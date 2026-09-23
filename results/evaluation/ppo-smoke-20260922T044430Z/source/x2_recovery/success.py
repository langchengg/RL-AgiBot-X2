"""Reward-independent standing measurements and continuous success semantics.

Call after EVERY checked_step (mj_step + mj_forward). The caller owns reset,
control, model immutability and trajectory continuity; measurements cannot prove
that a caller never teleported qpos. No simulation advancement or I/O here.
"""
from dataclasses import asdict, dataclass
import math
import mujoco as mj
import numpy as np
from .model import compiled_mapping, require
from .reset import (check_callbacks, floor_geometry, measurements as reset_measurements,
                    model_signature, ResetSettings, settled_failures)

# Calibrated effective-model identity, unchanged physics. Computed once at context creation.
MODEL_SIGNATURE = 'c168981e31f28d03ed9ca518d5bdb701ecea8f8df41ca03096a69e40bb0b0a3a'
MODEL_FINGERPRINT = 'bd9bfae8f3a5115cad202f989cf43d7ef0a6678346e4dd3c519de76c300316b0'


@dataclass(frozen=True)
class SuccessSettings:
    h_ref_m: float  # Required: independent calibration, never inferred in update.
    height_ratio_min: float = .90
    tilt_deg_max: float = 15.
    foot_weight_min: float = .05
    feet_weight_min: float = .80
    feet_weight_max: float = 1.20
    other_weight_max: float = .00001
    linear_m_s_max: float = .10
    angular_rad_s_max: float = .25
    joint_rad_s_max: float = .50
    pelvis_drift_m_max: float = .05
    foot_drift_m_max: float = .03
    hold_s: float = 2.
    floor_penetration_m_max: float = .001
    self_penetration_m_max: float = .0001
    support_gap_m_max: float = 2e-6
    joint_limit_rad_max: float = .0001

    def __post_init__(self):
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                    for v in asdict(self).values()), 'Settings must be finite and positive; calibration required')
        require(.1 < self.h_ref_m < 2. and 0 < self.height_ratio_min <= 1., 'Invalid reference height/ratio')
        require(0 < self.tilt_deg_max < 90 and 0 < self.foot_weight_min < .5,
                'Invalid tilt/foot threshold')
        require(2*self.foot_weight_min <= self.feet_weight_min < self.feet_weight_max <= 2.,
                'Invalid support bounds')
        require(self.other_weight_max <= .005 and self.linear_m_s_max <= 1.
                and self.angular_rad_s_max <= 2. and self.joint_rad_s_max <= 5., 'Unreasonable stability thresholds')
        require(self.pelvis_drift_m_max <= .1 and self.foot_drift_m_max <= .1
                and self.floor_penetration_m_max <= .002 and self.self_penetration_m_max <= .001
                and self.support_gap_m_max <= 1e-5 and self.joint_limit_rad_max <= .001,
                'Unreasonable steady geometry tolerance')


# Frozen from candidate 1, t=6..8 s: median pelvis body-origin height.
CALIBRATED_SETTINGS = SuccessSettings(h_ref_m=0.6724955472220092)


class StandingContext:
    """Verified fixed horizontal floor, dynamic subtree, exact foot-body proxies."""
    def __init__(self, loaded):
        m = loaded.model
        require(compiled_mapping(m) == loaded.mapping, 'Stale/wrong mapping')
        require(model_signature(m) == MODEL_SIGNATURE, 'Model changed: standing calibration is stale')
        check_callbacks()
        require(m.neq == 0 and m.nmocap == 0 and not np.any(m.body_gravcomp), 'External support mechanism')
        require(np.array_equal(m.opt.gravity, [0., 0., -9.81]), 'Unsupported gravity')
        self.loaded = loaded
        self.pelvis = m.body('pelvis').id
        self.torso = m.body('torso_link').id
        self.foot_bodies = tuple(m.body(side+'_ankle_roll_link').id for side in ('left', 'right'))
        require(m.jnt_bodyid[m.joint('floating_base_joint').id] == self.pelvis
                and m.body_parentid[self.pelvis] == 0, 'Expected free pelvis root')
        neutral = mj.MjData(m)
        mj.mj_forward(m, neutral)
        self.floor, self.normal, self.floor_origin, geoms = floor_geometry(m, neutral)
        require(np.array_equal(neutral.xmat[self.torso].reshape(3, 3), np.eye(3)),
                'Unverified torso +Z longitudinal axis')
        self.robot_bodies = frozenset(b for b in range(1, m.nbody)
                                     if m.body_rootid[b] == self.pelvis)
        require(len(self.robot_bodies) == m.nbody-1, 'Unknown dynamic subtree')
        self.robot_geoms = frozenset(g for g in geoms if m.geom_bodyid[g] in self.robot_bodies)
        require(self.robot_geoms == frozenset(geoms), 'Unknown floor-contact object')
        self.feet = tuple(frozenset(g for g in geoms if m.geom_bodyid[g] == b)
                          for b in self.foot_bodies)
        require(all(len(gs) == 12 for gs in self.feet) and not self.feet[0] & self.feet[1],
                'Expected disjoint 12-sphere feet')
        for gs in self.feet:
            require(all(m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE
                        and m.geom_size[g, 0] == .005 for g in gs), 'Unexpected foot proxy')
        require(not np.any(m.geom_margin) and not np.any(m.geom_gap), 'Unsupported contact margin/gap')
        self.weight_N = float(sum(m.body_mass[b] for b in self.robot_bodies)*np.linalg.norm(m.opt.gravity))
        require(np.isfinite(self.weight_N) and self.weight_N > 0, 'Weight must be positive')
        self.dt = float(m.opt.timestep)
        self.qadr = loaded.qpos_addresses
        self.vadr = loaded.dof_addresses
        self.ranges = np.array([r.position_range for r in loaded.mapping])
        self.efforts = np.array([r.effort_range for r in loaded.mapping])
        self.ctrladr = np.array([r.ctrl_index for r in loaded.mapping])


def contact_world_force(contact, force_local, floor):
    """mj_contactForce acts on geom2; frame rows are world contact axes.

    Keep signed world forces, including tangential components. Never abs(Fz).
    """
    require(floor in (contact.geom1, contact.geom2), 'Not a floor contact')
    return (1. if contact.geom1 == floor else -1.) * (contact.frame.reshape(3, 3).T @ force_local[:3])


def ground_forces(context, data):
    m = context.loaded.model
    feet = [0., 0.]
    other = 0.
    support = []
    floor_depth = self_depth = 0.
    for i, c in enumerate(data.contact):
        if context.floor not in (c.geom1, c.geom2):
            self_depth = max(self_depth, -float(c.dist))
            continue
        floor_depth = max(floor_depth, -float(c.dist))
        g = int(c.geom2 if c.geom1 == context.floor else c.geom1)
        require(g in context.robot_geoms, 'Unknown ground-contact geom')
        if c.efc_address < 0:
            continue
        force = np.empty(6)
        mj.mj_contactForce(m, data, i, force)
        require(np.isfinite(force).all(), 'Nonfinite contact force')
        world = contact_world_force(c, force, context.floor)
        magnitude = float(np.linalg.norm(world))
        if magnitude == 0.:
            continue
        require(world[2] >= -1e-9, 'Negative ground support: force/frame/model invalid')
        side = next((k for k, gs in enumerate(context.feet) if g in gs), None)
        if side is None:
            other += magnitude  # Sum of norms, no per-contact small-force filter.
        else:
            feet[side] += float(world[2])
        support.append({'geom': g, 'body': m.body(m.geom_bodyid[g]).name,
                        'group': ('left', 'right')[side] if side is not None else 'other',
                        'force_world_N': world.tolist(), 'norm_N': magnitude,
                        'distance_m': float(c.dist), 'efc_address': int(c.efc_address)})
    return feet, other, support, floor_depth, self_depth


def measure_standing(context, data):
    """Read synchronized solved state. mj_subtreeVel only updates derived data.

    Free-joint translational qvel is world linear velocity of the body origin;
    angular qvel is local, whose norm is rotation invariant. mj_objectVelocity
    returns [angular, linear], centered at body COM for mjOBJ_BODY, world axes
    with flg_local=0. Only its angular half is used for the torso.
    """
    c = context; m = c.loaded.model; d = data
    require(d.model is m, 'Wrong model/data')
    require(np.isfinite(d.time) and all(np.isfinite(a).all() for a in
            (d.qpos, d.qvel, d.qacc, d.ctrl, d.qfrc_constraint, d.qfrc_actuator)), 'Nonfinite state/time')
    require(abs(np.linalg.norm(d.qpos[3:7])-1) < 1e-9, 'Invalid base quaternion')
    require(not np.any(d.warning.number), 'MuJoCo numerical warning')
    require(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied), 'Illegal external force')
    ctrl = d.ctrl[c.ctrladr]
    require(np.all(ctrl >= c.efforts[:, 0]) and np.all(ctrl <= c.efforts[:, 1]), 'Unbounded control')
    require(np.max(abs(d.qfrc_actuator[c.vadr]-ctrl)) < 1e-8, 'Unexpected actuation')
    feet, other, support, floor_depth, self_depth = ground_forces(c, d)
    mj.mj_subtreeVel(m, d)
    velocity = np.empty(6)
    mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, c.torso, velocity, 0)
    dot = float(d.xmat[c.torso].reshape(3, 3)[:, 2] @ c.normal)
    q = d.qpos[c.qadr]
    return {'time_s': float(d.time), 'pelvis_height_m': float((d.xpos[c.pelvis]-c.floor_origin) @ c.normal),
            'upright_dot': dot, 'tilt_deg': float(np.rad2deg(np.arccos(np.clip(dot, -1, 1)))),
            'left_weight': feet[0]/c.weight_N, 'right_weight': feet[1]/c.weight_N,
            'other_weight': other/c.weight_N, 'other_force_N': other, 'support': support,
            'base_linear_m_s': float(np.linalg.norm(d.qvel[:3])),
            'com_linear_m_s': float(np.linalg.norm(d.subtree_linvel[c.pelvis])),
            'base_angular_rad_s': float(np.linalg.norm(d.qvel[3:6])),
            'torso_angular_rad_s': float(np.linalg.norm(velocity[:3])),
            'joint_rad_s': float(np.max(abs(d.qvel[c.vadr]))),
            'joint_limit_rad': float(max(0., np.max(c.ranges[:, 0]-q), np.max(q-c.ranges[:, 1]))),
            'floor_penetration_m': floor_depth, 'self_penetration_m': self_depth,
            'support_gap_m': max([0.] + [r['distance_m'] for r in support]),
            'pelvis_xy': d.xpos[c.pelvis, :2].tolist(),
            'left_xy': d.xpos[c.foot_bodies[0], :2].tolist(),
            'right_xy': d.xpos[c.foot_bodies[1], :2].tolist()}


def finite_measurement(value):
    if isinstance(value, dict):
        return all(finite_measurement(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_measurement(v) for v in value)
    return not isinstance(value, (float, int, np.number)) or bool(np.isfinite(value))


def standing_failures(v, s):
    require(finite_measurement(v), 'Nonfinite measurement')
    failures = []
    if v['pelvis_height_m']/s.h_ref_m < s.height_ratio_min: failures.append('height')
    if v['upright_dot'] < math.cos(math.radians(s.tilt_deg_max)): failures.append('torso_tilt')
    for side in ('left', 'right'):
        if v[side+'_weight'] < s.foot_weight_min: failures.append(side+'_support')
    if not s.feet_weight_min <= v['left_weight']+v['right_weight'] <= s.feet_weight_max:
        failures.append('feet_total')
    for key, limit in (('other_weight', s.other_weight_max),
                       ('base_linear_m_s', s.linear_m_s_max), ('com_linear_m_s', s.linear_m_s_max),
                       ('base_angular_rad_s', s.angular_rad_s_max), ('torso_angular_rad_s', s.angular_rad_s_max),
                       ('joint_rad_s', s.joint_rad_s_max), ('joint_limit_rad', s.joint_limit_rad_max),
                       ('floor_penetration_m', s.floor_penetration_m_max),
                       ('self_penetration_m', s.self_penetration_m_max), ('support_gap_m', s.support_gap_m_max)):
        if v[key] > limit: failures.append(key)
    return failures


class SuccessTracker:
    """No provenance boolean in update. Invalid execution is latched until reset.

    reset() starts a standing-only fixture. reset_from_supine() validates an actual
    reset handoff, preserving its residual state/time. This is a caller contract,
    not a tamper-proof history certificate. Call invalidate(reason) on any step
    exception, model mutation, teleport/reset or other violation in the caller.
    """
    TIME_EPS = 1e-10  # floating summation only: ten-millionth of the 1 ms step

    def __init__(self, settings, timestep_s, timeout_s=20.):
        require(np.isfinite(timestep_s) and 0 < timestep_s <= .01, 'Invalid physics timestep')
        require(np.isfinite(timeout_s) and timeout_s > 0, 'Invalid timeout')
        self.settings = settings; self.dt = timestep_s; self.timeout = timeout_s
        self.reset(0.)

    def reset(self, start_time_s):
        require(np.isfinite(start_time_s) and start_time_s >= 0, 'Invalid attempt start time')
        self.start = float(start_time_s); self.origin = None; self.last = None; self.result = None
        self.invalid_reason = None; self.timed_out = False; self.succeeded = False
        self._clear_window()

    def reset_from_supine(self, context, data, reset_result):
        # Deliberately only callable at the exact reset handoff, not after rising.
        self.reset(float(data.time))
        r = reset_result
        require(r['status'] == 'PASS' and r['episode_start_time'] == data.time
                and np.array_equal(r['final_qpos'], data.qpos)
                and np.array_equal(r['final_qvel'], data.qvel), 'Not the successful reset handoff')
        require(model_signature(context.loaded.model) == MODEL_SIGNATURE, 'Changed model at handoff')
        require(not settled_failures(reset_measurements(context.loaded, data),
                    ResetSettings(**r['criteria']), context.weight_N), 'Invalid supine origin')
        measure_standing(context, data)  # also rejects illegal forces and numerical errors
        self.origin = {'time_s': float(data.time), 'qpos': tuple(data.qpos), 'qvel': tuple(data.qvel)}

    def _clear_window(self):
        self.stable_since = None; self.reference = None
        self.drift = {'pelvis': 0., 'left': 0., 'right': 0.}

    def invalidate(self, reason):
        self.invalid_reason = self.invalid_reason or str(reason)
        self.succeeded = False; self._clear_window()

    def update(self, v):
        import copy
        if not finite_measurement(v): self.invalidate('nonfinite measurement/time')
        t = v['time_s']
        if not self.invalid_reason:
            previous = self.last['time_s'] if self.last is not None else self.start
            delta = t-previous
            if delta < -self.TIME_EPS: self.invalidate('time rollback/reset')
            elif self.last is not None and abs(delta) <= self.TIME_EPS:
                if {k: x for k, x in v.items() if k != 'time_s'} != {k: x for k, x in self.last.items() if k != 'time_s'}:
                    self.invalidate('inconsistent duplicate sample')
                else: return copy.deepcopy(self.result)
            elif abs(delta-self.dt) > self.TIME_EPS and not (self.last is None and abs(delta) <= self.TIME_EPS):
                self.invalidate('missing/nonuniform physics sample')
        if self.invalid_reason:
            self.result = {'instant_standing_ok': False, 'standing_held': False, 'recovery_success': False,
                           'stable_duration_s': 0., 'failures': [], 'invalid_reason': self.invalid_reason,
                           'drift_m': self.drift.copy(), 'timed_out': self.timed_out}
            return copy.deepcopy(self.result)
        bad = standing_failures(v, self.settings)
        instant = not bad
        if instant and self.reference is not None:
            for key in self.drift:
                distance = float(np.linalg.norm(np.array(v[key+'_xy'])-self.reference[key]))
                self.drift[key] = max(self.drift[key], distance)
                limit = self.settings.pelvis_drift_m_max if key == 'pelvis' else self.settings.foot_drift_m_max
                if self.drift[key] > limit: bad.append(key+'_drift')
        observed_drift = self.drift.copy()
        if bad:
            self._clear_window()
        elif self.stable_since is None:
            self.stable_since = t
            self.reference = {key: np.array(v[key+'_xy']) for key in self.drift}
        duration = 0. if self.stable_since is None else float(t-self.stable_since)
        held = not bad and duration+self.TIME_EPS >= self.settings.hold_s
        elapsed = t-self.start
        # At the deadline, success gets first refusal; later samples cannot rescue timeout.
        if self.origin is not None and held and not self.timed_out and elapsed <= self.timeout+self.TIME_EPS:
            self.succeeded = True
        if not self.succeeded and elapsed >= self.timeout-self.TIME_EPS: self.timed_out = True
        self.result = {'instant_standing_ok': instant, 'standing_held': held,
                       'recovery_success': self.succeeded, 'stable_duration_s': duration,
                       'failures': bad, 'invalid_reason': None, 'drift_m': observed_drift,
                       'timed_out': self.timed_out}
        self.last = copy.deepcopy(v)
        return copy.deepcopy(self.result)
