"""Read-only physical-step diagnostics; no alternative controller or task logic.

The observer preserves checked_step's existing mj_step -> mj_forward sequence.
The intervening copy distinguishes integration-interval forces from the following
post-state forward solve. Use in one isolated process, never alongside ROS threads.
"""
from contextlib import AbstractContextManager
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from .model import URDF, require
from .reset import integration_state


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')


def excess(q, lower, upper):
    q, lower, upper = np.asarray(q), np.asarray(lower), np.asarray(upper)
    require(np.all(lower < upper) and np.isfinite(q).all(), 'Invalid joint bounds/state')
    return np.maximum(np.maximum(lower - q, q - upper), 0.)


def intervals(mask, post_times, dt):
    """True post sample k represents (t[k]-dt,t[k]]; adjacent samples form a run."""
    mask, times = np.asarray(mask, bool), np.asarray(post_times, float)
    require(mask.shape == times.shape and mask.ndim == 1 and dt > 0, 'Invalid interval samples')
    if not len(mask):
        return []
    require(np.allclose(np.diff(times), dt, rtol=0, atol=1e-10), 'Missing/nonuniform samples')
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [dict(start_s=float(times[a]-dt), end_s=float(times[b-1]),
                 sampled_duration_s=float((b-a)*dt), samples=int(b-a))
            for a, b in bounds.reshape(-1, 2)]


def directional_ratio(force, limits):
    force, limits = np.asarray(force, float), np.asarray(limits, float)
    require(np.all(limits[:, 0] < 0) and np.all(limits[:, 1] > 0), 'Need signed force limits')
    return np.where(force >= 0, force/limits[:, 1], force/limits[:, 0])


def discount_rewards(rewards, terms, gamma, *, terminated, truncated, terminal_value=None):
    """Observed environment prefix; optional critic tail is explicitly an estimate."""
    rewards = np.asarray(rewards, float)
    require(rewards.ndim == 1 and np.isfinite(rewards).all() and 0 < gamma < 1, 'Invalid rewards/gamma')
    require(not (terminated and truncated), 'Conflicting terminal flags')
    weights = gamma ** np.arange(len(rewards))
    parts = {k: float(weights @ np.asarray(v, float)) for k, v in terms.items()}
    require(all(len(v) == len(rewards) for v in terms.values()), 'Reward component length mismatch')
    require(np.allclose(np.sum(list(terms.values()), axis=0), rewards, atol=1e-9, rtol=1e-9),
            'Reward components disagree')
    require(terminal_value is None or (truncated and np.isfinite(terminal_value)),
            'Critic tail allowed only for time truncation')
    tail = None if terminal_value is None else float(gamma**len(rewards)*terminal_value)
    return dict(raw_return=float(rewards.sum()), observed_discounted_prefix=float(weights @ rewards),
                discounted_terms=parts, gamma_per_policy_transition=float(gamma), transitions=len(rewards),
                terminated=bool(terminated), truncated=bool(truncated), critic_tail_estimate=tail,
                historical_rollout_bootstrap=None,
                semantics='No post-terminal reward; critic tail is not observed reward or historical bootstrap.')


def _contacts(model, data, floor):
    worst = [dict(depth_m=0., geom_ids=None, geom_names=None, bodies=None) for _ in range(2)]
    total_z = total_norm = max_force = 0.
    max_pair = None
    for i, contact in enumerate(data.contact):
        pair = [int(contact.geom1), int(contact.geom2)]
        group = 0 if floor in pair else 1
        depth = max(0., -float(contact.dist))
        if depth > worst[group]['depth_m']:
            worst[group] = dict(depth_m=depth, geom_ids=pair,
                geom_names=[model.geom(g).name for g in pair],
                bodies=[model.body(int(model.geom_bodyid[g])).name for g in pair])
        if group == 0 and contact.efc_address >= 0:
            local = np.zeros(6)
            mj.mj_contactForce(model, data, i, local)
            world = (1. if contact.geom1 == floor else -1.) * (contact.frame.reshape(3, 3).T @ local[:3])
            norm = float(np.linalg.norm(world))
            total_z += float(world[2]); total_norm += norm
            if norm > max_force:
                max_force, max_pair = norm, pair
    return dict(floor=worst[0], self=worst[1], ground_vertical_N=total_z,
                ground_sum_norm_N=total_norm, largest_ground_contact_N=max_force,
                largest_ground_contact_geom_ids=max_pair)


class PhysicalObserver(AbstractContextManager):
    """Scope-bound forwarding wrappers, restricted to one exact MjData instance.

    No callbacks, state writes, extra stepping, or extra active-data forward calls.
    Existing reset and environment guards remain in force. Fields copied after
    mj_step and before checked_step's mj_forward describe the integration solve;
    qpos/qvel at that point are already post-integration. Contacts still describe
    the pre-state (for the required non-RK integrator).
    """
    def __init__(self, base):
        self.base = base
        self.records = []
        self.phase = 'reset'
        self.control_index = -1
        self.residual_gate = 0.
        self.pending = None
        self._installed = False
        require(base.model.opt.integrator in (mj.mjtIntegrator.mjINT_EULER,
                mj.mjtIntegrator.mjINT_IMPLICIT, mj.mjtIntegrator.mjINT_IMPLICITFAST),
                'Audit timing requires a verified single-stage integrator')

    def __enter__(self):
        require(not self._installed, 'Observer already installed')
        self._step = mj.mj_step
        env_module = sys.modules[type(self.base).__module__]
        self._checked = env_module.checked_step
        reset_module = sys.modules[self._checked.__module__]
        self._namespaces = tuple(dict.fromkeys((env_module, reset_module)))
        self._originals = [(module, module.checked_step) for module in self._namespaces]
        mj.mj_step = self._observed_step
        for module, _ in self._originals:
            module.checked_step = self._observed_checked
        self._installed = True
        return self

    def __exit__(self, *args):
        if self._installed:
            mj.mj_step = self._step
            for module, original in self._originals:
                module.checked_step = original
            self._installed = False
        return False

    def _observed_step(self, model, data, *args, **kwargs):
        if data is not self.base.data:
            return self._step(model, data, *args, **kwargs)
        require(not args and not kwargs, 'Audit expects exactly one physical step')
        base = self.base
        q, dq = base.loaded.read_state(data)
        target = base._target.copy() if self.phase == 'recovery' else np.full(len(q), np.nan)
        raw = base.kp*(target-q)-base.kd*dq if self.phase == 'recovery' else np.zeros(len(q))
        row = dict(pre_time_s=float(data.time), pre_qpos=data.qpos.copy(), pre_qvel=data.qvel.copy(),
                   target_rad=target, tau_raw_Nm=raw, ctrl_Nm=data.ctrl[base.context.ctrladr].copy(),
                   phase=self.phase, control_index=self.control_index, residual_gate=self.residual_gate)
        self._step(model, data)
        row.update(post_time_s=float(data.time), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                   integration_actuator_force_Nm=data.actuator_force.copy(),
                   integration_qfrc_actuator_Nm=data.qfrc_actuator.copy(),
                   integration_contact=_contacts(model, data, base.context.floor))
        self.pending = row

    def _observed_checked(self, loaded, data, deadline):
        self._checked(loaded, data, deadline)
        if data is self.base.data:
            require(self.pending is not None, 'Missing single physical step')
            row, self.pending = self.pending, None
            require(row['post_time_s'] == float(data.time)
                    and np.array_equal(row['qpos'], data.qpos)
                    and np.array_equal(row['qvel'], data.qvel), 'Post-forward unexpectedly changed physical state')
            row.update(postforward_actuator_force_Nm=data.actuator_force.copy(),
                       postforward_qfrc_actuator_Nm=data.qfrc_actuator.copy(),
                       postforward_contact=_contacts(loaded.model, data, self.base.context.floor))
            self.records.append(row)


def _joint_rows(base, records, origin, standing_tolerance):
    names = [r.joint_name for r in base.loaded.mapping]
    q = np.array([r['qpos'][base.context.qadr] for r in records])
    dq = np.array([r['qvel'][base.context.vadr] for r in records])
    times = np.array([r['post_time_s']-origin for r in records])
    target = np.array([r['target_rad'] for r in records])
    force = np.array([r['integration_qfrc_actuator_Nm'][base.context.vadr] for r in records])
    raw = np.array([r['tau_raw_Nm'] for r in records])
    ctrl = np.array([r['ctrl_Nm'] for r in records])
    ex = excess(q, base.q_min, base.q_max)
    ratio = directional_ratio(force, base.context.efforts)
    urdf = ET.parse(base.loaded.asset_repo / URDF).getroot()
    output = []
    for i, name in enumerate(names):
        limit = urdf.find(f"joint[@name='{name}']/limit")
        speed = None if limit is None or limit.get('velocity') is None else float(limit.get('velocity'))
        k = int(np.argmax(ex[:, i]))
        spans = intervals(ex[:, i] > 0, times, base.physics_dt)
        valid_target = target[:, i][np.isfinite(target[:, i])]
        margin = None if not len(valid_target) else float(min(np.min(valid_target-base.q_min[i]),
                                                            np.min(base.q_max[i]-valid_target)))
        output.append(dict(joint_name=name, q_lower_rad=base.q_min[i], q_upper_rad=base.q_max[i],
            q_min_observed_rad=float(q[:, i].min()), q_max_observed_rad=float(q[:, i].max()),
            max_lower_excess_rad=float(max(0., base.q_min[i]-q[:, i].min())),
            max_upper_excess_rad=float(max(0., q[:, i].max()-base.q_max[i])),
            max_excess_rad=float(ex[k, i]), time_of_max_excess_s=float(times[k]),
            time_above_zero_excess_s=float(np.count_nonzero(ex[:, i] > 0)*base.physics_dt),
            time_above_standing_tolerance_s=float(np.count_nonzero(ex[:, i] > standing_tolerance)*base.physics_dt),
            longest_consecutive_excess_s=max((s['sampled_duration_s'] for s in spans), default=0.),
            max_abs_velocity_rad_s=float(np.max(abs(dq[:, i]))), declared_velocity_limit_rad_s=speed,
            velocity_limit_source=None if speed is None else URDF+'#'+name+'/limit@velocity (declaration; not MJCF clamp)',
            max_velocity_ratio=None if speed is None or speed <= 0 else float(np.max(abs(dq[:, i]))/speed),
            max_positive_actuation_Nm=float(max(0., force[:, i].max())),
            max_negative_actuation_Nm=float(min(0., force[:, i].min())),
            max_actuation_limit_ratio=float(ratio[:, i].max()),
            torque_saturation_fraction=float(np.mean(raw[:, i] != ctrl[:, i])),
            min_target_limit_margin_rad=margin,
            urdf_lower_rad=None if limit is None else float(limit.get('lower')),
            urdf_upper_rad=None if limit is None else float(limit.get('upper')),
            urdf_effort_Nm=None if limit is None else float(limit.get('effort')),
            effective_effort_lower_Nm=base.context.efforts[i, 0],
            effective_effort_upper_Nm=base.context.efforts[i, 1]))
    return output


def _segment(base, rows, origin):
    if not rows:
        return dict(samples=0)
    q = np.array([r['qpos'][base.context.qadr] for r in rows])
    dq = np.array([r['qvel'][base.context.vadr] for r in rows])
    ex = excess(q, base.q_min, base.q_max)
    k, j = np.unravel_index(int(ex.argmax()), ex.shape)
    force = np.array([r['integration_qfrc_actuator_Nm'][base.context.vadr] for r in rows])
    result = dict(samples=len(rows), sampled_duration_s=len(rows)*base.physics_dt,
        max_joint_excess_rad=float(ex[k, j]), worst_joint=base.loaded.mapping[j].joint_name,
        worst_joint_elapsed_s=rows[k]['post_time_s']-origin,
        any_joint_above_zero_s=float(np.count_nonzero(ex.max(axis=1) > 0)*base.physics_dt),
        any_joint_above_standing_tolerance_s=float(np.count_nonzero(ex.max(axis=1) > .0001)*base.physics_dt),
        max_joint_speed_rad_s=float(abs(dq).max()),
        max_actuation_limit_ratio=float(directional_ratio(force, base.context.efforts).max()))
    for scope in ('integration', 'postforward'):
        contacts = [r[scope+'_contact'] for r in rows]
        scoped = {}
        for kind in ('floor', 'self'):
            index = max(range(len(rows)), key=lambda n: contacts[n][kind]['depth_m'])
            scoped[kind] = dict(contacts[index][kind],
                elapsed_s=rows[index]['pre_time_s' if scope == 'integration' else 'post_time_s']-origin)
        index = max(range(len(rows)), key=lambda n: contacts[n]['ground_vertical_N'])
        scoped.update(peak_ground_vertical_N=contacts[index]['ground_vertical_N'],
            peak_ground_vertical_bodyweights=contacts[index]['ground_vertical_N']/base.context.weight_N,
            peak_ground_elapsed_s=rows[index]['pre_time_s' if scope == 'integration' else 'post_time_s']-origin,
            peak_ground_sum_norm_N=max(c['ground_sum_norm_N'] for c in contacts),
            largest_ground_contact_N=max(c['largest_ground_contact_N'] for c in contacts))
        result[scope+'_contacts'] = scoped
    return result


def _standing_margins(base, samples):
    settings = base.tracker.settings
    if not samples:
        return None
    margins = {}
    def record(key, value):
        margins[key] = min(margins.get(key, float('inf')), float(value))
    for s in samples:
        v, tracker = s['measurement'], s['success']
        record('height_m', v['pelvis_height_m']-settings.h_ref_m*settings.height_ratio_min)
        record('tilt_deg', settings.tilt_deg_max-v['tilt_deg'])
        for side in ('left', 'right'):
            record(side+'_support_bodyweights', v[side+'_weight']-settings.foot_weight_min)
        feet = v['left_weight']+v['right_weight']
        record('feet_lower_bodyweights', feet-settings.feet_weight_min)
        record('feet_upper_bodyweights', settings.feet_weight_max-feet)
        for key, setting in [('other_weight', 'other_weight_max'), ('base_linear_m_s', 'linear_m_s_max'),
                ('com_linear_m_s', 'linear_m_s_max'), ('base_angular_rad_s', 'angular_rad_s_max'),
                ('torso_angular_rad_s', 'angular_rad_s_max'), ('joint_rad_s', 'joint_rad_s_max'),
                ('joint_limit_rad', 'joint_limit_rad_max'), ('floor_penetration_m', 'floor_penetration_m_max'),
                ('self_penetration_m', 'self_penetration_m_max'), ('support_gap_m', 'support_gap_m_max')]:
            record(key, getattr(settings, setting)-v[key])
        for key, value in tracker['drift_m'].items():
            record(key+'_drift_m', (settings.pelvis_drift_m_max if key == 'pelvis' else settings.foot_drift_m_max)-value)
    return margins


def compact_physical_arrays(arrays, lower, upper, effort_limits):
    """Keep step-level extrema and timing; calculate from full precision vectors first."""
    qadr, vadr = arrays['joint_qpos_addresses'], arrays['joint_dof_addresses']
    joint_excess = excess(arrays['qpos'][:, qadr], lower, upper)
    result = {k: v for k, v in arrays.items() if k not in {
        'pre_qpos', 'pre_qvel', 'qpos', 'qvel', 'target_rad', 'tau_raw_Nm', 'ctrl_Nm',
        'integration_actuator_force_Nm', 'integration_qfrc_actuator_Nm',
        'postforward_actuator_force_Nm', 'postforward_qfrc_actuator_Nm'}}
    result.update(max_joint_excess_rad=joint_excess.max(axis=1),
        worst_joint_index=joint_excess.argmax(axis=1),
        max_abs_joint_velocity_rad_s=abs(arrays['qvel'][:, vadr]).max(axis=1),
        max_actuation_limit_ratio=directional_ratio(
            arrays['integration_qfrc_actuator_Nm'][:, vadr], effort_limits).max(axis=1))
    return result


def reset_match(actual, expected, atol):
    """Validate a real reset before any controller step; never restore physical state."""
    require(set(actual) == set(expected) and atol >= 0, 'Invalid reset comparison fields/tolerance')
    differences = {}
    for key, value in actual.items():
        value, reference = np.asarray(value), np.asarray(expected[key])
        require(value.shape == reference.shape and np.isfinite(value).all()
                and np.isfinite(reference).all(), 'Invalid reset comparison: '+key)
        differences[key] = float(np.max(abs(value-reference))) if value.size else 0.
    require(all(v <= atol for v in differences.values()), 'Reset handoff mismatch: '+str(differences))
    return differences


def run_diagnostic(controller, policy, *, seed, output, zero_residual=False, gamma=.999,
                   source_identity=None, trace_level='full', expected_reset=None, reset_atol=1e-10):
    """One NEW complete deterministic diagnostic episode using a prepared controller.

    Caller owns validated loading and close(). This helper never loads a checkpoint,
    changes config, sets physical state, trains, or writes to input directories.
    """
    require(trace_level in ('full', 'compact'), 'Invalid physical trace level')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = controller.unwrapped
    old_capture = base.capture_substeps
    base.capture_substeps = True
    records, controls, samples = [], [], []
    started = time.monotonic()
    error = None
    final_info = {}
    origin = 0.
    reset_observation = reset_qpos = reset_qvel = None
    reset_differences = None
    terminated = truncated = False
    try:
        with PhysicalObserver(base) as observer:
            records = observer.records
            obs, reset_info = controller.reset(seed=int(seed))
            origin = float(base.episode_start_time)
            reset_observation, reset_qpos, reset_qvel = obs.copy(), base.data.qpos.copy(), base.data.qvel.copy()
            if expected_reset is not None:
                reset_differences = reset_match(dict(observation=obs, qpos=base.data.qpos,
                    qvel=base.data.qvel, integration_state=integration_state(base.model, base.data)),
                    expected_reset, reset_atol)
            observer.phase = 'recovery'
            while True:
                observer.control_index = len(controls)
                before_obs = obs.copy()
                if zero_residual:
                    action = np.zeros(controller.action_space.shape, dtype=np.float32)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                before_count = len(records)
                obs, reward, terminated, truncated, final_info = controller.step(action)
                gate = final_info.get('controller', {}).get('residual_gate', 1.)
                for r in records[before_count:]:
                    r['residual_gate'] = float(gate)
                samples.extend(base.last_substeps)
                controls.append(dict(observation=before_obs, action=np.asarray(action).copy(),
                    next_observation=obs.copy(), reward=float(reward), reward_terms=dict(final_info['reward_terms']),
                    terminated=bool(terminated), truncated=bool(truncated),
                    elapsed_s=float(final_info['elapsed_sim_s']), qpos=base.data.qpos.copy(), qvel=base.data.qvel.copy(),
                    physics_steps=int(final_info['physics_steps_executed']), controller=final_info.get('controller', {})))
                if terminated or truncated:
                    break
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        base.capture_substeps = old_capture
        summary = dict(schema='constraint-diagnostic-v1', status='ERROR' if error else 'COMPLETE', error=error,
            seed=int(seed), controller='zero_residual_reference' if zero_residual else 'reference_residual',
            source_identity=source_identity, wall_seconds=time.monotonic()-started, trace_level=trace_level,
            trace_retention='All metrics calculated from full float64 physical vectors. '
                'Compact traces retain every physical sample scalar and joint extrema, joint CSV, '
                'and full control-boundary observations/actions/states; full traces also retain physical vectors.',
            reset_handoff_absolute_s=origin, success=bool(final_info.get('is_success', False)),
            reset_handoff_comparison_max_abs=reset_differences,
            sim_duration_s=final_info.get('elapsed_sim_s'), final_info=final_info,
            physics_dt_s=base.physics_dt, standing_tolerance_rad=base.tracker.settings.joint_limit_rad_max,
            duration_rule='Right endpoint indicators: sum(mask[k])*dt for (post_time[k]-dt,post_time[k]]. '
                          'Consecutive true samples form runs; extrema are physical-step sampled, not continuous extrema.',
            force_timing='integration_* copies after native mj_step and before the EXISTING active mj_forward. '
                         'Their contacts/forces use the pre-state time; qpos/qvel use post-time. postforward_* '
                         'is the synchronized post-state recomputation used by existing success checks, not the applied interval solve.',
            force_definition='Ground vertical is the signed sum of contact force world-Z on robot; bodyweights=force/(mass*9.81). '
                             'Sum-norm and largest-contact force are separate; no force ratio is electrical power.',
            observer='Scoped Python forwarding wrappers only; no added mj_step or active mj_forward; no physics-state writes.',
            timing_source='https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_forward.c',
            noninterference_tolerances=dict(state_atol=1e-10, action_atol=1e-7, rtol=0),
            standing_settings=asdict(base.tracker.settings))
        if records:
            recovery = [r for r in records if r['phase'] == 'recovery']
            reset = [r for r in records if r['phase'] == 'reset']
            hold_start = None
            if final_info.get('is_success'):
                hold_start = float(base.data.time-base.tracker.settings.hold_s)
            hold = [] if hold_start is None else [r for r in recovery if r['post_time_s'] >= hold_start-1e-10]
            segments = dict(reset_settling=reset, recovery=recovery,
                residual_disabled=[r for r in recovery if r['residual_gate'] == 0.],
                residual_ramp=[r for r in recovery if 0. < r['residual_gate'] < 1.],
                residual_full=[r for r in recovery if r['residual_gate'] >= 1.], final_standing_window=hold)
            summary['segments'] = {k: _segment(base, rows, origin) for k, rows in segments.items()}
            summary['segment_definition'] = 'Residual stage uses actual gate held from pre-control boundary; final window includes both endpoint samples (2001 samples for a 2s hold). Segment memberships overlap.'
            summary['final_window_start_elapsed_s'] = None if hold_start is None else hold_start-origin
            summary['standing_window_minimum_margins'] = _standing_margins(base,
                [] if hold_start is None else [s for s in samples if s['measurement']['time_s'] >= hold_start-1e-10])
            times = np.array([r['post_time_s']-origin for r in recovery])
            if len(recovery):
                # Strict no-solved-ground-force definition, rather than an arbitrary light-load cutoff.
                airborne = np.array([r['postforward_contact']['ground_vertical_N'] <= 1e-9 for r in recovery])
                summary['airborne_intervals'] = intervals(airborne, times, base.physics_dt)
                summary['landing_events'] = [dict(elapsed_s=float(times[i]),
                    ground_vertical_N=recovery[i]['postforward_contact']['ground_vertical_N'])
                    for i in range(1, len(recovery)) if airborne[i-1] and not airborne[i]]
                landing_times = [e['elapsed_s'] for e in summary['landing_events']]
                landing = [r for r in recovery if any(t <= r['post_time_s']-origin <= t+.1+1e-10
                                                      for t in landing_times)]
                summary['segments']['landing_100ms'] = _segment(base, landing, origin)
                summary['airborne_landing_definition'] = ('Airborne = synchronized postforward total ground vertical force <=1e-9 N; '
                    'landing = the next loaded sample; landing_100ms is the union of 0.1 s windows beginning at those samples. '
                    'This is a force-based diagnostic, not a claim of complete geometric separation.')
                joints = _joint_rows(base, recovery, origin, base.tracker.settings.joint_limit_rad_max)
                with (output/'joint_constraints.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(joints[0])); writer.writeheader(); writer.writerows(joints)
                summary['constraint_envelope_checks'] = dict(
                    joint_excess=bool(max(j['max_excess_rad'] for j in joints) <= .0001),
                    directional_actuation=bool(max(j['max_actuation_limit_ratio'] for j in joints) <= 1.+1e-10),
                    declared_velocity=bool(all(j['max_velocity_ratio'] is None or j['max_velocity_ratio'] <= 1.+1e-10 for j in joints)),
                    velocity_limits_declared_count=sum(j['max_velocity_ratio'] is not None for j in joints))
                summary['constraint_envelope_pass'] = all(summary['constraint_envelope_checks'][k]
                    for k in ('joint_excess', 'directional_actuation', 'declared_velocity'))
                summary['constraint_envelope_definition'] = 'Recovery nominal joint-range excess <=0.0001 rad, directional actuation ratio<=1+1e-10 and each declared URDF speed ratio<=1+1e-10. The joint tolerance is borrowed from standing, not a hardware safety allowance. Candidate impact non-regression requires separate comparison with the unchanged-controller baseline.'
                summary['admissible_recovery'] = bool(summary['success'] and summary['constraint_envelope_pass'])
            arrays = {key: np.asarray([r[key] for r in records]) for key in
                ('pre_time_s', 'post_time_s', 'pre_qpos', 'pre_qvel', 'qpos', 'qvel', 'target_rad',
                 'tau_raw_Nm', 'ctrl_Nm', 'integration_actuator_force_Nm', 'integration_qfrc_actuator_Nm',
                 'postforward_actuator_force_Nm', 'postforward_qfrc_actuator_Nm', 'control_index', 'residual_gate')}
            arrays['is_recovery'] = np.array([r['phase'] == 'recovery' for r in records])
            for scope in ('integration', 'postforward'):
                for key in ('ground_vertical_N', 'ground_sum_norm_N', 'largest_ground_contact_N'):
                    arrays[scope+'_'+key] = np.array([r[scope+'_contact'][key] for r in records])
                for kind in ('floor', 'self'):
                    arrays[scope+'_'+kind+'_depth_m'] = np.array([r[scope+'_contact'][kind]['depth_m'] for r in records])
                    arrays[scope+'_'+kind+'_geom_ids'] = np.array([r[scope+'_contact'][kind]['geom_ids'] or [-1, -1] for r in records])
            arrays['joint_qpos_addresses'] = base.context.qadr
            arrays['joint_dof_addresses'] = base.context.vadr
            arrays['joint_names'] = np.array([r.joint_name for r in base.loaded.mapping])
            if samples:
                keys = sorted(k for k, v in samples[0]['measurement'].items() if isinstance(v, (int, float)))
                arrays['measurement_scalar_names'] = np.array(keys)
                arrays['recovery_measurement_scalars'] = np.array([[s['measurement'][k] for k in keys] for s in samples])
                arrays['recovery_stable_duration_s'] = np.array([s['success']['stable_duration_s'] for s in samples])
                arrays['recovery_standing_qualified'] = np.array([s['qualified'] for s in samples])
                arrays['recovery_drift_pelvis_left_right_m'] = np.array([[s['success']['drift_m'][k]
                    for k in ('pelvis', 'left', 'right')] for s in samples])
                if hold_start is not None:
                    window = [s['measurement'] for s in samples if s['measurement']['time_s'] >= hold_start-1e-10]
                    summary['standing_window_sway'] = dict(
                        torso_tilt_range_deg=float(np.ptp([s['tilt_deg'] for s in window])),
                        maximum_torso_tilt_deg=max(s['tilt_deg'] for s in window),
                        pelvis_xy_peak_to_peak_m=np.ptp([s['pelvis_xy'] for s in window], axis=0),
                        max_base_angular_rad_s=max(s['base_angular_rad_s'] for s in window),
                        max_torso_angular_rad_s=max(s['torso_angular_rad_s'] for s in window))
            summary['actuation_consistency'] = dict(
                integration_vs_postforward_max_abs_Nm=float(np.max(abs(arrays['integration_qfrc_actuator_Nm']-arrays['postforward_qfrc_actuator_Nm']))),
                mapped_joint_vs_ctrl_max_abs_Nm=float(np.max(abs(arrays['integration_qfrc_actuator_Nm'][:, base.context.vadr]-arrays['ctrl_Nm']))),
                actuator_vs_ctrl_max_abs_Nm=float(np.max(abs(arrays['integration_actuator_force_Nm']-arrays['ctrl_Nm']))),
                mapping='31 stateless scalar hinge motors; fixed unit gain/gear; no actuator-output clamp; input and joint clamps preserved.')
            if trace_level == 'compact':
                arrays = compact_physical_arrays(arrays, base.q_min, base.q_max, base.context.efforts)
            np.savez_compressed(output/'physical_trace.npz', **arrays)
        if controls:
            term_keys = sorted({k for c in controls for k in c['reward_terms']})
            terms = {k: np.array([c['reward_terms'].get(k, 0.) for c in controls]) for k in term_keys}
            arrays = {key: np.array([c[key] for c in controls]) for key in
                      ('observation', 'action', 'next_observation', 'reward', 'terminated', 'truncated',
                       'elapsed_s', 'qpos', 'qvel', 'physics_steps')}
            arrays.update(reset_observation=reset_observation, reset_qpos=reset_qpos, reset_qvel=reset_qvel,
                          reward_term_names=np.array(term_keys), reward_terms=np.array([terms[k] for k in term_keys]).T)
            np.savez_compressed(output/'control_trace.npz', **arrays)
            summary['reward'] = discount_rewards(arrays['reward'], terms, gamma, terminated=terminated, truncated=truncated)
            summary['maximum_stable_hold_s'] = max(s['success']['stable_duration_s'] for s in samples)
        summary['artifacts_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in output.iterdir() if p.is_file()}
        write_json(output/'summary.json', summary)
    return summary
