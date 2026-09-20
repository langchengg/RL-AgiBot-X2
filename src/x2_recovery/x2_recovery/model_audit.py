"""Bounded Step 3 evidence. Run through runtime_check audit; not a recovery reset."""

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from .model import (COMMIT, REPOSITORY, ROBOT, SCENE, SOURCE_HASHES, URDF, VARIANT,
                    coordinate_check, git, load_effective_model, require, resolve_assets,
                    rotation, urdf_inertia)

# Declared before the formal suite. No thresholds adapt to observations.
TOLERANCES = {
    "force_Nm": 1e-10, "quaternion_norm": 1e-10, "time_s": 1e-10,
    "mass_export_kg": 5e-5, "inertia_export_kg_m2": 1e-6, "com_m": 1e-9,
    "freefall_position_m": 1e-8, "freefall_velocity_m_s": 1e-8,
    "initial_floor_clearance_m": .002, "floor_penetration_m": .015,
    "self_penetration_m": .005, "active_normal_force_N": .01,
    "limit_initial_penetration_rad": .002, "limit_peak_penetration_rad": .005,
    "limit_final_penetration_rad": .001,
}
AREAS = ("floating_base", "mass_inertia", "collision_coverage", "joint_ranges",
         "actuator_semantics", "effective_limits", "indexing")
REGIONS = {
    "back_torso": ("torso_link",),
    "left_forearm": ("left_elbow_link",), "right_forearm": ("right_elbow_link",),
    "left_hand_proxy": ("left_wrist_pitch_link", "left_wrist_roll_link"),
    "right_hand_proxy": ("right_wrist_pitch_link", "right_wrist_roll_link"),
    "left_knee_shin": ("left_knee_link",), "right_knee_shin": ("right_knee_link",),
    "left_foot": ("left_ankle_roll_link",), "right_foot": ("right_ankle_roll_link",),
}


def result(ok, reason, scope="effective_model", **measurements):
    return {"status": "PASS" if ok else "FAIL", "reason": reason,
            "scope": scope, **measurements}


def json_value(value, invalid=None, path="root"):
    if type(value).__module__ == 'mujoco._enums':
        return int(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v, invalid, path + "." + str(k)) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v, invalid, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, float) and not np.isfinite(value):
        if invalid is not None:
            invalid.append(path)
        return {"status": "FAIL", "reason": "Nonfinite measurement", "value": None}
    return str(value) if isinstance(value, Path) else value


def save_report(path, report):
    invalid = []
    clean = json_value(report, invalid)
    if invalid:
        clean["nonfinite_measurements"] = invalid
        clean["verdict"] = "PARTIAL/BLOCKED"
        clean["checks"]["strict_json"] = result(False, "Nonfinite measurements recorded", paths=invalid)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return not invalid


def fields(obj, names):
    return {name: json_value(getattr(obj, name)) for name in names.split()}


def fresh(model, high=False):
    d = mj.MjData(model)  # qpos0 contains a valid w,x,y,z quaternion.
    if high:
        d.qpos[2] = 3
    d.ctrl[:] = 0
    d.act[:] = 0
    d.qfrc_applied[:] = 0
    d.xfrc_applied[:] = 0
    mj.mj_forward(model, d)
    return d


def coherent(model, data, expected_time):
    """Forward refresh is required after integration before sampling forces."""
    mj.mj_forward(model, data)
    for name in ("qpos", "qvel", "qacc", "qfrc_actuator", "qfrc_constraint", "actuator_force"):
        require(np.isfinite(getattr(data, name)).all(), f"Nonfinite {name}")
    require(not np.any(data.warning.number), f"MuJoCo warnings: {data.warning.number}")
    require(abs(data.time - expected_time) < TOLERANCES['time_s'], "Time rollback/reset or wrong step count")
    require(abs(np.linalg.norm(data.qpos[3:7]) - 1) < TOLERANCES['quaternion_norm'], "Invalid base quaternion")


def step(model, data, index, deadline):
    require(time.monotonic() < deadline, "Probe wall-time deadline exceeded")
    mj.mj_step(model, data)
    coherent(model, data, (index + 1) * model.opt.timestep)


def identity(project):
    versions = {}
    for package in ("mujoco", "numpy", "gymnasium", "stable-baselines3", "torch", "Pillow",
                    "rclpy", "colcon-core", "setuptools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable in current package metadata"
    return {"python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
            "base_prefix": sys.base_prefix, "platform": platform.platform(),
            "machine": platform.machine(), "os_release": platform.freedesktop_os_release(),
            "virtualization": subprocess.run(["systemd-detect-virt"], capture_output=True,
                                              text=True, timeout=5).stdout.strip(),
            "versions": versions, "mujoco_module": mj.__file__, "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "project": {"path": str(project), "commit": git(project, "rev-parse", "HEAD"),
                        "branch": git(project, "branch", "--show-current"),
                        "dirty": git(project, "status", "--porcelain"),
                        "remotes": git(project, "remote", "-v"),
                        "source_sha256": {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted((project / 'src').rglob('*.py'))
                                          if '__pycache__' not in p.parts}}}


def provenance(path):
    scene = ET.parse(path / SCENE).getroot()
    robot = ET.parse(path / ROBOT).getroot()
    includes = [x.get('file') for x in scene.findall('include')]
    require(includes == ['x2_ultra.xml'] and not robot.findall('.//include'), "Unexpected include chain")
    referenced = set(SOURCE_HASHES)
    for mesh in robot.findall('asset/mesh'):
        referenced.add(str(Path(VARIANT) / 'meshes' / mesh.get('file')))
    for mesh in ET.parse(path / URDF).getroot().findall('.//mesh'):
        referenced.add(str(Path(VARIANT) / mesh.get('filename')))
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in sorted(referenced)}
    return {"repository": REPOSITORY, "commit": git(path, 'rev-parse', 'HEAD'), "path": path,
            "urdf": URDF, "robot_xml": ROBOT, "scene": SCENE, "include_chain": [SCENE, ROBOT],
            "sha256": hashes, "all_checkout_changes": git(path, 'status', '--porcelain', '--untracked-files=all'),
            "license": "AgibotTech; upstream LICENSE: Mulan PSL v2, retained without modification",
            "notices": git(path, 'ls-tree', '-r', '--name-only', 'HEAD', '--', 'LICENSE', '*NOTICE*'),
            "source_pinning": "XML/license SHA256 and clean pinned Git tree; all referenced mesh files hashed"}


def inventory(model, spec):
    d = fresh(model)
    return {"dimensions": fields(model, 'nq nv nu nbody njnt ngeom nmesh neq ntendon nmocap na nsensor'),
            "options": fields(model.opt, 'timestep gravity integrator solver cone jacobian tolerance iterations '
                              'ls_tolerance ls_iterations noslip_tolerance noslip_iterations ccd_tolerance '
                              'ccd_iterations disableflags enableflags disableactuator density viscosity wind'),
            "option_names": {"integrator": mj.mjtIntegrator(model.opt.integrator).name,
                             "solver": mj.mjtSolver(model.opt.solver).name,
                             "cone": mj.mjtCone(model.opt.cone).name},
            "compiler": fields(spec.compiler, 'inertiafromgeom inertiagrouprange boundmass boundinertia '
                               'balanceinertia settotalmass alignfree fusestatic autolimits discardvisual'),
            "qpos0": model.qpos0.copy(), "total_robot_mass_kg": float(sum(model.body_mass[1:])),
            "com_world_at_qpos0_m": d.subtree_com[model.body('pelvis').id].copy()}


def body_inertia(model, b):
    R = rotation(model.body_iquat[b])
    return {"mass_kg": float(model.body_mass[b]), "com_body_m": model.body_ipos[b].copy(),
            "principal_kg_m2": model.body_inertia[b].copy(), "principal_quaternion_wxyz": model.body_iquat[b].copy(),
            "tensor_at_com_body_frame_kg_m2": R @ np.diag(model.body_inertia[b]) @ R.T}


def mass_audit(raw, effective, spec, urdf):
    rows = []
    ok = True
    for b in range(1, effective.nbody):
        name = effective.body(b).name
        mass, com, tensor = urdf_inertia(urdf.find(f"link[@name='{name}']"))
        r, e = body_inertia(raw, b), body_inertia(effective, b)
        eig = e['principal_kg_m2']
        valid = (np.isfinite([e['mass_kg'], *eig, *e['com_body_m'], *e['principal_quaternion_wxyz']]).all()
                 and e['mass_kg'] > 0 and min(eig) > 0 and 2*max(eig) <= sum(eig)+1e-12)
        errors = {"mass_kg": abs(e['mass_kg'] - mass),
                  "com_m": float(max(abs(e['com_body_m'] - com))),
                  "tensor_kg_m2": float(np.max(abs(e['tensor_at_com_body_frame_kg_m2'] - tensor)))}
        matches = (errors['mass_kg'] <= TOLERANCES['mass_export_kg']
                   and errors['com_m'] <= TOLERANCES['com_m']
                   and errors['tensor_kg_m2'] <= TOLERANCES['inertia_export_kg_m2'])
        rows.append({"body_id": b, "name": name, "raw": r, "effective": e,
                     "urdf": {"mass_kg": mass, "com_body_m": com, "tensor_kg_m2": tensor},
                     "errors": errors, "principal_condition_ratio": float(max(eig)/min(eig)),
                     "status": "PASS" if valid and matches else "FAIL",
                     "classification": "corrected mesh inference" if name == 'pelvis' else "export precision"})
        ok &= bool(valid and matches)
    fixed = []
    for joint in urdf.findall('joint'):
        if joint.get('type') == 'fixed':
            child = urdf.find(f"link[@name='{joint.find('child').get('link')}']")
            fixed.append({"name": joint.get('name'), "parent": joint.find('parent').get('link'),
                          "child": child.get('name'), "has_inertial": child.find('inertial') is not None})
    # The only fixed child carrying mass is pelvis under a massless base_link.
    require(all(not r['has_inertial'] or r['child'] == 'pelvis' for r in fixed),
            "New fixed-link inertial aggregation requires investigation")
    base_link = urdf.find("link[@name='base_link']")
    require(base_link.find('inertial') is None, "Unexpected root fixed-frame inertia")
    p = spec.body('pelvis')
    source_geoms = [{"mesh": g.meshname, "density_kg_m3": g.density,
                     "group": g.group, "contype": g.contype, "conaffinity": g.conaffinity}
                    for g in p.geoms]
    no_masks = spec.copy()
    for g in no_masks.body('pelvis').geoms:
        g.contype = g.conaffinity = 0
    no_mask_mass = float(no_masks.compile().body('pelvis').mass[0])
    doubled = spec.copy()
    for g in doubled.body('pelvis').geoms:
        g.density *= 2
    doubled_mass = float(doubled.compile().body('pelvis').mass[0])
    raw_mass = float(raw.body('pelvis').mass[0])
    ok &= abs(no_mask_mass-raw_mass) < 1e-10 and abs(doubled_mass-2*raw_mass) < 1e-10
    symmetry = [{"left": r['name'], "right": r['name'].replace('left_', 'right_', 1),
                 "mass_difference_kg": r['effective']['mass_kg'] - float(effective.body(
                     r['name'].replace('left_', 'right_', 1)).mass[0])}
                for r in rows if r['name'].startswith('left_')]
    return result(ok, "32 dynamic bodies; common body axes at each COM, reconstructed full tensors; "
                  "remaining differences within declared export precision. Source consistency, not hardware calibration.",
                  bodies=rows, fixed_frames=fixed, bilateral_mass_differences=symmetry,
                  aggregation="Eight sensor frames and base_link have no inertia; no positive mass discarded/aggregated. "
                              "world/floor is static, excluded from robot mass; no parallel-axis shift needed.",
                  inference_experiment={"scope": "isolated_probe", "source_geoms": source_geoms,
                                        "raw_pelvis_kg": raw_mass, "masks_disabled_pelvis_kg": no_mask_mass,
                                        "density_doubled_pelvis_kg": doubled_mass,
                                        "reason": "AUTO inference uses density and inertia groups, independently of contact masks"},
                  raw_total_mass_kg=float(sum(raw.body_mass[1:])),
                  effective_total_mass_kg=float(sum(effective.body_mass[1:])))


def range_audit(raw, loaded, urdf):
    rows = []
    for joint in urdf.findall('joint'):
        if joint.get('type') == 'fixed':
            continue
        name = joint.get('name')
        row = loaded.joint(name)
        errors = coordinate_check(raw, joint)
        lim = joint.find('limit')
        u = np.array([float(lim.get('lower')), float(lim.get('upper'))])
        r = raw.jnt_range[row.joint_id]
        if np.array_equal(r, u):
            classification = 'equal'
        elif np.max(abs(r-u)) <= 1e-4:
            classification = 'rounding'
        elif r[0] >= u[0] and r[1] <= u[1]:
            classification = 'narrower'
        elif r[0] <= u[0] and r[1] >= u[1]:
            classification = 'wider'
        elif max(r[0], u[0]) < min(r[1], u[1]):
            classification = 'overlapping'
        else:
            classification = 'unresolved'
        desired = [max(r[0], u[0]), min(r[1], u[1])]
        valid = np.allclose(row.position_range, desired, atol=1e-12, rtol=0)
        rows.append({"joint": name, "joint_id": row.joint_id, "type": joint.get('type'),
                     "axis_local": raw.jnt_axis[row.joint_id].copy(), "coordinate_errors": errors,
                     "zero_reference_rad": float(raw.qpos0[row.qpos_address]), "units": "rad",
                     "urdf_range": u, "raw_range": r.copy(), "effective_range": row.position_range,
                     "classification": classification, "limited": bool(loaded.model.jnt_limited[row.joint_id]),
                     "damping_Nm_s_rad": float(raw.dof_damping[row.dof_address]),
                     "frictionloss_Nm": float(raw.dof_frictionloss[row.dof_address]),
                     "armature_kg_m2": float(raw.dof_armature[row.dof_address]),
                     "urdf_dynamics": None if joint.find('dynamics') is None else joint.find('dynamics').attrib,
                     "velocity_limit_rad_s": float(lim.get('velocity')), "velocity_enforced": False,
                     "decision": 'keep stricter existing range' if np.array_equal(r, row.position_range)
                                 else 'project conservative intersection',
                     "status": "PASS" if valid and classification != 'unresolved' else "FAIL"})
    counts = {c: sum(r['classification'] == c for r in rows)
              for c in ('equal', 'rounding', 'narrower', 'wider', 'overlapping', 'unresolved')}
    return result(all(r['status'] == 'PASS' for r in rows),
                  "Same axis/sign, zero and parent-child transforms within 2e-6 export tolerance; "
                  "no speed clamp in MJCF. Absent URDF dynamics are not hardware damping/friction specifications.",
                  counts=counts, discrepancy_count=sum(r['classification'] != 'equal' for r in rows),
                  joints=rows, unmatched_movable=[], mimic_or_coupled=[], unactuated_scalar_joints=[])


def expected_actuation(model, row, control):
    """Verified stateless hinge formula, input clamp -> gain -> gear -> joint clamp."""
    a, j = row.ctrl_index, row.joint_id
    u = float(np.clip(control, *model.actuator_ctrlrange[a]))
    force = row.gain*u
    if model.actuator_forcelimited[a]:
        force = float(np.clip(force, *model.actuator_forcerange[a]))
    torque = float(np.clip(row.gear*force, *model.jnt_actfrcrange[j]))
    return force, torque


def verify_actuation(model, data, row, force, torque):
    a, v = row.ctrl_index, row.dof_address
    out = int(model.actuator_outadr[a])
    require(abs(data.actuator_force[out]-force) <= TOLERANCES['force_Nm'], "Actuator force mismatch")
    expected = np.zeros(model.nv)
    expected[v] = torque
    require(np.allclose(data.qfrc_actuator, expected, atol=TOLERANCES['force_Nm'], rtol=0),
            "Transmitted joint actuation mismatch")
    adr, nnz = data.moment_rowadr[out], data.moment_rownnz[out]
    columns = data.moment_colind[adr:adr+nnz]
    values = data.actuator_moment[adr:adr+nnz]
    require(np.array_equal(columns, [v]) and np.allclose(values, [row.gear], atol=1e-12),
            "Unexpected sparse transmission moment")


def actuator_audit(loaded, urdf):
    m = loaded.model
    inventory_rows, trials = [], []
    deadline = time.monotonic()+30
    discrepancies = []
    for r in loaded.mapping:
        a, j = r.ctrl_index, r.joint_id
        urdf_effort = float(urdf.find(f"joint[@name='{r.joint_name}']/limit").get('effort'))
        attrs = {name: getattr(m, 'actuator_'+name)[a].copy()
                 for name in ('trnid', 'trntype', 'gear', 'gainprm', 'gaintype', 'biasprm', 'biastype',
                              'dynprm', 'dyntype', 'ctrlrange', 'ctrllimited', 'forcerange', 'forcelimited',
                              'actadr', 'actnum', 'actrange', 'actlimited', 'group', 'ctrladr', 'ctrlnum',
                              'outadr', 'outnum', 'delay', 'armature', 'damping', 'dampingpoly',
                              'plugin', 'actearly', 'ctrlspec')}
        attrs.update(name=r.actuator_name, id=a, joint=r.joint_name,
                     joint_actuatorfrcrange=m.jnt_actfrcrange[j].copy(),
                     joint_actuatorfrclimited=bool(m.jnt_actfrclimited[j]), urdf_effort_Nm=urdf_effort,
                     effective_effort_Nm=r.effort_range,
                     joint_clamp_independently_exercised=False,
                     joint_clamp_reason="Hidden behind equal or stricter input clamp; normal envelope verified",
                     decision="Retain conservative source control bounds; no extra clamp")
        inventory_rows.append(attrs)
        if not np.array_equal(r.effort_range, [-urdf_effort, urdf_effort]):
            discrepancies.append({"joint": r.joint_name, "input": r.control_range,
                                  "joint_effort_envelope_Nm": r.effort_range, "urdf_effort_Nm": urdf_effort})
        require(max(abs(np.array(r.effort_range))) <= urdf_effort, "Effort envelope exceeds paired URDF")
        lo, hi = r.control_range
        for u in (0., lo*.4, hi*.4, lo*1.05, hi*1.05):
            require(time.monotonic()<deadline, "Forward actuator suite timed out")
            d = fresh(m, high=True)
            d.qpos[loaded.qpos_addresses] = interior_positions(loaded)
            d.qvel[loaded.dof_addresses] = np.arange(1, m.nu+1)*.0003
            d.ctrl[a] = u  # Deliberately no wrapper clipping.
            coherent(m, d, 0.)
            force, torque = expected_actuation(m, r, u)
            verify_actuation(m, d, r, force, torque)
            require(d.ctrl[a] == u, "Unexpected ctrl buffer overwrite")
            trials.append({"actuator_id": a, "input": u, "stored_ctrl": float(d.ctrl[a]),
                           "actuator_force": float(d.actuator_force[m.actuator_outadr[a]]),
                           "qfrc_actuator": float(d.qfrc_actuator[r.dof_address]),
                           "expected_force": force, "expected_torque": torque, "status": "PASS"})
    return result(True, "155 fresh forward checks; gain=gear=1, bias=0, no activation state; "
                  "ctrl input in N m equals motor torque before joint clamp. Total physical joint load also includes "
                  "gravity, passive and contact forces, which are not qfrc_actuator.",
                  definition={"steps": 0, "inputs": "0, 40% each bound, 105% each bound", "wall_limit_s": 30,
                              "force_tolerance_Nm": TOLERANCES['force_Nm'], "all_other_controls": 0},
                  inventory=inventory_rows, trials=trials, discrepancy_count=len(discrepancies),
                  discrepancies=discrepancies, global_settings=fields(m.opt, 'disableflags disableactuator'))


def interior_positions(loaded):
    values = []
    for i, row in enumerate(loaded.mapping):
        lo, hi = row.position_range
        values.append(float(np.clip(.01 + i*.002, lo+.05, hi-.05)))
    return np.array(values)


def mapping_audit(loaded):
    m = loaded.model
    d = fresh(m, high=True)
    positions = interior_positions(loaded)
    # Elbows need distinct negative interior positions too.
    for i, r in enumerate(loaded.mapping):
        if positions[i] <= r.position_range[0]+.05 or positions[i] >= r.position_range[1]-.05:
            positions[i] = r.position_range[1]-.06-i*.001
    velocities = np.arange(1, m.nu+1)*-.001
    for r, q, v in zip(loaded.mapping, positions, velocities):
        # Independent name lookup for the write side, validated address arrays for read side.
        j = m.joint(r.joint_name).id
        d.qpos[m.jnt_qposadr[j]], d.qvel[m.jnt_dofadr[j]] = q, v
    coherent(m, d, 0.)
    measured_q, measured_v = loaded.read_state(d)
    require(len(np.unique(positions)) == m.nu, "Mapping test positions must be distinct")
    require(np.array_equal(measured_q, positions) and np.array_equal(measured_v, velocities), "State ordering mismatch")
    r = loaded.joint('head_yaw_joint')
    d.ctrl[r.ctrl_index] = .2
    coherent(m, d, 0.)
    verify_actuation(m, d, r, .2, .2)
    return result(True, "All 31 unique hinge joints covered; 6 free-base DOFs excluded. Direct named-state access; no sensors assumed.",
                  mapping=[asdict(r) for r in loaded.mapping], distinct_positions_rad=positions,
                  distinct_velocities_rad_s=velocities,
                  traversal_order=[m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE],
                  order_differs=[r.joint_id for r in loaded.mapping] != list(range(1, m.njnt)),
                  example={**asdict(r), "qpos": d.qpos[r.qpos_address], "qvel": d.qvel[r.dof_address],
                           "ctrl": d.ctrl[r.ctrl_index], "actuator_force": d.actuator_force[m.actuator_outadr[r.ctrl_index]],
                           "qfrc_actuator": d.qfrc_actuator[r.dof_address]})


def base_audit(loaded):
    m = loaded.model
    b = m.body('pelvis').id
    j = m.joint('floating_base_joint').id
    callbacks = {name: getattr(mj, name)() is not None for name in dir(mj) if name.startswith('get_mjcb_')}
    require(not any(callbacks.values()), f"Unexpected installed callbacks: {callbacks}")
    require(m.neq == m.ntendon == m.nmocap == 0 and not np.any(m.body_gravcomp)
            and m.body_parentid[b] == 0, "Hidden base support/coupling")
    require(np.array_equal(m.opt.gravity, [0, 0, -9.81]) and m.opt.disableflags == 0,
            "Ordinary gravity/physics unavailable")
    d = fresh(m, high=True)
    initial = d.subtree_com[b].copy()
    q0 = d.qpos.copy()
    history = []
    deadline = time.monotonic()+10
    for k in range(100):
        step(m, d, k, deadline)
        mj.mj_subtreeVel(m, d)
        require(d.ncon == 0 and not np.any(d.ctrl) and not np.any(d.qfrc_applied)
                and not np.any(d.xfrc_applied), "Freefall contaminated by support/contact")
        history.append({"time": float(d.time), "com_world_m": d.subtree_com[b].copy(),
                        "com_velocity_world_m_s": d.subtree_linvel[b].copy(), "ncon": d.ncon})
    expected_delta = .5*m.opt.gravity*d.time*(d.time+m.opt.timestep)  # Euler semi-implicit position
    displacement = d.subtree_com[b]-initial
    pos_error = float(max(abs(displacement-expected_delta)))
    vel_error = float(max(abs(d.subtree_linvel[b]-m.opt.gravity*d.time)))
    require(pos_error < TOLERANCES['freefall_position_m'] and vel_error < TOLERANCES['freefall_velocity_m_s'],
            "COM does not follow analytical gravity")
    # Frame test at a non-identity attitude, independent from dynamics/gravity.
    d = fresh(m, high=True)
    mj.mju_axisAngle2Quat(d.qpos[3:7], np.array([0., 0., 1.]), .6)
    mj.mj_forward(m, d)
    q = d.qpos.copy()
    dq = np.zeros(m.nv)
    dq[:6] = [.02, -.01, .03, .04, -.03, .02]
    dt = .1
    mj.mj_integratePos(m, q, dq, dt)
    expected_quat = d.qpos[3:7].copy()
    delta = np.empty(4)
    mj.mju_axisAngle2Quat(delta, dq[3:6]/np.linalg.norm(dq[3:6]), np.linalg.norm(dq[3:6])*dt)
    mj.mju_mulQuat(expected_quat, d.qpos[3:7], delta)  # body-local angular increment, right multiplication
    require(np.allclose(q[:3], d.qpos[:3]+dq[:3]*dt, atol=1e-12, rtol=0)
            and np.allclose(q[3:7], expected_quat, atol=1e-12, rtol=0), "Free-joint integration frame mismatch")
    d.qpos[:] = q
    d.qvel[:] = dq
    coherent(m, d, 0.)
    world_velocity = np.empty(6)
    mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, world_velocity, 0)
    require(np.allclose(world_velocity[:3], rotation(q[3:7])@dq[3:6], atol=1e-12, rtol=0),
            "Angular velocity frame mismatch")
    require(d.ncon == 0 and not np.any(d.qfrc_actuator[:6]), "Base is contacted/actuated")
    return result(True, "Six unactuated DOFs; COM ballistic fall and independent valid translation/rotation verified.",
                  root_body={"id": b, "name": m.body(b).name},
                  free_joint={"id": j, "name": m.joint(j).name, "qpos_address": int(m.jnt_qposadr[j]),
                              "dof_address": int(m.jnt_dofadr[j]),
                              "qpos": "world xyz + unit quaternion wxyz (7)",
                              "qvel": "world linear xyz + body-local angular xyz (6)"},
                  supports={"equality": m.neq, "tendons": m.ntendon, "mocap": m.nmocap,
                            "gravcomp": m.body_gravcomp.copy(), "callbacks": callbacks,
                            "ongoing_pose_overwrite": False},
                  definition={"qpos": q0, "qvel": "zero", "ctrl": "zero", "steps": 100,
                              "wall_limit_s": 10, "gravity": m.opt.gravity.copy(), "model_changes": []},
                  displacement_m=displacement, expected_displacement_m=expected_delta,
                  position_error_m=pos_error, velocity_error_m_s=vel_error, samples=history,
                  frame_probe={"qpos": q, "velocity": dq[:6], "world_object_angular_velocity": world_velocity[:3],
                               "integration_interval_s": dt, "steps": 0})


def limit_audit(loaded):
    m = copy.copy(loaded.model)
    # Only options change, no structural constants; no mj_setConst needed.
    m.opt.gravity[:] = 0
    m.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_CONTACT) | int(mj.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    trials = []
    deadline = time.monotonic()+30
    for r in loaded.mapping:
        for side in (-1, 1):
            d = fresh(m, high=True)
            bound = r.position_range[0 if side < 0 else 1]
            d.qpos[r.qpos_address] = bound+side*TOLERANCES['limit_initial_penetration_rad']
            d.ctrl[r.ctrl_index] = side*.1
            coherent(m, d, 0.)
            indices = np.flatnonzero((d.efc_type == mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)
                                     & (d.efc_id == r.joint_id))
            association = {"constraint_indices": indices.copy(), "constraint_force": d.efc_force[indices].copy(),
                           "qfrc_constraint_Nm": float(d.qfrc_constraint[r.dof_address]),
                           "qacc_rad_s2": float(d.qacc[r.dof_address])}
            exercised = (len(indices)>0 and np.any(d.efc_force[indices]>0)
                         and side*d.qfrc_constraint[r.dof_address]<0 and side*d.qacc[r.dof_address]<0)
            peak = TOLERANCES['limit_initial_penetration_rad']
            samples = []
            for k in range(80):
                step(m, d, k, deadline)
                pen = max(0., side*(d.qpos[r.qpos_address]-bound))
                peak = max(peak, pen)
                if k in (0, 9, 39, 79):
                    samples.append({"time_s": float(d.time), "qpos_rad": float(d.qpos[r.qpos_address]),
                                    "qvel_rad_s": float(d.qvel[r.dof_address]), "penetration_rad": pen})
            ok = bool(exercised and peak <= TOLERANCES['limit_peak_penetration_rad']
                      and pen < TOLERANCES['limit_final_penetration_rad'])
            trials.append(result(ok, "Active named limit and restoring acceleration, then penetration decreases",
                                 'isolated_probe', joint=r.joint_name, side='lower' if side<0 else 'upper',
                                 initial=association, peak_penetration_rad=peak, final_penetration_rad=pen,
                                 samples=samples, simulated_time_s=float(d.time), warnings=d.warning.number.copy()))
    return result(all(t['status']=='PASS' for t in trials),
                  "62 sides exercised under modest outward loading. Isolation proves soft joint-limit response; "
                  "not reachability of every bound with full collision/gravity enabled.", 'isolated_probe',
                  definition={"changes": ["gravity=0", "disable CONTACT", "disable FRICTIONLOSS"],
                              "unchanged": "inertia, armature, timestep, solver, joint/control/effort bounds",
                              "initial_state": "qpos0, base z=3m, tested joint 0.002rad outside bound, zero velocities",
                              "input_Nm": "+/-0.1 outward on tested motor only", "steps_each": 80,
                              "wall_limit_s": 30, "tolerances": {k:v for k,v in TOLERANCES.items() if k.startswith('limit_')}},
                  effective_static_limits_enabled=bool(np.all(loaded.model.jnt_limited[1:])
                                                      and loaded.model.opt.disableflags == 0), trials=trials)


def geom_label(m, g):
    b = int(m.geom_bodyid[g])
    return {"id": int(g), "name": m.geom(g).name or None, "body_id": b, "body": m.body(b).name,
            "body_local_index": int(g-m.body_geomadr[b])}


def floor_eligible(m, g):
    f = m.geom('floor').id
    return bool((m.geom_contype[g] & m.geom_conaffinity[f]) or (m.geom_contype[f] & m.geom_conaffinity[g]))


def geometry_inventory(m, spec):
    rows = []
    for g in range(m.ngeom):
        mesh = int(m.geom_dataid[g]) if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH else -1
        rows.append({**geom_label(m, g), **{name: getattr(m, 'geom_'+name)[g].copy()
                     for name in ('type', 'pos', 'quat', 'size', 'contype', 'conaffinity', 'condim', 'friction',
                                  'margin', 'gap', 'solref', 'solimp', 'solmix', 'priority', 'group')},
                     "mesh": m.mesh(mesh).name if mesh>=0 else None,
                     "mesh_scale": spec.mesh(m.mesh(mesh).name).scale.copy() if mesh>=0 else None,
                     "floor_eligible": floor_eligible(m, g) if g else False,
                     "classification": "contact" if m.geom_contype[g] or m.geom_conaffinity[g] else "visual only"})
    return {"geoms": rows, "pairs": m.npair, "exclusions": m.nexclude,
            "rules": "No explicit pairs/exclusions. Mask OR test plus same-body/welded-body and parent-child "
                     "filters for self collisions. Plane floor has infinite extent despite display size.",
            "contact_surface": "Mesh collision uses convex hulls, not detailed concave visual triangles. "
                               "Each foot has 12 spheres of radius 0.005m; the detailed foot mesh is visual only. "
                               "No articulated hand/finger joints: wrist geometry is the available hand proxy.",
            "regions": {key: [geom_label(m,g) for g in range(1,m.ngeom)
                               if m.body(m.geom_bodyid[g]).name in bodies and floor_eligible(m,g)]
                        for key,bodies in REGIONS.items()}}


def lowest_geom_z(m, d, g):
    p, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3,3)
    typ = m.geom_type[g]
    if typ == mj.mjtGeom.mjGEOM_MESH:
        mesh = m.geom_dataid[g]
        vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
        return float(np.min(vertices @ R[2] + p[2]))
    if typ == mj.mjtGeom.mjGEOM_SPHERE:
        return float(p[2]-m.geom_size[g,0])
    if typ == mj.mjtGeom.mjGEOM_CYLINDER:
        return float(p[2]-m.geom_size[g,0]*np.linalg.norm(R[2,:2])-m.geom_size[g,1]*abs(R[2,2]))
    raise ValueError(f"Unexpected contact geom type {typ}")


def contact_pose(loaded, pose):
    """Audit-only valid poses. Never used as an episode reset or support fixture."""
    m = loaded.model
    d = fresh(m, high=True)
    edits = {}
    axis, angle = np.array([0.,1.,0.]), 0.
    if pose == 'back':
        angle = -np.pi/2
    elif pose != 'feet':
        side = pose.split('_')[0]
        sign = 1 if side == 'left' else -1
        axis, angle = np.array([1.,0.,0.]), -sign*np.pi/2
        if pose.endswith('shin'):
            edits = {side+'_shoulder_pitch_joint': -3., side+'_hip_roll_joint': sign*.3,
                     side+'_knee_joint': 1.9, side+'_ankle_roll_joint': -sign*.2}
        elif pose.endswith('arm'):
            # Move the forearm forward of the hip hull; arm-at-side probes below
            # retain the measured trapping failure as an explicit investigation.
            edits = {side+'_shoulder_pitch_joint': -.6}
    mj.mju_axisAngle2Quat(d.qpos[3:7], axis, angle)
    for name, value in edits.items():
        d.qpos[loaded.joint(name).qpos_address] = value
    for r in loaded.mapping:
        require(r.position_range[0] <= d.qpos[r.qpos_address] <= r.position_range[1], "Invalid diagnostic pose")
    mj.mj_forward(m, d)
    lows = [(lowest_geom_z(m,d,g),g) for g in range(1,m.ngeom) if floor_eligible(m,g)]
    lowest, geom = min(lows)
    d.qpos[2] += TOLERANCES['initial_floor_clearance_m']-lowest
    coherent(m,d,0.)
    require(d.ncon == 0, "Diagnostic pose starts in floor/self contact")
    return d, {"pose": pose, "joint_edits_rad": edits, "qpos": d.qpos.copy(),
               "qvel": d.qvel.copy(), "initial_lowest_geom": geom_label(m,geom),
               "placement": "lowest actual collision mesh vertex/primitive support point + 0.002m", "ncon": d.ncon}


def contact_samples(m,d):
    floor = m.geom('floor').id
    rows = []
    for ci,c in enumerate(d.contact):
        force = np.empty(6)
        mj.mj_contactForce(m,d,ci,force)
        require(np.isfinite(force).all(), "Nonfinite contact force")
        is_floor = floor in (c.geom1,c.geom2)
        frame = c.frame.reshape(3,3)
        # Contact normal points from geom1 to geom2; returned force acts on geom2.
        on_robot_world = (1 if c.geom1 == floor else -1) * (frame.T @ force[:3]) if is_floor else None
        if is_floor and force[0] > TOLERANCES['active_normal_force_N']:
            require(on_robot_world[2] > 0, "Floor normal-force sign convention mismatch")
        rows.append({"geom1": int(c.geom1), "geom2": int(c.geom2), "floor": is_floor,
                     "robot_body": m.body(m.geom_bodyid[c.geom2 if c.geom1==floor else c.geom1]).name if is_floor else None,
                     "distance_m": float(c.dist), "efc_address": int(c.efc_address),
                     "active": bool(c.efc_address>=0 and force[0]>TOLERANCES['active_normal_force_N']),
                     "normal_force_N": float(force[0]), "force_contact_frame": force.copy(),
                     "force_on_robot_world_N": on_robot_world, "position_world_m": c.pos.copy(),
                     "normal_geom1_to_geom2_world": frame[0].copy()})
    return rows


def render_collision(loaded, snapshots, output):
    from PIL import Image

    require(os.environ.get('MUJOCO_GL') == 'osmesa', "Audit render requires MUJOCO_GL=osmesa before importing mujoco")
    m = copy.copy(loaded.model)
    # Display grouping only on a disposable model; contact masks/physics untouched.
    for g in range(m.ngeom):
        m.geom_group[g] = 0 if (g==0 or floor_eligible(m,g)) else 5
    option = mj.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    option.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL] = True
    frames = []
    for pose in ('back','feet','left_arm','left_shin'):
        snap = snapshots[pose]
        d = fresh(m)
        d.qpos[:], d.qvel[:], d.time = snap['qpos'], snap['qvel'], snap['time']
        mj.mj_forward(m,d)
        camera = mj.MjvCamera()
        camera.lookat[:] = d.subtree_com[m.body('pelvis').id]
        camera.distance = 1.8 if pose=='feet' else 2.
        camera.azimuth, camera.elevation = (125,-22) if pose=='feet' else (135,-35)
        with mj.Renderer(m,height=480,width=640) as renderer:
            renderer.update_scene(d,camera=camera,scene_option=option)
            pixels = renderer.render()
        require(pixels.shape==(480,640,3) and pixels.std()>1, "Invalid offscreen image")
        file = output/f'collision-{pose}.png'
        Image.fromarray(pixels).save(file)
        frames.append({"path": str(file), "pose": pose, "simulated_time_s": snap['time'],
                       "qpos": snap['qpos'], "image_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                       "display": "collision hulls/proxies only, contact points and force arrows",
                       "visual_inspection": "NOT_TESTED by automated command; inspect displayed files"})
    return frames


def collision_audit(loaded, spec, output):
    m = loaded.model
    geometry = geometry_inventory(m,spec)
    coverage = {key: {"available": bool(geometry['regions'][key]), "exercised": False,
                      "active_force": False, "peak_normal_force_N": 0., "evidence": []} for key in REGIONS}
    trials, investigations, snapshots = [], [], {}
    for pose in ('feet','back','left_arm','right_arm','left_shin','right_shin',
                 'left_arm_at_side','right_arm_at_side'):
        required = not pose.endswith('at_side')
        d, initial = contact_pose(loaded,pose)
        deadline = time.monotonic()+15
        peak_floor = peak_self = 0.
        self_pairs = {}
        active_floor_samples = 0
        sample_history = []
        for k in range(180):
            step(m,d,k,deadline)
            contacts = contact_samples(m,d)
            if any(c['floor'] and c['active'] for c in contacts):
                active_floor_samples += 1
            for c in contacts:
                if c['floor']:
                    peak_floor = max(peak_floor,-c['distance_m'])
                    for region,bodies in (REGIONS.items() if required else []):
                        if c['robot_body'] in bodies:
                            r = coverage[region]
                            r['exercised'] = True
                            r['active_force'] |= c['active']
                            if c['normal_force_N']>r['peak_normal_force_N']:
                                r['peak_normal_force_N'] = c['normal_force_N']
                                r['evidence'] = [{"pose": pose, "time_s": float(d.time), **c}]
                else:
                    peak_self = max(peak_self,-c['distance_m'])
                    key = f"{c['geom1']}:{c['geom2']}"
                    if key not in self_pairs or c['distance_m']<self_pairs[key]['distance_m']:
                        self_pairs[key] = {**c, "bodies": [m.body(m.geom_bodyid[g]).name for g in (c['geom1'],c['geom2'])]}
            if k in (19,49,99,179):
                sample_history.append({"time_s": float(d.time), "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                                       "qacc_max_abs": float(np.max(abs(d.qacc))), "contacts": contacts})
            if k == 99:
                snapshots[pose] = {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "time": float(d.time)}
        expected_self = all(set(c['bodies']) in ({'pelvis','left_hip_roll_link'}, {'pelvis','right_hip_roll_link'})
                            for c in self_pairs.values())
        ok = (peak_floor <= TOLERANCES['floor_penetration_m']
              and peak_self <= TOLERANCES['self_penetration_m'] and expected_self and active_floor_samples>0)
        trial = result(ok, "Uncontrolled free-base whole model, no fixtures or ongoing state correction",
                             required=required,
                             definition={"initial": initial, "steps": 180, "wall_limit_s": 15,
                                         "controls": "all zero", "applied_forces": "all zero", "model_changes": []},
                             max_floor_penetration_m=peak_floor, max_self_penetration_m=peak_self,
                             self_contacts=list(self_pairs.values()),
                             self_contact_classification=("Pelvis/hip-roll contact during passive leg collapse; "
                                 "active self-collision retained, <=5mm. No initial overlap." if expected_self else
                                 "Arm at side becomes trapped against hip hull during passive impact; "
                                 "exceeds declared 5mm self-penetration tolerance. No numerical warning/reset."),
                             unexpected_self_contacts=not expected_self, active_floor_samples=active_floor_samples,
                             simulated_time_s=float(d.time), warnings=d.warning.number.copy(), samples=sample_history)
        (trials if required else investigations).append(trial)
    for key,r in coverage.items():
        r['status'] = 'PASS' if r['available'] and r['exercised'] and r['active_force'] else 'FAIL'
    frames = render_collision(loaded,snapshots,output)
    return result(all(t['status']=='PASS' for t in trials) and all(r['status']=='PASS' for r in coverage.values()),
                  "Required anatomical regions exercised with measured active robot-floor forces. "
                  "Bounded transient impacts; not a settled supine reset, hardware contact calibration, or recovery evidence.",
                  geometry=geometry, regions=coverage, trials=trials, images=frames,
                  pose_investigation={"trials": investigations,
                      "resolution": "Retain failed arm-at-side measurements. Required arm-floor probes place "
                                    "the shoulder at -0.6rad to avoid the identified hip/wrist trapping. "
                                    "Same 180 steps, physics and tolerances. These impacts do not certify arbitrary poses."},
                  force_convention="mj_contactForce local frame, normal first; frame.T maps force to world, "
                                   "sign chosen for force on robot and verified upward for plane contact")


PHYSICS_FIELDS = ('qpos0', 'body_mass', 'body_ipos', 'body_iquat', 'body_inertia', 'body_pos', 'body_quat',
                  'body_gravcomp',
                  'jnt_type', 'jnt_axis', 'jnt_range', 'jnt_limited', 'jnt_actfrcrange', 'jnt_actfrclimited',
                  'jnt_qposadr', 'jnt_dofadr', 'dof_armature', 'dof_damping', 'dof_frictionloss',
                  'geom_pos', 'geom_quat', 'geom_size', 'geom_type', 'geom_contype', 'geom_conaffinity',
                  'geom_friction', 'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap',
                  'mesh_vert', 'actuator_trnid', 'actuator_trntype',
                  'actuator_ctrlrange', 'actuator_ctrllimited', 'actuator_gear', 'actuator_gainprm',
                  'actuator_forcerange', 'actuator_forcelimited', 'actuator_biasprm', 'actuator_biastype',
                  'actuator_dynprm', 'actuator_dyntype', 'actuator_ctrladr', 'actuator_outadr')


def fingerprint(loaded):
    h = hashlib.sha256()
    for name in PHYSICS_FIELDS:
        h.update(name.encode())
        h.update(getattr(loaded.model,name).tobytes())
    h.update(str(loaded.model.opt).encode())
    h.update(json.dumps([asdict(r) for r in loaded.mapping],sort_keys=True).encode())
    return h.hexdigest()


def repeat_audit(loaded, initial_fingerprint, source_before):
    after = load_effective_model(loaded.asset_repo)
    original = fingerprint(loaded)
    again = fingerprint(after)
    require(original == initial_fingerprint == again, "Probe contamination or non-reproducible fresh model")
    # A different process/cwd verifies resolution; the main process never chdirs.
    code = ('from x2_recovery.model import load_effective_model; '
            'from x2_recovery.model_audit import fingerprint; '
            'import sys; print(fingerprint(load_effective_model(sys.argv[1])))')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    with tempfile.TemporaryDirectory(prefix='x2-cwd-') as cwd:
        completed = subprocess.run([sys.executable,'-c',code,str(loaded.asset_repo)],cwd=cwd,env=env,
                                   text=True,capture_output=True,check=True,timeout=30)
    require(completed.stdout.strip()==again, "Model depends on current working directory")
    source_after = provenance(loaded.asset_repo)
    require(source_before['sha256']==source_after['sha256']
            and source_before['all_checkout_changes']==source_after['all_checkout_changes'], "Upstream files changed")
    require(after.model is not loaded.model, "Loader returned shared mutable model")
    return result(True,"Fresh loads, different cwd and source hashes agree after all disposable probes",
                  fingerprint=again, compared_fields=PHYSICS_FIELDS, upstream_unchanged=True)


def visual_review(frames, review):
    if not review:
        return {"status": "NOT_TESTED", "reason": "Inspect all four collision views and record hash-bound observations"}
    require(len(frames) == 4, "Visual review requires all four newly rendered images")
    for frame in frames:
        entry = review.get(Path(frame['path']).name, {})
        require(entry.get('sha256') == frame['image_sha256'] and bool(entry.get('observation')),
                f"Missing/stale visual review: {frame['path']}")
        frame['visual_inspection'] = entry['observation']
    return result(True, "Human/agent visual observations match all four freshly rendered image hashes",
                  observations=review)


def run_audit(asset_repo, output, project=None, image_review_from=None):
    output = Path(output).resolve()
    candidate_assets = Path(asset_repo or os.environ.get('X2_ASSET_REPO') or
                            Path.home()/'.cache/hrs-x2-recovery/agibot_x2_urdf').expanduser().resolve()
    require(not output.is_relative_to(candidate_assets), "Reports cannot be written into upstream assets")
    review, review_error = {}, None
    if image_review_from:
        try:
            review = json.loads(Path(image_review_from).read_text()).get('image_review', {})
        except (OSError, ValueError) as exc:
            review_error = str(exc)
    output.mkdir(parents=True,exist_ok=True)
    path = output/'report.json'
    report = {"schema": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_completed": False, "verdict": "PARTIAL/BLOCKED", "tolerances": TOLERANCES,
              "checks": {name: {"status":"NOT_TESTED", "reason":"Current run has not reached this check"}
                         for name in AREAS},
              "limitations": {"hardware_fidelity": "NOT_TESTED", "recovery_trainability": "NOT_TESTED",
                              "episode_ready_supine_reset": "NOT_TESTED; next step",
                              "A1": "Pending", "A2_A6": "Pending", "evaluation": "Not evaluated"}}
    report['image_review'] = review
    report['checks']['visual_review'] = {"status":"NOT_TESTED", "reason":"Images not yet inspected in current run"}
    save_report(path,report)  # Invalidate a previous successful run immediately.
    try:
        project = Path(project or git(Path(__file__).resolve().parent,'rev-parse','--show-toplevel'))
        report['runtime'] = identity(project)
        headers = Path(mj.__file__).parent/'include/mujoco'
        report['api_evidence'] = {
            'installed_header_sha256': {name: hashlib.sha256((headers/name).read_bytes()).hexdigest()
                                       for name in ('mujoco.h','mjspec.h','mjmodel.h','mjdata.h')},
            'matching_official_sources': [
                'https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_forward.c',
                'https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_support.c',
                'https://github.com/google-deepmind/mujoco/blob/3.13.0/doc/overview.rst'],
            'verification': 'Installed MjSpec fullinertia recompilation; sparse moment row API; input/output addresses; '
                            'force clamp order and local angular quaternion integration checked numerically. '
                            'No network access occurs during audit/loading.'}
        assets = resolve_assets(asset_repo)
        require(not output.is_relative_to(assets), "Reports must not be written into the upstream checkout")
        report['source'] = provenance(assets)
        urdf = ET.parse(assets/URDF).getroot()
        spec = mj.MjSpec.from_file(str(assets/SCENE))
        # Explicit audit-only baseline, compiled/inventoried before loading the effective model.
        raw = spec.compile()
        report['upstream_baseline'] = inventory(raw,spec)
        loaded = load_effective_model(assets)
        report['effective_model'] = inventory(loaded.model,spec)
        report['overrides'] = loaded.overrides
        before = fingerprint(loaded)
        checks = (("mass_inertia",lambda: mass_audit(raw,loaded.model,spec,urdf)),
                  ("joint_ranges",lambda: range_audit(raw,loaded,urdf)),
                  ("actuator_semantics",lambda: actuator_audit(loaded,urdf)),
                  ("indexing",lambda: mapping_audit(loaded)),
                  ("floating_base",lambda: base_audit(loaded)),
                  ("effective_limits",lambda: limit_audit(loaded)),
                  ("collision_coverage",lambda: collision_audit(loaded,spec,output)),
                  ("visual_review",lambda: visual_review(report['checks']['collision_coverage'].get('images',[]),review)),
                  ("reproducibility",lambda: repeat_audit(loaded,before,report['source'])))
        for name,check in checks:
            try:
                report['checks'][name] = check()
            except Exception as exc:
                report['checks'][name] = result(False,str(exc),traceback=traceback.format_exc())
            save_report(path,report)
            print(json.dumps({"check":name,"status":report['checks'][name]['status'],
                              "reason":report['checks'][name]['reason']}),flush=True)
        if review_error:
            report['checks']['visual_review'] = result(False, review_error)
        report['run_completed'] = True
        report['verdict'] = 'COMPLETE' if all(r['status']=='PASS' for r in report['checks'].values()) else 'PARTIAL/BLOCKED'
    except BaseException as exc:
        report['checks']['run'] = result(False,str(exc),traceback=traceback.format_exc())
        report['verdict'] = 'PARTIAL/BLOCKED'
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        valid_json = save_report(path,report)
    success = report['run_completed'] and report['verdict']=='COMPLETE' and valid_json
    print(json.dumps({"verdict":report['verdict'],"run_completed":report['run_completed'],"report":str(path)}),flush=True)
    return 0 if success else 1
