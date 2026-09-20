"""Pinned X2 Ultra v1.3.0 physics and simulator interface; no ROS or audit imports.

Upstream assets remain external, attributed to AgibotTech under Mulan PSL v2.
Only an in-memory MjSpec is edited. Each call compiles an independent model.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

REPOSITORY = "https://github.com/AgibotTech/agibot_x2_urdf"
COMMIT = "60c5de582c523cd188f563819e62d34cfdc3d2d0"
VARIANT = "X2_URDF-v1.3.0"
URDF = f"{VARIANT}/x2_ultra.urdf"
ROBOT = f"{VARIANT}/x2_ultra.xml"
SCENE = f"{VARIANT}/scene.xml"
SOURCE_HASHES = {
    "LICENSE": "d5777bea50dc2cd111845d8a9c51ab990d188ce343b843e5ad3e653b5fa8fdab",
    URDF: "f628a969aaf3d5785b8c8901611a85f358d7eb62a8b06b1216dc42291a72944e",
    ROBOT: "3be016eaa127c79ac547754d6e25c0930887efcf9577af33e9f5bb4ff4ed90d4",
    SCENE: "7e067253f0e4b0e023614d63a6f15b9c7b9108d7a4bf63bb9cee5cf2b5f24439",
}
# Project policy: intersect coordinate-equivalent source intervals, never widen.
RANGE_EDITS = {
    "waist_yaw_joint": ((-3.43, 2.382), (-3.43, 2.2078)),
    "head_yaw_joint": ((-.366, .366), (-.349, .349)),
    "left_wrist_pitch_joint": ((-.558, .558), (-.5236, .5236)),
    "right_wrist_pitch_joint": ((-.558, .558), (-.5236, .5236)),
    "left_wrist_roll_joint": ((-1.571, .724), (-1.5097, .724)),
    "right_wrist_roll_joint": ((-.724, 1.571), (-.724, 1.5097)),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, timeout=15).stdout.strip()


def resolve_assets(asset_repo=None):
    """Validate the pinned checkout offline, without modifying it or changing cwd."""
    path = Path(asset_repo or os.environ.get("X2_ASSET_REPO") or
                Path.home() / ".cache/hrs-x2-recovery/agibot_x2_urdf").expanduser().resolve()
    require(path.is_dir(), f"Missing asset repository: {path}; set X2_ASSET_REPO")
    try:
        require(git(path, "rev-parse", "HEAD") == COMMIT, "Asset revision is not the pinned commit")
        dirty = git(path, "status", "--porcelain", "--untracked-files=all", "--", VARIANT, "LICENSE")
        require(not dirty, f"Relevant upstream files must be unchanged: {dirty}")
        for relative, expected in SOURCE_HASHES.items():
            require(hashlib.sha256((path / relative).read_bytes()).hexdigest() == expected,
                    f"Source hash precondition failed: {relative}")
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Cannot verify pinned assets at {path}: {exc}") from exc
    return path


def rotation(quaternion):
    matrix = np.empty(9)
    mj.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def rpy_rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def urdf_inertia(link):
    inertial = link.find("inertial")
    origin = inertial.find("origin")
    com = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    rot = rpy_rotation(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    v = {k: float(x) for k, x in inertial.find("inertia").attrib.items()}
    tensor = np.array([[v['ixx'], v['ixy'], v['ixz']], [v['ixy'], v['iyy'], v['iyz']],
                       [v['ixz'], v['iyz'], v['izz']]])
    return float(inertial.find("mass").get("value")), com, rot @ tensor @ rot.T


def coordinate_check(model, joint):
    """Same parent/child frame, hinge axis/sign and zero; export precision allowed."""
    name = joint.get("name")
    j = model.joint(name).id
    b = model.jnt_bodyid[j]
    origin = joint.find("origin")
    pos = np.fromstring(origin.get("xyz"), sep=" ")
    rot = rpy_rotation(np.fromstring(origin.get("rpy"), sep=" "))
    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    errors = {"origin_m": float(np.max(abs(model.body_pos[b] - pos))),
              "rotation_matrix": float(np.max(abs(rotation(model.body_quat[b]) - rot))),
              "axis": float(np.max(abs(model.jnt_axis[j] - axis)))}
    ok = (joint.get("type") == "revolute" and model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE
          and model.body(b).name == joint.find("child").get("link")
          and model.body(model.body_parentid[b]).name == joint.find("parent").get("link")
          and np.max(abs(model.jnt_pos[j])) < 1e-12
          and model.qpos0[model.jnt_qposadr[j]] == 0
          and max(errors.values()) <= 2e-6 and joint.find("mimic") is None)
    require(ok, f"Unresolved joint coordinate equivalence: {name}: {errors}")
    return errors


def apply_overrides(spec, urdf):
    """Named, checked pre-compilation edits; no changes to source assets."""
    pelvis = spec.body("pelvis")
    require(not pelvis.explicitinertial and pelvis.mass == 0 and np.isnan(pelvis.fullinertia[0]),
            "Pelvis override precondition failed: expected inferred upstream inertia")
    mass, com, tensor = urdf_inertia(urdf.find("link[@name='pelvis']"))
    require(mass == 3.523487 and np.linalg.eigvalsh(tensor).min() > 0,
            "Unexpected URDF pelvis inertia")
    pelvis.mass, pelvis.ipos = mass, com
    pelvis.fullinertia = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    pelvis.explicitinertial = True
    edits = [{"target": "pelvis.inertial", "old": "absent; inferred from density-1000 collision mesh",
              "new": {"mass": mass, "com": com.tolist(), "tensor_at_com_in_body_frame": tensor.tolist()},
              "units": "kg, m, kg m^2", "source": URDF + "#pelvis/inertial",
              "reason": "Use explicit paired-source mass, COM and full tensor instead of mesh-density inference"}]
    for name, (old, new) in RANGE_EDITS.items():
        j = spec.joint(name)
        require(np.array_equal(j.range, old), f"Range override precondition failed: {name}")
        j.range = new
        edits.append({"target": name + ".range", "old": list(old), "new": list(new),
                      "units": "rad", "source": URDF + "#" + name,
                      "reason": "Project conservative intersection policy; not a hardware specification"})
    waist = spec.joint("waist_pitch_joint")
    require(waist.margin == 0, "Waist limit margin precondition failed")
    waist.margin = .005
    edits.append({"target": "waist_pitch_joint.margin", "old": 0., "new": .005,
                  "units": "rad", "source": "Step 4 natural settling diagnosis",
                  "reason": "Activate the unchanged soft limit early: gravity otherwise produces "
                            "0.00137 rad steady upper-bound violation. No range/effort widening."})
    return tuple(edits)


@dataclass(frozen=True)
class JointControl:
    ctrl_index: int
    actuator_name: str
    joint_name: str
    joint_id: int
    qpos_address: int
    dof_address: int
    gear: float
    gain: float
    position_range: tuple
    control_range: tuple
    effort_range: tuple
    joint_effort_range: tuple
    joint_type: str = "hinge"
    transmission: str = "joint"
    control_units: str = "N m (unit gain and gear)"
    limiting_stages: tuple = ("input ctrlrange", "joint actuatorfrcrange")


@dataclass
class LoadedModel:
    model: mj.MjModel
    mapping: tuple[JointControl, ...]
    asset_repo: Path
    overrides: tuple

    @property
    def qpos_addresses(self):
        return np.array([r.qpos_address for r in self.mapping], dtype=int)

    @property
    def dof_addresses(self):
        return np.array([r.dof_address for r in self.mapping], dtype=int)

    def joint(self, name):
        for row in self.mapping:
            if row.joint_name == name:
                return row
        raise KeyError(name)

    def read_state(self, data):
        """Independent arrays in actuator order (rad, rad/s); excludes free base."""
        return data.qpos[self.qpos_addresses].copy(), data.qvel[self.dof_addresses].copy()


def compiled_mapping(model):
    """Fail closed on transmissions outside the verified model's simple motors."""
    require((model.nq, model.nv, model.nu, model.njnt, model.na) == (38, 37, 31, 32, 0),
            "Unexpected X2 dimensions/mapping")
    require(model.opt.disableflags == 0 and model.opt.disableactuator == 0,
            "Unexpected disabled physics/control clamping")
    rows = []
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        require(model.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT and 0 <= j < model.njnt,
                f"Unexpected transmission for actuator {a}")
        require(model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]
                and model.jnt_actfrclimited[j], f"Unexpected joint or limits for actuator {a}")
        require(model.actuator_dyntype[a] == mj.mjtDyn.mjDYN_NONE
                and model.actuator_gaintype[a] == mj.mjtGain.mjGAIN_FIXED
                and model.actuator_biastype[a] == mj.mjtBias.mjBIAS_NONE
                and model.actuator_gainprm[a, 0] == 1
                and np.array_equal(model.actuator_gear[a], [1, 0, 0, 0, 0, 0])
                and model.actuator_ctrllimited[a] and not model.actuator_forcelimited[a],
                f"Unsupported motor semantics for actuator {a}")
        require(model.actuator_ctrladr[a] == a and model.actuator_ctrlnum[a] == 1
                and model.actuator_outadr[a] == a and model.actuator_outnum[a] == 1
                and model.actuator_delay[a] == 0, f"Unexpected input/output layout for actuator {a}")
        name = model.joint(j).name
        require(model.actuator(a).name == "motor_" + name, f"Unexpected actuator name: {a}")
        lo, hi = model.actuator_ctrlrange[a]
        jl, jh = model.jnt_actfrcrange[j]
        require(np.isfinite([lo, hi, jl, jh, *model.jnt_range[j]]).all()
                and lo < hi and jl < jh, f"Invalid bounds for {name}")
        rows.append(JointControl(a, model.actuator(a).name, name, j,
                                 int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), 1., 1.,
                                 tuple(model.jnt_range[j]), (lo, hi), (max(lo, jl), min(hi, jh)),
                                 (jl, jh)))
    expected = set(np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_HINGE))
    require(len({r.joint_id for r in rows}) == model.nu and {r.joint_id for r in rows} == expected,
            "Duplicate, unmatched or unactuated scalar-joint mapping")
    require(len({r.qpos_address for r in rows}) == model.nu
            and len({r.dof_address for r in rows}) == model.nu, "Duplicate state address")
    base = model.joint("floating_base_joint").id
    require(model.jnt_type[base] == mj.mjtJoint.mjJNT_FREE
            and model.jnt_qposadr[base] == 0 and model.jnt_dofadr[base] == 0,
            "Unexpected floating-base layout")
    return tuple(rows)


def load_effective_model(asset_repo=None):
    """Offline, fresh, deterministic X2 model. Does not render, probe or write files."""
    require(mj.__version__ == "3.13.0", "This loader is verified for MuJoCo 3.13.0")
    path = resolve_assets(asset_repo)
    urdf = ET.parse(path / URDF).getroot()
    spec = mj.MjSpec.from_file(str(path / SCENE))
    overrides = apply_overrides(spec, urdf)
    model = spec.compile()
    for joint in urdf.findall("joint"):
        if joint.get("type") != "fixed":
            coordinate_check(model, joint)
    mapping = compiled_mapping(model)
    mass, com, tensor = urdf_inertia(urdf.find("link[@name='pelvis']"))
    b = model.body("pelvis").id
    rot = rotation(model.body_iquat[b])
    require(model.body_mass[b] == mass and np.allclose(model.body_ipos[b], com, atol=1e-12, rtol=0)
            and np.allclose(rot @ np.diag(model.body_inertia[b]) @ rot.T, tensor, atol=1e-10, rtol=0),
            "Compiled pelvis override verification failed")
    return LoadedModel(model, mapping, path, overrides)
