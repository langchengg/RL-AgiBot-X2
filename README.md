# AgiBot X2 Ground Recovery

## Overview and current status

HRS take-home: design an RL environment for recovery from lying on the back,
run a training experiment, and integrate recovery and telemetry with ROS 2.
Repository setup and requirements capture are complete. Native MuJoCo loading and a
short headless stepping check pass for one official X2 model candidate.
Recovery simulation, RL environment, ROS nodes, training and evaluation remain **Pending**.
Evaluation: **Not evaluated**.

The two-page task brief, *AgiBot X2 Ground Recovery — Environment Design and ROS 2
Integration*, defines acceptance. References below provide implementation context.

## Target environment

Ubuntu 24.04 ARM64 in Parallels, ROS 2 Jazzy, and MuJoCo.
These are project choices, not additional requirements imposed by the task PDF.

## Observed environment

Dependency validation on 2026-09-20, executed inside the target VM:

| Item | Observed fact |
| --- | --- |
| Platform | Ubuntu 24.04.5 LTS; Linux 7.0.0-31-generic; aarch64; dpkg architecture arm64 |
| Virtualization | `systemd-detect-virt`: parallels; platform identifies as Parallels ARM Virtual Machine |
| VM resources | 8 available CPUs; `free -h`: 11 GiB RAM, 4.0 GiB swap (rounded) |
| Python | System `/usr/bin/python3` and project `.venv/bin/python`: Python 3.12.3 |
| ROS 2 | `/opt/ros/jazzy/setup.bash` exists; sourced in a child shell; ROS_DISTRO=jazzy |
| ROS tools | `ros2` and `colcon` found; system and venv Python import Jazzy rclpy and all three required interface types |
| MuJoCo | 3.13.0 Python bindings and native library load in `.venv`; absent from system Python |
| Dependencies | NumPy 1.26.4; `.venv/bin/python -m pip check` passes |
| RL libraries | Gymnasium, Stable-Baselines3 and PyTorch are not installed or validated |

The venv inherits system packages for ROS compatibility. Adding cffi 2.1.1 inside the
venv resolved an inherited PyNaCl dependency check failure; system packages were unchanged.
ROS communication, rendering, GPU capability and the training stack remain unverified.

## Acceptance criteria

The matrix defines required evidence; Pending items have not passed acceptance.
Source abbreviations refer to sections of the supplied task PDF:
**SIM** = p. 1, Simulation and reinforcement learning; **ROS** = p. 1, ROS 2 integration;
**IF** = p. 2, ROS 2 interfaces; **VAL** = p. 2, Validation;
**DOC** = p. 2, README requirements.

| ID | Deliverable | Acceptance criteria / required evidence | Source | Status |
| --- | --- | --- | --- | --- |
| A1 | Simulation and robot model | Select an AgiBot X2 URDF and simulator; load a floating base on a flat floor. Each episode resets to resting on the back without floor intersection. Configure collisions and respect joint/actuator limits. Document source, version, modifications and simulation settings; provide model-load, reset and limit evidence. Upstream XML availability alone is insufficient. | SIM; DOC | Pending |
| A2 | RL environment | Define observations, actions, reward, reset and episode end conditions. Explain reward terms, weights, rationale and recovery incentives, plus environment simplifications. Dimensions, gains, weights and thresholds must follow actual design and validation. | SIM; DOC | Pending |
| A3 | Training experiment | Actually run PPO or another RL algorithm. Save any produced policy checkpoint and a real training reward plot. Record algorithm, training settings and compute resources. No fabricated or placeholder data. Zero success does not waive training; document any genuine training blocker and apply A5's baseline fallback. | SIM; VAL; DOC | Pending |
| A4 | ROS 2 integration | Python or C++ package with recovery and telemetry nodes; colcon build and one launch file for both. Connect a real simulator. Accept before executing one episode with an available checkpoint or scripted baseline; reject concurrent requests. Publish simulator-derived status and actual joint states. Configurable timeout sends unsuccessful attempts to FAILED. Telemetry subscribes to both topics and logs status plus one joint position. Command targets or fabricated values are not measured telemetry. Docker is optional. | ROS; IF | Pending |
| A5 | Evaluation | Run five simulation episodes with the available policy; report success count and failures. Success means upright, stable standing on both feet without other body support. Define and document concrete success checks; no numerical thresholds are prescribed. Zero successes is acceptable with explanation. If training is blocked, evaluate and clearly label a scripted baseline. Reasonable hand contact during rising is not itself a failure of the final standing check. | VAL; DOC | Pending |
| A6 | Reproducibility and end-to-end validation | Verify a fresh ROS 2 build, one launch command for both nodes, and a CLI request that actually starts recovery in the simulator with live joint telemetry from that episode. Verify busy rejection and unsuccessful timeout to FAILED. Record actual commands and outcomes. Final README must cover dependencies/setup, simulation/training/evaluation commands, model and compute resources, environment/reward design, training settings, success checks/results, failure analysis/improvements, node responsibilities and build/launch/service/topic commands plus simulator integration. | VAL; DOC | Pending |

An empty-workspace build is not evidence for A6. Evaluation has not been run.

## ROS 2 interface contract

| Interface | Type | Required behavior |
| --- | --- | --- |
| `/x2/start_recovery` | `std_srvs/srv/Trigger` | Return `success=true` when accepted, before episode execution; `success=false` when busy. Acceptance does not mean the robot has stood up. |
| `/x2/recovery_status` | `std_msgs/msg/String` | Publish `IDLE`, `RUNNING`, `SUCCEEDED` or `FAILED`. |
| `/x2/joint_states` | `sensor_msgs/msg/JointState` | Publish joint names, actual simulator positions and timestamps while running. |

Each accepted request executes exactly one episode. An unsuccessful attempt reaches
FAILED at its configurable timeout. Keep the required Trigger service and message types.
The telemetry node subscribes to joint states and recovery status and logs both the
current status and one joint position. Source: PDF p. 1 ROS; p. 2 IF and VAL.

## Planned implementation choices

- Use MuJoCo's native Python API on the target Ubuntu/Jazzy ARM64 environment.
- Consider Gymnasium with Stable-Baselines3 PPO, initially using CPU training.
- Implement both nodes in Python/rclpy.
- Prefer direct calls from the recovery node into a shared environment implementation.
- Determine the final model version, actuator setup, reward, training budget, success
  thresholds and dependency versions only after inspection and verification.

These are plans, not implemented features or PDF mandates. Reference projects do not
make ONNX, legged_control2, ros2_control or a separate simulator bridge dependencies.

## Development and verification status

The following commands validate dependencies and an unmodified official model. They do
not implement a recovery reset, controller, ROS package or training experiment.

### Dependency and model checks

From the project root, create the venv and install the checked versions:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --only-binary=:all: mujoco==3.13.0 cffi==2.1.1
source /opt/ros/jazzy/setup.bash
.venv/bin/python -c 'import mujoco, rclpy; from sensor_msgs.msg import JointState; from std_srvs.srv import Trigger; from std_msgs.msg import String; print(mujoco.__version__, mujoco.mj_versionString())'
.venv/bin/python -m pip check
```

Observed: both MuJoCo versions are 3.13.0; imports pass; no broken requirements.
Local supporting packages: absl-py 2.5.0, etils 1.14.0, fsspec 2026.9.0, glfw 2.10.2,
PyOpenGL 3.1.10 and pycparser 3.0. This is a validated snapshot, not a complete lockfile.

Keep the upstream checkout outside this repository. For the initial download:

```bash
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
git clone --filter=blob:none --no-checkout https://github.com/AgibotTech/agibot_x2_urdf.git "$X2_ASSET_REPO"
git -C "$X2_ASSET_REPO" sparse-checkout set X2_URDF-v1.3.0
git -C "$X2_ASSET_REPO" checkout --detach 60c5de582c523cd188f563819e62d34cfdc3d2d0
```

The checkout retains the upstream Mulan PSL v2 license and has no local changes.
v1.3.0 Ultra is the initial loading candidate, not a final recovery-model selection.
It supplies a paired URDF, native MJCF and scene, avoiding conversion for this check.

Headless load and stepping check (no renderer required):

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import mujoco as mj
import numpy as np
scene = Path(os.environ["X2_ASSET_REPO"]) / "X2_URDF-v1.3.0/scene.xml"
m = mj.MjModel.from_xml_path(str(scene))
d = mj.MjData(m)
mj.mj_forward(m, d)
print("nq nv nu:", m.nq, m.nv, m.nu, "initial contacts:", d.ncon)
for _ in range(100):
    mj.mj_step(m, d)
assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
assert np.isclose(d.time, 0.1) and not np.any(d.warning.number)
print("time:", d.time, "warnings:", d.warning.number.tolist())
PY
```

Observed: `nq=38`, `nv=37`, `nu=31`, zero initial contacts, 0.1 s advanced and
all warning counters zero. This checks loading and short stepping, not standing or recovery.

Model audit findings at the pinned revision:

- One free joint, 31 limited hinge joints, 31 control-limited motors and a plane floor;
  timestep 0.001 s. No separate actuator force limits are enabled.
- All 31 actuated joint names match the URDF. The first hinge uses qpos index 7 and
  qvel index 6: floating-base coordinates must not be published as named hinge positions.
- Twelve joint position ranges and thirteen motor control ranges differ from the URDF.
  For example, waist yaw has upper bound 2.382 rad in MJCF versus 2.2078 in URDF;
  head yaw spans ±0.366 versus ±0.349 rad. Hip/knee motors use ±118 versus URDF ±120;
  wrist pitch/roll use ±2.2 versus URDF ±4.8. Reconcile limits before recovery control.
- The default base is upright at z=0.68 m, not resting supine. Collision masks include
  both disabled visual geoms and enabled collision geoms; a valid supine reset and
  collision/limit behavior still require verification. No model edits were made.

### Development history

The repository was established before implementation. Preserve meaningful commits and
push completed work throughout development (PDF p. 2, Commit history).

## References

Reviewed README content and GitHub directory trees on 2026-09-20:

- [MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html) and
  [PyPI distribution](https://pypi.org/project/mujoco/3.13.0/): native bindings and ARM64 wheel.
- Task PDF: p. 1, Simulation and reinforcement learning / ROS 2 integration;
  p. 2, ROS 2 interfaces / Validation / GitHub repository submission / README requirements.
- [AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf), inspected revision
  `60c5de582c523cd188f563819e62d34cfdc3d2d0`. Root and version READMEs and tree confirm
  [v1.3.0](https://github.com/AgibotTech/agibot_x2_urdf/tree/60c5de582c523cd188f563819e62d34cfdc3d2d0/X2_URDF-v1.3.0):
  `x2_ultra.urdf`, `x2_ultra_simple_collision.urdf`, `x2_ultra.xml`, `x2_fist.urdf`,
  `x2_fist.xml`, `scene.xml`; and
  [v1.4.0](https://github.com/AgibotTech/agibot_x2_urdf/tree/60c5de582c523cd188f563819e62d34cfdc3d2d0/X2_URDF-v1.4.0):
  `X2-Ultra.urdf`, `X2-EDU.urdf`, their `_simple_collision.urdf` variants,
  `X2-Ultra.xml`, `X2-EDU.xml`, `scene.xml`. Both have mesh directories.
  Scene XML reads confirm each includes its Ultra XML and a plane floor.
  v1.3.0 loading results and unresolved model checks are recorded above; v1.4.0 was not loaded.
- [ioai-tech/humanoid_controller](https://github.com/ioai-tech/humanoid_controller), inspected revision
  `2f94ff0ebd072e255d90bbbb2122d0f1a622a0b3`. README describes ROS 2 Humble,
  legged_control2 and ONNX inference with MuJoCo launch; tree includes
  `launch/mujoco.launch.py` and `config/agibot/x2/sim.yaml`.
  Its jammy-humble-amd64 installation route is not transferable to Jazzy ARM64.
  No apt sources or dependencies were adopted; compatibility and execution are unverified.
- [Woolfrey/mujoco_ros2](https://github.com/Woolfrey/mujoco_ros2), inspected revision
  `43ae110ea8aff9e5b482aa1e206b1f63217645e5`. README describes JointState publishing
  and command-topic input; tree includes `src/mujoco_ros.cpp`, `src/nodes/mujoco_node.cpp`
  and `include/mujoco_ros2/mujoco_ros.hpp`. Its declared tested stack is Ubuntu 22.04,
  ROS 2 Humble and MuJoCo 3.2.0. This is not evidence for Jazzy ARM64 or X2 compatibility.
  Reading `update_simulation()` confirmed qpos-to-position mapping, timestamps and publishing.
  Its joint indexing has not been validated for X2's floating base.
  Used only as a state-publication reference; no source was copied or executed.
