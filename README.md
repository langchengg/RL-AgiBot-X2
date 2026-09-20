# AgiBot X2 Ground Recovery

## Overview and current status

HRS take-home: design an RL environment for recovery from lying on the back,
run a training experiment, and integrate recovery and telemetry with ROS 2.
**Step 2 core runtime verified** in the target VM. CPU PPO, native MuJoCo and a
real ROS diagnostic package work together. OSMesa offscreen rendering passes;
the interactive viewer is unstable on this VM.
X2 recovery control, its Gymnasium environment, final ROS nodes, training and evaluation
remain **Pending**. The diagnostic package does not implement recovery behavior.
Evaluation: **Not evaluated**.

The two-page task brief, *AgiBot X2 Ground Recovery — Environment Design and ROS 2
Integration*, defines acceptance. References below provide implementation context.

## Target environment

Ubuntu 24.04 ARM64 in Parallels, ROS 2 Jazzy, and MuJoCo.
These are project choices, not additional requirements imposed by the task PDF.

## Observed environment

Checks rerun on 2026-09-20 inside the existing Parallels VM:

| Item | Observed value |
| --- | --- |
| OS / virtualization | Ubuntu 24.04.5 LTS; Linux 7.0.0-31-generic; aarch64/arm64; Parallels |
| VM allocation | 8 CPUs; approximately 11 GiB RAM and 4 GiB swap |
| Disk | 62 GiB filesystem, 40 GiB available after installation (`df -h`, rounded) |
| Python | System `/usr/bin/python3` and project `.venv/bin/python`: 3.12.3 |
| Runtime identity | Installed executable uses `<checkout>/.venv/bin/python`; prefix `<checkout>/.venv`; base prefix `/usr` |
| ROS | Jazzy in `/opt/ros/jazzy`; rclpy 7.1.12; sensor_msgs, std_msgs, std_srvs 5.3.8 |
| Build tools | colcon-core 0.21.3; setuptools 68.1.2; ament-package 0.16.5 |
| Simulation / RL | MuJoCo 3.13.0; Gymnasium 1.3.0; Stable-Baselines3 2.9.0; PyTorch 2.8.0+cpu |
| Other direct dependencies | NumPy 1.26.4; cffi 2.1.1; Pillow 10.2.0 |
| Graphics | Virtio virtual VGA and renderD128; Mesa virgl, desktop OpenGL 4.0; DISPLAY=:0 |
| Rendering libraries | Python glfw 2.10.2; PyOpenGL 3.1.10; libosmesa6 25.1.7 |
| PyTorch backends | CPU used explicitly; CUDA=false, MPS=false; 2 compute threads, 1 interop thread |

The venv uses system site-packages: NumPy/Pillow come from Ubuntu, rclpy and interface
modules from Jazzy, and MuJoCo/Gymnasium/SB3/Torch from the venv. The installed diagnostic
prints module paths and interpreter identity in one process. No system Python packages
were replaced. Virtual graphics capability does not establish a training GPU.

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

## Implementation route

These project choices are separate from the PDF requirements:

- Official X2 Ultra v1.3.0 at `60c5de582c523cd188f563819e62d34cfdc3d2d0`, using native
  MuJoCo Python. Selecting this version does not approve its physical limits.
- Future environment: Gymnasium; learning: SB3 PPO with MlpPolicy, initially CPU.
- One ament_python package, eventually containing Python/rclpy recovery and telemetry
  nodes plus one launch file. Only the diagnostic entry point exists now.
- Recovery will directly use the same environment class as training/evaluation.
- Actuation semantics, reward, X2 training budget and success thresholds remain future work.

No simulator bridge, ONNX, ros2_control or alternative simulation framework is required.

## Development and verification status

### Setup and fresh build

Run from the repository root. Jazzy and the existing Ubuntu Python/colcon packages are
prerequisites; do not install rclpy from PyPI. The inspected `/usr/bin/colcon` shebang uses
system Python, so invoke it explicitly with the venv interpreter.

```bash
source /opt/ros/jazzy/setup.bash
# Only if .venv does not already exist:
# /usr/bin/python3 -m venv --system-site-packages .venv
touch .venv/COLCON_IGNORE
.venv/bin/python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
.venv/bin/python -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python /usr/bin/colcon list --base-paths src
STEP2_DIR="$(mktemp -d /tmp/hrs-x2-step2.XXXXXX)"
.venv/bin/python /usr/bin/colcon --log-base "$STEP2_DIR/final-log" build \
  --base-paths src --packages-select x2_recovery --symlink-install \
  --build-base "$STEP2_DIR/final-build" --install-base "$STEP2_DIR/final-install"
source "$STEP2_DIR/final-install/setup.bash"
head -1 "$STEP2_DIR/final-install/x2_recovery/lib/x2_recovery/runtime_check"
```

PyPI metadata and official wheel listings were checked before installation: Gymnasium
and SB3 have universal wheels; Torch has a cp312 Linux aarch64 CPU wheel. SB3 2.9.0
accepts Torch >=2.8 and Gymnasium <2.0; these pins retain working MuJoCo/NumPy and avoid
CUDA packages or source builds. requirements.txt pins direct dependencies and the cffi
compatibility requirement from inherited PyNaCl; it is not a transitive lockfile.
No apt/build-tool changes were needed. Build outputs and raw logs stay in the temporary
directory; colcon discovers only src, and the venv has COLCON_IGNORE.

### Model source

The existing external checkout is reused, not vendored or modified:

```bash
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
git -C "$X2_ASSET_REPO" rev-parse HEAD
git -C "$X2_ASSET_REPO" status --short
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
```

Observed revision: `60c5de582c523cd188f563819e62d34cfdc3d2d0`; worktree clean.
For a new checkout, the previously verified download sequence is:

```bash
git clone --filter=blob:none --no-checkout https://github.com/AgibotTech/agibot_x2_urdf.git "$X2_ASSET_REPO"
git -C "$X2_ASSET_REPO" sparse-checkout set X2_URDF-v1.3.0
git -C "$X2_ASSET_REPO" checkout --detach 60c5de582c523cd188f563819e62d34cfdc3d2d0
```

The upstream Mulan PSL v2 license stays with the assets. No project license is granted
by this work; ROS package metadata uses UNLICENSED.

### Executed checks

Using the sourced overlay and the variables above:

```bash
timeout --kill-after=5s 120s ros2 run x2_recovery runtime_check runtime --x2-scene "$X2_SCENE"
MUJOCO_GL=osmesa timeout --kill-after=5s 40s ros2 run x2_recovery runtime_check render \
  --x2-scene "$X2_SCENE" --output "$STEP2_DIR/frames-osmesa"
```

The PPO dependency/runtime smoke test uses one Pendulum-v1 environment without rendering,
seed 42, MlpPolicy, device=cpu, n_steps=128, batch_size=64, n_epochs=2 and 1024 timesteps.
It checks finite data and parameters, an actual parameter change, checkpoint save/load,
and 32 deterministic prediction steps. PPO has a 60 s internal deadline and the complete
runtime command a 120 s external bound. Its temporary checkpoint is removed automatically.
These are compatibility checks, not X2 hyperparameters, convergence evidence, A3 training,
or any of the five A5 evaluation episodes.

Cross-process communication was checked with matching Fast DDS, domain 42, localhost-only
discovery and reliable/volatile keep-last-10 topic QoS. No global networking changes:

```bash
export ROS_DOMAIN_ID=42 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
timeout --kill-after=5s 35s ros2 run x2_recovery runtime_check serve --seconds 20 > "$STEP2_DIR/server.log" 2>&1 &
smoke_pid=$!
trap 'kill "$smoke_pid" 2>/dev/null || true' EXIT
timeout --kill-after=5s 25s ros2 run x2_recovery runtime_check client --seconds 15
client_exit=$?
wait "$smoke_pid"
server_exit=$?
trap - EXIT
cat "$STEP2_DIR/server.log"
test "$client_exit" -eq 0 && test "$server_exit" -eq 0
```

The client waits for discovery and repeated publications, verifies String content,
JointState names/finite changing positions/timestamps, then checks the Trigger response.
The server reads `probe_hinge` qpos from the tiny simulated scene. Endpoints are only
`/x2_smoke/status`, `/x2_smoke/joint_states`, `/x2_smoke/ping`; none implement the final
recovery interfaces, busy rejection or recovery timeout behavior.

| Check (rerun this step) | Status | Observed evidence |
| --- | --- | --- |
| Complete stack in one process | PASS | Installed ros2-run entry point imports all requested modules from the intended runtime |
| pip check; NumPy/Torch interop | PASS | No broken requirements; shared-array conversion and finite CPU forward/backward gradients |
| Fresh ROS package build | PASS | One x2_recovery ament_python package; final fresh build 0.72 s; entry point shebang uses venv |
| Simple MuJoCo physics | PASS | 100 warm-up + 2000 timed steps; 4.0 simulated s; finite state, no warnings |
| Simple-scene timing | PASS | Load 0.000776 s; timed stepping 0.001944 s, approximately 1.03 million steps/s; loading/warm-up excluded |
| Pinned X2 regression | PASS | nq=38, nv=37, nu=31; 100 steps / 0.1 s, finite state, no warnings |
| PPO dependency/runtime smoke test | PASS | 1024 steps, 16 optimization epochs; 1.305 s setup/learning, 1.329 s including reload/rollout; max parameter change 0.0106144 |
| Cross-process ROS topics/service | PASS | 3 String and 3 JointState messages received; positions 0.392532, 0.383565, 0.371238 rad; Trigger success=true; both processes exit 0 |
| OSMesa offscreen frames | PASS | 320x240 simple/X2 PNGs saved and visually inspected: visible hinge/floor and complete robot/floor |
| Automatic GLFW desktop rendering | FAIL | Image-validity assertion failed; forcing X11 on the selected Wayland library was unsupported |
| Interactive viewer | FAIL | X11 variant opened for 2 s, but process exited with segmentation fault; software retry hit GLXBadDrawable and 20 s timeout |
| Recovery, robot training/evaluation | NOT RUN | A1-A6 Pending; evaluation Not evaluated |

The simple-scene timing is a short runtime measurement, not an X2 training-speed estimate.
OSMesa uses the already installed libosmesa6 25.1.7; no graphics packages/drivers were
changed. Interactive capability is unverified beyond opening a window. The bounded failed
viewer attempts used `PYGLFW_LIBRARY_VARIANT=x11 MUJOCO_GL=glfw` and then additionally
`LIBGL_ALWAYS_SOFTWARE=1`, each with `timeout --kill-after=5s 20s ros2 run x2_recovery runtime_check viewer`.
All diagnostic processes were bounded and have exited. No core checks are blocked;
the graphics limitation is isolated to desktop rendering/viewer reliability.

### Model limitations retained from the earlier audit

The selected model's 12 joint-range and 13 actuator-control-range discrepancies against
its URDF remain unresolved. Control ranges are not approved joint torque limits:
transmission/gear semantics must be inspected before configuring actuation. Its default
upright base at z=0.68 m is not a valid supine reset. Collision/limit behavior and reset
still require verification. Step 2 made no model edits and does not pass A1.

### Development history

The repository was established before implementation. Preserve meaningful commits and
push completed work throughout development (PDF p. 2, Commit history).

## References

Reviewed README content and GitHub directory trees on 2026-09-20:

- [SB3 package metadata](https://pypi.org/project/stable-baselines3/2.9.0/),
  [Gymnasium metadata](https://pypi.org/project/gymnasium/1.3.0/), and
  [official PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/).
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
