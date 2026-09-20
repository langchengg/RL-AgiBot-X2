# AgiBot X2 Ground Recovery

## Overview and current status

HRS take-home: design an RL environment for recovery from lying on the back,
run a training experiment, and integrate recovery and telemetry with ROS 2.
**Step 4 physics verified; visualization closeout in progress** in the target VM. The pinned official model has
one effective loader, source-consistent pelvis inertia, conservative joint ranges and
verified state/control mappings. CPU PPO and ROS compatibility were verified in Step 2.
OSMesa recording works. The Step 4 live window uses native MuJoCo visualization and
existing X11 GLFW with process-local Mesa software rendering; virgl gives black X2 frames.
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
- Simulator torque semantics are verified below. RL actions, controller, reward, X2
  training budget and success thresholds remain future work.

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

The existing external checkout is reused, not vendored or modified. Source:
[AgibotTech/agibot_x2_urdf](https://github.com/AgibotTech/agibot_x2_urdf),
commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`, **X2 Ultra v1.3.0**.
Reference URDF: `X2_URDF-v1.3.0/x2_ultra.urdf`; robot MJCF:
`X2_URDF-v1.3.0/x2_ultra.xml`; scene: `X2_URDF-v1.3.0/scene.xml`
(includes only that robot MJCF). Runtime loading is offline after setup:

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

### Step 2 checks (historical)

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

| Check (rerun in Step 2) | Status | Observed evidence |
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

### Step 3: effective model and evidence (historical baseline at 0c627d7)

`x2_recovery.model.load_effective_model(asset_repo=None)` is the only normal X2
loading path, including existing runtime/render diagnostics. It accepts a repository
path or `X2_ASSET_REPO`, otherwise the cache path under `Path.home()`. It checks the
pinned Git revision, relevant dirty/untracked files and XML/license hashes. It uses
MuJoCo 3.13.0 `MjSpec.from_file`, named edits and recompilation, preserving relative
includes/meshes without changing cwd. Every call returns a fresh model; no ROS,
Torch, SB3, rendering, probes, reports or network access are imported/performed by
this module. The audit alone compiles an explicitly labeled unmodified baseline.
Incompatible assets, patch preconditions and mappings raise explicit errors.

Seven in-memory overrides are applied, with old/new values and reasons in the report:

- **Pelvis inertial:** source MJCF omits it. AUTO mass inference includes the
  density-1000 collision mesh; visual meshes have density zero. Disabling contact
  masks leaves inferred mass unchanged, and doubling density doubles it in a
  disposable compilation. Raw pelvis mass **5.031810659 kg** becomes URDF
  **3.523487 kg**; robot total **43.474796659 → 41.966473 kg**.
  COM changes from `[-0.00152667, 0.00012933, 0.00289836]` to
  `[-0.001209, -0.000023, -0.002011]` m in the pelvis frame.
  The full tensor at COM, in pelvis axes, is
  `[[0.0126, 0, 0.000163], [0, 0.007924, -0.000005],
  [0.000163, -0.000005, 0.012417]]` kg m². `fullinertia` preserves the
  off-diagonals; the loader verifies the reconstructed compiled tensor.
- **Six range intersections (rad):** waist yaw `[-3.43,2.382] → [-3.43,2.2078]`;
  head yaw `±0.366 → ±0.349`; both wrist pitches `±0.558 → ±0.5236`;
  left wrist roll `[-1.571,0.724] → [-1.5097,0.724]`; right wrist roll
  `[-0.724,1.571] → [-0.724,1.5097]`. These are conservative **project policies**,
  not new official hardware specifications. Axis/sign/zero and parent-child
  transforms were checked first. No mass scaling, balancing, collision geometry,
  friction, solver or actuator-bound changes were made.

Recomputed differences: **12 joint ranges**, comprising six narrower retained
intervals and six wider intersected intervals; **zero rounding-only differences**
at the declared 1e-4 rad classification threshold. **13 effort-envelope differences**:
eight hip/knee motors plus waist yaw retain ±118 N m versus URDF ±120; four wrist
pitch/roll motors retain ±2.2 versus ±4.8. Other bounds agree. All 31 motors have
unit gain/gear, no bias or activation state, enabled input limits and enabled joint
actuation limits. Input clipping dominates the equal/wider downstream joint clamp;
that downstream clamp was **not independently exercised**. Absence of an actuator
`forcerange` does not mean unlimited joint effort. URDF speed limits are reported
but **not enforced** by this route; no hidden speed clamp was added.

| Audit area | Result | Measured evidence and scope |
| --- | --- | --- |
| Floating base | PASS | root pelvis (body 1), free joint 0; qpos address 0 (7 coordinates), DOF address 0 (6 velocities); nq=38, nv=37, nu=31; no equality, mocap, tendon, gravcomp or callbacks. Contact-free whole-robot COM drops 0.0495405 m in 100 ms, matching semi-implicit Euler gravity; valid independent translation/rotation tested. |
| Mass/inertia | PASS | 32 dynamic bodies; effective 41.966473 kg; qpos0 COM world `[0.00168761,0.00024727,0.71173956]` m. Positive principal moments and triangle inequalities; full common-frame tensors match URDF within export precision (max mass error 3.8e-5 kg, tensor error 4.65e-7 kg m²). Fixed sensor/base frames add no omitted mass. |
| Collision coverage | PASS, bounded poses | Back, both forearms, wrist/hand proxies, knee/shin meshes and feet have actual active floor force. Peak individual normal forces: back 448.54 N; forearms L/R 468.11/477.70 N; wrist proxies 77.21/141.48 N; shins 564.27/509.91 N; feet 344.44/386.04 N. Six 180 ms whole-model probes; largest floor penetration 10.99 mm (<15 mm), required-pose self penetration 2.66 mm (<5 mm); no warnings/reset. Four collision-only OSMesa images inspected. |
| Joint ranges | PASS | 31 coordinate-equivalent revolute joints, no unmatched movable/mimic joints; six restrictions, six stricter source ranges retained, 19 exact agreements. |
| Actuator semantics | PASS | 155 fresh forward checks: zero, ±40% and ±105% source inputs for every motor; sparse transmission moments, sign and measured actuator/joint forces agree within 1e-10 N m. Raw ctrl remains unclipped in its buffer. |
| Effective limits | PASS, stated scope | All finite joint bounds enabled. 62 lower/upper probes with 0.1 N m outward input, 80 ms each; explicit named limit constraints and restoring acceleration. Disposable copies disable contact/frictionloss and gravity only. Initial/peak penetration 0.002 rad (<0.005); max final 0.000270 rad (<0.001). Does not certify full-collision reachability of every bound. |
| Indexing | PASS | 31 unique controlled hinges; distinct nonzero positions/velocities and isolated inputs verified. Head motors occur before arms in ctrl order but after arms in joint traversal. No magic scalar-state slice or joint-ID/ctrl-index assumption. |

Settings remain timestep 0.001 s, gravity `[0,0,-9.81]` m/s², Euler integrator,
Newton solver, pyramidal cone, tolerance 1e-8, 100 iterations, all global disable
flags zero. The strict JSON report includes complete settings, inertias, geoms,
limits, actuator/mapping tables, source and project identities/hashes, definitions,
measurements, warning/time checks and evidence scopes. Quaternions are unit `wxyz`;
free-base linear velocity is world-frame and angular velocity is body-local.
The installed header/API behavior and
[MuJoCo 3.13.0 source](https://github.com/google-deepmind/mujoco/tree/3.13.0)
were checked, including sparse moments, clamp order and quaternion integration.

Mesh contact uses convex hulls. Each foot uses twelve 5 mm spheres, not its detailed
visual mesh; the floor is an infinite plane. No articulated hands/fingers exist.
Two original arm-at-side impact probes remain labeled **FAIL** in the report's
pose investigation: wrists become trapped against hip hulls (6.21/5.94 mm soft
penetration). The required arm-floor probes move the shoulder to -0.6 rad to
separate these contacts; physics, duration and thresholds stay unchanged.
This classifies the failure rather than claiming arbitrary-pose robustness.
The later supine reset must explicitly avoid such limb trapping. No episode-ready
reset, hardware calibration or recovery trainability is established. **A1-A6 stay
Pending; evaluation stays Not evaluated.** Next: implement and validate a reusable
supine reset, without starting training or controller design in this step.

These historical measurements and original reports are preserved. Step 4 adds a
conservative waist limit activation margin in the same loader; see its new fingerprint,
revalidation and reset evidence below. Old visual approvals are not reused for that model.

### Step 3 closeout review

The closeout fixed a real evidence-reuse gap: earlier reviews checked only image
bytes. It also records ground penetration at initialization, every-step maximum
(with time/pair), and the final coherent state. Previously only the maximum scalar
and four sparse contact snapshots were saved, so the peak pair was unrecoverable.
The one requested fresh-directory audit supplied that missing evidence; no extra
pose search or model/physics changes were made. The effective-model fingerprint
is unchanged. Four targeted closeout tests passed, including same-path invalidation,
changed physics/probes with identical pixels, changed pixels, and Euler expectations.

Ground penetration below is in **mm**, sampled initially and after every 1 ms step;
final means 180 ms. All initial states had **0 penetration, no contact pair**, and
2 mm clearance to the listed nearest floor/robot pair. `0` is the floor geom.
These are the maxima across all robot-floor contacts, not just the named region.

| Pose | Initial nearest pair (no contact) | Whole-run maximum: mm / pair / time | Final: mm / pair |
| --- | --- | --- | --- |
| feet | 0–15 | 1.394 / 0–18 / 40 ms | 0.265 / 0–18 |
| back | 0–53 | 10.991 / 0–67 / 162 ms | 8.848 / 0–2 |
| left_arm | 0–67 | 9.863 / 0–22 / 154 ms | 4.324 / 0–24 |
| right_arm | 0–80 | 10.763 / 0–44 / 155 ms | 4.997 / 0–46 |
| left_shin | 0–11 | 7.223 / 0–56 / 132 ms | 1.326 / 0–56 |
| right_shin | 0–34 | 7.246 / 0–69 / 132 ms | 1.373 / 0–69 |
| left_arm_at_side (FAIL retained) | 0–67 | 10.495 / 0–22 / 171 ms | 9.485 / 0–22 |
| right_arm_at_side (FAIL retained) | 0–80 | 10.191 / 0–44 / 172 ms | 9.413 / 0–44 |

Geom IDs: 2 pelvis; 11/34 left/right knee-shin mesh; 15/18/22/24 left foot
sphere proxies; 44/46 right foot sphere proxies; 53 torso; 56/69 left/right shoulder
pitch hulls; 67/80 left/right wrist-roll hulls. Complete body/local-index labels
remain in the JSON geom inventory. All **62 joint-limit sides** already recorded
whole-run and final violations: every maximum is 0.002 rad (the deliberately
initialized overshoot); largest final violation is **0.000269485464 rad**, left
wrist yaw lower bound. Individual final values remain in the existing limit table.

The **15 mm** floor threshold is a permissive project screening cutoff for these
180 ms passive impacts under the unchanged source soft-contact settings. It was
chosen before the formal suite to bound transient overlap; it is **not derived
from an official specification, hardware calibration or a proven accuracy bound**.
It does not establish contact fidelity, zero penetration or settled support, and
must not be transferred to future reset acceptance. In particular, the back probe's
8.848 mm final floor penetration means this trajectory is not an accepted resting
supine reset.

The two original arm-at-side failures start from valid, contact-free states. Their
wrist/hip-yaw geom pairs **9–67 / 32–80** first contact at **116 / 115 ms**, exceed
the unchanged 5 mm self-contact threshold at **121 / 120 ms**, and peak at
**6.214 / 5.937 mm at 125 / 123 ms**. They are reproducible, pose-dependent
soft-constraint violations during passive limb trapping, not initial intersections
or numerical resets (finite states and zero warnings). Both remain FAIL in the
original report and new report's investigation; the arm-forward PASS does not
replace them. The selected **back diagnostic pose** has no self-contact throughout
its 180 ms record, so the observed wrist/hip failure is not demonstrated there.
Longer settling and an episode-ready supine initial state remain untested; their
future validation must exclude trapping and assess penetration separately.
There is no newly unresolved blocker to Step 3's bounded model/interface scope;
**Step 3 remains COMPLETE**, while A1–A6 and reset acceptance remain Pending.

COM comparison uses the actual `mjINT_EULER` integrator, now explicitly checked:
with zero initial velocity, N=100 and h=0.001 s,
`g*h²*N*(N+1)/2 = -0.0495405 m`. The continuous value `g*(N*h)²/2 = -0.04905 m`
is reported separately and is not used as the discrete acceptance target.

Executed closeout commands (existing symlink-installed package; no rebuild needed):

```bash
source /opt/ros/jazzy/setup.bash
source audit-output/step3-build/install/setup.bash
MUJOCO_GL=osmesa PYTHONPATH=src/x2_recovery/test:$PYTHONPATH .venv/bin/python \
  -m unittest test_model.CloseoutTests -v
# This directory did not exist; no --image-review-from or prior report was used.
MUJOCO_GL=osmesa timeout --kill-after=5s 60s ros2 run x2_recovery runtime_check audit \
  --output audit-output/step3-closeout
```

The fresh run finished every physical check and rendered four images, then correctly
exited **1**, visual review **NOT_TESTED**. After actual inspection, only visual
review/completion was updated with the existing Python helpers and verified hashes;
no physical checks were rerun. `audit-output/step3-closeout/report.json` retains
that initial outcome in `closeout_validation` and the final COMPLETE result.
The original `audit-output/step3/report.json` is preserved. Future regenerations
can use `--image-review-from audit-output/step3-closeout/report.json` with matching
context; context changes require actual image review again.

### Reproduce Step 3 locally

Reuse the existing venv; do not repeat dependency installation for these checks.
All outputs below are under the explicitly ignored `audit-output/` directory.
The original Step 3 build was fresh and used the verified venv interpreter; its ten
regression tests passed, including missing assets, wrong hashes/patch conditions,
incompatible mapping, wrong expected force, repeated loads and stale/failed-report
handling. The closeout reused that symlink install and ran only the four targeted
tests above, plus the one explicitly requested fresh-directory audit.

```bash
source /opt/ros/jazzy/setup.bash
export MUJOCO_GL=osmesa
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
.venv/bin/python /usr/bin/colcon --log-base audit-output/step3-build/log build \
  --base-paths src --packages-select x2_recovery --symlink-install \
  --build-base audit-output/step3-build/build --install-base audit-output/step3-build/install
source audit-output/step3-build/install/setup.bash
.venv/bin/python -m unittest discover -s src/x2_recovery/test -v
timeout --kill-after=5s 30s ros2 run x2_recovery runtime_check model
timeout --kill-after=5s 30s ros2 run x2_recovery runtime_check render \
  --x2-scene "$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml" --output audit-output/step3/render
timeout --kill-after=5s 60s ros2 run x2_recovery runtime_check audit \
  --output audit-output/step3-closeout --image-review-from audit-output/step3-closeout/report.json
```

The last command regenerates `audit-output/step3-closeout/report.json` and
`collision-{back,feet,left_arm,left_shin}.png`. It reuses only prior visual
observations whose image SHA256 **and model/probe context digest** match the new
run; all numerical checks rerun. The context binds the compiled physics/mapping
fingerprint, pinned source identity, MuJoCo version, actual probe definitions and
tolerances, and loader/audit implementation hashes. Identical pixels alone are
insufficient. Legacy observations without a context digest are rejected. The
same read/write report path is safe: observations are read into memory first,
then the report is replaced with an incomplete current-run record; no old numerical
PASS or completion status is imported. Validation of all four reviews finishes
before any individual image is marked reviewed.

On a fresh checkout omit `--image-review-from`: physics and rendering still finish,
but the command returns nonzero with visual review NOT_TESTED. Inspect all four
PNGs, then record `image_review` entries keyed by filename with the actual `sha256`,
the image record's `review_context_sha256` stored as `context_sha256`, and a specific
`observation` of what was seen. Only record these after actual inspection. A normal
rerun verifies them; for review-only completion of an already finished run, the
existing Python `visual_review`/`save_report` helpers can update that report after
checking current model/probe/code and file hashes, without rerunning physics.
Missing/stale observations cannot yield COMPLETE.
No prior numerical PASS or completion verdict is trusted. A failed/incomplete
current run replaces the old report and exits nonzero; NaN/Infinity becomes a
failed measurement in strict JSON. Repeated loads, a different cwd and unchanged
upstream hashes are verified after the probes. PPO and ROS communication were not
rerun because these changes do not affect their implementations.

The controlled joints are both legs (6 each), waist (3), head (2) and both arms
including wrists (7 each). Use the compiled mapping to read actual state and write
control; this is the simulator interface, not an RL action space:

```python
from x2_recovery.model import load_effective_model
import mujoco

loaded = load_effective_model()  # independent MjModel, verified mapping
model = loaded.model
data = mujoco.MjData(model)
row = loaded.joint("head_yaw_joint")
position = data.qpos[row.qpos_address]  # discovered address 36, rad
velocity = data.qvel[row.dof_address]   # discovered address 35, rad/s
data.ctrl[row.ctrl_index] = 0.2         # motor_head_yaw_joint: ctrl 15, joint ID 30
mujoco.mj_forward(model, data)
torque = data.qfrc_actuator[row.dof_address]  # measured +0.2 N m
positions, velocities = loaded.read_state(data)  # copies in ctrl order
```

The audit's nonzero example reads 0.04 rad and -0.016 rad/s, then measures +0.2 N m
from +0.2 input. Robot loading does not maintain a pose or implement a controller.

### Step 4: physical supine reset and bounded actuation

Base: local `main` at `0c627d7ac28da57ff08a5b233b3f3cfdaa4eeb96`, containing
`226ca31f34df6b61ae33c1e6af35834290a97c31` and
`242fa640994b98de681329cf5a73e31d34b90fbc`. Work stays on
`feat/step4-supine-reset`; no automatic merge or remote write.

`reset.py` supplies `ResetSettings`, `construct_pose`, measured predicates and
`reset_supine(loaded, data, seed=..., settings=..., observer=...)`. It uses the existing
loader/mapping and `mj_resetData`; it compiles nothing, writes nothing, imports no
ROS/RL/graphics code, and never caches a successful settled snapshot. Reset-local
counters, dwell reference and RNG are new on every call. The installed API clears
controls, forces, velocities, warm start/history and equality activation defaults;
this model has no activation, history, mocap, equality or plugin state to exercise.
Unexpected control/passive/contact callbacks are rejected. Observation callbacks
receive copied arrays at 25 simulated Hz, with integration-state/physics guards.

Neutral URDF axes are +X front, +Y left and +Z towards the head (verified body
transforms and model views). Rotation of -pi/2 about +Y sends front upwards and the
longitudinal axis horizontally towards -X. All 31 hinges are explicit in
`nominal_pose`, with no unlisted defaults:

| Group | Nominal positions, rad |
| --- | --- |
| Both legs | hip pitch 0.01; hip roll/yaw 0; knee 0.03; ankle pitch -0.37; ankle roll 0 |
| Waist/head | waist pitch 0.30; waist yaw/roll and both head joints 0 |
| Both arms | shoulder pitch 0.42; roll L +0.50 / R -0.50; yaw 0; elbow -0.65 |
| Wrists | yaw 0; pitch -0.22; roll L +0.05 / R -0.05 |

This low-drop pose uses measured passive resting geometry to reduce impact energy;
it is constructed afresh and physically settled on every call. It is not a stored
MjData/qpos reset. Arms are spread away from hip hulls. Optional uniform independent
joint noise is at most 0.005 rad, default OFF, with an explicit local integer seed.
There is one candidate per seed, no resampling, no base kicks or model randomization.

Placement uses the actual fixed floor plane and all eligible collision shapes:
transformed compiled mesh vertices (scaled/recentered by MuJoCo), sphere support and
cylinder support. The minimum linear support equals the convex-hull support.
`mj_geomDistance` independently checks all plane distances with an unsaturated cutoff;
cutoff returns are never called measured distances. Initial clearance is 2 mm with
2 micrometre numerical epsilon. Every eligible robot-robot pair is queried; same-body
and parent/weld filters are reported separately, not newly whitelisted. Nominal
wrist/hip gaps are 285.18 / 286.10 mm. Collision hulls/spherical foot proxies differ
from visible shells; visually detailed hands have no articulated finger DOFs.

Acceptance settings were fixed before the successful 20+20 batch:

| Predicate | Project criterion |
| --- | --- |
| Settling/dwell/extra hold | at most 5 s to qualify; continuous 0.5 s dwell plus 1 s unassisted hold; 45 s wall deadline |
| Torso and pelvis | front within 20 degrees of up; longitudinal axis within 20 degrees of horizontal |
| Linear/angular speed | COM and base <=0.01 m/s; torso and base <=0.05 rad/s |
| Every hinge speed | <=0.02 rad/s, stricter than the initial 0.05 suggestion to bound complete-window drift |
| Complete dwell+hold drift | base translation <=3 mm; base rotation and each hinge <=0.01 rad |
| Settled penetration | ground <=1 mm; eligible self contact <=0.1 mm; joint bounds within 1e-6 rad |
| Real support | total vertical ground force 80–120% of weight; torso carries >=10%; positive gap <=2 micrometres |
| Transient safety | ground <=12 mm, self <=3 mm, joint violation <=0.005 rad, hinge speed <=12 rad/s |

The 1 mm steady and 12 mm transient criteria are project numerical/safety choices,
not PDF/hardware specifications or the old Step 3 15 mm impact screen. The final
state has measured soft-contact compliance, not mathematically zero intersection.
Every step refreshes derived state, contacts and forces, checks finite values,
warning counters, unit quaternion and exact time progression. Force is obtained
with `mj_contactForce`; the contact-frame force on geom2 is rotated to world and
signed for the robot. Near contact alone cannot count as support. No control,
external support, model switching, velocity scaling or per-step pose overwrite is used.
The same qpos **and nonzero qvel** remain in the caller's MjData. Its actual time is
returned as `episode_start_time`; future episode time is `data.time - origin`.

Only model change: `waist_pitch_joint.margin` 0 -> 0.005 rad, in `model.py` for **all**
dynamics. Natural long settling originally loaded the soft upper limit to about
0.31537 rad, beyond the unchanged 0.314 bound. Earlier activation leaves final waist
pitch about 0.31037 rad inside that range. Solref/solimp, masses/inertias, damping,
contact masks, exclusions, gravity, 1 ms Euler, friction and torque limits are unchanged.
Active contacts retain combined `solref=[0.02,1]` and the original impedance; measured
combined values are in the report. Fingerprinting now includes joint margin and
limit solver parameters, closing an identity gap exposed by this change.

New model fingerprint: `bd9bfae8f3a5115cad202f989cf43d7ef0a6678346e4dd3c519de76c300316b0`.
Affected Step 3 checks (including all 62 limit sides and collision probes), other
model/interface checks and four newly generated collision images were revalidated;
the old reports and the two original wrist/hip side-lying FAILs remain intact.

Successful development batch: **20/20 fixed + 20/20 seeds 100–119**, each after
explicit previous-episode motion/control/force/warm-start contamination. Repeats of
100, 107, 119 after different contamination matched requested poses, final q/dq,
metrics and timing exactly here (declared tolerance 1e-10, no cross-platform promise).
Worst total reset time 1.814 s; whole dwell+hold maximum ground penetration 0.385257 mm,
transient maximum 3.932839 mm, largest final penetration 0.196431 mm. No eligible
self penetration, final bound violation, instability warning or time rollback.
Window worst linear speed 0.005469 m/s, angular speed 0.011068 rad/s, hinge speed
0.019953 rad/s; drift 0.135136 mm / 0.009658 rad. Worst final hinge speed is
0.006280 rad/s, retained at handoff. Final ground support is 411.648–411.691 N
(weight 411.691 N), including 113.401–114.470 N on the torso. Final torso face-up
angle <=17.698 degrees, pelvis <=0.105 degrees. All initial clearances were 2 mm.
Per-trial pair/time extrema and sampled traces remain in strict JSON.

**Preserved failures:** `natural-baseline.json` records slow waist/hip drift;
`pose-development*.json` and `low-energy-development.json` retain rejected candidates.
The first real acceptance run, `verification/run-20260920T085519-56c2e7/report.json`,
failed **0/20 fixed and 0/20 seeded passes** on knee impact exceeding 0.005 rad at
about 105 ms. The initial pose was then changed to reduce the fall, with no relaxation
of the acceptance criteria. These are development tests, not five recovery episodes.

`step4.py` performs 31 named joint causal tests. Zero, positive and negative rollouts
start from `mj_copyData` full integration-state copies of the same accepted reset.
Inputs are torque in N m (gain=gear=1), not desired angles. A sin-squared 120 ms pulse
is followed by 120 ms neutral input; effort and q/dq are sampled through the mapping.
Per-joint ceilings are 4 N m hip/knee, 1.5 ankle/shoulder, 2 waist, 1 elbow, 0.7 wrist,
0.6 head, further bounded by half the retained effective effort interval. Start at
half ceiling; escalate once only for insufficient causal motion, never after a safety
stop. Diagnostic guards: 0.2 rad displacement, 4 rad/s hinge speed, 0.1 mrad bound
violation, 12/3 mm ground/self penetration. Inputs return to zero even on errors.

All 31 joints passed at the first amplitude in the accepted supine pose; no fixture
was needed. Maximum sampled causal q differences ranged from 0.00006635 rad (waist
pitch) to 0.01428 rad (hip yaw). Small displacement includes soft-contact/friction
compliance; this is input-to-effort/motion evidence, not useful range or recovery
capability. Four further bounded sequences cover legs, arms/wrists, waist/head and
all joints together. The live/recorded demo uses longer 350 ms head-yaw pulses plus
250 ms neutral intervals for visible motion, then the same small grouped sequences.
Zero torque does not instantaneously stop motion. No standing controller is present.

Commands (existing venv, no dependency installation):

```bash
source /opt/ros/jazzy/setup.bash
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
.venv/bin/python /usr/bin/colcon --log-base audit-output/step4/final-build/log build \
  --base-paths src --packages-select x2_recovery --symlink-install \
  --build-base audit-output/step4/final-build/build --install-base audit-output/step4/final-build/install
source audit-output/step4/final-build/install/setup.bash
MUJOCO_GL=osmesa .venv/bin/python -m unittest discover -s src/x2_recovery/test -v
MUJOCO_GL=osmesa ros2 run x2_recovery runtime_check step4 --output audit-output/step4/verification
MUJOCO_GL=glfw PYGLFW_LIBRARY_VARIANT=x11 LIBGL_ALWAYS_SOFTWARE=1 \
  ros2 run x2_recovery runtime_check step4-live --repeat 2 --output audit-output/step4/live
MUJOCO_GL=osmesa ros2 run x2_recovery runtime_check step4-record \
  --repeat 2 --output audit-output/step4/recording
```

`--repeat 20` watches twenty actual resets; it does not replace headless acceptance.
Backend variables must be set before import, in separate processes. The live window
uses supported `mjv_updateScene`/`mjr_render` in a GLFW desktop window with one main
physics loop. C toggles collision hulls; successive resets begin in normal/collision
views. Mouse forces and GUI control editing are absent. Camera/graphics do not change
physics. On-screen phase/reset/seed/time/episode time/torque and measured head q/dq
are synchronized. Native window readback is saved before swapping that same buffer;
black images fail explicitly. The observer checks integration and model state.

Rendering targets 25 FPS without altering the 1 ms physics timestep. Software
rendering is slower than real time; measured rate is reported, not silently fixed
by changing physics. Shadows/reflections are off only in the live display. The
OSMesa GIF is labeled **PHYSICS REPLAY**, rendered on separate visualization data
from copied states of a single continuous reset-to-actuation run, including initial
placement. It never feeds state back into physics. No installed H.264 encoder was
available, so the existing Pillow animated format is used. Reports start incomplete;
media generation and process exit never automatically grant visual approval.

Reproducible bug notes and sources:

- Soft waist limit drift: checked [matching limit-margin documentation](https://github.com/google-deepmind/mujoco/blob/3.13.0/doc/XMLreference.rst)
  and [soft-constraint model](https://mujoco.readthedocs.io/en/stable/modeling.html).
  Applied only the conservative early-activation margin; old failures retained.
- Reset state/geometry: verified installed 3.13.0 headers/API and matching
  [reset implementation](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_io.c)
  and [signed-distance implementation](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_support.c).
  A versioned rendered documentation page was unavailable, so matching source was used.
- Viewer: an X11 passive smoke test exited cleanly after explicitly joining its
  background renderer, but this alone did not prove images. Actual X2 framebuffer
  readback was black on virgl (Apple M5 Pro Compat), even without shadows/meshes in
  the minimal reproduction. An existing llvmpipe process renders the same scene.
  The final compact viewer uses main-thread native rendering and clean context
  destruction. No driver/dependency/global-display changes. Sources:
  [MuJoCo visualization API](https://mujoco.readthedocs.io/en/stable/programming/visualization.html),
  [pyGLFW backend selection](https://github.com/FlorianRhiem/pyGLFW),
  [Mesa process environment](https://docs.mesa3d.org/envvars.html).
  `gl-clear-probe.log`, `mujoco-gl-probe.log`, `x2-gl-probe.log` and native black frames
  preserve reproduction evidence; the first high-resolution software run hit its
  external 40 s timeout. Optimized software rendering completed the full sequence.

A2–A6 remain Pending. X2 training is **Not run**, evaluation **Not evaluated**.
Next work is environment observation/action/reward design; none is implemented here.

### Development history

The repository was established before implementation. Preserve meaningful commits and
record completed work in local commits (PDF p. 2, Commit history). The current
user instruction overrides the PDF push request: **no remote writes without explicit
approval of the specific push/PR operation**. Step 3 is finished locally only.

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
  v1.3.0 effective-model results and remaining scope limitations are recorded above; v1.4.0 was not loaded.
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
