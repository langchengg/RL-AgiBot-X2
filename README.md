# AgiBot X2 Ground Recovery

## Overview

A native MuJoCo and Gymnasium implementation for AgiBot X2 ground recovery, developed
for the HRS take-home task. The floating-base model, resting supine reset, bounded
joint control, independent success detector and environment have been validated in
an Ubuntu ARM64 Parallels VM. No policy has yet been shown to recover from supine to
standing. Real-X2 PPO sampling, optimization and checkpoint reload now pass a bounded
engineering smoke test. A larger formal experiment is deferred because training
episodes consistently abort within one simulated second. Five policy evaluation
episodes remain NOT_EVALUATED; ROS integration is validated with the scripted baseline.

| Component | Status |
| --- | --- |
| MuJoCo model and supine reset | Validated within documented numerical tolerances |
| Joint control and standing-success detector | Validated |
| Gymnasium environment | Validated |
| PPO recovery training | 2048-transition real-X2 smoke passed; formal experiment deferred on sampling quality |
| Five-episode recovery evaluation | Not evaluated |
| ROS recovery and telemetry nodes | Validated with real X2 simulation and scripted_baseline |
| ROS end-to-end integration validation | Passed; scripted baseline did not recover to standing |

## System Architecture

One `ament_python` package contains the native simulator environment and validation
utilities. ROS recovery control and PPO training use the same environment; evaluation
will use this implementation as well. No simulator bridge, ONNX conversion, ros2_control, Gazebo or alternative
simulation framework is required.

| Module | Responsibility |
| --- | --- |
| [model.py](src/x2_recovery/x2_recovery/model.py) | Offline pinned-asset verification, effective model construction and joint/control mapping |
| [reset.py](src/x2_recovery/x2_recovery/reset.py) | Geometric placement, physical supine settling and checked stepping |
| [success.py](src/x2_recovery/x2_recovery/success.py) | Read-only standing measurements and continuous, reward-independent success tracking |
| [env.py](src/x2_recovery/x2_recovery/env.py) | Gymnasium reset/step, observations, bounded PD, rewards, lifecycle and lazy rendering |
| [train.py](src/x2_recovery/x2_recovery/train.py) | Bounded PPO training, sampling diagnostics, optimization evidence, checkpoint reload and reward plots |
| [model_audit.py](src/x2_recovery/x2_recovery/model_audit.py) | Disposable model/interface probes and evidence validation |
| [simulation_validation.py](src/x2_recovery/x2_recovery/simulation_validation.py) | Reset batches, causal actuation probes, live observation and actual-state recording |
| [success_validation.py](src/x2_recovery/x2_recovery/success_validation.py) | Independent standing calibration fixtures and physical positive/negative validation |
| [diagnostics.py](src/x2_recovery/x2_recovery/diagnostics.py) | `runtime_check` command dispatch, environment checks and ROS dependency probes |

The environment owns model/data, RNG, tracker, previous action and rendering resources
per instance. Core measurements and tracking do not perform audit I/O or import ROS,
training or recording components. Standing fixtures are validation tools, not recovery
policies or normal environment initial states.

## Requirements

Target: Ubuntu 24.04 ARM64 in Parallels, ROS 2 Jazzy and native MuJoCo Python.
These are project choices, not additional constraints imposed by the task brief.

Runtime observed on 2026-09-20 inside the existing Parallels VM:

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

## Setup

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
VALIDATION_BUILD_DIR="$(mktemp -d /tmp/hrs-x2-validation.XXXXXX)"
.venv/bin/python /usr/bin/colcon --log-base "$VALIDATION_BUILD_DIR/final-log" build \
  --base-paths src --packages-select x2_recovery --symlink-install \
  --build-base "$VALIDATION_BUILD_DIR/final-build" --install-base "$VALIDATION_BUILD_DIR/final-install"
source "$VALIDATION_BUILD_DIR/final-install/setup.bash"
head -1 "$VALIDATION_BUILD_DIR/final-install/x2_recovery/lib/x2_recovery/runtime_check"
```

PyPI metadata and official wheel listings were checked before installation: Gymnasium
and SB3 have universal wheels; Torch has a cp312 Linux aarch64 CPU wheel. SB3 2.9.0
accepts Torch >=2.8 and Gymnasium <2.0; these pins retain working MuJoCo/NumPy and avoid
CUDA packages or source builds. requirements.txt pins direct dependencies and the cffi
compatibility requirement from inherited PyNaCl; it is not a transitive lockfile.
No apt/build-tool changes were needed. Build outputs and raw logs stay in the temporary
directory; colcon discovers only src, and the venv has COLCON_IGNORE.

## Robot Model and Simulation

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

The upstream Mulan PSL v2 license stays with the assets. No project license is granted; ROS package metadata uses UNLICENSED.

### Model construction and modifications

`x2_recovery.model.load_effective_model(asset_repo=None)` is the only normal X2
loading path, including existing runtime/render diagnostics. It accepts a repository
path or `X2_ASSET_REPO`, otherwise the cache path under `Path.home()`. It checks the
pinned Git revision, relevant dirty/untracked files and XML/license hashes. It uses
MuJoCo 3.13.0 `MjSpec.from_file`, named edits and recompilation, preserving relative
includes/meshes without changing cwd. Every call returns a fresh model; no ROS,
Torch, SB3, rendering, probes, reports or network access are imported/performed by
this module. The audit alone compiles an explicitly labeled unmodified baseline.
Incompatible assets, patch preconditions and mappings raise explicit errors.

The pelvis and six joint-range overrides are applied, with old/new values and reasons in the report:

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

An additional in-memory override sets `waist_pitch_joint.margin` 0 -> 0.005 rad, in `model.py` for **all**
dynamics. Natural long settling originally loaded the soft upper limit to about
0.31537 rad, beyond the unchanged 0.314 bound. Earlier activation leaves final waist
pitch about 0.31037 rad inside that range. Solref/solimp, masses/inertias, damping,
contact masks, exclusions, gravity, 1 ms Euler, friction and torque limits are unchanged.
Active contacts retain combined `solref=[0.02,1]` and the original impedance; measured
combined values are in the report. Fingerprinting includes joint margin and
limit solver parameters, closing an identity gap exposed by this change.

Effective-model fingerprint: `bd9bfae8f3a5115cad202f989cf43d7ef0a6678346e4dd3c519de76c300316b0`.
This fingerprint binds the effective physical model and frozen success calibration.

Simulation settings are timestep 0.001 s, gravity `[0,0,-9.81]` m/s², Euler integrator,
Newton solver, pyramidal cone, tolerance 1e-8, 100 iterations, all global disable
flags zero. The strict JSON report includes complete settings, inertias, geoms,
limits, actuator/mapping tables, source and project identities/hashes, definitions,
measurements, warning/time checks and evidence scopes. Quaternions are unit `wxyz`;
free-base linear velocity is world-frame and angular velocity is body-local.
The installed header/API behavior and
[MuJoCo 3.13.0 source](https://github.com/google-deepmind/mujoco/tree/3.13.0)
were checked, including sparse moments, clamp order and quaternion integration.

### Collision, joint and actuator limits

The source comparison identifies **12 joint ranges**, comprising six narrower retained
intervals and six wider intersected intervals; **zero rounding-only differences**
at the declared 1e-4 rad classification threshold. **13 effort-envelope differences**:
eight hip/knee motors plus waist yaw retain ±118 N m versus URDF ±120; four wrist
pitch/roll motors retain ±2.2 versus ±4.8. Other bounds agree. All 31 motors have
unit gain/gear, no bias or activation state, enabled input limits and enabled joint
actuation limits. Input clipping dominates the equal/wider downstream joint clamp;
that downstream clamp was **not independently exercised**. Absence of an actuator
`forcerange` does not mean unlimited joint effort. URDF speed limits are reported
but **not enforced** by this route; no hidden speed clamp was added.

Mesh contact uses convex hulls. Each foot uses twelve 5 mm collision spheres, not
its detailed visual mesh; the floor is an infinite plane. Hands/fingers are not
articulated. Passive wrist/hip trapping failures and transient penetration measurements
are retained under [Validation and Reproducibility](#validation-and-reproducibility).

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

## Recovery Environment

`env.py` implements `X2RecoveryEnv` with independent model/data, RNG, success tracker, controller history
and lazy rendering resources. It does not import ROS or the audit/standing fixture.
It uses the effective X2 model, 31-joint mapping and frozen success calibration.
The ROS nodes below reuse this implementation. PPO has produced an updated checkpoint, but no successful learned recovery has been observed.

```python
import numpy as np
from x2_recovery.env import X2RecoveryEnv

with X2RecoveryEnv() as env:
    observation, info = env.reset(seed=60)
    observation, reward, terminated, truncated, info = env.step(np.zeros(31, np.float32))
    # Inspect terminated/truncated; explicitly reset before another episode.
```

### Reset

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

Reset criteria (validated with 20 fixed and 20 seeded trials):

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
not PDF/hardware specifications or the 15 mm passive-impact screening cutoff. The final
state has measured soft-contact compliance, not mathematically zero intersection.
Every step refreshes derived state, contacts and forces, checks finite values,
warning counters, unit quaternion and exact time progression. Force is obtained
with `mj_contactForce`; the contact-frame force on geom2 is rotated to world and
signed for the robot. Near contact alone cannot count as support. No control,
external support, model switching, velocity scaling or per-step pose overwrite is used.
The same qpos **and nonzero qvel** remain in the caller's MjData. Its actual time is
returned as `episode_start_time`; future episode time is `data.time - origin`.

Minimal caller (after loading once with `load_effective_model()`):

```python
import mujoco
from x2_recovery.reset import reset_supine

data = mujoco.MjData(loaded.model)
metadata = reset_supine(loaded, data, seed=100)
episode_start_time = metadata["episode_start_time"]
# Continue stepping this same data; do not zero qvel or reset data.time.
```

Reset calls Gymnasium's seed initialization and draws an actual reset seed from
`self.np_random`. `seed=None` continues that RNG sequence. The existing supine reset
hands over the **same MjData, qpos, residual qvel and nonzero time**; that instant starts
episode timing and provenance, excluding settling. Default perturbation is zero;
optional `reset_perturb_rad=0.005` uses the previously validated reset range. Reset
failure raises without retry. Only absent/empty reset options are supported. Previous
action is initialized to zero as a “no policy action yet” convention; no standing
initialization is exposed.

### Observation space

Every return is an independent, finite float32 `(117,)` array in
`Box(-inf,inf)`. q scaling uses the effective mechanical midpoint/half-range; physical
limit excess is observable, not clipped away. R is pelvis body-to-world rotation.

| Slice | Meaning / scaling |
| --- | --- |
| 0:31 | `(q-midpoint)/half_range`, actuator order |
| 31:62 | joint velocity / 5 rad/s |
| 62:65 | unit gravity in pelvis frame, `R.T @ [0,0,-1]` |
| 65:68 | pelvis **origin** linear velocity in pelvis frame / 1 m/s |
| 68:71 | pelvis angular velocity in pelvis frame / 2 rad/s |
| 71:72 | pelvis floor-relative height / frozen 0.6724955472220092 m |
| 72:74 | left/right world vertical foot force / current robot weight |
| 74:75 | sum of individual non-foot ground force magnitudes / weight |
| 75:106 | previous actually adopted policy action; current action after step |
| 106:107 | continuous stable duration / 2.0 s |
| 107:108 | tracker window valid flag |
| 108:117 | pelvis/left/right displacement from tracker window start, three vectors |

For the last nine channels, world horizontal displacement `[dx,dy,0]` is rotated by
`R.T`, then divided by 0.05 m for pelvis or 0.03 m for each fixed foot-body origin.
Without a valid window, flag, timer and all displacements are zero. References come
from the existing tracker, with no second window algorithm. MuJoCo free-joint qvel
translation is world-frame origin velocity, while rotation is already body-local;
the latter is not rotated twice. A known-yaw test compares the origin velocity with
`mj_objectVelocity`'s body COM velocity and its angular lever-arm correction. Contact
channels reuse the independent success detector’s actual constraint forces. Returned info also owns its data.
These observations use simulation-accessible states/contact forces; hardware
availability and complete observability have not been established.

### Action space

`Box(-1,1,(31,),float32)` follows `loaded.mapping`, without a
second joint ordering. Zero means the fixed q_ref target, not zero torque or holding
the current state. Reference angles match the independently calibrated near-straight standing pose:
both hip pitch -0.05, knees +0.10, ankle pitch -0.05, shoulder pitch -0.15, elbows
-0.30 rad; shoulder roll +0.15 left / -0.15 right; all others zero. For nonnegative
`a`, target is `q_ref + a*(q_max-q_ref)`; for negative `a`, it is
`q_ref + a*(q_ref-q_min)`. Thus the target range covers the **whole effective mechanical
range**, including asymmetric sides. Examples: knees [0,2.4073], hip pitch
[-2.704,2.556], elbows [-2.3556,0] rad. Tests also cover reset targets and a candidate
with knee 1.8, hip pitch -1.2 and elbow -1.4 rad; this is target coverage, not a proof
of a feasible recovery trajectory. Wrong shape, nonfinite or out-of-range actions
are rejected before physics; no out-of-range tolerance is applied.

### Controller

| PD group | Kp (N m/rad) | Kd (N m s/rad) |
| --- | ---: | ---: |
| Hip, knee | 600 | 16.97056275 |
| Ankle, waist | 300 | 11.31370850 |
| Shoulder, elbow | 100 | 4.24264069 |
| Head, wrist | 10 | 0.42426407 |

`tau_raw=Kp*(target-q)-Kd*dq`; tau is clipped to the verified effective effort interval
and written through each ctrl address. These direct motors have unit gear/gain;
measured joint actuator torque equals clipped ctrl. Limits remain the loader's
0.6–118 N m magnitudes, depending on joint, not the small diagnostic pulse amplitudes. The gains
are initial candidates retained after bounded supine tests, not optimized recovery
parameters. Saturation reports distinguish saturated joint/substep pairs divided by
`31*executed_substeps`, per-joint time fractions, and time with any saturation. Target
boundary fraction uses joint/policy-action pairs; raw and measured torques are separate.

`EnvConfig` is the single editable configuration source. `env.resolved_config()` and
an `env-check` report expose all 31 names, qpos/dof/ctrl addresses, target ranges,
reference angles, gains, effort limits and success thresholds in actuator order.
Defaults are **0.001 s physics**, **20 substeps**, **0.020 s control / 50 Hz**, and
**20.0 s external timeout**. Timeout must be an integer multiple of the physical dt;
the environment never changes dt. Each substep recomputes PD from current q/dq, runs
`checked_step` (including state synchronization), measures contacts and updates the
original tracker. It stops at the first successful, safety-aborted or timed-out
substep, with success taking priority at the exact deadline. There is no hidden step,
state clipping, automatic reset, action smoothing or target integration.

### Episode termination and execution errors

Successful recovery terminates with `is_success=True`; timeout truncates with
`time_limit`; normal lying, low height, tilted torso and non-foot contact do not end
an attempt. Separate task safety guards terminate on floor penetration >30 mm,
self penetration >15 mm, joint-limit excess >0.05 rad or joint speed >30 rad/s.
These are engineering guardrails for gross deformation/high-rate stress, **not**
hardware speed ratings, proof of irrecoverability or relaxed standing criteria.
They retain contact-rich motion (observed up to 11.90 mm floor penetration and
0.0320 rad soft limit excess) but stop the tested full-target impulses and severe
self intersection. Their suitability for useful learned recovery remains unverified.

Nonfinite state/action, numerical warnings, illegal forces, changed model/mapping,
unexpected state edits, missed samples/time reversal or watchdog expiry invalidate
the tracker, require reset and raise `EnvExecutionError` with detached diagnostic
evidence. NaN/Inf are explicitly encoded in that evidence, never repaired as zeros.
Model identity is checked at policy/reset/render boundaries; lightweight state,
forces/time/standing checks run every millisecond. Wall budgets are 45 s per reset
and 5 s per step call; waiting between calls or doing network updates consumes no
simulation timeout. The environment is the sole owner of normal rollout state;
these checks are not a general tamper-proof execution certificate.

### Rendering and resources

`render_mode=None` creates no graphics resources or sleeps. `rgb_array` renders the
current state; `human` uses native mjv/mjr + GLFW mechanism without mutable
viewer panels or mouse perturbation. Rendering cannot advance physics or write controls.
Select OSMesa versus GLFW **before process start**; close is idempotent. GIF labels
identify actual env rollout states, not a standing initialization or recovered policy.

### Scripted baseline

`baseline.py` runs one real supine recovery attempt without ROS or training libraries.
It copies the settled reset joint positions once, constructs named targets in the
verified actuator order, and uses quintic smoothstep interpolation. The stages are
handover (0–0.4 s), tuck (0.4–3.4 s), brace (3.4–5.4 s), shift (5.4–8.4 s), and extend
(8.4–12.4 s), followed by indefinite final-target holding. Names describe intent;
support and standing come from actual measurements. Unspecified joints, including
head and wrists, inherit the initial/preceding target. Candidate angles, durations,
any target clamps and initial reference projections are exported in the summary.
The measured run needed neither clipping nor projection; peak reference speed was
0.70046 rad/s, which is not a physical speed limit or a hardware specification.

Only `env.action_for_targets()` → `env.step()` controls the episode. The environment
retains bounded PD, per-physics-step success/safety checks and its simulation timeout.
The sequence uses relative simulation time, excluding reset settling. Changing the
timeout does not rescale the sequence, and finishing it does not imply recovery.
Physics, reset, gains, action mapping and success/safety thresholds are unchanged.

From the repository root, with the existing venv and cached model:

```bash
PYTHONPATH="$PWD/src/x2_recovery${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -m x2_recovery.baseline \
  --seed 60 --timeout-s 20 \
  --output-dir "audit-output/scripted-baseline/$(date -u +%Y%m%dT%H%M%S)-$$"
```

Default is headless and needs no DISPLAY or network. Omitting `--output-dir` creates
a unique directory; explicitly supplied existing directories are rejected. Each run
saves strict `trajectory.jsonl` (real reset, then actual post-step states with action
intervals, contacts, last-substep torques and environment info) and `summary.json`
(outcome, reset evidence, resolved configuration, joint order, model/Git identity and
scoped statistics). Exit 0 means a valid episode ended, including recovery failure;
check `recovery_success`. Errors, cancellation or evidence-write failures exit nonzero,
with `execution_completed=false` and `recovery_success=null`; partial evidence is
retained when writable. Optional `--human` uses the existing
[GLFW settings](#environment-integration-rewards-and-performance).

Development validation in the Ubuntu ARM64 VM on 2026-09-20, code `ac7867d`, clean at
run time (these are not the formal five evaluation episodes):

| Run under `audit-output/scripted-baseline/` | Actual simulation / control / physics steps | Result |
| --- | --- | --- |
| `main-20260920` | 20.000 s / 1,000 / 20,000 | completed; recovery false; `time_limit` |
| `timeout-20260920` | 2.003 s / 101 / 2,003 | completed; recovery false; `time_limit`; final call 3 substeps |
| `repeat-20260920` | 20.000 s / 1,000 / 20,000 | same seed/configuration; all trajectory fields identical |

Seed 60 drew reset seed 374032080; legal settling ended at absolute time 1.661 s.
**Measured:** joint excursion reached 1.42961 rad, applied torque peaked at 48 N m,
control-sampled pelvis height peaked at 0.08515 m, and sampled standing dwell stayed
zero. At timeout, pelvis height was 0.07636 m and torso tilt 73.5896°; left/right
vertical foot forces were 0.17866/0.17799 body weights, while non-foot force norms
summed to 0.64335 body weights. Elbow collision contacts really carried load during
brace (about 80 N each at its endpoint); final support still included pelvis/torso.
No numerical warnings or safety termination occurred. Substep peak floor penetration
was 0.5301 mm, self penetration and joint-range excess were zero; 1.3724% of
joint/substep pairs saturated. **Inference:** this open-loop target sequence changes
limb posture but does not transfer support into upright standing. **Next experiment:**
use these contact and torso measurements to revise the support-transfer targets;
these data do not establish a geometric impossibility or reliable recovery.

The repeat comparison fixed absolute tolerance 1e-10 and relative tolerance zero
before rerunning; 382,095 floating values across full trajectories and selected
physical/configuration summary fields had maximum difference zero. Paths, file times
and Git metadata were excluded. All 14 baseline tests and the full 87-test suite
passed:

```bash
PYTHONPATH="$PWD/src/x2_recovery" MUJOCO_GL=osmesa \
  .venv/bin/python -m unittest discover -s src/x2_recovery/test -v
```

A fresh isolated venv/colcon build using
[Setup](#setup) passed; the installed module ran from outside the repository with
`--timeout-s 2.003`, matching the source trajectory. No console-script addition is
needed. Logs and comparison details are under `audit-output/scripted-baseline/verification/`.
Six actual RGB samples from a separate `visual-20260920` episode were inspected;
its trajectory matched the main run. This was OSMesa observation, not live desktop
viewer validation. This baseline evidence is separate from the PPO training and ROS
integration results documented below.

## Reward Design

Let `H=clip(height/h_ref,0,1)` and
`U=(1+clip(torso_upright_dot,-1,1))/2`. Gamma is **0.999 per env.step transition**,
including a shortened final step; PPO explicitly checks and uses the same gamma.

| Contribution | Actual weight | Raw term |
| --- | ---: | --- |
| Height shaping | 3.0 | `gamma*H(next)-H(previous)` |
| Upright shaping | 1.0 | `gamma*U(next)-U(previous)` |
| Qualified standing time | 1.0 | sum of qualifying substep dt, seconds |
| Torque-square cost | -0.02 | sum of mean `(actual_torque/effort_magnitude)^2 * dt`, seconds |
| Action-change cost | -0.002 | mean `(adopted_action-previous_action)^2`, once per transition |
| Recovery completion | 50.0 | 1 only on the first true `recovery_success` transition |

Height/upright shaping supplies progress feedback, qualified hold encourages the
actual target, torque/action costs discourage excessive actuation and abrupt changes,
and the completion bonus favors finishing. Initial and final weights are identical;
bounded diagnostics support the wiring, not optimality. Hold uses right-endpoint
integration of **all** instantaneous and drift checks, with no invalid execution;
it does not replace or give an extra timestep to the tracker's continuous timer.
Torque square is not physical energy. `reward_terms_raw` and weighted `reward_terms`
reconstruct the scalar reward; the total is never clipped nonnegative.

True `terminated` transitions use zero absorbing-state next potential in shaping,
while keeping the **real observation**. External timeout uses `truncated=True`, the
real next potential and terminal observation for SB3 bootstrap. This is an external
sampling limit, not a finite-horizon task requiring a remaining-time observation.
`Monitor`/`DummyVecEnv` preserve `terminal_observation` and `TimeLimit.truncated`;
the base environment still returns Gymnasium's five values, without another TimeLimit.

## Success Criteria

`success.py` has no reward, ROS, training, rendering, network or file-output dependency.
`measure_standing` reads synchronized MuJoCo state; `standing_failures` combines
necessary predicates; `SuccessTracker` handles continuity, drift, provenance and timeout.
`success_validation.py` is only a bounded standing calibration/test fixture and evidence command.

Recovery success requires an actual successful `reset_supine` handoff, followed by
bounded joint actuation and uninterrupted normal physics without external support,
teleport, unintended reset or model changes, reaching and holding all standing checks
for **2.0 simulated seconds** by the configured deadline (tracker default **20 s**).
`instant_standing_ok` checks the current sample; `standing_held` additionally requires
continuous duration and drift; `recovery_success` additionally requires a valid origin
and execution before timeout. Directly initialized standing can only satisfy the first
two. `reset_from_supine(context, data, reset_result)` checks the actual returned q/dq,
time and supine state, retains residual velocities, and starts at handoff, excluding
reset settling. The caller owns stepping, model immutability and the prohibition on
teleportation; no boolean/hash can prove unseen execution history. It must invalidate
the attempt on any stepping exception or execution violation.

| Frozen criterion | Value and unit |
| --- | --- |
| Pelvis body-origin height / reference | >=0.90; **h_ref=0.6724955472220092 m** |
| Torso local +Z versus world up | <=15 degrees, signed dot (inversion fails); yaw unrestricted |
| Each foot world vertical force / W | >=0.05 |
| Sum of both foot vertical forces / W | 0.80–1.20 |
| Sum of individual non-foot ground force magnitudes / W | <=0.00001 (about **0.0041169 N**) |
| Base / whole robot COM 3D linear speed | each <=0.10 m/s |
| Base / torso angular speed | each <=0.25 rad/s |
| Maximum absolute controlled-joint speed | <=0.50 rad/s |
| Maximum horizontal displacement from window start | pelvis <=0.05 m; each fixed foot-body origin <=0.03 m |
| Steady floor / self penetration; joint-limit excess | <=1 mm / 0.1 mm; <=0.1 mrad |
| Load-bearing contact positive gap | <=2 micrometres |
| Continuous hold | >=2.0 s, every 0.001 s physics sample |

`W` is computed from the actual pelvis dynamic subtree and gravity: **411.69110013 N**.
The verified foot groups are exactly the twelve 5 mm collision spheres on each
`{left,right}_ankle_roll_link`; visual meshes and other leg geoms do not qualify.
`mj_contactForce` is transformed by `contact.frame.T`, with the geom1/geom2 sign applied.
Inactive/zero-force contacts do not support; every nonzero non-foot force is accumulated
without a small-contact filter or vector cancellation. Self contacts are excluded from
ground support. All proxy margins/gaps are zero. Contact forces agree with independent
`mj_rnePostConstraint` force aggregation to <=5.7e-14 N. COM velocity uses the robot
subtree; body/angular velocities were checked against actual X2 Jacobians and mapping.

Every `checked_step` is followed by a measurement/update. The first qualifying sample
starts at duration zero. Failed instantaneous conditions or excessive drift clear the
window; failed rising poses are not invalid episodes. Repeated identical samples add
no time; changed duplicate samples, time rollback, missing substeps and nonfinite data
invalidate the attempt. Time comparisons allow only 1e-10 s floating error. At the
exact deadline a completed hold wins; incomplete holds time out and cannot later succeed.
External forces, unbounded control, numerical warnings and incompatible models raise
explicit errors. Static model validation is cached, not repeated at every millisecond.

Calibration used independently specified nearly straight legs (hip -0.05, knee +0.10,
ankle pitch -0.05 rad), neutral waist/head, shoulders pitch -0.15 and roll +/-0.15,
elbows -0.30, other joints zero. Exact all-joint targets and gains are saved in the report.
PD Kp/Kd: hip/knee 600/16.97056, ankle/waist 300/11.31371, shoulders/elbows
100/4.24264, head/wrists 10/0.424264 (N m/rad, N m s/rad), updated every 1 ms,
clipped to actual effective actuator limits. Initial placement uses actual hull/proxy
geometry with 2 mm clearance; only initialization writes q/dq. No base restraint,
gravity compensation, control feedback beyond joint PD, or physics modification is used.
The predeclared search bound was six candidates, 8 s simulation / 45 s wall each;
candidate 0 fell, candidate 1 stood, and search stopped. Both traces remain in
the [calibration trial records](audit-output/step5/development-103134/). The 6–8 s stable segment, independently checked
for upright posture, foot-only support and low motion, had pelvis height
0.67249380–0.67249581 m and tilt 4.86944–4.87325 degrees. Its median became h_ref,
distinct from whole-body COM height. Zero non-foot contacts/force supported tightening
0.005W to 0.00001W. This is a numerical allowance, not proof of exactly zero support.
Reference and thresholds are committed as `CALIBRATED_SETTINGS`; model/code/config
changes invalidate saved acceptance, and physics changes require recalibration.

## Training

`train_recovery` (or `python -m x2_recovery.train`) runs a single real X2 environment
through Monitor, DummyVecEnv and VecCheckNan. It uses SB3 PPO 2.9.0 on CPU with fixed
environment observation scaling. No VecNormalize, extra reward/observation clipping,
rendering, ROS service stepping or scripted actions participate in training.

The initial PPO configuration uses 512 transitions per rollout, batch size 64,
5 epochs, learning rate 3e-4, gamma **0.999** (checked against shaping gamma),
GAE lambda 0.95, policy clip 0.2, no value clipping, normalized advantages,
entropy coefficient 0, value coefficient 0.5, gradient norm 0.5 and target KL 0.03.
Separate actor/value MLPs have two 128-unit Tanh layers with orthogonal initialization;
Adam uses epsilon 1e-5 and betas (0.9, 0.999). Exploration starts at log std -1.0;
the one measured comparison uses -1.5. Torch uses 2 compute threads and 1 interop
thread on the existing 8-vCPU, approximately 11-GiB ARM64 Parallels VM.
The complete effective settings, including optimizer defaults, are exported per run.

Commands executed in this VM (existing run directories are deliberately rejected):

```bash
export PYTHONPATH="$PWD/src/x2_recovery"
.venv/bin/python -m x2_recovery.train probe --policy zero --seed 220922 \
  --episodes 2 --max-transitions 128 --max-wall-seconds 180 \
  --run-dir runs/ppo_supine/probe-zero-20260922
.venv/bin/python -m x2_recovery.train probe --policy untrained --seed 220922 \
  --episodes 20 --max-transitions 128 --max-wall-seconds 180 \
  --run-dir runs/ppo_supine/probe-std-minus1-20260922
.venv/bin/python -m x2_recovery.train probe --policy untrained --seed 220922 \
  --log-std-init -1.5 --episodes 20 --max-transitions 256 --max-wall-seconds 180 \
  --run-dir runs/ppo_supine/probe-std-minus1p5-20260922
.venv/bin/python -m x2_recovery.train smoke --seed 220923 --log-std-init -1.5 \
  --total-timesteps 2048 --max-wall-seconds 1200 \
  --run-dir runs/ppo_supine/smoke-std-minus1p5-20260922
```

Initial diagnostics are separate from training. At log std -1.0, 17/20 episodes
ended in safety abort within two transitions; all 20 ended on joint speed.
At -1.5, that fraction fell to 7/20, with median length 3 and maximum length 15
(0.290 simulated seconds); 16 ended on joint speed and 4 on joint limits.
Terminal observations identify left/right hip yaw as the speed offenders.
Zero actions survived a 128-transition (2.56-second) diagnostic cutoff without
standing. Zero is a PD reference target, not zero torque. That incomplete trajectory
is excluded from episode outcome statistics. Reduced exploration improves length
but these short trajectories remain a substantial learning limitation.

The selected smoke run `smoke-std-minus1p5-20260922` completed on 2026-09-22:

| Measurement | Observed result |
| --- | --- |
| Seed / requested / sampled / optimized rollout transitions | 220923 / 2048 / 2048 / 2048 |
| Complete rollouts / optimization rounds / actual Adam steps | 4 / 4 / 30 (increments 3, 8, 9, 10; KL early stopping) |
| SB3 epoch-attempt counter | 7; distinct from the 30 optimizer steps |
| Actor mean / critic / log-std parameter L2 change | 0.246114 / 0.645779 / 0.012161 |
| `learn()` / end-to-end throughput | 352.862 s / 5.804 transitions/s |
| Full rollout cycle throughput range | 5.358–6.296 transitions/s |
| Reset wall time / calls | 315.273 s / 424; 89.35% of `learn()` |
| Total run wall time, including reload | 374.842 s |
| Recovery physical steps / simulated time | 37,616 / 37.616 s; excludes reset settling |
| Complete episodes / median length / maximum length | 423 / 4 / 32 transitions |
| Safety endings | 363 joint-speed; 60 joint-limit; no training timeouts or successes |
| Unfinished data | No partial rollout; a final 2-transition partial episode is retained separately |
| Independent reload | PID 197760; 16 real observations, exact deterministic action agreement; full 1000-transition, 20-second timeout; no standing success |

All seven rollout arrays, 390 backward gradient tensors, every optimizer input,
updated parameter and Adam-state tensor passed finite checks. Recorded losses are
update-level SB3 metrics, not a claim to have captured every intermediate loss.
Monitor and diagnostics agree on episode order/count; maximum return discrepancy
is 5.25e-7. The final update is present in `progress.csv` and the SB3 log.

The observed maximum control-sampled pelvis height, 0.21975 m, occurred at 92.34°
tilt, with no foot support; maximum stable duration was zero. Mean episode return
was -0.99466. Longer episodes accumulated more cost (length/return correlation
-0.998); return alone would be misleading here. Joint/substep torque saturation
was 31.10%, despite no clipped-action or target-boundary samples at log std -1.5.
These measurements do not establish sitting up, rolling over or recovering.

**Formal experiment: NOT_RUN.** All 423 complete episodes ended in safety abort
within 0.633 seconds (median 0.071 s). Passing the narrow two-transition gate does
not make this sampling suitable for more compute. The CLI also rejects expansion
when all 20 recent episodes are subsecond safety aborts; this is a conservative
resource guard, not proof that the task cannot be learned. A candidate 600-second
budget at the measured rate gives `512 * floor(0.8 * 600 * 5.803974 / 512) = 2560`
planned transitions; zero formal transitions were executed. The actual guard
invocation exited 1 before creating a run directory. Details are retained in
`audit-output/ppo-validation/budget-decision.json` and the smoke run's
`sampling_analysis.json`. No physics, action mapping, PD, reset, rewards, safety
thresholds, success logic, baseline or ROS implementation changed.

Checkpoint: `runs/ppo_supine/smoke-std-minus1p5-20260922/policy_final.zip`, SHA-256
`e8fa7e642e8ccd8d6dd38aaf9d810eee3c00d39d4188f82702c5b8e6124b213e`.

Each local run under `runs/ppo_supine/` retains its source snapshot/diff, exact
configuration, manifest, raw transition/episode/Monitor logs, completed rollout
arrays, update-level `progress.csv`, reward PNG and console log. Optimized runs also
save `policy_final.zip`, a real-observation `reload_probe.npz`, and a new-process
`reload_check.json` with a full deterministic recovery attempt. Checkpoint saving
requires actual actor-mean and critic parameter changes and observed optimizer
steps. The diagnostic PPO subclass delegates the optimization loop to SB3;
ordinary `PPO.load(..., device="cpu")` loads its checkpoint.

The reward plot uses raw complete episode returns before timeout bootstrap, with a
trailing mean only when at least 20 episodes exist. Heights/tilts/holds are sampled
at control-step boundaries; torque saturation is weighted by actual physical steps.
A wall cutoff leaves partial rollout samples unoptimized and partial episodes out
of complete-episode statistics. The 20-episode, 80%-within-two-transitions safety
gate stops expansion at an update boundary. Longer but still very early failures
must also be reviewed before allocating a formal budget.

Recheck the saved checkpoint and regenerate its plot without training:

```bash
.venv/bin/python -m x2_recovery.train reload-check \
  --run-dir runs/ppo_supine/smoke-std-minus1p5-20260922 \
  --output runs/ppo_supine/smoke-std-minus1p5-20260922/reload_recheck.json
.venv/bin/python -m x2_recovery.train plot \
  --run-dir runs/ppo_supine/smoke-std-minus1p5-20260922
```

A future formal run requires a passed smoke/reload and acceptable sampling evidence.
The following command was exercised and **rejected by the sampling guard** (exit 1);
no formal run directory or policy was created. For an eligible future configuration,
it initializes a fresh model/seed, reads the saved configuration, and plans whole rollouts
as `512 * floor(0.8 * budget_seconds * measured_transitions_per_second / 512)`.
The 0.8 factor is compute headroom, not a learning or statistical guarantee. Use a
new output directory; the CLI rejects budgets above 3600 seconds and mismatched
measured configurations. A ten-minute budget does not imply adequate training:

```bash
.venv/bin/python -m x2_recovery.train train --seed 220924 \
  --config runs/ppo_supine/smoke-std-minus1p5-20260922/resolved_config.json \
  --validated-run runs/ppo_supine/smoke-std-minus1p5-20260922 \
  --max-wall-seconds 600 --run-dir runs/ppo_supine/formal-quality-gated-20260922
```

Final verification: **125 tests passed, no failures/errors/skips** (60.195 s),
including 19 training bookkeeping/fault tests; these synthetic tests are not X2
training evidence. A fresh colcon build in
`audit-output/ppo-validation/release-{build,install,log}` passed in 0.73 s, and the
installed CLI ran from `/tmp` with the project venv. `pip check` reported no broken
requirements. Logs, the earlier corrected ROS-path invocation failure, the formal
budget rejection, and a real-X2 one-transition budget cutoff (exit 2, zero updates,
no checkpoint, environment closed) remain in `audit-output/ppo-validation/`.
The cutoff's 0.01-second cooperative budget actually took 0.884 seconds in `learn()`;
the existing bounded reset must return before the callback can stop it.

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="$PWD/src/x2_recovery${PYTHONPATH:+:$PYTHONPATH}" MUJOCO_GL=osmesa \
  .venv/bin/python -m unittest discover -s src/x2_recovery/test -v
.venv/bin/python -m pip check
```

Five formal policy evaluation episodes remain **NOT_EVALUATED**. Reload validation
is a separate execution check and does not count toward those five. ROS recovery
still uses `scripted_baseline`; no trained checkpoint has been connected to the node.

## Evaluation

Formal recovery evaluation is **not evaluated**. It requires five actual episodes
using the selected learned checkpoint identified above. Report successes and failures
against the independent criteria above; a learned policy that fails to recover is a
valid negative result, not a reason to replace it with the scripted baseline.
Reasonable hand contact during rising is allowed, but final stable standing must have no non-foot support above the
measurement tolerance. Environment rollouts and directly initialized standing
fixtures do not count as those five attempts.

## ROS 2 Integration

`recovery_node.py` owns one headless `X2RecoveryEnv`. It loads and validates the model
before offering the service; startup does not reset or execute an episode.
`telemetry_node.py` is a separate ROS process that only subscribes, caches valid
messages and logs the named `left_knee_joint` at 1 Hz. It imports neither MuJoCo nor
training libraries. `recovery.launch.py` starts both processes.

| Interface | Type | Meaning |
| --- | --- | --- |
| `/x2/start_recovery` | `std_srvs/srv/Trigger` | `true` = accepted; `false` = already running. Acceptance is not recovery success. |
| `/x2/recovery_status` | `std_msgs/msg/String` | Exactly `IDLE`, `RUNNING`, `SUCCEEDED`, `FAILED`; changes plus 1 Hz heartbeat. |
| `/x2/joint_states` | `sensor_msgs/msg/JointState` | Actual mapped joint names, measured positions/velocities and acquisition ROS timestamp; effort is empty. |

The execution owner is a `SingleThreadedExecutor`. The short Trigger callback sets
busy (including the pending window) and returns its response. The installed Jazzy
executor calls standard `send_response` before executing the next timer callback.
That callback performs one real supine reset and builds keyframes from its actual
joint state, then returns. Later callbacks each perform at most one
`scripted_targets` → `env.action_for_targets` → `env.step` → `loaded.read_state`
transition. The timer period is `env.control_dt` (20 ms). There is no catch-up loop,
extra PD implementation, simulator bridge or second environment. High-level targets
are an **open-loop scripted_baseline**; the existing low-level bounded PD uses actual
joint feedback every physics substep. Physics, reset settling, rewards and success
thresholds are unchanged.

Only environment termination with `is_success=true` can produce `SUCCEEDED`.
Safety termination, simulation truncation, invalid state and execution exceptions
produce `FAILED`, with the original reason/evidence in logs. Finishing the target
sequence continues its existing final-target hold, without implying success.
A terminal state persists; simulation stops advancing and joint samples stop.
Another explicit request reuses the instance but performs a fresh reset and builds
fresh keyframes. There is no queue, automatic retry or automatic return to IDLE.

| Startup parameter | Default | Semantics |
| --- | --- | --- |
| `seed` | `60` | Nonnegative integer, passed to each explicit reset. |
| `episode_timeout_s` | `20.0` | Passed to `EnvConfig`; simulation time after settled reset. Must align with the existing physics timestep. |
| `recovery_timeout_s` | `30.0` | Monotonic wall budget from acceptance, including pending/reset/control. |

Parameters are read-only after startup; invalid values fail startup. Both nodes reject
`use_sim_time=true`; no `/clock` is provided. Timer scheduling uses a steady clock.
Baseline progress and standing checks use relative simulation time. JointState uses
ROS acquisition time, never a monotonic value disguised as a ROS timestamp.
Wall deadlines are checked before and immediately after reset/step, using `>=`.
Late environment success remains visible in the log but the ROS outcome is FAILED
with `recovery_timeout`. Timely results are not invalidated by later logging/DDS delay.
The timeout is cooperative: it cannot preempt an executing reset/step. The existing
45 s reset and 5 s step watchdogs remain intact. This is neither hard real time nor
proof of a safe physical-robot stop. Ctrl+C unwinds execution before closing the
environment; terminal ROS delivery after context shutdown is only best effort.

Status QoS on both ends is RELIABLE / TRANSIENT_LOCAL / KEEP_LAST / depth 1.
Joint QoS is RELIABLE / VOLATILE / KEEP_LAST / depth 10; Trigger uses standard service
QoS. Topic deliveries have no cross-topic transaction or total-order guarantee.
Telemetry prints `waiting`/`no_sample` before data and `last_sample` plus sample age
at terminal states. After an observed new RUNNING transition, an older sample is
explicitly marked as waiting for a current sample. Late subscribers receive retained
status while the publisher lives, but receive no fabricated joint snapshot.

Build using the existing interpreter (the system colcon shebang is `/usr/bin/python3`):

```bash
cd /home/lang/RL-AgiBot-X2
source /opt/ros/jazzy/setup.bash
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
export ROS_DOMAIN_ID=73
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
VALIDATION_DIR="$(mktemp -d /tmp/x2-ros-validation.XXXXXX)"
.venv/bin/python /usr/bin/colcon --log-base "$VALIDATION_DIR/log" build \
  --base-paths src --packages-select x2_recovery --symlink-install \
  --build-base "$VALIDATION_DIR/build" --install-base "$VALIDATION_DIR/install"
source "$VALIDATION_DIR/install/setup.bash"
ros2 pkg prefix x2_recovery
ros2 pkg executables x2_recovery
head -1 "$VALIDATION_DIR/install/x2_recovery/lib/x2_recovery/recovery_node"
cd /tmp
ros2 launch x2_recovery recovery.launch.py \
  seed:=60 episode_timeout_s:=20.0 recovery_timeout_s:=30.0
```

In another terminal, source the same Jazzy/installation and export the same model,
domain, discovery and RMW settings. The domain isolates these tests from other nodes;
Fast DDS was already the working RMW. Actual CLI commands used (bounded echo):

```bash
ros2 service type /x2/start_recovery
ros2 topic info /x2/recovery_status --verbose
ros2 topic info /x2/joint_states --verbose
ros2 topic echo /x2/recovery_status std_msgs/msg/String \
  --qos-reliability reliable --qos-durability transient_local --once --timeout 5
ros2 service call /x2/start_recovery std_srvs/srv/Trigger "{}"
# Run again while the episode is RUNNING: success=False, Recovery already running
ros2 service call /x2/start_recovery std_srvs/srv/Trigger "{}"
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once --timeout 5
```

The repeatable cross-process probe launches these installed executables from `/tmp`
without injecting source PYTHONPATH. It has bounded discovery, RPC and process waits,
failure exit codes, and finally cleanup of only its own processes. A separate
instrumented real-X2 subprocess wraps standard `send_response`, reset/step and
`read_state` solely to record server ordering and exact timestamp-matched simulator
samples. Fault injection is explicitly synthetic and is not a successful recovery.

```bash
cd /tmp
MUJOCO_GL=osmesa timeout --signal=INT --kill-after=15s 300s \
  /home/lang/RL-AgiBot-X2/.venv/bin/python -m unittest discover \
  -s /home/lang/RL-AgiBot-X2/src/x2_recovery/test -v
timeout --signal=INT --kill-after=15s 240s \
  /home/lang/RL-AgiBot-X2/.venv/bin/python \
  /home/lang/RL-AgiBot-X2/src/x2_recovery/test/test_ros_integration.py \
  --output-dir /home/lang/RL-AgiBot-X2/audit-output/ros-integration/new-run
# Independent wall-time scenario used by the probe:
ros2 launch x2_recovery recovery.launch.py \
  seed:=60 episode_timeout_s:=20.0 recovery_timeout_s:=3.0
```

Actual acceptance on 2026-09-21 in the existing Ubuntu 24.04.5 aarch64 Parallels VM,
Python 3.12.3 / ROS Jazzy / MuJoCo 3.13.0 / Gymnasium 1.3.0:

| Check | Actual result |
| --- | --- |
| Fresh build / installed launch | PASS; colcon 0.74 s; prefix `/tmp/x2-ros-validation.s2cp2X/install/x2_recovery`; both executable shebangs use the existing venv; launched from `/tmp`. |
| CLI / QoS / acceptance | PASS; Trigger type verified, IDLE observed, `success=True` acceptance and concurrent `success=False` rejection. Both topics and the independent Telemetry process had compatible explicit QoS. |
| Normal baseline | FAILED as expected: environment `time_limit`, 20.000 s simulation, 1,000 control steps, 20.673 s wall; no wall timeout or safety error. |
| Terminal hold / same-instance repeat | PASS; no extra joint samples after terminal, FAILED remains observable; next request performs reset again: 20.000 s simulation, 1,000 steps, 20.638 s wall, `time_limit`. |
| Independent wall timeout | PASS; budget 3.0 s, episode limit 20.0 s; 116 steps, 2.320 s simulation, 3.004352 s wall; `FAILED / recovery_timeout`, overshoot 4.352 ms. |
| Actual telemetry | PASS; 31 mapped joints move; 9 received snapshots matched names/q/dq from that same simulator read at the same ROS timestamp exactly (maximum difference 0). |
| Busy phases / response ordering | PASS; pending checked in logic tests; real request sent inside recorded reset interval rejected before episode end; normal stepping requests rejected. Standard response-send completion preceded reset entry. |
| Late Telemetry | PASS; retained FAILED received; VOLATILE joint topic gives `no_sample`, not invented position. |
| Errors / cleanup | PASS; real invalid-parameter and missing-model startup failures; synthetic reset/action/step/nonfinite/deadline cases; real stepping followed by an injected error, then successful explicit reset/retry; active Ctrl+C unwinds before close. All owned processes and the test-created CLI daemon exited. |
| Regression | 106 unittest tests passed, 0 failed, 0 skipped; 68 cross-process assertions passed. Missing-dependency injection also exited 1 as expected. Synthetic successful termination tests are not real recoveries. |

The single-thread choice was measured, not inferred from historical timings:
creation 0.637 s, real reset 0.571 s, 100 headless steps: median 17.55 ms,
P95 20.84 ms, maximum 22.89 ms. Final warm-client RPC measurements:

| Measurement | Samples | Median / P95 / maximum |
| --- | ---: | --- |
| Acceptance RPC | 6 | 0.578 / 1.319 / 1.528 ms |
| Busy RPC during stepping | 16 | 41.344 / 42.467 / 42.719 ms |
| Busy RPC sent during reset | 1 | maximum 589.831 ms; no meaningful percentile estimate |
| Instrumented complete service callback, excluding response send | 6 | 0.218 / 0.233 / 0.235 ms |

Every observed acceptance/busy RPC met the fixed **1.0 s** local headless regression
budget. This is a project budget, not a PDF metric or ROS real-time guarantee.
The CLI acceptance/busy processes took 0.590/0.577 s, including interpreter startup
and discovery; those are not service callback timings. The wall-time budget of 30 s
was sufficient for the full 20 s simulation on this VM.

**ROS 2 integration and scripted-baseline end-to-end validation: COMPLETE.**
**ROS integration passed; the scripted baseline did not recover to standing.**
Both full episodes had `is_success=false`, zero standing dwell, final pelvis height
0.07636 m and torso tilt 73.59 degrees; non-foot support remained 0.64335 body weights.
These are ROS integration runs, not the five formal policy evaluations. Training
has not been run (not declared blocked), and no trained checkpoint exists. This does
not demonstrate learned recovery or physical-robot control.

Commands, raw CLI outputs, node/client logs, matched samples and strict JSON summaries
are retained under `audit-output/ros-integration/run-20260921-231426-Xwwi/` (ignored
raw evidence). The authoritative final run is `integration-acceptance/`; top-level
`summary.json`, `acceptance-build-and-install.log`, `regression-acceptance.log` and
`acceptance-code-identity.json` bind the results to the tested code. Base HEAD was
`559fd0e63e2fff45d650089ab0a71b0c146437f3` with only this integration work uncommitted;
the per-file SHA-256 manifest identifies the tested implementation independently of
the subsequent local commit. Earlier attempts remain separately named and unchanged.


## Validation and Reproducibility

### Build, imports and regression

Use the existing venv and the overlay built in [Setup](#setup). Output paths for new
runs use component names. All 73 regression tests passed without skips in the target
VM, along with a fresh package build. Module/import compatibility was also checked
against baseline `2958bc1`: default, perturbed-reset and timeout rollouts produced
byte-identical observations, rewards, info, qpos/qvel, controls, per-substep records
and configuration. Test assertions and simulation algorithms were not changed.

```bash
source /opt/ros/jazzy/setup.bash
source "$VALIDATION_BUILD_DIR/final-install/setup.bash"
MUJOCO_GL=osmesa timeout --kill-after=5s 180s .venv/bin/python \
  -m unittest discover -s src/x2_recovery/test -v
MUJOCO_GL=osmesa timeout --kill-after=5s 30s ros2 run x2_recovery runtime_check model
MUJOCO_GL=osmesa timeout --kill-after=5s 30s ros2 run x2_recovery runtime_check render \
  --asset-repo "$X2_ASSET_REPO" --output audit-output/model-render
```

### Model audit and evidence review

The preserved passive model audit was recorded at implementation `0c627d7`, before
the conservative waist-margin correction. Its collision/limit and interface checks
were subsequently rerun for the effective model fingerprint above. Quantitative
scope and original failures remain distinct from settled-reset and standing criteria.

| Audit area | Result | Measured evidence and scope |
| --- | --- | --- |
| Floating base | PASS | root pelvis (body 1), free joint 0; qpos address 0 (7 coordinates), DOF address 0 (6 velocities); nq=38, nv=37, nu=31; no equality, mocap, tendon, gravcomp or callbacks. Contact-free whole-robot COM drops 0.0495405 m in 100 ms, matching semi-implicit Euler gravity; valid independent translation/rotation tested. |
| Mass/inertia | PASS | 32 dynamic bodies; effective 41.966473 kg; qpos0 COM world `[0.00168761,0.00024727,0.71173956]` m. Positive principal moments and triangle inequalities; full common-frame tensors match URDF within export precision (max mass error 3.8e-5 kg, tensor error 4.65e-7 kg m²). Fixed sensor/base frames add no omitted mass. |
| Collision coverage | PASS, bounded poses | Back, both forearms, wrist/hand proxies, knee/shin meshes and feet have actual active floor force. Peak individual normal forces: back 448.54 N; forearms L/R 468.11/477.70 N; wrist proxies 77.21/141.48 N; shins 564.27/509.91 N; feet 344.44/386.04 N. Six 180 ms whole-model probes; largest floor penetration 10.99 mm (<15 mm), required-pose self penetration 2.66 mm (<5 mm); no warnings/reset. Four collision-only OSMesa images inspected. |
| Joint ranges | PASS | 31 coordinate-equivalent revolute joints, no unmatched movable/mimic joints; six restrictions, six stricter source ranges retained, 19 exact agreements. |
| Actuator semantics | PASS | 155 fresh forward checks: zero, ±40% and ±105% source inputs for every motor; sparse transmission moments, sign and measured actuator/joint forces agree within 1e-10 N m. Raw ctrl remains unclipped in its buffer. |
| Effective limits | PASS, stated scope | All finite joint bounds enabled. 62 lower/upper probes with 0.1 N m outward input, 80 ms each; explicit named limit constraints and restoring acceleration. Disposable copies disable contact/frictionloss and gravity only. Initial/peak penetration 0.002 rad (<0.005); max final 0.000270 rad (<0.001). Does not certify full-collision reachability of every bound. |
| Indexing | PASS | 31 unique controlled hinges; distinct nonzero positions/velocities and isolated inputs verified. Head motors occur before arms in ctrl order but after arms in joint traversal. No magic scalar-state slice or joint-ID/ctrl-index assumption. |

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
does not apply to the stricter settled-reset criteria. In particular, the back probe's
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
These short passive probes do not establish an episode-ready supine state. The
separate settling validation below excludes trapping and applies stricter steady
penetration criteria.

COM comparison uses the actual `mjINT_EULER` integrator, now explicitly checked:
with zero initial velocity, N=100 and h=0.001 s,
`g*h²*N*(N+1)/2 = -0.0495405 m`. The continuous value `g*(N*h)²/2 = -0.04905 m`
is reported separately and is not used as the discrete acceptance target.

Fresh model-audit command:

```bash
MUJOCO_GL=osmesa timeout --kill-after=5s 60s ros2 run x2_recovery runtime_check audit \
  --output audit-output/model-audit
```

A completed physical audit still exits 1 until its generated images are reviewed.
For a later rerun, `--image-review-from` may name a reviewed report. Only observations
with matching image SHA256 **and model/probe context digest** can be reused. The
context binds compiled physics/mapping, pinned assets, MuJoCo version, probe definitions,
tolerances and loader/audit hashes. Legacy observations without that digest are
rejected. The same read/write path is safe: observations are read first, then a new
incomplete report replaces the old result; numerical PASS is never imported.

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
upstream hashes are verified after the probes.

### Supine reset and causal actuation

Recorded reset validation, implementation `9566fd5390ffe0ebb66e586fcf679f043228b229`:
**20/20 fixed + 20/20 seeds 100–119**, each after
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
The worst transient was floor/geom 80 (`right_wrist_roll_link`), seed 106 at 0.078 s;
the worst window value was floor/geom 67 (`left_wrist_roll_link`), seed 110 at 0.144 s;
the largest final value was floor/geom 80, seed 117 at 1.653 s. Initial penetration
was zero, with the independently verified 2 mm clearance.

**Preserved failures:** `natural-baseline.json` records slow waist/hip drift;
`pose-development*.json` and `low-energy-development.json` retain rejected candidates.
The first real acceptance run, `verification/run-20260920T085519-56c2e7/report.json`,
failed **0/20 fixed and 0/20 seeded passes** on knee impact exceeding 0.005 rad at
about 105 ms. The initial pose was then changed to reduce the fall, with no relaxation
of the acceptance criteria. These are reset experiments, not recovery evaluation episodes.

`simulation_validation.py` performs 31 named joint causal tests. Zero, positive and negative rollouts
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
Zero torque does not instantaneously stop motion. The causal diagnostic contains no standing controller.
Measured transmitted-effort error was zero; worst individual-test hinge speed was
0.258136 rad/s. Grouped tests peaked at 0.036139 rad/s and 0.194557 mm floor penetration,
with no measured self penetration. The visible head-yaw demonstration reached about
0.13064 rad and returned towards 0.0338 rad after the negative pulse.

```bash
MUJOCO_GL=osmesa timeout --kill-after=5s 180s ros2 run x2_recovery runtime_check simulation-check \
  --output audit-output/simulation-validation
MUJOCO_GL=glfw PYGLFW_LIBRARY_VARIANT=x11 LIBGL_ALWAYS_SOFTWARE=1 \
  timeout --kill-after=5s 90s ros2 run x2_recovery runtime_check simulation-live \
  --repeat 2 --output audit-output/simulation-live
MUJOCO_GL=osmesa timeout --kill-after=5s 180s ros2 run x2_recovery runtime_check simulation-record \
  --repeat 2 --output audit-output/simulation-record
```

Live/recorded views use copied actual simulation states and never feed rendered state
back into physics. Reports distinguish generated media from inspected media.

| Area | Status | Evidence |
| --- | --- | --- |
| RESET | PASS | `verification/run-20260920T091936-386c73/report.json`: 20+20, three same-seed repeats, full dwell/hold and preserved final state |
| ACTUATION | PASS | Same report: 31 causal joint checks, no clearance fixtures, four grouped sequences |
| LIVE_VIEWER | PASS | `live/run-20260920T091948-96ae14/report.json`: exit 0, two complete resets/sequences, 19.369 s wall time, 190 frames; normal and collision images actually inspected |
| RECORDING | PASS | `recording/run-20260920T092545-7e098b/report.json` and `demonstration.gif`: exit 0, generation 84.327 s; actual-state replay inspected and played locally |

These report paths are relative to the [simulation evidence archive](audit-output/step4/).

The live backend was GLFW X11 on real desktop `DISPLAY=:0` (Wayland session), using
already installed llvmpipe LLVM 20.1.2 with process-local `LIBGL_ALWAYS_SOFTWARE=1`.
Observed rate was **9.81 FPS** while other checks/rendering ran, below the 25 FPS target;
there is no real-time-performance claim. The 1 ms physics step was unchanged. The GIF
has 190 frames, 25 nominal FPS and 7.6 s playback; each explicit reset covers simulation
time 0–3.661 s (7.322 s total). Extra phase-boundary samples explain the small playback
duration difference. No recorded frames were dropped; trajectory timestamps are saved.
The existing GTK3/GdkPixbuf native image player mapped a window on `:0` and advanced
all 190 distinct frames during 10.007 s of playback, exit 0.

### Standing detector validation

Recorded standing-detector verification, using frozen settings and fresh initialization:

| Evidence layer | Actual result |
| --- | --- |
| Constructed logic | 22 tests PASS: boundaries, interruptions, drift, force transform/aggregation, duplicate/gap/timeouts, fixture/origin gates |
| Actual X2 API / handoff tests | 5 PASS: mappings, Jacobian velocities, invalid models/forces, unchanged state, real residual supine handoff |
| Free-base standing; actual yaw +1.1 rad; independent repeat | 3/3 PASS, each continuous **3.938 s** (t=0.062–4.000); recovery_success always false |
| Supine, sitting, kneeling, left wrist support, single-foot, airborne, brief standing | 7/7 exercised and rejected; no skipped physical cases |
| Regression at the recorded detector validation | **49 PASS**, no skips; reset/model/actuation tests retained |
| Installed entry point / visual evidence | Fresh isolated colcon build PASS; venv and source paths verified; 18 normal/collision PNGs inspected |

Across positive windows: pelvis 0.671282–0.673721 m, tilt <=7.292 degrees,
left/right loads 0.4780–0.5164W / 0.4764–0.5044W; non-foot force zero; base/COM
speed <=0.06513/0.07602 m/s; base/torso angular speed <=0.13714/0.17422 rad/s;
joint speed <=0.32571 rad/s. Maximum pelvis drift 47.565 mm, feet <0.575 mm,
floor penetration <0.981 mm, no self penetration or joint-limit excess. This has
limited drift margin and establishes only the tested initializations, not robustness.
Negative witnesses include pelvis-supported sitting (1983.37 N total non-foot force),
kneeling with actual distal thigh/knee-cap hull support (1028.87 N; knee joints 2.2 rad),
left wrist support (38.77 N), right foot 6.180 mm above the floor with zero load,
and airborne upright state with low speed but no support. These are transient negative
witnesses, often failing several predicates; individual necessity is tested logically.
The brief standing trace holds only 0.938 s. No fixture is a recovery attempt.

```bash
MUJOCO_GL=osmesa timeout --kill-after=5s 180s ros2 run x2_recovery runtime_check success-check \
  --output audit-output/success-validation
# Set SUCCESS_RUN to the new run directory printed by success-check.
# Inspect every normal/collision PNG, then write observations.json there:
# {"image-filename.png": "specific observation after actual inspection", ...}
ros2 run x2_recovery runtime_check success-review \
  --output "$SUCCESS_RUN" --image-review-from "$SUCCESS_RUN/observations.json"
```

The check creates a unique directory, starts INCOMPLETE and exits 1 for missing/failed
checks or pending visual review. `success-review` performs no new physics: it verifies
the completed checks, current model/code/config identity, all image hashes and explicit
observations before returning 0. A new implementation identity requires new evidence;
old reports are never cosmetically rewritten. Each report retains per-millisecond
traces, copied actual-state snapshots, geometry checks and labeled replay images.

### Environment integration, rewards and performance

| Evidence | Result |
| --- | --- |
| Existing model/reset/success regression | 49 PASS, unchanged physical model and success settings |
| Environment test suite | 24 PASS: 10 real X2/API/resource tests, 14 synthetic reward/fault/wiring tests; no skips |
| Gymnasium 1.3.0 / SB3 2.9.0 checkers | PASS; only Gymnasium's two expected infinite-bound warnings; render checked separately |
| Real reset, seeds, ownership | >=10 reset/step cycles; exact repeated-seed trajectory; independent instances; preserved residual handoff; perturb 0 and 0.005 tested |
| Control / time / reward | 20 recomputed PD torques and tracker samples per normal step; 23 ms timeout stops at third internal substep; qualified-time/torque integrals and action cost checked |
| Errors / success branch | Fault injection PASS, including 1 ms interruption, immediate stop and success-over-timeout priority; synthetic success is not physical recovery |
| Actual default timeout | 20,000 physics steps / 1,000 actions; 20.000 s, truncated only |
| Independent real standing fixture | 2.938 s continuous hold in a 3 s rollout; standing_held true, recovery_success false |
| Rendering / cleanup | 480x360 uint8 OSMesa; read-only native GLFW/X11 window; six actual trajectory PNGs inspected; 51-frame GIF played locally; repeated RGB creation/close and headless mode checked |
| Fresh package build | isolated venv colcon PASS; installed entry points resolve this checkout and its venv |

The real action tests below all began from verified supine seed 60 (random actions
use a separate fixed RNG seed 600). Saturation is the joint/substep-pair fraction.
Different durations and endings make their returns **diagnostics, not policy rankings**.
Both returns use the same per-transition gamma; all numerical warning counts were zero.

| Action diagnostic | Actual seconds / ending | Saturation | Undiscounted / discounted return |
| --- | --- | ---: | ---: |
| Zero (q_ref target) | 20.000 / timeout | 3.23% | -1.0060 / -0.6387 |
| Random uniform [-0.1,0.1] | 2.000 / bounded stop | 4.38% | -0.1090 / -0.1005 |
| Two-stage hip/knee/elbow targets via action interface | 2.000 / bounded stop | 0.045% | -0.0993 / -0.0946 |
| Legs +/-0.3 target segment | 0.119 / self penetration guard; negative segment not reached | 17.46% | -0.9955 / -0.9897 |
| Arms +/-0.3 | 0.400 / bounded stop | 10.70% | 0.0214 / 0.0218 |
| Waist/head +/-0.3 | 0.400 / bounded stop | 4.99% | 0.2772 / 0.2737 |
| All +1 / all -1 / mixed +/-1 | 0.011 / 0.010 / 0.013, speed guard | 90.03% / 87.42% / 86.10% | about -0.9912 each |

Full-target impulses reached 30.53–31.58 rad/s before the first sampled safety stop;
raw requests reached 2057.46 N m while actual applied torque remained <=118 N m and
inside each individual limit. The legs test reached 15.198 mm self penetration.
These are handled safety terminations, not evidence that full-range exploration is
benign. Targets were not narrowed and gains/physics were not changed to hide them.

Constructed reward checks (not physical recovery): stationary seated potential for
1000 steps returns -1.9500 / -1.2330; a 100-transition rise/fall sequence returns
-0.2825 / -0.1857. Discounted shaping telescopes to
`-Phi(start)+gamma**N*Phi(end)` for `Phi=3H+U`. Immediate synthetic success gives
46.0 / 46.0 versus 45.8 / 43.5603 after 50 unqualified steps. Early safety abort can
be preferable to waiting with costs and no eventual success (-1.9504 / -1.9504 vs
-2.06555 / -1.96952 over 50 transitions). This known incentive risk needs training-time
investigation; finite diagnostics do not rule out all reward hacking. A separate 1 ms
window-interruption case, starting 2 ms before completion, returns
47.601 / 43.1446 over 101 transitions versus immediate 46.002 / 46.002; consistent
discounting favors completion even when the longer path collects hold reward.
No real success trajectory was invented to tune rewards.

```bash
MUJOCO_GL=osmesa timeout --kill-after=5s 180s ros2 run x2_recovery runtime_check env-check \
  --output audit-output/environment-validation
MUJOCO_GL=osmesa timeout --kill-after=5s 120s ros2 run x2_recovery runtime_check env-record \
  --seconds 2 --output audit-output/environment-recording
MUJOCO_GL=glfw PYGLFW_LIBRARY_VARIANT=x11 LIBGL_ALWAYS_SOFTWARE=1 \
  timeout --kill-after=5s 90s ros2 run x2_recovery runtime_check env-live \
  --seconds 1 --output audit-output/environment-live
```

Each diagnostic creates a new run directory and strict report, starting INCOMPLETE;
errors exit 1 with available evidence. `env-check` exit 0 means automated acceptance;
media commands explicitly require subsequent visual review, rather than implying
that generation proves inspection. Policy traces include real reward/control/state
statistics, with per-physics-step traces enabled only for bounded scripted/stress
cases. Normal environment calls perform no audit I/O. Original failed test logs are
retained (synthetic timestamp setup and exception-message assertions were corrected).

On the final zero-action episode, after excluding loading, reset, the first five
policy steps, logging and rendering: **15.022 ms/step**, **1331.4 physics steps/s**,
**66.57 policy steps/s**, **1.331 simulated seconds/wall-second** over 995 actions.
Separate model load was **0.228 s** with warmed asset/filesystem caches; resets across
nine cases cost **0.550–0.570 s**. Earlier verification measured 1.249–1.283 simulation
seconds/wall-second. These are local bounded measurements, not a real-time guarantee
or a prediction of training duration.

### Runtime dependencies and diagnostic ROS transport

The PPO dependency/runtime smoke test uses one Pendulum-v1 environment without rendering,
seed 42, MlpPolicy, device=cpu, n_steps=128, batch_size=64, n_epochs=2 and 1024 timesteps.
It checks finite data and parameters, an actual parameter change, checkpoint save/load,
and 32 deterministic prediction steps. PPO has a 60 s internal deadline and the complete
runtime command a 120 s external bound. Its temporary checkpoint is removed automatically.
These are compatibility checks, not X2 hyperparameters, convergence evidence, recovery training,
or any of the five formal evaluation episodes.

Cross-process communication was checked with matching Fast DDS, domain 42, localhost-only
discovery and reliable/volatile keep-last-10 topic QoS. No global networking changes:

```bash
export ROS_DOMAIN_ID=42 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
timeout --kill-after=5s 35s ros2 run x2_recovery runtime_check serve --seconds 20 > "$VALIDATION_BUILD_DIR/server.log" 2>&1 &
smoke_pid=$!
trap 'kill "$smoke_pid" 2>/dev/null || true' EXIT
timeout --kill-after=5s 25s ros2 run x2_recovery runtime_check client --seconds 15
client_exit=$?
wait "$smoke_pid"
server_exit=$?
trap - EXIT
cat "$VALIDATION_BUILD_DIR/server.log"
test "$client_exit" -eq 0 && test "$server_exit" -eq 0
```

The client waits for discovery and repeated publications, verifies String content,
JointState names/finite changing positions/timestamps, then checks the Trigger response.
The server reads `probe_hinge` qpos from the tiny simulated scene. Endpoints are only
`/x2_smoke/status`, `/x2_smoke/joint_states`, `/x2_smoke/ping`; none implement the final
recovery interfaces, busy rejection or recovery timeout behavior.

The `runtime` command includes that bounded Pendulum learning smoke test; it is not
needed to use or validate the X2 environment. To reproduce the dependency check:

```bash
timeout --kill-after=5s 120s ros2 run x2_recovery runtime_check runtime \
  --asset-repo "$X2_ASSET_REPO"
```

| Runtime check | Status | Observed evidence |
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
| X2 recovery training/evaluation | NOT RUN | Dependency smoke is not recovery evidence |

The simple-scene throughput is not an X2 training-speed estimate. OSMesa uses the
existing system library; graphics package/driver configuration was not changed.
The failed passive viewer route remains separate from the validated native renderer.

### Evidence locations and provenance

Generated audit evidence is intentionally ignored, not shipped as model assets or
rewritten during source cleanup. Links below retain the original directory names
because they identify actual runs. Embedded provenance labels, code hashes and past
command names describe those runs; they are not current component names. In particular,
the model override's original settling-diagnosis label and the calibration record's
source directory remain literal provenance. New runs use the semantic commands above.

| Evidence | Original local artifact |
| --- | --- |
| Model passive probes and review context | [Model audit](audit-output/step3-closeout/report.json); [original model report](audit-output/step3/report.json) |
| Repeated reset / causal actuation | [Simulation report](audit-output/step4/verification/run-20260920T091936-386c73/report.json) |
| Rejected reset attempt | [Original failed batch](audit-output/step4/verification/run-20260920T085519-56c2e7/report.json) |
| Native live simulation | [Live report](audit-output/step4/live/run-20260920T091948-96ae14/report.json) |
| Actual-state simulation replay | [Recording](audit-output/step4/recording/run-20260920T092545-7e098b/demonstration.gif) |
| Standing detector positives/negatives | [Success report](audit-output/step5/verification/run-20260920T103825-c8b74c/report.json) |
| Environment configuration / results | [Environment report](audit-output/step6/final-verification/run-20260920T115046-8512ac/report.json); [combined verification](audit-output/step6/acceptance.json) |
| Actual environment motion | [2 s recording](audit-output/step6/recording/run-20260920T114224-057043/env-rollout.gif); [native live frames](audit-output/step6/live/run-20260920T114327-515f59/) |
| Module/import regression and deterministic environment comparison | [Regression evidence](audit-output/presentation-cleanup/run-20260920-semantic-names/) |

The environment recording was inspected at the start, middle and end; native playback
on DISPLAY=:0 advanced 250 frames / 51 distinct images in 10.008 seconds. Reports retain
interpreter/source identity, full resolved configuration, commands, exit codes and
review observations. Original failed tests and incomplete runs remain available.
The two-page HRS task PDF was inspected locally; its content is not redistributed.
Commit history records implementation milestones without squashing or rewriting them.

## Limitations and Future Work

No real supine-to-standing recovery was observed. The bounded scripts exercise the
environment; they have not recovered the robot to standing. ROS service-to-simulation
integration has passed with this baseline. PPO optimization and reload now pass, but
poor sampling has deferred the larger formal training experiment. Five formal
evaluation episodes also remain outstanding; integration and standing fixtures do
not complete those requirements.

Simulation uses convex collision hulls, spherical foot proxies and soft constraints;
measured nonzero penetration is not mathematical nonintersection. Finite checks do
not establish every joint-bound pose's collision-free reachability or whole-action-space
safety. Full-target actions can terminate almost immediately at the documented safety
guards. PD gains/reward weights remain initial choices. PPO throughput is measured
for the documented configuration; learned recovery robustness, the behavioral effect
of early-abort returns, and observations available on hardware remain unverified.
No hardware calibration, GPU capability or real-time training guarantee is inferred
from the VM tests.

The original arm-at-side contact failures remain relevant to motion planning: wrists
can become trapped against hip hulls. A safe spread-arm supine reset and successful
arm-forward probes do not erase those failures.

### Known numerical and rendering issues

- **Soft waist limit drift:** source soft-limit loading produced a steady excess;
  the conservative activation margin above corrects the reset case. The unchanged
  [limit-margin semantics](https://github.com/google-deepmind/mujoco/blob/3.13.0/doc/XMLreference.rst)
  and [soft-constraint model](https://mujoco.readthedocs.io/en/stable/modeling.html)
  explain why contact/limit penetration is not mathematically zero.
- **Reset geometry and API compatibility:** installed 3.13.0 headers and matching
  [reset](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_io.c)/
  [signed-distance](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_support.c)
  source were checked when a versioned rendered API page was unavailable.
- **Desktop graphics:** virgl (Apple M5 Pro Compat) produced black X2 framebuffer
  readback, including minimal scenes without shadows or meshes. The native renderer
  works with existing X11 GLFW and process-local `LIBGL_ALWAYS_SOFTWARE=1` (llvmpipe).
  The virgl defect is not fixed. Passive viewer attempts suffered a segmentation
  fault, GLXBadDrawable or timeout; opening a window or joining a renderer alone did
  not establish valid images. Use `simulation-live` or `env-live` for the validated
  route, not the dependency probe's `viewer` command. No system graphics settings
  or drivers were changed.
- **Preserved graphics failures:** `gl-clear-probe.log`, `mujoco-gl-probe.log`,
  `x2-gl-probe.log` and black frames in the simulation archive retain reproductions.
  A high-resolution software run hit a 40 s external timeout; the compact native
  renderer completed its bounded sequence. A recording with complete media exited
  143 for an unestablished reason
  ([original report](audit-output/step4/recording/run-20260920T091949-6bf409/report.json));
  the separately verified recording exited 0 with byte-identical trajectory/GIF and
  passed native playback. Failed passive probes used X11/GLFW with and without the
  software-rendering environment flag, bounded by a 20 s timeout.

Rendering references: [MuJoCo visualization API](https://mujoco.readthedocs.io/en/stable/programming/visualization.html),
[pyGLFW backend selection](https://github.com/FlorianRhiem/pyGLFW),
[Mesa process environment](https://docs.mesa3d.org/envvars.html).

## References

Targeted references read: HoST's
[`eval_ground.py` at 58ea00e](https://github.com/InternRobotics/HoST/blob/58ea00e8541911540a5ac53879ad98827c8e9fb5/legged_gym/legged_gym/scripts/eval/eval_ground.py)
(height-history metrics); dm_control's
[`humanoid.py` at 87e046b](https://github.com/google-deepmind/dm_control/blob/87e046bfeab1d6c1ffb40f9ee2a7459a38778c74/dm_control/suite/humanoid.py)
(torso-axis/subtree measurements, reward not acceptance); installed Gymnasium 1.3.0
`humanoidstandup_v5.py` and [documentation](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/)
(height reward, no success termination). API semantics were checked against installed
MuJoCo 3.13.0 headers, [matching engine source](https://github.com/google-deepmind/mujoco/blob/3.13.0/src/engine/engine_core_smooth.c),
[API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) and
[simulation synchronization](https://mujoco.readthedocs.io/en/stable/programming/simulation.html).
No reference framework was installed or another robot's thresholds copied.

References read for mechanisms only: installed Gymnasium **1.3.0** `core.py`,
`utils/env_checker.py`, `envs/mujoco/humanoidstandup_v5.py`, and
[Env API](https://gymnasium.farama.org/api/env/)/
[time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/);
installed SB3 **2.9.0** checker, Monitor and DummyVecEnv plus
[custom-env documentation](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
(the online master was newer; installed source defined compatibility).
[legged_robot.py at 7aeb9ee](https://github.com/leggedrobotics/legged_gym/blob/7aeb9ee4e987d7a9cf180e1e79921fc3f51bcf06/legged_gym/envs/base/legged_robot.py)
confirmed per-substep PD recomputation;
[go1/getup.py at 8a4b464](https://github.com/google-deepmind/mujoco_playground/blob/8a4b4642d8eba8a80ac99ed125cb62c16e1457ad/mujoco_playground/_src/locomotion/go1/getup.py)
illustrated gating stillness near the goal. No framework, robot weights, low-body
termination or reward implementation was copied. MuJoCo **3.13.0** installed API
headers and actual velocity/Jacobian tests defined frame semantics (the versioned
web API URL was unavailable). [Ng, Harada and Russell, 1999, Eq. 2/Theorem 1](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
informed potential shaping and the telescoping test; this does not assert policy
invariance for the additional hold/cost/bonus terms or prove full observability.

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
