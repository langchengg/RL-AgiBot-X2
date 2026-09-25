# AgiBot X2 Ground Recovery

Native MuJoCo/Gymnasium supine recovery for AgiBot X2. The recommended **v6** controller
combines a repaired reference motion, bounded torso-to-ankle analytical feedback and a
trained PPO residual. It achieved **5/5 fixed-supine recoveries** and **20/20 small
perturbations**, including the original two-second standing criterion and the declared
whole-recovery constraint checks. Reference + analytical feedback with **zero PPO
residual also passed 20/20**: PPO benefit is not established. ROS runs the same frozen
bundle; `scripted_baseline` remains an explicit debug choice. These are finite simulation
results, not broad-pose robustness or hardware certification.

### Demo — v6 recovery

[![AgiBot X2 v6 recovering from a supine pose and holding a stable stance][v6-gif]][v6-video]

[Watch or download the full MP4][v6-video] · [Recording provenance][demo] · [Recording script][recorder]

One independent real-simulation reproduction, seed 221030: **6.616 simulated seconds**,
including the final **2.000-second continuous standing hold**, at approximately 1× speed.
The GIF loops this one episode; it is neither a ROS recording nor another formal evaluation.
The [v5 video][v5-video] remains a separately labeled historical reproduction.

## Setup and Quick Start

### First installation

Target: **Ubuntu 24.04 ARM64 Parallels**, system **Python 3.12.3**, **ROS 2 Jazzy**.
System prerequisites: `git`, `ca-certificates`, `python3`, `python3-venv`, `python3-pip`,
`python3-setuptools`, `python3-numpy`, `python3-pil`, `python3-matplotlib`, `libosmesa6`,
`python3-colcon-common-extensions`, `ros-jazzy-ros-base`, `ros-jazzy-rmw-fastrtps-cpp`.
Jazzy must supply `rclpy`, `rcl_interfaces`, `sensor_msgs`, `std_msgs`, `std_srvs`,
`launch`, `launch_ros`, `ament_index_python` and the package/node/topic/service/launch CLIs.
The commands below assume these system packages are already installed.

[requirements.txt](requirements.txt) pins the direct Python dependencies, not every
transitive dependency. `--system-site-packages` inherits Ubuntu's plotting/build packages;
sourcing Jazzy exposes ROS modules. cffi satisfies inherited PyNaCl. Keep the versions:
NumPy 1.26.4, MuJoCo 3.13.0, Gymnasium 1.3.0, SB3 2.9.0, Torch 2.8.0+cpu,
Pillow 10.2.0, cffi 2.1.1 and matplotlib 3.6.3. Do not install PyPI `rclpy` or use `sudo pip`.

Use a fresh shell without an old ROS overlay. A **full clone** includes the published
controller inputs and history; a source-only checkout cannot run the policy. These are
first-install commands. For an existing environment, skip to the shared runtime block.
For existing model assets, skip their clone/checkout and verify the fixed revision instead.

```bash
set -e
cd "$HOME"
git -c http.version=HTTP/1.1 clone https://github.com/langchengg/RL-AgiBot-X2.git RL-AgiBot-X2
export WS="$HOME/RL-AgiBot-X2"
cd "$WS"
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m venv --system-site-packages .venv
touch .venv/COLCON_IGNORE
export PY="$WS/.venv/bin/python"
"$PY" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu'
"$PY" -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
"$PY" -m pip check
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
mkdir -p "$(dirname "$X2_ASSET_REPO")"
git -c http.version=HTTP/1.1 clone --filter=blob:none --no-checkout \
  https://github.com/AgibotTech/agibot_x2_urdf.git "$X2_ASSET_REPO"
git -C "$X2_ASSET_REPO" sparse-checkout set X2_URDF-v1.3.0
git -C "$X2_ASSET_REPO" checkout --detach 60c5de582c523cd188f563819e62d34cfdc3d2d0
git -C "$X2_ASSET_REPO" status --short
"$PY" /usr/bin/colcon --log-base "$WS/log" build --base-paths "$WS/src" \
  --packages-select x2_recovery --symlink-install \
  --build-base "$WS/build" --install-base "$WS/install"
```

Use the venv Python to invoke colcon; `/usr/bin/colcon` has a system-Python shebang.
The package installs ament metadata, the resource marker, launch file and node entry points.
Model assets and results remain external to package/share. [Installation evidence][installation]
records the reused system components; this is not a claim of validation on a new machine.
For offscreen rendering or the regression suite, set `MUJOCO_GL=osmesa` **before**
starting Python; the default GLFW backend needs a display. Non-rendering recovery does not.

### Shared runtime context

Run this block **once in every new terminal**, then execute any current-controller
example below in that same shell. Set `WS` to your checkout; set `OVERLAY` to its actual
install directory if you used an isolated build. No example depends on the author's home path.

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export OVERLAY="$WS/install"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$OVERLAY/setup.bash"
export INPUT="$WS/results/control-repair/20260924T151819Z/training-run"
export POLICY_SHA=af70fe7ed8ae6c8b1a2ed1f73cd247b116789488120586fc1d8dc949fbca5540
cd /tmp
ros2 pkg prefix x2_recovery
"$PY" -c 'import sys, x2_recovery.env; print(sys.executable, x2_recovery.env.__file__)'
```

Module paths must resolve to this checkout/build, without an old source `PYTHONPATH`.
The [complete v6 bundle][v6-inputs] contains checkpoint, resolved reference/feedback/control
configuration, manifest, progress, reload validation and saved observation/action probe.
Strict loading checks their identities and the model; a lone ZIP is insufficient.

### Run one published recovery

Continue after the shared block. `check` does not reset, step or train; `run` copies the
six required inputs to a new writable directory before executing one real episode.

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -m x2_recovery.reproduce check --training-run "$INPUT" \
  --expected-checkpoint-sha256 "$POLICY_SHA" --output "$WS/runs/check-$RUN_ID"
"$PY" -m x2_recovery.reproduce run --training-run "$INPUT" \
  --expected-checkpoint-sha256 "$POLICY_SHA" --seed 221030 --output "$WS/runs/replay-$RUN_ID"
```

Outputs must be new and disjoint from inputs. Read
`runs/replay-<id>/training-input/development-221030/summary.json`: `success`, `reason`,
`sim_duration_s` and `max_stable_seconds` describe the episode. `execution.json` records
the child exit code; exit 0 alone is not recovery success. Nothing installs or trains implicitly.

### Historical v5 reproduction

Use a separate shell with the shared block, then the following commands. Historical v5
requires matching source; only these child processes receive its `PYTHONPATH`.
The full clone contains this commit. If using an existing shallow clone and the object
is missing, first run `git -C "$WS" fetch origin f542ec6414ae1366bc17868c0f43de80c580c8fc`.

```bash
set -o pipefail
HISTORY_COMMIT=f542ec6414ae1366bc17868c0f43de80c580c8fc
HISTORY="$(mktemp -d "${TMPDIR:-/tmp}/x2-v5-XXXXXX")"
git -C "$WS" archive "$HISTORY_COMMIT" src/x2_recovery/x2_recovery | tar -x -C "$HISTORY"
OLD_INPUT="$WS/results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run"
OLD_SHA=11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
PYTHONPATH="$HISTORY/src/x2_recovery" "$PY" -m x2_recovery.reproduce check \
  --training-run "$OLD_INPUT" --expected-checkpoint-sha256 "$OLD_SHA" \
  --output "$WS/runs/history-v5-check-$RUN_ID"
PYTHONPATH="$HISTORY/src/x2_recovery" "$PY" -m x2_recovery.reproduce run \
  --training-run "$OLD_INPUT" --expected-checkpoint-sha256 "$OLD_SHA" --seed 221030 \
  --output "$WS/runs/history-v5-replay-$RUN_ID"
```

No installed v6 source or frozen manifest is replaced. Historical full publication
inventories must likewise be checked at their declared historical tree.

## Robot Model and Simulation

**X2 Ultra v1.3.0** comes from [AgibotTech/agibot_x2_urdf][upstream], commit
`60c5de582c523cd188f563819e62d34cfdc3d2d0`: `X2_URDF-v1.3.0/x2_ultra.urdf`,
`x2_ultra.xml` and `scene.xml`. Assets remain external under **Mulan PSL v2**.
Project package metadata is **UNLICENSED**; no additional project license is granted.

[model.py](src/x2_recovery/x2_recovery/model.py) applies these in-memory modifications:

- Supply the URDF pelvis mass, COM and full inertia tensor missing from the MJCF.
  Pelvis mass changes 5.031810659 → 3.523487 kg; total mass is 41.966473 kg.
- Intersect six wider MJCF ranges with URDF limits: waist/head yaw and both wrist
  pitches/rolls. Keep the other, already narrower model ranges.
- Activate the waist-pitch upper soft limit 0.005 rad early to avoid settled-reset
  drift beyond the unchanged 0.314 rad bound. This is a project modeling choice.

The floating base has 7 position/6 velocity coordinates; `nq=38`, `nv=37`, with
31 hinged joints and direct torque motors. Joint/control/DOF mapping is discovered
and checked. Motor effort limits span 0.6–118 N m; sourced URDF speed ratings are
measured, not enforced by hardware speed clamps. No external recovery force or base restraint is used.
Physics: **1 ms Euler**, gravity −9.81 m/s², Newton solver, pyramidal friction cone,
100 iterations, tolerance 1e-8. Control: **50 Hz**, PD recomputed at every physics step.
Original friction, collision and effort settings are retained. Mesh collisions use
convex hulls, each foot uses twelve 5 mm spheres, and fingers are not articulated.
Soft contacts permit penetration; these approximations limit transfer to hardware.
Full ranges, gains and identities are in the **[current v6 configuration][v6-config]**.

Experiments used 8 virtual CPUs, approximately 11 GiB RAM/4 GiB swap and CPU PPO:
2 Torch compute threads, 1 interop thread, 1 compute thread per training worker.
Headless physics needs no display; the real demo uses OSMesa. Virtual graphics do not
imply GPU training. Reference provenance includes HumanUP motion retargeting and local
X2 refinement. [Attribution and raw-motion licensing limits][notices] remain explicit;
the upstream repository license is not asserted to grant rights to its external raw motion.
Codex assisted development/documentation. The local two-page HRS task brief was checked,
not redistributed; all empirical claims refer to saved execution evidence.

## Environment Design

The current chain is `ControlledRecoveryEnv(X2RecoveryEnv(saved_env), saved_control)`:
**`reference_residual` / `targets-v6` / `independentlegs17`**.
Reset starts face-up with spread arms and 2 mm clearance, then settles under zero control:
up to 5 s to qualify, 0.5 s continuous dwell and 1 s unassisted hold. Accepted rest requires
face-up tilt ≤20°, linear/angular/joint speed ≤0.01 m/s / 0.05 rad/s / 0.02 rad/s,
floor/self penetration ≤1/ 0.1 mm and support 80–120% of body weight including torso contact.
The settled qpos/qvel and MuJoCo time carry into control. Recovery time excludes settling.
Nominal perturbation is zero; the reset supports per-joint uniform noise up to ±0.005 rad,
while the published paired test uses ±0.002 rad. This is not general fallen-pose randomization.
Reset clears previous action, adopted target, reference history, phase and standing/drift memory.

The **149 float32 observations** use fixed scaling, without VecNormalize or extra clipping.
`R` is pelvis body-to-world rotation, `gravity_unit=[0,0,-1]`, and `W` is robot weight.
All measurements come from the simulator.

| Components, in order | Count | Scaling / meaning |
| --- | ---: | --- |
| Joint q, dq | 31 + 31 | `(q-midpoint)/half_range`; `dq/5 rad/s`, actuator order |
| Gravity, base linear/angular velocity | 3 + 3 + 3 | `R.T*gravity_unit`; `R.T*v/1 m/s`; body angular velocity / 2 rad/s |
| Pelvis height and support | 1 + 3 | Height / 0.6724955472220092 m; left/right vertical foot loads and other contact-force norms / W |
| Previous native action | 31 | Applied target-mapping action, not the policy output |
| Standing memory | 1 + 1 + 9 | Hold / 2 s, anchor-valid flag, pelvis/feet drift in pelvis frame / 0.05/ 0.03/ 0.03 m |
| Adopted target and phase | 31 + 1 | Target normalized by joint range; elapsed recovery / 20 s |
| **Total** | **149** | Native 117 plus 32 controller channels |

`Box(-1,1,(17,))` has six coordinates per leg, waist pitch and four bilateral arm
synergies. **Only four ankle pitch/roll outputs currently have nonzero PPO authority**:
columns 4, 5, 10, 11, each bounded to ±0.002 rad; other residual scales are zero. All **31 physical
actuators** still execute reference/PD control. Historical v5 also used 149/17 but different
reference and residual permissions; the native baseline interface remains 117/31.

The saved reference has 158 linear knots, revised arm-support/hip-knee-ankle transitions,
a flat-foot terminal stance and a waist pulse at 3.05–3.45 s. It queries elapsed time plus
one 20 ms interval and holds its last target after 5.399296 s. The separate head-progress
reward table ends at 8.199296 s; neither reference endpoint is a success condition.
Analytical feedback reconstructs torso tilt/rate from the current observation and waist
chain, adding equal ankle-pitch offsets `clip(0.3*theta+0.03*theta_dot, ±0.08 rad)`.
Its quintic gate starts at 3.5 s over 0.2 s; PPO has a separate 5.5 s/ 0.2 s gate.
This memoryless analytical feedback is **not PPO**. Both gates use simulated time.

Reference + feedback + mapped residual is joint-clipped, then rate-limited: hip/knee/
shoulder/elbow 3, ankle 2, waist/head/wrist 1.2 rad/s. The adopted target enters native
`env.step()` through its existing action mapping. At 1 kHz, `tau=clip(Kp*(target-q)-Kd*dq,
effort_limits)`: hip/knee 600/16.97056, ankle/waist 300/11.31371, shoulder/elbow 100/4.24264,
head/wrist 10/ 0.424264 (N m/rad, N m s/rad). Legal targets do not guarantee legal actual q.
Success terminates; the 20 s horizon truncates. Safety aborts retain floor penetration
>30 mm, self penetration >15 mm, joint excess >0.05 rad or speed >30 rad/s. Numerical errors
require reset; reset/step watchdogs are 45/5 wall seconds. Non-foot support is allowed
during recovery, but not in the final standing window. No state teleport or extra force is used.

Shared model/reset/success/env boundaries, the two ROS nodes and train/evaluate/reproduce
remain separate. `baseline.py` supplies reference data; `model_audit.fingerprint` is used
by training and source identity, so neither is an unused test helper.

## Rewards and Training

### Current objective and selected checkpoint

**`reference-balance-density10-v1`** scales the four repeatable positive densities of
historical `reference-balance-v1` to 10%. Negative costs and terminal rewards are unchanged.
It is not potential shaping. Densities multiply **actual executed dt**, normally 0.02 s;
the nominal final transition is 0.016 s. Let `L,R,O` denote support fractions, `F=L+R`,
`H=clip(height/h_ref,0,1)`, `G=clip((upright_dot-0.5)/ 0.5,0,1)`, and
`p=clip((head_z-0.2043093023)/(1.2258631472-0.2043093023),0,1)`.
`p_ref` comes from the saved head-progress table; q23 excludes head and wrist joints.

| Term | Actual weighted contribution |
| --- | --- |
| Pose guide | `dt*0.1*(1-p_ref)*exp(-mean((q23-reference23)^2)/ 0.35^2)` |
| Head tracking | `dt*0.2*exp(-((p-p_ref)/ 0.2)^2)*clip(F+O,0,1)` |
| Balance | `dt*0.6*H*G*B*N*A*V` |
| Qualified standing | `dt*0.5*qualified`, including native drift predicates |
| Overload / self penetration | `-dt*0.5*G*clip(F-1.2,0,3)` / `-dt*0.5*G*clip(self_depth/ 0.003,0,1)` |
| Torque | `-0.02*sum_substeps(mean31((applied_torque/limit)^2)*physics_dt)` |
| Target change | `-0.01*mean31((adopted_target-previous_target)^2)`, once per transition |
| Success / safety abort | `+50` / `-2`, once at termination |

Here `B=clip(min(L,R)/ 0.2,0,1)`, `N=exp(-(distance(F,[0.8,1.2])/ 0.4)^2)`,
`A=exp(-(O/ 0.1)^2)`, `V=1/(1+(COM_speed/ 0.3)^2+(torso_angular_speed/ 0.75)^2)`.
Tracking encourages progress; bilateral support and low speed encourage balance. Torque
square is not energy; balance costs vanish at tilt ≥60°, and imitation remains rewarding.
Learning reward at 50 Hz is separate from success checks at every 1 ms physics sample.

| Setting | Selected v6 training |
| --- | --- |
| Algorithm/network | SB3 PPO 2.9.0; separate 128×128 Tanh actor/value MLPs, orthogonal initialization |
| Rollout/batch | 2 envs × 256 steps = 512 transitions; minibatch 64 |
| Optimization | lr 0.0001; up to 5 epochs; target KL 0.03; Adam epsilon 1e-5, betas (0.9,0.999) |
| Returns/clipping | gamma 0.999 per transition, GAE 0.95; policy clip 0.2, no value clip; normalized advantages |
| Loss/gradient | entropy 0, value 0.5; gradient norm limit 0.5 |
| Exploration | Diagonal Gaussian, initial log std −2.3; no gSDE or additional action noise |
| Initialization | Fresh actor/critic/log std/Adam; no weight migration, old rollouts or VecNormalize |

The [training manifest][v6-training] records **4096 transitions, 8 complete rollouts and 174
actual Adam updates**; SB3's 25 epoch-attempt counter is different. All 12 completed training
episodes stood successfully; training outcomes are not an independent test. The [reward
curve][v6-reward] plots raw Monitor episode returns against global sampled transitions,
before timeout bootstrap; fewer than 20 episodes means no trailing 20-episode mean.
[Historical v5 training][v5-training] was a separate 4096-transition/138-update run;
[earlier branch totals][totals] are not this checkpoint's continuous training.

The [old reward audit][reward-audit] found a v5 stochastic failure with raw return 191.958311.
Its full sequence is missing, but signed totals prove discounted return ≥72.413169, above
that version's deterministic success 67.173051. This is trajectory-level mismatch, not
proof of deliberate reward exploitation. Reduced densities address that ranking risk.
For current v6, [real reference+feedback / plus-PPO diagnostics][v6-reward-check] give raw
returns 53.859569/53.859579 and discounted 39.037416/39.037423; both recover at 6.616 s.
The positive-density upper bound is 1.4/s; 20 s failure-prefix/latest-success bounds are
17.704528/36.107703. These are theoretical bounds, not additional physical trajectories.
`sum(gamma**t*r_t)` discounts policy transitions, including the short final step. Monitor
return, discounted observed prefix, timeout critic bootstrap and GAE differ; critic tails
are estimates, and no post-terminal rewards or unavailable historical bootstrap are invented.

### Optional training, continuation and evaluation

Use the **shared runtime block** first. Training is not needed to view or run the published
controller. Re-training need not recreate its checkpoint. These commands preserve the input:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
NEW_RUN="$WS/runs/v6-ppo-$RUN_ID"
"$PY" -m x2_recovery.train control-block --run-dir "$NEW_RUN" \
  --control-config "$INPUT/resolved_config.json" --total-timesteps 4096 \
  --max-wall-seconds 1200 --seed 26092490 --workers 2 \
  --learning-rate 0.0001 --log-std-init -2.3 --target-kl 0.03 --n-epochs 5
"$PY" -m x2_recovery.train control-reload --run-dir "$NEW_RUN" --seed 221030 --max-wall-seconds 180
NEW_SHA="$(sha256sum "$NEW_RUN/policy_final.zip" | cut -d ' ' -f 1)"
"$PY" -m x2_recovery.evaluate --training-run "$NEW_RUN" \
  --expected-checkpoint-sha256 "$NEW_SHA" \
  --reload-validation "$NEW_RUN/development-221030/summary.json" \
  --seeds 26092501 26092502 26092503 26092504 26092505 --deterministic \
  --output "$WS/results/evaluation/new-training-$RUN_ID"
```

Inspect reload `success`/reason before evaluation; 0/5 remains a valid reported result.
For **genuine continuation**, this command restores the published optimizer/counters and
global RNG state, then starts fresh legal episodes (not a bitwise worker/simulator resume):

```bash
NEXT_RUN="$WS/runs/v6-resume-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -m x2_recovery.train control-block --run-dir "$NEXT_RUN" \
  --resume-run "$INPUT" --control-config "$INPUT/resolved_config.json" \
  --total-timesteps 4096 --max-wall-seconds 1200 --seed 26092490 --workers 2 \
  --learning-rate 0.0001 --log-std-init -2.3 --target-kl 0.03 --n-epochs 5
```

## Evaluation and Limitations

Success requires a legal supine origin followed by **2.0 uninterrupted simulated seconds**
satisfying every predicate at 1 kHz. A failure clears the window; disconnected intervals
never add up. Recovery time excludes reset settling and includes this final hold.

| Standing predicate | Unchanged threshold |
| --- | --- |
| Pelvis height / torso tilt | ≥0.90×0.6724955472220092 m / ≤15° |
| Each foot / combined vertical load | ≥0.05 W each / 0.80–1.20 W; W=411.69110013 N |
| Other body support | Sum of non-foot ground-force norms ≤0.00001 W |
| Base and COM linear / base and torso angular speed | ≤0.10 m/s / ≤0.25 rad/s |
| Joint speed / pelvis and foot drift | ≤0.50 rad/s / ≤0.05 m and 0.03 m |
| Floor / self penetration / joint excess | ≤1 mm / 0.1 mm / 0.0001 rad |
| Load-bearing contact gap | ≤2 micrometres |

The **additional** [envelope protocol][v6-protocol] requires every recovery physics sample
joint excess ≤0.0001 rad, direction-normalized actual motor effort ≤1 and speed ≤ its sourced
URDF rating. The 0.0001 rad target is a conservative project choice, not a PDF requirement
or hardware tolerance. `admissible_recovery` further requires standing success and five
quality comparisons against exact old nominal peaks: vertical load, summed contact-force
norms, floor/self penetration and joint speed. Remaining supine or stopping early cannot pass.

| Frozen protocol | Standing | Envelope | Standing + envelope + impact |
| --- | ---: | ---: | ---: |
| Historical pure PPO, fixed five | 0/5 | Not this protocol | 0/5 |
| Historical v5 reference + PPO, fixed five | 5/5 | Fails later matching diagnostic | Not admissible; original records unchanged |
| Old ±0.002 rad development pairs, reference/PPO | 2/20 / 3/20 | 0/20 / 0/20 | 0/20 / 0/20 |
| Current v6 + PPO, fixed five | 5/5 | 5/5 | 5/5 |
| New held-out pairs, reference+feedback / plus PPO | 20/20 / 20/20 | 20/20 / 20/20 | 20/20 / 20/20 |

The early pure-PPO smoke policy reached all five 20 s time limits without a qualified
standing sample; its peak pelvis height was 0.0788 m. Its results remain a failed baseline.

The [five individual results][v6-five] use seeds 26092550–26092554; **each** succeeds at
6.616 s with 2.000 s continuous hold, 331 control transitions and 6,616 physics steps.
Identical initial states/trajectories establish repeatability, not five different poses.
Peak pelvis height 0.627942 m is a transient maximum, not a stable-standing height.
The [paired protocol/results][v6-pairs] preregistered seeds 26092500–26092539. Its first 20
resets were legal and physically distinct; independent A/B reset handoffs matched.
Both groups used the same reference, feedback and original ±0.002 rad joint-noise reset.
All 40 episodes remain: 0 A-only wins, 0 B-only wins, 20 ties; common-success time differences 0.
This met the predeclared 16/20 standing/all-constraints goal, not a population reliability guarantee.

| Nominal whole recovery | Historical v5 | Current trained v6 |
| --- | ---: | ---: |
| Maximum actual joint excess | Left ankle pitch 0.0451852 rad | 0 rad across 31 joints |
| Left ankle excess / longest consecutive excess | 0.650 / 0.393 s | 0 / 0 s |
| Maximum joint speed / motor effort ratio | 8.34714 rad/s / 1.0 | 3.83110 rad/s / 1.0 |
| Floor / self penetration | 9.04692 / 3.18148 mm | 3.36709 / 1.66294 mm |
| Ground vertical / summed contact-norm peak | 1626.681 / 1845.448 N | 829.084 / 1134.171 N |
| Duration including the same hold | 4.856 s | 6.616 s |

These are 1 kHz sampled maxima. q/dq are post-integration; forces/contacts are copied
from the integration solve before the existing forward recomputation. Post-forward
estimates are separate. Durations sum 1 ms indicators. The 829 N peak includes mixed
contacts, not isolated foot impact. [Contact comparison][contacts] shows reduced transient
loads and zero-ground-load duration; zero load alone does not prove geometric flight.
The new prefix changes coupled support transitions; no single causal joint is established.
Motor saturation and soft-contact penetration remain, without relaxed physical/success limits.

Across the 40 paired recoveries, speed reached 3.94160 rad/s, floor/self penetration
3.49847/ 1.67070 mm and vertical load 853.263 N. Final margins were small: minimum joint-speed
margin 0.000730 rad/s and floor-depth margin 0.020621 mm. PPO changes actual targets by up to
0.000088201 rad in the nominal diagnostic, proving influence, **not improved success**.
The old/new perturbation batches use different seeds. Broader initial poses, model error
and real hardware remain unvalidated; next work should test new preregistered perturbations,
contact/model uncertainty and larger standing margins. Reward improvement alone is not control repair.

For a new standing-only five-episode repetition, after the shared runtime block:

```bash
"$PY" -m x2_recovery.reproduce evaluate --training-run "$INPUT" \
  --expected-checkpoint-sha256 "$POLICY_SHA" --seeds 26092550 26092551 26092552 26092553 26092554 \
  --output "$WS/results/evaluation/reproduction-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

This delegates to the original evaluator. The published [final validation runner][validation]
adds the separate physical-step envelope/impact checks; ordinary evaluator success alone
does not certify those checks. Demonstrations, ROS episodes and formal batches stay separate.

| Evidence | Authoritative entry |
| --- | --- |
| Current inputs, training and reward curve | [bundle][v6-inputs] · [manifest][v6-training] · [curve][v6-reward] |
| Current fixed five, paired comparison and physical audit | [five][v6-five] · [pairs][v6-pairs] · [diagnostic][v6-diagnostic] |
| Current build/ROS and delivery checks | [v6 summary][v6-summary] · [ROS][v6-ros] · [delivery review][delivery] |
| Historical failures, v5 and later audits | [pure PPO 0/5][ppo-failure] · [v5 inputs/results][v5-results] · [constraint/reward/pair review][old-review] |
| History and attribution | [publication inventory][inventory] · [reference notices][notices] · [baseline ROS][baseline-ros] |

Old absolute paths are execution provenance, not required user paths. Frozen files,
failed candidates and source snapshots remain intact. Search plans/journals are archived
with [round-trip hashes][archives]; they are unnecessary for controller loading/running.

## ROS 2 Integration and Validation

`Recovery` owns the simulator and full controller; `Telemetry` subscribes and logs status
and the measured left-knee position at 1 Hz. Default `reference_residual` requires an explicit
bundle. **Always pass the v6 hash**: the startup hash default retains the old v5 pin.
`scripted_baseline` is an explicit choice and never substitutes for failed policy loading.

| Interface | Behavior |
| --- | --- |
| `/x2/start_recovery` (`std_srvs/srv/Trigger`) | True means accepted; false while pending/running |
| `/x2/recovery_status` (`std_msgs/msg/String`) | IDLE, RUNNING, SUCCEEDED, FAILED; transitions and heartbeat |
| `/x2/joint_states` (`sensor_msgs/msg/JointState`) | Same simulator's 31 actual q/dq values and ROS acquisition stamp; effort empty |

Strict `evaluate.prepare` validates the full bundle, model and 149/17 probe once at startup,
without reset/step. Invalid inputs leave diagnostic FAILED, `controller_ready=false` and
a rejecting service; fix startup-only parameters and restart. One executor sends acceptance
before a later reset-only callback, then predicts deterministically and executes at most
one wrapped control step per timer callback. Phase uses simulation time; no wall-time
catch-up or target telemetry is substituted. Original environment success produces SUCCEEDED.
Terminal states stop stepping; a new request resets phase, targets and standing history.
Status QoS: reliable/transient-local depth 1. Joints: reliable/volatile depth 10.
ROS time stamps acquisition; MuJoCo time governs physics. `recovery_timeout_s` is a
monotonic, cooperative wall watchdog, not hard real-time: it checks around calls, records
overshoot and rejects late success. Policy `episode_timeout_s` must equal the frozen 20 s.

The first-install colcon command above builds both nodes and launch resources. For an
isolated fresh build, after defining `WS`/`PY`, choose a new directory and record it:

```bash
FRESH="$WS/runs/ros-build-$(date -u +%Y%m%dT%H%M%SZ)-$$"
test ! -e "$FRESH"
"$PY" /usr/bin/colcon --log-base "$FRESH/log" build --base-paths "$WS/src" \
  --packages-select x2_recovery --symlink-install \
  --build-base "$FRESH/build" --install-base "$FRESH/install"
printf 'Use OVERLAY=%s in both terminals\n' "$FRESH/install"
```

In **both terminals**, run the shared runtime block, replacing its `OVERLAY` assignment with
that printed install path (or keeping the initial `$WS/install`). Then, in both, select the same unused domain:

```bash
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Terminal 1 launches both processes. Terminal 2 waits for IDLE, observes topics and requests:

```bash
# Terminal 1
ros2 launch x2_recovery recovery.launch.py controller:=reference_residual \
  controller_run:="$INPUT" expected_checkpoint_sha256:="$POLICY_SHA" \
  seed:=221030 episode_timeout_s:=20.0 recovery_timeout_s:=180.0
```

```bash
# Terminal 2
ros2 pkg executables x2_recovery
timeout --signal=INT --kill-after=5s 70s ros2 topic echo /x2/recovery_status std_msgs/msg/String \
  --qos-reliability reliable --qos-durability transient_local &
timeout --signal=INT --kill-after=5s 70s ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState \
  --qos-reliability reliable --qos-durability volatile &
timeout --signal=INT --kill-after=5s 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

A running request returns false; a request after a terminal state starts a fresh episode.
Acceptance or CLI exit 0 is not success: observe terminal status and node reason/hold logs.
For a wall-timeout check, stop launch and repeat with `recovery_timeout_s:=3.0`; keep 20 s
simulation horizon. For the independent debug/failure path, stop policy launch and run:

```bash
ros2 launch x2_recovery recovery.launch.py controller:=scripted_baseline seed:=60
```

The [frozen v6 acceptance][v6-ros] used this VM's dependencies and a fresh build from `/tmp`:
142 policy/92 baseline cross-process checks, two same-process SUCCEEDED episodes at 6.616 s,
18 timestamp-matched 31-joint snapshots with zero q/dq error, and exact standalone/ROS
observation/action/state/reward/time equality. Busy, real wall timeout, wrong artifacts,
post-step exception/retry and both child processes' Ctrl+C cleanup passed. The [historical
236-test regression][regression] is separate from those real ROS episodes. Counts are
checks, not recoveries; these are dated published results, not automatically rerun tests.
The [delivery review][delivery] records this revision's actual commands and scope.
Neither ROS success nor these finite trials prove hardware safety or PPO superiority.

To run the existing regression suite after the shared runtime block:

```bash
MUJOCO_GL=osmesa "$PY" -m unittest discover -s "$WS/src/x2_recovery/test" -p 'test_*.py' -v
```

It includes synthetic training fixtures and bounded physical tests. It does not run the
separate opt-in cross-process ROS runner or repeat the published formal evaluations.

[upstream]: https://github.com/AgibotTech/agibot_x2_urdf/tree/60c5de582c523cd188f563819e62d34cfdc3d2d0
[v6-inputs]: results/control-repair/20260924T151819Z/training-run
[v6-config]: results/control-repair/20260924T151819Z/training-run/resolved_config.json
[v6-training]: results/control-repair/20260924T151819Z/training-run/manifest.json
[v6-reward]: results/control-repair/20260924T151819Z/training-run/training_reward.png
[v6-reward-check]: results/control-repair/20260924T151819Z/trained-policy-diagnostic/reward_comparison.json
[v6-protocol]: results/control-repair/20260924T151819Z/manifest.json
[v6-five]: results/control-repair/20260924T151819Z/fixed-five/episodes.csv
[v6-pairs]: results/control-repair/20260924T151819Z/held-out-paired/validation_summary.json
[v6-diagnostic]: results/control-repair/20260924T151819Z/trained-policy-diagnostic/summary.json
[v6-summary]: results/control-repair/20260924T151819Z/summary.json
[v6-ros]: results/control-repair/20260924T151819Z/ros-acceptance/summary.json
[v6-gif]: results/demonstrations/v6-recovery-20260925T060741Z/recovery-v6.gif
[v6-video]: results/demonstrations/v6-recovery-20260925T060741Z/recovery-v6.mp4
[demo]: results/demonstrations/v6-recovery-20260925T060741Z/manifest.json
[recorder]: results/demonstrations/v6-recovery-20260925T060741Z/record_demo.py
[v5-video]: results/evaluation/ppo-reference-residual-20260922T212457Z/demonstration/trained-recovery-reproduction.mp4
[v5-training]: results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run/manifest.json
[v5-results]: results/evaluation/ppo-reference-residual-20260922T212457Z
[ppo-failure]: results/evaluation/ppo-smoke-20260922T044430Z
[old-review]: results/controller-review/20260924T132353Z
[reward-audit]: results/controller-review/20260924T132353Z/reward-audit/summary.json
[totals]: results/evaluation/ppo-reference-residual-20260922T212457Z/analysis/experiment_totals.json
[contacts]: results/control-repair/20260924T151819Z/contact_transition_comparison.json
[archives]: results/control-repair/20260924T151819Z/validation/search-archives.json
[validation]: results/control-repair/20260924T151819Z/final_validation.py
[notices]: results/publication/20260923/THIRD_PARTY_NOTICES.md
[inventory]: results/publication/20260923/manifest.json
[installation]: results/ros-acceptance/20260923T125208Z-patched-v3/commands.json
[baseline-ros]: results/ros-acceptance/20260923T125208Z-patched-v3/summary.json
[regression]: results/control-repair/20260924T151819Z/validation/regression.log
[delivery]: results/delivery-review/20260925T065042Z/summary.json
