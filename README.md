# AgiBot X2 Ground Recovery

Native MuJoCo/Gymnasium supine recovery for AgiBot X2. The selected controller combines
an X2 reference motion with a trained PPO residual and succeeded in **5/5 fixed-supine
evaluations**. All five initial states and trajectories were identical. The reference
alone also succeeds: PPO changes the motion, but improved success rate is not established.
The earlier pure-PPO checkpoint scored 0/5. A later physical-step review found transient
joint-limit excess and landing impacts despite the valid final standing window. ROS 2 now
loads this complete hybrid controller explicitly; the default remains `scripted_baseline`.
This is a simulation demonstration with unresolved whole-episode constraint risks.

## Setup and Quick Start

Target: **Ubuntu 24.04 ARM64 Parallels**, system Python **3.12.3**, ROS 2 **Jazzy**.
Existing system prerequisites: `git`, `ca-certificates`, `python3`, `python3-venv`,
`python3-pip`, `python3-setuptools`, `python3-numpy`, `python3-pil`, `python3-matplotlib`,
`libosmesa6`, `python3-colcon-common-extensions`, `ros-jazzy-ros-base`,
`ros-jazzy-rmw-fastrtps-cpp`. Jazzy must provide `rclpy`, `rcl_interfaces`, `sensor_msgs`,
`std_msgs`, `std_srvs`, `launch`, `launch_ros`, `ament_index_python` and the ROS package/node/topic/service/launch CLIs.
The commands below install no system packages.

[requirements.txt](requirements.txt) pins NumPy 1.26.4, MuJoCo 3.13.0, Gymnasium 1.3.0,
SB3 2.9.0, Torch 2.8.0+cpu, Pillow 10.2.0, cffi 2.1.1 and matplotlib 3.6.3; it is not a
transitive lockfile. `--system-site-packages` inherits Ubuntu NumPy/Pillow/matplotlib
and build tools; sourcing Jazzy exposes ROS modules. cffi satisfies inherited PyNaCl.
Use compatible system Python; do not upgrade pins, use `sudo pip` or install PyPI `rclpy`.

Start a clean shell without an old overlay. Use a new checkout, or reuse an existing
installation. A **full clone** includes all policy inputs; source-only clones do not.
Change `WS` consistently if using another location. For an existing model, skip its
clone/checkout commands and verify the pinned revision and clean relevant files.

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
source "$WS/install/setup.bash"
cd /tmp
ros2 pkg prefix x2_recovery
ros2 pkg executables x2_recovery
"$PY" -c 'import sys, x2_recovery.env; print(sys.executable, x2_recovery.env.__file__)'
```

Use `"$PY"` for colcon: its shebang selects system Python. Installed resources and symlinks
must resolve into this build/checkout, without an old `PYTHONPATH`; model hashes are checked
offline. [Prior installation evidence][ros-commands] records system reuse and download retries.

### Load and run the published controller

The thin `reproduce` entry delegates to the existing validated loader and execution paths.
Keep the zip, manifest, resolved reference/configuration, progress, reload validation and
probe together. `check` only loads and checks saved-observation actions; it never resets,
steps or trains. `run` makes a verified writable input copy before one real episode.

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd /tmp
INPUT="$WS/results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run"
POLICY_SHA=11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -m x2_recovery.reproduce check --training-run "$INPUT" \
  --expected-checkpoint-sha256 "$POLICY_SHA" --output "$WS/runs/check-$RUN_ID"
"$PY" -m x2_recovery.reproduce run --training-run "$INPUT" \
  --expected-checkpoint-sha256 "$POLICY_SHA" --seed 221030 \
  --output "$WS/runs/replay-$RUN_ID"
```

All three subcommands require a new output path and refuse overlap with the frozen input.
The one-episode result is under `replay-<id>/training-input/development-221030/`;
`execution.json` records the real child exit code. Nothing installs or trains implicitly.

Inspect `status`, `success`, `reason`, `sim_duration_s`; exit 0 alone is not recovery success.
The [video][video] is a separate real reproduction; training/evaluation and ROS commands follow.

## Robot Model and Simulation

Model: **X2 Ultra v1.3.0**, from [AgibotTech/agibot_x2_urdf][upstream] at
`60c5de582c523cd188f563819e62d34cfdc3d2d0`. Under `X2_URDF-v1.3.0`, the reference
URDF is `x2_ultra.urdf`; `scene.xml` includes `x2_ultra.xml` and a flat infinite floor.
Assets remain external under their **Mulan PSL v2** license. Project package metadata
is **UNLICENSED**; no additional project license is granted.

[model.py](src/x2_recovery/x2_recovery/model.py) applies named, in-memory edits:

- Supply the URDF pelvis mass, COM and full inertia tensor omitted by the MJCF.
  Pelvis mass changes 5.031810659 → 3.523487 kg; total mass becomes 41.966473 kg.
- Intersect six wider MJCF joint ranges with the URDF: waist yaw, head yaw,
  both wrist pitches and both wrist rolls. Retain already narrower source ranges.
- Activate the waist-pitch upper soft limit 0.005 rad early to prevent settled-reset
  drift beyond the unchanged 0.314 rad bound. This is a project modeling choice.

The floating pelvis has 7 position coordinates/6 velocities; the model has `nq=38`,
`nv=37`, **31 hinged joints and 31 direct torque motors**. Soft joint-limit constraints
and clamped motor-effort bounds (0.6–118 N m, depending on joint) remain active; URDF speed ratings are
reported but not enforced as hardware speed clamps. Mapping uses discovered qpos,
DOF and control addresses, not assumed contiguous joint IDs. No base restraint,
external recovery force, gravity compensation or mass scaling is used.

Physics is 1 ms Euler integration with gravity −9.81 m/s², Newton solver, pyramidal
friction cone, 100 iterations and tolerance 1e-8. Original friction/contact parameters
and effort bounds are retained. Control targets update every 20 substeps (**50 Hz**);
PD is recomputed every substep. Mesh collisions use convex hulls, feet use twelve
5 mm spheres each, and fingers are not articulated. Soft contacts permit small
penetrations; collision proxies and simulated force measurements limit hardware transfer.
Full inertias/ranges and identity checks remain in the loader and [resolved configuration][config].

Experiments used 8 virtual CPUs, approximately 11 GiB RAM and 4 GiB swap; PPO ran on
CPU with 2 Torch compute threads, 1 interop thread and 1 compute thread per worker.
Virtual graphics do not imply GPU training. Headless operation needs no display;
OSMesa was used for offscreen rendering. Earlier virgl/passive-viewer failures remain
historical limitations; native X11 software rendering was separately validated.
Reference provenance includes HumanUP motion retargeting and local X2 dynamic/stance
refinement; [attribution and raw-motion licensing limits][notices] are retained.
This README was consolidated with Codex assistance; measured results are identified by
saved execution evidence. The two-page HRS task brief was checked locally, not redistributed.

## Environment Design

The selected path is `ControlledRecoveryEnv(X2RecoveryEnv(...), saved ControlConfig)`:
**`reference_residual` / `targets-v5` / `independentlegs17`**. The native environment
owns physics, reset, bounded PD and the independent success tracker.

Reset constructs a fresh face-up, spread-arm pose with 2 mm floor clearance, then
settles under zero control: up to 5 s to qualify, a continuous 0.5 s dwell and 1 s
unassisted hold. Accepted rest requires face-up torso/pelvis within 20°, linear
speed ≤0.01 m/s, angular speed ≤0.05 rad/s, hinge speed ≤0.02 rad/s, floor/self
penetration ≤1/0.1 mm and real support at 80–120% of body weight, including torso load.
The same state, residual velocity and nonzero MuJoCo time are handed to the episode;
settling is excluded from recovery duration. Selected reset perturbation is **zero**.
Optional per-joint uniform noise is limited to ±0.005 rad; there is no broad pose randomization.
Previous action, tracker/window anchors and controller history reset each episode;
adopted targets start from settled joints, references are rebuilt, and phase starts at zero.

The **149 float32 observations** use fixed scaling, with no VecNormalize. Let `R`
be pelvis body-to-world rotation and `W` robot weight. Values are not clipped to hide
physical limit violations; these are simulator states, not a validated hardware sensor set.

| Components | Count | Meaning / scaling |
| --- | ---: | --- |
| Joint positions and velocities | 31 + 31 | `(q-midpoint)/half_range`; `dq/5 rad/s`, actuator order |
| Gravity, linear and angular velocity | 3 + 3 + 3 | `R.T*[0,0,-1]`; pelvis-origin body-frame velocity / 1 m/s; body angular velocity / 2 rad/s |
| Pelvis height and support | 1 + 3 | Height / 0.6724955472220092 m; left/right vertical foot force and summed non-foot force norms / W |
| Previous physical action | 31 | Adopted normalized native action, not the 17 network coordinates |
| Standing memory | 1 + 1 + 9 | Hold / 2 s; valid-window flag; pelvis/feet horizontal drift rotated into pelvis frame, divided by 0.05/0.03/0.03 m |
| Adopted target and phase | 31 + 1 | Target normalized by joint midpoint/half-range; elapsed recovery / 20 s |
| **Total** | **149** | Native 117 plus 32 controller channels |

The policy action space is `Box(-1,1,(17,))`: six independent coordinates for each leg
(hip pitch/roll/yaw, knee, ankle pitch/roll), plus waist pitch and bilateral shoulder
pitch/roll/yaw and elbow coordinates. Roll/yaw mirror between arms; pitch/elbow share
sign. The map produces residuals for 21 joints; head, wrists and waist roll/yaw retain
reference control. **17 policy outputs do not mean 17 robot actuators.**

A 76-stage reference is linearly interpolated at 50 Hz, queried at elapsed time plus
one control interval; its final target is held after 3.2 s. Residual authority starts
at 2.25 s with a 0.2 s quintic ramp and remains active during standing. Full-scale
corrections are 0.03 rad at hips/knees, 0.02 at ankles/shoulders/elbows and 0.01 at waist
pitch. Reference plus mapped residual is joint-clipped, then target-rate-limited
(hip/knee/shoulder/elbow 3, ankle 2, waist/head/wrist 1.2 rad/s). Targets are converted
to native normalized actions and passed through the original `env.step()`. SB3 clips sampled Gaussian actions to the action
bounds before control; PPO retains the original samples/log probabilities for training.

PD uses `tau=clip(Kp*(target-q)-Kd*dq, effort_limits)` at 1 kHz: hip/knee gains
600/16.97056; ankle/waist 300/11.31371; shoulder/elbow 100/4.24264; head/wrist
10/0.424264 (N m/rad, N m s/rad). No per-step pose assignment or extra force is used.
Success terminates; 20 s simulation timeout truncates for bootstrap. Floor penetration
>30 mm, self penetration >15 mm, joint excess >0.05 rad or speed >30 rad/s terminate
as safety aborts. Invalid/numerical execution raises and requires reset. Watchdogs
bound reset/step calls to 45/5 wall seconds. Lying or non-foot support alone is allowed
while attempting recovery; standing qualification is stricter.

| Interface | Observation / action | Use |
| --- | --- | --- |
| Selected controlled environment | 149 / 17 | Reference plus PPO residual; versioned learning reward |
| Native `X2RecoveryEnv` | 117 / 31 | Full-range target actions; original potential reward; old smoke and ROS baseline |

Shared core remains in `model.py`, `reset.py`, `success.py`, `env.py`; user entry points
are `train.py`, `evaluate.py` and the two ROS nodes. `baseline.py` supplies reference
keyframe logic as well as the ROS script. `motion_search.py` and validation modules
retain development tools. `train.py` imports `model_audit.fingerprint`, and evaluation
snapshots include that module; it is part of the runtime identity.

## Rewards and Training

### Selected reward and PPO settings

The trained objective is **`reference-balance-v1`**, implemented in
[env.py](src/x2_recovery/x2_recovery/env.py). It replaces the native height/upright
potential reward; the native total remains diagnostic, while its torque-cost term is reused.
All densities below use post-control-step measurements times **actual executed dt**
(normally 0.02 s, possibly shorter at termination). Let `L,R,O` be the support channels,
`F=L+R`, `H=clip(height/h_ref,0,1)`, `G=clip((upright_dot-0.5)/0.5,0,1)`.
Head progress `p=clip((head_z-0.2043093023)/(1.2258631472-0.2043093023),0,1)` uses the
head-pitch body origin; `p_ref` is the saved linearly interpolated head-progress table.

| Term | Weighted contribution |
| --- | --- |
| Pose guide | `dt*(1-p_ref)*exp(-mean((q23-reference23)^2)/0.35^2)`; excludes head/wrists |
| Head tracking | `dt*2*exp(-((p-p_ref)/0.2)^2)*clip(F+O,0,1)` |
| Balance | `dt*6*H*G*B*N*A*V` (factors below) |
| Foot overload | `dt*(-0.5)*G*clip(F-1.2,0,3)` |
| Self penetration | `dt*(-0.5)*G*clip(self_penetration/0.003,0,1)` |
| Qualified standing | `dt*5*qualified`; current native predicate including window-drift checks |
| Torque cost | `-0.02*sum_substeps(mean31((applied_torque/effort_limit)^2)*physics_dt)` |
| Target change | `-0.01*mean31((adopted_target-previous_target)^2)`, once per transition |
| Success / safety | `+50` on recovery completion; `-2` on safety abort |

Balance factors are `B=clip(min(L,R)/0.2,0,1)`,
`N=exp(-(distance(F,[0.8,1.2])/0.4)^2)`, `A=exp(-(O/0.1)^2)` and
`V=1/(1+(COM_speed/0.3)^2+(torso_angular_speed/0.75)^2)`.
Reference terms guide the rise; bilateral support and low-speed factors favor balance;
costs discourage excessive torque, abrupt targets, overload and collision.
The balance costs vanish at torso tilt ≥60°, tracking can reward reference imitation,
and torque-square is not energy. This is not potential-based shaping or a policy-invariance
claim. Control-boundary reward qualification does not replace the 1 ms success detector.

| Setting | Selected value |
| --- | --- |
| Algorithm / network | SB3 PPO 2.9.0, CPU; separate 128×128 Tanh actor/value MLPs, orthogonal initialization |
| Parallel rollout | `n_envs=2`, `n_steps=256` each → 512 transitions/update; minibatch 64 |
| Optimization | Learning rate 0.0001; up to 5 epochs; target KL 0.03; Adam epsilon 1e-5, betas (0.9,0.999) |
| Returns / clipping | Gamma 0.999 per transition, GAE 0.95, policy clip 0.2, no value clip, normalized advantages |
| Loss / gradient | Entropy coefficient 0, value coefficient 0.5, gradient norm limit 0.5 |
| Exploration | gSDE, resample every 25 steps, initial log std −2.5 (noise-weight std, not marginal action std) |
| Preprocessing | Fixed environment scaling; no BC pretraining, VecNormalize or extra action noise |

The [selected training manifest][train-manifest] records **4096 new transitions**,
8 complete rollouts and **138 actual Adam steps**; SB3's counter of 21 epoch attempts
is different. This was a fresh run, not continuation. Actor/critic L2 changes were
0.291120/1.963901. `learn()` took 38.458 s (106.505 transitions/s including rollout,
reset, inference, optimization and logging). Its six completed stochastic episodes
ended in five joint-limit aborts and one timeout; deterministic inference later succeeded.
The [reward plot][reward] shows those six raw complete-episode returns against global
sampled transitions, before timeout bootstrap; fewer than 20 episodes means no trailing
20-episode mean. It is not a success curve and is not comparable to the old smoke reward.

[Experiment totals][totals] verify 26 discovery blocks: 450,560 new completed-rollout
transitions and 12,838 Adam steps across independent initializations and explicit resume
chains. They exclude the earlier 2,048-transition/30-Adam-step smoke and do not count
unpersisted interrupted partial rollouts. These totals are not the selected checkpoint's
training scale. A separate lower-noise branch continued 4,096 → 16,384 transitions
and was rejected after only 0.569 s of qualified hold; the selected checkpoint was not
continued. Re-running training does not guarantee the published checkpoint or outcome.

The [discounted-reward review][review-reward] uses complete real reward sequences and
`sum(gamma**t * r_t)` with one exponent per policy transition, including the short final
transition. Raw Monitor totals, observed discounted prefixes, truncated critic tails and
rollout GAE are distinct; no missing historical bootstrap is inferred from today's critic.

| Real trajectory under `reference-balance-v1` | Raw return | Observed discounted return |
| --- | ---: | ---: |
| Selected deterministic success | 82.867603 | 67.173051 |
| Zero-residual reference success | 83.085549 | 67.252247 |
| Continued-policy safety failure, same control configuration | 27.315238 | 23.403243 |
| Broader-residual 20 s timeout, different control configuration | 17.304419 | 15.196736; critic tail unavailable |
| Selected training episode 6, joint-limit abort | 191.958311 | Exact sequence unavailable; rigorous lower bound 72.413169 |

The last row's signed component totals and known terminal safety penalty prove its
lower bound exceeds the fast success return. They do **not** reconstruct its sequence:
`G >= 0.999**974*(194.0051577-2)-0.0468471`. That stochastic episode crossed several
actor updates, so this confirms a task/reward ranking mismatch for those trajectories,
not intentional exploitation or a performance claim about the final actor.

Two offline candidates reduce repeatable positive densities while keeping success +50,
safety −2 and every physical/task condition unchanged. `reference-balance-density10-v1`
scales pose/head/balance/standing by 0.1, with no added time cost. Its conservative 20 s
nonterminal positive-return bound is 17.704528; the same fast success rescored is 42.009844.
The historical failure's upper bound becomes 18.628065. These bounds cover repeated
posture/hold rewards; they do not guarantee all exploration outranks immediate failure.
A single local 4096-transition pilot transferred the original actor/log std, initialized a
fresh critic/Adam, and reused no rollout. It completed eight rollouts and 97 actual Adam
updates in about 123 s including diagnostics. Before/after deterministic development
recovery was 4.856/4.803 s, but the same pre-residual constraint violations remained;
stochastic training had zero successful completed episodes. The new objective and
checkpoint are **unadopted local experiments**, not replacements for the published inputs.
A separately [frozen new five][review-pilot-five] then achieved 5/5 original standing success
at 4.803 s, but 0/5 constraint-envelope and 0/5 admissible recoveries. All five trajectories
were again identical fixed-condition repeats; new weights and full candidate inputs remain local.

### New training, continuation and evaluation

This is an optional **new experiment**, not required to use the published policy.
In a new shell, use the selected configuration and explicit exploration settings;
plain `train`/`smoke` and CLI defaults do not select this final controller.

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd "$WS"
INPUT="$WS/results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
NEW_RUN="$WS/runs/reference-ppo-$RUN_ID"
"$PY" -m x2_recovery.train control-block --run-dir "$NEW_RUN" \
  --control-config "$INPUT/resolved_config.json" --total-timesteps 4096 \
  --max-wall-seconds 180 --seed 221900 --workers 2 --use-sde --sde-sample-freq 25 \
  --learning-rate 0.0001 --log-std-init -2.5 --target-kl 0.03 --n-epochs 5
"$PY" -m x2_recovery.train control-reload --run-dir "$NEW_RUN" --seed 222611 --max-wall-seconds 180
NEW_SHA="$(sha256sum "$NEW_RUN/policy_final.zip" | cut -d ' ' -f 1)"
"$PY" -m x2_recovery.evaluate --training-run "$NEW_RUN" \
  --expected-checkpoint-sha256 "$NEW_SHA" \
  --reload-validation "$NEW_RUN/development-222611/summary.json" \
  --seeds 221030 221031 221032 221033 221034 --deterministic \
  --output "$WS/results/evaluation/new-training-$RUN_ID"
```

Inspect the training/reload summaries before interpreting evaluation. The evaluator
requires a complete compatible reload, not a successful recovery; 0/5 is a valid result.
Missing checkpoints, budget stops or execution errors must not be treated as trained success.
For **continuation instead of fresh initialization**, use the same shell/configuration:

```bash
"$PY" -m x2_recovery.train control-block \
  --run-dir "$WS/runs/reference-ppo-resume-$(date -u +%Y%m%dT%H%M%SZ)-$$" \
  --resume-run "$INPUT" --control-config "$INPUT/resolved_config.json" \
  --total-timesteps 4096 --max-wall-seconds 180 --seed 221900 --workers 2 \
  --use-sde --sde-sample-freq 25 --learning-rate 0.0001 --log-std-init -2.5 \
  --target-kl 0.03 --n-epochs 5
```

This adds 4096 transitions, restoring weights, optimizer/counters and published global
RNG state. It begins fresh legal supine episodes, not a bitwise simulator/worker resume.
New training and evaluation directories must not exist; the CLIs refuse overwrite.

## Evaluation and Limitations

### Success and recorded outcomes

Success requires legal supine origin and **2.0 uninterrupted simulated seconds** meeting
all [success.py](src/x2_recovery/x2_recovery/success.py) predicates at every 1 ms sample.
Any failed predicate/drift clears the window; disconnected stable intervals do not add up.
Success at the exact timeout wins; invalid execution cannot qualify.

| Necessary predicate | Frozen threshold |
| --- | --- |
| Pelvis height / torso tilt | ≥0.90×0.6724955472220092 m; torso +Z within 15° of world up |
| Each foot / combined vertical load | ≥0.05 W each; combined 0.80–1.20 W |
| Other body support | Sum of individual non-foot ground-force norms ≤0.00001 W |
| Linear / angular speeds | Base and COM ≤0.10 m/s; base and torso ≤0.25 rad/s |
| Hinge speeds / horizontal drift | ≤0.50 rad/s; pelvis ≤0.05 m, each foot origin ≤0.03 m from window start |
| Floor / self penetration; joint excess | ≤1 mm / 0.1 mm; ≤0.1 mrad |
| Load-bearing contact gap | ≤2 micrometres; W=411.69110013 N |

The frozen [CSV][success-csv] and [summary][success-summary] report:

| Seed | Success | Recovery duration, including hold | Continuous hold | Peak pelvis height |
| --- | ---: | ---: | ---: | ---: |
| 221030 | 1 | 4.856 s | 2.000 s | 0.631539 m |
| 221031 | 1 | 4.856 s | 2.000 s | 0.631539 m |
| 221032 | 1 | 4.856 s | 2.000 s | 0.631539 m |
| 221033 | 1 | 4.856 s | 2.000 s | 0.631539 m |
| 221034 | 1 | 4.856 s | 2.000 s | 0.631539 m |

Each attempt executed 243 control calls / 4,856 physics steps, terminated successfully
without truncation, and excluded reset settling. The final hold spans 2.856–4.856 s;
**4.856 s includes the final two-second hold**. Peak height occurs airborne, not in
stable standing. Matched trained-policy [hold evidence][hold] measures pelvis
0.615588–0.617180 m, torso tilt ≤1.8674° and zero non-foot support in that final window.
Instantaneous support or peak height elsewhere does not establish sustained standing.

The [original pure-PPO evaluation][old-summary] was **0/5**: seeds 220930–220934 all
truncated at 20 s, peak pelvis 0.078813 m and zero qualified hold. Its small training
run produced early safety failures; deterministic targets then approached a low-height
state with persistent non-foot support and saturated waist effort. A low-height
control equilibrium and contact-constrained waist are hypotheses, not proven causes.
Useful separate diagnostics would freeze late actions or measure generalized constraint
forces; increasing torque limits is not justified by these observations.

In the [matched ablation][ablation], trained residual and reference alone both succeeded
(4.856 versus 4.887 s). Learned output changed actual targets by up to 0.005623 rad
across 130 transitions without parameter/optimizer updates. This proves control influence,
not success-rate improvement. The formal five repeats share identical initial states
and trajectories, so neither comparison establishes broad fallen-pose robustness.
The new [physical-step audit][review-audit] reproduces the **mixed controller itself**.
Adding the observer changed control-boundary observations, actions, qpos/qvel and rewards
by exactly zero. It copies forces immediately after native `mj_step`, before the existing
`mj_forward`, and separately labels the synchronized post-state recomputation. It adds
no physics step, forward solve or state write. Extrema are 1 kHz sample maxima; durations
sum 1 ms right-endpoint indicators, not continuous-time mathematical bounds.

| Whole recovery, excluding reset settling | Published mixed controller |
| --- | --- |
| Worst actual joint excess | Left ankle pitch: 0.0451852 rad below −0.803 rad at 2.206 s |
| That joint / longest consecutive excess | 0.650 s / 0.393 s; any joint exceeds standing tolerance for 1.064 s |
| Maximum joint speed / declared URDF speed ratio | 8.34714 rad/s / 0.699325; 31 velocity declarations available |
| Actual mapped motor effort / limit | Maximum directional ratio 1.0; targets stayed within nominal limits |
| Floor / self penetration | 9.04692 mm / 3.18148 mm |
| Landing vertical load | Actual integration solve: 3.95122 W; post-forward recomputation: 3.95815 W |
| Final continuous standing window | Original predicates pass; floor-penetration margin only 1.349 micrometres |

Of 1,064 over-tolerance samples, 1,063 precede residual activation; maximum excess,
speed and penetration also occur before activation. This primarily implicates the
reference prefix and its execution, not an inactive residual. Reset settling has zero
nominal-range excess. The separate [zero-residual audit][review-zero] uses the same inputs.

The review predeclared a conservative **whole-recovery** goal of ≤0.0001 rad joint excess,
legal directional effort and declared URDF speeds; this reuses the standing tolerance,
not a PDF or hardware safety allowance. Original standing success remains valid, but
this additional envelope fails. Six isolated reference-margin/timing candidates were
compared internally with/without the original actor (12 real episodes). None met the
combined standing, constraint and impact criteria. The 0.03 rad left-ankle-margin case
still stood but reduced excess only to 0.0440961 rad and increased peak load/self contact.
Other candidates lost success or worsened impact; [all outcomes][review-mitigation] remain.
No altered reference or checkpoint replaces the published system. Broad fallen-pose
robustness and real-robot deployment remain unvalidated.

The [registered paired batch][review-pairs] perturbs all 31 initial joint targets uniformly
within ±0.002 rad through the existing legal reset. The first 20 of 40 predeclared seeds
all passed reset-only screening before outcomes; all 20 handoffs were physically distinct.
Each A/B run independently reset to exactly matching qpos/qvel/observation and full MuJoCo
integration state. All 20 planned pairs completed: zero residual **2/20**, frozen residual
**3/20**; B-only success 1, A-only 0, ties 19. Among the two common successes, B−A recovery
time was +1 ms and −24 ms (mean −11.5 ms). All 40 failed the conservative whole-episode
envelope. This small development batch reveals substantial sensitivity and does not
establish residual superiority. No failed seed was replaced; a table-serialization bug
was repaired from complete saved physical records without rerunning any episode.

### Reproduce or audit without overwriting evidence

For a **new five-episode repetition of the published policy**, this independent block
uses the portable inputs, original checkpoint hash and a new output directory:

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd "$WS"
"$PY" -m x2_recovery.reproduce evaluate \
  --training-run "$WS/results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run" \
  --expected-checkpoint-sha256 11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64 \
  --seeds 221030 221031 221032 221033 221034 \
  --output "$WS/results/evaluation/reproduction-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

This is a new batch, not broader initial-state coverage. Alternatively, this independent
block audits existing records without new episodes, using a disposable full clone:

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
AUDIT="$(mktemp -d /tmp/x2-evidence.XXXXXX)"
git clone --no-hardlinks "$WS" "$AUDIT/repository"
git -C "$AUDIT/repository" checkout --detach f23a1c1ceb888756d94e8b6768bfcd7f1de2ca8e
cd "$AUDIT/repository"
gzip -dk results/evaluation/ppo-smoke-20260922T044430Z/trajectory.jsonl.gz
gzip -dk results/evaluation/ppo-reference-residual-20260922T212457Z/trajectory.jsonl.gz
"$PY" results/publication/20260923/verify_delivery.py
```

Run that historical inventory audit at its compatible pre-integration tree, as above;
new ROS source is intentionally different. The verifier checks the 94-file [publication inventory][inventory], gzip byte round trips,
both saved batches and saved-observation inference in separate processes, with reset,
step and optimization forbidden. It uses frozen source for the old 117/31 PPO and
current compatible source for the selected 149/17 policy. Do not load the old policy
through the new controller or bypass identity checks. Raw trajectories expand to about
188 MiB only in this disposable clone; originals, gzip files and manifests stay unchanged.
The inventory covers runtime/policy evidence, not the current README. Historical ROS
README hashes describe their declared base plus patches, not later documentation.

| Published evidence | Location |
| --- | --- |
| Selected policy, complete inputs, configuration, training and reward | [inputs][inputs] · [manifest][train-manifest] · [reward plot][reward] |
| Five successful episodes, full compressed trajectory and frozen source | [CSV][success-csv] · [summary][success-summary] · [batch directory][success-dir] |
| Original 0/5 policy and batch | [summary][old-summary] · [complete batch][old-dir] |
| Reference/residual comparison and experiment accounting | [ablation][ablation] · [hold][hold] · [mechanism][mechanism] · [totals][totals] |
| Publication/source identity and prior validation | [inventory][inventory] · [validation][publication-validation] · [ROS source identity][ros-identity] |
| Historical ROS acceptance and commands | [summary][ros-summary] · [commands][ros-commands] |
| New constraint/reward review, paired trials and ROS policy | [constraint audit][review-audit] · [reward][review-reward] · [paired][review-pairs] · [ROS][review-ros] |

`runs/`, `audit-output/` and bulk research logs are ignored local development data,
not required published inputs. Old absolute paths in evidence identify original
execution locations. Historical snapshots, failed attempts and manifests are preserved;
the old full README remains in Git history.

## ROS 2 Integration and Validation

`Recovery` owns the simulation; `Telemetry` logs status and the measured left knee at
1 Hz. The default is still `scripted_baseline`. Explicit `reference_residual` uses the
same complete `evaluate.prepare` loader as standalone inference: checkpoint, reference,
config/source/model identity, preprocessing and saved 149/17 actions are checked before
the service is offered. Loading constructs the model but does not reset or step.
Missing inputs, wrong hashes, missing reference or incompatible interfaces fail startup;
there is no fallback. Both modes expose the same 31 real joint measurements.

| Interface | Type / behavior |
| --- | --- |
| `/x2/start_recovery` | `std_srvs/srv/Trigger`: true means accepted; false while pending/running |
| `/x2/recovery_status` | `std_msgs/msg/String`: IDLE, RUNNING, SUCCEEDED, FAILED; changes plus heartbeat |
| `/x2/joint_states` | `sensor_msgs/msg/JointState`: 31 names, actual q/dq, ROS acquisition stamp; effort empty |

The exclusive executor sends the acceptance response before a later timer resets.
Each subsequent timer performs at most one control transition. Policy mode keeps the
full `ControlledRecoveryEnv`, saves its observation, predicts a deterministic CPU action,
and steps that wrapper; joint readback comes from the same underlying physical state.
Phase follows simulation time, never timer delays. No catch-up stepping is performed.
The original independent standing criterion alone controls SUCCEEDED. A finished
reference, high reward or accepted service request cannot establish recovery.

Terminal states stop sampling and stepping. A new request resets target/reference/phase
and controller history; recoverable exceptions produce FAILED and permit explicit retry.
Ctrl+C unwinds execution and closes resources. `episode_timeout_s=20` excludes settling;
policy mode requires the saved value because changing it would also change observation
phase. Test shorter failures with `recovery_timeout_s`, a monotonic wall watchdog including
reset, or with baseline mode. The watchdog cannot preempt a currently executing bounded
reset/step. Busy requests during reset can consequently take over one second to respond.

Joint stamps use ROS acquisition time, not MuJoCo time. No `/clock` is supplied;
`use_sim_time=true` is rejected. Status QoS is reliable/transient-local/keep-last-1,
and joints reliable/volatile/keep-last-10. Late subscribers receive retained status
while its publisher lives. This is not hard real-time control or hardware deployment.

In **each new terminal**, use this setup; choose an unused domain:

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
INPUT="$WS/results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run"
POLICY_SHA=11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64
cd /tmp
```

Terminal 1 launches both business nodes with the historical successful controller:

```bash
ros2 launch x2_recovery recovery.launch.py controller:=reference_residual \
  training_run:="$INPUT" expected_checkpoint_sha256:="$POLICY_SHA" \
  seed:=221030 episode_timeout_s:=20.0 recovery_timeout_s:=30.0
```

For the original failure/regression path, stop that launch and use
`ros2 launch x2_recovery recovery.launch.py controller:=scripted_baseline seed:=60`.
The installed ament package retains its standard resource marker, launch file and four
console entries; `evaluate` and `reproduce` are Python module entries.

Terminal 2 starts bounded observers and requests recovery once IDLE is visible:

```bash
timeout --signal=INT --kill-after=5s 70s ros2 topic echo /x2/recovery_status std_msgs/msg/String \
  --qos-reliability reliable --qos-durability transient_local &
timeout --signal=INT --kill-after=5s 70s ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState \
  --qos-reliability reliable --qos-durability volatile &
timeout --signal=INT --kill-after=5s 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

Repeat the service call during RUNNING to observe busy rejection, and after SUCCEEDED
or FAILED to request a fresh episode. For wall timeout, stop the launch and restart the
same policy command with `recovery_timeout_s:=3.0`; keep policy simulation timeout at 20 s.
Alternatively, stop manual processes and run the installed-package cross-process check:

```bash
ROS_CHECK="$(mktemp -d /tmp/x2-ros-check.XXXXXX)"
timeout --signal=INT --kill-after=15s 240s "$PY" \
  "$WS/src/x2_recovery/test/test_ros_integration.py" --output-dir "$ROS_CHECK/policy" \
  --training-run "$INPUT" --expected-checkpoint-sha256 "$POLICY_SHA" \
  --reference-trace "$WS/results/controller-review/20260924T132353Z/unobserved/control_trace.npz"
```

The [new ROS acceptance][review-ros] used a fresh build/install in the existing VM,
reusing the verified `.venv`, system ROS packages and pinned external model. It was
not a new OS/dependency installation. Original launch, read-only observation wrappers,
explicit fault injection and unit tests are recorded separately.

| Current acceptance scope | Actual result |
| --- | --- |
| Policy / baseline cross-process checks | 111 / 85 passed; both runner exit codes 0, all owned children exited |
| Real policy recovery and retry | 4.856 s, 243 control calls; original launch, observed retry and post-fault retry succeeded |
| Response before reset | Server send completed 14.89 ms before reset began |
| Actual JointState | 14 timestamp-matched q/dq snapshots, maximum error 0 |
| Standalone vs ROS / complete retry | All actions, observations, states, rewards and simulation times match exactly |
| Timeouts, startup errors, stepping fault, Ctrl+C | Expected FAILED/rejection/cleanup paths verified; no hidden fallback |

These are functional scenarios, not a new five-seed robustness experiment. SUCCEEDED
still carries the whole-episode risks measured above. Initial attempts exposed a bounded
DDS-discovery miss, reset-busy latency above one second, and a test hash parsed as a YAML
integer; the [attempt records][review-ros-attempts] preserve them. Busy functionality and
the one-second timing goal are reported separately; final baseline still missed that
optional latency goal. No production timeout or task-success threshold was relaxed.

The current regression suite includes calculator tests using synthetic data and real
MuJoCo environment tests. With the terminal setup above, use the required offscreen
backend (a default graphics backend can return blank images on this VM):

```bash
MUJOCO_GL=osmesa "$PY" -m unittest discover -s "$WS/src/x2_recovery/test" -v
```

The [2026-09-23 baseline acceptance][ros-summary] remains separate historical evidence:
85 cross-process checks and 183 regression tests, not 268 recovery attempts. Its earlier
[shutdown][ros-first] and [discovery][ros-second] fixes remain in the runner.

[upstream]: https://github.com/AgibotTech/agibot_x2_urdf/tree/60c5de582c523cd188f563819e62d34cfdc3d2d0
[inputs]: results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run
[config]: results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run/resolved_config.json
[train-manifest]: results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run/manifest.json
[reward]: results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run/training_reward.png
[video]: results/evaluation/ppo-reference-residual-20260922T212457Z/demonstration/trained-recovery-reproduction.mp4
[success-csv]: results/evaluation/ppo-reference-residual-20260922T212457Z/episodes.csv
[success-summary]: results/evaluation/ppo-reference-residual-20260922T212457Z/summary.json
[success-dir]: results/evaluation/ppo-reference-residual-20260922T212457Z
[old-summary]: results/evaluation/ppo-smoke-20260922T044430Z/summary.json
[old-dir]: results/evaluation/ppo-smoke-20260922T044430Z
[ablation]: results/evaluation/ppo-reference-residual-20260922T212457Z/analysis/network_ablation.json
[hold]: results/evaluation/ppo-reference-residual-20260922T212457Z/analysis/hold_comparison.json
[mechanism]: results/evaluation/ppo-reference-residual-20260922T212457Z/analysis/reference_mechanism.json
[totals]: results/evaluation/ppo-reference-residual-20260922T212457Z/analysis/experiment_totals.json
[inventory]: results/publication/20260923/manifest.json
[publication-validation]: results/publication/20260923/validation.json
[notices]: results/publication/20260923/THIRD_PARTY_NOTICES.md
[ros-summary]: results/ros-acceptance/20260923T125208Z-patched-v3/summary.json
[ros-commands]: results/ros-acceptance/20260923T125208Z-patched-v3/commands.json
[ros-identity]: results/ros-acceptance/20260923T125208Z-patched-v3/source-identity.json
[ros-first]: results/ros-acceptance/20260923T121422Z-release-71037f7/summary.json
[ros-second]: results/ros-acceptance/20260923T122443Z-patched-v2/summary.json
[review-audit]: results/controller-review/20260924T132353Z/published-policy/summary.json
[review-zero]: results/controller-review/20260924T132353Z/zero-residual-reference/summary.json
[review-mitigation]: results/controller-review/20260924T132353Z/mitigation/summary.json
[review-reward]: results/controller-review/20260924T132353Z/reward-audit/summary.json
[review-pairs]: results/controller-review/20260924T132353Z/paired/summary.json
[review-ros]: results/controller-review/20260924T132353Z/ros-acceptance/policy-r4/summary.json
[review-ros-attempts]: results/controller-review/20260924T132353Z/ros-acceptance/build-and-unit.json
[review-pilot-five]: results/controller-review/20260924T132353Z/reward-audit/pilot_five_summary.json
