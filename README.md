# AgiBot X2 Ground Recovery

Native MuJoCo/Gymnasium supine recovery for AgiBot X2. The current simulation controller
combines a repaired reference motion, bounded torso-to-ankle feedback and a trained PPO
residual. It passed **5/5 fixed-supine recoveries** and **20/20 newly registered small
perturbations**, including the original two-second standing criterion and the declared
whole-recovery joint, motor and speed envelope. The same reference and feedback with
**zero PPO residual also passed 20/20**; PPO benefit is not established.

The earlier controller's fixed 5/5 standing result remains valid, but its transient
joint-limit excess failed the additional envelope and its old perturbation result was
3/20 (reference alone 2/20). All historical evidence is retained. ROS runs the same
new frozen input bundle; `scripted_baseline` remains an explicit debug/fallback choice.
These are finite simulation results, not broad-pose robustness or hardware certification.

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

### Load and run the validated v6 controller

The recommended simulation controller is **reference motion + analytical torso/ankle feedback + trained PPO
residual**, with its complete [training input](results/control-repair/20260924T151819Z/training-run).
Its fixed-five and small-perturbation validation is reported below; loading alone is not recovery.
The thin `reproduce` entry uses the existing strict loader. Keep the checkpoint, manifest,
resolved configuration/reference, progress, reload validation and action probe together.
`check` loads and compares saved-observation actions without reset, stepping or training.
`run` copies verified inputs to a new writable directory before one real episode.

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd /tmp
INPUT="$WS/results/control-repair/20260924T151819Z/training-run"
POLICY_SHA=af70fe7ed8ae6c8b1a2ed1f73cd247b116789488120586fc1d8dc949fbca5540
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -m x2_recovery.reproduce check --training-run "$INPUT" \
  --expected-checkpoint-sha256 "$POLICY_SHA" --output "$WS/runs/check-$RUN_ID"
"$PY" -m x2_recovery.reproduce run --training-run "$INPUT" \
  --expected-checkpoint-sha256 "$POLICY_SHA" --seed 221030 \
  --output "$WS/runs/replay-$RUN_ID"
```

Each output must be new and disjoint from the input. The episode summary is under
`replay-<id>/training-input/development-221030/`; `execution.json` records the child
exit code. Inspect `success`, `reason` and `sim_duration_s`: exit 0 alone is not success.
Nothing installs or trains implicitly. The [existing video][video] depicts the historical
v5 controller in a separate real reproduction, not the new controller.

### Reproduce the historical v5 controller

The old [5/5 inputs][inputs] remain byte-identical, with checkpoint SHA
`11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64`.
New `env.py` changes its source/controller identity, so those inputs must run with the
matching frozen source. This explicit historical process uses a source-only archive;
it does not replace the current checkout or its installed package.

```bash
set -e
set -o pipefail
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
cd /tmp
HISTORY_COMMIT=f542ec6414ae1366bc17868c0f43de80c580c8fc
HISTORY="$(mktemp -d "${TMPDIR:-/tmp}/x2-v5-XXXXXX")"
git -C "$WS" archive "$HISTORY_COMMIT" src/x2_recovery/x2_recovery | tar -x -C "$HISTORY"
INPUT="$WS/results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run"
POLICY_SHA=11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
PYTHONPATH="$HISTORY/src/x2_recovery" "$PY" -m x2_recovery.reproduce check \
  --training-run "$INPUT" --expected-checkpoint-sha256 "$POLICY_SHA" \
  --output "$WS/runs/history-v5-check-$RUN_ID"
PYTHONPATH="$HISTORY/src/x2_recovery" "$PY" -m x2_recovery.reproduce run \
  --training-run "$INPUT" --expected-checkpoint-sha256 "$POLICY_SHA" --seed 221030 \
  --output "$WS/runs/history-v5-replay-$RUN_ID"
```

Only these historical child processes receive that `PYTHONPATH`; current v6 commands use
the current installation. Old manifests/hashes are not rewritten to claim compatibility.

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

The current controller uses `ControlledRecoveryEnv(X2RecoveryEnv(...), ControlConfig)`:
**`reference_residual` / `targets-v6` / `independentlegs17`**. The native environment
still owns physics, legal reset, bounded PD and the unchanged independent success tracker.

Reset starts face-up with spread arms and 2 mm floor clearance, then settles under zero
control: up to 5 s to qualify, a continuous 0.5 s dwell and 1 s unassisted hold. Accepted
rest requires torso/pelvis within 20° of face-up, linear/angular/hinge speeds at most
0.01 m/s, 0.05 rad/s and 0.02 rad/s, floor/self penetration at most 1/0.1 mm, and support
at 80–120% of body weight including torso contact. The actual settled state, residual
velocity and MuJoCo time carry into the episode; settling is excluded from recovery time.
Nominal reset perturbation is **zero**. Optional per-joint uniform noise is bounded by
±0.005 rad; this is not broad fallen-pose randomization. Reset clears previous action,
tracker/window anchors, adopted target and reference history; phase restarts at zero.

The **149 float32 observations** use fixed scaling, with no VecNormalize. Let `R` be
pelvis body-to-world rotation and `W` robot weight. Values are not clipped to conceal
violations. These simulator measurements are not a validated hardware sensor pipeline.

| Components | Count | Meaning / scaling |
| --- | ---: | --- |
| Joint positions and velocities | 31 + 31 | `(q-midpoint)/half_range`; `dq/5 rad/s`, actuator order |
| Gravity, linear and angular velocity | 3 + 3 + 3 | `R.T*[0,0,-1]`; pelvis-origin body-frame velocity / 1 m/s; body angular velocity / 2 rad/s |
| Pelvis height and support | 1 + 3 | Height / 0.6724955472220092 m; left/right vertical foot force and summed non-foot force norms / W |
| Previous physical action | 31 | Adopted normalized native action, not network coordinates |
| Standing memory | 1 + 1 + 9 | Hold / 2 s; valid-window flag; pelvis/feet horizontal drift in pelvis frame, divided by 0.05/0.03/0.03 m |
| Adopted target and phase | 31 + 1 | Target normalized by joint midpoint/half-range; elapsed recovery / 20 s |
| **Total** | **149** | Native 117 plus 32 controller channels |

The retained `Box(-1,1,(17,))` policy interface has six coordinates per leg (hip
pitch/roll/yaw, knee, ankle pitch/roll), plus waist pitch and four bilateral arm synergies.
The current PPO has **physical authority only at the four ankle pitch/roll joints**:
action columns 4, 5, 10 and 11, each at most ±0.002 rad. Other mapped residual scales are
zero; their reference control remains active. There are still **31 physical actuators**.

The [saved controller](results/control-repair/20260924T151819Z/training-run/resolved_config.json)
contains 158 linearly interpolated reference knots with revised support timing, a waist
reference pulse and a new standing target. It samples elapsed time plus one 20 ms control
interval and holds the final target after 5.399296 s. The separate saved head-progress
reward table reaches its endpoint at 8.199296 s; it is not a recovery-success timer.

Built-in analytical feedback reconstructs torso gravity and angular velocity from pelvis
observations and the waist yaw/pitch/roll chain. With sagittal torso tilt `theta` and
its rate, it adds equal ankle-pitch offsets `clip(0.3*theta + 0.03*theta_dot, ±0.08 rad)`.
It preserves the validated float32 action path and uses a quintic gate starting at 3.5 s
with a 0.2 s ramp. This feedback is **not PPO**. The independently gated PPO residual
starts at 5.5 s with another 0.2 s quintic ramp. Both phases derive from actual simulation
time; feedback adds no hidden filter or controller memory.

Reference + gated analytical feedback + gated mapped PPO residual is joint-clipped,
then target-rate-limited: hip/knee/shoulder/elbow 3, ankle 2 and waist/head/wrist 1.2 rad/s.
The adopted target becomes a native normalized action for the original `env.step()`.
SB3 clips sampled Gaussian actions before execution but retains original samples and log
probabilities for PPO. Legal targets alone do not guarantee actual joints remain in range.

At 1 kHz, PD uses `tau=clip(Kp*(target-q)-Kd*dq, effort_limits)`: hip/knee gains
600/16.97056; ankle/waist 300/11.31371; shoulder/elbow 100/4.24264; head/wrist
10/0.424264 (N m/rad, N m s/rad). There are no pose assignments or added external forces.
Success terminates; the original 20 s simulation timeout truncates. Floor penetration
>30 mm, self penetration >15 mm, joint excess >0.05 rad or speed >30 rad/s terminate as
safety aborts. Numerical errors require reset; reset/step watchdogs remain 45/5 wall seconds.
Non-foot support is allowed during recovery; final standing qualification is stricter.

| Version | Observation / action | Control and scope |
| --- | --- | --- |
| Current v6 controller | 149 / 17 | Revised reference + analytical feedback + four-ankle PPO residual; density10 reward |
| Historical v5 checkpoint | 149 / 17 | Original 76-stage reference + 21-joint mapped PPO permissions; original balance reward; frozen source required |
| Native environment | 117 / 31 | Direct target interface, original potential reward; historical smoke and ROS baseline |

Shared core remains in `model.py`, `reset.py`, `success.py`, `env.py`; `train.py`,
`evaluate.py` and the two ROS nodes are user entry points. `baseline.py` supplies
reference keyframes as well as the baseline script. Development tools remain separate;
`model_audit.fingerprint` is imported by training and included in source identity.

## Rewards and Training

### Current objective and measured training

The new **`reference-balance-density10-v1`** objective reduces four repeatable positive
terms of historical `reference-balance-v1` to 10%. Negative costs, success +50, safety −2,
physics and success conditions are unchanged. This is a new objective, not potential-based
shaping. The native reward remains diagnostic; its integrated torque cost is reused.

Densities use post-control-step measurements times **actual executed dt** (normally
0.02 s; the trained nominal diagnostic ends with a 0.016 s step). Let `L,R,O` be support
channels, `F=L+R`, `H=clip(height/h_ref,0,1)`, `G=clip((upright_dot-0.5)/0.5,0,1)`.
Head progress is `p=clip((head_z-0.2043093023)/(1.2258631472-0.2043093023),0,1)`;
`p_ref` comes from the saved linear head-progress table.

| Term | Current weighted contribution |
| --- | --- |
| Pose guide | `dt*0.1*(1-p_ref)*exp(-mean((q23-reference23)^2)/0.35^2)`; excludes head/wrists |
| Head tracking | `dt*0.2*exp(-((p-p_ref)/0.2)^2)*clip(F+O,0,1)` |
| Balance | `dt*0.6*H*G*B*N*A*V` |
| Qualified standing | `dt*0.5*qualified`; native predicate including drift checks |
| Foot overload | `dt*(-0.5)*G*clip(F-1.2,0,3)` |
| Self penetration | `dt*(-0.5)*G*clip(self_penetration/0.003,0,1)` |
| Torque cost | `-0.02*sum_substeps(mean31((applied_torque/effort_limit)^2)*physics_dt)` |
| Target change | `-0.01*mean31((adopted_target-previous_target)^2)`, once per transition |
| Success / safety | `+50` on completion; `-2` on safety abort |

Balance factors are `B=clip(min(L,R)/0.2,0,1)`,
`N=exp(-(distance(F,[0.8,1.2])/0.4)^2)`, `A=exp(-(O/0.1)^2)` and
`V=1/(1+(COM_speed/0.3)^2+(torso_angular_speed/0.75)^2)`.
Reference tracking encourages the rise; support and low speed favor balance. Torque-square
is not energy, balance costs vanish at torso tilt ≥60°, and imitation remains rewarding.
The 50 Hz reward does not replace independent success checks at every 1 ms physics step.

| Setting | Current trained controller |
| --- | --- |
| Algorithm / network | SB3 PPO 2.9.0, CPU; separate 128×128 Tanh actor/value MLPs, orthogonal initialization |
| Rollout | 2 environments × 256 steps = 512 transitions; minibatch 64 |
| Optimization | Learning rate 0.0001; up to 5 epochs; target KL 0.03; Adam epsilon 1e-5, betas (0.9,0.999) |
| Returns / clipping | Gamma 0.999 per policy transition, GAE 0.95, policy clip 0.2, no value clip, normalized advantages |
| Loss / gradient | Entropy coefficient 0, value coefficient 0.5, gradient norm limit 0.5 |
| Exploration | Diagonal Gaussian; initial log std −2.3; no gSDE or extra action noise |
| Initialization / preprocessing | Fresh actor, critic, log std and Adam; no weight migration, old rollout reuse or VecNormalize |

The [new training manifest](results/control-repair/20260924T151819Z/training-run/manifest.json)
records **4096 transitions, 8 complete rollouts and 174 actual Adam updates**; SB3's
25 epoch-attempt counter is different. Total wall time was 37.08 s, including 33.68 s
in `learn()`. All 12 completed stochastic episodes met the original standing criterion;
these are training episodes, not an independent robustness test. The
[reward plot](results/control-repair/20260924T151819Z/training-run/training_reward.png)
shows raw completed-episode returns versus global sampled transitions, before timeout
bootstrap; fewer than 20 episodes means no trailing 20-episode mean. It is not evidence
that PPO improves success over the reference controller.

The previous [v5 training manifest][train-manifest] remains a separate fresh 4096-transition
run with 138 Adam updates and gSDE. [Earlier experiment totals][totals] describe 450,560
transitions across 26 discovery blocks and explicit resume chains, not continuous training
of either selected checkpoint. The earlier density10 actor-transfer pilot and its
[5/5 standing, 0/5 constraint results][review-pilot-five] remain an unadopted experiment.

### Discounted reward audit

`G=sum(gamma**t*r_t)` uses one exponent per **policy transition**, even for a short final
step. Raw Monitor returns, observed discounted prefixes, SB3 timeout bootstrap and GAE
are distinct. No post-terminal reward is added; an unavailable historical bootstrap is
not replaced with today's critic estimate.

The [historical audit][review-reward] found one stochastic v5 training failure with raw
return 191.958311. Its exact reward sequence was not saved, but signed component totals
prove `G >= 72.413169`, above the old deterministic success's measured 67.173051.
This establishes a trajectory-level reward/task mismatch, not intentional exploitation
or a claim about the final actor's distribution. Reduced positive densities address this
ranking risk without an added time cost or inflated terminal reward.

| Same new reference and analytical feedback, nominal seed 221030 | Raw density10 return | Discounted density10 return |
| --- | ---: | ---: |
| Zero PPO | 53.859569 | 39.037416 |
| Trained PPO residual | 53.859579 | 39.037423 |

Both [real diagnostics](results/control-repair/20260924T151819Z/trained-policy-diagnostic/reward_comparison.json)
complete at 6.616 s. Their very small return difference does not establish PPO benefit.
For this objective, repeatable positive density is at most 1.4/s: the conservative 20 s
failure-prefix bound is 17.704528 and the latest-success bound including +50 is 36.107703,
both below the measured nominal success. These are theoretical bounds, not physical
trajectories or a guarantee that every exploration path beats immediate safety failure.
Changing reward does not itself repair a physical constraint violation.

### New training, continuation and evaluation

The following optional command starts a **fresh experiment**, preserving the frozen input.
Re-training is distinct from replay and need not reproduce its checkpoint or result.
Plain `train`/`smoke` and CLI defaults do not select this controller.

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd "$WS"
INPUT="$WS/results/control-repair/20260924T151819Z/training-run"
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

Inspect training/reload results before evaluation. A complete compatible reload is
required, but 0/5 is a legitimate evaluation result; exit status alone is insufficient.
For genuine continuation of the published v6 controller, use the same settings and
explicitly restore its optimizer/counters and global RNG state:

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd /tmp
INPUT="$WS/results/control-repair/20260924T151819Z/training-run"
NEXT_RUN="$WS/runs/v6-resume-$(date -u +%Y%m%dT%H%M%SZ)-$$"
"$PY" -m x2_recovery.train control-block --run-dir "$NEXT_RUN" \
  --resume-run "$INPUT" --control-config "$INPUT/resolved_config.json" \
  --total-timesteps 4096 --max-wall-seconds 1200 --seed 26092490 --workers 2 \
  --learning-rate 0.0001 --log-std-init -2.3 --target-kl 0.03 --n-epochs 5
```

Continuation starts fresh legal supine episodes, not a bitwise worker/simulator resume.
All commands require new output directories; none overwrites published evidence.

## Evaluation and Limitations

Success still requires a legal supine origin and **2.0 uninterrupted simulated seconds**
meeting every [success.py](src/x2_recovery/x2_recovery/success.py) predicate at each 1 ms
sample. A failed predicate resets the window; disconnected intervals do not add up.
Recovery duration excludes reset settling and includes the final two-second hold.

| Necessary standing predicate | Unchanged threshold |
| --- | --- |
| Pelvis height / torso tilt | ≥0.90×0.6724955472220092 m; torso within 15° of upright |
| Each foot / combined vertical load | ≥0.05 W each; combined 0.80–1.20 W, W=411.69110013 N |
| Other body support | Sum of non-foot ground-force norms ≤0.00001 W |
| Linear / angular speeds | Base and COM ≤0.10 m/s; base and torso ≤0.25 rad/s |
| Joint speed / horizontal drift | ≤0.50 rad/s; pelvis ≤0.05 m, each foot ≤0.03 m |
| Floor / self penetration / joint excess | ≤1 mm / 0.1 mm / 0.0001 rad |
| Load-bearing contact gap | ≤2 micrometres |

The additional [predeclared envelope and comparison protocol](results/control-repair/20260924T151819Z/manifest.json)
requires every recovery physics sample to have nominal joint excess ≤0.0001 rad,
actual direction-normalized motor effort ≤1 and velocity ≤its sourced URDF rating.
The 0.0001 rad value is a conservative internal target borrowed from standing, not a
PDF requirement or hardware tolerance. `admissible_recovery` additionally requires a
complete standing success and all five impact comparisons: vertical ground load,
sum of contact-force norms, floor/self penetration and maximum joint speed must not
exceed the exact historical nominal values. Early termination or remaining supine cannot pass.

### Frozen results and limitations

| Version and protocol | Standing | Whole-recovery envelope | Standing + envelope + impact |
| --- | ---: | ---: | ---: |
| Historical pure PPO, fixed five | 0/5 | Not this protocol | 0/5 |
| Historical v5 reference + PPO, fixed five | 5/5 | Fails in matching later 1 kHz diagnostic | Not admissible; original five were not rerun |
| Historical ±0.002 rad development pairs, reference / PPO | 2/20 / 3/20 | 0/20 / 0/20 | 0/20 / 0/20 |
| New v6 reference + PPO, fixed five | 5/5 | 5/5 | 5/5 |
| New held-out pairs, reference+feedback / plus PPO | 20/20 / 20/20 | 20/20 / 20/20 | 20/20 / 20/20 |

The [new fixed-five CSV](results/control-repair/20260924T151819Z/fixed-five/episodes.csv)
uses seeds 26092550–26092554: all terminate successfully at **6.616 s**, including the
last 2.000 s, with 331 control transitions and 6,616 physics steps. Initial states and
trajectories are identical, so this is repeatability, not five fallen poses. Peak pelvis
height 0.627942 m is a transient maximum, not the stable standing height.

The [new paired batch](results/control-repair/20260924T151819Z/held-out-paired/validation_summary.json)
selected the first 20 legal resets from the predeclared 26092500–26092539 sequence,
before observing any controller outcome. All first 20 were valid and physically distinct;
all independently reset A/B handoffs matched. Each joint's reset target perturbation is
uniform within ±0.002 rad through the original reset. All 40 episodes are retained.
There were 0 A-only wins, 0 B-only wins and 20 ties; all common-success time differences
were 0. Both groups used the **same reference and analytical feedback**; only PPO differed.
This exceeds the preregistered 16/20 standing and all-constraints target in this batch.
It does not establish superiority over the reference, a population success probability,
or performance outside this small perturbation range. Across all 40 new recoveries,
maximum speed was 3.94160 rad/s, floor/self penetration 3.49847/1.67070 mm and ground
vertical load 853.263 N, with zero sampled joint excess. Final standing margins remain
limited: the smallest joint-speed margin was 0.000730 rad/s and floor-depth margin
0.020621 mm. Old and new batches use different
seeds, so their percentages are not a controlled estimate of PPO improvement.

| Nominal whole recovery, reset settling excluded | Historical mixed v5 | New trained v6 |
| --- | ---: | ---: |
| Maximum actual joint excess | Left ankle pitch 0.0451852 rad at 2.206 s | 0 rad across all 31 joints |
| Left ankle excess / longest consecutive excess | 0.650 s / 0.393 s | 0 s / 0 s |
| Maximum joint speed | 8.34714 rad/s | 3.83110 rad/s |
| Maximum declared velocity ratio / motor effort ratio | 0.699325 / 1.0 | 0.383487 / 1.0 |
| Floor / self penetration | 9.04692 / 3.18148 mm | 3.36709 / 1.66294 mm |
| Ground vertical / summed contact-norm peak | 1626.681 / 1845.448 N | 829.084 / 1134.171 N |
| Recovery duration including unchanged hold | 4.856 s | 6.616 s |

These are **1 kHz sampled maxima**, not continuous-time mathematical bounds. Positions
and velocities are post-integration; contacts/forces are copied from the integration
solve before the existing forward recomputation. Post-forward estimates are stored
separately and never substituted in the comparison. Duration sums 1 ms sample indicators.
The new controller is slower, but completes the full episode; no safety threshold,
physical model, torque bound or success criterion was relaxed. The 829 N figure is the
whole-body ground-load maximum, not a separately isolated foot-landing impact. Motor saturation still
occurs, and soft-contact penetrations remain. This is not a proof of mechanical safety.

The repair changes the coupled prefix, support targets and terminal stance, rather than
asking an inactive network to fix earlier motion. The old 1,063/1,064 over-tolerance
samples preceded residual activation. Development changed arm-support targets and coordinated hip/knee/ankle transitions,
used a flat-foot terminal stance, and added the
3.05–3.45 s waist-target pulse and 3.5 s bounded ankle feedback described above.
Contact/load traces establish changed support transitions; they do not identify a
single causal joint. The 829 N peak involves a mixed-contact configuration, not isolated
arm or foot loading. Under the same diagnostic rule (last zero-foot-load return followed
by 100 ms), all-body vertical load peaks fall from 1626.681 to 498.222 N. Integration
zero-ground-load samples total 79 ms, longest 12 ms (old: 348/215 ms); zero load does not
prove complete geometric flight. See the [contact comparison](results/control-repair/20260924T151819Z/contact_transition_comparison.json). The
[search record](results/control-repair/20260924T151819Z/search_summary.json) preserves
failed candidates, calibration runs and development data. Candidate counts are not
independent initial states. IK geometry alone was not accepted as physical validation.
The new search plans and journals are stored as byte-identical gzip archives; raw
originals stay local. Their [round-trip hashes](results/control-repair/20260924T151819Z/validation/search-archives.json)
are recorded. Before replaying historical search commands, restore just these files:
`find "$WS/results/control-repair/20260924T151819Z" -type f \( -name plan.json.gz -o -name results.jsonl.gz \) -exec gzip -dk {} +`.
This is unnecessary for normal loading, training, evaluation or ROS: their complete
input bundle is directly available without extraction.

The trained residual changes actual targets by up to 0.000088201 rad in the nominal
run; state differences begin after its gate activates. That proves control influence,
not a success-rate benefit. Training, analytical feedback and reference search are
separate sources of behavior. Broader perturbations, alternate fallen orientations,
model error and hardware remain unvalidated.

### Reproduction and evidence

For a new five-episode repetition, use a fresh output directory and the installed package:

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd /tmp
"$PY" -m x2_recovery.reproduce evaluate \
  --training-run "$WS/results/control-repair/20260924T151819Z/training-run" \
  --expected-checkpoint-sha256 af70fe7ed8ae6c8b1a2ed1f73cd247b116789488120586fc1d8dc949fbca5540 \
  --seeds 26092550 26092551 26092552 26092553 26092554 \
  --output "$WS/results/evaluation/reproduction-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

This entry preserves the original standing evaluation; the published
[final validation runner](results/control-repair/20260924T151819Z/final_validation.py)
adds the read-only physical-step envelope and impact summaries. Its `formal` and `paired`
commands require explicit protocol, input, checksum and new output paths. The original
five, new five, paired development/held-out batches and ROS episodes remain separate.
Historical full publication audits belong to their declared historical tree; do not
rewrite old hashes to accept new source. The Quick Start provides an isolated v5 load.

| Evidence | Location |
| --- | --- |
| New complete input, reference, training and reward | [bundle](results/control-repair/20260924T151819Z/training-run) · [control](results/control-repair/20260924T151819Z/final_control_config.json) · [training](results/control-repair/20260924T151819Z/training-run/manifest.json) |
| New physical-step trace and joint constraints | [diagnostic](results/control-repair/20260924T151819Z/trained-policy-diagnostic) · [joint CSV](results/control-repair/20260924T151819Z/trained-policy-diagnostic/joint_constraints.csv) |
| New fixed five and held-out pairs | [five](results/control-repair/20260924T151819Z/fixed-five/validation_summary.json) · [pairs](results/control-repair/20260924T151819Z/held-out-paired/paired_results.csv) |
| New source/build/ROS delivery | [review](results/control-repair/20260924T151819Z/summary.json) · [ROS](results/control-repair/20260924T151819Z/ros-acceptance/summary.json) |
| Historical successful/failed evaluations | [v5 inputs][inputs] · [5/5 CSV][success-csv] · [0/5 batch][old-dir] · [old real reproduction video][video] |
| Historical constraint/reward/pair audits | [constraints][review-audit] · [reward][review-reward] · [pairs][review-pairs] |
| Historical publication and ROS | [inventory][inventory] · [2026-09-23 baseline][ros-summary] · [earlier policy acceptance][deployment-summary] |

Old absolute paths identify execution provenance, not recipient working directories.
Frozen inputs, failed experiments and historical snapshots remain unchanged. Model
and contact simplifications in the simulation section still apply after ROS integration.

## ROS 2 Integration and Validation

`Recovery` owns one simulator and the complete controller; `Telemetry` logs its measured
left knee/status at 1 Hz. Default mode `reference_residual` requires an explicit input
bundle. Always provide its checkpoint hash as below; the startup hash default retains
the historical v5 pin. `scripted_baseline` is an explicit debug/fallback selection and
never substitutes automatically for a failed policy load.

| Interface | Behavior |
| --- | --- |
| `/x2/start_recovery`, `std_srvs/srv/Trigger` | True means accepted; pending/running requests return false |
| `/x2/recovery_status`, `std_msgs/msg/String` | IDLE, RUNNING, SUCCEEDED, FAILED; changes plus heartbeat |
| `/x2/joint_states`, `sensor_msgs/msg/JointState` | Same simulator's 31 actual q/dq values, ROS acquisition stamp; effort empty |

The strict `evaluate.prepare` loader checks bundle/source/model identities and saved
149/17 action consistency once at startup, without reset or step. Missing or incompatible
inputs produce observable FAILED, `controller_ready=false` and a rejecting service.
Correct startup-only parameters and restart; no request retries loading.
The single executor sends acceptance before a later reset-only callback. Each subsequent
callback predicts deterministically from the current wrapped observation and performs
at most one wrapped control step. Joint data come from that wrapper's physical environment.
Phase follows simulated time, never timer delay; no catch-up stepping changes control.
Original environment success alone produces SUCCEEDED. Terminal episodes stop advancing
until an explicit request resets phase, targets and standing history.

Status QoS is reliable/transient-local depth 1; joints are reliable/volatile depth 10.
ROS stamps are acquisition times, MuJoCo time governs dynamics, and a monotonic wall
clock governs `recovery_timeout_s`. Policy `episode_timeout_s` must equal the frozen
20 s configuration. Deadline checks around reset/predict/step are cooperative: they
cannot interrupt a Python/C call; overshoot is reported and late success is rejected.

Build in a **new** directory using the existing verified Python, Jazzy and model:

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export FRESH="$WS/runs/ros-deployment"
test ! -e "$FRESH"
source /opt/ros/jazzy/setup.bash
"$WS/.venv/bin/python" /usr/bin/colcon --log-base "$FRESH/log" build \
  --base-paths "$WS/src" --packages-select x2_recovery --symlink-install \
  --build-base "$FRESH/build" --install-base "$FRESH/install"
source "$FRESH/install/setup.bash"
cd /tmp
ros2 pkg prefix x2_recovery
ros2 pkg executables x2_recovery
```

In **each terminal**, source that same fresh installation and choose an unused domain:

```bash
set -e
export WS="$HOME/RL-AgiBot-X2" PYTHONNOUSERSITE=1
export PY="$WS/.venv/bin/python" FRESH="$WS/runs/ros-deployment"
export X2_ASSET_REPO="$HOME/.cache/hrs-x2-recovery/agibot_x2_urdf"
source /opt/ros/jazzy/setup.bash
source "$FRESH/install/setup.bash"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
INPUT="$WS/results/control-repair/20260924T151819Z/training-run"
POLICY_SHA=af70fe7ed8ae6c8b1a2ed1f73cd247b116789488120586fc1d8dc949fbca5540
cd /tmp
```

Terminal 1 starts the same new frozen controller used in standalone evaluation:

```bash
ros2 launch x2_recovery recovery.launch.py controller:=reference_residual \
  controller_run:="$INPUT" expected_checkpoint_sha256:="$POLICY_SHA" \
  seed:=221030 episode_timeout_s:=20.0 recovery_timeout_s:=180.0
```

Terminal 2 observes and requests once IDLE is visible:

```bash
timeout --signal=INT --kill-after=5s 70s ros2 topic echo /x2/recovery_status std_msgs/msg/String \
  --qos-reliability reliable --qos-durability transient_local &
timeout --signal=INT --kill-after=5s 70s ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState \
  --qos-reliability reliable --qos-durability volatile &
timeout --signal=INT --kill-after=5s 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

Repeat the service request during RUNNING for busy rejection or after a terminal state
for a fresh episode. For a wall-timeout test, stop launch and restart the policy command
with `recovery_timeout_s:=3.0`, retaining the frozen 20 s simulation horizon.
For baseline regression, stop launch and explicitly run:

```bash
ros2 launch x2_recovery recovery.launch.py controller:=scripted_baseline seed:=60
```

The [new acceptance](results/control-repair/20260924T151819Z/ros-acceptance/summary.json)
used a fresh candidate build, the existing VM dependencies and explicit new inputs,
from `/tmp`. It is not a new OS installation. Original launch, read-only measurement
wrappers, fault injection and synthetic unit tests are distinguished in the evidence.

| New deployment validation | Actual result |
| --- | --- |
| Policy / baseline cross-process checks | 142 / 92 passed; both runners exit 0 |
| Same-process original launch episodes 1 and 2 | Both SUCCEEDED; 6.616 s, 331 control transitions, continuous 2 s standing |
| Standalone / observed ROS state and retry | Observation/action/qpos/qvel/reward/time arrays match exactly |
| Actual telemetry | 18 timestamp-matched 31-joint snapshots; q/dq error 0 |
| Policy execution | 662 predict/step calls across two measured episodes; 112 active-residual transitions; all 13 parameter tensors unchanged |
| Wall timeout / service order | FAILED at 3.000078 s (0.078 ms overshoot); server response sent 4.360 ms before reset |
| Failure handling | Bad artifacts, genuine post-step exception and explicit retry verified; no fallback |
| Cleanup | Normal exit and RUNNING Ctrl+C verified, including both business children; no owned process remains |

Busy during reset took 0.644 s in policy mode and 0.624 s in baseline mode; both
met the existing one-second response goal in this run. This is not a hard real-time guarantee.
These counts are checks, not recovery episodes. The [236-test regression](results/control-repair/20260924T151819Z/validation/regression.log)
includes synthetic control-flow tests separately from real ROS/MuJoCo acceptance. The
historical 138/92 policy/baseline acceptance, earlier 111/85 checks and 183/204 regression
records remain historical; none is relabeled as new-controller evidence.
ROS success neither adds to the formal five nor proves hardware safety or PPO superiority.

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
[deployment-summary]: results/ros-acceptance/20260924T145619Z-reference-residual/summary.json
