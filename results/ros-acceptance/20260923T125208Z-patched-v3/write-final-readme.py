import datetime,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);r=json.loads((root/'evidence/summary.json').read_text());i=json.loads((root/'evidence/integration/summary.json').read_text());p=root/'repository/README.md'
assert r['result']=='COMPLETE'
assert r['patch']=='tested-v3.patch' and r['production_sources_unchanged']
assert hashlib.sha256((root/'evidence'/r['patch']).read_bytes()).hexdigest()==r['patch_sha256']
base_path='results/ros-acceptance/'+r['run_id']
history=[json.loads((pathlib.Path(x)/'evidence/summary.json').read_text()) for x in ('/tmp/x2-final-ros.zvwycfdg','/tmp/x2-final-ros.pukricvy')]
assert all(x['result']!='COMPLETE' for x in history)
release_path='results/ros-acceptance/'+history[0]['run_id']
second_path='results/ros-acceptance/'+history[1]['run_id']
run_date=datetime.datetime.fromisoformat(r['run_started_utc']).date().isoformat()
checks=i['checks'];normal=checks['simulation_timeout'];wall=checks['wall_timeout'];audit=i['audited_execution'];tele=i['telemetry_by_scenario']['normal-episode-1'];match=checks['same_snapshot_actual_readback']
normal_episode=next(x for x in r['metrics']['episodes'] if x['log']=='integration/normal-launch.log' and x['episode']==1)
wall_episode=next(x for x in r['metrics']['episodes'] if x['log']=='integration/wall-launch.log')
assert r['metrics']['maximum_actual_readback_absolute_error']==match['max_absolute_error']
accepted=[x['round_trip_s']*1000 for x in i['rpc'] if x['success']];busy=[x['round_trip_s']*1000 for x in i['rpc'] if 'stepping-busy' in x['label']];reset_busy=next(x for x in i['rpc'] if x['label']=='real-reset-busy')
cli={pathlib.Path(x['log']).stem:x['process_wall_s']*1000 for x in i['cli']}
new=f'''### Final isolated ROS acceptance — {run_date}

The final run is **COMPLETE**, with a fresh build, {r['regression']['passed']}/{r['regression']['tests']}
regression tests ({r['regression']['failures']} failures, {r['regression']['errors']} errors, {r['regression']['skipped']} skips), and a separately executed integration
runner with {r['integration']['passed']}/{r['integration']['checks']} checks and outer watchdog exit 0.
Evidence: [final summary]({base_path}/summary.json),
[environment and module origins]({base_path}/environment.json),
[actual commands]({base_path}/commands.json),
[integration records]({base_path}/integration/summary.json).
The original temporary records remain at `{root}`; paths inside raw logs retain
that execution identity. Published evidence includes file hashes; the original successful and failed attempt directories are retained.

Scope: **the same Ubuntu ARM64 VM with new source, project venv, independently
downloaded model, clean shell and build/install products; the declared Ubuntu/ROS
system dependencies were reused**. This is not a new OS or new-machine installation.
NumPy/Pillow/matplotlib/setuptools/colcon were inherited from Ubuntu; rclpy and ROS
interfaces came from `/opt/ros/jazzy`. Simulator/RL packages and cffi came from the new
venv. No system package was installed, upgraded or replaced, and no old project source,
venv, overlay or model cache was loaded. The interpreter's `/usr/bin/python3.12`
realpath is expected; `sys.executable`, `sys.prefix` and both node shebangs identify
the new project venv. The new installed prefix is `{root}/install/x2_recovery`;
symlink-installed production modules resolve into this run's new clone.

Code identity: published base `71037f7d5530a65da220345f9037ac4d29f32908` plus the explicit
[tested patch]({base_path}/{r['patch']})
(SHA-256 `{r['patch_sha256']}`). The patch changes README setup guidance and the existing
integration runner only. All production modules, launch, package/setup declarations,
requirements and physics/model/control/success definitions remain byte-identical to
the release; [source identities]({base_path}/source-identity.json)
contain full SHA-256 values. The final result prose was added after testing; it does
not change the tested commands or runtime/test implementation.

The [unmodified-release attempt]({release_path}/summary.json)
is retained as **FAIL for clean shutdown**, even though its original runner reported
68 checks passed and exited 0. Its normal/wall launch children exited `-2` during
cleanup: the noninteractive runner sent SIGINT to the entire group, then Jazzy launch
forwarded SIGINT again. The runner now signals the launch parent once, lets launch
forward it, checks both business child exit codes, and records owned process-group
cleanup. All three final raw launch scenarios have two business children exiting 0.
The new active-launch case also proves shutdown while real physics is running.
No production node change was needed.

The [second independent attempt]({second_path}/summary.json) failed before accepting
any recovery request. Its new CLI graph observer returned an empty node list within
the installed default 0.5 s discovery wait, although the runner's existing observer
and both business nodes were ready. With no prior daemon, Jazzy's NodeStrategy
starts a daemon and reads through a new DirectNode; the evidence does not establish
that a daemon cache caused the empty result. The v2 signal fix already gave both
startup-only business processes exit 0. This partial attempt is not counted as a
completed ROS acceptance or active-episode shutdown test.

The final v3 runner explicitly uses installed, help-verified `--no-daemon --spin-time 2`
for node list and verbose topic information. This is a declared discovery-protocol
change from the default 0.5 s, not a claim that the failed earlier protocol passed.
Installed `ros2 service type` supports neither option, so its original command is
retained after starting this run's domain daemon and verifying its service graph
with a bounded observer. That observer's XMLRPC socket timeout is capped at 5 s and
the remaining 10 s readiness budget. The original 10 s CLI, 1 s discovered-client RPC,
30 s normal wall and 210/240 s runner/watchdog budgets are unchanged. Every CLI
command and actual output, including daemon readiness, is saved in the final records.

The old README commented out venv creation, omitted explicit inherited Ubuntu
prerequisites and mixed historical paths with fresh-install commands. Setup now
states and executes those steps. The final v3 attempt is the third independent
source clone, project venv, model checkout and build tree; all previous attempts remain
available with their original outcomes. Network and package-download failures are retained with their actual nonzero exits.
The second environment rejected a truncated Torch wheel. In the third environment,
two pip transfers produced the same truncated MuJoCo wheel and failed its official
SHA-256 check. The complete identical wheel was then independently downloaded from
the official files.pythonhosted.org URL using bounded curl transport, checked against
PyPI's declared byte length and SHA-256, and installed before repeating the unchanged
requirements command and pip check. These transport recovery commands are recorded;
no version, source, or integrity requirement was changed, and no old cache was used.
The final dependency stage is based on successful installation commands, actual
module origins and `pip check`, with failed attempts preserved. The
HTTP/1.1 partial clone checks out top-level files plus all `src`; omitted historical
`results` assets are a download-scope choice, not a source-code change. No proxy or
certificate exception was used. This was not an uninterrupted first-try network install.

| Final measured scenario | Result |
| --- | --- |
| Original launch, first CLI request and busy CLI request | Both nodes ready; IDLE before request; accepted true, busy false; CLI total {cli['cli-accept']:.3f} / {cli['cli-busy']:.3f} ms including Python startup and DDS discovery |
| Discovered-client accepted RPCs / stepping-busy RPCs | {min(accepted):.3f}–{max(accepted):.3f} ms / {min(busy):.3f}–{max(busy):.3f} ms |
| Busy request issued during real reset | {reset_busy['round_trip_s']*1000:.3f} ms; processed after the single-threaded reset returned, without episode/reset mutation |
| Normal simulation timeout | FAILED / time_limit; {normal['control_steps']} control steps / {normal_episode['physics_steps_derived_from_elapsed_sim']} physics substeps derived from elapsed time and model timestep; {normal['elapsed_sim_s']:.12f} s simulation, {normal['wall_s']:.6f} s wall |
| Independent 3 s wall timeout | FAILED / recovery_timeout; {wall['control_steps']} control steps / {wall_episode['physics_steps_derived_from_elapsed_sim']} derived physics substeps; {wall['elapsed_sim_s']:.6f} s simulation, {wall['wall_s']:.9f} s wall; overshoot {wall['timeout_overshoot_s']*1000:.6f} ms |
| Original-launch reset durations | Normal {normal_episode['reset_completed']['duration_s']:.6f} s; wall-timeout scenario {wall_episode['reset_completed']['duration_s']:.6f} s; simulation time limits exclude reset settling |
| Read-only audit | Actual send_response completion < reset begin < reset end < first step; reset {audit['reset_duration_s']:.6f} s; {audit['actual_completed_control_calls']} control calls / {audit['actual_physics_steps']} directly observed physics steps |
| Same-snapshot telemetry | {match['matched']} matched timestamped snapshots; names/position/velocity match; actual maximum absolute error {match['max_absolute_error']} |
| Normal first-episode telemetry | {tele['samples']} received samples; acquisition interval median {tele['acquisition_intervals']['median_s']:.6f} s, range {tele['acquisition_intervals']['minimum_s']:.6f}–{tele['acquisition_intervals']['maximum_s']:.6f} s |
| Retry, errors and cleanup | Same-instance new reset after terminal and after synthetic step fault; invalid startup/model failures; active Ctrl+C; all owned processes and created daemon exited |

Raw launch scenarios use the unchanged production launch and native simulator. A
separate read-only observation wrapper records real reset/step/read_state/send_response
calls without adding physics operations. Its 2.003 s audit proves same-sample readback
and server ordering; it is labeled separately from the original launch. The step
exception is **synthetic fault injection after real physics**, not a naturally observed
simulator failure. It executes the third physical control call before raising; the
node retains its last valid two-step summary rather than querying failed dynamics.
Startup parameter and missing-model failures are actual installed-node child exits,
not launch-parent return codes. Model-hash mismatch is a separately identified unit
fixture that changes the expected hash, not the downloaded model. Pending-state busy rejection is separately supported by the logical
node unit test; reset-window and stepping busy requests are real cross-process RPCs.

JointState stamps are ROS acquisition time; recovery progress is MuJoCo relative time;
watchdog budgets use monotonic wall time. Sampling intervals are measured, not a
strict 50 Hz claim. Terminal samples stop; late telemetry receives retained FAILED
and `no_sample`, and queued network delivery is distinguished from new sampling.
Shutdown logs prove server cleanup; they do not claim the subscriber received a
post-context-shutdown terminal message. Wall timeout is cooperative and cannot
preempt a reset/step; the measured overshoot is retained. The 1 s RPC, 30 s normal
wall and 210/240 s runner/watchdog budgets are project test settings, not PDF standards.

ROS remains **scripted_baseline**, and the normal baseline episodes did not stand up.
Expected FAILED/time_limit verifies integration without establishing policy recovery.
The published reference + PPO residual fixed-supine **5/5** remains an independent
historical result; neither retraining nor that evaluation was repeated, and the hybrid
was not connected to ROS. ROS checkpoint-path error testing is **NOT_APPLICABLE**:
this implementation has no checkpoint/controller loading parameter. The regression's
bounded synthetic optimization fixtures are not retraining the successful checkpoint.

'''
s=p.read_text();start=s.index('### Final isolated ROS acceptance');end=s.index('## Validation and Reproducibility',start);p.write_text(s[:start]+new+'\n'+s[end:]);print('Updated measured README section')
