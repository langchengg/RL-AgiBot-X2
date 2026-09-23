import datetime,hashlib,json,pathlib,re,sys,math
root=pathlib.Path(sys.argv[1]); e=root/'evidence'; patched=sys.argv[2]=='patched'
load=lambda p:json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
commands=[load_line for l in (e/'commands.jsonl').read_text().splitlines() if (load_line:=json.loads(l,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s))))]
commands.sort(key=lambda c:datetime.datetime.fromisoformat(c['start']))
for c in commands:
 c['shell_steps']=[]
 for line in pathlib.Path(c['log']).read_text().splitlines():
  m=re.match(r'^\+ ([0-9.]+) cwd=(\S+) (.+)$',line)
  if m:c['shell_steps'].append(dict(unix_time_s=float(m[1]),cwd=m[2],expanded_command=m[3]))
 c['evidence_log']=str(pathlib.Path(c['log']).relative_to(e))
(e/'commands.json').write_text(json.dumps(commands,indent=2,allow_nan=False)+'\n')
env=load(e/'environment.json'); integ=load(e/'integration/summary.json'); post=load(e/'postflight.json')
source_identity=load(e/'source-identity.json')
production_identity={p:v for p,v in source_identity['files'].items() if p=='requirements.txt' or p.startswith('src/x2_recovery/x2_recovery/') or p.startswith('src/x2_recovery/launch/') or p in ('src/x2_recovery/setup.py','src/x2_recovery/setup.cfg','src/x2_recovery/package.xml')}
production_unchanged=bool(production_identity) and all(v['identical_to_release'] for v in production_identity.values())
def phase_commands(predicate):
 return [dict(label=c['label'],exit_code=c.get('exit_code'),log=c['evidence_log']) for c in commands if predicate(c)]
torch_commands=phase_commands(lambda c:'torch' in c['label'])
requirements_commands=phase_commands(lambda c:'requirements' in c['label'])
build_commands=phase_commands(lambda c:c['label'].endswith('-build'))
runtime_commands=phase_commands(lambda c:c['label'].endswith('-runtime-identity'))
def final_success(items):return bool(items) and items[-1]['exit_code']==0
pip_check_seen=any('No broken requirements found.' in (e/c['log']).read_text() for c in requirements_commands if c['exit_code']==0)
dependency_ok=final_success(torch_commands) and final_success(requirements_commands) and pip_check_seen and env['prefix']==str(root/'repository/.venv') and 'include-system-site-packages = true' in env['pyvenv_cfg']
build_ok=final_success(build_commands)
runtime_ok=final_success(runtime_commands) and not env['forbidden_path_matches'] and env['package_prefix']==str(root/'install/x2_recovery') and all(env['entrypoints'].get(n)=='#!'+env['executable'] for n in ('recovery_node','telemetry_node'))
stage_evidence=dict(dependencies=dict(status='PASS' if dependency_ok else 'FAIL',torch_attempts=torch_commands,requirements_attempts=requirements_commands,pip_check_no_broken_requirements=pip_check_seen),build=dict(status='PASS' if build_ok else 'FAIL',commands=build_commands),installed_runtime=dict(status='PASS' if runtime_ok else 'FAIL',commands=runtime_commands,forbidden_path_matches=env['forbidden_path_matches']))
reglog=next(e.glob('*-regression.log'),None)
icmd=next(c for c in commands if c['label'].endswith('-integration'))
regression=dict(status='NOT_RUN',reason='Cross-process runner failed early; this attempt is not a completed acceptance.')
if reglog:
 reg=reglog.read_text();n=re.search(r'Ran (\d+) tests in ([0-9.]+)s',reg)
 rcmd=next(c for c in commands if c['label'].endswith('-regression'))
 outcome=re.search(r'^(OK|FAILED)(?: \(([^\n]*)\))?\s*$',reg,re.M)
 counts={key:int(value) for key,value in re.findall(r'(failures|errors|skipped|expected failures|unexpected successes)=(\d+)',outcome[2] or '')} if outcome else {}
 tests=int(n[1]) if n else None
 regression=dict(status='PASS' if n and outcome and outcome[1]=='OK' and rcmd['exit_code']==0 else 'FAIL',tests=tests,passed=None if tests is None else tests-sum(counts.values()),failures=counts.get('failures',0),errors=counts.get('errors',0),skipped=counts.get('skipped',0),expected_failures=counts.get('expected failures',0),unexpected_successes=counts.get('unexpected successes',0),test_seconds=float(n[2]) if n else None,exit_code=rcmd['exit_code'],log=reglog.name,scope='Existing full unittest suite, including bounded physical tests and synthetic PPO optimization fixtures; no successful policy retraining or five-policy evaluation rerun.')
raw=[]
for p in sorted((e/'integration').glob('*launch.log')):
 text=p.read_text()
 children=[]
 for name,pid in re.findall(r'\[([^\]]+)\]: process started with pid \[(\d+)\]',text):
  fail=re.search(r'process has died \[pid '+pid+r', exit code (-?\d+)',text)
  clean=f'process has finished cleanly [pid {pid}]' in text
  children.append(dict(name=name,pid=int(pid),returncode=0 if clean else int(fail[1]) if fail else None))
 raw.append(dict(log=str(p.relative_to(e)),children=children))
readbacks=load(e/'integration/matched-readback.json') if (e/'integration/matched-readback.json').exists() else [];errors=[abs(a-b) for r in readbacks for key in ['q','dq'] for a,b in zip(r['simulator'][key],r['received'][key])]
readback_identity_ok=bool(readbacks) and all(r['simulator']['stamp']==r['received']['stamp'] and r['simulator']['names']==r['received']['names'] and all(len(r['simulator'][key])==len(r['received'][key])==len(r['received']['names']) for key in ('q','dq')) for r in readbacks)
metrics=dict(rpc=integ['rpc'],cli=integ['cli'],matched_readback_samples=len(readbacks),matched_readback_identity_ok=readback_identity_ok,matched_readback_scalar_values=len(errors),maximum_actual_readback_absolute_error=max(errors) if errors else None,telemetry_by_scenario=integ.get('telemetry_by_scenario'),audited_execution=integ.get('audited_execution'),response_reset_step_order=integ['checks'].get('response_reset_step_order'),graph_discovery_protocol=integ.get('graph_discovery_protocol'),service_type_daemon_readiness=integ.get('service_type_daemon_readiness'))
joint_integrity_ok=None
if (e/'integration/received-joints.jsonl').exists():
 joints=[json.loads(l) for l in (e/'integration/received-joints.jsonl').read_text().splitlines()]
 metrics['joint_message_integrity']=dict(samples=len(joints),all_names_match_model_order=all(j['names']==env['model']['joint_names'] for j in joints),all_names_nonempty_unique=all(j['names'] and all(j['names']) and len(set(j['names']))==len(j['names']) for j in joints),all_arrays_valid=all(len(j['q'])==len(j['names']) and all(len(j[k]) in (0,len(j['names'])) for k in ['dq','effort']) and all(math.isfinite(v) for k in ['q','dq','effort'] for v in j[k]) for j in joints))
 if joints:
  joint_integrity_ok=all(v for k,v in metrics['joint_message_integrity'].items() if k!='samples')
 else:
  metrics['joint_message_integrity']=dict(samples=0,status='NOT_OBSERVED')
episodes=[]
for p in (e/'integration').glob('*.log'):
 rows=[]
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line[line.index('{'):]))
  except (ValueError,json.JSONDecodeError):pass
 for r in rows:
  if r.get('event')!='finished':continue
  item={k:r.get(k) for k in ['episode','controller','status','reason','control_steps','wall_s','elapsed_sim_s','timeout_overshoot_s','error']};item['log']=str(p.relative_to(e))
  accepted=next((a for a in rows if a.get('event')=='accepted' and a.get('episode')==r['episode']),None)
  reset=next((a for a in rows if a.get('event')=='reset_completed' and a.get('episode')==r['episode']),None)
  reset_start=next((a for a in rows if a.get('event')=='reset_started' and a.get('episode')==r['episode']),None)
  item.update(accepted=accepted,reset_completed=reset,reset_started=reset_start)
  if r.get('elapsed_sim_s') is not None and r.get('reason') not in ['step_error','reset_error']:
   item['physics_steps_derived_from_elapsed_sim']=round(r['elapsed_sim_s']/env['model']['physics_timestep_s'])
   item['physics_count_scope']='Derived from real elapsed simulation time / pinned 0.001 s timestep; direct env._physics_steps counts are separately saved in instrumented audit events.'
  if accepted:item['completion_monotonic_s_derived']=accepted['accepted_monotonic_s']+r['wall_s']
  episodes.append(item)
metrics['episodes']=episodes
fault_path=e/'integration/fault-events.jsonl'
if fault_path.exists():
 fault_events=[json.loads(line) for line in fault_path.read_text().splitlines()]
 executed=[v for v in fault_events if v['event']=='step_end' and v['episode']==1]
 saved=next((v for v in episodes if v['reason']=='step_error' and v['episode']==1),None)
 metrics['synthetic_fault_accounting']=dict(real_completed_step_calls=len(executed),actual_last_elapsed_sim_s=executed[-1]['elapsed_sim_s'] if executed else None,direct_physics_steps=executed[-1].get('physics_steps') if executed else None,last_valid_node_control_steps=saved['control_steps'] if saved else None,last_valid_node_elapsed_sim_s=saved['elapsed_sim_s'] if saved else None,scope='Synthetic exception is injected after the third real step returns but before RecoveryNode accounts/publishes it. Node retains the last valid two-step evidence rather than requerying failed dynamics.')
metrics['startup_error_children']=[dict(pid=p['pid'],command=p['command'],returncode=p['returncode'],log=str(pathlib.Path(p['log']).relative_to(e)),diagnostic=pathlib.Path(p['log']).read_text().strip(),scope='Installed recovery_node process, not a ros2 launch parent return code') for p in integ['processes'] if pathlib.Path(p['log']).stem.startswith('invalid-startup-') or pathlib.Path(p['log']).stem=='missing-model']
raw_child_exit_ok=bool(raw) and all(len(r['children'])==2 and {c['name'].split('-',1)[0] for c in r['children']}=={'recovery_node','telemetry_node'} and all(c['returncode']==0 for c in r['children']) for r in raw)
assert raw
required=['single_launch_two_processes','idle_observed','cli_accept_true','cli_busy_false','same_instance_new_episode','simulation_timeout','wall_timeout','same_snapshot_actual_readback','request_actually_sent_during_reset','busy_during_verified_stepping_interval','late_subscriber','terminal_held_no_more_samples','synthetic_step_fault_ends_request','real_reset_after_synthetic_fault','startup_failures_no_service','model_load_failure','active_ctrl_c_cleanup','close_after_execution_unwound']
if patched:required+=['business_nodes_discovered','response_reset_step_order','audited_busy_no_episode_reset_or_step_mutation','no_new_sampling_after_audited_terminal','original_launch_active_ctrl_c_cleanup']
required_scenarios={name:integ['checks'].get(name,dict(result='NOT_RUN'))['result'] for name in required}
complete=(dependency_ok and build_ok and runtime_ok and production_unchanged and readback_identity_ok and (not patched or joint_integrity_ok is True) and integ.get('result')=='PASS' and icmd['exit_code']==0 and all(v['result']=='PASS' for v in integ['checks'].values()) and all(v=='PASS' for v in required_scenarios.values()) and integ.get('all_children_exited') and not integ.get('cleanup_errors') and not integ.get('remaining_owned_group_members') and raw_child_exit_ok and post['result']=='PASS' and regression['status']=='PASS')
patches=sorted(e.glob('tested-v*.patch'),key=lambda p:int(re.search(r'v(\d+)',p.name)[1]))
patch_name=patches[-1].name if patched and patches else None
if patched and patch_name is None:complete=False
started=datetime.datetime.fromisoformat(commands[0]['start']).astimezone(datetime.timezone.utc)
suffix=('patched-'+re.search(r'v\d+',patch_name)[0]) if patch_name else 'release-'+env['commit'][:7]
run_id=started.strftime('%Y%m%dT%H%M%SZ')+'-'+suffix
summary=dict(result='COMPLETE' if complete else 'FAIL',run_id=run_id,run_started_utc=started.isoformat(),summary_scope='ROS acceptance; Git publication is recorded separately after push',date=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_commit=env['commit'],patch=patch_name if patched else None,patch_sha256=hashlib.sha256((e/patch_name).read_bytes()).hexdigest() if patch_name else None,production_sources_unchanged=production_unchanged,controller='scripted_baseline',isolation_scope=env['isolation_scope'],environment='environment.json',commands='commands.json',fresh_dependency_setup=stage_evidence['dependencies']['status'],fresh_build=stage_evidence['build']['status'],installed_runtime=stage_evidence['installed_runtime']['status'],stage_evidence=stage_evidence,source_identity='source-identity.json',production_source_hashes={p:v['sha256'] for p,v in production_identity.items()},modified_source_files=[p for p,v in source_identity['files'].items() if not v['identical_to_release']],regression=regression,integration=dict(explicit_runner_executed=True,raw_runner_result=integ['result'],checks=len(integ['checks']),passed=sum(v['result']=='PASS' for v in integ['checks'].values()),watchdog_exit_code=icmd['exit_code'],runner_completed=icmd['exit_code'] in (0,1) and integ.get('error',{}).get('type') not in ('KeyboardInterrupt','SystemExit'),all_required_scenarios_passed=all(v=='PASS' for v in required_scenarios.values()),required_scenarios=required_scenarios,summary='integration/summary.json',launch_business_child_exits=raw,all_business_child_exit_codes_zero=raw_child_exit_ok,cleanup=post),metrics=metrics,not_applicable={'ROS_checkpoint_path_error':'No checkpoint/controller parameter or ROS policy-loading entry point exists.'},logical_validation={'pending_busy':'test_ros_nodes.RecoveryLogicTests.test_acceptance_pending_busy_no_environment_calls_or_mutation in regression.log; logical fixture, not a cross-process reset window','model_hash_mismatch':'test_model.ModelTests.test_missing_path_and_wrong_pin_hash_fail_clearly; synthetic expected-hash mismatch. Missing model path is also tested in a real child process.'},validation_layers={'original_launch':'Unmodified recovery.launch.py starts real recovery_node + telemetry_node. CLI and both timeout scenarios use real native simulation.','read_only_audit':'A separate subprocess wraps real send_response/reset/step/read_state and samples without extra physics; proves server ordering and same-stamp actual state.','synthetic_fault':'An explicit one-shot exception after a real physics step; tests error handling/retry, not natural simulator failure.','unit_tests':'Logical node fixture tests plus bounded real physics and synthetic optimizer fixtures.'},limits=['Same existing VM, not a new OS or separate machine installation.','--system-site-packages intentionally inherits declared Ubuntu and ROS packages.','ROS scripted baseline did not recover; terminal FAILED is expected for these integration scenarios.','Standalone reference + PPO residual frozen fixed-supine 5/5 is historical and not rerun or wired into ROS.','Cooperative wall timeout cannot preempt reset/step; measured overshoot is reported.','Terminal network delivery can lag creation; closing a ROS context does not guarantee subscriber delivery of shutdown status.','Acquisition timestamps use ROS clock; simulation progress uses MuJoCo time; budgets use monotonic wall time. No /clock is published.'],failures_or_incomplete=[])
if not raw_child_exit_ok:
 summary['failures_or_incomplete'].append(dict(issue='Original runner false-positive cleanup summary',detail='Noninteractive launch received group SIGINT and forwarded another SIGINT to its children; both normal/wall children exited -2 during cleanup. Raw runner PASS is preserved but this final acceptance attempt is FAIL.',fix='Retest in a fresh independent environment with explicit runner patch: signal launch parent once, inspect business child exits, record group cleanup.'))
summary['nonzero_preparation_attempts']=[dict(label=c['label'],exit_code=c['exit_code'],log=c['evidence_log']) for c in commands if c['exit_code'] != 0 and not any(x in c['label'] for x in ['integration','regression'])]
summary['package_integrity_rejections']=[dict(label=c['label'],exit_code=c['exit_code'],log=c['evidence_log']) for c in commands if c['exit_code']!=0 and 'DO NOT MATCH THE HASHES' in pathlib.Path(c['log']).read_text()]
summary['preparation_retry_notes']=['All network retries are explicit commands with separate logs; no assertion budget was relaxed.','Root1 full/shallow Git clones and one filtered handshake failed; HTTP/1.1 sparse clone succeeded at the exact base commit.','Root1 python3-cffi apt query returned nonzero because that optional inventory name was absent; declared cffi was installed in the venv, and python3-cffi-backend system package was recorded.','Root2 had Git TLS failures and one manually bounded interrupted stalled download. Its first Torch transfer was truncated (56.4/101.9 MB) and rejected by the official hash; a --no-cache-dir retry kept the same source, version and integrity check.'] if patched else ['Git transfer/TLS failures retained; explicit HTTP/1.1 sparse clone retry tested identical source commit.','The missing python3-cffi apt inventory name was not a missing project prerequisite: requirements installs cffi in the venv; system python3-cffi-backend was separately recorded.']
if integ.get('result')!='PASS':
 summary['failures_or_incomplete'].append(dict(issue='Cross-process runner failure',error=integ.get('error'),failed_checks={k:v for k,v in integ['checks'].items() if v['result']!='PASS'},detail='The first independent CLI DirectNode observer returned an empty graph within its default 0.5 s discovery wait, although the integration client and two business nodes were ready. No daemon preexisted, so the installed NodeStrategy spawned one and read through a new DirectNode; daemon-cache causation is not established. This attempt accepted no request. The third fresh run explicitly changes supported direct discovery waits to 2 s and bounds daemon graph readiness for unchanged service-type CLI; 10 s CLI, 1 s discovered-client RPC and 210 s runner budgets remain unchanged.' if root.name=='x2-final-ros.pukricvy' else None))
summary['environment_notes']=['NumPy/Pillow/matplotlib/setuptools/colcon come from declared system packages; simulator/RL packages and cffi installed in the new venv.','Python executable realpath /usr/bin/python3.12 is expected: executable and sys.prefix point to the new venv.','torch.classes/torch.ops __file__ values are synthetic relative labels; implementations are torch._classes/torch._ops inside the new venv. Ubuntu sitecustomize is a system apport hook, not a user startup file.']
summary['module_origins']=env['modules']
summary['model_identity']=env['model']
summary['integration']['success']=integ.get('result')=='PASS' and icmd['exit_code']==0
summary['validation_layer_results']=dict(original_launch='PASS' if all(integ['checks'].get(k,{}).get('result')=='PASS' for k in ('simulation_timeout','wall_timeout','cli_accept_true','cli_busy_false')) else 'PARTIAL' if integ['checks'].get('single_launch_two_processes',{}).get('result')=='PASS' else 'NOT_RUN',read_only_audit=integ['checks'].get('same_snapshot_actual_readback',{}).get('result','NOT_RUN'),synthetic_fault=integ['checks'].get('synthetic_step_fault_ends_request',{}).get('result','NOT_RUN'),unit_tests=regression['status'])
summary['validation_layers']['startup_errors']='Illegal parameters and missing model run installed recovery_node child processes with actual nonzero process exits. They do not claim nonzero ros2 launch parent exits.'
summary['preparation_retry_notes'].append('This attempt\'s failed command list and stage_evidence retain actual chronology. Stage PASS requires the final explicit dependency command to exit 0 plus pip check and runtime identity; it never means an uninterrupted first attempt.')
summary['logical_validation_execution']='PASS' if regression['status']=='PASS' else 'NOT_ESTABLISHED_BY_THIS_ATTEMPT'
summary['limits']+=['Reported process exit times are parent observation times, not exact kernel exit instants.','Original launch cumulative physics counts are derived from elapsed simulation time and model timestep; separate audited env._physics_steps counts are directly observed.','Scenario labels record receiver phase; ROS acquisition stamps distinguish queued deliveries from newly created samples.']
for phase,details in stage_evidence.items():
 if details['status']!='PASS':summary['failures_or_incomplete'].append(dict(issue=phase,detail=details))
if not production_unchanged:summary['failures_or_incomplete'].append(dict(issue='Production source identity differs or is missing'))
if regression['status']!='PASS':summary['failures_or_incomplete'].append(dict(issue='Full regression incomplete or failed',detail=regression))
if not readback_identity_ok:summary['failures_or_incomplete'].append(dict(issue='No complete same-stamp readback identity evidence in this attempt'))
if patched and joint_integrity_ok is not True:summary['failures_or_incomplete'].append(dict(issue='Complete valid raw JointState stream not observed in this attempt'))
if (e/'publication-document-identity.json').exists():
 summary['publication_document_identity']=load(e/'publication-document-identity.json')
 summary['publication_document_identity_file']='publication-document-identity.json'
if (e/'mujoco-download-verification.json').exists():
 summary['network_download_recovery']=dict(identity='mujoco-download-identity.json',verification=load(e/'mujoco-download-verification.json'),scope='Two truncated pip downloads were rejected. Bounded curl downloaded the same official wheel independently into this new root; official size and SHA-256 both matched before local-wheel installation and the unchanged requirements/pip-check command. No old cache or version substitution.')
summary['evidence_sha256']={str(p.relative_to(e)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(e.rglob('*')) if p.is_file() and p != e/'summary.json' and '__pycache__' not in p.parts}
# Include the runner's summary; only the top-level summary cannot hash itself.
summary['evidence_sha256']['integration/summary.json']=hashlib.sha256((e/'integration/summary.json').read_bytes()).hexdigest()
(e/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:summary[k] for k in ['result','run_id','test_commit','patch_sha256','regression','failures_or_incomplete']},indent=2))
