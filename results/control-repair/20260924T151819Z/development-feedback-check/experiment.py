"""Five preregistered development replays of one analytical-feedback candidate.

No training, published-manifest alteration, active-state restoration or filtering.
"""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict,replace
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import argparse,hashlib,importlib.util,json,shutil,signal,sys,time
import numpy as np
from x2_recovery.env import X2RecoveryEnv,ControlledRecoveryEnv,ControlConfig
from x2_recovery.evaluate import prepare,verify_policy
from x2_recovery.motion_search import constraint_trial
from x2_recovery.train import write_json,json_value,source_hashes

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(json_value(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();root=args.repo.resolve();out=args.output.resolve();run=root/'results/control-repair/20260924T151819Z';source=run/'tail-feedback-g33-ik';planpath=source/'plan.json';p=json.loads(planpath.read_text());name='ik-feedback-18-g33-9-kp0.3-kd0.03';item=next(c for c in p['candidates'] if c['name']==name);previouspath=source/name/'summary.json';previous=json.loads(previouspath.read_text());inputs=(root/p['input_directory']).resolve();runnerpath=source/'tail_feedback_pilot.py';helperpath=source/'x2_tail_ankle_feedback.py'
 if out.exists() or out.is_relative_to(inputs) or inputs.is_relative_to(out):raise ValueError('Output must be new and disjoint from frozen input')
 assert previous['success'] and previous['feasible'];assert digest(item['control_config'])==item['control_config_sha256']==previous['control_sha256'];assert sha(runnerpath)==p['pilot_sha256'] and sha(helperpath)==p['helper_sha256'];assert sha(root/'src/x2_recovery/x2_recovery/motion_search.py')==p['motion_search_sha256']
 trials=[dict(name='nominal-221030',seed=221030,reset_perturb_rad=0.)]+[dict(name=f'perturbed-{s}',seed=s,reset_perturb_rad=.002) for s in [26092400,26092405,26092412,26092413]]
 started=time.monotonic();out.mkdir(parents=True);shutil.copyfile(__file__,out/'experiment.py');shutil.copyfile(runnerpath,out/runnerpath.name);shutil.copyfile(helperpath,out/helperpath.name);core=source_hashes();sources={str(x):sha(x) for x in [planpath,previouspath,runnerpath,helperpath,Path(__file__),root/'src/x2_recovery/x2_recovery/motion_search.py']}
 manifest=dict(schema='analytical-feedback-development-check-v1',status='PREREGISTERED',created_utc=datetime.now(timezone.utc).isoformat(),controller='reference_motion_plus_observation_only_analytical_ankle_feedback',candidate=item,trials=trials,planned=5,development_only=True,held_out=False,training=False,weight_transfer=False,ppo_control_inference=False,input_directory=str(inputs),source_sha256=sources,source_core=core,allowlisted_difference={'reset_perturb_rad':[0.,.002]},reset_protocol='Independent original legal supine reset for each trial, uniform supported joint perturbation in plus/minus0.002rad; no seed selection or replacements. Existing development seeds.',comparison=p['comparison'],horizon_s=20.,measurement='Existing StreamingConstraintObserver every physics step; complete success/safety/time-limit episode, no early diagnostic cutoff.',reference_hashes={k:previous[k] for k in ['physical_state_sha256','policy_action_sha256','actual_target_sha256']},workers=1,max_wall_seconds=120,source_classification='Original published inputs strictly validated once unchanged. The in-memory saved.training.env copy is only constraint_trial argument plumbing with one explicit reset difference; not a published manifest or original execution identity.',command=[sys.executable,*sys.argv]);write_json(out/'manifest.json',manifest)
 runner=module(runnerpath,'development_observed_feedback');formula=module(helperpath,'development_feedback_formula');owner=None;frozen=original=None;results=[];exceptions=[];fatal=None;input_hashes={};nominal_match=None
 def budget(signum,frame):raise TimeoutError('Preregistered120s development wall budget exhausted')
 oldhandler=signal.signal(signal.SIGALRM,budget);signal.setitimer(signal.ITIMER_REAL,120.)
 try:
  with ExitStack() as guards:
   for n in ['mujoco.mj_step','x2_recovery.env.X2RecoveryEnv.reset','x2_recovery.env.X2RecoveryEnv.step','x2_recovery.env.ControlledRecoveryEnv.reset','x2_recovery.env.ControlledRecoveryEnv.step']:guards.enter_context(patch(n,side_effect=RuntimeError('Strict prepare must be read-only')))
   owner,frozen,original,prepared=prepare(inputs,expected_checkpoint_sha256=p['expected_checkpoint_sha256'])
  input_hashes={str(x):sha(x) for x in prepared['input_sources'].values()};manifest.update(status='RUNNING',strict_original_load_action_check=prepared['consistency'],input_sha256=prepared['hashes'],original_env_config=asdict(prepared['config'].env));write_json(out/'manifest.json',manifest)
  for index,trial in enumerate(trials):
   if time.monotonic()-started>=120:break
   directory=out/trial['name'];directory.mkdir();base=None;feedback=None;observer=None;result=None;reset_record={'called':False,'completed':False};cfg=replace(prepared['config'].env,reset_perturb_rad=trial['reset_perturb_rad']);originalcfg=asdict(prepared['config'].env);trialcfg=asdict(cfg);differences={k:dict(original=originalcfg[k],experiment=trialcfg[k]) for k in originalcfg if originalcfg[k]!=trialcfg[k]};assert set(differences)<= {'reset_perturb_rad'};saved=deepcopy(prepared['saved']);saved['training']['env']=json_value(trialcfg)
   try:
    assert all(sha(x)==h for x,h in input_hashes.items());base=X2RecoveryEnv(config=cfg,render_mode=None,capture_substeps=False);control=ControlConfig.from_dict(item['control_config']);preview=ControlledRecoveryEnv(base,control);c=dict(p['interface'],residual_scale_rad=item['cap_rad'],start_s=item['start_s'],ramp_s=item['ramp_s']);scales=np.zeros(31);scales[c['ankle_joint_indices']]=c['residual_scale_rad'];assert np.array_equal(preview.residual_scale,scales);assert base.config.episode_timeout_s==c['episode_timeout_s']==20.;native_reset=base.reset
    def record_reset(*a,**kw):
     reset_record['called']=True
     try:
      result=native_reset(*a,**kw);reset_record['completed']=True;return result
     except Exception as exc:
      reset_record['error']=dict(type=type(exc).__name__,message=str(exc),evidence=getattr(exc,'evidence',None));raise
     finally:
      q=base.data.qpos.copy();dq=base.data.qvel.copy();reset_record.update(qpos_sha256=hashlib.sha256(np.asarray(q,dtype='<f8').tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(np.asarray(dq,dtype='<f8').tobytes()).hexdigest(),state_sha256=hashlib.sha256(np.r_[q,dq].astype('<f8').tobytes()).hexdigest(),absolute_sim_time_s=float(base.data.time),reset_evidence=getattr(base,'reset_evidence',None));np.savez_compressed(directory/'reset_state.npz',qpos=q,qvel=dq);write_json(directory/'reset.json',json_value(reset_record))
    base.reset=record_reset;feedback=runner.ObservedFeedback(base,formula,c,item['kp'],item['kd']);observer=runner.TargetObserver(base)
    with observer:result=constraint_trial(saved,control,seed=trial['seed'],policy=feedback,base=base,zero_residual=False,comparison=p['comparison'],name=trial['name'])
    assert len(feedback.observations)==result['transitions']==len(observer.rows);assert reset_record['completed'];assert asdict(base.config)==trialcfg
    result.update(controller=manifest['controller'],weight_transfer=False,training=False,ppo_control_inference=False,source_candidate=name,reset_configuration_changes=differences,reset_handoff=reset_record,observation_probe=dict(calls=len(feedback.observations),max_decoding_errors=feedback.errors,pregate_nonzero_actions=feedback.pregate_nonzero),target_pipeline_probe=observer.summary(),control_config_sha256=item['control_config_sha256'])
    if index==0:
     nominal_match={k:result[k]==v for k,v in manifest['reference_hashes'].items()};result['nominal_three_hashes_match']=nominal_match
     if not all(nominal_match.values()):raise ValueError('Nominal replay did not exactly match original state/action/target hashes')
    verify_policy(frozen,original);write_json(directory/'summary.json',result);results.append(result)
    with (out/'results.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['name','success','feasible','sim_seconds','max_stable_hold_s','physical_state_sha256']}),flush=True)
   except Exception as exc:
    error=dict(index=index,trial=trial,type=type(exc).__name__,message=str(exc),reset=reset_record);exceptions.append(error);write_json(directory/'error.json',json_value(error))
    if result is not None:result.update(status='INVALID',validation_error=error);write_json(directory/'summary.json',json_value(result))
    print(json.dumps(dict(name=trial['name'],invalid=True,type=type(exc).__name__,message=str(exc))),flush=True)
    if isinstance(exc,TimeoutError) or index==0 or not reset_record['called'] or reset_record['completed']:raise
    # A failed legal reset is retained; proceed only to the remaining preregistered seeds.
   finally:
    if observer is not None:observer.save(directory/'target_pipeline.npz')
    if feedback is not None:np.savez_compressed(directory/'feedback_probe.npz',observations=np.asarray(feedback.observations,dtype=np.float32),actions=np.asarray(feedback.actions,dtype=np.float32),signals=np.asarray(feedback.signals,dtype=np.float64))
    if base is not None:base.close()
  verify_policy(frozen,original)
 except Exception as exc:fatal=dict(type=type(exc).__name__,message=str(exc))
 finally:
  signal.setitimer(signal.ITIMER_REAL,0.);signal.signal(signal.SIGALRM,oldhandler)
  if owner is not None:
   try:verify_policy(frozen,original)
   finally:owner.close()
 unchanged=core==source_hashes() and all(sha(x)==h for x,h in sources.items());inputsunchanged=bool(input_hashes) and all(sha(x)==h for x,h in input_hashes.items());status='FAILED' if fatal or not unchanged or not inputsunchanged else 'COMPLETE' if len(results)+len(exceptions)==5 else 'BUDGET_STOP';perturbed=[r for r in results if r['name'].startswith('perturbed')];summary=dict(status=status,planned=5,completed=len(results),invalid=len(exceptions),not_executed=5-len(results)-len(exceptions),nominal_three_hashes_match=nominal_match,perturbed_planned=4,perturbed_valid=len(perturbed),perturbed_success=sum(r['success'] for r in perturbed),perturbed_feasible=sum(r['feasible'] for r in perturbed),distinct_valid_handoff_states=len({r['reset_handoff']['state_sha256'] for r in results}),fatal=fatal,exceptions=exceptions,inputs_unchanged=inputsunchanged,source_unchanged=unchanged,wall_seconds=time.monotonic()-started,controller=manifest['controller'],training=False,weight_transfer=False,ppo_control_inference=False,interpretation='Small preregistered development check only; not formal five, not held-out robustness and not evidence of PPO contribution. All illegal resets retained; no replacements.');write_json(out/'summary.json',json_value(summary));manifest.update(status=status,completed_utc=datetime.now(timezone.utc).isoformat());write_json(out/'manifest.json',manifest);print(json.dumps(summary),flush=True);return int(status!='COMPLETE')
if __name__=='__main__':sys.exit(main())
