"""Fixed analytical ankle-feedback experiment through the existing17D controller.

No core edits, PPO control, training, deployment bundle or simulator state writes.
Plan is static; run needs explicit fresh output and executes complete episodes.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,importlib.util,json
from pathlib import Path
import shutil,sys,time
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def helper_module(path):
 spec=importlib.util.spec_from_file_location('experimental_tail_ankle_formula',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def build_plan(root,helper,source_plan,design):
 from x2_recovery.env import ControlConfig
 source=json.loads(source_plan.read_text())
 if source['schema']!='experimental-tail-ankle-feedback-v1' or source['anchors']!=['g19-29','g23-31']:raise ValueError('Expected frozen g19/g23 anchor plan')
 if sha(helper)!=source['helper_sha256']:raise ValueError('Original pure helper differs')
 plan=deepcopy(source);plan.update(schema='experimental-tail-ankle-feedback-list-v1',created_utc=datetime.now(timezone.utc).isoformat(),pilot_sha256=sha(__file__),
  prior_anchor_plan=dict(path=str(source_plan),sha256=sha(source_plan)),source_pilot_sha256=source['pilot_sha256'],design=design)
 if design=='fine-pd':plan['anchors']=['g19-29'];plan['profiles']={'g19-29':plan['profiles']['g19-29']}
 plan.pop('gain_pairs',None)
 saved=json.loads((root/plan['input_directory']/'resolved_config.json').read_text());physical=saved['environment']['base']
 bounds=np.array([physical['q_min_rad'],physical['q_max_rad']]).T
 def entry(label,kp,kd,cap,start,calibration,index):
  profile=plan['profiles'][label];cfg=deepcopy(profile['control_config']);cfg['residual_start_s']=start;cfg['residual_ramp_s']=.2;cfg['residual_joint_multipliers']=[0.]*31
  delta=np.minimum(np.max(abs(np.asarray(physical['effort_limits_Nm'])),axis=1)/physical['kp_Nm_rad'],np.ptp(bounds,axis=1)*.25)*cfg['delta_multiplier']
  for joint in plan['interface']['ankle_joint_indices']:cfg['residual_joint_multipliers'][joint]=float(cap/(delta[joint]*cfg['residual_multiplier']))
  original=deepcopy(profile['control_config'])
  for key,change in profile['control_changes'].items():original[key]=change['original']
  if digest(original)!=profile['original_control_sha256']:raise ValueError('Anchor reconstruction identity mismatch')
  changes={key:dict(original=original[key],experiment=cfg[key]) for key in cfg if cfg[key]!=original[key]}
  if set(changes)!={'residual_start_s','residual_ramp_s','residual_joint_multipliers'}:raise ValueError('Unexpected control changes')
  ControlConfig.from_dict(cfg)
  return dict(name=f"authority-{index:02d}-{label}-cap{cap:g}-start{start:g}-kd{kd:g}"+('-zero' if calibration else ''),anchor=label,
   kp=0. if calibration else kp,kd=0. if calibration else kd,cap_rad=cap,start_s=start,ramp_s=.2,calibration=calibration,
   control_config=cfg,control_config_sha256=digest(cfg),control_changes=changes)
 candidates=[]
 calibration_cap=.08 if design=='fine-pd' else .16
 for label in plan['anchors']:candidates.append(entry(label,0.,0.,calibration_cap,3.5,True,len(candidates)))
 if design=='fine-pd':
  for kp in [.08,.12,.15,.18,.22,.30]:
   for kd in [.035,.045,.055,.065]:candidates.append(entry('g19-29',kp,kd,.08,3.5,False,len(candidates)))
  factors=dict(kp_grid=[.08,.12,.15,.18,.22,.30],kd_grid=[.035,.045,.055,.065],cap_rad=.08,start_s=3.5,ramp_s=.2)
  order='One zero-gain g19-29 baseline first; then fixed Kp outer and Kd inner Cartesian grid,24 analytical candidates. No outcome-based selection or replacement. Authority/timing remain exactly fixed.'
 else:
  conditions=[(.04,3.0),(.04,3.2),(.04,3.5),(.08,3.5),(.12,3.5),(.16,3.5)]
  for label in plan['anchors']:
   for cap,start in conditions:
    for kd in [0.,.03]:candidates.append(entry(label,.15,kd,cap,start,False,len(candidates)))
  factors=dict(separated_conditions=conditions,kp=.15,kd_grid=[0.,.03],ramp_s=.2)
  order='Two zero-gain baseline calibrations first, each with late cap.16 configuration; then anchor outer, six(cap,start)conditions, Kd0/.03 inner.24 analytical candidates plus2 calibrations, no replacement.'
 plan.update(candidates=candidates,planned=len(candidates),factors=factors,candidate_list_sha256=digest(candidates),candidate_order=order,
  target_measurement='A read-only temporary wrapper calls original ControlledRecoveryEnv.step exactly once and returns its original tuple. It samples the original pure reference/gate function before step, reconstructs pre-clamp request and checks exact equality with actual returned requested/adopted targets. Clip and rate-limit counts are control transitions, not physical-step durations.',
  physical_risk='Late cap.16 is close to the g13-10 observed initial .166rad target margin; newanchors can differ. Permission may be consumed by existing nominal target clamping. Earlier starts use only.04rad authority. Actual range/impact results remain decisive.',
  motion_search_sha256=sha(root/'src/x2_recovery/x2_recovery/motion_search.py'))
 return plan


class ObservedFeedback:
 def __init__(self,base,formula,interface,kp,kd):
  self.base=base;self.c=interface;self.actor=formula.TailAnkleFeedback(interface['q_lower'],interface['q_upper'],interface['waist_indices'],kp,kd,
   cap_rad=interface['residual_scale_rad'],start_s=interface['start_s'],episode_timeout_s=interface['episode_timeout_s'],joint_velocity_scale=interface['joint_velocity_scale_rad_s'],base_angular_scale=interface['base_angular_scale_rad_s'])
  self.observations=[];self.actions=[];self.signals=[];self.errors={key:0. for key in interface['assertion_tolerances']};self.pregate_nonzero=0
 def predict(self,observation,deterministic=True):
  if deterministic is not True:raise ValueError('Only deterministic analytical inference supported')
  obs=np.asarray(observation)
  if obs.shape!=(149,) or obs.dtype!=np.float32 or not np.isfinite(obs).all():raise ValueError('Expected current float32 finite149D observation')
  b=self.base;c=self.c;q,dq=b.loaded.read_state(b.data);R=b.data.xmat[b.context.pelvis].reshape(3,3)
  lower=np.asarray(c['q_lower']);upper=np.asarray(c['q_upper']);decoded=(lower+upper)/2+obs[:31].astype(float)*(upper-lower)/2
  errors=dict(q_rad=float(np.max(abs(decoded-q))),dq_rad_s=float(np.max(abs(obs[31:62].astype(float)*c['joint_velocity_scale_rad_s']-dq))),
   pelvis_gravity=float(np.max(abs(obs[62:65]-(R.T@np.array([0.,0.,-1.]))))),
   pelvis_angular_rad_s=float(np.max(abs(obs[68:71].astype(float)*c['base_angular_scale_rad_s']-b.data.qvel[3:6]))),
   elapsed_s=abs(float(obs[148])*c['episode_timeout_s']-float(b.data.time-b.episode_start_time)))
  # torso xmat is already current from the environment's existing forward pass.
  signals=self.actor.signals(obs)
  torso=b.data.xmat[b.context.torso].reshape(3,3)
  errors['torso_gravity']=float(np.max(abs(signals['torso_gravity']-torso.T@np.array([0.,0.,-1.]))))
  for key,error in errors.items():
   self.errors[key]=max(self.errors[key],error)
   if error>c['assertion_tolerances'][key]:raise ValueError(f'Observation decoding mismatch {key}={error}')
  action,_=self.actor.predict(obs,deterministic=True)
  if action.shape!=(17,) or action.dtype!=np.float32 or not np.isfinite(action).all() or np.any(abs(action)>1):raise ValueError('Invalid analytical action')
  other=[i for i in range(17) if i not in c['ankle_action_columns']]
  if np.any(action[other]!=0):raise ValueError('Analytical action changed another joint')
  if signals['elapsed_s']<c['start_s'] and np.any(action!=0):self.pregate_nonzero+=1;raise ValueError('Feedback changed protected prefix')
  self.observations.append(obs.copy());self.actions.append(action.copy());self.signals.append([signals[k] for k in ('elapsed_s','torso_tilt_rad','torso_tilt_rate_rad_s','pelvis_tilt_rad','desired_pelvis_tilt_rad')])
  return action,None


class TargetObserver:
 """Observe original target pipeline once; never modify input/output or state."""
 def __init__(self,base):
  self.base=base;self.rows=[];self.clip_counts=np.zeros(31,dtype=int);self.rate_counts=np.zeros(31,dtype=int);self.max_clip=np.zeros(31);self.min_margin=np.full(31,np.inf)
 def __enter__(self):
  from x2_recovery.env import ControlledRecoveryEnv,controller_reference_targets,residual_phase_gate
  self.cls=ControlledRecoveryEnv;self.original=self.cls.step
  def observed(env,action):
   if env.env is not self.base:return self.original(env,action)
   c=env.control_config;e=env.env
   if c.mode!='reference_residual' or c.tracking_error_ratio is not None:raise ValueError('Unsupported target measurement pipeline')
   elapsed=float(e.data.time-e.episode_start_time);previous=env.target.copy()
   reference,_=controller_reference_targets(elapsed+e.control_dt,env.reference,c.reference_interpolation)
   mapped=env.policy_to_joint@np.asarray(action);gate=residual_phase_gate(c,elapsed)
   residual=env.residual_scale*mapped*gate;unclipped=reference+residual
   expected_requested=np.clip(unclipped,e.q_min,e.q_max)
   expected_adopted=previous+np.clip(expected_requested-previous,-env.rate*e.control_dt,env.rate*e.control_dt)
   result=self.original(env,action);info=result[4]['controller'];requested=np.asarray(info['requested_target_rad']);adopted=np.asarray(info['adopted_target_rad'])
   if not np.array_equal(requested,expected_requested) or not np.array_equal(adopted,expected_adopted):raise ValueError('Measured original target pipeline differs from read-only reconstruction')
   self.clip_counts+=unclipped!=requested;self.rate_counts+=requested!=adopted
   self.max_clip=np.maximum(self.max_clip,abs(unclipped-requested));self.min_margin=np.minimum(self.min_margin,np.minimum(adopted-e.q_min,e.q_max-adopted))
   self.rows.append((elapsed,unclipped.copy(),requested.copy(),adopted.copy(),residual.copy()))
   return result
  self.cls.step=observed;return self
 def __exit__(self,*args):self.cls.step=self.original;return False
 def summary(self):
  return dict(control_transitions=len(self.rows),joint_names=[r.joint_name for r in self.base.loaded.mapping],target_clip_count_per_joint=self.clip_counts.tolist(),
   target_rate_limit_count_per_joint=self.rate_counts.tolist(),max_target_clip_magnitude_rad=self.max_clip.tolist(),min_adopted_target_margin_rad=[float(x) if np.isfinite(x) else None for x in self.min_margin],
   exact_original_requested_and_adopted_checks=True,rule='Counts are exact control-transition comparisons; no invented1kHz duration.')
 def save(self,path):
  np.savez_compressed(path,elapsed_s=np.array([x[0] for x in self.rows]),unclipped_reference_plus_residual=np.array([x[1] for x in self.rows]),requested_target=np.array([x[2] for x in self.rows]),adopted_target=np.array([x[3] for x in self.rows]),gated_residual=np.array([x[4] for x in self.rows]))


def run_plan(root,planfile,output,helper,max_wall_seconds):
 from x2_recovery.env import ControlledRecoveryEnv,ControlConfig
 from x2_recovery.evaluate import prepare,physical_env,verify_policy
 from x2_recovery.motion_search import constraint_trial
 from x2_recovery.train import write_json,source_hashes
 plan=json.loads(planfile.read_text());inputs=(root/plan['input_directory']).resolve();output=output.resolve()
 if output.exists() or output.is_relative_to(inputs) or inputs.is_relative_to(output):raise ValueError('Output must be new and disjoint from frozen inputs')
 if not 0<max_wall_seconds<=600:raise ValueError('Scheduling budget must be in(0,600] seconds')
 if plan['schema']!='experimental-tail-ankle-feedback-list-v1' or not 1<=plan['planned']<=32 or plan['planned']!=len(plan['candidates']):raise ValueError('Invalid fixed feedback plan')
 if plan['pilot_sha256']!=sha(__file__) or plan['helper_sha256']!=sha(helper):raise ValueError('Formula/runner source differs from plan')
 if plan['motion_search_sha256']!=sha(root/'src/x2_recovery/x2_recovery/motion_search.py'):raise ValueError('Measurement source differs from plan')
 if digest(plan['candidates'])!=plan['candidate_list_sha256'] or len({x['name'] for x in plan['candidates']})!=plan['planned']:raise ValueError('Frozen candidate list differs or names are duplicated')
 calibrations=plan['candidates'][:len(plan['anchors'])]
 if [(x['anchor'],x['kp'],x['kd'],x['calibration']) for x in calibrations]!=[(label,0.,0.,True) for label in plan['anchors']]:raise ValueError('Every anchor requires its zero calibration first')
 for item in plan['candidates']:
  if item['anchor'] not in plan['profiles'] or not np.isfinite([item['kp'],item['kd'],item['cap_rad'],item['start_s']]).all() or not (0<=item['kp']<=.6 and 0<=item['kd']<=.1 and 0<item['cap_rad']<=.16 and 2.8<=item['start_s']<=3.5):raise ValueError('Candidate outside explicitly bounded experiment space')
  if digest(item['control_config'])!=item['control_config_sha256']:raise ValueError('Candidate configuration changed')
  parsed=ControlConfig.from_dict(item['control_config'])
  if parsed.residual_start_s!=item['start_s'] or parsed.residual_ramp_s!=item['ramp_s'] or item['ramp_s']!=.2:raise ValueError('Candidate gate differs')
 for name,identity in plan['sources'].items():
  if sha(root/name)!=identity:raise ValueError('Plan source changed:'+name)
 output.mkdir(parents=True);shutil.copyfile(planfile,output/'plan.json');shutil.copyfile(__file__,output/'tail_feedback_pilot.py');shutil.copyfile(helper,output/helper.name)
 formula=helper_module(helper);core=source_hashes();started=time.monotonic();owner=None;results=[];exceptions=[];fatal=None;input_hashes={}
 manifest=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(planfile),pilot_sha256=sha(__file__),helper_sha256=sha(helper),
  motion_search_sha256=plan['motion_search_sha256'],core_sources=core,controller=plan['controller'],input_directory=str(inputs),
  expected_checkpoint_sha256=plan['expected_checkpoint_sha256'],training=False,weight_transfer=False,ppo_control_inference=False,max_wall_seconds=max_wall_seconds,
  budget_scope=plan['bounds'],command=[sys.executable,*sys.argv]);write_json(output/'manifest.json',manifest)
 try:
  owner,frozen,original,prepared=prepare(inputs,expected_checkpoint_sha256=plan['expected_checkpoint_sha256']);base=physical_env(owner)
  input_hashes={str(path):sha(path) for path in prepared['input_sources'].values()}
  manifest.update(strict_input_identity=prepared['identity'],strict_saved_action_check=prepared['consistency'],input_sha256=prepared['hashes']);write_json(output/'manifest.json',manifest)
  for index,item in enumerate(plan['candidates']):
   if time.monotonic()-started>=max_wall_seconds:break
   if not all(sha(path)==expected for path,expected in input_hashes.items()):raise ValueError('Frozen input changed before trial')
   profile=plan['profiles'][item['anchor']];control=ControlConfig.from_dict(item['control_config']);preview=ControlledRecoveryEnv(base,control);c=dict(plan['interface'],residual_scale_rad=item['cap_rad'],start_s=item['start_s'],ramp_s=item['ramp_s'])
   scales=np.zeros(31);scales[c['ankle_joint_indices']]=c['residual_scale_rad']
   if not np.array_equal(preview.residual_scale,scales):raise ValueError('Actual controller residual authority differs')
   if base.config.joint_velocity_scale_rad_s!=c['joint_velocity_scale_rad_s'] or base.config.base_angular_scale_rad_s!=c['base_angular_scale_rad_s'] or base.config.episode_timeout_s!=c['episode_timeout_s']:raise ValueError('Observation scaling differs')
   feedback=ObservedFeedback(base,formula,c,item['kp'],item['kd']);directory=output/item['name'];directory.mkdir();result=None;target_observer=TargetObserver(base)
   try:
    with target_observer:
     result=constraint_trial(prepared['saved'],control,seed=plan['seed'],policy=feedback,base=base,zero_residual=False,comparison=plan['comparison'],name=item['name'])
    result.update(controller=plan['controller'],weight_transfer=False,training=False,ppo_control_inference=False,anchor=item['anchor'],kp=item['kp'],kd=item['kd'],cap_rad=item['cap_rad'],start_s=item['start_s'],calibration=item['calibration'],target_pipeline_probe=target_observer.summary(),
     observation_probe=dict(calls=len(feedback.observations),max_decoding_errors=feedback.errors,pregate_nonzero_actions=feedback.pregate_nonzero,
      signal_columns=['elapsed_s','torso_tilt_rad','torso_tilt_rate_rad_s','pelvis_tilt_rad','desired_pelvis_tilt_rad']))
    if len(feedback.observations)!=result['transitions'] or len(target_observer.rows)!=result['transitions']:raise ValueError('Predict/transition count mismatch')
    if item['kp']==item['kd']==0:
     result['zero_gain_baseline_equivalence']={k:result[k]==v for k,v in profile['baseline_hashes'].items()}
     if not all(result['zero_gain_baseline_equivalence'].values()):raise ValueError('Zero-gain baseline state/action/target identity differs')
    verify_policy(frozen,original);result['original_policy_state_unchanged']=True
    if asdict(base.config)!=asdict(prepared['config'].env):raise ValueError('Original physical configuration not restored')
    result['original_environment_config_restored']=True
    write_json(directory/'summary.json',result);results.append(result)
    with (output/'results.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['name','success','sim_seconds','feasible','physical_state_sha256','actual_target_sha256']}),flush=True)
   except Exception as exc:
    error=dict(index=index,name=item['name'],anchor=item['anchor'],kp=item['kp'],kd=item['kd'],cap_rad=item['cap_rad'],start_s=item['start_s'],calibration=item['calibration'],target_pipeline_probe=target_observer.summary(),type=type(exc).__name__,message=str(exc));exceptions.append(error);write_json(directory/'error.json',error)
    if result is not None:
     result.update(status='INVALID',validation_error=error);write_json(directory/'summary.json',result)
    # Do not proceed after a baseline or measurement/identity failure.
    raise
   finally:
    target_observer.save(directory/'target_pipeline.npz')
    np.savez_compressed(directory/'feedback_probe.npz',observations=np.asarray(feedback.observations,dtype=np.float32),actions=np.asarray(feedback.actions,dtype=np.float32),signals=np.asarray(feedback.signals,dtype=np.float64))
  verify_policy(frozen,original)
 except Exception as exc:fatal=dict(type=type(exc).__name__,message=str(exc))
 finally:
  if owner is not None:
   try:verify_policy(frozen,original)
   except Exception as exc:fatal=dict(type=type(exc).__name__,message=str(exc))
   finally:owner.close()
 unchanged=core==source_hashes() and plan['motion_search_sha256']==sha(root/'src/x2_recovery/x2_recovery/motion_search.py') and plan['pilot_sha256']==sha(__file__) and plan['helper_sha256']==sha(helper)
 inputs_unchanged=bool(input_hashes) and all(sha(path)==expected for path,expected in input_hashes.items())
 status='FAILED' if fatal or not unchanged or not inputs_unchanged else ('COMPLETE' if len(results)==plan['planned'] else 'BUDGET_STOP')
 summary=dict(status=status,planned=plan['planned'],completed=len(results),invalid=len(exceptions),not_executed=plan['planned']-len(results)-len(exceptions),
  successful=sum(x['success'] for x in results),feasible=sum(x['feasible'] for x in results),exception=fatal,exceptions=exceptions,
  source_unchanged=unchanged,inputs_unchanged=inputs_unchanged,wall_seconds=time.monotonic()-started,
  controller=plan['controller'],weight_transfer=False,training=False,ppo_control_inference=False,original_checkpoint_sha256_after=sha(inputs/'policy_final.zip'))
 write_json(output/'summary.json',summary);print(json.dumps(summary),flush=True)
 return 0 if status=='COMPLETE' else 1


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True,type=Path);p.add_argument('--helper',type=Path,default=Path(__file__).with_name('x2_tail_ankle_feedback.py'))
 sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('plan');q.add_argument('--source-plan',type=Path,default=Path('/tmp/x2-tail-feedback-g19-g23-plan.json'));q.add_argument('--output',required=True,type=Path);q.add_argument('--design',choices=['authority','fine-pd'],default='authority')
 q=sub.add_parser('run');q.add_argument('--plan',required=True,type=Path);q.add_argument('--output',required=True,type=Path);q.add_argument('--max-wall-seconds',type=float,default=600.)
 args=p.parse_args();root=args.repo.resolve();sys.path.insert(0,str(root/'src/x2_recovery'))
 if args.command=='plan':
  plan=build_plan(root,args.helper.resolve(),args.source_plan.resolve(),args.design)
  with args.output.open('x') as f:json.dump(plan,f,indent=2,allow_nan=False);f.write('\n')
  print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),planned=plan['planned'],pilot_sha256=plan['pilot_sha256'],helper_sha256=plan['helper_sha256'])));return 0
 return run_plan(root,args.plan.resolve(),args.output,args.helper.resolve(),args.max_wall_seconds)


if __name__=='__main__':raise SystemExit(main())
