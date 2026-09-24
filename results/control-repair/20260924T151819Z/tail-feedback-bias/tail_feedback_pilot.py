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

def build_plan(root,helper,source_plan):
 from x2_recovery.env import ControlConfig
 source=json.loads(source_plan.read_text());folder=source_plan.parent;summary_file=folder/'summary.json';results_file=folder/'results.jsonl'
 summary=json.loads(summary_file.read_text())
 if source['design']!='fine-pd' or source['anchors']!=['g19-29'] or summary['status']!='COMPLETE' or summary['invalid']!=0:raise ValueError('Require completed valid g19 fine-PD source')
 rows=[json.loads(line) for line in results_file.read_text().splitlines()]
 pairs=[(.12,.035),(.15,.035),(.18,.055),(.22,.065)]
 selected={pair:next(row for row in rows if (row['kp'],row['kd'])==pair) for pair in pairs}
 parent_candidates={pair:next(row for row in source['candidates'] if (row['kp'],row['kd'])==pair) for pair in pairs}
 plan=deepcopy(source);plan.update(schema='experimental-tail-ankle-bias-feedback-v1',created_utc=datetime.now(timezone.utc).isoformat(),
  pilot_sha256=sha(__file__),helper_sha256=sha(helper),helper_filename=helper.name,controller='experimental_reference_torso_bias_ankle_feedback',
  source_helper_sha256=source['helper_sha256'],source_pilot_sha256=source['pilot_sha256'],source_fine_pd_plan=dict(path=str(source_plan.relative_to(root)),sha256=sha(source_plan)),design='explicit_torso_tilt_bias')
 def entry(pair,bias,calibration,index):
  row=parent_candidates[pair];cfg=deepcopy(row['control_config']);ControlConfig.from_dict(cfg)
  if row['control_config_sha256']!=digest(cfg) or selected[pair]['control_sha256']!=digest(cfg):raise ValueError('Parent full control identity mismatch')
  return dict(name=f'bias-{index:02d}-g19-29-kp{0. if calibration else pair[0]:g}-kd{0. if calibration else pair[1]:g}-theta{bias:g}',anchor='g19-29',
   kp=0. if calibration else pair[0],kd=0. if calibration else pair[1],theta_des_rad=bias,cap_rad=.08,start_s=3.5,ramp_s=.2,
   calibration=calibration,bias_zero_calibration=bool(not calibration and bias==0.),control_config=cfg,control_config_sha256=digest(cfg),
   control_changes=row['control_changes'],parent_candidate=selected[pair]['name'],parent_control_sha256=selected[pair]['control_sha256'],
   expected_bias_zero_hashes={k:selected[pair][k] for k in ['physical_state_sha256','policy_action_sha256','actual_target_sha256']} if not calibration and bias==0 else None)
 candidates=[entry(pairs[0],0.,True,0)]
 # These four calibrations are the theta_des=0 members of the requested20 grid.
 for pair in pairs:candidates.append(entry(pair,0.,False,len(candidates)))
 for pair in pairs:
  for bias in [-.03,-.06,-.10,-.15]:candidates.append(entry(pair,bias,False,len(candidates)))
 for path in [source_plan,summary_file,results_file]:plan['sources'][str(path.relative_to(root))]=sha(path)
 plan.update(candidates=candidates,planned=21,candidate_list_sha256=digest(candidates),
  factors=dict(kp_kd_pairs=pairs,theta_des_rad=[-.03,-.06,-.10,-.15,0.],cap_rad=.08,start_s=3.5,ramp_s=.2),
  candidate_order='One zero-gain baseline, then four theta_des=0 cases against the exact measured fine-PD state/action/target hashes, then fixed gain-pair outer and[-.03,-.06,-.10,-.15] bias inner. These are21 total cases, not extra zero-bias episodes.',
  formula='Unchanged149-observation torso tilt/rate reconstruction. After existing gate onset, delta=clip(Kp*(theta-theta_des)+Kd*theta_dot,+/-0.08). theta_des is an explicit per-candidate constant; no new memory, sensor or state estimate.',
  physical_risk='Negative desired torso sagittal tilt is a hypothesis to shift balance away from observed forward tipping. It can worsen rise, support or contacts. Original success and full-episode constraints remain separate mandatory measurements.',
  motion_search_sha256=sha(root/'src/x2_recovery/x2_recovery/motion_search.py'))
 return plan


class ObservedFeedback:
 def __init__(self,base,formula,interface,kp,kd):
  self.base=base;self.c=interface;self.actor=formula.TailAnkleFeedback(interface['q_lower'],interface['q_upper'],interface['waist_indices'],kp,kd,
   cap_rad=interface['residual_scale_rad'],start_s=interface['start_s'],episode_timeout_s=interface['episode_timeout_s'],joint_velocity_scale=interface['joint_velocity_scale_rad_s'],base_angular_scale=interface['base_angular_scale_rad_s'],theta_des_rad=interface['theta_des_rad'])
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
  self.observations.append(obs.copy());self.actions.append(action.copy());self.signals.append([signals[k] for k in ('elapsed_s','torso_tilt_rad','torso_tilt_rate_rad_s','pelvis_tilt_rad','desired_pelvis_tilt_rad')]+[self.c['theta_des_rad']])
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
 if plan['schema']!='experimental-tail-ankle-bias-feedback-v1' or not 1<=plan['planned']<=32 or plan['planned']!=len(plan['candidates']):raise ValueError('Invalid fixed feedback plan')
 if plan['pilot_sha256']!=sha(__file__) or plan['helper_sha256']!=sha(helper):raise ValueError('Formula/runner source differs from plan')
 if plan['motion_search_sha256']!=sha(root/'src/x2_recovery/x2_recovery/motion_search.py'):raise ValueError('Measurement source differs from plan')
 if digest(plan['candidates'])!=plan['candidate_list_sha256'] or len({x['name'] for x in plan['candidates']})!=plan['planned']:raise ValueError('Frozen candidate list differs or names are duplicated')
 calibrations=plan['candidates'][:len(plan['anchors'])]
 if [(x['anchor'],x['kp'],x['kd'],x['calibration']) for x in calibrations]!=[(label,0.,0.,True) for label in plan['anchors']]:raise ValueError('Every anchor requires its zero calibration first')
 for item in plan['candidates']:
  if not np.isfinite(item['theta_des_rad']) or item['theta_des_rad'] not in [-.03,-.06,-.10,-.15,0.] or (item['cap_rad'],item['start_s'],item['ramp_s'])!=(.08,3.5,.2):raise ValueError('Bias candidate differs from frozen authority/timing')
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
   profile=plan['profiles'][item['anchor']];control=ControlConfig.from_dict(item['control_config']);preview=ControlledRecoveryEnv(base,control);c=dict(plan['interface'],residual_scale_rad=item['cap_rad'],start_s=item['start_s'],ramp_s=item['ramp_s'],theta_des_rad=item['theta_des_rad'])
   scales=np.zeros(31);scales[c['ankle_joint_indices']]=c['residual_scale_rad']
   if not np.array_equal(preview.residual_scale,scales):raise ValueError('Actual controller residual authority differs')
   if base.config.joint_velocity_scale_rad_s!=c['joint_velocity_scale_rad_s'] or base.config.base_angular_scale_rad_s!=c['base_angular_scale_rad_s'] or base.config.episode_timeout_s!=c['episode_timeout_s']:raise ValueError('Observation scaling differs')
   feedback=ObservedFeedback(base,formula,c,item['kp'],item['kd']);directory=output/item['name'];directory.mkdir();result=None;target_observer=TargetObserver(base)
   try:
    with target_observer:
     result=constraint_trial(prepared['saved'],control,seed=plan['seed'],policy=feedback,base=base,zero_residual=False,comparison=plan['comparison'],name=item['name'])
    result.update(controller=plan['controller'],weight_transfer=False,training=False,ppo_control_inference=False,anchor=item['anchor'],kp=item['kp'],kd=item['kd'],cap_rad=item['cap_rad'],start_s=item['start_s'],calibration=item['calibration'],theta_des_rad=item['theta_des_rad'],target_pipeline_probe=target_observer.summary(),
     observation_probe=dict(calls=len(feedback.observations),max_decoding_errors=feedback.errors,pregate_nonzero_actions=feedback.pregate_nonzero,
      signal_columns=['elapsed_s','torso_tilt_rad','torso_tilt_rate_rad_s','pelvis_tilt_rad','desired_pelvis_tilt_rad','theta_des_rad']))
    if len(feedback.observations)!=result['transitions'] or len(target_observer.rows)!=result['transitions']:raise ValueError('Predict/transition count mismatch')
    if item['kp']==item['kd']==0:
     result['zero_gain_baseline_equivalence']={k:result[k]==v for k,v in profile['baseline_hashes'].items()}
     if not all(result['zero_gain_baseline_equivalence'].values()):raise ValueError('Zero-gain baseline state/action/target identity differs')
    if item['bias_zero_calibration']:
     result['bias_zero_parent_equivalence']={key:result[key]==expected for key,expected in item['expected_bias_zero_hashes'].items()}
     if not all(result['bias_zero_parent_equivalence'].values()):raise ValueError('Zero-bias state/action/target differs from measured parent')
    verify_policy(frozen,original);result['original_policy_state_unchanged']=True
    if asdict(base.config)!=asdict(prepared['config'].env):raise ValueError('Original physical configuration not restored')
    result['original_environment_config_restored']=True
    write_json(directory/'summary.json',result);results.append(result)
    with (output/'results.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['name','success','sim_seconds','feasible','physical_state_sha256','actual_target_sha256']}),flush=True)
   except Exception as exc:
    error=dict(index=index,name=item['name'],anchor=item['anchor'],kp=item['kp'],kd=item['kd'],cap_rad=item['cap_rad'],start_s=item['start_s'],calibration=item['calibration'],theta_des_rad=item['theta_des_rad'],target_pipeline_probe=target_observer.summary(),type=type(exc).__name__,message=str(exc));exceptions.append(error);write_json(directory/'error.json',error)
    if result is not None:
     result.update(status='INVALID',validation_error=error);write_json(directory/'summary.json',result)
    # Do not proceed after a baseline or measurement/identity failure.
    raise
   finally:
    target_observer.save(directory/'target_pipeline.npz')
    np.savez_compressed(directory/'feedback_probe.npz',observations=np.asarray(feedback.observations,dtype=np.float32),actions=np.asarray(feedback.actions,dtype=np.float32),signals=np.asarray(feedback.signals,dtype=np.float64),theta_des_rad=np.full(len(feedback.signals),item['theta_des_rad']))
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True,type=Path);p.add_argument('--helper',type=Path,default=Path(__file__).with_name('x2_tail_ankle_bias_feedback.py'))
 sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('plan');q.add_argument('--source-plan',type=Path,default=Path('/home/lang/RL-AgiBot-X2/results/control-repair/20260924T151819Z/tail-feedback-fine-pd/plan.json'));q.add_argument('--output',required=True,type=Path)
 q=sub.add_parser('run');q.add_argument('--plan',required=True,type=Path);q.add_argument('--output',required=True,type=Path);q.add_argument('--max-wall-seconds',type=float,default=600.)
 args=p.parse_args();root=args.repo.resolve();sys.path.insert(0,str(root/'src/x2_recovery'))
 if args.command=='plan':
  plan=build_plan(root,args.helper.resolve(),args.source_plan.resolve())
  with args.output.open('x') as f:json.dump(plan,f,indent=2,allow_nan=False);f.write('\n')
  print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),planned=plan['planned'],pilot_sha256=plan['pilot_sha256'],helper_sha256=plan['helper_sha256'])));return 0
 return run_plan(root,args.plan.resolve(),args.output,args.helper.resolve(),args.max_wall_seconds)


if __name__=='__main__':raise SystemExit(main())
