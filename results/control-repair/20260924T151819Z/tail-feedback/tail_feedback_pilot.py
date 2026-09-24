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

def build_plan(root,helper,specfile,anchors):
 from x2_recovery.env import ControlConfig,INDEPENDENT_LEG_ACTION_NAMES,independent_legs_action_matrix
 run=root/'results/control-repair/20260924T151819Z';inputs=root/'results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run'
 saved=json.loads((inputs/'resolved_config.json').read_text());protocol=json.loads((run/'manifest.json').read_text());spec=json.loads(specfile.read_text())
 if sha(helper)!=spec['helper_sha256']:raise ValueError('Pure feedback helper differs from verified specification')
 gains=[(kp,kd) for kp in [0.,.15,.3,.6] for kd in [0.,.03,.08]]
 if gains!=[(row['kp'],row['kd']) for row in spec['gain_pairs']]:raise ValueError('Gain grid differs from static specification')
 physical=saved['environment']['base'];names=[r['joint_name'] for r in physical['mapping']]
 indices=[names.index(side+'_ankle_pitch_joint') for side in ('left','right')]
 columns=[list(INDEPENDENT_LEG_ACTION_NAMES).index(side+'_ankle_pitch') for side in ('left','right')]
 if columns!=[4,10]:raise ValueError('Existing analytical helper action map differs')
 matrix=independent_legs_action_matrix(names)
 for joint,column in zip(indices,columns):
  if matrix[joint,column]!=1 or np.count_nonzero(matrix[:,column])!=1:raise ValueError('Unexpected ankle action transmission')
 bounds=np.array([physical['q_min_rad'],physical['q_max_rad']]).T
 if not np.array_equal(bounds,np.asarray(spec['joint_bounds'])):raise ValueError('Joint bounds differ from formula specification')
 waist=[names.index('waist_'+part+'_joint') for part in ('yaw','pitch','roll')]
 if waist!=spec['waist_indices_yaw_pitch_roll']:raise ValueError('Waist order mismatch')
 sources=[inputs/'resolved_config.json',run/'manifest.json'];base_profiles={}
 for label in anchors:
  generation,index=(11,21) if label=='g11-21' else (13,10)
  planfile=run/f'generation-{generation:02d}'/'plan.json';resultfile=planfile.with_name('candidates.jsonl');sources.extend([planfile,resultfile])
  row=json.loads(planfile.read_text())['candidates'][index];original=row['control_config'];cfg=deepcopy(original)
  measured=next(json.loads(line) for line in resultfile.read_text().splitlines() if json.loads(line)['index']==index)
  if digest(original)!=measured['control_sha256']:raise ValueError('Anchor result/config identity mismatch')
  cfg['residual_start_s']=3.5;cfg['residual_ramp_s']=.2;cfg['residual_joint_multipliers']=[0.]*31
  delta=np.minimum(np.max(abs(np.asarray(physical['effort_limits_Nm'])),axis=1)/physical['kp_Nm_rad'],np.ptp(bounds,axis=1)*.25)*cfg['delta_multiplier']
  for joint in indices:cfg['residual_joint_multipliers'][joint]=float(.08/(delta[joint]*cfg['residual_multiplier']))
  ControlConfig.from_dict(cfg)
  changes={key:dict(original=original[key],experiment=cfg[key]) for key in cfg if cfg[key]!=original[key]}
  if set(changes)!={'residual_start_s','residual_ramp_s','residual_joint_multipliers'}:raise ValueError('Unexpected deployment differences')
  base_profiles[label]=dict(source_candidate=row['name'],original_control_sha256=digest(original),experimental_control_sha256=digest(cfg),
   control_config=cfg,control_changes=changes,baseline_hashes={key:measured[key] for key in ['physical_state_sha256','policy_action_sha256','actual_target_sha256']})
 candidates=[dict(name=f'tail-feedback-{label}-kp{kp:g}-kd{kd:g}',anchor=label,kp=kp,kd=kd) for kp,kd in gains for label in anchors]
 return dict(schema='experimental-tail-ankle-feedback-v1',created_utc=datetime.now(timezone.utc).isoformat(),controller='experimental_reference_torso_ankle_feedback',
  development_only=True,deployable_policy=False,weight_transfer=False,training=False,ppo_control_inference=False,
  input_directory=str(inputs.relative_to(root)),expected_checkpoint_sha256=protocol['checkpoint_sha256'],seed=protocol['development_seeds'][0],comparison=protocol['comparison'],
  anchors=list(anchors),profiles=base_profiles,candidates=candidates,planned=len(candidates),gain_pairs=gains,
  candidate_order='Gain pair outer order, anchor inner order; both zero-gain baseline validations precede any nonzero gain. No resampling.',
  helper_sha256=sha(helper),helper_filename=helper.name,formula_spec_sha256=sha(specfile),formula=spec['formula'],pilot_sha256=sha(__file__),
  interface=dict(q_lower=physical['q_min_rad'],q_upper=physical['q_max_rad'],waist_indices=waist,ankle_joint_indices=indices,ankle_action_columns=columns,
   joint_velocity_scale_rad_s=saved['training']['env']['joint_velocity_scale_rad_s'],base_angular_scale_rad_s=saved['training']['env']['base_angular_scale_rad_s'],
   episode_timeout_s=saved['training']['env']['episode_timeout_s'],residual_scale_rad=.08,start_s=3.5,ramp_s=.2,
   assertion_tolerances=dict(q_rad=1e-6,dq_rad_s=1e-5,pelvis_gravity=1e-6,pelvis_angular_rad_s=1e-5,elapsed_s=2e-6,torso_gravity=2e-6)),
  sources={str(path.relative_to(root)):sha(path) for path in sources},
  motion_search_sha256=sha(root/'src/x2_recovery/x2_recovery/motion_search.py'),
  measurement='Existing constraint_trial / StreamingConstraintObserver, every physical step, original complete episode and guards. No extra active forward or state writes. Pre-control live reads only validate observation decoding, never enter the feedback formula.',
  semantics='Existing controller applies its own residual phase gate once, legal target clipping and target rates, original PD. Only two ankle pitch actions may be nonzero; exactly zero before3.5s. Original PPO is strict-loaded and saved-probe verified only; analytical controller never calls it.',
  bounds='Scheduling budget at most600s; an in-flight original full episode is not cut short. Exceptions recorded separately, no retry or replacement.',
  limitation='Analytical development experiment, not PPO gain or a deployed policy. Tail action cannot repair earlier constraint violations. Dynamic behavior may invalidate fixed-foot sign assumptions.')


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


def run_plan(root,planfile,output,helper,max_wall_seconds):
 from x2_recovery.env import ControlledRecoveryEnv,ControlConfig
 from x2_recovery.evaluate import prepare,physical_env,verify_policy
 from x2_recovery.motion_search import constraint_trial
 from x2_recovery.train import write_json,source_hashes
 plan=json.loads(planfile.read_text());inputs=(root/plan['input_directory']).resolve();output=output.resolve()
 if output.exists() or output.is_relative_to(inputs) or inputs.is_relative_to(output):raise ValueError('Output must be new and disjoint from frozen inputs')
 if not 0<max_wall_seconds<=600:raise ValueError('Scheduling budget must be in(0,600] seconds')
 if plan['schema']!='experimental-tail-ankle-feedback-v1' or plan['planned'] not in [12,24]:raise ValueError('Invalid fixed feedback plan')
 if plan['pilot_sha256']!=sha(__file__) or plan['helper_sha256']!=sha(helper):raise ValueError('Formula/runner source differs from plan')
 if plan['motion_search_sha256']!=sha(root/'src/x2_recovery/x2_recovery/motion_search.py'):raise ValueError('Measurement source differs from plan')
 expected=[(label,kp,kd) for kp,kd in [(kp,kd) for kp in [0.,.15,.3,.6] for kd in [0.,.03,.08]] for label in plan['anchors']]
 if [(x['anchor'],x['kp'],x['kd']) for x in plan['candidates']]!=expected:raise ValueError('Candidate order differs from fixed grid')
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
   profile=plan['profiles'][item['anchor']];control=ControlConfig.from_dict(profile['control_config']);preview=ControlledRecoveryEnv(base,control);c=plan['interface']
   scales=np.zeros(31);scales[c['ankle_joint_indices']]=c['residual_scale_rad']
   if not np.array_equal(preview.residual_scale,scales):raise ValueError('Actual controller residual authority differs')
   if base.config.joint_velocity_scale_rad_s!=c['joint_velocity_scale_rad_s'] or base.config.base_angular_scale_rad_s!=c['base_angular_scale_rad_s'] or base.config.episode_timeout_s!=c['episode_timeout_s']:raise ValueError('Observation scaling differs')
   feedback=ObservedFeedback(base,formula,c,item['kp'],item['kd']);directory=output/item['name'];directory.mkdir();result=None
   try:
    result=constraint_trial(prepared['saved'],control,seed=plan['seed'],policy=feedback,base=base,zero_residual=False,comparison=plan['comparison'],name=item['name'])
    result.update(controller=plan['controller'],weight_transfer=False,training=False,ppo_control_inference=False,anchor=item['anchor'],kp=item['kp'],kd=item['kd'],
     observation_probe=dict(calls=len(feedback.observations),max_decoding_errors=feedback.errors,pregate_nonzero_actions=feedback.pregate_nonzero,
      signal_columns=['elapsed_s','torso_tilt_rad','torso_tilt_rate_rad_s','pelvis_tilt_rad','desired_pelvis_tilt_rad']))
    if len(feedback.observations)!=result['transitions']:raise ValueError('Predict/transition count mismatch')
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
    error=dict(index=index,name=item['name'],anchor=item['anchor'],kp=item['kp'],kd=item['kd'],type=type(exc).__name__,message=str(exc));exceptions.append(error);write_json(directory/'error.json',error)
    if result is not None:
     result.update(status='INVALID',validation_error=error);write_json(directory/'summary.json',result)
    # Do not proceed after a baseline or measurement/identity failure.
    raise
   finally:
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
 sub=p.add_subparsers(dest='command',required=True);q=sub.add_parser('plan');q.add_argument('--spec',type=Path,default=Path('/tmp/x2-tail-ankle-feedback-spec.json'));q.add_argument('--output',required=True,type=Path);q.add_argument('--anchors',nargs='+',choices=['g11-21','g13-10'],default=['g11-21','g13-10'])
 q=sub.add_parser('run');q.add_argument('--plan',required=True,type=Path);q.add_argument('--output',required=True,type=Path);q.add_argument('--max-wall-seconds',type=float,default=600.)
 args=p.parse_args();root=args.repo.resolve();sys.path.insert(0,str(root/'src/x2_recovery'))
 if args.command=='plan':
  if len(set(args.anchors))!=len(args.anchors):raise ValueError('Duplicate anchors')
  plan=build_plan(root,args.helper.resolve(),args.spec.resolve(),args.anchors)
  with args.output.open('x') as f:json.dump(plan,f,indent=2,allow_nan=False);f.write('\n')
  print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),planned=plan['planned'],pilot_sha256=plan['pilot_sha256'],helper_sha256=plan['helper_sha256'])));return 0
 return run_plan(root,args.plan.resolve(),args.output,args.helper.resolve(),args.max_wall_seconds)


if __name__=='__main__':raise SystemExit(main())
