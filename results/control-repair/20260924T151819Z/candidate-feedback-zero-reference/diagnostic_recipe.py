"""Two newly authorized full deterministic diagnostic episodes; frozen source actor unused."""
from pathlib import Path
import importlib.util,json,hashlib,shutil,csv,time
import numpy as np
from x2_recovery.evaluate import prepare,physical_env,verify_policy
from x2_recovery.env import ControlledRecoveryEnv,ControlConfig
from x2_recovery.constraint_audit import run_diagnostic,discount_rewards
from x2_recovery.train import write_json
r=Path('/home/lang/RL-AgiBot-X2');run=r/'results/control-repair/20260924T151819Z';src=run/'tail-feedback-g33-ik';plan=json.load(open(src/'plan.json'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pilot=load(src/'tail_feedback_pilot.py','frozen_analytical_runner');formula=load(src/'x2_tail_ankle_feedback.py','frozen_analytical_formula');assert sha(src/'tail_feedback_pilot.py')==plan['pilot_sha256'];assert sha(src/'x2_tail_ankle_feedback.py')==plan['helper_sha256'];inputs=r/plan['input_directory'];original_input_hashes={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*')if p.is_file()};started=time.monotonic();rows=[];detail={};summaries={}
owner,frozen,original,prepared=prepare(inputs,expected_checkpoint_sha256=plan['expected_checkpoint_sha256']);base=physical_env(owner)
try:
 for index,outname in [(18,'candidate-feedback-feasible'),(1,'candidate-feedback-zero-reference')]:
  item=plan['candidates'][index];cfg=ControlConfig.from_dict(item['control_config']);env=ControlledRecoveryEnv(base,cfg);interface=dict(plan['interface'],residual_scale_rad=item['cap_rad'],start_s=item['start_s'],ramp_s=item['ramp_s']);feedback=pilot.ObservedFeedback(base,formula,interface,item['kp'],item['kd']);output=run/outname
  expected=json.load(open(src/item['name']/'summary.json'));source=dict(controller='experimental_IK_reference_torso_ankle_feedback',ppo_control_inference=False,training=False,weight_transfer=False,original_checkpoint_sha256=plan['expected_checkpoint_sha256'],plan_sha256=sha(src/'plan.json'),selected_candidate=item,pilot_sha256=plan['pilot_sha256'],helper_sha256=plan['helper_sha256'],original_input_sha256=original_input_hashes)
  observer=pilot.TargetObserver(base)
  recovered=output.exists()
  if recovered:
   assert index==18 and (output/'physical_trace.npz').is_file()
   s=json.load(open(output/'summary.json'));assert s['status']=='COMPLETE' and s['source_identity']['selected_candidate']['name']==item['name']
  else:
   with observer:
    s=run_diagnostic(env,feedback,seed=plan['seed'],output=output,zero_residual=False,gamma=.999,source_identity=source,trace_level='full')
  s['controller']='experimental_IK_reference_torso_ankle_feedback';s['ppo_control_inference']=False;s['training']=False;s['weight_transfer']=False;s['analytical_gains']={'kp':item['kp'],'kd':item['kd']};s['zero_feedback_reference']=index==1;s['target_pipeline_probe']=json.load(open(src/item['name']/'summary.json'))['target_pipeline_probe'] if recovered else observer.summary()
  if not recovered:observer.save(output/'target_pipeline.npz')
  if not recovered:np.savez_compressed(output/'feedback_probe.npz',observations=np.asarray(feedback.observations,dtype=np.float32),actions=np.asarray(feedback.actions,dtype=np.float32),signals=np.asarray(feedback.signals,dtype=np.float64))
  n=np.load(output/'physical_trace.npz');c=np.load(output/'control_trace.npz');mask=n['is_recovery'];state=hashlib.sha256()
  for q,v in zip(n['qpos'][mask],n['qvel'][mask]):state.update(np.asarray(q,dtype='<f8').tobytes());state.update(np.asarray(v,dtype='<f8').tobytes())
  hashes=dict(physical_state_sha256=state.hexdigest(),actual_target_sha256=hashlib.sha256(np.asarray(n['target_rad'][mask],dtype='<f8').tobytes()).hexdigest(),policy_action_sha256=hashlib.sha256(np.asarray(c['action'],dtype='<f8').tobytes()).hexdigest());checks={k:hashes[k]==expected[k] for k in hashes}
  prior=np.load(src/item['name']/'feedback_probe.npz');prior_target=np.load(src/item['name']/'target_pipeline.npz');current_target=np.load(output/'target_pipeline.npz');checks.update(original_observations_exact=np.array_equal(c['observation'],prior['observations']),original_actions_exact=np.array_equal(c['action'],prior['actions']),original_adopted_target_exact=np.array_equal(prior_target['adopted_target'],current_target['adopted_target']),original_requested_target_exact=np.array_equal(prior_target['requested_target'],current_target['requested_target']))
  assert all(checks.values()),checks
  verify_policy(frozen,original);s.update(streaming_equivalence=checks,streaming_hashes=hashes,original_policy_state_unchanged=True,observation_probe=dict(max_decoding_errors=(expected['observation_probe']['max_decoding_errors'] if recovered else feedback.errors),pregate_nonzero=(expected['observation_probe']['pregate_nonzero_actions'] if recovered else feedback.pregate_nonzero)));shutil.copyfile(__file__,output/('diagnostic_resume_recipe.py'if recovered else'diagnostic_recipe.py'));shutil.copyfile(src/'tail_feedback_pilot.py',output/'tail_feedback_pilot.py');shutil.copyfile(src/'x2_tail_ankle_feedback.py',output/'x2_tail_ankle_feedback.py');(output/'control_config.json').write_text(json.dumps(item['control_config'],indent=2)+'\n');s['artifacts_sha256']={p.name:sha(p) for p in output.iterdir()if p.is_file()and p.name!='summary.json'};write_json(output/'summary.json',s);summaries[outname]=dict(success=s['success'],sim_seconds=s['sim_duration_s'],hold_s=s['maximum_stable_hold_s'],constraint_envelope_pass=s['constraint_envelope_pass'],streaming_equivalence=checks)
  originalterms={str(k):c['reward_terms'][:,j] for j,k in enumerate(c['reward_term_names'])};assert np.allclose(sum(originalterms.values()),c['reward'],rtol=1e-12,atol=1e-12)
  for version,alpha in [('reference-balance-v1',1.),('reference-balance-density10-v1',.1)]:
   terms={k:v*(alpha if k in ('pose_guide','head_track','balance','standing') else 1.)for k,v in originalterms.items()};rew=sum(terms.values());rr=discount_rewards(rew,terms,.999,terminated=bool(c['terminated'][-1]),truncated=bool(c['truncated'][-1]));detail[outname+'/'+version]=rr;rows.append(dict(controller=outname,reward_version=version,success=s['success'],transitions=len(rew),sim_seconds=s['sim_duration_s'],raw_return=rr['raw_return'],discounted_prefix=rr['observed_discounted_prefix'],terminated=bool(c['terminated'][-1]),truncated=bool(c['truncated'][-1]),critic_tail='unavailable_not_added'))
  print(json.dumps(dict(output=outname,**summaries[outname],reward_rows=rows[-2:])),flush=True)
finally:
 verify_policy(frozen,original);owner.close()
assert original_input_hashes=={str(p.relative_to(inputs)):sha(p)for p in inputs.rglob('*')if p.is_file()}
bounds={}
for version,alpha in [('reference-balance-v1',1.),('reference-balance-density10-v1',.1)]:
 upper=.02*14*alpha*sum(.999**np.arange(1000));bounds[version]=dict(positive_density_bound_per_second=14*alpha,max_transition_positive=.28*alpha,max_20s_failure_discounted_prefix_ignoring_negative=upper,max_20s_success_discounted_return_ignoring_negative=upper+50*.999**999,one_step_delay_immediately_before_success_upper_advantage=.28*alpha-(1-.999)*50,immediate_safety_failure=-2.,max_nonterminal_cycles_included=True,scope='Mathematical upper bounds, not actual physics;no assumed critic bootstrap')
result=dict(schema='feasible-feedback-reward-audit-v1',episodes=summaries,exact_reward_rankings=detail,rows=rows,theoretical_bounds=bounds,original_inputs_unchanged=True,original_policy_unchanged=True,training=False,wall_seconds=time.monotonic()-started,gamma_semantics='gamma=.999 per policy transition; shortened final step retains one exponent. No missing future reward after termination; timeout prefix only; final critic not used.',interpretation='Analytical feedback plus new IK reference, not PPO. Both are same physical model/reset/reward version; control differences are explicit. Reduced-density values are offline rescoring, not learned results.')
output=run/'candidate-feedback-feasible';(output/'reward_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
with (output/'reward_ranking.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(result,indent=2))
