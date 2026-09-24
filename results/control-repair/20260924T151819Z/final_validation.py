#!/usr/bin/env python3
"""Thin final protocol runner: existing strict evaluation and paired reset execution.

No training, installation, source changes, extra reset/step or identity bypass.
Source the selected installation before invocation from any working directory.
"""
import argparse,csv,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

KEYS=('peak_ground_vertical_N','peak_ground_sum_norm_N','floor_penetration_m','self_penetration_m','max_joint_speed_rad_s')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 from x2_recovery.train import write_json
 write_json(path,value)

def impact(values,comparison):
 if set(KEYS)-values.keys() or set(KEYS)-comparison.keys():raise ValueError('Missing impact metrics')
 return {key:values[key]<=comparison[key]+1e-10 for key in KEYS}

def protocol(args):
 p=Path(args.protocol).resolve();spec=json.loads(p.read_text());out=Path(args.output).resolve();inputs=Path(args.training_run).resolve()
 if out.exists() or out.is_relative_to(inputs) or inputs.is_relative_to(out):raise ValueError('Output must be new and disjoint from inputs')
 target=spec['evaluation_target'];comparison=spec['comparison']
 if target['fixed_five']['seeds']!=list(range(26092550,26092555)):raise ValueError('Unexpected preregistered five seeds')
 held=target['held_out']
 if held['candidate_seeds']!=list(range(26092500,26092540)) or held['pairs']!=20 or held['maximum_reset_candidates']!=40 or held['perturb_rad']!=.002:raise ValueError('Unexpected heldout protocol')
 for key in KEYS:
  if not isinstance(comparison[key],(int,float)) or not comparison[key]>0:raise ValueError('Invalid comparison')
 return p,spec,out,inputs

def formal(args):
 from x2_recovery import evaluate
 from x2_recovery.motion_search import StreamingConstraintObserver
 p,spec,out,inputs=protocol(args);rows=[];active=[None];owner=[None]
 original_prepare=evaluate.prepare
 def prepare(*aa,**kw):
  env,model,weights,prepared=original_prepare(*aa,**kw);base=evaluate.physical_env(env);owner[0]=env
  reset_original,step_original=env.reset,env.step
  def reset(*a,**k):
   if active[0] is not None:raise ValueError('Unclosed observer before episode')
   if not rows:
    write(out/'validation_protocol.json',dict(protocol=spec['evaluation_target']['fixed_five'],comparison=spec['comparison'],protocol_path=str(p),protocol_sha256=sha(p),runner_sha256=sha(__file__),model_loaded_by='unchanged evaluate.prepare',measurement='StreamingConstraintObserver: post-integration q/dq; integration-solve forces/contacts; reset/recovery separate; no extra step/forward.'))
   obs=StreamingConstraintObserver(base);obs.__enter__();active[0]=obs
   result=reset_original(*a,**k);obs.phase='recovery';obs.validation_seed=k.get('seed');return result
  def step(action):
   result=step_original(action)
   if result[2] or result[3]:
    obs=active[0];r=obs.summary('recovery');z=obs.summary('reset');info=result[4]
    if r['samples']!=base._physics_steps:raise ValueError('Physical sample count mismatch')
    envelope=bool(r['max_joint_excess_rad']<=.0001+1e-10 and r['max_actuation_limit_ratio']<=1+1e-10 and r['velocity_limits_complete'] and r['max_velocity_ratio']<=1+1e-10)
    checks=impact(r,spec['comparison']);passed=all(checks.values())
    row=dict(episode=len(rows)+1,seed=obs.validation_seed,success=bool(info['is_success']),constraint_envelope_pass=envelope,impact_pass=passed,admissible_recovery=bool(info['is_success'] and envelope and passed),impact_checks=checks,sim_duration_s=float(info['elapsed_sim_s']),maximum_hold_s=obs.max_hold,physical_state_sha256=obs.state_hash.hexdigest(),actual_target_sha256=obs.target_hash.hexdigest(),reset_settling=z,recovery=r)
    rows.append(row);obs.__exit__(None,None,None);active[0]=None
    write(out/'constraint_episodes.json',rows)
   return result
  env.reset,env.step=reset,step
  return env,model,weights,prepared
 try:
  with patch.object(evaluate,'prepare',prepare):
   measured=evaluate.run(inputs,spec['evaluation_target']['fixed_five']['seeds'],out,command=[sys.executable,*sys.argv],episode_wall_seconds=args.episode_wall_seconds,expected_checkpoint_sha256=args.expected_checkpoint_sha256)
 finally:
  if active[0] is not None:active[0].__exit__(*sys.exc_info());active[0]=None
 if len(rows)!=5 or [r['seed'] for r in rows]!=spec['evaluation_target']['fixed_five']['seeds']:raise ValueError('Wrong formal count/order')
 summary=dict(status='COMPLETE',planned=5,completed=5,standing=sum(r['success'] for r in rows),envelope=sum(r['constraint_envelope_pass'] for r in rows),impact=sum(r['impact_pass'] for r in rows),admissible=sum(r['admissible_recovery'] for r in rows),evaluator_status=measured['status'],scope='Recovery only; reset settling retained separately. Admissible = original standing AND envelope AND all five integration-contact/speed comparisons. Raw evaluator success is unchanged.',runner_sha256=sha(__file__))
 write(out/'validation_summary.json',summary);print(json.dumps(summary),flush=True)
 return 0 if summary['admissible']==5 else 2

def paired(args):
 p,spec,out,inputs=protocol(args);held=spec['evaluation_target']['held_out'];runner=Path(args.pairs_runner).resolve()
 loader=importlib.util.spec_from_file_location('unchanged_existing_pair_runner',runner);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
 adapted=dict(paired_protocol=dict(planned_pairs=20,max_reset_candidates=40,candidate_seeds=held['candidate_seeds'],allowed_base_config_difference=['reset_perturb_rad'],perturb_rad=held['perturb_rad'],physical_pair_atol=held['physical_handoff_atol'],rtol=0,A='same targets-v6 reference and explicit analytical feedback; zero PPO residual',B='same controller; frozen trained deterministic PPO residual',selection=held['select']),source_protocol=dict(path=str(p),sha256=sha(p)),comparison=spec['comparison'])
 path=out.with_name(out.name+'-protocol.json')
 if path.exists():raise FileExistsError(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(adapted,f,indent=2);f.write('\n')
 original_metrics=module.metrics
 def metrics(summary):
  row=original_metrics(summary);r=summary['segments']['recovery'];c=r['integration_contacts']
  values=dict(peak_ground_vertical_N=c['peak_ground_vertical_N'],peak_ground_sum_norm_N=c['peak_ground_sum_norm_N'],floor_penetration_m=c['floor']['depth_m'],self_penetration_m=c['self']['depth_m'],max_joint_speed_rad_s=r['max_joint_speed_rad_s'])
  checks=impact(values,spec['comparison']);envelope=bool(summary['constraint_envelope_pass'] and summary['constraint_envelope_checks']['velocity_limits_declared_count']==31)
  row.update({('integration_'+key):value for key,value in values.items()});row.update(constraint_envelope_pass=envelope,impact_pass=all(checks.values()),impact_checks=json.dumps(checks,sort_keys=True),admissible_recovery=bool(summary['success'] and envelope and all(checks.values())))
  return row
 with patch.object(module,'metrics',metrics):
  code=module.execute(SimpleNamespace(manifest=str(path),training_run=str(inputs),expected_checkpoint_sha256=args.expected_checkpoint_sha256,output=str(out)))
 raw=json.loads((out/'summary.json').read_text());rows=list(csv.DictReader((out/'paired_results.csv').open()))
 groups={}
 for label in ('A_reference','B_residual'):
  rr=[r for r in rows if r['controller']==label and r['status']=='COMPLETE']
  groups[label]=dict(completed=len(rr),standing=sum(r['success']=='True' for r in rr),envelope=sum(r['constraint_envelope_pass']=='True' for r in rr),impact=sum(r['impact_pass']=='True' for r in rr),admissible=sum(r['admissible_recovery']=='True' for r in rr))
 summary=dict(status='COMPLETE' if code==0 else 'INCOMPLETE',planned_pairs=20,valid_pairs=raw['valid_pairs'],invalid_or_blocked_pairs=raw['invalid_or_blocked_pairs'],controllers=groups,pair_outcomes=raw['success_outcomes'],both_success_pairs=raw['both_success_pairs'],B_minus_A_success_time_s=raw['B_minus_A_success_time_s'],runner_sha256=sha(__file__),existing_pair_runner_sha256=sha(runner),protocol_sha256=sha(p),scope='Requirement pass follows the preregistered manifest exactly:20 valid pairs, all40 envelope passes, and at least16 B original standing successes. Impact and admissible counts are separate quality dimensions. All20 reset-only selections precede A/B outcomes; no replacement. New CSV admissible = standing AND envelope with31 declared velocities AND five integration-impact comparisons. Individual raw diagnostic summaries retain their older standing+envelope admissible definition; this summary and CSV explicitly add impact.',requirement_pass=bool(raw['valid_pairs']==20 and all(g['envelope']==20 for g in groups.values()) and groups['B_residual']['standing']>=held['required_standing_B']),all_40_impact_pass=bool(all(g['completed']==20 and g['impact']==20 for g in groups.values())),B_admissible_at_least_16=bool(groups['B_residual']['admissible']>=16))
 write(out/'validation_summary.json',summary);print(json.dumps(summary),flush=True)
 return 0 if summary['requirement_pass'] else 2

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['formal','paired'])
 for name in ('protocol','training-run','expected-checkpoint-sha256','output'):parser.add_argument('--'+name,required=True)
 parser.add_argument('--episode-wall-seconds',type=float,default=180.);parser.add_argument('--pairs-runner')
 args=parser.parse_args()
 if args.command=='paired' and not args.pairs_runner:parser.error('paired requires --pairs-runner')
 return formal(args) if args.command=='formal' else paired(args)
if __name__=='__main__':raise SystemExit(main())
