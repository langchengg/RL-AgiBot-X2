"""Audit frozen evaluation and portable inputs without reset or physical steps."""
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
import csv
import os
import stat
import sys
import json
import numpy as np
import torch
import x2_recovery.evaluate as evaluation
from x2_recovery.evaluate import audit_saved, verify_saved, prepare, validate_reset, verify_policy, strict_json
from x2_recovery.env import X2RecoveryEnv, ControlledRecoveryEnv, require
from x2_recovery.success import CALIBRATED_SETTINGS, SuccessTracker, standing_failures
from x2_recovery.train import sha256, write_json, utc_now, _state_digest

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
manifest = json.loads((OUT/'manifest.json').read_text())
summary = json.loads((OUT/'summary.json').read_text())
input_dir = OUT/manifest['portable_training_directory']
before = {name:sha256(OUT/name) for name in ('manifest.json','summary.json','episodes.csv','trajectory.jsonl')}
require((OUT/'manifest.json').stat().st_mode & (stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH) == 0, 'Manifest is writable')
require(manifest['seeds'] == [221030,221031,221032,221033,221034], 'Formal seed protocol differs')
require(summary['status']=='COMPLETE' and all(summary[k]==5 for k in ('planned','attempted','valid_completed','successes_observed')), 'Incomplete batch')
bookkeeping = audit_saved(OUT)
disk = verify_saved(OUT)
require(bookkeeping['status']=='PASS' and disk['valid_completed']==disk['successes_observed']==5, 'Saved-file aggregation failed')
for key,value in disk.items():require(summary[key]==value,'Summary aggregation differs: '+key)
for name,digest in manifest['source_sha256'].items():
    require(sha256(OUT/manifest['source_snapshot']/name)==digest, 'Frozen source differs')
    require(sha256(Path(manifest['source_directory'])/name)==digest,'Current executed module differs')
for name,digest in manifest['input_sha256'].items():
    require(sha256(input_dir/name)==digest and not (input_dir/name).is_symlink(),'Portable input is missing/changed/link')
old_path = REPO/'runs/ppo_supine/smoke-std-minus1p5-20260922/resolved_config.json'
old = json.loads(old_path.read_text())
physical = manifest['identity']['resolved_env']
require(physical==old['identity']['resolved_env'],'Original smoke physical/control/reset/safety/success settings differ')
require(manifest['identity']['model_fingerprint']==old['identity']['model_fingerprint'],'Old smoke model differs')
require(physical['success_settings']==asdict(CALIBRATED_SETTINGS),'Original success settings differ')
counts = dict(reset=0,step=0)
def forbidden_reset(*args,**kwargs):
    counts['reset']+=1
    raise RuntimeError('Physical reset forbidden during this audit')
def forbidden_step(*args,**kwargs):
    counts['step']+=1
    raise RuntimeError('Physical step forbidden during this audit')
env = None
with patch.object(X2RecoveryEnv,'reset',forbidden_reset), patch.object(X2RecoveryEnv,'step',forbidden_step), \
     patch.object(ControlledRecoveryEnv,'reset',forbidden_reset), patch.object(ControlledRecoveryEnv,'step',forbidden_step):
    try:
        env,model,original,prepared = prepare(input_dir,
            expected_checkpoint_sha256=manifest['input_sha256']['policy_final.zip'],
            reload_validation=input_dir/'reload_validation.json')
        require(prepared['identity']==manifest['identity'],'Portable prepared controller identity differs')
        require(verify_policy(model,original)==manifest['initial_policy_state_sha256']==summary['final_policy_state_sha256'],'Policy identity differs')
        optimizer_before = _state_digest(model.policy.optimizer.state_dict())
        kp,kd,limits = (np.asarray(physical[k]) for k in ('kp_Nm_rad','kd_Nm_s_rad','effort_limits_Nm'))
        dt = physical['physics_dt']
        episode_count = terminal_count = total_transitions = total_physics = 0
        active = False
        rows = []
        original_state = None
        state_errors = []
        max_prediction_error = max_pd_error = 0.0
        with (OUT/'trajectory.jsonl').open() as stream:
            for line in stream:
                record = strict_json(line)
                if record['kind']=='reset':
                    require(not active and episode_count<5,'Unexpected reset')
                    episode_count+=1
                    require(record['episode']==episode_count and record['seed']==manifest['seeds'][episode_count-1],'Reset order mismatch')
                    validate_reset(record)
                    active=True
                    start=record['handoff_time_s']
                    tracker=SuccessTracker(CALIBRATED_SETTINGS,dt,physical['config']['episode_timeout_s'])
                    tracker.reset(start)
                    tracker.origin=record['recovery_origin']  # Persisted-source replay, not a new physical claim.
                    require(tracker.update(record['measurement'])==record['tracker'],'Reset tracker mismatch')
                    observation=np.asarray(record['observation'],np.float32)
                    transitions=physics=0
                    window=[]
                    reset=record
                    if original_state is None:original_state=record
                    state_errors.append({k:float(np.max(abs(np.asarray(record[k])-original_state[k])))for k in ('qpos','qvel','observation')})
                elif record['kind']=='transition':
                    require(active and record['episode']==episode_count,'Transition outside active episode')
                    transitions+=1
                    total_transitions+=1
                    require(record['control_step']==transitions,'Control-step order differs')
                    with torch.inference_mode():action,_=model.predict(observation,deterministic=True)
                    error=float(np.max(abs(action-record['action'])))
                    max_prediction_error=max(max_prediction_error,error)
                    require(error==0,'Actual formal action does not equal loaded deterministic policy')
                    observation=np.asarray(record['observation'],np.float32)
                    require(len(record['substeps'])==record['info']['physics_steps_executed'],'Substep count differs')
                    for sample in record['substeps']:
                        physics+=1
                        total_physics+=1
                        v,t=sample['measurement'],sample['tracker']
                        require(tracker.update(v)==t,'Original SuccessTracker replay mismatch')
                        require(t['invalid_reason'] is None,'Invalid native episode')
                        if not t['failures'] and t['instant_standing_ok']:
                            require(not standing_failures(v,CALIBRATED_SETTINGS),'Original standing predicate failed')
                            window.append(v)
                        else:window=[]
                    c=record['last_substep_control']
                    raw=kp*(np.asarray(c['target_rad'])-c['pre_q_rad'])-kd*np.asarray(c['pre_dq_rad_s'])
                    clipped=np.clip(raw,limits[:,0],limits[:,1])
                    max_pd_error=max(max_pd_error,float(np.max(abs(raw-c['pre_tau_raw_Nm']))),float(np.max(abs(clipped-c['post_applied_torque_Nm']))),float(np.max(abs(clipped-record['post_step']['ctrl_Nm']))))
                    require(abs(c['post_time_s']-c['pre_time_s']-dt)<1e-10,'PD pre/post time alignment differs')
                    last=record
                elif record['kind']=='terminal':
                    require(active and last['terminated'] and not last['truncated'],'Invalid terminal')
                    terminal_count+=1
                    require(record['episode']==episode_count and record['seed']==reset['seed'],'Terminal identity differs')
                    require(last['info']['termination_reason']=='success' and tracker.result['recovery_success'],'Native success absent')
                    require(len(window)==2001 and tracker.result['stable_duration_s']>=2.,'Incomplete native hold')
                    require(abs(physics*dt-last['info']['elapsed_sim_s'])<1e-9,'Recovery duration differs')
                    require(record['policy_state_sha256']==manifest['initial_policy_state_sha256'],'Runtime policy state changed')
                    rows.append(dict(episode=episode_count,seed=reset['seed'],reset_seed=reset['reset_seed'],transitions=transitions,
                        physics_steps=physics,sim_duration_s=last['info']['elapsed_sim_s'],last_transition_physics_steps=len(last['substeps']),
                        native_hold_s=tracker.result['stable_duration_s'],native_hold_samples=len(window),
                        native_hold_start_s=window[0]['time_s']-start,native_hold_end_s=window[-1]['time_s']-start,
                        hold_pelvis_min_m=min(v['pelvis_height_m']for v in window),hold_pelvis_max_m=max(v['pelvis_height_m']for v in window),
                        hold_tilt_max_deg=max(v['tilt_deg']for v in window),hold_other_weight_max=max(v['other_weight']for v in window),
                        hold_floor_penetration_max_m=max(v['floor_penetration_m']for v in window),
                        success=True,terminated=True,truncated=False,termination_reason='success'))
                    active=False
                else:raise ValueError('Unexpected formal record: '+record['kind'])
        require(episode_count==terminal_count==5 and not active,'Formal batch boundary incomplete')
        require(total_transitions==1215 and total_physics==24280,'Unexpected total interaction counts')
        require(max_prediction_error==0 and max_pd_error<1e-12,'Formal policy/control mismatch')
        require(verify_policy(model,original)==manifest['initial_policy_state_sha256'],'Audit inference changed policy')
        require(_state_digest(model.policy.optimizer.state_dict())==optimizer_before,'Audit inference changed optimizer')
        portable=dict(status='PASS',pid=os.getpid(),python=sys.executable,loaded_evaluator=evaluation.__file__,
            loaded_checkpoint=str(input_dir/'policy_final.zip'),action_consistency=prepared['consistency'],
            controller_identity_equal=True,policy_state_sha256=verify_policy(model,original),optimizer_state_unchanged=True,
            reset_calls=counts['reset'],step_calls=counts['step'])
    finally:
        if env is not None:env.close()
require(counts==dict(reset=0,step=0),'Audit ran physical episode')
require(all(sha256(OUT/name)==digest for name,digest in before.items()),'Existing formal artifacts modified')
require((OUT/'manifest.json').stat().st_mode & 0o222==0,'Manifest permissions changed')
write_json(OUT/'independent_final_audit.json',dict(status='PASS',created_utc=utc_now(),script_sha256=sha256(__file__),
    planned=5,attempted=episode_count,valid_completed=terminal_count,successes_observed=5,result='5/5',
    rows=rows,counts=dict(resets=episode_count,terminals=terminal_count,transitions=total_transitions,physics_steps=total_physics),
    existing_EpisodeMetrics_disk_audit=bookkeeping,original_tracker_replay='EXACT_ALL_FIELDS_FOR_EVERY_PHYSICS_SAMPLE',
    formal_deterministic_actions_max_abs_error=max_prediction_error,pd_aligned_last_substep_max_abs_error_Nm=max_pd_error,
    state_comparison_errors=state_errors,recorded_physical_transitions_identical=disk['all_recorded_physical_transitions_identical'],
    frozen_manifest_mode=oct((OUT/'manifest.json').stat().st_mode & 0o777),existing_artifact_hashes_unchanged=before,
    physical_configuration_equal_to_original_smoke=True,original_smoke_configuration_sha256=sha256(old_path),
    model_fingerprint=manifest['identity']['model_fingerprint'],portable_input_prepare=portable,environment_closed=True,
    scope='Five completed frozen trained reference+PPO residual episodes. Audit uses saved observations/measurements; no new recovery trials.',
    limitations=['Five seeds produced identical physical/controller initial states and recorded trajectories: fixed-initial-state repeatability, not broad robustness.',
                'Network predictions match all1215 executed actions exactly. Reference-only ablation also succeeds; no claim that PPO created recovery ability.',
                'Original PD checked from aligned full vectors at every control transition final physical substep; scalar success measurements cover all24280 physical substeps.',
                'Portable input files are real copied files. The separately sourced model asset remains required at its recorded fixed version.']))
print(json.dumps(dict(status='PASS',counts=dict(resets=episode_count,terminals=terminal_count,transitions=total_transitions,physics_steps=total_physics),result='5/5',
    action_error=max_prediction_error,pd_error=max_pd_error,portable_prepare=portable,rows=rows),indent=2))
