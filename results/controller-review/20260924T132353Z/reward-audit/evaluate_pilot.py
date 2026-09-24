"""Freeze and evaluate exactly five episodes of the explicitly LOCAL reward pilot.

The original input loader validates the unchanged complete execution system.
The new actor is then validated against the pilot manifest, probe and spaces;
it is never passed off as the original published checkpoint.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from x2_recovery.evaluate import prepare, policy_copy, policy_hash, validate_environment, verify_policy
from x2_recovery.constraint_audit import run_diagnostic, write_json
from pilot import DensityObjective, VERSION

SEEDS = [26092460,26092461,26092462,26092463,26092464]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--pilot-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source,pilot,output=args.training_run.resolve(),args.pilot_run.resolve(),args.output.resolve()
    if source==output or source in output.parents:raise ValueError('Output must be outside frozen original input')
    output.mkdir(parents=True,exist_ok=False)
    m=json.loads((pilot/'manifest.json').read_text());configuration=json.loads((pilot/'pilot_configuration.json').read_text())
    assert m['status']=='COMPLETE' and m['actual_transitions']==4096 and not m['adopted']
    assert m['reward_version']==VERSION==configuration['new_reward_version']
    assert sha(source/'policy_final.zip')==m['source_checkpoint_sha256']
    assert sha(source/'resolved_config.json')==m['source_config_sha256']
    assert sha(pilot/'policy_pilot.zip')==m['checkpoint_sha256']
    assert sha(Path(__file__).with_name('pilot.py'))==m['script_sha256']
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    actor=PPO.load(pilot/'policy_pilot.zip',device='cpu');actor.policy.set_training_mode(False)
    assert policy_hash(policy_copy(actor))==m['policy_state_sha256']
    observations=np.load(source/'reload_probe.npz',allow_pickle=False)['observations']
    expected=actor.predict(observations,deterministic=True)[0]
    np.savez_compressed(output/'pilot_action_probe.npz',observations=observations,actions=expected)
    import x2_recovery.env as env_module
    modules=Path(env_module.__file__).parent
    core_names=['model.py','reset.py','success.py','env.py','baseline.py','train.py','evaluate.py','model_audit.py','constraint_audit.py']
    protected={str(p):sha(p) for p in source.iterdir() if p.is_file()}
    protected.update({str(modules/name):sha(modules/name) for name in core_names})
    protected.update({str(p):sha(p) for p in (pilot/'policy_pilot.zip',pilot/'manifest.json',pilot/'pilot_configuration.json',Path(__file__).resolve(),Path(__file__).with_name('pilot.py'),output/'pilot_action_probe.npz')})
    frozen=dict(schema='local-pilot-five-evaluation-v1',created_before_episodes_utc=datetime.now(timezone.utc).isoformat(),
        seeds=SEEDS,planned_episodes=5,checkpoint_sha256=m['checkpoint_sha256'],policy_state_sha256=m['policy_state_sha256'],
        input_original_checkpoint_sha256=m['source_checkpoint_sha256'],original_input_config_sha256=m['source_config_sha256'],
        configuration=configuration,protected_dependencies_sha256=protected,
        reward_version=VERSION,inference='CPU deterministic',reset='Original legal fixed supine, perturb_rad=0; distinct seeds do not imply distinct physical initial states.',
        frozen_differences_from_original=dict(actor='New weights after 4096 transitions with fresh critic/optimizer',
            reward='Positive density components scaled by0.1; terminal rewards unchanged',control='none',physics='none',
            observation='none',action='none',success='none',episode_timeout='none:20s'),
        criteria=dict(standing_success='Original 1kHz tracker and continuous2s hold',
            constraint_envelope='All recovery samples: actual joint excess<=0.0001rad, signed actuator limit ratio<=1+1e-10, official URDF velocity ratios<=1+1e-10.',
            admissible_recovery='standing_success AND constraint_envelope_pass',
            tolerance_origin='Conservative review objective borrowed from standing-window tolerance, not a hardware safety threshold.'),
        formal_scope='New fixed-initial-state five-episode evaluation of local unadopted pilot; not the historical5/5 and not a robustness test.',
        adoption='No automatic adoption; original published input remains default.')
    write_json(output/'manifest.json',frozen)
    rows=[];identities=[];states=[];started=time.monotonic()
    for index,seed in enumerate(SEEDS,1):
        assert all(sha(p)==value for p,value in protected.items())
        env,_,_,prepared=prepare(source,expected_checkpoint_sha256=m['source_checkpoint_sha256'])
        torch.set_num_threads(1)
        policy=PPO.load(pilot/'policy_pilot.zip',device='cpu');policy.policy.set_training_mode(False)
        initial=policy_copy(policy);assert policy_hash(initial)==m['policy_state_sha256']
        validate_environment(env,prepared['saved'],policy)
        np.testing.assert_allclose(policy.predict(observations,deterministic=True)[0],expected,atol=1e-7,rtol=0)
        destination=output/f'episode-{index}'
        try:
            s=run_diagnostic(DensityObjective(env),policy,seed=seed,output=destination,trace_level='compact',
                source_identity=dict(frozen_manifest_sha256=sha(output/'manifest.json'),checkpoint_sha256=m['checkpoint_sha256'],new_objective=VERSION))
            verify_policy(policy,initial)
            np.testing.assert_allclose(policy.predict(observations,deterministic=True)[0],expected,atol=1e-7,rtol=0)
        finally:env.close()
        joints=list(csv.DictReader((destination/'joint_constraints.csv').open()))
        velocity=max(float(r['max_velocity_ratio']) for r in joints if r['max_velocity_ratio'])
        recovery=s['segments']['recovery'];envelope=bool(recovery['max_joint_excess_rad']<=.0001 and recovery['max_actuation_limit_ratio']<=1+1e-10 and velocity<=1+1e-10)
        contacts=recovery['integration_contacts']
        row=dict(episode=index,seed=seed,success=s['success'],constraint_envelope_pass=envelope,
            admissible_recovery=bool(s['success'] and envelope),sim_duration_s=s['sim_duration_s'],
            maximum_stable_hold_s=s['maximum_stable_hold_s'],max_joint_excess_rad=recovery['max_joint_excess_rad'],
            worst_joint=recovery['worst_joint'],any_joint_above_standing_tolerance_s=recovery['any_joint_above_standing_tolerance_s'],
            max_joint_speed_rad_s=recovery['max_joint_speed_rad_s'],max_official_velocity_ratio=velocity,
            max_actuation_limit_ratio=recovery['max_actuation_limit_ratio'],max_floor_penetration_m=contacts['floor']['depth_m'],
            max_self_penetration_m=contacts['self']['depth_m'],peak_ground_vertical_bodyweights=contacts['peak_ground_vertical_bodyweights'],
            raw_return=s['reward']['raw_return'],discounted_return=s['reward']['observed_discounted_prefix'])
        rows.append(row);a=np.load(destination/'control_trace.npz');states.append((a['reset_qpos'],a['reset_qvel'],a['qpos'],a['qvel'],a['action']))
        identities.append({p.name:sha(p) for p in destination.iterdir() if p.is_file()})
        print(json.dumps(row),flush=True)
    assert all(sha(p)==value for p,value in protected.items())
    with (output/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(status='COMPLETE',planned=5,completed=len(rows),invalid_or_blocked=0,
        success=sum(r['success'] for r in rows),constraint_envelope_pass=sum(r['constraint_envelope_pass'] for r in rows),
        admissible_recovery=sum(r['admissible_recovery'] for r in rows),
        checkpoint_sha256=m['checkpoint_sha256'],new_reward_version=VERSION,
        input_and_policy_identities_unchanged=True,frozen_manifest_sha256=sha(output/'manifest.json'),
        episode_artifacts_sha256=identities,wall_seconds=time.monotonic()-started,
        all_five_initial_and_control_state_trajectories_identical=all(all(np.array_equal(x,y) for x,y in zip(states[0],v)) for v in states[1:]),
        checkpoint_published=False,adopted=False,
        interpretation='This fixed-condition repeatability result does not establish perturbed robustness or repair full-process constraints. Original historical5/5 is unchanged.')
    write_json(output/'summary.json',summary)


if __name__=='__main__':main()
