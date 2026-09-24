"""One explicitly local reward-objective pilot; never replaces released inputs.

Uses the original strictly validated controller unchanged, transfers only actor
and exploration parameters to a fresh PPO, and applies one declared reward scale.
No old rollout, critic, optimizer state or running statistics are reused.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from x2_recovery.evaluate import prepare, policy_copy, policy_hash
from x2_recovery.constraint_audit import run_diagnostic, write_json

VERSION = 'reference-balance-density10-v1'
POSITIVE = ('pose_guide','head_track','balance','standing')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DensityObjective(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        old = dict(info['reward_terms'])
        assert set(POSITIVE) <= set(old)
        terms = {k: v*(.1 if k in POSITIVE else 1.) for k,v in old.items()}
        info['original_reward'] = reward
        info['original_reward_terms'] = old
        info['reward_terms'] = terms
        info['reward_version'] = VERSION
        return obs, float(sum(terms.values())), terminated, truncated, info


class Audit(BaseCallback):
    def __init__(self, deadline):
        super().__init__();self.deadline=deadline
        self.rewards=[];self.terms=[];self.dones=[];self.truncated=[];self.elapsed=[]
        self.physics=[];self.episodes=[];self.buffers=[];self.keys=None
    def _on_step(self):
        if time.monotonic()>self.deadline:
            raise TimeoutError('Pilot wall budget exceeded; no candidate adoption')
        infos=self.locals['infos'];rewards=self.locals['rewards'];dones=self.locals['dones']
        if self.keys is None:self.keys=sorted(infos[0]['reward_terms'])
        self.rewards.append(np.array(rewards).copy())
        self.terms.append([[i['reward_terms'][k] for k in self.keys] for i in infos])
        self.dones.append(np.array(dones).copy())
        self.truncated.append([i.get('TimeLimit.truncated',False) for i in infos])
        self.elapsed.append([i['elapsed_sim_s'] for i in infos])
        self.physics.append([i['physics_steps_executed'] for i in infos])
        for worker,(info,done) in enumerate(zip(infos,dones)):
            if done:self.episodes.append(dict(worker=worker,global_transition=self.num_timesteps,
                seed=info['reset_seed'],success=info['is_success'],duration_s=info['elapsed_sim_s'],
                stable_s=info['stable_duration_s'],reason=info['termination_reason'] or info['truncation_reason']))
        return True
    def _on_rollout_end(self):
        b=self.model.rollout_buffer
        assert b.full and all(np.isfinite(getattr(b,k)).all() for k in ('observations','actions','rewards','values','advantages','returns','log_probs'))
        self.buffers.append({k:getattr(b,k).copy() for k in ('rewards','values','advantages','returns')})


def actor_key(key):
    return key=='log_std' or key.startswith(('mlp_extractor.policy_net.','action_net.'))


def compact(summary):
    return {k:summary[k] for k in ('success','sim_duration_s','maximum_stable_hold_s','segments',
        'standing_window_minimum_margins','standing_window_sway','reward','actuation_consistency') if k in summary}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.training_run.resolve();output=args.output.resolve()
    if source==output or source in output.parents:raise ValueError('Output must be outside frozen training inputs')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+1200.
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    config=json.loads((source/'resolved_config.json').read_text());train=config['training']
    environments=[]
    for i in range(2):
        env,actor,_,prepared=prepare(source,expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        env.unwrapped.capture_substeps=False
        environments.append(DensityObjective(env))
        if i==0:original=actor
    torch.set_num_threads(1)
    vector=DummyVecEnv([lambda e=e:e for e in environments])
    pilot=PPO('MlpPolicy',vector,n_steps=train['n_steps'],batch_size=train['batch_size'],
        n_epochs=train['n_epochs'],learning_rate=train['learning_rate'],gamma=train['gamma'],
        gae_lambda=train['gae_lambda'],clip_range=train['clip_range'],clip_range_vf=None,
        normalize_advantage=train['normalize_advantage'],ent_coef=train['ent_coef'],vf_coef=train['vf_coef'],
        max_grad_norm=train['max_grad_norm'],target_kl=train['target_kl'],
        use_sde=config['use_sde'],sde_sample_freq=config['sde_sample_freq'],
        policy_kwargs=copy.deepcopy(original.policy_kwargs),seed=26092451,device='cpu',verbose=0)
    initial=policy_copy(pilot);old=policy_copy(original)
    transplanted={k:old[k] if actor_key(k) else v for k,v in initial.items()}
    pilot.policy.load_state_dict(transplanted,strict=True)
    assert not pilot.policy.optimizer.state
    assert all(torch.equal(pilot.policy.state_dict()[k],initial[k]) for k in initial if not actor_key(k))
    probe=np.load(source/'reload_probe.npz',allow_pickle=False)['observations']
    actual=pilot.predict(probe,deterministic=True)[0];expected=original.predict(probe,deterministic=True)[0]
    max_error=float(np.max(abs(actual-expected)))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
    initial_transferred=policy_copy(pilot)
    manifest=dict(status='RUNNING',kind='local_unadopted_reward_pilot',reward_version=VERSION,
        source_training_run=str(source),source_checkpoint_sha256=sha(source/'policy_final.zip'),
        source_config_sha256=sha(source/'resolved_config.json'),script_sha256=sha(__file__),
        checkpoint_will_not_be_published=True,source_identity=prepared['identity'],
        objective=dict(positive_density_scale=.1,scaled_components=list(POSITIVE),success=50,safety=-2,
                       added_time_cost=0,base_reward='reference-balance-v1'),
        control_changes={},physical_changes={},reset_changes={},observation_changes={},
        initialization=dict(actor='copied exactly',log_std='copied exactly',critic='fresh seeded initialization',
            optimizer='fresh Adam, empty state',statistics='fixed scaling only; no VecNormalize',
            old_rollouts_reused=False,actor_probe_max_abs_error=max_error,probe_count=len(probe),atol=1e-7,rtol=0),
        budget=dict(total_transitions=4096,workers=2,n_steps=256,rollout_transitions=512,
                    rollouts=8,hard_wall_seconds=1200,torch_threads=1),
        hyperparameters={k:v for k,v in train.items() if k not in ('env',)},seed=26092451,
        interpretation='A separate new optimization objective; not a continuation of old critic/optimizer, not a fix for pre-residual constraints, and not a release candidate.')
    write_json(output/'manifest.json',manifest)
    # Run a real pre-training diagnostic on a separate strictly prepared environment.
    diagnostic,_,_,_=prepare(source,expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    torch.set_num_threads(1)
    try:
        before=run_diagnostic(DensityObjective(diagnostic),pilot,seed=221030,output=output/'before',
                              source_identity={'new_objective':VERSION,'actor_transfer':True},trace_level='compact')
    finally:diagnostic.close()
    callback=Audit(deadline);updates=[];learn_start=time.monotonic()
    try:
        for block in range(8):
            if time.monotonic()>deadline:raise TimeoutError('Budget exhausted before next full rollout')
            pilot.learn(total_timesteps=512,reset_num_timesteps=False,callback=callback,progress_bar=False)
            assert pilot.num_timesteps==(block+1)*512
            assert all(torch.isfinite(v).all() for v in pilot.policy.state_dict().values())
            adam=[int(v['step'].item()) for v in pilot.policy.optimizer.state.values() if 'step' in v]
            updates.append(dict(rollout=block+1,transitions=pilot.num_timesteps,
                actual_adam_steps_min=min(adam),actual_adam_steps_max=max(adam),
                sb3_epoch_attempt_counter=pilot._n_updates,elapsed_wall_s=time.monotonic()-learn_start))
            print(json.dumps(updates[-1]),flush=True)
        pilot.save(output/'policy_pilot.zip')
        policy_before_reload=policy_copy(pilot)
        loaded=PPO.load(output/'policy_pilot.zip',device='cpu')
        assert policy_hash(policy_copy(loaded))==policy_hash(policy_before_reload)
        np.testing.assert_allclose(loaded.predict(probe,deterministic=True)[0],pilot.predict(probe,deterministic=True)[0],rtol=0,atol=1e-7)
        diagnostic,_,_,_=prepare(source,expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        torch.set_num_threads(1)
        try:
            after=run_diagnostic(DensityObjective(diagnostic),loaded,seed=221030,output=output/'after',
                                source_identity={'new_objective':VERSION,'actor_transfer':True,
                                    'pilot_checkpoint_sha256':sha(output/'policy_pilot.zip')},trace_level='compact')
        finally:diagnostic.close()
        raw=np.array(callback.rewards);buffer=np.concatenate([b['rewards'] for b in callback.buffers])
        np.savez_compressed(output/'training_rewards.npz',raw_environment_reward=raw,
            reward_terms=np.array(callback.terms),reward_term_names=np.array(callback.keys),
            terminated=np.array(callback.dones)&~np.array(callback.truncated),
            truncated=np.array(callback.truncated),elapsed_s=np.array(callback.elapsed),
            physics_steps=np.array(callback.physics),rollout_rewards=buffer,
            timeout_bootstrap_adjustment=buffer-raw,
            values=np.concatenate([b['values'] for b in callback.buffers]),
            advantages=np.concatenate([b['advantages'] for b in callback.buffers]),
            returns=np.concatenate([b['returns'] for b in callback.buffers]))
        final=policy_copy(pilot)
        manifest.update(status='COMPLETE',actual_transitions=pilot.num_timesteps,updates=updates,
            rollout_count=len(callback.buffers),optimizer_steps=updates[-1]['actual_adam_steps_max'],
            learning_wall_seconds=time.monotonic()-learn_start,wall_seconds=time.monotonic()-started,
            checkpoint_sha256=sha(output/'policy_pilot.zip'),policy_state_sha256=policy_hash(final),
            numerical_checks_finite=True,old_checkpoint_unchanged=sha(source/'policy_final.zip')==args.expected_checkpoint_sha256,
            parameter_changes={group:float(np.sqrt(sum(float(torch.sum((final[k]-initial_transferred[k])**2))
                for k in final if actor_key(k)==is_actor))) for group,is_actor in [('actor_and_log_std',True),('critic',False)]},
            completed_training_episodes=callback.episodes,before=compact(before),after=compact(after),
            successful_reload=True,adopted=False,
            conclusion='One fixed-budget development pilot; no seed selection, tuning, formal five-episode evaluation or evidence of general improvement.')
    except BaseException as exc:
        manifest.update(status='ERROR',error={'type':type(exc).__name__,'message':str(exc)},updates=updates)
        raise
    finally:
        write_json(output/'manifest.json',manifest);vector.close()


if __name__=='__main__':main()
