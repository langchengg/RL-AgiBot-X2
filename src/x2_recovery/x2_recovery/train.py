"""Bounded, single-environment SB3 PPO experiments on the frozen X2 environment.

Instrumentation observes SB3's own rollouts and optimizer. It never changes the
buffer, reward, actions, gradients, reset semantics or physical environment.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import csv

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

from .env import EnvConfig, X2RecoveryEnv
from .model import COMMIT, REPOSITORY, SCENE, require
from .model_audit import fingerprint
from .success import MODEL_FINGERPRINT


CORE_FILES = ('env.py', 'model.py', 'reset.py', 'success.py', 'baseline.py')
REWARD_KEYS = ('height', 'upright', 'standing_hold', 'torque_cost', 'action_change', 'success')


def json_value(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    # Serialize first: a bad number must not truncate existing evidence.
    encoded = json.dumps(json_value(value), indent=2, allow_nan=False) + '\n'
    pending = Path(str(path) + '.pending')
    pending.write_text(encoded)
    pending.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 220922
    total_timesteps: int = 2048
    max_wall_seconds: float = 1200.
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = .999
    gae_lambda: float = .95
    clip_range: float = .2
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    ent_coef: float = 0.
    vf_coef: float = .5
    max_grad_norm: float = .5
    target_kl: float = .03
    log_std_init: float = -1.
    net_arch: tuple = (128, 128)
    torch_threads: int = 2
    interop_threads: int = 1
    reload_timeout_seconds: float = 180.
    env: EnvConfig = field(default_factory=EnvConfig)

    def __post_init__(self):
        for key in ('seed', 'total_timesteps', 'n_steps', 'batch_size', 'n_epochs',
                    'torch_threads', 'interop_threads'):
            v = getattr(self, key)
            require(type(v) is int and v >= (0 if key == 'seed' else 1), 'Invalid integer: ' + key)
        require(self.seed < 2**32, 'Seed must be below 2**32')
        require(isinstance(self.env, EnvConfig), 'Expected EnvConfig')
        for key in ('max_wall_seconds', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range',
                    'ent_coef', 'vf_coef', 'max_grad_norm', 'target_kl', 'log_std_init',
                    'reload_timeout_seconds'):
            v = getattr(self, key)
            require(type(v) in (int, float) and math.isfinite(v), 'Invalid real: ' + key)
        for key in ('max_wall_seconds', 'learning_rate', 'clip_range', 'vf_coef',
                    'max_grad_norm', 'target_kl', 'reload_timeout_seconds'):
            require(getattr(self, key) > 0, 'Must be positive: ' + key)
        require(0 < self.gamma < 1 and 0 < self.gae_lambda <= 1 and self.ent_coef >= 0,
                'Invalid gamma, GAE lambda or entropy coefficient')
        require(self.gamma == self.env.shaping_gamma, 'PPO gamma must equal environment shaping_gamma')
        require(self.clip_range_vf is None, 'This experiment uses no value clipping')
        require(self.normalize_advantage is True, 'normalize_advantage must be True')
        require(self.n_steps > 1 and self.batch_size > 1 and self.n_steps % self.batch_size == 0,
                'batch_size must divide n_steps and both must exceed one')
        require(self.total_timesteps % self.n_steps == 0, 'total_timesteps must be a multiple of n_steps')
        require(type(self.net_arch) is tuple and len(self.net_arch) == 2
                and all(type(n) is int and n > 0 for n in self.net_arch), 'Expected two positive layer sizes')
        require(-10 <= self.log_std_init <= 2, 'Unreasonable initial log_std')

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict), 'Expected configuration object')
        value = dict(value)
        env = value.pop('env', {})
        require(isinstance(env, dict), 'Expected environment configuration object')
        env = dict(env)
        for key in ('reference_angles', 'pd_gains'):
            if key in env:
                require(isinstance(env[key], (list, tuple))
                        and all(isinstance(row, (list, tuple)) for row in env[key]), 'Invalid ' + key)
                env[key] = tuple(tuple(row) for row in env[key])
        if 'net_arch' in value:
            require(isinstance(value['net_arch'], (list, tuple)), 'Invalid net_arch')
            value['net_arch'] = tuple(value['net_arch'])
        try:
            return cls(env=EnvConfig(**env), **value)
        except TypeError as exc:
            raise ValueError('Invalid configuration keys or structure') from exc


def configure_threads(config):
    torch.set_num_threads(config.torch_threads)
    if torch.get_num_interop_threads() != config.interop_threads:
        torch.set_num_interop_threads(config.interop_threads)


def policy_kwargs(config):
    return dict(net_arch=dict(pi=list(config.net_arch), vf=list(config.net_arch)),
                activation_fn=torch.nn.Tanh, ortho_init=True, log_std_init=config.log_std_init,
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5, betas=(.9, .999), weight_decay=0., amsgrad=False),
                squash_output=False, share_features_extractor=True, normalize_images=True)


def make_model(config, vec, cls=PPO):
    return cls('MlpPolicy', vec, device='cpu', seed=config.seed, verbose=1,
               n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
               learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
               clip_range=config.clip_range, clip_range_vf=config.clip_range_vf,
               normalize_advantage=config.normalize_advantage, ent_coef=config.ent_coef,
               vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm, target_kl=config.target_kl,
               use_sde=False, sde_sample_freq=-1, stats_window_size=100,
               policy_kwargs=policy_kwargs(config))


class TimedMonitor(Monitor):
    """Standard Monitor semantics, with wall time around the unmodified reset."""
    def __init__(self, env, filename):
        super().__init__(env, filename=str(filename))
        self.reset_seconds = 0.
        self.reset_calls = 0

    def reset(self, **kwargs):
        start = time.perf_counter()
        try:
            return super().reset(**kwargs)
        finally:
            self.reset_seconds += time.perf_counter() - start
            self.reset_calls += 1


def make_env(config, directory, stack):
    env = X2RecoveryEnv(config.env, render_mode=None, capture_substeps=False)
    stack.callback(env.close)
    monitor = TimedMonitor(env, directory / 'monitor.csv')
    stack.callback(monitor.close)
    vec = VecCheckNan(DummyVecEnv([lambda: monitor]), raise_exception=True, warn_once=False)
    stack.callback(vec.close)
    vec.seed(config.seed)
    return env, monitor, vec


def source_hashes():
    return {name: sha256(Path(__file__).resolve().parent / name) for name in CORE_FILES}


def env_identity(env):
    def space(s):
        # Infinite declared bounds are strings, never invalid JSON numbers.
        return dict(shape=list(s.shape), dtype=str(s.dtype),
                    low=[float(v) if np.isfinite(v) else '-inf' for v in s.low],
                    high=[float(v) if np.isfinite(v) else 'inf' for v in s.high])
    measured = fingerprint(env.loaded)
    require(measured == MODEL_FINGERPRINT, 'Effective model fingerprint changed')
    return json_value(dict(model_fingerprint=measured, core_source_hashes=source_hashes(),
                           resolved_env=env.resolved_config(), observation=space(env.observation_space),
                           action=space(env.action_space)))


def provenance(directory):
    import importlib.metadata
    root = Path(subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parent),
                                         'rev-parse', '--show-toplevel'], text=True).strip())
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    snapshot = directory / 'source'
    snapshot.mkdir()
    paths = list((root / 'src/x2_recovery/x2_recovery').glob('*.py'))
    paths += [root / p for p in ('requirements.txt', 'src/x2_recovery/setup.py',
                                'src/x2_recovery/package.xml', '.gitignore')]
    hashes = {}
    for path in paths:
        relative = path.relative_to(root)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(target)
    (snapshot / 'tracked.diff').write_text(git('diff', 'HEAD', '--', *(str(p.relative_to(root)) for p in paths)))
    memory = {line.split(':')[0]: line.split(':')[1].strip()
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}
    return dict(git=dict(root=str(root), branch=git('branch', '--show-current'), head=git('rev-parse', 'HEAD'),
                         status=git('status', '--porcelain', '--untracked-files=all')),
                source_snapshot='source', source_hashes=hashes, cwd=str(Path.cwd()),
                command=[sys.executable, '-m', 'x2_recovery.train', *sys.argv[1:]],
                python=dict(executable=sys.executable, version=sys.version, prefix=sys.prefix),
                system=dict(platform=platform.platform(), architecture=platform.machine(),
                            os_release=Path('/etc/os-release').read_text(), cpu_count=os.cpu_count(),
                            virtualization=subprocess.check_output(['systemd-detect-virt'], text=True).strip(),
                            memory=memory, disk_free_bytes=shutil.disk_usage(directory).free),
                dependencies={name: importlib.metadata.version(name) for name in
                              ('mujoco', 'gymnasium', 'stable-baselines3', 'torch', 'numpy', 'matplotlib')},
                torch=dict(device='cpu', compute_threads=torch.get_num_threads(),
                           interop_threads=torch.get_num_interop_threads()),
                model_source=dict(repository=REPOSITORY, commit=COMMIT, scene=SCENE))


def resolved_configuration(config, env, model):
    return dict(training=asdict(config), environment=env.resolved_config(), identity=env_identity(env),
                algorithm='stable_baselines3.PPO', policy='MlpPolicy', n_envs=1, device='cpu',
                normalization='fixed_env_scaling', render_mode=None, capture_substeps=False,
                wrappers=['Monitor (reset timing only)', 'DummyVecEnv', 'VecCheckNan'],
                extra_clipping=False, use_sde=False,
                policy_resolved=dict(representation=str(model.policy), activation='torch.nn.Tanh',
                                     ortho_init=model.policy.ortho_init,
                                     features_extractor=type(model.policy.features_extractor).__name__,
                                     squash_output=model.policy.squash_output,
                                     optimizer=type(model.policy.optimizer).__name__,
                                     optimizer_defaults=model.policy.optimizer.defaults),
                sampling_gate=dict(window=20, safety_le2_fraction=.8,
                                   formal_subsecond_gate='all 20 recent episodes safety-aborted in under one second',
                                   additional_review='Inspect safety endings at 3-5 transitions as well'),
                metric_scope=dict(height_tilt_hold='control-step-sampled; not physical-substep extrema',
                                  control_peaks='existing info control peaks over actual physical substeps',
                                  torque_saturation='weighted by actual physics_steps_executed',
                                  raw_return='environment rewards before SB3 timeout bootstrap'))


def csv_output(stack, path, fields):
    stream = stack.enter_context(path.open('x', newline='', buffering=1))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    return writer


def distribution(values):
    return dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                    np.percentile(values, [0, 25, 50, 75, 100]).tolist())) if values else None


def sampling_quality(episodes):
    recent = episodes[-20:]
    short = lambda row: row['length'] <= 2 and str(row['reason']).startswith('safety_abort:')
    fraction = sum(short(row) for row in recent) / len(recent) if recent else None
    degenerate = len(recent) >= 20 and fraction >= .8
    # A separate warning makes 3-5-step failures visible without moving the declared gate.
    short5 = sum(row['length'] <= 5 and str(row['reason']).startswith('safety_abort:') for row in recent)
    return dict(completed_episodes=len(episodes), recent_sample_size=len(recent),
                recent_safety_le2_fraction=fraction, sampling_degenerate=degenerate,
                recent_safety_le5_fraction=short5 / len(recent) if recent else None,
                all_recent_safety_under_one_second=len(recent) >= 20 and all(
                    str(row['reason']).startswith('safety_abort:') and row['simulated_seconds'] < 1.
                    for row in recent),
                status='sampling_degenerate' if degenerate else
                ('insufficient_episodes' if len(recent) < 20 else 'no_defined_degeneracy_detected'),
                length=distribution([r['length'] for r in episodes]),
                simulated_seconds=distribution([r['simulated_seconds'] for r in episodes]),
                episode_return=distribution([r['return'] for r in episodes]),
                reasons=dict(Counter(r['reason'] for r in episodes)),
                time_limit_fraction=sum(r['truncated'] for r in episodes) / len(episodes) if episodes else None,
                successes=sum(r['success'] for r in episodes))


class EpisodeLog:
    """Aggregate only transition info, which survives vector auto reset intact."""
    fields = ('episode_id', 'env_index', 'reset_seed', 'global_transition', 'length', 'physics_steps',
              'simulated_seconds', 'return', 'terminated', 'truncated', 'reason', 'success',
              'max_height_m', 'tilt_at_max_height_deg', 'min_tilt_deg', 'max_stable_seconds',
              'torque_saturation_fraction', 'raw_action_oob_fraction', 'action_boundary_fraction',
              'target_boundary_fraction', 'action_change_mean', 'max_joint_speed_rad_s',
              'max_joint_limit_rad', 'max_floor_penetration_m', 'max_self_penetration_m',
              'max_raw_torque_Nm', 'max_applied_torque_Nm', 'reward_terms', 'reward_terms_raw')

    def __init__(self, directory, stack, physics_dt):
        self.writer = csv_output(stack, directory / 'episode_diagnostics.csv', self.fields)
        self.stream = stack.enter_context((directory / 'transitions.jsonl').open('x', buffering=1))
        self.dt = physics_dt
        self.episodes = []
        self.physics_steps = 0
        self.transitions = 0
        self.observations = []
        self.current = None

    def add(self, observation, raw_action, action, reward, terminated, truncated, info):
        require(np.isfinite(reward), 'Nonfinite environment reward')
        require(np.isclose(sum(info['reward_terms'].values()), reward, rtol=1e-6, atol=2e-7),
                'Reward components disagree with environment reward')
        if len(self.observations) < 16:
            self.observations.append(np.array(observation, dtype=np.float32, copy=True))
        if self.current is None:
            self.current = dict(length=0, physics_steps=0, return_value=0., saturation=0., raw_oob=0.,
                                boundary=0., target_boundary=0., action_change=0.,
                                max_height_m=-float('inf'), min_tilt_deg=float('inf'),
                                max_stable_seconds=0., peaks={},
                                reward_terms={k: 0. for k in REWARD_KEYS},
                                reward_terms_raw={k: 0. for k in REWARD_KEYS})
        c = self.current
        self.transitions += 1
        steps = info['physics_steps_executed']
        self.physics_steps += steps
        c['length'] += 1
        c['physics_steps'] += steps
        c['return_value'] += float(reward)
        c['saturation'] += info['torque_saturation_fraction'] * steps
        c['raw_oob'] += float(np.mean(np.abs(raw_action) > 1)) if raw_action is not None else 0.
        c['boundary'] += float(np.mean(np.abs(action) >= 1))
        c['target_boundary'] += info['control']['target_boundary_fraction']
        c['action_change'] += info['reward_terms_raw']['action_change']
        if info['state']['pelvis_height_m'] > c['max_height_m']:
            c['max_height_m'] = info['state']['pelvis_height_m']
            c['tilt_at_max_height_deg'] = info['state']['tilt_deg']
        c['min_tilt_deg'] = min(c['min_tilt_deg'], info['state']['tilt_deg'])
        c['max_stable_seconds'] = max(c['max_stable_seconds'], info['stable_duration_s'])
        for key in ('joint_rad_s', 'joint_limit_rad', 'floor_penetration_m', 'self_penetration_m',
                    'raw_torque_abs_max_Nm', 'applied_torque_abs_max_Nm'):
            c['peaks'][key] = max(c['peaks'].get(key, 0.), info['control'][key])
        for group in ('reward_terms', 'reward_terms_raw'):
            for key in REWARD_KEYS:
                c[group][key] += info[group][key]
        record = dict(global_transition=self.transitions, raw_action=raw_action, action=action,
                      reward=float(reward), terminated=bool(terminated), truncated=bool(truncated), info=info)
        self.stream.write(json.dumps(json_value(record), allow_nan=False) + '\n')
        if terminated or truncated:
            row = dict(episode_id=len(self.episodes) + 1, env_index=0, reset_seed=info['reset_seed'],
                       global_transition=self.transitions, length=c['length'], physics_steps=c['physics_steps'],
                       simulated_seconds=c['physics_steps'] * self.dt, **{'return': c['return_value']},
                       terminated=bool(terminated), truncated=bool(truncated),
                       reason=info['termination_reason'] or info['truncation_reason'], success=bool(info['is_success']),
                       **{k: c[k] for k in ('max_height_m', 'tilt_at_max_height_deg', 'min_tilt_deg', 'max_stable_seconds')},
                       torque_saturation_fraction=c['saturation'] / c['physics_steps'],
                       raw_action_oob_fraction=c['raw_oob'] / c['length'] if raw_action is not None else None,
                       action_boundary_fraction=c['boundary'] / c['length'],
                       target_boundary_fraction=c['target_boundary'] / c['length'],
                       action_change_mean=c['action_change'] / c['length'],
                       max_joint_speed_rad_s=c['peaks']['joint_rad_s'], max_joint_limit_rad=c['peaks']['joint_limit_rad'],
                       max_floor_penetration_m=c['peaks']['floor_penetration_m'],
                       max_self_penetration_m=c['peaks']['self_penetration_m'],
                       max_raw_torque_Nm=c['peaks']['raw_torque_abs_max_Nm'],
                       max_applied_torque_Nm=c['peaks']['applied_torque_abs_max_Nm'],
                       reward_terms=c['reward_terms'], reward_terms_raw=c['reward_terms_raw'])
            if 'episode' in info:
                require(info['episode']['l'] == row['length'] and
                        np.isclose(info['episode']['r'], row['return'], rtol=1e-6, atol=2e-6),
                        'Monitor and diagnostic episode disagree')
            self.episodes.append(row)
            self.writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, dict) else v
                                  for k, v in row.items()})
            self.current = None

    def partial(self, reason):
        return dict(complete=False, reason=reason, transitions=0 if self.current is None else self.current['length'],
                    trajectory=self.current)


def parameter_copy(policy):
    return {key: value.detach().cpu().clone() for key, value in policy.named_parameters()}


def parameter_changes(before, policy):
    groups = dict(actor_mean=[], critic=[], log_std=[])
    for name, value in policy.named_parameters():
        require(torch.isfinite(value).all().item(), 'Nonfinite parameter: ' + name)
        delta = (value.detach().cpu() - before[name]).double().flatten()
        group = ('log_std' if name == 'log_std' else
                 'actor_mean' if name.startswith(('mlp_extractor.policy_net.', 'action_net.')) else
                 'critic' if name.startswith(('mlp_extractor.value_net.', 'value_net.')) else None)
        require(group is not None, 'Unclassified trainable parameter: ' + name)
        groups[group].append(delta)
    return {key: dict(l2=float(torch.linalg.vector_norm(torch.cat(parts))),
                      max_abs=float(torch.cat(parts).abs().max())) for key, parts in groups.items()}


def validate_updates(changes, optimizer_steps, completed_updates):
    require(optimizer_steps > 0 and completed_updates > 0, 'No complete optimizer update')
    for name in ('actor_mean', 'critic'):
        require(changes[name]['l2'] > 0 and changes[name]['max_abs'] > 0, 'Unchanged ' + name)


class ObservedPPO(PPO):
    """Delegate every algorithm operation to SB3; observe after train returns."""
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['evidence']

    def collect_rollouts(self, *args, **kwargs):
        if self.evidence.stop_reason is not None:
            return False
        return super().collect_rollouts(*args, **kwargs)

    def train(self):
        started = time.perf_counter()
        super().train()
        self.evidence.after_update(self, time.perf_counter() - started)


class TrainingEvidence(BaseCallback):
    def __init__(self, config, directory, stack, episodes, run_type='smoke'):
        super().__init__()
        self.config, self.directory, self.episodes = config, directory, episodes
        self.run_type = run_type
        self.completed_rollouts = self.completed_updates = self.optimizer_steps = 0
        self.gradient_tensors = 0
        self.gradient_max_abs = 0.
        self.stop_reason = None
        self.learn_start = None
        self.cycle_start = None
        self.rollout_records = []
        self.updates = []
        self.progress = csv_output(stack, directory / 'progress.csv',
                                   ('rollout', 'sampled_transitions', 'trained_rollout_transitions',
                                    'optimizer_steps', 'sb3_n_updates', 'learn_seconds', 'cycle_seconds',
                                    'optimization_seconds', 'transitions_per_second', 'policy_loss',
                                    'value_loss', 'entropy_loss', 'total_loss', 'approx_kl',
                                    'explained_variance', 'undefined_reason'))

    def install_hooks(self, model, stack):
        def gradient(grad):
            require(torch.isfinite(grad).all().item(), 'Nonfinite backward gradient')
            self.gradient_tensors += 1
            self.gradient_max_abs = max(self.gradient_max_abs, float(grad.detach().abs().max()))
            return None  # Read-only: retain the original gradient.
        for param in model.policy.parameters():
            stack.callback(param.register_hook(gradient).remove)

        def before_step(optimizer, args, kwargs):
            for group in optimizer.param_groups:
                for param in group['params']:
                    require(param.grad is not None and torch.isfinite(param.grad).all().item(),
                            'Missing/nonfinite post-clipping gradient')

        def after_step(optimizer, args, kwargs):
            self.optimizer_steps += 1
            for group in optimizer.param_groups:
                for param in group['params']:
                    require(torch.isfinite(param).all().item(), 'Nonfinite updated parameter')
            self.optimizer_state(optimizer)

        stack.callback(model.policy.optimizer.register_step_pre_hook(before_step).remove)
        stack.callback(model.policy.optimizer.register_step_post_hook(after_step).remove)

    def optimizer_state(self, optimizer):
        steps = []
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    require(torch.isfinite(value).all().item(), 'Nonfinite optimizer state: ' + key)
            if 'step' in state:
                steps.append(int(state['step'].item()))
        require(not steps or min(steps) == max(steps) == self.optimizer_steps,
                'Adam per-parameter step disagrees with global hook count')
        return dict(parameter_states=len(steps), step_min=min(steps) if steps else 0,
                    step_max=max(steps) if steps else 0, global_hook_steps=self.optimizer_steps)

    def _on_rollout_start(self):
        if self.cycle_start is None:
            self.cycle_start = self.learn_start  # Includes the first reset in learn().

    def _on_step(self):
        loc = self.locals
        info = loc['infos'][0]
        done = bool(loc['dones'][0])
        truncated = bool(info.get('TimeLimit.truncated', False))
        if done:
            require('terminal_observation' in info, 'Missing terminal observation')
            require(truncated == bool(info['truncation_reason']), 'Truncation semantics disagree')
        self.episodes.add(loc['obs_tensor'][0].detach().cpu().numpy(), loc['actions'][0],
                          loc['clipped_actions'][0], float(loc['rewards'][0]),
                          done and not truncated, truncated, info)
        if time.perf_counter() - self.learn_start >= self.config.max_wall_seconds:
            self.stop_reason = 'wall_budget'
            # Complete the current rollout when this is its last transition.
            return (loc['n_steps'] + 1) == self.config.n_steps
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        shapes = dict(observations=(self.config.n_steps, 1, 117), actions=(self.config.n_steps, 1, 31))
        arrays, checks = {}, {}
        for name in ('observations', 'actions', 'rewards', 'values', 'log_probs', 'advantages', 'returns'):
            arr = getattr(buffer, name)
            require(arr.shape == shapes.get(name, (self.config.n_steps, 1)), 'Wrong rollout shape: ' + name)
            require(arr.dtype == np.float32 and np.isfinite(arr).all(), 'Invalid rollout values/dtype: ' + name)
            arrays[name] = arr
            checks[name] = dict(shape=list(arr.shape), dtype=str(arr.dtype), finite=True,
                                min=float(arr.min()), max=float(arr.max()))
        self.completed_rollouts += 1
        np.savez_compressed(self.directory / f'rollout_{self.completed_rollouts:04d}.npz', **arrays)
        self.rollout_records.append(checks)

    def after_update(self, model, optimization_seconds):
        metrics = model.logger.name_to_value
        names = dict(policy_loss='train/policy_gradient_loss', value_loss='train/value_loss',
                     entropy_loss='train/entropy_loss', total_loss='train/loss', approx_kl='train/approx_kl')
        losses = {k: float(metrics[v]) for k, v in names.items()}
        require(np.isfinite(list(losses.values())).all(), 'Nonfinite recorded update loss/KL')
        variance = float(metrics['train/explained_variance'])
        reason = ''
        if not math.isfinite(variance):
            require(np.var(model.rollout_buffer.returns) == 0, 'Unexpected undefined explained variance')
            variance, reason = None, 'returns variance is zero'
        self.completed_updates += 1
        elapsed = time.perf_counter() - self.learn_start
        cycle = time.perf_counter() - self.cycle_start
        row = dict(rollout=self.completed_updates, sampled_transitions=model.num_timesteps,
                   trained_rollout_transitions=self.completed_updates * self.config.n_steps,
                   optimizer_steps=self.optimizer_steps, sb3_n_updates=model._n_updates,
                   learn_seconds=elapsed, cycle_seconds=cycle, optimization_seconds=optimization_seconds,
                   transitions_per_second=self.config.n_steps / cycle, **losses,
                   explained_variance=variance, undefined_reason=reason)
        self.progress.writerow(row)
        self.updates.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        quality = sampling_quality(self.episodes.episodes)
        if quality['sampling_degenerate']:
            self.stop_reason = 'sampling_degenerate'
        elif self.run_type == 'formal' and quality['all_recent_safety_under_one_second']:
            self.stop_reason = 'sampling_early_safety'
        elif elapsed >= self.config.max_wall_seconds:
            self.stop_reason = 'wall_budget'
        self.cycle_start = time.perf_counter()

    def summary(self, model):
        trained = self.completed_updates * self.config.n_steps
        return dict(requested_transitions=self.config.total_timesteps,
                    sampled_transitions=self.episodes.transitions,
                    transitions_in_completed_training_rollouts=trained,
                    completed_rollout_count=self.completed_rollouts, completed_optimization_rounds=self.completed_updates,
                    partial_rollout_transitions=self.episodes.transitions - trained,
                    optimizer_steps=self.optimizer_steps, sb3_n_updates=model._n_updates,
                    adam_state=self.optimizer_state(model.policy.optimizer), rollout_checks=self.rollout_records,
                    numerical_checks=dict(rollout='all seven arrays of every complete rollout',
                                          losses='SB3 update-level means; total loss is last evaluated minibatch',
                                          gradients='every backward parameter tensor and every post-clip step input',
                                          gradient_tensors=self.gradient_tensors,
                                          max_abs_backward_gradient=self.gradient_max_abs,
                                          parameters_and_optimizer='after every observed optimizer step', finite=True),
                    updates=self.updates)


def plot_reward(directory):
    """Rebuild solely from saved raw episode records, never buffer rewards."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    resolved = json.loads((directory / 'resolved_config.json').read_text())
    with (directory / 'episode_diagnostics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    try:
        x = np.array([int(row['global_transition']) for row in rows])
        y = np.array([float(row['return']) for row in rows])
        require(np.isfinite(y).all(), 'Nonfinite raw episode return in plot input')
        ax.plot(x, y, '.', alpha=.55, markersize=4, label='Raw complete episode return')
        if len(y) >= 20:
            ax.plot(x[19:], np.convolve(y, np.ones(20) / 20, mode='valid'),
                    label='Trailing mean: 20 complete episodes', linewidth=1.5)
        if not len(y):
            ax.text(.5, .5, 'No complete episodes recorded', ha='center', transform=ax.transAxes)
        config_id = sha256(directory / 'resolved_config.json')[:10]
        ax.set(title=f"{directory.name}\nseed={resolved['training']['seed']} | config={config_id}",
               xlabel='Cumulative sampled training transitions', ylabel='Raw episode return')
        ax.grid(alpha=.2)
        ax.legend(loc='best')
        fig.savefig(directory / 'training_reward.png', dpi=150)
    finally:
        plt.close(fig)
    return dict(complete_episodes=len(rows), path='training_reward.png',
                sha256=sha256(directory / 'training_reward.png'))


def compare_actions(model, observations, expected, *, atol=1e-7, rtol=1e-6):
    require(observations.ndim == 2 and observations.shape[1] == 117
            and observations.dtype == np.float32 and np.isfinite(observations).all(), 'Invalid saved observations')
    actual, _ = model.predict(observations, deterministic=True)
    require(actual.shape == expected.shape == (len(observations), 31), 'Reload action shape mismatch')
    require(np.isfinite(expected).all() and np.isfinite(actual).all()
            and np.all(np.abs(actual) <= 1), 'Invalid reload actions')
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), 'Reload deterministic action mismatch')
    return dict(max_abs_error=float(np.max(np.abs(actual - expected))), atol=atol, rtol=rtol,
                observations=len(observations), action_shape=list(actual.shape))


def reload_check(directory, output=None):
    directory = Path(directory).resolve()
    output = Path(output) if output else directory / 'reload_check.json'
    result = dict(status='ERROR', episode_label='reload_validation', formal_evaluation='NOT_EVALUATED',
                  pid=os.getpid(), python_executable=sys.executable,
                  checkpoint_path=str(directory / 'policy_final.zip'), started_utc=utc_now())
    start = time.perf_counter()
    try:
        resolved = json.loads((directory / 'resolved_config.json').read_text())
        config = TrainConfig.from_dict(resolved['training'])
        configure_threads(config)
        manifest = json.loads((directory / 'manifest.json').read_text())
        require(sha256(directory / 'resolved_config.json') == manifest['config_sha256'], 'Config hash mismatch')
        require(sha256(directory / 'policy_final.zip') == manifest['checkpoint']['sha256'], 'Checkpoint hash mismatch')
        # Ordinary PPO.load is intentionally used; no diagnostic subclass is needed.
        model = PPO.load(directory / 'policy_final.zip', device='cpu')
        with np.load(directory / 'reload_probe.npz', allow_pickle=False) as probe:
            result['action_consistency'] = compare_actions(model, probe['observations'], probe['actions'])
        with X2RecoveryEnv(config.env, render_mode=None, capture_substeps=False) as env:
            before = env_identity(env)
            require(before == resolved['identity'], 'Environment/model/interface compatibility mismatch')
            result['compatibility'] = before
            obs, info = env.reset(seed=config.seed + 1)
            result['reset_seed'] = info['reset_seed']
            steps = physics_steps = 0
            raw_return = 0.
            with (directory / (output.stem + '_trajectory.jsonl')).open('w', buffering=1) as stream:
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    physics_steps += info['physics_steps_executed']
                    raw_return += reward
                    stream.write(json.dumps(json_value(dict(transition=steps, action=action,
                                                           reward=reward, terminated=terminated,
                                                           truncated=truncated, info=info)), allow_nan=False) + '\n')
                    if terminated or truncated:
                        break
                    require(time.perf_counter() - start < config.reload_timeout_seconds,
                            'Reload wall budget expired before a complete episode')
            require(env_identity(env) == before, 'Environment changed during reload')
            result.update(status='PASS', complete_episode=True, transitions=steps, physics_steps=physics_steps,
                          simulated_seconds=physics_steps * env.physics_dt, raw_return=raw_return,
                          terminated=terminated, truncated=truncated, final_info=info)
        result['checkpoint_sha256'] = sha256(directory / 'policy_final.zip')
        return result
    except BaseException as exc:
        result.update(status='ERROR', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), error_evidence=repr(getattr(exc, 'evidence', None)))
        raise
    finally:
        result.update(ended_utc=utc_now(), wall_seconds=time.perf_counter() - start)
        write_json(output, result)


def launch_reload(directory, timeout):
    command = [sys.executable, '-m', 'x2_recovery.train', 'reload-check', '--run-dir', str(directory)]
    try:
        child = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        (directory / 'reload_stdout.log').write_text(child.stdout)
        (directory / 'reload_stderr.log').write_text(child.stderr)
        result = json.loads((directory / 'reload_check.json').read_text()) if (directory / 'reload_check.json').exists() else {}
        result.update(command=command, returncode=child.returncode)
        require(child.returncode == 0 and result.get('status') == 'PASS', 'Independent reload failed: ' + repr(result))
        require(result['pid'] != os.getpid(), 'Reload did not run in a new process')
        return result
    except subprocess.TimeoutExpired as exc:
        (directory / 'reload_stdout.log').write_bytes(exc.stdout or b'')
        (directory / 'reload_stderr.log').write_bytes(exc.stderr or b'')
        write_json(directory / 'reload_check.json', dict(status='INTERRUPTED', reason='external_watchdog_timeout',
                                                        timeout_seconds=timeout, complete_episode=False, command=command))
        raise RuntimeError('Independent reload exceeded its watchdog; no complete episode') from exc


def formal_plan(smoke_directory, config):
    directory = Path(smoke_directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest['run_type'] == 'smoke' and manifest['status'] == 'PASS'
            and manifest['reload']['status'] == 'PASS', 'Formal experiment requires a passed smoke and reload')
    quality = manifest['sampling_quality']
    require(not quality['sampling_degenerate'] and quality['recent_safety_le5_fraction'] is not None
            and quality['recent_safety_le5_fraction'] < .8, 'Sampling quality requires investigation before expansion')
    # This additional conservative review follows the observed X2 failure mode:
    # passing the two-transition gate cannot excuse uniformly subsecond failures.
    all_subsecond = quality.get('all_recent_safety_under_one_second', False)
    if not all_subsecond and quality.get('completed_episodes', 0) >= 20:
        all_subsecond = (quality['simulated_seconds']['max'] < 1. and
                         all(str(reason).startswith('safety_abort:') for reason in quality['reasons']))
    require(not all_subsecond, 'All recent episodes are subsecond safety aborts; do not expand the budget')
    measured = json.loads((directory / 'resolved_config.json').read_text())['training']
    current = json_value(asdict(config))
    for key in current:
        if key not in ('seed', 'max_wall_seconds', 'total_timesteps'):
            require(current[key] == measured[key], 'Formal configuration differs from measured smoke: ' + key)
    require(config.seed != measured['seed'], 'Formal experiment requires a fresh seed')
    require(config.max_wall_seconds <= 3600, 'Formal learn budget exceeds 60 minutes')
    rate = manifest['timing']['transitions_per_second']
    planned = config.n_steps * math.floor(.8 * config.max_wall_seconds * rate / config.n_steps)
    require(planned >= config.n_steps, 'Budget cannot support one complete rollout')
    return dict(measured_run=str(directory.resolve()), measured_transitions_per_second=rate,
                learn_budget_seconds=config.max_wall_seconds, reserve_factor=.8, rollout_batch=config.n_steps,
                planned_transitions=planned, formula='B * floor(0.8 * T * r / B)',
                qualification='Compute planning only; not evidence of learning sufficiency')


def run_experiment(config, directory, run_type, *, probe_policy='untrained', probe_episodes=20,
                   probe_transitions=128, planning=None):
    require(isinstance(config, TrainConfig), 'Expected TrainConfig')
    require(run_type in ('probe', 'smoke', 'formal'), 'Unknown run type')
    require(type(probe_episodes) is int and probe_episodes > 0 and type(probe_transitions) is int
            and probe_transitions > 0, 'Invalid probe budget')
    directory = Path(directory).expanduser().absolute()
    require(not directory.exists(), 'Run directory already exists: ' + str(directory))
    require(config.env == EnvConfig(), 'This experiment must retain the frozen default environment')
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = dict(run_id=directory.name, run_type=run_type, seed=config.seed, status='RUNNING',
                    started_utc=utc_now(), normalization='fixed_env_scaling', formal_evaluation='NOT_EVALUATED',
                    planning=planning, timing={}, checkpoint=None, reload=None, environment_closed=False)
    env = episodes = evidence = model = None
    before = source_hashes()
    error = None
    with (directory / 'stdout.log').open('x', buffering=1) as stdout:
        with redirect_stdout(stdout), redirect_stderr(stdout):
            try:
                configure_threads(config)
                manifest['provenance'] = provenance(directory)
                manifest['core_sources_before_creation'] = before
                write_json(directory / 'manifest.json', manifest)
                with ExitStack() as stack:
                    construction_start = time.perf_counter()
                    env, monitor, vec = make_env(config, directory, stack)
                    model = make_model(config, vec, PPO if run_type == 'probe' else ObservedPPO)
                    identity = env_identity(env)
                    require(identity['core_source_hashes'] == before, 'Environment source changed during construction')
                    manifest['identity'] = identity
                    manifest['model_path'] = str(env.loaded.asset_repo / SCENE)
                    resolved = resolved_configuration(config, env, model)
                    write_json(directory / 'resolved_config.json', resolved)
                    manifest['config_sha256'] = sha256(directory / 'resolved_config.json')
                    episodes = EpisodeLog(directory, stack, env.physics_dt)
                    manifest['timing']['construction_seconds'] = time.perf_counter() - construction_start
                    if run_type == 'probe':
                        probe_start = time.perf_counter()
                        obs = vec.reset()
                        stop = 'episode_budget'
                        while len(episodes.episodes) < probe_episodes:
                            if episodes.transitions >= probe_transitions or time.perf_counter() - probe_start >= config.max_wall_seconds:
                                stop = 'diagnostic_cutoff'
                                break
                            if probe_policy == 'zero':
                                raw, action = None, np.zeros((1, 31), dtype=np.float32)
                            else:
                                with torch.no_grad():
                                    raw = model.policy(model.policy.obs_to_tensor(obs)[0])[0].cpu().numpy()
                                require(not model.policy.squash_output, 'Unexpected squashed PPO policy')
                                action = np.clip(raw, vec.action_space.low, vec.action_space.high)
                            previous = obs[0].copy()
                            obs, reward, done, infos = vec.step(action)
                            truncated = bool(infos[0]['TimeLimit.truncated'])
                            episodes.add(previous, raw[0] if raw is not None else None, action[0], float(reward[0]),
                                         bool(done[0]) and not truncated, truncated, infos[0])
                        manifest.update(status='COMPLETE', policy_label=probe_policy,
                                        initial_log_std=config.log_std_init if probe_policy != 'zero' else None,
                                        stop_reason=stop, sampled_transitions=episodes.transitions,
                                        requested_episodes=probe_episodes, requested_transition_cap=probe_transitions)
                        manifest['timing']['probe_seconds'] = time.perf_counter() - probe_start
                        write_json(directory / 'partial_episode.json', episodes.partial(stop))
                    else:
                        before_parameters = parameter_copy(model.policy)
                        evidence = TrainingEvidence(config, directory, stack, episodes, run_type)
                        model.evidence = evidence
                        hooks = stack.enter_context(ExitStack())
                        evidence.install_hooks(model, hooks)
                        logger = configure(str(directory / 'sb3'), ['csv'])
                        stack.callback(logger.close)
                        model.set_logger(logger)
                        evidence.learn_start = time.perf_counter()
                        try:
                            model.learn(config.total_timesteps, callback=evidence, log_interval=None)
                        finally:
                            manifest['timing']['learn_seconds'] = time.perf_counter() - evidence.learn_start
                            hooks.close()
                        # Explicitly flush the last actual optimization values after train().
                        model.logger.dump(step=model.num_timesteps)
                        changes = parameter_changes(before_parameters, model.policy)
                        manifest['training'] = evidence.summary(model)
                        manifest['training']['parameter_changes'] = changes
                        elapsed = manifest['timing']['learn_seconds']
                        manifest['timing'].update(transitions_per_second=episodes.transitions / elapsed,
                                                 wall_budget_seconds=config.max_wall_seconds,
                                                 budget_overrun_seconds=max(0., elapsed - config.max_wall_seconds))
                        manifest['stop_reason'] = evidence.stop_reason or 'requested_transitions'
                        write_json(directory / 'partial_episode.json', episodes.partial(manifest['stop_reason']))
                        if evidence.completed_updates:
                            validate_updates(changes, evidence.optimizer_steps, evidence.completed_updates)
                            save_start = time.perf_counter()
                            model.save(directory / 'policy_final.zip')
                            observations = np.stack(episodes.observations)
                            actions, _ = model.predict(observations, deterministic=True)
                            np.savez(directory / 'reload_probe.npz', observations=observations, actions=actions)
                            manifest['checkpoint'] = dict(path='policy_final.zip', sha256=sha256(directory / 'policy_final.zip'),
                                                           state='after last complete optimization round',
                                                           trained_rollout_transitions=evidence.completed_updates * config.n_steps,
                                                           sampled_transitions=episodes.transitions,
                                                           discarded_partial_rollout=episodes.transitions - evidence.completed_updates * config.n_steps)
                            manifest['timing']['checkpoint_save_seconds'] = time.perf_counter() - save_start
                        manifest['status'] = 'PASS' if episodes.transitions == config.total_timesteps else 'INTERRUPTED'
                    manifest['sampling_quality'] = sampling_quality(episodes.episodes)
                    manifest['physics_steps'] = episodes.physics_steps
                    manifest['recovery_simulated_seconds'] = episodes.physics_steps * env.physics_dt
                    manifest['timing'].update(reset_seconds=monitor.reset_seconds, reset_calls=monitor.reset_calls,
                                             reset_scope='all reset calls, including vector auto reset after final done; settling excluded from recovery simulated seconds')
                    require(env_identity(env) == identity, 'Environment/model/source changed during run')
                    manifest['identity_after_execution'] = env_identity(env)
                manifest['environment_closed'] = env._closed
                plot_start = time.perf_counter()
                manifest['reward_plot'] = plot_reward(directory)
                manifest['timing']['plot_seconds'] = time.perf_counter() - plot_start
                final_status = manifest['status']
                if manifest['checkpoint'] is not None:
                    manifest['status'] = 'VERIFYING_RELOAD'
                write_json(directory / 'manifest.json', manifest)
                if manifest['checkpoint'] is not None:
                    reload_start = time.perf_counter()
                    manifest['reload'] = launch_reload(directory, config.reload_timeout_seconds)
                    manifest['timing']['independent_reload_seconds'] = time.perf_counter() - reload_start
                    manifest['status'] = final_status
            except BaseException as exc:
                error = exc
                manifest.update(status='ERROR', stop_reason='execution_error', error_type=type(exc).__name__,
                                error=str(exc), traceback=traceback.format_exc(),
                                error_evidence=repr(getattr(exc, 'evidence', None)))
                traceback.print_exc()
                if evidence is not None:
                    manifest['failure_counters'] = dict(sampled_transitions=episodes.transitions,
                                                        completed_rollouts=evidence.completed_rollouts,
                                                        completed_updates=evidence.completed_updates,
                                                        optimizer_steps=evidence.optimizer_steps)
            finally:
                if env is not None:
                    manifest['environment_closed'] = env._closed
                if episodes is not None:
                    manifest['sampling_quality'] = sampling_quality(episodes.episodes)
                    write_json(directory / 'partial_episode.json', episodes.partial(manifest.get('stop_reason')))
                manifest.update(ended_utc=utc_now(), core_sources_after_run=source_hashes())
                manifest['timing']['total_seconds'] = time.perf_counter() - start
                write_json(directory / 'manifest.json', manifest)
    if error is not None:
        raise error
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('probe', 'smoke', 'train'):
        sub = commands.add_parser(name)
        sub.add_argument('--run-dir', '--output-dir', required=True, type=Path)
        sub.add_argument('--config', type=Path, help='Saved resolved_config.json or a TrainConfig JSON object')
        sub.add_argument('--seed', type=int)
        sub.add_argument('--total-timesteps', type=int)
        sub.add_argument('--max-wall-seconds', type=float)
        sub.add_argument('--log-std-init', type=float)
        if name == 'probe':
            sub.add_argument('--policy', choices=('zero', 'untrained'), default='untrained')
            sub.add_argument('--episodes', type=int, default=20)
            sub.add_argument('--max-transitions', type=int, default=128)
        if name == 'train':
            sub.add_argument('--validated-run', required=True, type=Path)
    sub = commands.add_parser('reload-check')
    sub.add_argument('--run-dir', required=True, type=Path)
    sub.add_argument('--output', type=Path)
    sub = commands.add_parser('plot')
    sub.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'reload-check':
            result = reload_check(args.run_dir, args.output)
            print(json.dumps({k: result[k] for k in ('status', 'pid', 'python_executable', 'transitions')}, allow_nan=False))
            return 0
        if args.command == 'plot':
            print(json.dumps(plot_reward(args.run_dir), allow_nan=False))
            return 0
        values = {}
        if args.config:
            values = json.loads(args.config.read_text())
            require(isinstance(values, dict), 'Expected configuration object')
            values = values.get('training', values)
            require(isinstance(values, dict), 'Expected training configuration object')
        for key in ('seed', 'total_timesteps', 'max_wall_seconds', 'log_std_init'):
            if getattr(args, key) is not None:
                values[key] = getattr(args, key)
        config = TrainConfig.from_dict(values)
        planning = None
        if args.command == 'train':
            require(args.config is not None, 'Formal training requires an explicit saved configuration')
            planning = formal_plan(args.validated_run, config)
            if args.total_timesteps is None:
                values['total_timesteps'] = planning['planned_transitions']
                config = TrainConfig.from_dict(values)
            require(config.total_timesteps <= planning['planned_transitions'], 'Requested transitions exceed measured budget plan')
        result = run_experiment(config, args.run_dir, 'formal' if args.command == 'train' else args.command,
                                probe_policy=getattr(args, 'policy', 'untrained'),
                                probe_episodes=getattr(args, 'episodes', 20),
                                probe_transitions=getattr(args, 'max_transitions', 128), planning=planning)
        print(json.dumps(dict(run_dir=str(args.run_dir), status=result['status'],
                              stop_reason=result['stop_reason'], sampling_quality=result['sampling_quality']), allow_nan=False))
        return 0 if result['status'] in ('PASS', 'COMPLETE') else 2
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
