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
import random

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

from .env import ControlConfig, ControlledRecoveryEnv, EnvConfig, X2RecoveryEnv
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
    package = Path(__file__).resolve().parent
    root = Path(subprocess.check_output(['git', '-C', str(package),
                                         'rev-parse', '--show-toplevel'], text=True).strip())
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    snapshot = directory / 'source'
    snapshot.mkdir()
    logical_package = Path('src/x2_recovery/x2_recovery')
    paths = {logical_package / path.name: path for path in package.glob('*.py')}
    # A continuation may import a previous run's source snapshot. Record the
    # actual modules, including an explicit override if imports span directories.
    for name, module in tuple(sys.modules.items()):
        if name != 'x2_recovery' and not name.startswith('x2_recovery.'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None or Path(filename).suffix != '.py':
            continue
        actual = Path(filename).resolve()
        suffix = Path(*name.split('.')[1:])
        relative = (suffix / '__init__.py' if actual.name == '__init__.py'
                    else suffix.with_suffix('.py'))
        paths[logical_package / relative] = actual
    layout_root = package.parents[2] if package.parts[-3:] == ('src', 'x2_recovery', 'x2_recovery') else root
    for relative in map(Path, ('requirements.txt', 'src/x2_recovery/setup.py',
                              'src/x2_recovery/package.xml', '.gitignore')):
        candidate = layout_root / relative
        paths[relative] = candidate if candidate.is_file() else root / relative
    hashes = {}
    for relative, path in sorted(paths.items()):
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = sha256(target)
    (snapshot / 'tracked.diff').write_text(git('diff', 'HEAD', '--', *(str(p) for p in sorted(paths))))
    memory = {line.split(':')[0]: line.split(':')[1].strip()
              for line in Path('/proc/meminfo').read_text().splitlines()
              if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}
    return dict(git=dict(root=str(root), branch=git('branch', '--show-current'), head=git('rev-parse', 'HEAD'),
                         status=git('status', '--porcelain', '--untracked-files=all')),
                source_snapshot='source', source_hashes=hashes,
                source_paths={str(relative): str(path) for relative, path in sorted(paths.items())},
                tracked_diff_scope='current Git checkout; source snapshot follows actual imported package/module paths',
                cwd=str(Path.cwd()),
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


class ControlEpisodeLog:
    """Per-worker raw episode statistics; all samples are control-step boundaries."""
    fields = ('episode_id', 'env_index', 'reset_seed', 'global_transition', 'length',
              'physics_steps', 'simulated_seconds', 'return', 'terminated', 'truncated',
              'reason', 'success', 'max_height_m', 'tilt_at_max_height_deg', 'min_tilt_deg',
              'max_stable_seconds', 'max_bilateral_weight', 'other_weight_at_max_height',
              'torque_saturation_fraction', 'max_tracking_error_rad', 'raw_action_oob_fraction',
              'action_boundary_fraction', 'reward_terms')

    def __init__(self, directory, stack, physics_dt, workers):
        self.writer = csv_output(stack, directory / 'episode_diagnostics.csv', self.fields)
        self.current = [None] * workers
        self.trajectories = [[] for _ in range(workers)]
        self.dt, self.transitions, self.physics_steps = physics_dt, 0, 0
        self.episodes, self.observations = [], []

    def add(self, index, observation, raw_action, action, reward, terminated, truncated, info, global_step):
        require(np.isfinite(reward) and np.isclose(sum(info['reward_terms'].values()), reward,
                atol=2e-6, rtol=2e-6), 'Invalid raw controller reward/components')
        if len(self.observations) < 16:
            self.observations.append(np.array(observation, dtype=np.float32, copy=True))
        require(len(self.trajectories[index]) < 1000, 'Controller episode exceeds the 20-second trajectory bound')
        self.trajectories[index].append(json_value(dict(
            control_step=len(self.trajectories[index])+1, global_transition=global_step,
            observation=observation, raw_policy_action=raw_action, applied_policy_action=action,
            reward=float(reward), terminated=bool(terminated), truncated=bool(truncated), info=info)))
        if self.current[index] is None:
            self.current[index] = dict(length=0, physics_steps=0, return_value=0., terms={}, saturation=0.,
                height=-1., tilt=180., hold=0., bilateral=0., tracking=0., raw_oob=0., boundary=0.)
        c, state = self.current[index], info['state']
        steps = int(info['physics_steps_executed'])
        self.transitions += 1
        self.physics_steps += steps
        c['length'] += 1
        c['physics_steps'] += steps
        c['return_value'] += float(reward)
        c['saturation'] += info['torque_saturation_fraction'] * steps
        c['raw_oob'] += float(np.mean(np.abs(raw_action) > 1))
        c['boundary'] += float(np.mean(np.abs(action) >= 1))
        c['tracking'] = max(c['tracking'], info['controller']['target_tracking_max_rad'])
        c['tilt'] = min(c['tilt'], state['tilt_deg'])
        c['hold'] = max(c['hold'], info['stable_duration_s'])
        c['bilateral'] = max(c['bilateral'], min(state['left_weight'], state['right_weight']))
        if state['pelvis_height_m'] > c['height']:
            c.update(height=state['pelvis_height_m'], tilt_at_height=state['tilt_deg'],
                     other_at_height=state['other_weight'])
        for key, value in info['reward_terms'].items():
            c['terms'][key] = c['terms'].get(key, 0.) + value
        if terminated or truncated:
            require(np.isclose(c['physics_steps'] * self.dt, info['elapsed_sim_s'], atol=1e-8, rtol=0),
                    'Episode physical duration mismatch')
            row = dict(episode_id=len(self.episodes)+1, env_index=index, reset_seed=info['reset_seed'],
                global_transition=global_step, length=c['length'], physics_steps=c['physics_steps'],
                simulated_seconds=c['physics_steps']*self.dt, return_value=c['return_value'],
                terminated=terminated, truncated=truncated,
                reason=info['termination_reason'] or info['truncation_reason'], success=info['is_success'],
                max_height_m=c['height'], tilt_at_max_height_deg=c['tilt_at_height'], min_tilt_deg=c['tilt'],
                max_stable_seconds=c['hold'], max_bilateral_weight=c['bilateral'],
                other_weight_at_max_height=c['other_at_height'],
                torque_saturation_fraction=c['saturation']/c['physics_steps'],
                max_tracking_error_rad=c['tracking'], raw_action_oob_fraction=c['raw_oob']/c['length'],
                action_boundary_fraction=c['boundary']/c['length'], reward_terms=c['terms'])
            row['return'] = row.pop('return_value')
            self.episodes.append(row)
            written = dict(row, reward_terms=json.dumps(json_value(row['reward_terms']), allow_nan=False))
            self.writer.writerow(written)
            self.current[index] = None
            trajectory = self.trajectories[index]
            self.trajectories[index] = []
            return trajectory
        return None


def action_distribution_metrics(model, observations):
    """Observe marginal action variance without drawing actions or resetting gSDE noise."""
    observations = np.asarray(observations).reshape((-1, *model.observation_space.shape))
    require(len(observations) > 0 and np.isfinite(observations).all(), 'Invalid distribution diagnostic observations')
    indices = np.linspace(0, len(observations)-1, min(32, len(observations)), dtype=int)
    with torch.inference_mode():
        distribution = model.policy.get_distribution(torch.as_tensor(observations[indices], device=model.device))
        std = distribution.distribution.stddev.detach().cpu().numpy()
        weight_std = float(torch.exp(model.policy.log_std.detach()).mean())
    require(np.isfinite(std).all() and np.all(std > 0) and math.isfinite(weight_std),
            'Invalid action distribution standard deviation')
    return dict(noise_weight_std_mean=weight_std, action_distribution_std_min=float(std.min()),
                action_distribution_std_median=float(np.median(std)), action_distribution_std_max=float(std.max()),
                action_distribution_observations=len(indices))


class ControlTrainingEvidence(TrainingEvidence):
    """Reuse gradient hooks while accounting for vector workers and inherited Adam state."""
    def __init__(self, config, directory, stack, episodes, workers, inherited_steps, initial_timesteps):
        BaseCallback.__init__(self)
        self.config, self.directory, self.episodes = config, directory, episodes
        self.workers, self.inherited_steps, self.initial_timesteps = workers, inherited_steps, initial_timesteps
        self.completed_rollouts = self.completed_updates = self.optimizer_steps = 0
        self.gradient_tensors, self.gradient_max_abs = 0, 0.
        self.stop_reason = self.learn_start = self.cycle_start = None
        self.rollout_records, self.updates = [], []
        self.first_success = None
        self.progress = csv_output(stack, directory / 'progress.csv', (
            'rollout', 'sampled_transitions', 'cumulative_model_timesteps', 'trained_rollout_transitions',
            'optimizer_steps', 'cumulative_adam_steps', 'sb3_n_updates', 'learn_seconds', 'cycle_seconds',
            'optimization_seconds', 'transitions_per_second', 'policy_loss', 'value_loss', 'entropy_loss',
            'total_loss', 'approx_kl', 'clip_fraction', 'policy_std_mean', 'noise_weight_std_mean',
            'action_distribution_std_min', 'action_distribution_std_median', 'action_distribution_std_max',
            'action_distribution_observations', 'explained_variance', 'undefined_reason'))

    def optimizer_state(self, optimizer):
        steps = []
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    require(torch.isfinite(value).all().item(), 'Nonfinite optimizer state: '+key)
            if 'step' in state:
                steps.append(int(state['step'].item()))
        expected = self.inherited_steps + self.optimizer_steps
        require(not steps or min(steps) == max(steps) == expected, 'Adam step count disagrees with observed updates')
        return dict(step_min=min(steps) if steps else 0, step_max=max(steps) if steps else 0,
                    inherited_steps=self.inherited_steps, block_optimizer_steps=self.optimizer_steps)

    def _on_step(self):
        loc = self.locals
        for i in range(self.workers):
            info, done = loc['infos'][i], bool(loc['dones'][i])
            truncated = bool(info.get('TimeLimit.truncated', False))
            if done:
                require('terminal_observation' in info and truncated == bool(info['truncation_reason']),
                        'Vector terminal semantics changed')
            trajectory = self.episodes.add(i, loc['obs_tensor'][i].detach().cpu().numpy(), loc['actions'][i],
                loc['clipped_actions'][i], float(loc['rewards'][i]), done and not truncated, truncated,
                info, self.model.num_timesteps)
            active_trace = trajectory if trajectory is not None else self.episodes.trajectories[i]
            active_trace[-1]['cumulative_adam_steps_at_action'] = self.inherited_steps+self.optimizer_steps
            if info['is_success'] and self.first_success is None:
                require(done and not truncated and trajectory is not None,
                        'Success must belong to a complete terminated episode')
                # This callback runs before the rollout's PPO optimization. These
                # are the exact weights that sampled the successful actions.
                checkpoint = self.directory/'policy_first_success.zip'
                self.model.save(checkpoint, exclude=['train'])
                write_json(self.directory/'first_success_trajectory.json', dict(
                    scope='control-step observations/actions/info; success checked by the original tracker at every physical step',
                    env_index=i, seed=self.config.seed, reset_seed=info['reset_seed'], transitions=trajectory))
                self.first_success = dict(checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint),
                    trajectory='first_success_trajectory.json',
                    trajectory_sha256=sha256(self.directory/'first_success_trajectory.json'),
                    cumulative_model_timesteps=self.model.num_timesteps, env_index=i,
                    rollout=self.completed_rollouts+1, completed_optimization_rounds=self.completed_updates,
                    optimizer_steps_in_block=self.optimizer_steps, seed=self.config.seed, reset_seed=info['reset_seed'],
                    episode_transitions=len(trajectory), physics_steps=sum(t['info']['physics_steps_executed'] for t in trajectory),
                    episode_adam_versions=sorted({t['cumulative_adam_steps_at_action'] for t in trajectory}),
                    weights_scope='weights at the success transition; a long episode may span prior PPO updates',
                    controller_identity=json.loads((self.directory/'resolved_config.json').read_text())['identity'],
                    stochastic_success=True, deterministic_recovery='NOT_YET_VERIFIED')
                write_json(self.directory/'first_success.json', self.first_success)
        if self.stop_reason is not None:
            return False
        if time.perf_counter()-self.learn_start >= self.config.max_wall_seconds:
            self.stop_reason = 'wall_budget'
            return loc['n_steps'] + 1 == self.config.n_steps
        return True

    def _on_rollout_end(self):
        buffer, checks = self.model.rollout_buffer, {}
        shapes = dict(observations=(self.config.n_steps, self.workers, *self.model.observation_space.shape),
                      actions=(self.config.n_steps, self.workers, *self.model.action_space.shape))
        for name in ('observations', 'actions', 'rewards', 'values', 'log_probs', 'advantages', 'returns'):
            arr = getattr(buffer, name)
            require(arr.shape == shapes.get(name, (self.config.n_steps, self.workers)), 'Wrong rollout shape: '+name)
            require(arr.dtype == np.float32 and np.isfinite(arr).all(), 'Invalid rollout tensor: '+name)
            checks[name] = dict(shape=list(arr.shape), dtype=str(arr.dtype), finite=True,
                                min=float(arr.min()), max=float(arr.max()))
        self.completed_rollouts += 1
        self.rollout_records.append(checks)

    def after_update(self, model, optimization_seconds):
        metrics = model.logger.name_to_value
        names = dict(policy_loss='train/policy_gradient_loss', value_loss='train/value_loss',
                     entropy_loss='train/entropy_loss', total_loss='train/loss', approx_kl='train/approx_kl',
                     clip_fraction='train/clip_fraction')
        losses = {key: float(metrics[value]) for key, value in names.items()}
        require(np.isfinite(list(losses.values())).all(), 'Nonfinite update loss/KL/clip fraction')
        variance, reason = float(metrics['train/explained_variance']), ''
        if not math.isfinite(variance):
            require(np.var(model.rollout_buffer.returns) == 0, 'Undefined explained variance with nonconstant returns')
            variance, reason = None, 'returns variance is zero'
        self.completed_updates += 1
        elapsed, cycle = time.perf_counter()-self.learn_start, time.perf_counter()-self.cycle_start
        batch = self.config.n_steps*self.workers
        distribution_metrics = action_distribution_metrics(model, model.rollout_buffer.observations)
        row = dict(rollout=self.completed_updates, sampled_transitions=self.episodes.transitions,
            cumulative_model_timesteps=model.num_timesteps, trained_rollout_transitions=self.completed_updates*batch,
            optimizer_steps=self.optimizer_steps, cumulative_adam_steps=self.inherited_steps+self.optimizer_steps,
            sb3_n_updates=model._n_updates, learn_seconds=elapsed, cycle_seconds=cycle,
            optimization_seconds=optimization_seconds, transitions_per_second=batch/cycle,
            policy_std_mean=distribution_metrics['noise_weight_std_mean'],
            explained_variance=variance, undefined_reason=reason, **losses, **distribution_metrics)
        self.progress.writerow(row)
        self.updates.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        quality = sampling_quality(self.episodes.episodes)
        if quality['sampling_degenerate']:
            self.stop_reason = 'sampling_degenerate'
        elif elapsed >= self.config.max_wall_seconds:
            self.stop_reason = 'wall_budget'
        self.cycle_start = time.perf_counter()

    def summary(self, model):
        trained = self.completed_updates*self.config.n_steps*self.workers
        return dict(requested_transitions=self.config.total_timesteps, sampled_transitions=self.episodes.transitions,
            inherited_model_timesteps=self.initial_timesteps, cumulative_model_timesteps=model.num_timesteps,
            transitions_in_completed_training_rollouts=trained, completed_rollout_count=self.completed_rollouts,
            completed_optimization_rounds=self.completed_updates,
            partial_rollout_transitions=self.episodes.transitions-trained, optimizer_steps=self.optimizer_steps,
            sb3_n_updates=model._n_updates, adam_state=self.optimizer_state(model.policy.optimizer),
            rollout_checks=self.rollout_records, updates=self.updates,
            first_success=self.first_success,
            numerical_checks=dict(finite=True, rollout='seven arrays of every complete rollout',
                losses='update-level means and last evaluated minibatch total loss',
                gradients='every backward parameter tensor and every post-clip optimizer input',
                gradient_tensors=self.gradient_tensors, max_abs_backward_gradient=self.gradient_max_abs,
                parameters_and_optimizer='after every optimizer step'))


def controller_identity(env):
    identity = env_identity(env.env)
    identity['controller'] = json_value(env.resolved_config())
    identity['observation']['shape'] = list(env.observation_space.shape)
    identity['observation']['dtype'] = str(env.observation_space.dtype)
    identity['observation']['low'] = ['-inf']*env.observation_space.shape[0]
    identity['observation']['high'] = ['inf']*env.observation_space.shape[0]
    identity['action'] = dict(shape=list(env.action_space.shape), dtype=str(env.action_space.dtype),
                              low=env.action_space.low.tolist(), high=env.action_space.high.tolist())
    return identity


def _controller_factory(config_dict, control_dict, directory, index):
    def create():
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        config = TrainConfig.from_dict(config_dict)
        env = ControlledRecoveryEnv(X2RecoveryEnv(config.env, render_mode=None, capture_substeps=False),
                                    ControlConfig.from_dict(control_dict))
        return TimedMonitor(env, Path(directory)/f'worker_{index:02d}.monitor.csv')
    return create


def _adam_steps(model):
    steps = [int(v['step'].item()) for v in model.policy.optimizer.state.values() if 'step' in v]
    require(not steps or min(steps) == max(steps), 'Inconsistent inherited Adam steps')
    return max(steps) if steps else 0


def _save_rng(path):
    # Only locally generated state is loaded; this file is not an untrusted input format.
    torch.save(dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()), path)


def _load_rng(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def save_valid_update(directory, model, training, changes):
    """Publish an immutable checkpoint/RNG pair only after both files exist.

    An interrupted save leaves an unreferenced generation for diagnosis, while
    the previous pointer and its complete files remain available for recovery.
    """
    directory = Path(directory)
    generation = directory / 'checkpoints' / f"update-{training['completed_optimization_rounds']:06d}"
    generation.parent.mkdir(exist_ok=True)
    generation.mkdir()  # Never overwrite a previous generation or failed attempt.
    checkpoint, rng = generation / 'policy.zip', generation / 'rng.pt'
    model.save(checkpoint, exclude=['train'])
    _save_rng(rng)
    saved = dict(training=training, parameter_changes=changes,
                 checkpoint_path=str(checkpoint.relative_to(directory)),
                 rng_path=str(rng.relative_to(directory)),
                 checkpoint_sha256=sha256(checkpoint), rng_sha256=sha256(rng))
    write_json(directory / 'last_valid_update.json', saved)
    return saved


def valid_update_files(directory, saved):
    """Resolve and validate a published pair, including the older flat layout."""
    directory = Path(directory).resolve()
    paths = []
    for key, fallback in (('checkpoint', 'policy_last_update.zip'), ('rng', 'rng_last_update.pt')):
        relative = Path(saved.get(key + '_path', fallback))
        path = (directory / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(directory),
                'Last valid update path is outside its run directory')
        require(sha256(path) == saved[key + '_sha256'], 'Last valid ' + key + ' changed')
        paths.append(path)
    return tuple(paths)


def _state_digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def initialize_exploration_phase(model, factor, observations, expected_actions):
    """One declared log-std intervention, before this block's training baseline.

    Actor/critic, optimizer moments/counters, and RNG stay unchanged. SB3 resets
    gSDE noise at rollout start using the adjusted standard-deviation parameters.
    """
    require(type(factor) in (float, int) and math.isfinite(factor) and 0 < factor <= 1,
            'Initial exploration std factor must be finite and in (0, 1]')
    observations, expected_actions = np.asarray(observations), np.asarray(expected_actions)
    require(observations.ndim == 2 and len(observations) > 0
            and observations.shape[1:] == model.observation_space.shape
            and observations.dtype == np.float32 and np.isfinite(observations).all(),
            'Invalid exploration phase observations')
    require(expected_actions.shape == (len(observations), *model.action_space.shape)
            and np.isfinite(expected_actions).all(), 'Invalid exploration phase expected actions')
    before = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    require('log_std' in before and torch.isfinite(before['log_std']).all().item(),
            'Exploration phase requires finite policy log_std')
    optimizer_before = _state_digest(model.policy.optimizer.state_dict())
    rng = lambda: dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    rng_before = _state_digest(rng())
    counters_before = (model.num_timesteps, model._n_updates, _adam_steps(model))
    with torch.inference_mode():
        actions_before, _ = model.predict(observations, deterministic=True)
    require(np.allclose(actions_before, expected_actions, atol=1e-7, rtol=1e-6),
            'Parent deterministic action probe changed')
    if factor != 1:
        with torch.no_grad():
            model.policy.log_std.add_(math.log(factor))
    with torch.inference_mode():
        actions_after, _ = model.predict(observations, deterministic=True)
    require(np.array_equal(actions_before, actions_after), 'Exploration phase changed deterministic actions')
    after = model.policy.state_dict()
    require(before.keys() == after.keys() and all(torch.equal(value, after[key])
            for key, value in before.items() if key != 'log_std'),
            'Exploration phase changed actor, critic, or buffers')
    require(torch.isfinite(after['log_std']).all().item(), 'Exploration phase created nonfinite log_std')
    require(_state_digest(model.policy.optimizer.state_dict()) == optimizer_before,
            'Exploration phase changed optimizer state')
    require(_state_digest(rng()) == rng_before, 'Exploration phase consumed RNG')
    require((model.num_timesteps, model._n_updates, _adam_steps(model)) == counters_before,
            'Exploration phase changed update counters')
    def stats(value):
        array = value.detach().cpu().numpy()
        return dict(shape=list(array.shape), dtype=str(array.dtype),
                    sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                    min=float(array.min()), mean=float(array.mean()), max=float(array.max()),
                    exp_log_std_mean=float(np.exp(array).mean()))
    return dict(kind='one_time_exploration_std_scale', applied=factor != 1, factor=float(factor),
                operation='policy.log_std += log(factor); no optimizer step',
                before=stats(before['log_std']), after=stats(after['log_std']),
                optimizer_state_sha256=optimizer_before, optimizer_state_unchanged=True,
                rng_state_sha256=rng_before, rng_state_unchanged=True,
                inherited_timesteps=counters_before[0], inherited_sb3_updates=counters_before[1],
                inherited_adam_steps=counters_before[2], actor_critic_and_buffers_unchanged=True,
                action_consistency=dict(observations=len(observations),
                    parent_probe_max_abs_error=float(np.max(abs(actions_before-expected_actions))),
                    intervention_max_abs_error=float(np.max(abs(actions_after-actions_before))),
                    parent_atol=1e-7, parent_rtol=1e-6, intervention_exact=True),
                training_delta_baseline='captured after this intervention',
                subsequent_resume='factor defaults to 1; this intervention is not automatically replayed')


def run_control_block(config, control, directory, *, workers=1, use_sde=False, sde_sample_freq=8,
                      resume_run=None, initial_exploration_std_factor=1.):
    require(type(initial_exploration_std_factor) in (float, int)
            and math.isfinite(initial_exploration_std_factor) and 0 < initial_exploration_std_factor <= 1,
            'Initial exploration std factor must be finite and in (0, 1]')
    require(initial_exploration_std_factor == 1 or resume_run is not None,
            'Initial exploration std reduction requires --resume-run')
    require(workers in (1, 2, 4) and type(workers) is int, 'workers must be 1, 2 or 4')
    require(type(use_sde) is bool and type(sde_sample_freq) is int and sde_sample_freq > 0, 'Invalid exploration settings')
    require(config.max_wall_seconds <= 1800, 'A control training block is limited to 1800 seconds')
    require(config.total_timesteps % (config.n_steps*workers) == 0, 'Transitions must divide full vector rollout size')
    require(config.env == EnvConfig(), 'Control experiments retain the original physical task configuration')
    directory = Path(directory).resolve()
    require(not directory.exists(), 'Output directory already exists: '+str(directory))
    parent = None
    algorithm = dict(training=asdict(config), workers=workers, use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                     controller=asdict(control))
    if resume_run is not None:
        resume_run = Path(resume_run).resolve()
        parent = json.loads((resume_run/'manifest.json').read_text())
        require(sha256(resume_run/'resolved_config.json') == parent['config_sha256'],
                'Parent resolved configuration changed')
        old = json.loads((resume_run/'resolved_config.json').read_text())
        comparable = json_value(algorithm)
        previous = {key: old[key] for key in comparable}
        for value in (comparable, previous):
            value['training'] = dict(value['training'])
            for key in ('total_timesteps', 'max_wall_seconds'):
                value['training'].pop(key)
        require(comparable == previous, 'Resume configuration differs beyond block budgets')
        require(parent.get('checkpoint') is not None and parent['checkpoint']['sha256'] == sha256(resume_run/'policy_final.zip'),
                'Parent checkpoint missing or changed')
        require(parent['rng_sha256'] == sha256(resume_run/'rng_state.pt'), 'Parent RNG state changed')
    directory.mkdir(parents=True)
    started, begin = utc_now(), time.perf_counter()
    manifest = dict(run_id=directory.name, run_type='controller_ppo_block', status='RUNNING', started_utc=started,
                    normalization='fixed_env_scaling', formal_evaluation='NOT_EVALUATED', timing={},
                    checkpoint=None, resume_from=None if parent is None else dict(
                        path=str(resume_run), checkpoint_sha256=parent['checkpoint']['sha256'],
                        state='model/optimizer/global RNG restored; fresh legal supine episodes, not bitwise simulator resume'))
    model = evidence = episodes = vec = None
    error = None
    with ExitStack() as stack:
        log = stack.enter_context((directory/'stdout.log').open('x', buffering=1))
        with redirect_stdout(log), redirect_stderr(log):
            try:
                configure_threads(config)
                manifest['provenance'] = provenance(directory)
                probe = ControlledRecoveryEnv(X2RecoveryEnv(config.env), control)
                stack.callback(probe.close)
                identity = controller_identity(probe)
                if parent is not None:
                    require(old['identity'] == identity, 'Parent model/control/core identity differs')
                factories = [_controller_factory(asdict(config), asdict(control), str(directory), i) for i in range(workers)]
                plain = DummyVecEnv(factories) if workers == 1 else SubprocVecEnv(factories, start_method='spawn')
                stack.callback(plain.close)
                vec = VecCheckNan(plain, raise_exception=True, warn_once=False)
                # The worker factories use one thread; keep parent optimization at the explicit budget.
                configure_threads(config)
                vec.seed(config.seed)
                if parent is None:
                    model = PPO('MlpPolicy', vec, device='cpu', seed=config.seed, verbose=0,
                        n_steps=config.n_steps, batch_size=config.batch_size, n_epochs=config.n_epochs,
                        learning_rate=config.learning_rate, gamma=config.gamma, gae_lambda=config.gae_lambda,
                        clip_range=config.clip_range, clip_range_vf=config.clip_range_vf,
                        normalize_advantage=config.normalize_advantage, ent_coef=config.ent_coef,
                        vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm, target_kl=config.target_kl,
                        use_sde=use_sde, sde_sample_freq=sde_sample_freq, policy_kwargs=policy_kwargs(config))
                else:
                    model = PPO.load(resume_run/'policy_final.zip', env=vec, device='cpu')
                    require(model.num_timesteps == parent['training']['cumulative_model_timesteps'], 'Inherited timestep count differs')
                    _load_rng(resume_run/'rng_state.pt')
                phase = dict(kind='one_time_exploration_std_scale', applied=False, factor=1.)
                if initial_exploration_std_factor != 1:
                    with np.load(resume_run/'reload_probe.npz', allow_pickle=False) as parent_probe:
                        phase = initialize_exploration_phase(model, initial_exploration_std_factor,
                            parent_probe['observations'], parent_probe['actions'])
                    phase['parent_checkpoint_sha256'] = parent['checkpoint']['sha256']
                    phase['parent_probe_sha256'] = sha256(resume_run/'reload_probe.npz')
                    require(sha256(resume_run/'policy_final.zip') == parent['checkpoint']['sha256'],
                            'Parent checkpoint changed during phase initialization')
                manifest['initialization_phase'] = phase
                logger = configure(str(directory/'sb3'), ['csv'])
                stack.callback(logger.close)
                model.set_logger(logger)
                resolved = dict(**algorithm, schema='x2-controller-ppo-v1', identity=identity,
                    initialization_phase=phase,
                    environment=probe.resolved_config(), device='cpu', normalization='fixed_env_scaling',
                    module_paths=dict(train=str(Path(__file__).resolve()),
                        env=str(Path(sys.modules[X2RecoveryEnv.__module__].__file__).resolve()),
                        ppo=str(Path(sys.modules[PPO.__module__].__file__).resolve())),
                    policy_resolved=dict(representation=str(model.policy), optimizer=type(model.policy.optimizer).__name__,
                        optimizer_defaults=model.policy.optimizer.defaults, activation='Tanh', ortho_init=True),
                    capture_substeps=False, reset='original legal supine reset',
                    exploration_metric_definitions=dict(
                        policy_std_mean='legacy alias for noise_weight_std_mean; not gSDE marginal action standard deviation',
                        noise_weight_std_mean='mean exp(log_std); gSDE noise matrix weights or diagonal Gaussian action standard deviation',
                        action_distribution_std='marginal Gaussian standard deviation after the update on up to32 evenly spaced real rollout observations, all action dimensions; no random samples'),
                    metrics='control-step-sampled height/support/hold; physical-step-weighted saturation; raw episode reward')
                write_json(directory/'resolved_config.json', resolved)
                manifest['config_sha256'] = sha256(directory/'resolved_config.json')
                manifest['identity'] = identity
                manifest['timing']['construction_seconds'] = time.perf_counter()-begin
                episodes = ControlEpisodeLog(directory, stack, probe.env.physics_dt, workers)
                evidence = ControlTrainingEvidence(config, directory, stack, episodes, workers, _adam_steps(model), model.num_timesteps)
                evidence.install_hooks(model, stack)
                before = parameter_copy(model.policy)
                original_train = model.train
                def observed_train():
                    start = time.perf_counter()
                    original_train()
                    evidence.after_update(model, time.perf_counter()-start)
                    validate_updates(parameter_changes(before, model.policy), evidence.optimizer_steps,
                                     evidence.completed_updates)
                    save_valid_update(directory, model, evidence.summary(model),
                                      parameter_changes(before, model.policy))
                model.train = observed_train
                stack.callback(lambda: model.__dict__.pop('train', None))
                evidence.learn_start = time.perf_counter()
                try:
                    model.learn(config.total_timesteps, callback=evidence, reset_num_timesteps=parent is None)
                finally:
                    manifest['timing']['learn_seconds'] = time.perf_counter()-evidence.learn_start
                manifest['training'] = evidence.summary(model)
                manifest['training']['parameter_changes'] = parameter_changes(before, model.policy)
                manifest['sampling_quality'] = sampling_quality(episodes.episodes)
                manifest['stop_reason'] = evidence.stop_reason or 'requested_transitions'
                validate_updates(manifest['training']['parameter_changes'], evidence.optimizer_steps, evidence.completed_updates)
                save_start = time.perf_counter()
                model.save(directory/'policy_final.zip', exclude=['train'])
                _save_rng(directory/'rng_state.pt')
                manifest['rng_sha256'] = sha256(directory/'rng_state.pt')
                manifest['checkpoint'] = dict(path='policy_final.zip', sha256=sha256(directory/'policy_final.zip'),
                    trained_rollout_transitions=manifest['training']['transitions_in_completed_training_rollouts'],
                    cumulative_model_timesteps=model.num_timesteps, actual_block_optimizer_steps=evidence.optimizer_steps,
                    discarded_partial_rollout=manifest['training']['partial_rollout_transitions'])
                model.policy.set_training_mode(False)
                observations = np.asarray(episodes.observations, dtype=np.float32)
                with torch.inference_mode():
                    actions, _ = model.predict(observations, deterministic=True)
                np.savez_compressed(directory/'reload_probe.npz', observations=observations, actions=actions)
                manifest['timing']['save_seconds'] = time.perf_counter()-save_start
                require(controller_identity(probe) == identity, 'Controller identity changed during training')
                manifest['reset_cost'] = dict(seconds=vec.get_attr('reset_seconds'), calls=vec.get_attr('reset_calls'))
                manifest['training']['transitions_per_second'] = episodes.transitions/manifest['timing']['learn_seconds']
                manifest['training']['physics_steps'] = episodes.physics_steps
                manifest['training']['recovery_simulated_seconds'] = episodes.physics_steps*probe.env.physics_dt
                manifest['status'] = 'COMPLETE' if evidence.stop_reason is None else 'BUDGET_STOP' if evidence.stop_reason == 'wall_budget' else 'STOPPED'
            except Exception as exc:
                error = exc
                manifest.update(status='ERROR', stop_reason=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
                if evidence is not None:
                    manifest['observed_before_error'] = dict(sampled_transitions=episodes.transitions,
                        completed_rollouts=evidence.completed_rollouts, optimizer_steps=evidence.optimizer_steps)
                if (directory/'last_valid_update.json').exists():
                    safe = json.loads((directory/'last_valid_update.json').read_text())
                    checkpoint, rng = valid_update_files(directory, safe)
                    shutil.copyfile(checkpoint, directory/'policy_final.zip')
                    shutil.copyfile(rng, directory/'rng_state.pt')
                    manifest['rng_sha256'] = safe['rng_sha256']
                    manifest['training'] = safe['training']
                    manifest['training']['parameter_changes'] = safe['parameter_changes']
                    manifest['checkpoint'] = dict(path='policy_final.zip', sha256=safe['checkpoint_sha256'],
                        state='last complete validated optimization before error; see observed_before_error',
                        recovered_from=str(checkpoint.relative_to(directory)),
                        recovered_rng_from=str(rng.relative_to(directory)),
                        cumulative_model_timesteps=safe['training']['cumulative_model_timesteps'],
                        trained_rollout_transitions=safe['training']['transitions_in_completed_training_rollouts'])
                traceback.print_exc()
            finally:
                if episodes is not None:
                    write_json(directory/'partial_episodes.json', episodes.current)
                manifest.update(ended_utc=utc_now(), core_sources_after_run=source_hashes())
                manifest['timing']['total_seconds'] = time.perf_counter()-begin
                write_json(directory/'manifest.json', manifest)
    if error is not None:
        raise error
    plot_start = time.perf_counter()
    manifest['reward_plot'] = plot_reward(directory)
    manifest['timing']['plot_seconds'] = time.perf_counter()-plot_start
    manifest['timing']['total_seconds'] = time.perf_counter()-begin
    write_json(directory/'manifest.json', manifest)
    return manifest


def control_reload_check(directory, *, seed=221020, max_wall_seconds=180.):
    """Independent-process development validation, never a formal five-episode batch."""
    directory = Path(directory).resolve()
    resolved = json.loads((directory/'resolved_config.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    require(resolved['schema'] == 'x2-controller-ppo-v1', 'Unexpected controller checkpoint schema')
    require(sha256(directory/'resolved_config.json') == manifest['config_sha256'], 'Configuration changed')
    require(sha256(directory/'policy_final.zip') == manifest['checkpoint']['sha256'], 'Checkpoint changed')
    require(type(seed) is int and 0 <= seed < 2**32 and math.isfinite(max_wall_seconds) and 0 < max_wall_seconds <= 300,
            'Invalid development check budget/seed')
    output = directory/f'development-{seed}'
    require(not output.exists(), 'Development evidence already exists')
    output.mkdir()
    config = TrainConfig.from_dict(resolved['training'])
    configure_threads(config)
    env = ControlledRecoveryEnv(X2RecoveryEnv(config.env, capture_substeps=True), ControlConfig.from_dict(resolved['controller']))
    result = dict(status='RUNNING', seed=seed, label='development_reload_validation', pid=os.getpid(),
                  python_executable=sys.executable, checkpoint=str(directory/'policy_final.zip'),
                  checkpoint_sha256=sha256(directory/'policy_final.zip'), transitions=0, physics_steps=0)
    start = time.perf_counter()
    try:
        require(controller_identity(env) == resolved['identity'], 'Saved environment/controller identity differs')
        model = PPO.load(directory/'policy_final.zip', device='cpu')
        model.policy.set_training_mode(False)
        before = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
        with np.load(directory/'reload_probe.npz', allow_pickle=False) as probe:
            with torch.inference_mode():
                actual, _ = model.predict(probe['observations'], deterministic=True)
            require(actual.shape == probe['actions'].shape and np.isfinite(actual).all() and np.all(abs(actual) <= 1),
                    'Invalid loaded deterministic actions')
            require(np.allclose(actual, probe['actions'], atol=1e-7, rtol=1e-6), 'Reload deterministic actions differ')
            result['action_consistency'] = dict(max_abs_error=float(np.max(abs(actual-probe['actions']))), atol=1e-7, rtol=1e-6)
        with (output/'trajectory.jsonl').open('x', buffering=1) as stream:
            observation, info = env.reset(seed=seed)
            result.update(reset_seed=info['reset_seed'], reset_status=env.env.reset_evidence['status'],
                max_height_m=info['state']['pelvis_height_m'], min_tilt_deg=info['state']['tilt_deg'], max_stable_seconds=0.)
            stream.write(json.dumps(json_value(dict(kind='reset', seed=seed, info=info,
                evidence=env.env.reset_evidence, observation=observation,
                qpos=env.env.data.qpos.copy(), qvel=env.env.data.qvel.copy())), allow_nan=False)+'\n')
            while True:
                require(time.perf_counter()-start < max_wall_seconds, 'Development reload wall budget exceeded')
                with torch.inference_mode():
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                result['transitions'] += 1
                result['physics_steps'] += info['physics_steps_executed']
                for sample in env.env.last_substeps:
                    v, tracker = sample['measurement'], sample['success']
                    result['max_height_m'] = max(result['max_height_m'], v['pelvis_height_m'])
                    result['min_tilt_deg'] = min(result['min_tilt_deg'], v['tilt_deg'])
                    result['max_stable_seconds'] = max(result['max_stable_seconds'], tracker['stable_duration_s'])
                post_q, post_dq = env.env.loaded.read_state(env.env.data)
                last = env.env.last_substeps[-1]
                stream.write(json.dumps(json_value(dict(kind='transition', control_step=result['transitions'],
                    action=action, reward=reward, terminated=terminated, truncated=truncated, info=info,
                    post_step=dict(time_s=float(env.env.data.time), q_rad=post_q, dq_rad_s=post_dq,
                                   ctrl_Nm=env.env.data.ctrl[env.env.context.ctrladr].copy()),
                    last_substep_control=dict(pre_time_s=last['measurement']['time_s']-env.env.physics_dt,
                        post_time_s=last['measurement']['time_s'], target_rad=last['target_rad'],
                        pre_q_rad=last['q_rad'], pre_dq_rad_s=last['dq_rad_s'],
                        pre_tau_raw_Nm=last['tau_raw_Nm'], post_applied_torque_Nm=last['applied_torque_Nm']),
                    substeps=[dict(measurement=s['measurement'], success=s['success']) for s in env.env.last_substeps])), allow_nan=False)+'\n')
                if terminated or truncated:
                    require(np.isclose(result['physics_steps']*env.env.physics_dt, info['elapsed_sim_s'],
                                       atol=1e-8, rtol=0), 'Development physical duration mismatch')
                    result.update(status='COMPLETE', success=info['is_success'], terminated=terminated, truncated=truncated,
                        reason=info['termination_reason'] or info['truncation_reason'], sim_duration_s=info['elapsed_sim_s'], final_info=info)
                    break
        require(all(torch.equal(value, before[key]) for key, value in model.policy.state_dict().items()), 'Inference changed policy state')
        require(controller_identity(env) == resolved['identity'], 'Environment changed during development check')
        require(sha256(directory/'policy_final.zip') == result['checkpoint_sha256'], 'Checkpoint file changed')
        result['policy_state_unchanged'] = True
    except Exception as exc:
        result.update(status='ERROR', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        env.close()
        result['wall_seconds'] = time.perf_counter()-start
        write_json(output/'summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('control-block', help='Bounded PPO on an explicitly versioned learning controller')
    sub.add_argument('--run-dir', required=True, type=Path)
    sub.add_argument('--control-config', required=True, type=Path)
    sub.add_argument('--total-timesteps', required=True, type=int)
    sub.add_argument('--max-wall-seconds', required=True, type=float)
    sub.add_argument('--seed', required=True, type=int)
    sub.add_argument('--workers', choices=(1, 2, 4), type=int, default=1)
    sub.add_argument('--resume-run', type=Path)
    sub.add_argument('--initial-exploration-std-factor', type=float, default=1.,
                     help='One-time post-load std scaling in (0,1]; reductions require --resume-run')
    sub.add_argument('--use-sde', action='store_true')
    sub.add_argument('--sde-sample-freq', type=int, default=8)
    sub.add_argument('--learning-rate', type=float, default=1e-4)
    sub.add_argument('--log-std-init', type=float, default=-1.5)
    sub.add_argument('--target-kl', type=float, default=.03)
    sub.add_argument('--n-epochs', type=int, default=5)
    sub = commands.add_parser('control-reload', help='One complete deterministic development attempt in this process')
    sub.add_argument('--run-dir', required=True, type=Path)
    sub.add_argument('--seed', type=int, default=221020)
    sub.add_argument('--max-wall-seconds', type=float, default=180.)
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
        if args.command == 'control-block':
            values = json.loads(args.control_config.read_text())
            require(isinstance(values, dict), 'Expected controller configuration object')
            control = ControlConfig.from_dict(values.get('controller', values))
            config = TrainConfig(seed=args.seed, total_timesteps=args.total_timesteps,
                max_wall_seconds=args.max_wall_seconds, n_steps=512//args.workers,
                learning_rate=args.learning_rate, log_std_init=args.log_std_init,
                target_kl=args.target_kl, n_epochs=args.n_epochs)
            result = run_control_block(config, control, args.run_dir, workers=args.workers,
                use_sde=args.use_sde, sde_sample_freq=args.sde_sample_freq, resume_run=args.resume_run,
                initial_exploration_std_factor=args.initial_exploration_std_factor)
            print(json.dumps(dict(run_dir=str(args.run_dir), status=result['status'],
                training=result['training'], stop_reason=result['stop_reason']), allow_nan=False))
            return 0 if result['status'] in ('COMPLETE', 'BUDGET_STOP', 'STOPPED') else 2
        if args.command == 'control-reload':
            result = control_reload_check(args.run_dir, seed=args.seed, max_wall_seconds=args.max_wall_seconds)
            print(json.dumps(result, allow_nan=False))
            return 0
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
