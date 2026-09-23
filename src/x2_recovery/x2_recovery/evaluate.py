"""Five frozen, deterministic PPO attempts; no training and no automatic reset.

The environment owns physics, reset and success. This module records its evidence
and independently checks the bookkeeping, not a second standing criterion.
"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict, fields
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO

from .env import ControlConfig, ControlledRecoveryEnv, EnvConfig, X2RecoveryEnv, _diagnostic
from .model import COMMIT, REPOSITORY, require
from .train import (TrainConfig, compare_actions, configure_threads, controller_identity, env_identity,
                    json_value, sha256, utc_now, write_json)

CHECKPOINT_SHA256 = 'e8fa7e642e8ccd8d6dd38aaf9d810eee3c00d39d4188f82702c5b8e6124b213e'
INPUT_FILES = ('policy_final.zip', 'resolved_config.json', 'manifest.json',
               'reload_check.json', 'reload_probe.npz', 'progress.csv')
SOURCE_FILES = ('__init__.py', 'evaluate.py', 'train.py', 'env.py', 'reset.py',
                'success.py', 'model.py', 'model_audit.py', 'baseline.py')
COLUMNS = ('episode', 'seed', 'reset_seed', 'controller', 'success', 'sim_duration_s',
           'max_pelvis_height_m', 'max_stable_hold_s', 'terminated', 'truncated',
           'termination_reason')
TIME_ATOL = 1e-8  # Bookkeeping roundoff only; never changes the success tracker.
STATE_ATOL = 1e-10
CONTROLLER_SCHEMA = 'x2-controller-ppo-v1'


def strict_json(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), 'Nonfinite JSON number: ' + value)
        return number
    return json.loads(text, parse_constant=invalid, parse_float=finite_float)


def read_json(path):
    return strict_json(Path(path).read_text())


def encoded(value):
    return json.dumps(json_value(value), allow_nan=False, separators=(',', ':'))


def validate_seeds(seeds):
    require(len(seeds) == 5, 'Exactly five seeds are required')
    require(all(type(s) is int and 0 <= s < 2**32 for s in seeds),
            'Seeds must be integers in [0, 2**32)')
    require(len(set(seeds)) == 5, 'Seeds must be distinct')
    return list(seeds)


def input_sources(directory, controlled, reload_validation=None):
    """Give external reload evidence a fixed, portable name in the input bundle."""
    directory = Path(directory).resolve()
    if not controlled:
        require(reload_validation is None, 'Legacy smoke evaluation uses its recorded reload_check.json')
        return {name: directory / name for name in INPUT_FILES}
    names = [name for name in INPUT_FILES if name != 'reload_check.json']
    if reload_validation is None:
        portable = directory / 'reload_validation.json'
        reload_validation = portable if portable.is_file() else directory / 'development-221020' / 'summary.json'
    return {**{name: directory / name for name in names},
            'reload_validation.json': Path(reload_validation).resolve()}


def controller_label(saved):
    return 'ppo_' + saved['controller']['mode'] if saved.get('schema') == CONTROLLER_SCHEMA else 'ppo'


def physical_env(env):
    return env.env if isinstance(env, ControlledRecoveryEnv) else env


def load_inputs(directory, *, expected_checkpoint_sha256=None, reload_validation=None):
    directory = Path(directory).resolve()
    require((directory / 'resolved_config.json').is_file(), 'Missing input: ' + str(directory / 'resolved_config.json'))
    saved = read_json(directory / 'resolved_config.json')
    controlled = saved.get('schema') == CONTROLLER_SCHEMA
    sources = input_sources(directory, controlled, reload_validation)
    for path in sources.values():
        require(path.is_file(), 'Missing input: ' + str(path))
    hashes = {name: sha256(path) for name, path in sources.items()}
    training = read_json(directory / 'manifest.json')
    reload = read_json(sources['reload_validation.json' if controlled else 'reload_check.json'])
    if controlled:
        require(isinstance(expected_checkpoint_sha256, str) and len(expected_checkpoint_sha256) == 64
                and all(c in '0123456789abcdef' for c in expected_checkpoint_sha256),
                'Controller evaluation requires an explicit expected checkpoint SHA-256')
        require(training['run_type'] == 'controller_ppo_block'
                and training['status'] in ('COMPLETE', 'BUDGET_STOP', 'STOPPED', 'INTERRUPTED'),
                'Expected a controller PPO experiment with a validated checkpoint')
    else:
        require(expected_checkpoint_sha256 in (None, CHECKPOINT_SHA256), 'Legacy smoke checkpoint is fixed')
        expected_checkpoint_sha256 = CHECKPOINT_SHA256
        require(training['run_type'] == 'smoke' and training['status'] == 'PASS',
                'Expected the verified small smoke experiment')
    require(hashes['policy_final.zip'] == expected_checkpoint_sha256 == training['checkpoint']['sha256'],
            'Checkpoint identity mismatch')
    require(hashes['resolved_config.json'] == training['config_sha256'], 'Configuration hash mismatch')
    evidence = training['training']
    require(evidence['optimizer_steps'] > 0 and evidence['transitions_in_completed_training_rollouts'] > 0,
            'Missing actual optimization evidence')
    for group in ('actor_mean', 'critic'):
        require(evidence['parameter_changes'][group]['l2'] > 0, 'Missing update: ' + group)
    require(evidence['numerical_checks']['finite'] is True, 'Training numerical checks did not pass')
    require(saved['normalization'] == training['normalization'] == 'fixed_env_scaling',
            'Unsupported normalization')
    if not controlled:
        require(saved['extra_clipping'] is False and saved['algorithm'] == 'stable_baselines3.PPO',
                'Unsupported inference pipeline')
    else:
        require(saved['device'] == 'cpu' and saved['capture_substeps'] is False
                and type(saved['use_sde']) is bool and type(saved['workers']) is int
                and saved['workers'] in (1, 2, 4), 'Unsupported controller training pipeline')
    require(set(saved['training']) == {f.name for f in fields(TrainConfig)}, 'Incomplete training configuration')
    require(set(saved['training']['env']) == {f.name for f in fields(EnvConfig)}, 'Incomplete environment configuration')
    config = TrainConfig.from_dict(saved['training'])
    require(saved['identity'] == training['identity'], 'Recorded environment identities disagree')
    if controlled:
        control_fields = {field.name for field in fields(ControlConfig)}
        supplied_fields = set(saved['controller'])
        version = saved['controller'].get('version')
        optional_legacy = {'reference_interpolation'}
        if version not in ('targets-v4', 'targets-v5'):
            optional_legacy.update(('residual_start_s', 'residual_ramp_s'))
        if saved['controller'].get('reward_version') not in ('reference-head-support-v1', 'reference-balance-v1'):
            optional_legacy.update(('head_reference', 'head_progress_range_m'))
        if version in ('targets-v1', 'targets-v2'):
            optional_legacy.add('action_layout')
        if version == 'targets-v1':
            optional_legacy.add('residual_joint_multipliers')
        require(supplied_fields <= control_fields and control_fields - supplied_fields <= optional_legacy,
                'Incomplete controller configuration')
        control = ControlConfig.from_dict(saved['controller'])
        require(saved['environment'] == saved['identity']['controller'], 'Saved controller environment mismatch')
        require(json_value(asdict(control)) == json_value(asdict(ControlConfig.from_dict(saved['environment']['controller']))),
                'Saved controller configuration mismatch')
        environment = saved['environment']['base']
        require(environment == saved['identity']['resolved_env'], 'Saved physical environment mismatch')
        action_shape = [{'joint31': 31, 'bilateral11': 11, 'independentlegs17': 17}[control.action_layout]]
        require(saved['environment']['observation_shape'] == [149]
                and saved['environment']['action_shape'] == action_shape
                and saved['identity']['observation']['shape'] == [149]
                and saved['identity']['action']['shape'] == action_shape, 'Controller interface shape mismatch')
        require(reload['status'] == 'COMPLETE' and reload['label'] == 'development_reload_validation'
                and reload['reset_status'] == 'PASS' and reload['policy_state_unchanged'] is True
                and reload['checkpoint_sha256'] == hashes['policy_final.zip'], 'Missing verified controller reload')
        require(type(reload['terminated']) is bool and type(reload['truncated']) is bool
                and reload['terminated'] != reload['truncated'] and reload['transitions'] > 0
                and reload['physics_steps'] > 0, 'Reload did not complete a valid episode')
        reason = reload['reason']
        require((reload['truncated'] and reason == 'time_limit')
                or (reload['terminated'] and (reason == 'success' or str(reason).startswith('safety_abort:'))),
                'Invalid controller reload termination')
        require(reload['success'] == (reason == 'success') == reload['final_info']['is_success'],
                'Controller reload success evidence disagrees')
        require(math.isclose(reload['physics_steps'] * environment['physics_dt'], reload['sim_duration_s'],
                             abs_tol=TIME_ATOL, rel_tol=0), 'Reload physical duration mismatch')
        require(0 < reload['sim_duration_s'] <= config.env.episode_timeout_s + TIME_ATOL
                and (not reload['truncated'] or math.isclose(reload['sim_duration_s'], config.env.episode_timeout_s,
                                                            abs_tol=TIME_ATOL, rel_tol=0)),
                'Controller reload did not reach its recorded episode boundary')
        require(math.isclose(reload['final_info']['elapsed_sim_s'], reload['sim_duration_s'],
                             abs_tol=TIME_ATOL, rel_tol=0)
                and (reload['final_info']['termination_reason'] or reload['final_info']['truncation_reason']) == reason,
                'Controller reload terminal info disagrees')
        require(reload['action_consistency']['atol'] == 1e-7 and reload['action_consistency']['rtol'] == 1e-6,
                'Controller reload tolerances differ from the verified protocol')
    else:
        environment = saved['environment']
        require(saved['identity'] == reload['compatibility'], 'Recorded environment identities disagree')
        require(environment == saved['identity']['resolved_env'], 'Saved environment mismatch')
        require(reload['status'] == 'PASS' and reload['complete_episode'] is True
                and reload['checkpoint_sha256'] == hashes['policy_final.zip'], 'Missing verified reload')
    require(json_value(config.env.__dict__) == environment['config'], 'Environment reconstruction mismatch')
    source = training['provenance']['model_source']
    require(source['repository'] == REPOSITORY and source['commit'] == COMMIT, 'Model source mismatch')
    for name in ('mujoco', 'gymnasium', 'stable-baselines3', 'torch', 'numpy'):
        require(importlib.metadata.version(name) == training['provenance']['dependencies'][name],
                'Dependency changed since training: ' + name)
    return directory, config, saved, training, reload, hashes


def policy_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}


def policy_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(torch.isfinite(value).all().item(), 'Nonfinite policy tensor: ' + name)
        h.update(encoded([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_policy(model, original):
    current = model.policy.state_dict()
    require(current.keys() == original.keys(), 'Policy state keys changed')
    for name, value in current.items():
        require(torch.equal(value.detach().cpu(), original[name]), 'Policy state changed: ' + name)
    require(not model.policy.training, 'Policy left inference mode')
    return policy_hash(current)


def validate_environment(env, saved, model):
    identity = controller_identity(env) if saved.get('schema') == CONTROLLER_SCHEMA else env_identity(env)
    require(identity == saved['identity'], 'Environment/model/interface identity mismatch')
    for name in ('observation_space', 'action_space'):
        actual, loaded = getattr(env, name), getattr(model, name)
        require(actual.shape == loaded.shape and actual.dtype == loaded.dtype
                and np.array_equal(actual.low, loaded.low) and np.array_equal(actual.high, loaded.high),
                'Checkpoint space mismatch: ' + name)
    return identity


def compare_controller_actions(model, observations, expected, *, atol, rtol):
    """The saved probe uses the controller's observation interface, not native117."""
    require(observations.ndim == 2 and len(observations) > 0
            and observations.shape[1:] == model.observation_space.shape == (149,)
            and observations.dtype == np.float32 and np.isfinite(observations).all(),
            'Invalid saved controller observations')
    actual, _ = model.predict(observations, deterministic=True)
    require(model.action_space.shape in ((31,), (11,), (17,))
            and actual.shape == expected.shape == (len(observations), *model.action_space.shape),
            'Reload action shape mismatch')
    require(np.isfinite(actual).all() and np.isfinite(expected).all()
            and np.all(abs(actual) <= 1) and np.all(abs(expected) <= 1), 'Invalid reload controller actions')
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), 'Reload deterministic action mismatch')
    return dict(max_abs_error=float(np.max(abs(actual-expected))), atol=atol, rtol=rtol,
                observations=len(observations), action_shape=list(actual.shape))


def runtime_provenance():
    modules = {}
    for name in ('x2_recovery.evaluate', 'x2_recovery.env', 'x2_recovery.reset',
                 'x2_recovery.success', 'x2_recovery.model', 'x2_recovery.train',
                 'mujoco', 'gymnasium', 'stable_baselines3', 'torch'):
        modules[name] = str(Path(importlib.import_module(name).__file__).resolve())
    return dict(python=sys.executable, prefix=sys.prefix, python_version=sys.version,
                os=platform.platform(), architecture=platform.machine(), modules=modules,
                dependencies={n: importlib.metadata.version(n) for n in
                              ('mujoco', 'gymnasium', 'stable_baselines3', 'torch', 'numpy')},
                device='cpu', torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                cpu_count=__import__('os').cpu_count())


def prepare(training_run, *, expected_checkpoint_sha256=None, reload_validation=None):
    """Compatibility and saved-observation wiring check. Never resets or steps."""
    directory, config, saved, training, reload, hashes = load_inputs(training_run,
        expected_checkpoint_sha256=expected_checkpoint_sha256, reload_validation=reload_validation)
    configure_threads(config)
    env = X2RecoveryEnv(config=config.env, render_mode=None, capture_substeps=True)
    try:
        controlled = saved.get('schema') == CONTROLLER_SCHEMA
        if controlled:
            env = ControlledRecoveryEnv(env, ControlConfig.from_dict(saved['controller']))
        model = PPO.load(directory / 'policy_final.zip', device='cpu')
        model.policy.set_training_mode(False)
        original = policy_copy(model)
        identity = validate_environment(env, saved, model)
        tolerances = reload['action_consistency']
        with np.load(directory / 'reload_probe.npz', allow_pickle=False) as probe, torch.inference_mode():
            compare = compare_controller_actions if controlled else compare_actions
            consistency = compare(model, probe['observations'], probe['actions'],
                                  atol=tolerances['atol'], rtol=tolerances['rtol'])
        verify_policy(model, original)
        return env, model, original, dict(directory=directory, config=config, saved=saved,
                training=training, hashes=hashes, identity=identity, consistency=consistency,
                input_sources=input_sources(directory, controlled, reload_validation),
                model_asset_directory=str(physical_env(env).loaded.asset_repo.resolve()))
    except BaseException:
        env.close()
        raise


def freeze(output, seeds, prepared, model, original, command, wall_seconds):
    output = Path(output).resolve()
    require(not output.exists(), 'Output directory already exists: ' + str(output))
    require(type(wall_seconds) in (float, int) and math.isfinite(wall_seconds) and wall_seconds > 0,
            'Invalid episode wall budget')
    validate_seeds(seeds)
    output.mkdir(parents=True, exist_ok=False)
    source_dir = Path(__file__).resolve().parent
    snapshot = output / 'source' / 'x2_recovery'
    snapshot.mkdir(parents=True)
    source_hashes = {}
    for name in SOURCE_FILES:
        shutil.copyfile(source_dir / name, snapshot / name)
        source_hashes[name] = sha256(source_dir / name)
        require(sha256(snapshot / name) == source_hashes[name], 'Source copy mismatch')
    input_dir = output / 'inputs' / 'training-run'
    input_dir.mkdir(parents=True)
    for name, digest in prepared['hashes'].items():
        source = prepared.get('input_sources', {}).get(name, prepared['directory'] / name)
        shutil.copyfile(source, input_dir / name)
        require(sha256(input_dir / name) == digest, 'Input copy mismatch: ' + name)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source_dir), *args], text=True).strip()
    root = git('rev-parse', '--show-toplevel')
    patch = subprocess.check_output(['git', '-C', root, 'diff', 'HEAD', '--',
                                    'README.md', 'src/x2_recovery', '.gitignore'], text=True)
    (output / 'source' / 'working-tree.patch').write_text(patch)
    train = prepared['training']
    controlled = prepared['saved'].get('schema') == CONTROLLER_SCHEMA
    manifest = dict(schema_version=1, run_id=output.name, created_utc=utc_now(), command=command,
        cwd=str(Path.cwd()), seeds=seeds, controller=controller_label(prepared['saved']), deterministic=True,
        training_source=dict(run_id=train['run_id'], run_type=train['run_type'] if controlled else 'small PPO smoke experiment',
            sampled_transitions=train['training']['sampled_transitions'],
            trained_rollout_transitions=train['training']['transitions_in_completed_training_rollouts'],
            optimizer_steps=train['training']['optimizer_steps']),
        original_training_directory=str(prepared['directory']), portable_training_directory='inputs/training-run',
        input_sha256=prepared['hashes'], runtime=runtime_provenance(),
        git=dict(root=root, head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'), status=git('status', '--short')),
        source_snapshot='source/x2_recovery', source_directory=str(source_dir), source_sha256=source_hashes,
        model_source=train['provenance']['model_source'], identity=prepared['identity'],
        model_asset_directory=prepared['model_asset_directory'],
        normalization='fixed_env_scaling', capture_substeps=True, render_mode=None,
        initial_policy_state_sha256=policy_hash(original), reload_probe=prepared['consistency'],
        episode_wall_seconds=wall_seconds,
        watchdog='Cooperative, includes reset and recording; checked after bounded calls; interruption is not time_limit.',
        metrics=dict(sim_duration_s='Environment elapsed_sim_s; cross-checked with actual physics steps * dt; excludes settling.',
            max_pelvis_height_m='Pelvis body origin above floor: reset handoff and every executed physics substep.',
            max_stable_hold_s='Maximum existing SuccessTracker stable_duration_s, never a sum.',
            success='Environment is_success and existing tracker recovery_success at terminal transition.',
            substeps='Post-integration measurement and tracker at physics_dt. Contact details omitted; support loads retained.',
            control='Last substep q/dq/tau_raw are pre-integration at t-dt; actuator torque and post_step state are at t.',
            comparison_time_atol_s=TIME_ATOL, initial_state_atol=STATE_ATOL, initial_state_rtol=0.,
            state_hash='qpos float64, qvel float64, observation float32 only, excludes seeds, clocks and paths.',
            trajectory_hash='All recorded physical transitions excluding episode, seed and reset_seed; not unrecorded simulator internals.'),
        exception_rule='Stop batch, preserve partial attempt; no replacement episodes. New implementation requires a new full batch.',
        evaluation_scope='Five registered deterministic attempts, never training or ROS integration validation.')
    resolved = prepared['identity']['resolved_env']
    manifest['action_mapping'] = [dict(joint_name=row['joint_name'], order=i,
        **{key: resolved[key][i] for key in ('q_ref_rad', 'q_min_rad', 'q_max_rad',
            'scale_positive_rad', 'scale_negative_rad', 'kp_Nm_rad', 'kd_Nm_s_rad', 'effort_limits_Nm')})
        for i, row in enumerate(resolved['mapping'])]
    manifest['observation_layout'] = [
        [0, 31, 'joint position relative to range midpoint / half-range, actuator order'],
        [31, 62, 'joint velocity / saved joint_velocity_scale_rad_s'],
        [62, 65, 'gravity unit vector in pelvis frame'],
        [65, 68, 'base linear velocity in pelvis frame / saved base_linear_scale_m_s'],
        [68, 71, 'base angular velocity / saved base_angular_scale_rad_s'],
        [71, 75, 'pelvis height / calibrated h_ref, left/right/other weight fractions'],
        [75, 106, 'previous applied action, actuator order'],
        [106, 108, 'tracker stable duration / hold_s, tracker reference valid'],
        [108, 117, 'pelvis/left/right drift in pelvis frame / calibrated drift limits']]
    if controlled:
        controller = prepared['saved']['environment']
        manifest['execution_controller'] = controller
        manifest['training_source'].update(
            cumulative_model_timesteps=train['training']['cumulative_model_timesteps'],
            inherited_model_timesteps=train['training']['inherited_model_timesteps'],
            cumulative_adam_steps=train['training']['adam_state']['step_max'])
        manifest['reload_validation_input'] = 'inputs/training-run/reload_validation.json'
        manifest['portable_evaluation_arguments'] = dict(
            expected_checkpoint_sha256=prepared['hashes']['policy_final.zip'],
            reload_validation='inputs/training-run/reload_validation.json')
        manifest['observation_layout'][6][2] = 'previous native target-mapping action, not the residual policy output'
        manifest['observation_layout'] += [
            [117, 148, 'adopted joint target relative to range midpoint / half-range, actuator order'],
            [148, 149, 'elapsed recovery time / saved episode timeout']]
        residual = controller.get('residual_joint_full_scale_rad', controller.get('residual_scale_rad'))
        if residual is None:
            residual = (np.asarray(controller['measured_delta_rad'])
                        * controller['controller']['residual_multiplier']).tolist()
        for index, mapping in enumerate(manifest['action_mapping']):
            mapping['target_rate_rad_s'] = controller['target_rate_rad_s'][index]
            if controller['controller']['mode'] == 'reference_residual':
                mapping['policy_residual_full_scale_rad'] = residual[index]
                if 'policy_to_joint_matrix' in controller:
                    mapping['policy_action_coefficients'] = dict(zip(controller['policy_action_names'],
                                                                     controller['policy_to_joint_matrix'][index]))
    write_json(output / 'manifest.json', manifest)
    require(read_json(output / 'manifest.json') == json_value(manifest), 'Manifest readback mismatch')
    (output / 'manifest.json').chmod(0o444)
    return manifest


def initial_state_hash(record):
    h = hashlib.sha256()
    for key, dtype in (('qpos', '<f8'), ('qvel', '<f8'), ('observation', '<f4')):
        h.update(np.asarray(record[key], dtype=dtype).tobytes())
    return h.hexdigest()


def reset_record(env, observation, info, episode, seed):
    record = dict(kind='reset', episode=episode, seed=seed, reset_seed=info['reset_seed'],
        observation=observation, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
        absolute_time_s=float(env.data.time), handoff_time_s=env.episode_start_time,
        reset_evidence=env.reset_evidence, recovery_origin=env.tracker.origin,
        measurement=env._measurement, tracker=env.tracker.result, info=info)
    record = json_value(record)
    validate_reset(record)
    record['physical_state_sha256'] = initial_state_hash(record)
    return record


def validate_reset(record):
    require(record['reset_evidence']['status'] == 'PASS', 'Supine reset did not pass')
    origin = record['recovery_origin']
    require(origin is not None and record['tracker']['invalid_reason'] is None, 'Unverified recovery origin')
    for key in ('qpos', 'qvel'):
        require(np.array_equal(origin[key], record[key]), 'Recovery origin state mismatch: ' + key)
        require(np.array_equal(record['reset_evidence']['final_' + key], record[key]), 'Reset evidence mismatch: ' + key)
    require(origin['time_s'] == record['absolute_time_s'] == record['handoff_time_s'], 'Handoff time mismatch')
    require(record['measurement']['time_s'] == record['handoff_time_s']
            == record['reset_evidence']['episode_start_time'], 'Reset measurement clock mismatch')
    require(record['info']['reset_seed'] == record['reset_seed'], 'Reset seed mismatch')
    require(record['reset_evidence']['initial']['seed'] == record['reset_seed'], 'Reset provenance seed mismatch')
    require(abs(record['info']['elapsed_sim_s']) <= TIME_ATOL, 'Reset elapsed time is not zero')
    require(record['info']['physics_steps_executed'] == 0 and not record['info']['is_success'], 'Invalid reset boundary')
    require(record['info']['termination_reason'] is None and record['info']['truncation_reason'] is None,
            'Reset retained terminal reason')
    require(record['tracker']['stable_duration_s'] == 0 and not record['tracker']['recovery_success'], 'Reset retained success')
    encoded(record)  # Reject nonfinite data before the first policy action.


def transition_record(env, observation, action, reward, terminated, truncated, info, episode, seed, step):
    require(env.last_substeps, 'Missing physical substep evidence')
    last = env.last_substeps[-1]
    q, dq = env.loaded.read_state(env.data)
    return json_value(dict(kind='transition', episode=episode, seed=seed, control_step=step,
        observation=observation, action=action, reward=reward, terminated=bool(terminated),
        truncated=bool(truncated), info=info,
        substeps=[dict(measurement={k: v for k, v in s['measurement'].items() if k != 'support'},
                       tracker=s['success']) for s in env.last_substeps],
        last_substep_control=dict(pre_time_s=last['measurement']['time_s'] - env.physics_dt,
            post_time_s=last['measurement']['time_s'], target_rad=last['target_rad'],
            pre_q_rad=last['q_rad'], pre_dq_rad_s=last['dq_rad_s'], pre_tau_raw_Nm=last['tau_raw_Nm'],
            post_applied_torque_Nm=last['applied_torque_Nm']),
        post_step=dict(time_s=float(env.data.time), q_rad=q, dq_rad_s=dq,
                       ctrl_Nm=env.data.ctrl[env.context.ctrladr].copy())))


class EpisodeMetrics:
    """Aggregate existing measurements; make no additional success decisions."""
    def __init__(self, reset, dt, controller='ppo'):
        validate_reset(reset)
        self.reset = reset
        self.dt = dt
        self.controller = controller
        self.steps = self.physics_steps = 0
        self.time = reset['handoff_time_s']
        self.max_height = reset['measurement']['pelvis_height_m']
        self.max_hold = reset['tracker']['stable_duration_s']
        self.ended = False
        self.raw_return = 0.
        self.reward_terms = Counter()
        self.failure_samples = Counter()
        self.saturated_steps = self.action_change_sum = 0.
        self.previous_action = None
        self.peak = dict(elapsed_sim_s=0., measurement=reset['measurement'], tracker=reset['tracker'])
        self.qualified_samples = 0
        self.hold_interruptions = 0
        self.previous_hold = self.max_hold

    def add(self, record):
        require(not self.ended, 'Transition after terminal')
        require(record['control_step'] == self.steps + 1, 'Control step order mismatch')
        info, samples = record['info'], record['substeps']
        require(info['reset_seed'] == self.reset['reset_seed'], 'Episode reset seed changed')
        require(len(samples) == info['physics_steps_executed'] > 0, 'Incorrect substep count')
        for sample in samples:
            v, tracker = sample['measurement'], sample['tracker']
            require(math.isclose(v['time_s'] - self.time, self.dt, abs_tol=TIME_ATOL, rel_tol=0), 'Physical timestamp gap')
            require(tracker['invalid_reason'] is None, 'Invalid tracker sampling')
            self.time = v['time_s']
            self.physics_steps += 1
            if v['pelvis_height_m'] > self.max_height:
                self.max_height = v['pelvis_height_m']
                self.peak = dict(elapsed_sim_s=self.physics_steps * self.dt, measurement=v, tracker=tracker)
            self.max_hold = max(self.max_hold, tracker['stable_duration_s'])
            self.failure_samples.update(tracker['failures'])
            self.qualified_samples += int(tracker['instant_standing_ok'] and not tracker['failures'])
            self.hold_interruptions += int(self.previous_hold > 0 and tracker['stable_duration_s'] == 0)
            self.previous_hold = tracker['stable_duration_s']
        require(math.isclose(info['elapsed_sim_s'], self.physics_steps * self.dt, abs_tol=TIME_ATOL, rel_tol=0),
                'Episode duration disagrees with physics steps')
        require(math.isclose(self.time - self.reset['handoff_time_s'], info['elapsed_sim_s'], abs_tol=TIME_ATOL, rel_tol=0),
                'Episode clock mismatch')
        self.steps += 1
        require(info['stable_duration_s'] == samples[-1]['tracker']['stable_duration_s'], 'Hold measurement mismatch')
        require(all(info['state'][key] == samples[-1]['measurement'][key] for key in info['state']),
                'Control boundary measurement mismatch')
        self.raw_return += record['reward']
        self.reward_terms.update(info['reward_terms'])
        require(math.isclose(sum(info['reward_terms'].values()), record['reward'], abs_tol=1e-9, rel_tol=1e-9),
                'Reward terms disagree')
        self.saturated_steps += info['torque_saturation_fraction'] * len(samples)
        action = np.asarray(record['action'], dtype=float)
        require(action.ndim == 1 and action.size in (11, 17, 31) and np.isfinite(action).all(),
                'Invalid recorded policy action')
        if self.previous_action is None:
            self.previous_action = np.zeros_like(action)
        require(action.shape == self.previous_action.shape, 'Policy action shape changed within episode')
        self.action_change_sum += float(np.mean((action - self.previous_action)**2))
        self.previous_action = action
        self.ended = record['terminated'] or record['truncated']
        if not self.ended:
            require(not info['is_success'] and info['termination_reason'] is None
                    and info['truncation_reason'] is None, 'Nonterminal reason mismatch')
            return None
        require(not (record['terminated'] and record['truncated']), 'Conflicting episode flags')
        require(not record['truncated'] or samples[-1]['tracker']['timed_out'], 'Timeout tracker mismatch')
        reason = info['termination_reason'] or info['truncation_reason']
        success = info['is_success']
        require(success == samples[-1]['tracker']['recovery_success'] == (reason == 'success'), 'Success evidence mismatch')
        require((record['truncated'] and reason == 'time_limit' and info['termination_reason'] is None)
                or (record['terminated'] and info['truncation_reason'] is None
                    and (reason == 'success' or str(reason).startswith('safety_abort:'))), 'Terminal semantics mismatch')
        return dict(episode=self.reset['episode'], seed=self.reset['seed'], reset_seed=self.reset['reset_seed'],
            controller=self.controller, success=int(success), sim_duration_s=info['elapsed_sim_s'],
            max_pelvis_height_m=self.max_height, max_stable_hold_s=self.max_hold,
            terminated=int(record['terminated']), truncated=int(record['truncated']), termination_reason=reason)

    def diagnostics(self):
        return dict(control_steps=self.steps, physics_steps=self.physics_steps, raw_return=self.raw_return,
            reward_terms=dict(self.reward_terms), peak_height_sample=self.peak,
            failed_predicate_substeps=dict(self.failure_samples), qualified_substeps=self.qualified_samples,
            hold_interruptions=self.hold_interruptions,
            torque_saturation_physics_weighted=self.saturated_steps / self.physics_steps,
            mean_squared_action_change_per_transition=self.action_change_sum / self.steps)


def append_record(stream, record):
    stream.write(encoded(record) + '\n')
    stream.flush()


def verify_saved(output, require_complete=True):
    """Recompute CSV metrics by streaming the saved trajectory, not live memory."""
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    rows, resets, diagnostics, digests = [], [], [], []
    active = None
    pending_row = None
    with (output / 'trajectory.jsonl').open() as stream:
        for line in stream:
            record = strict_json(line)
            kind = record['kind']
            if kind == 'reset':
                require(active is None and len(resets) < 5, 'Extra or overlapping reset')
                index = len(resets)
                require(record['episode'] == index + 1 and record['seed'] == manifest['seeds'][index], 'Reset order mismatch')
                require(initial_state_hash(record) == record['physical_state_sha256'], 'Physical state hash mismatch')
                resets.append(record)
                active = EpisodeMetrics(record, manifest['identity']['resolved_env']['physics_dt'],
                                        manifest.get('controller', 'ppo'))
                digest = hashlib.sha256()
            elif kind == 'transition':
                require(active is not None and record['episode'] == active.reset['episode']
                        and record['seed'] == active.reset['seed'], 'Transition outside episode')
                pending_row = active.add(record)
                physical = {k: v for k, v in record.items() if k not in ('episode', 'seed')}
                physical['info'] = {k: v for k, v in record['info'].items() if k != 'reset_seed'}
                digest.update(encoded(physical).encode())
            elif kind == 'terminal':
                require(active is not None and pending_row is not None and record['row'] == pending_row,
                        'Terminal record mismatch')
                require(record['episode'] == active.reset['episode'] and record['seed'] == active.reset['seed'],
                        'Terminal identity mismatch')
                require(record['policy_state_sha256'] == manifest['initial_policy_state_sha256'], 'Policy changed')
                require(record['diagnostics'] == active.diagnostics(), 'Diagnostic aggregation mismatch')
                rows.append(pending_row)
                diagnostics.append(active.diagnostics())
                digests.append(digest.hexdigest())
                active, pending_row = None, None
            elif kind == 'error':
                require(not require_complete, 'Error in completed trajectory')
            else:
                raise ValueError('Unknown trajectory record: ' + kind)
    with (output / 'episodes.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(COLUMNS), 'CSV header mismatch')
        csv_rows = list(reader)
    require(len(csv_rows) == len(rows), 'CSV/trajectory row count mismatch')
    for actual, expected in zip(csv_rows, rows):
        for key, value in expected.items():
            require(actual[key] == str(value), 'CSV/trajectory mismatch: ' + key)
    if require_complete:
        require(len(rows) == 5 and len(resets) == 5 and active is None, 'Incomplete five-episode batch')
    comparisons = []
    for record in resets:
        comparisons.append(dict(episode=record['episode'], physical_state_sha256=record['physical_state_sha256'],
            exact_equal_to_first=all(np.array_equal(record[k], resets[0][k]) for k in ('qpos', 'qvel', 'observation')),
            max_abs_delta={k: float(np.max(np.abs(np.asarray(record[k]) - resets[0][k])))
                           for k in ('qpos', 'qvel', 'observation')}))
    return dict(valid_completed=len(rows), successes_observed=sum(r['success'] for r in rows),
        episodes=rows, diagnostics=diagnostics, initial_state_comparison=comparisons,
        all_initial_states_equal_within_tolerance=bool(resets) and all(
            max(r['max_abs_delta'].values()) <= STATE_ATOL for r in comparisons),
        physical_transition_sha256=digests, all_recorded_physical_transitions_identical=len(set(digests)) == 1,
        verified_from_disk=True)


def execute_batch(output, env, model, boundary_check):
    """Run the already-frozen protocol. Fake envs may test this bookkeeping only."""
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    native = physical_env(env)
    start = time.perf_counter()
    summary = dict(status='INCOMPLETE', planned=5, attempted=0, valid_completed=0,
                   successes_observed=0, started_utc=utc_now())
    step = 0
    record = None
    try:
        with (output / 'trajectory.jsonl').open('x') as trajectory, (output / 'episodes.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            stream.flush()
            try:
                for episode, seed in enumerate(manifest['seeds'], 1):
                    boundary_check()
                    summary['attempted'] += 1
                    step = 0
                    episode_start = time.perf_counter()
                    observation, info = env.reset(seed=seed)
                    record = reset_record(native, observation, info, episode, seed)
                    append_record(trajectory, record)
                    metrics = EpisodeMetrics(record, native.physics_dt, manifest.get('controller', 'ppo'))
                    while True:
                        require(time.perf_counter() - episode_start < manifest['episode_wall_seconds'],
                                'Evaluation wall watchdog interrupted episode')
                        with torch.inference_mode():
                            action, _ = model.predict(observation, deterministic=True)
                        # predict() already applies SB3's action semantics; pass it unchanged.
                        observation, reward, terminated, truncated, info = env.step(action)
                        step += 1
                        record = transition_record(native, observation, action, reward, terminated, truncated,
                                                   info, episode, seed, step)
                        append_record(trajectory, record)
                        row = metrics.add(record)
                        require(time.perf_counter() - episode_start < manifest['episode_wall_seconds'],
                                'Evaluation wall watchdog interrupted episode')
                        if row is not None:
                            digest = boundary_check()
                            append_record(trajectory, dict(kind='terminal', episode=episode, seed=seed,
                                row=row, diagnostics=metrics.diagnostics(), policy_state_sha256=digest,
                                wall_seconds=time.perf_counter() - episode_start))
                            writer.writerow(row)
                            stream.flush()
                            summary['valid_completed'] += 1
                            summary['successes_observed'] += row['success']
                            print(encoded(row), flush=True)
                            break
                boundary_check()
            except BaseException as exc:
                summary['error'] = dict(type=type(exc).__name__, message=str(exc),
                    attempted_episode=summary['attempted'], control_step=step, traceback=traceback.format_exc(),
                    environment_last_error=_diagnostic(getattr(native, 'last_error', None)))
                summary['error']['last_record'] = _diagnostic(record)
                try:
                    append_record(trajectory, dict(kind='error', **summary['error']))
                except Exception as recording_error:
                    summary['error']['recording_error'] = repr(recording_error)
                raise
        verification = verify_saved(output)
        require(verification['valid_completed'] == summary['valid_completed']
                and verification['successes_observed'] == summary['successes_observed'], 'Summary mismatch')
        summary.update(verification)
        summary.update(status='COMPLETE', result=f"{summary['successes_observed']}/5",
                       final_policy_state_sha256=boundary_check())
        return summary
    except BaseException as exc:
        summary['status'] = 'ERROR'
        if 'error' not in summary:
            summary['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        env.close()
        summary.update(ended_utc=utc_now(), wall_seconds=time.perf_counter() - start, environment_closed=True)
        summary['artifact_sha256'] = {p.name: sha256(p) for p in
            (output / 'manifest.json', output / 'episodes.csv', output / 'trajectory.jsonl') if p.is_file()}
        write_json(output / 'summary.json', summary)


def audit_saved(output):
    """Final read-only audit, including the summary written after resources close."""
    output = Path(output)
    manifest, summary = read_json(output / 'manifest.json'), read_json(output / 'summary.json')
    require(summary['status'] == 'COMPLETE' and summary['planned'] == summary['attempted'] == 5,
            'Batch is not complete')
    verification = verify_saved(output)
    for key, value in verification.items():
        require(summary[key] == value, 'Saved summary mismatch: ' + key)
    require(summary['result'] == str(verification['successes_observed']) + '/5', 'Result mismatch')
    require(summary['environment_closed'] is True, 'Environment cleanup missing')
    require(summary['final_policy_state_sha256'] == manifest['initial_policy_state_sha256'],
            'Policy identity changed')
    for name, digest in summary['artifact_sha256'].items():
        require(sha256(output / name) == digest, 'Saved artifact changed: ' + name)
    for name, digest in manifest['input_sha256'].items():
        require(sha256(output / manifest['portable_training_directory'] / name) == digest,
                'Copied training input changed: ' + name)
    for name, digest in manifest['source_sha256'].items():
        require(sha256(output / manifest['source_snapshot'] / name) == digest, 'Source snapshot changed: ' + name)
    return dict(status='PASS', valid_completed=verification['valid_completed'],
                successes_observed=verification['successes_observed'], summary_sha256=sha256(output / 'summary.json'))


def run(training_run, seeds, output, command=None, episode_wall_seconds=180., *,
        expected_checkpoint_sha256=None, reload_validation=None):
    seeds = validate_seeds(seeds)
    output = Path(output).resolve()
    require(not output.exists(), 'Output directory already exists: ' + str(output))
    require(type(episode_wall_seconds) in (int, float) and math.isfinite(episode_wall_seconds)
            and episode_wall_seconds > 0, 'Invalid episode wall budget')
    env, model, original, prepared = prepare(training_run,
        expected_checkpoint_sha256=expected_checkpoint_sha256, reload_validation=reload_validation)
    try:
        manifest = freeze(output, seeds, prepared, model, original,
                          command or [sys.executable, '-m', 'x2_recovery.evaluate', *sys.argv[1:]], episode_wall_seconds)
        manifest_hash = sha256(output / 'manifest.json')
        def boundary_check():
            require(sha256(output / 'manifest.json') == manifest_hash, 'Frozen manifest changed')
            for name, digest in manifest['input_sha256'].items():
                source = prepared.get('input_sources', {}).get(name, prepared['directory'] / name)
                require(sha256(source) == digest
                        and sha256(output / 'inputs' / 'training-run' / name) == digest, 'Input changed: ' + name)
            for name, digest in manifest['source_sha256'].items():
                require(sha256(Path(__file__).resolve().parent / name) == digest
                        and sha256(output / manifest['source_snapshot'] / name) == digest, 'Source changed: ' + name)
            validate_environment(env, prepared['saved'], model)
            return verify_policy(model, original)
        summary = execute_batch(output, env, model, boundary_check)
        try:
            audit_saved(output)
        except Exception as exc:
            summary.update(status='ERROR', disk_audit_error=repr(exc))
            summary.pop('result', None)
            write_json(output / 'summary.json', summary)
            raise
        return summary
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True, type=Path)
    parser.add_argument('--seeds', required=True, nargs=5, type=int)
    parser.add_argument('--deterministic', required=True, action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episode-wall-seconds', type=float, default=180.,
                        help='Cooperative wall watchdog per attempt, including reset (default: 180).')
    parser.add_argument('--expected-checkpoint-sha256',
                        help='Required explicit checkpoint identity for controller PPO experiments; the legacy smoke identity stays fixed.')
    parser.add_argument('--reload-validation', type=Path,
                        help='Completed control-reload summary; defaults to portable reload_validation.json or development-221020/summary.json.')
    args = parser.parse_args(argv)
    summary = run(args.training_run, args.seeds, args.output, episode_wall_seconds=args.episode_wall_seconds,
                  expected_checkpoint_sha256=args.expected_checkpoint_sha256, reload_validation=args.reload_validation)
    print(encoded(dict(status=summary['status'], result=summary['result'], output=str(args.output.resolve()))))


if __name__ == '__main__':
    main()
