"""Synthetic fixtures test bookkeeping only; real X2 evidence lives in run directories."""
from contextlib import ExitStack
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from x2_recovery import train as tr


class SyntheticEnv(gym.Env):
    """No physics; intentionally separate from the real-X2 acceptance runs."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (117,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (31,), dtype=np.float32)
    physics_dt = .001

    def __init__(self, *args, terminate=False, fail=False, **kwargs):
        self.terminate, self.fail = terminate, fail
        self.resets = self.steps = 0
        self._closed = False
        self.loaded = SimpleNamespace(asset_repo=Path('/synthetic'))

    def resolved_config(self):
        return {'synthetic_fixture': True}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.steps = 0
        return np.full(117, self.resets / 10, dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Synthetic fixture received an unclipped or malformed action')
        if self.fail:
            raise RuntimeError('synthetic step failure')
        self.steps += 1
        done = self.steps == 2
        reward = 1. + .01 * float(action[0])
        terms = {key: 0. for key in tr.REWARD_KEYS}
        terms['height'] = reward
        info = dict(physics_steps_executed=20, reset_seed=self.resets, elapsed_sim_s=.02 * self.steps,
                    termination_reason='safety_abort:synthetic' if done and self.terminate else None,
                    truncation_reason='time_limit' if done and not self.terminate else None,
                    is_success=False, reward_terms=terms, reward_terms_raw=terms.copy(),
                    torque_saturation_fraction=.25, stable_duration_s=0.,
                    state=dict(pelvis_height_m=.1, tilt_deg=80.),
                    control=dict(target_boundary_fraction=0., joint_rad_s=1., joint_limit_rad=0.,
                                 floor_penetration_m=0., self_penetration_m=0.,
                                 raw_torque_abs_max_Nm=2., applied_torque_abs_max_Nm=1.))
        return np.full(117, self.steps, dtype=np.float32), reward, done and self.terminate, done and not self.terminate, info

    def close(self):
        self._closed = True


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tr.configure_threads(tr.TrainConfig())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = tr.TrainConfig(total_timesteps=16, n_steps=8, batch_size=4, n_epochs=2)

    def fixture(self, *, terminate=False, config=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        env = SyntheticEnv(terminate=terminate)
        monitor = tr.TimedMonitor(env, self.root / 'monitor.csv')
        stack.callback(monitor.close)
        vec = VecCheckNan(DummyVecEnv([lambda: monitor]), raise_exception=True)
        stack.callback(vec.close)
        config = config or self.config
        model = tr.make_model(config, vec, tr.ObservedPPO)
        logger = configure(str(self.root / 'sb3'), ['csv'])
        stack.callback(logger.close)
        model.set_logger(logger)
        episodes = tr.EpisodeLog(self.root, stack, .001)
        evidence = tr.TrainingEvidence(config, self.root, stack, episodes)
        evidence.install_hooks(model, stack)
        model.evidence = evidence
        evidence.learn_start = time.perf_counter()
        return env, vec, model, evidence, episodes

    def test_gamma_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'gamma'):
            replace(self.config, gamma=.99)

    def test_invalid_types_shapes_and_budgets_rejected(self):
        for kw in (dict(batch_size=3), dict(n_steps=True), dict(seed=-1), dict(seed=1.5),
                   dict(max_wall_seconds=0), dict(max_wall_seconds=float('nan')), dict(n_epochs='5'),
                   dict(total_timesteps=9), dict(net_arch=[128, 128]), dict(net_arch=(128,)),
                   dict(normalize_advantage=1), dict(learning_rate=True)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                replace(self.config, **kw)
        with self.assertRaises(ValueError):
            tr.TrainConfig.from_dict({'unknown_setting': 4})
        with self.assertRaises(ValueError):
            tr.TrainConfig.from_dict({'env': None})

    def test_config_roundtrip_restores_immutable_nested_tuples(self):
        serialized = json.loads(json.dumps(asdict(self.config)))
        restored = tr.TrainConfig.from_dict(serialized)
        self.assertEqual(self.config, restored)
        self.assertIsInstance(restored.env.pd_gains[0], tuple)
        self.assertIsInstance(restored.env.reference_angles[0], tuple)

    def test_existing_directory_and_invalid_parent_are_not_overwritten(self):
        marker = self.root / 'keep'
        marker.write_text('user content')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            tr.run_experiment(self.config, self.root, 'smoke')
        self.assertEqual(marker.read_text(), 'user content')
        with self.assertRaises((ValueError, OSError)):
            tr.run_experiment(self.config, marker / 'child', 'smoke')

    def test_initial_or_critic_only_or_log_std_only_is_not_training_evidence(self):
        _, _, model, _, _ = self.fixture()
        before = tr.parameter_copy(model.policy)
        with self.assertRaises(ValueError):
            tr.validate_updates(tr.parameter_changes(before, model.policy), 0, 0)
        with torch.no_grad():
            model.policy.log_std.add_(.1)
            model.policy.value_net.bias.add_(.1)
        with self.assertRaisesRegex(ValueError, 'actor_mean'):
            tr.validate_updates(tr.parameter_changes(before, model.policy), 1, 1)

    def test_terminated_truncated_terminal_observation_and_auto_reset(self):
        for terminated in (False, True):
            with tempfile.TemporaryDirectory() as path:
                env = SyntheticEnv(terminate=terminated)
                vec = DummyVecEnv([lambda: tr.TimedMonitor(env, Path(path) / 'monitor.csv')])
                try:
                    vec.reset()
                    vec.step(np.zeros((1, 31), np.float32))
                    obs, _, done, info = vec.step(np.zeros((1, 31), np.float32))
                    self.assertTrue(done[0])
                    self.assertEqual(info[0]['TimeLimit.truncated'], not terminated)
                    np.testing.assert_array_equal(info[0]['terminal_observation'], np.full(117, 2., np.float32))
                    np.testing.assert_array_equal(obs[0], np.full(117, .2, np.float32))
                    self.assertEqual(env.resets, 2)
                finally:
                    vec.close()
                self.assertTrue(env._closed)

    def test_timeout_bootstrap_does_not_pollute_episode_returns(self):
        _, _, model, evidence, episodes = self.fixture()
        with torch.no_grad():
            for parameter in model.policy.mlp_extractor.value_net.parameters():
                parameter.zero_()
            model.policy.value_net.weight.zero_()
            model.policy.value_net.bias.fill_(2.)
        model.learn(8, callback=evidence, log_interval=None)
        with np.load(self.root / 'rollout_0001.npz') as rollout:
            self.assertGreater(float(rollout['rewards'][1, 0]), 2.9)
        for row in episodes.episodes:
            self.assertLess(row['return'], 2.03)
        with (self.root / 'monitor.csv').open() as stream:
            next(stream)
            monitor_rows = list(csv.DictReader(stream))
        np.testing.assert_allclose([row['return'] for row in episodes.episodes],
                                   [float(row['r']) for row in monitor_rows], atol=2e-6)

    def test_partial_rollout_and_episode_are_not_completed(self):
        config = replace(self.config, max_wall_seconds=1e-9)
        _, _, model, evidence, episodes = self.fixture(config=config)
        model.learn(config.total_timesteps, callback=evidence, log_interval=None)
        summary = evidence.summary(model)
        self.assertEqual(summary['sampled_transitions'], 1)
        self.assertEqual(summary['transitions_in_completed_training_rollouts'], 0)
        self.assertEqual(summary['partial_rollout_transitions'], 1)
        self.assertEqual(summary['optimizer_steps'], 0)
        self.assertEqual(episodes.episodes, [])
        self.assertEqual(episodes.partial('wall_budget')['transitions'], 1)

    def test_partial_after_complete_update_retains_only_trained_rollout_count(self):
        _, _, model, evidence, episodes = self.fixture()
        original = evidence._on_step
        def cutoff():
            if episodes.transitions == 8:
                evidence.learn_start -= self.config.max_wall_seconds
            return original()
        with patch.object(evidence, '_on_step', side_effect=cutoff):
            model.learn(16, callback=evidence, log_interval=None)
        summary = evidence.summary(model)
        self.assertEqual(summary['sampled_transitions'], 9)
        self.assertEqual(summary['transitions_in_completed_training_rollouts'], 8)
        self.assertEqual(summary['partial_rollout_transitions'], 1)
        self.assertEqual(summary['completed_optimization_rounds'], 1)
        self.assertEqual(len(episodes.episodes), 4)
        self.assertEqual(episodes.partial('wall_budget')['transitions'], 1)

    def test_raw_gaussian_actions_are_retained_in_buffer(self):
        config = replace(self.config, log_std_init=2.)
        _, _, model, evidence, _ = self.fixture(config=config)
        model.learn(8, callback=evidence, log_interval=None)
        with np.load(self.root / 'rollout_0001.npz') as data:
            self.assertGreater(float(np.abs(data['actions']).max()), 1.)
        records = [json.loads(line) for line in (self.root / 'transitions.jsonl').read_text().splitlines()]
        self.assertTrue(all(np.max(np.abs(row['action'])) <= 1 for row in records))

    def test_nonfinite_gradient_stops_before_optimizer_step(self):
        _, _, model, evidence, _ = self.fixture()
        hook = model.policy.action_net.weight.register_hook(lambda grad: grad * float('nan'))
        try:
            with self.assertRaisesRegex(ValueError, 'gradient'):
                model.learn(8, callback=evidence, log_interval=None)
        finally:
            hook.remove()
        self.assertEqual(evidence.optimizer_steps, 0)
        self.assertEqual(evidence.completed_updates, 0)

    def test_real_optimizer_counts_gradients_parameter_changes_and_last_log(self):
        _, _, model, evidence, _ = self.fixture()
        before = tr.parameter_copy(model.policy)
        model.learn(16, callback=evidence, log_interval=None)
        changes = tr.parameter_changes(before, model.policy)
        tr.validate_updates(changes, evidence.optimizer_steps, evidence.completed_updates)
        self.assertEqual(evidence.completed_updates, 2)
        self.assertGreater(evidence.optimizer_steps, 0)
        self.assertLessEqual(evidence.optimizer_steps, 8)
        self.assertEqual(evidence.optimizer_state(model.policy.optimizer)['step_max'], evidence.optimizer_steps)
        self.assertGreater(evidence.gradient_tensors, 0)
        with (self.root / 'progress.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(int(rows[-1]['sampled_transitions']), 16)
        self.assertTrue(np.isfinite(float(rows[-1]['total_loss'])))

    def test_plain_ppo_load_in_new_process_matches_synthetic_observations(self):
        _, _, model, evidence, episodes = self.fixture()
        model.learn(8, callback=evidence, log_interval=None)
        model.save(self.root / 'policy_final.zip')
        obs = np.stack(episodes.observations)
        actions, _ = model.predict(obs, deterministic=True)
        np.savez(self.root / 'probe.npz', observations=obs, actions=actions)
        code = """
import numpy as np, sys
from pathlib import Path
from stable_baselines3 import PPO
from x2_recovery.train import compare_actions
p=Path(sys.argv[1]); m=PPO.load(p/'policy_final.zip', device='cpu')
with np.load(p/'probe.npz', allow_pickle=False) as d:
    print(compare_actions(m,d['observations'],d['actions']))
"""
        child = subprocess.run([sys.executable, '-c', code, str(self.root)], capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertIn('max_abs_error', child.stdout)
        loaded = PPO.load(self.root / 'policy_final.zip', device='cpu')
        self.assertFalse(hasattr(loaded, 'evidence'))
        self.assertEqual(len(loaded.policy.optimizer._optimizer_step_post_hooks), 0)

    def test_sampling_gate_needs_twenty_complete_episodes(self):
        row = dict(length=1, simulated_seconds=.012, reason='safety_abort:joint_rad_s',
                   truncated=False, success=False, **{'return': -1.})
        self.assertFalse(tr.sampling_quality([row] * 19)['sampling_degenerate'])
        self.assertTrue(tr.sampling_quality([row] * 20)['sampling_degenerate'])
        three = {**row, 'length': 3}
        q = tr.sampling_quality([three] * 20)
        self.assertFalse(q['sampling_degenerate'])
        self.assertEqual(q['recent_safety_le5_fraction'], 1.)
        self.assertTrue(q['all_recent_safety_under_one_second'])

    def test_plot_from_saved_logs(self):
        _, _, model, evidence, _ = self.fixture()
        model.learn(8, callback=evidence, log_interval=None)
        tr.write_json(self.root / 'resolved_config.json', dict(training=asdict(self.config)))
        result = tr.plot_reward(self.root)
        self.assertEqual(result['complete_episodes'], 4)
        from PIL import Image
        with Image.open(self.root / 'training_reward.png') as image:
            self.assertEqual(image.size, (1350, 750))
            self.assertGreater(np.asarray(image).std(), 5)

    def test_exception_cleanup_and_no_initial_checkpoint(self):
        env = SyntheticEnv(fail=True)
        identity = dict(core_source_hashes=tr.source_hashes(), synthetic_fixture=True)
        with patch.object(tr, 'X2RecoveryEnv', return_value=env), \
                patch.object(tr, 'env_identity', return_value=identity), \
                patch.object(tr, 'provenance', return_value={'synthetic_fixture': True}):
            with self.assertRaisesRegex(RuntimeError, 'synthetic step failure'):
                tr.run_experiment(self.config, self.root / 'failed', 'smoke')
        self.assertTrue(env._closed)
        manifest = json.loads((self.root / 'failed/manifest.json').read_text())
        self.assertEqual(manifest['status'], 'ERROR')
        self.assertTrue(manifest['environment_closed'])
        self.assertFalse((self.root / 'failed/policy_final.zip').exists())
        # Opening for append proves our own streams have completed their lifecycle.
        with (self.root / 'failed/stdout.log').open('a') as stream:
            stream.write('synthetic cleanup verified\n')

    def test_budget_before_first_update_is_interrupted_and_closes_resources(self):
        env = SyntheticEnv()
        identity = dict(core_source_hashes=tr.source_hashes(), synthetic_fixture=True)
        config = replace(self.config, max_wall_seconds=1e-9)
        with patch.object(tr, 'X2RecoveryEnv', return_value=env), \
                patch.object(tr, 'env_identity', return_value=identity), \
                patch.object(tr, 'provenance', return_value={'synthetic_fixture': True}), \
                patch.object(tr, 'launch_reload') as reload:
            result = tr.run_experiment(config, self.root / 'budget', 'smoke')
        self.assertEqual(result['status'], 'INTERRUPTED')
        self.assertEqual(result['stop_reason'], 'wall_budget')
        self.assertEqual(result['training']['optimizer_steps'], 0)
        self.assertEqual(result['training']['partial_rollout_transitions'], 1)
        self.assertTrue(env._closed)
        self.assertIsNone(result['checkpoint'])
        self.assertFalse((self.root / 'budget/policy_final.zip').exists())
        reload.assert_not_called()

    def test_strict_json_does_not_hide_invalid_numbers(self):
        path = self.root / 'evidence.json'
        tr.write_json(path, {'valid': True})
        with self.assertRaises(ValueError):
            tr.write_json(path, {'loss': float('nan')})
        self.assertEqual(json.loads(path.read_text()), {'valid': True})

    def test_formal_plan_requires_verified_smoke_quality_and_matching_config(self):
        measured = replace(self.config, seed=1)
        manifest = dict(run_type='smoke', status='PASS', reload={'status': 'PASS'},
                        sampling_quality={'sampling_degenerate': False, 'recent_safety_le5_fraction': .1},
                        timing={'transitions_per_second': 5.})
        tr.write_json(self.root / 'manifest.json', manifest)
        tr.write_json(self.root / 'resolved_config.json', {'training': asdict(measured)})
        config = replace(self.config, max_wall_seconds=60.)
        plan = tr.formal_plan(self.root, config)
        self.assertEqual(plan['planned_transitions'], 240)
        with self.assertRaisesRegex(ValueError, 'differs'):
            tr.formal_plan(self.root, replace(config, log_std_init=-1.5))
        with self.assertRaisesRegex(ValueError, 'fresh seed'):
            tr.formal_plan(self.root, replace(config, seed=1))
        manifest['sampling_quality']['recent_safety_le5_fraction'] = .9
        tr.write_json(self.root / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Sampling quality'):
            tr.formal_plan(self.root, config)
        manifest['sampling_quality']['recent_safety_le5_fraction'] = .65
        manifest['sampling_quality']['all_recent_safety_under_one_second'] = True
        tr.write_json(self.root / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'subsecond safety'):
            tr.formal_plan(self.root, config)


class SyntheticControlledEnv(SyntheticEnv):
    """149D accounting fixture; this is not an X2 recovery simulation."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (149,), dtype=np.float32)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.env = SimpleNamespace(physics_dt=.001)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        return np.r_[observation, np.zeros(32, np.float32)], info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info['state'].update(left_weight=.1, right_weight=.2, other_weight=.7)
        info['controller'] = dict(target_tracking_max_rad=.2)
        info['reward_terms'] = {'new_controller_term': reward}
        return np.r_[observation, np.zeros(32, np.float32)], reward, terminated, truncated, info


class ControlBlockTests(unittest.TestCase):
    def test_provenance_snapshots_actual_imported_sources_at_logical_paths(self):
        repository = Path(tr.__file__).resolve().parents[3]
        actual_env = Path(sys.modules[tr.X2RecoveryEnv.__module__].__file__).resolve()
        command = subprocess.check_output
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root/'old-source'/'src'/'x2_recovery'/'x2_recovery'
            package.mkdir(parents=True)
            old_train = package/'train.py'
            old_train.write_text('# a previous execution source, not the current checkout\n')
            (package/'env.py').write_text('# not the imported environment module\n')
            (root/'old-source'/'requirements.txt').write_text('# saved package requirements\n')
            output = root/'new-run'
            output.mkdir()
            def checked_output(args, **kwargs):
                if args[:3] == ['git', '-C', str(package)]:
                    self.assertEqual(args[3:], ['rev-parse', '--show-toplevel'])
                    return str(repository)+'\n'
                return command(args, **kwargs)
            with patch.object(tr, '__file__', str(old_train)), patch.object(tr.subprocess, 'check_output', checked_output):
                saved = tr.provenance(output)
            train_path = 'src/x2_recovery/x2_recovery/train.py'
            env_path = 'src/x2_recovery/x2_recovery/env.py'
            self.assertEqual((output/'source'/train_path).read_bytes(), old_train.read_bytes())
            self.assertEqual(saved['source_paths'][train_path], str(old_train))
            self.assertEqual((output/'source'/env_path).read_bytes(), actual_env.read_bytes())
            self.assertEqual(saved['source_paths'][env_path], str(actual_env))
            self.assertEqual((output/'source'/'requirements.txt').read_text(), '# saved package requirements\n')
            for logical, digest in saved['source_hashes'].items():
                self.assertEqual(tr.sha256(output/'source'/logical), digest)

    def test_interrupted_update_publication_keeps_previous_checkpoint_pair(self):
        class SavedModel:
            def save(self, path, **kwargs):
                Path(path).write_bytes(b'checkpoint fixture')

        for failure in ('rng', 'pointer'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                saved = tr.save_valid_update(root, SavedModel(), {'completed_optimization_rounds': 1}, {})
                pointer = (root/'last_valid_update.json').read_bytes()
                checkpoint, rng = tr.valid_update_files(root, saved)
                originals = checkpoint.read_bytes(), rng.read_bytes()
                def fail_rng(path):
                    Path(path).write_bytes(b'interrupted RNG write')
                    raise OSError('injected RNG failure')
                with ExitStack() as stack:
                    if failure == 'rng':
                        stack.enter_context(patch.object(tr, '_save_rng', fail_rng))
                    else:
                        stack.enter_context(patch.object(tr, 'write_json', side_effect=OSError('injected pointer failure')))
                    with self.assertRaisesRegex(OSError, 'injected'):
                        tr.save_valid_update(root, SavedModel(), {'completed_optimization_rounds': 2}, {})
                self.assertEqual((root/'last_valid_update.json').read_bytes(), pointer)
                self.assertEqual(tr.valid_update_files(root, saved), (checkpoint, rng))
                self.assertEqual((checkpoint.read_bytes(), rng.read_bytes()), originals)
                self.assertTrue((root/'checkpoints'/'update-000002'/'policy.zip').exists())
                with self.assertRaises(FileExistsError):
                    tr.save_valid_update(root, SavedModel(), {'completed_optimization_rounds': 2}, {})

    def test_valid_update_checks_both_hashes_and_accepts_legacy_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, rng = root/'policy_last_update.zip', root/'rng_last_update.pt'
            checkpoint.write_bytes(b'old checkpoint')
            rng.write_bytes(b'old RNG')
            saved = dict(checkpoint_sha256=tr.sha256(checkpoint), rng_sha256=tr.sha256(rng))
            self.assertEqual(tr.valid_update_files(root, saved), (checkpoint, rng))
            rng.write_bytes(b'changed RNG')
            with self.assertRaisesRegex(ValueError, 'rng changed'):
                tr.valid_update_files(root, saved)

    def test_synthetic_step_error_recovers_published_generation_and_closes_env(self):
        class FailingFixture(SyntheticControlledEnv):
            instances = []
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.total_steps = 0
                self.instances.append(self)
            def step(self, action):
                self.total_steps += 1
                if self.total_steps == 9:
                    raise RuntimeError('injected later rollout failure')
                return super().step(action)

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)/'run'
            stack.enter_context(patch.object(tr, 'X2RecoveryEnv', SyntheticEnv))
            stack.enter_context(patch.object(tr, 'ControlledRecoveryEnv', FailingFixture))
            stack.enter_context(patch.object(tr, 'controller_identity', lambda env: {'fixture': 'synthetic149'}))
            stack.enter_context(patch.object(tr, 'provenance', lambda directory: {'fixture': 'synthetic'}))
            config = tr.TrainConfig(total_timesteps=16, n_steps=8, batch_size=4, n_epochs=2)
            with self.assertRaisesRegex(RuntimeError, 'later rollout failure'):
                tr.run_control_block(config, tr.ControlConfig(), root)
            saved = json.loads((root/'last_valid_update.json').read_text())
            checkpoint, rng = tr.valid_update_files(root, saved)
            manifest = json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'ERROR')
            self.assertEqual(manifest['training']['completed_optimization_rounds'], 1)
            self.assertEqual(manifest['observed_before_error']['sampled_transitions'], 8)
            self.assertEqual(manifest['checkpoint']['recovered_from'], saved['checkpoint_path'])
            self.assertEqual((root/'policy_final.zip').read_bytes(), checkpoint.read_bytes())
            self.assertEqual((root/'rng_state.pt').read_bytes(), rng.read_bytes())
            self.assertTrue(all(env._closed for env in FailingFixture.instances))

    def test_distribution_diagnostic_is_read_only_and_not_noise_weight_std(self):
        with ExitStack() as stack:
            env=DummyVecEnv([SyntheticControlledEnv]);stack.callback(env.close)
            model=PPO('MlpPolicy',env,n_steps=8,batch_size=4,use_sde=True,seed=123,
                      policy_kwargs={'net_arch':[32,32],'log_std_init':-1.5},device='cpu')
            observations=np.full((40,149),.5,dtype=np.float32)
            before={key:value.clone() for key,value in model.policy.state_dict().items()}
            rng=torch.get_rng_state().clone()
            matrices=model.policy.action_dist.exploration_matrices.clone()
            metrics=tr.action_distribution_metrics(model,observations)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(matrices,model.policy.action_dist.exploration_matrices))
            self.assertTrue(all(torch.equal(before[key],value) for key,value in model.policy.state_dict().items()))
            self.assertEqual(metrics['action_distribution_observations'],32)
            self.assertGreater(metrics['action_distribution_std_median'],metrics['noise_weight_std_mean'])
            self.assertLessEqual(metrics['action_distribution_std_min'],metrics['action_distribution_std_median'])
            self.assertLessEqual(metrics['action_distribution_std_median'],metrics['action_distribution_std_max'])

    def test_synthetic_first_success_retains_pre_update_weights_and_episode(self):
        class SuccessfulFixture(SyntheticControlledEnv):
            def step(self, action):
                observation, reward, terminated, truncated, info = super().step(action)
                if truncated:
                    terminated, truncated = True, False
                    info.update(is_success=True, termination_reason='success', truncation_reason=None)
                return observation, reward, terminated, truncated, info

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(tr, 'X2RecoveryEnv', SyntheticEnv))
            stack.enter_context(patch.object(tr, 'ControlledRecoveryEnv', SuccessfulFixture))
            stack.enter_context(patch.object(tr, 'controller_identity', lambda env: {'fixture': 'synthetic149'}))
            stack.enter_context(patch.object(tr, 'provenance', lambda directory: {'fixture': 'synthetic'}))
            config = tr.TrainConfig(total_timesteps=16, n_steps=8, batch_size=4, n_epochs=2)
            result = tr.run_control_block(config, tr.ControlConfig(), root/'run')
            saved = json.loads((root/'run'/'first_success.json').read_text())
            self.assertEqual(saved['optimizer_steps_in_block'], 0)
            self.assertEqual(saved['completed_optimization_rounds'], 0)
            self.assertEqual(saved['cumulative_model_timesteps'], 2)
            self.assertEqual(saved['episode_adam_versions'], [0])
            trajectory = json.loads((root/'run'/'first_success_trajectory.json').read_text())['transitions']
            self.assertEqual(len(trajectory), 2)
            self.assertEqual(saved['physics_steps'], 40)
            self.assertEqual(len(trajectory[0]['observation']), 149)
            self.assertTrue(trajectory[-1]['info']['is_success'])
            initial = PPO.load(root/'run'/'policy_first_success.zip', device='cpu')
            final = PPO.load(root/'run'/'policy_final.zip', device='cpu')
            self.assertGreater(tr.parameter_changes(tr.parameter_copy(initial.policy), final.policy)['actor_mean']['l2'], 0)
            self.assertEqual(result['training']['first_success'], saved)

    def test_synthetic_block_resume_keeps_optimizer_and_dynamic_rollout_shapes(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(tr, 'X2RecoveryEnv', SyntheticEnv))
            stack.enter_context(patch.object(tr, 'ControlledRecoveryEnv', SyntheticControlledEnv))
            stack.enter_context(patch.object(tr, 'controller_identity', lambda env: {'fixture': 'synthetic149'}))
            stack.enter_context(patch.object(tr, 'provenance', lambda directory: {'fixture': 'synthetic'}))
            config = tr.TrainConfig(total_timesteps=16, n_steps=8, batch_size=4, n_epochs=2)
            first = tr.run_control_block(config, tr.ControlConfig(), root/'first')
            self.assertEqual(first['status'], 'COMPLETE')
            self.assertEqual(first['training']['sampled_transitions'], 16)
            self.assertEqual(first['training']['partial_rollout_transitions'], 0)
            self.assertEqual(first['training']['rollout_checks'][0]['observations']['shape'], [8, 1, 149])
            self.assertEqual(first['training']['optimizer_steps'], first['training']['adam_state']['step_max'])
            with (root/'first'/'episode_diagnostics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(json.loads(row['reward_terms']).keys() == {'new_controller_term'} for row in rows))
            second = tr.run_control_block(config, tr.ControlConfig(), root/'second', resume_run=root/'first')
            self.assertEqual(second['training']['inherited_model_timesteps'], 16)
            self.assertEqual(second['training']['cumulative_model_timesteps'], 32)
            self.assertEqual(second['training']['adam_state']['inherited_steps'], first['training']['optimizer_steps'])
            self.assertGreater(second['training']['parameter_changes']['actor_mean']['l2'], 0)
            self.assertEqual(len(second['training']['updates']), 2)
            with (root/'second'/'progress.csv').open() as stream:
                progress = list(csv.DictReader(stream))
            self.assertEqual(progress[-1]['cumulative_model_timesteps'], '32')
            loaded = PPO.load(root/'second'/'policy_final.zip', device='cpu')
            self.assertNotIn('train', loaded.__dict__)
            correlated = tr.run_control_block(config, tr.ControlConfig(), root/'sde', use_sde=True)
            self.assertGreater(correlated['training']['optimizer_steps'], 0)
            correlated_model = PPO.load(root/'sde'/'policy_final.zip', device='cpu')
            self.assertTrue(correlated_model.use_sde)
            self.assertEqual(correlated_model.sde_sample_freq, 8)
            parent_hash = tr.sha256(root/'sde'/'policy_final.zip')
            with patch.object(tr, 'initialize_exploration_phase', wraps=tr.initialize_exploration_phase) as initialize:
                branch = tr.run_control_block(config, tr.ControlConfig(), root/'sde-low-std',
                    use_sde=True, resume_run=root/'sde', initial_exploration_std_factor=.25)
                self.assertEqual(initialize.call_count, 1)
                continued = tr.run_control_block(config, tr.ControlConfig(), root/'sde-low-std-continued',
                    use_sde=True, resume_run=root/'sde-low-std')
                self.assertEqual(initialize.call_count, 1)
            self.assertEqual(tr.sha256(root/'sde'/'policy_final.zip'), parent_hash)
            self.assertTrue(branch['initialization_phase']['applied'])
            self.assertFalse(continued['initialization_phase']['applied'])
            self.assertEqual(branch['training']['adam_state']['inherited_steps'], correlated['training']['optimizer_steps'])
            self.assertEqual(continued['training']['cumulative_model_timesteps'], 48)
            phase = json.loads((root/'sde-low-std'/'resolved_config.json').read_text())['initialization_phase']
            self.assertEqual(phase['factor'], .25)
            with self.assertRaisesRegex(ValueError, 'Resume configuration differs'):
                tr.run_control_block(replace(config, learning_rate=.001), tr.ControlConfig(), root/'bad', resume_run=root/'first')
            self.assertFalse((root/'bad').exists())
            changed = json.loads((root/'first'/'resolved_config.json').read_text())
            changed['unrelated_metadata'] = 'changed after saving'
            tr.write_json(root/'first'/'resolved_config.json', changed)
            with self.assertRaisesRegex(ValueError, 'configuration.*changed|Configuration.*changed'):
                tr.run_control_block(config, tr.ControlConfig(), root/'changed-config', resume_run=root/'first')
            self.assertFalse((root/'changed-config').exists())

    def test_exploration_phase_real_sb3_policy_without_env_or_optimizer_step(self):
        from stable_baselines3.common.policies import ActorCriticPolicy
        obs_space = gym.spaces.Box(-np.inf, np.inf, (149,), dtype=np.float32)
        action_space = gym.spaces.Box(-1., 1., (11,), dtype=np.float32)
        policy = ActorCriticPolicy(obs_space, action_space, lambda _: 1e-4,
            net_arch=dict(pi=[16, 16], vf=[16, 16]), use_sde=True, log_std_init=-1.5)
        # Synthetic, explicit Adam state exercises preservation without running
        # a training step. These values are not X2 optimization evidence.
        for parameter in policy.parameters():
            policy.optimizer.state[parameter] = dict(step=torch.tensor(7.),
                exp_avg=torch.full_like(parameter, .01), exp_avg_sq=torch.full_like(parameter, .02))
        model = SimpleNamespace(policy=policy, predict=policy.predict, observation_space=obs_space,
            action_space=action_space, num_timesteps=16384, _n_updates=40)
        observations = np.full((5, 149), .2, dtype=np.float32)
        actions, _ = model.predict(observations, deterministic=True)
        initial_log_std = policy.log_std.detach().clone()
        with patch.object(policy.optimizer, 'step', side_effect=AssertionError('No optimizer step allowed')):
            phase = tr.initialize_exploration_phase(model, .25, observations, actions)
        self.assertTrue(torch.equal(policy.log_std, initial_log_std + np.log(.25)))
        self.assertTrue(phase['optimizer_state_unchanged'] and phase['rng_state_unchanged'])
        self.assertEqual(phase['inherited_adam_steps'], 7)
        self.assertEqual(phase['action_consistency']['intervention_max_abs_error'], 0)
        self.assertNotEqual(phase['before']['sha256'], phase['after']['sha256'])
        np.testing.assert_allclose(phase['after']['exp_log_std_mean']/phase['before']['exp_log_std_mean'], .25, rtol=1e-6)
        before = policy.log_std.detach().clone()
        noop = tr.initialize_exploration_phase(model, 1., observations, actions)
        self.assertFalse(noop['applied'])
        self.assertTrue(torch.equal(policy.log_std, before))
        with self.assertRaisesRegex(ValueError, 'Parent deterministic'):
            tr.initialize_exploration_phase(model, .25, observations, actions + .1)
        self.assertTrue(torch.equal(policy.log_std, before))

    def test_exploration_phase_validation_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in (0, -1., 1.01, float('nan'), float('inf'), True, '.25'):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'factor'):
                    tr.run_control_block(tr.TrainConfig(), tr.ControlConfig(), root/'invalid',
                        resume_run=root/'absent', initial_exploration_std_factor=value)
                self.assertFalse((root/'invalid').exists())
            with self.assertRaisesRegex(ValueError, 'requires --resume-run'):
                tr.run_control_block(tr.TrainConfig(), tr.ControlConfig(), root/'fresh',
                    initial_exploration_std_factor=.25)
            self.assertFalse((root/'fresh').exists())

    def test_control_budget_and_path_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, '1800'):
                tr.run_control_block(tr.TrainConfig(max_wall_seconds=1801), tr.ControlConfig(), root/'new')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                tr.run_control_block(tr.TrainConfig(), tr.ControlConfig(), root)
            self.assertFalse((root/'new').exists())


if __name__ == '__main__':
    unittest.main()
