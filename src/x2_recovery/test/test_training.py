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


if __name__ == '__main__':
    unittest.main()
