"""Real mapping/physics tests plus explicitly synthetic lifecycle fault tests.

Synthetic success below verifies runner wiring only, never physical recovery.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from x2_recovery import baseline as b
from x2_recovery.env import EnvConfig, EnvExecutionError, X2RecoveryEnv
from x2_recovery.reset import integration_state


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = X2RecoveryEnv()
        cls.env.reset(seed=60)
        cls.initial, _ = cls.env.loaded.read_state(cls.env.data)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_boundaries_continuity_range_and_final_hold(self):
        e = self.env
        f = b.build_keyframes(self.initial, e.loaded.mapping)
        for i, t in enumerate(f.times_s):
            q, phase = b.scripted_targets(t, f)
            np.testing.assert_array_equal(q, f.targets_rad[i])
            if i < len(f.phases):
                self.assertEqual(phase['name'], f.phases[i])
                self.assertEqual(phase['progress'], 0.)
            if t > 0:
                np.testing.assert_allclose(b.scripted_targets(t-1e-6, f)[0], q, atol=1e-12, rtol=0)
            np.testing.assert_allclose(b.scripted_targets(t+1e-6, f)[0], q, atol=1e-12, rtol=0)
        for t in np.linspace(0., 30., 301):
            q, _ = b.scripted_targets(t, f)
            self.assertTrue(np.all(q >= e.q_min) and np.all(q <= e.q_max))
            np.testing.assert_allclose(e.action_targets(e.action_for_targets(q))[1], q, atol=2e-7, rtol=0)
        q, phase = b.scripted_targets(1000., f)
        np.testing.assert_array_equal(q, f.targets_rad[-1])
        self.assertEqual(phase, dict(name='hold', progress=1., sequence_finished=True))

    def test_current_initial_copy_purity_and_inheritance(self):
        e = self.env
        before = integration_state(e.model, e.data)
        initial = self.initial.copy()
        f = b.build_keyframes(initial, e.loaded.mapping)
        initial[:] = 0
        np.testing.assert_array_equal(f.targets_rad[0], self.initial)
        self.assertFalse(np.array_equal(f.targets_rad[0], e.q_ref))
        names = [r.joint_name for r in e.loaded.mapping]
        for i, name in enumerate(names):
            if 'head' in name or 'wrist' in name:
                np.testing.assert_array_equal(f.targets_rad[:, i], np.full(len(f.times_s), self.initial[i]))
        for t in np.linspace(0, 20, 51):
            q, _ = b.scripted_targets(t, f)
            q[:] = 10
        np.testing.assert_array_equal(before, integration_state(e.model, e.data))
        changed = self.initial.copy()
        changed[names.index('head_yaw_joint')] += .01
        next_frames = b.build_keyframes(changed, e.loaded.mapping)
        np.testing.assert_array_equal(next_frames.targets_rad[0], changed)
        np.testing.assert_array_equal(f.targets_rad[0], self.initial)

    def test_names_follow_actual_mapping_not_update_insertion_order(self):
        mapping = self.env.loaded.mapping
        updates = {'right_knee_joint': .9, 'left_hip_pitch_joint': -.6}
        a = b.build_keyframes(self.initial, mapping, stages=(('move', 2., updates),))
        rev = dict(reversed(list(updates.items())))
        c = b.build_keyframes(self.initial, mapping, stages=(('move', 2., rev),))
        np.testing.assert_array_equal(a.targets_rad, c.targets_rad)
        order = np.arange(31)[::-1]
        d = b.build_keyframes(self.initial[order], tuple(mapping[i] for i in order),
                              stages=(('move', 2., updates),))
        np.testing.assert_array_equal(a.targets_rad[:, order], d.targets_rad)

    def test_reject_invalid_parameters(self):
        mapping = self.env.loaded.mapping
        for duration in (0., -1., np.nan, np.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                b.build_keyframes(self.initial, mapping, stages=(('move', duration, {}),))
        for updates in ({'typo_joint': .2}, {'left_knee_joint': np.nan}, {'left_knee_joint': np.inf}):
            with self.assertRaises(ValueError):
                b.build_keyframes(self.initial, mapping, stages=(('move', 1., updates),))
        for q in (np.zeros(30), np.zeros((31, 1)), np.full(31, np.nan), np.full(31, np.inf)):
            with self.assertRaises(ValueError):
                b.build_keyframes(q, mapping)
        for times in ([0, 0], [0, -1], [1, 2], [0, np.inf]):
            with self.assertRaises(ValueError):
                b.Keyframes(times, np.zeros((2, 31)), ('move',))
        for targets in (np.zeros((3, 31)), np.zeros((2, 30)), np.full((2, 31), np.nan)):
            with self.assertRaises(ValueError):
                b.Keyframes([0, 1], targets, ('move',))
        f = b.build_keyframes(self.initial, mapping)
        for t in (-1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                b.scripted_targets(t, f)

    def test_projection_uses_existing_tolerance_and_records_clamps(self):
        e = self.env
        initial = self.initial.copy()
        initial[0] = e.q_min[0]-5e-7
        before = initial.copy()
        f = b.build_keyframes(initial, e.loaded.mapping,
                              stages=(('move', 1., {'left_knee_joint': 3.}),))
        np.testing.assert_array_equal(initial, before)
        self.assertEqual(f.adjustments[0]['kind'], 'initial_reference_projection')
        self.assertAlmostEqual(f.adjustments[0]['delta_rad'], 5e-7)
        self.assertEqual(f.adjustments[1]['kind'], 'candidate_target_clamp')
        initial[0] = e.q_min[0]-1.1e-6
        with self.assertRaisesRegex(ValueError, 'reset joint tolerance'):
            b.build_keyframes(initial, e.loaded.mapping)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'run'

    def read_outputs(self):
        summary = json.loads((self.path/'summary.json').read_text())
        rows = [json.loads(line) for line in (self.path/'trajectory.jsonl').read_text().splitlines()]
        return summary, rows

    def test_real_short_timeout_partial_substeps_and_closed_resources(self):
        e = X2RecoveryEnv(EnvConfig(episode_timeout_s=.043))
        with patch.object(b, 'X2RecoveryEnv', return_value=e), \
                patch.object(e, 'reset', wraps=e.reset) as reset, patch.object(e, 'step', wraps=e.step) as step:
            result = b.run_episode(output_dir=self.path, timeout_s=.043)
        summary, rows = self.read_outputs()
        self.assertEqual(reset.call_count, 1)
        self.assertEqual(step.call_count, 3)
        self.assertEqual(len(rows), 4)
        self.assertTrue(e._closed)
        self.assertTrue(result['execution_completed'])
        self.assertFalse(result['recovery_success'])
        self.assertEqual(result['truncation_reason'], 'time_limit')
        self.assertEqual(result['physics_steps'], 43)
        self.assertEqual(rows[-1]['info']['physics_steps_executed'], 3)
        self.assertEqual(summary['elapsed_sim_s'], rows[-1]['elapsed_sim_s'])
        self.assertEqual(summary['final_info'], rows[-1]['info'])
        self.assertGreater(rows[0]['sim_end_s'], 0.)
        self.assertEqual(rows[0]['elapsed_sim_s'], 0.)
        np.testing.assert_array_equal(rows[0]['q_rad'], summary['keyframes']['targets_rad'][0])
        np.testing.assert_array_equal(rows[-1]['q_rad'], e.loaded.read_state(e.data)[0])
        self.assertGreater(max(abs(x) for x in rows[-1]['ctrl_Nm']), 0.)
        for a, c in zip(rows, rows[1:]):
            self.assertEqual(a['sim_end_s'], c['sim_start_s'])
        with patch.object(b, 'X2RecoveryEnv') as create:
            self.assertEqual(b.main(['--output-dir', str(self.path)]), 1)
            create.assert_not_called()

    def test_sequence_end_is_not_success_and_keeps_stepping_real_env(self):
        original = b.build_keyframes
        def brief(initial, mapping):
            return original(initial, mapping, stages=(('brief', .001, {}),))
        with patch.object(b, 'build_keyframes', side_effect=brief):
            result = b.run_episode(output_dir=self.path, timeout_s=.063)
        _, rows = self.read_outputs()
        self.assertEqual(result['control_steps'], 4)
        self.assertFalse(result['recovery_success'])
        self.assertTrue(rows[2]['phase']['sequence_finished'])
        self.assertFalse(rows[2]['terminated'] or rows[2]['truncated'])

    def test_synthetic_success_and_safety_stop_once(self):
        for reason in ('success', 'safety_abort:synthetic'):
            e = X2RecoveryEnv()
            original = e.step
            def synthetic(action):
                obs, reward, _, _, info = original(action)
                info.update(is_success=reason == 'success', termination_reason=reason)
                return obs, reward, True, False, info
            path = self.path/reason
            with patch.object(b, 'X2RecoveryEnv', return_value=e), \
                    patch.object(e, 'reset', wraps=e.reset) as reset, \
                    patch.object(e, 'step', side_effect=synthetic) as step:
                result = b.run_episode(output_dir=path)
            self.assertTrue(result['execution_completed'])
            self.assertEqual(result['recovery_success'], reason == 'success')
            self.assertEqual(step.call_count, 1)
            self.assertEqual(reset.call_count, 1)
            self.assertTrue(e._closed)

    def test_step_error_preserves_first_error_partial_evidence_and_close(self):
        e = X2RecoveryEnv()
        original_step, original_close = e.step, e.close
        calls = 0
        def failing_step(action):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise EnvExecutionError('first error', {'invalid_numeric': 'nan'})
            return original_step(action)
        def failing_close():
            original_close()
            raise RuntimeError('secondary close error')
        with patch.object(b, 'X2RecoveryEnv', return_value=e), \
                patch.object(e, 'step', side_effect=failing_step), patch.object(e, 'close', side_effect=failing_close):
            code = b.main(['--output-dir', str(self.path)])
        summary, rows = self.read_outputs()
        self.assertEqual(code, 1)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(summary['recovery_success'])
        self.assertFalse(summary['execution_completed'])
        self.assertEqual(summary['execution_error']['message'], 'first error')
        self.assertEqual(summary['secondary_errors'][0]['message'], 'secondary close error')
        self.assertTrue(e._closed)

    def test_reset_error_and_cancellation_are_incomplete(self):
        for index, error in enumerate((EnvExecutionError('illegal reset', {'reason': 'fixture'}), KeyboardInterrupt())):
            e = X2RecoveryEnv()
            with patch.object(b, 'X2RecoveryEnv', return_value=e), \
                    patch.object(e, 'reset', side_effect=error) as reset, patch.object(e, 'step') as step:
                result = b.run_episode(output_dir=self.path/str(index))
            self.assertEqual(reset.call_count, 1)
            step.assert_not_called()
            self.assertFalse(result['execution_completed'])
            self.assertIsNone(result['recovery_success'])
            self.assertTrue(e._closed)
            self.assertEqual(result['cancelled'], index == 1)

    def test_cancel_during_episode_returns_nonzero_and_keeps_written_state(self):
        e = X2RecoveryEnv()
        with patch.object(b, 'X2RecoveryEnv', return_value=e), \
                patch.object(e, 'step', side_effect=KeyboardInterrupt()) as step:
            code = b.main(['--output-dir', str(self.path)])
        summary, rows = self.read_outputs()
        self.assertEqual(code, 130)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(e._closed)
        self.assertTrue(summary['cancelled'])
        self.assertIsNone(summary['recovery_success'])

    def test_nonfinite_measured_output_is_diagnostic_error(self):
        original = b._state
        def nonfinite(env):
            state = original(env)
            state['q_rad'][0] = np.nan  # copied log data only, synthetic fault
            return state
        with patch.object(b, '_state', side_effect=nonfinite):
            result = b.run_episode(output_dir=self.path)
        self.assertFalse(result['execution_completed'])
        self.assertIsNone(result['recovery_success'])
        self.assertIn('Nonfinite', result['execution_error']['message'])
        summary, rows = self.read_outputs()
        self.assertEqual(rows, [])
        self.assertTrue(summary['environment_closed'])

    def test_trajectory_write_error_and_nonfinite_data_do_not_become_failure_outcomes(self):
        original = b._write_record
        for index, error in enumerate((OSError('disk full'), ValueError('Nonfinite output value'))):
            e = X2RecoveryEnv()
            calls = 0
            def write(stream, record):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise error
                return original(stream, record)
            with patch.object(b, 'X2RecoveryEnv', return_value=e), patch.object(b, '_write_record', side_effect=write):
                result = b.run_episode(output_dir=self.path/str(index))
            self.assertFalse(result['execution_completed'])
            self.assertIsNone(result['recovery_success'])
            self.assertEqual(result['last_recorded_control_step'], 0)
            self.assertTrue(e._closed)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            b._json_value({'state': np.array([np.nan])})

    def test_summary_failure_nonzero_and_bad_parameters(self):
        original = Path.open
        def fail_summary(path, *args, **kwargs):
            if path.name.startswith('summary.'):
                raise OSError('summary unavailable')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', fail_summary):
            self.assertEqual(b.main(['--timeout-s', '.023', '--output-dir', str(self.path)]), 1)
        for value in ('nan', '0', '-1', '.0205'):
            self.assertNotEqual(b.main(['--timeout-s', value, '--output-dir', str(self.path/value)]), 0)


if __name__ == '__main__':
    unittest.main()
