"""Synthetic/fault-injection logic tests; none count as real X2 recoveries."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from x2_recovery import recovery_node as rn
from x2_recovery.telemetry_node import TelemetryNode, validate_joint_state
from x2_recovery.model import load_effective_model


class SyntheticEnv:
    def __init__(self, mapping, clock):
        self.loaded = SimpleNamespace(mapping=mapping, read_state=self.read_state, asset_repo='synthetic')
        self.clock = clock
        self.control_dt = .02
        self.data = SimpleNamespace(time=0.)
        self.episode_start_time = None
        self.resets = self.steps = self.closes = 0
        self.q = np.zeros(31)
        self.dq = np.arange(31, dtype=float)/100
        self.reset_delay = self.step_delay = 0.
        self.reset_error = self.step_error = None
        self.terminated = self.truncated = self.success = False

    def read_state(self, data):
        return self.q.copy(), self.dq.copy()

    def info(self):
        return dict(is_success=self.success, elapsed_sim_s=self.data.time-self.episode_start_time,
                    termination_reason='success' if self.success else 'safety_abort:synthetic',
                    truncation_reason='time_limit', reset_seed=123)

    def reset(self, *, seed):
        self.resets += 1
        self.clock.now += self.reset_delay
        if self.reset_error:
            raise self.reset_error
        self.data.time = self.episode_start_time = 1.661
        self.q[:] = [min(.01*self.resets, row.position_range[1]) for row in self.loaded.mapping]
        return None, self.info()

    def action_for_targets(self, q):
        self.target = q.copy()
        return q.copy()

    def step(self, action):
        self.steps += 1
        self.clock.now += self.step_delay
        if self.step_error:
            raise self.step_error
        self.data.time += self.control_dt
        self.q += .002
        return None, 999., self.terminated, self.truncated, self.info()

    def close(self):
        self.closes += 1


class RecoveryLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load_effective_model().mapping

    def setUp(self):
        self.context = Context()
        rclpy.init(context=self.context)
        self.clock = SimpleNamespace(now=100.)
        self.clock.monotonic = lambda: self.clock.now
        self.env = SyntheticEnv(self.mapping, self.clock)
        self.time_patch = patch.object(rn, 'time', self.clock)
        self.env_patch = patch.object(rn, 'X2RecoveryEnv', return_value=self.env)
        self.time_patch.start(); self.env_patch.start()
        self.addCleanup(self.time_patch.stop); self.addCleanup(self.env_patch.stop)
        self.node = rn.RecoveryNode(context=self.context)
        self.addCleanup(self.context.shutdown)
        self.addCleanup(self.node.destroy_node)
        self.addCleanup(self.node.close)
        self.logs = []
        self.node._log = lambda event, **kw: self.logs.append(dict(event=event, **kw))

    def start(self):
        return self.node._start(Trigger.Request(), Trigger.Response())

    def finish_logs(self):
        return [r for r in self.logs if r['event'] == 'finished']

    def test_acceptance_pending_busy_no_environment_calls_or_mutation(self):
        self.assertEqual((self.env.resets, self.env.steps), (0, 0))
        self.assertTrue(self.start().success)
        before = (self.node.seed, self.node.deadline, self.node.accepted_at,
                  self.node.episode, self.node.control_steps, self.node.keyframes)
        self.clock.now += .001
        response = self.start()
        self.assertFalse(response.success)
        self.assertEqual(response.message, 'Recovery already running')
        self.assertEqual(before, (self.node.seed, self.node.deadline, self.node.accepted_at,
                                 self.node.episode, self.node.control_steps, self.node.keyframes))
        self.assertEqual((self.env.resets, self.env.steps), (0, 0))

    def test_reset_separate_callback_each_step_single_and_fresh_keyframes(self):
        self.start(); self.node._advance()
        self.assertEqual((self.env.resets, self.env.steps), (1, 0))
        np.testing.assert_array_equal(self.node.keyframes.targets_rad[0], self.env.q)
        self.node._advance(); self.assertEqual(self.env.steps, 1)
        self.node._advance(); self.assertEqual(self.env.steps, 2)
        self.env.truncated = True; self.node._advance()
        self.assertTrue(self.start().success); self.node._advance()
        self.assertEqual(self.env.resets, 2)
        np.testing.assert_array_equal(self.node.keyframes.targets_rad[0], self.env.q)
        self.assertEqual(self.env.q[0], .02)
        self.assertEqual(self.node.control_steps, 0)

    def test_success_safety_and_truncation_mapping(self):
        for terminated, success, truncated, expected, reason in [
            (True, True, False, 'SUCCEEDED', 'success'),
            (True, False, False, 'FAILED', 'safety_abort:synthetic'),
            (False, False, True, 'FAILED', 'time_limit')]:
            self.start(); self.node._advance()
            self.env.terminated, self.env.success, self.env.truncated = terminated, success, truncated
            self.node._advance()
            self.assertEqual(self.node.status, expected)
            self.assertEqual(self.finish_logs()[-1]['reason'], reason)
            n = self.env.steps; ends = len(self.finish_logs())
            for _ in range(3): self.node._advance()
            self.assertEqual(self.env.steps, n)
            self.assertEqual(len(self.finish_logs()), ends)
            self.assertFalse(self.node.busy or self.node.pending)

    def test_sequence_finished_and_large_reward_are_not_success(self):
        self.start(); self.node._advance()
        self.env.data.time = self.env.episode_start_time+100.
        self.node._advance()
        self.assertEqual(self.node.status, 'RUNNING')
        np.testing.assert_array_equal(self.env.target, self.node.keyframes.targets_rad[-1])

    def test_before_reset_and_before_step_deadline_boundary(self):
        self.start(); self.clock.now = self.node.deadline; self.node._advance()
        self.assertEqual(self.env.resets, 0)
        self.assertEqual(self.finish_logs()[-1]['reason'], 'recovery_timeout')
        self.start(); self.node._advance(); self.clock.now = self.node.deadline
        self.node._advance(); self.assertEqual(self.env.steps, 0)

    def test_after_reset_deadline_and_after_action_deadline(self):
        self.env.reset_delay = 30.
        self.start(); self.node._advance()
        self.assertEqual(self.node.status, 'FAILED'); self.assertEqual(self.env.steps, 0)
        self.env.reset_delay = 0.
        self.start(); self.node._advance()
        def slow(q):
            self.clock.now = self.node.deadline
            return q
        self.env.action_for_targets = slow
        self.node._advance(); self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.finish_logs()[-1]['reason'], 'recovery_timeout')

    def test_late_environment_success_keeps_evidence_but_fails_ros(self):
        self.start(); self.node._advance()
        self.env.terminated = self.env.success = True
        self.env.step_delay = 30.125
        self.node._advance()
        end = self.finish_logs()[-1]
        self.assertEqual((self.node.status, end['reason']), ('FAILED', 'recovery_timeout'))
        self.assertTrue(end['environment_result']['is_success'])
        self.assertEqual(end['timeout_overshoot_s'], .125)

    def test_exact_deadline_success_fails_and_timely_success_survives_later_logging(self):
        self.start(); self.node._advance()
        self.env.terminated = self.env.success = True; self.env.step_delay = 30.
        self.node._advance(); self.assertEqual(self.node.status, 'FAILED')
        self.env.step_delay = 29.9
        self.start(); self.node._advance()
        original = self.node._sample
        def slow_sample():
            value = original(); self.clock.now += 1.; return value
        with patch.object(self.node, '_sample', side_effect=slow_sample): self.node._advance()
        self.assertEqual(self.node.status, 'SUCCEEDED')
        self.assertAlmostEqual(self.finish_logs()[-1]['wall_s'], 29.9)

    def test_reset_step_action_and_nonfinite_faults_allow_explicit_retry(self):
        for where in ('reset', 'step', 'action', 'sample'):
            self.env.reset_error = self.env.step_error = None
            self.env.q[:] = 0.
            self.start()
            if where == 'reset': self.env.reset_error = RuntimeError('first reset fault')
            self.node._advance()
            if where != 'reset':
                if where == 'step': self.env.step_error = RuntimeError('first step fault')
                if where == 'sample': self.env.q[0] = np.nan
                if where == 'action':
                    with patch.object(self.env, 'action_for_targets', side_effect=ValueError('action fault')):
                        self.node._advance()
                else: self.node._advance()
            self.assertEqual(self.node.status, 'FAILED')
            self.assertIn('error', self.finish_logs()[-1])
            self.assertFalse(self.node.busy)

    def test_invalid_state_overrides_success_and_is_not_published(self):
        self.start(); self.node._advance()
        self.env.q[0] = np.nan; self.env.terminated = self.env.success = True
        with patch.object(self.node.joint_pub, 'publish') as publish:
            self.node._advance(); publish.assert_not_called()
        self.assertEqual(self.node.status, 'FAILED')
        self.assertIn('position', self.finish_logs()[-1]['error']['message'])
        self.assertTrue(self.finish_logs()[-1]['environment_result']['is_success'])

    def test_first_error_retains_environment_evidence_without_requery(self):
        from x2_recovery.env import EnvExecutionError
        self.start(); self.node._advance()
        evidence = {'first_fault': 'synthetic', 'partial_steps': 3}
        self.env.step_error = EnvExecutionError('original failure', evidence)
        with patch.object(self.env.loaded, 'read_state', side_effect=AssertionError('must not requery')):
            self.node._advance()
        error = self.finish_logs()[-1]['error']
        self.assertEqual(error, dict(type='EnvExecutionError', message='original failure', evidence=evidence))
        self.assertFalse(self.node.busy)

    def test_deadline_can_expire_during_reset_start_logging(self):
        self.start()
        original = self.node._log
        def delayed(event, **fields):
            if event == 'reset_started': self.clock.now = self.node.deadline
            original(event, **fields)
        self.node._log = delayed
        self.node._advance()
        self.assertEqual(self.env.resets, 0)
        self.assertEqual(self.finish_logs()[-1]['reason'], 'recovery_timeout')

    def test_joint_snapshot_is_actual_mapping_and_detached(self):
        self.start(); self.node._advance(); self.node._advance()
        m = self.node._sample()
        self.assertEqual(m.name, [r.joint_name for r in self.mapping])
        np.testing.assert_array_equal(m.position, self.env.q)
        np.testing.assert_array_equal(m.velocity, self.env.dq)
        self.assertFalse(m.effort)
        self.assertGreater(m.header.stamp.sec, 0)
        self.assertFalse(m.header.frame_id)
        self.env.q[:] = 8.
        self.assertNotEqual(m.position[0], 8.)

    def test_shutdown_cancels_pending_and_closes_once(self):
        self.start(); self.node.close(); self.node._advance(); self.node.close()
        self.assertEqual((self.env.resets, self.env.steps, self.env.closes), (0, 0, 1))
        self.assertTrue(self.node.timer.is_canceled())
        self.assertEqual(self.finish_logs()[-1]['reason'], 'shutdown')
        self.assertFalse(self.start().success)

    def test_unrecoverable_resource_error_stops_accepting(self):
        self.start(); self.env.reset_error = MemoryError('synthetic resource failure')
        self.node._advance()
        self.assertTrue(self.node.fatal_error and self.node.closing)
        self.assertFalse(self.start().success)

    def test_startup_cleanup_does_not_replace_first_error(self):
        with patch.object(self.env.loaded, 'mapping', ()), \
                patch.object(self.env, 'close', side_effect=RuntimeError('secondary cleanup')):
            with self.assertRaisesRegex(ValueError, 'Invalid controlled joint mapping'):
                rn.RecoveryNode(context=self.context)

    def test_startup_parameters_and_runtime_changes_rejected(self):
        for name in ('seed', 'episode_timeout_s', 'recovery_timeout_s', 'use_sim_time'):
            descriptor = self.node.describe_parameter(name)
            self.assertEqual(descriptor.name, name)
            self.assertEqual(descriptor.type, self.node.get_parameter(name).type_.value)
            self.assertTrue(descriptor.read_only)
        for name, value in [('seed', -1), ('recovery_timeout_s', 0.),
                            ('recovery_timeout_s', float('nan')), ('episode_timeout_s', -1.),
                            ('use_sim_time', True)]:
            with self.assertRaises((ValueError, RuntimeError)):
                rn.RecoveryNode(context=self.context, parameter_overrides=[Parameter(name, value=value)])
        for name, value in [('seed', 61), ('episode_timeout_s', 3.),
                            ('recovery_timeout_s', 2.), ('use_sim_time', True)]:
            result = self.node.set_parameters([Parameter(name, value=value)])
            self.assertFalse(result[0].successful)


class TelemetryLogicTests(unittest.TestCase):
    def setUp(self):
        self.context = Context(); rclpy.init(context=self.context)
        self.node = TelemetryNode(context=self.context)
        self.addCleanup(self.context.shutdown); self.addCleanup(self.node.destroy_node)

    def sample(self, names=None):
        m = JointState(); m.name = names or ['head_joint', 'left_knee_joint']
        m.position = [1., .23]; m.header.stamp = self.node.get_clock().now().to_msg()
        return m

    def test_waiting_missing_joint_terminal_and_new_running_old_sample(self):
        self.assertIn('status=waiting', self.node.description())
        self.assertIn('no_sample', self.node.description())
        self.node._joint(self.sample(['a', 'b']))
        self.assertIn('missing_joint', self.node.description())
        self.node._joint(self.sample()); self.node._status(String(data='FAILED'))
        self.assertIn('last_sample position_rad=0.230000 sample_age_s=', self.node.description())
        self.node._status(String(data='RUNNING'))
        self.assertIn('waiting_for_current_sample', self.node.description())
        self.node._joint(self.sample())
        self.assertNotIn('last_sample', self.node.description())
        self.node._status(String(data='BUSY'))
        self.assertEqual(self.node.status, 'RUNNING')

    def test_invalid_messages_rejected_keep_last_valid_without_restamping(self):
        m = self.sample(); self.node._joint(m); before = self.node.received_at
        for mutate in [lambda x: setattr(x, 'name', ['a', 'a']),
                       lambda x: setattr(x, 'position', [1.]),
                       lambda x: setattr(x, 'velocity', [1.]),
                       lambda x: setattr(x, 'effort', [1.]),
                       lambda x: setattr(x, 'position', [float('nan'), 1.]),
                       lambda x: setattr(x, 'velocity', [float('inf'), 1.]),
                       lambda x: setattr(x, 'name', ['', 'a'])]:
            bad = copy.deepcopy(m); mutate(bad)
            with self.assertRaises(ValueError): validate_joint_state(bad)
            self.node._joint(bad)
            self.assertIs(self.node.sample, m); self.assertEqual(self.node.received_at, before)
        validate_joint_state(m)  # Empty velocity/effort are legal.


if __name__ == '__main__':
    unittest.main()
