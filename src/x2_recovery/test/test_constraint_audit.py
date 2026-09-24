"""Calculator and observation plumbing tests; synthetic data is not robot evidence."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco as mj
import numpy as np

from x2_recovery.constraint_audit import (
    PhysicalObserver, _contacts, _joint_rows, directional_ratio,
    compact_physical_arrays, discount_rewards, excess, intervals, json_value, reset_match, run_diagnostic,
)
from x2_recovery.model import URDF


class CalculatorTests(unittest.TestCase):
    def test_sway_numpy_vectors_are_json_serializable(self):
        sway = dict(pelvis_xy_peak_to_peak_m=np.array([.001, .002]),
                    torso_tilt_range_deg=np.float64(.3))
        self.assertEqual(json.loads(json.dumps(json_value(sway))),
                         dict(pelvis_xy_peak_to_peak_m=[.001, .002], torso_tilt_range_deg=.3))

    def test_reset_match_rejects_physical_mismatch_without_restoring(self):
        actual = {'qpos': np.array([.1, .2]), 'qvel': np.array([0., 0.])}
        expected = {'qpos': np.array([.1, .2]), 'qvel': np.array([0., 0.])}
        self.assertEqual(reset_match(actual, expected, 1e-10), {'qpos': 0., 'qvel': 0.})
        expected['qpos'][0] += .001
        with self.assertRaisesRegex(ValueError, 'Reset handoff mismatch'):
            reset_match(actual, expected, 1e-10)
        np.testing.assert_array_equal(actual['qpos'], [.1, .2])

    def test_compact_retention_uses_actual_joint_mapping(self):
        source = dict(qpos=np.array([[3.1, 99., -2.2], [1., 99., 0.]]),
            qvel=np.array([[99., -2., 7.], [99., 5., 1.]]),
            integration_qfrc_actuator_Nm=np.array([[99., -8., 3.], [99., 4., -6.]]),
            joint_qpos_addresses=np.array([2, 0]), joint_dof_addresses=np.array([1, 2]),
            post_time_s=np.array([.001, .002]), postforward_floor_depth_m=np.array([.003, .004]))
        result = compact_physical_arrays(source, [-2., -1.], [1., 3.], [[-8., 4.], [-12., 3.]])
        self.assertNotIn('qpos', result)
        np.testing.assert_allclose(result['max_joint_excess_rad'], [.2, 0.])
        np.testing.assert_array_equal(result['worst_joint_index'], [0, 0])
        np.testing.assert_array_equal(result['max_abs_joint_velocity_rad_s'], [7., 5.])
        np.testing.assert_array_equal(result['max_actuation_limit_ratio'], [1., 1.])
        np.testing.assert_array_equal(result['postforward_floor_depth_m'], [.003, .004])
        self.assertIn('qpos', source)

    def test_asymmetric_limits_and_joint_order(self):
        q = np.array([[-2.5, 3.4], [1.2, -1.1]])
        np.testing.assert_allclose(excess(q, [-2., -1.], [1., 3.]), [[.5, .4], [.2, .1]])
        np.testing.assert_allclose(directional_ratio([[2., -6.], [-4., 3.]], [[-8., 4.], [-12., 3.]]),
                                   [[.5, .5], [.5, 1.]])

    def test_right_endpoint_intervals_and_consecutive_segments(self):
        times = np.arange(1, 7)*.001
        spans = intervals([True, True, False, True, False, True], times, .001)
        self.assertEqual([s['samples'] for s in spans], [2, 1, 1])
        self.assertEqual(spans[0]['start_s'], 0.)
        self.assertEqual(spans[0]['end_s'], .002)
        self.assertEqual(sum(s['sampled_duration_s'] for s in spans), .004)
        self.assertEqual(intervals([False]*6, times, .001), [])
        with self.assertRaisesRegex(ValueError, 'Missing/nonuniform'):
            intervals([True, True], [.001, .003], .001)

    def test_discount_terms_and_short_final_transition(self):
        # Last transition can be 1ms; gamma is still once per transition.
        r = discount_rewards([2., 3., 5.], {'density': [2., 3., 0.], 'terminal': [0., 0., 5.]},
                             .9, terminated=True, truncated=False)
        self.assertEqual(r['raw_return'], 10.)
        self.assertAlmostEqual(r['observed_discounted_prefix'], 2+.9*3+.9**2*5)
        self.assertAlmostEqual(sum(r['discounted_terms'].values()), r['observed_discounted_prefix'])
        self.assertIsNone(r['critic_tail_estimate'])
        self.assertIsNone(r['historical_rollout_bootstrap'])

    def test_truncated_critic_is_separate_and_no_tail_after_termination(self):
        r = discount_rewards([1., 2.], {'task': [1., 2.]}, .9,
                             terminated=False, truncated=True, terminal_value=4.)
        self.assertAlmostEqual(r['observed_discounted_prefix'], 2.8)
        self.assertAlmostEqual(r['critic_tail_estimate'], .9**2*4.)
        with self.assertRaisesRegex(ValueError, 'only for time truncation'):
            discount_rewards([1.], {'task': [1.]}, .9, terminated=True,
                             truncated=False, terminal_value=4.)
        with self.assertRaisesRegex(ValueError, 'components disagree'):
            discount_rewards([1.], {'task': [2.]}, .9, terminated=False, truncated=True)

    def test_existing_output_rejected_before_reset_or_loading(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(FileExistsError):
                run_diagnostic(None, None, seed=1, output=path)

    def test_joint_csv_uses_mapping_addresses_and_declared_velocity(self):
        with tempfile.TemporaryDirectory() as path:
            urdf = Path(path)/URDF
            urdf.parent.mkdir(parents=True)
            urdf.write_text('<robot><joint name="b"><limit lower="-2" upper="1" effort="4" velocity="5"/></joint>'
                            '<joint name="a"><limit lower="-1" upper="3" effort="3"/></joint></robot>')
            mapping = [SimpleNamespace(joint_name='b'), SimpleNamespace(joint_name='a')]
            base = SimpleNamespace(loaded=SimpleNamespace(mapping=mapping, asset_repo=Path(path)),
                context=SimpleNamespace(qadr=np.array([2, 0]), vadr=np.array([1, 2]), efforts=np.array([[-8., 4.], [-12., 3.]])),
                q_min=np.array([-2., -1.]), q_max=np.array([1., 3.]), physics_dt=.001)
            rows = [dict(qpos=np.array([0., 9., -2.2]), qvel=np.array([99., 5., 1.]), post_time_s=.001,
                        target_rad=np.array([-.2, .1]), integration_qfrc_actuator_Nm=np.array([0., -8., 3.]),
                        tau_raw_Nm=np.array([-9., 3.]), ctrl_Nm=np.array([-8., 3.])),
                    dict(qpos=np.array([3.1, 9., .5]), qvel=np.array([99., 2., 7.]), post_time_s=.002,
                        target_rad=np.array([-.2, .1]), integration_qfrc_actuator_Nm=np.array([0., 4., -6.]),
                        tau_raw_Nm=np.array([4., -6.]), ctrl_Nm=np.array([4., -6.]))]
            values = _joint_rows(base, rows, 0., .0001)
            self.assertEqual([r['joint_name'] for r in values], ['b', 'a'])
            self.assertAlmostEqual(values[0]['max_excess_rad'], .2)
            self.assertAlmostEqual(values[1]['max_excess_rad'], .1)
            self.assertEqual(values[0]['max_abs_velocity_rad_s'], 5.)
            self.assertEqual(values[0]['max_velocity_ratio'], 1.)
            self.assertIsNone(values[1]['declared_velocity_limit_rad_s'])
            self.assertIsNone(values[1]['max_velocity_ratio'])
            self.assertEqual(values[0]['torque_saturation_fraction'], .5)
            self.assertEqual(values[0]['max_negative_actuation_Nm'], -8.)
            self.assertEqual(values[1]['max_actuation_limit_ratio'], 1.)


# A synthetic forwarding chain tests the exact observation insertion boundary.
# It is deliberately independent from the robot's physics and success criteria.
def checked_step(loaded, data, deadline):
    mj.mj_step(loaded.model, data)
    data.actuator_force[:] = 9.
    data.qfrc_actuator[:] = 9.


class FakeBase:
    pass


class ObserverTests(unittest.TestCase):
    def make_base(self):
        base = FakeBase()
        base.model = SimpleNamespace(opt=SimpleNamespace(integrator=mj.mjtIntegrator.mjINT_EULER))
        base.data = SimpleNamespace(time=0., qpos=np.array([.2]), qvel=np.array([.3]),
                                   ctrl=np.array([.4]), actuator_force=np.zeros(1),
                                   qfrc_actuator=np.zeros(1), contact=[])
        base.loaded = SimpleNamespace(model=base.model,
            read_state=lambda data: (data.qpos.copy(), data.qvel.copy()))
        base.context = SimpleNamespace(ctrladr=np.array([0]), floor=0)
        base._target = np.array([.5]); base.kp = np.array([2.]); base.kd = np.array([.1])
        return base

    def test_one_native_step_pre_force_post_forward_and_restore(self):
        base = self.make_base()
        calls = []
        def native(model, data):
            calls.append(1)
            data.time += .001; data.qpos += .01
            data.actuator_force[:] = .4; data.qfrc_actuator[:] = .4
        module = sys.modules[__name__]
        original_checked = module.checked_step
        with patch.object(mj, 'mj_step', native):
            with PhysicalObserver(base) as observer:
                observer.phase = 'recovery'
                module.checked_step(base.loaded, base.data, 99.)
            self.assertIs(mj.mj_step, native)
        self.assertIs(module.checked_step, original_checked)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(observer.records), 1)
        r = observer.records[0]
        self.assertEqual(r['pre_time_s'], 0.)
        self.assertEqual(r['post_time_s'], .001)
        np.testing.assert_array_equal(r['pre_qpos'], [.2])
        np.testing.assert_allclose(r['qpos'], [.21])
        np.testing.assert_allclose(r['tau_raw_Nm'], [.57])
        np.testing.assert_array_equal(r['integration_actuator_force_Nm'], [.4])
        np.testing.assert_array_equal(r['postforward_actuator_force_Nm'], [9.])

    def test_exception_restores_every_wrapper(self):
        base = self.make_base()
        module = sys.modules[__name__]
        original_checked, original_step = module.checked_step, mj.mj_step
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with PhysicalObserver(base):
                raise RuntimeError('injected')
        self.assertIs(module.checked_step, original_checked)
        self.assertIs(mj.mj_step, original_step)

    def test_force_orientation_and_contact_geometry(self):
        model = SimpleNamespace(geom_bodyid=np.array([0, 1, 2]),
            geom=lambda i: SimpleNamespace(name='g'+str(i)), body=lambda i: SimpleNamespace(name='b'+str(i)))
        # Normal is local axis zero, directed along world Z.
        frame = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]).ravel()
        data = SimpleNamespace(contact=[SimpleNamespace(geom1=0, geom2=1, dist=-.002, efc_address=0, frame=frame),
                                       SimpleNamespace(geom1=1, geom2=2, dist=-.003, efc_address=1, frame=frame)])
        def force(model, data, i, out):
            out[:] = [7., 0., 0., 0., 0., 0.]
        with patch.object(mj, 'mj_contactForce', force):
            result = _contacts(model, data, 0)
        self.assertEqual(result['ground_vertical_N'], 7.)
        self.assertEqual(result['ground_sum_norm_N'], 7.)
        self.assertEqual(result['floor']['geom_ids'], [0, 1])
        self.assertEqual(result['self']['bodies'], ['b1', 'b2'])


if __name__ == '__main__':
    unittest.main()
