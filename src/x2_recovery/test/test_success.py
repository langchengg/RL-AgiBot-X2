"""Constructed measurements test logic only; X2 physics is runtime_check success-check."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS
import math
import unittest
from unittest.mock import patch
import numpy as np
from x2_recovery.success import (SuccessSettings, SuccessTracker, standing_failures,
                                  ground_forces, contact_world_force)


def good(t=0., **changes):
    v = dict(time_s=t, pelvis_height_m=.67, upright_dot=1., tilt_deg=0.,
             left_weight=.5, right_weight=.5, other_weight=0., other_force_N=0.,
             base_linear_m_s=0., com_linear_m_s=0., base_angular_rad_s=0.,
             torso_angular_rad_s=0., joint_rad_s=0., joint_limit_rad=0.,
             floor_penetration_m=0., self_penetration_m=0., support_gap_m=0.,
             pelvis_xy=[0., 0.], left_xy=[0., .1], right_xy=[0., -.1], support=[])
    v.update(changes)
    return v


class SuccessLogicTests(unittest.TestCase):
    def setUp(self):
        self.s = SuccessSettings(.67)
        self.tracker = SuccessTracker(self.s, .001)

    def feed(self, start, stop, tracker=None, **changes):
        tracker = tracker or self.tracker
        for k in range(start, stop+1): result = tracker.update(good(k*.001, **changes))
        return result

    def logical_origin(self, tracker=None):
        # Constructed origin is deliberately internal and ONLY proves the logical gate.
        (tracker or self.tracker).origin = {'time_s': 0., 'qpos': (), 'qvel': ()}

    def test_full_two_seconds_not_early(self):
        self.assertFalse(self.feed(0, 1999)['standing_held'])
        self.assertTrue(self.feed(2000, 2000)['standing_held'])
        self.assertFalse(self.tracker.result['recovery_success'])

    def test_first_sample_receives_no_free_timestep(self):
        self.assertFalse(self.feed(1, 2000)['standing_held'])
        self.assertTrue(self.feed(2001, 2001)['standing_held'])

    def test_interrupted_time_not_accumulated(self):
        self.feed(0, 1800)
        self.tracker.update(good(1.801, left_weight=0.))
        r = self.feed(1802, 2102)
        self.assertAlmostEqual(r['stable_duration_s'], .3)
        self.assertFalse(r['standing_held'])

    def test_each_scalar_upper_boundary(self):
        fields = {'other_weight': self.s.other_weight_max, 'base_linear_m_s': .1,
                  'com_linear_m_s': .1, 'base_angular_rad_s': .25, 'torso_angular_rad_s': .25,
                  'joint_rad_s': .5, 'joint_limit_rad': .0001, 'floor_penetration_m': .001,
                  'self_penetration_m': .0001, 'support_gap_m': 2e-6}
        for key, boundary in fields.items():
            for delta in (-1e-9, 0., 1e-9):
                with self.subTest(key=key, delta=delta):
                    self.assertEqual(key in standing_failures(good(**{key: boundary+delta}), self.s), delta > 0)

    def test_height_boundary(self):
        for delta in (-1e-9, 0., 1e-9):
            self.assertEqual('height' in standing_failures(good(pelvis_height_m=.67*.9+delta), self.s), delta < 0)

    def test_torso_boundary_and_inversion(self):
        for delta in (-1e-9, 0., 1e-9):
            dot = math.cos(math.radians(15))+delta
            self.assertEqual('torso_tilt' in standing_failures(good(upright_dot=dot), self.s), delta < 0)
        self.assertIn('torso_tilt', standing_failures(good(upright_dot=-1.), self.s))
        self.assertIn('torso_tilt', standing_failures(good(upright_dot=.5), self.s))

    def test_left_right_support_boundaries(self):
        for side in ('left', 'right'):
            for delta in (-1e-9, 0., 1e-9):
                v = good(**{side+'_weight': .05+delta})
                self.assertEqual(side+'_support' in standing_failures(v, self.s), delta < 0)
            self.assertIn(side+'_support', standing_failures(good(**{side+'_weight': 0.}), self.s))

    def test_total_support_bounds(self):
        for boundary in (.8, 1.2):
            for delta in (-1e-9, 0., 1e-9):
                total = boundary+delta
                self.assertEqual('feet_total' in standing_failures(good(left_weight=total/2, right_weight=total/2), self.s),
                                 total < .8 or total > 1.2)

    def test_drift_boundaries_and_maximum_from_origin(self):
        for key, limit in [('pelvis', .05), ('left', .03), ('right', .03)]:
            for delta in (-1e-9, 0., 1e-9):
                tr = SuccessTracker(self.s, .001);tr.update(good())
                xy = good()[key+'_xy'];xy[0] += limit+delta
                r = tr.update(good(.001, **{key+'_xy': xy}))
                self.assertEqual(key+'_drift' in r['failures'], delta > 0)
                if delta > 0: self.assertIsNone(tr.reference)
        self.tracker.update(good())
        self.tracker.update(good(.001, pelvis_xy=[.03, 0.]))
        r = self.tracker.update(good(.002, pelvis_xy=[.04, 0.]))
        self.assertAlmostEqual(r['drift_m']['pelvis'], .04)
        r = self.tracker.update(good(.003, pelvis_xy=[.01, 0.]))
        self.assertAlmostEqual(r['drift_m']['pelvis'], .04)

    def test_repeat_timestamp_no_credit_or_new_reference(self):
        v = good(); self.tracker.update(v); self.feed(1, 700)
        before = copy.deepcopy(self.tracker.reference)
        for _ in range(20): r = self.tracker.update(good(.7))
        self.assertAlmostEqual(r['stable_duration_s'], .7)
        np.testing.assert_array_equal(before['pelvis'], self.tracker.reference['pelvis'])
        r = self.tracker.update(good(.7, left_weight=.6))
        self.assertIn('inconsistent', r['invalid_reason'])

    def test_rollback_reset_gap_and_nonuniform_samples_invalid(self):
        for next_time in (-.001, .002, .020, .0005):
            tr = SuccessTracker(self.s, .001);tr.update(good())
            self.assertIsNotNone(tr.update(good(next_time))['invalid_reason'])
            self.assertFalse(tr.update(good(.021))['standing_held'])

    def test_substep_violation_cannot_hide_in_policy_step(self):
        self.feed(0, 1990)
        for k in range(1991, 2011):
            r = self.tracker.update(good(k*.001, left_weight=0. if k == 1995 else .5))
        self.assertFalse(r['standing_held'])
        self.assertLess(r['stable_duration_s'], .020)
        tr = SuccessTracker(self.s, .001);tr.update(good())
        self.assertIn('missing', tr.update(good(.02))['invalid_reason'])

    def test_new_episode_and_instance_isolation(self):
        self.feed(0, 2100)
        other = SuccessTracker(self.s, .001)
        self.assertEqual(other.update(good())['stable_duration_s'], 0.)
        self.tracker.reset(10.)
        self.assertFalse(self.tracker.update(good(10., pelvis_xy=[1., 2.]))['standing_held'])
        self.assertIsNone(self.tracker.origin)

    def test_deadline_before_exact_and_after(self):
        for timeout, success in [(2.001, True), (2., True), (1.999, False)]:
            tr = SuccessTracker(self.s, .001, timeout);self.logical_origin(tr)
            r = self.feed(0, 2000, tr)
            self.assertEqual(r['recovery_success'], success)
            self.assertEqual(r['timed_out'], not success)
        tr = SuccessTracker(self.s, .001, 2.)
        self.logical_origin(tr)
        self.assertFalse(self.feed(1, 2001, tr)['recovery_success'])
        self.assertTrue(tr.result['timed_out'])

    def test_standing_fixture_never_recovery(self):
        r = self.feed(0, 2200)
        self.assertTrue(r['standing_held']);self.assertFalse(r['recovery_success'])

    def test_origin_and_execution_gate_logic_only(self):
        self.logical_origin();self.assertTrue(self.feed(0, 2000)['recovery_success'])
        self.tracker.invalidate('caller reports teleport')
        r = self.feed(2001, 2001)
        self.assertFalse(r['recovery_success']);self.assertIn('teleport', r['invalid_reason'])

    def test_nonfinite_configuration_and_measurements(self):
        for key in good():
            if isinstance(good()[key], (float, int)):
                for value in (math.nan, math.inf, -math.inf):
                    tr = SuccessTracker(self.s, .001)
                    self.assertIsNotNone(tr.update(good(**{key: value}))['invalid_reason'])
        for key in self.s.__dataclass_fields__:
            for value in (math.nan, math.inf, 0., -1.):
                with self.assertRaises(ValueError): replace(self.s, **{key: value})
        for change in ({'feet_weight_min': 1.3}, {'tilt_deg_max': 100.}, {'other_weight_max': .01}):
            with self.assertRaises(ValueError): replace(self.s, **change)
        with self.assertRaises(TypeError): SuccessSettings()
        with self.assertRaises(ValueError): SuccessSettings(None)
        for dt, timeout in [(0., 20.), (.001, 0.), (math.nan, 20.), (.001, math.inf)]:
            with self.assertRaises(ValueError): SuccessTracker(self.s, dt, timeout)


class ContactLogicTests(unittest.TestCase):
    """Synthetic contact records / force reader, not physical integration evidence."""
    def contact(self, a=0, b=2, efc=0, frame=None):
        return NS(geom1=a, geom2=b, dist=-.0001, efc_address=efc,
                  frame=np.eye(3) if frame is None else np.array(frame))

    def context(self):
        m = NS(geom_bodyid=[0, 1, 2, 3], body=lambda b: NS(name=str(b)))
        return NS(loaded=NS(model=m), floor=0, robot_geoms={1, 2, 3}, feet=({1}, {3}))

    def forces(self, contacts, vectors):
        def reader(m, d, i, out): out[:]=[*vectors[i], 0., 0., 0.]
        with patch('x2_recovery.success.mj.mj_contactForce', side_effect=reader):
            return ground_forces(self.context(), NS(contact=contacts))

    def test_rotated_frame_and_floor_in_both_geom_positions(self):
        frame = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
        f = np.array([3., 2., 1., 0., 0., 0.])
        np.testing.assert_array_equal(contact_world_force(self.contact(frame=frame), f, 0), [2, 1, 3])
        reverse_frame = np.array(frame);reverse_frame[:2] *= -1  # proper 180-degree rotation
        reverse_force = f.copy();reverse_force[2] *= -1
        self.assertAlmostEqual(np.linalg.det(reverse_frame), 1.)
        np.testing.assert_array_equal(contact_world_force(self.contact(a=2, b=0, frame=reverse_frame), reverse_force, 0), [2, 1, 3])

    def test_many_small_contacts_accumulate(self):
        rows = [self.contact() for _ in range(10)]
        feet, other, support, *_ = self.forces(rows, [[0., 0., .001]]*10)
        self.assertAlmostEqual(other, .01);self.assertEqual(len(support), 10)
        self.assertIn('other_weight', standing_failures(good(other_weight=other/411.), SuccessSettings(.67)))

    def test_opposed_tangential_forces_do_not_cancel(self):
        feet, other, support, *_ = self.forces([self.contact(), self.contact()], [[1., 0., .001], [-1., 0., .001]])
        self.assertGreater(other, 2.)

    def test_self_contact_and_inactive_and_zero_force(self):
        feet, other, support, *_ = self.forces([self.contact(a=1, b=2), self.contact(efc=-1), self.contact(b=1)],
                                               [[0., 0., 100.], [0., 0., 100.], [0., 0., 0.]])
        self.assertEqual(feet, [0., 0.]);self.assertEqual(other, 0.);self.assertEqual(support, [])

    def test_signed_negative_vertical_fails(self):
        with self.assertRaisesRegex(ValueError, 'Negative'):
            self.forces([self.contact()], [[0., 0., -1.]])


class X2MeasurementTests(unittest.TestCase):
    """Actual compiled X2 / API checks, separate from constructed logic above."""
    @classmethod
    def setUpClass(cls):
        import mujoco as mj
        from x2_recovery.model import load_effective_model
        from x2_recovery.success import StandingContext
        cls.mj=mj;cls.loaded=load_effective_model();cls.ctx=StandingContext(cls.loaded)

    def test_mapping_weight_axes_and_read_only_measurement(self):
        from x2_recovery.success import measure_standing
        from x2_recovery.reset import integration_state
        m=self.loaded.model;d=self.mj.MjData(m);self.mj.mj_forward(m,d)
        before=integration_state(m,d).copy();v=measure_standing(self.ctx,d)
        np.testing.assert_array_equal(before,integration_state(m,d))
        self.assertEqual(tuple(map(len,self.ctx.feet)),(12,12))
        self.assertAlmostEqual(self.ctx.weight_N,m.body_subtreemass[self.ctx.pelvis]*np.linalg.norm(m.opt.gravity))
        self.assertEqual(v['upright_dot'],1.)
        self.assertEqual(v['pelvis_height_m'],d.xpos[self.ctx.pelvis,2]-self.ctx.floor_origin[2])

    def test_actual_velocity_api_against_jacobians(self):
        from x2_recovery.success import measure_standing
        m=self.loaded.model;d=self.mj.MjData(m)
        d.qpos[2]+=2.;d.qvel[:]=np.linspace(-.2,.3,m.nv);self.mj.mj_forward(m,d)
        v=measure_standing(self.ctx,d)
        jp=np.zeros((3,m.nv));jr=jp.copy();jc=jp.copy()
        self.mj.mj_jacBody(m,d,jp,jr,self.ctx.pelvis)
        self.assertAlmostEqual(v['base_linear_m_s'],np.linalg.norm(jp@d.qvel))
        self.assertAlmostEqual(v['base_angular_rad_s'],np.linalg.norm(jr@d.qvel))
        self.mj.mj_jacBody(m,d,jp,jr,self.ctx.torso)
        self.assertAlmostEqual(v['torso_angular_rad_s'],np.linalg.norm(jr@d.qvel))
        self.mj.mj_jacSubtreeCom(m,d,jc,self.ctx.pelvis)
        self.assertAlmostEqual(v['com_linear_m_s'],np.linalg.norm(jc@d.qvel))
        self.assertNotAlmostEqual(v['com_linear_m_s'],v['base_linear_m_s'])

    def test_changed_model_and_mapping_rejected(self):
        from x2_recovery.success import StandingContext
        x=copy.copy(self.loaded);x.model=copy.copy(x.model);x.model.opt.gravity[2]=0.
        with self.assertRaisesRegex(ValueError,'calibration'):StandingContext(x)
        x=copy.copy(self.loaded);x.mapping=(replace(x.mapping[0],dof_address=0),)+x.mapping[1:]
        with self.assertRaisesRegex(ValueError,'mapping'):StandingContext(x)
        x=copy.copy(self.loaded);x.model=copy.copy(x.model)
        x.model.geom_contype[next(iter(self.ctx.feet[0]))]=0
        with self.assertRaises(ValueError):StandingContext(x)

    def test_invalid_forces_state_and_controls_rejected(self):
        from x2_recovery.success import measure_standing
        m=self.loaded.model
        for kind in ['qfrc','xfrc','nan','quaternion','control','warning']:
            d=self.mj.MjData(m);self.mj.mj_forward(m,d)
            if kind=='qfrc':d.qfrc_applied[0]=.01
            if kind=='xfrc':d.xfrc_applied[1,2]=.01
            if kind=='nan':d.qvel[0]=np.nan
            if kind=='quaternion':d.qpos[3]=2.
            if kind=='control':d.ctrl[0]=119.
            if kind=='warning':d.warning.number[0]=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):measure_standing(self.ctx,d)

    def test_real_reset_handoff_origin_and_residual_state(self):
        from x2_recovery.reset import reset_supine, checked_step
        from x2_recovery.success import CALIBRATED_SETTINGS, measure_standing
        import time
        d=self.mj.MjData(self.loaded.model);r=reset_supine(self.loaded,d,seed=0)
        tr=SuccessTracker(CALIBRATED_SETTINGS,self.ctx.dt)
        tr.reset_from_supine(self.ctx,d,r)
        self.assertEqual(tr.start,r['episode_start_time'])
        np.testing.assert_array_equal(tr.origin['qvel'],r['final_qvel'])
        self.assertGreater(np.max(abs(d.qvel)),0.)
        result=tr.update(measure_standing(self.ctx,d))
        self.assertFalse(result['instant_standing_ok']);self.assertEqual(result['stable_duration_s'],0.)
        checked_step(self.loaded,d,time.monotonic()+5.)
        with self.assertRaisesRegex(ValueError,'handoff'):tr.reset_from_supine(self.ctx,d,r)


if __name__ == '__main__': unittest.main()
