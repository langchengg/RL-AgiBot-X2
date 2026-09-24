"""Synthetic search-metric tests; these are not physical recovery evidence."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from x2_recovery import motion_search as search


class RepairMetricsTests(unittest.TestCase):
    def observer(self):
        o = search.StreamingConstraintObserver.__new__(search.StreamingConstraintObserver)
        o.n = 2; o.phase = 'recovery'; o.target_hash = hashlib.sha256()
        o.base = SimpleNamespace(physics_dt=.001, episode_start_time=10.,
            q_min=np.zeros(2), q_max=np.ones(2),
            context=SimpleNamespace(efforts=np.array([[-2., 3.], [-4., 5.]]), weight_N=10.),
            loaded=SimpleNamespace(mapping=[SimpleNamespace(joint_name='a'), SimpleNamespace(joint_name='b')]))
        o.speed_limits = np.array([1., 2.]); o.velocity_sources = ['urdf/a', 'urdf/b']
        o.stats = {phase: o._empty() for phase in ('reset', 'recovery')}
        return o

    def row(self, q, index):
        return dict(q=np.array(q), dq=np.array([.5, -1.]), time=10.+index*.001,
                    pre_time=10.+(index-1)*.001, target=np.array([.2, .8]),
                    raw=np.array([4., -6.]), ctrl=np.array([3., -4.]),
                    force=np.array([3., -4.]), actuator=np.array([3., -4.]),
                    contacts=dict(ground_vertical_N=12., ground_sum_norm_N=13., floor=dict(depth_m=.002), self=dict(depth_m=.001)))

    def test_substep_extrema_duration_runs_and_directional_force(self):
        o = self.observer()
        for index, q in enumerate([[-.002, 1.001], [.5, .5], [-.003, .5], [-.004, .5]], 1):
            o.consume(self.row(q, index))
        result = o.summary('recovery'); a, b = result['joints']
        self.assertAlmostEqual(a['max_excess_rad'], .004)
        self.assertAlmostEqual(a['time_above_zero_excess_s'], .003)
        self.assertAlmostEqual(a['longest_consecutive_excess_s'], .002)
        self.assertAlmostEqual(b['time_above_standing_tolerance_s'], .001)
        self.assertAlmostEqual(result['first_violation_s'], .001)
        self.assertEqual(result['max_actuation_limit_ratio'], 1.)
        self.assertEqual(a['max_actuation_limit_ratio'], 1.)
        self.assertEqual(b['max_actuation_limit_ratio'], 1.)
        self.assertEqual(b['max_negative_actuation_Nm'], -4.)
        self.assertEqual(result['max_velocity_ratio'], .5)
        self.assertEqual(a['torque_saturation_fraction'], 1.)
        self.assertEqual(result['peak_ground_vertical_bodyweights'], 1.2)
        self.assertAlmostEqual(result['excess_integral_rad_s'], .000010)

    def test_wrong_motor_mapping_rejected(self):
        o = self.observer(); row = self.row([.5, .5], 1); row['actuator'][0] = 2.
        with self.assertRaisesRegex(ValueError, 'mapping changed'): o.consume(row)

    def test_actual_target_hash_detects_equivalence(self):
        a, b = self.observer(), self.observer()
        a.consume(self.row([.5, .5], 1)); b.consume(self.row([.6, .4], 1))
        self.assertEqual(a.target_hash.hexdigest(), b.target_hash.hexdigest())
        c = self.observer(); row = self.row([.5, .5], 1); row['target'][0] += .01; c.consume(row)
        self.assertNotEqual(a.target_hash.hexdigest(), c.target_hash.hexdigest())

    def candidate(self):
        comparison = dict(peak_ground_vertical_N=40., peak_ground_sum_norm_N=45., floor_penetration_m=.01,
                          self_penetration_m=.004, max_joint_speed_rad_s=8.)
        metrics = dict(comparison, max_joint_excess_rad=0., max_actuation_limit_ratio=1.,
                       max_velocity_ratio=.9, velocity_limits_complete=True,
                       excess_integral_rad_s=0., any_joint_above_tolerance_s=0.)
        result = dict(recovery=metrics, success=True, complete_episode=True, reason='success',
                      sim_seconds=5., episode_timeout_s=20., max_progress_quality=.9,
                      progress_integral_s=3., final_quality=.9, max_stable_hold_s=2.)
        return result, comparison

    def test_success_cannot_purchase_constraint_feasibility(self):
        result, comparison = self.candidate()
        self.assertTrue(search.repair_rank(result, comparison)['feasible'])
        result['complete_episode'] = False
        self.assertFalse(search.repair_rank(result, comparison)['feasible'])
        result['complete_episode'] = True
        result['recovery']['max_joint_excess_rad'] = .001
        self.assertFalse(search.repair_rank(result, comparison)['feasible'])
        result['recovery']['max_joint_excess_rad'] = 0.
        result['recovery']['max_velocity_ratio'] = 1.01
        self.assertFalse(search.repair_rank(result, comparison)['feasible'])

    def test_early_abort_and_no_motion_cannot_win_infeasible(self):
        result, comparison = self.candidate()
        result.update(success=False, reason='safety_abort:joint_limit_rad', sim_seconds=1.)
        self.assertFalse(search.repair_rank(result, comparison)['infeasible_eligible'])
        result.update(reason='time_limit', sim_seconds=20., max_progress_quality=.01)
        self.assertFalse(search.repair_rank(result, comparison)['infeasible_eligible'])

    def test_partial_batch_progress_uses_recorded_indices(self):
        candidates = [dict(name=str(i)) for i in range(5)]
        records = [dict(index=0, name='0')]
        state = search.repair_progress(candidates, records, [0, 1, 2, 3])
        self.assertEqual(state['recorded_indices'], [0])
        self.assertEqual(state['pending_indices'], [1, 2, 3])
        self.assertEqual(state['unrecorded_indices'], [1, 2, 3, 4])
        self.assertEqual(state['next_index'], 1)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            search.repair_progress(candidates, records+records, [0, 1])
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            search.repair_progress(candidates, [dict(index=0, name='wrong')], [0])
        state = search.repair_progress(candidates, [dict(index=i, name=str(i)) for i in range(5)], [])
        self.assertIsNone(state['next_index'])
        self.assertEqual(state['pending_indices'], [])

    def test_tracker_forwarding_preserves_result_and_measurement(self):
        o = self.observer()
        o.max_hold = o.max_progress = o.progress_integral = o.final_quality = 0.
        result = dict(stable_duration_s=.001)
        seen = []
        def original(measurement):
            seen.append(measurement)
            return result
        o.base.tracker = SimpleNamespace(update=original)
        measurement = dict(pelvis_height_m=.5, upright_dot=.8, left_weight=.3, right_weight=.4)
        initial = measurement.copy()
        with patch.object(search.PhysicalObserver, '__enter__', return_value=o), \
                patch.object(search.PhysicalObserver, '__exit__', return_value=False):
            with o:
                observed = o.base.tracker.update(measurement)
        self.assertIs(observed, result)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], measurement)
        self.assertEqual(measurement, initial)
        self.assertIs(o.base.tracker.update, original)
        self.assertEqual(o.max_hold, .001)
        self.assertEqual(o.peak_progress_measurement, initial)
        self.assertIsNot(o.peak_progress_measurement, measurement)

    def test_recorded_short_final_transition_counts_every_physics_step(self):
        # Actual published transition lengths, synthetic q/dq: calculator regression only.
        path = Path(__file__).resolve().parents[3]/'results/evaluation/ppo-reference-residual-20260922T212457Z/trajectory.jsonl.gz'
        counts = []; final = None
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('kind') != 'transition':
                    continue
                counts.append(row['info']['physics_steps_executed'])
                if row['terminated'] or row['truncated']:
                    final = row; break
        self.assertIsNotNone(final)
        self.assertLess(counts[-1], counts[0])
        observer = self.observer()
        for i in range(sum(counts)):
            observer.consume(self.row([-.001, .5], i+1))
        measured = observer.summary('recovery')
        self.assertEqual(measured['samples'], sum(counts))
        self.assertAlmostEqual(measured['sampled_duration_s'], final['info']['elapsed_sim_s'])
        self.assertAlmostEqual(measured['joints'][0]['time_above_zero_excess_s'], final['info']['elapsed_sim_s'])

    def test_exact_change_paths_and_historical_cli_route(self):
        before = {'reference_stages': [['stage', .2, {'a': 0.}]]}
        after = {'reference_stages': [['stage', .3, {'a': .1}]]}
        self.assertEqual([d['path'] for d in search.control_diff(before, after)],
                         ['controller.reference_stages[0][1]', 'controller.reference_stages[0][2].a'])
        with patch.object(search, 'search', return_value={'status': 'COMPLETE'}) as historical:
            search.main(['search', '--output', '/tmp/unused-repair-unit-path'])
            self.assertEqual(historical.call_args.kwargs['max_sim_s'], 8.)


class ExperimentalPDTests(unittest.TestCase):
    def fixture(self, multiplier=2.):
        original = search.X2RecoveryEnv.__init__.__globals__['EnvConfig']()
        rows = [list(row) for row in original.pd_gains]
        for row in rows:
            if row[0] == 'waist': row[2] *= multiplier
        names = ['waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint', 'left_ankle_pitch_joint']
        base = SimpleNamespace(config=original, kp=np.full(4, 300.), kd=np.full(4, original.pd_gains[3][2]),
            _ready=False, loaded=SimpleNamespace(mapping=[SimpleNamespace(joint_name=name) for name in names]),
            model=object(), data=SimpleNamespace(qpos=np.array([.1,.2]), qvel=np.array([.3,.4])))
        return original, {'pd_gains': rows}, base

    def test_explicit_override_and_resolved_arrays_restore_after_exception(self):
        original, overrides, base = self.fixture()
        saved = asdict(original); original_kp = base.kp.copy(); original_kd = base.kd.copy()
        kp_object, kd_object = base.kp, base.kd
        q, dq = base.data.qpos.copy(), base.data.qvel.copy()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with search.experimental_pd(base, original, overrides) as changed:
                self.assertEqual(changed.pd_gains[3][2], 2*original.pd_gains[3][2])
                self.assertIs(base.config, changed)
                np.testing.assert_array_equal(base.kp, original_kp)
                np.testing.assert_array_equal(base.kd[:3], original_kd[:3]*2)
                self.assertEqual(base.kd[3], original_kd[3])
                self.assertNotEqual(search._digest(asdict(changed)), search._digest(saved))
                raise RuntimeError('injected')
        self.assertIs(base.config, original)
        self.assertIs(base.kp, kp_object); self.assertIs(base.kd, kd_object)
        np.testing.assert_array_equal(base.kp, original_kp); np.testing.assert_array_equal(base.kd, original_kd)
        np.testing.assert_array_equal(base.data.qpos, q); np.testing.assert_array_equal(base.data.qvel, dq)
        self.assertEqual(asdict(original), saved)

    def test_original_multiplier_one_preserves_configuration_identity(self):
        original, overrides, base = self.fixture(1.)
        with search.experimental_pd(base, original, overrides) as candidate:
            self.assertEqual(search._digest(asdict(candidate)), search._digest(asdict(original)))
            np.testing.assert_array_equal(base.kd, np.full(4, original.pd_gains[3][2]))
        self.assertIs(base.config, original)
        self.assertIs(search.experimental_pd_config(original, None), original)

    def test_reject_unsupported_or_out_of_range_changes_before_mutation(self):
        original, overrides, base = self.fixture()
        variants = [dict(safety_joint_limit_rad=.1), {}, dict(overrides, reset_perturb_rad=.001)]
        for row_index, column, value in [(3, 0, 'torso'), (3, 1, 301.), (2, 2, 20.),
                                         (3, 2, 100.), (3, 2, .1), (3, 2, float('nan'))]:
            variant = deepcopy(overrides); variant['pd_gains'][row_index][column] = value; variants.append(variant)
        variant = deepcopy(overrides); variant['pd_gains'].reverse(); variants.append(variant)
        for bad in variants:
            with self.subTest(overrides=bad), self.assertRaises(ValueError):
                with search.experimental_pd(base, original, bad): pass
            self.assertIs(base.config, original)
            np.testing.assert_array_equal(base.kd, np.full(4, original.pd_gains[3][2]))

    def test_reject_active_episode_or_inconsistent_gain_arrays(self):
        original, overrides, base = self.fixture()
        base._ready = True
        with self.assertRaisesRegex(ValueError, 'active episode'):
            with search.experimental_pd(base, original, overrides): pass
        base._ready = False; base.kd[0] += .1
        with self.assertRaisesRegex(ValueError, 'arrays disagree'):
            with search.experimental_pd(base, original, overrides): pass

    def test_prefix_uses_same_samples_and_excludes_later_peaks(self):
        fixture = RepairMetricsTests(); o = fixture.observer()
        o.prefix_cutoff_s = .002
        o.prefix = dict(samples=0, covered_through_s=0., max_joint_excess_rad=0.,
            max_joint_speed_rad_s=0., any_joint_above_tolerance_s=0.,
            peak_ground_vertical_N=0., peak_ground_sum_norm_N=0.,
            floor_penetration_m=0., self_penetration_m=0.)
        o.consume(fixture.row([-.002,.5],1)); o.consume(fixture.row([-.001,.5],2))
        later = fixture.row([-.04,.5],3); later['contacts']['ground_vertical_N'] = 100.
        o.consume(later)
        self.assertEqual(o.prefix['samples'], 2)
        self.assertAlmostEqual(o.prefix['max_joint_excess_rad'], .002)
        self.assertAlmostEqual(o.prefix['any_joint_above_tolerance_s'], .002)
        self.assertEqual(o.prefix['peak_ground_vertical_N'], 12.)
        self.assertEqual(o.summary('recovery')['max_joint_excess_rad'], .04)
        self.assertEqual(o.summary('recovery')['peak_ground_vertical_N'], 100.)


if __name__ == '__main__': unittest.main()
