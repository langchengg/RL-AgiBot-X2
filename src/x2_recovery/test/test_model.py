"""Regression checks for the pinned loader, address mapping and falsifiable audit."""

import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco as mj
import numpy as np

from x2_recovery.model import (SCENE, SOURCE_HASHES, URDF, apply_overrides,
                               compiled_mapping, load_effective_model, resolve_assets)
from x2_recovery.model_audit import (expected_actuation, fingerprint, fresh, mapping_audit,
                                    run_audit, save_report, verify_actuation, visual_review,
                                    freefall_displacement, image_review_context, penetration_sample, result)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = load_effective_model()

    def test_pelvis_full_tensor_and_conservative_ranges(self):
        m = self.loaded.model
        self.assertAlmostEqual(m.body('pelvis').mass[0], 3.523487, places=12)
        self.assertAlmostEqual(sum(m.body_mass[1:]), 41.966473, places=10)
        self.assertEqual(self.loaded.joint('head_yaw_joint').position_range, (-.349, .349))
        self.assertEqual(self.loaded.joint('left_hip_yaw_joint').position_range, (-1.684, 3.43))
        self.assertEqual(self.loaded.joint('left_wrist_roll_joint').effort_range, (-2.2, 2.2))
        # Full tensor validation occurs on every load (including off-diagonals).
        self.assertEqual(len(self.loaded.overrides), 8)
        self.assertEqual(m.jnt_margin[m.joint('waist_pitch_joint').id], .005)

    def test_nonzero_named_mapping_and_example(self):
        evidence = mapping_audit(self.loaded)
        self.assertEqual(evidence['status'], 'PASS')
        self.assertTrue(evidence['order_differs'])
        example = evidence['example']
        self.assertEqual((example['ctrl_index'], example['joint_id'], example['qpos_address'],
                          example['dof_address']), (15, 30, 36, 35))
        self.assertEqual(example['qfrc_actuator'], .2)

    def test_independent_repeated_load_after_mutation(self):
        before = fingerprint(self.loaded)
        other = load_effective_model(self.loaded.asset_repo)
        self.assertIsNot(self.loaded.model, other.model)
        self.assertEqual(before, fingerprint(other))
        other.model.body_mass[1] = 123
        self.assertEqual(before, fingerprint(load_effective_model(self.loaded.asset_repo)))
        self.assertEqual(before, fingerprint(self.loaded))

    def test_offline_load_from_another_cwd_without_heavy_imports(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
        code = """
import sys
from x2_recovery.model import load_effective_model
x = load_effective_model(sys.argv[1])
if any(n in sys.modules for n in ('rclpy', 'torch', 'stable_baselines3', 'x2_recovery.model_audit')):
    raise RuntimeError('Heavy/audit import leaked into runtime loader')
print(x.joint('head_yaw_joint').qpos_address)
"""
        with tempfile.TemporaryDirectory() as cwd:
            completed = subprocess.run([sys.executable, '-c', code, str(self.loaded.asset_repo)],
                                       env=env, cwd=cwd, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), '36')

    def test_missing_path_and_wrong_pin_hash_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Missing asset repository'):
                load_effective_model(Path(directory)/'missing')
            with self.assertRaisesRegex(ValueError, 'Cannot verify pinned assets'):
                load_effective_model(directory)
        with patch.dict(SOURCE_HASHES, {SCENE: '0'*64}):
            with self.assertRaisesRegex(ValueError, 'hash precondition'):
                resolve_assets(self.loaded.asset_repo)

    def test_patch_preconditions_cannot_silently_fall_back(self):
        spec = mj.MjSpec.from_file(str(self.loaded.asset_repo/SCENE))
        urdf = ET.parse(self.loaded.asset_repo/URDF).getroot()
        spec.body('pelvis').explicitinertial = True
        with self.assertRaisesRegex(ValueError, 'Pelvis override precondition'):
            apply_overrides(spec, urdf)
        spec = mj.MjSpec.from_file(str(self.loaded.asset_repo/SCENE))
        spec.joint('waist_yaw_joint').range = [-3, 2]
        with self.assertRaisesRegex(ValueError, 'Range override precondition'):
            apply_overrides(spec, urdf)

    def test_unexpected_mapping_and_actuation_are_rejected(self):
        m = copy.copy(self.loaded.model)
        m.actuator_trnid[0, 0] = m.actuator_trnid[1, 0]
        with self.assertRaisesRegex(ValueError, 'Unexpected actuator name|mapping'):
            compiled_mapping(m)
        m = copy.copy(self.loaded.model)
        m.actuator_gear[0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'Unsupported motor semantics'):
            compiled_mapping(m)
        m = copy.copy(self.loaded.model)
        m.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_CLAMPCTRL)
        with self.assertRaisesRegex(ValueError, 'disabled physics'):
            compiled_mapping(m)

    def test_wrong_expected_force_fails_and_unclipped_input_saturates(self):
        m = self.loaded.model
        row = self.loaded.joint('left_wrist_roll_joint')
        d = fresh(m, high=True)
        d.ctrl[row.ctrl_index] = 2.31
        mj.mj_forward(m, d)
        force, torque = expected_actuation(m, row, 2.31)
        self.assertEqual((force, torque), (2.2, 2.2))
        verify_actuation(m, d, row, force, torque)
        self.assertEqual(d.ctrl[row.ctrl_index], 2.31)
        with self.assertRaisesRegex(ValueError, 'Actuator force mismatch'):
            verify_actuation(m, d, row, 4.8, 4.8)
        with self.assertRaisesRegex(ValueError, 'Transmitted joint actuation mismatch'):
            verify_actuation(m, d, row, 2.2, -2.2)

    def test_failed_run_replaces_stale_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            path.write_text('{"run_completed":true,"verdict":"COMPLETE"}')
            with patch('builtins.print'):
                code = run_audit(Path(directory)/'missing', directory,
                                 project=Path(__file__).resolve().parents[3])
            report = json.loads(path.read_text())
            self.assertEqual(code, 1)
            self.assertFalse(report['run_completed'])
            self.assertEqual(report['verdict'], 'PARTIAL/BLOCKED')
            self.assertEqual(report['checks']['run']['status'], 'FAIL')

    def test_strict_nonfinite_and_stale_visual_review_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            self.assertFalse(save_report(path, {'checks':{},'verdict':'COMPLETE','measurement':np.nan}))
            report = json.loads(path.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertEqual(report['verdict'], 'PARTIAL/BLOCKED')
            self.assertEqual(report['measurement']['status'], 'FAIL')
        self.assertEqual(visual_review([], {})['status'], 'NOT_TESTED')
        with self.assertRaisesRegex(ValueError, 'Missing/stale visual review|four'):
            visual_review([{'path':f'{i}.png','image_sha256':'actual'} for i in range(4)],
                          {f'{i}.png':{'sha256':'old','observation':'stale image'} for i in range(4)})


class CloseoutTests(unittest.TestCase):
    """Only the closeout changes; no dynamic probe or rendering reruns here."""

    def test_review_requires_matching_physics_probe_and_pixels(self):
        loaded = load_effective_model()
        definitions = [{'steps': 180, 'controls': 'zero'}]
        context = image_review_context(loaded, definitions)['sha256']
        frames = [{'path': f'{i}.png', 'image_sha256': f'pixels-{i}',
                   'review_context_sha256': context} for i in range(4)]
        review = {f'{i}.png': {'sha256': f'pixels-{i}', 'context_sha256': context,
                               'observation': 'Explicit test-only inspection fixture'} for i in range(4)}
        self.assertEqual(visual_review(copy.deepcopy(frames), review)['status'], 'PASS')
        changed = copy.copy(loaded)
        changed.model = copy.copy(loaded.model)
        changed.model.body_mass[1] += .01  # Physics can change without changing a rendered silhouette.
        contexts = [image_review_context(changed, definitions)['sha256'],
                    image_review_context(loaded, [{'steps': 181, 'controls': 'zero'}])['sha256']]
        for new_context in contexts:
            with self.subTest(context=new_context):
                stale = copy.deepcopy(frames)
                for frame in stale:
                    frame['review_context_sha256'] = new_context
                with self.assertRaisesRegex(ValueError, 'model or probe context'):
                    visual_review(stale, review)
                self.assertTrue(all('visual_inspection' not in frame for frame in stale))
        stale = copy.deepcopy(frames)
        stale[-1]['image_sha256'] = 'different pixels'
        with self.assertRaisesRegex(ValueError, 'Missing/stale visual review'):
            visual_review(stale, review)
        self.assertTrue(all('visual_inspection' not in frame for frame in stale))
        legacy = {k: {f:v for f,v in row.items() if f != 'context_sha256'} for k,row in review.items()}
        with self.assertRaisesRegex(ValueError, 'model or probe context'):
            visual_review(frames, legacy)

    def test_same_report_path_reads_review_before_invalidation(self):
        loaded = load_effective_model()
        frames = [{'path': f'{i}.png', 'image_sha256': f'pixels-{i}',
                   'review_context_sha256': 'test-context'} for i in range(4)]
        review = {f'{i}.png': {'sha256': f'pixels-{i}', 'context_sha256': 'test-context',
                               'observation': 'Explicit test-only inspection fixture'} for i in range(4)}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory)/'report.json'
            path.write_text(json.dumps({'run_completed': True, 'verdict': 'COMPLETE', 'image_review': review}))
            prefix = 'x2_recovery.model_audit.'
            for name in ('mass_audit', 'range_audit', 'actuator_audit', 'mapping_audit', 'base_audit',
                         'limit_audit', 'repeat_audit'):
                stack.enter_context(patch(prefix+name, return_value=result(True, 'Stub; no physics rerun')))
            stack.enter_context(patch(prefix+'identity', return_value={}))
            stack.enter_context(patch(prefix+'provenance', return_value={}))
            stack.enter_context(patch(prefix+'inventory', return_value={}))
            stack.enter_context(patch(prefix+'load_effective_model', return_value=loaded))
            stack.enter_context(patch(prefix+'mj.MjSpec.from_file',
                                      return_value=SimpleNamespace(compile=lambda: loaded.model)))
            stack.enter_context(patch('builtins.print'))

            def collision(*args):
                current = json.loads(path.read_text())
                self.assertFalse(current['run_completed'])
                self.assertEqual(current['verdict'], 'PARTIAL/BLOCKED')
                self.assertEqual(current['checks']['visual_review']['status'], 'NOT_TESTED')
                return result(True, 'Stub collision evidence', images=copy.deepcopy(frames))

            stack.enter_context(patch(prefix+'collision_audit', side_effect=collision))
            code = run_audit(loaded.asset_repo, directory, project=Path.cwd(), image_review_from=path)
            self.assertEqual(code, 0)
            current = json.loads(path.read_text())
            self.assertEqual(current['image_review'], review)
            self.assertEqual(current['checks']['visual_review']['status'], 'PASS')
            # Valid prior visual review must never carry an old numerical success into a failing run.
            with patch(prefix+'base_audit', return_value=result(False, 'Intentional numerical failure')):
                self.assertEqual(run_audit(loaded.asset_repo, directory, project=Path.cwd(),
                                          image_review_from=path), 1)
            self.assertEqual(json.loads(path.read_text())['verdict'], 'PARTIAL/BLOCKED')

    def test_penetration_pair_separates_floor_peak_and_final(self):
        contacts = [{'floor': False, 'geom1': 9, 'geom2': 67, 'distance_m': -.1},
                    {'floor': True, 'geom1': 0, 'geom2': 15, 'distance_m': -.002},
                    {'floor': True, 'geom1': 0, 'geom2': 18, 'distance_m': -.004}]
        peak = penetration_sample(contacts, .05)
        self.assertEqual((peak['penetration_m'], peak['geom_pair'], peak['time_s']), (.004, [0,18], .05))
        final = penetration_sample([contacts[1]], .18)
        self.assertEqual((final['penetration_m'], final['geom_pair']), (.002, [0,15]))
        initial = penetration_sample([], 0.)
        self.assertEqual((initial['penetration_m'], initial['geom_pair']), (0., None))
        self.assertEqual(penetration_sample(contacts, .05, floor=False)['geom_pair'], [9,67])

    def test_freefall_expectation_is_discrete_euler_and_rejects_other_integrators(self):
        model = SimpleNamespace(opt=SimpleNamespace(integrator=mj.mjtIntegrator.mjINT_EULER,
                                                    timestep=.001, gravity=np.array([0.,0.,-9.81])))
        self.assertAlmostEqual(freefall_displacement(model, 100)[2], -.0495405, places=15)
        self.assertAlmostEqual(.5*model.opt.gravity[2]*(100*model.opt.timestep)**2, -.04905, places=15)
        model.opt.integrator = mj.mjtIntegrator.mjINT_RK4
        with self.assertRaisesRegex(ValueError, 'actual Euler integrator'):
            freefall_displacement(model, 100)


if __name__ == '__main__':
    unittest.main()
