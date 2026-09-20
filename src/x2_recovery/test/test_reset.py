"""Targeted runtime invariants; full 20+20 acceptance is the installed diagnostic."""
import copy
from dataclasses import replace
import os
import time
import unittest
os.environ.setdefault('MUJOCO_GL','osmesa')
import mujoco as mj
import numpy as np
from x2_recovery.model import load_effective_model
from x2_recovery.reset import (ResetSettings,ResetFailure,construct_pose,reset_supine,
    measurements,settled_failures,initial_geometry,notify,checked_step,integration_state)
from x2_recovery.step4 import contaminate,pulse,ceiling,causal_joint


class ResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded=load_effective_model();cls.m=cls.loaded.model
        cls.settled=mj.MjData(cls.m)
        cls.result=reset_supine(cls.loaded,cls.settled,seed=100,settings=replace(ResetSettings(),perturb_rad=.005))

    def data(self):return mj.MjData(self.m)

    def test_clear_previous_episode_and_seeded_construction(self):
        a,b=self.data(),self.data()
        contaminate(self.loaded,a,17)
        settings=replace(ResetSettings(),perturb_rad=.005)
        aa=construct_pose(self.loaded,a,100,settings);bb=construct_pose(self.loaded,b,100,settings)
        np.testing.assert_array_equal(integration_state(self.m,a),integration_state(self.m,b))
        self.assertEqual(a.time,0.)
        self.assertEqual(aa['requested_joints_rad'],bb['requested_joints_rad'])
        self.assertAlmostEqual(np.linalg.norm(a.qpos[3:7]),1.,places=12)
        self.assertAlmostEqual(aa['geometry']['minimum_clearance_m'],.002,places=10)
        self.assertLess(aa['geometry']['query_support_max_error_m'],2e-6)
        self.assertEqual(len(aa['requested_joints_rad']),31)
        for j in self.loaded.mapping:self.assertTrue(j.position_range[0]<a.qpos[j.qpos_address]<j.position_range[1])
        cc=construct_pose(self.loaded,b,101,settings)
        self.assertNotEqual(aa['requested_joints_rad'],cc['requested_joints_rad'])

    def test_invalid_seed_settings_model_and_collision_are_rejected(self):
        d=self.data()
        for seed,settings in ((-1,ResetSettings()),(None,ResetSettings()),
                              (0,replace(ResetSettings(),perturb_rad=.02)),
                              (0,replace(ResetSettings(),floor_m=float('nan')))):
            with self.assertRaises(ValueError):construct_pose(self.loaded,d,seed,settings)
        other=load_effective_model()
        with self.assertRaisesRegex(ValueError,'different model'):construct_pose(other,d,0,ResetSettings())
        construct_pose(self.loaded,d,0,ResetSettings());d.qpos[2]-=.01;mj.mj_forward(self.m,d)
        with self.assertRaisesRegex(ValueError,'floor intersection'):initial_geometry(self.m,d,ResetSettings())

    def test_actual_handoff_state_time_and_callback_equivalence(self):
        d=self.data();contaminate(self.loaded,d,999);frames=[]
        result=reset_supine(self.loaded,d,seed=100,settings=replace(ResetSettings(),perturb_rad=.005),observer=frames.append)
        np.testing.assert_array_equal(d.qpos,self.settled.qpos)
        np.testing.assert_array_equal(d.qvel,self.settled.qvel)
        np.testing.assert_array_equal(result['final_qvel'],d.qvel)
        self.assertGreater(np.max(abs(d.qvel)),0.)
        self.assertEqual(result['episode_start_time'],d.time)
        self.assertEqual(result['window_maxima'],self.result['window_maxima'])
        self.assertEqual(frames[0]['phase'],'placement');self.assertEqual(frames[-1]['phase'],'ready')
        self.assertGreaterEqual(d.time-result['settled_start_s'],1.5-1e-10)
        origin=d.time;checked_step(self.loaded,d,time.monotonic()+5)
        self.assertAlmostEqual(d.time-origin,.001,places=10)
        self.assertFalse(np.shares_memory(frames[-1]['qpos'],d.qpos))

    def test_reject_prone_sideways_unsupported_motion_and_penetration(self):
        weight=sum(self.m.body_mass)*9.81;s=ResetSettings()
        for label,axis,angle in [('prone',[0.,1.,0.],np.pi/2),('side',[1.,0.,0.],np.pi/2)]:
            d=self.data();mj.mj_copyData(d,self.m,self.settled)
            mj.mju_axisAngle2Quat(d.qpos[3:7],np.array(axis),angle);mj.mj_forward(self.m,d)
            failures=settled_failures(measurements(self.loaded,d),s,weight)
            self.assertIn('torso_face_deg',failures,label)
        for label,z,v,expected in [('unsupported',1.,0.,'back load support'),('moving',0.,.2,'base_linear_m_s'),
                                    ('penetrating',-.02,0.,'floor penetration')]:
            d=self.data();mj.mj_copyData(d,self.m,self.settled);d.qpos[2]+=z;d.qvel[0]=v;mj.mj_forward(self.m,d)
            self.assertIn(expected,settled_failures(measurements(self.loaded,d),s,weight),label)

    def test_timeout_never_returns_ready(self):
        d=self.data()
        with self.assertRaises(ResetFailure) as caught:
            reset_supine(self.loaded,d,seed=0,settings=replace(ResetSettings(),max_settle_s=.001))
        self.assertEqual(caught.exception.evidence['status'],'FAIL')
        self.assertIsNone(caught.exception.evidence['episode_start_time'])

    def test_observer_interference_and_callback_are_not_hidden(self):
        d=self.data()
        with self.assertRaisesRegex(ValueError,'Observer/viewer'):
            notify(self.m,d,lambda frame:d.ctrl.fill(.1),'test')
        try:
            mj.set_mjcb_control(lambda m,d:None)
            with self.assertRaisesRegex(ValueError,'Unexpected MuJoCo callback'):
                construct_pose(self.loaded,d,0,ResetSettings())
        finally:mj.set_mjcb_control(None)

    def test_limit_margin_is_part_of_physics_identity(self):
        from x2_recovery.model_audit import fingerprint
        x=copy.copy(self.loaded);x.model=copy.copy(self.m)
        before=fingerprint(x);x.model.jnt_margin[self.m.joint('waist_pitch_joint').id]=0.
        self.assertNotEqual(before,fingerprint(x))

    def test_real_head_motion_and_neutral_controls_on_error(self):
        row=self.loaded.joint('head_yaw_joint')
        result=causal_joint(self.loaded,self.settled,row)
        self.assertEqual(result['status'],'PASS')
        self.assertGreater(min(result['attempts'][-1]['causal_max_displacement_rad']),1e-5)
        d=self.data();mj.mj_copyData(d,self.m,self.settled)
        wrong=replace(row,dof_address=self.loaded.joint('left_knee_joint').dof_address)
        with self.assertRaisesRegex(ValueError,'effort mismatch'):
            pulse(self.loaded,d,[(wrong,ceiling(row))])
        np.testing.assert_array_equal(d.ctrl,0.)
        np.testing.assert_array_equal(d.qfrc_applied,0.)
        np.testing.assert_array_equal(d.xfrc_applied,0.)


if __name__=='__main__':unittest.main()
