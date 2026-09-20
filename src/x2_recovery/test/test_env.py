"""Separate real-X2 integration and explicitly synthetic fault/reward tests."""
from dataclasses import replace
import copy
import json
import time
import unittest
from unittest.mock import patch
import mujoco as mj
import numpy as np
from x2_recovery.env import X2RecoveryEnv, EnvConfig, EnvExecutionError, _reward, _potential
from x2_recovery.reset import integration_state, nominal_pose
from x2_recovery.success import CALIBRATED_SETTINGS, SuccessTracker


class RealEnvTests(unittest.TestCase):
    def setUp(self):
        self.e=X2RecoveryEnv(capture_substeps=True)
        self.addCleanup(self.e.close)

    def test_reset_preserves_actual_handoff_and_history(self):
        e=self.e;data=e.data;obs,info=e.reset(seed=12);r=e.reset_evidence
        self.assertIs(data,e.data)
        np.testing.assert_array_equal(e.data.qpos,r['final_qpos'])
        np.testing.assert_array_equal(e.data.qvel,r['final_qvel'])
        self.assertEqual(e.data.time,r['episode_start_time']);self.assertGreater(max(abs(e.data.qvel)),0.)
        self.assertEqual(info['elapsed_sim_s'],0.);self.assertEqual(info['physics_steps_executed'],0)
        self.assertEqual(obs.shape,(117,));self.assertEqual(obs.dtype,np.float32)
        self.assertTrue(e.observation_space.contains(obs));np.testing.assert_array_equal(obs[75:],0.)
        e.step(np.full(31,.01,np.float32));again,info=e.reset(seed=12)
        np.testing.assert_array_equal(obs,again);self.assertEqual(e._physics_steps,0)
        np.testing.assert_allclose(e._potential,_potential(e._measurement))

    def test_ten_resets_seed_sequence_and_two_instances(self):
        e=self.e;initial=[];actual=[]
        for seed in range(10):
            obs,info=e.reset(seed=seed);initial.append(obs);actual.append(info['reset_seed'])
            e.step(np.zeros(31,np.float32))
        self.assertEqual(len(set(actual)),10)
        # Default reset perturbation is zero: seeds need not produce distinct physics.
        np.testing.assert_array_equal(initial[0],initial[-1])
        e.reset(seed=17);seed1=e.reset_seed;e.reset();seed2=e.reset_seed
        e.reset(seed=17);self.assertEqual(seed1,e.reset_seed);e.reset();self.assertEqual(seed2,e.reset_seed)
        self.assertNotEqual(seed1,seed2)
        with X2RecoveryEnv() as other:
            other.reset(seed=3);before=integration_state(other.model,other.data).copy()
            e.reset(seed=3);e.step(np.full(31,.02,np.float32))
            np.testing.assert_array_equal(before,integration_state(other.model,other.data))
            self.assertIsNot(e.model,other.model);self.assertIsNot(e.tracker,other.tracker)

    def test_validated_nonzero_perturbation_reproducible(self):
        with X2RecoveryEnv(replace(EnvConfig(),reset_perturb_rad=.005)) as e:
            a,_=e.reset(seed=100);b,_=e.reset(seed=101);again,_=e.reset(seed=100)
            self.assertFalse(np.array_equal(a,b));np.testing.assert_array_equal(a,again)

    def test_same_seed_and_action_trajectory_exact(self):
        e=self.e;rng=np.random.default_rng(4);actions=rng.uniform(-.04,.04,(6,31)).astype(np.float32)
        sequences=[]
        for _ in range(2):
            e.reset(seed=24);sequences.append([e.step(a) for a in actions])
        for a,b in zip(*sequences):
            np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1:],b[1:])

    def test_per_substep_pd_mapping_and_applied_torques(self):
        e=self.e;e.reset(seed=42);start=e.data.time
        with patch.object(e.tracker,'update',wraps=e.tracker.update) as update:
            obs,reward,t,u,info=e.step(np.zeros(31,np.float32))
        self.assertEqual(update.call_count,20);self.assertEqual(info['physics_steps_executed'],20)
        self.assertAlmostEqual(e.data.time-start,.02,places=10)
        records=e.last_substeps;self.assertEqual(len(records),20)
        self.assertGreater(np.max(np.ptp([r['tau_raw_Nm'] for r in records],axis=0)),.001)
        clipped=[];integral=0.
        for r in records:
            np.testing.assert_allclose(r['tau_raw_Nm'],e.kp*(np.array(r['target_rad'])-r['q_rad'])-e.kd*r['dq_rad_s'])
            expected=np.clip(r['tau_raw_Nm'],e.context.efforts[:,0],e.context.efforts[:,1])
            np.testing.assert_allclose(r['applied_torque_Nm'],expected,atol=1e-10)
            clipped.extend(np.array(r['tau_raw_Nm'])!=expected)
            integral+=np.mean((expected/e.effort_magnitude)**2)*e.physics_dt
        self.assertAlmostEqual(info['torque_saturation_fraction'],np.mean(clipped))
        self.assertAlmostEqual(info['reward_terms_raw']['torque_cost'],integral)
        self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))
        self.assertFalse(t or u)

    def test_full_target_range_and_candidate_pose_coverage(self):
        e=self.e
        for a,q in [(np.zeros(31),e.q_ref),(np.ones(31),e.q_max),(-np.ones(31),e.q_min)]:
            np.testing.assert_allclose(e.action_targets(a)[1],q,atol=1e-15)
        reset=nominal_pose(e.loaded);q=np.array([reset[r.joint_name] for r in e.loaded.mapping])
        a=e.action_for_targets(q);np.testing.assert_allclose(e.action_targets(a)[1],q,atol=1e-7)
        for i,row in enumerate(e.loaded.mapping):
            if 'knee' in row.joint_name:q[i]=1.8
            if 'hip_pitch' in row.joint_name:q[i]=-1.2
            if 'elbow' in row.joint_name:q[i]=-1.4
        np.testing.assert_allclose(e.action_targets(e.action_for_targets(q))[1],q,atol=2e-7)
        self.assertTrue(any(r.ctrl_index!=r.dof_address for r in e.loaded.mapping))

    def test_observation_mapping_frames_and_copies(self):
        e=self.e;obs,info=e.reset(seed=4);q,dq=e.loaded.read_state(e.data)
        np.testing.assert_allclose(obs[:31],(q-(e.q_min+e.q_max)/2)/((e.q_max-e.q_min)/2),atol=1e-7)
        np.testing.assert_allclose(obs[31:62],dq/5,atol=1e-7)
        R=e.data.xmat[e.context.pelvis].reshape(3,3)
        np.testing.assert_allclose(obs[62:65],R.T@[0,0,-1],atol=1e-7)
        np.testing.assert_allclose(obs[65:68],R.T@e.data.qvel[:3],atol=1e-7)
        np.testing.assert_allclose(obs[68:71],e.data.qvel[3:6]/2,atol=1e-7)
        a=np.linspace(-.02,.02,31).astype(np.float32)
        result=e.step(a);np.testing.assert_array_equal(result[0][75:106],a)
        self.assertAlmostEqual(result[-1]['reward_terms_raw']['action_change'],np.mean(a.astype(float)**2))
        self.assertEqual(e.step(a)[-1]['reward_terms_raw']['action_change'],0.)
        before=integration_state(e.model,e.data).copy();result[0][:]=9.;result[-1]['standing_failures'].clear()
        result[-1]['warnings'][0]=999;a[:]=1.
        np.testing.assert_array_equal(before,integration_state(e.model,e.data))
        self.assertFalse(np.all(e.previous_action==1.));self.assertNotEqual(e.data.warning.number[0],999)
        info['state']['other_weight']=9.;self.assertNotEqual(e._measurement['other_weight'],9.)

    def test_timeout_mid_policy_step_and_wrapper_terminal_observation(self):
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        config=replace(EnvConfig(),episode_timeout_s=.023)
        with X2RecoveryEnv(config) as e:
            e.reset(seed=1);e.step(np.zeros(31,np.float32));obs,r,t,u,info=e.step(np.zeros(31,np.float32))
            self.assertFalse(t);self.assertTrue(u);self.assertEqual(info['physics_steps_executed'],3)
            self.assertAlmostEqual(info['elapsed_sim_s'],.023)
            self.assertEqual(info['truncation_reason'],'time_limit')
            with self.assertRaisesRegex(ValueError,'reset'):e.step(np.zeros(31))
        vec=DummyVecEnv([lambda:Monitor(X2RecoveryEnv(config))]);self.addCleanup(vec.close)
        vec.seed(1);vec.reset();vec.step(np.zeros((1,31),np.float32));_,_,done,infos=vec.step(np.zeros((1,31),np.float32))
        self.assertTrue(done[0]);self.assertTrue(infos[0]['TimeLimit.truncated'])
        np.testing.assert_array_equal(infos[0]['terminal_observation'],obs)
        self.assertAlmostEqual(infos[0]['elapsed_sim_s'],.023)

    def test_no_render_resources_and_idempotent_close(self):
        e=self.e;e.reset(seed=1);before=integration_state(e.model,e.data).copy()
        self.assertIsNone(e.render());self.assertIsNone(e._renderer);self.assertIsNone(e._human)
        np.testing.assert_array_equal(before,integration_state(e.model,e.data))
        e.close();e.close()
        with self.assertRaisesRegex(ValueError,'closed'):e.reset()
        with self.assertRaises(ValueError):e.step(np.zeros(31))

    def test_rgb_render_current_state_and_repeated_cleanup(self):
        # Run this suite in a MUJOCO_GL=osmesa process; human is tested separately.
        for _ in range(2):
            with X2RecoveryEnv(render_mode='rgb_array') as e:
                e.reset(seed=1);first=e.render();e.step(np.zeros(31,np.float32))
                state=integration_state(e.model,e.data).copy();second=e.render()
                self.assertEqual(second.shape,(360,480,3));self.assertEqual(second.dtype,np.uint8)
                self.assertGreater(second.std(),1.);self.assertFalse(np.array_equal(first,second))
                np.testing.assert_array_equal(state,integration_state(e.model,e.data))
                second[:]=0;self.assertGreater(e.render().std(),1.)
            e.close();self.assertIsNone(e._renderer)


class SyntheticEnvTests(unittest.TestCase):
    """Fault injection / artificial measurements test wiring, NEVER recovery evidence."""
    def setUp(self):
        self.e=X2RecoveryEnv();self.addCleanup(self.e.close);self.e.reset(seed=9)

    def test_invalid_action_never_advances_physics(self):
        for a in (np.zeros((31,1)),np.zeros(30),np.full(31,np.nan),np.full(31,np.inf),np.full(31,1.0000001)):
            self.e.reset(seed=9);before=integration_state(self.e.model,self.e.data).copy()
            with self.assertRaises(EnvExecutionError) as caught:self.e.step(a)
            np.testing.assert_array_equal(before,integration_state(self.e.model,self.e.data))
            json.dumps(caught.exception.evidence,allow_nan=False)
            self.assertFalse(self.e._ready);self.assertIsNotNone(self.e.tracker.invalid_reason)

    def test_faults_preserve_first_error_and_require_reset(self):
        for kind in ('warning','rollback','gap','qfrc','xfrc','model','state','nonfinite','watchdog'):
            with self.subTest(kind=kind):
                e=self.e;e.reset(seed=9)
                if kind=='qfrc':e.data.qfrc_applied[0]=1.
                if kind=='xfrc':e.data.xfrc_applied[e.context.pelvis,2]=1.
                if kind=='model':e.model.body_mass[e.context.pelvis]+=.001
                if kind=='state':e.data.qpos[0]+=.1
                if kind=='nonfinite':e.data.qvel[0]=np.nan
                def bad_step(loaded,d,deadline):
                    if kind=='warning':d.warning.number[0]=1;raise ValueError('first numeric warning')
                    if kind=='rollback':d.time-=.001;mj.mj_forward(loaded.model,d)
                    if kind=='gap':d.time+=.002;mj.mj_forward(loaded.model,d)
                    if kind=='watchdog':raise ValueError('Step wall-time deadline')
                try:
                    if kind in ('warning','rollback','gap','watchdog'):
                        with patch('x2_recovery.env.checked_step',side_effect=bad_step),self.assertRaises(EnvExecutionError) as caught:
                            e.step(np.zeros(31))
                    else:
                        with self.assertRaises(EnvExecutionError) as caught:e.step(np.zeros(31))
                    self.assertFalse(e._ready);self.assertIsNotNone(e.tracker.invalid_reason)
                    json.dumps(caught.exception.evidence,allow_nan=False)
                    if kind=='warning':self.assertIn('first numeric warning',str(caught.exception))
                    if kind=='rollback':self.assertIn('time rollback',str(caught.exception))
                    if kind=='gap':self.assertIn('missing/nonuniform physics sample',str(caught.exception))
                finally:
                    if kind=='model':e.model.body_mass[e.context.pelvis]-=.001

    def test_reset_failure_no_retry_no_fake_observation(self):
        e=self.e
        with patch('x2_recovery.env.reset_supine',side_effect=ValueError('Reset wall-time deadline')) as reset:
            with self.assertRaisesRegex(EnvExecutionError,'Reset wall-time'):e.reset(seed=4)
        self.assertEqual(reset.call_count,1);self.assertFalse(e._ready)
        with self.assertRaisesRegex(ValueError,'options'):e.reset(options={'standing':True})

    def test_actual_watchdog_budget_and_float32_overflow(self):
        # Fault injection via config, not elapsed time between calls.
        with X2RecoveryEnv(replace(EnvConfig(),step_wall_s=1e-9)) as e:
            e.reset(seed=9)
            with self.assertRaisesRegex(EnvExecutionError,'[Ww]all-time'):e.step(np.zeros(31))
        with X2RecoveryEnv(replace(EnvConfig(),reset_wall_s=1e-9)) as e:
            with self.assertRaisesRegex(EnvExecutionError,'[Ww]all-time'):e.reset(seed=9)
        self.e.data.qvel[0]=1e50
        with self.assertRaisesRegex(ValueError,'float32 observation'):self.e._observation()

    def test_world_planar_history_transforms_to_body_and_clears(self):
        e=self.e;v=copy.deepcopy(e._measurement)
        e.tracker.reference={key:np.array(v[key+'_xy'])-np.array([.01,.02]) for key in ('pelvis','left','right')}
        e.tracker.stable_since=e.data.time-.3;e._result['stable_duration_s']=.3
        R=e.data.xmat[e.context.pelvis].reshape(3,3)
        obs=e._observation();self.assertEqual(obs[107],1.);self.assertAlmostEqual(obs[106],.15)
        for i,scale in enumerate((.05,.03,.03)):
            np.testing.assert_allclose(obs[108+3*i:111+3*i],R.T@np.array([.01,.02,0.])/scale,atol=1e-7)
        e.tracker._clear_window();np.testing.assert_array_equal(e._observation()[106:],0.)
        # Known yaw: linear velocity transforms, angular qvel is ALREADY body-local.
        mj.mju_axisAngle2Quat(e.data.qpos[3:7],np.array([0.,0.,1.]),np.pi/2)
        e.data.qvel[:6]=[1,0,0,.2,.4,.6];mj.mj_forward(e.model,e.data)
        obs=e._observation();np.testing.assert_allclose(obs[65:68],[0,-1,0],atol=1e-7)
        np.testing.assert_allclose(obs[68:71],[.1,.2,.3],atol=1e-7)
        velocity=np.zeros(6)
        mj.mj_objectVelocity(e.model,e.data,mj.mjtObj.mjOBJ_BODY,e.context.pelvis,velocity,0)
        R=e.data.xmat[e.context.pelvis].reshape(3,3)
        offset=e.data.xipos[e.context.pelvis]-e.data.xpos[e.context.pelvis]
        np.testing.assert_allclose(velocity[3:],e.data.qvel[:3]+np.cross(R@e.data.qvel[3:6],offset),atol=1e-12)
        self.assertGreater(np.linalg.norm(velocity[3:]-e.data.qvel[:3]),1e-4)
        # Physical range excess remains observable, not clipped to +/-1.
        e.data.qpos[e.context.qadr[0]]=e.q_max[0]+.01
        self.assertGreater(e._observation()[0],1.)

    def test_success_midstep_priority_single_bonus_synthetic(self):
        e=self.e;original=e.tracker.update;count=0
        def synthetic_success(v):
            nonlocal count
            count+=1;r=original(v)
            if count==3:r.update(recovery_success=True,standing_held=True,timed_out=True)
            return r
        with patch.object(e.tracker,'update',side_effect=synthetic_success):
            obs,reward,t,u,info=e.step(np.zeros(31))
        self.assertTrue(t);self.assertFalse(u);self.assertTrue(info['is_success'])
        self.assertEqual(info['physics_steps_executed'],3);self.assertAlmostEqual(info['elapsed_sim_s'],.003)
        self.assertEqual(info['reward_terms']['success'],50.)
        with self.assertRaisesRegex(ValueError,'reset'):e.step(np.zeros(31))

    def test_qualified_reward_integrates_executed_substeps_synthetic(self):
        e=self.e;original=e.tracker.update;count=0
        def sample(v):
            nonlocal count
            count+=1;r=original(v)
            r.update(instant_standing_ok=True,failures=[] if count%2 else ['pelvis_drift'])
            return r
        with patch.object(e.tracker,'update',side_effect=sample):
            _,_,_,_,info=e.step(np.full(31,.02,np.float32))
        self.assertAlmostEqual(info['reward_terms_raw']['standing_hold'],.010)
        self.assertAlmostEqual(info['reward_terms_raw']['action_change'],.02**2,places=10)

    def test_drift_qualified_reward_and_standing_only_gate_synthetic(self):
        e=self.e;original=e.tracker.update
        def sample(v):
            r=original(v);r.update(instant_standing_ok=True,standing_held=True,failures=['pelvis_drift'])
            return r
        with patch.object(e.tracker,'update',side_effect=sample):
            _,_,t,u,info=e.step(np.zeros(31))
        self.assertFalse(t or u);self.assertEqual(info['reward_terms']['standing_hold'],0.)
        self.assertEqual(info['reward_terms']['success'],0.)

    def test_one_ms_interruption_seen_inside_policy_step_synthetic(self):
        from test_success import good
        e=self.e;tr=SuccessTracker(CALIBRATED_SETTINGS,.001);tr.reset(e.data.time+10.-1.99)
        # Warm up the independent synthetic window; never modify recovery provenance.
        base=tr.start
        for k in range(1991):tr.update(good(base+k*.001))
        e.tracker=tr;count=0
        def measurement(ctx,d):
            nonlocal count
            count+=1
            return good(d.time+10.,left_weight=0. if count==5 else .5)
        with patch('x2_recovery.env.measure_standing',side_effect=measurement):
            _,_,t,u,info=e.step(np.zeros(31))
        self.assertEqual(count,20);self.assertFalse(t or u)
        self.assertAlmostEqual(info['stable_duration_s'],.014,places=9)

    def test_safety_is_terminated_and_absorbing_potential(self):
        e=self.e;previous=e._potential.copy();original=e.tracker.update
        with patch.object(e,'_safety_reason',return_value='safety_abort:synthetic'):
            _,reward,t,u,info=e.step(np.zeros(31))
        self.assertTrue(t);self.assertFalse(u);self.assertEqual(info['physics_steps_executed'],1)
        self.assertAlmostEqual(info['reward_terms_raw']['height'],-previous[0])
        self.assertAlmostEqual(info['reward_terms_raw']['upright'],-previous[1])
        self.assertFalse(info['is_success'])


class RewardLogicTests(unittest.TestCase):
    def test_configuration_validation(self):
        for change in ({'decimation':0},{'decimation':1.5},{'decimation':True},{'step_wall_s':0},
                       {'episode_timeout_s':np.nan},{'shaping_gamma':1.},{'reset_perturb_rad':.006},
                       {'torque_cost_weight':.01},{'reference_angles':(('bad',np.inf),)}):
            with self.assertRaises(ValueError):replace(EnvConfig(),**change)
        with self.assertRaisesRegex(ValueError,'align'):X2RecoveryEnv(replace(EnvConfig(),episode_timeout_s=.0205))

    def test_terminal_vs_truncation_and_fixed_gamma(self):
        c=EnvConfig();p=np.array([.3,.6]);n=np.array([.5,.8])
        _,raw,_=_reward(c,p,n,True,.003,.001,.2,False)
        self.assertEqual(raw['height'],-.3);self.assertEqual(raw['upright'],-.6)
        r,raw,weighted=_reward(c,p,n,False,.003,.001,.2,False)
        self.assertAlmostEqual(raw['height'],c.shaping_gamma*.5-.3)
        self.assertAlmostEqual(raw['standing_hold'],.003);self.assertAlmostEqual(r,sum(weighted.values()))

    def test_telescoping_cycles_and_stationary_states(self):
        c=EnvConfig();gamma=c.shaping_gamma
        seq=[np.array(p) for p in [(.2,.5),(.8,.9),(.2,.5),(.8,.9),(.2,.5)]]
        rewards=[_reward(c,a,b,False,0,0,0,False)[0] for a,b in zip(seq,seq[1:])]
        phi=lambda p:3*p[0]+p[1]
        expected=-phi(seq[0])+gamma**4*phi(seq[-1])
        self.assertAlmostEqual(sum(gamma**k*r for k,r in enumerate(rewards)),expected)
        self.assertLess(sum(rewards),0.)
        self.assertLess(_reward(c,seq[0],seq[0],False,0,0,0,False)[0],0.)

    def test_immediate_vs_delayed_success_and_abort_comparison(self):
        c=EnvConfig();p=np.ones(2)
        immediate=_reward(c,p,p,True,0,0,0,True)[0]
        delay=[_reward(c,p,p,False,0,0,0,False)[0]]*50
        delayed=sum(c.shaping_gamma**k*r for k,r in enumerate(delay))+c.shaping_gamma**50*immediate
        abort=_reward(c,p,p,True,0,0,0,False)[0]
        self.assertGreater(immediate,delayed);self.assertGreater(delayed,abort)
        # Same synthetic goal state, 2 ms short of success. A 1 ms interruption
        # then a new 2 s window delays the bonus, despite collecting hold reward.
        finish=_reward(c,p,p,True,.002,0,0,True)[0]
        interrupted=[_reward(c,p,p,False,.019,0,0,False)[0]]
        interrupted += [_reward(c,p,p,False,.020,0,0,False)[0]]*99
        interrupted += [finish]
        self.assertGreater(finish,sum(c.shaping_gamma**k*r for k,r in enumerate(interrupted)))
        # With no future success and running costs, early abort CAN be less negative.
        # This is an explicit limitation, not a fabricated guarantee against hacking.
        wait=[_reward(c,p,p,False,0,.02,0,False)[0]]*50
        later_abort=sum(c.shaping_gamma**k*r for k,r in enumerate(wait))+c.shaping_gamma**50*abort
        self.assertGreater(abort,later_abort)


if __name__=='__main__':unittest.main()
