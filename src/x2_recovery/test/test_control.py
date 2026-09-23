"""Checks for target generation, not a claim of recovery success."""
import unittest
import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from x2_recovery.env import X2RecoveryEnv, EnvConfig, ControlledRecoveryEnv, ControlConfig, BILATERAL_ACTION_NAMES, controller_reference_targets, head_support_task, balance_support_task, residual_phase_gate, INDEPENDENT_LEG_ACTION_NAMES, independent_legs_action_matrix
from x2_recovery.success import CALIBRATED_SETTINGS


class ControlTests(unittest.TestCase):
    def test_independent_legs_config_and_layout_are_explicit(self):
        from dataclasses import asdict
        config=ControlConfig(version='targets-v5',mode='reference_residual',action_layout='independentlegs17',
            reference_stages=(('hold',.3,{}),),residual_joint_multipliers=(.5,)*31,
            residual_start_s=.05,residual_ramp_s=.04)
        self.assertEqual(ControlConfig.from_dict(json.loads(json.dumps(asdict(config)))),config)
        for override in ({'version':'targets-v4'},{'version':[]},{'action_layout':'bilateral11'},
            {'action_layout':'joint31'},{'residual_joint_multipliers':None},{'residual_ramp_s':0.},
            {'mode':'rate','reference_stages':()}):
            with self.subTest(override=override),self.assertRaises(ValueError):replace(config,**override)
        self.assertEqual(residual_phase_gate(config,.05),0.)
        self.assertAlmostEqual(residual_phase_gate(config,.07),.5)
        self.assertEqual(residual_phase_gate(config,.10),1.)

    def test_independent_legs_zero_prefix_and_one_sided_action(self):
        from x2_recovery.reset import integration_state
        from x2_recovery.train import controller_identity
        old_config=ControlConfig(version='targets-v4',mode='reference_residual',action_layout='bilateral11',
            reference_stages=(('hold',.3,{}),),residual_joint_multipliers=(.5,)*31,
            residual_start_s=.02,residual_ramp_s=.02)
        old=ControlledRecoveryEnv(control=old_config)
        new=ControlledRecoveryEnv(control=replace(old_config,version='targets-v5',action_layout='independentlegs17'))
        try:
            initial,_=new.reset(seed=222401);old_initial,_=old.reset(seed=222401)
            np.testing.assert_array_equal(initial,old_initial)
            names=[row.joint_name for row in new.env.loaded.mapping]
            matrix=independent_legs_action_matrix(names)
            self.assertEqual(matrix.shape,(31,17));self.assertEqual(np.linalg.matrix_rank(matrix),17)
            leg_indices=[names.index(name+'_joint') for name in INDEPENDENT_LEG_ACTION_NAMES[:12]]
            np.testing.assert_array_equal(matrix[leg_indices,:12],np.eye(12))
            np.testing.assert_array_equal(matrix[:,12:],old.policy_to_joint[:,6:])
            for _ in range(3):
                a,ar,at,au,_=old.step(np.zeros(11,np.float32))
                b,br,bt,bu,_=new.step(np.zeros(17,np.float32))
                np.testing.assert_array_equal(a,b);self.assertEqual((ar,at,au),(br,bt,bu))
                np.testing.assert_array_equal(integration_state(old.env.model,old.env.data),
                    integration_state(new.env.model,new.env.data))
            action=np.zeros(17,np.float32);action[INDEPENDENT_LEG_ACTION_NAMES.index('right_ankle_pitch')]=.5
            saved=action.copy();observation,_,_,_,info=new.step(action)
            np.testing.assert_array_equal(action,saved)
            np.testing.assert_array_equal(info['controller']['policy_action'],action)
            mapped=np.asarray(info['controller']['mapped_policy_action'])
            self.assertEqual(np.count_nonzero(mapped),1)
            self.assertEqual(mapped[names.index('right_ankle_pitch_joint')],.5)
            self.assertEqual(mapped[names.index('left_ankle_pitch_joint')],0.)
            expected_target=(new.target-(new.env.q_min+new.env.q_max)/2)/((new.env.q_max-new.env.q_min)/2)
            np.testing.assert_array_equal(observation[117:148],expected_target.astype(np.float32))
            identity=controller_identity(new)
            self.assertEqual(identity['action']['shape'],[17]);self.assertEqual(identity['observation']['shape'],[149])
            self.assertEqual(identity['controller']['policy_action_names'],list(INDEPENDENT_LEG_ACTION_NAMES))
            np.testing.assert_array_equal(identity['controller']['policy_to_joint_matrix'],matrix)
            for bad in (np.zeros(11),np.zeros(31),np.ones(17)*1.01):
                with self.assertRaises(ValueError):new.step(bad)
            again,reset_info=new.reset(seed=222401)
            np.testing.assert_array_equal(again,initial)
            self.assertEqual(reset_info['controller']['residual_gate_time_s'],0.)
            self.assertIsNone(new.last_controller)
        finally:old.close();new.close()

    def test_independent_legs_ppo_checkpoint_rejects_bilateral_space(self):
        import tempfile
        from pathlib import Path
        import torch
        from stable_baselines3 import PPO
        from x2_recovery.evaluate import compare_controller_actions
        config=ControlConfig(version='targets-v5',mode='reference_residual',action_layout='independentlegs17',
            reference_stages=(('hold',.3,{}),),residual_joint_multipliers=(.5,)*31,
            residual_start_s=.02,residual_ramp_s=.02)
        env=ControlledRecoveryEnv(control=config)
        wrong=ControlledRecoveryEnv(control=replace(config,version='targets-v4',action_layout='bilateral11'))
        try:
            observation,_=env.reset(seed=222402)
            # Initialized test network only: there is no learning or recovery claim.
            model=PPO('MlpPolicy',env,n_steps=8,batch_size=4,device='cpu',seed=222402,
                policy_kwargs={'net_arch':dict(pi=[8],vf=[8])})
            model.policy.set_training_mode(False)
            before={name:value.clone() for name,value in model.policy.state_dict().items()}
            with torch.inference_mode(),tempfile.TemporaryDirectory() as temporary:
                action,_=model.predict(observation,deterministic=True)
                self.assertEqual(action.shape,(17,));self.assertTrue(np.all(abs(action)<=1))
                path=Path(temporary)/'initialized-test-only.zip';model.save(path)
                loaded=PPO.load(path,device='cpu')
                result=compare_controller_actions(loaded,observation[None],action[None],atol=1e-7,rtol=1e-6)
                self.assertEqual(result['action_shape'],[1,17]);self.assertEqual(result['max_abs_error'],0.)
                with self.assertRaisesRegex(ValueError,'Action spaces do not match'):
                    PPO.load(path,env=wrong,device='cpu')
            self.assertEqual(model.num_timesteps,0)
            self.assertTrue(all(torch.equal(value,before[name]) for name,value in model.policy.state_dict().items()))
        finally:env.close();wrong.close()

    def test_balance_density_rejects_airborne_peaks_and_bounds_costs(self):
        state=dict(pelvis_height_m=CALIBRATED_SETTINGS.h_ref_m, upright_dot=1.,
            left_weight=.5,right_weight=.5,other_weight=0.,com_linear_m_s=0.,
            torso_angular_rad_s=0.,self_penetration_m=0.)
        terms,factors=balance_support_task(state)
        self.assertEqual(terms,dict(balance=6.,overload=-0.,self_penetration=-0.))
        self.assertEqual(factors['balance_low_speed_factor'],1.)
        self.assertEqual(balance_support_task(dict(state,left_weight=0.,right_weight=0.))[0]['balance'],0.)
        slow=balance_support_task(dict(state,com_linear_m_s=.3,torso_angular_rad_s=.75))[0]
        self.assertEqual(slow['balance'],2.)
        self.assertLess(balance_support_task(dict(state,other_weight=.1))[0]['balance'],terms['balance'])
        impact=balance_support_task(dict(state,left_weight=10.,right_weight=10.,self_penetration_m=.1))[0]
        self.assertEqual(impact['overload'],-1.5);self.assertEqual(impact['self_penetration'],-.5)
        early=balance_support_task(dict(state,upright_dot=.5,left_weight=10.,right_weight=10.,
            other_weight=5.,self_penetration_m=.1))[0]
        self.assertEqual(sum(early.values()),0.)
        for name in state:
            for bad in (float('nan'),float('inf')):
                with self.subTest(name=name,bad=bad),self.assertRaises(ValueError):
                    balance_support_task(dict(state,**{name:bad}))

    def test_balance_reference_roundtrip_preserves_existing_head_guide(self):
        from dataclasses import asdict
        old=ControlConfig(mode='reference_residual',reference_stages=(('hold',.3,{}),),
            reward_version='reference-head-support-v1',head_reference=((0.,0.),(4.,1.)),head_progress_range_m=(.2,1.2))
        new=replace(old,reward_version='reference-balance-v1')
        self.assertEqual(ControlConfig.from_dict(json.loads(json.dumps(asdict(new)))),new)
        with self.assertRaises(ValueError):replace(new,head_reference=())
        with self.assertRaises(ValueError):replace(new,head_progress_range_m=())
        with self.assertRaises(ValueError):replace(new,mode='rate',reference_stages=())
        q=np.zeros(31);reference=np.linspace(-.2,.2,31);indices=np.arange(23)
        state=dict(pelvis_height_m=.3,upright_dot=.4,left_weight=.1,right_weight=.1,
            other_weight=.8,com_linear_m_s=.3,torso_angular_rad_s=.4,self_penetration_m=.002)
        for time in (0.,1.,4.,20.):
            original,oldinfo=head_support_task(old,q,reference,indices,state,False,.5,time)
            modified,info=head_support_task(new,q,reference,indices,state,False,.5,time)
            for name in ('pose_guide','head_track','standing'):
                self.assertEqual(modified[name],original[name])
            self.assertTrue(all(info[key]==value for key,value in oldinfo.items()))
            self.assertEqual(modified['balance'],0.)
            self.assertNotIn('upright_progress',modified);self.assertNotIn('foot_transfer',modified)

    def test_balance_live_short_timeout_changes_only_reward_and_uses_actual_duration(self):
        from x2_recovery.reset import integration_state
        config=ControlConfig(mode='reference_residual',reference_stages=(('hold',.3,{}),),
            reward_version='reference-head-support-v1',head_reference=((0.,0.),(4.,1.)),head_progress_range_m=(.2,1.2))
        # A shortened timeout is only a unit-test fixture for the 5ms terminal
        # transition. Recovery experiments retain their saved 20s timeout.
        physical=EnvConfig(episode_timeout_s=.025)
        old=ControlledRecoveryEnv(X2RecoveryEnv(config=physical),config)
        new=ControlledRecoveryEnv(X2RecoveryEnv(config=physical),replace(config,reward_version='reference-balance-v1'))
        try:
            a,_=old.reset(seed=222301);b,_=new.reset(seed=222301)
            np.testing.assert_array_equal(a,b)
            for physics_steps in (20,5):
                action=np.linspace(-.1,.1,31,dtype=np.float32)
                oldobs,_,ot,ou,oldinfo=old.step(action)
                obs,reward,t,u,info=new.step(action)
                np.testing.assert_array_equal(obs,oldobs)
                np.testing.assert_array_equal(integration_state(old.env.model,old.env.data),
                    integration_state(new.env.model,new.env.data))
                self.assertEqual((t,u),(ot,ou));self.assertEqual(info['physics_steps_executed'],physics_steps)
                self.assertEqual(info['physical_reward'],oldinfo['physical_reward'])
                for name in ('pose_guide','head_track','standing','torque_cost','target_change','success','safety'):
                    self.assertEqual(info['reward_terms'][name],oldinfo['reward_terms'][name])
                density,_=balance_support_task(new.env._measurement)
                for name,value in density.items():
                    self.assertEqual(info['reward_terms'][name],physics_steps*new.env.physics_dt*value)
                self.assertEqual(info['learning_task']['head_height_m'],float(new.env.data.xpos[new.head_body,2]))
                self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))
            self.assertFalse(t);self.assertTrue(u)
            task=new.resolved_config()['learning_task']
            self.assertEqual(task['density_weights']['balance'],6.)
            self.assertEqual(len(task['pose_joint_names']),23)
            json.dumps(new.resolved_config(),allow_nan=False)
        finally:old.close();new.close()

    def test_residual_phase_config_roundtrip_validation_and_grid_timing(self):
        from dataclasses import asdict
        config=ControlConfig(version='targets-v4',mode='reference_residual',action_layout='bilateral11',
            reference_stages=(('hold',3.,{}),),residual_joint_multipliers=(1.,)*31,
            residual_start_s=2.25,residual_ramp_s=.2)
        self.assertEqual(ControlConfig.from_dict(json.loads(json.dumps(asdict(config)))),config)
        self.assertEqual(residual_phase_gate(config,2.24),0.)
        self.assertEqual(residual_phase_gate(config,2.25),0.)
        self.assertGreater(residual_phase_gate(config,2.26),0.)
        self.assertAlmostEqual(residual_phase_gate(config,2.35),.5)
        self.assertEqual(residual_phase_gate(config,2.46),1.)
        self.assertEqual(residual_phase_gate(ControlConfig(),0.),1.)
        for override in ({'version':'targets-v3'},{'action_layout':'joint31'},
            {'residual_joint_multipliers':None},{'residual_start_s':-.1},{'residual_start_s':True},
            {'residual_start_s':'2.25'},{'residual_ramp_s':0.},{'residual_ramp_s':float('inf')},
            {'residual_ramp_s':float('nan')},{'mode':'rate','reference_stages':()}):
            with self.subTest(override=override),self.assertRaises(ValueError):replace(config,**override)
        for bad in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):residual_phase_gate(config,bad)
        for version in ('targets-v1','targets-v2','targets-v3'):
            kwargs=dict(version=version,mode='reference_residual',reference_stages=(('hold',3.,{}),))
            if version!='targets-v1':kwargs['residual_joint_multipliers']=(1.,)*31
            if version=='targets-v3':kwargs['action_layout']='bilateral11'
            legacy=ControlConfig(**kwargs)
            self.assertEqual(residual_phase_gate(legacy,1.),1.)
            with self.assertRaises(ValueError):replace(legacy,residual_start_s=.1)

    def test_phase_gate_live_prefix_is_exact_zero_residual_and_reset_clears_phase(self):
        from x2_recovery.reset import integration_state
        from x2_recovery.train import controller_identity
        old_config=ControlConfig(version='targets-v3',mode='reference_residual',action_layout='bilateral11',
            reference_stages=(('hold',.3,{}),),residual_joint_multipliers=(.5,)*31)
        gated_config=replace(old_config,version='targets-v4',residual_start_s=.05,residual_ramp_s=.04)
        plain=ControlledRecoveryEnv(control=old_config);gated=ControlledRecoveryEnv(control=gated_config)
        try:
            initial,_=plain.reset(seed=222081);actual,info=gated.reset(seed=222081)
            np.testing.assert_array_equal(initial,actual)
            self.assertEqual(info['controller']['residual_gate'],0.)
            action=np.linspace(-.8,.8,11,dtype=np.float32);saved=action.copy()
            # Three real transitions exercise time0,.02,.04 while the gate is0.
            # This bounded mapping test is not a recovery attempt or demonstration.
            for _ in range(3):
                expected,re,te,ue,ie=plain.step(np.zeros(11,np.float32))
                actual,ra,ta,ua,ia=gated.step(action)
                np.testing.assert_array_equal(expected,actual)
                np.testing.assert_array_equal(integration_state(plain.env.model,plain.env.data),
                    integration_state(gated.env.model,gated.env.data))
                self.assertEqual((re,te,ue),(ra,ta,ua))
                np.testing.assert_array_equal(ia['controller']['policy_action'],saved)
                np.testing.assert_array_equal(ia['controller']['gated_residual_rad'],np.zeros(31))
                self.assertEqual(ia['controller']['residual_gate'],0.)
            pre=float(gated.env.data.time-gated.env.episode_start_time)
            reference,_=controller_reference_targets(pre+gated.env.control_dt,gated.reference,gated_config.reference_interpolation)
            expected_residual=gated.residual_scale*(gated.policy_to_joint@action)*residual_phase_gate(gated_config,pre)
            _,_,_,_,info=gated.step(action)
            self.assertGreater(info['controller']['residual_gate'],0.)
            np.testing.assert_array_equal(info['controller']['gated_residual_rad'],expected_residual)
            np.testing.assert_array_equal(info['controller']['requested_target_rad'],
                np.clip(reference+expected_residual,gated.env.q_min,gated.env.q_max))
            np.testing.assert_array_equal(action,saved)
            identity=controller_identity(gated)
            self.assertEqual(identity['action']['shape'],[11]);self.assertEqual(identity['observation']['shape'],[149])
            self.assertEqual(identity['controller']['residual_gate']['start_s'],.05)
            again,info=gated.reset(seed=222081)
            np.testing.assert_array_equal(again,initial)
            self.assertEqual(info['controller']['residual_gate_time_s'],0.)
            self.assertIsNone(gated.last_controller)
            with self.assertRaisesRegex(ValueError,'timeout'):
                ControlledRecoveryEnv(gated.env,replace(gated_config,residual_start_s=20.))
        finally:plain.close();gated.close()

    def test_head_reference_config_roundtrip_and_rejection(self):
        from dataclasses import asdict
        config=ControlConfig(mode='reference_residual',reference_stages=(('hold',.3,{}),),
            reward_version='reference-head-support-v1',head_reference=((0.,0.),(4.,1.)),head_progress_range_m=(.2,1.2))
        self.assertEqual(ControlConfig.from_dict(json.loads(json.dumps(asdict(config)))),config)
        for table in ((),((0.,0.),),[[0.,0.],[4.,1.]],((0.,0.),(0.,1.)),
                      ((0.,0.),(4.,.9)),((.1,0.),(4.,1.)),((0.,0.),(4.,1.1)),
                      ((0.,False),(4.,1.)),((0.,0.),(float('nan'),1.))):
            with self.subTest(table=table),self.assertRaises(ValueError):replace(config,head_reference=table)
        for heights in ((),(.2,.2),(-.1,1.),(.2,float('inf')),(True,2.),[.2,1.2]):
            with self.subTest(heights=heights),self.assertRaises(ValueError):replace(config,head_progress_range_m=heights)
        with self.assertRaises(ValueError):replace(config,reward_version='track-v1')
        with self.assertRaises(ValueError):replace(config,mode='rate',reference_stages=())

    def test_head_reference_reward_distinguishes_bridge_airborne_and_late_lying(self):
        config=ControlConfig(mode='reference_residual',reference_stages=(('hold',.3,{}),),
            reward_version='reference-head-support-v1',head_reference=((0.,0.),(4.,1.)),head_progress_range_m=(.2,1.2))
        q=np.zeros(31);indices=np.arange(23)
        cases=dict(bridge=(.56,-.1,.5,.5,0.,False),airborne=(1.2,1.,0.,0.,0.,False),
                   late_lying=(.2,.3,.01,.01,.98,False),standing=(1.2,1.,.5,.5,0.,True))
        measured={}
        for name,(head,upright,left,right,other,qualified) in cases.items():
            terms,diagnostics=head_support_task(config,q,q,indices,
                dict(upright_dot=upright,left_weight=left,right_weight=right,other_weight=other),qualified,head,4.)
            measured[name]=terms
            self.assertEqual(diagnostics['head_height_m'],head)
            self.assertEqual(diagnostics['reference_progress'],1.)
            self.assertEqual(terms['standing'],5.*qualified)
        self.assertEqual(measured['bridge']['upright_progress'],0.)
        self.assertEqual(measured['bridge']['foot_transfer'],0.)
        self.assertLess(sum(measured['bridge'].values()),.001)
        self.assertEqual(sum(measured['airborne'].values()),0.)
        self.assertLess(sum(measured['late_lying'].values()),1e-9)
        self.assertEqual(sum(measured['standing'].values()),13.)
        # Reward durations use executed physics time, not a fixed20ms assumption.
        self.assertEqual(sum(.005*x for x in measured['standing'].values()),.065)
        with self.assertRaises(ValueError):
            head_support_task(config,q,q,indices,dict(upright_dot=1.,left_weight=.5,right_weight=.5,other_weight=0.),False,float('nan'),4.)

    def test_head_task_live_step_records_post_state_without_changing_physics(self):
        from x2_recovery.reset import integration_state
        config=ControlConfig(mode='reference_residual',reference_stages=(('hold',.3,{}),),
            reward_version='reference-head-support-v1',head_reference=((0.,0.),(4.,1.)),head_progress_range_m=(.2,1.2))
        new=ControlledRecoveryEnv(control=config)
        old=ControlledRecoveryEnv(control=replace(config,reward_version='original',head_reference=(),head_progress_range_m=()))
        try:
            new.reset(seed=221111);old.reset(seed=221111)
            ob,reward,terminated,truncated,info=new.step(np.zeros(31,np.float32))
            old_ob,_,old_terminated,old_truncated,_=old.step(np.zeros(31,np.float32))
            np.testing.assert_array_equal(ob,old_ob)
            np.testing.assert_array_equal(integration_state(new.env.model,new.env.data),integration_state(old.env.model,old.env.data))
            self.assertEqual((terminated,truncated),(old_terminated,old_truncated))
            self.assertEqual(info['learning_task']['head_height_m'],float(new.env.data.xpos[new.head_body,2]))
            self.assertEqual(info['learning_task']['elapsed_sim_s'],info['elapsed_sim_s'])
            self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))
            self.assertEqual(info['physics_steps_executed'],20)
            task=new.resolved_config()['learning_task']
            self.assertEqual(len(task['pose_joint_names']),23)
            self.assertFalse(any('head' in name or 'wrist' in name for name in task['pose_joint_names']))
        finally:new.close();old.close()

    def test_search_replay_restores_saved_controller_timing_and_objective_without_physics(self):
        import tempfile
        from pathlib import Path
        from x2_recovery import motion_search
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.json'
            source.write_text(json.dumps(dict(control_config=dict(reference_stages=[['hold',1.,{}]],
                rate_multiplier=6.,reference_interpolation='linear-v1'),objective='load-transfer-v1')))
            with patch.object(motion_search,'trial',return_value={'scope':'synthetic'}) as trial,patch('builtins.print'):
                motion_search.main(['replay','--input',str(source),'--output',str(root/'output')])
                kwargs=trial.call_args.kwargs
                self.assertEqual(kwargs['reference_interpolation'],'linear-v1')
                self.assertEqual(kwargs['rate_multiplier'],6.)
                self.assertEqual(kwargs['objective'],'load-transfer-v1')
                self.assertEqual(kwargs['custom_stages'],[['hold',1.,{}]])

    def test_reference_interpolation_dense_frames_and_phase_boundaries(self):
        from dataclasses import asdict
        from x2_recovery.baseline import Keyframes, scripted_targets
        times=np.linspace(0,8,58)
        values=np.sin(times[:,None]*np.linspace(.1,.5,31)[None,:])
        frames=Keyframes(times,values,tuple(f'frame{i}' for i in range(57)))
        for t in np.r_[times,np.linspace(0,8,133),8.1,20.]:
            actual,phase=controller_reference_targets(float(t),frames,'linear-v1')
            expected=np.array([np.interp(t,times,values[:,j]) for j in range(31)])
            np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=0)
            if t>=8:
                self.assertEqual(phase,dict(name='hold',progress=1.,sequence_finished=True))
            elif t in times:
                self.assertEqual(phase['name'],f'frame{int(np.searchsorted(times,t,side="right")-1)}')
                self.assertEqual(phase['progress'],0.)
            legacy=scripted_targets(float(t),frames)
            default=controller_reference_targets(float(t),frames,'quintic-v1')
            np.testing.assert_array_equal(default[0],legacy[0]);self.assertEqual(default[1],legacy[1])
        for t in (-.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):controller_reference_targets(t,frames,'linear-v1')
        with self.assertRaises(ValueError):controller_reference_targets(0.,frames,'unknown')
        stages=(('pose',.2,{}),)
        for version in ('targets-v1','targets-v2','targets-v3'):
            config=ControlConfig(version=version,mode='reference_residual',reference_stages=stages,
                action_layout='bilateral11' if version=='targets-v3' else 'joint31',
                residual_joint_multipliers=None if version=='targets-v1' else (1.,)*31,
                reference_interpolation='linear-v1')
            self.assertEqual(ControlConfig.from_dict(json.loads(json.dumps(asdict(config)))),config)
        with self.assertRaises(ValueError):ControlConfig(reference_interpolation='linear-v1')
        with self.assertRaises(ValueError):ControlConfig(reference_interpolation='unknown')

    def test_default_reference_keeps_original_target_and_physics_arithmetic(self):
        from x2_recovery.baseline import build_keyframes,scripted_targets
        from x2_recovery.reset import integration_state
        stages=(('move',.3,{'left_hip_pitch_joint':-.2,'right_hip_pitch_joint':-.2}),)
        control=ControlConfig(mode='reference_residual',reference_stages=stages)
        wrapped=ControlledRecoveryEnv(control=control);native=X2RecoveryEnv()
        try:
            wrapped.reset(seed=221110);native.reset(seed=221110)
            q,_=native.loaded.read_state(native.data);target=np.clip(q,native.q_min,native.q_max)
            keyframes=build_keyframes(q,native.loaded.mapping,stages=stages)
            scale=np.minimum(native.effort_magnitude/native.kp,(native.q_max-native.q_min)*.25)*control.residual_multiplier
            for sign in (1.,-1.,1.):
                action=np.linspace(-.2,.2,31,dtype=np.float32)*sign
                reference,_=scripted_targets(native.data.time-native.episode_start_time+native.control_dt,keyframes)
                requested=np.clip(reference+scale*action,native.q_min,native.q_max)
                target+=np.clip(requested-target,-wrapped.rate*native.control_dt,wrapped.rate*native.control_dt)
                expected=native.step(native.action_for_targets(target));actual=wrapped.step(action)
                np.testing.assert_array_equal(integration_state(native.model,native.data),integration_state(wrapped.env.model,wrapped.env.data))
                np.testing.assert_array_equal(actual[0][:117],expected[0])
                self.assertEqual(actual[1:4],expected[1:4])
                np.testing.assert_array_equal(actual[4]['controller']['requested_target_rad'],requested)
            self.assertEqual(wrapped.resolved_config()['reference_sampling']['interpolation'],'quintic-v1')
        finally:wrapped.close();native.close()

    def test_bilateral_version_roundtrip_and_rejection(self):
        from dataclasses import asdict
        config=ControlConfig(version='targets-v3',mode='reference_residual',action_layout='bilateral11',
            reference_stages=(('handoff',.3,{}),),residual_joint_multipliers=(1.,)*31)
        self.assertEqual(ControlConfig.from_dict(json.loads(json.dumps(asdict(config)))),config)
        for override in ({'version':'targets-v2'},{'version':'targets-v1'},
                         {'action_layout':'joint31'},{'action_layout':'unknown'},
                         {'residual_joint_multipliers':None},{'mode':'rate','reference_stages':()}):
            with self.subTest(override=override),self.assertRaises(ValueError):replace(config,**override)
        legacy=asdict(ControlConfig());del legacy['action_layout'];del legacy['residual_joint_multipliers']
        self.assertEqual(ControlConfig.from_dict(legacy),ControlConfig())

    def test_bilateral_mapping_preserves_raw_actions_and_exposes_joint_targets(self):
        import mujoco
        from x2_recovery.baseline import scripted_targets
        from x2_recovery.train import controller_identity
        e=ControlledRecoveryEnv(control=ControlConfig(version='targets-v3',mode='reference_residual',
            action_layout='bilateral11',reference_stages=(('handoff',.3,{}),),
            residual_joint_multipliers=(.5,)*31))
        try:
            initial,_=e.reset(seed=221107);initial_target=e.target.copy()
            names=[row.joint_name for row in e.env.loaded.mapping]
            expected_matrix=np.zeros((31,11))
            for column,name in enumerate(BILATERAL_ACTION_NAMES):
                for prefix in (('',) if name=='waist_pitch' else ('left_','right_')):
                    joint=prefix+name+'_joint';index=names.index(joint)
                    mirrored=prefix=='right_' and name.endswith(('_roll','_yaw'))
                    expected_matrix[index,column]=-1 if mirrored else 1
                    jid=mujoco.mj_name2id(e.env.model,mujoco.mjtObj.mjOBJ_JOINT,joint)
                    axis=[1,0,0] if name.endswith('_roll') else [0,0,1] if name.endswith('_yaw') else [0,1,0]
                    np.testing.assert_array_equal(e.env.model.jnt_axis[jid],axis)
            np.testing.assert_array_equal(e.policy_to_joint,expected_matrix)
            self.assertEqual(e.action_space.shape,(11,));self.assertEqual(initial.shape,(149,))
            action=np.linspace(-.4,.4,11,dtype=np.float32);unchanged=action.copy()
            reference,_=scripted_targets(e.env.control_dt,e.reference)
            mapped=expected_matrix@action
            requested=np.clip(reference+e.residual_scale*mapped,e.env.q_min,e.env.q_max)
            observation,_,_,_,info=e.step(action)
            np.testing.assert_array_equal(action,unchanged)
            np.testing.assert_array_equal(info['controller']['policy_action'],action)
            np.testing.assert_array_equal(info['controller']['mapped_policy_action'],mapped)
            np.testing.assert_array_equal(info['controller']['requested_target_rad'],requested)
            for index,name in enumerate(names):
                if 'wrist' in name or 'head' in name or name in ('waist_yaw_joint','waist_roll_joint'):
                    self.assertEqual(mapped[index],0.)
                    self.assertEqual(requested[index],np.clip(reference[index],e.env.q_min[index],e.env.q_max[index]))
            np.testing.assert_allclose(observation[117:148],
                (e.target-(e.env.q_min+e.env.q_max)/2)/((e.env.q_max-e.env.q_min)/2))
            identity=controller_identity(e)
            self.assertEqual(identity['action']['shape'],[11]);self.assertEqual(identity['action']['low'],[-1.]*11)
            self.assertEqual(identity['controller']['policy_action_names'],list(BILATERAL_ACTION_NAMES))
            np.testing.assert_array_equal(identity['controller']['policy_to_joint_matrix'],expected_matrix)
            np.testing.assert_array_equal(identity['controller']['residual_joint_full_scale_rad'],
                e.residual_scale*np.sum(abs(expected_matrix),axis=1))
            for bad in (np.zeros(31),np.zeros(10),np.ones(11)*2,np.zeros(11,dtype=complex)):
                with self.assertRaises(ValueError):e.step(bad)
            again,_=e.reset(seed=221107)
            np.testing.assert_array_equal(again,initial);np.testing.assert_array_equal(e.target,initial_target)
            self.assertIsNone(e.last_controller)
        finally:e.close()

    def test_bilateral_zero_residual_preserves_reference_dynamics(self):
        from x2_recovery.reset import integration_state
        control=ControlConfig(version='targets-v2',mode='reference_residual',
            reference_stages=(('handoff',.3,{}),),residual_joint_multipliers=(.5,)*31)
        a=ControlledRecoveryEnv(control=control)
        b=ControlledRecoveryEnv(control=replace(control,version='targets-v3',action_layout='bilateral11'))
        try:
            oa,_=a.reset(seed=221108);ob,_=b.reset(seed=221108)
            np.testing.assert_array_equal(oa,ob)
            for _ in range(3):
                oa,ra,ta,ua,ia=a.step(np.zeros(31,np.float32))
                ob,rb,tb,ub,ib=b.step(np.zeros(11,np.float32))
                np.testing.assert_array_equal(oa,ob)
                np.testing.assert_array_equal(integration_state(a.env.model,a.env.data),integration_state(b.env.model,b.env.data))
                self.assertEqual((ra,ta,ua),(rb,tb,ub))
                self.assertEqual({k:v for k,v in ia.items() if k!='controller'},
                                 {k:v for k,v in ib.items() if k!='controller'})
                np.testing.assert_array_equal(ia['controller']['adopted_target_rad'],ib['controller']['adopted_target_rad'])
        finally:a.close();b.close()

    def test_bilateral_real_ppo_predict_shape_without_learning(self):
        import tempfile
        from pathlib import Path
        import torch
        from stable_baselines3 import PPO
        from x2_recovery.evaluate import compare_controller_actions
        e=ControlledRecoveryEnv(control=ControlConfig(version='targets-v3',mode='reference_residual',
            action_layout='bilateral11',reference_stages=(('handoff',.3,{}),),residual_joint_multipliers=(.5,)*31))
        try:
            observation,_=e.reset(seed=221109)
            # This initialized test model is not a trained checkpoint or recovery evidence.
            model=PPO('MlpPolicy',e,n_steps=8,batch_size=4,device='cpu',seed=221109,
                      policy_kwargs={'net_arch':dict(pi=[8],vf=[8])})
            model.policy.set_training_mode(False)
            before={key:value.clone() for key,value in model.policy.state_dict().items()}
            with torch.inference_mode():
                action,_=model.predict(observation,deterministic=True)
                self.assertEqual(action.shape,(11,));self.assertTrue(np.isfinite(action).all())
                e.step(action)
                with tempfile.TemporaryDirectory() as temporary:
                    path=Path(temporary)/'initialized-test-only.zip';model.save(path)
                    loaded=PPO.load(path,device='cpu');loaded.policy.set_training_mode(False)
                    result=compare_controller_actions(loaded,observation[None],action[None],atol=1e-7,rtol=1e-6)
                    self.assertEqual(result['action_shape'],[1,11]);self.assertEqual(result['max_abs_error'],0.)
            self.assertEqual(model.num_timesteps,0)
            self.assertTrue(all(torch.equal(value,before[key]) for key,value in model.policy.state_dict().items()))
        finally:e.close()

    def test_config_roundtrip_and_rejection(self):
        from dataclasses import asdict
        for mode in ('rate','measured_delta'):
            c=ControlConfig(mode=mode)
            self.assertEqual(ControlConfig.from_dict(asdict(c)),c)
        for kwargs in ({'mode':'unknown'},{'rate_multiplier':0},{'delta_multiplier':float('nan')},
                       {'tracking_error_ratio':-1},{'reward_version':'track-v1'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):ControlConfig(**kwargs)

    def test_jointwise_residual_config_roundtrip_and_validation(self):
        from dataclasses import asdict
        stages=(('handoff',.3,{}),)
        values=dict(version='targets-v2',mode='reference_residual',reference_stages=stages,
                    residual_joint_multipliers=tuple(i/30 for i in range(31)))
        config=ControlConfig(**values)
        restored=ControlConfig.from_dict(json.loads(json.dumps(asdict(config))))
        self.assertEqual(config,restored)
        self.assertIs(type(restored.residual_joint_multipliers),tuple)
        for bad in (None,(),(1.,)*30,(1.,)*32,[1.]*31,
                    (True,)+(1.,)*30,(-.1,)+(1.,)*30,
                    (float('nan'),)+(1.,)*30,(float('inf'),)+(1.,)*30):
            with self.subTest(value=bad),self.assertRaises(ValueError):
                ControlConfig(**dict(values,residual_joint_multipliers=bad))
        for override in ({'version':'targets-v1'},
                         {'mode':'rate','reference_stages':()}):
            with self.subTest(override=override),self.assertRaises(ValueError):
                ControlConfig(**dict(values,**override))
        with self.assertRaises(ValueError):
            ControlConfig.from_dict(dict(values,residual_joint_multipliers=1.))

    def test_jointwise_residual_mapping_is_observed_and_reset(self):
        from x2_recovery.baseline import scripted_targets
        weights=tuple(i/30 for i in range(31))
        control=ControlConfig(version='targets-v2',mode='reference_residual',
                              reference_stages=(('handoff',.3,{}),),
                              residual_multiplier=2.,residual_joint_multipliers=weights)
        e=ControlledRecoveryEnv(control=control)
        try:
            initial,_=e.reset(seed=221105)
            initial_target=e.target.copy()
            reference,_=scripted_targets(e.env.control_dt,e.reference)
            action=np.linspace(-.5,.5,31,dtype=np.float32)
            expected_scale=e.delta*2.*np.asarray(weights)
            requested=np.clip(reference+expected_scale*action,e.env.q_min,e.env.q_max)
            expected=initial_target+np.clip(requested-initial_target,-e.rate*e.env.control_dt,e.rate*e.env.control_dt)
            observation,_,_,_,info=e.step(action)
            np.testing.assert_array_equal(info['controller']['policy_action'],action)
            np.testing.assert_array_equal(info['controller']['requested_target_rad'],requested)
            np.testing.assert_array_equal(e.target,expected)
            np.testing.assert_allclose(observation[117:148],
                (expected-(e.env.q_min+e.env.q_max)/2)/((e.env.q_max-e.env.q_min)/2))
            self.assertGreater(observation[-1],0)
            self.assertEqual(requested[0],reference[0])
            resolved=e.resolved_config()
            np.testing.assert_array_equal(resolved['residual_scale_rad'],expected_scale)
            self.assertEqual(resolved['residual_joint_names'],[row.joint_name for row in e.env.loaded.mapping])
            again,_=e.reset(seed=221105)
            np.testing.assert_array_equal(again,initial)
            np.testing.assert_array_equal(e.target,initial_target)
            self.assertIsNone(e.last_controller)
        finally:e.close()

    def test_unit_joint_multipliers_preserve_legacy_reference_dynamics(self):
        from x2_recovery.baseline import scripted_targets
        from x2_recovery.reset import integration_state
        old=ControlConfig(mode='reference_residual',reference_stages=(('handoff',.3,{}),),residual_multiplier=.2)
        new=replace(old,version='targets-v2',residual_joint_multipliers=(1.,)*31)
        a=ControlledRecoveryEnv(control=old)
        b=ControlledRecoveryEnv(control=new)
        try:
            oa,_=a.reset(seed=221106);ob,_=b.reset(seed=221106)
            np.testing.assert_array_equal(oa,ob)
            for sign in (1.,-1.):
                action=np.linspace(-.2,.2,31,dtype=np.float32)*sign
                reference,_=scripted_targets(a.env.data.time-a.env.episode_start_time+a.env.control_dt,a.reference)
                legacy_requested=np.clip(reference+a.delta*old.residual_multiplier*action,a.env.q_min,a.env.q_max)
                oa,ra,ta,ua,ia=a.step(action);ob,rb,tb,ub,ib=b.step(action)
                np.testing.assert_array_equal(ia['controller']['requested_target_rad'],legacy_requested)
                np.testing.assert_array_equal(oa,ob)
                np.testing.assert_array_equal(integration_state(a.env.model,a.env.data),integration_state(b.env.model,b.env.data))
                self.assertEqual((ra,ta,ua),(rb,tb,ub))
                self.assertEqual(ia,ib)
        finally:a.close();b.close()

    def test_reset_target_and_rate_are_observed_and_bounded(self):
        e=ControlledRecoveryEnv(control=ControlConfig(mode='rate'))
        try:
            obs,_=e.reset(seed=221101);q,_=e.env.loaded.read_state(e.env.data)
            np.testing.assert_array_equal(e.target,np.clip(q,e.env.q_min,e.env.q_max))
            self.assertEqual(obs.shape,(149,));self.assertEqual(obs[-1],0)
            before=e.target.copy();obs,_,_,_,info=e.step(np.ones(31,dtype=np.float32))
            self.assertTrue(np.all(abs(e.target-before)<=e.rate*e.env.control_dt+1e-12))
            np.testing.assert_allclose(obs[117:148],(e.target-(e.env.q_min+e.env.q_max)/2)/((e.env.q_max-e.env.q_min)/2))
            self.assertGreater(obs[-1],0)
            e.reset(seed=221101);np.testing.assert_array_equal(e.target,np.clip(q,e.env.q_min,e.env.q_max))
        finally:e.close()

    def test_measured_offset_semantics_and_invalid_actions(self):
        e=ControlledRecoveryEnv(control=ControlConfig(mode='measured_delta'))
        try:
            e.reset(seed=221102);q,_=e.env.loaded.read_state(e.env.data)
            a=np.full(31,.2,dtype=np.float32);expected=np.clip(q+e.delta*a,e.env.q_min,e.env.q_max)
            _,_,_,_,info=e.step(a);np.testing.assert_array_equal(e.target,expected)
            for bad in (np.zeros(30),np.ones(31)*2,np.zeros(31,dtype=complex)):
                with self.assertRaises(ValueError):e.step(bad)
        finally:e.close()

    def test_dense_reward_preserves_base_termination_and_step_duration(self):
        e=ControlledRecoveryEnv(control=ControlConfig(mode='measured_delta',reward_version='support-v1'))
        try:
            e.reset(seed=221103);_,reward,terminated,truncated,info=e.step(np.zeros(31))
            self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))
            self.assertEqual(info['physics_steps_executed'],20)
            self.assertFalse(terminated);self.assertFalse(truncated)
            self.assertIn('physical_reward_terms',info)
        finally:e.close()

    def test_upright_v2_changes_only_elevation_on_synthetic_measurements(self):
        e=ControlledRecoveryEnv(control=ControlConfig(mode='measured_delta',reward_version='support-v1'))
        try:
            observation,reset_info=e.reset(seed=221104)
            # Synthetic reward inputs only: no modified state is executed as a trajectory.
            for upright in (-.5,.5,1.):
                e.env._measurement=dict(e.env._measurement,pelvis_height_m=.4,upright_dot=upright)
                rewards={}
                for version in ('support-v1','upright-support-v2'):
                    e.control_config=replace(e.control_config,reward_version=version)
                    info=deepcopy(reset_info);info['physics_steps_executed']=20
                    info['reward_terms']={'torque_cost':0.}
                    with patch.object(e.env,'step',return_value=(observation[:117],0.,False,False,info)):
                        _,_,_,_,returned=e.step(np.zeros(31))
                    rewards[version]=returned['reward_terms']
                old,new=rewards['support-v1'],rewards['upright-support-v2']
                self.assertAlmostEqual(old['elevation'],.02*3*.4/CALIBRATED_SETTINGS.h_ref_m)
                self.assertAlmostEqual(new['elevation'],old['elevation']*max(upright,0))
                self.assertEqual({k:v for k,v in old.items() if k!='elevation'},
                                 {k:v for k,v in new.items() if k!='elevation'})
        finally:e.close()

if __name__=='__main__':unittest.main()
