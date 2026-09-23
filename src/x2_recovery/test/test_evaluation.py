"""Synthetic evaluator bookkeeping tests, not evidence of real X2 recovery."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from x2_recovery import evaluate as ev


SEEDS = [11, 12, 13, 14, 15]


class SyntheticPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer('running_value', torch.zeros(2))
        self.eval()

    def set_training_mode(self, mode):
        self.train(mode)


class SyntheticModel:
    observation_space = gym.spaces.Box(-np.inf, np.inf, (117,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (31,), dtype=np.float32)

    def __init__(self, fail_predict=None):
        self.policy = SyntheticPolicy()
        self.calls = 0
        self.fail_predict = fail_predict
        self.action = np.linspace(-.2, .2, 31, dtype=np.float32)

    def predict(self, observation, deterministic):
        self.calls += 1
        if self.calls == self.fail_predict:
            raise RuntimeError('synthetic prediction error')
        if not deterministic or self.policy.training or torch.is_grad_enabled():
            raise RuntimeError('inference contract changed')
        return self.action, None

    def learn(self, *args, **kwargs):
        raise AssertionError('Evaluation must never train')


class SyntheticEnv:
    """Scripted measurements exercise aggregation, not standing predicates."""
    physics_dt = 1.
    observation_space = SyntheticModel.observation_space
    action_space = SyntheticModel.action_space

    def __init__(self, fail_reset=None, fail_step=None):
        self.resets = []
        self.actions = []
        self.total_steps = 0
        self.fail_reset, self.fail_step = fail_reset, fail_step
        self.closed = False
        self.loaded = SimpleNamespace(read_state=lambda data: (np.zeros(31), np.zeros(31)))
        self.context = SimpleNamespace(ctrladr=np.arange(31))

    @staticmethod
    def tracker_value(hold=0., success=False):
        return dict(invalid_reason=None, stable_duration_s=hold,
                    recovery_success=success, instant_standing_ok=hold > 0,
                    failures=[] if hold > 0 else ['synthetic_predicate'], timed_out=False)

    def reset(self, *, seed):
        self.resets.append(seed)
        if len(self.resets) == self.fail_reset:
            raise RuntimeError('synthetic reset error')
        self.local_steps = 0
        self.done = False
        self.episode_start_time = 10.
        self.data = SimpleNamespace(time=10., qpos=np.zeros(3), qvel=np.zeros(3), ctrl=np.zeros(31))
        self.reset_evidence = dict(status='PASS', final_qpos=self.data.qpos.copy(), final_qvel=self.data.qvel.copy(),
                                  episode_start_time=10., initial={'seed': seed + 100})
        self.tracker = SimpleNamespace(origin=dict(time_s=10., qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy()),
                                       result=self.tracker_value())
        # First handoff is higher than every later substep; the next episode
        # must reset its peak rather than inherit this deliberately high value.
        self._measurement = dict(time_s=10., pelvis_height_m=.9 if len(self.resets) == 1 else .1)
        info = dict(reset_seed=seed + 100, elapsed_sim_s=0., physics_steps_executed=0, is_success=False,
                    termination_reason=None, truncation_reason=None)
        return np.zeros(117, dtype=np.float32), info

    def step(self, action):
        if self.done:
            raise AssertionError('Step after terminal')
        self.total_steps += 1
        if self.total_steps == self.fail_step:
            raise RuntimeError('synthetic step error')
        self.actions.append(action)
        self.local_steps += 1
        self.done = self.local_steps == 2
        reason = ['success', 'safety_abort:joint_velocity', 'time_limit', 'time_limit', 'time_limit'][len(self.resets) - 1]
        success = self.done and reason == 'success'
        # The second control transition is shortened to two physical steps.
        heights, holds = ([.3, .8, .2], [0., 1., 0.]) if self.local_steps == 1 else ([.4, .35], [1., 2.])
        self.last_substeps = []
        for index, (height, hold) in enumerate(zip(heights, holds)):
            self.data.time += self.physics_dt
            tracker = self.tracker_value(hold, success and index == len(heights) - 1)
            measurement = dict(time_s=self.data.time, pelvis_height_m=height, tilt_deg=20.)
            self.last_substeps.append(dict(measurement=measurement, success=tracker,
                target_rad=action.tolist(), q_rad=[0.] * 31, dq_rad_s=[0.] * 31,
                tau_raw_Nm=[0.] * 31, applied_torque_Nm=[0.] * 31))
        self._measurement = measurement
        self.tracker.result = tracker
        truncated = self.done and reason == 'time_limit'
        tracker['timed_out'] = truncated
        info = dict(physics_steps_executed=len(heights), elapsed_sim_s=self.data.time - 10.,
            is_success=success, reset_seed=self.resets[-1] + 100,
            termination_reason=reason if self.done and not truncated else None,
            truncation_reason='time_limit' if truncated else None,
            stable_duration_s=tracker['stable_duration_s'], state={'pelvis_height_m': measurement['pelvis_height_m']},
            reward_terms={'synthetic': 1.}, torque_saturation_fraction=.2 if self.local_steps == 1 else .5)
        return np.full(117, self.local_steps, dtype=np.float32), 1., self.done and not truncated, truncated, info

    def close(self):
        self.closed = True


class SyntheticControlledEnv(ev.ControlledRecoveryEnv):
    """Wrapper wiring fixture; deliberately does not initialize real physics."""
    def __init__(self, env=None, control=None):
        self.env = env if env is not None else SyntheticEnv()
        self.control_config = control
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (149,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.received_actions = []

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return np.r_[observation, np.zeros(32, np.float32)], info

    def step(self, action):
        self.received_actions.append(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return np.r_[observation, np.zeros(32, np.float32)], reward, terminated, truncated, info


class SyntheticControllerModel(SyntheticModel):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (149,), dtype=np.float32)

    def __init__(self, action_size=31):
        super().__init__()
        self.action_space=gym.spaces.Box(-1.,1.,(action_size,),dtype=np.float32)
        self.action=np.linspace(-.2,.2,action_size,dtype=np.float32)

    def predict(self, observation, deterministic):
        if observation.shape[-1] != 149:
            raise RuntimeError('Controller observation was not passed intact')
        action, state = super().predict(observation, deterministic)
        return (np.tile(action, (len(observation), 1)) if observation.ndim == 2 else action), state


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        quiet = patch.object(ev, 'print', create=True)
        quiet.start()
        self.addCleanup(quiet.stop)

    def batch(self, name='batch', env=None, model=None):
        output = self.root / name
        output.mkdir()
        env, model = env or SyntheticEnv(), model or SyntheticModel()
        original = ev.policy_copy(model)
        ev.write_json(output / 'manifest.json', dict(seeds=SEEDS,
            identity={'resolved_env': {'physics_dt': env.physics_dt}}, episode_wall_seconds=60.,
            initial_policy_state_sha256=ev.policy_hash(original)))
        boundary = lambda: ev.verify_policy(model, original)
        return output, env, model, boundary

    def input_fixture(self, name):
        directory = self.root / name
        directory.mkdir()
        checkpoint = b'Synthetic test input; not an executable PPO checkpoint.'
        (directory / 'policy_final.zip').write_bytes(checkpoint)
        config = asdict(ev.TrainConfig())
        environment = {'config': ev.json_value(config['env'])}
        environment['mapping'] = [{'joint_name': 'synthetic_joint'}]
        environment.update({key: [0.] for key in ('q_ref_rad', 'q_min_rad', 'q_max_rad',
            'scale_positive_rad', 'scale_negative_rad', 'kp_Nm_rad', 'kd_Nm_s_rad', 'effort_limits_Nm')})
        identity = {'resolved_env': environment, 'mapping': ['synthetic_joint']}
        saved = dict(training=config, environment=environment, identity=identity,
                     normalization='fixed_env_scaling', extra_clipping=False, algorithm='stable_baselines3.PPO')
        ev.write_json(directory / 'resolved_config.json', saved)
        digest = hashlib.sha256(checkpoint).hexdigest()
        training = dict(checkpoint={'sha256': digest}, config_sha256=ev.sha256(directory / 'resolved_config.json'),
            run_id='synthetic-smoke', run_type='smoke', status='PASS', normalization='fixed_env_scaling', identity=identity,
            training=dict(optimizer_steps=1, sampled_transitions=512, transitions_in_completed_training_rollouts=512,
                parameter_changes={k: {'l2': .1} for k in ('actor_mean', 'critic')}, numerical_checks={'finite': True}),
            provenance=dict(model_source={'repository': ev.REPOSITORY, 'commit': ev.COMMIT},
                dependencies={n: importlib.metadata.version(n) for n in
                              ('mujoco', 'gymnasium', 'stable-baselines3', 'torch', 'numpy')}))
        ev.write_json(directory / 'manifest.json', training)
        ev.write_json(directory / 'reload_check.json', dict(compatibility=identity, status='PASS',
                      complete_episode=True, checkpoint_sha256=digest))
        np.savez(directory / 'reload_probe.npz', observations=np.zeros((1, 117), dtype=np.float32),
                 actions=np.zeros((1, 31), dtype=np.float32))
        (directory / 'progress.csv').write_text('synthetic\n1\n')
        return directory, digest

    def controller_input_fixture(self, name, *, version='targets-v2', mode='reference_residual', reward_version='original'):
        directory, digest = self.input_fixture(name)
        config = asdict(ev.TrainConfig())
        control = ev.ControlConfig(version=version, mode=mode,
            reference_stages=(('handoff', .3, {}),) if mode == 'reference_residual' else (),
            residual_joint_multipliers=(.5,)*31 if version in ('targets-v2','targets-v3','targets-v4','targets-v5') else None,
            action_layout='independentlegs17' if version=='targets-v5' else 'bilateral11' if version in ('targets-v3','targets-v4') else 'joint31',
            residual_start_s=2.25 if version in ('targets-v4','targets-v5') else 0.,residual_ramp_s=.2 if version in ('targets-v4','targets-v5') else 0.,
            reward_version=reward_version,
            head_reference=((0.,0.),(4.,1.)) if reward_version in ('reference-head-support-v1','reference-balance-v1') else (),
            head_progress_range_m=(.2,1.2) if reward_version in ('reference-head-support-v1','reference-balance-v1') else ())
        action_size=17 if version=='targets-v5' else 11 if version in ('targets-v3','targets-v4') else 31
        base = {'config': ev.json_value(config['env']), 'physics_dt': .001,
                'mapping': [{'joint_name': f'synthetic_joint_{i}'} for i in range(31)]}
        base.update({key: [0.]*31 for key in ('q_ref_rad', 'q_min_rad', 'q_max_rad',
            'scale_positive_rad', 'scale_negative_rad', 'kp_Nm_rad', 'kd_Nm_s_rad', 'effort_limits_Nm')})
        environment = dict(base=base, controller=asdict(control), target_rate_rad_s=[1.]*31,
                           measured_delta_rad=[.1]*31, residual_scale_rad=[.01]*31,
                           observation_shape=[149], action_shape=[action_size])
        identity = dict(resolved_env=base, controller=environment,
                        observation={'shape': [149]}, action={'shape': [action_size]})
        saved = dict(schema=ev.CONTROLLER_SCHEMA, training=config, controller=asdict(control),
                     environment=environment, identity=identity, normalization='fixed_env_scaling',
                     device='cpu', capture_substeps=False, use_sde=True, workers=1)
        ev.write_json(directory/'resolved_config.json', saved)
        training = ev.read_json(directory/'manifest.json')
        training.update(run_type='controller_ppo_block', status='COMPLETE', identity=identity,
                        config_sha256=ev.sha256(directory/'resolved_config.json'))
        training['training'].update(inherited_model_timesteps=0, cumulative_model_timesteps=512,
                                    adam_state={'step_max': 1})
        ev.write_json(directory/'manifest.json', training)
        validation = self.root/(name+'-reload.json')
        ev.write_json(validation, dict(status='COMPLETE', label='development_reload_validation',
            reset_status='PASS', policy_state_unchanged=True, checkpoint_sha256=digest,
            terminated=False, truncated=True, transitions=1000, physics_steps=20000,
            reason='time_limit', success=False,
            final_info={'is_success': False, 'elapsed_sim_s': 20., 'termination_reason': None, 'truncation_reason': 'time_limit'},
            sim_duration_s=20.,
            action_consistency={'atol': 1e-7, 'rtol': 1e-6, 'max_abs_error': 0.}))
        model = SyntheticControllerModel(action_size)
        np.savez(directory/'reload_probe.npz', observations=np.zeros((1,149),np.float32), actions=model.action[None])
        return directory, digest, validation

    def test_controller_inputs_require_explicit_checkpoint_and_exact_reload_identity(self):
        for version,mode in (('targets-v1','rate'),('targets-v1','reference_residual'),
                             ('targets-v2','reference_residual'),('targets-v3','reference_residual'),('targets-v4','reference_residual'),('targets-v5','reference_residual')):
            with self.subTest(version=version,mode=mode):
                directory,digest,validation=self.controller_input_fixture(version+mode,version=version,mode=mode)
                with self.assertRaisesRegex(ValueError,'explicit expected checkpoint'):
                    ev.load_inputs(directory,reload_validation=validation)
                result=ev.load_inputs(directory,expected_checkpoint_sha256=digest,reload_validation=validation)
                self.assertEqual(ev.controller_label(result[2]),'ppo_'+mode)
                self.assertIn('reload_validation.json',result[-1])
        directory,digest,validation=self.controller_input_fixture('bilateral-mismatch',version='targets-v3')
        saved=ev.read_json(directory/'resolved_config.json');manifest=ev.read_json(directory/'manifest.json')
        saved['environment']['action_shape']=[31]
        saved['identity']['controller']['action_shape']=[31]
        saved['identity']['action']['shape']=[31];manifest['identity']=deepcopy(saved['identity'])
        ev.write_json(directory/'resolved_config.json',saved)
        manifest['config_sha256']=ev.sha256(directory/'resolved_config.json')
        ev.write_json(directory/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'interface shape mismatch'):
            ev.load_inputs(directory,expected_checkpoint_sha256=digest,reload_validation=validation)
        for case in ('checkpoint','controller','incomplete_controller','model','interface','reload',
                     'short_reload','tolerance','optimizer','numerical'):
            with self.subTest(case=case):
                directory,digest,validation=self.controller_input_fixture(case)
                saved=ev.read_json(directory/'resolved_config.json')
                manifest=ev.read_json(directory/'manifest.json')
                reload=ev.read_json(validation)
                if case=='checkpoint':digest='0'*64
                elif case=='controller':saved['controller']['residual_joint_multipliers'][0]=.25
                elif case=='incomplete_controller':del saved['controller']['rate_multiplier']
                elif case=='model':manifest['provenance']['model_source']['commit']='other'
                elif case=='interface':saved['identity']['action']['shape']=[30]
                elif case=='reload':reload['policy_state_unchanged']=False
                elif case=='short_reload':
                    reload.update(transitions=100,physics_steps=2000,sim_duration_s=2.)
                    reload['final_info']['elapsed_sim_s']=2.
                elif case=='tolerance':reload['action_consistency']['atol']=1e-3
                elif case=='optimizer':manifest['training']['optimizer_steps']=0
                elif case=='numerical':manifest['training']['numerical_checks']['finite']=False
                ev.write_json(directory/'resolved_config.json',saved)
                manifest['config_sha256']=ev.sha256(directory/'resolved_config.json')
                ev.write_json(directory/'manifest.json',manifest)
                ev.write_json(validation,reload)
                with self.assertRaises(ValueError):
                    ev.load_inputs(directory,expected_checkpoint_sha256=digest,reload_validation=validation)

    def test_balance_evaluation_requires_complete_head_reference_inputs(self):
        directory,digest,validation=self.controller_input_fixture('balance-input',version='targets-v4',
            reward_version='reference-balance-v1')
        result=ev.load_inputs(directory,expected_checkpoint_sha256=digest,reload_validation=validation)
        self.assertEqual(result[2]['controller']['reward_version'],'reference-balance-v1')
        saved=ev.read_json(directory/'resolved_config.json');manifest=ev.read_json(directory/'manifest.json')
        del saved['controller']['head_reference']
        ev.write_json(directory/'resolved_config.json',saved)
        manifest['config_sha256']=ev.sha256(directory/'resolved_config.json')
        ev.write_json(directory/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'Incomplete controller configuration'):
            ev.load_inputs(directory,expected_checkpoint_sha256=digest,reload_validation=validation)

    def test_controller_prepare_is_read_only_and_freeze_copies_portable_reload(self):
        directory,digest,validation=self.controller_input_fixture('new-prepare')
        saved=ev.read_json(directory/'resolved_config.json')
        native=SyntheticEnv()
        native.loaded.asset_repo=self.root/'synthetic-model'
        model=SyntheticControllerModel()
        with patch.object(ev,'X2RecoveryEnv',return_value=native), \
             patch.object(ev,'ControlledRecoveryEnv',SyntheticControlledEnv), \
             patch.object(ev,'controller_identity',return_value=saved['identity']), \
             patch.object(ev.PPO,'load',return_value=model) as load:
            env,model,original,prepared=ev.prepare(directory,expected_checkpoint_sha256=digest,reload_validation=validation)
            self.assertEqual(native.resets,[])
            load.assert_called_once_with(directory/'policy_final.zip',device='cpu')
            self.assertEqual(prepared['consistency']['action_shape'],[1,31])
            output=self.root/'new-frozen'
            manifest=ev.freeze(output,SEEDS,prepared,model,original,['synthetic-controller-test'],60.)
            self.assertEqual(manifest['controller'],'ppo_reference_residual')
            self.assertEqual(manifest['observation_layout'][-1],[148,149,'elapsed recovery time / saved episode timeout'])
            self.assertEqual(manifest['action_mapping'][0]['policy_residual_full_scale_rad'],.01)
            portable=output/'inputs/training-run'
            self.assertEqual(ev.sha256(portable/'reload_validation.json'),ev.sha256(validation))
            ev.load_inputs(portable,expected_checkpoint_sha256=digest)
            self.assertEqual(ev.verify_policy(model,original),ev.policy_hash(original))
            env.close()
        self.assertTrue(native.closed)

    def test_controller_probe_and_checkpoint_spaces_are_dynamic_and_strict(self):
        model=SyntheticControllerModel()
        observation=np.zeros((2,149),np.float32)
        expected=np.tile(model.action,(2,1))
        with torch.inference_mode():
            result=ev.compare_controller_actions(model,observation,expected,atol=1e-7,rtol=1e-6)
            self.assertEqual(result['max_abs_error'],0.)
            for bad in (np.zeros((2,117),np.float32),np.zeros((2,149),np.float64)):
                with self.assertRaises(ValueError):
                    ev.compare_controller_actions(model,bad,expected,atol=1e-7,rtol=1e-6)
        env=SyntheticControlledEnv()
        identity={'controller':{'reference':'fixed'}}
        with patch.object(ev,'controller_identity',return_value=identity):
            saved={'schema':ev.CONTROLLER_SCHEMA,'identity':identity}
            ev.validate_environment(env,saved,model)
            model.action_space=gym.spaces.Box(-1.,1.,(30,),dtype=np.float32)
            with self.assertRaisesRegex(ValueError,'space mismatch'):
                ev.validate_environment(env,saved,model)
        env.close()

    def test_bilateral_probe_batch_and_saved_action_metrics(self):
        model=SyntheticControllerModel(11)
        observations=np.zeros((2,149),np.float32)
        with torch.inference_mode():
            result=ev.compare_controller_actions(model,observations,np.tile(model.action,(2,1)),atol=1e-7,rtol=1e-6)
            self.assertEqual(result['action_shape'],[2,11])
            with self.assertRaisesRegex(ValueError,'action shape mismatch'):
                ev.compare_controller_actions(model,observations,np.zeros((2,31),np.float32),atol=1e-7,rtol=1e-6)
        output,native,_,_=self.batch('bilateral-wrapped')
        manifest=ev.read_json(output/'manifest.json');manifest['controller']='ppo_reference_residual'
        ev.write_json(output/'manifest.json',manifest)
        env=SyntheticControlledEnv(native);env.action_space=model.action_space
        original=ev.policy_copy(model)
        summary=ev.execute_batch(output,env,model,lambda:ev.verify_policy(model,original))
        self.assertEqual(summary['status'],'COMPLETE')
        self.assertEqual(native.resets,SEEDS)
        self.assertTrue(all(action is model.action for action in env.received_actions))
        for diagnostics in summary['diagnostics']:
            self.assertAlmostEqual(diagnostics['mean_squared_action_change_per_transition'],
                                   float(np.mean(model.action.astype(float)**2))/2)
        self.assertEqual(ev.verify_saved(output)['valid_completed'],5)
        self.assertTrue(native.closed)

    def test_independent_legs_probe_batch_and_saved_action_metrics(self):
        model=SyntheticControllerModel(17)
        observations=np.zeros((2,149),np.float32)
        with torch.inference_mode():
            result=ev.compare_controller_actions(model,observations,np.tile(model.action,(2,1)),atol=1e-7,rtol=1e-6)
            self.assertEqual(result['action_shape'],[2,17])
            with self.assertRaisesRegex(ValueError,'action shape mismatch'):
                ev.compare_controller_actions(model,observations,np.zeros((2,31),np.float32),atol=1e-7,rtol=1e-6)
        output,native,_,_=self.batch('independent-legs-wrapped')
        manifest=ev.read_json(output/'manifest.json');manifest['controller']='ppo_reference_residual'
        ev.write_json(output/'manifest.json',manifest)
        env=SyntheticControlledEnv(native);env.action_space=model.action_space
        original=ev.policy_copy(model)
        summary=ev.execute_batch(output,env,model,lambda:ev.verify_policy(model,original))
        self.assertEqual(summary['status'],'COMPLETE')
        self.assertEqual(native.resets,SEEDS)
        self.assertTrue(all(action is model.action for action in env.received_actions))
        for diagnostics in summary['diagnostics']:
            self.assertAlmostEqual(diagnostics['mean_squared_action_change_per_transition'],
                                   float(np.mean(model.action.astype(float)**2))/2)
        self.assertEqual(ev.verify_saved(output)['valid_completed'],5)
        self.assertTrue(native.closed)

    def test_controller_batch_uses_native_measurements_and_retains_policy_actions(self):
        output,native,_,_=self.batch('wrapped')
        manifest=ev.read_json(output/'manifest.json');manifest['controller']='ppo_reference_residual'
        ev.write_json(output/'manifest.json',manifest)
        env=SyntheticControlledEnv(native);model=SyntheticControllerModel();original=ev.policy_copy(model)
        summary=ev.execute_batch(output,env,model,lambda:ev.verify_policy(model,original))
        self.assertEqual(summary['status'],'COMPLETE')
        self.assertEqual(native.resets,SEEDS)
        self.assertEqual(len(env.received_actions),10)
        self.assertTrue(all(action is model.action for action in env.received_actions))
        self.assertTrue(all(row['controller']=='ppo_reference_residual' for row in summary['episodes']))
        self.assertEqual(ev.verify_saved(output)['valid_completed'],5)
        self.assertTrue(native.closed)

    def test_seed_and_path_validation_precedes_preparation(self):
        self.assertEqual(ev.validate_seeds([15, 11, 14, 13, 12]), [15, 11, 14, 13, 12])
        for seeds in ([], SEEDS[:4], SEEDS + [16], [1, 1, 2, 3, 4], [True, 2, 3, 4, 5],
                      [1., 2, 3, 4, 5], ['1', 2, 3, 4, 5], [-1, 2, 3, 4, 5], [2**32, 2, 3, 4, 5]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                ev.validate_seeds(seeds)
        with self.assertRaisesRegex(ValueError, 'Missing input'):
            ev.load_inputs(self.root / 'missing')
        with patch.object(ev, 'prepare') as prepare:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                ev.run(self.root / 'missing', SEEDS, self.root)
            for budget in (0, -1, float('nan'), float('inf'), True, '60'):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    ev.run(self.root / 'missing', SEEDS, self.root / 'new', episode_wall_seconds=budget)
            prepare.assert_not_called()

    def test_load_inputs_reconstructs_tuples_and_rejects_identity_changes(self):
        directory, digest = self.input_fixture('valid')
        with patch.object(ev, 'CHECKPOINT_SHA256', digest):
            _, config, *_ = ev.load_inputs(directory)
        self.assertIsInstance(config.env.reference_angles, tuple)
        self.assertIsInstance(config.env.reference_angles[0], tuple)
        cases = ('checkpoint', 'config_hash', 'identity', 'model', 'normalization', 'incomplete_config', 'optimizer', 'actor')
        for case in cases:
            with self.subTest(case=case):
                directory, digest = self.input_fixture(case)
                manifest = ev.read_json(directory / 'manifest.json')
                saved = ev.read_json(directory / 'resolved_config.json')
                if case == 'checkpoint':
                    (directory / 'policy_final.zip').write_bytes(b'tampered')
                elif case == 'config_hash':
                    (directory / 'resolved_config.json').write_text('{}')
                elif case == 'identity':
                    manifest['identity']['mapping'] = ['different_joint_same_dimension']
                elif case == 'model':
                    manifest['provenance']['model_source']['commit'] = 'wrong'
                elif case == 'normalization':
                    saved['normalization'] = 'VecNormalize'
                elif case == 'incomplete_config':
                    del saved['training']['env']['decimation']
                elif case == 'optimizer':
                    manifest['training']['optimizer_steps'] = 0
                elif case == 'actor':
                    manifest['training']['parameter_changes']['actor_mean']['l2'] = 0.
                if case in ('normalization', 'incomplete_config'):
                    ev.write_json(directory / 'resolved_config.json', saved)
                    manifest['config_sha256'] = ev.sha256(directory / 'resolved_config.json')
                ev.write_json(directory / 'manifest.json', manifest)
                with patch.object(ev, 'CHECKPOINT_SHA256', digest), self.assertRaises(ValueError):
                    ev.load_inputs(directory)

    def test_environment_identity_checks_semantics_and_bounds(self):
        env, model = SyntheticEnv(), SyntheticModel()
        identity = {'joint_order': ['a', 'b'], 'scales': [1., 2.], 'model_fingerprint': 'original'}
        with patch.object(ev, 'env_identity', return_value=identity):
            self.assertEqual(ev.validate_environment(env, {'identity': identity}, model), identity)
            for key, value in [('joint_order', ['b', 'a']), ('scales', [2., 1.]), ('model_fingerprint', 'other')]:
                altered = dict(identity, **{key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    ev.validate_environment(env, {'identity': altered}, model)
            model.action_space = gym.spaces.Box(-.5, .5, (31,), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, 'space mismatch'):
                ev.validate_environment(env, {'identity': identity}, model)

    def test_policy_parameters_and_buffers_remain_read_only(self):
        for name in ('weight', 'running_value'):
            with self.subTest(name=name):
                model = SyntheticModel()
                original = ev.policy_copy(model)
                self.assertEqual(ev.verify_policy(model, original), ev.policy_hash(original))
                with torch.no_grad():
                    getattr(model.policy, name).add_(1.)
                with self.assertRaisesRegex(ValueError, 'state changed'):
                    ev.verify_policy(model, original)
        with self.assertRaises(ValueError):
            ev.policy_hash({'invalid': torch.tensor(float('nan'))})

    def test_freeze_creates_read_only_manifest_and_verified_input_copies(self):
        directory, digest = self.input_fixture('freeze-input')
        with patch.object(ev, 'CHECKPOINT_SHA256', digest):
            directory, config, saved, training, reload, hashes = ev.load_inputs(directory)
        model = SyntheticModel()
        prepared = dict(directory=directory, config=config, saved=saved, training=training,
                        hashes=hashes, identity=saved['identity'], consistency={'synthetic': True},
                        model_asset_directory='/synthetic/model-not-loaded')
        output = self.root / 'frozen'
        manifest = ev.freeze(output, SEEDS, prepared, model, ev.policy_copy(model), ['synthetic-test'], 60.)
        self.assertEqual(manifest['seeds'], SEEDS)
        self.assertEqual((output / 'manifest.json').stat().st_mode & 0o777, 0o444)
        self.assertEqual(ev.read_json(output / 'manifest.json'), ev.json_value(manifest))
        for name, digest in hashes.items():
            self.assertEqual(ev.sha256(output / 'inputs' / 'training-run' / name), digest)
        for name, digest in manifest['source_sha256'].items():
            self.assertEqual(ev.sha256(output / manifest['source_snapshot'] / name), digest)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            ev.freeze(output, SEEDS, prepared, model, ev.policy_copy(model), ['synthetic-test'], 60.)

    def test_five_explicit_boundaries_substep_metrics_and_direct_actions(self):
        output, env, model, boundary = self.batch()
        summary = ev.execute_batch(output, env, model, boundary)
        self.assertEqual(summary['status'], 'COMPLETE')
        self.assertEqual(summary['result'], '1/5')
        self.assertEqual(env.resets, SEEDS)
        self.assertEqual(len(env.actions), 10)
        self.assertTrue(all(action is model.action for action in env.actions))
        self.assertTrue(env.closed)
        self.assertEqual([r['max_pelvis_height_m'] for r in summary['episodes']], [.9, .8, .8, .8, .8])
        self.assertEqual([r['max_stable_hold_s'] for r in summary['episodes']], [2.] * 5)
        self.assertEqual([r['sim_duration_s'] for r in summary['episodes']], [5.] * 5)
        self.assertEqual([(r['terminated'], r['truncated']) for r in summary['episodes']],
                         [(1, 0), (1, 0), (0, 1), (0, 1), (0, 1)])
        self.assertEqual([r['termination_reason'] for r in summary['episodes']][:3],
                         ['success', 'safety_abort:joint_velocity', 'time_limit'])
        for diagnostics in summary['diagnostics']:
            self.assertEqual(diagnostics['physics_steps'], 5)
            self.assertEqual(diagnostics['hold_interruptions'], 1)
            self.assertAlmostEqual(diagnostics['torque_saturation_physics_weighted'], (.2 * 3 + .5 * 2) / 5)
        verified = ev.verify_saved(output)
        self.assertEqual(verified['episodes'], summary['episodes'])
        self.assertEqual(ev.read_json(output / 'summary.json')['episodes'], verified['episodes'])

    def test_partial_transition_is_not_a_completed_episode(self):
        env, model = SyntheticEnv(), SyntheticModel()
        obs, info = env.reset(seed=11)
        reset = ev.reset_record(env, obs, info, 1, 11)
        metrics = ev.EpisodeMetrics(reset, env.physics_dt)
        obs, reward, terminated, truncated, info = env.step(model.action)
        transition = ev.transition_record(env, obs, model.action, reward, terminated, truncated, info, 1, 11, 1)
        self.assertIsNone(metrics.add(transition))
        self.assertEqual(metrics.physics_steps, 3)
        self.assertEqual(metrics.max_hold, 1.)
        bad = deepcopy(transition)
        bad['info']['elapsed_sim_s'] = 999.
        with self.assertRaisesRegex(ValueError, 'duration'):
            ev.EpisodeMetrics(reset, env.physics_dt).add(bad)
        bad = deepcopy(transition)
        bad['info']['is_success'] = True
        with self.assertRaisesRegex(ValueError, 'Nonterminal'):
            ev.EpisodeMetrics(reset, env.physics_dt).add(bad)

    def test_reset_predict_step_and_recording_errors_stop_without_fabrication(self):
        for failure in ('reset', 'predict', 'step', 'record'):
            with self.subTest(failure=failure):
                env = SyntheticEnv(fail_reset=2 if failure == 'reset' else None,
                                   fail_step=3 if failure == 'step' else None)
                model = SyntheticModel(fail_predict=3 if failure == 'predict' else None)
                output, env, model, boundary = self.batch(failure, env, model)
                original_append = ev.append_record
                def append(stream, record):
                    if failure == 'record' and record['kind'] == 'transition' and record['episode'] == 2:
                        raise OSError('synthetic recording error')
                    original_append(stream, record)
                with patch.object(ev, 'append_record', side_effect=append), self.assertRaises((RuntimeError, OSError)):
                    ev.execute_batch(output, env, model, boundary)
                summary = ev.read_json(output / 'summary.json')
                self.assertEqual(summary['status'], 'ERROR')
                self.assertEqual(summary['planned'], 5)
                self.assertEqual(summary['attempted'], 2)
                self.assertEqual(summary['valid_completed'], 1)
                self.assertNotIn('result', summary)
                self.assertTrue(env.closed)
                self.assertEqual(env.resets, SEEDS[:2])
                with (output / 'episodes.csv').open(newline='') as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 1)
                self.assertIn('"kind":"error"', (output / 'trajectory.jsonl').read_text())
                self.assertEqual(ev.verify_saved(output, require_complete=False)['valid_completed'], 1)
                with self.assertRaises(ValueError):
                    ev.verify_saved(output)

    def test_final_serialization_failure_closes_environment_and_cannot_return_complete(self):
        output, env, model, boundary = self.batch()
        with patch.object(ev, 'write_json', side_effect=OSError('synthetic summary write error')):
            with self.assertRaisesRegex(OSError, 'summary write'):
                ev.execute_batch(output, env, model, boundary)
        self.assertTrue(env.closed)
        self.assertFalse((output / 'summary.json').exists())

    def test_saved_csv_and_extra_reset_cannot_pass_verification(self):
        for change in ('csv', 'extra_reset', 'terminal'):
            with self.subTest(change=change):
                output, env, model, boundary = self.batch(change)
                ev.execute_batch(output, env, model, boundary)
                path = output / 'trajectory.jsonl'
                records = [ev.strict_json(line) for line in path.read_text().splitlines()]
                if change == 'csv':
                    csv_path = output / 'episodes.csv'
                    csv_path.write_text(csv_path.read_text().replace('0.9', '9.9', 1))
                elif change == 'extra_reset':
                    with path.open('a') as stream:
                        ev.append_record(stream, records[0])
                else:
                    terminal = next(r for r in records if r['kind'] == 'terminal')
                    terminal['row']['success'] = 0
                    path.write_text('\n'.join(ev.encoded(r) for r in records) + '\n')
                with self.assertRaises(ValueError):
                    ev.verify_saved(output)

    def test_saved_summary_and_input_snapshot_tampering_are_rejected(self):
        for change in ('summary', 'input', 'source'):
            with self.subTest(change=change):
                output, env, model, boundary = self.batch(change)
                for name in ('inputs', 'source'):
                    (output / name).mkdir()
                    (output / name / 'synthetic.txt').write_text('synthetic fixture')
                manifest = ev.read_json(output / 'manifest.json')
                manifest.update(portable_training_directory='inputs', source_snapshot='source',
                    input_sha256={'synthetic.txt': ev.sha256(output / 'inputs' / 'synthetic.txt')},
                    source_sha256={'synthetic.txt': ev.sha256(output / 'source' / 'synthetic.txt')})
                ev.write_json(output / 'manifest.json', manifest)
                ev.execute_batch(output, env, model, boundary)
                self.assertEqual(ev.audit_saved(output)['status'], 'PASS')
                if change == 'summary':
                    summary = ev.read_json(output / 'summary.json')
                    summary['successes_observed'] = 5
                    ev.write_json(output / 'summary.json', summary)
                else:
                    (output / ('inputs' if change == 'input' else 'source') / 'synthetic.txt').write_text('tampered')
                with self.assertRaises(ValueError):
                    ev.audit_saved(output)

    def test_invalid_reset_and_watchdog_preserve_incomplete_attempt(self):
        for failure in ('reset_evidence', 'watchdog'):
            with self.subTest(failure=failure):
                output, env, model, boundary = self.batch(failure)
                if failure == 'watchdog':
                    manifest = ev.read_json(output / 'manifest.json')
                    manifest['episode_wall_seconds'] = 1.
                    ev.write_json(output / 'manifest.json', manifest)
                original_reset = env.reset
                def reset(*, seed):
                    observation, info = original_reset(seed=seed)
                    if failure == 'reset_evidence':
                        env.reset_evidence['status'] = 'FAIL'
                    return observation, info
                env.reset = reset
                clock = patch.object(ev.time, 'perf_counter', side_effect=[0., 0., 2., 3.]) if failure == 'watchdog' else nullcontext()
                with clock, self.assertRaises(ValueError):
                    ev.execute_batch(output, env, model, boundary)
                summary = ev.read_json(output / 'summary.json')
                self.assertEqual(summary['status'], 'ERROR')
                self.assertEqual(summary['attempted'], 1)
                self.assertEqual(summary['valid_completed'], 0)
                self.assertEqual(summary['successes_observed'], 0)
                self.assertNotIn('result', summary)
                self.assertEqual(model.calls, 0)
                self.assertTrue(env.closed)

    def test_strict_json_rejects_nonfinite_values(self):
        for value in ('NaN', 'Infinity', '-Infinity', '1e999'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ev.strict_json('{"value":' + value + '}')
        with self.assertRaises(ValueError):
            ev.encoded({'value': float('nan')})


if __name__ == '__main__':
    unittest.main()
