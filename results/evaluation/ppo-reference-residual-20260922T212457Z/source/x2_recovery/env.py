"""Native MuJoCo Gymnasium recovery environment; no ROS, audit I/O or policy.

State and contact observations are simulator-accessible, not a hardware sensing
claim. Only reset_supine initializes physics; step uses bounded joint torques.
"""
from dataclasses import asdict, dataclass, replace
import copy
import time
import gymnasium as gym
import mujoco as mj
import numpy as np
from .model import load_effective_model, require
from .reset import ResetSettings, reset_supine, checked_step, model_signature, integration_state, check_callbacks
from .success import StandingContext, SuccessTracker, CALIBRATED_SETTINGS, measure_standing


# Keep this list identical to reset.model_signature; the regression test checks
# that its bytewise boundary checks cover every field used by the provenance hash.
_MODEL_SIGNATURE_ARRAYS = (
    'body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
    'body_gravcomp','jnt_range','jnt_margin','jnt_solref','jnt_solimp',
    'jnt_limited','jnt_actfrclimited','jnt_actfrcrange','jnt_axis',
    'dof_damping','dof_armature','dof_frictionloss','geom_type','geom_size',
    'geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction',
    'geom_solref','geom_solimp','geom_margin','geom_gap','geom_priority',
    'mesh_vert','actuator_gear','actuator_gainprm','actuator_biasprm',
    'actuator_ctrlrange','actuator_ctrllimited','eq_active0')
_MODEL_IDENTITY_ARRAYS = (
    'jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','body_parentid','body_rootid',
    'geom_bodyid','actuator_trnid','actuator_trntype','actuator_dyntype','actuator_gaintype',
    'actuator_ctrladr','actuator_outadr','actuator_forcelimited','actuator_forcerange')


@dataclass(frozen=True)
class EnvConfig:
    decimation: int = 20
    episode_timeout_s: float = 20.
    step_wall_s: float = 5.
    reset_wall_s: float = 45.
    reset_perturb_rad: float = 0.  # Only the validated supine-reset [0, .005] range.
    joint_velocity_scale_rad_s: float = 5.
    base_linear_scale_m_s: float = 1.
    base_angular_scale_rad_s: float = 2.
    shaping_gamma: float = .999
    height_weight: float = 3.
    upright_weight: float = 1.
    standing_hold_weight: float = 1.
    torque_cost_weight: float = -.02
    action_change_weight: float = -.002
    success_bonus: float = 50.
    # Task safety guards for gross soft-constraint deformation, not standing
    # tolerances or a claim that ordinary lying/contact states are unrecoverable.
    safety_floor_m: float = .03
    safety_self_m: float = .015
    safety_joint_limit_rad: float = .05
    safety_joint_speed_rad_s: float = 30.
    # Relative to the source neutral frame; every unspecified joint is zero.
    reference_angles: tuple = (('hip_pitch', -.05), ('knee', .10), ('ankle_pitch', -.05),
                               ('shoulder_pitch', -.15), ('elbow', -.30))
    shoulder_roll_rad: float = .15
    # group, Kp [N m/rad], Kd [N m s/rad]; candidate based on calibrated standing-fixture PD.
    pd_gains: tuple = (('hip', 600., 16.970562748477143), ('knee', 600., 16.970562748477143),
                      ('ankle', 300., 11.313708498984761), ('waist', 300., 11.313708498984761),
                      ('shoulder', 100., 4.242640687119286), ('elbow', 100., 4.242640687119286),
                      ('head', 10., .4242640687119285), ('wrist', 10., .4242640687119285))

    def __post_init__(self):
        require(type(self.decimation) is int and self.decimation > 0, 'decimation must be a positive integer')
        for key, value in asdict(self).items():
            if key not in ('reference_angles', 'pd_gains', 'decimation'):
                require(isinstance(value, (int, float)) and np.isfinite(value), 'Nonfinite config: '+key)
        for name in ('episode_timeout_s','step_wall_s','reset_wall_s','joint_velocity_scale_rad_s',
                     'base_linear_scale_m_s','base_angular_scale_rad_s','safety_floor_m','safety_self_m',
                     'safety_joint_limit_rad','safety_joint_speed_rad_s'):
            require(getattr(self, name)>0, 'Config must be positive: '+name)
        require(0<=self.reset_perturb_rad<=.005 and 0<self.shaping_gamma<1, 'Invalid reset perturbation/gamma')
        require(self.height_weight>=0 and self.upright_weight>=0 and self.standing_hold_weight>=0
                and self.torque_cost_weight<=0 and self.action_change_weight<=0 and self.success_bonus>0,
                'Invalid reward signs')
        require(isinstance(self.reference_angles,tuple) and isinstance(self.pd_gains,tuple)
                and all(isinstance(r,tuple) for r in self.reference_angles+self.pd_gains), 'Use immutable config tuples')
        require(len(dict(self.reference_angles))==len(self.reference_angles)
                and all(np.isfinite(v) for _,v in self.reference_angles), 'Invalid reference angles')
        require(len({r[0] for r in self.pd_gains})==len(self.pd_gains)
                and all(np.isfinite([p,d]).all() and p>0 and d>0 for _,p,d in self.pd_gains), 'Invalid PD gains')


class EnvExecutionError(RuntimeError):
    """Execution failed; evidence is detached and strict-JSON compatible."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _diagnostic(value):
    if isinstance(value, np.ndarray): return _diagnostic(value.tolist())
    if isinstance(value, np.generic): return _diagnostic(value.item())
    if isinstance(value, dict): return {str(k):_diagnostic(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_diagnostic(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return {'invalid_numeric':repr(value)}
    return value if value is None or isinstance(value,(str,bool,int,float)) else repr(value)


def _potential(v):
    return np.array([np.clip(v['pelvis_height_m']/CALIBRATED_SETTINGS.h_ref_m,0.,1.),
                     (1.+np.clip(v['upright_dot'],-1.,1.))/2.])


def _reward(config, previous, current, terminated, qualified_s, torque_integral, action_change, success):
    # One gamma per policy transition, including a shortened final transition.
    following = np.zeros(2) if terminated else current
    shaping = config.shaping_gamma*following-previous
    raw = dict(height=float(shaping[0]), upright=float(shaping[1]), standing_hold=float(qualified_s),
               torque_cost=float(torque_integral), action_change=float(action_change), success=float(success))
    weights = dict(height=config.height_weight, upright=config.upright_weight,
                   standing_hold=config.standing_hold_weight, torque_cost=config.torque_cost_weight,
                   action_change=config.action_change_weight, success=config.success_bonus)
    weighted={key:raw[key]*weights[key] for key in raw}
    require(np.isfinite(list(weighted.values())).all(), 'Nonfinite reward')
    return float(sum(weighted.values())), raw, weighted


class X2RecoveryEnv(gym.Env):
    metadata = {'render_modes':['rgb_array','human'], 'render_fps':50}

    def __init__(self, config=None, render_mode=None, asset_repo=None, *, capture_substeps=False):
        super().__init__()
        self.config = config if config is not None else EnvConfig()
        require(isinstance(self.config,EnvConfig), 'Expected EnvConfig')
        require(render_mode is None or render_mode in self.metadata['render_modes'], 'Unsupported render_mode')
        self.render_mode=render_mode; self.capture_substeps=bool(capture_substeps)
        self.loaded=load_effective_model(asset_repo);self.model=self.loaded.model
        self.data=mj.MjData(self.model);self.context=StandingContext(self.loaded)
        require(len(self.loaded.mapping)==31, 'Expected 31 controlled joints')
        self.physics_dt=float(self.model.opt.timestep)
        self.control_dt=self.physics_dt*self.config.decimation
        n=self.config.episode_timeout_s/self.physics_dt
        require(abs(n-round(n))<1e-8, 'Timeout must align to physics timestep')
        self.timeout_steps=int(round(n))
        self.metadata={**type(self).metadata,'render_fps':max(1,round(1/self.control_dt))}
        self.tracker=SuccessTracker(CALIBRATED_SETTINGS,self.physics_dt,self.config.episode_timeout_s)
        self.q_min=self.context.ranges[:,0].copy();self.q_max=self.context.ranges[:,1].copy()
        self.q_ref=np.zeros(31);self.kp=np.empty(31);self.kd=np.empty(31)
        pose={r.joint_name:0. for r in self.loaded.mapping}
        for side, sign in (('left',1),('right',-1)):
            for joint,value in self.config.reference_angles:
                require(side+'_'+joint+'_joint' in pose, 'Unknown reference joint '+joint)
                pose[side+'_'+joint+'_joint']=value
            pose[side+'_shoulder_roll_joint']=sign*self.config.shoulder_roll_rad
        for i,row in enumerate(self.loaded.mapping):
            self.q_ref[i]=pose[row.joint_name]
            group=row.joint_name.removeprefix('left_').removeprefix('right_').split('_')[0]
            candidates=[(p,d) for name,p,d in self.config.pd_gains if name==group]
            require(len(candidates)==1,'Missing PD group '+group)
            self.kp[i],self.kd[i]=candidates[0]
        require(np.all(self.q_ref>=self.q_min) and np.all(self.q_ref<=self.q_max),'Reference outside limits')
        self.scale_positive=self.q_max-self.q_ref;self.scale_negative=self.q_ref-self.q_min
        self.effort_magnitude=np.max(abs(self.context.efforts),axis=1)
        require(np.isfinite(self.effort_magnitude).all() and np.all(self.effort_magnitude>0),'Invalid effort denominator')
        self.action_space=gym.spaces.Box(-1.,1.,(31,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(117,),dtype=np.float32)
        self._mapping=self.loaded.mapping;self._signature=self._model_identity()
        # Check the same bytes at the same boundaries without rehashing the full
        # collision mesh twice per transition. uint8 comparison also detects
        # signed-zero changes that ordinary floating-point equality would miss.
        self._model_snapshot=[]
        for key in _MODEL_SIGNATURE_ARRAYS+_MODEL_IDENTITY_ARRAYS:
            original=getattr(self.model,key)
            snapshot=original.view(np.uint8).copy();snapshot.setflags(write=False)
            self._model_snapshot.append((key,original.dtype,original.shape,snapshot))
        self._model_snapshot=tuple(self._model_snapshot)
        self._model_options=str(self.model.opt)
        self._model_counts=(self.model.neq,self.model.nmocap)
        self._renderer=None;self._human=None;self._closed=False;self._ready=False
        self._measurement=None;self._result=None;self._expected_state=None
        self._seed=None;self.reset_seed=None;self.episode_start_time=None
        self.previous_action=np.zeros(31,dtype=np.float32);self._target=self.q_ref.copy()
        self._physics_steps=0;self.last_substeps=[];self.last_error=None;self.reset_evidence=None

    def _model_identity(self):
        # Cryptographic provenance identity; boundary checks compare these bytes.
        import hashlib
        h=hashlib.sha256(model_signature(self.model).encode())
        for key in _MODEL_IDENTITY_ARRAYS:
            h.update(getattr(self.model,key).tobytes())
        h.update(str((self.model.neq,self.model.nmocap)).encode())
        return h.digest()

    def _check_model(self):
        require(self.loaded.mapping==self._mapping
                and str(self.model.opt)==self._model_options
                and (self.model.neq,self.model.nmocap)==self._model_counts,
                'Model/mapping changed')
        for key,dtype,shape,snapshot in self._model_snapshot:
            current=getattr(self.model,key)
            require(current.dtype==dtype and current.shape==shape
                    and np.array_equal(current.view(np.uint8),snapshot),'Model/mapping changed')
        check_callbacks()

    def _fail(self, exc, action=None):
        self._ready=False;self.tracker.invalidate(str(exc))
        # Do not query derived dynamics after an error: preserve the FIRST exception.
        self.last_error=_diagnostic({'error':str(exc),'error_type':type(exc).__name__,
            'seed':self._seed,'reset_seed':self.reset_seed,'sim_time_s':float(self.data.time),
            'elapsed_sim_s':None if self.episode_start_time is None else float(self.data.time-self.episode_start_time),
            'action':action,'previous_action':self.previous_action.copy(),'target':self._target.copy(),
            'ctrl':self.data.ctrl.copy(),'qpos':self.data.qpos.copy(),'qvel':self.data.qvel.copy(),
            'warnings':self.data.warning.number.copy(),'last_measurement':self._measurement,
            'reset_evidence':getattr(exc,'evidence',None)})
        return EnvExecutionError(str(exc),copy.deepcopy(self.last_error))

    def reset(self, *, seed=None, options=None):
        require(not self._closed,'Environment closed')
        require(options is None or options=={}, 'Unsupported reset options; only supine episodes')
        super().reset(seed=seed)
        self._ready=False;self._seed=int(self.np_random_seed)
        self.reset_seed=int(self.np_random.integers(0,2**31-1))
        self.episode_start_time=None;self._measurement=None;self.last_error=None
        started=time.monotonic()
        try:
            self._check_model()
            settings=replace(ResetSettings(),perturb_rad=self.config.reset_perturb_rad,wall_s=self.config.reset_wall_s)
            result=reset_supine(self.loaded,self.data,seed=self.reset_seed,settings=settings)
            require(time.monotonic()-started<self.config.reset_wall_s,'Reset wall-time deadline')
            self.reset_evidence=copy.deepcopy(result)
            self.episode_start_time=float(self.data.time)
            self.tracker.reset_from_supine(self.context,self.data,result)
            self._measurement=measure_standing(self.context,self.data)
            self._result=self.tracker.update(self._measurement)
            self.previous_action=np.zeros(31,dtype=np.float32);self._target=self.q_ref.copy()
            self._potential=_potential(self._measurement);self._physics_steps=0;self.last_substeps=[]
            self._ready=True;self._expected_state=integration_state(self.model,self.data)
            observation=self._observation()
            info=self._info(0,None,None,{}, {},0.)
            if self.render_mode=='human':self.render()
            require(time.monotonic()-started<self.config.reset_wall_s,'Reset wall-time deadline')
            return observation,info
        except Exception as exc:raise self._fail(exc) from exc

    def action_targets(self, action):
        value=np.asarray(action)
        require(value.shape==(31,), 'Action must have exact shape (31,)')
        require(np.issubdtype(value.dtype,np.number) and not np.iscomplexobj(value),'Action must be real numeric')
        require(np.isfinite(value).all(),'Nonfinite action')
        require(np.all(value>=-1.) and np.all(value<=1.),'Action outside [-1, 1]')
        applied=value.astype(np.float32).copy()
        target=self.q_ref+applied*np.where(applied>=0,self.scale_positive,self.scale_negative)
        return applied,np.clip(target,self.q_min,self.q_max)

    def action_for_targets(self, targets):
        """Inverse for diagnostic/consumer targets, in the same actuator order."""
        q=np.asarray(targets,dtype=float)
        require(q.shape==(31,) and np.isfinite(q).all() and np.all(q>=self.q_min) and np.all(q<=self.q_max),
                'Targets outside effective joint ranges')
        delta=q-self.q_ref;scale=np.where(delta>=0,self.scale_positive,self.scale_negative)
        action=np.divide(delta,scale,out=np.zeros(31),where=scale>0)
        return action.astype(np.float32)

    def _safety_reason(self, v):
        for name,limit in [('floor_penetration_m',self.config.safety_floor_m),
                           ('self_penetration_m',self.config.safety_self_m),
                           ('joint_limit_rad',self.config.safety_joint_limit_rad),
                           ('joint_rad_s',self.config.safety_joint_speed_rad_s)]:
            if v[name]>limit:return 'safety_abort:'+name
        return None

    def step(self, action):
        require(not self._closed and self._ready,'Call reset before step: episode ended, invalid or closed')
        started=time.monotonic();deadline=started+self.config.step_wall_s
        try:
            applied,target=self.action_targets(action)
            self._check_model()
            require(not np.any(self.data.qfrc_applied) and not np.any(self.data.xfrc_applied),'Illegal external force')
            require(np.array_equal(self._expected_state,integration_state(self.model,self.data)),
                    'State changed between calls (teleport/reset/external control)')
            self._target=target;self.last_substeps=[]
            change=float(np.mean((applied.astype(float)-self.previous_action)**2))
            qualified_s=torque_integral=0.;saturated=0;executed=0
            saturated_time=0;joint_saturated=np.zeros(31,dtype=int)
            termination=None;truncation=None;raw_max=applied_max=0.
            peaks={key:0. for key in ('joint_rad_s','joint_limit_rad','floor_penetration_m','self_penetration_m')}
            for _ in range(self.config.decimation):
                q,dq=self.loaded.read_state(self.data)
                tau_raw=self.kp*(target-q)-self.kd*dq
                tau=np.clip(tau_raw,self.context.efforts[:,0],self.context.efforts[:,1])
                self.data.ctrl[self.context.ctrladr]=tau
                checked_step(self.loaded,self.data,deadline)
                v=measure_standing(self.context,self.data);result=self.tracker.update(v)
                require(result['invalid_reason'] is None,'Invalid success sampling: '+str(result['invalid_reason']))
                self._measurement=v;self._result=result;executed+=1;self._physics_steps+=1
                effort=self.data.qfrc_actuator[self.context.vadr].copy()
                mask=tau_raw!=tau;saturated+=int(np.count_nonzero(mask))
                saturated_time+=int(np.any(mask));joint_saturated+=mask
                raw_max=max(raw_max,float(max(abs(tau_raw))));applied_max=max(applied_max,float(max(abs(effort))))
                torque_integral+=float(np.mean((effort/self.effort_magnitude)**2))*self.physics_dt
                qualified=bool(result['instant_standing_ok'] and not result['failures'] and result['invalid_reason'] is None)
                qualified_s+=float(qualified)*self.physics_dt  # right-endpoint integration, NOT dwell timer
                for key in peaks:peaks[key]=max(peaks[key],v[key])
                if self.capture_substeps:
                    self.last_substeps.append(copy.deepcopy({'measurement':v,'success':result,'target_rad':target.tolist(),
                        'q_rad':q.tolist(),'dq_rad_s':dq.tolist(),'tau_raw_Nm':tau_raw.tolist(),
                        'applied_torque_Nm':effort.tolist(),'qualified':qualified}))
                if result['recovery_success']:termination='success'
                else:termination=self._safety_reason(v)
                if termination is None and result['timed_out']:truncation='time_limit'
                if termination or truncation:break
            self._check_model()
            require(time.monotonic()<deadline,'Step wall-time deadline')
            current=_potential(self._measurement)
            reward,raw,terms=_reward(self.config,self._potential,current,termination is not None,
                                    qualified_s,torque_integral,change,termination=='success')
            self._potential=current;self.previous_action=applied.copy()
            self._ready=not (termination or truncation)
            self._expected_state=integration_state(self.model,self.data)
            observation=self._observation()
            info=self._info(executed,termination,truncation,raw,terms,saturated/(executed*31))
            info.update(control={'any_saturation_time_fraction':saturated_time/executed,
                                 'joint_saturation_time_fraction':(joint_saturated/executed).tolist(),
                                 'target_boundary_fraction':float(np.mean((target==self.q_min)|(target==self.q_max))),
                                 'raw_torque_abs_max_Nm':raw_max,'applied_torque_abs_max_Nm':applied_max,**peaks})
            if self.render_mode=='human':self.render()
            require(time.monotonic()<deadline,'Step wall-time deadline')
            return observation,reward,termination is not None,truncation is not None,info
        except Exception as exc:raise self._fail(exc,action) from exc

    def _observation(self):
        c=self.config;d=self.data;v=self._measurement;r=self._result
        q,dq=self.loaded.read_state(d);R=d.xmat[self.context.pelvis].reshape(3,3)
        history=np.zeros(9);valid=self.tracker.reference is not None
        if valid:
            for i,key in enumerate(('pelvis','left','right')):
                delta=np.r_[np.array(v[key+'_xy'])-self.tracker.reference[key],0.]
                limit=CALIBRATED_SETTINGS.pelvis_drift_m_max if i==0 else CALIBRATED_SETTINGS.foot_drift_m_max
                history[3*i:3*i+3]=R.T@delta/limit
        obs=np.concatenate(((q-(self.q_min+self.q_max)/2)/((self.q_max-self.q_min)/2),
            dq/c.joint_velocity_scale_rad_s,R.T@np.array([0.,0.,-1.]),
            R.T@d.qvel[:3]/c.base_linear_scale_m_s,d.qvel[3:6]/c.base_angular_scale_rad_s,
            [v['pelvis_height_m']/CALIBRATED_SETTINGS.h_ref_m,v['left_weight'],v['right_weight'],v['other_weight']],
            self.previous_action,[r['stable_duration_s']/CALIBRATED_SETTINGS.hold_s if valid else 0.,float(valid)],history))
        require(obs.shape==(117,) and np.isfinite(obs).all(),'Invalid float64 observation')
        with np.errstate(over='ignore'):obs=obs.astype(np.float32)
        require(np.isfinite(obs).all() and self.observation_space.contains(obs),'Invalid float32 observation')
        return obs

    def _info(self, steps, termination, truncation, raw, terms, saturation):
        return copy.deepcopy({'is_success':termination=='success','termination_reason':termination,
            'truncation_reason':truncation,'elapsed_sim_s':float(self.data.time-self.episode_start_time),
            'physics_steps_executed':steps,'stable_duration_s':self._result['stable_duration_s'],
            'standing_failures':self._result['failures'],'reward_terms_raw':raw,'reward_terms':terms,
            'torque_saturation_fraction':float(saturation),'reset_seed':self.reset_seed,
            'state':{key:self._measurement[key] for key in ('pelvis_height_m','tilt_deg','left_weight','right_weight','other_weight')},
            'warnings':self.data.warning.number.tolist()})

    def resolved_config(self):
        return {'config':asdict(self.config),'physics_dt':self.physics_dt,'control_dt':self.control_dt,
                'success_settings':asdict(CALIBRATED_SETTINGS),'mapping':[asdict(r) for r in self.loaded.mapping],
                'q_ref_rad':self.q_ref.tolist(),'q_min_rad':self.q_min.tolist(),'q_max_rad':self.q_max.tolist(),
                'scale_positive_rad':self.scale_positive.tolist(),'scale_negative_rad':self.scale_negative.tolist(),
                'kp_Nm_rad':self.kp.tolist(),'kd_Nm_s_rad':self.kd.tolist(),
                'effort_limits_Nm':self.context.efforts.tolist()}

    def render(self):
        require(not self._closed,'Environment closed')
        if self.render_mode is None:return None
        require(self._measurement is not None and self.last_error is None,'Call reset before render: no valid state')
        before=integration_state(self.model,self.data)
        try:
            self._check_model()
            camera=mj.MjvCamera();camera.distance=2.2;camera.azimuth=125;camera.elevation=-28
            camera.lookat[:]=self.data.subtree_com[self.context.pelvis]
            if self.render_mode=='rgb_array':
                if self._renderer is None:self._renderer=mj.Renderer(self.model,height=360,width=480)
                self._renderer.update_scene(self.data,camera=camera);image=self._renderer.render().copy()
            else:
                if self._human is None:self._human=_HumanView(self.model)
                self._human.draw(self.model,self.data,camera);image=None
            require(np.array_equal(before,integration_state(self.model,self.data)),'Render changed integration state')
            self._check_model()
            return image
        except Exception as exc:raise self._fail(exc) from exc

    def close(self):
        if self._renderer is not None:self._renderer.close();self._renderer=None
        if self._human is not None:self._human.close();self._human=None
        self._closed=True;self._ready=False


class _HumanView:
    """Native mjv/mjr + GLFW mechanism; no GUI state editing or mouse forces."""
    users=0  # GLFW process lifetime only; no shared model/data/window.

    def __init__(self,model):
        import os
        import glfw
        require(os.environ.get('MUJOCO_GL')=='glfw','human mode needs a separate MUJOCO_GL=glfw process')
        self.glfw=glfw;self.window=None;self.context=None;self.frame=None
        require(glfw.init(),'GLFW init failed')
        try:
            self.window=glfw.create_window(480,360,'X2RecoveryEnv - read-only physics',None,None)
            require(self.window is not None,'GLFW window failed')
            glfw.make_context_current(self.window);glfw.swap_interval(0)
            self.context=mj.MjrContext(model,mj.mjtFontScale.mjFONTSCALE_100)
            self.scene=mj.MjvScene(model,maxgeom=2000);self.option=mj.MjvOption()
            type(self).users+=1
        except Exception:
            if self.context is not None:self.context.free()
            if self.window:glfw.destroy_window(self.window)
            if not type(self).users:glfw.terminate()
            raise

    def draw(self,model,data,camera):
        g=self.glfw;require(not g.window_should_close(self.window),'Human window closed')
        g.make_context_current(self.window)
        mj.mjv_updateScene(model,data,self.option,None,camera,mj.mjtCatBit.mjCAT_ALL,self.scene)
        self.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0;self.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION]=0
        width,height=g.get_framebuffer_size(self.window);viewport=mj.MjrRect(0,0,width,height)
        mj.mjr_render(viewport,self.scene,self.context)
        rgb=np.empty((height,width,3),dtype=np.uint8);mj.mjr_readPixels(rgb,None,viewport,self.context)
        self.frame=rgb[::-1].copy();g.swap_buffers(self.window);g.poll_events()

    def close(self):
        self.glfw.make_context_current(self.window);self.context.free()
        self.glfw.destroy_window(self.window);type(self).users-=1
        if not type(self).users:self.glfw.terminate()


BILATERAL_ACTION_NAMES = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch',
                          'ankle_roll', 'waist_pitch', 'shoulder_pitch', 'shoulder_roll',
                          'shoulder_yaw', 'elbow')


def bilateral_action_matrix(joint_names):
    """Map normalized bilateral commands into the pinned X2 joint order.

    The model uses +Y pitch/knee/elbow, +X roll and +Z yaw on both sides.
    A sagittal mirror therefore reverses right roll/yaw commands. The slightly
    asymmetric shoulder frames mean this is a coordination prior, not exact IK.
    Unlisted joints keep their reference targets without a learned residual.
    """
    require(len(joint_names) == 31 and len(set(joint_names)) == 31, 'Invalid physical joint order')
    matrix = np.zeros((31, len(BILATERAL_ACTION_NAMES)))
    for column, name in enumerate(BILATERAL_ACTION_NAMES):
        sides = ('',) if name == 'waist_pitch' else ('left_', 'right_')
        for side in sides:
            joint = side + name + '_joint'
            require(joint in joint_names, 'Missing bilateral joint: ' + joint)
            sign = -1. if side == 'right_' and name.endswith(('_roll', '_yaw')) else 1.
            matrix[joint_names.index(joint), column] = sign
    return matrix


INDEPENDENT_LEG_ACTION_NAMES = tuple(side+'_'+name for side in ('left', 'right')
                                     for name in BILATERAL_ACTION_NAMES[:6]) + BILATERAL_ACTION_NAMES[6:]


def independent_legs_action_matrix(joint_names):
    """Twelve independent leg commands and the existing five upper-body synergies."""
    bilateral = bilateral_action_matrix(joint_names)
    matrix = np.zeros((31, len(INDEPENDENT_LEG_ACTION_NAMES)))
    for column, name in enumerate(INDEPENDENT_LEG_ACTION_NAMES[:12]):
        matrix[joint_names.index(name+'_joint'), column] = 1.
    matrix[:, 12:] = bilateral[:, 6:]
    return matrix


@dataclass(frozen=True)
class ControlConfig:
    """Versioned learning controller; the wrapped physical task is unchanged."""
    version: str = 'targets-v1'
    mode: str = 'rate'
    rate_multiplier: float = 1.
    delta_multiplier: float = 1.
    tracking_error_ratio: float | None = None
    reward_version: str = 'original'
    reference_stages: tuple = ()
    residual_multiplier: float = .2
    residual_joint_multipliers: tuple | None = None
    action_layout: str = 'joint31'
    reference_interpolation: str = 'quintic-v1'
    head_reference: tuple = ()
    head_progress_range_m: tuple = ()
    residual_start_s: float = 0.
    residual_ramp_s: float = 0.

    def __post_init__(self):
        layouts = {'targets-v1': 'joint31', 'targets-v2': 'joint31', 'targets-v3': 'bilateral11',
                   'targets-v4': 'bilateral11', 'targets-v5': 'independentlegs17'}
        require(type(self.version) is str and self.version in layouts, 'Unknown controller version')
        require(self.action_layout in layouts.values(), 'Unknown policy action layout')
        require(self.action_layout == layouts[self.version], 'Controller version/action layout mismatch')
        require(self.mode in ('rate', 'measured_delta', 'reference_residual'), 'Unknown action semantics')
        for name in ('residual_start_s', 'residual_ramp_s'):
            value = getattr(self, name)
            require(type(value) in (int, float) and np.isfinite(value) and value >= 0,
                    'Residual phase times must be finite nonnegative numbers')
        if self.version in ('targets-v4', 'targets-v5'):
            require(self.mode == 'reference_residual' and self.residual_ramp_s > 0,
                    'targets-v4/v5 requires reference_residual and a positive residual ramp')
        else:
            require(self.residual_start_s == self.residual_ramp_s == 0,
                    'Residual phase gating requires targets-v4/v5')
        require(self.reference_interpolation in ('quintic-v1', 'linear-v1'), 'Unknown reference interpolation')
        require(self.reference_interpolation == 'quintic-v1' or self.mode == 'reference_residual',
                'Linear reference interpolation requires reference_residual')
        require(self.reward_version in ('original', 'support-v1', 'track-v1', 'upright-support-v2',
                                       'reference-head-support-v1', 'reference-balance-v1'),
                'Unknown reward version')
        require(type(self.head_reference) is tuple and type(self.head_progress_range_m) is tuple,
                'Head reference and progress range must be tuples')
        if self.reward_version in ('reference-head-support-v1', 'reference-balance-v1'):
            require(self.mode == 'reference_residual', 'Head reference reward requires reference_residual')
            require(len(self.head_progress_range_m) == 2
                    and all(type(x) in (int, float) and np.isfinite(x) for x in self.head_progress_range_m)
                    and 0 <= self.head_progress_range_m[0] < self.head_progress_range_m[1],
                    'Head progress range must be finite increasing nonnegative heights')
            require(len(self.head_reference) >= 2 and all(type(row) is tuple and len(row) == 2
                    and all(type(x) in (int, float) and np.isfinite(x) for x in row)
                    and row[0] >= 0 and 0 <= row[1] <= 1 for row in self.head_reference),
                    'Head reference must contain finite (time, progress) pairs')
            require(self.head_reference[0] == (0., 0.) and self.head_reference[-1][1] == 1.
                    and all(b[0] > a[0] for a, b in zip(self.head_reference, self.head_reference[1:])),
                    'Head reference must start at (0,0), end at progress1 and have increasing times')
        else:
            require(not self.head_reference and not self.head_progress_range_m,
                    'Head reference inputs require a head-guided reward version')
        for name in ('rate_multiplier', 'delta_multiplier', 'residual_multiplier'):
            value = getattr(self, name)
            require(type(value) in (int, float) and np.isfinite(value) and value > 0, 'Invalid '+name)
        require(self.tracking_error_ratio is None or (type(self.tracking_error_ratio) in (int, float)
                and np.isfinite(self.tracking_error_ratio) and self.tracking_error_ratio > 0), 'Invalid tracking error ratio')
        require(type(self.reference_stages) is tuple, 'Reference stages must be a tuple')
        require((self.mode == 'reference_residual') == bool(self.reference_stages), 'Reference/controller mismatch')
        require(self.reward_version != 'track-v1' or self.mode == 'reference_residual',
                'Tracking reward requires reference control')
        require((self.version in ('targets-v2', 'targets-v3', 'targets-v4', 'targets-v5')) == (self.residual_joint_multipliers is not None),
                'Jointwise residual multipliers require targets-v2/v3/v4/v5 and explicit values')
        if self.residual_joint_multipliers is not None:
            require(self.mode == 'reference_residual', 'Jointwise residual multipliers require reference_residual')
            require(type(self.residual_joint_multipliers) is tuple and len(self.residual_joint_multipliers) == 31,
                    'Jointwise residual multipliers must be a 31-element tuple in mapped joint order')
            require(all(type(value) in (int, float) and np.isfinite(value) and value >= 0
                        for value in self.residual_joint_multipliers),
                    'Jointwise residual multipliers must be finite nonnegative numbers')

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'reference_stages' in values:
            values['reference_stages'] = tuple((str(n), float(t), dict(q)) for n, t, q in values['reference_stages'])
        if values.get('residual_joint_multipliers') is not None:
            require(type(values['residual_joint_multipliers']) in (list, tuple),
                    'Jointwise residual multipliers must be a list or tuple')
            values['residual_joint_multipliers'] = tuple(values['residual_joint_multipliers'])
        for name in ('head_reference', 'head_progress_range_m'):
            if name in values:
                require(type(values[name]) in (list, tuple), 'Head reference fields must be lists or tuples')
                values[name] = tuple(tuple(row) if type(row) in (list, tuple) else row for row in values[name]) if name == 'head_reference' else tuple(values[name])
        return cls(**values)


def residual_phase_gate(control, elapsed_sim_s):
    """Read the 50Hz pre-control phase; never advance a separate phase state."""
    require(np.isfinite(elapsed_sim_s) and elapsed_sim_s >= 0, 'Invalid residual gate time')
    if control.version not in ('targets-v4', 'targets-v5'):
        return 1.
    u = float(np.clip((elapsed_sim_s-control.residual_start_s)/control.residual_ramp_s, 0., 1.))
    return u**3*(10.-15.*u+6.*u*u)


def controller_reference_targets(elapsed_sim_s, keyframes, interpolation):
    """Sample an explicit reference interpolation, without advancing controller state."""
    if interpolation == 'quintic-v1':
        from .baseline import scripted_targets
        return scripted_targets(elapsed_sim_s, keyframes)
    require(interpolation == 'linear-v1', 'Unknown reference interpolation')
    require(np.isfinite(elapsed_sim_s) and elapsed_sim_s >= 0, 'Invalid episode time')
    times, targets = keyframes.times_s, keyframes.targets_rad
    if elapsed_sim_s >= times[-1]:
        return targets[-1].copy(), dict(name='hold', progress=1., sequence_finished=True)
    index = int(np.searchsorted(times, elapsed_sim_s, side='right')-1)
    fraction = float(np.clip((elapsed_sim_s-times[index])/(times[index+1]-times[index]), 0., 1.))
    return ((1-fraction)*targets[index]+fraction*targets[index+1],
            dict(name=keyframes.phases[index], progress=fraction, sequence_finished=False))


def balance_support_task(state):
    """Near-upright balance densities; these are not standing/safety predicates."""
    names = ('pelvis_height_m', 'upright_dot', 'left_weight', 'right_weight', 'other_weight',
             'com_linear_m_s', 'torso_angular_rad_s', 'self_penetration_m')
    require(np.isfinite([state[name] for name in names]).all(), 'Nonfinite balance task measurement')
    height = float(np.clip(state['pelvis_height_m']/CALIBRATED_SETTINGS.h_ref_m, 0., 1.))
    gate = float(np.clip((state['upright_dot']-.5)/.5, 0., 1.))
    feet = state['left_weight']+state['right_weight']
    bilateral = float(np.clip(min(state['left_weight'], state['right_weight'])/.2, 0., 1.))
    distance = max(.8-feet, 0., feet-1.2)
    nominal_load = float(np.exp(-(distance/.4)**2))
    other_support = float(np.exp(-(state['other_weight']/.1)**2))
    low_speed = float(1./(1.+(state['com_linear_m_s']/.3)**2+(state['torso_angular_rad_s']/.75)**2))
    terms = dict(balance=6.*height*gate*bilateral*nominal_load*other_support*low_speed,
                 overload=-.5*gate*float(np.clip(feet-1.2, 0., 3.)),
                 self_penetration=-.5*gate*float(np.clip(state['self_penetration_m']/.003, 0., 1.)))
    require(np.isfinite(list(terms.values())).all(), 'Nonfinite balance task reward')
    return terms, dict(balance_height_factor=height, balance_upright_gate=gate,
        balance_bilateral_factor=bilateral, balance_nominal_load_factor=nominal_load,
        balance_other_support_factor=other_support, balance_low_speed_factor=low_speed,
        balance_feet_weight=float(feet))


def head_support_task(config, q, reference, pose_indices, state, qualified, head_height_m, elapsed_s):
    """Control-boundary task reward densities, independent of success detection."""
    require(config.reward_version in ('reference-head-support-v1', 'reference-balance-v1'),
            'Expected head-guided reward')
    require(np.isfinite(head_height_m) and np.isfinite(elapsed_s) and elapsed_s >= 0,
            'Nonfinite head task measurement or invalid time')
    z0, zstar = config.head_progress_range_m
    progress = float(np.clip((head_height_m-z0)/(zstar-z0), 0., 1.))
    table = np.asarray(config.head_reference)
    reference_progress = float(np.interp(elapsed_s, table[:, 0], table[:, 1]))
    pose_error = np.asarray(q)[pose_indices]-np.asarray(reference)[pose_indices]
    joint_match = float(np.exp(-np.mean(pose_error**2)/.35**2))
    head_match = float(np.exp(-((progress-reference_progress)/.2)**2))
    upright = float(np.clip(state['upright_dot'], 0., 1.))
    left, right, other = state['left_weight'], state['right_weight'], state['other_weight']
    contact = float(np.clip(left+right+other, 0., 1.))
    foot_transfer = float(np.clip(left+right, 0., 1.)*np.clip(min(left,right)/.25, 0., 1.)
                          *(1.-np.clip(other, 0., 1.)))
    terms = dict(pose_guide=(1.-reference_progress)*joint_match,
                 head_track=2.*head_match*contact, upright_progress=4.*progress*upright*contact,
                 foot_transfer=2.*progress*upright*foot_transfer, standing=5.*float(qualified))
    require(np.isfinite(list(terms.values())).all(), 'Nonfinite head task reward')
    diagnostics = dict(head_height_m=float(head_height_m), head_progress=progress,
        reference_progress=reference_progress, head_tracking_error=progress-reference_progress,
        head_tracking_error_m=float(head_height_m-(z0+reference_progress*(zstar-z0))),
        contact_factor=contact, upright=upright, foot_transfer_factor=foot_transfer,
        pose_error_rms_rad=float(np.sqrt(np.mean(pose_error**2))), elapsed_sim_s=float(elapsed_s),
        measurement_scope='post-step control boundary; head_pitch_link body origin in world Z')
    if config.reward_version == 'reference-balance-v1':
        balance, factors = balance_support_task(state)
        terms = dict(pose_guide=terms['pose_guide'], head_track=terms['head_track'],
                     **balance, standing=terms['standing'])
        diagnostics.update(factors)
    return terms, diagnostics


class ControlledRecoveryEnv(gym.Wrapper):
    """50 Hz target generation ahead of the original limited PD and MuJoCo step.

    This is intentionally not HoST's per-physics-step current-q offset. The new
    policy observes the adopted target and elapsed phase. All episodes still start
    through the original supine reset; the original tracker owns success.
    """
    def __init__(self, env=None, control=None):
        super().__init__(env if env is not None else X2RecoveryEnv())
        require(isinstance(self.env, X2RecoveryEnv), 'Expected the native X2 physical environment')
        self.control_config = control if control is not None else ControlConfig()
        require(isinstance(self.control_config, ControlConfig), 'Expected ControlConfig')
        if self.control_config.version in ('targets-v4', 'targets-v5'):
            require(self.control_config.residual_start_s+self.control_config.residual_ramp_s
                    <= self.env.config.episode_timeout_s, 'Residual ramp exceeds the original episode timeout')
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (149,), dtype=np.float32)
        joint_names = [row.joint_name for row in self.env.loaded.mapping]
        self.pose_indices = np.array([i for i, name in enumerate(joint_names)
                                      if 'head' not in name and 'wrist' not in name], dtype=int)
        self.head_body = None
        if self.control_config.reward_version in ('reference-head-support-v1', 'reference-balance-v1'):
            require(len(self.pose_indices) == 23, 'Expected 23 reference pose joints')
            self.head_body = self.env.model.body('head_pitch_link').id
        if self.control_config.action_layout == 'bilateral11':
            self.policy_action_names = BILATERAL_ACTION_NAMES
            self.policy_to_joint = bilateral_action_matrix(joint_names)
            self.action_space = gym.spaces.Box(-1., 1., (11,), dtype=np.float32)
        elif self.control_config.action_layout == 'independentlegs17':
            self.policy_action_names = INDEPENDENT_LEG_ACTION_NAMES
            self.policy_to_joint = independent_legs_action_matrix(joint_names)
            self.action_space = gym.spaces.Box(-1., 1., (17,), dtype=np.float32)
        else:
            self.policy_action_names = tuple(joint_names)
            self.policy_to_joint = np.eye(31)
        groups = {'hip': 1.5, 'knee': 1.5, 'ankle': 1., 'waist': .6,
                  'shoulder': 1.5, 'elbow': 1.5, 'wrist': .6, 'head': .6}
        self.rate = np.array([groups[r.joint_name.removeprefix('left_').removeprefix('right_').split('_')[0]]
                              for r in self.env.loaded.mapping]) * self.control_config.rate_multiplier
        # A position error producing one effort limit under P alone. Damping and
        # contact dynamics can still cause saturation; this is not a force guarantee.
        self.delta = np.minimum(self.env.effort_magnitude / self.env.kp,
                                (self.env.q_max-self.env.q_min)*.25) * self.control_config.delta_multiplier
        # Static mapped-joint coefficients do not add controller state. An exact
        # zero disables only that joint's learned residual, not its reference/PD.
        self.residual_scale = self.delta * self.control_config.residual_multiplier
        if self.control_config.residual_joint_multipliers is not None:
            self.residual_scale *= np.asarray(self.control_config.residual_joint_multipliers)
        require(np.isfinite(self.residual_scale).all(), 'Derived residual scales must be finite')
        self.target = self.env.q_ref.copy()
        self.reference = None
        self.last_controller = None

    def _augment(self, observation):
        e = self.env
        target = (self.target-(e.q_min+e.q_max)/2)/((e.q_max-e.q_min)/2)
        phase = (e.data.time-e.episode_start_time)/e.config.episode_timeout_s
        obs = np.r_[observation, target, phase].astype(np.float32)
        require(obs.shape == (149,) and np.isfinite(obs).all(), 'Invalid controller observation')
        return obs

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        q, _ = self.env.loaded.read_state(self.env.data)
        self.target = np.clip(q, self.env.q_min, self.env.q_max)
        self.reference = None
        if self.control_config.reference_stages:
            from .baseline import build_keyframes
            self.reference = build_keyframes(q, self.env.loaded.mapping, stages=self.control_config.reference_stages)
        self.last_controller = None
        info['controller'] = dict(version=self.control_config.version, mode=self.control_config.mode,
                                  action_layout=self.control_config.action_layout,
                                  initial_target_rad=self.target.tolist())
        if self.control_config.version in ('targets-v4', 'targets-v5'):
            info['controller'].update(residual_gate=0., residual_gate_time_s=0.)
        return self._augment(observation), info

    def step(self, action):
        e, c = self.env, self.control_config
        action = np.asarray(action)
        require(action.shape == self.action_space.shape and np.issubdtype(action.dtype, np.number)
                and not np.iscomplexobj(action) and np.isfinite(action).all() and np.all(abs(action) <= 1),
                'Controller action must be finite and in [-1,1] with shape ' + str(self.action_space.shape))
        # Keep legacy arithmetic intact; the new linear map acts once, inside
        # the environment. PPO retains the policy's original action/log-probability.
        mapped_action = self.policy_to_joint @ action if c.action_layout != 'joint31' else action
        q, _ = e.loaded.read_state(e.data)
        previous = self.target.copy()
        reference = None
        gate_time = float(e.data.time-e.episode_start_time)
        if c.mode == 'rate':
            _, requested = e.action_targets(action)
        elif c.mode == 'measured_delta':
            requested = np.clip(q+self.delta*action, e.q_min, e.q_max)
        else:
            reference, _ = controller_reference_targets(
                e.data.time-e.episode_start_time+e.control_dt, self.reference, c.reference_interpolation)
            if c.version in ('targets-v4', 'targets-v5'):
                gate = residual_phase_gate(c, gate_time)
                gated_residual = self.residual_scale*mapped_action*gate
                requested = np.clip(reference+gated_residual, e.q_min, e.q_max)
            else:
                requested = np.clip(reference+self.residual_scale*mapped_action, e.q_min, e.q_max)
        if c.tracking_error_ratio is not None:
            margin = e.effort_magnitude/e.kp*c.tracking_error_ratio
            requested = np.clip(requested, np.maximum(e.q_min,q-margin), np.minimum(e.q_max,q+margin))
        self.target = requested if c.mode == 'measured_delta' else previous+np.clip(
            requested-previous, -self.rate*e.control_dt, self.rate*e.control_dt)
        physical_action = e.action_for_targets(self.target)
        observation, original_reward, terminated, truncated, info = e.step(physical_action)
        self.last_controller = dict(policy_action=action.tolist(), mapped_policy_action=mapped_action.tolist(),
            action_layout=c.action_layout, requested_target_rad=requested.tolist(),
            adopted_target_rad=self.target.tolist(), pre_q_rad=q.tolist(),
            pre_time_s=float(e.data.time-e.episode_start_time-info['physics_steps_executed']*e.physics_dt),
            target_tracking_max_rad=float(np.max(abs(self.target-q))),
            target_tracking_rms_rad=float(np.sqrt(np.mean((self.target-q)**2))))
        if c.version in ('targets-v4', 'targets-v5'):
            self.last_controller.update(residual_gate=gate, residual_gate_time_s=gate_time,
                                        gated_residual_rad=gated_residual.tolist())
        info['controller'] = self.last_controller
        reward = original_reward
        if c.reward_version != 'original':
            # New task objective, not an optimal-policy-invariant shaping claim.
            v = e._measurement
            dt = info['physics_steps_executed']*e.physics_dt
            if c.reward_version in ('reference-head-support-v1', 'reference-balance-v1'):
                elapsed = float(e.data.time-e.episode_start_time)
                measured_reference, _ = controller_reference_targets(elapsed, self.reference, c.reference_interpolation)
                actual, _ = e.loaded.read_state(e.data)
                density, info['learning_task'] = head_support_task(c, actual, measured_reference,
                    self.pose_indices, v, e._result['instant_standing_ok'] and not e._result['failures'],
                    float(e.data.xpos[self.head_body, 2]), elapsed)
                terms = {name: dt*value for name, value in density.items()}
                terms.update(torque_cost=info['reward_terms']['torque_cost'],
                    target_change=-.01*float(np.mean((self.target-previous)**2)),
                    success=50.*float(info['is_success']),
                    safety=-2.*float(str(info['termination_reason']).startswith('safety_abort:')))
            else:
                height = np.clip(v['pelvis_height_m']/CALIBRATED_SETTINGS.h_ref_m,0,1.2)
                upright = np.clip(v['upright_dot'],0,1)
                feet = np.clip(v['left_weight']+v['right_weight'],0,1.2)
                bilateral = np.clip(min(v['left_weight'],v['right_weight'])/.25,0,1)
                terms = dict(posture=dt*2*upright, elevation=dt*3*height,
                    foot_support=dt*3*height*upright*feet,
                    bilateral=dt*height*upright*bilateral,
                    standing=dt*5*float(e._result['instant_standing_ok'] and not e._result['failures']),
                    torque_cost=info['reward_terms']['torque_cost'],
                    target_change=-.01*float(np.mean((self.target-previous)**2)),
                    success=50.*float(info['is_success']), safety=-2.*float(str(info['termination_reason']).startswith('safety_abort:')))
                if c.reward_version == 'upright-support-v2':
                    # A new objective: elevation only pays in proportion to uprightness.
                    # The earlier reward versions retain their recorded semantics.
                    terms['elevation'] *= upright
                if c.reward_version == 'track-v1':
                    require(reference is not None, 'Tracking reward requires reference')
                    actual, _ = e.loaded.read_state(e.data)
                    terms['reference_tracking'] = dt*float(np.exp(-4*np.mean((actual-reference)**2)))
            info['physical_reward_terms'] = info['reward_terms']
            info['physical_reward'] = original_reward
            info['reward_terms'] = terms
            reward = float(sum(terms.values()))
        return self._augment(observation), reward, terminated, truncated, info

    def resolved_config(self):
        resolved = dict(base=self.env.resolved_config(), controller=asdict(self.control_config),
            target_rate_rad_s=self.rate.tolist(), measured_delta_rad=self.delta.tolist(),
            residual_scale_rad=self.residual_scale.tolist(),
            residual_joint_full_scale_rad=(self.residual_scale*np.sum(abs(self.policy_to_joint), axis=1)).tolist(),
            residual_joint_names=[row.joint_name for row in self.env.loaded.mapping],
            policy_action_names=list(self.policy_action_names), policy_to_joint_matrix=self.policy_to_joint.tolist(),
            residual_action_semantics='reference + residual_scale_rad * (policy_to_joint_matrix @ policy_action); joint_multiplier=1 for targets-v1; then joint-range and target-rate limits',
            reference_sampling=dict(interpolation=self.control_config.reference_interpolation,
                frequency_hz=1/self.env.control_dt, time='elapsed recovery time + control_dt',
                endpoint='final legal reference target held after final keyframe'),
            action_schema=f'{self.action_space.shape[0]} normalized {self.control_config.action_layout} policy actions; controller transforms once per control step',
            observation_schema='native117 + adopted_target normalized31 + elapsed/timeout1',
            observation_shape=[149], action_shape=list(self.action_space.shape))
        if self.control_config.version in ('targets-v4', 'targets-v5'):
            resolved['residual_action_semantics'] = ('reference + residual_phase_gate(pre_control_elapsed) * '
                'residual_scale_rad * (policy_to_joint_matrix @ policy_action); then joint-range and target-rate limits')
            resolved['residual_gate'] = dict(start_s=self.control_config.residual_start_s,
                ramp_s=self.control_config.residual_ramp_s, interpolation='quintic 10u^3-15u^4+6u^5',
                u='clip((pre_control_elapsed-start_s)/ramp_s,0,1)',
                sampling='once at the beginning of each control step; held for its physical substeps',
                timing='off-grid start takes effect at the first subsequent control boundary',
                state='derived only from existing observed elapsed/timeout phase; reset adds no hidden state',
                residual_scale_scope='resolved residual scales are full authority when gate reaches1',
                policy_buffer='original policy actions and log probabilities are unchanged')
        if self.control_config.reward_version in ('reference-head-support-v1', 'reference-balance-v1'):
            resolved['learning_task'] = dict(head_body='head_pitch_link', head_point='body origin world Z',
                pose_joint_names=[self.env.loaded.mapping[i].joint_name for i in self.pose_indices],
                head_reference_interpolation='linear with endpoint hold', pose_sigma_rad=.35,
                head_progress_sigma=.2, density_weights=dict(pose_guide=1.,head_track=2.,upright_progress=4.,foot_transfer=2.,standing=5.),
                sampling='post-control-step measurements weighted by actual executed physics duration',
                observation='existing joint angles, pelvis gravity/height and elapsed phase determine the task inputs; normalization/table are fixed configuration')
            if self.control_config.reward_version == 'reference-balance-v1':
                resolved['learning_task'].update(
                    density_weights=dict(pose_guide=1., head_track=2., balance=6., overload=-.5,
                                         self_penetration=-.5, standing=5.),
                    balance_factors=dict(
                        height='clip(pelvis_height_m/h_ref_m,0,1)',
                        upright_gate='clip((upright_dot-.5)/.5,0,1)',
                        bilateral='clip(min(left_weight,right_weight)/.2,0,1)',
                        nominal_load='exp(-(distance(left_weight+right_weight,[.8,1.2])/.4)^2)',
                        other_support='exp(-(other_weight/.1)^2)',
                        low_speed='1/(1+(com_linear_m_s/.3)^2+(torso_angular_rad_s/.75)^2)'),
                    balance_density='6 * product(balance_factors)',
                    overload_density='-.5 * upright_gate * clip(left_weight+right_weight-1.2,0,3)',
                    self_penetration_density='-.5 * upright_gate * clip(self_penetration_m/.003,0,1)',
                    standing='unchanged native independent standing result; no new success or termination rule',
                    scope='New task objective; bounded balance costs vanish at torso tilt >=60 degrees; no optimal-policy invariance claim')
        return resolved
