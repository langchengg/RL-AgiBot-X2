"""Native MuJoCo Gymnasium recovery environment; no ROS, audit I/O or policy.

State and contact observations are simulator-accessible, not a hardware sensing
claim. Only reset_supine initializes physics; step uses bounded joint torques.
"""
from dataclasses import asdict, dataclass, field, replace
import copy
import time
import gymnasium as gym
import mujoco as mj
import numpy as np
from .model import load_effective_model, require
from .reset import ResetSettings, reset_supine, checked_step, model_signature, integration_state, check_callbacks
from .success import StandingContext, SuccessTracker, CALIBRATED_SETTINGS, measure_standing


@dataclass(frozen=True)
class EnvConfig:
    decimation: int = 20
    episode_timeout_s: float = 20.
    step_wall_s: float = 5.
    reset_wall_s: float = 45.
    reset_perturb_rad: float = 0.  # Only the Step 4 validated [0, .005] range.
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
    # group, Kp [N m/rad], Kd [N m s/rad]; candidate based on Step 5 posture PD.
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
    return value


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
        self.metadata={**type(self).metadata,'render_fps':round(1/self.control_dt)}
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
        self._renderer=None;self._human=None;self._closed=False;self._ready=False
        self._measurement=None;self._result=None;self._expected_state=None
        self._seed=None;self.reset_seed=None;self.episode_start_time=None
        self.previous_action=np.zeros(31,dtype=np.float32);self._target=self.q_ref.copy()
        self._physics_steps=0;self.last_substeps=[];self.last_error=None;self.reset_evidence=None

    def _model_identity(self):
        # Whole mesh identity at policy/reset/render boundaries, NEVER each substep.
        import hashlib
        h=hashlib.sha256(model_signature(self.model).encode())
        for key in ('jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','body_parentid','body_rootid',
                    'geom_bodyid','actuator_trnid','actuator_trntype','actuator_dyntype','actuator_gaintype',
                    'actuator_ctrladr','actuator_outadr','actuator_forcelimited','actuator_forcerange'):
            h.update(getattr(self.model,key).tobytes())
        h.update(str((self.model.neq,self.model.nmocap)).encode())
        return h.digest()

    def _check_model(self):
        require(self.loaded.mapping==self._mapping and self._model_identity()==self._signature,'Model/mapping changed')
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
                saturated+=int(np.count_nonzero(tau_raw!=tau))
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
            info.update(control={'target_boundary_fraction':float(np.mean((target==self.q_min)|(target==self.q_max))),
                                 'raw_torque_abs_max_Nm':raw_max,'applied_torque_abs_max_Nm':applied_max,**peaks})
            if self.render_mode=='human':self.render()
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
        require(self._measurement is not None,'Call reset before render')
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
    """Step 4 native mjv/mjr + GLFW mechanism; no GUI state editing or mouse forces."""
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
