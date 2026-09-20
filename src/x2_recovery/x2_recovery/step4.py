"""Bounded Step 4 diagnostics, live observer and OSMesa trajectory recording."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
import uuid
import mujoco as mj
import numpy as np
from .model import load_effective_model, require
from .reset import (ResetSettings, ResetFailure, reset_supine, construct_pose,
                    measurements, checked_step, notify, integration_state, model_signature)
from .model_audit import fingerprint, identity, provenance, json_value

SEEDS=tuple(range(100,120))


class PulseFailure(ValueError):
    def __init__(self,message,evidence):
        super().__init__(message)
        self.evidence=evidence


def write_json(path,value):
    path.write_text(json.dumps(json_value(value),indent=2,allow_nan=False)+'\n')


def contaminate(loaded,d,index):
    """Explicit test fixture, never called by runtime reset or ordinary stepping."""
    rng=np.random.default_rng(7000+index)
    d.ctrl[:]=rng.uniform(-.2,.2,loaded.model.nu)
    d.qvel[:]=rng.uniform(-.05,.05,loaded.model.nv)
    d.qfrc_applied[:]=rng.uniform(-.1,.1,loaded.model.nv)
    d.xfrc_applied[:]=rng.uniform(-.1,.1,d.xfrc_applied.shape)
    d.qacc_warmstart[:]=rng.uniform(-1.,1.,loaded.model.nv)
    d.act[:]=.1;d.history[:]=.1;d.userdata[:]=.2
    d.eq_active[:]=~loaded.model.eq_active0
    d.time=10.+index


def reset_batch(loaded,d):
    records=[]
    for mode,seeds,amplitude in (('fixed',range(20),0.),('seeded',SEEDS,.005)):
        for seed in seeds:
            contaminate(loaded,d,int(seed))
            try:r=reset_supine(loaded,d,seed=int(seed),settings=replace(ResetSettings(),perturb_rad=amplitude))
            except ResetFailure as exc:r=exc.evidence
            except ValueError as exc:r={'status':'FAIL','error':str(exc)}
            records.append({'mode':mode,'seed':int(seed),**r})
            print(json.dumps({'reset':mode,'seed':seed,'status':r['status'],
                              'time':r.get('episode_start_time'),'error':r.get('error')}),flush=True)
    repetitions=[]
    for seed in (100,107,119):
        contaminate(loaded,d,1000+seed)
        prior=next(r for r in records if r['mode']=='seeded' and r['seed']==seed)
        try:
            r=reset_supine(loaded,d,seed=seed,settings=replace(ResetSettings(),perturb_rad=.005))
            qerr=float(np.max(abs(np.array(prior['final_qpos'])-r['final_qpos'])))
            verr=float(np.max(abs(np.array(prior['final_qvel'])-r['final_qvel'])))
            same=(prior['initial']['requested_joints_rad']==r['initial']['requested_joints_rad']
                  and abs(prior['episode_start_time']-r['episode_start_time'])<1e-10
                  and qerr<1e-10 and verr<1e-10 and prior['window_maxima']==r['window_maxima'])
            repetitions.append({'seed':seed,'status':'PASS' if same else 'FAIL','q_error':qerr,'v_error':verr})
        except (ValueError,KeyError) as exc:repetitions.append({'seed':seed,'status':'FAIL','error':str(exc)})
    return {'status':'PASS' if all(r['status']=='PASS' for r in records+repetitions) else 'FAIL',
            'fixed_passes':sum(r['status']=='PASS' for r in records[:20]),
            'seeded_passes':sum(r['status']=='PASS' for r in records[20:]),
            'records':records,'same_seed_repetitions':repetitions,
            'repeat_tolerance':{'q_rad_or_m':1e-10,'v_rad_s_or_m_s':1e-10,'time_s':1e-10},
            'initial_diversity_max_rad':float(np.max(np.ptp([list(r['initial']['requested_joints_rad'].values())
                for r in records[20:] if 'initial' in r],axis=0))),
            'final_diversity_max_qpos':float(np.max(np.ptp([r['final_qpos'] for r in records[20:]
                if r['status']=='PASS'],axis=0))) if all(r['status']=='PASS' for r in records[20:]) else None}


def ceiling(row):
    n=row.joint_name
    value=(4. if 'hip_' in n or 'knee' in n else 1.5 if 'ankle' in n else
           2. if 'waist' in n else .6 if 'head' in n else .7 if 'wrist' in n else
           1. if 'elbow' in n else 1.5)
    return min(value, .5*min(abs(v) for v in row.effort_range))


def pulse(loaded,d,commands,*,sign=1.,observer=None,label='pulse',pulse_s=.12,neutral_s=.12):
    """One smooth torque pulse, then neutral. Pose changes only through mj_step."""
    m=loaded.model;start=d.time;reference=d.qpos.copy();trace=[];peak_floor=peak_self=peak_speed=0.
    deadline=time.monotonic()+30.;last_force_error=0.
    try:
        for k in range(round((pulse_s+neutral_s)/m.opt.timestep)):
            t=k*m.opt.timestep
            shape=np.sin(np.pi*t/pulse_s)**2 if t<pulse_s else 0.
            d.ctrl[:]=0.
            for row,amplitude in commands:
                requested=sign*amplitude*shape
                require(row.effort_range[0]<=requested<=row.effort_range[1],'Torque outside bounds')
                d.ctrl[row.ctrl_index]=requested
            checked_step(loaded,d,deadline)
            v=measurements(loaded,d)
            q,dq=loaded.read_state(d)
            peak_floor=max(peak_floor,v['floor']['penetration_m']);peak_self=max(peak_self,v['self']['penetration_m'])
            peak_speed=max(peak_speed,v['joint_rad_s'])
            require(peak_floor<.012 and peak_self<.003,'Actuation intersection guard')
            require(v['joint_violation_rad']<1e-4,'Actuation joint-limit guard')
            require(peak_speed<4. and np.max(abs(d.qpos[loaded.qpos_addresses]-reference[loaded.qpos_addresses]))<.2,
                    'Actuation displacement/speed guard')
            for row,amplitude in commands:
                error=abs(d.qfrc_actuator[row.dof_address]-d.ctrl[row.ctrl_index])
                last_force_error=max(last_force_error,float(error))
                require(error<1e-10,'Input/transmitted effort mismatch')
            if k%10==0 or k==round((pulse_s+neutral_s)/m.opt.timestep)-1:
                trace.append({'elapsed_s':float(d.time-start),'q':q,'dq':dq,'ctrl':d.ctrl.copy(),
                              'effort':d.qfrc_actuator[loaded.dof_addresses].copy(),
                              'floor_m':v['floor']['penetration_m'],'self_m':v['self']['penetration_m']})
            if k%40==0:
                notify(m,d,observer,label if t<pulse_s else 'zero-input',{'episode_elapsed_s':float(d.time-start)})
        return {'status':'PASS','label':label,'sign':sign,'trace':trace,'max_effort_error_Nm':last_force_error,
                'peak_ground_m':peak_floor,'peak_self_m':peak_self,'peak_joint_rad_s':peak_speed,
                'final_q':loaded.read_state(d)[0],'final_dq':loaded.read_state(d)[1]}
    except ValueError as exc:
        raise PulseFailure(str(exc),{'status':'GUARD_STOP','label':label,'sign':sign,
            'time_s':float(d.time),'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),
            'ctrl':d.ctrl.copy(),'trace':trace,'metrics':measurements(loaded,d)}) from exc
    finally:
        d.ctrl[:]=0.
        mj.mj_forward(m,d)


def causal_joint(loaded,reference,row,scope='settled_supine'):
    m=loaded.model;attempts=[]
    for scale in (.5,1.):
        amplitude=ceiling(row)*scale;rollouts=[]
        try:
            for sign in (0.,1.,-1.):
                d=mj.MjData(m);mj.mj_copyData(d,m,reference)
                require(np.array_equal(integration_state(m,d),integration_state(m,reference)), 'Incomplete diagnostic state copy')
                mj.mj_forward(m,d)
                rollouts.append(pulse(loaded,d,[(row,amplitude)],sign=sign,label=row.joint_name))
            base,positive,negative=rollouts
            # Compare sampled trajectories, not merely final sign (gravity/contact can dominate).
            diffs=[max(abs(a['q'][row.ctrl_index]-b['q'][row.ctrl_index]) for a,b in zip(r['trace'],base['trace']))
                   for r in (positive,negative)]
            attempts.append({'amplitude_Nm':amplitude,'causal_max_displacement_rad':diffs,'rollouts':rollouts})
            if min(diffs)>1e-5:
                return {'joint':row.joint_name,'scope':scope,'status':'PASS','attempts':attempts}
        except ValueError as exc:
            attempts.append({'amplitude_Nm':amplitude,'status':'GUARD_STOP','error':str(exc),
                             'failed_rollout':getattr(exc,'evidence',None),'completed_rollouts':rollouts})
            break  # A safety stop is not a reason to escalate.
    return {'joint':row.joint_name,'scope':scope,'status':'BLOCKED','attempts':attempts}


def actuation_checks(loaded,d):
    reset=reset_supine(loaded,d,seed=0);reference=mj.MjData(loaded.model);mj.mj_copyData(reference,loaded.model,d)
    rows=[]
    for row in loaded.mapping:
        record=causal_joint(loaded,reference,row)
        if record['status']!='PASS':
            # Disposable free-fall fixture: same model, gravity, collision and limits.
            # Restore a full reset-constructed state, raise it BEFORE the rollout.
            fixture=mj.MjData(loaded.model);construct_pose(loaded,fixture,0,ResetSettings())
            fixture.qpos[2]+=2.
            fixture.qpos[loaded.joint('waist_pitch_joint').qpos_address]=0.
            mj.mj_forward(loaded.model,fixture)
            record['fixture']=causal_joint(loaded,fixture,row,'disposable free-fall clearance; not a reset')
        rows.append(record)
        print(json.dumps({'joint':row.joint_name,'status':record['status'],
                          'fixture':record.get('fixture',{}).get('status')}),flush=True)
    groups=[]
    for name,predicate in (('legs',lambda n:'hip' in n or 'knee' in n or 'ankle' in n),
                           ('arms',lambda n:any(t in n for t in ('shoulder','elbow','wrist'))),
                           ('waist_head',lambda n:'waist' in n or 'head' in n),('full_body',lambda n:True)):
        mj.mj_copyData(d,loaded.model,reference)
        commands=[(r,.2*ceiling(r)) for r in loaded.mapping if predicate(r.joint_name)]
        try:groups.append({'group':name,**pulse(loaded,d,commands,pulse_s=.08,neutral_s=.12)})
        except ValueError as exc:groups.append({'group':name,'status':'FAIL','error':str(exc),
                                                'evidence':getattr(exc,'evidence',None)})
    return {'status':'PASS' if all(r['status']=='PASS' or r.get('fixture',{}).get('status')=='PASS' for r in rows)
            and all(g['status']=='PASS' for g in groups) else 'FAIL','joints':rows,'groups':groups,
            'initial_reset':reset,'mapping':[asdict(r) for r in loaded.mapping],
            'motion_threshold_rad':1e-5,'ceilings_Nm':{r.joint_name:ceiling(r) for r in loaded.mapping}}


class LiveObserver:
    """MuJoCo's supported mjv/mjr API in a real GLFW desktop window.

    Rendering lives in the observer/main thread, so no second physics loop or
    shutdown race exists. Readback is from the very window buffer we swap onto
    the desktop, not an offscreen approximation of what the viewer displayed.
    """
    def __init__(self,loaded,d,output):
        import glfw
        from OpenGL import GL
        require(os.environ.get('MUJOCO_GL')=='glfw','Live demo requires a separate GLFW process')
        self.glfw=glfw;self.m,self.d=loaded.model,d;self.output=output
        self.head=loaded.joint('head_yaw_joint');self.last=None;self.frames=0
        self.closed=False;self.collision=False;self.saved=[]
        require(glfw.init(),'GLFW initialization failed')
        self.window=glfw.create_window(640,480,'X2 Step 4 - MuJoCo live physics',None,None)
        require(self.window is not None,'GLFW window/context creation failed')
        glfw.make_context_current(self.window);glfw.swap_interval(1)
        self.context=mj.MjrContext(self.m,mj.mjtFontScale.mjFONTSCALE_100)
        self.scene=mj.MjvScene(self.m,maxgeom=2000);self.camera=mj.MjvCamera();self.option=mj.MjvOption()
        self.camera.distance=2.2;self.camera.azimuth=125;self.camera.elevation=-28
        self.groups=self.m.geom_group.copy()
        self.m.geom_group[:]=[1 if g==0 else 0 if self.m.geom_contype[g] or self.m.geom_conaffinity[g] else 5
                             for g in range(self.m.ngeom)]
        glfw.set_key_callback(self.window,self.key)
        self.backend={'MUJOCO_GL':'glfw','glfw_library':glfw._glfw._name,
                      'DISPLAY':os.environ.get('DISPLAY'),'WAYLAND_DISPLAY':os.environ.get('WAYLAND_DISPLAY'),
                      'OpenGL_renderer':GL.glGetString(GL.GL_RENDERER).decode(),
                      'LIBGL_ALWAYS_SOFTWARE':os.environ.get('LIBGL_ALWAYS_SOFTWARE'),
                      'target_fps':25,'render_effects':'shadow/reflection off; visual and collision groups separated',
                      'API':'MuJoCo mjv_updateScene/mjr_render, GLFW real desktop window; main-thread renderer'}

    def key(self,window,key,scancode,action,mods):
        if action==self.glfw.PRESS:
            if key==self.glfw.KEY_C:self.collision=not self.collision
            if key==self.glfw.KEY_ESCAPE:self.glfw.set_window_should_close(window,True)

    def __call__(self,frame):
        from PIL import Image
        g=self.glfw
        require(not g.window_should_close(self.window),'Live viewer closed before demo completed')
        if self.last is not None:
            wait=.04-(time.monotonic()-self.last)
            if wait>0:time.sleep(wait)
        self.last=time.monotonic();self.camera.lookat[:]=self.d.subtree_com[1]
        if frame['phase']=='placement':self.collision=bool(frame.get('reset_number',0)%2)
        self.option.geomgroup[:]=0
        self.option.geomgroup[1]=1
        self.option.geomgroup[0 if self.collision else 5]=1
        self.option.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL]=self.collision
        self.option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT]=self.collision
        mj.mjv_updateScene(self.m,self.d,self.option,None,self.camera,mj.mjtCatBit.mjCAT_ALL,self.scene)
        self.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0
        self.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION]=0
        width,height=g.get_framebuffer_size(self.window);viewport=mj.MjrRect(0,0,width,height)
        mj.mjr_render(viewport,self.scene,self.context)
        lines=(f"{frame['phase']}\nreset {frame.get('reset_number',0)} | seed {frame.get('seed','-')}\nt={frame['time_s']:.3f}s | episode={frame.get('episode_elapsed_s',0):.3f}s",
               f"max ctrl={max(abs(frame['ctrl'])):.3f} Nm\nhead q={self.d.qpos[self.head.qpos_address]:.4f} rad\nhead dq={self.d.qvel[self.head.dof_address]:.4f} rad/s | C: hulls")
        mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,mj.mjtGridPos.mjGRID_TOPLEFT,viewport,*lines,self.context)
        # Native window readback provides inspectable evidence even when the desktop
        # compositor denies global screenshots. It is the same rendered buffer.
        if self.frames%4==0:
            rgb=np.empty((height,width,3),dtype=np.uint8)
            mj.mjr_readPixels(rgb,None,viewport,self.context)
            path=self.output/f"live-{self.frames:04d}.png"
            Image.fromarray(rgb[::-1]).save(path)
            self.saved.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                               'phase':frame['phase'],'time_s':frame['time_s']})
            require(float(rgb.std())>1.,'Invalid/black live window framebuffer; see '+str(path))
        g.swap_buffers(self.window);g.poll_events();self.frames+=1

    def close(self):
        self.context.free();self.glfw.destroy_window(self.window);self.glfw.terminate()
        self.m.geom_group[:]=self.groups;self.closed=True


def demonstration(loaded,d,observer,repeat):
    resets=[];pulses=[]
    for number in range(repeat):
        origin=None
        def observe(frame):
            observer({**frame,'seed':number,'reset_number':number,
                      'episode_elapsed_s':0. if origin is None else max(0.,frame['time_s']-origin)})
        r=reset_supine(loaded,d,seed=number,observer=observe);resets.append(r)
        origin=d.time
        row=loaded.joint('head_yaw_joint')
        for sign in (1.,-1.):pulses.append(pulse(loaded,d,[(row,ceiling(row))],sign=sign,
            observer=observe,label='head yaw '+str(sign),pulse_s=.35,neutral_s=.25))
        for name,predicate in (('legs',lambda n:'hip' in n or 'knee' in n or 'ankle' in n),
                               ('arms',lambda n:'shoulder' in n or 'elbow' in n or 'wrist' in n),
                               ('waist/head',lambda n:'waist' in n or 'head' in n),('full body',lambda n:True)):
            pulses.append(pulse(loaded,d,[(j,.2*ceiling(j)) for j in loaded.mapping if predicate(j.joint_name)],
                                observer=observe,label=name,pulse_s=.08,neutral_s=.12))
        notify(loaded.model,d,observe,'demo end',{'episode_elapsed_s':float(d.time-origin)})
    return {'status':'PASS','resets':resets,'pulses':pulses}


def record_gif(loaded,frames,output):
    """Clearly labeled replay on separate MjData; never fed back to physics."""
    require(os.environ.get('MUJOCO_GL')=='osmesa','Recording requires separate OSMesa process')
    from PIL import Image,ImageDraw
    m=loaded.model;d=mj.MjData(m);images=[];camera=mj.MjvCamera()
    head_qadr=loaded.joint('head_yaw_joint').qpos_address
    camera.distance=2.2;camera.azimuth=125;camera.elevation=-28
    with mj.Renderer(m,height=360,width=480) as renderer:
        for i,frame in enumerate(frames):
            d.qpos[:]=frame['qpos'];d.qvel[:]=frame['qvel'];d.time=frame['time_s'];d.ctrl[:]=frame['ctrl']
            mj.mj_forward(m,d);camera.lookat[:]=d.subtree_com[1]
            renderer.update_scene(d,camera=camera);im=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(im);draw.rectangle((0,0,480,34),fill='black')
            draw.text((5,3),f"PHYSICS REPLAY | reset {frame['reset_number']} | {frame['phase']} | t={d.time:.3f}s",fill='white')
            draw.text((5,18),f"ctrl max {max(abs(d.ctrl)):.3f} Nm | head q {d.qpos[head_qadr]:.4f} rad",fill='white')
            images.append(im)
    path=output/'demonstration.gif'
    images[0].save(path,save_all=True,append_images=images[1:],duration=40,loop=0)
    # A small set of exact trajectory frames aids inspection without fabricating curves.
    for i in sorted(set([0,len(images)//4,len(images)//2,3*len(images)//4,len(images)-1])):
        images[i].save(output/f'frame-{i:04d}.png')
    return {'status':'GENERATED_NOT_REVIEWED','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'backend':'OSMesa','fps':25,'frames':len(images),'dropped_frames':0,
            'coverage':{'first':frames[0]['time_s'],'last':frames[-1]['time_s'],
                        'resets':sorted({f['reset_number'] for f in frames})},
            'method':'Separate-data replay of copied actual physics callback states; includes reset placement/settling/hold/actuation'}


def run(mode,asset_repo,output,repeat=2):
    require(1<=repeat<=20,'Repeat must be 1..20')
    output=Path(output).resolve()/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6])
    output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[3];loaded=load_effective_model(asset_repo);m=loaded.model;d=mj.MjData(m)
    report={'run_id':output.name,'mode':mode,'verdict':'PARTIAL/BLOCKED','run_completed':False,'runtime':identity(project),
            'source':provenance(loaded.asset_repo),'model_fingerprint':fingerprint(loaded),
            'model_signature':model_signature(m),'criteria':asdict(ResetSettings()),'overrides':loaded.overrides,
            'RESET':{'status':'NOT_TESTED'},'ACTUATION':{'status':'NOT_TESTED'},
            'LIVE_VIEWER':{'status':'NOT_TESTED'},'RECORDING':{'status':'NOT_TESTED'}}
    write_json(output/'report.json',report);live=None;start=time.monotonic()
    try:
        if mode=='step4':
            report['RESET']=reset_batch(loaded,d);write_json(output/'report.json',report)
            report['ACTUATION']=actuation_checks(loaded,d)
        else:
            frames=[]
            if mode=='step4-live':live=LiveObserver(loaded,d,output)
            def observer(frame):
                frames.append(frame)
                if live:live(frame)
            demo=demonstration(loaded,d,observer,repeat)
            report['demo']=demo
            write_json(output/'trajectory.json',frames)
            if live:
                live.close();report['LIVE_VIEWER']={'status':'RAN_NOT_REVIEWED','completed':True,
                    'wall_s':time.monotonic()-start,'frames':live.frames,'window_frames':live.saved,**live.backend}
            else:report['RECORDING']=record_gif(loaded,frames,output)
        report['run_completed']=True
    except Exception as exc:
        report.update(error=repr(exc),run_completed=False)
        area={'step4-live':'LIVE_VIEWER','step4-record':'RECORDING'}.get(mode)
        if area:
            report[area]={'status':'FAIL','error':repr(exc)}
            if live:report[area].update(backend=live.backend,window_frames=live.saved)
        if isinstance(exc,ResetFailure):report['failed_reset']=exc.evidence
        if isinstance(exc,PulseFailure):report['failed_pulse']=exc.evidence
        if live and not live.closed:
            try:live.close()
            except Exception as close_error:report['close_error']=repr(close_error)
        print(repr(exc),file=sys.stderr)
    finally:
        report['wall_s']=time.monotonic()-start
        if report['model_signature']!=model_signature(m):
            report.update(run_completed=False,error='Unexpected model change')
        write_json(output/'report.json',report)
        print(json.dumps({'report':str(output/'report.json'),'completed':report.get('run_completed'),
                          'RESET':report['RESET']['status'],'ACTUATION':report['ACTUATION']['status'],
                          'LIVE_VIEWER':report['LIVE_VIEWER']['status'],'RECORDING':report['RECORDING']['status']}),flush=True)
    if mode=='step4':return 0 if report['run_completed'] and report['RESET']['status']==report['ACTUATION']['status']=='PASS' else 1
    return 0 if report.get('run_completed') else 1
