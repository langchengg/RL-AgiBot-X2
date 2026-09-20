"""Step 5 bounded calibration / acceptance. Standing fixture, NOT recovery policy."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import time
import uuid
import mujoco as mj
import numpy as np
from .model import load_effective_model, require
from .reset import (ResetSettings, reset_supine, checked_step, floor_geometry,
                    plane_gap, initial_geometry, model_signature)
from .success import (StandingContext, SuccessSettings, SuccessTracker, measure_standing,
                      standing_failures, MODEL_FINGERPRINT, CALIBRATED_SETTINGS)
from .model_audit import fingerprint, identity, json_value


def write_json(path, value):
    invalid = []
    value = json_value(value, invalid)
    require(not invalid, 'Nonfinite diagnostic data: '+str(invalid))
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def standing_pose(loaded, knee=.10):
    pose = {r.joint_name: 0. for r in loaded.mapping}
    for side, sign in [('left', 1), ('right', -1)]:
        for name, angle in {'hip_pitch': -knee/2, 'knee': knee, 'ankle_pitch': -knee/2,
                            'shoulder_pitch': -.15, 'shoulder_roll': sign*.15,
                            'elbow': -.30}.items():
            pose[f'{side}_{name}_joint'] = angle
    return pose


def gains(loaded, scale=2.):
    kp=[];kd=[]
    for r in loaded.mapping:
        n=r.joint_name
        p,d = ((300.,12.) if 'hip' in n or 'knee' in n else (150.,8.) if 'ankle' in n else
               (150.,8.) if 'waist' in n else (5.,.3) if 'head' in n or 'wrist' in n else (50.,3.))
        kp.append(p*scale);kd.append(d*np.sqrt(scale))
    return np.array(kp),np.array(kd)


def place(loaded, pose, yaw=0., pitch=0., lift=0.):
    """Initialization only; lowest actual collision hull/proxy gets 2 mm clearance."""
    m=loaded.model;d=mj.MjData(m)
    require(set(pose)=={r.joint_name for r in loaded.mapping}, 'Incomplete fixture pose')
    for r in loaded.mapping:
        require(r.position_range[0]<=pose[r.joint_name]<=r.position_range[1], 'Invalid fixture target')
        d.qpos[r.qpos_address]=pose[r.joint_name]
    qy=np.zeros(4);qp=np.zeros(4)
    mj.mju_axisAngle2Quat(qy,np.array([0.,0.,1.]),yaw)
    mj.mju_axisAngle2Quat(qp,np.array([0.,1.,0.]),pitch)
    mj.mju_mulQuat(d.qpos[3:7],qy,qp)
    d.qpos[:3]=0.;mj.mj_forward(m,d)
    f,n,o,gs=floor_geometry(m,d)
    d.qpos[:3]+=n*(.002+lift-min(plane_gap(m,d,g,n,o) for g in gs))
    mj.mj_forward(m,d)
    geometry=initial_geometry(m,d,ResetSettings())
    return d,geometry


def pd_control(context,d,pose,kp,kd):
    target=np.array([pose[r.joint_name] for r in context.loaded.mapping])
    torque=kp*(target-d.qpos[context.qadr])-kd*d.qvel[context.vadr]
    d.ctrl[context.ctrladr]=np.clip(torque,context.efforts[:,0],context.efforts[:,1])
    return float(np.max(np.abs(torque)/np.maximum(abs(context.efforts[:,0]),abs(context.efforts[:,1]))))


def rollout(context,d,pose,kp,kd,seconds,settings,trace_path=None,tracker=None,witness=None):
    tracker=tracker or SuccessTracker(settings,context.dt)
    if tracker.origin is None:tracker.reset(d.time)
    rows=[];snapshots={};deadline=time.monotonic()+45.;signature=model_signature(context.loaded.model)
    start=d.time;peak_saturation=0.;handle=open(trace_path,'w') if trace_path else None
    try:
        for k in range(round(seconds/context.dt)):
            peak_saturation=max(peak_saturation,pd_control(context,d,pose,kp,kd))
            checked_step(context.loaded,d,deadline)
            v=measure_standing(context,d);result=tracker.update(v)
            row={**v,'height_ratio':v['pelvis_height_m']/settings.h_ref_m,**result}
            rows.append(row)
            if handle:handle.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            if result['invalid_reason']:raise ValueError(result['invalid_reason'])
            if result['standing_held'] and 'held' not in snapshots: snapshots['held']=snapshot(d,row)
            if result['instant_standing_ok'] and 'instant' not in snapshots: snapshots['instant']=snapshot(d,row)
            if witness is not None and witness(row) and 'witness' not in snapshots: snapshots['witness']=snapshot(d,row)
            snapshots['final']=snapshot(d,row)
        require(signature==model_signature(context.loaded.model), 'Model changed during rollout')
    except Exception as exc:
        tracker.invalidate(str(exc));raise
    finally:
        if handle:handle.close()
    return rows,snapshots,{'elapsed_s':float(d.time-start),'max_unclipped_effort_fraction':peak_saturation}


def snapshot(d,row):
    return {'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'ctrl':d.ctrl.copy(),'time_s':float(d.time),'measurement':row}


CALIBRATION = {
    'reference': 'candidate 1, 2026-09-20, independent upright posture; median pelvis height at t=6..8 s',
    'development_evidence': 'audit-output/step5/development-103134',
    'search_limit': {'candidates': 6, 'sim_s_per_candidate': 8., 'wall_s_per_candidate': 45.},
    'attempted': [{'candidate': 0, 'gain_scale': 1., 'max_standing_s': .345,
                   'result': 'fell; insufficient posture stiffness, final height .10169 m, tilt 88.67 deg'},
                  {'candidate': 1, 'gain_scale': 2., 'max_standing_s': 7.938, 'result': 'stable; selected'}],
    'stable_segment_s': [6., 8.],
    'pelvis_height_range_m': [.6724937999650762, .672495805450893],
    'tilt_range_deg': [4.86943614241918, 4.873249218885313],
    'other_contacts': 0, 'other_force_N': 0.,
    'other_tolerance_basis': 'Zero contacts/force in real standing; 1e-5 W numerical allowance, '
                             '500x below candidate .005 W. Does not prove mathematical zero support.',
    'steady_geometry_basis': 'Observed floor penetration .46233 mm, self penetration 0, joint excess 0; '
                             '1 mm floor / .1 mm self / .1 mrad joint tolerance, distinct from impact guards.',
}


def code_identity():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('model.py','reset.py','success.py','step5.py','diagnostics.py')}


def summary(rows,info):
    scalar=['pelvis_height_m','height_ratio','tilt_deg','left_weight','right_weight','other_weight',
            'other_force_N','base_linear_m_s','com_linear_m_s','base_angular_rad_s',
            'torso_angular_rad_s','joint_rad_s','floor_penetration_m','self_penetration_m','joint_limit_rad']
    qualified=[r for r in rows if r['instant_standing_ok'] and not r['failures']]
    # Reconstruct the longest continuous window (not a union of disjoint good samples).
    longest=max(rows,key=lambda r:r['stable_duration_s'])
    window=[r for r in rows if longest['time_s']-longest['stable_duration_s']-1e-10 <= r['time_s'] <= longest['time_s']]
    return {**info,'samples':len(rows),'ever_held':any(r['standing_held'] for r in rows),
            'ever_recovery':any(r['recovery_success'] for r in rows),
            'qualified_samples':len(qualified),'max_hold_s':longest['stable_duration_s'],
            'window_s':[window[0]['time_s'],window[-1]['time_s']],
            'window_ranges':{key:[min(r[key] for r in window),max(r[key] for r in window)] for key in scalar},
            'window_drift_max_m':{key:max(r['drift_m'][key] for r in window) for key in ('pelvis','left','right')},
            'final_failures':rows[-1]['failures']}


def negative_fixture(loaded,name):
    from .model_audit import contact_pose
    pose=standing_pose(loaded);kwargs={};seconds=.8
    if name=='sitting':
        for side in ('left','right'):
            pose[side+'_hip_pitch_joint']=-1.5
            pose[side+'_knee_joint']=.1
            pose[side+'_ankle_pitch_joint']=.35
        witness=lambda r:r['pelvis_height_m']<.3 and any(c['body']=='pelvis' for c in r['support'])
    elif name=='kneeling':
        for side in ('left','right'):
            pose[side+'_hip_pitch_joint']=-.15
            pose[side+'_knee_joint']=2.2
            pose[side+'_ankle_pitch_joint']=-.75
        # The actual thigh hull's distal knee cap is lower than the shin hull.
        witness=lambda r:(r['pelvis_height_m']<.5 and r['tilt_deg']<15. and
            any(c['body'] in ('left_hip_yaw_link','right_hip_yaw_link','left_knee_link','right_knee_link')
                and c['norm_N']>1. for c in r['support']))
    elif name=='arm_support':
        d,geometry=contact_pose(loaded,'left_arm')
        pose={r.joint_name:float(d.qpos[r.qpos_address]) for r in loaded.mapping}
        witness=lambda r:any(c['body'] in ('left_elbow_link','left_wrist_roll_link','left_wrist_pitch_link')
                            and c['norm_N']>1. for c in r['support'])
    elif name=='single_foot':
        pose.update(right_hip_pitch_joint=-.15,right_knee_joint=.30,right_ankle_pitch_joint=-.15)
        witness=lambda r:r['left_weight']>.2 and r['right_weight']<.05 and r['pelvis_height_m']>.60
    elif name=='airborne':
        kwargs['lift']=.15;seconds=.01
        witness=lambda r:(r['left_weight']==0. and r['right_weight']==0. and r['height_ratio']>.9
                          and r['tilt_deg']<15. and r['base_linear_m_s']<.1)
    elif name=='brief':
        seconds=1.;witness=lambda r:r['instant_standing_ok'] and 0.<r['stable_duration_s']<2.
    else:raise ValueError('Unknown negative fixture '+name)
    if name!='arm_support':d,geometry=place(loaded,pose,**kwargs)
    return d,pose,geometry,seconds,witness


def render_evidence(loaded,snapshots,output):
    """Actual-state replay on separate model/data; never feeds back to rollouts."""
    from PIL import Image,ImageDraw
    require(os.environ.get('MUJOCO_GL')=='osmesa','Set MUJOCO_GL=osmesa before import')
    records=[]
    for collision in (False,True):
        m=copy.copy(loaded.model);d=mj.MjData(m);option=mj.MjvOption()
        if collision:
            for g in range(m.ngeom):
                m.geom_group[g]=0 if g==m.geom('floor').id or m.geom_contype[g] or m.geom_conaffinity[g] else 5
            option.geomgroup[:]=0;option.geomgroup[0]=1
            option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT]=True
            option.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE]=True
            option.flags[mj.mjtVisFlag.mjVIS_CONVEXHULL]=True
        with mj.Renderer(m,height=360,width=480) as renderer:
            for name,snap in snapshots.items():
                d.qpos[:]=snap['qpos'];d.qvel[:]=snap['qvel'];d.ctrl[:]=snap['ctrl'];d.time=snap['time_s']
                mj.mj_forward(m,d)
                camera=mj.MjvCamera();camera.lookat[:]=d.subtree_com[m.body('pelvis').id]
                camera.distance=2.1;camera.azimuth=125;camera.elevation=-20
                renderer.update_scene(d,camera=camera,scene_option=option)
                pixels=renderer.render();require(float(pixels.std())>1.,'Invalid render')
                im=Image.fromarray(pixels);draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,480,34),fill='black')
                draw.text((4,3),f'ACTUAL STATE REPLAY | {name} | t={d.time:.3f}s',fill='white')
                draw.text((4,18),f'{"COLLISION" if collision else "NORMAL"} | standing fixture, NOT recovery',fill='white')
                file=output/f'{name}-{"collision" if collision else "normal"}.png';im.save(file)
                records.append({'path':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                                'case':name,'time_s':float(d.time),'view':'collision' if collision else 'normal'})
    return records


def run(asset_repo,output):
    output=Path(output).resolve()/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6])
    output.mkdir(parents=True,exist_ok=False)
    report={'run_id':output.name,'verdict':'INCOMPLETE','run_completed':False,'checks':{},
            'visual_review':'PENDING','command':sys.argv,'implementation_sha256':code_identity(),
            'settings':asdict(CALIBRATED_SETTINGS),'calibration':CALIBRATION,
            'recovery_training':'NOT RUN','five_episode_recovery_evaluation':'NOT EVALUATED'}
    path=output/'report.json';write_json(path,report)
    print('Step 5 evidence: '+str(output),flush=True)
    try:
        project=Path(__file__).resolve().parents[3]
        report['runtime']=identity(project)
        # No unittest skip can accidentally make the formal command pass.
        import unittest
        with (output/'success-tests.log').open('w') as stream:
            suite=unittest.defaultTestLoader.discover(str(project/'src/x2_recovery/test'),pattern='test_success.py')
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report['checks']['tests']={'status':'PASS' if result.wasSuccessful() and not result.skipped and result.testsRun>=27 else 'FAIL',
                                    'count':result.testsRun,'skipped':len(result.skipped)}
        require(report['checks']['tests']['status']=='PASS','Required tests failed/skipped')
        snapshots={}
        for name,yaw in [('standing',0.),('yaw',1.1),('repeat',0.)]:
            loaded=load_effective_model(asset_repo);ctx=StandingContext(loaded);m=loaded.model
            require(fingerprint(loaded)==MODEL_FINGERPRINT,'Stale calibration fingerprint')
            pose=standing_pose(loaded);kp,kd=gains(loaded);d,geometry=place(loaded,pose,yaw=yaw)
            report['model_fingerprint']=fingerprint(loaded)
            report['model_signature']=model_signature(m)
            report['weight_N']=ctx.weight_N
            report['mapping']={'floor':ctx.floor,'pelvis':ctx.pelvis,'torso':ctx.torso,
                               'foot_bodies':ctx.foot_bodies,'foot_geoms':[sorted(gs) for gs in ctx.feet],
                               'reference_points':'ankle_roll_link body origins, fixed to each foot',
                               'torso_longitudinal_axis':'local +Z, source neutral axes verified'}
            report['controller']={'pose_rad':pose,'kp_Nm_rad':kp,'kd_Nm_s_rad':kd,'period_s':ctx.dt,
                                  'clipping':'effective mapping effort_range','initial_clearance_m':.002,
                                  'initial_qvel':'zero, initialization only','free_base':True,'gravity':m.opt.gravity}
            rows,snaps,info=rollout(ctx,d,pose,kp,kd,4.,CALIBRATED_SETTINGS,output/(name+'.jsonl'))
            entry=summary(rows,info)
            entry.update(yaw_rad=yaw,initial_geometry=geometry,status='PASS' if entry['ever_held'] and not entry['ever_recovery'] else 'FAIL')
            # Independent engine force aggregation includes all contacts (self forces cancel).
            mj.mj_rnePostConstraint(m,d)
            measured=sum((np.array(c['force_world_N']) for c in rows[-1]['support']),start=np.zeros(3))
            engine=np.sum(d.cfrc_ext[list(ctx.robot_bodies),3:],axis=0)
            entry['contact_balance_error_N']=float(np.linalg.norm(measured-engine))
            require(entry['contact_balance_error_N']<1e-8,'Contact world-force mismatch')
            report['checks'][name]=entry
            require(entry['status']=='PASS',name+' has no full physical standing window')
            snapshots[name]=snaps['final']
            write_json(path,report);print(name+': PASS',flush=True)
        # A real reset's returned residual state/time establishes the origin, never the fixture.
        d=mj.MjData(m);reset=reset_supine(loaded,d,seed=0)
        tr=SuccessTracker(CALIBRATED_SETTINGS,ctx.dt);tr.reset_from_supine(ctx,d,reset)
        pose={r.joint_name:float(d.qpos[r.qpos_address]) for r in loaded.mapping}
        rows,snaps,info=rollout(ctx,d,pose,np.zeros(m.nu),np.zeros(m.nu),.1,CALIBRATED_SETTINGS,
                              output/'supine.jsonl',tracker=tr)
        require(all(not r['instant_standing_ok'] and not r['recovery_success'] for r in rows),'Supine false positive')
        require(all('height' in r['failures'] and 'torso_tilt' in r['failures'] and 'other_weight' in r['failures'] for r in rows),
                'Supine expected measurements missing')
        report['checks']['supine']={'status':'PASS',**summary(rows,info),'handoff':reset,
                                   'origin_time_s':tr.start,'witness':rows[0]}
        snapshots['supine']=snaps['final']
        for name in ('sitting','kneeling','arm_support','single_foot','airborne','brief'):
            d,pose,geometry,seconds,witness=negative_fixture(loaded,name)
            kp,kd=gains(loaded)
            rows,snaps,info=rollout(ctx,d,pose,kp,kd,seconds,CALIBRATED_SETTINGS,output/(name+'.jsonl'),witness=witness)
            entry=summary(rows,info)
            entry.update(status='NOT_EXERCISED' if 'witness' not in snaps else 'FAIL',initial_geometry=geometry,
                         pose_rad=pose,witness=snaps.get('witness',{}).get('measurement'))
            if 'witness' in snaps and not entry['ever_held'] and not entry['ever_recovery']:
                required={'sitting':'height','kneeling':'other_weight','arm_support':'other_weight',
                          'single_foot':'right_support','airborne':'feet_total','brief':None}[name]
                if required is None or required in entry['witness']['failures']:entry['status']='PASS'
            if name=='kneeling' and 'witness' in snaps:
                entry['knee_joint_angles_rad']={side:float(snaps['witness']['qpos'][loaded.joint(side+'_knee_joint').qpos_address])
                                                 for side in ('left','right')}
                entry['contact_note']='Kneeling distal thigh/knee caps are hip_yaw_link collision hulls in this model.'
                require(min(entry['knee_joint_angles_rad'].values())>2.,'Not a kneeling posture')
            report['checks'][name]=entry;write_json(path,report)
            require(entry['status']=='PASS',name+': '+entry['status'])
            if name!='brief':snapshots[name]=snaps['witness']
            print(name+': PASS',flush=True)
        require(fingerprint(loaded)==MODEL_FINGERPRINT,'Physics changed during acceptance')
        write_json(output/'snapshots.json',snapshots)
        report['images']=render_evidence(loaded,snapshots,output)
        report.update(run_completed=True,automated_checks='PASS',verdict='AWAITING_VISUAL_REVIEW',exit_code=1)
        write_json(path,report)
        print('Automated checks PASS; inspect images then use step5-review. '+str(path),flush=True)
        return 1
    except Exception as exc:
        report.update(error=str(exc),exit_code=1)
        write_json(path,report);print('Step 5 INCOMPLETE: '+str(exc),flush=True)
        return 1


def review(output,observations):
    """Close out an existing run after actual inspection; no simulation rerun.

    observations JSON maps every image filename to a specific human/agent visual
    observation. This records inspection, not an automated image-understanding claim.
    """
    directory=Path(output).resolve();path=directory/'report.json'
    report=json.loads(path.read_text());notes=json.loads(Path(observations).read_text())
    require(report.get('run_completed') and report.get('automated_checks')=='PASS', 'Physical checks incomplete')
    require(all(c['status']=='PASS' for c in report['checks'].values()),'Failed/unexercised check')
    require(report['implementation_sha256']==code_identity() and report['settings']==asdict(CALIBRATED_SETTINGS),
            'Code/config changed since run; results stale')
    require(report['model_fingerprint']==fingerprint(load_effective_model()),'Model changed; results stale')
    require(set(notes)=={i['path'] for i in report['images']},'Inspect every normal/collision image')
    for image in report['images']:
        require(hashlib.sha256((directory/image['path']).read_bytes()).hexdigest()==image['sha256'], 'Image changed')
        require(isinstance(notes[image['path']],str) and len(notes[image['path']].strip())>20,'Specific observation required')
    report.update(visual_review=notes,verdict='COMPLETE',review_command=sys.argv,review_exit_code=0)
    write_json(path,report);print('Step 5 COMPLETE: '+str(path))
    return 0
