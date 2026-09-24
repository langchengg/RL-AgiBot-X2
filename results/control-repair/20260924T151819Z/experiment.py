"""Generate bounded coupled-reference candidates; no physics execution/training.

Example (repository root, existing environment):
PYTHONPATH=src/x2_recovery .venv/bin/python results/control-repair/20260924T151819Z/experiment.py --output /tmp/x2-repair-g0.json
Continuation: --generation 1 --previous-results RUN/results.csv [--previous-plan RUN/plan.json]
Explicit --center / --sigma accept JSON arrays, comma-separated values, or a JSON
file with mean/sigma keys. All target changes are validated; none are clipped.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from x2_recovery.env import ControlConfig
from x2_recovery.model import URDF, SOURCE_HASHES

NAMES = ['early_left_ankle_relief_rad','early_right_ankle_relief_rad',
         'early_waist_negative_rad','early_time_scale','push_time_scale',
         'landing_left_ankle_relief_rad','landing_right_ankle_relief_rad',
         'landing_left_knee_flexion_rad','landing_right_knee_flexion_rad']
LOW = np.array([0.,0.,0.,.90,.90,0.,0.,0.,0.])
HIGH = np.array([.10,.10,.10,1.25,1.25,.16,.12,.18,.18])
CENTER = np.array([.030,.030,.035,1.08,1.06,.045,.025,.040,.035])
SIGMA = np.array([.020,.020,.025,.075,.075,.035,.025,.035,.030])
RNG_SEED = 26092471


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def rotation(axis, angle):
    a=np.asarray(axis,float);a=a/np.linalg.norm(a);x,y,z=a
    k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(angle)*k+(1.-np.cos(angle))*(k@k)


class StaticAxes:
    """URDF orientation FK only: no MuJoCo data, mj_forward or mj_step."""
    def __init__(self,path):
        root=ET.parse(path).getroot();self.joints=[]
        children={j.find('child').get('link') for j in root.findall('joint')}
        self.root=next(link.get('name') for link in root.findall('link') if link.get('name') not in children)
        for joint in root.findall('joint'):
            origin=joint.find('origin');rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            matrix=rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0])
            axis=joint.find('axis');a=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0',sep=' ')
            self.joints.append((joint.get('name'),joint.find('parent').get('link'),joint.find('child').get('link'),matrix,a,joint.get('type')))
    def axes(self,q):
        links={self.root:np.eye(3)};axes={};todo=list(self.joints)
        while todo:
            before=len(todo)
            for entry in todo[:]:
                name,parent,child,origin,axis,kind=entry
                if parent not in links:continue
                basis=links[parent]@origin;axes[name]=basis@axis
                links[child]=basis@(rotation(axis,q.get(name,0.)) if kind!='fixed' else np.eye(3))
                todo.remove(entry)
            if len(todo)==before:raise ValueError('Disconnected/cyclic URDF joint graph')
        return axes


def smooth(u):
    # Scalar construction of the reference basis only, not clipping candidates.
    if u<=0:return 0.
    if u>=1:return 1.
    return float(u*u*u*(10.+u*(-15.+6.*u)))


def bump(t,a,b,c,d):
    if t<=a or t>=d:return 0.
    if t<b:return smooth((t-a)/(b-a))
    if t<=c:return 1.
    return 1.-smooth((t-c)/(d-c))


def warp(t,early,push):
    # Piecewise affine old-reference time -> new physical-reference time.
    begin=.7;boundary=1.577870;end=1.94
    return float(t+(early-1.)*max(0.,min(t,boundary)-begin)
                 +(push-1.)*max(0.,min(t,end)-boundary))


def parse_vector(text,key,size=len(NAMES)):
    if text is None:return None
    try:path=Path(text);exists=path.is_file()
    except OSError:exists=False
    if exists:value=json.loads(path.read_text())
    elif text.lstrip().startswith('['):value=json.loads(text)
    else:value=[float(x) for x in text.split(',')]
    if isinstance(value,dict):value=value[key]
    array=np.asarray(value,float)
    if array.shape!=(size,) or not np.isfinite(array).all():raise ValueError(f'Expected {size} finite '+key+' values')
    return array


def target_bounds(config,bounds,original):
    for i,(_,_,target) in enumerate(config['reference_stages']):
        for joint,value in target.items():
            lower,upper=bounds[joint]
            if not np.isfinite(value):raise ValueError('Nonfinite target')
            if lower<=value<=upper:continue
            # Retain an unchanged original boundary float only when it is within
            # the existing verified model/reset export precision tolerance.
            before=original['reference_stages'][i][2].get(joint)
            if value==before and lower-1e-6<=value<=upper+1e-6:continue
            raise ValueError(f'stage{i}:{joint} target {value} outside [{lower},{upper}]')


def prepare_basis(original,kinematics):
    end_times=np.cumsum([stage[1] for stage in original['reference_stages']])
    matrices={};conditions={}
    for index,t in enumerate(end_times):
        if bump(t,1.94,2.18,2.24,2.64)==0:continue
        q=original['reference_stages'][index][2];axes=kinematics.axes(q)
        for side in ('left','right'):
            hip=[side+'_'+name+'_joint' for name in ('hip_pitch','hip_roll','hip_yaw')]
            h=np.column_stack([axes[name] for name in hip])
            condition=float(np.linalg.cond(h))
            if condition>3.:raise ValueError('Landing hip Jacobian ill-conditioned')
            known=np.column_stack([axes[side+'_ankle_pitch_joint'],axes[side+'_knee_joint']])
            matrices[index,side]=-np.linalg.solve(h,known);conditions[f'{index}:{side}']=condition
    return end_times,matrices,conditions


def candidate(parameters,original,bounds,basis):
    p=np.asarray(parameters,float)
    if p.shape!=LOW.shape or not np.isfinite(p).all() or np.any(p<LOW) or np.any(p>HIGH):
        raise ValueError('Parameter outside preregistered bounds')
    config=copy.deepcopy(original);times,matrices,_=basis;previous_time=0.;changed_targets=0
    for index,(stage,t) in enumerate(zip(config['reference_stages'],times)):
        name,_,q=stage
        early=bump(float(t),.4,.7,1.577870,1.94)
        waist=bump(float(t),.7877192982456141,.9631578947368421,1.1385964912280702,1.4017543859649122)
        landing=bump(float(t),1.94,2.18,2.24,2.64)
        if q:
            for side,relief in [('left',p[0]),('right',p[1])]:
                q[side+'_ankle_pitch_joint']+=float(relief*early)
                q[side+'_knee_joint']-=float(relief*early)
            q['waist_pitch_joint']-=float(p[2]*waist)
            for side,a,c in [('left',p[5],p[7]),('right',p[6],p[8])]:
                if landing:
                    q[side+'_ankle_pitch_joint']+=float(a*landing)
                    q[side+'_knee_joint']+=float(c*landing)
                    delta=matrices[index,side]@np.array([a*landing,c*landing])
                    if np.max(abs(delta))>.40:raise ValueError('Landing hip compensation exceeds0.40rad bound')
                    for joint,value in zip(('hip_pitch','hip_roll','hip_yaw'),delta):q[side+'_'+joint+'_joint']+=float(value)
            changed_targets+=sum(value!=original['reference_stages'][index][2][joint] for joint,value in q.items())
        new_time=warp(float(t),p[3],p[4]);duration=new_time-previous_time
        if duration<=0:raise ValueError('Nonpositive warped stage duration')
        config['reference_stages'][index]=[name,float(duration),q];previous_time=new_time
    config['head_reference']=[[warp(float(t),p[3],p[4]),value] for t,value in original['head_reference']]
    config['residual_start_s']=warp(original['residual_start_s'],p[3],p[4])
    full=warp(original['residual_start_s']+original['residual_ramp_s'],p[3],p[4])
    config['residual_ramp_s']=full-config['residual_start_s']
    target_bounds(config,bounds,original)
    ControlConfig.from_dict(config)
    changed=[key for key in config if config[key]!=original[key]]
    allowed={'reference_stages','head_reference','residual_start_s','residual_ramp_s'}
    if not set(changed)<=allowed:raise ValueError('Unexpected controller field change')
    return config,dict(fields=changed,changed_target_values=changed_targets,
        reference_end_s=previous_time,residual_start_s=config['residual_start_s'],
        residual_full_s=full,head_reference_end_s=config['head_reference'][-1][0])


def read_previous(results_path,plan_path):
    result_file=Path(results_path).resolve();plan_file=Path(plan_path).resolve() if plan_path else result_file.parent/'plan.json'
    old=json.loads(plan_file.read_text());rows=list(csv.DictReader(result_file.open()))
    if old.get('parameter_space',{}).get('names')!=NAMES:raise ValueError('CEM continuation requires the original parameter basis; this plan uses a different explicit family')
    def truth(value):return str(value).lower() in ('true','1')
    eligible=[row for row in rows if row['status']=='COMPLETE' and
              (truth(row.get('feasible')) or truth(row.get('infeasible_eligible')))]
    if len(eligible)<2:raise ValueError('Fewer than two eligible complete candidates; choose an explicit center/sigma instead of selecting early aborts')
    eligible.sort(key=lambda row:(truth(row.get('feasible')),
        -float(row['sim_seconds']) if truth(row.get('feasible')) else float(row['infeasible_score'])),reverse=True)
    chosen=eligible[:min(6,max(2,len(eligible)//4))]
    vectors=np.array([old['candidates'][int(row['index'])]['parameters'] for row in chosen])
    mean=vectors.mean(axis=0);sigma=np.maximum(vectors.std(axis=0),.035*(HIGH-LOW))
    sigma=np.minimum(sigma,.30*(HIGH-LOW))
    return mean,sigma,old,dict(results=str(result_file),results_sha256=sha(result_file),plan=str(plan_file),
        plan_sha256=sha(plan_file),elite_names=[row['name'] for row in chosen],eligible_count=len(eligible))



def local_direction_population():
    """32 fixed small deviations from original; no result-dependent selection."""
    base=np.array([0.,0.,0.,1.,1.,0.,0.,0.,0.])
    result=[]
    early=[(.0015,.0015,0.),(.003,.003,0.),(.006,.006,0.),
           (0.,0.,.002),(0.,0.,.004),(0.,0.,.008),
           (.003,.0015,.002),(.0015,.003,.002)]
    for values in early:
        p=base.copy();p[:3]=values;result.append(('early-only',p))
    landing=[(.001,0.,0.,0.),(0.,.001,0.,0.),(0.,0.,.0015,.0015),
             (.002,.001,.0015,.0015),(.004,.002,0.,0.),
             (.001,.001,.004,.004),(0.,0.,.004,0.),(0.,0.,0.,.004)]
    for values in landing:
        p=base.copy();p[5:]=values;result.append(('landing-only',p))
    for coordinate in (3,4):
        for offset in (-.005,.005,-.01,.01):
            p=base.copy();p[coordinate]+=offset;result.append(('time-only',p))
    for scale in (.015,.03,.05,.075):
        result.append(('weak-coupled',base+scale*(CENTER-base)))
    weak=[(.002,.001,.001,1.002,.998,.001,.0005,.001,.001),
          (.001,.002,.001,.998,1.002,.0005,.001,.001,.001),
          (.0015,.0015,.002,1.003,1.003,.001,.001,.0015,.0015),
          (.0015,.0015,.002,.997,.997,.001,.001,.0015,.0015)]
    result.extend(('weak-coupled',np.array(values)) for values in weak)
    assert len(result)==32
    return result



PREFIX_NAMES=['initial_ramp_scale','early_time_scale','waist_toward_reset_mix',
              'left_hip_return_scale','right_hip_return_scale',
              'left_knee_progress_scale','right_knee_progress_scale',
              'hip_pitch_delay_s','knee_delay_s']
PREFIX_LOW=np.array([1.,1.,0.,.6,.6,.5,.5,-.04,-.04])
PREFIX_HIGH=np.array([2.,2.2,.8,1.,1.,1.,1.,.04,.04])


def prefix_population(original,reset_waist):
    base=np.array([1.,1.,0.,1.,1.,1.,1.,0.,0.]);trough=original['reference_stages'][1][2]['waist_pitch_joint']
    def point(initial=1.,early=1.,waist=None,hip=1.,knee=1.,hip_delay=0.,knee_delay=0.):
        p=base.copy();p[0:2]=initial,early
        p[2]=0. if waist is None else (waist-trough)/(reset_waist-trough)
        p[3:5]=hip;p[5:7]=knee;p[7:9]=hip_delay,knee_delay
        return p
    rows=[]
    for initial in (1.4,1.6,1.75,2.):
        for waist in (None,-.24):rows.append(('initial-waist',point(initial=initial,waist=waist)))
    for knee in (.6,.7):
        for hip in (.7,.85):
            for initial in (1.,1.75):rows.append(('leg-drive',point(initial=initial,hip=hip,knee=knee)))
    for early in (1.4,1.6,1.9,2.2):rows.append(('timing-waist',point(early=early)))
    for waist in (-.15,0.,.15):rows.append(('timing-waist',point(waist=waist)))
    rows.append(('timing-waist',point(initial=1.75,early=1.6,waist=-.15)))
    for initial in (1.6,1.75):
        for knee in (.6,.7):
            for hip_delay,knee_delay in ((.02,0.),(0.,.02)):
                rows.append(('coupled-phase',point(initial=initial,waist=-.24,hip=.85,knee=knee,
                                                  hip_delay=hip_delay,knee_delay=knee_delay)))
    assert len(rows)==32
    return rows


def prefix_candidate(p,original,bounds,basis,reset_angles):
    p=np.asarray(p,float)
    if p.shape!=PREFIX_LOW.shape or not np.isfinite(p).all() or np.any(p<PREFIX_LOW) or np.any(p>PREFIX_HIGH):
        raise ValueError('Prefix-drive parameter out of bounds')
    times=basis[0];boundary=float(times[12]);anchor_time=float(times[2]);release_time=float(times[6])
    original_targets=[];carry=dict(reset_angles)
    for _,_,q in original['reference_stages']:
        carry.update(q);original_targets.append(dict(carry))
    def warp_prefix(t):
        return float(t+(p[0]-1.)*max(0.,min(t,.7)-.4)
                      +(p[1]-1.)*max(0.,min(t,boundary)-.7))
    def sample(joint,t):
        return float(np.interp(t,times,[q[joint] for q in original_targets]))
    config=copy.deepcopy(original);previous=0.;changed=0
    for index,(stage,t) in enumerate(zip(config['reference_stages'],times)):
        name,old_duration,q=stage
        if q:
            wphase=bump(float(t),.7,.8754385964912281,1.2263157894736842,boundary)
            # Amplitude is relative to the duplicated stage2 anchor. Only the
            # positive return/flexion excursion is attenuated; initial negative
            # hip dip remains. Release continuously to the exact old suffix.
            for side,hip_scale,knee_scale in [('left',p[3],p[5]),('right',p[4],p[6])]:
                for part,scale,delay in [('hip_pitch',hip_scale,p[7]),('knee',knee_scale,p[8])]:
                    joint=side+'_'+part+'_joint';tau=float(t-delay*wphase)
                    value=sample(joint,tau);anchor=original_targets[2][joint]
                    weight=0. if tau<=anchor_time or tau>=boundary else (1. if tau<=release_time else 1.-smooth((tau-release_time)/(boundary-release_time)))
                    value+=(scale-1.)*weight*max(0.,value-anchor)
                    q[joint]=float(value)
            wwaist=bump(float(t),.4,.7,release_time,1.4)
            q['waist_pitch_joint']+=float(p[2]*wwaist*(reset_angles['waist_pitch_joint']-q['waist_pitch_joint']))
            changed+=sum(value!=original['reference_stages'][index][2][joint] for joint,value in q.items())
        new_time=warp_prefix(float(t));duration=new_time-previous
        if duration<=0:raise ValueError('Nonpositive prefix time warp')
        # When timing is unchanged retain original duration bytes exactly.
        config['reference_stages'][index]=[name,old_duration if p[0]==p[1]==1. else float(duration),q];previous=new_time
    if p[0]!=1. or p[1]!=1.:
        config['head_reference']=[[warp_prefix(float(t)),value] for t,value in original['head_reference']]
        config['residual_start_s']=warp_prefix(original['residual_start_s'])
        config['residual_ramp_s']=warp_prefix(original['residual_start_s']+original['residual_ramp_s'])-config['residual_start_s']
    target_bounds(config,bounds,original);ControlConfig.from_dict(config)
    changes=dict(fields=[k for k in config if config[k]!=original[k]],changed_target_values=changed,
        reference_end_s=previous,initial_ramp_duration_s=.3*p[0],early_interval_duration_s=(boundary-.7)*p[1],
        waist_target_at_original_phase_070_s=config['reference_stages'][1][2]['waist_pitch_joint'],
        residual_start_s=config['residual_start_s'],residual_full_s=config['residual_start_s']+config['residual_ramp_s'],
        suffix_target_identity=all(a[2]==b[2] for a,b in zip(config['reference_stages'][13:],original['reference_stages'][13:])))
    if not changes['suffix_target_identity']:raise ValueError('Unexpected old suffix target change')
    return config,changes



ROUTE_A_NAMES=['early_time_scale','waist_to_zero_mix','ankle_knee_relief_rad',
               'push_time_scale','landing_time_scale','landing_knee_flexion_rad','ankle_roll_to_zero_mix']
ROUTE_A_LOW=np.array([2.2,.3,0.,1.,1.,0.,0.])
ROUTE_A_HIGH=np.array([3.2,.8,.12,1.8,2.2,.10,.2])


def route_a_population():
    """A fixed factorial design: no fitting to outcomes within this batch."""
    rows=[]
    for early in (2.2,2.6,3.0,3.2):
        for waist in (.3,.8):
            rows.append(('slow-waist',np.array([early,waist,0.,1.,1.,0.,0.])))
    for early in (2.6,3.2):
        for waist in (.55,.8):
            for ankle in (.04,.12):
                rows.append(('slow-geometry',np.array([early,waist,ankle,1.,1.,0.,0.])))
    for push in (1.4,1.8):
        for landing in (1.4,1.8):
            for crouch in (0.,.06):
                rows.append(('support-middle',np.array([2.6,.55,.08,push,landing,crouch,.1])))
    for push in (1.4,1.8):
        for landing in (1.6,2.2):
            for crouch in (.04,.10):
                rows.append(('support-slower',np.array([3.2,.8,.12,push,landing,crouch,.2])))
    assert len(rows)==32
    return rows


def route_a_candidate(p,original,bounds,basis):
    p=np.asarray(p,float)
    if p.shape!=ROUTE_A_LOW.shape or not np.isfinite(p).all() or np.any(p<ROUTE_A_LOW) or np.any(p>ROUTE_A_HIGH):
        raise ValueError('Route-A parameter out of bounds')
    times,matrices,_=basis;boundary=float(times[12]);config=copy.deepcopy(original);previous=0.;changed=0
    def wt(t):
        return float(t+.75*max(0.,min(t,.7)-.4)
            +(p[0]-1.)*max(0.,min(t,boundary)-.7)
            +(p[3]-1.)*max(0.,min(t,1.94)-boundary)
            +(p[4]-1.)*max(0.,min(t,2.64)-1.94))
    for index,(stage,t) in enumerate(zip(config['reference_stages'],times)):
        name,_,q=stage;t=float(t)
        waist=bump(t,.4,.7,1.94,2.70)
        geometry=bump(t,.4,.7,float(times[13]),2.64)
        landing=bump(t,1.94,2.14,2.38,2.64)
        if q:
            q['waist_pitch_joint']*=float(1.-p[1]*waist)
            for side in ('left','right'):
                q[side+'_ankle_pitch_joint']+=float(p[2]*geometry)
                q[side+'_knee_joint']-=float(p[2]*geometry)
                q[side+'_ankle_roll_joint']*=float(1.-p[6]*geometry)
                if landing and p[5]:
                    flex=float(p[5]*landing);q[side+'_knee_joint']+=flex
                    delta=matrices[index,side]@np.array([0.,flex])
                    if np.max(abs(delta))>.4:raise ValueError('Landing hip compensation exceeds0.4rad')
                    for part,value in zip(('hip_pitch','hip_roll','hip_yaw'),delta):
                        q[side+'_'+part+'_joint']+=float(value)
            changed+=sum(value!=original['reference_stages'][index][2][joint] for joint,value in q.items())
        new_time=wt(t);duration=new_time-previous
        if duration<=0:raise ValueError('Nonpositive Route-A stage duration')
        config['reference_stages'][index]=[name,float(duration),q];previous=new_time
    config['head_reference']=[[wt(float(t)),v] for t,v in original['head_reference']]
    config['residual_start_s']=wt(original['residual_start_s'])
    config['residual_ramp_s']=wt(original['residual_start_s']+original['residual_ramp_s'])-config['residual_start_s']
    target_bounds(config,bounds,original);ControlConfig.from_dict(config)
    if config['reference_stages'][-1][2]!=original['reference_stages'][-1][2]:raise ValueError('Final standing target changed')
    changes=dict(fields=[k for k in config if config[k]!=original[k]],changed_target_values=changed,
        reference_end_s=previous,initial_ramp_scale=1.75,initial_ramp_duration_s=.525,
        early_interval_duration_s=(boundary-.7)*p[0],push_interval_duration_s=(1.94-boundary)*p[3],
        landing_interval_duration_s=.7*p[4],residual_start_s=config['residual_start_s'],
        residual_full_s=config['residual_start_s']+config['residual_ramp_s'],final_standing_target_identity=True)
    if not set(changes['fields'])<=set(('reference_stages','head_reference','residual_start_s','residual_ramp_s')):
        raise ValueError('Unexpected Route-A controller change')
    return config,changes



LOCAL_NAMES=['early_time_scale','waist_early_to_zero_mix','waist_late_to_zero_mix',
             'push_time_scale','landing_time_scale']
LOCAL_LOW=np.array([2.35,.20,.40,1.60,.90])
LOCAL_HIGH=np.array([2.85,.60,.70,2.10,1.50])
LOCAL_CENTER=np.array([2.6,.55,.55,1.8,1.4])
LOCAL_SIGMA=np.array([.12,.08,.06,.12,.10])


def local_cem_candidate(p,original,bounds,basis):
    p=np.asarray(p,float)
    if p.shape!=LOCAL_LOW.shape or not np.isfinite(p).all() or np.any(p<LOCAL_LOW) or np.any(p>LOCAL_HIGH):
        raise ValueError('Local-CEM parameter outside frozen bounds')
    times=basis[0];boundary=float(times[12]);config=copy.deepcopy(original);previous=0.;changed=0
    def wt(t):
        return float(t+.75*max(0.,min(t,.7)-.4)
            +(p[0]-1.)*max(0.,min(t,boundary)-.7)
            +(p[3]-1.)*max(0.,min(t,1.94)-boundary)
            +(p[4]-1.)*max(0.,min(t,2.64)-1.94))
    for index,(stage,t) in enumerate(zip(config['reference_stages'],times)):
        name,_,q=stage;t=float(t);waist=bump(t,.4,.7,1.94,2.70)
        mix=float(p[1]+(p[2]-p[1])*smooth((t-1.14)/.40))
        geometry=bump(t,.4,.7,float(times[13]),2.64)
        if q:
            q['waist_pitch_joint']*=float(1.-mix*waist)
            for side in ('left','right'):
                q[side+'_ankle_pitch_joint']+=float(.08*geometry)
                q[side+'_knee_joint']-=float(.08*geometry)
                q[side+'_ankle_roll_joint']*=float(1.-.1*geometry)
            changed+=sum(value!=original['reference_stages'][index][2][joint] for joint,value in q.items())
        new_time=wt(t);duration=new_time-previous
        if duration<=0:raise ValueError('Nonpositive local-CEM stage duration')
        config['reference_stages'][index]=[name,float(duration),q];previous=new_time
    config['head_reference']=[[wt(float(t)),v] for t,v in original['head_reference']]
    config['residual_start_s']=wt(original['residual_start_s'])
    config['residual_ramp_s']=wt(original['residual_start_s']+original['residual_ramp_s'])-config['residual_start_s']
    target_bounds(config,bounds,original);ControlConfig.from_dict(config)
    changes=dict(fields=[k for k in config if config[k]!=original[k]],changed_target_values=changed,
        reference_end_s=previous,residual_start_s=config['residual_start_s'],
        residual_full_s=config['residual_start_s']+config['residual_ramp_s'],final_standing_target_identity=True)
    if config['reference_stages'][-1][2]!=original['reference_stages'][-1][2]:raise ValueError('Final standing target changed')
    if not set(changes['fields'])<=set(('reference_stages','head_reference','residual_start_s','residual_ramp_s')):
        raise ValueError('Unexpected local-CEM controller change')
    return config,changes


def local_cem_update(results_path,plan_path,anchor):
    """Use complete real-episode metrics and their original common scores."""
    result_file=Path(results_path).resolve();plan_file=Path(plan_path).resolve() if plan_path else result_file.parent/'plan.json'
    old=json.loads(plan_file.read_text());trace=result_file.parent/'candidates.jsonl';manifest_file=result_file.parent/'manifest.json'
    if old.get('family_design')!='local-cem' or old['parameter_space']['names']!=LOCAL_NAMES:
        raise ValueError('Local-CEM continuation requires its own five-dimensional parameter basis')
    if json.loads(manifest_file.read_text())['plan_sha256']!=sha(plan_file):raise ValueError('Executed plan hash mismatch')
    rows=[json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    indices=[int(row['index']) for row in rows]
    if sorted(indices)!=list(range(16)):raise ValueError('Continuation requires all16 unique recorded candidates; no partial-batch fitting')
    csv_rows=list(csv.DictReader(result_file.open()))
    if sorted(int(row['index']) for row in csv_rows)!=list(range(16)):raise ValueError('CSV and full trial records disagree')
    csv_by_index={int(row['index']):row for row in csv_rows};eligible=[];excluded=[]
    for row in rows:
        index=int(row['index']);planned=old['candidates'][index]
        expected=hashlib.sha256(canonical(planned['control_config']).encode()).hexdigest()
        if planned['control_config_sha256']!=expected or row['control_sha256']!=expected:
            raise ValueError('Actual controller identity mismatch at index '+str(index))
        if row['name']!=planned['name'] or csv_by_index[index]['name']!=row['name']:
            raise ValueError('CSV, plan and full-result names disagree')
        reason=None
        if row['status']!='COMPLETE' or not row.get('complete_episode'):
            reason='not_complete_episode'
        elif not row['success'] and row['reason']!='time_limit':
            reason='safety_or_other_early_termination'
        elif not row['success'] and (row['max_progress_quality']<.9*anchor['max_progress_quality'] or
                                    row['progress_integral_s']<.9*anchor['progress_integral_s']):
            reason='insufficient_recovery_progress_relative_to_fixed_g04_20_anchor'
        elif not all(np.isfinite(row[k]) for k in ('progress_score','violation_score','infeasible_score')):
            reason='nonfinite_common_scores'
        if reason:excluded.append(dict(index=index,name=row['name'],reason=reason))
        else:eligible.append(row)
    if not eligible:raise ValueError('No complete progress-qualified trials; retain existing state and investigate, never fit low-motion/early-abort candidates')
    # Pareto dimensions use the same experiment-wide measured feasibility and
    # existing progress/violation scores. No engineering acceptance is relaxed.
    feasible=[r for r in eligible if r['feasible']]
    pool=feasible if feasible else eligible
    def vec(r):return np.array([float(r['success']),r['progress_score'],-r['violation_score']])
    front=[r for r in pool if not any(np.all(vec(s)>=vec(r)) and np.any(vec(s)>vec(r)) for s in pool)]
    front.sort(key=lambda r:(bool(r['feasible']),bool(r['success']),r['infeasible_score'],r['progress_score']),reverse=True)
    elite=front[:4];vectors=np.array([old['candidates'][int(r['index'])]['parameters'] for r in elite])
    old_mean=np.array(old['search_state']['mean']);old_sigma=np.array(old['search_state']['sigma'])
    mean=.5*old_mean+.5*vectors.mean(axis=0)
    dispersion=vectors.std(axis=0) if len(elite)>1 else .8*old_sigma
    sigma=np.maximum(.15*LOCAL_SIGMA,np.minimum(.30*(LOCAL_HIGH-LOCAL_LOW),.6*old_sigma+.4*dispersion))
    best_feasible=max(feasible,key=lambda r:r['infeasible_score'])['name'] if feasible else old['search_state'].get('best_feasible')
    details=dict(results=str(result_file),results_sha256=sha(result_file),trials=str(trace),trials_sha256=sha(trace),
        manifest_sha256=sha(manifest_file),plan=str(plan_file),plan_sha256=sha(plan_file),
        eligible_names=[r['name'] for r in eligible],excluded=excluded,pareto_names=[r['name'] for r in front],
        elite_names=[r['name'] for r in elite],mean_update_fraction=.5,sigma_update_fraction=.4,
        best_feasible=best_feasible)
    return mean,sigma,old,details


def generate_local_cem(args,original,bounds,basis,manifest_path,resolved,urdf,bounds_path):
    if args.count!=16:raise ValueError('Local-CEM uses exactly16 complete candidates per generation')
    if args.center or args.sigma:raise ValueError('Local-CEM starts from frozen g04-20 or recorded CEM state; no unrecorded manual override')
    repo=args.repo.resolve();output=args.output.resolve();anchor_dir=manifest_path.parent/'generation-04'
    anchor_plan=json.loads((anchor_dir/'plan.json').read_text());anchor_rows=[json.loads(x) for x in (anchor_dir/'candidates.jsonl').read_text().splitlines()]
    anchor=next(x for x in anchor_rows if x['index']==20)
    if anchor['name']!='g04-support-middle-20' or anchor['status']!='COMPLETE' or not anchor['complete_episode']:
        raise ValueError('Required measured g04-20 anchor is unavailable')
    center_cfg,_=local_cem_candidate(LOCAL_CENTER,original,bounds,basis)
    if canonical(center_cfg)!=canonical(anchor_plan['candidates'][20]['control_config']):
        raise ValueError('Five-dimensional center is not exactly g04-20 controller')
    mean=LOCAL_CENTER.copy();sigma=LOCAL_SIGMA.copy();previous=None;old=None
    if args.previous_results:
        mean,sigma,old,previous=local_cem_update(args.previous_results,args.previous_plan,anchor)
        if args.generation!=old['generation']+1:raise ValueError('Continuation generation must equal previous+1')
    elif args.generation!=6:raise ValueError('Initial local-CEM generation is6; later generations require actual previous results')
    rng=np.random.default_rng(np.random.SeedSequence([RNG_SEED,args.generation]))
    if old is not None:rng.bit_generator.state=old['search_state']['rng_state_after']
    before=copy.deepcopy(rng.bit_generator.state);items=[];rejected=[];seen=set();attempt=0
    while len(items)<16:
        if attempt>=10000:raise ValueError('Local-CEM proposal rejection budget exhausted')
        p=mean.copy() if attempt==0 else rng.normal(mean,sigma);attempt+=1
        try:
            cfg,changes=local_cem_candidate(p,original,bounds,basis)
            digest=hashlib.sha256(canonical(cfg).encode()).hexdigest()
            if digest in seen:raise ValueError('Duplicate full controller')
        except ValueError as exc:
            rejected.append(dict(attempt=attempt,parameters=p.tolist(),reason=str(exc)));continue
        seen.add(digest);index=len(items)
        items.append(dict(name=f'g{args.generation:02d}-local-cem-{index:02d}',family='local-cem',
            parameters=p.tolist(),named_parameters=dict(zip(LOCAL_NAMES,p.tolist())),changes=changes,
            control_config_sha256=digest,control_config=cfg))
    protocol=json.loads(manifest_path.read_text())
    plan=dict(schema='coupled-reference-candidates-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        generation=args.generation,family_design='local-cem',candidates=items,comparison=protocol['comparison'],
        seed=221030,zero_residual=True,original_control_config=original,
        source=dict(resolved_config=str(resolved),resolved_config_sha256=sha(resolved),script_sha256=sha(__file__),
            protocol_manifest_sha256=sha(manifest_path),urdf=str(urdf),urdf_sha256=sha(urdf),
            effective_bounds_source=str(bounds_path),effective_bounds_sha256=sha(bounds_path),
            anchor_plan_sha256=sha(anchor_dir/'plan.json'),anchor_trials_sha256=sha(anchor_dir/'candidates.jsonl'),
            anchor_control_sha256=anchor['control_sha256']),
        parameter_space=dict(names=LOCAL_NAMES,lower=LOCAL_LOW.tolist(),upper=LOCAL_HIGH.tolist()),
        basis=dict(waist='Mix starts at early; quintic interpolation to late over ORIGINAL phase[1.14,1.54]; multiply by C2window[.4,.7,1.94,2.70] and contract original waist target toward0.',
            timewarp='initial[.4,.7] fixed1.75, early[.7,stage12end], push[stage12end,1.94], landing[1.94,2.64]; translate later times. All76nodes/head/gate transformed together.',
            fixed_geometry='Bilateral ankle_pitch+.08*w, knee-.08*w, ankle_roll*(1-.1*w), w=C2window[.4,.7,stage13end,2.64]; no extra landing crouch.',
            fixed_physics='Original physics, PD, effort/target-rate limits,20s horizon,elapsed/20 observation,success and safety thresholds; final31joint standing target unchanged.',
            validation='Full ControlConfig.from_dict and strict changed-target bounds; reject rather than clip; only unchanged original boundary float tolerance retained.',
            anchor='g04-20; generator center full canonical ControlConfig equality checked before sampling',
            continuation='Complete16 actual trials only. Exclude safety/other early termination and low-progress trials. Pareto on success,existing progress_score,negative existing violation_score; up to4 front elites sorted by existing infeasible_score. Smooth means and variances; never claim engineering feasibility from optimizer eligibility.',
            progress_guard=dict(kind='search eligibility only, not new engineering PASS criteria',anchor='g04-20',fraction=.9,
                minimum_max_progress_quality=.9*anchor['max_progress_quality'],minimum_progress_integral_s=.9*anchor['progress_integral_s'],
                success_exempt=True),feasibility='Unchanged joint1e-4rad, official speed/effort bounds and common impact limits plus original standing success'),
        search_state=dict(mean=mean.tolist(),sigma=sigma.tolist(),rng_seed=RNG_SEED,rng_state_before=before,
            rng_state_after=rng.bit_generator.state,attempts=attempt,accepted=16,rejections=rejected,
            population_method='One current mean +15 bounded independent Gaussian proposals; static generation only',
            previous=previous,best_feasible=previous['best_feasible'] if previous else None,
            anchor_existing_standing_success=False,anchor_engineering_feasible=False),
        hypothesis='Local support/waist timing search around the measured zero-ankle-excess but unsuccessful g04-20. Preserves all common full-episode metrics; no successful safe controller assumed.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),count=16,attempts=attempt,rejected=len(rejected),sha256=sha(output)),indent=2))



TAIL_NAMES=['bilateral_hip_pitch_offset_rad','bilateral_knee_extension_rad','ankle_orientation_compensation']


def tail_candidate(p,anchor,original,bounds,axes_fk):
    hip,knee,comp=p
    if not .025<=hip<=.15 or not 0.<=knee<=.10 or comp not in (0.,1.):raise ValueError('Tail family parameter out of bounds')
    config=copy.deepcopy(anchor);old_times=np.cumsum([s[1] for s in original['reference_stages']])
    changed=0;residual=[];peak_comp=0.
    for index,(stage,t) in enumerate(zip(config['reference_stages'],old_times)):
        weight=smooth((float(t)-1.58)/(.36));q=stage[2]
        if not q or not weight:continue
        original_q=anchor['reference_stages'][index][2];axes=axes_fk.axes(original_q)
        for side in ('left','right'):
            hp=side+'_hip_pitch_joint';kn=side+'_knee_joint';ap=side+'_ankle_pitch_joint';ar=side+'_ankle_roll_joint'
            dh=hip*weight;dk=-knee*weight;q[hp]+=dh;q[kn]+=dk
            angle=axes[hp]*dh+axes[kn]*dk
            if comp:
                ankle_axes=np.column_stack([axes[ap],axes[ar]])
                delta=np.linalg.lstsq(ankle_axes,-angle,rcond=None)[0]
                if np.max(abs(delta))>.25:raise ValueError('Tail ankle compensation exceeds0.25rad')
                q[ap]+=float(delta[0]);q[ar]+=float(delta[1]);angle+=ankle_axes@delta
                peak_comp=max(peak_comp,float(np.max(abs(delta))))
            residual.append(float(np.linalg.norm(angle)))
        changed+=sum(value!=original_q[joint] for joint,value in q.items())
    target_bounds(config,bounds,anchor);ControlConfig.from_dict(config)
    # No pre-tail state/reference, timing, reward, action or physical settings change.
    prefix=[a==b for a,b,t in zip(config['reference_stages'],anchor['reference_stages'],old_times) if t<=1.58]
    if not all(prefix):raise ValueError('Tail family unexpectedly changed prefix')
    fields=[k for k in config if config[k]!=anchor[k]]
    if fields!=['reference_stages']:raise ValueError('Tail family changed another controller field')
    return config,dict(fields_vs_anchor=fields,changed_target_values=changed,
        prefix_through_original_phase_s=1.58,prefix_exact_identity=True,
        full_bias_at_original_phase_s=1.94,offset_persists_in_final_stance=True,
        max_ankle_compensation_rad=peak_comp,max_first_order_uncompensated_orientation_rad=max(residual),
        original_standing_target_changed=True,reference_end_s=sum(s[1] for s in config['reference_stages']),
        final_knee_targets_rad=[config['reference_stages'][-1][2][s+'_knee_joint'] for s in ('left','right')])


def generate_tail_support(args,original,bounds,manifest_path,resolved,urdf,bounds_path):
    if args.count!=32 or args.generation!=8 or args.center or args.sigma or args.previous_results:
        raise ValueError('Initial tail family is frozen generation8,32 fixed factorial candidates; no adaptive replacement')
    root=manifest_path.parent;anchor_dir=root/'generation-04';anchor_plan=json.loads((anchor_dir/'plan.json').read_text())
    anchor=anchor_plan['candidates'][20]['control_config'];row=next(json.loads(x) for x in (anchor_dir/'candidates.jsonl').read_text().splitlines() if json.loads(x)['index']==20)
    if row['control_sha256']!=hashlib.sha256(canonical(anchor).encode()).hexdigest():raise ValueError('Tail anchor executed control identity mismatch')
    if row['name']!='g04-support-middle-20' or not row['complete_episode']:raise ValueError('Wrong tail anchor')
    fk=StaticAxes(urdf);items=[];rejections=[]
    for comp in (0.,1.):
        for hip in (.025,.05,.10,.15):
            for knee in (0.,.03,.06,.10):
                p=[hip,knee,comp]
                try:cfg,changes=tail_candidate(p,anchor,original,bounds,fk)
                except ValueError as exc:
                    rejections.append(dict(parameters=p,reason=str(exc)))
                    continue
                index=len(items);digest=hashlib.sha256(canonical(cfg).encode()).hexdigest()
                items.append(dict(name=f'g08-tail-{int(comp)}-{index:02d}',family='tail-ankle-compensated' if comp else 'tail-direct',
                    parameters=p,named_parameters=dict(zip(TAIL_NAMES,p)),changes=changes,
                    control_config_sha256=digest,control_config=cfg))
    if rejections:raise ValueError('Fixed tail design has rejected targets; no silent clipping/replacement: '+json.dumps(rejections))
    if len({x['control_config_sha256'] for x in items})!=32:raise ValueError('Duplicate tail controller')
    protocol=json.loads(manifest_path.read_text());output=args.output.resolve()
    plan=dict(schema='coupled-reference-candidates-v1',created_utc=datetime.now(timezone.utc).isoformat(),generation=8,
        family_design='tail-support',candidates=items,comparison=protocol['comparison'],seed=221030,zero_residual=True,
        original_control_config=original,anchor_control_config=anchor,
        source=dict(resolved_config=str(resolved),resolved_config_sha256=sha(resolved),script_sha256=sha(__file__),
            protocol_manifest_sha256=sha(manifest_path),urdf=str(urdf),urdf_sha256=sha(urdf),
            effective_bounds_source=str(bounds_path),effective_bounds_sha256=sha(bounds_path),
            anchor_plan_sha256=sha(anchor_dir/'plan.json'),anchor_trials_sha256=sha(anchor_dir/'candidates.jsonl'),
            anchor_control_sha256=row['control_sha256']),
        parameter_space=dict(names=TAIL_NAMES,lower=[.025,0.,0.],upper=[.15,.10,1.],
            discrete_axes={'hip':[.025,.05,.10,.15],'knee_extension':[0.,.03,.06,.10],'ankle_compensation':[0.,1.]}),
        basis=dict(anchor='Fixed g04-20, not outcome-dependent gen7',
            window='Quintic ramp from0 at ORIGINAL phase1.58 to1 at1.94, held through2.64/3.2 and afterward; sampled into existing76 linear reference nodes. Reference/head/gate timestamps unchanged from anchor.',
            hips='Same positive pitch offset on both hips; hypothesis: move support/body forward during the measured backward drift, direction requires real simulation.',
            knees='Same negative target offset on both knees, including new final stance; strict q>=0 and all other model bounds, never clipped.',
            foot_compensation='Optional two-ankle-axis least-squares first-order orientation compensation for imposed hip/knee increments at anchor URDF pose; residual3D rotation explicitly recorded, not exact foot-orientation preservation.',
            frozen='Prefix through originalphase1.58, physical/PD/rate/effort/success/safety/reward/action/observation/horizon settings, all arm targets and all timing.',
            limitation='Earlier positive waist violation of g04-20 remains outside this tail intervention; do not expect tail control alone to establish full-envelope feasibility.',
            validation='Full ControlConfig.from_dict, strict changed-target range rejection, exact prefix identity; complete1kHz physics measurements still required.'),
        search_state=dict(mean=None,sigma=None,rng_seed=None,rng_state_before=None,rng_state_after=None,
            population_method='Fixed4hip x4knee x2compensation factorial',accepted=32,rejections=[],best_feasible=None,
            continuation='Use complete actual task/constraint/impact outcomes to define a bounded tail CEM family; no automatic mixing with preceding5D family.'),
        hypothesis='Measured g04-23 backward COM drift with feet still loaded, insufficient pelvis height and torso tilt motivates preemptive tail hip-pitch and knee-extension bias. These are control hypotheses, not verified mechanisms.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),count=len(items),rejected=0,sha256=sha(output)),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root_default=Path(__file__).resolve().parents[3]
    parser.add_argument('--repo',type=Path,default=root_default)
    parser.add_argument('--asset-repo',type=Path,default=Path.home()/'.cache/hrs-x2-recovery/agibot_x2_urdf')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generation',type=int,default=0)
    parser.add_argument('--family',choices=('coupled-cem','local-directions','prefix-drive','slow-support','local-cem','tail-support'),default='coupled-cem')
    parser.add_argument('--count',type=int)
    parser.add_argument('--center');parser.add_argument('--sigma')
    parser.add_argument('--previous-results',type=Path);parser.add_argument('--previous-plan',type=Path)
    args=parser.parse_args();repo=args.repo.resolve();output=args.output.resolve()
    if args.count is None:args.count=16 if args.family=='local-cem' else 32
    if output.exists():raise FileExistsError(output)
    if args.generation<0 or not 1<=args.count<=32:raise ValueError('Invalid generation/count')
    manifest_path=Path(__file__).with_name('manifest.json');manifest=json.loads(manifest_path.read_text())
    resolved=repo/manifest['frozen_controller_input']/'resolved_config.json';saved=json.loads(resolved.read_text());original=saved['controller']
    urdf=args.asset_repo.resolve()/URDF
    if sha(urdf)!=SOURCE_HASHES[URDF]:raise ValueError('Fixed upstream URDF identity mismatch')
    bounds_path=repo/'results/controller-review/20260924T132353Z/published-policy/joint_constraints.csv'
    bounds={r['joint_name']:(float(r['q_lower_rad']),float(r['q_upper_rad'])) for r in csv.DictReader(bounds_path.open())}
    target_bounds(original,bounds,original);ControlConfig.from_dict(original)
    basis=prepare_basis(original,StaticAxes(urdf))
    if args.family=='tail-support':
        generate_tail_support(args,original,bounds,manifest_path,resolved,urdf,bounds_path)
        return
    if args.family=='local-cem':
        generate_local_cem(args,original,bounds,basis,manifest_path,resolved,urdf,bounds_path)
        return
    if args.family in ('local-directions','prefix-drive','slow-support') and (args.center or args.sigma or args.previous_results or args.count!=32):
        raise ValueError('This family is an explicit32-point design; center/sigma/previous-results are not used')
    mean=CENTER.copy() if args.family=='coupled-cem' else np.array([0.,0.,0.,1.,1.,0.,0.,0.,0.])
    sigma=SIGMA.copy();previous=None;old=None
    if args.previous_results:
        mean,sigma,old,previous=read_previous(args.previous_results,args.previous_plan)
        if args.generation!=old['generation']+1:raise ValueError('Continuation generation must equal previous+1')
    explicit_mean=parse_vector(args.center,'mean');explicit_sigma=parse_vector(args.sigma,'sigma')
    if explicit_mean is not None:mean=explicit_mean
    if explicit_sigma is not None:sigma=explicit_sigma
    if np.any(mean<LOW) or np.any(mean>HIGH) or np.any(sigma<=0):raise ValueError('Invalid mean/sigma')
    rng=np.random.default_rng(np.random.SeedSequence([RNG_SEED,args.generation]))
    if old is not None:rng.bit_generator.state=old['search_state']['rng_state_after']
    rng_before=copy.deepcopy(rng.bit_generator.state);items=[];rejections=[];seen=set();attempt=0
    fixed=local_direction_population() if args.family=='local-directions' else (route_a_population() if args.family=='slow-support' else None)
    reset_angles=None;prefix_source=None
    if args.family=='prefix-drive':
        trace_root=repo/'results/controller-review/20260924T132353Z/published-policy'
        with np.load(trace_root/'physical_trace.npz',allow_pickle=False) as mapping,np.load(trace_root/'control_trace.npz',allow_pickle=False) as trace:
            reset_angles=dict(zip(mapping['joint_names'].tolist(),trace['reset_qpos'][mapping['joint_qpos_addresses']].tolist()))
        fixed=prefix_population(original,reset_angles['waist_pitch_joint'])
        prefix_source=dict(control_trace_sha256=sha(trace_root/'control_trace.npz'),reset_waist_rad=reset_angles['waist_pitch_joint'])
        mean=np.array([1.,1.,0.,1.,1.,1.,1.,0.,0.])
    while len(items)<args.count:
        if attempt>=10000:raise ValueError('Candidate rejection budget exhausted')
        # First point is the new coupled center, not a repeat of baseline calibration.
        if fixed is not None:
            if attempt>=len(fixed):raise ValueError('An explicit local candidate failed; no silent replacement')
            family,p=fixed[attempt]
        elif attempt==0:family,p='coupled-cem',mean.copy()
        else:family,p='coupled-cem',rng.normal(mean,sigma*[.5,1.,1.5][attempt%3])
        attempt+=1
        try:
            cfg,changes=(prefix_candidate(p,original,bounds,basis,reset_angles) if args.family=='prefix-drive' else
                route_a_candidate(p,original,bounds,basis) if args.family=='slow-support' else candidate(p,original,bounds,basis))
            digest=hashlib.sha256(canonical(cfg).encode()).hexdigest()
            if digest in seen:raise ValueError('Duplicate full control configuration')
        except ValueError as exc:
            rejections.append(dict(attempt=attempt,parameters=p.tolist(),reason=str(exc)));continue
        seen.add(digest);number=len(items)
        items.append(dict(name=f'g{args.generation:02d}-{family}-{number:02d}',family=family,parameters=p.tolist(),
            named_parameters=dict(zip(PREFIX_NAMES if args.family=='prefix-drive' else ROUTE_A_NAMES if args.family=='slow-support' else NAMES,p.tolist())),changes=changes,control_config_sha256=digest,control_config=cfg))
    plan=dict(schema='coupled-reference-candidates-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        generation=args.generation,family_design=args.family,candidates=items,comparison=manifest['comparison'],seed=221030,zero_residual=True,
        original_control_config=original,source=dict(resolved_config=str(resolved),resolved_config_sha256=sha(resolved),
            script_sha256=sha(__file__),previous_generator_sha256='d612735886f621b482f024739aa96ccdc771518241b2699367eb26597650b66f',protocol_manifest_sha256=sha(manifest_path),urdf=str(urdf),urdf_sha256=sha(urdf),
            effective_bounds_source=str(bounds_path),effective_bounds_sha256=sha(bounds_path)),
        parameter_space=dict(names=PREFIX_NAMES if args.family=='prefix-drive' else NAMES,lower=(PREFIX_LOW if args.family=='prefix-drive' else LOW).tolist(),upper=(PREFIX_HIGH if args.family=='prefix-drive' else HIGH).tolist()),
        basis=dict(early_ankle_knee='[+a,-a] on ankle/knee preserves reference foot orientation because their axes are parallel; actual contact response is unproven.',
            early_window=[.4,.7,1.577870,1.94],waist_negative_window=[.7877192982456141,.9631578947368421,1.1385964912280702,1.4017543859649122],
            landing_window=[1.94,2.18,2.24,2.64],smoothstep='6u^5-15u^4+10u^3; C2 bump at joins',
            landing_hip_compensation='First-order world-orientation Jacobian from verified URDF at original reference nodes; solve three hip axes against imposed knee/ankle axes, never actual-state writes.',
            landing_hip_compensation_max_abs_rad=.40,landing_J_condition_numbers=basis[2],
            timewarp='Piecewise affine old->new time: scale[.7,1.577870] by early and[1.577870,1.94] by push; translate subsequent times. Warp every reference node, head-reference time and gate endpoints.',
            observation_phase='Actual elapsed/20 unchanged; no warped observation clock',
            physics_success_safety_timeout='Unchanged',target_validation='No target clipping. Reject changed out-of-bound targets; permit unchanged original boundary export floats within existing1e-6 tolerance.',
            arms='Unchanged in this first coupled search; no hand-support claim.',baseline_candidate='Omitted because root already completed independent metric calibration.'),
        search_state=dict(mean=mean.tolist(),sigma=sigma.tolist(),rng_seed=RNG_SEED,
            population_method='Explicit32-point near-original design; no RNG draws' if fixed is not None else 'Bounded Gaussian mixture',
            sigma_used_for_sampling=fixed is None,
            rng_state_before=rng_before,rng_state_after=rng.bit_generator.state,attempts=attempt,
            accepted=len(items),rejections=rejections,previous=previous,
            continuation='Use --previous-results and next --generation. Only eligible complete trials inform elites; no silent recovery from too few eligible trials.'),
        hypothesis='Coordinated reference geometry and timing may reduce ankle/waist dynamic load and landing shock. No claim of physical improvement before full-episode measurements.')
    if args.family=='prefix-drive':
        plan['basis']=dict(initial_ramp_interval=[.4,.7],early_interval=[.7,float(basis[0][12])],
            timewarp='Stretch initial and early intervals separately; translate all subsequent times. All76nodes/head-table/gate endpoints move together;20s physics horizon and actual elapsed/20 observation unchanged.',
            waist='Convex blend of original waist target toward actual frozen reset waist angle; C2 window[.4,.7,stage6_end,1.4] in ORIGINAL reference phase; unchanged by original phase1.4, whose physical time is warped.',
            leg_amplitude='Stage2 reference is anchor. Scale positive hip-return/knee-flexion excursion by named factors through stage6; quintic release to original by stage12. Negative hip dip unchanged.',
            phase='Sample each bilateral hip_pitch/knee original target at old_time-delay*window[.7,.8754385964912281,1.2263157894736842,stage12_end]; signed phase delays explicit.',
            suffix='Every target dictionary from stage13 onward exactly equals original; only absolute schedule may shift. Landing and arm targets untouched.',
            coefficient_sources=prefix_source,FK='Not needed for this prefix-only family',
            target_validation='Strict changed-target bound rejection; no clipping; original boundary float exception unchanged.',
            population='8initial/waist +8leg-drive +8timing/waist +8coupled-phase; no result-conditioned replacement; baseline already calibrated separately.',
            hypotheses='Initial1.75 reduces nominal initial waist slope below1.2rad/s; knee progress0.6/0.7 is intended to break the old+3rad/s limiter plateau. Actual adopted slopes and mechanics must be measured; no outcome predicted.')
    if args.family=='slow-support':
        plan['parameter_space']=dict(names=ROUTE_A_NAMES,lower=ROUTE_A_LOW.tolist(),upper=ROUTE_A_HIGH.tolist())
        plan['search_state']['mean']=[2.6,.55,.08,1.4,1.4,.04,.1]
        plan['search_state']['sigma']=None
        plan['search_state']['continuation']='Explicit preregistered factorial design; never infer CEM coordinates from another family.'
        plan['source']['previous_generator_sha256']='d11b77bc189c6402ebb5ae12854bb4b2c56cd5cb84f8ec43af7defa01d016521'
        plan['basis']=dict(initial_ramp_scale=1.75,
            timewarp='Stretch [.4,.7] by1.75, [.7,stage12end] by early, [stage12end,1.94] by push, [1.94,2.64] by landing; translate later nodes. Warp all76reference/head/gate times;20s horizon and elapsed/20 unchanged.',
            waist='Contract reference waist toward zero by mix*C2window[.4,.7,1.94,2.70]. Covers both negative excursions, releases to exact original final standing posture. This differs deliberately from gen2 toward-reset blend.',
            geometry='Bilateral ankle_pitch+=a*w and knee-=a*w, w=C2window[.4,.7,stage13end,2.64]. Parallel pitch axes preserve foot orientation geometrically, not foot position/contact.',
            ankle_roll='Contract ankle_roll target toward zero by mix*w in support families only; changes foot roll deliberately; no claim of maintaining support before measurement.',
            landing='Add bilateral knee flexion c*C2window[1.94,2.14,2.38,2.64], with three-hip first-order orientation compensation from original verified URDF node Jacobian. No actual state writes.',
            hip_J_condition_numbers=basis[2],target_validation='Strict changed-target range rejection; no clipping; unchanged original boundary float exception only.',
            family_counts={'slow-waist':8,'slow-geometry':8,'support-middle':8,'support-slower':8},
            population='Fixed factorial contrasts, no result-conditioned replacements, no old baseline repetition.',
            unchanged='PD/effort/rate limits, all physical/success/safety settings, original20s horizon and final31joint target; no auxiliary support forces.',
            hypotheses='Gen2 early stretch reduced waist sampled excess but did not solve later ankle/standing. This batch tests coordinated stronger stretch, centered waist excursion and preserved-foot-pitch ankle/knee clearance; landing timing/crouch may reduce impulsive transfer. Full episodes must still judge standing and all constraints separately.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),count=len(items),attempts=attempt,rejected=len(rejections),sha256=sha(output)),indent=2))


if __name__=='__main__':main()
