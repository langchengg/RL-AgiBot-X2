"""Reusable, zero-control physical supine reset. No I/O, graphics or ROS.

An observer is called at 25 simulated Hz with copies. It must not mutate the
captured model/data. The same MjData, including residual velocity/time, is handed on.
"""
from dataclasses import dataclass, asdict
import hashlib
import time
import mujoco as mj
import numpy as np
from .model import require


@dataclass(frozen=True)
class ResetSettings:
    clearance_m: float = .002
    geometry_epsilon_m: float = 2e-6
    max_settle_s: float = 5.
    dwell_s: float = .5
    hold_s: float = 1.
    wall_s: float = 45.
    face_up_deg: float = 20.
    linear_m_s: float = .01
    angular_rad_s: float = .05
    joint_rad_s: float = .02
    drift_m: float = .003
    drift_rad: float = .01
    floor_m: float = .001
    self_m: float = .0001
    joint_epsilon_rad: float = 1e-6
    transient_floor_m: float = .012
    transient_self_m: float = .003
    perturb_rad: float = 0.


class ResetFailure(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def nominal_pose(loaded):
    # All 31 joints are explicit, in their actual (not assumed mirrored) convention.
    pose = {name: 0. for name in ('waist_yaw_joint', 'waist_roll_joint',
                                'head_yaw_joint', 'head_pitch_joint')}
    pose['waist_pitch_joint'] = .30
    for side, sign in (('left', 1), ('right', -1)):
        for joint, value in {'hip_pitch': .01, 'hip_roll': 0., 'hip_yaw': 0.,
                             'knee': .03, 'ankle_pitch': -.37, 'ankle_roll': 0.,
                             'shoulder_pitch': .42, 'shoulder_roll': sign*.5,
                             'shoulder_yaw': 0., 'elbow': -.65, 'wrist_yaw': 0.,
                             'wrist_pitch': -.22, 'wrist_roll': sign*.05}.items():
            pose[f'{side}_{joint}_joint'] = value
    require(set(pose) == {r.joint_name for r in loaded.mapping}, 'Incomplete nominal pose')
    return pose


def model_signature(m):
    # Also detects changes made by viewer GUI/observer. Includes collision mesh data.
    h = hashlib.sha256(str(m.opt).encode())
    for name in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                 'body_gravcomp','jnt_range','jnt_margin','jnt_solref','jnt_solimp',
                 'jnt_limited','jnt_actfrclimited','jnt_actfrcrange','jnt_axis',
                 'dof_damping','dof_armature','dof_frictionloss','geom_type','geom_size',
                 'geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction',
                 'geom_solref','geom_solimp','geom_margin','geom_gap','geom_priority',
                 'mesh_vert','actuator_gear','actuator_gainprm','actuator_biasprm',
                 'actuator_ctrlrange','actuator_ctrllimited','eq_active0'):
        h.update(getattr(m,name).tobytes())
    return h.hexdigest()


def integration_state(m, d):
    sig = mj.mjtState.mjSTATE_INTEGRATION
    out = np.empty(mj.mj_stateSize(m, sig))
    mj.mj_getState(m, d, out, sig)
    return out


def check_callbacks():
    for name in ('control','passive','contactfilter','act_bias','act_dyn','act_gain','sensor'):
        require(getattr(mj, 'get_mjcb_'+name)() is None, 'Unexpected MuJoCo callback: '+name)


def notify(m, d, callback, phase, extra=None):
    if callback is None:
        return
    before, physics = integration_state(m,d), model_signature(m)
    callback({'phase': phase, 'time_s': float(d.time), 'qpos': d.qpos.copy(),
              'qvel': d.qvel.copy(), 'ctrl': d.ctrl.copy(), **(extra or {})})
    require(np.array_equal(before,integration_state(m,d)) and physics == model_signature(m),
            'Observer/viewer changed integration state or physics')
    check_callbacks()


def floor_geometry(m, d):
    f = m.geom('floor').id
    require(m.geom_type[f] == mj.mjtGeom.mjGEOM_PLANE and m.geom_bodyid[f] == 0,
            'Expected fixed floor plane')
    n = d.geom_xmat[f].reshape(3,3)[:,2].copy()
    require(np.allclose(n, [0,0,1], atol=1e-12), 'Expected horizontal upward floor')
    geoms = [g for g in range(m.ngeom) if g != f and
             ((m.geom_contype[g]&m.geom_conaffinity[f]) or
              (m.geom_contype[f]&m.geom_conaffinity[g]))]
    return f, n, d.geom_xpos[f].copy(), geoms


def plane_gap(m, d, g, normal, origin):
    """Exact support along plane normal for this model's compiled shapes.

    Compiled mesh_vert has scale/recentering baked in; geom_xmat/geom_xpos maps
    those vertices to world. A linear support minimum equals the hull minimum.
    """
    R=d.geom_xmat[g].reshape(3,3); local=R.T@normal
    center=float((d.geom_xpos[g]-origin)@normal); typ=m.geom_type[g]
    if typ == mj.mjtGeom.mjGEOM_MESH:
        mesh=m.geom_dataid[g]; start=m.mesh_vertadr[mesh]
        vertices=m.mesh_vert[start:start+m.mesh_vertnum[mesh]]
        return center+float(np.min(vertices@local))
    if typ == mj.mjtGeom.mjGEOM_SPHERE:
        return center-float(m.geom_size[g,0])
    if typ == mj.mjtGeom.mjGEOM_CYLINDER:
        return center-float(m.geom_size[g,0]*np.linalg.norm(local[:2])+m.geom_size[g,1]*abs(local[2]))
    raise ValueError('Unsupported collision shape '+str(typ))


def initial_geometry(m,d,s):
    f,n,origin,geoms=floor_geometry(m,d)
    gaps=[(plane_gap(m,d,g,n,origin),g) for g in geoms]
    query_errors=[]
    for gap,g in gaps:
        cutoff=max(2.,gap+.1)
        query=mj.mj_geomDistance(m,d,f,g,cutoff,None)
        require(query < cutoff, 'Distance cutoff is not a measured gap')
        query_errors.append(abs(query-gap))
    require(max(query_errors)<s.geometry_epsilon_m, 'Plane support/query disagreement')
    require(min(gap for gap,g in gaps)>=-s.geometry_epsilon_m, 'Initial floor intersection')
    excluded=[]; nearby=[]
    require(m.npair==0 and m.nexclude==0, 'Unexpected pair/exclusion policy')
    for i,g in enumerate(geoms):
        for h in geoms[i+1:]:
            if not ((m.geom_contype[g]&m.geom_conaffinity[h]) or
                    (m.geom_contype[h]&m.geom_conaffinity[g])): continue
            b,c=int(m.body_weldid[m.geom_bodyid[g]]),int(m.body_weldid[m.geom_bodyid[h]])
            filtered=(b==c or m.body_weldid[m.body_parentid[b]]==c
                      or m.body_weldid[m.body_parentid[c]]==b)
            distance=float(mj.mj_geomDistance(m,d,g,h,.02,None))
            if distance < .02:
                row={'pair':[g,h],'distance_m':distance}
                if filtered:
                    if distance<0: excluded.append(row)
                else:
                    nearby.append(row)
                    require(distance>=-s.geometry_epsilon_m, f'Initial self intersection {g,h}: {distance}')
    return {'minimum_clearance_m':min(gaps)[0], 'nearest_pair':[f,min(gaps)[1]],
            'query_support_max_error_m':max(query_errors),'nearby_eligible_pairs':nearby,
            'adjacent_or_same_body_overlaps':excluded,
            'wrist_hip_distances_m':{f'{a}-{b}':float(mj.mj_geomDistance(m,d,a,b,2.,None))
                                    for a,b in ((9,67),(32,80))}}


def construct_pose(loaded,d,seed,settings):
    m=loaded.model; s=settings
    require(all(np.isfinite(v) and (v>=0 if k=='perturb_rad' else v>0)
                for k,v in asdict(s).items()), 'Reset settings must be finite and positive')
    require(d.model is m, 'MjData belongs to a different model')
    require(isinstance(seed,(int,np.integer)) and seed>=0, 'Explicit nonnegative integer seed required')
    require(0<=s.perturb_rad<=.005, 'Joint perturbation must be in [0, .005] rad')
    check_callbacks()
    mj.mj_resetData(m,d)
    # These defaults come from mj_resetData, not a partial hand-written reset.
    require(not np.any(d.ctrl) and not np.any(d.qvel) and not np.any(d.act)
            and not np.any(d.qacc_warmstart) and not np.any(d.qfrc_applied)
            and not np.any(d.xfrc_applied) and not np.any(d.history)
            and np.array_equal(d.eq_active,m.eq_active0), 'MuJoCo reset defaults violated')
    rng=np.random.default_rng(int(seed)); pose=nominal_pose(loaded)
    for row in loaded.mapping:
        value=pose[row.joint_name]+(rng.uniform(-s.perturb_rad,s.perturb_rad) if s.perturb_rad else 0.)
        require(row.position_range[0]<value<row.position_range[1], 'Requested joint at/outside bound')
        d.qpos[row.qpos_address]=value;pose[row.joint_name]=float(value)
    # URDF neutral: +X front, +Y left, +Z towards head. Rotation about +Y
    # by -pi/2 sends front to +Z and longitudinal +Z to -X (face up).
    mj.mju_axisAngle2Quat(d.qpos[3:7], np.array([0.,1.,0.]), -np.pi/2)
    d.qpos[:3]=0.;mj.mj_forward(m,d)
    f,n,origin,geoms=floor_geometry(m,d)
    lowest=min(plane_gap(m,d,g,n,origin) for g in geoms)
    d.qpos[:3]+=n*(s.clearance_m-lowest)
    mj.mj_forward(m,d)
    require(abs(np.linalg.norm(d.qpos[3:7])-1)<1e-12,'Invalid base quaternion')
    return {'seed':int(seed),'attempts':1,'requested_joints_rad':pose,
            'qpos':d.qpos.copy(),'geometry':initial_geometry(m,d,s)}


def measurements(loaded,d):
    m=loaded.model; f=m.geom('floor').id
    floor={'penetration_m':0.,'pair':None,'time_s':float(d.time)}
    selfp=floor.copy(); support=[]; total=back=0.
    for i,c in enumerate(d.contact):
        isfloor=f in (c.geom1,c.geom2); worst=floor if isfloor else selfp
        if -c.dist>worst['penetration_m']:
            worst.update(penetration_m=float(-c.dist),pair=[int(c.geom1),int(c.geom2)])
        force=np.empty(6);mj.mj_contactForce(m,d,i,force)
        require(np.isfinite(force).all(),'Nonfinite contact force')
        if isfloor and c.efc_address>=0 and force[0]>.01:
            robot=int(c.geom2 if c.geom1==f else c.geom1)
            world=(1 if c.geom1==f else -1)*(c.frame.reshape(3,3).T@force[:3])
            body=m.body(m.geom_bodyid[robot]).name; vertical=float(world[2])
            require(vertical>=-.01,'Unexpected ground force direction')
            total+=vertical
            if body=='torso_link': back+=vertical
            support.append({'pair':[int(c.geom1),int(c.geom2)],'body':body,
                            'vertical_N':vertical,'distance_m':float(c.dist),
                            'solref':c.solref.copy(),'solimp':c.solimp.copy()})
    mj.mj_subtreeVel(m,d)
    torso=m.body('torso_link').id;pelvis=m.body('pelvis').id
    vel=np.empty(6);mj.mj_objectVelocity(m,d,mj.mjtObj.mjOBJ_BODY,torso,vel,0)
    axes={}
    for name,b in (('torso',torso),('pelvis',pelvis)):
        R=d.xmat[b].reshape(3,3)
        axes[name+'_face_deg']=float(np.rad2deg(np.arccos(np.clip(R[2,0],-1,1))))
        axes[name+'_longitudinal_deg']=float(np.rad2deg(np.arcsin(np.clip(abs(R[2,2]),0,1))))
    q,dq=loaded.read_state(d)
    violation=max(max(r.position_range[0]-v,v-r.position_range[1],0.) for r,v in zip(loaded.mapping,q))
    return {'time_s':float(d.time),**axes,'base_linear_m_s':float(np.linalg.norm(d.qvel[:3])),
            'com_linear_m_s':float(np.linalg.norm(d.subtree_linvel[pelvis])),
            'base_angular_rad_s':float(np.linalg.norm(d.qvel[3:6])),
            'torso_angular_rad_s':float(np.linalg.norm(vel[:3])),
            'joint_rad_s':float(np.max(abs(dq))),'joint_violation_rad':float(violation),
            'floor':floor,'self':selfp,'ground_support_N':total,'back_support_N':back,
            'support':support,'com':d.subtree_com[pelvis].copy()}


def settled_failures(v,s,weight):
    failures=[]
    for key in ('torso_face_deg','pelvis_face_deg','torso_longitudinal_deg','pelvis_longitudinal_deg'):
        if v[key]>s.face_up_deg:failures.append(key)
    for key,limit in (('base_linear_m_s',s.linear_m_s),('com_linear_m_s',s.linear_m_s),
                      ('base_angular_rad_s',s.angular_rad_s),('torso_angular_rad_s',s.angular_rad_s),
                      ('joint_rad_s',s.joint_rad_s),('joint_violation_rad',s.joint_epsilon_rad)):
        if v[key]>limit:failures.append(key)
    if v['floor']['penetration_m']>s.floor_m:failures.append('floor penetration')
    if v['self']['penetration_m']>s.self_m:failures.append('self penetration')
    if not .8*weight<=v['ground_support_N']<=1.2*weight:failures.append('total ground support')
    if v['back_support_N']<.1*weight:failures.append('back load support')
    if any(c['distance_m']>s.geometry_epsilon_m for c in v['support']):failures.append('hovering contact')
    return failures


def checked_step(loaded,d,deadline):
    m=loaded.model; before=float(d.time)
    require(time.monotonic()<deadline,'Wall-time deadline')
    require(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied),'External support/force')
    mj.mj_step(m,d);mj.mj_forward(m,d)
    require(abs(d.time-before-m.opt.timestep)<1e-10,'Unexpected time progression/reset')
    require(not np.any(d.warning.number),'MuJoCo instability warning')
    require(all(np.isfinite(a).all() for a in (d.qpos,d.qvel,d.qacc,d.qfrc_constraint,d.qfrc_actuator)),
            'Nonfinite state/forces')
    require(abs(np.linalg.norm(d.qpos[3:7])-1)<1e-9,'Nonunit quaternion')


def reset_supine(loaded,d,*,seed,settings=ResetSettings(),observer=None):
    """Reset or raise ResetFailure with all collected evidence; no hidden retry."""
    s=settings;m=loaded.model
    require(s.clearance_m>0 and s.dwell_s>0 and s.hold_s>0 and s.max_settle_s>0 and s.wall_s>0,
            'Reset times and clearance must be positive')
    initial=construct_pose(loaded,d,seed,s);signature=model_signature(m)
    evidence={'status':'FAIL','criteria':asdict(s),'initial':initial,'trace':[],
              'episode_start_time':None,'window_resets':0}
    v=measurements(loaded,d);evidence['initial_metrics']=v
    peak={key:v[key].copy() for key in ('floor','self')}
    weight=float(sum(m.body_mass)*np.linalg.norm(m.opt.gravity))
    deadline=time.monotonic()+s.wall_s;start=None;hold=False;window=[];reference=None;window_peak={}
    stride=max(1,round(.04/m.opt.timestep))
    notify(m,d,observer,'placement',{'seed':seed})
    try:
        for k in range(int(np.ceil((s.max_settle_s+s.hold_s)/m.opt.timestep))):
            require(not np.any(d.ctrl),'Nonneutral reset controls')
            checked_step(loaded,d,deadline);v=measurements(loaded,d)
            for key in peak:
                if v[key]['penetration_m']>peak[key]['penetration_m']:peak[key]=v[key].copy()
            require(v['floor']['penetration_m']<=s.transient_floor_m,'Transient floor safety threshold')
            require(v['self']['penetration_m']<=s.transient_self_m,'Transient self safety threshold')
            require(v['joint_violation_rad']<=.005,'Transient joint-limit safety threshold')
            require(v['joint_rad_s']<=12.,'Transient joint-speed safety threshold')
            bad=settled_failures(v,s,weight)
            drift=np.zeros(m.nv)
            if reference is not None:
                mj.mj_differentiatePos(m,drift,1.,reference,d.qpos)
                if np.linalg.norm(drift[:3])>s.drift_m or max(np.linalg.norm(drift[3:6]),max(abs(drift[6:])))>s.drift_rad:
                    bad.append('window pose drift')
            if bad:
                if hold:raise ValueError('Unassisted hold failed: '+', '.join(bad))
                if start is not None:evidence['window_resets']+=1
                start=None;reference=None;window=[];window_peak={}
            else:
                if start is None:start=d.time;reference=d.qpos.copy()
                for key in ('floor','self'):
                    if key not in window_peak or v[key]['penetration_m']>window_peak[key]['penetration_m']:
                        window_peak[key]=v[key].copy()
                window.append({'time_s':float(d.time),'floor_m':v['floor']['penetration_m'],
                               'self_m':v['self']['penetration_m'],'joint_rad_s':v['joint_rad_s'],
                               'linear_m_s':max(v['com_linear_m_s'],v['base_linear_m_s']),
                               'angular_rad_s':max(v['torso_angular_rad_s'],v['base_angular_rad_s']),
                               'drift_m':float(np.linalg.norm(drift[:3])),
                               'drift_rad':float(max(np.linalg.norm(drift[3:6]),max(abs(drift[6:]))))})
                if d.time-start>=s.dwell_s-1e-10:hold=True
            phase='hold' if hold else ('dwell' if start is not None else 'settling')
            if k%stride==0:
                evidence['trace'].append({'phase':phase,**v,'failed_predicates':bad})
                notify(m,d,observer,phase,{'seed':seed})
            evidence.update(final=v,transient_maximum=peak)
            if hold and d.time-start>=s.dwell_s+s.hold_s-1e-10:
                require(signature==model_signature(m),'Model changed during reset')
                evidence.update(status='PASS',episode_start_time=float(d.time),
                                settled_start_s=float(start),window_maxima={key:max(row[key] for row in window)
                                    for key in window[0] if key!='time_s'},
                                window_penetration=window_peak,hold_start_s=float(start+s.dwell_s),
                                final_qpos=d.qpos.copy(),final_qvel=d.qvel.copy(),warnings=d.warning.number.copy())
                notify(m,d,observer,'ready',{'seed':seed,'episode_start_time':float(d.time)})
                return evidence
            if not hold and d.time>=s.max_settle_s-1e-10:raise ValueError('Settling timeout: '+', '.join(bad))
        raise ValueError('Settling/hold timeout')
    except ValueError as exc:
        evidence.update(error=str(exc),final=measurements(loaded,d),transient_maximum=peak)
        raise ResetFailure(str(exc),evidence) from exc
