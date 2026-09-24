"""Bounded low-dimensional reference discovery through real supine dynamics.

Search candidates are diagnostic references, never trained policies. Every trial
replays its prefix from the original reset; no state restoration is used.
"""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .env import X2RecoveryEnv, ControlledRecoveryEnv, ControlConfig
from .model import require
from .success import CALIBRATED_SETTINGS
from .train import json_value, write_json, source_hashes, sha256, utc_now, env_identity

_BASE = None


def _worker():
    global _BASE
    import torch
    torch.set_num_threads(1)
    _BASE = X2RecoveryEnv(capture_substeps=False)
    atexit.register(_BASE.close)


def bilateral(**angles):
    result = {}
    for side, sign in (('left', 1), ('right', -1)):
        for joint, angle in angles.items():
            result[f'{side}_{joint}_joint'] = float(angle * sign if 'roll' in joint or 'yaw' in joint else angle)
    return result


def stand_target():
    return dict(waist_pitch_joint=0.,waist_roll_joint=0.,waist_yaw_joint=0.,
        **bilateral(hip_pitch=-.05,hip_roll=0.,hip_yaw=0.,knee=.1,ankle_pitch=-.05,ankle_roll=0.,
                    shoulder_pitch=-.15,shoulder_roll=.15,shoulder_yaw=0.,elbow=-.3,
                    wrist_yaw=0.,wrist_roll=0.,wrist_pitch=0.))


def stages_from_parameters(p, route='symmetric', full=False):
    p = np.asarray(p, dtype=float)
    require(p.shape == (15,) and np.isfinite(p).all(), 'Expected 15 finite reference parameters')
    hip,knee,ankle,shoulder,roll,elbow,waist,tuck_s,brace_s,hip2,knee2,ankle2,shoulder2,elbow2,rise_s = p
    tuck = dict(waist_pitch_joint=float(waist), **bilateral(hip_pitch=hip,knee=knee,ankle_pitch=ankle))
    brace = bilateral(shoulder_pitch=shoulder,shoulder_roll=roll,elbow=elbow)
    rise = dict(waist_pitch_joint=float(waist*.5), **bilateral(hip_pitch=hip2,knee=knee2,ankle_pitch=ankle2,
                                        shoulder_pitch=shoulder2,elbow=elbow2))
    if route == 'side':
        # A registered alternate route: waist yaw is longitudinal in supine.
        tuck['waist_yaw_joint'] = float(np.clip(shoulder,-1,1))
        tuck['right_hip_pitch_joint'] = float(hip*.45)
        brace.update(left_shoulder_pitch_joint=float(shoulder2),right_shoulder_pitch_joint=float(-.3),
                     left_elbow_joint=float(elbow2),right_elbow_joint=float(-1.2))
        rise['waist_yaw_joint'] = float(np.clip(shoulder*.4,-1,1))
    require(route in ('symmetric','side'), 'Unknown route')
    stages = [('handoff',.3,{}),('tuck',float(tuck_s),tuck),('brace',float(brace_s),brace),
              ('transfer',float(rise_s),rise)]
    if full:
        stages += [('extend',4.,stand_target())]
    return tuple(stages)


CENTER = np.array([-1.1,2.35,.27,.9,.45,-.6,.25,2.,1.5,-.8,2.30,-.65,1.2,-.3,2.])
LOW = np.array([-1.5,1.7,-.1,.3,.15,-1.3,-.05,1.2,.7,-1.5,1.7,-.803,.6,-1.0,1.])
HIGH = np.array([-.65,2.40,.45,1.5,.9,-.1,.30,3.0,2.5,-.3,2.40,.3,1.8,-.05,3.])


def quality(v, hold, objective='posture-v1'):
    h = float(np.clip(v['pelvis_height_m']/CALIBRATED_SETTINGS.h_ref_m,0,1.2))
    u = float(np.clip(v['upright_dot'],0,1))
    feet = float(np.clip(v['left_weight']+v['right_weight'],0,1.2))
    require(objective in ('posture-v1','load-transfer-v1'), 'Unknown search objective')
    if objective == 'load-transfer-v1':
        # Require coincident elevation, useful orientation and foot load. This
        # discovery score is not a success detector or a PPO reward.
        return 3*h*u+2*h*u*feet+.25*u+.5*h*feet+5*hold
    return 2*h+1.5*u+1.5*h*u*feet+5*hold


def trial(parameters, *, route='symmetric', full=False, max_sim_s=8., seed=221100,
          trajectory=None, rate_multiplier=2., custom_stages=None, prefix_stages=None,
          objective='posture-v1', reference_interpolation='quintic-v1'):
    base = _BASE if _BASE is not None else X2RecoveryEnv(capture_substeps=trajectory is not None)
    base.capture_substeps = trajectory is not None
    if prefix_stages is not None:
        hip,knee,ankle,shoulder,elbow,waist,duration,roll = parameters
        target=dict(waist_pitch_joint=float(waist), **bilateral(hip_pitch=hip,knee=knee,ankle_pitch=ankle,
                      shoulder_pitch=shoulder,elbow=elbow,shoulder_roll=roll))
        stages=tuple(prefix_stages)+(('unload-arms',float(duration),target),)
        if full:stages+=(('extend',3.,stand_target()),)
    else:
        stages = tuple(custom_stages) if custom_stages is not None else stages_from_parameters(parameters,route,full)
    cfg=ControlConfig(mode='reference_residual',reference_stages=stages,rate_multiplier=rate_multiplier,
                      reference_interpolation=reference_interpolation)
    env=ControlledRecoveryEnv(base,cfg)
    started=time.perf_counter();stream=None
    try:
        obs,info=env.reset(seed=seed)
        if trajectory:
            from .evaluate import reset_record, transition_record, append_record
            stream=Path(trajectory).open('x')
            append_record(stream,reset_record(base,obs,info,1,seed))
        steps=physics=0;maximum=base._measurement['pelvis_height_m'];hold=0.;raw_return=0.
        peak=base._measurement.copy();scores=[];snapshots=[];saturation=0.
        while True:
            action=np.zeros(env.action_space.shape,dtype=np.float32)
            obs,reward,terminated,truncated,info=env.step(action)
            steps+=1;physics+=info['physics_steps_executed'];raw_return+=reward
            samples=[s['measurement'] for s in base.last_substeps] if trajectory else [base._measurement]
            for v in samples:
                if v['pelvis_height_m']>maximum:maximum=v['pelvis_height_m'];peak=v.copy()
            observed_hold = max(s['success']['stable_duration_s'] for s in base.last_substeps) if trajectory else info['stable_duration_s']
            hold=max(hold,observed_hold);scores.append(quality(base._measurement,hold,objective))
            saturation+=info['torque_saturation_fraction']*info['physics_steps_executed']
            if steps%25==0 or terminated or truncated:
                q,dq=base.loaded.read_state(base.data)
                snapshots.append(dict(step=steps,elapsed_sim_s=info['elapsed_sim_s'],measurement=base._measurement,
                    q_rad=q.tolist(),dq_rad_s=dq.tolist(),target_rad=env.target.tolist(),controller=info['controller']))
            if stream:
                append_record(stream,transition_record(base,obs,action,reward,terminated,truncated,info,1,seed,steps))
            if terminated or truncated or info['elapsed_sim_s']+1e-8>=max_sim_s:break
        reason=info['termination_reason'] or info['truncation_reason'] or 'diagnostic_cutoff'
        safe=not str(reason).startswith('safety_abort:')
        # End support/posture dominates transient height. Early unsafe motions are
        # penalized, and an airborne peak alone cannot earn a standing score.
        score=(1000 if info['is_success'] else 0)+float(np.mean(scores[-min(len(scores),50):]))
        score+=.2*max(scores)
        if not safe:score-=2.
        result=dict(seed=seed,reset_seed=info['reset_seed'],controller='reference_search',parameters=list(parameters),
            route=route,full=full,control_config=asdict(cfg),max_sim_s=max_sim_s,objective=objective,
            complete_episode=bool(terminated or truncated),terminated=terminated,truncated=truncated,
            reason=reason,success=info['is_success'],score=score,transitions=steps,physics_steps=physics,
            sim_seconds=info['elapsed_sim_s'],max_height=maximum,max_stable_hold_s=hold,peak=peak,
            final=base._measurement,standing_failures=info['standing_failures'],raw_return=raw_return,
            saturation_physics_weighted=saturation/physics,snapshots=snapshots,
            metric_scope='all physical substeps for detailed replay; control-boundary samples for search ranking',
            wall_seconds=time.perf_counter()-started)
        if stream:append_record(stream,dict(kind='terminal' if terminated or truncated else 'diagnostic_cutoff',result=result))
        return json_value(result)
    finally:
        if stream:stream.close()
        if _BASE is None:env.close()


def _candidate(job):
    index,parameters,kwargs=job
    return index,trial(parameters,**kwargs)


def search(directory, *, candidates=32, generations=2, workers=2, max_wall_seconds=1800.,
           max_sim_s=8., seed=901, route='symmetric', center=None, prefix=None,objective='posture-v1'):
    directory=Path(directory);require(not directory.exists(),'Output exists')
    require(0<max_wall_seconds<=1800 and 1<=workers<=4 and candidates>=4 and generations>=1,'Invalid search budget')
    directory.mkdir(parents=True);started=time.perf_counter()
    low,high=LOW,HIGH
    prefix_stages=None
    if prefix is not None:
        prefix_stages=json.loads(Path(prefix).read_text())
        low=np.array([-2.3,.7,-.8,.5,-2.2,-.2,1.,.1])
        high=np.array([-.6,2.4,.45,2.2,-.05,.3,3.,.6])
        default=np.array([-1.5,1.7,.1,1.5,-1.,.2,2.,.15])
    else:default=CENTER
    rng=np.random.default_rng(seed);mean=default.copy() if center is None else np.asarray(center,dtype=float)
    require(mean.shape==low.shape,'Search center dimensions differ')
    sigma=(high-low)*.18
    manifest=dict(kind='development_reference_search',started_utc=utc_now(),seed=seed,route=route,
        requested_candidates=candidates*generations,workers=workers,max_wall_seconds=max_wall_seconds,
        max_sim_s=max_sim_s,source_hashes=source_hashes(),search_source_sha256=sha256(__file__),
        hypothesis='Deep tuck and arm loading can rotate pelvis toward foot support; physical success unchanged',
        parameters_lower=low,parameters_upper=high,initial_center=mean,initial_sigma=sigma,
        prefix_stages=prefix_stages,prefix_sha256=sha256(prefix) if prefix is not None else None,objective=objective)
    write_json(directory/'manifest.json',manifest)
    best=None;count=0;records=[]
    with (directory/'candidates.jsonl').open('x',buffering=1) as stream:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_worker) as pool:
            for generation in range(generations):
                if time.perf_counter()-started>=max_wall_seconds:break
                population=np.clip(rng.normal(mean,sigma,(candidates,len(mean))),low,high);population[0]=mean
                batch=[]
                # Bounded batches allow budget decisions without queueing an entire
                # generation after the wall deadline. Each task is itself <=20 sim s.
                for offset in range(0,candidates,workers):
                    if time.perf_counter()-started>=max_wall_seconds:break
                    jobs=[(count+i,p,dict(route=route,max_sim_s=max_sim_s,prefix_stages=prefix_stages,objective=objective)) for i,p in enumerate(population[offset:offset+workers])]
                    for index,result in pool.map(_candidate,jobs):
                        count+=1;result.update(candidate=index,generation=generation)
                        stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush();batch.append(result)
                        records.append({k:v for k,v in result.items() if k not in ('snapshots','control_config')})
                        if best is None or result['score']>best['score']:
                            best=result;write_json(directory/'best.json',best)
                            print(json.dumps({k:best[k] for k in ('candidate','score','reason','max_height','sim_seconds','success')}),flush=True)
                if not batch:break
                elite=sorted(batch,key=lambda x:x['score'],reverse=True)[:max(2,len(batch)//5)]
                vectors=np.array([r['parameters'] for r in elite]);mean=vectors.mean(axis=0)
                sigma=np.maximum(vectors.std(axis=0),.04*(high-low))
                write_json(directory/'search_state.json',dict(next_mean=mean,next_sigma=sigma,rng_state=rng.bit_generator.state))
    summary=dict(status='COMPLETE',completed_candidates=count,wall_seconds=time.perf_counter()-started,
        best_score=None if best is None else best['score'],successes=sum(r['success'] for r in records),
        total_transitions=sum(r['transitions'] for r in records),reason_counts={key:sum(r['reason']==key for r in records) for key in set(r['reason'] for r in records)},
        best_path='best.json',source_hashes_after=source_hashes())
    write_json(directory/'summary.json',summary);return summary


# Repair search is deliberately separate from the historical short-prefix search.
# It reuses the published full configuration and always runs complete episodes.
_REPAIR = None


def control_diff(before, after, path='controller'):
    """Exact scalar changes, including indexed reference stages; no hidden defaults."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                result.append(dict(path=path+'.'+key, before=before.get(key), after=after.get(key)))
            else:
                result.extend(control_diff(before[key], after[key], path+'.'+key))
        return result
    if isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)) and len(before) == len(after):
        return [change for index, (a, b) in enumerate(zip(before, after))
                for change in control_diff(a, b, f'{path}[{index}]')]
    return [] if before == after else [dict(path=path, before=before, after=after)]


def _digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


from .constraint_audit import PhysicalObserver


class StreamingConstraintObserver(PhysicalObserver):
    """Same checked-step forwarding, but bounded memory and one contact solve read.

    q/dq are post integration. Contacts and actuator force are copied immediately
    after mj_step, before the existing mj_forward: the actual integration solve.
    The original tracker update is forwarded once to retain exact substep hold.
    """
    def __init__(self, base):
        import hashlib
        import xml.etree.ElementTree as ET
        from .model import URDF
        super().__init__(base)
        self.target_hash = hashlib.sha256()
        self.state_hash = hashlib.sha256()
        self.n = len(base.loaded.mapping)
        self.speed_limits = np.full(self.n, np.nan)
        self.velocity_sources = []
        urdf = ET.parse(base.loaded.asset_repo/URDF).getroot()
        for i, row in enumerate(base.loaded.mapping):
            limit = urdf.find(f"joint[@name='{row.joint_name}']/limit")
            if limit is not None and limit.get('velocity') is not None:
                self.speed_limits[i] = float(limit.get('velocity'))
            self.velocity_sources.append(URDF+'#'+row.joint_name+'/limit@velocity'
                                         if np.isfinite(self.speed_limits[i]) else None)
        self.stats = {phase: self._empty() for phase in ('reset', 'recovery')}
        self.max_hold = self.max_progress = self.progress_integral = self.final_quality = 0.
        self.final_measurement = None
        self.peak_progress_measurement = None
        self.prefix_cutoff_s = None
        self.prefix = dict(samples=0, covered_through_s=0., max_joint_excess_rad=0.,
            max_joint_speed_rad_s=0., any_joint_above_tolerance_s=0.,
            peak_ground_vertical_N=0., peak_ground_sum_norm_N=0.,
            floor_penetration_m=0., self_penetration_m=0.)

    def _empty(self):
        return dict(samples=0, qmin=np.full(self.n, np.inf), qmax=np.full(self.n, -np.inf),
                    excess=np.zeros(self.n), max_time=np.zeros(self.n), duration=np.zeros(self.n),
                    duration_tolerance=np.zeros(self.n), current_run=np.zeros(self.n),
                    longest_run=np.zeros(self.n), excess_integral=np.zeros(self.n),
                    speed=np.zeros(self.n), torque_positive=np.zeros(self.n), torque_negative=np.zeros(self.n),
                    torque_ratio=np.zeros(self.n), saturation=np.zeros(self.n),
                    target_margin=np.full(self.n, np.inf), any_excess_s=0., any_tolerance_s=0.,
                    first_violation_s=None, peak_ground_vertical_N=0., peak_ground_sum_norm_N=0., floor_penetration_m=0.,
                    self_penetration_m=0., worst_floor=None, worst_self=None)

    def __enter__(self):
        super().__enter__()
        self._tracker_update = self.base.tracker.update
        def tracked(measurement):
            result = self._tracker_update(measurement)
            if self.phase == 'recovery':
                self.max_hold = max(self.max_hold, result['stable_duration_s'])
                h = float(np.clip(measurement['pelvis_height_m']/CALIBRATED_SETTINGS.h_ref_m, 0., 1.))
                u = float(np.clip(measurement['upright_dot'], 0., 1.))
                feet = float(np.clip(measurement['left_weight']+measurement['right_weight'], 0., 1.))
                self.final_quality = h*u*feet
                if self.final_quality > self.max_progress:
                    self.max_progress = self.final_quality
                    self.peak_progress_measurement = measurement.copy()
                self.progress_integral += self.final_quality*self.base.physics_dt
                self.final_measurement = measurement.copy()
            return result
        self.base.tracker.update = tracked
        return self

    def __exit__(self, *args):
        self.base.tracker.update = self._tracker_update
        return super().__exit__(*args)

    def _observed_step(self, model, data, *args, **kwargs):
        from .constraint_audit import _contacts
        if data is not self.base.data:
            return self._step(model, data, *args, **kwargs)
        require(not args and not kwargs, 'Repair observer requires a single physical step')
        base = self.base
        target = base._target.copy() if self.phase == 'recovery' else None
        q, dq = base.loaded.read_state(data)
        raw = None if target is None else base.kp*(target-q)-base.kd*dq
        ctrl = data.ctrl[base.context.ctrladr].copy()
        pre_time = float(data.time)
        self._step(model, data)
        if self.phase == 'recovery':
            self.state_hash.update(np.asarray(data.qpos, dtype='<f8').tobytes())
            self.state_hash.update(np.asarray(data.qvel, dtype='<f8').tobytes())
        self.pending = dict(q=data.qpos[base.context.qadr].copy(),
            dq=data.qvel[base.context.vadr].copy(), time=float(data.time), pre_time=pre_time,
            target=target, raw=raw, ctrl=ctrl,
            force=data.qfrc_actuator[base.context.vadr].copy(),
            actuator=data.actuator_force[base.context.ctrladr].copy(),
            contacts=_contacts(model, data, base.context.floor))

    def _observed_checked(self, loaded, data, deadline):
        self._checked(loaded, data, deadline)
        if data is not self.base.data:
            return
        row, self.pending = self.pending, None
        require(row is not None and row['time'] == float(data.time), 'Missing aligned physical sample')
        require(np.array_equal(row['q'], data.qpos[self.base.context.qadr])
                and np.array_equal(row['dq'], data.qvel[self.base.context.vadr]),
                'Forward solve unexpectedly changed physical state')
        self.consume(row)

    def consume(self, row):
        from .constraint_audit import excess, directional_ratio
        base = self.base; dt = base.physics_dt; stats = self.stats[self.phase]
        t = row['time']-(base.episode_start_time if self.phase == 'recovery' else 0.)
        ex = excess(row['q'], base.q_min, base.q_max)
        stats['samples'] += 1
        stats['qmin'] = np.minimum(stats['qmin'], row['q']); stats['qmax'] = np.maximum(stats['qmax'], row['q'])
        new_peak = ex > stats['excess']; stats['max_time'][new_peak] = t
        stats['excess'] = np.maximum(stats['excess'], ex)
        stats['duration'] += (ex > 0)*dt
        stats['duration_tolerance'] += (ex > CALIBRATED_SETTINGS.joint_limit_rad_max)*dt
        stats['current_run'] = np.where(ex > 0, stats['current_run']+dt, 0.)
        stats['longest_run'] = np.maximum(stats['longest_run'], stats['current_run'])
        stats['excess_integral'] += ex*dt
        stats['speed'] = np.maximum(stats['speed'], abs(row['dq']))
        stats['torque_positive'] = np.maximum(stats['torque_positive'], row['force'])
        stats['torque_negative'] = np.minimum(stats['torque_negative'], row['force'])
        stats['torque_ratio'] = np.maximum(stats['torque_ratio'], directional_ratio(row['force'], base.context.efforts))
        # The pinned model validates unit gear and one motor per joint. Check that
        # actuator force maps to the joint generalized force actually integrated.
        require(np.allclose(row['force'], row['actuator'], atol=1e-10, rtol=0), 'Actuator/joint force mapping changed')
        if row['target'] is not None:
            stats['saturation'] += row['raw'] != row['ctrl']
            stats['target_margin'] = np.minimum(stats['target_margin'],
                np.minimum(row['target']-base.q_min, base.q_max-row['target']))
            self.target_hash.update(np.asarray(row['target'], dtype='<f8').tobytes())
        stats['any_excess_s'] += float(np.any(ex > 0))*dt
        stats['any_tolerance_s'] += float(np.any(ex > CALIBRATED_SETTINGS.joint_limit_rad_max))*dt
        if stats['first_violation_s'] is None and np.any(ex > CALIBRATED_SETTINGS.joint_limit_rad_max):
            stats['first_violation_s'] = t
        contacts = row['contacts']
        if (self.phase == 'recovery' and getattr(self, 'prefix_cutoff_s', None) is not None
                and t <= self.prefix_cutoff_s+1e-10):
            p = self.prefix
            p['samples'] += 1; p['covered_through_s'] = t
            p['max_joint_excess_rad'] = max(p['max_joint_excess_rad'], float(ex.max()))
            p['max_joint_speed_rad_s'] = max(p['max_joint_speed_rad_s'], float(abs(row['dq']).max()))
            p['any_joint_above_tolerance_s'] += float(np.any(ex > CALIBRATED_SETTINGS.joint_limit_rad_max))*dt
            for key in ('ground_vertical_N', 'ground_sum_norm_N'):
                p['peak_'+key] = max(p['peak_'+key], contacts[key])
            for kind in ('floor', 'self'):
                p[kind+'_penetration_m'] = max(p[kind+'_penetration_m'], contacts[kind]['depth_m'])
        stats['peak_ground_vertical_N'] = max(stats['peak_ground_vertical_N'], contacts['ground_vertical_N'])
        stats['peak_ground_sum_norm_N'] = max(stats['peak_ground_sum_norm_N'], contacts['ground_sum_norm_N'])
        for kind, key in [('floor', 'floor_penetration_m'), ('self', 'self_penetration_m')]:
            if contacts[kind]['depth_m'] > stats[key]:
                stats[key] = contacts[kind]['depth_m']
                stats['worst_'+kind] = dict(contacts[kind], integration_pre_time_s=row['pre_time']-
                    (base.episode_start_time if self.phase == 'recovery' else 0.))

    def summary(self, phase):
        s = self.stats[phase]; n = s['samples']
        require(n > 0, 'No physical samples for '+phase)
        speed_ratio = np.divide(s['speed'], self.speed_limits)
        known = np.isfinite(speed_ratio) & (self.speed_limits > 0)
        rows = []
        for i, joint in enumerate(self.base.loaded.mapping):
            rows.append(dict(joint_name=joint.joint_name, q_lower_rad=float(self.base.q_min[i]),
                q_upper_rad=float(self.base.q_max[i]), q_min_observed_rad=float(s['qmin'][i]),
                q_max_observed_rad=float(s['qmax'][i]), max_excess_rad=float(s['excess'][i]),
                max_lower_excess_rad=float(max(0., self.base.q_min[i]-s['qmin'][i])),
                max_upper_excess_rad=float(max(0., s['qmax'][i]-self.base.q_max[i])),
                time_of_max_excess_s=float(s['max_time'][i]), time_above_zero_excess_s=float(s['duration'][i]),
                time_above_standing_tolerance_s=float(s['duration_tolerance'][i]),
                longest_consecutive_excess_s=float(s['longest_run'][i]),
                excess_integral_rad_s=float(s['excess_integral'][i]),
                max_abs_velocity_rad_s=float(s['speed'][i]),
                declared_velocity_limit_rad_s=float(self.speed_limits[i]) if known[i] else None,
                velocity_limit_source=self.velocity_sources[i],
                max_velocity_ratio=float(speed_ratio[i]) if known[i] else None,
                max_positive_actuation_Nm=float(s['torque_positive'][i]),
                max_negative_actuation_Nm=float(s['torque_negative'][i]),
                max_actuation_limit_ratio=float(s['torque_ratio'][i]),
                torque_saturation_fraction=float(s['saturation'][i]/n),
                min_target_limit_margin_rad=float(s['target_margin'][i]) if np.isfinite(s['target_margin'][i]) else None))
        worst = int(np.argmax(s['excess']))
        return dict(samples=n, sampled_duration_s=n*self.base.physics_dt, joints=rows,
            worst_joint=rows[worst]['joint_name'], max_joint_excess_rad=float(s['excess'][worst]),
            max_joint_speed_rad_s=float(s['speed'].max()),
            max_velocity_ratio=float(speed_ratio[known].max()) if known.any() else None,
            velocity_limits_complete=bool(known.all()), max_actuation_limit_ratio=float(s['torque_ratio'].max()),
            excess_integral_rad_s=float(s['excess_integral'].sum()),
            any_joint_above_zero_s=s['any_excess_s'], any_joint_above_tolerance_s=s['any_tolerance_s'],
            first_violation_s=s['first_violation_s'], peak_ground_vertical_N=s['peak_ground_vertical_N'],
            peak_ground_vertical_bodyweights=s['peak_ground_vertical_N']/self.base.context.weight_N,
            peak_ground_sum_norm_N=s['peak_ground_sum_norm_N'],
            floor_penetration_m=s['floor_penetration_m'], self_penetration_m=s['self_penetration_m'],
            worst_floor=s['worst_floor'], worst_self=s['worst_self'])


def repair_rank(result, comparison):
    """Feasibility first; no standing bonus can buy a constraint violation."""
    keys = ('peak_ground_vertical_N', 'peak_ground_sum_norm_N', 'floor_penetration_m', 'self_penetration_m', 'max_joint_speed_rad_s')
    require(all(k in comparison and np.isfinite(comparison[k]) and comparison[k] > 0 for k in keys),
            'Missing preregistered baseline comparison')
    m = result['recovery']; tol = 1e-10
    envelope = (m['max_joint_excess_rad'] <= .0001+tol and m['max_actuation_limit_ratio'] <= 1.+tol
                and m['velocity_limits_complete'] and m['max_velocity_ratio'] <= 1.+tol)
    impact = all(m[k] <= comparison[k]+tol for k in keys)
    feasible = bool(result['complete_episode'] and result['success'] and envelope and impact)
    completed = result['complete_episode']
    eligible = bool(completed and (result['success'] or
        (result['reason'] == 'time_limit' and result['max_progress_quality'] >= .25)))
    timeout = result['episode_timeout_s']
    progress = result['progress_integral_s']/timeout+result['final_quality']+result['max_stable_hold_s']/2.
    violation = (m['max_joint_excess_rad']/.05+m['excess_integral_rad_s']/(.05*timeout)
                 +m['any_joint_above_tolerance_s']/timeout
                 +max(0., (m['max_velocity_ratio'] or 0.)-1.)
                 +max(0., m['max_actuation_limit_ratio']-1.)
                 +sum(max(0., m[k]/comparison[k]-1.) for k in keys))
    early_abort_penalty = 0. if result['success'] or result['reason'] == 'time_limit' else 2.*max(0., 1.-result['sim_seconds']/timeout)
    return dict(feasible=feasible, constraint_envelope_pass=bool(envelope), impact_pass=bool(impact),
                infeasible_eligible=eligible, progress_score=float(progress), violation_score=float(violation),
                infeasible_score=float(progress-violation-early_abort_penalty),
                ranking_scope='Only original standing + envelope + nonworse impact is feasible. '
                'No-motion and safety-aborted trials cannot become best_infeasible; all remain recorded.')


def experimental_pd_config(original, overrides):
    """One explicit experiment: preserve all groups/Kp, waist Kd in [1,6]x."""
    if overrides is None:
        return original
    require(type(overrides) is dict and set(overrides) == {'pd_gains'},
            'Only explicit pd_gains environment overrides are permitted')
    rows = overrides['pd_gains']
    require(type(rows) in (list, tuple) and len(rows) == len(original.pd_gains),
            'Expected complete PD groups in original order')
    require(all(type(r) in (list, tuple) and len(r) == 3 for r in rows), 'Invalid PD rows')
    for row, old in zip(rows, original.pd_gains):
        require(row[0] == old[0] and row[1] == old[1], 'PD group order/names and Kp must remain unchanged')
        require(type(row[2]) in (int, float) and np.isfinite(row[2]), 'Nonfinite/non-numeric Kd')
        if row[0] == 'waist':
            require(old[2] <= row[2] <= 6.*old[2], 'Waist Kd multiplier must be in [1,6]')
        else:
            require(row[2] == old[2], 'Only waist Kd may change')
    require(sum(r[0] == 'waist' for r in rows) == 1, 'Expected exactly one waist group')
    return replace(original, pd_gains=tuple(tuple(row) for row in rows))


@contextmanager
def experimental_pd(base, original, overrides):
    """Change only declared gain arrays/config before reset; always restore them."""
    require(asdict(base.config) == asdict(original), 'Physical/reset configuration changed')
    require(not base._ready, 'Cannot change PD in an active episode')
    candidate = experimental_pd_config(original, overrides)
    groups = [row.joint_name.removeprefix('left_').removeprefix('right_').split('_')[0]
              for row in base.loaded.mapping]
    def arrays(config):
        gains = {group: (kp, kd) for group, kp, kd in config.pd_gains}
        return np.asarray([gains[group] for group in groups], dtype=float).T
    expected_kp, expected_kd = arrays(original)
    require(np.array_equal(base.kp, expected_kp) and np.array_equal(base.kd, expected_kd),
            'Original configuration and actual gain arrays disagree')
    old_config, old_kp, old_kd = base.config, base.kp.copy(), base.kd.copy()
    try:
        base.config = candidate
        base.kp[:], base.kd[:] = arrays(candidate)
        yield candidate
    finally:
        base.config = old_config
        base.kp[:], base.kd[:] = old_kp, old_kd


def constraint_trial(resolved_config, control, *, seed, policy=None, base=None,
                     zero_residual=True, comparison=None, name=None, env_control_overrides=None):
    """Complete real-reset trial under the saved physical configuration, no cutoff.

    Passing a prepared base/policy reuses resources only; no state is restored.
    A changed control with frozen actor weights is explicitly a weight transfer.
    """
    from .train import TrainConfig
    saved = json.loads(Path(resolved_config).read_text()) if isinstance(resolved_config, (str, Path)) else resolved_config
    require(saved.get('schema') == 'x2-controller-ppo-v1', 'Expected full saved controller configuration')
    require(type(seed) is int and 0 <= seed < 2**32, 'Invalid legal reset seed')
    require(isinstance(control, ControlConfig), 'Explicit complete ControlConfig required')
    require(zero_residual or policy is not None, 'Frozen policy required for residual trials')
    cfg = TrainConfig.from_dict(saved['training'])
    own = base is None
    base = X2RecoveryEnv(cfg.env, capture_substeps=False) if own else base
    old_capture = base.capture_substeps; base.capture_substeps = False
    import hashlib
    action_hash = hashlib.sha256()
    started = time.monotonic(); steps = 0; total_reward = 0.
    try:
        with experimental_pd(base, cfg.env, env_control_overrides) as experimental_config:
            env = ControlledRecoveryEnv(base, control)
            with StreamingConstraintObserver(base) as observer:
                observer.prefix_cutoff_s = sum(stage[1] for stage in control.reference_stages[:13])
                obs, info = env.reset(seed=seed)
                reset_seed = info['reset_seed']; require(base.reset_evidence['status'] == 'PASS', 'Illegal reset')
                observer.phase = 'recovery'
                while True:
                    action = np.zeros(env.action_space.shape, np.float32) if zero_residual else policy.predict(obs, deterministic=True)[0]
                    action_hash.update(np.asarray(action, dtype='<f8').tobytes())
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1; total_reward += reward
                    if terminated or truncated: break
            require(observer.stats['recovery']['samples'] == base._physics_steps, 'Physical sample count mismatch')
            changes = control_diff(saved['controller'], json_value(asdict(control)))
            env_changes = control_diff(json_value(asdict(cfg.env)), json_value(asdict(experimental_config)), path='environment')
            resolved = base.resolved_config()
            result = dict(name=name, status='COMPLETE', seed=seed, reset_seed=reset_seed,
                controller='zero_residual_reference' if zero_residual else 'frozen_policy_transfer',
                weight_transfer=bool((changes or env_changes) and not zero_residual), control_sha256=_digest(asdict(control)),
                env_control_overrides=env_control_overrides, environment_control_changes=env_changes,
                experiment_env_config=asdict(experimental_config), experiment_env_sha256=_digest(asdict(experimental_config)),
                resolved_environment_sha256=_digest(resolved),
                resolved_pd_gains=dict(joint_names=[r.joint_name for r in base.loaded.mapping],
                    kp_Nm_rad=base.kp.tolist(), kd_Nm_s_rad=base.kd.tolist()),
                changed_fields=changes, actual_target_sha256=observer.target_hash.hexdigest(),
                physical_state_sha256=observer.state_hash.hexdigest(), policy_action_sha256=action_hash.hexdigest(),
                fingerprint_encoding='Concatenated little-endian float64: full post-step qpos then qvel per physical step; action per transition.',
                complete_episode=bool(terminated or truncated), terminated=bool(terminated), truncated=bool(truncated),
                success=bool(info['is_success']), reason=info['termination_reason'] or info['truncation_reason'],
                sim_seconds=float(info['elapsed_sim_s']), episode_timeout_s=cfg.env.episode_timeout_s,
                transitions=steps, physics_steps=base._physics_steps, raw_return=float(total_reward),
                max_stable_hold_s=observer.max_hold, max_progress_quality=observer.max_progress,
                progress_integral_s=observer.progress_integral, final_quality=observer.final_quality,
                final_measurement=observer.final_measurement, standing_failures=info['standing_failures'],
                peak_progress_measurement=observer.peak_progress_measurement,
                reset=observer.summary('reset'), recovery=observer.summary('recovery'),
                prefix=dict(observer.prefix, cutoff_s=observer.prefix_cutoff_s,
                    complete=float(info['elapsed_sim_s'])+1e-10 >= observer.prefix_cutoff_s,
                    definition='Recovery post-step samples at or before the first13 reference stage durations; partial if episode ended earlier.'),
                wall_seconds=time.monotonic()-started,
                sampling='Every physics step; right-endpoint durations. Integration-solve force/contact at pre-time, '
                         'joint q/dq at post-time. No extra active forward/step or state writes.')
            if comparison is not None: result.update(repair_rank(result, comparison))
            return json_value(result)
    finally:
        base.capture_substeps = old_capture
        if own: base.close()


def _repair_worker(control_input, expected_checkpoint):
    global _REPAIR
    from .evaluate import prepare, physical_env
    env, policy, original, prepared = prepare(control_input, expected_checkpoint_sha256=expected_checkpoint)
    _REPAIR = dict(env=env, base=physical_env(env), policy=policy, original=original, prepared=prepared)
    atexit.register(env.close)


def _repair_candidate(job):
    index, candidate, seed, zero_residual, comparison = job
    try:
        result = constraint_trial(_REPAIR['prepared']['saved'], ControlConfig.from_dict(candidate['control_config']),
            seed=seed, policy=_REPAIR['policy'], base=_REPAIR['base'], zero_residual=zero_residual,
            comparison=comparison, name=candidate['name'],
            env_control_overrides=candidate.get('env_control_overrides'))
        from .evaluate import verify_policy
        verify_policy(_REPAIR['policy'], _REPAIR['original'])
        require(asdict(_REPAIR['base'].config) == asdict(_REPAIR['prepared']['config'].env),
                'Original environment configuration was not restored')
        result.update(index=index, policy_state_unchanged=True, original_environment_config_restored=True)
        return result
    except Exception as exc:
        return dict(index=index, name=candidate['name'], status='ERROR', error=dict(type=type(exc).__name__, message=str(exc)))


def repair_progress(candidates, recorded, scheduled):
    """Journal records are authoritative; never skip a partially completed batch."""
    indices = [r['index'] for r in recorded]
    require(len(indices) == len(set(indices)), 'Duplicate recorded candidate index')
    require(all(type(i) is int and 0 <= i < len(candidates) for i in indices), 'Invalid recorded candidate index')
    require(all(r['name'] == candidates[r['index']]['name'] for r in recorded), 'Recorded candidate identity mismatch')
    completed = set(indices); remaining = sorted(set(range(len(candidates)))-completed)
    return dict(completed=len(completed), recorded_indices=sorted(completed),
        pending_indices=sorted(set(scheduled)-completed), unrecorded_indices=remaining,
        next_index=remaining[0] if remaining else None,
        resume_rule='Reconstruct remaining candidates from candidates.jsonl unique indices and plan identities; next_index is a hint only.')


def repair_search(directory, control_input, expected_checkpoint, plan, *, workers=4, max_wall_seconds=1800.):
    import csv
    import shutil
    directory = Path(directory).resolve(); control_input = Path(control_input).resolve(); plan_path = Path(plan).resolve()
    require(not directory.exists() and not directory.is_relative_to(control_input), 'Unsafe/existing output')
    require(1 <= workers <= 4 and 0 < max_wall_seconds <= 1800., 'Invalid repair block budget')
    specification = json.loads(plan_path.read_text()); candidates = specification['candidates']
    require(1 <= len(candidates) <= 32 and len({c['name'] for c in candidates}) == len(candidates),
            'Expected 1..32 uniquely named candidates (normal search batch 20..32)')
    require(type(specification['zero_residual']) is bool, 'Explicit controller selection required')
    from .train import TrainConfig
    original_env = TrainConfig.from_dict(json.loads((control_input/'resolved_config.json').read_text())['training']).env
    feature = specification.get('required_search_feature')
    require(feature in (None, 'explicit_env_pd_gains_override_v1'), 'Unsupported required search feature')
    if any('env_control_overrides' in c for c in candidates):
        require(feature == 'explicit_env_pd_gains_override_v1', 'PD overrides require an explicit plan feature')
    for candidate in candidates:
        experimental = experimental_pd_config(original_env, candidate.get('env_control_overrides'))
        if 'experiment_env_config' in candidate:
            require(json_value(asdict(experimental)) == candidate['experiment_env_config'], 'Experiment environment config disagrees with overrides')
        require(isinstance(candidate['name'], str) and candidate['name'], 'Invalid candidate name')
        require(set(candidate['control_config']) == set(asdict(ControlConfig())), 'Every candidate requires a complete ControlConfig')
        ControlConfig.from_dict(candidate['control_config'])
    comparison = specification['comparison']; seed = specification['seed']
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, directory/'plan.json')
    manifest = dict(kind='full_episode_constraint_repair_search', started_utc=utc_now(),
        control_input=str(control_input), expected_checkpoint_sha256=expected_checkpoint,
        input_sha256={p.name: sha256(p) for p in control_input.iterdir() if p.is_file()},
        plan_sha256=sha256(plan_path), source_sha256=sha256(__file__), core_sources=source_hashes(),
        workers=workers, max_wall_seconds=max_wall_seconds, requested_candidates=len(candidates),
        comparison=comparison, seed=seed, zero_residual=specification['zero_residual'],
        supported_environment_control_overrides='Complete original PD groups; only waist Kd in [1,6]x, all Kp and other groups unchanged.',
        required_search_feature=feature,
        feasibility_scope='Recovery physical samples only; reset settling is separately measured and retained.',
        budget_scope='No task cutoff; scheduling stops at block deadline. Already-running complete episodes may overrun.')
    write_json(directory/'manifest.json', manifest)
    start = time.monotonic(); results = []; best_feasible = best_infeasible = None
    scalar_fields = ['index','name','status','success','reason','sim_seconds','max_stable_hold_s',
                     'feasible','constraint_envelope_pass','impact_pass','infeasible_eligible','infeasible_score',
                     'max_joint_excess_rad','any_joint_above_tolerance_s','excess_integral_rad_s',
                     'max_joint_speed_rad_s','max_velocity_ratio','max_actuation_limit_ratio',
                     'peak_ground_vertical_N','peak_ground_sum_norm_N','floor_penetration_m','self_penetration_m','actual_target_sha256','wall_seconds']
    with (directory/'candidates.jsonl').open('x', buffering=1) as stream, (directory/'results.csv').open('x', newline='') as csvstream:
        writer = csv.DictWriter(csvstream, fieldnames=scalar_fields); writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                initializer=_repair_worker, initargs=(str(control_input), expected_checkpoint)) as pool:
            for offset in range(0, len(candidates), workers):
                if time.monotonic()-start >= max_wall_seconds: break
                jobs = [(i, candidates[i], seed, specification['zero_residual'], comparison)
                        for i in range(offset, min(offset+workers, len(candidates)))]
                scheduled = [job[0] for job in jobs]
                write_json(directory/'search_state.json', repair_progress(candidates, results, scheduled))
                for result in pool.map(_repair_candidate, jobs):
                    results.append(result); stream.write(json.dumps(result, allow_nan=False)+'\n')
                    flattened = dict(result, **result.get('recovery', {}))
                    writer.writerow({key: flattened.get(key) for key in scalar_fields}); csvstream.flush()
                    if result['status'] == 'COMPLETE':
                        if result['feasible'] and (best_feasible is None or result['sim_seconds'] < best_feasible['sim_seconds']):
                            best_feasible = result
                            write_json(directory/'best_feasible.json', dict(result, control_config=candidates[result['index']]['control_config']))
                        elif not result['feasible'] and result['infeasible_eligible'] and (best_infeasible is None or
                                result['infeasible_score'] > best_infeasible['infeasible_score']):
                            best_infeasible = result
                            write_json(directory/'best_infeasible.json', dict(result, control_config=candidates[result['index']]['control_config']))
                    write_json(directory/'search_state.json', dict(repair_progress(candidates, results, scheduled),
                        best_feasible=None if best_feasible is None else best_feasible['name'],
                        best_infeasible=None if best_infeasible is None else best_infeasible['name']))
                    print(json.dumps({key: flattened.get(key) for key in scalar_fields}, allow_nan=False), flush=True)
    complete = [r for r in results if r['status'] == 'COMPLETE']
    hashes = {}
    for r in complete: hashes.setdefault(r['actual_target_sha256'], []).append(r['name'])
    summary = dict(status='COMPLETE' if len(results) == len(candidates) else 'BUDGET_STOP',
        completed_candidates=len(complete), error_candidates=len(results)-len(complete),
        planned_candidates=len(candidates), unscheduled_candidates=len(candidates)-len(results),
        successful_recoveries=sum(r['success'] for r in complete), feasible_candidates=sum(r['feasible'] for r in complete),
        best_feasible=None if best_feasible is None else best_feasible['name'],
        best_infeasible=None if best_infeasible is None else best_infeasible['name'],
        target_equivalence_groups=[names for names in hashes.values() if len(names) > 1],
        wall_seconds=time.monotonic()-start, core_sources_after=source_hashes(),
        source_sha256_after=sha256(__file__))
    write_json(directory/'summary.json', summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('search');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--candidates',type=int,default=24);p.add_argument('--generations',type=int,default=2)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-wall-seconds',type=float,default=1800)
    p.add_argument('--max-sim-seconds',type=float,default=8);p.add_argument('--seed',type=int,default=901)
    p.add_argument('--route',choices=('symmetric','side'),default='symmetric');p.add_argument('--center',type=Path)
    p.add_argument('--prefix',type=Path,help='Fixed, previously executed stages; search only eight suffix parameters')
    p.add_argument('--objective',choices=('posture-v1','load-transfer-v1'),default='posture-v1')
    p=sub.add_parser('repair-search');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--control-input',required=True,type=Path);p.add_argument('--expected-checkpoint-sha256',required=True)
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-wall-seconds',type=float,default=1800.)
    p=sub.add_parser('replay');p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--max-sim-seconds',type=float,default=20.)
    p.add_argument('--seed',type=int,default=221100)
    args=parser.parse_args(argv)
    if args.command=='repair-search':
        result=repair_search(args.output,args.control_input,args.expected_checkpoint_sha256,args.plan,
                             workers=args.workers,max_wall_seconds=args.max_wall_seconds)
    elif args.command=='search':
        center=json.loads(args.center.read_text())['parameters'] if args.center else None
        result=search(args.output,candidates=args.candidates,generations=args.generations,workers=args.workers,
            max_wall_seconds=args.max_wall_seconds,max_sim_s=args.max_sim_seconds,seed=args.seed,route=args.route,center=center,prefix=args.prefix,objective=args.objective)
    else:
        require(not args.output.exists(),'Output exists');args.output.mkdir(parents=True)
        source=json.loads(args.input.read_text()) if args.input else {}
        stages=source.get('control_config',{}).get('reference_stages')
        if args.full and stages is not None:
            stages=list(stages)+[('extend',3.,stand_target())]
        result=trial(source.get('parameters',CENTER),route=source.get('route','symmetric'),full=args.full,
            max_sim_s=args.max_sim_seconds,seed=args.seed,trajectory=args.output/'trajectory.jsonl',custom_stages=stages,
            rate_multiplier=source.get('control_config',{}).get('rate_multiplier',2.),
            reference_interpolation=source.get('control_config',{}).get('reference_interpolation','quintic-v1'),
            objective=source.get('objective','posture-v1'))
        write_json(args.output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('snapshots','peak','final','control_config')},allow_nan=False))

if __name__=='__main__':main()
