"""Bounded low-dimensional reference discovery through real supine dynamics.

Search candidates are diagnostic references, never trained policies. Every trial
replays its prefix from the original reset; no state restoration is used.
"""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('search');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--candidates',type=int,default=24);p.add_argument('--generations',type=int,default=2)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--max-wall-seconds',type=float,default=1800)
    p.add_argument('--max-sim-seconds',type=float,default=8);p.add_argument('--seed',type=int,default=901)
    p.add_argument('--route',choices=('symmetric','side'),default='symmetric');p.add_argument('--center',type=Path)
    p.add_argument('--prefix',type=Path,help='Fixed, previously executed stages; search only eight suffix parameters')
    p.add_argument('--objective',choices=('posture-v1','load-transfer-v1'),default='posture-v1')
    p=sub.add_parser('replay');p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--max-sim-seconds',type=float,default=20.)
    p.add_argument('--seed',type=int,default=221100)
    args=parser.parse_args(argv)
    if args.command=='search':
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
