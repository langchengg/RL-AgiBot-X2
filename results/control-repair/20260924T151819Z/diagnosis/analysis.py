#!/usr/bin/env python3
"""Rebuild the recorded-pose diagnosis, with no simulation stepping or reset.

Use the existing project Python and installed x2_recovery package (or an explicit
source PYTHONPATH). From any cwd:

  MUJOCO_GL=osmesa /path/to/repo/.venv/bin/python \
    /path/to/repo/results/control-repair/20260924T151819Z/diagnosis/analysis.py \
    --repo /path/to/repo --output /tmp/x2-offline-diagnosis-NEW

The output must be new. Source identities are checked against the adjacent
published diagnosis.json. Strict prepare verifies the full original controller;
no compatibility checks are disabled. Saved states are copied only into new,
independent MjData instances for forward geometry and rendering. Force evidence
comes from recorded integration samples, never from those reconstructed states.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
from x2_recovery.evaluate import prepare,physical_env,verify_policy
from x2_recovery.baseline import build_keyframes
from x2_recovery.env import controller_reference_targets, X2RecoveryEnv, ControlledRecoveryEnv
from x2_recovery.constraint_audit import write_json
from x2_recovery.model import require

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
parser.add_argument('--output',type=Path,required=True,help='New directory; existing output is refused')
args=parser.parse_args()
root=args.repo.resolve()
old=root/'results/controller-review/20260924T132353Z/published-policy'
hist=root/'results/evaluation/ppo-reference-residual-20260922T212457Z/trajectory.jsonl.gz'
inputdir=hist.parent/'inputs/training-run'
out=args.output.resolve()
require(not out.exists(),'Diagnostic output must not already exist')
require(not out.is_relative_to(old) and not out.is_relative_to(hist.parent), 'Output overlaps recorded inputs')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected={str(p.relative_to(root)):sha(p) for p in (old/'physical_trace.npz',old/'control_trace.npz',old/'joint_constraints.csv',old/'summary.json',hist,inputdir/'policy_final.zip',inputdir/'resolved_config.json')}
recorded=json.loads(Path(__file__).with_name('diagnosis.json').read_text())
require(protected==recorded['input_sha256'],'Recorded source identity mismatch')
out.mkdir(parents=True,exist_ok=False)
s=json.loads((old/'summary.json').read_text());origin=s['reset_handoff_absolute_s']
with np.load(old/'physical_trace.npz') as z:a={k:z[k] for k in z.files}
with np.load(old/'control_trace.npz') as z:c={k:z[k] for k in z.files}
joints=list(csv.DictReader((old/'joint_constraints.csv').open()))
ii=np.flatnonzero(a['is_recovery']);t=a['post_time_s'][ii]-origin
qadr=a['joint_qpos_addresses'];vadr=a['joint_dof_addresses'];names=list(a['joint_names'])
q=a['qpos'][ii][:,qadr];dq=a['qvel'][ii][:,vadr]
lo=np.array([float(r['q_lower_rad']) for r in joints]);hi=np.array([float(r['q_upper_rad']) for r in joints]);ex=np.maximum(np.maximum(lo-q,q-hi),0.)
measurements=[];transitions=[]
with gzip.open(hist,'rt') as f:
 for line in f:
  r=json.loads(line)
  if r['episode']!=1:break
  if r['kind']=='transition':
   measurements.extend(x['measurement'] for x in r['substeps']);transitions.append(r)
require(len(measurements)==len(t) and len(transitions)==len(c['action']),'Historical alignment length')
alignment={}
for key,current in [('time_s',a['post_time_s'][ii]),('joint_limit_rad',ex.max(axis=1)),('floor_penetration_m',a['postforward_floor_depth_m'][ii]),('self_penetration_m',a['postforward_self_depth_m'][ii])]:
 alignment[key]=float(abs(np.array([v[key] for v in measurements])-current).max())
 require(alignment[key]<=1e-12,'Historical scalar alignment failed: '+key)
for key,current in [('q_rad',c['qpos'][:,qadr]),('dq_rad_s',c['qvel'][:,vadr])]:
 alignment[key]=float(abs(np.array([v['post_step'][key] for v in transitions])-current).max());require(alignment[key]<=1e-12,'Historical state alignment failed')
alignment['actions']=float(abs(np.array([r['action'] for r in transitions])-c['action']).max());require(alignment['actions']==0,'Actions changed')
implicated=[j for j in range(31) if ex[:,j].max()>.0001]
events=[]
for j in implicated:
 for label,k in [('first_excess',np.flatnonzero(ex[:,j]>.0001)[0]),('max_excess',ex[:,j].argmax())]:events.append(dict(label=label,joint=names[j],index=int(k),elapsed_s=float(t[k])))
longest=max(s['airborne_intervals'],key=lambda v:v['sampled_duration_s'])
for label,when in [('long_airborne_start',longest['start_s']),('long_airborne_last_sample',longest['end_s']),('landing',longest['end_s']+.001),('peak_integration_ground_load',float(t[a['integration_ground_vertical_N'][ii].argmax()])),('first_residual_active',float(t[np.flatnonzero(a['residual_gate'][ii]>0)[0]]))]:
 k=int(abs(t-when).argmin());events.append(dict(label=label,joint=None,index=k,elapsed_s=float(t[k])))
selected={}
for e in events:
 for offset in (-.02,0,.02):
  k=int(abs(t-(e['elapsed_s']+offset)).argmin());selected.setdefault(k,[]).append(e['label']+':'+str(e['joint'])+f':offset={offset}')
guards=ExitStack()
for operation in ('mujoco.mj_step','x2_recovery.env.X2RecoveryEnv.reset',
                  'x2_recovery.env.X2RecoveryEnv.step','x2_recovery.env.ControlledRecoveryEnv.reset',
                  'x2_recovery.env.ControlledRecoveryEnv.step'):
 guards.enter_context(patch(operation,side_effect=RuntimeError('Offline diagnosis forbids reset/step')))
env=None
try:
 env,policy,original,p=prepare(inputdir,expected_checkpoint_sha256='11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64')
except BaseException:
 guards.close()
 raise
try:
 base=physical_env(env);m=base.model
 require(np.array_equal(qadr,base.context.qadr) and np.array_equal(vadr,base.context.vadr),'Recorded joint mapping differs')
 body_names=[m.body(i).name for i in range(m.nbody)]
 hands=[i for i,n in enumerate(body_names) if 'wrist' in n or 'hand' in n]
 def snapshot(when):
  k=int(abs(t-when).argmin());g=ii[k];d=mj.MjData(m)
  d.qpos[:]=a['qpos'][g];d.qvel[:]=a['qvel'][g]
  d.ctrl[base.context.ctrladr]=a['ctrl_Nm'][g];d.time=float(a['post_time_s'][g]);mj.mj_forward(m,d)
  require(np.array_equal(d.qpos,a['qpos'][g]) and np.array_equal(d.qvel,a['qvel'][g]),'Snapshot changed q/dq')
  return d,k
 def floor_contacts(d):
  result={}
  for ct in d.contact:
   if base.context.floor in (ct.geom1,ct.geom2) and ct.dist<=0:
    geom=int(ct.geom2 if ct.geom1==base.context.floor else ct.geom1)
    name=body_names[int(m.geom_bodyid[geom])];result[name]=max(result.get(name,0.),-float(ct.dist))
  return result
 keyframes=build_keyframes(c['reset_qpos'][qadr],base.loaded.mapping,stages=env.control_config.reference_stages)
 designed=[];native_errors=[];adoption_errors=[];previous=np.clip(c['reset_qpos'][qadr],lo,hi)
 for k,r in enumerate(transitions):
  ctl=r['info']['controller'];pre=ctl['pre_time_s'];reference,phase=controller_reference_targets(pre+base.control_dt,keyframes,env.control_config.reference_interpolation)
  designed.append(reference);requested=np.array(ctl['requested_target_rad']);adopted=np.array(ctl['adopted_target_rad'])
  predicted=previous+np.clip(requested-previous,-env.rate*base.control_dt,env.rate*base.control_dt)
  adoption_errors.append(float(abs(predicted-adopted).max()));previous=adopted
  g=ii[np.flatnonzero(a['control_index'][ii]==k)[0]];native_errors.append(float(abs(a['target_rad'][g]-adopted).max()))
 require(max(adoption_errors)<1e-12,'Adopted target limiter mismatch')
 designed=np.array(designed);requested=np.array([r['info']['controller']['requested_target_rad'] for r in transitions]);adopted=np.array([r['info']['controller']['adopted_target_rad'] for r in transitions]);gates=np.array([r['info']['controller']['residual_gate'] for r in transitions])
 require(float(abs(designed[gates==0]-requested[gates==0]).max())<1e-10,'Disabled residual requested target differs from reference')
 limiter=[]
 for j in implicated:
  gap=abs(requested[:,j]-adopted[:,j]);limiter.append(dict(joint=names[j],target_rate_limit_rad_s=float(env.rate[j]),maximum_requested_vs_adopted_gap_rad=float(gap.max()),rate_limited_control_steps=int(np.count_nonzero(gap>1e-10)),max_reference_slope_rad_s=float(abs(np.diff(keyframes.targets_rad[:,j])/np.diff(keyframes.times_s)).max()),maximum_target_step_rad=float(abs(np.diff(adopted[:,j])).max())))
 timeline=[]
 for k,labels in sorted(selected.items()):
  g=ii[k];v=measurements[k];ctl=int(a['control_index'][g]);phase=controller_reference_targets(float(transitions[ctl]['info']['controller']['pre_time_s'])+.02,keyframes,'linear-v1')[1]
  for j in implicated:
   timeline.append(dict(event_labels='|'.join(labels),post_elapsed_s=float(t[k]),pre_elapsed_s=float(a['pre_time_s'][g]-origin),joint=names[j],reference_phase=phase['name'],q_lower_rad=lo[j],q_upper_rad=hi[j],pre_q_rad=a['pre_qpos'][g,qadr[j]],post_q_rad=q[k,j],pre_dq_rad_s=a['pre_qvel'][g,vadr[j]],post_dq_rad_s=dq[k,j],excess_rad=ex[k,j],reference_at_control_endpoint_rad=designed[ctl,j],requested_target_rad=requested[ctl,j],adopted_native_target_rad=a['target_rad'][g,j],raw_pd_Nm=a['tau_raw_Nm'][g,j],clipped_control_Nm=a['ctrl_Nm'][g,j],integration_joint_torque_Nm=a['integration_qfrc_actuator_Nm'][g,vadr[j]],integration_actuator_force_Nm=a['integration_actuator_force_Nm'][g,base.context.ctrladr[j]],left_foot_weight=v['left_weight'],right_foot_weight=v['right_weight'],other_contact_norm_weight=v['other_weight'],integration_ground_vertical_N=a['integration_ground_vertical_N'][g],postforward_ground_vertical_N=a['postforward_ground_vertical_N'][g],pelvis_height_m=v['pelvis_height_m'],torso_tilt_deg=v['tilt_deg'],com_speed_m_s=v['com_linear_m_s'],torso_angular_rad_s=v['torso_angular_rad_s'],floor_penetration_m=v['floor_penetration_m'],residual_gate=a['residual_gate'][g]))
 with (out/'event_timeline.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(timeline[0]),lineterminator='\n');w.writeheader();w.writerows(timeline)
 snapshot_times=[.7,1.,1.5,1.9,2.15,2.21]
 snapshots=[];panels=[]
 renderer=mj.Renderer(m,height=320,width=480)
 camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[.0,0,.47];camera.distance=2.3;camera.azimuth=90;camera.elevation=-7
 options=mj.MjvOption();options.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT]=True
 try:
  for when in snapshot_times:
   d,k=snapshot(when)
   foot=np.array([d.xpos[b] for b in base.context.foot_bodies])
   com=d.subtree_com[base.context.pelvis].copy();contacts=floor_contacts(d)
   v=measurements[k];snapshots.append(dict(elapsed_s=float(t[k]),whole_robot_COM_xyz_m=com,pelvis_xyz_m=d.xpos[base.context.pelvis].copy(),torso_xyz_m=d.xpos[base.context.torso].copy(),foot_body_origin_xyz_m=foot,COM_from_midfoot_body_origin_xyz_m=com-foot.mean(axis=0),penetrating_floor_bodies_and_max_depth_m=contacts,recorded_support_bodyweights={n:v[n] for n in ('left_weight','right_weight','other_weight')},recorded_torso_tilt_deg=v['tilt_deg'],hand_or_wrist_body_origin_xyz_m={body_names[i]:d.xpos[i].copy() for i in hands}))
   renderer.update_scene(d,camera=camera,scene_option=options);panels.append(Image.fromarray(renderer.render()))
 finally:renderer.close()
 fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');font=ImageFont.truetype(str(fontpath),17) if fontpath.exists() else ImageFont.load_default()
 montage=Image.new('RGB',(1440,750),'white');draw=ImageDraw.Draw(montage)
 for i,panel in enumerate(panels):
  x=(i%3)*480;y=(i//3)*355;montage.paste(panel,(x,y));v=snapshots[i]['recorded_support_bodyweights'];draw.text((x+10,y+322),f"t={snapshot_times[i]:.2f}s  L/R/other={v['left_weight']:.2f}/{v['right_weight']:.2f}/{v['other_weight']:.2f} W",fill='black',font=font)
 draw.text((10,716),'Offline recorded-pose snapshots; independent MjData + forward; contact markers recomputed; NOT a new rollout.',fill='black',font=font)
 montage.save(out/'offline_pose_montage.png')
 rows=[]
 for when in np.arange(.01,2.251,.01):
  d,k=snapshot(when);ct=floor_contacts(d)
  rows.append(dict(elapsed_s=float(t[k]),penetrating_floor_bodies=sorted(ct)))
 intervals=[]
 for row in rows:
  if intervals and intervals[-1]['bodies']==row['penetrating_floor_bodies']:
   intervals[-1]['last_sample_s']=row['elapsed_s'];intervals[-1]['sample_count']+=1
  else:
   intervals.append(dict(first_sample_s=row['elapsed_s'],last_sample_s=row['elapsed_s'],sample_count=1,bodies=row['penetrating_floor_bodies']))
 contact_timeline=dict(semantics='Independent recorded-pose geometry at 10ms sampling; dist<=0 floor contact. Body names identify geometric contact, not load. Shorter contact events can be missed. Intervals use first/last sample and may include gaps shorter than 10ms. No mj_step.',sample_count=len(rows),intervals=intervals,hand_or_wrist_floor_contact_samples=sum(any(('hand' in n or 'wrist' in n) for n in r['penetrating_floor_bodies']) for r in rows))
 report=dict(schema='control-repair-offline-diagnosis-v1',input_sha256=protected,historical_scalar_and_control_alignment_max_abs=alignment,
   source_scope='Physical vectors are from the later unchanged-controller diagnostic. Support/tilt/COM-speed scalars are from historical episode 1 only after exact time/state/action and scalar alignment checks.',
   timing='q/dq are explicitly pre/post; integration torque and ground load belong to the pre-state integration solve. Postforward support loads are synchronized recomputations. 1ms physical samples; no continuous extrema claim.',
   first_and_peak_joint_events=events,implicated_joint_csv=[r for r in joints if r['joint_name'] in [names[j] for j in implicated]],
   longest_force_free_interval=longest,target_limiter_audit=dict(config_rate_multiplier=env.control_config.rate_multiplier,tracking_error_ratio=env.control_config.tracking_error_ratio,maximum_reconstructed_adoption_error_rad=max(adoption_errors),maximum_native_float32_action_roundtrip_error_rad=max(native_errors),joints=limiter),
   offline_floor_body_contact_timeline_10ms=contact_timeline,
   independent_snapshot_geometry=snapshots,snapshot_method='Create a separate fresh MjData per selected recorded state; copy saved qpos/qvel/ctrl/time; mj_forward only on that snapshot. Body/COM positions and contact markers recomputed. No recomputed contact forces used as integration evidence. Foot body origins do not define the true support polygon.',
   facts=['All five implicated joints first exceed 0.0001rad before residual activation.', 'Earliest ankle-pitch targets lie exactly on the lower model bounds while actuation remains unsaturated.', 'Waist upper-limit event occurs with opposing PD already clipped at the -48Nm motor bound.', 'Worst late left ankle pitch/roll excess occurs after the long force-free interval with target safely inside nominal limits and restoring torque saturated.'],
   inferences=['Small early ankle target margins may remove target-induced near-boundary loading, but cannot alone remove late impact excess when targets already have large margins.', 'The waist recoil and late ankle impact require load/momentum/support-transition changes rather than merely higher proportional gains.'],
   untested_hypotheses=[dict(group='early 0.4–1.05s setup',parameters=['bilateral ankle pitch lower-bound margin 0.02–0.08rad','ankle roll boundary margin 0.01–0.03rad','jointly lengthen catch-knot-1 and catch-knot-3..6 durations by 10–30%'],motivation='Address early nominal-bound targets while reducing the knee/hip-driven waist recoil; preserve a coherent prefix rather than isolated endpoint edits.'),dict(group='takeoff and landing 1.58–2.25s',parameters=['reduce or slow coupled hip/knee extension during the transfer into 1.94s boundary','delay knee-extension/waist-straightening sequence until foot load returns, or predeclare small shifted landing timing','maintain ankle pitch/roll recovery margin and reduce asymmetry through knee/hip targets'],motivation='Reduce COM launch and landing impulse so capped ankle torques can arrest rotation. Outcome and stability must be tested; no success claim from offline geometry.')],
   outputs=['event_timeline.csv','offline_pose_montage.png'],execution=dict(no_new_control_trial=True,no_reset=True,no_mj_step=True,active_environment_state_unmodified=True,policy_weights_unchanged=True,script_sha256=sha(Path(__file__)),reset_and_step_calls_blocked=True))
 verify_policy(policy,original)
 require({name:sha(root/name) for name in protected}==protected,'Protected input changed')
 write_json(out/'diagnosis.json',report)
 print(json.dumps(dict(output=str(out),events=len(events),timeline_rows=len(timeline),alignment=alignment,limiter=limiter),indent=2))
finally:
 try:
  if env is not None:env.close()
 finally:guards.close()
