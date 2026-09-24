"""Read-only matched1kHz contact transitions; no physics or force re-solve."""
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import gzip,json,hashlib
import numpy as np,mujoco as mj
from x2_recovery.env import X2RecoveryEnv
r=Path('/home/lang/RL-AgiBot-X2');run=r/'results/control-repair/20260924T151819Z';hist=r/'results/evaluation/ppo-reference-residual-20260922T212457Z/trajectory.jsonl.gz';old=r/'results/controller-review/20260924T132353Z/published-policy';new=run/'trained-policy-diagnostic';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();W=411.69110013;eps=1e-9;dt=.001
with gzip.open(hist,'rt')as f:
 rows=[]
 for line in f:
  item=json.loads(line)
  if item.get('kind')=='transition' and item.get('episode')==1:rows.append(item)
  elif rows and item.get('episode',1)>1:break
c=np.load(old/'control_trace.npz');pold=np.load(old/'physical_trace.npz');mold=pold['is_recovery'];qadr=pold['joint_qpos_addresses'];vadr=pold['joint_dof_addresses'];postms=[s['measurement']for x in rows for s in x['substeps']]
identity={
 'transition_count_equal':len(rows)==len(c['action']),
 'historical_poststep_observations_match_diagnostic_next_observation_exact':np.array_equal(np.array([x['observation']for x in rows]),c['next_observation']),
 'actions_exact':np.array_equal(np.array([x['action']for x in rows]),c['action']),
 'rewards_exact':np.array_equal(np.array([x['reward']for x in rows]),c['reward']),
 'joint_positions_exact':np.array_equal(np.array([x['post_step']['q_rad']for x in rows]),c['qpos'][:,qadr]),
 'joint_velocities_exact':np.array_equal(np.array([x['post_step']['dq_rad_s']for x in rows]),c['qvel'][:,vadr]),
 'all_physics_times_exact':np.array_equal(np.array([x['time_s']for x in postms]),pold['post_time_s'][mold]),
 'post_floor_depth_exact':np.array_equal(np.array([x['floor_penetration_m']for x in postms]),pold['postforward_floor_depth_m'][mold])}
assert all(identity.values()),identity
result={'schema':'contact-transition-comparison-v1','no_new_reset_step_training_or_force_solve':True,'historical_measurement_alignment':identity,'source_historical_trajectory_sha256':sha(hist),'definitions':{'global_peak':'Maximum sum of ground-contact vertical forces from the original integration solve, across all robot bodies. It is not automatically a foot-landing peak.','integration_timing':'Force and geometry are computed at pre_time and used over [pre_time, post_time). Saved qpos/qvel are post-integration states.','postforward_timing':'The existing checked_step mj_forward computes contact forces and measurements at post_time. These synchronized recomputations differ from forces used during integration.','zero_ground_load':'Sum of all ground-contact force norms <= 1e-9 N; also compare total vertical force <= 1e-9 N. This measures force, not complete geometric separation.','post_sample_intervals':'A truepostsample k isrepresented by(post_time[k]−.001,post_time[k]]; duration iscount*.001. Integration intervals use actual[pre,post).','load_return':'First loaded sample immediately after a zero-load sample. No event is inferred at row 0.','foot_load':'Left/right vertical foot loads from original synchronized post-step measurements at 1 kHz. Their sum <= 1e-9 N defines zero foot load; other bodies may still contact.','bilateral_support_return':'First post-step sample with both feet >= 0.05 W after that predicate was false. This uses the original foot-support threshold and does not imply full standing.','hundred_ms_window':'Union of inclusive records where post_time is in [event_time, event_time + 0.1 s], with 1e-10 s floating-point tolerance. Original integration forces from those records retain their pre_time. The window peak need not occur at its first sample; all bodies contribute.','geometric_snapshot':'Independent MjData at saved pre_qpos; kinematics and collision only, without a forward dynamics solve. Contact bodies describe geometry, not original per-body forces.'},'controllers':{}}

def spans(mask,pre,post):
 e=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))).reshape(-1,2)
 return [dict(start_s=float(pre[a]),end_s=float(post[b-1]),samples=int(b-a),duration_s=float((b-a)*dt))for a,b in e]
def events(mask,t):return [int(i)for i in range(1,len(mask))if mask[i-1]and not mask[i]]
def window(mask_event,t,p,allmask):
 e=events(mask_event,t);union=np.zeros(len(t),bool)
 for i in e:union|=(t>=t[i]-1e-10)&(t<=t[i]+.1+1e-10)
 if not np.any(union):return dict(event_count=len(e),events_elapsed_s=[],union_samples=0)
 idx=np.flatnonzero(union);values=p['integration_ground_vertical_N'][allmask];ip=idx[np.argmax(values[idx])];ipost=idx[np.argmax(p['postforward_ground_vertical_N'][allmask][idx])]
 return dict(event_count=len(e),events_elapsed_s=[float(t[i])for i in e],union_samples=int(union.sum()),union_sampled_duration_s=float(union.sum()*dt),integration_allbody_peak_vertical_N=float(values[ip]),integration_peak_pre_elapsed_s=float(t[ip]-dt),integration_peak_post_elapsed_s=float(t[ip]),integration_sum_contact_norm_peak_N=float(p['integration_ground_sum_norm_N'][allmask][idx].max()),postforward_allbody_peak_vertical_N=float(p['postforward_ground_vertical_N'][allmask][ipost]),postforward_peak_elapsed_s=float(t[ipost]),integration_floor_max_m=float(p['integration_floor_depth_m'][allmask][idx].max()))
with ExitStack()as guards:
 for fn in ['mujoco.mj_step','x2_recovery.env.X2RecoveryEnv.reset','x2_recovery.env.X2RecoveryEnv.step']:guards.enter_context(patch(fn,side_effect=RuntimeError('Read-only contact review')))
 base=X2RecoveryEnv()
 try:
  model=base.model;data=mj.MjData(model);assert abs(base.context.weight_N-W)<1e-9
  for label,directory in [('historical_published_mixed',old),('new_trained_v6',new)]:
   p=np.load(directory/'physical_trace.npz');s=json.load(open(directory/'summary.json'));mask=p['is_recovery'];origin=s['reset_handoff_absolute_s'];pre=p['pre_time_s'][mask]-origin;post=p['post_time_s'][mask]-origin
   if label=='historical_published_mixed':ms=postms
   else:ms=[dict(zip(p['measurement_scalar_names'],row))for row in p['recovery_measurement_scalars']]
   assert len(ms)==len(post)and np.array_equal(np.array([x['time_s']for x in ms]),p['post_time_s'][mask])
   loads=np.array([[x['left_weight'],x['right_weight'],x['other_weight']]for x in ms]);ig=p['integration_ground_sum_norm_N'][mask]<=eps;pg=p['postforward_ground_sum_norm_N'][mask]<=eps;fv=(loads[:,:2].sum(axis=1)*W)<=eps;bil=np.all(loads[:,:2]>=.05,axis=1)
   conditions={}
   for name,low in [('integration_allground',ig),('postforward_allground',pg),('postforward_feet_only',fv)]:
    ss=spans(low,pre,post);conditions[name]=dict(interval_count=len(ss),total_sampled_duration_s=float(low.sum()*dt),longest_sampled_duration_s=max([x['duration_s']for x in ss],default=0))
    if name!='postforward_feet_only':conditions[name]['intervals']=ss
   peak=int(np.argmax(p['integration_ground_vertical_N'][mask]));j=np.flatnonzero(mask)[peak];data.qpos[:]=p['pre_qpos'][j];mj.mj_kinematics(model,data);mj.mj_comPos(model,data);mj.mj_collision(model,data);geometry=[]
   for co in data.contact[:data.ncon]:
    if base.context.floor not in [co.geom1,co.geom2]:continue
    other=int(co.geom2 if co.geom1==base.context.floor else co.geom1);body=model.body(int(model.geom_bodyid[other])).name;geometry.append(dict(geom_id=other,body=body,distance_m=float(co.dist)))
   # The old audit summary's landing window is postforward-vertical-triggered; recomputeexactly.
   oldmask=p['postforward_ground_vertical_N'][mask]<=eps;oldwindow=window(oldmask,post,p,mask);assert abs(oldwindow['integration_allbody_peak_vertical_N']-s['segments']['landing_100ms']['integration_contacts']['peak_ground_vertical_N'])<1e-9
   footwindow=window(fv,post,p,mask);bilwindow=window(~bil,post,p,mask)
   for compact_window in (footwindow,bilwindow):compact_window.pop('events_elapsed_s',None)
   # Afterthelastzero-foot sample, all remainingpost samples showpositivefootload.
   lastzero=np.flatnonzero(fv);lastreturn=None if not len(lastzero)or lastzero[-1]==len(post)-1 else float(post[lastzero[-1]+1]);late=None
   if lastreturn is not None:
    i=int(lastzero[-1]+1);indices=np.flatnonzero((post>=post[i]-1e-10)&(post<=post[i]+.1+1e-10));ip=indices[np.argmax(p['integration_ground_vertical_N'][mask][indices])];late=dict(event_elapsed_s=lastreturn,window_end_s=lastreturn+.1,integration_allbody_peak_vertical_N=float(p['integration_ground_vertical_N'][mask][ip]),peak_pre_elapsed_s=float(pre[ip]),peak_post_elapsed_s=float(post[ip]),left_right_weight_at_return=loads[i,:2].tolist(),integration_sum_contact_norm_peak_N=float(p['integration_ground_sum_norm_N'][mask][indices].max()))
   result['controllers'][label]=dict(source_physical_trace=str((directory/'physical_trace.npz').relative_to(r)),source_physical_sha256=sha(directory/'physical_trace.npz'),post_measurement_source=('Historical 5/5 episode 1, after exact state/action/time alignment'if label.startswith('historical')else'Current physical-trace 1 kHz measurement array'),global_integration_peak=dict(vertical_N=float(p['integration_ground_vertical_N'][mask][peak]),bodyweights=float(p['integration_ground_vertical_N'][mask][peak]/W),interval_start_elapsed_s=float(pre[peak]),interval_end_elapsed_s=float(post[peak]),contemporaneous_postforward_left_right_other_weights=loads[peak].tolist(),other_weight_definition='Sum of non-foot ground-force vector norms / W, not vertical force.',prestate_geometric_floor_contacts=geometry,perbody_original_integration_force='Original per-body integration forces were not preserved; aggregate force and independent geometry do not provide a left/right/body force decomposition.'),zero_load=conditions,vertical_vs_norm_zero_masks_identical=dict(integration=bool(np.array_equal(ig,p['integration_ground_vertical_N'][mask]<=eps)),postforward=bool(np.array_equal(pg,oldmask))),existing_summary_postforward_triggered_window_recomputed=oldwindow,integration_zero_load_return_window=window(ig,post,p,mask),postforward_zero_foot_load_return_window=footwindow,bilateral_005W_support_return_window=bilwindow,last_positive_foot_load_return_before_continuous_loading=late)
 finally:base.close()
result['conclusions']=['Global peak denotes all-body ground vertical load. Foot-landing analysis requires an explicit event and window; peak magnitude alone does not define a landing.','Previously published airborne/landing labels are force-based. Short zero-load intervals alone do not prove complete geometric flight.','Historical episode 1 preserves 1 kHz foot measurements. Exact post-observation, action, reward, joint-state, physical-time and floor-depth checks support aligned use with the diagnostic integration-force arrays.','Original per-body integration forces were not saved. Post-forward foot/non-foot measurements and independent contact geometry are not substitutes for that decomposition.']
result['analysis_recipe_sha256']=sha(__file__)
write=run/'contact_transition_comparison.json';write.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:{'global':v['global_integration_peak'],'zeroload_counts':{a:{b:c for b,c in d.items()if b!='intervals'}for a,d in v['zero_load'].items()},'existing_window':v['existing_summary_postforward_triggered_window_recomputed'],'lastfootreturn':v['last_positive_foot_load_return_before_continuous_loading']}for k,v in result['controllers'].items()},indent=2));print('OUTPUT',write)
