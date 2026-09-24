import json,time,sys
from pathlib import Path
import numpy as np
from x2_recovery.evaluate import prepare,physical_env,verify_policy
from x2_recovery.train import write_json
root=Path('/home/lang/RL-AgiBot-X2')
out=root/'results/controller-review/20260924T132353Z/unobserved'
out.mkdir(exist_ok=False)
start=time.monotonic()
env,model,original,prepared=prepare(root/'results/evaluation/ppo-reference-residual-20260922T212457Z/inputs/training-run',expected_checkpoint_sha256='11d8f3e203936d2bd49b3b6cfb62e91909c6a8c8825425f540dacc712bb54c64')
e=physical_env(env); arrays={k:[] for k in ['observations','actions','qpos','qvel','reward','time_s']}
try:
 obs,info=env.reset(seed=221030)
 arrays['observations'].append(obs.copy());arrays['qpos'].append(e.data.qpos.copy());arrays['qvel'].append(e.data.qvel.copy());arrays['time_s'].append(0.)
 while True:
  action,_=model.predict(obs,deterministic=True)
  obs,reward,term,trunc,info=env.step(action)
  arrays['observations'].append(obs.copy()); arrays['actions'].append(action.copy());arrays['qpos'].append(e.data.qpos.copy());arrays['qvel'].append(e.data.qvel.copy());arrays['reward'].append(reward);arrays['time_s'].append(info['elapsed_sim_s'])
  if term or trunc: break
 verify_policy(model,original)
 np.savez_compressed(out/'control_trace.npz',**{k:np.asarray(v) for k,v in arrays.items()})
 write_json(out/'summary.json',dict(label='unmodified published policy without new observer',seed=221030,success=info['is_success'],reason=info['termination_reason'] or info['truncation_reason'],sim_s=info['elapsed_sim_s'],transitions=len(arrays['actions']),raw_return=sum(arrays['reward']),wall_s=time.monotonic()-start,reload=prepared['consistency'],policy_unchanged=True))
 print((out/'summary.json').read_text(),flush=True)
finally: env.close()
