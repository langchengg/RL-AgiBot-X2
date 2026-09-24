"""Bounded reference-only changes; old actor transferred explicitly, never retrained.
Run using installed x2_recovery or the repository's src/x2_recovery PYTHONPATH.
All original validation runs before creating a new in-memory control configuration.
"""
import argparse
import csv
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import shutil

from x2_recovery.constraint_audit import run_diagnostic,write_json
from x2_recovery.env import ControlledRecoveryEnv
from x2_recovery.evaluate import prepare,verify_policy,physical_env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(), 'Refusing existing output'
    args.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(args.plan.read_text())
    rows=[]
    for candidate in plan['candidates']:
        for zero in (True,False):
            mode='zero_residual_reference' if zero else 'transferred_residual'
            out=args.output/candidate['name']/mode
            env,policy,original,loaded=prepare(args.training_run,expected_checkpoint_sha256=args.checkpoint_sha256)
            base=physical_env(env)
            mapping={r.joint_name:i for i,r in enumerate(base.loaded.mapping)}
            stages=[]
            for name,duration,angles in env.control_config.reference_stages:
                altered=dict(angles)
                for joint,margin in candidate.get('margins',{}).items():
                    if joint in altered:
                        i=mapping[joint]
                        altered[joint]=min(max(altered[joint],base.q_min[i]+margin),base.q_max[i]-margin)
                stages.append((name,candidate.get('duration_override',{}).get(name,duration),altered))
            config=replace(env.control_config,reference_stages=tuple(stages))
            revised=ControlledRecoveryEnv(base,config)
            identity=dict(source_checkpoint_sha256=args.checkpoint_sha256,
                original_input_hashes=loaded['hashes'],operation='original trained weights transferred to changed reference; no training',
                changed_fields=['controller.reference_stages'],candidate=candidate,
                unchanged='model, physics, reset, PD, residual mapping/scales/gate, reward, original success and safety settings',
                new_control=asdict(config))
            try:
                summary=run_diagnostic(revised,policy,seed=plan['seed'],output=out,zero_residual=zero,
                    source_identity=identity)
                verify_policy(policy,original)
                write_json(out/'control_config.json',asdict(config))
                x=summary['segments']['recovery'];c=x['integration_contacts']
                row=dict(candidate=candidate['name'],controller=mode,success=summary['success'],
                    sim_s=summary['sim_duration_s'],hold_s=summary['maximum_stable_hold_s'],
                    reason=summary['final_info']['termination_reason'] or summary['final_info']['truncation_reason'],
                    joint_excess_rad=x['max_joint_excess_rad'],worst_joint=x['worst_joint'],
                    excess_duration_s=x['any_joint_above_standing_tolerance_s'],speed_rad_s=x['max_joint_speed_rad_s'],
                    torque_ratio=x['max_actuation_limit_ratio'],floor_penetration_m=c['floor']['depth_m'],
                    self_penetration_m=c['self']['depth_m'],peak_ground_force_bodyweights=c['peak_ground_vertical_bodyweights'],
                    constraint_envelope_pass=summary['constraint_envelope_pass'],admissible=summary['admissible_recovery'],
                    config_sha256=hashlib.sha256((out/'control_config.json').read_bytes()).hexdigest())
                rows.append(row)
                with (args.output/'results.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
                print(json.dumps(row),flush=True)
            finally: revised.close()
    write_json(args.output/'summary.json',dict(label='bounded development candidates, not a replacement release',
        candidates=len(plan['candidates']),runs=len(rows),original_actor_unchanged=True,training=False,
        admissible_runs=sum(r['admissible'] for r in rows),rows=rows))

if __name__=='__main__':main()
