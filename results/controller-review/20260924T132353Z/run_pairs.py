#!/usr/bin/env python3
"""Execute the preregistered review protocol; input loading stays in evaluate.prepare.

Requires the repository Python environment and x2_recovery on its import path.
This script never installs dependencies, edits frozen inputs, trains, or restores states.
"""
import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from x2_recovery.constraint_audit import json_value, run_diagnostic, write_json
from x2_recovery.evaluate import prepare, physical_env, verify_policy
from x2_recovery.model import require
from x2_recovery.reset import integration_state


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(summary):
    r = summary['segments']['recovery']
    p = r['postforward_contacts']
    return dict(success=summary['success'], recovery_sim_s=summary['sim_duration_s'],
        max_continuous_hold_s=summary['maximum_stable_hold_s'],
        joint_excess_max_rad=r['max_joint_excess_rad'], worst_joint=r['worst_joint'],
        joint_excess_duration_s=r['any_joint_above_standing_tolerance_s'],
        max_velocity_rad_s=r['max_joint_speed_rad_s'], torque_ratio=r['max_actuation_limit_ratio'],
        floor_penetration_m=p['floor']['depth_m'], self_penetration_m=p['self']['depth_m'],
        floor_contact_load_N=p['peak_ground_vertical_N'],
        integration_floor_contact_load_N=r['integration_contacts']['peak_ground_vertical_N'],
        floor_contact_load_bodyweights=p['peak_ground_vertical_bodyweights'],
        constraint_envelope_pass=summary['constraint_envelope_pass'],
        admissible_recovery=summary['admissible_recovery'],
        stable_margins=json.dumps(summary['standing_window_minimum_margins'], sort_keys=True),
        standing_sway=json.dumps(json_value(summary.get('standing_window_sway')), sort_keys=True),
        termination_reason=summary['final_info'].get('termination_reason'),
        truncation_reason=summary['final_info'].get('truncation_reason'))


def execute(args):
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest['paired_protocol']
    require(protocol['planned_pairs'] == 20 and protocol['max_reset_candidates'] == 40,
            'Unexpected preregistered pair count')
    require(protocol['allowed_base_config_difference'] == ['reset_perturb_rad'], 'Unexpected override allowlist')
    output = Path(args.output).expanduser().resolve()
    training = Path(args.training_run).expanduser().resolve()
    require(not output.exists(), 'Output already exists')
    require(output != training and training not in output.parents, 'Cannot write inside controller input')
    env, policy, original, prepared = prepare(training,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    base = physical_env(env)
    started = time.monotonic()
    try:
        original_config = asdict(base.config)
        base.config = replace(base.config, reset_perturb_rad=float(protocol['perturb_rad']))
        deployment_config = asdict(base.config)
        differences = {k: {'published': original_config[k], 'deployment': v}
                       for k, v in deployment_config.items() if original_config[k] != v}
        require(set(differences) == {'reset_perturb_rad'}, 'Unapproved deployment config difference')
        require(env.control_config.mode == 'reference_residual' and env.action_space.shape == (17,),
                'Unexpected controller interface')
        output.mkdir(parents=True, exist_ok=False)
        import x2_recovery.constraint_audit as audit
        source_identity = dict(input_sha256=prepared['hashes'], deployment_difference=differences,
            input_interface=prepared['consistency'], controller_config_sha256=hashlib.sha256(
                json.dumps(asdict(env.control_config), sort_keys=True).encode()).hexdigest(),
            audit_source_sha256=sha256(audit.__file__), runner_sha256=sha256(__file__),
            protocol_manifest_sha256=sha256(manifest_path), deterministic=True, device='cpu')
        write_json(output/'protocol.json', dict(protocol=protocol, source_identity=source_identity,
            full_deployment_base_config=deployment_config,
            statement='Original checkpoint and controller unchanged; only validated reset perturbation distribution differs.'))
        selected, attempts = [], []
        # Complete ALL selection before observing either controller's results.
        for seed in protocol['candidate_seeds'][:protocol['max_reset_candidates']]:
            record = dict(seed=seed, accepted=False)
            try:
                observation, _ = env.reset(seed=seed)
                require(base.reset_evidence['status'] == 'PASS', 'Reset evidence not PASS')
                state = dict(observation=observation.copy(), qpos=base.data.qpos.copy(),
                    qvel=base.data.qvel.copy(), integration_state=integration_state(base.model, base.data))
                selected.append((seed, state))
                record.update(accepted=True, reset_seed=base.reset_seed,
                    reset_handoff_absolute_s=float(base.data.time),
                    window_maxima=base.reset_evidence['window_maxima'],
                    final_reset_metrics=base.reset_evidence['final'])
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc),
                    evidence=getattr(exc, 'evidence', base.last_error))
            attempts.append(record)
            print(f"preflight {seed}: {'accepted' if record['accepted'] else 'REJECTED'} ({len(selected)}/20)", flush=True)
            if len(selected) == protocol['planned_pairs']:
                break
        write_json(output/'reset_preflight.json', dict(attempts=attempts,
            selected_seeds=[s for s, _ in selected], completed_before_any_controller=True))
        states = {key: np.array([state[key] for _, state in selected])
                  for key in ('observation', 'qpos', 'qvel', 'integration_state')}
        states['seed'] = np.array([seed for seed, _ in selected], dtype=np.int64)
        np.savez_compressed(output/'reset_handoffs.npz', **states)
        require(len(selected) == protocol['planned_pairs'], 'Fewer than 20 legal preregistered resets')
        minimum_difference = min(float(max(abs(a['qpos']-b['qpos']).max(), abs(a['qvel']-b['qvel']).max()))
            for i, (_, a) in enumerate(selected) for _, b in selected[i+1:])
        require(minimum_difference > protocol['physical_pair_atol'], 'Initial physical states not distinct')
        write_json(output/'selection_frozen.json', dict(selected_seeds=states['seed'],
            minimum_pairwise_handoff_qpos_qvel_max_abs_difference=minimum_difference,
            reset_preflight_sha256=sha256(output/'reset_preflight.json'),
            reset_handoffs_sha256=sha256(output/'reset_handoffs.npz'),
            rule='No replacement seeds or reset retries based on controller outcomes.'))
        rows, detailed, pairs = [], [], []
        for pair_index, (seed, reference_state) in enumerate(selected):
            outcomes = {}
            for label, zero in (('A_reference', True), ('B_residual', False)):
                destination = output/f'pair-{pair_index+1:02d}-{seed}'/label
                record = dict(pair=pair_index+1, seed=seed, controller=label, status='COMPLETE')
                try:
                    summary = run_diagnostic(env, policy, seed=seed, output=destination,
                        zero_residual=zero, gamma=prepared['config'].gamma, source_identity=source_identity,
                        trace_level='compact', expected_reset=reference_state,
                        reset_atol=protocol['physical_pair_atol'])
                    record.update(metrics(summary))
                    outcomes[label] = summary
                except Exception as exc:
                    record.update(status='INVALID_OR_BLOCKED', error_type=type(exc).__name__, error=str(exc))
                rows.append(record)
                detailed.append(record)
                write_json(output/'paired_progress.json', detailed)
                print(f"pair {pair_index+1}/20 {seed} {label}: {record['status']} success={record.get('success')} time={record.get('recovery_sim_s')}", flush=True)
            pair = dict(pair=pair_index+1, seed=seed, valid=len(outcomes) == 2)
            if pair['valid']:
                a, b = outcomes['A_reference'], outcomes['B_residual']
                with np.load(output/f'pair-{pair_index+1:02d}-{seed}/A_reference/control_trace.npz') as aa, \
                     np.load(output/f'pair-{pair_index+1:02d}-{seed}/B_residual/control_trace.npz') as bb:
                    pair['A_B_handoff_max_abs'] = {k: float(abs(aa[k]-bb[k]).max())
                        for k in ('reset_qpos', 'reset_qvel', 'reset_observation')}
                pair['valid'] = all(v <= protocol['physical_pair_atol'] for v in pair['A_B_handoff_max_abs'].values())
                pair.update(A_success=a['success'], B_success=b['success'],
                    success_outcome='tie' if a['success'] == b['success'] else ('B_win' if b['success'] else 'A_win'),
                    B_minus_A_success_time_s=(b['sim_duration_s']-a['sim_duration_s']) if a['success'] and b['success'] else None,
                    B_minus_A_joint_excess_rad=b['segments']['recovery']['max_joint_excess_rad']-a['segments']['recovery']['max_joint_excess_rad'],
                    B_minus_A_peak_ground_N=b['segments']['recovery']['postforward_contacts']['peak_ground_vertical_N']-a['segments']['recovery']['postforward_contacts']['peak_ground_vertical_N'])
            pairs.append(pair)
        columns = sorted({k for row in rows for k in row})
        with (output/'paired_results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
        valid = [p for p in pairs if p['valid']]
        success_times = [p['B_minus_A_success_time_s'] for p in valid if p['B_minus_A_success_time_s'] is not None]
        by_controller = {}
        for label in ('A_reference', 'B_residual'):
            rr = [r for r in rows if r['controller'] == label and r['status'] == 'COMPLETE']
            by_controller[label] = dict(completed=len(rr), successes=sum(r['success'] for r in rr),
                constraint_envelope_passes=sum(r['constraint_envelope_pass'] for r in rr),
                admissible_recoveries=sum(r['admissible_recovery'] for r in rr),
                ranges={key: dict(min=min(r[key] for r in rr), max=max(r[key] for r in rr))
                    for key in ('joint_excess_max_rad', 'joint_excess_duration_s', 'max_velocity_rad_s',
                        'torque_ratio', 'floor_penetration_m', 'self_penetration_m', 'floor_contact_load_N')} if rr else {})
        verify_policy(policy, original)
        summary = dict(schema='paired-reset-review-v1', planned=20, valid_pairs=len(valid),
            invalid_or_blocked_pairs=20-len(valid), reset_candidates_attempted=len(attempts),
            reset_rejections=sum(not r['accepted'] for r in attempts),
            minimum_pairwise_handoff_difference=minimum_difference, pairs=pairs, controllers=by_controller,
            success_outcomes={k: sum(p['success_outcome'] == k for p in valid) for k in ('A_win', 'B_win', 'tie')},
            both_success_pairs=len(success_times), B_minus_A_success_time_s=None if not success_times else dict(
                min=min(success_times), max=max(success_times), mean=float(np.mean(success_times)), median=float(np.median(success_times))),
            wall_seconds=time.monotonic()-started, source_identity=source_identity,
            policy_weights_unchanged=True, claims='20 distinct physical initial states, 40 controller runs; paired development data. '
                'Success and the conservative full-process constraint envelope are separate. '
                'Success differences are descriptive; this batch does not establish broad robustness or a causal training benefit.')
        write_json(output/'summary.json', summary)
        print(json.dumps({k: summary[k] for k in ('planned', 'valid_pairs', 'success_outcomes', 'both_success_pairs', 'B_minus_A_success_time_s', 'wall_seconds')}, indent=2), flush=True)
        return 0 if len(valid) == 20 else 2
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(execute(parser.parse_args()))
