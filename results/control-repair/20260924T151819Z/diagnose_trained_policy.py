"""Strictly load one trained bundle and capture one unchanged deterministic episode.

Run using the matching installed source/environment. No training, migration,
control patch, threshold change, implicit install, or overwrite is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from x2_recovery.constraint_audit import discount_rewards, run_diagnostic
from x2_recovery.evaluate import prepare, verify_policy, policy_hash
from x2_recovery.train import json_value, source_hashes, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(trace, gamma):
    terms = {str(k): trace['reward_terms'][:, i]
             for i, k in enumerate(trace['reward_term_names'])}
    np.testing.assert_allclose(sum(terms.values()), trace['reward'], rtol=1e-12, atol=1e-12)
    result = {}
    for version, scale in [('reference-balance-density10-v1', 1.), ('reference-balance-v1', 10.)]:
        reweighted = {k: v*(scale if k in ('pose_guide', 'head_track', 'balance', 'standing') else 1.)
                      for k, v in terms.items()}
        result[version] = discount_rewards(sum(reweighted.values()), reweighted, gamma,
            terminated=bool(trace['terminated'][-1]), truncated=bool(trace['truncated'][-1]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True, type=Path)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--reference-directory', required=True, type=Path)
    parser.add_argument('--comparison-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    inputs, output, reference = args.training_run.resolve(), args.output.resolve(), args.reference_directory.resolve()
    if output.exists() or output.is_relative_to(inputs) or inputs.is_relative_to(output):
        raise ValueError('Output must be new and disjoint from the complete training inputs')
    saved_reference = json.loads((reference/'summary.json').read_text())
    reference_control = json.loads((reference/'control_config.json').read_text())
    reference_trace = np.load(reference/'control_trace.npz', allow_pickle=False)
    sources_before = source_hashes()
    env, policy, original, prepared = prepare(inputs,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    input_hashes = {str(path): sha(path) for path in prepared['input_sources'].values()}
    initial_policy_hash = policy_hash(original)
    try:
        saved = prepared['saved']
        if saved['controller'] != reference_control or saved['controller']['reward_version'] != 'reference-balance-density10-v1':
            raise ValueError('Reference controller/reward differs from the trained controller')
        if args.seed != saved_reference['seed']:
            raise ValueError('This comparison requires the same declared reset seed')
        gamma = prepared['config'].gamma
        identity = dict(controller='targets-v6 reference plus analytic feedback plus trained PPO residual',
            training=False, deterministic=True, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            strict_identity=prepared['identity'], strict_action_probe=prepared['consistency'],
            strict_input_hashes=prepared['hashes'], policy_state_sha256_before=initial_policy_hash,
            source_hashes=sources_before, reference_summary_sha256=sha(reference/'summary.json'),
            recipe_sha256=sha(__file__), command=[sys.executable, *sys.argv])
        summary = run_diagnostic(env, policy, seed=args.seed, output=output,
            zero_residual=False, gamma=gamma, source_identity=identity, trace_level='full')
        final_policy_hash = verify_policy(policy, original)
    finally:
        verify_policy(policy, original)
        env.close()
    trace = np.load(output/'control_trace.npz', allow_pickle=False)
    reset_difference = {key: float(np.max(abs(trace['reset_'+key]-reference_trace['reset_'+key])))
                        for key in ('observation', 'qpos', 'qvel')}
    if any(value > 1e-10 for value in reset_difference.values()):
        raise ValueError('Actual reset comparison failed: '+str(reset_difference))
    inputs_unchanged = all(sha(path) == digest for path, digest in input_hashes.items())
    if not inputs_unchanged or source_hashes() != sources_before:
        raise ValueError('Frozen inputs or running core sources changed during diagnostic')
    ranking = dict(trained_policy=score(trace, gamma), zero_PPO_reference=score(reference_trace, gamma),
        gamma_per_transition=gamma, last_trained_transition_physics_steps=int(trace['physics_steps'][-1]),
        same_reset_max_abs=reset_difference,
        interpretation='Same new reference, built-in feedback, physical settings and reward. Original reward is an offline re-score. No critic tail or historical bootstrap fabricated. One deterministic pair does not establish residual benefit.')
    comparison = json.loads(args.comparison_manifest.read_text())['comparison']
    def metrics(s):
        recovery = s['segments']['recovery']
        contacts = recovery['integration_contacts']
        return dict(peak_ground_vertical_N=contacts['peak_ground_vertical_N'],
            peak_ground_sum_norm_N=contacts['peak_ground_sum_norm_N'],
            floor_penetration_m=contacts['floor']['depth_m'],
            self_penetration_m=contacts['self']['depth_m'],
            max_joint_speed_rad_s=recovery['max_joint_speed_rad_s'])
    joint_metrics = metrics(summary)
    impact = {key: joint_metrics[key] <= value+1e-10 for key, value in comparison.items()}
    transitions = int(np.ceil(prepared['config'].env.episode_timeout_s /
                              prepared['saved']['environment']['base']['control_dt']))
    step_dt = prepared['saved']['environment']['base']['control_dt']
    late_bound = step_dt*1.4*sum(gamma**np.arange(transitions))+50*gamma**(transitions-1)
    ranking.update(density10_late_success_upper_ignoring_costs=float(late_bound),
        trained_observed_G_above_late_upper=bool(summary['success'] and
            ranking['trained_policy']['reference-balance-density10-v1']['observed_discounted_prefix'] > late_bound),
        theoretical_bound_is_not_success_criterion=True)
    summary.update(controller='reference_motion_analytic_feedback_and_trained_PPO_residual',
        policy_state_sha256_before=initial_policy_hash, policy_state_sha256_after=final_policy_hash,
        policy_state_unchanged=initial_policy_hash == final_policy_hash, input_bytes_unchanged=inputs_unchanged,
        original_reset_comparison_max_abs=reset_difference, impact_checks_against_original_mixed=impact,
        impact_pass=all(impact.values()),
        admissible_recovery_with_impact=bool(summary['success'] and summary['constraint_envelope_pass'] and all(impact.values())),
        reward_comparison=ranking, reference_summary=dict(success=saved_reference['success'],
            duration_s=saved_reference['sim_duration_s'], constraints=saved_reference['constraint_envelope_pass'],
            metrics=metrics(saved_reference)), training_executed_by_diagnostic=False)
    write_json(output/'reward_comparison.json', ranking)
    summary['artifacts_sha256'] = {path.name: sha(path) for path in output.iterdir()
                                  if path.is_file() and path.name != 'summary.json'}
    write_json(output/'summary.json', summary)
    print(json.dumps(json_value(dict(output=str(output), success=summary['success'],
        duration_s=summary['sim_duration_s'], constraint_envelope=summary['constraint_envelope_pass'],
        impact=all(impact.values()), policy_unchanged=summary['policy_state_unchanged'],
        raw_reward=summary['reward']['raw_return'], discounted_reward=summary['reward']['observed_discounted_prefix'],
        reference_duration_s=saved_reference['sim_duration_s'], reset_difference=reset_difference)), indent=2))


if __name__ == '__main__':
    main()
