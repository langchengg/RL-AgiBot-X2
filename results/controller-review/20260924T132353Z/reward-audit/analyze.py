"""Recompute real-trajectory reward audit. No environment execution or training.

PYTHONPATH=src/x2_recovery .venv/bin/python <this-file> [--extract --repo ROOT --output NEW]
Default is read-only and uses the published compact extraction; --output must be new.
Ignored historical run folders
are unnecessary. --extract additionally verifies the original local log sources.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from x2_recovery.constraint_audit import discount_rewards

GAMMA = .999
POSITIVE = ('pose_guide', 'head_track', 'balance', 'standing')
CANDIDATES = {'reference-balance-v1': 1., 'reference-balance-density10-v1': .1,
              'reference-balance-density05-v1': .05}
BASE = 'runs/recovery_discovery/20260922T072024Z/'
PUBLISHED = 'results/evaluation/ppo-reference-residual-20260922T212457Z/'
SOURCES = {
    'published_success': (PUBLISHED+'trajectory.jsonl.gz', PUBLISHED+'inputs/training-run', 1),
    'zero_residual_success': (BASE+'success-small4k-network-ablation/zero_residual/trajectory.jsonl', BASE+'ppo-success-reference-small-4k', 1),
    'continued_policy_safety_failure': (BASE+'ppo-success-reference-low-noise-16k/development-222621/trajectory.jsonl', BASE+'ppo-success-reference-low-noise-16k', None),
    'broad_residual_timeout': (BASE+'ppo-success-reference-broad-4k/development-222600/trajectory.jsonl', BASE+'ppo-success-reference-broad-4k', None),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def extract(repo, output):
    arrays, metadata = {}, {}
    selected = json.loads((repo/PUBLISHED/'inputs/training-run/resolved_config.json').read_text())
    selected_controller = selected['controller']
    for name, (relative, training, episode) in SOURCES.items():
        source, inputs = repo/relative, repo/training
        opener = gzip.open if source.suffix == '.gz' else open
        with opener(source, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        transitions = [r for r in rows if r['kind'] == 'transition' and
                       (episode is None or r.get('episode') == episode)]
        reset = next(r for r in rows if r['kind'] == 'reset' and
                     (episode is None or r.get('episode') == episode))
        terms = sorted(transitions[0]['info']['reward_terms'])
        arrays[name+'__rewards'] = np.array([r['reward'] for r in transitions])
        arrays[name+'__terms'] = np.array([[r['info']['reward_terms'][k] for k in terms] for r in transitions])
        arrays[name+'__time_s'] = np.array([r['info']['elapsed_sim_s'] for r in transitions])
        arrays[name+'__physics_steps'] = np.array([r['info']['physics_steps_executed'] for r in transitions])
        config = json.loads((inputs/'resolved_config.json').read_text())
        manifest = json.loads((inputs/'manifest.json').read_text())
        assert config['controller']['reward_version'] == 'reference-balance-v1'
        assert config['training']['gamma'] == GAMMA
        assert config['training']['env'] == selected['training']['env']
        controller_diff = {k: {'selected': selected_controller.get(k), 'trajectory': v}
                           for k,v in config['controller'].items() if v != selected_controller.get(k)}
        last = transitions[-1]
        evidence = reset.get('reset_evidence', reset.get('evidence'))
        arrays[name+'__reset_qpos'] = np.array(evidence['final_qpos'])
        arrays[name+'__reset_qvel'] = np.array(evidence['final_qvel'])
        metadata[name] = dict(source=relative, source_sha256=sha(source), terms=terms,
            training_run=training, configuration_sha256=sha(inputs/'resolved_config.json'),
            checkpoint_sha256=sha(inputs/'policy_final.zip'),
            manifest_sha256=sha(inputs/'manifest.json'), reward_version='reference-balance-v1',
            gamma=GAMMA, reward_preprocessing=manifest['normalization'],
            controller_differences_from_selected=controller_diff, seed=reset.get('seed'),
            reset_seed=reset.get('reset_seed', reset.get('info', {}).get('reset_seed')),
            reset_perturb_rad=config['training']['env']['reset_perturb_rad'],
            policy_mode='zero_residual' if name.startswith('zero') else 'deterministic',
            success=last['info']['is_success'], terminated=last['terminated'], truncated=last['truncated'],
            termination_reason=last['info']['termination_reason'] or last['info']['truncation_reason'],
            transitions=len(transitions), simulated_seconds=last['info']['elapsed_sim_s'],
            last_transition_physics_steps=last['info']['physics_steps_executed'],
            terminal_observation_saved='observation' in last.get('post_step', {}),
            exact_historical_bootstrap_available=False)
    diagnostics = repo/PUBLISHED/'inputs/training-run/episode_diagnostics.csv'
    row = next(r for r in csv.DictReader(diagnostics.open()) if r['episode_id']=='6')
    manifest_path = repo/PUBLISHED/'inputs/training-run/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rawterms = json.loads(row['reward_terms'])
    episode = dict(row, reward_terms=rawterms, source=str(diagnostics.relative_to(repo)),
                   source_sha256=sha(diagnostics), manifest_sha256=sha(manifest_path),
                   exact_reward_sequence_available=False,
                   policy_adam_steps_during_episode=[40,52,78,91,108],
                   policy_version_evidence='Two workers; end global transition 3700 and length 975 imply global start 1750; completed rollout boundaries in manifest.training.updates.',
                   absent_data='Only completed episode aggregates and rollout min/max/shapes were retained; first-success-only trace was not emitted for this failure. Checkpoints are not a reward sequence.')
    np.savez_compressed(output/'real_reward_sequences.npz', **arrays)
    dump(output/'sources.json', dict(trajectories=metadata, historical_training_episode=episode,
         extraction='Exact float values copied from real logs, not simulated or reconstructed rewards.',
         array_sha256=sha(output/'real_reward_sequences.npz')))


def analyze(inputs, output=None):
    source = json.loads((inputs/'sources.json').read_text())
    assert sha(inputs/'real_reward_sequences.npz') == source['array_sha256']
    data = np.load(inputs/'real_reward_sequences.npz', allow_pickle=False)
    rows, detailed = [], {}
    for name, meta in source['trajectories'].items():
        original = {k: data[name+'__terms'][:,i] for i,k in enumerate(meta['terms'])}
        for version, scale in CANDIDATES.items():
            terms = {k: v*(scale if k in POSITIVE else 1.) for k,v in original.items()}
            reward = sum(terms.values())
            if scale == 1.:
                np.testing.assert_allclose(reward, data[name+'__rewards'], rtol=1e-12, atol=1e-12)
            result = discount_rewards(reward, terms, GAMMA,
                                      terminated=meta['terminated'], truncated=meta['truncated'])
            key = name+'/'+version
            detailed[key] = result
            rows.append(dict(trajectory=name, reward_version=version, success=meta['success'],
                transitions=meta['transitions'], duration_s=meta['simulated_seconds'],
                raw_return=result['raw_return'], discounted_prefix=result['observed_discounted_prefix'],
                critic_tail='', exact_historical_bootstrap='',
                scope='real trajectory offline rescore' if scale != 1. else 'original real rewards'))
    starts = {name: {key: float(np.max(abs(data[name+'__reset_'+key]-data['published_success__reset_'+key])))
                     for key in ('qpos','qvel')} for name in source['trajectories']}
    assert all(error == 0. for values in starts.values() for error in values.values())
    episode = source['historical_training_episode']
    signed = episode['reward_terms']
    positive = sum(signed[k] for k in POSITIVE)
    negative = sum(v for k,v in signed.items() if k not in POSITIVE and k not in ('success','safety'))
    terminal = signed['safety']
    assert positive >= 0 and negative <= 0 and terminal == -2 and float(signed['success']) == 0
    assert abs(positive+negative+terminal-float(episode['return'])) < 1e-9
    steps = int(episode['length']); final_weight = GAMMA**(steps-1)
    bounds = {}
    theoretical = {}
    for version, scale in CANDIDATES.items():
        lower = final_weight*(scale*positive+terminal)+negative
        upper = scale*positive+final_weight*(terminal+negative)
        success_g = detailed['published_success/'+version]['observed_discounted_prefix']
        bounds[version] = dict(lower_bound=lower, upper_bound=upper, exact_discounted_return=None,
            exceeds_selected_success_even_at_lower_bound=lower>success_g,
            below_selected_success_even_at_upper_bound=upper<success_g)
        density = 14.*scale  # pose <=1, head <=2, balance <=6, standing <=5.
        step_upper = .02*density
        horizon_sum = float(np.sum(GAMMA**np.arange(1000)))
        max_failure_prefix = step_upper*horizon_sum
        # These are mathematical sequences, not measured robot trajectories.
        late_success_upper = max_failure_prefix+50.*GAMMA**999
        inserted_hold_advantage = step_upper - (1.-GAMMA)*50.
        theoretical[version] = dict(positive_density_upper_per_second=density,
            positive_step_upper=step_upper, max_20s_nonterminal_discounted_positive=max_failure_prefix,
            max_20s_success_discounted_return_ignoring_costs=late_success_upper,
            measured_fast_success_discounted_return=success_g,
            max_20s_failure_below_measured_fast_success=max_failure_prefix<success_g,
            max_20s_success_below_measured_fast_success=late_success_upper<success_g,
            immediate_before_success_one_step_delay_upper_advantage=inserted_hold_advantage,
            immediate_safety_failure_reward=-2., added_time_cost=0.,
            cycle_bound='All repeated posture/hold rewards are included in the 20s positive-density bound; no one-time resettable stage bonus is introduced.',
            exploration_vs_death='No new time penalty is introduced. This does not prove every exploration trajectory outranks immediate failure: physical/target costs and safety termination still matter.',
            scope='Theoretical reward bound/test sequences, not real robot trajectories or learning guarantees.')
    fieldnames=list(rows[0])
    if output is not None:
        with (output/'reward_ranking.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    summary=dict(status='COMPLETE', gamma_per_policy_transition=GAMMA,
        formula='G=sum(gamma**t*r_t), t=0..T-1; actual dt weights original density rewards, but a shortened final transition still receives one gamma exponent.',
        discounted_terms=detailed, historical_episode_bounds=bounds,
        scope='Offline audit and rescoring only; pilot_result.json separately records new training and frozen five evaluation.',
        historical_episode_standing_reward=dict(raw_component=signed['standing'],control_boundary_qualified_duration_equivalent_s=signed['standing']/5.,maximum_continuous_physics_step_hold_s=float(episode['max_stable_seconds']),meaning='The duration equivalent weights each post-control-boundary qualification by its transition dt. It is not a reconstructed 1kHz standing trace and is not continuous hold time.'),
        historical_episode_bound_proof=dict(positive_nonterminal_sum=positive, negative_nonterminal_sum=negative,
            terminal_safety=terminal, last_weight=final_weight, T=steps,
            lower='All positive samples have weight >=gamma^(T-1); all nonterminal negative samples have weight <=1; safety is at T-1.',
            upper='All positive samples have weight <=1; all negative samples have weight >=gamma^(T-1).',
            exact_sequence_missing=True, bound_is_not_reconstructed_discounted_return=True),
        theoretical=theoretical, reset_handoff_max_abs_difference=starts,
        candidate_definition='Only pose_guide/head_track/balance/standing are scaled by alpha (.1 or .05); all costs, success=50, safety=-2, termination, truncation, physics, and controller are unchanged. These are new objectives, not potential-based shaping.',
        conclusion='The historical failure provably outranks the deterministic selected success under the same reward/gamma, even though its exact G is unavailable. This confirms a task/reward ranking mismatch for those trajectories, not exploitation by the final actor or a policy-distribution claim. Reduced-density offline objectives remove this ordering in the available bounds; training/control benefit remains unproven.',
        unavailable=['Exact rewards and discounted return for historical episode 6.',
            'Historical rollout bootstrap values/GAE for these complete episodes.',
            'Terminal observation for broad-residual timeout, so neither current nor historical critic tail is reported.',
            'A complete real long-near-standing failed episode sequence matching episode 6; its aggregates are retained separately.'],
        distinctions='Environment rewards and Monitor return are raw. SB3 may add gamma*V(terminal_observation) to a truncated transition in the rollout buffer after callback logging. Rollout GAE/returns use critic estimates and are distinct from the observed episodic discounted prefix. No post-terminal rewards are added here.',
        sampled_exploration_vs_immediate_safety_failure={version:{name:detailed[name+'/'+version]['observed_discounted_prefix'] > -2.
            for name in ('continued_policy_safety_failure','broad_residual_timeout')} for version in CANDIDATES},
        training_performed=False, proposed_new_objective_not_adopted=True)
    if output is not None: dump(output/'summary.json',summary)
    print(json.dumps(dict(rankings=rows, historical_bounds=bounds),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--output',type=Path,help='New output directory; absent means read-only stdout.')
    args=parser.parse_args();inputs=Path(__file__).resolve().parent
    output=None if args.output is None else args.output.resolve()
    if output is not None:output.mkdir(parents=True,exist_ok=False)
    if args.extract:
        if output is None:parser.error('--extract requires a new --output directory')
        extract(args.repo.resolve(),output);inputs=output
    analyze(inputs,output)


if __name__=='__main__': main()
