"""One open-loop, joint-target recovery attempt through the existing environment.

Stage names describe intent, not measured support or recovery. No ROS, direct
physics stepping, state placement, PD implementation or success logic lives here.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

from .env import EnvConfig, X2RecoveryEnv
from .model import COMMIT, REPOSITORY, SCENE, SOURCE_HASHES, require
from .reset import ResetSettings
from .success import MODEL_FINGERPRINT, MODEL_SIGNATURE, measure_standing


def candidate_stages():
    """(intent, duration seconds, named absolute radians), independent of timeout.

    Verified source axes: hip/knee/ankle/waist pitch and elbows are +Y on both
    sides; negative hip pitch flexes forward, positive knee bends back. Shoulder
    pitch is +Y in the slightly rolled shoulder frames. Shoulder roll is +X:
    outward is positive left, negative right. Forearm/wrist collision hulls can
    support the body; visual fingers alone cannot establish physical support.
    """
    tuck = {}
    brace = {}
    shift = {'waist_pitch_joint': .12}
    extend = {'waist_pitch_joint': 0.}
    for side, outward in (('left', 1), ('right', -1)):
        def named(values):
            return {f'{side}_{joint}_joint': value for joint, value in values.items()}
        tuck.update(named({'hip_pitch': -.65, 'knee': 1.15, 'ankle_pitch': -.45}))
        brace.update(named({'shoulder_pitch': .65, 'shoulder_roll': outward*.65,
                            'elbow': -.95}))
        shift.update(named({'hip_pitch': -.95, 'knee': 1.45, 'ankle_pitch': -.55,
                            'shoulder_pitch': .8, 'elbow': -.65}))
        extend.update(named({'hip_pitch': -.05, 'knee': .10, 'ankle_pitch': -.05,
                             'shoulder_pitch': -.15, 'shoulder_roll': outward*.35,
                             'elbow': -.30}))
    return (('handover', .4, {}), ('tuck', 3., tuck), ('brace', 2., brace),
            ('shift', 3., shift), ('extend', 4., extend))


@dataclass(frozen=True)
class Keyframes:
    times_s: np.ndarray
    targets_rad: np.ndarray
    phases: tuple
    adjustments: tuple = ()

    def __post_init__(self):
        times = np.array(self.times_s, dtype=float, copy=True)
        targets = np.array(self.targets_rad, dtype=float, copy=True)
        require(times.ndim == 1 and len(times) >= 2 and np.isfinite(times).all()
                and times[0] == 0 and np.all(np.diff(times) > 0),
                'Keyframe times must start at zero and strictly increase')
        require(targets.ndim == 2 and targets.shape[0] == len(times)
                and targets.shape[1] == 31 and np.isfinite(targets).all(),
                'Keyframe targets must be finite with shape (nodes, 31)')
        require(len(self.phases) == len(times)-1
                and all(isinstance(p, str) and p for p in self.phases), 'Invalid phase names')
        times.setflags(write=False)
        targets.setflags(write=False)
        object.__setattr__(self, 'times_s', times)
        object.__setattr__(self, 'targets_rad', targets)
        object.__setattr__(self, 'phases', tuple(self.phases))

    def configuration(self):
        return dict(times_s=self.times_s, targets_rad=self.targets_rad, phases=self.phases,
                    adjustments=self.adjustments,
                    segment_peak_reference_speed_rad_s=1.875*np.abs(np.diff(self.targets_rad, axis=0))
                    / np.diff(self.times_s)[:, None],
                    interpolation='quintic smoothstep; final target held indefinitely',
                    speed_scope='Reference only; not a physical joint speed limit')


def build_keyframes(q_initial, mapping, *, stages=None):
    """Copy this reset's state once; inherit unspecified joints in mapping order.

    Only the reference may be projected within the existing reset tolerance.
    Candidate endpoint clipping is explicit evidence, never a physical state edit.
    """
    names = [row.joint_name for row in mapping]
    require(len(names) == 31 and len(set(names)) == 31, 'Expected 31 unique mapped joints')
    indices = {name: i for i, name in enumerate(names)}
    limits = np.array([row.position_range for row in mapping], dtype=float)
    require(limits.shape == (31, 2) and np.isfinite(limits).all()
            and np.all(limits[:, 0] < limits[:, 1]), 'Invalid mapped limits')
    q = np.array(q_initial, dtype=float, copy=True)
    require(q.shape == (31,) and np.isfinite(q).all(), 'Invalid initial joint state')
    projected = np.clip(q, limits[:, 0], limits[:, 1])
    require(np.max(abs(projected-q)) <= ResetSettings().joint_epsilon_rad,
            'Initial state exceeds reset joint tolerance')
    changes = []

    def record_changes(before, after, kind, phase):
        for i in np.flatnonzero(before != after):
            changes.append(dict(kind=kind, phase=phase, joint=names[i],
                                requested_rad=float(before[i]), target_rad=float(after[i]),
                                delta_rad=float(after[i]-before[i])))

    record_changes(q, projected, 'initial_reference_projection', 'handover')
    targets, times, phases = [projected], [0.], []
    for name, duration, updates in candidate_stages() if stages is None else stages:
        require(np.isfinite(duration) and duration > 0, 'Stage duration must be finite and positive')
        desired = targets[-1].copy()
        for joint, angle in updates.items():
            require(joint in indices, 'Unknown joint: '+joint)
            require(np.isscalar(angle) and np.isreal(angle) and np.isfinite(angle),
                    'Nonfinite/nonreal target: '+joint)
            desired[indices[joint]] = angle
        legal = np.clip(desired, limits[:, 0], limits[:, 1])
        record_changes(desired, legal, 'candidate_target_clamp', name)
        targets.append(legal)
        times.append(times[-1]+duration)
        phases.append(name)
    return Keyframes(times, targets, tuple(phases), tuple(changes))


def scripted_targets(elapsed_sim_s, keyframes):
    """Pure reference sampling; exact nodes, next phase at boundaries, no outcome."""
    require(np.isfinite(elapsed_sim_s) and elapsed_sim_s >= 0, 'Invalid episode time')
    t, q = keyframes.times_s, keyframes.targets_rad
    if elapsed_sim_s >= t[-1]:
        return q[-1].copy(), dict(name='hold', progress=1., sequence_finished=True)
    i = int(np.searchsorted(t, elapsed_sim_s, side='right')-1)
    u = float(np.clip((elapsed_sim_s-t[i])/(t[i+1]-t[i]), 0., 1.))
    s = float(np.clip(10*u**3-15*u**4+6*u**5, 0., 1.))
    return (1-s)*q[i]+s*q[i+1], dict(name=keyframes.phases[i], progress=u, sequence_finished=False)


def _json_value(value, *, diagnostic=False):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), diagnostic=diagnostic)
    if isinstance(value, np.generic):
        return _json_value(value.item(), diagnostic=diagnostic)
    if isinstance(value, dict):
        return {k: _json_value(v, diagnostic=diagnostic) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v, diagnostic=diagnostic) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        if diagnostic:
            return {'invalid_numeric': repr(value)}
        raise ValueError('Nonfinite output value')
    return value


def _error(exc):
    return _json_value(dict(type=type(exc).__name__, message=str(exc),
                            evidence=getattr(exc, 'evidence', None)), diagnostic=True)


def _git_identity():
    # Resolve symlink installations back to their tested source checkout.
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), *args],
                                  check=True, capture_output=True, text=True, timeout=15).stdout.strip()
        status = git('status', '--porcelain', '--untracked-files=all')
        return dict(sha=git('rev-parse', 'HEAD'), dirty=bool(status), status=status)
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(sha=None, dirty=None, unavailable_reason=str(exc))


def _state(env):
    q, dq = env.loaded.read_state(env.data)
    return dict(q_rad=q.copy(), dq_rad_s=dq.copy(),
                base_position_world_m=env.data.qpos[:3].copy(),
                base_quaternion_wxyz=env.data.qpos[3:7].copy(),
                base_qvel=env.data.qvel[:6].copy(),
                ctrl_Nm=env.data.ctrl[env.context.ctrladr].copy(),
                qfrc_actuator_Nm=env.data.qfrc_actuator[env.context.vadr].copy(),
                measurement=measure_standing(env.context, env.data))


def _write_record(stream, record):
    stream.write(json.dumps(_json_value(record), allow_nan=False, separators=(',', ':'))+'\n')
    stream.flush()


def _write_summary(directory, summary, *, diagnostic=False):
    encoded = json.dumps(_json_value(summary, diagnostic=diagnostic), allow_nan=False, indent=2)+'\n'
    pending = directory/('summary.diagnostic.pending' if diagnostic else 'summary.pending')
    with pending.open('x', encoding='utf-8') as output:
        output.write(encoded)
    # The run directory is exclusively owned; publish only a complete JSON file.
    pending.rename(directory/'summary.json')


def run_episode(*, seed=60, timeout_s=20., output_dir=None, human=False):
    """Exactly one env/reset/episode. Errors and cancellation are incomplete runs.

    Output creation failures raise to the CLI before any environment is created.
    Subsequent failures retain the first error, partial trajectory and best-effort
    summary. No automatic retries, outcome inference, or wall-clock scheduling.
    """
    config = EnvConfig(episode_timeout_s=timeout_s)
    require(type(seed) is int and seed >= 0, 'Seed must be a nonnegative integer')
    directory = Path(output_dir) if output_dir is not None else Path('audit-output/scripted-baseline') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:10])
    directory.mkdir(parents=True, exist_ok=False)
    stream = None
    env = None
    summary = dict(controller='scripted_baseline', output_dir=str(directory.resolve()),
                   execution_completed=False, recovery_success=None, termination_reason=None,
                   truncation_reason=None, execution_error=None, secondary_errors=[], cancelled=False,
                   requested_seed=seed, requested_timeout_s=timeout_s, reset_seed=None, reset_evidence=None,
                   initial_sim_time_s=None, final_sim_time_s=None, elapsed_sim_s=None,
                   control_steps=0, physics_steps=0, cumulative_reward=0.,
                   git=_git_identity(), python_executable=sys.executable,
                   environment_closed=False, render_mode='human' if human else None,
                   sampling=dict(state='reset and post-step; q/dq at interval end',
                                 action='sampled at interval start, held until interval end',
                                 torque='last physical substep only, not interval mean',
                                 base='position world; quaternion wxyz body-to-world; raw qvel: world linear then body-local angular',
                                 support='left/right_weight: signed vertical N / robot weight N; other_weight: sum of non-foot force norms / robot weight N; support.force_world_N in N',
                                 statistics='height/stability: reset plus recorded control returns; control peaks/fractions: all substeps of recorded returns',
                                 final_time='last fully written trajectory record; an execution error may contain a later failed/unsaved state'))
    peak_control = {}
    heights, holds = [], []
    saturated_pairs = saturated_steps = 0.
    recorded_physics_steps = 0
    last_record = None
    attempted = None

    def fail(exc):
        error = _error(exc)
        if summary['execution_error'] is None:
            summary['execution_error'] = error
        else:
            summary['secondary_errors'].append(error)
        summary['execution_completed'] = False
        summary['recovery_success'] = None
        if isinstance(exc, KeyboardInterrupt):
            summary['cancelled'] = True

    try:
        # Exclusive creation checks writability before model loading/reset.
        stream = (directory/'trajectory.jsonl').open('x', encoding='utf-8')
        env = X2RecoveryEnv(config, render_mode='human' if human else None)
        summary.update(resolved_env_config=env.resolved_config(),
                       joint_names=[r.joint_name for r in env.loaded.mapping],
                       model=dict(repository=REPOSITORY, commit=COMMIT, scene=SCENE,
                                  asset_repo=str(env.loaded.asset_repo), source_hashes=SOURCE_HASHES,
                                  overrides=env.loaded.overrides, signature=MODEL_SIGNATURE,
                                  effective_fingerprint=MODEL_FINGERPRINT), robot_weight_N=env.context.weight_N)
        _, info = env.reset(seed=seed)
        summary.update(reset_seed=env.reset_seed, reset_evidence=env.reset_evidence,
                       initial_sim_time_s=float(env.data.time))
        initial = _state(env)
        record = dict(record_type='reset', control_step=0, sim_start_s=float(env.data.time),
                      sim_end_s=float(env.data.time), elapsed_start_s=0., elapsed_sim_s=0.,
                      phase=dict(name='reset', progress=0., sequence_finished=False),
                      target_rad=None, action=None, applied_target_rad=None, reward=0.,
                      terminated=False, truncated=False, **initial, info=info)
        _write_record(stream, record)
        last_record = record
        heights.append(info['state']['pelvis_height_m'])
        holds.append(info['stable_duration_s'])
        frames = build_keyframes(initial['q_rad'], env.loaded.mapping)
        summary['keyframes'] = frames.configuration()
        while True:
            start = float(env.data.time)
            elapsed = info['elapsed_sim_s']
            desired, phase = scripted_targets(elapsed, frames)
            action = env.action_for_targets(desired)
            _, applied_target = env.action_targets(action)
            error = float(np.max(abs(applied_target-desired)))
            require(error <= 2e-7, 'Action/target round trip exceeds 2e-7 rad')
            attempted = dict(sim_start_s=start, elapsed_start_s=elapsed, phase=phase,
                             target_rad=desired, action=action)
            _, reward, terminated, truncated, info = env.step(action)
            summary['control_steps'] += 1
            steps = info['physics_steps_executed']
            summary['physics_steps'] += steps
            summary['cumulative_reward'] += reward
            summary['last_successful_return'] = dict(sim_time_s=float(env.data.time), info=info)
            record = dict(record_type='transition', control_step=summary['control_steps'],
                          **attempted, sim_end_s=float(env.data.time),
                          elapsed_sim_s=info['elapsed_sim_s'], applied_target_rad=applied_target,
                          action_roundtrip_max_error_rad=error, reward=reward,
                          terminated=terminated, truncated=truncated, **_state(env), info=info)
            _write_record(stream, record)
            last_record = record
            recorded_physics_steps += steps
            heights.append(info['state']['pelvis_height_m'])
            holds.append(info['stable_duration_s'])
            saturated_pairs += info['torque_saturation_fraction']*steps*len(env.loaded.mapping)
            saturated_steps += info['control']['any_saturation_time_fraction']*steps
            for key in ('raw_torque_abs_max_Nm', 'applied_torque_abs_max_Nm', 'joint_rad_s',
                        'joint_limit_rad', 'floor_penetration_m', 'self_penetration_m'):
                peak_control[key] = max(peak_control.get(key, 0.), info['control'][key])
            if terminated or truncated:
                summary.update(execution_completed=True, recovery_success=bool(info['is_success']),
                               termination_reason=info['termination_reason'], truncation_reason=info['truncation_reason'])
                break
    except (Exception, KeyboardInterrupt) as exc:
        fail(exc)
        summary['attempted_transition'] = _json_value(attempted, diagnostic=True)
    finally:
        if env is not None:
            try:
                env.close()
                summary['environment_closed'] = True
            except (Exception, KeyboardInterrupt) as exc:
                fail(exc)
        if stream is not None:
            try:
                stream.close()
            except (Exception, KeyboardInterrupt) as exc:
                fail(exc)

    if last_record is not None:
        summary.update(final_sim_time_s=last_record['sim_end_s'],
                       elapsed_sim_s=last_record['elapsed_sim_s'],
                       last_recorded_control_step=last_record['control_step'],
                       final_info=last_record['info'])
    n = recorded_physics_steps
    summary['statistics'] = dict(recorded_physics_steps=n,
                                 control_sample_max_pelvis_height_m=max(heights) if heights else None,
                                 control_sample_max_stable_duration_s=max(holds) if holds else None,
                                 substep_peaks=peak_control,
                                 substep_joint_saturation_fraction=saturated_pairs/(n*31) if n else None,
                                 substep_any_saturation_fraction=saturated_steps/n if n else None)
    if not summary['execution_completed']:
        summary['interpretation'] = 'Incomplete execution; no valid recovery outcome. See first execution_error and last recorded state.'
    else:
        v = last_record['info']['state']
        summary['interpretation'] = (
            f"Environment ended with {summary['termination_reason'] or summary['truncation_reason']}; "
            f"final pelvis {v['pelvis_height_m']:.6f} m, tilt {v['tilt_deg']:.3f} deg, "
            f"left/right vertical support {v['left_weight']:.6f}/{v['right_weight']:.6f} body weights, "
            f"non-foot force norm sum {v['other_weight']:.6f} body weights. "
            'Stage names describe reference intent; they do not establish support or standing.')
    # Serialize before opening to avoid truncating a summary on invalid numbers.
    try:
        _write_summary(directory, summary)
    except (Exception, KeyboardInterrupt) as exc:
        fail(exc)
        print('Summary write failed: '+str(exc), file=sys.stderr)
        # A diagnostic-only fallback may contain explicit invalid_numeric tags.
        # Never overwrite a partial file or pretend evidence was saved.
        if not (directory/'summary.json').exists():
            try:
                _write_summary(directory, summary, diagnostic=True)
            except (Exception, KeyboardInterrupt) as secondary:
                fail(secondary)
                print('Diagnostic summary write failed: '+str(secondary), file=sys.stderr)
    return _json_value(summary, diagnostic=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=60)
    parser.add_argument('--timeout-s', type=float, default=EnvConfig().episode_timeout_s)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--human', action='store_true', help='Use the existing read-only GLFW viewer')
    args = parser.parse_args(argv)
    try:
        result = run_episode(seed=args.seed, timeout_s=args.timeout_s, output_dir=args.output_dir, human=args.human)
    except KeyboardInterrupt:
        print('Cancelled before episode output was available', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ('output_dir', 'execution_completed', 'recovery_success',
                                           'termination_reason', 'truncation_reason', 'execution_error',
                                           'elapsed_sim_s', 'control_steps', 'physics_steps')}, allow_nan=False))
    return 0 if result['execution_completed'] else (130 if result['cancelled'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
