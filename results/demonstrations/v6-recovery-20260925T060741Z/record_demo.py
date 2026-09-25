"""Record one strict-loaded, deterministic v6 episode; never train or alter physics.

Source ROS Jazzy and a colcon install of this repository first. Set MUJOCO_GL=osmesa.
Use a new --output every time. PNG intermediates stay in a temporary directory.
ffmpeg/ffprobe must already be available (explicit executable paths are supported).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import pickle
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('MUJOCO_GL', 'osmesa')  # before MuJoCo's first import
import mujoco as mj
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from x2_recovery import evaluate
from x2_recovery.evaluate import prepare, physical_env, verify_policy, policy_hash
from x2_recovery.reset import integration_state
from x2_recovery.train import json_value, sha256, _state_digest


def write_json(path, data):
    path.write_text(json.dumps(json_value(data), indent=2, allow_nan=False) + '\n')


def checked(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return command


def encode(args, raw, frames, duration, output):
    # VFR preserves regular 40 ms samples AND the true short final transition.
    concat = raw / 'frames.ffconcat'
    lines = ['ffconcat version 1.0']
    for i, frame in enumerate(frames):
        lines += [f"file '{i:04d}.png'", 'option framerate 1000']
        if i + 1 < len(frames):
            lines.append(f"duration {frames[i+1]['time_s']-frame['time_s']:.9f}")
    concat.write_text('\n'.join(lines) + '\n')
    mp4 = output / 'recovery-v6.mp4'
    commands = [checked([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-n',
        '-safe', '0', '-f', 'concat', '-i', str(concat), '-fps_mode', 'vfr',
        '-enc_time_base', '1:1000', '-c:v', 'libx264', '-crf', '22', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-video_track_timescale', '1000', '-movflags', '+faststart',
        '-an', str(mp4)])]
    probe_command = [args.ffprobe, '-v', 'error', '-count_frames', '-show_streams',
                     '-show_format', '-of', 'json', str(mp4)]
    probe = json.loads(subprocess.check_output(probe_command))
    stream = probe['streams'][0]
    assert int(stream['nb_read_frames']) == len(frames)
    assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
    assert abs(float(probe['format']['duration'])-duration) < .04
    commands += [probe_command, checked([args.ffmpeg, '-v', 'error', '-i', str(mp4),
                                        '-fps_mode', 'passthrough', '-enc_time_base', '1:1000', '-f', 'null', '-'])]
    # Same raw samples, nominal 12.5 fps. Quantize actual terminal PTS to 10 ms;
    # preserve the true terminal image instead of extending a previous still.
    gif_ids = [i for i, f in enumerate(frames[:-1])
               if abs(f['time_s']/.08-round(f['time_s']/.08)) < 1e-6]
    gif_ids.append(len(frames)-1)
    pts = [round(frames[i]['time_s']*100)*10 for i in gif_ids]
    end_ms = math.ceil(duration*25-1e-8)*40
    end_ms = max(end_ms, pts[-1]+10)
    delays = np.diff(pts + [end_ms]).tolist()
    assert min(delays) > 0 and abs(end_ms/1000-duration) < .08
    images = [Image.open(raw / f'{i:04d}.png').convert('RGB') for i in gif_ids]
    # One global palette avoids frame-to-frame palette flicker.
    sheet = Image.new('RGB', (160*10, 120*math.ceil(len(images)/10)))
    for i, im in enumerate(images):
        thumb = im.copy(); thumb.thumbnail((160, 120))
        sheet.paste(thumb, ((i%10)*160, (i//10)*120))
    palette = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    gif = output / 'recovery-v6.gif'
    with gif.open('xb') as handle:
        indexed[0].save(handle, format='GIF', save_all=True, append_images=indexed[1:],
                        duration=delays, loop=0, disposal=1, optimize=True)
    with Image.open(gif) as decoded:
        actual_delays = []
        for i in range(decoded.n_frames):
            decoded.seek(i); decoded.load(); actual_delays.append(decoded.info['duration'])
        assert decoded.n_frames == len(gif_ids) and actual_delays == delays
        assert decoded.info['loop'] == 0
    assert mp4.stat().st_size < 15*1024**2 and gif.stat().st_size < 8*1024**2
    return dict(commands=commands, mp4=probe, gif=dict(width=args.width, height=args.height,
        frames=len(gif_ids), nominal_fps=12.5, durations_ms=delays, loop=0,
        duration_s=sum(delays)/1000, terminal_presentation_s=pts[-1]/1000),
        gif_source_frame_indices=gif_ids,
        encoding_method='H264/libx264; Pillow global128color GIF from the same live PNG frames',
        timing='MP4 regular25fps plus true terminal at millisecond precision (VFR); '
               'GIF regular80ms with shortened final intervals on the centisecond grid. '
               'No intro/outro, synthetic posture, interpolation, or duplicated hold.'), gif_ids, pts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--expected-checkpoint-sha256', required=True)
    p.add_argument('--seed', type=int, default=221030)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--azimuth', type=float, default=125.)
    p.add_argument('--elevation', type=float, default=-20.)
    p.add_argument('--distance', type=float, default=2.8)
    p.add_argument('--lookat', type=float, nargs=3, default=[0., 0., .65])
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--ffprobe', default='ffprobe')
    p.add_argument('--preview-gate', type=Path, help='Optional nonexistent file: create after inspecting initial PNG; 180s limit')
    args = p.parse_args()
    assert os.environ['MUJOCO_GL'] == 'osmesa'
    output = args.output.resolve()
    assert not output.exists(), 'Refusing to overwrite any output'
    training_input = args.training_run.resolve()
    assert not output.is_relative_to(training_input) and not training_input.is_relative_to(output), \
        'Output and frozen input must be disjoint'
    assert args.width%2 == args.height%2 == 0
    for tool in (args.ffmpeg, args.ffprobe):
        subprocess.run([tool, '-version'], check=True, stdout=subprocess.DEVNULL)
    if args.preview_gate:
        assert not args.preview_gate.exists()
    source = Path(evaluate.__file__).resolve().parent
    root = Path(subprocess.check_output(['git', '-C', str(source), 'rev-parse', '--show-toplevel'], text=True).strip())
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    source_hashes = {f: sha256(source/f) for f in evaluate.SOURCE_FILES}
    started = time.monotonic(); env = renderer = None
    raw = Path(tempfile.mkdtemp(prefix='x2-v6-demo-frames-'))
    frames = []; data = {k: [] for k in ('observation','next_observation','action','qpos','qvel',
        'reward','elapsed_s','stable_hold_s','terminated','truncated','physics_steps','gated_residual_rad')}
    output.mkdir(parents=True)
    try:
        env, model, original, prepared = prepare(args.training_run,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        base = physical_env(env)
        assert env.control_config.mode == 'reference_residual' and env.control_config.version == 'targets-v6'
        assert env.observation_space.shape == (149,) and env.action_space.shape == (17,)
        assert args.width <= base.model.vis.global_.offwidth and args.height <= base.model.vis.global_.offheight
        optimizer_before = _state_digest(model.policy.optimizer.state_dict())
        policy_before = policy_hash(original)
        renderer = mj.Renderer(base.model, height=args.height, width=args.width)
        camera = mj.MjvCamera()
        camera.lookat[:] = args.lookat
        camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
        def memory():
            return hashlib.sha256(pickle.dumps([env.target, env.reference, env.last_controller,
                base.previous_action, base._target, base._result, vars(base.tracker)], protocol=5)).hexdigest()
        def capture(info, step):
            before = integration_state(base.model, base.data); control_before = memory()
            base._check_model()
            renderer.update_scene(base.data, camera=camera)
            pixels = renderer.render().copy()
            assert np.array_equal(before, integration_state(base.model, base.data))
            assert memory() == control_before
            base._check_model()
            im = Image.fromarray(pixels); draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, args.width, 49), fill=(15, 21, 30))
            draw.text((10, 5), 'X2 v6 | Reference + feedback + PPO residual', font=font, fill=(240, 243, 246))
            label = 'Recovery success' if info['is_success'] else 'Independent simulation'
            draw.text((10, 27), f"Sim: {info['elapsed_sim_s']:.3f}s  |  Hold: {info['stable_duration_s']:.3f}/2.000s  |  {label}",
                      font=font, fill=(130, 235, 175) if info['is_success'] else (200, 211, 224))
            im.save(raw / f'{len(frames):04d}.png')
            frames.append(dict(index=len(frames), control_step=step, time_s=info['elapsed_sim_s'],
                               stable_hold_s=info['stable_duration_s'], success=info['is_success']))
        obs, info = env.reset(seed=args.seed)
        reset_arrays = dict(reset_observation=obs.copy(), reset_qpos=base.data.qpos.copy(), reset_qvel=base.data.qvel.copy())
        reset = {k:v for k,v in base.reset_evidence.items() if k not in ('trace','final_qpos','final_qvel')}
        capture(info, 0)
        print(json.dumps(dict(initial_frame=str(raw/'0000.png'), reset_status=reset['status'],
                              framebuffer=[args.width,args.height], raw_frames=str(raw))), flush=True)
        if args.preview_gate:
            deadline = time.monotonic()+180
            while not args.preview_gate.exists():
                if time.monotonic() > deadline: raise TimeoutError('Initial-frame inspection timeout')
                time.sleep(.2)
        while True:
            if time.monotonic()-started > 900: raise TimeoutError('Recording wall budget exceeded')
            assert obs.shape == (149,) and np.isfinite(obs).all()
            with torch.inference_mode(): action, _ = model.predict(obs, deterministic=True)
            assert action.shape == (17,) and np.isfinite(action).all()
            previous = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            values = dict(observation=previous, next_observation=obs.copy(), action=action.copy(),
                qpos=base.data.qpos.copy(), qvel=base.data.qvel.copy(), reward=reward,
                elapsed_s=info['elapsed_sim_s'], stable_hold_s=info['stable_duration_s'],
                terminated=terminated, truncated=truncated, physics_steps=info['physics_steps_executed'],
                gated_residual_rad=info['controller']['gated_residual_rad'])
            for k in data: data[k].append(values[k])
            step = len(data['reward'])
            if step%2 == 0 or terminated or truncated: capture(info, step)
            if terminated or truncated: break
        duration = info['elapsed_sim_s']
        policy_after = verify_policy(model, original)
        assert policy_after == policy_before and _state_digest(model.policy.optimizer.state_dict()) == optimizer_before
        assert all(sha256(source/f) == h for f,h in source_hashes.items())
        assert all(sha256(prepared['input_sources'][f]) == h for f,h in prepared['hashes'].items())
        assert evaluate.validate_environment(env, prepared['saved'], model) == prepared['identity']
        assert info['is_success'], 'Actual episode failed; preserve evidence, do not publish as success'
        media, gif_ids, gif_pts = encode(args, raw, frames, duration, output)
        np.savez_compressed(output/'trace.npz', **{k:np.asarray(v) for k,v in data.items()}, **reset_arrays,
            frame_control_step=[f['control_step'] for f in frames], frame_time_s=[f['time_s'] for f in frames],
            frame_hold_s=[f['stable_hold_s'] for f in frames], frame_success=[f['success'] for f in frames],
            gif_source_frame_index=gif_ids, gif_presentation_ms=gif_pts)
        manifest = dict(kind='independent demonstration reproduction', formal_evaluation=False,
            recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_commit=head,
            controller='repaired reference + bounded torso/ankle analytical feedback + trained PPO residual',
            training_run=os.path.relpath(prepared['directory'], root), input_sha256=prepared['hashes'],
            source_sha256=source_hashes, model_source=prepared['training']['provenance']['model_source'],
            model_fingerprint=prepared['identity']['model_fingerprint'],
            versions=dict(python=platform.python_version(), mujoco=mj.__version__, torch=torch.__version__,
                          os=platform.platform(), architecture=platform.machine()),
            strict_probe=prepared['consistency'], seed=args.seed, reset_seed=info['reset_seed'], reset=reset,
            success=info['is_success'], termination_reason=info['termination_reason'],
            truncation_reason=info['truncation_reason'], simulated_duration_s=duration,
            longest_continuous_hold_s=max(data['stable_hold_s']), control_steps=len(data['reward']),
            physics_steps=sum(data['physics_steps']), final_transition_physics_steps=data['physics_steps'][-1],
            reset_settling_s=base.episode_start_time, terminal_info=info,
            render=dict(backend='OSMesa', fixed_camera=True, camera=dict(lookat=args.lookat,
                azimuth=args.azimuth,elevation=args.elevation,distance=args.distance),
                size_px=[args.width,args.height], nominal_fps=25, frames=len(frames),
                frame_schedule='Initial handoff, every40ms of real simulation, plus true terminal',
                full_integration_and_controller_memory_exactly_unchanged_every_render=True),
            policy_state_sha256_before=policy_before, policy_state_sha256_after=policy_after,
            optimizer_unchanged=True, inputs_unchanged=True, core_source_unchanged=True,
            recording_command=[sys.executable, *sys.argv], recording_cwd=str(Path.cwd()),
            media=media, artifacts={n:dict(bytes=(output/n).stat().st_size,sha256=sha256(output/n))
                for n in ('recovery-v6.mp4','recovery-v6.gif','trace.npz')},
            recorder_sha256=sha256(__file__), wall_seconds=time.monotonic()-started,
            local_visual_review='Pending separate full playback and sampled-frame inspection',
            public_page_check='Not asserted before publication; post-push check reported separately')
        write_json(output/'manifest.json', manifest)
        print(json.dumps(dict(success=info['is_success'],duration=duration,hold=info['stable_duration_s'],
                              output=str(output),raw_frames=str(raw))), flush=True)
    except BaseException as exc:
        if data['reward']:
            np.savez_compressed(output/'partial-trace.npz', **{k:np.asarray(v) for k,v in data.items()})
        write_json(output/'error.json', dict(error=repr(exc),frames=len(frames),steps=len(data['reward']),raw_frames=str(raw)))
        raise
    finally:
        if renderer is not None: renderer.close()
        if env is not None: env.close()


if __name__ == '__main__':
    main()
