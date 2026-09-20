"""Bounded runtime, model, standing and Gymnasium environment diagnostics."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

# Must be selected before MuJoCo/PyOpenGL import in this process.
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco as mj
import numpy as np

from .model import SCENE, load_effective_model, require

SIMPLE_SCENE = """
<mujoco model="runtime_probe">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 -2 3"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.3 0.4 0.5 1"/>
    <body pos="0 0 1">
      <joint name="probe_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.6" size="0.05" mass="1"
            rgba="0.9 0.3 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def report(check, **values):
    print(json.dumps({"check": check, **values}), flush=True)


def runtime_identity():
    import gymnasium as gym
    import rclpy
    from rclpy.utilities import get_rmw_implementation_identifier
    from sensor_msgs.msg import JointState
    import stable_baselines3 as sb3
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report("identity", python=sys.version.split()[0], executable=sys.executable, prefix=sys.prefix,
           base_prefix=sys.base_prefix, ros_distro=os.environ.get("ROS_DISTRO"),
           rmw=get_rmw_implementation_identifier(), threads=torch.get_num_threads(),
           cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available())
    for module in (mj, np, gym, sb3, torch, rclpy):
        report("module", name=module.__name__, version=getattr(module, "__version__", None),
               path=module.__file__)
    for message in (JointState, String, Trigger):
        report("interface", name=message.__name__,
               path=sys.modules[message.__module__].__file__)
    tensor = torch.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
    layer = torch.nn.Linear(3, 2, device="cpu")
    loss = layer(tensor).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    assert tensor.data_ptr() == tensor.numpy().__array_interface__["data"][0]
    report("numpy_torch_forward_backward", result="PASS", loss=loss.item())


def simple_model():
    model = mj.MjModel.from_xml_string(SIMPLE_SCENE)
    data = mj.MjData(model)
    data.qpos[0] = 0.4
    mj.mj_forward(model, data)
    return model, data


def finite_physics(data):
    require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all(), "Nonfinite physics state")
    require(not np.any(data.warning.number), f"MuJoCo warnings: {data.warning.number}")


def physics_checks(asset_repo):
    start = time.perf_counter()
    model, data = simple_model()
    load_seconds = time.perf_counter() - start
    for _ in range(100):
        mj.mj_step(model, data)
    finite_physics(data)
    start_time = data.time
    start = time.perf_counter()
    for _ in range(2000):
        mj.mj_step(model, data)
    elapsed = time.perf_counter() - start
    finite_physics(data)
    require(np.isclose(data.time - start_time, 2000 * model.opt.timestep), "Wrong simple-scene time progression")
    report("simple_physics", result="PASS", steps=2000, warmup_steps=100,
           load_seconds=load_seconds, step_seconds=elapsed,
           steps_per_second=2000 / elapsed, simulated_seconds=data.time - start_time)
    model = load_effective_model(asset_repo).model
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    for _ in range(100):
        mj.mj_step(model, data)
    finite_physics(data)
    require(np.isclose(data.time, 0.1), "Wrong X2 time progression")
    report("effective_x2", result="PASS", nq=model.nq, nv=model.nv, nu=model.nu,
           steps=100, simulated_seconds=data.time)


def ppo_check():
    import gymnasium as gym
    import stable_baselines3 as sb3
    import torch

    env = gym.make("Pendulum-v1", render_mode=None)
    start = time.monotonic()
    try:
        obs, info = env.reset(seed=42)
        assert isinstance(info, dict) and env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        model = sb3.PPO("MlpPolicy", env, device="cpu", seed=42, n_steps=128,
                        batch_size=64, n_epochs=2, verbose=0)
        before = torch.nn.utils.parameters_to_vector(model.policy.parameters()).detach().clone()

        def check_rollout(local_vars, _):
            if time.monotonic() - start > 60:
                raise TimeoutError("PPO smoke test exceeded 60 seconds")
            for key in ("new_obs", "rewards", "actions"):
                assert np.isfinite(local_vars[key]).all(), key
            return True

        model.learn(total_timesteps=1024, callback=check_rollout)
        elapsed = time.monotonic() - start
        after = torch.nn.utils.parameters_to_vector(model.policy.parameters()).detach()
        change = float((after - before).abs().max())
        assert model.num_timesteps == 1024 and model._n_updates > 0
        assert torch.isfinite(after).all() and change > 0 and elapsed < 60
        with tempfile.TemporaryDirectory(prefix="x2-ppo-smoke-") as directory:
            checkpoint = Path(directory) / "pendulum.zip"
            model.save(checkpoint)
            loaded = sb3.PPO.load(checkpoint, device="cpu")
            obs, _ = env.reset(seed=43)
            expected, _ = model.predict(obs, deterministic=True)
            actual, _ = loaded.predict(obs, deterministic=True)
            assert np.allclose(expected, actual)
            for _ in range(32):
                action, _ = loaded.predict(obs, deterministic=True)
                assert np.isfinite(action).all() and env.action_space.contains(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                assert np.isfinite(obs).all() and np.isfinite(reward)
                if terminated or truncated:
                    obs, _ = env.reset()
        report("PPO dependency/runtime smoke test", result="PASS", timesteps=model.num_timesteps,
               optimization_epochs=model._n_updates, learn_seconds=elapsed,
               total_seconds=time.monotonic() - start, max_parameter_change=change,
               prediction_steps=32, device=str(model.device), seed=42)
    finally:
        env.close()


def render_checks(asset_repo, output):
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    for name in ("simple", "x2"):
        if name == "simple":
            model, data = simple_model()
        else:
            model = load_effective_model(asset_repo).model
            data = mj.MjData(model)
            mj.mj_forward(model, data)
        camera = mj.MjvCamera()
        camera.lookat[:] = [0, 0, 0.65]
        camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -20
        with mj.Renderer(model, height=240, width=320) as renderer:
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
        path = output / f"{name}.png"
        Image.fromarray(pixels).save(path)
        report("offscreen_frame", backend=os.environ.get("MUJOCO_GL"), path=str(path),
               shape=list(pixels.shape), pixel_std=float(pixels.std()),
               result="SAVED; visual inspection required")
        require(pixels.shape == (240, 320, 3) and pixels.std() > 1, "Invalid offscreen image")


def viewer_check():
    import mujoco.viewer

    model, data = simple_model()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.monotonic()
        while time.monotonic() - start < 2:
            assert viewer.is_running(), "Viewer closed before the availability check finished"
            viewer.sync()
            time.sleep(0.02)
    report("interactive_viewer", result="OPENED AND CLOSED; process exit must also succeed", open_seconds=2,
           backend=os.environ.get("MUJOCO_GL"))


def serve(seconds):
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node("x2_smoke_server")
    try:
        model, data = simple_model()
        status_pub = node.create_publisher(String, "/x2_smoke/status", 10)
        joint_pub = node.create_publisher(JointState, "/x2_smoke/joint_states", 10)
        published = 0
        requests = 0

        def publish():
            nonlocal published
            for _ in range(10):
                mj.mj_step(model, data)
            finite_physics(data)
            message = JointState()
            message.header.stamp = node.get_clock().now().to_msg()
            message.name = ["probe_hinge"]
            address = model.jnt_qposadr[model.joint("probe_hinge").id]
            message.position = [float(data.qpos[address])]
            joint_pub.publish(message)
            status_pub.publish(String(data="MUJOCO_SMOKE_RUNNING"))
            published += 1

        def ping(_, response):
            nonlocal requests
            requests += 1
            response.success = True
            response.message = "MuJoCo diagnostic alive"
            return response

        node.create_timer(0.1, publish)
        node.create_service(Trigger, "/x2_smoke/ping", ping)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        assert published > 0 and requests > 0, "No diagnostic service request received"
        report("smoke_server", result="PASS", publications=published, requests=requests,
               simulated_seconds=data.time, final_position=float(data.qpos[0]))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def client(seconds):
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    rclpy.init()
    node = rclpy.create_node("x2_smoke_client")
    try:
        statuses = []
        joints = []
        node.create_subscription(String, "/x2_smoke/status", statuses.append, 10)
        node.create_subscription(JointState, "/x2_smoke/joint_states", joints.append, 10)
        service = node.create_client(Trigger, "/x2_smoke/ping")
        deadline = time.monotonic() + seconds
        assert service.wait_for_service(timeout_sec=min(10, seconds)), "Service discovery timed out"
        while time.monotonic() < deadline and (not statuses or len(joints) < 3):
            rclpy.spin_once(node, timeout_sec=0.1)
        assert statuses and len(joints) >= 3, "Topic receipt timed out"
        assert all(msg.data == "MUJOCO_SMOKE_RUNNING" for msg in statuses)
        for msg in joints:
            assert msg.name == ["probe_hinge"] and len(msg.position) == 1
            assert np.isfinite(msg.position).all()
            assert msg.header.stamp.sec > 0
        assert np.ptp([msg.position[0] for msg in joints]) > 1e-8
        future = service.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=max(0, deadline-time.monotonic()))
        assert future.done(), "Service response timed out"
        response = future.result()
        assert response.success and response.message == "MuJoCo diagnostic alive"
        report("cross_process_ros", result="PASS", string_messages=len(statuses),
               joint_messages=len(joints), positions=[msg.position[0] for msg in joints],
               trigger_success=response.success, trigger_message=response.message)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def env_diagnostic(mode, asset_repo, output, seconds):
    """Bounded Step 6 checks/media; environment itself never writes evidence."""
    from dataclasses import asdict
    from datetime import datetime, timezone
    import copy
    import hashlib
    import uuid
    import warnings
    from .env import X2RecoveryEnv, EnvConfig, _diagnostic, _reward
    from .model_audit import identity, fingerprint
    from .reset import integration_state
    output=Path(output).resolve()/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6])
    output.mkdir(parents=True,exist_ok=False)
    evidence={'verdict':'INCOMPLETE','run_completed':False,'mode':mode,'command':sys.argv,
              'runtime':identity(Path(__file__).resolve().parents[3]),'checks':{},'rollouts':[],
              'recovery_training':'NOT RUN','five_episode_evaluation':'NOT EVALUATED'}
    def save():
        (output/'report.json').write_text(json.dumps(_diagnostic(evidence),indent=2,allow_nan=False)+'\n')
    save();print('Step 6 evidence: '+str(output),flush=True)
    env=None
    try:
        if mode=='env-check':
            from gymnasium.utils.env_checker import check_env as gym_check
            from stable_baselines3.common.env_checker import check_env as sb3_check
            for name,checker in [('gymnasium_checker',gym_check),('sb3_checker',sb3_check)]:
                with X2RecoveryEnv(asset_repo=asset_repo) as env:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always');checker(env,skip_render_check=True)
                    messages=[str(w.message) for w in captured]
                    for message in messages:print(name+' warning: '+message,flush=True)
                    evidence['checks'][name]={'status':'PASS','warnings':messages,'render':'separately tested'}
                save();print(name+': PASS',flush=True)
            started=time.monotonic();load_effective_model(asset_repo)
            evidence['model_load_wall_s']=time.monotonic()-started
            cases=[('zero',20.),('small_random',2.),('script',2.),('group_legs',.4),
                   ('group_arms',.4),('group_waist_head',.4),('positive',.2),('negative',.2),('mixed_full',.2)]
        else:cases=[('script',seconds)]
        for name,length in cases:
            mode_render='rgb_array' if mode=='env-record' else 'human' if mode=='env-live' else None
            started=time.monotonic()
            env=X2RecoveryEnv(render_mode=mode_render,asset_repo=asset_repo,capture_substeps=name not in ('zero','small_random'))
            creation=time.monotonic()-started
            evidence['resolved_config']=env.resolved_config();evidence['model_fingerprint']=fingerprint(env.loaded)
            evidence['saturation_denominators']={'aggregate':'saturated (joint,physics-step) pairs / (31 * executed steps)',
                'per_joint':'saturated steps for each joint / executed steps',
                'any_joint':'physics steps with any saturated joint / executed steps',
                'target_boundary':'boundary (joint,policy-action) pairs / (31 * executed policy actions)'}
            started=time.monotonic();obs,info=env.reset(seed=60);reset_cost=time.monotonic()-started
            initial=env.loaded.read_state(env.data)[0];rng=np.random.default_rng(600)
            rewards=[];rows=[];elapsed_calls=[];steps=[];images=[];frame_states=[]
            obs_min=obs.copy();obs_max=obs.copy()
            def frame():
                from PIL import Image, ImageDraw
                before=integration_state(env.model,env.data).copy()
                pixels=env.render() if mode=='env-record' else env._human.frame.copy()
                require(np.array_equal(before,integration_state(env.model,env.data)),'Rendering altered physics')
                require(pixels.dtype==np.uint8 and pixels.shape[-1]==3 and pixels.std()>1.,'Black/invalid render')
                im=Image.fromarray(pixels);draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,480,20),fill='black')
                draw.text((4,4),f'ACTUAL ENV ROLLOUT | elapsed={env.data.time-env.episode_start_time:.3f}s | NOT recovery',fill='white')
                images.append(im);frame_states.append({'time_s':float(env.data.time),'elapsed_sim_s':float(env.data.time-env.episode_start_time),
                    'qpos':env.data.qpos.copy(),'qvel':env.data.qvel.copy(),'ctrl':env.data.ctrl.copy()})
            if mode_render:frame()
            with (output/(name+'-policy.jsonl')).open('w') as policy_log, (output/(name+'-substeps.jsonl')).open('w') as physics_log:
                for k in range(int(np.ceil(length/env.control_dt))):
                    action=np.zeros(31,np.float32)
                    if name=='small_random':action=rng.uniform(-.1,.1,31).astype(np.float32)
                    if name=='positive':action[:]=1.
                    if name=='negative':action[:]=-1.
                    if name=='mixed_full':action[::2]=1.;action[1::2]=-1.
                    if name.startswith('group_'):
                        for j,r in enumerate(env.loaded.mapping):
                            selected=(('hip' in r.joint_name or 'knee' in r.joint_name) if name=='group_legs' else
                                      any(x in r.joint_name for x in ('shoulder','elbow','wrist')) if name=='group_arms' else
                                      any(x in r.joint_name for x in ('waist','head')))
                            if selected:action[j]=.3 if k<10 else -.3
                    if name=='script':
                        target=initial.copy()
                        for j,r in enumerate(env.loaded.mapping):
                            if 'knee' in r.joint_name:target[j]=.4 if k<50 else .8
                            if 'hip_pitch' in r.joint_name:target[j]=-.2 if k<50 else -.4
                            if 'elbow' in r.joint_name:target[j]=-.9
                        action=env.action_for_targets(target)
                    before=time.monotonic();obs,reward,terminated,truncated,info=env.step(action)
                    elapsed_calls.append(time.monotonic()-before);steps.append(info['physics_steps_executed'])
                    obs_min=np.minimum(obs_min,obs);obs_max=np.maximum(obs_max,obs)
                    rewards.append(reward);rows.append(info)
                    policy_log.write(json.dumps({'action':action.tolist(),'reward':reward,'terminated':terminated,'truncated':truncated,**info},allow_nan=False)+'\n')
                    for sample in env.last_substeps:physics_log.write(json.dumps(sample,allow_nan=False)+'\n')
                    if mode_render and (k%2==1 or terminated or truncated):frame()
                    if terminated or truncated:break
            total_steps=sum(steps);gamma=env.config.shaping_gamma
            case={'name':name,'seed':60,'reset_seed':env.reset_seed,'status':'PASS','policy_steps':len(rows),
                'physics_steps':total_steps,'sim_s':rows[-1]['elapsed_sim_s'],
                'termination_reason':rows[-1]['termination_reason'],'truncation_reason':rows[-1]['truncation_reason'],
                'is_success':rows[-1]['is_success'],'environment_creation_wall_s':creation,'reset_wall_s':reset_cost,
                'step_calls_wall_s':sum(elapsed_calls),'return_undiscounted':sum(rewards),
                'return_discounted':sum(gamma**k*r for k,r in enumerate(rewards)),
                'reward_terms':{key:{'min':min(r['reward_terms'][key] for r in rows),'max':max(r['reward_terms'][key] for r in rows),
                     'sum':sum(r['reward_terms'][key] for r in rows),'discounted_sum':sum(gamma**k*r['reward_terms'][key] for k,r in enumerate(rows))}
                    for key in rows[0]['reward_terms']},
                'saturation_fraction':sum(r['torque_saturation_fraction']*n for r,n in zip(rows,steps))/total_steps,
                'any_saturation_time_fraction':sum(r['control']['any_saturation_time_fraction']*n for r,n in zip(rows,steps))/total_steps,
                'joint_saturation_time_fraction':(sum((np.array(r['control']['joint_saturation_time_fraction'])*n for r,n in zip(rows,steps)),start=np.zeros(31))/total_steps).tolist(),
                'target_boundary_fraction':float(np.mean([r['control']['target_boundary_fraction'] for r in rows])),
                'peaks':{key:max(r['control'][key] for r in rows) for key in ('joint_rad_s','joint_limit_rad','floor_penetration_m',
                         'self_penetration_m','raw_torque_abs_max_Nm','applied_torque_abs_max_Nm')},
                'last_joint_velocity_rad_s':env.loaded.read_state(env.data)[1].tolist(),
                'final_state':rows[-1]['state'],'observation_min':obs_min.tolist(),'observation_max':obs_max.tolist(),
                'warnings':env.data.warning.number.tolist()}
            require(not any(case['warnings']),'Numerical warnings in real rollout')
            if name=='zero':
                require(truncated and not terminated and abs(case['sim_s']-20)<1e-9,'Default 20 s timeout not exercised')
                wall=sum(elapsed_calls[5:]);n=sum(steps[5:]);count=len(steps[5:])
                evidence['performance']={'excluded_warmup_policy_steps':5,'policy_steps':count,'physics_steps':n,
                    'wall_s':wall,'step_mean_ms':wall/count*1000,'physics_steps_per_s':n/wall,
                    'policy_steps_per_s':count/wall,'sim_s_per_wall_s':n*env.physics_dt/wall,
                    'excludes':'model loading, reset, first 5 policy steps, logging, rendering'}
            if images:
                path=output/'env-rollout.gif';images[0].save(path,save_all=True,append_images=images[1:],duration=40,loop=0)
                files=[]
                for i in sorted(set([0,len(images)//2,len(images)-1])):
                    p=output/f'frame-{i:03d}.png';images[i].save(p);files.append(p.name)
                (output/'frame-states.json').write_text(json.dumps(_diagnostic(frame_states),allow_nan=False))
                evidence['media']={'status':'GENERATED_REQUIRES_INSPECTION','path':path.name,'frames':len(images),'pngs':files,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'method':'Images of actual current env.reset/step state; render does not advance physics',
                    'backend':os.environ.get('MUJOCO_GL'),'DISPLAY':os.environ.get('DISPLAY')}
            evidence['rollouts'].append(case);env.close();env.close();save()
            print(name+': '+str(case['termination_reason'] or case['truncation_reason'] or 'bounded rollout complete'),flush=True)
        if mode=='env-check':
            # Independent standing-only fixture. Never installed into an environment.
            from .step5 import standing_pose, gains, place, rollout, summary
            from .success import StandingContext, CALIBRATED_SETTINGS
            x=load_effective_model(asset_repo);ctx=StandingContext(x);pose=standing_pose(x);d,_=place(x,pose);kp,kd=gains(x)
            samples,_,details=rollout(ctx,d,pose,kp,kd,3.,CALIBRATED_SETTINGS)
            fixture=summary(samples,details)
            require(fixture['ever_held'] and not fixture['ever_recovery'],'Standing fixture failed')
            evidence['checks']['independent_standing_fixture']={'status':'PASS',**fixture,'scope':'physical standing ONLY; not an env recovery'}
            # Synthetic comparisons have matching start potential, and report BOTH returns.
            c=EnvConfig();p=np.array([.4,.75]);g=c.shaping_gamma
            def comparison(states,terminate=False,success=False,torque=0.,hold=0.):
                values=[]
                for i,(a,b) in enumerate(zip(states,states[1:])):
                    final=i==len(states)-2
                    values.append(_reward(c,np.array(a),np.array(b),terminate and final,hold,torque,0.,success and final)[0])
                return {'transitions':len(values),'undiscounted':sum(values),'discounted':sum(g**i*v for i,v in enumerate(values))}
            evidence['reward_synthetic']={
                'scope':'constructed potential histories; NO physical recovery',
                'stationary_seated_1000':comparison([p]*1001),
                'rise_fall_100':comparison([p,np.array([.9,1.])]*50+[p]),
                'immediate_success':comparison([np.ones(2)]*2,True,True),
                'delayed_success_50_unqualified':comparison([np.ones(2)]*52,True,True),
                'early_abort':comparison([p]*2,True,False,torque=.02),
                'late_abort_50':comparison([p]*51,True,False,torque=.02),
                'limitation':'Early abort can avoid future running costs when success never occurs; no claim of reward-hacking immunity.'}
        evidence.update(verdict='AUTOMATED_PASS' if mode=='env-check' else 'AWAITING_VISUAL_REVIEW',run_completed=True,exit_code=0)
        save();return 0
    except Exception as exc:
        evidence.update(error=str(exc),error_evidence=getattr(exc,'evidence',None),exit_code=1)
        save();print('Step 6 INCOMPLETE: '+str(exc),flush=True);return 1
    finally:
        if env is not None:env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["runtime", "model", "audit", "render", "viewer", "serve", "client",
                                        "step4", "step4-live", "step4-record", "step5", "step5-review", "env-check", "env-record", "env-live"])
    parser.add_argument("--repeat", type=int, default=2, help="Step 4 demo reset count, 1..20")
    parser.add_argument("--x2-scene", type=Path, help="Legacy alias: must be the pinned Ultra scene")
    parser.add_argument("--asset-repo", type=Path)
    parser.add_argument("--image-review-from", type=Path,
                        help="Reuse visual observations from a report only if new image hashes match")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("--seconds must be in (0, 60]")
    if args.mode in ("render", "audit", "step4", "step4-live", "step4-record", "step5", "step5-review", "env-check", "env-record", "env-live") and args.output is None:
        parser.error("--output is required")
    if args.x2_scene is not None:
        scene = args.x2_scene.expanduser().resolve()
        candidate = scene.parent.parent
        if scene != candidate / SCENE:
            parser.error("--x2-scene must identify the pinned X2_URDF-v1.3.0/scene.xml")
        if args.asset_repo is not None and candidate != args.asset_repo.expanduser().resolve():
            parser.error("--asset-repo and --x2-scene disagree")
        args.asset_repo = candidate
    if args.mode.startswith("env-"):
        return env_diagnostic(args.mode, args.asset_repo, args.output, args.seconds)
    if args.mode == "step5":
        from .step5 import run
        return run(args.asset_repo, args.output)
    if args.mode == "step5-review":
        if args.image_review_from is None:
            parser.error("step5-review requires --image-review-from observations.json")
        from .step5 import review
        return review(args.output, args.image_review_from)
    if args.mode.startswith("step4"):
        from .step4 import run
        return run(args.mode, args.asset_repo, args.output, args.repeat)
    if args.mode == "runtime":
        runtime_identity()
        physics_checks(args.asset_repo)
        ppo_check()
    elif args.mode == "model":
        physics_checks(args.asset_repo)
    elif args.mode == "audit":
        from .model_audit import run_audit
        return run_audit(args.asset_repo, args.output, image_review_from=args.image_review_from)
    elif args.mode == "render":
        render_checks(args.asset_repo, args.output)
    elif args.mode == "viewer":
        viewer_check()
    elif args.mode == "serve":
        serve(args.seconds)
    else:
        client(args.seconds)
