"""Bounded Step 2 checks, independent of the future X2 recovery environment."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import gymnasium as gym
import mujoco as mj
import numpy as np
import rclpy
from rclpy.utilities import get_rmw_implementation_identifier
from sensor_msgs.msg import JointState
import stable_baselines3 as sb3
from std_msgs.msg import String
from std_srvs.srv import Trigger
import torch

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
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert not np.any(data.warning.number), data.warning.number


def physics_checks(scene):
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
    assert np.isclose(data.time - start_time, 2000 * model.opt.timestep)
    report("simple_physics", result="PASS", steps=2000, warmup_steps=100,
           load_seconds=load_seconds, step_seconds=elapsed,
           steps_per_second=2000 / elapsed, simulated_seconds=data.time - start_time)
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    for _ in range(100):
        mj.mj_step(model, data)
    finite_physics(data)
    assert np.isclose(data.time, 0.1)
    report("official_x2", result="PASS", nq=model.nq, nv=model.nv, nu=model.nu,
           steps=100, simulated_seconds=data.time)


def ppo_check():
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


def render_checks(scene, output):
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    for name in ("simple", "x2"):
        if name == "simple":
            model, data = simple_model()
        else:
            model = mj.MjModel.from_xml_path(str(scene))
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
        assert pixels.shape == (240, 320, 3) and pixels.std() > 1


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["runtime", "render", "viewer", "serve", "client"])
    parser.add_argument("--x2-scene", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("--seconds must be in (0, 60]")
    if args.mode in ("runtime", "render") and args.x2_scene is None:
        parser.error("--x2-scene is required")
    if args.mode == "render" and args.output is None:
        parser.error("--output is required")
    runtime_identity()
    if args.mode == "runtime":
        physics_checks(args.x2_scene)
        ppo_check()
    elif args.mode == "render":
        render_checks(args.x2_scene, args.output)
    elif args.mode == "viewer":
        viewer_check()
    elif args.mode == "serve":
        serve(args.seconds)
    else:
        client(args.seconds)
