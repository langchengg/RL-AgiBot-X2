"""Headless baseline or frozen reference-residual episodes on one executor.

The service only accepts work. Jazzy's executor sends its returned response before
another callback can reset/step. Timers never catch up with multiple control steps.
"""
import json
import math
import sys
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .baseline import build_keyframes, scripted_targets
from .env import EnvConfig, X2RecoveryEnv
from .telemetry_node import STATUS_QOS, JOINT_QOS, validate_joint_state, fixed_wall_clock


def prepare_policy(training_run, expected_checkpoint_sha256):
    # Importing the training library is unnecessary for the baseline ROS path.
    from .evaluate import prepare, physical_env
    env, model, _, prepared = prepare(training_run,
        expected_checkpoint_sha256=expected_checkpoint_sha256)
    return env, physical_env(env), model, prepared


class RecoveryNode(Node):
    def __init__(self, **kwargs):
        super().__init__('recovery_node', **kwargs)
        self.env = self.physical_env = self.policy = None
        self.observation = self.last_action = None
        self.controller = 'scripted_baseline'
        self.closing = False
        self.fatal_error = False
        self.busy = self.pending = False
        self.status = 'IDLE'
        self.episode = 0
        self.control_steps = 0
        self.keyframes = None
        self.last_info = None
        self.last_result = None
        try:
            self.steady_clock = fixed_wall_clock(self)
            for name, default in (('controller', 'scripted_baseline'), ('training_run', ''),
                                  ('expected_checkpoint_sha256', ''), ('seed', 60),
                                  ('episode_timeout_s', 20.), ('recovery_timeout_s', 30.)):
                # rclpy retains/mutates descriptors, so each parameter needs its own.
                descriptor = ParameterDescriptor(read_only=True, description='Startup-only configuration')
                setattr(self, name, self.declare_parameter(name, default, descriptor).value)
            if type(self.seed) is not int or self.seed < 0:
                raise ValueError('seed must be a nonnegative integer')
            for name in ('episode_timeout_s', 'recovery_timeout_s'):
                value = getattr(self, name)
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(name+' must be finite and positive')
            started = time.monotonic()
            # The existing constructor enforces physics-step alignment; no rounding here.
            identity = {}
            if self.controller == 'scripted_baseline':
                if self.training_run or self.expected_checkpoint_sha256:
                    raise ValueError('Policy inputs require controller:=reference_residual')
                self.env = self.physical_env = X2RecoveryEnv(
                    EnvConfig(episode_timeout_s=self.episode_timeout_s), render_mode=None)
            elif self.controller == 'reference_residual':
                if not self.training_run or not self.expected_checkpoint_sha256:
                    raise ValueError('reference_residual requires training_run and expected_checkpoint_sha256')
                self.env, self.physical_env, self.policy, prepared = prepare_policy(
                    self.training_run, self.expected_checkpoint_sha256)
                saved = prepared['saved']
                if (saved.get('controller', {}).get('mode') != 'reference_residual'
                        or self.env.observation_space.shape != (149,)
                        or self.env.action_space.shape != (17,)):
                    raise ValueError('reference_residual requires the validated 149/17 controller interface')
                if self.episode_timeout_s != self.physical_env.config.episode_timeout_s:
                    raise ValueError('Policy episode_timeout_s must equal its saved configuration; '
                                     'use recovery_timeout_s for a wall-clock watchdog')
                identity = dict(training_run=str(prepared['directory']),
                    input_hashes=prepared['hashes'], observation_dimension=149, action_dimension=17,
                    policy_device='cpu', deterministic=True,
                    saved_action_consistency=prepared['consistency'],
                    model_identity=saved['identity']['model_fingerprint'],
                    scope='Simulation demonstration; full-episode constraint risks are not resolved')
            else:
                raise ValueError('controller must be scripted_baseline or reference_residual')
            self.names = [row.joint_name for row in self.physical_env.loaded.mapping]
            if not self.names or any(not n for n in self.names) or len(set(self.names)) != len(self.names):
                raise ValueError('Invalid controlled joint mapping')
            self.status_pub = self.create_publisher(String, '/x2/recovery_status', STATUS_QOS)
            self.joint_pub = self.create_publisher(JointState, '/x2/joint_states', JOINT_QOS)
            self.timer = self.create_timer(self.physical_env.control_dt, self._advance, clock=self.steady_clock)
            self.heartbeat = self.create_timer(1., self._publish_status, clock=self.steady_clock)
            self.service = self.create_service(Trigger, '/x2/start_recovery', self._start)
            self._publish_status()
            self._log('ready', create_s=time.monotonic()-started, executable=sys.executable,
                      module=__file__, seed=self.seed, episode_timeout_s=self.episode_timeout_s,
                      recovery_timeout_s=self.recovery_timeout_s, control_dt=self.physical_env.control_dt,
                      asset_repo=str(self.physical_env.loaded.asset_repo), controlled_joints=len(self.names),
                      **identity)
        except BaseException:
            try:
                if self.env is not None:
                    self.env.close()
            except Exception as exc:
                print(f'Secondary startup cleanup error: {type(exc).__name__}: {exc}', file=sys.stderr)
            finally:
                self.destroy_node()
            raise  # Preserve the startup failure even if cleanup also failed.

    def _log(self, event, **fields):
        message = json.dumps(dict(event=event, controller=self.controller,
                                  episode=self.episode, **fields), allow_nan=False)
        if self.context.ok():
            self.get_logger().info(message)
        else:
            print(message, flush=True)  # shutdown evidence; no claim of ROS delivery

    def _publish_status(self):
        self.status_pub.publish(String(data=self.status))

    def _start(self, request, response):
        started = time.monotonic()
        if self.busy or self.closing:
            response.success = False
            response.message = 'Recovery already running'
            self._log('rejected', status=self.status, callback_body_s=time.monotonic()-started)
            return response
        self.busy = self.pending = True
        self.episode += 1
        self.accepted_at = started
        self.deadline = started+self.recovery_timeout_s
        self.control_steps = 0
        self.keyframes = None
        self.observation = self.last_action = None
        self.last_info = self.last_result = None
        self.status = 'RUNNING'
        self._publish_status()
        response.success = True
        response.message = 'Recovery accepted'
        self._log('accepted', accepted_monotonic_s=started, deadline_monotonic_s=self.deadline,
                  callback_body_s=time.monotonic()-started)
        return response

    def _sample(self):
        # This callback and reset/step have the same, exclusive executor owner.
        q, dq = self.physical_env.loaded.read_state(self.physical_env.data)
        stamp = self.get_clock().now().to_msg()
        message = JointState()
        message.header.stamp = stamp
        message.name = self.names
        message.position = q.tolist()
        message.velocity = dq.tolist()
        message.effort = []
        validate_joint_state(message)
        return message

    def _expired(self, completed_at):
        return completed_at >= self.deadline

    def _advance(self):
        if not self.busy or self.closing:
            return
        started = time.monotonic()
        if self._expired(started):
            self._finish('FAILED', 'recovery_timeout', started)
            return
        operation = 'reset' if self.pending else 'step'
        completed_at = None
        try:
            if self.pending:
                self.pending = False
                self._log('reset_started', monotonic_s=started)
                before_reset = time.monotonic()
                if self._expired(before_reset):
                    self._finish('FAILED', 'recovery_timeout', before_reset)
                    return
                self.observation, info = self.env.reset(seed=self.seed)
                completed_at = time.monotonic()  # deadline decision excludes logging/DDS latency
                self.last_info = info
                expired = self._expired(completed_at)
                message = self._sample()  # invalid state always overrides a success outcome
                if self.controller == 'scripted_baseline':
                    self.keyframes = build_keyframes(message.position, self.physical_env.loaded.mapping)
                self.joint_pub.publish(message)
                self._log('reset_completed', monotonic_s=completed_at, duration_s=completed_at-started,
                          elapsed_sim_s=info['elapsed_sim_s'], episode_start_time=self.physical_env.episode_start_time,
                          reset_seed=info.get('reset_seed'))
                if expired:
                    self._finish('FAILED', 'recovery_timeout', completed_at)
                return  # first callback never performs a control step
            if self.controller == 'reference_residual':
                action, _ = self.policy.predict(self.observation, deterministic=True)
            else:
                elapsed = float(self.physical_env.data.time-self.physical_env.episode_start_time)
                targets, _ = scripted_targets(elapsed, self.keyframes)
                action = self.env.action_for_targets(targets)
            self.last_action = action.copy()
            # Target generation may take time; do not start step after the deadline.
            before_step = time.monotonic()
            if self._expired(before_step):
                self._finish('FAILED', 'recovery_timeout', before_step)
                return
            self.observation, _, terminated, truncated, info = self.env.step(action)
            completed_at = time.monotonic()
            self.control_steps += 1
            self.last_info = info
            self.last_result = dict(terminated=terminated, truncated=truncated, **info)
            expired = self._expired(completed_at)
            message = self._sample()
            self.joint_pub.publish(message)
            if expired:
                self._finish('FAILED', 'recovery_timeout', completed_at)
            elif terminated:
                success = info['is_success'] is True
                self._finish('SUCCEEDED' if success else 'FAILED', info['termination_reason'], completed_at)
            elif truncated:
                self._finish('FAILED', info['truncation_reason'], completed_at)
        except Exception as exc:
            failed_at = time.monotonic()
            # Use detached evidence already carried by the exception; never query broken dynamics.
            cause = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            self.fatal_error = isinstance(cause, (MemoryError, OSError))
            self._finish('FAILED', operation+'_error', failed_at,
                         error=dict(type=type(exc).__name__, message=str(exc),
                                    evidence=getattr(exc, 'evidence', None)),
                         call_completed_monotonic_s=completed_at)
            if self.fatal_error:
                self.closing = True
                self.timer.cancel()
                self.destroy_service(self.service)

    def _finish(self, status, reason, completed_at, **details):
        if not self.busy:
            return
        self.busy = self.pending = False
        self.keyframes = None
        self.observation = self.last_action = None
        self.status = status
        self._log('finished', status=status, reason=reason, control_steps=self.control_steps,
                  wall_s=completed_at-self.accepted_at,
                  timeout_overshoot_s=max(0., completed_at-self.deadline),
                  elapsed_sim_s=None if self.last_info is None else self.last_info.get('elapsed_sim_s'),
                  environment_result=self.last_result, **details)
        if self.context.ok():
            self._publish_status()

    def close(self):
        # Called only after spin has returned: no concurrent callback/environment use.
        if self.env is None:
            return
        self.closing = True
        self.timer.cancel()
        self.heartbeat.cancel()
        try:
            self._finish('FAILED', 'shutdown', time.monotonic())
        finally:
            env, self.env = self.env, None
            env.close()
            self._log('closed')


def main(args=None):
    node = executor = None
    code = 0
    try:
        rclpy.init(args=args)
        node = RecoveryNode()
        executor = SingleThreadedExecutor(context=node.context)
        executor.add_node(node)
        while node.context.ok() and not node.fatal_error:
            executor.spin_once(timeout_sec=1.)
        code = int(node.fatal_error)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        print(f'Recovery startup/execution failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        code = 1
    finally:
        try:
            if node is not None:
                node.close()
        except Exception as exc:
            print(f'Recovery close failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            code = 1
        finally:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return code


if __name__ == '__main__':
    sys.exit(main())
