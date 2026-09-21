"""Read-only ROS telemetry; no simulator, policy or training imports."""
import copy
import math
import sys
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String


STATUS_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
JOINT_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE)


def validate_joint_state(message):
    names = message.name
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError('Joint names must be nonempty and unique')
    if len(message.position) != len(names):
        raise ValueError('Joint name/position length mismatch')
    for field in ('position', 'velocity', 'effort'):
        values = getattr(message, field)
        if (values and len(values) != len(names)) or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid joint '+field)


def fixed_wall_clock(node):
    if node.get_parameter('use_sim_time').value is not False:
        raise ValueError('use_sim_time=true is unsupported; no /clock is published')
    descriptor = copy.deepcopy(node.describe_parameter('use_sim_time'))
    descriptor.read_only = True
    node.set_descriptor('use_sim_time', descriptor)
    return Clock(clock_type=ClockType.STEADY_TIME)


class TelemetryNode(Node):
    def __init__(self, **kwargs):
        super().__init__('telemetry_node', **kwargs)
        try:
            self.steady_clock = fixed_wall_clock(self)
            self.joint_name = self.declare_parameter(
                'joint_name', 'left_knee_joint', ParameterDescriptor(read_only=True)).value
            if not isinstance(self.joint_name, str) or not self.joint_name:
                raise ValueError('joint_name must be nonempty')
            self.status = None
            self.sample = None
            self.received_at = None
            self.running_since_ns = None
            self.create_subscription(String, '/x2/recovery_status', self._status, STATUS_QOS)
            self.create_subscription(JointState, '/x2/joint_states', self._joint, JOINT_QOS)
            self.timer = self.create_timer(1., self._report, clock=self.steady_clock)
        except BaseException:
            self.destroy_node()
            raise

    def _status(self, message):
        if message.data not in ('IDLE', 'RUNNING', 'SUCCEEDED', 'FAILED'):
            self.get_logger().warning('Ignoring invalid recovery status: '+message.data)
            return
        if message.data == 'RUNNING' and self.status != 'RUNNING':
            # Topic deliveries are not ordered across topics. Conservatively require
            # a sample acquired after this observed RUNNING transition.
            self.running_since_ns = self.get_clock().now().nanoseconds
        self.status = message.data

    def _joint(self, message):
        try:
            validate_joint_state(message)
        except ValueError as exc:
            self.get_logger().warning('Ignoring invalid joint sample: '+str(exc))
            return
        self.sample = message
        self.received_at = time.monotonic()
        # Acquisition age starts in ROS time, then advances monotonically. Never
        # convert a monotonic-clock value into a ROS timestamp.
        stamp_ns = message.header.stamp.sec*10**9+message.header.stamp.nanosec
        self.age_at_receipt = max(0., (self.get_clock().now().nanoseconds-stamp_ns)/1e9)

    def description(self):
        prefix = f'status={self.status or "waiting"} joint={self.joint_name}'
        if self.sample is None:
            return prefix+' no_sample'
        if self.joint_name not in self.sample.name:
            return prefix+' missing_joint no_sample'
        stamp = self.sample.header.stamp
        stamp_ns = stamp.sec*10**9+stamp.nanosec
        previous = (self.status == 'RUNNING' and self.running_since_ns is not None
                    and stamp_ns < self.running_since_ns)
        label = 'last_sample' if self.status != 'RUNNING' or previous else 'sample'
        if previous:
            label += ' waiting_for_current_sample'
        value = self.sample.position[self.sample.name.index(self.joint_name)]
        age = self.age_at_receipt+time.monotonic()-self.received_at
        return prefix+f' {label} position_rad={value:.6f} sample_age_s={age:.3f}'

    def _report(self):
        self.get_logger().info(self.description())


def main(args=None):
    node = executor = None
    code = 0
    try:
        rclpy.init(args=args)
        node = TelemetryNode()
        executor = SingleThreadedExecutor(context=node.context)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        print(f'Telemetry startup/execution failed: {type(exc).__name__}: {exc}', file=sys.stderr)
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
