"""Opt-in, bounded cross-process validation against an installed package.

Run from outside the checkout, after sourcing the isolated colcon install:
  python /absolute/path/test_ros_integration.py --output-dir /new/evidence/path
The clean launch scenarios do not patch production code. A separate real-X2 run
wraps send_response/read_state/reset/step solely to record ordering and readback.
No source PYTHONPATH, extra simulator step, state edits or alternative environment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    # Read-only reset evidence contains detached NumPy arrays/scalars.
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def process_group_members(group):
    """Inspect only this runner's new process group; never signal unrelated ROS."""
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                members.append(dict(pid=int(path.parent.name), state=fields[0],
                    parent_pid=int(fields[1]), process_group=int(fields[2]),
                    start_ticks=int(fields[19]),
                    command=(path.parent/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return members


def is_cli_daemon(member):
    command = member['command']
    return ('ros2-daemon' in command and '--ros-domain-id' in command
            and command[command.index('--ros-domain-id')+1] == os.environ.get('ROS_DOMAIN_ID', '0'))


def signal_owned_group(group, signum):
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        pass  # The owned process/group exited between inspection and signaling.


def instrumented_node(events_path, inject_step_error=False):
    from unittest.mock import patch
    import numpy as np
    from x2_recovery import recovery_node as rn
    stream = open(events_path, 'x', buffering=1)
    traces = {}

    def event(kind, **fields):
        stream.write(json.dumps(dict(event=kind, monotonic_s=time.monotonic(), **fields),
                                default=json_value, allow_nan=False)+'\n')

    class RecordedRecovery(rn.RecoveryNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.audit_reset_count = 0
            send = self.service.send_response
            def sending(response, header):
                event('send_begin', success=response.success, episode=self.episode)
                result = send(response, header)
                event('send_end', success=response.success, episode=self.episode)
                return result
            self.service.send_response = sending
            for operation in ('reset', 'step'):
                original = getattr(self.env, operation)
                def call(*args, _op=operation, _original=original, **kw):
                    if _op == 'reset':
                        self.audit_reset_count += 1
                    event(_op+'_begin', episode=self.episode, reset_count=self.audit_reset_count)
                    try:
                        result = _original(*args, **kw)
                        event(_op+'_end', episode=self.episode, elapsed_sim_s=result[-1]['elapsed_sim_s'],
                              physics_steps=self.physical_env._physics_steps,
                              reset_evidence=self.physical_env.reset_evidence if _op == 'reset' else None)
                        trace = traces.setdefault(self.episode, {name: [] for name in
                            ('observations', 'actions', 'qpos', 'qvel', 'reward', 'time_s')})
                        trace['observations'].append(np.asarray(result[0]).copy())
                        trace['qpos'].append(self.physical_env.data.qpos.copy())
                        trace['qvel'].append(self.physical_env.data.qvel.copy())
                        trace['time_s'].append(result[-1]['elapsed_sim_s'])
                        if _op == 'step':
                            trace['actions'].append(np.asarray(args[0]).copy())
                            trace['reward'].append(result[1])
                        if inject_step_error and self.episode == 1 and _op == 'step' and self.control_steps == 2:
                            event('fault_injection', fault_type='step_error', episode=self.episode)
                            raise RuntimeError('synthetic one-shot step failure after real physics')
                        return result
                    except BaseException as exc:
                        event(_op+'_exception', episode=self.episode, error=type(exc).__name__)
                        raise
                setattr(self.env, operation, call)
            original_close = self.env.close
            def close():
                event('close_begin'); original_close(); event('close_end')
            self.env.close = close

        def _log(self, kind, **fields):
            event('node_event', name=kind, episode=self.episode, fields=fields)
            return super()._log(kind, **fields)

        def _start(self, request, response):
            begin = time.monotonic()
            before = dict(episode=self.episode, reset_count=self.audit_reset_count,
                          control_steps=self.control_steps, busy=self.busy, pending=self.pending,
                          environment_id=id(self.env))
            result = super()._start(request, response)
            event('callback_return', success=result.success, duration_s=time.monotonic()-begin,
                  episode=self.episode, before=before,
                  after=dict(episode=self.episode, reset_count=self.audit_reset_count,
                             control_steps=self.control_steps, busy=self.busy, pending=self.pending,
                             environment_id=id(self.env)))
            return result

        def _sample(self):
            captured = []
            read = self.physical_env.loaded.read_state
            def readback(data):
                q, dq = read(data)
                captured.append(dict(names=[r.joint_name for r in self.physical_env.loaded.mapping],
                                     q=q.tolist(), dq=dq.tolist(), sim_time_s=float(data.time)))
                return q, dq
            with patch.object(self.physical_env.loaded, 'read_state', side_effect=readback):
                message = super()._sample()
            event('sample_created', episode=self.episode, step=self.control_steps,
                  stamp=[message.header.stamp.sec, message.header.stamp.nanosec])
            if self.control_steps < 5 or self.control_steps % 25 == 0:
                assert len(captured) == 1
                event('readback', episode=self.episode, step=self.control_steps,
                      stamp=[message.header.stamp.sec, message.header.stamp.nanosec], **captured[0])
            return message

    try:
        with patch.object(rn, 'RecoveryNode', RecordedRecovery):
            return rn.main()
    finally:
        stream.close()
        for episode, trace in traces.items():
            np.savez_compressed(Path(events_path).with_name(
                Path(events_path).stem+f'-episode-{episode}.npz'),
                **{name: np.asarray(values) for name, values in trace.items()})


def records(path):
    result = []
    if not path.exists(): return result
    for line in path.read_text().splitlines():
        try:
            start = line.index('{'); result.append(json.loads(line[start:]))
        except (ValueError, json.JSONDecodeError): pass
    return result


def run(output, training_run=None, checkpoint=None, reference_trace=None):
    import hashlib
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.utilities import get_rmw_implementation_identifier
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    from x2_recovery.telemetry_node import STATUS_QOS, JOINT_QOS, validate_joint_state
    from x2_recovery import recovery_node
    from ros2cli.node.daemon import DaemonNode, is_daemon_running, shutdown_daemon
    from ros2cli.daemon import get_xmlrpc_server_url
    from types import SimpleNamespace
    from xmlrpc.client import ServerProxy, Transport

    daemon_args = SimpleNamespace()
    daemon_was_running = is_daemon_running(daemon_args)
    output.mkdir(parents=True, exist_ok=False)
    policy_mode = training_run is not None
    controller = 'reference_residual' if policy_mode else 'scripted_baseline'
    policy_launch = ([f'controller:={controller}', f'training_run:={training_run}',
                      f'expected_checkpoint_sha256:={checkpoint}'] if policy_mode else [])
    policy_parameters = [part for value in policy_launch for part in ('-p', value)]
    report = dict(controller=controller, checks={}, processes=[], rpc=[], cli=[], episodes=[],
                  identity=dict(executable=sys.executable, cwd=os.getcwd(),
                                module=recovery_node.__file__, prefix=get_package_prefix('x2_recovery'),
                                launch=str(Path(get_package_share_directory('x2_recovery'))/'launch/recovery.launch.py'),
                                rmw=get_rmw_implementation_identifier(),
                                domain=os.environ.get('ROS_DOMAIN_ID'),
                                discovery=os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')))
    report['identity']['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), Path(recovery_node.__file__).resolve(),
                     Path(report['identity']['launch']).resolve())}
    report['graph_discovery_protocol'] = dict(previous_default_direct_spin_s=.5,
        explicit_direct_spin_s=2., service_type='Unchanged CLI after bounded daemon graph readiness',
        reason='An independent CLI observer may not be discovered when the integration client is ready',
        acceptance_and_stepping_rpc_limit_s=1.,
        reset_busy_timing='Recorded separately: one executor cannot preempt its bounded reset',
        runner_budget_s=300. if policy_mode else 210.)
    processes = []
    scenario = 'startup'
    received_joints = []
    status_stream = (output/'received-status.jsonl').open('x', buffering=1)
    joint_stream = (output/'received-joints.jsonl').open('x', buffering=1)
    deadline = time.monotonic()+(300. if policy_mode else 210.)
    rclpy.init()
    node = Node('ros_integration_client')
    executor = SingleThreadedExecutor(); executor.add_node(node)
    statuses, joints = [], []
    def status(message):
        received = time.monotonic()
        statuses.append((received, message.data))
        status_stream.write(json.dumps(dict(scenario=scenario, received_monotonic_s=received,
                                            status=message.data), allow_nan=False)+'\n')
    node.create_subscription(String, '/x2/recovery_status', status, STATUS_QOS)
    def joint(message):
        validate_joint_state(message)
        item = dict(scenario=scenario, received_monotonic_s=time.monotonic(),
                           received_ros_ns=node.get_clock().now().nanoseconds,
                           stamp=[message.header.stamp.sec, message.header.stamp.nanosec],
                           names=message.name, q=list(message.position), dq=list(message.velocity),
                           effort=list(message.effort), frame_id=message.header.frame_id)
        joints.append(item); received_joints.append(item)
        joint_stream.write(json.dumps(item, allow_nan=False)+'\n')
    node.create_subscription(JointState, '/x2/joint_states', joint, JOINT_QOS)
    client = node.create_client(Trigger, '/x2/start_recovery')

    def check(name, condition, **detail):
        report['checks'][name] = dict(result='PASS' if condition else 'FAIL', **detail)
        dump(output/'summary.json', report)
        if not condition: raise AssertionError(name+': '+str(detail))

    def wait(predicate, seconds=10.):
        until = min(deadline, time.monotonic()+seconds)
        while time.monotonic() < until:
            if predicate(): return
            executor.spin_once(timeout_sec=.01)
        raise TimeoutError(f'Condition timed out after {seconds}s')

    def pump(seconds):
        until = time.monotonic()+seconds
        wait(lambda: time.monotonic() >= until, seconds+1.)

    def spawn(command, name):
        log = output/(name+'.log')
        handle = log.open('x')
        started_utc, started = utc_now(), time.monotonic()
        p = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd='/tmp',
                             start_new_session=True)
        entry = dict(pid=p.pid, process_group=p.pid, command=command, cwd='/tmp',
                     started_utc=started_utc, started_monotonic_s=started,
                     environment={key: os.environ.get(key) for key in (
                         'HOME','PATH','PYTHONPATH','PYTHONNOUSERSITE','VIRTUAL_ENV','ROS_DISTRO',
                         'ROS_DOMAIN_ID','ROS_AUTOMATIC_DISCOVERY_RANGE','RMW_IMPLEMENTATION',
                         'AMENT_PREFIX_PATH','CMAKE_PREFIX_PATH','COLCON_PREFIX_PATH',
                         'X2_ASSET_REPO','X2_SCENE','MUJOCO_GL')},
                     log=str(log), returncode=None)
        if command[0] == 'env':
            for assignment in command[1:]:
                if '=' not in assignment:
                    break
                key, value = assignment.split('=', 1)
                if key in entry['environment']:
                    entry['environment'][key] = value
        processes.append((p, handle, entry)); report['processes'].append(entry)
        return p, log

    def stop(p):
        entry = next(e for proc, _, e in processes if proc is p)
        def send(signum, *, launch_parent_only=False):
            entry.setdefault('signals', []).append(dict(signal=signum.name,
                target_kind='launch_pid' if launch_parent_only else 'owned_process_group',
                target=p.pid, sent_utc=utc_now(), sent_monotonic_s=time.monotonic()))
            if launch_parent_only:
                try:
                    p.send_signal(signum)
                except ProcessLookupError:
                    pass
            else:
                signal_owned_group(p.pid, signum)
        members = process_group_members(p.pid)
        if members:
            entry['members_before_stop'] = members
        escalated = False
        if p.poll() is None:
            # Noninteractive Jazzy launch forwards SIGINT to its own children.
            # A group SIGINT here would hit them twice and interrupt their cleanup.
            send(signal.SIGINT, launch_parent_only=entry['command'][:2] == ['ros2', 'launch'])
            try: p.wait(timeout=8.)
            except subprocess.TimeoutExpired:
                escalated = True
                send(signal.SIGTERM)
                try: p.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    send(signal.SIGKILL); p.wait(timeout=3.)
        # A launch parent exiting does not itself prove its business children exited.
        remaining = [m for m in process_group_members(p.pid) if not is_cli_daemon(m)]
        if remaining:
            escalated = True
            send(signal.SIGTERM)
            until = time.monotonic()+3.
            while any(not is_cli_daemon(m) for m in process_group_members(p.pid)) and time.monotonic() < until:
                time.sleep(.02)
            if any(not is_cli_daemon(m) for m in process_group_members(p.pid)):
                send(signal.SIGKILL)
                until = time.monotonic()+2.
                while process_group_members(p.pid) and time.monotonic() < until:
                    time.sleep(.02)
        for proc, handle, entry in processes:
            if proc is p:
                entry['returncode'] = p.returncode; handle.flush()
                entry.setdefault('exit_observed_utc', utc_now())
                entry.setdefault('exit_observed_monotonic_s', time.monotonic())
                # Jazzy CLI daemon inherits its spawning CLI's process group.
                # Leave it to the existing domain-specific shutdown_daemon below.
                current_members = process_group_members(p.pid)
                entry['remaining_group_members'] = [m for m in current_members if not is_cli_daemon(m)]
                entry['deferred_cli_daemon_members'] = [m for m in current_members if is_cli_daemon(m)]
                text = Path(entry['log']).read_text()
                entry['launch_children'] = []
                for name, pid in re.findall(r'\[([^\]]+)\]: process started with pid \[(\d+)\]', text):
                    clean = f'process has finished cleanly [pid {pid}]' in text
                    failure = re.search(r'process has died \[pid '+pid+r', exit code (-?\d+)', text)
                    entry['launch_children'].append(dict(name=name, pid=int(pid),
                        returncode=0 if clean else int(failure.group(1)) if failure else None))
                break
        if escalated or entry['remaining_group_members']:
            raise AssertionError('Normal SIGINT cleanup required escalation or left group members')
        return p.returncode

    def clean_launch_children(p):
        entry = next(e for proc, _, e in processes if proc is p)
        children = entry['launch_children']
        return (len(children) == 2 and
                {c['name'].split('-', 1)[0] for c in children} == {'recovery_node', 'telemetry_node'}
                and all(c['returncode'] == 0 for c in children))

    def cli(args, name, seconds=10.):
        start = time.monotonic()
        p, log = spawn(['ros2', *args], name)
        wait(lambda: p.poll() is not None, seconds)
        stop(p)
        text = log.read_text()
        report['cli'].append(dict(command=['ros2', *args], log=str(log), exit_code=p.returncode,
                                  process_wall_s=time.monotonic()-start,
                                  scope='Includes Python startup and DDS discovery; not callback/RPC latency'))
        check(name, p.returncode == 0, output=text[:300])
        return text

    def rpc(label, expected):
        assert client.service_is_ready(), 'Client must be discovered before timing RPC'
        started = time.monotonic(); future = client.call_async(Trigger.Request())
        wait(future.done, 2.)
        finished = time.monotonic(); response = future.result()
        item = dict(label=label, start_monotonic_s=started, end_monotonic_s=finished,
                    round_trip_s=finished-started, success=response.success, message=response.message)
        report['rpc'].append(item)
        within_one_second = finished-started <= 1.
        if label == 'real-reset-busy':
            report.setdefault('performance_metrics', {})[label] = dict(
                one_second_goal_met=within_one_second, round_trip_s=finished-started,
                result='MET' if within_one_second else 'UNMET',
                explanation='Busy response waits for the current bounded reset on the exclusive executor; '
                            'the initial acceptance response is tested separately before reset')
        check(label, response.success is expected and (within_one_second or label == 'real-reset-busy'),
              one_second_goal_met=within_one_second, **item)
        return item

    def service_type_daemon_ready():
        # Jazzy `service type` lacks --no-daemon/--spin-time. Verify its actual
        # observer before invoking that unchanged CLI; do not assume our client
        # or a different CLI participant's discovery implies daemon readiness.
        started = time.monotonic()
        until = min(deadline, started+10.)
        evidence = dict(start_monotonic_s=started, deadline_monotonic_s=until,
                        socket_timeout_cap_s=5., observations=[])
        report['service_type_daemon_readiness'] = evidence

        class BoundedTransport(Transport):
            def make_connection(self, host):
                connection = super().make_connection(host)
                timeout = max(.001, min(5., until-time.monotonic()))
                connection.timeout = timeout
                if connection.sock is not None:
                    connection.sock.settimeout(timeout)
                return connection

        # DaemonNode has no timeout parameter in installed Jazzy. Replace only
        # this observer's initially unconnected proxy with a bounded transport.
        with DaemonNode(daemon_args) as graph:
            graph._proxy = ServerProxy(get_xmlrpc_server_url(), allow_none=True,
                                       transport=BoundedTransport())
            def discovered():
                services = graph.get_service_names_and_types()
                types = [types for name, types in services if name == '/x2/start_recovery']
                evidence['observations'].append(dict(monotonic_s=time.monotonic(), types=types))
                return types == [['std_srvs/srv/Trigger']]
            wait(discovered, max(.001, until-time.monotonic()))
        evidence['end_monotonic_s'] = time.monotonic()
        check('service_type_cli_daemon_ready', bool(evidence['observations']) and
              evidence['observations'][-1]['types'] == [['std_srvs/srv/Trigger']], **evidence)

    def ready(log):
        wait(lambda: client.service_is_ready() and any(r.get('event') == 'ready' for r in records(log))
             and any(s == 'IDLE' for _, s in statuses), 15.)
        wait(lambda: len(node.get_subscriptions_info_by_topic('/x2/joint_states')) >= 2)
        for p, _, entry in processes:
            if entry['log'] == str(log):
                entry['members_when_ready'] = process_group_members(p.pid)

    def finished(log, n):
        wait(lambda: len([r for r in records(log) if r.get('event') == 'finished']) >= n, 40.)
        record = [r for r in records(log) if r.get('event') == 'finished'][n-1]
        report['episodes'].append(record)
        wait(lambda: statuses and statuses[-1][1] == record['status'])
        return record

    prefix = Path(get_package_prefix('x2_recovery'))/'lib/x2_recovery'
    try:
        check('installed_outside_checkout', Path.cwd() == Path('/tmp') and 'install' in str(prefix)
              and Path(report['identity']['launch']).exists()
              and Path(prefix/'recovery_node').read_text().splitlines()[0] == '#!'+sys.executable)
        statuses.clear(); joints.clear()
        scenario = 'normal-episode-1'
        launch, log = spawn(['ros2', 'launch', 'x2_recovery', 'recovery.launch.py',
                              'seed:=221030' if policy_mode else 'seed:=60', 'episode_timeout_s:=20.0',
                              'recovery_timeout_s:=30.0', *policy_launch], 'normal-launch')
        ready(log)
        check('single_launch_two_processes', 'process started with pid' in log.read_text()
              and 'telemetry_node-' in log.read_text() and 'recovery_node-' in log.read_text())
        cli(['daemon', 'start'], 'daemon-start')
        nodes = cli(['node', 'list', '--no-daemon', '--spin-time', '2'], 'node-list')
        for attempt in range(2, 4):
            if '/recovery_node' in nodes and '/telemetry_node' in nodes:
                break
            # Each independent CLI observer needs discovery; retry is bounded and
            # never substitutes a missing business process with a synthetic name.
            nodes = cli(['node', 'list', '--no-daemon', '--spin-time', '2'], f'node-list-{attempt}')
        check('business_nodes_discovered', '/recovery_node' in nodes and '/telemetry_node' in nodes,
              direct_observer_spin_s=2., maximum_attempts=3)
        service_type_daemon_ready()
        service_type = cli(['service', 'type', '/x2/start_recovery'], 'service-type')
        check('trigger_service_type', service_type.strip() == 'std_srvs/srv/Trigger')
        cli(['topic', 'info', '/x2/recovery_status', '--verbose', '--no-daemon', '--spin-time', '2'], 'status-qos')
        cli(['topic', 'info', '/x2/joint_states', '--verbose', '--no-daemon', '--spin-time', '2'], 'joint-qos')
        for topic, wanted in [('/x2/recovery_status', STATUS_QOS), ('/x2/joint_states', JOINT_QOS)]:
            pubs = node.get_publishers_info_by_topic(topic)
            subs = node.get_subscriptions_info_by_topic(topic)
            check('qos-'+topic.rsplit('/',1)[1], len(pubs) == 1 and all(
                p.qos_profile.reliability == wanted.reliability and p.qos_profile.durability == wanted.durability
                for p in pubs+subs))
            expected_type = 'std_msgs/msg/String' if topic.endswith('recovery_status') else 'sensor_msgs/msg/JointState'
            check('type-'+topic.rsplit('/',1)[1], all(p.topic_type == expected_type for p in pubs+subs),
                  expected_type=expected_type)
        text = cli(['topic','echo','/x2/recovery_status','std_msgs/msg/String',
                    '--qos-reliability','reliable','--qos-durability','transient_local','--once','--timeout','5'], 'idle-echo')
        check('idle_observed', 'IDLE' in text and not joints)
        text = cli(['service','call','/x2/start_recovery','std_srvs/srv/Trigger','{}'], 'cli-accept')
        check('cli_accept_true', 'success=True' in text)
        wait(lambda: len(joints) >= 5)
        for i in range(8): rpc(f'normal-stepping-busy-{i}', False)
        text = cli(['service','call','/x2/start_recovery','std_srvs/srv/Trigger','{}'], 'cli-busy')
        check('cli_busy_false', 'success=False' in text)
        cli(['topic','echo','/x2/joint_states','sensor_msgs/msg/JointState','--once','--timeout','5'], 'joint-echo')
        result = finished(log, 1)
        check('real_policy_success' if policy_mode else 'simulation_timeout',
              (result['status'] == 'SUCCEEDED' and result['reason'] == 'success'
               and abs(result['elapsed_sim_s']-4.856) < 1e-8 and result['control_steps'] == 243
               if policy_mode else result['status'] == 'FAILED' and result['reason'] == 'time_limit'
               and abs(result['elapsed_sim_s']-20.) < 1e-8 and result['control_steps'] == 1000)
              and result['wall_s'] < 30., **result)
        if policy_mode:
            identity = next(r for r in records(log) if r.get('event') == 'ready')
            check('policy_startup_identity', identity['controller'] == controller
                  and identity['input_hashes']['policy_final.zip'] == checkpoint
                  and identity['observation_dimension'] == 149 and identity['action_dimension'] == 17
                  and identity['controlled_joints'] == 31 and identity['deterministic'] is True
                  and identity['policy_device'] == 'cpu', identity=identity)
        check('actual_moving_telemetry', len(joints) > 20 and
              max(abs(a-b) for a,b in zip(joints[0]['q'], joints[-1]['q'])) > .1
              and all(len(j['names']) == 31 and not j['effort'] and not j['frame_id'] for j in joints)
              and 'position_rad=' in log.read_text(), samples=len(joints))
        normal_records = records(log)
        check('normal_busy_keeps_episode_and_reset',
              len([r for r in normal_records if r.get('event') == 'accepted']) == 1
              and len([r for r in normal_records if r.get('event') == 'reset_started']) == 1
              and len([r for r in normal_records if r.get('event') == 'reset_completed']) == 1
              and all(r['episode'] == 1 and r['status'] == 'RUNNING' for r in normal_records
                      if r.get('event') == 'rejected')
              and len([r for r in normal_records if r.get('event') == 'rejected']) == 9)
        pump(.25); count = len(joints); pump(1.2)
        check('terminal_held_no_more_samples', len(joints) == count and statuses[-1][1] == result['status'])
        late, late_log = spawn([str(prefix/'telemetry_node'), '--ros-args', '-r', '__node:=late_telemetry'], 'late-telemetry')
        wait(lambda: 'status='+result['status'] in late_log.read_text() and 'no_sample' in late_log.read_text(), 5.)
        check('late_subscriber', 'position_rad=' not in late_log.read_text())
        stop(late)
        scenario = 'normal-episode-2'
        rpc('same-instance-accept', True)
        second = finished(log, 2)
        check('same_instance_new_episode', second['episode'] == 2 and second['control_steps'] == (243 if policy_mode else 1000)
              and second['reason'] == ('success' if policy_mode else 'time_limit') and len([r for r in records(log)
              if r.get('event') == 'reset_completed']) == 2)
        check('normal_cleanup', stop(launch) == 0 and '"event": "closed"' in log.read_text()
              and clean_launch_children(launch),
              children=next(e['launch_children'] for p, _, e in processes if p is launch))
        wait(lambda: not client.service_is_ready(), 5.)
        statuses.clear(); joints.clear()
        scenario = 'wall-episode-1'
        wall, wall_log = spawn(['ros2','launch','x2_recovery','recovery.launch.py',
                                'seed:=221030' if policy_mode else 'seed:=60','episode_timeout_s:=20.0',
                                'recovery_timeout_s:=3.0', *policy_launch], 'wall-launch')
        ready(wall_log); rpc('wall-accept', True)
        wait(lambda: len(joints) >= 5)
        for i in range(4): rpc(f'wall-stepping-busy-{i}', False)
        result = finished(wall_log, 1)
        check('wall_timeout', result['status'] == 'FAILED' and result['reason'] == 'recovery_timeout'
              and 0. < result['elapsed_sim_s'] < 20. and result['control_steps'] >= 3
              and len(joints) >= 4, **result)
        check('wall_cleanup', stop(wall) == 0 and '"event": "closed"' in wall_log.read_text()
              and clean_launch_children(wall),
              children=next(e['launch_children'] for p, _, e in processes if p is wall))
        wait(lambda: not client.service_is_ready(), 5.)
        statuses.clear(); joints.clear()
        scenario = 'original-launch-active-shutdown'
        active_launch, active_log = spawn(['ros2','launch','x2_recovery','recovery.launch.py',
            'seed:=221030' if policy_mode else 'seed:=60','episode_timeout_s:=20.0',
            'recovery_timeout_s:=30.0', *policy_launch], 'active-shutdown-launch')
        ready(active_log); rpc('original-launch-shutdown-accept', True)
        wait(lambda: len(joints) >= 5)
        check('original_launch_running_before_ctrl_c', statuses[-1][1] == 'RUNNING'
              and not any(r.get('event') == 'finished' for r in records(active_log)), samples=len(joints))
        active_code = stop(active_launch)
        active_finished = [r for r in records(active_log) if r.get('event') == 'finished']
        report['episodes'].extend(active_finished)
        active_process = next(e for p, _, e in processes if p is active_launch)
        check('original_launch_active_ctrl_c_cleanup', active_code == 0
              and len(active_finished) == 1 and active_finished[0]['reason'] == 'shutdown'
              and active_finished[0]['control_steps'] >= 3
              and active_finished[0]['elapsed_sim_s'] > 0.
              and '"event": "closed"' in active_log.read_text()
              and clean_launch_children(active_launch)
              and not active_process['remaining_group_members'],
              finished=active_finished, children=active_process['launch_children'],
              subscriber_last_status=statuses[-1][1],
              scope='Server shutdown/close and child exits; no claim shutdown status reached subscriber')
        wait(lambda: not client.service_is_ready(), 5.)
        statuses.clear(); joints.clear()
        scenario = 'readback-episode-1'
        events_path = output/'readback-events.jsonl'
        audited, audit_log = spawn([sys.executable, str(Path(__file__).resolve()),
                                    '--instrumented-node', str(events_path), '--ros-args',
                                    '-p','episode_timeout_s:=20.0' if policy_mode else 'episode_timeout_s:=2.003',
                                    '-p','seed:=221030' if policy_mode else 'seed:=60',
                                    *policy_parameters], 'readback-recovery')
        telemetry, telemetry_log = spawn([str(prefix/'telemetry_node')], 'readback-telemetry')
        ready(audit_log)
        rpc('readback-accept', True)
        wait(lambda: any(r['event'] == 'reset_begin' for r in records(events_path)), 3.)
        busy_reset = rpc('real-reset-busy', False)
        wait(lambda: len(joints) >= 5)
        for i in range(4): rpc(f'readback-stepping-busy-{i}', False)
        result = finished(audit_log, 1)
        if policy_mode:
            check('instrumented_real_policy_success', result['status'] == 'SUCCEEDED'
                  and result['reason'] == 'success', **result)
        events = records(events_path)
        begin = next(e for e in events if e['event'] == 'reset_begin')
        end = next(e for e in events if e['event'] == 'reset_end')
        sent = next(e for e in events if e['event'] == 'send_end' and e['success'])
        check('acceptance_sent_before_reset', sent['monotonic_s'] < begin['monotonic_s'])
        first_step = next(e for e in events if e['event'] == 'step_begin')
        check('response_reset_step_order', sent['monotonic_s'] < begin['monotonic_s']
              < end['monotonic_s'] < first_step['monotonic_s'], send_end=sent['monotonic_s'],
              reset_begin=begin['monotonic_s'], reset_end=end['monotonic_s'],
              first_step_begin=first_step['monotonic_s'])
        check('audited_supine_reset_qualified', end['reset_evidence']['status'] == 'PASS',
              reset_duration_s=end['monotonic_s']-begin['monotonic_s'],
              settled_start_s=end['reset_evidence']['settled_start_s'])
        check('request_actually_sent_during_reset', begin['monotonic_s'] <= busy_reset['start_monotonic_s']
              < end['monotonic_s'] and busy_reset['end_monotonic_s'] < next(
                  e['monotonic_s'] for e in reversed(events) if e['event'] == 'step_end'),
              reset_begin=begin['monotonic_s'], reset_end=end['monotonic_s'], **busy_reset)
        by_stamp = {tuple(j['stamp']): j for j in joints}
        compared = []
        for read in (e for e in events if e['event'] == 'readback'):
            msg = by_stamp.get(tuple(read['stamp']))
            if msg is not None:
                check('readback-'+str(read['step']), read['names'] == msg['names'] and
                      read['q'] == msg['q'] and read['dq'] == msg['dq'])
                compared.append(dict(simulator=read, received=msg))
        dump(output/'matched-readback.json', compared)
        errors = [abs(a-b) for pair in compared for field, key in [('q','q'),('dq','dq')]
                  for a,b in zip(pair['simulator'][field], pair['received'][key])]
        check('same_snapshot_actual_readback', len(compared) >= 4, matched=len(compared),
              max_absolute_error=max(errors) if errors else None)
        stepping = [r for r in report['rpc'] if r['label'].startswith('readback-stepping-busy')]
        last_step = max(e['monotonic_s'] for e in events if e['event'] == 'step_end')
        check('busy_during_verified_stepping_interval', all(end['monotonic_s'] < r['start_monotonic_s']
              < r['end_monotonic_s'] < last_step for r in stepping), samples=len(stepping))
        rejected = [e for e in events if e['event'] == 'callback_return' and not e['success']]
        check('audited_busy_no_episode_reset_or_step_mutation', len(rejected) == 5 and
              all(e['before'] == e['after'] and e['episode'] == 1 for e in rejected)
              and len([e for e in events if e['event'] == 'reset_begin']) == 1,
              callbacks=len(rejected))
        finished_at = next(e['monotonic_s'] for e in events if e['event'] == 'node_event'
                           and e['name'] == 'finished')
        pump(.2)
        terminal_events = records(events_path)
        check('no_new_sampling_after_audited_terminal', all(e['monotonic_s'] <= finished_at
              for e in terminal_events if e['event'] == 'sample_created'),
              terminal_monotonic_s=finished_at,
              samples=len([e for e in terminal_events if e['event'] == 'sample_created']))
        report['callback_samples_s'] = [e['duration_s'] for e in events if e['event'] == 'callback_return']
        report['audited_execution'] = dict(
            reset_duration_s=end['monotonic_s']-begin['monotonic_s'],
            actual_completed_control_calls=len([e for e in events if e['event'] == 'step_end']),
            actual_physics_steps=max(e['physics_steps'] for e in events if e['event'] == 'step_end'),
            elapsed_sim_s=result['elapsed_sim_s'], wall_s=result['wall_s'],
            timeout_overshoot_s=result['timeout_overshoot_s'])
        if policy_mode:
            scenario = 'readback-complete-retry-episode-2'
            rpc('readback-complete-retry-accept', True)
            policy_retry = finished(audit_log, 2)
            check('instrumented_policy_retry_success', policy_retry['status'] == 'SUCCEEDED'
                  and policy_retry['control_steps'] == 243
                  and abs(policy_retry['elapsed_sim_s']-4.856) < 1e-8, **policy_retry)
        shutdown_episode = 3 if policy_mode else 2
        scenario = f'readback-active-shutdown-episode-{shutdown_episode}'
        rpc('shutdown-active-accept', True)
        wait(lambda: any(e['event'] == 'reset_begin' and e['episode'] == shutdown_episode for e in records(events_path)))
        check('active_ctrl_c_cleanup', stop(audited) == 0 and
              any(e['event'] == 'close_end' for e in records(events_path)))
        close_events = records(events_path)
        close_begin = next(e['monotonic_s'] for e in close_events if e['event'] == 'close_begin')
        check('close_after_execution_unwound', all(e['monotonic_s'] < close_begin for e in close_events
              if e['event'] in ('reset_end','reset_exception','step_end','step_exception')))
        stop(telemetry)
        if policy_mode:
            import numpy as np
            with np.load(output/'readback-events-episode-1.npz', allow_pickle=False) as first, np.load(
                    output/'readback-events-episode-2.npz', allow_pickle=False) as retry_trace:
                for field in ('observations', 'actions', 'qpos', 'qvel', 'reward', 'time_s'):
                    left, right = first[field], retry_trace[field]
                    error = float(np.max(np.abs(left-right))) if left.shape == right.shape else None
                    check('retry_resets_policy_history_'+field, error == 0., max_absolute_error=error)
        if policy_mode and reference_trace is not None:
            import numpy as np
            with np.load(reference_trace, allow_pickle=False) as standalone, np.load(
                    output/'readback-events-episode-1.npz', allow_pickle=False) as ros_trace:
                differences = {}
                for field in ('observations', 'actions', 'qpos', 'qvel', 'reward', 'time_s'):
                    left, right = standalone[field], ros_trace[field]
                    tolerance = 1e-7 if field == 'actions' else 1e-10
                    error = float(np.max(np.abs(left-right))) if left.shape == right.shape else None
                    differences[field] = dict(standalone_shape=list(left.shape), ros_shape=list(right.shape),
                                              max_absolute_error=error, atol=tolerance, rtol=0)
                    check('standalone_policy_trajectory_'+field,
                          error is not None and error <= tolerance, **differences[field])
                report['standalone_comparison'] = dict(reference=str(reference_trace), fields=differences,
                    scope='Simulation state/action comparisons; ROS and wall timestamps intentionally differ')
        wait(lambda: not client.service_is_ready(), 5.)
        statuses.clear(); joints.clear()
        scenario = 'synthetic-fault-episode-1'
        faulty, fault_log = spawn([sys.executable, str(Path(__file__).resolve()),
                                  '--instrumented-node', str(output/'fault-events.jsonl'), '--inject-step-error',
                                  '--ros-args', '-p', 'episode_timeout_s:=20.0' if policy_mode else 'episode_timeout_s:=0.203',
                                  '-p','seed:=221030' if policy_mode else 'seed:=60',
                                  *policy_parameters], 'fault-injection-recovery')
        fault_telemetry, _ = spawn([str(prefix/'telemetry_node')], 'fault-injection-telemetry')
        ready(fault_log); rpc('fault-injection-accept', True)
        failed = finished(fault_log, 1)
        check('synthetic_step_fault_ends_request', failed['status'] == 'FAILED' and
              failed['reason'] == 'step_error' and 'synthetic one-shot' in failed['error']['message'])
        scenario = 'retry-after-synthetic-fault-episode-2'
        rpc('retry-after-fault-accept', True)
        retry = finished(fault_log, 2)
        check('real_reset_after_synthetic_fault', retry['reason'] == ('success' if policy_mode else 'time_limit') and
              retry['episode'] == 2 and retry['control_steps'] == (243 if policy_mode else 11))
        stop(faulty); stop(fault_telemetry)
        wait(lambda: not client.service_is_ready(), 5.)
        for index, args in enumerate([
            ['-p','seed:=-1'], ['-p','episode_timeout_s:=0.0205'],
            ['-p','recovery_timeout_s:=0.0'], ['-p','use_sim_time:=true'],
            ['-p','episode_timeout_s:=0.0'], ['-p','episode_timeout_s:=-1.0']]):
            scenario = f'invalid-startup-{index}'
            bad, bad_log = spawn([str(prefix/'recovery_node'),'--ros-args',*args], f'invalid-startup-{index}')
            wait(lambda: bad.poll() is not None, 10.)
            stop(bad)
            check(f'invalid-startup-{index}', bad.returncode != 0 and '"event": "ready"' not in bad_log.read_text()
                  and 'Recovery startup/execution failed:' in bad_log.read_text())
        if policy_mode:
            import hashlib
            import shutil
            import tempfile
            # All corrupt fixtures are disposable copies. Frozen policy inputs are read only.
            with tempfile.TemporaryDirectory(prefix='x2-ros-invalid-inputs-') as temporary:
                fixtures = []
                for label in ('missing-reference', 'incompatible-interface'):
                    directory = Path(temporary)/label
                    directory.mkdir()
                    for name in ('policy_final.zip', 'resolved_config.json', 'manifest.json',
                                 'reload_validation.json', 'reload_probe.npz', 'progress.csv'):
                        shutil.copyfile(training_run/name, directory/name)
                    config = json.loads((directory/'resolved_config.json').read_text())
                    manifest = json.loads((directory/'manifest.json').read_text())
                    if label == 'missing-reference':
                        del config['controller']['reference_stages']
                    else:
                        config['environment']['action_shape'] = [31]
                        config['identity']['controller']['action_shape'] = [31]
                        config['identity']['action']['shape'] = [31]
                        manifest['identity'] = config['identity']
                    dump(directory/'resolved_config.json', config)
                    manifest['config_sha256'] = hashlib.sha256(
                        (directory/'resolved_config.json').read_bytes()).hexdigest()
                    dump(directory/'manifest.json', manifest)
                    fixtures.append((label, ['-p', f'training_run:={directory}'],
                        'Incomplete controller configuration' if label == 'missing-reference'
                        else 'Controller interface shape mismatch'))
                cases = [
                    ('missing-policy-path', ['-p', 'training_run:=/nonexistent/x2-controller'], 'Missing input:'),
                    ('wrong-policy-hash', ['-p', 'expected_checkpoint_sha256:='+'f'*64], 'Checkpoint identity mismatch'),
                    ('policy-timeout-mismatch', ['-p', 'episode_timeout_s:=2.0'], 'saved configuration'),
                    *fixtures]
                for label, override, expected_message in cases:
                    scenario = label
                    bad, bad_log = spawn([str(prefix/'recovery_node'), '--ros-args',
                                          *policy_parameters, *override], label)
                    wait(lambda: bad.poll() is not None, 20.)
                    stop(bad)
                    check(label, bad.returncode != 0 and '"event": "ready"' not in bad_log.read_text()
                          and expected_message in bad_log.read_text(), expected_error=expected_message,
                          scope='Real startup rejection; tampered fixtures are temporary copies')
        # Missing assets are a real startup fault, isolated to this child process.
        bad, bad_log = spawn(['env','X2_ASSET_REPO=/nonexistent/x2-assets',str(prefix/'recovery_node')], 'missing-model')
        wait(lambda: bad.poll() is not None, 10.)
        stop(bad)
        check('model_load_failure', bad.returncode != 0 and '"event": "ready"' not in bad_log.read_text()
              and 'Missing asset repository:' in bad_log.read_text())
        check('startup_failures_no_service', not client.service_is_ready())
        report['result'] = 'PASS'
        return 0
    except BaseException as exc:
        report['result'] = 'FAIL'; report['error'] = dict(type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
        return 1
    finally:
        cleanup_errors = []
        for p, handle, entry in reversed(processes):
            try:
                stop(p)
            except BaseException as exc:
                cleanup_errors.append(str(exc))
            entry['returncode'] = p.poll(); handle.close()
        try:
            report['daemon_was_running'] = daemon_was_running
            if not daemon_was_running:
                report['own_cli_daemon_stopped'] = shutdown_daemon(daemon_args, timeout=5.)
                if is_daemon_running(daemon_args):
                    cleanup_errors.append('Runner-created CLI daemon is still running')
        except Exception as exc:
            cleanup_errors.append('CLI daemon cleanup: '+str(exc))
        report['all_children_exited'] = all(p.poll() is not None for p,_,_ in processes)
        until = time.monotonic()+2.
        while any(process_group_members(p.pid) for p, _, _ in processes) and time.monotonic() < until:
            time.sleep(.02)
        report['remaining_owned_group_members'] = [member for p, _, _ in processes
                                                    for member in process_group_members(p.pid)]
        if not report['all_children_exited'] or report['remaining_owned_group_members']:
            cleanup_errors.append('Runner-owned processes remain')
        report['telemetry_by_scenario'] = {}
        for label in dict.fromkeys(j['scenario'] for j in received_joints):
            samples = [j for j in received_joints if j['scenario'] == label]
            stamps = [j['stamp'][0]+j['stamp'][1]/1e9 for j in samples]
            arrivals = [j['received_monotonic_s'] for j in samples]
            def intervals(values):
                differences = [b-a for a,b in zip(values, values[1:])]
                return dict(count=len(differences), minimum_s=min(differences) if differences else None,
                            median_s=statistics.median(differences) if differences else None,
                            maximum_s=max(differences) if differences else None)
            knee = [j['q'][j['names'].index('left_knee_joint')] for j in samples]
            report['telemetry_by_scenario'][label] = dict(samples=len(samples),
                acquisition_intervals=intervals(stamps), receipt_intervals=intervals(arrivals),
                stamp_semantics='ROS acquisition clock; not MuJoCo simulation time',
                knee_joint='left_knee_joint', knee_min_rad=min(knee), knee_max_rad=max(knee),
                maximum_age_at_receipt_s=max((j['received_ros_ns']-(j['stamp'][0]*10**9+j['stamp'][1]))/1e9
                                             for j in samples))
        report['telemetry_terminal_log_lines'] = [dict(log=entry['log'], line=line)
            for _, _, entry in processes for line in Path(entry['log']).read_text().splitlines()
            if ('last_sample' in line or 'no_sample' in line or 'waiting_for_current_sample' in line)]
        for name, cleanup in [('executor', executor.shutdown), ('client_node', node.destroy_node),
                              ('ros_context', rclpy.shutdown)]:
            try:
                cleanup()
            except BaseException as exc:
                cleanup_errors.append(name+': '+str(exc))
        status_stream.close(); joint_stream.close()
        report['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            report['result'] = 'FAIL'
            report['cleanup_incomplete'] = True
        dump(output/'summary.json', report)
        if cleanup_errors: raise RuntimeError('; '.join(cleanup_errors))


if __name__ == '__main__':
    if '--instrumented-node' in sys.argv:
        index = sys.argv.index('--instrumented-node')
        path = sys.argv[index+1]; del sys.argv[index:index+2]
        inject = '--inject-step-error' in sys.argv
        if inject: sys.argv.remove('--inject-step-error')
        sys.exit(instrumented_node(path, inject))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, help='Enable real reference-residual acceptance')
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--reference-trace', type=Path, help='Standalone control_trace.npz for numeric comparison')
    args = parser.parse_args()
    if bool(args.training_run) != bool(args.expected_checkpoint_sha256):
        parser.error('--training-run and --expected-checkpoint-sha256 are required together')
    sys.exit(run(args.output_dir.resolve(),
                 args.training_run.resolve() if args.training_run else None,
                 args.expected_checkpoint_sha256,
                 args.reference_trace.resolve() if args.reference_trace else None))
