"""Opt-in, bounded cross-process validation against an installed package.

Run from outside the checkout, after sourcing the isolated colcon install:
  python /absolute/path/test_ros_integration.py --output-dir /new/evidence/path
The clean launch scenarios do not patch production code. A separate real-X2 run
wraps send_response/read_state/reset/step solely to record ordering and readback.
No source PYTHONPATH, extra simulator step, state edits or alternative environment.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def instrumented_node(events_path, inject_step_error=False):
    from unittest.mock import patch
    from x2_recovery import recovery_node as rn
    stream = open(events_path, 'x', buffering=1)

    def event(kind, **fields):
        stream.write(json.dumps(dict(event=kind, monotonic_s=time.monotonic(), **fields), allow_nan=False)+'\n')

    class RecordedRecovery(rn.RecoveryNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
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
                    event(_op+'_begin', episode=self.episode)
                    try:
                        result = _original(*args, **kw)
                        event(_op+'_end', episode=self.episode, elapsed_sim_s=result[-1]['elapsed_sim_s'])
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

        def _start(self, request, response):
            begin = time.monotonic()
            result = super()._start(request, response)
            event('callback_return', success=result.success, duration_s=time.monotonic()-begin,
                  episode=self.episode)
            return result

        def _sample(self):
            captured = []
            read = self.env.loaded.read_state
            def readback(data):
                q, dq = read(data)
                captured.append(dict(names=[r.joint_name for r in self.env.loaded.mapping],
                                     q=q.tolist(), dq=dq.tolist(), sim_time_s=float(data.time)))
                return q, dq
            with patch.object(self.env.loaded, 'read_state', side_effect=readback):
                message = super()._sample()
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


def records(path):
    result = []
    if not path.exists(): return result
    for line in path.read_text().splitlines():
        try:
            start = line.index('{'); result.append(json.loads(line[start:]))
        except (ValueError, json.JSONDecodeError): pass
    return result


def run(output):
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
    from ros2cli.node.daemon import is_daemon_running, shutdown_daemon
    from types import SimpleNamespace

    daemon_args = SimpleNamespace()
    daemon_was_running = is_daemon_running(daemon_args)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(controller='scripted_baseline', checks={}, processes=[], rpc=[], cli=[], episodes=[],
                  identity=dict(executable=sys.executable, cwd=os.getcwd(),
                                module=recovery_node.__file__, prefix=get_package_prefix('x2_recovery'),
                                launch=str(Path(get_package_share_directory('x2_recovery'))/'launch/recovery.launch.py'),
                                rmw=get_rmw_implementation_identifier(),
                                domain=os.environ.get('ROS_DOMAIN_ID'),
                                discovery=os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')))
    processes = []
    deadline = time.monotonic()+210.
    rclpy.init()
    node = Node('ros_integration_client')
    executor = SingleThreadedExecutor(); executor.add_node(node)
    statuses, joints = [], []
    node.create_subscription(String, '/x2/recovery_status', lambda m: statuses.append((time.monotonic(), m.data)), STATUS_QOS)
    def joint(message):
        validate_joint_state(message)
        joints.append(dict(received_monotonic_s=time.monotonic(),
                           stamp=[message.header.stamp.sec, message.header.stamp.nanosec],
                           names=message.name, q=list(message.position), dq=list(message.velocity),
                           effort=list(message.effort), frame_id=message.header.frame_id))
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
        p = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd='/tmp',
                             start_new_session=True)
        entry = dict(pid=p.pid, command=command, log=str(log), returncode=None)
        processes.append((p, handle, entry)); report['processes'].append(entry)
        return p, log

    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
            try: p.wait(timeout=8.)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=5.)
                raise AssertionError('Normal SIGINT cleanup exceeded 8s')
        for proc, handle, entry in processes:
            if proc is p:
                entry['returncode'] = p.returncode; handle.flush()
        return p.returncode

    def cli(args, name, seconds=10.):
        start = time.monotonic()
        p, log = spawn(['ros2', *args], name)
        wait(lambda: p.poll() is not None, seconds)
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
        check(label, response.success is expected and finished-started <= 1., **item)
        return item

    def ready(log):
        wait(lambda: client.service_is_ready() and any(r.get('event') == 'ready' for r in records(log))
             and any(s == 'IDLE' for _, s in statuses), 15.)
        wait(lambda: len(node.get_subscriptions_info_by_topic('/x2/joint_states')) >= 2)

    def finished(log, n):
        wait(lambda: len([r for r in records(log) if r.get('event') == 'finished']) >= n, 40.)
        record = [r for r in records(log) if r.get('event') == 'finished'][n-1]
        report['episodes'].append(record)
        wait(lambda: statuses and statuses[-1][1] == record['status'])
        return record

    prefix = Path(get_package_prefix('x2_recovery'))/'lib/x2_recovery'
    try:
        check('installed_outside_checkout', Path.cwd() == Path('/tmp') and '/install/' in str(prefix)
              and Path(report['identity']['launch']).exists()
              and Path(prefix/'recovery_node').read_text().splitlines()[0] == '#!'+sys.executable)
        statuses.clear(); joints.clear()
        launch, log = spawn(['ros2', 'launch', 'x2_recovery', 'recovery.launch.py',
                              'seed:=60', 'episode_timeout_s:=20.0', 'recovery_timeout_s:=30.0'], 'normal-launch')
        ready(log)
        check('single_launch_two_processes', 'process started with pid' in log.read_text()
              and 'telemetry_node-' in log.read_text() and 'recovery_node-' in log.read_text())
        cli(['service', 'type', '/x2/start_recovery'], 'service-type')
        cli(['topic', 'info', '/x2/recovery_status', '--verbose'], 'status-qos')
        cli(['topic', 'info', '/x2/joint_states', '--verbose'], 'joint-qos')
        for topic, wanted in [('/x2/recovery_status', STATUS_QOS), ('/x2/joint_states', JOINT_QOS)]:
            pubs = node.get_publishers_info_by_topic(topic)
            subs = node.get_subscriptions_info_by_topic(topic)
            check('qos-'+topic.rsplit('/',1)[1], len(pubs) == 1 and all(
                p.qos_profile.reliability == wanted.reliability and p.qos_profile.durability == wanted.durability
                for p in pubs+subs))
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
        check('simulation_timeout', result['status'] == 'FAILED' and result['reason'] == 'time_limit'
              and abs(result['elapsed_sim_s']-20.) < 1e-8 and result['control_steps'] == 1000
              and result['wall_s'] < 30., **result)
        check('actual_moving_telemetry', len(joints) > 20 and
              max(abs(a-b) for a,b in zip(joints[0]['q'], joints[-1]['q'])) > .1
              and all(len(j['names']) == 31 and not j['effort'] and not j['frame_id'] for j in joints)
              and 'position_rad=' in log.read_text(), samples=len(joints))
        pump(.25); count = len(joints); pump(1.2)
        check('terminal_held_no_more_samples', len(joints) == count and statuses[-1][1] == 'FAILED')
        late, late_log = spawn([str(prefix/'telemetry_node'), '--ros-args', '-r', '__node:=late_telemetry'], 'late-telemetry')
        wait(lambda: 'status=FAILED' in late_log.read_text() and 'no_sample' in late_log.read_text(), 5.)
        check('late_subscriber', 'position_rad=' not in late_log.read_text())
        stop(late)
        rpc('same-instance-accept', True)
        second = finished(log, 2)
        check('same_instance_new_episode', second['episode'] == 2 and second['control_steps'] == 1000
              and second['reason'] == 'time_limit' and len([r for r in records(log)
              if r.get('event') == 'reset_completed']) == 2)
        check('normal_cleanup', stop(launch) == 0 and '"event": "closed"' in log.read_text())
        wait(lambda: not client.service_is_ready(), 5.)
        statuses.clear(); joints.clear()
        wall, wall_log = spawn(['ros2','launch','x2_recovery','recovery.launch.py',
                                'seed:=60','episode_timeout_s:=20.0','recovery_timeout_s:=3.0'], 'wall-launch')
        ready(wall_log); rpc('wall-accept', True)
        wait(lambda: len(joints) >= 5)
        for i in range(4): rpc(f'wall-stepping-busy-{i}', False)
        result = finished(wall_log, 1)
        check('wall_timeout', result['status'] == 'FAILED' and result['reason'] == 'recovery_timeout'
              and 0. < result['elapsed_sim_s'] < 20. and result['control_steps'] >= 3
              and len(joints) >= 4, **result)
        check('wall_cleanup', stop(wall) == 0 and '"event": "closed"' in wall_log.read_text())
        wait(lambda: not client.service_is_ready(), 5.)
        statuses.clear(); joints.clear()
        events_path = output/'readback-events.jsonl'
        audited, audit_log = spawn([sys.executable, str(Path(__file__).resolve()),
                                    '--instrumented-node', str(events_path), '--ros-args',
                                    '-p','episode_timeout_s:=2.003'], 'readback-recovery')
        telemetry, telemetry_log = spawn([str(prefix/'telemetry_node')], 'readback-telemetry')
        ready(audit_log)
        rpc('readback-accept', True)
        wait(lambda: any(r['event'] == 'reset_begin' for r in records(events_path)), 3.)
        busy_reset = rpc('real-reset-busy', False)
        wait(lambda: len(joints) >= 5)
        for i in range(4): rpc(f'readback-stepping-busy-{i}', False)
        result = finished(audit_log, 1)
        events = records(events_path)
        begin = next(e for e in events if e['event'] == 'reset_begin')
        end = next(e for e in events if e['event'] == 'reset_end')
        sent = next(e for e in events if e['event'] == 'send_end' and e['success'])
        check('acceptance_sent_before_reset', sent['monotonic_s'] < begin['monotonic_s'])
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
        check('same_snapshot_actual_readback', len(compared) >= 4, matched=len(compared), max_absolute_error=0.)
        stepping = [r for r in report['rpc'] if r['label'].startswith('readback-stepping-busy')]
        last_step = max(e['monotonic_s'] for e in events if e['event'] == 'step_end')
        check('busy_during_verified_stepping_interval', all(end['monotonic_s'] < r['start_monotonic_s']
              < r['end_monotonic_s'] < last_step for r in stepping), samples=len(stepping))
        report['callback_samples_s'] = [e['duration_s'] for e in events if e['event'] == 'callback_return']
        rpc('shutdown-active-accept', True)
        wait(lambda: any(e['event'] == 'reset_begin' and e['episode'] == 2 for e in records(events_path)))
        check('active_ctrl_c_cleanup', stop(audited) == 0 and
              any(e['event'] == 'close_end' for e in records(events_path)))
        close_events = records(events_path)
        close_begin = next(e['monotonic_s'] for e in close_events if e['event'] == 'close_begin')
        check('close_after_execution_unwound', all(e['monotonic_s'] < close_begin for e in close_events
              if e['event'] in ('reset_end','reset_exception','step_end','step_exception')))
        stop(telemetry)
        wait(lambda: not client.service_is_ready(), 5.)
        statuses.clear(); joints.clear()
        faulty, fault_log = spawn([sys.executable, str(Path(__file__).resolve()),
                                  '--instrumented-node', str(output/'fault-events.jsonl'), '--inject-step-error',
                                  '--ros-args', '-p', 'episode_timeout_s:=0.203'], 'fault-injection-recovery')
        fault_telemetry, _ = spawn([str(prefix/'telemetry_node')], 'fault-injection-telemetry')
        ready(fault_log); rpc('fault-injection-accept', True)
        failed = finished(fault_log, 1)
        check('synthetic_step_fault_ends_request', failed['status'] == 'FAILED' and
              failed['reason'] == 'step_error' and 'synthetic one-shot' in failed['error']['message'])
        rpc('retry-after-fault-accept', True)
        retry = finished(fault_log, 2)
        check('real_reset_after_synthetic_fault', retry['reason'] == 'time_limit' and
              retry['episode'] == 2 and retry['control_steps'] == 11)
        stop(faulty); stop(fault_telemetry)
        wait(lambda: not client.service_is_ready(), 5.)
        for index, args in enumerate([
            ['-p','seed:=-1'], ['-p','episode_timeout_s:=0.0205'],
            ['-p','recovery_timeout_s:=0.0'], ['-p','use_sim_time:=true']]):
            bad, bad_log = spawn([str(prefix/'recovery_node'),'--ros-args',*args], f'invalid-startup-{index}')
            wait(lambda: bad.poll() is not None, 10.)
            check(f'invalid-startup-{index}', bad.returncode != 0 and '"event": "ready"' not in bad_log.read_text())
        # Missing assets are a real startup fault, isolated to this child process.
        bad, bad_log = spawn(['env','X2_ASSET_REPO=/nonexistent/x2-assets',str(prefix/'recovery_node')], 'missing-model')
        wait(lambda: bad.poll() is not None, 10.)
        check('model_load_failure', bad.returncode != 0 and '"event": "ready"' not in bad_log.read_text())
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
        except Exception as exc:
            cleanup_errors.append('CLI daemon cleanup: '+str(exc))
        report['cleanup_errors'] = cleanup_errors
        report['all_children_exited'] = all(p.poll() is not None for p,_,_ in processes)
        executor.shutdown(); node.destroy_node(); rclpy.shutdown()
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
    args = parser.parse_args()
    sys.exit(run(args.output_dir.resolve()))
