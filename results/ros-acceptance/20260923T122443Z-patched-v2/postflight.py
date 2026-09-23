import datetime,json,os,pathlib,time
root=pathlib.Path(os.environ['ACCEPT_ROOT'])
import rclpy
from ros2cli.node.daemon import is_daemon_running
from types import SimpleNamespace
owned=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
  env=dict(x.split('=',1) for x in p.joinpath('environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
  runtime=any(pathlib.Path(a).name in ['ros2','recovery_node','telemetry_node','test_ros_integration.py','ros2-daemon'] for a in args)
  if env.get('ACCEPT_ROOT')==str(root) and runtime:owned.append(dict(pid=int(p.name),command=args))
 except (OSError,ValueError):pass
rclpy.init();node=rclpy.create_node('x2_final_postflight')
try:
 end=time.monotonic()+4.
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
 report=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),domain=os.environ.get('ROS_DOMAIN_ID'),rmw=os.environ.get('RMW_IMPLEMENTATION'),owned_runtime_processes=owned,nodes=node.get_node_names_and_namespaces(),services=node.get_service_names_and_types(),status_publishers=len(node.get_publishers_info_by_topic('/x2/recovery_status')),joint_publishers=len(node.get_publishers_info_by_topic('/x2/joint_states')),daemon_running=is_daemon_running(SimpleNamespace()))
 report['no_residual_business_graph']=not any(n in ['recovery_node','telemetry_node','late_telemetry'] for n,_ in report['nodes']) and not any(n=='/x2/start_recovery' for n,_ in report['services']) and report['status_publishers']==report['joint_publishers']==0
 report['result']='PASS' if not owned and report['no_residual_business_graph'] and not report['daemon_running'] else 'FAIL'
 (root/'evidence/postflight.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
 assert report['result']=='PASS'
finally:node.destroy_node();rclpy.shutdown()
