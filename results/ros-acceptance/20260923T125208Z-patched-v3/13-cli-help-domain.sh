set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
ros2 topic echo --help
ros2 node list --help
ros2 service call --help
ros2 service type --help
ros2 topic info --help
ros2 daemon start --help
/usr/bin/python3 - <<'INNER'
import json,os,time,rclpy
from rclpy.utilities import get_rmw_implementation_identifier
from ros2cli.node.daemon import is_daemon_running
from types import SimpleNamespace
from pathlib import Path
report={'domain':os.environ['ROS_DOMAIN_ID'],'rmw':get_rmw_implementation_identifier(),'discovery':os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'],'daemon_already_running':is_daemon_running(SimpleNamespace())}
rclpy.init();node=rclpy.create_node('x2_domain_preflight')
try:
 end=time.monotonic()+4
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.1)
 report['nodes']=node.get_node_names_and_namespaces();report['services']=node.get_service_names_and_types();report['topics']=node.get_topic_names_and_types()
 report['status_publishers']=len(node.get_publishers_info_by_topic('/x2/recovery_status'));report['joint_publishers']=len(node.get_publishers_info_by_topic('/x2/joint_states'))
 report['conflict']=any(n in ['recovery_node','telemetry_node'] for n,_ in report['nodes']) or any(n=='/x2/start_recovery' for n,_ in report['services']) or report['status_publishers']>0 or report['joint_publishers']>0
 Path(os.environ['ACCEPT_ROOT'],'evidence','domain-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));assert not report['conflict']
finally:node.destroy_node();rclpy.shutdown()
INNER



