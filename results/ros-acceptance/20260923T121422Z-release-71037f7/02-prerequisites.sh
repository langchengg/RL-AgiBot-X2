set -eo pipefail
python3 - <<'INNER'
import os,sys,json,platform
fields=['PATH','PYTHONPATH','VIRTUAL_ENV','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','PYTHONNOUSERSITE','PIP_CONFIG_FILE','ROS_DISTRO','ROS_DOMAIN_ID','RMW_IMPLEMENTATION','X2_ASSET_REPO']
print(json.dumps({'stage':'before_source','executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'sys_path':sys.path,'environment':{k:os.environ.get(k) for k in fields}},indent=2))
INNER
source /opt/ros/jazzy/setup.bash
python3 - <<'INNER'
import os,sys,json,importlib.metadata,importlib
fields=['PATH','PYTHONPATH','VIRTUAL_ENV','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','PYTHONNOUSERSITE','PIP_CONFIG_FILE','ROS_DISTRO','ROS_DOMAIN_ID','RMW_IMPLEMENTATION','X2_ASSET_REPO']
print(json.dumps({'stage':'after_ros_source','executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'sys_path':sys.path,'environment':{k:os.environ.get(k) for k in fields}},indent=2))
for name in ['rclpy','numpy','PIL','matplotlib','setuptools','colcon_core','ament_package','venv','pip']:
 try:
  m=importlib.import_module(name);print(name,getattr(m,'__version__',None),m.__file__)
 except Exception as e: print(name,type(e).__name__,str(e))
INNER
dpkg-query -W -f='${binary:Package} ${Version} ${Architecture} ${db:Status-Status}\n' python3 python3-venv python3-pip python3-setuptools python3-colcon-common-extensions python3-numpy python3-pil python3-matplotlib python3-cffi python3-nacl libosmesa6 ros-jazzy-ros-base ros-jazzy-rclpy ros-jazzy-sensor-msgs ros-jazzy-std-msgs ros-jazzy-std-srvs ros-jazzy-launch-ros ros-jazzy-rmw-fastrtps-cpp
apt-cache policy python3 python3-venv python3-pip python3-setuptools python3-colcon-common-extensions python3-numpy python3-pil python3-matplotlib libosmesa6 ros-jazzy-rclpy ros-jazzy-rmw-fastrtps-cpp

