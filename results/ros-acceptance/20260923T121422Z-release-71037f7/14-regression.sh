set -eo pipefail
source /opt/ros/jazzy/setup.bash
source "$ACCEPT_ROOT/install/setup.bash"
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd /tmp
MUJOCO_GL=osmesa timeout --signal=INT --kill-after=15s 300s "$PY" -m unittest discover -s "$WS/src/x2_recovery/test" -v

