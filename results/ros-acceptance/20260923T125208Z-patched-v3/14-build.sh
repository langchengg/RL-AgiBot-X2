set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd "$WS"
test ! -e "$ACCEPT_ROOT/build"
test ! -e "$ACCEPT_ROOT/install"
test ! -e "$ACCEPT_ROOT/log"
"$PY" /usr/bin/colcon list --base-paths src
"$PY" /usr/bin/colcon --log-base "$ACCEPT_ROOT/log" build --base-paths src --packages-select x2_recovery --symlink-install --build-base "$ACCEPT_ROOT/build" --install-base "$ACCEPT_ROOT/install"


