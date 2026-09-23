set -eo pipefail
source /opt/ros/jazzy/setup.bash
source "$ACCEPT_ROOT/install/setup.bash"
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd /tmp
"$PY" "$ACCEPT_ROOT/evidence/postflight.py"
git -C "$WS" status --short
git -C "$X2_ASSET_REPO" status --short

