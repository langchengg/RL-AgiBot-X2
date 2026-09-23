set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source "$ACCEPT_ROOT/install/setup.bash"
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd /tmp
ros2 pkg prefix x2_recovery
ros2 pkg executables x2_recovery
head -1 "$ACCEPT_ROOT/install/x2_recovery/lib/x2_recovery/recovery_node"
head -1 "$ACCEPT_ROOT/install/x2_recovery/lib/x2_recovery/telemetry_node"
"$PY" "$ACCEPT_ROOT/evidence/capture-identity.py"
ros2 launch x2_recovery recovery.launch.py --show-args


/usr/bin/python3 "$ACCEPT_ROOT/evidence/collect-extra.py" "$ACCEPT_ROOT"

