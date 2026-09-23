set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
timeout --signal=INT --kill-after=10s 120s git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 clone --depth 1 --filter=blob:none --no-checkout https://github.com/langchengg/RL-AgiBot-X2.git "$WS"
timeout --signal=INT --kill-after=10s 120s git -C "$WS" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 sparse-checkout set src
timeout --signal=INT --kill-after=10s 120s git -C "$WS" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 checkout --detach 71037f7d5530a65da220345f9037ac4d29f32908
git -C "$WS" rev-parse HEAD
git -C "$WS" status --short
