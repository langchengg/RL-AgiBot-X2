set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
WHEEL_URL="$("$PY" -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["ACCEPT_ROOT"])/"evidence/mujoco-download-identity.json").read_text())["url"])')"
curl --version
curl --http1.1 --fail --location --connect-timeout 15 --max-time 45 --head "$WHEEL_URL" > "$ACCEPT_ROOT/evidence/mujoco-official-head.txt"
cat "$ACCEPT_ROOT/evidence/mujoco-official-head.txt"
curl --http1.1 --fail --location --connect-timeout 15 --max-time 45 --range 0-15 --dump-header "$ACCEPT_ROOT/evidence/mujoco-range-headers.txt" --output "$ACCEPT_ROOT/downloads/mujoco-range-probe.bin" "$WHEEL_URL"
cat "$ACCEPT_ROOT/evidence/mujoco-range-headers.txt"
wc -c "$ACCEPT_ROOT/downloads/mujoco-range-probe.bin"
