set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd "$WS"
"$PY" -m pip install --no-cache-dir --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
"$PY" -m pip check
"$PY" -m pip freeze --all > "$ACCEPT_ROOT/evidence/python-packages.txt"
cp .venv/pyvenv.cfg "$ACCEPT_ROOT/evidence/pyvenv.cfg"
sha256sum "$ACCEPT_ROOT/evidence/python-packages.txt" "$ACCEPT_ROOT/evidence/pyvenv.cfg"
