set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
WHEEL_URL="$("$PY" -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["ACCEPT_ROOT"])/"evidence/mujoco-download-identity.json").read_text())["url"])')"
WHEEL_NAME="$("$PY" -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["ACCEPT_ROOT"])/"evidence/mujoco-download-identity.json").read_text())["filename"])')"
curl --http1.1 --fail --location --connect-timeout 15 --max-time 120 --retry 2 --retry-all-errors --retry-delay 2 --retry-max-time 360 --continue-at - --dump-header "$ACCEPT_ROOT/evidence/mujoco-curl-attempt-1.headers" --output "$ACCEPT_ROOT/downloads/$WHEEL_NAME" "$WHEEL_URL"
"$PY" - <<'PYCODE'
from pathlib import Path
import hashlib,json,os,datetime
root=Path(os.environ['ACCEPT_ROOT']);record=json.loads((root/'evidence/mujoco-download-identity.json').read_text())
path=root/'downloads'/record['filename']
actual_size=path.stat().st_size
with path.open('rb') as stream: actual_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
report=dict(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),path=str(path),
 actual_size=actual_size,actual_sha256=actual_sha256,expected_size=record['expected_size'],expected_sha256=record['expected_sha256'],
 size_matches=actual_size==record['expected_size'],sha256_matches=actual_sha256==record['expected_sha256'])
(root/'evidence/mujoco-download-verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
assert report['size_matches'] and report['sha256_matches']
PYCODE
