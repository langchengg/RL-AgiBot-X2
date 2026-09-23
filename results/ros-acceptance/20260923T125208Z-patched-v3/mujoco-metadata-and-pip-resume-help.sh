set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
"$PY" -m pip --version
"$PY" -m pip download --help > "$ACCEPT_ROOT/evidence/pip-download-help.txt"
"$PY" - <<'PYCODE'
from pathlib import Path
import hashlib,json,os,urllib.request,datetime
root=Path(os.environ['ACCEPT_ROOT'])
text=(root/'evidence/pip-download-help.txt').read_text()
print('pip_download_resume_options=',[line.strip() for line in text.splitlines() if 'resume' in line.lower()])
url='https://pypi.org/pypi/mujoco/3.13.0/json'
with urllib.request.urlopen(url, timeout=30) as response:
    data=response.read()
(root/'evidence/mujoco-pypi-metadata.json').write_bytes(data)
metadata=json.loads(data)
filename='mujoco-3.13.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl'
matches=[item for item in metadata['urls'] if item['filename']==filename]
assert len(matches)==1
item=matches[0]
record=dict(source_url=url,retrieved_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 metadata_sha256=hashlib.sha256(data).hexdigest(),filename=filename,url=item['url'],expected_size=item['size'],
 expected_sha256=item['digests']['sha256'],transport_scope='New official artifact download after two pip hash failures; no old cache, pin change or hash bypass.')
(root/'evidence/mujoco-download-identity.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
(root/'downloads').mkdir(exist_ok=True)
print(json.dumps(record,indent=2,allow_nan=False))
PYCODE
