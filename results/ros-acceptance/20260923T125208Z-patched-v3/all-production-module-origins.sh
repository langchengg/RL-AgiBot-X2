set -eo pipefail
source /opt/ros/jazzy/setup.bash
source "$ACCEPT_ROOT/install/setup.bash"
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd /tmp
"$PY" - <<'PYCODE'
import sys
sys.dont_write_bytecode=True
import datetime,hashlib,importlib,json,os,subprocess
from pathlib import Path
root=Path(os.environ['ACCEPT_ROOT']);ws=root/'repository';base='71037f7d5530a65da220345f9037ac4d29f32908'
items=[]
for file in sorted((ws/'src/x2_recovery/x2_recovery').glob('*.py')):
    name='x2_recovery' if file.stem=='__init__' else 'x2_recovery.'+file.stem
    module=importlib.import_module(name)
    raw=Path(module.__file__);actual=raw.resolve();relative=file.relative_to(ws)
    expected=subprocess.check_output(['git','-C',str(ws),'show',base+':'+str(relative)])
    item=dict(module=name,file=str(raw),realpath=str(actual),sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
              base_sha256=hashlib.sha256(expected).hexdigest(),inside_new_source=actual.is_relative_to(ws),
              imported_bytes_match_release=actual.read_bytes()==expected)
    assert item['inside_new_source'] and item['imported_bytes_match_release'],item
    items.append(item)
report=dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=base,cwd=os.getcwd(),
 executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,sys_path=sys.path,
 scope='Imports only through sourced install; no rclpy.init, ROS node, model load, reset/step, training or evaluation; pyc writing disabled.',
 count=len(items),modules=items)
out=root/'evidence/all-production-module-origins.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
PYCODE
