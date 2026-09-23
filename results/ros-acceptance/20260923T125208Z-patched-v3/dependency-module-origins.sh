set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd /tmp
"$PY" - <<'PYCODE'
import datetime,hashlib,importlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path
root=Path(os.environ['ACCEPT_ROOT'])
pairs=[('rclpy','rclpy'),('mujoco','mujoco'),('gymnasium','gymnasium'),('stable_baselines3','stable-baselines3'),('torch','torch'),('numpy','numpy'),('matplotlib','matplotlib'),('PIL','Pillow'),('cffi','cffi'),('setuptools','setuptools'),('colcon_core','colcon-core'),('ament_package','ament-package'),('ament_index_python','ament-index-python')]
records=[]
for module,distribution in pairs:
    imported=importlib.import_module(module)
    file=str(Path(imported.__file__).resolve())
    try: version=importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError: version=getattr(imported,'__version__',None)
    if file.startswith(str(root/'repository/.venv')+'/'): source='new_project_venv'
    elif file.startswith('/opt/ros/jazzy/'): source='reused_ros_jazzy'
    elif file.startswith('/usr/lib/python3/'): source='reused_ubuntu_system_python'
    else: source='other_recorded_path'
    records.append(dict(module=module,distribution=distribution,version=version,file=file,source=source))
report=dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scope='Import/module provenance only: no rclpy.init, no node, no model load, no simulator reset/step.',
 executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,sys_path=sys.path,
 os=platform.platform(),architecture=platform.machine(),ROS_DISTRO=os.environ.get('ROS_DISTRO'),modules=records)
for r in records:
    assert '/home/lang/RL-AgiBot-X2' not in r['file'],r
    assert not ('/tmp/x2-final-ros.' in r['file'] and not r['file'].startswith(str(root)+'/')),r
out=root/'evidence/dependency-module-origins.json'
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
print('evidence_sha256',hashlib.sha256(out.read_bytes()).hexdigest())
PYCODE
