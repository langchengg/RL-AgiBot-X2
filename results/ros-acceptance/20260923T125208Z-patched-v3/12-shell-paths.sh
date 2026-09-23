set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
python3 - <<'INNER'
import os,sys,json,pathlib
fields=['HOME','PATH','PYTHONPATH','VIRTUAL_ENV','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','ROS_DISTRO','X2_ASSET_REPO','PYTHONNOUSERSITE','PIP_CONFIG_FILE']
r=[dict(stage='clean_before_source',executable=sys.executable,prefix=sys.prefix,sys_path=sys.path,environment={k:os.environ.get(k) for k in fields})]
pathlib.Path(os.environ['ACCEPT_ROOT'],'evidence/shell-paths.json').write_text(json.dumps(r,indent=2)+'\n')
INNER
source /opt/ros/jazzy/setup.bash
python3 - <<'INNER'
import os,sys,json,pathlib
p=pathlib.Path(os.environ['ACCEPT_ROOT'],'evidence/shell-paths.json');r=json.loads(p.read_text());fields=list(r[0]['environment'])
r.append(dict(stage='after_ros_source',executable=sys.executable,prefix=sys.prefix,sys_path=sys.path,environment={k:os.environ.get(k) for k in fields}))
p.write_text(json.dumps(r,indent=2)+'\n');print(p.read_text())
INNER


