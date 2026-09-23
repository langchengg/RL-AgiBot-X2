import datetime,hashlib,importlib,importlib.metadata as md,json,os,pathlib,platform,subprocess,sys
root=pathlib.Path(os.environ['ACCEPT_ROOT']);ws=pathlib.Path(os.environ['WS'])
fields=['HOME','PATH','PYTHONPATH','VIRTUAL_ENV','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','PYTHONNOUSERSITE','PIP_CONFIG_FILE','ROS_DISTRO','ROS_DOMAIN_ID','RMW_IMPLEMENTATION','ROS_AUTOMATIC_DISCOVERY_RANGE','X2_ASSET_REPO','X2_SCENE','MUJOCO_GL']
report=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),isolation_scope='同一 Ubuntu ARM64 VM 上，新的源码、项目虚拟环境、模型副本、shell 和构建产物；复用已声明的系统 ROS/Ubuntu 依赖。',new=['GitHub partial/sparse clone: root files and complete src; historical results not checked out','project .venv with system-site-packages','independently downloaded pinned model','env -i equivalent subprocess, bash --noprofile --norc, new HOME','build/install/log/evidence'],reused=['existing Parallels Ubuntu 24.04 ARM64 VM','/usr/bin/python3','/opt/ros/jazzy','declared Ubuntu/ROS packages in system inventory'],executable=sys.executable,real_executable=os.path.realpath(sys.executable),prefix=sys.prefix,base_prefix=sys.base_prefix,sys_path=sys.path,platform=platform.platform(),machine=platform.machine(),os_release=pathlib.Path('/etc/os-release').read_text(),environment={k:os.environ.get(k) for k in fields},pyvenv_cfg=(ws/'.venv/pyvenv.cfg').read_text(),modules={})
for name,dist in [('rclpy','rclpy'),('mujoco','mujoco'),('gymnasium','gymnasium'),('stable_baselines3','stable-baselines3'),('torch','torch'),('numpy','numpy'),('matplotlib','matplotlib'),('PIL','Pillow'),('cffi','cffi'),('setuptools','setuptools'),('pip','pip'),('colcon_core','colcon-core'),('ament_package','ament-package'),('ament_index_python','ament-index-python'),('sensor_msgs','sensor-msgs'),('std_msgs','std-msgs'),('std_srvs','std-srvs'),('launch','launch'),('launch_ros','launch-ros'),('x2_recovery.recovery_node','x2-recovery'),('x2_recovery.telemetry_node','x2-recovery')]:
 m=importlib.import_module(name)
 try: version=md.version(dist)
 except md.PackageNotFoundError: version=getattr(m,'__version__',None)
 p=m.__file__; report['modules'][name]=dict(version=version,path=p,realpath=os.path.realpath(p))
from ament_index_python.packages import get_package_prefix,get_package_share_directory
report['package_prefix']=get_package_prefix('x2_recovery');report['package_share']=get_package_share_directory('x2_recovery')
report['entrypoints']={n:(pathlib.Path(report['package_prefix'])/'lib/x2_recovery'/n).read_text().splitlines()[0] for n in ['recovery_node','telemetry_node']}
report['source_sha256']={str(p.relative_to(ws)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ws.glob('src/**/*')) if p.is_file() and p.suffix in ('.py','.xml','.cfg') and '__pycache__' not in p.parts and not any('.egg-info' in x for x in p.parts)}
report['launch_sha256']=hashlib.sha256((pathlib.Path(report['package_share'])/'launch/recovery.launch.py').read_bytes()).hexdigest()
for key,args in [('commit',['git','rev-parse','HEAD']),('status',['git','status','--porcelain=v1']),('diff',['git','diff','--stat'])]:report[key]=subprocess.check_output(args,cwd=ws,text=True).strip()
from x2_recovery.model import load_effective_model,COMMIT,REPOSITORY,SOURCE_HASHES
from x2_recovery.model_audit import fingerprint
loaded=load_effective_model()
report['model']=dict(repository=REPOSITORY,commit=COMMIT,asset_repo=str(loaded.asset_repo),fingerprint=fingerprint(loaded),source_sha256={p:hashlib.sha256((loaded.asset_repo/p).read_bytes()).hexdigest() for p in SOURCE_HASHES},license_present=(loaded.asset_repo/'LICENSE').exists(),joint_names=[r.joint_name for r in loaded.mapping],physics_timestep_s=loaded.model.opt.timestep)
report['loaded_module_files']={n:os.path.realpath(m.__file__) for n,m in sorted(sys.modules.items()) if getattr(m,'__file__',None)}
report['relative_module_file_labels']={n:dict(raw_file=m.__file__,exists_as_file=pathlib.Path(m.__file__).is_file(),implementation_source=getattr(sys.modules.get('torch._classes' if n=='torch.classes' else 'torch._ops' if n=='torch.ops' else ''),'__file__',None)) for n,m in sorted(sys.modules.items()) if getattr(m,'__file__',None) and not os.path.isabs(m.__file__)}
report['system_sitecustomize_note']='Ubuntu /etc/python3.12/sitecustomize.py installs apport hook if available; not a personal startup file. torch.classes and torch.ops expose synthetic relative __file__ labels; implementation is in the new venv.'
forbidden=['/home/lang/RL-AgiBot-X2','/home/lang/.local','/home/lang/.cache/hrs-x2-recovery']
report['forbidden_path_matches']=[p for p in report['loaded_module_files'].values() if any(x in p for x in forbidden)]
assert not report['forbidden_path_matches'],report['forbidden_path_matches']
assert sys.prefix==str(ws/'.venv')
assert report['package_prefix']==str(root/'install/x2_recovery')
assert all(s=='#!'+sys.executable for s in report['entrypoints'].values())
for name in ['x2_recovery.recovery_node','x2_recovery.telemetry_node']:
 assert report['modules'][name]['realpath'].startswith(str(ws)+'/' )
(root/'evidence/environment.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('loaded_module_files','source_sha256')},indent=2,allow_nan=False))
