import datetime, json, os, pathlib, signal, subprocess, sys, time
ROOT=pathlib.Path(sys.argv[1]); label=sys.argv[2]; script=sys.argv[3]; phase=sys.argv[4] if len(sys.argv)>4 else 'clean'
E=ROOT/'evidence'; E.mkdir(exist_ok=True)
env=dict(HOME=str(ROOT/'home'),USER='lang',LOGNAME='lang',LANG='C.UTF-8',LC_ALL='C.UTF-8',PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',ACCEPT_ROOT=str(ROOT),WS=str(ROOT/'repository'),PYTHONNOUSERSITE='1',PIP_CONFIG_FILE='/dev/null')
setup='set -eo pipefail\n'
if phase in ('ros','installed'): setup+='source /opt/ros/jazzy/setup.bash\n'
if phase=='installed': setup+='source "$ACCEPT_ROOT/install/setup.bash"\n'
if phase in ('ros','installed'):
 setup+='export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"\nexport X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"\nexport ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n'
script_path=E/(label+'.sh'); script_path.write_text(setup+"PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x\n"+script+'\n')
log=E/(label+'.log'); start=datetime.datetime.now(datetime.timezone.utc).isoformat(); t=time.monotonic()
entry=dict(label=label,cwd='/tmp',argv=['/bin/bash','--noprofile','--norc',str(script_path)],script=str(script_path),script_text=script_path.read_text(),environment=env,start=start,log=str(log))
p=None
try:
 with log.open('x') as h:
  p=subprocess.Popen(entry['argv'],cwd='/tmp',env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  entry['pid']=p.pid
  entry['exit_code']=p.wait(timeout=900)
except BaseException as exc:
 entry['driver_error']=repr(exc)
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGINT)
  try: p.wait(timeout=15)
  except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
 entry['exit_code']=p.returncode if p else None
finally:
 entry['end']=datetime.datetime.now(datetime.timezone.utc).isoformat(); entry['wall_s']=time.monotonic()-t
 with (E/'commands.jsonl').open('a') as h: h.write(json.dumps(entry,allow_nan=False)+'\n')
 print(json.dumps({k:entry.get(k) for k in ['label','pid','exit_code','wall_s','log','driver_error']},allow_nan=False)); print(log.read_text()[-7000:])
sys.exit(entry['exit_code'] or 0)
