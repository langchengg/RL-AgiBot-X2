import datetime,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);e=root/'evidence';ws=root/'repository'
files=['README.md','requirements.txt','src/x2_recovery/package.xml','src/x2_recovery/setup.py','src/x2_recovery/setup.cfg','src/x2_recovery/launch/recovery.launch.py']
files.extend(str(p.relative_to(ws)) for p in (ws/'src/x2_recovery/x2_recovery').glob('*.py'))
files.extend(str(p.relative_to(ws)) for p in (ws/'src/x2_recovery/test').glob('*.py'))
rows={}
for f in files:
 p=ws/f;current=hashlib.sha256(p.read_bytes()).hexdigest()
 base=hashlib.sha256(subprocess.check_output(['git','show','71037f7d5530a65da220345f9037ac4d29f32908:'+f],cwd=ws)).hexdigest()
 rows[f]=dict(sha256=current,base_sha256=base,identical_to_release=current==base)
report=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit='71037f7d5530a65da220345f9037ac4d29f32908',files=rows)
(e/'source-identity.json').write_text(json.dumps(report,indent=2)+'\n')
print('Files checked:',len(rows),'Changes:',[f for f,r in rows.items() if not r['identical_to_release']])
# Exact apt list for packages owning actual inherited module files.
env=json.loads((e/'environment.json').read_text())
origins={}
for name,row in env['modules'].items():
 path=row['realpath']
 if path.startswith(('/usr/','/opt/ros/')):
  p=subprocess.run(['dpkg-query','-S',path],capture_output=True,text=True)
  origins[name]=dict(path=path,exit_code=p.returncode,owning_package=p.stdout.strip(),stderr=p.stderr.strip())
(e/'module-system-package-owners.json').write_text(json.dumps(origins,indent=2)+'\n')
# Source evidence for the observed original driver's double-SIGINT defect.
refs={}
for file,lo,hi in [('/opt/ros/jazzy/lib/python3.12/site-packages/launch/actions/execute_local.py',438,449),('/opt/ros/jazzy/lib/python3.12/site-packages/ros2launch/command/launch.py',71,80)]:
 path=pathlib.Path(file)
 refs[file]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),first_line=lo,last_line=hi,excerpt='\n'.join(path.read_text().splitlines()[lo-1:hi]))
(e/'launch-signal-semantics.json').write_text(json.dumps(refs,indent=2)+'\n')
