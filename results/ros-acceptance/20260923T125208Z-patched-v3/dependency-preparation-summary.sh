set -eo pipefail
source /opt/ros/jazzy/setup.bash
export PY="$WS/.venv/bin/python" X2_ASSET_REPO="$ACCEPT_ROOT/assets/agibot_x2_urdf"
export X2_SCENE="$X2_ASSET_REPO/X2_URDF-v1.3.0/scene.xml"
export ROS_DOMAIN_ID=86 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
"$PY" - <<'PYCODE'
import datetime,hashlib,json,os
from pathlib import Path
root=Path(os.environ['ACCEPT_ROOT']);evidence=root/'evidence'
labels={'source-clone-attempt-1','source-checkout-attempt-2','model-clone-attempt-1','venv-and-torch-attempt-1','requirements-check-freeze-attempt-1','requirements-check-freeze-attempt-2','mujoco-metadata-and-pip-resume-help','mujoco-http-head-range-probe','mujoco-curl-download-attempt-1','verified-wheel-and-requirements-attempt-3','dependency-module-origins'}
commands=[json.loads(line) for line in (evidence/'commands.jsonl').read_text().splitlines()]
commands=[c for c in commands if c['label'] in labels]
files=['readme-applied.patch','python-packages.txt','pyvenv.cfg','dependency-module-origins.json','mujoco-pypi-metadata.json','mujoco-download-identity.json','mujoco-download-verification.json','mujoco-official-head.txt','mujoco-range-headers.txt','pip-download-help.txt']
files += [c['label']+'.log' for c in commands] + [c['label']+'.sh' for c in commands]
report=dict(status='DEPENDENCY_PREPARATION_COMPLETE',recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),
 scope='New GitHub clone, new project system-site-packages venv, independent fixed model download and clean driver shells. This subtask did not build, launch ROS, load a model or run physics/tests.',
 base_commit='71037f7d5530a65da220345f9037ac4d29f32908',model_commit='60c5de582c523cd188f563819e62d34cfdc3d2d0',
 pip_check='No broken requirements found.',python_packages_path=str(evidence/'python-packages.txt'),
 preparation_stages=[{key:c.get(key) for key in ('label','start','end','wall_s','exit_code','pid','log')} for c in commands],
 failures_preserved=[dict(stage='source-clone-attempt-1',exit_code=128,reason='TLS failure while lazily fetching a checkout blob; explicit checkout attempt 2 succeeded.'),dict(stage='requirements-check-freeze-attempt-1',exit_code=1,reason='MuJoCo wheel received only approximately 3.8/19.1MB; pip rejected hash e5c1246f6a4213a0b80dc51af89194f034a07e93210934870921d8c460998eef instead of ce292cdf888dcadf1d1440c3debfdd38519ea01d8e4376f86d1e7754bfe0b069.'),dict(stage='requirements-check-freeze-attempt-2',exit_code=1,reason='Independent no-cache pip retry reproduced the same rejected incomplete MuJoCo bytes.')],
 transport_recovery=dict(pip_version='24.0',pip_resume_option_available=False,curl_version='8.5.0',
 official_metadata_url='https://pypi.org/pypi/mujoco/3.13.0/json',
 independent_curl_download=True,used_old_cache=False,used_proxy_or_certificate_exception=False,used_query_parameter=False,
 verified=json.loads((evidence/'mujoco-download-verification.json').read_text()),
 note='Same pinned official wheel, full size and official SHA verified before local installation; unchanged requirements then installed from PyPI and pip check passed. Download transport recovery, not a dependency version or hash exemption.'),
 modules=json.loads((evidence/'dependency-module-origins.json').read_text())['modules'],
 system_packages_modified=False,
 artifacts_sha256={name:hashlib.sha256((evidence/name).read_bytes()).hexdigest() for name in files},
 publication_exclusion='Do not publish root/downloads wheel/range files, model assets or .venv; only evidence metadata/logs are needed.')
path=evidence/'dependency-preparation-summary.json'
path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(path);print(hashlib.sha256(path.read_bytes()).hexdigest())
PYCODE
