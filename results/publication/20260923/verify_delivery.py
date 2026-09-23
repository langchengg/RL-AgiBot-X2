"""Read-only publication audit; restore trajectory gzip files as in README first.

Run with the project venv. Each historical policy is loaded in a separate process
with its own source identity. Reset, stepping, learning and optimizer updates are
explicitly forbidden. This audits saved experiments, not new recovery attempts.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


LOAD_CHECK = r'''
import json, sys
from pathlib import Path
from unittest.mock import patch
import torch
from stable_baselines3 import PPO
import x2_recovery.evaluate as evaluation
from x2_recovery.env import X2RecoveryEnv
from x2_recovery.train import json_value
directory, expected_source = (Path(value).resolve() for value in sys.argv[1:])
module = Path(evaluation.__file__).resolve()
if not module.is_relative_to(expected_source):
    raise ValueError('Wrong module path: ' + str(module))
manifest = json.loads((directory/'manifest.json').read_text())
calls = dict(reset=0, step=0, learn=0, train=0, optimizer=0)
def forbidden(name):
    def fail(*args, **kwargs):
        calls[name] += 1
        raise RuntimeError('Forbidden during publication audit: ' + name)
    return fail
with patch.object(X2RecoveryEnv, 'reset', forbidden('reset')), \
     patch.object(X2RecoveryEnv, 'step', forbidden('step')), \
     patch.object(PPO, 'learn', forbidden('learn')), \
     patch.object(PPO, 'train', forbidden('train')), \
     patch.object(torch.optim.Adam, 'step', forbidden('optimizer')):
    options = {} if directory.name.startswith('ppo-smoke-') else dict(
        expected_checkpoint_sha256=manifest['input_sha256']['policy_final.zip'])
    env, model, original, prepared = evaluation.prepare(
        directory/'inputs/training-run', **options)
    try:
        policy_hash = evaluation.verify_policy(model, original)
        if policy_hash != manifest['initial_policy_state_sha256']:
            raise ValueError('Loaded policy state differs from frozen evaluation')
        result = dict(module=str(module), python=sys.executable,
                      action_consistency=json_value(prepared['consistency']),
                      policy_state_sha256=policy_hash, forbidden_calls=calls,
                      saved_batch_audit=evaluation.audit_saved(directory))
    finally:
        env.close()
print(json.dumps(result, allow_nan=False))
'''


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parents[3]
    publication = json.loads(Path(__file__).with_name('manifest.json').read_text())
    for entry in publication['files']:
        path = root / entry['path']
        if path.is_symlink() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Published file identity mismatch: ' + entry['path'])
    for entry in publication['compressed_trajectories']:
        with gzip.open(root / entry['published'], 'rb') as stream:
            restored = hashlib.file_digest(stream, 'sha256').hexdigest()
        if restored != entry['original_sha256']:
            raise ValueError('Gzip roundtrip mismatch: ' + entry['published'])
    reports = {}
    for name in publication['evaluation_runs']:
        directory = root / 'results/evaluation' / name
        source = directory / 'source' if name.startswith('ppo-smoke-') else root / 'src/x2_recovery'
        environment = dict(os.environ, PYTHONPATH=str(source))
        child = subprocess.run([sys.executable, '-c', LOAD_CHECK, str(directory), str(source)],
                               cwd=root, env=environment, text=True, capture_output=True, timeout=180)
        if child.returncode:
            raise RuntimeError(f'{name}: exit {child.returncode}\n{child.stderr}\n{child.stdout}')
        reports[name] = json.loads(child.stdout)
    print(json.dumps(dict(status='PASS', root=str(root), checked_files=len(publication['files']),
                          batches=reports, scope='Saved evidence and inference; no new episodes'),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
