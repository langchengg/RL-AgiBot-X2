"""Thin published-controller entry: strict check, one copied-input run, or five evaluations.

No training, installation, alternate environment, or compatibility bypass is added.
Use absolute input/output paths from an arbitrary working directory.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from . import evaluate
from .model import require
from .train import sha256,write_json


def output_path(training_run, output):
    source=Path(training_run).expanduser().resolve()
    destination=Path(output).expanduser().resolve()
    require(source.is_dir(), 'Training input directory does not exist: '+str(source))
    require(not destination.exists(), 'Refusing existing output: '+str(destination))
    require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
            'Output and frozen input must be disjoint directories')
    return source,destination


def checked_inputs(source, expected):
    env,model,original,prepared=evaluate.prepare(source,expected_checkpoint_sha256=expected)
    try:
        require(prepared['saved'].get('schema')==evaluate.CONTROLLER_SCHEMA
                and prepared['saved']['controller']['mode']=='reference_residual',
                'This entry requires the complete reference-residual controller')
        evaluate.verify_policy(model,original)
        report=dict(status='PASS',scope='read-only loading; no reset, step, or training',
                    controller='reference_residual',training_run=str(source),
                    checkpoint_sha256=prepared['hashes']['policy_final.zip'],
                    input_sha256=prepared['hashes'],saved_action_check=prepared['consistency'],
                    observation_shape=list(env.observation_space.shape),
                    policy_action_shape=list(env.action_space.shape),
                    actuators=len(evaluate.physical_env(env).loaded.mapping),
                    model_asset_directory=prepared['model_asset_directory'],
                    source_identity=prepared['saved']['identity']['core_source_hashes'],
                    limitation='Historical simulation controller; standing success does not certify whole-episode constraints.')
        return report,dict(prepared['input_sources'])
    finally:
        env.close()


def execute(args):
    source,destination=output_path(args.training_run,args.output)
    if args.operation=='evaluate':
        evaluate.validate_seeds(args.seeds)
    require(0<=args.seed<2**32, 'Seed must be in [0,2**32)')
    require(0<args.episode_wall_seconds<=300, 'Episode wall budget must be in (0,300] seconds')
    report,sources=checked_inputs(source,args.expected_checkpoint_sha256)
    destination.mkdir(parents=True,exist_ok=False)
    write_json(destination/'check.json',report)
    if args.operation=='check':
        print(evaluate.encoded(report))
        return 0
    if args.operation=='run':
        copied=destination/'training-input'
        copied.mkdir()
        for name,path in sources.items():
            shutil.copyfile(path,copied/name)
            require(sha256(copied/name)==report['input_sha256'][name], 'Input copy mismatch: '+name)
        command=[sys.executable,'-m','x2_recovery.train','control-reload',
                 '--run-dir',str(copied),'--seed',str(args.seed),
                 '--max-wall-seconds',str(args.episode_wall_seconds)]
        result_path=copied/f'development-{args.seed}'/'summary.json'
    else:
        command=[sys.executable,'-m','x2_recovery.evaluate','--training-run',str(source),
                 '--expected-checkpoint-sha256',args.expected_checkpoint_sha256,
                 '--seeds',*[str(seed) for seed in args.seeds],'--deterministic',
                 '--output',str(destination/'evaluation'),
                 '--episode-wall-seconds',str(args.episode_wall_seconds)]
        result_path=destination/'evaluation'/'summary.json'
    # Lists preserve argument boundaries; inherited installed environment is intentional.
    with (destination/'stdout.log').open('x') as stdout,(destination/'stderr.log').open('x') as stderr:
        completed=subprocess.run(command,stdout=stdout,stderr=stderr,check=False)
    receipt=dict(controller=report['controller'],source=report['training_run'],command=command,
                 returncode=completed.returncode,result=str(result_path),trained=False)
    if result_path.is_file():
        result=json.loads(result_path.read_text())
        receipt.update(status=result.get('status'),success=result.get('success'),reason=result.get('reason'))
    write_json(destination/'execution.json',receipt)
    print(evaluate.encoded(receipt))
    if completed.returncode:
        print('Controller execution failed; see '+str(destination/'stderr.log'),file=sys.stderr)
    return completed.returncode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('check','run','evaluate'))
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--seed',type=int,default=221030)
    parser.add_argument('--seeds',type=int,nargs=5)
    parser.add_argument('--episode-wall-seconds',type=float,default=180.)
    args=parser.parse_args(argv)
    if args.operation=='evaluate' and args.seeds is None:
        parser.error('evaluate requires exactly five explicit --seeds')
    if args.operation!='evaluate' and args.seeds is not None:
        parser.error('--seeds is only for evaluate; use --seed for run')
    try:
        return execute(args)
    except (ValueError,OSError) as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
