"""Run the 20³–100³ Julia and Python sweeps with per-size wall limits.

The script owns logs, timeouts, and a completion marker. Timings are inside
the workers; process startup and compilation are recorded only as wall time.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(command, log, timeout, env):
    start = perf_counter()
    with log.open('w') as output:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            status = 'ok' if code == 0 else 'error'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
            status = 'timeout'
    return dict(status=status, wall_seconds=perf_counter()-start,
                returncode=code, log=str(log))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--sizes', nargs='+', type=int, default=[20, 40, 60, 80, 100])
    p.add_argument('--julia-timeout', type=float, default=120)
    p.add_argument('--python-timeout', type=float, default=60)
    p.add_argument('--dense-memory-mib', type=float, default=1024)
    p.add_argument('--chisel', choices=('universal', 'adjoint'), default='universal')
    p.add_argument('--julia-methods', help='comma-separated Julia methods to run')
    p.add_argument('--languages', default='python,julia',
                   help='comma-separated python,julia (default both)')
    p.add_argument('--python-warm-target', action='store_true')
    p.add_argument('--dimension-only', action='store_true',
                   help='run reduction and dense svdvals, skipping solvers')
    args = p.parse_args()
    if any(n < 2 or n % 4 for n in args.sizes):
        p.error('sizes must be positive multiples of four')
    languages = args.languages.split(',')
    if not languages or any(language not in ('python', 'julia') for language in languages):
        p.error('languages must be python,julia or one of them')
    args.folder.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', JULIA_NUM_THREADS='1')
    with (args.folder / 'inputs.log').open('w') as output:
        subprocess.run([sys.executable, str(HERE / 'large_sweep_inputs.py'),
                        str(args.folder), '--sizes', *map(str, args.sizes)],
                       cwd=ROOT, env=environment, stdout=output, check=True)
    manifest = args.folder / 'manifest.jsonl'
    manifest.write_text('')
    for n in args.sizes:
        for language in languages:
            log = args.folder / f'{language}-{n}-{args.chisel}.log'
            if language == 'python':
                command = [sys.executable, str(HERE / 'large_sweep.py'),
                           str(args.folder), '--sizes', str(n),
                           '--timeout', str(args.python_timeout),
                           '--chisel', args.chisel,
                           '--dense-memory-mib', str(args.dense_memory_mib)]
                if args.python_warm_target:
                    command.append('--warm-target')
                if args.dimension_only:
                    command.append('--dimension-only')
                timeout = args.python_timeout + 10
            else:
                command = ['julia', '--compiled-modules=existing', '--project=.',
                           str(HERE / 'large_sweep.jl'), str(args.folder), str(n),
                           str(args.dense_memory_mib), args.chisel]
                if args.dimension_only:
                    command.append('none')
                elif args.julia_methods:
                    command.append(args.julia_methods)
                timeout = args.julia_timeout
            print(f'RUN {language} n={n} chisel={args.chisel}', flush=True)
            result = run(command, log, timeout, environment)
            with manifest.open('a') as output:
                output.write(json.dumps(dict(language=language, size=n,
                                             chisel=args.chisel, **result)) + '\n')
            print(f'{result["status"].upper()} {language} n={n} chisel={args.chisel} '
                  f'wall={result["wall_seconds"]:.1f}s log={log}', flush=True)
    (args.folder / 'COMPLETE').write_text('large sweep finished\n')
    print(f'DONE {manifest}', flush=True)


if __name__ == '__main__':
    main()
