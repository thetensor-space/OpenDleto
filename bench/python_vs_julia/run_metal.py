"""Run Julia Metal one size per process with durable, resumable result logs."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, default=[8,16,24,32,48,64,96,128,192,256,384,512])
    parser.add_argument('--julia', default='julia')
    parser.add_argument('--project', default='.', help='Julia project with Dleto and Metal')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    script = Path(__file__).resolve().with_name('compare_apple_metal.jl')
    for n in args.sizes:
        checkpoint = args.input / f'metal_{n}.log'
        if args.resume and checkpoint.exists() and 'RESULT n=' in checkpoint.read_text():
            print(f'SKIP n={n} checkpoint={checkpoint}', flush=True)
            continue
        partial = checkpoint.with_suffix('.log.partial')
        print(f'START n={n} log={partial}', flush=True)
        with partial.open('w') as log:
            process = subprocess.run([args.julia, '--compiled-modules=existing', f'--project={args.project}', str(script), str(args.input), str(n)], stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise SystemExit(f'Metal n={n} failed with exit {process.returncode}; see {partial}')
        if 'RESULT n=' not in partial.read_text():
            raise SystemExit(f'Metal n={n} produced no result; see {partial}')
        os.replace(partial, checkpoint)
        print(checkpoint.read_text(), flush=True)


if __name__ == '__main__':
    main()
