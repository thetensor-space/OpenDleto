#!/usr/bin/env python3
"""Own the numerical job, its timeout, streamed log, and completion notification.
Run this in a terminal; the model need not remain active to monitor it.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / 'bench/reports/symmetric-sphere'
out.mkdir(parents=True, exist_ok=True)
log = out / 'comparison.log'
(out / 'completion.txt').write_text('running\n')
limit = int(os.environ.get('SPHERE_TIMEOUT_SECONDS', '900'))
cmd = [os.environ.get('JULIA', 'julia'), '--project=' + str(root), '--threads=4', str(root / 'bench/SymmetricSphereSweep.jl'), str(out)]
print('Running:', ' '.join(cmd), '\nLog:', log, flush=True)
with log.open('w', buffering=1) as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=root, start_new_session=True)
    try:
        code = p.wait(timeout=limit)
        status = 'complete' if code == 0 else f'failed (exit {code})'
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
        import signal
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        code = 124 if isinstance(e, subprocess.TimeoutExpired) else 130
        status = f'timed out after {limit}s' if code == 124 else 'interrupted'
    f.write('\nJOB ' + status + '\n')
(out / 'completion.txt').write_text(status + '\n')
print('\aSphere sweep ' + status + '. Results: ' + str(out), flush=True)
if sys.platform == 'darwin':
    subprocess.run(['osascript', '-e', 'display notification "Sphere sweep finished; see terminal and results log." with title "OpenDleto"'], capture_output=True)
sys.exit(code)
