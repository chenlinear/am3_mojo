"""Run every script of tests/literature_checks/ and tests/solver_checks/, and tools/ky_angular_extract.py, with the
arguments that give the README's figures (`make checks`).

usage: .venv/bin/python tests/run_checks.py [-j N] [--out DIR] [name ...]
Each script's output goes to DIR/<name>.txt (default build/checks/). The exit code and run time of every script
are printed and written to DIR/summary.txt. With -j 6 (the default) a run takes about 5 minutes on an Apple M5,
the time of its longest script, triplet/machinery.py --high-k. The two runs of regime_boundary.py with the C++
AM3 are skipped without `make reference`. Exit status 1 if a script failed.
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

from support.paths import ROOT, available

L = 'tests/literature_checks/'
S = 'tests/solver_checks/'
# name, script and arguments, implementation it needs (longest first, for the parallel runs)
SCRIPTS = [
    ('triplet_machinery', [L + 'triplet/machinery.py', '--high-k'], None),
    ('triplet_x_scaling', [L + 'triplet/x_scaling.py'], None),
    ('triplet_compare_exact', [L + 'triplet/compare_exact.py', '--mc', '4.03', '7', '10', '12', '45'], None),
    ('triplet_below_grid', [S + 'triplet_below_grid.py'], None),
    ('derivations', [L + 'derivations/run.py'], None),
    ('triplet_inelasticity', [L + 'triplet/inelasticity.py'], None),
    ('triplet_extrapolation', [L + 'triplet/extrapolation.py'], None),
    ('kaon_production_two_body', [L + 'kaon_production/two_body.py'], None),
    ('kaon_production_multibody_strangeness', [L + 'kaon_production/multibody_strangeness.py'], None),
    ('kaon_production_m4_steps', [L + 'kaon_production/m4_steps.py'], None),
    ('kaon_production_ky_angular_extract', ['tools/ky_angular_extract.py'], None),
    ('derivations_inventory', [L + 'derivations/inventory.py'], None),
    ('regime_boundary', [S + 'regime_boundary.py'], None),
    ('regime_boundary_cpp_fixed', [S + 'regime_boundary.py', '--impl', 'cpp-fixed'], 'cpp-fixed'),
    ('regime_boundary_cpp', [S + 'regime_boundary.py', '--impl', 'cpp'], 'cpp'),
    ('window_top_undershoot', [S + 'window_top_undershoot.py'], None),
    ('triplet_steady_states', [S + 'triplet_steady_states.py'], None),
]


def run(name, args, out):
    t0 = time.time()
    with open(os.path.join(out, name + '.txt'), 'w') as f:
        f.write('$ ' + ' '.join(args) + '\n')
        f.flush()
        rc = subprocess.run([sys.executable] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    line = f'{name:40s} {"ok" if rc == 0 else f"FAILED (exit {rc})":16s} {time.time() - t0:7.1f} s'
    print(line, flush=True)
    return line, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=6, help='scripts run in parallel')
    ap.add_argument('--out', default=os.path.join(ROOT, 'build', 'checks'))
    ap.add_argument('names', nargs='*', help='run only these (default: all)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo, lines = [], []
    for name, args, impl in SCRIPTS:
        if a.names and name not in a.names:
            continue
        if impl and not available(impl):
            lines.append(f'{name:40s} skipped (needs `make reference`)')
            print(lines[-1])
            continue
        todo.append((name, args))
    with ThreadPoolExecutor(a.j) as ex:
        res = list(ex.map(lambda t: run(*t, a.out), todo))
    lines += [line for line, _ in res]
    with open(os.path.join(a.out, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    failed = [line for line, rc in res if rc != 0]
    print(f'\n{len(res) - len(failed)} of {len(res)} scripts ran without error; outputs in {a.out}/')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
