"""Benchmark am3_mojo against the C++ AM3 (serial build) on the scenarios of
tests/support/scenarios.py. The C++ is the original with the fixes of
tests/reference/am3_fixes.patch (`cpp-fixed`), so both codes solve the same
equations and do the same work.

usage: python tests/benchmark/benchmark.py [--repeats 3] [--scenarios a,b,...]

Each (implementation, scenario) pair runs in a fresh process. Reported: the
minimum over the repeats of the kernel initialisation (init_kernels) and of
the time per evolution step, and the process's peak memory. Against the C++,
am3_mojo runs without its added processes (`mojo-am3`, the same physics); the
last column has them on (its default). Results: results/benchmark/benchmark.json
and benchmark.md.
"""
import argparse, json, os, sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.paths import available, results_dir  # noqa: E402
from support.runs import time_scenario  # noqa: E402
from support.scenarios import SCENARIOS  # noqa: E402

RUNS = [  # label, implementation, environment
    ('C++ AM3 (serial)', 'cpp-fixed', {}),
    ('am3_mojo, 4 threads', 'mojo-am3', {'AM3_MOJO_THREADS': '4'}),
    ('am3_mojo, 1 thread', 'mojo-am3', {'AM3_MOJO_THREADS': '1'}),
    ('am3_mojo + added processes, 4 threads', 'mojo', {'AM3_MOJO_THREADS': '4'}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--scenarios', default=','.join(SCENARIOS))
    a = ap.parse_args()
    runs = [r for r in RUNS if available(r[1])]
    if not any(r[1] == 'cpp-fixed' for r in runs):
        print('C++ reference not built (make reference): benchmarking am3_mojo only')
    res = {}
    for sc in a.scenarios.split(','):
        for label, impl, env in runs:
            d = time_scenario(impl, sc, a.repeats, env)
            x = res.setdefault(sc, {})[label] = dict(
                init_s=min(d['init']), ms_per_step=1e3 * min(d['evolve']) / d['steps'],
                steps=d['steps'], max_rss_gb=d['max_rss'] / 1e9)
            print(f"{sc:18s} {label:38s} init {x['init_s']:7.3f} s   {x['ms_per_step']:8.3f} ms/step"
                  f"   peak memory {x['max_rss_gb']:.2f} GB", flush=True)
    out = results_dir('benchmark')
    json.dump(res, open(os.path.join(out, 'benchmark.json'), 'w'), indent=1)
    labels = [r[0] for r in runs]
    with open(os.path.join(out, 'benchmark.md'), 'w') as f:
        f.write('| Scenario | ' + ' | '.join(labels) + ' | Speed-up (4 threads vs C++) |\n')
        f.write('|---' * (len(labels) + 2) + '|\n')
        for sc, r in res.items():
            cells = [f"{r[l]['ms_per_step']:.2f} ms ({r[l]['init_s']:.2f} s, {r[l]['max_rss_gb']:.2f} GB)" for l in labels]
            sp = (r['C++ AM3 (serial)']['ms_per_step'] / r['am3_mojo, 4 threads']['ms_per_step']
                  if 'C++ AM3 (serial)' in r else float('nan'))
            f.write(f'| {sc} | ' + ' | '.join(cells) + f' | {sp:.1f}x |\n')
        f.write('\nms per evolution step (kernel initialisation, peak memory of one simulation); minimum of '
                f'{a.repeats} runs in one fresh process per implementation and scenario.\n')
    print(f'\nwritten {out}/benchmark.md')


if __name__ == '__main__':
    main()
