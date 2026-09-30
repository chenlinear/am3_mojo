"""Run one scenario with one implementation and save every getter at every
checkpoint.

usage: python tests/support/run_scenario.py --impl mojo|cpp|cpp-oobfix
           --scenario NAME --out FILE.npz [--steps N --every M]
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.paths import load  # noqa: E402
from support.scenarios import SCENARIOS, dump_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True)
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--every', type=int)
    a = ap.parse_args()
    mod = load(a.impl)
    kw = {}
    if a.steps:
        kw = dict(n_steps=a.steps, every=a.every or a.steps)
    obj = mod.AM3()
    res = {}
    t0 = time.perf_counter()
    for step in SCENARIOS[a.scenario](obj, **kw):
        for k, v in dump_state(obj).items():
            res[f'{step:06d}/{k}'] = v
    np.savez_compressed(a.out, **res)
    print(f'{a.scenario} [{a.impl}]: {time.perf_counter() - t0:.2f} s, {len(res)} arrays -> {a.out}',
          file=sys.stderr)


if __name__ == '__main__':
    main()
