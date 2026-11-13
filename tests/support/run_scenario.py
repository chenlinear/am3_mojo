"""Run one scenario with one implementation, and save every getter at every
checkpoint.

usage: python tests/support/run_scenario.py --impl mojo|mojo-am3|cpp|cpp-fixed
           --scenario NAME --out FILE.npz [--steps N --every M]
"""

import argparse, os, sys, time

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.paths import new_am3
from support.runs import save_npz
from support.scenarios import SCENARIOS, dump_state


def whole_number(text):
    """Without the limit of 1, --every 0 counts as no option, and the other
    counts below 1 make scenarios.evolve raise ValueError or run no step."""
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not 1 or more")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=whole_number)
    ap.add_argument("--every", type=whole_number)
    a = ap.parse_args()
    # Without this check, --every alone is dropped for the scenario's own
    # checkpoints.
    if a.every is not None and a.steps is None:
        ap.error("--every needs --steps")
    kw = {}
    if a.steps is not None:
        kw = dict(n_steps=a.steps, every=a.every or a.steps)
    obj, res = new_am3(a.impl), {}
    t0 = time.perf_counter()
    for step in SCENARIOS[a.scenario](obj, **kw):
        for k, v in dump_state(obj).items():
            res[f"{step:06d}/{k}"] = v
    save_npz(a.out, **res)
    print(
        f"{a.scenario} [{a.impl}]: {time.perf_counter() - t0:.2f} s, "
        f"{len(res)} arrays -> {a.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
