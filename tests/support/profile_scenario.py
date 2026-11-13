"""Profile one scenario with one implementation (run in its own process): AM3's
built-in profiler (set_profile_timing / get_profiled_times) over --steps steps
after 20 warm-up steps.

usage: python tests/support/profile_scenario.py --impl IMPL --scenario NAME
       --steps N
prints one JSON line: {"labels": [...], "ms": [...]} (ms per step of every
profiler slot)
"""

import argparse, json, os, sys

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.output import silenced
from support.paths import new_am3
from support.scenarios import SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--steps", type=int, required=True)
    a = ap.parse_args()
    obj, n = new_am3(a.impl), a.steps
    with silenced():
        next(SCENARIOS[a.scenario](obj, n_steps=20, every=20))
        obj.set_profile_timing(True)
        tot = [0.0] * len(obj.get_profiled_time_labels())
        for _ in range(n):
            obj.evolve_step()
            for k, t in enumerate(obj.get_profiled_times()):
                tot[k] += t
    print(
        json.dumps(
            dict(
                labels=list(obj.get_profiled_time_labels()),
                ms=[t / 1e6 / n for t in tot],
            )
        )
    )


if __name__ == "__main__":
    main()
