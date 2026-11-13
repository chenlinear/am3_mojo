"""Time one scenario with one implementation (run in its own process).

usage: python tests/support/time_scenario.py --impl IMPL --scenario NAME
       [--repeats R]
prints one JSON line:
{"init": [...], "evolve": [...], "steps": n, "max_rss": bytes}
"""

import argparse, gc, json, os, resource, sys, time

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from support.output import silenced
from support.paths import new_am3
from support.scenarios import SCENARIOS


class Timed:
    """The set-up is init_kernels, set_mag_field before the first of the steps,
    and the first evolve_step after init_kernels. The C++ builds the kernels
    that depend on the field in set_mag_field. am3_mojo builds them in that
    step, where it also packs the Bethe-Heitler kernel."""

    def __init__(self, obj):
        self._o, self.t_init, self.t_evolve, self.steps = obj, 0.0, 0.0, 0
        self.first = False

    def __getattr__(self, name):
        return getattr(self._o, name)

    def init_kernels(self):
        t = time.perf_counter()
        self._o.init_kernels()
        self.t_init += time.perf_counter() - t
        self.first = True

    def set_mag_field(self, b):
        t = time.perf_counter()
        self._o.set_mag_field(b)
        if self.steps == 0:
            self.t_init += time.perf_counter() - t
        else:
            self.t_evolve += time.perf_counter() - t

    def evolve_step(self):
        t = time.perf_counter()
        self._o.evolve_step()
        if self.first:
            self.t_init += time.perf_counter() - t
            self.first = False
        else:
            self.t_evolve += time.perf_counter() - t
            self.steps += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    res = {"init": [], "evolve": [], "steps": 0}
    for _ in range(a.repeats):
        obj = Timed(new_am3(a.impl))
        with silenced():
            for _ in SCENARIOS[a.scenario](obj):
                pass
        res["init"].append(obj.t_init)
        res["evolve"].append(obj.t_evolve)
        res["steps"] = obj.steps
        # free this simulation before the next is built, so the peak memory is
        # one simulation's
        obj = None
        gc.collect()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes (Linux reports KiB)
    res["max_rss"] = rss if sys.platform == "darwin" else rss * 1024
    print(json.dumps(res))


if __name__ == "__main__":
    main()
