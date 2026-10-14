"""Time one scenario with one implementation (run in its own process).

usage: python tests/support/time_scenario.py --impl IMPL --scenario NAME [--repeats R]
prints one JSON line: {"init": [...], "evolve": [...], "steps": n, "max_rss": bytes}
"""
import argparse, gc, json, os, resource, sys, time

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tests/
from support.output import silenced  # noqa: E402
from support.paths import new_am3  # noqa: E402
from support.scenarios import SCENARIOS  # noqa: E402


class Timed:
    """Forwarding proxy that accumulates the time spent in init_kernels and
    evolve_step."""

    def __init__(self, obj):
        object.__setattr__(self, '_o', obj)
        object.__setattr__(self, 't_init', 0.0)
        object.__setattr__(self, 't_evolve', 0.0)
        object.__setattr__(self, 'steps', 0)

    def __getattr__(self, name):
        attr = getattr(self._o, name)
        if name == 'init_kernels':
            def f():
                t = time.perf_counter(); attr()
                object.__setattr__(self, 't_init', self.t_init + time.perf_counter() - t)
            return f
        if name == 'evolve_step':
            def g():
                t = time.perf_counter(); attr()
                object.__setattr__(self, 't_evolve', self.t_evolve + time.perf_counter() - t)
                object.__setattr__(self, 'steps', self.steps + 1)
            return g
        return attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', required=True)
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--repeats', type=int, default=1)
    a = ap.parse_args()
    res = {'init': [], 'evolve': [], 'steps': 0}
    for _ in range(a.repeats):
        obj = Timed(new_am3(a.impl))
        with silenced():
            for _ in SCENARIOS[a.scenario](obj):
                pass
        res['init'].append(obj.t_init)
        res['evolve'].append(obj.t_evolve)
        res['steps'] = obj.steps
        # release this simulation before the next one is built, so that the
        # peak memory is that of one simulation
        obj = None
        gc.collect()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res['max_rss'] = rss if sys.platform == 'darwin' else rss * 1024  # bytes (Linux reports KiB)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
