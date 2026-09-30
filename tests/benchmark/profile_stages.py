"""Per-stage time of one evolution step, C++ AM3 (serial) vs am3_mojo
(4 threads), from AM3's built-in profiler (set_profile_timing /
get_profiled_times, which both implementations provide).

usage: python tests/benchmark/profile_stages.py [--scenario blazar_simple] [--steps 300]
Writes results/benchmark/stages_<scenario>.md.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from support.paths import available, results_dir  # noqa: E402

CHILD = r'''
import contextlib, io, json, sys
sys.path.insert(0, sys.argv[1])
from support.paths import load
from support.scenarios import SCENARIOS
mod = load(sys.argv[2]); a = mod.AM3(); n = int(sys.argv[4])
with contextlib.redirect_stdout(io.StringIO()):
    g = SCENARIOS[sys.argv[3]](a, n_steps=20, every=20); next(g)  # set up + warm up
    a.set_profile_timing(True)
    tot = [0.0] * 36
    for _ in range(n):
        a.evolve_step()
        for k, t in enumerate(a.get_profiled_times()):
            tot[k] += t
print(json.dumps(dict(labels=list(a.get_profiled_time_labels()), ms=[t / 1e6 / n for t in tot])))
'''


def profile(impl, scenario, steps, env):
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run([sys.executable, '-c', CHILD, os.path.dirname(HERE), impl, scenario, str(steps)],
                       capture_output=True, text=True, env=e, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='blazar_simple')
    ap.add_argument('--steps', type=int, default=300)
    a = ap.parse_args()
    if not available('cpp'):
        sys.exit('C++ reference not built: run `make reference`')
    cpp = profile('cpp', a.scenario, a.steps, {})
    moj = profile('mojo', a.scenario, a.steps, {'AM3_MOJO_THREADS': '4'})
    rows = sorted(zip(cpp['labels'], cpp['ms'], moj['ms']), key=lambda r: -r[1])
    path = os.path.join(results_dir('benchmark'), f'stages_{a.scenario}.md')
    with open(path, 'w') as f:
        f.write('| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |\n|---|---|---|---|\n')
        for label, c, m in rows:
            if c < 0.02 and m < 0.02:
                continue
            f.write(f'| {label} | {c:.3f} ms | {m:.3f} ms | {c / m if m > 0 else float("inf"):.1f}x |\n')
        f.write('| (stages below 0.02 ms in both codes omitted) | | | |\n')
        f.write(f'| **total of stages** | **{sum(cpp["ms"]):.2f} ms** | **{sum(moj["ms"]):.2f} ms** | '
                f'**{sum(cpp["ms"]) / sum(moj["ms"]):.1f}x** |\n')
    print(open(path).read())


if __name__ == '__main__':
    main()
