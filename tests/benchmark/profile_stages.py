"""Per-stage time of one evolution step, C++ AM3 (serial, with the fixes of
tests/reference/am3_fixes.patch) vs am3_mojo (4 threads), from AM3's built-in
profiler (set_profile_timing / get_profiled_times), which both codes provide.

usage: python tests/benchmark/profile_stages.py [--scenario blazar_simple] [--steps 300]
Writes results/benchmark/stages_<scenario>.md.

The profiler slots are nested. AM3's slots 0-20 are the stages of a step, and
21-35 are parts of slot 18 ("calc. coeff. photon"). Of am3_mojo's slots, 38
(neutron decay), 39 and 40 (the kaons) are stages. The others are parts: 36
(e+e- annihilation) of slot 15 ("calc. coeff. pairs"), 41 and 42 of slot 18;
37 (triplet pair production) sums parts of slots 15, 16 and 19. The total of
a step is the sum of the stages only.

The profiler averages --steps steps after 20 warm-up steps. benchmark.py's
time per step averages all steps, including the first, which takes 7-26 times
as long as a later one in both codes. So the two differ by up to 8 %; the rest
is run-to-run variation of 1-2 %, profiler overhead included. A few stages at
or below about 0.1 ms are slightly slower in am3_mojo: the cost of handing work
to the thread pool.
"""
import argparse, os, sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.paths import available, results_dir  # noqa: E402
from support.runs import profile_scenario  # noqa: E402

STAGES = list(range(21)) + [38, 39, 40]
PHOTON_COEFF, PHOTON_PARTS = 18, range(21, 36)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='blazar_simple')
    ap.add_argument('--steps', type=int, default=300)
    a = ap.parse_args()
    if not available('cpp-fixed'):
        sys.exit('C++ reference not built: run `make reference`')
    cpp = profile_scenario('cpp-fixed', a.scenario, a.steps)
    moj = profile_scenario('mojo-am3', a.scenario, a.steps, {'AM3_MOJO_THREADS': '4'})
    ms = lambda r, k: r['ms'][k] if k < len(r['ms']) else 0.0
    label = lambda k: moj['labels'][k]
    path = os.path.join(results_dir('benchmark'), f'stages_{a.scenario}.md')

    def row(f, k, name):
        c, m = ms(cpp, k), ms(moj, k)
        if c >= 0.02 or m >= 0.02:
            f.write(f'| {name} | {c:.3f} ms | {m:.3f} ms | {c / m if m > 0 else float("inf"):.1f}x |\n')

    with open(path, 'w') as f:
        f.write('| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |\n|---|---|---|---|\n')
        for k in sorted(STAGES, key=lambda k: -ms(cpp, k)):
            row(f, k, label(k))
            if k == PHOTON_COEFF:  # its parts, which its time includes
                for j in sorted(PHOTON_PARTS, key=lambda j: -ms(cpp, j)):
                    row(f, j, '&nbsp;&nbsp;&nbsp;↳ ' + label(j).replace('photon coeff: ', ''))
        f.write('| (stages below 0.02 ms in both codes omitted) | | | |\n')
        tc, tm = sum(ms(cpp, k) for k in STAGES), sum(ms(moj, k) for k in STAGES)
        f.write(f'| **total of a step** | **{tc:.2f} ms** | **{tm:.2f} ms** | **{tc / tm:.1f}x** |\n')
    print(open(path).read())


if __name__ == '__main__':
    main()
