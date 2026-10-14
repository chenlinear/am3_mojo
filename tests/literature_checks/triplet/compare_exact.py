"""Triplet pair production, README "Triplet pair production" (Checks against the literature): the code's total
cross section sigma(k) and the Monte Carlo nodes of its tables against the exact cross section of Lee, Lyubyakin &
Stotsky (eq. 4.4), and sigma(k) against Haug's numerical values (1975, Z. Naturforsch. 30a, 1099, Table 1: 58 values
for 4 < k <= 5000, to three or four digits). Haug's points on a node test the Monte Carlo, those between nodes the
interpolation. Also: the expansions against the tables, and where sigma(k) exceeds the Klein-Nishina cross section.

The code's sigma(k) (physics/triplet.mojo, through tests/harness/triplet_mojo.mojo) is Lee et al.'s threshold
expansion for k <= 4.02, the table nodes up to k = 1000 with a cubic spline of ln sigma in ln(k - 4), and the
high-energy expansion (rounded coefficients) above. derivations/triplet.py gives eq. 4.4 (in 30 digits), the
README's expansions, Haug's Table 1 and Klein-Nishina. It also compares the expansions and Haug's values with the
exact cross section and each other, and finds where the exact cross section exceeds Klein-Nishina's.

usage: compare_exact.py [--mc k ...]   (about 30 s; --mc: the QED Monte Carlo of tools/triplet at those k, about
30 s each, against the exact cross section, the code and Haug's values)"""
import os, sys
import numpy as np
import mpmath as mp
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.harness import Triplet  # noqa: E402
from support.paths import load_derivation, use_tool  # noqa: E402

derivation = load_derivation('triplet')
ALPHA = float(derivation.ALPHA)
K_LOW, K_ASYM = derivation.numeric_expansions()[-1]   # the ends of the expansions: k = 4.02 (the tables' first node), 1e3


def haug():
    """k and sigma_t / (alpha r_e^2) of Haug's Table 1."""
    rows = derivation.haug()
    return np.array([float(k) for k, _ in rows]), np.array([float(s) for _, s in rows])


def mc_sigma(k):
    """sigma(k) / (alpha r_e^2) from the QED Monte Carlo of tools/triplet."""
    import gvar
    import vegas
    use_tool('triplet')
    from produce import events
    gvar.ranseed(0)  # vegas draws from gvar's generator: reproducible results
    integ = vegas.Integrator(4 * [[0.0, 1.0]])
    f = vegas.batchintegrand(lambda u: events(np.asarray(u), k)[0])
    integ(f, nitn=10, neval=200000)
    return integ(f, nitn=10, neval=200000)


T = Triplet()
kh, sh = haug()
dense = [k for k in 4 + np.geomspace(K_LOW - 4, K_ASYM - 4, 400) if T.k[0] < k < K_ASYM]
mc_ks = [float(a) for a in sys.argv[sys.argv.index('--mc') + 1:]] if '--mc' in sys.argv else []
needed = sorted({float(k) for k in list(T.k) + list(kh) + dense + mc_ks})
with mp.workdps(derivation.DPS):   # eq. 4.4 at every k needed, in one pass
    EXACT = dict(zip(needed, (float(v) for v in derivation.lee_sigma([mp.mpf(repr(k)) for k in needed]))))
    low, high, _ = derivation.readme_expansions()
    thr = lambda k: low(mp.mpf(repr(float(k))))             # noqa: E731  (eq. 4.5)
    asym_rounded = lambda k: high(mp.mpf(repr(float(k))))   # noqa: E731  (eq. 4.13)
ex = lambda k: EXACT[float(k)]  # noqa: E731
print('=== Monte Carlo nodes vs exact')
dn = []
for k, s in zip(T.k, T.sig):
    e = ex(k)
    dn.append((k, s / e - 1))
dn = np.array(dn)
for k, d in dn:
    print(f'  k={k:<12.6g} MC/exact-1 {d:+.2e}')
print('  max |MC/exact-1| all nodes %.2e; k <= 10: %.2e; 10 < k <= 1e3: %.2e; k > 1e3: %.2e; rms(k>1e3) %.2e' % (
    np.max(np.abs(dn[:, 1])), np.max(np.abs(dn[dn[:, 0] <= 10, 1])), np.max(np.abs(dn[(dn[:, 0] > 10) & (dn[:, 0] <= K_ASYM), 1])),
    np.max(np.abs(dn[dn[:, 0] > K_ASYM, 1])), np.sqrt(np.mean(dn[dn[:, 0] > K_ASYM, 1] ** 2))))
print("=== code vs exact at Haug's energies")
code = T.sigma(kh)
rows = []
for k, c in zip(kh, code):
    e = ex(k)
    rows.append(c / e - 1)
    print(f'  k={k:<7g} code/exact-1 {c / e - 1:+.2e}')
print('  max |code/exact-1| at Haug points %.2e' % np.max(np.abs(rows)))
print("=== code sigma(k) vs Haug's Table 1")
node = np.array([np.any(np.isclose(x, T.k, rtol=1e-9)) for x in kh])
where = np.where(kh <= T.k[0], 'threshold exp.', np.where(kh >= K_ASYM, 'high-energy exp.',
                                                        np.where(node, 'node', 'between nodes')))
print('    k      Haug [alpha r_e^2]   code/Haug - 1   code uses')
for x, s, c, w in zip(kh, sh, code, where):
    print(f'{x:8g}   {s:16.6g}   {c / s - 1:+12.1e}     {w}')
for label, sel in [('on table nodes', (where == 'node')),
                   ('between table nodes', (where == 'between nodes'))]:
    r = code[sel] / sh[sel] - 1
    print(f'{label:34s} ({sel.sum():2d} points): code within {np.max(np.abs(r)):.1e} '
          f'(from {r.min():+.4f} at k = {kh[sel][np.argmin(r)]:g} to {r.max():+.4f} at k = {kh[sel][np.argmax(r)]:g})')
print('=== code sigma(k) vs exact on a dense grid, along the spline (4.02 < k < 1000)')
v = np.array([(k, c / ex(k) - 1) for k, c in zip(dense, T.sigma(dense))])
i = np.argmax(np.abs(v[:, 1]))
print(f'  spline: max |code/exact-1| = {abs(v[i, 1]):.2e} at k = {v[i, 0]:.5g}')
print('=== expansions vs the tables')
print('  threshold expansion / table at %g - 1: %+.2e' % (K_LOW, float(thr(K_LOW)) / T.sig[0] - 1))
hi = T.k >= K_ASYM
r = np.array([float(asym_rounded(k)) for k in T.k[hi]]) / T.sig[hi] - 1
print('  HE expansion / tables for k >= 1e3: max |.| %.2e (at k = %.4g); values %s' % (np.max(np.abs(r)), T.k[hi][np.argmax(np.abs(r))], np.array2string(r[:8], precision=1)))
print('=== Klein-Nishina crossing')
print('  the code sigma exceeds sigma_KN above k = ', end='')
lo, hi_ = 100.0, 1000.0
for _ in range(6):   # the bracket narrowed 32-fold per run of the harness
    ks = np.geomspace(lo, hi_, 33)
    above = T.sigma(ks) * ALPHA > np.array([float(derivation.kn_sigma(k)) for k in ks])   # both in r_e^2
    i = int(np.argmax(above))   # the first k where the code's sigma exceeds Klein-Nishina's
    lo, hi_ = ks[i - 1], ks[i]
print('%.4g' % np.sqrt(lo * hi_))
if '--mc' in sys.argv:
    print('=== Monte Carlo at the given k')
    for x in [float(a) for a in sys.argv[sys.argv.index('--mc') + 1:]]:
        res = mc_sigma(x)
        e = ex(x)
        c = T.sigma([x])[0]
        line = (f'k = {x:g}: Monte Carlo {res} (Q = {res.Q:.2f}); code {c:.6g}; Monte Carlo / exact - 1 = '
                f'{res.mean / e - 1:+.1e}, code / Monte Carlo - 1 = {c / res.mean - 1:+.1e}')
        if x in kh:
            ref = sh[list(kh).index(x)]
            line += f'; Monte Carlo / Haug - 1 = {res.mean / ref - 1:+.1e}, code / Haug - 1 = {c / ref - 1:+.1e}'
        print(line, flush=True)
