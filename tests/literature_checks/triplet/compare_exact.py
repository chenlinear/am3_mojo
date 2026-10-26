"""Triplet pair production (tests/literature_checks/triplet/README.md, "Cross section": "The exact result" and "In the
code"): the code's total cross section sigma(k) and its tables' Monte Carlo nodes against the exact cross
section of Lee, Lyubyakin & Stotsky (eq. 4.4, with the factor 2 in eq. 4.3's J_11 that the README derives), and
sigma(k) against Haug's numerical values (1975, Z. Naturforsch. 30a, 1099, Table 1: 58 values for
4 < k <= 5000, to three or four digits). Haug's points on a node test the Monte Carlo, those between nodes the
interpolation. Also: the code's expansions against the exact cross section at Haug's k, and the expansions against
the tables.

The code's sigma(k) (physics/triplet.mojo, via tests/harness/triplet_mojo.mojo) is Lee et al.'s threshold expansion
for k <= 4.02, the table nodes up to k = 1000 with a cubic spline of ln sigma in ln(k - 4), and the high-energy
expansion (rounded coefficients) above. Eq. 4.4 (in 30 digits, lee_sigma) and Haug's Table 1 (haug) are below;
triplet/triplet.py gives the README's expansions.

usage: compare_exact.py [--mc k ...]   (about 30 s; --mc: tools/triplet's QED Monte Carlo at those k, about 30 s
each, against the exact cross section, the code and Haug's values)"""
import functools, os, sys
import numpy as np
import mpmath as mp
import sympy as sp
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.harness import Triplet
from support.paths import load_derivation, use_tool

derivation = load_derivation('triplet')              # the README's expansions
L = load_derivation('latex')                         # the TeX parser
mojo_constants = load_derivation('mojo_constants')   # the code's constants
K_LOW, K_ASYM = derivation.numeric_expansions()[-1]   # the expansions' ends: k = 4.02 (the tables' first node), 1e3

# Lee, Lyubyakin & Stotsky's exact cross section (eq. 4.4), from the TeX source of arXiv:2010.15430v1
s, t = sp.symbols('s t', positive=True)
# eq. 4.3, from the TeX source: for each k the kernel of J_k = int_9^s ds~ J_1(s~) K_k(s, s~), J_11's as printed
ell5 = sp.log((sp.sqrt(s - 5) + sp.sqrt(s - 1)) / (sp.sqrt(t - 5) + sp.sqrt(t - 1)))
ell3 = sp.log((sp.sqrt(s - 1) + sp.sqrt(s + 3)) / (sp.sqrt(t - 1) + sp.sqrt(t + 3)))
K9 = (7 * t - 15) / (t - 9) / sp.sqrt((t - 5) * (t - 1))
KERNELS = {3: 2 / (t - 9), 4: 2 / (t - 9) * sp.log((s - 1) / (t - 1)), 5: 1 / (t - 1),
           6: sp.Rational(1, 2) / (t - 1) * sp.log(s / t), 7: -3 / (t - 9) * sp.log(s / t),
           8: 2 * (7 * t - 15) / ((t - 9) * (t - 1)) * ell3, 9: K9,
           10: K9 * sp.log((s - 1) / (t - 1)) - 32 * t / ((t - 9) * (t - 1)) * ell5,
           11: K9 * ell5,
           12: K9 * sp.log((s - 5) / (t - 5)) + 2 * (5 * t + 3) / ((t - 9) * (t - 1)) * ell5}


# eq. 4.4, from the TeX source, with its y, z, J_1 and J_2
LEE_44 = [r'\sigma_{\egeee}=\frac{\alpha ^3}{s-1}',
          r'\frac{4\left(168 s^4-2714 s^3+8697 s^2+1128 s+81\right) J_1}{27(s-1)^2 s}',
          r'-\frac{2 \left(872 s^4-11289 s^3+29949 s^2+729s+27\right) J_2}{27 (s-1)^2 s}',
          r'-\frac{32 \left(s^2-5 s-2\right)J_6}{(s-1)^2}',
          r'+\frac{4 \left(24 s^6-38s^5-228 s^4+507 s^3+1161 s^2-45 s+27\right) J_3}{3 (s-1)^2s^2 (s+3)}',
          r'-\frac{16 \left(9 s^2-46 s+17\right)J_4}{(s-1)^2}',
          r'-\frac{8 \left(3 s^6-3 s^5-59 s^4-94 s^3-159 s^2+33s-9\right) J_5}{3 (s-1)^2 s^2 (s+3)}',
          r'+\frac{16 \left(2 s^2-8 s-1\right)J_7}{(s-1)^2}',
          r'+\frac{32 \left(s^4-s^3-21 s^2-27s+16\right) y J_8}{(s-1)^3 (s+3)}',
          r'-\frac{8 \left(3 s^4-30s^3+116 s^2-146 s-7\right) z J_9}{3 (s-5) (s-1)^2}',
          r'-\frac{8 (s+1) z J_{10}}{s-1}',
          r'+\frac{8 \left(3 s^2-6 s-5\right)J_{11}}{(s-1)^2}',
          r'-\frac{16 \left(s^3-3 s^2-9 s+19\right) zJ_{12}}{(s-5) (s-1)^2}']
LEE_YZ = r'$y=\sqrt{\frac{s-1}{s+3}}$ and $z=\sqrt{\frac{s-5}{s-1}}$'
LEE_J1 = (r'J_1(s)=\frac{s-9}{6 \left(\sqrt{s}-1\right)^{3/2} \left(\sqrt{s}+3\right)^{1/2}}'
          r'\mathrm{K}\left(\frac{\left(\sqrt{s}-3\right) '
          r'\left(\sqrt{s}+1\right)^3}{\left(\sqrt{s}-1\right)^3 \left(\sqrt{s}+3\right)}\right)')
LEE_J2 = (r'J_2(s)=\frac{1}{8s} \left(\sqrt{s}-1\right)^{3/2} \left(\sqrt{s}+3\right)^{1/2}\, '
          r'\mathrm{E}\left(\frac{\left(\sqrt{s}-3\right) '
          r'\left(\sqrt{s}+1\right)^3}{\left(\sqrt{s}-1\right)^3 \left(\sqrt{s}+3\right)}\right)',
          r'-\frac{2J_1(s)}{ \sqrt{s}(\sqrt{s}-3)}')
DPS, NODES = 30, 20      # digits; Gauss-Legendre nodes per panel of the integrals J_3 ... J_12
_J = {}                  # J_3 ... J_12 (J_11 as printed) by s, computed once


@functools.cache
def _lee_setup():
    """Eq. 4.4 with J_1, J_2, y and z parsed from the TeX, as a function of s and J_3 ... J_12. KERNELS become sums
    of a(s) b(t), logarithms of ratios split, so that one pass over s~ gives every J_k."""
    js = {kk: sp.Symbol(f'J{kk}') for kk in range(1, 13)}
    tokens = {'s': s, r'\alpha': 1} | {(f'J_{kk}' if kk < 10 else f'J_{{{kk}}}'): js[kk] for kk in js}
    tokens |= {L.sides(p)[0]: L.to_sympy(L.sides(p)[1], {'s': s}) for p in LEE_YZ.split('$')[1::2]}   # y, z
    j1 = L.to_sympy(LEE_J1.split('=', 1)[1], {'s': s}, funcs={r'\mathrm{K}': sp.elliptic_k})
    j2 = L.to_sympy(LEE_J2[0].split('=', 1)[1] + LEE_J2[1], {'s': s},
                    funcs={r'\mathrm{E}': sp.elliptic_e, 'J_1': lambda a: j1.subs(s, a)})
    sigma = L.to_sympy(LEE_44[0].split('=')[1], tokens) * L.to_sympy(''.join(LEE_44[1:]), tokens)
    terms, basis = {}, []
    for kk, kernel in KERNELS.items():
        terms[kk] = []
        for term in sp.Add.make_args(sp.expand(sp.expand_log(kernel, force=True))):
            a, b = term.as_independent(t, as_Add=False)
            basis += [b] if b not in basis else []
            terms[kk].append((sp.lambdify(s, a, 'mpmath'), basis.index(b)))
    return dict(j1=sp.lambdify(s, j1, 'mpmath'), basis=sp.lambdify(t, basis, 'mpmath'), n=len(basis), terms=terms,
                sigma=sp.lambdify([s] + [js[kk] for kk in range(3, 13)], sigma.subs({js[1]: j1, js[2]: j2}), 'mpmath'))


def _integrals(svals, setup):
    """{s: [J_3 ... J_12]} (J_11 as printed) for svals, in one pass over s~ from 9: Gauss-Legendre panels of 1/4 on
    9 <= s~ <= 10 and of at most 0.3 in ln s~ above, split at every s in svals."""
    X, W = mp.gauss_quadrature(NODES, 'legendre')
    top = max(svals)
    edges = {9 + mp.mpf(i) / 4 for i in range(5)} | set(svals)
    if top > 10:
        n_ = int(mp.ceil(mp.log(top / 10) / mp.mpf('0.3')))
        edges |= {10 * (top / 10) ** (mp.mpf(i) / n_) for i in range(n_ + 1)}
    edges = sorted(e for e in edges if e <= top)
    total, out = [mp.mpf(0)] * setup['n'], {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        h, mid = (hi - lo) / 2, (hi + lo) / 2
        for xv, wv in zip(X, W):
            weight = wv * h * setup['j1'](mid + h * xv)
            for i, b in enumerate(setup['basis'](mid + h * xv)):
                total[i] += weight * b
        if hi in svals:
            out[hi] = [sum(a(hi) * total[i] for a, i in setup['terms'][kk]) for kk in range(3, 13)]
    return out


def lee_sigma(ks):
    """sigma/(alpha r_e^2) of eq. 4.4 (alpha^3 with m_e = 1) at the photon energies ks (s = 1 + 2k), with the factor 2
    in eq. 4.3's J_11 that eq. 4.2 requires. Call within mp.workdps(DPS), with every k needed in one call (one pass
    over s~)."""
    setup = _lee_setup()
    svals = [1 + 2 * mp.mpf(kv) for kv in ks]
    new = {sv for sv in svals if sv not in _J}
    if new:
        _J.update(_integrals(new, setup))
    return [setup['sigma'](sv, *[jv * (2 if kk == 11 else 1) for kk, jv in zip(range(3, 13), _J[sv])])
            for sv in svals]


# Haug (1975, Z. Naturforsch. 30a, 1099), Table 1: the photon energy k (m_e c^2, target electron at rest) and the total
# cross section, in the table's units (10^-10 b for k <= 4.05, mub for 4.1 <= k <= 7.8, mb above)
HAUG = [(1e-7, [(4.001, 0.03255), (4.002, 0.1307), (4.003, 0.2951), (4.004, 0.5265), (4.005, 0.8256), (4.01, 3.361),
                (4.02, 13.91), (4.03, 32.34), (4.04, 59.3), (4.05, 95.5)]),
        (1e-3, [(4.1, 0.0436), (4.2, 0.2133), (4.4, 1.110), (4.6, 2.940), (5.0, 9.85), (5.4, 21.22), (5.8, 36.80),
                (6.0, 46.0), (6.2, 56.1), (6.6, 78.7), (7.0, 104.0), (7.4, 131.5), (7.8, 161.0)]),
        (1.0, [(8, 0.1764), (9, 0.2576), (10, 0.3439), (11, 0.433), (12, 0.522), (13, 0.611), (14, 0.700), (16, 0.871),
               (18, 1.035), (20, 1.191), (25, 1.545), (30, 1.856), (35, 2.131), (40, 2.376), (45, 2.598), (50, 2.800),
               (60, 3.154), (70, 3.459), (80, 3.726), (90, 3.963), (100, 4.176), (200, 5.58), (300, 6.40), (400, 6.98),
               (500, 7.42), (600, 7.78), (700, 8.09), (800, 8.35), (900, 8.58), (1000, 8.78), (1500, 9.56),
               (2000, 10.11), (2500, 10.53), (3000, 10.88), (5000, 11.83)])]


def haug():
    """Haug's k and sigma/(alpha r_e^2), alpha r_e^2 the code's ALPHA_RE2 sigma_T in its MILLIBARN."""
    code = mojo_constants.read(('ALPHA_RE2', 'SIGMA_T', 'MILLIBARN'))
    unit = mp.mpf(code['ALPHA_RE2'] * code['SIGMA_T'] / code['MILLIBARN'])
    return [(mp.mpf(repr(kv)), mp.mpf(repr(sv)) * mp.mpf(repr(to_mb)) / unit)
            for to_mb, rows in HAUG for kv, sv in rows]


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
kh, sh = (np.array([float(v) for v in col]) for col in zip(*haug()))   # Haug's k, sigma_t / (alpha r_e^2)
expansion = (kh <= T.k[0]) | (kh >= K_ASYM)   # Haug's k where the code uses an expansion
dense = [k for k in 4 + np.geomspace(K_LOW - 4, K_ASYM - 4, 400) if T.k[0] < k < K_ASYM]
mc_ks = [float(a) for a in sys.argv[sys.argv.index('--mc') + 1:]] if '--mc' in sys.argv else []
needed = sorted({float(k) for k in list(T.k) + list(kh[expansion]) + dense + mc_ks})
with mp.workdps(DPS):   # eq. 4.4 at every k needed, in one pass
    EXACT = dict(zip(needed, (float(v) for v in lee_sigma([mp.mpf(repr(k)) for k in needed]))))
    low, high, _ = derivation.readme_expansions()
    thr = lambda k: low(mp.mpf(repr(float(k))))             # eq. 4.5
    asym_rounded = lambda k: high(mp.mpf(repr(float(k))))   # eq. 4.13
ex = lambda k: EXACT[float(k)]
print('=== Monte Carlo nodes vs exact')
dn = np.array([s / ex(k) - 1 for k, s in zip(T.k, T.sig)])
for k, d in zip(T.k, dn):
    print(f'  k={k:<12.6g} MC/exact-1 {d:+.2e}')
mx = lambda sel: np.max(np.abs(dn[sel]))
print('  max |MC/exact-1| all nodes %.2e; k <= 10: %.2e; 10 < k <= 1e3: %.2e; k > 1e3: %.2e; rms(k>1e3) %.2e' % (
    np.max(np.abs(dn)), mx(T.k <= 10), mx((T.k > 10) & (T.k <= K_ASYM)), mx(T.k > K_ASYM),
    np.sqrt(np.mean(dn[T.k > K_ASYM] ** 2))))
print(f"=== code's expansions vs exact at Haug's energies (k <= {K_LOW:g}, k >= {K_ASYM:g})")
code = T.sigma(kh)
for k, c in zip(kh[expansion], code[expansion]):
    print(f'  k={k:<7g} code/exact-1 {c / ex(k) - 1:+.2e}')
print("=== code sigma(k) vs Haug's Table 1")
node = np.array([np.any(np.isclose(x, T.k, rtol=1e-9)) for x in kh])
where = np.where(kh <= T.k[0], 'threshold exp.', np.where(kh >= K_ASYM, 'high-energy exp.',
                                                        np.where(node, 'node', 'between nodes')))
print('    k      Haug [alpha r_e^2]   code/Haug - 1   code uses')
for x, h, c, w in zip(kh, sh, code, where):
    print(f'{x:8g}   {h:16.6g}   {c / h - 1:+12.1e}     {w}')
for label, w in [('on table nodes', 'node'), ('between table nodes', 'between nodes')]:
    sel = where == w
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
print('  HE expansion / tables for k >= 1e3: max |.| %.2e (at k = %.4g); values %s' % (
    np.max(np.abs(r)), T.k[hi][np.argmax(np.abs(r))], np.array2string(r[:8], precision=1)))
if '--mc' in sys.argv:
    print('=== Monte Carlo at the given k')
    for x in mc_ks:
        res = mc_sigma(x)
        c = T.sigma([x])[0]
        line = (f'k = {x:g}: Monte Carlo {res} (Q = {res.Q:.2f}); code {c:.6g}; Monte Carlo / exact - 1 = '
                f'{res.mean / ex(x) - 1:+.1e}, code / Monte Carlo - 1 = {c / res.mean - 1:+.1e}')
        if x in kh:
            ref = sh[list(kh).index(x)]
            line += f'; Monte Carlo / Haug - 1 = {res.mean / ref - 1:+.1e}, code / Haug - 1 = {c / ref - 1:+.1e}'
        print(line, flush=True)
