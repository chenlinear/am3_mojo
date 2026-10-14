"""Triplet pair production: checks of the QED machinery of tools/triplet (README "Triplet pair
production"). Spinor completeness, the current and slash helpers, and Compton scattering with the
same spinors and gamma matrices against Klein-Nishina. Up to k = 1e7: the triplet amplitude's Ward
identity (|M|^2 vanishes for eps -> k), its invariance under a longitudinal boost, the accuracy of
the invariants passed to it, momentum conservation, and amplitude_sq_fast of qed_triplet.py (which
made the tables) against amplitude_sq (used above).
Then why the tables stop at k = 1e10: the Ward identity at k = 1e8 ... 1e14 (over 300 events). With
--high-k, the seeded Monte Carlo of tools/triplet gives the total cross section, against Lee et al.'s
high-energy expansion with exact coefficients (eq. 4.12 of the journal version, 4.11 of arXiv v1),
and k<x>/ln^2(2k) of the energies the tables hold (0: the positron, 1: the less energetic electron,
2: the recoil's loss xi = x+ + x_low), each with a VEGAS map adapted to its energy-weighted rate, as
in tools/triplet/produce.py.

usage: machinery.py [--high-k [k ...]]   (--high-k: default k = 1e8 1e10 1e12 1e14, a few minutes)"""
import os, sys
import numpy as np
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.paths import load_derivation, use_tool  # noqa: E402

use_tool('triplet')
from qed_triplet import *  # noqa: E402,F403
from kinematics import generate  # noqa: E402
derivation = load_derivation('triplet')   # the Klein-Nishina cross section, Lee et al.'s eq. 4.12
rng = np.random.default_rng(1)
# 1. spinor completeness: sum_s u ubar = pslash + m, sum_s v vbar = pslash - m
p = onshell(np.array([3.7]), np.array([0.4]), np.array([-1.1]))
u = u_spinors(p)[0]; v = v_spinors(p)[0]
Su = sum(np.outer(u[s], bar(u[s][None])[0]) for s in range(2))
Sv = sum(np.outer(v[s], bar(v[s][None])[0]) for s in range(2))
print('completeness u:', np.abs(Su - (slash(p)[0] + np.eye(4))).max(), ' v:', np.abs(Sv - (slash(p)[0] - np.eye(4))).max())
# current vs explicit gamma matrices: J.a = ubar aslash w for a test vector a
a = np.array([[1.3, 0.7, -0.2, 0.5]])
J = current(bar(u[0][None]), u[1][None])
print('current:', abs(dot(J, a)[0] - sandwich(bar(u[0][None]), slash(a), u[1][None])[0]))
# 2. Compton scattering e(p1 at rest) + gamma(k along z) -> e(p2) + gamma(k2), vs Klein-Nishina
def compton_dsig(k, cth):
    kp = k / (1 + k * (1 - cth))
    sth = np.sqrt(1 - cth**2)
    k2 = np.array([[kp * (1 + cth), kp * (1 - cth), kp * sth, 0.0]])   # photon out, light-cone
    kin = np.array([[2 * k, 0.0, 0.0, 0.0]])
    p1 = np.array([[1.0, 1.0, 0.0, 0.0]])
    p2 = p1 + kin - k2
    u1 = u_spinors(p1)[0]; u2 = u_spinors(p2)[0]
    # incoming photon polarisations (k along z): x and y
    e_in = [np.array([[0, 0, 1.0, 0]]), np.array([[0, 0, 0, 1.0]])]
    # outgoing: two vectors orthogonal to k2 and to each other (transverse). For k2 at angle th in x-z:
    # eps = (cos th, 0, -sin th) and (0,1,0) in Cartesian -> light-cone
    ex = np.array([[(-sth), (sth), cth, 0.0]])   # (eps+, eps-, ex, ey): eps0 = 0, epsz = -sin th
    ey = np.array([[0, 0, 0, 1.0]])
    tot = 0.0
    for ei in e_in:
        for eo in (ex, ey):
            Ei, Eo = slash(ei)[0], slash(eo)[0]   # real polarisations: eps* = eps
            Ps = slash(p1 + kin)[0] + np.eye(4); Pu = slash(p1 - k2)[0] + np.eye(4)
            Ds = dot(p1 + kin, p1 + kin)[0] - 1; Du = dot(p1 - k2, p1 - k2)[0] - 1
            X = Eo @ Ps @ Ei / Ds + Ei @ Pu @ Eo / Du
            for s1 in range(2):
                for s2 in range(2):
                    tot += abs(bar(u2[s2][None])[0] @ X @ u1[s1]) ** 2
    # dsigma/dOmega = |M|^2_avg / (64 pi^2) (k'/k)^2 (m = 1), |M|^2 has e^4 = (4 pi alpha)^2
    avg = tot / 4 * (4 * np.pi) ** 2
    ds = avg / (64 * np.pi ** 2) * (kp / k) ** 2      # in units alpha^2/m^2 = r_e^2
    return ds, derivation.kn_dsigma(k, cth)
for k, c in [(0.3, 0.2), (2.0, -0.7), (50.0, 0.95)]:
    ds, kn = compton_dsig(k, c)
    print(f'Compton k={k} cos={c}: |M|^2 machinery {ds:.10f}  Klein-Nishina {kn:.10f}  rel {ds/kn-1:.1e}')
# 3. Ward identity and longitudinal-boost invariance for triplet events
for k in [5.0, 50.0, 1e4, 1e7]:
    u = rng.random((200, 4))
    ok, jac, p1, kv, p2, p3, p4, inv = generate(u, k)
    sel = ok
    args = [x[sel] for x in (p1, kv, p2, p3, p4)]
    d = {key: val[sel] for key, val in inv.items()}
    m2 = amplitude_sq(*args, None, d)
    g = amplitude_sq(*args, None, d, check_gauge=True)
    fast = np.max(np.abs(amplitude_sq_fast(*args, None, d) / m2 - 1))
    # boost: p+ -> lam p+, p- -> p-/lam (invariants unchanged)
    lam = 7.3
    def bst(x):
        y = x.copy(); y[..., 0] *= lam; y[..., 1] /= lam; return y
    db = dict(d); db['p3mk'] = bst(d['p3mk']); db['kmp4'] = bst(d['kmp4']); db['p2mk'] = bst(d['p2mk'])
    m2b = amplitude_sq(*[bst(x) for x in args], None, db)
    # momentum conservation
    cons = np.abs((p1 + kv - p2 - p3 - p4)[sel]).max(axis=0) / np.array([2 * k, 1, 1, 1])
    print(f'k={k:g}: {sel.sum()} events; Ward |M(eps->k)|^2/|M|^2 max {np.max(g/m2):.1e}; boost max rel {np.max(np.abs(m2b/m2-1)):.1e}; '
          f'momentum conservation (rel +, abs -, x, y) {cons}; amplitude_sq_fast against amplitude_sq {fast:.1e}')
    # accurate invariants vs direct dot products
    print('   t23 vs dot', np.max(np.abs(inv['t23'][sel] - dot(p1-p2, p1-p2)[sel]) / np.abs(inv['t23'][sel])),
          ' kp3 vs dot', np.max(np.abs(inv['kp3'][sel] - dot(kv, p3)[sel]) / inv['kp3'][sel]),
          ' kp4 vs dot', np.max(np.abs(inv['kp4'][sel] - dot(kv, p4)[sel]) / inv['kp4'][sel]))
# 4. at high k the gauge cancellations between the diagrams approach the limit of double precision: the Ward
#    identity measures the part of |M|^2 that breaks it (its square root that of the amplitude)
for k in [1e8, 1e10, 1e12, 1e14]:
    u = np.random.default_rng(1).random((300, 4))
    ok, jac, p1, kv, p2, p3, p4, inv = generate(u, k)
    args = [x[ok] for x in (p1, kv, p2, p3, p4)]
    d = {key: val[ok] for key, val in inv.items()}
    m2 = amplitude_sq(*args, None, d)
    ward = np.max(amplitude_sq(*args, None, d, check_gauge=True) / m2)
    fast = np.max(np.abs(amplitude_sq_fast(*args, None, d) / m2 - 1))
    print(f'k = {k:.0e}: Ward identity {ward:.1e}; amplitude_sq_fast against amplitude_sq {fast:.1e}')
if '--high-k' in sys.argv:
    import gvar, vegas
    from produce import events  # noqa: E402
    asym_exact = derivation.eq412()
    ks = [float(a) for a in sys.argv[sys.argv.index('--high-k') + 1:]] or [1e8, 1e10, 1e12, 1e14]
    for k in ks:
        gvar.ranseed(2)
        integ = vegas.Integrator(4 * [[0.0, 1.0]])
        f0 = vegas.batchintegrand(lambda u: events(np.asarray(u), k)[0])
        integ(f0, nitn=8, neval=100000)
        s = integ(f0, nitn=8, neval=100000)
        out = [f'k = {k:.0e}: sigma {s} alpha r_e^2 (expansion {float(asym_exact(k)):.6g})']
        for role in (0, 1, 2):
            gvar.ranseed(3 + role)
            integ = vegas.Integrator(4 * [[0.0, 1.0]])
            fr = vegas.batchintegrand(lambda u, r=role: (lambda w, x: w * x[r])(*events(np.asarray(u), k)))
            integ(fr, nitn=10, neval=100000)
            m = integ(fr, nitn=10, neval=150000)
            out.append(f'k<x_{role}>/ln^2(2k) {k * m / s / np.log(2 * k) ** 2}')
        print('; '.join(out), flush=True)
