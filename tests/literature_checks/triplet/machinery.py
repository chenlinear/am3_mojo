"""Triplet pair production (tests/literature_checks/triplet/README.md, "Energies of the products", where the code
extrapolates above the last node): why tools/triplet's tables stop at k = 1e10. At high k the gauge cancellations
between the diagrams approach the limit of double precision: the Ward identity of the triplet amplitude
(|M|^2 vanishes for eps -> k) measures the part of |M|^2 that breaks it, at k = 1e8 ... 1e14 (over 300 events).
With --high-k, tools/triplet's seeded Monte Carlo gives the total cross section, against Lee et al.'s high-energy
expansion with exact coefficients (eq. 4.12 of the journal version, 4.11 of arXiv v1). It also gives k<x>/ln^2(2k) of
the energies the tables hold (0: the positron, 1: the less energetic electron, 2: the recoil's loss xi = x+ + x_low),
each with a VEGAS map adapted to its energy-weighted rate, as in tools/triplet/produce.py.

usage: machinery.py [--high-k]   (--high-k: at k = 1e8, 1e10, 1e12 and 1e14, a few minutes)"""
import os, sys
import numpy as np
import sympy as sp
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.paths import load_derivation, use_tool

use_tool('triplet')
from qed_triplet import *
from kinematics import generate
derivation = load_derivation('triplet')   # Lee et al.'s eq. 4.12 as transcribed, and its parser


def eq412():
    """Lee et al.'s eq. 4.12 (exact coefficients): sigma/(alpha r_e^2) as an mpmath function of k."""
    return sp.lambdify(derivation.k, derivation.lee(derivation.LEE_412, 'k'), 'mpmath')


for k in [1e8, 1e10, 1e12, 1e14]:
    u = np.random.default_rng(1).random((300, 4))
    ok, jac, p1, kv, p2, p3, p4, inv = generate(u, k)
    args = [x[ok] for x in (p1, kv, p2, p3, p4)]
    d = {key: val[ok] for key, val in inv.items()}
    ward = np.max(amplitude_sq(*args, d, check_gauge=True) / amplitude_sq(*args, d))
    print(f'k = {k:.0e}: Ward identity {ward:.1e}')
if '--high-k' in sys.argv:
    import gvar, vegas
    from produce import events
    asym_exact = eq412()
    for k in [1e8, 1e10, 1e12, 1e14]:
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
