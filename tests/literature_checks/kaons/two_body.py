"""Photo-hadronic kaons: the two-body channels gamma N -> K Lambda, K Sigma
(kaons/README.md "Two-body production near threshold"). The script reads
tests/support/measured/ky_angular.py: the kaon's cross section and mean CM
angle per energy bin of the published data (made by
tools/ky_angular_extract.py).
Printed:
  1. The steps of the code's table: sigma and chi of every channel column, and
     chi against its isotropic value (the page: p -> K+ 5 % below it near
     threshold and by more than half at 3-5 GeV, n -> K0 7-8 % above it below
     1.25 GeV). Also <cos theta> of K+ Lambda near threshold and at 3.5 GeV.
  2. The data and beyond: each data set's range (the page's table of the data).
The script does not read the code (tests/unit/test_added_processes.py compares
the code's table with these values). kaons/kaon_production.py checks the page's
steps (kp.steps), the scale factors of the analogous gamma p reactions
(kp.analogues) and the power laws above all data (kp.rule4) against the same
data."""

import os, sys

import numpy as np

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
from support.measured import ky_angular as K

print("1. The steps: sigma [mubarn], chi, its isotropic value and their ratio")
T = K.steps_table()
for i, (a, b) in enumerate(K.STEPS):
    cells = []
    for c in K.COLUMNS:
        s, x = T[c][i]
        iso = K.step_chi(c, i, isotropic=True)
        cells.append(
            f"{c} {s:6.3f} {x:.3f} (iso {iso:.3f}, ratio {x / iso:.3f})"
            if s > 0
            else f"{c}  -"
        )
    print(f"  {a:.2f}-{b:.2f} GeV: " + ";  ".join(cells))
E = np.linspace(*K.STEPS[0], 2001)
s = K.sigma("K+L", E)
print(
    "  K+ Lambda <cos theta>: "
    f"{np.trapezoid(s * K.cos_theta('K+L', E), E) / np.trapezoid(s, E):+.3f} "
    "in the first step (sigma-weighted), "
    f"{K.cos_theta('K+L', np.array([3.5]))[0]:+.2f} at 3.5 GeV"
)

print(
    "\n2. The data and beyond: the range of every data set [GeV, first and "
    "last bin centre]"
)
for r, sets in K.DATA.items():
    print(
        f"  {r:5s} (threshold {K.threshold(r):.3f}): "
        + ";  ".join(
            f"{name} {pts[0][0]:.3f}-{pts[-1][0]:.3f}" for name, pts in sets
        )
    )
