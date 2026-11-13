"""Triplet pair production, tests/literature_checks/triplet/README.md ("Kinetic
terms": the continuous loss): the code's inelasticity <x+ + x_low> at fixed k,
from its QED tables (via tests/harness/triplet_mojo.mojo). The inelasticity is
the mean share of the lepton's energy that the pair takes. The script also
prints k<x+ + x_low>/ln^2(2k) (README: k<x+ + x_low> at k = 1e2, 1e3 and 1e4
against Anguelov et al.'s Table 2).
In monoenergetic isotropic soft photons, the script takes the energy-loss rate
from the code's kernel rows on the default grid. It compares this rate with the
inverse-Compton loss in its Klein-Nishina limit (README: larger above
gamma eps = 2.6e8, where <x+ + x_low> = 5.6e-7).
The energy that the recoiling lepton loses per interaction is xi = x+ + x_low.
The script prints its mean in isotropic photons (README: 0.085 at
gamma eps = 1e2, 0.018 at 1e3). At fixed k, it prints the share of the loss in
interactions with xi > 0.095 (more than a grid bin) and with xi > 0.5 (README:
41, 24 and 20 % at k = 1e3, 1e5 and 1e6, and 3-6 %).
The inverse-Compton loss in its Klein-Nishina limit (ic_loss) and the xi of more
than a grid bin (readme_xi_bin) are read from this folder's README.md."""

import os, sys
from pathlib import Path
from re import search as re_search
import mpmath as mp
import numpy as np
import sympy as sp
from scipy.interpolate import CubicSpline

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
from support.harness import Triplet
from support.paths import load_derivation

derivation = load_derivation("triplet")
L = load_derivation("latex")
PAGE = Path(__file__).resolve().parent / "README.md"
n = sp.Symbol("n", positive=True)  # the photon density


def readme_match(pattern, pin):
    """The match of the pattern in the README line that holds the pin (a pin
    without the numbers and equations read)."""
    found = re_search(pattern, L.line(pin, PAGE))
    if found is None:
        raise LookupError(f"{PAGE}: pattern not found in the line of {pin}")
    return found


def readme_kn_ic():
    m_ = readme_match(
        r"its Klein–Nishina limit, \$-\\dot\\gamma\$ "
        r"\$= ([^$]+)\$ \$([^$]+)\$, for",
        derivation.IC_LINE,
    )
    tokens = derivation.TOKENS | {"n": n}
    return L.to_sympy(m_[1], tokens) * L.to_sympy(m_[2], tokens)


def readme_xi_bin():
    tex = readme_match(
        r"loses more than a grid bin \(\$\\xi \\gt ([\d.]+)\$\)",
        "Interactions where the lepton loses more than a grid bin",
    )[1]
    return mp.mpf(tex), len(tex.replace(".", "").lstrip("0"))


def ic_loss():
    P = sp.Symbol("P", positive=True)
    c, sT, gam, eps = (
        derivation.c,
        derivation.sT,
        derivation.gam,
        derivation.eps,
    )
    expr = sp.simplify((readme_kn_ic() / (c * sT * n * gam)).subs(eps, P / gam))
    assert expr.free_symbols <= {P}, expr
    return sp.lambdify(P, expr, "numpy")


T = Triplet()
ic = ic_loss()
XI_BIN = float(readme_xi_bin()[0])
PAIR = {
    {"+": 0, "low": 1}[r]: float(c)
    for r, c in derivation.readme_loss_fractions().items()
}


def ln_y(f):
    return T.ly0 + (np.arange(f.shape[-1]) + 0.5) * T.dly


def pair_fraction_k(ks):
    return np.array(
        [
            sum(
                c * f[r] @ np.exp(ln_y(f)) / f[r].sum() / k
                for r, c in PAIR.items()
            )
            for k, f in zip(ks, T.distributions(ks))
        ]
    )


rows = T.rows()
on = rows["R"] > 0
LOSS = CubicSpline(np.log(rows["P"][on]), np.log(rows["L"][on]))
FRAC = CubicSpline(np.log(rows["P"][on]), np.log(rows["L"][on] / rows["R"][on]))
# -gamma_dot/(gamma c sigma_T n) = R <x+ + x_low>
loss = lambda P: float(np.exp(LOSS(np.log(P))))
pair_fraction = lambda P: float(np.exp(FRAC(np.log(P))))

print("   k        <x_pair> (QED)   k<x_pair>/ln^2(2k)")
nodes = T.k[T.k >= 10]
for k, eta in zip(nodes, pair_fraction_k(nodes)):
    print(f"{k:9.4g}   {eta:13.5g}   {k * eta / np.log(2 * k) ** 2:.4f}")
print()
print(
    "energy loss in monoenergetic isotropic photons: triplet / inverse "
    "Compton (Klein-Nishina limit)"
)
for P in [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 3e9]:
    print(
        f"  gamma eps = {P:8.3g}: triplet / IC = {loss(P) / ic(P):.3f}; "
        f"<x+ + x_low> = {pair_fraction(P):.3g}"
    )
lo, hi = 1e6, 3e9
for _ in range(60):
    mid = np.sqrt(lo * hi)
    lo, hi = (lo, mid) if loss(mid) > ic(mid) else (mid, hi)
print(
    "  the triplet loss exceeds the inverse-Compton loss above gamma eps "
    f"= {mid:.2g} (<x+ + x_low> = {pair_fraction(mid):.2g} there)"
)
print()
print("the energy the recoiling lepton loses per interaction, xi = x+ + x_low")
for P in [1e2, 1e3]:
    print(
        f"  in isotropic photons, gamma eps = {P:g}: <xi> = "
        f"{pair_fraction(P):.3g}"
    )
for k, d in zip([1e3, 1e5, 1e6], T.distributions([1e3, 1e5, 1e6])):
    f = d[2]  # the recoil
    xi = np.exp(ln_y(f)) / k
    e = f * xi
    print(
        f"  at k = {k:.0e}: of the energy lost, "
        f"{e[xi > XI_BIN].sum() / e.sum():.0%} in interactions with xi > "
        f"{XI_BIN:g}, {e[xi > 0.5].sum() / e.sum():.1%} with xi > 0.5"
    )
