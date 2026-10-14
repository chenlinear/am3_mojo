"""Triplet pair production, README "Triplet pair production": the code's mean energy fraction of the pair,
<x+ + x_low> (the inelasticity: the share of the lepton's energy the pair takes), against the literature:
  Mastichiadis' fit as Lee (1998, eq. 25) quotes it:  eta = 1.768 (s/m_e^2)^(-3/4), s = 1 + 2k
  Anguelov et al. (1999), mean energy per produced particle: 0.195 ln^2(2k) / k (k > 100)
The code's values (through tests/harness/triplet_mojo.mojo) come at fixed k from its QED tables, in isotropic
photons from its kernel rows on the default grid, gamma eps = e^(0.1 m). Also the energy-loss rate in
monoenergetic isotropic soft photons against the estimate of Dermer & Schlickeiser (1991, A&A 252, 414), as Dermer
(2021, arXiv:2111.09106) quotes it: -dgamma/dt = (4/3) alpha_f c sigma_T n (gamma/eps)^(1/2), "accurate to a factor
of 3 over more than 5 orders of magnitude" in eps gamma against Mastichiadis and colleagues' numerical results. And
against the inverse-Compton loss (README: the triplet loss is larger above gamma eps = 2.6e8). Then Mastichiadis' own
fit and loss rate (his eqs. 26-27; README: 1.6 times too high at gamma eps = 1e3, 2.15 times at 1e6). Finally the
energy the recoiling lepton loses per interaction, xi = x+ + x_low: its mean in isotropic photons (README: 0.085 at
gamma eps = 1e2, 0.018 at 1e3), and at fixed k the share of the loss in interactions with xi > 0.095 (more than a
grid bin) and xi > 0.5 (README: 41, 24 and 20 % at k = 1e3, 1e5, 1e6; 3-6 %).
The literature's forms are literature_forms of derivations/triplet.py (read from the Results, checked against the
papers). It also derives the formulas quoted, and where Mastichiadis' eq. 27 equals the inverse-Compton loss
(his eq. 28)."""
import os, sys
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.harness import Triplet  # noqa: E402
from support.paths import load_derivation  # noqa: E402

derivation = load_derivation('triplet')

T = Triplet()
F = derivation.literature_forms()
XI_BIN = float(derivation.readme_xi_bin()[0])   # more than a grid bin (the Results' 0.095)
PAIR = {{'+': 0, 'low': 1}[r]: float(c) for r, c in derivation.readme_loss_fractions().items()}   # x+ + x_low


def ln_y(f):
    return T.ly0 + (np.arange(f.shape[-1]) + 0.5) * T.dly


def pair_fraction_k(ks):
    """<x+ + x_low> at fixed k, from the code's distributions."""
    return np.array([sum(c * f[r] @ np.exp(ln_y(f)) / f[r].sum() / k for r, c in PAIR.items())
                     for k, f in zip(ks, T.distributions(ks))])


# the code's kernel rows, interpolated in ln(gamma eps) between them (cubic, in the logarithms)
rows = T.rows()
on = rows['R'] > 0
LOSS = CubicSpline(np.log(rows['P'][on]), np.log(rows['L'][on]))
FRAC = CubicSpline(np.log(rows['P'][on]), np.log(rows['L'][on] / rows['R'][on]))
loss = lambda P: float(np.exp(LOSS(np.log(P))))            # noqa: E731  -gamma_dot/(gamma c sigma_T n) = R <x+ + x_low>
pair_fraction = lambda P: float(np.exp(FRAC(np.log(P))))   # noqa: E731  <x+ + x_low>, the energy the pair takes

print('   k        <x_pair> (QED)   Mastichiadis/Lee   ratio   Anguelov et al.   ratio   k<x_pair>/ln^2(2k)')
nodes = T.k[T.k >= 10]
for k, eta in zip(nodes, pair_fraction_k(nodes)):
    m, a = F['lee'](k), F['anguelov'](k)
    print(f'{k:9.4g}   {eta:13.5g}   {m:16.5g}   {m / eta:5.2f}   {a:15.5g}   {a / eta:5.2f}   {k * eta / np.log(2 * k) ** 2:.4f}')
print()
print('energy loss in monoenergetic isotropic photons: exact / Dermer & Schlickeiser (1991)')
for P in [10, 30, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
    exact = loss(P)
    ds = F['ds'](P)
    print(f'  gamma eps = {P:8.3g}: exact {exact:.4g}, DS {ds:.4g}, exact/DS {exact / ds:.3f}')
Ps = np.exp(np.linspace(np.log(30), np.log(3e5), 41))
over = np.array([F['ds'](P) / loss(P) for P in Ps])
print(f'  DS / exact for 30 <= gamma eps <= 3e5: {over.min():.2f} (at {Ps[np.argmin(over)]:.3g}) to {over.max():.2f}'
      f' (at {Ps[np.argmax(over)]:.3g})')
print()
# against the inverse-Compton loss in its Klein-Nishina limit (Blumenthal & Gould 1970), which the exact Klein-Nishina
# rate approaches within 0.2 % for gamma eps >= 1e3 (derivations/am3_cooling.py, am3.ic_kn_limit)
print('energy loss in monoenergetic isotropic photons: triplet / inverse Compton (Klein-Nishina limit)')
tpp = loss
ic = F['ic']
for P in [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 3e9]:
    print(f'  gamma eps = {P:8.3g}: triplet / IC = {tpp(P) / ic(P):.3f}; <x+ + x_low> = {pair_fraction(P):.3g}')
lo, hi = 1e6, 3e9
for _ in range(60):
    mid = np.sqrt(lo * hi)
    lo, hi = (lo, mid) if tpp(mid) > ic(mid) else (mid, hi)
print(f'  the triplet loss exceeds the inverse-Compton loss above gamma eps = {mid:.2g} '
      f'(<x+ + x_low> = {pair_fraction(mid):.2g} there)')
print()
# Mastichiadis (1991, MNRAS 253, 235): eq. 26, the pair's share of the energy, and eq. 27, the loss rate
print('Mastichiadis (1991) in monoenergetic isotropic photons: pair fraction (eq. 26) and loss rate (eq. 27) / exact')
m26, m27 = F['mast26'], F['mast27']
for P in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
    print(f'  gamma eps = {P:8.3g}: eq. 26 / exact {m26(P) / pair_fraction(P):.2f}, eq. 27 / exact {m27(P) / tpp(P):.2f}')
print()
print('the energy the recoiling lepton loses per interaction, xi = x+ + x_low')
for P in [1e2, 1e3]:
    print(f'  in isotropic photons, gamma eps = {P:g}: <xi> = {pair_fraction(P):.3g}')
for k, d in zip([1e3, 1e5, 1e6], T.distributions([1e3, 1e5, 1e6])):
    f = d[2]   # the recoil: y = xi k
    xi = np.exp(ln_y(f)) / k
    e = f * xi  # the energy lost, per bin of ln xi
    print(f'  at k = {k:.0e}: of the energy lost, {e[xi > XI_BIN].sum() / e.sum():.0%} in interactions with xi > {XI_BIN:g},'
          f' {e[xi > 0.5].sum() / e.sum():.1%} with xi > 0.5')
