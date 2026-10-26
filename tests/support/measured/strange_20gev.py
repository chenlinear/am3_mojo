"""Inclusive strange-particle photoproduction at 20 GeV: the measured values behind the kaons of the multi-pion type M4
(5-50 GeV; tests/literature_checks/kaons/README.md, "Kaons in multi-pion production"), and what follows from them.

SLAC Hybrid Facility Photon Collaboration, gamma p at 20 GeV in a hydrogen bubble chamber. Cross sections are corrected
for unseen decays and normalised to the total 115 mubarn. The 1984 distributions and 1985 cross sections are those of
the HEPData records (kaons/m4_steps.py checks them); the other values were checked against the preprints.
  * Abe et al. 1985 (Phys. Rev. D 32, 2869; SLAC-PUB-3634; HEPData ins213727), 284,000 hadronic events: inclusive cross
    sections, and the pairs of neutral strange particles in one event (preprint Table 2, not in HEPData).
  * Abe et al. 1984 (Phys. Rev. D 29, 1877; SLAC-PUB-3162; HEPData ins190724), an earlier 97,100 of these events: the
    inclusive K0_S cross section (Table II), K*+-(890) (Sec. 6), and the K0_S distributions in x, z and pT^2
    (Tables III, V and VI).

From them: the four kaon species that balance the hyperons' strangeness, the K0 among the kaons, and the share chi of
the nucleon energy that a K0_S takes, at 20 GeV and, by Feynman scaling, at other photon energies.
"""

import re
from functools import lru_cache

import numpy as np
import sympy as sp

from support.paths import load_derivation

readme = load_derivation('kaon_production')
L = load_derivation('latex')
PAGE = 'kaons'   # tests/literature_checks/kaons/README.md (a page of latex.FILES)

M_P, M_K0 = readme.MASS['p'], readme.MASS['K0']  # GeV, PDG 2026
E_GAMMA = 20.0  # GeV
SIGMA_TOT = (115.0, 2.0)  # mubarn (Caldwell et al. 1978, their Ref. 25)
# total gamma p cross sections [mubarn] of the SLAC 82-inch bubble chamber at its photon energies [GeV] (Ballam et al.
# 1972, Table V; Bingham et al. 1973, Table IV)
SIGMA_TOT_SLAC = ((1.44, 145.1), (2.8, 132.0), (4.7, 127.0), (9.3, 124.0))

# Abe et al. 1985: inclusive cross sections [mubarn] and errors. The Lambda include those from Sigma0 -> Lambda gamma
# and from Sigma*(1385) and Xi decays (as in Abe et al. 1984, Sec. 3); the anti-Lambda likewise.
K0S = (9.663, 0.272)
LAMBDA = (5.603, 0.180)
ANTI_LAMBDA = (0.389, 0.036)
SIGMA0 = (1.65, 0.44)
XI_MINUS = (0.117, 0.017)
ANTI_XI_PLUS = (0.010, 0.004)
SIGMA1385_PLUS = (0.63, 0.06)
SIGMA1385_MINUS = (0.33, 0.05)
OMEGA_MINUS_LIMIT = 0.017
# Abe et al. 1985, preprint Table 2: pairs of neutral strange particles in one event [mubarn], corrected for unseen
# decays as the single particles are (counted in parts of the decay angle cos theta_cm, corrected for the rest)
K0S_K0S = (0.973, 0.040)
K0S_LAMBDA = (1.125, 0.059)
K0S_ANTI_LAMBDA = (0.023, 0.009)
LAMBDA_ANTI_LAMBDA = (0.126, 0.038)
LAMBDA_LAMBDA = (0.028, 0.014)
# Abe et al. 1984, Table II: the K0_S cross section from the sample of the distributions below. Sec. 6: K*+(890) and
# K*-(890) [mubarn] (systematic errors of 15 % and 30 %)
K0S_1984 = (9.447, 0.318)
KSTAR_PLUS = (3.27, 0.35)
KSTAR_MINUS = (1.90, 0.28)

# Abe et al. 1984, Table III: (2 E*/(pi sqrt s)) dsigma/dx [mubarn] of the K0_S in bins of x = 2 p_L*/sqrt(s) in the
# centre-of-momentum frame, positive along the photon
X_EDGES = np.round(np.arange(-0.7, 0.9001, 0.1), 1)
F_K0S = 1e-2 * np.array([3.5, 10.4, 16.0, 30.1, 56.1, 71.0, 100.8, 134.1, 142.0, 121.2, 84.5, 57.3, 39.7, 19.2,
                         5.6, 1.0])

# Abe et al. 1984, Table V: (1/sigma_T) dsigma/dz of the K0_S in bins of z = E_K/E_gamma in the rest frame of the proton
Z_EDGES = np.array([0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.65, 0.75, 0.90])
DNDZ_K0S = 1e-3 * np.array([317.8, 343.2, 281.9, 224.1, 168.2, 127.2, 97.7, 70.7, 50.4, 31.0, 15.5, 6.21, 1.65])

# Abe et al. 1984, Table VI: dsigma/dpT^2 [mubarn/GeV^2] of the K0_S in bins of pT^2 [GeV^2]
PT2_EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
                      1.05, 1.15, 1.35, 1.55, 1.80])
DSDPT2_K0S = np.array([43.35, 31.18, 22.26, 17.97, 13.66, 9.80, 9.17, 6.50, 4.85, 4.21, 2.79, 1.77, 1.29, 0.945,
                       0.488, 0.411, 0.180, 0.117, 0.060])

# Shares in ABBHHM's strange-particle events (Erbe et al. 1969, Table VI; photons up to 5.8 GeV), derived and checked
# by kaons/multibody_strangeness.py. K0 among the kaons of K anti-K events: 0.45 without the reactions seen
# only when a K+- decays, 0.37 with them; anti-K0 among the antikaons: 0.53 and 0.44, taken as 0.5. K0 with a Sigma+
# and a Sigma- (events with further particles): 0.38 and 0.32; with a Lambda or Sigma0, 0.48, which the K0_S Lambda
# pairs at 20 GeV replace. Lambda or Sigma0 among the hyperons of events with further particles: 0.76.
K0_SHARE, ANTI_K0_SHARE = 0.45, 0.5
K0_SHARE_ALL, ANTI_K0_SHARE_RANGE = 0.37, (0.44, 0.53)
K0_SHARE_SIGMA_PLUS, K0_SHARE_SIGMA_MINUS, K0_SHARE_LAMBDA = 0.38, 0.32, 0.48
LAMBDA_SHARE_ABBHHM = 0.76
# The charged Sigma, not measured: Sigma+ + Sigma- = SIGMA_PM x Sigma0. Isospin gives 2 if the Sigma are made alike,
# as do any production linear in I3 and quark counting (a strange quark replacing the proton's d gives a Sigma+, one
# replacing a u a Lambda or Sigma0, none a Sigma-). Sigma+ : Sigma- as the Sigma*(1385); as many Xi0 as Xi-.
SIGMA_PM, XI0_OVER_XIM = 2.0, 1.0
SIGMA_PM_ALT = 1.5   # Sigma+ = Sigma0, Sigma+ / Sigma- as the measured Sigma*+(1385) / Sigma*-(1385), 1.9: 1 + 1/1.9


def kstar_ratio():
    """K*+(890) / K*-(890) at 20 GeV and its statistical error."""
    r = KSTAR_PLUS[0] / KSTAR_MINUS[0]
    return r, r * np.hypot(KSTAR_PLUS[1] / KSTAR_PLUS[0], KSTAR_MINUS[1] / KSTAR_MINUS[0])


def s_of(e_gamma):
    """Squared centre-of-momentum energy [GeV^2] of a photon of energy e_gamma [GeV] on a proton at rest (the
    s of kaons/README.md)."""
    return readme.numeric_s()(M_P, e_gamma)


def hyperon_strangeness(sigma_pm=SIGMA_PM, xi0_over_xim=XI0_OVER_XIM):
    """Strangeness [mubarn] of the hyperons less the antihyperons at 20 GeV, which the kaons less the antikaons
    balance. The Lambda (with those from decays) and the charged Sigma count once, a Xi twice (once through its
    Lambda), the antihyperons likewise (their charged Sigma in the hyperons' proportion). The Omega- (below
    0.017 mubarn) is left out."""
    sig_pm = sigma_pm * SIGMA0[0]
    xi = (1.0 + xi0_over_xim) * XI_MINUS[0]
    anti_xi = (1.0 + xi0_over_xim) * ANTI_XI_PLUS[0]
    hyperons = LAMBDA[0] + sig_pm + xi
    antihyperons = ANTI_LAMBDA[0] * (1.0 + sig_pm / LAMBDA[0]) + anti_xi
    return hyperons - antihyperons


def lambda_share():
    """The Lambda (with those from Sigma0) among the hyperons at 20 GeV, with the charged Sigma from the Sigma0."""
    return LAMBDA[0] / (LAMBDA[0] + SIGMA_PM * SIGMA0[0])


def k0_share_with_lambda(extra_pairs=0.0, xi0_over_xim=XI0_OVER_XIM):
    """The K0 share a of the kaons that come with a Lambda at 20 GeV, from the K0_S Lambda pairs. A Lambda comes with
    one kaon, those from a Xi with two, those of a Lambda anti-Lambda pair with none. extra_pairs K anti-K pairs per
    Lambda add a K0_S with probability p (from the K0 and anti-K0 shares in pairs). A K0 is a K0_S half the time:
    K0_S Lambda = n1 (a/2 + extra_pairs p) + n_xi (1 - (1 - a/2)^2)."""
    n_xi = (1.0 + xi0_over_xim) * XI_MINUS[0]
    n1 = LAMBDA[0] - LAMBDA_ANTI_LAMBDA[0] - n_xi
    b, c = K0_SHARE, ANTI_K0_SHARE
    p = 1.0 - (1.0 - b / 2.0) * (1.0 - c / 2.0)
    # n_xi a^2 / 4 - (n1 / 2 + n_xi) a + (K0S_LAMBDA - n1 extra_pairs p) = 0, the smaller root
    qa, qb, qc = n_xi / 4.0, -(n1 / 2.0 + n_xi), K0S_LAMBDA[0] - n1 * extra_pairs * p
    return (-qb - np.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)


def k0_share_with_hyperons(extra_pairs=0.0, sigma_pm=SIGMA_PM, xi0_over_xim=XI0_OVER_XIM):
    """The K0 among the kaons that come with the hyperons at 20 GeV: those with a Lambda (with those of Sigma0 and
    Xi decays) from the K0_S Lambda pairs, those with a charged Sigma as in ABBHHM's events."""
    a = k0_share_with_lambda(extra_pairs, xi0_over_xim)
    n_xi = (1.0 + xi0_over_xim) * XI_MINUS[0]
    n1 = LAMBDA[0] - LAMBDA_ANTI_LAMBDA[0] - n_xi
    ratio = SIGMA1385_PLUS[0] / SIGMA1385_MINUS[0]
    sig_p = sigma_pm * SIGMA0[0] * ratio / (1.0 + ratio)
    sig_m = sigma_pm * SIGMA0[0] / (1.0 + ratio)
    k0 = (n1 + 2.0 * n_xi) * a + sig_p * K0_SHARE_SIGMA_PLUS + sig_m * K0_SHARE_SIGMA_MINUS
    return k0 / (n1 + 2.0 * n_xi + sig_p + sig_m)


def species(k0_with_hyperons=None, k0_in_pairs=K0_SHARE, anti_k0_share=ANTI_K0_SHARE, sigma_pm=SIGMA_PM,
            xi0_over_xim=XI0_OVER_XIM):
    """Cross sections [mubarn] of (K+, K0, K-, anti-K0) at 20 GeV. Strangeness balance: the kaons (K+, K0) equal
    the antikaons (K-, anti-K0) plus hyperon_strangeness(). The kaons balancing the hyperons are K0 in the share
    k0_with_hyperons (default: k0_share_with_hyperons()), those paired with the antikaons in the share k0_in_pairs;
    the antikaons are anti-K0 in the share anti_k0_share. K0 and anti-K0 together are twice the measured K0_S."""
    hyperons = hyperon_strangeness(sigma_pm, xi0_over_xim)
    a = k0_share_with_hyperons(0.0, sigma_pm, xi0_over_xim) if k0_with_hyperons is None else k0_with_hyperons
    anti = (2.0 * K0S[0] - a * hyperons) / (k0_in_pairs + anti_k0_share)
    k0 = a * hyperons + k0_in_pairs * anti
    return np.array([anti + hyperons - k0, k0, (1.0 - anti_k0_share) * anti, anti_k0_share * anti])


def _sub(edges, n):
    """Midpoints of n equal parts of every bin, the index of their bin and the width of a part."""
    lo, hi = edges[:-1], edges[1:]
    f = (np.arange(n) + 0.5) / n
    pts = (lo[:, None] + (hi - lo)[:, None] * f[None, :]).ravel()
    return pts, np.repeat(np.arange(lo.size), n), np.repeat((hi - lo) / n, n)


# chi and x of the kaons of M4 as kaons/README.md defines them, for chi_samples
Es, W = sp.symbols('E_star W', positive=True)   # E* and sqrt(s)
pL = sp.Symbol('p_L', real=True)                # p*_L
TOK_M4 = {r'E^\ast': Es, r'p^\ast_L': pL, 's': W**2}


def readme_chi_M4():
    """The page's chi = (E* - p*_L)/sqrt(s) of the kaons of M4."""
    span = re.search(r'\$(\\chi = .+?)\$',
                     L.line(r"A kaon's share of the nucleon energy is $\chi = ", PAGE)).group(1)
    return L.to_sympy(L.sides(span)[1], TOK_M4)


def readme_x_M4():
    """The page's x = 2 p*_L/sqrt(s)."""
    return L.to_sympy(L.sides(L.inline(r'x = 2p^\ast_L/\sqrt{s}', PAGE))[1], TOK_M4)


def chi_of_x(mT, xr):
    """readme_chi_M4 with E* = sqrt(m_T^2 + p*_L^2) and p*_L from readme_x_M4 = xr."""
    return readme_chi_M4().subs(Es, sp.sqrt(mT**2 + pL**2)).subs(pL, sp.solve(readme_x_M4() - xr, pL)[0])


@lru_cache(None)
def numeric_chi_M4():
    """chi_of_x as a numpy function of (x, m_T, sqrt(s))."""
    mT, xr = sp.Symbol('m_T', positive=True), sp.Symbol('x', real=True)
    return sp.lambdify((xr, mT, W), chi_of_x(mT, xr), 'numpy')


def chi_samples(e_gamma=E_GAMMA):
    """(chi, x, weight) of the K0_S at the photon energy e_gamma [GeV]; the weights add up to the cross section
    [mubarn] at 20 GeV. The invariant cross section E* d3sigma/dp3 is taken as a function of x (Table III, which
    is its integral over pT^2) times one of pT^2, fixed so that dsigma/dpT^2 is that of Table VI at 20 GeV. With
    E* = (sqrt(s)/2) q and q = sqrt(x^2 + 4 mT^2/s), dsigma = pi F(x) h(pT^2) dx dpT^2 / q. chi is the page's
    (E* - p*_L)/sqrt(s) (= (q - x)/2)."""
    x, ix, dx = _sub(X_EDGES, 40)
    pt2, ipt, dpt2 = _sub(PT2_EDGES, 8)
    mt2 = M_K0 ** 2 + pt2

    def over_q(s):
        return np.sqrt(x[:, None] ** 2 + 4.0 * mt2[None, :] / s)

    q0 = over_q(s_of(E_GAMMA))
    # dsigma/dpT^2 at 20 GeV for h = 1, per bin of Table VI: h is what brings it to the table
    per_part = np.pi * np.sum(F_K0S[ix][:, None] * dx[:, None] / q0, axis=0)
    per_bin = np.array([per_part[ipt == j].mean() for j in range(DSDPT2_K0S.size)])
    h = (DSDPT2_K0S / per_bin)[ipt]
    q = over_q(s_of(e_gamma))
    w = np.pi * F_K0S[ix][:, None] * dx[:, None] * h[None, :] * dpt2[None, :] / q
    chi = numeric_chi_M4()(x[:, None], np.sqrt(mt2)[None, :], np.sqrt(s_of(e_gamma)))
    return chi.ravel(), np.repeat(x, pt2.size), w.ravel()


def mean_chi(e_gamma=E_GAMMA, x_below=None, x_above=None):
    """Mean chi of the K0_S at the photon energy e_gamma [GeV] and their share of all K0_S, for those with x
    below x_below and above x_above (all of them by default)."""
    chi, x, w = chi_samples(e_gamma)
    sel = np.ones(x.size, bool)
    if x_below is not None:
        sel &= x < x_below
    if x_above is not None:
        sel &= x >= x_above
    return float(np.sum(chi[sel] * w[sel]) / np.sum(w[sel])), float(np.sum(w[sel]) / np.sum(w))


def mean_chi_from_z():
    """Mean chi of the K0_S at 20 GeV from their distribution in z = E_K/E_gamma (Table V) and in pT^2 (Table
    VI), taken as independent: a K0_S of energy E in the rest frame of the proton, where it moves forward
    (x > -0.7), has chi = (E - p_L)/m_p with p_L = sqrt(E^2 - mT^2) (the page's chi in that frame).
    Returns (mean chi, K0_S per interaction in the table)."""
    pt2, ipt, dpt2 = _sub(PT2_EDGES, 8)
    h = DSDPT2_K0S[ipt] * dpt2
    mt2 = M_K0 ** 2 + pt2
    z, iz, dz = _sub(Z_EDGES, 200)
    num = den = 0.0
    for zz, j, d in zip(z, iz, dz):
        e = E_GAMMA * zz
        ok = e * e > mt2  # the pT^2 that a kaon of this energy can have
        if not ok.any():
            continue
        chi = (e - np.sqrt(e * e - mt2[ok])) / M_P
        w = DNDZ_K0S[j] * d * h[ok] / h[ok].sum()
        num += np.sum(chi * w)
        den += np.sum(w)
    return float(num / den), float(den)


def trend(e, e_low, low):
    """The level at the photon energy e [GeV] on the line in ln E through `low` at e_low and the four species at
    20 GeV."""
    high = species()
    low = np.asarray(low, float)
    return low + (high - low) * np.log(e / e_low) / np.log(E_GAMMA / e_low)


def ln_mean(channels, lo, hi):
    """Mean over the photon energies lo ... hi [GeV], with the weight 1/E, of the sum of the channels
    (eps_min, eps_max, values): each holds its values between eps_min and eps_max."""
    total = 0.0
    for emin, emax, values in channels:
        a, b = max(emin, lo), min(emax, hi)
        if b > a:
            total = total + np.asarray(values, float) * np.log(b / a)
    return total / np.log(hi / lo)
