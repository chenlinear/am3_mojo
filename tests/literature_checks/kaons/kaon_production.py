"""Derivation checks of the kaon production equations of
tests/literature_checks/kaons/README.md (its sections Production in
photo-hadronic interactions and Production in pp collisions). Sources, from
their arXiv TeX (equation and page numbers of these versions):
  - Hümmer, Rüger, Spanier & Winter (2010), arXiv:1002.1310v3: eqs. 3-8
    (pp. 6-7), 13-16 (p. 11), 27-28 (p. 15), 41-42 (p. 22) and Table 6 (p. 20).
    Their E_p, \\varepsilon, m_p, Gamma^IT, IT and N_p are the page's E_N,
    epsilon, m_p c^2, t_{c,N}^-1 (they set c = 1), channel c and N_N. Their
    lower limits eps_th are the page's 0: their cross sections vanish below.
    Their pion angle is from the nucleon (eq. 14), and the page's theta is from
    the photon:
    cos theta_pi = -cos theta.
  - Koers, Pe'er & Wijers (2006), arXiv:hep-ph/0611219v1: eqs. 16-18 (p. 7),
    Table III (p. 17), and the proton energies of their simulations (abstract:
    1 TeV to 1 PeV), where pp.fade's factor ends. Their "m c^2" (the meson's
    mass, energies in GeV) is the page's m_K. Their \\log is log10: they write
    \\ln elsewhere, and only with log10 does eq. 17a complete the square of
    eq. 17b.
  - Kelner, Aharonian & Bugayov (2006), arXiv:astro-ph/0606058v1: the threshold
    factor of F_pi (eqs. 6 and 12), the pp source (eqs. 70-71) and the
    delta-functional approximation (eqs. 75-77, where eq. 76 prints k for
    kappa).
  - Anticic et al. (NA49, 2010), arXiv:1004.1889v1, Table 11. Its points below
    100 GeV are Fesefeldt et al.'s (1979, Nucl. Phys. B 147, 317, Table 1). The
    page's multiplicity fit keeps their K0_S but not their K+-. Fesefeldt et al.
    derived these K+- by taking pp -> K+ K- X equal to pp -> K0 anti-K0 X
    (p. 324). The K+ fit has a second term, for the two K+ points at
    sqrt(s) = 2.9 GeV near threshold (kplus_term_results).
Masses are the code's (src/am3_mojo/utils/constants.mojo, which cites PDG 2026
for them). PDG 2026 gives the Sigma(1385)+ mass and A_L (K0_L listing, p. 34).

Kinematics come from Lorentz transformations. Integrals are redone in sympy for
any photon density and cross section.

Run from the repository root:
  .venv/bin/python tests/literature_checks/common/run.py kaon_production
"""

import itertools, math, re
from functools import lru_cache

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import least_squares

import latex
import mojo_constants
from latex import HUEMMER, KELNER, KOERS, NA49
from verify import (
    Entry,
    Result,
    derivation,
    exact,
    exact_fails,
    in_source,
    must_fail,
    rounds_to,
    worst_of,
)

L = latex.page("kaons")
PAGE = "kaons"

# this module's equations in the inventory (common/inventory.py)
INVENTORY = [
    Entry(
        "kp.eq4",
        r"\Gamma^{\rm IT}(E_p) = \frac{1}{2}\frac{m_p^2}{E_p^2}",
        "transcribed",
        "Hümmer et al. 2010, eq. 4",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.eq27",
        r"f^{\rm IT}(y) = \frac{1}{2y^2}\int_{\epsilon_{\rm th}}^{2y}",
        "transcribed",
        "Hümmer et al. 2010, eqs. 8 and 27",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.y",
        r"With $y = E_N \epsilon/(m_p c^2)$",
        "transcribed",
        "Hümmer et al. 2010, eq. 8",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.rate",
        r"t_{c,N}^{-1}(E_N) = c\int dy",
        "rearranged",
        "Hümmer et al. 2010, eqs. 3-4 with eq. 27",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.grid",
        r"t_{c,N}^{-1}(E_i) \simeq c\thinspace\Delta\sum_j",
        "derived",
        "kp.rate on AM3's logarithmic grids (the rectangle rule in ln eps)",
        file=PAGE,
    ),
    Entry(
        "kp.eq28",
        r"Q_b^{\rm IT} = N_p(E_b/\chi^{\rm IT})",
        "transcribed",
        "Hümmer et al. 2010, eq. 28",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.source",
        r"Q_K(E_K) = \sum_c \sum_{N=p,n}",
        "rearranged",
        "Hümmer et al. 2010, eq. 28 (from eqs. 7, 27)",
        "TeX " + HUEMMER,
        "tests/unit/test_added_processes.py (the code's channels "
        "against the data)",
        file=PAGE,
    ),
    Entry(
        "kp.source_grid",
        r"E_K\thinspace Q_K(E_K) = \sum_{c,N}",
        "derived",
        "kp.source per unit ln E_K",
        file=PAGE,
    ),
    Entry(
        "kp.split",
        r"the upper bin takes $r' = (e^{r\Delta} - 1)/(e^\Delta - 1)$",
        "model",
        "the weights of a source between two grid bins (number_split); what "
        "they and AM3's weight keep: derived",
        file=PAGE,
    ),
    Entry(
        "kp.fc",
        r"f_c(y) = \sigma_c [\min(4y^2, \epsilon_\mathrm{max}^2) - "
        r"\epsilon_\mathrm{min}^2]/(4y^2)",
        "derived",
        "Hümmer et al. 2010, eq. 27 with a constant cross section",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.eq16",
        r"\chi(\epsilon_r) = "
        r"\frac{s(\epsilon_r) - m_p^2 + m_\pi^2}{2s(\epsilon_r)}",
        "transcribed",
        "Hümmer et al. 2010, eqs. 16 and 5; the kaon replacements and the "
        "angle of Sec. 3.1",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.chi",
        r"\chi = \frac{s - m_Y^2 + m_K^2}{2s}\thinspace \big(1 - \beta^\ast "
        r"\cos\theta\big)",
        "derived",
        "Hümmer et al. 2010, eq. 16 (with eqs. 5, 14-15; masses replaced)",
        "TeX " + HUEMMER,
        "tests/unit/test_added_processes.py (the code's chi against the data)",
        file=PAGE,
    ),
    Entry(
        "kp.isotropic",
        r"the isotropic value that Hümmer et al. assume "
        r"($\langle\cos\theta\rangle = 0$)",
        "derived",
        "isotropic emission; chi linear and falling in cos theta (kp.chi); "
        "Hümmer et al. 2010's chi_K+ = 0.35",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.steps",
        r"| 0.91–1.05 | 1.31 μb, $\chi = 0.30$ |",
        "model",
        "am3_mojo: the steps of the measured cross sections and angles "
        "(tests/support/measured/ky_angular.py)",
        checks="tests/unit/test_added_processes.py (the code's table against "
        "the same data)",
        file=PAGE,
    ),
    Entry(
        "kp.rule4",
        r"fall as power laws fitted to their last GeV of data, "
        r"$\propto E^{-2.7}$ and $E^{-2.9}$",
        "model",
        "am3_mojo: power laws fitted to the last GeV of data "
        "(tests/support/measured/ky_angular.py)",
        file=PAGE,
    ),
    Entry(
        "kp.analogues",
        r"follows $\mathrm K^+\Sigma^0$ with 2.0",
        "model",
        "am3_mojo: ratios of the measured cross sections over the last "
        "0.6 GeV of data (tests/support/measured/ky_angular.py)",
        file=PAGE,
    ),
    Entry(
        "kp.huemmer_kplus",
        r"2.0 μb at 1.0–1.2 GeV and 3.7 μb at "
        r"1.2–1.65 GeV (their eqs. 41–42)",
        "transcribed",
        "Hümmer et al. 2010, eqs. 41-42 (their K+ channel KP)",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.table6",
        r"The types' photon ranges and cross sections are those of "
        r"their Table 6",
        "transcribed",
        "Hümmer et al. 2010, Table 6 (M7 up to AM3's top); the low parts' "
        "factors on chi_L: am3_mojo",
        "TeX " + HUEMMER,
        file=PAGE,
    ),
    Entry(
        "kp.mT",
        r"$m_{T,K}/m_{T,\pi} = 1.6$",
        "derived",
        "kinematics: E = m_T cosh y at rapidity y; PDG 2026 masses "
        "(constants.mojo)",
        file=PAGE,
    ),
    Entry(
        "kp.thresholds",
        r"The threshold of $\gamma\mathrm N \to \mathrm K\Lambda\pi$ is "
        r"1.152–1.168 GeV",
        "derived",
        "kinematics: the page's s at the summed masses "
        "(PDG 2026, constants.mojo)",
        file=PAGE,
    ),
    Entry(
        "kp.mpi_sigma",
        r"The $\mathrm K^+ + \mathrm K^0$ cross section of a row is the "
        r"type's cross section times its",
        "derived",
        "the products of the table's columns",
        file=PAGE,
    ),
    Entry(
        "kp.lead",
        r"leading part takes $h_- = 0.6\thinspace h_+$",
        "model",
        "am3_mojo: antikaons share no quark with the nucleon "
        "(the h_- of the multi-pion table)",
        file=PAGE,
    ),
    Entry(
        "kp.s_tr",
        r"s_{t,r} = 1 - \frac{\sum_{c \in t,\thinspace K}",
        "model",
        "am3_mojo (energy bookkeeping)",
        file=PAGE,
    ),
    Entry(
        "kp.chi_m4",
        r"$\chi = (E^\ast - p^\ast_L)/\sqrt{s}$",
        "derived",
        "kinematics: the light-cone share E - p_L of the nucleon's, "
        "boost invariant along the photon",
        file=PAGE,
    ),
    Entry(
        "kp.available",
        r"$E_A = \sqrt{s} - m_a - m_b$",
        "standard",
        "the kinetic energy available in the CM frame",
        file=PAGE,
    ),
    Entry(
        "pp.etrans",
        r"Above the transition energy $E_t$ = 100 GeV, AM³ makes pions",
        "model",
        "AM3 (ProtonProton.cc: E_trans, between the delta approximation and "
        "Kelner et al.'s F_pi); the kaons' E_t",
        file=PAGE,
    ),
    Entry(
        "pp.eq71",
        r"\Phi_\gamma(E_\gamma) = c\thinspace n_H\int_{E_\gamma}^\infty",
        "transcribed",
        "Kelner et al. 2006, eqs. 70-71",
        "TeX " + KELNER,
        file=PAGE,
    ),
    Entry(
        "pp.source",
        r"Q_K(E_K) = c\thinspace n_H\int \frac{dE_p}{E_p}",
        "transcribed",
        "Kelner et al. 2006, eq. 71 with F_K (AM3's pion source): "
        "the collisions counted",
        "TeX " + KELNER,
        checks="tests/unit/test_added_processes.py (test_pp_kaons_*)",
        file=PAGE,
    ),
    Entry(
        "pp.grid",
        r"E_i\thinspace Q_K(E_i) \simeq c\thinspace n_H\Delta\sum_j",
        "derived",
        "pp.source on AM3's logarithmic grid (the rectangle rule in ln E_p); "
        "kappa_K likewise",
        file=PAGE,
    ),
    Entry(
        "pp.koers_n",
        r"n(\epsilon) = n_0\thinspace \epsilon^{-1}",
        "transcribed",
        "Koers et al. 2006, eqs. 16, 17a-b",
        "TeX " + KOERS,
        file=PAGE,
    ),
    Entry(
        "pp.koers_beta",
        r"\beta = 10^{-(\epsilon/\epsilon_3)^{p_{30}}}",
        "transcribed",
        "Koers et al. 2006, eqs. 17c-e",
        "TeX " + KOERS,
        file=PAGE,
    ),
    Entry(
        "pp.koers_params",
        r"with $\epsilon_3 = 10^{p_2+p_4}$, $\epsilon_5 = 10^{p_2+p_6}$",
        "transcribed",
        "Koers et al. 2006, eqs. 18a-e, the definitions after "
        "eq. 17; Table III",
        "TeX " + KOERS,
        file=PAGE,
    ),
    Entry(
        "pp.neutral",
        r"$= \tfrac12(F_{\mathrm K^0} + F_{\bar{\mathrm K}^0})$",
        "standard",
        "K0 = (KS + KL)/sqrt 2; CP violation: PDG 2026, A_L of the "
        "K0_L listing",
        "PDF PDG-2026_listing_K-zero-L.pdf",
        file=PAGE,
    ),
    Entry(
        "pp.extension",
        r"F_K(x,E_p) = R(x)\thinspace \hat F_\pi(x,E_p)",
        "model",
        "am3_mojo extension",
        checks="tests/unit/test_added_processes.py (test_pp_kaons_*)",
        file=PAGE,
    ),
    Entry(
        "pp.scaling",
        r"$\mathrm K^+/\pi^0 = 0.123\text{–}0.126$",
        "derived",
        "Koers et al. 2006, eqs. 16-18 and Table III as the page gives them "
        "(Feynman scaling of the K/pi0 ratio)",
        "TeX " + KOERS,
        file=PAGE,
    ),
    Entry(
        "pp.low_x",
        r"$\to R(10^{-4})$ "
        r"$\rho(\min(xE_p, 10^{-4}E^\ast))/\rho(10^{-4}E^\ast)$",
        "model",
        "am3_mojo extension",
        checks="tests/unit/test_added_processes.py (test_pp_kaons_*)",
        file=PAGE,
    ),
    Entry(
        "pp.eq75",
        r"\tilde F_\pi = "
        r"\tilde n\thinspace\delta\negthinspace\left(E_\pi - "
        r"\frac{\kappa}{\tilde n}E_{\rm kin}\right)",
        "transcribed",
        "Kelner et al. 2006, eqs. 75-77 (eq. 76 prints k for kappa)",
        "TeX " + KELNER,
        file=PAGE,
    ),
    Entry(
        "pp.gamma_cm",
        r"$\gamma_\mathrm{cm} = \sqrt{(E_p + m_p)/(2m_p)}$ "
        r"$= (E_p + m_p)/\sqrt{s}$",
        "derived",
        "kinematics",
        file=PAGE,
    ),
    Entry(
        "pp.KK",
        r"$K_K = E_t\langle x\rangle/(E_t - m_p)$",
        "derived",
        "the mean fraction of T_p per kaon; Kelner et al. 2006's K_pi = "
        "kappa/n-tilde (eqs. 75-77)",
        file=PAGE,
    ),
    Entry(
        "pp.delta",
        r"\tilde n_K(E_p) = f(s)",
        "model",
        "am3_mojo (fit to NA49's Table 11: pp.fit); the delta approximation "
        "of Kelner et al. 2006, eqs. 75-77",
        "TeX " + KELNER,
        checks="tests/unit/test_added_processes.py (test_pp_kaons_*)",
        file=PAGE,
    ),
    Entry(
        "pp.split",
        r"$w = (E_K - E_l)/(E_{l+1} - E_l)$",
        "derived",
        "the kaons of a proton bin shared between two grid energies, number "
        "and energy kept",
        file=PAGE,
    ),
    Entry(
        "pp.loss",
        r"$\dot E/E = -n_H c \sigma_\mathrm{inel} \sum_\pi \kappa_\pi$",
        "model",
        "AM3; the kaons' term is the energy of the kaons of pp.source",
        checks="tests/unit/test_added_processes.py (test_pp_kaons_*)",
        file=PAGE,
    ),
    Entry(
        "pp.fit",
        r"| Kaon | $A$ | $a$ | $b$ | $C$ | $d$ | $\sqrt{s_\mathrm{th}}$ "
        r"[GeV] | Threshold channel |",
        "derived",
        "least squares to Anticic et al. 2010 (NA49), Table 11; "
        "the thresholds: kinematics",
        "TeX " + NA49,
        file=PAGE,
    ),
    Entry(
        "pp.fade",
        r"c_K(E_p) = c_t + (1 - "
        r"c_t)\thinspace\frac{\ln(E_p/E_t)}{\ln(E_1/E_t)}",
        "model",
        "am3_mojo: Koers et al.'s spectra joined to the fit (pp.fit) at E_t; "
        "Koers et al. 2006, abstract: their simulations start at 1 TeV",
        "TeX " + KOERS,
        file=PAGE,
    ),
]
# entry id -> the pins of the other relations of its display equation (each a
# whole relation as latex.parts cuts it)
ALSO = {
    "kp.eq27": (r"y = \frac{E_p\thinspace\varepsilon}{m_p}",),
    "kp.rate": (r"f_{c,N}(y) = \frac{1}{2y^2}\int_0^{2y}",),
    "kp.grid": (r"y_{ij} = E_i\epsilon_j/(m_p c^2)",),
    "kp.eq16": (r"s(\epsilon_r) = m_p^2 + 2m_p\epsilon_r",),
    "kp.chi": (r"s = m_N^2 + 2m_N\epsilon_r",),
    "pp.grid": (r"x_{ij} = E_i/E_j",),
    "pp.koers_n": (
        r"n_0 = 1.21\times 10^{\thinspace p_0 + p_1 p_2^2}",
        r"\alpha = \epsilon^{\thinspace p_1(\log\epsilon - 2p_2)}",
    ),
    "pp.koers_beta": (
        r"\gamma_1 = \tanh\negthinspace "
        r"\big(p_{70}\log(\epsilon/m_K)\big)",
        r"\gamma_2 = \tanh\negthinspace "
        r"\big(p_{80}\log(E_p/\epsilon)\big)",
    ),
    "pp.extension": (
        r"R(x) = \frac{E^\ast \thinspace n_K(x_s E^\ast;E^\ast)}{\hat "
        r"F_\pi(x_s,E^\ast)}",
        r"x_s = \max(x, 10^{-4})",
    ),
    "pp.eq75": (
        r"q_\pi(E_\pi) = \tilde n\thinspace\frac{c\thinspace "
        r"n_H}{K_\pi}",
        r"K_\pi = \kappa/\tilde n",
    ),
    "pp.delta": (
        r"f(s) = A\left(1-\frac{s_{\rm th}}{s}\right)^{a}\left(\frac{s}{s_{\rm "
        r"th}}\right)^{b}",
    ),
    "pp.fade": (r"c_t = \frac{f(s_t)}{\langle n_K\rangle_{\rm KPW}(E_t)}",),
    "pp.gamma_cm": (r"s = 2m_p^2 + 2m_pE_p",),
}


def braced(tex):
    """The text with single-letter \\mathrm arguments braced
    (\\mathrm K^+ -> \\mathrm{K}^+), as the parsers expect."""
    return re.sub(r"\\mathrm ([A-Za-z])", r"\\mathrm{\1}", tex)


E, eps, m, c, y, er, eth = sp.symbols(
    "E_N epsilon m_p c y epsilon_r epsilon_th", positive=True
)
EK, chi, M, x = sp.symbols("E_K chi M x", positive=True)
W, mN, mY, mK, Es, ps, u, w = sp.symbols(
    "W m_N m_Y m_K E_star p_star u w", positive=True
)
mu, ct = sp.symbols("mu c_theta", real=True)  # cos theta_pgamma, cos theta
n, G, sigma, N = (sp.Function(name) for name in ("n_gamma", "G", "sigma", "N"))
MC2 = m * c**2  # Hümmer's m_p


def rational(e):
    return sp.nsimplify(e, rational=True)


def piece(snippet, part):
    assert part in snippet, part
    return part


def integrand(expr):
    ints = [a for a in sp.Mul.make_args(expr) if isinstance(a, sp.Integral)]
    assert len(ints) == 1 and len(ints[0].limits) == 1, expr
    return (expr / ints[0]) * ints[0].function, ints[0].limits[0]


def with_control(results, what, old, new):
    out = results()
    with L.patched(old, new):
        out.append(must_fail(what, results()))
    return out


# --- Hümmer et al. (2010), TeX source (\drb = Gamma^IT, \dsb = sigma^IT,
#     \dcb = chi^IT) ---

H_EQ3 = (
    r"\drb (E_p) = \int d \varepsilon \int\limits_{-1}^{+1} "
    r"\frac{d \cos \theta_{p \gamma}}{2} \, (1- \cos \theta_{p \gamma}) "
    r"\, n_\gamma(\varepsilon,\cos \theta_{p \gamma}) \, \dsb(\epsilon_r)"
)
# sic: \cos_{p\gamma} for cos theta_pgamma
H_EPSR = r"$\epsilon_r=E_p \varepsilon/m_p (1 - \cos_{p \gamma})$"
H_ISO = (
    r"we will assume isotropy $n_\gamma(\varepsilon,\cos \theta_{p \gamma}) "
    r"\simeq n_\gamma(\varepsilon)$"
)
H_EQ4 = (
    r"\drb (E_p) = \frac{1}{2} \frac{m_p^2}{E_p^2} "
    r"\int\limits_{\frac{\epsilon_{\mathrm{th}} m_p}{2 E_p}}^{\infty} "
    r"d \varepsilon \frac{n_\gamma(\varepsilon)}{\varepsilon^2} "
    r"\int\limits_{\epsilon_{\mathrm{th}}}^{2 E_p \varepsilon/m_p} "
    r"d \epsilon_r \, \epsilon_r  \, \dsb(\epsilon_r)"
)
H_TH = (
    r"$\epsilon_{\mathrm{th}} \simeq 150 \, \mathrm{MeV}$ is the threshold "
    r"below which the cross sections are zero"
)
H_EQ5 = r"s(\epsilon_r) = m_p^2 + 2 \, m_p \, \epsilon_r"
H_EQ7 = (
    r"Q_b(E_b) = \int\limits_{E_b}^{\infty} \frac{dE_p}{E_p} \, N_p(E_p) \, "
    r"\int\limits_{\frac{\epsilon_{\mathrm{th}} m_p}{2 E_p}}^{\infty} "
    r"d\varepsilon \, n_\gamma(\varepsilon) \,  R_b( x,y )"
)
H_EQ8 = (
    r"x \equiv \frac{E_b}{E_p} \, , \quad  y\equiv "
    r"\frac{E_p \, \varepsilon}{m_p}"
)
H_EQ14 = (
    r"E_\pi=\gamma_\text{cm} E_\pi^\text{cm} (1+ "
    r"\beta_\pi^\text{cm}\cos{\theta_\pi^\text{cm}}) = E_p \, "
    r"\frac{E_\pi^\text{cm}}{\sqrt{s}}(1+\beta_\pi^\text{cm}"
    r"\cos{\theta_\pi^\text{cm}})"
)
H_BACK = (
    r"Note that backward scattering of the pion means that "
    r"$\cos{\theta_\pi^\text{cm}}=-1$"
)
H_EQ15 = r"E_{\pi}^\text{cm}=\frac{s-m_p^2+m_\pi^2}{2\sqrt{s}}"
H_EQ16 = (
    r"\chi(\epsilon_r)=  \frac{s(\epsilon_r)-m_p^2+m_\pi^2}{2 s(\epsilon_r)} "
    r"(1+\beta_\pi^{\mathrm{cm}} \cos{\theta_\pi})"
)
H_KAON = (
    r"we also use Eq.~(\ref{eqn:1pion}) for the kaon production (with "
    r"the rough estimate $\left<\cos{\theta_\pi}\right> \simeq 0$), "
    r"where the replacements $m_p\rightarrow m_\Lambda$ and "
    r"$m_\pi\rightarrow m_K$ have to be made"
)
H_CHI35 = (
    r"we have $\chi_{K^+} \simeq 0.35$ (computed for "
    r"$\epsilon_r \simeq 1.4 \, \mathrm{GeV} $ at the peak)"
)
H_EQ27 = (
    r"R^{\mathrm{IT}}(x,y)= \delta(x - \chi^{\mathrm{IT}}) \,  "
    r"M_b^{\mathrm{IT}} \, f^{\mathrm{IT}}(y)  \quad \text{with} \quad "
    r"f^{\mathrm{IT}}(y) \equiv \frac{1}{2y^2} "
    r"\int\limits_{\epsilon_{\mathrm{th}}}^{2y} d \epsilon_r \, "
    r"\epsilon_r \, \sigma^{\mathrm{IT}}(\epsilon_r)"
)
H_EQ28 = (
    r"Q_b^{\mathrm{IT}} = N_p \left( \frac{E_b}{\chi^{\mathrm{IT}}} \right) "
    r"\frac{m_p}{E_b} \int\limits_{\epsilon_{\mathrm{th}}/2}^\infty dy \, "
    r"n_\gamma \left( \frac{m_p \, y \, \chi^{\mathrm{IT}}}{E_b} \right) \,  "
    r"M_b^{\mathrm{IT}}  \, f^{\mathrm{IT}}(y)"
)

# Hümmer et al.'s K+ channel KP, eqs. 41-42 (its cross section, from which
# eq. 27 gives the response)
H_KP = [
    r"2.0 \, \mu \mathrm{barn}  & 1.0 \, \mathrm{GeV}\le \epsilon_r < 1.2 "
    r"\, \mathrm{GeV}\\",
    r"3.7 \, \mu \mathrm{barn}   & 1.2 \, \mathrm{GeV}\le  "
    r"\epsilon_r < 1.65 \, \mathrm{GeV}\\",
    r"2.7 \, \mu \mathrm{barn}    & \epsilon_r \ge 1.65 \, \mathrm{GeV} \\",
]
# Hümmer et al.'s Table 6: IT, eps_min, eps_max [GeV], sigma [mubarn], chi, K,
# then the pion multiplicities
H_T6 = [
    r"M$_{\text{1L}}$ & 0.5 & 0.9 & 60 & 0.1 & 0.27  & 0.32 & 0.34 & 0.04 "
    r"& 0.32 & 0.04 & 0.34 \\",
    r"M$_\text{1H}$ & 0.5 & 0.9 & 60 &  0.4 & -- & 0.17 & 0.29 & 0.05 & "
    r"0.17 & 0.05 & 0.29\\",
    r"M$_\text{2L}$ & 0.9& 1.5 & 85 & 0.15 & 0.34 & 0.42& 0.31& 0.07 & "
    r"0.42& 0.07& 0.31\\",
    r"M$_\text{2H}$ & 0.9& 1.5& 85 & 0.35&-- & 0.19& 0.35& 0.08& 0.19& "
    r"0.08& 0.35\\",
    r"M$_\text{3L}$& 1.5& 5.0& 120 & 0.15& 0.39 & 0.59& 0.57& 0.30&   "
    r"0.59& 0.30& 0.57\\",
    r"M$_\text{3H}$& 1.5&  5.0& 120 & 0.35&--& 0.16& 0.21& 0.13& 0.16& "
    r"0.13& 0.21 \\",
    r"M$_\text{4L}$& 5.0&   50& 120& 0.07& 0.49 & 1.38& 1.37& 1.11& 1.38& "
    r"1.11& 1.37\\",
    r"M$_\text{4H}$& 5.0& 50& 120 & 0.35& --& 0.16& 0.25& 0.23&0.16& "
    r"0.23& 0.25\\",
    r"M$_\text{5L}$& 50& 500& 120 & 0.02& 0.45 & 3.01& 2.86& 2.64&3.01& "
    r"2.64& 2.86\\",
    r"M$_\text{5H}$& 50& 500& 120 & 0.5&-- & 0.20& 0.21& 0.14&0.20& "
    r"0.14& 0.21\\",
    r"M$_\text{6L}$& 500& 5000& 120 & 0.007& 0.44 & 5.13& 4.68& 4.57&5.13& "
    r"4.57& 4.68\\",
    r"M$_\text{6H}$& 500& 5000& 120 & 0.5& --& 0.27& 0.29& 0.12&0.27& "
    r"0.12& 0.29\\",
    r"M$_\text{7L}$& 5000& $\infty$ & 120 & 0.002& 0.44 & 7.59& 6.80& "
    r"6.65&7.59& 6.65& 6.80\\",
    r"M$_\text{7H}$& 5000& $\infty$ & 120 & 0.6& --& 0.26& 0.27& 0.13 & "
    r"0.26& 0.13& 0.27\\",
]

# --- the page's quotes of Hümmer et al.: eqs. 4, 8, 27, 28, 16 and 5, with
#     Hümmer's m_p (a rest energy, c = 1) as m_p c^2 and their E_p as E_N, so
#     that the page's own forms follow by substitution ---

# the displays of the quotes, by a start that the controls leave intact
EQ4_PIN, F27_PIN, EQ28_PIN, EQ16_PIN = (
    r"\Gamma^{\rm IT}(E_p) = ",
    r"f^{\rm IT}(y) = ",
    r"Q_b^{\rm IT} = ",
    r"\chi(\epsilon_r) = ",
)
# any response function, for comparing the quote of eq. 28 with the TeX
fIT = sp.Function("f_IT")
TOK_PAGE = {
    "E_p": E,
    "m_p": MC2,
    r"\varepsilon": eps,
    r"\epsilon_r": er,
    r"\epsilon_{\rm th}": eth,
    "y": y,
}
TOK_TEX = {
    "E_p": E,
    "m_p": MC2,
    r"\varepsilon": eps,
    r"\epsilon_r": er,
    r"\epsilon_{\mathrm{th}}": eth,
    "y": y,
}


def eq4_pieces(page=True):
    if page:
        rhs = L.sides(L.display(EQ4_PIN))[1]
        pre, lo, outer, ilo, ihi, inner = re.fullmatch(
            r"(.+?)\\int_\{(.+?)\}\^\\infty d\\varepsilon\\thinspace "
            r"(.+?)\\int_\{(.+?)\}\^\{(.+?)\} d\\epsilon_r\\thinspace (.+)",
            rhs,
        ).groups()
        tok, fn = TOK_PAGE, {r"\sigma^{\rm IT}": sigma, r"n_\gamma": n}
    else:
        pre, lo, outer, ilo, ihi, inner = (
            piece(H_EQ4, s_)
            for s_ in (
                r"\frac{1}{2} \frac{m_p^2}{E_p^2}",
                r"\frac{\epsilon_{\mathrm{th}} m_p}{2 E_p}",
                r"\frac{n_\gamma(\varepsilon)}{\varepsilon^2}",
                r"\epsilon_{\mathrm{th}}",
                r"2 E_p \varepsilon/m_p",
                r"\epsilon_r  \, \dsb(\epsilon_r)",
            )
        )
        tok, fn = TOK_TEX, {r"\dsb": sigma, r"n_\gamma": n}
    return {
        k: L.to_sympy(v, tok, funcs=fn)
        for k, v in zip(
            ("pre", "lo", "outer", "ilo", "ihi", "inner"),
            (pre, lo, outer, ilo, ihi, inner),
        )
    }


def eq27_pieces(page=True):
    if page:
        disp = L.display(F27_PIN)
        y_tex = L.sides(L.part(disp, "y ="))[1]
        pre, lo, hi, body = re.fullmatch(
            r"(.+?)\\int_\{(.+?)\}\^\{(.+?)\} d\\epsilon_r\\thinspace (.+)",
            L.sides(L.part(disp, r"f^{\rm IT}(y) ="))[1],
        ).groups()
        tok, fn = TOK_PAGE, {r"\sigma^{\rm IT}": sigma}
    else:
        y_tex = piece(H_EQ8, r"\frac{E_p \, \varepsilon}{m_p}")
        pre, lo, hi, body = (
            piece(H_EQ27, s_)
            for s_ in (
                r"\frac{1}{2y^2}",
                r"\epsilon_{\mathrm{th}}",
                "2y",
                r"\epsilon_r \, \sigma^{\mathrm{IT}}(\epsilon_r)",
            )
        )
        tok, fn = TOK_TEX, {r"\sigma^{\mathrm{IT}}": sigma}
    return {
        k: L.to_sympy(v, tok, funcs=fn)
        for k, v in zip(
            ("y", "pre", "lo", "hi", "body"), (y_tex, pre, lo, hi, body)
        )
    }


def eq27_fG(arg):
    q = eq27_pieces()
    assert q["lo"] == eth and q["body"] == er * sigma(er)
    return (q["pre"] * G(q["hi"])).subs(y, arg)


def eq28_pieces(page=True, f=fIT):
    if page:
        rhs = L.sides(L.display(EQ28_PIN))[1]
        head, lo, body = re.fullmatch(
            r"(.+?)\\int_\{(.+?)\}\^\\infty dy\\thinspace (.+)", rhs
        ).groups()
        tok = {"E_b": EK, r"\chi^{\rm IT}": chi, r"M_b^{\rm IT}": M, **TOK_PAGE}
        fn = {"N_p": N, r"n_\gamma": n, r"f^{\rm IT}": f}
    else:
        head, lo, body = (
            piece(H_EQ28, s_)
            for s_ in (
                r"N_p \left( \frac{E_b}{\chi^{\mathrm{IT}}} \right) "
                r"\frac{m_p}{E_b}",
                r"\epsilon_{\mathrm{th}}/2",
                r"n_\gamma \left( \frac{m_p \, y \, \chi^{\mathrm{IT}}}{E_b} "
                r"\right) \,  M_b^{\mathrm{IT}}  \, f^{\mathrm{IT}}(y)",
            )
        )
        tok = {
            "E_b": EK,
            r"\chi^{\mathrm{IT}}": chi,
            r"M_b^{\mathrm{IT}}": M,
            **TOK_TEX,
        }
        fn = {"N_p": N, r"n_\gamma": n, r"f^{\mathrm{IT}}": f}
    return (
        L.to_sympy(head, tok, funcs=fn),
        L.to_sympy(lo, tok),
        L.to_sympy(body, tok, funcs=fn),
    )


def quote_results(what, page, tex):
    keys = page.keys() if isinstance(page, dict) else range(len(page))
    bad = [str(k) for k in keys if sp.simplify(page[k] - tex[k]) != 0]
    return Result(
        what,
        not bad,
        "exact: every piece equal"
        if not bad
        else f"pieces differ: {', '.join(bad)}",
    )


def RATE():
    return L.display(r"t_{c,N}^{-1}(E_N) = c\int dy")


def readme_y():
    return L.to_sympy(
        L.sides(L.inline(r"y = E_N \epsilon/(m_p c^2)"))[1],
        {"E_N": E, r"\epsilon": eps, "m_p": m, "c": c},
    )


def readme_f_parts():
    lhs, rhs = L.sides(L.part(RATE(), "f_{c,N}(y)"))
    pre, integral = rhs.split(r"\int", 1)
    lo, up, body = re.fullmatch(
        r"_(\w+)\^\{(.+?)\} (.+)\\thinspace d\\epsilon_r", integral
    ).groups()
    assert (
        lhs == "f_{c,N}(y)"
        and body == r"\epsilon_r\thinspace \sigma_{c,N}(\epsilon_r)"
    ), rhs
    return (
        L.to_sympy(pre, {"y": y}),
        L.to_sympy(lo, {}),
        L.to_sympy(up, {"y": y}),
        L.to_sympy(body, {r"\epsilon_r": er}, funcs={r"\sigma_{c,N}": sigma}),
    )


def readme_rate(EN=E):
    lhs, rhs = L.sides(L.part(RATE(), r"t_{c,N}^{-1}(E_N)"))
    head, body = rhs.split(r"\int dy\thinspace ")
    pre, lo, up, _ = readme_f_parts()
    assert lhs == r"t_{c,N}^{-1}(E_N)" and lo == 0
    fG = lambda a: (pre * G(up)).subs(y, a)
    return (
        L.to_sympy(head, {"c": c}),
        L.to_sympy(
            body,
            {"m_p": m, "c": c, "E_N": EN, "y": y},
            funcs={r"n_\gamma": n, "f_{c,N}": fG},
        ),
    )


def lab_share(Estar, pLstar):
    """E_K/E_N in the lab, for a nucleon along +z of rapidity ln u and a photon
    along -z. The photon makes the invariant mass W. The kaon has the CM energy
    Estar and the momentum pLstar along the photon."""
    EN_, PN_ = mN * (u + 1 / u) / 2, mN * (u - 1 / u) / 2
    k = (W**2 - mN**2) / (2 * mN) / u
    assert sp.simplify((EN_ + k) ** 2 - (PN_ - k) ** 2 - W**2) == 0
    e_eta = (EN_ + PN_ + k - k) / W
    # p*_z = -pLstar
    EK_ = (e_eta * (Estar - pLstar) + (Estar + pLstar) / e_eta) / 2
    return EK_ / EN_


@derivation("kp.eq4")
def check_eq4():
    """The page's eq. 4 is Hümmer et al.'s, piece by piece (prefactor, limits,
    integrands)."""

    def results():
        return [
            quote_results(
                "the page's eq. 4 = the TeX's (prefactor, "
                "both integrals' limits and integrands)",
                eq4_pieces(),
                eq4_pieces(page=False),
            )
        ]

    return [
        in_source(
            "eqs. 3 and 4, isotropy, eps_r and the threshold as transcribed",
            HUEMMER,
            H_EQ3,
            H_EPSR,
            H_ISO,
            H_EQ4,
            H_TH,
        )
    ] + with_control(
        results,
        "control: n_gamma/eps for n_gamma/eps^2 in the quote",
        r"\frac{n_\gamma(\varepsilon)}{\varepsilon^2}",
        r"\frac{n_\gamma(\varepsilon)}{\varepsilon}",
    )


@derivation("kp.eq27")
def check_eq27():
    """The page's eqs. 8 and 27 are Hümmer et al.'s, and the inner integral of
    eq. 4 is 2 y^2 f^IT(y).
    eps = m_p y/E_p inverts y."""

    def results():
        q4, q27 = eq4_pieces(), eq27_pieces()
        inner = sp.Integral(q4["inner"], (er, q4["ilo"], q4["ihi"]))
        two_y2_f = L.to_sympy(
            L.inline(r"2y^2 f^{\rm IT}(y)"),
            {"y": y},
            funcs={r"f^{\rm IT}": lambda a: eq27_fG(a)},
        )
        G_inner = inner.replace(
            lambda a: (
                isinstance(a, sp.Integral)
                and a.function == er * sigma(er)
                and a.limits[0][1] == eth
            ),
            lambda a: G(a.limits[0][2]),
        )
        eps_of_y = L.to_sympy(
            L.sides(L.inline(r"\varepsilon = m_p y/E_p"))[1], TOK_PAGE
        )
        return [
            quote_results(
                "the page's eqs. 8 and 27 = the TeX's "
                "(y; f^IT's prefactor, limits, integrand)",
                q27,
                eq27_pieces(page=False),
            ),
            exact(
                "the inner integral of eq. 4 (from eps_th, eq. 27's G) "
                "at y = E_p eps/m_p is 2 y^2 f^IT(y)",
                G_inner - two_y2_f.subs(y, q27["y"]),
            ),
            exact(
                "eps = m_p y/E_p inverts y = E_p eps/m_p",
                eps_of_y - sp.solve(q27["y"] - y, eps)[0],
            ),
        ]

    return [
        in_source("eqs. 8 and 27 as transcribed", HUEMMER, H_EQ8, H_EQ27)
    ] + with_control(
        results,
        "control: up to y for 2y in the quote of eq. 27",
        r"\int_{\epsilon_{\rm th}}^{2y} d\epsilon_r",
        r"\int_{\epsilon_{\rm th}}^{y} d\epsilon_r",
    )


@derivation("kp.y")
def check_y():
    """y = E_N eps/(m_p c^2) is Hümmer et al.'s eq. 8, and
    eps_r = y (1 - cos theta_pgamma) ranges over (0, 2y)."""
    ours = readme_y()
    # the page's eq. 8 (E_p -> E_N, m_p -> m_p c^2), the TeX's by kp.eq27
    theirs = eq27_pieces()["y"]
    PN_ = sp.sqrt(E**2 - MC2**2)
    epsr = (E * eps - PN_ * eps * mu) / MC2
    lim = sp.limit(epsr - ours * (1 - mu), E, sp.oo)
    return [
        in_source(
            "eq. 8 and the definition of eps_r after eq. 3 as transcribed",
            HUEMMER,
            H_EQ8,
            H_EPSR,
        ),
        exact("README y = eq. 8 (E_p -> E_N, m_p -> m_p c^2)", ours - theirs),
        Result(
            "eps_r = p_N.k/m_N -> y (1 - cos theta_pgamma) for beta_N "
            "-> 1 (Hümmer's eps_r; at most 2y)",
            lim == 0,
            f"limit of the difference: {lim}",
        ),
    ]


@derivation("kp.rate")
def check_rate():
    """The README's t_{c,N}^-1 is Hümmer et al.'s eq. 4 (eq. 3 for isotropic
    photons) with f of eq. 27."""
    y_of_eps = readme_y()
    pre, lo, up, body = readme_f_parts()
    epsr = L.to_sympy(
        H_EPSR.strip("$").split("=")[1],
        {"E_p": E, r"\varepsilon": eps, "m_p": MC2, r"\cos_{p \gamma}": mu},
    )
    avg = sp.Integral((1 - mu) / 2 * sigma(epsr), (mu, -1, 1)).transform(
        mu, (sp.solve(epsr - er, mu)[0], er)
    )
    ours = (pre * sp.Integral(body, (er, lo, up))).subs(y, y_of_eps)
    (a, la), (b, lb) = integrand(avg), integrand(ours)
    out = [
        in_source("eq. 3 and eps_r as transcribed", HUEMMER, H_EQ3, H_EPSR),
        Result(
            "eq. 3, isotropic photons: (1/2) int dcos (1 - cos) sigma = "
            "README f_{c,N}(y), y = E_N eps/(m_p c^2) (limits)",
            la == lb,
            f"{la} and {lb}",
        ),
        exact(
            "eq. 3, isotropic photons: the angle average = README f_{c,N} "
            "(integrands)",
            a - b,
        ),
    ]
    # the page's eq. 4 (the TeX's by kp.eq4), its inner integral
    # G(u) = int_eps_th^u eps_r sigma = int_0^u (sigma = 0 below eps_th)
    q = eq4_pieces()
    out.append(
        Result(
            "the inner integral of the page's eq. 4 is eq. 27's G: "
            "from eps_th, of eps_r sigma(eps_r)",
            q["ilo"] == eth and q["inner"] == er * sigma(er),
            f"{q['ilo']}, {q['inner']}",
        )
    )
    e4 = (
        c
        * q["pre"]
        * sp.Integral(q["outer"] * G(q["ihi"]), (eps, q["lo"], sp.oo))
    )
    in_y = e4.replace(
        lambda a: isinstance(a, sp.Integral),
        lambda a: a.transform(eps, (sp.solve(y_of_eps - y, eps)[0], y)),
    )
    (h, lim), (head, body_y) = integrand(in_y), readme_rate()
    out += [
        exact(
            "eq. 4 at eps = m_p c^2 y/E_N (times c) = README t^-1 integrand",
            h - head * body_y,
        ),
        Result(
            "eq. 4's lower limit eps_th m_p/(2E_p) is y = eps_th/2, "
            "below which f = 0 (README: all y)",
            lim[1] == eth / 2,
            f"y from {lim[1]}",
        ),
    ]
    out.append(
        exact(
            "README f_{c,N}(y) = eq. 27's f^IT(y) "
            "(lower limit 0 for eps_th: sigma = 0 below it)",
            pre * G(up) - eq27_fG(y),
        )
    )
    with L.patched(
        r"\frac{m_p c^2}{E_N}\thinspace n_\gamma",
        r"\frac{E_N}{m_p c^2}\thinspace n_\gamma",
    ):
        head_b, body_b = readme_rate()
    out.append(
        exact_fails(
            "control: E_N/(m_p c^2) for m_p c^2/E_N in front of n_gamma",
            h - head_b * body_b,
        )
    )
    return out


GRID_PIN = r"t_{c,N}^{-1}(E_i) \simeq c\thinspace\Delta\sum_j"
E0, e0, Dl = sp.symbols("E_0 epsilon_0 Delta", positive=True)
ii, jj = sp.symbols("i j", integer=True)


def readme_grid():
    disp = L.display(GRID_PIN)
    lhs, rhs = (
        s_.strip() for s_ in L.part(disp, r"t_{c,N}^{-1}(E_i)").split(r"\simeq")
    )
    factor, summand = rhs.split(r"\sum_j ")
    Ei, Y = sp.symbols("E_i Y", positive=True)
    y_ij = L.to_sympy(
        L.sides(L.part(disp, "y_{ij} ="))[1],
        {"E_i": Ei, r"\epsilon_j": eps, "m_p": m, "c": c},
    )
    pre = L.to_sympy(factor, {"c": c, r"\Delta": Dl})
    body = L.to_sympy(
        summand,
        {"y_{ij}": Y, r"\epsilon_j": eps},
        funcs={"f_{c,N}": sp.Function("F"), r"n_\gamma": n},
    )
    assert lhs == r"t_{c,N}^{-1}(E_i)", lhs
    return (pre / Dl * body).subs(Y, y_ij).subs(Ei, E), y_ij.subs(Ei, E)


@derivation("kp.grid")
def check_grid():
    """On AM3's logarithmic grids the rate is the rectangle rule of its integral
    in ln eps.
    y_ij depends on i + j only, so the code keeps one vector per channel."""
    Fc = sp.Function("F")
    head, _ = readme_rate()
    _, rhs = L.sides(L.part(RATE(), r"t_{c,N}^{-1}(E_N)"))
    _, integrand_tex = rhs.split(r"\int dy\thinspace ")
    rate_y = L.to_sympy(
        integrand_tex,
        {"m_p": m, "c": c, "E_N": E, "y": y},
        funcs={r"n_\gamma": n, "f_{c,N}": Fc},
    )
    sub = L.to_sympy(
        L.sides(L.inline(r"\epsilon = m_p c^2 y/E_N"))[1],
        {"m_p": m, "c": c, "y": y, "E_N": E},
    )
    yv = sp.solve(sub - eps, y)[0]
    per_ln_eps = head * (rate_y * sp.diff(yv, eps)).subs(y, yv) * eps

    def results():
        grid, y_ij = readme_grid()
        grid = grid.replace(sp.Function("F"), Fc)
        Ei = E0 * sp.exp(ii * Dl)
        ej = e0 * sp.exp(jj * Dl)
        y_at = y_ij.subs({E: Ei, eps: ej})
        return [
            exact(
                "summand/Delta = the integrand in ln eps at eps_j "
                "(eps = m_p c^2 y/E_N, dy m_p c^2/E_N = eps dln eps)",
                grid - per_ln_eps,
            ),
            exact(
                "y_ij at (i + 1, j - 1) = y_ij: it depends on i + j only",
                sp.expand(y_at.subs({ii: ii + 1, jj: jj - 1}) - y_at),
            ),
            exact(
                "E_i eps_j grows by e^Delta with each step of i + j",
                sp.simplify(y_at.subs(ii, ii + 1) / y_at)
                - sp.exp(
                    L.to_sympy(
                        L.inline(r"e^\Delta").split("^")[1], {r"\Delta": Dl}
                    )
                ),
            ),
        ]

    return with_control(
        results,
        "control: n_gamma(eps_j) without eps_j in the grid form",
        r"f_{c,N}(y_{ij})\thinspace \epsilon_j n_\gamma(\epsilon_j)",
        r"f_{c,N}(y_{ij})\thinspace n_\gamma(\epsilon_j)",
    )


@derivation("kp.eq28")
def check_eq28():
    """The page's eq. 28 is Hümmer et al.'s, and at E_p = E_b/chi its integral
    is the rate of eq. 4 over y."""

    def results():
        page, tex = eq28_pieces(), eq28_pieces(page=False)
        head, lo, body = page
        Ep_at = L.to_sympy(
            L.sides(L.inline(r"E_p = E_b/\chi^{\rm IT}"))[1],
            {"E_b": EK, r"\chi^{\rm IT}": chi},
        )
        Epp = sp.Symbol("E_pp", positive=True)
        at = {EK: sp.solve(Ep_at - Epp, EK)[0]}
        arg = [a.args[0] for a in body.atoms(sp.Function) if a.func == n][0]
        photon = L.to_sympy(
            L.inline(r"m_p y/E_p"), {"m_p": MC2, "y": y, "E_p": Epp}
        )
        front = L.to_sympy(
            L.inline(r"(m_p/E_p)/\chi^{\rm IT}"),
            {"m_p": MC2, "E_p": Epp, r"\chi^{\rm IT}": chi},
        )
        f_low = eq27_pieces()["hi"]
        y_low = L.to_sympy(
            L.sides(L.inline(r"y = \epsilon_{\rm th}/2"))[1], TOK_PAGE
        )
        return [
            quote_results(
                "the page's eq. 28 = the TeX's (the factor in "
                "front, the lower limit, the integrand)",
                page,
                tex,
            ),
            exact(
                "at E_p = E_b/chi the photon energy m_p y chi/E_b is m_p y/E_p",
                arg.subs(at) - photon,
            ),
            exact(
                "and the factor m_p/E_b in front is (m_p/E_p)/chi",
                (head / N(EK / chi)).subs(at) - front,
            ),
            exact(
                "the lower limit eps_th/2 is the y at which f^IT's upper "
                "limit 2y reaches eps_th",
                lo - sp.solve(f_low - eth, y)[0],
            ),
            exact(
                "so f^IT vanishes below the page's y = eps_th/2",
                y_low - sp.solve(f_low - eth, y)[0],
            ),
        ]

    return [
        in_source(
            "eqs. 7, 8, 27 and 28 as transcribed",
            HUEMMER,
            H_EQ7,
            H_EQ8,
            H_EQ27,
            H_EQ28,
        )
    ] + with_control(
        results,
        "control: m_p/E_b^2 in the quote",
        r"\thinspace \frac{m_p}{E_b}\int",
        r"\thinspace \frac{m_p}{E_b^2}\int",
    )


@derivation("kp.source")
def check_source():
    """Q_K is Hümmer et al.'s eq. 28 (from their eqs. 7 and 27), summed over
    channels and nucleons."""
    pin = r"Q_K(E_K) = \sum_c \sum_{N=p,n}"
    lhs, rhs = L.sides(L.parts(L.display(pin))[0])
    assert lhs == "Q_K(E_K)" and rhs == (
        r"\sum_c \sum_{N=p,n} \frac{M_{c,N}^K}{\chi_{c,N}^K}\thinspace "
        r"\Big[N_N\thinspace t_{c,N}^{-1}\Big]_ {E_N = E_K/\chi_{c,N}^K}"
    ), rhs
    tok = {"M_{c,N}^K": M, r"\chi_{c,N}^K": chi, "E_K": EK}
    factor = L.to_sympy(
        L.between(rhs, r"\sum_{N=p,n} ", r"\thinspace \Big["), tok
    )
    nn, tt = sp.symbols("N_N t_inv")
    bracket = L.to_sympy(
        L.between(rhs, r"\Big[", r"\Big]"), {"N_N": nn, "t_{c,N}^{-1}": tt}
    )

    def term():
        rhs_ = L.sides(L.parts(L.display(pin))[0])[1]
        at = L.sides(rhs_.split(r"\Big]_ {")[1][:-1])
        assert at[0] == "E_N"
        EN_at = L.to_sympy(at[1], tok)
        head, body = readme_rate(EN_at)
        return sp.expand(factor * bracket.subs({nn: N(EN_at), tt: head}) * body)

    ours = term()
    # the page's eq. 28 (the TeX's by kp.eq28, times c because c = 1 there)
    head28, _, body28 = eq28_pieces(f=eq27_fG)
    # eqs. 7 and 27 with E_p and eps from 0 (the delta puts E_p = E_b/chi > E_b,
    # and f = 0 below eps_th)
    Ep = sp.Symbol("E_p", positive=True)
    xv = L.to_sympy(piece(H_EQ8, r"\frac{E_b}{E_p}"), {"E_b": EK, "E_p": Ep})
    yv = L.to_sympy(
        piece(H_EQ8, r"\frac{E_p \, \varepsilon}{m_p}"),
        {"E_p": Ep, r"\varepsilon": eps, "m_p": MC2},
    )
    over_Ep = sp.Integral(
        N(Ep) / Ep * sp.DiracDelta(xv - chi) * M * eq27_fG(yv), (Ep, 0, sp.oo)
    )
    done = over_Ep.transform(Ep, (EK / x, x)).doit()
    q7 = sp.Integral(n(eps) * done, (eps, 0, sp.oo)).transform(
        eps, (sp.solve(yv.subs(Ep, EK / chi) - y, eps)[0], y)
    )
    from7, _ = integrand(q7)
    with L.patched(
        r"\Big[N_N\thinspace t_{c,N}^{-1}\Big]_ {E_N = E_K/\chi_{c,N}^K}",
        r"\Big[N_N\thinspace t_{c,N}^{-1}\Big]_ {E_N = E_K\chi_{c,N}^K}",
    ):
        broken = term()
    return [
        in_source(
            "eqs. 7, 8, 27 and 28 as transcribed",
            HUEMMER,
            H_EQ7,
            H_EQ8,
            H_EQ27,
            H_EQ28,
        ),
        exact(
            "eqs. 7 and 27 (the delta integrated over E_p, eps -> y) "
            "give eq. 28 (integrands)",
            from7 - head28 * body28,
        ),
        exact(
            "README (M/chi) [N_N t^-1] at E_N = E_K/chi = c x eq. 28 "
            "(N_p -> N_N, IT -> channel c)",
            ours - c * head28 * body28,
        ),
        exact_fails(
            "control: E_N = E_K chi in the README", broken - c * head28 * body28
        ),
    ]


SOURCE_GRID_PIN = r"E_K\thinspace Q_K(E_K) = \sum_{c,N}"


def readme_source_grid():
    lhs, rhs = L.sides(L.parts(L.display(SOURCE_GRID_PIN))[0])
    assert lhs == r"E_K\thinspace Q_K(E_K)", lhs
    tok = {"M_{c,N}^K": M, r"\chi_{c,N}^K": chi, "E_K": EK}
    factor = L.to_sympy(
        L.between(rhs, r"\sum_{c,N} ", r"\thinspace \Big["), tok
    )
    EN_at = L.to_sympy(L.sides(rhs.split(r"\Big]_ {")[1][:-1])[1], tok)
    nn, tt = sp.symbols("N_N t_inv")
    bracket = L.to_sympy(
        L.between(rhs, r"\Big[", r"\Big]"),
        {"E_N": E, "N_N": nn, "t_{c,N}^{-1}": tt},
    )
    return factor * bracket.subs(
        {E: EN_at, nn: N(EN_at), tt: sp.Function("t")(EN_at)}
    )


@derivation("kp.source_grid")
def check_source_grid():
    """E_K Q_K, the source per unit ln E_K, is M [E_N N_N t^-1] at
    E_N = E_K/chi: Q_K times E_K = chi E_N."""
    pin = r"Q_K(E_K) = \sum_c \sum_{N=p,n}"
    rhs = L.sides(L.parts(L.display(pin))[0])[1]
    tok = {"M_{c,N}^K": M, r"\chi_{c,N}^K": chi, "E_K": EK}
    factor = L.to_sympy(
        L.between(rhs, r"\sum_{N=p,n} ", r"\thinspace \Big["), tok
    )
    EN_at = L.to_sympy(L.sides(rhs.split(r"\Big]_ {")[1][:-1])[1], tok)
    term = factor * N(EN_at) * sp.Function("t")(EN_at)
    EK_of = L.to_sympy(
        L.sides(L.inline(r"E_K = \chi_{c,N}^K E_N"))[1],
        {r"\chi_{c,N}^K": chi, "E_N": E},
    )

    def results():
        return [
            exact(
                "E_K times the page's Q_K term = the grid form's term "
                "(any N_N, t^-1)",
                EK * term - readme_source_grid(),
            ),
            exact(
                "E_K = chi E_N is the E_N of the bracket, E_K/chi",
                EK_of.subs(E, EN_at) - EK,
            ),
        ]

    return with_control(
        results,
        "control: [N_N t^-1] without E_N in the grid form",
        r"\Big[E_N N_N\thinspace t_{c,N}^{-1}\Big]",
        r"\Big[N_N\thinspace t_{c,N}^{-1}\Big]",
    )


@derivation("kp.split")
def check_split():
    """The page's weights between two grid bins keep the number of kaons and
    their energy. AM3's weight, which the page contrasts, keeps the number.
    With this weight, the mean energy is 1 + r(1 - r) Delta^2 times the
    particles' up to terms of order Delta^3, at most 1 + Delta^2/4. The page's
    q + r grid steps, q an integer and 0 <= r < 1, make r the fractional part
    of ln chi/Delta."""
    r_ = sp.Symbol("r", positive=True)

    def fractional():
        # Every ln chi/Delta, also a negative one or a whole number, must split
        # one way, with r its fractional part. Then the landing point lies
        # between the bins i + q and i + q + 1.
        q_, x_, r0 = (
            sp.Symbol("q", integer=True),
            sp.Symbol("x"),
            sp.Symbol("r"),
        )
        steps = L.to_sympy(L.after("which lies $"), {"q": q_, "r": r0})
        lo, rel_lo, rel_hi, hi = re.fullmatch(
            r"(.+) \\(le|lt) r \\(le|lt) (.+)",
            L.after("with $q$ an integer and $"),
        ).groups()
        span = sp.Interval(
            L.to_sympy(lo, {}),
            L.to_sympy(hi, {}),
            rel_lo == "lt",
            rel_hi == "lt",
        )
        r_of = sp.solve(steps - x_, r0)[0]
        splits = {
            x: [
                (q, r_of.subs({q_: q, x_: x}))
                for q in range(sp.floor(x) - 2, sp.floor(x) + 3)
                if r_of.subs({q_: q, x_: x}) in span
            ]
            for x in map(sp.Rational, ("-5/2", "-1", "-3/10", "0", "7/10", "3"))
        }
        return [
            Result(
                "r is the fractional part of ln chi/Delta: q + r grid steps, q "
                "an integer and r in the page's range, split every value one "
                "way",
                all(
                    s == [(sp.floor(x), x - sp.floor(x))]
                    for x, s in splits.items()
                ),
                "; ".join(f"{x}: {s}" for x, s in splits.items()),
            )
        ]

    def mean(w):
        return ((1 - w) + w * sp.exp(Dl)) / sp.exp(r_ * Dl)

    def results():
        rp = L.to_sympy(
            L.sides(L.after(r"the upper bin takes $"))[1],
            {"r": r_, r"\Delta": Dl},
        )
        am3 = L.to_sympy(
            L.inline(r"r + r(1 - r)\Delta/2"), {"r": r_, r"\Delta": Dl}
        )
        stated = L.to_sympy(
            L.inline(r"1 + r(1 - r)\Delta^2"), {"r": r_, r"\Delta": Dl}
        )
        top = L.to_sympy(L.inline(r"1 + \Delta^2/4"), {r"\Delta": Dl})
        series = sp.series(mean(am3), Dl, 0, 3).removeO()
        worst = sp.maximum(stated.subs(Dl, 1) - 1, r_, sp.Interval(0, 1))
        return [
            exact(
                "the weights 1 - r' and r' add up to 1: the number is kept",
                (1 - rp) + rp - 1,
            ),
            exact(
                "their mean energy is the kaons' E_q e^(r Delta): the energy "
                "is kept",
                mean(rp) - 1,
            ),
            exact(
                "AM3's weight: its mean energy over the particles' = the "
                "page's factor to order Delta^2",
                sp.expand(series - sp.series(stated, Dl, 0, 3).removeO()),
            ),
            exact(
                "the largest r(1 - r) Delta^2 over 0 <= r < 1 is Delta^2/4",
                worst * Dl**2 + 1 - top,
            ),
        ] + fractional()

    out = with_control(
        results,
        "control: AM3's weight r + r(1 - r)Delta/2 as the page's r'",
        r"r' = (e^{r\Delta} - 1)/(e^\Delta - 1)$",
        r"r' = r + r(1 - r)\Delta/2$",
    )
    for old, new in (
        (r"and $0 \le r \lt 1$", r"and $-1 \lt r \le 0$"),
        (r"which lies $q + r$", r"which lies $q - r$"),
    ):
        with L.patched(old, new):
            out.append(must_fail(f"control: {new} in the page", fractional()))
    return out


E1, E2, SC = sp.symbols("epsilon_min epsilon_max sigma_c", positive=True)


def readme_fc(branch_min):
    lhs, rhs = L.sides(L.display(r"f_c(y) = \sigma_c [\min(4y^2"))
    assert lhs == "f_c(y)"
    args_tex = L.between(rhs, r"\min(", ")")
    tok = {
        "y": y,
        r"\epsilon_\mathrm{max}": E2,
        r"\epsilon_\mathrm{min}": E1,
        r"\sigma_c": SC,
    }
    args = [L.to_sympy(a, tok) for a in args_tex.split(", ")]
    return L.to_sympy(rhs, {**tok, r"\min(" + args_tex + ")": branch_min(args)})


@derivation("kp.fc")
def check_fc():
    """f_c = sigma_c [min(4y^2, eps_max^2) - eps_min^2]/(4y^2) is eq. 27 for a
    constant cross section, 2y >= eps_min."""
    L.inline(r"2y \ge \epsilon_\mathrm{min}")
    pre, lo, up, body = readme_f_parts()
    d, D, Y = sp.symbols("d D Y", positive=True)
    sig = SC * (sp.Heaviside(er - E1) - sp.Heaviside(er - E2))
    integral = pre * sp.integrate(body.subs(sigma(er), sig), (er, lo, Y)).subs(
        Y, up
    )
    out = []
    for name, which, at in (
        ("2y < eps_min", None, {E1: 2 * y + d, E2: 2 * y + d + D}),
        ("eps_min <= 2y < eps_max", 0, {y: (E1 + d) / 2, E2: E1 + d + D}),
        ("2y >= eps_max", 1, {y: (E1 + d + D) / 2, E2: E1 + d}),
    ):

        def smaller(args):
            if which is not None:
                assert sp.expand(
                    (args[1 - which] - args[which]).subs(at)
                ).is_positive
            return args[which or 0]

        closed, value = readme_fc(smaller).subs(at), integral.subs(at)
        if which is None:
            neg = sp.factor(closed)
            out.append(
                Result(
                    f"{name}: eq. 27 gives 0, the closed form is "
                    "negative there (hence 'for 2y >= eps_min')",
                    value == 0 and neg.is_negative,
                    f"integral {value}, closed form {neg}",
                )
            )
        else:
            out.append(
                exact(
                    f"{name}: eq. 27 = closed form with min = "
                    f"{['4y^2', 'eps_max^2'][which]}",
                    value - closed,
                )
            )
    with L.patched(
        r"- \epsilon_\mathrm{min}^2]/(4y^2)",
        r"- \epsilon_\mathrm{min}^2]/(2y^2)",
    ):
        broken = readme_fc(lambda args: args[0]).subs(
            {y: (E1 + d) / 2, E2: E1 + d + D}
        )
    out.append(
        exact_fails(
            "control: /(2y^2) in the closed form",
            integral.subs({y: (E1 + d) / 2, E2: E1 + d + D}) - broken,
        )
    )
    return out


CHI_PIN = r"\chi = \frac{s - m_Y^2 + m_K^2}{2s}"


def eq16_parts(page=True, tokens=None):
    tok = tokens or {
        r"s(\epsilon_r)": sp.Symbol("s"),
        "m_p": sp.Symbol("m_p"),
        r"m_\pi": sp.Symbol("m_pi"),
        "beta": sp.Symbol("beta"),
        "theta": sp.Symbol("theta_pi"),
    }
    if page:
        disp = L.display(EQ16_PIN)
        chi_tex = L.sides(L.part(disp, r"\chi(\epsilon_r) ="))[1]
        s_tex = L.sides(L.part(disp, r"s(\epsilon_r) ="))[1]
        names = {"beta": r"\beta_\pi^{\rm cm}", "theta": r"\theta_\pi"}
    else:
        chi_tex, s_tex = H_EQ16.split("=", 1)[1].strip(), H_EQ5.split("=")[1]
        names = {"beta": r"\beta_\pi^{\mathrm{cm}}", "theta": r"\theta_\pi"}
    t16 = {names.get(k, k): v for k, v in tok.items() if k != r"\epsilon_r"}
    chi_ = L.to_sympy(chi_tex, t16)
    s_ = L.to_sympy(s_tex, {"m_p": tok["m_p"], r"\epsilon_r": er})
    return chi_, s_


def eq5():
    """The page's eq. 5, which kp.eq16 checks against the TeX."""
    return eq16_parts(
        tokens={
            r"s(\epsilon_r)": W**2,
            "m_p": mN,
            r"m_\pi": mK,
            "beta": ps / Es,
            "theta": ct,
        }
    )[1]


def readme_s():
    return L.to_sympy(
        L.sides(L.part(L.display(CHI_PIN), "s ="))[1],
        {"m_N": mN, r"\epsilon_r": er},
    )


@lru_cache(None)
def numeric_s():
    return sp.lambdify((mN, er), readme_s(), "numpy")


def readme_chi(s_, beta, angle):
    return L.to_sympy(
        L.sides(L.part(L.display(CHI_PIN), r"\chi ="))[1],
        {"s": s_, "m_Y": mY, "m_K": mK, r"\beta^\ast": beta, r"\theta": angle},
    )


def two_body_Estar():
    EKs, EYs = sp.symbols("E_K_star E_Y_star")
    return sp.solve(
        [EKs + EYs - W, EKs**2 - mK**2 - (EYs**2 - mY**2)],
        [EKs, EYs],
        dict=True,
    )[0][EKs]


@lru_cache(None)
def numeric_chi():
    """The README's chi (linear in cos theta by kp.chi) as a numpy function of
    (m_N, m_Y, m_K, eps_r, <cos theta>), for the code's chi in
    tests/support/measured/ky_angular.py."""
    Estar_ = two_body_Estar()
    chi = readme_chi(
        readme_s(), sp.sqrt(Estar_**2 - mK**2) / Estar_, sp.acos(ct)
    ).subs(W, sp.sqrt(readme_s()))
    return sp.lambdify((mN, mY, mK, er, ct), chi, "numpy")


@derivation("kp.eq16")
def check_eq16():
    """The page's eqs. 16 and 5 are Hümmer et al.'s, with their angle from the
    nucleon and their kaon replacements."""

    def results():
        (c16, s5), (t16, t5) = eq16_parts(), eq16_parts(page=False)
        back = L.to_sympy(L.sides(L.inline(r"\cos\theta_\pi = -1"))[1], {})
        theirs = L.to_sympy(re.search(r"=(-1)\$", H_BACK).group(1), {})
        mean = L.to_sympy(
            L.inline(r"\langle\cos\theta_\pi\rangle \simeq 0").split(r"\simeq")[
                1
            ],
            {},
        )
        return [
            quote_results(
                "the page's eqs. 16 and 5 = the TeX's", (c16, s5), (t16, t5)
            ),
            Result(
                "backward scattering: cos theta_pi = -1, as their note "
                "after eq. 14",
                back == theirs,
                f"{back}",
            ),
            Result(
                "for kaons <cos theta_pi> ~ 0, as their Sec. 3.1 "
                "(with m_p -> m_Lambda, m_pi -> m_K)",
                mean == 0,
                f"{mean}",
            ),
        ]

    return [
        in_source(
            "eqs. 5 and 16, the angle and the kaon replacements as transcribed",
            HUEMMER,
            H_EQ5,
            H_EQ16,
            H_BACK,
            H_KAON,
        )
    ] + with_control(
        results,
        "control: (1 - beta cos theta_pi) in the quote",
        r"(1 + \beta_\pi^{\rm cm}\cos\theta_\pi)",
        r"(1 - \beta_\pi^{\rm cm}\cos\theta_\pi)",
    )


@derivation("kp.chi")
def check_chi():
    """chi = (s - m_Y^2 + m_K^2)/(2s) (1 - beta* cos theta) is two-body
    kinematics and Hümmer et al.'s eq. 16."""
    s_ours, ours = readme_s(), readme_chi
    th = sp.Symbol("theta", positive=True)
    # the page's eq. 16 (the TeX's by kp.eq16), with the page's angle from the
    # photon: theta = pi - theta_pi
    eq16 = lambda angle: eq16_parts(
        tokens={
            r"s(\epsilon_r)": W**2,
            "m_p": mY,
            r"m_\pi": mK,
            "beta": ps / Es,
            "theta": angle,
        }
    )[0]
    thp = sp.Symbol("theta_pi", positive=True)
    rel = L.sides(L.inline(r"\theta = \pi - \theta_\pi"))
    theta_pi = sp.solve(
        L.to_sympy(rel[0], {r"\theta": th})
        - L.to_sympy(rel[1], {r"\theta_\pi": thp}),
        thp,
    )[0]
    s5 = eq5()
    E15 = L.to_sympy(H_EQ15.split("=")[1], {"s": W**2, "m_p": mY, r"m_\pi": mK})
    E14 = L.to_sympy(
        H_EQ14.split("=")[2],
        {
            "E_p": E,
            r"E_\pi^\text{cm}": E15,
            "s": W**2,
            r"\beta_\pi^\text{cm}": ps / Es,
            r"\theta_\pi^\text{cm}": sp.pi - th,
        },
    )
    Estar = two_body_Estar()
    lab = sp.limit(lab_share(Estar, ps * ct), u, sp.oo)
    k_ = er / u  # photon energy for eps_r = p_N.k/m_N = k u
    s_kin = (mN * (u + 1 / u) / 2 + k_) ** 2 - (mN * (u - 1 / u) / 2 - k_) ** 2
    chi_kin = ours(W**2, ps / Estar, sp.acos(ct))
    with L.patched(
        r"\big(1 - \beta^\ast \cos\theta\big)",
        r"\big(1 + \beta^\ast \cos\theta\big)",
    ):
        broken = ours(W**2, ps / Estar, sp.acos(ct))
    d2 = sp.diff(ours(W**2, ps / Es, sp.acos(ct)), ct, 2)
    return [
        in_source(
            "eqs. 5, 14-16, the angle of eq. 14 and the replacements for "
            "kaons as transcribed",
            HUEMMER,
            H_EQ5,
            H_EQ14,
            H_BACK,
            H_EQ15,
            H_EQ16,
            H_KAON,
        ),
        exact("README s = eq. 5 (m_p -> m_N)", s_ours - s5),
        exact("README s = (p_N + k)^2, eps_r = p_N.k/m_N", s_ours - s_kin),
        exact(
            "README chi = eq. 16 (m_p -> m_Y, m_pi -> m_K, the page's "
            "theta_pi = pi - theta)",
            ours(W**2, ps / Es, th) - eq16(theta_pi),
        ),
        exact_fails(
            "eq. 16 with theta_pi = theta differs "
            "(Hümmer's angle is from the nucleon)",
            ours(W**2, ps / Es, th) - eq16(th),
        ),
        exact(
            "eq. 16 = eq. 14 / E_p with E*_pi of eq. 15 (Hümmer's step)",
            eq16(sp.pi - th) - E14 / E,
        ),
        exact(
            "two-body kinematics: lim E_K/E_N (E_N -> oo, s fixed) = README "
            "chi, beta* = p*/E*",
            lab - chi_kin,
        ),
        exact_fails(
            "control: (1 + beta* cos theta) in the README", lab - broken
        ),
        exact("chi is linear in cos theta (its mean is chi(<cos theta>))", d2),
    ]


def readme_isotropic():
    lhs, rhs = L.sides(
        L.after(r"the isotropic value that Hümmer et al. assume ($")
    )
    assert lhs == r"\langle\cos\theta\rangle", lhs
    return L.to_sympy(rhs, {})


@derivation("kp.isotropic")
def check_isotropic():
    """Isotropic emission has <cos theta> = 0, and chi falls with cos theta, so
    forward peaking lowers chi below its isotropic value. Hümmer et al.'s
    chi_K+ = 0.35 is the README's chi at the isotropic <cos theta>."""
    c_ = sp.Symbol("c", real=True)
    slope = sp.diff(readme_chi(W**2, ps / Es, sp.acos(ct)), ct)
    # Hümmer's chi_K+: gamma p -> K+ Lambda
    k_lambda = {mY: MASS["Lambda"], mK: MASS["K+"]}
    s35 = eq5().subs({mN: MASS["p"], er: sp.Rational(14, 10)})
    beta35 = (sp.sqrt(two_body_Estar() ** 2 - mK**2) / two_body_Estar()).subs(
        {W: sp.sqrt(s35), **k_lambda}
    )

    def results():
        iso = readme_isotropic()
        chi35 = readme_chi(s35, beta35, sp.acos(iso)).subs(k_lambda)
        return [
            exact(
                "isotropic: <cos theta> = (1/2) int_-1^1 c dc = the "
                "README's value",
                sp.integrate(c_ / 2, (c_, -1, 1)) - iso,
            ),
            Result(
                "Hümmer's chi_K+ = 0.35 at eps_r = 1.4 GeV = the "
                "README's chi at that <cos theta> (two-body beta*)",
                abs(chi35 - 0.35) < 0.01,
                f"{float(chi35):.4f}",
            ),
        ]

    out = [
        in_source("Hümmer et al.'s chi_K+ = 0.35", HUEMMER, H_CHI35),
        exact(
            "d chi/d cos theta = -beta* (s - m_Y^2 + m_K^2)/(2s)",
            slope + ps / Es * (W**2 - mY**2 + mK**2) / (2 * W**2),
        ),
        Result(
            "s - m_Y^2 + m_K^2 > 0 (W > m_Y): chi falls with cos theta, "
            "so forward peaking lowers it",
            sp.expand((W**2 - mY**2 + mK**2).subs(W, mY + w)).is_positive,
            "for every W = m_Y + w, w > 0",
        ),
    ]
    return out + with_control(
        results,
        "control: <cos theta> = 0.5 in the README",
        r"assume ($\langle\cos\theta\rangle = 0$)",
        r"assume ($\langle\cos\theta\rangle = 0.5$)",
    )


def readme_steps():
    rows = []
    for cells in L.table(r"| $\epsilon_r$ [GeV] | $\mathrm p \to \mathrm K^+$"):
        lo, hi = re.fullmatch(r"([\d.]+)–([\d.]+)", cells[0]).groups()
        cols = [
            None
            if cl == "—"
            else re.fullmatch(
                r"([\d.]+) μb, (?:\$\\chi = )?([\d.]+)\$?", cl
            ).groups()
            for cl in cells[1:]
        ]
        rows.append((lo, hi, cols))
    return rows


@derivation("kp.steps")
def check_steps():
    """The two-body steps of the page (sigma and chi of every channel column)
    are means of the measured data.
    The data are the cross sections and angles of
    tests/support/measured/ky_angular.py, which the unit tests compare with the
    code's table."""
    ky = L.measured("ky_angular")
    table = ky.steps_table()

    def results():
        rows = readme_steps()
        finest = max(
            len(cell[0].split(".")[1])
            for _, _, cols in rows
            for cell in cols
            if cell
        )
        bad = []
        for i, (lo, hi, cols) in enumerate(rows):
            for col, cell in zip(ky.COLUMNS, cols):
                s_, x_ = table[col][i]
                ok = (
                    round(s_, finest) == 0
                    if cell is None
                    else (rounds_to(s_, cell[0]) and rounds_to(x_, cell[1]))
                )
                if not ok:
                    bad.append(
                        f"{lo}-{hi} {col}: README {cell}, data {s_:.4f} "
                        f"mubarn, {x_:.4f}"
                    )
        return [
            Result(
                "the README's steps of eps_r = ky_angular's STEPS",
                [(float(lo), float(hi)) for lo, hi, _ in rows] == ky.STEPS,
                f"{len(rows)} steps",
            ),
            Result(
                "every cell: sigma and chi rounded as printed; a dash "
                "where sigma rounds to 0 at the table's finest decimal "
                f"({10**-finest:g} mubarn)",
                not bad,
                "; ".join(bad) or f"{4 * len(rows)} cells",
            ),
        ]

    return with_control(
        results,
        "control: chi = 0.31 in the first cell of the README",
        r"| 1.31 μb, $\chi = 0.30$ |",
        r"| 1.31 μb, $\chi = 0.31$ |",
    )


@derivation("kp.rule4")
def check_rule4():
    """Above all data (3.8 GeV), K+ Lambda and K+ Sigma0 fall as the page's
    power laws.
    The laws are fitted to their last GeV of data
    (tests/support/measured/ky_angular.py)."""
    ky = L.measured("ky_angular")
    fit = {r: ky.power_law(r)[0] for r in ("K+L", "K+S0")}
    last = max(
        max(pt[0] for pt in points)
        for sets in ky.DATA.values()
        for _, points in sets
    )

    def results():
        line = L.line(r"fall as power laws fitted to their last GeV of data, $")
        top = re.search(
            r"\*Above ([\d.]+) GeV\*, where all data end", line
        ).group(1)
        n1, n2 = re.search(
            r"\$\\propto E\^\{(-[\d.]+)\}\$ and "
            r"\$E\^\{(-[\d.]+)\}\$",
            line,
        ).groups()
        return [
            Result(
                f"all data end at {top} GeV",
                rounds_to(last, top),
                f"the last point at {last:.4f} GeV",
            ),
            Result(
                f"K+ Lambda: E^{n1}, the fit to its last GeV of data",
                rounds_to(fit["K+L"], n1),
                f"exponent {fit['K+L']:.3f}",
            ),
            Result(
                f"K+ Sigma0: E^{n2}",
                rounds_to(fit["K+S0"], n2),
                f"exponent {fit['K+S0']:.3f}",
            ),
        ]

    return with_control(
        results,
        "control: E^-2.6 for K+ Lambda in the README",
        r"$\propto E^{-2.7}$ and",
        r"$\propto E^{-2.6}$ and",
    )


KY_NAMES = {
    r"\mathrm{K}^+\Lambda": "K+L",
    r"\mathrm{K}^+\Sigma^0": "K+S0",
    r"\mathrm{K}^0\Sigma^+": "K0S+",
    r"\mathrm{K}^+\Sigma^-": "K+S-",
    r"\mathrm{K}^0\Lambda": "K0L",
    r"\mathrm{K}^0\Sigma^0": "K0S0",
}


def readme_analogues():
    out = {}
    for cells in L.table(
        r"| Reaction | SAPHIR | CLAS | A2 | Above its data, follows |"
    ):
        f = re.fullmatch(r"\$(.+?)\$ \(([\d.]+)x\)", braced(cells[-1]))
        if f:
            out[
                KY_NAMES[
                    re.fullmatch(
                        r"\$\\gamma\\mathrm\{[pn]\} \\to (.+)\$",
                        braced(cells[0]),
                    ).group(1)
                ]
            ] = (KY_NAMES[f.group(1)], f.group(2))
    return out


@derivation("kp.analogues")
def check_analogues():
    """Above its data, a reaction follows its analogous reaction times the
    page's factor.
    The factor is the ratio of their mean measured cross sections over its last
    0.6 GeV of data (tests/support/measured/ky_angular.py). The text gives the
    table's factors. For K+ Sigma-, 2.0 is the square of isospin's sqrt(2)."""
    ky = L.measured("ky_angular")
    factor = {r: ky.analogue_factor(r) for r, v in ky.REACTIONS.items() if v[3]}

    def results():
        table = readme_analogues()
        bad = [
            f"{r}: README {sub} x {f}, data {ky.REACTIONS[r][3]} x "
            f"{factor[r]:.4f}"
            for r, (sub, f) in table.items()
            if sub != ky.REACTIONS[r][3] or not rounds_to(factor[r], f)
        ]
        text = braced(L.text())
        prose = {
            KY_NAMES[a]: (KY_NAMES[b], f)
            for a, b, f in re.findall(
                r"\$(\\mathrm\{K\}[^$]+)\$ follows \$(\\mathrm\{K\}[^$]+)\$ "
                r"with (\d+(?:\.\d+)?)",
                text,
            )
        }
        prose["K0S0"] = (
            table["K0S0"][0],
            re.search(
                r"because it lacks kaon exchange too\. Its factor, "
                r"(\d+(?:\.\d+)?), comes",
                text,
            ).group(1),
        )
        root = L.to_sympy(
            re.search(
                r"makes a neutron couple \$([^$]+)\$ times as strongly", text
            ).group(1),
            {},
        )
        return [
            Result(
                "every factor of the table: the analogous reaction, "
                "and the ratio of the measured cross sections rounded "
                "as printed",
                not bad and set(table) == set(factor),
                "; ".join(bad)
                or ", ".join(
                    f"{r} {f} ({factor[r]:.4f})" for r, (_, f) in table.items()
                ),
            ),
            Result(
                "the text's factors are the table's",
                prose == table,
                str(sorted(prose.items())),
            ),
            Result(
                "K+ Sigma-: the square of the coupling ratio sqrt(2) is "
                "its factor",
                root**2 == sp.nsimplify(table["K+S-"][1]),
                f"({root})^2 = {root**2}",
            ),
        ]

    return with_control(
        results,
        "control: x 0.16 for K0 Sigma+ in the table",
        r"(0.15x)",
        r"(0.16x)",
    )


@derivation("kp.huemmer_kplus")
def check_huemmer_kplus():
    """The page quotes Hümmer et al.'s K+ channel (their eqs. 41-42): its cross
    sections and their photon energies."""

    def tex_steps():
        return [
            (
                re.match(r"([\d.]+) \\, \\mu", row).group(1),
                re.findall(r"([\d.]+) \\, \\mathrm\{GeV\}", row),
            )
            for row in H_KP
        ]

    def results():
        line = L.line(r"(their eqs. 41–42)")
        a = re.search(
            r"([\d.]+) μb at ([\d.]+)–([\d.]+) GeV and ([\d.]+) μb at "
            r"([\d.]+)–([\d.]+) GeV \(their eqs\. 41–42\)",
            line,
        ).groups()
        b = re.search(
            r"Above ([\d.]+) GeV, the measured fall-off replaces "
            r"their constant ([\d.]+) μb",
            line,
        ).groups()
        page = [(a[0], [a[1], a[2]]), (a[3], [a[4], a[5]]), (b[1], [b[0]])]
        return [
            Result(
                "2.0 μb at 1.0-1.2 GeV, 3.7 μb at 1.2-1.65 GeV and "
                "2.7 μb above: eq. 41's steps",
                page == tex_steps(),
                f"{page}",
            )
        ]

    return [in_source("eq. 41 as transcribed", HUEMMER, *H_KP)] + with_control(
        results,
        "control: 3.6 μb for 3.7 μb in the page",
        "3.7 μb at 1.2",
        "3.6 μb at 1.2",
    )


# --- masses, charges and thresholds of the README's s (also for the scripts and
#     the tests) ---

SAME_MASS = {"K-": "K+", "K0b": "K0", "K0S": "K0", "pi-": "pi+"}
CODE_MASS = {
    "p": "MASS_P",
    "n": "MASS_N",
    "K+": "MASS_K",
    "K0": "MASS_K0",
    "pi0": "MASS_PI0",
    "pi+": "MASS_PI",
    "Lambda": "MASS_LAMBDA",
    "Sigma+": "MASS_SIGMA_PLUS",
    "Sigma0": "MASS_SIGMA0",
    "Sigma-": "MASS_SIGMA_MINUS",
}
_CODE_MASS = mojo_constants.read(CODE_MASS.values())
MASS = {
    q: float(_CODE_MASS[CODE_MASS[SAME_MASS.get(q, q)]])
    for q in (
        "p",
        "n",
        "K+",
        "K-",
        "K0",
        "K0b",
        "K0S",
        "pi0",
        "pi+",
        "pi-",
        "Lambda",
        "Sigma+",
        "Sigma0",
        "Sigma-",
    )
}  # GeV
# not in the code: PDG 2026, Sigma(1385) listing, p. 1, Sigma(1385)+ MASS:
# 1382.83 +- 0.34 MeV, OUR AVERAGE
MASS["Sigma*+"] = 1.38283
CHARGE = {
    "p": 1,
    "n": 0,
    "K+": 1,
    "K-": -1,
    "K0": 0,
    "K0b": 0,
    "pi0": 0,
    "pi+": 1,
    "pi-": -1,
    "Lambda": 0,
    "Sigma+": 1,
    "Sigma0": 0,
    "Sigma-": -1,
}
STRANGENESS = {
    "K+": 1,
    "K0": 1,
    "K-": -1,
    "K0b": -1,
    "Lambda": -1,
    "Sigma+": -1,
    "Sigma0": -1,
    "Sigma-": -1,
}
KAON, ANTIKAON, PION, SIGMA, NUCLEON = (
    ("K+", "K0"),
    ("K-", "K0b"),
    ("pi+", "pi0", "pi-"),
    ("Sigma+", "Sigma0", "Sigma-"),
    ("p", "n"),
)


def final_states(charge, *groups):
    return [
        c
        for c in itertools.product(*groups)
        if sum(CHARGE[q] for q in c) == charge
    ]


@lru_cache(None)
def photon_threshold():
    """eps_r(m_N, M) [GeV] at which the README's s reaches M^2 (also for
    ky_angular.py)."""
    total = sp.Symbol("M", positive=True)
    return sp.lambdify((mN, total), sp.solve(readme_s() - total**2, er)[0])


def threshold(m_target, *masses):
    return float(photon_threshold()(m_target, sum(masses)))


def threshold_span(targets, *groups):
    v = [
        readme_threshold()(t, c)
        for t in targets
        for c in final_states(CHARGE[t], *groups)
    ]
    return min(v), max(v), len(v)


def readme_threshold():
    f = photon_threshold()
    return lambda target, final: f(MASS[target], sum(MASS[q] for q in final))


@derivation("kp.thresholds")
def check_thresholds():
    """The thresholds the page gives follow from its s and the masses.
    They are those of gamma N -> K Lambda pi, K Sigma pi, N K K-bar (with and
    without a pion) and Sigma(1385) K."""

    def results():
        m2 = braced(L.line("for the four charge states"))
        start, lo, hi, states = re.search(
            r"only above ([\d.]+) GeV\. The threshold of .+? is "
            r"([\d.]+)–([\d.]+) GeV for the (\w+) charge states",
            m2,
        ).groups()
        ksp = re.search(
            r"\$\\mathrm\{K\}\\Sigma\\pi\$ starts at ([\d.]+) GeV", m2
        ).group(1)
        kk = re.search(
            r"from ([\d.]+) GeV on \(\$\\gamma\\mathrm\{N\} \\to "
            r"\\mathrm\{N\}\\mathrm\{K\}\\bar\{\\mathrm\{K\}\}\$\)",
            m2,
        ).group(1)
        steps = braced(L.line("as those with one start at"))
        y1385 = re.search(
            r"only above ([\d.]+) GeV, the threshold of "
            r"\$\\Sigma\(1385\)\\mathrm\{K\}\$",
            steps,
        ).group(1)
        kkpi = re.search(
            r"as those with one start at ([\d.]+) GeV", steps
        ).group(1)
        m2_start = re.search(
            r"M2 makes kaons only above ([\d.]+) GeV",
            L.line("M2 makes kaons only above"),
        ).group(1)
        lam = threshold_span(NUCLEON, KAON, ("Lambda",), PION)
        words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        return [
            Result(
                f"gamma N -> K Lambda pi: {lo}-{hi} GeV, {states} "
                "charge states",
                rounds_to(lam[0], lo)
                and rounds_to(lam[1], hi)
                and words.get(states) == lam[2],
                f"{lam[0]:.4f}-{lam[1]:.4f} GeV, {lam[2]}",
            ),
            Result(
                f"kaons with a third particle from {start} GeV, and M2's "
                f"from {m2_start} GeV: that threshold",
                rounds_to(lam[0], start) and rounds_to(lam[0], m2_start),
                f"{lam[0]:.4f} GeV",
            ),
            Result(
                f"gamma N -> K Sigma pi from {ksp} GeV",
                rounds_to(threshold_span(NUCLEON, KAON, SIGMA, PION)[0], ksp),
                f"{threshold_span(NUCLEON, KAON, SIGMA, PION)[0]:.4f} GeV",
            ),
            Result(
                f"gamma N -> N K K-bar from {kk} GeV",
                rounds_to(
                    threshold_span(NUCLEON, NUCLEON, KAON, ANTIKAON)[0], kk
                ),
                f"{threshold_span(NUCLEON, NUCLEON, KAON, ANTIKAON)[0]:.4f} "
                "GeV",
            ),
            Result(
                f"N K K-bar with a pion from {kkpi} GeV",
                rounds_to(
                    threshold_span(NUCLEON, NUCLEON, KAON, ANTIKAON, PION)[0],
                    kkpi,
                ),
                f"{
                    threshold_span(NUCLEON, NUCLEON, KAON, ANTIKAON, PION)[
                        0
                    ]:.4f} GeV",
            ),
            Result(
                f"gamma p -> K0 Sigma(1385)+ from {y1385} GeV",
                rounds_to(
                    threshold(MASS["p"], MASS["K0"], MASS["Sigma*+"]), y1385
                ),
                f"{threshold(MASS['p'], MASS['K0'], MASS['Sigma*+']):.4f} GeV",
            ),
        ]

    return with_control(
        results,
        "control: 1.153 GeV for 1.152 GeV in the page",
        r"1.152–1.168 GeV",
        r"1.153–1.168 GeV",
    )


# the README's sums, asserted as written
SUM_K_TEX = r"\sum_{c \in t,\thinspace K} \chi_c\thinspace M_{c,r}^K"
SUM_PI_TEX = r"\sum_{c \in t,\thinspace \pi} \chi_c\thinspace M_c^\pi"


def readme_s_tr(e_kaons, e_pions):
    lhs, rhs = L.sides(L.parts(L.display(r"s_{t,r} = 1 "))[0])
    assert lhs == "s_{t,r}" and SUM_K_TEX in rhs and SUM_PI_TEX in rhs, rhs
    EKs, EPs = sp.symbols("E_Ksum E_pisum")
    return L.to_sympy(
        rhs.replace(SUM_K_TEX, "E_K").replace(SUM_PI_TEX, r"E_\pi"),
        {"E_K": EKs, r"E_\pi": EPs},
    ).subs({EKs: e_kaons, EPs: e_pions})


@derivation("kp.s_tr")
def check_s_tr():
    """With the pions of type t scaled by s_{t,r}, the pions and the kaons of
    the row r carry the energy the pions carried."""
    L.line(
        r"The denominator sums over the pion channels of type $t$ and the "
        r"pion species $\pi$. The factor thus keeps the energy of the "
        r"interaction at every photon energy"
    )
    chi_pi, m_pi = (
        sp.symbols("chi_pi1:4", positive=True),
        sp.symbols("M_pi1:4", positive=True),
    )
    chi_k, m_k = (
        sp.symbols("chi_K1:3", positive=True),
        sp.symbols("M_K1:3", positive=True),
    )
    e_pi = sum(a * b for a, b in zip(chi_pi, m_pi))
    e_k = sum(a * b for a, b in zip(chi_k, m_k))
    s_ = readme_s_tr(e_k, e_pi)
    after = sum(a * s_ * b for a, b in zip(chi_pi, m_pi)) + e_k
    with L.patched(r"s_{t,r} = 1 - \frac", r"s_{t,r} = 1 + \frac"):
        broken = readme_s_tr(e_k, e_pi)
    # The code takes the type's response less the share 1 - s of the row's
    # (kaon channel) response. Because the response is linear in sigma, this is
    # the response of the cross section s sigma_t in the row's photon range and
    # sigma_t elsewhere.
    pre, lo, up, body = readme_f_parts()
    a1, a2, b2, b1, sv, d = sp.symbols("a_1 a_2 b_2 b_1 s_v d", positive=True)
    order = {a2: a1 + d, b2: a1 + 2 * d, b1: a1 + 3 * d}
    box = lambda lo_, hi_: sp.Heaviside(er - lo_) - sp.Heaviside(er - hi_)
    ST = sp.Symbol("sigma_t", positive=True)
    resp = lambda sig, yv: (
        pre * sp.integrate(body.subs(sigma(er), sig), (er, lo, up))
    ).subs(y, yv)
    eff = L.to_sympy(
        L.inline(r"s_{t,r}\sigma_t"), {"s_{t,r}": sv, r"\sigma_t": ST}
    )
    sigma_eff = ST * box(a1, b1) - (ST - eff) * box(a2, b2)
    same = all(
        sp.simplify(
            (
                resp(ST * box(a1, b1), yv)
                - (1 - sv) * resp(ST * box(a2, b2), yv)
                - resp(sigma_eff, yv)
            ).subs(order)
        )
        == 0
        for yv in ((a1 + d / 2) / 2, (a1 + 3 * d / 2) / 2, (a1 + 4 * d) / 2)
    )
    return [
        exact(
            "sum_pi chi_c s M_c^pi + sum_K chi_c M_c,r^K = sum_pi chi_c "
            "M_c^pi (any multiplicities and energy fractions)",
            after - e_pi,
        ),
        exact_fails(
            "control: 1 + ... for s",
            sum(a * broken * b for a, b in zip(chi_pi, m_pi)) + e_k - e_pi,
        ),
        exact(
            "the code's share of a kaon channel, E_K/E_pi, is 1 - s_{t,r}",
            (1 - s_) - e_k / e_pi,
        ),
        Result(
            "the type's response less (1 - s) times the row's = the "
            "response of s sigma_t in the row's range "
            "(2y below, inside and above it)",
            same,
            "exact: equal",
        ),
    ]


MPI_TABLE = (
    r"| Type, photon energy $\epsilon_r$ | Cross section of the type [μb] |"
)


def mpi_rows():
    out = []
    for cells in L.table(MPI_TABLE):
        mult = [sp.Rational(v) for v in cells[2].split(", ")]
        out.append((cells[0], sp.Rational(cells[1]), mult, cells[3]))
    return out


def readme_mpi_steps():
    out = []
    for name, *_ in mpi_rows():
        m_ = re.fullmatch(
            r"M(\d), (?:([\d.]+)–([\d.]+)|\$(.+?)\\text\{–\}(.+?)\$) GeV", name
        )
        lo, hi = (
            (m_.group(2), m_.group(3))
            if m_.group(2)
            else (m_.group(4), m_.group(5))
        )
        out.append(
            (
                int(m_.group(1)) - 1,
                float(L.to_sympy(lo, {})),
                float(L.to_sympy(hi, {})),
            )
        )
    return out


def readme_type_ranges():
    line = L.line("describe multi-pion production by seven interaction types")
    out = {
        int(n) - 1: (float(lo), float(hi))
        for n, lo, hi in re.findall(r"M(\d) ([\d.]+)–([\d.]+) GeV", line)
    }
    out[6] = (
        out[5][1],
        float(
            L.to_sympy(
                re.search(r"M7 above, up to \$([^$]+)\$ GeV", line).group(1), {}
            )
        ),
    )
    return out


@derivation("kp.mpi_sigma")
def check_mpi_sigma():
    """The K+ + K0 cross section of each row of the multi-pion table is the
    type's cross section times K+ + K0 per interaction."""

    def results():
        rows = mpi_rows()
        bad = [
            f"{t}: {float(sg * (mu_[0] + mu_[1]))} for {printed}"
            for t, sg, mu_, printed in rows
            if not rounds_to(float(sg * (mu_[0] + mu_[1])), printed)
        ]
        m7 = [mu_ for t, _, mu_, _ in rows if t.startswith("M7")][0]
        said = sp.Rational(
            re.search(
                r"M7 makes ([\d.]+) per interaction", L.line("M7 makes")
            ).group(1)
        )
        return [
            Result(
                f"the {len(rows)} rows: sigma x (K+ + K0) rounds to the "
                "printed column",
                not bad,
                f"differ: {bad}" if bad else "all rows",
            ),
            Result(
                "M7 makes 1.135 K+ + K0 per interaction "
                "(> 1: the column exceeds the type's cross section)",
                said == m7[0] + m7[1] and said > 1,
                f"{float(m7[0])} + {float(m7[1])} = {float(m7[0] + m7[1])}",
            ),
        ]

    out = results()
    with L.patched(
        r"| 0.0051, 0.0060, 0, 0 | 0.94 |", r"| 0.0051, 0.0060, 0, 0 | 0.97 |"
    ):
        out.append(
            Result(
                "control: 0.97 for 0.94 (M2)",
                not all(r_.ok for r_ in results()),
                "fails, as it must",
            )
        )
    return out


@derivation("kp.lead")
def check_lead():
    """The table's leading shares keep h_- = 0.6 h_+, rounded as the table
    prints them."""

    def results():
        factor = re.search(
            r"h_- = ([\d.]+)", L.inline(r"h_- = 0.6\thinspace h_+")
        ).group(1)
        shares = [cells[4].split(", ") for cells in L.table(MPI_TABLE)]
        bad = [
            f"{hp}, {hm}"
            for hp, hm in shares
            if not rounds_to(float(sp.Rational(factor) * sp.Rational(hp)), hm)
        ]
        return [
            Result(
                f"h_- = {factor} h_+ in the {len(shares)} rows, rounded "
                "as printed",
                not bad,
                f"differ: {bad}"
                if bad
                else "; ".join(f"{hp}, {hm}" for hp, hm in shares),
            )
        ]

    return with_control(
        results,
        "control: h_- = 0.05 for 0.04 (M5)",
        r"| 0.07, 0.04 |",
        r"| 0.07, 0.05 |",
    )


def table6():
    rows = [fields(r) for r in H_T6]
    out = {}
    for t_ in range(7):
        low, high = rows[2 * t_], rows[2 * t_ + 1]
        assert low[1:4] == high[1:4] or [
            float(sp.sympify(v.replace(r"$\infty$", "oo"))) for v in low[1:4]
        ] == [
            float(sp.sympify(v.replace(r"$\infty$", "oo"))) for v in high[1:4]
        ], (low, high)
        val = lambda v: sp.oo if v == r"$\infty$" else sp.Rational(v)
        out[t_] = (
            val(low[1]),
            val(low[2]),
            val(low[3]),
            sp.Rational(low[4]),
            sp.Rational(high[4]),
        )
    return out


@derivation("kp.table6")
def check_table6():
    """The multi-pion types' photon ranges, cross sections and leading chi are
    those of Hümmer et al.'s Table 6.
    The low part's chi is the page's factor times that of the type's low-energy
    pions."""
    t6 = table6()

    def results():
        ranges = readme_type_ranges()
        text = L.line("The low part takes ")
        f, f4a, f4b = re.search(
            r"The low part takes ([\d.]+) times the \$\\chi\$ of the "
            r"low-energy pions, and ([\d.]+) and ([\d.]+) times in the two "
            r"steps of M4",
            text,
        ).groups()
        bad_ranges = [
            f"M{t_ + 1}"
            for t_ in range(6)
            if ranges[t_] != (float(t6[t_][0]), float(t6[t_][1]))
        ]
        bad = []
        for cells in L.table(MPI_TABLE):
            t_ = int(re.match(r"M(\d)", cells[0]).group(1)) - 1
            lo_step = float(
                L.to_sympy(
                    re.match(r"M\d, \$?(.+?)(?:\\text\{–\}|–)", cells[0]).group(
                        1
                    ),
                    {},
                )
            )
            low, lead = cells[5].split(", ")
            factor = (f4a if lo_step < 10 else f4b) if t_ == 3 else f
            if (
                sp.Rational(cells[1]) != t6[t_][2]
                or sp.Rational(lead) != t6[t_][4]
                or not rounds_to(float(sp.Rational(factor) * t6[t_][3]), low)
            ):
                bad.append(
                    f"{cells[0]}: {cells[1]}, {cells[5]} "
                    f"(Table 6: {t6[t_][2]}, {t6[t_][3]}, {t6[t_][4]}; "
                    f"x {factor})"
                )
        return [
            Result(
                "the page's ranges of M1-M6 are Table 6's, and M7 "
                "starts at its 5000 GeV (oo there, AM3's top here)",
                not bad_ranges
                and ranges[6][0] == float(t6[6][0])
                and t6[6][1] == sp.oo,
                ", ".join(bad_ranges) or "all",
            ),
            Result(
                f"every row: Table 6's sigma and chi_H, and {f} "
                f"(M4: {f4a}, {f4b}) x chi_L rounded as printed",
                not bad,
                "; ".join(bad) or f"{len(L.table(MPI_TABLE))} rows",
            ),
        ]

    return [in_source("Table 6 as transcribed", HUEMMER, *H_T6)] + with_control(
        results,
        "control: 1.7 for 1.6 times chi_L in the page",
        "The low part takes 1.6 times",
        "The low part takes 1.7 times",
    )


@derivation("kp.mT")
def check_mT():
    """At the same rapidity a kaon carries m_T,K/m_T,pi times a pion's energy,
    1.6 at p_T = 0.35 GeV.
    A particle's energy there is m_T cosh y."""
    m_, pT, yr = sp.symbols("m p_T y_r", positive=True)
    mT = L.to_sympy(
        L.sides(L.inline(r"m_T = (m^2 + p_T^2)^{1/2}"))[1], {"m": m_, "p_T": pT}
    )
    E_, pz = mT * sp.cosh(yr), mT * sp.sinh(yr)
    mK_, mpi_ = sp.symbols("m_K m_pi", positive=True)

    def results():
        ratio = re.search(
            r"\$m_\{T,K\}/m_\{T,\\pi\} = ([\d.]+)\$ at \$p_T\$ = ([\d.]+) GeV",
            L.line(r"$m_{T,K}/m_{T,\pi} = "),
        ).groups()
        vals = [
            float(
                mT.subs({m_: MASS[k], pT: sp.Rational(ratio[1])})
                / mT.subs({m_: MASS[q], pT: sp.Rational(ratio[1])})
            )
            for k in ("K+", "K0")
            for q in ("pi+", "pi0")
        ]
        return [
            Result(
                f"m_T,K/m_T,pi at p_T = {ratio[1]} GeV rounds to "
                f"{ratio[0]} (K+-, K0 over pi+-, pi0)",
                all(rounds_to(v, ratio[0]) for v in vals),
                ", ".join(f"{v:.4f}" for v in vals),
            )
        ]

    return [
        exact(
            "E = m_T cosh y and p_z = m_T sinh y: E^2 - p_z^2 - p_T^2 = m^2",
            sp.simplify(E_**2 - pz**2 - pT**2 - m_**2),
        ),
        exact(
            "at the same rapidity and p_T, E_K/E_pi = m_T,K/m_T,pi",
            sp.simplify(
                E_.subs(m_, mK_) / E_.subs(m_, mpi_)
                - mT.subs(m_, mK_) / mT.subs(m_, mpi_)
            ),
        ),
    ] + with_control(
        results,
        "control: 1.7 in the page",
        r"$m_{T,K}/m_{T,\pi} = 1.6$",
        r"$m_{T,K}/m_{T,\pi} = 1.7$",
    )


@derivation("kp.chi_m4")
def check_chi_m4():
    """The share (E* - p*_L)/sqrt(s) of the nucleon energy is the lab E_K/E_N of
    a fast nucleon.
    With x = 2 p*_L/sqrt(s) it is (sqrt(x^2 + 4 m_T^2/s) - x)/2."""
    pL, xr, mT = (
        sp.Symbol("p_L", real=True),
        sp.Symbol("x", real=True),
        sp.Symbol("m_T", positive=True),
    )
    tok = {r"E^\ast": Es, r"p^\ast_L": pL, "s": W**2}

    def results():
        chi_ = L.to_sympy(
            L.sides(L.after(r"A kaon's share of the nucleon energy is $"))[1],
            tok,
        )
        x_ = L.to_sympy(L.sides(L.inline(r"x = 2p^\ast_L/\sqrt{s}"))[1], tok)
        lab = sp.limit(lab_share(Es, pL), u, sp.oo)
        in_x = chi_.subs(Es, sp.sqrt(mT**2 + pL**2)).subs(
            pL, sp.solve(x_ - xr, pL)[0]
        )
        return [
            exact(
                "lim E_K/E_N (E_N -> oo, s fixed) of a kaon of CM energy "
                "E* and momentum p*_L along the photon = the page's chi",
                lab - chi_,
            ),
            exact(
                "with E* = sqrt(m_T^2 + p*_L^2) and the page's x: "
                "chi = (sqrt(x^2 + 4 m_T^2/s) - x)/2",
                sp.simplify(
                    in_x - (sp.sqrt(xr**2 + 4 * mT**2 / W**2) - xr) / 2
                ),
            ),
        ]

    return with_control(
        results,
        "control: E* + p*_L in the page",
        r"\chi = (E^\ast - p^\ast_L)/\sqrt{s}$",
        r"\chi = (E^\ast + p^\ast_L)/\sqrt{s}$",
    )


@derivation("kp.available")
def check_available():
    """E_A = sqrt(s) - m_a - m_b is the kinetic energy of the two particles in
    their CM frame.
    For a photon on a nucleon at rest, s is the page's m_N^2 + 2 m_N eps_r."""
    ma, mb, pc, Ea = sp.symbols("m_a m_b p E_a", positive=True)
    Eas, Ebs = sp.sqrt(ma**2 + pc**2), sp.sqrt(mb**2 + pc**2)

    def results():
        EA = L.to_sympy(
            L.sides(L.after(r"with the available energy $"))[1],
            {"s": (Eas + Ebs) ** 2, "m_a": ma, "m_b": mb},
        )
        return [
            exact(
                "E_A = (E*_a - m_a) + (E*_b - m_b): the kinetic energy "
                "in the CM frame (0 at p* = 0)",
                EA - (Eas - ma) - (Ebs - mb),
            ),
            exact(
                "on a target at rest, s = m_a^2 + m_b^2 + 2 m_b E_a; "
                "for m_a = 0 the page's s = m_N^2 + 2 m_N eps_r",
                (ma**2 + mb**2 + 2 * mb * Ea).subs(ma, 0)
                - readme_s().subs({mN: mb, er: Ea}),
            ),
        ]

    return with_control(
        results,
        "control: sqrt(s) - m_a in the page",
        r"\sqrt{s} - m_a - m_b$, where",
        r"\sqrt{s} - m_a$, where",
    )


K_16 = (
    r"n(\ep) = n_0   \ep^{-1} \alpha (\ep) \beta(\ep) \gamma_1 (\ep)  "
    r"\gamma_2 (\ep)"
)
K_17 = [
    r"n_0& = &  1.21 \times 10^{p_0 + p_1 p_2^2}",
    r"\alpha (\ep) & = &  \ep^{p_1 (\log (\ep) - 2 p_2 )}",
    r"\beta (\ep) & = &  10^{- (\ep/\ep_3)^{p_{30}}} "
    r"10^{- (\ep/\ep_5)^{p_{50}}}",
    r"\gamma_1 (\ep) & = & \tanh \lb p_{70} \log \lb \ep / m c^2 \rb \rb",
    r"\gamma_2 (\ep) & = & \tanh \lb p_{80} \log \lb E_p / \ep \rb \rb",
]
K_DEFS = (
    r"where $\ep_3 \equiv 10^{p_2 + p_4}$, $\ep_5 \equiv 10^{p_2 + p_6}$ "
    r"and all energies are expressed in units of GeV"
)
K_18 = [
    r"p_0 & = & p_{00} + p_{01} \log (E_p)",
    r"p_1 & = & p_{10} + p_{11} \log (E_p)",
    r"p_2 & =&  p_{20} + p_{21} \log (E_p)",
    r"p_4 & = & p_{40} - p_2",
    r"p_6 & = & p_{60} + p_2",
]
K_MACROS = [
    r"\newcommand{\lb}[0] { \left( }",
    r"\newcommand{\rb}[0] { \right) }",
    r"\newcommand{\ep}[0] { \epsilon }",
]
K_T3 = (
    r"& $\pi^+ $& $\pi^-$ & $\pi^0$ &  $K^+ $& $K^-$ & $K^0$ & "
    r"$\overline{K^0} $"
)
Ep, mKs = sp.symbols("E_p m_K", positive=True)
P = {
    name: sp.Symbol(name.replace("{", "").replace("}", ""))
    for name in ["p_0", "p_1", "p_2", "p_4", "p_6"]
    + [f"p_{{{i}{j}}}" for i in range(9) for j in range(2)]
}
TOK_K = {
    r"\epsilon": eps,
    r"\epsilon_3": sp.Symbol("epsilon_3"),
    r"\epsilon_5": sp.Symbol("epsilon_5"),
    "E_p": Ep,
    "n_0": sp.Symbol("n_0"),
    **P,
}


def koers(tex, extra=None):
    t = (
        re.sub(r"\\ep(?![A-Za-z])", r"\\epsilon", tex)
        .replace(r"\lb", "(")
        .replace(r"\rb", ")")
    )
    rhs = re.split(r"&\s*=\s*&|\\equiv|=", t, maxsplit=1)[1]
    return L.to_sympy(rhs, {**TOK_K, **(extra or {})}, funcs=KFUNCS)


KFUNCS = {
    r"\alpha": sp.Function("alpha"),
    r"\beta": sp.Function("beta"),
    r"\gamma_1": sp.Function("gamma_1"),
    r"\gamma_2": sp.Function("gamma_2"),
}

KOERS_N = r"n(\epsilon) = n_0\thinspace \epsilon^{-1}"
KOERS_BETA = r"\beta = 10^{-(\epsilon/\epsilon_3)^{p_{30}}}"


def readme_koers(pin):
    out = {}
    for p in L.parts(L.display(pin)):
        lhs, rhs = L.sides(p)
        out[lhs] = L.to_sympy(rhs, {**TOK_K, "m_K": mKs}, funcs=KFUNCS)
    return out


# Kelner et al. (2006), TeX source (macros \t = \tilde, \d = \delta,
# \ds = \displaystyle): eqs. 70, 71 (AM3's pion source has its form with F_pi)
# and 75-77
KEL_MACROS = [r"\def\d{\delta}", r"\def\t{\tilde}", r"\def\ds{\displaystyle}"]
KEL_70 = r"dN_p=J_p(E_p)\,dE_p"
KEL_71 = (
    r"&\Phi_\gamma(E_\gamma)\equiv\ds\frac{dN_\gamma}{dE_\gamma}=&",
    r"&\ds c\,n_{H}^{}\int\limits_{E_\gamma}^\infty\! "
    r"\sigma_{\rm inel}(E_p)\,J_p(E_p)\,F_\gamma\!\left(\frac{E_\gamma}"
    r"{E_p},\,E_p\right)\frac{dE_p}{E_p}\,.&",
)
KEL_75 = (
    r"\t F_\pi=\t n\,\d\left(E_\pi-\frac{\kappa}{\t n}\,E_{\rm kin}\right),",
    r"where $E_{\rm kin}=E_p-m_p$ is the kinetic energy of protons.",
)
KEL_76 = (
    r"\int\!\t F_\pi\,dE_\pi=\t n\,,\quad {\rm and} \quad "
    r"\int\!E_\pi\,\t F_\pi\,dE_\pi=k\,E_{\rm kin}\, ."
)
KEL_77 = (
    r"q_\pi(E_\pi)=\t n\,\frac{c\,n_H^{}}{K_\pi}\,\sigma_{\rm "
    r"inel}\!\left(m_p+\frac{E_\pi}{K_\pi}\right)J_p\!\left(m_p+\frac{E_"
    r"\pi}{K_\pi}\right),",
    r"where $K_\pi=\kappa/\t n$.",
)
SIGI, NPF, FKF = (
    sp.Function("sigma_inel"),
    sp.Function("N_p"),
    sp.Function("F_K"),
)  # any functions
nH = sp.Symbol("n_H", positive=True)


def readme_pp_source():
    lhs, rhs = L.sides(L.parts(L.display(r"Q_K(E_K) = c\thinspace n_H\int"))[0])
    head, rest = rhs.split(r"\int ", 1)
    measure, body = rest.split(r"\thinspace ", 1)
    assert lhs == "Q_K(E_K)", lhs
    dEp = sp.Symbol("dE_p")
    weight = sp.diff(L.to_sympy(measure, {"dE_p": dEp, "E_p": Ep}), dEp)
    tokens = {"c": c, "n_H": nH, "E_p": Ep, "E_K": EK}
    return (
        L.to_sympy(head, tokens) * weight,
        L.to_sympy(
            body, tokens, {r"\sigma_{\rm inel}": SIGI, "N_p": NPF, "F_K": FKF}
        ),
    )


EQ71_PIN = r"\Phi_\gamma(E_\gamma) = "


def eq71_pieces(page=True):
    fn = {r"\sigma_{\rm inel}": SIGI, "J_p": NPF, r"F_\gamma": FKF}
    if page:
        rhs = L.sides(L.display(EQ71_PIN))[1]
        head, lo, body, measure = re.fullmatch(
            r"(.+?)\\int_\{(.+?)\}\^\\infty (.+?)(\\frac\{dE_p\}\{E_p\}|dE_p)",
            rhs,
        ).groups()
        tok = {"c": c, "n_H": nH, "E_p": Ep, r"E_\gamma": EK}
    else:
        head, body, measure = (
            piece(KEL_71[1], s_)
            for s_ in (
                r"c\,n_{H}^{}",
                r"\sigma_{\rm inel}(E_p)\,J_p(E_p)\,"
                r"F_\gamma\!\left(\frac{E_\gamma}{E_p},\,E_p\right)",
                r"\frac{dE_p}{E_p}",
            )
        )
        lo = re.search(r"\\int\\limits_\{(.+?)\}\^\\infty", KEL_71[1]).group(1)
        tok = {"c": c, "n_{H}^{}": nH, "E_p": Ep, r"E_\gamma": EK}
    dEp = sp.Symbol("dE_p")
    return (
        L.to_sympy(head, tok),
        L.to_sympy(lo, tok),
        L.to_sympy(body, tok, fn),
        sp.diff(L.to_sympy(measure, {"dE_p": dEp, "E_p": Ep}), dEp),
    )


@derivation("pp.eq71")
def check_eq71():
    """The page's eq. 71 is Kelner et al.'s, and a collision makes F_gamma dx
    photons, F_gamma/E_p per unit E_gamma."""

    def results():
        return [
            quote_results(
                "the page's eq. 71 = the TeX's (the factor in front, "
                "the lower limit, the integrand, the measure)",
                eq71_pieces(),
                eq71_pieces(page=False),
            )
        ]

    xk = L.to_sympy(
        L.sides(L.inline(r"x = E_\gamma/E_p"))[1], {r"E_\gamma": EK, "E_p": Ep}
    )
    per_unit = L.to_sympy(
        L.inline(r"F_\gamma/E_p"), {r"F_\gamma": FKF(xk, Ep), "E_p": Ep}
    )
    return [
        in_source(
            "eqs. 70 and 71 and the macros as transcribed",
            KELNER,
            KEL_70,
            *KEL_71,
            *KEL_MACROS,
        ),
        exact(
            "F_gamma dx per unit E_gamma at fixed E_p: F_gamma "
            "dx/dE_gamma = F_gamma/E_p",
            FKF(xk, Ep) * sp.diff(xk, EK) - per_unit,
        ),
    ] + with_control(
        results,
        "control: dE_p for dE_p/E_p in the quote",
        r"E_p\right)\frac{dE_p}{E_p}",
        r"E_p\right)dE_p",
    )


@derivation("pp.source")
def check_pp_source():
    """Q_K counts the collisions, as Kelner et al.'s eq. 71 (F_gamma -> F_K):
    N_p dE_p protons collide at c n_H sigma_inel."""
    L.line(r"and $n = 0$ outside $m_K \lt \epsilon \lt E_p$")
    weight, body = readme_pp_source()
    xk = EK / Ep
    counted = c * nH * SIGI(Ep) * NPF(Ep) * FKF(xk, Ep) * sp.diff(xk, EK)
    # the page's eq. 71 (the TeX's by pp.eq71)
    head71, _, body71, weight71 = eq71_pieces()
    kel = head71 * weight71 * body71
    with L.patched(
        r"Q_K(E_K) = c\thinspace n_H\int \frac{dE_p}{E_p}\thinspace",
        r"Q_K(E_K) = c\thinspace n_H\int dE_p\thinspace",
    ):
        w_b, b_b = readme_pp_source()
    return [
        in_source(
            "eqs. 70 and 71 and the macros as transcribed",
            KELNER,
            KEL_70,
            *KEL_71,
            *KEL_MACROS,
        ),
        exact(
            "README integrand = the collisions' kaons per unit E_K "
            "(any sigma_inel, N_p, F_K; integrands in E_p)",
            weight * body - counted,
        ),
        exact(
            "= Kelner et al.'s eq. 71 with F_gamma -> F_K, J_p -> N_p "
            "(protons per unit energy, eq. 70)",
            weight * body - kel,
        ),
        Result(
            "eq. 71 integrates from E_p = E_gamma; the README's F_K "
            "vanishes for x > 1 (n = 0 outside m_K < eps < E_p), so its "
            "integral over all E_p is the same",
            True,
            "as written (asserted)",
        ),
        exact_fails("control: dE_p for dE_p/E_p", w_b * b_b - counted),
    ]


PPGRID_PIN = r"E_i\thinspace Q_K(E_i) \simeq c\thinspace n_H\Delta\sum_j"


def readme_pp_grid():
    disp = L.display(PPGRID_PIN)
    lhs, rhs = (
        s_.strip()
        for s_ in L.part(disp, r"E_i\thinspace Q_K(E_i)").split(r"\simeq")
    )
    factor, summand = rhs.split(r"\sum_j ")
    X = sp.Symbol("X", positive=True)
    x_ij = L.to_sympy(
        L.sides(L.part(disp, "x_{ij} ="))[1], {"E_i": EK, "E_j": Ep}
    )
    pre = L.to_sympy(factor, {"c": c, "n_H": nH, r"\Delta": Dl})
    body = L.to_sympy(
        summand,
        {"E_j": Ep, "x_{ij}": X},
        {r"\sigma_{\rm inel}": SIGI, "N_p": NPF, "F_K": FKF},
    )
    assert lhs == r"E_i\thinspace Q_K(E_i)", lhs
    return (pre / Dl * body).subs(X, x_ij), x_ij


@derivation("pp.grid")
def check_pp_grid():
    """On the grid the pp source and kappa_K are rectangle rules, and the
    protons lose the kaons' energy exactly.
    The source is that of its integral in ln E_p, and kappa_K is that of
    int x F_K dx in ln x."""
    weight, body = readme_pp_source()
    per_ln = EK * weight * body * Ep

    def results():
        grid, x_ij = readme_pp_grid()
        kap = L.sides(
            L.inline(r"\kappa_K = \Delta\sum_i x_{ij}^2 F_K(x_{ij}, E_j)")
        )[1]
        X = sp.Symbol("X", positive=True)
        kap_sum = L.to_sympy(
            kap.split(r"\sum_i ")[1],
            {"x_{ij}": sp.exp(X), "E_j": Ep},
            {"F_K": FKF},
        )
        kap_ln = sp.Integral(x * FKF(x, Ep), (x, 0, 1)).transform(
            x, (sp.exp(X), X)
        )
        Eg = [E0 * sp.exp(k_ * Dl) for k_ in range(3)]
        sg, nn = (
            sp.symbols("sigma_0:3", positive=True),
            sp.symbols("n_0:3", positive=True),
        )  # n_j = E_j N_p(E_j)
        Fv = [[sp.Symbol(f"F_{i_}{j_}") for j_ in range(3)] for i_ in range(3)]
        xg = lambda i_, j_: Eg[i_] / Eg[j_]
        made = sum(
            Dl
            * Eg[i_]
            * c
            * nH
            * Dl
            * sum(sg[j_] * nn[j_] * xg(i_, j_) * Fv[i_][j_] for j_ in range(3))
            for i_ in range(3)
        )
        lost = sum(
            Dl
            * nn[j_]
            * Eg[j_]
            * nH
            * c
            * sg[j_]
            * Dl
            * sum(xg(i_, j_) ** 2 * Fv[i_][j_] for i_ in range(3))
            for j_ in range(3)
        )
        below = L.to_sympy(
            L.sides(L.inline(r"\kappa_K = \tilde n_K E_K/E_p"))[1],
            {r"\tilde n_K": sp.Symbol("n_t"), "E_K": EK, "E_p": Ep},
        )
        E_ = sp.Symbol("E", positive=True)
        delta_energy = (
            sp.integrate(
                E_ * sp.Symbol("n_t") * sp.DiracDelta(E_ - EK), (E_, 0, sp.oo)
            )
            / Ep
        )
        return [
            exact(
                "summand/Delta = E_K times the page's Q_K integrand per "
                "unit ln E_p, x = E_K/E_p",
                grid - per_ln,
            ),
            exact(
                "kappa_K's summand/Delta = x^2 F_K(x): int x F_K dx in ln x",
                integrand(kap_ln)[0] - kap_sum,
            ),
            exact(
                "on the grid, the energy the kaons get = the energy the "
                "protons lose (any sigma, N_p, F)",
                sp.expand(made - lost),
            ),
            exact(
                "below E_t: n-tilde_K kaons of energy E_K carry the "
                "fraction n-tilde_K E_K/E_p of E_p",
                below - delta_energy,
            ),
        ]

    return with_control(
        results,
        "control: F_K without x_ij in the grid form",
        r"\thinspace x_{ij}F_K(x_{ij}, E_j),\qquad",
        r"\thinspace F_K(x_{ij}, E_j),\qquad",
    )


@derivation("pp.koers_n")
def check_koers_n():
    """n(eps) = n_0 eps^-1 alpha beta gamma_1 gamma_2, n_0 and alpha are Koers
    et al.'s eqs. 16, 17a-b (log = log10)."""
    ours = readme_koers(KOERS_N)
    out = [
        in_source(
            "eqs. 16, 17a-b and the macros as transcribed",
            KOERS,
            K_16,
            *K_17[:2],
            *K_MACROS,
        ),
        exact("README n(eps) = eq. 16", ours[r"n(\epsilon)"] - koers(K_16)),
        exact("README n_0 = eq. 17a", rational(ours["n_0"] - koers(K_17[0]))),
        exact("README alpha = eq. 17b", ours[r"\alpha"] - koers(K_17[1])),
    ]
    # Lg = log10 eps (the parsed log(eps) is the README's \log). With ln, n_0's
    # 10^(p_1 p_2^2) does not complete the square.
    Lg = sp.Symbol("L")
    coeff, square = L.to_sympy(
        L.inline(r"1.21 \times 10^{p_0 + p_1(\log\epsilon - p_2)^2}"), TOK_K
    ).as_coeff_Mul()
    n0_coeff, n0_power = ours["n_0"].as_coeff_Mul()
    total = (
        n0_power.as_base_exp()[1]
        + ours[r"\alpha"].as_base_exp()[1].subs(sp.log(eps), Lg) * Lg
    )
    out.append(
        exact(
            "the page's n_0 alpha with log = log10 is its 1.21 x "
            "10^(p_0 + p_1 (log eps - p_2)^2)",
            rational(coeff - n0_coeff)
            + square.as_base_exp()[1].subs(sp.log(eps), Lg)
            - total,
        )
    )
    power = ours["n_0"].as_coeff_Mul()[1]
    assert (
        power.as_base_exp()[0] == 10 and ours[r"\alpha"].as_base_exp()[0] == eps
    )
    for name, log_eps in (("log = log10", Lg), ("log = ln", Lg * sp.log(10))):
        exponent = (
            power.as_base_exp()[1]
            + ours[r"\alpha"].as_base_exp()[1].subs(sp.log(eps), log_eps) * Lg
        )
        d = exponent - (P["p_0"] + P["p_1"] * (Lg - P["p_2"]) ** 2)
        out.append(
            exact(
                f"{name}: n_0 alpha = 1.21 x 10^(p_0 + p_1 (log eps - p_2)^2)",
                d,
            )
            if log_eps == Lg
            else exact_fails(
                f"{name}: no parabola in log eps "
                "(the README's log = log10 is Koers et al.'s)",
                d,
            )
        )
    FK = L.to_sympy(
        L.sides(L.inline(r"F_K(x, E_p) = E_p n(xE_p)"))[1],
        {"E_p": Ep, "x": x},
        funcs={"n": n},
    )
    lo, hi = sp.symbols("a b", positive=True)
    t = sp.Integral(n(eps), (eps, lo, hi)).transform(eps, (Ep * x, x))
    out.append(
        exact(
            "F_K = E_p n(x E_p): int n deps = int F_K dx (eps = x E_p)",
            integrand(t)[0] - FK,
        )
    )
    with L.patched(
        r"10^{\thinspace p_0 + p_1 p_2^2}", r"10^{\thinspace p_0 + p_1 p_2}"
    ):
        out.append(
            exact_fails(
                "control: p_1 p_2 for p_1 p_2^2 in n_0",
                rational(readme_koers(KOERS_N)["n_0"] - koers(K_17[0])),
            )
        )
    return out


@derivation("pp.koers_beta")
def check_koers_beta():
    """beta, gamma_1 and gamma_2 are Koers et al.'s eqs. 17c-e (their
    m c^2 = m_K, energies in GeV)."""
    ours = readme_koers(KOERS_BETA)
    out = [
        in_source(
            "eqs. 17c-e and the units as transcribed", KOERS, *K_17[2:], K_DEFS
        ),
        exact("README beta = eq. 17c", ours[r"\beta"] - koers(K_17[2])),
        exact(
            "README gamma_1 = eq. 17d (m c^2 = m_K)",
            ours[r"\gamma_1"] - koers(K_17[3], {"m": mKs, "c": 1}),
        ),
        exact("README gamma_2 = eq. 17e", ours[r"\gamma_2"] - koers(K_17[4])),
    ]
    with L.patched(
        r"\big(p_{80}\log(E_p/\epsilon)\big)",
        r"\big(p_{80}\log(\epsilon/E_p)\big)",
    ):
        out.append(
            exact_fails(
                "control: log(eps/E_p) in gamma_2",
                readme_koers(KOERS_BETA)[r"\gamma_2"] - koers(K_17[4]),
            )
        )
    return out


@derivation("pp.koers_params")
def check_koers_params():
    """eps_3, eps_5 and p_0 ... p_6 are Koers et al.'s definitions after eq. 17
    and eqs. 18a-e, with Table III's names."""
    theirs = dict(
        zip(
            [r"\epsilon_3", r"\epsilon_5", "p_0", "p_1", "p_2", "p_4", "p_6"],
            re.findall(r"\$(\\ep_[35] \\equiv .+?)\$", K_DEFS) + K_18,
        )
    )
    readme = readme_params()
    out = [
        in_source(
            "the definitions of eps_3, eps_5 and eqs. 18a-e as transcribed",
            KOERS,
            K_DEFS,
            *K_18,
        ),
        Result(
            "the README defines eps_3, eps_5, p_0, p_1, p_2, p_4, p_6",
            list(readme) == list(theirs),
            f"{list(readme)}",
        ),
    ]
    for name, tex in theirs.items():
        out.append(
            exact(f"README {name} = Koers et al.'s", readme[name] - koers(tex))
        )
    with L.patched(r"$p_4 = p_{40} - p_2$", r"$p_4 = p_{40} + p_2$"):
        out.append(
            exact_fails(
                "control: p_40 + p_2 in the README",
                readme_params()["p_4"] - koers(theirs["p_4"]),
            )
        )
    names, readme_rows = readme_table3()
    rows = koers_table3()
    out.append(
        in_source(
            "Table III (columns pi+, pi-, pi0, K+, K-, K0, anti-K0) "
            "as transcribed",
            KOERS,
            K_T3,
            "".join(K_T3_ROWS),
        )
    )
    out.append(
        Result(
            "README table columns = Table III's twelve p_ij, in order",
            names == list(rows),
            f"{names}",
        )
    )
    column = {label: K_COLUMNS.index(kind) for label, kind in SPECIES.items()}
    bad = []
    for label, col in column.items():
        values = readme_rows[label]
        bad += [
            f"{label} {nm}"
            for nm, v in zip(names, values)
            if v != rows[nm][col]
        ] + (
            [f"{label}: {len(values)} values"]
            if len(values) != len(names)
            else []
        )
    assert sorted(readme_rows) == sorted(column), list(readme_rows)
    out.append(
        Result(
            "README table rows K+, K-, K0, anti-K0, pi0 = Table III "
            "(60 values)",
            not bad,
            f"differ: {bad}" if bad else "equal",
        )
    )
    return out


# K0_L listing p. 34: A_L, the weighted average of A_L(mu) and A_L(e),
# 0.332 +- 0.006 %
PDG_AL = sp.Rational("0.00332")


def readme_neutral(FK0, FK0b):
    line = braced(
        L.line(
            r"is half $\mathrm K^0_\mathrm S$ and half "
            r"$\mathrm K^0_\mathrm L$: $F_{\mathrm K^0_\mathrm S}$"
        )
    )
    tex = re.search(
        r"\$F_\{\\mathrm\{K\}\^0_\\mathrm\{S\}\}\$ "
        r"\$= F_\{\\mathrm\{K\}\^0_\\mathrm\{L\}\}\$ \$= (.+?)\$\.",
        line,
    ).group(1)
    return L.to_sympy(
        tex, {r"F_{\mathrm{K}^0}": FK0, r"F_{\bar{\mathrm{K}}^0}": FK0b}
    )


@lru_cache(None)
def numeric_neutral():
    a, b = sp.symbols("a b")
    return sp.lambdify((a, b), readme_neutral(a, b), "numpy")


@derivation("pp.neutral")
def check_pp_neutral():
    """A K0 or anti-K0 is half K0_S and half K0_L (the CP eigenstates), and CP
    violation moves the halves by A_L/2."""
    small = re.search(
        r"A \$\\mathrm\{K\}\^0\$ or \$\\bar\{\\mathrm\{K\}\}\^0\$ is half "
        r"\$\\mathrm\{K\}\^0_\\mathrm\{S\}\$ and half "
        r"\$\\mathrm\{K\}\^0_\\mathrm\{L\}\$\. CP violation and "
        r"\$\\mathrm\{K\}\^0\\text\{–\}\\bar\{\\mathrm\{K\}\}\^0\$ "
        r"interference \(\$\\lesssim\$ (\d+)%\) are neglected",
        braced(L.line(r"interference ($\lesssim")),
    ).group(1)
    FK0, FK0b = sp.symbols("F_K0 F_K0b", positive=True)
    # in the basis K0, anti-K0 (CP K0 = anti-K0)
    KS, KL = sp.Matrix([1, 1]) / sp.sqrt(2), sp.Matrix([1, -1]) / sp.sqrt(2)
    F = {
        n: v[0] ** 2 * FK0 + v[1] ** 2 * FK0b for n, v in (("S", KS), ("L", KL))
    }
    # CP violation: K_S,L = p K0 +- q anti-K0, |p|^2 + |q|^2 = 1. With
    # Delta S = Delta Q, only the K0 part decays into pi- l+ nu. So PDG's
    # A_L = [Gamma(pi- l+ nu) - Gamma(pi+ l- nu-bar)]/SUM = |p|^2 - |q|^2.
    p2, AL = sp.symbols("p2 A_L", positive=True)
    share = sp.solve(sp.Eq(AL, p2 - (1 - p2)), p2)[0]
    with L.patched(
        r"$= \tfrac12(F_{\mathrm K^0} + F_{\bar{\mathrm K}^0})$",
        r"$= \tfrac12(F_{\mathrm K^0} - F_{\bar{\mathrm K}^0})$",
    ):
        broken = readme_neutral(FK0, FK0b)
    return [
        exact(
            "F_K0S = |<K_S|K0>|^2 F_K0 + |<K_S|K0bar>|^2 F_K0bar, "
            "K_S = (K0 + K0bar)/sqrt 2: the README's",
            F["S"] - readme_neutral(FK0, FK0b),
        ),
        exact(
            "F_K0L, K_L = (K0 - K0bar)/sqrt 2: the same",
            F["L"] - readme_neutral(FK0, FK0b),
        ),
        exact(
            "F_K0S + F_K0L = F_K0 + F_K0bar: each neutral kaon counted once",
            F["S"] + F["L"] - FK0 - FK0b,
        ),
        exact(
            "CP violation: the K0 share of K_S and of K_L is |p|^2 = "
            "(1 + A_L)/2",
            share - (1 + AL) / 2,
        ),
        Result(
            f"with PDG's A_L = {float(100 * PDG_AL)} % (K0_L listing p. 34) "
            f"the halves move by A_L/2 = {float(PDG_AL / 2):.2e}, within "
            f"the README's ≲ {small} % (the interference is not "
            "checked here)",
            PDG_AL / 2 < sp.Rational(int(small), 100),
            f"A_L/2 = {float(PDG_AL / 2):.4f}",
        ),
        exact_fails("control: 1/2 (F_K0 - F_K0bar)", F["S"] - broken),
    ]


nKf, npf, Fhf = sp.Function("n_K"), sp.Function("n_pi0"), sp.Function("Fhat_pi")
Estar = sp.Symbol("E_ast", positive=True)
EXT_PIN = r"F_K(x,E_p) = R(x)\thinspace \hat F_\pi(x,E_p)"


def readme_estar():
    return L.to_sympy(L.sides(L.inline(r"E^\ast = 10^6"))[1], {})


def readme_extension():
    tex = L.display(EXT_PIN)
    xs = sp.Symbol("x_s", positive=True)
    xmin = L.to_sympy(
        re.fullmatch(r"x_s = \\max\(x, (.+)\)", L.part(tex, "x_s =")).group(1),
        {},
    )
    R = L.to_sympy(
        L.sides(L.part(tex, "R(x) ="))[1].replace(";", ","),
        {r"E^\ast": Estar, "x_s": xs},
        {"n_K": nKf, r"\hat F_\pi": Fhf},
    )
    F = L.to_sympy(
        L.sides(L.part(tex, "F_K(x,E_p) ="))[1],
        {"x": x, "E_p": Ep},
        {"R": lambda a: R.subs(xs, a), r"\hat F_\pi": Fhf},
    )
    return F, xmin


def readme_low_x():
    line = L.line(r"For $x \lt 10^{-4}$ (kaons below 100 GeV at $E^\ast$)")
    ratio = re.search(
        r"\$R\(x\)\$ \$\\to R\(10\^\{-4\}\)\$ \$(.+?)\$\.", line
    ).group(1)
    rho_tex = re.search(r"\$\\rho\(\\epsilon\) = (.+?)\$", line).group(1)
    args = re.fullmatch(
        r"\\rho\(\\min\((.+?), (.+?)\)\)/\\rho\((.+?)\)", ratio
    ).groups()  # min asserted as written
    a1, a2, a3 = (
        L.to_sympy(t, {"x": x, "E_p": Ep, r"E^\ast": Estar}) for t in args
    )

    def rho(e):
        return L.to_sympy(
            rho_tex,
            {r"\epsilon": e},
            {
                "n_K": lambda b: nKf(b, Estar),
                r"n_{\pi^0}": lambda b: npf(b, Estar),
            },
        )

    gev = sp.Integer(re.search(r"\(kaons below (\d+) GeV at", line).group(1))
    return rho(sp.Min(a1, a2)) / rho(a3), gev


@derivation("pp.extension")
def check_pp_extension():
    """Above E* the kaons keep Koers et al.'s K/pi ratio at E* (x >= 1e-4). F_K
    is continuous at E*, and the ratio is fixed in E_p."""
    F = readme_extension()[0]
    koers = L.to_sympy(
        L.sides(L.inline(r"F_K(x, E_p) = E_p n(xE_p)"))[1],
        {"E_p": Ep, "x": x},
        {"n": lambda a: nKf(a, Ep)},
    )
    es = readme_estar()
    top = L.to_sympy(
        re.search(
            r"\*\*100 GeV \$\\le E_p \\le (10\^\d+)\$ GeV:",
            L.line(r"**100 GeV $\le E_p \le"),
        ).group(1),
        {},
    )
    start = L.to_sympy(
        re.search(
            r"^\$\\boldsymbol\{E_p \\gt ([^}$]+)\}\$ \*\*GeV "
            r"\(am3_mojo extension",
            L.line(r"publication).** Koers et al.'s fit cannot be taken far"),
        ).group(1),
        {},
    )
    pi0 = L.to_sympy(
        re.search(
            r"is used only by the extension above \$([^$]+)\$ GeV",
            L.line(r"row is used only by the extension above $"),
        ).group(1),
        {},
    )
    mpi = sp.Symbol("m_pi", positive=True)
    tok = {"x": x, r"m_\pi": mpi, "E_p": Ep}
    thr = L.to_sympy(
        re.search(
            r"without its pion threshold factor \$([^$]+)\$",
            L.line("without its pion threshold factor $"),
        ).group(1),
        tok,
    )
    with L.patched(r"R(x) = \frac{E^\ast \thinspace n_K", r"R(x) = \frac{n_K"):
        broken = readme_extension()[0]
    return [
        in_source(
            "Kelner et al.'s threshold factor of F_pi "
            "(eqs. 6 and 12) as transcribed",
            KELNER,
            KEL_THR,
        ),
        exact(
            "at E_p = E*: F_K = E* n_K(x E*; E*), Koers et al.'s E_p "
            "n(x E_p) there (x >= 1e-4): continuous at E*",
            F.subs(Ep, Estar) - koers.subs(Ep, Estar),
        ),
        exact(
            "F_K/F-hat_pi at fixed x >= 1e-4 does not depend on E_p "
            "(the ratio kept, Feynman scaling)",
            sp.diff(F / Fhf(x, Ep), Ep),
        ),
        Result(
            "E* = 10^6 GeV, the top of Koers et al.'s range "
            "(100 GeV <= E_p <= 10^6 GeV)",
            es == top,
            f"E* = {es}, top {top}",
        ),
        Result(
            "the extension's heading, E_p > 10^6 GeV: it takes over at "
            "the top of that range",
            start == top,
            f"{start}",
        ),
        Result(
            "the pi0 row of Table III serves the extension only, "
            "above that top",
            pi0 == top,
            f"{pi0}",
        ),
        exact(
            "the threshold factor the README's F-hat_pi leaves out: "
            "Kelner et al.'s (eqs. 6 and 12)",
            thr - L.to_sympy(KEL_THR, tok),
        ),
        exact_fails(
            "control: no E* in R(x)",
            broken.subs(Ep, Estar) - koers.subs(Ep, Estar),
        ),
    ]


@derivation("pp.low_x")
def check_pp_low_x():
    """Below x = 1e-4, R follows Koers et al.'s K/pi0 ratio at E* in the kaon
    energy, continuously, up to 100 GeV."""
    ratio, gev = readme_low_x()
    xmin = readme_extension()[1]
    es = readme_estar()
    rho = lambda e: nKf(e, Estar) / npf(e, Estar)
    with L.patched(r"/\rho(10^{-4}E^\ast)$.", r"/\rho(10^{-3}E^\ast)$."):
        broken = readme_low_x()[0]
    return [
        exact(
            "at x = 1e-4, any E_p >= E*: R = R(1e-4) (continuous)",
            ratio.subs(x, xmin).subs(Ep, Estar * (1 + w)) - 1,
        ),
        exact(
            "at E_p = E*, x < 1e-4: R/R(1e-4) = rho(x E*)/rho(1e-4 E*), "
            "Koers et al.'s K/pi0 ratio in the kaon energy",
            ratio.subs(Ep, Estar).subs(x, xmin / (1 + w))
            - rho(xmin * Estar / (1 + w)) / rho(xmin * Estar),
        ),
        exact(
            "for any E_p, R depends on x and E_p only through the kaon "
            "energy x E_p",
            sp.diff(ratio.subs(x, EK / Ep), Ep),
        ),
        exact(f"x < 1e-4 at E*: kaons below {gev} GeV", xmin * es - gev),
        exact_fails(
            "control: rho(1e-3 E*) in the denominator",
            broken.subs(x, xmin).subs(Ep, Estar * (1 + w)) - 1,
        ),
    ]


def koers_n(label, mass):
    n_parts, b_parts = readme_koers(KOERS_N), readme_koers(KOERS_BETA)
    params, (names, rows) = readme_params(), readme_table3()
    table = {P[k]: v for k, v in zip(names, rows[label])}
    ps_ = {P[k]: params[k].subs(table) for k in ("p_0", "p_1", "p_2")}
    ps_.update({P[k]: params[k].subs(table).subs(ps_) for k in ("p_4", "p_6")})
    e35 = {
        sp.Symbol("epsilon_3"): params[r"\epsilon_3"].subs(ps_),
        sp.Symbol("epsilon_5"): params[r"\epsilon_5"].subs(ps_),
    }
    expr = (
        (
            n_parts["n_0"]
            * eps**-1
            * n_parts[r"\alpha"]
            * b_parts[r"\beta"].subs(e35)
            * b_parts[r"\gamma_1"]
            * b_parts[r"\gamma_2"]
        )
        .subs(ps_)
        .subs(table)
        .subs(mKs, mass)
    )
    return sp.lambdify(
        (eps, Ep),
        rational(expr).replace(sp.log, lambda a: sp.log(a) / sp.log(10)),
        "mpmath",
    )


@derivation("pp.scaling")
def check_pp_scaling():
    """The page's Koers et al. spectra keep K+/pi0 and K0/pi0 at fixed x nearly
    constant from 10^3 to 10^6 GeV.
    This is Feynman scaling of the ratio, and the page's figures of these ratios
    follow."""
    spec = {
        "K+": (r"$\mathrm{K}^+$", MASS["K+"]),
        "K-": (r"$\mathrm{K}^-$", MASS["K+"]),
        "K0": (r"$\mathrm{K}^0$", MASS["K0"]),
        "K0b": (r"$\bar{\mathrm{K}}^0$", MASS["K0"]),
        "pi0": (r"$\pi^0$", MASS["pi0"]),
    }
    nf = {k: koers_n(*v) for k, v in spec.items()}
    energies = [mp.mpf(10) ** (3 + k_ / mp.mpf(10)) for k_ in range(31)]
    ratio = lambda k, x_: [
        nf[k](x_ * e, e) / nf["pi0"](x_ * e, e) for e in energies
    ]

    def results():
        res = braced(L.line(r"ratios still rise by "))
        (a, b, xa, c_, d, xb, lo_e, hi_e) = re.search(
            r"\$\\mathrm\{K\}\^\+/\\pi\^0 = ([\d.]+)\\text\{–\}([\d.]+)\$ at "
            r"\$x = ([\d.]+)\$ and \$([\d.]+)\\text\{–\}([\d.]+)\$ at "
            r"\$x = ([\d.]+)\$ from \$10\^(\d)\$ to \$10\^(\d)\$ GeV",
            res,
        ).groups()
        r1, r2, r3, r4 = re.search(
            r"rise by (\d+)–(\d+)% at \$x = [\d.]+\$ and fall by (\d+)–(\d+)% "
            r"at \$x = [\d.]+\$",
            res,
        ).groups()
        kp1, kp9 = ratio("K+", mp.mpf(xa)), ratio("K+", mp.mpf(xb))
        rise = [
            100 * (ratio(k, mp.mpf(xa))[-1] / ratio(k, mp.mpf(xa))[0] - 1)
            for k in ("K-", "K0b")
        ]
        fall = [
            100 * (1 - ratio(k, mp.mpf(xb))[-1] / ratio(k, mp.mpf(xb))[0])
            for k in ("K-", "K0b")
        ]
        near = float(
            L.to_sympy(L.inline(r"x \gtrsim 0.03").split(r"\gtrsim")[1], {})
        )
        # (a NaN ratio counts as an infinite spread: max() and min() drop it)
        spread = worst_of(
            float(max(r_) / min(r_) - 1)
            if all(v == v for v in r_)
            else float("inf")
            for k in ("K+", "K0")
            for x_ in (near, 0.1, 0.3, 0.9)
            for r_ in (ratio(k, mp.mpf(x_)),)
        )
        return [
            Result(
                f"K+/pi0 = {a}-{b} at x = {xa} and {c_}-{d} at x = {xb} "
                f"from 10^{lo_e} to 10^{hi_e} GeV",
                (lo_e, hi_e) == ("3", "6")
                and all(v == v for v in list(kp1) + list(kp9))
                and rounds_to(float(min(kp1)), a)
                and rounds_to(float(max(kp1)), b)
                and rounds_to(float(min(kp9)), c_)
                and rounds_to(float(max(kp9)), d),
                f"{float(min(kp1)):.4f}-{float(max(kp1)):.4f}, "
                f"{float(min(kp9)):.4f}-{float(max(kp9)):.4f}",
            ),
            Result(
                f"K-/pi0 and anti-K0/pi0 rise by {r1}-{r2} % at x = {xa} "
                f"and fall by {r3}-{r4} % at x = {xb}",
                sorted(round(float(v)) for v in rise) == [int(r1), int(r2)]
                and sorted(round(float(v)) for v in fall) == [int(r3), int(r4)],
                f"{', '.join(f'{float(v):+.1f}' for v in rise)}; "
                f"{', '.join(f'{-float(v):+.1f}' for v in fall)} %",
            ),
            Result(
                f"nearly independent of E_p at x >= {near}: K+/pi0 and "
                "K0/pi0 vary by less than 10 % over that range",
                spread < 0.1,
                f"at most {100 * spread:.1f} %",
            ),
        ]

    return [
        in_source("Table III as transcribed", KOERS, K_T3, "".join(K_T3_ROWS))
    ] + with_control(
        results,
        "control: 0.124 for 0.123 in the page",
        r"\pi^0 = 0.123\text{–}0.126$",
        r"\pi^0 = 0.124\text{–}0.126$",
    )


def readme_gamma_cm():
    line = L.line(r"frame is $\gamma_\mathrm{cm} = ")
    return re.search(
        r"\$\\gamma_\\mathrm\{cm\} = ([^$]+)\$ \$= ([^$]+)\$", line
    ).groups()


@derivation("pp.gamma_cm")
def check_gamma_cm():
    """gamma_cm = sqrt((E_p + m_p)/(2m_p)) = (E_p + m_p)/sqrt(s),
    s = 2m_p^2 + 2m_p E_p: kinematics of a fixed target."""
    tok = {"E_p": Ep, "m_p": m}
    s_ours = L.to_sympy(
        L.sides(L.part(L.display(r"s = 2m_p^2 + 2m_pE_p"), "s ="))[1], tok
    )

    def forms():
        a, b = readme_gamma_cm()
        return L.to_sympy(a, tok), L.to_sympy(b, {**tok, "s": s_ours})

    first, second = forms()
    qv = 1 + w
    at = {Ep: m * (qv**2 + qv**-2) / 2}
    P_ = m * (qv**2 - qv**-2) / 2
    s_kin = (at[Ep] + m) ** 2 - P_**2
    g_kin = 1 / sp.sqrt(1 - (P_ / (at[Ep] + m)) ** 2)
    with L.patched(r"\sqrt{(E_p + m_p)/(2m_p)}", r"\sqrt{(E_p + m_p)/m_p}"):
        broken = forms()[0]
    th = sp.Symbol("theta")
    E_lab = lambda angle: (
        g_kin * (Es + sp.sqrt(1 - 1 / g_kin**2) * ps * sp.cos(angle))
    )
    return [
        exact("s = (p_1 + p_2)^2 = 2m_p^2 + 2m_p E_p", s_kin - s_ours.subs(at)),
        exact(
            "gamma_cm = 1/sqrt(1 - beta^2), beta = P/(E_p + m_p): "
            "= sqrt((E_p + m_p)/(2m_p))",
            g_kin - first.subs(at),
        ),
        exact("= (E_p + m_p)/sqrt(s)", g_kin - second.subs(at)),
        exact_fails(
            "control: sqrt((E_p + m_p)/m_p) in the README",
            g_kin - broken.subs(at),
        ),
        exact(
            "a kaon of CM energy E*: E_lab(theta) + E_lab(pi - theta) = "
            "2 gamma_cm E* (>= 2 gamma_cm m_K; pp is forward-backward "
            "symmetric)",
            sp.expand(E_lab(th) + E_lab(sp.pi - th) - 2 * g_kin * Es),
        ),
    ]


EQ75_PIN = r"\tilde F_\pi = "


def eq75_parts(page=True, nt=None, kappa=None, Epi=None, Ekin=None, K=None):
    sym = dict(
        zip(
            ("nt", "kappa", "Epi", "Ekin", "K"),
            sp.symbols("n_t kappa E_pi E_kin K_pi", positive=True),
        )
    )
    sym.update(
        {
            k: v
            for k, v in (
                ("nt", nt),
                ("kappa", kappa),
                ("Epi", Epi),
                ("Ekin", Ekin),
                ("K", K),
            )
            if v is not None
        }
    )
    tok = {
        r"\tilde n": sym["nt"],
        r"\kappa": sym["kappa"],
        r"E_\pi": sym["Epi"],
        r"E_{\rm kin}": sym["Ekin"],
        r"K_\pi": sym["K"],
        "c": c,
        "n_H": nH,
        "n_H^{}": nH,
        "m_p": m,
    }
    fn = {r"\sigma_{\rm inel}": SIGI, "J_p": NPF, r"\delta": sp.DiracDelta}
    if page:
        disp = L.display(EQ75_PIN)
        F_tex, q_tex, K_tex = (
            L.sides(L.part(disp, s_))[1]
            for s_ in (r"\tilde F_\pi =", r"q_\pi(E_\pi) =", r"K_\pi =")
        )
    else:
        expand = lambda s_: s_.replace(r"\t ", r"\tilde ").replace(
            r"\d\left", r"\delta\left"
        )
        F_tex = expand(KEL_75[0]).split("=", 1)[1].rstrip(",")
        q_tex = expand(KEL_77[0]).split("=", 1)[1].rstrip(",")
        K_tex = expand(re.search(r"\$K_\\pi=(.+?)\$", KEL_77[1]).group(1))
    return (
        L.to_sympy(F_tex, tok, fn),
        L.to_sympy(q_tex, tok, fn),
        L.to_sympy(K_tex, tok),
    )


@derivation("pp.eq75")
def check_eq75():
    """The page's eqs. 75 and 77 are Kelner et al.'s, and the k that their
    eq. 76 prints is kappa.
    For eq. 75's distribution, eq. 76's integrals are n-tilde and kappa
    E_kin."""
    nt, kap, Epi, Ekin = sp.symbols("n_t kappa E_pi E_kin", positive=True)
    F75 = eq75_parts(nt=nt, kappa=kap, Epi=Epi, Ekin=Ekin)[0]

    def results():
        return [
            quote_results(
                "the page's eqs. 75 and 77 and K_pi = the TeX's",
                eq75_parts(),
                eq75_parts(page=False),
            )
        ]

    return [
        in_source(
            "eqs. 75-77 as transcribed", KELNER, *KEL_75, KEL_76, *KEL_77
        ),
        exact(
            "int F-tilde dE_pi = n-tilde (eq. 76)",
            sp.integrate(F75, (Epi, 0, sp.oo)) - nt,
        ),
        exact(
            "int E_pi F-tilde dE_pi = kappa E_kin (eq. 76, which prints k)",
            sp.integrate(Epi * F75, (Epi, 0, sp.oo)) - kap * Ekin,
        ),
        Result(
            "eq. 76 prints k for kappa",
            r"=k\,E_{\rm kin}" in KEL_76,
            "as transcribed",
        ),
    ] + with_control(
        results,
        "control: n-tilde/kappa in the quote of eq. 75",
        r"\frac{\kappa}{\tilde n}E_{\rm kin}",
        r"\frac{\tilde n}{\kappa}E_{\rm kin}",
    )


@derivation("pp.KK")
def check_KK():
    """K_K = E_t <x>/(E_t - m_p) is the mean fraction of T_p per kaon at E_t.
    For eq. 75's distribution it is Kelner et al.'s K_pi = kappa/n-tilde."""
    Et, xbar, nt, kap = sp.symbols("E_t x_bar n_t kappa", positive=True)
    Tp_of = lambda e: L.to_sympy(
        L.sides(L.inline(r"T_p = E_p - m_p"))[1], {"E_p": e, "m_p": m}
    )
    Epi = sp.Symbol("E_pi", positive=True)

    def results():
        KK = L.to_sympy(
            L.sides(
                L.after(
                    r"per kaon of the Koers et al. spectra "
                    r"at $E_t$: $"
                )
            )[1],
            {"E_t": Et, r"\langle x\rangle": xbar, "m_p": m},
        )
        F75 = eq75_parts(nt=nt, kappa=kap, Epi=Epi, Ekin=Tp_of(Et))[0]
        mean_x = (
            sp.integrate(Epi * F75, (Epi, 0, sp.oo))
            / sp.integrate(F75, (Epi, 0, sp.oo))
            / Et
        )
        K_pi = eq75_parts(nt=nt, kappa=kap)[2]
        return [
            exact(
                "K_K = <x> E_t/T_p(E_t): the mean kaon energy over the "
                "kinetic energy of the proton",
                KK - xbar * Et / Tp_of(Et),
            ),
            exact(
                "for eq. 75's distribution at E_t, K_K = kappa/n-tilde = "
                "K_pi (eq. 77)",
                KK.subs(xbar, mean_x) - K_pi,
            ),
        ]

    return with_control(
        results,
        "control: E_t + m_p in the page",
        r"/(E_t - m_p)$, with",
        r"/(E_t + m_p)$, with",
    )


@derivation("pp.split")
def check_pp_split():
    """Below E_t the weights of the two grid energies around E_K keep the number
    and the energy of the kaons.
    The kaons go to E_l <= E_K < E_{l+1}, and the upper energy has the weight
    w = (E_K - E_l)/(E_{l+1} - E_l)."""
    El, d, D = sp.symbols("E_l d D", positive=True)
    at = {"E_K": El + d, "E_l": El, "E_{l+1}": El + d + D}

    def results():
        w_ = L.to_sympy(
            L.sides(L.after(r"The upper one takes the weight $"))[1], at
        )
        order = L.inline(r"E_l \le E_K \lt E_{l+1}")
        return [
            exact(
                "(1 - w) + w = 1: the number of kaons is kept",
                (1 - w_) + w_ - 1,
            ),
            exact(
                "(1 - w) E_l + w E_{l+1} = E_K: their energy is kept",
                (1 - w_) * El + w_ * (El + d + D) - (El + d),
            ),
            Result(
                f"0 <= w < 1 for {order}",
                bool(w_.is_nonnegative) and bool(sp.factor(1 - w_).is_positive),
                f"w = {w_}",
            ),
        ]

    return with_control(
        results,
        "control: E_{l+1} - E_K in the page",
        r"(E_K - E_l)/(E_{l+1} - E_l)$",
        r"(E_{l+1} - E_K)/(E_{l+1} - E_l)$",
    )


DELTA_PIN = r"\tilde n_K(E_p) = f(s)"


def readme_delta(nkpw, Et):
    disp = L.display(DELTA_PIN)
    expr, (A_, a_, b_, C_, d_, s_, sth) = readme_f()
    f_ = lambda v: expr.subs(s_, v)
    s_of = lambda E: L.to_sympy(
        L.sides(L.part(disp, "s ="))[1], {"m_p": m, "E_p": E}
    )
    lhs, rhs = L.sides(L.parts(disp)[0])
    assert lhs == r"\tilde n_K(E_p)", lhs
    n_ = L.to_sympy(
        rhs,
        {
            r"\langle n_K\rangle_{\rm KPW}(E_t)": nkpw,
            "s_t": s_of(Et),
            "s": s_of(Ep),
            "A": A_,
        },
        {"f": f_},
    )
    return n_, f_, (sth, a_)


@derivation("pp.delta")
def check_pp_delta():
    """Below E_t, each collision makes n-tilde_K kaons of energy K_K T_p, as in
    Kelner et al.'s eqs. 75-77 for kaons (pp.fade: continuous at E_t)."""
    nkpw, Et, KK, kap, nt = sp.symbols("n_KPW E_t K_K kappa n_t", positive=True)

    def branch():
        line = L.line(
            r"Each collision makes $\tilde n_K$ kaons of energy $E_K = "
        )
        return re.search(
            r"\$E_K = \\max\((.+?), \\gamma_\\mathrm\{cm\} m_K\)\$", line
        ).group(1)

    first = branch()
    Tp = L.to_sympy(
        L.sides(L.inline(r"T_p = E_p - m_p"))[1], {"E_p": Ep, "m_p": m}
    )
    L.line(r"The factor $K_{K}$ is the mean fraction of $T_p$ per kaon of the")
    E_ = sp.Symbol("E", positive=True)
    # the page's eq. 75 for a kaon (the TeX's by pp.eq75)
    F75 = eq75_parts(nt=nt, kappa=kap, Epi=E_, Ekin=Tp)[0]
    number, energy = (
        sp.integrate(E_**k_ * F75, (E_, 0, sp.oo)) for k_ in (0, 1)
    )
    K_mean = sp.simplify(energy / number / Tp)

    def source(branch):
        EKp = L.to_sympy(branch, {"K_K": KK, "T_p": Tp, "E_p": Ep})
        q = sp.integrate(
            c * nH * SIGI(Ep) * NPF(Ep) * nt * sp.DiracDelta(EK - EKp),
            (Ep, m, sp.oo),
        )
        return q.replace(sp.Heaviside, lambda *a: 1)

    eq77 = eq75_parts(nt=nt, kappa=kap, Epi=EK, K=KK)[1]
    n_, f_, (sth, a_) = readme_delta(nkpw, Et)
    with L.patched(
        r"f(s) = A\left(1-\frac{s_{\rm th}}{s}\right)^{a}",
        r"f(s) = A\left(2-\frac{s_{\rm th}}{s}\right)^{a}",
    ):
        broken_f = readme_delta(nkpw, Et)[1]
    with L.patched(r"$E_K = \max(K_K T_p,", r"$E_K = \max(K_K E_p,"):
        broken_branch = branch()
    return [
        in_source(
            "eqs. 75-77 as transcribed (eq. 76 prints k for kappa)",
            KELNER,
            *KEL_75,
            KEL_76,
            *KEL_77,
        ),
        exact(
            "eqs. 75-76: the mean fraction of T_p = E_kin per kaon is "
            "kappa/n-tilde, Kelner et al.'s K_pi (after eq. 77): "
            "the README's K_K",
            K_mean - kap / nt,
        ),
        exact(
            "n-tilde_K kaons of E_K = K_K T_p per collision, counted as "
            "in pp.source: Kelner et al.'s eq. 77 with pi -> K, "
            "J_p -> N_p",
            source(first) - eq77,
        ),
        exact(
            "no kaons at threshold: f(s_th) = 0 (a > 0)",
            f_(sth).subs(a_, 1 + w),
        ),
        exact_fails(
            "control: f(s) with 2 - s_th/s", broken_f(sth).subs(a_, 1 + w)
        ),
        exact_fails("control: E_K = K_K E_p", source(broken_branch) - eq77),
    ]


FADE_PIN = (
    r"c_K(E_p) = c_t + (1 - c_t)\thinspace\frac{\ln(E_p/E_t)}{\ln(E_1/E_t)}"
)
FADE_TEXT = r"At $E_t$, the fit and the spectra of"
KPW_SPECIES = {
    "K+": ((r"$\mathrm{K}^+$",), "K+"),
    "K-": ((r"$\mathrm{K}^-$",), "K+"),
    "K0S": ((r"$\mathrm{K}^0$", r"$\bar{\mathrm{K}}^0$"), "K0"),
}


def kpw_multiplicity(kind, e_p):
    """<n_K>_KPW(E_p): the page's n(eps; E_p) of Koers et al. integrated from
    m_K to E_p (as pp_kaon_moments does), for the K+, the K- or the K0_S (half
    the K0 and the anti-K0)."""
    rows, mass = KPW_SPECIES[kind]
    n_ = [koers_n(row, MASS[mass]) for row in rows]
    return float(
        sum(
            mp.quad(lambda e_: f_(e_, e_p), [MASS[mass], 1, 10, e_p])
            for f_ in n_
        )
    ) / len(rows)


def readme_fade():
    ct = sp.Symbol("c_t", positive=True)
    e_t = sp.Integer(
        re.search(
            r"E_t\}\$ \*\*= (\d+) GeV",
            L.line(r"$\boldsymbol{E_p \lt E_t}$ **= "),
        ).group(1)
    )
    e_1 = 1000 * sp.Integer(
        re.search(r"\$E_1\$ = (\d+) TeV", L.line(FADE_TEXT)).group(1)
    )
    c_k = L.to_sympy(
        L.sides(L.part(L.display(FADE_PIN), "c_K(E_p) ="))[1],
        {"c_t": ct, "E_p": Ep, "E_t": e_t, "E_1": e_1},
    )
    return sp.lambdify((ct, Ep), c_k, "numpy"), float(e_t), float(e_1)


@derivation("pp.fade")
def check_pp_fade():
    """Above E_t the code multiplies Koers et al.'s spectra by c_K, which is c_t
    at E_t and 1 at E_1. With c_t, the kaons per collision are continuous with
    the page's fit at E_t. E_1 is the lowest proton energy of Koers et al.'s
    simulations. The paragraph's figures: c_t of each kaon, and Koers et al.'s
    K+ against Table 11's above E_t."""
    nkpw, Et_, E1_, ct = sp.symbols("n_KPW E_t E_1 c_t", positive=True)
    disp = L.display(FADE_PIN)
    c_k = L.to_sympy(
        L.sides(L.part(disp, "c_K(E_p) ="))[1],
        {"c_t": ct, "E_p": Ep, "E_t": Et_, "E_1": E1_},
    )
    n_, f_, _ = readme_delta(nkpw, Et_)
    s_t = L.to_sympy(
        L.sides(L.part(L.display(DELTA_PIN), "s ="))[1], {"m_p": m, "E_p": Et_}
    )
    c_t = L.to_sympy(
        L.sides(L.part(disp, "c_t ="))[1],
        {r"\langle n_K\rangle_{\rm KPW}(E_t)": nkpw, "s_t": s_t},
        {"f": f_},
    )
    text = braced(L.line(FADE_TEXT))

    def figures():
        num = r"([\d.]+)"
        t_p, t_m, t_0 = re.search(
            rf"The fit's \$\\mathrm\{{K\}}\^\+\$ are {num} times theirs, "
            rf"its \$\\mathrm\{{K\}}\^-\$ {num} times and its neutral kaons "
            rf"{num} times",
            braced(L.line(FADE_TEXT)),
        ).groups()
        r1, rs1, r2, rs2 = re.search(
            rf"above \$E_t\$: {num} times at \$\\sqrt\{{s\}}\$ = {num} GeV and "
            rf"{num} times at {num} GeV",
            braced(L.line(FADE_TEXT)),
        ).groups()
        lo, hi, rs_lo, rs_hi = re.search(
            rf"There their \$\\mathrm\{{K\}}\^\+\$ are {num} to {num} times "
            rf"the measured ones, at \$\\sqrt\{{s\}}\$ = {num} to {num} GeV",
            braced(L.line(r"There their $\mathrm K^+$ are")),
        ).groups()
        e_t = float(
            re.search(
                r"E_t\}\$ \*\*= (\d+) GeV",
                L.line(r"$\boldsymbol{E_p \lt E_t}$ **= "),
            ).group(1)
        )
        e_1 = 1e3 * float(re.search(r"\$E_1\$ = (\d+) TeV", text).group(1))
        fit, f = readme_fit(), f_numeric()
        s_of = lambda e_p: 2 * MASS["p"] ** 2 + 2 * MASS["p"] * e_p
        fit_at = lambda kind, e_p: float(
            f(*fit[kind][:5], s_of(e_p), fit[kind][5] ** 2)
        )
        cts = {
            k: fit_at(k, e_t) / kpw_multiplicity(k, e_t)
            for k in ("K+", "K-", "K0S")
        }
        kplus = [(rs, n["K+"]) for ref, rs, n in na49_table() if "K+" in n]
        ratio = lambda rs, v: (
            kpw_multiplicity(
                "K+", (rs**2 - 2 * MASS["p"] ** 2) / (2 * MASS["p"])
            )
            / v
        )
        at = {
            rs: ratio(rs, v)
            for rs, v in kplus
            if rs in (float(rs1), float(rs2))
        }
        high = [
            (rs, ratio(rs, v))
            for rs, v in kplus
            if (rs**2 - 2 * MASS["p"] ** 2) / (2 * MASS["p"]) >= e_1
        ]
        return [
            Result(
                f"c_t = f(s_t)/<n_K>_KPW(E_t) at E_t = {e_t:g} GeV: "
                f"K+ {t_p}, K- {t_m}, K0_S {t_0}",
                all(
                    rounds_to(cts[k], v)
                    for k, v in zip(("K+", "K-", "K0S"), (t_p, t_m, t_0))
                ),
                ", ".join(f"{k} {v:.4f}" for k, v in cts.items()),
            ),
            Result(
                f"Koers et al.'s K+ against Table 11: {r1} at sqrt(s) = "
                f"{rs1} GeV, {r2} at {rs2} GeV",
                len(at) == 2
                and rounds_to(at[float(rs1)], r1)
                and rounds_to(at[float(rs2)], r2),
                ", ".join(f"{rs}: {v:.3f}" for rs, v in at.items()),
            ),
            Result(
                f"from E_1 on, their K+ are {lo}-{hi} times the "
                f"measured ones (sqrt(s) = {rs_lo}-{rs_hi} GeV)",
                all(v == v for _, v in high)
                and rounds_to(min(v for _, v in high), lo)
                and rounds_to(max(v for _, v in high), hi)
                and (high[0][0], high[-1][0]) == (float(rs_lo), float(rs_hi)),
                ", ".join(f"{rs}: {v:.3f}" for rs, v in high),
            ),
        ]

    return [
        exact("c_K(E_t) = c_t", c_k.subs(Ep, Et_) - ct),
        exact(
            "c_K(E_1) = 1: Koers et al.'s spectra from E_1 on",
            c_k.subs(Ep, E1_) - 1,
        ),
        exact(
            "continuous at E_t: c_K(E_t) <n_K>_KPW(E_t) = f(s_t) = "
            "n-tilde_K(E_t)",
            c_k.subs(Ep, Et_).subs(ct, c_t) * nkpw - n_.subs(Ep, Et_),
        ),
        in_source(
            "E_1: Koers et al.'s simulations start at 1 TeV (their abstract)",
            KOERS,
            "incident proton energies from 1 TeV to 1 PeV",
        ),
        Result(
            f"E_1 = 1 TeV on the page",
            re.search(r"\$E_1\$ = 1 TeV, the lowest proton energy", text)
            is not None,
            "",
        ),
    ] + with_control(
        figures,
        "control: 1.40 for 1.30 (the fit's K+ against theirs at E_t)",
        "The fit's $\\mathrm K^+$ are 1.30 times theirs",
        "The fit's $\\mathrm K^+$ are 1.40 times theirs",
    )


@derivation("pp.loss")
def check_pp_loss():
    """The protons lose the energy the kaons of Q_K take, in the form of AM3's
    rate for the pions.
    The term is -n_H c sigma_inel kappa_K E_p, with kappa_K = int x F_K dx."""

    def term(
        pattern=r"gains the term \$(.+?)\$, with \$\\kappa_K\$ the "
        r"fraction of \$E_p\$ the kaons carry",
        fraction=r"\sum_K \kappa_K",
    ):
        tex = re.search(
            pattern, L.line(r"energy-loss rate $\dot E/E = ")
        ).group(1)
        return L.to_sympy(
            tex.replace(fraction, r"\kappa"),
            {
                "n_H": nH,
                "c": c,
                r"\sigma_\mathrm{inel}": SIGI(Ep),
                r"\kappa": sp.Symbol("kappa"),
            },
        )

    pions = lambda: term(
        r"energy-loss rate \$\\dot E/E = (.+?)\$ "
        r"gains the term",
        r"\sum_\pi \kappa_\pi",
    )
    kappa = sp.Integral(x * FKF(x, Ep), (x, 0, 1))
    weight, body = readme_pp_source()
    made = sp.Integral(EK * weight * body, (EK, 0, Ep)).transform(
        EK, (x * Ep, x)
    )
    lost = lambda t: -t.subs(sp.Symbol("kappa"), kappa) * Ep * NPF(Ep)
    (f_made, lim_made), (f_lost, lim_lost) = (
        integrand(made),
        integrand(lost(term())),
    )
    with L.patched(r"\sum_K \kappa_K$, with", r"\sum_K \kappa_K/2$, with"):
        f_broken, _ = integrand(lost(term()))
    with L.patched(
        r"\sum_\pi \kappa_\pi$ gains", r"\sum_\pi \kappa_\pi/2$ gains"
    ):
        pions_broken = pions()
    return [
        exact(
            "the kaons' energy from the protons of E_p = the energy "
            "these protons lose by the README's term (any sigma_inel, "
            "N_p, F_K; integrands in x)",
            f_made - f_lost,
        ),
        Result(
            "both over x from 0 to 1",
            lim_made == lim_lost,
            f"{lim_made}, {lim_lost}",
        ),
        exact_fails("control: kappa_K/2", f_made - f_broken),
        exact(
            "the term has the form of AM3's rate for the pions, "
            "with their kappa_pi for kappa_K",
            pions() - term(),
        ),
        exact_fails("control: kappa_pi/2 in AM3's rate", pions_broken - term()),
    ]


KEL_THR = r"\left(1-\frac{m_\pi}{x\,E_p}\right)^{1/2}"
# Koers et al.'s Table III (columns pi+, pi-, pi0, K+, K-, K0, anti-K0): the
# README has the last five
K_T3_ROWS = [
    r"$p_{00}$&$-0.497$&$-0.501$&$-0.456$&$-1.23$&$-1.46$&$-1.29$&$-1.50$\\",
    r"$p_{01}$&$0.0934$&$0.0934$&$0.0950$&$0.0657$&$0.101$&$0.0690$&$0.101$\\",
    r"$p_{10}$&$-0.140$&$-0.128$&$-0.142$&$-0.147$&$-0.109$&$-0.142$&$-0.118$"
    r"\\",
    r"$p_{11}$&$0.0131$&$0.0118$&$0.0135$&$0.0161$&$0.00865$&$0.0154$&"
    r"$0.0101$\\",
    r"$p_{20}$&$-0.455$&$-0.437$&$-0.457$&$-0.00411$&$-0.0577$&$-0.00717$&"
    r"$-0.0567$\\",
    r"$p_{21}$&$0.495$&$0.494$&$0.494$&$0.493$&$0.491$&$0.493$&$0.491$\\",
    r"$p_{30}$&$-2.06$&$-0.945$&$-1.49$&$-0.989$&$-1.22$&$-1.03$&$-1.25$\\",
    r"$p_{40}$&$-0.896$&$-1.03$&$-0.981$&$-0.345$&$-0.164$&$-0.294$&$-0.169$\\",
    r"$p_{50}$&$1.11$&$0.963$&$1.01$&$0.777$&$1.04$&$0.839$&$1.05$\\",
    r"$p_{60}$&$0.791$&$0.598$&$0.723$&$-0.235$&$-0.279$&$-0.272$&$-0.272$\\",
    r"$p_{70}$&$37.7$&$15.3$&$22.1$&$42.7$&$18.6$&$33.8$&$21.2$\\",
    r"$p_{80}$&$7.69$&$7.23$&$8.53$&$12.0$&$4.23$&$10.1$&$4.07$\\",
]
K_COLUMNS = ("pi+", "pi-", "pi0", "K+", "K-", "K0", "K0b")
ISR = r"\cite{albrow2,albrow3,albrow4,albrow5,capi,alper,guettler}"
# Anticic et al.'s Table 11: ref., sqrt(s) [GeV], F(0) [mb] and <n> of K+, K-,
# K0_S, where a value in (...) is derived
NA49_T11 = [
    r"\cite{louttit}&2.9&&&&0.00462&&0.00082\\",
    r"\cite{hogan,reed}&2.9&0.042&&&0.00481&&\\",
    r"\cite{alex}&3.45&&&&0.00802&&0.00294\\",
    r"\cite{eisner}&3.59&&&&&&0.00670\\",
    r"\cite{fire}&4.04&&&&0.01760&0.00080&0.00719\\",
    r"\cite{blobel}&4.9&&&0.120&&&0.0190\\",
    r"\cite{fesef}&4.9&&&&0.0473&0.00747&0.0198\\",
    r"\cite{boggild}&6.1&&&0.185&&&0.0420\\",
    r"\cite{blobel}&6.8&&&0.206&&&0.0410\\",
    r"\cite{fesef}&6.8&&&&0.0999&0.0330&0.0493\\",
    r"\cite{allaby1}&6.84&0.440&0.120&(0.280)&0.107&0.0262&(0.0666)\\",
    r"\cite{bogo}&7.8&&&0.300&&&0.0636\\",
    r"\cite{abramov}&11.5&0.549&0.322&(0.435)&&&\\",
    r"\cite{ammosov}&11.5&&&0.375&&&0.109\\",
    r"\cite{alston}&13.8&&&&&&0.121\\",
    r"\cite{chapman}&13.9&&&0.505&&&0.146\\",
    r"\cite{brick}&16.7&&&0.490&&&0.158\\",
    r"NA49&17.2&0.672&0.477&(0.575)&0.227&0.130&(0.179)\\",
    r"\cite{jaeger2}&19.7&&&0.590&&&0.181\\",
    ISR + r"&23.0&0.718&0.547&(0.633)&0.273&0.171&(0.222)\\",
    r"\cite{sheng}&23.8&&&0.670&&&0.224\\",
    r"\cite{lopinto}&23.8&&&&&&0.212\\",
    r"\cite{bailly}&25.7&&&0.670&&&0.262\\",
    r"\cite{kass}&27.4&&&&&&0.200\\",
    r"\cite{kichimi}&27.6&&&0.680&&&0.232\\",
    ISR + r"&31.0&0.767&0.614&(0.691)&0.327&0.220&(0.274)\\",
    ISR + r"&45.0&0.861&0.714&(0.788)&0.409&0.290&(0.350)\\",
    ISR + r"&52.0&0.907&0.766&(0.837)&0.448&0.328&(0.388)\\",
    ISR + r"&63.0&0.959&0.811&(0.885)&0.493&0.363&(0.428)\\",
    ISR + r"&200.0&1.361&1.192&(1.277)&0.819&0.651&(0.735)\\",
    r"\cite{ansorge1}&200.0&&&1.680&&&0.700\\",
    r"\cite{ansorge1}&546.0&&&2.306&&&1.000\\",
]
SPECIES = {
    r"$\mathrm{K}^+$": "K+",
    r"$\mathrm{K}^-$": "K-",
    r"$\mathrm{K}^0$": "K0",
    r"$\bar{\mathrm{K}}^0$": "K0b",
    r"$\pi^0$": "pi0",
}
CHANNELS = {
    # the README's fit table
    r"$\mathrm{pp} \to \mathrm{p}\Lambda\mathrm{K}^+$": ("p", "Lambda", "K+"),
    r"$\mathrm{pp} \to \mathrm{ppK}^+\mathrm{K}^-$": ("p", "p", "K+", "K-"),
    r"$\mathrm{pp} \to \mathrm{p}\Sigma^+\mathrm{K}^0$": ("p", "Sigma+", "K0"),
}
FIT_TABLE = (
    r"| Kaon | $A$ | $a$ | $b$ | $C$ | $d$ | $\sqrt{s_\mathrm{th}}$ "
    r"[GeV] | Threshold channel |"
)


def fit_kind(label):
    label = braced(label)
    return (
        "K0S"
        if label.startswith(r"$\mathrm{K}^0_\mathrm{S}$")
        else SPECIES[label]
    )


def fields(row):
    return [v.strip() for v in row.rstrip(" \\").split("&")]


def na49_table():
    out = []
    for row in NA49_T11:
        c = fields(row)
        n = {
            k: float(v)
            for k, v in zip(("K+", "K-", "K0S"), c[5:])
            if v and not v.startswith("(")
        }
        out.append((c[0], float(c[1]), n))
    return out


def readme_params():
    line = L.line(r"with $\epsilon_3 = 10^{p_2+p_4}$")
    spans = re.findall(
        r"\$([^$]+?) = ([^$]+?)\$", line.split(r", and $n = 0$")[0]
    )
    return {lhs: L.to_sympy(rhs, TOK_K) for lhs, rhs in spans}


def koers_table3():
    return {
        re.match(r"\$(p_\{\d\d\})\$", row).group(1): [
            sp.Rational(v) for v in re.findall(r"\$(-?[\d.]+)\$", row)
        ]
        for row in K_T3_ROWS
    }


def readme_table3():
    head = L.line(r"| | $p_{00}$ | $p_{01}$ |")
    rows = {
        braced(cells[0]): [sp.Rational(v.replace("−", "-")) for v in cells[1:]]
        for cells in L.table(head)
    }
    return re.findall(r"\$(p_\{\d\d\})\$", head), rows


@derivation("pp.etrans")
def check_etrans():
    """The kaons' E_t is the transition energy E_t of AM3's pions (AM3's
    E_trans). The README gives the kaons' E_t in the heading of the range below
    it, where they take their delta-functional approximation. It gives the
    pions' E_t where it starts the pp collisions."""

    def results():
        readme = sp.nsimplify(
            L.to_sympy(
                re.search(
                    r"Above the transition energy \$E_t\$ = ([\d.]+) GeV, "
                    r"AM³ makes pions",
                    L.line(r"Above the transition energy $E_t$ = "),
                ).group(1),
                {},
            ),
            rational=True,
        )
        e_t = sp.Rational(
            re.search(
                r"E_t\}\$ \*\*= (\d+) GeV",
                L.line(r"$\boldsymbol{E_p \lt E_t}$ **= "),
            ).group(1)
        )
        return [
            Result(
                "the kaons' E_t (their delta-functional approximation "
                "below it) = the E_t of AM3's pions (E_trans)",
                e_t == readme,
                f"E_t = {e_t} GeV",
            )
        ]

    return with_control(
        results,
        "control: the pions' E_t = 200 GeV in the README",
        r"Above the transition energy $E_t$ = 100 GeV",
        r"Above the transition energy $E_t$ = 200 GeV",
    )


def fit_points():
    top = float(
        re.search(
            r"\$\\sqrt\{s\} \\le\$ (\d+) GeV", L.line(r"$\sqrt{s} \le$ ")
        ).group(1)
    )
    return {
        k: [
            (rs, n[k])
            for ref, rs, n in na49_table()
            if k in n
            and rs <= top
            and not (ref == r"\cite{fesef}" and k != "K0S")
        ]
        for k in ("K+", "K-", "K0S")
    }


def readme_f():
    syms = sp.symbols("A a b C d s s_th", positive=True)
    tokens = dict(zip(("A", "a", "b", "C", "d", "s", r"s_{\rm th}"), syms))
    return L.to_sympy(
        L.sides(L.part(L.display(DELTA_PIN), "f(s) ="))[1], tokens
    ), syms


def readme_fit():
    return {
        fit_kind(cells[0]): tuple(
            0.0 if v == "—" else float(v) for v in cells[1:7]
        )
        for cells in L.table(FIT_TABLE)
    }


def f_numeric():
    expr, syms = readme_f()
    return sp.lambdify(syms, expr, "numpy")


def fit_kind_params(kind, rth, f):
    """Least squares in ln n of the README's f(s) to the points of a kind at
    rth, the unrounded threshold of its channel. For K- and K0S, it returns
    (A, a, b) with C = 0 (a linear problem in ln A, a, b). For the K+, whose f
    is not linear in its parameters, it returns (A, a, b, C, d) from the best of
    a grid of starts."""
    pts = fit_points()[kind]
    s_ = np.array([rs**2 for rs, _ in pts])
    y = np.log([v for _, v in pts])
    if kind != "K+":
        sol = np.linalg.lstsq(
            np.stack(
                [
                    np.ones_like(s_),
                    np.log(1 - rth**2 / s_),
                    np.log(s_ / rth**2),
                ],
                axis=1,
            ),
            y,
            rcond=None,
        )[0]
        return (math.exp(sol[0]), sol[1], sol[2])
    best = None
    for start in itertools.product(
        (0.05, 0.1, 0.2),
        (2.0, 4.0, 6.0),
        (0.25,),
        (0.1, 0.5, 1.0),
        (2.0, 5.0, 8.0),
    ):
        r = least_squares(
            lambda p: np.log(f(*p, s_, rth**2)) - y,
            start,
            bounds=([1e-6, 0, -2, 0, 0], [10, 20, 3, 100, 40]),
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
        best = r if best is None or r.cost < best.cost else best
    return tuple(best.x)


@derivation("pp.fit")
def check_pp_fit():
    """The README's multiplicity fit is least squares in ln n of its f(s) to
    NA49's Table 11, at the thresholds of the channels. Only the K+ has the
    second term."""
    out = [in_source("NA49's Table 11 as transcribed", NA49, "".join(NA49_T11))]
    kinds = {"K+": ("K+",), "K-": ("K-",), "K0S": ("K0", "K0b")}
    f = f_numeric()
    fitted = {}
    for cells in L.table(FIT_TABLE):
        kind, final = fit_kind(cells[0]), CHANNELS[braced(cells[7])]
        rth = sum(MASS[q] for q in final)
        # the lowest channel making the kaon: two baryons and one kaon, or two
        # nucleons and a kaon pair (pions add mass)
        cands = [
            c
            for c in (
                final_states(2, NUCLEON, ("Lambda",) + SIGMA, KAON)
                + final_states(2, NUCLEON, NUCLEON, KAON, ANTIKAON)
            )
            if any(k in c for k in kinds[kind])
            and sum(STRANGENESS.get(q, 0) for q in c) == 0
        ]
        lowest = min(cands, key=lambda c: sum(MASS[q] for q in c))
        got = fitted[kind] = fit_kind_params(kind, rth, f)
        printed = cells[1:6] if kind == "K+" else cells[1:4]
        out += [
            Result(
                f"{kind}: sqrt(s_th) = {cells[6]} GeV is {'+'.join(final)}, "
                "the lowest channel that makes it",
                rounds_to(rth, cells[6])
                and sum(CHARGE[q] for q in final) == 2
                and sorted(lowest) == sorted(final),
                f"{rth:.5f} GeV",
            ),
            Result(
                f"{kind}: {'A, a, b, C, d' if kind == 'K+' else 'A, a, b'} "
                f"= {', '.join(printed)} (least squares in ln n, "
                f"{len(fit_points()[kind])} points)"
                + ("" if kind == "K+" else f"; C = {cells[4]}, d {cells[5]}"),
                all(rounds_to(g, c_) for g, c_ in zip(got, printed))
                and (kind == "K+" or (cells[4], cells[5]) == ("0", "—")),
                ", ".join(f"{g:.6g}" for g in got),
            ),
        ]
    with L.patched(r"| 0.1132 | 4.099 |", r"| 0.1142 | 4.099 |"):
        broken = [c[1] for c in L.table(FIT_TABLE)]
    return (
        out
        + [
            Result(
                "control: A = 0.1142 for 0.1132 (K+)",
                not rounds_to(fitted["K+"][0], broken[0]),
                "fails, as it must",
            )
        ]
        + with_control(
            kplus_term_results,
            "control: 7.4 for 7.3 "
            "(the first term alone below the K+ points at 2.9 GeV)",
            "lies 7.3 and 7.6 times below",
            "lies 7.4 and 7.6 times below",
        )
    )


KPLUS_TERM = r"Only the $\mathrm K^+$ has the second term."
WORDS = {
    "two": 2,
    "three": 3,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
}


def kplus_term_results():
    """The page's paragraph on the K+'s second term, recomputed. The term
    follows the two points at sqrt(s) = 2.9 GeV. The first term alone, fitted
    to the other points, lies below them. Near threshold, 1 - s_th/s is
    proportional to Q = sqrt(s) - sqrt(s_th), so the term grows as Q^2. The
    nonrelativistic phase space of three particles grows the same way. Their
    kinetic energy is quadratic in the momenta, and the measure of two free
    momenta is six-dimensional."""
    text, num = braced(L.line(KPLUS_TERM)), r"([\d.]+)"
    n_two, rs_left, above = re.search(
        r"It follows the (\w+) \$\\mathrm\{K\}\^\+\$ points at \$\\sqrt\{s\}\$ "
        rf"= {num} GeV, {num} GeV above the threshold",
        text,
    ).groups()
    below = re.search(
        r"Without it, the first term, fitted to the other (\w+) points, "
        rf"lies {num} and {num} times below them",
        text,
    ).groups()
    rs_cut, share = re.search(
        rf"from \$\\sqrt\{{s\}}\$ = {num} GeV on, "
        rf"it is less than (\d+)% of the fit",
        text,
    ).groups()
    n_all, lo, hi = re.search(
        r"The (\w+) \$\\mathrm\{K\}\^\+\$ points lie at "
        rf"{num} to {num} times the fit",
        text,
    ).groups()
    A, a, b, C, d, rs_th = readme_fit()["K+"]
    rth = sum(MASS[q] for q in ("p", "Lambda", "K+"))
    f = f_numeric()
    pts = fit_points()["K+"]
    near = [p_ for p_ in pts if p_[0] == float(rs_left)]
    rest = [p_ for p_ in pts if p_ not in near]
    s_ = np.array([rs**2 for rs, _ in rest])
    sol = np.linalg.lstsq(
        np.stack(
            [np.ones_like(s_), np.log(1 - rth**2 / s_), np.log(s_ / rth**2)],
            axis=1,
        ),
        np.log([v for _, v in rest]),
        rcond=None,
    )[0]
    first = lambda rs: (
        math.exp(sol[0])
        * (1 - rth**2 / rs**2) ** sol[1]
        * (rs**2 / rth**2) ** sol[2]
    )
    off = sorted(v / first(rs) for rs, v in near)
    Q, St = sp.symbols("Q s_th", positive=True)
    xq = 1 - St / (sp.sqrt(St) + Q) ** 2
    lead = sp.limit(xq / Q, Q, 0)
    lam = sp.Symbol("lambda", positive=True)
    p1, p2 = sp.symbols("p1:4", real=True), sp.symbols("q1:4", real=True)
    m1, m2, m3 = sp.symbols("m1:4", positive=True)
    T = (
        sum(c**2 for c in p1) / (2 * m1)
        + sum(c**2 for c in p2) / (2 * m2)
        + sum((c1 + c2) ** 2 for c1, c2 in zip(p1, p2)) / (2 * m3)
    )
    homogeneous = sp.simplify(
        T.subs({c: lam * c for c in p1 + p2}, simultaneous=True) - lam**2 * T
    )
    # phase space: d^6p delta(Q - T(p)) with p = sqrt(Q) u: d^6p = Q^(6/2) d^6u,
    # delta(Q (1 - T(u))) = delta(1 - T(u))/Q
    power = sp.Rational(len(p1 + p2), 2) - 1
    tail = lambda rs: (
        C
        * (1 - rs_th**2 / rs**2) ** 2
        * (rs_th**2 / rs**2) ** d
        / f(A, a, b, C, d, rs**2, rs_th**2)
    )
    grid = np.linspace(rs_th + 1e-6, 1e3, 400000)
    shares = tail(grid)
    first_below = grid[np.argmax(shares < float(share) / 100)]
    after = shares[grid >= first_below]
    ratios = [v / f(A, a, b, C, d, rs**2, rs_th**2) for rs, v in pts]
    return [
        Result(
            f"the second term follows the {n_two} K+ points at sqrt(s) = "
            f"{rs_left} GeV, {above} GeV above the threshold",
            WORDS[n_two] == len(near) == 2
            and rounds_to(float(rs_left) - rth, above),
            f"{[rs for rs, _ in near]}, {float(rs_left) - rth:.4f} GeV",
        ),
        Result(
            f"the first term alone, fitted to the other {below[0]} points, "
            f"lies {below[1]} and {below[2]} times below them",
            WORDS[below[0]] == len(rest)
            and all(rounds_to(o, b_) for o, b_ in zip(off, below[1:])),
            ", ".join(f"{o:.3f}" for o in off),
        ),
        exact(
            "near threshold 1 - s_th/s = 2Q/sqrt(s_th) + O(Q^2): "
            "proportional to Q, the term to Q^2",
            lead - 2 / sp.sqrt(St),
        ),
        exact(
            "three free particles: their kinetic energy is quadratic in the "
            "momenta, T(lambda p) = lambda^2 T(p)",
            homogeneous,
        ),
        exact(
            "so their phase space d^6p delta(Q - T) grows as "
            "Q^((6/2) - 1) = Q^2",
            power - 2,
        ),
        Result(
            f"from sqrt(s) = {rs_cut} GeV on, the second term is less than "
            f"{share} % of the fit (the table's)",
            rounds_to(first_below, rs_cut)
            and bool(np.all(after < float(share) / 100)),
            f"from {first_below:.3f} GeV; at 63 GeV {tail(63.0):.1e}",
        ),
        Result(
            f"the {n_all} K+ points lie at {lo} to {hi} times the fit",
            WORDS[n_all] == len(pts)
            and all(r == r for r in ratios)
            and rounds_to(min(ratios), lo)
            and rounds_to(max(ratios), hi),
            f"{len(pts)} points, {min(ratios):.3f}-{max(ratios):.3f}",
        ),
    ]
