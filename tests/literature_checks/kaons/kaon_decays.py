"""Derivation checks of the kaon decays (the section "Decays" of
tests/literature_checks/kaons/README.md) against their sources:
  - the PDG 2026 listings of K+- and K0_L: the expansions and definitions that
    the page quotes, transcribed below with their printed page numbers. For
    K0_L -> 3pi0, the listing gives only h, the coefficient of u^2 (p. 22). The
    page's 1 + h(u^2 + v^2/3) is derived from PDG's expansion (p. 20).
  - Chounet, Gaillard & Gaillard (1972, Phys. Rep. 4, 199), eqs. 6.6-6.9
    (pp. 249-250), transcribed below from the scan in their notation: kaon
    momentum k and mass M, pion momentum q and mass mu, lepton mass m and
    energy E_mu (also for K_e3), and E_nu = M - E_pi - E_mu.
  - Lipari, Lusignoli & Meloni (2007), the TeX source of arXiv:0704.0718v1:
    eqs. 13, 15, 16, 18 and 19.
The kinematics that the page states without a source is derived again exactly.
K -> mu nu, where angular momentum makes the mu+ left-handed, fixes the sign
convention of the spin s (kd.W). The numeric checks take the masses from
src/am3_mojo/utils/constants.mojo, as inputs.

branching_ratios() reads the page's branching ratios for
kaons/multibody_strangeness.py.

Run from the repository root:
  .venv/bin/python tests/literature_checks/common/run.py kaon_decays
"""

import re
from itertools import permutations

import mpmath as mp
import sympy as sp

import latex
import mojo_constants
from latex import LIPARI
from verify import (
    PDG,
    Entry,
    Result,
    derivation,
    dirac,
    exact,
    exact_fails,
    in_source,
    mdot,
    slash,
    worst_of,
)

L = latex.page("kaons")

KD = "kaons"
CGG = "scan chounet1972.pdf"
# this module's equations in the inventory (common/inventory.py)
INVENTORY = [
    Entry(
        "kd.asym",
        r"probability $\left(1 + A_{L}\right)/2$",
        "standard",
        "charge asymmetry: PDG 2026, K0_L listing pp. 34-35",
        "PDF PDG-2026_listing_K-zero-L.pdf",
        file=KD,
    ),
    Entry(
        "kd.modes",
        r"F_{s\to X}(x) = \frac{\sum_m B_m\thinspace \nu_{mX}\thinspace "
        r"F_{mX}(x)}{\sum_m B_m}",
        "model",
        "the modes of the table (PDG 2026), renormalised to their sum",
        PDG,
        checks="kaons/multibody_strangeness.py (the K0_S -> pi+ pi- ratio)",
        file=KD,
    ),
    Entry(
        "kd.current",
        r"$H = f_+(t)(p_K + p_\pi) + f_-(t)(p_K - p_\pi)$",
        "standard",
        "K_l3 hadronic current: Chounet, Gaillard & Gaillard 1972, "
        "eqs. 6.7-6.8",
        CGG,
        file=KD,
    ),
    Entry(
        "kd.W",
        r"\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s)",
        "derived",
        "V-A trace; W_0: Chounet, Gaillard & Gaillard 1972, eq. 6.9 "
        "(kd.cgg_w0)",
        CGG,
        file=KD,
    ),
    Entry(
        "kd.spin_sum",
        r"summed over its spin, $\vert\mathcal M\vert^2 \propto 2W_0$",
        "derived",
        "the spin sum of W_0 + W_pol(s) (W_pol linear in s)",
        file=KD,
    ),
    Entry(
        "kd.cgg",
        r"\rho \propto Af_+^2 + Bf_+f_- + Cf_-^2",
        "transcribed",
        "Chounet, Gaillard & Gaillard 1972, eq. 6.9 (pp. 249-250)",
        CGG,
        file=KD,
    ),
    Entry(
        "kd.cgg_w0",
        r"$W_0 = 4M\thinspace\rho$",
        "derived",
        "the V-A density W_0 against Chounet, Gaillard & Gaillard "
        "1972, eq. 6.9",
        CGG,
        file=KD,
    ),
    Entry(
        "kd.fminus",
        r"$f_- = (f_0 - f_+)(M^2 - m_\pi^2)/t$",
        "rearranged",
        "kd.f0 solved for f_-",
        PDG,
        file=KD,
    ),
    Entry(
        "kd.f0",
        r"$f_0 = f_+ + f_-\thinspace t/(m_K^2 - m_\pi^2)$",
        "transcribed",
        "PDG 2026, K± listing p. 37 (K0_L p. 23): the definition of f_0",
        PDG,
        file=KD,
    ),
    Entry(
        "kd.formfactors",
        r"f_+(t) = 1 + \lambda'_ +\frac{t}{m_{\pi^+}^2}",
        "transcribed",
        "PDG 2026, K± listing p. 37 (K0_L p. 23)",
        PDG,
        file=KD,
    ),
    Entry(
        "kd.fminus_code",
        r"f_-(t) = (M^2 - m_\pi^2)\left[\frac{\lambda_0 - \lambda'_ "
        r"+}{m_{\pi^+}^2}",
        "derived",
        "kd.fminus with the expansions of kd.formfactors",
        file=KD,
    ),
    Entry(
        "kd.dalitz",
        r"\vert \mathcal M\vert^2 \propto 1 + g\thinspace u + h\thinspace "
        r"u^2 + k\thinspace v^2",
        "transcribed",
        "PDG 2026, K± listing p. 32 (note, eq. 1)",
        PDG,
        note="its values: kd.dalitz_values",
        file=KD,
    ),
    Entry(
        "kd.dalitz_values",
        r"$= (-0.21134, 0.01848, -0.00463)$",
        "transcribed",
        "PDG 2026, K0_L listing p. 42: the CP-violating j and f the page "
        "leaves out (the values: K± listing pp. 33-35, K0_L listing "
        "pp. 21-22)",
        PDG,
        file=KD,
    ),
    Entry(
        "kd.KL3pi0",
        r"$1 + h(u^2 + v^2/3)$",
        "derived",
        "PDG 2026 (K0_L listing pp. 20, 22) with Bose symmetry",
        "PDF PDG-2026_listing_K-zero-L.pdf",
        file=KD,
    ),
    Entry(
        "kd.phase_space",
        r"$d\Gamma \propto \rho\thinspace dE_i\thinspace dE_j$",
        "standard",
        "three-body phase space",
        file=KD,
    ),
    Entry(
        "kd.frame",
        r"p_K\negthinspace\cdot\negthinspace p_a = ME_a",
        "derived",
        "kinematics (kaon frame, p_K = p_a + p_b + p_c)",
        file=KD,
    ),
    Entry(
        "kd.spin",
        r"s = \frac{1}{m_\ell}\left(\vert\vec p_\ell\vert,\thinspace "
        r"E_\ell\thinspace\hat p_\ell\right)",
        "standard",
        "the spin 4-vector of helicity +1/2 and its products",
        file=KD,
    ),
    Entry(
        "kd.line",
        r"E_j^{\rm lo,hi} = "
        r"\frac{(M - E^\ast)\thinspace E_j' \mp p^\ast p_j'}{m_{jk}}",
        "derived",
        "kinematics (the boundary of the Dalitz plot)",
        file=KD,
    ),
    Entry(
        "kd.g",
        r"g(E^\ast) = \frac{1}{Z}\int_{E_j^{\rm lo}}^{E_j^{\rm hi}} "
        r"\rho\thinspace dE_j",
        "derived",
        "the Dalitz density (kd.phase_space) projected on one energy",
        file=KD,
    ),
    Entry(
        "kd.quad",
        r"$E^\ast = m + (E_{\max} - m)\sin^2(\pi u/2)$",
        "model",
        "the rest-frame quadratures of decays.mojo (GLRules)",
        checks="tests/unit/test_added_processes.py "
        "(test_decays_energy_and_positivity)",
        file=KD,
    ),
    Entry(
        "kd.flat",
        r"$x \in [(E^\ast - p^\ast)/M, (E^\ast + p^\ast)/M]$ with density "
        r"$M/(2p^\ast)$",
        "derived",
        "kinematics (isotropic decay, ultra-relativistic parent)",
        file=KD,
    ),
    Entry(
        "kd.F",
        r"F(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}^{E_{\max}} "
        r"\frac{g(E^\ast)}{p^\ast}",
        "derived",
        "kinematics",
        file=KD,
    ),
    Entry(
        "kd.two_body",
        r"$E^\ast = (M^2 + m^2 - m_b^2)/(2M)$",
        "standard",
        "two-body kinematics",
        file=KD,
    ),
    Entry(
        "kd.htilde",
        r"$\tilde h = (E^\ast M x - m^2)/(p^\ast M x)$",
        "derived",
        "kinematics (decay-plane average)",
        file=KD,
    ),
    Entry(
        "kd.FRL",
        r"F_{R,L}(x) = \frac{M}{2}\int",
        "derived",
        "kinematics",
        file=KD,
    ),
    Entry(
        "kd.lipari",
        r"F_{\mu_R^+}(x) = \frac{r\thinspace (1 - x)}{(1 - r)^2\thinspace x}",
        "derived",
        "kd.FRL for K -> mu nu = Lipari, Lusignoli & Meloni 2007, eqs. 18-19 "
        "(m_pi -> m_K); eqs. 15-16",
        "TeX " + LIPARI,
        file=KD,
    ),
    Entry(
        "kd.scaling",
        r"\frac{dn_{a \to b}}{dE}(E_b; E_a) = \frac{1}{E_a}",
        "transcribed",
        "Lipari, Lusignoli & Meloni 2007, eq. 13",
        "TeX " + LIPARI,
        file=KD,
    ),
    Entry(
        "kd.source",
        r"Q_X(E) = \sum_s \int_E^\infty",
        "standard",
        "decay source: the decays counted, with kd.scaling (as nd.source)",
        file=KD,
    ),
    Entry(
        "kd.persource",
        r"E\thinspace Q_X(E) = \sum_s \int_{\ln E}^\infty",
        "derived",
        "kd.source per unit ln E",
        file=KD,
    ),
    Entry(
        "kd.grid",
        r"E_iQ_X(E_i) = \Delta\sum_s\sum_j",
        "derived",
        "kd.persource on the logarithmic grid",
        file=KD,
    ),
    Entry(
        "kd.table",
        r"T_k = \frac{e_k - x_{k+1}n_k}{\Delta\thinspace (x_k - x_{k+1})}",
        "model",
        "the sharing of decays.mojo (kaon_decay_tables), keeping number "
        "and energy",
        checks="tests/unit/test_added_processes.py "
        "(test_decays_energy_and_positivity)",
        file=KD,
    ),
    Entry(
        "kd.node",
        r"\int_{x_a}^{x_b} \frac{M}{2p^\ast}\left(w_g + w_G\thinspace\tilde "
        r"h\right) dx = ",
        "derived",
        "kd.flat and kd.htilde integrated over a bin (decays.mojo, _bin_node)",
        checks="tests/unit/test_added_processes.py "
        "(test_decays_energy_and_positivity)",
        file=KD,
    ),
    Entry(
        "kd.weights",
        r"w_g = \tfrac12\mathcal B\thinspace g(E^\ast_n)\thinspace\omega_n",
        "derived",
        "the quadrature of kd.FRL",
        file=KD,
    ),
]
ALSO = {
    "kd.W": (
        r"W_0 = 2\thinspace (p_\nu\negthinspace",
        r"W_{\rm pol}(s) = m_\ell\left[2\thinspace",
    ),
    "kd.cgg": (
        r"A = M(2E_\mu E_\nu - ME'_ \pi) + m^2\left(\tfrac14 E'_ \pi - "
        r"E_\nu\right)",
        r"B = m^2\left(E_\nu - \tfrac12 E'_ \pi\right)",
        r"C = \tfrac14 m^2E'_ \pi",
    ),
    "kd.formfactors": (r"f_0(t) = 1 + \lambda_0\frac{t}{m_{\pi^+}^2}",),
    "kd.dalitz": (
        r"u = \frac{s_3 - s_0}{m_{\pi^+}^2}",
        r"v = \frac{s_2 - s_1}{m_{\pi^+}^2}",
        r"s_i = (p_K - p_i)^2",
        r"s_0 = \tfrac13\textstyle\sum_i s_i",
    ),
    "kd.frame": (
        r"2\thinspace p_a\negthinspace\cdot\negthinspace p_b = "
        r"M^2 - 2ME_c + m_c^2 - m_a^2 - m_b^2",
    ),
    "kd.spin": (
        r"p_a\negthinspace\cdot\negthinspace s = "
        r"\frac{E_a\vert\vec p_\ell\vert - E_\ell\thinspace\vec "
        r"p_a\negthinspace\cdot\negthinspace\hat p_\ell}{m_\ell}",
        r"2\thinspace\vec p_a\negthinspace\cdot\negthinspace\vec "
        r"p_\ell = \vert\vec p_b\vert^2 - \vert\vec p_a\vert^2 - "
        r"\vert\vec p_\ell\vert^2",
    ),
    "kd.line": (
        r"m_{jk}^2 = M^2 + m^2 - 2ME^\ast",
        r"E_j' = \frac{m_{jk}^2 + m_j^2 - m_k^2}{2m_{jk}}",
        r"E_{\max} = \frac{M^2 + m^2 - (m_j + m_k)^2}{2M}",
    ),
    "kd.g": (
        r"G(E^\ast) = \frac{1}{Z}\int_{E_j^{\rm lo}}^{E_j^{\rm hi}} "
        r"\rho\thinspace\mathcal P\thinspace dE_j",
        r"Z = \iint\rho\thinspace dE_i\thinspace dE_j",
    ),
    "kd.F": (r"E_{\rm lo}(x) = \frac{M^2x^2 + m^2}{2Mx}\ \ (\ge m)",),
    "kd.lipari": (r"F_{\mu_L^+}(x) = \frac{x - r}{(1 - r)^2\thinspace x}",),
    "kd.persource": (r"T(x) = x\thinspace F(x)",),
    "kd.grid": (r"x_k = e^{-k\Delta} = \frac{E_i}{E_j}",),
    "kd.table": (
        r"n_q = \int_{x_{q+1}}^{x_q} F\thinspace dx",
        r"e_q = \int_{x_{q+1}}^{x_q} xF\thinspace dx",
    ),
    "kd.node": (
        r"\int_{x_a}^{x_b} \frac{M}{2p^\ast}\left(w_g + "
        r"w_G\thinspace\tilde h\right) x\thinspace dx = ",
    ),
    "kd.weights": (
        r"w_G = \pm\tfrac12\mathcal B\thinspace "
        r"G(E^\ast_n)\thinspace\omega_n",
    ),
}

# Es, ps: E*, p* in the kaon frame
M, m, x, Es, ps, q = sp.symbols("M m x E_s p_s q", positive=True)
c = sp.Symbol("c", real=True)  # cos(theta*) in the kaon frame
g_, G_ = sp.symbols("g G", real=True)
# the kaon's gamma and gamma beta, q = e^rapidity -> oo
GAM, GB = (q + 1 / q) / 2, (q - 1 / q) / 2


def boost(a):
    """The kaon moves along z."""
    return (GAM * a[0] + GB * a[3], a[1], a[2], GAM * a[3] + GB * a[0])


def parse(tex, raw, tokens=None, funcs=None, sign=None):
    """The pieces of LaTeX of `raw` are as written on the page, for example
    with \\negthinspace, which the normalization of to_sympy changes."""
    tokens = dict(tokens or {})
    for n, (piece, value) in enumerate(raw.items()):
        if piece not in tex:
            raise LookupError(f"{piece} not in {tex}")
        tex = tex.replace(piece, r" \Upsilon_{%d} " % n)
        tokens[r"\Upsilon_{%d}" % n] = value
    return L.to_sympy(tex, tokens, funcs, sign)


def relations(pin):
    out = {}
    for part in L.parts(L.display(pin)):
        sides = L.sides(part)
        out[sides[0]] = sides[-1] if len(sides) == 2 else sides[1:]
    return out


# --- Chounet, Gaillard & Gaillard (1972), transcribed from the scan ---


def cgg_rho(E_pi, E_mu, M_, mu, m_, fp, fm):
    """Eq. 6.9 (p. 249), rho ∝ A f+^2 + B f+ f- + C f-^2, with (p. 250)
    A = M(2 E_mu E_nu - M E'_pi) + m^2(E'_pi/4 - E_nu), B = m^2(E_nu - E'_pi/2),
    C = m^2 E'_pi/4, E'_pi = E_pi^max - E_pi = (M^2 + mu^2 - m^2)/(2M) - E_pi.
    Eq. 6.8: <pi(q)|V|K(k)> ∝ (k + q) f+(t) + (k - q) f-(t). Eq. 6.7:
    t = (k - q)^2 = M^2 + mu^2 - 2 M E_pi, m^2 < t < (M - mu)^2."""
    E_nu = M_ - E_pi - E_mu
    Ep = (M_**2 + mu**2 - m_**2) / (2 * M_) - E_pi
    A = M_ * (2 * E_mu * E_nu - M_ * Ep) + m_**2 * (Ep / 4 - E_nu)
    B = m_**2 * (E_nu - Ep / 2)
    C = m_**2 * Ep / 4
    return A * fp**2 + B * fp * fm + C * fm**2


# --- PDG 2026 listings, transcribed from the PDFs ---
# K± listing p. 37, K±_l3 form factors (the K0_L listing p. 23 has the same with
# m_K0, m_pi+):
#   f0 = f+ + f- t/(m_K+^2 - m_pi0^2). "lambda+ and lambda0 are the linear
#   expansion coefficients of f+ and f0": f+(t) = f+(0)(1 + lambda+ t/m_pi+^2)
#   (printed), and likewise f0(t) = f0(0)(1 + lambda0 t/m_pi+^2) (not
#   printed). The quadratic expansion (as used by KTeV):
#   f+(t) = f+(0)(1 + lambda'+ t/m_pi+^2 + (lambda''+/2) t^2/m_pi+^4).
# K± listing p. 32, note "Dalitz plot parameters for K -> 3pi decays", eq. 1:
#   |M|^2 ∝ 1 + g (s3 - s0)/m_pi+^2 + h [(s3 - s0)/m_pi+^2]^2
#           + j (s2 - s1)/m_pi+^2 + k [(s2 - s1)/m_pi+^2]^2
#           + f (s2 - s1)/m_pi+^2 (s3 - s0)/m_pi+^2 + ...,
#   s_i = (P_K - P_i)^2 = (m_K - m_i)^2 - 2 m_K T_i (T_i the kinetic energy of
#   pion i),
#   s0 = (1/3) sum_i s_i = (1/3)(m_K^2 + m_1^2 + m_2^2 + m_3^2). Index 3 is the
#   odd pion. The listings' headings (K± p. 33 and K0_L p. 20:
#   1 + gu + hu^2 + jv + kv^2 + fuv) write u = (s3 - s0)/m_pi^2,
#   v = (s2 - s1)/m_pi^2.
# K0_L listing pp. 34-35:
#   A_L(mu) = [Gamma(pi- mu+ nu) - Gamma(pi+ mu- nu-bar)]/SUM, A_L(e) likewise.


def pdg_dalitz(s, mpi, g, h, j, k, f):
    """Eq. 1 of the note (p. 32), to second order."""
    s1, s2, s3 = s
    s0 = (s1 + s2 + s3) / 3
    U, V = (s3 - s0) / mpi**2, (s2 - s1) / mpi**2
    return 1 + g * U + h * U**2 + j * V + k * V**2 + f * V * U


# --- Lipari, Lusignoli & Meloni (2007), the TeX source (arXiv:0704.0718v1),
#     copied ---
LIP_13 = (
    r"\frac{dn_{a \to b}}{dE} (E_b; E_a) = \frac{1}{E_a} ~ F_{a \to b} "
    r"\left  ( \frac{E_b}{E_a}\right )"
)
LIP_15 = r"F_{\pi^+ \to \mu^+} (x) = {1 \over 1 -r_\pi } ~ \Theta (x-r_\pi)"
LIP_16 = (
    r"h_{\pi^+ \to \mu^+}(x) = \frac{2\,r_\pi}{\left( 1 - r_\pi \right) \,x} "
    r"-\frac{1 + r_\pi}{1 - r_\pi}"
)
LIP_18 = (
    r"F_{\pi^+ \to \mu^+_R} (x) = F_{\pi^- \to \mu^-_L} (x) = "
    r"\frac{r_\pi (1-x)}{(1-r_\pi)^2 \, x} ~ \Theta (x-r_\pi)"
)
# mu^+_R for mu^-_R: the misprint the page notes
LIP_19 = (
    r"F_{\pi^+ \to \mu^+_L} (x) = F_{\pi^- \to \mu^+_R} (x) = "
    r"\frac{x - r_\pi}{(1-r_\pi)^2 \, x} ~ \Theta (x-r_\pi)"
)
LIP_TEXT = (
    r"$r_\pi = (m_\mu/m_\pi)^2$",
    r"is also valid for the decay mode $K^+ \to \mu^+ \nu_\mu$ (and "
    r"charge conjugate), with the simple replacement $m_\pi \to m_K$",
)


def lipari(tex):
    r = sp.Symbol("r", positive=True)
    return L.to_sympy(tex.replace("~", " "), {r"r_\pi": r, "x": x}), r


PRODUCT = {r"E^\ast": Es, r"p^\ast": ps, "M": M, "m": m, "x": x}
TABLE = "| Kaon | Mode | Branching ratio |"
KAONS = {
    r"$\mathrm K^+$": "K+",
    r"$\mathrm K^0_\mathrm S$": "K0S",
    r"$\mathrm K^0_\mathrm L$": "K0L",
}


def readme_range():
    tex = L.after("is spread uniformly over $")
    assert tex.startswith(r"x \in [") and tex.endswith("]"), tex
    return [
        L.to_sympy(e, PRODUCT)
        for e in L.split_top(tex[len(r"x \in [") : -1], ",")
    ]


def readme_density():
    return L.to_sympy(L.after("/M]$ with density $"), PRODUCT)


def readme_htilde(E=Es, p=ps):
    return L.to_sympy(
        L.after(r"with $\tilde h = "), {**PRODUCT, r"E^\ast": E, r"p^\ast": p}
    )


F_PIN = r"F(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}"


def readme_Elo():
    p = L.sides(L.part(L.display(F_PIN), r"E_{\rm lo}(x)"))
    assert p[0] == r"E_{\rm lo}(x)" and p[1].endswith(r"\ \ (\ge m)"), p
    return L.to_sympy(p[1][: -len(r"\ \ (\ge m)")], {"M": M, "m": m, "x": x})


def readme_F(sign=None, g=g_, G=G_, ht=None, E=Es, p=ps):
    pin, lhs = (
        (F_PIN, "F(x)")
        if sign is None
        else (r"F_{R,L}(x) = \frac{M}{2}\int", "F_{R,L}(x)")
    )
    part = L.parts(L.display(pin))[0]
    left, right = L.sides(part)
    pre, integrand = right.split(r"\int_{E_{\rm lo}(x)}^{E_{\max}} ")
    assert left == lhs and integrand.endswith(r"\thinspace dE^\ast"), part
    funcs = {"g": lambda a: g, "G": lambda a: G, r"\tilde h": lambda a, b: ht}
    tokens = {r"E^\ast": E, "x": x, r"p^\ast": p}
    return L.to_sympy(pre, {"M": M}) * L.to_sympy(
        integrand[: -len(r"\thinspace dE^\ast")], tokens, funcs, sign
    )


def readme_two_body(mb=0):
    """E* of the page's two-body decay (partner mass mb) and p* for a massless
    partner."""
    E0 = L.to_sympy(
        L.sides(L.after(r"Two-body decays have one $"))[1],
        {"M": M, "m": m, "m_b": mb},
    )
    return E0, (M**2 - m**2) / (2 * M)


def readme_P():
    return L.to_sympy(L.sides(L.after(r"For $\mathrm K \to \mu\nu$, $"))[1], {})


DOT = re.compile(r"\(([^()]+?)\\negthinspace \\cdot\\negthinspace ([^()]+?)\)")


def readme_W(vectors, dot, ml):
    line = L.display(r"\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s)")
    assert (
        L.parts(line)[0]
        == r"\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s)"
    ), line
    out = []
    for lhs in ("W_0 =", r"W_{\rm pol}(s) =")[: 2 if "s" in vectors else 1]:
        tokens = {r"m_\ell": ml, "H^2": dot(vectors["H"], vectors["H"])}

        def value(match):
            name = r"\Xi_{%d}" % len(tokens)
            tokens[name] = dot(vectors[match.group(1)], vectors[match.group(2)])
            return " " + name + " "

        out.append(
            L.to_sympy(DOT.sub(value, L.sides(L.part(line, lhs))[1]), tokens)
        )
    return out


def readme_H(fp, fm, pK, ppi):
    tex = L.after(r"with the hadronic current $")
    return L.to_sympy(
        L.sides(tex)[1],
        {"p_K": pK, r"p_\pi": ppi, "t": sp.Symbol("t")},
        funcs={"f_+": lambda a: fp, "f_-": lambda a: fm},
    )


def DALITZ():
    return L.display(
        r"\vert \mathcal M\vert^2 \propto 1 + g\thinspace u + "
        r"h\thinspace u^2 + k\thinspace v^2"
    )


def readme_uv(s, mpi):
    line = DALITZ()
    assert L.part(line, "s_i =") == "s_i = (p_K - p_i)^2", line
    third, rest = L.sides(L.part(line, "s_0 ="))[1].split(r"\textstyle")
    assert rest == r"\sum_i s_i", rest
    tokens = {
        "s_1": s[0],
        "s_2": s[1],
        "s_3": s[2],
        "s_0": L.to_sympy(third, {}) * sum(s),
        r"m_{\pi^+}": mpi,
    }
    return [
        L.to_sympy(L.sides(L.part(line, n + " ="))[1], tokens) for n in "uv"
    ]


def readme_dalitz(s, mpi, g, h, k):
    u, v = readme_uv(s, mpi)
    return L.to_sympy(
        L.parts(DALITZ())[0].split(r"\propto")[1],
        {"g": g, "h": h, "k": k, "u": u, "v": v},
    )


def laurent_integral(expr, var, lo, hi):
    total = 0
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff, n = term.as_coeff_exponent(var)
        assert not coeff.has(var), term
        total += coeff * (
            sp.log(hi / lo)
            if n == -1
            else (hi ** (n + 1) - lo ** (n + 1)) / (n + 1)
        )
    return total


FRAME_PIN = r"p_K\negthinspace\cdot\negthinspace p_a = ME_a"
SPIN_PIN = r"s = \frac{1}{m_\ell}\left(\vert\vec p_\ell\vert,"


def readme_frame():
    rel = relations(FRAME_PIN)
    Ea, Ec, ma, mb, mc = sp.symbols("E_a E_c m_a m_b m_c", real=True)
    one = L.to_sympy(
        rel[r"p_K\negthinspace\cdot\negthinspace p_a"], {"M": M, "E_a": Ea}
    )
    two = (
        L.to_sympy(
            rel[r"2\thinspace p_a\negthinspace\cdot\negthinspace p_b"],
            {"M": M, "E_c": Ec, "m_a": ma, "m_b": mb, "m_c": mc},
        )
        / 2
    )
    return (
        (lambda E: one.subs(Ea, E)),
        (lambda E, a, b, c_: two.subs({Ec: E, ma: a, mb: b, mc: c_})),
    )


def readme_spin():
    rel = relations(SPIN_PIN)
    P, El, ml, Ea, D, A2, B2, P2 = sp.symbols(
        "P E_l m_l E_a D A2 B2 P2", real=True
    )
    pm = re.fullmatch(
        r"(\\frac\{1\}\{m_\\ell\})\\left\((.*),\\thinspace "
        r"(.*)\\right\)",
        rel["s"],
    )
    pre = L.to_sympy(pm.group(1), {r"m_\ell": ml})
    s0 = pre * parse(pm.group(2), {r"\vert\vec p_\ell\vert": P})
    s3 = pre * parse(pm.group(3), {r"\hat p_\ell": 1}, {r"E_\ell": El})
    dot = parse(
        rel[r"p_a\negthinspace\cdot\negthinspace s"],
        {
            r"\vec p_a\negthinspace\cdot\negthinspace\hat p_\ell": D,
            r"\vert\vec p_\ell\vert": P,
        },
        {"E_a": Ea, r"E_\ell": El, r"m_\ell": ml},
    )
    vec = (
        parse(
            rel[
                r"2\thinspace\vec p_a\negthinspace\cdot\negthinspace\vec p_\ell"
            ],
            {
                r"\vert\vec p_b\vert": sp.sqrt(B2),
                r"\vert\vec p_a\vert": sp.sqrt(A2),
                r"\vert\vec p_\ell\vert": sp.sqrt(P2),
            },
        )
        / 2
    )
    sym = (P, El, ml, Ea, D, A2, B2, P2)
    return (s0, s3), dot, vec, sym


def l3_dot(Epi, El, mpi, ml, P=None):
    kdot, pair = readme_frame()
    E = {"pi": Epi, "l": El, "nu": M - Epi - El}
    mass = {"pi": mpi, "l": ml, "nu": 0}
    gram = {("K", "K"): M**2}
    for a in E:
        gram[("K", a)] = kdot(E[a])
        gram[(a, a)] = mass[a] ** 2
    for a, b in (("pi", "l"), ("pi", "nu"), ("l", "nu")):
        c_ = ({"pi", "l", "nu"} - {a, b}).pop()
        gram[(a, b)] = pair(E[c_], mass[a], mass[b], mass[c_])
    if P is not None:
        _, sdot, vdot, (Ps, Els, mls, Ea, D, A2, B2, P2) = readme_spin()
        mom2 = {a: E[a] ** 2 - mass[a] ** 2 for a in E}
        for a in ("K", "pi", "l", "nu"):
            if a == "K":
                d = 0  # p_K has no 3-momentum
            elif a == "l":
                d = P  # p_l . p_l-hat
            else:
                b = "nu" if a == "pi" else "pi"
                d = vdot.subs({A2: mom2[a], B2: mom2[b], P2: P**2}) / P
            gram[("s", a)] = sdot.subs(
                {Ea: M if a == "K" else E[a], D: d, Ps: P, Els: El, mls: ml}
            )

    def dot(u, v):
        return sum(
            cu * cv * gram.get((a, b), gram.get((b, a)))
            for a, cu in u.items()
            for b, cv in v.items()
        )

    return dot


def l3_vectors(fp, fm):
    PK, PPI = sp.symbols("P_K P_pi")
    H = sp.expand(readme_H(fp, fm, PK, PPI))
    return {
        r"p_\nu": {"nu": 1},
        r"p_\ell": {"l": 1},
        "H": {"K": H.coeff(PK), "pi": H.coeff(PPI)},
        "s": {"s": 1},
    }


def readme_l3(fp, fm, Epi, El, mpi, ml, P=None):
    vec = l3_vectors(fp, fm)
    if P is None:
        del vec["s"]
    return readme_W(vec, l3_dot(Epi, El, mpi, ml, P), ml)


ML = sp.Symbol("m_l", positive=True)
EPI, EL, MPI, FP, FM = sp.symbols("E_pi E_l m_pi f_p f_m", real=True)
PK, PPI = sp.symbols("P_K P_pi")  # k and q of CGG


def kaon_frame():
    gdot = l3_dot(EPI, EL, MPI, ML)
    Hx = sp.expand(readme_H(FP, FM, PK, PPI))
    t_tex = L.sides(L.inline(r"t = (p_K - p_\pi)^2"))[1]
    assert t_tex.endswith("^2"), t_tex
    t_vec = sp.expand(L.to_sympy(t_tex[:-2], {"p_K": PK, r"p_\pi": PPI}))
    return (
        gdot,
        Hx,
        {"K": Hx.coeff(PK), "pi": Hx.coeff(PPI)},
        {"K": t_vec.coeff(PK), "pi": t_vec.coeff(PPI)},
    )


PARTICLE = re.compile(
    r"(\d?)(\\pi|\\mu|\\nu|\\mathrm e)(\^(?:\+|-|0|\\pm|\\mp))?(_(?:\\mu|e))?"
)
CHARGE = {
    "^+": (1, 1),
    "^-": (-1, -1),
    "^0": (0, 0),
    r"^\pm": (1, -1),
    r"^\mp": (-1, 1),
    "": (0, 0),
}


def number(cell):
    return sp.Rational(
        cell.strip("$").replace(r" \times 10^{", "e").replace("}", "")
    )


def readme_modes():
    out, kaon = {}, None
    for cells in L.table(TABLE):
        kaon = KAONS[cells[0]] if cells[0] else kaon
        tex = re.match(r"\$([^$]+)\$", cells[1]).group(1)
        parts, pos, lepton = [], 0, None
        for p in PARTICLE.finditer(tex):
            assert p.start() == pos, (tex, pos)
            pos = p.end()
            charge = (
                (0, 0) if p.group(2) == r"\nu" else CHARGE[p.group(3) or ""]
            )
            parts.append((int(p.group(1) or 1), *charge))
            if charge[0] != charge[1] and p.group(2) != r"\pi":
                lepton = "e" if p.group(2) == r"\mathrm e" else "mu"
        assert pos == len(tex), tex
        out.setdefault(kaon, []).append((number(cells[2]), lepton, parts))
    return out


def readme_AL():
    m_ = re.search(
        r"Its averages are \$A_L = (.+?)\$ \(\$\\mathrm e\$, p\. "
        r"\d+\) and \$(.+?)\$ \(\$\\mu\$, p\. \d+\)",
        L.line("Its averages are $A_L = "),
    )
    return {"e": number(m_.group(1)), "mu": number(m_.group(2))}


def branching_ratios():
    out, kaon = {}, None
    for cells in L.table(TABLE):
        kaon = KAONS[cells[0]] if cells[0] else kaon
        value = (
            cells[2].strip("$").replace(r" \times 10^{", "e").replace("}", "")
        )
        out.setdefault(kaon, {})[cells[1]] = float(value)
    return out


MODES_PIN = r"F_{s\to X}(x) = \frac{\sum_m B_m\thinspace \nu_{mX}"


def readme_per_decay(kaon, per_mode):
    num, den = re.fullmatch(
        r"\\frac\{(.*)\}\{(.*)\}", relations(MODES_PIN)[r"F_{s\to X}(x)"]
    ).groups()
    assert num == r"\sum_m B_m\thinspace \nu_{mX}\thinspace F_{mX}(x)", num
    shares, AL = readme_asym()[0], readme_AL()
    total_num, total_den = 0, 0
    for B, lepton, parts in readme_modes()[kaon]:
        if lepton is None:
            states = [(1, [(n, up) for n, up, _ in parts])]
        # the page's probabilities: (1 +- A_L)/2 for K0_L, split evenly for K0_S
        else:
            A = AL[lepton] if kaon == "K0L" else 0
            states = [
                (
                    shares[0].subs(sp.Symbol("A_L"), A),
                    [(n, up) for n, up, _ in parts],
                ),
                (
                    shares[1].subs(sp.Symbol("A_L"), A),
                    [(n, low) for n, _, low in parts],
                ),
            ]
        total_num += B * per_mode(states)
        total_den += (
            L.to_sympy(den.replace(r"\sum_m B_m", r"B_m"), {"B_m": B})
            if r"\sum_m" in den
            else 0
        )
    if r"\sum_m" not in den:
        total_den = L.to_sympy(den, {})
    return total_num / total_den


@derivation("kd.modes")
def check_modes():
    """The modes of the table conserve charge, and, renormalized to their sum
    (F_s->X), every decay carries the kaon's charge and energy. The table's sums
    are the page's coverages."""
    charge = {"K+": 1, "K0S": 0, "K0L": 0}
    modes = readme_modes()
    conserve = all(
        sum(n * up for n, up, _ in parts) == charge[k]
        and sum(n * low for n, _, low in parts) == charge[k]
        for k, rows in modes.items()
        for _, _, parts in rows
    )
    q_per_decay = {
        k: readme_per_decay(
            k, lambda st: sum(p * sum(n * ch for n, ch in ps_) for p, ps_ in st)
        )
        for k in modes
    }
    # each mode's products carry the kaon's energy (kd.F: the mean x of a
    # product is E*/M, and the E* add up to M)
    energy = {
        k: readme_per_decay(k, lambda st: sum(p for p, _ in st)) for k in modes
    }
    sums = {k: sum(B for B, _, _ in rows) for k, rows in modes.items()}
    cover = re.search(
        r"cover ([\d.]+)% of \$\\mathrm K\^\\pm\$ decays, all of "
        r"\$\\mathrm K\^0_\\mathrm S\$ and ([\d.]+)% of "
        r"\$\\mathrm K\^0_\\mathrm L\$\.",
        L.line("Together they cover "),
    ).groups()
    count = lambda k, B, q_: sum(
        n
        for n, up, _ in dict((b, p_) for b, _, p_ in modes[k])[sp.Rational(B)]
        if up == q_
    )
    with L.patched(r"F_{mX}(x)}{\sum_m B_m}", r"F_{mX}(x)}{1}"):
        unnormalised = readme_per_decay("K0L", lambda st: sum(p for p, _ in st))
    return [
        Result(
            "every mode of the table conserves charge "
            "(both charge states of pi-+ l+- nu)",
            conserve,
            f"{sum(len(r) for r in modes.values())} modes",
        ),
        exact(
            "per decay, the charge of K+, K0_S, "
            "K0_L (F_s->X with the page's probabilities)",
            sum((q_per_decay[k] - charge[k]) ** 2 for k in modes),
        ),
        exact(
            "per decay, the products carry the kaon's energy "
            "(each mode's do; F_s->X divides by the sum)",
            sum((energy[k] - 1) ** 2 for k in modes),
        ),
        Result(
            "the coverages: K+- 99.995 %, K0_S all, K0_L 99.93 % "
            "(the table's sums, rounded as written)",
            abs(sp.Rational(cover[0]) - 100 * sums["K+"])
            < sp.Rational(5, 10**4)
            and sums["K0S"] >= 1
            and abs(sp.Rational(cover[1]) - 100 * sums["K0L"])
            < sp.Rational(5, 10**3),
            ", ".join(f"{k} {float(v):.6f}" for k, v in sums.items()),
        ),
        Result(
            "nu_mX: 2 pi+ in K+ -> pi+ pi+ pi-, 3 pi0 in K0_L -> 3 pi0 "
            "(the table's modes, as the page says)",
            count("K+", "0.05583", 1) == 2
            and count("K0L", "0.1952", 0) == 3
            and "2 for the $\\pi^+$ of $\\pi^+\\pi^+\\pi^-$ and 3 "
            "for the $\\pi^0$ of $3\\pi^0$"
            in L.line(r"Here $\nu_{mX}$ is the number of $X$"),
            "as written",
        ),
        exact_fails(
            "control: without the sum in the denominator, a K0_L "
            "decay carries 0.99933 of its energy",
            unnormalised - 1,
        ),
    ]


def readme_asym():
    line = L.line(
        r"semileptonic decays give $\pi^-\ell^+\nu$ with probability $"
    )
    AL = sp.Symbol("A_L")
    shares = re.search(
        r"give \$\\pi\^-\\ell\^\+\\nu\$ with probability \$(.+?)\$ and "
        r"\$\\pi\^\+\\ell\^-\\bar\\nu\$ with \$(.+?)\$\.",
        line,
    ).groups()
    return [L.to_sympy(t, {"A_{L}": AL}) for t in shares], AL


@derivation("kd.asym")
def check_asym():
    """PDG's A_L = [Gamma(pi- l+ nu) - Gamma(pi+ l- nu-bar)]/SUM gives the
    shares (1 +- A_L)/2."""
    G1, G2 = sp.symbols("Gamma_1 Gamma_2", positive=True)
    (plus, minus), AL = readme_asym()
    pdg = (G1 - G2) / (G1 + G2)  # K0_L listing pp. 34-35
    with L.patched(
        r"$\left(1 + A_{L}\right)/2$", r"$\left(1 + A_{L}\right)/4$"
    ):
        broken = readme_asym()[0][0]
    return [
        exact(
            "Gamma(pi- l+ nu)/SUM = the page's (1 + A_L)/2",
            plus.subs(AL, pdg) - G1 / (G1 + G2),
        ),
        exact(
            "Gamma(pi+ l- nu-bar)/SUM = the page's (1 - A_L)/2",
            minus.subs(AL, pdg) - G2 / (G1 + G2),
        ),
        Result(
            "the page states PDG's definition: the difference of the two "
            "rates over their sum",
            "is the difference of the two rates over their sum"
            in L.line("PDG's charge asymmetry $A_{L}$"),
            "as written",
        ),
        exact_fails(
            "control: (1 + A_L)/4", broken.subs(AL, pdg) - G1 / (G1 + G2)
        ),
    ]


@derivation("kd.W")
def check_W():
    """W_0 +- W_pol(s) is the V-A trace for an l+ of spin s (s along the
    momentum: helicity +1/2). K -> mu nu fixes the sign."""
    _, g5, one = dirac()
    H, n, l, s = (
        sp.symbols(f"{v}0:4", real=True) for v in ("H", "n", "l", "s")
    )

    def trace(spin):
        rho_l = (slash(l) - ML * one) * (one + g5 * slash(spin)) / 2
        return sp.expand(
            (
                slash(n) * slash(H) * (one - g5) * rho_l * slash(H) * (one - g5)
            ).trace()
        )

    T, Tm = trace(s), trace(tuple(-a for a in s))
    vecs = {r"p_\nu": n, r"p_\ell": l, "H": H, "s": s}
    W0, Wp = readme_W(vecs, mdot, ML)
    with L.patched(r"m_\ell\left[2\thinspace", r"m_\ell\left[4\thinspace"):
        Wp_broken = readme_W(vecs, mdot, ML)[1]
    L.line(
        r"$\mathrm K^-$ decays are the CP conjugates of $\mathrm K^+$ decays "
        r"(a left-handed $\mu^+$ becomes a right-handed $\mu^-$)"
    )
    rho_minus = (slash(l) + ML * one) * (one + g5 * slash(s)) / 2
    T_minus = sp.expand(
        (
            rho_minus * slash(H) * (one - g5) * slash(n) * (one + g5) * slash(H)
        ).trace()
    )
    out = [
        exact(
            "trace = 4 [W_0 + W_pol(s)] (any H, p_nu, p_l, s)",
            T - 4 * (W0 + Wp),
        ),
        exact("summed over s, -s: 4 x 2 W_0", T + Tm - 8 * W0),
        exact(
            "an l- (CP conjugate): 4 [W_0 - W_pol(s)], its helicities "
            "those of the l+ exchanged",
            T_minus - 4 * (W0 - Wp),
        ),
        exact_fails("control: 4 for 2 in W_pol", T - 4 * (W0 + Wp_broken)),
    ]
    # The spin-0 kaon and the left-handed nu make the mu+ left-handed, so s
    # along the momentum (helicity +1/2) must give no rate.
    E0, p0 = readme_two_body()
    pK, pmu, pnu, s_up = (
        (M, 0, 0, 0),
        (E0, 0, 0, p0),
        (p0, 0, 0, -p0),
        (p0 / m, 0, 0, E0 / m),
    )
    W0_2, Wp_2 = readme_W(
        {r"p_\nu": pnu, r"p_\ell": pmu, "H": pK, "s": s_up}, mdot, m
    )
    out += [
        exact(
            "K -> mu nu: p_K = p_mu + p_nu, s.p_mu = 0, s^2 = -1",
            sum((a - b - d) ** 2 for a, b, d in zip(pK, pmu, pnu))
            + mdot(s_up, pmu) ** 2
            + (mdot(s_up, s_up) + 1) ** 2,
        ),
        exact(
            "K -> mu nu: W_0 + W_pol(s) = 0 for helicity +1/2 "
            "(the mu+ is left-handed)",
            W0_2 + Wp_2,
        ),
        exact(
            "K -> mu nu: P = W_pol(s)/W_0 = the page's P = -1",
            Wp_2 / W0_2 - readme_P(),
        ),
    ]
    return out


@derivation("kd.current")
def check_current():
    """H is CGG's eq. 6.8, and t = (p_K - p_pi)^2 = (p_l + p_nu)^2. The f_- term
    gives m_l times the scalar lepton current."""
    gdot, Hx, _, tv = kaon_frame()
    # The spin sum of |u_nu-bar X v_l|^2, X = q-slash (1 - g5) + m_l (1 + g5),
    # vanishes on shell, so each amplitude does.
    gam, g5, one = dirac()
    n, l = (sp.symbols(f"{v}1:4", real=True) for v in ("n", "l"))
    pn = (sp.sqrt(sum(a**2 for a in n)),) + tuple(n)
    pl = (sp.sqrt(sum(a**2 for a in l) + ML**2),) + tuple(l)
    spins = lambda X_: sp.expand(
        (
            slash(pn) * X_ * (slash(pl) - ML * one) * gam[0] * X_.H * gam[0]
        ).trace()
    )
    q_ = slash([a + b for a, b in zip(pn, pl)])
    total, wrong = (
        spins(q_ * (one - g5) + ML * (one + g5)),
        spins(q_ * (one - g5) - ML * (one + g5)),
    )
    with L.patched(
        r"f_-(t)(p_K - p_\pi)$, where", r"f_-(t)(p_K + p_\pi)$, where"
    ):
        broken = sp.expand(readme_H(FP, FM, PK, PPI))
    return [
        exact(
            "H = (k + q) f+ + (k - q) f- (CGG eq. 6.8, k = p_K, q = p_pi)",
            Hx - ((PK + PPI) * FP + (PK - PPI) * FM),
        ),
        exact(
            "t = (p_K - p_pi)^2 = CGG's (k - q)^2 = M^2 + mu^2 - 2 M "
            "E_pi (eq. 6.7)",
            gdot(tv, tv) - (M**2 + MPI**2 - 2 * M * EPI),
        ),
        exact(
            "t = (p_l + p_nu)^2 (p_K = p_pi + p_l + p_nu)",
            gdot(tv, tv) - gdot({"l": 1, "nu": 1}, {"l": 1, "nu": 1}),
        ),
        exact(
            "sum over spins |u_nu-bar [(p_l + p_nu)-slash (1 - g5) + m_l "
            "(1 + g5)] v_l|^2 = 0 on shell: the f_- term is -m_l f_- "
            "u_nu-bar (1 + g5) v_l, a scalar current that vanishes with "
            "m_l (K_e3)",
            total,
        ),
        exact_fails(
            "control: + m_l u_nu-bar (1 + g5) v_l does not cancel it", wrong
        ),
        exact_fails(
            "control: f_-(t)(p_K + p_pi)",
            broken - ((PK + PPI) * FP + (PK - PPI) * FM),
        ),
    ]


@derivation("kd.spin_sum")
def check_spin_sum():
    """Summed over the lepton's spin (s and -s), the page's W_0 + W_pol(s) is
    its 2 W_0: W_pol is linear in s."""
    H, n, l, s = (
        sp.symbols(f"{v}0:4", real=True) for v in ("H", "n", "l", "s")
    )
    vectors = lambda spin: {r"p_\nu": n, r"p_\ell": l, "H": H, "s": spin}
    W0, Wp = readme_W(vectors(s), mdot, ML)
    Wm = readme_W(vectors(tuple(-a for a in s)), mdot, ML)[1]

    def summed():
        lhs, rhs = L.after(r"and, summed over its spin, $").split(r"\propto")
        assert lhs.strip() == r"\vert\mathcal M\vert^2", lhs
        return L.to_sympy(rhs, {"W_0": W0})

    out = [
        exact("W_pol(-s) = -W_pol(s) (any p_nu, p_l, H, s)", Wp + Wm),
        exact(
            "the page: W_0 + W_pol(s) summed over s and -s = its 2 W_0",
            (W0 + Wp) + (W0 + Wm) - summed(),
        ),
    ]
    with L.patched(
        r"summed over its spin, $\vert\mathcal M\vert^2 \propto 2W_0$",
        r"summed over its spin, $\vert\mathcal M\vert^2 \propto W_0$",
    ):
        out.append(
            exact_fails(
                "control: W_0 for 2 W_0 on the page",
                (W0 + Wp) + (W0 + Wm) - summed(),
            )
        )
    return out


CGG_PIN = r"\rho \propto Af_+^2 + Bf_+f_- + Cf_-^2"


def readme_cgg(E_pi, E_mu, M_, mu, m_, fp, fm):
    rel = relations(CGG_PIN)
    Emax = L.to_sympy(
        L.after(r"with $E_\pi^{\max}$ $= "), {"M": M_, r"\mu": mu, "m": m_}
    )
    Ep = L.to_sympy(
        L.after(r"and $E'_ \pi$ is $"), {r"E_\pi^{\max}": Emax, r"E_\pi": E_pi}
    )
    tokens = {
        "M": M_,
        "m": m_,
        r"E_\mu": E_mu,
        r"E_\nu": M_ - E_pi - E_mu,
        r"E'_ \pi": Ep,
    }
    coeff = {k: L.to_sympy(rel[k], tokens) for k in "ABC"}
    lhs, rhs = L.parts(L.display(CGG_PIN))[0].split(r"\propto")
    assert lhs.strip() == r"\rho", lhs
    return L.to_sympy(rhs, {**coeff, "f_+": fp, "f_-": fm})


@derivation("kd.cgg")
def check_cgg():
    """The page quotes CGG's eq. 6.9 (pp. 249-250) as transcribed from the scan:
    rho, A, B, C, E'_pi and E_pi^max."""
    mu, mm = sp.symbols("mu m", positive=True)
    ours = readme_cgg(EPI, EL, M, mu, mm, FP, FM)
    with L.patched(
        r"B = m^2\left(E_\nu - \tfrac12 E'_ \pi\right)",
        r"B = m^2\left(E_\nu - \tfrac14 E'_ \pi\right)",
    ):
        broken = readme_cgg(EPI, EL, M, mu, mm, FP, FM)
    return [
        exact(
            "the page's rho = A f+^2 + B f+ f- + C f-^2 of eq. 6.9 "
            "(A, B, C, E'_pi of p. 250), any f+, f-",
            ours - cgg_rho(EPI, EL, M, mu, mm, FP, FM),
        ),
        Result(
            "in CGG's notation: kaon mass M, pion mass mu, lepton mass m",
            "in their notation: the kaon mass $M$, the pion mass $\\mu$ and "
            "the lepton mass $m$" in L.line("in their notation:"),
            "as written",
        ),
        exact_fails(
            "control: E'_pi/4 for E'_pi/2 in B",
            broken - cgg_rho(EPI, EL, M, mu, mm, FP, FM),
        ),
    ]


@derivation("kd.cgg_w0")
def check_cgg_w0():
    """W_0 of the page (V-A, kd.W) is 4M times CGG's rho (eq. 6.9) at every
    point of the Dalitz plot."""
    W0 = readme_l3(FP, FM, EPI, EL, MPI, ML)[0]
    L.inline(r"W_0 = 4M\thinspace\rho")  # the pin, once
    ratio = lambda: L.to_sympy(
        L.sides(L.after(r"With these energies, $"))[1], {"M": M, r"\rho": 1}
    )
    rho = readme_cgg(EPI, EL, M, MPI, ML, FP, FM)
    with L.patched(r"$W_0 = 4M\thinspace\rho$", r"$W_0 = 2M\thinspace\rho$"):
        broken = ratio()
    return [
        exact(
            "W_0 = 4M rho in the energies E_pi, E_l (any f+, f-, masses)",
            W0 - ratio() * rho,
        ),
        exact_fails("control: 2M rho", W0 - broken * rho),
    ]


def readme_f0(fp, fm, t, mK, mpi):
    left, right = L.sides(L.after(r"PDG's definition $"))
    assert left == "f_0", left
    return L.to_sympy(
        right, {"f_+": fp, "f_-": fm, "t": t, "m_K": mK, r"m_\pi": mpi}
    )


@derivation("kd.f0")
def check_f0():
    """The page quotes PDG's definition f_0 = f_+ + f_- t/(m_K^2 - m_pi^2) (K±
    listing p. 37, K0_L p. 23)."""
    fp, fm, t, mK, mpi = sp.symbols("f_p f_m t m_K m_pi", real=True)
    # K± p. 37 (m_K+, m_pi0), K0_L p. 23 (m_K0, m_pi+)
    pdg = fp + fm * t / (mK**2 - mpi**2)
    L.inline(r"f_0 = f_+ + f_-\thinspace t/(m_K^2 - m_\pi^2)")  # the pin, once
    with L.patched(
        r"$f_0 = f_+ + f_-\thinspace t/", r"$f_0 = f_+ - f_-\thinspace t/"
    ):
        broken = readme_f0(fp, fm, t, mK, mpi)
    return [
        exact(
            "the page's f_0 = PDG's (any f_+, f_-, t, masses)",
            readme_f0(fp, fm, t, mK, mpi) - pdg,
        ),
        Result(
            "the masses of the mode: M for m_K",
            "with $M$ for the kaon mass $m_K$"
            in L.line("PDG's definition $f_0"),
            "as written",
        ),
        exact_fails("control: f_+ - f_- t/(...)", broken - pdg),
    ]


@derivation("kd.fminus")
def check_fminus():
    """f_- = (f_0 - f_+)(M^2 - m_pi^2)/t is the page's quotation of PDG's
    definition of f_0 solved for f_-."""
    f0, fp, fm, t, mpi = sp.symbols("f_0 f_p f_m t m_pi", real=True)
    tokens = {"f_0": f0, "f_+": fp, "M": M, r"m_\pi": mpi, "t": t}
    pdg = sp.solve(sp.Eq(f0, readme_f0(fp, fm, t, M, mpi)), fm)
    ours = lambda: L.to_sympy(L.after(r"fits, with $f_- = "), tokens)
    with L.patched(r"(M^2 - m_\pi^2)/t", r"(M^2 + m_\pi^2)/t"):
        broken = ours()
    return [
        exact(
            "the page's f_- = its f_0 definition solved for f_- "
            "(m_K = M, m_pi of the mode)",
            ours() - pdg[0],
        ),
        exact_fails("control: M^2 + m_pi^2", broken - pdg[0]),
    ]


FF_PIN = r"f_+(t) = 1 + \lambda'_ +\frac{t}{m_{\pi^+}^2}"
T_, MPIP, L1, L2, L0 = sp.symbols("t m_pip lambda1 lambda2 lambda0", real=True)
FF_TOKENS = {
    r"\lambda'_ +": L1,
    r"\lambda''_ +": L2,
    r"\lambda_0": L0,
    "t": T_,
    r"m_{\pi^+}": MPIP,
}


def readme_fplus():
    return L.to_sympy(
        L.sides(L.part(L.display(FF_PIN), "f_+(t)"))[1], FF_TOKENS
    )


def readme_fzero():
    return L.to_sympy(
        L.sides(L.part(L.display(FF_PIN), "f_0(t)"))[1], FF_TOKENS
    )


@derivation("kd.formfactors")
def check_formfactors():
    """f_+ and f_0 are PDG's expansions (quadratic, or linear for
    lambda''_+ = 0) normalized to f_+(0) = f_0(0) = 1."""
    f0 = sp.Symbol("f0", real=True)
    # K± p. 37, K0_L p. 23 (f0 = f+(0))
    pdg_quad = f0 * (1 + L1 * T_ / MPIP**2 + L2 / 2 * T_**2 / MPIP**4)
    pdg_lin = f0 * (1 + L1 * T_ / MPIP**2)
    pdg_f0 = f0 * (1 + L0 * T_ / MPIP**2)
    with L.patched(r"\frac{\lambda''_ +}{2}\left(", r"\lambda''_ +\left("):
        broken = readme_fplus()
    return [
        exact(
            "the page's f_+ = PDG's quadratic f_+(t)/f_+(0)",
            readme_fplus() - pdg_quad / f0,
        ),
        exact(
            "with lambda''_+ = 0: PDG's linear f_+(t)/f_+(0), lambda_+ = "
            "lambda'_+ (the K± fit)",
            readme_fplus().subs(L2, 0) - pdg_lin / f0,
        ),
        exact(
            "the page's f_0 = PDG's linear f_0(t)/f_0(0), f_0(0) = "
            "f_+(0) (f_0 = f_+ + f_- t/(m_K^2 - m_pi^2))",
            readme_fzero() - pdg_f0 / f0,
        ),
        exact_fails(
            "control: lambda''_+ for lambda''_+/2", broken - pdg_quad / f0
        ),
    ]


FMC_PIN = r"f_-(t) = (M^2 - m_\pi^2)\left["


def readme_fminus_code(mpi=MPI):
    """The page's f_- in the expansion coefficients (the form the code
    evaluates)."""
    left, right = L.sides(L.display(FMC_PIN))
    assert left == "f_-(t)", left
    return L.to_sympy(right.rstrip("."), {**FF_TOKENS, "M": M, r"m_\pi": mpi})


@derivation("kd.fminus_code")
def check_fminus_code():
    """With the expansions of f_+ and f_0, f_- = (f_0 - f_+)(M^2 - m_pi^2)/t is
    the page's form, finite at t = 0."""
    fm = L.to_sympy(
        L.after(r"fits, with $f_- = "),
        {
            "f_0": readme_fzero(),
            "f_+": readme_fplus(),
            "M": M,
            r"m_\pi": MPI,
            "t": T_,
        },
    )
    ours = readme_fminus_code()
    with L.patched(
        r"\frac{\lambda_0 - \lambda'_ +}{m_{\pi^+}^2}",
        r"\frac{\lambda_0 + \lambda'_ +}{m_{\pi^+}^2}",
    ):
        broken = readme_fminus_code()
    return [
        exact(
            "(f_0 - f_+)(M^2 - m_pi^2)/t with the page's expansions = "
            "the page's f_-(t)",
            fm - ours,
        ),
        Result(
            "finite at t = 0 (no 1/t)",
            not sp.denom(sp.together(ours)).has(T_),
            f"f_-(0) = {ours.subs(T_, 0)}",
        ),
        exact_fails("control: lambda_0 + lambda'_+", fm - broken),
    ]


@derivation("kd.dalitz")
def check_dalitz():
    """The K -> 3pi expansion is PDG's eq. 1 (K± listing, p. 32) without its
    CP-violating terms j v and f u v."""
    s = sp.symbols("s_1:4", real=True)
    mpi, mK, g, h, k = sp.symbols("m_pi m_K g h k", real=True)
    E, mi = sp.symbols("E_1:4", real=True), sp.symbols("m_1:4", positive=True)
    u, v = readme_uv(s, mpi)
    ours = readme_dalitz(s, mpi, g, h, k)
    s0 = sum(s) / 3
    s_kin = [mK**2 + mi[i] ** 2 - 2 * mK * E[i] for i in range(3)]
    on_shell = {E[2]: mK - E[0] - E[1]}
    with L.patched(r"u = \frac{s_3 - s_0}", r"u = \frac{s_1 - s_0}"):
        broken = readme_uv(s, mpi)[0]
    return [
        exact(
            "the page's |M|^2 = eq. 1 with j = f = 0",
            ours - pdg_dalitz(s, mpi, g, h, 0, k, 0),
        ),
        exact(
            "u = (s_3 - s_0)/m_pi+^2 and v = (s_2 - s_1)/m_pi+^2 "
            "(eq. 1; the sign of v matters for j, f)",
            (u - (s[2] - s0) / mpi**2) ** 2 + (v - (s[1] - s[0]) / mpi**2) ** 2,
        ),
        exact(
            "(p_K - p_i)^2 = PDG's (m_K - m_i)^2 - 2 m_K T_i (T_i = E_i - m_i)",
            sum(
                (s_kin[i] - ((mK - mi[i]) ** 2 - 2 * mK * (E[i] - mi[i]))) ** 2
                for i in range(3)
            ),
        ),
        exact(
            "s_0 = (1/3) sum s_i = PDG's (m_K^2 + m_1^2 + m_2^2 + m_3^2)/3",
            (sum(s_kin) / 3).subs(on_shell)
            - (mK**2 + sum(a**2 for a in mi)) / 3,
        ),
        Result(
            "pion 3 is the odd one (PDG: index 3 for the odd pion); "
            "eq. 1 of the note, p. 32",
            "where pion 3 is the odd one"
            in L.line("where pion 3 is the odd one")
            and "eq. 1 of PDG's note on the Dalitz-plot parameters "
            "($\\mathrm K^\\pm$ listing, p. 32)"
            in L.line("This is eq. 1 of PDG's note"),
            "as written",
        ),
        exact_fails("control: s_1 for s_3 in u", broken - (s[2] - s0) / mpi**2),
    ]


@derivation("kd.KL3pi0")
def check_KL3pi0():
    """1 + h(u^2 + v^2/3) is PDG's expansion (K0_L listing p. 20) symmetric
    under the permutations of three pi0."""
    s = sp.symbols("s_1:4", real=True)
    mpi, g, h, j, k, f, U, V = sp.symbols("m_pi g h j k f u v", real=True)
    u, v = readme_uv(s, mpi)

    def general(si):
        uu, vv = readme_uv(si, mpi)
        return 1 + g * uu + h * uu**2 + j * vv + k * vv**2 + f * uu * vv

    conditions = []
    for perm in permutations(s):
        conditions += sp.Poly(
            sp.expand(general(perm) - general(s)), *s
        ).coeffs()
    sym = sp.solve(conditions, [g, j, k, f], dict=True)
    readme = lambda: L.to_sympy(
        L.after(r"\to 3\pi^0$ has $"), {"h": h, "u": U, "v": V}
    ).subs({U: u, V: v})
    with L.patched(r"1 + h(u^2 + v^2/3)", r"1 + h(u^2 + v^2/2)"):
        broken = readme()
    return [
        Result(
            "symmetric under the 6 permutations: g = j = f = 0, k = h/3",
            sym == [{g: 0, j: 0, f: 0, k: h / 3}],
            f"solution {sym}",
        ),
        exact(
            "the page's form = the symmetric expansion "
            "(h = the coefficient of u^2, as PDG's h)",
            readme() - general(s).subs(sym[0]),
        ),
        exact_fails("control: v^2/2", broken - general(s).subs(sym[0])),
    ]


# PDG 2026, K0_L listing p. 42 (CP violation): K0_L -> pi+ pi- pi0, linear
# coefficient j = 0.0012 +- 0.0008 (OUR AVERAGE), quadratic coefficient
# f = 0.0045 +- 0.0024 +- 0.0059 (ANGELOPOULOS 98C): the CP-violating terms
# j v and f u v of |M|^2 ~ 1 + g u + h u^2 + j v + k v^2 + f u v.
PDG_JF = {
    "j": (sp.Rational("0.0012"), sp.Rational("0.0008")),
    "f": (
        sp.Rational("0.0045"),
        sp.sqrt(sp.Rational("0.0024") ** 2 + sp.Rational("0.0059") ** 2),
    ),
}


@derivation("kd.dalitz_values")
def check_dalitz_values():
    """The terms the page leaves out for K0_L -> pi+ pi- pi0 are PDG's
    CP-violating ones, consistent with zero."""
    u, v, j, f = sp.symbols("u v j f")
    cp = lambda: L.to_sympy(
        L.after("the CP-violating terms $"), {"j": j, "f": f, "u": u, "v": v}
    )
    with L.patched(r"terms $j\thinspace v + f", r"terms $j\thinspace u + f"):
        cp_broken = cp()
    return [
        exact(
            "the terms left out: PDG's CP-violating j v + f u v",
            cp() - (j * v + f * u * v),
        ),
        Result(
            "PDG's j and f are within 2 sigma of zero (K0_L listing p. 42)",
            all(abs(c_) < 2 * e for c_, e in PDG_JF.values()),
            ", ".join(
                f"{n} = {float(c_)} +- {float(e):.4f}"
                for n, (c_, e) in PDG_JF.items()
            ),
        ),
        exact_fails(
            "control: j u on the page", cp_broken - (j * v + f * u * v)
        ),
    ]


@derivation("kd.phase_space")
def check_phase_space():
    """Three-body phase space d^3p_1 d^3p_2 delta(M - E_1 - E_2 - E_3)/(E_1 E_2
    E_3) is uniform in E_1, E_2 (and in any pair)."""
    E1, E2, p1, p2, m3, cth = sp.symbols(
        "E_1 E_2 p_1 p_2 m_3 c_12", positive=True
    )
    E3 = sp.sqrt(m3**2 + p1**2 + p2**2 + 2 * p1 * p2 * cth)
    jac = 1 / sp.diff(E3, cth)
    measure = (p1 * E1) * (p2 * E2) * jac / (E1 * E2 * E3)
    Msq, dEi, dEj = sp.symbols("M2 dE_i dE_j", positive=True)

    def readme():
        lhs, rhs = L.after(r"so the decay rate is $").split(r"\propto")
        dens_l, dens_r = L.after(r"dE_j$, with the density $").split(r"\propto")
        assert lhs.strip() == r"d\Gamma" and dens_l.strip() == r"\rho", (
            lhs,
            dens_l,
        )
        density = parse(dens_r, {r"\vert\mathcal M\vert^2": Msq})
        return L.to_sympy(
            rhs, {r"\rho": density, "dE_i": dEi, "dE_j": dEj, "E_j": E2}
        )

    # the pin, once
    L.inline(r"d\Gamma \propto \rho\thinspace dE_i\thinspace dE_j")
    derived = Msq * sp.simplify(measure) * dEi * dEj
    with L.patched(
        r"\thinspace dE_i\thinspace dE_j$, with the density",
        r"\thinspace dE_i\thinspace E_j\thinspace dE_j$, with the density",
    ):
        broken = readme()
    # (E_1, E_2) -> (E_2, E_3 = M - E_1 - E_2)
    swap = sp.Matrix([[0, 1], [-1, -1]]).det()
    return [
        exact(
            "|M|^2 d^3p_1 d^3p_2 delta(...)/(E_1 E_2 E_3) = the page's "
            "rho dE_i dE_j, rho ∝ |M|^2 (constant angles)",
            readme() - derived,
        ),
        Result(
            "any other pair of energies: unit Jacobian",
            abs(swap) == 1,
            f"det = {swap}",
        ),
        exact_fails("control: E_j dE_j", broken - derived),
    ]


@derivation("kd.frame")
def check_frame():
    """p_K . p_a = M E_a (kaon at rest), and 2 p_a . p_b follows from
    (p_a + p_b)^2 = (p_K - p_c)^2."""
    Ea, Eb, Pa, Pb, cb = sp.symbols("E_a E_b P_a P_b c_b", real=True)
    sb = sp.sqrt(1 - cb**2)
    p = {"a": (Ea, 0, 0, Pa), "b": (Eb, Pb * sb, 0, Pb * cb)}
    p["c"] = (M - Ea - Eb, -Pb * sb, 0, -Pa - Pb * cb)
    pK = (M, 0, 0, 0)
    mass2 = {k: sp.expand(mdot(v, v)) for k, v in p.items()}
    kdot, pair = readme_frame()
    out = [
        exact(
            "p_K . p_a = M E_a for each product",
            sum((mdot(pK, v) - kdot(v[0])) ** 2 for v in p.values()),
        )
    ]
    for a, b in (("a", "b"), ("a", "c"), ("b", "c")):
        cc = ({"a", "b", "c"} - {a, b}).pop()
        ours = pair(p[cc][0], *(sp.sqrt(mass2[k]) for k in (a, b, cc)))
        out.append(
            exact(
                f"p_{a} . p_{b} from the energy of {cc} and the three masses",
                sp.expand(mdot(p[a], p[b]) - ours),
            )
        )
    with L.patched(r"= M^2 - 2ME_c + m_c^2", r"= M^2 + 2ME_c + m_c^2"):
        broken = readme_frame()[1](
            p["c"][0], *(sp.sqrt(mass2[k]) for k in "abc")
        )
    out.append(
        exact_fails(
            "control: + 2 M E_c", sp.expand(mdot(p["a"], p["b"]) - broken)
        )
    )
    return out


@derivation("kd.spin")
def check_spin():
    """s is the rest-frame spin along the momentum, boosted (helicity +1/2):
    s.p_l = 0, s^2 = -1. p_a.s and the 3-vector relation are a Minkowski product
    and momentum conservation squared."""
    (s0, s3), sdot, vdot, (P, El, ml, Ea, D, A2, B2, P2) = readme_spin()
    v = sp.Symbol("v", positive=True)
    lam = {El: ml * (v**2 + v**-2) / 2, P: ml * (v**2 - v**-2) / 2}
    n = sp.symbols("n1:4", real=True)
    s = (s0, *(s3 * a for a in n))
    pl = (El, *(P * a for a in n))
    unit = {n[2]: sp.sqrt(1 - n[0] ** 2 - n[1] ** 2)}
    rest_t = (El * s0 - P * s3) / ml
    rest_l = (El * s3 - P * s0) / ml
    a = sp.symbols("a1:4", real=True)
    pa = (Ea, *a)
    with L.patched(
        r"\frac{E_a\vert\vec p_\ell\vert - E_\ell",
        r"\frac{E_a\vert\vec p_\ell\vert + E_\ell",
    ):
        broken = readme_spin()[1]
    return [
        exact("s.p_l = 0", sp.expand(mdot(s, pl)).subs(unit).subs(lam)),
        exact("s.s = -1", (sp.expand(mdot(s, s)).subs(unit) + 1).subs(lam)),
        exact(
            "in the lepton's rest frame s = (0, p_l-hat): spin along the "
            "momentum, helicity +1/2",
            (rest_t.subs(lam)) ** 2 + (rest_l.subs(lam) - 1) ** 2,
        ),
        exact(
            "p_a . s = (E_a |p_l| - E_l p_a . p_l-hat)/m_l "
            "(Minkowski product; p_K: no 3-momentum)",
            sp.expand(
                mdot(pa, s) - sdot.subs(D, sum(x_ * y for x_, y in zip(a, n)))
            ),
        ),
        exact(
            "2 p_a . p_l = |p_b|^2 - |p_a|^2 - |p_l|^2 for p_b = -(p_a + p_l)",
            vdot.subs(
                {
                    A2: sum(y**2 for y in a),
                    P2: sum(y**2 for y in n),
                    B2: sum((y + z) ** 2 for y, z in zip(a, n)),
                }
            )
            - sum(y * z for y, z in zip(a, n)),
        ),
        exact_fails(
            "control: + E_l in p_a . s",
            sp.expand(
                mdot(pa, s) - broken.subs(D, sum(x_ * y for x_, y in zip(a, n)))
            ),
        ),
    ]


LINE_PIN = r"E_j^{\rm lo,hi} = \frac{(M - E^\ast)"
MJ, MK_ = sp.symbols("m_j m_k", positive=True)


def readme_line(E=Es, p=ps, mi=m, mj=MJ, mk=MK_):
    rel = relations(LINE_PIN)
    tokens = {"M": M, "m": mi, "m_j": mj, "m_k": mk, r"E^\ast": E, r"p^\ast": p}
    mjk2 = L.to_sympy(rel["m_{jk}^2"], tokens)
    Ejp = L.to_sympy(rel["E_j'"], {**tokens, "m_{jk}": sp.sqrt(mjk2)})
    Emax = L.to_sympy(rel[r"E_{\max}"], tokens)
    ends = [
        L.to_sympy(
            rel[r"E_j^{\rm lo,hi}"],
            {
                **tokens,
                "E_j'": Ejp,
                "p_j'": sp.sqrt(Ejp**2 - mj**2),
                "m_{jk}": sp.sqrt(mjk2),
            },
            sign=sg,
        )
        for sg in (1, -1)
    ]
    return ends[0], ends[1], mjk2, Ejp, Emax


@derivation("kd.line")
def check_line():
    """The Dalitz line of product i: m_jk^2 = (p_K - p_i)^2, E_j' of the pair's
    rest frame, the ends and E_max. The ends are at the collinear
    configurations, and the line closes at E_max."""
    Ej = sp.Symbol("E_j", real=True)
    lo, hi, mjk2, Ejp, Emax = readme_line()
    on_shell = {ps**2: Es**2 - m**2}
    pk2, pj2 = (M - Es - Ej) ** 2 - MK_**2, Ej**2 - MJ**2
    quad = sp.Poly(sp.expand((pk2 - ps**2 - pj2) ** 2 - 4 * ps**2 * pj2), Ej)
    a2, a1, a0 = quad.all_coeffs()
    out = [
        exact(
            "m_jk^2 = (p_K - p_i)^2 = M^2 - 2 M E* + m^2",
            mjk2 - (M**2 - 2 * M * Es + m**2),
        ),
        exact(
            "E_j' = (m_jk^2 + m_j^2 - m_k^2)/(2 m_jk): the pair's "
            "two-body decay at rest",
            ((sp.sqrt(mjk2) - Ejp) ** 2 - MK_**2) - (Ejp**2 - MJ**2),
        ),
        exact(
            "lo + hi = the sum of the boundary's roots in E_j",
            sp.expand((lo + hi) * a2 + a1).subs(on_shell),
        ),
        exact(
            "lo hi = the product of the boundary's roots",
            sp.expand(lo * hi * a2 - a0).subs(on_shell),
        ),
        exact(
            "E_max: the E* where m_jk = m_j + m_k (p_j' = 0, the line closes)",
            sp.solve(sp.Eq(mjk2, (MJ + MK_) ** 2), Es)[0] - Emax,
        ),
        exact("at E_max, E_j' = m_j", Ejp.subs(Es, Emax) - MJ),
    ]
    with L.patched(r"\mp p^\ast p_j'}{m_{jk}}", r"\mp 2p^\ast p_j'}{m_{jk}}"):
        blo, bhi = readme_line()[:2]
    out.append(
        exact_fails(
            "control: 2 p* p_j'", sp.expand(blo * bhi * a2 - a0).subs(on_shell)
        )
    )
    return out


def l3_values():
    label = r"$\mathrm K^\pm$"
    m_ = re.search(
        re.escape(f"The {label} values are ")
        + r"\$\\lambda'_ \+ = (.+?)\$, \$\\lambda''_ \+ = (.+?)\$ "
        r"and \$\\lambda_0 = (.+?)\$",
        L.line(f"The {label} values are"),
    )
    return {
        L1: sp.Rational(m_.group(1)),
        L2: sp.Rational(m_.group(2)),
        L0: sp.Rational(m_.group(3)),
    }


MASSES = mojo_constants.read(["MASS_K", "MASS_PI", "MASS_PI0", "MASS_MU"])


def dalitz_moments(rho, masses, M_, polar=None, dps=40):
    """rho: a sympy expression in the energies E = (E_1, E_2, E_3). polar:
    (i, rho P, P), where P is the symbol of |p_i| in rho P."""
    E = sp.symbols("E_1:4", real=True)
    Esym, Ej = sp.symbols("E_star E_j", real=True)
    out = {"Z": [], "E": [], "gG": []}
    with mp.workdps(dps):
        for i in range(3):
            j, k = (1, 2) if i == 0 else (0, 2) if i == 1 else (0, 1)
            sub = {E[i]: Esym, E[j]: Ej, E[k]: M_ - Esym - Ej}
            prim = sp.integrate(sp.expand(rho.subs(sub)), Ej)
            lo, hi, _, _, Emax = readme_line(
                Esym,
                sp.sqrt(Esym**2 - masses[i] ** 2),
                masses[i],
                masses[j],
                masses[k],
            )
            lo, hi, Emax = (e.subs(M, M_) for e in (lo, hi, Emax))
            line = sp.lambdify(
                Esym, prim.subs(Ej, hi) - prim.subs(Ej, lo), "mpmath"
            )
            bounds = [
                mp.mpf(sp.Float(masses[i], dps + 5)),
                mp.mpf(sp.Float(Emax, dps + 5)),
            ]
            out["Z"].append(mp.quad(line, bounds))
            out["E"].append(mp.quad(lambda e: e * line(e), bounds))
            if polar is not None and polar[0] == i:
                pprim = sp.integrate(sp.expand(polar[1].subs(sub)), Ej)
                pline = sp.lambdify(
                    Esym,
                    (pprim.subs(Ej, hi) - pprim.subs(Ej, lo)).subs(
                        polar[2], sp.sqrt(Esym**2 - masses[i] ** 2)
                    ),
                    "mpmath",
                )
                out["gG"] = [
                    (line(e), pline(e))
                    for e in (
                        bounds[0] + (bounds[1] - bounds[0]) * f
                        for f in (mp.mpf("0.1"), mp.mpf("0.5"), mp.mpf("0.9"))
                    )
                ]
    return out


def readme_g_norm():
    """The integration bounds and the integrands are asserted as written, so g
    and G are the line integrals of kd.line."""
    rel = relations(r"g(E^\ast) = \frac{1}{Z}\int")
    Zs = sp.Symbol("Z", positive=True)
    out = []
    for lhs, body in (
        ("g(E^\\ast)", r"\rho\thinspace dE_j"),
        ("G(E^\\ast)", r"\rho\thinspace\mathcal P\thinspace dE_j"),
    ):
        pre, rest = rel[lhs].split(r"\int_{E_j^{\rm lo}}^{E_j^{\rm hi}} ")
        assert rest == body, rest
        out.append(L.to_sympy(pre, {"Z": Zs}))
    assert rel["Z"] == r"\iint\rho\thinspace dE_i\thinspace dE_j", rel["Z"]
    return out, Zs


@derivation("kd.g")
def check_g():
    """g and G integrate the density along the Dalitz lines of kd.line. Z is the
    same from every product's lines, and each g has the integral 1. The mean
    energies add up to M, and |G| <= g (40 digits)."""
    mK, mpi, mpi0, mmu = (
        MASSES[k] for k in ("MASS_K", "MASS_PI", "MASS_PI0", "MASS_MU")
    )
    E = sp.symbols("E_1:4", real=True)
    (pre_g, pre_G), Zs = readme_g_norm()
    gk = [
        sp.Rational(v)
        for v in L.after(r"$\left(g, h, k\right)$ $= ").strip("()").split(", ")
    ]
    s = [mK**2 + mpi**2 - 2 * mK * e for e in E]
    rho3 = readme_dalitz(s, mpi, *gk)  # pion 3 (E_3) the odd one
    res3 = dalitz_moments(rho3, (mpi, mpi, mpi), mK)
    P = sp.Symbol("P", positive=True)
    ff = l3_values()
    tval = mK**2 + mpi0**2 - 2 * mK * E[0]
    fp = readme_fplus().subs({**ff, T_: tval, MPIP: mpi})
    fm = readme_fminus_code(mpi0).subs({**ff, T_: tval, MPIP: mpi, M: mK})
    W0, Wp = (
        w.subs(M, mK) for w in readme_l3(fp, fm, E[0], E[1], mpi0, mmu, P)
    )
    resl = dalitz_moments(W0, (mpi0, mmu, 0), mK, polar=(1, Wp, P))
    out = []
    fg, fG = (sp.lambdify(Zs, f, "mpmath") for f in (pre_g, pre_G))
    for name, res in (("K+ -> pi+ pi+ pi-", res3), ("K+ -> pi0 mu+ nu", resl)):
        Z = res["Z"][0]
        with mp.workdps(40):
            norm = [fg(Z) * z for z in res["Z"]]
            energy = sum(fg(Z) * e for e in res["E"])
        worst = worst_of(
            [abs(n_ - 1) for n_ in norm]
            + [abs(energy / mp.mpf(sp.Float(mK, 45)) - 1)]
        )
        out.append(
            Result(
                f"{name}: Z the same from each product, int g dE* = 1, "
                "sum of the mean energies = M",
                worst < mp.mpf(10) ** -30,
                f"max rel. difference {mp.nstr(worst, 3)}, 40 digits",
            )
        )
    ratio = [
        abs(fG(resl["Z"][0]) * G) / (fg(resl["Z"][0]) * g)
        for g, G in resl["gG"]
    ]
    out.append(
        Result(
            "K+ -> pi0 mu+ nu: |G| <= g on the muon's line (|P| <= 1)",
            all(r <= 1 for r in ratio),
            "|G|/g = " + ", ".join(mp.nstr(r, 4) for r in ratio),
        )
    )
    with L.patched(r"\mp p^\ast p_j'}{m_{jk}}", r"\mp 2p^\ast p_j'}{m_{jk}}"):
        bad = dalitz_moments(rho3, (mpi, mpi, mpi), mK, dps=20)
    spread = max(abs(z / bad["Z"][0] - 1) for z in bad["Z"]) + abs(
        sum(bad["E"]) / bad["Z"][0] / mp.mpf(sp.Float(mK, 25)) - 1
    )
    out.append(
        Result(
            "control: with 2 p* p_j' in the line's ends, Z or the "
            "energy sum is off",
            spread > 1e-3,
            f"off by {mp.nstr(spread, 3)}",
        )
    )
    return out


QUAD_PIN = r"E^\ast = m + (E_{\max} - m)\sin^2(\pi u/2)"


@derivation("kd.quad")
def check_quad():
    """The rest-frame quadratures: along a Dalitz line, the densities are
    polynomials of degree 2 for K -> 3pi and at most 5 for K_l3. The 24
    Gauss-Legendre points are exact up to degree 47. In u, the line's length
    has no square-root edge."""
    out = []
    E = sp.symbols("E_1:4", real=True)
    y, Ei, mp_, mpi_, ml_, P = sp.symbols(
        "y E_i m_pip m_pi m_l P", positive=True
    )
    g, h, k = sp.symbols("g h k", real=True)
    mi = sp.symbols("m_1:4", positive=True)
    s = [M**2 + mi[i] ** 2 - 2 * M * E[i] for i in range(3)]
    rho3 = readme_dalitz(s, mp_, g, h, k)
    lines = lambda i: {E[i]: Ei, E[(i + 1) % 3]: y, E[(i + 2) % 3]: M - Ei - y}
    deg3 = max(
        sp.Poly(sp.expand(rho3.subs(lines(i))), y).degree() for i in range(3)
    )
    tval = M**2 + mpi_**2 - 2 * M * E[0]
    fp = readme_fplus().subs({T_: tval, MPIP: mp_})
    fm = readme_fminus_code(mpi_).subs({T_: tval, MPIP: mp_})
    W0, Wp = readme_l3(fp, fm, E[0], E[1], mpi_, ml_, P)
    degl = [sp.Poly(sp.expand(W0.subs(lines(i))), y).degree() for i in range(3)]
    # W_pol on the lepton's line: |p_l| = P is fixed there
    degp = sp.Poly(sp.expand(Wp.subs(lines(1))), y).degree()
    m_ = re.search(
        r"of degree (\d+) for \$\\mathrm K \\to 3\\pi\$ and at most (\d+) for "
        r"\$\\mathrm K_ \{l3\}\$\. The code's (\d+)-point Gauss–Legendre rule",
        L.line("Along a Dalitz line, the densities are polynomials"),
    )
    n3, nl, npts = (int(v) for v in m_.groups())
    out += [
        Result(
            "K -> 3pi: degree in E_j along each line = the page's",
            deg3 == n3,
            f"degree {deg3}",
        ),
        Result(
            "K_l3: W_0 on the three lines and W_pol on the lepton's line, "
            "at most the page's degree",
            max(degl + [degp]) == nl,
            f"W_0 {degl}, W_pol {degp}",
        ),
        Result(
            "the page's Gauss-Legendre rule is exact up to degree 2n - 1",
            2 * npts - 1 >= max(n3, nl),
            f"{npts} points: degree {2 * npts - 1}",
        ),
    ]
    s2 = sp.Symbol("s2", positive=True)
    left, right = L.sides(L.inline(QUAD_PIN))
    assert left == r"E^\ast", left
    lo, hi, mjk2, Ejp, Emax = readme_line()
    sub_u = lambda expr: sp.together(
        expr.subs(
            Es,
            L.to_sympy(
                right,
                {
                    "m": m,
                    r"E_{\max}": Emax,
                    "u": 2 * sp.asin(sp.sqrt(s2)) / sp.pi,
                },
            ),
        )
    )
    length2 = sp.expand((hi - lo) ** 2).subs(ps**2, Es**2 - m**2)
    ratio = sp.factor(sub_u(length2) / (s2 * (1 - s2)))
    vals = [sp.simplify(ratio.subs(s2, v)) for v in (0, 1)]
    mK, mpi = MASSES["MASS_K"], MASSES["MASS_PI"]
    sample = [float(v.subs({M: mK, m: mpi, MJ: mpi, MK_: mpi})) for v in vals]
    mjk_top = sp.factor(mjk2 - (MJ + MK_) ** 2 - 2 * M * (Emax - Es))
    out += [
        exact(
            "m_jk^2 - (m_j + m_k)^2 = 2M (E_max - E*): p_j'^2 vanishes "
            "linearly in E* at the top",
            mjk_top,
        ),
        Result(
            "massive products: (hi - lo)^2/(sin^2 cos^2) is finite and "
            "non-zero at u = 0 and 1 (square-root edges, "
            "both turned smooth)",
            all(
                v.is_finite is not False
                and v != 0
                and not v.has(sp.zoo, sp.nan)
                for v in vals
            )
            and all(0 < abs(v) < float("inf") for v in sample),
            "a pion's line of K+ -> pi+ pi+ pi-: "
            + ", ".join(f"{v:.4g} GeV^2" for v in sample),
        ),
        exact(
            "a massless product i (m = 0): the ratio vanishes at u = 0, "
            "a linear edge (sin^2 after the substitution)",
            sp.limit(ratio.subs(m, 0), s2, 0),
        ),
        exact(
            "a massless partner (m_k = 0): the ratio vanishes at u = 1, "
            "a linear edge (p_j' ∝ m_jk^2 - m_j^2)",
            sp.limit(ratio.subs(MK_, 0), s2, 1),
        ),
    ]
    with L.patched("and at most 5 for", "and at most 4 for"):
        n_broken = int(
            re.search(
                r"at most (\d+) for",
                L.line("Along a Dalitz line, the densities are polynomials"),
            ).group(1)
        )
    out.append(
        Result(
            "control: 'at most 4' fails the degree",
            max(degl + [degp]) != n_broken,
            f"{max(degl + [degp])} != {n_broken}",
        )
    )
    return out


@derivation("kd.flat")
def check_flat():
    """x = E/E_K is (E* + p* cos theta*)/M for gamma -> oo: uniform over the
    page's range with density M/(2p*)."""
    # cos(theta*) = c, uniform (isotropic)
    xc = sp.limit(boost((Es, 0, 0, ps * c))[0] / (GAM * M), q, sp.oo)
    lo, hi = readme_range()
    with L.patched(r"with density $M/(2p^\ast)$", r"with density $M/p^\ast$"):
        broken = readme_density()
    return [
        Result(
            "x is linear in cos theta* (so uniform)",
            sp.diff(xc, c, 2) == 0,
            f"x = {xc}",
        ),
        exact(
            "x at cos theta* = -1, +1: the page's ends (E* -+ p*)/M",
            (xc.subs(c, -1) - lo) ** 2 + (xc.subs(c, 1) - hi) ** 2,
        ),
        exact(
            "(1/2)/(dx/dcos theta*) = the page's density M/(2p*)",
            sp.Rational(1, 2) / sp.diff(xc, c) - readme_density(),
        ),
        exact_fails(
            "control: density M/p*", sp.Rational(1, 2) / sp.diff(xc, c) - broken
        ),
    ]


@derivation("kd.F")
def check_F():
    """F(x) adds the flat spectra of the nodes E* whose range holds x: those
    with E* >= E_lo(x) (>= m)."""
    E, v, w1 = sp.symbols("E v w_1", positive=True)
    Elo = readme_Elo()
    lo, hi = (e.subs({Es: E, ps: sp.sqrt(E**2 - m**2)}) for e in readme_range())
    inside = sp.expand((x - lo) * (x - hi))
    ends = sp.solve(inside, E)
    out = [
        Result(
            "x is at an end of the range of one node E* only",
            len(ends) == 1,
            f"E* = {ends}",
        ),
        exact("it is the page's E_lo(x)", ends[0] - Elo),
        exact(
            "M^2 (x - x_lo)(x - x_hi) = -2Mx (E* - E_lo): x is in the "
            "range of the node iff E* >= E_lo",
            M**2 * inside + 2 * M * x * (E - Elo),
        ),
        exact(
            "E_lo - m = (Mx - m)^2/(2Mx) >= 0 (the page's >= m)",
            Elo - m - (M * x - m) ** 2 / (2 * M * x),
        ),
        exact(
            "the page's integrand (M/2) g/p* = g times the flat density "
            "of kd.flat",
            readme_F() - g_ * readme_density(),
        ),
    ]
    # The page's F keeps the number and energy of g = p* (a0 + a1 E* + a2 E*^2)
    # on [m, E_max].
    a = sp.symbols("a0:3", real=True)
    poly = a[0] + a[1] * E + a[2] * E**2
    Ev, pv, vmax = m * (v**2 + v**-2) / 2, m * (v**2 - v**-2) / 2, 1 + w1
    Emax, pmax = Ev.subs(v, vmax), pv.subs(v, vmax)

    def moments(Elo_):
        F = laurent_integral(
            readme_F(g=ps * poly.subs(E, Es)).subs(Es, E), E, Elo_, Emax
        )
        xs = ((Emax - pmax) / M, (Emax + pmax) / M)
        return [laurent_integral(F * x**k, x, *xs) for k in (0, 1)]

    gv = pv * poly.subs(E, Ev) * sp.diff(Ev, v)
    truth = [
        laurent_integral(gv, v, 1, vmax),
        laurent_integral(gv * Ev / M, v, 1, vmax),
    ]
    ours = moments(Elo)
    with L.patched(r"M^2x^2 + m^2", r"M^2x^2 - m^2"):
        broken = moments(readme_Elo())
    out += [
        exact(
            "int F dx = int g dE* (g = p* (a0 + a1 E* + a2 E*^2))",
            ours[0] - truth[0],
        ),
        exact(
            "int x F dx = int (E*/M) g dE* (each flat range has the "
            "mean x = E*/M)",
            ours[1] - truth[1],
        ),
        exact_fails(
            "control: E_lo = (M^2 x^2 - m^2)/(2Mx) loses number",
            broken[0] - truth[0],
        ),
    ]
    return out


@derivation("kd.two_body")
def check_two_body():
    """E* = (M^2 + m^2 - m_b^2)/(2M) by energy-momentum conservation. There is
    one E*, so F is flat, and its x never exceeds 1."""
    mb = sp.Symbol("m_b", positive=True)
    E0 = readme_two_body(mb)[0]
    solved = sp.solve((M - Es) ** 2 - mb**2 - (Es**2 - m**2), Es)
    p2 = E0**2 - m**2
    p0 = sp.Symbol("p_0", positive=True)
    flat = readme_F(g=1, E=E0, p=p0)  # g = delta(E* - E0): one node
    with L.patched(r"- m_b^2)/(2M)$, so", r"+ m_b^2)/(2M)$, so"):
        broken = readme_two_body(mb)[0]
    return [
        Result("one solution E*", len(solved) == 1, f"{solved}"),
        exact(
            "E* = (M^2 + m^2 - m_b^2)/(2M): M = E* + E_b, |p_b| = |p*|",
            E0 - solved[0],
        ),
        exact(
            "p*^2 = [M^2 - (m + m_b)^2][M^2 - (m - m_b)^2]/(4M^2)",
            p2 - (M**2 - (m + mb) ** 2) * (M**2 - (m - mb) ** 2) / (4 * M**2),
        ),
        Result(
            "one node: F = M/(2p*) on its range (kd.flat), the same at every x",
            flat == M / (2 * p0) and not flat.has(x),
            f"F = {flat}",
        ),
        exact(
            "(M^2 - m^2 + m_b^2)^2 - 4 M^2 p*^2 = 4 M^2 m_b^2 >= 0: "
            "(E* + p*)/M <= 1 (= 1 for m_b = 0), so x <= 1",
            (M**2 - m**2 + mb**2) ** 2 - 4 * M**2 * p2 - 4 * M**2 * mb**2,
        ),
        exact_fails("control: + m_b^2", broken - solved[0]),
    ]


@derivation("kd.htilde")
def check_htilde():
    """A muon polarized along its kaon-frame momentum has the lab helicity P h~
    (gamma -> oo). Transverse polarization averages out."""
    phi = sp.Symbol("phi", real=True)
    mm, sth = sp.sqrt(Es**2 - ps**2), sp.sqrt(1 - c**2)
    lab = boost((Es, ps * sth, 0, ps * c))
    plab = sp.sqrt(lab[0] ** 2 - mm**2)
    # spin 4-vector a: a^0 = (|p|/m) (rest-frame spin . p^)
    helicity = lambda a: mm * boost(a)[0] / plab
    sL = (ps / mm, Es / mm * sth, 0, Es / mm * c)
    e1, e2 = (0, c, 0, -sth), (0, 0, 1, 0)
    trans = [
        tuple(sp.cos(phi) * a + sp.sin(phi) * b for a, b in zip(e1, e2)),
        tuple(-sp.sin(phi) * a + sp.cos(phi) * b for a, b in zip(e1, e2)),
    ]
    hL = sp.limit(helicity(sL), q, sp.oo)
    cx = sp.solve(sp.Eq(sp.limit(lab[0] / (GAM * M), q, sp.oo), x), c)
    ht = readme_htilde().subs(m, mm)
    lo, hi = readme_range()
    with L.patched(
        r"(E^\ast M x - m^2)/(p^\ast M x)", r"(E^\ast M x + m^2)/(p^\ast M x)"
    ):
        broken = readme_htilde().subs(m, mm)
    return [
        exact(
            "s_L: s.p = 0, s^2 = -1",
            mdot(sL, (Es, ps * sth, 0, ps * c)) ** 2 + (mdot(sL, sL) + 1) ** 2,
        ),
        Result("one cos theta* for each x", len(cx) == 1, f"cos theta* = {cx}"),
        exact(
            "lab helicity of s_L for gamma -> oo = the page's h~",
            (hL - ht).subs(c, cx[0]),
        ),
        exact(
            "polarisation normal to the momentum (in and out of the decay "
            "plane): 0 on average over its orientation",
            sum(
                sp.integrate(helicity(a), (phi, 0, 2 * sp.pi)) ** 2
                for a in trans
            ),
        ),
        exact(
            "h~ = -1 at x = (E* - p*)/M (backward), +1 at (E* + p*)/M",
            (ht.subs(x, lo) + 1) ** 2 + (ht.subs(x, hi) - 1) ** 2,
        ),
        exact(
            "dh~/dx = m^2/(p* M x^2) > 0 (|h~| <= 1 on the range)",
            sp.diff(ht, x) - mm**2 / (ps * M * x**2),
        ),
        exact_fails("control: E* M x + m^2", (hL - broken).subs(c, cx[0])),
    ]


@derivation("kd.FRL")
def check_FRL():
    """F_R,L: a muon is right-handed with probability (1 + P h~)/2. Along the
    Dalitz line, that gives (g +- G h~)/2."""
    y, y1, y2 = sp.symbols("y y_1 y_2", real=True)
    a, b = sp.symbols("a0:3", real=True), sp.symbols("b0:2", real=True)
    rho, P = a[0] + a[1] * y + a[2] * y**2, b[0] + b[1] * y
    g, G = sp.integrate(rho, (y, y1, y2)), sp.integrate(rho * P, (y, y1, y2))
    ht = readme_htilde()
    out = []
    for sign, n in ((1, "R"), (-1, "L")):
        events = sp.integrate(
            rho * readme_density() * (1 + sign * P * ht) / 2, (y, y1, y2)
        )
        out.append(
            exact(
                f"the page's F_{n} integrand = int along the line of "
                f"rho (M/2p*) (1 {'+-'[sign < 0]} P h~)/2",
                readme_F(sign, g, G, ht) - events,
            )
        )
    out.append(
        exact(
            "F_R + F_L = F",
            readme_F(1, ht=ht) + readme_F(-1, ht=ht) - readme_F(),
        )
    )
    with L.patched(r"{2\thinspace p^\ast}", r"{p^\ast}"):
        out.append(
            exact_fails(
                "control: p* for 2p* in F_R,L",
                readme_F(1, ht=ht) + readme_F(-1, ht=ht) - readme_F(),
            )
        )
    return out


LIPARI_PIN = r"F_{\mu_R^+}(x) = "


def readme_lipari(r):
    rel = relations(LIPARI_PIN)
    return [
        L.to_sympy(rel[k], {"r": r, "x": x})
        for k in (r"F_{\mu_R^+}(x)", r"F_{\mu_L^+}(x)")
    ]


@derivation("kd.lipari")
def check_lipari():
    """For K -> mu nu, F_R,L (P = -1) give the page's F_mu+R,L = Lipari et al.'s
    eqs. 18-19 with m_pi -> m_K. P h~ is their eq. 16, and F_R + F_L is their
    eq. 15."""
    r = sp.Symbol("r", positive=True)
    rdef = L.to_sympy(
        L.sides(L.inline(r"r = m_\mu^2/M^2"))[1], {r"m_\mu": m, "M": M}
    )
    E0, p0 = readme_two_body()
    P = readme_P()
    ht = readme_htilde(E0, p0)
    # one node, g = 1 (two-body), G = P g
    ours = [readme_F(sg, 1, P, ht, E0, p0) for sg in (1, -1)]
    page = readme_lipari(r)
    to_r = {m: M * sp.sqrt(r)}
    eq18, rl = lipari(LIP_18.split(" = ")[2].split(" ~ ")[0])
    eq19, _ = lipari(LIP_19.split(" = ")[2].split(" ~ ")[0])
    eq16, _ = lipari(LIP_16.split(" = ")[1])
    eq15, _ = lipari(
        LIP_15.split(" = ")[1]
        .split(" ~ ")[0]
        .replace(r"{1 \over 1 -r_\pi }", r"\frac{1}{1 - r_\pi}")
    )
    lo, hi = (e.subs({Es: E0, ps: p0}) for e in readme_range())
    with L.patched(
        r"\frac{r\thinspace (1 - x)}", r"\frac{r\thinspace (1 + x)}"
    ):
        broken = readme_lipari(r)[0]
    return [
        in_source(
            "eqs. 13, 15, 16, 18 and 19 and the text as copied",
            LIPARI,
            LIP_13,
            LIP_15,
            LIP_16,
            LIP_18,
            LIP_19,
            *LIP_TEXT,
        ),
        exact(
            "r = m_mu^2/M^2 (their r_pi = (m_mu/m_pi)^2 with m_pi -> M)",
            rdef - m**2 / M**2,
        ),
        exact(
            "the page's F_R,L for K -> mu nu (P = -1) = its F_mu+R, F_mu+L",
            sum(
                (sp.simplify(o.subs(to_r)) - p_) ** 2
                for o, p_ in zip(ours, page)
            ),
        ),
        exact(
            "= Lipari et al.'s eqs. 18 (F_R) and 19 (F_L), r_pi -> r",
            (page[0] - eq18.subs(rl, r)) ** 2
            + (page[1] - eq19.subs(rl, r)) ** 2,
        ),
        exact(
            "F_R + F_L = eq. 15's 1/(1 - r) on r <= x <= 1 "
            "(the range of kd.flat)",
            (page[0] + page[1] - eq15.subs(rl, r)) ** 2
            + (sp.simplify(lo.subs(to_r)) - r) ** 2
            + (sp.simplify(hi.subs(to_r)) - 1) ** 2,
        ),
        exact(
            "P h~ = eq. 16's helicity, -1 at x = 1",
            sp.simplify((P * ht).subs(to_r)) - eq16.subs(rl, r),
        ),
        in_source(
            "eq. 19 prints F_{pi- -> mu+_R} for the conjugate decay",
            LIPARI,
            r"F_{\pi^- \to \mu^+_R}",
        ),
        Result(
            "the page notes eq. 19's mu+_R for the mu-_R of the conjugate "
            "decay",
            r"with $\mu^+_ R$ printed for $\mu^-_ R$"
            in L.line("Their eq. 19 also gives the conjugate decay"),
            "as written",
        ),
        exact_fails("control: r(1 + x)", broken - eq18.subs(rl, r)),
    ]


SCALING_PIN = r"\frac{dn_{a \to b}}{dE}(E_b; E_a) = "


def readme_scaling(Ea, Eb, F):
    rel = relations(SCALING_PIN)
    left = r"\frac{dn_{a \to b}}{dE}(E_b; E_a)"
    return L.to_sympy(rel[left], {"E_a": Ea, "E_b": Eb}, {r"F_{a \to b}": F})


@derivation("kd.scaling")
def check_scaling():
    """The page quotes Lipari et al.'s eq. 13 (the scaling form of a decay
    spectrum) as printed."""
    Ea, Eb = sp.symbols("E_a E_b", positive=True)
    F = sp.Function("F")
    left, right = LIP_13.replace("~", " ").split(" = ")
    paper = L.to_sympy(right, {"E_a": Ea, "E_b": Eb}, {r"F_{a \to b}": F})
    with L.patched(
        r"= \frac{1}{E_a}\thinspace F_{a \to b}",
        r"= \frac{1}{E_b}\thinspace F_{a \to b}",
    ):
        broken = readme_scaling(Ea, Eb, F)
    return [
        in_source("eq. 13 as copied", LIPARI, LIP_13),
        Result(
            "the same left side",
            re.sub(r"\s", "", left)
            == re.sub(
                r"\s", "", L.sides(L.parts(L.display(SCALING_PIN))[0])[0]
            ),
            "dn_{a -> b}/dE (E_b; E_a)",
        ),
        exact(
            "the page's right side = eq. 13's",
            readme_scaling(Ea, Eb, F) - paper,
        ),
        exact_fails("control: 1/E_b", broken - paper),
    ]


def readme_decay_source(EK, Ns, FX, tau):
    lhs, rhs = L.sides(L.parts(L.display(r"Q_X(E) = \sum_s \int_E^\infty"))[0])
    head, rest = rhs.split(r"\int_E^\infty ", 1)
    measure, body = rest.split(r"\thinspace ", 1)
    assert lhs == "Q_X(E)" and head.strip() == r"\sum_s", rhs
    ms, cl, dEK = sp.symbols("m_s c dE_K", positive=True)
    gamma = L.to_sympy(
        L.sides(L.inline(r"\gamma_K = E_K/(m_s c^2)"))[1],
        {"E_K": EK, "m_s": ms, "c": cl},
    )
    weight = sp.diff(L.to_sympy(measure, {"dE_K": dEK, "E_K": EK}), dEK)
    tokens = {
        "E": sp.Symbol("E", positive=True),
        "E_K": EK,
        r"\gamma_K": gamma,
        r"\tau_s": tau,
    }
    return (
        weight,
        L.to_sympy(body, tokens, {"N_s": Ns, r"F_{s\to X}": FX}),
        gamma,
    )


@derivation("kd.source")
def check_source():
    """Q_X counts the decays: N_s dE_K kaons decay at the time-dilated rate,
    and each decay makes dn/dE = F(x)/E_K products (eq. 13)."""
    EK, tau, ms, cl = sp.symbols("E_K tau_s m_s c", positive=True)
    E = sp.Symbol("E", positive=True)
    Ns, FX = sp.Function("N_s"), sp.Function("F")
    weight, body, gamma = readme_decay_source(EK, Ns, FX, tau)
    xv = L.to_sympy(L.sides(L.inline(r"x = E/E_K"))[1], {"E": E, "E_K": EK})
    counted = Ns(EK) / (EK / (ms * cl**2) * tau) * FX(xv) * sp.diff(xv, E)
    rate = L.to_sympy(
        L.inline(r"1/(\gamma_K\tau_s)"), {r"\gamma_K": gamma, r"\tau_s": tau}
    )
    per_decay = readme_scaling(EK, E, FX)  # eq. 13: a -> the kaon, b -> X
    with L.patched(
        r"\int_E^\infty \frac{dE_K}{E_K}\thinspace",
        r"\int_E^\infty dE_K\thinspace",
    ):
        w_b, b_b, _ = readme_decay_source(EK, Ns, FX, tau)
    return [
        exact(
            "the page's Q_X = the count of the decays per unit energy "
            "(any N_s, F; integrands in E_K)",
            weight * body - counted,
        ),
        exact(
            "= sum over the kaons of N_s/(gamma tau_s) times eq. 13's dn/dE",
            weight * body - Ns(EK) * rate * per_decay,
        ),
        exact(
            "the page's rate 1/(gamma_K tau_s) is the time-dilated one, "
            "gamma_K = E_K/(m_s c^2)",
            rate - ms * cl**2 / (EK * tau),
        ),
        Result(
            "the integral from E_K = E: no product has x > 1 "
            "((E* + p*)/M <= 1, kd.two_body)",
            True,
            "as written (asserted)",
        ),
        exact_fails("control: dE_K for dE_K/E_K", w_b * b_b - counted),
    ]


PERSOURCE_PIN = r"E\thinspace Q_X(E) = \sum_s \int_{\ln E}^\infty"


def readme_persource(EK, E, N, T, F, tau, gamma):
    rel = relations(PERSOURCE_PIN)
    rhs = rel[r"E\thinspace Q_X(E)"]
    head, rest = rhs.split(r"\int_{\ln E}^\infty ", 1)
    assert head.strip() == r"\sum_s" and rest.endswith(r" d\ln E_K"), rhs
    integrand = L.to_sympy(
        rest[: -len(r" d\ln E_K")],
        {"E_K": EK, "E": E, r"\gamma_K": gamma, r"\tau_s": tau},
        {"N_s": N, r"T_{s\to X}": T},
    )
    Tx = L.to_sympy(rel["T(x)"], {"x": x}, {"F": F})
    return integrand, Tx


@derivation("kd.persource")
def check_persource():
    """E Q_X per d ln E_K: the page's Q_X (dE_K/E_K = d ln E_K) times E, with
    T(x) = x F(x)."""
    EK, E, tau = sp.symbols("E_K E tau_s", positive=True)
    N, F, T = sp.Function("N_s"), sp.Function("F"), sp.Function("T")
    weight, body, gamma = readme_decay_source(EK, N, F, tau)

    def per_lnE():
        integrand, Tx = readme_persource(EK, E, N, T, F, tau, gamma)
        return integrand.replace(T, lambda a: Tx.subs(x, a))

    target = E * EK * weight * body
    with L.patched(r"T(x) = x\thinspace F(x)", r"T(x) = F(x)"):
        broken = per_lnE()
    return [
        exact(
            "E Q_X: the per-ln-E integrand = E E_K times the page's Q_X "
            "integrand per dE_K",
            per_lnE() - target,
        ),
        Result(
            "from E_K = E: ln E_K from ln E",
            r"\int_{\ln E}^\infty" in L.display(PERSOURCE_PIN),
            "as written",
        ),
        exact_fails("control: T(x) = F(x)", broken - target),
    ]


GRID_PIN = r"E_iQ_X(E_i) = "


def readme_grid(Ej, N, gj, tau, Tk, D):
    rel = relations(GRID_PIN)
    rhs = rel["E_iQ_X(E_i)"]
    pre, summand = rhs.split(r"\sum_s\sum_j ")
    summand = parse(
        summand,
        {r"T_{s\to X,\thinspace k}": Tk},
        {"E_j": Ej, r"\gamma_j": gj, r"\tau_s": tau},
        {"N_s": N},
    )
    return (
        (L.to_sympy(pre, {r"\Delta": D}) if pre.strip() else sp.Integer(1)),
        summand,
    )


@derivation("kd.grid")
def check_grid():
    """On the grid E_j = e^(j Delta), with x_k = E_i/E_j = e^(-k Delta)
    (k = j - i), Delta times the sum is the rectangle rule in ln E_K. This rule
    keeps the products' number and energy per decay exactly."""
    D, tau, gam = sp.symbols("Delta tau_s gamma_j", positive=True)
    i, j, kk = sp.symbols("i j k", integer=True)
    Ej, EK, E = sp.symbols("E_j E_K E", positive=True)
    N, T = sp.Function("N_s"), sp.Function("T")
    Tk = sp.Symbol("T_k")
    grid = lambda n: L.to_sympy(
        L.sides(L.inline(r"E_j = e^{j\Delta}"))[1], {"j": n, r"\Delta": D}
    )
    xk = relations(GRID_PIN)[r"x_k"]
    xk_e = L.to_sympy(xk[0], {"k": kk, r"\Delta": D})
    xk_r = L.to_sympy(xk[1], {"E_i": grid(i), "E_j": grid(j)})
    pre, summand = readme_grid(Ej, N, gam, tau, Tk, D)
    integrand, Tx = readme_persource(
        EK, E, N, T, sp.Function("F"), tau, sp.Symbol("gamma_K", positive=True)
    )
    at_node = integrand.subs(
        {EK: Ej, sp.Symbol("gamma_K", positive=True): gam}
    ).replace(T, lambda a: Tk)
    J, K = 4, 3
    w = sp.symbols(f"w0:{J}", positive=True)
    T_ = sp.symbols(f"t0:{K}", positive=True)
    xs = [sp.exp(-n * D) for n in range(K)]
    Eg = [grid(n) for n in range(-K, J)]
    src = {
        n: pre * sum(T_[jj - n] * w[jj] for jj in range(J) if 0 <= jj - n < K)
        for n in range(-K, J)
    }
    number = D * sum(src.values()) - (D * sum(T_)) * (D * sum(w))
    energy = D * sum(Eg[n + K] * src[n] for n in src) - (
        D * sum(a * b for a, b in zip(xs, T_))
    ) * (D * sum(grid(n) * w[n] for n in range(J)))
    with L.patched(
        r"E_iQ_X(E_i) = \Delta\sum_s\sum_j", r"E_iQ_X(E_i) = \sum_s\sum_j"
    ):
        b_pre = readme_grid(Ej, N, gam, tau, Tk, D)[0]
    return [
        exact(
            "x_k = e^(-k Delta) = E_i/E_j on the grid E_j = e^(j Delta) "
            "GeV, k = j - i",
            (xk_e.subs(kk, j - i) - xk_r) ** 2,
        ),
        exact(
            "the summand = the per-ln-E integrand at E_K = E_j "
            "(gamma_j, T at x_k)",
            summand - at_node,
        ),
        exact(
            "Delta in front: the rectangle rule in ln E_K on the grid of "
            "step Delta",
            pre - D,
        ),
        exact(
            "number: Delta sum_i E_i Q_i = (Delta sum_k T_k)(Delta sum_j "
            "E_j N_j/(gamma_j tau_s))",
            sp.expand(number),
        ),
        exact(
            "energy: Delta sum_i E_i E_i Q_i = (Delta sum_k x_k "
            "T_k)(Delta sum_j E_j E_j N_j/(gamma_j tau_s))",
            sp.expand(energy),
        ),
        exact_fails(
            "control: no Delta in front",
            sp.expand(
                D
                * sum(
                    b_pre
                    * sum(
                        T_[jj - n] * w[jj] for jj in range(J) if 0 <= jj - n < K
                    )
                    for n in range(-K, J)
                )
                - (D * sum(T_)) * (D * sum(w))
            ),
        ),
    ]


TABLE_PIN = r"T_k = \frac{e_k"


def readme_table(k, xs, ns, es, D):
    rel = relations(TABLE_PIN)
    tokens = {
        "x_k": xs[k],
        "x_{k+1}": xs[k + 1],
        "x_{k-1}": xs[k - 1] if k > 0 else 2 * xs[0],
        "e_k": es.get(k, 0),
        "n_k": ns.get(k, 0),
        "e_{k-1}": es.get(k - 1, 0),
        "n_{k-1}": ns.get(k - 1, 0),
        r"\Delta": D,
    }
    for name, tex in (
        ("n_q", r"\int_{x_{q+1}}^{x_q} F\thinspace dx"),
        ("e_q", r"\int_{x_{q+1}}^{x_q} xF\thinspace dx"),
    ):
        assert rel[name] == tex, rel[name]
    return L.to_sympy(rel["T_k"], tokens)


@derivation("kd.table")
def check_table():
    """T_k shares each bin's products between its two points linearly in x. The
    table keeps number and energy and has no negative entry. T_k -> x_k F(x_k)
    (the T of kd.persource) to second order in Delta."""
    D = sp.Symbol("Delta", positive=True)
    Q = 4
    xs = [sp.exp(-n * D) for n in range(Q + 2)]
    ns, es = (
        dict(zip(range(Q), sp.symbols(f"n0:{Q}", positive=True))),
        dict(zip(range(Q), sp.symbols(f"e0:{Q}", positive=True))),
    )
    T = [readme_table(k, xs, ns, es, D) for k in range(Q + 1)]
    xx, a0, a1, a2 = sp.symbols("x a0:3", positive=True)
    F = a0 + a1 * xx + a2 * xx**2
    lo_, hi_ = sp.symbols("x_lo x_hi", positive=True)
    n1, e1 = (
        sp.integrate(F, (xx, lo_, hi_)),
        sp.integrate(xx * F, (xx, lo_, hi_)),
    )
    up = sp.integrate((xx - lo_) / (hi_ - lo_) * F, (xx, lo_, hi_))
    single = readme_table(0, [hi_, lo_], {0: n1}, {0: e1}, 1)
    single_lo = readme_table(1, [hi_, lo_, lo_ / 2], {0: n1}, {0: e1}, 1)
    y = sp.Symbol("y", positive=True)
    xk = y
    pts = [xk * sp.exp(D), xk, xk * sp.exp(-D)]
    nq = {
        0: sp.integrate(F, (xx, pts[1], pts[0])),
        1: sp.integrate(F, (xx, pts[2], pts[1])),
    }
    eq_ = {
        0: sp.integrate(xx * F, (xx, pts[1], pts[0])),
        1: sp.integrate(xx * F, (xx, pts[2], pts[1])),
    }
    Tmid = readme_table(1, pts + [pts[2] * sp.exp(-D)], nq, eq_, D)
    series = sp.series(
        sp.simplify(Tmid - xk * F.subs(xx, xk)), D, 0, 2
    ).removeO()
    with L.patched(r"\frac{e_k - x_{k+1}n_k}", r"\frac{e_k - x_kn_k}"):
        broken = [readme_table(k, xs, ns, es, D) for k in range(Q + 1)]
    return [
        exact(
            "x_q's share of bin q: int (x - x_{q+1})/(x_q - x_{q+1}) F "
            "dx = the first term's (e_q - x_{q+1} n_q)/(x_q - x_{q+1})",
            single - up,
        ),
        exact(
            "x_{q+1}'s share: the rest, the second term at k = q + 1",
            single_lo - (n1 - up),
        ),
        exact(
            "Delta sum_k T_k = sum_q n_q (number per decay)",
            D * sum(T) - sum(ns.values()),
        ),
        exact(
            "Delta sum_k x_k T_k = sum_q e_q (energy per decay)",
            sp.expand(D * sum(a * b for a, b in zip(xs, T)) - sum(es.values())),
        ),
        exact(
            "no negative entry: e_q - x_{q+1} n_q = int (x - x_{q+1}) "
            "F dx >= 0, x_q n_q - e_q = int (x_q - x) F dx >= 0",
            (e1 - lo_ * n1 - sp.integrate((xx - lo_) * F, (xx, lo_, hi_))) ** 2
            + (hi_ * n1 - e1 - sp.integrate((hi_ - xx) * F, (xx, lo_, hi_)))
            ** 2,
        ),
        exact(
            "T_k - x_k F(x_k) = O(Delta^2): the table of T(x) = x F(x) "
            "per unit ln x",
            series,
        ),
        exact_fails(
            "control: e_k - x_k n_k", D * sum(broken) - sum(ns.values())
        ),
    ]


NODE_PIN = (
    r"\int_{x_a}^{x_b} \frac{M}{2p^\ast}\left(w_g + "
    r"w_G\thinspace\tilde h\right) dx = "
)
NODE_E_PIN = (
    r"\int_{x_a}^{x_b} \frac{M}{2p^\ast}\left(w_g + "
    r"w_G\thinspace\tilde h\right) x\thinspace dx = "
)
WG, WGG, XA, XB = sp.symbols("w_g w_G x_a x_b", positive=True)


def readme_node(pin, ht):
    left, right = L.sides(L.display(pin))
    head, integrand = left.split(r"\int_{x_a}^{x_b} ")
    assert head == "", left
    integrand = integrand.removesuffix(" dx").removesuffix(r"\thinspace")
    tokens = {
        "M": M,
        r"p^\ast": ps,
        r"E^\ast": Es,
        "m": m,
        "w_g": WG,
        "w_G": WGG,
        "x_a": XA,
        "x_b": XB,
        "x": x,
    }
    return (
        parse(integrand, {r"\tilde h": ht}, tokens),
        L.to_sympy(right.rstrip(".,"), tokens),
    )


@derivation("kd.node")
def check_node():
    """The bin integrals of a node's flat spectrum M/(2p*) (w_g + w_G h~), and
    of x times it, in closed form."""
    ht = readme_htilde()
    out = []
    for name, pin in (("number", NODE_PIN), ("energy", NODE_E_PIN)):
        integrand, closed = readme_node(pin, ht)
        exact_int = sp.integrate(sp.expand(integrand), (x, XA, XB))
        out.append(
            exact(
                f"{name}: int over [x_a, x_b] = the page's closed form",
                exact_int - closed,
            )
        )
    out.append(
        Result(
            "the energy integrand is x times the number's",
            sp.simplify(
                readme_node(NODE_E_PIN, ht)[0] / readme_node(NODE_PIN, ht)[0]
                - x
            )
            == 0,
            "as written",
        )
    )
    with L.patched(r"\ln\frac{x_b}{x_a}", r"\ln\frac{x_a}{x_b}"):
        integrand, closed = readme_node(NODE_PIN, ht)
        out.append(
            exact_fails(
                "control: ln(x_a/x_b)",
                sp.integrate(sp.expand(integrand), (x, XA, XB)) - closed,
            )
        )
    return out


WEIGHTS_PIN = (
    r"w_g = \tfrac12\mathcal B\thinspace g(E^\ast_n)\thinspace\omega_n"
)


def readme_weights(sign):
    rel = relations(WEIGHTS_PIN)
    Bc, om, En = sp.symbols("B omega_n E_n", positive=True)
    tokens = {r"\mathcal B": Bc, r"\omega_n": om, r"E^\ast_n": En}
    funcs = {"g": lambda a: g_, "G": lambda a: G_}
    return (
        L.to_sympy(rel["w_g"], tokens, funcs),
        L.to_sympy(rel["w_G"], tokens, funcs, sign),
        Bc,
        om,
    )


@derivation("kd.weights")
def check_weights():
    """With w_g = B g omega/2 and w_G = +-B G omega/2, the node densities are
    the quadrature of B F_R,L. Twice w_g and no w_G give B F. A two-body node
    (g omega = 1, G = P g) gives B times Lipari et al.'s spectra (kd.lipari)."""
    ht = readme_htilde()
    out = []
    node = lambda wg, wG: M / (2 * ps) * (wg + wG * ht)
    assert (
        sp.simplify(
            readme_node(NODE_PIN, ht)[0].subs({WG: g_, WGG: G_}) - node(g_, G_)
        )
        == 0
    )
    for sign, n in ((1, "R"), (-1, "L")):
        wg, wG, Bc, om = readme_weights(sign)
        out.append(
            exact(
                f"{n}: the node's density = B omega_n (M/2) "
                f"(g {'+-'[sign < 0]} G h~)/(2p*), the F_{n} integrand",
                node(wg, wG) - Bc * om * readme_F(sign, g_, G_, ht),
            )
        )
    wg, wG, Bc, om = readme_weights(1)
    out.append(
        exact(
            "another product: 2 w_g and no w_G give B omega_n (M/2) "
            "g/p*, the F integrand",
            node(2 * wg, 0) - Bc * om * readme_F(None, g_, G_, ht),
        )
    )
    r = sp.Symbol("r", positive=True)
    E0, p0 = readme_two_body()
    P = readme_P()
    htwo = readme_htilde(E0, p0)
    two = [
        (M / (2 * p0) * (w1 + w2 * htwo)).subs({g_: 1 / om, G_: P / om})
        for w1, w2 in (readme_weights(1)[:2], readme_weights(-1)[:2])
    ]
    lip = readme_lipari(r)
    out.append(
        exact(
            "two-body (g omega_n = 1, G = P g): B times the page's "
            "F_mu+R, F_mu+L of K -> mu nu",
            sum(
                (sp.simplify(t.subs(m, M * sp.sqrt(r))) - Bc * l_) ** 2
                for t, l_ in zip(two, lip)
            ),
        )
    )
    out.append(
        Result(
            "the page: twice this w_g and no w_G for the others; g omega_n = 1 "
            "and G = P g for two bodies",
            "Another product has twice this $w_g$ and no $w_G$."
            in L.line("Another product has twice")
            and (
                r"$g\thinspace\omega_n$ is 1 and $G$ is $\mathcal P g$"
                in L.line("Another product has twice")
            ),
            "as written",
        )
    )
    with L.patched(
        r"w_G = \pm\tfrac12\mathcal B\thinspace G",
        r"w_G = \pm\mathcal B\thinspace G",
    ):
        wgb, wGb, _, _ = readme_weights(1)
    out.append(
        exact_fails(
            "control: B G omega_n without the 1/2",
            node(wgb, wGb) - Bc * om * readme_F(1, g_, G_, ht),
        )
    )
    return out
