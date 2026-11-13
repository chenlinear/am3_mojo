"""Derivation checks of tests/literature_checks/annihilation/README.md
(electron–positron annihilation): from Svensson (1982, ApJ 258, 321) to the
sinks and the photon source on the code's grids. The page writes Svensson's
symbols. His equations are transcribed here in his notation from the NASA ADS
scan (the sv_ functions, with journal page and equation numbers). The page
quotes them as printed, and a check compares each quote with its transcription.
His eq. 6 (Dirac's cross section) has a typo. The argument of its logarithm is
printed as (1 + beta_cm)/(1 + beta_cm) instead of (1 + beta_cm)/(1 - beta_cm).
The checks read the page and the sources, not the code.

Run from the repository root:
  .venv/bin/python tests/literature_checks/common/run.py annihilation
"""

from re import (
    escape as re_escape,
    fullmatch as re_fullmatch,
    search as re_search,
    split as re_split,
)

import mpmath as mp
import sympy as sp
from sympy.integrals.quadrature import gauss_legendre as sympy_gauss_legendre
from sympy.integrals.rationaltools import ratint

import latex
from verify import (
    SVENSSON,
    Entry,
    Result,
    control,
    derivation,
    dirac,
    exact,
    exact_fails,
    log_uniform,
    mdot,
    must_fail,
    numeric,
    rng,
    slash,
    worst_of,
)

PAGE = "annihilation"
L = latex.page(PAGE)
UNIT = "tests/unit/test_added_processes.py"

# this module's equations in the inventory (common/inventory.py), in the order
# of the page
INVENTORY = [
    Entry(
        "ann.notation",
        r"$\sigma_T = 8\pi r_e^2/3$",
        "standard",
        "notation: Svensson 1982 (p. 322); sigma_T (PDG 2026, Table 1.1)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq6",
        r"\sigma(\gamma_{cm}) = "
        r"r_e^2\thinspace\frac{\pi}{4\beta_{cm}\gamma_{cm}^2}",
        "transcribed",
        "Svensson 1982, eq. 6 (p. 323), as printed; the misprint of "
        "its logarithm",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq1a",
        r"v_+\cdot v_- = \gamma_+\gamma_-(1 - \beta_+\beta_-\mu) = "
        r"2\gamma_{cm}^2 - 1 = \gamma_r",
        "transcribed",
        "Svensson 1982, eq. 1a (p. 322); the four-momenta",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.relations",
        r"\gamma_{cm} = \left[\frac{\gamma_r + 1}{2}\right]^{1/2}",
        "derived",
        "Svensson 1982, eq. 1a and p. 328; his L(beta) of eq. 18",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq13",
        r"\overline{v\sigma}(\gamma_+,\gamma_-) \equiv "
        r"\frac{1}{\gamma_+\gamma_-}\int_{-1}^{1}",
        "transcribed",
        "Svensson 1982, eq. 13 (p. 324)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq16",
        r"\overline{v\sigma}(\gamma_+,\gamma_-) = "
        r"\frac{1}{\beta_+\gamma_+^2\beta_-\gamma_-^2}"
        r"\int_{\gamma_{cm}^{\min}}",
        "transcribed",
        "Svensson 1982, eq. 16 (p. 324), eq. 13 in gamma_cm by eq. 1a",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq17",
        r"\gamma_{cm}^{\max} = "
        r"\left[\tfrac{1}{2}(1 + \gamma_+\gamma_- + "
        r"\gamma_+\gamma_-\beta_+\beta_-)\right]^{1/2}",
        "transcribed",
        "Svensson 1982, eqs. 17a-b (p. 324)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq18",
        r"\overline{v\sigma}(\gamma_+,\gamma_-) = "
        r"cr_e^2\thinspace\frac{\pi}"
        r"{\beta_+\gamma_+^2\beta_-\gamma_-^2}\Big[",
        "transcribed",
        "Svensson 1982, eq. 18 (p. 324), the integral of eq. 16; eq. 19a",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.gr_limits",
        r"\gamma_r^{\max} = \gamma_+\gamma_-(1 + \beta_+\beta_-)",
        "derived",
        "Svensson 1982, eqs. 1a and 17",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.integrand",
        r"2\beta_{cm}\gamma_{cm}^2\thinspace\sigma(\gamma_{cm}) = "
        r"\frac{\pi r_e^2}{1 + \gamma_r}",
        "derived",
        "Svensson 1982, eqs. 6 (corrected), 13 and 1a; the e+e- -> gamma "
        "gamma trace",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.quadrature",
        r"$= [\gamma_r^{\max}(1 - \mu) + \gamma_r^{\min}(1 + \mu)]/2$",
        "derived",
        "Svensson 1982, eqs. 1a and 13; the Gauss-Legendre rule",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.series",
        r"\frac{L(\beta_{cm})}{\beta_r\gamma_r} = 1 - \frac{\gamma_r - 1}{3}",
        "derived",
        "the series of L(beta_cm)/(beta_r gamma_r) at gamma_r = 1",
        file=PAGE,
    ),
    Entry(
        "ann.eq14",
        r"\dot n_+ = n_+n_-\langle\overline{v\sigma}\rangle",
        "transcribed",
        "Svensson 1982, eqs. 14-15 (p. 324)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.sinks",
        r"\frac{d\dot n_+}{d\gamma_+} = n_+n_-f_+(\gamma_+)",
        "rearranged",
        "Svensson 1982, eqs. 14-15: the annihilations per unit Lorentz factor",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq25",
        r"\dot n(k) = "
        r"n_+n_-\Big\langle\overline{v\frac{d\sigma}{dk}}(k)\Big\rangle",
        "transcribed",
        "Svensson 1982, eqs. 23-25 (p. 325)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.norm",
        r"\int\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)\thinspace "
        r"dk = 2\thinspace\overline{v\sigma}(\gamma_+,\gamma_-)",
        "derived",
        "Svensson 1982, eqs. 18, 21 and 55-58 (p. 331)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq30",
        r"k_{\max}k_{\min} = \gamma_{cm}^2",
        "derived",
        "Svensson 1982, eqs. 2a, 29 and 30 (p. 326)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq31",
        r"\gamma_{cm}^{\ast 2} = k\thinspace(\gamma_+ + \gamma_- - k)",
        "transcribed",
        "Svensson 1982, eq. 31 (p. 326)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq32",
        r"\gamma_{cm}^{(U)} = \min\left[\gamma_{cm}^{\max},\ "
        r"\gamma_{cm}^\ast\right]",
        "transcribed",
        "Svensson 1982, eq. 32 (p. 326)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.kLU",
        r"$k_L = \tfrac12[\gamma_+(1-\beta_+) + \gamma_-(1-\beta_-)]$",
        "transcribed",
        "Svensson 1982, eqs. 33a and 33d (p. 326)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.peaks",
        r"k_\pm = \tfrac{1}{2}\left[\gamma_\pm(1+\beta_\pm) + "
        r"\gamma_\mp(1-\beta_\mp)\right]",
        "transcribed",
        "Svensson 1982, eqs. 33b-c (the peaks, p. 331)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.k_stable",
        r"$\gamma(1 - \beta) = 1/[\gamma(1 + \beta)]$",
        "derived",
        "Svensson's gamma and beta",
        file=PAGE,
    ),
    Entry(
        "ann.eq26",
        r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-) = "
        r"\frac{1}{\beta_+\gamma_+^2\beta_-\gamma_-^2}\int",
        "transcribed",
        "Svensson 1982, eq. 26 (p. 325)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq50",
        r"\frac{d\sigma}{dk} = "
        r"r_e^2\thinspace\frac{1}{4\beta_{cm}\gamma_{cm}^3}",
        "transcribed",
        "Svensson 1982, eq. 50 (p. 330)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq51",
        r"G_\pm = \left(2\gamma_{cm}^2 + 2 - "
        r"\frac{1}{\gamma_{cm}^2}\right)\frac{1}{u_\pm}",
        "transcribed",
        "Svensson 1982, eq. 51 (p. 330)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.cdu",
        r"d_\pm = \gamma_\mp(\gamma_+ + \gamma_-) \pm k(\gamma_+ - \gamma_-)",
        "transcribed",
        "Svensson 1982, eqs. 52-54 (p. 330)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq55",
        r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-) = "
        r"cr_e^2\thinspace\frac{\pi}"
        r"{\beta_+\gamma_+^2\beta_-\gamma_-^2}\Big\lbrace",
        "transcribed",
        "Svensson 1982, eq. 55 (p. 330), the integral of eq. 26 with eq. 50; "
        "his tests of p. 331",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.H",
        r"H_\pm = \left(2 + "
        r"\frac{1 - \gamma_{cm}^{\ast 2}}{c_\pm}\right)I_\pm",
        "transcribed",
        "Svensson 1982, eq. 56 (p. 330)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.I",
        r"I_\pm = \begin{cases}",
        "transcribed",
        "Svensson 1982, eq. 57 (p. 330)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq58",
        r"H_\pm = \left(\tfrac{2}{3}\gamma_{cm}^3 + 2\gamma_{cm}",
        "transcribed",
        "Svensson 1982, eq. 58 (p. 331), the limit c -> 0 of eq. 56",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.u_zero",
        r"There the second term of $H_\pm$ tends to zero",
        "derived",
        "Svensson 1982, p. 331; eqs. 52-56",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq41",
        r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_r,1) = "
        r"cr_e^2\thinspace\frac{\pi}{\beta_r\gamma_r^2}",
        "transcribed",
        "Svensson 1982, eq. 41 (p. 328)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.rest_primitives",
        r"P(k) = cr_e^2\thinspace\frac{\pi}{\beta_r\gamma_r^2}\left["
        r"\frac{1}{k}",
        "derived",
        "Svensson 1982, eq. 41; p. 328 (the rate on a particle at rest)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.eq59",
        r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-) \approx",
        "transcribed",
        "Svensson 1982, eq. 59 (p. 331)",
        SVENSSON,
        file=PAGE,
    ),
    Entry(
        "ann.sink_grid",
        r"\Gamma_\mp(\gamma_j) = \sigma_T c\thinspace\Delta\sum_i "
        r"R_{ij}\thinspace N_\pm(\gamma_i)",
        "model",
        "the page's annihilations per unit Lorentz factor, with the "
        "particles of each bin at its grid point",
        checks=UNIT + ", tests/unit/test_fixes.py",
        file=PAGE,
    ),
    Entry(
        "ann.deposit",
        r"(1 - w)\thinspace k_l + w\thinspace k_{l+1} = \bar k",
        "model",
        "photons shared between two grid points, number and energy kept",
        checks=UNIT,
        file=PAGE,
    ),
    Entry(
        "ann.cos_map",
        r"t = \frac{1 - \cos\pi s}{2} = \sin^2\frac{\pi s}{2}",
        "standard",
        "the map of the quadrature nodes",
        file=PAGE,
    ),
    Entry(
        "ann.source_grid",
        r"Q_\gamma(k_l) = \sigma_T c\thinspace\Delta\sum_{i,j} "
        r"N_+(\gamma_i)\thinspace N_-(\gamma_j)\thinspace K_{ij}(l)",
        "model",
        "Svensson's eqs. 23-25 with the particles of each bin at its "
        "grid point",
        checks=UNIT,
        file=PAGE,
    ),
]
# entry id -> the pins of the other relations of its display equation (each a
# whole relation, as latex.parts cuts it) and of the inline relations that its
# check reads
ALSO = {
    "ann.relations": (
        r"\beta_{cm} = "
        r"\left[\frac{\gamma_r - 1}{\gamma_r + 1}\right]^{1/2}",
        r"L(\beta_{cm}) = \ln[\gamma_r(1 + \beta_r)]",
        r"$2\beta_{cm}\gamma_{cm}^2$ is $\beta_r\gamma_r$",
    ),
    "ann.eq17": (
        r"\gamma_{cm}^{\min} = \left[\tfrac{1}{2}(1 + "
        r"\gamma_+\gamma_- - "
        r"\gamma_+\gamma_-\beta_+\beta_-)\right]^{1/2}",
    ),
    "ann.eq18": (
        r"L(\beta) = \ln\frac{1 + \beta}{1 - \beta}",
        r"$4\beta_{cm}\gamma_{cm}^3\sigma(\gamma_{cm})/(\pi r_e^2)$",
    ),
    "ann.gr_limits": (
        r"\gamma_r^{\min} = \gamma_+\gamma_-(1 - "
        r"\beta_+\beta_-) = "
        r"\frac{\gamma_+^2 + \gamma_-^2 - 1}{\gamma_r^{\max}}",
        r"$\gamma_r^{\max}\gamma_r^{\min} = \gamma_+^2 + "
        r"\gamma_-^2 - 1$",
    ),
    "ann.eq14": (
        r"\langle\overline{v\sigma}\rangle \equiv "
        r"\int_1^\infty d\gamma_+",
    ),
    "ann.sinks": (r"\frac{d\dot n_+}{d\gamma_-} = n_+n_-f_-(\gamma_-)",),
    "ann.eq25": (
        r"\Big\langle\overline{v\frac{d\sigma}{dk}}(k)\Big\rangle "
        r"\equiv \int_{\gamma_+^{(L)}}",
    ),
    "ann.norm": (
        r"\int k\thinspace\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)"
        r"\thinspace dk = (\gamma_+ + "
        r"\gamma_-)\thinspace\overline{v\sigma}(\gamma_+,\gamma_-)",
    ),
    "ann.eq30": (
        r"k_{\max} + k_{\min} = \gamma_+ + \gamma_-",
        r"$\gamma_c = (\gamma_+ + \gamma_-)/(2\gamma_{cm})$",
    ),
    "ann.eq32": (r"\gamma_{cm}^{(L)} = \gamma_{cm}^{\min}",),
    "ann.kLU": (
        r"$k_U = \tfrac12[\gamma_+(1+\beta_+) + \gamma_-(1+\beta_-)]$",
    ),
    "ann.cdu": (
        r"u_\pm = (c_\pm\gamma_{cm}^2 + \gamma_{cm}^{\ast 2})^{1/2}",
        r"c_\pm = (\gamma_\mp - k)^2 - 1",
    ),
    "ann.eq55": (
        r"$dH_\pm/d\gamma_{cm} = G_\pm$",
        r"$k_T = (\gamma_+ + \gamma_-)/2$",
    ),
    "ann.u_zero": (r"$\gamma_{cm} = \gamma_{cm}^\ast$", r"$k = \gamma_\mp$"),
    "ann.eq41": (
        r"[\gamma_r(1 - \beta_r) + 1]/2 \le k \le [\gamma_r(1 + "
        r"\beta_r) + 1]/2",
    ),
    "ann.rest_primitives": (
        r"E(k) = cr_e^2\thinspace\frac{\pi}{\beta_r\gamma_r^2}\left[-(3 + "
        r"\gamma_r)(1 + \gamma_r)",
        r"$= c\beta_r\sigma(\gamma_{cm})$",
        r"$1 + \gamma_r - k_U = k_L$",
    ),
    "ann.eq59": (r"$k_+ + k_- = \gamma_+ + \gamma_{-}$",),
    "ann.sink_grid": (
        r"R_{ij} = \frac{\overline{v\sigma}(\gamma_i,\gamma_j)}{\sigma_T c}",
        r"$N_\pm(\gamma) = \gamma\thinspace n_\pm f_\pm(\gamma)$",
        r"$= -\Gamma_\mp(\gamma_j)\thinspace N_\mp(\gamma_j)$",
    ),
    "ann.deposit": (r"w = \frac{\bar k - k_l}{k_{l+1} - k_l}",),
    "ann.cos_map": (r"dt = \frac{\pi}{2}\sin(\pi s)\thinspace ds",),
    "ann.source_grid": (r"$K_{ji} = K_{ij}$", r"$\sum_l K_{ij}(l) = 2R_{ij}$"),
}

gp, gm, g, k, x, c, re = sp.symbols(
    "gamma_p gamma_m gamma k x c r_e", positive=True
)
gcm, gst = sp.symbols("gamma_cm gamma_cm_star", positive=True)
C, D, U, I_ = sp.symbols("c d u I", real=True)
cp = sp.Symbol("c_p", positive=True)  # sqrt|c|: c = cp^2 or -cp^2
w1, w2 = sp.symbols("w_1 w_2", positive=True)
UNITS = {c: 1, re: 1}  # numeric checks: c = r_e = 1


def lorentz(w):
    """A rational parametrization (see verify.zero_numerator)."""
    v = 1 + w
    return (v**2 + v**-2) / 2


GCM = (1 + w1 + 1 / (1 + w1)) / 2  # gamma_cm of gamma_r = lorentz(w1)
AB = {gp: lorentz(w1), gm: lorentz(w2)}
BRANCHES = ((0, "c > 0", cp**2), (1, "c < 0", -(cp**2)))  # the branches of I

EE, PP, ct, st = sp.symbols("E P c_theta s_theta", real=True)


def ee_gg_trace():
    """Spin and polarization sum of |M|^2/e^4 for e-(p_1) e+(p_2) -> gamma(k_1)
    gamma(k_2) in the CM frame (m = 1). The polarization sums are -g_mu nu
    (gauge invariance)."""
    gam, _, one = dirac()
    met = (1, -1, -1, -1)
    p1, p2 = (EE, 0, 0, PP), (EE, 0, 0, -PP)
    k1, k2 = (EE, EE * st, 0, EE * ct), (EE, -EE * st, 0, -EE * ct)
    d1, d2 = -2 * mdot(p1, k1), -2 * mdot(p1, k2)
    q1 = slash([a - b for a, b in zip(p1, k1)]) + one
    q2 = slash([a - b for a, b in zip(p1, k2)]) + one
    total = 0
    for m in range(4):
        for n in range(4):
            vertex = gam[n] * q1 * gam[m] * d2 + gam[m] * q2 * gam[n] * d1
            total += (
                met[m]
                * met[n]
                * sp.expand(
                    (
                        (slash(p2) - one)
                        * vertex
                        * (slash(p1) + one)
                        * gam[0]
                        * vertex.H
                        * gam[0]
                    ).trace()
                )
            )
    return sp.expand(sp.expand(total).subs(st**2, 1 - ct**2)), (d1 * d2) ** 2


def spin_averaged_M2():
    """The spin-averaged |M|^2/e^4 of e+e- -> gamma gamma in the CM frame: a
    quarter of ee_gg_trace() (ann.integrand). Here m = 1, and the photon is at
    cos theta = c to the e-."""
    pk1, pk2 = EE * (EE - PP * ct), EE * (EE + PP * ct)
    a_ = 1 / pk1 + 1 / pk2
    return 2 * (pk2 / pk1 + pk1 / pk2 + 2 * a_ - a_**2)


def cm_dsigma_domega(M2):
    """dsigma/dOmega = <|M|^2>/(64 pi^2 s) |k|/|p| in the CM frame for the
    spin-averaged |M|^2/e^4 = M2, with e^4 = 16 pi^2 r_e^2 (m = 1), s = 4E^2,
    |k| = E, |p| = P."""
    return M2 * 16 * sp.pi**2 * re**2 / (64 * sp.pi**2 * 4 * EE**2) * EE / PP


def sigma_from_trace(trace, denominator):
    """sigma(E) = (1/2) int dOmega dsigma/dOmega (1/2 for identical photons),
    with <|M|^2> = trace e^4/(4 denominator)."""
    dsdO = sp.cancel(cm_dsigma_domega(trace / (4 * denominator)))
    prim = ratint(dsdO, ct)
    return (
        sp.Rational(1, 2) * 2 * sp.pi * (prim.subs(ct, 1) - prim.subs(ct, -1))
    )


# --- Svensson (1982), transcribed from the scan (journal page numbers) ---


def sv_beta(gam):
    """The velocity of the four-velocity gamma(1, beta) (p. 322)."""
    return sp.sqrt(1 - 1 / gam**2)


def sv_L(b):
    """L(beta) (p. 324, after eq. 18)."""
    return sp.log((1 + b) / (1 - b))


def sv_eq1a(mu, a=gp, b=gm):
    """Eq. 1a (p. 322)."""
    return a * b * (1 - sv_beta(a) * sv_beta(b) * mu)


def sv_p328(gr):
    """P. 328, from eq. 1a."""
    return sp.sqrt((gr + 1) / 2), sp.sqrt((gr - 1) / (gr + 1))


def sv_eq8(gr):
    """Eq. 8 (p. 323), the rate of two beams: its factor
    [(v_+ . v_-)^2 - 1]^(1/2), with v_+ . v_- = gamma_r (eq. 1a)."""
    return sp.sqrt(gr**2 - 1)


def sv_gc(gam):
    """Eq. 2a (p. 322)."""
    return (gp + gm) / (2 * gam)


def sv_sigma(gam, printed=False):
    """Eq. 6 (p. 323). As printed, its logarithm is
    ln[(1 + beta)/(1 + beta)] = 0."""
    b = sv_beta(gam)
    lg = 0 if printed else sv_L(b)
    return (
        re**2
        * sp.pi
        / (4 * b * gam**2)
        * ((2 + 2 / gam**2 - 1 / gam**4) * lg / b - 2 - 2 / gam**2)
    )


def sv_eq13(gam, sigma):
    """Eq. 13 (p. 324): the integrand of int_-1^1 dmu with its factors."""
    return (
        1
        / (gp * gm)
        * sp.Rational(1, 2)
        * 2
        * sv_beta(gam)
        * gam**2
        * c
        * sigma(gam)
    )


def sv_eq14(f_p, f_m, rate, n_p, n_m):
    """Eqs. 14-15 (p. 324): the integrand of the double integral."""
    return n_p * n_m * f_p(gp) * f_m(gm) * rate(gp, gm)


def sv_eq16(gam, sigma):
    """Eq. 16 (p. 324): the prefactor and the integrand of the gamma_cm
    integral."""
    return (
        1 / (sv_beta(gp) * gp**2 * sv_beta(gm) * gm**2),
        4 * sv_beta(gam) * gam**3 * c * sigma(gam),
    )


def sv_eq16_integrand(gam):
    """Eq. 16 (p. 324), the integrand (eq. 6 corrected)."""
    return 4 * sv_beta(gam) * gam**3 * c * sv_sigma(gam)


def sv_eq18_bracket(gam):
    """Eq. 18 (p. 324): the bracket."""
    b = sv_beta(gam)
    return (
        b**3 * gam**2 * sv_L(b) - 2 * gam**2 + sp.Rational(3, 4) * sv_L(b) ** 2
    )


def sv_gcm(sign, a=gp, b=gm):
    """Eqs. 17a (sign +1, gamma_cm^max) and 17b (sign -1, gamma_cm^min)."""
    return sp.sqrt((1 + a * b + sign * a * b * sv_beta(a) * sv_beta(b)) / 2)


def sv_prefactor(a=gp, b=gm):
    """Eqs. 18 and 55."""
    return c * re**2 * sp.pi / (sv_beta(a) * a**2 * sv_beta(b) * b**2)


def sv_eq19a():
    """Eq. 19a (p. 325): the rate of slow pairs
    (alpha << beta_+, beta_- << 1)."""
    return c * re**2 * sp.pi


def sv_eq25(f_p, f_m, emissivity, n_p, n_m):
    """Eqs. 23-25 (p. 325): the integrand of the double integral."""
    return n_p * n_m * f_p(gp) * f_m(gm) * emissivity(k, gp, gm)


def sv_eq26(gam, dsdk):
    """Eq. 26 (p. 325): the prefactor and the integrand of the gamma_cm
    integral."""
    return (
        1 / (sv_beta(gp) * gp**2 * sv_beta(gm) * gm**2),
        4 * sv_beta(gam) * gam**3 * c * dsdk,
    )


def sv_eq29(sign, gam):
    """Eq. 29 (p. 326): k_max (sign +1) and k_min (-1)."""
    gc = sv_gc(gam)
    return gam * gc * (1 + sign * sv_beta(gc))


def sv_gstar2():
    """Eq. 31 (p. 326)."""
    return k * (gp + gm - k)


def sv_k(name, a=gp, b=gm):
    """Eqs. 33a-d (p. 326)."""
    ba, bb = sv_beta(a), sv_beta(b)
    return {
        "k_U": (a * (1 + ba) + b * (1 + bb)) / 2,
        "k_+": (a * (1 + ba) + b * (1 - bb)) / 2,
        "k_-": (a * (1 - ba) + b * (1 + bb)) / 2,
        "k_L": (a * (1 - ba) + b * (1 - bb)) / 2,
    }[name]


def sv_eq41(kk=k, gr=g):
    """Eq. 41 (p. 328), one lepton at rest, and its range."""
    br = sv_beta(gr)
    spec = (
        c
        * re**2
        * sp.pi
        / (br * gr**2)
        * (
            (-(3 + gr) / (1 + gr) + (3 + gr) / kk - 1 / kk**2)
            / (1 - kk / (1 + gr)) ** 2
            - 2
        )
    )
    return spec, (gr * (1 - br) + 1) / 2, (gr * (1 + br) + 1) / 2


def sv_eq50(gam, g_p, g_m):
    """Eq. 50 (p. 330)."""
    return (
        re**2
        / (4 * sv_beta(gam) * gam**3)
        * sp.pi
        * (-4 * gam / sp.sqrt((gp + gm) ** 2 - 4 * gam**2) + g_p + g_m)
    )


def sv_G51(gam, gs, cc, dd):
    """Eq. 51 (p. 330), with u of eq. 53."""
    u = sp.sqrt(cc * gam**2 + gs**2)
    return (2 * gam**2 + 2 - 1 / gam**2) / u + (2 * gam**2 - dd) / (2 * u**3)


def sv_cdu(sign):
    """Eqs. 52-54: c, d, u of the given sign (gamma_mp is gamma_- for +)."""
    other = gm if sign > 0 else gp
    cc = (other - k) ** 2 - 1
    dd = other * (gp + gm) + sign * k * (gp - gm)
    return cc, dd, sp.sqrt(cc * gcm**2 + gst**2)


def sv_I(cc, uu):
    """Eq. 57: the branches for c > 0 and c < 0."""
    return (
        sp.log(gcm * sp.sqrt(cc) + uu) / sp.sqrt(cc),
        sp.asin(gcm / gst * sp.sqrt(-cc)) / sp.sqrt(-cc),
    )


def sv_H(gam, gs, cc, dd, uu, ii):
    """Eq. 56."""
    return (
        (2 + (1 - gs**2) / cc) * ii
        + (1 / gam - gam / cc + gam / (2 * gs**2) * (2 * cc - dd)) / uu
        + gam / cc * uu
    )


def sv_H58(gam, gs, dd):
    """Eq. 58 (p. 331), H for c = 0."""
    return (sp.Rational(2, 3) * gam**3 + 2 * gam + 1 / gam) / gs + (
        sp.Rational(2, 3) * gam**3 - dd * gam
    ) / (2 * gs**3)


def sv_equal(gam):
    """The spectrum at k = gamma_+ = gamma_- = gamma (p. 331, item iv)."""
    b = sv_beta(gam)
    return (
        c
        * re**2
        * 2
        * sp.pi
        / (b**2 * gam**4)
        * ((gam**2 + 1) * (sp.pi / 2 - sp.asin(1 / gam)) - b / gam)
    )


def sv_eq59(rate, kk=k):
    """Eq. 59 (p. 331), k_+- of eqs. 33b-c."""
    return rate * (
        sp.DiracDelta(kk - sv_k("k_+")) + sp.DiracDelta(kk - sv_k("k_-"))
    )


# --- the page, read at each call (so that L.patched applies) ---


def _display(pin):
    return lambda: L.display(pin)


def _tex(s):
    """The page's LaTeX as latex.to_sympy reads it."""
    return s.replace(r"\lbrace", "(").replace(r"\rbrace", ")")


EQ6 = _display(
    r"\sigma(\gamma_{cm}) = "
    r"r_e^2\thinspace\frac{\pi}{4\beta_{cm}\gamma_{cm}^2}"
)
EQ1A = _display(r"v_+\cdot v_- = \gamma_+\gamma_-(1 - \beta_+\beta_-\mu)")
RELS = _display(r"\gamma_{cm} = \left[\frac{\gamma_r + 1}{2}\right]^{1/2}")
EQ13 = _display(r"\overline{v\sigma}(\gamma_+,\gamma_-) \equiv")
EQ16 = _display(r"\int_{\gamma_{cm}^{\min}}^{\gamma_{cm}^{\max}}d\gamma_{cm}")
EQ17 = _display(r"\gamma_{cm}^{\max} = \left[\tfrac{1}{2}(1 + \gamma_+\gamma_-")
EQ18 = _display(r"\overline{v\sigma}(\gamma_+,\gamma_-) = cr_e^2")
GRLIM = _display(r"\gamma_r^{\max} = \gamma_+\gamma_-(1 + \beta_+\beta_-)")
INTEG = _display(r"2\beta_{cm}\gamma_{cm}^2\thinspace\sigma(\gamma_{cm}) = ")
SERIES = _display(r"\frac{L(\beta_{cm})}{\beta_r\gamma_r} = 1")
EQ14 = _display(r"\dot n_+ = n_+n_-\langle\overline{v\sigma}\rangle")
SINKS = _display(r"\frac{d\dot n_+}{d\gamma_+} = ")
EQ25 = _display(r"\dot n(k) = n_+n_-\Big\langle")
NORM = _display(
    r"\int\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)\thinspace dk = "
)
EQ30 = _display(r"k_{\max} + k_{\min} = \gamma_+ + \gamma_-")
EQ31 = _display(r"\gamma_{cm}^{\ast 2} = k\thinspace(")
EQ32 = _display(r"\gamma_{cm}^{(U)} = ")
PEAKS = _display(r"k_\pm = \tfrac{1}{2}\left[\gamma_\pm(1+\beta_\pm)")
EQ26 = _display(r"\int_{\gamma_{cm}^{(L)}}^{\gamma_{cm}^{(U)}}d\gamma_{cm}")
EQ50 = _display(r"\frac{d\sigma}{dk} = r_e^2")
EQ51 = _display(r"G_\pm = \left(2\gamma_{cm}^2 + 2")
CDU = _display(r"d_\pm = \gamma_\mp(\gamma_+ + \gamma_-)")
EQ55 = _display(r"H_+ + H_-\Big\rbrace_ {")
HLINE = _display(
    r"H_\pm = \left(2 + \frac{1 - \gamma_{cm}^{\ast 2}}{c_\pm}\right)I_\pm"
)
ILINE = _display(r"I_\pm = \begin{cases}")
EQ58 = _display(r"H_\pm = \left(\tfrac{2}{3}\gamma_{cm}^3")
EQ41 = _display(r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_r,1) = ")
PRIM_P = _display(r"P(k) = cr_e^2")
PRIM_E = _display(r"E(k) = cr_e^2")
EQ59 = _display(r"\approx \overline{v\sigma}(\gamma_+,\gamma_-)")
SINKGRID = _display(r"\Gamma_\mp(\gamma_j) = ")
DEPOSIT = _display(r"(1 - w)\thinspace k_l + w\thinspace k_{l+1} = \bar k")
COSMAP = _display(r"dt = \frac{\pi}{2}\sin(\pi s)")
QGRID = _display(r"Q_\gamma(k_l) = ")
NODES = r"= [\gamma_r^{\max}(1 - \mu) + \gamma_r^{\min}(1 + \mu)]/2"


def relation(display, start=None):
    parts = L.parts(display())
    if start is None:
        assert len(parts) == 1, parts
        return L.sides(parts[0])
    return L.sides(L.part(display(), start))


def sigma_T():
    return L.to_sympy(
        L.sides(L.after("with the Thomson cross section $"))[1], {"r_e": re}
    )


def leptons(a=gp, b=gm):
    return {
        r"\gamma_+": a,
        r"\gamma_-": b,
        r"\gamma_{-}": b,
        r"\beta_+": sv_beta(a),
        r"\beta_-": sv_beta(b),
    }


def cm(gam):
    return {r"\gamma_{cm}": gam, r"\beta_{cm}": sv_beta(gam)}


def readme_eq6(corrected=False):
    """If corrected is true, the logarithm takes the argument that the page
    gives instead of the misprinted one."""
    lhs, rhs = relation(EQ6)
    assert lhs == r"\sigma(\gamma_{cm})", lhs
    if corrected:
        printed = r"\left(\frac{1 + \beta_{cm}}{1 + \beta_{cm}}\right)"
        assert rhs.count(printed) == 1, rhs
        rhs = rhs.replace(
            printed, r"\left(" + L.after("The argument is $") + r"\right)"
        )
    return L.to_sympy(rhs, {**cm(gcm), "r_e": re})


def page_sigma(gam):
    return readme_eq6(corrected=True).subs(gcm, gam)


def eq1a():
    s = relation(EQ1A)
    assert s[0] == r"v_+\cdot v_-" and s[3] == r"\gamma_r", s
    return s


def readme_gr_mu(mu, a=gp, b=gm):
    return L.to_sympy(eq1a()[1], {**leptons(a, b), r"\mu": mu})


def readme_relations(gr):
    tokens = {r"\gamma_r": gr, r"\beta_r": sv_beta(gr)}
    return tuple(
        L.to_sympy(relation(RELS, start)[1], tokens)
        for start in (r"\gamma_{cm}", r"\beta_{cm}", r"L(\beta_{cm})")
    )


def readme_eq17(sign, a=gp, b=gm):
    return L.to_sympy(
        relation(
            EQ17, r"\gamma_{cm}^{\max}" if sign > 0 else r"\gamma_{cm}^{\min}"
        )[1],
        leptons(a, b),
    )


def readme_L(b):
    return L.to_sympy(relation(EQ18, r"L(\beta)")[1], {r"\beta": b})


def eq18_tokens(gam):
    b = sv_beta(gam)
    return {
        r"\gamma_{cm}": gam,
        r"\beta_{cm}": b,
        r"L(\beta_{cm})": readme_L(b),
        r"L^2(\beta_{cm})": readme_L(b) ** 2,
    }


def eq18_parts():
    rhs = relation(EQ18, r"\overline{v\sigma}")[1]
    pre, rest = rhs.split(r"\Big[", 1)
    bracket, limits = rest.split(r"\Big]")
    assert limits == r"_ {\gamma_{cm}^{\min}}^{\gamma_{cm}^{\max}}", limits
    return pre, bracket


def readme_eq18():
    pre, bracket = eq18_parts()
    return (
        L.to_sympy(pre, {**leptons(), "c": c, "r_e": re}),
        lambda gam: L.to_sympy(bracket, eq18_tokens(gam)),
    )


def readme_rate(a=gp, b=gm, ends="eq. 17"):
    """The page's eq. 18 between the limits of its eq. 17 (ends "eq. 17"), or
    as the code takes it (ends "gamma_r"). Both give the same values
    (ann.relations, ann.gr_limits). The second avoids the cancellations in
    eq. 17b and in 1 - beta_cm for fast particles."""
    pre, bracket = readme_eq18()
    if ends == "gamma_r":
        tex = eq18_parts()[1]

        def at(gr):
            gam, b_cm, l_cm = readme_relations(gr)
            return L.to_sympy(
                tex,
                {
                    r"\gamma_{cm}": gam,
                    r"\beta_{cm}": b_cm,
                    r"L(\beta_{cm})": l_cm,
                    r"L^2(\beta_{cm})": l_cm**2,
                },
            )

        up, lo = (at(readme_gr_limit(w, a, b)) for w in ("max", "min"))
    else:
        up, lo = bracket(readme_eq17(1, a, b)), bracket(readme_eq17(-1, a, b))
    return pre.subs({gp: a, gm: b}) * (up - lo)


def readme_gr_limit(which, a=gp, b=gm):
    """gamma_r^min (in the last, cancellation-free form) of the page."""
    if which == "max":
        return L.to_sympy(relation(GRLIM, r"\gamma_r^{\max}")[1], leptons(a, b))
    s = relation(GRLIM, r"\gamma_r^{\min}")
    return L.to_sympy(
        s[2],
        {**leptons(a, b), r"\gamma_r^{\max}": readme_gr_limit("max", a, b)},
    )


def readme_integrand(gr):
    lhs, rhs = relation(INTEG)
    gam, b_cm, l_cm = readme_relations(gr)
    left = L.to_sympy(
        lhs,
        {r"\gamma_{cm}": gam, r"\beta_{cm}": b_cm},
        funcs={r"\sigma": page_sigma},
    )
    right = L.to_sympy(
        rhs,
        {
            r"\gamma_r": gr,
            r"\beta_r": sv_beta(gr),
            r"L(\beta_{cm})": l_cm,
            "r_e": re,
        },
    )
    return left, right


def readme_gs2(kk=k, a=gp, b=gm):
    return L.to_sympy(relation(EQ31)[1], {**leptons(a, b), "k": kk})


def readme_cdu(sign, gs=gst):
    tokens = {
        **leptons(),
        "k": k,
        r"\gamma_{cm}": gcm,
        r"\gamma_{cm}^{\ast 2}": gs**2,
    }
    cc = L.to_sympy(relation(CDU, r"c_\pm")[1], tokens, sign=sign)
    dd = L.to_sympy(relation(CDU, r"d_\pm")[1], tokens, sign=sign)
    uu = L.to_sympy(
        relation(CDU, r"u_\pm")[1], {**tokens, "c_+": cc, "c_-": cc}, sign=sign
    )
    return cc, dd, uu


def readme_I(sign, cc, uu, gs=gst):
    part = L.parts(ILINE())[0]
    assert part.startswith(r"I_\pm = \begin{cases}") and part.endswith(
        r"\end{cases}"
    ), part
    cases = L.between(part, r"\begin{cases}", r"\end{cases}")
    rows = [
        [s.strip() for s in L.split_top(row, "&")]
        for row in L.split_top(cases, r"\cr")
    ]
    assert [r[1] for r in rows] == [r"c_\pm \gt 0", r"c_\pm \lt 0"], rows
    tokens = {
        r"\gamma_{cm}": gcm,
        r"\gamma_{cm}^\ast": gs,
        "c_+": cc,
        "c_-": cc,
        "u_+": uu,
        "u_-": uu,
    }
    return tuple(L.to_sympy(r[0], tokens, sign=sign) for r in rows)


def h_tokens(sign, gam, gs, cc=C, dd=D, uu=U, ii=I_):
    s = "+" if sign > 0 else "-"
    return {
        r"\gamma_{cm}": gam,
        r"\gamma_{cm}^\ast": gs,
        r"\gamma_{cm}^{\ast 2}": gs**2,
        r"\gamma_{cm}^{\ast 3}": gs**3,
        f"c_{s}": cc,
        f"d_{s}": dd,
        f"u_{s}": uu,
        f"I_{s}": ii,
    }


def readme_H(sign, gam, gs, cc, dd, uu, ii):
    return L.to_sympy(
        relation(HLINE)[1], h_tokens(sign, gam, gs, cc, dd, uu, ii), sign=sign
    )


def readme_H58(sign, gam, gs, dd):
    return L.to_sympy(
        relation(EQ58)[1], h_tokens(sign, gam, gs, dd=dd), sign=sign
    )


def readme_G51(sign, gam, dd, uu):
    s = "+" if sign > 0 else "-"
    return L.to_sympy(
        relation(EQ51)[1],
        {r"\gamma_{cm}": gam, f"d_{s}": dd, f"u_{s}": uu},
        sign=sign,
    )


def readme_k(name, a=gp, b=gm):
    if name in ("k_L", "k_U"):
        tex = re_search(
            r"\$(" + name + r" = [^$]+)\$",
            L.line("the kinematic limits $k_L = "),
        ).group(1)
        return L.to_sympy(L.sides(tex)[1], leptons(a, b))
    return L.to_sympy(
        relation(PEAKS)[1], leptons(a, b), sign=1 if name == "k_+" else -1
    )


def readme_eq41():
    p = L.parts(EQ41())
    tokens = {"k": k, r"\gamma_r": g, r"\beta_r": sv_beta(g), "c": c, "r_e": re}
    lo, mid, hi = (s.strip() for s in re_split(r"\\le(?![A-Za-z])", p[1]))
    assert mid == "k", p[1]
    return (
        L.to_sympy(_tex(L.sides(p[0])[1]), tokens),
        L.to_sympy(lo, tokens),
        L.to_sympy(hi, tokens),
    )


def eq55_parts():
    lhs, rhs = relation(EQ55)
    pre, rest = rhs.split(r"\Big\lbrace", 1)
    brace, limits = rest.split(r"\Big\rbrace")
    assert limits == r"_ {\gamma_{cm}^{(L)}}^{\gamma_{cm}^{(U)}}", limits
    return L.to_sympy(pre, {**leptons(), "c": c, "r_e": re}), brace


def _lambdify(args, expr):
    return sp.lambdify(args, expr.subs(UNITS), "mpmath")


def spectrum_function(source):
    if source == "README":
        pre, brace = eq55_parts()
        gmax, gmin, gs = readme_eq17(1), readme_eq17(-1), sp.sqrt(readme_gs2())
        sign_c = [readme_cdu(s)[0] for s in (1, -1)]

        def F(bp, bm):
            GS = sp.Symbol("GS")
            hs = {}
            for sign, branch in ((1, bp), (-1, bm)):
                cc, dd, uu = readme_cdu(sign, gs=GS)
                hs[sign] = readme_H(
                    sign,
                    gcm,
                    GS,
                    cc,
                    dd,
                    uu,
                    readme_I(sign, cc, uu, gs=GS)[branch],
                )
            expr = L.to_sympy(
                _tex(brace),
                {**leptons(), r"\gamma_{cm}": gcm, "H_+": hs[1], "H_-": hs[-1]},
            )
            return expr.subs(GS, gs)
    else:
        pre = sv_prefactor()
        gmax, gmin, gs = sv_gcm(1), sv_gcm(-1), sp.sqrt(sv_gstar2())
        sign_c = [sv_cdu(s)[0] for s in (1, -1)]

        def F(bp, bm):
            terms = sp.sqrt((gp + gm) ** 2 - 4 * gcm**2)
            for sign, branch in ((1, bp), (-1, bm)):
                cc, dd, uu = sv_cdu(sign)
                terms += sv_H(gcm, gst, cc, dd, uu, sv_I(cc, uu)[branch])
            return terms.subs(gst, gs)

    args = (gcm, k, gp, gm)
    f = {(bp, bm): _lambdify(args, F(bp, bm)) for bp in (0, 1) for bm in (0, 1)}
    pre_f, gmax_f, gmin_f, gs_f = (
        _lambdify((k, gp, gm), e) for e in (pre, gmax, gmin, gs)
    )
    c_f = [_lambdify((k, gp, gm), e) for e in sign_c]

    def spectrum(kk, a, b):
        up, lo = min(gmax_f(kk, a, b), gs_f(kk, a, b)), gmin_f(kk, a, b)
        if up <= lo:
            return mp.mpf(0)
        branch = tuple(0 if cf(kk, a, b) > 0 else 1 for cf in c_f)
        return pre_f(kk, a, b) * (
            f[branch](up, kk, a, b) - f[branch](lo, kk, a, b)
        )

    return spectrum


def sample_points(n, seed):
    """Half of the points have k within 0.9 of gamma_-, where c_+ < 0 (the
    arcsin branch)."""
    r = rng(seed)
    kL, kU = (_lambdify((gp, gm), sv_k(n_)) for n_ in ("k_L", "k_U"))
    out = []
    while len(out) < n:
        a, b = log_uniform(r, 1.01, 1e4), log_uniform(r, 1.01, 1e4)
        lo, hi = kL(a, b), kU(a, b)
        kk = lo + (hi - lo) * mp.mpf(r.random())
        if len(out) % 2:
            kk = b + mp.mpf(0.9) * (2 * mp.mpf(r.random()) - 1)
            if not lo < kk < hi:
                continue
        out.append((kk, a, b))
    return out


@derivation("ann.notation")
def check_notation():
    """The page's sigma_T, the unit of the code's rates, makes Svensson's pi
    r_e^2 c of slow pairs (eq. 19a) 3/8 of sigma_T c."""

    def ratio():
        return sv_eq19a() / (c * sigma_T())

    with L.patched(r"$\sigma_T = 8\pi r_e^2/3$", r"$\sigma_T = 8\pi r_e^2/5$"):
        broken = ratio()
    return [
        exact(
            "pi r_e^2 c (eq. 19a) = (3/8) sigma_T c with the page's "
            "sigma_T = 8 pi r_e^2/3",
            ratio() - sp.Rational(3, 8),
        ),
        exact_fails(
            "control: sigma_T = 8 pi r_e^2/5", broken - sp.Rational(3, 8)
        ),
    ]


@derivation("ann.eq6")
def check_eq6():
    """The page quotes eq. 6 as printed, with its misprint, and the argument
    that it gives instead makes the logarithm Svensson's L(beta_cm)."""
    printed, corrected = readme_eq6(), readme_eq6(corrected=True)
    with L.patched(
        r"- 2 - \frac{2}{\gamma_{cm}^2}\right].",
        r"- 2 - \frac{3}{\gamma_{cm}^2}\right].",
    ):
        broken = readme_eq6()
    return [
        exact(
            "the quote = eq. 6 as printed (its logarithm ln[(1 + "
            "beta_cm)/(1 + beta_cm)] = 0)",
            (printed - sv_sigma(gcm, printed=True)).subs(gcm, GCM),
        ),
        Result(
            "as printed, the cross section is negative (gamma_cm = 2)",
            printed.subs(gcm, 2) < 0,
            f"sigma(2) = {sp.nsimplify(printed.subs(gcm, 2))}",
        ),
        exact(
            "with the page's argument (1 + beta_cm)/(1 - beta_cm), "
            "it is eq. 6 with L(beta_cm) of eq. 18",
            (corrected - sv_sigma(gcm)).subs(gcm, GCM),
        ),
        exact_fails(
            "control: 3/gamma_cm^2 in the quote",
            (broken - sv_sigma(gcm, printed=True)).subs(gcm, GCM),
        ),
    ]


@derivation("ann.eq1a")
def check_eq1a():
    """Eq. 1a as quoted: gamma_r is the product of the four-velocities, and each
    particle's cm Lorentz factor obeys 2 gamma_cm^2 - 1 = gamma_r."""
    a, b = lorentz(w1), lorentz(w2)
    mu = sp.Symbol("mu", real=True)
    p_a, p_b = sp.sqrt(a**2 - 1), sp.sqrt(b**2 - 1)
    pp_, pm_ = (a, 0, 0, p_a), (b, p_b * sp.sqrt(1 - mu**2), 0, p_b * mu)
    # the energy of one particle in the other's rest frame (m = 1)
    gr = mdot(pp_, pm_)
    tot = [u + v for u, v in zip(pp_, pm_)]
    # each particle's cm energy squared
    cm2 = [mdot(q, tot) ** 2 / mdot(tot, tot) for q in (pp_, pm_)]

    def gcm_of(gr_):  # gamma_cm from the quote's 2 gamma_cm^2 - 1 = gamma_r
        y = sp.Symbol("y", positive=True)
        sol = sp.solve(L.to_sympy(eq1a()[2], {r"\gamma_{cm}": y}) - gr_, y)
        assert len(sol) == 1, sol
        return sol[0]

    with L.patched(
        r"= 2\gamma_{cm}^2 - 1 = \gamma_r", r"= 2\gamma_{cm}^2 - 2 = \gamma_r"
    ):
        broken = gcm_of(gr) ** 2
    return [
        exact("the quote = eq. 1a", readme_gr_mu(mu, a, b) - sv_eq1a(mu, a, b)),
        exact(
            "gamma_+ gamma_- (1 - beta_+ beta_- mu) = p_+.p_- "
            "(four-momenta, m = 1)",
            readme_gr_mu(mu, a, b) - gr,
        ),
        exact(
            "the positron's gamma_cm^2, (p_+.P)^2/P^2 with P = p_+ + "
            "p_-, = (1 + gamma_r)/2 of 2 gamma_cm^2 - 1 = gamma_r",
            cm2[0] - gcm_of(gr) ** 2,
        ),
        exact("the electron's, the same", cm2[1] - gcm_of(gr) ** 2),
        exact_fails("control: 2 gamma_cm^2 - 2", cm2[0] - broken),
    ]


@derivation("ann.relations")
def check_relations():
    """gamma_cm and beta_cm of gamma_r follow from eq. 1a, as Svensson gives
    them (p. 328). L(beta_cm) is ln[gamma_r (1 + beta_r)]. The factor
    2 beta_cm gamma_cm^2 of eq. 13 is beta_r gamma_r, that is the factor
    [(v_+.v_-)^2 - 1]^(1/2) of eq. 8."""
    t = lorentz(w1)  # gamma_r
    gam, b_cm, l_cm = readme_relations(t)
    y = sp.Symbol("y", positive=True)
    from_1a = sp.solve(L.to_sympy(eq1a()[2], {r"\gamma_{cm}": y}) - t, y)
    assert len(from_1a) == 1, from_1a
    factor = re_search(
        r"the factor \$([^$]+)\$ is \$([^$]+)\$",
        L.line(r"and the factor $2\beta_{cm}\gamma_{cm}^2$ is $"),
    )
    f_cm = L.to_sympy(factor.group(1), cm(gam))
    f_r = L.to_sympy(factor.group(2), {r"\beta_r": sv_beta(t), r"\gamma_r": t})
    p328 = sv_p328(t)
    with L.patched(
        r"\beta_{cm} = \left[\frac{\gamma_r - 1}{\gamma_r + 1}\right]^{1/2}",
        r"\beta_{cm} = \left[\frac{\gamma_r - 1}{\gamma_r + 2}\right]^{1/2}",
    ):
        broken = readme_relations(t)[1]
    return [
        exact(
            "gamma_cm = [(gamma_r + 1)/2]^(1/2) solves 2 gamma_cm^2 - 1 "
            "= gamma_r of eq. 1a",
            gam - from_1a[0],
        ),
        exact(
            "beta_cm = [(gamma_r - 1)/(gamma_r + 1)]^(1/2) is the "
            "velocity of that gamma_cm",
            b_cm - sv_beta(gam),
        ),
        exact(
            "both as Svensson gives them (p. 328)",
            (gam - p328[0]) ** 2 + (b_cm - p328[1]) ** 2,
        ),
        exact(
            "L(beta_cm) = ln[(1 + beta_cm)/(1 - beta_cm)] (eq. 18) = "
            "ln[gamma_r (1 + beta_r)]",
            readme_L(b_cm) - l_cm,
        ),
        exact("2 beta_cm gamma_cm^2 = beta_r gamma_r", f_cm - f_r),
        exact(
            "= [(v_+.v_-)^2 - 1]^(1/2) of eq. 8 at v_+.v_- = gamma_r (eq. 1a)",
            f_r - sv_eq8(t),
        ),
        exact_fails(
            "control: (gamma_r - 1)/(gamma_r + 2)", broken - sv_beta(gam)
        ),
    ]


def eq13_parts(sigma):
    lhs, rhs = (s.strip() for s in relation(EQ13)[0].split(r"\equiv"))
    assert lhs == r"\overline{v\sigma}(\gamma_+,\gamma_-)", lhs
    head, body = rhs.split(r"\int_{-1}^{1}\frac{d\mu}{2}\thinspace ")
    tokens = {**leptons(), **cm(gcm), "c": c}
    return L.to_sympy(head, tokens), L.to_sympy(
        body, tokens, funcs={r"\sigma": sigma}
    )


@derivation("ann.eq13")
def check_eq13():
    """The page quotes eq. 13, Svensson's angle average of the rate."""
    sigma = sp.Function("sigma")

    def integrand():
        head, body = eq13_parts(sigma)
        return head * sp.Rational(1, 2) * body

    with L.patched(
        r"\thinspace 2\beta_{cm}\gamma_{cm}^2\thinspace c",
        r"\thinspace 2\beta_{cm}\gamma_{cm}^3\thinspace c",
    ):
        broken = integrand()
    return [
        exact(
            "the quote's integrand of int_-1^1 dmu/2, with its "
            "1/(gamma_+ gamma_-), = eq. 13 (any sigma)",
            (integrand() - sv_eq13(gcm, sigma)).subs({gcm: GCM, **AB}),
        ),
        exact_fails(
            "control: gamma_cm^3 in the quote",
            (broken - sv_eq13(gcm, sigma)).subs({gcm: GCM, **AB}),
        ),
    ]


def readme_eq16(sigma):
    lhs, rhs = relation(EQ16)
    assert lhs == r"\overline{v\sigma}(\gamma_+,\gamma_-)", lhs
    head, body = rhs.split(
        r"\int_{\gamma_{cm}^{\min}}^{\gamma_{cm}^{\max}}d"
        r"\gamma_{cm}\thinspace "
    )
    tokens = {**leptons(), **cm(gcm), "c": c}
    return L.to_sympy(head, tokens), L.to_sympy(
        body, tokens, funcs={r"\sigma": sigma}
    )


@derivation("ann.eq16")
def check_eq16():
    """The page quotes eq. 16, which is eq. 13 with gamma_cm for mu (eq. 1a)."""
    sigma = sp.Function("sigma")
    mu = sp.Symbol("mu", real=True)
    pre, body = readme_eq16(sigma)
    sv_pre, sv_body = sv_eq16(gcm, sigma)
    # eq. 1a: mu falls as gamma_cm rises
    mu_of = sp.solve(sp.Eq(2 * gcm**2 - 1, sv_eq1a(mu)), mu)
    assert len(mu_of) == 1, mu_of
    from13 = sv_eq13(gcm, sigma) * (-sp.diff(mu_of[0], gcm))
    with L.patched(
        r"d\gamma_{cm}\thinspace 4\beta_{cm}\gamma_{cm}^3\thinspace "
        r"c\thinspace\sigma(\gamma_{cm}).",
        r"d\gamma_{cm}\thinspace 2\beta_{cm}\gamma_{cm}^3\thinspace "
        r"c\thinspace\sigma(\gamma_{cm}).",
    ):
        broken = readme_eq16(sigma)[1]
    return [
        exact(
            "the quote's prefactor 1/(beta_+ gamma_+^2 beta_- gamma_-^2) "
            "= eq. 16's",
            (pre - sv_pre).subs(AB),
        ),
        exact(
            "the quote's integrand 4 beta_cm gamma_cm^3 c sigma = eq. 16's "
            "(any sigma)",
            (body - sv_body).subs(gcm, GCM),
        ),
        exact(
            "= eq. 13's integrand times -dmu/dgamma_cm of eq. 1a (any sigma)",
            (pre * body - from13).subs({gcm: GCM, **AB}),
        ),
        exact_fails(
            "control: 2 beta_cm gamma_cm^3", (broken - sv_body).subs(gcm, GCM)
        ),
    ]


@derivation("ann.eq17")
def check_eq17():
    """The page quotes eq. 17, and its limits are the cm Lorentz factors of
    eq. 1a at mu = -1 and 1."""
    out = []
    for sign, name, eq in ((1, "max", "17a"), (-1, "min", "17b")):
        gam_of = readme_relations(readme_gr_mu(-sign))[0]
        out += [
            exact(
                f"gamma_cm^{name} = eq. {eq}",
                (readme_eq17(sign) - sv_gcm(sign)).subs(AB),
            ),
            exact(
                f"gamma_cm^{name} = [(gamma_r + 1)/2]^(1/2) at mu = "
                f"{-sign} of eq. 1a",
                (readme_eq17(sign) - gam_of).subs(AB),
            ),
        ]
    with L.patched(
        r"\left[\tfrac{1}{2}(1 + \gamma_+\gamma_- - "
        r"\gamma_+\gamma_-\beta_+\beta_-)\right]",
        r"\left[\tfrac{1}{2}(1 + \gamma_+\gamma_- - "
        r"2\gamma_+\gamma_-\beta_+\beta_-)\right]",
    ):
        out.append(
            exact_fails(
                "control: 2 gamma_+ gamma_- beta_+ beta_- in gamma_cm^min",
                (readme_eq17(-1) - sv_gcm(-1)).subs(AB),
            )
        )
    return out


def eq13_function():
    mu = sp.Symbol("mu", real=True)
    head, body = eq13_parts(page_sigma)
    gam = readme_relations(readme_gr_mu(mu))[0]
    f = _lambdify(
        (mu, gp, gm), (head * sp.Rational(1, 2) * body).subs(gcm, gam)
    )

    def rate(a, b):
        with mp.workdps(30):
            return mp.quad(lambda m: f(m, a, b), [-1, 0, 1])

    return rate


def cold_rate(rate):
    """vsigma of cold pairs against Svensson's eq. 19a, pi r_e^2 c. rate is the
    page's eq. 18 as an mpmath function (c = r_e = 1)."""
    with mp.workdps(40):
        dev = rate(1 + mp.mpf(10) ** -16, 1 + 2 * mp.mpf(10) ** -16) / mp.pi - 1
    return Result(
        "slow particles: eq. 18 -> pi r_e^2 c, eq. 19a "
        "(gamma_+- = 1 + 1e-16, 1 + 2e-16)",
        abs(dev) < 1e-15,
        f"numeric: rel. difference {mp.nstr(dev, 3)}",
    )


@derivation("ann.eq18")
def check_eq18():
    """The page quotes eq. 18, which is the integral of eq. 16 with the
    corrected eq. 6. Eq. 18 equals eq. 13 integrated numerically, and it tends
    to eq. 19a for slow particles."""
    pre, bracket = readme_eq18()
    deriv = L.to_sympy(
        L.after(r"the derivative of the bracket in $\gamma_{cm}$ is $"),
        {**cm(gcm), "r_e": re},
        funcs={r"\sigma": page_sigma},
    )
    step = sp.diff(re**2 * sp.pi * c * bracket(gcm), gcm) - sv_eq16_integrand(
        gcm
    )
    ours = _lambdify((gp, gm), readme_rate())
    r = rng(6)
    pts = [
        (log_uniform(r, 1.001, 1e6), log_uniform(r, 1.001, 1e6))
        for _ in range(12)
    ]
    with L.patched(
        r"- 2\gamma_{cm}^2 + \tfrac{3}{4}L^2(\beta_{cm})\Big]_ {",
        r"- 3\gamma_{cm}^2 + \tfrac{3}{4}L^2(\beta_{cm})\Big]_ {",
    ):
        broken = _lambdify((gp, gm), readme_rate())
    return [
        exact(
            "the quote's prefactor = eq. 18's", (pre - sv_prefactor()).subs(AB)
        ),
        exact(
            "the quote's bracket = eq. 18's",
            (bracket(gcm) - sv_eq18_bracket(gcm)).subs(gcm, GCM),
        ),
        exact(
            "L(beta) = ln[(1 + beta)/(1 - beta)] (p. 324)",
            (readme_L(sv_beta(gcm)) - sv_L(sv_beta(gcm))).subs(gcm, GCM),
        ),
        exact(
            "d/dgamma_cm of the bracket = the page's 4 beta_cm "
            "gamma_cm^3 sigma/(pi r_e^2), eq. 6 corrected",
            (sp.diff(bracket(gcm), gcm) - deriv).subs(gcm, GCM),
        ),
        exact(
            "so c r_e^2 pi times the bracket is a primitive of "
            "eq. 16's integrand",
            step.subs(gcm, GCM),
        ),
        numeric(
            "eq. 18 between the limits of eq. 17 = eq. 13 integrated "
            "over mu (30 digits)",
            ours,
            eq13_function(),
            pts,
            dps=30,
        ),
        control(
            "control: 3 gamma_cm^2 in the quote's bracket",
            ours,
            broken,
            pts,
            dps=30,
        ),
        cold_rate(ours),
    ]


@derivation("ann.gr_limits")
def check_gr_limits():
    """gamma_r^max and gamma_r^min are eq. 1a at mu = -1 and 1,
    gamma_r^min = (gamma_+^2 + gamma_-^2 - 1)/gamma_r^max, and with eq. 1a they
    give eq. 17's limits."""
    prod = L.sides(
        L.inline(
            r"\gamma_r^{\max}\gamma_r^{\min} = "
            r"\gamma_+^2 + \gamma_-^2 - 1"
        )
    )
    grmax, grmin = readme_gr_limit("max"), readme_gr_limit("min")
    direct = L.to_sympy(relation(GRLIM, r"\gamma_r^{\min}")[1], leptons())
    with L.patched(
        r"\frac{\gamma_+^2 + \gamma_-^2 - 1}{\gamma_r^{\max}}",
        r"\frac{\gamma_+^2 + \gamma_-^2 + 1}{\gamma_r^{\max}}",
    ):
        broken = readme_gr_limit("min")
    return [
        exact(
            "gamma_r^max = eq. 1a at mu = -1, and gamma_+ gamma_- "
            "(1 - beta_+ beta_-) at mu = 1",
            (
                (grmax - readme_gr_mu(-1)) ** 2
                + (direct - readme_gr_mu(1)) ** 2
            ).subs(AB),
        ),
        exact(
            "gamma_r^max gamma_r^min = gamma_+^2 + gamma_-^2 - 1",
            (grmax * direct - L.to_sympy(prod[1], leptons())).subs(AB),
        ),
        exact(
            "(gamma_+^2 + gamma_-^2 - 1)/gamma_r^max = gamma_r^min",
            (grmin - direct).subs(AB),
        ),
        exact(
            "[(1 + gamma_r^max)/2]^(1/2) and [(1 + gamma_r^min)/2]^(1/2) "
            "= eq. 17's gamma_cm^max and gamma_cm^min",
            (
                (readme_relations(grmax)[0] - readme_eq17(1)) ** 2
                + (readme_relations(grmin)[0] - readme_eq17(-1)) ** 2
            ).subs(AB),
        ),
        exact_fails(
            "control: + 1 in the numerator", (broken - direct).subs(AB)
        ),
    ]


@derivation("ann.integrand")
def check_integrand():
    """Eq. 13's integrand in gamma_r is 2 beta_cm gamma_cm^2 times eq. 6
    corrected, and the cross section of the e+e- -> gamma gamma trace. The
    integrand tends to pi r_e^2 for slow particles (eq. 19a)."""
    trace, denominator = ee_gg_trace()
    sigma = sigma_from_trace(trace, denominator)
    # 4 <|M|^2>/e^4, x (d_1 d_2)^2
    textbook = 4 * spin_averaged_M2() * denominator
    # E = gamma_cm, P = beta_cm gamma_cm of gamma_r = lorentz(w1)
    on_shell = {EE: GCM, PP: (1 + w1 - 1 / (1 + w1)) / 2}
    t = lorentz(w1)
    left, right = readme_integrand(t)
    e = sp.Symbol("e", positive=True)
    slow = sp.limit(readme_integrand(1 + e)[1], e, 0, "+")
    with L.patched(
        r"\left[(\gamma_r^2 + 4\gamma_r + 1)\thinspace",
        r"\left[(\gamma_r^2 + 4\gamma_r - 1)\thinspace",
    ):
        broken = readme_integrand(t)[1]
    return [
        exact(
            "the right side = the left, eq. 6 corrected at the gamma_cm of "
            "gamma_r (the page's relations)",
            left - right,
        ),
        exact(
            "the trace (explicit Dirac matrices) = "
            "8 e^4 [p_1.k_2/p_1.k_1 + p_1.k_1/p_1.k_2 + "
            "2(1/p_1.k_1 + 1/p_1.k_2) - (1/p_1.k_1 + 1/p_1.k_2)^2], "
            "the spin-averaged |M|^2 x 4",
            (trace - textbook).subs(on_shell),
        ),
        exact(
            "the right side = 2 beta_cm gamma_cm^2 sigma with sigma = (1/2) "
            "int dOmega <|M|^2>/(64 pi^2 s) E/P from the trace (e^4 = 16 "
            "pi^2 r_e^2)",
            right - (2 * PP * EE * sigma).subs(on_shell),
        ),
        exact(
            "slow particles: -> pi r_e^2 as gamma_r -> 1 (eq. 19a over c)",
            slow - sv_eq19a() / c,
        ),
        exact_fails("control: gamma_r^2 + 4 gamma_r - 1", broken - left),
    ]


def gauss_legendre(old=None, new=None):
    """The page's n-node Gauss-Legendre rule in mu for eq. 13, against its
    eq. 18. The largest error must be below the page's figure. old/new: a patch
    of the page's text (a control)."""
    pin = "There the code takes the integral of eq. 13 with the "
    if old:
        with L.patched(pin + old, pin + new):
            return gauss_legendre()
    line = L.line(pin)
    assert L.line(
        r"Where $\beta_+\gamma_+\beta_-\gamma_-$ is below a hundredth of "
        r"$\gamma_r^{\max}$, the brackets at "
        "the two limits nearly cancel."
    )
    n = int(
        re_search(
            re_escape(pin) + r"(\d+)-node Gauss–Legendre rule in \$\\mu\$", line
        ).group(1)
    )
    bound = mp.mpf(
        str(
            sp.N(
                L.to_sympy(
                    re_search(
                        r"whose error is below \$([^$]+)\$ there", line
                    ).group(1),
                    {},
                ),
                20,
            )
        )
    )
    nodes = sympy_gauss_legendre(n, 70)
    gr_, mu = sp.Symbol("gamma_r", positive=True), sp.Symbol("mu", real=True)
    integrand = _lambdify((gr_,), readme_integrand(gr_)[1])
    at_node = _lambdify(
        (mu, gp, gm),
        L.to_sympy(
            L.sides(L.inline(NODES))[1],
            {
                r"\gamma_r^{\max}": readme_gr_limit("max"),
                r"\gamma_r^{\min}": readme_gr_limit("min"),
                r"\mu": mu,
            },
        ),
    )
    rate = _lambdify((gp, gm), readme_rate())
    r, worst = rng(7), mp.mpf(0)
    with mp.workdps(60):
        mus, ws = ([mp.mpf(str(v)) for v in q] for q in nodes)
        for i in range(40):
            a = mp.mpf(10) ** mp.mpf(r.uniform(0.0005, 8))
            # beta_+ gamma_+ beta_- gamma_-/gamma_r^max
            f = (
                mp.mpf(10) ** -2
                if i % 4 == 0
                else mp.mpf(10) ** mp.mpf(r.uniform(-12, -2))
            )
            # the velocity of the electron that gives this f
            bm = f * a / (mp.sqrt(a * a - 1) * (1 - f))
            if bm >= 1:
                continue
            b = 1 / mp.sqrt(1 - bm * bm)
            mean = sum(
                wq * integrand(at_node(mq, a, b)) for mq, wq in zip(mus, ws)
            ) / (2 * a * b)
            worst = worst_of((worst, abs(mean / rate(a, b) - 1)))
    return Result(
        f"the {n}-node Gauss-Legendre rule in mu for eq. 13 = eq. 18 to below "
        f"{mp.nstr(bound, 2)}",
        worst < bound,
        f"numeric: largest rel. error {mp.nstr(worst, 3)} (60 digits)",
    )


@derivation("ann.quadrature")
def check_quadrature():
    """The code's 8-node rule gives eq. 13 to the page's 1e-30 where
    beta_+ gamma_+ beta_- gamma_- is below a hundredth of gamma_r^max. At the
    nodes of the rule, gamma_r is eq. 1a's."""
    mu = sp.Symbol("mu", real=True)

    def nodes():
        # "$\gamma_r$ $= ...$": two spans
        lhs, rhs = L.sides(L.after(r"At its nodes, $\gamma_r$ $"))
        return L.to_sympy(
            rhs,
            {
                r"\gamma_r^{\max}": readme_gr_limit("max"),
                r"\gamma_r^{\min}": readme_gr_limit("min"),
                r"\mu": mu,
            },
        )

    with L.patched(
        r"[\gamma_r^{\max}(1 - \mu) + \gamma_r^{\min}(1 + \mu)]/2",
        r"[\gamma_r^{\max}(1 + \mu) + \gamma_r^{\min}(1 - \mu)]/2",
    ):
        broken = nodes()
    return [
        exact(
            "gamma_r at the nodes = eq. 1a's gamma_+ gamma_- "
            "(1 - beta_+ beta_- mu)",
            (nodes() - readme_gr_mu(mu)).subs(AB),
        ),
        gauss_legendre(),
        must_fail(
            "control: the 4-node rule", [gauss_legendre("8-node", "4-node")]
        ),
        exact_fails(
            "control: 1 + mu and 1 - mu exchanged",
            (broken - readme_gr_mu(mu)).subs(AB),
        ),
    ]


@derivation("ann.series")
def check_series():
    """The series of L(beta_cm)/(beta_r gamma_r) at gamma_r = 1, and the size of
    the first term that the code leaves out."""
    e = sp.Symbol("e", positive=True)

    def terms():
        lhs, rhs = relation(SERIES)
        assert lhs == r"\frac{L(\beta_{cm})}{\beta_r\gamma_r}" and rhs.endswith(
            r" - \ldots"
        ), (lhs, rhs)
        return L.to_sympy(rhs[: -len(r" - \ldots")], {r"\gamma_r": 1 + e})

    l_cm = readme_relations(1 + e)[2]
    ratio = l_cm / (sv_beta(1 + e) * (1 + e))
    series = sp.series(ratio, e, 0, 3).removeO()
    line = L.line(
        "the code takes the first two terms of the series of the "
        "ratio in the bracket"
    )
    near = L.to_sympy(
        re_search(r"For \$\\gamma_r\$ within \$([^$]+)\$ of 1", line).group(1),
        {},
    )
    left_out = L.to_sympy(
        re_search(
            r"the first term it leaves out is below \$([^$]+)\$",
            L.line("and the first term it leaves out is below $"),
        ).group(1),
        {},
    )
    two = sp.lambdify(
        e,
        sp.expand(terms()).coeff(e, 0) + sp.expand(terms()).coeff(e, 1) * e,
        "mpmath",
    )
    exact_ratio = sp.lambdify(e, ratio, "mpmath")
    with mp.workdps(50):
        at, left_out = (mp.mpf(str(sp.N(q, 30))) for q in (near, left_out))
        dev = abs(exact_ratio(at) - two(at))
    with L.patched(
        r"= 1 - \frac{\gamma_r - 1}{3} +", r"= 1 - \frac{\gamma_r - 1}{6} +"
    ):
        broken = terms()
    return [
        exact(
            "L(beta_cm)/(beta_r gamma_r) at gamma_r = 1 + e = 1 - e/3 + "
            "2e^2/15 + O(e^3)",
            sp.expand(terms() - series),
        ),
        Result(
            f"at gamma_r - 1 = {near}, the two terms are within "
            f"{mp.nstr(left_out, 2)}",
            dev < left_out,
            f"numeric: {mp.nstr(dev, 3)} (50 digits)",
        ),
        exact_fails("control: (gamma_r - 1)/6", sp.expand(broken - series)),
    ]


# Svensson's normalized distributions: any functions
FP, FM = sp.Function("f_+"), sp.Function("f_-")
NP, NM = sp.symbols("n_p n_m", positive=True)  # his total densities
SV = sp.Function("V")  # vsigma(gamma_+, gamma_-): any
# the densities of a line of e+ and one of e-
AN, BN = sp.symbols("A B", positive=True)


def eq14_body():
    p1, p2 = L.parts(EQ14())
    l1, r1 = L.sides(p1)
    l2, r2 = (s.strip() for s in p2.split(r"\equiv"))
    assert (
        l1 == r"\dot n_+"
        and l2 == r1.split("n_+n_-")[1] == r"\langle\overline{v\sigma}\rangle"
    ), (p1, p2)
    body = r2.replace(r"\int_1^\infty d\gamma_+\thinspace ", "", 1).replace(
        r"\int_1^\infty d\gamma_-\thinspace ", "", 1
    )
    assert (
        r2.index(r"d\gamma_+") < r2.index(r"d\gamma_-") and r"\int" not in body
    ), r2
    tokens = {
        "n_+": NP,
        "n_-": NM,
        r"\gamma_+": gp,
        r"\gamma_-": gm,
        r"\overline{v\sigma}(\gamma_+,\gamma_-)": SV(gp, gm),
    }
    return L.to_sympy(r1.split(r"\langle")[0], tokens) * L.to_sympy(
        body, tokens, {"f_+": FP, "f_-": FM}
    )


@derivation("ann.eq14")
def check_eq14():
    """The page quotes eqs. 14-15."""
    with L.patched(
        r"f_-(\gamma_-)\thinspace\overline{v\sigma}(\gamma_+,\gamma_-).",
        r"f_+(\gamma_-)\thinspace\overline{v\sigma}(\gamma_+,\gamma_-).",
    ):
        broken = eq14_body()
    theirs = sv_eq14(FP, FM, SV, NP, NM)
    return [
        exact(
            "the quote: n_dot_+ = n_+ n_- int dgamma_+ f_+ int dgamma_- "
            "f_- vsigma = eqs. 14-15",
            eq14_body() - theirs,
        ),
        exact_fails("control: f_+(gamma_-) in the quote", broken - theirs),
    ]


def readme_sinks(which):
    lhs, rhs = relation(SINKS, r"\frac{d\dot n_+}{d\gamma_" + which + "}")
    other = "-" if which == "+" else "+"
    head, body = rhs.split(r"\int_1^\infty d\gamma_" + other + r"\thinspace ")
    tokens = {
        "n_+": NP,
        "n_-": NM,
        r"\gamma_+": gp,
        r"\gamma_-": gm,
        r"\overline{v\sigma}(\gamma_+,\gamma_-)": SV(gp, gm),
    }
    funcs = {"f_+": FP, "f_-": FM}
    return L.to_sympy(head, tokens, funcs), L.to_sympy(body, tokens, funcs)


def on_lines(expr, a, b):
    """The lines are n_+ f_+ = A delta(gamma - a) and
    n_- f_- = B delta(gamma - b)."""
    return (
        expr.subs({gp: a, gm: b})
        .replace(FP, lambda z: AN / NP if z == a else 0)
        .replace(FM, lambda z: BN / NM if z == b else 0)
    )


@derivation("ann.sinks")
def check_sinks():
    """The annihilations per unit gamma_+ and per unit gamma_- are the
    integrands of eq. 14's outer integral in either order. Both integrate to
    n_dot_+, and on lines they remove one positron and one electron per
    annihilation."""
    theirs = sv_eq14(FP, FM, SV, NP, NM)
    out = []
    for which in ("+", "-"):
        head, body = readme_sinks(which)
        out.append(
            exact(
                f"d n_dot_+/d gamma_{which} times its integrand = "
                "eq. 14's double integrand (any f_+-, vsigma)",
                head * body - theirs,
            )
        )
    out.append(
        Result(
            "the page: integrated over gamma_+ or over gamma_-, "
            "both give n_dot_+",
            r"both give $\dot n_+$"
            in L.line(r"Integrated over $\gamma_+$ or over $\gamma_-$"),
            "as written (asserted)",
        )
    )
    a, b = lorentz(w1), lorentz(w2)
    removed = [on_lines(sp.Mul(*readme_sinks(w)), a, b) for w in ("+", "-")]
    out.append(
        exact(
            "lines (A at gamma_1, B at gamma_2): positrons removed = "
            "electrons removed = A B vsigma(gamma_1, gamma_2), "
            "the annihilations of eq. 14",
            sum((q - AN * BN * SV(a, b)) ** 2 for q in removed),
        )
    )
    with L.patched(
        r"\frac{d\dot n_+}{d\gamma_-} = n_+n_-f_-(\gamma_-)",
        r"\frac{d\dot n_+}{d\gamma_-} = n_+n_-f_+(\gamma_-)",
    ):
        out.append(
            exact_fails(
                "control: f_+(gamma_-) for f_-(gamma_-)",
                sp.Mul(*readme_sinks("-")) - theirs,
            )
        )
    return out


@derivation("ann.eq25")
def check_eq25():
    """The page quotes eqs. 23-25."""
    emis = sp.Function("E")  # Svensson's emissivity per pair

    def quoted():
        p1, p2 = L.parts(EQ25())
        l1, r1 = L.sides(p1)
        l2, r2 = (s.strip() for s in p2.split(r"\equiv"))
        avg = r"\Big\langle\overline{v\frac{d\sigma}{dk}}(k)\Big\rangle"
        assert l1 == r"\dot n(k)" and r1 == "n_+n_-" + avg and l2 == avg, (
            p1,
            p2,
        )
        limits = (
            r"\int_{\gamma_+^{(L)}}^{\gamma_+^{(U)}}d\gamma_+\thinspace ",
            r"\int_{\gamma_-^{(L)}}^{\gamma_-^{(U)}}d\gamma_-\thinspace ",
        )
        # eq. 37's limits, as written
        assert r2.startswith(limits[0]) and r2.count(limits[1]) == 1, r2
        body = r2.replace(limits[0], "", 1).replace(limits[1], "", 1)
        tokens = {
            r"\gamma_+": gp,
            r"\gamma_-": gm,
            r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)": emis(
                k, gp, gm
            ),
        }
        return NP * NM * L.to_sympy(body, tokens, {"f_+": FP, "f_-": FM})

    theirs = sv_eq25(FP, FM, emis, NP, NM)
    with L.patched(
        r"f_-(\gamma_-)\thinspace\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,"
        r"\gamma_-).",
        r"f_+(\gamma_-)\thinspace\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,"
        r"\gamma_-).",
    ):
        broken = quoted()
    return [
        exact(
            "the quote: n_dot(k) = n_+ n_- int dgamma_+ f_+ int dgamma_- "
            "f_- (v dsigma/dk) = eqs. 23-25",
            quoted() - theirs,
        ),
        exact_fails("control: f_+(gamma_-) in the quote", broken - theirs),
    ]


def readme_counts(a, b):
    """The photon number and energy per annihilation are the page's integrals
    of the emissivity over k, divided by vsigma(a, b)."""
    tokens = {
        r"\overline{v\sigma}(\gamma_+,\gamma_-)": 1,
        r"\gamma_+": a,
        r"\gamma_-": b,
    }
    return (
        L.to_sympy(relation(NORM, r"\int\overline")[1], tokens),
        L.to_sympy(relation(NORM, r"\int k")[1], tokens),
    )


@derivation("ann.norm")
def check_norm():
    """The quoted eq. 55 integrates to 2 vsigma photons with the energy
    (gamma_+ + gamma_-) vsigma. Each annihilation thus makes two photons with
    the energy of the pair that it removes."""
    spec = spectrum_function("README")
    rate = _lambdify((gp, gm), readme_rate(ends="gamma_r"))
    ks = {
        n: _lambdify((gp, gm), readme_k(n))
        for n in ("k_L", "k_-", "k_+", "k_U")
    }
    pairs = [
        (mp.mpf(a), mp.mpf(b))
        for a, b in (
            (1.5, 1.02),
            (3, 7),
            (20, 2.5),
            (1e3, 30),
            (1e4, 1e4 * 1.1),
        )
    ]
    # the tolerance of verify.numeric at 30 digits
    digits, tol = 30, mp.mpf(10) ** -20
    with mp.workdps(digits):
        integrals = []
        for a, b in pairs:
            nodes = sorted(f(a, b) for f in ks.values())
            integrals.append(
                (
                    mp.quad(lambda kk: spec(kk, a, b), nodes),
                    mp.quad(lambda kk: kk * spec(kk, a, b), nodes),
                )
            )

    def errors():
        factors = [_lambdify((gp, gm), q) for q in readme_counts(gp, gm)]
        with mp.workdps(digits):
            return [
                [
                    abs(q[i] / (factors[i](a, b) * rate(a, b)) - 1)
                    for (a, b), q in zip(pairs, integrals)
                ]
                for i in (0, 1)
            ]

    number, energy = errors()
    with L.patched(
        r"\thinspace dk = 2\thinspace\overline{v\sigma}",
        r"\thinspace dk = 3\thinspace\overline{v\sigma}",
    ):
        broken = errors()[0]
    figure = mp.mpf(
        str(
            sp.N(
                L.to_sympy(
                    re_search(
                        r"photon number and energy above, "
                        r"to \$([^$]+)\$ in a 30-digit quadrature",
                        L.line("in a 30-digit quadrature"),
                    )
                    .group(1)
                    .replace(r"\times", r"\cdot"),
                    {},
                ),
                20,
            )
        )
    )
    worst = worst_of(number + energy)
    a, b = lorentz(w1), lorentz(w2)
    n_photons, e_photons = (AN * BN * q * SV(a, b) for q in readme_counts(a, b))
    removed = [on_lines(sp.Mul(*readme_sinks(w)), a, b) for w in ("+", "-")]
    return [
        Result(
            what,
            worst_of(errs) <= tol,
            f"numeric: quadrature ({digits} digits) within "
            f"{mp.nstr(worst_of(errs), 3)} for 5 pairs, gamma 1.02 to 1.1e4 "
            f"(tolerance {mp.nstr(tol, 1)})",
        )
        for what, errs in (
            ("photon number = 2 vsigma (eq. 18)", number),
            ("photon energy = (gamma_+ + gamma_-) vsigma", energy),
        )
    ] + [
        Result(
            f"the page's figure {mp.nstr(figure, 2)} is the largest of "
            "these, rounded",
            mp.nstr(worst, 2) == mp.nstr(figure, 2),
            f"numeric: {mp.nstr(worst, 3)}",
        ),
        exact(
            "lines: photons made = 2 x the annihilations "
            "(the positrons removed)",
            n_photons - 2 * removed[0],
        ),
        exact(
            "their energy = gamma_1 (positrons removed) + gamma_2 "
            "(electrons removed)",
            e_photons - a * removed[0] - b * removed[1],
        ),
        Result(
            "control: 3 vsigma photons",
            max(broken) > tol,
            f"control: off by {mp.nstr(max(broken), 3)} (must fail)",
        ),
    ]


@derivation("ann.eq30")
def check_eq30():
    """k_max k_min = gamma_cm^2 and k_max + k_min = gamma_+ + gamma_- follow
    from eqs. 2a and 29 as quoted."""
    gc_tex = L.sides(
        L.inline(r"\gamma_c = (\gamma_+ + \gamma_-)/(2\gamma_{cm})")
    )[1]
    k29 = L.after("between $k_{\\min}$ and $k_{\\max}$, which are $")
    gc = L.to_sympy(gc_tex, {**leptons(), r"\gamma_{cm}": gcm})

    def energies():  # k_min (upper sign of the page's -+) and k_max
        return [
            L.to_sympy(
                k29,
                {r"\gamma_{cm}": gcm, r"\gamma_c": gc, r"\beta_c": sv_beta(gc)},
                sign=s,
            )
            for s in (1, -1)
        ]

    kmin, kmax = energies()

    def relations():
        p1, p2 = (
            relation(EQ30, r"k_{\max}k_{\min}"),
            relation(EQ30, r"k_{\max} +"),
        )
        tokens = {
            r"k_{\max}": kmax,
            r"k_{\min}": kmin,
            r"\gamma_{cm}": gcm,
            **leptons(),
        }
        return [
            L.to_sympy(s[0], tokens) - L.to_sympy(s[1], tokens)
            for s in (p1, p2)
        ]

    rel = relations()
    with L.patched(
        r"k_{\max}k_{\min} = \gamma_{cm}^2,", r"k_{\max}k_{\min} = \gamma_{cm},"
    ):
        broken = relations()[0]
    v = {gcm: sp.Symbol("q", positive=True)}
    return [
        exact("gamma_c = eq. 2a", gc - sv_gc(gcm)),
        exact(
            "k_min, k_max = eq. 29",
            (kmin - sv_eq29(-1, gcm)) ** 2 + (kmax - sv_eq29(1, gcm)) ** 2,
        ),
        exact(
            "k_max k_min = gamma_cm^2 (gamma_c^2 (1 - beta_c^2) = 1)",
            sp.simplify(rel[0].subs(v)),
        ),
        exact(
            "k_max + k_min = gamma_+ + gamma_- (2 gamma_cm gamma_c)",
            sp.simplify(rel[1].subs(v)),
        ),
        exact_fails(
            "control: k_max k_min = gamma_cm", sp.simplify(broken.subs(v))
        ),
    ]


@derivation("ann.eq31")
def check_eq31():
    """The page quotes eq. 31, and eliminating the other energy of eq. 30 gives
    eq. 31 at k = k_max or k_min."""
    kmax, kmin = sv_eq29(1, gcm), sv_eq29(-1, gcm)
    with L.patched(
        r"\gamma_{cm}^{\ast 2} = k\thinspace(\gamma_+ + \gamma_- - k)",
        r"\gamma_{cm}^{\ast 2} = k\thinspace(\gamma_+ + \gamma_- + k)",
    ):
        broken = readme_gs2()
    return [
        exact(
            "gamma_cm*^2 = k (gamma_+ + gamma_- - k) = eq. 31",
            readme_gs2() - sv_gstar2(),
        ),
        exact(
            "at k = k_max and k_min of a pair with gamma_cm, "
            "k (gamma_+ + gamma_- - k) = gamma_cm^2 (eq. 30)",
            (readme_gs2(kmax) - gcm**2) ** 2 + (readme_gs2(kmin) - gcm**2) ** 2,
        ),
        exact_fails("control: + k", broken - sv_gstar2()),
    ]


def readme_limits(kk, a, b):
    values = {
        r"\gamma_{cm}^{\max}": readme_eq17(1, a, b),
        r"\gamma_{cm}^{\min}": readme_eq17(-1, a, b),
        r"\gamma_{cm}^\ast": sp.sqrt(readme_gs2(kk, a, b)),
    }
    out = {}
    for name in ("U", "L"):
        lhs, rhs = relation(EQ32, r"\gamma_{cm}^{(" + name + ")}")
        m = re_fullmatch(r"\\(min|max)\\left\[(.+),\\ (.+)\\right\]", rhs)
        out[name] = (
            getattr(sp, m.group(1).capitalize())(
                values[m.group(2)], values[m.group(3)]
            )
            if m
            else values[rhs]
        )
    return out["L"], out["U"]


@derivation("ann.eq32")
def check_eq32():
    """The page quotes eq. 32, and its range of gamma_cm is not empty exactly
    between k_L and k_U."""

    def property_holds():
        r, ok = rng(8), True
        for _ in range(30):
            a, b = (
                sp.Float(log_uniform(r, 1.01, 1e3), 30),
                sp.Float(log_uniform(r, 1.01, 1e3), 30),
            )
            kk = sp.Float((a + b) * r.random(), 30)
            lo, hi = readme_limits(kk, a, b)
            kL, kU = (readme_k(n, a, b) for n in ("k_L", "k_U"))
            ok &= bool(hi > lo) == bool(kL < kk < kU)
        return ok

    with L.patched(
        r"\gamma_{cm}^{(U)} = \min\left[", r"\gamma_{cm}^{(U)} = \max\left["
    ):
        broken = property_holds()
    return [
        Result(
            "the quote: gamma_cm^(U) = min[gamma_cm^max, gamma_cm*], "
            "gamma_cm^(L) = gamma_cm^min (eq. 32)",
            relation(EQ32, r"\gamma_{cm}^{(U)}")[1]
            == r"\min\left[\gamma_{cm}^{\max},\ \gamma_{cm}^\ast\right]"
            and relation(EQ32, r"\gamma_{cm}^{(L)}")[1]
            == r"\gamma_{cm}^{\min}",
            "as written (asserted)",
        ),
        Result(
            "the range is not empty exactly for k_L < k < k_U "
            "(30 random pairs and k)",
            property_holds(),
            "numeric",
        ),
        Result(
            "control: max[gamma_cm^max, gamma_cm*] fails",
            not broken,
            "fails, as it must",
        ),
    ]


def k_results(names, sign):
    m, out = "max" if sign > 0 else "min", []
    for n, e in names:
        out += [
            exact(f"{n} = eq. {e}", (readme_k(n) - sv_k(n)).subs(AB)),
            exact(
                f"gamma_cm*^2({n}) = (gamma_cm^{m})^2",
                (readme_gs2(readme_k(n)) - readme_eq17(sign) ** 2).subs(AB),
            ),
        ]
    return out


@derivation("ann.kLU")
def check_kLU():
    """k_L and k_U are Svensson's eqs. 33d and 33a, where
    gamma_cm* = gamma_cm^min (eqs. 17b and 31)."""
    with L.patched(
        r"$k_L = \tfrac12[\gamma_+(1-\beta_+)",
        r"$k_L = \tfrac12[\gamma_+(1+\beta_+)",
    ):
        broken = (readme_k("k_L") - sv_k("k_L")).subs(AB)
    return k_results((("k_L", "33d"), ("k_U", "33a")), -1) + [
        exact_fails("control: gamma_+(1 + beta_+) in k_L", broken)
    ]


@derivation("ann.peaks")
def check_peaks():
    """k_+- are Svensson's eqs. 33b-c, where gamma_cm* = gamma_cm^max, and the
    spectrum peaks there with equal values."""
    out = k_results((("k_+", "33b"), ("k_-", "33c")), 1)
    spec = spectrum_function("README")
    kp, km = (_lambdify((gp, gm), readme_k(n)) for n in ("k_+", "k_-"))
    peaks, ratios = True, []
    with mp.workdps(50):
        for a_, b_ in ((2, 1.3), (10, 3), (300, 50), (5e3, 20)):
            a_, b_ = mp.mpf(a_), mp.mpf(b_)
            for kf in (kp, km):
                k0 = kf(a_, b_)
                # (each neighbor on its own: max() can drop a NaN one)
                peaks &= all(
                    spec(k0, a_, b_) > spec(k0 * (1 + d), a_, b_)
                    for d in (-(mp.mpf(10) ** -6), mp.mpf(10) ** -6)
                )
            ratios.append(
                abs(spec(kp(a_, b_), a_, b_) / spec(km(a_, b_), a_, b_) - 1)
            )
    same = worst_of(ratios)
    out.append(
        Result(
            "local maxima at k_+- with equal values (Svensson p. 331, i)",
            peaks and same < mp.mpf(10) ** -40,
            f"numeric: 4 pairs, values equal within {mp.nstr(same, 3)}",
        )
    )
    with L.patched(
        r"+ \gamma_\mp(1-\beta_\mp)\right]", r"+ \gamma_\mp(1+\beta_\mp)\right]"
    ):
        out.append(
            exact_fails(
                "control: gamma_-+(1 + beta_-+) in k_+-",
                (readme_k("k_+") - sv_k("k_+")).subs(AB),
            )
        )
    return out


@derivation("ann.k_stable")
def check_k_stable():
    """gamma(1 - beta) = 1/[gamma(1 + beta)]: the page's k_L, k_U and k_+-
    without the cancellation in 1 - beta."""
    gv = lorentz(w1)
    tokens = {r"\gamma": gv, r"\beta": sv_beta(gv)}

    def difference():
        lhs, rhs = L.sides(L.after("and $k_U$ with $"))
        return L.to_sympy(lhs, tokens) - L.to_sympy(rhs, tokens)

    with L.patched(
        r"$\gamma(1 - \beta) = 1/[\gamma(1 + \beta)]$",
        r"$\gamma(1 - \beta) = 1/[\gamma(1 - \beta)]$",
    ):
        broken = difference()
    lhs = L.sides(L.after("and $k_U$ with $"))[0]
    cancels = L.to_sympy(L.after("which avoids the cancellation in $"), tokens)
    return [
        exact(
            "gamma(1 - beta) = 1/[gamma(1 + beta)], as gamma^2 "
            "(1 - beta^2) = 1",
            difference(),
        ),
        exact(
            "the factor the page names, 1 - beta, is the one of "
            "gamma(1 - beta)",
            L.to_sympy(lhs, tokens) - gv * cancels,
        ),
        exact_fails("control: 1/[gamma(1 - beta)]", broken),
    ]


DSDK = sp.Symbol("dsdk")  # dsigma/dk(k, gamma_+, gamma_-, gamma_cm) of eq. 26


def readme_eq26():
    lhs, rhs = relation(EQ26)
    assert lhs == r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)", lhs
    head, body = rhs.split(
        r"\int_{\gamma_{cm}^{(L)}}^{\gamma_{cm}^{(U)}}d\gamma_{cm}\thinspace "
    )
    tokens = {
        **leptons(),
        **cm(gcm),
        "c": c,
        r"\frac{d\sigma}{dk}(k,\gamma_+,\gamma_-,\gamma_{cm})": DSDK,
    }
    return L.to_sympy(head, tokens), L.to_sympy(body, tokens)


@derivation("ann.eq26")
def check_eq26():
    """The page quotes eq. 26, the emissivity of a pair as an integral over
    gamma_cm."""
    pre, body = readme_eq26()
    sv_pre, sv_body = sv_eq26(gcm, DSDK)
    with L.patched(
        r"d\gamma_{cm}\thinspace 4\beta_{cm}\gamma_{cm}^3\thinspace "
        r"c\thinspace\frac",
        r"d\gamma_{cm}\thinspace 2\beta_{cm}\gamma_{cm}^3\thinspace "
        r"c\thinspace\frac",
    ):
        broken = readme_eq26()[1]
    return [
        exact(
            "prefactor 1/(beta_+ gamma_+^2 beta_- gamma_-^2) = eq. 26's",
            (pre - sv_pre).subs(AB),
        ),
        exact(
            "integrand 4 beta_cm gamma_cm^3 c dsigma/dk = eq. 26's",
            (body - sv_body).subs(gcm, GCM),
        ),
        Result(
            "the limits gamma_cm^(L), gamma_cm^(U) of eq. 32",
            True,
            "as written (asserted)",
        ),
        exact_fails(
            "control: 2 beta_cm gamma_cm^3", (broken - sv_body).subs(gcm, GCM)
        ),
    ]


def readme_eq50(gam, g_p, g_m):
    lhs, rhs = relation(EQ50)
    assert lhs == r"\frac{d\sigma}{dk}", lhs
    return L.to_sympy(
        _tex(rhs), {**leptons(), **cm(gam), "r_e": re, "G_+": g_p, "G_-": g_m}
    )


@derivation("ann.eq50")
def check_eq50():
    """The page quotes eq. 50, the cross section of a pair integrated over the
    photon's directions."""
    Gp_, Gm_ = sp.symbols("G_p G_m")
    with L.patched(
        r"\frac{-4\gamma_{cm}}{[(\gamma_+ + \gamma_-)^2",
        r"\frac{-2\gamma_{cm}}{[(\gamma_+ + \gamma_-)^2",
    ):
        broken = readme_eq50(gcm, Gp_, Gm_)
    return [
        exact(
            "the quote = eq. 50",
            (readme_eq50(gcm, Gp_, Gm_) - sv_eq50(gcm, Gp_, Gm_)).subs(
                gcm, GCM
            ),
        ),
        exact_fails(
            "control: -2 gamma_cm",
            (broken - sv_eq50(gcm, Gp_, Gm_)).subs(gcm, GCM),
        ),
    ]


@derivation("ann.eq51")
def check_eq51():
    """The page quotes eq. 51, G_+-."""
    out = []
    for sign, s in ((1, "+"), (-1, "-")):
        uu = sp.sqrt(C * gcm**2 + gst**2)
        out.append(
            exact(
                f"G_{s} = eq. 51",
                readme_G51(sign, gcm, D, uu) - sv_G51(gcm, gst, C, D),
            )
        )
    with L.patched(
        r"\tfrac{1}{2}\left(2\gamma_{cm}^2 - d_\pm\right)\frac{1}{u_\pm^3}",
        r"\tfrac{1}{2}\left(2\gamma_{cm}^2 + d_\pm\right)\frac{1}{u_\pm^3}",
    ):
        out.append(
            exact_fails(
                "control: 2 gamma_cm^2 + d",
                readme_G51(1, gcm, D, sp.sqrt(C * gcm**2 + gst**2))
                - sv_G51(gcm, gst, C, D),
            )
        )
    return out


@derivation("ann.cdu")
def check_cdu():
    """c, d, u of the page are Svensson's eqs. 52-54."""
    out = []
    for sign, s in ((1, "+"), (-1, "-")):
        ours, theirs = readme_cdu(sign), sv_cdu(sign)
        out += [
            exact(f"{n}_{s} = eq. {e}", o - t)
            for n, e, o, t in zip("cdu", (54, 52, 53), ours, theirs)
        ]
    with L.patched(
        r"\pm k(\gamma_+ - \gamma_-),\qquad",
        r"\pm k(\gamma_+ + \gamma_-),\qquad",
    ):
        out.append(
            exact_fails(
                "control: k(gamma_+ + gamma_-) in d",
                readme_cdu(1)[1] - sv_cdu(1)[1],
            )
        )
    return out


def eq55_brace(gs):
    pre, brace = eq55_parts()
    dp, dm = sp.symbols("d_p d_m", real=True)
    out = {}
    for bp, name_p, cc_p in BRANCHES:
        for bm, name_m, cc_m in BRANCHES:
            hs, gs_ = {}, {}
            for sign, branch, cc, dd in ((1, bp, cc_p, dp), (-1, bm, cc_m, dm)):
                uu = sp.sqrt(cc * gcm**2 + gs**2)
                ii = readme_I(sign, cc, uu, gs=gs)[branch]
                hs[sign] = readme_H(sign, gcm, gs, cc, dd, uu, ii)
                gs_[sign] = readme_G51(sign, gcm, dd, uu)
            ours = L.to_sympy(
                _tex(brace),
                {**leptons(), r"\gamma_{cm}": gcm, "H_+": hs[1], "H_-": hs[-1]},
            )
            out[(name_p, name_m)] = ours, gs_
    return pre, out


@derivation("ann.eq55")
def check_eq55():
    """The page quotes eq. 55, the integral of eq. 26 with eq. 50 as
    dH_+-/dgamma_cm = G_+-. Between the limits of eq. 32, the quote is
    Svensson's, and it passes his tests of p. 331."""
    dH = L.sides(L.inline(r"dH_\pm/d\gamma_{cm} = G_\pm"))
    pre, braces = eq55_brace(gst)
    pre26, body26 = readme_eq26()
    # the first term of eq. 50's brace
    first = -4 * gcm / sp.sqrt((gp + gm) ** 2 - 4 * gcm**2)
    # eq. 50 without its brace
    factor50 = sp.cancel(readme_eq50(gcm, 0, 0) / first)
    out = [
        exact(
            "the quote's prefactor c r_e^2 pi/(beta_+ gamma_+^2 beta_- "
            "gamma_-^2) = eq. 55's",
            (pre - sv_prefactor()).subs(AB),
        ),
        exact(
            "= the quoted eq. 26's prefactor x its 4 beta_cm gamma_cm^3 c "
            "x eq. 50's r_e^2 pi/(4 beta_cm gamma_cm^3)",
            (pre26 * body26.subs(DSDK, factor50) - pre).subs({gcm: GCM, **AB}),
        ),
    ]
    for (name_p, name_m), (brace, gs_) in braces.items():
        target = (
            -4 * gcm / sp.sqrt((gp + gm) ** 2 - 4 * gcm**2) + gs_[1] + gs_[-1]
        )
        out.append(
            exact(
                "d/dgamma_cm of the brace = eq. 50's brace "
                f"({name_p} for I_+, {name_m} for I_-)",
                (sp.diff(brace, gcm) - target).subs(
                    {gp: lorentz(w1), gm: lorentz(w2)}
                ),
            )
        )
    out.append(
        Result(
            "dH_+-/dgamma_cm = G_+- (ann.H)",
            dH == [r"dH_\pm/d\gamma_{cm}", r"G_\pm"],
            "as written (asserted)",
        )
    )
    ours, theirs = spectrum_function("README"), spectrum_function("Svensson")
    pts = sample_points(24, 3)
    kT = _lambdify(
        (gp, gm),
        L.to_sympy(
            L.sides(L.inline(r"k_T = (\gamma_+ + \gamma_-)/2"))[1], leptons()
        ),
    )
    spec41, lo41, hi41 = readme_eq41()
    rest = _lambdify((k, g), spec41)
    equal = _lambdify((g,), sv_equal(g))
    r = rng(4)
    at_rest = []
    for _ in range(8):
        gam = log_uniform(r, 1.01, 1e4)
        lo, hi = (
            (gam - mp.sqrt(gam**2 - 1) + 1) / 2,
            (gam + mp.sqrt(gam**2 - 1) + 1) / 2,
        )
        at_rest.append((lo + (hi - lo) * mp.mpf(r.random()), gam))
    gams = [(log_uniform(r, 1.01, 1e4),) for _ in range(8)]
    out += [
        numeric(
            "the quote between the limits of eq. 32 = Svensson's "
            "eqs. 55-57 (both branches of I)",
            ours,
            theirs,
            pts,
        ),
        numeric(
            "symmetric about k_T = (gamma_+ + gamma_-)/2 (Svensson p. 331, i)",
            ours,
            lambda kk, a_, b_: ours(2 * kT(a_, b_) - kk, a_, b_),
            pts,
        ),
        numeric(
            "symmetric in gamma_+ and gamma_- (both orders of the charges)",
            ours,
            lambda kk, a_, b_: ours(kk, b_, a_),
            pts,
        ),
        numeric(
            "gamma_- -> 1 (1 + 1e-40): the quote of eq. 41 (Svensson p. 331)",
            lambda kk, gam: ours(kk, gam, 1 + mp.mpf(10) ** -40),
            rest,
            at_rest,
            dps=90,
            tol=mp.mpf(10) ** -15,
        ),
        numeric(
            "k = gamma_+ = gamma_- (k = gamma (1 + 1e-30)): Svensson's "
            "closed form (p. 331, iv)",
            lambda gam: ours(gam * (1 + mp.mpf(10) ** -30), gam, gam),
            equal,
            gams,
            dps=80,
            tol=mp.mpf(10) ** -12,
        ),
    ]
    with L.patched(
        r"\Big\lbrace[(\gamma_+ + \gamma_-)^2 - 4\gamma_{cm}^2]^{1/2} + H_+",
        r"\Big\lbrace[(\gamma_+ + \gamma_-)^2 - 2\gamma_{cm}^2]^{1/2} + H_+",
    ):
        brace, gs_ = eq55_brace(gst)[1][("c > 0", "c > 0")]
        target = (
            -4 * gcm / sp.sqrt((gp + gm) ** 2 - 4 * gcm**2) + gs_[1] + gs_[-1]
        )
        out.append(
            exact_fails(
                "control: 2 gamma_cm^2 in the brace",
                (sp.diff(brace, gcm) - target).subs(
                    {gp: lorentz(w1), gm: lorentz(w2)}
                ),
            )
        )
    with L.patched(r"(2c_\pm - d_\pm)\right]", r"(2c_\pm + d_\pm)\right]"):
        broken = spectrum_function("README")
    out.append(control("control: 2c + d in H", ours, broken, pts))
    return out


@derivation("ann.H")
def check_H():
    """The page quotes eq. 56, H_+-, whose derivatives in gamma_cm are its G_+-
    of eq. 51."""
    out = []
    for sign, s in ((1, "+"), (-1, "-")):
        out.append(
            exact(
                f"H_{s} = eq. 56",
                readme_H(sign, gcm, gst, C, D, U, I_)
                - sv_H(gcm, gst, C, D, U, I_),
            )
        )
        for branch, name, cc in BRANCHES:
            uu = sp.sqrt(cc * gcm**2 + gst**2)
            ii = readme_I(sign, cc, uu)[branch]
            dH = sp.diff(
                readme_H(sign, gcm, gst, cc, D, uu, ii), gcm
            ) - readme_G51(sign, gcm, D, uu)
            out.append(
                exact(
                    f"dH_{s}/dgamma_cm = G_{s} of the page's eq. 51 ({name})",
                    dH,
                )
            )
    with L.patched(r"(2c_\pm - d_\pm)\right]", r"(2c_\pm + d_\pm)\right]"):
        out.append(
            exact_fails(
                "control: 2c + d in H",
                readme_H(1, gcm, gst, C, D, U, I_)
                - sv_H(gcm, gst, C, D, U, I_),
            )
        )
    return out


@derivation("ann.I")
def check_I():
    """The page quotes eq. 57, I_+- in its two branches."""
    out = []
    for sign, s in ((1, "+"), (-1, "-")):
        for branch, name, cc in BRANCHES:
            uu = sp.sqrt(cc * gcm**2 + gst**2)
            out.append(
                exact(
                    f"I_{s} ({name}) = eq. 57",
                    readme_I(sign, cc, uu)[branch] - sv_I(cc, uu)[branch],
                )
            )
    with L.patched(
        r"\ln[\gamma_{cm}(c_\pm)^{1/2} + u_\pm]",
        r"\ln[\gamma_{cm}(c_\pm)^{1/2} - u_\pm]",
    ):
        uu = sp.sqrt(cp**2 * gcm**2 + gst**2)
        out.append(
            exact_fails(
                "control: - u in the logarithm",
                readme_I(1, cp**2, uu)[0] - sv_I(cp**2, uu)[0],
            )
        )
    return out


@derivation("ann.eq58")
def check_eq58():
    """The page quotes eq. 58, which is eq. 56 at c = 0, up to a constant in
    gamma_cm (which cancels in eq. 55)."""
    out = [
        exact(
            f"H_{s} of eq. 58 = Svensson's",
            readme_H58(sign, gcm, gst, D) - sv_H58(gcm, gst, D),
        )
        for sign, s in ((1, "+"), (-1, "-"))
    ]
    out.append(
        exact(
            "d(eq. 58)/dgamma_cm = G of the page's eq. 51 at c = 0 "
            "(Svensson's own derivation; with dH/dgamma = G for every "
            "c, see ann.H, eq. 58 is the limit)",
            sp.diff(readme_H58(1, gcm, gst, D), gcm)
            - readme_G51(1, gcm, D, sp.sqrt(gst**2)),
        )
    )
    r = rng(5)
    for branch, name, sgn in ((0, "c > 0", 1), (1, "c < 0", -1)):
        cc = sgn * mp.mpf(10) ** -30
        uu = sp.sqrt(C * gcm**2 + gst**2)
        H = _lambdify(
            (gcm, gst, C, D),
            readme_H(1, gcm, gst, C, D, uu, readme_I(1, C, uu)[branch]),
        )
        H58 = _lambdify((gcm, gst, D), readme_H58(1, gcm, gst, D))
        pts = []
        for _ in range(10):
            gs_ = log_uniform(r, 1.5, 1e3)
            g1_, g2_ = (
                1 + (gs_ - 1) * mp.mpf(r.random()),
                1 + (gs_ - 1) * mp.mpf(r.random()),
            )
            pts.append((g1_, g2_, gs_, (gs_**2 + 1) * mp.mpf(r.random()) * 4))
        out.append(
            numeric(
                f"[H - eq. 58](gamma_1) = [H - eq. 58](gamma_2) at {name}, "
                "|c| = 1e-30",
                lambda g1_, g2_, gs_, dd: (
                    H(g1_, gs_, cc, dd) - H(g2_, gs_, cc, dd)
                ),
                lambda g1_, g2_, gs_, dd: H58(g1_, gs_, dd) - H58(g2_, gs_, dd),
                pts,
                dps=80,
                tol=mp.mpf(10) ** -25,
            )
        )
    with L.patched(
        r"\left(\tfrac{2}{3}\gamma_{cm}^3 + 2\gamma_{cm} +",
        r"\left(\tfrac{2}{3}\gamma_{cm}^3 + 3\gamma_{cm} +",
    ):
        out.append(
            exact_fails(
                "control: 3 gamma_cm",
                readme_H58(1, gcm, gst, D) - sv_H58(gcm, gst, D),
            )
        )
    return out


@derivation("ann.u_zero")
def check_u_zero():
    """u_+- is real and vanishes only at gamma_cm = gamma_cm* for k = gamma_-+,
    where the second term of H_+- tends to 0."""
    where = (
        L.sides(L.inline(r"\gamma_{cm} = \gamma_{cm}^\ast")),
        L.sides(L.inline(r"k = \gamma_\mp")),
    )
    # k = gamma_- for the terms of H_+
    kk = L.to_sympy(where[1][1], leptons(), sign=1)
    cc, dd, uu = readme_cdu(1)
    gs = sp.sqrt(readme_gs2(kk))  # gamma_cm* at that k

    def bracket():
        line = relation(HLINE)[1]
        tex = L.between(line, r"+ \left[", r"\right]\frac{1}{u_\pm}")
        return L.to_sympy(
            tex, h_tokens(1, gcm, gs, cc.subs(k, kk), dd.subs(k, kk)), sign=1
        )

    with L.patched(r"(2c_\pm - d_\pm)\right]", r"(2c_\pm + d_\pm)\right]"):
        broken = bracket()
    least = L.to_sympy(
        L.after(r"On the range of the integral, $c_\pm$ is at least $"), {}
    )
    return [
        Result(
            f"c_+ is at least {least} (its minimum over k, at k = gamma_-)",
            sp.minimum(cc, k) == least and cc.subs(k, gm) == least,
            f"exact: minimum {sp.minimum(cc, k)}",
        ),
        exact(
            "u_+^2 - (gamma_cm*^2 - gamma_cm^2) = (c_+ + 1) gamma_cm^2 and "
            "c_+ + 1 = (gamma_- - k)^2: u_+ is real, and zero only at "
            "gamma_cm = gamma_cm* with c_+ = -1",
            sp.expand(uu**2 - (gst**2 - gcm**2) - (gm - k) ** 2 * gcm**2),
        ),
        Result(
            "where: gamma_cm = gamma_cm*, k = gamma_mp",
            where[0] == [r"\gamma_{cm}", r"\gamma_{cm}^\ast"]
            and kk == gm
            and cc.subs(k, kk) == -1,
            "as written (asserted); c_+ = -1 there",
        ),
        exact(
            "at k = gamma_-, the bracket of the second term of H_+ is "
            "(gamma_cm*^2 - gamma_cm^2)/(gamma_cm gamma_cm*^2) = "
            "u_+^2/(gamma_cm gamma_cm*^2): the term, "
            "u_+/(gamma_cm gamma_cm*^2), tends to 0",
            bracket() - (gs**2 - gcm**2) / (gcm * gs**2),
        ),
        Result(
            "control: 2c + d in H: the bracket stays finite at gamma_cm = "
            "gamma_cm*, the term diverges",
            sp.simplify(broken.subs(gcm, gs)) != 0,
            "exact: not zero, as it must be",
        ),
    ]


@derivation("ann.eq41")
def check_eq41():
    """The page quotes eq. 41 and its range, which runs from k_L to k_U of
    eqs. 33 with gamma_- = 1."""
    spec, lo, hi = readme_eq41()
    s_spec, s_lo, s_hi = sv_eq41()
    gv = lorentz(w1)
    at_rest = {gp: g, gm: 1}
    kL, kU = (readme_k(n).subs(at_rest) for n in ("k_L", "k_U"))
    with L.patched(
        r"+ (3 + \gamma_r)/k - 1/k^2}", r"+ (3 + \gamma_r)/k - 2/k^2}"
    ):
        broken = readme_eq41()[0]
    return [
        exact("the quote = eq. 41", (spec - s_spec).subs(g, gv)),
        exact(
            "its range = eq. 41's",
            ((lo - s_lo) ** 2 + (hi - s_hi) ** 2).subs(g, gv),
        ),
        exact(
            "the range is [k_L, k_U] of eqs. 33 with gamma_- = 1",
            ((lo - kL) ** 2 + (hi - kU) ** 2).subs(g, gv),
        ),
        exact_fails("control: 2/k^2", (broken - s_spec).subs(g, gv)),
    ]


def readme_primitives(gv=g):
    tokens = {
        "k": k,
        r"\gamma_r": gv,
        r"\beta_r": sv_beta(gv),
        "c": c,
        "r_e": re,
    }
    return tuple(L.to_sympy(relation(d)[1], tokens) for d in (PRIM_P, PRIM_E))


def rest_rate():
    """The page's eq. 18 as gamma_- -> 1 is its c beta_r sigma(gamma_cm)
    (Svensson p. 328)."""
    rate = _lambdify((gp, gm), readme_rate())
    rel = L.sides(L.inline(r"= c\beta_r\sigma(\gamma_{cm})"))[1]
    rest = _lambdify(
        (g,),
        L.to_sympy(
            rel,
            {
                "c": c,
                r"\beta_r": sv_beta(g),
                r"\gamma_{cm}": readme_relations(g)[0],
            },
            funcs={r"\sigma": page_sigma},
        ),
    )
    r, worst = rng(9), mp.mpf(0)
    with mp.workdps(60):
        for _ in range(8):
            gam = log_uniform(r, 1.01, 1e6)
            worst = worst_of(
                (worst, abs(rate(gam, 1 + mp.mpf(10) ** -30) / rest(gam) - 1))
            )
    return Result(
        "eq. 18 at (gamma_r, 1 + 1e-30) = c beta_r sigma(gamma_cm), 8 gamma_r "
        "from 1.01 to 1e6",
        worst < 1e-20,
        f"numeric: largest rel. difference {mp.nstr(worst, 3)} (60 digits)",
    )


@derivation("ann.rest_primitives")
def check_rest_primitives():
    """The code's primitives P(k) and E(k) of eq. 41: their derivatives, their
    totals, 1 + gamma_r - k_U = k_L and the logarithms between the ends of a
    segment."""
    gv = lorentz(w1)
    spec, lo, hi = (q.subs(g, gv) for q in readme_eq41())
    P, E = readme_primitives(gv)
    rel = L.sides(L.inline(r"= c\beta_r\sigma(\gamma_{cm})"))[1]
    rate1 = L.to_sympy(
        rel,
        {
            "c": c,
            r"\beta_r": sv_beta(gv),
            r"\gamma_{cm}": readme_relations(gv)[0],
        },
        funcs={r"\sigma": page_sigma},
    )
    totals = re_search(
        r"they give \$([^$]+)\$ photons with the energy \$([^$]+)\$",
        L.line("Over the whole range, they give $"),
    )
    R1 = sp.Symbol("R1")
    n_tot, e_tot = (
        L.to_sympy(s, {r"\overline{v\sigma}(\gamma_r, 1)": R1, r"\gamma_r": gv})
        for s in totals.groups()
    )
    top = L.sides(L.inline(r"1 + \gamma_r - k_U = k_L"))
    # 1 + gamma_r - k_hi = q_1, 1 + gamma_r - k_lo = q_2
    GR, KH, Q1, Q2 = sp.symbols("G_r k_hi q_1 q_2", positive=True)
    term = L.to_sympy(
        L.between(L.line("terms become $"), "Their $", "$"),
        {r"\gamma_r": GR, "k": k},
    )
    ratio = L.to_sympy(
        L.after("terms become $"),
        {r"\gamma_r": GR, r"k_{\rm hi}": KH, r"k_{\rm lo}": sp.Symbol("k_lo")},
    )
    on_segment = {GR: KH + Q1 - 1, sp.Symbol("k_lo"): KH + Q1 - Q2}
    out = [
        exact("dP/dk = eq. 41 (the quote)", sp.diff(P, k) - spec),
        exact("dE/dk = k times eq. 41", sp.diff(E, k) - k * spec),
        exact(
            "P(k_U) - P(k_L) = 2 vsigma(gamma_r, 1), with vsigma(gamma_r, "
            "1) = c beta_r sigma(gamma_cm)",
            P.subs(k, hi) - P.subs(k, lo) - n_tot.subs(R1, rate1),
        ),
        exact(
            "E(k_U) - E(k_L) = (1 + gamma_r) vsigma(gamma_r, 1)",
            E.subs(k, hi) - E.subs(k, lo) - e_tot.subs(R1, rate1),
        ),
        exact(
            "1 + gamma_r - k_U = k_L (eq. 41's range)",
            L.to_sympy(top[0], {r"\gamma_r": gv, "k_U": hi})
            - L.to_sympy(top[1], {"k_L": lo}),
        ),
        exact(
            "the ln(1 + gamma_r - k) terms between k_lo and k_hi: "
            "ln[(1 + gamma_r - k_hi)/(1 + gamma_r - k_lo)]",
            (term.subs(k, KH) - term.subs(k, sp.Symbol("k_lo")) - ratio).subs(
                on_segment
            ),
        ),
        rest_rate(),
    ]
    with L.patched(
        r"\left(3 + \gamma_r - \frac{2}{1 + \gamma_r}\right)",
        r"\left(3 + \gamma_r - \frac{1}{1 + \gamma_r}\right)",
    ):
        out.append(
            exact_fails(
                "control: 3 + gamma_r - 1/(1 + gamma_r)",
                sp.diff(readme_primitives(gv)[0], k) - spec,
            )
        )
    return out


@derivation("ann.eq59")
def check_eq59():
    """The page quotes eq. 59, which keeps the number and the energy of the
    photons, as
    k_+ + k_- = gamma_+ + gamma_-."""
    R, kp, km, kk = sp.symbols("R k_p k_m kk", real=True)

    def quoted():
        lhs, rhs = (s.strip() for s in relation(EQ59)[0].split(r"\approx"))
        assert lhs == r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)", (
            lhs
        )
        return L.to_sympy(
            rhs,
            {
                r"\overline{v\sigma}(\gamma_+,\gamma_-)": R,
                "k": kk,
                "k_+": kp,
                "k_-": km,
                "k_L": sp.Symbol("k_L"),
            },
            funcs={r"\delta": sp.DiracDelta},
        )

    rhs = quoted()
    number, energy = (
        sp.integrate(f * rhs, (kk, -sp.oo, sp.oo)) for f in (1, kk)
    )
    peaks = {kp: readme_k("k_+"), km: readme_k("k_-")}
    lhs_e, rhs_e = L.sides(L.inline(r"k_+ + k_- = \gamma_+ + \gamma_{-}"))
    sum_k = L.to_sympy(
        lhs_e, {"k_+": readme_k("k_+"), "k_-": readme_k("k_-")}
    ) - L.to_sympy(rhs_e, leptons())
    n_page, e_page = readme_counts(gp, gm)
    with L.patched(
        r"[\delta(k - k_+) + \delta(k - k_-)]",
        r"[\delta(k - k_+) + \delta(k - k_L)]",
    ):
        broken = sp.integrate(kk * quoted(), (kk, -sp.oo, sp.oo)).subs(
            {**peaks, sp.Symbol("k_L"): readme_k("k_L")}
        )

    # a distribution in k on exp(-k^2): equal distributions give equal numbers
    def tested(dist):
        return sp.integrate(dist * sp.exp(-(kk**2)), (kk, -sp.oo, sp.oo))

    return [
        exact(
            "the quote at the page's k_+- = eq. 59 with eqs. 33b-c "
            "(both tested on exp(-k^2))",
            (tested(rhs.subs(peaks)) - tested(sv_eq59(R, kk))).subs(AB),
        ),
        exact(
            "photons: int dk = 2 vsigma, the page's normalisation",
            number - n_page * R,
        ),
        exact(
            "k_+ + k_- = gamma_+ + gamma_- (the page's k_+-)", sum_k.subs(AB)
        ),
        exact(
            "energy: int k dk = (k_+ + k_-) vsigma = "
            "(gamma_+ + gamma_-) vsigma",
            (energy.subs(peaks) - e_page * R).subs(AB),
        ),
        exact_fails(
            "control: delta(k - k_L) for delta(k - k_-)",
            (broken - e_page * R).subs(AB),
        ),
    ]


def grid_tokens(n):
    """The symbols of n grid points: Lorentz factors, densities per unit
    ln gamma of e+ and e-, and the grid step."""
    G_ = sp.symbols(f"gamma_1:{n + 1}", positive=True)
    return (
        G_,
        sp.symbols(f"P_1:{n + 1}", positive=True),
        sp.symbols(f"M_1:{n + 1}", positive=True),
        sp.Symbol("Delta", positive=True),
    )


@derivation("ann.sink_grid")
def check_sink_grid():
    """With the particles of each bin at its grid point, the page's
    annihilations per unit Lorentz factor become its rates Gamma_-+ per
    particle. These rates are sums over the grid, and the particles decrease
    as N_dot = -Gamma N."""
    G_, P_, M_, Dl = grid_tokens(3)
    st = sp.Symbol("sigma_T c", positive=True)
    rate = lambda a, b: SV(a, b) + SV(b, a)  # a symmetric vsigma, as eq. 18's
    per_ln = L.sides(
        L.inline(r"N_\pm(\gamma) = \gamma\thinspace n_\pm f_\pm(\gamma)")
    )
    decrease = L.sides(
        L.inline(r"= -\Gamma_\mp(\gamma_j)\thinspace N_\mp(\gamma_j)")
    )[1]
    assert r"$\dot N_\mp(\gamma_j)$ $= -\Gamma_\mp(\gamma_j)" in L.line(
        r"The particles of grid point $j$ thus decrease as $"
    )

    def gamma_page(sign, j, dens):
        head, summand = relation(SINKGRID, r"\Gamma_\mp")[1].split(r"\sum_i")
        R_rhs = relation(SINKGRID, "R_{ij}")[1]
        total = 0
        for i in range(3):
            Rij = L.to_sympy(
                R_rhs,
                {r"\gamma_i": G_[i], r"\gamma_j": G_[j], r"\sigma_T c": st},
                funcs={r"\overline{v\sigma}": rate},
            )
            total += L.to_sympy(
                summand,
                {
                    "R_{ij}": Rij,
                    r"N_+(\gamma_i)": dens[i],
                    r"N_-(\gamma_i)": dens[i],
                },
                sign=sign,
            )
        return L.to_sympy(head, {r"\sigma_T c": st, r"\Delta": Dl}) * total

    def continuous(which, j, own, other):
        """gamma_j times the page's annihilations per unit gamma_-+ at gamma_j
        (the particles removed per unit ln gamma). The other charge is on
        lines, n f = N Delta delta(gamma - gamma_i), and the own charge is at
        gamma_j, n f = N/gamma."""
        head, body = readme_sinks(which)
        own_f, other_f = (FM, FP) if which == "-" else (FP, FM)
        own_n, other_n = (NM, NP) if which == "-" else (NP, NM)
        total = 0
        for i in range(3):
            at = (
                {gm: G_[j], gp: G_[i]}
                if which == "-"
                else {gp: G_[j], gm: G_[i]}
            )
            term = (head * body).subs(at).replace(SV, rate)
            term = term.replace(
                other_f, lambda z: other[i] * Dl / other_n if z == G_[i] else 0
            )
            total += term.replace(
                own_f, lambda z: own[j] / (G_[j] * own_n) if z == G_[j] else 0
            )
        return G_[j] * total

    out = []
    for sign, which, own, other, name in (
        (1, "-", M_, P_, "e-"),
        (-1, "+", P_, M_, "e+"),
    ):
        for j in range(3):
            tokens = {
                r"\Gamma_-(\gamma_j)": gamma_page(sign, j, other),
                r"\Gamma_+(\gamma_j)": gamma_page(sign, j, other),
                r"N_-(\gamma_j)": own[j],
                r"N_+(\gamma_j)": own[j],
            }
            out.append(
                exact(
                    f"{name} at grid point {j + 1}: removed per unit "
                    "ln gamma = -N_dot = Gamma N of the page",
                    continuous(which, j, own, other)
                    + L.to_sympy(decrease, tokens, sign=sign),
                )
            )
    out.append(
        Result(
            "N = gamma n f (per unit ln gamma)",
            per_ln
            == [r"N_\pm(\gamma)", r"\gamma\thinspace n_\pm f_\pm(\gamma)"],
            "as written (asserted)",
        )
    )
    al, tt = sp.symbols("Gamma t", positive=True)
    tau = L.to_sympy(L.after("and the timescales $"), {r"\Gamma_\mp": al})
    out.append(
        exact(
            "the timescale 1/Gamma: N_dot = -Gamma N gives N(0) "
            "e^(-Gamma t), which falls by e in it",
            sp.diff(sp.exp(-al * tt), tt)
            + al * sp.exp(-al * tt)
            + (al * tau - 1) ** 2,
        )
    )
    with L.patched(
        r"\sigma_T c\thinspace\Delta\sum_i R_{ij}",
        r"\sigma_T c\thinspace\sum_i R_{ij}",
    ):
        tokens = {
            r"\Gamma_-(\gamma_j)": gamma_page(1, 0, P_),
            r"N_-(\gamma_j)": M_[0],
        }
        out.append(
            exact_fails(
                "control: no Delta",
                continuous("-", 0, M_, P_)
                + L.to_sympy(decrease, tokens, sign=1),
            )
        )
    return out


@derivation("ann.deposit")
def check_deposit():
    """The photons of a segment shared between the grid points k_l and k_l+1
    keep their number and energy."""
    kb, kl, kl1 = sp.symbols("kbar k_l k_l1", positive=True)

    def weight():
        lhs, rhs = relation(DEPOSIT, "w =")
        return L.to_sympy(rhs, {r"\bar k": kb, "k_l": kl, "k_{l+1}": kl1})

    lhs, rhs = relation(DEPOSIT, "(1 - w)")
    w = weight()
    energy = L.to_sympy(lhs, {"w": w, "k_l": kl, "k_{l+1}": kl1}) - L.to_sympy(
        rhs, {r"\bar k": kb}
    )
    with L.patched(
        r"w = \frac{\bar k - k_l}{k_{l+1} - k_l}",
        r"w = \frac{\bar k - k_l}{k_{l+1} + k_l}",
    ):
        broken = (
            L.to_sympy(lhs, {"w": weight(), "k_l": kl, "k_{l+1}": kl1}) - kb
        )
    return [
        exact(
            "(1 - w) k_l + w k_l+1 = kbar with the page's w: the energy kept",
            energy,
        ),
        Result(
            "number: (1 - w) + w = 1; w = 0 at kbar = k_l, 1 at k_l+1",
            w.subs(kb, kl) == 0 and sp.simplify(w.subs(kb, kl1)) == 1,
            "exact",
        ),
        exact_fails("control: k_l+1 + k_l in w", broken),
    ]


@derivation("ann.cos_map")
def check_cos_map():
    """The map t = (1 - cos pi s)/2 = sin^2(pi s/2), dt = (pi/2) sin(pi s) ds,
    turns sqrt(t) into sin(pi s/2)."""
    s = sp.Symbol("s", real=True)

    def forms():
        sides_ = relation(COSMAP, "t =")
        dl, dr = relation(COSMAP, "dt")
        assert (
            sides_[0] == "t" and dl == "dt" and dr.endswith(r"\thinspace ds")
        ), (sides_, dl, dr)
        return [L.to_sympy(q, {"s": s}) for q in sides_[1:]] + [
            L.to_sympy(dr[: -len(r"\thinspace ds")], {"s": s})
        ]

    a, b, dt = forms()
    smooth = L.to_sympy(
        L.after(
            r"The map smooths a square-root edge at an "
            r"end: $\sqrt{t}$ becomes $"
        ),
        {"s": s},
    )
    with L.patched(
        r"t = \frac{1 - \cos\pi s}{2}", r"t = \frac{1 - \cos 2\pi s}{2}"
    ):
        broken = forms()
    vals = [sp.Rational(i, 8) for i in range(9)]
    return [
        Result(
            "(1 - cos pi s)/2 = sin^2(pi s/2)",
            sp.simplify(a - b) == 0,
            "exact (sympy simplify)",
        ),
        Result(
            "dt/ds = (pi/2) sin(pi s)",
            sp.simplify(sp.diff(a, s) - dt) == 0,
            "exact",
        ),
        Result(
            "sqrt(t) = sin(pi s/2) on [0, 1] (s = 0, 1/8, ..., 1)",
            all(
                sp.simplify(sp.sqrt(b.subs(s, v)) - smooth.subs(s, v)) == 0
                for v in vals
            ),
            "exact",
        ),
        Result(
            "t runs from 0 to 1",
            a.subs(s, 0) == 0 and a.subs(s, 1) == 1,
            "exact",
        ),
        Result(
            "control: cos 2 pi s",
            sp.simplify(broken[0] - broken[1]) != 0,
            "fails, as it must",
        ),
    ]


@derivation("ann.source_grid")
def check_source_grid():
    """With the particles of each bin at its grid point, Svensson's eqs. 23-25
    become the page's photon source, a sum over the pairs of grid points. The
    kernel serves both charge orders as K_ji = K_ij, and its photons are 2
    R_ij."""
    G_, P_, M_, Dl = grid_tokens(3)
    st = sp.Symbol("sigma_T c", positive=True)
    Ksym = [
        [sp.Symbol(f"K_{min(i, j)}{max(i, j)}") for j in range(3)]
        for i in range(3)
    ]  # K_ji = K_ij
    emis = sp.Function("E")
    # eq. 25 as quoted, with lines
    # n_+ f_+ = sum_i N_+i Delta delta(gamma - gamma_i) and
    # n_- f_- = sum_j N_-j Delta delta(gamma - gamma_j), per unit ln k averaged
    # over bin l: (1/Delta) int_bin ... dk, with
    # int_bin E dk = sigma_T c K_ij(l)
    p1, p2 = L.parts(EQ25())
    r2 = p2.split(r"\equiv")[1].strip()
    body = r2.replace(
        r"\int_{\gamma_+^{(L)}}^{\gamma_+^{(U)}}d\gamma_+\thinspace ", "", 1
    ).replace(
        r"\int_{\gamma_-^{(L)}}^{\gamma_-^{(U)}}d\gamma_-\thinspace ", "", 1
    )
    tokens = {
        r"\gamma_+": gp,
        r"\gamma_-": gm,
        r"\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)": emis(k, gp, gm),
    }
    integrand = NP * NM * L.to_sympy(body, tokens, {"f_+": FP, "f_-": FM})
    cont = 0
    for i in range(3):
        for j in range(3):
            term = (
                integrand.subs({gp: G_[i], gm: G_[j]})
                .replace(FP, lambda z: P_[i] * Dl / NP if z == G_[i] else 0)
                .replace(FM, lambda z: M_[j] * Dl / NM if z == G_[j] else 0)
            )
            cont += term.replace(
                emis, lambda kk, a, b: st * Ksym[G_.index(a)][G_.index(b)]
            )
    cont /= Dl

    def discrete():
        head, summand = relation(QGRID)[1].split(r"\sum_{i,j}")
        total = 0
        for i in range(3):
            for j in range(3):
                total += L.to_sympy(
                    summand,
                    {
                        r"N_+(\gamma_i)": P_[i],
                        r"N_-(\gamma_j)": M_[j],
                        "K_{ij}(l)": Ksym[i][j],
                    },
                )
        return L.to_sympy(head, {r"\sigma_T c": st, r"\Delta": Dl}) * total

    half = L.to_sympy(
        L.between(
            L.line("whose logarithm lies within $"),
            "whose logarithm lies within $",
            "$",
        ),
        {r"\Delta": Dl},
    )
    sym = L.sides(L.inline(r"K_{ji} = K_{ij}"))
    norm = L.sides(L.inline(r"\sum_l K_{ij}(l) = 2R_{ij}"))
    assert norm[0] == r"\sum_l K_{ij}(l)", norm
    a, b = lorentz(w1), lorentz(w2)
    # int E dk of the page's normalization
    number = readme_counts(a, b)[0] * readme_rate(a, b)
    R_rhs = relation(SINKGRID, "R_{ij}")[1]
    Rab = L.to_sympy(
        R_rhs,
        {r"\gamma_i": a, r"\gamma_j": b, r"\sigma_T c": st},
        funcs={r"\overline{v\sigma}": lambda u, v: readme_rate(u, v)},
    )
    out = [
        exact(
            "eq. 25 with lines, per unit ln k averaged over bin l = the "
            "page's sum over i, j",
            cont - discrete(),
        ),
        exact(
            "bins of half-width Delta/2 around grid points Delta apart in "
            "ln k touch: they tile the photon energies",
            2 * half - Dl,
        ),
        Result(
            "K_ji = K_ij: eq. 55 is symmetric in gamma_+ and gamma_- "
            "(ann.eq55)",
            sym == ["K_{ji}", "K_{ij}"],
            "as written (asserted)",
        ),
        exact(
            "sum_l K_ij(l) sigma_T c = int E dk = 2 vsigma = 2 R_ij "
            "sigma_T c (the page's R_ij)",
            number / st - L.to_sympy(norm[1], {"R_{ij}": Rab}),
        ),
    ]
    with L.patched(
        r"Q_\gamma(k_l) = \sigma_T c\thinspace\Delta\sum_{i,j}",
        r"Q_\gamma(k_l) = \sigma_T c\thinspace\sum_{i,j}",
    ):
        out.append(exact_fails("control: no Delta", cont - discrete()))
    return out
