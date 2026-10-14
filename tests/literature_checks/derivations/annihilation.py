"""Derivation checks of the e+e- annihilation equations of README.md (Electron–positron annihilation) against
Svensson (1982, ApJ 258, 321), transcribed below in his notation from the NASA ADS scan (journal page numbers). His
gamma_cm and gamma_cm* are the README's gamma_ast and gamma_s; beta = (1 - 1/gamma^2)^(1/2). His eq. 6, Dirac's cross
section, prints the argument of its logarithm as (1 + beta_cm)/(1 + beta_cm), which makes the cross section negative.
With (1 + beta_cm)/(1 - beta_cm), the L(beta) of his eq. 18, it is the README's sigma_rest.

The checks also redo Svensson's steps where they are exact, and test the README's equations against each other.
Without a paper, they derive sigma_rest from the Dirac trace of the two diagrams of e+e- -> gamma gamma, <sigma v> as
its Moller angle average, gamma_ast from the four-momenta, and the kinetic terms as counts of the annihilations.
Berestetskii, Lifshitz & Pitaevskii (Quantum Electrodynamics, §89) give the three-photon fraction. The other codes'
treatments (Results of tests/literature_checks/README.md) are recomputed against the README's rate: ATHEnA's rate fit
(Mastichiadis & Kirk 1995, eq. 59) and Boettcher & Schlickeiser (1996, eqs. 4-12, 16 and 33, from the TeX source of
arXiv:astro-ph/9506112v1).

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/annihilation.py
"""
from re import search as re_search, split as re_split

import mpmath as mp
import sympy as sp
from sympy.integrals.rationaltools import ratint

import latex as L
from latex import BS96
from pdg import INV_ALPHA
from verify import Result, control, derivation, exact, exact_fails, in_source, log_uniform, mdot, numeric, rng, run

gp, gm, g, k, x, c, re = sp.symbols("gamma_p gamma_m gamma k x c r_e", positive=True)
gcm, gst = sp.symbols("gamma_cm gamma_cm_star", positive=True)
C, D, U, I_ = sp.symbols("c d u I", real=True)
cp = sp.Symbol("c_p", positive=True)   # |c|: c = cp^2 or -cp^2
w1, w2 = sp.symbols("w_1 w_2", positive=True)
SIGMA_T = sp.Rational(8, 3) * sp.pi * re**2   # the README's sigma_T, in Svensson's r_e
UNITS = {c: 1, re: 1}   # numeric checks: c = r_e = 1


def lorentz(w):
    """A rational parametrisation of a Lorentz factor (see verify.zero_numerator): v = 1 + w,
    gamma = (v^2 + v^-2)/2, so that p = (v^2 - v^-2)/2 and gamma_cm = (v + 1/v)/2 for gamma_r."""
    v = 1 + w
    return (v**2 + v**-2) / 2


# --- Svensson (1982), transcribed from the scan ---

def sv_beta(gam):
    return sp.sqrt(1 - 1 / gam**2)


def sv_L(b):
    return sp.log((1 + b) / (1 - b))


def sv_sigma(gam, printed=False):
    """Eq. 6 (p. 323), sigma(gamma_cm). As printed, its logarithm is ln[(1 + beta)/(1 + beta)] = 0."""
    b = sv_beta(gam)
    lg = 0 if printed else sv_L(b)
    return re**2 * sp.pi / (4 * b * gam**2) * ((2 + 2 / gam**2 - 1 / gam**4) * lg / b - 2 - 2 / gam**2)


def sv_eq16_integrand(gam):
    """Eq. 16 (p. 324), the integrand 4 beta_cm gamma_cm^3 c sigma(gamma_cm) (eq. 6 corrected)."""
    return 4 * sv_beta(gam) * gam**3 * c * sv_sigma(gam)


def sv_eq18_bracket(gam):
    """Eq. 18 (p. 324): the bracket beta^3 gamma^2 L(beta) - 2 gamma^2 + (3/4) L^2(beta)."""
    b = sv_beta(gam)
    return b**3 * gam**2 * sv_L(b) - 2 * gam**2 + sp.Rational(3, 4) * sv_L(b)**2


def sv_gcm(sign, a=gp, b=gm):
    """Eqs. 17a (sign +1, gamma_cm^max) and 17b (sign -1, gamma_cm^min)."""
    return sp.sqrt((1 + a * b + sign * a * b * sv_beta(a) * sv_beta(b)) / 2)


def sv_rate(a=gp, b=gm):
    """Eq. 18, the angle-averaged rate."""
    pre = c * re**2 * sp.pi / (sv_beta(a) * a**2 * sv_beta(b) * b**2)
    return pre * (sv_eq18_bracket(sv_gcm(1, a, b)) - sv_eq18_bracket(sv_gcm(-1, a, b)))


def sv_gstar2(kk=k, a=gp, b=gm):
    """Eq. 31 (p. 326), gamma_cm*^2."""
    return kk * (a + b - kk)


def sv_k(name, a=gp, b=gm):
    """Eqs. 33a-d (p. 326)."""
    ba, bb = sv_beta(a), sv_beta(b)
    return {"k_U": (a * (1 + ba) + b * (1 + bb)) / 2, "k_+": (a * (1 + ba) + b * (1 - bb)) / 2,
            "k_-": (a * (1 - ba) + b * (1 + bb)) / 2, "k_L": (a * (1 - ba) + b * (1 - bb)) / 2}[name]


def sv_eq41(kk=k, gr=g):
    """Eq. 41 (p. 328), one lepton at rest, and its range [gamma_r(1 -+ beta_r) + 1]/2."""
    br = sv_beta(gr)
    spec = c * re**2 * sp.pi / (br * gr**2) * ((-(3 + gr) / (1 + gr) + (3 + gr) / kk - 1 / kk**2)
                                               / (1 - kk / (1 + gr))**2 - 2)
    return spec, (gr * (1 - br) + 1) / 2, (gr * (1 + br) + 1) / 2


def sv_G51(gam, gs, cc, dd):
    """Eq. 51 (p. 330), with u of eq. 53."""
    u = sp.sqrt(cc * gam**2 + gs**2)
    return (2 * gam**2 + 2 - 1 / gam**2) / u + (2 * gam**2 - dd) / (2 * u**3)


def sv_cdu(sign, kk=k, a=gp, b=gm, gam=gcm, gs=gst):
    """Eqs. 52-54: c, d, u of the given sign (gamma_mp is gamma_- for +)."""
    other = b if sign > 0 else a
    cc = (other - kk)**2 - 1
    dd = other * (a + b) + sign * kk * (a - b)
    return cc, dd, sp.sqrt(cc * gam**2 + gs**2)


def sv_I(cc, uu, gam=gcm, gs=gst):
    """Eq. 57: the branches for c > 0 and c < 0."""
    return sp.log(gam * sp.sqrt(cc) + uu) / sp.sqrt(cc), sp.asin(gam / gs * sp.sqrt(-cc)) / sp.sqrt(-cc)


def sv_H(gam, gs, cc, dd, uu, ii):
    """Eq. 56."""
    return ((2 + (1 - gs**2) / cc) * ii + (1 / gam - gam / cc + gam / (2 * gs**2) * (2 * cc - dd)) / uu
            + gam / cc * uu)


def sv_H58(gam, gs, dd):
    """Eq. 58 (p. 331), H for c = 0."""
    return ((sp.Rational(2, 3) * gam**3 + 2 * gam + 1 / gam) / gs
            + (sp.Rational(2, 3) * gam**3 - dd * gam) / (2 * gs**3))


def sv_prefactor(a=gp, b=gm):
    """Eq. 55: c r_e^2 pi/(beta_+ gamma_+^2 beta_- gamma_-^2)."""
    return c * re**2 * sp.pi / (sv_beta(a) * a**2 * sv_beta(b) * b**2)


def sv_equal(gam):
    """The spectrum at k = gamma_+ = gamma_- = gamma (p. 331, item iv)."""
    b = sv_beta(gam)
    return c * re**2 * 2 * sp.pi / (b**2 * gam**4) * ((gam**2 + 1) * (sp.pi / 2 - sp.asin(1 / gam)) - b / gam)


# --- Boettcher & Schlickeiser (1996), from the TeX source of arXiv:astro-ph/9506112v1 (numeric, c = r_e = 1) ---

BS_TEX = (r"c_{\pm} = (k - \gmp)^2 \, - \, 1, \eqno(4)", r"\gc^{*2} = k \> (\gp + \gm - k), \eqno(5)",
          r"\vdswr (k, \gp, \gm) = {c \, \pi \, r_e^2 \over \gp^2 \, \gm^2} \cdot",
          r"\cdot \left( \sqrt{(\gp + \gm)^2 - 4 \gc^2} \, + \, H_+ \, + \, H_- \right) \Biggr\vert_{\gc^L}^{\gc^U} \eqno(6)",
          r"H_{\pm} = {\gc \over c_{\pm}} \sqrt{\gc^{*2} + c_{\pm} \gc^2} \> +",
          r"+ \> {\gc^{*2} \over {\left\vert c_{\pm} \right\vert}^{3/2}} \cdot \cases{\arcsin\left ( {\sqrt{-c_{\pm}} \> \gc "
          r"\over \gc^*}\right ) & for $c_{\pm} < 0$ \cr -$arsinh$\left ({\sqrt{c_{\pm}} \> \gc \over \gc^*}\right ) & for "
          r"$c_{\pm} > 0$ \cr}, \eqno(7)",
          r"\gc^L = \gc^{min} = \sqrt{{1 \over 2} \left( 1 + \gp \gm [ 1 - \bp \bm] \right)} \eqno(9 a)",
          r"\gc^U = \min \, \{\gc^{max}, \gc^*\} \eqno(9 b)",
          r"\gc^{max} = \sqrt{{1 \over 2} \left(1 + \gp\gm \left[1 + \bp\bm\right]\right)} \eqno(10)",
          r"= {1 \over 2} \left( \gp [1 + \bp] + \gm [1 + \bm]\right) \eqno(12)",
          r"\approx {c \pi r_e^2 \over \gp^2 \gm^2} \cdot \left[ {\left ( \gc^U \right )^2 \over \vert k - \gp \vert + {2 \over "
          r"\pi} } + {\left ( \gc^U \right )^2 \over \vert k - \gm \vert + {2 \over \pi} } \right ]. \eqno(16)",
          r"\left( \ln[ 4 \gamma_+ \gamma_-] - 2 \right).  \eqno(33)",
          r"used Svensson's formula (19 b) for the angle averaged reaction rate")


def bs_p(a):
    return mp.sqrt(a * a - 1)


def bs_gcm(sign, a, b):
    """Eq. 10 (sign +1, gamma_cm^max) and eq. 9a (sign -1, gamma_cm^L = gamma_cm^min)."""
    return mp.sqrt((1 + a * b * (1 + sign * bs_p(a) / a * bs_p(b) / b)) / 2)


def bs_kU(a, b):
    """Eq. 12."""
    return (a + bs_p(a) + b + bs_p(b)) / 2


def bs_H(g_, cc, gs2):
    """Eq. 7 (c != 0)."""
    gs = mp.sqrt(gs2)
    first = g_ / cc * mp.sqrt(max(gs2 + cc * g_ * g_, 0))
    if cc < 0:
        return first + gs2 / (-cc)**mp.mpf(1.5) * mp.asin(min(mp.sqrt(-cc) * g_ / gs, 1))
    return first - gs2 / cc**mp.mpf(1.5) * mp.asinh(mp.sqrt(cc) * g_ / gs)


def bs_eq6(kk, a, b):
    """Eq. 6, with c_+- of eq. 4 (c_+ = (k - gamma_-)^2 - 1), gamma_cm*^2 of eq. 5 and the limits of eqs. 9-10, for k < k_U."""
    gs2 = kk * (a + b - kk)
    if gs2 <= 0 or kk >= bs_kU(a, b):
        return mp.mpf(0)
    lo, up = bs_gcm(-1, a, b), min(bs_gcm(1, a, b), mp.sqrt(gs2))
    if up <= lo:
        return mp.mpf(0)
    cpl, cmi = (kk - b)**2 - 1, (kk - a)**2 - 1

    def prim(g_):
        return mp.sqrt(max((a + b)**2 - 4 * g_ * g_, 0)) + bs_H(g_, cpl, gs2) + bs_H(g_, cmi, gs2)
    return mp.pi / (a * a * b * b) * (prim(up) - prim(lo))


def bs_eq16(kk, a, b):
    """Eq. 16, with gamma_cm^U of eq. 9b, for k < k_U (eq. 12)."""
    gs2 = kk * (a + b - kk)
    if gs2 <= 0 or kk >= bs_kU(a, b):
        return mp.mpf(0)
    gu2 = min(bs_gcm(1, a, b)**2, gs2)
    return mp.pi / (a * a * b * b) * (gu2 / (abs(kk - a) + 2 / mp.pi) + gu2 / (abs(kk - b) + 2 / mp.pi))


def bs_eq33(a, b):
    """The rate eq. 33 uses: Svensson's eq. 19b, pi r_e^2 c [ln(4 gamma_+ gamma_-) - 2]/(gamma_+ gamma_-)."""
    return mp.pi * (mp.log(4 * a * b) - 2) / (a * b)


# --- the physics: the Dirac cross section and the Moller average ---

EE, PP, ct, st = sp.symbols("E P c_theta s_theta", real=True)


def ee_gg_trace():
    """e-(p_1) e+(p_2) -> gamma(k_1) gamma(k_2) in the centre-of-momentum frame (m = 1), with explicit Dirac matrices
    (Dirac representation): leptons of energy E and momentum P along +-z, photons at the angle theta (cos = c_theta,
    sin = s_theta). Returns the numerator (with s_theta^2 = 1 - c_theta^2) and the denominator (d_1 d_2)^2 of the
    spin and polarisation sum of |M|^2/e^4, with the two diagrams (propagators 1/d, d = -2 p_1.k) over d_1 d_2.
    Polarisation sums -g_mu nu (the amplitude is gauge invariant); spin sums p_1-slash + 1 and p_2-slash - 1."""
    I2, Z2 = sp.eye(2), sp.zeros(2)
    pauli = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
    gam = [sp.Matrix(sp.BlockMatrix([[I2, Z2], [Z2, -I2]]))] + [sp.Matrix(sp.BlockMatrix([[Z2, s_], [-s_, Z2]])) for s_ in pauli]
    one, met = sp.eye(4), (1, -1, -1, -1)

    def slash(a):
        return sum((met[m] * a[m] * gam[m] for m in range(4)), sp.zeros(4))
    p1, p2 = (EE, 0, 0, PP), (EE, 0, 0, -PP)
    k1, k2 = (EE, EE * st, 0, EE * ct), (EE, -EE * st, 0, -EE * ct)
    d1, d2 = -2 * mdot(p1, k1), -2 * mdot(p1, k2)
    q1 = slash([a - b for a, b in zip(p1, k1)]) + one
    q2 = slash([a - b for a, b in zip(p1, k2)]) + one
    total = 0
    for m in range(4):
        for n in range(4):
            vertex = gam[n] * q1 * gam[m] * d2 + gam[m] * q2 * gam[n] * d1
            total += met[m] * met[n] * sp.expand(((slash(p2) - one) * vertex * (slash(p1) + one)
                                                  * gam[0] * vertex.H * gam[0]).trace())
    return sp.expand(sp.expand(total).subs(st**2, 1 - ct**2)), (d1 * d2)**2


def spin_averaged_M2(E_=EE, P_=PP, c_=ct):
    """The spin-averaged |M|^2/e^4 of e+e- -> gamma gamma in the centre-of-momentum frame (m = 1; each lepton of energy E
    and momentum P, the photon at cos theta = c to the e-): 2 [p_1.k_2/p_1.k_1 + p_1.k_1/p_1.k_2 + 2(1/p_1.k_1 +
    1/p_1.k_2) - (1/p_1.k_1 + 1/p_1.k_2)^2], a quarter of ee_gg_trace() (ann.sigma_rest)."""
    pk1, pk2 = E_ * (E_ - P_ * c_), E_ * (E_ + P_ * c_)
    a_ = 1 / pk1 + 1 / pk2
    return 2 * (pk2 / pk1 + pk1 / pk2 + 2 * a_ - a_**2)


def cm_dsigma_domega(M2):
    """dsigma/dOmega = <|M|^2>/(64 pi^2 s) |k|/|p| in the centre-of-momentum frame for the spin-averaged |M|^2/e^4 = M2,
    with e^4 = 16 pi^2 r_e^2 (m = 1), s = 4E^2, |k| = E, |p| = P."""
    return M2 * 16 * sp.pi**2 * re**2 / (64 * sp.pi**2 * 4 * EE**2) * EE / PP


def sigma_from_trace(trace, denominator):
    """sigma(E) = (1/2) int dOmega dsigma/dOmega (the 1/2 for the identical photons), with
    <|M|^2> = trace e^4/(4 denominator), the angle integral in closed form."""
    dsdO = sp.cancel(cm_dsigma_domega(trace / (4 * denominator)))
    prim = ratint(dsdO, ct)
    return sp.Rational(1, 2) * 2 * sp.pi * (prim.subs(ct, 1) - prim.subs(ct, -1))


# --- the README, read from README.md ---

def P_DEF():
    return L.inline(r"p = \sqrt{\gamma^2 - 1}")


def p_of(gam):
    return L.to_sympy(L.sides(P_DEF())[1], {r"\gamma": gam})


def beta_of(gam):
    return L.to_sympy(L.sides(L.inline(r"\beta = p/\gamma"))[1], {"p": p_of(gam), r"\gamma": gam})


def leptons(a=gp, b=gm):
    return {r"\gamma_+": a, r"\gamma_-": b, "p_+": p_of(a), "p_-": p_of(b), r"\beta_+": beta_of(a),
            r"\beta_-": beta_of(b)}


def RATE():
    return L.display(r"\langle\sigma v\rangle(\gamma_+,\gamma_-) = c\thinspace \frac{G(\gamma_r^+)")


def GLINE():
    return L.display(r"G(x) = \int_1^{x}\sqrt{t^2-1}")


def SIGMA():
    return L.display(r"\sigma_{\rm rest}(t) = \frac{3\sigma_T}{8\thinspace (t+1)}")


def SPEC():
    return L.display(r"v\frac{d\sigma}{dk}(k;\gamma_+,\gamma_-) = \frac{3\sigma_T\thinspace c}{8\thinspace p_+\gamma_+\thinspace p_-\gamma_-}")


def DEFS():
    return L.display(r"S = \gamma_+ + \gamma_-,\qquad \gamma_s^2 = k\thinspace (S-k)")


def HLINE():
    return L.display(r"H_\pm(\gamma_\ast) = \left(2 + \frac{1-\gamma_s^2}{c_\pm}\right) I_\pm")


def CDU():
    return L.display(r"c_\pm = (\gamma_\mp - k)^2 - 1,\quad d_\pm")


def readme_gr(sign, a=gp, b=gm):
    """gamma_r^+- of the README."""
    return L.to_sympy(L.sides(L.part(RATE(), r"\gamma_r^\pm"))[1], leptons(a, b), sign=sign)


def readme_sigma(arg):
    return L.to_sympy(L.sides(L.part(SIGMA(), r"\sigma_{\rm rest}(t)"))[1], {"t": arg, r"\sigma_T": SIGMA_T})


def readme_G(arg):
    s = L.sides(L.part(GLINE(), "G(x)"))
    assert s[1] == r"\int_1^{x}\sqrt{t^2-1}\thinspace \sigma_{\rm rest}(t)\thinspace dt", s[1]
    assert L.part(GLINE(), r"\ell") == r"\ell = \mathrm{arccosh}\thinspace x"
    return L.to_sympy(s[2], {"x": arg, r"\sigma_T": SIGMA_T, r"\ell": sp.acosh(arg)})


def readme_rate(a=gp, b=gm):
    tex = L.sides(L.part(RATE(), r"\langle\sigma v\rangle"))[1]
    tokens = {**leptons(a, b), "c": c, r"\gamma_r^+": readme_gr(1, a, b), r"\gamma_r^-": readme_gr(-1, a, b)}
    return L.to_sympy(tex, tokens, funcs={"G": readme_G})


def readme_S(a=gp, b=gm):
    return L.to_sympy(L.sides(L.part(DEFS(), "S ="))[1], leptons(a, b))


def readme_gs2(kk=k, a=gp, b=gm):
    return L.to_sympy(L.sides(L.part(DEFS(), r"\gamma_s^2"))[1], {"k": kk, "S": readme_S(a, b)})


def readme_gstar(sign, a=gp, b=gm):
    """gamma_ast^max (sign +1) and gamma_ast^min (-1): the README pairs max, min with gamma_r^+, gamma_r^-."""
    p = L.part(DEFS(), r"\gamma_\ast^{\max,\min}")
    s = "+" if sign > 0 else "-"
    return L.to_sympy(L.sides(p)[1], {rf"\gamma_r^{s}": readme_gr(sign, a, b)}, sign=sign)


def readme_cdu(sign, kk=k, a=gp, b=gm, gam=gcm, gs=gst):
    """c, d, u of the README (gamma_ast = gam, gamma_s = gs)."""
    tokens = {**leptons(a, b), "k": kk, "S": readme_S(a, b), r"\gamma_\ast": gam, r"\gamma_s": gs}
    cc = L.to_sympy(L.sides(L.part(CDU(), r"c_\pm"))[1], tokens, sign=sign)
    dd = L.to_sympy(L.sides(L.part(CDU(), r"d_\pm"))[1], tokens, sign=sign)
    uu = L.to_sympy(L.sides(L.part(CDU(), r"u_\pm"))[1], {**tokens, "c_+": cc, "c_-": cc}, sign=sign)
    return cc, dd, uu


def readme_I(sign, cc, uu, gam=gcm, gs=gst):
    """The two branches of the README's I (c > 0, c < 0)."""
    cases = L.between(L.part(CDU(), r"I_\pm"), r"\begin{cases}", r"\end{cases}")
    rows = [[s.strip() for s in L.split_top(row, "&")] for row in L.split_top(cases, r"\cr")]
    assert [r[1] for r in rows] == [r"c_\pm \gt 0", r"c_\pm \lt 0"], rows
    tokens = {r"\gamma_\ast": gam, r"\gamma_s": gs, "c_+": cc, "c_-": cc, "u_+": uu, "u_-": uu}
    return tuple(L.to_sympy(r[0], tokens, sign=sign) for r in rows)


def readme_H(sign, gam, gs, cc, dd, uu, ii):
    tex = L.sides(L.part(HLINE(), r"H_\pm(\gamma_\ast)"))[1]
    s = "+" if sign > 0 else "-"
    return L.to_sympy(tex, {r"\gamma_\ast": gam, r"\gamma_s": gs, f"c_{s}": cc, f"d_{s}": dd, f"u_{s}": uu,
                            f"I_{s}": ii}, sign=sign)


def readme_spectrum_pieces():
    """Prefactor and bracket of the README's spectrum; the limits are asserted as written."""
    spec = SPEC()
    limits = spec.split(r"\Big]")[1]
    assert limits == r"_ {\gamma_\ast^{\min}}^{\min(\gamma_\ast^{\max},\ \gamma_s)},", limits
    pre = L.to_sympy(L.between(spec, "= ", r"\Big["), {**leptons(), "c": c, r"\sigma_T": SIGMA_T})
    return pre, L.between(spec, r"\Big[", r"\Big]")


def readme_k(name, a=gp, b=gm):
    """k_L, k_U (inline) and k_+, k_- (display) of the README."""
    if name in ("k_L", "k_U"):
        tex = {"k_L": r"k_L = \tfrac12[\gamma_+(1-\beta_+) + \gamma_-(1-\beta_-)]",
               "k_U": r"k_U = \tfrac12[\gamma_+(1+\beta_+) + \gamma_-(1+\beta_-)]"}[name]
        return L.to_sympy(L.sides(L.inline(tex))[1], leptons(a, b))
    tex = L.sides(L.display(r"k_\pm = \tfrac{1}{2}\left[\gamma_\pm(1+\beta_\pm)"))[1]
    return L.to_sympy(tex, leptons(a, b), sign=1 if name == "k_+" else -1)


def readme_eq41(kk=k, gam=g):
    """The README's eq. 41 and its range."""
    p = L.parts(L.display(r"v\frac{d\sigma}{dk}(k;\gamma) = \frac{3\sigma_T\thinspace c}{8\thinspace p\thinspace \gamma}"))
    tokens = {"k": kk, r"\gamma": gam, "p": p_of(gam), "c": c, r"\sigma_T": SIGMA_T}
    lo, mid, hi = (s.strip() for s in re_split(r"\\le(?![A-Za-z])", p[1]))
    assert mid == "k", p[1]
    return L.to_sympy(L.sides(p[0])[1], tokens), L.to_sympy(lo, tokens), L.to_sympy(hi, tokens)


# --- numeric spectra (mpmath functions of k, gamma_+, gamma_-) ---

def _lambdify(args, expr):
    return sp.lambdify(args, expr.subs(UNITS), "mpmath")


def spectrum_function(source):
    """The README's spectrum (source "README", from its text) or Svensson's eqs. 55-57 with the limits
    of eq. 32 ("Svensson"), assembled from the pieces, with the branch of I chosen by the sign of c."""
    if source == "README":
        pre, bracket = readme_spectrum_pieces()
        gmax, gmin, gs = readme_gstar(1), readme_gstar(-1), sp.sqrt(readme_gs2())
        sign_c = [readme_cdu(s)[0] for s in (1, -1)]

        def F(bp, bm):
            def H(sign, branch):
                cc, dd, uu = readme_cdu(sign, gs=sp.Symbol("GS"))
                ii = readme_I(sign, cc, uu, gs=sp.Symbol("GS"))[branch]
                return lambda arg: readme_H(sign, arg, sp.Symbol("GS"), cc, dd, uu, ii).subs(gcm, arg)
            expr = L.to_sympy(bracket, {"S": readme_S(), r"\gamma_\ast": gcm},
                              funcs={"H_+": H(1, bp), "H_-": H(-1, bm)})
            return expr.subs(sp.Symbol("GS"), gs)
    else:
        pre = sv_prefactor()
        gmax, gmin, gs = sv_gcm(1), sv_gcm(-1), sp.sqrt(sv_gstar2())
        sign_c = [sv_cdu(s)[0] for s in (1, -1)]

        def F(bp, bm):
            terms = sp.sqrt((gp + gm)**2 - 4 * gcm**2)
            for sign, branch in ((1, bp), (-1, bm)):
                cc, dd, uu = sv_cdu(sign)
                terms += sv_H(gcm, gst, cc, dd, uu, sv_I(cc, uu)[branch])
            return terms.subs(gst, gs)
    args = (gcm, k, gp, gm)
    f = {(bp, bm): _lambdify(args, F(bp, bm)) for bp in (0, 1) for bm in (0, 1)}
    pre_f, gmax_f, gmin_f, gs_f = (_lambdify((k, gp, gm), e) for e in (pre, gmax, gmin, gs))
    c_f = [_lambdify((k, gp, gm), e) for e in sign_c]

    def spectrum(kk, a, b):
        up, lo = min(gmax_f(kk, a, b), gs_f(kk, a, b)), gmin_f(kk, a, b)
        if up <= lo:
            return mp.mpf(0)
        branch = tuple(0 if cf(kk, a, b) > 0 else 1 for cf in c_f)
        return pre_f(kk, a, b) * (f[branch](up, kk, a, b) - f[branch](lo, kk, a, b))
    return spectrum


def sample_points(n, seed, both_branches=True):
    """(k, gamma_+, gamma_-): Lorentz factors from 1.01 to 1e4, k inside (k_L, k_U); half of the points
    with k within 0.9 of gamma_-, where c_+ < 0 (the arcsin branch)."""
    r = rng(seed)
    kL, kU = (_lambdify((gp, gm), sv_k(n_)) for n_ in ("k_L", "k_U"))
    out = []
    while len(out) < n:
        a, b = log_uniform(r, 1.01, 1e4), log_uniform(r, 1.01, 1e4)
        lo, hi = kL(a, b), kU(a, b)
        kk = lo + (hi - lo) * mp.mpf(r.random())
        if both_branches and len(out) % 2:
            kk = b + mp.mpf(0.9) * (2 * mp.mpf(r.random()) - 1)
            if not lo < kk < hi:
                continue
        out.append((kk, a, b))
    return out


# --- the checks ---

@derivation("ann.sigma_rest")
def check_sigma_rest():
    """sigma_rest(t) is Svensson's eq. 6 (Dirac's cross section) with t = gamma_r = 2 gamma_cm^2 - 1, and the e+e- -> gamma gamma trace."""
    to_t = (readme_sigma(2 * gcm**2 - 1) - sv_sigma(gcm)).subs(gcm, (1 + w1 + 1 / (1 + w1)) / 2)
    printed = (readme_sigma(2 * gcm**2 - 1) - sv_sigma(gcm, printed=True)).subs(gcm, (1 + w1 + 1 / (1 + w1)) / 2)
    # the cross section from the trace: E = gamma_cm = (v + 1/v)/2, P = (v - 1/v)/2, t = p_+.p_- = s/2 - 1 = 2E^2 - 1
    trace, denominator = ee_gg_trace()
    sigma = sigma_from_trace(trace, denominator)
    textbook = 4 * spin_averaged_M2() * denominator   # 4 <|M|^2>/e^4, x (d_1 d_2)^2
    on_shell = {EE: (1 + w1 + 1 / (1 + w1)) / 2, PP: (1 + w1 - 1 / (1 + w1)) / 2}
    with L.patched(r"\frac{t^2+4t+1}{t^2-1}", r"\frac{t^2+4t-1}{t^2-1}"):
        broken = (readme_sigma(2 * gcm**2 - 1) - sv_sigma(gcm)).subs(gcm, (1 + w1 + 1 / (1 + w1)) / 2)
        broken_trace = readme_sigma(2 * on_shell[EE]**2 - 1) - sigma.subs(on_shell)
    return [exact("README sigma_rest = eq. 6 with ln[(1 + beta)/(1 - beta)]", to_t),
            exact_fails("eq. 6 as printed, ln[(1 + beta)/(1 + beta)] = 0, differs (a misprint)", printed),
            exact("the trace (explicit Dirac matrices) = 8 e^4 [p_1.k_2/p_1.k_1 + p_1.k_1/p_1.k_2 + 2(1/p_1.k_1 + 1/p_1.k_2) "
                  "- (1/p_1.k_1 + 1/p_1.k_2)^2], the spin-averaged |M|^2 x 4", (trace - textbook).subs(on_shell)),
            exact("README sigma_rest(t) = (1/2) int dOmega <|M|^2>/(64 pi^2 s) E/P from the trace (e^4 = 16 pi^2 r_e^2, "
                  "t = s/2 - 1)", readme_sigma(2 * on_shell[EE]**2 - 1) - sigma.subs(on_shell)),
            exact_fails("control: t^2 + 4t - 1 in the README", broken),
            exact_fails("control: t^2 + 4t - 1 against the trace", broken_trace),
            exact("cold pairs: beta sigma_rest -> 3 sigma_T/8 = pi r_e^2 for t -> 1 (Dirac's low-velocity limit)",
                  sp.limit(sp.sqrt(lorentz(w1)**2 - 1) / lorentz(w1) * readme_sigma(lorentz(w1)), w1, 0, "+")
                  - sp.Rational(3, 8) * SIGMA_T)]


@derivation("ann.G")
def check_G():
    """G(x) is the primitive of sqrt(t^2 - 1) sigma_rest(t) from 1, and Svensson's eq. 18 bracket."""
    Gx = readme_G(x)
    integrand = sp.sqrt(x**2 - 1) * readme_sigma(x)
    xv = lorentz(w1)   # x = gamma_r
    bracket = (Gx - sp.Rational(3, 2) * SIGMA_T - sp.Rational(3, 4) * SIGMA_T * sv_eq18_bracket(sp.sqrt((1 + x) / 2)))
    with L.patched(r"\frac{3}{4}\thinspace \ell^2", r"\frac{3}{8}\thinspace \ell^2"):
        broken = sp.diff(readme_G(x), x) - integrand
    return [exact("dG/dx = sqrt(x^2 - 1) sigma_rest(x)", (sp.diff(Gx, x) - integrand).subs(x, xv)),
            Result("G(1) = 0", sp.simplify(Gx.subs(x, 1)) == 0, f"G(1) = {sp.simplify(Gx.subs(x, 1))}"),
            exact("G(x) - 3 sigma_T/2 = (3 sigma_T/4) [eq. 18 bracket] at gamma_cm = ((1 + x)/2)^(1/2)",
                  bracket.subs(x, xv)),
            exact_fails("control: (3/8) l^2 in G", broken.subs(x, xv))]


@derivation("ann.rate")
def check_rate():
    """<sigma v> is Svensson's eq. 18; eq. 18 is the integral of eq. 16 with eq. 6."""
    step = sp.diff(re**2 * sp.pi * c * sv_eq18_bracket(gcm), gcm) - sv_eq16_integrand(gcm)
    a, b = lorentz(w1), lorentz(w2)
    gr = [exact(f"(1 + gamma_r^{s})/2 = (gamma_cm^{m})^2 (eq. 17{l})",
                ((1 + readme_gr(sign)) / 2 - sv_gcm(sign)**2).subs({gp: a, gm: b}))
          for sign, s, m, l in ((1, "+", "max", "a"), (-1, "-", "min", "b"))]
    ours = _lambdify((gp, gm), readme_rate())
    theirs = _lambdify((gp, gm), sv_rate())
    r = rng(2)
    pts = [(log_uniform(r, 1.001, 1e6), log_uniform(r, 1.001, 1e6)) for _ in range(20)]
    with L.patched(r"\frac{G(\gamma_r^+) - G(\gamma_r^-)}{2\thinspace p_+p_-",
                   r"\frac{G(\gamma_r^+) - G(\gamma_r^-)}{4\thinspace p_+p_-"):
        broken = _lambdify((gp, gm), readme_rate())
        broken_moller = moller()[1]
    return [exact("d/dgamma_cm [c r_e^2 pi (eq. 18 bracket)] = eq. 16 integrand (eq. 6 corrected)",
                  step.subs(gcm, (1 + w1 + 1 / (1 + w1)) / 2)),
            *gr,
            numeric("README <sigma v> = eq. 18, gamma_+- from 1.001 to 1e6", ours, theirs, pts),
            control("control: 4 p_+ p_- in the denominator", ours, broken, pts),
            *moller()[0],
            exact_fails("control: 4 p_+ p_- against the Moller average", broken_moller),
            cold_rate()]


def cold_rate():
    """<sigma v> of cold pairs (gamma_+- = 1 + 1e-16, 1 + 2e-16) against pi r_e^2 c = 3 sigma_T c/8 (c = r_e = 1)."""
    with mp.workdps(40):
        dev = rate_mp(1 + mp.mpf(10)**-16, 1 + 2 * mp.mpf(10)**-16) / mp.pi - 1
    return Result("cold pairs: <sigma v> -> pi r_e^2 c = 3 sigma_T c/8 (gamma_+- = 1 + 1e-16, 1 + 2e-16)", abs(dev) < 1e-15,
                  f"numeric: rel. difference {mp.nstr(dev, 3)}")


def v_moller(gr, a=gp, b=gm):
    """The Moller velocity c [(p_+.p_-)^2 - 1]^(1/2)/(gamma_+ gamma_-) (m = 1) for gr = p_+.p_-."""
    return c * sp.sqrt(gr**2 - 1) / (a * b)


def moller():
    """<sigma v> as the Moller average (1/2) int_-1^1 dmu sigma_rest(gamma_r) v_Moller over the angle arccos mu between
    the momenta (p_+ along z, gamma_r = p_+.p_-), for ANY sigma_rest. By Leibniz' rule and the README's
    G(x) = int_1^x (t^2 - 1)^(1/2) sigma_rest(t) dt, Phi(mu) = -c G(gamma_r(mu))/(2 p_+p_- gamma_+gamma_-) is a
    primitive of the integrand, and Phi(1) - Phi(-1) is the README's <sigma v>. Returns the two Results and the
    difference README - [Phi(1) - Phi(-1)] (for the control)."""
    mu, t_ = sp.Symbol("mu", real=True), sp.Symbol("t", positive=True)
    sigma = sp.Function("sigma_rest")
    pp, pm = p_of(gp), p_of(gm)
    gr_mu = mdot((gp, 0, 0, pp), (gm, pm * sp.sqrt(1 - mu**2), 0, pm * mu))

    def G_def(X):
        return sp.Integral(sp.sqrt(t_**2 - 1) * sigma(t_), (t_, 1, X))
    Gf = sp.Function("G")
    tokens = {**leptons(), "c": c, r"\gamma_r^+": readme_gr(1), r"\gamma_r^-": readme_gr(-1)}
    rate = L.to_sympy(L.sides(L.part(RATE(), r"\langle\sigma v\rangle"))[1], tokens, funcs={"G": Gf})
    Phi = lambda m_: -c * Gf(gr_mu.subs(mu, m_)) / (2 * pp * pm * gp * gm)   # noqa: E731
    ends = sp.expand(rate - (Phi(1) - Phi(-1)))
    slope = sp.diff(Phi(mu).replace(Gf, G_def), mu) - sigma(gr_mu) * v_moller(gr_mu) / 2
    return [exact("README <sigma v> = Phi(1) - Phi(-1): gamma_r^-+ = p_+.p_- at mu = +-1, Phi = -c G(p_+.p_-)/(2 p_+p_- "
                  "gamma_+gamma_-)", ends),
            exact("dPhi/dmu = (1/2) sigma_rest(gamma_r) c [gamma_r^2 - 1]^(1/2)/(gamma_+gamma_-) (Moller), any sigma_rest: "
                  "<sigma v> is the Moller angle average", slope.doit())], ends


@derivation("ann.gr_minus")
def check_gr_minus():
    """(gamma_+^2 + gamma_-^2 - 1)/gamma_r^+ = gamma_+ gamma_- - p_+ p_-."""
    tex = L.inline(r"(\gamma_+^2 + \gamma_-^2 - 1)/\gamma_r^+", "checks")
    form = L.to_sympy(tex, {**leptons(), r"\gamma_r^+": readme_gr(1)})
    a, b = lorentz(w1), lorentz(w2)
    return [exact("= gamma_r^-", (form - readme_gr(-1)).subs({gp: a, gm: b}))]


@derivation("ann.spectrum_defs")
def check_spectrum_defs():
    """S, gamma_s and gamma_ast^max,min are Svensson's eqs. 31 and 17."""
    a, b = lorentz(w1), lorentz(w2)
    readme_spectrum_pieces()   # asserts the limits as written
    return [exact("gamma_s^2 = k(S - k) = eq. 31", readme_gs2() - sv_gstar2()),
            *[exact(f"gamma_ast^{m} = gamma_cm^{m} (eq. 17{l})", (readme_gstar(s) - sv_gcm(s)).subs({gp: a, gm: b}))
              for s, m, l in ((1, "max", "a"), (-1, "min", "b"))],
            Result("limits [gamma_ast^min, min(gamma_ast^max, gamma_s)] = eq. 32", True,
                   "as written (asserted): eq. 32 has min[gamma_cm^max, gamma_cm*] and gamma_cm^min")]


def readme_gstar2(gr):
    """gamma_ast^2 of the README for the relative Lorentz factor gr."""
    found = re_search(r"\(\$\\gamma_\\ast\^2 = ([^$]+)\$ for the relative Lorentz factor",
                      L.line("the Lorentz factor of each lepton in the centre-of-momentum frame ($"))
    return L.to_sympy(found.group(1), {r"\gamma_r": gr})


@derivation("ann.gstar")
def check_gstar():
    """gamma_ast^2 = (1 + gamma_r)/2: each lepton's Lorentz factor in the centre-of-momentum frame, gamma_r = p_+.p_-."""
    a, b = lorentz(w1), lorentz(w2)
    mu = sp.Symbol("mu", real=True)   # the cosine of the angle between the momenta
    pp_, pm_ = (a, 0, 0, p_of(a)), (b, p_of(b) * sp.sqrt(1 - mu**2), 0, p_of(b) * mu)
    gr = mdot(pp_, pm_)                                    # the energy of one lepton in the other's rest frame (m = 1)
    tot = [u + v for u, v in zip(pp_, pm_)]
    cm = [mdot(q, tot)**2 / mdot(tot, tot) for q in (pp_, pm_)]   # (p.P)^2/P^2: each lepton's energy squared in the CM frame
    with L.patched(r"$\gamma_\ast^2 = (1 + \gamma_r)/2$", r"$\gamma_\ast^2 = (2 + \gamma_r)/2$"):
        broken = readme_gstar2(gr)
    return [exact("the positron's gamma^2 in the CM frame, (p_+.P)^2/P^2 with P = p_+ + p_-, = (1 + gamma_r)/2",
                  cm[0] - readme_gstar2(gr)),
            exact("the electron's, the same", cm[1] - readme_gstar2(gr)),
            exact("gamma_r = p_+.p_- runs from gamma_r^- (mu = 1) to gamma_r^+ (mu = -1), the README's gamma_r^-+: gamma_ast^min,max "
                  "of the spectrum are this gamma_ast at its ends", (gr.subs(mu, 1) - readme_gr(-1, a, b))**2
                  + (gr.subs(mu, -1) - readme_gr(1, a, b))**2),
            exact_fails("control: (2 + gamma_r)/2", cm[0] - broken)]


@derivation("ann.H")
def check_H():
    """H_+- are Svensson's eq. 56, and the integrals over gamma_cm of Svensson's eq. 51."""
    out = []
    for sign, s in ((1, "+"), (-1, "-")):
        out.append(exact(f"H_{s} = eq. 56", readme_H(sign, gcm, gst, C, D, U, I_) - sv_H(gcm, gst, C, D, U, I_)))
        for branch, name, cc in ((0, "c > 0", cp**2), (1, "c < 0", -cp**2)):
            uu = sp.sqrt(cc * gcm**2 + gst**2)
            ii = readme_I(sign, cc, uu)[branch]
            dH = sp.diff(readme_H(sign, gcm, gst, cc, D, uu, ii), gcm) - sv_G51(gcm, gst, cc, D)
            out.append(exact(f"dH_{s}/dgamma_ast = G_{s} of eq. 51 ({name})", dH))
    with L.patched(r"(2c_\pm - d_\pm)", r"(2c_\pm + d_\pm)"):
        out.append(exact_fails("control: 2c + d in H", readme_H(1, gcm, gst, C, D, U, I_) - sv_H(gcm, gst, C, D, U, I_)))
    return out


@derivation("ann.cdu")
def check_cdu():
    """c, d, u and I of the README are Svensson's eqs. 52-54 and 57."""
    out = []
    for sign, s in ((1, "+"), (-1, "-")):
        ours, theirs = readme_cdu(sign), sv_cdu(sign)
        out += [exact(f"{n}_{s} = eq. {e}", o - t) for n, e, o, t in zip("cdu", (54, 52, 53), ours, theirs)]
        for branch, name, cc in ((0, "c > 0", cp**2), (1, "c < 0", -cp**2)):
            uu = sp.sqrt(cc * gcm**2 + gst**2)
            out.append(exact(f"I_{s} ({name}) = eq. 57", readme_I(sign, cc, uu)[branch] - sv_I(cc, uu)[branch]))
    return out


@derivation("ann.eq58")
def check_eq58():
    """Svensson's eq. 58 is eq. 56 at c = 0, up to a constant in gamma_cm (which cancels in eq. 55)."""
    out = [exact("d(eq. 58)/dgamma_cm = G of eq. 51 at c = 0 (Svensson's own derivation; with dH/dgamma = G "
                 "for every c, see ann.H, eq. 58 is the limit)", sp.diff(sv_H58(gcm, gst, D), gcm) - sv_G51(gcm, gst, 0, D))]
    r = rng(5)
    for branch, name, sgn in ((0, "c > 0", 1), (1, "c < 0", -1)):
        cc = sgn * mp.mpf(10)**-30
        uu = sp.sqrt(C * gcm**2 + gst**2)
        H = _lambdify((gcm, gst, C, D), readme_H(1, gcm, gst, C, D, uu, readme_I(1, C, uu)[branch]))
        H58 = _lambdify((gcm, gst, D), sv_H58(gcm, gst, D))
        pts = []
        for _ in range(10):
            gs_ = log_uniform(r, 1.5, 1e3)
            g1, g2 = 1 + (gs_ - 1) * mp.mpf(r.random()), 1 + (gs_ - 1) * mp.mpf(r.random())
            pts.append((g1, g2, gs_, (gs_**2 + 1) * mp.mpf(r.random()) * 4))
        out.append(numeric(f"[H - eq. 58](gamma_1) = [H - eq. 58](gamma_2) at {name}, |c| = 1e-30",
                           lambda g1, g2, gs_, dd: H(g1, gs_, cc, dd) - H(g2, gs_, cc, dd),
                           lambda g1, g2, gs_, dd: H58(g1, gs_, dd) - H58(g2, gs_, dd), pts, dps=80, tol=mp.mpf(10)**-25))
    return out


@derivation("ann.spectrum")
def check_spectrum():
    """The README's spectrum is Svensson's eq. 55, and agrees with his special cases."""
    a, b = lorentz(w1), lorentz(w2)
    pre, _ = readme_spectrum_pieces()
    ours, theirs = spectrum_function("README"), spectrum_function("Svensson")
    pts = sample_points(24, 3)
    with L.patched(r"(2c_\pm - d_\pm)", r"(2c_\pm + d_\pm)"):
        broken = spectrum_function("README")
    eq41 = _lambdify((k, g), readme_eq41()[0])
    equal = _lambdify((g,), sv_equal(g))
    r = rng(4)
    rest = []
    for _ in range(8):
        gam = log_uniform(r, 1.01, 1e4)
        lo, hi = (gam - mp.sqrt(gam**2 - 1) + 1) / 2, (gam + mp.sqrt(gam**2 - 1) + 1) / 2
        rest.append((lo + (hi - lo) * mp.mpf(r.random()), gam))
    sym = [(kk, a_, b_) for kk, a_, b_ in pts]
    gams = [(log_uniform(r, 1.01, 1e4),) for _ in range(8)]
    return [exact("prefactor 3 sigma_T c/(8 p_+ gamma_+ p_- gamma_-) = eq. 55's", (pre - sv_prefactor()).subs({gp: a, gm: b})),
            numeric("README spectrum = eq. 55 (both branches of I)", ours, theirs, pts),
            control("control: 2c + d in H", ours, broken, pts),
            numeric(
                "symmetry k <-> gamma_+ + gamma_- - k (Svensson p. 331, i)", lambda kk, a_, b_: ours(kk, a_, b_),
                    lambda kk, a_, b_: ours(a_ + b_ - kk, a_, b_), sym),
            numeric("gamma_- -> 1 (1 + 1e-40): the README's eq. 41 (Svensson p. 331)",
                    lambda kk, gam: ours(kk, gam, 1 + mp.mpf(10)**-40), eq41, rest, dps=90, tol=mp.mpf(10)**-15),
            numeric("k = gamma_+ = gamma_- (k = gamma (1 + 1e-30)): Svensson's closed form (p. 331, iv)",
                    lambda gam: ours(gam * (1 + mp.mpf(10)**-30), gam, gam), equal, gams, dps=80, tol=mp.mpf(10)**-12)]


NPOS, NELE = sp.Function("n_+"), sp.Function("n_-")    # the e+, e- densities per unit gamma: any functions
SV, VDS = sp.Function("S"), sp.Function("V")            # <sigma v>(gamma_+, gamma_-) and v dsigma/dk(k; gamma_+, gamma_-): any
g1 = sp.Symbol("gamma'", positive=True)
AN, BN = sp.symbols("A B", positive=True)               # the weights of the lines of e+ and e-


def SINKS():
    return L.display(r"\dot n_\mp(\gamma) = -\thinspace n_\mp(\gamma)")


def SOURCE():
    return L.display(r"\dot n_\gamma(k) = \iint d\gamma_+")


def readme_sink(sign, rate=SV):
    """The README's sink of e- (sign +1) or e+ (sign -1) per unit gamma and gamma': the factor in front times the integrand
    of its integral over gamma', with <sigma v> = rate."""
    lhs, rhs = L.sides(L.parts(SINKS())[0])
    assert lhs == r"\dot n_\mp(\gamma)", lhs
    head, body = rhs.split(r"\int d\gamma'\thinspace ")
    tokens, funcs = {r"\gamma'": g1, r"\gamma": g}, {"n_+": NPOS, "n_-": NELE, r"\langle\sigma v\rangle": rate}
    return L.to_sympy(head, tokens, funcs, sign) * L.to_sympy(body, tokens, funcs, sign)


def readme_source(spec=VDS):
    """The README's photon source per unit k, gamma_+ and gamma_-: the integrand of its double integral."""
    lhs, rhs = L.sides(L.parts(SOURCE())[0])
    head, body = rhs.split(r"\iint d\gamma_+\thinspace d\gamma_-\thinspace ")
    assert lhs == r"\dot n_\gamma(k)" and head == "", rhs
    tokens = {r"\gamma_+": gp, r"\gamma_-": gm, "k": k}
    return L.to_sympy(body.replace(";", ","), tokens, {"n_+": NPOS, "n_-": NELE, r"v\frac{d\sigma}{dk}": spec})


def readme_counts(a, b):
    """The number and the energy of the photons of one annihilation of an e+ of a and an e- of b, per <sigma v>(a, b): the
    README's normalisation, int v dsigma/dk dk = 2 <sigma v> and int k v dsigma/dk dk = (gamma_+ + gamma_-) <sigma v>."""
    line = L.line("which counts both photons: $")
    number = re_search(r"\$\\int v\\ \\frac\{d\\sigma\}\{dk\}\\ dk = ([^$]+)\$", line).group(1)
    energy = re_search(r"\$\\int k\\ v\\ \\frac\{d\\sigma\}\{dk\}\\ dk\$ \$= ([^$]+)\$", line).group(1)
    tokens = {r"\langle\sigma v\rangle": 1, r"\gamma_+": a, r"\gamma_-": b}
    return L.to_sympy(number, tokens), L.to_sympy(energy, tokens)


def on_lines(expr, x_, y_, a, b):
    """int dx_ int dy_ expr for a line of e+ at a (A delta(gamma - a)) and a line of e- at b (B delta(gamma - b)); expr is
    linear in the densities n_+ and n_- at x_ and y_."""
    total = 0
    for u in (a, b):
        for v in (a, b):
            total += (expr.subs({x_: u, y_: v}).replace(NPOS, lambda z: AN if z == a else 0)
                      .replace(NELE, lambda z: BN if z == b else 0))
    return total


@derivation("ann.sinks")
def check_sinks():
    """The sinks count the annihilations, n_+(gamma') n_-(gamma) <sigma v> per unit gamma and gamma', one e- and one e+ each."""
    # <sigma v> takes the positron's gamma first, as in the README. The e- of gamma meet the e+ of any gamma', and the
    # e+ of gamma the e- of any gamma'
    counted = {1: -NELE(g) * NPOS(g1) * SV(g1, g), -1: -NPOS(g) * NELE(g1) * SV(g, g1)}
    a, b = lorentz(w1), lorentz(w2)
    out = [exact("README e- sink = the annihilations of the e- of gamma (any <sigma v>; integrands in gamma')",
                 readme_sink(1) - counted[1]),
           exact("README e+ sink = those of the e+ of gamma, but with <sigma v>(gamma', gamma) for <sigma v>(gamma, gamma') "
                 "(any <sigma v>)", readme_sink(-1) - counted[-1] + NPOS(g) * NELE(g1) * (SV(g1, g) - SV(g, g1))),
           exact("the README's <sigma v>(gamma_+, gamma_-) is symmetric, so the e+ sink is the count too",
                 readme_rate(a, b) - readme_rate(b, a))]
    removed = [-on_lines(readme_sink(s, lambda u, v: readme_rate(u, v)), g, g1, a, b) for s in (1, -1)]
    out.append(exact("lines of e+ (A at gamma_1) and e- (B at gamma_2): e- removed = e+ removed = A B <sigma v>(gamma_1, gamma_2), "
                     "the annihilations", (removed[0] - AN * BN * readme_rate(a, b))**2 + (removed[1] - AN * BN * readme_rate(a, b))**2))
    with L.patched(r"\int d\gamma'\thinspace n_\pm(\gamma')", r"\int d\gamma'\thinspace n_\mp(\gamma')"):
        out.append(exact_fails("control: n_mp(gamma') in the integral (the leptons' own charge)", readme_sink(1) - counted[1]))
    return out


@derivation("ann.source")
def check_source():
    """The photon source counts the annihilations' photons: two per annihilation, with the energy the sinks remove."""
    # per annihilation of an e+ of gamma_+ and an e- of gamma_-, (v dsigma/dk)/<sigma v> photons per unit k (both photons)
    counted = NPOS(gp) * NELE(gm) * SV(gp, gm) * (VDS(k, gp, gm) / SV(gp, gm))
    a, b = lorentz(w1), lorentz(w2)
    number, energy = readme_counts(a, b)
    rate = AN * BN * readme_rate(a, b)                          # the annihilations on the lines (ann.sinks)
    # the photons on the lines: int dk of the source, A B int v dsigma/dk dk, by the README's normalisation (ann.norm)
    made = on_lines(readme_source(), gp, gm, a, b).subs(VDS(k, a, b), sp.Symbol("Vint"))
    made_n, made_e = (made.subs(sp.Symbol("Vint"), q * readme_rate(a, b)) for q in (number, energy))
    lost = -sum(on_lines(g * readme_sink(s, lambda u, v: readme_rate(u, v)), g, g1, a, b) for s in (1, -1))
    out = [exact("README source = the annihilations n_+ n_- <sigma v> times their photons (v dsigma/dk)/<sigma v> (any "
                 "<sigma v> and spectrum; per unit gamma_+, gamma_-)", readme_source() - counted),
           exact("lines (A at gamma_1, B at gamma_2): photons made = 2 x the annihilations of the sinks", made_n - 2 * rate),
           exact("their energy = the energy of the e- and e+ the sinks remove, (gamma_1 + gamma_2) A B <sigma v>", made_e - lost)]
    with L.patched(r"n_+(\gamma_+)\thinspace n_-(\gamma_-)", r"n_+(\gamma_+)\thinspace n_+(\gamma_-)"):
        out.append(exact_fails("control: n_+(gamma_-) for n_-(gamma_-)", readme_source() - counted))
    with L.patched(r"$= (\gamma_+ + \gamma_-)\langle\sigma v\rangle$", r"$= 2(\gamma_+ + \gamma_-)\langle\sigma v\rangle$"):
        out.append(exact_fails("control: photon energy 2 (gamma_+ + gamma_-) <sigma v>",
                               made.subs(sp.Symbol("Vint"), readme_counts(a, b)[1] * readme_rate(a, b)) - lost))
    return out


def norm_claim():
    """(precision, digits) of the normalisation as tests/literature_checks/README.md states it (Results)."""
    m = re_search(r"The normalisation of the spectrum \(to \$([\d.]+) \\times 10\^\{(-\d+)\}\$, in (\d+)-digit quadrature\)",
                  L.line("The normalisation of the spectrum (to", "checks"))
    return mp.mpf(m.group(1)) * mp.mpf(10)**int(m.group(2)), int(m.group(3))


@derivation("ann.norm")
def check_norm():
    """The README's spectrum integrates to 2 <sigma v> photons with the energy (gamma_+ + gamma_-) <sigma v>."""
    L.line(r"which counts both photons: $\int v\ \frac{d\sigma}{dk}\ dk = 2\langle\sigma v\rangle$ and "
           r"$\int k\ v\ \frac{d\sigma}{dk}\ dk$ $= (\gamma_+ + \gamma_-)\langle\sigma v\rangle$.")
    spec = spectrum_function("README")
    rate = _lambdify((gp, gm), readme_rate())
    ks = {n: _lambdify((gp, gm), readme_k(n)) for n in ("k_L", "k_-", "k_+", "k_U")}
    pairs = [(mp.mpf(a), mp.mpf(b)) for a, b in ((1.5, 1.02), (3, 7), (20, 2.5), (1e3, 30), (1e4, 1e4 * 1.1))]
    out = []
    stated, digits = norm_claim()
    with mp.workdps(digits):
        number, energy = [], []
        for a, b in pairs:
            nodes = sorted(f(a, b) for f in ks.values())
            n_ = mp.quad(lambda kk: spec(kk, a, b), nodes)
            e_ = mp.quad(lambda kk: kk * spec(kk, a, b), nodes)
            number.append(abs(n_ / (2 * rate(a, b)) - 1))
            energy.append(abs(e_ / ((a + b) * rate(a, b)) - 1))

    def results(stated):
        return [Result(what, mp.nstr(max(errs), 2) == mp.nstr(stated, 2),
                       f"numeric: quadrature ({digits} digits) within {mp.nstr(max(errs), 3)} for 5 pairs, gamma 1.02 to 1.1e4 "
                       f"(tests/literature_checks/README.md: {mp.nstr(stated, 2)})")
                for what, errs in (("photon number = 2 <sigma v>", number),
                                   ("photon energy = (gamma_+ + gamma_-) <sigma v>", energy))]
    out = results(stated)
    with L.patched(r"(to $1.4 \times 10^{-30}$, in 30-digit", r"(to $1.4 \times 10^{-31}$, in 30-digit", "checks"):
        bad = [r.what for r in results(norm_claim()[0]) if not r.ok]
        out.append(Result("control: 1.4e-31 stated", bool(bad), "control: fails, as it must" if bad else "control: passes"))
    return out


@derivation("ann.kLU")
def check_kLU():
    """k_L and k_U are Svensson's eqs. 33d and 33a, where gamma_s = gamma_ast^min (eqs. 17b and 31)."""
    a, b = lorentz(w1), lorentz(w2)
    out = []
    for n, e in (("k_L", "33d"), ("k_U", "33a")):
        out.append(exact(f"{n} = eq. {e}", (readme_k(n) - sv_k(n)).subs({gp: a, gm: b})))
        out.append(exact(f"gamma_s^2({n}) = (gamma_ast^min)^2", (readme_gs2(readme_k(n)) - readme_gstar(-1)**2).subs({gp: a, gm: b})))
    return out


@derivation("ann.peaks")
def check_peaks():
    """k_+- are Svensson's eqs. 33b-c, where gamma_s = gamma_ast^max; the spectrum peaks there, equally."""
    a, b = lorentz(w1), lorentz(w2)
    out = []
    for n, e in (("k_+", "33b"), ("k_-", "33c")):
        out.append(exact(f"{n} = eq. {e}", (readme_k(n) - sv_k(n)).subs({gp: a, gm: b})))
        out.append(exact(f"gamma_s^2({n}) = (gamma_ast^max)^2", (readme_gs2(readme_k(n)) - readme_gstar(1)**2).subs({gp: a, gm: b})))
    spec = spectrum_function("README")
    kp, km = (_lambdify((gp, gm), readme_k(n)) for n in ("k_+", "k_-"))
    peaks, same = True, mp.mpf(0)
    with mp.workdps(50):
        for a_, b_ in ((2, 1.3), (10, 3), (300, 50), (5e3, 20)):
            a_, b_ = mp.mpf(a_), mp.mpf(b_)
            for kf in (kp, km):
                k0 = kf(a_, b_)
                peaks &= spec(k0, a_, b_) > max(spec(k0 * (1 - mp.mpf(10)**-6), a_, b_),
                                                spec(k0 * (1 + mp.mpf(10)**-6), a_, b_))
            same = max(same, abs(spec(kp(a_, b_), a_, b_) / spec(km(a_, b_), a_, b_) - 1))
    out.append(Result("local maxima at k_+- with equal values (Svensson p. 331, i)", peaks and same < mp.mpf(10)**-40,
                      f"numeric: 4 pairs, values equal within {mp.nstr(same, 3)}"))
    return out


@derivation("ann.rest")
def check_rest():
    """The README's eq. 41 and its range are Svensson's eq. 41 (gamma_r = gamma, beta_r gamma_r^2 = p gamma)."""
    spec, lo, hi = readme_eq41()
    s_spec, s_lo, s_hi = sv_eq41()
    gv = lorentz(w1)
    return [exact("v dsigma/dk = eq. 41", (spec - s_spec).subs(g, gv)),
            exact("lower limit = [gamma_r(1 - beta_r) + 1]/2", (lo - s_lo).subs(g, gv)),
            exact("upper limit = [gamma_r(1 + beta_r) + 1]/2", (hi - s_hi).subs(g, gv))]


@derivation("ann.rest_primitives")
def check_rest_primitives():
    """N(k) and E(k) are primitives of eq. 41 and of k times eq. 41."""
    defs = {}
    tokens = {"k": k, r"\gamma": g, "p": p_of(g), r"\sigma_T": SIGMA_T}
    for name, tex in (("D", r"D = 1 + \gamma"), ("B", r"B = 3 + \gamma"), ("A", r"A = -B/D"),
                      ("C", r"C = 3\sigma_T/(8p\gamma)"), ("L", r"L = \ln k - \ln(D - k)")):
        defs[name] = L.to_sympy(L.sides(L.inline(tex, "checks"))[1], {**tokens, **defs})
    N = L.to_sympy(L.sides(L.inline(r"N(k) = C[1/k + (B - 2/D) L - 1/(D - k) - 2k]", "checks"))[1], {**tokens, **defs})
    E = L.to_sympy(L.sides(L.inline(r"E(k) = C[A D^2 \ln(D - k) - L - D/(D - k) - k^2]", "checks"))[1], {**tokens, **defs})
    spec = readme_eq41()[0] / c
    gv = lorentz(w1)
    return [exact("dN/dk = eq. 41/c", (sp.diff(N, k) - spec).subs(g, gv)),
            exact("dE/dk = k eq. 41/c", (sp.diff(E, k) - k * spec).subs(g, gv))]


@derivation("ann.three_photon")
def check_three_photon():
    """4(pi^2 - 9) alpha/(3 pi) is BLP's sigma_3gamma/sigma_2gamma (eqs. 89.6, 89.1), about 1/371."""
    al, v_ = sp.symbols("alpha v", positive=True)
    tex = L.inline(r"4(\pi^2 - 9)\alpha/(3\pi) \approx 1/371", "checks")
    form = L.to_sympy(tex.split(r"\approx")[0], {r"\alpha": al})
    s2 = sp.pi * re**2 * c / v_                                          # (89.1), e^2/mc^2 = r_e
    s3 = 4 * (sp.pi**2 - 9) * c / (3 * v_) * al * re**2                 # (89.6)
    value = form.subs(al, 1 / sp.Float(INV_ALPHA, 30))
    return [exact("= sigma_3gamma/sigma_2gamma (BLP eqs. 89.6, 89.1)", form - s3 / s2),
            Result("about 1/371", abs(1 / value - 371) < 0.5, f"1/{sp.N(1 / value, 6)}")]


@derivation("ann.coppi")
def check_coppi():
    """ATHEnA's rate is Mastichiadis & Kirk's eq. 59, R_ann = 3/(8 gamma) [gamma^-1/2 + ln gamma]."""
    form = L.to_sympy(L.inline(r"3/(8\gamma)\thinspace[\gamma^{-1/2} + \ln\gamma]"), {r"\gamma": g})
    mk = sp.Rational(3, 8) / g * (g**sp.Rational(-1, 2) + sp.log(g))    # M&K 1995, eq. 59 (p. 8 of the arXiv PDF)
    return [exact("= eq. 59", form - mk)]


_RATE = {}


def rate_mp(a, b):
    """The README's <sigma v>(gamma_+, gamma_-) (c = r_e = 1) at 40 digits. Equal Lorentz factors are split by 1e-12
    (relatively), as their gamma_r^- = 1 could round below 1."""
    key = (RATE(), GLINE(), SIGMA(), P_DEF())
    if key not in _RATE:
        _RATE[key] = _lambdify((gp, gm), readme_rate())
    with mp.workdps(40):
        a, b = mp.mpf(a), mp.mpf(b)
        return +_RATE[key](a, b * (1 + mp.mpf(10)**-12) if a == b else b)


def results_line():
    """The bullet 'The treatments of the other codes' of the Results of tests/literature_checks/README.md."""
    return L.line("The treatments of the other codes were compared with the exact result", "checks")


def golden(f, lo, hi, sign):
    """The extremum of f between lo and hi (a maximum for sign +1, a minimum for -1), by golden-section search."""
    r = (mp.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(60):
        x1, x2 = b - r * (b - a), a + r * (b - a)
        if sign * f(x1) > sign * f(x2):
            b = x2
        else:
            a = x1
    return (a + b) / 2


@derivation("ann.coppi_error")
def check_coppi_error():
    """ATHEnA's rate fit against the README's rate on a lepton at rest: -14 % at gamma = 2.2, +6 % at gamma ~ 100, its extremes."""
    def claims():
        return re_search(r"This rate is \$(-\d+)\\text\{％\}\$ at \$\\gamma = ([\d.]+)\$ and \$\+(\d+)\\text\{％\}\$ at "
                         r"\$\\gamma \\approx (\d+)\$", results_line()).groups()
    fit = sp.lambdify(g, L.to_sympy(L.inline(r"3/(8\gamma)\thinspace[\gamma^{-1/2} + \ln\gamma]", "checks"), {r"\gamma": g}),
                      "mpmath")
    with mp.workdps(20):
        def r(u):   # u = ln gamma; the README's rate with gamma_- = 1 + 1e-30 (one lepton at rest), sigma_T c = 1
            gam = mp.exp(u)
            with mp.workdps(40):
                rest = rate_mp(gam, 1 + mp.mpf(10)**-30)
            return fit(gam) * 8 * mp.pi / 3 / rest - 1
        us = [mp.log(mp.mpf("1.001")) + (mp.log(mp.mpf(10)**8) - mp.log(mp.mpf("1.001"))) * i / 160 for i in range(161)]
        vals = [r(u_) for u_ in us]
        i_min, i_max = vals.index(min(vals)), vals.index(max(vals))
        u_min, u_max = (golden(r, us[i - 1], us[i + 1], s_) for i, s_ in ((i_min, -1), (i_max, 1)))
        g_min, g_max, r_min, r_max = mp.exp(u_min), mp.exp(u_max), r(u_min), r(u_max)

    def result(lo, g_lo, hi, g_hi):
        return Result(f"{lo} % at gamma = {g_lo} (the minimum) and +{hi} % at gamma ~ {g_hi} (the maximum), gamma 1.001 to 1e8",
                      f"{100 * float(r_min):.0f}" == lo and f"{float(g_min):.2g}" == g_lo and f"{100 * float(r_max):.0f}" == hi
                      and float(f"{float(g_max):.1g}") == float(g_hi) and float(vals[0]) > float(r_min) and float(vals[-1]) < float(r_max),
                      f"numeric: {mp.nstr(100 * r_min, 3)} % at gamma = {mp.nstr(g_min, 3)}, {mp.nstr(100 * r_max, 3)} % at "
                      f"gamma = {mp.nstr(g_max, 3)}")
    out = [result(*claims())]
    with L.patched(r"This rate is $-14\text{％}$", r"This rate is $-12\text{％}$", "checks"):
        broken = result(*claims())
    out.append(Result("control: -12 %", not broken.ok, "control: fails, as it must" if not broken.ok else "control: passes, but must fail"))
    return out


def bs_claims():
    """(the excess of eq. 16's photons, low and high in %; the Lorentz factors, low and high; the rates of eqs. 6 and 33 in %, at gamma)."""
    a_ = re_search(r"has \$(\d+)\\text\{–\}(\d+)\\text\{％\}\$ too many photons for Lorentz factors from (\d+) to \$10\^(\d+)\$",
                   results_line()).groups()
    b_ = re_search(r"The rates of their eqs\. 6 and 33 are \$(-\d+)\\text\{％\}\$ and \$(-\d+)\\text\{％\}\$ at \$\\gamma = (\d+)\$",
                   results_line()).groups()
    return a_[:2], (int(a_[2]), 10**int(a_[3])), b_


def bs_nodes(a, b):
    """The kinks and branch points of eqs. 6 and 16 inside [k_L, k_U]: k_-+, gamma_+-, gamma_+- -+ 1 (c_+- = 0), S/2."""
    lo, hi = (a - bs_p(a) + b - bs_p(b)) / 2, bs_kU(a, b)
    inner = [(a - bs_p(a) + b + bs_p(b)) / 2, (a + bs_p(a) + b - bs_p(b)) / 2, a, b, (a + b) / 2, a - 1, a + 1, b - 1, b + 1]
    return [lo] + sorted(set(p_ for p_ in inner if lo < p_ < hi)) + [hi]


@derivation("ann.bs96")
def check_bs96():
    """Boettcher & Schlickeiser's eq. 16 has 3-15 % too many photons (Lorentz factors 5 to 1e4); the rates of their eqs. 6 and 33 are -12 % and -17 % at gamma = 5."""
    kk = sp.Symbol("k", positive=True)
    a, b = sp.symbols("a b", positive=True)
    eq16 = lambda q: (sp.Min(sp.Symbol("G2"), q * (a + b - q)) / (sp.Abs(q - a) + 2 / sp.pi)   # noqa: E731
                      + sp.Min(sp.Symbol("G2"), q * (a + b - q)) / (sp.Abs(q - b) + 2 / sp.pi))
    symmetric = sp.simplify(eq16(a + b - kk) - eq16(kk)) == 0
    (lo_s, hi_s), (g_lo, g_hi), (r6_s, r33_s, g5) = bs_claims()
    grid = [mp.mpf(g_lo) * (mp.mpf(g_hi) / g_lo)**(mp.mpf(i) / 12) for i in range(13)]
    excess = []
    with mp.workdps(15):
        for i, a_ in enumerate(grid):
            for b_ in grid[i:]:
                n16 = mp.quad(lambda q: bs_eq16(q, a_, b_), bs_nodes(a_, b_))
                excess.append((n16 / (2 * rate_mp(a_, b_)) - 1, a_, b_))
        excess.sort()
        a5 = mp.mpf(g5)
        r6 = mp.quad(lambda q: bs_eq6(q, a5, a5), bs_nodes(a5, a5)) / 2 / rate_mp(a5, a5) - 1
        r33 = bs_eq33(a5, a5) / rate_mp(a5, a5) - 1

    def results(lo_s, hi_s, r6_s, r33_s):
        lo, hi = excess[0], excess[-1]
        return [Result(f"eq. 16: {lo_s}-{hi_s} % too many photons, N/(2 <sigma v>) - 1 on a grid of 91 pairs with gamma_+- "
                       f"from {g_lo} to {g_hi:.0e}", f"{100 * float(lo[0]):.0f}" == lo_s and f"{100 * float(hi[0]):.0f}" == hi_s,
                       f"numeric: {mp.nstr(100 * lo[0], 3)} % at ({mp.nstr(lo[1], 3)}, {mp.nstr(lo[2], 3)}), "
                       f"{mp.nstr(100 * hi[0], 3)} % at ({mp.nstr(hi[1], 3)}, {mp.nstr(hi[2], 3)})"),
                Result(f"the rates of eqs. 6 (half its photons) and 33: {r6_s} % and {r33_s} % at gamma_+ = gamma_- = {g5}",
                       f"{100 * float(r6):.0f}" == r6_s and f"{100 * float(r33):.0f}" == r33_s,
                       f"numeric: {mp.nstr(100 * r6, 3)} %, {mp.nstr(100 * r33, 3)} %")]
    out = [in_source("eqs. 4-7, 9-10, 12, 16 and 33 (Svensson's eq. 19b) as transcribed", BS96, *BS_TEX),
           Result("eq. 16 is symmetric under k <-> gamma_+ + gamma_- - k: its photons carry (gamma_+ + gamma_-)/2 on "
                  "average, so its photon energy is off as its photon number", symmetric, "exact: symmetric")]
    out += results(lo_s, hi_s, r6_s, r33_s)
    with L.patched(r"$3\text{–}15\text{％}$ too many photons", r"$3\text{–}13\text{％}$ too many photons", "checks"):
        (lo_b, hi_b), _, (r6_b, r33_b, _) = bs_claims()
        bad = [r_.what for r_ in results(lo_b, hi_b, r6_b, r33_b) if not r_.ok]
        out.append(Result("control: 3-13 % (the range of 9 sampled pairs)", bool(bad),
                          "control: fails, as it must" if bad else "control: passes, but must fail"))
    return out


if __name__ == "__main__":
    run()
