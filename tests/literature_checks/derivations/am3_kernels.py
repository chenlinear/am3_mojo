"""Derivation checks of AM3's synchrotron and inverse-Compton kernels in tests/reference/README.md (Misprints
of the AM3 paper; Kept as in AM3: the quantum suppression of synchrotron cooling; Bugs: quantum-synchrotron
cooling of heavy particles) against their sources:
  - Klinger et al. 2024 (the AM3 paper), appendix A of the TeX source of arXiv:2312.13371v2, copied and parsed:
    sigma_T and B_crit of a particle of mass m_X, the synchrotron kernels and cooling, and the inverse-Compton
    kernels of e+- and hadrons (with the head-on cross section of Dermer & Menon 2009);
  - Gao, Pohl & Winter 2017, the TeX source of arXiv:1610.05306v2 (appendix, inverse Compton);
  - Berestetskii, Lifshitz & Pitaevskii (BLP), Quantum Electrodynamics, 2nd ed., §90, and Blumenthal & Gould
    1970, p. 243, transcribed below from their scans (BLP: PDF page = book page + 17);
  - the C++ AM3 source (am3-master.zip in the repository root), for what "the code has".

BLP put c = 1 and keep hbar (§90). Their epsilon is the electron's energy gamma m, hbar omega the photon's
(the README's epsilon m c^2), epsilon' = epsilon - hbar omega, H the field, H_0 = m^2c^3/(|e| hbar) = 4.4e13 G
(the README's B_cr) and chi = (H/H_0)|p|/m ~ H epsilon/(H_0 m) = gamma B/B_cr (eq. 90.3, p. 376). Phi is the
Airy function of QM §b, Phi = sqrt(pi) Ai (checked through their Phi'(0)):
  p. 382    I_cl = 2 e^2 m^2 chi^2/(3 hbar^2), the classical intensity;
  (90.23)   dI/domega = -(e^2 m^2 omega/(sqrt(pi) epsilon^2)) {int_x^oo Phi(xi) dxi
                        + (2/x + (hbar omega/epsilon) chi sqrt(x)) Phi'(x)},
  (90.24)   x = (hbar omega/(epsilon' chi))^(2/3);
  (90.25, p. 383)  I = -(e^2 m^2 chi^2/(2 sqrt(pi) hbar^2)) int_0^oo (4 + 5 chi x^(3/2) + 4 chi^2 x^3)
                        /(1 + chi x^(3/2))^4 Phi'(x) x dx;
  p. 383    int_0^oo x^nu Phi'(x) dx = -(1/(2 sqrt(pi))) 3^((4 nu - 1)/6) Gamma(nu/3 + 1) Gamma(nu/3 + 1/3),
            Phi'(0) = -3^(1/6) Gamma(2/3)/(2 sqrt(pi));
  (90.26)   I = I_cl (1 - (55 sqrt(3)/16) chi + 48 chi^2 - ...);
  (90.27, p. 384)  I ~ (32 Gamma(2/3) e^2 m^2/(243 hbar^2)) (3 chi)^(2/3) = 0.37 (e^2 m^2/hbar^2)(H eps/(H_0 m))^(2/3).
BLP print I, not I/I_cl: the README's 0.556 chi^(-4/3) is eq. 90.27 divided by I_cl.
Blumenthal & Gould (p. 243): eq. 2.47 eps_1 = gamma m c^2 E_1; eq. 2.48 dN/(dt dE_1) = (2 pi r_0^2 m c^3/gamma)
(n(eps) deps/eps) [2q ln q + (1 + 2q)(1 - q) + (1/2)(Gamma_eps q)^2 (1 - q)/(1 + Gamma_eps q)]; eq. 2.49
Gamma_eps = 4 eps gamma/(m c^2), q = E_1/(Gamma_eps (1 - E_1)).

Found besides the README's misprints: the AM3 paper prints the total inverse-Compton rate (the integral of
its kernel c sigma_T,e R~) without sigma_T,e, and calls R~ Blumenthal & Gould's "Eq. 2.49", though it is
their eq. 2.48. Gao et al. define u = 2 gamma_e,i, without the target photon's energy. The paper's R_syn
counts photons per unit time and ln(eps), as the code does (its power is then the cooling for electrons).

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/am3_kernels.py
"""
import re
import zipfile

import mpmath as mp
import sympy as sp

import latex as L
from latex import GAO, KLINGER
from pdg import INV_ALPHA
from verify import Result, control, derivation, exact, exact_fails, in_source, log_uniform, numeric, rng, run

CODE = L.ROOT / "am3-master.zip"
chi, s, t, z, w, u, v, g, eps, eo = sp.symbols("chi s t z w u v gamma varepsilon varepsilon_o", positive=True)
mX, me, c, B, e, hb, FB, X, XI, Y, sT, Rh = sp.symbols("m_X m_e c B e hbar F_B x xi y sigma_T Rhat", positive=True)
# CODATA 2022 (SI; e, c, h and the eV exact), for the Gaussian-unit numbers
E_SI, C_SI, H_SI, ME_SI = (sp.Rational(n) for n in ("1.602176634e-19", "299792458", "6.62607015e-34", "9.1093837139e-31"))
MP_ME = sp.Rational("1836.152673426")   # m_p/m_e


def in_code(what, member, *snippets):
    """The snippets occur in the C++ file `member` of am3-master.zip (ignoring whitespace). Without the zip, the
    check says so."""
    if not CODE.exists():
        return Result(what, True, "not re-checked: am3-master.zip is not in the repository root")
    with zipfile.ZipFile(CODE) as zf:
        flat = re.sub(r"\s+", "", zf.read("am3-master/" + member).decode())
    found = all(re.sub(r"\s+", "", sn) in flat for sn in snippets)
    return Result(what, found, f"in {member} of am3-master.zip" if found else f"not found in {member}")


def exact_gamma(what, difference):
    """exact, with every Gamma(n/3) written as a rational times Gamma(1/3) or Gamma(2/3), and
    Gamma(2/3) = 2 pi/(sqrt3 Gamma(1/3)) (the reflection formula, which sympy does not apply across terms)."""
    third = sp.gamma(sp.Rational(1, 3))
    return exact(what, sp.expand_func(difference).subs(sp.gamma(sp.Rational(2, 3)), 2 * sp.pi / (sp.sqrt(3) * third)))


def exact_all(what, *differences):
    """exact for each of the differences."""
    bad = [i for i, d in enumerate(differences) if not exact("", d).ok]
    return Result(what, not bad, "exact: 0" if not bad else f"exact: not zero (pieces {bad})")


def pinned(context, span):
    """The inline span $span$, asserted in the README line that holds `context` (for spans that occur twice)."""
    assert "$" + span + "$" in L.line(context, "reference"), span
    return span


def percents(span):
    """The percentage of a README span "2\\text{％}", or the range of "7\\text{–}10\\text{％}"."""
    return [float(p) for p in span.replace(r"\text{％}", "").split(r"\text{–}")]


# --- the quantum suppression of synchrotron cooling (Kept as in AM3) ---

QS_PIN = r"it tends to $\chi^{-4/3}/2$, their eq. 90.27 to $0.556\ \chi^{-4/3}$"


def am3_suppression():
    """AM3's factor, from tests/reference/README.md (Kept as in AM3)."""
    span = r"1/(1 + \sqrt{2}\thinspace \chi^{2/3})^2"
    return L.to_sympy(pinned("The quantum suppression of synchrotron cooling, $" + span + "$", span), {r"\chi": chi})


def blp_mellin(nu):
    """BLP p. 383: int_0^oo x^nu Phi'(x) dx."""
    nu = sp.sympify(nu)
    return -3**((4 * nu - 1) / 6) * sp.gamma(nu / 3 + 1) * sp.gamma(nu / 3 + sp.Rational(1, 3)) / (2 * sp.sqrt(sp.pi))


def blp_over_icl():
    """Over I_cl (p. 382): the prefactor of eq. 90.25's integral, eq. 90.27 times chi^(4/3), and the
    coefficient of (e^2 m^2/hbar^2) chi^(2/3) in eq. 90.27 (its second form rounds it to 0.37)."""
    em, mm, hh = sp.symbols("e m hbar", positive=True)
    i_cl = 2 * em**2 * mm**2 * chi**2 / (3 * hh**2)
    pre_25 = -em**2 * mm**2 * chi**2 / (2 * sp.sqrt(sp.pi) * hh**2)
    i_27 = sp.Rational(32, 243) * sp.gamma(sp.Rational(2, 3)) * em**2 * mm**2 / hh**2 * (3 * chi)**sp.Rational(2, 3)
    return (sp.simplify(pre_25 / i_cl), sp.simplify(i_27 / i_cl * chi**sp.Rational(4, 3)),
            sp.simplify(i_27 * hh**2 / (em**2 * mm**2 * chi**sp.Rational(2, 3))))


BLP_F = (4 + 5 * t + 4 * t**2) / (1 + t)**4          # eq. 90.25's integrand without x Phi'(x), t = chi x^(3/2)
BLP_PRE, BLP_27, BLP_037 = blp_over_icl()           # I/I_cl = BLP_PRE int_0^oo x Phi'(x) F(chi x^(3/2)) dx
BLP_PHI0 = -3**sp.Rational(1, 6) * sp.gamma(sp.Rational(2, 3)) / (2 * sp.sqrt(sp.pi))   # p. 383


def blp_ratio(chi_, dps=20):
    """I/I_cl of eq. 90.25 by quadrature (Phi' = sqrt(pi) Ai')."""
    pre = BLP_PRE * sp.sqrt(sp.pi)                    # -3/4
    with mp.workdps(dps):
        x_ = mp.mpf(chi_)
        f = lambda q: q * mp.airyai(q, derivative=1) * (4 + 5 * x_ * q**1.5 + 4 * x_**2 * q**3) / (1 + x_ * q**1.5)**4
        knee = x_**(-mp.mpf(2) / 3)                   # where chi x^(3/2) = 1
        nodes = sorted({mp.mpf(0), knee / 10, knee, 10 * knee, mp.mpf(1), mp.mpf(5)})
        return mp.mpf(pre.p) / pre.q * mp.quad(f, [n for n in nodes if n < 20] + [mp.mpf(20), mp.inf])


def blp_large_chi():
    """The coefficient of chi^(-4/3) in eq. 90.25 / I_cl for chi >> 1, with Phi'(x) -> Phi'(0) (BLP, p. 383):
    x = (t/chi)^(2/3) turns int x F(chi x^(3/2)) dx into (2/3) chi^(-4/3) int_0^oo t^(1/3) F(t) dt."""
    return BLP_PRE * BLP_PHI0 * sp.Rational(2, 3) * sp.integrate(t**sp.Rational(1, 3) * BLP_F, (t, 0, sp.oo))


def limit_result(what, span=r"\chi^{-4/3}/2"):
    lim = L.to_sympy(L.inline(span, "reference"), {r"\chi": chi})
    ratio = sp.limit(am3_suppression() / lim, chi, sp.oo)
    return Result(what, ratio == 1, f"exact: lim AM3/README = {ratio}")


@derivation("am3.qs_limits")
def check_qs_limits():
    """AM3's factor tends to chi^(-4/3)/2, BLP's eq. 90.25 to 0.556 chi^(-4/3) (eq. 90.27 / I_cl): 10% more."""
    assert "(§90, eq. 90.25)" in L.line(QS_PIN, "reference")
    const = blp_large_chi()
    readme_const = float(L.to_sympy(L.inline(r"0.556\ \chi^{-4/3}", "reference"), {r"\chi": 1}))
    out = [limit_result("AM3's 1/(1 + sqrt2 chi^(2/3))^2 -> chi^(-4/3)/2"),
           in_code("the code has 1/(1 + sqrt2 x^(2/3))^2", "src/Solver.cc",
                   "Asy[i] *= 1/pow((1+sqrt(2)*pow(x, (2.0/3.0))), 2.0);"),
           exact_gamma("BLP's Phi'(0) = sqrt(pi) Ai'(0) (Phi = sqrt(pi) Ai)", sp.sqrt(sp.pi) * sp.airyaiprime(0) - BLP_PHI0),
           exact_gamma("eq. 90.25 / I_cl for chi >> 1 (Phi' -> Phi'(0)) = eq. 90.27 / I_cl = (16/81) 3^(2/3) Gamma(2/3) chi^(-4/3)",
                       const - BLP_27),
           Result("= 0.556 chi^(-4/3) (README); eq. 90.27's second form has 0.37 = 32 Gamma(2/3) 3^(2/3)/243",
                  round(float(const), 3) == readme_const and round(float(BLP_037), 2) == 0.37,
                  f"{float(const):.6f} and {float(BLP_037):.4f}")]
    d6, d8 = (abs(blp_ratio(x_) * mp.mpf(x_)**(mp.mpf(4) / 3) - float(const)) for x_ in (1e6, 1e8))
    out.append(Result("eq. 90.25 by quadrature: chi^(4/3) I/I_cl -> it as chi^(-2/3)", d8 < 1e-5 and 19 < d6 / d8 < 24,
                      f"off by {mp.nstr(d6, 3)} at chi = 1e6, {mp.nstr(d8, 3)} at 1e8"))
    # "too slow by" = 1 - AM3/exact, rounded to whole percents
    am3_f = sp.lambdify(chi, am3_suppression(), "mpmath")
    slow = lambda x_: 100 * (1 - am3_f(mp.mpf(x_)) / blp_ratio(x_))
    at_1e3 = percents(pinned(r"too slow by $2\text{％}$ at", r"2\text{％}"))[0]
    x3 = float(L.to_sympy(L.sides(L.inline(r"\chi = 10^{-3}", "reference"))[1], {}))
    out.append(Result("too slow by 2% at chi = 1e-3", round(slow(x3)) == at_1e3, f"{mp.nstr(slow(x3), 3)}%"))
    for span, rng_span in ((r"7\text{–}10\text{％}", r"0.01 \le \chi \le 0.3"), (r"5\text{–}6\text{％}", r"1 \le \chi \le 10")):
        lo_p, hi_p = percents(L.inline(span, "reference"))
        lo, _, hi = (float(L.to_sympy(p_, {r"\chi": 0})) for p_ in re.split(r"\\le(?![A-Za-z])", L.inline(rng_span, "reference")))
        vals = [slow(lo * (hi / lo)**(k / 30)) for k in range(31)]
        out.append(Result(f"too slow by {lo_p:g}-{hi_p:g}% for {lo:g} <= chi <= {hi:g}",
                          round(min(vals)) == lo_p and round(max(vals)) == hi_p,
                          f"{mp.nstr(min(vals), 3)}% to {mp.nstr(max(vals), 3)}% (31 points)"))
    inf_p = percents(pinned(r"and $10\text{％}$ as $\chi \to \infty$", r"10\text{％}"))[0]
    out.append(Result("too slow by 10% as chi -> oo: 1 - (1/2)/0.5564", round(100 * (1 - 0.5 / float(const))) == inf_p,
                      f"{100 * (1 - 0.5 / float(const)):.2f}%"))
    with L.patched(r"$\chi^{-4/3}/2$", r"$\chi^{-4/3}/3$", "reference"):
        out.append(Result("control: chi^(-4/3)/3 in the README", not limit_result("", r"\chi^{-4/3}/3").ok,
                          "the limit check fails, as it must"))
    return out


def readme_small(span=r"1 - (55\sqrt{3}/16)\thinspace \chi"):
    return L.to_sympy(L.inline(span, "reference"), {r"\chi": chi})


@derivation("am3.qs_small")
def check_qs_small():
    """AM3's factor starts as 1 - 2 sqrt2 chi^(2/3); eq. 90.25 as 1 - (55 sqrt3/16) chi (eq. 90.26)."""
    am3_series = sp.series(am3_suppression().subs(chi, s**3), s, 0, 4).removeO()     # chi = s^3, to O(chi^(4/3))
    readme_am3 = L.to_sympy(L.inline(r"1 - 2\sqrt{2}\thinspace \chi^{2/3}", "reference"), {r"\chi": s**3})
    coeffs = [sp.gammasimp(BLP_PRE * sp.series(BLP_F, t, 0, 3).removeO().coeff(t, n) * blp_mellin(1 + sp.Rational(3, 2) * n))
              for n in range(3)]
    series = sum(cf * chi**n for n, cf in enumerate(coeffs))
    first = series.coeff(chi, 0) + series.coeff(chi, 1) * chi
    with mp.workdps(30):
        mellin = max(abs(mp.quad(lambda q: q**mp.mpf(float(nu)) * mp.sqrt(mp.pi) * mp.airyai(q, derivative=1),
                                 [0, 1, 5, 20, mp.inf]) / mp.mpf(sp.N(blp_mellin(nu), 40)) - 1)
                     for nu in (sp.Integer(1), sp.Rational(5, 2), sp.Integer(4)))
    x4 = float(L.to_sympy(L.sides(L.inline(r"\chi = 10^{-4}", "reference"))[1], {}))
    exact_x4 = blp_ratio(x4, 30)
    off = exact_x4 - mp.mpf(sp.N(series.subs(chi, sp.Rational(1, 10**4)), 40))
    am3_pct, true_pct = 100 * (1 - float(am3_suppression().subs(chi, x4))), float(100 * (1 - exact_x4))
    with L.patched(r"(55\sqrt{3}/16)", r"(55\sqrt{3}/15)", "reference"):
        broken = readme_small(r"1 - (55\sqrt{3}/15)\thinspace \chi")
    return [exact("AM3's factor = README's 1 - 2 sqrt2 chi^(2/3) + O(chi^(4/3))", am3_series - readme_am3),
            Result("BLP's formula for int x^nu Phi'(x) dx (p. 383), nu = 1, 5/2, 4", mellin < 1e-25,
                   f"numeric: quadrature within {mp.nstr(mellin, 3)} (30 digits)"),
            exact("eq. 90.25 expanded in chi with it: 1 - (55 sqrt3/16) chi + 48 chi^2 (eq. 90.26)",
                  series - (1 - 55 * sp.sqrt(3) / 16 * chi + 48 * chi**2)),
            exact("= README's 1 - (55 sqrt3/16) chi to first order", first - readme_small()),
            Result("eq. 90.25 by quadrature at chi = 1e-4 = eq. 90.26 + O(chi^3)", abs(off) < 1e-9,
                   f"differs by {mp.nstr(off, 3)}"),
            Result("suppression at chi = 1e-4: AM3 0.6%, exact 0.06%",
                   (round(am3_pct, 1), round(true_pct, 2)) == (percents(L.inline(r"0.6\text{％}", "reference"))[0],
                                                               percents(L.inline(r"0.06\text{％}", "reference"))[0]),
                   f"{am3_pct:.3f}% and {true_pct:.4f}%"),
            exact_fails("control: 55 sqrt3/15 in the README", first - broken)]


@derivation("am3.qs_const")
def check_qs_const():
    """m_e c^2/B_cr = e hbar/(m_e c) = 1.158e-8 eV/G (B_cr = m^2c^3/(e hbar), CODATA 2022); AM3's 1.2e-8 is 3.7% high."""
    m, cc, ee, h_ = sp.symbols("m c e hbar", positive=True)
    tokens = {"m": m, "c": cc, "e": ee, r"\hbar": h_}
    bcr = L.to_sympy(L.sides(L.inline(r"B_\mathrm{cr} = m^2c^3/(e\hbar) \propto m^2", "reference").split(r"\propto")[0])[1], tokens)
    lhs, rhs = L.sides(L.inline(r"m_e c^2/B_\mathrm{cr} = 1.158 \times 10^{-8}", "reference"))
    ratio = L.to_sympy(lhs, {"m_e": m, "c": cc, r"B_\mathrm{cr}": bcr})
    gauss = {m: ME_SI * 1000, cc: C_SI * 100, ee: E_SI * C_SI * 10, h_: H_SI * 10**7 / (2 * sp.pi)}   # g, cm/s, statC, erg s
    ev_g = float(ratio.subs(gauss) / (E_SI * 10**7))      # erg/G -> eV/G
    stated = float(L.to_sympy(rhs, {}))
    am3 = float(L.to_sympy(L.inline(r"1.2 \times 10^{-8}", "reference"), {}))
    high = percents(L.inline(r"3.7\text{％}", "reference"))[0]
    b_cr = float(bcr.subs(gauss))
    with L.patched(r"= 1.158 \times 10^{-8}$", r"= 1.168 \times 10^{-8}$", "reference"):
        broken = float(L.to_sympy(L.sides(L.inline(r"m_e c^2/B_\mathrm{cr} = 1.168 \times 10^{-8}", "reference"))[1], {}))
    return [exact("m_e c^2/B_cr = e hbar/(m_e c) (twice the Bohr magneton)", ratio - ee * h_ / (m * cc)),
            Result("= 1.158e-8 eV/G", f"{ev_g:.3e}" == f"{stated:.3e}", f"{ev_g:.6e} eV/G"),
            Result("B_cr = 4.4e13 G (BLP's H_0, eq. 90.3)", f"{b_cr:.1e}" == "4.4e+13", f"{b_cr:.4e} G"),
            Result("AM3's 1.2e-8 eV/G is 3.7% high", round(100 * (am3 / ev_g - 1), 1) == high, f"{100 * (am3 / ev_g - 1):.2f}%"),
            Result("control: 1.168e-8 in the README", f"{ev_g:.3e}" != f"{broken:.3e}", "the comparison fails, as it must"),
            in_code("the code has 1.2e-8 B gamma/(m c^2 [eV])", "src/Solver.cc",
                    "double x = 1.2 * pow(10, -8.0) * b_gauss / mass / pow(10, 9.0) * gamma;")]


# --- the quantum-synchrotron emission kernel ---

QS_KERNEL = (r"R_\mathrm{syn}(\varepsilon, \gamma_{e^\pm}) =  P_\mathrm{syn}(\varepsilon, \gamma_{e^\pm})= "
             r"\sqrt{3} \alpha m_e c^2 F_B "
             r"F_\xi\left(\frac{2 \varepsilon}{3 \gamma_{e^\pm}^2 F_B} \right)")
F_XI = r"F_\xi (x) = x \left[ \int_{z}^\infty K_{5/3}(x') dx' + zx \xi^2 K_{5/3}(z)) \right]"
QS_DEFS = (r"z = \frac{x}{1-x\xi},", r"\xi = \frac{3}{2 F_B \gamma_{e^\pm}},", r"x = \frac{\varepsilon}{ \gamma_{e^\pm} \xi}")
QS_TOKENS = {r"\varepsilon": eps, r"\gamma_{e^\pm}": g, "F_B": FB, "x": X, r"\xi": XI, "z": z}


def qs_tex():
    """The paper's argument of F_xi, z(x, xi), xi, x(xi) and the coefficient z x xi^2 of the second term."""
    rhs = lambda tex_: L.to_sympy(tex_.split("=")[1].rstrip(","), QS_TOKENS)
    return (L.to_sympy(r"\frac{2 \varepsilon}{3 \gamma_{e^\pm}^2 F_B}", QS_TOKENS), *map(rhs, QS_DEFS),
            L.to_sympy(r"zx \xi^2", QS_TOKENS))


def blp_photons(gam, ep, fb, alpha):
    """(1/hbar) dI/domega of eq. 90.23 (m = c = hbar = 1, e^2 = alpha): photons per unit time and ln(eps)."""
    chi_ = gam * fb
    x_ = (ep / ((gam - ep) * chi_))**(mp.mpf(2) / 3)
    tail = mp.sqrt(mp.pi) * mp.quad(mp.airyai, [x_, x_ + 1, x_ + 10, mp.inf])
    return -(alpha * ep / (mp.sqrt(mp.pi) * gam**2)) * (tail + (2 / x_ + (ep / gam) * chi_ * mp.sqrt(x_))
                                                          * mp.sqrt(mp.pi) * mp.airyai(x_, derivative=1))


@derivation("am3.qs_kernel")
def check_qs_kernel():
    """The quantum-synchrotron kernel needs xi = (3/2)F_B gamma and K_{2/3}: it is then BLP's eq. 90.23."""
    arg, z_of, xi_tex, x_of, coef = qs_tex()
    tok = {r"\varepsilon": eps, r"\gamma": g, "F_B": FB, r"\xi": XI, "z": z, "x": X, "y": Y}
    xi_printed = L.to_sympy(L.sides(L.inline(r"\xi = 3/(2F_B\gamma)", "reference"))[1], tok)
    xi_right = L.to_sympy(pinned(r"has $\xi = 3/(2F_B\gamma)$ for $(3/2)F_B\gamma$", r"(3/2)F_B\gamma"), tok)
    x_readme = L.to_sympy(L.sides(L.inline(r"x = \varepsilon/(\gamma\xi)", "reference"))[1], tok)
    arg_readme = L.to_sympy(L.inline(r"2\varepsilon/(3\gamma^2F_B)", "reference"), tok)
    ident_l, ident_r = (L.to_sympy(p_, tok) for p_ in L.sides(L.inline(r"z x \xi^2 = y^2/(1 - y)", "reference")))
    y_def = L.to_sympy(L.sides(L.inline(r"y = \varepsilon/\gamma", "reference"))[1], tok)
    below = {eps: g / (1 + w)}                                       # y = 1/(1 + w) < 1
    zz = z_of.subs(X, x_of).subs(XI, xi_right)
    orders = [sp.Rational(*map(int, re.match(r"K_\{(\d)/(\d)\}\(z\)", L.inline(k_, "reference")).groups()))
              for k_ in (r"K_{5/3}(z)", r"K_{2/3}(z)")]                     # as printed, as the code has it
    # the prefactors: the paper's sqrt3 alpha m_e c^2, the code's sqrt3 alpha m_e c^2/h
    al, h_ = sp.symbols("alpha h", positive=True)
    pre_tok = {r"\alpha": al, "m_e": me, "c": c, "h": h_}
    assert r"= \sqrt{3} \alpha m_e c^2 F_B" in QS_KERNEL
    paper_pre = L.to_sympy(L.inline(r"\sqrt{3}\thinspace\alpha m_e c^2", "reference"), pre_tok)
    code_pre = sp.sqrt(3) * al * L.to_sympy(L.inline(r"m_e c^2/h", "reference"), pre_tok)
    pre_f = sp.lambdify(al, code_pre.subs({me: 1, c: 1, h_: 2 * sp.pi}), "mpmath")           # m = c = hbar = 1
    out = [in_source("the kernel, F_xi, z, xi and x as transcribed", KLINGER, QS_KERNEL, F_XI, *QS_DEFS),
           exact_all("README's xi = 3/(2 F_B gamma), x = eps/(gamma xi), 2 eps/(3 gamma^2 F_B), z x xi^2, sqrt3 alpha m_e c^2 "
                     "are the paper's", xi_printed - xi_tex, x_readme - x_of, arg_readme - arg, ident_l - coef,
                     paper_pre - L.to_sympy(r"\sqrt{3} \alpha m_e c^2", pre_tok)),
           exact("with xi = (3/2) F_B gamma, x = eps/(gamma xi) is the argument 2 eps/(3 gamma^2 F_B)",
                 x_of.subs(XI, xi_right) - arg),
           exact_fails("with the printed xi = 3/(2 F_B gamma) it is not", x_of.subs(XI, xi_printed) - arg),
           exact("z x xi^2 = y^2/(1 - y), y = eps/gamma (the paper's z and x, any xi)",
                 (coef.subs(z, z_of).subs(X, x_of) - ident_r.subs(Y, y_def)).subs(below)),
           exact("z = (2/3) x_BLP^(3/2), BLP's Airy argument (eq. 90.24, chi = gamma F_B, eq. 90.3)",
                 (zz - sp.Rational(2, 3) * (eps / ((g - eps) * g * FB))).subs(below))]
    # numeric: sqrt(3) alpha (m_e c^2/h) F_B F_xi(x) (the code's rate) against eq. 90.23
    fz = sp.lambdify((eps, g, FB, XI), z_of.subs(X, arg), "mpmath")
    fc = sp.lambdify((eps, g, FB, XI), coef.subs(z, z_of).subs(X, arg), "mpmath")
    fx = sp.lambdify((eps, g, FB), arg, "mpmath")
    xi_f = {name: sp.lambdify((g, FB), xi_, "mpmath") for name, xi_ in (("right", xi_right), ("printed", xi_printed))}
    first = sp.Rational(*map(int, re.search(r"\\int_\{z\}\^\\infty K_\{(\d)/(\d)\}\(x'\)", F_XI).groups()))   # 5/3
    tails = {}

    def rate(gam, ep, fb, alpha, order, xi_name="right"):
        xi_ = xi_f[xi_name](gam, fb)
        z_ = fz(ep, gam, fb, xi_)
        if z_ <= 0:                                   # beyond the kinematic limit (the code returns 0)
            return mp.mpf(0)
        if z_ not in tails:
            tails[z_] = mp.quad(lambda q: mp.besselk(mp.mpf(first.p) / first.q, q), [z_, z_ + 1, z_ + 10, mp.inf])
        f_xi = fx(ep, gam, fb) * (tails[z_] + fc(ep, gam, fb, xi_) * mp.besselk(mp.mpf(order.p) / order.q, z_))
        return pre_f(alpha) * fb * f_xi
    r = rng(7)
    pts = []
    with mp.workdps(30):
        while len(pts) < 10:                          # chi = gamma F_B from 1e-3 to 1e3, z from 1e-3 to 30
            gam, chi_, y_ = log_uniform(r, 1e3, 1e9), log_uniform(r, 1e-3, 1e3), mp.mpf(r.random())
            if 1e-3 < 2 * y_ / (3 * (1 - y_) * chi_) < 30:
                pts.append((gam, gam * y_, chi_ / gam, 1 / mp.mpf(INV_ALPHA)))
        ref = {p_: blp_photons(*p_) for p_ in pts}
    out += [numeric("sqrt3 alpha (m_e c^2/h) F_B F_xi with (3/2)F_B gamma and K_2/3 = (1/hbar) dI/domega of eq. 90.23",
                    lambda *p_: rate(*p_, orders[1]), lambda *p_: ref[p_], pts, dps=30),
            control("control: K_5/3 as printed", lambda *p_: rate(*p_, orders[0]), lambda *p_: ref[p_], pts, dps=30),
            control("control: xi = 3/(2 F_B gamma) as printed", lambda *p_: rate(*p_, orders[1], "printed"),
                    lambda *p_: ref[p_], pts, dps=30)]
    nu0 = float(ME_SI * C_SI**2 / H_SI)
    tex = L.tex_source(KLINGER)
    out += [exact("the paper's prefactor = h x the code's: its kernel is h (1/hbar) dI/domega = dI/dnu, an energy",
                  paper_pre - h_ * code_pre),
            Result("the paper does not mention a pitch-angle average", tex is None or "pitch" not in tex.lower(),
                   "not re-checked: no TeX source" if tex is None else "no \"pitch\" in the TeX source"),
            in_code("the code has (3/2)F_B gamma, K_2/3 and sqrt3 alpha (m_e c^2/h) F_B, averaged over the pitch angle",
                    "src/Synchrotron.cc", "double ksi0 = 1.5 * b_ratio * gamma_e;",
                    "double y2 = z>1.0e2 ? 0.0 : x*ksi*ksi*z*gsl_sf_bessel_Knu(2.0/3.0, z);", "double nu_0 = 1.235e20;",
                    "return sqrt(3.0) * nu_0 * alpha_0 * b_dimless * I_Qsyn_avg(",
                    "double I_K53 = 0.5 * rk4_1d(I_Qsyn_mu_1, -0.99, 0.99, N_INTG, b_ratio, gamma_e, epsilon);"),
            Result("the code's nu_0 = 1.235e20 Hz is m_e c^2/h", abs(1.235e20 / nu0 - 1) < 1e-3, f"m_e c^2/h = {nu0:.5e} Hz")]
    return out


# --- the synchrotron kernel (classical) ---

SIGMA_T = r"\sigma_\mathrm{T} = \frac{8 \pi}{3} \left(\frac{e^2}{m_X c^2}\right)^2"
B_CRIT = r"B_\mathrm{crit}=\frac{m_X^2 c^3 }{e\hbar}"
RSYN = (r"\frac{3\sqrt{3}}{\pi} \frac{\sigma_\mathrm{T} c}{m_X c^2} \frac{B B_\mathrm{crit}}{8\pi} "
        r"\left(\frac{m_X}{m_e}\right)^2 \hat{R}(z)",
        r"2\sqrt{3} \frac{\alpha m_X c^2}{h} \left(\frac{m_X}{m_e}\right)^2\frac{B}{B_\mathrm{crit}}\, \hat{R}(z)")
Z_SYN = r"z = \frac{\varepsilon m_e/m_X}{ 3\gamma_X^2 B/ B_\mathrm{crit}}"
COOL = (r"\dfrac{4}{3}\frac{\sigma_\mathrm{T} c}{m_Xc^2} \frac{B^2}{8\pi} \gamma_X^{2}",
        r"\frac{8\pi}{9} \frac{\alpha m_X c^2}{h} \left( \frac{B}{B_\mathrm{crit}} \right)^2 \gamma_X^2")
RHAT = r"\hat{R}(z) &= z^2 \left\{K_{4/3}(z)K_{1/3}(z)-\frac{3}{5}z\left[K_{4/3}^{2}(z)-K_{1/3}^{2}(z)\right]\right\}"
RHAT_SMALL = r"\frac{2^{-1/3}}{5} \Gamma(1/3)^2 z^{1/3}"
RHAT_LARGE = r"\frac{\pi}{4} e^{-2z} \left( 1- \frac{99}{165}\frac{1}{2z}\right) \hspace{27mm} \mathrm{for} \;  2z > 10"


def particle(mass):
    """sigma_T and B_crit of the paper for a particle of mass `mass`."""
    tok = {"e": e, "m_X": mass, "c": c, r"\hbar": hb}
    return L.to_sympy(SIGMA_T.split("=")[1], tok), L.to_sympy(B_CRIT.split("=")[1], tok)


def rhat(k43, k13):
    """R^(z) of the paper with the given K_{4/3} and K_{1/3}. latex.py reads neither K^2(z) (a power of a function
    before its argument) nor \\{ \\}, so they are rewritten here."""
    tex = RHAT.split("&=")[1].replace(r"\left\{", "(").replace(r"\right\}", ")")
    tex = re.sub(r"(K_\{\d/3\})\^\{2\}\(z\)", r"\1(z)^{2}", tex)
    return L.to_sympy(tex, {"z": z}, funcs={"K_{4/3}": k43, "K_{1/3}": k13})


def hankel(nu, arg, terms=6):
    """K_nu(arg) ~ (pi/(2 arg))^(1/2) e^(-arg) sum_k a_k(nu) arg^(-k) (DLMF 10.40.2), truncated."""
    total, a_k = 0, sp.Integer(1)
    for k_ in range(terms):
        total += a_k / arg**k_
        a_k *= (4 * nu**2 - (2 * k_ + 1)**2) / (8 * (k_ + 1))
    return sp.sqrt(sp.pi / (2 * arg)) * sp.exp(-arg) * total


def mellin_kk(s_, a, b):
    """int_0^oo t^(s-1) K_a(t) K_b(t) dt = 2^(s-3)/Gamma(s) prod Gamma((s +- a +- b)/2)."""
    return (2**(s_ - 3) / sp.gamma(s_) * sp.gamma((s_ + a + b) / 2) * sp.gamma((s_ + a - b) / 2)
            * sp.gamma((s_ - a + b) / 2) * sp.gamma((s_ - a - b) / 2))


def syn_power_ratio(factor=1):
    """The kernel's emitted power, int dln(eps) (eps m_e c^2) R_syn / factor, over the cooling m_X c^2 (-dgamma/dt)."""
    sT_, bc = particle(mX)
    tok = {r"\sigma_\mathrm{T}": sT_, "c": c, "m_X": mX, "m_e": me, r"B_\mathrm{crit}": bc, "B": B, r"\hat{R}(z)": Rh,
           r"\alpha": e**2 / (hb * c), "h": 2 * sp.pi * hb, r"\gamma_X": g, r"\varepsilon": eps}
    kernel = L.to_sympy(RSYN[0], tok) / factor
    zz = L.to_sympy(Z_SYN.split("=")[1], tok)
    power = me * c**2 * kernel.subs(Rh, 1) / sp.diff(zz, eps) * (4 * sp.pi / (27 * sp.sqrt(3)))   # int R^ dz, see below
    return sp.simplify(power / (mX * c**2 * L.to_sympy(COOL[0], tok))), [L.to_sympy(x_, tok) for x_ in RSYN + COOL]


@derivation("am3.syn_mX")
def check_syn_mX():
    """With sigma_T and B_crit of the particle, the synchrotron kernel radiates (m_X/m_e)^2 times its cooling."""
    factor = L.to_sympy(L.inline(r"(m_X/m_e)^2", "reference"), {"m_X": mX, "m_e": me})
    ratio, (k1, k2, c1, c2) = syn_power_ratio()
    k43, k13 = (lambda a: sp.besselk(sp.Rational(4, 3), a)), (lambda a: sp.besselk(sp.Rational(1, 3), a))
    # int_0^oo R^ dz from the Mellin transforms of z^2 K_4/3 K_1/3, z^3 K_4/3^2 and z^3 K_1/3^2
    third = sp.Rational(1, 3)
    integral = mellin_kk(3, 4 * third, third) - sp.Rational(3, 5) * (mellin_kk(4, 4 * third, 4 * third) - mellin_kk(4, third, third))
    f = sp.lambdify(z, rhat(k43, k13), "mpmath")
    with mp.workdps(30):
        quad = mp.quad(f, [0, 1, 10, 50, mp.inf])
    stated = float(L.to_sympy(L.inline(r"3.4 \times 10^6", "reference"), {}))
    proton = float(factor.subs({mX: MP_ME, me: 1}))
    with L.patched(r"$(m_X/m_e)^2$", r"$(m_X/m_e)^4$", "reference"):
        broken = syn_power_ratio(L.to_sympy(L.inline(r"(m_X/m_e)^4", "reference"), {"m_X": mX, "m_e": me}))[0]
    sT_e, bc_e = particle(me)
    sT_x, bc_x = particle(mX)
    return [in_source("sigma_T, B_crit, R_syn, z and the cooling as transcribed", KLINGER, SIGMA_T, B_CRIT, *RSYN, Z_SYN, *COOL),
            exact("the paper's two forms of R_syn agree (alpha = e^2/(hbar c), h = 2 pi hbar)", k1 - k2),
            exact("the paper's two forms of the cooling agree", c1 - c2),
            exact_gamma("int_0^oo R^(z) dz = 4 pi/(27 sqrt3) (Mellin transforms of K K)", integral - 4 * sp.pi / (27 * sp.sqrt(3))),
            Result("= the quadrature of R^", abs(quad / mp.mpf(sp.N(integral, 40)) - 1) < 1e-25, f"numeric: {mp.nstr(quad, 20)}"),
            exact("power of the kernel (photons per unit time and ln eps) / cooling = (m_X/m_e)^2", ratio - factor),
            exact("without the README's factor (m_X/m_e)^2: = 1 (so for e+- the paper's kernel is right)",
                  syn_power_ratio(factor)[0] - 1),
            Result("a proton radiates about 3.4e6 times its cooling", f"{proton:.1e}" == f"{stated:.1e}",
                   f"(m_p/m_e)^2 = {proton:.4e}"),
            exact_fails("control: (m_X/m_e)^4 in the README", broken - 1),
            exact("the code's norm (sigma_T and B_crit of e+-) = the paper's first form without (m_X/m_e)^2",
                  sT_e * bc_e - sT_x * bc_x),
            in_code("the code's hadron kernel: (3 sqrt3/pi) sigma_T c B B_crit/(8 pi m)", "src/HadronicSynchrotron.cc",
                    "double norm = (3.0 * 1.732 / mathpi) * sigmaT * c0 * b_gauss * b_crit / (8.0 * mathpi) * erg_GeV / mass;")]


def readme_99(pin="has $99/{165}$ for $99/{162}$"):
    """The README's printed and correct coefficient (has a/b for c/d)."""
    found = re.search(r"has \$(\d+)/\{(\d+)\}\$ for \$(\d+)/\{(\d+)\}\$", L.line(pin, "reference")).groups()
    return sp.Rational(int(found[0]), int(found[1])), sp.Rational(int(found[2]), int(found[3]))


@derivation("am3.Rhat")
def check_Rhat():
    """R^(z) = (pi/4) e^(-2z) [1 - (99/162)/(2z) + ...] from the Bessel asymptotics; the paper prints 99/165."""
    printed, right = readme_99()
    assert r"The asymptotic form of $\hat R(z)$ for $2z \gt 10$ has" in L.line("has $99/{165}$ for $99/{162}$", "reference")
    series_r = rhat(lambda a: hankel(sp.Rational(4, 3), a), lambda a: hankel(sp.Rational(1, 3), a))
    expansion = sp.expand(sp.series(sp.simplify(series_r * 4 / sp.pi * sp.exp(2 * z)).subs(z, 1 / w), w, 0, 3).removeO())
    coeff = 2 * expansion.coeff(w, 1)                                    # of 1/(2z)
    paper = L.to_sympy(RHAT_LARGE.split(r"\hspace")[0], {"z": z})
    paper_coeff = 2 * sp.expand(paper * 4 / sp.pi * sp.exp(2 * z)).coeff(z, -1)
    k43, k13 = (lambda a: sp.besselk(sp.Rational(4, 3), a)), (lambda a: sp.besselk(sp.Rational(1, 3), a))
    exact_r = rhat(k43, k13)
    f = sp.lambdify(z, exact_r, "mpmath")
    with mp.workdps(40):
        at = {zz: 2 * zz * (f(mp.mpf(zz)) * 4 / mp.pi * mp.exp(2 * mp.mpf(zz)) - 1) for zz in (10**3, 10**5)}
    small = L.to_sympy(RHAT_SMALL, {"z": z, r"\Gamma(1/3)": sp.gamma(sp.Rational(1, 3))})
    with L.patched("has $99/{165}$ for $99/{162}$", "has $99/{162}$ for $99/{165}$", "reference"):
        swapped = readme_99("has $99/{162}$ for $99/{165}$")[1]
    return [in_source("R^, its asymptotic forms for 2z < 1e-2 and 2z > 10 as transcribed", KLINGER, RHAT, RHAT_SMALL, RHAT_LARGE),
            exact("the paper's asymptotic form has README's printed 99/165", paper_coeff + printed),
            exact("Hankel expansions of K_4/3, K_1/3: coefficient of 1/(2z) = -99/162 (README)", coeff + right),
            exact_fails("and not -99/165", coeff + printed),
            Result("numeric: 2z[(4/pi) e^(2z) R^(z) - 1] -> -99/162 = -0.6111 (-99/165 = -0.6)",
                   abs(at[10**5] + float(right)) < 1e-5 < 1e-2 < abs(at[10**5] + float(printed)),
                   f"{mp.nstr(at[10**3], 6)} at z = 1e3, {mp.nstr(at[10**5], 8)} at 1e5"),
            exact_gamma("the paper's small-z form (2^(-1/3)/5) Gamma(1/3)^2 z^(1/3) is R^'s limit",
                        sp.limit(exact_r / z**sp.Rational(1, 3), z, 0) - small / z**sp.Rational(1, 3)),
            exact_fails("control: the README's numbers swapped", coeff + swapped),
            in_code("the code has 99/162 (e+-)", "src/Synchrotron.cc", "(1. - 99./162. /2./xbar)"),
            in_code("the code has 99/162 (hadrons)", "src/HadronicSynchrotron.cc", "(1. - 99./162. /2./xbar)")]


# --- the e+- inverse-Compton kernel ---

DM09 = (r"\frac{3}{8}\frac{\sigma_{\mathrm{T}, e}}{\gamma_{e^\pm} x^{3}} "
        r"\left[ yx^{2}+1+2x+\frac{1}{y}\left(x^2-2x-2\right)+\frac{1}{y^2} \right]",
        r"H\left( y-\frac{1}{1+2x} \right) H\left(1-\frac{x}{2\gamma_{e^\pm}^2}-y\right)")
IC_XY = (r"where $x=\gamma_{e^\pm} \varepsilon_{i}(1-\mu)$ is the photon energy in the particle rest frame, "
         r"$y=1-\varepsilon_{o}/\gamma_{e^\pm}$")
IC_AVG = r"\dfrac{c}{2}\int \dd \mu (1-\mu) \: \dfrac{\dd^{2}\sigma_\mathrm{IC}}{\dd\varepsilon_{o}\dd\mu}"
IC_RANGE = (r"c \sigma_{\mathrm{T}, e} \widetilde{R}(u,v) & {\rm~for~} \dfrac{u}{2\gamma_{e^\pm}^{2}}<v<\dfrac{2u}{1+2u} "
            r"{\rm ~and~} u<2\gamma_{e^\pm}^{2}-\frac{1}{2}")
IC_PRINTED = (r"\dfrac{ 6(1-u)\dfrac{u}{v}\ln\left[\dfrac{u}{2v(1-u)}\right] - 3\left( \dfrac{u}{v}-2+2u \right) "
              r"\left(\dfrac{u}{v}+1+\dfrac{u^{2}}{2}-u\right)}{4v\gamma_{e^\pm}(u-1)^{3}}")
IC_UV = (r"where $u=2\gamma_{e^\pm}\varepsilon_{i}$, $v=\varepsilon_{o}/\gamma_{e^\pm}$. We note that this is the same result as "
         r"Eq.~2.49 in \cite{GouldBlumenthal1970}.")
IC_TOTAL = (r"\dfrac{3c}{8u^{3}(1+2u)}\left[-2u(4+9u+u^{2})+(4+u)(1+2u)^{2}\ln(1+2u)+4u(1+2u)\mathrm{Li}_2(-2u) \right]")
GAO_PRINTED = IC_PRINTED.replace(r"\gamma_{e^\pm}", r"\gamma_{e,i}")
GAO_UV = r"where $u=2\gamma_{e,i}$, $v=\gamma_{f,o}/\gamma_{e,i}$."
IC_TOKENS = {r"\sigma_{\mathrm{T}, e}": sT, r"\gamma_{e^\pm}": g, "x": X, "y": Y, "u": u, "v": v, r"\gamma_{e,i}": g,
             r"\varepsilon_{i}": eps, r"\varepsilon_{o}": eo, r"\mu": sp.Symbol("mu"), "c": c}
RT_PIN = r"\tilde R = [6(1 - v)(v/u) \ln(v/[2u(1 - v)])"


def readme_Rtilde():
    return L.to_sympy(L.sides(L.parts(L.display(RT_PIN, "reference"))[0])[1], {"u": u, "v": v, r"\gamma": g})


def tex_rhs(snippet):
    """The right side of "$a=b$" in a TeX snippet."""
    return L.to_sympy(snippet.split("=", 1)[1].strip("$ ,"), IC_TOKENS)


def eps_of_uv():
    """eps_i(u) and eps_o(v) from the paper's u = 2 gamma eps_i, v = eps_o/gamma."""
    u_def, v_def = (tex_rhs(p_) for p_ in re.findall(r"\$([^$]*)\$", IC_UV)[:2])
    return sp.solve(u_def - u, eps)[0], sp.solve(v_def - v, eo)[0]


def bg_248(eps_o, eps_i, gam):
    """Blumenthal & Gould's eq. 2.48 for one target photon (n(eps) deps = 1, eps = eps_i), per unit eps_o and
    over c sigma_T, with m c^2 = 1: dE_1 = deps_o/gamma (eq. 2.47) and 2 pi r_0^2 = (3/4) sigma_T."""
    e1, big = eps_o / gam, 4 * eps_i * gam                           # E_1 (eq. 2.47), Gamma_eps (eq. 2.49)
    q = e1 / (big * (1 - e1))                                        # eq. 2.49
    return (sp.Rational(3, 4) / (gam * eps_i) / gam
            * (2 * q * sp.log(q) + (1 + 2 * q) * (1 - q) + sp.Rational(1, 2) * (big * q)**2 / (1 + big * q) * (1 - q)))


def angle_average():
    """(c/2) int dmu (1 - mu) d^2sigma/(deps_o dmu) / (c sigma_T,e) of the paper, in u, v: with x = gamma eps_i
    (1 - mu), dmu = -dx/(gamma eps_i); the H functions keep x > x_lo(y) and x < x_hi(y), and mu >= -1 keeps
    x <= 2 gamma eps_i = u, the upper limit in the paper's range (checked)."""
    x_def, y_def = (tex_rhs(p_) for p_ in re.findall(r"\$([^$]*)\$", IC_XY))
    eps_i, eps_o = eps_of_uv()
    sigma = L.to_sympy(DM09[0], IC_TOKENS)
    h1, h2 = (L.to_sympy(a_, IC_TOKENS) for a_ in re.findall(r"H\\left\((.*?)\\right\)", DM09[1]))
    yv = y_def.subs(eo, eps_o)
    x_lo, x_hi = sp.solve(h1.subs(Y, yv), X)[0], sp.solve(h2.subs(Y, yv), X)[0]
    x_top = x_def.subs(sp.Symbol("mu"), -1).subs(eps, eps_i)
    gei = g * eps_i                                               # x = gei (1 - mu)
    average = sp.integrate(sp.expand((c / 2) * (X / gei) * sigma.subs(Y, yv) / gei), (X, x_lo, x_top)) / (c * sT)
    return average, x_lo, x_hi, x_top


@derivation("am3.Rtilde")
def check_Rtilde():
    """The code's R~ is the angle average of the paper's own cross section and Blumenthal & Gould's eq. 2.48."""
    ours = readme_Rtilde()
    average, x_lo, x_hi, x_top = angle_average()
    vmin, vmax = (L.to_sympy(a_, IC_TOKENS) for a_ in L.between(IC_RANGE, r"{\rm~for~}", r"{\rm ~and~}").split("<v<"))
    u_max = L.to_sympy(IC_RANGE.split("u<")[1], IC_TOKENS)
    in_range = {v: 1 / (1 + w)}                                     # 0 < v < 1
    eps_i, eps_o = eps_of_uv()
    bg = bg_248(eps_o, eps_i, g)
    printed = L.to_sympy(IC_PRINTED, IC_TOKENS)
    L.inline(r"\tilde R(u, v)", "reference")
    old, new = L.inline(r"(1 - v)^2", "reference"), L.inline(r"(u - 1)^3", "reference").replace("u", "v")   # "(u - 1)^3 for (1 - v)^2", before u <-> v
    with L.patched(old + "].", new + "].", "reference"):
        swapped = readme_Rtilde().xreplace({u: v, v: u})
    with L.patched(r"v^2/2 - v)]", r"v^2/3 - v)]", "reference"):
        broken = readme_Rtilde()
    # the code (src/InverseCompton.cc, s_diff_full): L = u, K = v, K_ = 1 - v, R = K/L; a3 (a1 + a2)
    a1 = (v / u - 2 + 2 * v) * (v / u + 1 + v**2 / 2 - v)
    a2 = -2 * (1 - v) * (v / u) * sp.log(v / u / 2 / (1 - v))
    code = -sp.Rational(3, 4) / (g * u * (1 - v)**2) * (a1 + a2)
    # normalisation: int R~ deps_o (gamma -> oo: v from 0 to vmax; gamma R~ does not depend on gamma)
    f, tot = sp.lambdify((u, v), g * ours, "mpmath"), sp.lambdify(u, L.to_sympy(
        IC_TOTAL.replace(r"\mathrm{Li}_2", "Li"), IC_TOKENS, funcs={"Li": lambda a: sp.polylog(2, a)}).subs(c, 1), "mpmath")
    with mp.workdps(50):
        norm = max(abs(mp.quad(lambda vv: f(uu, vv), [0, 2 * uu / (1 + 2 * uu)]) / tot(uu) - 1)
                   for uu in (mp.mpf("0.01"), mp.mpf("0.3"), mp.mpf(2), mp.mpf(50)))
    return [in_source("cross section, x, y, the average, the range, R~, u, v and the total rate as transcribed",
                      KLINGER, *DM09, IC_XY, IC_AVG, IC_RANGE, IC_PRINTED, IC_UV, IC_TOTAL),
            in_source("Gao et al.'s R~ and u, v as transcribed (u = 2 gamma_e,i lacks the photon energy)", GAO, GAO_PRINTED, GAO_UV),
            exact("README R~ = (c/2) int dmu (1 - mu) d^2sigma/(deps_o dmu) / (c sigma_T,e) (the paper's own cross section)",
                  (average - ours).subs(in_range)),
            exact_all("the paper's range: x_lo = u at v = 2u/(1 + 2u), x_hi = u at v = u/(2 gamma^2), they meet at "
                      "u = 2 gamma^2 - 1/2", (x_lo - x_top).subs(v, vmax), (x_hi - x_top).subs(v, vmin),
                      (vmin - vmax).subs(u, u_max)),
            exact("README R~ = Blumenthal & Gould's eq. 2.48 per unit eps_o, over c sigma_T", (bg - ours).subs(in_range)),
            exact("README R~ = the code's s_diff_full", (code - ours).subs(in_range)),
            Result("int R~ deps_o (gamma -> oo) = the paper's total rate over c sigma_T,e (printed without sigma_T,e)",
                   norm < 1e-35, f"numeric: within {mp.nstr(norm, 3)} at u = 0.01, 0.3, 2, 50 (50 digits)"),
            exact_fails("the printed R~ differs (Klinger et al.)", (printed - ours).subs(in_range)),
            exact("Gao et al. print the same", L.to_sympy(GAO_PRINTED, IC_TOKENS) - printed),
            exact("printed = README with u <-> v and (u - 1)^3 for (1 - v)^2", (printed - swapped).subs(u, 1 / (1 + w))),
            exact_fails("control: v^2/3 in the README", (average - broken).subs(in_range)),
            in_code("the code's s_diff_full", "src/InverseCompton.cc",
                    "double a1 = (R - 2.0 + 2.0 * K) * (R + 1.0 + 0.5 * K * K - K);",
                    "double a2 = -2.0 * K_ * R * log(0.5 * R / K_);", "double a3 = -0.75 / g / L / K_ / K_;")]


# --- the hadrons' inverse Compton ---

HADRON_IC = (r"\frac{\mathrm{d}\sigma_\mathrm{IC}}{\mathrm{d}\varepsilon_{o}} = "
             r"\sigma_T \delta(\varepsilon_{o} - \gamma_{X,i}^2 \varepsilon_{i}) "
             r"H \left( 1- \frac{1}{2}\gamma_{X, i} \varepsilon_{i} \frac{m_e}{m_X} \right)")
HADRON_TOKENS = {r"\varepsilon_{o}": eo, r"\varepsilon_{i}": eps, r"\gamma_{X,i}": g, r"\gamma_{X, i}": g, "m_e": me, "m_X": mX,
                 r"\gamma": g, r"\varepsilon": eps}


def largest_share(cut):
    """The largest eps_o over the particle's energy gamma m_X/m_e (in m_e c^2) that the cut H(cut) allows,
    with the paper's eps_o = gamma^2 eps_i (the delta function)."""
    delta = L.to_sympy(L.between(HADRON_IC, r"\delta(", ") H"), HADRON_TOKENS)
    eps_max = sp.solve(cut, eps)[0]
    return sp.simplify(sp.solve(delta.subs(eps, eps_max), eo)[0] / (g * mX / me))


def readme_cut(span):
    return L.to_sympy(L.inline(span, "reference"), HADRON_TOKENS, funcs={"H": lambda a: a})


@derivation("am3.hadron_ic")
def check_hadron_ic():
    """With eps_o = gamma^2 eps, the printed cut lets eps_o reach twice the particle's energy, the code's half of it."""
    L.line(r"which lets a photon take up to twice the particle's energy; the code cuts at", "reference")
    printed = readme_cut(r"H(1 - \tfrac{1}{2}\gamma\varepsilon m_e/m_X)")
    code = readme_cut(r"H(1 - 2\gamma\varepsilon m_e/m_X)")
    at, value = (L.to_sympy(p_, HADRON_TOKENS) for p_ in L.sides(L.inline(r"\gamma\varepsilon m_e/m_X = 1/2", "reference")))
    paper = L.to_sympy(L.between(HADRON_IC, r"H \left(", r"\right)"), HADRON_TOKENS)
    with L.patched(r"$H(1 - \tfrac{1}{2}\gamma", r"$H(1 - \tfrac{1}{3}\gamma", "reference"):
        broken = largest_share(readme_cut(r"H(1 - \tfrac{1}{3}\gamma\varepsilon m_e/m_X)"))
    return [in_source("the step cross section as transcribed", KLINGER, HADRON_IC),
            exact("README's H argument = the paper's", printed - paper),
            Result("printed cut: eps_o up to twice the particle's energy", largest_share(printed) == 2,
                   f"exact: eps_o/E = {largest_share(printed)}"),
            Result("the code's cut gamma eps m_e/m_X = 1/2 is H(1 - 2 gamma eps m_e/m_X): eps_o up to half of it",
                   largest_share(code) == sp.Rational(1, 2) and code.subs(eps, sp.solve(at - value, eps)[0]) == 0,
                   f"exact: eps_o/E = {largest_share(code)}"),
            Result("control: 1/3 for 1/2 in the README", broken != 2, f"eps_o/E = {broken}, not 2"),
            in_code("the code: eps_o = gamma^2 eps_i, and eps_i [GeV] < 0.5 m_p/gamma", "src/HadronicInverseCompton.cc",
                    "int dEp = (int)(log(gp * gp ) / dat.dlnE);",
                    "int jmax1_p  = (int)(log(0.5 * massp/gp ) / dat.dlnE) + dat.ind_min_energy_pho_hadgrid;")]


if __name__ == "__main__":
    run()
