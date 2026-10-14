"""Derivation checks of AM3's solver and inverse-Compton cooling: the equations of tests/reference/README.md (Bugs of
AM3 v1.2.0 that am3_mojo fixes: the solver bullets, the series of the cooling functions; Misprints of the AM3 paper;
Numerics: the cooling function F; Kept as in AM3: the loss rate) and of the local citation index citations.md
(Chang & Cooper 1970; Jones 1965). Sources:
  - Jones 1965 (Phys. Rev. 137, B1306), pp. B1307-B1309, transcribed from the journal scan: eqs. 13-15, Table I and
    eq. 21 (his Thomson loss); r_0 = e^2/m c^2 (after eq. 10), and gamma and alpha are energies in units of the
    scattering particle's rest energy. His preprint (NASA X-641-64-28, Oct. 1964, pp. 9-10) prints the same eqs. 13-15
    and Table I;
  - Klinger et al. 2024 (the AM3 paper), the TeX source of arXiv:2312.13371v2: eq. A1 (p. 20 of the arXiv PDF) and
    Sec. A.2 (p. 22: eqs. A19-A23, and sigma_T ∝ m_X^-4 after eq. A18), parsed;
  - Chang & Cooper 1970 (J. Comput. Phys. 6, 1), transcribed from the scan: eqs. 1 (p. 1), 15-17 (p. 9), and 18 with
    the sentence after it (p. 10). Their eq. 19 (p. 10), which starts their proof of non-negative densities
    (eqs. 19-21), is fully implicit, as citations.md says;
  - the Klein-Nishina cross section, without a paper: Jones' eq. 13 is the loss in isotropic photons
    (am3.ic_klein_nishina); its Klein-Nishina limit is in the Results of tests/literature_checks/README.md.

The solver bullets (am3.green, am3.source_integral, am3.zfactor, am3.regime4) follow from the cooling equation in
x = ln E, dn/dt = -d(A n)/dx - alpha n + q, with the cooling rate A = dx/dt < 0, the escape and decay rate alpha and
the injection q constant within a step. Along the characteristics dX/dt = A(X), d(A n)/dt = -alpha (A n) + A q, so
the particles at y at the start of the step reach x at its end if int_x^y dx'/|A| = dt, and
  n(x) = n(y) A(y)/A(x) exp(-int_x^y alpha/|A|) + int_x^y q(x') exp(-int_x^x' alpha/|A| dx'') dx'/|A(x)|.
The checks verify this exactly, for arbitrary functions A, alpha, q and initial densities n_0.

Regular expressions cut the README's pieces out of the line found by a phrase the controls leave alone (L.inline
would pin the very span a control patches). For integrals, which latex.to_sympy cannot read, they capture the limits,
the sign of the exponent and the divisor. Without the citation index (git-ignored), its two checks say so.

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/am3_cooling.py
"""
import re

import mpmath as mp
import sympy as sp

import latex as L
from latex import KLINGER
from verify import Result, derivation, exact, exact_fails, in_source, must_fail, run, zero_numerator

z, eps, gam, al_, c, sT, r0, B, C, dx, w = sp.symbols("z epsilon gamma alpha c sigma_T r_0 B C Delta_x w", positive=True)
E_ = sp.Symbol("E", positive=True)   # e^w in identities that are rational in w and e^w
G = 1 + sp.Symbol("v", positive=True)   # gbar > 1: gamma = (G + 1/G)/2, (gamma^2 - 1)^(1/2) = (G - 1/G)/2
GAMMA_G, P_G = (G + 1 / G) / 2, (G - 1 / G) / 2   # gamma and p for gbar = G
n_, F_, f1_, f2_ = (sp.Function(nm) for nm in ("n", "F", "f1", "f2"))
Li2 = lambda u: sp.polylog(2, u)


def _match(pattern, text):
    m = re.search(pattern, text)
    if m is None:
        raise LookupError(f"pattern not found: {pattern[:80]}")
    return m


def without_citations(what):
    """[a Result saying so] if the citation index (git-ignored) is not there, else []."""
    return [] if L.FILES["citations"].exists() else [Result(what, True, "not re-checked: knowledge/citations.md is not there")]


def exact_all(what, *differences):
    """Every difference is identically zero (verify.exact for several at once)."""
    left = [str(n)[:60] for n in map(zero_numerator, differences) if n != 0]
    return Result(what, not left, "exact: 0" if not left else f"exact: not zero ({left[0]})")


def by_exp(expr, var=w):
    """expr with var = ln E: an identity in var and e^var becomes rational in E and ln E (see zero_numerator)."""
    return sp.expand(sp.expand(expr).subs(var, sp.log(E_)))


# --- Jones (1965), transcribed from the journal scan ---

def jones_f1(u, li2=2):
    """Eq. 14 (p. B1308): f1(z) = (z + 6 + 3/z) ln(1 + 2z) - (22z^3/3 + 24z^2 + 18z + 4)(1 + 2z)^-2 - 2 + 2 Li2(-2z)."""
    return ((u + 6 + 3 / u) * sp.log(1 + 2 * u) - (sp.Rational(22, 3) * u**3 + 24 * u**2 + 18 * u + 4) / (1 + 2 * u)**2
            - 2 + li2 * Li2(-2 * u))


def jones_f2(u, li2=1):
    """Eq. 15 (p. B1308): f2(z) = (z + 31/6 + 5/z + 3/2z^2) ln(1 + 2z) - (22z^3/3 + 28z^2 + 103z/3 + 17 + 3/z)
    x (1 + 2z)^-2 - 2 + Li2(-2z). Jones' "3/2z^2" is 3/(2z^2), as in the AM3 paper's \\frac{3}{2z^2}."""
    return ((u + sp.Rational(31, 6) + 5 / u + sp.Rational(3, 2) / u**2) * sp.log(1 + 2 * u)
            - (sp.Rational(22, 3) * u**3 + 28 * u**2 + sp.Rational(103, 3) * u + 17 + 3 / u) / (1 + 2 * u)**2
            - 2 + li2 * Li2(-2 * u))


def jones_beta(g):
    """beta = (gamma^2 - 1)^(1/2)/gamma (p. B1307, after eq. 6)."""
    return sp.sqrt(g**2 - 1) / g


def jones_gbar(g):
    """gbar = gamma (1 + beta) (p. B1308, after eq. 13)."""
    return g * (1 + jones_beta(g))


def jones_F(alpha, g, f1, f2, gbar):
    """Eq. 13: F(alpha, gamma) = gamma [f1(alpha gbar) - f1(alpha/gbar)] - alpha [f2(alpha gbar) - f2(alpha/gbar)]."""
    return g * (f1(alpha * gbar) - f1(alpha / gbar)) - alpha * (f2(alpha * gbar) - f2(alpha / gbar))


def jones_loss(alpha, g, dens, F, radius=r0):
    """Eq. 13: the integrand of <-dgamma/dt>_av = (1/2) pi r_0^2 c int_0^oo dalpha n(alpha) F(alpha, gamma)/(beta gamma^2 alpha^2)."""
    return sp.pi * radius**2 * c / 2 * dens * F / (jones_beta(g) * g**2 * alpha**2)


JONES_21 = "3.555"   # eq. 21 (p. B1309), the zero-order loss -<dE/dt>_av = 3.555 pi r_0^2 c rho (p/m_e c)^2, rho the
                     # photon energy density. 3.555 is 32/9: (32/9) pi r_0^2 = (4/3) sigma_T, the Thomson loss


def jones_eq21(rho, p, radius=r0):
    """Eq. 21 with 32/9 for 3.555 (p the momentum in units of m c)."""
    return sp.Rational(32, 9) * sp.pi * radius**2 * c * rho * p**2


# Table I (p. B1309): A_n of f1, B_n of f2, n = 0..10
TABLE_I = {"A": ["0.0", "0.0", "0.0", "0.88888827e0", "-0.27999992e1", "0.78400015e1", "-0.20520631e2", "0.51156461e2",
                 "-0.12304761e3", "0.28815807e3", "-0.66115232e3"],
           "B": ["0.0", "0.0", "0.13333320e1", "-0.28444412e1", "0.69999923e1", "-0.16784715e2", "0.39161810e2",
                 "-0.89469178e2", "0.20112988e3", "-0.44645477e3", "0.98096851e3"]}
_TAYLOR = {}


def taylor(name):
    """The Taylor coefficients c_0 ... c_10 of f1 (name "A") or f2 ("B"), exact: the series of eqs. 14-15."""
    if name not in _TAYLOR:
        ser = sp.series(jones_f1(z) if name == "A" else jones_f2(z), z, 0, 11).removeO()
        _TAYLOR[name] = [ser.coeff(z, k) for k in range(11)]
    return _TAYLOR[name]


def poly(name, upto):
    """The Taylor polynomial of f1 or f2 up to z^upto, as a function."""
    cs = taylor(name)
    return lambda u: sum(cs[k] * u**k for k in range(upto + 1))


MP = {"f1": sp.lambdify(z, jones_f1(z), "mpmath"), "f2": sp.lambdify(z, jones_f2(z), "mpmath")}


def mp_series(name, terms):
    """The series of f1 (from z^3) or f2 (from z^2) with the given number of terms, as an mpmath function."""
    start = 3 if name == "A" else 2
    cs = [mp.mpf(k.p) / k.q for k in taylor(name)]
    return lambda u: sum(cs[k] * u**k for k in range(start, start + terms))


# --- the AM3 paper (Klinger et al. 2024), from the TeX source ---

K_SIGMA = r"\sigma_\mathrm{T} = \frac{8 \pi}{3} \left(\frac{e^2}{m_X c^2}\right)^2"
K_M4 = r"Here we recall that $\sigma_T \propto m_X^{-4}$ is a function of the particle mass."
K_EPS = r"for photons we use the dimensionless energy $\varepsilon \equiv E_\gamma / m_e c^2$"
K_N = r"We recall our definition of $n = \dd^2 N / \dd E \dd V$"
K_LOSS = (r"- \dot{\gamma}_{X, \mathrm{IC}} = \frac{3}{16} \sigma_T c \int_0^\infty  \mathrm{d}\varepsilon_i "
          r"\frac{n_{\gamma}(\varepsilon_i)}{\gamma_X^2 \varepsilon_i} F(\varepsilon_i, \gamma_X)")
K_F = (r"F(\varepsilon, \gamma_X) = \gamma_X \left[f_1 (\varepsilon \bar{\gamma}_X) - f_1 (\varepsilon/ \bar{\gamma_X})\right]"
       r" - \varepsilon \left[f_2 (\varepsilon \bar{\gamma}_X) - f_2 (\varepsilon/ \bar{\gamma}_X) \right]")
K_GBAR = r"\bar{\gamma}_X &= \gamma_X +(\gamma_X^2 -1)^{1/2}"
K_F1 = (r"f_1(z) &= \qty(z+6+\frac{3}{z})\mathrm{\log}(1+2z) - \qty(\frac{22}{3}z^3 +24z^2 +18z +4)(1+2z)^{-2} -2 "
        r"- 2 \mathrm{Li}_2(-2z)")
K_F2 = (r"f_2(z) &= \qty(z + \frac{31}{6} + \frac{5}{z} + \frac{3}{2z^2})\mathrm{log}(1+2z) -\qty(\frac{22}{3}z^3 +28z^2 "
        r"+\frac{103 z}{3} +17 + \frac{3}{z})(1+2z)^{-2} -2 - 2 \mathrm{Li}_2(-2z)")


def klinger_f(snippet):
    """Eq. A22 or A23 as printed: \\qty( is a parenthesis, log the natural logarithm (Jones' ln)."""
    rhs = snippet.split("&=")[1].replace(r"\qty(", "(").replace(r"\mathrm{\log}", r"\ln").replace(r"\mathrm{log}", r"\ln")
    return L.to_sympy(rhs, {"z": z}, funcs={r"\mathrm{Li}_2": Li2})


def klinger_sigma(m):
    """Eq. A1, sigma_T = (8 pi/3) (e^2/(m_X c^2))^2 (e the charge, here the symbol q_e)."""
    return L.to_sympy(K_SIGMA.split(" = ")[1], {"e": sp.Symbol("q_e", positive=True), "m_X": m, "c": c})


# --- the README (and the citation index), read from the files ---

def readme_F_tex():
    """F and gbar of 'Inverse-Compton cooling near gamma = 1' (Numerics), as LaTeX."""
    m = _match(r"The cooling function \$F\(\\varepsilon, \\gamma\)\$ \$=(.*?\$) \(\$\\bar\\gamma = (.*?)\$, \$\\varepsilon\$ "
               r"in units of the particle's rest energy; Jones 1965, eqs\. 13–15\)", L.line("**Inverse-Compton cooling near", "reference"))
    return m.group(1).replace("$", "").strip(), m.group(2)


def readme_gbar(g):
    return L.to_sympy(readme_F_tex()[1], {r"\gamma": g})


def readme_F(f1, f2, g, e, gbar):
    """The README's F(eps, gamma) with the given f1, f2 (functions) and gbar."""
    return L.to_sympy(readme_F_tex()[0], {r"\gamma": g, r"\varepsilon": e, r"\bar\gamma": gbar}, funcs={"f_1": f1, "f_2": f2})


def readme_F_mp(f1="f1", f2="f2"):
    """The README's F(eps, gamma) as an mpmath function, with gbar as the README defines it."""
    expr = readme_F(f1_, f2_, gam, eps, readme_gbar(gam))
    impl = {"f1": MP[f1] if isinstance(f1, str) else f1, "f2": MP[f2] if isinstance(f2, str) else f2}
    return sp.lambdify((eps, gam), expr, [impl, "mpmath"])


def readme_p(g):
    return L.to_sympy(L.sides(L.inline(r"p = \sqrt{\gamma^2 - 1}"))[1], {r"\gamma": g})


def li2_term(tex):
    """A term like +2\\ \\mathrm{Li}_ 2(-2z) of the README or citations.md."""
    return L.to_sympy(tex[1:] if tex.startswith("+") else tex, {"z": z}, funcs={r"\mathrm{Li}_ 2": Li2})


def numerics_point10():
    """The pieces of the first bullet of "Misprints of the AM3 paper" (tests/reference/README.md): the AM3 paper's
    inverse-Compton misprints."""
    line = L.line("In the inverse-Compton loss, the cooling functions", "reference")
    f = _match(r"the cooling functions \$f_1\$ and \$f_2\$ have \$(.*?)\$ instead of \$(.*?)\$ and \$(.*?)\$\. They go to "
               r"zero only linearly as \$z \\to 0\$ \(as \$(.*?)\$ and \$(.*?)\$ instead of \$(.*?)\$ and \$(.*?)\$, so they "
               r"miss the Thomson limit\)", line)
    d = _match(r"The loss rate has \$(.*?)\$ where its photon density per unit \$\\varepsilon\$ needs \$(.*?)\$\.", line)
    s = _match(r"The same section also writes \$(.*?)\$ for the \$(.*?)\$ of its own definition\. It takes \$\\varepsilon\$ "
               r"in units of \$m_e c\^2\$, where the formula needs it in units of the particle's rest energy", line)
    return {"printed": f.group(1), "f1": f.group(2), "f2": f.group(3), "lin": (f.group(4), f.group(5)),
            "lead": (f.group(6), f.group(7)), "density": (d.group(1), d.group(2)), "sigma": (s.group(1), s.group(2))}


def thomson_leading(f1, f2):
    """(power, coefficient) of the lowest order in eps of the README's F with the given f1, f2 (polynomials), for
    gbar = G and gamma = (G + 1/G)/2."""
    expr = sp.expand(readme_F(f1, f2, (G + 1 / G) / 2, eps, G))
    for k in range(1, 6):
        cf = sp.factor(expr.coeff(eps, k))
        if cf != 0:
            return k, cf
    raise ValueError("no term up to eps^5")


# --- the cooling equation and its solution along the characteristics ---

xs, ts, s = sp.symbols("x t s", real=True)   # x = ln E, t the time into the step, s = x' (integration variable)
a, alpha, q, n0 = (sp.Function(nm) for nm in ("a", "alpha", "q", "n_0"))   # a = |A| = -A


class S(sp.Function):
    """An antiderivative of alpha/|A|: int_u^v alpha/|A| dx'' = S(v) - S(u)."""
    def fdiff(self, argindex=1):
        return alpha(self.args[0]) / a(self.args[0])


class T(sp.Function):
    """An antiderivative of 1/|A|: particles cool from v down to u in the time T(v) - T(u)."""
    def fdiff(self, argindex=1):
        return 1 / a(self.args[0])


class Y(sp.Function):
    """y(x, t), the energy from which particles reach x in the time t: int_x^y dx'/|A| = T(y) - T(x) = t,
    y(x, 0) = x. Its derivatives dy/dx = |A(y)|/|A(x)| and dy/dt = |A(y)| are checked against this."""
    def fdiff(self, argindex=1):
        return a(self) / a(self.args[0]) if argindex == 1 else a(self)


YXT = Y(xs, ts)


def residual(P, g, lo, hi, esc=alpha):
    """The residual dn/dt + d(A n)/dx + alpha n - q (A = -a) of n = P + int_lo^hi g ds, in two parts that both vanish
    for a solution: the terms at the limits (Leibniz' rule) with those of P, and the integrand of the rest."""
    gl, gh = g.subs(s, lo), g.subs(s, hi)
    at_limits = (sp.diff(P, ts) - sp.diff(a(xs) * P, xs) + esc(xs) * P - q(xs)
                 + gh * (sp.diff(hi, ts) - a(xs) * sp.diff(hi, xs)) - gl * (sp.diff(lo, ts) - a(xs) * sp.diff(lo, xs)))
    return at_limits, sp.diff(g, ts) - sp.diff(a(xs) * g, xs) + esc(xs) * g


def at_start(expr):
    """expr at t = 0, where y = x."""
    return sp.sympify(expr).subs(ts, 0).xreplace({Y(xs, 0): xs})


def solves(what, P, g, lo, hi, esc=alpha, start=None):
    """A Result: n = P + int_lo^hi g ds solves the cooling equation (and starts from `start` at t = 0)."""
    lim, itg = (zero_numerator(e) for e in residual(P, g, lo, hi, esc))
    ok = lim == 0 and itg == 0
    detail = "exact: residual 0 at the limits and in the integrand" if ok else \
        f"exact: residual at the limits {str(lim)[:60]}, integrand {str(itg)[:60]}"
    if start is not None:
        init = zero_numerator(at_start(P) - start)
        same = sp.simplify(at_start(lo) - at_start(hi)) == 0
        ok = ok and init == 0 and same
        detail += f"; n(x, 0) - {start}: {init}" + ("" if same else "; the integral does not vanish at t = 0")
    return Result(what, ok, detail)


def readme_green_signs():
    """The signs of AM3's weight and of the decaying one in 'Sign of the Green's function'."""
    m = _match(r"weights the injection with \$\\exp\(([-+])\\int \\alpha/\\vert A\\vert\\thinspace dx\)\$ instead of the "
               r"decaying \$\\exp\(([-+])\\int \\alpha/\\vert A\\vert\\thinspace dx\)\$ \(\$\\alpha\$: escape and decay rate, "
               r"\$A\$: cooling rate, \$x = \\ln E\$\)", L.line("**Sign of the Green's function.**", "reference"))
    return tuple(1 if sg == "+" else -1 for sg in m.groups())


def readme_source(antider=S):
    """The README's source integral of regime 3 (Quadrature of the source integral), read from its text:
    (lower limit, upper limit, integrand in s = x'), with int_u^v alpha/|A| = antider(v) - antider(u)."""
    m = _match(r"The source integral there is \$\\int_\{?(\w'?)\}?\^\{?(\w'?)\}? q\(x'\) \\exp\(([-+])\\int_\{?(\w'?)\}?\^\{?(\w'?)\}? "
               r"\\alpha/\\vert A\\vert\\ dx''\)\$ \$dx'/\\vert A\((\w'?)\)\\vert\$, with \$y\$ the energy from which "
               r"particles reach \$x\$ within the step", L.line("The source integral there is", "reference"))
    lo, hi, sign, ilo, ihi, div = m.groups()
    names = {"x": xs, "y": YXT, "x'": s}
    weight = sp.exp((1 if sign == "+" else -1) * (antider(names[ihi]) - antider(names[ilo])))
    return names[lo], names[hi], q(s) * weight / a(names[div])


def readme_regime4():
    """(README's, AM3's) particles that cool into x from y (Sign of the cooled particles in regime 4)."""
    m = _match(r"The particles that cool into bin \$x\$ from \$y\$ are \$(.*?)\$; AM³ adds \$(.*?)\$\.",
               L.line("**Sign of the cooled particles in regime 4.**", "reference"))
    return tuple(L.to_sympy(tex, {"x": xs, "y": YXT}, funcs={"n": n0, "A": lambda u: -a(u)}) for tex in m.groups())


# --- the checks: the solver ---

@derivation("am3.green")
def check_green():
    """The decaying weight exp(-int_x^x' alpha/|A|) of the injection at x' > x solves the cooling equation; AM3's exp(+...) does not."""
    am3_sign, sign = readme_green_signs()
    W = lambda u, v, sg: sp.exp(sg * (S(v) - S(u)))   # the README's exp(-+int alpha/|A| dx) along the path from v down to u
    cooled = readme_regime4()[0]

    def results(sg, am3):
        out = [solves("README's weight: n = n(y) A(y)/A(x) W(x, y) + int_x^y q(x') W(x, x') dx'/|A(x)| solves the cooling "
                      "equation from n(x, 0) = n_0(x)", cooled * W(xs, YXT, sg), q(s) * W(xs, s, sg) / a(xs), xs, YXT,
                      start=n0(xs))]
        itg = residual(0, q(s) * W(xs, s, am3) / a(xs), xs, YXT)[1]
        out.append(exact_fails(f"AM3's weight does not solve it (the integrand leaves {sp.factor(zero_numerator(itg))})", itg))
        return out
    out = results(sign, am3_sign)
    with L.patched(r"instead of the decaying $\exp(-\int", r"instead of the decaying $\exp(+\int", "reference"):
        out.append(must_fail("control: the decaying weight with + in the README", results(*readme_green_signs()[::-1])))
    return out


@derivation("am3.source_integral")
def check_source_integral():
    """int_x^y q(x') exp(-int_x^x' alpha/|A| dx'') dx'/|A(x)| is the injection term of the solution along the characteristics."""
    rel = T(YXT) - T(xs) - ts

    def result(what):
        lo, hi, g = readme_source()
        return solves(what, 0, g, lo, hi, start=0)
    out = [exact_all("y: d/dt and d/dx of int_x^y dx'/|A| - t vanish (y(x, 0) = x), so dy/dt = |A(y)|, dy/dx = |A(y)|/|A(x)|",
                     sp.diff(rel, ts), sp.diff(rel, xs)),
           result("README's integral solves the cooling equation with n(x, 0) = 0")]
    with L.patched(r"$dx'/\vert A(x)\vert$, with $y$", r"$dx'/\vert A(x')\vert$, with $y$", "reference"):
        out.append(must_fail("control: dx'/|A(x')|", [result("broken")]))
    with L.patched(r"\exp(-\int_x^{x'} \alpha", r"\exp(-\int_{x'}^{x} \alpha", "reference"):
        out.append(must_fail("control: the inner integral from x' down to x", [result("broken")]))
    return out


@derivation("am3.zfactor")
def check_zfactor():
    """AM3's sum of q(x') exp(-int_x^{x'+Delta} alpha/|A|) Delta is z/(e^z - 1) times the source integral, z = Delta alpha/|A|."""
    k, D, q0, a0 = sp.symbols("k Delta q_0 a_0", positive=True)   # alpha/|A| = k, q = q0 and |A| = a0, constant
    xp = sp.Symbol("x'", real=True)

    def ratio_minus_factor(cells):
        """AM3's sum over the grid points x' = x + j Delta below y = x + cells Delta, over the README's integral
        (both without the common 1/|A(x)|), minus the README's z/(e^z - 1), with k = w/Delta."""
        rule = _match(r"a sum of \$q\(x'\) \\exp\(([-+])\\int_x\^\{([^}]*)\} \\alpha/\\vert A\\vert\)\\ \\Delta\$ over the grid "
                      r"points \$x'\$ below \$y\$", L.line("**AM³'s quadrature:**", "reference"))
        upper = L.to_sympy(rule.group(2), {"x'": xp, r"\Delta": D})
        point = q0 * sp.exp((1 if rule.group(1) == "+" else -1) * k * (upper - xs)) * D
        _, _, g = readme_source(antider=lambda u: k * u)
        integral = sp.integrate(g.replace(q, lambda u: q0).replace(a, lambda u: a0), (s, xs, xs + cells * D)) * a0
        zdef = L.to_sympy(L.sides(L.inline(r"z = \Delta\thinspace \alpha/\vert A\vert \gtrsim 1", "reference").split(r"\gtrsim")[0])[1],
                          {r"\Delta": D, r"\alpha": k * a0, r"\vert A\vert": a0})
        factor = L.to_sympy(L.inline(r"z/(e^z - 1)", "reference"), {"z": zdef})
        return by_exp((sum(point.subs(xp, xs + j * D) for j in range(cells)) / integral - factor).subs(k, w / D))
    out = [exact_all("q and alpha/|A| constant, y = x + N Delta, N = 1 to 5 cells (each point attenuated by the cell above it): "
                     "AM3's sum / the README integral = z/(e^z - 1)", *[ratio_minus_factor(nc) for nc in range(1, 6)])]
    zz = sp.Symbol("z", positive=True)
    first = sp.series(L.to_sympy(L.inline(r"z/(e^z - 1)", "reference"), {"z": zz}), zz, 0, 2).removeO()
    out.append(exact("the rule is first order: z/(e^z - 1) = 1 - z/2 + O(z^2)", first - (1 - zz / 2)))
    # the README's steady-state numbers (a simulation) against the one-cell factor, Delta = 0.1 (bin i at gamma = e^(i Delta))
    delta = L.to_sympy(L.sides(L.inline(r"\Delta = 0.1", "reference"))[1], {})
    line = L.line("**Escape-dominated cells**", "reference")
    m = _match(r"they are \$(.*?)\$ of the exact density just above regime 1 \(escape (\d+) times faster than cooling\), "
               r"([\d.]+) at 1 GeV \(\$\\vert A\\vert/\\alpha = ([\d.]+)\$\), ([\d.]+) at 3 GeV and ([\d.]+) at 10 GeV", line)
    assert "electrons with synchrotron cooling" in line   # |A| = |dE/dt|/E ∝ E, escape constant
    fac = lambda zv: float(zv / (sp.exp(zv) - 1))
    ratio = float(L.to_sympy(m.group(4), {}))
    ours = [fac(delta / (ratio * e_)) for e_ in (1, 3, 10)]
    theirs = [float(m.group(i)) for i in (3, 5, 6)]
    worst = max(abs(o / t_ - 1) for o, t_ in zip(ours, theirs))
    edge, edge_readme = fac(delta * int(m.group(2))), float(L.to_sympy(m.group(1).replace(r"\times", r"\cdot"), {}))
    out.append(Result("the README's steady-state densities (simulation) are about z/(e^z - 1): 1, 3, 10 GeV within 10 %, "
                      "just above regime 1 within a factor 2", worst < 0.1 and 0.5 < edge / edge_readme < 2,
                      f"numeric: {', '.join(f'{o:.3f}' for o in ours)} (README {', '.join(m.group(i) for i in (3, 5, 6))}, "
                      f"largest deviation {100 * worst:.1f} %); {edge:.2g} at z = {float(delta) * int(m.group(2)):g} "
                      f"(README {edge_readme:.0e})"))
    with L.patched(r"\exp(-\int_x^{x'+\Delta} \alpha/\vert A\vert)", r"\exp(-\int_x^{x'} \alpha/\vert A\vert)", "reference"):
        out.append(must_fail("control: each point attenuated only to its own position",
                             [exact("broken", ratio_minus_factor(1))]))
    return out


@derivation("am3.regime4")
def check_regime4():
    """The particles that cool into x from y are n(y) A(y)/A(x) (number conserved along the characteristics), not -n(y) A(y)/A(x)."""
    ours, am3 = readme_regime4()
    lo, hi, g = readme_source(antider=lambda u: 0)   # regime 4: no escape
    out = [exact("n(y) A(y)/A(x) = n_0(y) dy/dx: the particles between y(x) and y(x + dx) at the start", ours - n0(YXT) * sp.diff(YXT, xs)),
           solves("n(y) A(y)/A(x) + int_x^y q dx'/|A(x)| solves dn/dt = -d(A n)/dx + q from n(x, 0) = n_0(x)", ours, g, lo, hi,
                  esc=lambda u: 0, start=n0(xs)),
           exact_fails("AM3's -n(y) A(y)/A(x) starts from -n_0(x)", at_start(am3) - n0(xs))]
    with L.patched(r"are $n(y)\thinspace A(y)/A(x)$", r"are $n(y)\thinspace A(x)/A(y)$", "reference"):
        out.append(must_fail("control: n(y) A(x)/A(y)", [exact("broken", readme_regime4()[0] - n0(YXT) * sp.diff(YXT, xs))]))
    return out


# --- the checks: Chang & Cooper ---

CC_LINE = "the weights of AM³'s regime-2 solver"


def cc_weights():
    """The citation's delta and w (as LaTeX)."""
    return _match(r"the weights of AM³'s regime-2 solver: \$\\delta = (.*?)\$, \$w = (.*?)\$ \(eq\. 18\), for ",
                  L.line(CC_LINE, "citations")).groups()


def cc_delta(wv):
    return L.to_sympy(cc_weights()[0], {"w": wv})


def cc_w():
    return L.to_sympy(cc_weights()[1], {r"\Delta x": dx, "B": B, "C": C})


@derivation("am3.chang_cooper")
def check_chang_cooper():
    """delta = 1/w - 1/(e^w - 1), w = Delta x B/C, is Chang & Cooper's eq. 18: their eq. 17 then gives the exact equilibrium e^-w."""
    skip = without_citations("Chang & Cooper's eqs. 1, 15-19")
    if skip:
        return skip
    line = L.line(CC_LINE, "citations")
    u, du, u0, u1 = sp.symbols("u du u_j u_j+1")

    def step():
        """Eq. 17 (p. 9): u_{j+1}/u_j = (C/Dx - delta B)/(C/Dx + (1 - delta) B) at equilibrium, with the citation's
        delta and w, minus the exact e^-w of eq. 15, in w (C = Dx B/w)."""
        d = cc_delta(cc_w())
        ratio = (C / dx - d * B) / (C / dx + (1 - d) * B)
        return by_exp((ratio - sp.exp(-cc_w())).subs(C, dx * B / w))
    out = [exact("delta(w) = eq. 18, 1/w - 1/(exp(w) - 1) (p. 10)", cc_delta(w) - (1 / w - 1 / (sp.exp(w) - 1))),
           exact("w = Delta x B/C = eq. 18's w_j = Delta x B_j+1/2/C_j+1/2", cc_w() - dx * B / C),
           exact("eq. 18 makes eq. 17's equilibrium ratio (p. 9) the exact exp(-Delta x B/C) of eq. 15", step())]
    # eq. 1 (p. 1): du/dt = A(x)^-1 d/dx [B u + C du/dx]; the citation's current, zero for u = exp(-B x/C)
    cur = _match(r"for \$\\partial u/\\partial t\$ \$= A\^\{-1\}\\ \\partial/\\partial x\\ \((.*?)\)\$ \(eq\. 1; \$B\$ the friction, "
                 r"\$C\$ the dispersion\)", line).group(1)
    J = L.to_sympy(cur, {"B": B, "C": C, "u": u, r"\partial u/\partial x": du})   # u, du: symbols for u(x), du/dx
    ueq = sp.exp(-B * xs / C)
    out.append(exact_all("eq. 1's current B u + C du/dx vanishes for u = exp(-B x/C), which falls by exp(-w) over Delta x (eq. 15)",
                         J.subs({du: sp.diff(ueq, xs), u: ueq}),
                         by_exp((ueq.subs(xs, xs + dx) / ueq - sp.exp(-cc_w())).subs(C, dx * B / w))))
    lim = _match(r"\$\\delta\$ falling from (\S+) to (\S+) as \$w\$ goes from (\S+) to \$\\infty\$", line).groups()
    at0, atoo, start = (L.to_sympy(v.strip("$"), {}) for v in lim)
    d = cc_delta(w)
    limits = (sp.limit(d, w, start, "+"), sp.limit(d, w, sp.oo))
    slope = zero_numerator(by_exp(sp.diff(d, w) - (1 / (4 * sp.sinh(w / 2)**2) - 1 / w**2).rewrite(sp.exp)))
    dfun = sp.lambdify(w, sp.diff(d, w), "mpmath")
    with mp.workdps(50):
        falls = all(dfun(mp.mpf(10) ** (k / mp.mpf(4))) < 0 for k in range(-32, 13))
    out.append(Result(f"delta falls from {at0} to {atoo} as w goes from {start} to oo (p. 10): the limits, and d delta/dw = "
                      "1/(4 sinh^2(w/2)) - 1/w^2 < 0 (sinh u > u)", limits == (at0, atoo) and slope == 0 and falls,
                      f"exact: limits {limits[0]}, {limits[1]}; slope identity {slope}; numeric: negative for w = 1e-8 to 1e3"))
    small = _match(r"AM³ passes its cooling rate for \$B\$ and \$(.*?)\$ for \$C\$, so \$\\delta \\approx 0\$ and the flux "
                   r"between two cells comes from the upper one", line).group(1)
    flux = (((1 - d) * B + C / dx) * u1 - (C / dx - d * B) * u0).subs(C, dx * B / w)   # eq. 16 (p. 9)
    eq19 = (C / dx * w / (sp.exp(w) - 1) * (sp.exp(w) * u1 - u0)).subs(C, dx * B / w)   # eq. 19's bracket: (C/Dx) W (e^w u_j+1 - u_j)
    bound = L.to_sympy(small, {}) / sp.Rational(1, 10**50)   # 1/w = C/(Delta x B) at Delta x B = 1e-50
    out.append(Result(f"eq. 16 with the citation's delta is eq. 19's current (C/Delta x) W (e^w u_j+1 - u_j), W = w/(e^w - 1) "
                      f"(p. 10); C = {small}: w >> 1, 0 < delta < 1/w (1e-50 at Delta x B = 1e-50), and the current tends to "
                      "B u_j+1 (from the upper cell)",
                      zero_numerator(by_exp(flux - eq19)) == 0 and zero_numerator(by_exp(1 / w - d - 1 / (sp.exp(w) - 1))) == 0
                      and sp.limit(eq19, w, sp.oo) == B * u1 and bound == sp.Rational(1, 10**50),
                      "exact: the current, 1/w - delta = 1/(e^w - 1) > 0, the limit"))
    with L.patched(r"$\delta = 1/w - 1/(e^w - 1)$", r"$\delta = 1/w + 1/(e^w - 1)$", "citations"):
        out.append(must_fail("control: delta = 1/w + 1/(e^w - 1)", [exact("broken", step())]))
    return out


# --- the checks: inverse Compton ---

@derivation("am3.ic_F")
def check_ic_F():
    """F(eps, gamma) and gbar = gamma + (gamma^2 - 1)^(1/2) are Jones' eq. 13 with alpha = eps (and the AM3 paper's eqs. A20-A21)."""
    gb = sp.Symbol("gbar", positive=True)
    jones = jones_F(eps, gam, f1_, f2_, gb)
    gk = sp.Symbol("gamma_X", positive=True)
    k_F = L.to_sympy(K_F.split(" = ")[1], {r"\gamma_X": gam, r"\bar{\gamma}_X": gb, r"\bar{\gamma_X}": gb, r"\varepsilon": eps},
                     funcs={"f_1": f1_, "f_2": f2_})
    k_gbar = L.to_sympy(K_GBAR.split("&=")[1], {r"\gamma_X": gk})
    out = [exact("README gbar = Jones' gamma (1 + beta) (gamma = (G + 1/G)/2: both G)",
                 readme_gbar(GAMMA_G) - jones_gbar(GAMMA_G)),
           exact("README F = eq. 13 with alpha = eps (any f1, f2)", readme_F(f1_, f2_, gam, eps, gb) - jones),
           in_source("eqs. A20-A21 of the AM3 paper as transcribed", KLINGER, K_F, K_GBAR),
           exact_all("the AM3 paper's F and gbar (eqs. A20-A21) are the same", k_F - jones,
                     (k_gbar - readme_gbar(gk)).subs(gk, GAMMA_G))]
    with L.patched(r"$- \varepsilon$ $[f_2(", r"$+ \varepsilon$ $[f_2(", "reference"):
        out.append(must_fail("control: + eps [f2 ...]", [exact("broken", readme_F(f1_, f2_, gam, eps, gb) - jones)]))
    return out


def readme_loss():
    """(prefactor, integrand) of the README's -dgamma/dt (Kept as in AM3), with n and F functions."""
    m = _match(r"The inverse-Compton loss \$-\\dot\\gamma\$ \$= (.*?) \\int d\\varepsilon\\ (.*?)\$ \$(.*?)\$ \(\$n\$ per unit "
               r"\$\\varepsilon\$, \$\\varepsilon\$ in units of the particle's rest energy",
               L.line("The ultra-relativistic forms of the continuous losses.", "reference"))
    tok = {r"\sigma_T": sT, "c": c, r"\varepsilon": eps, r"\gamma": gam}
    return L.to_sympy(m.group(1), tok), L.to_sympy(m.group(2) + " " + m.group(3), tok, funcs={"n": n_, "F": F_})


@derivation("am3.ic_loss")
def check_ic_loss():
    """-dgamma/dt = (3/16) sigma_T c int deps n F/(gamma^2 eps^2) is beta times Jones' eq. 13 ((1/2) pi r_0^2 = (3/16) sigma_T)."""
    m = sp.Symbol("m", positive=True)
    radius = sp.solve(sp.Eq(klinger_sigma(m), sp.Rational(8, 3) * sp.pi * r0**2), r0)
    r0_of_sT = sp.sqrt(3 * sT / (8 * sp.pi))
    pre, integrand = readme_loss()
    p = readme_p(GAMMA_G)
    bg2 = L.sides(L.inline(r"\beta\gamma^2 = \gamma p", "reference"))
    out = [exact("eq. A1's sigma_T = (8 pi/3) r_0^2 with Jones' r_0 = e^2/m c^2 (p. B1307), so (1/2) pi r_0^2 = (3/16) sigma_T",
                 sum(r_ - sp.Symbol("q_e", positive=True)**2 / (m * c**2) for r_ in radius)),
           exact("README integrand = beta x eq. 13's (alpha = eps, n(alpha) = n(eps); AM3 drops 1/beta)",
                 (pre * integrand - jones_beta(gam) * jones_loss(eps, gam, n_(eps), F_(eps, gam), r0_of_sT)).subs(gam, GAMMA_G)),
           exact("beta gamma^2 = gamma p (README, p = (gamma^2 - 1)^(1/2)): eq. 13's denominator is gamma p",
                 (L.to_sympy(bg2[0], {r"\beta": jones_beta(GAMMA_G), r"\gamma": GAMMA_G})
                  - L.to_sympy(bg2[1], {r"\gamma": GAMMA_G, "p": p})))]
    with L.patched(r"$= (3/16)\ \sigma_T c \int d\varepsilon", r"$= (3/8)\ \sigma_T c \int d\varepsilon", "reference"):
        pre_b, integrand_b = readme_loss()
        out.append(must_fail("control: (3/8) sigma_T", [exact("broken", (pre_b * integrand_b - jones_beta(gam) * jones_loss(
            eps, gam, n_(eps), F_(eps, gam), r0_of_sT)).subs(gam, GAMMA_G))]))
    return out


# --- the Klein-Nishina loss in isotropic photons (from the cross section, not from a paper) ---

xk, uk = sp.symbols("x u", positive=True)


def kn_density(g, e):
    """K(x), with the loss rate per photon (per unit photon density) P_KN = int_{e/gbar}^{e gbar} K(x) dx of a
    particle of Lorentz factor g in isotropic photons of energy e (in units of the particle's rest energy):
      P_KN = c int_-1^1 dmu/2 (1 - beta mu) int dOmega' dsigma_KN/dOmega' (e_1 - e).
    x = g e (1 - beta mu) is the photon's rest-frame energy, so dmu (1 - beta mu)/2 = x dx/(2 g^2 e^2 beta), and
    dsigma_KN/dOmega' = (r_0^2/2) r^2 (r + 1/r - sin^2 theta'), r = 1/u, u = 1 + x (1 - cos theta'). Averaged over the
    azimuth, the scattered photon's lab energy is e_1 = g x r (1 + beta mu' cos theta'), with the incident photon's
    rest-frame direction mu' = (mu - beta)/(1 - beta mu) = [e/(g x) - 1]/beta. Over cos theta' (dcos = -du/x), the
    integrand is a Laurent polynomial in u, integrated term by term from 1 to 1 + 2x."""
    b = sp.sqrt(g**2 - 1) / g
    cos = 1 - (uk - 1) / xk
    mu_rest = (e / (g * xk) - 1) / b
    integrand = (2 * sp.pi * r0**2 / 2 / uk**2 * (1 / uk + uk - (1 - cos**2)) * (g * xk / uk * (1 + b * mu_rest * cos) - e)) / xk
    num, den = sp.fraction(sp.cancel(sp.together(integrand * uk**6)))
    assert uk not in den.free_symbols, den
    top = 1 + 2 * xk
    inner = sum(cf / den * (sp.log(top) if k - 6 == -1 else (top**(k - 5) - 1) / (k - 5))
                for (k,), cf in sp.Poly(num, uk).terms())
    return c * xk / (2 * g**2 * e**2 * b) * inner


def kn_direct(g, e):
    """P_KN by direct quadrature over mu and cos theta' (sigma_T = c = 1, r_0^2 = 3/(8 pi)), at the working precision."""
    b = mp.sqrt(1 - 1 / g**2)

    def f(mu, cos):
        x_ = g * e * (1 - b * mu)
        r = 1 / (1 + x_ * (1 - cos))
        dsigma = 3 / (8 * mp.pi) / 2 * r * r * (r + 1 / r - (1 - cos * cos)) * 2 * mp.pi
        return (1 - b * mu) / 2 * dsigma * (g * x_ * r * (1 + b * (mu - b) / (1 - b * mu) * cos) - e)
    return mp.quad(f, [-1, 1], [-1, 1])


def kn_precision():
    """The precision of the direct integration that Kept as in AM3 states."""
    m = _match(r"It is checked against a direct integration of the Klein–Nishina cross section to \$10\^\{(-\d+)\}\$",
               L.line("The ultra-relativistic forms of the continuous losses.", "reference"))
    return mp.mpf(10)**int(m.group(1))


@derivation("am3.ic_klein_nishina")
def check_ic_klein_nishina():
    """(3/16) sigma_T c F/(gamma^2 eps^2) per photon is beta times the Klein-Nishina loss in isotropic photons: exactly, and by direct integration."""
    gb = sp.Symbol("gbar", positive=True)
    phi = lambda u_: gam * f1_(u_) - eps * f2_(u_)                                          # noqa: E731
    K = kn_density(GAMMA_G, eps).subs(r0, sp.sqrt(3 * sT / (8 * sp.pi)))
    dphi = sp.expand_func(sp.diff(GAMMA_G * jones_f1(xk) - eps * jones_f2(xk), xk))      # f1, f2 of Jones' eqs. 14-15
    b = jones_beta(GAMMA_G)

    def per_photon(F):
        """The README's loss for n(eps') = delta(eps' - eps), with F(eps, gamma) -> F."""
        pre, integrand = readme_loss()
        return (pre * integrand.subs(n_(eps), 1).replace(F_, lambda e_, g_: F)).subs(gam, GAMMA_G)
    out = [exact("README F = Phi(eps gbar) - Phi(eps/gbar), Phi(x) = gamma f1(x) - eps f2(x) (any f1, f2)",
                 readme_F(f1_, f2_, gam, eps, gb) - (phi(eps * gb) - phi(eps / gb))),
           exact_all("x = gamma eps (1 - beta mu) runs from eps gbar (mu = -1) to eps/gbar (mu = 1), with the README's gbar",
                     GAMMA_G * eps * (1 + b) - eps * readme_gbar(GAMMA_G), GAMMA_G * eps * (1 - b) - eps / readme_gbar(GAMMA_G)),
           exact("README's loss with F -> dPhi/dx = beta K(x): the loss per photon is beta x the Klein-Nishina loss "
                 "int_{eps/gbar}^{eps gbar} K dx (any gamma, eps)", per_photon(dphi) - b * K)]
    tol = kn_precision()
    F = readme_F_mp()
    pre = readme_loss()[0].subs({sT: 1, c: 1})
    worst = mp.mpf(0)
    with mp.workdps(20):
        for g_ in ("1.001", "2", "10", "100"):
            for e_ in ("1e-3", "10"):
                g_, e_ = mp.mpf(g_), mp.mpf(e_)
                with mp.workdps(50):
                    ours = mp.mpf(sp.N(pre, 30)) * F(e_, g_) / (g_**2 * e_**2) / (mp.sqrt(g_**2 - 1) / g_)
                worst = max(worst, abs(ours / kn_direct(g_, e_) - 1))
    out.append(Result(f"the README's loss/beta against a direct 2-D integration of the Klein-Nishina cross section (20 digits; "
                      f"F at 50) to {mp.nstr(tol, 1)}", worst <= tol,
                      f"numeric: max rel. difference {mp.nstr(worst, 3)} at 8 points, gamma 1.001 to 100, eps 1e-3 and 10"))
    with L.patched(r"$= (3/16)\ \sigma_T c \int d\varepsilon", r"$= (3/8)\ \sigma_T c \int d\varepsilon", "reference"):
        out.append(must_fail("control: (3/8) sigma_T", [exact("broken", per_photon(dphi) - b * K)]))
    return out


def kn_limit_claims():
    """(the deviation in %, the limit's two LaTeX pieces, the exponent of the lowest gamma eps) of 'Against inverse Compton'."""
    return _match(r"is within \$([\d.]+)\\text\{％\}\$ of its Klein–Nishina limit, \$-\\dot\\gamma\$ \$= (.*?)\$ \$(.*?)\$, for "
                  r"\$\\gamma\\varepsilon \\ge 10\^(\d+)\$", L.line("of its Klein–Nishina limit,", "checks")).groups()


@derivation("am3.ic_kn_limit")
def check_ic_kn_limit():
    """The Klein-Nishina loss (the README's over beta) is within 0.22 % of (3/8) c sigma_T n [ln(4 gamma eps) - 11/6]/eps for gamma eps >= 1e3."""
    nn = sp.Symbol("n", positive=True)
    F = readme_F_mp()
    pre = readme_loss()[0].subs({sT: 1, c: 1})

    def deviations():
        pct, front, bracket, power = kn_limit_claims()
        limit = sp.lambdify((gam, eps), (L.to_sympy(front, {"c": c, r"\sigma_T": sT, "n": nn, r"\varepsilon": eps})
                                         * L.to_sympy(bracket, {r"\gamma": gam, r"\varepsilon": eps})).subs({c: 1, sT: 1, nn: 1}),
                            "mpmath")
        out = []
        with mp.workdps(50):
            g_ = mp.mpf(10)**10
            for P in (int(power), int(power) + 1, 5, 6, 9):
                e_ = mp.mpf(10)**P / g_
                exact_ = mp.mpf(sp.N(pre, 30)) * F(e_, g_) / (g_**2 * e_**2) / (mp.sqrt(g_**2 - 1) / g_)
                out.append(abs(exact_ / limit(g_, e_) - 1))
        return pct, power, out

    def result():
        pct, power, devs = deviations()
        return Result(f"|exact/limit - 1| = {pct} % at gamma eps = 1e{power}, decreasing above (gamma = 1e10)",
                      f"{100 * float(devs[0]):.2f}" == pct and all(a_ > b_ for a_, b_ in zip(devs, devs[1:])),
                      "numeric: " + ", ".join(f"{mp.nstr(100 * d, 3)} %" for d in devs) + " at gamma eps = 1e3, 1e4, 1e5, 1e6, 1e9")
    out = [result()]
    with L.patched(r"is within $0.22\text{％}$ of its Klein–Nishina limit", r"is within $0.20\text{％}$ of its Klein–Nishina limit",
                   "checks"):
        out.append(must_fail("control: 0.20 %", [result()]))
    return out


def citations_jones():
    """The pieces of the Jones (1965) row of the citation index."""
    line = L.line("Jones 1965, Phys. Rev. 137, B1306", "citations")
    m = _match(r"the loss rate \$(.*?) \\int d\\alpha\\ (.*?)\$ \$(.*?)\$ \(eq\. 13; \$(.*?)\$, as the README says\) with \$f_1\$, "
               r"\$f_2\$ ending in \$(.*?)\$ and \$(.*?)\$ \(eqs\. 14–15\)", line)
    t = _match(r"the Taylor coefficients of \$f_1\$, \$f_2\$ \(Table I, to \$z\^\{(\d+)\}\$\), of which AM³ uses the first (\w+) "
               r"\(\$([\d.]+)\\text\{％\}\$ off at \$z = ([\d.]+)\$; .*?, which Table I gives to \$(.*?)\$\)", line)
    return m.groups(), t.groups()


WORDS = {"two": 2, "three": 3, "four": 4}


def table_I_precision():
    """The largest relative difference between Table I and the exact Taylor coefficients (to z^10)."""
    return max(abs(mp.mpf(TABLE_I[nm][k]) / (mp.mpf(cf.p) / cf.q) - 1) for nm in ("A", "B")
               for k, cf in enumerate(taylor(nm)) if cf != 0)


def series_error(name, terms, zv):
    f = MP["f1" if name == "A" else "f2"]
    with mp.workdps(40):
        return mp_series(name, terms)(mp.mpf(zv)) / f(mp.mpf(zv)) - 1


@derivation("am3.ic_jones13")
def check_ic_jones13():
    """citations.md quotes Jones' eq. 13 and the endings of eqs. 14-15; Table I gives the exact coefficients to 3e-6."""
    skip = without_citations("Jones' eqs. 13-15 and Table I")
    if skip:
        return skip
    (pre, dens, rest, bg2, end1, end2), (order, terms, pct, zv, prec) = citations_jones()
    Fs = sp.Symbol("F")   # F(alpha, gamma), substituted after parsing
    tok = {"r_0": r0, "c": c, r"\alpha": al_, r"\beta": jones_beta(gam), r"\gamma": gam, "F": Fs}
    quoted = (L.to_sympy(pre, tok) * L.to_sympy(dens, tok, funcs={"n": n_}) * L.to_sympy(rest, tok)).subs(Fs, F_(al_, gam))
    p = readme_p(GAMMA_G)
    zeros = [k for nm in ("A", "B") for k, cf in enumerate(taylor(nm)) if (cf == 0) != (float(TABLE_I[nm][k]) == 0)]
    prec_v, worst = float(L.to_sympy(prec.replace(r"\times", r"\cdot"), {})), table_I_precision()
    err = series_error("A", WORDS[terms], zv)
    out = [exact("the quoted loss rate = eq. 13 (p. B1308)", (quoted - jones_loss(al_, gam, n_(al_), F_(al_, gam)))
                 .subs(gam, GAMMA_G)),
           exact("beta gamma^2 = gamma p", (L.to_sympy(L.sides(bg2)[0], {r"\beta": jones_beta(GAMMA_G), r"\gamma": GAMMA_G})
                                            - L.to_sympy(L.sides(bg2)[1], {r"\gamma": GAMMA_G, "p": p}))),
           exact_all("f1 ends in +2 Li2(-2z), f2 in +Li2(-2z) (eqs. 14-15)", li2_term(end1) - (jones_f1(z) - jones_f1(z, 0)),
                     li2_term(end2) - (jones_f2(z) - jones_f2(z, 0))),
           Result(f"Table I, to z^{order}: the exact coefficients of eqs. 14-15 to {prec_v:.0e}",
                  int(order) == 10 and not zeros and f"{float(worst):.0e}" == f"{prec_v:.0e}",
                  f"numeric: largest relative difference {mp.nstr(worst, 3)}"),
           Result(f"AM3's first {terms} terms of f1 are {pct} % off at z = {zv}", f"{100 * float(err):.1f}" == pct,
                  f"numeric: {mp.nstr(100 * err, 4)} %")]
    with L.patched(r"$\tfrac12\pi r_0^2 c \int d\alpha", r"$\tfrac14\pi r_0^2 c \int d\alpha", "citations"):
        (pre_b, dens_b, rest_b, *_), _ = citations_jones()
        broken = (L.to_sympy(pre_b, tok) * L.to_sympy(dens_b, tok, funcs={"n": n_}) * L.to_sympy(rest_b, tok)).subs(Fs, F_(al_, gam))
        out.append(must_fail("control: (1/4) pi r_0^2", [exact("broken", (broken - jones_loss(al_, gam, n_(al_), F_(al_, gam)))
                                                               .subs(gam, GAMMA_G))]))
    return out


def sign_change(F, g, lo="1e-12", hi="1e12"):
    """The photon energy above which F(eps, gamma) < 0 (bisection in ln eps)."""
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    for _ in range(70):
        mid = mp.sqrt(lo * hi)
        lo, hi = (mid, hi) if F(mid, g) > 0 else (lo, mid)
    return mp.sqrt(lo * hi)


@derivation("am3.ic_negative")
def check_ic_negative():
    """F < 0 above eps = (4/3)(gamma^2 - 1) as gamma -> 1 (the series of eqs. 14-15), 0.87 gamma at gamma = 3, 0.98 gamma at 10."""
    tg = sp.Symbol("t", positive=True)
    u = sp.Symbol("u", positive=True)

    def claims():
        """(eps_0/gamma at one gamma, that gamma, at another, that gamma, the boundary as gamma -> 1), as LaTeX."""
        return _match(r"\$F\$ is negative where the photons are more energetic than the particle \(\$\\varepsilon \\gt ([\d.]+)\\gamma\$ "
                      r"at \$\\gamma = (\d+)\$, \$\\varepsilon \\gt ([\d.]+)\\gamma\$ at \$\\gamma = (\d+)\$, tending to "
                      r"\$\\varepsilon \\gt \\gamma\$ at large \$\\gamma\$; \$\\varepsilon \\gtrsim (.*?)\$ as \$\\gamma \\to 1\$\)",
                      L.line("The ultra-relativistic forms of the continuous losses.", "reference")).groups()

    def boundary():
        """eps = t (gamma^2 - 1) with t from the README; F(t p^2) to order u^9 for gbar = 1 + u (f1, f2 to z^6
        suffice: z ~ u^2): (lower coefficients, the u^9 coefficient as a polynomial in t, the README's t)."""
        t_readme = sp.simplify(L.to_sympy(claims()[4], {r"\gamma": gam}) / (gam**2 - 1))
        g_ = 1 + u
        Fu = readme_F(poly("A", 6), poly("B", 6), (g_ + 1 / g_) / 2, tg * ((g_ - 1 / g_) / 2)**2, g_)
        ser = sp.series(Fu, u, 0, 10).removeO()
        return [sp.simplify(ser.coeff(u, k)) for k in range(9)], sp.expand(ser.coeff(u, 9)), t_readme
    lower, c9, t_readme = boundary()
    roots = [r_ for r_ in sp.solve(c9, tg) if r_.is_positive]
    out = [Result(f"eps = t (gamma^2 - 1), gamma -> 1 (gbar = 1 + u): F = c9(t) u^9 + O(u^10), c9 changes sign only at the "
                  f"README's t = {t_readme} (F < 0 above)", all(cf == 0 for cf in lower) and roots == [t_readme]
                  and c9.subs(tg, t_readme / 2) > 0 and c9.subs(tg, 2 * t_readme) < 0,
                  f"exact: c9 = {sp.factor(c9)}, positive root {roots}")]
    F = readme_F_mp()
    with mp.workdps(60):
        near = [(g_, sign_change(F, mp.mpf(g_)) / (mp.mpf(g_)**2 - 1)) for g_ in ("1.00001", "1.0001", "1.001")]

    def far_result():
        r_a, g_a, r_b, g_b, _ = claims()
        with mp.workdps(60):
            far = [(g_, sign_change(F, mp.mpf(g_)) / mp.mpf(g_)) for g_ in (g_a, g_b, "100", "1e4")]
        return Result(f"eps_0/gamma = {r_a} at gamma = {g_a}, {r_b} at {g_b}, rising to 1 (gamma = 100, 1e4)",
                      f"{float(far[0][1]):.2f}" == r_a and f"{float(far[1][1]):.2f}" == r_b
                      and all(a[1] < b[1] for a, b in zip(far, far[1:])) and far[-1][1] > 1 - 1e-6 and far[-1][1] <= 1,
                      "numeric: " + ", ".join(mp.nstr(v, 5) for _, v in far))
    out.append(Result(f"the sign change at gamma = 1 + 1e-5, 1e-4, 1e-3: eps_0/(gamma^2 - 1) -> {t_readme}",
                      abs(near[0][1] - mp.mpf(t_readme.p) / t_readme.q) < 1e-3 and near[0][1] > near[1][1] > near[2][1],
                      "numeric (60 digits): " + ", ".join(mp.nstr(v, 6) for _, v in near)))
    out.append(far_result())
    with L.patched(r"$\varepsilon \gt 0.87\gamma$ at $\gamma = 3$", r"$\varepsilon \gt 0.9\gamma$ at $\gamma = 3$", "reference"):
        out.append(must_fail("control: 0.9 gamma at gamma = 3", [far_result()]))
    with L.patched(r"$\varepsilon \gtrsim (4/3)(\gamma^2 - 1)$", r"$\varepsilon \gtrsim (3/4)(\gamma^2 - 1)$", "reference"):
        _, c9_b, t_b = boundary()
        out.append(must_fail("control: (3/4)(gamma^2 - 1)", [Result("broken", c9_b.subs(tg, t_b) == 0, "")]))
    return out


def series_claims():
    """The numbers of 'Series of the inverse-Compton cooling functions' (tests/reference/README.md, "Bugs of AM3
    v1.2.0 that am3_mojo fixes")."""
    line = L.line("**Series of the inverse-Compton cooling functions.**", "reference")
    a_ = _match(r"keeps only the first (\w+) terms of Jones' series \(his Table I\), which are \$([\d.]+)\\text\{％\}\$ \(\$f_1\$\) "
                r"and \$([\d.]+)\\text\{％\}\$ \(\$f_2\$\) too large at \$z = ([\d.]+)\$", line).groups()
    b_ = _match(r"by \$([\d.]+)\\text\{％\}\$ for \$\\gamma \\gtrsim (\d+)\$ \(\$([\d.]+)\\text\{％\}\$ at \$\\gamma = ([\d.]+)\$, "
                r"\$([\d.]+)\\text\{％\}\$ at \$\\gamma = ([\d.]+)\$ and \$([\d.]+)\\text\{％\}\$ at \$\\gamma = ([\d.]+)\$, the "
                r"lowest .*?; \$([\d.]+)\\text\{％\}\$ at \$\\varepsilon\\bar\\gamma = ([\d.]+)\$\)", line).groups()
    c_ = _match(r"am3_mojo uses Jones' series to \$z\^\{(\d+)\}\$ \(with exact coefficients\), accurate to \$(.*?)\$ at "
                r"\$z = ([\d.]+)\$", line).groups()
    return a_, b_, c_


def series_results():
    (terms, p1, p2, zv), (pf, glo, p10, g10, p15, g15, p122, g122, p05, zb05), (order, acc, zacc) = series_claims()
    n3 = WORDS[terms]
    e1, e2 = series_error("A", n3, zv), series_error("B", n3, zv)
    full = int(order)
    acc_v = float(L.to_sympy(acc.replace(r"\times", r"\cdot"), {}))
    errs = [abs(series_error("A", full - 2, zacc)), abs(series_error("B", full - 1, zacc))]
    F = readme_F_mp()
    F3 = readme_F_mp(mp_series("A", n3), mp_series("B", n3))

    def fast(g_, zb):
        """F with the three-term series / F - 1 just below eps gbar = zb."""
        with mp.workdps(40):
            g_ = mp.mpf(g_)
            e_ = mp.mpf(zb) / (g_ + mp.sqrt(g_**2 - 1)) * (1 - mp.mpf(10)**-25)
            return 100 * (F3(e_, g_) / F(e_, g_) - 1)
    big = [fast(g_, zv) for g_ in (glo, str(2 * int(glo)), "100", "1e4", "1e8")]   # decreasing to the limit
    at = {g10: fast(g10, zv), g15: fast(g15, zv), g122: fast(g122, zv)}
    half = fast("100", zb05)
    return [Result(f"AM3's {terms} terms at z = {zv}: f1 {p1} %, f2 {p2} % too large",
                   f"{100 * float(e1):.1f}" == p1 and f"{100 * float(e2):.1f}" == p2,
                   f"numeric: {mp.nstr(100 * e1, 4)} %, {mp.nstr(100 * e2, 4)} %"),
            Result(f"F just below eps gbar = {zv}: {pf} % for gamma >~ {glo} (falling from there), {p10} % at {g10}, {p15} % at "
                   f"{g15}, {p122} % at {g122}; {p05} % at eps gbar = {zb05}",
                   all(f"{float(v):.1f}" == pf for v in big) and all(a > b for a, b in zip(big, big[1:]))
                   and all(f"{float(at[g_]):.1f}" == p_ for g_, p_ in ((g10, p10), (g15, p15), (g122, p122)))
                   and f"{float(half):.1f}" == p05,
                   f"numeric: {', '.join(mp.nstr(v, 5) for v in big)} % (gamma = {glo}, {2 * int(glo)}, 100, 1e4, 1e8), "
                   f"{mp.nstr(at[g10], 4)} %, {mp.nstr(at[g15], 4)} %, {mp.nstr(at[g122], 4)} %, {mp.nstr(half, 3)} % (gamma = 100)"),
            Result(f"the series to z^{order} at z = {zacc}: accurate to {acc_v:.0e}",
                   f"{float(max(errs)):.0e}" == f"{acc_v:.0e}",
                   f"numeric: f1 {mp.nstr(errs[0], 3)}, f2 {mp.nstr(errs[1], 3)}")]


@derivation("am3.jones_series")
def check_jones_series():
    """Jones' series to z^10: exact coefficients from eqs. 14-15 (Table I to 3e-6); AM3's three terms are 2.4 % and 1.2 % off at z = 0.1."""
    A, Bc = taylor("A"), taylor("B")
    out = [Result("eqs. 14-15: f1 = (8/9) z^3 + ..., f2 = (4/3) z^2 + ... with rational coefficients to z^10 (A_0..A_2 = "
                  "B_0 = B_1 = 0, as Table I has)", A[:3] == [0, 0, 0] and Bc[:2] == [0, 0]
                  and all(cf.is_Rational for cf in A + Bc),
                  "exact: A_3..A_10 = " + ", ".join(map(str, A[3:])) + "; B_2..B_10 = " + ", ".join(map(str, Bc[2:]))),
           Result("Table I (p. B1309) prints these to 3e-6", f"{float(table_I_precision()):.0e}" == "3e-06",
                  f"numeric: largest relative difference {mp.nstr(table_I_precision(), 3)}")]
    out += series_results()
    with L.patched(r"which are $2.4\text{％}$ ($f_1$)", r"which are $3.4\text{％}$ ($f_1$)", "reference"):
        out.append(must_fail("control: 3.4 % for f1", series_results()))
    with L.patched(r"for $\gamma \gtrsim 14$ ($2.5", r"for $\gamma \gtrsim 10$ ($2.5", "reference"):
        out.append(must_fail("control: 2.4 % for gamma >~ 10 (2.455 % there)", series_results()))
    return out


@derivation("am3.f12_misprint")
def check_f12_misprint():
    """The AM3 paper's f1, f2 (eqs. A22-A23) have -2 Li2(-2z) for Jones' +2 Li2(-2z), +Li2(-2z), and vanish as 8z, 6z."""
    def results():
        pieces = numerics_point10()
        k1, k2 = klinger_f(K_F1), klinger_f(K_F2)
        printed, plus2, plus1 = (li2_term(pieces[k]) for k in ("printed", "f1", "f2"))
        lin = [L.to_sympy(t_, {"z": z}) for t_ in pieces["lin"]]
        lead = [L.to_sympy(t_, {"z": z}) for t_ in pieces["lead"]]
        return [exact("eq. A22 - eq. 14 = (-2 Li2(-2z)) - (+2 Li2(-2z)), as the README says", k1 - jones_f1(z) - (printed - plus2)),
                exact("eq. A23 - eq. 15 = (-2 Li2(-2z)) - (+Li2(-2z))", k2 - jones_f2(z) - (printed - plus1)),
                exact_all("eqs. A22-A23 as printed: 8z + O(z^2) and 6z + O(z^2)",
                          *[sp.series(k_, z, 0, 2).removeO() - l_ for k_, l_ in zip((k1, k2), lin)]),
                exact_all("eqs. 14-15: (8/9) z^3 + O(z^4) and (4/3) z^2 + O(z^3)",
                          sp.series(jones_f1(z), z, 0, 4).removeO() - lead[0], sp.series(jones_f2(z), z, 0, 3).removeO() - lead[1])]
    out = [in_source("eqs. A22-A23 as transcribed", KLINGER, K_F1, K_F2)] + results()
    # the Thomson limit: with eqs. 14-15, F = (64/9) gamma p^3 eps^3 + ..., and eq. 13 becomes Jones' zero-order eq. 21
    power, cf = thomson_leading(poly("A", 5), poly("B", 5))
    out.append(exact_all("eqs. 14-15: F = (64/9) gamma p^3 eps^3 + O(eps^4), and eq. 13 tends to eq. 21 (p. B1309), "
                         "-dgamma/dt = (32/9) pi r_0^2 c p^2 int eps n deps, the Thomson loss", sp.Integer(power - 3),
                         cf - sp.Rational(64, 9) * GAMMA_G * P_G**3,
                         jones_loss(eps, GAMMA_G, n_(eps), cf * eps**power) - jones_eq21(eps * n_(eps), P_G)))
    out.append(Result(f"eq. 21's {JONES_21} is 32/9 (= (4/3) sigma_T/(pi r_0^2)) to 2e-4", abs(float(JONES_21) * 9 / 32 - 1) < 2e-4,
                      f"numeric: {float(JONES_21) * 9 / 32 - 1:.1e}"))
    k_poly = {nm: sp.series(klinger_f(k_), z, 0, 6).removeO() for nm, k_ in (("A", K_F1), ("B", K_F2))}
    power_k, cf_k = thomson_leading(lambda u_: k_poly["A"].subs(z, u_), lambda u_: k_poly["B"].subs(z, u_))
    left = zero_numerator(cf_k - 16 * GAMMA_G * P_G)
    out.append(Result("eqs. A22-A23: F = 16 gamma p eps + O(eps^2), so they miss the Thomson limit (eq. 13 would give a loss "
                      "∝ int n/eps deps)", power_k == 1 and left == 0, f"exact: lowest order eps^{power_k}, coefficient - 16 gamma p = {left}"))
    with L.patched(r"$+\mathrm{Li}_ 2(-2z)$. They go", r"$+2\ \mathrm{Li}_ 2(-2z)$. They go", "reference"):
        out.append(must_fail("control: +2 Li2(-2z) for f2", results()))
    return out


@derivation("am3.ic_density_misprint")
def check_ic_density_misprint():
    """The AM3 paper's loss (eq. A19) has n/(gamma^2 eps); n per unit eps (its own definition) needs n/(gamma^2 eps^2), as Jones' eq. 13."""
    tok_k = {r"\varepsilon_i": eps, r"\gamma_X": gam, r"\sigma_T": sT, "c": c}
    k_pre = L.to_sympy(L.between(K_LOSS, "= ", r" \int"), tok_k)
    k_density = L.to_sympy(L.between(K_LOSS, r"\mathrm{d}\varepsilon_i ", r" F("), tok_k, funcs={r"n_{\gamma}": n_})
    thomson = sp.Rational(64, 9) * GAMMA_G * P_G**3 * eps**3   # F to lowest order in eps (am3.f12_misprint)
    eq21 = jones_beta(GAMMA_G) * jones_eq21(eps * n_(eps), P_G, sp.sqrt(3 * sT / (8 * sp.pi)))   # beta x eq. 21 per unit eps

    def results():
        printed, needed = numerics_point10()["density"]
        tok = {r"\varepsilon": eps, r"\gamma": gam}
        ours_p, ours_n = (L.to_sympy(t_, tok, funcs={r"n_\gamma": n_}).subs(gam, GAMMA_G) for t_ in (printed, needed))
        jones = jones_beta(GAMMA_G) * jones_loss(eps, GAMMA_G, n_(eps), 1) / (sp.pi * r0**2 * c / 2)   # beta n/(beta gamma^2 eps^2)
        return [exact("the README's printed n_gamma/(gamma^2 eps) = eq. A19's", ours_p - k_density.subs(gam, GAMMA_G)),
                exact("the README's n_gamma/(gamma^2 eps^2) = eq. 13's n(alpha)/(beta gamma^2 alpha^2) x beta", ours_n - jones),
                exact_fails("eq. A19 as printed differs (by a factor eps)", ours_p - ours_n),
                exact("n/(gamma^2 eps^2): (3/16) sigma_T c F n/(gamma^2 eps^2) -> beta x eq. 21, (32/9) pi r_0^2 c p^2 eps n "
                      "per unit eps (its integral, the energy density)", k_pre * thomson * ours_n - eq21),
                exact("n/(gamma^2 eps): eps times that (eps^2 n, not an energy density)", k_pre * thomson * ours_p - eps * eq21)]
    pre, integrand = readme_loss()
    out = [in_source("eq. A19 and the paper's n = d^2N/dE dV, as transcribed", KLINGER, K_LOSS, K_N),
           exact("eq. A19 = eps x the README's loss (Kept as in AM3): the density factor is the only difference",
                 k_pre * k_density * F_(eps, gam) - eps * pre * integrand)] + results()
    with L.patched(r"needs $n_\gamma(\varepsilon)/(\gamma^2\varepsilon^2)$", r"needs $n_\gamma(\varepsilon)/(\gamma^2\varepsilon^3)$", "reference"):
        out.append(must_fail("control: n/(gamma^2 eps^3)", results()))
    return out


@derivation("am3.sigmaT")
def check_sigmaT():
    """The AM3 paper's sigma_T (eq. A1) is ∝ m_X^-2; Sec. A.2 writes m_X^-4. It takes eps in units of m_e c^2."""
    m = sp.Symbol("m_X", positive=True)
    exponent = lambda expr: sp.simplify(m * sp.diff(sp.log(expr), m))

    def results():
        quoted, correct = numerics_point10()["sigma"]
        lhs, rhs = quoted.split(r"\propto")
        assert lhs.strip() == r"\sigma_T", lhs
        return [exact("the README's m_X^-2 is eq. A1's scaling, d ln sigma_T/d ln m_X",
                      exponent(L.to_sympy(correct, {"m_X": m})) - exponent(klinger_sigma(m))),
                exact_fails("the printed sigma_T ∝ m_X^-4 (the README's quotation) differs",
                            exponent(L.to_sympy(rhs, {"m_X": m})) - exponent(klinger_sigma(m))),
                exact("the README quotes Sec. A.2's exponent", exponent(L.to_sympy(rhs, {"m_X": m}))
                      - exponent(L.to_sympy(K_M4.split(r"\propto")[1].split("$")[0], {"m_X": m})))]
    out = [in_source("eq. A1, sigma_T ∝ m_X^-4 (after eq. A18) and eps = E_gamma/m_e c^2 (Sec. A), as transcribed", KLINGER,
                     K_SIGMA, K_M4, K_EPS)] + results()
    # eps in units of M c^2: the README's loss with F to lowest order (am3.f12_misprint), per unit photon energy E (eps = E/(M c^2),
    # n(eps) = M c^2 n_E(E)), times m_X c^2, against beta x Jones' eq. 21 with rho = int E n_E dE and r_0 of the particle
    E, M, mX, me = sp.symbols("E M m_X m_e", positive=True)
    nE = sp.Function("n_E")
    pre, integrand = readme_loss()
    per_eps = pre * integrand.replace(F_, lambda e_, g_: sp.Rational(64, 9) * g_ * (g_**2 - 1)**sp.Rational(3, 2) * e_**3)
    loss = lambda unit: (per_eps.subs(n_(eps), unit * c**2 * nE(E)).subs(eps, E / (unit * c**2)) / (unit * c**2)
                         * mX * c**2).subs(gam, GAMMA_G)
    eq21 = jones_beta(GAMMA_G) * jones_eq21(E * nE(E), P_G, sp.sqrt(3 * sT / (8 * sp.pi)))
    out += [exact("eps in units of the particle's rest energy: the energy loss tends to beta x eq. 21 (the Thomson loss)",
                  loss(mX) - eq21),
            exact("eps in units of m_e c^2 (the paper's): m_X/m_e times that", loss(me) - mX / me * eq21)]
    with L.patched(r"for the $m_X^{-2}$ of its own definition", r"for the $m_X^{-1}$ of its own definition", "reference"):
        out.append(must_fail("control: m_X^-1", results()))
    return out


if __name__ == "__main__":
    run()
