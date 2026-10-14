"""Derivation checks of the triplet pair production equations of README.md (Triplet pair production) against:
  - Lee, Lyubyakin & Stotsky (2021), the TeX source of arXiv:2010.15430v1, whose expansions are read and parsed.
    Sec. 4 of the journal version (JHEP 01 (2021) 144) differs only in numbering, a footnote and references: its
    eqs. 4.11-4.13 are v1's eqs. 4.10-4.11, as v1 gives eqs. 4.12-4.13 one number;
  - Lee (1998, arXiv:astro-ph/9604098, eq. 25) and Anguelov et al. (1999, arXiv:astro-ph/0001252, eq. 9), from their
    TeX sources;
  - Blumenthal & Gould (1970, eq. 2.57, p. 244), Mastichiadis (1991, eqs. 26-28, pp. 240-241) and Dermer &
    Schlickeiser (1991, eqs. 6-7, p. 415), transcribed below from their scans;
  - Haug (1975, Table 1), transcribed below from the journal's PDF.
In both versions, Lee et al.'s threshold expansion (eq. 4.5) lacks the factor alpha^3 of their exact cross section
(eq. 4.4) and high-energy expansions (eqs. 4.11-4.13). The README's sigma/(alpha r_e^2) has it (alpha r_e^2 =
alpha^3/m_e^2), as tr.lee_exact confirms. Their eq. 4.3 lacks a factor 2 in J_11, which their differential system
(eq. 4.2) requires (tr.lee_J11).

The exact cross section (eq. 4.4) is evaluated here at 30 digits: J_1, J_2 and its bracket are parsed from the TeX
source, and J_3 ... J_12 are the integrals of eq. 4.3 (the kernels below). With J_11's factor 2, it gives the figures
under Results in tests/literature_checks/README.md. triplet/compare_exact.py (the code's cross section) and
triplet/machinery.py (the Monte Carlo at high k) use this evaluation, the README's expansions and eq. 4.12.

G_r and the kinetic terms (README, Rates and Kinetic terms) are checked against a direct count of the collisions of
leptons (gamma' >> 1) with isotropic photons, for any sigma, f_r and mean fractions (collisions(), direct_source(),
direct_loss(), direct_sink()). The counts the kinetic terms conserve (one e+, one e- and one photon per interaction;
the pair's energy) are checked on lines of leptons and photons (densities N delta(ln gamma - ln gamma_i)).

The README's k is the photon's energy in the lepton's rest frame, p_1.k, for beta -> 1; its threshold makes s = 9.
The fraction x = p.k/p_1.k differs from a product's share of the lepton's lab energy by exactly p'.(n' + beta z),
with p' the product's momentum and n' the photon's direction in the rest frame. This is at most |p'_par| eps/(gamma k)
along the photon and |p'_perp| (2 eps/(gamma k))^(1/2) across it.

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/triplet.py
"""
import functools
from re import match as re_match, search as re_search

import mpmath as mp
import numpy as np
import sympy as sp

import latex as L
import mojo_constants
from latex import ANGUELOV, LEE, LEE98
from pdg import INV_ALPHA, R_E_CM
from verify import Result, derivation, exact, exact_fails, in_source, mdot, must_fail, run, zero_numerator

k, s, t, P, x, al, re, sT, c, n, eps, gam, m = sp.symbols(
    "k s t P x alpha r_e sigma_T c n epsilon gamma m", positive=True)
Lg = sp.Symbol("L")   # ln 2k
ALPHA = 1 / sp.Float(INV_ALPHA, 30)


def readme_tex(pin):
    return L.display(pin)


# --- Lee, Lyubyakin & Stotsky: the expansions, from the TeX source ---

LEE_45 = r"\frac{\pi}{3 \sqrt{3}}  \left[\frac{4 r^2}{27}+\frac{602 r^3}{243}+\frac{4217 r^4}{4374}+O(r^5)\right]"
LEE_R = r"$r=\frac{s-9}{s-1}$"
LEE_K = r"photon energy $k=(s-1)/2$ in the rest frame of the initial electron"
LEE_411 = (r"\frac{28 \ln{s}}{9}-\frac{218}{27}",
           r"-\frac{1}{s}\left(\frac{8}{3} \ln^3{s}-\frac{49 }{6}\ln ^2{s}-\left(2 \pi ^2-\frac{409}{9}\right)\ln{s}"
           r"+12 \zeta_3+\frac{31 \pi^2}{18}-\frac{1081}{12}\right)")
LEE_412 = (r"\frac{28}{9} \ln (2 k)-\frac{218}{27}",
           r"+\frac{1}{k}\left(-\frac{4}{3}  \ln ^3(2 k)+\frac{49}{12} \ln ^2(2 k)+\left(\pi ^2 -\frac{409}{18}\right)"
           r" \ln (2 k)-6 \zeta_3-\frac{31 \pi ^2}{36}+\frac{3355}{72}\right)")
LEE_413 = (r"\frac{28}{9} \ln (2 k)-\frac{218}{27}-\frac{1}{k}\left(\frac{4}{3}  \ln ^3(2 k)-4.08\ln ^2(2 k)"
           r"+12.85 \ln (2 k)-30.89\right)")
ZETA = {r"\zeta_3": sp.zeta(3)}


def lee(tex, variable):
    return L.to_sympy(tex, {variable: {"s": s, "k": k, "r": sp.Symbol("r")}[variable], **ZETA})


def high_energy_form(expr):
    """{power of 1/k: polynomial in L = ln 2k} of an expansion in k."""
    e = sp.expand(expr.subs(sp.log(2 * k), Lg))
    return {p: sp.expand(e.coeff(k, -p)) for p in (0, 1)}


@derivation("tr.threshold")
def check_threshold():
    """The threshold expansion is Lee et al.'s eq. 4.5 (times alpha^3, see the module docstring)."""
    line = readme_tex(r"\frac{\sigma}{\alpha r_e^2} = \frac{\pi}{3\sqrt3}")
    ours = L.to_sympy(L.sides(L.parts(line)[0])[1], {"r": sp.Symbol("r")})
    r_def = L.to_sympy(L.sides(L.part(line, "r ="))[1], {"s": s})
    s_def = L.to_sympy(L.sides(L.part(line, "s ="))[1], {"k": k})
    theirs = lee(LEE_45.replace("+O(r^5)", ""), "r")
    return [in_source("eq. 4.5, r = (s - 9)/(s - 1) and k = (s - 1)/2 as transcribed", LEE, LEE_45, LEE_R, LEE_K),
            exact("README expansion = eq. 4.5 in r", ours - theirs),
            exact("README r(s) = (s - 9)/(s - 1)", r_def - (s - 9) / (s - 1)),
            exact("README s(k) = 1 + 2k (k = (s - 1)/2)", s_def - (1 + 2 * k))]


@derivation("tr.high")
def check_high():
    """The high-energy expansion is Lee et al.'s eq. 4.13, which rounds eq. 4.12, which is eq. 4.11 in k."""
    line = readme_tex(r"\frac{\sigma}{\alpha r_e^2} = \frac{28}{9}\ln 2k - \frac{218}{27}")
    ours = L.to_sympy(L.sides(L.parts(line)[0])[1], {"k": k})
    e413 = lee(LEE_413, "k")
    e412 = lee(LEE_412[0], "k") + lee(LEE_412[1], "k")
    e411 = lee(LEE_411[0], "s") + lee(LEE_411[1], "s")
    out = [in_source("eqs. 4.11-4.13 as transcribed", LEE, *LEE_411, *LEE_412, LEE_413),
           exact("README = eq. 4.13", ours - e413)]
    exact_form, rounded = high_energy_form(e412), high_energy_form(e413)
    worst = 0
    for p in (0, 1):
        for j in range(4):
            a, b = exact_form[p].coeff(Lg, j), rounded[p].coeff(Lg, j)
            worst = max(worst, abs(float(a - b)))
    out.append(Result("eq. 4.13 rounds eq. 4.12 to two decimals", worst <= 0.005,
                      f"largest difference of a coefficient {worst:.4f}"))
    # eq. 4.11 with s = 1 + 2k, expanded in u = 1/k with L = ln 2k held fixed: ln s = L + ln(1 + u/2)
    u = sp.Symbol("u", positive=True)
    in_k = e411.subs(sp.log(s), Lg + sp.log(1 + u / 2)).subs(s, 1 + 2 / u)
    series = sp.series(in_k, u, 0, 2).removeO()
    target = e412.subs(sp.log(2 * k), Lg).subs(k, 1 / u)
    out.append(exact("eq. 4.11 at s = 1 + 2k = eq. 4.12 to O(1/k)", sp.expand(series - target)))
    return out


LNK = "the cross section grows as $"


@derivation("tr.lnk")
def check_lnk():
    """(28/9) alpha r_e^2 ln 2k is the leading term of eq. 4.12 (alpha^3 = alpha r_e^2 for m_e = 1)."""
    lead = high_energy_form(lee(LEE_412[0], "k") + lee(LEE_412[1], "k"))[0]

    def claims():
        ours = L.to_sympy(L.between(L.line(LNK), LNK, "$"), {r"\alpha": al, "r_e": re, "k": k})
        return [exact("= alpha r_e^2 x (the term of eq. 4.12 growing with k)",
                      ours - al * re**2 * lead.coeff(Lg, 1) * sp.log(2 * k))]
    return claims() + _controls(claims, [(r"grows as $(28/9)\alpha", r"grows as $(14/9)\alpha")], "README")


# eq. 4.2 (the differential system) and eq. 4.3 (the integrals), from the TeX source: y, z, and for
# each k the coefficients of J_m in J_k' and the kernel of J_k = int_9^s ds~ J_1(s~) K_k(s, s~).
y = sp.sqrt((s - 1) / (s + 3))
z = sp.sqrt((s - 5) / (s - 1))
ell5 = sp.log((sp.sqrt(s - 5) + sp.sqrt(s - 1)) / (sp.sqrt(t - 5) + sp.sqrt(t - 1)))
ell3 = sp.log((sp.sqrt(s - 1) + sp.sqrt(s + 3)) / (sp.sqrt(t - 1) + sp.sqrt(t + 3)))
K9 = (7 * t - 15) / (t - 9) / sp.sqrt((t - 5) * (t - 1))
KERNELS = {3: 2 / (t - 9), 4: 2 / (t - 9) * sp.log((s - 1) / (t - 1)), 5: 1 / (t - 1),
           6: sp.Rational(1, 2) / (t - 1) * sp.log(s / t), 7: -3 / (t - 9) * sp.log(s / t),
           8: 2 * (7 * t - 15) / ((t - 9) * (t - 1)) * ell3, 9: K9,
           10: K9 * sp.log((s - 1) / (t - 1)) - 32 * t / ((t - 9) * (t - 1)) * ell5,
           11: K9 * ell5,
           12: K9 * sp.log((s - 5) / (t - 5)) + 2 * (5 * t + 3) / ((t - 9) * (t - 1)) * ell5}
SYSTEM = {3: {1: 2 / (s - 9)}, 4: {3: 1 / (s - 1)}, 5: {1: 1 / (s - 1)}, 6: {5: 1 / (2 * s)},
          7: {3: -sp.Rational(3, 2) / s}, 8: {3: 3 * y / (s - 1), 5: y / (s - 1)},
          9: {1: (7 * s - 15) * z / ((s - 9) * (s - 5))},
          10: {5: 2 * z / (s - 5), 3: -9 * z / (s - 5), 9: 1 / (s - 1)},      # J_15 = 0
          11: {9: z / (s - 5)}, 12: {3: 3 * z / (s - 5), 5: -z / (s - 5), 9: 1 / (s - 5)}}
LEE_SYSTEM = [r"$y=\sqrt{\frac{s-1}{s+3}}$ and $z=\sqrt{\frac{s-5}{s-1}}$",
              r"J_3'=\frac{2J_1}{s-9}", r"J_4'=\frac{J_3}{s-1}", r"J_5'=\frac{J_1}{s-1}", r"J_6'=\frac{J_5}{2 s}",
              r"J_7'=-\frac{3 J_3}{2s}", r"J_8'=\frac{ y }{s-1}(3J_3+J_5)",
              r"J_9'=\frac{(7 s-15) z J_1}{(s-9) (s-5)}",
              r"J_{10}'=\frac{z}{s-5}\left(2J_5-9J_3+2J_{15}\right)+\frac{J_9}{s-1}",
              r"J_{11}'=\frac{z J_9}{s-5}", r"J_{12}'=\frac{3 z J_3-zJ_5+J_9-zJ_{15}}{s-5}",
              r"J_{13}=J_{14}=J_{15}=J_{16}=J_{17}=J_{18}=0"]
LEE_INTEGRALS = [
    r"J_3(s)=2\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-9}",
    r"J_4(s)=2\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-9}\ln\left(\frac{s-1}{\tilde{s}-1}\right)",
    r"J_5(s)=\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-1}",
    r"J_6(s)=\frac12\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-1}\ln \left(\frac{s}{\tilde{s}}\right)",
    r"J_7(s)=-3\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-9}\ln \left(\frac{s}{\tilde{s}}\right)",
    r"J_8(s)=2\intop_9^s d\tilde{s} \frac{(7\tilde{s}-15)J_1(\tilde{s})}{\left(\tilde{s}-9\right) \left(\tilde{s}-1\right)}"
    r"\ln\left(\frac{\sqrt{s-1}+\sqrt{s+3}}{\sqrt{\tilde{s}-1}+\sqrt{\tilde{s}+3}}\right)",
    r"J_9(s)=\intop_9^s d\tilde{s} \frac{\frac{7\tilde{s}-15}{\tilde{s}-9}J_1(\tilde{s})}{\sqrt{(\tilde{s}-5)(\tilde{s}-1)}}",
    r"J_{10}(s)=\intop_9^s d\tilde{s} \frac{\frac{7 \tilde{s}-15}{\tilde{s}-9}J_1(\tilde{s})}{\sqrt{(\tilde{s}-5)(\tilde{s}-1)}}"
    r" \ln\left(\frac{s-1}{\tilde{s}-1}\right) -32\intop_9^s d\tilde{s} \frac{\tilde{s}J_1(\tilde{s})}{(\tilde{s}-9)(\tilde{s}-1)}"
    r" \ln \left(\frac{\sqrt{s-5}+\sqrt{s-1}}{\sqrt{\tilde{s}-5}+\sqrt{\tilde{s}-1}}\right)",
    r"J_{11}(s)=\intop_9^s d\tilde{s} \frac{J_1(\tilde{s})}{\sqrt{(\tilde{s}-5)(\tilde{s}-1)}} \frac{7 \tilde{s}-15}{\tilde{s}-9}"
    r" \ln \left(\frac{\sqrt{s-5}+\sqrt{s-1}}{\sqrt{\tilde{s}-5}+\sqrt{\tilde{s}-1}}\right)",
    r"J_{12}(s)=\intop_9^s d\tilde{s} \frac{\frac{7 \tilde{s}-15}{\tilde{s}-9}J_1(\tilde{s})}{\sqrt{(\tilde{s}-5)(\tilde{s}-1)}}"
    r" \ln \left(\frac{s-5}{\tilde{s}-5}\right)",
    r"+2\intop_9^s d\tilde{s} \frac{(5 \tilde{s}+3)J_1(\tilde{s})}{\left(\tilde{s}-9\right) \left(\tilde{s}-1\right)}"
    r" \ln \left(\frac{\sqrt{s-5}+\sqrt{s-1}}{\sqrt{\tilde{s}-5}+\sqrt{\tilde{s}-1}}\right)"]


def _rational_s(expr, roots):
    """Substitutes for s a parametrisation in which the two square roots are rational: s - 5, s - 1
    (roots 5) or s - 1, s + 3 (roots 3) are ((q -+ 4/q)/2)^2 with q = 2 + w > 2."""
    q = 2 + sp.Symbol("w", positive=True)
    sv = ((q + 4 / q) / 2)**2 + 1 if roots == 5 else ((q - 4 / q) / 2)**2 + 1
    return expr.subs(s, sv)


def kernel_identity(kk, kernels):
    """dK_k/ds minus the right-hand side of eq. 4.2, and K_k(s, s) minus the coefficient of J_1. If both are zero,
    J_k of eq. 4.3 solves eq. 4.2 for every J_1 (differentiating under the integral)."""
    rhs = sum(cf * kernels[m_] for m_, cf in SYSTEM[kk].items() if m_ != 1)
    roots = 3 if kk == 8 else 5
    return (zero_numerator(_rational_s(sp.diff(kernels[kk], s) - rhs, roots)),
            zero_numerator(_rational_s((kernels[kk] - SYSTEM[kk].get(1, 0)).subs(t, s), roots)))


@derivation("tr.lee_J11")
def check_lee_J11():
    """Lee et al.'s eq. 4.3 solves their eq. 4.2 except J_11, which needs a factor 2."""
    out = [in_source("eqs. 4.2 and 4.3 as transcribed", LEE, *LEE_SYSTEM, *LEE_INTEGRALS)]
    failing = []
    for kk in KERNELS:
        d, b = kernel_identity(kk, KERNELS)
        if d != 0 or b != 0:
            failing.append(kk)
    out.append(Result("J_3 ... J_12 of eq. 4.3 solve eq. 4.2 except J_11", failing == [11],
                      f"exact: kernels that fail: J_{failing}"))
    fixed = {**KERNELS, 11: 2 * KERNELS[11]}
    d, b = kernel_identity(11, fixed)
    out.append(Result("J_11 with the factor 2 solves eq. 4.2", d == 0 and b == 0, "exact"))
    ratio = sp.simplify(_rational_s(sp.diff(KERNELS[11], s) / (SYSTEM[11][9] * KERNELS[9]), 5))
    out.append(Result("J_11 as printed: dJ_11/ds is half of eq. 4.2's", ratio == sp.Rational(1, 2), f"ratio {ratio}"))
    return out


# --- the exact cross section (eqs. 4.3-4.4, evaluated) against the expansions, Haug's Table 1 and Klein-Nishina ---

LEE_44 = [r"\sigma_{\egeee}=\frac{\alpha ^3}{s-1}",
          r"\frac{4\left(168 s^4-2714 s^3+8697 s^2+1128 s+81\right) J_1}{27(s-1)^2 s}",
          r"-\frac{2 \left(872 s^4-11289 s^3+29949 s^2+729s+27\right) J_2}{27 (s-1)^2 s}",
          r"-\frac{32 \left(s^2-5 s-2\right)J_6}{(s-1)^2}",
          r"+\frac{4 \left(24 s^6-38s^5-228 s^4+507 s^3+1161 s^2-45 s+27\right) J_3}{3 (s-1)^2s^2 (s+3)}",
          r"-\frac{16 \left(9 s^2-46 s+17\right)J_4}{(s-1)^2}",
          r"-\frac{8 \left(3 s^6-3 s^5-59 s^4-94 s^3-159 s^2+33s-9\right) J_5}{3 (s-1)^2 s^2 (s+3)}",
          r"+\frac{16 \left(2 s^2-8 s-1\right)J_7}{(s-1)^2}",
          r"+\frac{32 \left(s^4-s^3-21 s^2-27s+16\right) y J_8}{(s-1)^3 (s+3)}",
          r"-\frac{8 \left(3 s^4-30s^3+116 s^2-146 s-7\right) z J_9}{3 (s-5) (s-1)^2}",
          r"-\frac{8 (s+1) z J_{10}}{s-1}",
          r"+\frac{8 \left(3 s^2-6 s-5\right)J_{11}}{(s-1)^2}",
          r"-\frac{16 \left(s^3-3 s^2-9 s+19\right) zJ_{12}}{(s-5) (s-1)^2}"]
LEE_J1 = (r"J_1(s)=\frac{s-9}{6 \left(\sqrt{s}-1\right)^{3/2} \left(\sqrt{s}+3\right)^{1/2}}\mathrm{K}\left(\frac{\left(\sqrt{s}-3\right) "
          r"\left(\sqrt{s}+1\right)^3}{\left(\sqrt{s}-1\right)^3 \left(\sqrt{s}+3\right)}\right)")
LEE_J2 = (r"J_2(s)=\frac{1}{8s} \left(\sqrt{s}-1\right)^{3/2} \left(\sqrt{s}+3\right)^{1/2}\, \mathrm{E}\left(\frac{\left(\sqrt{s}-3\right) "
          r"\left(\sqrt{s}+1\right)^3}{\left(\sqrt{s}-1\right)^3 \left(\sqrt{s}+3\right)}\right)", r"-\frac{2J_1(s)}{ \sqrt{s}(\sqrt{s}-3)}")
LEE_J1_HIGH = r"\frac{\ln{s}}{8}-\frac{3\ln{s}+2}{4 s}-\frac{6 \ln{s}-5}{4 s^2}+O\left(s^{-3}\right)"
DPS, NODES = 30, 20      # digits; Gauss-Legendre nodes per panel of the integrals J_3 ... J_12
_LEE, _J = {}, {}        # eq. 4.4 parsed; J_3 ... J_12 (J_11 as printed) by s, computed once


def _lee_setup():
    """Eq. 4.4 with J_1, J_2, y and z parsed from the TeX, as a function of s and J_3 ... J_12. The kernels of eq. 4.3
    (KERNELS) become sums of a(s) b(t), with logarithms of ratios split, so that one pass over s~ gives every J_k."""
    if not _LEE:
        js = {kk: sp.Symbol(f"J{kk}") for kk in range(1, 13)}
        tokens = {"s": s, r"\alpha": 1} | {(f"J_{kk}" if kk < 10 else f"J_{{{kk}}}"): js[kk] for kk in js}
        tokens |= {L.sides(p)[0]: L.to_sympy(L.sides(p)[1], {"s": s}) for p in LEE_SYSTEM[0].split("$")[1::2]}   # y, z
        j1 = L.to_sympy(LEE_J1.split("=", 1)[1], {"s": s}, funcs={r"\mathrm{K}": sp.elliptic_k})
        j2 = L.to_sympy(LEE_J2[0].split("=", 1)[1] + LEE_J2[1], {"s": s},
                        funcs={r"\mathrm{E}": sp.elliptic_e, "J_1": lambda a: j1.subs(s, a)})
        sigma = L.to_sympy(LEE_44[0].split("=")[1], tokens) * L.to_sympy("".join(LEE_44[1:]), tokens)
        terms, basis = {}, []
        for kk, kernel in KERNELS.items():
            terms[kk] = []
            for term in sp.Add.make_args(sp.expand(sp.expand_log(kernel, force=True))):
                a, b = term.as_independent(t, as_Add=False)
                basis += [b] if b not in basis else []
                terms[kk].append((sp.lambdify(s, a, "mpmath"), basis.index(b)))
        _LEE.update(j1_sympy=j1, j1=sp.lambdify(s, j1, "mpmath"), basis=sp.lambdify(t, basis, "mpmath"), n=len(basis),
                    terms=terms, sigma=sp.lambdify([s] + [js[kk] for kk in range(3, 13)], sigma.subs({js[1]: j1, js[2]: j2}), "mpmath"))
    return _LEE


def _integrals(svals, setup):
    """{s: [J_3 ... J_12]} (J_11 as printed) at the s of svals, in one pass over s~ from 9: Gauss-Legendre panels of a
    quarter on 9 <= s~ <= 10 and of at most 0.3 in ln s~ above, split at every s asked for."""
    X, W = mp.gauss_quadrature(NODES, "legendre")
    top = max(svals)
    edges = {9 + mp.mpf(i) / 4 for i in range(5)} | set(svals)
    if top > 10:
        n_ = int(mp.ceil(mp.log(top / 10) / mp.mpf("0.3")))
        edges |= {10 * (top / 10) ** (mp.mpf(i) / n_) for i in range(n_ + 1)}
    edges = sorted(e for e in edges if e <= top)
    total, out = [mp.mpf(0)] * setup["n"], {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        h, mid = (hi - lo) / 2, (hi + lo) / 2
        for xv, wv in zip(X, W):
            weight = wv * h * setup["j1"](mid + h * xv)
            for i, b in enumerate(setup["basis"](mid + h * xv)):
                total[i] += weight * b
        if hi in svals:
            out[hi] = [sum(a(hi) * total[i] for a, i in setup["terms"][kk]) for kk in range(3, 13)]
    return out


def lee_sigma(ks, j11=2):
    """sigma/(alpha r_e^2) of eq. 4.4 (alpha^3 with m_e = 1) at the photon energies ks (s = 1 + 2k), with J_11 of eq. 4.3
    times j11: 2 as eq. 4.2 requires (tr.lee_J11), 1 as printed. Call within mp.workdps(DPS), with all the k needed in
    one call (one pass over s~); triplet/compare_exact.py uses it too."""
    setup = _lee_setup()
    svals = [1 + 2 * mp.mpf(kv) for kv in ks]
    new = {sv for sv in svals if sv not in _J}
    if new:
        _J.update(_integrals(new, setup))
    return [setup["sigma"](sv, *[jv * (j11 if kk == 11 else 1) for kk, jv in zip(range(3, 13), _J[sv])]) for sv in svals]


def eq412():
    """Lee et al.'s eq. 4.12 (exact coefficients), sigma/(alpha r_e^2) as an mpmath function of k (machinery.py uses it too)."""
    return sp.lambdify(k, lee(LEE_412[0], "k") + lee(LEE_412[1], "k"), "mpmath")


def readme_expansion_forms():
    """The README's threshold (eq. 4.5) and high-energy (eq. 4.13) expansions as sympy expressions in k, and the ends
    of the ranges where the code uses them (k <= 4.02, k >= 10^3), read from the display equations."""
    low_tex = readme_tex(r"\frac{\sigma}{\alpha r_e^2} = \frac{\pi}{3\sqrt3}")
    high_tex = readme_tex(r"\frac{\sigma}{\alpha r_e^2} = \frac{28}{9}\ln 2k - \frac{218}{27}")
    low = L.to_sympy(L.sides(L.parts(low_tex)[0])[1], {"r": sp.Symbol("r")}).subs(
        sp.Symbol("r"), L.to_sympy(L.sides(L.part(low_tex, "r ="))[1], {"s": s})).subs(
        s, L.to_sympy(L.sides(L.part(low_tex, "s ="))[1], {"k": k}))
    high = L.to_sympy(L.sides(L.parts(high_tex)[0])[1], {"k": k})
    ends = [L.to_sympy(L.part(tex, "(k").strip("()").split(rel)[1], {})
            for tex, rel in ((low_tex, r"\le"), (high_tex, r"\ge"))]
    return low, high, ends


def readme_expansions():
    """readme_expansion_forms() as mpmath functions of k, and the ends as mpf."""
    low, high, ends = readme_expansion_forms()
    return sp.lambdify(k, low, "mpmath"), sp.lambdify(k, high, "mpmath"), [mp.mpf(str(e)) for e in ends]


@functools.cache
def numeric_expansions():
    """readme_expansion_forms() for numpy: sigma/(alpha r_e^2) of each expansion with its slope
    d ln sigma / d ln(k - 4) = (k - 4) sigma'/sigma (the ends of the code's spline), and the ends as floats."""
    out = []
    for form in readme_expansion_forms()[:2]:
        out += [sp.lambdify(k, form, "numpy"), sp.lambdify(k, (k - 4) * sp.diff(form, k) / form, "numpy")]
    return (*out, [float(e) for e in readme_expansion_forms()[2]])


def quoted(tex):
    """A number as the README writes it (3 \\times 10^{-10}, 10^{-3}, -28, 272), and its significant digits."""
    tex = tex.strip()
    digits = 1 if tex.startswith("10^") else len(re_match(r"-?(\d+(?:\.\d+)?)", tex).group(1).replace(".", "").lstrip("0"))
    return mp.mpf(str(float(L.to_sympy(tex, {})))), digits


def rounds_to(x, q):
    """x, rounded to the significant digits of the quoted number q, is that number."""
    return f"{float(x):.{q[1] - 1}e}" == f"{float(q[0]):.{q[1] - 1}e}"


def shown(tex):
    """The quoted number in plain text (3e-10)."""
    value, digits = quoted(tex)
    return f"{float(value):.{digits}g}"


def _controls(claims, patches, file="checks"):
    out = []
    for old, new in patches:
        with L.patched(old, new, file):
            out.append(must_fail(f"control: README with {new}", claims()))
    return out


LEE_EXACT = "Without the factor, eq. 4.4 is $"


def lee_exact_claims():
    """The README's figures of eq. 4.4 against the expansions, with and without J_11's factor 2."""
    m = re_search(r"Without the factor, eq\. 4\.4 is \$([^$]+)\$ times the threshold expansion at \$k = ([^$]+)\$\. With it, "
                  r"eq\. 4\.4 agrees with the threshold expansion to \$([^$]+)\$ there and with the high-energy expansion to "
                  r"\$O\(k\^\{([^}]+)\}\)\$", L.line(LEE_EXACT, "checks"))
    low = readme_expansions()[0]
    k0 = mp.mpf(m[2])
    printed, fixed = lee_sigma([k0], 1)[0] / low(k0), lee_sigma([k0])[0] / low(k0) - 1
    out = [Result(f"J_11 as printed: eq. 4.4 is {shown(m[1])} times the README's threshold expansion at k = {m[2]}",
                  rounds_to(printed, quoted(m[1])), f"{mp.nstr(printed, 4)}"),
           Result(f"with the factor 2: within {shown(m[3])} of it (so the README's sigma/(alpha r_e^2) carries eq. 4.4's alpha^3)",
                  rounds_to(abs(fixed), quoted(m[3])), f"eq. 4.4/expansion - 1 = {mp.nstr(fixed, 3)}")]
    e412 = eq412()
    power = -quoted(m[4])[0]
    ks_ = [mp.mpf(10) ** j for j in (4, 6, 8)]
    c_ = [kv ** power * abs(sv - e412(kv)) for kv, sv in zip(ks_, lee_sigma(ks_))]
    n_ = [mp.log(c_[i + 1] / c_[i]) / mp.log(mp.log(2 * ks_[i + 1]) / mp.log(2 * ks_[i])) for i in (0, 1)]
    out.append(Result(f"sigma - eq. 4.12 = O(k^{shown(m[4])}) up to logarithms: k^{float(power):g} |sigma - eq. 4.12| grows "
                      "as ln^n 2k with 0 <= n <= 4", all(0 <= v <= 4 for v in n_),
                      f"n = {mp.nstr(n_[0], 3)}, {mp.nstr(n_[1], 3)} (k = 1e4 to 1e6, 1e6 to 1e8)"))
    return out


@derivation("tr.lee_exact")
def check_lee_exact():
    """Eq. 4.4 (TeX) with J_11's factor 2 is within 3e-10 of eq. 4.5 at k = 4.001 and meets eq. 4.12 to O(k^-2); without it, -28 times eq. 4.5."""
    out = [in_source("eq. 4.4, J_1, J_2 and J_1's high-energy asymptotics as transcribed", LEE, *LEE_44, LEE_J1, *LEE_J2,
                     LEE_J1_HIGH)]
    with mp.workdps(DPS):
        setup = _lee_setup()
        high = sp.lambdify(s, lee(LEE_J1_HIGH.replace(r"+O\left(s^{-3}\right)", ""), "s"), "mpmath")
        modulus = sp.lambdify(s, setup["j1_sympy"].replace(sp.elliptic_k, lambda a: sp.elliptic_k(a**2)), "mpmath")
        pts = [mp.mpf(10) ** 6, mp.mpf(10) ** 8]
        worst = max(abs(setup["j1"](sv) / high(sv) - 1) for sv in pts)
        out += [Result("J_1, K(m) with the parameter m (mpmath's ellipk), has their asymptotics (s = 1e6, 1e8)", worst < 1e-15,
                       f"rel. difference {mp.nstr(worst, 2)}"),
                Result("control: K with the modulus, K(m^2), does not", max(abs(modulus(sv) / high(sv) - 1) for sv in pts) > 1e-3,
                       "differs, as it must")]
        out += lee_exact_claims()
        out += _controls(lee_exact_claims, [(r"eq. 4.4 is $-28$ times", r"eq. 4.4 is $-26$ times"),
                                            (r"to $3 \times 10^{-10}$ there", r"to $3 \times 10^{-11}$ there"),
                                            (r"expansion to $O(k^{-2})$", r"expansion to $O(k^{-1})$")])
    return out


EXPANSIONS = "The threshold expansion is $"


def expansion_claims():
    """The README's figures of its expansions against eq. 4.4: at two energies, and the largest difference over the ranges
    where the code uses them."""
    line = L.line(EXPANSIONS, "checks")
    m = re_search(r"The threshold expansion is \$([^$]+)\$ below the exact cross section at \$k = ([^$]+)\$ and \$([^$]+)\$ "
                  r"at ([\d.]+)\.", line)
    code = re_search(r"agrees to \$([^$]+)\$ in the threshold expansion, .*? and \$([^$]+)\$ in the high-energy expansion", line)
    low, high, (k_low, k_high) = readme_expansions()
    out = []
    for tol, kv in ((m[1], m[2]), (m[3], m[4])):
        dev = low(mp.mpf(kv)) / lee_sigma([kv])[0] - 1
        out.append(Result(f"eq. 4.5 is {shown(tol)} below eq. 4.4 at k = {kv}", dev < 0 and rounds_to(-dev, quoted(tol)), mp.nstr(dev, 3)))
    for name, f, pts, tol in (("eq. 4.5", low, [k_low] + [4 + (k_low - 4) / d for d in (2, 20, 200)], code[1]),
                              ("eq. 4.13", high, [k_high * 10**j for j in (0, 1, 3, 5, 7)], code[2])):
        devs = [abs(f(kv) / sv - 1) for kv, sv in zip(pts, lee_sigma(pts))]
        out.append(Result(f"{name} within {shown(tol)} of eq. 4.4 where the code uses it (from k = {float(pts[0]):g}), largest there",
                          rounds_to(max(devs), quoted(tol)) and max(devs) == devs[0],
                          ", ".join(f"{mp.nstr(d, 3)} at {float(kv):g}" for kv, d in zip(pts, devs))))
    return out


@derivation("tr.expansions")
def check_expansions():
    """The README's expansions against eq. 4.4: 1e-3 and 8e-3 below at k = 4.2 and 4.5; within 2e-6 (k <= 4.02) and 2.3e-5 (k >= 1e3)."""
    with mp.workdps(DPS):
        return (expansion_claims()
                + _controls(expansion_claims, [(r"and $8 \times 10^{-3}$ at 4.5", r"and $8 \times 10^{-3}$ at 4.6"),
                                               (r"$2.3 \times 10^{-5}$ in the high", r"$2.1 \times 10^{-5}$ in the high")])
                + _controls(expansion_claims, [(r"\qquad (k \ge 10^3)", r"\qquad (k \ge 10^2)")], "README"))


# Haug (1975, Z. Naturforsch. 30a, 1099), Table 1: the photon energy k (m_e c^2, target electron at rest) and the total
# cross section, in the table's units (10^-10 b for k <= 4.05, mub for 4.1 <= k <= 7.8, mb above)
HAUG = [(1e-7, [(4.001, 0.03255), (4.002, 0.1307), (4.003, 0.2951), (4.004, 0.5265), (4.005, 0.8256), (4.01, 3.361),
                (4.02, 13.91), (4.03, 32.34), (4.04, 59.3), (4.05, 95.5)]),
        (1e-3, [(4.1, 0.0436), (4.2, 0.2133), (4.4, 1.110), (4.6, 2.940), (5.0, 9.85), (5.4, 21.22), (5.8, 36.80),
                (6.0, 46.0), (6.2, 56.1), (6.6, 78.7), (7.0, 104.0), (7.4, 131.5), (7.8, 161.0)]),
        (1.0, [(8, 0.1764), (9, 0.2576), (10, 0.3439), (11, 0.433), (12, 0.522), (13, 0.611), (14, 0.700), (16, 0.871),
               (18, 1.035), (20, 1.191), (25, 1.545), (30, 1.856), (35, 2.131), (40, 2.376), (45, 2.598), (50, 2.800),
               (60, 3.154), (70, 3.459), (80, 3.726), (90, 3.963), (100, 4.176), (200, 5.58), (300, 6.40), (400, 6.98),
               (500, 7.42), (600, 7.78), (700, 8.09), (800, 8.35), (900, 8.58), (1000, 8.78), (1500, 9.56), (2000, 10.11),
               (2500, 10.53), (3000, 10.88), (5000, 11.83)])]
R_E = mp.mpf(R_E_CM)
HAUG_PIN = "Haug's values are $"


def haug():
    """Haug's k and sigma/(alpha r_e^2) (a millibarn is 1e-27 cm^2)."""
    unit = mp.mpf(1) / mp.mpf(INV_ALPHA) * R_E**2 / mp.mpf("1e-27")
    return [(mp.mpf(repr(kv)), mp.mpf(repr(sv)) * mp.mpf(repr(to_mb)) / unit) for to_mb, rows in HAUG for kv, sv in rows]


def haug_claims():
    """The README's figures of Haug's Table 1 against eq. 4.4 and against the expansions."""
    table = re_search(r"Table 1: (\d+) values for \$4 \\lt k \\le ([^$]+)\$.*within \$([^$]+)\$ for \$k \\le ([^$]+)\$ and "
                      r"\$k \\ge ([^$]+)\$\)", L.line("The expansions the code uses also agree to Haug", "checks"))
    m = re_search(r"Haug's values are \$([^$]+)\$ low on average \((\d+) of the (\d+) lie below the exact cross section, by up to "
                  r"\$([^$]+)\$\)", L.line(HAUG_PIN, "checks"))
    rows = haug()
    ks_ = [kv for kv, _ in rows]
    dev = [sv / ex - 1 for (_, sv), ex in zip(rows, lee_sigma(ks_))]
    low, high, _ = readme_expansions()
    k_low, k_high = (mp.mpf(str(L.to_sympy(v, {}))) for v in (table[4], table[5]))
    exp_dev = [abs(f(kv) / sv - 1) for kv, sv in rows for f, used in ((low, kv <= k_low), (high, kv >= k_high)) if used]
    below = [-d for d in dev if d < 0]
    mean = sum(dev) / len(dev)
    return [Result(f"the transcription: {table[1]} values for 4 < k <= {table[2]}", len(rows) == int(table[1]) == int(m[3])
                   and min(ks_) > 4 and max(ks_) == quoted(table[2])[0], f"{len(rows)} values, k = {float(min(ks_)):g} to {float(max(ks_)):g}"),
            Result(f"Haug's values are {shown(m[1])} low on average", mean < 0 and rounds_to(-mean, quoted(m[1])), mp.nstr(mean, 3)),
            Result(f"{m[2]} of the {m[3]} lie below eq. 4.4, by up to {shown(m[4])}", len(below) == int(m[2]) and
                   rounds_to(max(below), quoted(m[4])), f"{len(below)}, by up to {mp.nstr(max(below), 3)}"),
            Result(f"the README's expansions within {shown(table[3])} of Haug's values for k <= {shown(table[4])} and k >= {shown(table[5])} "
                   f"({len(exp_dev)} values)", rounds_to(max(exp_dev), quoted(table[3])), f"up to {mp.nstr(max(exp_dev), 3)}")]


@derivation("tr.haug")
def check_haug():
    """Haug's Table 1 against eq. 4.4: 2.4e-4 low on average, 46 of 58 below by up to 8e-4; the expansions within 6e-4 of it."""
    with mp.workdps(DPS):
        return haug_claims() + _controls(haug_claims, [("(46 of the 58 lie below", "(45 of the 58 lie below"),
                                                       (r"$2.4 \times 10^{-4}$ low", r"$2.6 \times 10^{-4}$ low"),
                                                       (r"within $6 \times 10^{-4}$ for", r"within $4 \times 10^{-4}$ for")])


def kn_dsigma(kv, mu_):
    """The Klein-Nishina cross section per solid angle, in r_e^2, at the photon energy kv (m_e c^2) and cos theta = mu_."""
    p_ = 1 / (1 + kv * (1 - mu_))
    return p_**2 * (p_ + 1 / p_ - (1 - mu_**2)) / 2


def kn_sigma(kv):
    """The total Klein-Nishina cross section, in r_e^2: 2 pi int kn_dsigma dmu (mpmath; triplet/compare_exact.py uses it)."""
    return 2 * mp.pi * mp.quad(lambda u: kn_dsigma(kv, u), [-1, 1 - 1 / kv, 1])


def kn_cross_claims():
    """Where the README says eq. 4.4 exceeds the Klein-Nishina cross section: the sign of alpha sigma - sigma_KN (in r_e^2)
    half a unit of the last digit below and above."""
    kc = L.between(L.line("exceeds the Klein–Nishina cross section above $k = ", "checks"),
                   "exceeds the Klein–Nishina cross section above $k = ", "$")
    value, digits = quoted(kc)
    half = mp.mpf(10) ** (mp.floor(mp.log10(abs(value))) - digits + 1) / 2
    pts = [value - half, value + half]
    diff = [sv / mp.mpf(INV_ALPHA) - kn_sigma(kv) for kv, sv in zip(pts, lee_sigma(pts))]
    return [Result(f"eq. 4.4 exceeds the Klein-Nishina cross section above k = {kc}", diff[0] < 0 < diff[1],
                   f"alpha sigma - sigma_KN = {mp.nstr(diff[0], 3)}, {mp.nstr(diff[1], 3)} r_e^2 at k = "
                   f"{mp.nstr(pts[0], 5)}, {mp.nstr(pts[1], 5)}")]


@derivation("tr.kn_cross")
def check_kn_cross():
    """Eq. 4.4 exceeds the Klein-Nishina cross section above k = 272."""
    mu_ = sp.Symbol("mu", real=True)
    out = [exact("Klein-Nishina at k = 0: 2 pi int dmu dsigma/dOmega = sigma_T/r_e^2 = 8 pi/3",
                 sp.integrate(2 * sp.pi * kn_dsigma(0, mu_), (mu_, -1, 1)) - 8 * sp.pi / 3)]
    with mp.workdps(DPS):
        return out + kn_cross_claims() + _controls(kn_cross_claims, [("above $k = 272$", "above $k = 274$")])


# --- the rates and the kinetic terms (README, Rates and Kinetic terms) against a direct count of the collisions ---

mu = sp.Symbol("mu", real=True)                                  # cos theta
y, gp = sp.symbols("y gamma'", positive=True)                    # y = xk; gamma' the target lepton
SIG = sp.Function("sigma")                                       # the cross section: any function, 0 below k = 4
F = {r: sp.Function(f"f_{r}") for r in ("r", "+", "low")}        # the distributions f_r(k, y) per ln y: any functions
X = {r: sp.Function(f"X_{r}") for r in ("+", "low")}             # their mean fractions <x_r> at fixed k
DENS = {"-": sp.Function("n_-"), "+": sp.Function("n_+")}         # e-, e+ per ln gamma
NG = sp.Function("n_gamma")                                      # photons per ln eps
CONJUGATE = {"-": "+", "+": "-"}
PRODUCTS = {"-": {"+": "+", "-": "low"}}   # an e- target makes a positron (x_+) and an electron of lower energy (x_low)
PRODUCTS["+"] = {CONJUGATE[q]: r for q, r in PRODUCTS["-"].items()}   # an e+ target makes their charge conjugates
PAIRS = r"\dot n_{+}(\gamma) = c\thinspace \sigma_T\int d\ln\gamma'"
LOSS = r"-\frac{\dot\gamma}{\gamma} = c\thinspace \sigma_T\int d\ln\varepsilon"
SINK = r"\dot n_\gamma(\varepsilon) = -\thinspace n_\gamma(\varepsilon)\thinspace c\thinspace \sigma_T"
ELECTRONS = r"likewise for the electrons, with $n_{-}$ and $n_+$ exchanged ($n_- G_\mathrm{low} + n_+ G_+$)"
MEAN = r"with $\langle\ldots\rangle$ the mean over $k$ with the weight $"
# test functions for the counts: sigma with the threshold; f_+ and f_low normalised to one in ln y on 0 < y <= k (x <= 1),
# the shape of f_+ changing with k; lines of e- (gamma_-), e+ (gamma_+) and photons (eps_1)
TEST_SIGMA = lambda kv: sT * (kv - 4)**2                         # noqa: E731
TEST_F = {"+": lambda kv, yv: (1 - 1 / kv) * yv / kv + 2 / kv * (yv / kv)**2, "low": lambda kv, yv: 2 * (yv / kv)**2}
gl_m, gl_p, e1, N1, N2, Ng = sp.symbols("gamma_- gamma_+ epsilon_1 N_- N_+ N_gamma", positive=True)
LINES = {"-": (gl_m, N1), "+": (gl_p, N2)}


def readme_rate_function(name, P_, x_=None, role="r"):
    """The README's R(P) (name "R(P)") or G_r(P, x) (with the f of role) at P_, x_: its factor times an integral over k."""
    rhs = L.sides(L.part(readme_tex(r"R(P) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k"), name))[1]
    pre, rest = rhs.split(r"\int_", 1)
    lo, rest = rest.split("^{", 1)
    hi, body = rest.split(r"} dk\thinspace ", 1)
    tokens = {"k": k, r"\sigma_T": sT} | ({"x": x_} if x_ is not None else {})
    integrand = L.to_sympy(body, tokens, funcs={r"\sigma": SIG, "f_r": F[role]})
    return L.to_sympy(pre, {"P": P_}) * sp.Integral(integrand, (k, L.to_sympy(lo, {}), L.to_sympy(hi, {"P": P_})))


def readme_k(gl, e):
    """The README's k for a lepton gl and a photon e at the angle theta (cos theta = mu)."""
    return L.to_sympy(L.sides(L.inline(r"k = \gamma\varepsilon(1 - \cos\theta)"))[1],
                      {r"\gamma": gl, r"\varepsilon": e, r"\theta": sp.acos(mu)})


def readme_y(xv, kv):
    """y = xk of the README."""
    return L.to_sympy(L.sides(L.inline(r"y = xk = p\cdot k"))[1], {"x": xv, "k": kv})


def collisions(weight, gl, e):
    """c (1/2) int dmu (1 - mu) sigma(k) weight(k), per lepton gl and photon e, as an integral over k: from 0 to 2 gl e,
    then from the README's threshold, below which sigma = 0."""
    kv = readme_k(gl, e)
    avg = sp.Integral((1 - mu) / 2 * SIG(kv) * weight(kv), (mu, -1, 1)).transform(mu, (sp.solve(kv - k, mu)[0], k))
    (var, lo, hi), = avg.limits
    assert var == k and lo == 0 and hi == 2 * gl * e, avg.limits
    return c * sp.Integral(avg.function, (k, L.to_sympy(L.inline("4m_e c^2"), {"m_e": 1, "c": 1}), hi))


def pulled(expr):
    """The constant factors of every integral over k taken out of it."""
    def split(a):
        cf, rest = a.function.as_independent(k, as_Add=False)
        return cf * sp.Integral(rest, *a.limits)
    return expr.replace(lambda a: isinstance(a, sp.Integral), split)


def integrands(difference):
    """A difference of two sums of integrals over k as the difference of their integrands (with their factors), or None
    if the integrals do not all have one range of k or the difference is not linear in them."""
    d = sp.expand(pulled(difference))
    if d == 0:
        return d

    def linear(term):   # one integral as a factor, none elsewhere
        fs = sp.Mul.make_args(term)
        return sum(isinstance(f, sp.Integral) for f in fs) == 1 and all(isinstance(f, sp.Integral) or not f.has(sp.Integral)
                                                                       for f in fs)
    ints = d.atoms(sp.Integral)
    if not all(linear(term) for term in sp.Add.make_args(d)) or len({i.limits for i in ints}) > 1:
        return None
    return d.xreplace({i: i.function for i in ints})


def same_integrands(what, a, b, must_differ=False):
    d = integrands(a - b)
    if d is None:
        return Result(what, must_differ, "not the same integrals over k" + (", as it must be" if must_differ else ""))
    return (exact_fails if must_differ else exact)(what, d)


def readme_kinetic(pin, lhs, measure):
    """(factor, integrand) of a kinetic term lhs = factor int measure integrand, as LaTeX; lhs and the measure as written."""
    left, rhs = L.sides(L.parts(readme_tex(pin))[0])
    assert left == lhs, left
    head, body = rhs.split(measure)
    return head, body


def readme_pairs():
    """The README's positron and electron sources per ln gamma' and ln eps: (factor, positrons, electrons)."""
    G = {r: (lambda a, b, r=r: readme_rate_function("G_r(P, x)", a, b, r)) for r in ("+", "low")}
    tokens = {r"\gamma'": gp, r"\varepsilon": eps, r"\gamma": gam, "c": c, r"\sigma_T": sT}
    funcs = {r"n_\gamma": NG, "n_-": DENS["-"], "n_+": DENS["+"], "G_{+}": G["+"], r"G_{\rm low}": G["low"]}
    head, body = readme_kinetic(PAIRS, r"\dot n_{+}(\gamma)", r"\int d\ln\gamma'\int d\ln\varepsilon\thinspace ")
    # the electrons: the bracket of the text, with the arguments of dot n_+'s G_+
    args = [L.to_sympy(a, tokens) for a in L.split_top(L.between(body, r"G_{+}\negthinspace \left(", r"\right)"), ",")]
    bracket = L.between(L.line(ELECTRONS), "exchanged ($", "$)")
    ele = (L.to_sympy(L.between(body, "", r"\left["), tokens, funcs=funcs)
           * L.to_sympy(bracket, {"n_-": DENS["-"](gp), "n_+": DENS["+"](gp), r"G_\mathrm{low}": G["low"](*args),
                                  "G_+": G["+"](*args)}))
    return L.to_sympy(head, tokens), L.to_sympy(body, tokens, funcs=funcs), ele


def readme_mean(q, P_):
    """<q>(P_) of the README: the mean over k with its weight, over R's range of k."""
    weight = L.to_sympy(L.between(L.line(MEAN), MEAN, "$, as in $R$"), {"k": k}, funcs={r"\sigma": SIG})
    (_, lo, hi), = readme_rate_function("R(P)", P_).atoms(sp.Integral).pop().limits
    return sp.Integral(weight * q, (k, lo, hi)) / sp.Integral(weight, (k, lo, hi))


def readme_loss_fractions():
    """The energy fractions the README's loss averages, x_+ + x_low, as {product ("+", "low"): coefficient}."""
    head, body = readme_kinetic(LOSS, r"-\frac{\dot\gamma}{\gamma}", r"\int d\ln\varepsilon\thinspace ")
    xs = sp.symbols("x_p x_low")
    q = L.to_sympy(L.between(body, r"\langle ", r"\rangle"), {"x_+": xs[0], r"x_{\rm low}": xs[1]})
    assert sp.Poly(q, *xs).total_degree() == 1 and q.subs({xs[0]: 0, xs[1]: 0}) == 0, q   # linear: its mean at k is that of the fractions
    return {"+": q.coeff(xs[0]), "low": q.coeff(xs[1])}


def readme_loss():
    """The README's -gamma_dot/gamma per ln eps: (factor, integrand), <x_r> at fixed k = X_r(k)."""
    head, body = readme_kinetic(LOSS, r"-\frac{\dot\gamma}{\gamma}", r"\int d\ln\varepsilon\thinspace ")
    inner, arg = L.between(body, r"\langle ", r"\rangle"), L.between(body, r"\rangle(", ")")
    tokens = {r"\gamma": gam, r"\varepsilon": eps, "c": c, r"\sigma_T": sT}
    tokens["Q"] = readme_mean(sum(cf * X[r](k) for r, cf in readme_loss_fractions().items()), L.to_sympy(arg, tokens))
    body = body.replace(r"\langle " + inner + r"\rangle(" + arg + ")", "Q")
    funcs = {r"n_\gamma": NG, "R": lambda a: readme_rate_function("R(P)", a)}
    return L.to_sympy(head, tokens), L.to_sympy(body, tokens, funcs=funcs)


def readme_sink():
    """The README's photon sink per ln gamma: (factor, integrand)."""
    tokens = {r"\gamma": gam, r"\varepsilon": eps, "c": c, r"\sigma_T": sT}
    funcs = {r"n_\gamma": NG, "n_-": DENS["-"], "n_+": DENS["+"], "R": lambda a: readme_rate_function("R(P)", a)}
    head, body = readme_kinetic(SINK, r"\dot n_\gamma(\varepsilon)", r"\int d\ln\gamma\thinspace ")
    return L.to_sympy(head, tokens, funcs=funcs), L.to_sympy(body, tokens, funcs=funcs)


def direct_source(q):
    """Products of species q per ln gamma, per ln gamma' and ln eps, counted: the target t makes the product of role
    PRODUCTS[t][q] with gamma = x gamma', f_r(k, y) per ln y (times d ln y/d ln gamma at fixed k and gamma')."""
    xv = gam / gp
    total = 0
    for t in "-+":
        def per_ln_gamma(kv, f_=F[PRODUCTS[t][q]]):
            return f_(kv, readme_y(xv, kv)) * gam * sp.diff(sp.log(readme_y(xv, kv)), gam)
        total += DENS[t](gp) * collisions(per_ln_gamma, gp, eps)
    return NG(eps) * total


def direct_loss():
    """-gamma_dot/gamma per ln eps, counted: a collision costs the lepton that comes back xi gamma, with xi at k the
    mean of the README's xi = 1 - x_high = x_+ + x_low."""
    xi = L.sides(L.inline(r"\xi = 1 - x_\mathrm{high} = x_+ + x_\mathrm{low}"))[2]
    return NG(eps) * collisions(lambda kv: L.to_sympy(xi, {"x_+": X["+"](kv), r"x_\mathrm{low}": X["low"](kv)}), gam, eps)


def direct_sink():
    """Photons removed per ln eps, per ln gamma, counted: every collision removes the photon; the isotropic leptons meet it
    with the same average (an e+ has the e-'s cross section)."""
    return -NG(eps) * (DENS["-"](gam) + DENS["+"](gam)) * collisions(lambda kv: 1, gam, eps)


def on_lines(expr, t, gl):
    """int d ln gl n_t(gl) int d ln eps n_gamma(eps) expr (expr linear in them) for the line of target t and the photon
    line, with the test functions."""
    g_t, N_t = LINES[t]
    e_ = expr.subs({gl: g_t, eps: e1}).replace(DENS[t], lambda a: N_t).replace(DENS[CONJUGATE[t]], lambda a: 0)
    e_ = e_.replace(NG, lambda a: Ng).replace(SIG, TEST_SIGMA)
    for r, f_ in TEST_F.items():
        e_ = e_.replace(F[r], f_).replace(X[r], lambda a, f_=f_: sp.integrate(f_(a, y) / a, (y, 0, a)))   # <x> = int dlny f x
    return e_.doit()


def made(source, power=0):
    """int d ln gamma gamma^power of a source of products on the lines (the products of the target t have gamma <= gamma_t)."""
    return sum(sp.integrate(gam**power * on_lines(source, t, gp) / gam, (gam, 0, LINES[t][0])) for t in "-+")


def interactions():
    """The interactions on the lines, from the README's rate of a lepton, c sigma_T int d ln eps n_gamma(eps) R(gamma eps)."""
    first, second = r"c\sigma_T \int d\ln\varepsilon\ n_\gamma(\varepsilon)", r"R(\gamma\varepsilon)"
    L.line(f"interacts at the rate ${first}$ ${second}$")
    head, nphot = first.split(r" \int d\ln\varepsilon\ ")
    tokens = {r"\gamma": gam, r"\varepsilon": eps, "c": c, r"\sigma_T": sT}
    rate = (L.to_sympy(head, tokens) * L.to_sympy(nphot, tokens, funcs={r"n_\gamma": NG})
            * L.to_sympy(second, tokens, funcs={"R": lambda a: readme_rate_function("R(P)", a)}))
    return sum(on_lines((DENS["-"](gam) + DENS["+"](gam)) * rate, t, gam) for t in "-+")


def lab_lepton():
    """(gamma, gamma beta, cos theta, sin theta) as rational functions of v, u > 0: gamma = (v + 1/v)/2, so that
    (gamma^2 - 1)^(1/2) = (v - 1/v)/2, and cos theta = (1 - u^2)/(1 + u^2)."""
    v, u = sp.symbols("v u", positive=True)
    return (v + 1 / v) / 2, (v - 1 / v) / 2, (1 - u**2) / (1 + u**2), 2 * u / (1 + u**2)


def threshold_k():
    """The threshold the README's text states (4 m_e c^2), in units of m_e c^2."""
    found = re_search(r"if the photon's energy in the lepton's rest frame exceeds \$(\d+)m_e c\^2\$",
                      L.line("An electron or positron that hits a photon makes"))
    return sp.Integer(found.group(1))


@derivation("tr.k")
def check_k():
    """k = gamma eps (1 - cos theta) is the photon's energy in the lepton's rest frame for gamma >> 1; it must exceed 4."""
    gl, gb, ct, st = lab_lepton()
    p1, ph = (gl, 0, 0, gb), (eps, eps * st, 0, eps * ct)        # the lepton along z (m_e = c = 1), the photon at theta
    k_rest = mdot(p1, ph)                                        # the photon's energy in the lepton's rest frame, p_1.k/m_e
    readme = readme_k(gl, eps).subs(mu, ct)
    s_inv = mdot(*[[a + b for a, b in zip(p1, ph)]] * 2)         # (p_1 + k)^2

    def s_of(kv):
        """The README's s = 1 + 2k (with Lee et al.'s threshold expansion)."""
        return L.to_sympy(L.sides(L.part(readme_tex(r"r = \frac{s-9}{s-1}"), "s ="))[1], {"k": kv})
    v = sp.Symbol("v", positive=True)
    ratio = sp.limit(readme / k_rest, v, sp.oo)
    with L.patched(r"exceeds $4m_e c^2$", r"exceeds $2m_e c^2$"):
        broken = threshold_k()
    return [exact("p_1.k = gamma eps (1 - beta cos theta)", k_rest - gl * eps * (1 - gb / gl * ct)),
            Result("the README's k / p_1.k -> 1 for gamma -> oo at any cos theta < 1 (gamma >> 1)", sp.simplify(ratio) == 1,
                   f"limit {sp.simplify(ratio)}"),
            exact("s = (p_1 + k)^2 = 1 + 2 p_1.k, the README's s = 1 + 2k (Lee et al.)", s_inv - s_of(k_rest)),
            exact("threshold: s = (3 m_e)^2 (three leptons at rest in the CM frame) at the README's k = 4 m_e c^2",
                  s_of(threshold_k()) - 9),
            exact_fails("control: 2 m_e c^2", s_of(broken) - 9)]


def readme_x(pk, p1k):
    """The README's x = p.k/p_1.k for the values pk = p.k, p1k = p_1.k."""
    tex = L.sides(L.display(r"$$x = \frac{"))[1].replace(r"\cdot", " ")
    return L.to_sympy(tex, {"p": sp.Symbol("pk"), "p_1": sp.Symbol("p1k"), "k": 1}).subs({sp.Symbol("pk"): pk,
                                                                                          sp.Symbol("p1k"): p1k})


@derivation("tr.x")
def check_x():
    """x = p.k/p_1.k is a product's share of the lepton's lab energy, up to terms that vanish for gamma >> 1, eps << gamma."""
    L.line(r"**Energies of the products.** For $\gamma \gg 1$ and $\varepsilon \ll \gamma$, each product carries the fraction")
    gl, gb, ct, st = lab_lepton()
    be = gb / gl
    E_, px, py, pz = sp.symbols("E p_x p_y p_z", real=True)       # a product in the lepton's rest frame
    lab = (gl * E_ + gb * pz, px, py, gl * pz + gb * E_)          # boosted to the lab (the lepton along z, m_e = 1)
    p1, ph = (gl, 0, 0, gb), (eps, eps * st, 0, eps * ct)
    xv = readme_x(mdot(lab, ph), mdot(p1, ph))
    share = lab[0] / gl                                           # its share of the lepton's lab energy gamma m_e c^2
    rest = (gl * ph[0] - gb * ph[3], ph[1], ph[2], gl * ph[3] - gb * ph[0])   # the photon in the rest frame
    kk = rest[0]
    nh = [q / kk for q in rest[1:]]                               # its direction there
    d = [nh[0], nh[1], nh[2] + be]                                # n' + beta z
    along = sum(a * b for a, b in zip(d, nh))
    across2 = sum(a * a for a in d) - along**2                    # |n'| = 1 (k is null)
    with L.patched(r"$$x = \frac{p\cdot k}{p_1\cdot k}$$", r"$$x = \frac{p_1\cdot k}{p\cdot k}$$"):
        broken = readme_x(mdot(lab, ph), mdot(p1, ph))
    return [exact("x = E' - p'.n' in the rest frame (n' the photon's direction): it depends on the products and the photon "
                  "there, not on the photon's lab direction", xv - (E_ - px * nh[0] - py * nh[1] - pz * nh[2])),
            exact("lab share - x = p'.(n' + beta z) (the lab share is E' + beta p'_z)", share - xv - (px * d[0] + py * d[1] + pz * d[2])),
            exact("(n' + beta z).n' = eps/(gamma k), k = p_1.k: along the photon, Delta = p'_par eps/(gamma k) <= eps/gamma "
                  "(|p'| < E' <= k - 1 of the 1 + k a product can take)", along - eps / (gl * kk)),
            exact("across: |n' + beta z|_perp^2 = (beta eps sin theta/k)^2 = beta^2 eps sin^2 theta/(gamma k (1 - beta cos theta)) "
                  "(|n'| = 1)", (across2 - (be * eps * st / kk)**2)**2 + (across2 - be**2 * eps * st**2 / (gl * kk * (1 - be * ct)))**2
                  + (sum(q * q for q in nh) - 1)**2),
            exact("2 (1 - beta cos theta) - sin^2 theta = (cos theta - beta)^2 + 1/gamma^2 >= 0: |Delta_perp| <= p'_perp (2 eps/"
                  "(gamma k))^(1/2), k >= 4: both parts vanish for gamma >> 1 and eps << gamma",
                  2 * (1 - be * ct) - st**2 - ((ct - be)**2 + 1 / gl**2)),
            exact_fails("control: x = p_1.k/p.k", share - broken - (px * d[0] + py * d[1] + pz * d[2]))]


def readme_lepton_rate(gl, e):
    """The README's interaction rate of a lepton gl per ln eps (with a photon density n_gamma per ln eps)."""
    found = re_search(r"interacts at the rate \$(.+?) \\int d\\ln\\varepsilon\\ (.+?)\$ \$(.+?)\$, with \$n_\\gamma\(\\varepsilon\)\$ "
                      r"the photon density per \$\\ln\\varepsilon\$", L.line("A lepton therefore interacts at the rate"))
    tokens = {r"\gamma": gl, r"\varepsilon": e, "c": c, r"\sigma_T": sT}
    return (L.to_sympy(found.group(1), tokens) * L.to_sympy(found.group(2), tokens, funcs={r"n_\gamma": NG})
            * L.to_sympy(found.group(3), tokens, funcs={"R": lambda a: readme_rate_function("R(P)", a)}))


@derivation("tr.rate")
def check_rate():
    """A lepton interacts at c sigma_T int d ln eps n_gamma R(gamma eps): the collisions with isotropic photons, counted."""
    counted = NG(eps) * collisions(lambda kv: 1, gam, eps)
    out = [same_integrands("README rate = the collisions c (1 - cos theta) sigma(k) averaged over isotropic photons, per ln "
                           "eps (any sigma; integrands in k)", readme_lepton_rate(gam, eps), counted)]
    with L.patched(r"$R(\gamma\varepsilon)$, with", r"$R(\varepsilon)$, with"):
        out.append(same_integrands("control: R(eps)", readme_lepton_rate(gam, eps), counted, must_differ=True))
    return out


TRIPLET_SRC = L.ROOT / "src" / "am3_mojo" / "physics" / "triplet.mojo"


def code_sig_unit(src):
    """The code's SIG_UNIT (its tables' sigma/(alpha r_e^2) -> sigma/sigma_T), exact, with the code's alpha_PDG and its
    3.141592653589793 taken as pi."""
    tex = re_search(r"comptime SIG_UNIT: Float64 = ([^#]+)#", src).group(1).replace("3.141592653589793", "pi")
    return sp.sympify(" ".join(tex.split()), locals={**mojo_constants.read(("alpha_PDG",)), "pi": sp.pi}, rational=True)


@derivation("tr.R")
def check_R():
    """R(P) and G_r(P, x) are isotropic averages of the collision rate, and the code's SIG_UNIT uses PDG 2026's alpha.
    The average is (1/2) int dmu (1 - mu) sigma(P (1 - mu))."""
    L.inline(r"k = \gamma\varepsilon(1 - \cos\theta)")
    line = readme_tex(r"R(P) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k")
    assert L.sides(L.parts(line)[0])[1] == r"\frac{1}{2P^2}\int_4^{2P} dk\thinspace k\thinspace \frac{\sigma(k)}{\sigma_T}"
    out = []
    v = sp.Symbol("v", positive=True)   # 1 - cos(theta)
    for power in (0, 1, 2, 3):
        sigma = sT * (k - 4)**power                              # a test cross section with the threshold
        average = sp.integrate(v * sigma.subs(k, P * v) / sT, (v, 4 / P, 2)) / 2
        readme = sp.integrate(k * sigma / sT, (k, 4, 2 * P)) / (2 * P**2)
        out.append(exact(f"sigma = sigma_T (k - 4)^{power}: README R(P) = the average", average - readme))
    # G_r: the products of these collisions per ln x, f_r(k, y) per ln y with y = xk (d ln y = d ln x at fixed k)
    per_ln_x = lambda kv: F["r"](kv, readme_y(x, kv)) * x * sp.diff(sp.log(readme_y(x, kv)), x)   # noqa: E731
    counted = collisions(per_ln_x, gp, eps) / (c * sT)
    out.append(same_integrands("README G_r(gamma' eps, x) = the average of sigma f_r(k, y), y = xk, per ln x "
                               "(any sigma, f_r; integrands in k)", readme_rate_function("G_r(P, x)", gp * eps, x), counted))
    with L.patched(r"f_r(k, xk)", r"f_r(k, x)"):
        out.append(same_integrands("control: f_r(k, x) in G_r", readme_rate_function("G_r(P, x)", gp * eps, x), counted,
                                   must_differ=True))
    # sigma_T = (8 pi/3) r_e^2, so sigma/sigma_T = (sigma/(alpha r_e^2)) 3 alpha/(8 pi), with PDG 2026's alpha
    src = TRIPLET_SRC.read_text()
    unit = 1 / sp.Rational(INV_ALPHA) * re**2 / (sp.Rational(8, 3) * sp.pi * re**2)
    out.append(exact("the code's SIG_UNIT = alpha r_e^2/sigma_T with PDG 2026's alpha (CODATA 2022)",
                     code_sig_unit(src) - unit))
    broken = src.replace("3.0 * alpha_PDG / (", "3.0 * (1.0 / 137.035999084) / (")
    out.append(exact_fails("control: CODATA 2018's alpha, 1/137.035999084", code_sig_unit(broken) - unit))
    return out


@derivation("tr.x_rest")
def check_x_rest():
    """x = p.k/p_1.k is E - p_z in the lepton's rest frame with the photon along z."""
    E_, px, py, pz, w_ = sp.symbols("E p_x p_y p_z omega", real=True)
    dot = lambda a, b: a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    p, p1, kv = (E_, px, py, pz), (1, 0, 0, 0), (w_, 0, 0, w_)
    ours = L.to_sympy(L.sides(readme_tex(r"x = \frac{p\cdot k}{p_1\cdot k}"))[1].replace(r"\cdot", " "),
                      {"p": sp.Symbol("pk"), "p_1": sp.Symbol("p1k"), "k": 1})
    x_ = ours.subs({sp.Symbol("pk"): dot(p, kv), sp.Symbol("p1k"): dot(p1, kv)})
    return [exact("x = E - p_z", sp.simplify(x_ - (E_ - pz)))]


@derivation("tr.xi")
def check_xi():
    """x_+ + x_low + x_high = 1 (momentum conservation, k^2 = 0), so xi = 1 - x_high = x_+ + x_low."""
    comps = [sp.symbols(f"{nm}0:4", real=True) for nm in ("a", "b")]
    w_ = sp.Symbol("omega", positive=True)
    dot = lambda a, b: a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    p1, kv = (1, 0, 0, 0), (w_, 0, 0, w_)
    a, b = comps
    c_ = tuple(p1[i] + kv[i] - a[i] - b[i] for i in range(4))   # the third product
    total = sum(dot(q, kv) for q in (a, b, c_)) / dot(p1, kv)
    return [exact("sum of the three x = 1", sp.simplify(total - 1))]


@derivation("tr.pairs")
def check_pairs():
    """The pair sources count the products of the collisions (charge conjugates for e+ targets), one e+ and one e- each."""
    head, pos, ele = readme_pairs()
    swap = {DENS["-"](gp): DENS["+"](gp), DENS["+"](gp): DENS["-"](gp)}
    out = [same_integrands("README dot n_+ = the positrons of the collisions (any sigma, f_r; per ln gamma' and ln eps, "
                           "integrands in k)", head * pos, direct_source("+")),
           same_integrands("README electrons (n_- G_low + n_+ G_+) = the electrons of the collisions", head * ele,
                           direct_source("-")),
           exact("the electrons' bracket = dot n_+'s with n_- and n_+ exchanged", pos.xreplace(swap) - ele)]
    assert all(sp.simplify(sp.integrate(f_(k, y) / y, (y, 0, k)) - 1) == 0 for f_ in TEST_F.values())
    n_int = interactions()
    out += [exact("lines of e-, e+ and photons, test sigma and f_r: positrons made = interactions (README's rate of a lepton)",
                  made(head * pos) - n_int),
            exact("electrons made = interactions", made(head * ele) - n_int)]
    with L.patched(r"n_+(\gamma')\thinspace G_{\rm low}", r"n_+(\gamma')\thinspace G_{+}"):
        h_b, pos_b, _ = readme_pairs()
    out.append(same_integrands("control: G_+ for the positrons of e+ targets", h_b * pos_b, direct_source("+"),
                               must_differ=True))
    with L.patched(r"G_{+}\negthinspace \left(\gamma'\varepsilon,\ \frac{\gamma}{\gamma'}\right)",
                   r"G_{+}\negthinspace \left(\gamma'\varepsilon,\ \frac{\gamma'}{\gamma}\right)"):
        h_b, pos_b, _ = readme_pairs()
    out.append(same_integrands("control: x = gamma'/gamma in G_+", h_b * pos_b, direct_source("+"), must_differ=True))
    return out


@derivation("tr.loss")
def check_loss():
    """The continuous loss is the mean energy the collisions cost the lepton that comes back, the energy the pairs take."""
    head, loss = readme_loss()
    out = [same_integrands("README -gamma_dot/gamma = the mean of xi gamma over the collisions, per gamma (any sigma, <x_r>; "
                           "per ln eps, integrands in k)", head * loss, direct_loss())]
    h_p, pos, ele = readme_pairs()
    gained = made(h_p * (pos + ele), power=1)

    def lost(h, l_):
        return sum(on_lines((DENS["-"](gam) + DENS["+"](gam)) * gam * h * l_, t, gam) for t in "-+")
    out.append(exact("lines, test sigma and f_r: the energy the leptons lose = the energy of the pairs made",
                     lost(head, loss) - gained))
    with L.patched(r"the weight $k\sigma(k)$, as in $R$", r"the weight $\sigma(k)$, as in $R$"):
        h_b, loss_b = readme_loss()
    out += [same_integrands("control: the weight sigma(k) in <...>", h_b * loss_b, direct_loss(), must_differ=True),
            exact_fails("control: the weight sigma(k): energy lost = gained", lost(h_b, loss_b) - gained)]
    return out


@derivation("tr.sink")
def check_sink():
    """The photon sink: every collision removes its photon; one photon per interaction."""
    head, body = readme_sink()
    out = [same_integrands("README dot n_gamma = the photons the collisions remove (any sigma; per ln gamma, integrands "
                           "in k)", head * body, direct_sink())]

    def absorbed(h, b):
        return -sum(on_lines(h * b, t, gam) for t in "-+")
    out.append(exact("lines, test sigma: photons absorbed = interactions (README's rate of a lepton)",
                     absorbed(head, body) - interactions()))
    with L.patched(r"\bigl(n_-(\gamma) + n_+(\gamma)\bigr)", r"n_-(\gamma)"):
        h_b, b_b = readme_sink()
    out += [same_integrands("control: n_- alone in the sink", h_b * b_b, direct_sink(), must_differ=True),
            exact_fails("control: n_- alone: photons absorbed = interactions", absorbed(h_b, b_b) - interactions())]
    return out


def readme_match(pattern, pin, file="checks"):
    """The match of the pattern in the README line that holds the pin (a pin without the numbers and equations read)."""
    found = re_search(pattern, L.line(pin, file))
    if found is None:
        raise LookupError(f"{file}: pattern not found in the line of {pin}")
    return found


IC_LINE = "**Against inverse Compton:**"


def readme_kn_ic(eps_):
    """-dgamma/dt of inverse Compton in its Klein-Nishina limit, as the Results quote it."""
    m_ = readme_match(r"its Klein–Nishina limit, \$-\\dot\\gamma\$ \$= ([^$]+)\$ \$([^$]+)\$, for", IC_LINE)
    tokens = {"c": c, r"\sigma_T": sT, "n": n, r"\varepsilon": eps_, r"\gamma": gam}
    return L.to_sympy(m_[1], tokens) * L.to_sympy(m_[2], tokens)


def kn_ic_claims():
    eps_ = sp.Symbol("varepsilon", positive=True)
    ours = readme_kn_ic(eps_)
    # B&G eq. 2.57 (scan, p. 244): -dE/dt = pi r_0^2 m^2 c^5 int n(e)/e (ln(4 e gamma/m c^2) - 11/6) de,
    # e the photon energy; for n(e) = n delta(e - eps m c^2): divided by m c^2 for -dgamma/dt
    r0 = sp.sqrt(3 * sT / (8 * sp.pi))                            # pi r_0^2 = (3/8) sigma_T
    e0 = eps_ * m * c**2
    bg = sp.pi * r0**2 * m**2 * c**5 * n / e0 * (sp.log(4 * e0 * gam / (m * c**2)) - sp.Rational(11, 6)) / (m * c**2)
    return [exact("README = eq. 2.57 / (m c^2), with pi r_0^2 = 3 sigma_T/8", ours - bg)]


@derivation("tr.kn_ic")
def check_kn_ic():
    """The Klein-Nishina limit of the IC loss is Blumenthal & Gould's eq. 2.57 for photons of one energy."""
    return kn_ic_claims() + _controls(kn_ic_claims, [(r"$= (3/8) c\sigma_T n\varepsilon^{-1}$", r"$= (3/4) c\sigma_T n\varepsilon^{-1}$"),
                                                     (r"- 11/6]$, for", r"- 11/3]$, for")])


def bg257_claims():
    eps_, r0_ = sp.Symbol("varepsilon", positive=True), sp.Symbol("r_0")
    m_ = readme_match(r"eq\. 2\.57 of Blumenthal & Gould 1970, \$-dE/d\{t\}\$ \$= (.+?) \\int d\\varepsilon\\ n\(\\varepsilon\)\$ "
                      r"\$([^$]+)\$ \$([^$]+)\$, which", IC_LINE)
    ours = (L.to_sympy(m_[1], {"r_0": r0_, "m": m, "c": c})
            * L.to_sympy(m_[2] + m_[3], {"m": m, "c": c, r"\varepsilon": eps_, r"\gamma": gam}))
    scan = sp.pi * r0_**2 * m**2 * c**5 / eps_ * (sp.log(4 * eps_ * gam / (m * c**2)) - sp.Rational(11, 6))
    return [exact("README's quotation = eq. 2.57 as read from the scan (integrand per n(eps) d eps)", ours - scan)]


@derivation("tr.bg257")
def check_bg257():
    """The README quotes Blumenthal & Gould's eq. 2.57 as printed (scan, p. 244)."""
    return bg257_claims() + _controls(bg257_claims, [(r"m^2c^5 \int d\varepsilon", r"m^2c^4 \int d\varepsilon"),
                                                     (r"mc^2) - 11/6]$", r"mc^2) - 11/3]$")])


def mast28_claims():
    eps_ = sp.Symbol("varepsilon", positive=True)
    m_ = readme_match(r"which Mastichiadis writes as \$([^$]+)\$ \$([^$]+)\$, his eq\. 28", IC_LINE)
    ours = (L.to_sympy(m_[1], {"r_e": re, "c": c, "n": n, r"\varepsilon": eps_})
            * L.to_sympy(m_[2], {r"\gamma": gam, r"\varepsilon": eps_}))
    # eq. 28 (scan, p. 241), with L = m_e c^2 (-dgamma/dt) and energies in m_e c^2
    q = sp.log(4 * gam * eps_) - sp.Rational(11, 6)
    loss = q * sp.pi * re**2 * m * c**3 * n / eps_ / (m * c**2)
    return [exact("README = eq. 28 solved for -dgamma/dt", ours - loss)]


@derivation("tr.mast28")
def check_mast28():
    """Mastichiadis' eq. 28, q_ics = L eps_0/(pi r_0^2 m_e c^3 n_0) = ln(4 E_0 eps_0) - 11/6, as quoted."""
    return mast28_claims() + _controls(mast28_claims, [(r"writes as $\pi r_e^2 c n", r"writes as $\pi r_e^2 c^2 n"),
                                                       (r"$[\ln(4\gamma\varepsilon) - 11/6]$, his", r"$[\ln(2\gamma\varepsilon) - 11/6]$, his")])


MAST26 = "Mastichiadis' own fit (his eq. 26, isotropic photons) gives a mean energy of $"


def readme_mast26(eps_):
    """(the mean energy per produced particle, the pair's share of the lepton's energy) of Mastichiadis' eq. 26, as quoted."""
    tokens = {r"\gamma": gam, r"\varepsilon": eps_}
    m_ = readme_match(r"gives a mean energy of \$([^$]+)\$ per produced particle, so the pair takes \$([^$]+)\$ of the lepton's "
                      r"energy", MAST26)
    return L.to_sympy(m_[1], tokens), L.to_sympy(m_[2], tokens)


def mast26_claims():
    eps_ = sp.Symbol("varepsilon", positive=True)
    mean, share = readme_mast26(eps_)
    eq26 = sp.Rational(5, 2) / eps_ * (gam * eps_)**sp.Rational(1, 4)          # scan, p. 240
    return [exact("README mean energy = eq. 26", mean - eq26),
            exact("the pair's share 2 <E_+>/gamma = 5 (gamma eps)^(-3/4)", 2 * mean / gam - share)]


@derivation("tr.mast26")
def check_mast26():
    """Mastichiadis' eq. 26, <E_+> = 2.5 (E_0 eps_0)^(1/4)/eps_0: the pair takes 5 (gamma eps)^(-3/4)."""
    return mast26_claims() + _controls(mast26_claims, [(r"of $2.5(\gamma\varepsilon)^{1/4}", r"of $2.4(\gamma\varepsilon)^{1/4}"),
                                                       (r"takes $5(\gamma\varepsilon)^{-3/4}$", r"takes $5(\gamma\varepsilon)^{-1/4}$")])


LEE25 = "Mastichiadis' fit as Lee (1998) quotes it, $"
LEE25_TEX = (r"\eta(s) \simeq 1.768 (s/m_e^2)^{-3/4}", r"where $s$ is the squared center of mass (CM) energy")


def readme_lee25(me):
    """(eta(s), s(k)) of Mastichiadis' fit as Lee (1998) quotes it."""
    m_ = readme_match(r"quotes it, \$\\eta = ([^$]+)\$ \(his eq\. 25, with \$s = ([^$]+)\$\)", LEE25)
    return L.to_sympy(m_[1], {"s": s, "m_e": me}), L.to_sympy(m_[2], {"m_e": me, "k": k})


def lee25_claims():
    me = sp.Symbol("m_e", positive=True)
    ours, s_def = readme_lee25(me)
    theirs = L.to_sympy(LEE25_TEX[0].split(r"\simeq")[1], {"s": s, "m_e": me})
    # the squared CM energy of an electron at rest and a photon of energy k m_e c^2 along z (c = 1)
    total = (me + k * me, 0, 0, k * me)
    cm = total[0]**2 - total[1]**2 - total[2]**2 - total[3]**2
    return [exact("README = eq. 25", ours - theirs),
            exact("README s = (p_1 + k)^2, Lee's squared CM energy, with k in m_e c^2", s_def - cm)]


@derivation("tr.lee25")
def check_lee25():
    """Lee's eq. 25, eta = 1.768 (s/m_e^2)^(-3/4), with s = m_e^2 (1 + 2k) his squared CM energy."""
    return ([in_source("eq. 25 and s as transcribed", LEE98, *LEE25_TEX)] + lee25_claims()
            + _controls(lee25_claims, [(r"1.768(s/m_e^2)^{-3/4}$ (his", r"1.768(s/m_e^2)^{-1/4}$ (his"),
                                       (r"$s = m_e^2(1 + 2k)$", r"$s = m_e^2(1 + k)$")]))


ANGUELOV_PIN = "The fit of Anguelov et al. (1999) is $"
ANGUELOV_TEX = (r"f_1(s) &=&E_m(E_0,s)s/E_0=0.195\ln ^2(2s)", r"s=E_0\varepsilon _0(1-\beta \cos \theta )")


def readme_anguelov():
    """(the mean energy fraction per produced particle of Anguelov et al.'s fit, the README's figure for the pair)."""
    m_ = readme_match(r"is \$([^$]+)\$ per produced particle \(their eq\. 9; ([\d.]+) for the pair\)", ANGUELOV_PIN)
    return L.to_sympy(m_[1], {"k": k}), m_[2]


def anguelov_claims():
    ours, pair_tex = readme_anguelov()
    theirs = L.to_sympy(ANGUELOV_TEX[0].split("=")[-1], {"s": k}) / k    # E_m/E_0 = f_1/s, s = k
    pair = sp.simplify(2 * ours * k / sp.log(2 * k)**2)
    return [exact("README = eq. 9 / s", ours - theirs),
            Result(f"{pair_tex} for the pair: twice the coefficient of ln^2(2k)/k", not pair.free_symbols and rounds_to(pair, quoted(pair_tex)),
                   f"{pair}")]


@derivation("tr.anguelov")
def check_anguelov():
    """Anguelov et al.'s eq. 9, f_1(s) = E_m s/E_0 = 0.195 ln^2(2s), with their s = k: 0.39 ln^2(2k)/k for the pair."""
    return ([in_source("eq. 9 and s = E_0 eps_0 (1 - beta cos theta) as transcribed", ANGUELOV,
                       *ANGUELOV_TEX)] + anguelov_claims()
            + _controls(anguelov_claims, [(r"is $0.195 \ln^2(2k)/k$ per", r"is $0.195 \ln^2(2k)/k^2$ per"),
                                          ("0.39 for the pair", "0.37 for the pair")]))


DS = "the analytic rate of Dermer & Schlickeiser (1991), $"
EPS, E_STAR = sp.symbols("varepsilon e_star", positive=True)
DS_EQ7 = 4 * al / 3 * (c * sT * n) * (gam / EPS)**sp.Rational(1, 2)          # scan, p. 415, t_C = 1/(c sigma_T n)


def readme_ds():
    """-dgamma/dt of Dermer & Schlickeiser's eq. 7, as quoted."""
    m_ = readme_match(r"Dermer & Schlickeiser \(1991\), \$-\\dot\\gamma\$ \$= ([^$]+)\$ \(their eq\. 7\)", DS)
    return L.to_sympy(m_[1], {r"\alpha_f": al, "c": c, r"\sigma_T": sT, "n": n, r"\gamma": gam, r"\varepsilon": EPS})


def ds_claims():
    ours = readme_ds()
    return [exact("README = eq. 7 (E in m_e c^2)", ours - DS_EQ7)]


@derivation("tr.ds")
def check_ds():
    """Dermer & Schlickeiser's eq. 7, -dE/dt = (4 alpha_f/3 t_C) (E/eps)^(1/2), the leading term of their eq. 6."""
    eq6 = 2**sp.Rational(-1, 2) * al * c * sT * n / (EPS**2 * gam) * sp.integrate(sp.sqrt(E_STAR), (E_STAR, 4, 2 * EPS * gam))
    return (ds_claims()
            + [Result("eq. 7 is eq. 6 with f = 1 for eps E >> 1", sp.limit(eq6 / DS_EQ7, gam, sp.oo) == 1,
                      f"eq. 6/eq. 7 = {sp.simplify(eq6 / DS_EQ7)}")]
            + _controls(ds_claims, [(r"(\gamma/\varepsilon)^{1/2}$ (their eq. 7)", r"(\gamma/\varepsilon)^{1/3}$ (their eq. 7)"),
                                    (r"$= (4/3)\alpha_f c", r"$= (4/9)\alpha_f c")]))


CC952 = "In its Chang & Cooper regime (slow cooling), the leptons lose $"
GRID_STEP = r"the step of the grid in $\ln E$"
XI_BIN = "In interactions where the lepton loses more than a grid bin"


def grid_step():
    """Delta, the step of the grids in ln E, as tests/reference/README.md states it (x_scaling.py uses it)."""
    return L.to_sympy(readme_match(r"with \$\\Delta = ([^$]+)\$ the step of the grid in \$\\ln E\$", GRID_STEP, "reference")[1], {})


def readme_xi_bin():
    """The xi of "more than a grid bin" as the Results quote it (inelasticity.py uses it), and its digits."""
    return quoted(readme_match(r"loses more than a grid bin \(\$\\xi \\gt ([\d.]+)\$\)", XI_BIN)[1])


def cc952_claims():
    m_ = readme_match(r"the leptons lose \$([^$=]+)= ([^$]+)\$ of the pairs' energy", CC952)
    delta = grid_step()
    form = float(L.to_sympy(m_[1], {r"\Delta": delta}))
    bin_ = 1 - mp.exp(-mp.mpf(str(delta)))
    return [Result(f"= {m_[2]} for the grid's Delta = {float(delta):g} (tests/reference/README.md)", rounds_to(form, quoted(m_[2])),
                   f"{form:.5f}"),
            Result(f"more than a grid bin: xi > 1 - e^-Delta, {mp.nstr(readme_xi_bin()[0], 2)} as the Results quote it",
                   rounds_to(bin_, readme_xi_bin()), f"{mp.nstr(bin_, 5)}")]


@derivation("tr.cc952")
def check_cc952():
    """(1 - e^-Delta)/Delta = 0.952 for the grid's Delta = 0.1, and xi > 0.095 is more than a grid bin."""
    return (cc952_claims()
            + _controls(cc952_claims, [("= 0.952$ of the pairs'", "= 0.962$ of the pairs'"),
                                       (r"lose $(1 - e^{-\Delta})/\Delta =", r"lose $(1 - e^{-2\Delta})/\Delta ="),
                                       (r"a grid bin ($\xi \gt 0.095$)", r"a grid bin ($\xi \gt 0.085$)")])
            + _controls(cc952_claims, [(r"with $\Delta = 0.1$ the step", r"with $\Delta = 0.2$ the step")], "reference"))


MAST_CROSS = r"Mastichiadis' loss rate (eq. 27) exceeds the inverse Compton loss above $\gamma\varepsilon = "


def mast27(X, alpha=1 / mp.mpf(INV_ALPHA), log=mp.log, pi=mp.pi):
    """Mastichiadis' eq. 27 (scan, p. 241), x = E_0 eps_0: q_tpp = (5 alpha/pi) x^(1/4) [(28/9) ln 2x - 218/27], the loss in
    units of pi r_0^2 c n/eps (log, pi: mpmath's or numpy's)."""
    return 5 * alpha / pi * X**0.25 * (84 * log(2 * X) - 218) / 27


def mast28(X, log=mp.log):
    """Mastichiadis' eq. 28 (scan, p. 241): q_ics = ln 4x - 11/6, in the same units."""
    return (6 * log(4 * X) - 11) / 6


@derivation("tr.mast_cross")
def check_mast_cross():
    """(4 alpha_f)^-4 = 1.4e6 is Mastichiadis' estimate of where his eqs. 27 and 28 meet; they meet at 9e5."""
    q = lambda X: mast27(X) - mast28(X)   # noqa: E731
    cross = mp.findroot(q, 9e5)

    def claim():
        m_ = readme_match(r"loss above \$\\gamma\\varepsilon = ([^$]+)\$ \(he gives \$([^$=]+)= ([^$]+)\$\)", MAST_CROSS)
        estimate = float(L.to_sympy(m_[2], {r"\alpha_f": ALPHA}))
        return [Result(f"eq. 27 = eq. 28 at gamma eps = {shown(m_[1])} (README: eq. 27 exceeds the inverse Compton loss above it)",
                       rounds_to(cross, quoted(m_[1])) and q(2 * cross) > 0, f"at gamma eps = {float(cross):.3g}"),
                Result(f"his estimate {m_[2].strip()} = {shown(m_[3])}", rounds_to(estimate, quoted(m_[3])), f"{estimate:.4g}")]
    return claim() + _controls(claim, [(r"$\gamma\varepsilon = 9 \times 10^5$ (he", r"$\gamma\varepsilon = 8 \times 10^5$ (he"),
                                       (r"= 1.4 \times 10^6$)", r"= 1.6 \times 10^6$)"),
                                       (r"$(4\alpha_f)^{-4} =", r"$(4\alpha_f)^{-3} =")])



# --- the forms of the literature as numpy functions, for triplet/inelasticity.py ---

def literature_forms():
    """{name: numpy function} of the forms the Results quote, checked above. The pair's share of the lepton's energy
    at k: "lee" (Mastichiadis' fit as Lee quotes it, m_e = 1), "anguelov" (twice their value per particle). At
    P = gamma eps in monoenergetic isotropic photons: the share "mast26" (Mastichiadis' eq. 26) and the energy-loss
    rates per c sigma_T n gamma, "ds" (Dermer & Schlickeiser's eq. 7), "ic" (inverse Compton, Klein-Nishina limit)
    and "mast27" (Mastichiadis' eq. 27, pi r_e^2 = 3 sigma_T/8)."""
    P_ = sp.Symbol("P", positive=True)
    eta, s_def = readme_lee25(1)
    per_loss = lambda loss: sp.simplify((loss / (c * sT * n * gam)).subs(EPS, P_ / gam).subs(sp.Symbol("varepsilon", positive=True), P_ / gam))   # noqa: E731
    share26 = readme_mast26(sp.Symbol("varepsilon", positive=True))[1].subs(sp.Symbol("varepsilon", positive=True), P_ / gam)
    forms = {"lee": (k, eta.subs(s, s_def)), "anguelov": (k, 2 * readme_anguelov()[0]), "mast26": (P_, sp.simplify(share26)),
             "ds": (P_, per_loss(readme_ds()).subs(al, ALPHA)), "ic": (P_, per_loss(readme_kn_ic(sp.Symbol("varepsilon", positive=True))))}
    out = {}
    for name, (var, expr) in forms.items():
        assert expr.free_symbols <= {var}, (name, expr)
        out[name] = sp.lambdify(var, expr, "numpy")
    out["mast27"] = lambda P: 3 / 8 * mast27(P, float(ALPHA), np.log, np.pi) / P
    return out

if __name__ == "__main__":
    run()
