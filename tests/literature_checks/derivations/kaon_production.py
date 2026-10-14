"""Derivation checks of the kaon production equations of README.md (Production in photo-hadronic interactions,
Production in pp collisions) and of the AM3 paper's misprint of Kafexhiu et al.'s cross section
(tests/reference/README.md, Misprints of the AM3 paper). Sources, read from their arXiv TeX (equation and page
numbers of the arXiv versions):
  - Hümmer, Rüger, Spanier & Winter (2010), arXiv:1002.1310v3: eqs. 3-8 (pp. 6-7), 13-16 (p. 11), 27-28 (p. 15) and
    their K+ channel KP (eqs. 41-42, Sec. 4.5, p. 24). Their E_p, \\varepsilon, m_p, Gamma^IT, IT and N_p are the
    README's E_N, epsilon, m_p c^2, t_{c,N}^-1 (they set c = 1), the channel c and N_N. Their cross sections vanish
    below eps_th, so their lower limits eps_th are the README's 0. Their pion angle is measured from the nucleon
    (eq. 14), the README's theta from the photon: cos theta_pi = -cos theta.
  - Koers, Pe'er & Wijers (2006), arXiv:hep-ph/0611219v1: eqs. 16-18 (p. 7), Table III (p. 17), and for the figures
    Tables I-II and their multiplicity fit (eq. 15). Their "m c^2" (the meson's mass, energies in GeV) is the
    README's m_K. Their \\log is log10: they write \\ln elsewhere, and eq. 17a completes the square of eq. 17b only
    for log10.
  - Kafexhiu et al. (2014), arXiv:1406.7369v2, eq. 1 (p. 2), and Klinger et al. (2024, the AM3 paper),
    arXiv:2312.13371v2, eq. A52 (p. 26), which quotes it with T_p/E_th inverted in the threshold factor.
  - Kelner, Aharonian & Bugayov (2006), arXiv:astro-ph/0606058v1: F_pi (eqs. 6, QGSJET, and 12, SIBYLL), the pp source
    (eqs. 70-71) and the delta-functional approximation (eqs. 75-77; eq. 76 prints k for kappa).
  - Anticic et al. (NA49, 2010), arXiv:1004.1889v1, Table 11. Its points below 100 GeV are Fesefeldt et al.'s (1979,
    Nucl. Phys. B 147, 317, Table 1). The multiplicity fit keeps their K0_S and leaves out their K+-, derived by
    taking pp -> K+ K- X equal to pp -> K0 anti-K0 X (p. 324).
The figures quoted for the pp kaons and the thresholds of M2 and M3 are recomputed from the README's equations and
tables, with these sources and PDG 2026 (the masses; A_L of the K0_L listing, p. 34).

Hümmer et al.'s eq. 42 (the response function of KP) does not follow from their eqs. 27 and 41 in its second and third
step: eq. 27 gives 1.2022 and 0.6391 GeV^2 where 0.3 and 0.16 are printed, and the printed function jumps at 2y = 1.2
and 1.65 GeV, where the integral is continuous.

The code's choices in the README are checked for their consequences (energy and counts kept, continuity). Kinematics
come from Lorentz transformations: the kaon's share of the nucleon energy is the limit E_N -> oo (fixed s) of E_K/E_N
in the lab, exactly for a head-on photon and numerically for an oblique one. The integrals are redone with sympy for
any photon density and cross section.

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/kaon_production.py
"""
import itertools
import math
import re
from functools import lru_cache

import mpmath as mp
import numpy as np
import sympy as sp
from sympy.physics.quantum.cg import CG
from scipy.integrate import quad
from scipy.optimize import brentq

import latex as L
import mojo_constants
from latex import HUEMMER, KAFEXHIU, KELNER, KLINGER, KOERS, NA49
from pdg import MASS_MEV
from verify import Result, control, derivation, exact, exact_fails, in_source, log_uniform, numeric, rng, run


E, eps, m, c, y, er, eth = sp.symbols("E_N epsilon m_p c y epsilon_r epsilon_th", positive=True)
EK, chi, M, x = sp.symbols("E_K chi M x", positive=True)
W, mN, mY, mK, Es, ps, u, w = sp.symbols("W m_N m_Y m_K E_star p_star u w", positive=True)
mu, ct, pL = sp.symbols("mu c_theta p_L", real=True)   # cos theta_pgamma, cos theta, p*_L
n, G, sigma, N = (sp.Function(name) for name in ("n_gamma", "G", "sigma", "N"))
MC2 = m * c**2   # Hümmer's m_p


def rational(e):
    """The expression with its decimals (1.9, 2.0 mubarn) as exact rationals."""
    return sp.nsimplify(e, rational=True)


def piece(snippet, part):
    """A part of a snippet copied from a paper (asserted to be in it), to be parsed on its own."""
    assert part in snippet, part
    return part


def integrand(expr):
    """(integrand times the factors in front, (variable, lower, upper)) of a product with one Integral."""
    ints = [a for a in sp.Mul.make_args(expr) if isinstance(a, sp.Integral)]
    assert len(ints) == 1 and len(ints[0].limits) == 1, expr
    return (expr / ints[0]) * ints[0].function, ints[0].limits[0]


# --- Hümmer et al. (2010), copied from the TeX source (\drb = Gamma^IT, \dsb = sigma^IT, \dcb = chi^IT) ---

H_EQ3 = (r"\drb (E_p) = \int d \varepsilon \int\limits_{-1}^{+1} \frac{d \cos \theta_{p \gamma}}{2} \, "
         r"(1- \cos \theta_{p \gamma}) \, n_\gamma(\varepsilon,\cos \theta_{p \gamma}) \, \dsb(\epsilon_r)")
H_EPSR = r"$\epsilon_r=E_p \varepsilon/m_p (1 - \cos_{p \gamma})$"     # sic: \cos_{p\gamma} for cos theta_pgamma
H_ISO = r"we will assume isotropy $n_\gamma(\varepsilon,\cos \theta_{p \gamma}) \simeq n_\gamma(\varepsilon)$"
H_EQ4 = (r"\drb (E_p) = \frac{1}{2} \frac{m_p^2}{E_p^2} \int\limits_{\frac{\epsilon_{\mathrm{th}} m_p}{2 E_p}}^{\infty} "
         r"d \varepsilon \frac{n_\gamma(\varepsilon)}{\varepsilon^2} "
         r"\int\limits_{\epsilon_{\mathrm{th}}}^{2 E_p \varepsilon/m_p} d \epsilon_r \, \epsilon_r  \, \dsb(\epsilon_r)")
H_TH = r"$\epsilon_{\mathrm{th}} \simeq 150 \, \mathrm{MeV}$ is the threshold below which the cross sections are zero"
H_EQ5 = r"s(\epsilon_r) = m_p^2 + 2 \, m_p \, \epsilon_r"
H_EQ7 = (r"Q_b(E_b) = \int\limits_{E_b}^{\infty} \frac{dE_p}{E_p} \, N_p(E_p) \, "
         r"\int\limits_{\frac{\epsilon_{\mathrm{th}} m_p}{2 E_p}}^{\infty} d\varepsilon \, n_\gamma(\varepsilon) \,  R_b( x,y )")
H_EQ8 = r"x \equiv \frac{E_b}{E_p} \, , \quad  y\equiv \frac{E_p \, \varepsilon}{m_p}"
H_EQ14 = (r"E_\pi=\gamma_\text{cm} E_\pi^\text{cm} (1+ \beta_\pi^\text{cm}\cos{\theta_\pi^\text{cm}}) "
          r"= E_p \, \frac{E_\pi^\text{cm}}{\sqrt{s}}(1+\beta_\pi^\text{cm}\cos{\theta_\pi^\text{cm}})")
H_BACK = r"Note that backward scattering of the pion means that $\cos{\theta_\pi^\text{cm}}=-1$"
H_EQ15 = r"E_{\pi}^\text{cm}=\frac{s-m_p^2+m_\pi^2}{2\sqrt{s}}"
H_EQ16 = r"\chi(\epsilon_r)=  \frac{s(\epsilon_r)-m_p^2+m_\pi^2}{2 s(\epsilon_r)} (1+\beta_\pi^{\mathrm{cm}} \cos{\theta_\pi})"
H_KAON = (r"we also use Eq.~(\ref{eqn:1pion}) for the kaon production (with the rough estimate "
          r"$\left<\cos{\theta_\pi}\right> \simeq 0$), where the replacements $m_p\rightarrow m_\Lambda$ and "
          r"$m_\pi\rightarrow m_K$ have to be made")
H_CHI35 = r"we have $\chi_{K^+} \simeq 0.35$ (computed for $\epsilon_r \simeq 1.4 \, \mathrm{GeV} $ at the peak)"
H_EQ27 = (r"R^{\mathrm{IT}}(x,y)= \delta(x - \chi^{\mathrm{IT}}) \,  M_b^{\mathrm{IT}} \, f^{\mathrm{IT}}(y)  \quad "
          r"\text{with} \quad f^{\mathrm{IT}}(y) \equiv \frac{1}{2y^2} \int\limits_{\epsilon_{\mathrm{th}}}^{2y} d \epsilon_r "
          r"\, \epsilon_r \, \sigma^{\mathrm{IT}}(\epsilon_r)")
H_EQ28 = (r"Q_b^{\mathrm{IT}} = N_p \left( \frac{E_b}{\chi^{\mathrm{IT}}} \right) \frac{m_p}{E_b} "
          r"\int\limits_{\epsilon_{\mathrm{th}}/2}^\infty dy \, "
          r"n_\gamma \left( \frac{m_p \, y \, \chi^{\mathrm{IT}}}{E_b} \right) \,  M_b^{\mathrm{IT}}  \, f^{\mathrm{IT}}(y)")
H_EQ41 = [r"0 & \epsilon_r < 1.0 \, \mathrm{GeV}",
          r"2.0 \, \mu \mathrm{barn}  & 1.0 \, \mathrm{GeV}\le \epsilon_r < 1.2 \, \mathrm{GeV}",
          r"3.7 \, \mu \mathrm{barn}   & 1.2 \, \mathrm{GeV}\le  \epsilon_r < 1.65 \, \mathrm{GeV}",
          r"2.7 \, \mu \mathrm{barn}    & \epsilon_r \ge 1.65 \, \mathrm{GeV}"]
H_EQ42 = [r"0 & 2y < 1.0 \, \mathrm{GeV}",
          r"2.0 \, \mu \mathrm{barn}  \, \left(1- \frac{1.0 \, \mathrm{GeV}^2}{(2y)^2} \right) "
          r"& 1.0 \, \mathrm{GeV} \le 2y < 1.2 \, \mathrm{GeV}",
          r"3.7 \, \mu \mathrm{barn}  \, \left(1- \frac{0.3 \, \mathrm{GeV}^2}{(2y)^2} \right) "
          r"& 1.2 \, \mathrm{GeV} \le 2y < 1.65 \, \mathrm{GeV}",
          r"2.7 \, \mu \mathrm{barn}  \, \left(1- \frac{0.16 \, \mathrm{GeV}^2}{(2y)^2} \right)  & 2y \ge 1.65 \, \mathrm{GeV}"]
UNITS = {r"\mu \mathrm{barn}": 1, r"\mathrm{GeV}": 1}   # cross sections in mubarn, energies in GeV


def eq41_steps():
    """[(sigma, lower edge, upper edge)] of eq. 41, read from its rows (sigma = 0 below the first edge)."""
    steps = []
    for row in H_EQ41[1:]:
        value, condition = row.split("&")
        edges = [sp.Rational(v) for v in re.findall(r"\d+\.\d+", condition)]
        steps.append((rational(L.to_sympy(value, UNITS)), edges[0], edges[1] if len(edges) > 1 else sp.oo))
    return steps


def eq42_branches():
    """[(printed f, lower edge of 2y)] of eq. 42."""
    out = []
    for row in H_EQ42[1:]:
        value, condition = row.split("&")
        out.append((rational(L.to_sympy(value, {**UNITS, "y": y})), sp.Rational(re.findall(r"\d+\.\d+", condition)[0])))
    return out


def eq27_fG(arg):
    """Eq. 27's f^IT at arg, its integral written as G(2y) (G(u) = int_eps_th^u eps_r sigma d eps_r)."""
    pre = L.to_sympy(piece(H_EQ27, r"\frac{1}{2y^2}"), {"y": y})
    top = L.to_sympy(piece(H_EQ27, "^{2y}")[2:-1], {"y": y})
    return (pre * G(top)).subs(y, arg)


def eq27_f(steps, branch):
    """Hümmer's eq. 27, f = 1/(2y^2) int_eps_th^2y eps_r sigma, for the step cross section of eq. 41, on the
    branch where 2y lies in step `branch` (the Heaviside functions of the edges up to its lower one are 1)."""
    Y = sp.Symbol("Y", positive=True)
    sig = sum(s * (sp.Heaviside(er - lo) - (sp.Heaviside(er - hi) if hi != sp.oo else 0)) for s, lo, hi in steps)
    G_ = sp.integrate(er * sig, (er, 0, Y))
    f_ = eq27_fG(y).replace(G, lambda a: G_.subs(Y, a))
    return sp.expand(f_.replace(sp.Heaviside, lambda a, *h: 1 if -a.subs(y, 0) <= steps[branch][1] else 0))


# --- the README, read from README.md ---

def RATE():
    return L.display(r"t_{c,N}^{-1}(E_N) = c\int dy")


def readme_y(EN=E):
    return L.to_sympy(L.sides(L.inline(r"y = E_N \epsilon/(m_p c^2)"))[1], {"E_N": EN, r"\epsilon": eps, "m_p": m, "c": c})


def readme_f_parts():
    """The README's f_{c,N}(y) = pre(y) int_lo^up body(eps_r) d eps_r: (pre, lo, up, body)."""
    lhs, rhs = L.sides(L.part(RATE(), "f_{c,N}(y)"))
    pre, integral = rhs.split(r"\int", 1)
    lo, up, body = re.fullmatch(r"_(\w+)\^\{(.+?)\} (.+)\\thinspace d\\epsilon_r", integral).groups()
    assert lhs == "f_{c,N}(y)" and body == r"\epsilon_r\thinspace \sigma_{c,N}(\epsilon_r)", rhs
    return (L.to_sympy(pre, {"y": y}), L.to_sympy(lo, {}), L.to_sympy(up, {"y": y}),
            L.to_sympy(body, {r"\epsilon_r": er}, funcs={r"\sigma_{c,N}": sigma}))


def readme_rate(EN=E):
    """(factor in front, integrand in y) of the README's t_{c,N}^-1(E_N), with f_{c,N}(y) = pre(y) G(up(y)),
    G(u) the integral of eps_r sigma from 0 to u."""
    lhs, rhs = L.sides(L.part(RATE(), r"t_{c,N}^{-1}(E_N)"))
    head, body = rhs.split(r"\int dy\thinspace ")
    pre, lo, up, _ = readme_f_parts()
    assert lhs == r"t_{c,N}^{-1}(E_N)" and lo == 0
    fG = lambda a: (pre * G(up)).subs(y, a)   # noqa: E731
    return (L.to_sympy(head, {"c": c}),
            L.to_sympy(body, {"m_p": m, "c": c, "E_N": EN, "y": y}, funcs={r"n_\gamma": n, "f_{c,N}": fG}))


# --- the kinematics ---

def lab_share(Estar, pLstar, Wv=W, mass=mN):
    """E_K/E_N in the lab for a nucleon along +z with e^rapidity = u (E_N = m_N (u + 1/u)/2), a photon along -z
    giving the invariant mass Wv (eps_r = (Wv^2 - m_N^2)/(2 m_N)), and a kaon of CM energy Estar and momentum
    pLstar along the photon. E +- p_z = e^+-eta (E* +- p*_z), with eta the CM frame's rapidity:
    e^eta = (E + p_z)_total/sqrt(s) = m_N u/Wv."""
    EN_, PN_ = mass * (u + 1 / u) / 2, mass * (u - 1 / u) / 2
    k = (Wv**2 - mass**2) / (2 * mass) / u
    assert sp.simplify((EN_ + k)**2 - (PN_ - k)**2 - Wv**2) == 0
    e_eta = (EN_ + PN_ + k - k) / Wv
    EK_ = (e_eta * (Estar - pLstar) + (Estar + pLstar) / e_eta) / 2     # p*_z = -pLstar
    return EK_ / EN_


def boost(v, p):
    """The 4-vector p = (E, px, py, pz) in the frame moving with the 3-velocity v (mpmath)."""
    v2 = sum(a * a for a in v)
    g, vp = 1 / mp.sqrt(1 - v2), sum(a * b for a, b in zip(v, p[1:]))
    return [g * (p[0] - vp)] + [p[i + 1] + ((g - 1) * vp / v2 - g * p[0]) * v[i] for i in range(3)]


def oblique_share(gN, psi, epsr, pl, pt, phi, mass=mp.mpf("0.93827"), mk=mp.mpf("0.49368")):
    """(E_K/E_N in the lab, sqrt(s)) for a nucleon along z with Lorentz factor gN, a photon at the angle psi to it
    with energy epsr in the nucleon rest frame, and a kaon of CM momentum pl along the photon and pt at the azimuth
    phi around it."""
    EN_, PN_ = gN * mass, mass * mp.sqrt(gN**2 - 1)
    k_ = epsr * mass / (EN_ - PN_ * mp.cos(psi))
    k4 = [k_, k_ * mp.sin(psi), 0, k_ * mp.cos(psi)]
    P = [EN_ + k4[0], k4[1], k4[2], PN_ + k4[3]]
    v = [P[i] / P[0] for i in (1, 2, 3)]
    unit = lambda a: [t / mp.sqrt(sum(s * s for s in a)) for t in a]   # noqa: E731
    nh = unit(boost(v, k4)[1:])                                       # the photon's direction in the CM frame
    e1 = unit([nh[2], 0, -nh[0]])                                     # n x y
    e2 = [nh[1] * e1[2] - nh[2] * e1[1], nh[2] * e1[0] - nh[0] * e1[2], nh[0] * e1[1] - nh[1] * e1[0]]
    pk = [pl * nh[i] + pt * (mp.cos(phi) * e1[i] + mp.sin(phi) * e2[i]) for i in range(3)]
    lab = boost([-a for a in v], [mp.sqrt(mk**2 + pl**2 + pt**2)] + pk)
    return lab[0] / EN_, mp.sqrt(P[0]**2 - P[1]**2 - P[2]**2 - P[3]**2)


# --- the checks: photo-hadronic production ---

@derivation("kp.y")
def check_y():
    """y = E_N eps/(m_p c^2) is Hümmer et al.'s eq. 8; eps_r = y (1 - cos theta_pgamma) ranges over (0, 2y)."""
    ours = readme_y()
    theirs = L.to_sympy(piece(H_EQ8, r"\frac{E_p \, \varepsilon}{m_p}"), {"E_p": E, r"\varepsilon": eps, "m_p": MC2})
    PN_ = sp.sqrt(E**2 - MC2**2)
    epsr = (E * eps - PN_ * eps * mu) / MC2                       # p_N.k/m_N for the angle theta_pgamma
    lim = sp.limit(epsr - ours * (1 - mu), E, sp.oo)
    return [in_source("eq. 8 and the definition of eps_r after eq. 3 as transcribed", HUEMMER, H_EQ8, H_EPSR),
            exact("README y = eq. 8 (E_p -> E_N, m_p -> m_p c^2)", ours - theirs),
            Result("eps_r = p_N.k/m_N -> y (1 - cos theta_pgamma) for beta_N -> 1 (Hümmer's eps_r; at most 2y)",
                   lim == 0, f"limit of the difference: {lim}")]


@derivation("kp.rate")
def check_rate():
    """The README's t_{c,N}^-1 is Hümmer et al.'s eq. 4 (eq. 3 for isotropic photons) with f of eq. 27."""
    y_of_eps = readme_y()
    pre, lo, up, body = readme_f_parts()
    # eq. 3: the angle average of (1 - cos) sigma(eps_r), with eps_r = E_p eps/m_p (1 - cos) (text after eq. 3)
    epsr = L.to_sympy(H_EPSR.strip("$").split("=")[1], {"E_p": E, r"\varepsilon": eps, "m_p": MC2, r"\cos_{p \gamma}": mu})
    avg = sp.Integral((1 - mu) / 2 * sigma(epsr), (mu, -1, 1)).transform(mu, (sp.solve(epsr - er, mu)[0], er))
    ours = (pre * sp.Integral(body, (er, lo, up))).subs(y, y_of_eps)
    (a, la), (b, lb) = integrand(avg), integrand(ours)
    out = [in_source("eqs. 3 and 4, isotropy, eps_r and the threshold as transcribed", HUEMMER, H_EQ3, H_EPSR, H_ISO,
                     H_EQ4, H_TH),
           in_source("eq. 27 as transcribed", HUEMMER, H_EQ27),
           Result("eq. 3, isotropic photons: (1/2) int dcos (1 - cos) sigma = README f_{c,N}(y), y = E_N eps/(m_p c^2) "
                  "(limits)", la == lb, f"{la} and {lb}"),
           exact("eq. 3, isotropic photons: the angle average = README f_{c,N} (integrands)", a - b)]
    # eq. 4 with G(u) = int_eps_th^u eps_r sigma = int_0^u (sigma = 0 below eps_th), then eps = m_p c^2 y/E_N
    e4 = (c * L.to_sympy(piece(H_EQ4, r"\frac{1}{2} \frac{m_p^2}{E_p^2}"), {"m_p": MC2, "E_p": E})
          * sp.Integral(L.to_sympy(piece(H_EQ4, r"\frac{n_\gamma(\varepsilon)}{\varepsilon^2}"), {r"\varepsilon": eps},
                                   funcs={r"n_\gamma": n})
                        * G(L.to_sympy(piece(H_EQ4, r"2 E_p \varepsilon/m_p"), {"E_p": E, r"\varepsilon": eps, "m_p": MC2})),
                        (eps, L.to_sympy(piece(H_EQ4, r"\frac{\epsilon_{\mathrm{th}} m_p}{2 E_p}"),
                                         {r"\epsilon_{\mathrm{th}}": eth, "m_p": MC2, "E_p": E}), sp.oo)))
    in_y = e4.replace(lambda a: isinstance(a, sp.Integral), lambda a: a.transform(eps, (sp.solve(y_of_eps - y, eps)[0], y)))
    (h, lim), (head, body_y) = integrand(in_y), readme_rate()
    out += [exact("eq. 4 at eps = m_p c^2 y/E_N (times c) = README t^-1 integrand", h - head * body_y),
            Result("eq. 4's lower limit eps_th m_p/(2E_p) is y = eps_th/2, below which f = 0 (README: all y)",
                   lim[1] == eth / 2, f"y from {lim[1]}")]
    out.append(exact("README f_{c,N}(y) = eq. 27's f^IT(y) (lower limit 0 for eps_th: sigma = 0 below it)",
                     pre * G(up) - eq27_fG(y)))
    with L.patched(r"\frac{m_p c^2}{E_N}\thinspace n_\gamma", r"\frac{E_N}{m_p c^2}\thinspace n_\gamma"):
        head_b, body_b = readme_rate()
    out.append(exact_fails("control: E_N/(m_p c^2) for m_p c^2/E_N in front of n_gamma", h - head_b * body_b))
    return out


@derivation("kp.source")
def check_source():
    """Q_K is Hümmer et al.'s eq. 28, which follows from their eqs. 7 and 27, summed over channels and nucleons."""
    src = L.parts(L.display(r"Q_K(E_K) = \sum_c \sum_{N=p,n}"))[0]
    lhs, rhs = L.sides(src)
    assert lhs == "Q_K(E_K)" and rhs == (r"\sum_c \sum_{N=p,n} \frac{M_{c,N}^K}{\chi_{c,N}^K}\thinspace \Big[N_N\thinspace "
                                         r"t_{c,N}^{-1}\Big]_ {E_N = E_K/\chi_{c,N}^K}"), rhs
    tok = {"M_{c,N}^K": M, r"\chi_{c,N}^K": chi, "E_K": EK}
    factor = L.to_sympy(L.between(rhs, r"\sum_{N=p,n} ", r"\thinspace \Big["), tok)
    at = L.sides(rhs.split(r"\Big]_ {")[1][:-1])
    assert at[0] == "E_N"
    EN_at = L.to_sympy(at[1], tok)
    nn, tt = sp.symbols("N_N t_inv")
    bracket = L.to_sympy(L.between(rhs, r"\Big[", r"\Big]"), {"N_N": nn, "t_{c,N}^{-1}": tt})
    head, body = readme_rate(EN_at)
    ours = sp.expand(factor * bracket.subs({nn: N(EN_at), tt: head}) * body)       # README term, integrand in y
    # eq. 28 (c = 1 there: times c), parsed
    tok28 = {"E_b": EK, r"\chi^{\mathrm{IT}}": chi, "m_p": MC2, "y": y, r"M_b^{\mathrm{IT}}": M}
    head28 = L.to_sympy(piece(H_EQ28, r"N_p \left( \frac{E_b}{\chi^{\mathrm{IT}}} \right) \frac{m_p}{E_b}"), tok28,
                        funcs={"N_p": N})
    body28 = L.to_sympy(piece(H_EQ28, r"n_\gamma \left( \frac{m_p \, y \, \chi^{\mathrm{IT}}}{E_b} \right) \,  "
                                      r"M_b^{\mathrm{IT}}  \, f^{\mathrm{IT}}(y)"), tok28,
                        funcs={r"n_\gamma": n, r"f^{\mathrm{IT}}": eq27_fG})
    # eqs. 7 and 27: Q = int dE_p/E_p N_p int d eps n R, R = delta(x - chi) M f(y), x and y of eq. 8. Integrals over
    # all E_p (the delta puts E_p = E_b/chi > E_b) and over eps from 0 (f = 0 below eps_th).
    Ep = sp.Symbol("E_p", positive=True)
    xv = L.to_sympy(piece(H_EQ8, r"\frac{E_b}{E_p}"), {"E_b": EK, "E_p": Ep})
    yv = L.to_sympy(piece(H_EQ8, r"\frac{E_p \, \varepsilon}{m_p}"), {"E_p": Ep, r"\varepsilon": eps, "m_p": MC2})
    over_Ep = sp.Integral(N(Ep) / Ep * sp.DiracDelta(xv - chi) * M * eq27_fG(yv), (Ep, 0, sp.oo))
    done = over_Ep.transform(Ep, (EK / x, x)).doit()
    q7 = sp.Integral(n(eps) * done, (eps, 0, sp.oo)).transform(eps, (sp.solve(yv.subs(Ep, EK / chi) - y, eps)[0], y))
    from7, _ = integrand(q7)
    with L.patched(r"E_N = E_K/\chi_{c,N}^K}", r"E_N = E_K\chi_{c,N}^K}"):
        rhs_b = L.sides(L.parts(L.display(r"Q_K(E_K) = \sum_c \sum_{N=p,n}"))[0])[1]
        EN_b = L.to_sympy(L.sides(rhs_b.split(r"\Big]_ {")[1][:-1])[1], tok)
        head_b, body_b = readme_rate(EN_b)
        broken = sp.expand(factor * bracket.subs({nn: N(EN_b), tt: head_b}) * body_b)
    return [in_source("eqs. 7, 8, 27 and 28 as transcribed", HUEMMER, H_EQ7, H_EQ8, H_EQ27, H_EQ28),
            exact("eqs. 7 and 27 (the delta integrated over E_p, eps -> y) give eq. 28 (integrands)",
                  from7 - head28 * body28),
            exact("README (M/chi) [N_N t^-1] at E_N = E_K/chi = c x eq. 28 (N_p -> N_N, IT -> channel c)",
                  ours - c * head28 * body28),
            exact_fails("control: E_N = E_K chi in the README", broken - c * head28 * body28)]


E1, E2, SC = sp.symbols("epsilon_min epsilon_max sigma_c", positive=True)


def readme_fc(branch_min):
    """The README's closed form, its min(...) replaced by branch_min(its two arguments)."""
    lhs, rhs = L.sides(L.display(r"f_c(y) = \sigma_c [\min(4y^2"))
    assert lhs == "f_c(y)"
    args_tex = L.between(rhs, r"\min(", ")")
    tok = {"y": y, r"\epsilon_\mathrm{max}": E2, r"\epsilon_\mathrm{min}": E1, r"\sigma_c": SC}
    args = [L.to_sympy(a, tok) for a in args_tex.split(", ")]
    return L.to_sympy(rhs, {**tok, r"\min(" + args_tex + ")": branch_min(args)})


@derivation("kp.fc")
def check_fc():
    """f_c = sigma_c [min(4y^2, eps_max^2) - eps_min^2]/(4y^2) is eq. 27 for a constant cross section, 2y >= eps_min."""
    L.inline(r"2y \ge \epsilon_\mathrm{min}")
    pre, lo, up, body = readme_f_parts()
    d, D, Y = sp.symbols("d D Y", positive=True)
    sig = SC * (sp.Heaviside(er - E1) - sp.Heaviside(er - E2))           # sigma_c from eps_min to eps_max
    integral = pre * sp.integrate(body.subs(sigma(er), sig), (er, lo, Y)).subs(Y, up)
    out = []
    # each branch as an ordering of positive quantities: d, D > 0
    for name, which, at in (("2y < eps_min", None, {E1: 2 * y + d, E2: 2 * y + d + D}),
                            ("eps_min <= 2y < eps_max", 0, {y: (E1 + d) / 2, E2: E1 + d + D}),
                            ("2y >= eps_max", 1, {y: (E1 + d + D) / 2, E2: E1 + d})):
        def smaller(args):
            if which is not None:   # the branch's min, by its ordering
                assert sp.expand((args[1 - which] - args[which]).subs(at)).is_positive
            return args[which or 0]
        closed, value = readme_fc(smaller).subs(at), integral.subs(at)
        if which is None:
            neg = sp.factor(closed)
            out.append(Result(f"{name}: eq. 27 gives 0, the closed form is negative there (hence 'for 2y >= eps_min')",
                              value == 0 and neg.is_negative, f"integral {value}, closed form {neg}"))
        else:
            out.append(exact(f"{name}: eq. 27 = closed form with min = {['4y^2', 'eps_max^2'][which]}", value - closed))
    with L.patched(r"- \epsilon_\mathrm{min}^2]/(4y^2)", r"- \epsilon_\mathrm{min}^2]/(2y^2)"):
        broken = readme_fc(lambda args: args[0]).subs({y: (E1 + d) / 2, E2: E1 + d + D})
    out.append(exact_fails("control: /(2y^2) in the closed form", integral.subs({y: (E1 + d) / 2, E2: E1 + d + D}) - broken))
    L.line(r"On AM³'s logarithmic grids $f_c$ depends only on the sum of the nucleon and photon bin indices", "checks")
    i, j, k, r = sp.symbols("i j k r", positive=True)
    y_ij = lambda a, b: readme_y(E * r**a).subs(eps, eps * r**b)   # noqa: E731  (E_N = E r^i, eps = eps r^j)
    out.append(exact("y (so f_c) on logarithmic grids depends on i + j only: y(i + k, j - k) = y(i, j)",
                     y_ij(i + k, j - k) - y_ij(i, j)))
    return out


@derivation("kp.eq42")
def check_eq42():
    """Hümmer et al.'s eq. 42 follows from eqs. 27 and 41 only in its first step: 1.20 and 0.64 GeV^2, not 0.3, 0.16."""
    steps, printed = eq41_steps(), eq42_branches()
    out = [in_source("eqs. 41 and 42 as transcribed", HUEMMER, r" \\ ".join(H_EQ41) + r" \\", r" \\ ".join(H_EQ42) + r" \\"),
           in_source("eq. 27 as transcribed", HUEMMER, H_EQ27),
           Result("eq. 42's steps start where eq. 41's do", [lo for _, lo in printed] == [lo for _, lo, _ in steps],
                  f"{[float(lo) for _, lo in printed]}")]
    exact_f = [eq27_f(steps, k) for k in range(3)]
    consts = [sp.nsimplify(sp.expand((2 * y)**2 * (1 - fk / steps[k][0]))) for k, fk in enumerate(exact_f)]
    for k, (fk, (pk, lo)) in enumerate(zip(exact_f, printed)):
        name = f"step {k + 1} (2y from {float(lo)} GeV): eq. 27 gives {float(steps[k][0])} [1 - {float(consts[k]):.4f}/(2y)^2]"
        if k == 0:
            out.append(exact(name + " = eq. 42 as printed", fk - pk))
        else:
            out.append(exact_fails(name + f", eq. 42 prints {float(sp.expand((2 * y)**2 * (1 - pk / steps[k][0]))):g}", fk - pk))
    out.append(Result("eq. 27's f is sigma_k [1 - C_k/(2y)^2] on each step, C_k independent of y",
                      all(cst.is_number for cst in consts), f"C = {consts}"))
    cont, jump = [], []
    for k in (1, 2):
        edge = steps[k][1] / 2
        cont.append(sp.simplify(exact_f[k - 1].subs(y, edge) - exact_f[k].subs(y, edge)))
        jump.append(float(printed[k - 1][0].subs(y, edge) - printed[k][0].subs(y, edge)))
    out.append(Result("eq. 27's f is continuous at 2y = 1.2 and 1.65 GeV; eq. 42 as printed jumps there",
                      cont == [0, 0] and all(abs(j) > 0.1 for j in jump),
                      f"jumps of eq. 42: {jump[0]:.3f}, {jump[1]:.3f} mubarn"))

    def readme_constants():
        line = L.line(r"Eq. 27 gives 3.7 μb $", "checks")
        mt = re.search(r"Eq\. 27 gives ([\d.]+) μb \$(.+?)\$ and ([\d.]+) μb \$(.+?)\$, against the printed ([\d.]+) and "
                       r"([\d.]+) GeV²", line)
        amp = [sp.Rational(mt.group(i)) for i in (1, 3)]
        brk = [rational(L.to_sympy(mt.group(i), {r"\mathrm{GeV}": 1, "y": y})) for i in (2, 4)]
        return amp, [sp.expand((2 * y)**2 * (1 - b)) for b in brk], [sp.Rational(mt.group(i)) for i in (5, 6)]
    amp, ours, said = readme_constants()

    def rounded(amp_, ours_):   # the two decimals of Results are eq. 27's constants, rounded
        return (amp_ == [steps[1][0], steps[2][0]]
                and all(abs(o - consts[k]) <= sp.Rational(1, 200) for o, k in zip(ours_, (1, 2))))
    ok = rounded(amp, ours)
    out.append(Result("Results 3.7 mubarn [1 - 1.20/(2y)^2] and 2.7 mubarn [1 - 0.64/(2y)^2]: eq. 27's constants rounded",
                      ok, f"Results {[float(v) for v in ours]}, eq. 27 {[round(float(consts[k]), 4) for k in (1, 2)]}"))
    printed_c = [sp.expand((2 * y)**2 * (1 - printed[k][0] / steps[k][0])) for k in (1, 2)]
    out.append(Result("Results' 'printed 0.3 and 0.16 GeV^2' = eq. 42's constants", said == printed_c, f"{said}"))
    with L.patched(r"[1 - 1.20\ \mathrm{GeV}^2/(2y)^2]", r"[1 - 1.30\ \mathrm{GeV}^2/(2y)^2]", "checks"):
        amp_b, ours_b, _ = readme_constants()
    out.append(Result("control: 1.30 for 1.20 in Results", not rounded(amp_b, ours_b), "fails, as it must"))
    # Results: the values quoted from eq. 41, and the '17 % more int sigma d eps_r over 0.91-1.65 GeV' of the steps
    # of the README's table (against eq. 41)
    line = L.line(r"their constant 2.7 μb to the highest energies", "checks")
    q = re.search(r"\(([\d.]+) μb at \$([\d.]+)\\text\{–\}([\d.]+)\$ GeV and "
                  r"([\d.]+) μb at \$([\d.]+)\\text\{–\}([\d.]+)\$ GeV, their eqs\. 41–42\)", line)
    quoted = [tuple(sp.Rational(q.group(i)) for i in (1, 2, 3)), tuple(sp.Rational(q.group(i)) for i in (4, 5, 6))]
    last = sp.Rational(re.search(r"their constant ([\d.]+) μb to the highest energies", line).group(1))
    out.append(Result("Results quotes eq. 41: 2.0 mubarn at 1.0-1.2, 3.7 at 1.2-1.65, 2.7 above",
                      quoted == [steps[0], steps[1]] and (last, sp.oo) == (steps[2][0], steps[2][2]), f"{quoted}, {last}"))
    pct, a, b = re.search(r"\$(\d+)\\text\{％\}\$ more \$\\int\\sigma\\thinspace d\\epsilon_r\$ "
                          r"over \$([\d.]+)\\text\{–\}([\d.]+)\$ GeV", line).groups()
    table = 0
    for row in (r"| $0.91\text{–}1.05$ |", r"| $1.05\text{–}1.25$ |", r"| $1.25\text{–}1.65$ |"):
        lo_, hi_, s_ = re.match(r"\| \$([\d.]+)\\text\{–\}([\d.]+)\$ \| ([\d.]+) μb", L.line(row)).groups()
        table += sp.Rational(s_) * (sp.Rational(hi_) - sp.Rational(lo_))
    hum = sum(s_ * (min(hi_, sp.Rational(b)) - max(lo_, sp.Rational(a))) for s_, lo_, hi_ in steps if lo_ < sp.Rational(b))
    ratio = table / hum
    out.append(Result(f"Results: the README's p -> K+ steps have {pct} % more int sigma d eps_r over {a}-{b} GeV than eq. 41",
                      round(100 * (ratio - 1)) == int(pct),
                      f"{float(table):.4f} / {float(hum):.4f} mubarn GeV = {float(ratio):.4f}"))
    return out


CHI_PIN = r"\chi = \frac{s - m_Y^2 + m_K^2}{2s}"


def readme_s():
    """The README's s = m_N^2 + 2 m_N eps_r."""
    return L.to_sympy(L.sides(L.part(L.display(CHI_PIN), "s ="))[1], {"m_N": mN, r"\epsilon_r": er})


@lru_cache(None)
def numeric_s():
    """readme_s as a numpy function of (m_N, eps_r)."""
    return sp.lambdify((mN, er), readme_s(), "numpy")


def readme_chi(s_, beta, angle):
    """The README's chi for s, beta* and the angle theta."""
    return L.to_sympy(L.sides(L.part(L.display(CHI_PIN), r"\chi ="))[1],
                      {"s": s_, "m_Y": mY, "m_K": mK, r"\beta^\ast": beta, r"\theta": angle})


def two_body_Estar():
    """The kaon's E* of two-body kinematics: E*_K + E*_Y = sqrt(s) = W, p*_K = p*_Y."""
    EKs, EYs = sp.symbols("E_K_star E_Y_star")
    return sp.solve([EKs + EYs - W, EKs**2 - mK**2 - (EYs**2 - mY**2)], [EKs, EYs], dict=True)[0][EKs]


@lru_cache(None)
def numeric_chi():
    """The README's chi with beta* = p*/E* of two-body kinematics, as a numpy function of (m_N, m_Y, m_K, eps_r,
    <cos theta>); chi is linear in cos theta (kp.chi). tests/support/measured/ky_angular.py computes the code's chi
    with it."""
    Estar_ = two_body_Estar()
    chi = readme_chi(readme_s(), sp.sqrt(Estar_**2 - mK**2) / Estar_, sp.acos(ct)).subs(W, sp.sqrt(readme_s()))
    return sp.lambdify((mN, mY, mK, er, ct), chi, "numpy")


@derivation("kp.chi")
def check_chi():
    """chi = (s - m_Y^2 + m_K^2)/(2s) (1 - beta* cos theta) is two-body kinematics and Hümmer et al.'s eq. 16."""
    s_ours, ours = readme_s(), readme_chi
    th = sp.Symbol("theta", positive=True)
    eq16 = lambda angle: L.to_sympy(piece(H_EQ16, H_EQ16.split("=", 1)[1].strip()),   # noqa: E731
                                    {r"s(\epsilon_r)": W**2, "m_p": mY, r"m_\pi": mK, r"\beta_\pi^{\mathrm{cm}}": ps / Es,
                                     r"\theta_\pi": angle})
    eq5 = L.to_sympy(H_EQ5.split("=")[1], {"m_p": mN, r"\epsilon_r": er})
    # eq. 16 from eqs. 14 and 15 (Hümmer's step): E_pi/E_p with E*_pi of eq. 15
    E15 = L.to_sympy(H_EQ15.split("=")[1], {"s": W**2, "m_p": mY, r"m_\pi": mK})
    E14 = L.to_sympy(H_EQ14.split("=")[2], {"E_p": E, r"E_\pi^\text{cm}": E15, "s": W**2,
                                            r"\beta_\pi^\text{cm}": ps / Es, r"\theta_\pi^\text{cm}": sp.pi - th})
    # two-body kinematics: E* from E*_K + E*_Y = sqrt(s), p*_K = p*_Y; s = (p_N + k)^2 in the lab
    Estar = two_body_Estar()
    lab = sp.limit(lab_share(Estar, ps * ct), u, sp.oo)
    k_ = er / u                                                   # photon energy for eps_r = p_N.k/m_N = k u
    s_kin = (mN * (u + 1 / u) / 2 + k_)**2 - (mN * (u - 1 / u) / 2 - k_)**2
    chi_kin = ours(W**2, ps / Estar, sp.acos(ct))
    with L.patched(r"\big(1 - \beta^\ast \cos\theta\big)", r"\big(1 + \beta^\ast \cos\theta\big)"):
        broken = ours(W**2, ps / Estar, sp.acos(ct))
    # Hümmer's chi_K+ = 0.35 at eps_r = 1.4 GeV, cos theta = 0, with m_Lambda and m_K+ (GeV, PDG)
    m_p, m_L, m_Kp = sp.Float("0.93827"), sp.Float("1.11568"), sp.Float("0.49368")
    chi35 = ours(eq5.subs({mN: m_p, er: sp.Rational(14, 10)}), 0, sp.pi / 2).subs({mY: m_L, mK: m_Kp})
    d2 = sp.diff(ours(W**2, ps / Es, sp.acos(ct)), ct, 2)
    return [in_source("eqs. 5, 14-16, the angle of eq. 14 and the replacements for kaons as transcribed", HUEMMER,
                      H_EQ5, H_EQ14, H_BACK, H_EQ15, H_EQ16, H_KAON, H_CHI35),
            exact("README s = eq. 5 (m_p -> m_N)", s_ours - eq5),
            exact("README s = (p_N + k)^2, eps_r = p_N.k/m_N", s_ours - s_kin),
            exact("README chi = eq. 16 (m_p -> m_Y, m_pi -> m_K, theta_pi = pi - theta)",
                  ours(W**2, ps / Es, th) - eq16(sp.pi - th)),
            exact_fails("eq. 16 with theta_pi = theta differs (Hümmer's angle is from the nucleon)",
                        ours(W**2, ps / Es, th) - eq16(th)),
            exact("eq. 16 = eq. 14 / E_p with E*_pi of eq. 15 (Hümmer's step)", eq16(sp.pi - th) - E14 / E),
            exact("two-body kinematics: lim E_K/E_N (E_N -> oo, s fixed) = README chi, beta* = p*/E*", lab - chi_kin),
            exact_fails("control: (1 + beta* cos theta) in the README", lab - broken),
            exact("chi is linear in cos theta (its mean is chi(<cos theta>))", d2),
            Result("Hümmer's chi_K+ = 0.35 at eps_r = 1.4 GeV (cos theta = 0, m_Lambda, m_K+)", abs(chi35 - 0.35) < 0.01,
                   f"eq. 16 gives {float(chi35):.4f}")]


KSTAR_DECAY = "The CBELSA/TAPS cross section is counted together with the "


def readme_kstar_k0pi0():
    """The share of the K*0 that decays into K0 pi0, as the Results state it ("the third"); multibody_strangeness.py
    uses it."""
    word = re.search(r"together with the (\w+) of their \$\\mathrm\{K\}\^\{\\ast 0\}\\Sigma\^\+\$ that decays into "
                     r"\$\\mathrm\{K\}\^0\\pi\^0\$", L.line(KSTAR_DECAY, "checks")).group(1)
    return {"half": sp.Rational(1, 2), "third": sp.Rational(1, 3), "quarter": sp.Rational(1, 4)}[word]


@derivation("kp.kstar_decay")
def check_kstar_decay():
    """A third of the K*0 decays into K0 pi0, two thirds into K+ pi- (isospin; K pi is all of its decays)."""
    h = sp.Rational(1, 2)
    k0pi0, kppim = CG(h, -h, 1, 0, h, -h).doit()**2, CG(h, h, 1, -1, h, -h).doit()**2   # <K pi | K*0 (I3 = -1/2)>^2
    with L.patched("together with the third of their", "together with the half of their", "checks"):
        broken = readme_kstar_k0pi0()
    return [exact("K0 pi0 share = <1/2,-1/2;1,0|1/2,-1/2>^2 = the Results' third", k0pi0 - readme_kstar_k0pi0()),
            exact("the two charge states add up to one", k0pi0 + kppim - 1),
            exact_fails("control: half in the Results", k0pi0 - broken)]


KSTAR = r"Pure $\mathrm{K}^\ast$ exchange would make it "


def readme_kstar_factor():
    """The factor of K0 Sigma0 against K0 Sigma+ with pure K* exchange, as the README states it (two_body.py uses it)."""
    return sp.Rational(re.search(r"Pure \$\\mathrm\{K\}\^\\ast\$ exchange would make it ([\d.]+) at high energies",
                                 L.line(KSTAR)).group(1))


@derivation("kp.kstar")
def check_kstar():
    """With pure K* exchange, gamma n -> K0 Sigma0 has half the cross section of gamma p -> K0 Sigma+ (isospin)."""
    # in both reactions the photon makes a K0 and a K-bar*0 (I = 1/2, I3 = +1/2) at the same vertex. The K-bar*0 and the
    # nucleon (I = 1/2) make the Sigma (I = 1), with the amplitude <1/2, I3_N; 1/2, 1/2 | 1, I3_Sigma>.
    h = sp.Rational(1, 2)
    ratio = (CG(h, -h, h, h, 1, 0).doit() / CG(h, h, h, h, 1, 1).doit())**2     # n -> Sigma0 over p -> Sigma+
    with L.patched("exchange would make it 0.5 at", "exchange would make it 0.25 at"):
        broken = readme_kstar_factor()
    return [exact("sigma(K0 Sigma0)/sigma(K0 Sigma+) = |<1/2,-1/2;1/2,1/2|1,0> / <1/2,1/2;1/2,1/2|1,1>|^2 = the README's "
                  "factor", ratio - readme_kstar_factor()),
            exact_fails("control: 0.25 in the README", ratio - broken)]


TOK_M4 = {r"E^\ast": Es, r"p^\ast_L": pL, "s": W**2}


def readme_chi_M4():
    """The Results' chi = (E* - p*_L)/sqrt(s) of the kaons of M4."""
    span = re.search(r"\$(\\chi = .+?)\$", L.line(r"A kaon's share of the nucleon energy is $\chi = ", "checks")).group(1)
    return L.to_sympy(L.sides(span)[1], TOK_M4)


def readme_x_M4():
    """The Results' x = 2 p*_L/sqrt(s)."""
    return L.to_sympy(L.sides(L.inline(r"x = 2p^\ast_L/\sqrt{s}", "checks"))[1], TOK_M4)


def chi_of_x(mT, xr):
    """readme_chi_M4 with E* = sqrt(m_T^2 + p*_L^2) and p*_L from readme_x_M4 = xr."""
    return readme_chi_M4().subs(Es, sp.sqrt(mT**2 + pL**2)).subs(pL, sp.solve(readme_x_M4() - xr, pL)[0])


@lru_cache(None)
def numeric_chi_M4():
    """chi_of_x as a numpy function of (x, m_T, sqrt(s)) (kp.chi_M4); tests/support/measured/strange_20gev.py uses it."""
    mT, xr = sp.Symbol("m_T", positive=True), sp.Symbol("x", real=True)
    return sp.lambdify((xr, mT, W), chi_of_x(mT, xr), "numpy")


@derivation("kp.chi_M4")
def check_chi_M4():
    """chi = (E* - p*_L)/sqrt(s), the lab limit of E_K/E_N, is [sqrt(x^2 + 4 m_T^2/s) - x]/2 with x = 2p*_L/sqrt(s)."""
    ours = readme_chi_M4
    lab = sp.limit(lab_share(Es, pL), u, sp.oo)
    with L.patched(r"share of the nucleon energy is $\chi = (E^\ast - p^\ast_L)/\sqrt{s}$",
                   r"share of the nucleon energy is $\chi = (E^\ast + p^\ast_L)/\sqrt{s}$", "checks"):
        broken = ours()
    # an oblique photon: (angle to the nucleon, eps_r [GeV], p*_L, p_T, azimuth), momenta up to ~sqrt(s)/2
    r, mk = rng(7), mp.mpf("0.49368")
    pts = []
    for _ in range(12):
        epsr = log_uniform(r, 1.2, 50)
        half = mp.sqrt(mp.mpf("0.93827")**2 + 2 * mp.mpf("0.93827") * epsr) / 2
        pts.append((mp.mpf(r.uniform(0.05, 3.1)), epsr, half * mp.mpf(r.uniform(-0.9, 0.9)), half * mp.mpf(r.uniform(0, 0.4)),
                    mp.mpf(r.uniform(0, 6.28))))
    gN = mp.mpf(10)**20
    chi_f = sp.lambdify((Es, pL, W), ours(), "mpmath")
    readme_at = lambda psi, epsr, pl, pt, phi: chi_f(mp.sqrt(mk**2 + pl**2 + pt**2), pl,   # noqa: E731
                                                     oblique_share(gN, psi, epsr, pl, pt, phi)[1])
    # x and p_T: E* = sqrt(m_T^2 + p*_L^2) with p*_L from the README's x
    mT, xr = sp.Symbol("m_T", positive=True), sp.Symbol("x", real=True)
    chi_x = chi_of_x(mT, xr)
    # the nucleon rest frame: the boost along the photon's axis with gamma = E*_N/m_N, gamma v = p_L/m_N (the
    # nucleon's CM energy and momentum along the photon) stops the nucleon
    ENs, pNs = (W**2 + mN**2) / (2 * W), -(W**2 - mN**2) / (2 * W)
    to_rest = lambda E_, p_: ((ENs * E_ - pNs * p_) / mN, (ENs * p_ - pNs * E_) / mN)   # noqa: E731
    nucleon, kaon = to_rest(ENs, pNs), to_rest(Es, pL)
    return [exact("lim E_K/E_N (head-on photon, E_N -> oo at fixed s) = README chi, any kaon momentum", lab - ours()),
            exact_fails("control: (E* + p*_L) in the README", lab - broken),
            numeric("oblique photon (0.05-3.1 rad from the nucleon), kaon with p_T: E_K/E_N at gamma_N = 1e20",
                    lambda *a: oblique_share(gN, *a)[0], readme_at, pts, dps=100, tol=mp.mpf(10)**-15),
            control("control: at gamma_N = 1e10 the oblique E_K/E_N still differs (the limit is approached as "
                    "1/gamma_N)", lambda *a: oblique_share(mp.mpf(10)**10, *a)[0], readme_at, pts, dps=100,
                    tol=mp.mpf(10)**-15),
            exact("with x = 2p*_L/sqrt(s): chi = [sqrt(x^2 + 4 m_T^2/s) - x]/2 (m_T^2 = m_K^2 + p_T^2; any sign of x)",
                  chi_x - (sp.sqrt(xr**2 + 4 * mT**2 / W**2) - xr) / 2),
            Result("the boost along the photon's axis brings the nucleon to rest", sp.simplify(nucleon[0] - mN) == 0
                   and sp.simplify(nucleon[1]) == 0, f"nucleon: {sp.simplify(nucleon[0])}, {sp.simplify(nucleon[1])}"),
            exact("in the nucleon rest frame chi = (E - p_L)/m_N (the form used with Abe et al.'s z = E_K/E_gamma)",
                  ours() - (kaon[0] - kaon[1]) / mN)]


# --- the checks: the thresholds of photo-hadronic kaon production ---

SAME_MASS = {"K-": "K+", "K0b": "K0", "K0S": "K0", "pi-": "pi+"}   # charge conjugates, and K0_S
MASS = {q: float(sp.Rational(MASS_MEV[SAME_MASS.get(q, q)]) / 1000)
        for q in ("p", "n", "K+", "K-", "K0", "K0b", "K0S", "pi0", "pi+", "pi-", "Lambda", "Sigma+", "Sigma0", "Sigma-",
                  "Sigma*+")}   # GeV, PDG 2026
CHARGE = {"p": 1, "n": 0, "K+": 1, "K-": -1, "K0": 0, "K0b": 0, "pi0": 0, "pi+": 1, "pi-": -1, "Lambda": 0,
          "Sigma+": 1, "Sigma0": 0, "Sigma-": -1, "Sigma*+": 1}
STRANGENESS = {"K+": 1, "K0": 1, "K-": -1, "K0b": -1, "Lambda": -1, "Sigma+": -1, "Sigma0": -1, "Sigma-": -1}
KAON, ANTIKAON, PION, SIGMA, NUCLEON = ("K+", "K0"), ("K-", "K0b"), ("pi+", "pi0", "pi-"), ("Sigma+", "Sigma0", "Sigma-"), ("p", "n")


def final_states(charge, *groups):
    """The final states with one particle of each group and the total charge `charge`."""
    return [c for c in itertools.product(*groups) if sum(CHARGE[q] for q in c) == charge]


def rounds_to(value, text):
    """value, rounded to the decimals of the number `text` (as the README prints it), is that number."""
    return round(value, len(text.split(".")[1]) if "." in text else 0) == float(text)


@lru_cache(None)
def photon_threshold():
    """eps_r(m_N, M) [GeV] at which the README's s = m_N^2 + 2 m_N eps_r reaches M^2. Also used by
    tests/support/measured/ky_angular.py, tools/ky_angular_extract.py, multibody_strangeness.py and m4_steps.py."""
    total = sp.Symbol("M", positive=True)
    return sp.lambdify((mN, total), sp.solve(readme_s() - total**2, er)[0])


def threshold(m_target, *masses):
    """eps_r [GeV] at which the README's s on a target of mass m_target reaches the squared sum of these masses."""
    return float(photon_threshold()(m_target, sum(masses)))


def available_energy(m_a, m_b, E_a):
    """The Results' available energy E_A = sqrt(s) - m_a - m_b of a beam of mass m_a and energy E_a on a target of mass
    m_b at rest, with s = (p_a + p_b)^2 = m_a^2 + m_b^2 + 2 m_b E_a (the README's s for a photon, m_a = 0)."""
    tex = L.sides(L.inline(r"E_A = \sqrt{s} - m_a - m_b", "checks"))[1]
    return float(L.to_sympy(tex, {"s": m_a**2 + m_b**2 + 2 * m_b * E_a, "m_a": m_a, "m_b": m_b}))


def threshold_span(targets, *groups):
    """(lowest, highest threshold [GeV], number of charge states) of the final states with one particle of each group,
    on these targets (PDG 2026 masses)."""
    v = [readme_threshold()(t, c) for t in targets for c in final_states(CHARGE[t], *groups)]
    return min(v), max(v), len(v)


def readme_threshold():
    """eps_r(target, final state) [GeV] at which the README's s reaches the squared sum of the final masses."""
    f = photon_threshold()
    return lambda target, final: f(MASS[target], sum(MASS[q] for q in final))


@derivation("kp.thresholds")
def check_thresholds():
    """The thresholds of K Lambda pi, K Sigma pi, N K Kbar (pi) and Sigma(1385) K: the README's s, PDG 2026 masses."""
    th, span = readme_threshold(), threshold_span
    klp, ksp = span(NUCLEON, KAON, ("Lambda",), PION), span(NUCLEON, KAON, SIGMA, PION)
    pair, pair_pi = span(NUCLEON, NUCLEON, KAON, ANTIKAON), span(("p",), NUCLEON, KAON, ANTIKAON, PION)
    y1385 = th("p", ("K0", "Sigma*+"))

    def stated():
        m2 = L.line("A kaon can be made with a third particle only above", "checks")
        a = re.search(r"only above ([\d.]+) GeV\. The threshold of \$\\gamma\\mathrm\{N\} \\to \\mathrm\{K\}\\Lambda\\pi\$ "
                      r"is \$([\d.]+)\\text\{–\}([\d.]+)\$ GeV for the four charge states, and \$\\mathrm\{K\}\\Sigma\\pi\$ "
                      r"starts at ([\d.]+) GeV\. An antikaon comes only with a kaon pair, from ([\d.]+) GeV on", m2).groups()
        b = re.search(r"only above ([\d.]+) GeV, the threshold of \$\\Sigma\(1385\)\\mathrm\{K\}\$", m2).group(1)
        c = re.search(r"those with one start at ([\d.]+) GeV", L.line("those with one start at", "checks")).group(1)
        d = re.search(r"M2 makes kaons only above ([\d.]+) GeV, the threshold of \$\\gamma\\mathrm\{N\} \\to "
                      r"\\mathrm\{K\}\\Lambda\\pi\$", L.line("M2 makes kaons only above")).group(1)
        return (*a, b, c, d)

    def compare(st):
        third, lo, hi, ks, pr, y, pr_pi, readme = st
        return [Result(f"a kaon with a third particle from {third} GeV (gamma N -> K Lambda pi, K Sigma pi, N K Kbar)",
                       rounds_to(min(klp[0], ksp[0], pair[0]), third), f"{min(klp[0], ksp[0], pair[0]):.4f} GeV"),
                Result(f"gamma N -> K Lambda pi: {lo}-{hi} GeV for its four charge states",
                       klp[2] == 4 and rounds_to(klp[0], lo) and rounds_to(klp[1], hi), f"{klp[0]:.4f}-{klp[1]:.4f} GeV"),
                Result(f"README: M2 makes kaons from {readme} GeV, the threshold of gamma N -> K Lambda pi",
                       rounds_to(klp[0], readme), f"{klp[0]:.4f} GeV"),
                Result(f"K Sigma pi from {ks} GeV (its {ksp[2]} charge states on p and n)", rounds_to(ksp[0], ks),
                       f"{ksp[0]:.4f} GeV"),
                Result(f"an antikaon only with a kaon pair, N K Kbar from {pr} GeV (M2 ends at 1.5 GeV)",
                       rounds_to(pair[0], pr) and pair[0] > 1.5, f"{pair[0]:.4f} GeV"),
                Result(f"gamma p -> N K Kbar pi from {pr_pi} GeV", rounds_to(pair_pi[0], pr_pi), f"{pair_pi[0]:.4f} GeV"),
                Result(f"gamma p -> Sigma+(1385) K0 from {y} GeV", rounds_to(y1385, y), f"{y1385:.4f} GeV")]
    out = compare(stated())
    with L.patched(r"$1.152\text{–}1.168$", r"$1.152\text{–}1.178$", "checks"):
        broken = compare(stated())
    return out + [Result("control: 1.178 for 1.168 GeV in Results", not all(r.ok for r in broken), "fails, as it must")]


# --- the checks: the multi-pion kaons and the energy bookkeeping ---

SUM_K_TEX = r"\sum_{c \in t,\thinspace K} \chi_c\thinspace M_{c,r}^K"     # the README's sums, asserted as written
SUM_PI_TEX = r"\sum_{c \in t,\thinspace \pi} \chi_c\thinspace M_c^\pi"


def readme_s_tr(e_kaons, e_pions):
    """The README's s_{t,r} with its sums over the kaon and the pion channels set to e_kaons and e_pions."""
    lhs, rhs = L.sides(L.parts(L.display(r"s_{t,r} = 1 "))[0])
    assert lhs == "s_{t,r}" and SUM_K_TEX in rhs and SUM_PI_TEX in rhs, rhs
    EKs, EPs = sp.symbols("E_Ksum E_pisum")
    return L.to_sympy(rhs.replace(SUM_K_TEX, "E_K").replace(SUM_PI_TEX, r"E_\pi"), {"E_K": EKs, r"E_\pi": EPs}).subs(
        {EKs: e_kaons, EPs: e_pions})


@derivation("kp.s_tr")
def check_s_tr():
    """With the pions of type t scaled by s_{t,r}, the pions and the kaons of the row r carry the energy the pions carried."""
    L.line(r"(the kaons of the row, the pions of the type), so that the energy of the interaction is kept at every photon energy")
    chi_pi, m_pi = sp.symbols("chi_pi1:4", positive=True), sp.symbols("M_pi1:4", positive=True)    # three pion channels
    chi_k, m_k = sp.symbols("chi_K1:3", positive=True), sp.symbols("M_K1:3", positive=True)        # two kaon channels of row r
    e_pi = sum(a * b for a, b in zip(chi_pi, m_pi))       # the share of the nucleon energy in the type's pions (all of it)
    e_k = sum(a * b for a, b in zip(chi_k, m_k))
    s_ = readme_s_tr(e_k, e_pi)
    after = sum(a * s_ * b for a, b in zip(chi_pi, m_pi)) + e_k     # the pions' multiplicities scaled, the kaons added
    with L.patched(r"s_{t,r} = 1 - \frac", r"s_{t,r} = 1 + \frac"):
        broken = readme_s_tr(e_k, e_pi)
    return [exact("sum_pi chi_c s M_c^pi + sum_K chi_c M_c,r^K = sum_pi chi_c M_c^pi (any multiplicities and energy fractions)",
                  after - e_pi),
            exact_fails("control: 1 + ... for s", sum(a * broken * b for a, b in zip(chi_pi, m_pi)) + e_k - e_pi)]


MPI_TABLE = r"| Type, photon energy $\epsilon_r$ | Cross section of the type [μb] |"


def mpi_rows():
    """[(type, its cross section, kaons per interaction from protons, the K+ + K0 cross section as printed)] of the table."""
    out = []
    for cells in readme_table(MPI_TABLE):
        mult = [sp.Rational(v) for v in cells[2].split(", ")]
        out.append((cells[0], sp.Rational(cells[1]), mult, cells[3]))
    return out


def readme_mpi_steps():
    """[(type index t, eps_min, eps_max [GeV])] of the rows of the multi-pion table (M(t + 1))."""
    out = []
    for name, *_ in mpi_rows():
        m_ = re.fullmatch(r"M(\d), \$(.+?)\\text\{–\}(.+?)\$ GeV", name)
        out.append((int(m_.group(1)) - 1, float(L.to_sympy(m_.group(2), {})), float(L.to_sympy(m_.group(3), {}))))
    return out


TYPES = "describe multi-pion production by seven interaction types"


def readme_type_ranges():
    """{type index t: (eps_min, eps_max) [GeV]} of the multi-pion types M(t + 1) as the README gives them (Hümmer et al.'s
    Table 6; M7 up to its end in AM3)."""
    line = L.line(TYPES)
    out = {int(n) - 1: tuple(float(L.to_sympy(v, {})) for v in tex.split(r"\text{–}"))
           for n, tex in re.findall(r"M(\d) \$([^$]+)\$ GeV", line)}
    out[6] = (out[5][1], float(L.to_sympy(re.search(r"M7 above, up to \$([^$]+)\$ GeV", line).group(1), {})))
    return out


@derivation("kp.mpi_sigma")
def check_mpi_sigma():
    """The K+ + K0 cross section of each row of the multi-pion table is the type's cross section times K+ + K0 per interaction."""
    def results():
        rows = mpi_rows()
        bad = [f"{t}: {float(sg * (mu_[0] + mu_[1]))} for {printed}" for t, sg, mu_, printed in rows
               if not rounds_to(float(sg * (mu_[0] + mu_[1])), printed)]
        m7 = [mu_ for t, _, mu_, _ in rows if t.startswith("M7")][0]
        said = sp.Rational(re.search(r"M7 makes ([\d.]+) per interaction", L.line("M7 makes")).group(1))
        return [Result(f"the {len(rows)} rows: sigma x (K+ + K0) rounds to the printed column", not bad, f"differ: {bad}" if bad
                       else "all rows"),
                Result("M7 makes 1.135 K+ + K0 per interaction (> 1: the column exceeds the type's cross section)",
                       said == m7[0] + m7[1] and said > 1, f"{float(m7[0])} + {float(m7[1])} = {float(m7[0] + m7[1])}")]
    out = results()
    with L.patched(r"| 0.0051, 0.0060, 0, 0 | 0.94 |", r"| 0.0051, 0.0060, 0, 0 | 0.97 |"):
        out.append(Result("control: 0.97 for 0.94 (M2)", not all(r_.ok for r_ in results()), "fails, as it must"))
    return out


# --- the checks: pp collisions ---

K_16 = r"n(\ep) = n_0   \ep^{-1} \alpha (\ep) \beta(\ep) \gamma_1 (\ep)  \gamma_2 (\ep)"
K_17 = [r"n_0& = &  1.21 \times 10^{p_0 + p_1 p_2^2}", r"\alpha (\ep) & = &  \ep^{p_1 (\log (\ep) - 2 p_2 )}",
        r"\beta (\ep) & = &  10^{- (\ep/\ep_3)^{p_{30}}} 10^{- (\ep/\ep_5)^{p_{50}}}",
        r"\gamma_1 (\ep) & = & \tanh \lb p_{70} \log \lb \ep / m c^2 \rb \rb",
        r"\gamma_2 (\ep) & = & \tanh \lb p_{80} \log \lb E_p / \ep \rb \rb"]
K_DEFS = r"where $\ep_3 \equiv 10^{p_2 + p_4}$, $\ep_5 \equiv 10^{p_2 + p_6}$ and all energies are expressed in units of GeV"
K_18 = [r"p_0 & = & p_{00} + p_{01} \log (E_p)", r"p_1 & = & p_{10} + p_{11} \log (E_p)",
        r"p_2 & =&  p_{20} + p_{21} \log (E_p)", r"p_4 & = & p_{40} - p_2", r"p_6 & = & p_{60} + p_2"]
K_MACROS = [r"\newcommand{\lb}[0] { \left( }", r"\newcommand{\rb}[0] { \right) }", r"\newcommand{\ep}[0] { \epsilon }"]
K_T3 = r"& $\pi^+ $& $\pi^-$ & $\pi^0$ &  $K^+ $& $K^-$ & $K^0$ & $\overline{K^0} $"
Ep, mKs = sp.symbols("E_p m_K", positive=True)
P = {name: sp.Symbol(name.replace("{", "").replace("}", "")) for name in
     ["p_0", "p_1", "p_2", "p_4", "p_6"] + [f"p_{{{i}{j}}}" for i in range(9) for j in range(2)]}
TOK_K = {r"\epsilon": eps, r"\epsilon_3": sp.Symbol("epsilon_3"), r"\epsilon_5": sp.Symbol("epsilon_5"), "E_p": Ep,
         "n_0": sp.Symbol("n_0"), **P}


def koers(tex, extra=None):
    """The right side of an equation of Koers et al. (TeX source), their macros expanded, parsed."""
    t = re.sub(r"\\ep(?![A-Za-z])", r"\\epsilon", tex).replace(r"\lb", "(").replace(r"\rb", ")")
    rhs = re.split(r"&\s*=\s*&|\\equiv|=", t, maxsplit=1)[1]
    return L.to_sympy(rhs, {**TOK_K, **(extra or {})}, funcs=KFUNCS)


KFUNCS = {r"\alpha": sp.Function("alpha"), r"\beta": sp.Function("beta"), r"\gamma_1": sp.Function("gamma_1"),
          r"\gamma_2": sp.Function("gamma_2")}


def readme_koers(pin):
    """{left side: parsed right side} of a README display equation of Koers et al.'s spectra."""
    out = {}
    for p in L.parts(L.display(pin)):
        lhs, rhs = L.sides(p)
        out[lhs] = L.to_sympy(rhs, {**TOK_K, "m_K": mKs}, funcs=KFUNCS)
    return out


# Kelner, Aharonian & Bugayov (2006), from the TeX source (macros \t = \tilde, \d = \delta, \ds = \displaystyle): the
# protons per unit energy (eq. 70), the source of gamma-rays (eq. 71; AM3's pion source has the form with F_pi), the
# delta-functional approximation (eqs. 75-77; eq. 76 prints k for kappa)
KEL_MACROS = [r"\def\d{\delta}", r"\def\t{\tilde}", r"\def\ds{\displaystyle}"]
KEL_70 = r"dN_p=J_p(E_p)\,dE_p"
KEL_71 = (r"&\Phi_\gamma(E_\gamma)\equiv\ds\frac{dN_\gamma}{dE_\gamma}=&",
          r"&\ds c\,n_{H}^{}\int\limits_{E_\gamma}^\infty\! \sigma_{\rm inel}(E_p)\,J_p(E_p)\,"
          r"F_\gamma\!\left(\frac{E_\gamma}{E_p},\,E_p\right)\frac{dE_p}{E_p}\,.&")
KEL_75 = (r"\t F_\pi=\t n\,\d\left(E_\pi-\frac{\kappa}{\t n}\,E_{\rm kin}\right),",
          r"where $E_{\rm kin}=E_p-m_p$ is the kinetic energy of protons.")
KEL_76 = r"\int\!\t F_\pi\,dE_\pi=\t n\,,\quad {\rm and} \quad \int\!E_\pi\,\t F_\pi\,dE_\pi=k\,E_{\rm kin}\, ."
KEL_77 = (r"q_\pi(E_\pi)=\t n\,\frac{c\,n_H^{}}{K_\pi}\,\sigma_{\rm inel}\!\left(m_p+\frac{E_\pi}{K_\pi}\right)"
          r"J_p\!\left(m_p+\frac{E_\pi}{K_\pi}\right),", r"where $K_\pi=\kappa/\t n$.")
SIGI, NPF, FKF = sp.Function("sigma_inel"), sp.Function("N_p"), sp.Function("F_K")   # any functions
nH = sp.Symbol("n_H", positive=True)


def readme_pp_source():
    """(the weight of the measure dE_p, the integrand) of the README's Q_K, with sigma_inel, N_p and F_K any functions."""
    lhs, rhs = L.sides(L.parts(L.display(r"Q_K(E_K) = c\thinspace n_H\int"))[0])
    head, rest = rhs.split(r"\int ", 1)
    measure, body = rest.split(r"\thinspace ", 1)
    assert lhs == "Q_K(E_K)", lhs
    dEp = sp.Symbol("dE_p")
    weight = sp.diff(L.to_sympy(measure, {"dE_p": dEp, "E_p": Ep}), dEp)
    tokens = {"c": c, "n_H": nH, "E_p": Ep, "E_K": EK}
    return L.to_sympy(head, tokens) * weight, L.to_sympy(body, tokens, {r"\sigma_{\rm inel}": SIGI, "N_p": NPF, "F_K": FKF})


@derivation("pp.source")
def check_pp_source():
    """Q_K counts the collisions, as Kelner et al.'s eq. 71 (F_gamma -> F_K): N_p dE_p protons collide at c n_H sigma_inel."""
    L.line(r"and $n = 0$ outside $m_K \lt \epsilon \lt E_p$")
    weight, body = readme_pp_source()
    xk = EK / Ep
    # a collision makes F_K(x, E_p) dx kaons, x = E_K/E_p: F_K dx/dE_K = F_K/E_p per unit E_K at fixed E_p
    counted = c * nH * SIGI(Ep) * NPF(Ep) * FKF(xk, Ep) * sp.diff(xk, EK)
    tok = {"c": c, "n_{H}^{}": nH, "E_p": Ep, r"E_\gamma": EK, "dE_p": 1}
    kel = (L.to_sympy(piece(KEL_71[1], r"c\,n_{H}^{}"), tok) * L.to_sympy(piece(KEL_71[1], r"\frac{dE_p}{E_p}"), tok)
           * L.to_sympy(piece(KEL_71[1], r"\sigma_{\rm inel}(E_p)\,J_p(E_p)\,F_\gamma\!\left(\frac{E_\gamma}{E_p},\,E_p\right)"),
                        tok, {r"\sigma_{\rm inel}": SIGI, "J_p": NPF, r"F_\gamma": FKF}))
    with L.patched(r"Q_K(E_K) = c\thinspace n_H\int \frac{dE_p}{E_p}\thinspace", r"Q_K(E_K) = c\thinspace n_H\int dE_p\thinspace"):
        w_b, b_b = readme_pp_source()
    return [in_source("eqs. 70 and 71 and the macros as transcribed", KELNER, KEL_70, *KEL_71, *KEL_MACROS),
            exact("README integrand = the collisions' kaons per unit E_K (any sigma_inel, N_p, F_K; integrands in E_p)",
                  weight * body - counted),
            exact("= Kelner et al.'s eq. 71 with F_gamma -> F_K, J_p -> N_p (protons per unit energy, eq. 70)", weight * body - kel),
            Result("eq. 71 integrates from E_p = E_gamma; the README's F_K vanishes for x > 1 (n = 0 outside m_K < eps < E_p), so "
                   "its integral over all E_p is the same", True, "as written (asserted)"),
            exact_fails("control: dE_p for dE_p/E_p", w_b * b_b - counted)]


@derivation("pp.koers_n")
def check_koers_n():
    """n(eps) = n_0 eps^-1 alpha beta gamma_1 gamma_2, n_0 and alpha are Koers et al.'s eqs. 16, 17a-b (log = log10)."""
    pin = r"n(\epsilon) = n_0\thinspace \epsilon^{-1}"
    ours = readme_koers(pin)
    out = [in_source("eqs. 16, 17a-b and the macros as transcribed", KOERS, K_16, *K_17[:2], *K_MACROS),
           exact("README n(eps) = eq. 16", ours[r"n(\epsilon)"] - koers(K_16)),
           exact("README n_0 = eq. 17a", rational(ours["n_0"] - koers(K_17[0]))),
           exact("README alpha = eq. 17b", ours[r"\alpha"] - koers(K_17[1]))]
    # log = log10: n_0 alpha = 1.21 x 10^(p_0 + p_1 (log eps - p_2)^2), a parabola in log eps. With ln, the
    # 10^(p_1 p_2^2) of n_0 does not complete the square. Lg = log10 eps; the parsed log(eps) is the README's \log.
    Lg = sp.Symbol("L")
    power = ours["n_0"].as_coeff_Mul()[1]
    assert power.as_base_exp()[0] == 10 and ours[r"\alpha"].as_base_exp()[0] == eps
    for name, log_eps in (("log = log10", Lg), ("log = ln", Lg * sp.log(10))):
        exponent = power.as_base_exp()[1] + ours[r"\alpha"].as_base_exp()[1].subs(sp.log(eps), log_eps) * Lg
        d = exponent - (P["p_0"] + P["p_1"] * (Lg - P["p_2"])**2)
        out.append(exact(f"{name}: n_0 alpha = 1.21 x 10^(p_0 + p_1 (log eps - p_2)^2)", d) if log_eps == Lg else
                   exact_fails(f"{name}: no parabola in log eps (the README's log = log10 is Koers et al.'s)", d))
    # F_K(x, E_p) = E_p n(x E_p): the same kaons per collision, dN/dx = E_p dN/deps
    FK = L.to_sympy(L.sides(L.inline(r"F_K(x, E_p) = E_p n(xE_p)"))[1], {"E_p": Ep, "x": x}, funcs={"n": n})
    lo, hi = sp.symbols("a b", positive=True)
    t = sp.Integral(n(eps), (eps, lo, hi)).transform(eps, (Ep * x, x))
    out.append(exact("F_K = E_p n(x E_p): int n deps = int F_K dx (eps = x E_p)", integrand(t)[0] - FK))
    with L.patched(r"10^{\thinspace p_0 + p_1 p_2^2}", r"10^{\thinspace p_0 + p_1 p_2}"):
        out.append(exact_fails("control: p_1 p_2 for p_1 p_2^2 in n_0",
                               rational(readme_koers(pin)["n_0"] - koers(K_17[0]))))
    return out


@derivation("pp.koers_beta")
def check_koers_beta():
    """beta, gamma_1 and gamma_2 are Koers et al.'s eqs. 17c-e (their m c^2 = m_K, energies in GeV)."""
    pin = r"\beta = 10^{-(\epsilon/\epsilon_3)^{p_{30}}}"
    ours = readme_koers(pin)
    out = [in_source("eqs. 17c-e and the units as transcribed", KOERS, *K_17[2:], K_DEFS),
           exact("README beta = eq. 17c", ours[r"\beta"] - koers(K_17[2])),
           exact("README gamma_1 = eq. 17d (m c^2 = m_K)", ours[r"\gamma_1"] - koers(K_17[3], {"m": mKs, "c": 1})),
           exact("README gamma_2 = eq. 17e", ours[r"\gamma_2"] - koers(K_17[4]))]
    with L.patched(r"\big(p_{80}\log(E_p/\epsilon)\big)", r"\big(p_{80}\log(\epsilon/E_p)\big)"):
        out.append(exact_fails("control: log(eps/E_p) in gamma_2", readme_koers(pin)[r"\gamma_2"] - koers(K_17[4])))
    return out


@derivation("pp.koers_params")
def check_koers_params():
    """eps_3, eps_5 and p_0 ... p_6 are Koers et al.'s definitions after eq. 17 and eqs. 18a-e; Table III's names."""
    theirs = dict(zip([r"\epsilon_3", r"\epsilon_5", "p_0", "p_1", "p_2", "p_4", "p_6"],
                      re.findall(r"\$(\\ep_[35] \\equiv .+?)\$", K_DEFS) + K_18))
    ours = readme_params
    readme = ours()
    out = [in_source("the definitions of eps_3, eps_5 and eqs. 18a-e as transcribed", KOERS, K_DEFS, *K_18),
           Result("the README defines eps_3, eps_5, p_0, p_1, p_2, p_4, p_6", list(readme) == list(theirs), f"{list(readme)}")]
    for name, tex in theirs.items():
        out.append(exact(f"README {name} = Koers et al.'s", readme[name] - koers(tex)))
    with L.patched(r"$p_4 = p_{40} - p_2$", r"$p_4 = p_{40} + p_2$"):
        out.append(exact_fails("control: p_40 + p_2 in the README", ours()["p_4"] - koers(theirs["p_4"])))
    # Table III: the twelve parameter names, and the README's five rows against its columns
    names, readme_rows = readme_table3()
    rows = {re.match(r"\$(p_\{\d\d\})\$", row).group(1): [sp.Rational(v) for v in re.findall(r"\$(-?[\d.]+)\$", row)]
            for row in K_T3_ROWS}
    out.append(in_source("Table III (columns pi+, pi-, pi0, K+, K-, K0, anti-K0) as transcribed", KOERS, K_T3,
                         "".join(K_T3_ROWS)))
    out.append(Result("README table columns = Table III's twelve p_ij, in order", names == list(rows), f"{names}"))
    column = {r"$\mathrm{K}^+$": 3, r"$\mathrm{K}^-$": 4, r"$\mathrm{K}^0$": 5, r"$\bar{\mathrm{K}}^0$": 6, r"$\pi^0$": 2}
    bad = []
    for label, col in column.items():
        values = readme_rows[label]
        bad += [f"{label} {nm}" for nm, v in zip(names, values) if v != rows[nm][col]] + \
               ([f"{label}: {len(values)} values"] if len(values) != len(names) else [])
    assert sorted(readme_rows) == sorted(column), list(readme_rows)
    out.append(Result("README table rows K+, K-, K0, anti-K0, pi0 = Table III (60 values)", not bad,
                      f"differ: {bad}" if bad else "equal"))
    return out


PDG_AL = sp.Rational("0.00332")   # K0_L listing p. 34: A_L, the weighted average of A_L(mu) and A_L(e), 0.332 +- 0.006 %


def readme_neutral(FK0, FK0b):
    """The README's F_K0S = F_K0L for the neutral kaons' F_K0, F_K0b."""
    line = L.line(r"is half $\mathrm{K}^0_\mathrm{S}$ and half $\mathrm{K}^0_\mathrm{L}$: $F_{\mathrm{K}^0_\mathrm{S}}$")
    tex = re.search(r"\$F_\{\\mathrm\{K\}\^0_\\mathrm\{S\}\}\$ \$= F_\{\\mathrm\{K\}\^0_\\mathrm\{L\}\}\$ \$= (.+?)\$\.", line).group(1)
    return L.to_sympy(tex, {r"F_{\mathrm{K}^0}": FK0, r"F_{\bar{\mathrm{K}}^0}": FK0b})


@lru_cache(None)
def numeric_neutral():
    """readme_neutral as a function of the numbers (or arrays) of K0 and anti-K0: the K0_S (= K0_L) among them."""
    a, b = sp.symbols("a b")
    return sp.lambdify((a, b), readme_neutral(a, b), "numpy")


@derivation("pp.neutral")
def check_pp_neutral():
    """A K0 or anti-K0 is half K0_S and half K0_L (the CP eigenstates); CP violation moves the halves by A_L/2."""
    small = re.search(r"\*\*Neutral kaons\.\*\* A \$\\mathrm\{K\}\^0\$ or \$\\bar\{\\mathrm\{K\}\}\^0\$ is half \$\\mathrm\{K\}\^0_\\mathrm\{S\}\$ "
                      r"and half \$\\mathrm\{K\}\^0_\\mathrm\{L\}\$\. CP violation and \$\\mathrm\{K\}\^0\\text\{–\}\\bar\{\\mathrm\{K\}\}\^0\$ "
                      r"interference \(\$\\lesssim (\d+)\\text\{％\}\$\) are neglected", L.line(r"**Neutral kaons.**")).group(1)
    FK0, FK0b = sp.symbols("F_K0 F_K0b", positive=True)
    KS, KL = sp.Matrix([1, 1]) / sp.sqrt(2), sp.Matrix([1, -1]) / sp.sqrt(2)   # in the basis K0, anti-K0 (CP K0 = anti-K0)
    F = {n: v[0]**2 * FK0 + v[1]**2 * FK0b for n, v in (("S", KS), ("L", KL))}  # |<K_S,L|K0>|^2 F_K0 + |<K_S,L|K0bar>|^2 F_K0bar
    # CP violation: K_S,L = p K0 +- q anti-K0, |p|^2 + |q|^2 = 1. With Delta S = Delta Q only the K0 part decays into
    # pi- l+ nu, so PDG's A_L = [Gamma(pi- l+ nu) - Gamma(pi+ l- nu-bar)]/SUM = |p|^2 - |q|^2.
    p2, AL = sp.symbols("p2 A_L", positive=True)
    share = sp.solve(sp.Eq(AL, p2 - (1 - p2)), p2)[0]
    with L.patched(r"$= \tfrac12(F_{\mathrm{K}^0} + F_{\bar{\mathrm{K}}^0})$", r"$= \tfrac12(F_{\mathrm{K}^0} - F_{\bar{\mathrm{K}}^0})$"):
        broken = readme_neutral(FK0, FK0b)
    return [exact("F_K0S = |<K_S|K0>|^2 F_K0 + |<K_S|K0bar>|^2 F_K0bar, K_S = (K0 + K0bar)/sqrt 2: the README's", F["S"] - readme_neutral(FK0, FK0b)),
            exact("F_K0L, K_L = (K0 - K0bar)/sqrt 2: the same", F["L"] - readme_neutral(FK0, FK0b)),
            exact("F_K0S + F_K0L = F_K0 + F_K0bar: each neutral kaon counted once", F["S"] + F["L"] - FK0 - FK0b),
            exact("CP violation: the K0 share of K_S and of K_L is |p|^2 = (1 + A_L)/2", share - (1 + AL) / 2),
            Result(f"with PDG's A_L = {float(100 * PDG_AL)} % (K0_L listing p. 34) the halves move by A_L/2 = "
                   f"{float(PDG_AL / 2):.2e}, within the README's ≲ {small} % (the interference is not checked here)",
                   PDG_AL / 2 < sp.Rational(int(small), 100), f"A_L/2 = {float(PDG_AL / 2):.4f}"),
            exact_fails("control: 1/2 (F_K0 - F_K0bar)", F["S"] - broken)]


nKf, npf, Fhf = sp.Function("n_K"), sp.Function("n_pi0"), sp.Function("Fhat_pi")   # n(eps; E_p) of the kaon and the pi0, F-hat_pi
Estar = sp.Symbol("E_ast", positive=True)
EXT_PIN = r"F_K(x,E_p) = R(x)\thinspace \hat F_\pi(x,E_p)"


def readme_extension():
    """(F_K(x, E_p) above E* for x >= x_min, R(x_min), x_min) of the README, n_K and F-hat_pi any functions."""
    tex = L.display(EXT_PIN)
    xs = sp.Symbol("x_s", positive=True)
    xmin = L.to_sympy(re.fullmatch(r"x_s = \\max\(x, (.+)\)", L.part(tex, "x_s =")).group(1), {})
    R = L.to_sympy(L.sides(L.part(tex, "R(x) ="))[1].replace(";", ","), {r"E^\ast": Estar, "x_s": xs},
                   {"n_K": nKf, r"\hat F_\pi": Fhf})
    F = L.to_sympy(L.sides(L.part(tex, "F_K(x,E_p) ="))[1], {"x": x, "E_p": Ep}, {"R": lambda a: R.subs(xs, a), r"\hat F_\pi": Fhf})
    return F, R.subs(xs, xmin), xmin


LOW_X = r"For $x \lt 10^{-4}$ (kaons below 100 GeV at $E^\ast$)"


def readme_low_x():
    """R(x)/R(x_min) for x < x_min as the README writes it, rho(min(x E_p, x_min E*))/rho(x_min E*), rho = n_K/n_pi0 at E*,
    and the kaon energy the text names (100 GeV)."""
    line = L.line(LOW_X)
    ratio = re.search(r"\$R\(x\)\$ \$\\to R\(10\^\{-4\}\)\$ \$(.+?)\$\.", line).group(1)
    rho_tex = re.search(r"\$\\rho\(\\epsilon\) = (.+?)\$", line).group(1)
    args = re.fullmatch(r"\\rho\(\\min\((.+?), (.+?)\)\)/\\rho\((.+?)\)", ratio).groups()   # min asserted as written
    a1, a2, a3 = (L.to_sympy(t, {"x": x, "E_p": Ep, r"E^\ast": Estar}) for t in args)

    def rho(e):
        return L.to_sympy(rho_tex, {r"\epsilon": e}, {"n_K": lambda b: nKf(b, Estar), r"n_{\pi^0}": lambda b: npf(b, Estar)})
    gev = sp.Integer(re.search(r"\(kaons below (\d+) GeV at", line).group(1))
    return rho(sp.Min(a1, a2)) / rho(a3), gev


def numeric_F(expr, species, model="SIBYLL"):
    """The README's F_K expression as a function of (x, E_p), with E* and the functions of pp_setup (Koers et al.'s n,
    Kelner et al.'s F-hat)."""
    n, fhat, _, _, es, _, _ = pp_setup()
    return sp.lambdify((x, Ep), expr.subs(Estar, es), modules=[{"n_K": n[species], "n_pi0": n["pi0"],
                                                                 "Fhat_pi": fhat[model][0]}, "math"])


def against_helper(f_readme, species, points):
    """Largest relative difference between f_readme and F_K (the function the numbers of the pp kaons use)."""
    return max(abs(f_readme(xv, E) / F_K(species, xv, E, "SIBYLL") - 1) for xv, E in points)


@derivation("pp.extension")
def check_pp_extension():
    """Above E* the kaons keep Koers et al.'s K/pi ratio at E* (x >= 1e-4): F_K is continuous at E*, the ratio fixed in E_p."""
    F, _, xmin = readme_extension()
    koers = L.to_sympy(L.sides(L.inline(r"F_K(x, E_p) = E_p n(xE_p)"))[1], {"E_p": Ep, "x": x}, {"n": lambda a: nKf(a, Ep)})
    es = L.to_sympy(L.sides(L.inline(r"E^\ast = 10^6"))[1], {})
    top = L.to_sympy(re.search(r"\*\*100 GeV \$\\le E_p \\le (10\^\d+)\$ GeV:", L.line(r"**100 GeV $\le E_p \le")).group(1), {})
    r = rng(11)
    pts = [(float(log_uniform(r, float(xmin), 0.9)), float(log_uniform(r, 1.01e6, 1e11))) for _ in range(12)]
    worst = max(against_helper(numeric_F(F, sp_), sp_, pts) for sp_ in ("K+", "K-", "K0", "K0b"))
    with L.patched(r"R(x) = \frac{E^\ast \thinspace n_K", r"R(x) = \frac{n_K"):
        broken = readme_extension()[0]
    return [exact("at E_p = E*: F_K = E* n_K(x E*; E*), Koers et al.'s E_p n(x E_p) there (x >= 1e-4): continuous at E*",
                  F.subs(Ep, Estar) - koers.subs(Ep, Estar)),
            exact("F_K/F-hat_pi at fixed x >= 1e-4 does not depend on E_p (the ratio kept, Feynman scaling)", sp.diff(F / Fhf(x, Ep), Ep)),
            Result("E* = 10^6 GeV, the top of Koers et al.'s range (100 GeV <= E_p <= 10^6 GeV)", es == top, f"E* = {es}, top {top}"),
            Result("the README's form (Koers et al.'s n, Kelner et al.'s F-hat_pi, SIBYLL) = the F_K of the pp numbers, K+, K-, K0, "
                   "anti-K0 at 12 points, x >= 1e-4, E_p 1e6-1e11 GeV", worst < 1e-12, f"numeric: max rel. difference {worst:.1e}"),
            exact_fails("control: no E* in R(x)", broken.subs(Ep, Estar) - koers.subs(Ep, Estar))]


@derivation("pp.low_x")
def check_pp_low_x():
    """Below x = 1e-4, R follows Koers et al.'s K/pi0 ratio at E* in the kaon energy, continuously, up to 100 GeV."""
    ratio, gev = readme_low_x()
    _, Rmin, xmin = readme_extension()
    es = L.to_sympy(L.sides(L.inline(r"E^\ast = 10^6"))[1], {})
    rho = lambda e: nKf(e, Estar) / npf(e, Estar)   # noqa: E731
    r = rng(12)
    pts = [(float(log_uniform(r, 1e-6, float(xmin) * 0.999)), float(log_uniform(r, 1.01e6, 1e11))) for _ in range(12)]
    worst = max(against_helper(numeric_F(Rmin * ratio * Fhf(x, Ep), sp_), sp_, pts) for sp_ in ("K+", "K-", "K0", "K0b"))
    with L.patched(r"/\rho(10^{-4}E^\ast)$.", r"/\rho(10^{-3}E^\ast)$."):
        broken = readme_low_x()[0]
    return [exact("at x = 1e-4, any E_p >= E*: R = R(1e-4) (continuous)", ratio.subs(x, xmin).subs(Ep, Estar * (1 + w)) - 1),
            exact("at E_p = E*, x < 1e-4: R/R(1e-4) = rho(x E*)/rho(1e-4 E*), Koers et al.'s K/pi0 ratio in the kaon energy",
                  ratio.subs(Ep, Estar).subs(x, xmin / (1 + w)) - rho(xmin * Estar / (1 + w)) / rho(xmin * Estar)),
            exact("for any E_p, R depends on x and E_p only through the kaon energy x E_p", sp.diff(ratio.subs(x, EK / Ep), Ep)),
            exact(f"x < 1e-4 at E*: kaons below {gev} GeV", xmin * es - gev),
            Result("the README's form = the F_K of the pp numbers, K+, K-, K0, anti-K0 at 12 points, x 1e-6 to 1e-4, E_p 1e6-1e11 "
                   "GeV", worst < 1e-12, f"numeric: max rel. difference {worst:.1e}"),
            exact_fails("control: rho(1e-3 E*) in the denominator", broken.subs(x, xmin).subs(Ep, Estar * (1 + w)) - 1)]


@derivation("pp.gamma_cm")
def check_gamma_cm():
    """gamma_cm = sqrt((E_p + m_p)/(2m_p)) = (E_p + m_p)/sqrt(s), s = 2m_p^2 + 2m_p E_p: kinematics of a fixed target."""
    tok = {"E_p": Ep, "m_p": m}
    s_ours = L.to_sympy(L.sides(L.part(L.display(r"s = 2m_p^2 + 2m_pE_p"), "s ="))[1], tok)

    def forms():
        """The README's two forms, '$\\gamma_\\mathrm{cm} = a$ $= b$'."""
        a, b = re.search(r"\$\\gamma_\\mathrm\{cm\} = ([^$]+)\$ \$= ([^$]+)\$",
                         L.line(r"; $\gamma_\mathrm{cm} = ")).groups()
        return L.to_sympy(a, tok), L.to_sympy(b, {**tok, "s": s_ours})
    first, second = forms()
    qv = 1 + w                                                     # E_p = m_p (q^2 + q^-2)/2, P = m_p (q^2 - q^-2)/2
    at = {Ep: m * (qv**2 + qv**-2) / 2}
    P_ = m * (qv**2 - qv**-2) / 2
    s_kin = (at[Ep] + m)**2 - P_**2                                # (p_1 + p_2)^2, target at rest
    g_kin = 1 / sp.sqrt(1 - (P_ / (at[Ep] + m))**2)                # the CM frame moves with beta = P/(E_p + m_p)
    with L.patched(r"\sqrt{(E_p + m_p)/(2m_p)}", r"\sqrt{(E_p + m_p)/m_p}"):
        broken = forms()[0]
    th = sp.Symbol("theta")
    E_lab = lambda angle: g_kin * (Es + sp.sqrt(1 - 1 / g_kin**2) * ps * sp.cos(angle))   # noqa: E731
    return [exact("s = (p_1 + p_2)^2 = 2m_p^2 + 2m_p E_p", s_kin - s_ours.subs(at)),
            exact("gamma_cm = 1/sqrt(1 - beta^2), beta = P/(E_p + m_p): = sqrt((E_p + m_p)/(2m_p))", g_kin - first.subs(at)),
            exact("= (E_p + m_p)/sqrt(s)", g_kin - second.subs(at)),
            exact_fails("control: sqrt((E_p + m_p)/m_p) in the README", g_kin - broken.subs(at)),
            exact("a kaon of CM energy E*: E_lab(theta) + E_lab(pi - theta) = 2 gamma_cm E* (>= 2 gamma_cm m_K; pp is "
                  "forward-backward symmetric)", sp.expand(E_lab(th) + E_lab(sp.pi - th) - 2 * g_kin * Es))]


DELTA_PIN = r"\tilde n_K(E_p) = \langle n_K\rangle_{\rm KPW}(E_t)"


def readme_delta(nkpw, Et):
    """(n-tilde_K(E_p), f(s) as a function, the symbols (s_th, a)) of the README's multiplicity below E_t."""
    disp = L.display(DELTA_PIN)
    A_, a_, b_, s_, sth = sp.symbols("A a b s s_th", positive=True)
    f_ = lambda v: L.to_sympy(L.sides(L.part(disp, "f(s) ="))[1], {"A": A_, "a": a_, "b": b_, "s": s_,   # noqa: E731
                                                                   r"s_{\rm th}": sth}).subs(s_, v)
    s_of = lambda E: L.to_sympy(L.sides(L.part(disp, "s ="))[1], {"m_p": m, "E_p": E})   # noqa: E731
    lhs, rhs = L.sides(L.parts(disp)[0])
    assert lhs == r"\tilde n_K(E_p)", lhs
    n_ = L.to_sympy(rhs, {r"\langle n_K\rangle_{\rm KPW}(E_t)": nkpw, "s_t": s_of(Et), "s": s_of(Ep), "A": A_}, {"f": f_})
    return n_, f_, (sth, a_)


@derivation("pp.delta")
def check_pp_delta():
    """Below E_t: n-tilde_K kaons of energy K_K T_p per collision, Kelner et al.'s eqs. 75-77 for kaons; continuous at E_t."""
    nkpw, Et, KK, kap, nt = sp.symbols("n_KPW E_t K_K kappa n_t", positive=True)
    line = L.line(r"Each collision makes $\tilde n_K$ kaons of energy $E_K = ")
    first = re.search(r"\$E_K = \\max\((.+?), \\gamma_\\mathrm\{cm\} m_K\)\$", line).group(1)   # the branch above the floor
    Tp = L.to_sympy(L.sides(L.inline(r"T_p = E_p - m_p"))[1], {"E_p": Ep, "m_p": m})
    L.line(r"The factor $K_{K}$ is the mean fraction of $T_p$ per kaon of the")
    E_ = sp.Symbol("E", positive=True)
    # eq. 75 for a kaon: n-tilde delta(E - (kappa/n-tilde) E_kin); K_K = (int E F dE/int F dE)/T_p (the README's definition)
    F75 = nt * sp.DiracDelta(E_ - kap / nt * Tp)
    number, energy = (sp.integrate(E_**k_ * F75, (E_, 0, sp.oo)) for k_ in (0, 1))
    K_mean = sp.simplify(energy / number / Tp)

    def source(branch):
        """Q_K(E_K) of the collisions, each making n-tilde_K kaons of the energy `branch` (pp.source's count)."""
        EKp = L.to_sympy(branch, {"K_K": KK, "T_p": Tp, "E_p": Ep})
        q = sp.integrate(c * nH * SIGI(Ep) * NPF(Ep) * nt * sp.DiracDelta(EK - EKp), (Ep, m, sp.oo))
        return q.replace(sp.Heaviside, lambda *a: 1)
    tok77 = {r"\t n": nt, "c": c, "n_H^{}": nH, r"K_\pi": KK, "m_p": m, r"E_\pi": EK}
    eq77 = L.to_sympy(KEL_77[0].split("=", 1)[1].rstrip(","), tok77, {r"\sigma_{\rm inel}": SIGI, "J_p": NPF})
    n_, f_, (sth, a_) = readme_delta(nkpw, Et)
    with L.patched(r"\frac{f(s)}{f(s_t)}", r"\frac{f(s)}{A}"):
        broken = readme_delta(nkpw, Et)[0]
    with L.patched(r"$E_K = \max(K_K T_p,", r"$E_K = \max(K_K E_p,"):
        broken_branch = re.search(r"\$E_K = \\max\((.+?), \\gamma_\\mathrm\{cm\} m_K\)\$",
                                  L.line(r"Each collision makes $\tilde n_K$ kaons of energy $E_K = ")).group(1)
    return [in_source("eqs. 75-77 as transcribed (eq. 76 prints k for kappa)", KELNER, *KEL_75, KEL_76, *KEL_77),
            exact("eqs. 75-76: the mean fraction of T_p = E_kin per kaon is kappa/n-tilde, Kelner et al.'s K_pi (after eq. 77): "
                  "the README's K_K", K_mean - kap / nt),
            exact("n-tilde_K kaons of E_K = K_K T_p per collision, counted as in pp.source: Kelner et al.'s eq. 77 with pi -> K, "
                  "J_p -> N_p", source(first) - eq77),
            exact("continuous at E_t: n-tilde_K(E_t) = <n_K>_KPW(E_t), s_t = s(E_t)", n_.subs(Ep, Et) - nkpw),
            exact("no kaons at threshold: f(s_th) = 0 (a > 0)", f_(sth).subs(a_, 1 + w)),
            exact_fails("control: f(s)/A", broken.subs(Ep, Et) - nkpw),
            exact_fails("control: E_K = K_K E_p", source(broken_branch) - eq77)]


@derivation("pp.loss")
def check_pp_loss():
    """The protons lose the energy the kaons of Q_K take: -n_H c sigma_inel kappa_K E_p, kappa_K = int x F_K dx."""
    def term():
        tex = re.search(r"gains the term \$(.+?)\$, with \$\\kappa_K\$ the fraction of \$E_p\$ the kaons carry",
                        L.line(r"energy-loss rate $\dot E/E = ")).group(1)
        return L.to_sympy(tex.replace(r"\sum_K \kappa_K", r"\kappa"), {"n_H": nH, "c": c, r"\sigma_\mathrm{inel}": SIGI(Ep),
                                                                       r"\kappa": sp.Symbol("kappa")})
    kappa = sp.Integral(x * FKF(x, Ep), (x, 0, 1))           # the fraction of E_p the kaons of one collision carry
    weight, body = readme_pp_source()
    # the energy of the kaons made per unit time by the protons per unit E_p: int dE_K E_K (README integrand), E_K = x E_p
    made = sp.Integral(EK * weight * body, (EK, 0, Ep)).transform(EK, (x * Ep, x))
    lost = lambda t: -t.subs(sp.Symbol("kappa"), kappa) * Ep * NPF(Ep)   # noqa: E731  # -(dE/dt) N_p, (dE/dt)/E = the term
    (f_made, lim_made), (f_lost, lim_lost) = integrand(made), integrand(lost(term()))
    with L.patched(r"\sum_K \kappa_K$, with", r"\sum_K \kappa_K/2$, with"):
        f_broken, _ = integrand(lost(term()))
    return [exact("the kaons' energy from the protons of E_p = the energy these protons lose by the README's term (any "
                  "sigma_inel, N_p, F_K; integrands in x)", f_made - f_lost),
            Result("both over x from 0 to 1", lim_made == lim_lost, f"{lim_made}, {lim_lost}"),
            exact_fails("control: kappa_K/2", f_made - f_broken)]


K_EQ1 = (r"\sigma_{\rm inel}=&\left[ 30.7 - 0.96\,\log\left(\frac{T_p}{T_p^{\rm th}}\right) + 0.18\,\log^2\left("
         r"\frac{T_p}{T_p^{\rm th}}\right) \right] \\ & ~~~~~~~~~~~~~~~~~~~~~~~\times\left[1-\left(\frac{T_p^{\rm th}}{T_p} "
         r"\right)^{1.9}\right]^3~{\rm mb}")
K_TH = r"$T_p^{\rm th}= 2m_\pi+m_\pi^2/2m_p\approx0.2797$~GeV is the threshold kinetic energy"
A52 = (r"\sigma_\mathrm{pp, \mathrm{inel}} (\gamma_p) = \left[ 30.7  - 0.96 \mathrm{log}\qty(\frac{T_p}{E_\mathrm{th}}) "
       r"+0.18 \mathrm{log}^2\qty(\frac{T_p}{E_ \mathrm{th}}) \right ] \times \left[1 - \qty(\frac{T_p}{E_\mathrm{th}})^{1.9} "
       r"\right]^3 \, \mathrm{mb}")
A52_TH = r"Here $E_\mathrm{th} = 2 m_\pi + m_\pi^2 / 2 m_p$ is the threshold energy depending on the pion mass"


@derivation("pp.kafexhiu")
def check_kafexhiu():
    """The AM3 paper's [1 - (T_p/E_th)^1.9]^3 (eq. A52) is negative above threshold; Kafexhiu et al.'s eq. 1 has E_th/T_p."""
    T, Eth, mpi = sp.symbols("T_p E_th m_pi", positive=True)
    line = L.line(r"Kafexhiu et al.'s eq. 1 and the code have", "reference")
    printed_tex, correct_tex = re.findall(r"\$(\[1 - .+?\]\^3)\$", line)
    tok = {"T_p": T, r"E_\mathrm{th}": Eth}
    printed, correct = (rational(L.to_sympy(t, tok)) for t in (printed_tex, correct_tex))
    # \qty( ) of the physics package, unknown to parse_latex, is a plain parenthesis
    klinger = rational(L.to_sympy(piece(A52, r"\left[1 - \qty(\frac{T_p}{E_\mathrm{th}})^{1.9} \right]^3").replace(r"\qty", ""),
                                  tok))
    kafexhiu = rational(L.to_sympy(piece(K_EQ1, r"\left[1-\left(\frac{T_p^{\rm th}}{T_p} \right)^{1.9}\right]^3"),
                                   {r"T_p^{\rm th}": Eth, "T_p": T}))
    def above_threshold(factor):
        """(power, coefficients of the numerator in w, denominator) of [inner]^3 at T_p = E_th (1 + w)^10, w > 0,
        where (T_p/E_th)^1.9 = (1 + w)^19: the factor has the sign of the numerator's coefficients if they agree."""
        inner, power = factor.subs(T, Eth * (1 + w)**10).as_base_exp()
        num, den = sp.fraction(sp.together(inner))
        return power, sp.Poly(sp.expand(num), w).coeffs(), den
    (p1, c1, d1), (p2, c2, d2) = above_threshold(printed), above_threshold(correct)
    T_th = sp.solve(2 * m**2 + 2 * m * (m + T) - (2 * m + mpi)**2, T)[0]   # s = (2m_p + m_pi)^2: p p pi0 at rest in the CM
    tok_th = {r"m_\pi": mpi, "m_p": m}
    with L.patched(r"have $[1 - (E_\mathrm{th}/T_p)^{1.9}]^3$", r"have $[1 - (T_p/E_\mathrm{th})^{1.9}]^3$", "reference"):
        broken = rational(L.to_sympy(re.findall(r"\$(\[1 - .+?\]\^3)\$", L.line(r"Kafexhiu et al.'s eq. 1 and the code have", "reference"))[1],
                                     tok))
    return [in_source("eq. 1 and its threshold as transcribed", KAFEXHIU, K_EQ1, K_TH),
            in_source("eq. A52 and its threshold as transcribed", KLINGER, A52, A52_TH),
            exact("README's printed factor = Klinger et al.'s eq. A52", printed - klinger),
            exact("README's correct factor = Kafexhiu et al.'s eq. 1 (T_p^th = E_th)", correct - kafexhiu),
            exact_fails("the two differ", printed - correct),
            Result("above threshold the printed factor is negative (a cube of a polynomial in w > 0 with negative "
                   "coefficients), eq. 1's positive", p1 == p2 == 3 and all(cf < 0 for cf in c1) and d1 == 1
                   and all(cf > 0 for cf in c2) and d2 == (1 + w)**19, f"numerators {c1[:3]}..., {c2[:3]}..."),
            exact("both papers' E_th = 2m_pi + m_pi^2/(2m_p) = T_p at s = (2m_p + m_pi)^2",
                  L.to_sympy(piece(K_TH, r"2m_\pi+m_\pi^2/2m_p"), tok_th) - T_th),
            exact("(Klinger et al.'s E_th is the same)",
                  L.to_sympy(piece(A52_TH, r"2 m_\pi + m_\pi^2 / 2 m_p"), tok_th) - T_th),
            exact_fails("control: the README's correct factor replaced by the printed one", broken - kafexhiu)]



# --- the checks: the numbers of the pp kaon spectra ---
#
# They are computed from the README's equations and tables and from the papers' TeX the module docstring lists. The
# integrals over x are adaptive quadratures in ln x (scipy, to about 1e-8).

KEL_THR = r"\left(1-\frac{m_\pi}{x\,E_p}\right)^{1/2}"
KEL = {   # Kelner et al.: eq. 12 (SIBYLL, AM3's default) and eq. 6 (QGSJET) with their parameters, from the TeX
    "SIBYLL": dict(tex=[r"F_\pi(x,\,E_p)=4\alpha B_\pi x^{\alpha-1}\left(\frac{1-x^\alpha}{1+rx^\alpha(1-x^\alpha)}\right)^4 "
                        r"\times\nonumber\\ \left(\frac{1}{1-x^\alpha}+\frac{r\,(1-2x^\alpha)}{1+rx^\alpha(1-x^\alpha)}\right)"
                        r"\left(1-\frac{m_\pi}{x\,E_p}\right)^{1/2}\,.",
                        r"B_\pi=a+0.25\,,\quad \alpha=\frac{0.98}{\sqrt{a}}\,,\,r=\frac{2.6}{\sqrt{a}}\,,",
                        r"a=3.67+0.83\,L+0.075\,L^2\,, \quad L=\ln(E_p/1\,{\rm TeV})\,."],
                   f=[r"4\alpha B_\pi x^{\alpha-1}\left(\frac{1-x^\alpha}{1+rx^\alpha(1-x^\alpha)}\right)^4",
                      r"\left(\frac{1}{1-x^\alpha}+\frac{r\,(1-2x^\alpha)}{1+rx^\alpha(1-x^\alpha)}\right)"],
                   params=[(r"B_\pi", "a+0.25"), (r"\alpha", r"\frac{0.98}{\sqrt{a}}"), ("r", r"\frac{2.6}{\sqrt{a}}"),
                           ("a", r"3.67+0.83\,L+0.075\,L^2"), ("L", r"\ln(E_p/1\,{\rm TeV})")]),
    "QGSJET": dict(tex=[r"F_\pi(x,\,E_p)=4\,\alpha B_\pi x^{\alpha-1}\left(\frac{1-x^\alpha}{(1+r\,x^\alpha)^3}\right)^4"
                        r"\times\nonumber\\ \left(\frac1{1-x^\alpha}+\frac{3\,r}{1+rx^\alpha}\right)"
                        r"\left(1-\frac{m_\pi}{x\,E_p}\right)^{1/2}.",
                        r"B_\pi=5.58+0.78\,L+0.10\,L^2\,,",
                        r"r=\frac{3.1}{B_\pi^{3/2}}\,,\quad \alpha=\frac{0.89}{B_\pi^{1/2}\left(1-e^{-0.33\,B_\pi}\right)}\,,",
                        r"where $L=\ln(E_p/1\,{\rm TeV})$"],
                   f=[r"4\,\alpha B_\pi x^{\alpha-1}\left(\frac{1-x^\alpha}{(1+r\,x^\alpha)^3}\right)^4",
                      r"\left(\frac1{1-x^\alpha}+\frac{3\,r}{1+rx^\alpha}\right)"],
                   params=[(r"B_\pi", r"5.58+0.78\,L+0.10\,L^2"), ("r", r"\frac{3.1}{B_\pi^{3/2}}"),
                           (r"\alpha", r"\frac{0.89}{B_\pi^{1/2}\left(1-e^{-0.33\,B_\pi}\right)}"),
                           ("L", r"\ln(E_p/1\,{\rm TeV})")])}
K_T3_ROWS = [   # Koers et al.'s Table III (columns pi+, pi-, pi0, K+, K-, K0, anti-K0): the README has the last five
    r"$p_{00}$&$-0.497$&$-0.501$&$-0.456$&$-1.23$&$-1.46$&$-1.29$&$-1.50$\\",
    r"$p_{01}$&$0.0934$&$0.0934$&$0.0950$&$0.0657$&$0.101$&$0.0690$&$0.101$\\",
    r"$p_{10}$&$-0.140$&$-0.128$&$-0.142$&$-0.147$&$-0.109$&$-0.142$&$-0.118$\\",
    r"$p_{11}$&$0.0131$&$0.0118$&$0.0135$&$0.0161$&$0.00865$&$0.0154$&$0.0101$\\",
    r"$p_{20}$&$-0.455$&$-0.437$&$-0.457$&$-0.00411$&$-0.0577$&$-0.00717$&$-0.0567$\\",
    r"$p_{21}$&$0.495$&$0.494$&$0.494$&$0.493$&$0.491$&$0.493$&$0.491$\\",
    r"$p_{30}$&$-2.06$&$-0.945$&$-1.49$&$-0.989$&$-1.22$&$-1.03$&$-1.25$\\",
    r"$p_{40}$&$-0.896$&$-1.03$&$-0.981$&$-0.345$&$-0.164$&$-0.294$&$-0.169$\\",
    r"$p_{50}$&$1.11$&$0.963$&$1.01$&$0.777$&$1.04$&$0.839$&$1.05$\\",
    r"$p_{60}$&$0.791$&$0.598$&$0.723$&$-0.235$&$-0.279$&$-0.272$&$-0.272$\\",
    r"$p_{70}$&$37.7$&$15.3$&$22.1$&$42.7$&$18.6$&$33.8$&$21.2$\\",
    r"$p_{80}$&$7.69$&$7.23$&$8.53$&$12.0$&$4.23$&$10.1$&$4.07$\\"]
K_T1 = (r"&$N_p$&$N_s$&$\gamma$&${\pi^+}$&${\pi^-}$&${\pi^0}$&${K^+}$&${K^-}$&${K^0}$&${\overline{K^0}}$\\",
        r"$f$&0.56&0.033&0.013&0.13&0.095&0.12&0.016&0.011&0.014&0.011\\")   # the energy fractions (Table I)
K_EQ15 = r"\mathcal{M}_{i} = c_0 + c_1 \log s + c_2 \log^2 s"
K_T2 = [r"$c_0$&$4.5$&$3.8$&$4.9$&$0.49$&$0.32$&$0.36$&$0.29$\\",       # columns as in Table III
        r"$c_1$&$-1.7$&$-1.7$&$-2.1$&$-0.23$&$-0.20$&$-0.17$&$-0.18$\\",
        r"$c_2$&$0.50$&$0.50$&$0.58$&$0.063$&$0.060$&$0.054$&$0.054$\\"]
K_COLUMNS = ("pi+", "pi-", "pi0", "K+", "K-", "K0", "K0b")
ISR = r"\cite{albrow2,albrow3,albrow4,albrow5,capi,alper,guettler}"
NA49_T11 = [   # Anticic et al.'s Table 11: ref., sqrt(s) [GeV], F(0) [mb] and <n> of K+, K-, K0_S; (...) derived
    r"\cite{louttit}&2.9&&&&0.00462&&0.00082\\", r"\cite{hogan,reed}&2.9&0.042&&&0.00481&&\\",
    r"\cite{alex}&3.45&&&&0.00802&&0.00294\\", r"\cite{eisner}&3.59&&&&&&0.00670\\",
    r"\cite{fire}&4.04&&&&0.01760&0.00080&0.00719\\", r"\cite{blobel}&4.9&&&0.120&&&0.0190\\",
    r"\cite{fesef}&4.9&&&&0.0473&0.00747&0.0198\\", r"\cite{boggild}&6.1&&&0.185&&&0.0420\\",
    r"\cite{blobel}&6.8&&&0.206&&&0.0410\\", r"\cite{fesef}&6.8&&&&0.0999&0.0330&0.0493\\",
    r"\cite{allaby1}&6.84&0.440&0.120&(0.280)&0.107&0.0262&(0.0666)\\", r"\cite{bogo}&7.8&&&0.300&&&0.0636\\",
    r"\cite{abramov}&11.5&0.549&0.322&(0.435)&&&\\", r"\cite{ammosov}&11.5&&&0.375&&&0.109\\",
    r"\cite{alston}&13.8&&&&&&0.121\\", r"\cite{chapman}&13.9&&&0.505&&&0.146\\",
    r"\cite{brick}&16.7&&&0.490&&&0.158\\", r"NA49&17.2&0.672&0.477&(0.575)&0.227&0.130&(0.179)\\",
    r"\cite{jaeger2}&19.7&&&0.590&&&0.181\\", ISR + r"&23.0&0.718&0.547&(0.633)&0.273&0.171&(0.222)\\",
    r"\cite{sheng}&23.8&&&0.670&&&0.224\\", r"\cite{lopinto}&23.8&&&&&&0.212\\",
    r"\cite{bailly}&25.7&&&0.670&&&0.262\\", r"\cite{kass}&27.4&&&&&&0.200\\",
    r"\cite{kichimi}&27.6&&&0.680&&&0.232\\", ISR + r"&31.0&0.767&0.614&(0.691)&0.327&0.220&(0.274)\\",
    ISR + r"&45.0&0.861&0.714&(0.788)&0.409&0.290&(0.350)\\", ISR + r"&52.0&0.907&0.766&(0.837)&0.448&0.328&(0.388)\\",
    ISR + r"&63.0&0.959&0.811&(0.885)&0.493&0.363&(0.428)\\", ISR + r"&200.0&1.361&1.192&(1.277)&0.819&0.651&(0.735)\\",
    r"\cite{ansorge1}&200.0&&&1.680&&&0.700\\", r"\cite{ansorge1}&546.0&&&2.306&&&1.000\\"]
SPECIES = {r"$\mathrm{K}^+$": "K+", r"$\mathrm{K}^-$": "K-", r"$\mathrm{K}^0$": "K0", r"$\bar{\mathrm{K}}^0$": "K0b",
           r"$\pi^0$": "pi0"}
CHANNELS = {r"$\mathrm{pp} \to \mathrm{p}\Lambda\mathrm{K}^+$": ("p", "Lambda", "K+"),   # the README's fit table
            r"$\mathrm{pp} \to \mathrm{ppK}^+\mathrm{K}^-$": ("p", "p", "K+", "K-"),
            r"$\mathrm{pp} \to \mathrm{p}\Sigma^+\mathrm{K}^0$": ("p", "Sigma+", "K0")}
SUM_K = (("K+", 1), ("K-", 1), ("K0S", 2))   # sum_K kappa_K: K0_S and K0_L alike
FIT_TABLE = r"| Kaon | $A$ | $a$ | $b$ | $\sqrt{s_\mathrm{th}}$ [GeV] | Threshold channel |"   # the header of the README's multiplicity fit table


def na49_table():
    """[(ref., sqrt(s), {"K+": <n>, "K-": <n>, "K0S": <n>})] of Table 11, measured values only (not the bracketed)."""
    out = []
    for row in NA49_T11:
        c = row[:-2].split("&")
        n = {k: float(v) for k, v in zip(("K+", "K-", "K0S"), c[5:]) if v and not v.startswith("(")}
        out.append((c[0], float(c[1]), n))
    return out


def readme_params():
    """{left side: right side} of the README's '$a = b$' spans before 'and $n = 0$ outside' (eps_3, eps_5, p_0 ... p_6)."""
    line = L.line(r"with $\epsilon_3 = 10^{p_2+p_4}$")
    spans = re.findall(r"\$([^$]+?) = ([^$]+?)\$", line.split(r", and $n = 0$")[0])
    return {lhs: L.to_sympy(rhs, TOK_K) for lhs, rhs in spans}


def readme_table3():
    """(the twelve parameter names, {row label: values}) of the README's Table III."""
    head = L.line(r"| | $p_{00}$ | $p_{01}$ |")
    block = L.text()[L.text().index(head):].split("\n\n")[0].splitlines()[2:]      # the rows below |---|
    rows = {cells[0]: [sp.Rational(v.replace("−", "-")) for v in cells[1:]] for cells in
            ([cl.strip() for cl in ln.strip().strip("|").split("|")] for ln in block)}
    return re.findall(r"\$(p_\{\d\d\})\$", head), rows


def readme_table(head, file="README"):
    """The cells of the rows of the README table whose header line holds `head`."""
    block = L.text(file)[L.text(file).index(L.line(head, file)):].split("\n\n")[0].splitlines()[2:]
    return [[cl.strip() for cl in ln.strip().strip("|").split("|")] for ln in block]


@lru_cache(None)
def pp_setup():
    """The README's pp kaons as functions: n {species: n(eps, E_p)}, F-hat {model: (F_pi without, with its threshold
    factor)}, the fit {species: f(E_p)}, gamma_cm(E_p), and E*, x_min, E_t (m_p of PDG 2026)."""
    first = readme_koers(r"n(\epsilon) = n_0\thinspace \epsilon^{-1}")
    second = readme_koers(r"\beta = 10^{-(\epsilon/\epsilon_3)^{p_{30}}}")
    expr = first[r"n(\epsilon)"]
    for name, f in KFUNCS.items():
        expr = expr.replace(f, lambda a, part=(first if name == r"\alpha" else second)[name]: part.subs(eps, a))
    defs = {TOK_K["n_0"]: first["n_0"], **{TOK_K[k]: v for k, v in readme_params().items()}}
    for _ in range(4):        # eps_3 = 10^(p_2 + p_4), p_4 = p_40 - p_2, p_2 = p_20 + p_21 log E_p
        expr = expr.xreplace(defs)
    expr = expr.replace(sp.log, lambda a: sp.log(a) / sp.log(10))       # the README's log is log10 (pp.koers_n)
    assert expr.free_symbols <= {eps, Ep, mKs} | set(P.values()), expr.free_symbols
    names, rows = readme_table3()
    columns = {SPECIES[label]: values for label, values in rows.items()}
    for row in K_T3_ROWS:                                                # Koers et al.'s pi+ and pi- (TeX)
        values = re.findall(r"\$(-?[\d.]+)\$", row)
        columns.setdefault("pi+", []).append(sp.Rational(values[0]))
        columns.setdefault("pi-", []).append(sp.Rational(values[1]))
    assert [re.match(r"\$(p_\{\d\d\})\$", row).group(1) for row in K_T3_ROWS] == names
    n = {}
    for species, values in columns.items():
        f = sp.lambdify((eps, Ep), expr.subs({mKs: MASS[species], **{P[nm]: sp.Float(v) for nm, v in zip(names, values)}}),
                        "math")
        n[species] = (lambda f, mass: lambda e, E: f(e, E) if mass < e < E else 0.0)(f, MASS[species])  # n = 0 outside
    fhat = {}
    al, B, r_, a_, Lk, mpi = sp.symbols("alpha B_pi r a L m_pi", positive=True)
    tok = {"x": x, r"\alpha": al, r"B_\pi": B, "r": r_, "a": a_, "L": Lk, r"m_\pi": mpi, "E_p": Ep, r"{\rm TeV}": 1000}
    for model, d in KEL.items():
        assert all(any(p in t for t in d["tex"]) for p in d["f"] + [KEL_THR] + [p for _, p in d["params"]])
        f = sp.Mul(*[L.to_sympy(p.replace(r"\frac1{", r"\frac{1}{"), tok) for p in d["f"]])
        defs = {tok[name]: L.to_sympy(p, tok) for name, p in d["params"]}
        for _ in range(4):
            f = f.xreplace(defs)
        assert f.free_symbols == {x, Ep}, f.free_symbols
        thr = sp.lambdify((x, Ep, mpi), L.to_sympy(KEL_THR, tok), "math")
        fh = sp.lambdify((x, Ep), f, "math")
        fhat[model] = (fh, (lambda fh, thr: lambda x_, E, m: fh(x_, E) * thr(x_, E, m) if x_ * E > m else 0.0)(fh, thr))
    disp = L.display(r"\tilde n_K(E_p) = \langle n_K\rangle_{\rm KPW}(E_t)")
    A_, b_, s_, sth = sp.symbols("A b s s_th", positive=True)
    fit_tex, s_tex = L.sides(L.part(disp, "f(s) ="))[1], L.sides(L.part(disp, "s ="))[1]
    fit_f = sp.lambdify((s_, A_, a_, b_, sth), L.to_sympy(fit_tex, {"A": A_, "a": a_, "b": b_, "s": s_, r"s_{\rm th}": sth}))
    mp_ = sp.Symbol("m_p", positive=True)
    s_mp = sp.lambdify((Ep, mp_), L.to_sympy(s_tex, {"m_p": mp_, "E_p": Ep}))   # m_p an argument: s in floating point
    s_of = lambda E: s_mp(E, MASS["p"])   # noqa: E731
    fit = {}
    for cells in readme_table(FIT_TABLE):
        species = "K0S" if cells[0].startswith(r"$\mathrm{K}^0_\mathrm{S}$") else SPECIES[cells[0]]
        A, a, b, rth = (float(v) for v in cells[1:5])
        fit[species] = (lambda A, a, b, st: lambda E: fit_f(s_of(E), A, a, b, st) if s_of(E) > st else 0.0)(A, a, b, rth**2)
    g_tex = re.search(r"\$\\gamma_\\mathrm\{cm\} = ([^$]+)\$ \$= ", L.line(r"; $\gamma_\mathrm{cm} = ")).group(1)
    gcm = sp.lambdify(Ep, L.to_sympy(g_tex, {"E_p": Ep, "m_p": MASS["p"]}))
    L.inline(r"E_K = \max(K_K T_p, \gamma_\mathrm{cm} m_K)")   # as delta() takes it
    xmin = float(L.to_sympy(re.fullmatch(r"x_s = \\max\(x, (.+)\)",
                                         L.part(L.display(r"F_K(x,E_p) = R(x)\thinspace \hat F_\pi(x,E_p)"), "x_s =")).group(1), {}))
    es = float(L.to_sympy(L.sides(L.inline(r"E^\ast = 10^6"))[1], {}))
    et = float(re.search(r"E_t = (\d+)", L.inline(r"\boldsymbol{E_p \lt E_t = 100}")).group(1))
    return n, fhat, fit, gcm, es, xmin, et


def F_K(species, x_, E, model):
    """The README's F_K(x, E_p) for E_p >= E_t: Koers et al.'s E_p n(x E_p) up to E*, above it R(x) F-hat_pi(x, E_p)
    with x_s = max(x, x_min) and, below x_min, R following rho = n_K/n_pi0 at E* in the kaon energy."""
    if species == "K0S":
        return (F_K("K0", x_, E, model) + F_K("K0b", x_, E, model)) / 2
    n, fhat, _, _, es, xmin, _ = pp_setup()
    if E <= es:
        return E * n[species](x_ * E, E)
    f = fhat[model][0]
    xs = max(x_, xmin)
    R = es * n[species](xs * es, es) / f(xs, es)
    if x_ < xmin:
        rho = lambda e: n[species](e, es) / n["pi0"](e, es) if n["pi0"](e, es) > 0 else 0.0   # noqa: E731
        R *= rho(min(x_ * E, xmin * es)) / rho(xmin * es)
    return R * f(x_, E)


def moments(species, E, model="SIBYLL"):
    """(kaons per inelastic collision, the fraction of E_p they carry) = (int F_K dx, int x F_K dx)."""
    return _moments(species, E, model if E > pp_setup()[4] else "SIBYLL")


@lru_cache(None)
def _moments(species, E, model):
    xmin = pp_setup()[5]
    lo = math.log(MASS[species] / E)
    pts = [math.log(xmin)] if E > pp_setup()[4] and math.log(xmin) > lo else None
    f = lambda lx: F_K(species, math.exp(lx), E, model) * math.exp(lx)   # noqa: E731
    return (quad(f, lo, 0, limit=400, points=pts)[0],
            quad(lambda lx: f(lx) * math.exp(lx), lo, 0, limit=400, points=pts)[0])


def kappa_K(E, model="SIBYLL"):
    """sum_K kappa_K: the fraction of E_p all kaons carry (above E_t), or of the delta-functional approximation below."""
    if E < pp_setup()[6]:
        return sum(w * nt * EK / E for species, w in SUM_K for nt, EK in [delta(species, E)])
    return sum(w * moments(species, E, model)[1] for species, w in SUM_K)


@lru_cache(None)
def kappa_pi(E, model):
    """The fraction of E_p that AM3's pions carry: Kelner et al.'s F_pi (with its threshold factor) for pi+, pi- and pi0."""
    f = pp_setup()[1][model][1]
    return sum(quad(lambda lx: f(math.exp(lx), E, MASS[pi]) * math.exp(2 * lx), math.log(MASS[pi] / E), 0, limit=400)[0]
               for pi in ("pi+", "pi-", "pi0"))


def delta(species, E):
    """(n-tilde_K, E_K) of the README's delta-functional approximation at E_p = E < E_t (E_K as pp_setup asserts it)."""
    _, _, fit, gcm, _, _, et = pp_setup()
    N, kappa = moments(species, et)
    K_K = kappa * et / N / (et - MASS["p"])      # the mean fraction of T_p per kaon at E_t
    return N * fit[species](E) / fit[species](et), max(K_K * (E - MASS["p"]), gcm(E) * MASS[species])


def E_of_sqrt_s(rs):
    """E_p [GeV] of a fixed-target collision at sqrt(s) (the README's s = 2m_p^2 + 2m_p E_p)."""
    return (rs**2 - 2 * MASS["p"]**2) / (2 * MASS["p"])


def percent(span):
    """The numbers of a README span "13\\text{–}16\\text{％}" (or "5.2\\text{％}"), as strings."""
    return span.replace(r"\text{％}", "").split(r"\text{–}")


def control_result(what, compare, stated, old, new, file="checks"):
    """The comparison with one stated number broken must fail."""
    with L.patched(old, new, file):
        broken = compare(stated())
    return Result(f"control: {what}", not all(r.ok for r in broken), "fails, as it must")


@derivation("pp.table")
def check_pp_table():
    """The table of kaons per inelastic pp collision (Results): the README's spectra integrated (SIBYLL above E*)."""
    pin = r"Kaons per inelastic $\mathrm{pp}$ collision and the fraction of the proton energy they carry"
    es = pp_setup()[4]

    def stated():
        t = L.text("checks")
        block = t[t.index(L.line(pin, "checks")):].split("\n\n")[1].splitlines()
        assert block[0].startswith(r"| $E_p$ | $\langle \mathrm{K}^+\rangle$ | $\langle \mathrm{K}^-\rangle$ |"), block[0]
        rows = []
        for ln in block[2:]:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            e = re.fullmatch(r"(?:(\d+)|\$10\^\{?(\d+)\}?\$) GeV( \(extension, SIBYLL\))?", cells[0])
            rows.append((float(e.group(1)) if e.group(1) else 10.0 ** int(e.group(2)), bool(e.group(3)), cells[1:]))
        return rows

    def compare(rows):
        out = []
        for E, ext, cells in rows:
            got = [moments(s, E)[0] for s in ("K+", "K-", "K0S")] + [kappa_K(E)]
            out.append(Result(f"E_p = {E:g} GeV: <K+>, <K->, <K0_S>, sum kappa_K = {', '.join(cells)}",
                              all(rounds_to(g, c) for g, c in zip(got, cells)) and ext == (E > es),
                              ", ".join(f"{g:.4f}" for g in got)))
        return out
    return compare(stated()) + [control_result("0.388 for <K+> = 0.378 at 1e3 GeV", compare, stated,
                                               r"| $10^3$ GeV | 0.378 |", r"| $10^3$ GeV | 0.388 |")]




def koers_eq15():
    """Koers et al.'s multiplicity fit, eq. 15 with Table II: M(species, E_p) (their log is log10, s in GeV^2)."""
    S, c0, c1, c2 = sp.symbols("s c_0 c_1 c_2", positive=True)
    m15 = L.to_sympy(K_EQ15.split("=")[1].replace(r"\log^2 s", r"(\log s)^2"), {"c_0": c0, "c_1": c1, "c_2": c2, "s": S})
    m15 = sp.lambdify((S, c0, c1, c2), m15.replace(sp.log, lambda a: sp.log(a) / sp.log(10)), "math")
    rows = {row.split("&")[0]: [float(v) for v in re.findall(r"\$(-?[\d.]+)\$", row)] for row in K_T2}
    k = lambda species: K_COLUMNS.index(species)   # noqa: E731
    return lambda species, E: m15(2 * MASS["p"]**2 + 2 * MASS["p"] * E, *(rows[f"$c_{i}$"][k(species)] for i in range(3)))


@derivation("pp.kpw_range")
def check_pp_kpw_range():
    """Koers et al.'s spectra from 100 to 1e6 GeV: against their own multiplicity fit (eq. 15) and NA49's Table 11."""
    et = pp_setup()[6]
    m15 = koers_eq15()
    grid = [10.0 ** (3 + k / 4) for k in range(13)]
    dev15 = {E: max(abs(moments(s, E)[0] / m15(s, E) - 1) for s in ("K+", "K-", "K0", "K0b")) for E in grid}
    table = na49_table()
    na49 = next(n for ref, rs, n in table if ref == "NA49")

    def deviation(kaons):   # the largest |Koers / Table 11 - 1| at E_p >= E_t, and the sqrt(s) of these rows
        devs = [(abs(moments(k, E_of_sqrt_s(rs))[0] / v - 1), rs) for ref, rs, n in table for k, v in n.items()
                if k in kaons and E_of_sqrt_s(rs) >= et]
        return max(d for d, _ in devs), min(r for _, r in devs), max(r for _, r in devs)
    dev_kpm, dev_k0s = deviation(("K+", "K-")), deviation(("K0S",))

    def stated():
        line = L.line("At 158 GeV their spectra give", "checks")
        a = re.search(r"their own multiplicity fit \(their eq\. 15\) within \$(\d+)\\text\{％\}\$, and within "
                      r"\$(\d+)\\text\{％\}\$ from \$10\^(\d)\$ GeV on", line).groups()
        b = re.search(r"At (\d+) GeV their spectra give \$\\langle \\mathrm\{K\}\^\+\\rangle = ([\d.]+)\$ and "
                      r"\$\\langle \\mathrm\{K\}\^-\\rangle = ([\d.]+)\$ per inelastic collision, against ([\d.]+) and "
                      r"([\d.]+) measured by NA49", line).groups()
        c = re.search(r"deviate by up to \$(\d+)\\text\{％\}\$ from the \$\\mathrm\{K\}\^\\pm\$ multiplicities", line).group(1)
        d = re.search(r"deviate by up to \$(\d+)\\text\{％\}\$ from the compilation's \$\\mathrm\{K\}\^0_\\mathrm\{S\}\$ "
                      r"multiplicities \(\$\\sqrt\{s\} = (\d+)\\text\{–\}(\d+)\$ GeV\)", line).groups()
        return a, b, c, d

    def compare(st):
        (all15, from15, e15), (e158, kp, km, kp_m, km_m), c, (d, lo, hi) = st
        late = max(v for E, v in dev15.items() if E >= 10.0 ** int(e15) * (1 - 1e-9))
        n158 = [moments(k, float(e158))[0] for k in ("K+", "K-")]
        return [Result(f"their spectra integrate to their eq. 15 within {all15}% (1e3-1e6 GeV), {from15}% from 1e{e15} GeV",
                       rounds_to(100 * max(dev15.values()), all15) and rounds_to(100 * late, from15),
                       f"{100 * max(dev15.values()):.2f}% and {100 * late:.2f}% (4 energies per decade; K+, K-, K0, anti-K0)"),
                Result(f"<K+> = {kp}, <K-> = {km} at {e158} GeV", rounds_to(n158[0], kp) and rounds_to(n158[1], km),
                       f"{n158[0]:.4f}, {n158[1]:.4f}"),
                Result(f"NA49 measured {kp_m} and {km_m} (Table 11, sqrt(s) = 17.2 GeV)",
                       (float(kp_m), float(km_m)) == (na49["K+"], na49["K-"]), f"{na49['K+']}, {na49['K-']}"),
                Result(f"K+- of Table 11 (E_p >= E_t): up to {c}%", rounds_to(100 * dev_kpm[0], c), f"{100 * dev_kpm[0]:.2f}%"),
                Result(f"K0_S of Table 11 (E_p >= E_t, sqrt(s) = {lo}-{hi} GeV): up to {d}%",
                       rounds_to(100 * dev_k0s[0], d) and rounds_to(dev_k0s[1], lo) and rounds_to(dev_k0s[2], hi),
                       f"{100 * dev_k0s[0]:.2f}%, sqrt(s) = {dev_k0s[1]}-{dev_k0s[2]} GeV")]
    out = [in_source("Koers et al.'s eq. 15 and Table II as transcribed", KOERS, K_EQ15, "".join(K_T2)),
           in_source("NA49's Table 11 as transcribed", NA49, "".join(NA49_T11))]
    return out + compare(stated()) + [control_result("0.208 for <K+> = 0.198 at 158 GeV", compare, stated,
                                                     r"\langle \mathrm{K}^+\rangle = 0.198$", r"\langle \mathrm{K}^+\rangle = 0.208$")]


@derivation("pp.scaling")
def check_pp_scaling():
    """Above 1e6 GeV: Koers et al.'s pion energy fraction, their K/pi0 ratios at fixed x, the continuity at E*."""
    n, _, _, _, es, _, _ = pp_setup()

    def pion_fraction(E):
        return sum(quad(lambda lx: E * n[pi](math.exp(lx) * E, E) * math.exp(2 * lx), math.log(MASS[pi] / E), 0,
                        limit=400)[0] for pi in ("pi+", "pi-", "pi0"))
    one = math.exp(brentq(lambda le: pion_fraction(math.exp(le)) - 1, math.log(1e9), math.log(1e10), xtol=1e-6))
    ratio = lambda k, x_, E: n[k](x_ * E, E) / n["pi0"](x_ * E, E)   # noqa: E731
    grid = [10.0 ** (3 + k / 8) for k in range(25)]                   # 1e3 ... 1e6 GeV
    above = es * (1 + 1e-7)
    jump = {model: {k: [moments(k, above, model)[i] / moments(k, es)[i] - 1 for i in (0, 1)] for k in ("K+", "K-", "K0S")}
            for model in KEL}
    kappa_jump = {model: kappa_K(above, model) / kappa_K(es) - 1 for model in KEL}

    def stated():
        line = L.line(r"(3.5 at $10^{11}$ GeV)", "checks")
        a = re.search(r"reaches 1 at about \$([\d.]+) \\times 10\^(\d+)\$ GeV \(([\d.]+) at \$10\^\{(\d+)\}\$ GeV\)", line).groups()
        b = re.search(r"\\mathrm\{K\}\^\+/\\pi\^0 = ([\d.]+)\\text\{–\}([\d.]+)\$ at \$x = ([\d.]+)\$ and \$([\d.]+)\\text\{–\}"
                      r"([\d.]+)\$ at \$x = ([\d.]+)\$ from \$10\^(\d)\$ to \$10\^(\d)\$ GeV", line).groups()
        c = re.search(r"ratios still rise by \$(\d+)\\text\{–\}(\d+)\\text\{％\}\$ at \$x = ([\d.]+)\$ and fall by "
                      r"\$(\d+)\\text\{–\}(\d+)\\text\{％\}\$ at \$x = ([\d.]+)\$", line).groups()
        d = re.search(r"multiplicity drops by \$([\d.]+)\\text\{–\}([\d.]+)\\text\{％\}\$ \(with QGSJET at most "
                      r"\$([\d.]+)\\text\{％\}\$\)", line).groups()
        e = re.search(r"reaches 1 at about \$([\d.]+) \\times 10\^(\d+)\$ GeV\.",
                      L.line("Its parameters drift so that the energy fraction of its pions")).groups()
        return a, b, c, d, e

    def compare(st):
        (m1, p1, f2, p2), (lo1, hi1, x1, lo2, hi2, x2, e_lo, e_hi), (r1, r2, xr, f1_, f2_, xf), (d1, d2, dq), main = st
        out = [Result(f"Koers et al.'s pion energy fraction is 1 at about {m1}e{p1} GeV (README: {main[0]}e{main[1]}), "
                      f"{f2} at 1e{p2} GeV", rounds_to(one / 10.0 ** int(p1), m1) and (m1, p1) == main
                      and rounds_to(pion_fraction(10.0 ** int(p2)), f2),
                      f"1 at {one:.4g} GeV, {pion_fraction(10.0 ** int(p2)):.4f} at 1e{p2} GeV")]
        span = [E for E in grid if 10.0 ** int(e_lo) * (1 - 1e-9) <= E <= 10.0 ** int(e_hi) * (1 + 1e-9)]
        for lo, hi, xx in ((lo1, hi1, x1), (lo2, hi2, x2)):
            v = [ratio("K+", float(xx), E) for E in span]
            out.append(Result(f"K+/pi0 = {lo}-{hi} at x = {xx}, 1e{e_lo}-1e{e_hi} GeV", rounds_to(min(v), lo) and rounds_to(max(v), hi),
                              f"{min(v):.4f}-{max(v):.4f} ({len(span)} energies)"))
        for a, b, xx, sign in ((r1, r2, xr, 1), (f1_, f2_, xf, -1)):
            ch = [100 * sign * (ratio(k, float(xx), 10.0 ** int(e_hi)) / ratio(k, float(xx), 10.0 ** int(e_lo)) - 1) for k in ("K-", "K0b")]
            out.append(Result(f"K-/pi0 and anti-K0/pi0 {'rise' if sign > 0 else 'fall'} by {a}-{b}% at x = {xx}",
                              rounds_to(min(ch), a) and rounds_to(max(ch), b), f"{ch[0]:.2f}%, {ch[1]:.2f}%"))
        drops = {model: [-100 * j[k][0] for k in j] for model, j in jump.items()}
        steps = [abs(v[1]) for j in jump.values() for v in j.values()] + [abs(v) for v in kappa_jump.values()]
        out += [Result("the energy fraction of the kaons is continuous at E* (each kind and all, SIBYLL and QGSJET)",
                       max(steps) < 1e-4, f"steps of at most {max(steps):.1e} of it"),
                Result(f"their multiplicity drops by {d1}-{d2}% (SIBYLL), at most {dq}% (QGSJET) at E* (K+, K-, K0_S)",
                       rounds_to(min(drops["SIBYLL"]), d1) and rounds_to(max(drops["SIBYLL"]), d2)
                       and rounds_to(max(drops["QGSJET"]), dq) and min(drops["QGSJET"]) > 0,
                       f"SIBYLL {', '.join(f'{v:.3f}' for v in drops['SIBYLL'])}%, QGSJET "
                       f"{', '.join(f'{v:.3f}' for v in drops['QGSJET'])}%")]
        return out
    return compare(stated()) + [control_result("0.133-0.136 for K+/pi0 = 0.123-0.126", compare, stated,
                                               r"= 0.123\text{–}0.126$", r"= 0.133\text{–}0.136$")]


@derivation("pp.below")
def check_pp_below():
    """Below 100 GeV: the README's multiplicity fit against NA49's Table 11, its normalisation and the kaons it gives."""
    _, _, fit, _, _, _, et = pp_setup()
    table = na49_table()
    fesef = {k: sorted(100 * (n[k] / fit[k](E_of_sqrt_s(rs)) - 1) for ref, rs, n in table if ref == r"\cite{fesef}")
             for k in ("K+", "K-")}
    low = sorted(n["K+"] / fit["K+"](E_of_sqrt_s(rs)) for ref, rs, n in table if rs < 3 and "K+" in n)
    rms = {k: math.sqrt(sum(math.log(fit[k](E_of_sqrt_s(rs)) / v) ** 2 for rs, v in pts) / len(pts))
           for k, pts in fit_points().items()}
    norm = {k: moments(k, et)[0] / fit[k](et) for k in fit}
    na49 = next(n for ref, rs, n in table if ref == "NA49")

    def most(E):        # the most kaons of a kind below E (the fit rises with s)
        return max(delta(k, e)[0] for k in fit for e in (E * (1 - 1e-9) * (j / 400) ** 0.5 for j in range(1, 401)))

    def stated():
        line = L.line("The normalisation to Koers et al. at 100 GeV scales the fitted", "checks")
        return (percent(re.search(r"Their \$\\mathrm\{K\}\^-\$ points lie \$(.+?)\$ above it", line).group(1)),
                percent(re.search(r"and their \$\\mathrm\{K\}\^\+\$ points \$(.+?)\$\.", line).group(1)),
                re.search(r"points at \$\\sqrt\{s\} = [\d.]+\$ GeV, which lie ([\d.]+) and ([\d.]+) times above", line).groups(),
                re.search(r"remaining points to \$(\d+)\\text\{％\}\$ \(\$\\mathrm\{K\}\^\+\$\), \$(\d+)\\text\{％\}\$ "
                          r"\(\$\\mathrm\{K\}\^-\$\) and \$(\d+)\\text\{％\}\$ \(\$\\mathrm\{K\}\^0_\\mathrm\{S\}\$\) rms", line).groups(),
                re.search(r"multiplicity by ([\d.]+) \(\$\\mathrm\{K\}\^-\$ by ([\d.]+), \$\\mathrm\{K\}\^0_\\mathrm\{S\}\$ "
                          r"by ([\d.]+)\)", line).groups(),
                re.search(r"Below (\d+) GeV a \$\\mathrm\{pp\}\$ collision makes fewer than ([\d.]+) kaons of each kind, "
                          r"and below (\d+) GeV fewer than ([\d.]+)\.", line).groups(),
                re.search(r"within \$(\d+)\\text\{％\}\$ at \$\\sqrt\{s\} = (\d+)\$ GeV \(\$\\mathrm\{K\}\^\\pm\$, (\d+) GeV\)",
                          line).groups())

    def compare(st):
        km, kp, (t1, t2), (r1, r2, r3), (n1, n2, n3), (e1, b1, e2, b2), (w, rs, e158) = st
        dev158 = max(abs(moments(k, float(e158))[0] / na49[k] - 1) for k in ("K+", "K-"))
        return [Result(f"Fesefeldt et al.'s K- {km[0]}-{km[1]}% and K+ {kp[0]}-{kp[1]}% above the fit (left out of it)",
                       all(rounds_to(v, s_) for v, s_ in zip(fesef["K-"] + fesef["K+"], km + kp)),
                       f"K- {fesef['K-'][0]:.1f}, {fesef['K-'][1]:.1f}%; K+ {fesef['K+'][0]:.1f}, {fesef['K+'][1]:.1f}%"),
                Result(f"the K+ points at sqrt(s) = 2.9 GeV: {t1} and {t2} times the fit (left out of it)",
                       rounds_to(low[0], t1) and rounds_to(low[1], t2), f"{low[0]:.3f}, {low[1]:.3f}"),
                Result(f"the fit reproduces its points to {r1}, {r2}, {r3}% rms in ln n (K+, K-, K0_S)",
                       all(rounds_to(100 * rms[k], v) for k, v in zip(("K+", "K-", "K0S"), (r1, r2, r3))),
                       ", ".join(f"{100 * rms[k]:.2f}% ({len(fit_points()[k])} points)" for k in ("K+", "K-", "K0S"))),
                Result(f"normalised to Koers et al. at E_t: x {n1}, {n2}, {n3} (K+, K-, K0_S)",
                       all(rounds_to(norm[k], v) for k, v in zip(("K+", "K-", "K0S"), (n1, n2, n3))),
                       ", ".join(f"{norm[k]:.4f}" for k in ("K+", "K-", "K0S"))),
                Result(f"fewer than {b1} kaons of each kind below {e1} GeV, {b2} below {e2} GeV",
                       most(float(e1)) < float(b1) and most(float(e2)) < float(b2), f"{most(float(e1)):.4f}, {most(float(e2)):.4f}"),
                Result(f"Koers et al. within {w}% of NA49 at sqrt(s) = {rs} GeV ({e158} GeV; K+-)",
                       100 * dev158 < float(w) and rounds_to(100 * dev158, w), f"{100 * dev158:.2f}%")]
    return compare(stated()) + [control_result("0.91 for the K+ factor 0.81", compare, stated,
                                               r"$\mathrm{K}^+$ multiplicity by 0.81", r"$\mathrm{K}^+$ multiplicity by 0.91")]


def fit_points():
    """{kind: [(sqrt(s), <n>)]}: the points of Table 11 the multiplicity fit takes. These are the measured values with
    sqrt(s) <= 63 GeV (README), without Fesefeldt et al.'s charged kaons and the K+ at sqrt(s) = 2.9 GeV (Results)."""
    top = float(re.fullmatch(r"\\sqrt\{s\} \\le (\d+)", L.inline(r"\sqrt{s} \le 63")).group(1))
    return {k: [(rs, n[k]) for ref, rs, n in na49_table() if k in n and rs <= top
                and not (ref == r"\cite{fesef}" and k != "K0S") and not (k == "K+" and rs < 3)] for k in ("K+", "K-", "K0S")}


@derivation("pp.fit")
def check_pp_fit():
    """The README's multiplicity fit: least squares in ln n to NA49's Table 11 at the thresholds of its channels."""
    rows = readme_table(FIT_TABLE)
    out = [in_source("NA49's Table 11 as transcribed", NA49, "".join(NA49_T11))]
    kinds = {"K+": ("K+",), "K-": ("K-",), "K0S": ("K0", "K0b")}
    for cells in rows:
        kind = "K0S" if cells[0].startswith(r"$\mathrm{K}^0_\mathrm{S}$") else SPECIES[cells[0]]
        final = CHANNELS[cells[5]]
        rth = sum(MASS[q] for q in final)
        # the lowest channel making the kaon: two baryons and one kaon, or two nucleons and a kaon pair (pions add mass)
        cands = [c for c in final_states(2, NUCLEON, ("Lambda",) + SIGMA, KAON) + final_states(2, NUCLEON, NUCLEON, KAON, ANTIKAON)
                 if any(k in c for k in kinds[kind]) and sum(STRANGENESS.get(q, 0) for q in c) == 0]
        lowest = min(cands, key=lambda c: sum(MASS[q] for q in c))
        sol = np.linalg.lstsq(*fit_system(kind), rcond=None)[0]
        got = (math.exp(sol[0]), sol[1], sol[2])
        out += [Result(f"{kind}: sqrt(s_th) = {cells[4]} GeV is {'+'.join(final)}, the lowest channel that makes it",
                       rounds_to(rth, cells[4]) and sum(CHARGE[q] for q in final) == 2 and sorted(lowest) == sorted(final), f"{rth:.5f} GeV"),
                Result(f"{kind}: A, a, b = {', '.join(cells[1:4])} (least squares in ln n, {len(fit_points()[kind])} points)",
                       all(rounds_to(g, c_) for g, c_ in zip(got, cells[1:4])), ", ".join(f"{g:.6g}" for g in got))]

    def stated():
        return [c[1] for c in readme_table(FIT_TABLE)]
    with L.patched(r"| 0.0942 | 3.435 |", r"| 0.0952 | 3.435 |"):
        broken = stated()
    a_kp = math.exp(np.linalg.lstsq(*fit_system("K+"), rcond=None)[0][0])
    return out + [Result("control: A = 0.0952 for 0.0942 (K+)", not rounds_to(a_kp, broken[0]), "fails, as it must")]


PP_SRC = L.ROOT / "src" / "am3_mojo" / "physics" / "proton_proton.mojo"
T3_ROWS = (r"$\mathrm{K}^+$", r"$\mathrm{K}^-$", r"$\mathrm{K}^0$", r"$\bar{\mathrm{K}}^0$", r"$\pi^0$")   # the rows of _KPW


def code_pp(src):
    """The code's Table III {README row label: 12 values}, its fit {kind: (A, a, b, sqrt(s_th))} and its ranges
    (E_p from, to; x from), exact (decimal literals as rationals)."""
    block = re.search(r"comptime _KPW: Array\[Float64, 60\] = \[(.*?)\]", src, re.S).group(1)
    t3 = [sp.Rational(v) for v in re.findall(r"-?\d+\.\d+", block)]
    assert len(t3) == 60, len(t3)
    fit = re.findall(r"A = ([\d.]+)\s+a = ([\d.]+)\s+b = ([\d.]+)", src[src.index("def pp_kaon_mult_fit"):])[:3]
    th = re.findall(r"\d+\.\d+", src[src.index("def pp_kaon_sqrt_s_th"):src.index("def pp_kaon_mult_fit")])
    assert len(fit) == 3 and len(th) == 3, (fit, th)                    # K+, K-, then the neutral kaons
    ranges = [sp.Rational(re.search(rf"comptime {n} = ([\d.e+-]+)", src).group(1))
              for n in ("KPW_E_MIN", "KPW_E_MAX", "KPW_X_MIN")]
    return ({lab: t3[12 * i:12 * i + 12] for i, lab in enumerate(T3_ROWS)},
            {k: tuple(sp.Rational(v) for v in f) + (sp.Rational(t),) for k, f, t in zip(("K+", "K-", "K0S"), fit, th)},
            ranges)


def code_proton_mass(src):
    """The proton masses of the code's pp kaons (every `var mp = ...` of the source), exact, with the code's constants."""
    named = mojo_constants.read(("massP_PDG",))
    return {named[v] if v in named else sp.Rational(v) for v in re.findall(r"var mp = (\S+)", src)}


@derivation("pp.code_tables")
def check_code_tables():
    """The code has the README's Table III, multiplicity fit and ranges, and PDG 2026's proton mass.

    The code is src/am3_mojo/physics/proton_proton.mojo; the proton mass is that of s = 2 m_p^2 + 2 m_p E_p and of
    the kernels below E_t."""
    def readme():
        rows = {("K0S" if c[0].startswith(r"$\mathrm{K}^0_\mathrm{S}$") else SPECIES[c[0]]):
                tuple(sp.Rational(v) for v in c[1:5]) for c in readme_table(FIT_TABLE)}
        m = re.search(r"\*\*(\d+) GeV \$\\le E_p \\le 10\^(\d+)\$ GeV: Koers et al\. \(2006\)",
                      L.line(r"GeV: Koers et al. (2006), eqs. 16–18 and their Table III.**"))
        x = re.search(r"For \$x \\lt 10\^\{(-\d+)\}\$ \(kaons below", L.line(r"(kaons below 100 GeV at $E^\ast$)"))
        return readme_table3()[1], rows, [sp.Integer(m.group(1)), sp.Integer(10) ** int(m.group(2)),
                                          sp.Integer(10) ** int(x.group(1))]

    def results(src, ours):
        t3, fit, ranges = code_pp(src)
        r3, rfit, rranges = ours
        return [Result("Table III in the code (60 values; K+, K-, K0, anti-K0, pi0) = the README's", t3 == {k: r3[k] for k in T3_ROWS},
                       "exact: equal" if t3 == {k: r3[k] for k in T3_ROWS} else
                       f"differ: {[k for k in T3_ROWS if t3[k] != r3[k]]}"),
                Result("the code's A, a, b and sqrt(s_th) (K+, K-, K0_S) = the README's fit table", fit == rfit,
                       "exact: " + "; ".join(f"{k} {', '.join(map(str, v))}" for k, v in fit.items())),
                Result("the code's E_p range of Koers et al. (100 GeV to 1e6 GeV) and x >= 1e-4 = the README's",
                       ranges == rranges, f"exact: {ranges}")]
    src = PP_SRC.read_text()
    out = results(src, readme())

    def mass(src_):
        masses = code_proton_mass(src_)
        return Result("the code's m_p (the fit's s, the kernels below E_t) = PDG 2026's, in every place",
                      masses == {sp.Rational(str(MASS["p"]))}, f"exact: {', '.join(str(sp.N(v, 12)) for v in masses)} GeV")
    out.append(mass(src))
    out.append(Result("control: the old literal 0.93827208 in one place", not mass(src.replace("var mp = massP_PDG",
                                                                                                "var mp = 0.93827208", 1)).ok,
                      "fails, as it must"))
    broken_src = src.replace("comptime _KPW: Array[Float64, 60] = [\n    -1.23,", "comptime _KPW: Array[Float64, 60] = [\n    -1.24,")
    with L.patched(r"| 0.0942 | 3.435 |", r"| 0.0952 | 3.435 |"):
        broken_readme = results(src, readme())[1]
    return out + [Result("control: p_00 of K+ = -1.24 in the source", broken_src != src and not results(broken_src, readme())[0].ok,
                         "fails, as it must"),
                  Result("control: A = 0.0952 for 0.0942 (K+) in the README", not broken_readme.ok, "fails, as it must")]


def fit_system(kind):
    """(design matrix, ln n) of the least-squares fit of a kind, at the unrounded threshold of its README channel."""
    for cells in readme_table(FIT_TABLE):
        if kind == ("K0S" if cells[0].startswith(r"$\mathrm{K}^0_\mathrm{S}$") else SPECIES[cells[0]]):
            rth = sum(MASS[q] for q in CHANNELS[cells[5]])
    pts = fit_points()[kind]
    s_ = np.array([rs**2 for rs, _ in pts])
    return np.stack([np.ones_like(s_), np.log(1 - rth**2 / s_), np.log(s_ / rth**2)], axis=1), np.log([v for _, v in pts])


@derivation("pp.energy")
def check_pp_energy():
    """The energy fraction of the pp kaons, and its share of that of AM3's pions (SIBYLL, QGSJET), 10 GeV to 4e13 GeV."""
    es = pp_setup()[4]
    grid = sorted([10.0 ** (2 + k / 4) for k in range(41)] + [es * (1 + 1e-7)])   # 100 GeV ... 1e12 GeV
    kap = {m: [kappa_K(E, m) for E in grid] for m in KEL}
    share = {m: [kappa_K(E, m) / kappa_pi(E, m) for E in grid] for m in KEL}
    top = 4e13

    def stated():
        line = L.line(r"*Energy:* from $E_p = 100$ GeV to $10^{12}$ GeV, all kaons together carry", "checks")
        a = re.search(r"carry \$([^$]+)\$ of the proton energy \(\$([^$]+)\$ in Koers et al\.'s Table I\)\. That is \$([^$]+)\$ of the "
                      r"energy", line).groups()
        b = re.search(r"With QGSJET, [^$]+\$10\^6\$ GeV then follows, they are \$([^$]+)\$ and \$([^$]+)\$\.", line).groups()
        c = re.search(r"It reaches \$([^$]+)\$ \(SIBYLL\) and \$([^$]+)\$ \(QGSJET\) at \$4 \\times 10\^\{13\}\$ GeV", line).groups()
        d = re.search(r"Below 100 GeV it falls faster \(\$([^$]+)\$ at (\d+) GeV, \$([^$]+)\$ at (\d+) GeV\)", line).groups()
        return [percent(v) for v in a + b + c] + [percent(d[0]), d[1], percent(d[2]), d[3]]

    def compare(st):
        k_s, t1, sh_s, k_q, sh_q, top_s, top_q, b1, e1, b2, e2 = st

        def within(values, span, scale=100):
            return rounds_to(scale * min(values), span[0]) and rounds_to(scale * max(values), span[1])
        t1_sum = sum(float(v) for v in K_T1[1].rstrip("\\").split("&")[7:])
        return [Result(f"SIBYLL: the kaons carry {'-'.join(k_s)}% of E_p (100-1e12 GeV), {'-'.join(sh_s)}% of the pions' energy",
                       within(kap["SIBYLL"], k_s) and within(share["SIBYLL"], sh_s),
                       f"{100 * min(kap['SIBYLL']):.3f}-{100 * max(kap['SIBYLL']):.3f}%, "
                       f"{100 * min(share['SIBYLL']):.2f}-{100 * max(share['SIBYLL']):.2f}% ({len(grid)} energies)"),
                Result(f"QGSJET: {'-'.join(k_q)}% and {'-'.join(sh_q)}%", within(kap["QGSJET"], k_q) and within(share["QGSJET"], sh_q),
                       f"{100 * min(kap['QGSJET']):.3f}-{100 * max(kap['QGSJET']):.3f}%, "
                       f"{100 * min(share['QGSJET']):.2f}-{100 * max(share['QGSJET']):.2f}%"),
                Result(f"Koers et al.'s Table I: the kaons carry {t1[0]}%", rounds_to(100 * t1_sum, t1[0]), f"{t1_sum:.3f}"),
                Result(f"at 4e13 GeV: {top_s[0]}% (SIBYLL), {top_q[0]}% (QGSJET), at the same share of the pions' energy",
                       rounds_to(100 * kappa_K(top), top_s[0]) and rounds_to(100 * kappa_K(top, "QGSJET"), top_q[0])
                       and all(float(lo) <= round(100 * kappa_K(top, m) / kappa_pi(top, m)) <= float(hi)
                               for m, (lo, hi) in (("SIBYLL", sh_s), ("QGSJET", sh_q))),
                       f"{100 * kappa_K(top):.3f}%, {100 * kappa_K(top, 'QGSJET'):.3f}%; shares "
                       f"{100 * kappa_K(top) / kappa_pi(top, 'SIBYLL'):.2f}%, {100 * kappa_K(top, 'QGSJET') / kappa_pi(top, 'QGSJET'):.2f}%"),
                Result(f"below E_t (delta-functional approximation): {b1[0]}% at {e1} GeV, {b2[0]}% at {e2} GeV",
                       rounds_to(100 * kappa_K(float(e1)), b1[0]) and rounds_to(100 * kappa_K(float(e2)), b2[0]),
                       f"{100 * kappa_K(float(e1)):.3f}%, {100 * kappa_K(float(e2)):.3f}%")]
    out = [in_source("Koers et al.'s Table I as transcribed", KOERS, *K_T1),
           in_source("Kelner et al.'s eqs. 6 and 12 with their parameters as transcribed", KELNER,
                     *[t for d in KEL.values() for t in d["tex"]])]
    return out + compare(stated()) + [control_result("3.5% for 3.4% at 4e13 GeV (QGSJET)", compare, stated,
                                                     r"and $3.4\text{％}$ (QGSJET)", r"and $3.5\text{％}$ (QGSJET)")]


@derivation("pp.sigma_pions")
def check_pp_sigma_pions():
    """AM3's inelastic cross section for the charged pions (Kafexhiu et al.'s eq. 1 at the pi+- threshold) against eq. 1."""
    T, Tth, mpi = sp.symbols("T_p T_th m_pi", positive=True)
    eq1 = L.to_sympy(piece(K_EQ1, r"\left[ 30.7 - 0.96\,\log\left(\frac{T_p}{T_p^{\rm th}}\right) + 0.18\,\log^2\left("
                                  r"\frac{T_p}{T_p^{\rm th}}\right) \right]").replace(r"\log^2\left(\frac{T_p}{T_p^{\rm th}}\right)",
                                                                                      r"\left(\log\left(\frac{T_p}{T_p^{\rm th}}\right)\right)^2"),
                     {"T_p": T, r"T_p^{\rm th}": Tth})
    eq1 *= L.to_sympy(piece(K_EQ1, r"\left[1-\left(\frac{T_p^{\rm th}}{T_p} \right)^{1.9}\right]^3"), {"T_p": T, r"T_p^{\rm th}": Tth})
    th = L.to_sympy(piece(K_TH, r"2m_\pi+m_\pi^2/2m_p"), {r"m_\pi": mpi, "m_p": MASS["p"]})
    sigma = sp.lambdify((T, Tth), eq1, "math")
    t_th = sp.lambdify(mpi, th)
    pdg = float(re.search(r"\\approx([\d.]+)", K_TH).group(1))

    def stated():
        return re.search(r"mass, is \$(.+?)\$ below Kafexhiu et al\.'s from ([\d.]+) GeV up", L.line(
            r"Its inelastic cross section for the charged pions, whose threshold uses the $\pi^\pm$ mass", "checks")).groups()

    def compare(st):
        span, lo = percent(st[0]), float(st[1])
        E = [lo * (4e13 / lo) ** (k / 4000) for k in range(4001)]   # to the top of the default hadron grid
        r = [100 * (1 - sigma(e - MASS["p"], t_th(MASS["pi+"])) / sigma(e - MASS["p"], t_th(MASS["pi0"]))) for e in E]
        return [Result(f"{span[0]}-{span[1]}% below from {st[1]} GeV (to 4e13 GeV)",
                       rounds_to(min(r), span[0]) and rounds_to(max(r), span[1]), f"{min(r):.4f}-{max(r):.4f}%")]
    return [Result(f"Kafexhiu et al.'s threshold {pdg} GeV is that of the pi0 (PDG 2026 masses)",
                   rounds_to(t_th(MASS["pi0"]), str(pdg)), f"{t_th(MASS['pi0']):.5f} GeV")] + compare(stated()) + \
        [control_result("0.37 for 0.27%", compare, stated, r"$0.06\text{–}0.27\text{％}$ below", r"$0.06\text{–}0.37\text{％}$ below")]


if __name__ == "__main__":
    run()
