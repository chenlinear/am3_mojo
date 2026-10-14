"""Derivation checks of the neutron decay equations of README.md (Neutron decay) against their sources:
  - Hayen et al. (2018, Rev. Mod. Phys. 90, 015008), the TeX source of arXiv:1709.07530v2 (energies in units of
    m_e c^2): the allowed spectrum (eq. 4), the point-charge Fermi function (eqs. 6-7), Sirlin's outer radiative
    correction (eqs. 48, 50-51) and the ratio of bound to free decay;
  - Lipari, Lusignoli & Meloni (2007), the TeX source of arXiv:0704.0718v1, Sec. VII.C: the mean energy
    fraction and the end point of the antineutrinos of ultra-relativistic neutrons;
  - Huemmer et al. (2010), the TeX source of arXiv:1002.1310v3, Sec. 5.2, eq. 50: the line approximation;
  - Dimitrakoudis et al. (2012), the TeX source of arXiv:1209.0413v1, Sec. 2.2 (eqs. 13, 22-23): ATHEnA's
    neutron decay;
  - Dubbers & Schmidt (2011, Rev. Mod. Phys. 83, 1111, arXiv:1105.3694), Sec. VI.A: the proton's end-point
    energy, 750 eV;
  - the PDG 2026 listings of the neutron, the proton and the electron, transcribed below: the masses, lambda,
    the proton's charge radius, and the zero-recoil relations of the correlation coefficients a_0, A and B to
    lambda = g_A/g_V.
The kinematics the README states without a source is redone exactly: E_0, the products' lab energy fractions,
the antineutrinos' end point and Q_X as a count of the decays. The V-A trace with explicit Dirac matrices gives
the correlation coefficients at zero recoil, which fixes the sign convention of lambda in PDG's relations, and,
with the proton recoiling, the |M|^2 of the exact three-body decay.

The figures quoted in the Results of tests/literature_checks/README.md are read from there and recomputed
from the README's spectrum (quadrature, scipy), including the exact three-body decay (the Dalitz plot), the
radiative corrections, ATHEnA's treatment and Huemmer et al.'s line for power-law decays.

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/neutron_decay.py
"""
import functools
import re

import mpmath as mp
import numpy as np
import sympy as sp
from scipy import integrate, special

import latex as L
import mojo_constants
from latex import DIMITRAKOUDIS, HAYEN, HUEMMER, LIPARI
from pdg import INV_ALPHA, MASS_MEV
from verify import Result, derivation, exact, exact_fails, in_source, numeric, run


# --- PDG 2026 (MeV) ---
PDG_MASSES = {"m_n": MASS_MEV["n"], "m_p": MASS_MEV["p"], "m_e": MASS_MEV["e"]}
M_N, M_P, M_E = (mp.mpf(PDG_MASSES[k]) for k in ("m_n", "m_p", "m_e"))
ALPHA = 1 / mp.mpf(INV_ALPHA)
LAMBDA = mp.mpf("-1.2753")              # n listing: lambda = g_A/g_V, OUR AVERAGE -1.2753 +- 0.0013 (S = 2.7)
R_P = mp.mpf("0.84075")                 # p listing: p CHARGE RADIUS (fm), MOHR 25
HBARC = mp.mpf("197.3269804")           # MeV fm (2022 CODATA)
# Hayen et al.'s cut-off radius for the proton: a uniform sphere of its charge radius, in units hbar/(m_e c)
R_SPHERE = mp.sqrt(mp.mpf(5) / 3) * R_P * M_E / HBARC
# n listing, the n -> p e- nu_e-bar decay parameters a, A and B (pp. 11-15), as printed (Standard Model, zero
# recoil order, g_A and g_V real)
PDG_a = lambda lam: (1 - lam**2) / (1 + 3 * lam**2)            # noqa: E731
PDG_A = lambda lam: -2 * lam * (lam + 1) / (1 + 3 * lam**2)    # noqa: E731
PDG_B = lambda lam: 2 * lam * (lam - 1) / (1 + 3 * lam**2)     # noqa: E731

# --- Hayen et al. 2018, transcribed from the TeX source (units m_e = c = hbar = 1; hayen_tex() holds the TeX) ---
# eq. 4: N(W) dW = (G_V^2 V_ud^2 / 2 pi^3) F0(Z, W) L0 U D_FS R R_N Q S X r C D_C p W (W0 - W)^2 dW,
#   W the total energy of the beta particle, W0 its total energy at the end point, p = sqrt(W^2 - 1);
# eq. 6: F0(Z, W) = 4 (2pR)^(2(gamma - 1)) e^(pi y) |Gamma(gamma + i y)|^2 / (Gamma(1 + 2 gamma))^2,
# eq. 7: gamma = sqrt(1 - (alpha Z)^2), y = +- alpha Z W/p (+ for beta-).


def hayen_tex():
    """Eqs. 4, 6 and 7 as their TeX source writes them (the transcription above)."""
    return (r"N(W)dW & = \frac{G_V^2 V_{ud}^2}{2\pi^3} ~ F_{0}(Z, W)", r"p W (W_0 - W)^2 ~ dW",
            r"F_0(Z, W) = 4 (2pR)^{2(\gamma - 1)} e^{\pi y} \frac{|\Gamma(\gamma + i y)|^2}{\left( \Gamma(1 + 2\gamma) \right)^2}",
            r"\gamma = \sqrt{1 - (\alpha Z)^2}, \quad y = \pm \alpha Z W / p")


def hayen_F0(Z, W, R, alpha, g=None):
    """Eq. 6 (with eq. 7); g overrides gamma (the README's limit gamma -> 1)."""
    p = mp.sqrt(W**2 - 1)
    g = mp.sqrt(1 - (alpha * Z)**2) if g is None else g
    y = alpha * Z * W / p
    return 4 * (2 * p * R)**(2 * (g - 1)) * mp.e**(mp.pi * y) * abs(mp.gamma(g + 1j * y))**2 / mp.gamma(1 + 2 * g)**2


mn, mp_, me = sp.symbols("m_n m_p m_e", positive=True)
E, p, Es, ps, Ev = sp.symbols("E p E_s p_s E_v", positive=True)
b = sp.Symbol("beta", positive=True)
c = sp.Symbol("c", real=True)          # cos(theta*) in the neutron's rest frame
eta, al = sp.symbols("eta alpha", positive=True)


def E0_LINE():
    return L.display(r"{2m_n} = 1.29258\ \mathrm{MeV}")


def readme_E0():
    s = L.sides(L.parts(E0_LINE())[0])
    assert s[0] == "E_0" and s[2] == r"1.29258\ \mathrm{MeV}", s
    return L.to_sympy(s[1], {"m_n": mn, "m_p": mp_, "m_e": me}), s[2]


def SPEC_LINE():
    return L.display(r"\frac{dN}{dE} \propto F(E)")


def readme_spectrum(E0=sp.Symbol("E_0")):
    """The README's dN/dE (up to its constant, F as a function), E_nu-bar, F(E) and eta."""
    ps_ = L.parts(SPEC_LINE())
    lhs, rhs = ps_[0].split(r"\propto")
    assert lhs.strip() == r"\frac{dN}{dE}", lhs
    Fsym = sp.Function("F")
    spec = L.to_sympy(rhs, {"p": p, "E": E, "E_0": E0}, funcs={"F": lambda a: Fsym(a)})
    nu = L.sides(L.part(SPEC_LINE(), r"E_{\bar\nu} ="))
    Enu = L.to_sympy(nu[1], {"E": E, "E_0": E0})
    Fs = L.sides(L.part(SPEC_LINE(), "F(E) ="))
    F = L.to_sympy(Fs[1], {r"\eta": eta})
    es = L.sides(L.part(SPEC_LINE(), r"\eta ="))
    eta_ = L.to_sympy(es[1], {r"\alpha": al, "E": E, "p": p})
    return spec, Fsym, Enu, F, eta_


def BOX_LINE():
    return L.display(r"x \in \left[\frac{E^\ast - \beta_n p^\ast}{m_n}")


def readme_box():
    tex = BOX_LINE()
    inner = L.between(tex, r"\left[", r"\right]")
    lo, hi = L.split_top(inner, r",\ ")
    tok = {r"E^\ast": Es, r"\beta_n": b, r"p^\ast": ps, "m_n": mn}
    return L.to_sympy(lo, tok), L.to_sympy(hi, tok)


def readme_density():
    return L.to_sympy(L.inline(r"m_n/(2\beta_n p^\ast)"), {"m_n": mn, r"\beta_n": b, r"p^\ast": ps})


# --- the checks ---

TAU_N = sp.Rational("878.3")             # n listing: n MEAN LIFE, OUR AVERAGE 878.3 +- 0.4 s (scale factor 1.8)
DM_NP = sp.Rational("1.29333251")        # n listing: m_n - m_p (MeV)


def code_inputs(src):
    """The neutron decay inputs as constants.mojo writes them (GeV, s), exact (decimal literals as rationals)."""
    return mojo_constants.read(("massN_PDG", "massP_PDG", "massE_PDG", "tdec_n", "alpha_PDG"), src)


@derivation("nd.inputs")
def check_inputs():
    """The code's masses, mean life and alpha (src/am3_mojo/utils/constants.mojo) are PDG 2026's, as the README says."""
    L.line("the code uses the masses and the mean life of the PDG 2026 listings", "checks")
    pdg = {f"mass{k[2].upper()}_PDG": sp.Rational(v) / 1000 for k, v in PDG_MASSES.items()}   # GeV
    pdg["alpha_PDG"] = 1 / sp.Rational(INV_ALPHA)
    src = mojo_constants.PATH.read_text()

    def tau_readme():
        return sp.Rational(re.search(r"with the mean life \$\\tau_n = ([\d.]+)\$ s",
                                     L.line(r"with the mean life $\tau_n = ")).group(1))

    def results(src_):
        code = code_inputs(src_)
        return [Result("m_n, m_p, m_e and alpha of the code = PDG 2026's (n, p, e listings; Table 1.1)",
                       all(code[k] == v for k, v in pdg.items()),
                       "exact: " + ", ".join(f"{k} = {code[k] if code[k].q == 1 else sp.N(code[k], 15)}" for k in pdg)),
                Result("the code's tdec_n = the README's tau_n = PDG 2026's mean life", code["tdec_n"] == tau_readme() == TAU_N,
                       f"exact: {code['tdec_n']} s")]
    broken_src = src.replace("massN_PDG: Float64 = 0.93956542194", "massN_PDG: Float64 = 0.93956542195")
    with L.patched(r"$\tau_n = 878.3$", r"$\tau_n = 879.3$"):
        broken_tau = results(src)[1]
    return results(src) + [
        Result("PDG's m_n - m_p (n listing) = the difference of its masses",
               sp.Rational(PDG_MASSES["m_n"]) - sp.Rational(PDG_MASSES["m_p"]) == DM_NP, f"exact: {DM_NP} MeV"),
        Result("control: m_n one digit off in the source", broken_src != src and not results(broken_src)[0].ok,
               "the comparison fails, as it must"),
        Result("control: tau_n = 879.3 s in the README", not broken_tau.ok, "the comparison fails, as it must")]


@derivation("nd.E0")
def check_E0():
    """E_0 is the electron's largest energy: the proton and the antineutrino recoil together (their mass m_p)."""
    readme, value = readme_E0()
    mx2 = sp.Symbol("m_x2", positive=True)          # invariant mass^2 of the proton and the antineutrino
    two_body = (mn**2 + me**2 - mx2) / (2 * mn)     # the electron's energy against a system of mass^2 mx2
    num = ((M_N - M_P) * (M_N + M_P) + M_E**2) / (2 * M_N)
    with L.patched(r"E_0 = \frac{m_n^2 - m_p^2 + m_e^2}{2m_n}", r"E_0 = \frac{m_n^2 - m_p^2 - m_e^2}{2m_n}"):
        broken = readme_E0()[0]
    return [Result("E_e(m_x^2) falls with m_x^2: largest at the smallest m_x = m_p + m_nu = m_p",
                   sp.diff(two_body, mx2) == -1 / (2 * mn), f"dE/dm_x^2 = {sp.diff(two_body, mx2)}"),
            exact("the README's E_0 = (m_n^2 + m_e^2 - m_p^2)/(2 m_n)", readme - two_body.subs(mx2, mp_**2)),
            Result("PDG 2026: E_0 = 1.29258 MeV, as printed", mp.nstr(num, 6) == "1.29258", f"E_0 = {mp.nstr(num, 12)} MeV"),
            Result("kinetic end point 781.6 keV, the proton's largest kinetic energy 751 eV (as stated)",
                   abs(1e3 * (num - M_E) - mp.mpf("781.6")) < 0.05 and abs(1e6 * (M_N - M_P - num) - 751) < 0.5,
                   f"{mp.nstr(1e3 * (num - M_E), 7)} keV, {mp.nstr(1e6 * (M_N - M_P - num), 6)} eV"),
            exact_fails("control: m_e^2 with the wrong sign", broken - two_body.subs(mx2, mp_**2))]


@derivation("nd.spectrum")
def check_spectrum():
    """The README's spectrum is Hayen et al.'s eq. 4 without its corrections, with F their eq. 6 for gamma -> 1 (Z = 1)."""
    spec, Fsym, Enu, F, eta_ = readme_spectrum()
    W, W0, pp = sp.symbols("W W_0 p_W", positive=True)    # Hayen's units: E = W m_e, p = p_W m_e
    E0 = sp.Symbol("E_0")
    hayen = Fsym(W * me) * pp * W * (W0 - W)**2           # eq. 4 with L0 = U = ... = C = 1 (F0 as F)
    ratio = sp.simplify((spec / hayen).subs({E: W * me, p: pp * me, E0: W0 * me}))
    out = [in_source("Hayen et al. eqs. 4, 6 and 7 as transcribed", HAYEN, *hayen_tex()),
           Result("README dN/dE = m_e^4 x Hayen's eq. 4 without its corrections (a constant)", ratio == me**4,
                  f"ratio {ratio}"),
           exact("E_nu-bar = E_0 - E: the leptons share E_0 (energy conservation, proton recoil neglected)",
                 Enu - (E0 - E)),
           exact("eta = alpha E/p is Hayen's y = alpha Z W/p for Z = 1 (any units)", eta_ - al * E / p)]
    # F0 at gamma = 1 (Z = 1: gamma = sqrt(1 - alpha^2) -> 1): 4 e^(pi y) |Gamma(1 + iy)|^2 / Gamma(3)^2 =
    # e^(pi y) pi y / sinh(pi y) = 2 pi y / (1 - e^(-2 pi y))
    y = sp.Symbol("y", positive=True)
    limit = 4 * sp.exp(sp.pi * y) * (sp.pi * y / sp.sinh(sp.pi * y)) / sp.gamma(3)**2
    out.append(exact("with |Gamma(1 + iy)|^2 = pi y/sinh(pi y): 4 e^(pi y)|Gamma(1+iy)|^2/Gamma(3)^2 = the README's F",
                     sp.simplify((limit - F.subs(eta, y)).rewrite(sp.exp))))
    rng = __import__("random").Random(3)
    pts = [(mp.mpf(10) ** mp.mpf(rng.uniform(-3, 1)),) for _ in range(20)]
    out.append(numeric("|Gamma(1 + iy)|^2 = pi y / sinh(pi y) (mpmath, 50 digits)",
                       lambda yy: abs(mp.gamma(1 + 1j * yy))**2, lambda yy: mp.pi * yy / mp.sinh(mp.pi * yy), pts))
    # Hayen's F0 (as transcribed) at Z = 1 with gamma = 1 is the README's F(E), to 50 digits
    Wp = [(1 + mp.mpf(10) ** mp.mpf(rng.uniform(-4, 0.5)),) for _ in range(10)]
    readme_F = lambda yy: 2 * mp.pi * yy / -mp.expm1(-2 * mp.pi * yy)  # noqa: E731
    out.append(numeric("Hayen's eq. 6 with gamma = 1 (Z = 1, any R) = the README's F at eta = alpha W/p",
                       lambda w: hayen_F0(1, w, mp.mpf(3), ALPHA, g=1),
                       lambda w: readme_F(ALPHA * w / mp.sqrt(w**2 - 1)), Wp))
    # the Z = 1 value at gamma = sqrt(1 - alpha^2): within 6e-4 of the README's F (R = sqrt(5/3) r_p = 1.085 fm in m_e units)
    R = R_SPHERE
    worst = max(abs(hayen_F0(1, w, R, ALPHA) / readme_F(ALPHA * w / mp.sqrt(w**2 - 1)) - 1) for (w,) in Wp)
    out.append(Result("Hayen's F0 at Z = 1 (gamma = sqrt(1 - alpha^2), R of the proton) / README's F - 1",
                      worst < 1e-3, f"at most {mp.nstr(worst, 3)} (the README: the spectrum changes by at most 2.4e-4)"))
    with L.patched(r"F(E)\thinspace p\thinspace E\thinspace (E_0 - E)^2", r"F(E)\thinspace p\thinspace E\thinspace (E_0 - E)^3"):
        broken = readme_spectrum()[0]
    rb = sp.simplify((broken / hayen).subs({E: W * me, p: pp * me, E0: W0 * me}))
    out.append(Result("control: (E_0 - E)^3 is not Hayen's eq. 4", rb.has(W), f"ratio {rb}"))
    return out


@derivation("nd.box")
def check_box():
    """x = E/E_n of an isotropic product of a neutron of any velocity: uniform over the README's range."""
    g = 1 / sp.sqrt(1 - b**2)
    x = g * (Es + b * ps * c) / (g * mn)          # lab energy over E_n = gamma m_n (boost along the product's z)
    lo, hi = readme_box()
    return [Result("x is linear in cos theta* (so uniform for an isotropic decay)", sp.diff(x, c, 2) == 0, f"x = {sp.simplify(x)}"),
            exact("x at cos theta* = -1, +1: the README's ends", (x.subs(c, -1) - lo)**2 + (x.subs(c, 1) - hi)**2),
            exact("for beta -> 1 the ends of the kaon decays (kd.flat): (E* -+ p*)/m_n",
                  (lo.subs(b, 1) - (Es - ps) / mn)**2 + (hi.subs(b, 1) - (Es + ps) / mn)**2)]


@derivation("nd.density")
def check_density():
    """The density over the range: (1/2)/(dx/dcos theta*) = m_n/(2 beta_n p*)."""
    x = (Es + b * ps * c) / mn
    with L.patched(r"with density $m_n/(2\beta_n p^\ast)$", r"with density $m_n/(\beta_n p^\ast)$"):
        broken = L.to_sympy(L.inline(r"m_n/(\beta_n p^\ast)"), {"m_n": mn, r"\beta_n": b, r"p^\ast": ps})
    return [exact("(1/2)/(dx/dc) = the README's density", sp.Rational(1, 2) / sp.diff(x, c) - readme_density()),
            exact_fails("control: m_n/(beta_n p*)", sp.Rational(1, 2) / sp.diff(x, c) - broken)]


@derivation("nd.xp")
def check_xp():
    """The proton takes m_n - E_0 (energy conservation), at rest in the neutron's frame: x_p = (m_n - E_0)/m_n."""
    E0 = sp.Symbol("E_0", positive=True)
    xp = L.to_sympy(L.sides(L.inline(r"x_p = (m_n - E_0)/m_n"))[1], {"m_n": mn, "E_0": E0})
    Ep = mn - E0                                   # m_n - (E_e + E_nu-bar), with E_e + E_nu-bar = E_0
    lo, hi = readme_box()
    return [exact("a product at rest (p* = 0) has the single x = E*/m_n (the range of nd.box)", (lo - hi).subs(ps, 0)),
            exact("x_p = (m_n - E_0)/m_n", xp - lo.subs({ps: 0, Es: Ep}))]


def SOURCE_LINE():
    return L.display(r"Q_X(E) = \int")


def readme_gamma_n(En, cl):
    """The README's gamma_n = E_n/(m_n c^2), c = cl."""
    return L.to_sympy(L.sides(L.inline(r"\gamma_n = E_n/(m_n c^2)"))[1], {"E_n": En, "m_n": mn, "c": cl})


def readme_source(En, Nn, FX, beta):
    """The README's Q_X(E) as (the weight of its measure dE_n, its integrand), for the functions Nn(E_n), FX(x, beta)
    and the neutron's velocity beta. Q_X and N_n are per unit energy, as the text says."""
    L.line(r"where $N_n$ is the neutron density per unit energy, $Q_{X}$ the source of product $X$ per unit energy")
    lhs, rhs = L.sides(L.parts(SOURCE_LINE())[0])
    assert lhs == "Q_X(E)", lhs
    measure, body = L.between(rhs, r"\int ", r"\thinspace "), rhs.split(r"\thinspace ", 1)[1]
    dEn = sp.Symbol("dE_n")
    m = L.to_sympy(measure, {"dE_n": dEn, "E_n": En})
    weight = sp.diff(m, dEn)
    assert sp.simplify(m - weight * dEn) == 0, measure
    tau, cl = sp.symbols("tau_n c_light", positive=True)
    tokens = {"E_n": En, "E": E, r"\gamma_n": readme_gamma_n(En, cl), r"\tau_n": tau, r"\beta_n": beta}
    integrand = L.to_sympy(body.replace(";", ","), tokens, funcs={"N_n": Nn, "F_X": FX})   # F_X(x; beta_n): its parameter
    return weight, integrand


@derivation("nd.source")
def check_source():
    """Q_X counts the decays: N_n dE_n neutrons decay at the time-dilated rate, each making F_X dx products, x = E/E_n."""
    En, tau, cl = sp.symbols("E_n tau_n c_light", positive=True)
    Nn, FX = sp.Function("N_n"), sp.Function("F_X")
    beta = sp.Function("beta_n")(En)
    gamma = En / (mn * cl**2)                                   # E_n = gamma m_n c^2: the neutron's Lorentz factor
    x = L.to_sympy(L.sides(L.inline(r"x = E/E_n"))[1], {"E": E, "E_n": En})
    # per dE_n: N_n dE_n neutrons decay at 1/(gamma tau_n) (mean life gamma tau_n in the lab), each making F_X(x) dx
    # products; dx = (dx/dE) dE at fixed E_n gives them per unit E
    counted = Nn(En) / (gamma * tau) * FX(x, beta) * sp.diff(x, E)
    weight, integrand = readme_source(En, Nn, FX, beta)
    rate = L.to_sympy(L.inline(r"1/(\gamma_n\tau_n)"), {r"\gamma_n": readme_gamma_n(En, cl), r"\tau_n": tau})
    out = [exact("README Q_X = the count of the decays per unit energy (any N_n, F_X; integrands in E_n)",
                 weight * integrand - counted),
           exact("the README's rate 1/(gamma_n tau_n) is the time-dilated one, gamma_n = E_n/(m_n c^2)", rate - 1 / (gamma * tau))]
    for old, new, what in ((r"Q_X(E) = \int \frac{dE_n}{E_n}\thinspace", r"Q_X(E) = \int dE_n\thinspace", "dE_n for dE_n/E_n"),
                           (r"\left(\frac{E}{E_n};\beta_n\right)", r"\left(\frac{E_n}{E};\beta_n\right)", "F_X(E_n/E)")):
        with L.patched(old, new):
            w_b, i_b = readme_source(En, Nn, FX, beta)
        out.append(exact_fails(f"control: {what}", w_b * i_b - counted))
    return out


@derivation("nd.endpoint")
def check_endpoint():
    """The antineutrino reaches x = 2(E_0 - m_e)/m_n (beta -> 1, its largest energy E_0 - m_e); its PDG 2026 value."""
    E0 = sp.Symbol("E_0", positive=True)
    s = L.sides(L.inline(r"x = 2(E_0 - m_e)/m_n = 1.664 \times 10^{-3}", "checks"))
    readme = L.to_sympy(s[1], {"E_0": E0, "m_e": me, "m_n": mn})
    lo, hi = readme_box()
    top = hi.subs(b, 1).subs(ps, Es).subs(Es, E0 - me)   # massless: p* = E*; the largest E_nu-bar = E_0 - m_e
    num = 2 * (((M_N - M_P) * (M_N + M_P) + M_E**2) / (2 * M_N) - M_E) / M_N
    return [exact("upper end of the antineutrino's range for beta -> 1 at E* = E_0 - m_e", readme - top),
            Result("1.664e-3 with PDG 2026", mp.nstr(num, 4) == "0.001664", f"{mp.nstr(num, 8)}")]


def lipari_tex():
    return (r"$\langle E_{\overline{\nu}_e}\rangle/E_n \simeq 5.1 \times 10^{-4}$",
            r"$E^{\rm max}_{\overline{\nu}_e}/E_n \simeq  2 (m_n - m_p - m_e)/m_n" + "\n" + r"\simeq 1.66 \times 10^{-3}$")


# --- the figures of the Results (file "checks"), read there and computed from the README's spectrum ---

FIG = r"([+-]?\d+(?:\.\d+)?(?: \\times 10\^\{-?\d+\})?)"   # a number as the README prints it
PCT = FIG + r"\\text\{％\}"                                   # a percentage


def figure(tex):
    """The value of a number as the README prints it ("0.3011", "-4.1 \\times 10^{-4}", "+0.4") and the place of
    its last digit."""
    m = re.fullmatch(r"([+-]?\d+)(?:\.(\d+))?(?: \\times 10\^\{(-?\d+)\})?", tex)
    if m is None:
        raise ValueError(f"not a number: {tex}")
    frac, exp = m.group(2) or "", int(m.group(3) or 0)
    return float(f"{m.group(1)}.{frac or '0'}e{exp}"), 10.0 ** (exp - len(frac))


def as_printed(what, value, tex, note=""):
    """value, rounded to the last digit of the README's figure tex, is that figure."""
    v, unit = figure(tex)
    short = re.sub(r" \\times 10\^\{(-?\d+)\}", r"e\1", tex)
    return Result(what, round(value / unit) == round(v / unit), f"{value:.6g} (README: {short}){note}")


def quoted(pattern, pin, file="checks"):
    """The groups of the one match of the regex in the line of the file that holds the pin."""
    found = re.findall(pattern, L.line(pin, file))
    if len(found) != 1:
        raise LookupError(f"{file}: {len(found)} matches of {pattern} in the line of: {pin}")
    return found[0]


def wrong_figure(what, results, old, new, file="checks"):
    """The control of a check of quoted figures: with one of them changed in the file, a result fails."""
    with L.patched(old, new, file):
        bad = [r.what for r in results() if not r.ok]
    return Result(f"control: {what}", bool(bad), f"fails: {bad[0]}" if bad else "passes, but must fail")


ME_F, MN_F, MP_F, AL_F = (float(v) for v in (M_E, M_N, M_P, ALPHA))


def numeric_endpoint(m_n, m_p, m_e):
    """The README's E_0 for the masses given (any unit), evaluated in 40 digits."""
    with mp.workdps(40):
        expr = readme_E0()[0]
        return float(sp.N(expr.subs({mn: sp.Float(repr(m_n), 40), mp_: sp.Float(repr(m_p), 40), me: sp.Float(repr(m_e), 40)}), 40))


E0_F = numeric_endpoint(MN_F, MP_F, ME_F)   # the README's E_0 [MeV] with PDG 2026's masses
T_TOP = E0_F - ME_F                         # the electron's largest kinetic energy


@functools.cache
def readme_shape():
    """The README's spectrum as numpy functions of the electron's E and p (MeV): its F, and the rest p E (E_0 - E)^2
    at E_0 of PDG 2026."""
    E0 = sp.Symbol("E_0")
    spec, Fsym, Enu, F, eta_ = readme_spectrum(E0)
    rest = sp.simplify(spec / Fsym(E))
    assert not rest.has(Fsym), rest
    Fe = F.subs(eta, eta_).subs(al, AL_F)
    return sp.lambdify((E, p), Fe, "numpy"), sp.lambdify((E, p, E0), rest, "numpy")


def allowed(T, F=None):
    """dN/dT of the README's spectrum, up to its constant (T = E - m_e, MeV); F replaces its Fermi function."""
    Fr, rest = readme_shape()
    E_, p_ = ME_F + T, np.sqrt(T * (T + 2 * ME_F))
    return (Fr if F is None else F)(E_, p_) * rest(E_, p_, E0_F)


def moments(f, top=T_TOP, k=2, **kw):
    """int f, int f T, ... over [0, top] (adaptive quadrature)."""
    o = dict(epsabs=0, epsrel=1e-12, limit=400) | kw
    return [integrate.quad(lambda t: f(t) * t**j, 0, top, **o)[0] for j in range(k)]


@functools.cache
def spectrum_moments():
    """int dN and int T dN of the README's spectrum (its normalisation and mean kinetic energy)."""
    return tuple(moments(allowed))


def mean_fraction():
    """<E_nu-bar>/m_n of the README's spectrum (PDG 2026), by quadrature: independent of the code."""
    m0, m1 = spectrum_moments()
    return (T_TOP - m1 / m0) / MN_F


@derivation("nd.lipari")
def check_lipari():
    """Lipari et al.'s mean fraction 5.1e-4 (Sec. VII.C), against the README's 5.11e-4 of its spectrum."""
    f = mean_fraction()

    def results():
        ours, theirs = quoted(r"the antineutrino takes \$" + FIG + r"\$ of the energy on average \(Lipari et al\.: "
                              r"\$\\simeq " + FIG + r"\$\)", "of the energy on average (Lipari et al.:")
        return [as_printed("README: the mean <E_nu-bar>/m_n of its spectrum", f, ours),
                as_printed("Lipari et al.'s figure is that, to the digits they give", f, theirs)]
    return [in_source("Lipari et al. Sec. VII.C: mean fraction and end point", LIPARI, *lipari_tex()), *results(),
            wrong_figure("5.12e-4 for 5.11e-4", results, r"$5.11 \times 10^{-4}$ of", r"$5.12 \times 10^{-4}$ of")]


@derivation("nd.lipari_endpoint")
def check_lipari_endpoint():
    """Lipari et al.'s end point 2(m_n - m_p - m_e)/m_n ~ 1.66e-3: as quoted, and 1.0 per mille above the exact 2(E_0 - m_e)/m_n."""
    readme = L.to_sympy(L.inline(r"\simeq 2(m_n - m_p - m_e)/m_n", "checks")[len(r"\simeq "):], {"m_n": mn, "m_p": mp_, "m_e": me})
    tex = L.to_sympy("2 (m_n - m_p - m_e)/m_n", {"m_n": mn, "m_p": mp_, "m_e": me})
    v = 2 * (M_N - M_P - M_E) / M_N
    exact_v = 2 * (((M_N - M_P) * (M_N + M_P) + M_E**2) / (2 * M_N) - M_E) / M_N
    return [in_source("Lipari et al.: the end point", LIPARI, lipari_tex()[1]),
            exact("the README quotes their formula", readme - tex),
            Result("their formula and the exact end point are 1.66e-3 to the digits they give; they differ by the recoil",
                   abs(v / mp.mpf("1.66e-3") - 1) < 0.005 and abs(exact_v / mp.mpf("1.66e-3") - 1) < 0.005,
                   f"theirs {mp.nstr(v, 6)}, exact {mp.nstr(exact_v, 6)} (ratio {mp.nstr(v / exact_v, 6)})")]


@derivation("nd.huemmer")
def check_huemmer():
    """Huemmer et al.'s eq. 50 as the Results of the checks write it: Q_nu(E) = Q_n(E/chi)/chi."""
    Q = sp.Function("Q_n")
    chi, Ex = sp.symbols("chi E", positive=True)
    readme = L.to_sympy(L.sides(L.inline(r"Q_{\bar\nu}(E) = Q_n(E/\chi)/\chi", "checks"))[1], {r"\chi": chi, "E": Ex},
                        funcs={"Q_n": lambda a: Q(a)})
    tex = r"\frac{1}{\chi} Q_n\left(\frac{E_\nu}{\chi}\right)"
    theirs = L.to_sympy(tex, {r"\chi": chi, r"E_\nu": Ex}, funcs={"Q_n": lambda a: Q(a)})
    return [in_source("Huemmer et al. eq. 50 and chi", HUEMMER,
                      r"Q_{\bar{\nu}_e}(E_\nu)=\frac{1}{\chi} Q_n\left(\frac{E_\nu}{\chi}\right)", r"$\chi\approx5.1\times10^{-4}$"),
            exact("the quoted form = their eq. 50", readme - theirs)]


def gammas():
    I2, Z2 = sp.eye(2), sp.zeros(2)
    s = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
    g0 = sp.diag(1, 1, -1, -1)
    gs = [g0] + [sp.Matrix(sp.BlockMatrix([[Z2, si], [-si, Z2]])) for si in s]
    return gs, sp.I * gs[0] * gs[1] * gs[2] * gs[3]


@derivation("nd.a_A")
def check_correlations():
    """a, A and B of PDG from the V-A trace at zero recoil, with the current p-bar gamma^mu (1 + lambda g5) n."""
    lam = sp.Symbol("lambda", real=True)
    Mn, Ee, pe, Enu, ct, st = sp.symbols("M E_e p_e E_nu c_t s_t", positive=True)
    G, g5 = gammas()
    one = sp.eye(4)
    met = [1, -1, -1, -1]
    sl = lambda q: sum((met[m] * q[m] * G[m] for m in range(4)), sp.zeros(4))  # noqa: E731
    bar = lambda A: G[0] * A.H * G[0]                              # noqa: E731
    pn = [Mn, 0, 0, 0]
    pe4 = [Ee, 0, 0, pe]                         # the electron along z
    pv4 = [Enu, Enu * st, 0, Enu * ct]           # the antineutrino at the angle theta_e-nu
    Lmn = [[(G[m] * (one - g5) * sl(pv4) * bar(G[n] * (one - g5)) * (sl(pe4) + sp.Symbol("m_e") * one)).trace()
            for n in range(4)] for m in range(4)]

    def M2(spin):
        proj = (sl(pn) + Mn * one) * (one + g5 * sl(spin)) / 2
        Hmn = [[(G[m] * (one + lam * g5) * proj * bar(G[n] * (one + lam * g5)) * (sl(pn) + Mn * one)).trace()
                for n in range(4)] for m in range(4)]
        return sp.expand(sum(met[m] * met[n] * Lmn[m][n] * Hmn[m][n] for m in range(4) for n in range(4)))
    up, down = M2([0, 0, 0, 1]), M2([0, 0, 0, -1])
    unpol = sp.expand((up + down) / 2)
    pol = sp.expand((up - down) / 2)
    base = unpol.subs(ct, 0)                     # (1 + 3 lambda^2) E_e E_nu up to a constant
    a = sp.simplify(sp.diff(unpol, ct) / base * Ee / pe)
    A = sp.simplify(pol.subs(ct, 0) / base * Ee / pe)      # s along the electron: A beta_e
    Bc = sp.simplify(sp.diff(pol, ct) / base)              # s along z: B cos theta_nu
    readme_a = L.to_sympy(L.sides(L.inline(r"a = (1 - \lambda^2)/(1 + 3\lambda^2)", "checks"))[1], {r"\lambda": lam})
    readme_A = L.to_sympy(L.sides(L.inline(r"A = -2\lambda(\lambda + 1)/(1 + 3\lambda^2)", "checks"))[1], {r"\lambda": lam})
    return [exact("unpolarised |M|^2 ∝ 1 + a beta_e cos theta: a from the trace = PDG's", a - PDG_a(lam)),
            exact("A (neutron spin along the electron) from the trace = PDG's", A - PDG_A(lam)),
            exact("B (neutron spin along the antineutrino) from the trace = PDG's", Bc - PDG_B(lam)),
            exact("the README's a = PDG's", readme_a - PDG_a(lam)),
            exact("the README's A = PDG's", readme_A - PDG_A(lam)),
            exact_fails("control: the current (1 - lambda g5) gives another A", A.subs(lam, -lam) - PDG_A(lam))]


# --- the figures of the Results: the spectrum, the Fermi function ---

@derivation("nd.means")
def check_means():
    """The mean energies of the README's spectrum (PDG 2026) and the mean lab energy fractions of its products."""
    m0, m1 = spectrum_moments()
    T_mean = m1 / m0
    lo, hi = readme_box()

    def results():
        Ts, Evs = quoted(r"mean kinetic energy is " + FIG + r" MeV and the antineutrino's mean energy " + FIG + r" MeV",
                         "the electron's mean kinetic energy is")
        xe, xp = quoted(r"On average the electron takes \$" + FIG + r"\$ and the proton " + FIG + r"\.",
                        "On average the electron takes")
        return [as_printed("the electron's mean kinetic energy <T> [MeV]", T_mean, Ts),
                as_printed("the antineutrino's mean energy E_0 - m_e - <T> [MeV]", T_TOP - T_mean, Evs),
                as_printed("the electron's mean lab fraction (m_e + <T>)/m_n", (ME_F + T_mean) / MN_F, xe),
                as_printed("the proton's x_p = (m_n - E_0)/m_n (nd.xp)", 1 - E0_F / MN_F, xp)]
    return [exact("the mean x over the range of nd.box is E*/m_n, for any beta_n", (lo + hi) / 2 - Es / mn), *results(),
            wrong_figure("0.3021 MeV for 0.3011 MeV", results, "is 0.3011 MeV", "is 0.3021 MeV")]


@derivation("nd.fermi")
def check_fermi():
    """The README's Fermi function against F = 1: the normalised spectrum at the energies quoted, and <T>."""
    m0, m1 = spectrum_moments()
    n0, n1 = moments(lambda t: allowed(t, lambda E_, p_: np.ones_like(E_)))
    Fr = readme_shape()[0]
    ratio = lambda T: Fr(ME_F + T, np.sqrt(T * (T + 2 * ME_F))) * n0 / m0  # noqa: E731  (with F / without, normalised)
    pin = "The Fermi function raises the normalised spectrum by"
    others = ", ".join(f"{100 * (ratio(T) - 1):+.2f}% at {1e3 * T:.0f} keV" for T in (0.1, 0.3, 0.6))
    nu = f"; <E_nu-bar> {100 * ((T_TOP - n1 / n0) / (T_TOP - m1 / m0) - 1):+.2f}%"

    def results():
        at = re.findall(PCT + r"\$ at (\d+) keV", L.line(pin, "checks"))
        assert len(at) == 2, at
        out = [as_printed(f"F raises it at {k} keV by [%]", 100 * (ratio(1e-3 * float(k)) - 1), v) for v, k in at]
        out.append(as_printed("and lowers it at the end point by [%]", 100 * (1 - ratio(T_TOP)),
                              quoted(r"lowers it by \$" + PCT + r"\$ near the end point", pin), f"; {others}"))
        out.append(as_printed("<T> without F is higher by [%]", 100 * (n1 / n0 / (m1 / m0) - 1),
                              quoted(r"the mean kinetic energy would be \$" + PCT + r"\$ higher", pin), nu))
        return out
    return [*results(), wrong_figure("9.5% for 8.5% at 10 keV", results, r"$8.5\text{％}$ at 10 keV",
                                     r"$9.5\text{％}$ at 10 keV")]


def hayen_rel(E_, p_):
    """Hayen et al.'s eq. 6 at Z = 1 with gamma_1 = sqrt(1 - alpha^2) and R = R_SPHERE (numpy; E and p in MeV)."""
    W, pw = E_ / ME_F, p_ / ME_F
    g1, y = np.sqrt(1 - AL_F**2), AL_F * W / pw
    return (4 * (2 * pw * float(R_SPHERE))**(2 * (g1 - 1)) * np.exp(np.pi * y + 2 * special.loggamma(g1 + 1j * y).real)
            / special.gamma(1 + 2 * g1)**2)


@derivation("nd.fermi_rel")
def check_fermi_rel():
    """Hayen et al.'s eq. 6 with gamma_1 and the proton's radius against the README's F: the normalised spectra."""
    m0, m1 = spectrum_moments()
    r0, r1 = moments(lambda t: allowed(t, hayen_rel))
    Fr = readme_shape()[0]
    T = T_TOP * np.concatenate([np.geomspace(1e-9, 1e-2, 36), np.linspace(0.01, 1, 400)])
    pe = np.sqrt(T * (T + 2 * ME_F))
    dev = np.abs(hayen_rel(ME_F + T, pe) / Fr(ME_F + T, pe) * m0 / r0 - 1)
    W = [(1 + mp.mpf(10)**k,) for k in (-4, -2, 0, mp.mpf("0.5"))]

    def results():
        fig = quoted(r"change the spectrum by at most \$" + FIG + r"\$", "The relativistic and finite-size terms of the Fermi")
        return [as_printed("the largest change of the normalised spectrum", dev.max(), fig,
                           f" at T = {1e3 * T[dev.argmax()]:.2g} keV; <T> {(r1 / r0) / (m1 / m0) - 1:+.1e}")]
    return [numeric("this numpy form = eq. 6 as transcribed (gamma_1 = sqrt(1 - alpha^2), R = sqrt(5/3) r_p)",
                    lambda w: hayen_F0(1, w, R_SPHERE, ALPHA),
                    lambda w: hayen_rel(float(w) * ME_F, float(mp.sqrt(w**2 - 1)) * ME_F), W, dps=20, tol=1e-12),
            *results(), wrong_figure("2.6e-4 for 2.4e-4", results, r"at most $2.4 \times 10^{-4}$",
                                     r"at most $2.6 \times 10^{-4}$")]


# --- the figures of the Results: the exact three-body decay (the proton's recoil) ---

def closed_form(lam, d, pn, pp, pe, pv, m_n, m_p):
    """|M|^2 of n -> p e nu-bar for the current p-bar gamma^mu (1 + lambda g5) n times e-bar gamma_mu (1 - g5) nu,
    summed over the spins (averaged over the neutron's), up to the factor 32; d is the Minkowski product."""
    return ((1 - lam)**2 * d(pn, pv) * d(pp, pe) + (1 + lam)**2 * d(pn, pe) * d(pp, pv)
            + (lam**2 - 1) * m_n * m_p * d(pe, pv))


def M2(Ee, Ev, lam, m_n=MN_F, m_p=MP_F, m_e=ME_F):
    """The closed form in the energies of the neutron's rest frame (all on shell, the antineutrino massless)."""
    Ep = m_n - Ee - Ev
    return ((1 - lam)**2 * m_n * Ev * (m_n**2 - 2 * m_n * Ev - m_p**2 - m_e**2) / 2
            + (1 + lam)**2 * m_n * Ee * (m_n**2 + m_e**2 - 2 * m_n * Ee - m_p**2) / 2
            + (lam**2 - 1) * m_n * m_p * (m_n**2 + m_p**2 - 2 * m_n * Ep - m_e**2) / 2)


@derivation("nd.trace")
def check_trace():
    """The |M|^2 of the exact decay (nd.recoil) is the V-A trace with explicit Dirac matrices, the proton recoiling."""
    L.line("This matrix element is the V−A trace with explicit Dirac matrices, exactly.", "checks")
    lam = sp.Symbol("lambda", real=True)
    Ee, pe, Ev, ct, st, v = sp.symbols("E_e p_e E_nu c_t s_t v", positive=True)
    G, g5 = gammas()
    one = sp.eye(4)
    met = [1, -1, -1, -1]
    sl = lambda q_: sum((met[m] * q_[m] * G[m] for m in range(4)), sp.zeros(4))  # noqa: E731
    bar = lambda A: G[0] * A.H * G[0]                                              # noqa: E731
    d = lambda a, b_: a[0] * b_[0] - a[1] * b_[1] - a[2] * b_[2] - a[3] * b_[3]   # noqa: E731
    pn, pe4, pv4 = [mn, 0, 0, 0], [Ee, 0, 0, pe], [Ev, Ev * st, 0, Ev * ct]
    pp4 = [pn[i] - pe4[i] - pv4[i] for i in range(4)]       # the proton (on shell only below)
    Lmn = [[(G[m] * (one - g5) * sl(pv4) * bar(G[n] * (one - g5)) * (sl(pe4) + me * one)).trace()
            for n in range(4)] for m in range(4)]
    Hmn = [[(G[m] * (one + lam * g5) * (sl(pn) + mn * one) / 2 * bar(G[n] * (one + lam * g5)) * (sl(pp4) + mp_ * one)).trace()
            for n in range(4)] for m in range(4)]
    tr = sum(met[m] * met[n] * Lmn[m][n] * Hmn[m][n] for m in range(4) for n in range(4))
    cf = closed_form(lam, d, pn, pp4, pe4, pv4, mn, mp_)
    # on shell: p_p^2 = m_p^2 fixes cos(theta_e-nu); the electron E_e, p_e = m_e (v^2 +- v^-2)/2 (rational in v)
    c_on = ((mn - Ee - Ev)**2 - Ev**2 - pe**2 - mp_**2) / (2 * pe * Ev)
    on = {Ee: me * (v**2 + v**-2) / 2, pe: me * (v**2 - v**-2) / 2}
    on_shell = cf.subs(st, sp.sqrt(1 - ct**2)).subs(ct, c_on).subs(on)
    return [exact("spin sum (the neutron's averaged) = 32 x the closed form, for any momenta", tr - 32 * cf),
            exact("on shell (p_n at rest): the closed form = its form in E_e, E_nu (the Dalitz plot of nd.recoil)",
                  on_shell - M2(on[Ee], Ev, lam, mn, mp_, me)),
            exact_fails("control: (1 - lambda)^2 and (1 + lambda)^2 swapped (the other sign convention of lambda)",
                        tr - 32 * closed_form(-lam, d, pn, pp4, pe4, pv4, mn, mp_))]


X4, W4 = np.polynomial.legendre.leggauss(4)
X64, W64 = np.polynomial.legendre.leggauss(64)


def electron_exact(T, lam):
    """dN/dT of the electron in the exact decay: F times |M|^2 integrated along the Dalitz line at fixed E_e (quadratic
    in E_nu there: 4 Gauss-Legendre points are exact)."""
    Ee, pe = ME_F + T, np.sqrt(T * (T + 2 * ME_F))
    mpv = np.sqrt(MN_F**2 + ME_F**2 - 2 * MN_F * Ee)        # the mass of the proton and the antineutrino
    Es_ = (mpv**2 - MP_F**2) / (2 * mpv)                       # the antineutrino's energy in their frame
    lo, hi = Es_ * (MN_F - Ee - pe) / mpv, Es_ * (MN_F - Ee + pe) / mpv
    Ev = 0.5 * (hi - lo) * X4 + 0.5 * (hi + lo)
    return readme_shape()[0](Ee, pe) * np.sum(W4 * M2(Ee, Ev, lam)) * 0.5 * (hi - lo)


def nu_exact(Ev, lam):
    """dN/dE_nu in the exact decay: F |M|^2 integrated along the line at fixed E_nu, in the electron's momentum
    (F dE = F p dp/E is smooth where F has its 1/p)."""
    mpe = np.sqrt(MN_F**2 - 2 * MN_F * Ev)                    # the mass of the proton and the electron
    Es_ = (mpe**2 + ME_F**2 - MP_F**2) / (2 * mpe)
    ps_ = np.sqrt(max(Es_**2 - ME_F**2, 0.0))
    Elo, Ehi = ((MN_F - Ev) * Es_ - Ev * ps_) / mpe, ((MN_F - Ev) * Es_ + Ev * ps_) / mpe
    plo, phi = np.sqrt(max(Elo**2 - ME_F**2, 0.0)), np.sqrt(max(Ehi**2 - ME_F**2, 0.0))
    pe = 0.5 * (phi - plo) * X64 + 0.5 * (phi + plo)
    Ee = np.sqrt(pe**2 + ME_F**2)
    return np.sum(W64 * readme_shape()[0](Ee, pe) * pe / Ee * M2(Ee, Ev, lam)) * 0.5 * (phi - plo)


@functools.cache
def exact_decay(lam):
    """(int dN, int T dN) of the electron and (int dN, int E dN) of the antineutrino of the exact decay."""
    top_nu = (MN_F**2 - (MP_F + ME_F)**2) / (2 * MN_F)     # its Dalitz boundary closes like a square root there
    return (tuple(moments(lambda t: electron_exact(t, lam))),
            tuple(moments(lambda t: nu_exact(t, lam), top=top_nu, epsrel=1e-10)))


@derivation("nd.recoil")
def check_recoil():
    """The README's spectra (no recoil) against the exact three-body decay; the proton's mean and largest energy."""
    lam = figure(quoted(r"using \$\\lambda = g_A/g_V = " + FIG + r"\$ of PDG 2026", "the exact three-body decay is"))[0]
    (e0, e1), (v0, v1) = exact_decay(lam)
    m0, m1 = spectrum_moments()
    T_mean, T_ex, Ev_ex = m1 / m0, e1 / e0, v1 / v0
    Tp = MN_F - MP_F - ME_F - T_ex - Ev_ex                 # the proton's mean kinetic energy (energy conservation)
    T_code = MN_F - MP_F - E0_F                            # every proton's in the code

    @functools.cache
    def worst(lo, hi):
        u = T_TOP * np.concatenate([np.geomspace(1e-4, 1e-2, 20) if lo == 0 else [], np.linspace(max(lo, 0.01), hi, 180)])
        return (max(abs(allowed(t) / m0 / (electron_exact(t, lam) / e0) - 1) for t in u),
                max(abs(allowed(T_TOP - t) / m0 / (nu_exact(t, lam) / v0) - 1) for t in u))
    beyond = "; from 90 to 99 %%: %.2g, %.2g" % worst(0.9, 0.99)
    E0 = readme_E0()[0]

    def results():
        fig, pct = quoted(r"differ by up to \$" + FIG + r"\$ below \$" + PCT + r"\$ of the end point", "the normalised spectra differ")
        we, wv = worst(0, figure(pct)[0] / 100)
        de, dv = quoted(r"mean energy differ by \$" + FIG + r"\$ and \$" + FIG + r"\$", "the antineutrino's mean energy differ by")
        tp, tc = quoted(r"The proton's mean kinetic energy is (\d+) eV, whereas the code gives every proton (\d+) eV",
                        "whereas the code gives every proton")
        dx = quoted(r"changes the proton's energy fraction by \$" + FIG + r"\$", "changes the proton's energy fraction by")
        most, ds = quoted(r"the proton recoils with at most ([\d.]+) eV \(Dubbers & Schmidt: (\d+) eV\)", "recoils with at most")
        return [Result("lambda of the README = PDG 2026's", lam == float(LAMBDA), f"{lam}"),
                as_printed("normalised spectra, the README's against the exact: largest |ratio - 1| below that",
                           max(we, wv), fig, f" (electron {we:.2g}, antineutrino {wv:.2g}{beyond})"),
                as_printed("<T> of the README's spectrum / the exact one - 1", T_mean / T_ex - 1, de),
                as_printed("<E_nu-bar> of the README's spectrum / the exact one - 1", (T_TOP - T_mean) / Ev_ex - 1, dv),
                as_printed("the proton's mean kinetic energy [eV]", 1e6 * Tp, tp),
                as_printed("the code's m_n - m_p - E_0 [eV]", 1e6 * T_code, tc,
                           f"; the antineutrino's end point {1e3 * (MN_F**2 - (MP_F + ME_F)**2) / (2 * MN_F):.4f} keV, "
                           f"the README's {1e3 * T_TOP:.4f} keV"),
                as_printed("x_p of the code - the exact mean (m_p + <T_p>)/m_n", (T_code - Tp) / MN_F, dx,
                           f"; electron {(T_mean - T_ex) / MN_F:+.1e}, antineutrino {(T_TOP - T_mean - Ev_ex) / MN_F:+.1e}"),
                as_printed("Inputs: the largest kinetic energy of the proton [eV]", 1e6 * T_code, most),
                Result("Dubbers & Schmidt's figure is it, to the two digits they give", round(1e6 * T_code, -1) == float(ds), ds)]
    return [exact("m_n - m_p - E_0 = the proton's largest kinetic energy (against e + nu-bar of the mass m_e)",
                  mn - mp_ - E0 - ((mn**2 + mp_**2 - me**2) / (2 * mn) - mp_)), *results(),
            wrong_figure("-6.6e-4 for -5.6e-4", results, r"and $-5.6 \times 10^{-4}$", r"and $-6.6 \times 10^{-4}$")]


# --- the figures of the Results: the radiative corrections ---

def hayen_50_tex():
    """Eqs. 48 (the outer factor), 50 and 51 as their TeX source writes them (g_sirlin transcribes them)."""
    return (r"1+\frac{\alpha}{2\pi}\left[g(W_0,W)-3\ln\frac{m_p}{2W_0}\right]",
            r"g(W_0, W) =~ &3 \ln (m_p) - \frac{3}{4} + \frac{4}{\beta} \text{L}_s \left(\frac{2 \beta}{1 + \beta} \right)"
            r" + 4 \left(\frac{\tanh^{-1} \beta}{\beta} - 1 \right) \left[\frac{W_0 - W}{3 W} - \frac{3}{2}"
            r" + \ln [ 2 (W_0 - W)] \right]",
            r"& + \frac{\tanh^{-1} \beta}{\beta} \left[ 2 (1+\beta^2) + \frac{(W_0-W)^2}{6W^2} - 4 \tanh^{-1} \beta \right]",
            r"\text{L}_s = \int_0^x \frac{\ln(1-t)}{t} dt = - \sum_{k=1}^{k=\infty} \frac{x^k}{k^2}  \equiv -\text{Li}_2(x)")


def g_sirlin(W, W0):
    """Hayen et al.'s eq. 50 (W in units of m_e, beta = p/W) without its constant 3 ln(m_p) - 3/4, which drops out
    of the normalised spectrum. L_s(x) = -Li_2(x) (eq. 51); scipy's spence(1 - x) is Li_2(x)."""
    b_ = np.sqrt(W * W - 1) / W
    at = np.arctanh(b_)
    Ls = -special.spence(1 - 2 * b_ / (1 + b_))
    return (4 / b_ * Ls + 4 * (at / b_ - 1) * ((W0 - W) / (3 * W) - 1.5 + np.log(2 * (W0 - W)))
            + at / b_ * (2 * (1 + b_ * b_) + (W0 - W)**2 / (6 * W * W) - 4 * at))


@derivation("nd.sirlin")
def check_sirlin():
    """Sirlin's outer radiative correction 1 + (alpha/2 pi) g(W_0, W) (Hayen et al., eqs. 48, 50-51) on the README's spectrum."""
    corr = lambda T: 1 + AL_F / (2 * np.pi) * g_sirlin((ME_F + T) / ME_F, E0_F / ME_F)  # noqa: E731
    m0, m1 = spectrum_moments()
    c0, c1 = moments(lambda t: allowed(t) * corr(t), points=[T_TOP * 0.999])   # ln(W_0 - W): integrable
    ratio = lambda T: corr(T) * m0 / c0  # noqa: E731  (the normalised spectrum with the correction / without)
    others = ", ".join(f"{100 * (ratio(T) - 1):+.2f}% at {1e3 * T:.0f} keV" for T in (0.1, 0.3, 0.6))
    pin = "The change would range from"

    def results():
        lo, klo, hi, khi = quoted(r"range from \$" + PCT + r"\$ at (\d+) keV to \$" + PCT + r"\$ at (\d+) keV", pin)
        dT = quoted(r"mean kinetic energy would change by \$" + FIG + r"\$", pin)
        return [as_printed(f"the normalised spectrum changes at {klo} keV by [%]", 100 * (ratio(1e-3 * float(klo)) - 1), lo),
                as_printed(f"and at {khi} keV by [%]", 100 * (ratio(1e-3 * float(khi)) - 1), hi, f"; {others}"),
                as_printed("<T> changes by", (c1 / c0) / (m1 / m0) - 1, dT)]
    return [in_source("Hayen et al. eqs. 48 (the outer factor), 50 and 51 as transcribed", HAYEN, *hayen_50_tex()),
            numeric("-spence(1 - x) = L_s(x) = int_0^x ln(1 - t)/t dt (eq. 51)", lambda x_: -special.spence(1 - float(x_)),
                    lambda x_: mp.quad(lambda t: mp.log(1 - t) / t, [0, x_]),
                    [(mp.mpf(k) / 10,) for k in (1, 3, 5, 7, 9)], dps=20, tol=1e-12),
            *results(), wrong_figure("+0.6% for +0.4% at 10 keV", results, r"$+0.4\text{％}$ at 10 keV",
                                     r"$+0.6\text{％}$ at 10 keV")]


HAYEN_BOUND = r"For free neutron decay, this ratio is approximately $4.2\cdot 10^{-6}$"


@derivation("nd.bound")
def check_bound():
    """Bound decay into a hydrogen atom: the README's share of the decays is Hayen et al.'s ratio of bound to free decay."""
    theirs = HAYEN_BOUND.split("$")[1].replace(r"\cdot", r" \times")

    def results():
        ours = quoted(r"Bound decay into a hydrogen atom, \$" + FIG + r"\$ of the decays", "Bound decay into a hydrogen atom")
        return [Result("the README's figure is theirs", figure(ours) == figure(theirs), f"{ours} and {theirs}")]
    return [in_source("Hayen et al.: the ratio of bound to free decay", HAYEN, HAYEN_BOUND), *results(),
            wrong_figure("4.3e-6 for 4.2e-6", results, r"$4.2 \times 10^{-6}$ of the decays", r"$4.3 \times 10^{-6}$ of the decays")]


# --- the figures of the Results: the other codes ---

def athena_tex():
    """Dimitrakoudis et al. 2012, Sec. 2.2: the mean life, and eq. 23 (protons of the neutron's Lorentz factor gamma)."""
    return (r"with a mean lifetime of $\tau=881.5 \pm 1.5 s $",
            r"Q_{\rm p}^{\rm n \rightarrow p}(\gamma,t) = {n_{\rm n}(\gamma,t) \over {\gamma \tau}}")


@derivation("nd.athena")
def check_athena():
    """ATHEnA's protons of the neutron's Lorentz factor (no electron, no antineutrino) and its tau, against the README's."""
    E0 = readme_E0()[0]
    xp = L.to_sympy(L.sides(L.inline(r"x_p = (m_n - E_0)/m_n"))[1], {"m_n": mn, "E_0": E0})
    m0, m1 = spectrum_moments()
    leptons = ((ME_F + m1 / m0) + (T_TOP - m1 / m0)) / MN_F        # their mean lab fractions (nd.means)
    theirs = figure(re.search(r"\\tau=([\d.]+) ", athena_tex()[0]).group(1))
    pin = "s protons, with the neutron's Lorentz factor, take"

    def results():
        dx = quoted(r"take \$m_p/m_n\$ of its energy, \$" + FIG + r"\$ less than in the code", pin)
        out = quoted(r"leaves out carry \$" + FIG + r"\$", pin)
        tau_a, slower = quoted(r"Its \$\\tau = " + FIG + r"\$ s .*? gives decay rates \$" + PCT + r"\$ lower", pin)
        tau_n = quoted(r"the mean life \$\\tau_n = " + FIG + r"\$ s", "A free neutron decays", "README")
        return [as_printed("x_p - m_p/m_n with PDG 2026", (MN_F - E0_F - MP_F) / MN_F, dx),
                as_printed("the electrons and antineutrinos: 1 - x_p = E_0/m_n", E0_F / MN_F, out,
                           f" (their mean lab fractions by quadrature: {leptons:.6g})"),
                Result("ATHEnA's tau = Dimitrakoudis et al.'s", figure(tau_a) == theirs, tau_a),
                as_printed("its decay rates are lower by [%]", 100 * (1 - figure(tau_n)[0] / figure(tau_a)[0]), slower)]
    return [in_source("Dimitrakoudis et al. Sec. 2.2: tau and eq. 23", DIMITRAKOUDIS, *athena_tex()),
            exact("ATHEnA's x = gamma m_p/(gamma m_n): the README's x_p - m_p/m_n = ((m_n - m_p)^2 - m_e^2)/(2 m_n^2) > 0",
                  xp - mp_ / mn - ((mn - mp_)**2 - me**2) / (2 * mn**2)),
            Result("the electron and the antineutrino carry E_0/m_n (energy conservation)", abs(leptons * MN_F / E0_F - 1) < 1e-12,
                   f"{leptons * MN_F / E0_F - 1:.1e}"),
            *results(), wrong_figure("9e-7 for 8e-7", results, r"$8 \times 10^{-7}$ less", r"$9 \times 10^{-7}$ less")]


@derivation("nd.line")
def check_line():
    """Huemmer et al.'s line against the README's spectrum and Q_X, for decays Q_n ∝ E_n^-q (beta_n -> 1).
    The ratio of the README's Q_X to the line is (2/m_n)^(q-1) <E*^(q-1)>/(q chi^(q-1))."""
    En, tau, cl = sp.symbols("E_n tau_n c_light", positive=True)
    q_, chi = sp.symbols("q chi", positive=True)
    Nn, FX = sp.Function("N_n"), sp.Function("F_X")
    weight, integrand = readme_source(En, Nn, FX, 1)
    lo, hi = (s.subs({b: 1, ps: Es}) for s in readme_box())        # a massless product, beta_n -> 1
    dens = readme_density().subs({b: 1, ps: Es})
    body = (weight * integrand).subs(Nn(En), En**-q_ * En / (mn * cl**2) * tau)   # N_n/(gamma_n tau_n) = E_n^-q
    body = body.replace(lambda a: isinstance(a, sp.Function) and a.func == FX, lambda a: dens)
    ours = sp.integrate(body, (En, sp.solve(sp.Eq(E / En, hi), En)[0], sp.oo))    # where x = E/E_n <= 2E*/m_n
    line = L.to_sympy(L.sides(L.inline(r"Q_{\bar\nu}(E) = Q_n(E/\chi)/\chi", "checks"))[1], {r"\chi": chi, "E": E},
                      funcs={"Q_n": lambda a: a**-q_})
    m0 = spectrum_moments()[0]

    def ratio(q, chi_):
        mom = moments(lambda t: allowed(t) * (T_TOP - t)**(q - 1), k=1)[0] / m0
        return (2 / MN_F)**(q - 1) * mom / (q * chi_**(q - 1))
    pin = "times as many antineutrinos as the line"

    def results():
        chi_ = figure(quoted(r"line at \$\\chi = " + FIG + r"\$", pin))[0]
        q0, tol = quoted(r"decays \$\\propto E_n\^\{-([\d.]+)\}\$ \(per unit energy\) to \$" + FIG + r"\$", pin)
        q1, q2, q3, r1, r2, r3 = quoted(r"decays \$\\propto E_n\^\{-([\d.]+)\}\$, \$E_n\^\{-([\d.]+)\}\$ and "
                                        r"\$E_n\^\{-([\d.]+)\}\$ make ([\d.]+), ([\d.]+) and ([\d.]+) times as many", pin)
        note = (f" (chi/<x> = {chi_ / mean_fraction():.4f}; q = 1, 1.5: {ratio(1.0, chi_):.4f}, "
                f"{ratio(1.5, chi_):.4f})")
        return [as_printed(f"q = {q0}: |ratio - 1|", abs(ratio(float(q0), chi_) - 1), tol, note),
                *(as_printed(f"q = {qq}: ratio", ratio(float(qq), chi_), rr) for qq, rr in ((q1, r1), (q2, r2), (q3, r3)))]
    return [exact("the range of a massless product starts at x = 0 for beta_n -> 1", lo),
            exact("README Q_X / the line = (2E*/m_n)^(q-1)/(q chi^(q-1)), any q > 0",
                  ours / line - (2 * Es / mn)**(q_ - 1) / (q_ * chi**(q_ - 1))),
            *results(), wrong_figure("1.40 for 1.50", results, "make 1.19, 1.50 and 1.97", "make 1.19, 1.40 and 1.97")]


if __name__ == "__main__":
    run()
