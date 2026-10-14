"""Derivation checks of the kaon evolution equations of README.md (Evolution) and of the claim, under the kaons'
Choice of method, that the kinetic equation gives Lipari, Lusignoli & Meloni's energy distribution at decay. Sources:
  - Lipari, Lusignoli & Meloni (2007), the TeX source of arXiv:0704.0718v1: eq. 32 (p. 9 of the arXiv PDF), the
    synchrotron loss averaged over the field's orientation (c = 1); eqs. 37-38 (p. 10), the energy at decay of a
    particle of mass m and mean life tau that loses energy at the rate f(E).
  - Physics without a paper (Gaussian units): the synchrotron loss is Larmor's formula averaged over isotropic pitch
    angles. The Thomson inverse-Compton loss is the power scattered (the same in the rest frame and in the lab) less
    that of the photons removed (Rybicki & Lightman, Sec. 7.2).
Both losses use the Thomson cross section of the particle's mass, sigma_T (m_e/m)^2, and are (4/3) sigma c U gamma^2
beta^2; the README writes their limit gamma >> 1. AM3's inverse-Compton loss tends to the same loss for any mass
(am3_cooling.py, am3.sigmaT).

The kinetic equation is AM3's (a model of the code). Its decay term is the time-dilated rate. With cooling
Edot = -f(E) and decay alone, its steady state for one particle injected per unit time at E_i decays at Lipari et al.'s
eq. 38, for any f. With synchrotron cooling (f = a E^2), every injected particle decays.

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/kaon_evolution.py
"""
import re

import sympy as sp

import latex as L
from latex import LIPARI
from verify import Result, derivation, exact, exact_fails, in_source, run

LIP_32 = (r"after  averaging over all possible orientation of the  magnetic  field  direction is given by the well known "
          r"expression:", r"-\frac{dE}{dt}  = \frac{4}{9} \frac{e^4 \, B^2}{m^4}\; E^2")
LIP_37 = r"-\frac{dE}{dt} (E) = f (E) ~,"
LIP_38 = (r"\frac{dp_{\rm dec}}{dE} (E; E_i) = \frac{m}{\tau} \; \frac{1}{f(E) \; E} ~\exp \left [-\frac{m}{\tau} \; "
          r"\int_{E}^{E_i} \frac{dE^\prime} {f(E^\prime) \, E^\prime} \right ].")
LIP_TEXT = (r"probability   density  for an unstable  particle  of  mass $m$ and lifetime $\tau$, created  with initial  "
            r"energy $E_i$ to  decay   with energy $E$.")

E, Ei, Ep, m, tau, cl, a = sp.symbols("E E_i E' m tau c a", positive=True)
f, N = sp.Function("f"), sp.Function("N")
q, B, me, mK, U = sp.symbols("q_e B m_e m_K U", positive=True)    # charge, field, masses, photon energy density
v_ = sp.Symbol("v", positive=True)
GAM, GB = (v_ + 1 / v_) / 2, (v_ - 1 / v_) / 2                     # gamma and gamma beta: (gamma^2 - 1)^(1/2) = (v - 1/v)/2
BETA = GB / GAM


def sigma_T(mass, cc=cl):
    """The Thomson cross section of a charge q_e of the given mass, (8 pi/3) (q_e^2/(mass c^2))^2."""
    return sp.Rational(8, 3) * sp.pi * (q**2 / (mass * cc**2))**2


# --- the kinetic equation (Evolution) and Lipari et al.'s eq. 38 ---

KIN_PIN = r"\frac{\partial N_s}{\partial t} = -\frac{\partial}{\partial E}"
COOLING = r"-\frac{\partial}{\partial E}\Big(\dot E\thinspace N_s\Big)"   # the README's cooling term, asserted as written


def readme_kinetic(cc=1):
    """The README's right-hand side for N_s = N(E) with Edot = -f(E), without escape (t_esc -> oo) and source (below E_i),
    and its gamma; c = cc."""
    disp = L.display(KIN_PIN)
    lhs, rhs = L.sides(L.parts(disp)[0])
    assert lhs == r"\frac{\partial N_s}{\partial t}" and rhs.startswith(COOLING), rhs
    gam = L.to_sympy(L.sides(L.part(disp, r"\gamma ="))[1], {"E": E, "m_s": m, "c": cc})
    rest = L.to_sympy(rhs[len(COOLING):], {"N_s": N(E), r"t_{\rm esc}": sp.oo, r"\gamma": gam, r"\tau_s": tau, "Q_s": 0})
    return -sp.diff(-f(E) * N(E), E) + rest, rest, gam


def lipari_38():
    """Lipari et al.'s eq. 38 (c = 1), from the TeX source: the prefactor parsed, the integral in the exponent asserted."""
    pre = L.to_sympy(r"\frac{m}{\tau} \; \frac{1}{f(E) \; E}", {"m": m, r"\tau": tau, "E": E}, {"f": f})
    assert r"\int_{E}^{E_i} \frac{dE^\prime} {f(E^\prime) \, E^\prime}" in LIP_38
    inner = L.to_sympy(r"\frac{1} {f(E^\prime) \, E^\prime}", {r"E^\prime": Ep}, {"f": f})
    return pre * sp.exp(L.to_sympy(r"-\frac{m}{\tau}", {"m": m, r"\tau": tau}) * sp.Integral(inner, (Ep, E, Ei)))


@derivation("ev.kinetic")
def check_kinetic():
    """With cooling and decay alone, the steady state of the kinetic equation decays at Lipari et al.'s eq. 38."""
    L.line(r"With synchrotron cooling and decay alone, this equation gives the energy distribution at decay of particles that "
           r"cool before they decay (Lipari, Lusignoli & Meloni 2007, eq. 38)")
    residual, rest, gam = readme_kinetic(cl)
    # one particle injected per unit time at E_i: below E_i, N = exp(-S)/f with S = int_E^E_i dE'/(gamma tau f), c = 1
    S = sp.Integral(m / (tau * Ep * f(Ep)), (Ep, E, Ei))
    steady = sp.exp(-S) / f(E)
    residual1, rest1, gam1 = readme_kinetic(1)
    on = lambda expr: expr.replace(N, lambda z: steady.subs(E, z)).doit()   # noqa: E731
    with L.patched(r"- \frac{N_s}{\gamma\thinspace \tau_s}", r"- \frac{N_s}{\tau_s}"):
        broken = readme_kinetic(1)[0]
    # synchrotron cooling, f = a E^2: the decays add up to the particles injected
    syn = lipari_38().subs(f(E), a * E**2).replace(f, lambda z: a * z**2).doit()
    total = sp.simplify(sp.integrate(syn, (E, 0, Ei)))
    return [in_source("eqs. 32, 37 and 38 and the definitions as transcribed", LIPARI, *LIP_32, LIP_37, LIP_38, LIP_TEXT),
            exact("the decay term: N/(gamma tau) with gamma = E/(m c^2), the decay rate in the lab (mean life gamma tau)",
                  rest / N(E) + m * cl**2 / (E * tau)),
            exact("N = exp(-S)/f, S = int_E^E_i m dE'/(tau E' f(E')): steady below E_i (any f; c = 1)", on(residual1)),
            exact("f(E_i) N(E_i) = 1: one particle injected per unit time at E_i flows down through E_i",
                  (f(E) * steady).subs(E, Ei).doit() - 1),
            exact("its decays per unit time and energy, N/(gamma tau), = Lipari et al.'s eq. 38", on(N(E) / (gam1 * tau)) - lipari_38().doit()),
            Result("synchrotron cooling, f = a E^2: int_0^E_i of eq. 38 = 1, every particle decays", total == 1, f"{total}"),
            exact_fails("control: N_s/tau_s (no time dilation)", on(broken))]


# --- the synchrotron and inverse-Compton losses ---

def readme_loss(prefix):
    """The README's -Edot after the prefix (a span '$-\\dot E$ $= ...$'), for a kaon of mass m_K and Lorentz factor gamma."""
    line = L.line(prefix)
    tex = re.search(re.escape(prefix) + r" \$-\\dot E\$ \$= ([^$]+)\$", line).group(1)
    tokens = {r"\sigma_T": sigma_T(me), "c": cl, "U_B": B**2 / (8 * sp.pi), r"U_\mathrm{rad}": U, "m_e": me, "m_K": mK,
              r"\gamma": GAM}
    return L.to_sympy(tex, tokens)


SYN = r"**Synchrotron:** cooling"
IC = r"The cooling on the target photons is continuous (in the Thomson regime"


@derivation("ev.syn")
def check_syn():
    """The kaons' synchrotron loss is Larmor's for the kaon's mass, averaged over pitch angles, in its limit gamma >> 1."""
    al = sp.Symbol("alpha", positive=True)   # the pitch angle
    a_perp = q * B * BETA * cl * sp.sin(al) / (GAM * mK * cl)   # |dv/dt| = (q/c) v B sin(alpha)/(gamma m) (Lorentz force)
    larmor = 2 * q**2 * GAM**4 * a_perp**2 / (3 * cl**3)        # Larmor's power for an acceleration normal to the velocity
    UB = B**2 / (8 * sp.pi)
    mean = sp.integrate(larmor * sp.sin(al), (al, 0, sp.pi)) / 2   # isotropic pitch angles
    readme = readme_loss(SYN)
    lip = L.to_sympy(LIP_32[1].split("=")[1], {"e": q, "B": B, "m": mK, "E": GAM * mK})   # c = 1, E = gamma m
    with L.patched(r"$= (4/3)\ \sigma_T c U_B (m_e/m_K)^2 \gamma^2$", r"$= (4/3)\ \sigma_T c U_B (m_e/m_K)^4 \gamma^2$"):
        broken = readme_loss(SYN)
    return [in_source("eq. 32 as transcribed", LIPARI, *LIP_32),
            exact("Larmor: 2 e^2 gamma^4 a_perp^2/(3 c^3) = 2 sigma_T,K c beta^2 gamma^2 U_B sin^2 alpha, sigma_T,K = sigma_T (m_e/m_K)^2",
                  larmor - 2 * sigma_T(mK) * cl * BETA**2 * GAM**2 * UB * sp.sin(al)**2),
            exact("sigma_T,K = (8 pi/3)(e^2/(m_K c^2))^2 = sigma_T (m_e/m_K)^2", sigma_T(mK) - sigma_T(me) * (me / mK)**2),
            exact("averaged over isotropic pitch angles: (4/3) sigma_T,K c beta^2 gamma^2 U_B",
                  mean - sp.Rational(4, 3) * sigma_T(mK) * cl * BETA**2 * GAM**2 * UB),
            exact("the README's (4/3) sigma_T c U_B (m_e/m_K)^2 gamma^2 is that with gamma^2 for beta^2 gamma^2 (gamma >> 1)",
                  readme * BETA**2 - mean),
            exact("= Lipari et al.'s eq. 32, (4/9) e^4 B^2 E^2/m^4 (c = 1, E = gamma m_K)", readme.subs(cl, 1) - lip),
            exact_fails("control: (m_e/m_K)^4", broken * BETA**2 - mean)]


@derivation("ev.ic")
def check_ic():
    """The kaons' Thomson loss in isotropic photons, (4/3) sigma_T,K c U gamma^2 beta^2, in its limit gamma >> 1."""
    mu, ct = sp.symbols("mu c_theta", real=True)   # cosines: photon's angle to the velocity (lab), scattering angle
    # in the rest frame, photons of lab direction mu have gamma (1 - beta mu) times their energy and density: the energy
    # density there, from isotropic photons of energy density U
    U_rest = GAM**2 * sp.integrate((1 - BETA * mu)**2, (mu, -1, 1)) / 2 * U
    emitted = sigma_T(mK) * cl * U_rest                          # Thomson scattering there; the lab power is the same
    removed = sigma_T(mK) * cl * sp.integrate(1 - BETA * mu, (mu, -1, 1)) / 2 * U   # the photons scattered, in the lab
    readme = readme_loss(IC)
    with L.patched(r"$= (4/3)\ \sigma_T c U_\mathrm{rad} (m_e/m_K)^2 \gamma^2$", r"$= (4/3)\ \sigma_T c U_\mathrm{rad} (m_e/m_K)^4 \gamma^2$"):
        broken = readme_loss(IC)
    return [exact("U' = gamma^2 (1 + beta^2/3) U", U_rest - GAM**2 * (1 + BETA**2 / 3) * U),
            exact("Thomson scattering is symmetric about the incident direction (dsigma/dOmega ∝ 1 + cos^2): no momentum in the "
                  "rest frame, so the lab power E/t = gamma E'/(gamma t') is the rest frame's", sp.integrate((1 + ct**2) * ct, (ct, -1, 1))),
            exact("the loss: sigma_T,K c U [gamma^2 (1 + beta^2/3) - 1] = (4/3) sigma_T,K c U gamma^2 beta^2",
                  emitted - removed - sp.Rational(4, 3) * sigma_T(mK) * cl * U * GAM**2 * BETA**2),
            exact("the README's (4/3) sigma_T c U_rad (m_e/m_K)^2 gamma^2 is that with gamma^2 for beta^2 gamma^2 (gamma >> 1)",
                  readme * BETA**2 - (emitted - removed)),
            exact_fails("control: (m_e/m_K)^4", broken * BETA**2 - (emitted - removed))]


if __name__ == "__main__":
    run()
