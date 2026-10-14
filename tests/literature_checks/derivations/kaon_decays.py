"""Derivation checks of the kaon decay equations of README.md (Decays) against their sources:
  - Lipari, Lusignoli & Meloni (2007), the TeX source of arXiv:0704.0718v1, read and parsed: eqs. 15-16 and 18-19
    (pi+ -> mu+ nu: the flat muon spectrum, the mu+ helicity, the spectra of right- and left-handed muons), with
    r_pi = (m_mu/m_pi)^2 and m_pi -> m_K for K+ -> mu+ nu, as they state. Their eq. 19 prints F(pi- -> mu+_R) for
    F(pi- -> mu-_R), a charge misprint (the CP relation after it has mu-) that does not enter here;
  - Chounet, Gaillard & Gaillard (1972, Phys. Rep. 4, 199), eqs. 6.6-6.9 (pp. 249-250), transcribed below
    from the scan in their notation: kaon k, M; pion q, mu; lepton mass m and energy E_mu (for K_e3 too);
    E_nu = M - E_pi - E_mu;
  - the PDG 2026 listings of K+- and K0_L, transcribed below with their printed page numbers. For K0_L -> 3pi0,
    whose listing gives only the coefficient h of u^2 (p. 22), the README's 1 + h(u^2 + v^2/3) is derived from
    PDG's expansion (p. 20).
The kinematics the README states without a source is redone exactly, from Q_X to the bin integrals and the
sharing between grid points. The V-A trace behind W_0 +- W_pol uses explicit Dirac matrices and the spin
projector (p_l-slash - m)(1 + g5 s-slash)/2 of the l+. K -> mu nu, where angular momentum makes the mu+
left-handed, fixes the sign convention of s physically. For an l- the trace is 4 [W_0 - W_pol(s)], so its
helicities are those of the l+ exchanged. The muon's transverse polarisation averages out over the
orientation of the decay plane.

At the end, branching_ratios() reads the README's branching ratios for kaon_production/multibody_strangeness.py.

Run from the repository root:
  .venv/bin/python tests/literature_checks/derivations/kaon_decays.py
"""
import functools
import re
from itertools import permutations

import sympy as sp

import latex as L
import mojo_constants
from latex import LIPARI
from pdg import MASS_MEV
from verify import Result, derivation, exact, exact_fails, in_source, mdot, run

M, m, x, Es, ps, q = sp.symbols("M m x E_s p_s q", positive=True)   # Es, ps: E*, p* in the kaon frame
c = sp.Symbol("c", real=True)                                        # cos(theta*) in the kaon frame
g_, G_, w_ = sp.symbols("g G w", real=True)
GAM, GB = (q + 1 / q) / 2, (q - 1 / q) / 2   # the kaon's gamma and gamma beta, q = e^rapidity -> oo


def boost(a):
    """A 4-vector of the kaon frame in the lab (the kaon moves along z)."""
    return (GAM * a[0] + GB * a[3], a[1], a[2], GAM * a[3] + GB * a[0])


# --- Lipari, Lusignoli & Meloni (2007), copied from the TeX source ---

LIP_15 = r"F_{\pi^+ \to \mu^+} (x) = {1 \over 1 -r_\pi } ~ \Theta (x-r_\pi)"
LIP_16 = r"h_{\pi^+ \to \mu^+}(x) = \frac{2\,r_\pi}{\left( 1 - r_\pi \right) \,x} -\frac{1 + r_\pi}{1 - r_\pi}"
LIP_18 = (r"F_{\pi^+ \to \mu^+_R} (x) = F_{\pi^- \to \mu^-_L} (x) = \frac{r_\pi (1-x)}{(1-r_\pi)^2 \, x}"
          r" ~ \Theta (x-r_\pi)")
LIP_19 = (r"F_{\pi^+ \to \mu^+_L} (x) = F_{\pi^- \to \mu^+_R} (x) = \frac{x - r_\pi}{(1-r_\pi)^2 \, x}"
          r" ~ \Theta (x-r_\pi)")
LIP_TEXT = [r"$r_\pi = (m_\mu/m_\pi)^2$", r"helicity  $h = P_R - P_L$,  where  $P_{R,L}$ is  the probability "
            r"that the muon has spin parallel (anti--parallel) to its momentum",
            r"takes the value $-1$ for $x \to 1$ (forward muon emission) and $+1$ for $x \to r_\pi$",
            r"the $\mu^+$ is created as a left--handed particle in the pion rest frame",
            r"$K^+ \to \mu^+ \nu_\mu$ (and charge conjugate), with the simple replacement $m_\pi \to m_K$"]


def lipari(snippet, r):
    """The right-hand side of one of the equations above (the step function cut off)."""
    rhs = snippet.split("=")[-1].replace(r"~ \Theta (x-r_\pi)", "").replace(r"{1 \over 1 -r_\pi }", r"\frac{1}{1 -r_\pi}")
    return L.to_sympy(rhs, {r"r_\pi": r, "x": x})


# --- Chounet, Gaillard & Gaillard (1972), transcribed from the scan ---

def cgg_rho(E_pi, E_mu, M_, mu, m_, fp, fm):
    """Eq. 6.9 (p. 249), rho ∝ A f+^2 + B f+ f- + C f-^2, with (p. 250) A = M(2 E_mu E_nu - M E'_pi)
    + m^2(E'_pi/4 - E_nu), B = m^2(E_nu - E'_pi/2), C = m^2 E'_pi/4, E'_pi = E_pi^max - E_pi =
    (M^2 + mu^2 - m^2)/(2M) - E_pi; eq. 6.8: <pi(q)|V|K(k)> ∝ (k + q) f+(t) + (k - q) f-(t);
    eq. 6.7: t = (k - q)^2 = M^2 + mu^2 - 2 M E_pi, m^2 < t < (M - mu)^2."""
    E_nu = M_ - E_pi - E_mu
    Ep = (M_**2 + mu**2 - m_**2) / (2 * M_) - E_pi
    A = M_ * (2 * E_mu * E_nu - M_ * Ep) + m_**2 * (Ep / 4 - E_nu)
    B = m_**2 * (E_nu - Ep / 2)
    C = m_**2 * Ep / 4
    return A * fp**2 + B * fp * fm + C * fm**2


# --- PDG 2026 listings, transcribed from the PDFs ---
# K± listing p. 37 (K±_l3 form factors; the K0_L listing p. 23 has the same with m_K0, m_pi+):
#   f0 = f+ + f- t/(m_K+^2 - m_pi0^2); "lambda+ and lambda0 are the linear expansion coefficients of
#   f+ and f0": f+(t) = f+(0)(1 + lambda+ t/m_pi+^2) (printed; f0(t) = f0(0)(1 + lambda0 t/m_pi+^2)
#   likewise, not printed); the quadratic expansion (as used by KTeV)
#   f+(t) = f+(0)(1 + lambda'+ t/m_pi+^2 + (lambda''+/2) t^2/m_pi+^4).
# K± listing p. 32, note "Dalitz plot parameters for K -> 3pi decays", eq. 1:
#   |M|^2 ∝ 1 + g (s3 - s0)/m_pi+^2 + h [(s3 - s0)/m_pi+^2]^2 + j (s2 - s1)/m_pi+^2
#           + k [(s2 - s1)/m_pi+^2]^2 + f (s2 - s1)/m_pi+^2 (s3 - s0)/m_pi+^2 + ...,
#   s_i = (P_K - P_i)^2 = (m_K - m_i)^2 - 2 m_K T_i (T_i the kinetic energy of pion i),
#   s0 = (1/3) sum_i s_i = (1/3)(m_K^2 + m_1^2 + m_2^2 + m_3^2), index 3 the odd pion. The listings'
#   headings (K± p. 33; K0_L p. 20: 1 + gu + hu^2 + jv + kv^2 + fuv) write u = (s3 - s0)/m_pi^2,
#   v = (s2 - s1)/m_pi^2.
# K0_L listing p. 22: "quadratic coefficient h for K0_L -> pi0 pi0 pi0": (0.59 +- 0.20 +- 1.16) x 10^-3.

def pdg_dalitz(s, mpi, g, h, j, k, f):
    """Eq. 1 of the note (p. 32), to second order."""
    s1, s2, s3 = s
    s0 = (s1 + s2 + s3) / 3
    U, V = (s3 - s0) / mpi**2, (s2 - s1) / mpi**2
    return 1 + g * U + h * U**2 + j * V + k * V**2 + f * V * U


# --- the README, read from README.md ---

def span(prefix, file="README"):
    """The inline math after the prefix (which ends with $), on the line holding the prefix."""
    ln = L.line(prefix, file)
    i = ln.index(prefix) + len(prefix)
    return ln[i:ln.index("$", i)]


PRODUCT = {r"E^\ast": Es, r"p^\ast": ps, "M": M, "m": m, "x": x}


def readme_range():
    tex = span("is spread uniformly over $")
    assert tex.startswith(r"x \in [") and tex.endswith("]"), tex
    return [L.to_sympy(e, PRODUCT) for e in L.split_top(tex[len(r"x \in ["):-1], ",")]


def readme_density():
    return L.to_sympy(span("/M]$ with density $"), PRODUCT)   # after the range (the neutron decay's has a density too)


def readme_htilde(E=Es, p=ps):
    return L.to_sympy(span(r"with $\tilde h = "), {**PRODUCT, r"E^\ast": E, r"p^\ast": p})


def F_LINE():
    return L.display(r"F(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}")


def FRL_LINE():
    return L.display(r"F_{R,L}(x) = \frac{M}{2}\int")


def readme_Elo():
    p = L.sides(L.part(F_LINE(), r"E_{\rm lo}(x)"))
    assert p[0] == r"E_{\rm lo}(x)" and p[1].endswith(r"\ \ (\ge m)"), p
    return L.to_sympy(p[1][:-len(r"\ \ (\ge m)")], {"M": M, "m": m, "x": x})


def readme_F(sign=None, g=g_, G=G_, ht=None, E=Es, p=ps):
    """Prefactor times integrand of the README's F (sign None) or F_R (+1), F_L (-1) at the node E* = E,
    p* = p; the limits of the integral are asserted as written."""
    part, lhs = (L.parts(F_LINE())[0], "F(x)") if sign is None else (L.parts(FRL_LINE())[0], "F_{R,L}(x)")
    left, right = L.sides(part)
    pre, integrand = right.split(r"\int_{E_{\rm lo}(x)}^{E_{\max}} ")
    assert left == lhs and integrand.endswith(r"\thinspace dE^\ast"), part
    funcs = {"g": lambda a: g, "G": lambda a: G, r"\tilde h": lambda a, b: ht}
    tokens = {r"E^\ast": E, "x": x, r"p^\ast": p}
    return L.to_sympy(pre, {"M": M}) * L.to_sympy(integrand[:-len(r"\thinspace dE^\ast")], tokens, funcs, sign)


def readme_two_body():
    """E* and p* of the README's two-body decay into a massless partner (m_b = 0)."""
    E0 = L.to_sympy(L.sides(span(r"Two-body decays have one $"))[1], {"M": M, "m": m, "m_b": 0})
    return E0, (M**2 - m**2) / (2 * M)


def readme_P():
    return L.to_sympy(L.sides(span(r"For $\mathrm{K} \to \mu\nu$, $"))[1], {})


DOT = re.compile(r"\(([^()]+?)\\negthinspace \\cdot\\negthinspace ([^()]+?)\)")


def readme_W(vectors, dot, ml):
    """W_0 and (if s is given) W_pol(s) of the README for the 4-vectors p_nu, p_l, H, s (dot: their
    scalar product); each (a . b) is replaced by its value before parsing."""
    line = L.display(r"\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s)")
    assert L.parts(line)[0] == r"\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s)", line
    out = []
    for lhs in ("W_0 =", r"W_{\rm pol}(s) =")[:2 if "s" in vectors else 1]:
        tokens = {r"m_\ell": ml, "H^2": dot(vectors["H"], vectors["H"])}

        def value(match):
            name = r"\Xi_{%d}" % len(tokens)
            tokens[name] = dot(vectors[match.group(1)], vectors[match.group(2)])
            return " " + name + " "
        out.append(L.to_sympy(DOT.sub(value, L.sides(L.part(line, lhs))[1]), tokens))
    return out


def readme_H(fp, fm, pK, ppi):
    tex = span(r"with the hadronic current $")
    return L.to_sympy(L.sides(tex)[1], {"p_K": pK, r"p_\pi": ppi, "t": sp.Symbol("t")},
                      funcs={"f_+": lambda a: fp, "f_-": lambda a: fm})


def DALITZ():
    return L.display(r"\vert \mathcal M\vert^2 \propto 1 + g\thinspace u + h\thinspace u^2 + k\thinspace v^2")


def readme_uv(s, mpi):
    """u and v of the README in the s_i (s_0 = (s_1 + s_2 + s_3)/3, as written)."""
    line = DALITZ()
    assert L.part(line, "s_i =") == "s_i = (p_K - p_i)^2", line
    third, rest = L.sides(L.part(line, "s_0 ="))[1].split(r"\textstyle")
    assert rest == r"\sum_i s_i", rest
    tokens = {"s_1": s[0], "s_2": s[1], "s_3": s[2], "s_0": L.to_sympy(third, {}) * sum(s), r"m_{\pi^+}": mpi}
    return [L.to_sympy(L.sides(L.part(line, n + " ="))[1], tokens) for n in "uv"]


def BIN(name):
    pin = {"N": r"N = \frac{M}{2p^\ast}\left[w_g\thinspace (x_b - x_a)",
           "E": r"E = \frac{M}{2p^\ast}\left[w_g\thinspace \frac{x_b^2 - x_a^2}{2}"}[name]
    return L.sides(L.parts(L.display(pin, "checks"))[0])[1]


x1, x2 = sp.symbols("x_a x_b", positive=True)
wg, wG = sp.symbols("w_g w_G", real=True)


def readme_bin(name):
    return L.to_sympy(BIN(name), {"M": M, r"p^\ast": ps, "w_g": wg, "w_G": wG, "x_a": x1, "x_b": x2,
                                  r"E^\ast": Es, "m": m})


def laurent_integral(expr, var, lo, hi):
    """The integral from lo to hi of a Laurent polynomial in var, term by term (1/var gives ln(hi/lo))."""
    total = 0
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff, n = term.as_coeff_exponent(var)
        assert not coeff.has(var), term
        total += coeff * (sp.log(hi / lo) if n == -1 else (hi**(n + 1) - lo**(n + 1)) / (n + 1))
    return total


def node_spectrum():
    """dn/dx of one node: flat (M/2p*, kd.flat) with weight w_g plus the helicity term h~ with weight w_G."""
    return readme_density() * (wg + wG * readme_htilde())


# --- the checks ---

def readme_decay_source(EK, Ns, FX, tau):
    """The README's Q_X(E) of one species s as (the weight of its measure dE_K, its integrand), for any functions N_s,
    F_s->X; gamma_K is the README's gamma = E/(m_s c^2) of the kaon (Evolution)."""
    lhs, rhs = L.sides(L.parts(L.display(r"Q_X(E) = \sum_s \int_E^\infty"))[0])
    head, rest = rhs.split(r"\int_E^\infty ", 1)
    measure, body = rest.split(r"\thinspace ", 1)
    assert lhs == "Q_X(E)" and head.strip() == r"\sum_s", rhs
    ms, cl, dEK = sp.symbols("m_s c dE_K", positive=True)
    gamma = L.to_sympy(L.sides(L.part(L.display(r"\frac{\partial N_s}{\partial t} = "), r"\gamma ="))[1], {"E": EK, "m_s": ms, "c": cl})
    weight = sp.diff(L.to_sympy(measure, {"dE_K": dEK, "E_K": EK}), dEK)
    tokens = {"E": sp.Symbol("E", positive=True), "E_K": EK, r"\gamma_K": gamma, r"\tau_s": tau}
    return weight, L.to_sympy(body, tokens, {"N_s": Ns, r"F_{s\to X}": FX}), gamma


@derivation("kd.source")
def check_source():
    """Q_X counts the decays: N_s dE_K kaons decay at the time-dilated rate, each making F dx products, x = E/E_K."""
    EK, tau, ms, cl = sp.symbols("E_K tau_s m_s c", positive=True)
    E = sp.Symbol("E", positive=True)
    Ns, FX = sp.Function("N_s"), sp.Function("F")
    weight, body, gamma = readme_decay_source(EK, Ns, FX, tau)
    xv = L.to_sympy(L.sides(L.inline(r"x = E/E_K"))[1], {"E": E, "E_K": EK})
    # per dE_K: N_s dE_K kaons decay at 1/(gamma tau_s) (mean life gamma tau_s in the lab), each making F(x) dx
    # products; dx = (dx/dE) dE at fixed E_K
    counted = Ns(EK) / (EK / (ms * cl**2) * tau) * FX(xv) * sp.diff(xv, E)
    rate = L.to_sympy(L.inline(r"1/(\gamma\tau_s)"), {r"\gamma": gamma, r"\tau_s": tau})
    with L.patched(r"\int_E^\infty \frac{dE_K}{E_K}\thinspace", r"\int_E^\infty dE_K\thinspace"):
        w_b, b_b, _ = readme_decay_source(EK, Ns, FX, tau)
    return [exact("README Q_X = the count of the decays per unit energy (any N_s, F; integrands in E_K)", weight * body - counted),
            exact("the README's rate 1/(gamma tau_s) is the time-dilated one, gamma = E_K/(m_s c^2) (Evolution)",
                  rate - ms * cl**2 / (EK * tau)),
            Result("the integral from E_K = E: no product has x > 1 ((E* + p*)/M <= 1, kd.two_body)", True, "as written (asserted)"),
            exact_fails("control: dE_K for dE_K/E_K", w_b * b_b - counted)]


PDG_AL = {"e": sp.Rational("0.00334"), "mu": sp.Rational("0.00304")}   # K0_L listing: A_L(e) 0.334 +- 0.007 % (p. 35),
                                                                        # A_L(mu) 0.304 +- 0.025 % (p. 34), OUR AVERAGE


def readme_asym():
    """The README's shares of pi- l+ nu and pi+ l- nu-bar as functions of A_L, and its values of A_L for e and mu."""
    line = L.line(r"semileptonic decays give $\pi^-\ell^+\nu$ with probability $")
    AL = sp.Symbol("A_L")
    shares = re.search(r"give \$\\pi\^-\\ell\^\+\\nu\$ with probability \$(.+?)\$ and \$\\pi\^\+\\ell\^-\\bar\\nu\$ with \$(.+?)\$\.",
                       line).groups()
    values = re.search(r"\$A_L = ([\d.]+) \\times 10\^\{(-\d)\}\$ \(\$\\mathrm\{e\}\$\) and \$([\d.]+) \\times 10\^\{(-\d)\}\$ "
                       r"\(\$\\mu\$\)", line).groups()
    return ([L.to_sympy(t, {"A_{L}": AL}) for t in shares], AL,
            {"e": sp.Rational(values[0]) * sp.Integer(10)**int(values[1]), "mu": sp.Rational(values[2]) * sp.Integer(10)**int(values[3])})


@derivation("kd.asym")
def check_asym():
    """PDG's A_L = [Gamma(pi- l+ nu) - Gamma(pi+ l- nu-bar)]/SUM gives the shares (1 +- A_L)/2; the values are PDG's."""
    G1, G2 = sp.symbols("Gamma_1 Gamma_2", positive=True)   # Gamma(pi- l+ nu), Gamma(pi+ l- nu-bar)
    (plus, minus), AL, values = readme_asym()
    pdg = (G1 - G2) / (G1 + G2)                              # K0_L listing pp. 34-35
    with L.patched(r"$\left(1 + A_{L}\right)/2$", r"$\left(1 + A_{L}\right)/4$"):
        broken = readme_asym()[0][0]
    return [exact("Gamma(pi- l+ nu)/SUM = the README's (1 + A_L)/2", plus.subs(AL, pdg) - G1 / (G1 + G2)),
            exact("Gamma(pi+ l- nu-bar)/SUM = the README's (1 - A_L)/2", minus.subs(AL, pdg) - G2 / (G1 + G2)),
            Result("A_L = 3.34e-3 (e) and 3.04e-3 (mu): PDG's 0.334 % (p. 35) and 0.304 % (p. 34)", values == PDG_AL,
                   f"README {[float(v) for v in values.values()]}"),
            exact_fails("control: (1 + A_L)/4", broken.subs(AL, pdg) - G1 / (G1 + G2))]


@derivation("kd.flat")
def check_flat():
    """x = E/E_K is (E* + p* cos theta*)/M for gamma -> oo: uniform over the README's range with density M/(2p*)."""
    xc = sp.limit(boost((Es, 0, 0, ps * c))[0] / (GAM * M), q, sp.oo)   # cos(theta*) = c, uniform (isotropic)
    lo, hi = readme_range()
    with L.patched(r"with density $M/(2p^\ast)$", r"with density $M/p^\ast$"):
        broken = readme_density()
    return [Result("x is linear in cos theta* (so uniform)", sp.diff(xc, c, 2) == 0, f"x = {xc}"),
            exact("x at cos theta* = -1, +1: the README's ends (E* -+ p*)/M", (xc.subs(c, -1) - lo) ** 2 + (xc.subs(c, 1) - hi) ** 2),
            exact("(1/2)/(dx/dcos theta*) = the README's density M/(2p*)", sp.Rational(1, 2) / sp.diff(xc, c) - readme_density()),
            exact_fails("control: density M/p*", sp.Rational(1, 2) / sp.diff(xc, c) - broken)]


@derivation("kd.F")
def check_F():
    """F(x) adds the flat spectra of the nodes E* whose range holds x: those with E* >= E_lo(x) (>= m)."""
    E, v, w1 = sp.symbols("E v w_1", positive=True)
    Elo = readme_Elo()
    lo, hi = (e.subs({Es: E, ps: sp.sqrt(E**2 - m**2)}) for e in readme_range())   # the README's range of node E
    inside = sp.expand((x - lo) * (x - hi))                       # <= 0 iff x is in the range of the node E
    ends = sp.solve(inside, E)
    out = [Result("x is at an end of the range of one node E* only", len(ends) == 1, f"E* = {ends}"),
           exact("it is the README's E_lo(x)", ends[0] - Elo),
           exact("M^2 (x - x_lo)(x - x_hi) = -2Mx (E* - E_lo): x is in the range of the node iff E* >= E_lo",
                 M**2 * inside + 2 * M * x * (E - Elo)),
           exact("E_lo - m = (Mx - m)^2/(2Mx) >= 0 (the README's >= m)", Elo - m - (M * x - m)**2 / (2 * M * x)),
           exact("README integrand (M/2) g/p* = g times the flat density of kd.flat", readme_F() - g_ * readme_density())]
    # the README's F carries the number and the energy of g, for g = p* (a0 + a1 E* + a2 E*^2) on
    # [m, E_max]: with E = m (v^2 + v^-2)/2 (p* = m (v^2 - v^-2)/2), E_max at v = 1 + w_1
    a = sp.symbols("a0:3", real=True)
    poly = a[0] + a[1] * E + a[2] * E**2
    Ev, pv, vmax = m * (v**2 + v**-2) / 2, m * (v**2 - v**-2) / 2, 1 + w1
    Emax, pmax = Ev.subs(v, vmax), pv.subs(v, vmax)

    def moments(Elo_):
        F = laurent_integral(readme_F(g=ps * poly.subs(E, Es)).subs(Es, E), E, Elo_, Emax)
        xs = ((Emax - pmax) / M, (Emax + pmax) / M)
        return [laurent_integral(F * x**k, x, *xs) for k in (0, 1)]
    gv = pv * poly.subs(E, Ev) * sp.diff(Ev, v)
    truth = [laurent_integral(gv, v, 1, vmax), laurent_integral(gv * Ev / M, v, 1, vmax)]
    ours = moments(Elo)
    with L.patched(r"M^2x^2 + m^2", r"M^2x^2 - m^2"):
        broken = moments(readme_Elo())
    out += [exact("int F dx = int g dE* (g = p* (a0 + a1 E* + a2 E*^2))", ours[0] - truth[0]),
            exact("int x F dx = int (E*/M) g dE* (each flat range has the mean x = E*/M)", ours[1] - truth[1]),
            exact_fails("control: E_lo = (M^2 x^2 - m^2)/(2Mx) loses number", broken[0] - truth[0])]
    return out


@derivation("kd.two_body")
def check_two_body():
    """E* = (M^2 + m^2 - m_b^2)/(2M) by energy-momentum conservation; one E*, so F is flat; its x never exceeds 1."""
    mb = sp.Symbol("m_b", positive=True)
    readme = lambda: L.to_sympy(L.sides(span(r"Two-body decays have one $"))[1], {"M": M, "m": m, "m_b": mb})   # noqa: E731
    E0 = readme()
    solved = sp.solve((M - Es)**2 - mb**2 - (Es**2 - m**2), Es)   # the partner has M - E* and the same momentum
    p2 = E0**2 - m**2
    p0 = sp.Symbol("p_0", positive=True)
    flat = readme_F(g=1, E=E0, p=p0)                              # g = delta(E* - E0): one node
    with L.patched(r"- m_b^2)/(2M)$, so", r"+ m_b^2)/(2M)$, so"):
        broken = readme()
    return [Result("one solution E*", len(solved) == 1, f"{solved}"),
            exact("E* = (M^2 + m^2 - m_b^2)/(2M): M = E* + E_b, |p_b| = |p*|", E0 - solved[0]),
            exact("p*^2 = [M^2 - (m + m_b)^2][M^2 - (m - m_b)^2]/(4M^2)",
                  p2 - (M**2 - (m + mb)**2) * (M**2 - (m - mb)**2) / (4 * M**2)),
            Result("one node: F = M/(2p*) on its range (kd.flat), the same at every x", flat == M / (2 * p0) and not flat.has(x),
                   f"F = {flat}"),
            exact("(M^2 - m^2 + m_b^2)^2 - 4 M^2 p*^2 = 4 M^2 m_b^2 >= 0: (E* + p*)/M <= 1 (= 1 for m_b = 0), so x <= 1",
                  (M**2 - m**2 + mb**2)**2 - 4 * M**2 * p2 - 4 * M**2 * mb**2),
            exact_fails("control: + m_b^2", broken - solved[0])]


@derivation("kd.htilde")
def check_htilde():
    """A muon polarised along its kaon-frame momentum has the lab helicity P h~ (gamma -> oo); transverse polarisation averages out."""
    phi = sp.Symbol("phi", real=True)
    mm, sth = sp.sqrt(Es**2 - ps**2), sp.sqrt(1 - c**2)         # the muon mass, sin(theta*)
    lab = boost((Es, ps * sth, 0, ps * c))
    plab = sp.sqrt(lab[0]**2 - mm**2)
    helicity = lambda a: mm * boost(a)[0] / plab   # spin 4-vector a: a^0 = (|p|/m) (rest-frame spin . p^)
    sL = (ps / mm, Es / mm * sth, 0, Es / mm * c)                # longitudinal in the kaon frame
    e1, e2 = (0, c, 0, -sth), (0, 0, 1, 0)                       # normal to the momentum
    trans = [tuple(sp.cos(phi) * a + sp.sin(phi) * b for a, b in zip(e1, e2)),
             tuple(-sp.sin(phi) * a + sp.cos(phi) * b for a, b in zip(e1, e2))]
    hL = sp.limit(helicity(sL), q, sp.oo)
    cx = sp.solve(sp.Eq(sp.limit(lab[0] / (GAM * M), q, sp.oo), x), c)    # cos(theta*) of x (kd.flat)
    ht = readme_htilde().subs(m, mm)
    lo, hi = readme_range()
    with L.patched(r"(E^\ast M x - m^2)/(p^\ast M x)", r"(E^\ast M x + m^2)/(p^\ast M x)"):
        broken = readme_htilde().subs(m, mm)
    return [exact("s_L: s.p = 0, s^2 = -1", mdot(sL, (Es, ps * sth, 0, ps * c))**2 + (mdot(sL, sL) + 1)**2),
            Result("one cos theta* for each x", len(cx) == 1, f"cos theta* = {cx}"),
            exact("lab helicity of s_L for gamma -> oo = the README's h~", (hL - ht).subs(c, cx[0])),
            exact("polarisation normal to the momentum (in and out of the decay plane): 0 on average over its "
                  "orientation", sum(sp.integrate(helicity(a), (phi, 0, 2 * sp.pi))**2 for a in trans)),
            exact("h~ = -1 at x = (E* - p*)/M (backward), +1 at (E* + p*)/M",
                  (ht.subs(x, lo) + 1)**2 + (ht.subs(x, hi) - 1)**2),
            exact("dh~/dx = m^2/(p* M x^2) > 0 (|h~| <= 1 on the range)", sp.diff(ht, x) - mm**2 / (ps * M * x**2)),
            exact_fails("control: E* M x + m^2", (hL - broken).subs(c, cx[0]))]


@derivation("kd.FRL")
def check_FRL():
    """F_R,L: a muon is right-handed with probability (1 + P h~)/2; along the Dalitz line that gives (g +- G h~)/2."""
    y, y1, y2 = sp.symbols("y y_1 y_2", real=True)          # a coordinate along the Dalitz line E_mu = E*
    a, b = sp.symbols("a0:3", real=True), sp.symbols("b0:2", real=True)
    rho, P = a[0] + a[1] * y + a[2] * y**2, b[0] + b[1] * y   # test density and kaon-frame polarisation
    g, G = sp.integrate(rho, (y, y1, y2)), sp.integrate(rho * P, (y, y1, y2))
    ht = readme_htilde()
    out = []
    for sign, n in ((1, "R"), (-1, "L")):
        events = sp.integrate(rho * readme_density() * (1 + sign * P * ht) / 2, (y, y1, y2))
        out.append(exact(f"README F_{n} integrand = int along the line of rho (M/2p*) (1 {'+-'[sign < 0]} P h~)/2",
                         readme_F(sign, g, G, ht) - events))
    out.append(exact("F_R + F_L = F", readme_F(1, ht=ht) + readme_F(-1, ht=ht) - readme_F()))
    with L.patched(r"{2\thinspace p^\ast}", r"{p^\ast}"):
        out.append(exact_fails("control: p* for 2p* in F_R,L", readme_F(1, ht=ht) + readme_F(-1, ht=ht) - readme_F()))
    return out


@derivation("kd.lipari")
def check_lipari():
    """For K -> mu nu (P = -1) F_R,L are Lipari et al.'s eqs. 18-19 (m_pi -> m_K), and P h~ their eq. 16 (h = -1 at x = 1)."""
    r = sp.Symbol("r", positive=True)
    E0, p0 = readme_two_body()
    P = readme_P()
    ht0 = readme_htilde(E0, p0)
    rk = m**2 / M**2                                            # r_pi = (m_mu/m_pi)^2, m_pi -> m_K
    eq15, eq16, eq18, eq19 = (lipari(s, r).subs(r, rk) for s in (LIP_15, LIP_16, LIP_18, LIP_19))
    hx = re.search(r"helicity \$h = (-?\d+)\$ at \$x = (\d+)\$, eq\. 16", L.line(r"helicity $h = -1$ at $x = 1$, eq. 16"))
    FR, FL = (readme_F(s, g=1, G=P, ht=ht0, E=E0, p=p0) for s in (1, -1))   # g = delta(E* - E0), G = P g
    with L.patched(r"\frac{g(E^\ast) \pm G(E^\ast)", r"\frac{g(E^\ast) \mp G(E^\ast)"):
        broken = readme_F(1, g=1, G=P, ht=ht0, E=E0, p=p0)
    return [in_source("eqs. 15, 16, 18, 19, r_pi, h = P_R - P_L, its limits, the left-handed mu+ and m_pi -> m_K",
                      LIPARI, LIP_15, LIP_16, LIP_18, LIP_19, *LIP_TEXT),
            exact("p* = (M^2 - m^2)/(2M) at the README's E* (m_b = 0)", p0**2 - (E0**2 - m**2)),
            Result("E_lo(x) <= E*: x from r = m^2/M^2 to 1 (eq. 15: Theta(x - r))",
                   set(sp.solve(sp.Eq(readme_Elo(), E0), x)) == {rk, 1}, f"ends {sp.solve(sp.Eq(readme_Elo(), E0), x)}"),
            exact("the flat density M/(2p*) = eq. 15, 1/(1 - r)", readme_density().subs(ps, p0) - eq15),
            exact("lab helicity P h~ = eq. 16", P * ht0 - eq16),
            exact("h = -1 at x = 1 (README) = eq. 16 at x = 1", sp.Integer(hx.group(1)) - eq16.subs(x, sp.Integer(hx.group(2)))),
            exact("F_R = eq. 18", FR - eq18),
            exact("F_L = eq. 19", FL - eq19),
            exact_fails("control: g -+ G h~ for F_R", broken - eq18)]


def dirac():
    """The gamma matrices (Dirac representation), gamma_5, the unit matrix and the slash of a 4-vector."""
    I2, Z2 = sp.eye(2), sp.zeros(2)
    pauli = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
    gam = [sp.Matrix(sp.BlockMatrix([[I2, Z2], [Z2, -I2]]))] + [sp.Matrix(sp.BlockMatrix([[Z2, s_], [-s_, Z2]])) for s_ in pauli]
    return gam, sp.I * gam[0] * gam[1] * gam[2] * gam[3], sp.eye(4), lambda a: a[0] * gam[0] - a[1] * gam[1] - a[2] * gam[2] - a[3] * gam[3]


ML = sp.Symbol("m_l", positive=True)
EPI, EL, MPI, FP, FM = sp.symbols("E_pi E_l m_pi f_p f_m", real=True)
PK, PPI = sp.symbols("P_K P_pi")                 # k and q of CGG


def kaon_frame():
    """K -> pi l nu in the kaon frame: the scalar products of p_K, p_pi, p_l and p_nu in E_pi, E_l (p_K = p_pi + p_l + p_nu;
    vectors as their coefficients of these four), the README's H (its coefficients of p_K and p_pi) and t."""
    Enu = M - EPI - EL
    gram = {("K", "K"): M**2, ("K", "pi"): M * EPI, ("K", "l"): M * EL, ("K", "nu"): M * Enu,
            ("pi", "pi"): MPI**2, ("l", "l"): ML**2, ("nu", "nu"): 0,
            ("pi", "l"): (M**2 - 2 * M * Enu - MPI**2 - ML**2) / 2, ("pi", "nu"): (M**2 - 2 * M * EL + ML**2 - MPI**2) / 2,
            ("l", "nu"): (M**2 - 2 * M * EPI + MPI**2 - ML**2) / 2}
    gdot = lambda u, v: sum(cu * cv * gram.get((a, b), gram.get((b, a))) for a, cu in u.items() for b, cv in v.items())
    Hx = sp.expand(readme_H(FP, FM, PK, PPI))
    t_tex = L.sides(L.inline(r"t = (p_K - p_\pi)^2"))[1]
    assert t_tex.endswith("^2"), t_tex
    t_vec = sp.expand(L.to_sympy(t_tex[:-2], {"p_K": PK, r"p_\pi": PPI}))   # t is the square of this 4-vector
    return gdot, Hx, {"K": Hx.coeff(PK), "pi": Hx.coeff(PPI)}, {"K": t_vec.coeff(PK), "pi": t_vec.coeff(PPI)}


@derivation("kd.W")
def check_W():
    """W_0 +- W_pol(s) is the V-A trace for an l+ of spin s (s along the momentum: helicity +1/2); W_0 = 4 M_K rho of CGG's eq. 6.9."""
    # (1) |M|^2 = |u_nu-bar H-slash (1 - g5) v_l|^2 summed over the neutrino spin, with v v-bar =
    #     (p_l-slash - m)(1 + g5 s-slash)/2, explicit Dirac matrices, every component a symbol
    gam, g5, one, slash = dirac()
    H, n, l, s = (sp.symbols(f"{v}0:4", real=True) for v in ("H", "n", "l", "s"))
    ml = ML

    def trace(spin):
        rho_l = (slash(l) - ml * one) * (one + g5 * slash(spin)) / 2
        return sp.expand((slash(n) * slash(H) * (one - g5) * rho_l * slash(H) * (one - g5)).trace())
    T, Tm = trace(s), trace(tuple(-a for a in s))
    W0, Wp = readme_W({r"p_\nu": n, r"p_\ell": l, "H": H, "s": s}, mdot, ml)
    with L.patched(r"m_\ell\left[2\thinspace", r"m_\ell\left[4\thinspace"):
        Wp_broken = readme_W({r"p_\nu": n, r"p_\ell": l, "H": H, "s": s}, mdot, ml)[1]
    # the l- of the CP-conjugate decays: |u_l-bar H-slash (1 - g5) v_nu-bar|^2, u u-bar = (p_l-slash + m)(1 + g5 s-slash)/2
    L.line(r"$\mathrm{K}^-$ decays are the CP conjugates of $\mathrm{K}^+$ decays (a left-handed $\mu^+$ becomes a right-handed $\mu^-$)")
    rho_minus = (slash(l) + ml * one) * (one + g5 * slash(s)) / 2
    T_minus = sp.expand((rho_minus * slash(H) * (one - g5) * slash(n) * (one + g5) * slash(H)).trace())
    out = [exact("trace = 4 [W_0 + W_pol(s)] (any H, p_nu, p_l, s)", T - 4 * (W0 + Wp)),
           exact("summed over s, -s: 4 x 2 W_0", T + Tm - 8 * W0),
           exact("an l- (CP conjugate): 4 [W_0 - W_pol(s)], its helicities those of the l+ exchanged", T_minus - 4 * (W0 - Wp)),
           exact_fails("control: 4 for 2 in W_pol", T - 4 * (W0 + Wp_broken))]
    # (2) K -> mu nu (H ∝ p_K), kaon frame, mu+ along z: the spin-0 kaon and the left-handed nu make
    #     the mu+ left-handed, so s along the momentum (helicity +1/2) must give no rate
    E0, p0 = readme_two_body()
    pK, pmu, pnu, s_up = (M, 0, 0, 0), (E0, 0, 0, p0), (p0, 0, 0, -p0), (p0 / m, 0, 0, E0 / m)
    W0_2, Wp_2 = readme_W({r"p_\nu": pnu, r"p_\ell": pmu, "H": pK, "s": s_up}, mdot, m)
    out += [exact("K -> mu nu: p_K = p_mu + p_nu, s.p_mu = 0, s^2 = -1",
                  sum((a - b - d)**2 for a, b, d in zip(pK, pmu, pnu)) + mdot(s_up, pmu)**2 + (mdot(s_up, s_up) + 1)**2),
            exact("K -> mu nu: W_0 + W_pol(s) = 0 for helicity +1/2 (the mu+ is left-handed)", W0_2 + Wp_2),
            exact("K -> mu nu: P = W_pol(s)/W_0 = the README's P = -1", Wp_2 / W0_2 - readme_P())]
    # (3) CGG: kaon-frame energies, the scalar products from p_K = p_pi + p_l + p_nu (kaon_frame; H and t: kd.current)
    gdot, _, Hv, _ = kaon_frame()
    rho = cgg_rho(EPI, EL, M, MPI, ML, FP, FM)
    factor = L.to_sympy(L.sides(L.inline(r"W_0 = 4M_K\thinspace\rho", "checks"))[1], {"M_K": M, r"\rho": rho})
    vec = {r"p_\nu": {"nu": 1}, r"p_\ell": {"l": 1}, "H": Hv}
    W0_cgg = readme_W(vec, gdot, ML)[0]
    with L.patched(r"- H^2\thinspace (p_\nu\negthinspace \cdot\negthinspace p_\ell)",
                   r"+ H^2\thinspace (p_\nu\negthinspace \cdot\negthinspace p_\ell)"):
        broken = readme_W(vec, gdot, ML)[0]
    out += [exact("W_0 = 4 M_K rho over the whole Dalitz plot (CGG eq. 6.9, pp. 249-250)", W0_cgg - factor),
            exact_fails("control: + H^2 (p_nu . p_l) in W_0", broken - factor)]
    return out


@derivation("kd.current")
def check_current():
    """H is CGG's eq. 6.8, t = (p_K - p_pi)^2 = (p_l + p_nu)^2; the f_- term gives m_l times the scalar lepton current."""
    gdot, Hx, _, tv = kaon_frame()
    # (1) f_- (p_K - p_pi) = f_- (p_l + p_nu) against the V-A lepton current u_nu-bar gamma^mu (1 - g5) v_l: by the Dirac
    #     equation it is -m_l u_nu-bar (1 + g5) v_l. Summed over the spins, |u_nu-bar X v_l|^2 with X = q-slash (1 - g5)
    #     + m_l (1 + g5) vanishes on shell, so each amplitude does
    gam, g5, one, slash = dirac()
    n, l = (sp.symbols(f"{v}1:4", real=True) for v in ("n", "l"))
    pn = (sp.sqrt(sum(a**2 for a in n)),) + tuple(n)                    # the neutrino, massless
    pl = (sp.sqrt(sum(a**2 for a in l) + ML**2),) + tuple(l)            # the charged lepton, mass m_l
    spins = lambda X_: sp.expand((slash(pn) * X_ * (slash(pl) - ML * one) * gam[0] * X_.H * gam[0]).trace())   # noqa: E731
    q_ = slash([a + b for a, b in zip(pn, pl)])
    total, wrong = spins(q_ * (one - g5) + ML * (one + g5)), spins(q_ * (one - g5) - ML * (one + g5))
    with L.patched(r"f_-(t)(p_K - p_\pi)$, where", r"f_-(t)(p_K + p_\pi)$, where"):
        broken = sp.expand(readme_H(FP, FM, PK, PPI))
    return [exact("H = (k + q) f+ + (k - q) f- (CGG eq. 6.8, k = p_K, q = p_pi)", Hx - ((PK + PPI) * FP + (PK - PPI) * FM)),
            exact("t = (p_K - p_pi)^2 = CGG's (k - q)^2 = M^2 + mu^2 - 2 M E_pi (eq. 6.7)",
                  gdot(tv, tv) - (M**2 + MPI**2 - 2 * M * EPI)),
            exact("t = (p_l + p_nu)^2 (p_K = p_pi + p_l + p_nu)", gdot(tv, tv) - gdot({"l": 1, "nu": 1}, {"l": 1, "nu": 1})),
            exact("sum over spins |u_nu-bar [(p_l + p_nu)-slash (1 - g5) + m_l (1 + g5)] v_l|^2 = 0 on shell: the f_- term is "
                  "-m_l f_- u_nu-bar (1 + g5) v_l, a scalar current that vanishes with m_l (K_e3)", total),
            exact_fails("control: + m_l u_nu-bar (1 + g5) v_l does not cancel it", wrong),
            exact_fails("control: f_-(t)(p_K + p_pi)", broken - ((PK + PPI) * FP + (PK - PPI) * FM))]


@derivation("kd.cgg")
def check_cgg():
    """The citation index quotes Chounet, Gaillard & Gaillard's eq. 6.9 as printed; its E'_pi and t are the kinematics of eq. 6.7."""
    if not L.FILES["citations"].exists():
        return [Result("eq. 6.9 as quoted", True, "not re-checked: knowledge/citations.md is not there")]
    A, B, C, fp, fm, Epi, El, mu, ml = sp.symbols("A B C f_p f_m E_pi E_mu mu m", real=True)
    line = L.line(r"their eq. 6.9 ($", "citations")
    quote = lambda: L.to_sympy(span(r"their eq. 6.9 ($", "citations").split(r"\propto")[1],
                               {"A": A, "B": B, "C": C, "f_+": fp, "f_-": fm})
    eq69 = A * fp**2 + B * fp * fm + C * fm**2
    with L.patched(r"B f_+f_- + C f_-^2$", r"B f_+f_- + C f_+^2$", "citations"):
        broken = quote()
    Emax = (M**2 + mu**2 - ml**2) / (2 * M)                    # E_pi^max of p. 250
    t = M**2 + mu**2 - 2 * M * Epi                               # eq. 6.7
    return [Result("the quotation names eq. 6.9 and pp. 249-250", "pp. 249–250" in line, "as written"),
            exact("quotation = eq. 6.9 (scan p. 249): A f+^2 + B f+ f- + C f-^2", quote() - eq69),
            exact_fails("control: C f+^2 in the quotation", broken - eq69),
            exact("E_pi^max = (M^2 + mu^2 - m^2)/(2M) is where t = m^2 (eq. 6.7: m^2 < t < (M - mu)^2)",
                  t.subs(Epi, Emax) - ml**2),
            exact("E_pi = mu (pion at rest) gives t = (M - mu)^2", t.subs(Epi, mu) - (M - mu)**2),
            exact("m = 0: B = C = 0, only f+ (as CGG note for K_e3, p. 250)",
                  cgg_rho(Epi, El, M, mu, 0, fp, fm) - cgg_rho(Epi, El, M, mu, 0, fp, 0))]


@derivation("kd.fminus")
def check_fminus():
    """f_- = (f_0 - f_+)(M^2 - m_pi^2)/t is PDG's f_0 = f_+ + f_- t/(m_K^2 - m_pi^2) solved for f_-."""
    f0, fp, fm, t, mpi = sp.symbols("f_0 f_p f_m t m_pi", real=True)
    tokens = {"f_0": f0, "f_+": fp, "M": M, r"m_\pi": mpi, "t": t}
    pdg = sp.solve(sp.Eq(f0, fp + fm * t / (M**2 - mpi**2)), fm)      # K± listing p. 37, K0_L p. 23
    ours = lambda: L.to_sympy(span(r"fits, with $f_- = "), tokens)
    with L.patched(r"(M^2 - m_\pi^2)/t", r"(M^2 + m_\pi^2)/t"):
        broken = ours()
    return [exact("README f_- = PDG's f_0 solved for f_- (m_K, m_pi of the mode)", ours() - pdg[0]),
            exact_fails("control: M^2 + m_pi^2", broken - pdg[0])]


@derivation("kd.formfactors")
def check_formfactors():
    """f_+ and f_0 are PDG's expansions (quadratic; linear for lambda''_+ = 0) normalised to f_+(0) = f_0(0) = 1."""
    t, mpi, l1, l2, l0, f0 = sp.symbols("t m_pi lambda1 lambda2 lambda0 f0", real=True)
    line = L.display(r"f_+(t) = 1 + \lambda'_ +\frac{t}{m_{\pi^+}^2}")
    tokens = {r"\lambda'_ +": l1, r"\lambda''_ +": l2, r"\lambda_0": l0, "t": t, r"m_{\pi^+}": mpi}
    fplus = lambda: L.to_sympy(L.sides(L.part(L.display(r"f_+(t) = 1 + \lambda'_ +\frac{t}{m_{\pi^+}^2}"), "f_+(t)"))[1], tokens)
    fzero = L.to_sympy(L.sides(L.part(line, "f_0(t)"))[1], tokens)
    pdg_quad = f0 * (1 + l1 * t / mpi**2 + l2 / 2 * t**2 / mpi**4)   # K± p. 37, K0_L p. 23 (f0 = f+(0))
    pdg_lin = f0 * (1 + l1 * t / mpi**2)                              # lambda+ = lambda'+
    pdg_f0 = f0 * (1 + l0 * t / mpi**2)                               # f0(0) = f+(0) by the definition of f0
    with L.patched(r"\frac{\lambda''_ +}{2}", r"\lambda''_ +"):
        broken = fplus()
    return [exact("README f_+ = PDG's quadratic f_+(t)/f_+(0)", fplus() - pdg_quad / f0),
            exact("with lambda''_+ = 0: PDG's linear f_+(t)/f_+(0), lambda_+ = lambda'_+ (the K± fit)",
                  fplus().subs(l2, 0) - pdg_lin / f0),
            exact("README f_0 = PDG's linear f_0(t)/f_0(0), f_0(0) = f_+(0) (f_0 = f_+ + f_- t/(m_K^2 - m_pi^2))",
                  fzero - pdg_f0 / f0),
            exact_fails("control: lambda''_+ for lambda''_+/2", broken - pdg_quad / f0)]


@derivation("kd.dalitz")
def check_dalitz():
    """The K -> 3pi expansion is PDG's eq. 1 (K± listing, p. 32) without its CP-violating terms j v and f u v."""
    s = sp.symbols("s_1:4", real=True)
    mpi, mK, g, h, j, k, f = sp.symbols("m_pi m_K g h j k f", real=True)
    E, mi = sp.symbols("E_1:4", real=True), sp.symbols("m_1:4", positive=True)
    u, v = readme_uv(s, mpi)
    ours = L.to_sympy(L.parts(DALITZ())[0].split(r"\propto")[1], {"g": g, "h": h, "k": k, "u": u, "v": v})
    s0 = sum(s) / 3
    # s_i = (p_K - p_i)^2 in the kaon frame = m_K^2 + m_i^2 - 2 m_K E_i, with sum E_i = m_K
    s_kin = [mK**2 + mi[i]**2 - 2 * mK * E[i] for i in range(3)]
    on_shell = {E[2]: mK - E[0] - E[1]}
    with L.patched(r"u = \frac{s_3 - s_0}", r"u = \frac{s_1 - s_0}"):
        broken = readme_uv(s, mpi)[0]
    return [exact("README |M|^2 = eq. 1 with j = f = 0", ours - pdg_dalitz(s, mpi, g, h, 0, k, 0)),
            exact("u = (s_3 - s_0)/m_pi+^2 and v = (s_2 - s_1)/m_pi+^2 (eq. 1; the sign of v matters for j, f)",
                  (u - (s[2] - s0) / mpi**2)**2 + (v - (s[1] - s[0]) / mpi**2)**2),
            exact("(p_K - p_i)^2 = PDG's (m_K - m_i)^2 - 2 m_K T_i (T_i = E_i - m_i)",
                  sum((s_kin[i] - ((mK - mi[i])**2 - 2 * mK * (E[i] - mi[i])))**2 for i in range(3))),
            exact("s_0 = (1/3) sum s_i = PDG's (m_K^2 + m_1^2 + m_2^2 + m_3^2)/3",
                  (sum(s_kin) / 3).subs(on_shell) - (mK**2 + sum(a**2 for a in mi)) / 3),
            Result("pion 3 is the odd one (PDG: index 3 for the odd pion)",
                   "where pion 3 is the odd one" in L.line("where pion 3 is the odd one"), "as written"),
            exact_fails("control: s_1 for s_3 in u", broken - (s[2] - s0) / mpi**2)]


@derivation("kd.KL3pi0")
def check_KL3pi0():
    """1 + h(u^2 + v^2/3) is PDG's expansion (K0_L listing p. 20) symmetric under the permutations of three pi0."""
    s = sp.symbols("s_1:4", real=True)
    mpi, g, h, j, k, f, U, V = sp.symbols("m_pi g h j k f u v", real=True)
    u, v = readme_uv(s, mpi)

    def general(si):                                   # 1 + gu + hu^2 + jv + kv^2 + fuv in the README's u, v
        uu, vv = readme_uv(si, mpi)
        return 1 + g * uu + h * uu**2 + j * vv + k * vv**2 + f * uu * vv
    conditions = []
    for perm in permutations(s):
        conditions += sp.Poly(sp.expand(general(perm) - general(s)), *s).coeffs()
    sym = sp.solve(conditions, [g, j, k, f], dict=True)
    readme = lambda: L.to_sympy(span(r"\to 3\pi^0$ has $"), {"h": h, "u": U, "v": V}).subs({U: u, V: v})
    value = L.to_sympy(L.sides(span(r"made symmetric in the three $\pi^0$, with $"))[1], {})
    with L.patched(r"1 + h(u^2 + v^2/3)", r"1 + h(u^2 + v^2/2)"):
        broken = readme()
    return [Result("symmetric under the 6 permutations: g = j = f = 0, k = h/3",
                   sym == [{g: 0, j: 0, f: 0, k: h / 3}], f"solution {sym}"),
            exact("README form = the symmetric expansion (h = the coefficient of u^2, as PDG's h)",
                  readme() - general(s).subs(sym[0])),
            Result("h = 0.00059 = PDG's (0.59 +- 0.20 +- 1.16) x 10^-3 (K0_L listing p. 22)",
                   abs(float(value) - 0.59e-3) < 1e-15, f"README h = {value}"),
            exact_fails("control: v^2/2", broken - general(s).subs(sym[0]))]


@derivation("kd.bin_N")
def check_bin_N():
    """N is the integral over [x_a, x_b] of one node's spectrum (M/2p*)(w_g + w_G h~), a node of F or F_R,L."""
    N = readme_bin("N")
    spec = node_spectrum()
    weights = {n: L.to_sympy(L.sides(L.inline(tex, "checks"))[1], {"g": g_, "G": G_, "w": w_}, sign=1)
               for n, tex in (("g", r"w_g = g\thinspace w"), ("g/2", r"w_g = g\thinspace w/2"), ("G", r"w_G = \pm G\thinspace w/2"))}
    wG_minus = L.to_sympy(L.sides(L.inline(r"w_G = \pm G\thinspace w/2", "checks"))[1], {"G": G_, "w": w_}, sign=-1)
    ht = readme_htilde()
    with L.patched(r"\ln\frac{x_b}{x_a}", r"\ln\frac{x_a}{x_b}", "checks"):
        broken = readme_bin("N")
    return [exact("dN/dx_b = the node's spectrum at x_b", sp.diff(N, x2) - spec.subs(x, x2)),
            exact("N = 0 at x_b = x_a", N.subs(x2, x1)),
            exact("w_g = g w, w_G = 0: w times the README's F integrand", spec.subs({wg: weights["g"], wG: 0}) - w_ * readme_F()),
            exact("w_g = g w/2, w_G = +-G w/2: w times the README's F_R, F_L integrands",
                  (spec.subs({wg: weights["g/2"], wG: weights["G"]}) - w_ * readme_F(1, ht=ht))**2
                  + (spec.subs({wg: weights["g/2"], wG: wG_minus}) - w_ * readme_F(-1, ht=ht))**2),
            exact_fails("control: ln(x_a/x_b)", sp.diff(broken, x2) - spec.subs(x, x2))]


@derivation("kd.bin_E")
def check_bin_E():
    """E is the integral over [x_a, x_b] of x times one node's spectrum (M/2p*)(w_g + w_G h~)."""
    E = readme_bin("E")
    spec = node_spectrum()
    with L.patched(r"\frac{m^2}{p^\ast M}(x_b - x_a)", r"\frac{m^2}{p^\ast M}\ln\frac{x_b}{x_a}", "checks"):
        broken = readme_bin("E")
    return [exact("dE/dx_b = x_b times the node's spectrum at x_b", sp.diff(E, x2) - x2 * spec.subs(x, x2)),
            exact("E = 0 at x_b = x_a", E.subs(x2, x1)),
            exact_fails("control: ln(x_b/x_a) for x_b - x_a", sp.diff(broken, x2) - x2 * spec.subs(x, x2))]


XK, XK1, NN, EN = sp.symbols("x_k x_k1 N E", positive=True)


def readme_grid_shares():
    """The README's shares of x_k and x_{k+1} of the products between them (number N, energy E)."""
    tokens = {"E": EN, "N": NN, "x_k": XK, "x_{k+1}": XK1}
    return (L.to_sympy(span(r"$x_k$ receives $", "checks"), tokens),
            L.to_sympy(span(r"$x_{k+1}$ receives $", "checks"), tokens))


@functools.cache
def numeric_grid():
    """The README's grid points x_k = e^{-k Delta}, as a numpy function of (k, Delta)."""
    kk, D = sp.symbols("k Delta", real=True)
    return sp.lambdify((kk, D), L.to_sympy(L.sides(L.inline(r"x_k = e^{-k\Delta}", "checks"))[1], {"k": kk, r"\Delta": D}),
                       "numpy")


@derivation("kd.share")
def check_share():
    """Sharing the products linearly in x between x_{k+1} and x_k gives the README's shares, which keep N and E."""
    xk, xk1, EE = XK, XK1, EN
    a, b = readme_grid_shares()
    spec = node_spectrum()
    bins = {NN: readme_bin("N"), EE: readme_bin("E")}
    linear = sp.integrate((x - xk1) / (xk - xk1) * spec, (x, x1, x2))     # a product at x gives x_k this part
    with L.patched(r"\left(E - x_{k+1}N\right)", r"\left(E - x_kN\right)", "checks"):
        broken = readme_grid_shares()[0]
    return [exact("number: the two shares add up to N", a + b - NN),
            exact("energy: x_k (share of x_k) + x_{k+1} (share of x_{k+1}) = E", xk * a + xk1 * b - EE),
            exact("share of x_k = int of (x - x_{k+1})/(x_k - x_{k+1}) over the node's spectrum, with the README's N, E",
                  a.subs(bins) - linear),
            exact_fails("control: E - x_k N", broken.subs(bins) - linear)]


# --- the kaons' masses and mean lives: PDG 2026, the README and the code ---

# PDG 2026 listings (knowledge/papers/), OUR FIT: the masses (pdg.py), the K+- mean life (1.2380 +- 0.0020)e-8 s (p. 7),
# the K0_S mean life (0.8954 +- 0.0004)e-10 s, assuming CPT (p. 1), and the K0_L mean life (5.116 +- 0.021)e-8 s (p. 2).
# Masses in MeV, mean lives in s, by the names of the code's constants.
PDG_KAONS = {"massK": sp.Rational(MASS_MEV["K+"]), "massK0": sp.Rational(MASS_MEV["K0"]), "tdec_K": sp.Rational("1.2380e-8"),
             "tdec_KS": sp.Rational("0.8954e-10"), "tdec_KL": sp.Rational("5.116e-8")}
KAON_INPUTS = "with the PDG 2026 masses and mean lives"


def readme_kaon_inputs():
    """The README's kaon masses [MeV] and mean lives [s], by the names of the code's constants."""
    num = r"([\d.]+)(?: \\times 10\^\{(-\d+)\})?"
    found = re.search(r"\$m_\{\\mathrm\{K\}\^\\pm\} = " + num + r"\$ MeV, \$\\tau = " + num + r"\$ s; \$m_\{\\mathrm\{K\}\^0\} = "
                      + num + r"\$ MeV, \$\\tau_\\mathrm\{S\} = " + num + r"\$ s, \$\\tau_\\mathrm\{L\} = " + num + r"\$ s",
                      L.line(KAON_INPUTS)).groups()
    values = [sp.Rational(found[i]) * sp.Integer(10)**int(found[i + 1] or 0) for i in range(0, 10, 2)]
    return dict(zip(("massK", "tdec_K", "massK0", "tdec_KS", "tdec_KL"), values))


@derivation("kd.inputs")
def check_inputs():
    """The README's kaon masses and mean lives are PDG 2026's fit values, and the code's (constants.mojo)."""
    def results(src):
        readme = readme_kaon_inputs()
        code = {n: v * 1000 if n.startswith("mass") else v for n, v in mojo_constants.read(PDG_KAONS, src).items()}   # MeV
        return [Result("the README's m_K+-, tau_K+-, m_K0, tau_S, tau_L = PDG 2026's (K+-, K0, K0_S, K0_L listings, OUR FIT)",
                       readme == PDG_KAONS, "exact: " + ", ".join(f"{n} = {sp.N(v, 6)}" for n, v in readme.items())),
                Result("the code's massK, massK0 (GeV), tdec_K, tdec_KS, tdec_KL = the README's", code == readme,
                       "exact: equal" if code == readme else f"differ: {[n for n in readme if code[n] != readme[n]]}")]
    src = mojo_constants.PATH.read_text()
    broken = src.replace("massK0: Float64 = 0.497611", "massK0: Float64 = 0.497612")
    with L.patched(r"$\tau_\mathrm{L} = 5.116", r"$\tau_\mathrm{L} = 5.117"):
        broken_readme = results(src)[0]
    return results(src) + [
        Result("control: massK0 one digit off in the source", broken != src and not results(broken)[1].ok, "fails, as it must"),
        Result("control: tau_L = 5.117e-8 s in the README", not broken_readme.ok, "fails, as it must")]


# --- the README's branching ratios, for kaon_production/multibody_strangeness.py ---

def branching_ratios():
    """{kaon: {mode: branching ratio}} of the README's table (its LaTeX labels: "K+", "K0S", "K0L"; mode as written)."""
    head = L.line("| Kaon | Mode | Branching ratio |")
    block = L.text()[L.text().index(head):].split("\n\n")[0].splitlines()[2:]
    out, kaon = {}, None
    names = {r"$\mathrm{K}^+$": "K+", r"$\mathrm{K}^0_\mathrm{S}$": "K0S", r"$\mathrm{K}^0_\mathrm{L}$": "K0L"}
    for row in block:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        kaon = names[cells[0]] if cells[0] else kaon
        value = cells[2].strip("$").replace(r" \times 10^{", "e").replace("}", "")
        out.setdefault(kaon, {})[cells[1]] = float(value)
    return out


if __name__ == "__main__":
    run()
