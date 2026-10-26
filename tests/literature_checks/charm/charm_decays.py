"""Derivation checks of the Decays section of charm/README.md (the decays of the charm hadrons) against their sources:
  - the PDG 2026 listings, transcribed with their pages in pdg_charm.py: the decay modes of D+, D0, Ds+, Lambda_c+
    and the tau, the inclusive e+ rates, the form-factor ratios r_V and r_2, the I(J^P) of the pole states, the
    tau's decay parameters and the decays of the short-lived products;
  - Ablikim et al. (BESIII 2015), TeX source of arXiv:1508.07560v1: the modified pole (eq. II.4) and its alpha in
    Tables XIII and XV (D -> K 0.309, D -> pi 0.279) and in the text (eqs. VI.25-26: 0.310, 0.279);
  - Richman & Burchat (1995), the full preprint UCSB-HEP-95-08 in knowledge/papers/ (PostScript; the arXiv entry has
    only the abstract and the contents), transcribed below (RB_*): the currents (eqs. 81-86), the pole form (eq. 88),
    the quantum numbers of the poles (pp. 73-74) and the P -> V distribution (eqs. 113, 115-116);
  - Meinel (2017), TeX source of arXiv:1611.09696v2 (Lambda_c -> Lambda): the z expansion (eq. 3) with its t_+ and
    pole masses (text), the nominal fit (Table IV) and its supplemental file;
  - Meinel (2018), TeX source of arXiv:1712.05783v4 (Lambda_c -> N): the helicity form factors (eqs. 1-2, 5-7), the
    z expansion (eqs. 10-13), the pole states (Table IV), the nominal fit (Table V) and its supplemental file, the
    differential rate (eq. 17) and B(Lambda_c -> n l+ nu) (eqs. 20-21).
Derived here: with explicit Dirac matrices in sympy (as kd.W), the V-A trace of a complex current, the spin rule, the
left-handed l+ of D(s)+ -> l+ nu, the tau's angular distributions and Michel spectra, the P -> V density of Richman &
Burchat's current and the Lambda_c -> B density of Meinel's current; the source on the grid, the tables' shares of
the bins, the boosts, the cascade of a short-lived product and the recursion of n-body phase space.

key() gives the final-state keys of tests/support/code_decays.py (tests/unit/test_added_processes.py,
kaons/multibody_strangeness.py).

Run from the repository root:
  .venv/bin/python tests/literature_checks/common/run.py charm_decays
"""
import functools
import re

import mpmath as mp
import sympy as sp
from sympy.physics.quantum.cg import CG

import latex
import pdg_charm as PDG
from verify import PDG_C, RB, Entry, Result, derivation, dirac, exact, exact_fails, in_source, mdot, rng, slash

PAGE = "charm"   # the page holding this module's equations (a key of latex.FILES)
L = latex.page(PAGE)

BESIII = "arXiv-1508.07560v1.tar.gz"       # Ablikim et al. 2015
MEINEL17 = "arXiv-1611.09696v2.tar.gz"     # Meinel 2017, Lambda_c -> Lambda
MEINEL18 = "arXiv-1712.05783v4.tar.gz"     # Meinel 2018, Lambda_c -> N
PDG_TAU = "PDF PDG-2026_listing_tau.pdf"
PDG_SHORT = ("PDF PDG-2026_listing_eta.pdf, _eta-prime-958.pdf, _omega-782.pdf, _phi-1020.pdf, _rho-770.pdf, "
             "_K-star-892.pdf, _f0-980.pdf, _a1-1260.pdf, _lambda.pdf, _sigma-plus.pdf, _sigma-zero.pdf, "
             "_sigma-minus.pdf, _xi-zero.pdf, _xi-minus.pdf")
PDG_POLES = ("PDF PDG-2026_listing_D-star-2010-plus-minus.pdf, _D0-star-2300.pdf, _D1-2420.pdf, "
             "_Ds-star-plus-minus.pdf, _Ds0-star-2317-plus-minus.pdf, _Ds1-2460-plus-minus.pdf")

# this module's equations in the inventory (common/inventory.py), an Entry each
INVENTORY = [
    Entry("cd.source", r"E\thinspace Q_X(E) = \sum_s \int_E^\infty", "derived",
          "the kaon decays' source, per unit ln E: the decays counted", file=PAGE),
    Entry("cd.grid", r"E_i\thinspace Q_X(E_i) = \Delta\sum_s\sum_j", "derived",
          "the source on the logarithmic grids (sum over ln E_C); number and energy kept", file=PAGE),
    Entry("cd.semileptonic", r"every exclusive mode above 0.1% that no other mode contains", "transcribed",
          "PDG 2026 semileptonic modes (pdg_charm.py); Meinel 2018, eqs. 20-21", PDG_C + ", TeX " + MEINEL18,
          checks="tests/unit/test_added_processes.py (e+ per decay)", file=PAGE),
    Entry("cd.leptonic", r"$\mathrm D^+$ and $\mathrm D_s^+$ $\to \mu^+\nu$, $\tau^+\nu$", "transcribed",
          "PDG 2026, D+- and Ds+- listings", "PDF PDG-2026_listing_D-plus-minus.pdf, _Ds-plus-minus.pdf",
          checks="tests/unit/test_added_processes.py (nu_tau per decay)", file=PAGE),
    Entry("cd.hadronic", r"every mode above 0.5%, renormalised to the rest of the width", "transcribed",
          "PDG 2026 hadronic modes (K0S x 2, overlap subtractions; pdg_charm.py)", PDG_C, file=PAGE),
    Entry("cd.k0s", r"$\Gamma(\mathrm K^0) = 2\Gamma(\mathrm K^0_\mathrm S)$", "transcribed",
          "PDG 2026, D+- listing p. 2 (decay table header)", "PDF PDG-2026_listing_D-plus-minus.pdf", file=PAGE),
    Entry("cd.coverage", r"The hadronic modes cover 60–78% of the rates before the renormalisation",
          "derived", "sums of the PDG 2026 branching ratios of the hadronic modes", PDG_C, file=PAGE),
    Entry("cd.summary", r"| Scale factor of the semileptonic modes |", "derived",
          "the rules applied to PDG 2026's modes and inclusive e+ rates (pdg_charm.py) and Meinel 2018, eqs. 20-21",
          PDG_C + ", TeX " + MEINEL18, file=PAGE),
    Entry("cd.counts", r"| Hadronic modes | 21 | 29 | 30 | 32 |", "derived", "the modes the rules give (PDG 2026)",
          PDG_C, file=PAGE),
    Entry("cd.sl_table", r"| Parent | Mode | $\mathcal B_\mathrm e$ | $\mathcal B_\mu$ | PDG 2026 | Matrix element |",
          "transcribed", "PDG 2026 semileptonic modes and form factors (pdg_charm.py); Ablikim et al. 2015, Tables "
          "XIII, XV; Meinel 2018, eqs. 20-21", PDG_C + ", TeX " + BESIII + ", TeX " + MEINEL18, file=PAGE),
    Entry("cd.trace", r"\vert\mathcal M\vert^2_{\rm sum} = 8\left[2(p_\nu\cdot J_r)(p_\ell\cdot J_r)", "derived",
          "V-A trace with explicit Dirac matrices (any complex current)", file=PAGE),
    Entry("cd.pol",
          r"\vert\mathcal M\vert^2(s) = \tfrac12\thinspace\vert\mathcal M\vert^2_{\rm sum}"
          r"(p_\ell \to p_\ell + m_\ell s)",
          "derived", "V-A trace with explicit Dirac matrices (the spin projector of the l+)", file=PAGE),
    Entry("cd.ppcurrent", r"$H = f_+(t)(p + p') + f_-(t)(p - p')$", "transcribed",
          "Richman & Burchat 1995, eqs. 81-82 (p. 73)", RB, file=PAGE),
    Entry("cd.fminus", r"$f_- = (f_0 - f_+)(M^2 - m^2)/t$", "rearranged", "Richman & Burchat 1995, eq. 83", RB,
          file=PAGE),
    Entry("cd.formfactors", r"f_+(t) = \frac{1}{(1 - t/m_{\rm p}^2)(1 - \alpha t/m_{\rm p}^2)}", "transcribed",
          "Ablikim et al. 2015 (BESIII), eq. II.4, Tables XIII and XV, eqs. VI.25-26; Richman & Burchat 1995, "
          "eqs. 81-83, 88", "TeX " + BESIII + ", " + RB, file=PAGE),
    Entry("cd.rb85", r"\langle\mathrm V\vert V^\mu - A^\mu\vert\mathrm P\rangle = ", "transcribed",
          "Richman & Burchat 1995, eqs. 85-86 (pp. 73-74)", RB, file=PAGE),
    Entry("cd.pv", r"V(t) = \frac{r_V}{1 - t/m_V^2}", "transcribed",
          "Richman & Burchat 1995, eq. 88 (n = 1) and p. 87 (r_V, r_2); PDG 2026 semileptonic form factors", RB,
          file=PAGE),
    Entry("cd.a0", r"$A_0(0) = [(M + m) - (M - m)r_2]/(2m)$", "derived",
          "Richman & Burchat 1995, eq. 86 with A_0(0) = A_3(0) (p. 74)", RB, file=PAGE),
    Entry("cd.pv_result", r"\sum_\varepsilon\vert\mathcal M\vert^2_{\rm sum} = 2(q^2 - m_\ell^2)", "derived",
          "V-A trace of Richman & Burchat's eq. 85; their eqs. 113, 115-116 (lepton mass terms derived)", RB,
          file=PAGE),
    Entry("cd.poles", r"Richman & Burchat assign $J^P = 1^-$ to $f_+$ and $V$", "model",
          "Richman & Burchat 1995, pp. 73-75; Meinel 2018, Table IV; PDG 2026 I(J^P) of the states",
          RB + ", TeX " + MEINEL18 + ", " + PDG_POLES, file=PAGE),
    Entry("cd.bcurrent", r"J^\mu = \bar u_\mathrm B\left[F\gamma^\mu + aP^\mu + bq^\mu", "rearranged",
          "Meinel 2018, eqs. 1-2 (Detmold, Lehner & Meinel 2015, eqs. 3-5)", "TeX " + MEINEL18, file=PAGE),
    Entry("cd.zexp", r"f(q^2) = \frac{a_0^f + a_1^fz + a_2^fz^2}{1 - q^2/(m_{\rm pole}^f)^2}", "transcribed",
          "Meinel 2017, eq. 3, text and Table IV; Meinel 2018, eqs. 10-13, Tables IV-V; supplemental files",
          "TeX " + MEINEL17 + ", TeX " + MEINEL18, file=PAGE),
    Entry("cd.zw", r"c = \frac{(M - m_\mathrm B)\thinspace w\left[", "derived",
          "the z expansion with g_+ and g_perp sharing a_0 and the pole (Meinel 2018, eq. 7)", "TeX " + MEINEL18,
          file=PAGE),
    Entry("cd.bilinear", r"\sum\vert\mathcal M\vert^2 = \tfrac12 G_F^2\vert V_{cq}\vert^2\thinspace W", "derived",
          "V-A trace of Meinel's current with explicit Dirac matrices", file=PAGE),
    Entry("cd.onshell", r"$P^2 = 2(M^2 + m_\mathrm B^2) - q^2$", "standard", "kinematics (baryons on shell)",
          file=PAGE),
    Entry("cd.free_quark", r"$W = 256\thinspace(p\cdot p_\ell)(p'\cdot p_\nu)$", "derived",
          "the bilinear form with all form factors 1 (the free-quark V-A current)", file=PAGE),
    Entry("cd.lc_rate", r"Integrated over the lepton energy at fixed $q^2$, $W$ gives Meinel's differential rate",
          "derived", "Meinel 2018, eq. 17, from the README's W (numerically, 30 digits)", "TeX " + MEINEL18,
          file=PAGE),
    Entry("cd.two_body", r"$E^\ast = (M^2 + m^2 - m_b^2)/(2M)$", "standard", "two-body kinematics", file=PAGE),
    Entry("cd.phase_space", r"g(E) \propto p\thinspace R_{n-1}(m_{\rm rec})", "standard",
          "n-body phase space (its recursion over two-body phase spaces)", file=PAGE),
    Entry("cd.kallen", r"$\lambda(a, b, c) = a^2 + b^2 + c^2 - 2ab - 2bc - 2ca$", "standard",
          "two-body kinematics (Kallen function)", file=PAGE),
    Entry("cd.tau_pol", r"The $\tau^+$ is left-handed in the parent frame.", "derived",
          "V-A trace with explicit Dirac matrices (angular momentum, as kd.W)", file=PAGE),
    Entry("cd.tau_decays", r"It decays by its eleven modes above 0.5%, renormalised.", "transcribed",
          "PDG 2026, tau listing (basis modes)", PDG_TAU, file=PAGE),
    Entry("cd.tau_table", r"| Mode | Branching ratio | PDG 2026 | Angular distribution |", "transcribed",
          "PDG 2026, tau listing pp. 6-11 (basis modes, charge conjugated)", PDG_TAU, file=PAGE),
    Entry("cd.tau_ang", r"\frac{dN}{d\cos\theta} = \frac{1 + a\thinspace\mathcal P\cos\theta}{2}", "derived",
          "V-A traces of tau+ -> h+ anti-nu (spin 0 and 1); PDG 2026 chirality parameters (tau listing pp. 88-89)",
          PDG_TAU, file=PAGE),
    Entry("cd.michel", r"\frac{dN_\ell}{dx} = 2x^2(3 - 2x)", "derived",
          "V-A trace of tau+ -> l+ nu anti-nu; Michel parameters (PDG 2026, tau listing pp. 83-87)", PDG_TAU,
          file=PAGE),
    Entry("cd.tau_frame", r"The $\tau^+$ decays in its parent's frame instead, with its polarisation.", "derived",
          "the polarised tau's decays are not isotropic (V-A trace)", file=PAGE),
    Entry("cd.tau_boost", r"E = \gamma_\tau\left(E' + \beta_\tau p'\cos\theta\right)", "standard",
          "Lorentz boost; two-body kinematics of D(s)+ -> tau+ nu", file=PAGE),
    Entry("cd.flat", r"is spread uniformly over $x$ between $(E^\ast - p^\ast)/M$ and $(E^\ast + p^\ast)/M$",
          "derived", "kinematics (isotropic decay, ultra-relativistic parent)", file=PAGE),
    Entry("cd.bins", r"\Delta\thinspace T(x_k) = \frac{\mathcal E_k - x_{k+1}\mathcal N_k}{x_k - x_{k+1}}", "model",
          "the tables' shares of the bins (number and energy kept)", file=PAGE),
    Entry("cd.short_lived", r"gives each its PDG 2026 modes of 0.5% and above, renormalised", "transcribed",
          "PDG 2026 listings of the short-lived products (pdg_charm.SHORT_LIVED); isospin", PDG_SHORT, file=PAGE),
    Entry("cd.cascade", r"F_{s\to X}(x) = F^{\rm direct}_ {s\to X}(x) + \sum_Y \int_x^1", "derived",
          "kinematics: light-cone fractions multiply (isotropic decays)", file=PAGE),
    Entry("cd.conv", r"T_{s\to X}(x_k) = T^{\rm direct}_ {s\to X}(x_k) + \Delta\sum_Y", "derived",
          "the cascade on the grid (number and energy kept)", file=PAGE),
    Entry("cd.aligned", r"but their decays into spinless products are even in the polar angle", "derived",
          "angular distribution of a polarised vector meson's decay into spinless products", file=PAGE),
]
# entry id -> the pins of the other relations of its display equations (each a whole relation as latex.parts cuts it)
ALSO = {
    "cd.source": (r"n_s = E_CN_s", r"T_{s\to X}(x) = x\thinspace F_{s\to X}(x)"),
    "cd.formfactors": (r"f_0(t) = \frac{1}{1 - t/m_0^2}",),
    "cd.rb85": (r"A_3 = \frac{(M + m)A_1 - (M - m)A_2}{2m}",),
    "cd.pv": (r"A_1(t) = \frac{1}{1 - t/m_A^2}", r"A_2(t) = \frac{r_2}{1 - t/m_A^2}",
              r"A_0(t) = \frac{A_0(0)}{1 - t/m_0^2}"),
    "cd.pv_result": (r"H_t = \frac{2Mp_\mathrm VA_0}{\sqrt{q^2}}",),
    "cd.bcurrent": (r"P = p + p'", r"q = p - p',\qquad s_\pm", r"s_\pm = (M \pm m_\mathrm B)^2 - q^2", r"F = f_\perp",
                    r"G = g_\perp", r"a = \frac{(M + m_\mathrm B)(f_+ - f_\perp)}{s_+}",
                    r"b = (M - m_\mathrm B)\left[\frac{f_0 - f_+}{q^2} - \frac{a}{M + m_\mathrm B}\right]",
                    r"c = \frac{(M - m_\mathrm B)(g_+ - g_\perp)}{s_-}",
                    r"d = (M + m_\mathrm B)\left[\frac{g_0 - g_+}{q^2} - \frac{c}{M - m_\mathrm B}\right]"),
    "cd.zexp": (r"z = \frac{\sqrt{t_+ - q^2} - \sqrt{t_+ - t_0}}{\sqrt{t_+ - q^2} + \sqrt{t_+ - t_0}}",
                r"t_0 = (M - m_\mathrm B)^2"),
    "cd.zw": (r"w = \frac{z}{s_-} = \frac{1}{\left(\sqrt{t_+ - q^2} + \sqrt{t_+ - t_0}\right)^2}",),
    "cd.bilinear": (r"W = K_g\thinspace p_\ell\cdot p_\nu + K_{PP}", r"K_{PP} = 32\left[F^2 + G^2",
                    r"K_{qq} = -32\left[", r"K_S = -64\left[", r"K_A = -128\thinspace FG",
                    r"K_g = -16\left[4Mm_\mathrm B"),
    "cd.phase_space": (r"m_{\rm rec}^2 = M^2 + m^2 - 2ME", r"R_2(m) = \frac{\lambda^{1/2}(m^2, m_1^2, m_2^2)}{m^2}",
                       r"R_k(m) = \int dm'\thinspace 2m'\thinspace R_2(m; m_1, m')\thinspace R_{k-1}(m')"),
    "cd.tau_ang": (r"a_\pi = a_\mathrm K = -1", r"a_\rho = -\frac{m_\tau^2 - 2m_\rho^2}{m_\tau^2 + 2m_\rho^2}"),
    "cd.michel": (r"a_\ell = \frac{2x - 1}{3 - 2x}", r"\frac{dN_\nu}{dx} = 12x^2(1 - x)", r"a_\nu = -1"),
    "cd.tau_boost": (r"\gamma_\tau = \frac{M^2 + m_\tau^2}{2Mm_\tau}",),
    "cd.conv": (r"x_jx_{k-j} = x_k",),
}

SPECIES = ("D+", "D0", "Ds+", "Lambda_c+")
SPECIES_TEX = {"D+": r"$\mathrm D^+$", "D0": r"$\mathrm D^0$", "Ds+": r"$\mathrm D_s^+$", "Lambda_c+": r"$\Lambda_c^+$"}


def key(names):
    """A final state as a sorted tuple; K0 and anti-K0 alike (the code's X_K0)."""
    return tuple(sorted("K0" if n == "K0bar" else n for n in names))


def rat(x):
    """A float of pdg_charm, exact as printed."""
    return sp.Rational(repr(x))


def control_fails(what, failed):
    """A Result: the check run on a broken copy fails, as it must."""
    return Result(f"control: {what}", failed, "fails, as it must")


def printed(cell, value):
    """Whether the number a table cell starts with is value rounded to the cell's digits."""
    s = re.match(r"\s*(-?[\d.]+)", cell).group(1)
    decimals = len(s.split(".")[1]) if "." in s else 0
    return abs(sp.Rational(s) - value) <= sp.Rational(1, 2) * sp.Integer(10) ** -decimals


def gamma_number(label):
    """n of a PDG label 'Gamma<n> ...'."""
    return int(re.match(r"Gamma(\d+)", label).group(1))


# --- Richman & Burchat (1995), UCSB-HEP-95-08, transcribed from the PostScript in knowledge/papers/ ---
# p. 73: <P'(p')|V^mu|P(p)> = F_1(q^2)[(p + p')^mu - (M^2 - m_P'^2)/q^2 q^mu] + F_0(q^2) (M^2 - m_P'^2)/q^2 q^mu (81),
#   "F_0(0) = F_1(0), so there is no singular behavior at q^2 = 0"; "F_0(q^2) and F_1(q^2) can be associated with the
#   exchange of particles with quantum numbers J^P = 0+ and J^P = 1-, respectively";
#   = f_+(q^2)(p + p')^mu + f_-(q^2)(p - p')^mu (82), f_+(q^2) = F_1(q^2), F_0(q^2) = f_+(q^2) + q^2/(M^2 - m_P'^2)
#   f_-(q^2) (83).
# p. 73: <V(p', eps)|V^mu - A^mu|P(p)> = 2i eps^{mu nu alpha beta}/(M + m_V) eps*_nu p'_alpha p_beta V(q^2)
#   - (M + m_V) eps*^mu A_1(q^2) + (eps*.q)/(M + m_V) (p + p')^mu A_2(q^2) + 2 m_V (eps*.q)/q^2 q^mu A_3(q^2)
#   - 2 m_V (eps*.q)/q^2 q^mu A_0(q^2) (85), with no convention for eps^{0123} stated;
# p. 74: A_3(q^2) = (M + m_V)/(2 m_V) A_1(q^2) - (M - m_V)/(2 m_V) A_2(q^2) (86) "with A_0(0) = A_3(0)"; "The form
#   factors A_1(q^2) and A_2(q^2) can be associated with the exchange of a particle with quantum numbers J^P = 1+,
#   whereas V(q^2) is associated with J^P = 1-".
# p. 75: f_i(q^2) = f_i(0)/(1 - q^2/m_pole^2)^n (88), "where n is an integer, usually one for mesons. The pole mass
#   m_pole is the mass of the lowest-lying Q qbar' meson with the quantum numbers appropriate to a given part of the
#   hadronic current."
# p. 85: eq. 113, for V -> P1 P2: d Gamma ∝ p_V q^2 {(1 - eta cos theta_l)^2 sin^2 theta_V |H_+|^2 + (1 + eta
#   cos theta_l)^2 sin^2 theta_V |H_-|^2 + 4 sin^2 theta_l cos^2 theta_V |H_0|^2 + (terms in cos chi)}, "the factor
#   eta is equal to +1 for B decays and -1 for D decays", theta_l the angle of the charged lepton to the direction of
#   the virtual W in the parent frame, measured in the W rest frame, the lepton massless; over theta_V and chi the
#   weights are sin^2 -> 4/3, 4 cos^2 -> 8/3, the chi terms vanish.
# p. 86: H_0 = [(M^2 - m_V^2 - q^2)(M + m_V) A_1 - 4 M^2 p_V^2/(M + m_V) A_2]/(2 m_V sqrt(q^2)) (115),
#   H_pm = (M + m_V) A_1 -+ 2 M p_V/(M + m_V) V (116), p_V the V momentum in the P frame.
# p. 87: "r_V ≡ V/A_1 and r_2 ≡ A_2/A_1".
RB_EQ88_N = 1
RB_JP = {"F_1": "1-", "F_0": "0+", "A_1": "1+", "A_2": "1+", "V": "1-"}   # pp. 73-74; F_1 = f_+ (eq. 83)


def RB_81(F1, F0, p, pp, M, m, q2):
    """Eq. 81, per component (p, pp: a component of p, p'; q = p - p')."""
    q = p - pp
    return F1 * ((p + pp) - (M**2 - m**2) / q2 * q) + F0 * (M**2 - m**2) / q2 * q


def RB_85(X, E, S, Q, eq, q2, M, mV, V, A1, A2, A3, A0):
    """Eq. 85: X = eps^{mu nu alpha beta} eps*_nu p'_alpha p_beta, E = eps*^mu, S = (p + p')^mu, Q = q^mu,
    eq = eps*.q."""
    return (2 * sp.I * X / (M + mV) * V - (M + mV) * E * A1 + eq / (M + mV) * S * A2 + 2 * mV * eq / q2 * Q * A3
            - 2 * mV * eq / q2 * Q * A0)


def RB_86(M, mV, A1, A2):
    return (M + mV) / (2 * mV) * A1 - (M - mV) / (2 * mV) * A2


def RB_113(c, Hp, Hm, H0):
    """Eq. 113 for D decays (eta = -1), integrated over theta_V and chi, over the common factor 4/3."""
    return (1 + c)**2 * Hp**2 + (1 - c)**2 * Hm**2 + 2 * (1 - c**2) * H0**2


def RB_helicity(M, mV, q2, pV, V, A1, A2):
    """Eqs. 115-116: H_+, H_-, H_0."""
    return ((M + mV) * A1 - 2 * M * pV / (M + mV) * V, (M + mV) * A1 + 2 * M * pV / (M + mV) * V,
            ((M**2 - mV**2 - q2) * (M + mV) * A1 - 4 * M**2 * pV**2 / (M + mV) * A2) / (2 * mV * sp.sqrt(q2)))


# --- source of the decay products ---

SOURCE_PIN = r"E\thinspace Q_X(E) = \sum_s \int_E^\infty"


def readme_source(E, EC, gC, tau, N, F):
    """The README's source per unit ln E of one species, as (weight of the measure dE_C, integrand) for any density
    N_s and distribution F (sympy functions), with n_s and T from the display's other relations."""
    disp = L.display(SOURCE_PIN)
    lhs, rhs = L.sides(L.parts(disp)[0])
    assert lhs == r"E\thinspace Q_X(E)", lhs
    head, rest = rhs.split(r"\int_E^\infty ", 1)
    assert head.strip() == r"\sum_s", head
    measure, body = rest.split(r"\thinspace", 1)
    dEC, x, Nsym = sp.Symbol("dE_C", positive=True), sp.Symbol("x", positive=True), sp.Symbol("N")
    weight = sp.diff(L.to_sympy(measure, {"dE_C": dEC, "E_C": EC}), dEC)
    n_l, n_r = L.sides(L.part(disp, "n_s ="))
    assert n_l == "n_s", n_l
    n_of = L.to_sympy(n_r, {"E_C": EC, "N_s": Nsym})
    t_l, t_r = L.sides(L.part(disp, r"T_{s\to X}(x) ="))
    assert t_l == r"T_{s\to X}(x)", t_l
    T_of = L.to_sympy(t_r, {"x": x}, {r"F_{s\to X}": F})
    n_fun = lambda a: n_of.subs({EC: a, Nsym: N(a)}, simultaneous=True)
    T_fun = lambda a: T_of.subs(x, a)
    integrand = L.to_sympy(body, {"E": E, "E_C": EC, r"\gamma_C": gC, r"\tau_s": tau},
                           {"n_s": n_fun, r"T_{s\to X}": T_fun})
    return weight, integrand, n_fun, T_fun


@derivation("cd.source")
def check_source():
    """The source per unit ln E counts the decays: N_s dE_C hadrons decay at 1/(gamma tau), making F dx products."""
    E, EC, gC, tau = sp.symbols("E E_C gamma_C tau_s", positive=True)
    N, F = sp.Function("N_s"), sp.Function("F")
    weight, integrand, _, _ = readme_source(E, EC, gC, tau, N, F)
    # per unit energy: dN/dE = int dE_C N(E_C)/(gamma tau) F(E/E_C) dx/dE, dx/dE = 1/E_C; per unit ln E: times E
    counted = E * N(EC) / (gC * tau) * F(E / EC) / EC
    rate = L.to_sympy(L.inline(r"1/(\gamma_C\tau_s)"), {r"\gamma_C": gC, r"\tau_s": tau})
    with L.patched(r"\int_E^\infty \frac{dE_C}{E_C}\thinspace", r"\int_E^\infty dE_C\thinspace"):
        w_b, i_b, _, _ = readme_source(E, EC, gC, tau, N, F)
    # x <= 1: the largest light-cone fraction (E* + p*)/M of a product, for a massless partner, is 1
    M, m, mb = sp.symbols("M m m_b", positive=True)
    Es = (M**2 + m**2 - mb**2) / (2 * M)
    xmax = (Es + sp.sqrt(Es**2 - m**2)) / M
    gen = rng(11)
    below = all(xmax.subs({M: 2, m: sp.Rational(gen.randint(1, 9), 10), mb: sp.Rational(gen.randint(1, 9), 10)}) < 1
                for _ in range(10))
    return [exact("E Q_X(E) = E x the decays counted per unit energy (any N_s, F; integrands in E_C)",
                  weight * integrand - counted),
            exact("the rate 1/(gamma_C tau_s) is the time-dilated decay rate", rate - 1 / (gC * tau)),
            exact("the integral from E_C = E: a product's x = E/E_C reaches 1 only against a massless partner",
                  sp.simplify(xmax.subs(mb, 0).subs(M, m + sp.Symbol("d", positive=True))) - 1),
            Result("and stays below 1 otherwise (10 random two-body decays)", below, "all"),
            exact_fails("control: dE_C for dE_C/E_C", w_b * i_b - counted)]


GRID_PIN = r"E_i\thinspace Q_X(E_i) = \Delta\sum_s\sum_j"


def readme_grid(T, n, Ei, Ej, gj, tau):
    """The summand of the README's source on the grid, for the functions T and n."""
    lhs, rhs = L.sides(L.display(GRID_PIN))
    assert lhs == r"E_i\thinspace Q_X(E_i)", lhs
    head = r"\Delta\sum_s\sum_j "
    assert rhs.startswith(head), rhs
    return L.to_sympy(rhs[len(head):], {"E_i": Ei, "E_j": Ej, r"\gamma_j": gj, r"\tau_s": tau},
                      {r"T_{s\to X}": T, "n_s": n})


def grid_conservation(summand_of):
    """(number, energy) differences on a toy grid: parent bins j = 0..2 at E_j = r^-j, tables t_0..t_4 at x_k = r^k,
    product bins wherever E_i/E_j is a table point; zero when the sum keeps the products' number and energy."""
    r, D, tau = sp.symbols("r Delta tau", positive=True)
    J, K = 3, 4
    t = sp.symbols(f"t0:{K + 1}")
    w = sp.symbols(f"w0:{J}")
    E = lambda j: r ** (-j)
    g = lambda j: sp.Symbol(f"g{j}", positive=True)

    def T(arg):
        k = sp.simplify(sp.log(arg) / sp.log(r))
        return t[int(k)] if k.is_integer and 0 <= k <= K else 0

    def n(arg):
        j = -sp.simplify(sp.log(arg) / sp.log(r))
        return w[int(j)] * g(int(j)) * tau
    Q = {}
    for i in range(-K, J):
        Q[i] = sum(D * summand_of(T, n, E(i), E(j), g(j), tau) for j in range(J))
    number = sum(D * Q[i] for i in Q) - (D * sum(w)) * (D * sum(t))
    energy = sum(D * E(i) * Q[i] for i in Q) - (D * sum(w[j] * E(j) for j in range(J))) * (
        D * sum(r**k * t[k] for k in range(K + 1)))
    return sp.expand(number), sp.expand(energy)


@derivation("cd.grid")
def check_grid():
    """On the logarithmic grids the source is the sum over ln E_C, and it keeps the products' number and energy."""
    E, EC, gC, tau = sp.symbols("E E_C gamma_C tau_s", positive=True)
    N, F = sp.Function("N_s"), sp.Function("F")
    # the README's source per unit ln E_C (cd.source, any N, F) at E = E_i, E_C = E_j, against the grid's summand
    weight, integrand, n_fun, T_fun = readme_source(E, EC, gC, tau, N, F)
    per_ln = sp.expand(weight * integrand * EC)
    summand = readme_grid(T_fun, n_fun, E, EC, gC, tau)
    out = [exact("the grid's summand = the integrand per unit ln E_C (E_i = E, E_j = E_C; d ln E_C -> Delta)",
                 sp.expand(summand) - per_ln)]
    nb, en = grid_conservation(lambda T_, n_, Ei, Ej, gj, tau_: readme_grid(T_, n_, Ei, Ej, gj, tau_))
    out += [exact("number: Delta sum_i E_i Q_X(E_i) = (Delta sum_j n_j/(gamma_j tau)) x (Delta sum_k T(x_k))", nb),
            exact("energy: Delta sum_i E_i [E_i Q_X(E_i)] = (Delta sum_j E_j n_j/(gamma_j tau)) x "
                  "(Delta sum_k x_k T(x_k))", en)]
    with L.patched(r"\left(\frac{E_i}{E_j}\right)\frac{n_s(E_j)}", r"\left(\frac{E_j}{E_i}\right)\frac{n_s(E_j)}"):
        nb_b, en_b = grid_conservation(lambda T_, n_, Ei, Ej, gj, tau_: readme_grid(T_, n_, Ei, Ej, gj, tau_))
    out.append(control_fails("README with T(E_j/E_i)", en_b != 0 or nb_b != 0))
    return out


# --- decay modes ---

def readme_threshold(prefix):
    """The threshold of a rule ("... above x%"), as a fraction."""
    return sp.Rational(L.between(L.line(prefix), prefix, "%")) / 100


def contains(total, part):
    """Whether the final state `total` is `part` (a mode with a short-lived particle) with that particle decayed."""
    for i, x in enumerate(part):
        for md in PDG.RESONANCE_DECAYS.get(x, []):
            if key(list(part[:i]) + list(md["fs"]) + list(part[i + 1:])) == key(total):
                return True
    return False


# Meinel 2018, eqs. 20-21 (in percent)
MEINEL_BR = (r"\mathcal{B}(\Lambda_c\to n\, e^+\nu_e)  &=& \left( 0.410 \pm 0.026_{\,\rm LQCD}",
             r"\mathcal{B}(\Lambda_c\to n\, \mu^+\nu_\mu)  &=& \left( 0.400 \pm 0.026_{\,\rm LQCD}")
SL_RULE = r"every exclusive mode above "


def meinel_n_rates():
    """B(Lambda_c -> n e+ nu), B(-> n mu+ nu) of Meinel 2018, eqs. 20-21, as fractions (from the snippets)."""
    return [sp.Rational(re.search(r"\\left\( ([\d.]+) \\pm", s).group(1)) / 100 for s in MEINEL_BR]


@functools.cache
def sl_modes(s, thr=sp.Rational(1, 1000)):
    """The semileptonic modes the README's rules select for species s: [dict(lepton, hadrons, br, pdg, md)], with
    Meinel's Lambda_c -> n l+ nu and the Ds+ mu+ modes that PDG lacks taken equal to the e+ modes (pdg None)."""
    out = []
    for md in PDG.DECAYS[s]["semileptonic"]:
        if md["in_sum"] and (rat(md["br"]) > thr or md["swave"]):
            out.append(dict(lepton=md["lepton"], hadrons=tuple(f for f in md["fs"] if f not in ("e+", "nu_e", "mu+",
                                                                                                "nu_mu")),
                            br=rat(md["br"]), pdg=md["pdg"], md=md))
    if s == "Lambda_c+":
        for lepton, br in zip(("e", "mu"), meinel_n_rates()):
            out.append(dict(lepton=lepton, hadrons=("n",), br=br, pdg=None, md=None))
    if s == "Ds+":
        mu = {m["hadrons"] for m in out if m["lepton"] == "mu"}
        out += [dict(m, lepton="mu", pdg=None, md=None) for m in out if m["lepton"] == "e" and m["hadrons"] not in mu]
    return out


def semileptonic_threshold(thr):
    """Per species: PDG's exclusive modes that no other contains, but the complements, are not below thr; those at it,
    which "above" leaves out, are named."""
    out = []
    for s in SPECIES:
        modes = [md for md in PDG.DECAYS[s]["semileptonic"] if md["in_sum"] and not md["swave"]]
        at = [md["pdg"] for md in modes if rat(md["br"]) == thr]
        out.append(Result(f"{s}: PDG's exclusive modes no other contains, but the complements, are not below the "
                          f"README's {float(thr * 100):g} %", all(rat(md["br"]) >= thr for md in modes),
                          f"{len(modes)} modes, smallest {min(md['br'] for md in modes)}"
                          + (f"; at it, so left out: {', '.join(at)}" if at else "")))
    return out


@derivation("cd.semileptonic")
def check_semileptonic():
    """Semileptonic: no exclusive mode below 0.1 %, totals left out contain listed modes, Lambda_c -> n is Meinel's."""
    out = semileptonic_threshold(readme_threshold(SL_RULE))
    with L.patched(r"every exclusive mode above 0.1%", r"every exclusive mode above 0.12%"):
        broken = semileptonic_threshold(readme_threshold(SL_RULE))
    out.append(control_fails("README threshold 0.12 %", not all(r.ok for r in broken)))
    # "that no other mode contains": every total left out contains a listed (resonant) mode
    for s in SPECIES:
        listed = [md for md in PDG.DECAYS[s]["semileptonic"] if md["in_sum"]]
        for md in PDG.DECAYS[s]["semileptonic"]:
            if not md["in_sum"]:
                inside = [o["pdg"] for o in listed if o["lepton"] == md["lepton"] and contains(md["fs"], o["fs"])]
                out.append(Result(f"{s}: {md['pdg']} (left out) contains {', '.join(inside) or 'no listed mode'}",
                                  bool(inside), "PDG 2026 final states"))
    out += [Result("Lambda_c+ -> n l+ nu: not among PDG 2026's modes (README: \"which PDG does not list\")",
                   not any("n" in md["fs"] for md in PDG.DECAYS["Lambda_c+"]["semileptonic"])
                   and "which PDG does not list, has the rates of Meinel (2018), eqs. 20–21" in
                   L.line("which PDG does not list"), "as written"),
            in_source("Meinel 2018, eqs. 20-21: B(Lambda_c -> n e+ nu) = 0.410 %, B(-> n mu+ nu) = 0.400 %",
                      MEINEL18, *MEINEL_BR)]
    # the D(s)+ mu+ modes PDG lacks: only Ds+ has e+ modes without a mu+ partner
    lacking = {s: sorted(m["hadrons"] for m in sl_modes(s) if m["lepton"] == "mu" and m["pdg"] is None
                         and s != "Lambda_c+") for s in SPECIES}
    out.append(Result("the e+ modes without a mu+ mode in PDG 2026 are those of the Ds+ (README: \"The mu+ modes of "
                      "the Ds+ that PDG lacks\")", all(not v for k, v in lacking.items() if k != "Ds+")
                      and len(lacking["Ds+"]) == 3, f"Ds+: {lacking['Ds+']}"))
    return out


LEPTONIC_SPECIES = {r"\mathrm D^+": "D+", r"\mathrm D_s^+": "Ds+", r"\mathrm D^0": "D0", r"\Lambda_c^+": "Lambda_c+"}
LEPTONIC_FS = {r"\mu^+\nu": key(["mu+", "nu_mu"]), r"\tau^+\nu": key(["tau+", "nu_tau"]),
               r"\mathrm e^+\nu": key(["e+", "nu_e"])}


def readme_leptonic():
    """{species: set of final states} of the README's leptonic rule ("**Leptonic:** $D^+$ and $D_s^+$ $\\to ..."))."""
    ln = L.line(r"**Leptonic:** $")
    head = re.match(r"- \*\*Leptonic:\*\* (.*?)\.(?: |$)", ln).group(1)
    species_tex, modes_tex = head.split(r" $\to ", 1)
    species = [LEPTONIC_SPECIES[s] for s in re.findall(r"\$([^$]+)\$", species_tex)]
    modes = {LEPTONIC_FS[m.strip()] for m in re.split(r"\$, \$", modes_tex.strip("$"))}
    return {s: modes for s in species}


def leptonic_results(readme):
    out = []
    for s in SPECIES:
        pdg, want = {key(md["fs"]) for md in PDG.DECAYS[s]["leptonic"]}, readme.get(s, set())
        limits = ", ".join(lim["pdg"] for lim in PDG.DECAYS[s]["leptonic_limits"]) or "none listed"
        detail = f"{sorted(pdg) or 'none'}" if pdg == want else f"PDG {sorted(pdg)}, README {sorted(want)}"
        out.append(Result(f"{s}: PDG 2026's leptonic modes = the README's (e+ nu: {limits})", pdg == want, detail))
    return out


@derivation("cd.leptonic")
def check_leptonic():
    """The leptonic modes are PDG's D+, Ds+ -> mu+ nu, tau+ nu (e+ nu only limits); none for D0 and Lambda_c+."""
    out = leptonic_results(readme_leptonic())
    with L.patched(r"$\to \mu^+\nu$, $\tau^+\nu$.", r"$\to \mathrm e^+\nu$, $\mu^+\nu$, $\tau^+\nu$."):
        broken = leptonic_results(readme_leptonic())
    out.append(control_fails("README with e+ nu as a mode", not all(r.ok for r in broken)))
    return out


def pdg_value(value):
    """The central value of a PDG value as printed: '(1.554 +- 0.033) % S=1.8', '(7.1 +- 0.5) x 10^-3', ..."""
    m = re.match(r"\(([\d.]+) (?:\+- [\d.]+|\+[\d.]+ -[\d.]+)\) (%|x 10\^(-?\d+))", value)
    v = sp.Rational(m.group(1))
    return v / 100 if m.group(2) == "%" else v * sp.Integer(10) ** int(m.group(3))


def gammas(md):
    """{n: printed value} of the PDG entries a hadronic mode's calc uses (its own and its inputs)."""
    out = {gamma_number(md["pdg"]): pdg_value(md["pdg_value"])}
    for inp in md.get("inputs") or []:
        n, value = re.match(r"Gamma(\d+) .*? = (\(.*\)[^(]*)(?: \(p\d+\))?$", inp).groups()
        out[int(n)] = pdg_value(value.strip())
    return out


def evaluate(calc, values):
    """A calc formula with G<n> and B_* replaced by values."""
    names = {f"G{n}": v for n, v in values.items()}
    names.update({b: rat(v[0]) for b, v in PDG.SUBTRACTION_BRS.items()})
    expr = re.sub(r"\b(G\d+|B_\w+)\b", lambda m_: f"({names[m_.group(1)]})", calc)
    return sp.sympify(expr, rational=True)


RESONANCE_OF = {"B_ETA": "eta", "B_ETAP": "etap", "B_OMEGA": "omega", "B_PHI": "phi"}
README_RESONANCE = {r"\eta": "eta", r"\eta'": "etap", r"\omega": "omega", r"\phi": "phi", r"\mathrm{K}^\ast": "Kstar"}


def subtracted_resonances(md):
    """The short-lived particles through which the listed modes subtracted from a total make its final state."""
    out = {RESONANCE_OF[b] for b in re.findall(r"\bB_[A-Z]+", md["calc"])}
    for inp in md.get("inputs") or []:
        n = gamma_number(inp)
        if n != gamma_number(md["pdg"]) and re.search(r"-\s*G" + str(n) + r"\b", md["calc"]):
            label = inp.split("=")[0]
            out |= {r for r, pat in (("Kstar", r"K\*\(892\)"), ("phi", r"\bphi\b"), ("omega", r"\bomega\b"),
                                     ("eta", r"\beta\b"), ("etap", r"eta'")) if re.search(pat, label)}
    return out


def readme_subtracted():
    """The short-lived particles of the README's subtraction rule ("... through an $\\eta$, ... or $\\phi$ is
    subtracted")."""
    ln = L.line("the part that another listed mode makes through an")
    span = re.search(r"the part that another listed mode makes through an (.*?) is subtracted", ln).group(1)
    return {README_RESONANCE[t] for t in re.findall(r"\$([^$]+)\$", span)}


@derivation("cd.hadronic")
def check_hadronic():
    """Hadronic: PDG's modes above 0.5 %, a K0S twice (two K0S four times), less the parts of other listed modes."""
    out = hadronic_results(readme_threshold(r"**Hadronic:** every mode above "))
    with L.patched(r"**Hadronic:** every mode above 0.5%",
                   r"**Hadronic:** every mode above 0.6%"):
        broken = hadronic_results(readme_threshold(r"**Hadronic:** every mode above "))
    out.append(control_fails("README threshold 0.6 %", not all(r.ok for r in broken)))
    return out


def hadronic_results(thr):
    """The hadronic modes of pdg_charm against the README's rules with the threshold thr."""
    allowed = readme_subtracted()
    out = []
    for s in SPECIES:
        modes = PDG.DECAYS[s]["hadronic"]
        bad_calc, bad_k0, low, foreign = [], [], [], []
        for md in modes:
            vals = gammas(md)
            br, calc = rat(md["br"]), evaluate(md["calc"], vals)
            ulp = sp.Rational(1, 2) * sp.Integer(10) ** (sp.floor(sp.log(br, 10)) - 4)
            if abs(calc - br) > ulp:
                bad_calc.append(md["pdg"])
            label = md["pdg"].split(" ", 1)[1]
            n = label.count("K0S") + label.count("2K0S")
            factor = sp.Integer(2) ** n
            if md["kind"] in ("k0s", "remainder") and n and not (md["calc"].startswith(f"{factor}*")
                                                                and sum(f in ("K0", "K0bar") for f in md["fs"]) >= n):
                bad_k0.append(md["pdg"])
            main = gamma_number(md["pdg"])
            alone = {g: (v if g == main else 0) for g, v in vals.items()}               # the total, nothing subtracted
            total = br if md["kind"] == "derived" else evaluate(md["calc"], alone)
            if total <= thr:
                low.append(md["pdg"])
            if md["kind"] == "remainder" and not subtracted_resonances(md) <= allowed:
                foreign.append(f"{md['pdg']} ({', '.join(sorted(subtracted_resonances(md) - allowed))})")
        out += [Result(f"{s}: every branching ratio = its calc from PDG's printed values (K0S factors, subtractions)",
                       not bad_calc, f"{len(modes)} modes" if not bad_calc else f"differ: {bad_calc}"),
                Result(f"{s}: a K0S counts twice (factor 2 per K0S, a K0 or anti-K0 in the final state)", not bad_k0,
                       "all" if not bad_k0 else f"not: {bad_k0}"),
                Result(f"{s}: every mode's PDG total above {thr * 100} %", not low,
                       "all" if not low else f"below: {low}"),
                Result(f"{s}: the subtractions are of parts made through the README's {sorted(allowed)}", not foreign,
                       "all" if not foreign else f"also through others: {foreign}")]
    return out


def readme_k0s():
    """The README's K0S relation, as Gamma(K0) - (its right side), and the symbols."""
    GK, GS = sp.symbols("Gamma_K0 Gamma_K0S", positive=True)
    lhs, rhs = L.sides(L.between(L.line("rule makes the assumption $"), "makes the assumption $", "$"))
    toks = {r"\Gamma(\mathrm K^0_\mathrm S)": GS, r"\Gamma(\mathrm K^0)": GK}
    return L.to_sympy(lhs, toks) - L.to_sympy(rhs, toks), GK, GS


@derivation("cd.k0s")
def check_k0s():
    """A K0S counts twice: PDG's assumption 2 Gamma(K0S) = Gamma(K0), a neutral kaon being half K0S, half K0L."""
    rel, GK, GS = readme_k0s()
    m = re.fullmatch(r"(\d) Gamma\(K0S\) = Gamma\(K0\)", PDG.K0S_ASSUMPTION["text"])
    pdg = int(m.group(1)) * GS - GK
    # K0 = (K0S + K0L)/sqrt(2) with CP conserved: |<K0S|K0>|^2 = 1/2 of the rate
    share = sp.Abs(1 / sp.sqrt(2))**2
    line = L.line(r"a neutral kaon is half a $\mathrm K^0_\mathrm S$ and half a $\mathrm K^0_\mathrm L$")
    with L.patched(r"$\Gamma(\mathrm K^0) = 2\Gamma", r"$\Gamma(\mathrm K^0) = 3\Gamma"):
        broken = readme_k0s()[0]
    page = int(re.search(r"that the \$\\mathrm D\^\\pm\$ listing names \(p\. (\d+)\)",
                         L.line("rule makes the assumption $")).group(1))
    return [exact(f"README relation = PDG's \"{PDG.K0S_ASSUMPTION['text']}\" (D+- listing, p. "
                  f"{PDG.K0S_ASSUMPTION['page']})", sp.expand(rel + pdg)),
            Result("README's page = the listing's", page == PDG.K0S_ASSUMPTION["page"], f"p. {page}"),
            exact("K0 = (K0S + K0L)/sqrt(2): a K0 is a K0S with probability 1/2, so Gamma(K0 X) = 2 Gamma(K0S X)",
                  (share * GK - GS).subs(GK, 2 * GS)),
            Result("README: a neutral kaon is half a K0S and half a K0L (also as a product)",
                   "half a $\\mathrm K^0_\\mathrm S$ and half a $\\mathrm K^0_\\mathrm L$" in line
                   and "counts half as a $\\mathrm K^0_\\mathrm S$ and half as a $\\mathrm K^0_\\mathrm L$" in
                   L.line("counts half as a"), "as written"),
            exact_fails("control: README with 3 Gamma(K0S)", sp.expand(broken + pdg))]


def readme_coverage():
    ln = L.line("of the rates before the renormalisation")
    lo, hi = re.search(r"The hadronic modes cover (\d+)–(\d+)% of the rates", ln).groups()
    return int(lo), int(hi)


@derivation("cd.coverage")
def check_coverage():
    """The hadronic modes cover 60-78 % of the rates (the sums of their PDG branching ratios, rounded)."""
    lo, hi = readme_coverage()
    pct = {s: float(sum(rat(md["br"]) for md in PDG.DECAYS[s]["hadronic"]) * 100) for s in SPECIES}
    got = (round(min(pct.values())), round(max(pct.values())))
    with L.patched(r"cover 60–78", r"cover 59–78"):
        broken = readme_coverage()
    return [Result("README's range = the smallest and largest sum of the PDG 2026 branching ratios of the hadronic "
                   "modes", (lo, hi) == got, ", ".join(f"{s} {v:.2f} %" for s, v in pct.items())),
            control_fails("README with 59", broken != got)]


SUMMARY_HEAD = r"| | $\mathrm D^+$ | $\mathrm D^0$ | $\mathrm D_s^+$ | $\Lambda_c^+$ |"


def readme_summary():
    """{row label: [cells of D+, D0, Ds+, Lambda_c+]} of the README's table of the modes."""
    return {row[0]: row[1:] for row in L.table(SUMMARY_HEAD)}


@functools.cache
def summary_values():
    """The table's rows as the rules give them from PDG 2026 (and Meinel 2018): {row label: {species: value}}."""
    out = {k: {} for k in ("modes", "sum_e", "incl", "scale", "sl", "lep", "lep_gammas", "had", "sum_had", "renorm")}
    for s in SPECIES:
        modes = sl_modes(s)
        e = [m for m in modes if m["lepton"] == "e"]
        mu = [m for m in modes if m["lepton"] == "mu"]
        se, smu = sum(m["br"] for m in e), sum(m["br"] for m in mu)
        incl = PDG.INCLUSIVE_E[s]
        scale = rat(incl["br"]) / se
        lep = PDG.DECAYS[s]["leptonic"]
        b_lep = sum((rat(md["br"]) for md in lep), sp.Integer(0))
        had = PDG.DECAYS[s]["hadronic"]
        b_had = sum(rat(md["br"]) for md in had)
        out["modes"][s] = (len(e), len(mu))
        out["sum_e"][s] = se
        out["incl"][s] = (rat(incl["br"]), gamma_number(incl["pdg"]), incl["page"])
        out["scale"][s] = scale
        out["sl"][s] = scale * (se + smu)
        out["lep"][s] = b_lep
        out["lep_gammas"][s] = sorted(gamma_number(md["pdg"]) for md in lep)
        out["had"][s] = len(had)
        out["sum_had"][s] = b_had
        out["renorm"][s] = (1 - scale * (se + smu) - b_lep) / b_had
    return out


def summary_results():
    """The README's table against the rules applied to PDG 2026: one Result per row."""
    rows, v = readme_summary(), summary_values()
    out = []

    def row(label, ok_of, detail):
        cells = rows[label]
        ok = len(cells) == 4 and all(ok_of(c, s) for c, s in zip(cells, SPECIES))
        out.append(Result(f"README row \"{label}\" = the rules' values", ok, detail))
    row(r"Semileptonic modes, $\mathrm e^+$ and $\mu^+$",
        lambda c, s: tuple(map(int, re.fullmatch(r"(\d+) and (\d+)", c).groups())) == v["modes"][s],
        str(v["modes"]))
    row(r"Sum of their $\mathrm e^+$ branching ratios", lambda c, s: printed(c, v["sum_e"][s]),
        ", ".join(f"{s} {v['sum_e'][s]}" for s in SPECIES))
    def incl_ok(c, s):
        m = re.fullmatch(r"([\d.]+) \(Γ(\d+), p\. (\d+)\)", c)
        return m is not None and (sp.Rational(m.group(1)), int(m.group(2)), int(m.group(3))) == v["incl"][s]
    row(r"PDG's inclusive $\mathrm e^+$ rate", incl_ok, str({s: v["incl"][s] for s in SPECIES}))
    row("Scale factor of the semileptonic modes", lambda c, s: printed(c, v["scale"][s]),
        ", ".join(f"{s} {float(v['scale'][s]):.6f}" for s in SPECIES))
    row("Semileptonic branching ratio, scaled", lambda c, s: printed(c, v["sl"][s]),
        ", ".join(f"{s} {float(v['sl'][s]):.6f}" for s in SPECIES))

    def lep_ok(c, s):
        m = re.fullmatch(r"([\d.]+)(?: \(Γ(\d+) \+ Γ(\d+)\))?", c)
        gs = sorted(int(g) for g in m.groups()[1:] if g)
        return m is not None and printed(m.group(1), v["lep"][s]) and gs == v["lep_gammas"][s]
    row("Leptonic branching ratio", lep_ok, ", ".join(f"{s} {v['lep'][s]} {v['lep_gammas'][s]}" for s in SPECIES))
    row("Sum of their branching ratios", lambda c, s: printed(c, v["sum_had"][s]),
        ", ".join(f"{s} {float(v['sum_had'][s]):.6f}" for s in SPECIES))
    row("Renormalisation factor of the hadronic modes", lambda c, s: printed(c, v["renorm"][s]),
        ", ".join(f"{s} {float(v['renorm'][s]):.6f}" for s in SPECIES))
    return out


@derivation("cd.summary")
def check_summary():
    """The table of the modes is what the rules make of PDG 2026: counts, sums, scale and renormalisation factors."""
    out = summary_results()
    with L.patched("| Scale factor of the semileptonic modes | 1.0313 |",
                   "| Scale factor of the semileptonic modes | 1.0314 |"):
        broken = summary_results()
    out.append(control_fails("README with a scale factor 1.0314 for D+", not all(r.ok for r in broken)))
    return out


def readme_hadronic_mode_counts():
    """{species: number} of the README's hadronic modes (row "Hadronic modes" of the table)."""
    return dict(zip(SPECIES, map(int, readme_summary()["Hadronic modes"])))


@derivation("cd.counts")
def check_counts():
    """The numbers of hadronic modes the README states are those of PDG 2026 by its rules."""
    counts = readme_hadronic_mode_counts()
    pdg = {s: len(PDG.DECAYS[s]["hadronic"]) for s in SPECIES}
    with L.patched("| Hadronic modes | 21 |", "| Hadronic modes | 22 |"):
        broken = readme_hadronic_mode_counts()
    return [Result("README's numbers = PDG 2026's modes by the rules (pdg_charm)", counts == pdg, f"{pdg}"),
            control_fails("README with 22 for D+", broken != pdg)]


# --- the semileptonic modes' table ---

SL_HEAD = r"| Parent | Mode | $\mathcal B_\mathrm e$ | $\mathcal B_\mu$ | PDG 2026 | Matrix element |"
README_PARTICLES = ((r"\bar{\mathrm K}^{\ast 0}", "Kstar0bar"), (r"\mathrm K^{\ast -}", "Kstar-"),
                    (r"\mathrm K^{\ast 0}", "Kstar0"), (r"\bar{\mathrm K}^0", "K0bar"), (r"\mathrm K^-", "K-"),
                    (r"\mathrm K^0", "K0"), (r"\pi^+", "pi+"), (r"\pi^-", "pi-"), (r"\pi^0", "pi0"),
                    (r"\eta'", "etap"), (r"\eta", "eta"), (r"\rho^0", "rho0"), (r"\rho^-", "rho-"),
                    (r"\omega", "omega"), (r"\phi", "phi"), (r"f_0(980)", "f0_980"), (r"\Lambda", "Lambda"),
                    (r"\mathrm n", "n"))
VECTORS = {"Kstar0bar", "Kstar-", "Kstar0", "rho0", "rho-", "omega", "phi"}
BESIII_ROWS = (r"$D^0\to K^- e^+\nu_e$ & $  0.7163\pm0.0024\pm 0.0034$ & $  0.309\pm0.020\pm 0.013$",
               r"$D^0\to \pi^- e^+\nu_e$ & $  0.1437\pm0.0017\pm 0.0008$ & $  0.279\pm0.035\pm 0.011$")
BESIII_TABLE15 = r"BESIII (This work)    & $0.309 \pm  0.020 \pm 0.013$ & $0.279 \pm 0.035 \pm 0.011$"
BESIII_TEXT = (r"\alpha^{D \rightarrow K}   = 0.310 \pm  0.020 \pm 0.013",
               r"\alpha^{D \rightarrow \pi} = 0.279 \pm 0.035 \pm 0.011")


def besiii_alpha():
    """alpha of D -> K and D -> pi in BESIII's Table XIII (the rows as transcribed)."""
    return [sp.Rational(re.findall(r"\$\s*([\d.]+)\\pm", r)[1]) for r in BESIII_ROWS]


def readme_mode(tex):
    """(hadrons, complement?) of a README mode such as $(\\mathrm K^-\\pi^+)\\ell^+\\nu$."""
    body = tex.strip().strip("$")
    assert body.endswith(r"\ell^+\nu"), body
    body = body[:-len(r"\ell^+\nu")]
    complement = body.startswith("(") and body.endswith(")")
    body = body[1:-1] if complement else body
    names = []
    while body:
        for t, name in README_PARTICLES:
            if body.startswith(t):
                names.append(name)
                body = body[len(t):]
                break
        else:
            raise ValueError(f"unknown particle in {tex}: {body}")
    return tuple(names), complement


def pdg_cell(m):
    """The README's PDG 2026 cell of a mode: Γn, 2Γn for a K0S mode, Γa + Γb for a sum."""
    md = m["md"]
    if md is None:
        return None
    n = gamma_number(md["pdg"])
    if md["kind"] == "k0s":
        return f"{md['calc'].split('*')[0]}Γ{n}"
    if md["kind"] == "derived":
        return " + ".join(f"Γ{g}" for g in sorted(int(x) for x in re.findall(r"G(\d+)", md["calc"])))
    return f"Γ{n}"


def expected_sl_rows():
    """{(species, hadrons): (B_e, B_mu, PDG cell, matrix element cell)} by the README's rules, from PDG 2026."""
    alpha_K, alpha_pi = besiii_alpha()
    out = {}
    for s in SPECIES:
        modes = sl_modes(s)
        for hadrons in dict.fromkeys(m["hadrons"] for m in modes):
            e = next((m for m in modes if m["hadrons"] == hadrons and m["lepton"] == "e"), None)
            mu = next((m for m in modes if m["hadrons"] == hadrons and m["lepton"] == "mu"), None)
            cells = [c for c in (pdg_cell(e) if e else None, pdg_cell(mu) if mu else None) if c]
            pdg = ", ".join(cells) if cells else "Meinel 2018"
            if s == "Lambda_c+":
                me = "lattice, Meinel 2017" if hadrons == ("Lambda",) else "lattice, Meinel 2018"
            elif len(hadrons) > 1:
                me = "phase space"
            elif hadrons[0] in VECTORS:
                ff = PDG.SL_FORM_FACTORS[(s, hadrons[0])]
                me = (ff["rV"][0], ff["r2"][0])
            else:
                kind = ({"K0bar": "K", "K-": "K", "pi0": "pi", "pi-": "pi"}.get(hadrons[0])
                        if s in ("D+", "D0") else None)
                me = {"K": alpha_K, "pi": alpha_pi, None: sp.Integer(0)}[kind]
            out[(s, hadrons)] = (e["br"] if e else None, mu["br"] if mu else None, pdg, me)
    return out


def readme_sl_rows():
    """{(species, hadrons): (B_e cell, B_mu cell, PDG cell, matrix element cell, complement?)} of the README."""
    out, parent = {}, None
    names = {v: k for k, v in SPECIES_TEX.items()}
    for row in L.table(SL_HEAD):
        parent = names[row[0]] if row[0] else parent
        hadrons, complement = readme_mode(row[1])
        if (parent, hadrons) in out:
            raise ValueError(f"two rows for {parent} {hadrons}")
        out[(parent, hadrons)] = (row[2], row[3], row[4], row[5], complement)
    return out


def matrix_element_ok(cell, me):
    if isinstance(me, tuple):
        m = re.fullmatch(r"\$r_V = ([\d.]+)\$, \$r_2 = ([\d.]+)\$", cell)
        return m is not None and (sp.Rational(m.group(1)), sp.Rational(m.group(2))) == tuple(map(rat, me))
    if isinstance(me, str):
        return cell == me
    m = re.fullmatch(r"\$\\alpha = ([\d.]+)\$", cell)
    return m is not None and sp.Rational(m.group(1)) == me


def sl_table_results():
    readme, want = readme_sl_rows(), expected_sl_rows()
    out = [Result("README rows = the modes the rules select, each once", set(readme) == set(want),
                  f"{len(want)} rows" if set(readme) == set(want) else
                  f"missing {sorted(set(want) - set(readme))}, extra {sorted(set(readme) - set(want))}")]
    bad = []
    for k in sorted(set(readme) & set(want)):
        be, bmu, pdg, me, complement = readme[k]
        we, wmu, wpdg, wme = want[k]
        ok = ((be == "—" if we is None else printed(be, we) and sp.Rational(be) == we)
              and (bmu == "—" if wmu is None else printed(bmu, wmu) and sp.Rational(bmu) == wmu)
              and pdg == wpdg and matrix_element_ok(me, wme)
              and complement == (len(k[1]) > 1 and any(m["md"] and m["md"]["swave"] for m in sl_modes(k[0])
                                                       if m["hadrons"] == k[1])))
        if not ok:
            bad.append((k, readme[k], want[k]))
    out.append(Result("every row's B_e, B_mu (PDG as printed, K0S x 2), PDG entries and matrix element inputs "
                      "(BESIII's alpha, PDG's r_V and r_2) = the sources'", not bad,
                      "all" if not bad else f"differ: {bad[:3]}"))
    return out


@derivation("cd.sl_table")
def check_sl_table():
    """The semileptonic table: PDG's modes and branching ratios, BESIII's alpha, PDG's r_V and r_2, Meinel's rates."""
    out = sl_table_results()
    with L.patched("| 0.0881 | 0.0868 |", "| 0.0882 | 0.0868 |"):
        broken = sl_table_results()
    out.append(control_fails("README with B_e = 0.0882 for D+ -> K0bar", not all(r.ok for r in broken)))
    # the pages of the entries
    sentence = L.line("entries are on pp.")
    m = re.search(r"The \$\\mathrm D\^\+\$ entries are on pp\. (\d+)–(\d+) of its listing\. Those of the "
                  r"\$\\mathrm D\^0\$ are on p\. (\d+), of the \$\\mathrm D_s\^\+\$ on p\. (\d+) and of the "
                  r"\$\\Lambda_c\^\+\$ on p\. (\d+)\.", sentence)
    pages = {s: sorted({mm["md"]["page"] for mm in sl_modes(s) if mm["md"]}) for s in SPECIES}
    readme_pages = {"D+": [int(m.group(1)), int(m.group(2))], "D0": [int(m.group(3))], "Ds+": [int(m.group(4))],
                    "Lambda_c+": [int(m.group(5))]}
    out.append(Result("README's pages = those of the PDG entries (pdg_charm)", pages == readme_pages, f"{pages}"))
    l1520 = [md for md in PDG.DECAYS["Lambda_c+"]["semileptonic"] if "Lambda1520" in md["fs"]]
    out.append(Result("Lambda_c+ -> Lambda(1520) e+ nu is at 0.1 % exactly (PDG 2026), so not above the threshold",
                      len(l1520) == 1 and rat(l1520[0]["br"]) == sp.Rational(1, 1000)
                      and "$\\Lambda_c^+ \\to \\Lambda(1520)\\mathrm e^+\\nu$, at 0.1% exactly, is not "
                          "above the threshold." in sentence, l1520[0]["pdg"] if l1520 else "none"))
    out.append(in_source("BESIII Table XIII: alpha = 0.309 (D0 -> K- e+ nu), 0.279 (D0 -> pi- e+ nu)", BESIII,
                         *BESIII_ROWS))
    # the pages of the form-factor ratios, and which are averages
    text = L.line("The ratios are PDG's averages, or single measurements")
    m = re.search(r"\(\$\\mathrm D\^\\pm\$ listing pp\. (\d+)–(\d+), \$\\mathrm D\^0\$ p\. (\d+), "
                  r"\$\\mathrm D_s\^\\pm\$ pp\. (\d+)–(\d+)\)", text)
    ranges = {"D+-": range(int(m.group(1)), int(m.group(2)) + 1), "D0": range(int(m.group(3)), int(m.group(3)) + 1),
              "Ds+-": range(int(m.group(4)), int(m.group(5)) + 1)}
    pages = {(ff["listing"], ff[r][3]) for ff in PDG.SL_FORM_FACTORS.values() for r in ("rV", "r2")}
    single = sorted(f"{k[0]} {k[1]}" for k, ff in PDG.SL_FORM_FACTORS.items() if ff["rV"][2] != "OUR AVERAGE")
    out.append(Result("README's pages hold the ratios (pdg_charm.SL_FORM_FACTORS); single measurements only where "
                      "PDG has no average", all(pg in ranges[li] for li, pg in pages)
                      and all(isinstance(ff["rV"][1], tuple) == (ff["rV"][2] != "OUR AVERAGE")
                              for ff in PDG.SL_FORM_FACTORS.values()),
                      f"pages {sorted(pages)}; single measurements: {single}"))
    out.append(Result("D0 -> rho- takes the average of the D+- listing (D+,D0 -> rho e+ nu)",
                      PDG.SL_FORM_FACTORS[("D0", "rho-")]["listing"] == "D+-"
                      and PDG.SL_FORM_FACTORS[("D0", "rho-")]["rV"] == PDG.SL_FORM_FACTORS[("D+", "rho0")]["rV"]
                      and r"$\mathrm D^0 \to \rho^-$ takes the average of the $\mathrm D^\pm$ listing." in
                      L.line(r"$\mathrm D^0 \to \rho^-$ takes the average"), "as written"))
    return out


# --- the V-A lepton side (Dirac algebra: verify.dirac, as kd.W) ---

def eps4(a, b, c, d):
    """eps_{mu nu rho sigma} a^mu b^nu c^rho d^sigma with eps_{0123} = +1."""
    return sp.Matrix([list(a), list(b), list(c), list(d)]).det()


def eps_raised(a, b, c):
    """e^mu for e_mu = eps_{mu nu rho sigma} a^nu b^rho c^sigma."""
    unit = [[int(i == j) for j in range(4)] for i in range(4)]
    low = [eps4(unit[i], a, b, c) for i in range(4)]
    return [low[0], -low[1], -low[2], -low[3]]


def projector(p, m, s):
    """v v-bar of an antifermion of momentum p, mass m and spin s: (p-slash - m)(1 + g5 s-slash)/2."""
    _, g5, one = dirac()
    return (slash(p) - m * one) * (one + g5 * slash(s)) / 2


def lepton_trace(Jr, Ji, nu, rho, chirality=-1):
    """Sum over the neutrino spin of |u_nu-bar J-slash (1 + chirality g5) v_l|^2 for the complex current J = Jr + i Ji
    and v_l v_l-bar = rho: Tr[nu-slash J-slash (1 - g5) rho (1 + g5) J*-slash] for V-A (chirality -1)."""
    _, g5, one = dirac()
    J = [a + sp.I * b for a, b in zip(Jr, Ji)]
    Jc = [a - sp.I * b for a, b in zip(Jr, Ji)]
    return sp.expand((slash(nu) * slash(J) * (one + chirality * g5) * rho * (one - chirality * g5) * slash(Jc)).trace())


@functools.cache
def generic_lepton_side():
    """Symbols of a generic current and generic 4-vectors, the V-A trace for an l+ of spin s, and the spin-summed
    trace as a function of the momentum put for p_l."""
    Jr, Ji, nu, l, s = (sp.symbols(f"{v}0:4", real=True) for v in ("r", "i", "n", "l", "s"))
    m = sp.Symbol("m", positive=True)
    one = dirac()[2]
    spin = lepton_trace(Jr, Ji, nu, projector(l, m, s))
    summed = lambda p, chirality=-1: lepton_trace(Jr, Ji, nu, slash(p) - m * one, chirality)
    return Jr, Ji, nu, l, s, m, spin, summed


TRACE_PIN = r"\vert\mathcal M\vert^2_{\rm sum} = 8\left[2(p_\nu\cdot J_r)(p_\ell\cdot J_r)"


def readme_trace(Jr, Ji, nu, l):
    """The README's spin-summed |M|^2 for the vectors Jr, Ji, p_nu, p_l."""
    lhs, rhs = L.sides(L.display(TRACE_PIN))
    assert lhs == r"\vert\mathcal M\vert^2_{\rm sum}", lhs
    toks = {r"(p_\nu\cdot J_r)": mdot(nu, Jr), r"(p_\ell\cdot J_r)": mdot(l, Jr), r"(p_\nu\cdot J_i)": mdot(nu, Ji),
            r"(p_\ell\cdot J_i)": mdot(l, Ji), r"(p_\nu\cdot p_\ell)": mdot(nu, l), "J_r^2": mdot(Jr, Jr),
            "J_i^2": mdot(Ji, Ji), r"\epsilon(J_r, J_i, p_\nu, p_\ell)": eps4(Jr, Ji, nu, l)}
    return L.to_sympy(rhs, toks)


@derivation("cd.trace")
def check_trace():
    """V-A, massless neutrino: the spin-summed trace of a complex current is the README's form, eps_{0123} = +1."""
    Jr, Ji, nu, l, s, m, spin, summed = generic_lepton_side()
    readme = readme_trace(Jr, Ji, nu, l)
    with L.patched(r"\right] + 16\thinspace\epsilon(J_r", r"\right] - 16\thinspace\epsilon(J_r"):
        broken = readme_trace(Jr, Ji, nu, l)
    conv = L.inline(r"\epsilon_{0123} = +1")
    H, zero = Jr, (0, 0, 0, 0)
    W0 = 2 * mdot(nu, H) * mdot(l, H) - mdot(H, H) * mdot(nu, l)            # kaons' K_l3 density (CGG eq. 6.9 form)
    return [exact("README's form = Tr[p_nu-slash J-slash (1 - g5)(p_l-slash - m)(1 + g5) J*-slash] (any J, p_nu, p_l)",
                  readme - summed(l)),
            Result("README's convention eps_{0123} = +1 is eps4's", conv == r"\epsilon_{0123} = +1"
                   and eps4((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)) == 1, "as written"),
            exact("real current (J_i = 0): the bracket is the kaons' K_l3 density W_0, |M|^2_sum = 8 W_0",
                  readme_trace(Jr, zero, nu, l) - 8 * W0),
            exact_fails("control: README with - 16 eps(...)", broken - summed(l))]


# --- the spin rule (display equation) ---

def readme_pol():
    """(coefficient, p_l + m_l s as a function of (p_l, m_l, s), componentwise) of the README's spin rule."""
    lhs, rhs = L.sides(L.parts(L.display(r"\vert\mathcal M\vert^2(s) = \tfrac12"))[0])
    assert lhs == r"\vert\mathcal M\vert^2(s)", lhs
    m = re.fullmatch(r"(.*)\\thinspace\\vert\\mathcal M\\vert\^2_\{\\rm sum\}\(p_\\ell \\to (.*)\)", rhs)
    pl, ml, s = sp.symbols("p_l m_l s")
    sub = L.to_sympy(m.group(2), {r"p_\ell": pl, r"m_\ell": ml, "s": s})

    def substituted(p, mass, spin):
        return [sub.subs({pl: a, ml: mass, s: b}) for a, b in zip(p, spin)]
    return L.to_sympy(m.group(1), {}), substituted


@derivation("cd.pol")
def check_pol():
    """V-A, massless neutrino: an l+ of spin s has 1/2 |M|^2_sum(p_l -> p_l + m_l s), for any hadronic current."""
    Jr, Ji, nu, l, s, m, spin, summed = generic_lepton_side()
    coef, sub = readme_pol()
    rule = coef * summed(sub(l, m, s))
    with L.patched(r"(p_\ell \to p_\ell + m_\ell s)", r"(p_\ell \to p_\ell - m_\ell s)"):
        coef_b, sub_b = readme_pol()
    va_plus = lepton_trace(Jr, Ji, nu, projector(l, m, s), chirality=1)
    # helicity +1/2 of an l+ along z: s_h = (P, 0, 0, E)/m, E = m (v + 1/v)/2, P = m (v - 1/v)/2
    v = sp.Symbol("v", positive=True)
    E, P = m * (v + 1 / v) / 2, m * (v - 1 / v) / 2
    pz, sh = (E, 0, 0, P), (P / m, 0, 0, E / m)
    neg = [a - m * b for a, b in zip(pz, sh)]                                      # p_l - m s_h, helicity -1/2
    # the README's helicity difference |M|^2_sum(p_l -> m_l s)
    diff_tex = L.inline(r"\vert\mathcal M\vert^2_{\rm sum}(p_\ell \to m_\ell s)")
    ms = L.to_sympy(re.fullmatch(r"\\vert\\mathcal M\\vert\^2_\{\\rm sum\}\(p_\\ell \\to (.*)\)", diff_tex).group(1),
                    {r"m_\ell": m, "s": sp.Symbol("s_")})
    return [exact("trace with v v-bar = (p-slash - m)(1 + g5 s-slash)/2 = README's 1/2 |M|^2_sum(p_l + m_l s) "
                  "(any complex current J, any p_l, s, p_nu)", spin - rule),
            exact_fails("control: README with p_l - m_l s", spin - coef_b * summed(sub_b(l, m, s))),
            exact("V+A instead (1 + g5): the rule needs p_l - m_l s, so the README's rule is that of V-A",
                  va_plus - coef * summed([a - m * b for a, b in zip(l, s)], chirality=1)),
            exact_fails("control: V+A with the README's p_l + m_l s",
                        va_plus - coef * summed(sub(l, m, s), chirality=1)),
            exact("the helicities +1/2 and -1/2 add up to |M|^2_sum (s -> -s)",
                  rule + coef * summed(sub(l, m, [-a for a in s])) - summed(l)),
            exact("they differ by the README's |M|^2_sum(p_l -> m_l s): the trace is linear in p_l",
                  rule - coef * summed(sub(l, m, [-a for a in s]))
                  - summed([ms.subs(sp.Symbol("s_"), b) for b in s])),
            exact("helicity -1/2: p_l - m_l s_h = (m/v)(1, 0, 0, -1) -> 0 as m_l -> 0: a massless l+ is right-handed",
                  sum((a - b)**2 for a, b in zip(neg, (m / v, 0, 0, -m / v))))]


# --- P -> P' l+ nu ---

def readme_pp_current(p, pp, fp, fm):
    """The README's H for one component of p, p' and the values f_+(t), f_-(t)."""
    lhs, rhs = L.sides(L.between(L.line("write the hadronic current as $H ="), "current as $", "$"))
    assert lhs == "H", lhs
    return L.to_sympy(rhs, {"f_+(t)": fp, "f_-(t)": fm, "p'": pp, "p": p})


@derivation("cd.ppcurrent")
def check_ppcurrent():
    """The README's P -> P' current is Richman & Burchat's eq. 82, the same current as their eq. 81 by eq. 83."""
    p, pp, fp, fm, M, m, q2 = sp.symbols("p pp f_p f_m M m q2")
    H = readme_pp_current(p, pp, fp, fm)
    rb82 = fp * (p + pp) + fm * (p - pp)
    F0 = fp + q2 / (M**2 - m**2) * fm                                            # eq. 83, F_1 = f_+
    with L.patched(r"$H = f_+(t)(p + p')", r"$H = f_+(t)(p - p')"):
        broken = readme_pp_current(p, pp, fp, fm)
    return [exact("README H = RB eq. 82 (per component)", H - rb82),
            exact("RB eq. 81 with F_1 = f_+ and F_0 of eq. 83 = eq. 82 (any q^2)",
                  RB_81(fp, F0, p, pp, M, m, q2) - rb82),
            exact_fails("control: README with f_+(t)(p - p')", broken - rb82)]


def readme_fminus(f0, fp, M, m, t):
    lhs, rhs = L.sides(L.between(L.line("which gives $f_- ="), "which gives $", "$"))
    assert lhs == "f_-", lhs
    return L.to_sympy(rhs, {"f_0": f0, "f_+": fp, "M": M, "m": m, "t": t})


@derivation("cd.fminus")
def check_fminus():
    """f_- follows from Richman & Burchat's eq. 83, F_0 = f_+ + q^2 f_-/(M^2 - m^2), with F_0 = f_0 and q^2 = t."""
    f0, fp, fm, M, m, t = sp.symbols("f_0 f_p f_m M m t")
    readme = readme_fminus(f0, fp, M, m, t)
    solved = sp.solve(sp.Eq(f0, fp + t / (M**2 - m**2) * fm), fm)[0]
    with L.patched(r"$f_- = (f_0 - f_+)(M^2 - m^2)/t$", r"$f_- = (f_0 - f_+)(M^2 + m^2)/t$"):
        broken = readme_fminus(f0, fp, M, m, t)
    return [exact("README f_- = eq. 83 solved for f_-", readme - solved),
            exact_fails("control: README with M^2 + m^2", broken - solved)]


BESIII_POLE = r"f_+(q^2)=\frac{f_+(0)}{(1-\frac{q^2}{M^2_{\rm pole}})(1-\alpha \frac{q^2}{M^2_{\rm pole}})}"


def readme_formfactors(t, mp_, alpha, m0):
    """f_+ and f_0 of the README's display equation (found by f_0's part, which the control leaves intact)."""
    tokens = {"t": t, r"m_{\rm p}": mp_, r"\alpha": alpha, "m_0": m0}
    disp = L.display(r"f_0(t) = \frac{1}{1 - t/m_0^2}")
    return [L.to_sympy(L.sides(L.part(disp, f))[1], tokens) for f in ("f_+(t)", "f_0(t)")]


@derivation("cd.formfactors")
def check_formfactors():
    """f_+ is BESIII's modified pole, f_0 a single pole (RB eq. 88), f_+(0) = f_0(0); BESIII's alpha."""
    t, mp_, m0, M, mh, Mp2 = sp.symbols("t m_p m_0 M m_h Mp2", positive=True)
    alpha, f0_ = sp.symbols("alpha f_0", real=True)
    fp, f0 = readme_formfactors(t, mp_, alpha, m0)
    bes = L.to_sympy(BESIII_POLE.split("=", 1)[1],
                     {"f_+(0)": f0_, "q": sp.sqrt(t), r"M^2_{\rm pole}": Mp2, r"\alpha": alpha})
    with L.patched(r"(1 - \alpha t/m_{\rm p}^2)}", r"(1 + \alpha t/m_{\rm p}^2)}"):
        fp_b, _ = readme_formfactors(t, mp_, alpha, m0)
    rb83 = sp.solve(sp.Eq(f0, fp + t / (M**2 - mh**2) * sp.Symbol("fm")), sp.Symbol("fm"))[0]   # f_- of RB eq. 83
    out = [in_source("BESIII eq. II.4 (modified pole model) as transcribed", BESIII, BESIII_POLE),
           exact("README f_+ = BESIII's f_+(q^2)/f_+(0), q^2 = t, M_pole = m_p", fp - (bes / f0_).subs(Mp2, mp_**2)),
           exact_fails("control: README with (1 + alpha t/m_p^2)", fp_b - (bes / f0_).subs(Mp2, mp_**2)),
           exact(f"README f_0 = RB eq. 88 with n = {RB_EQ88_N}, normalised", f0 - 1 / (1 - t / m0**2)**RB_EQ88_N),
           exact("f_+(0) = f_0(0) = 1 (RB eqs. 81-83: F_0(0) = F_1(0))",
                 (fp.subs(t, 0) - 1)**2 + (f0.subs(t, 0) - 1)**2),
           exact("f_- finite at t -> 0: (M^2 - m^2)(1/m_0^2 - (1 + alpha)/m_p^2)",
                 sp.limit(rb83, t, 0) - (M**2 - mh**2) * (1 / m0**2 - (1 + alpha) / mp_**2))]
    # alpha: BESIII's fits of D -> K and D -> pi
    readme_alpha = (r"The modified pole's $\alpha$ is from Ablikim et al. 2015 for $\mathrm{D} \to \mathrm{K}$ and "
                    r"$\mathrm{D} \to \pi$, and 0 otherwise")
    said = (r"For $\mathrm D \to \mathrm K$, the tables of Ablikim et al. give $\alpha = 0.309$ (Tables XIII and XV), "
            r"and their text 0.310 (eq. VI.25). The code uses the tables' value. For $\mathrm D \to \pi$ both give "
            r"0.279.")
    out += [Result("README: alpha of the modified pole from Ablikim et al. 2015 (BESIII) for D -> K, D -> pi, else 0",
                   readme_alpha in L.line("The modified pole's $\\alpha$ is from"), "as written"),
            in_source("BESIII Table XIII: alpha = 0.309 (D0 -> K- e+ nu), 0.279 (D0 -> pi- e+ nu)", BESIII,
                      *BESIII_ROWS),
            in_source("BESIII Table XV (this work): 0.309, 0.279", BESIII, BESIII_TABLE15),
            in_source("BESIII text, eqs. VI.25-26: alpha^{D->K} = 0.310, alpha^{D->pi} = 0.279", BESIII,
                      *BESIII_TEXT),
            Result("README: the tables give 0.309, the text 0.310, both 0.279 for D -> pi",
                   said in L.line("the tables of Ablikim et al. give") and besiii_alpha() == [sp.Rational("0.309"),
                                                                                         sp.Rational("0.279")],
                   "as written, with the tables' values")]
    return out


# --- P -> V l+ nu: Richman & Burchat's current, the single poles, A_0(0), the density ---

RB85_PIN = r"\langle\mathrm V\vert V^\mu - A^\mu\vert\mathrm P\rangle = "


def readme_rb85(sym):
    """The README's current (as a linear form in the vector tokens) and its A_3."""
    disp = L.display(RB85_PIN)
    lhs, rhs = L.sides(L.parts(disp)[0])
    toks = {r"\epsilon^{\mu\nu\alpha\beta}": sym["X"], r"\varepsilon^\ast_\nu p'_ \alpha p_\beta": 1,
            r"\varepsilon^{\ast\mu}": sym["E"], r"\varepsilon^\ast\cdot q": sym["eq"], r"(p + p')^\mu": sym["S"],
            r"q^\mu": sym["Q"], "q^2": sym["q2"], "A_1": sym["A1"], "A_2": sym["A2"], "A_3": sym["A3"],
            "A_0": sym["A0"], "V": sym["V"], "M": sym["M"], "m": sym["m"], "i": sp.I}
    current = L.to_sympy(rhs, toks)
    a3_l, a3_r = L.sides(L.part(disp, "A_3 ="))
    a3 = L.to_sympy(a3_r, {"A_1": sym["A1"], "A_2": sym["A2"], "M": sym["M"], "m": sym["m"]})
    return current, a3


@derivation("cd.rb85")
def check_rb85():
    """The README's P -> V current and A_3 are Richman & Burchat's eqs. 85-86, as transcribed."""
    sym = {k: sp.Symbol(k) for k in ("X", "E", "eq", "S", "Q", "q2", "A1", "A2", "A3", "A0", "V", "M", "m")}
    current, a3 = readme_rb85(sym)
    rb = RB_85(sym["X"], sym["E"], sym["S"], sym["Q"], sym["eq"], sym["q2"], sym["M"], sym["m"], sym["V"],
               sym["A1"], sym["A2"], sym["A3"], sym["A0"])
    with L.patched(r"+ \frac{\varepsilon^\ast\cdot q}{M + m}(p + p')^\mu A_2",
                   r"- \frac{\varepsilon^\ast\cdot q}{M + m}(p + p')^\mu A_2"):
        broken, _ = readme_rb85(sym)
    return [exact("README current = RB eq. 85 (A_3 and A_0 terms combined)", current - rb),
            exact("README A_3 = RB eq. 86", a3 - RB_86(sym["M"], sym["m"], sym["A1"], sym["A2"])),
            exact_fails("control: README with - A_2 term", broken - rb)]


PV_PIN = r"V(t) = \frac{r_V}{1 - t/m_V^2}"


def readme_poles(sym):
    disp = L.display(PV_PIN)
    toks = {"t": sym["t"], "r_V": sym["rV"], "r_2": sym["r2"], "m_V": sym["mV"], "m_A": sym["mA"], "m_0": sym["m0"],
            "A_0(0)": sym["A00"]}
    return {f: L.to_sympy(L.sides(L.part(disp, f + "(t) ="))[1], toks) for f in ("V", "A_1", "A_2", "A_0")}


@derivation("cd.pv")
def check_pv():
    """V, A_1, A_2, A_0 are RB eq. 88 single poles with PDG's r_V = V(0)/A_1(0), r_2 = A_2(0)/A_1(0); A_1, A_2 share."""
    sym = {k: sp.Symbol(k, positive=True) for k in ("t", "rV", "r2", "mV", "mA", "m0", "A00")}
    ff = readme_poles(sym)
    t = sym["t"]
    pole = lambda f0, m: f0 / (1 - t / m**2)**RB_EQ88_N                              # eq. 88
    with L.patched(r"A_2(t) = \frac{r_2}{1 - t/m_A^2}", r"A_2(t) = \frac{r_2}{1 - t/m_V^2}"):
        broken = readme_poles(sym)
    readme_sentence = L.line(r"These are single poles with the ratios")
    return [exact("V, A_1, A_2, A_0 = eq. 88 (n = 1) with the poles m_V (1-), m_A (1+), m_0 and V(0) = r_V, "
                  "A_1(0) = 1, A_2(0) = r_2", sum((ff[f] - pole(v, m))**2 for f, v, m in
                                                  (("V", sym["rV"], sym["mV"]), ("A_1", 1, sym["mA"]),
                                                   ("A_2", sym["r2"], sym["mA"]), ("A_0", sym["A00"], sym["m0"])))),
            exact("V(0)/A_1(0) = r_V and A_2(0)/A_1(0) = r_2: PDG's (and RB p. 87) definitions",
                  (ff["V"] / ff["A_1"] - sym["rV"]).subs(t, 0)**2 + (ff["A_2"] / ff["A_1"] - sym["r2"]).subs(t, 0)**2),
            exact("A_1 and A_2 share their pole (both 1+, RB p. 74): A_2/A_1 = r_2 at every t",
                  ff["A_2"] / ff["A_1"] - sym["r2"]),
            exact_fails("control: README with A_2 on the pole m_V", broken["A_2"] / broken["A_1"] - sym["r2"]),
            Result("README: single poles with PDG 2026's ratios r_V = V(0)/A_1(0), r_2 = A_2(0)/A_1(0)",
                   r"single poles with the ratios $r_V = V(0)/A_1(0)$ and $r_2 = A_2(0)/A_1(0)$ of PDG 2026"
                   in readme_sentence, "as written")]


def readme_a0(M, m, r2):
    lhs, rhs = L.sides(L.between(L.line("Hence $A_0(0) ="), "Hence $", "$"))
    assert lhs == "A_0(0)", lhs
    return L.to_sympy(rhs, {"M": M, "m": m, "r_2": r2})


@derivation("cd.a0")
def check_a0():
    """A_0(0) = A_3(0) (RB p. 74) with eq. 86 at t = 0; it keeps the q^mu term of the current finite at q^2 = 0."""
    M, m, r2, mA, m0, t = sp.symbols("M m r_2 m_A m_0 t", positive=True)
    a0 = readme_a0(M, m, r2)
    said = L.inline(r"A_0(0) = A_3(0)")
    A1, A2 = 1 / (1 - t / mA**2), r2 / (1 - t / mA**2)
    A3 = RB_86(M, m, A1, A2)
    with L.patched(r"$A_0(0) = [(M + m) - (M - m)r_2]/(2m)$", r"$A_0(0) = [(M + m) + (M - m)r_2]/(2m)$"):
        broken = readme_a0(M, m, r2)
    finite = sp.limit((A3 - a0 / (1 - t / m0**2)) / t, t, 0)
    diverges = sp.limit((A3 - 1 / (1 - t / m0**2)) / t, t, 0, "+")
    return [Result("README quotes RB p. 74: \"with A_0(0) = A_3(0)\"", said == "A_0(0) = A_3(0)", "as written"),
            exact("A_0(0) = A_3(0) of RB eq. 86 with A_1(0) = 1, A_2(0) = r_2", a0 - A3.subs(t, 0)),
            Result("(A_3 - A_0)/q^2 has a finite limit at q^2 = 0 with the README's A_0(0)", finite.is_finite,
                   f"limit {sp.simplify(finite)}"),
            Result("control: with A_0(0) = 1 the term diverges at q^2 = 0", not diverges.is_finite,
                   "fails, as it must"),
            exact_fails("control: README with + (M - m) r_2", broken - A3.subs(t, 0))]


@functools.cache
def pv_frame():
    """P -> V l+ nu in the W rest frame: the V along -z (rapidity parametrised by u), the l+ at cos theta =
    (1 - w^2)/(1 + w^2) to the W direction (+z); M a symbol with M^2 = Q^2 + 2 Q E_V + m_V^2."""
    Q, mV, u, w, m, M = sp.symbols("Q m_V u w m M", positive=True)
    EV, K = mV * (u + 1 / u) / 2, mV * (u - 1 / u) / 2
    k, p, q = (EV, 0, 0, -K), (Q + EV, 0, 0, -K), (Q, 0, 0, 0)
    c, s = (1 - w**2) / (1 + w**2), 2 * w / (1 + w**2)
    Es, ps = (Q**2 + m**2) / (2 * Q), (Q**2 - m**2) / (2 * Q)
    l, nu = (Es, ps * s, 0, ps * c), (ps, -ps * s, 0, -ps * c)
    pols = [(0, 1, 0, 0), (0, 0, 1, 0), (K / mV, 0, 0, -EV / mV)]
    return dict(Q=Q, mV=mV, m=m, M=M, EV=EV, K=K, k=k, p=p, q=q, c=c, s=s, l=l, nu=nu, pols=pols,
                rel=M**2 - (Q**2 + 2 * Q * EV + mV**2))


def pv_sum(fr, trace, V, A1, A2, A0, sign=1):
    """|M|^2 summed over the V polarisations and the l+ spins: `trace`, the V-A trace of generic_lepton_side summed
    over the l+ spin, for the current i x RB eq. 85 with eps_{0123} = +1 (sign = -1: the other convention)."""
    Jr, Ji, nu, l = generic_lepton_side()[:4]
    M, mV, p, k, q = fr["M"], fr["mV"], fr["p"], fr["k"], fr["q"]
    A3 = RB_86(M, mV, A1, A2)
    q2 = mdot(q, q)
    tot = 0
    for e in fr["pols"]:
        eq = mdot(e, q)
        J_re = [sign * 2 * V / (M + mV) * x for x in eps_raised(e, p, k)]
        J_im = [-((M + mV) * A1 * e[i] - A2 / (M + mV) * eq * (p[i] + k[i]) - 2 * mV / q2 * (A3 - A0) * eq * q[i])
                for i in range(4)]
        tot += trace.xreplace({**dict(zip(Jr, J_re)), **dict(zip(Ji, J_im)), **dict(zip(nu, fr["nu"])),
                               **dict(zip(l, fr["l"]))})
    return tot


def reduced(expr, fr):
    """The numerator of expr, reduced modulo M^2 = Q^2 + 2 Q E_V + m_V^2 (zero if expr vanishes on shell)."""
    return sp.expand(sp.rem(sp.expand(sp.numer(sp.together(expr))), fr["rel"], fr["M"]))


PV_RESULT_PIN = r"\sum_\varepsilon\vert\mathcal M\vert^2_{\rm sum} = 2(q^2 - m_\ell^2)"


def readme_pv_result(c, Q, m, Hp, Hm, H0, M, pV, A0):
    """The README's summed P -> V density, with its H_t."""
    disp = L.display(PV_RESULT_PIN)
    ht = L.to_sympy(L.sides(L.part(disp, "H_t ="))[1], {"M": M, r"p_\mathrm V": pV, "A_0": A0, "q^2": Q**2})
    rhs = L.sides(L.parts(disp)[0])[1]
    toks = {r"\cos\theta_\ell": c, r"\sin^2\theta_\ell": 1 - c**2, "H_+": Hp, "H_-": Hm, "H_0": H0, "H_t": ht,
            "q^2": Q**2, r"m_\ell": m}
    return L.to_sympy(rhs, toks)


@derivation("cd.pv_result")
def check_pv_result():
    """P -> V l+ nu: the V-A |M|^2 of Richman & Burchat's current (eq. 85) is their eq. 113, plus lepton-mass terms."""
    fr = pv_frame()
    _, _, _, l, _, _, _, summed = generic_lepton_side()
    trace = summed(l)                                                              # the l+ spins summed
    V, A1, A2, A0 = sp.symbols("V A_1 A_2 A_0", real=True)
    Q, mV, K, c, mm_, M_ = fr["Q"], fr["mV"], fr["K"], fr["c"], fr["m"], fr["M"]
    pV = Q * K / M_                                                                # M p_V = sqrt(q^2) K
    Hp, Hm, H0 = RB_helicity(M_, mV, Q**2, pV, V, A1, A2)
    readme = readme_pv_result(c, Q, mm_, Hp, Hm, H0, M_, pV, A0)
    total = pv_sum(fr, trace, V, A1, A2, A0)
    with L.patched(r"\left[(1 + \cos\theta_\ell)^2H_+^2 + (1 - \cos\theta_\ell)^2H_-^2",
                   r"\left[(1 - \cos\theta_\ell)^2H_+^2 + (1 + \cos\theta_\ell)^2H_-^2"):
        broken = readme_pv_result(c, Q, mm_, Hp, Hm, H0, M_, pV, A0)
    out = [exact("polarisation sum: the three real vectors give -g + k k/m_V^2",
                 sum((sum(e[i] * e[j] for e in fr["pols"]) - (-(1 if i == 0 else -1) * int(i == j)
                                                              + fr["k"][i] * fr["k"][j] / mV**2))**2
                     for i in range(4) for j in range(4))),
           Result("README's density (H_+-, H_0 of RB eqs. 115-116, its H_t) = the trace summed over the V "
                  "polarisations and l+ spins", reduced(total - readme, fr) == 0,
                  "exact (M^2 = q^2 + 2 sqrt(q^2) E_V + m_V^2)"),
           Result("m_l -> 0: 2 q^2 x RB's eq. 113 for D decays (eta = -1), integrated over theta_V and chi",
                  reduced((total - 2 * Q**2 * RB_113(c, Hp, Hm, H0)).subs(mm_, 0), fr) == 0, "exact"),
           Result("control: README with H_+ and H_- exchanged", reduced(total - broken, fr) != 0, "fails, as it must")]
    # control: the other eps convention (V -> -V) exchanges H_+ and H_-: wrong for D decays (V-A makes H_- dominant)
    pt = {Q: sp.Rational(7, 10), mV: sp.Rational(9, 10), fr["m"]: sp.Rational(1, 10), V: sp.Rational(3, 2), A1: 1,
          A2: sp.Rational(4, 5), A0: sp.Rational(6, 5), sp.Symbol("u", positive=True): 2,
          sp.Symbol("w", positive=True): sp.Rational(1, 3)}
    pt[M_] = sp.sqrt((Q**2 + 2 * Q * fr["EV"] + mV**2).subs(pt))
    bad = (pv_sum(fr, trace, V, A1, A2, A0, sign=-1) - readme).subs(pt)
    out.append(Result("control: V -> -V (eps_{0123} = -1) breaks it", abs(sp.N(bad, 30)) > 1e-10,
                      f"differs by {sp.N(bad, 4)}"))
    return out


# --- the pole states ---

POLES_PIN = r"Richman & Burchat assign $J^P = 1^-$ to $f_+$ and $V$"
MEINEL18_POLES = (r"$f_+$, $f_\perp$, $h_+$, $h_\perp$                         && $1^-$   && $2.010$",
                  r"$f_0$                                                      && $0^+$   && $2.351$",
                  r"$g_+$, $g_\perp$, $\widetilde{h}_+$, $\widetilde{h}_\perp$ && $1^+$   && $2.423$",
                  r"$g_0$                                                      && $0^-$   && $1.870$")
STATE_TEX = {r"\mathrm D^\ast(2010)": "D*(2010)", r"\mathrm D_0^\ast(2300)": "D0*(2300)",
             r"\mathrm D_1(2420)": "D1(2420)", r"\mathrm D": "D", r"\mathrm D_s^\ast": "Ds*",
             r"\mathrm D_{s0}^\ast(2317)": "Ds0*(2317)", r"\mathrm D_{s1}(2460)": "Ds1(2460)", r"\mathrm D_s": "Ds"}


def readme_poles_sentence():
    """({form factor: J^P}, [states for c -> d], [states for c -> s]) of the README."""
    text = L.line(POLES_PIN)
    m = re.search(r"assign \$J\^P = (\d)\^(.)\$ to \$f_\+\$ and \$V\$, \$(\d)\^(.)\$ to \$f_0\$ and \$(\d)\^(.)\$ to "
                  r"\$A_1\$ and \$A_2\$ \(pp\. 73–74\)\. The form factor \$A_0\$ is the time-like part of the axial "
                  r"current\. It takes \$(\d)\^(.)\$, as the baryon form factor \$g_0\$ \(below\) does in Table IV",
                  text)
    g = m.groups()
    jp = {"f_+": g[0] + g[1], "V": g[0] + g[1], "f_0": g[2] + g[3], "A_1": g[4] + g[5], "A_2": g[4] + g[5],
          "A_0": g[6] + g[7]}
    m2 = re.search(r"The code takes (.*) for \$\\mathrm c \\to \\mathrm d\$, and (.*) for \$\\mathrm c \\to "
                   r"\\mathrm s\$", text)
    states = [[STATE_TEX[s] for s in re.findall(r"\$([^$]+)\$", part)] for part in m2.groups()]
    return jp, states[0], states[1]


def poles_results():
    jp, cd, cs = readme_poles_sentence()
    order = ["f_+", "f_0", "A_1", "A_0"]                                          # 1-, 0+, 1+, 0- as the states
    rb = {"f_+": RB_JP["F_1"], "V": RB_JP["V"], "f_0": RB_JP["F_0"], "A_1": RB_JP["A_1"], "A_2": RB_JP["A_2"]}
    out = [Result("README's J^P of f_+, V, f_0, A_1, A_2 = Richman & Burchat's (pp. 73-74)",
                  all(jp[k] == v for k, v in rb.items()), f"{jp}")]
    pdg = {s: re.fullmatch(r"(\S+)\((\d)(.)\)", PDG.POLE_STATES[s][0]).groups() for s in PDG.POLE_STATES}
    ok = (len(cd) == len(cs) == 4
          and all(pdg[s][1] + pdg[s][2] == jp[f] and pdg[s][0] == "1/2" for s, f in zip(cd, order))
          and all(pdg[s][1] + pdg[s][2] == jp[f] and pdg[s][0] == "0" for s, f in zip(cs, order)))
    out.append(Result("the states have these J^P (PDG 2026, p. 1 of each listing), isospin 1/2 (c qbar) for c -> d "
                      "and 0 (c sbar) for c -> s", ok, f"c -> d {cd}, c -> s {cs}"))
    return out, jp


@derivation("cd.poles")
def check_poles():
    """The poles have the J^P of their part of the current (RB pp. 73-74, Meinel 2018 Table IV); PDG's states match."""
    out, jp = poles_results()
    out.append(in_source("Meinel 2018, Table IV: J^P 1- (f_+, f_perp), 0+ (f_0), 1+ (g_+, g_perp), 0- (g_0)",
                         MEINEL18, *MEINEL18_POLES))
    out.append(Result("A_0 takes 0-, as g_0 (the time-like axial form factor) in Meinel's Table IV", jp["A_0"] == "0-",
                      jp["A_0"]))
    with L.patched(r"$0^+$ to $f_0$ and", r"$1^+$ to $f_0$ and"):
        broken, _ = poles_results()
    out.append(control_fails("README with 1+ for f_0", not all(r.ok for r in broken)))
    return out


# --- Lambda_c -> B l+ nu: Meinel's current regrouped, the z expansion, the bilinear form ---

MEINEL_CURRENT = (r"f_0(q^2)\: (m_{\Lambda_c}-m_N)\frac{q^\mu}{q^2}",
                  r"+ f_+(q^2) \frac{m_{\Lambda_c}+m_N}{s_+}\left( p^\mu + p^{\prime \mu} - (m_{\Lambda_c}^2-m_N^2)"
                  r"\frac{q^\mu}{q^2}  \right)",
                  r"+ f_\perp(q^2) \left(\gamma^\mu - \frac{2m_N}{s_+} p^\mu - \frac{2 m_{\Lambda_c}}{s_+} "
                  r"p^{\prime \mu} \right)",
                  r"-\overline{u}_N(p^\prime,s^\prime) \:\gamma_5 \bigg[ g_0(q^2)\: (m_{\Lambda_c}+m_N)"
                  r"\frac{q^\mu}{q^2}",
                  r"+ g_+(q^2)\frac{m_{\Lambda_c}-m_N}{s_-}\left( p^\mu + p^{\prime \mu} - (m_{\Lambda_c}^2-m_N^2)"
                  r"\frac{q^\mu}{q^2}  \right)",
                  r"+ g_\perp(q^2) \left(\gamma^\mu + \frac{2m_N}{s_-} p^\mu - \frac{2 m_{\Lambda_c}}{s_-} "
                  r"p^{\prime \mu} \right)",
                  r"$s_\pm =(m_{\Lambda_c} \pm m_N)^2-q^2$")


def meinel_current(M, m, q2, f, sp_, sm):
    """Meinel 2018, eqs. 1-2 as transcribed: the vector bracket and the axial bracket (<|A|> = -u-bar g5 [...] u) as
    coefficients of (gamma^mu, p^mu, p'^mu), with q = p - p'."""
    vec = (f["fperp"],
           f["f0"] * (M - m) / q2 + f["fplus"] * (M + m) / sp_ * (1 - (M**2 - m**2) / q2) - f["fperp"] * 2 * m / sp_,
           -f["f0"] * (M - m) / q2 + f["fplus"] * (M + m) / sp_ * (1 + (M**2 - m**2) / q2) - f["fperp"] * 2 * M / sp_)
    ax = (f["gperp"],
          f["g0"] * (M + m) / q2 + f["gplus"] * (M - m) / sm * (1 - (M**2 - m**2) / q2) + f["gperp"] * 2 * m / sm,
          -f["g0"] * (M + m) / q2 + f["gplus"] * (M - m) / sm * (1 + (M**2 - m**2) / q2) - f["gperp"] * 2 * M / sm)
    return vec, ax


BCURRENT_PIN = r"J^\mu = \bar u_\mathrm B\left[F\gamma^\mu + aP^\mu + bq^\mu"
BCOEFFS = ("F", "G", "a", "b", "c", "d")


def readme_bcurrent(M, m, q2, f):
    """The README's regrouped current: the coefficients of (gamma, p, p') of its vector and axial parts, with
    F, G, a, b, c, d and s_+- from its displays."""
    disp = L.display(BCURRENT_PIN)
    lhs, rhs = L.sides(L.parts(disp)[0])
    inner = L.between(rhs, r"\left[", r"\right]u_")
    vec_tex, ax_tex = inner.split(r"\gamma_5\left(")
    ax_tex = ax_tex.rsplit(r"\right)", 1)[0]
    vec_tex = vec_tex.strip().rstrip("+").strip()
    G_, P_, Q_ = sp.symbols("Gam Pm Qm")
    co = {k: sp.Symbol("c_" + k) for k in BCOEFFS}
    toks = {r"\gamma^\mu": G_, r"P^\mu": P_, r"q^\mu": Q_, **co}
    vec, ax = L.to_sympy(vec_tex, toks), L.to_sympy(ax_tex, toks)
    p, pp = sp.symbols("p pp")
    PP = L.to_sympy(L.sides(L.part(disp, "P ="))[1], {"p'": pp, "p": p})
    QQ = L.to_sympy(L.sides(L.part(disp, "q ="))[1], {"p'": pp, "p": p})
    spm = L.sides(L.part(disp, r"s_\pm ="))[1]
    s_plus = L.to_sympy(spm, {"M": M, r"m_\mathrm B": m, "q^2": q2}, sign=1)
    s_minus = L.to_sympy(spm, {"M": M, r"m_\mathrm B": m, "q^2": q2}, sign=-1)
    coeffs = readme_bcoeffs(M, m, q2, f, s_plus, s_minus)

    def basis(e):
        e = sp.expand(e.subs({P_: PP, Q_: QQ}).subs(coeffs))
        return tuple(sp.expand(e.coeff(x)) for x in (G_, p, pp))
    return basis(vec), basis(ax), s_plus, s_minus


def readme_bcoeffs(M, m, q2, f, s_plus, s_minus):
    """{F, G, a, b, c, d} of the README's display of the coefficients (b after a, d after c)."""
    disp = L.display(r"F = f_\perp,\qquad G = g_\perp")
    toks = {"M": M, r"m_\mathrm B": m, "q^2": q2, "f_+": f["fplus"], r"f_\perp": f["fperp"], "f_0": f["f0"],
            "g_+": f["gplus"], r"g_\perp": f["gperp"], "g_0": f["g0"], "s_+": s_plus, "s_-": s_minus}
    out = {}
    for k in BCOEFFS:
        lhs, rhs = L.sides(L.part(disp, k + " ="))
        out[k] = L.to_sympy(rhs, {**toks, **{c: out[c] for c in out}})
    return {sp.Symbol("c_" + k): v for k, v in out.items()}


@derivation("cd.bcurrent")
def check_bcurrent():
    """The README's regrouped current, with F, G, a, b, c, d, is Meinel's 2018 eqs. 1-2 (the V - A combination)."""
    M, m, q2 = sp.symbols("M m_B q2", positive=True)
    f = {k: sp.Symbol(k) for k in ("fplus", "fperp", "f0", "gplus", "gperp", "g0")}
    vec, ax, s_plus, s_minus = readme_bcurrent(M, m, q2, f)
    mv, ma = meinel_current(M, m, q2, f, s_plus, s_minus)
    with L.patched(r"b = (M - m_\mathrm B)\left[\frac{f_0 - f_+}{q^2} - \frac{a}{M + m_\mathrm B}\right]",
                   r"b = (M - m_\mathrm B)\left[\frac{f_0 - f_+}{q^2} + \frac{a}{M + m_\mathrm B}\right]"):
        vec_b, _, _, _ = readme_bcurrent(M, m, q2, f)
    return [in_source("Meinel 2018, eqs. 1-2 and s_+- as transcribed", MEINEL18, *MEINEL_CURRENT),
            exact("s_+ - s_- = README's (M +- m_B)^2 - q^2 differ by 4 M m_B", s_plus - s_minus - 4 * M * m),
            exact("vector part: README (F, a + b, a - b) on (gamma, p, p') = Meinel's eq. 1",
                  sum((a - b)**2 for a, b in zip(vec, mv))),
            exact("axial part: J = V - A = u-bar[... + g5 (eq. 2's bracket)]u: README (G, c + d, c - d) = Meinel's",
                  sum((a - b)**2 for a, b in zip(ax, ma))),
            exact_fails("control: README b with + a/(M + m_B)", sum((a - b)**2 for a, b in zip(vec_b, mv)))]


ZEXP_PIN = r"f(q^2) = \frac{a_0^f + a_1^fz + a_2^fz^2}{1 - q^2/(m_{\rm pole}^f)^2}"
MEINEL_ZEXP = (r"f(q^2) = \frac{1}{1-q^2/(m_{\rm pole}^f)^2} \sum_{n=0}^{n_{\rm max}} a_n^f [z(q^2)]^n",)
MEINEL_Z = r"z(q^2) = \frac{\sqrt{t_+-q^2}-\sqrt{t_+-t_0}}{\sqrt{t_+-q^2}+\sqrt{t_+-t_0}}"
MEINEL17_TEXT = (r"with $t_0 = q^2_{\rm max} = (m_{\Lambda_c} - m_{\Lambda})^2$ and $t_+ = (m_D + m_K)^2$",
                 r"but with $n_{\rm max}=2$ instead of $n_{\rm max}=1$",
                 r"$m_{\rm pole}^{f_+,f_\perp}=2.112$ GeV,", r"$m_{\rm pole}^{f_0}=2.318$ GeV, "
                 r"$m_{\rm pole}^{g_+,g_\perp}=2.460$ GeV,", r"$m_{\rm pole}^{g_0}=1.968$ GeV")
MEINEL18_TEXT = (r"t_+ = (m_D + m_\pi)^2. \label{eq:tplus}", r"t_0 = q^2_{\rm max} = (m_{\Lambda_c} - m_N)^2",
                 r"with $n_{\rm max}=2$ and $n_{\rm max}=3$, respectively",
                 r"g_\perp(q^2_{\rm max}) &=& g_+(q^2_{\rm max}), \label{eq:FFC3}")
FIT_NAMES = {r"f_\perp": "fperp", "f_+": "fplus", "f_0": "f0", r"g_\perp,g_+": "gpp", r"g_\perp": "gperp",
             "g_+": "gplus", "g_0": "g0"}


def readme_zexp(q2, a, mpole, tp, M, m):
    """f(q^2) and z of the README's z expansion, with its t_0."""
    disp = L.display(ZEXP_PIN)
    t0 = L.to_sympy(L.sides(L.part(disp, "t_0 ="))[1], {"M": M, r"m_\mathrm B": m})
    z = L.to_sympy(L.sides(L.part(disp, "z ="))[1], {"t_+": tp, "t_0": t0, "q^2": q2})
    f = L.to_sympy(L.sides(L.parts(disp)[0])[1], {"a_0^f": a[0], "a_1^f": a[1], "a_2^f": a[2], "z": z, "q^2": q2,
                                                  r"m_{\rm pole}^f": mpole})
    return f, z, t0


def fit_table(source):
    """{name: nominal value} of the fit table of a TeX source (a_n^f rows; the shared a_0^{g_perp,g_+} as gpp)."""
    tex = latex.tex_source(source)
    if tex is None:
        return None
    table = tex[tex.index(r"\label{tab:fitresults}") - 4000:tex.index(r"\label{tab:fitresults}")]
    out = {}
    for n, name, value in re.findall(r"\$a_(\d)\^\{([^$]*)\}\$\s*&\s*&\s*\$(?:\\wm)?\s*([-\d.]+)\\pm", table):
        if name in FIT_NAMES:
            out[f"a{n}_{FIT_NAMES[name]}"] = value
    return out


def supplemental(source, member):
    text = latex.tar_member(source, member)
    return None if text is None else dict(line.split() for line in text.strip().splitlines())


@derivation("cd.zexp")
def check_zexp():
    """The README's z expansion is Meinel's (n_max = 2) with his t_0 and t_+; the fits' digits round to his tables."""
    q2, tp, M, m, mpole = sp.symbols("q2 t_p M m_B m_pole", positive=True)
    a = sp.symbols("a0:3", real=True)
    f, z, t0 = readme_zexp(q2, a, mpole, tp, M, m)
    zm = L.to_sympy(MEINEL_Z.split("=", 1)[1], {"t_+": tp, "t_0": t0, "q": sp.sqrt(q2)})
    fm = sum(a[n] * zm**n for n in range(3)) / (1 - q2 / mpole**2)              # eq. 10 with n_max = 2
    with L.patched(r"z = \frac{\sqrt{t_+ - q^2} - \sqrt{t_+ - t_0}}",
                   r"z = \frac{\sqrt{t_+ - q^2} + \sqrt{t_+ - t_0}}"):
        _, z_b, _ = readme_zexp(q2, a, mpole, tp, M, m)
    # u = sqrt(t_+ - q^2), v = sqrt(t_+ - t_0): rational in u, v
    u, v = sp.symbols("u v", positive=True)
    rational = {q2: tp - u**2, M: m + sp.sqrt(tp - v**2)}
    out = [in_source("Meinel 2018, eqs. 10-11 (and 2017 eq. 3) as transcribed", MEINEL18, *MEINEL_ZEXP, MEINEL_Z),
           in_source("Meinel 2017: t_0, t_+ = (m_D + m_K)^2, n_max = 2 of the nominal fit, the D_s pole masses",
                     MEINEL17, *MEINEL17_TEXT),
           in_source("Meinel 2018: t_+ = (m_D + m_pi)^2 (eq. 12), t_0 = (m_Lc - m_N)^2 (eq. 13), n_max = 2, eq. 7",
                     MEINEL18, *MEINEL18_TEXT),
           exact("README z = Meinel's z(q^2) (u = sqrt(t_+ - q^2), v = sqrt(t_+ - t_0))",
                 sp.simplify((z - zm).subs(rational))),
           exact("README f(q^2) = eq. 10 with n_max = 2", sp.simplify((f - fm).subs(rational))),
           exact_fails("control: README z with + in the numerator", sp.simplify((z_b - zm).subs(rational)))]
    tplus = {"Lambda": L.inline(r"(m_\mathrm D + m_\mathrm K)^2"), "n": L.inline(r"(m_\mathrm D + m_\pi)^2")}
    out.append(Result("README's t_+: (m_D + m_K)^2 for Lambda (Meinel 2017), (m_D + m_pi)^2 for n (2018 eq. 12)",
                      "$t_+$ is $(m_\\mathrm D + m_\\mathrm K)^2$ for $\\Lambda$ and $(m_\\mathrm D + m_\\pi)^2$ for "
                      "$\\mathrm n$" in L.line("$t_+$ is $"), f"{tplus}"))
    # the supplemental files' digits round to the tables' nominal values
    for source, member, label in ((MEINEL17, "anc/LambdacLambda_results.dat", "2017 Table IV"),
                                  (MEINEL18, "anc/Lambdacproton_results.dat", "2018 Table V")):
        table, dat = fit_table(source), supplemental(source, member)
        if table is None or dat is None:
            out.append(Result(f"Meinel {label}: supplemental file", True, f"not re-checked: {source} absent"))
            continue
        bad = [k for k, val in table.items()
               if abs(sp.Rational(dat[k]) - sp.Rational(val))
               > sp.Rational(1, 2) * sp.Integer(10) ** -len(val.split(".")[1])]
        out.append(Result(f"Meinel {label}: the {len(table)} nominal z-expansion parameters of {member.split('/')[1]} "
                          "round to the table", not bad and len(table) == 17, f"differ: {bad}" if bad else "all"))
    return out


ZW_PIN = r"c = \frac{(M - m_\mathrm B)\thinspace w\left["


def readme_zw(sym):
    disp = L.display(ZW_PIN)
    sides_w = L.sides(L.part(disp, "w ="))
    toks = {"z": sym["z"], "s_-": sym["sm"], "t_+": sym["tp"], "t_0": sym["t0"], "q^2": sym["q2"]}
    w_forms = [L.to_sympy(s_, toks) for s_ in sides_w[1:]]
    c = L.to_sympy(L.sides(L.parts(disp)[0])[1],
                   {"M": sym["M"], r"m_\mathrm B": sym["m"], "w": sym["w"], "z": sym["z"], "q^2": sym["q2"],
                    "a_1^{g_+}": sym["a1p"], r"a_1^{g_\perp}": sym["a1q"], "a_2^{g_+}": sym["a2p"],
                    r"a_2^{g_\perp}": sym["a2q"], r"m_{\rm pole}^{g_+}": sym["mg"]})
    return w_forms, c


@derivation("cd.zw")
def check_zw():
    """z = s_- w, and with g_+, g_perp sharing a_0 and the pole the README's c is (M - m)(g_+ - g_perp)/s_- (no 0/0)."""
    u, v, m, a0 = sp.symbols("u v m_B a0", positive=True)
    tp, mg = sp.symbols("t_p m_g", positive=True)
    a1p, a1q, a2p, a2q = sp.symbols("a1p a1q a2p a2q", real=True)
    q2, t0 = tp - u**2, tp - v**2                                                    # u = sqrt(t_+ - q^2), ...
    M = m + sp.sqrt(t0)                                                              # t_0 = (M - m)^2
    z = (u - v) / (u + v)
    sm = (M - m)**2 - q2                                                             # s_- (cd.bcurrent)
    sym = dict(z=z, sm=sm, tp=tp, t0=t0, q2=q2, M=M, m=m, w=sp.Symbol("w"), a1p=a1p, a1q=a1q, a2p=a2p, a2q=a2q, mg=mg)
    w_forms, c = readme_zw(sym)
    pole = 1 / (1 - q2 / mg**2)
    gp = (a0 + a1p * z + a2p * z**2) * pole
    gq = (a0 + a1q * z + a2q * z**2) * pole
    c_def = (M - m) * (gp - gq) / sm
    with L.patched(r"\left[a_1^{g_+} - a_1^{g_\perp} +", r"\left[a_1^{g_+} + a_1^{g_\perp} +"):
        _, c_b = readme_zw(sym)
    return [exact("w = z/s_- = 1/(sqrt(t_+ - q^2) + sqrt(t_+ - t_0))^2 (t_0 = (M - m_B)^2)",
                  (w_forms[0] - w_forms[1])**2),
            exact("README c (w the README's) = (M - m_B)(g_+ - g_perp)/s_- of the z expansions sharing a_0 and "
                  "the pole",
                  c.subs(sym["w"], w_forms[1]) - c_def),
            exact("g_+ = g_perp at q^2 = t_0 (Meinel 2018 eq. 7): z = 0 there", (gp - gq).subs(u, v)),
            exact_fails("control: README with a_1^{g+} + a_1^{gperp}", c_b.subs(sym["w"], w_forms[1]) - c_def)]


BILINEAR_PIN = r"\sum\vert\mathcal M\vert^2 = \tfrac12 G_F^2\vert V_{cq}\vert^2\thinspace W"
K_LINES = (r"K_{qq} = -32\left[", r"K_S = -64\left[", r"+ (aP + bq)^2\right)")   # one pin per line


def readme_W(dots, co, M, m, q2):
    """The README's W for the dot products `dots` ({README token: value}) and the coefficients co (F, G, a, b, c, d),
    with its K's and (aP + bq)^2 rule; and the prefactor of sum |M|^2."""
    disp = L.display(BILINEAR_PIN)
    pre = L.to_sympy(L.sides(L.parts(disp)[0])[1], {"G_F": 1, r"\vert V_{cq}\vert": 1, "W": 1})
    toks = {"M": M, r"m_\mathrm B": m, "s_+": (M + m)**2 - q2, "s_-": (M - m)**2 - q2, **co}
    P2 = 2 * (M**2 + m**2) - q2                                                      # README's P^2, P.q (cd.onshell)
    Pq = M**2 - m**2
    sq = {"(aP + bq)^2": co["a"]**2 * P2 + 2 * co["a"] * co["b"] * Pq + co["b"]**2 * q2,
          "(cP + dq)^2": co["c"]**2 * P2 + 2 * co["c"] * co["d"] * Pq + co["d"]**2 * q2}
    K = {}
    for pin in K_LINES:
        for part in L.parts(L.display(pin)):
            lhs, rhs = L.sides(part)
            K[lhs] = L.to_sympy(rhs, {**toks, **sq})
    w_rhs = L.sides(L.part(disp, "W ="))[1]
    return pre, L.to_sympy(w_rhs, {**dots, **{k: v for k, v in K.items()}})


def W_dot_tokens(P, q, l, k):
    return {r"p_\ell\cdot p_\nu": mdot(l, k), r"(P\cdot p_\ell)": mdot(P, l), r"(P\cdot p_\nu)": mdot(P, k),
            r"(q\cdot p_\ell)": mdot(q, l), r"(q\cdot p_\nu)": mdot(q, k)}


@functools.cache
def baryon_trace_point(seed):
    """At rational kinematics (p at rest, p' on shell, any lepton momenta): the trace H^{mu nu} L_{mu nu} of the
    current F g + a P + b q + g5 (G g + c P + d q) with symbolic F..d, and the README's W there."""
    gam, g5, one = dirac()
    gen = rng(seed)
    M = sp.Rational(gen.randint(20, 30), 10)
    m = sp.Rational(gen.randint(9, 13), 10)
    v = sp.Rational(gen.randint(11, 19), 10)
    w = sp.Rational(gen.randint(1, 9), 10)
    Ep, Pp = m * (v + 1 / v) / 2, m * (v - 1 / v) / 2
    p, pp = (M, 0, 0, 0), (Ep, Pp * 2 * w / (1 + w**2), 0, Pp * (1 - w**2) / (1 + w**2))
    l = tuple(sp.Rational(gen.randint(-9, 9), 10) for _ in range(4))
    k = tuple(sp.Rational(gen.randint(-9, 9), 10) for _ in range(4))
    P = [a + b for a, b in zip(p, pp)]
    q = [a - b for a, b in zip(p, pp)]
    co = {s: sp.Symbol(s, real=True) for s in BCOEFFS}
    g = [1, -1, -1, -1]
    Gam = [co["F"] * gam[i] + (co["a"] * P[i] + co["b"] * q[i]) * one
           + g5 * (co["G"] * gam[i] + (co["c"] * P[i] + co["d"] * q[i]) * one) for i in range(4)]
    Gbar = [gam[0] * G.H * gam[0] for G in Gam]
    Hm = [[((slash(pp) + m * one) * Gam[i] * (slash(p) + M * one) * Gbar[j]).trace() for j in range(4)]
          for i in range(4)]
    Lm = [[g[i] * g[j] * (slash(k) * gam[i] * (one - g5) * slash(l) * (one + g5) * gam[j]).trace() for j in range(4)]
          for i in range(4)]
    trace = sp.expand(sum(Hm[i][j] * Lm[i][j] for i in range(4) for j in range(4)))
    q2 = mdot(q, q)
    return trace, co, (M, m, q2, P, q, l, k, p, pp)


@derivation("cd.bilinear")
def check_bilinear():
    """Summed over the spins, Meinel's current with the V-A lepton tensor gives the README's W (explicit traces)."""
    out = []
    for seed in (1, 2, 3):
        trace, co, (M, m, q2, P, q, l, k, p, pp) = baryon_trace_point(seed)
        pre, W = readme_W(W_dot_tokens(P, q, l, k), co, M, m, q2)
        out.append(exact(f"point {seed}: H^{{mu nu}} L_{{mu nu}} = README W (any F, G, a, b, c, d; on-shell baryons)",
                         sp.expand(W) - trace))
    trace, co, (M, m, q2, P, q, l, k, p, pp) = baryon_trace_point(1)
    out.append(Result("the trace is real: no eps tensor survives", sp.im(trace) == 0 and not trace.has(sp.I),
                      "as written"))
    with L.patched(r"K_A = -128\thinspace FG", r"K_A = 128\thinspace FG"):
        _, W_b = readme_W(W_dot_tokens(P, q, l, k), co, M, m, q2)
    out.append(exact_fails("control: README with K_A = +128 FG", sp.expand(W_b) - trace))
    with L.patched(r"K_{PP} = 32\left[", r"K_{PP} = 16\left["):
        _, W_b = readme_W(W_dot_tokens(P, q, l, k), co, M, m, q2)
    out.append(exact_fails("control: README with K_PP = 16[...]", sp.expand(W_b) - trace))
    out.append(Result("README's prefactor: sum |M|^2 = (G_F^2 |V_cq|^2 / 2) W, H_eff = G_F/sqrt(2) V_cq [V - A][V - A]",
                      pre == sp.Rational(1, 2), f"{pre}"))
    return out


@derivation("cd.onshell")
def check_onshell():
    """P^2 and P.q of on-shell baryons, as the README states them for (aP + bq)^2."""
    M, m = sp.symbols("M m_B", positive=True)
    v = sp.Symbol("v", positive=True)
    Ep, Pp = m * (v + 1 / v) / 2, m * (v - 1 / v) / 2
    p, pp = (M, 0, 0, 0), (Ep, 0, 0, Pp)
    P = [a + b for a, b in zip(p, pp)]
    q = [a - b for a, b in zip(p, pp)]
    q2 = mdot(q, q)

    def readme():
        line = L.line("is $a^2P^2 + 2ab")
        P2 = L.sides(L.between(line, "with $P^2", "$"))[1]
        Pq = L.sides(L.between(line, r"and $P\cdot q", "$"))[1]
        return (L.to_sympy(P2, {"M": M, r"m_\mathrm B": m, "q^2": q2}), L.to_sympy(Pq, {"M": M, r"m_\mathrm B": m}),
                L.between(line, "$(aP + bq)^2$ is $", "$"))
    P2, Pq, sq = readme()
    with L.patched(r"$P^2 = 2(M^2 + m_\mathrm B^2) - q^2$", r"$P^2 = 2(M^2 + m_\mathrm B^2) + q^2$"):
        P2_b = readme()[0]
    return [exact("P^2 = (p + p')^2 = README's, p^2 = M^2, p'^2 = m_B^2", P2 - mdot(P, P)),
            exact("P.q = (p + p').(p - p') = README's", Pq - mdot(P, q)),
            Result("(aP + bq)^2 expands as the README writes it", sq == r"a^2P^2 + 2ab\thinspace P\cdot q + b^2q^2",
                   "as written"),
            exact_fails("control: README with + q^2", P2_b - mdot(P, P))]


@derivation("cd.free_quark")
def check_free_quark():
    """With all form factors 1 the current is the free-quark V - A current, and the README's W is 256 (p.l)(p'.nu)."""
    M, m, q2 = sp.symbols("M m_B q2", positive=True)
    f = {k: 1 for k in ("fplus", "fperp", "f0", "gplus", "gperp", "g0")}
    s_plus, s_minus = (M + m)**2 - q2, (M - m)**2 - q2
    co = {str(k)[2:]: v for k, v in readme_bcoeffs(M, m, q2, f, s_plus, s_minus).items()}
    p, pp, l, k = (sp.symbols(f"{v}0:4", real=True) for v in ("p", "pp", "l", "k"))
    P = [a + b for a, b in zip(p, pp)]
    q = [a - b for a, b in zip(p, pp)]
    _, W = readme_W(W_dot_tokens(P, q, l, k), co, M, m, mdot(q, q))
    def readme_free():
        lhs, rhs = L.sides(L.between(L.line("the free-quark current, $W ="), "the free-quark current, $", "$"))
        return L.to_sympy(rhs, {r"(p\cdot p_\ell)": mdot(p, l), r"(p'\cdot p_\nu)": mdot(pp, k)})
    free = readme_free()
    with L.patched(r"$W = 256\thinspace(p\cdot p_\ell)", r"$W = 128\thinspace(p\cdot p_\ell)"):
        free_b = readme_free()
    return [Result("all form factors 1: F = G = 1, a = b = c = d = 0 by the README's coefficients",
                   co == {"F": 1, "G": 1, "a": 0, "b": 0, "c": 0, "d": 0}, f"{co}"),
            exact("README W = README's 256 (p.p_l)(p'.p_nu) for any momenta", sp.expand(W - free)),
            exact_fails("control: README with 128 for 256", sp.expand(W - free_b))]


MEINEL_RATE = (   # eq. 17, one snippet per term
    r"\frac{G_F^2 |V_{cd}|^2 \sqrt{s_+s_-} }{768 \pi ^3 m_{\Lambda_c}^3 } \left(1-\frac{m_\ell^2}{q^2}\right)^2",
    r"\times\Bigg\{4 \left(m_\ell^2+2 q^2\right) \left( s_+ \left[  g_\perp  \right]^2 + s_- \left[f_\perp\right]^2"
    r" \right)",
    r"+2 \frac{m_\ell^2+2 q^2}{q^2} \left(s_+ \left[\left(m_{\Lambda_c}-m_N\right) g_+ \right]^2+s_- "
    r"\left[\left(m_{\Lambda_c}+m_N\right)f_+ \right]^2\right)",
    r"+\frac{6 m_\ell^2}{q^2} \left(s_+ \left[ \left(m_{\Lambda_c}-m_N\right) f_0 \right]^2 + s_- "
    r"\left[ \left(m_{\Lambda_c}+m_N\right) g_0  \right]^2\right)\Bigg\}.")


def meinel_rate(M, m, ml, q2, f):
    """Meinel 2018, eq. 17, as transcribed, over G_F^2 |V|^2."""
    sp_, sm = (M + m)**2 - q2, (M - m)**2 - q2
    return (mp.sqrt(sp_ * sm) / (768 * mp.pi**3 * M**3) * (1 - ml**2 / q2)**2
            * (4 * (ml**2 + 2 * q2) * (sp_ * f["gperp"]**2 + sm * f["fperp"]**2)
               + 2 * (ml**2 + 2 * q2) / q2 * (sp_ * ((M - m) * f["gplus"])**2 + sm * ((M + m) * f["fplus"])**2)
               + 6 * ml**2 / q2 * (sp_ * ((M - m) * f["f0"])**2 + sm * ((M + m) * f["g0"])**2)))


def readme_rate(M, m, ml, q2, f):
    """dGamma/dq^2 over G_F^2 |V|^2 from the README's W: (1/(512 pi^3 M^2)) int dE_l W (three-body phase space,
    d^2 Gamma/(dE_B dE_l) = |M|^2-bar/(64 pi^3 M), |M|^2-bar = W/4 and dq^2 = 2M dE_B), at mpmath precision."""
    E_B = (M**2 + m**2 - q2) / (2 * M)
    pB = mp.sqrt(E_B**2 - m**2)
    Q = mp.sqrt(q2)
    El_s, pl_s = (q2 + ml**2) / (2 * Q), (q2 - ml**2) / (2 * Q)
    lo, hi = ((M - E_B) * El_s - pB * pl_s) / Q, ((M - E_B) * El_s + pB * pl_s) / Q
    Ms, ms, q2s = sp.symbols("M m_B q2", positive=True)
    fs = {k: sp.Symbol(k) for k in f}
    co = {str(k)[2:]: v for k, v in readme_bcoeffs(Ms, ms, q2s, fs, (Ms + ms)**2 - q2s, (Ms - ms)**2 - q2s).items()}
    dots = sp.symbols("d_lk d_Pl d_Pk d_ql d_qk")
    tok = dict(zip((r"p_\ell\cdot p_\nu", r"(P\cdot p_\ell)", r"(P\cdot p_\nu)", r"(q\cdot p_\ell)", r"(q\cdot p_\nu)"),
                   dots))
    _, W = readme_W(tok, co, Ms, ms, q2s)
    Wf = sp.lambdify((*dots, Ms, ms, q2s, *fs.values()), W, "mpmath")

    def integrand(El):
        Enu = M - E_B - El
        pl = mp.sqrt(El**2 - ml**2)
        c = (Enu**2 - pB**2 - pl**2) / (2 * pB * pl)
        s = mp.sqrt(max(1 - c**2, 0))
        p, pp = (M, 0, 0, 0), (E_B, 0, 0, pB)
        lv = (El, pl * s, 0, pl * c)
        kv = tuple(a - b - c_ for a, b, c_ in zip(p, pp, lv))
        P = [a + b for a, b in zip(p, pp)]
        q = [a - b for a, b in zip(p, pp)]
        d = (mdot(lv, kv), mdot(P, lv), mdot(P, kv), mdot(q, lv), mdot(q, kv))
        return Wf(*d, M, m, q2, *f.values())
    return mp.quad(integrand, [lo, hi]) / (512 * mp.pi**3 * M**2)


@derivation("cd.lc_rate")
def check_lc_rate():
    """Integrated over the lepton energy, the README's W gives Meinel's dGamma/dq^2 (2018 eq. 17), 30 digits."""
    gen = rng(5)
    worst, out = mp.mpf(0), []
    with mp.workdps(40):
        for _ in range(3):
            M, m, ml = mp.mpf("2.28646"), mp.mpf("0.939565") + gen.random() / 5, mp.mpf("0.1056584") * gen.random()
            q2 = ml**2 + (mp.mpf(gen.random()) * 0.9 + 0.05) * ((M - m)**2 - ml**2)
            f = {k: mp.mpf(gen.uniform(-1.5, 1.5)) for k in ("fplus", "fperp", "f0", "gplus", "gperp", "g0")}
            a, b = readme_rate(M, m, ml, q2, f), meinel_rate(M, m, ml, q2, f)
            worst = max(worst, abs(a / b - 1))
    out = [in_source("Meinel 2018, eq. 17 as transcribed", MEINEL18, *MEINEL_RATE),
           Result("README W integrated over E_l (fixed q^2) = Meinel's eq. 17 (any form factors, m_l, m_B)",
                  worst < mp.mpf(10)**-30, f"max rel. difference {mp.nstr(worst, 3)} at 3 points, 40 digits"),
           Result("README: \"W gives Meinel's differential rate dGamma/dq^2 (2018 eq. 17)\"",
                  "$W$ gives Meinel's differential rate $d\\Gamma/dq^2$ (2018 eq. 17)" in
                  L.line("Integrated over the lepton energy at fixed"), "as written")]
    with mp.workdps(40):
        with L.patched(r"K_g = -16\left[4Mm_\mathrm B", r"K_g = -8\left[4Mm_\mathrm B"):
            M, m, ml, q2 = mp.mpf("2.28646"), mp.mpf("1.115683"), mp.mpf("0.1056584"), mp.mpf("0.8")
            f = {k: mp.mpf(v) for k, v in zip(("fplus", "fperp", "f0", "gplus", "gperp", "g0"), (0.7, 1.1, 0.6, 0.5,
                                                                                              0.6, 0.55))}
            bad = abs(readme_rate(M, m, ml, q2, f) / meinel_rate(M, m, ml, q2, f) - 1)
    out.append(Result("control: README with K_g = -8[...]", bad > 1e-6, f"differs by {mp.nstr(bad, 3)}"))
    return out


# --- energy distributions at rest ---

def readme_two_body(M, m, mb):
    lhs, rhs = L.sides(L.between(L.line("has the single energy $"), "has the single energy $", "$"))
    return L.to_sympy(rhs, {"M": M, "m": m, "m_b": mb})


@derivation("cd.two_body")
def check_two_body():
    """Two-body decay at rest: E* = (M^2 + m^2 - m_b^2)/(2M) gives both products the same momentum."""
    M, m, mb = sp.symbols("M m m_b", positive=True)
    E = readme_two_body(M, m, mb)
    with L.patched(r"$E^\ast = (M^2 + m^2 - m_b^2)/(2M)$", r"$E^\ast = (M^2 + m^2 + m_b^2)/(2M)$"):
        E_b = readme_two_body(M, m, mb)
    return [exact("E* + E_b* = M and p*^2 = E*^2 - m^2 = E_b*^2 - m_b^2", (E**2 - m**2) - ((M - E)**2 - mb**2)),
            exact_fails("control: README with + m_b^2", (E_b**2 - m**2) - ((M - E_b)**2 - mb**2))]


def readme_kallen(a, b, c):
    lhs, rhs = L.sides(L.between(L.line("and $\\lambda(a, b, c) ="), "and $", "$"))
    return L.to_sympy(rhs, {"a": a, "b": b, "c": c})


@derivation("cd.kallen")
def check_kallen():
    """The Kallen function gives the two-body momentum: lambda^(1/2)(M^2, m1^2, m2^2)/(2M) = p*."""
    M, m1, m2 = sp.symbols("M m_1 m_2", positive=True)
    lam = readme_kallen(M**2, m1**2, m2**2)
    E = (M**2 + m1**2 - m2**2) / (2 * M)
    with L.patched(r"- 2bc - 2ca$", r"- 2bc + 2ca$"):
        lam_b = readme_kallen(M**2, m1**2, m2**2)
    return [exact("lambda(M^2, m1^2, m2^2)/(4 M^2) = E*^2 - m1^2 (cd.two_body)", lam / (4 * M**2) - (E**2 - m1**2)),
            exact_fails("control: README with + 2ca", lam_b / (4 * M**2) - (E**2 - m1**2))]


PS_PIN = r"g(E) \propto p\thinspace R_{n-1}(m_{\rm rec})"


def readme_phase_space(M, m, E):
    """m_rec^2 and R_2 of the README's display, and its recursion as written."""
    disp = L.display(PS_PIN)
    mrec2 = L.to_sympy(L.sides(L.part(disp, r"m_{\rm rec}^2 ="))[1], {"M": M, "m": m, "E": E})
    mm, m1, m2 = sp.symbols("mm m1 m2", positive=True)
    lam = lambda a, b, c: readme_kallen(a, b, c)
    R2 = L.to_sympy(L.sides(L.part(disp, "R_2(m) ="))[1],
                    {r"\lambda^{1/2}(m^2, m_1^2, m_2^2)": sp.sqrt(lam(mm**2, m1**2, m2**2)), "m": mm})
    rec = L.sides(L.part(disp, "R_k(m) ="))[1]
    return mrec2, sp.Lambda((mm, m1, m2), R2), rec


@derivation("cd.phase_space")
def check_phase_space():
    """The README's recursion is n-body phase space: R_3 = 4 x the Dalitz area, p R_2(m_rec) = the Dalitz line."""
    M, m, E = sp.symbols("M m E", positive=True)
    mrec2, R2s, rec = readme_phase_space(M, m, E)
    out = [Result("README: phase-space modes are boosted as the kaon decays",
                  "Two-body and phase-space modes are boosted to the lab as the kaon decays are" in
                  L.line("Two-body and phase-space modes are boosted"), "as written"),
           exact("m_rec^2 = (P - p)^2 for the parent at rest, P = (M, 0), p = (E, p): M^2 + m^2 - 2ME",
                 mrec2 - (M**2 + m**2 - 2 * M * E)),
           Result("README's recursion: R_k(m) = int dm' 2m' R_2(m; m_1, m') R_{k-1}(m')",
                  rec == r"\int dm'\thinspace 2m'\thinspace R_2(m; m_1, m')\thinspace R_{k-1}(m')", "as written")]
    a_, b_, c_ = sp.symbols("a_ b_ c_", positive=True)
    lam_f = sp.lambdify((a_, b_, c_), sp.expand((R2s(a_, b_, c_) * a_**2)**2), "mpmath")    # the README's lambda
    R2 = lambda mm, m1, m2: mp.sqrt(max(lam_f(mm, m1, m2), 0)) / mm**2
    # three bodies: R_3(M) = int dm' 2m' R_2(M; m1, m') R_2(m'; m2, m3) against the Dalitz plot area in (E1, E2)
    gen = rng(3)
    worst = mp.mpf(0)
    with mp.workdps(30):
        for _ in range(4):
            Mv = mp.mpf(gen.uniform(1.5, 2.3))
            m1, m2, m3 = (mp.mpf(gen.uniform(0.0, 0.5)) for _ in range(3))
            R3 = mp.quad(lambda mm: 2 * mm * R2(Mv, m1, mm) * R2(mm, m2, m3), [m2 + m3, Mv - m1])

            def E2_range(E1):
                p1 = mp.sqrt(E1**2 - m1**2)
                s23 = Mv**2 + m1**2 - 2 * Mv * E1
                m23 = mp.sqrt(s23)
                E2s = (s23 + m2**2 - m3**2) / (2 * m23)
                p2s = mp.sqrt(max(E2s**2 - m2**2, 0))
                return ((Mv - E1) * E2s - p1 * p2s) / m23, ((Mv - E1) * E2s + p1 * p2s) / m23
            E1max = (Mv**2 + m1**2 - (m2 + m3)**2) / (2 * Mv)
            area = mp.quad(lambda E1: E2_range(E1)[1] - E2_range(E1)[0], [m1, E1max])
            worst = max(worst, abs(R3 / (4 * area) - 1))                  # Phi_3 = pi^2/(4M^2) int ds .. = pi^2 area
    out.append(Result("R_3 = 4 x the Dalitz area int dE1 dE2 (Phi_3 = pi^2 dE1 dE2 = (pi^2/4) R_3), any masses",
                      worst < 1e-20, f"max rel. difference {mp.nstr(worst, 3)} at 4 points, 30 digits"))
    # p R_2(m_rec) = the length of the line E_1 = E across the Dalitz plot, exactly
    m2_, m3_ = sp.symbols("m_2 m_3", positive=True)
    mr = sp.sqrt(mrec2)
    P = sp.sqrt(E**2 - m**2)
    E2s = (mrec2 + m2_**2 - m3_**2) / (2 * mr)
    p2s = sp.sqrt(E2s**2 - m2_**2)
    length = 2 * P * p2s / mr                                                  # hi - lo of the E_2 range at E_1 = E
    out.append(exact("p R_2(m_rec; m_2, m_3) = the length of the line of fixed E across the flat Dalitz plot",
                     sp.simplify((P * R2s(mr, m2_, m3_))**2 - length**2)))
    Mz, mrec = sp.symbols("M m_rec", positive=True)
    Ez = (Mz**2 - mrec**2) / (2 * Mz)                                     # massless: p = E, m_rec^2 = M^2 - 2 M E
    out.append(exact("massless three bodies: dN/dE = p R_2(m_rec) = E (the flat Dalitz plot)",
                     Ez * R2s(mrec, 0, 0) - Ez))
    with mp.workdps(30):
        M_, m1, m2, m3 = mp.mpf(2), mp.mpf("0.3"), mp.mpf("0.2"), mp.mpf("0.4")
        bad = mp.quad(lambda mm: R2(M_, m1, mm) * R2(mm, m2, m3), [m2 + m3, M_ - m1])
        good = mp.quad(lambda mm: 2 * mm * R2(M_, m1, mm) * R2(mm, m2, m3), [m2 + m3, M_ - m1])
    out.append(Result("control: dm' for dm' 2m' in the recursion", abs(bad / good - 1) > 1e-3,
                      f"differs by {mp.nstr(bad / good - 1, 3)}"))
    return out


# --- the tau ---

def leptonic_rate(lepton_mass, helicity, chirality=-1):
    """|M|^2 of a pseudoscalar's decay at rest into l+ nu (current ∝ its momentum), the l+ along z with helicity
    +-1/2 (helicity = +-1), from the explicit trace; E = m (v + 1/v)/2, P = m (v - 1/v)/2 of the l+."""
    m, v = lepton_mass, sp.Symbol("v", positive=True)
    E, P = m * (v + 1 / v) / 2, m * (v - 1 / v) / 2
    l, nu = (E, 0, 0, P), (P, 0, 0, -P)
    J = [a + b for a, b in zip(l, nu)]                                             # f p_parent
    s = (helicity * P / m, 0, 0, helicity * E / m)
    return lepton_trace(J, (0, 0, 0, 0), nu, projector(l, m, s), chirality)


@derivation("cd.tau_pol")
def check_tau_pol():
    """D(s)+ -> l+ nu: the spin-0 parent and the left-handed nu make the l+ left-handed (P = -1)."""
    m = sp.Symbol("m_l", positive=True)
    up, down = leptonic_rate(m, 1), leptonic_rate(m, -1)
    up_b = leptonic_rate(m, 1, chirality=1)
    said = r"The $\tau^+$ is left-handed in the parent frame."
    pol = sp.simplify((up - down) / (up + down))
    readme_P = L.to_sympy(L.sides(L.inline(r"\mathcal P = -1"))[1], {})
    return [Result("README: the tau+ is left-handed in the parent frame", said in L.line(said), "as written"),
            exact("|M|^2 of helicity +1/2 = 0 (any lepton mass): the l+ is left-handed", up),
            Result("P = (|M|^2(+) - |M|^2(-))/(sum) = -1 = the README's P", pol == -1 and readme_P == -1,
                   f"P = {pol}"),
            exact_fails("control: a V+A current makes a right-handed l+", up_b)]


@derivation("cd.tau_decays")
def check_tau_decays():
    """The tau+ modes are PDG's eleven basis modes above 0.5 %, charge conjugated."""
    rule = r"eleven modes above "
    words = {"eleven": 11, "ten": 10, "twelve": 12}
    count = words[re.search(r"It decays by its (\w+) modes above", L.line(rule)).group(1)]
    thr = readme_threshold(rule)
    pdg = PDG.TAU_DECAYS
    with L.patched(r"It decays by its eleven modes above 0.5%",
                   r"It decays by its eleven modes above 0.7%"):
        thr_b = readme_threshold(rule)
    ok = lambda t: (count == len(pdg) and all(rat(md["br"]) > t for md in pdg)
                    and rat(PDG.TAU_LARGEST_EXCLUDED["br"]) < t)
    return [Result(f"README: {count} modes = PDG 2026's tau- basis modes above {thr * 100} % (fs_tau_plus)", ok(thr),
                   f"{len(pdg)} modes, smallest {min(md['br'] for md in pdg)}, largest left out "
                   f"{PDG.TAU_LARGEST_EXCLUDED['pdg']} {PDG.TAU_LARGEST_EXCLUDED['br']}"),
            control_fails("README threshold 0.7 %", not ok(thr_b))]


TAU_HEAD = r"| Mode | Branching ratio | PDG 2026 | Angular distribution |"
TAU_TEX = ((r"\nu_\mathrm e", "nu_e"), (r"\nu_\mu", "nu_mu"), (r"\bar\nu_\tau", "nu_tau_bar"), (r"\mu^+", "mu+"),
           (r"\mathrm e^+", "e+"), (r"\pi^+", "pi+"), (r"\pi^-", "pi-"), (r"\pi^0", "pi0"), (r"\mathrm K^+", "K+"),
           (r"\mathrm K^0", "K0"), (r"\omega", "omega"))


def tau_mode(tex):
    body, names = tex.strip("$"), []
    while body:
        for t, name in TAU_TEX:
            if body.startswith(t):
                names.append(name)
                body = body[len(t):]
                break
        else:
            raise ValueError(f"unknown particle in {tex}: {body}")
    return tuple(sorted(names))


def tau_treatment(fs):
    """The README's angular distribution of a tau+ mode by its rules."""
    hadrons = [f for f in fs if not f.startswith("nu") and f not in ("e+", "mu+")]
    if not hadrons:
        return "Michel spectra"
    if hadrons in (["pi+"], ["K+"]):
        return "$a = -1$"
    if sorted(hadrons) == ["pi+", "pi0"]:
        return r"as $\rho^+\bar\nu_\tau$, $a = a_\rho$"
    if sorted(hadrons) in (["pi+", "pi0", "pi0"], ["pi+", "pi+", "pi-"]):
        return r"as $a_1^+\bar\nu_\tau$, $a = a_{a_1}$"
    return "phase space"


def tau_table_results():
    rows = L.table(TAU_HEAD)
    pdg = {tuple(sorted(md["fs_tau_plus"])): md for md in PDG.TAU_DECAYS}
    bad, seen = [], set()
    for mode_tex, br, gam, treat in rows:
        fs = tau_mode(mode_tex)
        md = pdg.get(fs)
        seen.add(fs)
        if md is None or not (printed(br, rat(md["br"])) and sp.Rational(br) == rat(md["br"])
                              and gam == f"Γ{gamma_number(md['pdg'])}" and treat == tau_treatment(fs)):
            bad.append((mode_tex, br, gam, treat))
    total = sum(rat(md["br"]) for md in PDG.TAU_DECAYS)
    tau_line = L.line("the charge conjugates of PDG's $\\tau^-$ basis modes")
    said = re.search(r"They are renormalised to their sum, ([\d.]+):", tau_line).group(1)
    pages = sorted({md["page"] for md in PDG.TAU_DECAYS})
    return [Result("README rows = PDG 2026's tau+ modes (branching ratio, Gamma), each once, with the README's "
                   "angular distributions", not bad and seen == set(pdg) and len(rows) == len(pdg),
                   "all" if not bad else f"differ: {bad}"),
            Result("the sum the modes are renormalised to", sp.Rational(said) == total, f"{total}"),
            Result("README's pages (pp. 6-11) = those of the entries", f"(pp. {pages[0]}–{pages[-1]})" in
                   L.line("the charge conjugates of PDG's $\\tau^-$ basis modes"), f"{pages}")]


@derivation("cd.tau_table")
def check_tau_table():
    """The tau+ table: PDG 2026's eleven basis modes, their sum, and the angular distribution of each."""
    out = tau_table_results()
    rho = next(md for md in PDG.TAU_DECAYS if gamma_number(md["pdg"]) == 14)
    out.append(Result("pi+ pi0 nu as rho+ nu: PDG's non-rho part is consistent with zero",
                      "non-rho part G15 = (3.0 +- 3.2) x 10^-3" in rho["note"], rho["note"]))
    with L.patched("| 0.1737 | Γ3 |", "| 0.1738 | Γ3 |"):
        broken = tau_table_results()
    out.append(control_fails("README with 0.1738 for mu+ nu nu", not all(r.ok for r in broken)))
    return out


@functools.cache
def tau_two_body(spin_one):
    """|M|^2 of tau+ -> h+ anti-nu_tau for a tau+ at rest with its spin along z, the hadron at cos theta = c to the
    spin: spin 0 (current f p_h) or spin 1 (current f eps, summed over -g + p_h p_h/m^2), explicit traces."""
    gam, g5, one = dirac()
    mt, mh = sp.symbols("m_tau m_h", positive=True)
    c = sp.Symbol("c", real=True)
    sn = sp.sqrt(1 - c**2)
    Eh, pp = (mt**2 + mh**2) / (2 * mt), (mt**2 - mh**2) / (2 * mt)
    ph, k = (Eh, pp * sn, 0, pp * c), (pp, -pp * sn, 0, -pp * c)
    rho = projector((mt, 0, 0, 0), mt, (0, 0, 0, 1))                              # the tau+ (v v-bar), spin up
    g = [1, -1, -1, -1]
    if not spin_one:
        M2 = (slash(ph) * (one - g5) * slash(k) * (one + g5) * slash(ph) * rho).trace()
    else:
        M2 = 0
        for a in range(4):
            for b in range(4):
                Pab = -g[a] * int(a == b) + ph[a] * ph[b] * g[a] * g[b] / mh**2      # lower indices
                if Pab != 0:
                    M2 += Pab * (gam[a] * (one - g5) * slash(k) * (one + g5) * gam[b] * rho).trace()
    M2 = sp.factor(sp.simplify(sp.expand(M2)))
    return M2, c, mt, mh


def asymmetry(density, c):
    """a of a density ∝ 1 + a c."""
    return sp.simplify(sp.diff(density, c) / density.subs(c, 0))


TAU_ANG_PIN = r"\frac{dN}{d\cos\theta} = \frac{1 + a\thinspace\mathcal P\cos\theta}{2}"


def readme_tau_ang(a, P, ct, mt, mr):
    disp = L.display(TAU_ANG_PIN)
    dist = L.to_sympy(L.sides(L.parts(disp)[0])[1], {"a": a, r"\mathcal P": P, r"\cos\theta": ct})
    sides = L.sides(L.part(disp, r"a_\pi ="))
    assert sides[:2] == [r"a_\pi", r"a_\mathrm K"], sides
    spin0 = [L.to_sympy(sides[2], {})]
    arho = L.to_sympy(L.sides(L.part(disp, r"a_\rho ="))[1], {r"m_\tau": mt, r"m_\rho": mr})
    return dist, spin0, arho


@derivation("cd.tau_ang")
def check_tau_ang():
    """tau+ -> h+ anti-nu: (1 + a P cos)/2 with a = -1 (spin 0) and a_rho (spin 1) from the traces; PDG's xi agree."""
    a, P, ct = sp.symbols("a P c")
    mt, mh = sp.symbols("m_tau m_h", positive=True)
    dist, spin0, arho = readme_tau_ang(a, P, ct, mt, mh)
    M2_0, c0, mt0, mh0 = tau_two_body(False)
    M2_1, c1, mt1, mh1 = tau_two_body(True)
    a0 = asymmetry(M2_0, c0)                                                       # spin along +z: P = +1
    a1 = asymmetry(M2_1, c1).subs({mt1: mt, mh1: mh})
    with L.patched(r"a_\rho = -\frac{m_\tau^2 - 2m_\rho^2}", r"a_\rho = \frac{m_\tau^2 - 2m_\rho^2}"):
        _, _, arho_b = readme_tau_ang(a, P, ct, mt, mh)
    pulls = {k: (PDG.TAU_DECAY_PARAMETERS[k][0] - PDG.TAU_VA[k]) / PDG.TAU_DECAY_PARAMETERS[k][1]
             for k in ("xi(pi)", "xi(rho)", "xi(a1)")}
    return [exact("the distribution is normalised: int (1 + a P cos)/2 dcos over [-1, 1] = 1",
                  sp.integrate(dist, (ct, -1, 1)) - 1),
            exact("tau+ -> pi+ (K+) anti-nu: |M|^2 = 2 m_tau^2 (m_tau^2 - m^2)(1 - cos theta), spin up",
                  M2_0 - 2 * mt0**2 * (mt0**2 - mh0**2) * (1 - c0)),
            exact("a_pi = a_K = -1 = the spin-0 trace's asymmetry", sum((s_ - a0)**2 for s_ in spin0)),
            exact("a_rho = README = the spin-1 trace's asymmetry, summed over the hadron's spin", arho - a1),
            exact_fails("control: README a_rho with + sign", arho_b - a1),
            exact("the anti-nu_tau of a two-body mode goes opposite: asymmetry -a (cos -> -cos)",
                  asymmetry(M2_0.subs(c0, -c0), c0) + a0),
            Result("PDG 2026's chirality parameters xi(pi), xi(rho), xi(a1) agree with the V-A value 1 within 2 sigma, "
                   "on the README's pages", all(abs(p_) < 2 for p_ in pulls.values())
                   and readme_pages("chirality parameters") == pages_of(("xi(pi)", "xi(rho)", "xi(a1)")),
                   ", ".join(f"{k} {v:+.2f} sigma" for k, v in pulls.items()))]


def readme_pages(after):
    """The page range "(pp. a–b)" that follows `after` on its README line, as (a, b)."""
    line = L.line(after)
    a, b = re.search(re.escape(after) + r".*?\(pp\. (\d+)–(\d+)\)", line).groups()
    return int(a), int(b)


def pages_of(names):
    pages = [PDG.TAU_DECAY_PARAMETERS[n][2] for n in names]
    return min(pages), max(pages)


@functools.cache
def michel_matrix_element():
    """|M|^2 of tau+(p, s) -> l+(b) nu_l(a) anti-nu_tau(c), V-A, massless products: two explicit traces."""
    gam, g5, one = dirac()
    a, b, c, p, s = (sp.symbols(f"{v}0:4", real=True) for v in "abcps")
    m = sp.Symbol("m", positive=True)
    g = [1, -1, -1, -1]
    rho = projector(p, m, s)
    L1 = [[(slash(a) * gam[i] * (one - g5) * slash(b) * (one + g5) * gam[j]).trace() for j in range(4)]
          for i in range(4)]
    L2 = [[(gam[i] * (one - g5) * slash(c) * (one + g5) * gam[j] * rho).trace() for j in range(4)] for i in range(4)]
    M2 = sp.expand(sum(L1[i][j] * L2[i][j] * g[i] * g[j] for i in range(4) for j in range(4)))
    return M2, (a, b, c, p, s, m)


def two_body_tensor():
    """I^{ab} = int d^3k1/(2E1) d^3k2/(2E2) delta^4(Q - k1 - k2) k1^a k2^b for massless k1, k2, in the Q rest frame
    (Q = (Q0, 0, 0, 0)): k1 = (w, w n), k2 = (w, -w n), w = Q0/2, the angles integrated."""
    Q0 = sp.Symbol("Q0", positive=True)
    th, ph = sp.symbols("theta phi", real=True)
    w = Q0 / 2
    n = (sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th))
    k1, k2 = (w, *[w * x for x in n]), (w, *[-w * x for x in n])
    # d^3k1/(2E1) delta(Q0 - 2|k1|) /(2E2) = w^2 dw dOmega/(4 w^2) delta(Q0 - 2w) = dOmega/8
    I = [[sp.integrate(sp.integrate(k1[i] * k2[j] * sp.sin(th) / 8, (th, 0, sp.pi)), (ph, 0, 2 * sp.pi))
          for j in range(4)] for i in range(4)]
    return I, Q0


MICHEL_PIN = r"\frac{dN_\ell}{dx} = 2x^2(3 - 2x)"


def readme_michel(x):
    disp = L.display(MICHEL_PIN)
    get = lambda start: L.to_sympy(L.sides(L.part(disp, start))[1], {"x": x})
    return get(r"\frac{dN_\ell}{dx} ="), get(r"a_\ell ="), get(r"\frac{dN_\nu}{dx} ="), get(r"a_\nu =")


@derivation("cd.michel")
def check_michel():
    """tau+ -> l+ nu anti-nu (V-A, massless): the README's spectra and asymmetries; Michel's with V-A parameters."""
    M2, (a, b, c4, p, s, m) = michel_matrix_element()
    ps = readme_pol()[1](p, m, s)                                                  # the tau+ also takes p + m s
    out = [exact("|M|^2 = 128 ((p + m s).p_nu_l)(p_l.p_anti-nu_tau), p + m s by the spin rule (traces)",
                 M2 - 128 * mdot(ps, a) * mdot(b, c4)),
           exact("|M|^2 symmetric in l+ <-> anti-nu_tau: the anti-nu_tau has the l+'s spectrum and asymmetry",
                 (128 * mdot(ps, a) * mdot(b, c4)) - (128 * mdot(ps, a) * mdot(c4, b)))]
    I, Q0 = two_body_tensor()
    out.append(exact("int k1^a k2^b over massless two-body phase space = (pi/24)(Q^2 g^ab + 2 Q^a Q^b)",
                     sum((I[i][j] - sp.pi / 24 * (Q0**2 * (1 if i == j == 0 else -1 if i == j else 0)
                                                  + 2 * (Q0 if i == 0 else 0) * (Q0 if j == 0 else 0)))**2
                         for i in range(4) for j in range(4))))
    x, ct, mm = sp.Symbol("x", positive=True), sp.Symbol("c", real=True), sp.Symbol("m", positive=True)
    E = x * mm / 2
    st = sp.sqrt(1 - ct**2)
    k = (E, E * st, 0, E * ct)                                                     # a product at cos theta = c
    pp_ = (mm, 0, 0, mm)                                                           # p + m s, spin along z
    Q = [mm - k[0], -k[1], -k[2], -k[3]]
    gQ = lambda u, v: mdot(Q, Q) * mdot(u, v) + 2 * mdot(Q, u) * mdot(Q, v)       # (24/pi) I_ab u^a v^b
    dens_l = sp.expand(E * gQ(pp_, k))                                             # l+ (anti-nu_tau): d^3k/2E -> E dE
    dens_nu = sp.expand(E * mdot(pp_, k) * 6 * mdot(Q, Q))                         # nu_l: g_ab I^ab = (pi/24) 6 Q^2
    spec = lambda d: sp.factor(sp.integrate(d, (ct, -1, 1)))
    norm = lambda f: f / sp.integrate(f, (x, 0, 1))
    Nl, al, Nn, an = readme_michel(x)
    with L.patched(r"\frac{dN_\nu}{dx} = 12x^2(1 - x)", r"\frac{dN_\nu}{dx} = 6x^2(1 - x)"):
        _, _, Nn_b, _ = readme_michel(x)
    out += [exact("l+: README dN/dx = the trace's spectrum, normalised", norm(spec(dens_l)) - Nl),
            exact("l+: README a_l = the trace's asymmetry about the tau spin", asymmetry(dens_l, ct) - al),
            exact("nu_l: README dN/dx = the trace's spectrum, normalised", norm(spec(dens_nu)) - Nn),
            exact("nu_l: README a_nu = the trace's asymmetry", asymmetry(dens_nu, ct) - an),
            exact("README's spectra are normalised to one product", (sp.integrate(Nl, (x, 0, 1)) - 1)**2
                  + (sp.integrate(Nn, (x, 0, 1)) - 1)**2),
            exact_fails("control: README with 6 x^2 (1 - x)", norm(spec(dens_nu)) - Nn_b)]
    # Michel's form with the V-A parameters rho = delta = 3/4, xi = 1, eta = 0
    said = L.line(r"These are Michel's spectra with the V−A parameters")
    va = re.search(r"\$\\rho = \\delta = (\d)/(\d)\$, \$\\xi = (\d)\$ and \$\\eta = (\d)\$", said).groups()
    rho = delta = sp.Rational(int(va[0]), int(va[1]))
    xi, eta = int(va[2]), int(va[3])
    michel = x**2 * (3 * (1 - x) + 2 * rho / 3 * (4 * x - 3) + ct * xi * ((1 - x) + 2 * delta / 3 * (4 * x - 3)))
    out.append(exact("Michel's x^2[3(1 - x) + (2 rho/3)(4x - 3) + xi cos ((1 - x) + (2 delta/3)(4x - 3))] with "
                     "the README's V-A values (eta = 0 term absent for massless l+) = README's l+ density",
                     (norm(spec(michel)) - Nl)**2 + (asymmetry(michel, ct) - al)**2 + eta**2))
    pulls = {k_: (PDG.TAU_DECAY_PARAMETERS[k_][0] - PDG.TAU_VA[k_]) / PDG.TAU_DECAY_PARAMETERS[k_][1]
             for k_ in ("rho(e or mu)", "xi(e or mu)", "eta(e or mu)", "(delta xi)(e or mu)")}
    out.append(Result("PDG 2026's fits of the tau Michel parameters (e or mu) agree with V-A within 2 sigma, on the "
                      "README's pages, and the README's values are PDG's V-A predictions",
                      all(abs(p_) < 2 for p_ in pulls.values()) and float(rho) == PDG.TAU_VA["rho(e or mu)"]
                      and xi == PDG.TAU_VA["xi(e or mu)"] and eta == PDG.TAU_VA["eta(e or mu)"]
                      and readme_pages("which PDG's fits match") == pages_of(tuple(pulls)),
                      ", ".join(f"{k_} {v:+.2f} sigma" for k_, v in pulls.items())))
    return out


@derivation("cd.tau_frame")
def check_tau_frame():
    """The cascade needs an isotropic decay; the polarised tau+ is not, so it decays in its parent's frame."""
    M2, c, mt, mh = tau_two_body(False)
    zt = sp.Symbol("z", positive=True)
    # a pi+ at cos theta* to the tau's flight (its spin points backwards for P = -1): light-cone fraction z
    Es, ps = (mt**2 + mh**2) / (2 * mt), (mt**2 - mh**2) / (2 * mt)
    cos_z = (zt * mt - Es) / ps
    weight = M2.subs(c, -cos_z)
    flatness = sp.simplify(sp.diff(sp.simplify(weight / M2.subs(c, 0)), zt))
    unpolarised = sp.diff(sp.simplify((M2.subs(c, cos_z) + M2.subs(c, -cos_z)) / (2 * M2.subs(c, 0))), zt)
    return [Result("README: the tau+ decays in its parent's frame with its polarisation",
                   "The $\\tau^+$ decays in its parent's frame instead, with its polarisation." in
                   L.line(r"The $\tau^+$ decays in its parent's frame instead"), "as written"),
            exact("isotropic decay: z = (E* + p* cos)/m flat (dz/dcos constant): F_Y->X depends on z alone",
                  sp.diff(sp.diff(Es / mt + ps / mt * c, c), c)),
            Result("tau+ -> pi+ anti-nu with P = -1: dN/dz is not flat (it rises with z), so no F_tau->pi(x/y) of the "
                   "tau alone", flatness != 0, f"d/dz of the weight = {flatness}"),
            control_fails("the same test on an unpolarised tau (the spins averaged) finds it flat", unpolarised == 0)]


TAU_BOOST_PIN = r"\gamma_\tau = \frac{M^2 + m_\tau^2}{2Mm_\tau}"


def readme_tau_boost(sym):
    disp = L.display(TAU_BOOST_PIN)
    E = L.to_sympy(L.sides(L.parts(disp)[0])[1], {r"\gamma_\tau": sym["g"], r"\beta_\tau": sym["b"], "E'": sym["E"],
                                                  "p'": sym["p"], r"\cos\theta": sym["c"]})
    g = L.to_sympy(L.sides(L.part(disp, r"\gamma_\tau ="))[1], {"M": sym["M"], r"m_\tau": sym["mt"]})
    return E, g


@derivation("cd.tau_boost")
def check_tau_boost():
    """A tau product's parent-frame energy is the boost of its tau-frame momentum; gamma_tau from D(s)+ -> tau+ nu."""
    q = sp.Symbol("q", positive=True)
    sym = dict(g=(q + 1 / q) / 2, b=(q - 1 / q) / (q + 1 / q), E=sp.Symbol("E1", positive=True),
               p=sp.Symbol("p1", positive=True), c=sp.Symbol("c", real=True), M=sp.Symbol("M", positive=True),
               mt=sp.Symbol("m_tau", positive=True))
    E, g = readme_tau_boost(sym)
    # the tau moves along +z with rapidity ln q; the product (E', p' sin, 0, p' cos) in the tau frame
    boosted = (q + 1 / q) / 2 * sym["E"] + (q - 1 / q) / 2 * sym["p"] * sym["c"]
    Etau = (sym["M"]**2 + sym["mt"]**2 - 0) / (2 * sym["M"])                       # two-body, massless nu
    with L.patched(r"\left(E' + \beta_\tau p'\cos\theta\right)", r"\left(E' - \beta_\tau p'\cos\theta\right)"):
        E_b, _ = readme_tau_boost(sym)
    return [exact("README E = the parent-frame energy of the boosted 4-momentum (gamma beta = sinh, gamma = cosh)",
                  E - boosted),
            exact("README gamma_tau = E_tau/m_tau, E_tau = (M^2 + m_tau^2)/(2M) of the two-body decay",
                  g - Etau / sym["mt"]),
            exact_fails("control: README with E' - beta p' cos", E_b - boosted)]


# --- tables on the grid ---

def readme_flat():
    ln = L.line("is spread uniformly over $x$ between")
    lo, hi = re.search(r"between \$(.+?)\$ and \$(.+?)\$ in the lab", ln).groups()
    toks = {r"E^\ast": sp.Symbol("E", positive=True), r"p^\ast": sp.Symbol("p", positive=True),
            "M": sp.Symbol("M", positive=True)}
    return L.to_sympy(lo, toks), L.to_sympy(hi, toks)


@derivation("cd.flat")
def check_flat():
    """An isotropic decay of an ultra-relativistic parent spreads a product uniformly over (E* -+ p*)/M."""
    E, p, M = sp.symbols("E p M", positive=True)
    beta, c = sp.symbols("beta c", real=True)
    lo, hi = readme_flat()
    x = sp.limit((E + beta * p * c) / M, beta, 1)                                 # gamma (E* + beta p* c)/(gamma M)
    with L.patched(r"and $(E^\ast + p^\ast)/M$ in the lab", r"and $(E^\ast + 2p^\ast)/M$ in the lab"):
        lo_b, hi_b = readme_flat()
    return [exact("x = (E* + p* cos)/M is linear in cos, so isotropy makes it uniform", sp.diff(x, c, 2)),
            exact("its ends at cos = -1, +1 are the README's", (x.subs(c, -1) - lo)**2 + (x.subs(c, 1) - hi)**2),
            exact_fails("control: README with (E* + 2p*)/M", (x.subs(c, -1) - lo_b)**2 + (x.subs(c, 1) - hi_b)**2)]


BINS_PIN = r"\Delta\thinspace T(x_k) = "


def readme_bins():
    sym = sp.symbols("xk xk1 xkm Ek Nk Ekm Nkm")
    lhs, rhs = L.sides(L.display(BINS_PIN))
    assert lhs == r"\Delta\thinspace T(x_k)", lhs
    toks = dict(zip(("x_k", "x_{k+1}", "x_{k-1}", r"\mathcal E_k", r"\mathcal N_k", r"\mathcal E_{k-1}",
                     r"\mathcal N_{k-1}"), sym))
    return L.to_sympy(rhs, toks), sym


def bins_sums(expr, sym, K=4):
    """(sum_k Delta T(x_k) - sum N, sum_k x_k Delta T(x_k) - sum E) for K bins of symbolic number and energy."""
    X = sp.symbols(f"X0:{K + 1}", positive=True)
    N = sp.symbols(f"N0:{K}", positive=True)
    En = sp.symbols(f"En0:{K}", positive=True)
    xk, xk1, xkm, Ek, Nk, Ekm, Nkm = sym
    pad = sp.symbols("Xm Xp", positive=True)
    xs = lambda k: pad[0] if k < 0 else pad[1] if k > K else X[k]
    shares = [expr.subs({xk: xs(k), xk1: xs(k + 1), xkm: xs(k - 1), Ek: En[k] if k < K else 0,
                         Nk: N[k] if k < K else 0, Ekm: En[k - 1] if k > 0 else 0, Nkm: N[k - 1] if k > 0 else 0},
                        simultaneous=True) for k in range(K + 1)]
    return (sp.simplify(sum(shares) - sum(N)), sp.simplify(sum(X[k] * shares[k] for k in range(K + 1)) - sum(En)))


@derivation("cd.bins")
def check_bins():
    """The shares of each bin keep its number and energy, and are positive when its mean x lies in it."""
    expr, sym = readme_bins()
    n, e = bins_sums(expr, sym)
    with L.patched(r"\frac{\mathcal E_k - x_{k+1}\mathcal N_k}{x_k - x_{k+1}}",
                   r"\frac{\mathcal E_k - x_k\mathcal N_k}{x_k - x_{k+1}}"):
        expr_b, sym_b = readme_bins()
    nb, eb = bins_sums(expr_b, sym_b)
    xk, xk1, xkm, Ek, Nk, Ekm, Nkm = sym
    upper = expr.subs({Ekm: 0, Nkm: 0})                                              # bin k's share at x_k
    lower = (expr.subs({Ek: 0, Nk: 0}))                                              # bin k-1's share at x_k
    gen = rng(7)
    pos = []
    for _ in range(20):
        a, b = sorted(sp.Rational(gen.randint(1, 999), 1000) for _ in range(2))
        mean = a + (b - a) * sp.Rational(gen.randint(0, 100), 100)
        pos.append(upper.subs({xk: b, xk1: a, Nk: 1, Ek: mean}) >= 0 and
                   lower.subs({xkm: b, xk: a, Nkm: 1, Ekm: mean}) >= 0)
    out = [exact("the shares add up to the products' number: sum_k Delta T(x_k) = sum N_k", n),
           exact("and their x-moments to the energy: sum_k x_k Delta T(x_k) = sum E_k", e),
           Result("both shares of a bin are >= 0 when its mean x E/N lies in it (20 random bins)", all(pos), "all"),
           Result("control: README with x_k for x_{k+1} in the first share", nb != 0 or eb != 0, "fails, as it must")]
    return out


# --- short-lived products ---

SHORT_PIN = r"gives each its PDG 2026 modes of "


def short_lived_results(thr):
    out, bad, lines = [], [], []
    for name, d in PDG.SHORT_LIVED.items():
        brs = [rat(md["br"]) for md in d["modes"]]
        low = [md["pdg"] for md in d["modes"] if rat(md["br"]) < thr]
        below_ok = d["below"] is None or rat(d["below"]["br"]) < thr
        if low or not below_ok:
            bad.append(f"{name}: {low or ''} {'' if below_ok else d['below']['pdg']}")
        lines.append(f"{name} {len(brs)}")
    out.append(Result(f"every listed mode of the short-lived products is >= {float(thr * 100):g} %, the largest left "
                      "out below (pdg_charm.SHORT_LIVED)", not bad, ", ".join(lines) if not bad else f"{bad}"))
    return out


@derivation("cd.short_lived")
def check_short_lived():
    """The short-lived products' modes: PDG's of 0.5 % and above (two named exceptions); isospin 2/3 and 1/3."""
    thr = sp.Rational(L.between(L.line(SHORT_PIN), "modes of ", "%")) / 100
    out = short_lived_results(thr)
    with L.patched(r"its PDG 2026 modes of 0.5%", r"its PDG 2026 modes of 0.6%"):
        thr_b = sp.Rational(L.between(L.line(SHORT_PIN), "modes of ", "%")) / 100
    out.append(control_fails("README threshold 0.6 %", not all(r.ok for r in short_lived_results(thr_b))))
    text = L.line(SHORT_PIN)
    exceptions = {k: d["not_final"] for k, d in PDG.SHORT_LIVED.items() if d["not_final"]}
    out.append(Result("the entries of 0.5 % and above that are not a final state: the eta's sums of modes, the rho's "
                      "radiative pi+ pi- gamma (inside its pi pi) and the omega's \"neutrals\", as the README says",
                      set(exceptions) == {"eta", "rho0", "omega"}
                      and r"The $\rho$ decays into $\pi\pi$ only, which PDG gives as about 100%" in text
                      and "the \"neutrals\" of the $\\omega$ are not a final state" in text, f"{exceptions}"))
    # isospin: K*+ -> K0 pi+ : K+ pi0 and f0 -> pi+ pi- : pi0 pi0 from the Clebsch-Gordan coefficients
    h = sp.Rational(1, 2)
    kstar = [CG(1, 1, h, -h, h, h).doit()**2, CG(1, 0, h, h, h, h).doit()**2]      # (pi+ K0), (pi0 K+)
    f0 = [2 * CG(1, 1, 1, -1, 0, 0).doit()**2, CG(1, 0, 1, 0, 0, 0).doit()**2]     # pi+ pi- (both orders), pi0 pi0
    said = re.search(r"isospin gives (\d)/(\d) and (\d)/(\d)", text).groups()
    shares = [sp.Rational(int(said[0]), int(said[1])), sp.Rational(int(said[2]), int(said[3]))]
    out.append(Result("isospin: K* -> K pi and f0(980) -> pi pi split 2/3 : 1/3 (Clebsch-Gordan), as the README says",
                      kstar == shares and f0 == shares, f"K*: {kstar}, f0: {f0}"))
    return out


CASCADE_PIN = r"F_{s\to X}(x) = F^{\rm direct}_ {s\to X}(x) + \sum_Y \int_x^1"


def readme_cascade(f1, f2, x, y):
    """The README's cascade term: (integrand in y, lower and upper limit)."""
    lhs, rhs = L.sides(L.parts(L.display(CASCADE_PIN))[0])
    assert lhs == r"F_{s\to X}(x)" and rhs.startswith(
        r"F^{\rm direct}_ {s\to X}(x) + \sum_Y \int_x^1 \frac{dy}{y}\thinspace"), rhs
    lo, hi = re.search(r"\\int_(\w)\^(\w)", rhs).groups()
    integrand = rhs.split(r"\frac{dy}{y}\thinspace", 1)[1]
    tokens = {"x": x, "y": y}
    expr = L.to_sympy(r"\frac{1}{y}\thinspace " + integrand, tokens, funcs={r"F_{s\to Y}": f1, r"F_{Y\to X}": f2})
    return expr, L.to_sympy(lo, tokens), L.to_sympy(hi, tokens)


@derivation("cd.cascade")
def check_cascade():
    """A short-lived Y: light-cone fractions multiply, so number and energy multiply."""
    x, y, z = sp.symbols("x y z", positive=True)
    a, b = sp.symbols("a0:3", real=True), sp.symbols("b0:3", real=True)
    F1 = lambda u: sum(a[k] * u**k for k in range(3))                            # any spectra on (0, 1] (polynomials)
    F2 = lambda u: sum(b[k] * u**(k + 1) for k in range(3))
    expr, lo, hi = readme_cascade(F1, F2, x, y)
    G = sp.integrate(sp.expand(expr), (y, lo, hi))
    with L.patched(r"\left(\frac{x}{y}\right)", r"\left(\frac{y}{x}\right)"):
        expr_b, lo_b, hi_b = readme_cascade(F1, F2, x, y)
    Gb = sp.integrate(sp.expand(expr_b), (y, lo_b, hi_b))
    # the delta-function composition x = y z: int x^n G dx = int int (y z)^n F1(y) F2(z) dy dz, every moment
    product = lambda n: sp.integrate(y**n * F1(y), (y, 0, 1)) * sp.integrate(z**n * F2(z), (z, 0, 1))
    out = []
    for n in range(4):
        what = {0: "number", 1: "energy"}.get(n, f"moment x^{n}")
        out.append(exact(f"{what}: int x^{n} F_cascade dx = int (yz)^{n} F_s->Y(y) F_Y->X(z) (delta(x - y z))",
                         sp.integrate(sp.expand(x**n * G), (x, 0, 1)) - product(n)))
    out.append(exact_fails("control: README with F_{Y->X}(y/x)",
                           sp.integrate(sp.expand(x * Gb), (x, 0, 1)) - product(1)))
    # light-cone fractions n.p (n light-like along the parent) multiply exactly: x_X = (n.p_X/n.p_Y)(n.p_Y/n.p_s)
    nl = (1, 0, 0, -1)
    pY, pX, ps = (sp.symbols(f"{v}0:4", real=True) for v in ("Y", "X", "S"))
    out.append(exact("x = y z for the light-cone fractions n.p/n.p_parent (any momenta)",
                     mdot(nl, pX) / mdot(nl, ps) - (mdot(nl, pX) / mdot(nl, pY)) * (mdot(nl, pY) / mdot(nl, ps))))
    # per unit ln x: x F_cascade = int d ln y [y F_s->Y(y)] [(x/y) F_Y->X(x/y)], a convolution in ln x
    out.append(exact("x times the integrand = (1/y)[y F_s->Y(y)][(x/y) F_Y->X(x/y)]: a convolution in ln x",
                     x * expr - (y * F1(y)) * ((x / y) * F2(x / y)) / y))
    return out


CONV_PIN = r"T_{s\to X}(x_k) = T^{\rm direct}_ {s\to X}(x_k) + \Delta\sum_Y"


def readme_conv():
    """The grid indices of the two tables in the README's grid cascade (as functions of j and k), and its relation of
    the grid points."""
    disp = L.display(CONV_PIN)
    rhs = L.sides(L.parts(disp)[0])[1]
    head = r"T^{\rm direct}_ {s\to X}(x_k) + \Delta\sum_Y\sum_{j=0}^k "
    assert rhs.startswith(head), rhs
    m = re.fullmatch(r"T_\{s\\to Y\}\(x_(\w|\{[^}]*\})\)\\thinspace T_\{Y\\to X\}\(x_(\w|\{[^}]*\})\)",
                     rhs[len(head):])
    j, k = sp.symbols("j k", integer=True)
    idx = [sp.sympify(g.strip("{}"), locals={"j": j, "k": k}) for g in m.groups()]
    return idx, (j, k), L.sides(L.part(disp, "x_jx_{k-j} ="))


def conv_sums(idx, jk, K=4):
    """(number, energy) differences of the grid cascade of tables A (of Y, K + 1 points) and B (of Y's products), with
    the README's grid indices: zero when number and energy multiply."""
    j, k = jk
    r, D = sp.symbols("r Delta", positive=True)
    A = sp.symbols(f"A0:{K + 1}")
    B = sp.symbols(f"B0:{K + 1}")
    at = lambda tab, i: tab[int(i)] if 0 <= i <= K else 0
    C = [D * sum(at(A, idx[0].subs({j: jj, k: kk})) * at(B, idx[1].subs({j: jj, k: kk})) for jj in range(kk + 1))
         for kk in range(2 * K + 1)]
    number = D * sum(C) - (D * sum(A)) * (D * sum(B))
    energy = D * sum(r**kk * C[kk] for kk in range(len(C))) - (D * sum(r**i * A[i] for i in range(K + 1))) * (
        D * sum(r**i * B[i] for i in range(K + 1)))
    return sp.expand(number), sp.expand(energy)


@derivation("cd.conv")
def check_conv():
    """The grid cascade is the convolution in ln x as a sum; number and energy multiply exactly on the grid."""
    idx, jk, (xl, xr) = readme_conv()
    n, e = conv_sums(idx, jk)
    k, j, D = sp.symbols("k j Delta", positive=True)
    xk = lambda i: sp.exp(-i * D)
    rel = L.to_sympy(xl, {"x_j": xk(j), "x_{k-j}": xk(k - j)}) - L.to_sympy(xr, {"x_k": xk(k)})
    with L.patched(r"T_{Y\to X}(x_{k-j}),\qquad", r"T_{Y\to X}(x_{k+j}),\qquad"):
        idx_b, jk_b, _ = readme_conv()
    nb, eb = conv_sums(idx_b, jk_b)
    j_, k_ = jk
    return [Result("README's indices: T_s->Y at x_j, T_Y->X at x_{k-j}",
                   idx[0] == j_ and sp.expand(idx[1] - (k_ - j_)) == 0, str(idx)),
            exact("number: sum_k Delta T_cascade(x_k) = (Delta sum_j T_s->Y)(Delta sum_l T_Y->X)", n),
            exact("energy: sum_k Delta x_k T_cascade(x_k) = (Delta sum x_j T_s->Y)(Delta sum x_l T_Y->X)", e),
            exact("README's x_j x_{k-j} = x_k on the grid x_k = e^(-k Delta)", sp.simplify(rel)),
            Result("control: README with x_{k+j}", nb != 0 or eb != 0, "fails, as it must")]


@derivation("cd.aligned")
def check_aligned():
    """A vector meson's decay into spinless products is even in the polar angle: mean energy exact, spread not."""
    th, ph, al = sp.symbols("theta phi alpha", real=True)
    r0 = sp.Symbol("rho_00", positive=True)                                       # the helicity-0 population
    # spin 1 -> 0 + 0 about the meson's axis z: |d^1_{+-1 0}|^2 = sin^2/2, |d^1_00|^2 = cos^2; populations 1 - r0, r0
    W = sp.Rational(3, 8) / sp.pi * ((1 - r0) * sp.sin(th)**2 + 2 * r0 * sp.cos(th)**2)
    phat = (sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th))
    nhat = (sp.sin(al), 0, sp.cos(al))                                              # the boost's direction, any angle
    avg = lambda f, w: sp.integrate(sp.integrate(sp.expand(f * w * sp.sin(th)), (ph, 0, 2 * sp.pi)), (th, 0, sp.pi))
    pn = sum(a * b for a, b in zip(phat, nhat))
    norm = sp.simplify(avg(1, W))
    first = sp.simplify(avg(pn, W))
    second = sp.simplify(avg(pn**2, W) - sp.Rational(1, 3))
    Wv = (1 + sp.Symbol("b") * sp.cos(th)) / (4 * sp.pi)                           # a forward-backward asymmetry
    shift = sp.simplify(avg(pn, Wv))
    said = L.line(r"but their decays into spinless products are even in the polar angle")
    return [Result("README: even in the polar angle, number and energy exact, only the spread not",
                   "The number and energy of their products are thus exact, and only their spread is not." in
                   L.line("The number and energy of their products are thus exact") and "spinless" in said,
                   "as written"),
            exact("the distribution is normalised for any populations", norm - 1),
            exact("<p-hat . n-hat> = 0 for any populations and any n-hat: the mean light-cone fraction is E'/m", first),
            Result("the second moment depends on the populations: the spread differs from isotropy", second != 0,
                   f"<(p.n)^2> - 1/3 = {second}"),
            control_fails("a distribution 1 + b cos theta (a vector polarisation analysed) shifts the mean",
                          shift != 0)]
