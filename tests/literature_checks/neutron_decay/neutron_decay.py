"""Derivation checks of the neutron decay page,
tests/literature_checks/neutron_decay/README.md, against its sources:
  - Hayen et al. (2018, Rev. Mod. Phys. 90, 015008), TeX source of
    arXiv:1709.07530v2 (energies in units of m_e c^2): the allowed spectrum
    (eq. 4), the point-charge Fermi function (eqs. 6-7) and the ratio of bound
    to free decay (Sec. VII.E, "Bound state beta decay")
  - Lipari, Lusignoli & Meloni (2007, PRD 75, 123005), TeX source of
    arXiv:0704.0718v1, Sec. VII.C: the mean energy share and the end point of
    the antineutrinos
  - Dubbers & Schmidt (2011, Rev. Mod. Phys. 83, 1111), arXiv:1105.3694v1, a PDF
    without TeX (transcribed below): the end-point energy of the decay protons
  - the PDG 2026 listing of p (transcribed below): the proton's charge radius.
The masses and alpha are the code's (src/am3_mojo/utils/constants.mojo, which
cites PDG 2026 for them).

Run from the repository root:
  .venv/bin/python tests/literature_checks/common/run.py neutron_decay
"""

import re

import mpmath as mp
import sympy as sp

import latex
import mojo_constants
from latex import HAYEN, LIPARI
from verify import (
    PDG_N,
    Entry,
    Result,
    derivation,
    exact,
    exact_fails,
    in_source,
    must_fail,
    numeric,
    rng,
    rounds_to,
    split_number,
    worst_of,
)

PAGE = "neutron_decay"
L = latex.page(PAGE)
TEX_H, TEX_L = "TeX " + HAYEN, "TeX " + LIPARI
DUBBERS_PDF = "PDF arXiv-1105.3694v1_Dubbers-Schmidt-2011.pdf"
PDG_P = "PDF PDG-2026_listing_p.pdf"

# this module's equations in the inventory (common/inventory.py)
INVENTORY = [
    Entry(
        "nd.E0",
        r"E_0 = \frac{m_n^2 - m_p^2 + m_e^2}{2m_n}",
        "derived",
        "kinematics of n -> p e nu (the electron's largest energy); "
        "PDG 2026 masses",
        PDG_N,
        file=PAGE,
    ),
    Entry(
        "nd.recoil",
        r"m_n - E_0 = (m_n^2 + m_p^2 - m_e^2)/(2m_n)",
        "derived",
        "kinematics (the proton's largest energy); Dubbers & Schmidt 2011, "
        "p. 48: 750 eV",
        DUBBERS_PDF,
        file=PAGE,
    ),
    Entry(
        "nd.hayen4",
        r"N(W)\thinspace dW = \frac{G_V^2 V_{ud}^2}{2\pi^3}",
        "transcribed",
        "Hayen et al. 2018, eq. 4 (its first form, all corrections)",
        TEX_H,
        file=PAGE,
    ),
    Entry(
        "nd.bound",
        r"bound decay into a hydrogen atom, $4.2 \times 10^{-6}$ of "
        r"the decays",
        "transcribed",
        "Hayen et al. 2018, Sec. VII.E (Bound state beta decay)",
        TEX_H,
        file=PAGE,
    ),
    Entry(
        "nd.hayen67",
        r"F_0(Z, W) = 4 (2pR)^{2(\gamma - 1)}",
        "transcribed",
        "Hayen et al. 2018, eqs. 6 and 7: the Fermi function of a point charge",
        TEX_H,
        file=PAGE,
    ),
    Entry(
        "nd.gamma1",
        r"their $\gamma = \sqrt{1 - \alpha^2}$ (written $\gamma_1$ "
        r"here) set to 1",
        "transcribed",
        "Hayen et al. 2018, eq. 7 at Z = 1 (the proton)",
        TEX_H,
        file=PAGE,
    ),
    Entry(
        "nd.limit",
        r"\Gamma(1 + iy)\thinspace\Gamma(1 - iy) = \frac{\pi y}{\sinh \pi y}",
        "derived",
        "Hayen et al. 2018, eq. 6 at Z = 1 for gamma -> 1 "
        "(the reflection formula of the gamma function)",
        TEX_H,
        file=PAGE,
    ),
    Entry(
        "nd.spectrum",
        r"\frac{dN}{dE} \propto F(E)\thinspace p\thinspace E\thinspace "
        r"(E_0 - E)^2",
        "derived",
        "Hayen et al. 2018, eq. 4 without its corrections, F their eq. 6 for "
        "gamma -> 1",
        TEX_H,
        file=PAGE,
    ),
    Entry(
        "nd.size",
        r"R = \sqrt{5/3}\thinspace r_p",
        "derived",
        "Hayen et al. 2018, eqs. 6-7 at Z = 1 against F; PDG 2026 p listing, "
        "p. 8: r_p (MOHR 25)",
        PDG_P,
        file=PAGE,
    ),
    Entry(
        "nd.Fp",
        r"F(E)\thinspace p = \frac{2\pi\alpha E}{1 - e^{-2\pi\alpha E/p}}",
        "derived",
        "the spectrum's F times p, finite at p = 0 "
        "(as decays.mojo neutron_beta_density computes it)",
        file=PAGE,
    ),
    Entry(
        "nd.nodes",
        r"E = m_e + (E_0 - m_e)\sin^2\frac{\pi u}{2}",
        "model",
        "am3_mojo's substitution for the rest-frame nodes "
        "(decays.mojo NDecBuild)",
        file=PAGE,
    ),
    Entry(
        "nd.weights",
        r"\omega_k \propto \frac{a_k}{2}\thinspace \frac{dE}{du}\thinspace "
        r"\frac{dN}{dE}",
        "model",
        "am3_mojo's Gauss-Legendre weights of the rest-frame nodes (NDecBuild)",
        file=PAGE,
    ),
    Entry(
        "nd.source",
        r"Q_X(E) = \int \frac{dE_n}{E_n}",
        "derived",
        "the decays counted (time-dilated rate, x = E/E_n, densities per "
        "unit energy)",
        file=PAGE,
        checks="tests/unit/test_added_processes.py (the decay rate in a "
        "closed box)",
    ),
    Entry(
        "nd.box",
        r"x \in \left[\frac{E^\ast - \beta_n p^\ast}{m_n}",
        "derived",
        "kinematics (isotropic decay, Lorentz boost of any beta)",
        file=PAGE,
    ),
    Entry(
        "nd.density",
        r"with density $m_n/(2\beta_n p^\ast)$",
        "derived",
        "kinematics",
        file=PAGE,
    ),
    Entry(
        "nd.FX",
        r"F_X(x;\beta_n) = \int dE^\ast\thinspace f_X(E^\ast)",
        "derived",
        "the uniform ranges of nd.box summed over the rest-frame spectrum",
        file=PAGE,
    ),
    Entry(
        "nd.antinu",
        r"x \in \left[\frac{(1 - \beta_n)\thinspace E_{\bar\nu}}{m_n}",
        "derived",
        "nd.box for a massless product",
        file=PAGE,
    ),
    Entry(
        "nd.xp",
        r"$x_p = (m_n - E_0)/m_n$",
        "derived",
        "energy conservation (no recoil in the lepton spectra)",
        file=PAGE,
        checks="tests/unit/test_added_processes.py (energy per decay)",
    ),
    Entry(
        "nd.stable",
        r"E^\ast - \beta_n p^\ast = \frac{m_e^2}{E^\ast + p^\ast} + (1 - "
        r"\beta_n)\thinspace p^\ast",
        "derived",
        "the lower ends of nd.box without cancellation "
        "(decays.mojo _ndec_column)",
        file=PAGE,
    ),
    Entry(
        "nd.means",
        r"\langle x\rangle = \langle E^\ast\rangle/m_n",
        "derived",
        "the mean shares from the spectrum; Lipari et al. 2007, "
        "Sec. VII.C: 5.1e-4",
        TEX_L,
        file=PAGE,
        checks="tests/unit/test_added_processes.py "
        "(the code's kernels: 5.1e-4)",
    ),
    Entry(
        "nd.ends",
        r"x = 2(E_0 - m_e)/m_n = 1.664 \times 10^{-3}",
        "derived",
        "the antineutrino's largest share; Lipari et al. 2007, "
        "Sec. VII.C: 1.66e-3",
        TEX_L,
        file=PAGE,
    ),
    Entry(
        "nd.lnE",
        r"E\thinspace Q_X(E) = \int d\ln E_n",
        "derived",
        "nd.source per unit ln E",
        file=PAGE,
    ),
    Entry(
        "nd.grid",
        r"w_i = \frac{n_n(E_i)}{\gamma_i\tau_n}",
        "model",
        "AM3's grid sums over ln E (decays.mojo NeutronDecay.Calc_NtoX)",
        file=PAGE,
    ),
    Entry(
        "nd.kernel",
        r"K_{ji} = \int dx\thinspace F_X(x;\beta_i)\thinspace \Lambda_j(x)",
        "model",
        "the products per decay at the grid points "
        "(NeutronDecay.initialize_kernels)",
        file=PAGE,
        checks="tests/unit/test_added_processes.py "
        "(counts and energy per decay)",
    ),
    Entry(
        "nd.hat",
        r"\Lambda_j(x) = \frac{x - x_{j-1}}{x_j - x_{j-1}}",
        "standard",
        "linear interpolation between grid points",
        file=PAGE,
    ),
    Entry(
        "nd.bin",
        r"\mathcal N_j = \omega_k\thinspace\frac{x_b - x_a}{b - a}",
        "derived",
        "the number and energy of a uniform box in an interval "
        "(decays.mojo _bin_box)",
        file=PAGE,
    ),
    Entry(
        "nd.share",
        r"K_j^\mathrm{lo} = \frac{x_{j+1}\mathcal N_j - \mathcal "
        r"E_j}{x_{j+1} - x_j}",
        "derived",
        "the box's integrals against the hat functions "
        "(decays.mojo _share_to_points)",
        file=PAGE,
    ),
    Entry(
        "nd.proton",
        r"s_i = \frac{E_0/m_n}{1 - E_{i-1}/E_i}",
        "derived",
        "the hat functions at x_p (NeutronDecay ps0, ps1)",
        file=PAGE,
        checks="tests/unit/test_added_processes.py (proton shares)",
    ),
]
ALSO = {
    "nd.E0": (
        r"E = (m_n^2 + m_e^2 - m_{p\bar\nu}^2)/(2m_n)",
        r"m_{p\bar\nu} = m_p",
    ),
    "nd.recoil": (
        r"$m_n - m_p - E_0$ = 751.2 eV",
        r"Dubbers & Schmidt (2011) give 750 eV",
    ),
    "nd.hayen4": (r"p = \sqrt{W^2 - 1}",),
    "nd.hayen67": (
        r"\gamma = \sqrt{1 - (\alpha Z)^2}",
        r"y = \pm \alpha Z W / p",
    ),
    "nd.gamma1": (r"\alpha^2 = 5.3 \times 10^{-5}",),
    "nd.limit": (
        r"F_0 \to e^{\pi y}\thinspace\frac{\pi y}{\sinh \pi y} = "
        r"\frac{2\pi y}{1 - e^{-2\pi y}}",
        r"$\Gamma(3)^2 = 4$",
        r"$\vert\Gamma(1 + iy)\vert^2$ "
        r"$= \Gamma(1 + iy)\thinspace\Gamma(1 - iy)$",
        r"\Gamma(1 + z) = z\thinspace\Gamma(z)",
        r"y = \alpha W/p",
    ),
    "nd.spectrum": (
        r"E_{\bar\nu} = E_0 - E",
        r"F(E) = \frac{2\pi\eta}{1 - e^{-2\pi\eta}}",
        r"\eta = \frac{\alpha E}{p}",
        r"W = E/m_e",
        r"W_0 = E_0/m_e",
    ),
    "nd.size": (r"r_p = 0.84075",),
    "nd.Fp": (r"\lim_{p \to 0} F(E)\thinspace p = 2\pi\alpha m_e",),
    "nd.nodes": (
        r"E_0 - E = (E_0 - m_e)\cos^2\frac{\pi u}{2}",
        r"\frac{dE}{du} = \pi\thinspace (E_0 - "
        r"m_e)\sin\frac{\pi u}{2}\cos\frac{\pi u}{2}",
        r"T = E - m_e",
    ),
    "nd.weights": (r"\sum_k \omega_k = 1", r"u_k = (1 + \xi_k)/2"),
    "nd.source": (r"dx = dE/E_n", r"F_X(x;\beta_n) = dn/dx"),
    "nd.box": (
        r"E = \gamma_n(E^\ast + \beta_n p^\ast\cos\theta^\ast)",
        r"x = (E^\ast + \beta_n p^\ast\cos\theta^\ast)/m_n",
    ),
    "nd.antinu": (r"p^\ast = E^\ast",),
    "nd.stable": (
        r"1 - \beta_n = \frac{1}{\gamma_n^2\thinspace (1 + \beta_n)}",
        r"\beta_n = \sqrt{(\gamma_n - 1)(\gamma_n + 1)}/\gamma_n",
        r"E^{\ast 2} - p^{\ast 2} = m_e^2",
    ),
    "nd.means": (
        r"$\langle E\rangle$ = 0.8121 MeV",
        r"$E_0 - \langle E\rangle$ = 0.4805 MeV",
        r"$x_p = 0.99862$",
        r"\langle E_{\bar\nu_e}\rangle/E_n \simeq 5.1 \times 10^{-4}",
    ),
    "nd.ends": (
        r"E_{\bar\nu}^{\max}/E_n$ $\simeq 2(m_n - m_p - "
        r"m_e)/m_n$ $\simeq 1.66 \times 10^{-3}",
    ),
    "nd.lnE": (r"n_n = E_n N_n",),
    "nd.grid": (
        r"q_X(E_j) = \sum_i K_{ji}\thinspace w_i",
        r"\gamma_i = E_i/m_n",
    ),
    "nd.kernel": (r"x_j = \frac{E_j}{E_i}",),
    "nd.hat": (
        r"\Lambda_j(x) = \frac{x_{j+1} - x}{x_{j+1} - x_j}",
        r"\sum_j x_j\Lambda_j(x) = x",
    ),
    "nd.bin": (
        r"\mathcal E_j = \mathcal N_j\thinspace\frac{x_a + x_b}{2}",
        r"x_a = \max(a, x_j)",
        r"x_b = \min(b, x_{j+1})",
    ),
    "nd.share": (
        r"K_j^\mathrm{hi} = "
        r"\frac{\mathcal E_j - x_j\mathcal N_j}{x_{j+1} - x_j}",
    ),
    "nd.proton": (
        r"q_p(E_i) = (1 - s_i)\thinspace w_i + "
        r"s_{i+1}\thinspace w_{i+1}",
    ),
}


def mpf(v):
    return mp.mpf(str(sp.N(v, 60)))


# the code's masses (MeV), alpha and hbar c (MeV fm)
CONST = mojo_constants.read(
    ("MASS_N", "MASS_P", "MASS_E", "ALPHA", "H_PLANCK", "C_LIGHT")
)
with mp.workdps(60):
    M_N, M_P, M_E = (
        mpf(CONST[n] * 1000) for n in ("MASS_N", "MASS_P", "MASS_E")
    )
    ALPHA = mpf(CONST["ALPHA"])
    # h [GeV s] c [cm/s] / 2 pi, MeV fm
    HBARC = mpf(CONST["H_PLANCK"] * CONST["C_LIGHT"] * 10**16) / (2 * mp.pi)
    E0_MP = ((M_N - M_P) * (M_N + M_P) + M_E**2) / (2 * M_N)
ALPHA_X = CONST["ALPHA"]  # exact (sympy)
# PDG 2026, p listing, p. 8, p CHARGE RADIUS: 0.84075 +- 0.00064 fm (MOHR 25,
# 2022 CODATA)
R_P_FM = mp.mpf("0.84075")
# Dubbers & Schmidt 2011 (arXiv:1105.3694v1, a PDF), Sec. VI.A.4, p. 48: "The
# decay protons have an endpoint energy of 750 eV and for detection must be
# accelerated to typically 30 keV."
DUBBERS_END_POINT_EV = 750

# Hayen et al. 2018, from the TeX source (units m_e = c = hbar = 1): eq. 4 (its
# first form, two lines of an align), eq. 6, eq. 7, the definition of p, the
# sign convention and the bound-to-free ratio of the free neutron
HAYEN_EQ4 = (
    r"N(W)dW & = \frac{G_V^2 V_{ud}^2}{2\pi^3} ~ F_{0}(Z, W) ~ L_0(Z, "
    r"W) ~ U(Z, W) ~ D_\text{FS}(Z, W, \beta_2) ~ R(W, W_0) ~ "
    r"R_N(W,W_0, M)  \nonumber \\",
    r"& ~~~~ \times ~ Q(Z, W) ~ S(Z, W) ~ X(Z, W) ~ r(Z, W) ~ C(Z, "
    r"W) ~ D_C(Z, W, \beta_2) ~ p W (W_0 - W)^2 ~ dW  \nonumber \\",
)
HAYEN_EQ6 = (
    r"F_0(Z, W) = 4 (2pR)^{2(\gamma - 1)} e^{\pi y} "
    r"\frac{|\Gamma(\gamma + i y)|^2}{\left( \Gamma(1 + 2\gamma) \right)^2}"
)
HAYEN_EQ7 = r"\gamma = \sqrt{1 - (\alpha Z)^2}, \quad y = \pm \alpha Z W / p"
HAYEN_P = r"$p = \sqrt{W^2 - 1}$ the $\beta$ particle momentum"
HAYEN_SIGN = r"with the upper(lower) sign for electron(positron) decay"
HAYEN_BOUND = (
    r"For free neutron decay, this ratio is approximately $4.2\cdot 10^{-6}$"
)
HAYEN_FEW = (
    r"It absorbs most of the $Z$ and $W$ dependence of the tree-order "
    r"Coulomb interaction, whereas additional corrections are "
    r"typically limited to at most a few percent."
)
# Lipari et al. 2007, Sec. VII.C, from the TeX source
LIPARI_MEAN = (
    r"$\langle E_{\overline{\nu}_e}\rangle/E_n \simeq 5.1 \times 10^{-4}$"
)
LIPARI_END = (
    r"$E^{\rm max}_{\overline{\nu}_e}/E_n \simeq  2 (m_n - m_p - m_e)/m_n",
    r"\simeq 1.66 \times 10^{-3}$",
)

mn, mp_, me = sp.symbols("m_n m_p m_e", positive=True)
E, p, Es, ps = sp.symbols("E p E_s p_s", positive=True)
b = sp.Symbol("beta", positive=True)
c = sp.Symbol("c", real=True)  # cos(theta*) in the neutron's rest frame
eta, al = sp.symbols("eta alpha", positive=True)
# Hayen et al.'s Z, W, W_0, p (units m_e c) and R
Z, W, W0, pH, R = sp.symbols("Z W W_0 p_H R", positive=True)
y, g = sp.symbols("y gamma", positive=True)  # their y and gamma
E0 = sp.Symbol("E_0", positive=True)
x = sp.Symbol("x", positive=True)
GAMMA = {r"\Gamma": sp.gamma}
EXACT = {
    mn: CONST["MASS_N"] * 1000,
    mp_: CONST["MASS_P"] * 1000,
    me: CONST["MASS_E"] * 1000,
}  # the code's, MeV


def flat(tex):
    """A TeX formula without its spacing and layout commands, for comparing a
    transcription with its source."""
    for cmd in (
        r"\nonumber",
        "\\\\",
        r"\times",
        r"\thinspace",
        r"\left",
        r"\right",
        "&",
        "~",
    ):
        tex = tex.replace(cmd, "")
    tex = (
        tex.replace(r"\text{", r"\mathrm{")
        .replace(r"\vert", "|")
        .replace("F_{0}", "F_0")
    )
    return re.sub(r"\s+", "", tex).rstrip(".,")


def number(text):
    mantissa, power = split_number(text)
    return mp.mpf(mantissa) * mp.mpf(10) ** power


# ----------------------------------------------------- the decay at rest
def readme_E0():
    s = L.sides(L.parts(L.display(r"{2m_n} = 1.29258\ \mathrm{MeV}"))[0])
    assert s[0] == "E_0" and s[2] == r"1.29258\ \mathrm{MeV}", s
    return L.to_sympy(s[1], {"m_n": mn, "m_p": mp_, "m_e": me})


def two_body(m_this, m_rest):
    """In the decay of a neutron at rest: (p_n - p)^2 = m_rest^2
    (four-momentum conservation)."""
    En = sp.Symbol("E_two")
    return sp.solve(sp.Eq(mn**2 - 2 * mn * En + m_this**2, m_rest**2), En)[0]


@derivation("nd.E0")
def check_E0():
    """E_0 is the electron's largest energy, two-body kinematics at the smallest
    mass m_p of the rest.
    The page's two-body energy is four-momentum conservation. With the code's
    masses, E_0 is 1.29258 MeV and its kinetic part is 781.6 keV, as printed."""
    M = sp.Symbol("M", positive=True)
    tex = L.sides(L.inline(r"E = (m_n^2 + m_e^2 - m_{p\bar\nu}^2)/(2m_n)"))[1]
    readme_tb = L.to_sympy(tex, {"m_n": mn, "m_e": me, r"m_{p\bar\nu}": M})
    smallest = L.to_sympy(
        L.sides(L.inline(r"m_{p\bar\nu} = m_p"))[1], {"m_p": mp_}
    )
    kinetic = re.search(
        r"Its largest kinetic energy, \$E_0 - m_e\$, is ([\d.]+) keV\.",
        L.line("Its largest kinetic energy, $E_0 - m_e$, is "),
    ).group(1)
    readme = readme_E0()
    with L.patched(
        r"E_0 = \frac{m_n^2 - m_p^2 + m_e^2}{2m_n}",
        r"E_0 = \frac{m_n^2 - m_p^2 - m_e^2}{2m_n}",
    ):
        broken = readme_E0()
    return [
        exact(
            "the page's E(m_p nu) = four-momentum conservation, "
            "(p_n - p_e)^2 = m_p nu^2",
            readme_tb - two_body(me, M),
        ),
        Result(
            "E falls with the mass of the rest: largest at the smallest, "
            "m_p + m_nu = m_p (massless nu)",
            bool(sp.diff(readme_tb, M).is_negative) and smallest == mp_,
            f"dE/dm = {sp.diff(readme_tb, M)}, smallest mass {smallest}",
        ),
        exact(
            "the page's E_0 = E at the smallest mass",
            readme - readme_tb.subs(M, smallest),
        ),
        Result(
            "with the code's masses: E_0 = 1.29258 MeV, as printed",
            mp.nstr(E0_MP, 6) == "1.29258",
            f"E_0 = {mp.nstr(E0_MP, 12)} MeV",
        ),
        Result(
            f"kinetic end point E_0 - m_e = {kinetic} keV, as printed",
            rounds_to(1000 * (E0_MP - M_E), kinetic),
            f"{mp.nstr(1000 * (E0_MP - M_E), 8)} keV",
        ),
        exact_fails(
            "control: m_e^2 with the wrong sign", broken - two_body(me, mp_)
        ),
    ]


@derivation("nd.recoil")
def check_recoil():
    """The proton's m_n - E_0 is its largest energy, against an e nu-bar pair of
    mass m_e, and its kinetic part is 751.2 eV.
    Two-body kinematics, the kinetic part with the code's masses, and Dubbers &
    Schmidt's 750 eV (their p. 48)."""

    def results():
        lhs, rhs = L.sides(
            L.inline(r"m_n - E_0 = (m_n^2 + m_p^2 - m_e^2)/(2m_n)")
        )
        tok = {"m_n": mn, "m_p": mp_, "m_e": me}
        left = L.to_sympy(lhs, {"m_n": mn, "E_0": readme_E0()})
        right = L.to_sympy(rhs, tok)
        line = L.line("Its kinetic part, $m_n - m_p - E_0$ = ")
        kin_lhs, kin = re.search(
            r"\$(m_n - m_p - E_0)\$ = ([\d.]+) eV,", line
        ).groups()
        value = mpf(
            10**6 * L.to_sympy(kin_lhs, {**tok, "E_0": readme_E0()}).subs(EXACT)
        )
        ds = int(
            re.search(
                r"Dubbers & Schmidt \(2011\) give (\d+) eV\.",
                L.line("Dubbers & Schmidt (2011) give "),
            ).group(1)
        )
        return [
            exact(
                "m_n - E_0 = the page's (m_n^2 + m_p^2 - m_e^2)/(2 m_n)",
                left - right,
            ),
            exact(
                "= the proton's energy against an e nu-bar pair of the "
                "smallest mass, m_e + m_nu = m_e",
                right - two_body(mp_, me),
            ),
            Result(
                f"its kinetic part with the code's masses: {kin} eV, "
                "as printed",
                rounds_to(value, kin),
                f"{mp.nstr(value, 10)} eV",
            ),
            Result(
                "Dubbers & Schmidt (p. 48): 750 eV as quoted, and the "
                "kinetic part rounds to it (to 10 eV)",
                ds == DUBBERS_END_POINT_EV and abs(value - ds) < 5,
                f"quoted {ds} eV",
            ),
        ]

    out = [*results()]
    with L.patched(
        r"$m_n - m_p - E_0$ = 751.2 eV", r"$m_n - m_p - E_0$ = 751.4 eV"
    ):
        out.append(must_fail("control: 751.4 eV", results()))
    return out


# ----------------------------------------------------- the spectrum
def readme_eq4():
    line = L.display(r"N(W)\thinspace dW = \frac{G_V^2 V_{ud}^2}{2\pi^3}")
    lhs, rhs = L.sides(L.parts(line)[0])
    assert lhs == r"N(W)\thinspace dW" and rhs.endswith(r"\thinspace dW"), (
        lhs,
        rhs[-20:],
    )
    names = (
        "L_0",
        "U",
        r"D_\mathrm{FS}",
        "R",
        "R_N",
        "Q",
        "S",
        "X",
        "r",
        "C",
        "D_C",
    )
    called = set()

    def correction(name):
        return lambda *a: called.add(name) or 1

    funcs = {
        "F_0": lambda z, w: sp.Function("F0")(z, w),
        **{n: correction(n) for n in names},
    }
    G, V, Mn, b2 = sp.symbols("G_V V_ud M beta_2", positive=True)
    tok = {
        "Z": Z,
        "W": W,
        "W_0": W0,
        "M": Mn,
        r"\beta_2": b2,
        "p": pH,
        "G_V": G,
        "V_{ud}": V,
    }
    return (
        L.to_sympy(rhs[: -len(r"\thinspace dW")], tok, funcs=funcs),
        len(called),
    )


@derivation("nd.hayen4")
def check_hayen4():
    """The page's eq. 4 is Hayen et al.'s eq. 4 (its first form) as printed, and
    so is its p = sqrt(W^2 - 1)."""
    page = lambda: flat(
        L.parts(
            L.display(r"N(W)\thinspace dW = \frac{G_V^2 V_{ud}^2}{2\pi^3}")
        )[0]
    )
    source = flat(HAYEN_EQ4[0] + HAYEN_EQ4[1])
    L.line(
        "Hayen et al. find that the Fermi function holds most of the "
        "Coulomb effect and that its further corrections are typically a "
        "few percent at most."
    )
    out = [
        in_source(
            "Hayen et al. eq. 4 (first form) and the definition of "
            "p as transcribed",
            HAYEN,
            *HAYEN_EQ4,
            HAYEN_P,
        ),
        in_source(
            "their remark on the Fermi function's further corrections "
            "(a few percent at most)",
            HAYEN,
            HAYEN_FEW,
        ),
        Result(
            "the page's eq. 4 = the TeX of eq. 4, factor by factor "
            "(spacing aside)",
            page() == source,
            f"{len(source)} characters"
            if page() == source
            else f"page {page()[:60]}...",
        ),
        Result(
            "the page's p = sqrt(W^2 - 1) is Hayen et al.'s",
            flat(L.inline(r"p = \sqrt{W^2 - 1}"))
            == flat(HAYEN_P.split("$")[1]),
            "as printed",
        ),
    ]
    with L.patched(
        r"p W (W_0 - W)^2\thinspace dW", r"p W (W_0 - W)^3\thinspace dW"
    ):
        out.append(
            Result(
                "control: (W_0 - W)^3 is not Hayen et al.'s eq. 4",
                page() != source,
                "differs",
            )
        )
    return out


@derivation("nd.bound")
def check_bound():
    """The page's 4.2e-6 of the decays bound into hydrogen is Hayen et al.'s
    ratio of bound to free decay."""

    def results():
        txt = re.search(
            r"bound decay into a hydrogen atom, \$([^$]+)\$ of the decays",
            L.line("bound decay into a hydrogen atom"),
        ).group(1)
        src = re.search(
            r"approximately \$([\d.]+)\\cdot 10\^\{(-\d+)\}\$", HAYEN_BOUND
        ).groups()
        return [
            Result(
                "the page's value = Hayen et al.'s",
                number(txt) == mp.mpf(src[0]) * mp.mpf(10) ** int(src[1]),
                f"page {txt}",
            )
        ]

    out = [
        in_source(
            "Hayen et al.'s ratio for the free neutron as transcribed",
            HAYEN,
            HAYEN_BOUND,
        ),
        *results(),
    ]
    with L.patched(
        r"$4.2 \times 10^{-6}$ of the decays",
        r"$4.3 \times 10^{-6}$ of the decays",
    ):
        out.append(must_fail("control: 4.3e-6 on the page", results()))
    return out


def readme_eq67():
    parts = L.parts(L.display(r"F_0(Z, W) = 4 (2pR)^{2(\gamma - 1)}"))
    assert [q.split("=")[0].strip() for q in parts] == [
        "F_0(Z, W)",
        r"\gamma",
        "y",
    ], parts
    return parts


ABS_GAMMA = r"\vert\Gamma(\gamma + i y)\vert^2"


def readme_F0(gamma_value=None):
    """|Gamma(gamma + iy)|^2 is Gamma(gamma + iy) Gamma(gamma - iy) for real
    gamma, y."""
    f0, gam, _ = readme_eq67()
    rhs = L.sides(f0)[1]
    assert rhs.count(ABS_GAMMA) == 1, rhs
    rhs = rhs.replace(
        ABS_GAMMA, r"\Gamma(\gamma + i y)\thinspace\Gamma(\gamma - i y)"
    )
    gval = (
        L.to_sympy(L.sides(gam)[1], {r"\alpha": al, "Z": Z})
        if gamma_value is None
        else gamma_value
    )
    expr = L.to_sympy(
        rhs, {"p": pH, "R": R, r"\gamma": g, "y": y, "i": sp.I}, funcs=GAMMA
    )
    return expr.subs(g, gval).subs(Z, 1)


def readme_y(sign=1):
    """The page's y of eq. 7 (the + instance for electrons)."""
    return L.to_sympy(
        L.sides(readme_eq67()[2])[1],
        {r"\alpha": al, "Z": Z, "W": W, "p": pH},
        sign=sign,
    )


@derivation("nd.hayen67")
def check_hayen67():
    """The page's eqs. 6 and 7 are Hayen et al.'s as printed, and the upper sign
    (electrons) makes F > 1 (attraction)."""

    def same():
        f0, gam, yy = readme_eq67()
        g_src, y_src = flat(HAYEN_EQ7).split(r",\quad")
        return [
            flat(f0) == flat(HAYEN_EQ6),
            flat(gam) == g_src,
            flat(yy) == y_src,
        ]

    ok = same()
    out = [
        in_source(
            "Hayen et al. eqs. 6 and 7 and their sign convention as "
            "transcribed",
            HAYEN,
            HAYEN_EQ6,
            HAYEN_EQ7,
            HAYEN_SIGN,
        ),
        Result(
            "the page's F_0 = eq. 6, gamma and y = eq. 7 "
            "(spacing and \\vert aside)",
            all(ok),
            f"{ok}",
        ),
    ]
    yv = readme_y(+1).subs({Z: 1})
    F1 = sp.lambdify(
        (W, pH),
        (2 * sp.pi * yv / (1 - sp.exp(-2 * sp.pi * yv))).subs(al, ALPHA_X),
        "mpmath",
    )
    rand = rng(5)
    pts = [
        (1 + mp.mpf(10) ** mp.mpf(rand.uniform(-4, 0.5)),) for _ in range(20)
    ]
    vals = [F1(w, mp.sqrt(w**2 - 1)) for (w,) in pts]
    out.append(
        Result(
            "upper sign: y > 0 and F > 1 at every point "
            "(the proton attracts the electron)",
            all(v > 1 for v in vals),
            f"F from {mp.nstr(min(vals), 6)} to {mp.nstr(max(vals), 6)}",
        )
    )
    with L.patched(r"e^{\pi y} \frac{\vert", r"e^{-\pi y} \frac{\vert"):
        out.append(
            Result(
                "control: e^(-pi y) is not Hayen et al.'s eq. 6",
                not same()[0],
                "differs",
            )
        )
    return out


@derivation("nd.gamma1")
def check_gamma1():
    """The page's F is that of the proton's charge, Z = 1, with
    gamma_1 = sqrt(1 - alpha^2) the gamma of eq. 7.
    Setting gamma_1 to 1 neglects the alpha^2 under its root, 5.3e-5 with the
    code's alpha, as printed."""

    def results():
        z = L.to_sympy(
            L.sides(L.after(r"the Fermi function of a point charge $"))[1], {}
        )
        g1 = L.to_sympy(
            L.sides(L.after(r"Hayen et al.'s eq. 6 with their $"))[1],
            {r"\alpha": al},
        )
        g7 = L.to_sympy(L.sides(readme_eq67()[1])[1], {r"\alpha": al, "Z": Z})
        a2_tex, a2_txt = L.sides(L.after("This approximation neglects $"))
        a2 = L.to_sympy(a2_tex, {r"\alpha": al})
        a2_val = mpf(a2.subs(al, ALPHA_X))
        return [
            Result(
                "Z = 1: the charge of the proton, the daughter of n -> "
                "p e- anti-nu_e",
                z == 1,
                f"page Z = {z}",
            ),
            exact(
                "the page's gamma_1 = eq. 7's gamma = sqrt(1 - (alpha "
                "Z)^2) at the page's Z",
                g1 - g7.subs(Z, z),
            ),
            exact(
                "gamma_1 is 1 without the page's alpha^2 under its root: "
                "what setting it to 1 neglects against 1",
                g1.subs(a2, 0) - 1,
            ),
            Result(
                f"alpha^2 = {a2_txt} with the code's alpha, as printed",
                rounds_to(a2_val, a2_txt),
                mp.nstr(a2_val, 8),
            ),
        ]

    out = [
        in_source("Hayen et al. eq. 7 as transcribed", HAYEN, HAYEN_EQ7),
        *results(),
    ]
    with L.patched(
        r"their $\gamma = \sqrt{1 - \alpha^2}$",
        r"their $\gamma = \sqrt{1 + \alpha^2}$",
    ):
        out.append(
            must_fail("control: sqrt(1 + alpha^2) on the page", results())
        )
    with L.patched(r"a point charge $Z = 1$", r"a point charge $Z = 2$"):
        out.append(must_fail("control: Z = 2 on the page", results()))
    with L.patched(
        r"\alpha^2 = 5.3 \times 10^{-5}", r"\alpha^2 = 5.4 \times 10^{-5}"
    ):
        out.append(
            must_fail("control: alpha^2 = 5.4e-5 on the page", results())
        )
    return out


def readme_limit():
    first, second = L.parts(
        L.display(r"\Gamma(1 + iy)\thinspace\Gamma(1 - iy) = \frac")
    )
    tok = {"y": y, "i": sp.I}
    lhs, rhs = (L.to_sympy(s, tok, funcs=GAMMA) for s in L.sides(first))
    head, tail = second.split(r"\to")
    assert head.strip() == "F_0", head
    mid, last = (L.to_sympy(s, tok) for s in L.sides(tail))
    return lhs, rhs, mid, last


def readme_F(eta_value):
    return readme_spectrum()[3].subs(eta, eta_value)


@derivation("nd.limit")
def check_limit():
    """Eq. 6 at Z = 1 with gamma set to 1 is the page's F, by the reflection
    formula of the gamma function.
    The cut-off radius drops out, and Gamma(3)^2 = 4 cancels the 4.
    Gamma(1 + iy) Gamma(1 - iy) = pi y/sinh(pi y), and their y is the page's
    eta."""

    def results():
        lhs, rhs, mid, last = readme_limit()
        f0 = readme_F0(gamma_value=1)
        g3 = L.to_sympy(
            L.sides(L.inline(r"\Gamma(3)^2 = 4"))[0], {}, funcs=GAMMA
        )
        n3 = L.to_sympy(L.sides(L.inline(r"\Gamma(3)^2 = 4"))[1], {})
        z = sp.Symbol("z", positive=True)
        rec = [
            L.to_sympy(s, {"z": z}, funcs=GAMMA)
            for s in L.sides(L.inline(r"\Gamma(1 + z) = z\thinspace\Gamma(z)"))
        ]
        y_line = L.to_sympy(
            L.sides(L.inline(r"y = \alpha W/p"))[1],
            {r"\alpha": al, "W": W, "p": pH},
        )
        z_y = int(
            re.search(
                r"At \$Z = (\d+)\$ their \$y = ",
                L.line("is the Sommerfeld parameter of the electron"),
            ).group(1)
        )
        fac = L.to_sympy(
            re.search(
                r"the factor \$([^$]+)\$ is 1, "
                r"which removes the cut-off radius",
                L.line("which removes the cut-off radius"),
            ).group(1),
            {"p": pH, "R": R, r"\gamma": g},
        )
        f0_gen = readme_F0(gamma_value=g)
        fac6 = f0_gen / f0_gen.subs(pH, 1 / (2 * R))
        return [
            Result(
                "the page's (2pR)^(2(gamma - 1)) is eq. 6's factor with "
                "p and R, and is 1 at gamma = 1",
                sp.simplify(sp.powsimp(fac - fac6, force=True)) == 0
                and fac.subs(g, 1) == 1,
                f"{fac6}",
            ),
            exact("Gamma(3)^2 = 4 as printed", g3 - n3),
            Result("... which is eq. 6's prefactor 4", n3 == 4, f"{n3}"),
            Result(
                "Gamma(1 + z) = z Gamma(z) as printed",
                sp.gammasimp(rec[0] - rec[1]) == 0,
                "gammasimp: 0",
            ),
            Result(
                "the page's Gamma(1 + iy) Gamma(1 - iy) = pi "
                "y/sinh(pi y) (reflection formula)",
                sp.simplify(sp.gammasimp(lhs - rhs)) == 0,
                "gammasimp: 0",
            ),
            Result(
                "eq. 6 at gamma = 1 = the page's e^(pi y) pi "
                "y/sinh(pi y), any R (the identity used)",
                sp.simplify(sp.gammasimp(f0 - mid)) == 0,
                f"eq. 6 at gamma = 1: {sp.gammasimp(f0)}",
            ),
            exact(
                "= 2 pi y/(1 - e^(-2 pi y)), as printed",
                (mid - last).rewrite(sp.exp),
            ),
            exact("= the page's F(E) at eta = y", last - readme_F(y)),
            exact(
                f"their y at the page's Z = {z_y} (upper sign) = its alpha W/p",
                readme_y(+1).subs(Z, z_y) - y_line,
            ),
        ]

    out = [*results()]
    rand = rng(3)
    pts = [(mp.mpf(10) ** mp.mpf(rand.uniform(-3, 1)),) for _ in range(20)]
    out.append(
        numeric(
            "|Gamma(1 + iy)|^2 = Gamma(1 + iy) Gamma(1 - iy) for real y "
            "(mpmath, 50 digits)",
            lambda yy: abs(mp.gamma(1 + 1j * yy)) ** 2,
            lambda yy: (mp.gamma(1 + 1j * yy) * mp.gamma(1 - 1j * yy)).real,
            pts,
        )
    )
    L.line(
        r"$\vert\Gamma(1 + iy)\vert^2$ $= \Gamma(1 + "
        r"iy)\thinspace\Gamma(1 - iy)$"
    )  # the identity as printed
    out.append(
        numeric(
            "... = pi y/sinh(pi y) (mpmath, 50 digits)",
            lambda yy: abs(mp.gamma(1 + 1j * yy)) ** 2,
            lambda yy: mp.pi * yy / mp.sinh(mp.pi * yy),
            pts,
        )
    )
    with L.patched(
        r"\frac{\pi y}{\sinh \pi y},\qquad", r"\frac{\pi y}{\cosh \pi y},\qquad"
    ):
        out.append(must_fail("control: cosh for sinh on the page", results()))
    return out


def readme_spectrum(e0=E0):
    line = L.display(r"\frac{dN}{dE} \propto F(E)")
    lhs, rhs = L.parts(line)[0].split(r"\propto")
    assert lhs.strip() == r"\frac{dN}{dE}", lhs
    Fsym = sp.Function("F")
    rhs_of = lambda name, tokens: L.to_sympy(
        L.sides(L.part(line, name))[1], tokens
    )
    return (
        L.to_sympy(rhs, {"p": p, "E": E, "E_0": e0}, funcs={"F": Fsym}),
        Fsym,
        rhs_of(r"E_{\bar\nu} =", {"E": E, "E_0": e0}),
        rhs_of("F(E) =", {r"\eta": eta}),
        rhs_of(r"\eta =", {r"\alpha": al, "E": E, "p": p}),
    )


def readme_units():
    w = L.to_sympy(L.sides(L.inline(r"W = E/m_e"))[1], {"E": E, "m_e": me})
    w0 = L.to_sympy(
        L.sides(L.inline(r"W_0 = E_0/m_e"))[1], {"E_0": E0, "m_e": me}
    )
    L.line(r"and their $p$ is the electron's momentum over $m_e$")
    return {W: w, W0: w0, pH: p / me}


@derivation("nd.spectrum")
def check_spectrum():
    """The page's spectrum is its eq. 4 without the eleven corrections and with
    F_0 in the limit of nd.limit.
    In energies, dN/dE = N(W)/m_e with W = E/m_e and p/m_e for their p. The
    ratio to the page's dN/dE is a constant."""

    def ratio(spec):
        eq4, n = readme_eq4()
        units = readme_units()
        last = readme_limit()[3]
        z = L.to_sympy(
            L.sides(L.after(r"the Fermi function of a point charge $"))[1], {}
        )
        hayen = (
            eq4.replace(
                sp.Function("F0"), lambda zz, w: last.subs(y, readme_y(+1))
            )
            .subs(Z, z)
            .subs(units)
            / me
        )
        page = spec.replace(Fsym, lambda e: F.subs(eta, eta_))
        return sp.simplify(page / hayen), n

    spec, Fsym, Enu, F, eta_ = readme_spectrum(E0)
    r, n = ratio(spec)
    words = {"ten": 10, "eleven": 11, "twelve": 12}
    said = re.search(
        r"The (\w+) other factors correct the spectrum",
        L.line("other factors correct the spectrum"),
    ).group(1)
    set_to_1 = re.search(
        r"am3_mojo sets all (\w+) factors to 1", L.line("am3_mojo sets all")
    ).group(1)
    out = [
        Result(
            f"the page's eq. 4 has {n} corrections besides F_0, as its "
            f"text says ({said}, all set to 1)",
            words.get(said) == n and set_to_1 == said,
            f"{n} in the equation",
        ),
        Result(
            "the page's dN/dE / (eq. 4 without corrections, in energies) "
            "is a constant (no E, no p)",
            not r.has(E) and not r.has(p),
            f"ratio {r}",
        ),
        exact(
            "E_nu-bar = E_0 - E: the leptons share E_0 "
            "(energy conservation, proton recoil neglected)",
            Enu - (E0 - E),
        ),
        exact(
            "eta = alpha E/p is Hayen et al.'s y at Z = 1 in energies "
            "(W = E/m_e, p -> p/m_e)",
            eta_ - readme_y(+1).subs(Z, 1).subs(readme_units()),
        ),
    ]
    with L.patched(
        r"F(E)\thinspace p\thinspace E\thinspace (E_0 - E)^2",
        r"F(E)\thinspace p\thinspace E\thinspace (E_0 - E)^3",
    ):
        rb = ratio(readme_spectrum(E0)[0])[0]
    out.append(
        Result("control: (E_0 - E)^3 is not eq. 4", rb.has(E), f"ratio {rb}")
    )
    return out


@derivation("nd.size")
def check_size():
    """Eq. 6 at Z = 1 with gamma_1 and the proton's size is within 10^-3 of the
    page's F above 0.1 keV.
    R = sqrt(5/3) r_p is the radius of a uniform sphere of charge radius r_p
    (PDG 2026)."""
    rr, rp = sp.symbols("r R_s", positive=True)
    mean_r2 = sp.integrate(rr**4, (rr, 0, rp)) / sp.integrate(
        rr**2, (rr, 0, rp)
    )
    r_p = sp.Symbol("r_p", positive=True)
    R_page = L.to_sympy(
        L.sides(L.inline(r"R = \sqrt{5/3}\thinspace r_p"))[1], {"r_p": r_p}
    )
    rp_page = number(L.sides(L.inline(r"r_p = 0.84075"))[1])

    def worst():
        m = re.search(
            r"changes \$F\$ by less than \$10\^\{(-\d+)\}\$ for "
            r"kinetic energies above ([\d.]+) keV",
            L.line("for kinetic energies above"),
        )
        # MeV
        bound, t_min = mp.mpf(10) ** int(m.group(1)), mp.mpf(m.group(2)) / 1000
        # in units hbar/(m_e c)
        R_units = mpf(R_page.subs(r_p, sp.Rational(str(rp_page)))) * M_E / HBARC
        F0 = sp.lambdify((y, pH, R), readme_F0().subs(al, ALPHA_X), "mpmath")
        Fp = sp.lambdify(eta, readme_F(eta), "mpmath")
        t = [
            t_min * (((E0_MP - M_E) / t_min) ** (mp.mpf(k) / 40))
            for k in range(41)
        ]
        dev = 0
        for T in t:
            w = 1 + T / M_E
            ph = mp.sqrt(w**2 - 1)
            yy = ALPHA * w / ph
            dev = worst_of((dev, abs(mp.re(F0(yy, ph, R_units)) / Fp(yy) - 1)))
        return dev, bound

    with mp.workdps(30):
        dev, bound = worst()
        out = [
            exact(
                "a uniform sphere of radius R has <r^2> = 3R^2/5: the page's "
                "R = sqrt(5/3) r_p has <r^2> = r_p^2",
                mean_r2.subs(rp, R_page) - r_p**2,
            ),
            Result(
                "r_p = 0.84075 fm: the PDG 2026 p listing (p. 8, MOHR 25)",
                rp_page == R_P_FM,
                f"{rp_page} fm",
            ),
            Result(
                "eq. 6 at Z = 1 (gamma_1, R) / F - 1 stays below the page's "
                "bound above its 0.1 keV",
                dev < bound,
                f"at most {mp.nstr(dev, 3)} at 41 energies from 0.1 keV to "
                "the end point",
            ),
        ]
        with L.patched(r"by less than $10^{-3}$", r"by less than $10^{-4}$"):
            dev_b, bound_b = worst()
        out.append(
            Result(
                "control: a bound of 10^-4 fails",
                not dev_b < bound_b,
                f"{mp.nstr(dev_b, 3)} > 1e-4",
            )
        )
    return out


@derivation("nd.Fp")
def check_Fp():
    """F p = 2 pi alpha E/(1 - e^(-2 pi alpha E/p)) with the page's F and eta,
    finite (2 pi alpha m_e) at p -> 0."""

    def results():
        first, second = L.parts(
            L.display(r"F(E)\thinspace p = \frac{2\pi\alpha E}")
        )
        lhs, rhs = L.sides(first)
        assert lhs == r"F(E)\thinspace p", lhs
        form = L.to_sympy(rhs, {r"\alpha": al, "E": E, "p": p})
        lim_lhs, lim_rhs = L.sides(second)
        assert lim_lhs == r"\lim_{p \to 0} F(E)\thinspace p", lim_lhs
        limit = L.to_sympy(lim_rhs, {r"\alpha": al, "m_e": me})
        Fp = readme_F(readme_spectrum()[4]) * p
        shell = form.subs(E, sp.sqrt(p**2 + me**2))
        grows = L.to_sympy(
            re.search(
                r"grows like \$([^$]+)\$ at the bottom", L.line("grows like")
            ).group(1),
            {"p": p},
        )
        return [
            exact(
                "the page's F(eta) p with eta = alpha E/p = its 2 pi "
                "alpha E/(1 - e^(-2 pi alpha E/p))",
                Fp - form,
            ),
            exact(
                "lim p -> 0 at E = sqrt(p^2 + m_e^2): the page's 2 "
                "pi alpha m_e",
                sp.limit(shell, p, 0, "+") - limit,
            ),
            exact(
                "F alone grows like the page's p^(-1) there: F/p^(-1) -> "
                "2 pi alpha m_e",
                sp.limit(shell / p / grows, p, 0, "+") - limit,
            ),
        ]

    out = [*results()]
    with L.patched(r"{1 - e^{-2\pi\alpha E/p}}", r"{1 + e^{-2\pi\alpha E/p}}"):
        out.append(must_fail("control: 1 + e^(...) on the page", results()))
    return out


def readme_nodes():
    parts = L.parts(L.display(r"E = m_e + (E_0 - m_e)\sin^2\frac{\pi u}{2}"))
    u = sp.Symbol("u", positive=True)
    tok = {"m_e": me, "E_0": E0, "u": u}
    sides = [L.sides(q) for q in parts]
    assert [s[0] for s in sides] == ["E", "E_0 - E", r"\frac{dE}{du}"], sides
    return u, [L.to_sympy(s[1], tok) for s in sides]


@derivation("nd.nodes")
def check_nodes():
    """The substitution E(u) maps u in [0, 1] onto the spectrum, and the page's
    E_0 - E and dE/du follow from it.
    T = E - m_e and E_0 - E are products of positive factors (no
    cancellation)."""

    def results():
        u, (Eu, rest, dEdu) = readme_nodes()
        T = L.to_sympy(
            L.sides(L.inline(r"T = E - m_e"))[1], {"E": Eu, "m_e": me}
        )
        return [
            exact(
                "E(0) = m_e and E(1) = E_0",
                (Eu.subs(u, 0) - me) ** 2 + (Eu.subs(u, 1) - E0) ** 2,
            ),
            exact(
                "the page's E_0 - E = E_0 - E(u) (sin^2 + cos^2 = 1)",
                sp.trigsimp(rest - (E0 - Eu)),
            ),
            exact("the page's dE/du = d E(u)/du", dEdu - sp.diff(Eu, u)),
            Result(
                "T = E - m_e and E_0 - E are products of positive "
                "factors (no difference of near numbers)",
                sp.factor(T).is_Mul and sp.factor(rest).is_Mul,
                f"T = {sp.factor(T)}, E_0 - E = {sp.factor(rest)}",
            ),
        ]

    out = [*results()]
    with L.patched(
        r"E_0 - E = (E_0 - m_e)\cos^2\frac{\pi u}{2}",
        r"E_0 - E = (E_0 - m_e)\sin^2\frac{\pi u}{2}",
    ):
        out.append(must_fail("control: sin^2 for cos^2 on the page", results()))
    return out


@derivation("nd.weights")
def check_weights():
    """The weights a_k/2 dE/du dN/dE are the Gauss-Legendre rule after
    u = (1 + xi)/2: they integrate dN/dE dE."""

    def results():
        xi = sp.Symbol("xi", real=True)
        u_page = L.to_sympy(
            L.sides(L.inline(r"u_k = (1 + \xi_k)/2"))[1], {r"\xi_k": xi}
        )
        lo, hi = (
            int(v)
            for v in re.search(
                r"for \$(-?\d+) \\le \\xi \\le (-?\d+)\$, node",
                L.line(r"\le \xi \le"),
            ).groups()
        )
        weight = L.parts(L.display(r"\omega_k \propto \frac{a_k}"))
        assert weight[1] == r"\sum_k \omega_k = 1", weight
        factor = L.to_sympy(
            re.match(
                r"\\omega_k \\propto (\\frac\{a_k\}\{\d\})\\thinspace",
                weight[0],
            ).group(1),
            {"a_k": sp.Integer(1)},
        )
        u, (Eu, _, dEdu) = readme_nodes()
        jac = factor * dEdu.subs(u, u_page)
        Eval = Eu.subs(u, u_page)
        num = {me: 1, E0: 3}
        integral = lambda f: sp.integrate(
            sp.expand_trig((jac * f).subs(num)), (xi, lo, hi)
        )
        return [
            exact(
                f"the rule's interval [{lo}, {hi}] maps onto u in [0, 1]",
                u_page.subs(xi, lo) ** 2 + (u_page.subs(xi, hi) - 1) ** 2,
            ),
            exact(
                "the page's factor 1/2 = du/dxi", factor - sp.diff(u_page, xi)
            ),
            exact(
                "sum of weights x 1: the integral of dE over the "
                "spectrum, E_0 - m_e",
                integral(1) - 2,
            ),
            exact(
                "... x E: the integral of E dE, (E_0^2 - m_e^2)/2",
                integral(Eval) - 4,
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\omega_k \propto \frac{a_k}{2}", r"\omega_k \propto \frac{a_k}{1}"
    ):
        out.append(must_fail("control: a_k without the 1/2", results()))
    return out


# ----------------------------------------------------- the lab frame
def readme_gamma_n(En):
    return L.to_sympy(
        L.sides(L.inline(r"\gamma_n = E_n/m_n"))[1], {"E_n": En, "m_n": mn}
    )


def readme_source(En, Nn, FX, beta):
    """Q_X and N_n are per unit energy, as the text says."""
    L.line(
        r"where $N_n$ is the neutron density per unit energy, $Q_{X}$ the "
        r"source of product $X$ per unit energy"
    )
    lhs, rhs = L.sides(
        L.parts(L.display(r"\frac{N_n(E_n)}{\gamma_n\thinspace \tau_n}"))[0]
    )
    assert lhs == "Q_X(E)", lhs
    measure, body = (
        L.between(rhs, r"\int ", r"\thinspace "),
        rhs.split(r"\thinspace ", 1)[1],
    )
    dEn = sp.Symbol("dE_n")
    m = L.to_sympy(measure, {"dE_n": dEn, "E_n": En})
    weight = sp.diff(m, dEn)
    assert sp.simplify(m - weight * dEn) == 0, measure
    tau = sp.Symbol("tau_n", positive=True)
    tokens = {
        "E_n": En,
        "E": E,
        r"\gamma_n": readme_gamma_n(En),
        r"\tau_n": tau,
        r"\beta_n": beta,
    }
    # F_X(x; beta_n): its parameter
    integrand = L.to_sympy(
        body.replace(";", ","), tokens, funcs={"N_n": Nn, "F_X": FX}
    )
    return weight, integrand


@derivation("nd.source")
def check_source():
    """Q_X counts the decays: N_n dE_n neutrons decay at the time-dilated rate,
    and each decay makes F_X dx products.
    x = E/E_n and dx = dE/E_n at a fixed E_n."""
    En, tau = sp.symbols("E_n tau_n", positive=True)
    Nn, FX = sp.Function("N_n"), sp.Function("F_X")
    beta = sp.Function("beta_n")(En)
    gamma = En / mn
    xx = L.to_sympy(L.sides(L.inline(r"x = E/E_n"))[1], {"E": E, "E_n": En})
    dE = sp.Symbol("dE")
    dx = L.to_sympy(L.sides(L.inline(r"dx = dE/E_n"))[1], {"dE": dE, "E_n": En})
    per_decay = L.sides(L.inline(r"F_X(x;\beta_n) = dn/dx"))
    # The rate is 1/(gamma tau_n) (lab mean life gamma tau_n). dx = (dx/dE) dE
    # at a fixed E_n gives the products per unit E.
    counted = Nn(En) / (gamma * tau) * FX(xx, beta) * sp.diff(xx, E)
    weight, integrand = readme_source(En, Nn, FX, beta)
    rate = L.to_sympy(
        L.inline(r"1/(\gamma_n\tau_n)"),
        {r"\gamma_n": readme_gamma_n(En), r"\tau_n": tau},
    )
    out = [
        exact(
            "the page's Q_X = the count of the decays per unit energy "
            "(any N_n, F_X; integrands in E_n)",
            weight * integrand - counted,
        ),
        exact(
            "the page's dx = dE/E_n: x = E/E_n at a fixed E_n",
            dx - sp.diff(xx, E) * dE,
        ),
        Result(
            "F_X is the page's dn/dx, the products per decay and unit x "
            "(the definition used)",
            per_decay == [r"F_X(x;\beta_n)", "dn/dx"],
            " = ".join(per_decay),
        ),
        exact(
            "the page's rate 1/(gamma_n tau_n) is the time-dilated one, "
            "gamma_n = E_n/m_n (c = 1)",
            rate - 1 / (gamma * tau),
        ),
    ]
    for old, new, what in (
        (
            r"Q_X(E) = \int \frac{dE_n}{E_n}\thinspace",
            r"Q_X(E) = \int dE_n\thinspace",
            "dE_n for dE_n/E_n",
        ),
        (
            r"\left(\frac{E}{E_n};\beta_n\right)",
            r"\left(\frac{E_n}{E};\beta_n\right)",
            "F_X(E_n/E)",
        ),
    ):
        with L.patched(old, new):
            w_b, i_b = readme_source(En, Nn, FX, beta)
        out.append(exact_fails(f"control: {what}", w_b * i_b - counted))
    return out


def readme_box():
    tex = L.part(
        L.display(r"x \in \left[\frac{E^\ast - \beta_n p^\ast}{m_n}"), r"x \in"
    )
    assert tex.startswith(r"x \in \left[") and tex.endswith(r"\right]"), tex
    lo, hi = L.split_top(L.between(tex, r"\left[", r"\right]"), r",\ ")
    tok = {r"E^\ast": Es, r"\beta_n": b, r"p^\ast": ps, "m_n": mn}
    return L.to_sympy(lo, tok), L.to_sympy(hi, tok)


@derivation("nd.box")
def check_box():
    """The boost makes x = E/E_n linear in cos theta*: uniform over the page's
    range for an isotropic decay."""
    gam = 1 / sp.sqrt(1 - b**2)
    tok = {
        r"\gamma_n": gam,
        r"E^\ast": Es,
        r"\beta_n": b,
        r"p^\ast": ps,
        r"\cos\theta^\ast": c,
        "m_n": mn,
    }
    E_page = L.to_sympy(
        L.sides(
            L.inline(
                r"E = \gamma_n(E^\ast + \beta_n "
                r"p^\ast\cos\theta^\ast)"
            )
        )[1],
        tok,
    )
    x_page = L.to_sympy(
        L.sides(L.inline(r"x = (E^\ast + \beta_n p^\ast\cos\theta^\ast)/m_n"))[
            1
        ],
        tok,
    )
    boost = sp.Matrix([[gam, gam * b], [gam * b, gam]]) * sp.Matrix(
        [Es, ps * c]
    )
    Ens = sp.Symbol("E_n", positive=True)
    En = sp.solve(sp.Eq(readme_gamma_n(Ens), gam), Ens)[0]
    ends = lambda lo, hi: (
        (x_page.subs(c, -1) - lo) ** 2 + (x_page.subs(c, 1) - hi) ** 2
    )
    lo, hi = readme_box()
    with L.patched(
        r"\frac{E^\ast + \beta_n p^\ast}{m_n}\right]",
        r"\frac{E^\ast + 2\beta_n p^\ast}{m_n}\right]",
    ):
        broken = readme_box()
    return [
        exact(
            "the page's lab energy = the Lorentz boost along the "
            "neutron's direction",
            E_page - boost[0],
        ),
        exact(
            "the page's x = E/E_n with E_n from its gamma_n = E_n/m_n (c = 1)",
            x_page - E_page / En,
        ),
        Result(
            "x is linear in cos theta* (so uniform for an isotropic decay)",
            sp.diff(x_page, c, 2) == 0,
            f"x = {sp.simplify(x_page)}",
        ),
        exact("x at cos theta* = -1, +1: the page's ends", ends(lo, hi)),
        exact_fails("control: an upper end E* + 2 beta p*", ends(*broken)),
    ]


@derivation("nd.density")
def check_density():
    """The density over the range: (1/2)/(dx/dcos theta*) = m_n/(2 beta_n
    p*)."""
    xx = (Es + b * ps * c) / mn
    density = lambda tex: L.to_sympy(
        L.inline(tex), {"m_n": mn, r"\beta_n": b, r"p^\ast": ps}
    )
    with L.patched(
        r"with density $m_n/(2\beta_n p^\ast)$",
        r"with density $m_n/(\beta_n p^\ast)$",
    ):
        broken = density(r"m_n/(\beta_n p^\ast)")
    return [
        exact(
            "(1/2)/(dx/dc) = the page's density",
            sp.Rational(1, 2) / sp.diff(xx, c)
            - density(r"m_n/(2\beta_n p^\ast)"),
        ),
        exact_fails(
            "control: m_n/(beta_n p*)",
            sp.Rational(1, 2) / sp.diff(xx, c) - broken,
        ),
    ]


def readme_FX():
    lhs, rhs = L.sides(
        L.parts(
            L.display(r"F_X(x;\beta_n) = \int dE^\ast\thinspace f_X(E^\ast)")
        )[0]
    )
    assert lhs == r"F_X(x;\beta_n)" and rhs.startswith(
        r"\int dE^\ast\thinspace f_X(E^\ast)\thinspace "
    ), rhs
    factor = L.between(rhs, r"f_X(E^\ast)\thinspace ", r"\thinspace H")
    arg = L.between(rhs, r"H\negthinspace\left(", r"\right)")
    absval = r"\vert m_n x - E^\ast\vert"
    assert arg.count(absval) == 1, arg
    tok = {
        "m_n": mn,
        r"\beta_n": b,
        r"p^\ast": ps,
        r"E^\ast": Es,
        "x": x,
        "A": sp.Abs(mn * x - Es),
    }
    return L.to_sympy(factor, tok), L.to_sympy(arg.replace(absval, "A"), tok)


@derivation("nd.FX")
def check_FX():
    """F_X sums the boxes of nd.box: its factor is the density, H is 1 on the
    range, and a box holds one product."""

    def results():
        factor, arg = readme_FX()
        lo, hi = readme_box()
        dens = L.to_sympy(
            L.inline(r"m_n/(2\beta_n p^\ast)"),
            {"m_n": mn, r"\beta_n": b, r"p^\ast": ps},
        )
        return [
            exact(
                "the factor = the page's density m_n/(2 beta_n p*)",
                factor - dens,
            ),
            exact(
                "H's argument vanishes at the page's ends of the range",
                arg.subs(x, lo) ** 2 + arg.subs(x, hi) ** 2,
            ),
            exact(
                "... and is beta_n p* > 0 in its middle, E*/m_n",
                arg.subs(x, (lo + hi) / 2) - b * ps,
            ),
            exact(
                "one product per box: the integral of the factor over "
                "the range is 1",
                sp.integrate(factor, (x, lo, hi)) - 1,
            ),
            exact(
                "its mean share is E*/m_n (the middle of the range)",
                sp.integrate(x * factor, (x, lo, hi)) - Es / mn,
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\frac{m_n}{2\beta_n p^\ast}\thinspace H",
        r"\frac{m_n}{\beta_n p^\ast}\thinspace H",
    ):
        out.append(must_fail("control: m_n/(beta_n p*) in F_X", results()))
    return out


@derivation("nd.xp")
def check_xp():
    """The proton takes the page's m_n - E_0 (energy conservation), at rest in
    the neutron's frame:
    x_p = (m_n - E_0)/m_n."""
    E0s = sp.Symbol("E_0", positive=True)
    xp = L.to_sympy(
        L.sides(L.inline(r"x_p = (m_n - E_0)/m_n"))[1], {"m_n": mn, "E_0": E0s}
    )
    Ep = L.to_sympy(
        L.after("The proton takes the remaining $"), {"m_n": mn, "E_0": E0s}
    )
    lo, hi = readme_box()
    with L.patched(r"$x_p = (m_n - E_0)/m_n$", r"$x_p = (m_n + E_0)/m_n$"):
        broken = L.to_sympy(
            L.sides(L.inline(r"x_p = (m_n + E_0)/m_n"))[1],
            {"m_n": mn, "E_0": E0s},
        )
    return [
        exact(
            "the proton's m_n - E_0: m_n less the electron's and "
            "antineutrino's E_0",
            Ep - (mn - E0s),
        ),
        exact(
            "a product at rest (p* = 0) has the single x = E*/m_n "
            "(the range of nd.box)",
            (lo - hi).subs(ps, 0),
        ),
        exact("x_p = (m_n - E_0)/m_n", xp - lo.subs({ps: 0, Es: Ep})),
        exact_fails(
            "control: x_p = (m_n + E_0)/m_n", broken - lo.subs({ps: 0, Es: Ep})
        ),
    ]


def readme_antinu():
    tex = L.part(L.display(r"x \in \left[\frac{(1 - \beta_n"), r"x \in")
    assert tex.startswith(r"x \in \left[") and tex.endswith(r"\right]"), tex
    lo, hi = L.split_top(L.between(tex, r"\left[", r"\right]"), r",\ ")
    Enu = sp.Symbol("E_nu", positive=True)
    tok = {r"E_{\bar\nu}": Enu, r"\beta_n": b, "m_n": mn}
    return Enu, L.to_sympy(lo, tok), L.to_sympy(hi, tok)


@derivation("nd.antinu")
def check_antinu():
    """The antineutrino's range is nd.box's for a massless product,
    p* = E* = E_nu-bar."""

    def results():
        Enu, lo, hi = readme_antinu()
        star = L.to_sympy(
            L.sides(L.inline(r"p^\ast = E^\ast"))[1], {r"E^\ast": Es}
        )
        blo, bhi = readme_box()
        sub = {ps: star}
        return [
            exact(
                "massless: E*^2 - p*^2 = 0 with the page's p* = E*",
                Es**2 - star**2,
            ),
            exact(
                "nd.box's ends at p* = E* = E_nu-bar: the page's range",
                (blo.subs(sub).subs(Es, Enu) - lo) ** 2
                + (bhi.subs(sub).subs(Es, Enu) - hi) ** 2,
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\frac{(1 - \beta_n)\thinspace E_{\bar\nu}}{m_n}",
        r"\frac{(1 - \beta_n^2)\thinspace E_{\bar\nu}}{m_n}",
    ):
        out.append(must_fail("control: (1 - beta_n^2) on the page", results()))
    return out


@derivation("nd.stable")
def check_stable():
    """The lower ends without cancellation: the page's two forms are exact
    identities."""
    v, t = (
        (1 + sp.Symbol("w", positive=True)),
        (1 + sp.Symbol("s", positive=True)),
    )
    Ev, pv = me * (v**2 + v**-2) / 2, me * (v**2 - v**-2) / 2
    gv = (t**2 + t**-2) / 2

    def results():
        first, second = L.parts(
            L.display(r"E^\ast - \beta_n p^\ast = \frac{m_e")
        )
        gm = sp.Symbol("gamma_n", positive=True)
        tok = {
            r"E^\ast": Ev,
            r"p^\ast": pv,
            r"\beta_n": b,
            "m_e": me,
            r"\gamma_n": gm,
        }
        beta = L.to_sympy(
            L.sides(
                L.inline(
                    r"\beta_n = \sqrt{(\gamma_n - 1)(\gamma_n + 1)}/\gamma_n"
                )
            )[1],
            {r"\gamma_n": gm},
        )
        shell = [
            L.to_sympy(
                s, {r"E^{\ast 2}": Ev**2, r"p^{\ast 2}": pv**2, "m_e": me}
            )
            for s in L.sides(L.inline(r"E^{\ast 2} - p^{\ast 2} = m_e^2"))
        ]
        l1, r1 = (L.to_sympy(s, tok) for s in L.sides(first))
        l2, r2 = (L.to_sympy(s, tok) for s in L.sides(second))
        bv = beta.subs(gm, gv)
        return [
            exact(
                "the parametrisation is on the page's mass shell E*^2 - "
                "p*^2 = m_e^2",
                shell[0] - shell[1],
            ),
            exact(
                "the page's beta_n = sqrt(1 - 1/gamma_n^2)",
                (beta**2 - (1 - 1 / gm**2)).subs(gm, gv),
            ),
            exact(
                "E* - beta p* = m_e^2/(E* + p*) + (1 - beta) p*, any beta",
                l1 - r1,
            ),
            exact(
                "1 - beta = 1/(gamma^2 (1 + beta)) with the page's beta(gamma)",
                (l2 - r2).subs(b, bv).subs(gm, gv),
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\frac{m_e^2}{E^\ast + p^\ast}", r"\frac{m_e}{E^\ast + p^\ast}"
    ):
        out.append(must_fail("control: m_e for m_e^2 on the page", results()))
    return out


def spectrum_mean():
    spec, Fsym, _, F, eta_ = readme_spectrum(E0)
    num = {al: ALPHA_X, me: EXACT[me], E0: readme_E0().subs(EXACT)}
    dens = sp.lambdify(
        E,
        spec.replace(Fsym, lambda e: F.subs(eta, eta_))
        .subs(p, sp.sqrt(E**2 - me**2))
        .subs(num),
        "mpmath",
    )
    with mp.workdps(50):
        norm = mp.quad(dens, [M_E, E0_MP])
        return mp.quad(lambda e: e * dens(e), [M_E, E0_MP]) / norm


@derivation("nd.means")
def check_means():
    """The mean shares are <E*>/m_n at any beta, and the page's 5.11e-4,
    8.64e-4 and x_p = 0.99862 add up to 1.
    <E> = 0.8121 MeV from the page's spectrum at 50 digits. Lipari et al. give
    5.1e-4."""
    with mp.workdps(50):
        mean = spectrum_mean()

    def results():
        mid = L.to_sympy(
            L.after("Each range has its middle at $"),
            {r"E^\ast": Es, "m_n": mn},
        )
        L.line(
            r"a product therefore has "
            r"$\langle x\rangle = \langle E^\ast\rangle/m_n$ for any $\beta_n$"
        )
        lo, hi = readme_box()
        e_txt, nu_txt = re.search(
            r"the mean energy \$\\langle E\\rangle\$ = ([\d.]+) MeV, and the "
            r"antineutrino \$E_0 - \\langle E\\rangle\$ = ([\d.]+) MeV",
            L.line("the mean energy"),
        ).groups()
        m = re.search(
            r"the antineutrinos take \$([^$]+)\$ of the neutron's energy and "
            r"the electrons \$([^$]+)\$",
            L.line("on average the antineutrinos take"),
        )
        xp_txt = L.sides(L.inline(r"x_p = 0.99862"))[1]
        xp = L.to_sympy(
            L.sides(L.inline(r"x_p = (m_n - E_0)/m_n"))[1],
            {"m_n": mn, "E_0": E0},
        )
        Enu = readme_spectrum(E0)[2]
        # the antineutrino's mean energy from the page's
        # E_nu-bar = E_0 - E (linear in E), at the spectrum's mean
        Ebar = sp.Symbol("Ebar")
        with mp.workdps(50):
            nu_mean = sp.lambdify((E0, E), Enu, "mpmath")(E0_MP, mean)
            xp_val = mpf(xp.subs(E0, readme_E0()).subs(EXACT))
        total = xp + (Ebar + Enu.subs(E, Ebar)) / mn
        lip = L.sides(
            L.inline(
                r"\langle E_{\bar\nu_e}\rangle/E_n \simeq 5.1 "
                r"\times 10^{-4}"
            ).replace(r"\simeq", "=")
        )[1]
        return [
            exact(
                "the mean share over the range is its middle, the page's "
                "E*/m_n (any beta)",
                (lo + hi) / 2 - mid,
            ),
            Result(
                f"<E> = {e_txt} MeV from the page's spectrum, as printed",
                rounds_to(mean, e_txt),
                f"<E> = {mp.nstr(mean, 10)} MeV",
            ),
            Result(
                f"<E_nu-bar> = E_0 - <E> = {nu_txt} MeV, as printed",
                rounds_to(nu_mean, nu_txt),
                f"{mp.nstr(nu_mean, 10)} MeV",
            ),
            Result(
                f"antineutrino share {m.group(1)}, electron share "
                f"{m.group(2)}, as printed",
                rounds_to(nu_mean / M_N, m.group(1))
                and rounds_to(mean / M_N, m.group(2)),
                f"{mp.nstr(nu_mean / M_N, 6)}, {mp.nstr(mean / M_N, 6)}",
            ),
            Result(
                f"x_p = {xp_txt}, as printed",
                rounds_to(xp_val, xp_txt),
                f"x_p = {mp.nstr(xp_val, 10)}",
            ),
            exact(
                "x_p + <x_e> + <x_nu> = 1 (energy per decay; any mean <E>)",
                total - 1,
            ),
            Result(
                "Lipari et al.'s 5.1e-4 as quoted, and the share rounds to it",
                number(lip) == mp.mpf("5.1e-4")
                and rounds_to(nu_mean / M_N, lip),
                f"quoted {lip}",
            ),
        ]

    out = [
        in_source(
            "Lipari et al.'s mean share as transcribed", LIPARI, LIPARI_MEAN
        ),
        *results(),
    ]
    with L.patched(
        r"$\langle E\rangle$ = 0.8121 MeV", r"$\langle E\rangle$ = 0.8131 MeV"
    ):
        out.append(
            must_fail("control: <E> = 0.8131 MeV on the page", results())
        )
    return out


@derivation("nd.ends")
def check_ends():
    """The antineutrino reaches x = 2(E_0 - m_e)/m_n = 1.664e-3 for beta -> 1,
    and Lipari et al.'s form leaves out the recoil. Their 2(m_n - m_p - m_e)/m_n
    puts m_n - m_p for E_0, and exceeds ours by twice the largest recoil over
    m_n."""

    def results():
        line = L.line(
            "For an ultra-relativistic neutron the antineutrino takes up to"
        )
        formula, val = re.search(
            r"takes up to \$x = ([^$=]+) = ([^$=]+)\$ of "
            r"the energy",
            line,
        ).groups()
        tok = {"E_0": E0, "m_e": me, "m_n": mn, "m_p": mp_}
        page = L.to_sympy(formula, tok)
        Enu, _, hi = readme_antinu()
        nu_max = readme_spectrum(E0)[2].subs(E, me)
        quote = L.line(
            r"E_{\bar\nu}^{\max}/E_n$ $\simeq 2(m_n - m_p - "
            r"m_e)/m_n$ $\simeq 1.66 \times 10^{-3}"
        )
        lip_formula, lip_val = re.search(
            r"E_\{\\bar\\nu\}\^\{\\max\}/E_n\$ \$\\simeq ([^$]+)\$ "
            r"\$\\simeq ([^$]+)\$",
            quote,
        ).groups()
        lip = L.to_sympy(lip_formula, tok)
        e0 = readme_E0()
        with mp.workdps(30):
            v_page, v_lip = (
                mpf(page.subs(E0, e0).subs(EXACT)),
                mpf(lip.subs(EXACT)),
            )
        return [
            exact(
                "the upper end of the page's antineutrino range at beta = 1 "
                "and the largest E_nu-bar",
                hi.subs({b: 1, Enu: nu_max}) - page,
            ),
            Result(
                f"= {val} with the code's masses, as printed",
                rounds_to(v_page, val),
                mp.nstr(v_page, 8),
            ),
            exact(
                "Lipari et al.'s form = ours with the page's m_n - m_p for "
                "E_0 (the recoil left out)",
                lip
                - page.subs(
                    E0,
                    L.to_sympy(
                        re.search(
                            r"with \$([^$]+)\$ for \$E_0\$, which leaves out "
                            r"the recoil",
                            quote,
                        ).group(1),
                        tok,
                    ),
                ),
            ),
            exact(
                "... so it exceeds ours by 2 (m_n - m_p - E_0)/m_n, "
                "twice the largest recoil over m_n",
                lip - page.subs(E0, e0) - 2 * (mn - mp_ - e0) / mn,
            ),
            Result(
                f"their {lip_val} as quoted; ours, {mp.nstr(v_page, 4)}, "
                f"rounds to it (their form gives {mp.nstr(v_lip, 4)})",
                number(lip_val) == mp.mpf("1.66e-3")
                and rounds_to(v_page, lip_val),
                f"quoted {lip_val}",
            ),
        ]

    out = [
        in_source(
            "Lipari et al.'s end point as transcribed", LIPARI, *LIPARI_END
        ),
        *results(),
    ]
    with L.patched(r"= 1.664 \times 10^{-3}", r"= 1.665 \times 10^{-3}"):
        out.append(must_fail("control: 1.665e-3 on the page", results()))
    return out


# ----------------------------------------------------- the sources on the grids
def readme_lnE(En, nn, FX, beta, tau, gamma):
    first, second = L.parts(L.display(r"E\thinspace Q_X(E) = \int d\ln E_n"))
    lhs, rhs = L.sides(first)
    assert lhs == r"E\thinspace Q_X(E)" and rhs.startswith(
        r"\int d\ln E_n\thinspace "
    ), (lhs, rhs)
    body = rhs[len(r"\int d\ln E_n\thinspace ") :].replace(";", ",")
    tok = {
        "x": E / En,
        r"\gamma_n": gamma,
        r"\tau_n": tau,
        r"\beta_n": beta,
        "E_n": En,
    }
    nlhs, nrhs = L.sides(second)
    assert nlhs == "n_n", nlhs
    return L.to_sympy(body, tok, funcs={"n_n": nn, "F_X": FX}), nrhs


@derivation("nd.lnE")
def check_lnE():
    """E Q_X(E) per d ln E_n is the page's Q_X times E, with dE_n/E_n = d ln E_n
    and n_n = E_n N_n."""
    En, tau = sp.symbols("E_n tau_n", positive=True)
    Nn, FX = sp.Function("N_n"), sp.Function("F_X")
    beta = sp.Function("beta_n")(En)
    gamma = readme_gamma_n(En)

    def results():
        nrhs = readme_lnE(En, lambda e: e, FX, beta, tau, gamma)[1]
        n_of = L.to_sympy(nrhs, {"E_n": En, "N_n": Nn(En)})
        integrand = readme_lnE(En, lambda e: n_of, FX, beta, tau, gamma)[0]
        weight, q_integrand = readme_source(En, Nn, FX, beta)
        return [
            exact(
                "the page's integrand per d ln E_n = E x "
                "(Q_X's integrand per dE_n) x E_n",
                integrand - E * weight * q_integrand * En,
            )
        ]

    out = [*results()]
    with L.patched(
        r"\thinspace x F_X(x;\beta_n),\qquad n_n",
        r"\thinspace F_X(x;\beta_n),\qquad n_n",
    ):
        out.append(must_fail("control: without the factor x", results()))
    return out


@derivation("nd.grid")
def check_grid():
    """w_i is the integrand's n_n/(gamma_n tau_n) at E_i, and
    q_X = sum_i K_ji w_i keeps one product per decay.
    gamma_i = E_i/m_n."""
    En, Ei, tau = sp.symbols("E_n E_i tau_n", positive=True)
    nn = sp.Function("n_n")

    def results():
        gi = L.to_sympy(
            L.sides(L.inline(r"\gamma_i = E_i/m_n"))[1], {"E_i": Ei, "m_n": mn}
        )
        rate_i = L.to_sympy(
            L.inline(r"1/(\gamma_i\tau_n)"), {r"\gamma_i": gi, r"\tau_n": tau}
        )
        rate_n = L.to_sympy(
            L.inline(r"1/(\gamma_n\tau_n)"),
            {r"\gamma_n": readme_gamma_n(En), r"\tau_n": tau},
        )
        first, second = L.parts(
            L.display(r"w_i = \frac{n_n(E_i)}{\gamma_i\tau_n}")
        )
        w_page = L.to_sympy(
            L.sides(first)[1],
            {r"\gamma_i": gi, r"\tau_n": tau, "E_i": Ei},
            funcs={"n_n": nn},
        )
        gn = readme_gamma_n(En)
        factor = readme_lnE(En, nn, lambda *a: 1, sp.Symbol("beta"), tau, gn)[
            0
        ].subs(E, En)  # x F_X -> 1 at x = 1
        m = re.fullmatch(
            r"q_X\(E_j\) = \\sum_i K_\{(\w)(\w)\}\\thinspace w_i", second
        )
        assert m, second
        row, col = m.groups()
        K = sp.Matrix(3, 3, lambda j, i: sp.Symbol(f"K{j}{i}"))
        w = sp.symbols("w0:3")
        # each column (one neutron point i) sums to 1
        cols = {K[2, i]: 1 - K[0, i] - K[1, i] for i in range(3)}
        q = [
            sum(
                (K[j, i] if (row, col) == ("j", "i") else K[i, j]) * w[i]
                for i in range(3)
            )
            for j in range(3)
        ]
        return [
            exact(
                "gamma_i = gamma_n at E_n = E_i (the page's gamma_n = E_n/m_n)",
                gi - gn.subs(En, Ei),
            ),
            exact(
                "the solver's rate 1/(gamma_i tau_n) = the page's "
                "1/(gamma_n tau_n) at E_n = E_i",
                rate_i - rate_n.subs(En, Ei),
            ),
            exact(
                "w_i = the page's n_n/(gamma_n tau_n) at E_n = E_i",
                w_page - factor.subs(En, Ei),
            ),
            exact(
                "sum_j q_X(E_j) = sum_i w_i: one product per decay",
                (sum(q) - sum(w)).subs(cols),
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\sum_i K_{ji}\thinspace w_i", r"\sum_i K_{ij}\thinspace w_i"
    ):
        out.append(must_fail("control: K_ij (rows for columns)", results()))
    return out


def readme_hat():
    parts = L.parts(L.display(r"\Lambda_j(x) = \frac{x - x_{j-1}}"))
    pieces = []
    for q, cond in zip(
        parts, (r"(x_{j-1} \le x \le x_j)", r"(x_j \le x \le x_{j+1})")
    ):
        lhs, rhs = L.sides(q)
        frac, tail = rhs.split(r"\quad ")
        assert lhs == r"\Lambda_j(x)" and tail == cond, (lhs, tail)
        pieces.append(frac)
    xm, x0, x1 = sp.symbols("x_m x_0 x_1", positive=True)
    tok = {"x": x, "x_{j-1}": xm, "x_j": x0, "x_{j+1}": x1}
    rise, fall = (L.to_sympy(f, tok) for f in pieces)
    return (
        lambda X, a, b_: rise.subs({x: X, xm: a, x0: b_}, simultaneous=True),
        lambda X, a, b_: fall.subs({x: X, x0: a, x1: b_}, simultaneous=True),
    )


@derivation("nd.hat")
def check_hat():
    """The page's Lambda_j is a hat (1 at x_j, 0 at its neighbors) that
    interpolates 1 and x on an interval.
    Lambda_j + Lambda_{j+1} = 1 and x_j Lambda_j + x_{j+1} Lambda_{j+1} = x, the
    page's sum."""
    xa_, xb_ = sp.symbols("x_a x_b", positive=True)

    def results():
        # rise(x; x_{j-1}, x_j), fall(x; x_j, x_{j+1})
        rise, fall = readme_hat()
        xs = L.sides(L.inline(r"\sum_j x_j\Lambda_j(x) = x"))
        assert xs == [r"\sum_j x_j\Lambda_j(x)", "x"], xs
        return [
            exact(
                "Lambda_j(x_{j-1}) = 0, Lambda_j(x_j) = 1 (rising piece)",
                rise(xa_, xa_, xb_) ** 2 + (rise(xb_, xa_, xb_) - 1) ** 2,
            ),
            exact(
                "Lambda_j(x_j) = 1, Lambda_j(x_{j+1}) = 0 (falling piece)",
                (fall(xa_, xa_, xb_) - 1) ** 2 + fall(xb_, xa_, xb_) ** 2,
            ),
            exact(
                "on [x_j, x_{j+1}]: Lambda_j + Lambda_{j+1} = 1",
                fall(x, xa_, xb_) + rise(x, xa_, xb_) - 1,
            ),
            exact(
                "... and x_j Lambda_j + x_{j+1} Lambda_{j+1} = x",
                xa_ * fall(x, xa_, xb_) + xb_ * rise(x, xa_, xb_) - x,
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\frac{x_{j+1} - x}{x_{j+1} - x_j}\quad",
        r"\frac{x_{j+1} - x}{x_{j+1} - x_{j-1}}\quad",
    ):
        out.append(
            must_fail("control: x_{j-1} in the falling piece", results())
        )
    return out


@derivation("nd.kernel")
def check_kernel():
    """With the page's hat functions, K of a uniform box keeps one product and
    its mean share, and K depends on E_i.
    x_j = E_j/E_i is the page's x at a grid point."""
    xs = sp.symbols("x0:4", positive=True)
    a, b_ = sp.symbols("a b", positive=True)

    def results():
        first, second = L.parts(
            L.display(
                r"K_{ji} = \int dx\thinspace F_X(x;\beta_i)\thinspace "
                r"\Lambda_j(x)"
            )
        )
        assert (
            first == r"K_{ji} = \int dx\thinspace F_X(x;\beta_i)\thinspace "
            r"\Lambda_j(x)"
        ), first
        Ej, Ei = sp.symbols("E_j E_i", positive=True)
        xj = L.to_sympy(L.sides(second)[1], {"E_j": Ej, "E_i": Ei})
        x_page = L.to_sympy(
            L.sides(L.inline(r"x = E/E_n"))[1], {"E": Ej, "E_n": Ei}
        )
        rise, fall = readme_hat()
        # a uniform box [a, b] with x0 < a < x1 < x2 < b < x3
        dens = 1 / (b_ - a)
        segs = [(a, xs[1], 0), (xs[1], xs[2], 1), (xs[2], b_, 2)]
        K = [0] * 4
        for lo_, hi_, k in segs:
            K[k] += sp.integrate(
                dens * fall(x, xs[k], xs[k + 1]), (x, lo_, hi_)
            )
            K[k + 1] += sp.integrate(
                dens * rise(x, xs[k], xs[k + 1]), (x, lo_, hi_)
            )
        lo_end = readme_box()[0].subs(
            b, sp.sqrt(1 - 1 / readme_gamma_n(Ei) ** 2)
        )
        return [
            exact(
                "x_j = the page's x = E/E_n at E = E_j, E_n = E_i", xj - x_page
            ),
            exact("sum_j K_j = 1: the box's product", sum(K) - 1),
            exact(
                "sum_j x_j K_j = (a + b)/2: its mean share",
                sum(xs[j] * K[j] for j in range(4)) - (a + b_) / 2,
            ),
            Result(
                "the range's ends depend on E_i through beta_i "
                "(not only on E_j/E_i)",
                sp.diff(lo_end, Ei) != 0,
                f"d(lower end)/dE_i = {sp.simplify(sp.diff(lo_end, Ei))}",
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\frac{x - x_{j-1}}{x_j - x_{j-1}}\quad",
        r"\frac{x - x_{j-1}}{x_{j+1} - x_{j-1}}\quad",
    ):
        out.append(
            must_fail("control: a rising piece over two intervals", results())
        )
    return out


def readme_bin(w, a, b_, xa_, xb_):
    first, second = L.parts(
        L.display(r"\mathcal N_j = \omega_k\thinspace\frac{x_b - x_a}{b - a}")
    )
    tok = {r"\omega_k": w, "a": a, "b": b_, "x_a": xa_, "x_b": xb_}
    N = L.to_sympy(L.sides(first)[1], tok)
    En_ = L.to_sympy(L.sides(second)[1], {**tok, r"\mathcal N_j": N})
    return N, En_


@derivation("nd.bin")
def check_bin():
    """The page's N_j and E_j are the number and energy of a uniform box on its
    overlap with an interval.
    The energies are in units of E_i. The overlaps with consecutive intervals
    hold the box's whole weight."""
    w, a, b_, xa_, xb_ = sp.symbols("omega a b x_a x_b", positive=True)

    def results():
        assert L.sides(L.inline(r"x_a = \max(a, x_j)"))[1] == r"\max(a, x_j)"
        assert (
            L.sides(L.inline(r"x_b = \min(b, x_{j+1})"))[1]
            == r"\min(b, x_{j+1})"
        )
        N, En_ = readme_bin(w, a, b_, xa_, xb_)
        dens = w / (b_ - a)
        # a < x_1 < x_2 < b
        x1, x2 = sp.symbols("x_1 x_2", positive=True)
        total = sum(
            N.subs({xa_: lo_, xb_: hi_})
            for lo_, hi_ in ((a, x1), (x1, x2), (x2, b_))
        )
        return [
            exact(
                "N_j = the integral of the density over the overlap",
                N - sp.integrate(dens, (x, xa_, xb_)),
            ),
            exact(
                "E_j = the integral of x times the density over it",
                En_ - sp.integrate(x * dens, (x, xa_, xb_)),
            ),
            exact(
                "the overlaps of consecutive intervals hold the box's weight",
                total - w,
            ),
        ]

    out = [*results()]
    with L.patched(
        r"\mathcal N_j\thinspace\frac{x_a + x_b}{2}",
        r"\mathcal N_j\thinspace\frac{x_a + x_b}{3}",
    ):
        out.append(must_fail("control: (x_a + x_b)/3", results()))
    return out


@derivation("nd.share")
def check_share():
    """The page's shares of an interval are the box's integrals against the hat
    functions: they keep number and energy."""
    w, a, b_, xa_, xb_, xj, xj1 = sp.symbols(
        "omega a b x_a x_b x_j x_j1", positive=True
    )

    def results():
        first, second = L.parts(L.display(r"K_j^\mathrm{lo} = \frac"))
        N, En_ = readme_bin(w, a, b_, xa_, xb_)
        tok = {
            r"\mathcal N_j": N,
            r"\mathcal E_j": En_,
            "x_j": xj,
            "x_{j+1}": xj1,
        }
        assert (
            L.sides(first)[0] == r"K_j^\mathrm{lo}"
            and L.sides(second)[0] == r"K_j^\mathrm{hi}"
        )
        klo, khi = (
            L.to_sympy(L.sides(first)[1], tok),
            L.to_sympy(L.sides(second)[1], tok),
        )
        rise, fall = readme_hat()
        dens = w / (b_ - a)
        return [
            exact(
                "K_lo = the integral of the box against Lambda_j "
                "(falling on [x_j, x_{j+1}])",
                klo - sp.integrate(dens * fall(x, xj, xj1), (x, xa_, xb_)),
            ),
            exact(
                "K_hi = ... against Lambda_{j+1} (rising there)",
                khi - sp.integrate(dens * rise(x, xj, xj1), (x, xa_, xb_)),
            ),
            exact("K_lo + K_hi = N_j (the number)", klo + khi - N),
            exact(
                "x_j K_lo + x_{j+1} K_hi = E_j (the energy)",
                xj * klo + xj1 * khi - En_,
            ),
        ]

    out = [*results()]
    with L.patched(
        r"K_j^\mathrm{lo} = \frac{x_{j+1}\mathcal N_j - \mathcal E_j}",
        r"K_j^\mathrm{lo} = \frac{\mathcal E_j - x_{j+1}\mathcal N_j}",
    ):
        out.append(
            must_fail(
                "control: the lower share's numerator reversed", results()
            )
        )
    return out


@derivation("nd.proton")
def check_proton():
    """The proton's down share s_i is Lambda_{i-1}(x_p), q_p keeps number and
    energy, and with the step 0.1, s_i = 0.0145.
    The down share of the lowest point is lost."""
    r = sp.Symbol("r", positive=True)  # E_{i-1}/E_i

    def results():
        first, second = L.parts(L.display(r"s_i = \frac{E_0/m_n}"))
        Eim, Ei = sp.symbols("E_im E_i", positive=True)
        s_page = L.to_sympy(
            L.sides(first)[1],
            {"E_0": readme_E0(), "m_n": mn, "E_{i-1}": Eim, "E_i": Ei},
        ).subs(Eim, r * Ei)
        xp = L.to_sympy(
            L.sides(L.inline(r"x_p = (m_n - E_0)/m_n"))[1],
            {"m_n": mn, "E_0": readme_E0()},
        )
        rise, fall = readme_hat()
        s, w = sp.symbols("s0:4"), sp.symbols("w0:4")
        qform = L.sides(second)
        assert qform[0] == "q_p(E_i)", qform
        # four hadron points E_k = r^-k, the top one without a point above it
        q = lambda i: L.to_sympy(
            qform[1],
            {
                "s_i": s[i],
                "w_i": w[i],
                "s_{i+1}": s[i + 1] if i < 3 else 0,
                "w_{i+1}": w[i + 1] if i < 3 else 0,
            },
        )
        q4 = [q(i) for i in range(4)]
        lost = s[0] * w[0]  # the protons below point 0
        decays_energy = (
            sum(w[k] * ((1 - s[k]) + s[k] * r) * r**-k for k in range(4))
            - lost * r
        )
        energy = sum(r**-k * q4[k] for k in range(4)) - decays_energy
        m = re.search(
            r"in \$\\ln E\$ \(([\d.]+)\)",
            L.line(r"with the same step $\Delta$ in $\ln E$"),
        )
        step = sp.Rational(m.group(1))
        share = re.search(
            r"every \$s_i\$ the same, ([\d.]+)\.",
            L.line("The equal steps make every"),
        ).group(1)
        with mp.workdps(30):
            value = mpf(s_page.subs(r, sp.exp(-step)).subs(EXACT))
        return [
            exact(
                "s_i = Lambda_{i-1}(x_p) (falling piece on [E_{i-1}/E_i, 1])",
                s_page - fall(xp, r, 1),
            ),
            exact(
                "1 - s_i = Lambda_i(x_p) (rising piece)",
                1 - s_page - rise(xp, r, 1),
            ),
            exact(
                "the energy: (1 - s_i) E_i + s_i E_{i-1} = x_p E_i",
                (1 - s_page) + s_page * r - xp,
            ),
            exact(
                "sum_i q_p(E_i) = the decays less the lowest point's "
                "down share",
                sum(q4) - (sum(w) - lost),
            ),
            exact(
                "sum_i E_i q_p(E_i) = the energy of the decays' shares, "
                "less the lost one",
                energy,
            ),
            Result(
                f"with the step {step}: every s_i = {share}, as printed",
                rounds_to(value, share),
                mp.nstr(value, 6),
            ),
        ]

    out = [*results()]
    with L.patched(
        r"s_i = \frac{E_0/m_n}{1 - E_{i-1}/E_i}",
        r"s_i = \frac{E_0/m_n}{1 + E_{i-1}/E_i}",
    ):
        out.append(must_fail("control: 1 + E_{i-1}/E_i", results()))
    return out
