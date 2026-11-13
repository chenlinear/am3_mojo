"""Derivation checks of tests/literature_checks/triplet/README.md (triplet pair
production): each equation of the page against its source, or the derivation
that leads to it from there.

Sources: Lee, Lyubyakin & Stotsky (2021), from the TeX source of
arXiv:2010.15430v1. Sec. 4 of the journal version, JHEP 01 (2021) 144, differs
only in numbering, a footnote and references. Its eqs. 4.11-4.13 are v1's
4.10-4.11, because v1 numbers 4.12-4.13 as one. The other sources are Haug
(1975, Z. Naturforsch. 30a, 1099) and Blumenthal & Gould (1970, Rev. Mod. Phys.
42, 237), whose scans are transcribed below with page and equation numbers.

The rate, G_r and the kinetic terms (README, Rates and Kinetic terms) are
checked against a direct count of the collisions of leptons (gamma' >> 1) with
isotropic photons. The count holds for any sigma, f_r and mean fractions. The
kinetic terms conserve one e+, one e- and one photon per interaction, and the
pair's energy. This is checked on lines of leptons and photons (densities
N delta(ln gamma - ln gamma_i)), and on the grid (README, On the grid).

The scripts in this folder and tests/unit/test_added_processes.py also take from
here the README's expansions (readme_expansions, numeric_expansions), the energy
fractions its loss averages (readme_loss_fractions), its symbols, the line of
its inverse-Compton loss (IC_LINE) and Lee et al.'s eq. 4.12 (LEE_412, parsed by
lee).

Run from the repository root:
  .venv/bin/python tests/literature_checks/common/run.py triplet
"""

import functools
import re as regexp

import mpmath as mp
import sympy as sp

import latex
import mojo_constants
from latex import LEE
from verify import (
    Entry,
    Result,
    absent,
    derivation,
    exact,
    exact_fails,
    in_source,
    mdot,
    must_fail,
    numeric,
    rng,
    rounds_to,
    worst_of,
)

L = latex.page("triplet")
PAGE = "triplet"

KINEMATICS = r"s = (p_1 + k)^2 = 1 + 2k"
THRESHOLD = r"\frac{\sigma}{\alpha r_e^2} = \frac{\pi}{3\sqrt3}"
EQ411 = r"\sigma = \alpha^3\left[\frac{28\ln s}{9}"
EQ412 = r"\sigma = \alpha^3\left[\frac{28}{9}\ln 2k"
HIGH = r"\frac{\sigma}{\alpha r_e^2} = \frac{28}{9}\ln 2k - \frac{218}{27}"
TABLE = r"| Photon energy | The code's $\sigma(k)$ | Code |"
SPLINE = r"v = (1-t)\thinspace v_n + t\thinspace v_{n+1}"
INNER = (
    r"h_{n-1}M_{n-1} + 2\thinspace (h_{n-1} + h_n)\thinspace M_n + h_n M_{n+1}"
)
ENDS = r"2h_0M_0 + h_0M_1 = 6\left(\frac{v_1 - v_0}{h_0} - v_0'\right)"
SLOPE_LOW = (
    r"= \frac{4\thinspace (k-4)}{k^2}\left(\frac{2}{r} + "
    r"\frac{p'(r)}{p(r)}\right)"
)
SLOPE_HIGH = (
    r"= \frac{k-4}{\sigma}\left(\frac{28}{9k} - "
    r"\frac{Q'(L) - Q(L)}{k^2}\right)"
)
XI = r"$\xi = 1 - x_\mathrm{high} = x_+ + x_\mathrm{low}$"
LAB = r"E_\mathrm{lab} = \gamma x + \frac{\varepsilon}{k}\thinspace p_z"
LAWS = r"f_r = g_0(y) + \frac{g_1(y)}{\ln k}\quad (y \le 100)"
ANGLE = r"R(P) = \frac{1}{2}\int_{-1}^{1} d\mu"
RATES = r"R(P) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k"
PAIRS = r"\dot n_{+}(\gamma) = c\thinspace \sigma_T\int d\ln\gamma'"
LOSS = r"-\frac{\dot\gamma}{\gamma} = c\thinspace \sigma_T\int d\ln\varepsilon"
SINK = (
    r"\dot n_\gamma(\varepsilon) = -\thinspace "
    r"n_\gamma(\varepsilon)\thinspace c\thinspace \sigma_T"
)
GRID = r"P_q = e^{(m_0 + q)\Delta}"
ROW_R = r"R_q = \frac{1}{2P_q^2}\int_{\ln 4}^{\ln 2P_q} d\ln k"
ROW_L = (
    r"L_q = \sum_o \bigl(W_{+,q,o} + W_{\mathrm{low},q,o}\bigr)\thinspace x_o"
)
SHARE = r"w = \frac{x_o - x}{x_o - x_{o+1}}"
STEP = r"Z_{r,i,o} = c\thinspace \sigma_T\Delta\sum_j"
SOURCE = r"\dot n_+(\gamma_l) = \sum_i"
IC_KN = (
    r"$-\dot\gamma$ $= (3/8) c\sigma_T n\varepsilon^{-1}$ "
    r"$[\ln(4\gamma\varepsilon) - 11/6]$"
)
# the line of IC_KN and its source, also for inelasticity.py (a pin the controls
# leave intact)
IC_LINE = (
    "compares the loss with the inverse-Compton loss in its Klein–Nishina limit"
)
BG_QUOTE = (
    r"$= \pi r_0^2 m^2c^5 \int d\varepsilon\ n(\varepsilon)$ "
    r"$\varepsilon^{-1}$ $[\ln(4\varepsilon\gamma/mc^2) - 11/6]$"
)
ELECTRONS = (
    r"likewise for the electrons, with $n_{-}$ and $n_+$ exchanged "
    r"($n_- G_\mathrm{low} + n_+ G_+$)"
)
MEAN = r"with $\langle\ldots\rangle$ the mean over $k$ with the weight $"

# this module's equations in the inventory (common/inventory.py)
INVENTORY = [
    Entry(
        "tr.s",
        KINEMATICS,
        "standard",
        "kinematics: s = (p_1 + k)^2, the photon's energy p_1.k in the "
        "lepton's rest frame (Haug 1975, p. 1109; Lee et al. 2021, "
        "after eq. 4.12)",
        file=PAGE,
    ),
    Entry(
        "tr.k",
        r"$k = \gamma\varepsilon(1 - \cos\theta)$",
        "standard",
        "kinematics (gamma >> 1); the threshold k = 4",
        file=PAGE,
    ),
    Entry(
        "tr.J11",
        r"$J_{11}' = zJ_9/(s - 5)$",
        "corrected",
        "Lee et al. 2021, eqs. 4.2-4.3 (J_11 lacks a factor 2)",
        "TeX " + LEE,
        "triplet/compare_exact.py (eq. 4.4 with the factor 2 against the "
        "threshold expansion)",
        file=PAGE,
    ),
    Entry(
        "tr.alpha3",
        r"$\alpha^3\hbar^2/(m_e c)^2 = \alpha r_e^2$",
        "standard",
        "units hbar = c = m_e = 1 of Lee et al. 2021 (eq. 4.5 omits the "
        "alpha^3 of eqs. 4.4, 4.11-4.13)",
        "TeX " + LEE,
        file=PAGE,
    ),
    Entry(
        "tr.threshold",
        THRESHOLD,
        "transcribed",
        "Lee et al. 2021, eq. 4.5",
        "TeX " + LEE,
        "triplet/compare_exact.py (against the tables at k = 4.02)",
        file=PAGE,
    ),
    Entry(
        "tr.votruba",
        r"$(\sqrt3\pi/972)\thinspace \alpha r_e^2 (k-4)^2$",
        "derived",
        "Haug 1975, eq. 9.1 (p. 1109) and Table 1 (p. 1110): Votruba's "
        "threshold approximation",
        "PDF 10.1515_zna-1975-0901.pdf",
        file=PAGE,
    ),
    Entry(
        "tr.eq411",
        EQ411,
        "transcribed",
        "Lee et al. 2021, eq. 4.11",
        "TeX " + LEE,
        file=PAGE,
    ),
    Entry(
        "tr.eq412",
        EQ412,
        "derived",
        "Lee et al. 2021, eq. 4.12 (eq. 4.11 in k)",
        "TeX " + LEE,
        file=PAGE,
    ),
    Entry(
        "tr.high",
        HIGH,
        "transcribed",
        "Lee et al. 2021, eq. 4.13 (rounds eq. 4.12)",
        "TeX " + LEE,
        "triplet/compare_exact.py (against the tables for k >= 1e3)",
        file=PAGE,
    ),
    Entry(
        "tr.table_range",
        r"For $k \le 4.02$ and $k \ge 10^3$, the code "
        r"uses the expansions",
        "model",
        "am3_mojo (the tables' range and where the expansions take over, "
        "as the displays' conditions state it)",
        file=PAGE,
    ),
    Entry(
        "tr.spline",
        SPLINE,
        "standard",
        "the clamped cubic spline of ln sigma in ln(k - 4) "
        "(the code's sigma(k))",
        checks="triplet/compare_exact.py (the spline against the exact cross "
        "section and Haug's values)",
        file=PAGE,
    ),
    Entry(
        "tr.slopes",
        SLOPE_LOW,
        "derived",
        "d ln sigma/d ln(k - 4) of the expansions (the spline's end slopes)",
        file=PAGE,
    ),
    Entry(
        "tr.sigma_T",
        r"$\alpha r_e^2/\sigma_T = 3\alpha/(8\pi)$",
        "standard",
        "sigma_T = 8 pi r_e^2/3; AM3's code units (c sigma_T = 1)",
        file=PAGE,
    ),
    Entry(
        "tr.x",
        r"x = \frac{p\cdot k}{p_1\cdot k}",
        "standard",
        "light-cone fraction; the lab energy share for gamma >> 1, "
        "eps << gamma",
        checks="triplet/x_scaling.py (independent: the QED events boosted)",
        file=PAGE,
    ),
    Entry("tr.x_rest", r"$x = E - p_z$", "derived", "kinematics", file=PAGE),
    Entry(
        "tr.xi",
        XI,
        "derived",
        "momentum conservation",
        file=PAGE,
    ),
    Entry(
        "tr.lab",
        LAB,
        "derived",
        "the boost of a product to the lab (what the lab energy gamma "
        "x neglects)",
        checks="triplet/x_scaling.py (its effect on the spectra, with the "
        "QED events)",
        file=PAGE,
    ),
    Entry(
        "tr.laws",
        LAWS,
        "model",
        "am3_mojo (the extrapolation above k = 1e10, fitted to the nodes "
        "from 1e6 on)",
        checks="triplet/extrapolation.py (the laws fitted to fewer nodes "
        "against the nodes left out)",
        file=PAGE,
    ),
    Entry(
        "tr.angle",
        ANGLE,
        "standard",
        "the collision rate c (1 - beta cos theta) sigma n averaged over "
        "isotropic photons",
        file=PAGE,
    ),
    Entry(
        "tr.R",
        RATES,
        "derived",
        "isotropic average of the rate times f_r "
        "(G_r; R(P): tr.rate, tr.angle)",
        checks="tests/unit/test_added_processes.py "
        "(the kernel rows add up to R)",
        file=PAGE,
    ),
    Entry(
        "tr.rate",
        r"interacts at the rate "
        r"$c\sigma_T \int d\ln\varepsilon\ n_\gamma(\varepsilon)$",
        "standard",
        "the collisions with isotropic photons counted, with R(P) of the "
        "tr.R display",
        file=PAGE,
    ),
    Entry(
        "tr.pairs",
        PAIRS,
        "derived",
        "the products of the collisions (charge conjugates for e+ targets)",
        checks="tests/unit/test_added_processes.py",
        file=PAGE,
    ),
    Entry(
        "tr.loss",
        LOSS,
        "derived",
        "the mean energy xi gamma the collisions cost "
        "(treated as continuous: am3_mojo's choice)",
        checks="tests/unit/test_added_processes.py",
        file=PAGE,
    ),
    Entry(
        "tr.ic_kn",
        IC_KN,
        "rearranged",
        "Blumenthal & Gould 1970, eq. 2.57 (p. 244)",
        "PDF blumenthal1970.pdf",
        "triplet/inelasticity.py",
        file=PAGE,
    ),
    Entry(
        "tr.xi_bin",
        r"$\xi \gt 1 - e^{-\Delta}$",
        "derived",
        "a loss of more than a grid bin",
        checks="triplet/inelasticity.py (reads the 0.095)",
        file=PAGE,
    ),
    Entry(
        "tr.sink",
        SINK,
        "derived",
        "one photon per collision",
        checks="tests/unit/test_added_processes.py",
        file=PAGE,
    ),
    Entry(
        "tr.grid",
        GRID,
        "model",
        "AM3's grids (step Delta in ln E) and the code's first row "
        "above threshold",
        file=PAGE,
    ),
    Entry(
        "tr.rows",
        ROW_R,
        "derived",
        "R(P) in ln k on the grid; the energy of the deposits",
        checks="tests/unit/test_added_processes.py (the code's rows: number "
        "and energy)",
        file=PAGE,
    ),
    Entry(
        "tr.share",
        SHARE,
        "standard",
        "the sharing of a product between two grid points "
        "(number and energy kept)",
        file=PAGE,
    ),
    Entry(
        "tr.step",
        STEP,
        "derived",
        "the kinetic terms as sums over the grid (the code's per-step terms)",
        checks="tests/unit/test_added_processes.py (a step's products, "
        "steady states)",
        file=PAGE,
    ),
]
# entry id -> the pins of its other relations: those of its display equation
# (each a whole relation as latex.parts cuts it) and the inline relations of the
# same derivation step, which its check proves too
ALSO = {
    "tr.s": (
        r"k = p_1\cdot k = \gamma\varepsilon\thinspace (1 - "
        r"\beta\cos\theta)",
        r"the invariant $s = (p_1 + k)^2$",
    ),
    "tr.k": (r"$s = 9$, so $k = 4$",),
    "tr.J11": (
        r"$z = \sqrt{(s-5)/(s-1)}$",
        r"$\ln(\sqrt{s-5} + \sqrt{s-1})$",
        r"$1/[2\sqrt{(s-5)(s-1)}]$",
        r"half of $z/{(s - 5)}$",
    ),
    "tr.alpha3": (r"$r_e = \alpha\hbar/(m_e c)$",),
    "tr.threshold": (
        r"r = \frac{s-9}{s-1}",
        r"s = 1 + 2k",
        r"$r = (s-9)/(s-1)$",
        r"$r = 1 - 4/k$",
        r"\sigma = \frac{\pi}{3\sqrt3}\left[",
    ),
    "tr.eq412": (r"$\ln s = \ln 2k + \ln[1 + 1/(2k)]$",),
    "tr.table_range": (r"(k \le 4.02)", r"(k \ge 10^3)"),
    "tr.spline": (
        INNER,
        ENDS,
        r"h_{N-1}M_{N-1} + 2h_{N-1}M_N = "
        r"6\left(v_N' - \frac{v_N - v_{N-1}}{h_{N-1}}\right)",
        r"$h_n = u_{n+1} - u_n$",
        r"$t = (u - u_n)/h_n$",
    ),
    "tr.slopes": (
        SLOPE_HIGH,
        r"$p(r) = 4/27 + 602r/243 + 4217r^2/4374$",
        r"$dr/dk = 4/k^2$",
        r"$\sigma/(\alpha r_e^2) = 28L/9 - 218/27 - Q(L)/k$",
        r"$L = \ln 2k$",
        r"$dL/dk = 1/k$",
    ),
    "tr.sigma_T": (r"$\sigma_T = 8\pi r_e^2/3$", r"$c\sigma_T = 1$"),
    "tr.lab": (
        r"$\vert p_z\vert \lt k$",
        r"$\sin^2\theta \le 2(1 - \beta\cos\theta)$",
        r"$p_\perp\sqrt{2\gamma\varepsilon/k}$",
    ),
    "tr.laws": (
        r"k f_r = A(x)\ln k + B(x)\quad (x \ge 10^{-5})",
        r"$f_r = (a_0\ln y + b_0)/y$",
        r"$a_0 = 10^{-5}A(10^{-5})$",
        r"$b_0 = 10^{-5}B(10^{-5}) - a_0\ln 10^{-5}$",
    ),
    "tr.angle": (r",\qquad P = \gamma\varepsilon,", r"$k = P(1 - \mu)$"),
    "tr.R": (r"G_r(P, x) = \frac{1}{2P^2}\int_4^{2P}", r"$d\ln y = d\ln x$"),
    "tr.pairs": (r"$x = \gamma/\gamma'$",),
    "tr.grid": (
        r"m_0 = \left\lfloor \frac{\ln 2}{\Delta}\right\rfloor + 1",
        r"\gamma_i\varepsilon_j = P_q\quad \text{for}\quad q = "
        r"i + j - j_0 - m_0",
        r"$\gamma_i = e^{i\Delta}$",
        r"$\varepsilon_j = e^{(j - j_0)\Delta}$",
        r"$\varepsilon_{j_0} = 1$",
        r"$k = 2P = 4$",
    ),
    "tr.rows": (ROW_L,),
    "tr.share": (
        r"$x_o = e^{-o\Delta}$",
        r"$o = \lfloor -(\ln x)/\Delta\rfloor$",
        r"$w + (1 - w) = 1$",
        r"$(1 - w)\thinspace x_o + w\thinspace x_{o+1} = x$",
        r"have $x = y/k$",
    ),
    "tr.step": (
        r"-\frac{\dot\gamma_i}{\gamma_i} = c\thinspace "
        r"\sigma_T\Delta\sum_j",
        r"\dot n_\gamma(\varepsilon_j) = -\thinspace "
        r"n_\gamma(\varepsilon_j)",
        SOURCE,
        r"$\Delta\thinspace G_r(P_q, x_o)$",
        r"$o = i - l$",
    ),
    "tr.ic_kn": (BG_QUOTE, r"$\pi r_e^2 = (3/8)\sigma_T$"),
    "tr.xi_bin": (r"($\xi \gt 0.095$)",),
}

k, s, x, al, re, sT, c, eps, gam = sp.symbols(
    "k s x alpha r_e sigma_T c epsilon gamma", positive=True
)
Lg = sp.Symbol("L")  # ln 2k
ZETA = {r"\zeta_3": sp.zeta(3)}

# Lee, Lyubyakin & Stotsky's expansions, from the TeX source
LEE_45 = (
    r"\frac{\pi}{3 \sqrt{3}}  \left[\frac{4 r^2}{27}+\frac{602 "
    r"r^3}{243}+\frac{4217 r^4}{4374}+O(r^5)\right]"
)
LEE_R = r"$r=\frac{s-9}{s-1}$"
LEE_K = r"photon energy $k=(s-1)/2$ in the rest frame of the initial electron"
LEE_411 = (
    r"\frac{28 \ln{s}}{9}-\frac{218}{27}",
    r"-\frac{1}{s}\left(\frac{8}{3} \ln^3{s}-\frac{49 }{6}\ln "
    r"^2{s}-\left(2 \pi ^2-\frac{409}{9}\right)\ln{s}+12 "
    r"\zeta_3+\frac{31 \pi^2}{18}-\frac{1081}{12}\right)",
)
LEE_412 = (
    r"\frac{28}{9} \ln (2 k)-\frac{218}{27}",
    r"+\frac{1}{k}\left(-\frac{4}{3}  \ln ^3(2 k)+\frac{49}{12} \ln "
    r"^2(2 k)+\left(\pi ^2 -\frac{409}{18}\right) \ln (2 k)-6 "
    r"\zeta_3-\frac{31 \pi ^2}{36}+\frac{3355}{72}\right)",
)
LEE_413 = (
    r"\frac{28}{9} \ln (2 k)-\frac{218}{27}-\frac{1}{k}\left(\frac{4}{3}  \ln "
    r"^3(2 k)-4.08\ln ^2(2 k)+12.85 \ln (2 k)-30.89\right)"
)
# the heads of eqs. 4.4, 4.5, 4.11 and 4.12 (alpha^3 in all but eq. 4.5)
LEE_HEADS = {
    "4.4": r"\sigma_{\egeee}=\frac{\alpha ^3}{s-1}",
    "4.5": r"\sigma_{\egeee}=\frac{\pi}{3 \sqrt{3}}",
    "4.11": r"\sigma_{\egeee}=\alpha ^3 \bigg[\frac{28 \ln{s}}{9}",
    "4.12": r"\sigma_{\egeee}=\alpha ^3 \bigg[\frac{28}{9} \ln (2 k)",
}
# eq. 4.2, the differential system of J_3 ... J_12 (J_13 ... J_18 vanish), and
# y and z, from the TeX source
LEE_42 = {
    3: r"J_3'=\frac{2J_1}{s-9}",
    4: r"J_4'=\frac{J_3}{s-1}",
    5: r"J_5'=\frac{J_1}{s-1}",
    6: r"J_6'=\frac{J_5}{2 s}",
    7: r"J_7'=-\frac{3 J_3}{2s}",
    8: r"J_8'=\frac{ y }{s-1}(3J_3+J_5)",
    9: r"J_9'=\frac{(7 s-15) z J_1}{(s-9) (s-5)}",
    10: r"J_{10}'=\frac{z}{s-5}\left(2J_5-9J_3+2J_{15}\right)+\frac{J_9}{s-1}",
    11: r"J_{11}'=\frac{z J_9}{s-5}",
    12: r"J_{12}'=\frac{3 z J_3-zJ_5+J_9-zJ_{15}}{s-5}",
}
LEE_ZERO = r"$J_{13}=J_{14}=J_{15}=J_{16}=J_{17}=J_{18}=0$"
LEE_YZ = (r"$y=\sqrt{\frac{s-1}{s+3}}$", r"$z=\sqrt{\frac{s-5}{s-1}}$")
# eq. 4.3, J_3 ... J_12 as one-fold integrals of J_1 over s~ from 9 to s, from
# the TeX source (J_12 in two pieces around the TeX's \raggedright{...})
LEE_43 = {
    3: (r"J_3(s)=2\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-9}",),
    4: (
        r"J_4(s)=2\intop_9^s "
        r"d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-9}\ln\left(\frac{s-1}"
        r"{\tilde{s}-1}\right)",
    ),
    5: (r"J_5(s)=\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-1}",),
    6: (
        r"J_6(s)=\frac12\intop_9^s "
        r"d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-1}\ln "
        r"\left(\frac{s}{\tilde{s}}\right)",
    ),
    7: (
        r"J_7(s)=-3\intop_9^s d\tilde{s}\frac{J_1(\tilde{s})}{\tilde{s}-9}\ln "
        r"\left(\frac{s}{\tilde{s}}\right)",
    ),
    8: (
        r"J_8(s)=2\intop_9^s d\tilde{s} "
        r"\frac{(7\tilde{s}-15)J_1(\tilde{s})}{\left(\tilde{s}-9\right) "
        r"\left(\tilde{s}-1\right)}\ln\left(\frac{\sqrt{s-1}+\sqrt{s+3}}{"
        r"\sqrt{\tilde{s}-1}+\sqrt{\tilde{s}+3}}\right)",
    ),
    9: (
        r"J_9(s)=\intop_9^s d\tilde{s} "
        r"\frac{\frac{7\tilde{s}-15}{\tilde{s}-9}J_1(\tilde{s})}"
        r"{\sqrt{(\tilde{s}-5)(\tilde{s}-1)}}",
    ),
    10: (
        r"J_{10}(s)=\intop_9^s d\tilde{s} "
        r"\frac{\frac{7 "
        r"\tilde{s}-15}{\tilde{s}-9}J_1(\tilde{s})}{\sqrt{(\tilde{s}-5)("
        r"\tilde{s}-1)}} \ln\left(\frac{s-1}{\tilde{s}-1}\right)-32\intop_9^s "
        r"d\tilde{s} "
        r"\frac{\tilde{s}J_1(\tilde{s})}{(\tilde{s}-9)(\tilde{s}-1)} \ln "
        r"\left(\frac{\sqrt{s-5}+\sqrt{s-1}}"
        r"{\sqrt{\tilde{s}-5}+\sqrt{\tilde{s}-1}}\right)",
    ),
    11: (
        r"J_{11}(s)=\intop_9^s d\tilde{s} "
        r"\frac{J_1(\tilde{s})}{\sqrt{(\tilde{s}-5)(\tilde{s}-1)}} "
        r"\frac{7 \tilde{s}-15}{\tilde{s}-9} \ln "
        r"\left(\frac{\sqrt{s-5}+\sqrt{s-1}}"
        r"{\sqrt{\tilde{s}-5}+\sqrt{\tilde{s}-1}}\right)",
    ),
    12: (
        r"J_{12}(s)=\intop_9^s d\tilde{s} "
        r"\frac{\frac{7 "
        r"\tilde{s}-15}{\tilde{s}-9}J_1(\tilde{s})}{\sqrt{(\tilde{s}-5)("
        r"\tilde{s}-1)}} \ln \left(\frac{s-5}{\tilde{s}-5}\right)",
        r"+2\intop_9^s d\tilde{s} "
        r"\frac{(5 \tilde{s}+3)J_1(\tilde{s})}{\left(\tilde{s}-9\right) "
        r"\left(\tilde{s}-1\right)} \ln "
        r"\left(\frac{\sqrt{s-5}+\sqrt{s-1}}"
        r"{\sqrt{\tilde{s}-5}+\sqrt{\tilde{s}-1}}\right)",
    ),
}

# Haug (1975), Z. Naturforsch. 30a, 1099 (scan): eq. (9.1), p. 1109, Votruba's
# threshold approximation (r_0 the classical electron radius, k the photon
# energy in the target's rest frame), and the sigma_V row of Table 1, p. 1110
# [10^-10 b], as printed
HAUG_91 = r"\sigma_V = \frac{\sqrt{3}\pi}{972}\alpha r_0^2(k-4)^2"
HAUG_SIGMA_V = (
    ("4.001", "0.03244"),
    ("4.002", "0.1297"),
    ("4.003", "0.2919"),
    ("4.004", "0.5190"),
    ("4.005", "0.811"),
    ("4.01", "3.244"),
    ("4.02", "12.97"),
    ("4.03", "29.19"),
    ("4.04", "51.9"),
    ("4.05", "81.1"),
)
# Blumenthal & Gould (1970), Rev. Mod. Phys. 42, 237 (scan): eq. (2.57), p. 244,
# the inverse-Compton loss of an electron of energy E = gamma m c^2 in the
# extreme Klein-Nishina limit (Gamma_eps >> 1), photons n(eps) d eps of energy
# eps
BG_257 = (
    r"-\frac{dE}{dt} = \pi r_0^2 m^2 c^5 \int "
    r"\frac{n(\epsilon)}{\epsilon}\left(\ln\frac{4\epsilon\gamma}{mc^2} "
    r"- \frac{11}{6}\right) d\epsilon"
)


def lee(tex, variable):
    tokens = {variable: {"s": s, "k": k, "r": sp.Symbol("r")}[variable], **ZETA}
    return sum(
        L.to_sympy(part, tokens)
        for part in (tex if isinstance(tex, tuple) else (tex,))
    )


def high_energy_form(expr):
    e = sp.expand(expr.subs(sp.log(2 * k), Lg))
    return {p: sp.expand(e.coeff(k, -p)) for p in (0, 1)}


def _controls(claims, patches):
    out = []
    for old, new in patches:
        with L.patched(old, new):
            out.append(must_fail(f"control: README with {new}", claims()))
    return out


def relation(pin, tokens, funcs=None, text=None):
    return [
        L.to_sympy(side, tokens, funcs=funcs)
        for side in L.sides(L.inline(pin) if text is None else text)
    ]


def rhs(pin, tokens, funcs=None, text=None):
    return L.to_sympy(
        L.sides(L.inline(pin) if text is None else text)[-1],
        tokens,
        funcs=funcs,
    )


def display_rhs(pin):
    return L.sides(L.parts(L.display(pin))[0])[-1]


def readme_delta():
    found = regexp.search(
        r"AM³'s grids have the step \$\\Delta = ([^$]+)\$ in",
        L.line("AM³'s grids have the step"),
    )
    return sp.nsimplify(L.to_sympy(found.group(1), {}), rational=True)


# ----------------------- kinematics
def lab_lepton():
    """(gamma, gamma beta, cos theta, sin theta) as rational functions of v,
    u > 0: gamma = (v + 1/v)/2, so that (gamma^2 - 1)^(1/2) = (v - 1/v)/2, and
    cos theta = (1 - u^2)/(1 + u^2)."""
    v, u = sp.symbols("v u", positive=True)
    gl, gb, ct, st = (
        (v + 1 / v) / 2,
        (v - 1 / v) / 2,
        (1 - u**2) / (1 + u**2),
        2 * u / (1 + u**2),
    )
    return gl, gb, ct, st, (gl, 0, 0, gb), (eps, eps * st, 0, eps * ct)


@derivation("tr.s")
def check_s():
    """s = (p_1 + k)^2 = 1 + 2k with k = p_1.k = gamma eps (1 - beta cos theta),
    the photon's energy in the lepton's rest frame (m_e = c = 1)."""
    gl, gb, ct, st, p1, ph = lab_lepton()
    k_rest = mdot(p1, ph)
    s_inv = mdot(*[[a + b for a, b in zip(p1, ph)]] * 2)

    def claims():
        s_part, k_part = L.parts(L.display(r"\qquad k = p_1\cdot k"))
        s_sides, k_sides = L.sides(s_part), L.sides(k_part)
        k_readme = L.to_sympy(
            k_sides[2],
            {
                r"\gamma": gl,
                r"\varepsilon": eps,
                r"\beta": gb / gl,
                r"\theta": sp.acos(ct),
            },
        )
        inv = L.sides(L.after("depends only on the invariant $"))
        return [
            Result(
                "notation: s = (p_1 + k)^2 (also in the text) and k = p_1.k",
                s_sides[:2] == ["s", "(p_1 + k)^2"]
                and k_sides[:2] == ["k", r"p_1\cdot k"]
                and inv == s_sides[:2],
                f"{s_sides[:2]}, {k_sides[:2]}, {inv}",
            ),
            exact(
                "p_1.k = the README's gamma eps (1 - beta cos theta)",
                k_rest - k_readme,
            ),
            exact(
                "(p_1 + k)^2 = the README's 1 + 2k with k = p_1.k",
                s_inv - L.to_sympy(s_sides[2], {"k": k_rest}),
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (r"= 1 + 2k,\qquad k = p_1", r"= 1 + 4k,\qquad k = p_1"),
            (r"(1 - \beta\cos\theta).", r"(1 + \beta\cos\theta)."),
        ],
    )


def readme_k(gl, e):
    return L.to_sympy(
        L.sides(L.inline(r"k = \gamma\varepsilon(1 - \cos\theta)"))[1],
        {r"\gamma": gl, r"\varepsilon": e, r"\theta": sp.acos(mu)},
    )


def threshold_k():
    found = regexp.search(
        r"if the photon's energy in the lepton's rest frame exceeds "
        r"\$(\d+)m_e c\^2\$",
        L.line("An electron or positron hits a photon and makes"),
    )
    return sp.Integer(found.group(1))


@derivation("tr.k")
def check_k():
    """k = gamma eps (1 - cos theta), the photon's energy in the lepton's rest
    frame for gamma >> 1, must exceed 4."""
    gl, gb, ct, st, p1, ph = lab_lepton()
    k_rest = mdot(p1, ph)
    readme = readme_k(gl, eps).subs(mu, ct)
    s_inv = mdot(*[[a + b for a, b in zip(p1, ph)]] * 2)
    s_of = lambda kv: readme_threshold()[2].subs(k, kv)
    ratio = sp.limit(readme / k_rest, sp.Symbol("v", positive=True), sp.oo)
    with L.patched(r"exceeds $4m_e c^2$", r"exceeds $2m_e c^2$"):
        broken = threshold_k()
    found = regexp.search(
        r"\$s = (\d+)\$, so \$k = (\d+)\$", L.line(r"$s = 9$, so $k = 4$")
    )
    s_t, k_t = sp.Integer(found.group(1)), sp.Integer(found.group(2))
    return [
        exact(
            "p_1.k = gamma eps (1 - beta cos theta)",
            k_rest - gl * eps * (1 - gb / gl * ct),
        ),
        Result(
            "the README's k / p_1.k -> 1 for gamma -> oo at any cos "
            "theta < 1 (gamma >> 1)",
            sp.simplify(ratio) == 1,
            f"limit {sp.simplify(ratio)}",
        ),
        exact(
            "s = (p_1 + k)^2 = 1 + 2 p_1.k, the README's s = 1 + 2k "
            "(Lee et al.)",
            s_inv - s_of(k_rest),
        ),
        exact(
            "threshold: s = (3 m_e)^2 (three leptons at rest in the CM "
            "frame) at the README's k = 4 m_e c^2",
            s_of(threshold_k()) - 9,
        ),
        exact(
            "the Kinematics text: s = 9 = (3 m_e)^2 at its k = 4, "
            "the threshold of the first paragraph",
            (s_t - 3**2) ** 2
            + (s_of(k_t) - s_t) ** 2
            + (k_t - threshold_k()) ** 2,
        ),
        exact_fails("control: 2 m_e c^2", s_of(broken) - 9),
    ]


# ----------------------- the exact cross section: eq. 4.3's J_11
def lee_yz():
    return [L.to_sympy(L.sides(t.strip("$"))[1], {"s": s}) for t in LEE_YZ]


def lee_kernel(n, t_):
    """K_n(s, s~) of eq. 4.3: J_n(s) = int_9^s ds~ J_1(s~) K_n(s, s~) (every
    term of J_n is such an integral)."""
    tex = (
        "".join(LEE_43[n])
        .split("=", 1)[1]
        .replace(r"\intop_9^s d\tilde{s}", " ")
    )
    return L.to_sympy(tex, {"s": s, r"\tilde{s}": t_, r"J_1(\tilde{s})": 1})


def lee_derivative(n):
    js = {j: sp.Symbol(f"J{j}") for j in range(1, 19)}
    y_, z_ = lee_yz()
    tokens = {"s": s, "y": y_, "z": z_} | {
        (f"J_{j}" if j < 10 else f"J_{{{j}}}"): js[j] for j in js
    }
    expr = sp.expand(
        L.to_sympy(LEE_42[n].split("=", 1)[1], tokens).subs(
            {js[j]: 0 for j in range(13, 19)}
        )
    )
    assert sp.expand(expr - sum(expr.coeff(js[j]) * js[j] for j in js)) == 0, (
        expr
    )  # linear in the J_j
    return {j: expr.coeff(js[j]) for j in js if expr.coeff(js[j]) != 0}


@derivation("tr.J11")
def check_J11():
    """Eq. 4.2 requires J_11' = z J_9/(s - 5), but eq. 4.3's J_11 gives half of
    it. The other integrals of eq. 4.3 solve eq. 4.2 for any J_1."""
    t_ = sp.Symbol("t", positive=True)  # s~
    snippets = [p for n in LEE_43 for p in LEE_43[n]] + list(LEE_42.values())
    out = [
        in_source(
            "eqs. 4.2-4.3 as transcribed, y and z, J_13 ... J_18 = 0",
            LEE,
            *snippets,
            *LEE_YZ,
            LEE_ZERO,
        )
    ]
    kernels = {n: lee_kernel(n, t_) for n in LEE_43}
    r = rng(11)
    pts = []
    for _ in range(6):
        sv = 9 + mp.mpf(r.random()) * 60
        pts.append((sv, 9 + (sv - 9) * mp.mpf(r.random())))
    good, half, tol = [], [], mp.mpf(10) ** -40
    for n, kn in kernels.items():
        d = lee_derivative(n)
        assert set(d) <= {1} | set(kernels), d
        f = sp.lambdify((s, t_), sp.diff(kn, s), "mpmath")
        g = sp.lambdify(
            (s, t_),
            sum(cf * kernels[j] for j, cf in d.items() if j != 1),
            "mpmath",
        )
        edge = sp.lambdify(s, kn.subs(t_, s) - d.get(1, 0), "mpmath")
        with mp.workdps(50):
            # (worst_of: max() drops a NaN after the first point)
            worst_edge = worst_of(abs(edge(p[0])) for p in pts)
            if worst_of(abs(f(*p) - g(*p)) for p in pts) < tol:
                good.append(n)
            if (
                worst_of(abs(f(*p) - g(*p) / 2) for p in pts) < tol
                and worst_of(abs(g(*p)) for p in pts) > tol
            ):
                half.append(n)
        out.append(
            Result(
                f"J_{n}: boundary term K_{n}(s, s) = eq. 4.2's "
                "coefficient of J_1",
                worst_edge < tol,
                f"max difference {mp.nstr(worst_edge, 3)}",
            )
        )
    out.append(
        Result(
            "dK_n/ds = eq. 4.2's integrand for every n but 11; "
            "for n = 11 it is half (50 digits, 6 points)",
            half == [11] and good == sorted(set(kernels) - {11}),
            f"agree: {good}; half: {half}",
        )
    )
    y_, z_ = lee_yz()
    J9 = sp.Symbol("J9")

    def claims():
        lhs, right = L.sides(L.after("which requires $"))
        z_readme = rhs(r"z = \sqrt{(s-5)/(s-1)}", {"s": s})
        ours = L.to_sympy(right, {"z": z_, "J_9": J9, "s": s})
        log_ = L.to_sympy(L.inline(r"\ln(\sqrt{s-5} + \sqrt{s-1})"), {"s": s})
        dlog = L.to_sympy(L.inline(r"1/[2\sqrt{(s-5)(s-1)}]"), {"s": s})
        zs = L.to_sympy(L.after("That is half of $"), {"z": z_, "s": s})
        pts9 = [(9 + mp.mpf(r.random()) * 100,) for _ in range(5)]
        return [
            Result("README J_11' (verbatim)", lhs == "J_{11}'", lhs),
            exact(
                "README J_11' = eq. 4.2's z J_9/(s - 5)",
                ours - lee_derivative(11)[9] * J9,
            ),
            exact("README z = eq. 4.4's z", z_readme - z_),
            numeric(
                "README: d/ds ln(sqrt(s-5) + sqrt(s-1)) = 1/[2 "
                "sqrt((s-5)(s-1))]",
                sp.lambdify(s, sp.diff(log_, s), "mpmath"),
                sp.lambdify(s, dlog, "mpmath"),
                pts9,
            ),
            numeric(
                "README: that is half of z/(s - 5)",
                sp.lambdify(s, dlog / zs, "mpmath"),
                lambda sv: mp.mpf(1) / 2,
                pts9,
            ),
        ]

    return (
        out
        + claims()
        + _controls(
            claims,
            [
                (r"$J_{11}' = zJ_9/(s - 5)$", r"$J_{11}' = zJ_9/(2s - 10)$"),
                (r"half of $z/{(s - 5)}$", r"half of $z/{(s - 3)}$"),
            ],
        )
    )


# ----------------------- the threshold expansion
def readme_threshold():
    tex = L.display(THRESHOLD)
    return (
        L.to_sympy(L.sides(L.parts(tex)[0])[1], {"r": sp.Symbol("r")}),
        L.to_sympy(L.sides(L.part(tex, "r ="))[1], {"s": s}),
        L.to_sympy(L.sides(L.part(tex, "s ="))[1], {"k": k}),
        tex,
    )


def _equation_text(tex, head):
    flat = regexp.sub(r"\s+", "", tex)
    start = flat.index(regexp.sub(r"\s+", "", head))
    return flat[start : flat.index(r"\end{equation}", start)]


@derivation("tr.alpha3")
def check_alpha3():
    """Lee et al.'s cross sections in alpha^3 (hbar = c = m_e = 1) are in units
    of alpha r_e^2: alpha^3 hbar^2/(m_e c)^2 = alpha r_e^2 with r_e = alpha
    hbar/(m_e c) = q^2/(m_e c^2). Eq. 4.5 omits the alpha^3 of eqs. 4.4 and
    4.11-4.12."""
    q, hb, me = sp.symbols("q_e hbar m_e", positive=True)
    # Gaussian units
    alpha_def, re_def = q**2 / (hb * c), q**2 / (me * c**2)
    tokens = {r"\alpha": alpha_def, r"\hbar": hb, "m_e": me, "c": c}
    tex = latex.tex_source(LEE)
    if tex is None:
        omitted = absent("eq. 4.5 has no alpha^3", LEE)
    else:
        eq45 = _equation_text(tex, LEE_HEADS["4.5"])
        omitted = Result(
            "eq. 4.5 has no alpha^3", r"\alpha" not in eq45, eq45[:60]
        )

    def claims():
        r_e = rhs(r"r_e = \alpha\hbar/(m_e c)", tokens)
        a3 = relation(
            None,
            tokens | {"r_e": re_def},
            text=L.after("A cross section $\\alpha^3$ is then $"),
        )
        return [
            exact(
                "README r_e = alpha hbar/(m_e c) = q^2/(m_e c^2) "
                "(alpha = q^2/(hbar c))",
                r_e - re_def,
            ),
            exact(
                "README alpha^3 hbar^2/(m_e c)^2 = alpha r_e^2", a3[0] - a3[1]
            ),
        ]

    return (
        [
            in_source(
                "eqs. 4.4, 4.11 and 4.12 carry alpha^3 (as transcribed)",
                LEE,
                *LEE_HEADS.values(),
            ),
            omitted,
        ]
        + claims()
        + _controls(
            claims,
            [(r"$\alpha^3\hbar^2/(m_e c)^2", r"$\alpha^2\hbar^2/(m_e c)^2")],
        )
    )


@derivation("tr.threshold")
def check_threshold():
    """The threshold expansion is Lee et al.'s eq. 4.5 (quoted as printed, to
    O(r^5)) in r = (s - 9)/(s - 1) = 1 - 4/k (s = 1 + 2k), in units of alpha
    r_e^2."""

    def claims():
        ours, r_def, s_def, _ = readme_threshold()
        theirs = lee(LEE_45.replace("+O(r^5)", ""), "r")
        left, right = L.sides(
            L.parts(L.display(r"\sigma = \frac{\pi}{3\sqrt3}\left["))[0]
        )
        quoted = L.to_sympy(
            right.replace(" + O(r^5)", ""), {"r": sp.Symbol("r")}
        )
        r_inline = rhs(r"r = (s-9)/(s-1)", {"s": s})
        r_k = rhs(None, {"k": k}, text=L.after("In terms of $k$, $"))
        return [
            Result(
                "README's quote of eq. 4.5: sigma = ... + O(r^5)",
                left == r"\sigma" and " + O(r^5)" in right,
                left,
            ),
            exact("README's quote = eq. 4.5 (to O(r^5))", quoted - theirs),
            exact(
                "README expansion = eq. 4.5 in r, in units of alpha r_e^2",
                ours - quoted,
            ),
            exact("README r(s) = (s - 9)/(s - 1)", r_def - (s - 9) / (s - 1)),
            exact("README s(k) = 1 + 2k (k = (s - 1)/2)", s_def - (1 + 2 * k)),
            exact(
                "README inline r = (s - 9)/(s - 1), the display's",
                r_inline - r_def,
            ),
            exact("README r = 1 - 4/k = r(s(k))", r_k - r_def.subs(s, s_def)),
        ]

    return (
        [
            in_source(
                "eq. 4.5, r = (s - 9)/(s - 1) and k = (s - 1)/2 as transcribed",
                LEE,
                LEE_45,
                LEE_R,
                LEE_K,
            )
        ]
        + claims()
        + _controls(
            claims,
            [
                (
                    r"\left(\frac{4r^2}{27} + \frac{602\thinspace r^3}{243}",
                    r"\left(\frac{4r^2}{27} + \frac{602\thinspace r^3}{343}",
                ),
                (r"{4374} + O(r^5)", r"{4375} + O(r^5)"),
                (r"$r = 1 - 4/k$", r"$r = 1 - 2/k$"),
            ],
        )
    )


@derivation("tr.votruba")
def check_votruba():
    """The threshold expansion's leading term for k -> 4 is Votruba's
    approximation, Haug's eq. 9.1, whose values his Table 1 lists next to the
    exact ones."""

    def claims():
        ours = L.to_sympy(
            L.after(r"Its leading term for $k \to 4$, $"),
            {r"\alpha": al, "r_e": re, "k": k},
        )
        # sigma/(alpha r_e^2) of eq. 4.5, in k
        low = readme_expansion_forms()[0]
        lead = sp.limit(low / (k - 4) ** 2, k, 4)
        haug = L.to_sympy(
            L.sides(HAUG_91)[1], {r"\alpha": al, "r_0": re, "k": k}
        )
        return [
            exact(
                "README term = alpha r_e^2 (k - 4)^2 times the limit k "
                "-> 4 of eq. 4.5 / (k - 4)^2",
                ours - al * re**2 * lead * (k - 4) ** 2,
            ),
            exact(
                "= Haug's eq. 9.1 (p. 1109, transcribed) with r_0 = r_e",
                ours - haug,
            ),
        ]

    code = mojo_constants.read(("ALPHA_RE2", "SIGMA_T"))
    # alpha r_e^2 in 10^-10 b (1 b = 1e-24 cm^2)
    unit = code["ALPHA_RE2"] * code["SIGMA_T"] * 10**34
    haug = (
        L.to_sympy(L.sides(HAUG_91)[1], {r"\alpha": 1, "r_0": 1, "k": k}) * unit
    )
    worst = max(
        abs(sp.N(haug.subs(k, sp.Rational(kv)), 30) - sp.Rational(v))
        / sp.Rational(1, 10 ** len(v.split(".")[1]))
        for kv, v in HAUG_SIGMA_V
    )
    return (
        claims()
        + [
            Result(
                "eq. 9.1 gives the sigma_V row of Haug's Table 1 "
                "(p. 1110) within one unit of its last digit "
                "(alpha r_e^2 of PDG 2026)",
                worst < 1,
                f"largest difference {float(worst):.2f} unit",
            )
        ]
        + _controls(
            claims,
            [(r"(\sqrt3\pi/972)\thinspace", r"(\sqrt3\pi/486)\thinspace")],
        )
    )


# ----------------------- the high-energy expansions
def readme_alpha3_form(pin, var):
    left, right = L.sides(L.parts(L.display(pin))[0])
    assert left == r"\sigma", left
    return L.to_sympy(right, {var: {"s": s, "k": k}[var], r"\alpha": al} | ZETA)


@derivation("tr.eq411")
def check_eq411():
    """The README's expansion in 1/s is Lee et al.'s eq. 4.11."""

    def claims():
        return [
            exact(
                "README = alpha^3 x eq. 4.11",
                readme_alpha3_form(EQ411, "s") - al**3 * lee(LEE_411, "s"),
            )
        ]

    return (
        [in_source("eq. 4.11 as transcribed", LEE, *LEE_411)]
        + claims()
        + _controls(claims, [(r"\frac{49}{6}\ln^2 s", r"\frac{49}{3}\ln^2 s")])
    )


@derivation("tr.eq412")
def check_eq412():
    """The README's eq. 4.12 is Lee et al.'s, and it is the README's eq. 4.11 at
    s = 1 + 2k to order 1/k."""

    def claims():
        e411, e412 = (
            readme_alpha3_form(EQ411, "s") / al**3,
            readme_alpha3_form(EQ412, "k") / al**3,
        )
        u = sp.Symbol("u", positive=True)
        ln_s = relation(
            r"\ln s = \ln 2k + \ln[1 + 1/(2k)]", {"s": 1 + 2 * k, "k": k}
        )
        in_k = e411.subs(sp.log(s), Lg + sp.log(1 + u / 2)).subs(s, 1 + 2 / u)
        series = sp.series(in_k, u, 0, 2).removeO()
        return [
            exact(
                "README eq. 4.12 = alpha^3 x Lee et al.'s eq. 4.12",
                e412 - lee(LEE_412, "k"),
            ),
            exact(
                "README: ln s = ln 2k + ln[1 + 1/(2k)] at s = 1 + 2k",
                sp.expand_log(ln_s[0] - ln_s[1], force=True),
            ),
            exact(
                "README eq. 4.11 at s = 1 + 2k, to order 1/k at fixed ln "
                "2k = README eq. 4.12",
                sp.expand(series - e412.subs(sp.log(2 * k), Lg).subs(k, 1 / u)),
            ),
        ]

    return (
        [in_source("eq. 4.12 as transcribed", LEE, *LEE_412)]
        + claims()
        + _controls(
            claims, [(r"\frac{49}{12}\ln^2 2k", r"\frac{49}{6}\ln^2 2k")]
        )
    )


@derivation("tr.high")
def check_high():
    """The code's high-energy expansion is Lee et al.'s eq. 4.13, which rounds
    the README's eq. 4.12 to two decimals."""

    def claims():
        ours = readme_expansion_forms()[1]
        exact_form = high_energy_form(readme_alpha3_form(EQ412, "k") / al**3)
        rounded = high_energy_form(ours)
        worst = worst_of(
            abs(float(exact_form[p].coeff(Lg, j) - rounded[p].coeff(Lg, j)))
            for p in (0, 1)
            for j in range(4)
        )
        return [
            exact("README = eq. 4.13", ours - lee(LEE_413, "k")),
            Result(
                "it rounds the README's eq. 4.12 to two decimals",
                worst <= 0.005,
                f"largest difference of a coefficient {worst:.4f}",
            ),
        ]

    return (
        [in_source("eq. 4.13 as transcribed", LEE, LEE_413)]
        + claims()
        + _controls(claims, [(r"4.08\ln^2 2k", r"4.18\ln^2 2k")])
    )


def readme_expansion_forms():
    in_r, r_def, s_def, low_tex = readme_threshold()
    high_tex = L.display(HIGH)
    high = L.to_sympy(L.sides(L.parts(high_tex)[0])[1], {"k": k})
    ends = [
        L.to_sympy(L.part(tex, "(k").strip("()").split(rel)[1], {})
        for tex, rel in ((low_tex, r"\le"), (high_tex, r"\ge"))
    ]
    return in_r.subs(sp.Symbol("r"), r_def).subs(s, s_def), high, ends


def readme_expansions():
    low, high, ends = readme_expansion_forms()
    return (
        sp.lambdify(k, low, "mpmath"),
        sp.lambdify(k, high, "mpmath"),
        [mp.mpf(str(e)) for e in ends],
    )


@functools.cache
def numeric_expansions():
    """sigma/(alpha r_e^2) of each expansion with its slope
    d ln sigma / d ln(k - 4) (the ends of the code's spline)."""
    *forms, ends = readme_expansion_forms()
    out = []
    for form in forms:
        out += [
            sp.lambdify(k, form, "numpy"),
            sp.lambdify(k, (k - 4) * sp.diff(form, k) / form, "numpy"),
        ]
    return (*out, [float(e) for e in ends])


# ----------------------- the code's sigma(k)
def readme_switches():
    found = regexp.search(
        r"For \$k (\\[lg][et]) ([^$]+)\$ and \$k (\\[lg][et]) ([^$]+)\$, the "
        r"code uses the expansions",
        L.line(r"The Monte Carlo's tables give $\sigma(k)$ at "),
    )
    rel_lo, lo, rel_hi, hi = found.groups()
    return [
        (rel, sp.nsimplify(L.to_sympy(v, {}), rational=True))
        for rel, v in ((rel_lo, lo), (rel_hi, hi))
    ]


def readme_table_ranges():
    out = []
    for cells in L.table(TABLE):
        rng_ = cells[0].strip("$")
        parts = [p.strip() for p in regexp.split(r"\\lt|\\le|\\ge", rng_)]
        signs = regexp.findall(r"\\lt|\\le|\\ge", rng_)
        val = lambda p: sp.nsimplify(L.to_sympy(p, {}), rational=True)
        if len(parts) == 3:
            out.append((val(parts[0]), val(parts[2]), cells[1]))
        elif signs == [r"\ge"]:
            out.append((val(parts[1]), None, cells[1]))
        else:
            out.append((None, val(parts[1]), cells[1]))
    return out


@derivation("tr.table_range")
def check_table_range():
    """The switches to the expansions in the README's prose, table and spline
    are the displays' conditions."""

    def claims():
        ends = [
            sp.nsimplify(e, rational=True) for e in readme_expansion_forms()[2]
        ]
        rows = readme_table_ranges()
        bounds = [(lo, hi) for lo, hi, _ in rows]
        want = [
            (None, threshold_k()),
            (threshold_k(), ends[0]),
            (ends[0], ends[1]),
            (ends[1], None),
        ]
        forms = [form for *_, form in rows]
        found = regexp.search(
            r"At the end nodes, \$k_0 = ([^$]+)\$ and \$k_N = ([^$]+)\$",
            L.line("At the end nodes, $k_0 = "),
        )
        nodes = [
            sp.nsimplify(L.to_sympy(v, {}), rational=True)
            for v in found.groups()
        ]
        return [
            Result(
                "the prose's switches = the display's conditions "
                "(k <= 4.02), (k >= 10^3)",
                readme_switches() == [(r"\le", ends[0]), (r"\ge", ends[1])],
                f"display {', '.join(f'{float(e):g}' for e in ends)}",
            ),
            Result(
                "the spline's end nodes k_0, k_N = the expansions' ends",
                nodes == ends,
                str(nodes),
            ),
            Result(
                "the table's ranges: below threshold, eq. 4.5 to 4.02, "
                "the spline, eq. 4.13 from 10^3",
                bounds == want
                and forms[1] == "eq. 4.5"
                and forms[3] == "eq. 4.13",
                f"{bounds}; {forms}",
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (r"For $k \le 4.02$ and", r"For $k \le 4.05$ and"),
            (
                r"For $k \le 4.02$ and $k \ge 10^3$",
                r"For $k \lt 4.02$ and $k \gt 10^3$",
            ),
            (r"| $4 \lt k \le 4.02$ |", r"| $4 \lt k \le 4.05$ |"),
        ],
    )


@derivation("tr.spline")
def check_spline():
    """The cubic spline of ln sigma in u = ln(k - 4) passes through the nodes
    with second derivatives M_n, and its rows set its slopes. The rows make the
    slope continuous at the inner nodes and equal to the expansions' slopes at
    the ends."""
    u, ua, ub = sp.symbols("u u_a u_b", real=True)
    names = (
        "v_{n-1} v_n v_{n+1} M_{n-1} M_n M_{n+1} h_{n-1} h_n v_0 v_1 v_N "
        "v_{N-1} M_0 M_1 M_N M_{N-1} h_0 h_{N-1} v_0' v_N'"
    )
    S = {nm: sp.Symbol(nm) for nm in names.split()}

    def piece(va, vb, ma, mb, h):
        tex = display_rhs(SPLINE)
        t_def = rhs(r"t = (u - u_n)/h_n", {"u": u, "u_n": ua, "h_n": h})
        h_def = rhs(r"h_n = u_{n+1} - u_n", {"u_{n+1}": ub, "u_n": ua})
        v = L.to_sympy(
            tex,
            {
                "t": sp.Symbol("t"),
                "v_n": va,
                "v_{n+1}": vb,
                "h_n": h,
                "M_n": ma,
                "M_{n+1}": mb,
            },
        )
        return v.subs(sp.Symbol("t"), t_def), h_def

    def claims():
        h = sp.Symbol("h", positive=True)
        v, h_def = piece(S["v_n"], S["v_{n+1}"], S["M_n"], S["M_{n+1}"], h)
        at = lambda e, uv: e.subs(u, uv)
        d2 = sp.diff(v, u, 2)
        out = [
            exact("README h_n = u_{n+1} - u_n", (h_def - h).subs(ub, ua + h)),
            exact(
                "v(u_n) = v_n and v(u_n + h_n) = v_{n+1}",
                (at(v, ua) - S["v_n"]) ** 2
                + (at(v, ua + h) - S["v_{n+1}"]) ** 2,
            ),
            exact(
                "v'' = M_n at u_n, M_{n+1} at u_{n+1}, linear between",
                (at(d2, ua) - S["M_n"]) ** 2
                + (at(d2, ua + h) - S["M_{n+1}"]) ** 2
                + sp.diff(d2, u, 2) ** 2,
            ),
        ]
        # the slopes at an inner node: the piece before it ends there, the piece
        # after it starts there
        left, _ = piece(
            S["v_{n-1}"], S["v_n"], S["M_{n-1}"], S["M_n"], S["h_{n-1}"]
        )
        right, _ = piece(
            S["v_n"], S["v_{n+1}"], S["M_n"], S["M_{n+1}"], S["h_n"]
        )
        jump = sp.diff(right, u).subs(u, ua) - sp.diff(left, u).subs(
            ua, ua - S["h_{n-1}"]
        ).subs(u, ua)
        row = [
            L.to_sympy(side, S)
            for side in L.sides(
                L.parts(L.display(r"= 6\left(\frac{v_{n+1} - v_n}{h_n}"))[0]
            )
        ]
        out.append(
            exact(
                "6 x (slope after - slope before the node) = README "
                "row: RHS - LHS",
                6 * jump - (row[1] - row[0]),
            )
        )
        first, last = L.parts(L.display(r"h_{N-1}M_{N-1} + 2h_{N-1}M_N"))
        start, _ = piece(S["v_0"], S["v_1"], S["M_0"], S["M_1"], S["h_0"])
        end, _ = piece(
            S["v_{N-1}"], S["v_N"], S["M_{N-1}"], S["M_N"], S["h_{N-1}"]
        )
        slope0 = sp.diff(start, u).subs(u, ua)
        slopeN = sp.diff(end, u).subs(u, ua + S["h_{N-1}"])
        r0 = [L.to_sympy(x_, S) for x_ in L.sides(first)]
        rN = [L.to_sympy(x_, S) for x_ in L.sides(last)]
        out += [
            exact(
                "README first row: LHS - RHS = 6 (v_0' - the slope at u_0)",
                sp.expand(6 * (S["v_0'"] - slope0) - (r0[0] - r0[1])),
            ),
            exact(
                "README last row: LHS - RHS = 6 (the slope at u_N - v_N')",
                sp.expand(6 * (slopeN - S["v_N'"]) - (rN[0] - rN[1])),
            ),
        ]
        return out

    return claims() + _controls(
        claims,
        [
            (
                r"+ 2\thinspace (h_{n-1} + h_n)\thinspace M_n",
                r"+ (h_{n-1} + h_n)\thinspace M_n",
            ),
            (r"2h_0M_0 + h_0M_1", r"h_0M_0 + h_0M_1"),
        ],
    )


@derivation("tr.slopes")
def check_slopes():
    """The README's end slopes of the spline, d ln sigma/d ln(k - 4), are those
    of its expansions (eq. 4.5 in r = 1 - 4/k, and eq. 4.13 with its cubic
    Q(L), L = ln 2k)."""
    r_ = sp.Symbol("r")
    in_r = readme_threshold()[0]
    low, high, _ = readme_expansion_forms()

    def claims():
        p_def = rhs(r"p(r) = 4/27 + 602r/243 + 4217r^2/4374", {"r": r_})
        drdk = rhs(r"dr/dk = 4/k^2", {"k": k})
        r_k = rhs(r"r = 1 - 4/k", {"k": k})
        slope_low = L.to_sympy(
            display_rhs(r"= \frac{4\thinspace (k-4)}{k^2}"),
            {"k": k, "r": r_},
            funcs={
                "p'": lambda a: sp.diff(p_def, r_).subs(r_, a),
                "p": lambda a: p_def.subs(r_, a),
            },
        )
        Q = sp.expand(
            (
                -k
                * (
                    high
                    - sp.Rational(28, 9) * sp.log(2 * k)
                    + sp.Rational(218, 27)
                )
            ).subs(sp.log(2 * k), Lg)
        )
        form = rhs(
            r"\sigma/(\alpha r_e^2) = 28L/9 - 218/27 - Q(L)/k",
            {"L": Lg, "k": k},
            funcs={"Q": lambda a: Q.subs(Lg, a)},
        )
        L_def = rhs(r"L = \ln 2k", {"k": k})
        dLdk = rhs(r"dL/dk = 1/k", {"k": k})
        slope_high = L.to_sympy(
            display_rhs(r"= \frac{k-4}{\sigma}\left(\frac{28}{9k}"),
            {"k": k, r"\sigma": high, "L": L_def},
            funcs={
                "Q'": lambda a: sp.diff(Q, Lg).subs(Lg, a),
                "Q": lambda a: Q.subs(Lg, a),
            },
        )
        return [
            exact(
                "README p(r) = the bracket of eq. 4.5 / r^2",
                sp.pi / (3 * sp.sqrt(3)) * r_**2 * p_def - in_r,
            ),
            Result(
                "Q is a polynomial in L = ln 2k (no k left)",
                not Q.has(k),
                str(Q),
            ),
            exact(
                "README: sigma/(alpha r_e^2) = 28L/9 - 218/27 - Q(L)/k "
                "is eq. 4.13",
                form.subs(Lg, L_def) - high,
            ),
            exact("README dr/dk = d(1 - 4/k)/dk", drdk - sp.diff(r_k, k)),
            exact("README dL/dk = d ln 2k/dk", dLdk - sp.diff(L_def, k)),
            exact(
                "README slope of eq. 4.5 = (k - 4) d ln sigma/dk",
                sp.together(
                    slope_low.subs(r_, r_k) - (k - 4) * sp.diff(sp.log(low), k)
                ),
            ),
            exact(
                "README slope of eq. 4.13 = (k - 4) d ln sigma/dk",
                sp.together(slope_high - (k - 4) * sp.diff(sp.log(high), k)),
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (
                r"\left(\frac{2}{r} + \frac{p'(r)}{p(r)}\right)",
                r"\left(\frac{1}{r} + \frac{p'(r)}{p(r)}\right)",
            ),
            (r"\frac{Q'(L) - Q(L)}{k^2}", r"\frac{Q'(L) + Q(L)}{k^2}"),
        ],
    )


@derivation("tr.sigma_T")
def check_sigma_T():
    """alpha r_e^2/sigma_T = 3 alpha/(8 pi) with sigma_T = 8 pi r_e^2/3
    (Thomson), and c sigma_T = 1 in AM3's code units."""

    def claims():
        sT_def = rhs(r"\sigma_T = 8\pi r_e^2/3", {"r_e": re})
        ratio = relation(
            None,
            {r"\alpha": al, "r_e": re, r"\sigma_T": sT_def},
            text=L.after("so they use $"),
        )
        found = regexp.search(
            r"lengths in \$(.+?)\$ and times in \$(.+?)\$",
            L.line("AM³'s code units make"),
        )
        length, time = (
            L.to_sympy(found.group(i), {r"\sigma_T": sT, "c": c})
            for i in (1, 2)
        )
        one = relation(
            r"c\sigma_T = 1",
            {"c": c * time / length, r"\sigma_T": sT / length**2},
        )
        return [
            exact(
                "README alpha r_e^2/sigma_T = 3 alpha/(8 pi) with "
                "sigma_T = 8 pi r_e^2/3",
                ratio[0] - ratio[1],
            ),
            exact(
                "c sigma_T in the README's code units (lengths, times) = 1",
                one[0] - one[1],
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (r"= 3\alpha/(8\pi)$", r"= 3\alpha/(4\pi)$"),
            (r"times in $\sqrt{\sigma_T}/c$", r"times in $\sqrt{\sigma_T}c$"),
        ],
    )


# ----------------------- energies of the products
def readme_x(pk, p1k):
    tex = L.sides(L.display(r"x = \frac{"))[1].replace(r"\cdot", " ")
    return L.to_sympy(
        tex, {"p": sp.Symbol("pk"), "p_1": sp.Symbol("p1k"), "k": 1}
    ).subs({sp.Symbol("pk"): pk, sp.Symbol("p1k"): p1k})


@derivation("tr.x")
def check_x():
    """x = p.k/p_1.k is a product's share of the lepton's lab energy for
    gamma >> 1 and eps << gamma.
    The two differ by exactly p'.(n' + beta z), with p' the product's momentum
    and n' the photon's direction (rest frame)."""
    L.line(
        r"For $\gamma \gg 1$ and $\varepsilon \ll \gamma$, each product "
        r"carries the fraction"
    )
    gl, gb, ct, st, p1, ph = lab_lepton()
    be = gb / gl
    E_, px, py, pz = sp.symbols("E p_x p_y p_z", real=True)
    lab = (gl * E_ + gb * pz, px, py, gl * pz + gb * E_)
    xv = readme_x(mdot(lab, ph), mdot(p1, ph))
    share = lab[0] / gl
    rest = (gl * ph[0] - gb * ph[3], ph[1], ph[2], gl * ph[3] - gb * ph[0])
    kk = rest[0]
    nh = [q / kk for q in rest[1:]]
    d = [nh[0], nh[1], nh[2] + be]
    along = sum(a * b for a, b in zip(d, nh))
    across2 = sum(a * a for a in d) - along**2  # |n'| = 1 (k is null)
    with L.patched(
        r"x = \frac{p\cdot k}{p_1\cdot k}", r"x = \frac{p_1\cdot k}{p\cdot k}"
    ):
        broken = readme_x(mdot(lab, ph), mdot(p1, ph))
    return [
        exact(
            "x = E' - p'.n' in the rest frame (n' the photon's direction): "
            "it depends on the products and the photon there, not on the "
            "photon's lab direction",
            xv - (E_ - px * nh[0] - py * nh[1] - pz * nh[2]),
        ),
        exact(
            "lab share - x = p'.(n' + beta z) (the lab share is E' + "
            "beta p'_z)",
            share - xv - (px * d[0] + py * d[1] + pz * d[2]),
        ),
        exact(
            "(n' + beta z).n' = eps/(gamma k), k = p_1.k: along the photon, "
            "Delta = p'_par eps/(gamma k) <= eps/gamma "
            "(|p'| < E' <= k - 1 of the 1 + k a product can take)",
            along - eps / (gl * kk),
        ),
        exact(
            "across: |n' + beta z|_perp^2 = (beta eps sin theta/k)^2 "
            "= beta^2 eps sin^2 theta/(gamma k (1 - beta cos theta)) "
            "(|n'| = 1)",
            (across2 - (be * eps * st / kk) ** 2) ** 2
            + (across2 - be**2 * eps * st**2 / (gl * kk * (1 - be * ct))) ** 2
            + (sum(q * q for q in nh) - 1) ** 2,
        ),
        exact(
            "2 (1 - beta cos theta) - sin^2 theta = (cos theta - beta)^2 + "
            "1/gamma^2 >= 0: |Delta_perp| <= p'_perp "
            "(2 eps/(gamma k))^(1/2), k >= 4: both parts vanish for gamma >> "
            "1 and eps << gamma",
            2 * (1 - be * ct) - st**2 - ((ct - be) ** 2 + 1 / gl**2),
        ),
        exact_fails(
            "control: x = p_1.k/p.k",
            share - broken - (px * d[0] + py * d[1] + pz * d[2]),
        ),
    ]


@derivation("tr.x_rest")
def check_x_rest():
    """x = p.k/p_1.k is the README's E - p_z in the lepton's rest frame with the
    photon along z."""
    E_, px, py, pz, w_ = sp.symbols("E p_x p_y p_z omega", real=True)
    kv = (w_, 0, 0, w_)
    x_ = readme_x(mdot((E_, px, py, pz), kv), mdot((1, 0, 0, 0), kv))

    def rest():
        left, right = L.sides(
            L.after(
                r"In the lepton's rest frame, with the "
                r"photon along the $z$ axis, we have $"
            )
        )
        assert left == "x", left
        return L.to_sympy(right, {"E": E_, "p_z": pz})

    with L.patched(r"we have $x = E - p_z$", r"we have $x = E + p_z$"):
        broken = rest()
    return [
        exact("x = the README's E - p_z", sp.simplify(x_ - rest())),
        exact_fails("control: E + p_z in the README", sp.simplify(x_ - broken)),
    ]


@derivation("tr.xi")
def check_xi():
    """x_+ + x_low + x_high = 1 (momentum conservation, k^2 = 0), so the
    README's xi = 1 - x_high = x_+ + x_low."""
    a, b = (sp.symbols(f"{nm}0:4", real=True) for nm in "ab")
    w_ = sp.Symbol("omega", positive=True)
    p1, kv = (1, 0, 0, 0), (w_, 0, 0, w_)
    c_ = tuple(p1[i] + kv[i] - a[i] - b[i] for i in range(4))
    xp, xl, xh = (mdot(q, kv) / mdot(p1, kv) for q in (a, b, c_))
    xi = sp.Symbol("xi")

    def claims():
        left, lost, products = relation(
            None,
            {
                r"\xi": xi,
                "x_+": xp,
                r"x_\mathrm{low}": xl,
                r"x_\mathrm{high}": xh,
            },
            text=L.after("This lepton loses the fraction $"),
        )
        assert left == xi, left
        return [
            exact(
                "the README's 1 - x_high = x_+ + x_low with the x of the three "
                "products",
                sp.simplify(lost - products),
            )
        ]

    return (
        [exact("sum of the three x = 1", sp.simplify(xp + xl + xh - 1))]
        + claims()
        + _controls(
            claims,
            [
                (XI, XI.replace(r"x_+ + x_", r"x_+ - x_")),
                (XI, XI.replace(r"1 - x_", r"1 + x_")),
            ],
        )
    )


@derivation("tr.lab")
def check_lab():
    """A product's exact lab energy is gamma x plus the two terms that the code
    neglects, and these terms have the README's bounds. The energy comes from
    the boost of p' = p_z n' + p_perp (cos phi e_1 + sin phi e_2) to the lab.
    Here n' is the photon's direction in the rest frame, and e_1 points toward
    the boost axis."""
    gl, gb, ct, st, p1, ph = lab_lepton()
    be = gb / gl
    w_ = sp.Symbol("w", positive=True)
    cphi, sphi = (1 - w_**2) / (1 + w_**2), 2 * w_ / (1 + w_**2)
    Ep, pz, pp = sp.symbols("E' p_z p_perp", real=True)
    kk = mdot(p1, ph)  # k (tr.s)
    rest = (gl * ph[0] - gb * ph[3], ph[1], ph[2], gl * ph[3] - gb * ph[0])
    # the photon's direction in the rest frame (aberration)
    n_ = (st / (gl * (1 - be * ct)), 0, (ct - be) / (1 - be * ct))
    # normal to n', in the plane of n' and z, e_1.z > 0
    e1 = (-n_[2], 0, n_[0])
    p_vec = [
        pz * n_[a] + pp * (cphi * e1[a] + sphi * (0, 1, 0)[a]) for a in range(3)
    ]
    E_lab = gl * Ep + gb * p_vec[2]

    def claims():
        readme = L.to_sympy(
            display_rhs(r"E_\mathrm{lab} = \gamma x"),
            {
                r"\gamma": gl,
                "x": Ep - pz,
                r"\varepsilon": eps,
                "k": kk,
                "p_z": pz,
                r"\beta": be,
                r"\theta": sp.acos(ct),
                r"p_\perp": pp,
                r"\phi": sp.acos(cphi),
            },
        )
        return [
            exact(
                "README E_lab = the boost of the product (x = E' - p_z, "
                "tr.x_rest)",
                readme - E_lab,
            )
        ]

    third = be * st / (1 - be * ct) * pp
    bound = L.to_sympy(
        L.inline(r"p_\perp\sqrt{2\gamma\varepsilon/k}"),
        {r"p_\perp": pp, r"\gamma": gl, r"\varepsilon": eps, "k": kk},
    )
    sin2 = [
        L.to_sympy(side.strip(), {r"\theta": sp.acos(ct), r"\beta": be})
        for side in L.inline(r"\sin^2\theta \le 2(1 - \beta\cos\theta)").split(
            r"\le"
        )
    ]
    pz_bound = L.inline(r"\vert p_z\vert \lt k")
    return (
        [
            exact(
                "n' = the photon's direction in the rest frame, and e_1, "
                "e_2 = y unit vectors normal to it",
                sum((rest[a + 1] / rest[0] - n_[a]) ** 2 for a in range(3))
                + (sum(q * q for q in e1) - 1) ** 2
                + sum(a_ * b_ for a_, b_ in zip(e1, n_)) ** 2,
            ),
            Result(
                "e_1 points towards the lepton's direction of motion: "
                "e_1.z = sin theta' > 0",
                True,
                f"e_1.z = {sp.simplify(e1[2])} (u, v > 0)",
            ),
        ]
        + claims()
        + [
            exact(
                "|p_z| < k: a product's E' <= k - 1 (the others have E >= 1 "
                "of the 1 + k), so p_z^2 < (k - 1)^2 - 1 + 1 < k^2 ("
                + pz_bound
                + "): the second term is below eps",
                sp.expand(k**2 - (k - 1) ** 2) - (2 * k - 1),
            ),
            exact(
                "README: 2 (1 - beta cos theta) - sin^2 theta = "
                "(cos theta - beta)^2 + 1/gamma^2 >= 0",
                sin2[1] - sin2[0] - ((ct - be) ** 2 + 1 / gl**2),
            ),
            exact(
                "README bound^2 - (third term)^2 = p_perp^2 "
                "[(cos theta - beta)^2 + (1 + sin^2 theta)/gamma^2]/(1 - "
                "beta cos theta)^2 >= 0",
                sp.together(
                    (bound**2 - third**2) * (1 - be * ct) ** 2
                    - pp**2 * ((ct - be) ** 2 + (1 + st**2) / gl**2)
                ),
            ),
        ]
        + _controls(
            claims,
            [
                (
                    r"\frac{\varepsilon}{k}\thinspace p_z",
                    r"\frac{\varepsilon}{2k}\thinspace p_z",
                ),
                (
                    r"{1 - \beta\cos\theta}\thinspace p_\perp",
                    r"{1 + \beta\cos\theta}\thinspace p_\perp",
                ),
            ],
        )
    )


@derivation("tr.laws")
def check_laws():
    """The extrapolation's laws: the middle one, (a_0 ln y + b_0)/y, joins
    k f_r = A(x) ln k + B(x) at x = 10^-5."""
    Af, Bf = sp.Function("A"), sp.Function("B")
    y_, a0, b0 = sp.symbols("y a_0 b_0", positive=True)

    def claims():
        first, second = L.parts(L.display(LAWS))
        law, cond = second.split(r"\quad")
        left, right = L.sides(law)
        X = L.to_sympy(L.between(cond, "(", ")").split(r"\ge")[1], {})
        f2 = L.to_sympy(
            right, {"x": x, "k": k}, funcs={"A": Af, "B": Bf}
        ) / L.to_sympy(left, {"k": k, "f_r": 1})
        mid = rhs(r"f_r = (a_0\ln y + b_0)/y", {"a_0": a0, "b_0": b0, "y": y_})
        found = regexp.search(
            r"with \$a_0 = ([^$]+)\$ and \$b_0 = ([^$]+)\$",
            L.line(r"$f_r = (a_0\ln y + b_0)/y$"),
        )
        a0v = L.to_sympy(found.group(1), {}, funcs={"A": Af})
        b0v = L.to_sympy(found.group(2), {"a_0": a0v}, funcs={"B": Bf})
        join = L.to_sympy(
            L.after("the middle law joins the second law at $").split("=")[1],
            {},
        )
        return [
            Result(
                "the join: x = 10^-5, the end of the second law",
                join == X,
                f"{join}, {X}",
            ),
            exact(
                "(a_0 ln y + b_0)/y = (A(x) ln k + B(x))/k at y = x k, "
                "x = 10^-5, any A, B and k",
                sp.expand_log(
                    mid.subs({a0: a0v, b0: b0v, y_: X * k}) - f2.subs(x, X),
                    force=True,
                ),
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (
                r"b_0 = 10^{-5}B(10^{-5}) - a_0\ln 10^{-5}",
                r"b_0 = 10^{-5}B(10^{-5}) + a_0\ln 10^{-5}",
            )
        ],
    )


# ----------------------- the rates and the kinetic terms against a direct count
mu = sp.Symbol("mu", real=True)  # cos theta
# y = xk and the target lepton's gamma'
y, gp = sp.symbols("y gamma'", positive=True)
SIG = sp.Function("sigma")  # the cross section: any function, 0 below k = 4
# the distributions f_r(k, y) per ln y: any functions
F = {r: sp.Function(f"f_{r}") for r in ("r", "+", "low")}
# their mean fractions <x_r> at fixed k
X = {r: sp.Function(f"X_{r}") for r in ("+", "low")}
# e-, e+ per ln gamma
DENS = {"-": sp.Function("n_-"), "+": sp.Function("n_+")}
NG = sp.Function("n_gamma")  # photons per ln eps
CONJUGATE = {"-": "+", "+": "-"}
# an e- target makes a positron (x_+) and an electron of lower energy (x_low)
PRODUCTS = {"-": {"+": "+", "-": "low"}}
# an e+ target makes their charge conjugates
PRODUCTS["+"] = {CONJUGATE[q]: r for q, r in PRODUCTS["-"].items()}
TOKENS = {r"\gamma": gam, r"\varepsilon": eps, "c": c, r"\sigma_T": sT}
# Test functions for the counts: sigma with the threshold, and f_+ and f_low,
# normalized to one in ln y on 0 < y <= k (x <= 1). The shape of f_+ changes
# with k.
TEST_SIGMA = lambda kv: sT * (kv - 4) ** 2
TEST_F = {
    "+": lambda kv, yv: (1 - 1 / kv) * yv / kv + 2 / kv * (yv / kv) ** 2,
    "low": lambda kv, yv: 2 * (yv / kv) ** 2,
}
gl_m, gl_p, e1, N1, N2, Ng = sp.symbols(
    "gamma_- gamma_+ epsilon_1 N_- N_+ N_gamma", positive=True
)
LINES = {"-": (gl_m, N1), "+": (gl_p, N2)}


def readme_rate(P_, x_=None, role="r"):
    right = L.sides(
        L.part(L.display(RATES), "R(P)" if x_ is None else "G_r(P, x)")
    )[1]
    pre, rest = right.split(r"\int_", 1)
    lo, rest = rest.split("^{", 1)
    hi, body = rest.split(r"} dk\thinspace ", 1)
    tokens = {"k": k, r"\sigma_T": sT} | ({"x": x_} if x_ is not None else {})
    integrand = L.to_sympy(body, tokens, funcs={r"\sigma": SIG, "f_r": F[role]})
    return L.to_sympy(pre, {"P": P_}) * sp.Integral(
        integrand, (k, L.to_sympy(lo, {}), L.to_sympy(hi, {"P": P_}))
    )


def readme_y(xv, kv):
    return L.to_sympy(
        L.sides(L.inline(r"y = xk = p\cdot k"))[1], {"x": xv, "k": kv}
    )


def collisions(weight, gl, e):
    """c (1/2) int dmu (1 - mu) sigma(k) weight(k), per lepton gl and photon e,
    as an integral over k: from 0 to 2 gl e, then from the README's threshold,
    below which sigma = 0."""
    kv = readme_k(gl, e)
    avg = sp.Integral(
        (1 - mu) / 2 * SIG(kv) * weight(kv), (mu, -1, 1)
    ).transform(mu, (sp.solve(kv - k, mu)[0], k))
    ((var, lo, hi),) = avg.limits
    assert var == k and lo == 0 and hi == 2 * gl * e, avg.limits
    return c * sp.Integral(
        avg.function,
        (k, L.to_sympy(L.inline("4m_e c^2"), {"m_e": 1, "c": 1}), hi),
    )


def pulled(expr):
    def split(a):
        cf, rest = a.function.as_independent(k, as_Add=False)
        return cf * sp.Integral(rest, *a.limits)

    return expr.replace(lambda a: isinstance(a, sp.Integral), split)


def integrands(difference):
    d = sp.expand(pulled(difference))
    if d == 0:
        return d

    def linear(term):
        fs = sp.Mul.make_args(term)
        return sum(isinstance(f, sp.Integral) for f in fs) == 1 and all(
            isinstance(f, sp.Integral) or not f.has(sp.Integral) for f in fs
        )

    ints = d.atoms(sp.Integral)
    if (
        not all(linear(term) for term in sp.Add.make_args(d))
        or len({i.limits for i in ints}) > 1
    ):
        return None
    return d.xreplace({i: i.function for i in ints})


def same_integrands(what, a, b, must_differ=False):
    d = integrands(a - b)
    if d is None:
        return Result(
            what,
            must_differ,
            "not the same integrals over k"
            + (", as it must be" if must_differ else ""),
        )
    return (exact_fails if must_differ else exact)(what, d)


def readme_angle():
    P_ = sp.Symbol("P", positive=True)
    right = L.sides(L.parts(L.display(ANGLE))[0])[1]
    pre, rest = right.split(r"\int_", 1)
    lo, rest = rest.split("^", 1)
    hi, body = rest.split(r" d\mu\thinspace ", 1)
    integrand = L.to_sympy(
        body, {r"\mu": mu, "P": P_, r"\sigma_T": sT}, funcs={r"\sigma": SIG}
    )
    return (
        P_,
        L.to_sympy(pre, {}),
        integrand,
        L.to_sympy(lo.strip("{}"), {}),
        L.to_sympy(hi.strip("{}"), {}),
    )


@derivation("tr.angle")
def check_angle():
    """The isotropic mean of the collision rate c (1 - beta cos theta) sigma(k)
    n (beta -> 1), per c sigma_T n, is the README's angle average and its R(P).
    R(P) follows from the angle average with k = P(1 - mu) and sigma = 0 below
    k = 4."""

    def claims():
        P_, pre, integrand, lo, hi = readme_angle()
        rate = L.to_sympy(
            L.after(r"coming from the angle $\theta$ at the rate $"),
            {
                "c": c,
                r"\beta": 1,
                r"\theta": sp.acos(mu),
                "k": P_ * (1 - mu),
                "n": 1,
            },
            funcs={r"\sigma": SIG},
        )
        p_def = L.to_sympy(L.sides(L.part(L.display(ANGLE), "P ="))[1], TOKENS)
        k_mu = rhs(r"k = P(1 - \mu)", {"P": P_, r"\mu": mu})
        in_k = sp.Integral(integrand, (mu, lo, hi)).transform(
            mu, (sp.solve(k_mu - k, mu)[0], k)
        )
        ((_, k_lo, k_hi),) = in_k.limits
        threshold = L.to_sympy(L.inline("4m_e c^2"), {"m_e": 1, "c": 1})
        return [
            exact("P = gamma eps", p_def - gam * eps),
            exact(
                "README k = P(1 - mu) = its k = gamma eps "
                "(1 - cos theta) at gamma eps = P",
                k_mu - readme_k(P_, 1),
            ),
            exact(
                "the README's angle average = the mean (1/2) int dmu of "
                "its rate c (1 - mu) sigma n, over c sigma_T n",
                pre * integrand - rate / (c * sT) / 2,
            ),
            Result(
                "the mean runs over all directions, -1 <= mu <= 1",
                (lo, hi) == (-1, 1),
                f"{lo} ... {hi}",
            ),
            Result(
                "with k = P(1 - mu): k from 0 to 2P",
                (k_lo, k_hi) == (0, 2 * P_),
                f"{k_lo} ... {k_hi}",
            ),
            same_integrands(
                "with sigma = 0 below k = 4: the README's R(P) "
                "(any sigma; integrands in k)",
                pre * sp.Integral(in_k.function, (k, threshold, k_hi)),
                readme_rate(P_),
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (
                r"d\mu\thinspace (1 - \mu)\thinspace",
                r"d\mu\thinspace (1 + \mu)\thinspace",
            )
        ],
    )


def readme_lepton_rate():
    found = regexp.search(
        r"interacts at the rate \$(.+?) \\int d\\ln\\varepsilon\\ (.+?)\$ "
        r"\$(.+?)\$, with \$n_\\gamma\(\\varepsilon\)\$ the photon density per "
        r"\$\\ln\\varepsilon\$",
        L.line("A lepton therefore interacts at the rate"),
    )
    return (
        L.to_sympy(found.group(1), TOKENS)
        * L.to_sympy(found.group(2), TOKENS, funcs={r"n_\gamma": NG})
        * L.to_sympy(found.group(3), TOKENS, funcs={"R": readme_rate})
    )


@derivation("tr.rate")
def check_rate():
    """The rate c sigma_T int d ln eps n_gamma R(gamma eps) counts a lepton's
    collisions with isotropic photons."""
    counted = NG(eps) * collisions(lambda kv: 1, gam, eps)
    out = [
        same_integrands(
            "README rate = the collisions c (1 - cos theta) "
            "sigma(k) averaged over isotropic photons, per ln "
            "eps (any sigma; integrands in k)",
            readme_lepton_rate(),
            counted,
        )
    ]
    with L.patched(r"$R(\gamma\varepsilon)$, with", r"$R(\varepsilon)$, with"):
        out.append(
            same_integrands(
                "control: R(eps)",
                readme_lepton_rate(),
                counted,
                must_differ=True,
            )
        )
    return out


@derivation("tr.R")
def check_R():
    """G_r(P, x) is the isotropic average of the collision rate times f_r per
    ln x (d ln y = d ln x at fixed k). tr.rate and tr.angle check R(P)."""
    # the products of the collisions per ln x, f_r(k, y) per ln y with y = xk
    # (d ln y = d ln x at fixed k)
    per_ln_x = lambda kv: (
        F["r"](kv, readme_y(x, kv)) * x * sp.diff(sp.log(readme_y(x, kv)), x)
    )
    counted = collisions(per_ln_x, gp, eps) / (c * sT)
    dln = L.sides(L.inline(r"d\ln y = d\ln x"))
    out = [
        same_integrands(
            "README G_r(gamma' eps, x) = the average of sigma "
            "f_r(k, y), y = xk, per ln x (any sigma, f_r; "
            "integrands in k)",
            readme_rate(gp * eps, x),
            counted,
        ),
        exact(
            f"README {dln[0]} = {dln[1]} at fixed k: d ln y/d ln x = 1 "
            "for its y = xk",
            x * sp.diff(sp.log(readme_y(x, k)), x) - 1
            if dln == [r"d\ln y", r"d\ln x"]
            else 1,
        ),
    ]
    with L.patched(r"f_r(k, xk)", r"f_r(k, x)"):
        out.append(
            same_integrands(
                "control: f_r(k, x) in G_r",
                readme_rate(gp * eps, x),
                counted,
                must_differ=True,
            )
        )
    return out


def readme_kinetic(pin, measure):
    left, right = L.sides(L.parts(L.display(pin))[0])
    assert left == pin.split(" = ")[0], left
    head, body = right.split(measure)
    return head, body


def readme_pairs(G=None):
    G = G or {r: (lambda a, b, r=r: readme_rate(a, b, r)) for r in ("+", "low")}
    tokens = {r"\gamma'": gp} | TOKENS
    funcs = {
        r"n_\gamma": NG,
        "n_-": DENS["-"],
        "n_+": DENS["+"],
        "G_{+}": G["+"],
        r"G_{\rm low}": G["low"],
    }
    head, body = readme_kinetic(
        PAIRS, r"\int d\ln\gamma'\int d\ln\varepsilon\thinspace "
    )
    # the electrons: the bracket of the text, with the arguments of dot n_+'s
    # G_+
    args = [
        L.to_sympy(a, tokens)
        for a in L.split_top(
            L.between(body, r"G_{+}\negthinspace \left(", r"\right)"), ","
        )
    ]
    bracket = L.between(L.line(ELECTRONS), "exchanged ($", "$)")
    ele = L.to_sympy(
        L.between(body, "", r"\left["), tokens, funcs=funcs
    ) * L.to_sympy(
        bracket,
        {
            "n_-": DENS["-"](gp),
            "n_+": DENS["+"](gp),
            r"G_\mathrm{low}": G["low"](*args),
            "G_+": G["+"](*args),
        },
    )
    return L.to_sympy(head, tokens), L.to_sympy(body, tokens, funcs=funcs), ele


def readme_mean(q, P_):
    weight = L.to_sympy(
        L.between(L.line(MEAN), MEAN, "$, as in $R$"),
        {"k": k},
        funcs={r"\sigma": SIG},
    )
    ((_, lo, hi),) = readme_rate(P_).atoms(sp.Integral).pop().limits
    return sp.Integral(weight * q, (k, lo, hi)) / sp.Integral(
        weight, (k, lo, hi)
    )


def readme_loss_fractions():
    _, body = readme_kinetic(LOSS, r"\int d\ln\varepsilon\thinspace ")
    xs = sp.symbols("x_p x_low")
    q = L.to_sympy(
        L.between(body, r"\langle ", r"\rangle"),
        {"x_+": xs[0], r"x_{\rm low}": xs[1]},
    )
    # linear: its mean at k is that of the fractions
    assert (
        sp.Poly(q, *xs).total_degree() == 1
        and q.subs({xs[0]: 0, xs[1]: 0}) == 0
    ), q
    return {"+": q.coeff(xs[0]), "low": q.coeff(xs[1])}


def readme_loss():
    head, body = readme_kinetic(LOSS, r"\int d\ln\varepsilon\thinspace ")
    inner, arg = (
        L.between(body, r"\langle ", r"\rangle"),
        L.between(body, r"\rangle(", ")"),
    )
    q = sum(cf * X[r](k) for r, cf in readme_loss_fractions().items())
    tokens = TOKENS | {"Q": readme_mean(q, L.to_sympy(arg, TOKENS))}
    body = body.replace(r"\langle " + inner + r"\rangle(" + arg + ")", "Q")
    return (
        L.to_sympy(head, tokens),
        L.to_sympy(body, tokens, funcs={r"n_\gamma": NG, "R": readme_rate}),
    )


def readme_sink():
    funcs = {
        r"n_\gamma": NG,
        "n_-": DENS["-"],
        "n_+": DENS["+"],
        "R": readme_rate,
    }
    head, body = readme_kinetic(SINK, r"\int d\ln\gamma\thinspace ")
    return (
        L.to_sympy(head, TOKENS, funcs=funcs),
        L.to_sympy(body, TOKENS, funcs=funcs),
    )


def direct_source(q):
    """Products of species q per ln gamma, per ln gamma' and ln eps, counted.
    The target t makes the product of role PRODUCTS[t][q] at gamma = x gamma',
    with f_r(k, y) per ln y (times d ln y/d ln gamma at fixed k and gamma')."""
    xv = gam / gp
    total = 0
    for t in "-+":

        def per_ln_gamma(kv, f_=F[PRODUCTS[t][q]]):
            return (
                f_(kv, readme_y(xv, kv))
                * gam
                * sp.diff(sp.log(readme_y(xv, kv)), gam)
            )

        total += DENS[t](gp) * collisions(per_ln_gamma, gp, eps)
    return NG(eps) * total


def direct_loss():
    """-gamma_dot/gamma per ln eps, counted: a collision costs the scattered
    lepton xi gamma, with xi at k the mean of the README's
    xi = 1 - x_high = x_+ + x_low."""
    xi = L.sides(L.inline(r"\xi = 1 - x_\mathrm{high} = x_+ + x_\mathrm{low}"))[
        2
    ]
    return NG(eps) * collisions(
        lambda kv: L.to_sympy(
            xi, {"x_+": X["+"](kv), r"x_\mathrm{low}": X["low"](kv)}
        ),
        gam,
        eps,
    )


def direct_sink():
    """Photons removed per ln eps, per ln gamma, counted: every collision
    removes the photon. The isotropic leptons meet it with the same average (an
    e+ has the e-'s cross section)."""
    return (
        -NG(eps)
        * (DENS["-"](gam) + DENS["+"](gam))
        * collisions(lambda kv: 1, gam, eps)
    )


def on_lines(expr, t, gl):
    """int d ln gl n_t(gl) int d ln eps n_gamma(eps) expr (expr linear in them)
    for the line of target t and the photon line, with the test functions."""
    g_t, N_t = LINES[t]
    e_ = (
        expr.subs({gl: g_t, eps: e1})
        .replace(DENS[t], lambda a: N_t)
        .replace(DENS[CONJUGATE[t]], lambda a: 0)
    )
    e_ = e_.replace(NG, lambda a: Ng).replace(SIG, TEST_SIGMA)
    for r, f_ in TEST_F.items():
        # <x> = int dlny f x
        e_ = e_.replace(F[r], f_).replace(
            X[r], lambda a, f_=f_: sp.integrate(f_(a, y) / a, (y, 0, a))
        )
    return e_.doit()


def made(source, power=0):
    """Target t's products have gamma <= gamma_t."""
    return sum(
        sp.integrate(
            gam**power * on_lines(source, t, gp) / gam, (gam, 0, LINES[t][0])
        )
        for t in "-+"
    )


def interactions():
    return sum(
        on_lines(
            (DENS["-"](gam) + DENS["+"](gam)) * readme_lepton_rate(), t, gam
        )
        for t in "-+"
    )


@derivation("tr.pairs")
def check_pairs():
    """The pair sources count the collisions' products (charge conjugates for e+
    targets), one e+ and one e- each."""
    head, pos, ele = readme_pairs()
    swap = {DENS["-"](gp): DENS["+"](gp), DENS["+"](gp): DENS["-"](gp)}
    _, body = readme_kinetic(
        PAIRS, r"\int d\ln\gamma'\int d\ln\varepsilon\thinspace "
    )
    arg_x = L.to_sympy(
        L.split_top(
            L.between(body, r"G_{+}\negthinspace \left(", r"\right)"), ","
        )[1],
        {r"\gamma'": gp} | TOKENS,
    )
    x_text = rhs(r"x = \gamma/\gamma'", {r"\gamma'": gp, r"\gamma": gam})
    out = [
        same_integrands(
            "README dot n_+ = the positrons of the collisions "
            "(any sigma, f_r; per ln gamma' and ln eps, "
            "integrands in k)",
            head * pos,
            direct_source("+"),
        ),
        same_integrands(
            "README electrons (n_- G_low + n_+ G_+) = the "
            "electrons of the collisions",
            head * ele,
            direct_source("-"),
        ),
        exact(
            "the electrons' bracket = dot n_+'s with n_- and n_+ exchanged",
            pos.xreplace(swap) - ele,
        ),
        exact(
            "the text's x = gamma/gamma' = the second argument of G_+ "
            "(the count's x)",
            (x_text - arg_x) ** 2 + (x_text - gam / gp) ** 2,
        ),
    ]
    assert all(
        sp.simplify(sp.integrate(f_(k, y) / y, (y, 0, k)) - 1) == 0
        for f_ in TEST_F.values()
    )
    n_int = interactions()
    out += [
        exact(
            "lines of e-, e+ and photons, test sigma and f_r: positrons "
            "made = interactions (README's rate of a lepton)",
            made(head * pos) - n_int,
        ),
        exact("electrons made = interactions", made(head * ele) - n_int),
    ]
    with L.patched(
        r"n_+(\gamma')\thinspace G_{\rm low}", r"n_+(\gamma')\thinspace G_{+}"
    ):
        h_b, pos_b, _ = readme_pairs()
    out.append(
        same_integrands(
            "control: G_+ for the positrons of e+ targets",
            h_b * pos_b,
            direct_source("+"),
            must_differ=True,
        )
    )
    with L.patched(
        r"G_{+}\negthinspace \left(\gamma'\varepsilon,\ "
        r"\frac{\gamma}{\gamma'}\right)",
        r"G_{+}\negthinspace \left(\gamma'\varepsilon,\ "
        r"\frac{\gamma'}{\gamma}\right)",
    ):
        h_b, pos_b, _ = readme_pairs()
    out.append(
        same_integrands(
            "control: x = gamma'/gamma in G_+",
            h_b * pos_b,
            direct_source("+"),
            must_differ=True,
        )
    )
    return out


@derivation("tr.loss")
def check_loss():
    """The continuous loss is the mean energy the collisions cost the scattered
    lepton, which the pairs take."""
    head, loss = readme_loss()
    out = [
        same_integrands(
            "README -gamma_dot/gamma = the mean of xi gamma over the "
            "collisions, per gamma "
            "(any sigma, <x_r>; per ln eps, integrands in k)",
            head * loss,
            direct_loss(),
        )
    ]
    h_p, pos, ele = readme_pairs()
    gained = made(h_p * (pos + ele), power=1)

    def lost(h, l_):
        return sum(
            on_lines((DENS["-"](gam) + DENS["+"](gam)) * gam * h * l_, t, gam)
            for t in "-+"
        )

    out.append(
        exact(
            "lines, test sigma and f_r: the energy the leptons lose = "
            "the energy of the pairs made",
            lost(head, loss) - gained,
        )
    )
    with L.patched(
        r"the weight $k\sigma(k)$, as in $R$",
        r"the weight $\sigma(k)$, as in $R$",
    ):
        h_b, loss_b = readme_loss()
    out += [
        same_integrands(
            "control: the weight sigma(k) in <...>",
            h_b * loss_b,
            direct_loss(),
            must_differ=True,
        ),
        exact_fails(
            "control: the weight sigma(k): energy lost = gained",
            lost(h_b, loss_b) - gained,
        ),
    ]
    return out


# ----------------------- the loss against inverse Compton, and its spread
@derivation("tr.ic_kn")
def check_ic_kn():
    """The inverse-Compton loss in its Klein-Nishina limit is
    Blumenthal & Gould's eq. 2.57 for photons of one energy (E = gamma m c^2,
    eps in m c^2). The README compares the continuous loss with this limit and
    uses pi r_e^2 = (3/8) sigma_T."""
    m_, n_, eB = sp.symbols("m n epsilon_B", positive=True)

    def claims():
        line = L.line(IC_LINE)
        tokens = {
            "c": c,
            r"\sigma_T": sT,
            "n": n_,
            r"\varepsilon": eps,
            r"\gamma": gam,
        }
        ours = regexp.search(
            r"its Klein–Nishina limit, \$-\\dot\\gamma\$ "
            r"\$= ([^$]+)\$ \$([^$]+)\$, for",
            line,
        )
        loss = L.to_sympy(ours.group(1), tokens) * L.to_sympy(
            ours.group(2), tokens
        )
        bg = regexp.search(
            r"\$-dE/d\{t\}\$ "
            r"\$= (.+?) \\int d\\varepsilon\\ n\(\\varepsilon\)\$ "
            r"\$\\varepsilon\^\{-1\}\$ \$(.+?)\$, with \$r_0\$",
            line,
        )
        bg_tokens = {
            "r_0": re,
            "m": m_,
            "c": c,
            r"\varepsilon": eB,
            r"\gamma": gam,
        }
        # n(eps) = n delta(eps - eps_B)
        bg_mono = (
            L.to_sympy(bg.group(1), bg_tokens)
            * n_
            / eB
            * L.to_sympy(bg.group(2), bg_tokens)
        )
        sT_r = relation(
            r"\pi r_e^2 = (3/8)\sigma_T", {"r_e": re, r"\sigma_T": sT}
        )
        left, right = L.sides(BG_257)
        pre, integrand = right.split(r"\int ", 1)
        bg_t = L.to_sympy(pre, bg_tokens) * L.to_sympy(
            integrand.replace(r" d\epsilon", ""),
            {r"\epsilon": eB, r"\gamma": gam, "m": m_, "c": c},
            funcs={"n": lambda a: n_},
        )
        sigma_T = 8 * sp.pi * re**2 / 3
        per_m = lambda e: (e / (m_ * c**2)).subs(eB, eps * m_ * c**2)
        return [
            exact(
                "README's quote of eq. 2.57 = the transcription (p. 244), "
                "photons of one energy",
                bg_mono - bg_t,
            ),
            exact(
                "README pi r_e^2 = (3/8) sigma_T with sigma_T = 8 pi r_e^2/3",
                (sT_r[0] - sT_r[1]).subs(sT, sigma_T),
            ),
            exact(
                "README -gamma_dot = eq. 2.57 / (m c^2), eps in m c^2, "
                "r_0 = r_e, sigma_T = 8 pi r_e^2/3",
                (loss - per_m(bg_mono)).subs(sT, sigma_T),
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (
                r"$= (3/8) c\sigma_T n\varepsilon^{-1}$",
                r"$= (1/8) c\sigma_T n\varepsilon^{-1}$",
            )
        ],
    )


@derivation("tr.xi_bin")
def check_xi_bin():
    """A lepton that loses more than a grid bin keeps less than e^-Delta of its
    energy: xi > 1 - e^-Delta, which is the 0.095 of the README for its
    Delta."""
    D = readme_delta()

    def claims():
        keeps = L.to_sympy(L.after("keeps less than $"), {r"\Delta": DELTA})
        left, right = (
            p_.strip() for p_ in L.after("of its energy, so $").split(r"\gt")
        )
        bound = L.to_sympy(right, {r"\Delta": DELTA})
        printed = regexp.search(
            r"loses more than a grid bin "
            r"\(\$\\xi \\gt ([\d.]+)\$\)",
            L.line("Interactions where the lepton loses more than a grid bin"),
        ).group(1)
        value = float(bound.subs(DELTA, D))
        return [
            Result("xi > ...", left == r"\xi", left),
            exact("1 - xi < e^-Delta: xi > 1 - e^-Delta", bound - (1 - keeps)),
            Result(
                f"1 - e^-Delta at the README's Delta = {D} is "
                f"{value:.5f}: the README's {printed}",
                rounds_to(value, printed),
                f"{value:.6f}",
            ),
        ]

    return claims() + _controls(
        claims,
        [(r"so $\xi \gt 1 - e^{-\Delta}$", r"so $\xi \gt 1 - e^{-2\Delta}$")],
    )


@derivation("tr.sink")
def check_sink():
    """The photon sink: every collision removes its photon, that is, one photon
    per interaction."""
    head, body = readme_sink()
    out = [
        same_integrands(
            "README dot n_gamma = the photons the "
            "collisions remove (any sigma; per ln gamma, "
            "integrands in k)",
            head * body,
            direct_sink(),
        )
    ]

    def absorbed(h, b):
        return -sum(on_lines(h * b, t, gam) for t in "-+")

    out.append(
        exact(
            "lines, test sigma: photons absorbed = interactions "
            "(README's rate of a lepton)",
            absorbed(head, body) - interactions(),
        )
    )
    with L.patched(r"\bigl(n_-(\gamma) + n_+(\gamma)\bigr)", r"n_-(\gamma)"):
        h_b, b_b = readme_sink()
    out += [
        same_integrands(
            "control: n_- alone in the sink",
            h_b * b_b,
            direct_sink(),
            must_differ=True,
        ),
        exact_fails(
            "control: n_- alone: photons absorbed = interactions",
            absorbed(h_b, b_b) - interactions(),
        ),
    ]
    return out


# ----------------------- on the grid
DELTA = sp.Symbol("Delta", positive=True)
GI, GJ, GJ0, GM0, GQ, GO, GL_ = sp.symbols("i j j_0 m_0 q o l", integer=True)


def floor_part(tex, tokens):
    """to_sympy has no floor."""
    found = regexp.search(r"(\\left)?\\lfloor (.+?)(\\right)?\\rfloor", tex)
    inner = L.to_sympy(found.group(2), tokens)
    return L.to_sympy(
        tex[: found.start()] + " F " + tex[found.end() :],
        tokens | {"F": sp.floor(inner)},
    )


def readme_grid(D=DELTA):
    tex = L.display(GRID)
    P_q = L.to_sympy(
        L.sides(L.part(tex, "P_q ="))[1], {"m_0": GM0, "q": GQ, r"\Delta": D}
    )
    m_0 = floor_part(L.sides(L.part(tex, "m_0 ="))[1], {r"\Delta": D})
    product, q_def = L.part(tex, r"\gamma_i\varepsilon_j").split(
        r"\quad \text{for}\quad "
    )
    q_ij = L.to_sympy(
        L.sides(q_def)[1], {"i": GI, "j": GJ, "j_0": GJ0, "m_0": GM0}
    )
    return dict(
        gam=rhs(r"\gamma_i = e^{i\Delta}", {"i": GI, r"\Delta": D}),
        eps=rhs(
            r"\varepsilon_j = e^{(j - j_0)\Delta}",
            {"j": GJ, "j_0": GJ0, r"\Delta": D},
        ),
        P=P_q,
        m0=m_0,
        q=q_ij,
        product=L.sides(product),
        q_name=L.sides(q_def)[0],
        x=rhs(r"x_o = e^{-o\Delta}", {"o": GO, r"\Delta": D}),
    )


@derivation("tr.grid")
def check_grid():
    """On AM3's grids, a lepton of bin i and a photon of bin j have
    gamma_i eps_j = P_q with q = i + j - j_0 - m_0. P_0 = e^(m_0 Delta) is the
    first grid value above the threshold P = 2 (k = 2P = 4 for head-on
    photons)."""
    r = rng(5)
    steps = [readme_delta()] + [
        sp.Rational(r.randint(1, 999), 1000) for _ in range(30)
    ]

    def claims():
        g = readme_grid()
        first_above = []
        for D in steps:
            m0 = int(readme_grid(D)["m0"])
            with mp.workdps(50):
                first_above.append(
                    mp.exp((m0 - 1) * mp.mpf(D.p) / D.q)
                    <= 2
                    < mp.exp(m0 * mp.mpf(D.p) / D.q)
                )
        found = regexp.search(
            r"above the threshold \$P = (\d+)\$, where head-on photons have "
            r"\$k = 2P = (\d+)\$",
            L.line("is the first grid value above the threshold"),
        )
        P_t, k_t = sp.Integer(found.group(1)), sp.Integer(found.group(2))
        k_head_on = L.to_sympy(
            L.sides(L.inline(r"k = \gamma\varepsilon(1 - \cos\theta)"))[1],
            {r"\gamma": P_t, r"\varepsilon": 1, r"\theta": sp.pi},
        )
        return [
            Result(
                "notation: gamma_i eps_j = P_q for q = ...",
                g["product"] == [r"\gamma_i\varepsilon_j", "P_q"]
                and g["q_name"] == "q",
                f"{g['product']}, {g['q_name']}",
            ),
            exact(
                "gamma_i eps_j = P_q at the README's q (any i, j, j_0, "
                "m_0, Delta)",
                g["gam"] * g["eps"] - g["P"].subs(GQ, g["q"]),
            ),
            exact(
                "eps_{j_0} = 1",
                rhs(r"\varepsilon_{j_0} = 1", {}) - g["eps"].subs(GJ, GJ0),
            ),
            Result(
                "m_0 makes P_0 the first grid value above P = 2 "
                f"(the README's Delta and {len(steps) - 1} others)",
                all(first_above),
                f"{sum(first_above)} of {len(steps)}",
            ),
            exact(
                "head-on photons: k = 2P = 4 at the threshold P = 2, "
                "the README's k = 4",
                (k_head_on - k_t) ** 2
                + (2 * P_t - k_t) ** 2
                + (k_t - threshold_k()) ** 2,
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (r"\right\rfloor + 1", r"\right\rfloor - 1"),
            (r"q = i + j - j_0 - m_0", r"q = i + j - j_0 + m_0"),
        ],
    )


def readme_row_rate(P_):
    pre, rest = display_rhs(r"R_q = \frac{1}{2P_q^2}").split(r"\int_", 1)
    lo, rest = rest.split("^", 1)
    hi, body = rest.split(r" d\ln k\thinspace ", 1)
    lam = sp.Symbol("lambda", real=True)
    integrand = L.to_sympy(
        body, {"k": sp.exp(lam), r"\sigma_T": sT}, funcs={r"\sigma": SIG}
    )
    limits = [L.to_sympy(b.strip("{}"), {"P_q": P_}) for b in (lo, hi)]
    return L.to_sympy(pre, {"P_q": P_}) * sp.Integral(
        integrand, (lam, *limits)
    ).transform(lam, (sp.log(k), k))


def readme_share(xv, D=DELTA, o=None):
    g = readme_grid(D)
    if o is None:
        o = floor_part(
            L.after("next to $x$, with $").split("=", 1)[1],
            {"x": xv, r"\Delta": D},
        )
    xo, xo1 = g["x"].subs(GO, o), g["x"].subs(GO, o + 1)
    w = L.to_sympy(
        display_rhs(r"w = \frac{"), {"x_o": xo, "x": xv, "x_{o+1}": xo1}
    )
    return o, xo, xo1, w


@derivation("tr.share")
def check_share():
    """The code shares a product at x = y/k between the grid fractions
    x_o >= x > x_{o+1}, and this sharing keeps its number and energy. The share
    w goes to x_{o+1}, and 1 - w goes to x_o."""
    xs, xo, xo1 = sp.symbols("x x_o x_{o+1}", positive=True)
    D = readme_delta()
    r = rng(7)

    def claims():
        w = L.to_sympy(
            display_rhs(r"w = \frac{"), {"x_o": xo, "x": xs, "x_{o+1}": xo1}
        )
        number = relation(r"w + (1 - w) = 1", {"w": w})
        energy = relation(
            r"(1 - w)\thinspace x_o + w\thinspace x_{o+1} = x",
            {"w": w, "x_o": xo, "x_{o+1}": xo1, "x": xs},
        )
        x_of = L.to_sympy(
            L.after("the products of the table bin $y$ have $").split("=")[1],
            {"y": y, "k": k},
        )
        inside = []
        for _ in range(40):
            xv = sp.Rational(r.randint(1, 10**6), 10**6)
            o, a, b, wv = readme_share(xv, D)
            inside.append(
                bool(b < xv)
                and bool(xv <= a)
                and bool(wv >= 0)
                and bool(wv < 1)
            )
        return [
            exact("number kept: w + (1 - w) = 1", number[0] - number[1]),
            exact(
                "energy kept: (1 - w) x_o + w x_{o+1} = x",
                energy[0] - energy[1],
            ),
            exact("x = y/k: the README's y = xk", readme_y(x_of, k) - y),
            Result(
                "the README's o puts x between x_{o+1} and x_o, "
                f"and 0 <= w < 1 (40 random x, Delta = {D})",
                all(inside),
                f"{sum(inside)} of {len(inside)}",
            ),
        ]

    return claims() + _controls(
        claims,
        [
            (
                r"w = \frac{x_o - x}{x_o - x_{o+1}}",
                r"w = \frac{x - x_{o+1}}{x_o - x_{o+1}}",
            ),
            (
                r"$o = \lfloor -(\ln x)/\Delta\rfloor$",
                r"$o = \lfloor -(\ln x)/\Delta\rfloor + 1$",
            ),
        ],
    )


@derivation("tr.rows")
def check_rows():
    """The kernel rows: R_q is R(P_q) written in ln k, and L_q, the energy of
    the deposits, is that of the products. The sharing keeps this energy. In
    the continuum, L_q is R(P) <x_+ + x_low>(P)."""
    P_ = sp.Symbol("P", positive=True)
    D = DELTA
    g = readme_grid(D)

    def summand_L():
        sides = L.sides(L.parts(L.display(r"L_q = \sum_o"))[0])
        summand = sides[1].split(r"\sum_o ", 1)[1]
        return (
            sides,
            L.to_sympy(
                summand,
                {
                    "W_{+,q,o}": sp.Symbol("Wp"),
                    r"W_{\mathrm{low},q,o}": sp.Symbol("Wl"),
                    "x_o": sp.Symbol("xo"),
                    "x_{o+1}": sp.Symbol("xo1"),
                },
            ),
        )

    def claims():
        sides, summand = summand_L()
        amounts = sp.symbols("a0:4", positive=True)
        thetas = sp.symbols("theta0:4", positive=True)
        W = {"+": [0] * 6, "low": [0] * 6}
        energy = 0
        for m, (role, o) in enumerate(
            (("+", 0), ("+", 2), ("low", 1), ("low", 2))
        ):
            xm = (
                g["x"].subs(GO, o) * (1 - thetas[m])
                + g["x"].subs(GO, o + 1) * thetas[m]
            )  # x_o >= x > x_{o+1}
            w = readme_share(xm, D, o)[3]
            W[role][o] += amounts[m] * (1 - w)
            W[role][o + 1] += amounts[m] * w
            energy += amounts[m] * xm
        L_q = sum(
            summand.subs(
                {
                    sp.Symbol("Wp"): W["+"][o],
                    sp.Symbol("Wl"): W["low"][o],
                    sp.Symbol("xo"): g["x"].subs(GO, o),
                    sp.Symbol("xo1"): g["x"].subs(GO, o + 1),
                }
            )
            for o in range(6)
        )
        frac = readme_loss_fractions()
        gained = 0
        for role, cf in frac.items():
            G = (
                readme_rate(P_, x, role)
                .replace(SIG, TEST_SIGMA)
                .replace(F[role], TEST_F[role])
                .doit()
            )
            gained += cf * sp.integrate(G, (x, 0, 1))
        Xs = {
            role: sp.integrate(TEST_F[role](k, y) / k, (y, 0, k))
            for role in frac
        }
        mean = (
            readme_mean(sum(cf * Xs[role] for role, cf in frac.items()), P_)
            .replace(SIG, TEST_SIGMA)
            .doit()
        )
        R_P = readme_rate(P_).replace(SIG, TEST_SIGMA).doit()
        return [
            same_integrands(
                "README R_q = R(P_q), its integral over ln k "
                "(any sigma; integrands in k)",
                readme_row_rate(P_),
                readme_rate(P_),
            ),
            exact(
                "README L_q = the energy of the products deposited "
                "(shared by the README's w, any amounts and x)",
                L_q - energy,
            ),
            Result(
                "README L_q's last side: R_q <x_+ + x_low>(P_q), "
                "the loss term's mean",
                sides[2]
                == (
                    r"R_q\thinspace \langle "
                    + L.between(
                        readme_kinetic(
                            LOSS, r"\int d\ln\varepsilon\thinspace "
                        )[1],
                        r"\langle ",
                        r"\rangle",
                    )
                    + r"\rangle(P_q)"
                ),
                sides[2],
            ),
            exact(
                "continuum: the energy of the products, sum_r int dx "
                "G_r(P, x) = R(P) <x_+ + x_low>(P) (test sigma, f_r)",
                sp.simplify(gained - R_P * mean),
            ),
        ]

    return claims() + _controls(
        claims,
        [(r"\bigr)\thinspace x_o = R_q", r"\bigr)\thinspace x_{o+1} = R_q")],
    )


def readme_step():
    z_part, loss_part, sink_part = L.parts(
        L.display(
            r"\dot n_\gamma(\varepsilon_j) = "
            r"-\thinspace n_\gamma(\varepsilon_j)"
        )
    )
    out = {}
    for name, part, index in (
        ("Z", z_part, "j"),
        ("loss", loss_part, "j"),
        ("sink", sink_part, "i"),
    ):
        lhs, right = L.sides(part)
        factor, summand = right.split(r"\sum_" + index + " ", 1)
        out[name] = (lhs, factor, summand)
    lhs, right = L.sides(L.parts(L.display(SOURCE))[0])
    out["source"] = (lhs, "", right.split(r"\sum_i ", 1)[1])
    return out


def grid_terms(W, R, Lq, n_m, n_p, n_g, D, j0, m0, n_l, n_o, below=True):
    """below: if True, the products below the grid go into bin 0 (as the README
    says). If False, they are left out."""
    st = readme_step()
    g = readme_grid(D)
    sym = {
        nm: sp.Symbol(nm)
        for nm in ("NPH", "W", "Lq", "NM", "NP", "Rq", "Zp", "Zl")
    }
    nph_tok, nm_tok, np_tok = (
        r"n_\gamma(\varepsilon_j)",
        r"n_-(\gamma_i)",
        r"n_+(\gamma_i)",
    )
    base = {"c": c, r"\sigma_T": sT, r"\Delta": D}
    zf = L.to_sympy(st["Z"][1], base)
    zs = L.to_sympy(st["Z"][2], {nph_tok: sym["NPH"], "W_{r,q,o}": sym["W"]})
    lf = L.to_sympy(st["loss"][1], base)
    ls = L.to_sympy(st["loss"][2], {nph_tok: sym["NPH"], "L_q": sym["Lq"]})
    sf = L.to_sympy(st["sink"][1], base | {nph_tok: sym["NPH"]})
    ss = L.to_sympy(
        st["sink"][2], {nm_tok: sym["NM"], np_tok: sym["NP"], "R_q": sym["Rq"]}
    )
    offsets = {}
    for role, tok in (("+", "+"), ("low", r"\mathrm{low}")):
        found = regexp.search(
            r"Z_\{" + regexp.escape(tok) + r",i,([^}]+)\}", st["source"][2]
        )
        offsets[role] = L.to_sympy(found.group(1), {"i": GI, "l": GL_})
    src = L.to_sympy(
        regexp.sub(
            r"Z_\{\+,i,[^}]+\}",
            "Zp",
            regexp.sub(r"Z_\{\\mathrm\{low\},i,[^}]+\}", "Zl", st["source"][2]),
        ),
        {
            nm_tok: sym["NM"],
            np_tok: sym["NP"],
            "Zp": sym["Zp"],
            "Zl": sym["Zl"],
        },
    )
    q_of = lambda i, j: int(g["q"].subs({GI: i, GJ: j, GJ0: j0, GM0: m0}))
    row = lambda table, q: table[q] if q >= 0 else 0
    Z = {
        r: [
            [
                zf
                * sum(
                    zs.subs(
                        {
                            sym["NPH"]: n_g[j],
                            sym["W"]: row(W[r], q_of(i, j))[o]
                            if q_of(i, j) >= 0
                            else 0,
                        }
                    )
                    for j in range(len(n_g))
                )
                for o in range(n_o)
            ]
            for i in range(len(n_m))
        ]
        for r in W
    }
    loss = [
        lf
        * sum(
            ls.subs({sym["NPH"]: n_g[j], sym["Lq"]: row(Lq, q_of(i, j))})
            for j in range(len(n_g))
        )
        for i in range(len(n_m))
    ]
    sink = [
        sf.subs(sym["NPH"], n_g[j])
        * sum(
            ss.subs(
                {
                    sym["NM"]: n_m[i],
                    sym["NP"]: n_p[i],
                    sym["Rq"]: row(R, q_of(i, j)),
                }
            )
            for i in range(len(n_m))
        )
        for j in range(len(n_g))
    ]

    def source(nm, np_):
        out = [0] * n_l
        for i in range(len(nm)):
            for o in range(n_o):
                l_ = i - o
                assert all(
                    int(off.subs({GI: i, GL_: l_})) == o
                    for off in offsets.values()
                ), offsets
                if l_ < 0 and not below:
                    continue
                term = src.subs(
                    {
                        sym["NM"]: nm[i],
                        sym["NP"]: np_[i],
                        sym["Zp"]: Z["+"][i][o],
                        sym["Zl"]: Z["low"][i][o],
                    }
                )
                out[max(l_, 0)] += term  # below the grid: bin 0
        return out

    return source(n_m, n_p), source(n_p, n_m), loss, sink, offsets


@derivation("tr.step")
def check_step():
    """The per-step sums of the README, with W = Delta G_r(P_q, x_o), are the
    integrals of the kinetic terms as sums over the bins. For any rows with each
    product once per interaction, they keep the number
    (positrons made = photons absorbed). With L_q, they also keep the energy
    (pairs' energy = the leptons' loss)."""
    D = DELTA
    g = readme_grid(D)
    n_l, n_g_bins, n_o, j0, m0 = 4, 3, 3, 2, 1
    gam_i = [g["gam"].subs(GI, i) for i in range(n_l)]
    eps_j = [g["eps"].subs({GJ: j, GJ0: j0}) for j in range(n_g_bins)]
    q_of = lambda i, j: int(g["q"].subs({GI: i, GJ: j, GJ0: j0, GM0: m0}))
    n_q = max(q_of(n_l - 1, n_g_bins - 1), 0) + 1
    GP, GLw = sp.Function("G_p"), sp.Function("G_low")

    def riemann():
        w_cont = L.inline(r"\Delta\thinspace G_r(P_q, x_o)")
        W = {
            r: [
                [
                    L.to_sympy(
                        w_cont,
                        {
                            r"\Delta": D,
                            "P_q": g["P"].subs({GQ: q, GM0: m0}),
                            "x_o": g["x"].subs(GO, o),
                        },
                        funcs={"G_r": G},
                    )
                    for o in range(n_o)
                ]
                for q in range(n_q)
            ]
            for r, G in (("+", GP), ("low", GLw))
        }
        n_m = [DENS["-"](gi) for gi in gam_i]
        n_p = [DENS["+"](gi) for gi in gam_i]
        n_g = [NG(ej) for ej in eps_j]
        pos = grid_terms(
            W,
            [0] * n_q,
            [0] * n_q,
            n_m,
            n_p,
            n_g,
            D,
            j0,
            m0,
            n_l,
            n_o,
            below=False,
        )[0]
        head, body, _ = readme_pairs({"+": GP, "low": GLw})
        cont = [0] * n_l
        for l_ in range(n_l):
            for i in range(l_, min(l_ + n_o, n_l)):
                for j in range(n_g_bins):
                    if q_of(i, j) >= 0:
                        cont[l_] += (
                            D**2
                            * head
                            * body.subs(
                                {gp: gam_i[i], eps: eps_j[j], gam: gam_i[l_]}
                            )
                        )
        return exact(
            "positron source: the README's sums = its integrals as "
            "sums over the bins (any n, G_r)",
            sum((sp.expand(a - b)) ** 2 for a, b in zip(pos, cont)),
        )

    def balances():
        W = {
            r: [list(sp.symbols(f"W{r[0]}{q}_0:{n_o}")) for q in range(n_q)]
            for r in ("+", "low")
        }
        # one positron per interaction
        R = [sum(W["+"][q]) for q in range(n_q)]
        for q in range(n_q):
            # and one electron of lower energy
            W["low"][q][0] = R[q] - sum(W["low"][q][1:])
        summand = L.to_sympy(
            L.sides(L.parts(L.display(r"L_q = \sum_o"))[0])[1].split(
                r"\sum_o ", 1
            )[1],
            {
                "W_{+,q,o}": sp.Symbol("Wp"),
                r"W_{\mathrm{low},q,o}": sp.Symbol("Wl"),
                "x_o": sp.Symbol("xo"),
            },
        )
        Lq = [
            sum(
                summand.subs(
                    {
                        sp.Symbol("Wp"): W["+"][q][o],
                        sp.Symbol("Wl"): W["low"][q][o],
                        sp.Symbol("xo"): g["x"].subs(GO, o),
                    }
                )
                for o in range(n_o)
            )
            for q in range(n_q)
        ]
        n_m, n_p, n_g = (
            list(sp.symbols(f"{nm}0:{n}", positive=True))
            for nm, n in (("nm", n_l), ("np", n_l), ("ng", n_g_bins))
        )
        pos, ele, loss, sink, _ = grid_terms(
            W, R, Lq, n_m, n_p, n_g, D, j0, m0, n_l, n_o
        )
        made_p, made_e = D * sum(pos), D * sum(ele)
        absorbed = -D * sum(sink)
        # energy: leptons only where all their products land on the grid (bins
        # i >= n_o - 1)
        on = {s_: 0 for s_ in n_m[: n_o - 1] + n_p[: n_o - 1]}
        pos_e, ele_e, loss_e, _, _ = grid_terms(
            W,
            R,
            Lq,
            [v if v not in on else 0 for v in n_m],
            [v if v not in on else 0 for v in n_p],
            n_g,
            D,
            j0,
            m0,
            n_l,
            n_o,
        )
        pairs_energy = D * sum(
            gam_i[l_] * (pos_e[l_] + ele_e[l_]) for l_ in range(n_l)
        )
        lost = D * sum(
            gam_i[i]
            * (
                (n_m[i] if n_m[i] not in on else 0)
                + (n_p[i] if n_p[i] not in on else 0)
            )
            * loss_e[i]
            for i in range(n_l)
        )
        return [
            exact(
                "number: positrons made = photons absorbed "
                "(any n, rows; products below the grid in bin 0)",
                sp.expand(made_p - absorbed),
            ),
            exact(
                "number: electrons made (n_- and n_+ exchanged) = "
                "photons absorbed",
                sp.expand(made_e - absorbed),
            ),
            exact(
                "energy: the pairs' energy = the leptons' loss, sum gamma_i "
                "n_i (-gamma_dot_i/gamma_i) (all on the grid)",
                sp.expand(pairs_energy - lost),
            ),
        ]

    def claims():
        st = readme_step()
        o_text = rhs(r"o = i - l", {"i": GI, "l": GL_})
        return [
            Result(
                "the left sides: Z_{r,i,o}, -gamma_dot_i/gamma_i, "
                "dot n_gamma(eps_j), dot n_+(gamma_l)",
                [st[nm][0] for nm in ("Z", "loss", "sink", "source")]
                == [
                    "Z_{r,i,o}",
                    r"-\frac{\dot\gamma_i}{\gamma_i}",
                    r"\dot n_\gamma(\varepsilon_j)",
                    r"\dot n_+(\gamma_l)",
                ],
                str([st[nm][0] for nm in ("Z", "loss", "sink", "source")]),
            ),
            exact("the text's offset o = i - l", o_text - (GI - GL_)),
            riemann(),
        ] + balances()

    return claims() + _controls(
        claims,
        [
            (
                r"Z_{r,i,o} = c\thinspace \sigma_T\Delta\sum_j",
                r"Z_{r,i,o} = c\thinspace \sigma_T\sum_j",
            ),
            (
                r"\thinspace Z_{+,i,i-l} + n_+(\gamma_i)\thinspace "
                r"Z_{\mathrm{low},i,i-l}",
                r"\thinspace Z_{\mathrm{low},i,i-l} + "
                r"n_+(\gamma_i)\thinspace Z_{+,i,i-l}",
            ),
        ],
    )
