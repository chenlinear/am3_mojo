"""Derivation checks of the production and evolution of the charm hadrons,
tests/literature_checks/charm/README.md (page "charm"): production in
photo-hadronic interactions (ids cp.*), production in pp collisions (cpp.*) and
the evolution (ch.*). Sources (table, figure and equation numbers of these
versions):
  - Hümmer, Rüger, Spanier & Winter (2010), arXiv:1002.1310v3: eqs. 4, 5, 8, 27
    and 28, Table 6 (the multi-pion types).
  - The measured values of tests/support/measured/charm_photoproduction.py (its
    docstrings give each source). Those of HEPData records are compared with the
    local copies in knowledge/HEPData/. Those of ZEUS (Derrick et al. 1995,
    arXiv:hep-ex/9502002v1) and H1 (Aid et al. 1996, arXiv:hep-ex/9604005v1)
    are compared with their TeX sources.
  - The HPR1R2 fit of sigma_tot(gamma p): the PDG 2016 review "Plots of cross
    sections and related quantities" (Sec. 51: its form on p. 8, its table on
    p. 9) and the curve that the PDG 2026 web page serves (digitized). The
    review is read from its PDF in knowledge/papers/ (the text of the content
    streams). The TeX fonts of the review write "." as ":", "±" as "§" and "-"
    as "¡". The log of the review is the natural log.
  - Goncalves, Machado & Meneses (2009), arXiv:0905.2067v1, Fig. 1 (digitized,
    three dipole models).
  - Lisovyi, Verbytskyi & Zenaiev (2016), arXiv:1509.01061v4, Table 12,
    "Constrained S" (the source file Tables/average_tab2.txt, the twelfth table
    that the text places).
  - Abe et al. (SLAC HFP, 1986), PRD 33, 1 (transcribed: no TeX).
  - Link et al. (FOCUS, 2004), arXiv:hep-ex/0311022v1 (abstract and data
    selection).
  - Frixione, Mangano, Nason & Ridolfi (1995), arXiv:hep-ph/9412348v1,
    Tables 1-2.
  - Anjos et al. (E691, 1989), HEPData ins289318, Tables 9-11.
  - pp: tests/support/measured/charm_pp.py, digitized and read from the arXiv
    sources of these papers:
    - Bhattacharya et al. (2015), arXiv:1502.01076v2, Fig. 2.
    - Bhattacharya et al. (2016), arXiv:1607.00193v2, Table 1 and Fig. 12.
    - Carpio et al. (2020), arXiv:2007.07945v2, Fig. 7 and Table I.
    - Kniehl & Kramer (2006), arXiv:hep-ph/0607306v1, eq. (1) and Table I (LO).
  - AM3's two-bin split of a shifted source, number_split of its math.cc (the
    C++ original, set up by tests/reference/setup_reference.sh in
    tests/reference/work/am3/ and git-ignored).
  - The PDG 2026 listings of D+-, D0, Ds+- and Lambda_c+ (masses and mean
    lives), read from their PDFs in knowledge/papers/.
Masses: those of p, n, pi, K and the light hyperons are the code's
(src/am3_mojo/utils/constants.mojo, which cites PDG 2026). The masses of the
charm hadrons are those of the page's table under "Evolution".

The page's numbers are recomputed from its formulas, read from the page, and
from these data. These numbers are the fit of sigma_cc and its continuation, the
steps' R, h_a, h_N and chi, and the refit of the pp closed form and what follows
from it. The numerics are deterministic.

Run from the repository root:
  .venv/bin/python tests/literature_checks/common/run.py charm_production
"""

import math, os, re, warnings, zlib
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import least_squares

import latex
import mojo_constants
from latex import HUEMMER
from verify import (
    PDG_C,
    Entry,
    Result,
    absent,
    derivation,
    exact,
    exact_fails,
    in_source,
    must_fail,
    rounds_to,
    split_number,
    worst_of,
)

L = latex.page("charm")
C = dict(file="charm")

# this module's equations in the inventory (common/inventory.py)
INVENTORY = [
    Entry(
        "cp.response",
        r"f_r(y) = \frac{1}{2y^2}\int_0^{2y}",
        "derived",
        "Hümmer et al. 2010, eq. 27 (y of eq. 8) for a constant cross "
        "section on the step",
        "TeX " + HUEMMER,
        **C,
    ),
    Entry(
        "cp.source",
        r"t_r^{-1}(E_N) = c\int dy\thinspace",
        "rearranged",
        "Hümmer et al. 2010, eq. 4 (the rate) and eq. 28 (the source), "
        "summed over steps, parts and nucleons",
        "TeX " + HUEMMER,
        **C,
    ),
    Entry(
        "cp.grid",
        r"E\thinspace Q_h(E) = \sum_r \sum_k",
        "derived",
        "the source per unit ln E; the two-bin split of number_split, and "
        "AM3's (math.cc, number_split)",
        "C++ tests/reference/work/am3/",
        **C,
    ),
    Entry(
        "cp.R",
        r"R(\epsilon_r) = \frac{\sigma_{c\bar c}(W)}{\sigma_{\gamma p}(W)}",
        "model",
        "am3_mojo: c cbar pairs per interaction, the step mean of "
        "sigma_cc/sigma_gp (the R column); W^2: Hümmer et al. 2010, eq. 5",
        **C,
    ),
    Entry(
        "cp.threshold",
        r"charm is made above $\epsilon_r$ = 8.71 GeV, the threshold of",
        "derived",
        "kinematics (Hümmer et al. 2010, eq. 5), PDG 2026 masses "
        "(gamma p -> Lambda_c+ anti-D0)",
        "TeX " + HUEMMER,
        **C,
    ),
    Entry(
        "cp.sigma_gp",
        r"\sigma_{\gamma p} = \delta_\gamma\Big[H\ln^2(s/s_M)",
        "rearranged",
        "PDG 2016 review, Sec. 51: the HPR1R2 form and its gamma "
        "p factorisation (p. 8), the parameters (p. 9); the curve PDG "
        "2026 serves",
        "PDF PDG-2016_review_cross-section-plots.pdf",
        **C,
    ),
    Entry(
        "cp.sigma_cc",
        r"$\sigma_{c\bar c}$ is fitted to the measurements from "
        r"20 GeV to HERA:",
        "derived",
        "fit to SLAC HFP, PEC, NA14/2, E691, BFP, E687 (prel.), ZEUS, "
        "H1 (tests/support/measured/charm_photoproduction.py)",
        "HEPData ins214735, ins230940, ins333271, ins289318, ins392566, "
        "ins417960; TeX arXiv-hep-ex9502002v1, arXiv-hep-ex9604005v1",
        **C,
    ),
    Entry(
        "cp.ipsat",
        r"Above $W$ = 275 GeV, where the HERA data end, $\sigma_{c\bar c}$ "
        r"follows the IP-SAT",
        "model",
        "Goncalves, Machado & Meneses 2009, Fig. 1 (digitised), scaled to "
        "the fit at 275 GeV",
        "TeX arXiv-0905.2067v1.tar.gz",
        **C,
    ),
    Entry(
        "cp.fractions",
        r"Leaving those out, am3_mojo renormalizes to one hadron:",
        "derived",
        "Lisovyi et al. 2016, Table 12 ('Constrained S'), renormalised",
        "TeX arXiv-1509.01061v4.tar.gz (Tables/average_tab2.txt)",
        **C,
    ),
    Entry(
        "cp.associated",
        r"\mathcal{A} = \frac{N_+ - N_-}{N_+ + N_-}",
        "derived",
        "FOCUS (Link et al. 2004) asymmetry; Abe et al. 1986 (0.71 at "
        "20 GeV); the h_a column",
        "TeX arXiv-hep-ex0311022v1.tar.gz",
        **C,
    ),
    Entry(
        "cp.resolved",
        r"| Resolved share | 0.0833 | 0.1923 | 0.2936 |",
        "derived",
        "Frixione, Mangano, Nason & Ridolfi 1995, Tables 1-2 (m_c = 1.5 GeV, "
        "mu = m_c); the h_N column",
        "TeX arXiv-hep-ph9412348v1.tar.gz",
        **C,
    ),
    Entry(
        "cp.chi",
        r"On the photon side, "
        r"$\chi = \langle E^\ast - p_L^\ast\rangle/\sqrt{s}$",
        "derived",
        "kinematics (the boost to the source frame); the chi columns of the "
        "photon side and the associated anti-D",
        **C,
    ),
    Entry(
        "cp.e691",
        r"d\sigma = \pi F(x_F)\thinspace h(p_T^2)",
        "derived",
        "E691 (Anjos et al. 1989, HEPData ins289318 Tables 10-11); invariant "
        "cross section, kinematics",
        "HEPData ins289318",
        **C,
    ),
    Entry(
        "cp.recoil",
        r"$E^\ast = (s + m^2 - m_X^2)/(2\sqrt{s})$",
        "derived",
        "two-body kinematics; charge, baryon number, charm and strangeness; "
        "PDG 2026 masses",
        **C,
    ),
    Entry(
        "cp.nucleon",
        r"\chi_N = \bar z\thinspace \frac{\int_{x_c}^1 g(x)",
        "derived",
        "the pp closed form and fragmentation (cpp.g, cpp.peterson); "
        "the nucleon side's chi column",
        **C,
    ),
    Entry(
        "cp.table",
        r"| Type, $\epsilon_r$ [GeV] | $R$ | $h_a$ | $h_N$ |",
        "derived",
        "the steps inside the types M4-M7 of Hümmer et al. 2010, Table 6; "
        "the columns: cp.R, cp.associated, cp.resolved, cp.chi, cp.nucleon",
        "TeX " + HUEMMER,
        "tests/unit/test_added_processes.py (test_charm_steps_*)",
        **C,
    ),
    Entry(
        "cp.parts",
        r"| Part | Hadrons per interaction | $\chi$ |",
        "model",
        "am3_mojo: the parts of a step (charm and baryon number; "
        "the neutrons' charge); the associated Lambda_c+ at one less the "
        "inelasticity of Hümmer et al. 2010, Table 6",
        "TeX " + HUEMMER,
        "tests/unit/test_added_processes.py (test_charm_steps_*)",
        **C,
    ),
    Entry(
        "cp.bookkeeping",
        r"s_{t,r} = 1 - \frac{\sum_k \sum_h",
        "model",
        "am3_mojo (energy bookkeeping, as for the kaons): the type's "
        "response less kappa_r f_r",
        **C,
    ),
    Entry(
        "cpp.source",
        r"Q_h(E) = c\thinspace n_H\int \frac{dE_p}{E_p}",
        "standard",
        "pp source: the collisions' hadrons counted (as for the kaons); "
        "the fragmentation of the quark spectrum",
        **C,
    ),
    Entry(
        "cpp.g",
        r"g(x) = A\thinspace P(x)\thinspace "
        r"P\negthinspace\left(\frac{x_c^2}{x}\right)",
        "model",
        "am3_mojo closed form; symmetric in rapidity (x -> x_c^2/x), "
        "s and x_c kinematics",
        **C,
    ),
    Entry(
        "cpp.fit",
        r"\log_{10} x_0 = b_0 + b_1 l + b_2 l^2",
        "derived",
        "fit to Bhattacharya et al. 2015, Fig. 2 (at the level of 2016, "
        "Fig. 12) and Carpio et al. 2020, Fig. 7",
        "TeX arXiv-1502.01076v2.tar.gz",
        "tests/unit/test_added_processes.py "
        "(test_pp_charm_spectrum_against_nlo)",
        **C,
    ),
    Entry(
        "cpp.peterson",
        r"D_h(z) = N_h\thinspace\frac{z(1-z)^2}{[(1-z)^2 + \epsilon_h z]^2}",
        "transcribed",
        "Kniehl & Kramer 2006, eq. (1) and Table I (LO, c); N_h for the "
        "photo-hadronic f_h",
        "TeX arXiv-hep-ph0607306v1.tar.gz",
        **C,
    ),
    Entry(
        "cpp.sigma_table",
        r"| $\sigma_{c\bar c}$ [μb] | 1.51 | 38.4 | 252 |",
        "transcribed",
        "Bhattacharya et al. 2016, Table 1 (pp, M_F,R prop. to m_T)",
        "TeX arXiv-1607.00193v2.tar.gz",
        "tests/unit/test_added_processes.py "
        "(test_pp_charm_cross_section_table)",
        **C,
    ),
    Entry(
        "cpp.continuation",
        r"Up to $10^{11}$ GeV, the energy of Carpio et al.'s spectra, "
        r"the slope of the last decade continues.",
        "model",
        "am3_mojo extension (Carpio et al. 2020: 1e11 GeV)",
        "TeX arXiv-2007.07945v2.tar.gz",
        **C,
    ),
    Entry(
        "cpp.threshold",
        r"\sigma_{c\bar c} = \sigma_2\left[\frac{1 - s_{\rm th}/s}{1 - "
        r"s_{\rm th}/s_2}\right]^{a}",
        "model",
        "am3_mojo: value and slope continuous at 100 GeV (a = 8.954); "
        "threshold pp -> p Lambda_c+ anti-D0",
        **C,
    ),
    Entry(
        "cpp.delta",
        r"**GeV (below the published spectra).** As for the kaons, "
        r"a $\delta$-functional approximation applies",
        "model",
        "the delta-functional approximation of Kelner et al. 2006, "
        "eqs. 75-77 (as for the kaons); continuous at E_t; the two-bin split "
        "keeps number and energy",
        **C,
    ),
    Entry(
        "cpp.kernel",
        r"E\thinspace Q_h(E) = c\thinspace n_H\int d\ln E_p",
        "derived",
        "the pp source and fragmentation per unit ln E and ln z, and their "
        "rectangle rules on the grid",
        **C,
    ),
    Entry(
        "cpp.loss",
        r"\sigma\kappa_C = 2\sum_h\int_0^1 x\thinspace d\sigma_h",
        "model",
        "am3_mojo (energy bookkeeping, as for the kaons): the protons lose "
        "the hadrons' energy",
        **C,
    ),
    Entry(
        "ch.evolution",
        r"| Hadron | Mass [MeV] | Mean life [s] | Evolved as | PDG "
        r"2026 listing |",
        "transcribed",
        "PDG 2026 listings of D+-, D0, Ds+- and Lambda_c+: masses (OUR FIT) "
        "and mean lives (OUR AVERAGE)",
        PDG_C,
        **C,
    ),
]
ALSO = {
    "cp.response": (r"y = E_N\epsilon/(m_pc^2)", r"\epsilon_r = 2y"),
    "cp.source": (
        r"Q_h(E) = \sum_r \sum_k \sum_{N=p,n} "
        r"\frac{M_{k,N}^h}{\chi_k^h}",
    ),
    "cp.grid": (
        r"w = (e^{f\Delta} - 1)/(e^\Delta - 1)",
        r"f + f(1 - f)\Delta/2",
    ),
    "cp.R": (
        r"\sigma_{c\bar c} = A\left(1 - \frac{W_{\rm",
        r"W^2 = m_p^2 + 2m_p\epsilon_r",
    ),
    "cp.threshold": (r"W_{\rm th} = m_{\Lambda_c} + m_{\mathrm{D}^0}",),
    "cp.sigma_gp": (r"s_M = (m_\rho + m_p + M)^2",),
    "cp.associated": (
        r"h_a = \frac{2\mathcal{A}f_{\Lambda_c}}{1 - \mathcal{A} "
        r"+ 2\mathcal{A}f_{\Lambda_c}}",
        r"N_+ = h_a + (1 - h_a)f_{\Lambda_c}",
        r"N_- = (1 - h_a)f_{\Lambda_c}",
        r"h_a = 0.71\thinspace(\epsilon_r/\epsilon_a)^{-1.765}",
    ),
    "cp.chi": (r"E_N^\ast + p_N^\ast = \sqrt{s}",),
    "cp.e691": (
        r"q = \frac{2E^\ast}{\sqrt{s}} = \sqrt{x_F^2 + a^2}",
        r"a = \frac{2m_T}{\sqrt{s}}",
        r"\chi = \frac{q - x_F}{2}",
        r"\int \frac{dx_F}{q} = \mathrm{arsinh}\frac{x_F}{a}",
        r"\int \chi\thinspace \frac{dx_F}{q} = \frac{x_F - q}{2}",
        r"x_F = 2p_L^\ast/\sqrt{s}",
    ),
    "cp.nucleon": (r"\bar z = \sum_h f_h\langle z\rangle_h",),
    "cp.bookkeeping": (r"\kappa_r = 1 - s_{t,r}",),
    "cpp.source": (r"x\frac{d\sigma_h}{dx} = \sigma_{c\bar c}\int_x^1",),
    "cpp.g": (
        r"P(u) = \frac{(1-u)^n}{1 + u/x_0}",
        r"x_c = \frac{m_T}{\sqrt{s}}",
        r"\int_{x_c^2}^1 g(x)\thinspace\frac{dx}{x} = 1",
        r"x = x_c e^{y^\ast}",
    ),
    "cpp.fit": (
        r"n = n_0 + n_1 l",
        r"l = "
        r"\min\negthinspace\left(\log_{10}\frac{E_p}"
        r"{10^6\thinspace\mathrm{GeV}}, 5\right)",
    ),
    "cpp.peterson": (r"\int_0^1 D_h(z)\thinspace dz = f_h",),
    "cpp.threshold": (
        r"\sqrt{s_{\rm th}} = m_p + m_{\Lambda_c} + "
        r"m_{\mathrm{D}^0}",
    ),
    "cpp.delta": (r"w = (E_h - E_l)/(E_{l+1} - E_l)",),
    "cpp.kernel": (
        r"x\frac{d\sigma_h}{dx} = \sigma_{c\bar c}\int d\ln z",
        r"E_iQ_h(E_i) \approx n_H\Delta\sum_j",
        r"K_{ij} = c\thinspace\sigma_{c\bar c}\thinspace d\sum_l",
        r"d\sum_m g(e^{-md}) = 1",
        r"d\sum_l z_lD_h(z_l) = f_h",
        r"\int g\thinspace d\ln x = 1",
        r"\int zD_h\thinspace d\ln z = f_h",
    ),
}

# the arXiv sources and PDFs in knowledge/papers/ (git-ignored)
LISOVYI = "arXiv-1509.01061v4.tar.gz"  # Lisovyi, Verbytskyi & Zenaiev
GMM = "arXiv-0905.2067v1.tar.gz"  # Goncalves, Machado & Meneses
FMNR = "arXiv-hep-ph9412348v1.tar.gz"  # Frixione, Mangano, Nason & Ridolfi
FOCUS = "arXiv-hep-ex0311022v1.tar.gz"  # Link et al. (FOCUS)
ZEUS95 = "arXiv-hep-ex9502002v1.tar.gz"  # Derrick et al. (ZEUS)
H1_96 = "arXiv-hep-ex9604005v1.tar.gz"  # Aid et al. (H1)
KK06 = "arXiv-hep-ph0607306v1.tar.gz"  # Kniehl & Kramer
# Bhattacharya, Enberg, Jeong, Kim, Reno, Sarcevic & Stasto
BERSS16 = "arXiv-1607.00193v2.tar.gz"
CARPIO = "arXiv-2007.07945v2.tar.gz"  # Carpio, Murase, Reno, Sarcevic & Stasto
PDG16 = "PDG-2016_review_cross-section-plots.pdf"
# AM3's C++ original (git-ignored): AM3_SOURCE_DIR, else the clone of
# tests/reference/setup_reference.sh in its work directory
_WORK = os.environ.get(
    "AM3_REFERENCE_DIR", str(L.ROOT / "tests" / "reference" / "work")
)
AM3_MATH = (
    Path(os.environ.get("AM3_SOURCE_DIR", os.path.join(_WORK, "am3")))
    / "src"
    / "math.cc"
)

# NaN chi where no hadron is allowed
warnings.filterwarnings("ignore", message="invalid value encountered")

PG, PPD = L.measured("charm_photoproduction"), L.measured("charm_pp")


def piece(snippet, part):
    assert part in snippet, part
    return part


def in_member(what, name, member, *snippets):
    """in_source for a file of an arXiv source that is not TeX (Lisovyi et al.'s
    tables are .txt files)."""
    found = L.contains(L.tar_member(name, member), *snippets)
    if found is None:
        return absent(what, name)
    return Result(
        what,
        found,
        f"in {member} of {name}"
        if found
        else f"not found in {member} of {name}",
    )


@lru_cache(None)
def pdf_text(name):
    """TeX fonts keep their own codes, and " | " joins the text objects."""
    path = L.PAPERS / name
    if not path.exists():
        return None
    data, out = path.read_bytes(), []
    for m in re.finditer(rb"stream\r?\n", data):
        try:
            stream = zlib.decompress(
                data[m.end() : data.find(b"endstream", m.end())]
            )
        except zlib.error:
            continue
        for block in re.findall(rb"BT(.*?)ET", stream, re.S):
            ops = [
                a or b
                for a, b in re.findall(
                    rb"\[(.*?)\]\s*TJ|(\((?:\\.|[^\\)])*\))\s*Tj", block, re.S
                )
            ]
            if ops:
                out.append(
                    b"".join(
                        re.findall(rb"\(((?:\\.|[^\\)])*)\)", b" ".join(ops))
                    )
                )
    return b" | ".join(out).decode("latin-1")


def in_pdf(what, name, *snippets):
    """The snippets (copied from pdf_text) occur in the PDF's text."""
    found = L.contains(pdf_text(name), *snippets)
    if found is None:
        return absent(what, name)
    return Result(
        what,
        found,
        f"in the text of {name}" if found else f"not found in {name}",
    )


def pdf_page(name, snippet, page_of):
    text = pdf_text(name)
    if text is None:
        return None
    i = text.find(snippet)
    return page_of(text, i) if i >= 0 else -1


def pdg16_page(text, i):
    """The running heads put the page number before the section's title on even
    pages and after it on odd ones."""
    heads = [
        (m.start(), int(m.group(1) or m.group(2)))
        for m in re.finditer(
            r"(?:\| |^)(\d*)51\.Plotsofcrosssectionsandrelatedquantities(\d*)",
            text,
        )
    ]
    return [p for pos, p in heads if pos < i][-1]


def pdg26_page(text, i):
    return next(
        int(m.group(1))
        for m in re.finditer(r"Page(\d+)Created", text)
        if m.start() > i
    )


def in_am3(what, *lines):
    """The lines (AM3's C++ as transcribed) occur in its math.cc."""
    found = L.contains(
        AM3_MATH.read_text() if AM3_MATH.exists() else None, *lines
    )
    if found is None:
        return Result(
            what,
            True,
            f"not re-checked: no {AM3_MATH} "
            "(tests/reference/setup_reference.sh)",
        )
    return Result(
        what,
        found,
        "in AM3's src/math.cc" if found else "not found in AM3's src/math.cc",
    )


def close(a, b):
    return len(a) == len(b) and all(
        abs(float(x) - float(y)) <= 1e-9 * max(1.0, abs(float(y)))
        for x, y in zip(a, b)
    )


def number(text):
    """The value of a number of the page, the float nearest to it."""
    return float("{}e{}".format(*split_number(text)))


def rounds_as(value, text):
    """rounds_to, but exact for a number without decimals, such as 1 or 0."""
    mantissa, e = split_number(text)
    if "." not in mantissa:
        return value == float(mantissa) * 10.0**e
    return rounds_to(value, text)


def search(pattern, pin):
    m = re.search(pattern, L.line(pin))
    if m is None:
        raise LookupError(
            f"charm: no match on the line of {pin[:60]}: {pattern[:80]}"
        )
    return m.groups()


EVOLUTION_TABLE = (
    r"| Hadron | Mass [MeV] | Mean life [s] | Evolved as | PDG 2026 listing |"
)
EVOLUTION_ROWS = {
    "D+": r"$\mathrm{D}^\pm$",
    "D0": r"$\mathrm{D}^0$, $\bar{\mathrm{D}}^0$",
    "Ds+": r"$\mathrm D_s^\pm$",
    "Lc+": r"$\Lambda_c^+$, $\bar\Lambda_c^-$",
}


def evolution_rows():
    rows = {r[0]: r for r in L.table(EVOLUTION_TABLE)}
    return {h: rows[tex] for h, tex in EVOLUTION_ROWS.items()}


def readme_charm_masses():
    return {h: sp.Rational(r[1]) / 1000 for h, r in evolution_rows().items()}


CODE_MASSES = {
    "p": "MASS_P",
    "n": "MASS_N",
    "pi+": "MASS_PI",
    "pi0": "MASS_PI0",
    "K+": "MASS_K",
    "K0": "MASS_K0",
    "Lambda": "MASS_LAMBDA",
    "Sigma+": "MASS_SIGMA_PLUS",
    "Sigma0": "MASS_SIGMA0",
    "Sigma-": "MASS_SIGMA_MINUS",
}


def masses():
    """{particle: mass [GeV]}, exact: the code's masses of the light hadrons,
    the page's of the charm hadrons."""
    code = mojo_constants.read(CODE_MASSES.values())
    m = {
        **{q: code[n] for q, n in CODE_MASSES.items()},
        **readme_charm_masses(),
    }
    m.update(
        {
            "pi-": m["pi+"],
            "K-": m["K+"],
            "anti-K0": m["K0"],
            "anti-D0": m["D0"],
            "D-": m["D+"],
            "Ds-": m["Ds+"],
            "anti-Lc-": m["Lc+"],
        }
    )
    return m


MASS_Q = masses()
MASS = {q: float(v) for q, v in MASS_Q.items()}
MP = MASS["p"]
HADRONS = ("D0", "D+", "Ds+", "Lc+")  # the page's order of the fractions
# (charge, baryon number, charm, strangeness) of the particles of the recoils
# and their candidates
QN = {
    "p": (1, 1, 0, 0),
    "n": (0, 1, 0, 0),
    "pi+": (1, 0, 0, 0),
    "pi0": (0, 0, 0, 0),
    "pi-": (-1, 0, 0, 0),
    "K+": (1, 0, 0, 1),
    "K0": (0, 0, 0, 1),
    "K-": (-1, 0, 0, -1),
    "anti-K0": (0, 0, 0, -1),
    "Lambda": (0, 1, 0, -1),
    "Sigma+": (1, 1, 0, -1),
    "Sigma0": (0, 1, 0, -1),
    "Sigma-": (-1, 1, 0, -1),
    "D0": (0, 0, 1, 0),
    "anti-D0": (0, 0, -1, 0),
    "D+": (1, 0, 1, 0),
    "D-": (-1, 0, -1, 0),
    "Ds+": (1, 0, 1, 1),
    "Ds-": (-1, 0, -1, -1),
    "Lc+": (1, 1, 1, 0),
    "anti-Lc-": (-1, -1, -1, 0),
}
GAMMA_P = (1, 1, 0, 0)  # the quantum numbers of gamma p

STEP_TABLE = r"| Type, $\epsilon_r$ [GeV] | $R$ | $h_a$ | $h_N$ |"
COLUMNS = ("R", "h_a", "h_N", "chi_D", "chi_Ds", "chi_Lc", "chi_aD", "chi_N")


def step_rows():
    rows = []
    for cells in L.table(STEP_TABLE):
        m = re.fullmatch(
            r"M(\d), (?:([\d.]+)–([\d.]+)|\$(.+)\\text\{–\}(.+)\$)", cells[0]
        )
        lo, hi = (
            (m.group(2), m.group(3)) if m.group(2) else (m.group(4), m.group(5))
        )
        chi = (
            [None] * 3
            if cells[4] == "—"
            else [c.strip() for c in cells[4].split(",")]
        )
        values = [cells[1], cells[2], cells[3], *chi, cells[5], cells[6]]
        rows.append(
            (
                int(m.group(1)) - 1,
                lo,
                hi,
                {k: (None if v == "—" else v) for k, v in zip(COLUMNS, values)},
            )
        )
    return rows


def readme_charm_steps():
    return [
        dict(
            t=t,
            eps_min=number(lo),
            eps_max=number(hi),
            **{k: (None if v is None else number(v)) for k, v in cells.items()},
        )
        for t, lo, hi, cells in step_rows()
    ]


FRACTIONS = r"Leaving those out, am3_mojo renormalizes to one hadron:"


def fraction_line():
    return search(
        r"becomes \$\\mathrm\{D\}\^0\$, \$\\mathrm\{D\}\^\+\$, "
        r"\$\\mathrm D_s\^\+\$ or \$\\Lambda_c\^\+\$ with the fractions "
        r"\$f_h\$ of Lisovyi et al\. \(2016\)\. An anti-quark becomes the "
        r"antiparticles with the same fractions\. Their Table (\d+), column "
        r"\"([^\"]+)\", averages \$\\mathrm\{e\}\^\+\\mathrm\{e\}\^-\$, ep and "
        r"pp data\. That column constrains the sum \$S\$ of all fractions to "
        r"one and gives ([\d.]+), ([\d.]+), ([\d.]+) and ([\d.]+)\. These add "
        r"up to ([\d.]+), and the charm-strange baryons make the rest, "
        r"([\d.]+) per \$\\Lambda_c\^\+\$\. Leaving those out, am3_mojo "
        r"renormalizes to one hadron: ([\d.]+), ([\d.]+), ([\d.]+) and "
        r"([\d.]+)\.",
        FRACTIONS,
    )


def fractions_exact():
    raw = {h: sp.Rational(v) for h, v in zip(HADRONS, fraction_line()[2:6])}
    total = sum(raw.values())
    return {h: raw[h] / total for h in HADRONS}


def readme_fractions():
    """{hadron: f_h} as floats (fractions_exact), not the page's rounded
    print."""
    return {h: float(v) for h, v in fractions_exact().items()}


PP_PARAMS = r"with $s = 2m_p(m_p + E_p)$, $m_T$ = "
EPS_LINE = r"with Peterson's parameter $\epsilon_h$ = "
SIGMA_TABLE = r"| $\sigma_{c\bar c}$ [μb] |"
# (without the words of the rule above the table, which a control of
#  cpp.continuation changes)
TOP_LINE = r"the energy of Carpio et al.'s spectra, the slope of the last"
DECADES = {"decade": 1, "two decades": 2, "three decades": 3}
A_LINE = (
    r"where $\sigma_2$ and $s_2$ are the cross section and $s$ at "
    r"100 GeV, and the exponent $a = "
)
ET_LINE = r"**GeV (below the published spectra).**"
FORWARD_LINE = r"of the hadrons whose quarks have $y^\ast \gt "
NAMES = ("mT", "b0", "b1", "b2", "n0", "n1")


def pp_param_texts():
    return search(
        r"\$m_T\$ = ([\d.]+) GeV, \$b_0 = (-?[\d.]+)\$, "
        r"\$b_1 = (-?[\d.]+)\$, \$b_2 = (-?[\d.]+)\$, "
        r"\$n_0 = (-?[\d.]+)\$ and \$n_1 = (-?[\d.]+)\$\.",
        PP_PARAMS,
    )


def readme_pp_charm():
    out = dict(zip(NAMES, map(float, pp_param_texts())))
    e = search(
        r"with Peterson's parameter \$\\epsilon_h\$ = ([\d.]+) "
        r"\(\$\\mathrm\{D\}\^\+\$\), ([\d.]+) \(\$\\mathrm\{D\}\^0\$\), "
        r"([\d.]+) \(\$\\mathrm D_s\^\+\$\) and ([\d.]+) "
        r"\(\$\\Lambda_c\^\+\$\)",
        EPS_LINE,
    )
    out["eps"] = dict(zip(("D+", "D0", "Ds+", "Lc+"), map(float, e)))
    lines = L.text().splitlines()
    i = lines.index(L.line(SIGMA_TABLE))
    head = [c.strip() for c in lines[i - 2].strip().strip("|").split("|")]
    vals = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    assert head[0] == "$E_p$ [GeV]" and vals[0] == r"$\sigma_{c\bar c}$ [μb]", (
        head[0],
        vals[0],
    )
    out["sigma"] = {
        10.0 ** int(re.fullmatch(r"\$10\^\{?(\d+)\}?\$", h).group(1)): float(v)
        for h, v in zip(head[1:], vals[1:])
    }
    out["a"] = float(
        search(
            r"the exponent \$a = ([\d.]+)\$ keeps the "
            r"slope there",
            A_LINE,
        )[0]
    )
    out["E_t"] = 10.0 ** int(
        search(r"\$\\boldsymbol\{E_p \\lt E_t = 10\^(\d+)\}\$", ET_LINE)[0]
    )
    out["E_top"] = 10.0 ** int(
        search(
            r"Up to \$10\^\{(\d+)\}\$ GeV, "
            r"the energy of Carpio",
            TOP_LINE,
        )[0]
    )
    return out


def readme_forward_ystar():
    return float(
        L.to_sympy(
            search(
                r"whose quarks have \$y\^\\ast \\gt ([^$]+)\$", FORWARD_LINE
            )[0],
            {},
        )
    )


# --- Hümmer et al. (2010), TeX source (\drb = Gamma^IT, \dsb = sigma^IT) ---

H_EQ4 = (
    r"\drb (E_p) = \frac{1}{2} \frac{m_p^2}{E_p^2} "
    r"\int\limits_{\frac{\epsilon_{\mathrm{th}} m_p}{2 E_p}}^{\infty} "
    r"d \varepsilon \frac{n_\gamma(\varepsilon)}{\varepsilon^2} "
    r"\int\limits_{\epsilon_{\mathrm{th}}}^{2 E_p \varepsilon/m_p} "
    r"d \epsilon_r \, \epsilon_r  \, \dsb(\epsilon_r)"
)
H_EQ5 = r"s(\epsilon_r) = m_p^2 + 2 \, m_p \, \epsilon_r"
H_EQ8 = (
    r"x \equiv \frac{E_b}{E_p} \, , \quad  y\equiv "
    r"\frac{E_p \, \varepsilon}{m_p}"
)
# after eq. 3 (sic: \cos_{p\gamma} for cos theta)
H_EPSR = r"$\epsilon_r=E_p \varepsilon/m_p (1 - \cos_{p \gamma})$"
H_EQ27 = (
    r"R^{\mathrm{IT}}(x,y)= \delta(x - \chi^{\mathrm{IT}}) \,  "
    r"M_b^{\mathrm{IT}} \, f^{\mathrm{IT}}(y)  \quad \text{with} \quad "
    r"f^{\mathrm{IT}}(y) \equiv \frac{1}{2y^2} "
    r"\int\limits_{\epsilon_{\mathrm{th}}}^{2y} d \epsilon_r \, "
    r"\epsilon_r \, \sigma^{\mathrm{IT}}(\epsilon_r)"
)
H_EQ28 = (
    r"Q_b^{\mathrm{IT}} = N_p \left( \frac{E_b}{\chi^{\mathrm{IT}}} \right) "
    r"\frac{m_p}{E_b} \int\limits_{\epsilon_{\mathrm{th}}/2}^\infty dy \, "
    r"n_\gamma \left( \frac{m_p \, y \, \chi^{\mathrm{IT}}}{E_b} \right) \,  "
    r"M_b^{\mathrm{IT}}  \, f^{\mathrm{IT}}(y)"
)
# Table 6: the rows of the types M4-M7 (IT, eps_min, eps_max [GeV],
# sigma [mubarn], chi, K, the pions from protons and from neutrons)
H_TABLE6 = (
    r"M$_\text{4L}$& 5.0&   50& 120& 0.07& 0.49 & 1.38& 1.37& 1.11& "
    r"1.38& 1.11& 1.37\\",
    r"M$_\text{4H}$& 5.0& 50& 120 & 0.35& --& 0.16& 0.25& 0.23&0.16& "
    r"0.23& 0.25\\",
    r"M$_\text{5L}$& 50& 500& 120 & 0.02& 0.45 & 3.01& 2.86& "
    r"2.64&3.01& 2.64& 2.86\\",
    r"M$_\text{5H}$& 50& 500& 120 & 0.5&-- & 0.20& 0.21& 0.14&0.20& "
    r"0.14& 0.21\\",
    r"M$_\text{6L}$& 500& 5000& 120 & 0.007& 0.44 & 5.13& 4.68& "
    r"4.57&5.13& 4.57& 4.68\\",
    r"M$_\text{6H}$& 500& 5000& 120 & 0.5& --& 0.27& 0.29& 0.12&0.27& "
    r"0.12& 0.29\\",
    r"M$_\text{7L}$& 5000& $\infty$ & 120 & 0.002& 0.44 & 7.59& 6.80& "
    r"6.65&7.59& 6.65& 6.80\\",
    r"M$_\text{7H}$& 5000& $\infty$ & 120 & 0.6& --& 0.26& 0.27& 0.13 "
    r"& 0.26& 0.13& 0.27\\",
)
H_TABLE6_HEAD = (
    r"IT & $\epsilon_{\mathrm{min}}^{\mathrm{IT}}$ [GeV] & "
    r"$\epsilon_{\mathrm{max}}^{\mathrm{IT}}$ [GeV] &  $\sigma$ &$\chi$  &  K "
    r"& $M_{\pi^0}$ & $M_{\pi^+}$ & $M_{\pi^-}$ &  $M_{\pi^0}$ & $M_{\pi^+}$ & "
    r"$M_{\pi^-}$\\",
    r"The inelasticities for the H-sample are included in the L-sample.",
)


def table6():
    out = {}
    for row in H_TABLE6:
        cells = [c.strip() for c in row.rstrip("\\").split("&")]
        t, part = (
            int(re.search(r"\{?(\d)([LH])", cells[0]).group(1)) - 1,
            re.search(r"\d([LH])", cells[0]).group(1),
        )
        if part == "L":
            out[t] = dict(
                lo=float(cells[1]),
                hi=math.inf if "infty" in cells[2] else float(cells[2]),
                sigma=float(cells[3]),
                K=sp.Rational(cells[5]),
            )
    return out


def table_number(tex, label):
    if tex is None:
        return None
    return (
        re.findall(r"\\begin\{table\}.*?\\label\{([^}]*)\}", tex, re.S).index(
            label
        )
        + 1
    )


yy, er_, eps_, c_, E_, chi_, M_ = sp.symbols(
    "y epsilon_r epsilon c E chi M", positive=True
)
EN_, ms_ = sp.symbols("E_N m_p", positive=True)
E1, E2, ST = sp.symbols("epsilon_min epsilon_max sigma_t", positive=True)
SIG, G, NG, NN, TT = (
    sp.Function(name) for name in ("sigma", "G", "n_gamma", "N", "t")
)
MC2 = ms_ * c_**2  # Hümmer's m_p (c = 1 there)

F_PIN = r"f_r(y) = \frac{1}{2y^2}\int_0^{2y}"
RATE_PIN = r"t_r^{-1}(E_N) = c\int dy\thinspace"
GRID_PIN = r"E\thinspace Q_h(E) = \sum_r \sum_k"
W_SPLIT = r"w = (e^{f\Delta} - 1)/(e^\Delta - 1)"
W_AM3 = r"f + f(1 - f)\Delta/2"
AM3_SPLIT = (
    "(*i) = (int)floor(x);",
    "(*r) = x - 1.0 * (*i);",
    "(*r) += 0.5 * (*r) * (1.0 - (*r)) * dlnE;",
)


def eq27_f(arg):
    """Eq. 27's f^IT at arg, its integral as G(2y), G(u) = int_eps_th^u eps_r
    sigma d eps_r."""
    pre = L.to_sympy(piece(H_EQ27, r"\frac{1}{2y^2}"), {"y": yy})
    top = L.to_sympy(piece(H_EQ27, "^{2y}")[2:-1], {"y": yy})
    return (pre * G(top)).subs(yy, arg)


def readme_response(branch=None):
    lhs, mid, closed = L.sides(L.display(F_PIN))
    assert lhs == "f_r(y)", lhs
    pre, integral = mid.split(r"\int", 1)
    lo, up, body = re.fullmatch(
        r"_(\w+)\^\{(.+?)\} (.+)\\thinspace d\\epsilon_r", integral
    ).groups()
    form = (
        L.to_sympy(pre, {"y": yy}),
        L.to_sympy(lo, {}),
        L.to_sympy(up, {"y": yy}),
        L.to_sympy(body, {r"\epsilon_r": er_}, {r"\sigma_r": SIG}),
    )
    if branch is None:
        return form, None
    args_tex = L.between(closed, r"\min(", ")")
    tok = {
        "y": yy,
        r"\epsilon_{\rm max}": E2,
        r"\epsilon_{\rm min}": E1,
        r"\sigma_t": ST,
    }
    args = [L.to_sympy(a, tok) for a in args_tex.split(", ")]
    return (
        form,
        L.to_sympy(closed, {**tok, r"\min(" + args_tex + ")": branch(*args)}),
    )


def step_integral(pre, lo, up, body, pieces):
    Yv = sp.Symbol("Y", positive=True)
    sig = sum(
        v * (sp.Heaviside(er_ - a) - sp.Heaviside(er_ - b))
        for v, a, b in pieces
    )
    return pre * sp.integrate(body.subs(SIG(er_), sig), (er_, lo, Yv)).subs(
        Yv, up
    )


def readme_y():
    return L.to_sympy(
        L.sides(L.inline(r"y = E_N\epsilon/(m_pc^2)"))[1],
        {"E_N": EN_, r"\epsilon": eps_, "m_p": ms_, "c": c_},
    )


def readme_rate(EN_at):
    lhs, rhs = L.sides(L.part(L.display(RATE_PIN), r"t_r^{-1}(E_N)"))
    head, body = rhs.split(r"\int dy\thinspace ")
    assert lhs == r"t_r^{-1}(E_N)", lhs
    return (
        L.to_sympy(head, {"c": c_}),
        L.to_sympy(
            body,
            {"m_p": ms_, "c": c_, "E_N": EN_at, "y": yy},
            {r"n_\gamma": NG, "f_r": eq27_f},
        ),
    )


def eq4_in_y(EN_at):
    """Hümmer et al.'s eq. 4 (c = 1 there: times c), the inner integral as G
    (sigma = 0 below eps_th)."""
    pre = L.to_sympy(
        piece(H_EQ4, r"\frac{1}{2} \frac{m_p^2}{E_p^2}"),
        {"m_p": MC2, "E_p": EN_at},
    )
    dens = L.to_sympy(
        piece(H_EQ4, r"\frac{n_\gamma(\varepsilon)}{\varepsilon^2}"),
        {r"\varepsilon": eps_},
        {r"n_\gamma": NG},
    )
    up = L.to_sympy(
        piece(H_EQ4, r"2 E_p \varepsilon/m_p"),
        {"E_p": EN_at, r"\varepsilon": eps_, "m_p": MC2},
    )
    eps_y = MC2 * yy / EN_at
    return c_ * (pre * dens * G(up)).subs(eps_, eps_y) * sp.diff(eps_y, yy)


SOURCE_TERM = (
    r"\\sum_r \\sum_k \\sum_\{N=p,n\} (.+?)\\thinspace "
    r"\\Big\[(.+?)\\Big\]_ \{E_N = (.+)\}"
)
SOURCE_TOK = {"M_{k,N}^h": M_, r"\chi_k^h": chi_, "E": E_}


def readme_source_term():
    lhs, rhs = L.sides(L.part(L.display(RATE_PIN), "Q_h(E)"))
    assert lhs == "Q_h(E)", lhs
    factor, bracket, at = re.fullmatch(SOURCE_TERM, rhs).groups()
    EN_at = L.to_sympy(at, SOURCE_TOK)
    head, body = readme_rate(EN_at)
    return sp.expand(
        L.to_sympy(factor, SOURCE_TOK)
        * L.to_sympy(bracket, {"N_N": NN(EN_at), "t_r^{-1}": head * body})
    )


def eq28():
    """c times Hümmer et al.'s eq. 28 (c = 1 there): N_p -> N, IT -> a part of
    a step."""
    tok28 = {
        "E_b": E_,
        r"\chi^{\mathrm{IT}}": chi_,
        "m_p": MC2,
        "y": yy,
        r"M_b^{\mathrm{IT}}": M_,
    }
    head = L.to_sympy(
        piece(
            H_EQ28,
            r"N_p \left( \frac{E_b}{\chi^{\mathrm{IT}}} "
            r"\right) \frac{m_p}{E_b}",
        ),
        tok28,
        {"N_p": NN},
    )
    body = L.to_sympy(
        piece(
            H_EQ28,
            r"n_\gamma \left( \frac{m_p \, y \, \chi^{\mathrm{IT}}}{E_b} "
            r"\right) \,  M_b^{\mathrm{IT}}  \, f^{\mathrm{IT}}(y)",
        ),
        tok28,
        {r"n_\gamma": NG, r"f^{\mathrm{IT}}": eq27_f},
    )
    return sp.expand(c_ * head * body)


def readme_grid_term():
    lhs, rhs = L.sides(L.display(GRID_PIN))
    assert lhs == r"E\thinspace Q_h(E)", lhs
    factor, bracket, at = re.fullmatch(SOURCE_TERM, rhs).groups()
    EN_at = L.to_sympy(at, SOURCE_TOK)
    return L.to_sympy(factor, SOURCE_TOK) * L.to_sympy(
        bracket, {"E_N": EN_at, "N_N": NN(EN_at), "t_r^{-1}": TT(EN_at)}
    )


def readme_source_terms_as_functions():
    """The page's term of Q_h with N_N and t_r^-1 as functions of E_N (for
    cp.grid)."""
    lhs, rhs = L.sides(L.part(L.display(RATE_PIN), "Q_h(E)"))
    factor, bracket, at = re.fullmatch(SOURCE_TERM, rhs).groups()
    EN_at = L.to_sympy(at, SOURCE_TOK)
    return L.to_sympy(factor, SOURCE_TOK) * L.to_sympy(
        bracket, {"N_N": NN(EN_at), "t_r^{-1}": TT(EN_at)}
    )


R_PIN = r"R(\epsilon_r) = \frac{\sigma_{c\bar c}(W)}{\sigma_{\gamma p}(W)}"
SIGMA_LINE = r"$\sigma_{c\bar c}$ is fitted to the measurements from "
THRESHOLD_LINE = r"charm is made above $\epsilon_r$ = "
WTH_LINE = r"The lightest final state with charm is "
GP_PARAMS = r"The review's table gives $M$ = "
IPSAT_LINE = r"GeV, where the HERA data end, "
ASSOC_LINE = (
    r"makes a share $h_a$ of the pairs. Abe et al. (1986) found $h_a = "
)
ASSOC_PIN = r"\mathcal{A} = \frac{N_+ - N_-}{N_+ + N_-}"
HA_LINE = r"with the fraction $f_{\Lambda_c} = "
RESOLVED_LINE = r"$h_N$ is half the resolved share, interpolated in $\ln W$."
FMNR_TABLE = r"| $W$ [GeV] | 30 | 100 | 280 |"
FMNR_LINE = (
    r"give the NLO cross sections of the point-like photon (their Table "
)
CHI_LINE = (
    r"On the photon side, $\chi = \langle E^\ast - p_L^\ast\rangle/\sqrt{s}$"
)
TYPES_LINE = r"Four of them reach the charm threshold: M4 covers "
CURVES = {"IIM": PG.GMM09_IIM, "bCGC": PG.GMM09_BCGC, "IP-SAT": PG.GMM09_IPSAT}

A_, n_, d_, W_, Wth_, m_, er2_ = sp.symbols(
    "A n delta W W_th m_p epsilon_r", positive=True
)


def readme_R():
    parts = L.parts(L.display(R_PIN))
    assert len(parts) == 3, parts
    lhs, rhs = L.sides(parts[1])
    assert lhs == r"\sigma_{c\bar c}", lhs
    sig = L.to_sympy(
        rhs,
        {
            "A": A_,
            "n": n_,
            r"\delta": d_,
            "W": W_,
            r"W_{\rm th}": Wth_,
            r"\mathrm{GeV}": 1,
        },
    )
    lhs, rhs = L.sides(parts[2])
    assert lhs == "W^2", lhs
    return parts[0], sig, L.to_sympy(rhs, {"m_p": m_, r"\epsilon_r": er2_})


def readme_pg():
    _, sig, w2 = readme_R()
    fit = search(
        r"from (\d+) GeV to HERA: \$A\$ = ([\d.]+) μb, "
        r"\$n = ([\d.]+)\$ and \$\\delta = ([\d.]+)\$\.",
        SIGMA_LINE,
    )
    wth_tex, wth_text = search(
        r"\$W_\{\\rm th\} = (m_\{\\Lambda_c\} \+ m_\{\\mathrm\{D\}\^0\})\$ = "
        r"([\d.]+) GeV",
        WTH_LINE,
    )
    above = search(
        r"Above \$W\$ = (\d+) GeV, where the HERA data end, "
        r"\$\\sigma_\{c\\bar c\}\$ follows the ([\w-]+) dipole-model curve of "
        r"Gonçalves, Machado & Meneses \(2009\) \(their Fig\. 1\)\. IP-SAT "
        r"stands for impact-parameter saturation\. The curve is scaled to the "
        r"fit at (\d+) GeV\. Beyond the figure's end at (\d+) TeV, "
        r"it continues as the power law of its last ([\d.]+) decade in \$W\$\.",
        IPSAT_LINE,
    )
    wth = L.to_sympy(
        wth_tex,
        {r"m_{\Lambda_c}": MASS_Q["Lc+"], r"m_{\mathrm{D}^0}": MASS_Q["D0"]},
    )
    assoc = search(
        r"found \$h_a = ([\d.]+)\$ at (\d+) GeV, from the excess of "
        r".+? At (\d+) GeV, FOCUS measured the asymmetry "
        r"\$\\mathcal\{A\} = ([\d.]+)\$ of",
        ASSOC_LINE,
    )
    res = search(
        r"It is held above \$W\$ = (\d+) GeV\. At low energies, "
        r"it falls to zero at (\d+) GeV, on the line through its "
        r"values at (\d+) and (\d+) GeV\. This zero lies at the "
        r"photon energy \$\\epsilon_r\$ = (\d+) GeV\.",
        RESOLVED_LINE,
    )
    e691 = search(
        r"The mean is over the D-meson spectra of E691 at (\d+) GeV", CHI_LINE
    )
    th = search(
        r"charm is made above \$\\epsilon_r\$ = ([\d.]+) GeV, the threshold of "
        r"\$\\gamma\\mathrm\{p\} \\to "
        r"\\Lambda_c\^\+\\bar\{\\mathrm\{D\}\}\^0\$\.",
        THRESHOLD_LINE,
    )
    return dict(
        sig=sig,
        w2=w2,
        wth=float(wth),
        wth_q=wth,
        wth_text=wth_text,
        fit_texts=fit[1:],
        e_low=float(fit[0]),
        wj=float(above[0]),
        curve=above[1],
        w_norm=float(above[2]),
        w_end=1e3 * float(above[3]),
        decade=float(above[4]),
        h_a=tuple(float(v) for v in assoc),
        h_n=tuple(float(v) for v in res[:2]),
        h_n_line=tuple(float(v) for v in res[2:4]),
        h_n_eps=res[4],
        e691=float(e691[0]),
        threshold=th[0],
    )


# The HPR1R2 form of the PDG 2016 review (Sec. 51, p. 8) as transcribed from its
# PDF (PDG16_SNIPPETS[5] and [7]), with s_M^ab = (m_a + m_b + M)^2 and
# m_gamma* = m_rho(770). For gamma p, "exact factorization":
# H log^2(s/s_M) + P^{gamma p} -> delta [H log^2(s/s_M) + P^{pp}]. The table has
# no R_2 for gamma/p.
PDG16_FORM = (
    r"H\ln^2(s/s_M) + P^{ab} + R_1^{ab}(s/s_M)^{-\eta_1} "
    r"\pm R_2^{ab}(s/s_M)^{-\eta_2}"
)
PDG16_SM = r"(m_a + m_b + M)^2"
HP = dict(
    zip(
        (
            "delta",
            "H",
            "s",
            "s_M",
            "P",
            "R1",
            "eta1",
            "R2",
            "eta2",
            "M",
            "m_rho",
            "m_a",
            "m_b",
        ),
        sp.symbols(
            "delta_gamma H s s_M P R_1 eta_1 R_2 eta_2 M m_rho m_a m_b",
            positive=True,
        ),
    )
)


def pdg16_gamma_p():
    tok = {
        "H": HP["H"],
        "s_M": HP["s_M"],
        "s": HP["s"],
        "P^{ab}": HP["P"],
        "R_1^{ab}": HP["R1"],
        r"\eta_1": HP["eta1"],
        "R_2^{ab}": HP["R2"],
        r"\eta_2": HP["eta2"],
    }
    form = L.to_sympy(PDG16_FORM, tok, sign=+1)
    gp = form.subs(
        {
            HP["H"]: HP["delta"] * HP["H"],
            HP["P"]: HP["delta"] * HP["P"],
            HP["R2"]: 0,
        }
    )
    sm = L.to_sympy(PDG16_SM, {"m_a": HP["m_rho"], "m_b": m_, "M": HP["M"]})
    return gp, sm


def readme_hpr1r2():
    # also with a control's change
    first, second = L.parts(L.display(r"\sigma_{\gamma p} = \delta_\gamma"))
    lhs, rhs = L.sides(first)
    assert lhs == r"\sigma_{\gamma p}", lhs
    tok = {
        r"\delta_\gamma": HP["delta"],
        "H": HP["H"],
        "s_M": HP["s_M"],
        "s": HP["s"],
        r"P^{\rm pp}": HP["P"],
        r"R_1^{\gamma p}": HP["R1"],
        r"\eta_1": HP["eta1"],
    }
    lhs2, rhs2 = L.sides(second)
    assert lhs2 == "s_M", lhs2
    return (
        L.to_sympy(rhs, tok),
        L.to_sympy(rhs2, {r"m_\rho": HP["m_rho"], "m_p": m_, "M": HP["M"]}),
    )


def readme_gp_params():
    g = search(
        r"\$M\$ = ([\d.]+) GeV, \$H\$ = ([\d.]+) mb, \$\\eta_1 = ([\d.]+)\$, "
        r"\$\\delta_\\gamma = ([\d.]+) \\times 10\^\{(-\d)\}\$, "
        r"\$P\^\{\\rm pp\}\$ = ([\d.]+) mb and \$R_1\^\{\\gamma p\}\$ = "
        r"([\d.]+) mb\. In \$s_M\$, the photon counts with the mass of the "
        r"\$\\rho\(770\)\$, \$m_\\rho\$ = ([\d.]+) MeV\.",
        GP_PARAMS,
    )
    return dict(
        M=float(g[0]),
        H=float(g[1]),
        eta1=float(g[2]),
        delta=float(g[3]) * 10.0 ** int(g[4]),
        P_pp=float(g[5]),
        R1_gp=float(g[6]),
        m_rho=float(g[7]) / 1e3,
    )


def hpr1r2_numeric(p=None):
    """sigma_tot(gamma p) [mubarn] at s [GeV^2]."""
    p = readme_gp_params() if p is None else p
    sig, sm = readme_hpr1r2()
    sm_v = sm.subs({HP["m_rho"]: p["m_rho"], m_: MP, HP["M"]: p["M"]})
    expr = sig.subs(
        {
            HP["delta"]: p["delta"],
            HP["H"]: p["H"],
            HP["P"]: p["P_pp"],
            HP["R1"]: p["R1_gp"],
            HP["eta1"]: p["eta1"],
            HP["s_M"]: sm_v,
        }
    )
    return sp.lambdify(HP["s"], 1e3 * expr, "numpy")


MEAS_TABLE = (
    r"| Experiment | Target | Energy [GeV] | $\sigma_{c\bar c}$ per "
    r"nucleon [μb] | Errors up, down [μb] | Data |"
)


def fit_points(w_of):
    """E687's preliminary D*+- cross sections are converted to sigma(c cbar) as
    FN-555 does."""
    out = []
    for p in PG.SIGMA_CC:
        pdf = p.get("pdf", (0.0, 0.0))
        up, down = (
            math.sqrt(p["stat"][k] ** 2 + p["syst"][k] ** 2 + pdf[k] ** 2)
            for k in (0, 1)
        )
        out.append(
            (
                p["key"],
                p["W"] if "W" in p else w_of(p["E"]),
                p["sigma"],
                up,
                down,
            )
        )
    e = PG.E687_FN555
    for E, _, sig, st, sy in e["dstar"]:
        c = 1.0 / (e["dstar_per_cc"] * e["a_be"] ** e["alpha"])
        err = math.sqrt(st**2 + sy**2 + (e["extra_syst"] * sig) ** 2) * c
        out.append((f"E687_{E:g}", w_of(E), sig * c, err, err))
    return out


def gmm_log(curve, w, decade):
    """ln sigma [nb] of a curve of Goncalves et al. at W. Beyond the end of the
    figure at 1 TeV, the curve follows the power law of its last `decade`
    (in log10 W)."""
    c = np.array(curve)
    lw, ls = np.log(c[:, 0]), np.log(c[:, 1])
    lw_top = np.log(10 ** (np.log10(c[-1, 0]) - decade))
    slope = (ls[-1] - np.interp(lw_top, lw, ls)) / (lw[-1] - lw_top)
    lx = np.log(np.asarray(w, float))
    return np.where(
        lx <= lw[-1], np.interp(lx, lw, ls), ls[-1] + slope * (lx - lw[-1])
    )


def log_midpoints(lo, hi, n):
    edges = np.linspace(math.log(lo), math.log(hi), n + 1)
    return np.exp(0.5 * (edges[1:] + edges[:-1]))


def _sub(edges, parts):
    edges = np.asarray(edges, float)
    lo, hi = edges[:-1], edges[1:]
    f = (np.arange(parts) + 0.5) / parts
    return (
        (lo[:, None] + (hi - lo)[:, None] * f[None, :]).ravel(),
        np.repeat(np.arange(lo.size), parts),
        np.repeat((hi - lo) / parts, parts),
    )


E691_PIN = r"d\sigma = \pi F(x_F)\thinspace h(p_T^2)"
INTEG_PIN = r"\int \frac{dx_F}{q} = \mathrm{arsinh}\frac{x_F}{a}"
XF_INLINE = r"x_F = 2p_L^\ast/\sqrt{s}"
PT2_LINE = r"with $F$ constant in each $x_F$ bin and $h$ in each "
SHARES = {
    "half": 2,
    "third": 3,
    "quarter": 4,
    "fifth": 5,
    "sixth": 6,
    "eighth": 8,
    "tenth": 10,
}
KINLIM = r"E^\ast = (s + m^2 - m_X^2)/(2\sqrt{s})"
RECOIL_TABLE = r"| Hadron | $\mathrm{D}^0$ | $\mathrm{D}^+$ |"
PARTICLE_TEX = {
    r"\bar{\mathrm{D}}^0": "anti-D0",
    r"\mathrm D_s^+": "Ds+",
    r"\mathrm D_s^-": "Ds-",
    r"\bar\Lambda_c^-": "anti-Lc-",
    r"\Lambda_c^+": "Lc+",
    r"\mathrm{D}^0": "D0",
    r"\mathrm{D}^+": "D+",
    r"\mathrm{D}^-": "D-",
    r"\mathrm{p}": "p",
    r"\mathrm{n}": "n",
    r"\pi^+": "pi+",
    r"\mathrm{K}^+": "K+",
}
SPECIES = ("D+", "D-", "D0", "anti-D0", "Ds+", "Ds-", "Lc+", "anti-Lc-")
HADRON_OF = dict(
    zip(SPECIES, ("D+", "D+", "D0", "D0", "Ds+", "Ds+", "Lc+", "Lc+"))
)


def particles(cell):
    s, out = cell.strip().strip("$"), []
    while s:
        tex = next(
            t
            for t in sorted(PARTICLE_TEX, key=len, reverse=True)
            if s.startswith(t)
        )
        out.append(PARTICLE_TEX[tex])
        s = s[len(tex) :]
    return out


def readme_recoils():
    head = [
        c.strip() for c in L.line(RECOIL_TABLE).strip().strip("|").split("|")
    ]
    (row,) = L.table(RECOIL_TABLE)
    assert head[0] == "Hadron" and row[0] == "Lightest recoil", (
        head[0],
        row[0],
    )
    return {particles(h)[0]: particles(r) for h, r in zip(head[1:], row[1:])}


@lru_cache(None)
def recoil_masses(page_hash):
    return {s: sum(MASS[p] for p in ps) for s, ps in readme_recoils().items()}


def pt2_parts():
    return SHARES[
        search(r"and \$h\$ in each (\w+) of a \$p_T\^2\$ bin", PT2_LINE)[0]
    ]


XF_EDGES = np.asarray(PG.E691_XF_EDGES, float)
PT2, IPT, DPT2 = _sub(PG.E691_PT2_EDGES, pt2_parts())

xF_, a_, q_, Es_, pL_, s_, mT_, m0_, mX_ = sp.symbols(
    "x_F a q E_star p_L s m_T m m_X", positive=True
)
dxF, dpT2, pT2_ = sp.symbols("dx_F dp_T2 p_T2", positive=True)
FF, HH = sp.Function("F"), sp.Function("h")


def readme_e691():
    d_part, q_part, a_part, chi_part = L.parts(L.display(E691_PIN))
    lhs, rhs = L.sides(d_part)
    assert lhs == r"d\sigma", lhs
    dsig = L.to_sympy(
        rhs,
        {"dx_F": dxF, "dp_T^2": dpT2, "x_F": xF_, "p_T^2": pT2_, "q": q_},
        {"F": FF, "h": HH},
    )
    lhs, q1, q2 = L.sides(q_part)
    assert lhs == "q", lhs
    q1 = L.to_sympy(q1, {r"E^\ast": Es_, "s": s_})
    q2 = L.to_sympy(q2, {"x_F": xF_, "a": a_})
    lhs, rhs = L.sides(a_part)
    assert lhs == "a", lhs
    a = L.to_sympy(rhs, {"m_T": mT_, "s": s_})
    lhs, rhs = L.sides(chi_part)
    assert lhs == r"\chi", lhs
    return dsig, q1, q2, a, L.to_sympy(rhs, {"q": q_, "x_F": xF_})


def readme_primitives():
    first, second = L.parts(L.display(INTEG_PIN))
    lhs, rhs = L.sides(first)
    assert lhs == r"\int \frac{dx_F}{q}" and rhs.startswith(
        r"\mathrm{arsinh}"
    ), first
    p1 = sp.asinh(
        L.to_sympy(rhs[len(r"\mathrm{arsinh}") :], {"x_F": xF_, "a": a_})
    )
    lhs, rhs = L.sides(second)
    assert lhs == r"\int \chi\thinspace \frac{dx_F}{q}", lhs
    _, _, q2, _, _ = readme_e691()
    return p1, L.to_sympy(rhs, {"x_F": xF_, "q": q2})


@lru_cache(None)
def numeric_primitives(page_hash):
    p1, p2 = readme_primitives()
    return (
        sp.lambdify((xF_, a_), p1, "numpy"),
        sp.lambdify((xF_, a_), p2, "numpy"),
    )


def x_integrals(s, m, m_rec):
    """The integrals end at the kinematic limit
    E* = (s + m^2 - m_rec^2)/(2 sqrt(s)) of a recoil of mass m_rec."""
    p1, p2 = numeric_primitives(hash(L.text()))
    rs = math.sqrt(s)
    e_max = (s + m * m - m_rec * m_rec) / (2.0 * rs)
    mt2 = m * m + PT2[None, :]
    a = 2.0 * np.sqrt(mt2) / rs
    lo, hi = XF_EDGES[:-1, None] + 0.0 * a, XF_EDGES[1:, None] + 0.0 * a
    hi = np.minimum(
        hi, 2.0 * np.sqrt(np.clip(e_max * e_max - mt2, 0.0, None)) / rs
    )
    lo = np.minimum(lo, hi)
    return p1(hi, a) - p1(lo, a), p2(hi, a) - p2(lo, a)


@lru_cache(None)
def e691_invariant(e_gamma, page_hash):
    tx = np.array([v for v, _ in PG.E691_DSDXF]) * np.diff(XF_EDGES)
    tp = np.array([v for v, _ in PG.E691_DSDPT2]) * np.diff(PG.E691_PT2_EDGES)
    tx, tp = tx / tx.sum(), tp / tp.sum()
    rec = recoil_masses(page_hash)
    m_d, recoil_d = (
        0.5 * (MASS["D0"] + MASS["D+"]),
        0.5 * (rec["D0"] + rec["D+"]),
    )
    n, _ = x_integrals(MP * MP + 2.0 * MP * e_gamma, m_d, recoil_d)
    F, h = np.ones(tx.size), np.ones(tp.size)
    for _ in range(300):
        w = F[:, None] * h[IPT][None, :] * DPT2[None, :] * n
        mx = w.sum(axis=1) / w.sum()
        F *= tx / mx
        w = F[:, None] * h[IPT][None, :] * DPT2[None, :] * n
        mp = np.bincount(IPT, w.sum(axis=0)) / w.sum()
        h *= tp / mp
    return (
        F,
        h[IPT],
        float(np.max(np.abs(mx - tx))),
        float(np.max(np.abs(mp - tp))),
    )


NUCLEON_PIN = r"\chi_N = \bar z\thinspace \frac{\int_{x_c}^1 g(x)"


def readme_nucleon_measures():
    lhs, rhs = L.sides(L.part(L.display(NUCLEON_PIN), r"\chi_N"))
    assert lhs == r"\chi_N", lhs
    m = re.fullmatch(
        r"\\bar z\\thinspace \\frac\{\\int_\{x_c\}\^1 g\(x\)\\thinspace "
        r"(dx(?:/x)?)\}\{\\int_\{x_c\}\^1 g\(x\)\\thinspace (dx(?:/x)?)\}",
        rhs,
    )
    return tuple({"dx": 0, "dx/x": -1}[g] for g in m.groups())


class Photohadronic:
    def __init__(self):
        r = self.readme = readme_pg()
        w2 = sp.lambdify((m_, er2_), r["w2"], "math")
        self.s_of = lambda eps: w2(MP, eps)
        self.w_of = lambda eps: np.sqrt(w2(MP, np.asarray(eps, float)))
        self.form = sp.lambdify((A_, n_, d_, Wth_, W_), r["sig"], "numpy")
        self.wth = r["wth"]
        self.eps_th = float(
            sp.solve(r["w2"].subs(m_, MASS_Q["p"]) - r["wth_q"] ** 2, er2_)[0]
        )
        self.points = fit_points(lambda E: float(self.w_of(E)))
        self.gp = hpr1r2_numeric()
        frac = fractions_exact()
        self.ff = {h: float(v) for h, v in frac.items()}
        self.anti_d0 = self.ff["D0"] / (self.ff["D0"] + self.ff["D+"])
        h0, e0, e1, asym = r["h_a"]
        self.h_a0, self.e_a0 = h0, e0
        self.h_a1 = float(
            h_from_asymmetry().subs(
                {AS_: sp.Rational(str(asym)), F_: frac["Lc+"]}
            )
        )
        self.p_ha = math.log(h0 / self.h_a1) / math.log(e1 / e0)
        fm = readme_fmnr()
        self.fmnr_w = np.array(fm["W"])
        self.fmnr_r = np.array(fm["resolved"]) / (
            np.array(fm["resolved"]) + np.array(fm["point"])
        )
        self.recoil = recoil_masses(hash(L.text()))
        self._fit = self._steps = None

    def sigma_tot(self, eps):
        """Total gamma p cross section [mubarn], the page's HPR1R2 fit."""
        return self.gp(self.s_of(np.asarray(eps, float)))

    def model(self, p, w):
        """The page's sigma_cc [mubarn] with p = (ln A, n, delta)."""
        w = np.asarray(w, float)
        return np.where(
            w > self.wth,
            self.form(
                np.exp(p[0]), p[1], p[2], self.wth, np.maximum(w, self.wth)
            ),
            0.0,
        )

    def fit(self):
        if self._fit is None:

            def resid(p):
                m = self.model(p, [q[1] for q in self.points])
                return np.array(
                    [
                        (mv - q[2]) / (q[3] if mv > q[2] else q[4])
                        for mv, q in zip(m, self.points)
                    ]
                )

            f = least_squares(resid, [0.0, 1.5, 1.0])
            self._fit = (f.x, float(np.sum(f.fun**2)), len(self.points) - 3)
        return self._fit

    def sigma_cc(self, w):
        """sigma(gamma p -> c cbar X) [mubarn]."""
        p = self.fit()[0]
        curve = CURVES[self.readme["curve"]]
        w_j, dec = self.readme["wj"], self.readme["decade"]
        w = np.asarray(w, float)
        out = self.model(p, w)
        hi = w > w_j
        if np.any(hi):
            norm = math.log(float(self.model(p, w_j))) - float(
                gmm_log(curve, w_j, dec)
            )
            out = np.where(
                hi,
                np.exp(norm + gmm_log(curve, np.where(hi, w, w_j), dec)),
                out,
            )
        return out

    def h_assoc(self, eps):
        return np.minimum(
            self.h_a0 * (np.asarray(eps, float) / self.e_a0) ** -self.p_ha, 1.0
        )

    def resolved_share(self, w):
        lw, lx = np.log(np.asarray(w, float)), np.log(self.fmnr_w)
        slope = (self.fmnr_r[1] - self.fmnr_r[0]) / (lx[1] - lx[0])
        return np.where(
            lw < lx[0],
            np.clip(self.fmnr_r[0] + slope * (lw - lx[0]), 0.0, None),
            np.interp(lw, lx, self.fmnr_r),
        )

    def resolved_zero(self):
        lx = np.log(self.fmnr_w)
        return math.exp(
            lx[0]
            - self.fmnr_r[0]
            * (lx[1] - lx[0])
            / (self.fmnr_r[1] - self.fmnr_r[0])
        )

    def h_nucleon(self, eps):
        return 0.5 * self.resolved_share(self.w_of(eps))

    def chi_photon(self, eps, species):
        """Mean chi of a photon-side hadron (E691's invariant cross section,
        Feynman scaling). The mean is NaN where the kinematics allow
        no hadron."""
        eps = np.atleast_1d(np.asarray(eps, float))
        F, H, _, _ = e691_invariant(self.readme["e691"], hash(L.text()))
        wgt = F[:, None] * H[None, :] * DPT2[None, :]
        m, m_rec = MASS[HADRON_OF[species]], self.recoil[species]
        out = np.full(eps.size, np.nan)
        for i, e in enumerate(eps):
            if self.w_of(e) <= m + m_rec:
                continue
            n, chi_n = x_integrals(float(self.s_of(e)), m, m_rec)
            if np.sum(wgt * n) > 0:
                out[i] = np.sum(wgt * chi_n) / np.sum(wgt * n)
        return out

    def step(self, lo, hi, n=400, nchi=40):
        e = log_midpoints(lo, hi, n)
        r = np.where(
            e > self.eps_th,
            self.sigma_cc(self.w_of(e)) / self.sigma_tot(e),
            0.0,
        )
        a, nn = self.h_assoc(e), self.h_nucleon(e)
        g = 1.0 - a - nn
        res = dict(R=r.mean(), Ra=(r * a).mean(), RN=(r * nn).mean())
        res["h_a"], res["h_N"] = res["Ra"] / res["R"], res["RN"] / res["R"]
        ec = log_midpoints(lo, hi, nchi)

        def mean_chi(weight, species):
            c = np.interp(np.log(e), np.log(ec), self.chi_photon(ec, species))
            ok = np.isfinite(c) & (weight > 0)
            return (
                float(np.sum((weight * c)[ok]) / np.sum(weight[ok]))
                if ok.any()
                else np.nan
            )

        res["chi_g"] = {s: mean_chi(r * g, s) for s in SPECIES}
        res["chi_a"] = {s: mean_chi(r * a, s) for s in ("anti-D0", "D-")}
        if (r * nn).sum() > 0:
            cn = np.interp(
                np.log(e),
                np.log(ec),
                [pp_model().projectile_mean_x(x) for x in ec],
            )
            res["chi_N"] = float(np.sum(r * nn * cn) / np.sum(r * nn))
        return res

    def row(self, st):
        def avg(*names):
            w = [
                (self.ff[HADRON_OF[s]], st["chi_g"][s])
                for s in names
                if np.isfinite(st["chi_g"][s])
            ]
            return (
                sum(a * c for a, c in w) / sum(a for a, _ in w) if w else None
            )

        chi_ad = (
            (
                self.anti_d0 * st["chi_a"]["anti-D0"]
                + (1 - self.anti_d0) * st["chi_a"]["D-"]
            )
            if st["Ra"] > 0
            else None
        )
        return dict(
            R=st["R"],
            h_a=st["h_a"],
            h_N=st["h_N"],
            chi_D=avg("D0", "D+", "anti-D0", "D-"),
            chi_Ds=avg("Ds+", "Ds-"),
            chi_Lc=avg("Lc+", "anti-Lc-"),
            chi_aD=chi_ad,
            chi_N=st.get("chi_N"),
        )

    def steps(self):
        if self._steps is None:
            self._steps = [
                (
                    t,
                    number(lo),
                    number(hi),
                    self.row(self.step(number(lo), number(hi))),
                )
                for t, lo, hi, _ in step_rows()
            ]
        return self._steps


AS_, HA_, F_ = sp.symbols("A_s h_a f", positive=True)


def readme_asymmetry():
    lhs, by_n, by_h = L.sides(L.part(L.display(ASSOC_PIN), r"\mathcal{A}"))
    assert lhs == r"\mathcal{A}", lhs
    return L.to_sympy(by_h, {"h_a": HA_, r"f_{\Lambda_c}": F_})


def readme_asymmetry_by_n():
    _, by_n, _ = L.sides(L.part(L.display(ASSOC_PIN), r"\mathcal{A}"))
    tok = {"h_a": HA_, r"f_{\Lambda_c}": F_}
    n_plus = L.to_sympy(
        L.sides(L.inline(r"N_+ = h_a + (1 - h_a)f_{\Lambda_c}"))[1], tok
    )
    n_minus = L.to_sympy(
        L.sides(L.inline(r"N_- = (1 - h_a)f_{\Lambda_c}"))[1], tok
    )
    return L.to_sympy(by_n, {"N_+": n_plus, "N_-": n_minus})


def readme_h_inverse():
    lhs, rhs = L.sides(L.part(L.display(ASSOC_PIN), "h_a ="))
    assert lhs == "h_a", lhs
    return L.to_sympy(rhs, {r"\mathcal{A}": AS_, r"f_{\Lambda_c}": F_})


def h_from_asymmetry():
    sol = sp.solve(sp.Eq(AS_, readme_asymmetry()), HA_)
    assert len(sol) == 1, sol
    return sol[0]


def readme_fmnr():
    head = [c.strip() for c in L.line(FMNR_TABLE).strip().strip("|").split("|")]
    rows = {r[0]: r[1:] for r in L.table(FMNR_TABLE)}
    return dict(
        W=[float(v) for v in head[1:]],
        point=[float(v) for v in rows["Point-like [μb]"]],
        resolved=[float(v) for v in rows["Resolved [μb]"]],
        share=rows["Resolved share"],
    )


@lru_cache(None)
def _model(cls, page_hash):
    return cls()


def pg_model():
    """The photo-hadronic model of the page as it reads now (a patched page
    gives its own model)."""
    return _model(Photohadronic, hash(L.text()))


def pp_model():
    return _model(PPCharm, hash(L.text()))


def compare_columns(*columns):
    out = []
    for (t, lo, hi, cells), (_, _, _, got) in zip(
        step_rows(), pg_model().steps()
    ):
        for k in columns:
            want, have = cells[k], got[k]
            if want is None:
                ok, detail = (
                    have is None,
                    "—" if have is None else f"{have:.4g}, the page has —",
                )
            else:
                ok, detail = (
                    have is not None and rounds_as(have, want),
                    "none" if have is None else f"{have:.5g}",
                )
            out.append(
                Result(
                    f"M{t + 1}, {lo}-{hi} GeV: {k} = {want or '—'}", ok, detail
                )
            )
    return out


@derivation("cp.response")
def check_response():
    """f_r is Hümmer et al.'s eq. 27 (y of eq. 8) for the step's constant cross
    section: the page's closed form."""
    L.inline(r"2y \ge \epsilon_{\rm min}")
    (pre, lo, up, body), _ = readme_response()
    pre27 = L.to_sympy(piece(H_EQ27, r"\frac{1}{2y^2}"), {"y": yy})
    up27 = L.to_sympy(piece(H_EQ27, "^{2y}")[2:-1], {"y": yy})
    body27 = L.to_sympy(
        piece(H_EQ27, r"\epsilon_r \, \sigma^{\mathrm{IT}}(\epsilon_r)"),
        {r"\epsilon_r": er_},
        {r"\sigma^{\mathrm{IT}}": SIG},
    )
    eq8 = L.to_sympy(
        piece(H_EQ8, r"\frac{E_p \, \varepsilon}{m_p}"),
        {"E_p": EN_, r"\varepsilon": eps_, "m_p": MC2},
    )
    mu = sp.Symbol("mu", real=True)
    epsr = L.to_sympy(
        H_EPSR.strip("$").split("=")[1],
        {"E_p": EN_, r"\varepsilon": eps_, "m_p": MC2, r"\cos_{p \gamma}": mu},
    )
    lhs_r, rhs_r = L.sides(L.inline(r"\epsilon_r = 2y"))
    top = L.to_sympy(rhs_r, {"y": readme_y()})
    out = [
        in_source(
            "eqs. 8 and 27 and eps_r after eq. 3 as transcribed",
            HUEMMER,
            H_EQ8,
            H_EQ27,
            H_EPSR,
        ),
        exact(
            "y = E_N eps/(m_p c^2): eq. 8 (E_p -> E_N, m_p -> m_p c^2)",
            readme_y() - eq8,
        ),
        exact(
            "eps_r = E_N eps (1 - cos theta)/(m_p c^2) reaches 2y head-on "
            "(cos theta = -1)",
            epsr.subs(mu, -1) - top,
        ),
        exact("the integral form's prefactor 1/(2y^2) = eq. 27's", pre - pre27),
        exact(
            "its upper limit 2y = eq. 27's (its lower limit 0 for eps_th: "
            "sigma_r = 0 below eps_min)",
            up - up27,
        ),
        exact("its integrand eps_r sigma_r(eps_r) = eq. 27's", body - body27),
    ]
    d, D = sp.symbols("d D", positive=True)
    integral = step_integral(pre, lo, up, body, [(ST, E1, E2)])
    for name, which, at in (
        ("2y < eps_min", None, {E1: 2 * yy + d, E2: 2 * yy + d + D}),
        ("eps_min <= 2y < eps_max", 0, {yy: (E1 + d) / 2, E2: E1 + d + D}),
        ("2y >= eps_max", 1, {yy: (E1 + d + D) / 2, E2: E1 + d}),
    ):

        def branch(*args, which=which, at=at):
            if which is not None:
                assert sp.expand(
                    (args[1 - which] - args[which]).subs(at)
                ).is_positive
            return args[which or 0]

        closed, value = readme_response(branch)[1].subs(at), integral.subs(at)
        if which is None:
            neg = sp.factor(closed)
            out.append(
                Result(
                    f"{name}: eq. 27 gives 0, the closed form is "
                    "negative there (hence 'for 2y >= eps_min')",
                    value == 0 and neg.is_negative,
                    f"integral {value}, closed form {neg}",
                )
            )
        else:
            out.append(
                exact(
                    f"{name}: eq. 27 = the closed form with min = "
                    f"{['4y^2', 'eps_max^2'][which]}",
                    value - closed,
                )
            )
    at = {yy: (E1 + d) / 2, E2: E1 + d + D}
    with L.patched(r"\frac{\sigma_t}{4y^2}", r"\frac{\sigma_t}{2y^2}"):
        broken = readme_response(lambda a, b: a)[1].subs(at)
    out.append(
        exact_fails(
            "control: sigma_t/(2y^2) in the closed form",
            integral.subs(at) - broken,
        )
    )
    return out


@derivation("cp.source")
def check_source():
    """t_r^-1 is Hümmer et al.'s eq. 4 and Q_h their eq. 28, the step's part
    k as their interaction type."""
    head, body = readme_rate(EN_)
    ours = readme_source_term()
    out = [
        in_source(
            "eqs. 4, 27 and 28 as transcribed", HUEMMER, H_EQ4, H_EQ27, H_EQ28
        ),
        exact(
            "t_r^-1 = eq. 4 at eps = m_p c^2 y/E_N (times c): "
            "the integrands in y, with f_r = eq. 27's",
            head * body - eq4_in_y(EN_),
        ),
        exact(
            "(M/chi) [N_N t_r^-1] at E_N = E/chi = c x eq. 28 "
            "(N_p -> N_N; the integrands in y)",
            ours - eq28(),
        ),
    ]
    with L.patched(
        r"\frac{m_p c^2}{E_N}\thinspace n_\gamma",
        r"\frac{E_N}{m_p c^2}\thinspace n_\gamma",
    ):
        h_b, b_b = readme_rate(EN_)
        out.append(
            exact_fails(
                "control: E_N/(m_p c^2) for m_p c^2/E_N in front of n_gamma",
                h_b * b_b - eq4_in_y(EN_),
            )
        )
    with L.patched(r"E_N = E/\chi_k^h}.", r"E_N = E\chi_k^h}."):
        out.append(
            exact_fails(
                "control: E_N = E chi in the source",
                readme_source_term() - eq28(),
            )
        )
    return out


@derivation("cp.grid")
def check_grid():
    """Per unit ln E, the factor 1/chi cancels, and the page's split share keeps
    the number and the energy. The weight that the page gives for AM3 is its
    number_split, which keeps the number, and the energy to first order."""
    f, Dl = sp.symbols("f Delta", positive=True)
    tok = {"f": f, r"\Delta": Dl}
    w = L.to_sympy(L.sides(L.inline(W_SPLIT))[1], tok)
    w_am3 = L.to_sympy(L.inline(W_AM3), tok)
    # AM3's number_split as transcribed
    am3 = (lambda r: r + sp.Rational(1, 2) * r * (1 - r) * Dl)(f)

    def excess(v):
        return (1 - v) + v * sp.exp(Dl) - sp.exp(f * Dl)

    series = sp.series(excess(am3), Dl, 0, 3).removeO()
    out = [
        exact(
            "E Q_h(E) = M [E_N N_N t_r^-1] at E_N = E/chi: E times the "
            "source's term (any N_N, t_r^-1)",
            E_ * readme_source_terms_as_functions() - readme_grid_term(),
        ),
        exact(
            "the shares 1 - w and w keep the number and the energy: (1 - w) "
            "+ w e^Delta = e^(f Delta)",
            excess(w),
        ),
        Result(
            "w -> f for Delta -> 0: the linear interpolation in ln E",
            sp.limit(w, Dl, 0) == f,
            "limit f",
        ),
        in_am3("AM3's number_split as transcribed", *AM3_SPLIT),
        exact(
            "the page's weight of AM3 = its number_split, f + f(1 - f) dlnE/2",
            w_am3 - am3,
        ),
        Result(
            "AM3's weights keep the number; their energy (1 - w) + w e^Delta "
            "= e^(f Delta) + O(Delta^2)",
            sp.expand(series.coeff(Dl, 1)) == 0
            and sp.expand(series.coeff(Dl, 2)) != 0,
            f"error {sp.factor(series)}",
        ),
    ]
    with L.patched(W_SPLIT, "w = " + W_AM3):
        broken = L.to_sympy(L.sides(L.inline("w = " + W_AM3))[1], tok)
        out.append(
            exact_fails(
                "control: AM3's weight as the page's share", excess(broken)
            )
        )
    with L.patched("$" + W_AM3 + "$", r"$f - f(1 - f)\Delta/2$"):
        bad = L.to_sympy(L.inline(r"f - f(1 - f)\Delta/2"), tok)
        out.append(
            exact_fails(
                "control: f - f(1 - f) Delta/2 as AM3's weight", bad - am3
            )
        )
    return out


@derivation("cp.R")
def check_R():
    """R = sigma_cc/sigma_gp at W^2 = m_p^2 + 2 m_p eps_r, averaged over the
    steps: the page's R column."""
    r_part, sig, w2 = readme_R()
    q = sp.Symbol("q", positive=True)
    p_gamma, p_nucleon = (er2_, 0, 0, er2_), (m_, 0, 0, 0)
    s_kin = sum(
        sgn * (a + b) ** 2
        for sgn, a, b in zip((1, -1, -1, -1), p_gamma, p_nucleon)
    )
    out = [
        Result(
            "R(eps_r) = sigma_cc(W)/sigma_gp(W): c cbar pairs per gamma "
            "p interaction",
            r_part == R_PIN,
            "as written (asserted)",
        ),
        exact(
            "W^2 = s = (p_gamma + p_N)^2 of a photon eps_r on a "
            "nucleon at rest",
            w2 - s_kin,
        ),
        exact(
            "sigma_cc vanishes at W = W_th (n > 0)",
            sig.subs(W_, Wth_).subs(n_, 1 + q),
        ),
        *compare_columns("R"),
    ]
    with L.patched(
        r"W^2 = m_p^2 + 2m_p\epsilon_r", r"W^2 = m_p^2 + m_p\epsilon_r"
    ):
        broken = readme_R()[2]
        out.append(
            exact_fails(
                "control: W^2 = m_p^2 + m_p eps_r in the page", broken - s_kin
            )
        )
        out.append(
            must_fail("control: R column with that W^2", compare_columns("R"))
        )
    return out


@derivation("cp.threshold")
def check_threshold():
    """eps_r = 8.71 GeV: the threshold of gamma p -> Lambda_c+ anti-D0, the
    lightest charmed final state."""

    def results():
        r, m = readme_pg(), MASS_Q
        eps_r = lambda w: sp.solve(r["w2"].subs(m_, m["p"]) - w**2, er2_)[0]
        eps = eps_r(m["Lc+"] + m["D0"])
        others = {
            "Lambda_c+ D- pi+": m["Lc+"] + m["D+"] + m["pi+"],
            "D0 anti-D0 p": 2 * m["D0"] + m["p"],
            "D+ D- p": 2 * m["D+"] + m["p"],
            "Ds+ Ds- p": 2 * m["Ds+"] + m["p"],
            "Lambda_c+ anti-Lambda_c- p": 2 * m["Lc+"] + m["p"],
        }
        lowest = all(m["Lc+"] + m["D0"] < v for v in others.values())
        steps = readme_charm_steps()
        return [
            Result(
                f"threshold eps_r = {r['threshold']} GeV "
                "(the page's W^2 and charm masses)",
                rounds_as(float(eps), r["threshold"]),
                f"{float(eps):.5f} GeV",
            ),
            Result(
                f"W_th = m_Lambda_c + m_D0 = {r['wth_text']} GeV",
                r["wth_q"] == m["Lc+"] + m["D0"]
                and rounds_as(r["wth"], r["wth_text"]),
                f"{r['wth']:.5f} GeV",
            ),
            Result(
                "the lightest final state with charm: lighter than "
                + ", ".join(others),
                lowest,
                "; ".join(
                    f"{k} {float(eps_r(v)):.2f} GeV" for k, v in others.items()
                ),
            ),
            Result(
                "the first step starts there",
                steps[0]["eps_min"] == float(r["threshold"]),
                f"{steps[0]['eps_min']} GeV",
            ),
        ]

    eq5 = L.to_sympy(H_EQ5.split("=")[1], {"m_p": m_, r"\epsilon_r": er2_})
    out = [
        in_source("eq. 5 as transcribed", HUEMMER, H_EQ5),
        exact("W^2 = s(eps_r) of eq. 5", readme_R()[2] - eq5),
        *results(),
    ]
    with L.patched(
        r"charm is made above $\epsilon_r$ = 8.71 GeV",
        r"charm is made above $\epsilon_r$ = 8.72 GeV",
    ):
        out.append(must_fail("control: 8.72 GeV in the page", results()))
    return out


PDG16_SNIPPETS = (
    "HPR1R2 | M=2:1206§0:0094[GeV]H=0:2720§0:0024[mb]",
    "´1=0:4473§0:0077´2=0:5486§0:0049",
    r"±=\(3:063§0:014\)£10¡3",
    r"Npt=1048 | Â2=nptbyGroups | 34:41§0:13 | 13:07§0:17 | 7:394§0:081 "
    r"| ¹p\(p\)=p",
    "0:0139§0:0011 | °=p",
    "¾a¨b=Hlog2Ãs | sabM!+Pab+Rab1Ãs | sabM!¡´1§Rab2Ãs | sabM!¡´2",
    r"sabM=\(ma+mb+M\)2areinGeV2;ma,mb,\(m°¤=m½\(770\)\)",
    r'¼Hlog2Ãs | s°\(p;d\)M!+P°\(p;d\)\)±"¼\(1;¸\)Hlog2Ãs | '
    r"s°\(p;d\)M!+Pp\(p;d\)#",
)


def pdg16_values():
    def val(s):
        return float(s.replace(":", "."))

    sn = PDG16_SNIPPETS
    return dict(
        M=val(re.search(r"M=([\d:]+)§", sn[0]).group(1)),
        H=val(re.search(r"H=([\d:]+)§", sn[0]).group(1)),
        eta1=val(re.match(r"´1=([\d:]+)§", sn[1]).group(1)),
        delta=val(re.search(r"\(([\d:]+)§", sn[2]).group(1)) * 1e-3,
        P_pp=val(
            re.search(
                r"\| ([\d:]+)§[\d:]+ \| [\d:]+§[\d:]+ \| "
                r"[\d:]+§[\d:]+ \| ¹p",
                sn[3],
            ).group(1)
        ),
        R1_gp=val(re.match(r"([\d:]+)§", sn[4]).group(1)),
    )


GP_FINDING = (
    r"The PDG 2016 parameters reproduce the curve that the PDG 2026 "
    r"web page serves within "
)


@derivation("cp.sigma_gp")
def check_sigma_gp():
    """sigma_gp: the review's HPR1R2 form with its gamma p factorization and
    parameters, which reproduce PDG 2026's."""
    ours, sm = readme_hpr1r2()
    theirs, sm_pdg = pdg16_gamma_p()
    stated, page = pdg16_values(), readme_gp_params()
    s = np.array([rs**2 for rs, _ in PG.PDG_CURVE_2026])
    data = np.array([v for _, v in PG.PDG_CURVE_2026])
    deviation = lambda p=None: float(
        np.max(np.abs(hpr1r2_numeric(p)(s) / data - 1))
    )
    dev = deviation()
    found = search(
        r"within ([\d.]+)%, from \$\\sqrt\{s\}\$ = (\d+) GeV to "
        r"(\d+) TeV\.",
        GP_FINDING,
    )
    pages = [pdf_page(PDG16, PDG16_SNIPPETS[k], pdg16_page) for k in (5, 7, 0)]
    out = [
        in_pdf(
            "the review's form, s_M = (m_a + m_b + M)^2 with m_rho(770) "
            "for the photon, P^gamma p -> delta P^pp, and its table "
            "(M, H, eta_1, delta, P of pbar(p)/p, R_1 of gamma/p)",
            PDG16,
            *PDG16_SNIPPETS,
        ),
        Result(
            "the form and the factorisation on p. 8, the table on p. 9",
            pages in ([None] * 3, [8, 8, 9]),
            "not re-checked" if pages[0] is None else f"pages {pages}",
        ),
        exact(
            "the page's sigma_gp = the review's form for gamma "
            "p (log = ln; no R_2 for gamma/p)",
            ours - theirs,
        ),
        exact(
            "s_M = (m_rho + m_p + M)^2: the review's, a = gamma (m_rho), b = p",
            sm - sm_pdg,
        ),
        Result(
            "the page's parameters = the review's table",
            all(
                math.isclose(page[k], v, rel_tol=1e-12)
                for k, v in stated.items()
            ),
            ", ".join(f"{k} {page[k]:g}" for k in stated),
        ),
        Result(
            "tests/support/measured/charm_photoproduction.py has these "
            "parameters and m_rho (PDG_FIT)",
            all(
                math.isclose(PG.PDG_FIT[k], page[k], rel_tol=1e-12)
                for k in (*stated, "m_rho")
            ),
            f"m_rho {PG.PDG_FIT['m_rho']} GeV",
        ),
        Result(
            "they reproduce the gamma p curve PDG 2026 serves within "
            f"{found[0]} % (a pixel: 0.34 %), sqrt(s) = {found[1]} "
            f"GeV-{found[2]} TeV",
            dev < 0.0034
            and rounds_as(100 * dev, found[0])
            and round(s[0] ** 0.5) == float(found[1])
            and round(s[-1] ** 0.5 / 1e3) == float(found[2]),
            f"max deviation {100 * dev:.3f} % at {len(s)} points, "
            f"{s[0] ** 0.5:.0f}-{s[-1] ** 0.5:.0f} GeV",
        ),
    ]
    with L.patched(r"$H$ = 0.2720 mb", r"$H$ = 0.2820 mb"):
        out.append(
            Result(
                "control: H = 0.2820 for 0.2720 in the page",
                deviation() > 0.0033,
                f"deviates by {100 * deviation():.1f} % (must fail)",
            )
        )
    with L.patched(
        r"\delta_\gamma\Big[H\ln^2(s/s_M) + P^{\rm pp}\Big]",
        r"\delta_\gamma H\ln^2(s/s_M) + P^{\rm pp}",
    ):
        out.append(
            exact_fails(
                "control: delta_gamma on the H term only",
                readme_hpr1r2()[0] - theirs,
            )
        )
    return out


HERA_KEYS = ("ZEUS163", "ZEUS243", "H1tag200", "H1untag142", "H1untag230")
ZEUS_TEX = (
    r"\mbox{$(6.3 \pm 2.2^{+6.3}_{-3.0}) \, \mu {\rm b} $} at "
    r"$\langle W \rangle = 163$ \GeV\ and "
    r"\mbox{$(16.9 \pm 5.2^{+13.9}_{-8.5}) \, \mu {\rm b} $} at "
    r"$\langle W \rangle = 243$ \GeV."
)
H1_TEX = (
    r"$\langle W_{\gamma p}\rangle \approx 200$\ GeV and $Q^2 < 0.01\,\gev^2$, "
    r"the result is $\sigma(\gamma p \rightarrow c \bar{c} X) = (13.2 \pm 2.2 "
    r"^{+2.1}_{-1.7}\, ^{+9.9}_{-4.8})\,\mu b$.",
    r"$\sigma(\gamma p \rightarrow c \bar{c} X) = ( 9.3 \pm 2.1  "
    r"^{+1.9}_{-1.8}\,  ^{+6.9}_{-3.2} )\,\mu b$ at $\langle "
    r"W_{\gamma p}\rangle \approx 142$\,GeV and $\sigma(\gamma p \rightarrow c "
    r"\bar{c} X) = ( 20.6 \pm 5.5 ^{+4.3}_{-3.9}\, ^{+15.4}_{-7.2})\,\mu "
    r"b$ at $\langle W_{\gamma p}\rangle \approx 230$\,GeV",
)


def hera_from_tex():
    num = r"(\d+\.?\d*)"
    z = re.findall(
        rf"\(\s*{num} \\pm {num}\^\{{\+{num}\}}_\{{-{num}\}}\) \\, "
        r"\\mu {\\rm b} \$} at \$\\langle W \\rangle = (\d+)\$",
        ZEUS_TEX,
    )
    h_pat = (
        rf"\(\s*{num} "
        rf"\\pm {num}\s*\^\{{\+{num}\}}_\{{-{num}\}}\\,\s*\^\{{\+{num}\}}_"
        rf"\{{-{num}\}}\s*\)"
    )
    tagged = re.search(h_pat, H1_TEX[0]).groups() + ("200",)
    untagged = [
        m.groups() + (w,)
        for m, w in zip(re.finditer(h_pat, H1_TEX[1]), ("142", "230"))
    ]
    out = {
        f"ZEUS{w}": (
            float(w),
            float(s),
            float(st),
            float(sp_),
            float(sm),
            0.0,
            0.0,
        )
        for s, st, sp_, sm, w in z
    }
    for key, g in zip(
        ("H1tag200", "H1untag142", "H1untag230"), [tagged] + untagged
    ):
        out[key] = (float(g[6]), *map(float, g[:6]))
    return out


def transcriptions(alpha_e691):
    pts = {p["key"]: p for p in PG.SIGMA_CC}
    out = []

    def record(what, ins, compare):
        rec = L.hepdata(ins)
        out.append(
            absent(what, f"HEPData-{ins}", L.HEPDATA)
            if rec is None
            else Result(what, *compare(rec))
        )

    def num(v):
        return float(v.replace("%", ""))

    def errors(p):
        """sigma and its errors, signed as the records give them."""
        return (
            p["sigma"],
            p["stat"][0],
            -p["stat"][1],
            p["syst"][0],
            -p["syst"][1],
        )

    def slac(rec):
        row, p = rec["Table1"][0][2][0], pts["SLAC86"]
        return (
            close(
                [num(v) for v in row], [p["E"], *(1e3 * v for v in errors(p))]
            ),
            "Table 1, first block [nb]",
        )

    def pec(rec):
        row, p = rec["Table1"][0][2][0], pts["PEC"]
        ok = (
            close([num(v) for v in row[:3]], [p["E"], *p["E_range"]])
            and close([num(row[3]), num(row[4])], [250.0, 57.0])
            and (p["sigma"], p["stat"][0]) == (0.230, 0.057)
        )
        return (
            ok,
            "the record has 250 +- 57 nb, the module the abstract's "
            "230 +- 57 (as Goncalves et al. plot it)",
        )

    def na14(rec):
        row, p = rec["Table3"][0][2][0], pts["NA14/2"]
        return close(
            [num(v) for v in row[:3]] + [num(v) / 1e3 for v in row[3:]],
            [p["E"], *p["E_range"], *errors(p)],
        ), "Table 3"

    def e691(rec):
        row, p = rec["Table9"][0][2][0], pts["E691"]
        return (
            close([num(v) for v in row], [alpha_e691, *errors(p)]),
            f"Table 9, the row of alpha = {alpha_e691} (the E691 choice, "
            "the page's power)",
        )

    def zeus(rec):
        rows = {num(r[0]): r for r in rec["Table3"][0][2]}
        return (
            all(
                close(
                    [num(v) for v in rows[pts[k]["W"]][:3]],
                    [pts[k]["W"], *pts[k]["W_range"]],
                )
                for k in ("ZEUS163", "ZEUS243")
            ),
            "Table 3: the W bins (115-205, 205-275 GeV)",
        )

    def h1(rec):
        bins = [
            tuple(num(v) for v in r[1:3])
            for t in ("Table3", "Table4")
            for r in rec[t][0][2]
        ]
        return all(
            tuple(pts[k]["W_range"]) in bins
            for k in ("H1tag200", "H1untag142", "H1untag230")
        ), "Tables 3-4: the W bins"

    record(
        "SLAC HFP 1986: E, sigma, errors (HEPData ins214735 Table 1)",
        "ins214735",
        slac,
    )
    record(
        "PEC: the record's value and the abstract's "
        "(HEPData ins230940 Table 1)",
        "ins230940",
        pec,
    )
    record("NA14/2 (HEPData ins333271 Table 3)", "ins333271", na14)
    record(
        f"E691 per nucleon with A^{alpha_e691} (HEPData ins289318 Table 9)",
        "ins289318",
        e691,
    )
    record("ZEUS W bins (HEPData ins392566 Table 3)", "ins392566", zeus)
    record("H1 W bins (HEPData ins417960 Tables 3-4)", "ins417960", h1)
    tex = hera_from_tex()
    same = all(
        close(
            tex[k],
            (
                pts[k]["W"],
                pts[k]["sigma"],
                pts[k]["stat"][0],
                pts[k]["syst"][0],
                pts[k]["syst"][1],
                *pts[k].get("pdf", (0.0, 0.0)),
            ),
        )
        for k in HERA_KEYS
    )
    out += [
        in_source("ZEUS's two cross sections as transcribed", ZEUS95, ZEUS_TEX),
        in_source("H1's three cross sections as transcribed", H1_96, *H1_TEX),
        Result(
            "the module's HERA values, errors and <W> = these",
            same,
            ", ".join(HERA_KEYS),
        ),
    ]
    return out


def page_order():
    """The fitted measurements in the page's order."""
    fixed = [
        (p["key"], ("epsilon_r", p["E"]), p["target"], p)
        for p in PG.SIGMA_CC
        if "E" in p
    ]
    e687 = [
        (f"E687_{E:g}", ("epsilon_r", E), "Be", None)
        for E, *_ in PG.E687_FN555["dstar"]
    ]
    hera = [
        (p["key"], ("W", p["W"]), p["target"], p)
        for p in PG.SIGMA_CC
        if "W" in p
    ]
    return fixed + e687 + hera


def data_ok(cell, p):
    """A HEPData table belongs to the measurement's record, and the value is the
    record's (not PEC's abstract). A table or the abstract belongs to the paper
    that the measurement's reference names."""
    # E687: FN-555, Table 1 (the docstring of the module's E687_FN555)
    if p is None:
        return cell == "Table 1"
    if cell.startswith("HEPData, "):
        return (
            bool(p.get("hepdata"))
            and cell == "HEPData, " + p["hepdata"][1]
            and "the HEPData record has" not in p.get("note", "")
        )
    if cell == "abstract":
        return "abstract" in p["ref"]
    return re.fullmatch(r"Table \d+", cell) is not None and cell in p["ref"]


def measurement_rows():
    pts = {q[0]: q for q in pg_model().points}
    rows = L.table(MEAS_TABLE)
    out = [
        Result(
            "the table has the fitted measurements",
            len(rows) == len(pts),
            f"{len(rows)} rows, {len(pts)} points",
        )
    ]
    for cells, (key, (kind, energy), target, p) in zip(rows, page_order()):
        _, _, sig, up, down = pts[key]
        m = re.fullmatch(r"\$(\\epsilon_r|W) = ([\d.]+)\$", cells[2])
        ups, downs = (v.strip() for v in cells[4].split(","))
        ok = (
            m.group(1).strip("\\") == kind
            and float(m.group(2)) == energy
            and cells[1] == target
            and rounds_as(sig, cells[3])
            and rounds_as(up, ups)
            and rounds_as(down, downs)
            and data_ok(cells[5], p)
        )
        out.append(
            Result(
                f"{key}: {cells[2]}, {cells[3]} (+{ups}, -{downs}) "
                f"mubarn, {cells[1]}, {cells[5]}",
                ok,
                f"{sig:.4g} (+{up:.4g}, -{down:.4g})",
            )
        )
    return out


E687_LINE = r"FN-555 (pp. 15–16) converts this to charm pairs per nucleon"
E691_POWER = r"E691 assume a cross section that grows as the power "
FIT_FINDING = r"refits $A$, $n$ and $\delta$, with $\chi^2 = "


@derivation("cp.sigma_cc")
def check_sigma_cc():
    """sigma_cc: the page's form fitted to the measurements of its table, from
    20 GeV to HERA, gives its A, n, delta."""
    alpha = float(
        search(
            r"grows as the power ([\d.]+) of the nucleon number", E691_POWER
        )[0]
    )

    def results():
        m = pg_model()
        r = m.readme
        p, chi2, ndf = m.fit()
        got = (math.exp(p[0]), p[1], p[2])
        lowest = min(float(q["E"]) for q in PG.SIGMA_CC if "E" in q)
        chi2_text, ndf_text = search(
            r"with \$\\chi\^2 = ([\d.]+)\$ for (\d+) "
            r"degrees of freedom\.",
            FIT_FINDING,
        )
        return [
            Result(
                f"A, n, delta = {', '.join(r['fit_texts'])} "
                f"(A in mubarn; {ndf + 3} points, asymmetric errors)",
                all(rounds_as(g, t) for g, t in zip(got, r["fit_texts"])),
                ", ".join(f"{g:.5g}" for g in got)
                + f"; chi^2/ndf = {chi2:.2f}/{ndf}",
            ),
            Result(
                f"chi^2 = {chi2_text} for {ndf_text} degrees of freedom "
                "(the page)",
                rounds_as(chi2, chi2_text) and ndf == int(ndf_text),
                f"{chi2:.3f}/{ndf}",
            ),
            Result(
                f"the measurements reach from {r['e_low']:g} GeV "
                "(SLAC HFP) to HERA",
                lowest == r["e_low"] and max(q[1] for q in m.points) > 100,
                f"lowest E {lowest:g} GeV",
            ),
            *measurement_rows(),
        ]

    e = PG.E687_FN555
    conv = search(
        r"dividing by ([\d.]+) \$\\mathrm\{D\}\^\{\\ast\\pm\}\$ per "
        r"pair and by \$([\d.]+)\^\{([\d.]+)\}\$, with a further "
        r"error of (\d+)%\.",
        E687_LINE,
    )
    out = (
        results()
        + [
            Result(
                "E687's conversion (FN-555, pp. 15-16): the page's 0.604 "
                "D*+- per pair, 9.01^0.93 and 5 % = the module's",
                all(
                    math.isclose(float(v), w)
                    for v, w in zip(
                        conv,
                        (
                            e["dstar_per_cc"],
                            e["a_be"],
                            e["alpha"],
                            100 * e["extra_syst"],
                        ),
                    )
                ),
                ", ".join(conv),
            )
        ]
        + transcriptions(alpha)
    )
    with L.patched(r"$A$ = 0.590 μb", r"$A$ = 0.600 μb"):
        out.append(must_fail("control: A = 0.600 in the page", results()))
    with L.patched(r"| 0.062 | 0.017, 0.013 |", r"| 0.063 | 0.017, 0.013 |"):
        out.append(
            must_fail(
                "control: 0.063 for SLAC's 0.062 in the table",
                measurement_rows(),
            )
        )
    return out


IPSAT_FINDING = r"as published, IP-SAT describes the HERA points best"


@derivation("cp.ipsat")
def check_ipsat():
    """Above W = 275 GeV, where the HERA data end, sigma_cc follows the IP-SAT
    curve of Goncalves et al., scaled there."""

    def results():
        m = pg_model()
        r = m.readme
        top = max(q["W_range"][1] for q in PG.SIGMA_CC if "W_range" in q)
        below, above = (
            float(m.sigma_cc(r["wj"] * f)) for f in (1 - 1e-9, 1 + 1e-9)
        )
        return [
            Result(
                f"W_J = {r['wj']:g} GeV: the upper edge of the HERA bins",
                r["wj"] == top,
                f"{top:g} GeV (ZEUS)",
            ),
            Result(
                "scaled to the fit at W_J: continuous there",
                r["w_norm"] == r["wj"] and abs(above / below - 1) < 1e-6,
                f"{below:.5g}, {above:.5g} mubarn",
            ),
            Result(
                f"the {r['curve']} curve of the page",
                r["curve"] in CURVES,
                ", ".join(CURVES),
            ),
            Result(
                f"the figure ends at {r['w_end']:g} GeV",
                all(c[-1][0] == r["w_end"] for c in CURVES.values()),
                f"{CURVES[r['curve']][-1][0]:g} GeV",
            ),
        ]

    m = pg_model()

    def chi2(curve):
        out_ = 0.0
        for q in m.points:
            if q[0] in HERA_KEYS:
                c = (
                    float(
                        np.exp(gmm_log(CURVES[curve], q[1], m.readme["decade"]))
                    )
                    / 1e3
                )
                out_ += ((q[2] - c) / (q[3] if c > q[2] else q[4])) ** 2
        return out_

    c2 = {c: chi2(c) for c in CURVES}
    best = search(
        r"of Gonçalves et al\. as published, ([\w-]+) describes the "
        r"HERA points best",
        IPSAT_FINDING,
    )[0]
    out = [
        in_source(
            "Fig. 1 left is charm_nucleon.eps; IP-SAT describes the "
            "high energies",
            GMM,
            r"\includegraphics[scale=0.5] {charm_nucleon.eps}",
            r"In contrast, the IP-SAT model describes the high energy "
            r"region but overestimate the low energy regime.",
        ),
        *results(),
        Result(
            f"of the three curves as published, {best} describes the "
            "HERA points best (chi^2 of the five)",
            best == m.readme["curve"]
            and c2[best] < min(v for k, v in c2.items() if k != best),
            ", ".join(f"{c} {v:.2f}" for c, v in c2.items()),
        ),
    ]
    with L.patched(
        r"Above $W$ = 275 GeV, where", r"Above $W$ = 250 GeV, where"
    ):
        out.append(must_fail("control: W = 250 GeV in the page", results()))
    return out


LISOVYI_ROWS = (
    r"$f(c\rightarrow{}D^+)$&$0.2404\pm0.0067$&$0.2439\pm0.0067$\\\hline",
    r"$f(c\rightarrow{}D^0)$&$0.6086\pm0.0076$&$0.6141\pm0.0073$\\\hline",
    r"$f(c\rightarrow{}D^{+}_{s})$&$0.0802\pm0.0040$&$0.0797\pm0.0040$\\\hline",
    r"$f(c\rightarrow{}\Lambda_c^{+})$&$0.0623\pm0.0041$&$0.0549\pm0.0026$\\"
    r"\hline",
)
LISOVYI_TABLE = r"\newcommand{\tabFINALaverage}{"
LISOVYI_LAMBDA = r"\newcommand{\EXTRABARYONS}{0.136\pm0.006}"


def lisovyi_table_number():
    tex = L.tar_member(LISOVYI, "FFC-txt.tex")
    if tex is None:
        return None
    order = list(dict.fromkeys(re.findall(r"\\tab([A-Z][A-Za-z]*)", tex)))
    return order.index("FINALaverage") + 1


@derivation("cp.fractions")
def check_fractions():
    """The fractions: Lisovyi et al.'s Table 12 ("Constrained S") renormalized
    to one hadron, as the page has them."""
    exact_f = fractions_exact()
    raw = {h: PPD.LISOVYI16_TABLE12[h][0] for h in HADRONS}
    number_ = lisovyi_table_number()
    tex = {
        h: sp.Rational(re.search(r"&\$([\d.]+)\\pm", row).group(1))
        for h, row in zip(("D+", "D0", "Ds+", "Lc+"), LISOVYI_ROWS)
    }
    lam = sp.Rational(re.search(r"\{([\d.]+)\\pm", LISOVYI_LAMBDA).group(1))

    def results():
        g = fraction_line()
        texts = dict(zip(HADRONS, g[8:12]))
        page_raw = dict(zip(HADRONS, map(sp.Rational, g[2:6])))
        total = sum(page_raw.values())
        return [
            Result(
                f"Table {g[0]}, column '{g[1]}': the averaged table, "
                "the text's twelfth",
                int(g[0]) == 12
                and number_ in (None, 12)
                and g[1] == "Constrained S",
                "not re-checked" if number_ is None else f"table {number_}",
            ),
            Result(
                "the page's four fractions = the table's",
                page_raw == tex,
                ", ".join(g[2:6]),
            ),
            Result(
                f"they add up to {g[6]}",
                rounds_as(float(total), g[6]),
                f"{float(total):.5f}",
            ),
            Result(
                f"the charm-strange baryons: {g[7]} per Lambda_c+ "
                "(their lambda), the rest of S = 1",
                sp.Rational(g[7]) == lam
                and abs(total + lam * page_raw["Lc+"] - 1)
                <= sp.Rational(5, 10000),
                f"S = {float(total + lam * page_raw['Lc+']):.5f}; "
                "lambda f(Lambda_c+) = "
                f"{100 * float(lam * page_raw['Lc+']):.2f} %",
            ),
            *(
                Result(
                    f"f({h}) = {texts[h]}",
                    rounds_as(float(exact_f[h]), texts[h]),
                    f"{float(exact_f[h]):.5f}",
                )
                for h in HADRONS
            ),
        ]

    out = [
        in_member(
            "Table 12's middle column (Tables/average_tab2.txt)",
            LISOVYI,
            "Tables/average_tab2.txt",
            *LISOVYI_ROWS,
        ),
        in_source(
            "Table 12 is the average_tab2.txt table, 'Constrained S' "
            "its middle column",
            LISOVYI,
            LISOVYI_TABLE + r"\begin{table}[htbp]\tabfontsize\centering"
            r"\begin{tabular}{|c|c|c|}\hline&\multirow{2}{*}{ "
            r"Constrained $S$}&Constrained $S$, \\",
            r"\input{Tables/average_tab2.txt}",
        ),
        in_member(
            "S (constrained to 1)",
            LISOVYI,
            "Tables/average_tab2.txt",
            r"$S$&$1.0000\pm0.0005$&$1.0000\pm0.0004$\\\hline",
        ),
        in_member(
            "lambda = 0.136, the charm-strange baryons per Lambda_c+",
            LISOVYI,
            "Tables/constants.txt",
            LISOVYI_LAMBDA,
        ),
        Result(
            "tests/support/measured/charm_pp.py has these values",
            {h: sp.Rational(str(v)) for h, v in raw.items()} == tex,
            ", ".join(f"{h} {raw[h]}" for h in HADRONS),
        ),
        *results(),
    ]
    with L.patched(
        r"renormalizes to one hadron: 0.614,",
        r"renormalizes to one hadron: 0.613,",
    ):
        out.append(must_fail("control: f(D0) = 0.613 in the page", results()))
    return out


FOCUS_TEX = (
    r"is $0.111 \pm 0.018 \pm 0.012$, significantly different from zero.",
    r"photons  ($\left<E\right>\approx 175$ GeV for events in which a charm "
    r"decay was reconstructed)",
)
POWER = r"h_a = 0.71\thinspace(\epsilon_r/\epsilon_a)^{-1.765}"


@derivation("cp.associated")
def check_associated():
    """h_a: 0.71 at 20 GeV, from FOCUS's asymmetry at 175 GeV, a power law
    between and above (at most 1):
    the h_a column."""
    sol = h_from_asymmetry()

    def results():
        m = pg_model()
        h0, e0, e1, asym = m.readme["h_a"]
        raw_text, sum_text, f_text, h_text, e_text = search(
            r"with the fraction \$f_\{\\Lambda_c\} = ([\d.]+)/([\d.]+) = "
            r"([\d.]+)\$ above, before rounding\. "
            r"This gives \$h_a = ([\d.]+)\$ at (\d+) GeV\.",
            HA_LINE,
        )
        species = fraction_line()
        # h_a from the printed fraction, which must give the printed h_a too
        h_printed = float(
            sol.subs({AS_: sp.Rational(str(asym)), F_: sp.Rational(f_text)})
        )
        lead, ea, power = (
            re.fullmatch(
                r"h_a = ([\d.]+)\\thinspace"
                r"\(\\epsilon_r/\\epsilon_a\)\^\{-([\d.]+)\}",
                L.inline(POWER),
            ).group(1),
            float(
                search(
                    r"with \$\\epsilon_a\$ = (\d+) GeV, at most 1\.", HA_LINE
                )[0]
            ),
            re.search(r"\^\{-([\d.]+)\}", POWER).group(1),
        )
        one = float(search(r"It reaches 1 below ([\d.]+) GeV\.", HA_LINE)[0])
        e_one = e0 * h0 ** (1 / m.p_ha)
        return [
            Result(
                f"h_a = {h0} at {e0:g} GeV: SLAC HFP (Abe et al. 1986, "
                "transcribed)",
                (h0, e0) == (PG.SLAC_ASSOCIATED[0], PG.SLAC_E_GAMMA),
                f"{PG.SLAC_ASSOCIATED}",
            ),
            Result(
                f"A = {asym} at {e1:g} GeV: FOCUS",
                (asym, e1) == (PG.FOCUS_LC_ASYMMETRY[0], PG.FOCUS_E_GAMMA[0]),
                f"f = {m.ff['Lc+']:.5f}",
            ),
            Result(
                f"f_Lambda_c = {raw_text}/{sum_text} = {f_text}, the "
                f"fraction above before rounding; h_a = {h_text} at "
                f"{e_text} GeV, from it and from the printed fraction",
                (raw_text, sum_text)
                == (dict(zip(HADRONS, species[2:6]))["Lc+"], species[6])
                and rounds_as(m.ff["Lc+"], f_text)
                and rounds_as(m.h_a1, h_text)
                and rounds_as(h_printed, h_text)
                and float(e_text) == e1,
                f"{m.h_a1:.5f}; {h_printed:.5f} from {f_text}",
            ),
            Result(
                f"the power law: {lead} (eps_r/{ea:g} GeV)^-{power} "
                f"through both, 1 below {one} GeV",
                float(lead) == h0
                and ea == e0
                and rounds_as(m.p_ha, power)
                and rounds_as(e_one, str(one)),
                f"power {m.p_ha:.5f}; 1 below {e_one:.3f} GeV",
            ),
        ] + compare_columns("h_a")

    out = [
        in_source(
            "FOCUS's asymmetry and photon energy as transcribed",
            FOCUS,
            *FOCUS_TEX,
        ),
        exact(
            "A = (N+ - N-)/(N+ + N-) with the page's N+ = h_a + (1 - h_a) "
            "f, N- = (1 - h_a) f: the page's last side",
            readme_asymmetry_by_n() - readme_asymmetry(),
        ),
        exact(
            "the page's h_a(A, f) is the one root of A = h_a/[h_a + "
            "2(1 - h_a) f]",
            readme_h_inverse() - sol,
        ),
        *results(),
    ]
    with L.patched(
        r"\frac{h_a}{h_a + 2(1 - h_a)f_{\Lambda_c}}",
        r"\frac{h_a}{h_a + (1 - h_a)f_{\Lambda_c}}",
    ):
        out.append(
            exact_fails(
                "control: (1 - h_a) f for 2(1 - h_a) f in the page",
                readme_asymmetry_by_n() - readme_asymmetry(),
            )
        )
        out.append(must_fail("control: the h_a column with it", results()))
    with L.patched(
        r"found $h_a = 0.71$ at 20 GeV", r"found $h_a = 0.72$ at 20 GeV"
    ):
        out.append(must_fail("control: 0.72 at 20 GeV in the page", results()))
    with L.patched(r"= 0.0628$ above,", r"= 0.063$ above,"):
        out.append(
            must_fail(
                "control: the rounded 0.063 in the h_a sentence", results()
            )
        )
    return out


FMNR_ROWS = (
    r"30 GeV& 2.582 & 2.104 & 1.712 & 1.321 & 1.090 & 0.907 & 0.735 & "
    r"0.611 & 0.515",
    r"100 GeV&6.558 & 5.056 & 4.057 & 3.415 & 2.826 & 2.356 & 2.009 & "
    r"1.721 & 1.467",
    r"280 GeV&13.89 & 10.13 & 8.008 & 7.314 & 5.858 & 4.831 & 4.389 & "
    r"3.685 & 3.120",
    r"30 GeV& 0.793 & 0.297 & 0.165 & 0.206 & 0.099 & 0.060 & 0.072 & "
    r"0.040 & 0.026",
    r"100 GeV&4.572 & 1.680 & 0.924 & 1.438 & 0.673 & 0.405 & 0.589 & "
    r"0.314 & 0.200",
    r"280 GeV&15.42 & 5.492 & 2.989 & 5.296 & 2.435 & 1.454 & 2.348 & "
    r"1.239 & 0.784",
)
FMNR_CONTEXT = (
    r"\multicolumn{10}{|c|}{Proton PDF set MRSA}",
    r"\multicolumn{10}{|c|}{Photon PDF set GRV--HO}",
    r"Pointlike component of the total charm cross sections "
    r"($\mu$b) in $\gamma p$",
    r"Hadronic component of the total charm cross sections "
    r"($\mu$b) in $\gamma p$",
)
FMNR_HEAD = (
    r"& \multicolumn{3}{c|}{$m_c=1.2$ GeV} & "
    r"\multicolumn{3}{c|}{$m_c=1.5$ GeV} & \multicolumn{3}{c|}{$m_c=1.8$ GeV}",
    r"$E_{\sss {\rm CM}}$ \ \ \ $\mur$ & $m_c/2$ &$m_c$ & 2$m_c$ & $m_c/2$ "
    r"&$m_c$ & 2$m_c$ & $m_c/2$ &$m_c$ & 2$m_c$",
)
ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
}


@derivation("cp.resolved")
def check_resolved():
    """h_N: half FMNR's resolved share, interpolated in ln W, held above
    280 GeV, zero below 12 GeV: the h_N column."""
    head = FMNR_HEAD
    masses_c = re.findall(r"\$m_c=([\d.]+)\$", head[0])
    scales = [s.strip().replace("$", "") for s in head[1].split("&")[1:4]]

    def results():
        m = pg_model()
        top, zero = m.readme["h_n"]
        w0 = m.resolved_zero()
        held = m.resolved_share([top, 2 * top, 1e4]).tolist()
        t1, t2, col, mc = search(
            r"\(their Table (\d)\) and of the resolved photon \(their "
            r"Table (\d)\)\. Their (\w+) column has the charm mass ([\d.]+) "
            r"GeV and the renormalization scale \$\\mu = m_c\$",
            FMNR_LINE,
        )
        column = ORDINALS[col]
        cols = [[float(v) for v in row.split("&")[1:]] for row in FMNR_ROWS]
        fm = readme_fmnr()
        point, hadr = (
            [c[column - 1] for c in cols[:3]],
            [c[column - 1] for c in cols[3:]],
        )
        shares = [h / (h + p) for h, p in zip(fm["resolved"], fm["point"])]
        tex = L.tex_source(FMNR)
        numbers = (
            None
            if tex is None
            else (
                table_number(tex, "c_point_tab"),
                table_number(tex, "c_hadr_tab"),
            )
        )
        eps0 = float(
            sp.solve(
                m.readme["w2"].subs(m_, MASS_Q["p"])
                - sp.Integer(int(zero)) ** 2,
                er2_,
            )[0]
        )
        return [
            Result(
                f"Tables {t1} and {t2}: the point-like and the hadronic "
                "(resolved) component",
                numbers in (None, (int(t1), int(t2))),
                "not re-checked" if numbers is None else f"{numbers}",
            ),
            Result(
                f"column {column}: m_c = {mc} GeV, mu = m_c",
                masses_c[(column - 1) // 3] == mc
                and scales[(column - 1) % 3] == "m_c",
                f"m_c {masses_c}, mu {scales}",
            ),
            Result(
                "the page's table = that column",
                (fm["W"], fm["point"], fm["resolved"])
                == ([30.0, 100.0, 280.0], point, hadr),
                f"{point}, {hadr} mubarn",
            ),
            Result(
                "the data module's = these",
                (tuple(point), tuple(hadr))
                == (PG.FMNR95_POINTLIKE, PG.FMNR95_HADRONIC)
                and tuple(fm["W"]) == PG.FMNR95_W,
                "FMNR95_*",
            ),
            Result(
                "the resolved shares",
                all(rounds_as(v, t) for v, t in zip(shares, fm["share"])),
                ", ".join(f"{v:.5f}" for v in shares),
            ),
            Result(
                f"held above W = {top:g} GeV, FMNR's highest energy",
                top == fm["W"][-1] and len(set(held)) == 1,
                f"{held[0]:.4f}",
            ),
            Result(
                f"zero at {zero:g} GeV on the line through "
                f"{m.readme['h_n_line'][0]:g} and "
                f"{m.readme['h_n_line'][1]:g} GeV (in ln W)",
                round(w0) == zero
                and m.readme["h_n_line"] == tuple(fm["W"][:2]),
                f"W = {w0:.3f} GeV",
            ),
            Result(
                f"W = {zero:g} GeV is eps_r = {m.readme['h_n_eps']} GeV",
                round(eps0) == int(m.readme["h_n_eps"]),
                f"{eps0:.2f} GeV",
            ),
        ] + compare_columns("h_N")

    out = [
        in_source(
            "Tables 1-2 of FMNR (MRSA; GRV-HO), their columns; the fifth",
            FMNR,
            *FMNR_ROWS,
            *FMNR_CONTEXT,
            *FMNR_HEAD,
        ),
        *results(),
    ]
    with L.patched(r"falls to zero at 12 GeV", r"falls to zero at 15 GeV"):
        out.append(must_fail("control: zero at 15 GeV in the page", results()))
    with L.patched(r"| Resolved [μb] | 0.099 |", r"| Resolved [μb] | 0.098 |"):
        out.append(
            must_fail("control: 0.098 for 0.099 mubarn in the table", results())
        )
    return out


@derivation("cp.chi")
def check_chi():
    """chi of the photon side, (E* - p*_L)/sqrt(s) in the source frame, and of
    the associated anti-D: the page's."""
    Es, pl, s, mN, t = sp.symbols("E_star p_L s m t", positive=True)
    rs = sp.sqrt(s)
    e_n, p_n = (s + mN**2) / (2 * rs), (s - mN**2) / (2 * rs)
    share = sp.limit((Es - t * pl) / (e_n + t * p_n), t, 1)
    pin = r"\chi = \langle E^\ast - p_L^\ast\rangle/\sqrt{s}"
    broken_pin = pin.replace("-", "+")

    def readme_chi(pin):
        tex = (
            L.sides(L.inline(pin))[1]
            .replace(r"\langle", "(")
            .replace(r"\rangle", ")")
        )
        return L.to_sympy(tex, {r"E^\ast": Es, r"p_L^\ast": pl, "s": s})

    lhs_n, rhs_n = L.sides(L.inline(r"E_N^\ast + p_N^\ast = \sqrt{s}"))
    sum_n = L.to_sympy(
        lhs_n, {r"E_N^\ast": e_n, r"p_N^\ast": p_n}
    ) - L.to_sympy(rhs_n, {"s": s})

    def results():
        m = pg_model()
        return [
            Result(
                f"E691 at {m.readme['e691']:g} GeV",
                m.readme["e691"] == PG.E691_E_GAMMA,
                f"{PG.E691_E_GAMMA} GeV",
            ),
            *compare_columns("chi_D", "chi_Ds", "chi_Lc", "chi_aD"),
        ]

    out = [
        exact(
            "E*_N + p*_N = sqrt(s) for the nucleon in the centre of momentum",
            sum_n,
        ),
        exact(
            "the source-frame energy fraction of the nucleon energy: "
            "(E* - p*_L)/sqrt(s) for E_N -> oo at fixed s",
            share - readme_chi(pin),
        ),
        *results(),
    ]
    with L.patched(
        r"| 0.1961, 0.2093, 0.2486 |", r"| 0.1971, 0.2093, 0.2486 |"
    ):
        out.append(
            must_fail("control: 0.1971 for 0.1961 in the table", results())
        )
    with L.patched(f"${pin}$", f"${broken_pin}$"):
        out.append(
            exact_fails(
                "control: E* + p*_L in the page", share - readme_chi(broken_pin)
            )
        )
    return out


@derivation("cp.e691")
def check_e691():
    """dsigma of the invariant cross section F(x_F) h(p_T^2), its x_F integrals
    in closed form, E691's spectra fitted."""
    dsig, q1, q2, a, chi = readme_e691()
    xf = L.to_sympy(
        L.sides(L.inline(XF_INLINE))[1], {r"p_L^\ast": pL_, "s": s_}
    )
    mt = sp.sqrt(Es_**2 - pL_**2)
    derived = (
        FF(xF_)
        * HH(pT2_)
        * sp.pi
        * dpT2
        * dxF
        * sp.diff(sp.solve(xf - xF_, pL_)[0], xF_)
        / Es_
    )
    p1, p2 = readme_primitives()
    out = [
        exact(
            "dsigma = pi F h dx_F dp_T^2/q of E d^3sigma/dp^3 = F(x_F) "
            "h(p_T^2), x_F = 2p*_L/sqrt(s), q = 2E*/sqrt(s)",
            dsig.subs(q_, q1) - derived,
        ),
        exact(
            "q = 2E*/sqrt(s) = sqrt(x_F^2 + a^2): E*^2 = p*_L^2 + "
            "m_T^2 (squares)",
            q1**2 - (q2**2).subs({xF_: xf, a_: a.subs(mT_, mt)}),
        ),
        exact(
            "chi = (q - x_F)/2 = (E* - p*_L)/sqrt(s)",
            chi.subs({q_: q1, xF_: xf}) - (Es_ - pL_) / sp.sqrt(s_),
        ),
        exact(
            "d/dx_F arsinh(x_F/a) = 1/q", sp.simplify(sp.diff(p1, xF_) - 1 / q2)
        ),
        exact(
            "d/dx_F (x_F - q)/2 = chi/q",
            sp.simplify(sp.diff(p2, xF_) - (chi / q_).subs(q_, q2)),
        ),
    ]
    rec = L.hepdata("ins289318")
    if rec is None:
        out.append(
            absent(
                "E691's x_F and p_T^2 distributions "
                "(HEPData ins289318 Tables 11, 10)",
                "HEPData-ins289318",
                L.HEPDATA,
            )
        )
    else:

        def same(table, values, edges):
            rows = rec[table][0][2]
            return close(
                [float(r[1]) for r in rows] + [float(r[2]) for r in rows],
                [v for v, _ in values] + [e for _, e in values],
            ) and close(
                [float(r[0]) for r in rows],
                0.5 * (np.array(edges[:-1]) + np.array(edges[1:])),
            )

        out.append(
            Result(
                "E691's x_F and p_T^2 distributions as transcribed "
                "(HEPData ins289318 Tables 11, 10)",
                same("Table11", PG.E691_DSDXF, PG.E691_XF_EDGES)
                and same("Table10", PG.E691_DSDPT2, PG.E691_PT2_EDGES),
                "bins and values",
            )
        )
    *_, dx, dp = e691_invariant(PG.E691_E_GAMMA, hash(L.text()))
    L.line(
        r"An invariant cross section $F(x_F)\thinspace h(p_T^2)$ reproduces "
        r"both spectra exactly"
    )
    out.append(
        Result(
            f"F(x_F) h(p_T^2), h in {pt2_parts()} parts of each p_T^2 bin, "
            "reproduces both spectra exactly (the page)",
            max(dx, dp) < 1e-12,
            f"largest residuals {dx:.1e}, {dp:.1e} of the normalised bins",
        )
    )
    with L.patched(
        r"\frac{dx_F\thinspace dp_T^2}{q}", r"dx_F\thinspace dp_T^2"
    ):
        out.append(
            exact_fails(
                "control: no 1/q in dsigma",
                readme_e691()[0].subs(q_, q1) - derived,
            )
        )
    with L.patched(r"= \frac{x_F - q}{2}.", r"= \frac{q - x_F}{2}."):
        out.append(
            exact_fails(
                "control: (q - x_F)/2 as the primitive of chi/q",
                sp.simplify(
                    sp.diff(readme_primitives()[1], xF_)
                    - (chi / q_).subs(q_, q2)
                ),
            )
        )
    return out


def recoil_candidates(need):
    names = list(QN)
    out = []
    for k in (1, 2):
        for combo in combinations_with_replacement(names, k):
            if tuple(map(sum, zip(*(QN[p] for p in combo)))) == need:
                out.append((sum(MASS[p] for p in combo), combo))
    return sorted(out)


@derivation("cp.recoil")
def check_recoil():
    """E* is the kinematic limit of a hadron against its recoil, and each recoil
    keeps the quantum numbers and is the lightest."""
    tex = L.sides(L.inline(KINLIM))
    assert tex[0] == r"E^\ast", tex
    ours = L.to_sympy(tex[1], {"s": s_, "m": m0_, "m_X": mX_})
    EX = sp.Symbol("E_X", positive=True)
    two_body = sp.solve(
        [Es_ + EX - sp.sqrt(s_), Es_**2 - m0_**2 - (EX**2 - mX_**2)],
        [Es_, EX],
        dict=True,
    )[0][Es_]

    def results():
        out_ = []
        for h, rec in readme_recoils().items():
            need = tuple(g - q for g, q in zip(GAMMA_P, QN[h]))
            have = tuple(map(sum, zip(*(QN[p] for p in rec))))
            lightest = recoil_candidates(need)[0] if have == need else None
            mass = sum(MASS[p] for p in rec)
            ok = have == need and abs(lightest[0] - mass) < 1e-12
            out_.append(
                Result(
                    f"{h} + {' '.join(rec)}: the charge, baryon "
                    "number, charm and strangeness of gamma p, "
                    "the lightest such state",
                    ok,
                    f"{mass:.5f} GeV"
                    + (
                        ""
                        if lightest is None
                        else f"; lightest {' '.join(lightest[1])}"
                    ),
                )
            )
        return out_

    out = [
        exact(
            "E* = (s + m^2 - m_X^2)/(2 sqrt(s)): two bodies of E* + E_X = "
            "sqrt(s) and equal momenta",
            ours - two_body,
        ),
        *results(),
    ]
    with L.patched(r"$\Lambda_c^+\pi^+$", r"$\Lambda_c^+$"):
        out.append(
            must_fail("control: Lambda_c+ alone as the recoil of D-", results())
        )
    return out


@derivation("cp.nucleon")
def check_nucleon():
    """chi_N: z-bar times the mean x of the quarks above x_c, z independent
    of x (forward hadrons): the page's column."""
    xq, z = sp.symbols("x_q z", positive=True)
    k_num, k_den = readme_nucleon_measures()
    assert (
        L.sides(L.part(L.display(NUCLEON_PIN), r"\bar z ="))[1]
        == r"\sum_h f_h\langle z\rangle_h"
    )
    g = xq * (1 - xq) ** 2  # test functions on (0, 1)
    D = (z**2 * (1 - z), z * (1 - z) ** 3)  # two species, unnormalized
    f = [sp.integrate(Dh, (z, 0, 1)) for Dh in D]
    lo = sp.Rational(1, 4)
    num = sum(
        sp.integrate(sp.integrate(z * xq * Dh, (z, 0, 1)) * g / xq, (xq, lo, 1))
        for Dh in D
    )
    den = sum(
        sp.integrate(sp.integrate(Dh, (z, 0, 1)) * g / xq, (xq, lo, 1))
        for Dh in D
    )
    zbar = sum(
        fh * sp.integrate(z * Dh, (z, 0, 1)) / fh for fh, Dh in zip(f, D)
    ) / sum(f)
    ours = (
        zbar
        * sp.integrate(g * xq**k_num, (xq, lo, 1))
        / sp.integrate(g * xq**k_den, (xq, lo, 1))
    )

    def results():
        y0 = readme_forward_ystar()
        return [
            Result(
                f"y* > {y0:g} is x > x_c (x = x_c e^y*, cpp.g)",
                y0 == 0,
                "x_c e^0 = x_c",
            ),
            *compare_columns("chi_N"),
        ]

    out = [
        exact(
            "chi_N = z-bar int_x_c^1 g dx / int_x_c^1 g dx/x: the mean x "
            "of the hadrons (any g, D_h; f_h-weighted z-bar, sum f_h = 1 "
            "normalised)",
            ours - num / den,
        ),
        *results(),
    ]
    with L.patched(
        r"\thinspace dx}{\int_{x_c}^1 g(x)\thinspace dx/x}",
        r"\thinspace dx/x}{\int_{x_c}^1 g(x)\thinspace dx}",
    ):
        k1, k2 = readme_nucleon_measures()
        broken = (
            zbar
            * sp.integrate(g * xq**k1, (xq, lo, 1))
            / sp.integrate(g * xq**k2, (xq, lo, 1))
        )
        out.append(
            exact_fails(
                "control: the measures swapped in the page", broken - num / den
            )
        )
        out.append(
            must_fail(
                "control: the nucleon side's column with them",
                compare_columns("chi_N"),
            )
        )
    with L.patched(
        r"whose quarks have $y^\ast \gt 0$",
        r"whose quarks have $y^\ast \gt 0.5$",
    ):
        out.append(
            must_fail(
                "control: the nucleon side's hadrons of the "
                "quarks with y* > 0.5",
                results(),
            )
        )
    return out


def readme_types():
    line = L.line(TYPES_LINE)
    out = {
        int(k) - 1: (float(lo), float(hi))
        for k, lo, hi in re.findall(
            r"M(\d)(?: covers)? "
            r"([\d.]+)–([\d.]+) GeV",
            line,
        )
    }
    m = re.search(
        r"M7 everything above, up to \$5 \\times 10\^6\$ GeV in AM³\.", line
    )
    out[6] = (out[5][1], 5e6 if m else None)
    sigma = float(
        re.search(
            r"Each type \$t\$ has the cross section "
            r"\$\\sigma_t\$ = (\d+) μb\.",
            line,
        ).group(1)
    )
    return out, sigma


@derivation("cp.table")
def check_table():
    """The page's types are Table 6's, and its ten steps fill them, contiguous,
    from the threshold to 5e6 GeV."""
    tex = L.tex_source(HUEMMER)
    number_ = table_number(tex, "tab:mtype")
    t6 = table6()

    def results():
        types, sigma = readme_types()
        rows = readme_charm_steps()
        edges = [(r["eps_min"], r["eps_max"]) for r in rows]
        return [
            Result(
                "Table 6 of Hümmer et al.",
                number_ in (None, 6),
                "not re-checked" if number_ is None else f"table {number_}",
            ),
            Result(
                "the types M4-M7 and their ranges are Table 6's "
                "(M7 to oo there, to 5e6 GeV in AM3)",
                all(
                    types[t][0] == t6[t]["lo"]
                    and (
                        types[t][1] == t6[t]["hi"]
                        or t == 6
                        and t6[t]["hi"] == math.inf
                    )
                    for t in range(3, 7)
                ),
                ", ".join(
                    f"M{t + 1} {types[t][0]:g}-{types[t][1]:g}"
                    for t in range(3, 7)
                ),
            ),
            Result(
                f"each with sigma_t = {sigma:g} mubarn (Table 6)",
                all(t6[t]["sigma"] == sigma for t in range(3, 7)),
                ", ".join(f"M{t + 1} {t6[t]['sigma']:g}" for t in range(3, 7)),
            ),
            Result("ten steps", len(rows) == 10, f"{len(rows)} rows"),
            Result(
                "contiguous, each inside its type, the last ending at 5e6 GeV",
                all(a[1] == b[0] for a, b in zip(edges, edges[1:]))
                and edges[-1][1] == types[6][1]
                and all(
                    types[r["t"]][0]
                    <= r["eps_min"]
                    < r["eps_max"]
                    <= types[r["t"]][1]
                    for r in rows
                ),
                f"{edges[0][0]}-{edges[-1][1]} GeV",
            ),
        ]

    out = [
        in_source(
            "Table 6 (the head, the rows of M4-M7) as transcribed",
            HUEMMER,
            *H_TABLE6_HEAD,
            *H_TABLE6,
        ),
        *results(),
    ]
    with L.patched(r"M5 50–500 GeV", r"M5 50–600 GeV"):
        out.append(must_fail("control: M5 to 600 GeV in the page", results()))
    return out


PARTS_TABLE = r"| Part | Hadrons per interaction | $\chi$ |"
LC_CHI_LINE = (
    r"Associated production: The associated $\Lambda_c^+$ takes "
    r"the energy fraction of the type's nucleon"
)
BARYON_LINE = r"which adds the baryon number $R h_a$ per interaction, at most $"


def readme_parts():
    R, ha, hN, fh, f0, fp = sp.symbols("R h_a h_N f_h f_D0 f_Dp", positive=True)
    tok = {
        "R": R,
        "h_a": ha,
        "h_N": hN,
        "f_h": fh,
        r"f_{\mathrm{D}^0}": f0,
        r"f_{\mathrm{D}^+}": fp,
    }
    rows = {r[0]: r for r in L.table(PARTS_TABLE)}
    first = lambda cell: L.to_sympy(
        re.match(r"\$([^$]+)\$", cell).group(1), tok
    )
    ad = re.fullmatch(
        r"\$([^$]+)\$ \$\\bar\{\\mathrm\{D\}\}\^0\$, \$([^$]+)\$ "
        r"\$\\mathrm\{D\}\^-\$",
        rows["Associated anti-D"][1],
    )
    return (
        dict(
            photon=first(rows["Photon side"][1]),
            nucleon=first(rows["Nucleon side"][1]),
            anti_d0=L.to_sympy(ad.group(1), tok),
            d_minus=L.to_sympy(ad.group(2), tok),
            lc=first(rows[r"Associated $\Lambda_c^+$"][1]),
        ),
        (R, ha, hN, fh, f0, fp),
    )


@derivation("cp.parts")
def check_parts():
    """Each interaction makes R charm and R anticharm quarks, and the associated
    Lambda_c+ adds the baryon number R h_a."""
    t6 = table6()

    def results():
        parts, (R, ha, hN, fh, f0, fp) = readme_parts()
        fs = sp.Symbol("f_Ds", positive=True)
        # sum f_h = 1
        frac = {"D0": f0, "D+": fp, "Ds+": fs, "Lc+": 1 - f0 - fp - fs}
        side = lambda e: sum(e.subs(fh, f) for f in frac.values())
        charm = side(parts["photon"]) + side(parts["nucleon"]) + parts["lc"]
        anti = (
            side(parts["photon"])
            + side(parts["nucleon"])
            + parts["anti_d0"]
            + parts["d_minus"]
        )
        rows = {r[0]: r[1] for r in L.table(PARTS_TABLE)}
        both = all(
            "of each hadron $h$ and of its antiparticle" in rows[k]
            for k in ("Photon side", "Nucleon side")
        )
        # the sides make as many Lambda_c+ as anti-Lambda_c- (each hadron and
        # its antiparticle)
        baryon = parts["lc"]
        worst = max(r["R"] * r["h_a"] for r in readme_charm_steps())
        limit = number(search(r"at most \$([^$]+)\$\.", BARYON_LINE)[0])
        fractions = search(
            r"Table 6\. This fraction is ([\d.]+) for M4, "
            r"([\d.]+) for M5 and ([\d.]+) for M6 and M7\.",
            LC_CHI_LINE,
        )
        q = {k: QN[k][0] for k in ("p", "n", "Lc+", "anti-D0", "D-")}
        return [
            Result(
                "R charm quarks per interaction: photon and nucleon "
                "sides and the associated Lambda_c+",
                sp.simplify(charm - R) == 0,
                f"{sp.factor(charm)}",
            ),
            Result(
                "R anticharm quarks: the antiparticles of both sides "
                "and the associated anti-D",
                sp.simplify(anti - R) == 0,
                f"{sp.factor(anti)}",
            ),
            Result(
                "the baryon number R h_a of the associated Lambda_c+, "
                f"at most {limit:g} per interaction (the steps); "
                "none from the sides, which make each hadron and its "
                "antiparticle alike",
                both and sp.simplify(baryon - R * ha) == 0 and worst <= limit,
                f"largest {worst:.4g}",
            ),
            Result(
                "from protons Lambda_c+ anti-D0, from neutrons "
                "Lambda_c+ D-: the charge kept (the other way not)",
                q["p"] == q["Lc+"] + q["anti-D0"]
                and q["n"] == q["Lc+"] + q["D-"]
                and q["n"] != q["Lc+"] + q["anti-D0"],
                "1 = 1 + 0, 0 = 1 - 1",
            ),
            Result(
                "the associated Lambda_c+ at 1 - K of Table 6: "
                f"{', '.join(fractions)} (M4, M5, M6 and M7)",
                [sp.Rational(v) for v in fractions]
                == [1 - t6[t]["K"] for t in (3, 4, 5)]
                and t6[6]["K"] == t6[5]["K"],
                ", ".join(f"M{t + 1} K = {t6[t]['K']}" for t in range(3, 7)),
            ),
        ]

    out = results()
    with L.patched(r"$R(1 - h_a - h_N)f_h$", r"$R(1 - h_a)f_h$"):
        out.append(
            must_fail("control: R (1 - h_a) f_h on the photon side", results())
        )
    return out


@derivation("cp.bookkeeping")
def check_bookkeeping():
    """s_{t,r} keeps each interaction's energy, and f_t - kappa_r f_r is the
    response of s_{t,r} sigma_t in the step."""
    lhs, rhs = L.sides(L.display(r"s_{t,r} = 1 - \frac{\sum_k \sum_h"))
    assert lhs == "s_{t,r}", lhs
    m = re.fullmatch(
        r"(.+?)\\frac\{\\sum_k \\sum_h (.+?)\}\{\\sum_\{c \\in "
        r"t\} (.+?)\}",
        rhs,
    )
    chis, Ms, chic, Mc = (
        sp.symbols(f"{n}1:3", positive=True)
        for n in ("chi_h", "M_h", "chi_c", "M_c")
    )
    num = sum(
        L.to_sympy(m.group(2), {r"\chi_k^h": a, "M_{k,N}^h": b})
        for a, b in zip(chis, Ms)
    )
    den = sum(
        L.to_sympy(m.group(3), {r"\chi_c": a, r"M_{c,N}^\pi": b})
        for a, b in zip(chic, Mc)
    )
    s = L.to_sympy(m.group(1) + "X", {"X": num / den})
    kappa = L.to_sympy(
        L.sides(L.inline(r"\kappa_r = 1 - s_{t,r}"))[1], {"s_{t,r}": s}
    )
    out = [
        exact(
            "s_{t,r} x the pions' energy + the charm hadrons' = the "
            "pions' energy: each interaction keeps its energy",
            s * den + num - den,
        )
    ]
    a, d1, d2, d3, sig, kap = sp.symbols(
        "a d_1 d_2 d_3 sigma_t kappa", positive=True
    )
    b, c, e = a + d1, a + d1 + d2, a + d1 + d2 + d3
    (pre, lo, up, body), _ = readme_response()
    inside, above = (
        readme_response(pick)[1] for pick in (lambda u, v: u, lambda u, v: v)
    )

    def f(where, lo_, hi_, y2):
        return (
            0
            if where == "below"
            else (inside if where == "in" else above).subs(
                {E1: lo_, E2: hi_, ST: sig, yy: y2 / 2}
            )
        )

    for name, y2, type_, step in (
        ("2y in the type below the step", a + d1 / 2, "in", "below"),
        ("2y in the step", b + d2 / 2, "in", "in"),
        ("2y in the type above the step", c + d3 / 2, "in", "above"),
        ("2y above the type", e + d3, "above", "above"),
    ):
        direct = step_integral(
            pre,
            lo,
            up,
            body,
            [(sig, a, b), (sig * (1 - kap), b, c), (sig, c, e)],
        ).subs(yy, y2 / 2)
        out.append(
            exact(
                f"{name}: f_t - kappa f_r = the response of sigma_t "
                "(1 - kappa) in the step, sigma_t elsewhere",
                f(type_, a, e, y2) - kap * f(step, b, c, y2) - direct,
            )
        )
    out.append(
        exact(
            "kappa_r = 1 - s_{t,r} = the charm hadrons' share of the "
            "pions' energy",
            kappa - num / den,
        )
    )
    with L.patched(
        r"s_{t,r} = 1 - \frac{\sum_k", r"s_{t,r} = 1 + \frac{\sum_k"
    ):
        _, rhs_b = L.sides(L.display(r"s_{t,r} = 1 + \frac{\sum_k \sum_h"))
        mb = re.fullmatch(
            r"(.+?)\\frac\{\\sum_k \\sum_h (.+?)\}\{\\sum_\{c "
            r"\\in t\} (.+?)\}",
            rhs_b,
        )
        broken = L.to_sympy(mb.group(1) + "X", {"X": num / den})
        out.append(
            exact_fails(
                "control: 1 + the share in the page", broken * den + num - den
            )
        )
    return out


def gamma_cm(E):
    return (E + MP) / math.sqrt(2 * MP * MP + 2 * MP * E)


def sigma_inel(E):
    """AM3's inelastic pp cross section [mb]: Kafexhiu et al. 2014, eq. 1, with
    the charged-pion threshold and the code's masses."""
    mpi = MASS["pi+"]
    t_th = 2 * mpi + mpi**2 / (2 * MP)
    tp = E - MP
    if tp <= t_th:
        return 0.0
    lg = math.log(tp / t_th)
    return (30.7 - 0.96 * lg + 0.18 * lg * lg) * (1 - (t_th / tp) ** 1.9) ** 3


# finds the display also with a control's broken argument
G_KEY = r"g(x) = A\thinspace P(x)\thinspace P"
X0_PIN = r"\log_{10} x_0 = b_0 + b_1 l + b_2 l^2"
D_PIN = r"D_h(z) = N_h\thinspace\frac{z(1-z)^2}{[(1-z)^2 + \epsilon_h z]^2}"
TH_PIN = (
    r"\sigma_{c\bar c} = \sigma_2\left[\frac{1 - s_{\rm th}/s}{1 - "
    r"s_{\rm th}/s_2}\right]^{a}"
)
TH_SYMS = sp.symbols("sigma_2 s_th s s_2 a")
Q_PIN = r"Q_h(E) = c\thinspace n_H\int \frac{dE_p}{E_p}"
u_, x_, x0_, xc_, mT2_, sP_, z_, eh_, Nh_, l_, Ep_ = sp.symbols(
    "u x x_0 x_c m_T s z epsilon_h N_h l E_p", positive=True
)
b0_, b1_, b2_, n0_, n1_ = sp.symbols("b_0 b_1 b_2 n_0 n_1", real=True)


def readme_closed_form():
    parts = L.parts(L.display(G_KEY))
    assert len(parts) == 4, parts
    lhs, rhs = L.sides(parts[1])
    assert lhs == "P(u)", lhs
    P = sp.Lambda(u_, L.to_sympy(rhs, {"u": u_, "n": n_, "x_0": x0_}))
    lhs, rhs = L.sides(parts[0])
    assert lhs == "g(x)", lhs
    g = L.to_sympy(rhs, {"A": A_, "x": x_, "x_c": xc_}, {"P": P})
    lhs, rhs = L.sides(parts[2])
    assert lhs == "x_c", lhs
    return P, g, L.to_sympy(rhs, {"m_T": mT2_, "s": sP_}), parts[3]


def readme_energy_dependence():
    parts = L.parts(L.display(X0_PIN))
    tok = {"b_0": b0_, "b_1": b1_, "b_2": b2_, "n_0": n0_, "n_1": n1_, "l": l_}
    lg, n = (L.to_sympy(L.sides(p)[1], tok) for p in parts[:2])
    m = re.fullmatch(
        r"l = \\min\\negthinspace\\left\(\\log_\{10\}\\frac\{E_p\}"
        r"\{10\^(\d+)\\thinspace\\mathrm\{GeV\}\}, (\d+)\\right\)",
        parts[2],
    )
    return lg, n, (10.0 ** int(m.group(1)), float(m.group(2)))


def readme_s():
    rhs = L.sides(
        re.search(r"with \$(s = [^$]+)\$, \$m_T", L.line(PP_PARAMS)).group(1)
    )[1]
    return L.to_sympy(rhs, {"m_p": m_, "E_p": Ep_})


def readme_peterson():
    parts = L.parts(L.display(D_PIN))
    lhs, rhs = L.sides(parts[0])
    assert lhs == "D_h(z)", lhs
    return L.to_sympy(rhs, {"N_h": Nh_, "z": z_, r"\epsilon_h": eh_}), parts[1]


class PPCharm:
    def __init__(self):
        self.par = readme_pp_charm()
        P, _, xc, _ = readme_closed_form()
        lg, n, (self.e_ref, self.l_max) = readme_energy_dependence()
        self.P = sp.lambdify((u_, n_, x0_), P(u_), "numpy")
        self.lg = sp.lambdify((b0_, b1_, b2_, l_), lg, "math")
        self.n = sp.lambdify((n0_, n1_, l_), n, "math")
        self.s = sp.lambdify((m_, Ep_), readme_s(), "math")
        self.xc = sp.lambdify((mT2_, sP_), xc, "math")
        D, _ = readme_peterson()
        self.D = sp.lambdify((z_, eh_), D.subs(Nh_, 1), "numpy")
        self.frac = readme_fractions()
        self.norm = {
            h: self.frac[h] / self._int_d(self.par["eps"][h]) for h in HADRONS
        }
        self.zmean = {
            h: self._int_d(self.par["eps"][h], 1)
            / self._int_d(self.par["eps"][h])
            for h in HADRONS
        }
        self.zbar = sum(self.frac[h] * self.zmean[h] for h in HADRONS) / sum(
            self.frac.values()
        )
        self.below = threshold_form(self.par["a"], lambda E: self.s(MP, E))
        self.ystar = readme_forward_ystar()
        self.k_num, self.k_den = readme_nucleon_measures()
        self._pmx = {}

    def _int_d(self, eps, power=0):
        return quad(
            lambda z: z**power * float(self.D(z, eps)),
            0,
            1,
            points=[1 - math.sqrt(eps)],
            limit=400,
            epsabs=0,
            epsrel=1e-11,
        )[0]

    def shape(self, E, p=None):
        return Shape(self, E, self.par if p is None else p)

    def sigma_cc(self, E):
        """sigma(pp -> c cbar X) [mubarn] of the page up to E_top."""
        tab = self.par["sigma"]
        es = np.array(sorted(tab))
        ss = np.array([tab[e] for e in es])
        if E < es[0]:
            return self.below(E, es[0], ss[0])
        lg = math.log10(E)
        if lg <= math.log10(es[-1]):
            return 10 ** float(np.interp(lg, np.log10(es), np.log10(ss)))
        return ss[-1] * 10 ** (
            math.log10(ss[-1] / ss[-2])
            / math.log10(es[-1] / es[-2])
            * (lg - math.log10(es[-1]))
        )

    def convolve(self, g, x, h):
        eps, out = self.par["eps"][h], np.zeros(len(x))
        w = math.sqrt(eps)
        zp = 1 - w
        for i, xi in enumerate(x):
            edges = sorted(
                {xi, 1.0}
                | {min(1.0, max(xi, v)) for v in (zp - 6 * w, zp - w, zp + w)}
            )
            for a, b in zip(edges, edges[1:]):
                if b > a:
                    z = a + (b - a) * (GX + 1) / 2
                    out[i] += np.sum(
                        GW
                        * (b - a)
                        / 2
                        * self.norm[h]
                        * self.D(z, eps)
                        * g(xi / z)
                    )
        return out

    def projectile_mean_x(self, E, p=None):
        key = (E, None if p is None else tuple(sorted(p.items())))
        if key not in self._pmx:
            et = self.par["E_t"]
            if E >= et:
                g = self.shape(E, p)
                # the quark of CM rapidity y* has x = x_c e^{y*} (cpp.g)
                lo = g.xc * math.exp(self.ystar)
                self._pmx[key] = (
                    self.zbar
                    * g.moment(self.k_num, lo)
                    / g.moment(self.k_den, lo)
                )
            else:
                kf = self.projectile_mean_x(et, p) * et / (et - MP) / self.zbar
                self._pmx[key] = (
                    sum(
                        self.frac[h]
                        * max(
                            self.zmean[h] * kf * (E - MP), gamma_cm(E) * MASS[h]
                        )
                        for h in HADRONS
                    )
                    / sum(self.frac.values())
                    / E
                )
        return self._pmx[key]


GX, GW = np.polynomial.legendre.leggauss(64)


class Shape:
    """g(x; E_p) of the page, normalized (int g dx/x = 1)."""

    def __init__(self, model, E, p):
        l = min(math.log10(E / model.e_ref), model.l_max)
        self.model = model
        self.xc = model.xc(p["mT"], model.s(MP, E))
        self.x0 = 10 ** model.lg(p["b0"], p["b1"], p["b2"], l)
        self.n = model.n(p["n0"], p["n1"], l)
        self.A = 1.0
        self.A = 1.0 / self.moment(-1)

    def __call__(self, x):
        x = np.asarray(x, float)
        lo = self.xc**2
        ok = (x > lo) & (x < 1)
        xs = np.where(ok, x, 0.5)
        return np.where(
            ok,
            self.A
            * self.model.P(xs, self.n, self.x0)
            * self.model.P(lo / xs, self.n, self.x0),
            0.0,
        )

    def moment(self, k, xmin=None):
        lo = math.log(self.xc**2) if xmin is None else math.log(xmin)
        t = np.linspace(lo, 0.0, 8001)
        f = self(np.exp(t)) * np.exp((k + 1) * t)
        h = t[1] - t[0]
        return (
            h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
        )


def threshold_form(a, s_of):
    expr, rth = readme_threshold()
    sth = float(rth**2)
    f = sp.lambdify(TH_SYMS, expr, "math")

    def sigma(E, E2, sig2):
        s, s2 = s_of(E), s_of(E2)
        return 0.0 if s <= sth else f(sig2, sth, s, s2, a)

    return sigma


def readme_threshold():
    parts = L.parts(L.display(TH_PIN))
    expr = L.to_sympy(
        L.sides(parts[0])[1],
        dict(zip((r"\sigma_2", r"s_{\rm th}", "s", "s_2", "a"), TH_SYMS)),
    )
    lhs, rhs = L.sides(parts[1])
    assert lhs == r"\sqrt{s_{\rm th}}", lhs
    m = MASS_Q
    return expr, L.to_sympy(
        rhs,
        {
            "m_p": m["p"],
            r"m_{\Lambda_c}": m["Lc+"],
            r"m_{\mathrm{D}^0}": m["D0"],
        },
    )


FIT_E = (1e3, 1e6, 1e9)  # Bhattacharya et al. 2015, Fig. 2
LEVEL_LINE = r"The fit: Bhattacharya et al. (2015)'s Fig. 2 lies a factor $1/{"


def carpio_x_max():
    return float(
        search(
            r"curve only for its shape, up to \$x = ([\d.]+)\$\.", LEVEL_LINE
        )[0]
    )


@lru_cache(None)
def fit_grid(page_hash):
    grid = {
        E: np.exp(np.linspace(math.log(0.01), math.log(0.5), 40)) for E in FIT_E
    }
    grid[1e11] = np.exp(
        np.linspace(math.log(1.2e-6), math.log(carpio_x_max()), 60)
    )
    return grid


def berss_ratio():
    """Bhattacharya et al. 2016 Fig. 12 "BERSS" / their 2015 Fig. 2
    (pb -> mubarn)."""
    out = []
    for E in (1e6, 1e9):
        b = np.array(PPD.BERSS16_FIG12[("BERSS", E)])
        m = b[:, 0] <= 0.3
        a = np.array(PPD.BERSS15_FIG2[E])
        out.append(
            b[m, 1]
            / (np.exp(np.interp(b[m, 0], a[:, 0], np.log(a[:, 1]))) * 1e-6)
        )
    return out


R_BERSS = round(float(np.median(np.concatenate(berss_ratio()))), 3)


def quark_data(E):
    """(x, x dsigma/dx [mubarn]) of Bhattacharya et al. 2015, Fig. 2, at the
    level of the 2016 replot (R_BERSS)."""
    a = np.array(PPD.BERSS15_FIG2[E])
    return a[:, 0], R_BERSS * a[:, 1] * 1e-6 * a[:, 0]


def carpio():
    a = np.array(PPD.CARPIO20_FIG7_NLO)
    return a[:, 0], a[:, 1]


def residuals(model, q, grid):
    p, cn = dict(zip(NAMES, q[:6])), q[6]
    out = []
    for E, x in grid.items():
        g = model.shape(E, p)
        if E < 1e10:
            xd, yd = quark_data(E)
            y, m = (
                np.exp(np.interp(x, xd, np.log(yd))) / model.sigma_cc(E),
                g(x),
            )
        else:
            xd, yd = carpio()
            y = np.exp(
                np.interp(np.log(x), np.log(xd), np.log(yd))
            ) / model.sigma_cc(E)
            m = model.convolve(g, x, "D0") * math.exp(cn)
        out.append(np.log(np.maximum(m, 1e-300) / y))
    return np.concatenate(out)


@lru_cache(None)
def refit(key):
    model, grid = pp_model(), fit_grid(key)
    q0, lb, ub = (
        [1.7, -1.3, 0.0, 0.0, 3.0, 0.0, 0.2],
        [0.5, -4.0, -0.5, -0.2, 0.0, -1.0, -1.0],
        [4.0, 0.0, 0.5, 0.2, 15.0, 1.0, 1.0],
    )
    return least_squares(
        lambda q: residuals(model, q, grid),
        q0,
        bounds=(lb, ub),
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )


def fitted():
    return refit(hash(L.text()))


NH, CC = sp.symbols("n_H c", positive=True)
F_H, S_CC, S_INEL, K_H, T_P = sp.symbols(
    "f_h sigma_cc sigma_inel K_h T_p", positive=True
)
NPF, DF, GF = (sp.Function(name) for name in ("N_p", "D", "g"))
SOURCE_HEAD = r"c\thinspace n_H\int \frac{dE_p}{E_p}\thinspace N_p(E_p)\left["
SOURCE_TAIL = r"\right]_ {x = E/E_p}"
PSI = sp.Function("dsigma_h")  # dsigma_h/dx, any function


def source_bracket(part):
    return re.fullmatch(
        re.escape(SOURCE_HEAD) + r"(.+)" + re.escape(SOURCE_TAIL),
        L.sides(part)[1],
    ).group(1)


def source_integrand(tex, E, Ep):
    """The page's integrand of Q_h (measure dE_p)."""
    xs = sp.Symbol("x", positive=True)
    body = L.to_sympy(
        tex.replace(r"\frac{d\sigma_h}{dx}", r"\Psi(x)"),
        {"x": xs},
        {r"\Psi": PSI},
    )
    return CC * NH * NPF(Ep) / Ep * body.subs(xs, E / Ep)


@derivation("cpp.source")
def check_pp_source():
    """Q_h counts the collisions' hadrons (as Q_K), and x dsigma_h/dx is the
    fragmented quark spectrum (number, energy)."""
    first, second = L.parts(L.display(Q_PIN))
    E, Ep, xc_q, zz = sp.symbols("E E_p x_c z", positive=True)
    assert L.sides(first)[0] == "Q_h(E)", first
    xs = sp.Symbol("x", positive=True)
    tex = source_bracket(first)
    readme = source_integrand(tex, E, Ep)
    counted = CC * NH * NPF(Ep) * PSI(E / Ep) / Ep
    lhs2, rhs2 = L.sides(second)
    integral = re.fullmatch(
        r"\\sigma_\{c\\bar c\}\\int_x\^1 dz\\thinspace (.+)", rhs2
    )
    assert lhs2 == r"x\frac{d\sigma_h}{dx}" and integral, second
    integrand_c = L.to_sympy(
        integral.group(1), {"z": zz, "x": xs}, {"D_h": DF, "g": GF}
    )
    fragmented = sp.integrate(
        S_CC * GF(xc_q) / xc_q * DF(zz) * sp.DiracDelta(xs - zz * xc_q),
        (xc_q, 0, sp.oo),
    )
    # polynomial test functions (g on x_c < 1, D on z < 1)
    gt, dt = xs * (1 - xs) ** 2, zz**2 * (1 - zz)
    inner = lambda k: sp.integrate(
        sp.integrate(
            integrand_c.subs({GF(xs / zz): gt.subs(xs, xs / zz), DF(zz): dt}),
            (zz, xs, 1),
        )
        * xs ** (k - 1),
        (xs, 0, 1),
    )
    num_ = sp.integrate(dt, (zz, 0, 1)) * sp.integrate(gt / xs, (xs, 0, 1))
    energy = sp.integrate(zz * dt, (zz, 0, 1)) * sp.integrate(gt, (xs, 0, 1))
    with L.patched(
        r"\left[" + tex + SOURCE_TAIL, r"\left[x" + tex + SOURCE_TAIL
    ):
        broken = source_integrand(
            source_bracket(L.parts(L.display(Q_PIN))[0]), E, Ep
        )
    return [
        exact(
            "page integrand = the collisions' hadrons per unit E "
            "(N_p and Q_h per unit energy, as N_p and Q_K of the kaons "
            "and Kelner et al.'s eq. 71)",
            readme - counted,
        ),
        exact(
            "x dsigma_h/dx = sigma_cc int_x^1 dz D_h(z) g(x/z): "
            "the quarks' g(x_c) dx_c/x_c fragmented, x dsigma_h/dx of "
            "int dx_c (g/x_c) D_h(z) delta(x - z x_c) dz",
            sp.simplify(xs * fragmented) - S_CC * DF(zz) * GF(xs / zz),
        ),
        exact(
            "Fubini: int (x dsigma_h/dx) dx/x = sigma_cc f_h "
            "(int g dx/x = 1, int D_h dz = f_h); test functions",
            inner(0) - num_,
        ),
        exact(
            "and int (x dsigma_h/dx) dx = sigma_cc f_h <z>_h <x_c>",
            inner(1) - energy,
        ),
        exact_fails(
            "control: a factor x more in the page's bracket", broken - counted
        ),
    ]


MEASURES = (
    "a quark (and alike an anti-quark) has the distribution $",
    r"\to -y^\ast$, the distribution $",
)


def readme_measure(pin, g):
    line = L.line(pin)
    tex = line[line.index(pin) + len(pin) :].split("$")[0]
    return L.to_sympy(
        tex.replace("dx/x", r"\frac{1}{x}"),
        {"x": x_, "g": g},
        {"g": lambda a: g.subs(x_, a)},
    )


@derivation("cpp.g")
def check_g():
    """g dx/x is unchanged under x -> x_c^2/x (y* -> -y*): half the quarks lie
    above x_c, and s and x_c follow from the kinematics."""
    _, g, xc, norm = readme_closed_form()
    assert norm == r"\int_{x_c^2}^1 g(x)\thinspace\frac{dx}{x} = 1", norm
    w, ys, v = sp.symbols("w y_s v", positive=True)
    mirrored = g.subs(x_, xc_**2 / x_)
    q = 1 + w
    e_p, p_p = m_ * (q**2 + q**-2) / 2, m_ * (q**2 - q**-2) / 2
    s_kin = (e_p + m_) ** 2 - p_p**2
    e2y = (e_p + p_p) / m_  # e^{2 y_cm} of the CM frame in the lab
    xplus = mT2_ * sp.exp(ys) * sp.sqrt(e2y) / (e_p + p_p)
    xc_prime = mT2_ / sp.sqrt(m_ * (e_p + p_p))
    x_page = L.to_sympy(
        L.sides(L.inline(r"x = x_c e^{y^\ast}"))[1],
        {"x_c": xc.subs(sP_, s_kin), r"y^\ast": ys},
    )
    halves = [
        sh.moment(-1, sh.xc)
        for sh in map(pp_model().shape, (1e3, 1e6, 1e9, 1e12))
    ]
    with L.patched(
        r"P\negthinspace\left(\frac{x_c^2}{x}\right)",
        r"P\negthinspace\left(\frac{x_c}{x}\right)",
    ):
        _, g_broken, _, _ = readme_closed_form()
    pair, mirror = (readme_measure(pin, g) for pin in MEASURES)
    return [
        exact(
            "the text's distribution per pair, g(x) dx/x: the measure of the "
            "normalisation",
            pair - g / x_,
        ),
        exact("g(x_c^2/x) = g(x)", mirrored - g),
        exact(
            "and dx/x -> -dx/x: the text's g dx/x unchanged, [x_c^2, "
            "x_c] <-> [x_c, 1]",
            mirror.subs(x_, xc_**2 / x_) * sp.diff(xc_**2 / x_, x_) + mirror,
        ),
        exact(
            "s = 2 m_p(m_p + E_p) = (p_1 + p_2)^2, a proton on a "
            "proton at rest",
            readme_s().subs(Ep_, e_p) - s_kin,
        ),
        exact(
            "x+ = (E + p_z)/(E_p + p_p) of a quark of CM rapidity y* = "
            "x_c' e^{y*}, x_c' = m_T/sqrt(m_p(E_p + p_p))",
            xplus - xc_prime * sp.exp(ys),
        ),
        Result(
            "x_c'/x_c -> 1 for E_p >> m_p (the page's x_c = m_T/sqrt(s)): "
            "x = x_c e^y*, and y* -> -y* is x -> x_c^2/x",
            sp.limit((xc_prime / xc.subs(sP_, s_kin)).subs(w, v - 1), v, sp.oo)
            == 1
            and sp.limit((xplus / x_page).subs(w, v - 1), v, sp.oo) == 1,
            "sqrt(s/(m_p (E_p + p_p))) -> 1",
        ),
        Result(
            "half the quarks lie above x_c (the page's parameters, "
            "1e3-1e12 GeV)",
            all(abs(v - 0.5) < 1e-6 for v in halves),
            ", ".join(f"{v:.7f}" for v in halves),
        ),
        exact_fails(
            "control: P(x_c/x) in the page",
            g_broken.subs(x_, xc_**2 / x_) - g_broken,
        ),
    ]


CARPIO_TEX = (
    r"for protons with energy $E_p=10^{11}$ GeV incident on a fixed "
    r"proton target."
)


@derivation("cpp.fit")
def check_fit():
    """The closed form refitted to the NLO spectra gives the page's m_T, b_0,
    b_1, b_2, n_0, n_1.

    The spectra: Bhattacharya et al. 2015, Fig. 2 (at the level of their 2016
    replot), and Carpio et al. 2020, Fig. 7 (D0 at 1e11 GeV)."""

    def results():
        texts = pp_param_texts()
        q = fitted().x
        grid = fit_grid(hash(L.text()))
        return [
            Result(
                f"m_T, b_0, b_1, b_2, n_0, n_1 = {', '.join(texts)} "
                "(least squares in ln(model/data), "
                f"{sum(len(v) for v in grid.values())} points)",
                all(rounds_as(v, t) for v, t in zip(q[:6], texts)),
                ", ".join(f"{v:.6g}" for v in q[:6]),
            ),
            Result(
                "l = min(log10(E_p/1e6 GeV), 5): 5 at 1e11 GeV, "
                "the highest energy fitted",
                pp_model().l_max == math.log10(max(grid) / pp_model().e_ref),
                f"{pp_model().l_max:g}",
            ),
        ]

    ratios = berss_ratio()
    par = readme_pp_charm()
    level = search(r"lies a factor \$1/\{([\d.]+)\}\$ above", LEVEL_LINE)[0]
    out = [
        in_source(
            "Carpio et al.'s D0 curve: fixed-target protons "
            "of 1e11 GeV; their D0 fragmentation has eps = 0.101",
            CARPIO,
            CARPIO_TEX,
            r"$D^0$ & 0.577 & 0.101\\",
        ),
        Result(
            "the eps_D0 of Carpio et al.'s Table I = Kniehl & Kramer's "
            "(the page's)",
            PPD.CARPIO20_TABLE1["D0"][1]
            == PPD.KK06_TABLE1_LO_C["D0"][1]
            == par["eps"]["D0"],
            "0.101",
        ),
        Result(
            "Fig. 2 at the level of its replot in Bhattacharya et "
            f"al. 2016 (Fig. 12, 'BERSS'): x {level} (the page)",
            R_BERSS == float(level),
            "medians "
            + ", ".join(f"{np.median(r):.3f}" for r in ratios)
            + " at 1e6, 1e9 GeV",
        ),
        *results(),
    ]
    with L.patched(r"$b_1 = 0.0354$", r"$b_1 = 0.0364$"):
        out.append(must_fail("control: b_1 = 0.0364 in the page", results()))
    return out


KK_TEX = (
    r"D_c(x,\mu_0^2)&=&N\frac{x(1-x)^2}{[(1-x)^2+\epsilon x]^2},",
    r"$D^0$         & LO  & $c$ & 0.694   & $\cdots$ & $\cdots$ & 0.101    \\",
    r"$D^+$         & LO  & $c$ & 0.282   & $\cdots$ & $\cdots$ & 0.104    \\",
    r"$D_s^+$       & LO  & $c$ & 0.0498  & $\cdots$ & $\cdots$ & 0.0322   \\",
    r"$\Lambda_c^+$ & LO  & $c$ & 0.00677 & $\cdots$ & $\cdots$ & 0.00418  \\",
)


@derivation("cpp.peterson")
def check_peterson():
    """D_h: Kniehl & Kramer's Peterson form with their LO eps_h (Table I),
    normalized to the page's f_h."""
    D, norm = readme_peterson()
    assert norm == r"\int_0^1 D_h(z)\thinspace dz = f_h", norm
    N, eps, xx = sp.symbols("N epsilon x", positive=True)
    kk = L.to_sympy(
        KK_TEX[0].split("&=&")[1].rstrip(","),
        {"N": N, "x": xx, r"\epsilon": eps},
    )
    kk_eps = {
        h: float(row.split("&")[-1].replace("\\", ""))
        for h, row in zip(("D0", "D+", "Ds+", "Lc+"), KK_TEX[1:])
    }

    def results():
        par = readme_pp_charm()
        return [
            Result(
                "eps_h = "
                f"{', '.join(f'{v:g} ({h})' for h, v in par['eps'].items())}: "
                "Table I, LO, c",
                all(
                    par["eps"][h] == kk_eps[h] == PPD.KK06_TABLE1_LO_C[h][1]
                    for h in kk_eps
                ),
                "the TeX, the data module",
            )
        ]

    model = pp_model()

    def normalised():
        # mpmath computes the integral at 30 digits, independent of the model's
        # quadrature that sets N_h.
        D_, norm_ = readme_peterson()
        lhs, rhs = L.sides(norm_)
        assert lhs == r"\int_0^1 D_h(z)\thinspace dz", lhs
        f_ = sp.Symbol("f_h")
        target = L.to_sympy(rhs, {"f_h": f_})
        d = sp.lambdify((z_, eh_), D_.subs(Nh_, 1), "mpmath")
        dev = 0.0
        with mp.workdps(30):
            for h in HADRONS:
                e = mp.mpf(model.par["eps"][h])
                total = model.norm[h] * mp.quad(
                    lambda z: d(z, e), [0, 1 - mp.sqrt(e), 1]
                )
                want = float(target.subs(f_, model.frac[h]))
                dev = worst_of((dev, abs(total / want - 1)))
        return [
            Result(
                "N_h normalises int_0^1 D_h dz to the page's f_h "
                "(the photo-hadronic fractions)",
                dev < 1e-9,
                ", ".join(f"N_{h} = {model.norm[h]:.5f}" for h in HADRONS)
                + f"; max deviation {float(dev):.1e}",
            )
        ]

    out = [
        in_source("eq. (1) and Table I (LO, c) as transcribed", KK06, *KK_TEX),
        exact(
            "the page's D_h = Kniehl & Kramer's D_c (z = x, N_h = N, "
            "eps_h = eps)",
            D.subs({Nh_: N, eh_: eps}) - kk.subs(xx, z_),
        ),
        *results(),
        *normalised(),
    ]
    with L.patched(
        r"0.104 ($\mathrm{D}^+$), 0.101 ($\mathrm{D}^0$)",
        r"0.105 ($\mathrm{D}^+$), 0.101 ($\mathrm{D}^0$)",
    ):
        out.append(
            must_fail("control: eps = 0.105 (D+) in the page", results())
        )
    with L.patched(
        r"\int_0^1 D_h(z)\thinspace dz = f_h",
        r"\int_0^1 D_h(z)\thinspace dz = 1",
    ):
        out.append(
            must_fail("control: int_0^1 D_h dz = 1 in the page", normalised())
        )
    return out


BERSS16_TEX = (
    r"$10^2$ & 1.51 & $1.87 $ &1.64& $1.99$ &1.09 & 1.06\\",
    r"$10^3$ & $3.84\times 10^1$  & $4.72\times 10^1$ "
    r"& $4.03\times 10^1$ & $4.92\times 10^1$ & 1.05 & 1.04\\",
    r"$10^4$ & $2.52\times 10^2$ & $3.06\times 10^2$ "
    r"& $2.52\times 10^2$ & $3.03\times 10^2$  & 1.00 & 0.99\\",
    r"$10^5$ & $8.58\times 10^2$ & $1.03\times 10^3$ "
    r"& $8.22\times 10^2$ & $9.77\times 10^2$ & 0.96 & 0.95\\",
    r"$10^6$ &  $2.25\times 10^3$ & $2.63\times 10^3$ "
    r"& $2.10\times 10^3$ & $2.43\times 10^3$  & 0.93 & 0.92\\",
    r"$10^7$ & $5.36\times 10^3$  & $5.92\times 10^3$ "
    r"& $4.90\times 10^3$ & $5.35\times 10^3$ & 0.91 & 0.90\\",
    r"$10^8$ &  $1.21\times 10^4$ & $1.23\times 10^4$ "
    r"& $1.08\times 10^4$ & $1.09\times 10^4$  & 0.89 & 0.89\\",
    r"$10^9$ & $2.67\times 10^4$  & $2.44\times 10^4$ "
    r"& $2.35\times 10^4$ & $2.11\times 10^4$  &0.88 & 0.86\\",
    r"$10^{10}$ & $5.66\times 10^4$  & $4.67\times 10^4$ "
    r"& $4.94\times 10^4$ & $3.91\times 10^4$ & 0.87 & 0.84\\",
)
BERSS16_HEAD = (
    r"\multicolumn{2}{c|} {$\sigma (pp\rightarrow c \bar{c} X)$ [$\mu$b]}",
    r"& $M_{F,R}\propto m_T$ & $M_{F,R}\propto m_c$  & $M_{F,R}\propto m_T$ &  "
    r"$M_{F,R}\propto m_c$",
)


@derivation("cpp.sigma_table")
def check_sigma_table():
    """sigma_cc: Bhattacharya et al. 2016, Table 1 (pp, scales prop. to
    m_T) = the page's table."""

    def tex_value(cell):
        s = cell.replace("$", "").strip()
        m = re.fullmatch(r"([\d.]+)\\times10\^(\d+)", s.replace(" ", ""))
        return (
            sp.Rational(m.group(1)) * 10 ** int(m.group(2))
            if m
            else sp.Rational(s)
        )

    def column(col):
        return {
            10
            ** int(re.match(r"\$10\^\{?(\d+)\}?\$", row).group(1)): tex_value(
                row.split("&")[col]
            )
            for row in BERSS16_TEX
        }

    tex = column(1)

    def results():
        # the column of the page's scales: pp at M_F,R prop. to m_T (1) or m_c
        # (2), as the header orders them
        scale = search(
            r"\(Table 1, \$\\mathrm\{pp\}\$, scales \$\\propto (m_T|m_c)\$\)",
            r"(Table 1, $\mathrm{pp}$, scales $\propto ",
        )[0]
        sig = readme_pp_charm()["sigma"]
        return [
            Result(
                "the page's table = Table 1's column of pp at the "
                f"page's scales, M_F,R prop. to {scale} [mubarn]",
                {int(e): sp.nsimplify(v, rational=True) for e, v in sig.items()}
                == column({"m_T": 1, "m_c": 2}[scale]),
                ", ".join(f"{e:.0e} {v:g}" for e, v in sig.items()),
            )
        ]

    out = [
        in_source(
            "Table 1 as transcribed (rows 1e2-1e10 GeV; columns pp m_T, "
            "pp m_c, pA/A ...)",
            BERSS16,
            *BERSS16_TEX,
            *BERSS16_HEAD,
        ),
        Result(
            "tests/support/measured/charm_pp.py has Table 1 (BERSS16_TABLE1)",
            all(
                sp.nsimplify(PPD.BERSS16_TABLE1[float(e)][0], rational=True)
                == v
                for e, v in tex.items()
            ),
            "first column",
        ),
        *results(),
    ]
    with L.patched(r"| 1.51 | 38.4 | 252 |", r"| 1.51 | 38.5 | 252 |"):
        out.append(must_fail("control: 38.5 at 1e3 GeV in the page", results()))
    with L.patched(r"scales $\propto m_T$)", r"scales $\propto m_c$)"):
        out.append(
            must_fail("control: scales prop. to m_c in the page", results())
        )
    return out


TOP_FINDING = r"There, $\sigma_{c\bar c}$ is "


@derivation("cpp.continuation")
def check_continuation():
    """Up to 1e11 GeV, Carpio et al.'s energy, the last decade's slope
    continues."""

    def results():
        par = readme_pp_charm()
        sig, e_top = par["sigma"], par["E_top"]
        top = pp_model().sigma_cc(e_top)
        per = top * 1e-3 / sigma_inel(e_top)
        pairs = search(
            r"There, \$\\sigma_\{c\\bar c\}\$ is ([\d.]+) pairs per "
            r"inelastic collision\.",
            TOP_FINDING,
        )[0]
        # the page's rule above its table, independent of the model's expression
        words = search(
            r"the slope of the last ([\w ]+?) continues\.", TOP_LINE
        )[0]
        e_hi = max(sig)
        e_lo = e_hi / 10.0 ** DECADES[words]
        slope = math.log(sig[e_hi] / sig[e_lo]) / math.log(e_hi / e_lo)
        energies = [e_hi * (e_top / e_hi) ** (j / 4) for j in range(1, 5)]
        dev = worst_of(
            abs(pp_model().sigma_cc(e) / (sig[e_hi] * (e / e_hi) ** slope) - 1)
            for e in energies
        )
        return [
            Result(
                f"E_top = {e_top:.0e} GeV: the energy of Carpio et "
                "al.'s spectra",
                e_top == 1e11,
                "Fig. 7",
            ),
            Result(
                f"the model's sigma_cc from {e_hi:.0e} GeV to E_top = the "
                f"slope of the table's last {words}, continued "
                f"({len(energies)} energies)",
                dev < 1e-12,
                f"slope {slope:.4f}, {top:.5g} mubarn at E_top; max "
                f"deviation {dev:.1e}",
            ),
            Result(
                f"{pairs} pairs per inelastic collision there (the page)",
                rounds_as(per, pairs),
                f"{per:.3f}",
            ),
        ]

    out = [
        in_source("Carpio et al.: protons of 1e11 GeV", CARPIO, CARPIO_TEX),
        *results(),
    ]
    with L.patched(
        r"Up to $10^{11}$ GeV, the energy of Carpio",
        r"Up to $10^{12}$ GeV, the energy of Carpio",
    ):
        out.append(must_fail("control: 1e12 GeV in the page", results()))
    with L.patched(
        r"the slope of the last decade continues",
        r"the slope of the last two decades continues",
    ):
        out.append(
            must_fail(
                "control: the slope of the last two decades in the page",
                results(),
            )
        )
    return out


@derivation("cpp.threshold")
def check_pp_threshold():
    """Below 100 GeV: value and slope continuous at 100 GeV give a = 8.954, and
    sigma_cc is zero at the threshold of pp -> p Lambda_c+ anti-D0."""
    expr, rth = readme_threshold()
    sig2, sth, s, s2, a = TH_SYMS
    E, E2, mp, k = sp.symbols("E E_2 m_p k", positive=True)
    s_of = readme_s().subs(Ep_, E).subs(m_, mp)
    f = expr.subs({s: s_of, s2: s_of.subs(E, E2)})
    slope = sp.simplify((sp.diff(sp.log(f), E) * E).subs(E, E2))
    a_sol = sp.solve(sp.Eq(slope, k), a)[0]

    def results():
        par = readme_pp_charm()
        sig = par["sigma"]
        es = sorted(sig)
        kv = sp.log(
            sp.nsimplify(sig[es[1]] / sig[es[0]], rational=True), 10
        ) / sp.log(sp.Integer(es[1]) / sp.Integer(es[0]), 10)
        av = a_sol.subs(
            {k: kv, E2: sp.Integer(es[0]), mp: MASS_Q["p"], sth: rth**2}
        )
        return [
            Result(
                f"a = {par['a']}: the slope of the first decade "
                "(log-log) at 100 GeV",
                rounds_as(float(av), str(par["a"])),
                f"{float(av):.5f}",
            )
        ]

    L.line(
        r"The threshold is that of "
        r"$\mathrm{pp} \to \mathrm{p}\Lambda_c^+\bar{\mathrm{D}}^0$."
    )
    out = [
        exact(
            "sigma_cc(E_2) = sigma_2: the value continuous",
            f.subs(E, E2) - sig2,
        ),
        exact(
            "d ln sigma/d ln E_p at E_2 = a s_th 2 m_p "
            "E_2/(s_2 (s_2 - s_th)): a = k s_2 (s_2 - s_th)/(s_th 2 m_p E_2)",
            a_sol
            - k
            * s_of.subs(E, E2)
            * (s_of.subs(E, E2) - sth)
            / (sth * 2 * mp * E2),
        ),
        Result(
            "zero at s = s_th (a > 0), sqrt(s_th) = m_p + m_Lambda_c + "
            "m_D0: pp -> p Lambda_c+ anti-D0, the lightest charmed state",
            rth == MASS_Q["p"] + MASS_Q["Lc+"] + MASS_Q["D0"]
            and expr.subs({s: sth, a: 1 + sp.Symbol("q", positive=True)}) == 0,
            f"{float(rth):.5f} GeV",
        ),
        *results(),
    ]
    with L.patched(
        r"exponent $a = 8.954$ keeps", r"exponent $a = 8.854$ keeps"
    ):
        out.append(must_fail("control: a = 8.854 in the page", results()))
    return out


DELTA_LINE = r"(Kelner et al. 2006, eqs. 75–77). Each collision makes $"
FLOOR_FINDING = r"At $E_t$, the floor $\gamma_{\rm cm}m_h$ takes over for "


def readme_delta():
    g = search(
        r"Each collision makes \$([^$]+)\$ hadrons \$h\$, and as many "
        r"antiparticles, of energy "
        r"\$\\max\(([^,]+), \\gamma_\{\\rm cm\}m_h\)\$\.",
        DELTA_LINE,
    )
    tok = {
        "f_h": F_H,
        r"\sigma_{c\bar c}": S_CC,
        r"\sigma_{\rm inel}": S_INEL,
        "K_h": K_H,
        "T_p": T_P,
    }
    return L.to_sympy(g[0], tok), L.to_sympy(g[1], tok)


@derivation("cpp.delta")
def check_delta():
    """Below E_t = 1e3 GeV: f_h sigma_cc/sigma_inel hadrons of energy max(K_h
    T_p, gamma_cm m_h), continuous at E_t.

    K_h is the closed form's mean fraction of T_p per hadron at E_t. The number
    and the mean energy continue those of the closed form, except where the
    floor gamma_cm m_h takes over. Two neighboring bins share each hadron and
    keep its number and energy."""

    def results():
        par = readme_pp_charm()
        return [
            Result(
                f"E_t = {par['E_t']:.0e} GeV: the lowest energy of the "
                "published spectra (Bhattacharya et al. 2015, Fig. 2)",
                par["E_t"] == min(PPD.BERSS15_FIG2),
                f"{min(PPD.BERSS15_FIG2):.0e} GeV",
            )
        ]

    E, Eh, Et, xm, El, Eu = sp.symbols("E E_h E_t x_m E_l E_u", positive=True)
    nper, energy = readme_delta()
    delta_n = sp.integrate(nper * sp.DiracDelta(E - Eh), (E, 0, sp.oo))
    delta_e = (
        sp.integrate(E * nper * sp.DiracDelta(E - Eh), (E, 0, sp.oo)) / delta_n
    )
    K_mean = xm * Et / (Et - m_)
    model = pp_model()
    et = model.par["E_t"]
    g = model.shape(et)
    Kv = {h: model.zmean[h] * g.moment(0) * et / (et - MP) for h in HADRONS}
    xq = quad(
        lambda t: float(g(math.exp(t))) * math.exp(t),
        math.log(g.xc**2),
        0.0,
        points=[math.log(g.xc)],
        limit=400,
        epsabs=0,
        epsrel=1e-10,
    )[0]
    floor = {h: gamma_cm(et) * MASS[h] / (Kv[h] * (et - MP)) for h in HADRONS}
    w = L.to_sympy(
        L.sides(L.inline(r"w = (E_h - E_l)/(E_{l+1} - E_l)"))[1],
        {"E_h": Eh, "E_l": El, "E_{l+1}": Eu},
    )
    names = search(
        r"takes over for \$\\mathrm\{D\}\^0\$ and \$\\mathrm\{D\}\^\+\$, whose "
        r"mean energy rises there by ([\d.]+)% and ([\d.]+)%\.",
        FLOOR_FINDING,
    )
    out = [
        exact(
            "number per collision: the page's n = f_h sigma_cc/sigma_inel "
            "= the closed form's",
            delta_n - F_H * S_CC / S_INEL,
        ),
        exact(
            "mean energy at E_t: K_h T_p with K_h = <x>_h E_t/T_p(E_t) is "
            "the closed form's <x>_h E_t",
            delta_e.subs(Eh, energy).subs({K_H: K_mean, T_P: Et - m_})
            - xm * Et,
        ),
        Result(
            "K_h = <z>_h <x_c> E_t/T_p(E_t) (the page's parameters; "
            "<x_c> by Simpson and adaptive quadrature)",
            abs(g.moment(0) / xq - 1) < 1e-6,
            ", ".join(f"{h} {Kv[h]:.5f}" for h in HADRONS)
            + f"; <x_c> {xq:.6f}",
        ),
        Result(
            "the floor gamma_cm m_h at E_t over K_h T_p: above 1 for D0 "
            f"and D+, whose mean energy rises by {names[0]} % and "
            f"{names[1]} % (the page)",
            floor["D0"] > 1
            and floor["D+"] > 1
            and floor["Ds+"] < 1
            and floor["Lc+"] < 1
            and rounds_as(100 * (floor["D0"] - 1), names[0])
            and rounds_as(100 * (floor["D+"] - 1), names[1]),
            ", ".join(f"{h} {v:.4f}" for h, v in floor.items()),
        ),
        exact("the split: (1 - w) + w = 1 hadron", (1 - w) + w - 1),
        exact(
            "and (1 - w) E_l + w E_{l+1} = E_h: its energy",
            (1 - w) * El + w * Eu - Eh,
        ),
        *results(),
    ]
    with L.patched(
        r"Each collision makes $f_h\sigma_{c\bar c}/\sigma_{\rm inel}$ hadrons",
        r"Each collision makes $f_h\sigma_{c\bar c}$ hadrons",
    ):
        broken = readme_delta()[0]
        out.append(
            exact_fails(
                "control: f_h sigma_cc hadrons per collision in the page",
                sp.integrate(broken * sp.DiracDelta(E - Eh), (E, 0, sp.oo))
                - F_H * S_CC / S_INEL,
            )
        )
    with L.patched(
        r"$\boldsymbol{E_p \lt E_t = 10^3}$",
        r"$\boldsymbol{E_p \lt E_t = 10^4}$",
    ):
        out.append(must_fail("control: E_t = 1e4 GeV in the page", results()))
    with L.patched(
        r"w = (E_h - E_l)/(E_{l+1} - E_l)", r"w = (E_h - E_l)/(E_{l+1} + E_l)"
    ):
        wb = L.to_sympy(
            L.sides(L.inline(r"w = (E_h - E_l)/(E_{l+1} + E_l)"))[1],
            {"E_h": Eh, "E_l": El, "E_{l+1}": Eu},
        )
        out.append(
            exact_fails(
                "control: E_{l+1} + E_l in the split",
                (1 - wb) * El + wb * Eu - Eh,
            )
        )
    return out


GRIDPP_PIN = r"E\thinspace Q_h(E) = c\thinspace n_H\int d\ln E_p"
KIJ_PIN = r"E_iQ_h(E_i) \approx n_H\Delta\sum_j"


@derivation("cpp.kernel")
def check_kernel():
    """E Q_h and x dsigma_h/dx are integrals over ln E_p and ln z, and the
    code's sums are their rectangle rules."""
    E, Ep, lam, zz, xs, Ei, Ej, zl = sp.symbols(
        "E E_p lambda z x E_i E_j z_l", positive=True
    )
    GT, DT = xs * (1 - xs) ** 2, zz**2 * (1 - zz)  # test functions
    first, second = L.parts(L.display(GRIDPP_PIN))
    lhs, rhs = L.sides(first)
    assert lhs == r"E\thinspace Q_h(E)", lhs
    head, weight, bracket = re.fullmatch(
        r"(.+?)\\int d\\ln E_p\\thinspace (.+?)\\left\[(.+?)\\right\]_ "
        r"\{x = E/E_p\}",
        rhs,
    ).groups()
    xpsi = lambda t: L.to_sympy(
        t.replace(r"\frac{d\sigma_h}{dx}", r"\Psi(x)"),
        {"x": xs},
        {r"\Psi": PSI},
    )
    tok = {"c": CC, "n_H": NH}
    ours1 = (
        L.to_sympy(head, tok)
        * L.to_sympy(weight, {"E_p": Ep}, {"N_p": NPF})
        * xpsi(bracket).subs(xs, E / Ep)
    )
    theirs1 = (
        E
        * source_integrand(source_bracket(L.parts(L.display(Q_PIN))[0]), E, Ep)
        * Ep
    )
    lhs2, rhs2 = L.sides(second)
    assert lhs2 == r"x\frac{d\sigma_h}{dx}", lhs2
    body2 = re.fullmatch(
        r"\\sigma_\{c\\bar c\}\\int d\\ln z\\thinspace (.+)", rhs2
    ).group(1)
    ours2 = L.to_sympy(body2, {"z": zz, "x": xs}, {"D_h": DF, "g": GF})
    src2 = L.to_sympy(
        re.fullmatch(
            r"\\sigma_\{c\\bar c\}\\int_x\^1 dz\\thinspace (.+)",
            L.sides(L.parts(L.display(Q_PIN))[1])[1],
        ).group(1),
        {"z": zz, "x": xs},
        {"D_h": DF, "g": GF},
    )
    disc, kij = L.parts(L.display(KIJ_PIN))
    lhs3, rhs3 = (t.strip() for t in disc.split(r"\approx"))
    assert lhs3 == r"E_iQ_h(E_i)", lhs3
    m3 = re.fullmatch(r"n_H\\Delta\\sum_j K_\{ij\}\\thinspace (.+)", rhs3)
    summand3 = L.to_sympy(m3.group(1), {"E_j": Ej}, {"N_p": NPF})
    lhs4, rhs4 = L.sides(kij)
    assert lhs4 == "K_{ij}", lhs4
    m4 = re.fullmatch(
        r"c\\thinspace\\sigma_\{c\\bar c\}\\thinspace d\\sum_l (.+)", rhs4
    )
    summand4 = L.to_sympy(
        m4.group(1), {"z_l": zl, "E_i": Ei, "E_j": Ej}, {"D_h": DF, "g": GF}
    )
    norms = [
        L.sides(L.inline(p))
        for p in (
            r"\int g\thinspace d\ln x = 1",
            r"\int zD_h\thinspace d\ln z = f_h",
            r"d\sum_m g(e^{-md}) = 1",
            r"d\sum_l z_lD_h(z_l) = f_h",
        )
    ]
    assert [n[0] for n in norms] == [
        r"\int g\thinspace d\ln x",
        r"\int zD_h\thinspace d\ln z",
        r"d\sum_m g(e^{-md})",
        r"d\sum_l z_lD_h(z_l)",
    ], norms
    out = [
        exact(
            "E Q_h = c n_H int d ln E_p E_p N_p [x dsigma_h/dx]: "
            "the source times E, dE_p = E_p d ln E_p",
            ours1 - theirs1,
        ),
        exact(
            "x dsigma_h/dx = sigma_cc int d ln z z D_h(z) g(x/z): "
            "dz = z d ln z",
            ours2 - zz * src2,
        ),
        exact(
            "int g d ln x = int g dx/x (a test g on 1/4 < x < 1)",
            sp.integrate(GT.subs(xs, sp.exp(lam)), (lam, -sp.log(4), 0))
            - sp.integrate(GT / xs, (xs, sp.Rational(1, 4), 1)),
        ),
        exact(
            "int z D_h d ln z = int D_h dz (a test D_h on 1/4 < z < 1)",
            sp.integrate((zz * DT).subs(zz, sp.exp(lam)), (lam, -sp.log(4), 0))
            - sp.integrate(DT, (zz, sp.Rational(1, 4), 1)),
        ),
        exact(
            "K_ij's summand at z_l = e^(-lambda), E_i/E_j = x: "
            "the integrand in ln z at ln z = -lambda "
            "(the rectangle rule with the step d)",
            summand4.subs({zl: sp.exp(-lam), Ei: xs * Ej})
            - ours2.subs(zz, sp.exp(-lam)),
        ),
        exact(
            "the source's summand K_ij E_j N_p(E_j) at E_j = E_p, "
            "with K_ij -> c [x dsigma_h/dx]: the integrand in ln E_p "
            "(the rectangle rule with the step Delta)",
            NH * CC * xs * PSI(xs) * summand3.subs(Ej, Ep)
            - ours1.subs(E, xs * Ep),
        ),
        Result(
            "d sum_m g(e^(-md)) and d sum_l z_l D_h(z_l): the rectangle "
            "rules of int g d ln x and int z D_h d ln z",
            [n[1] for n in norms] == ["1", "f_h", "1", "f_h"],
            "the same normalisations",
        ),
    ]
    with L.patched(
        r"\int d\ln z\thinspace zD_h(z)", r"\int d\ln z\thinspace D_h(z)"
    ):
        rb = L.sides(L.parts(L.display(GRIDPP_PIN))[1])[1]
        broken = L.to_sympy(
            re.fullmatch(
                r"\\sigma_\{c\\bar c\}\\int d\\ln z\\thinspace (.+)", rb
            ).group(1),
            {"z": zz, "x": xs},
            {"D_h": DF, "g": GF},
        )
        out.append(
            exact_fails(
                "control: no z in the integrand over ln z", broken - zz * src2
            )
        )
    return out


LOSS_PIN = r"\sigma\kappa_C = 2\sum_h\int_0^1 x\thinspace d\sigma_h"
LOSS_LINE = (
    r"AM³'s energy-loss rate $\dot E/E$ of $\mathrm{pp}$ collisions gains $"
)


def loss_pairs():
    return int(
        search(
            r"with \$\\sigma\\kappa_C = (\d+)\\sum_h\\int_0\^1 "
            r"x\\thinspace d\\sigma_h\$\.",
            LOSS_LINE,
        )[0]
    )


@derivation("cpp.loss")
def check_loss():
    """The protons lose the energy the hadrons take: n_H c sigma kappa_C E_p per
    proton, a hadron and its antiparticle."""
    E, Ep, xs = sp.symbols("E E_p x", positive=True)
    pairs = loss_pairs()
    gain = L.to_sympy(
        search(r"gains \$([^$]+)\$, with", LOSS_LINE)[0],
        {"n_H": NH, "c": CC, r"\sigma\kappa_C": sp.Symbol("SK")},
    )
    psi = xs * (1 - xs) ** 3 + 2 * xs**2  # a test dsigma_h/dx on (0, 1)
    integrand = source_integrand(
        source_bracket(L.parts(L.display(Q_PIN))[0]), E, Ep
    ).subs(NPF(Ep), 1)
    injected = 2 * sp.integrate(
        E * integrand.replace(PSI, sp.Lambda(xs, psi)), (E, 0, Ep)
    )
    lost = (
        -gain.subs(sp.Symbol("SK"), pairs * sp.integrate(xs * psi, (xs, 0, 1)))
        * Ep
    )
    Eh, n_, dE = sp.symbols("E_h n Delta_E", positive=True)
    below = sp.integrate(
        xs * n_ * sp.DiracDelta(xs - Eh / Ep), (xs, 0, 1)
    ).subs(Ep, Eh + dE)  # E_h < E_p
    out = [
        exact(
            "the protons' loss -Edot = n_H c sigma kappa_C E_p = the "
            "energy injected into h and its antiparticle "
            "(test dsigma_h/dx)",
            sp.simplify(lost - injected),
        ),
        Result(
            "below E_t: int x dsigma_h = f_h sigma_cc E_h/E_p, "
            "the hadrons' energy over E_p",
            sp.simplify(
                below.subs(n_, F_H * S_CC) - F_H * S_CC * Eh / (Eh + dE)
            )
            == 0,
            f"{below}",
        ),
    ]
    with L.patched(
        LOSS_PIN, r"\sigma\kappa_C = 1\sum_h\int_0^1 x\thinspace d\sigma_h"
    ):
        broken = (
            -gain.subs(
                sp.Symbol("SK"),
                loss_pairs() * sp.integrate(xs * psi, (xs, 0, 1)),
            )
            * Ep
        )
        out.append(
            exact_fails(
                "control: one hadron per pair in sigma kappa_C",
                sp.simplify(broken - injected),
            )
        )
    return out


PDG26 = {
    "D+": (
        "PDG-2026_listing_D-plus-minus.pdf",
        "1869:66\x060:05OURFIT",
        "1033\x065OURAVERAGE",
    ),
    "D0": (
        "PDG-2026_listing_D-zero.pdf",
        "1864:84\x060:04OURFIT",
        "410:3\x061:0OURAVERAGE",
    ),
    "Ds+": (
        "PDG-2026_listing_Ds-plus-minus.pdf",
        "1968:35\x060:07OURFIT",
        "501:2\x062:2OURAVERAGE",
    ),
    "Lc+": (
        "PDG-2026_listing_lambdac-plus.pdf",
        "2286:46\x060:14OURFIT",
        "202:6\x061:0OURAVERAGE",
    ),
}
"""The mass (MeV) and mean life (1e-15 s) lines of the PDG 2026 listings, as
their PDFs' text writes them ("." as ":", "±" as \\x06)."""
PDG26_UNITS = ("VALUE\\(MeV\\)", "VALUE\\(10\x0015s\\)")


@derivation("ch.evolution")
def check_evolution():
    """The page's masses and mean lives are the PDG 2026 listings' (pages as
    cited), and the charged hadrons evolve as the charged kaons."""

    def value(snippet):
        return sp.Rational(
            re.match(r"([\d:]+)\x06", snippet).group(1).replace(":", ".")
        )

    def results():
        out_ = []
        for h, row in evolution_rows().items():
            name, mass, life = PDG26[h]
            pages = [pdf_page(name, s, pdg26_page) for s in (mass, life)]
            cited = list(map(int, re.findall(r"\d", row[4].split(",")[-1])))
            cited = cited * 2 if len(cited) == 1 else cited
            charged = QN[h][0] != 0
            mantissa, e = split_number(row[2])
            ok = (
                sp.Rational(row[1]) == value(mass)
                and sp.Rational(mantissa) * sp.Integer(10) ** e
                == value(life) / 10**15
                and pages in ([None, None], cited)
                and row[3]
                == ("the charged kaons" if charged else "the neutral kaons")
            )
            out_.append(
                Result(
                    f"{h}: {row[1]} MeV, {row[2]} s, listing pp. {cited}, "
                    f"as the {'charged' if charged else 'neutral'} kaons",
                    ok,
                    "not re-checked" if pages[0] is None else f"pages {pages}",
                )
            )
        return out_

    out = [
        in_pdf(
            f"{h}: the listing's mass (OUR FIT) and mean life "
            "(OUR AVERAGE), their units",
            name,
            mass,
            life,
            *PDG26_UNITS,
        )
        for h, (name, mass, life) in PDG26.items()
    ] + results()
    with L.patched(r"| 1869.66 |", r"| 1869.67 |"):
        out.append(must_fail("control: 1869.67 MeV for the D+-", results()))
    return out
