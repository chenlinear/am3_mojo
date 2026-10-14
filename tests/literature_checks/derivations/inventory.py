"""Inventory of the equations of README.md: for each, the source it is cited from, how it relates to that source, the
checks that already cover it and its derivation check in this folder, if any. It also covers the local citation index
(where present), the formulas of other codes and papers quoted in the Results of tests/literature_checks/README.md
(file "checks") and the AM3 equations of tests/reference/README.md (file "reference").

Every entry pins a piece of the equation's LaTeX, copied from the file, that must occur exactly once there. Run from
the repository root:
  .venv/bin/python tests/literature_checks/derivations/inventory.py [--gaps]
It prints the table, with the line of each equation. It fails if a display equation ($$ line) has no entry, if a pin
no longer occurs exactly once (the equation was edited: check the entry and its derivation check again), or if an
equation of README.md has no derivation check. For a model or standard entry, that check derives what follows from
it: a count, a conservation law, a limit, a continuity, the kinematics. --gaps lists only the entries that relate to
a source (transcribed, rearranged, derived, corrected) and have no derivation check yet.

Relations:
  transcribed  the source's equation in the README's notation (units, symbols)
  rearranged   rewritten or combined from several equations of the source
  derived      derived here: an integral, primitive, limit or expansion of a cited equation, or a
               consequence of kinematics stated without a source
  corrected    the source has a misprint; the README states the correct form and why
  model        a modelling or numerical choice of AM3 or am3_mojo (nothing to derive; the tests
               and the reference comparison check the code against it)
  standard     textbook kinematics or definitions
"""
import sys
from dataclasses import dataclass

import latex as L
RELATIONS = ("transcribed", "rearranged", "derived", "corrected", "model", "standard")
TO_DERIVE = ("transcribed", "rearranged", "derived", "corrected")

# Local copies of the sources (git-ignored).
SVENSSON = "scan ApJ-258-321_Svensson-1982.pdf"
HUEMMER = "TeX " + L.HUEMMER
KOERS = "TeX " + L.KOERS
LEE21 = "TeX " + L.LEE
LIPARI = "TeX " + L.LIPARI
KAFEXHIU = "TeX " + L.KAFEXHIU
KLINGER = "TeX " + L.KLINGER
JONES = "scan jones1965.pdf, NASA-X-641-64-28_Jones-1965_inverse-Compton-preprint.pdf"
BLP = "book Landau-Lifshitz vol. 4"
PDG = "PDF PDG-2026_listing_K-plus-minus.pdf, _K-zero-L.pdf"
PDG_N = "PDF PDG-2026_listing_n.pdf"
HAYEN = "TeX " + L.HAYEN
KELNER06 = "TeX " + L.KELNER
MK95 = "PostScript arXiv-astro-ph9408077v2.ps.gz"


@dataclass
class Entry:
    id: str
    pin: str
    relation: str
    source: str
    paper: str = ""
    checks: str = ""
    file: str = "README"
    note: str = ""


E = Entry
INVENTORY = [
    # e+e- annihilation
    E("ann.sinks", r"\dot n_\mp(\gamma) = -\thinspace n_\mp(\gamma)", "standard",
      "Svensson 1982 (rate definition): the annihilations counted, one e- and one e+ each"),
    E("ann.source", r"\dot n_\gamma(k) = \iint d\gamma_+", "standard",
      "Svensson 1982: the annihilations' photons counted (two each, with the pair's energy)"),
    E("ann.norm", r"$\int v\ \frac{d\sigma}{dk}\ dk = 2\langle\sigma v\rangle$", "derived",
      "follows from eqs. 18, 55-58", SVENSSON),
    E("ann.rate", r"\langle\sigma v\rangle(\gamma_+,\gamma_-) = c\thinspace \frac{G(\gamma_r^+) - G(\gamma_r^-)}",
      "rearranged", "Svensson 1982, eq. 18; the Moller angle average of sigma_rest", SVENSSON),
    E("ann.G", r"G(x) = \int_1^{x}\sqrt{t^2-1}", "derived", "Svensson 1982, eq. 18", SVENSSON),
    E("ann.sigma_rest", r"\sigma_{\rm rest}(t) = \frac{3\sigma_T}{8\thinspace (t+1)}", "transcribed",
      "Svensson 1982 (Dirac cross section); the e+e- -> gamma gamma trace", SVENSSON),
    E("ann.gstar", r"$\gamma_\ast^2 = (1 + \gamma_r)/2$", "standard", "kinematics (each lepton's energy in the CM frame)"),
    E("ann.spectrum", r"v\frac{d\sigma}{dk}(k;\gamma_+,\gamma_-) = \frac{3\sigma_T\thinspace c}{8\thinspace p_+\gamma_+\thinspace p_-\gamma_-}",
      "rearranged", "Svensson 1982, eqs. 55-57", SVENSSON),
    E("ann.spectrum_defs", r"S = \gamma_+ + \gamma_-,\qquad \gamma_s^2 = k\thinspace (S-k)", "rearranged",
      "Svensson 1982, eqs. 55-57", SVENSSON),
    E("ann.H", r"H_\pm(\gamma_\ast) = \left(2 + \frac{1-\gamma_s^2}{c_\pm}\right) I_\pm", "rearranged",
      "Svensson 1982, eqs. 56-57", SVENSSON),
    E("ann.cdu", r"c_\pm = (\gamma_\mp - k)^2 - 1,\quad d_\pm", "rearranged", "Svensson 1982, eqs. 56-57",
      SVENSSON),
    E("ann.eq58", r"**Photon spectrum** (Svensson eqs. 55–58)", "derived",
      "Svensson 1982, eq. 58 (the limit c -> 0)", SVENSSON),
    E("ann.kLU", r"$k_L = \tfrac12[\gamma_+(1-\beta_+) + \gamma_-(1-\beta_-)]$", "transcribed",
      "Svensson 1982, eqs. 33a-d", SVENSSON),
    E("ann.peaks", r"k_\pm = \tfrac{1}{2}\left[\gamma_\pm(1+\beta_\pm) + \gamma_\mp(1-\beta_\mp)\right]",
      "transcribed", "Svensson 1982, eqs. 33b-c (the peaks of eq. 59)", SVENSSON),
    E("ann.rest", r"v\frac{d\sigma}{dk}(k;\gamma) = \frac{3\sigma_T\thinspace c}{8\thinspace p\thinspace \gamma}", "transcribed",
      "Svensson 1982, eq. 41", SVENSSON),
    E("ann.gr_minus", r"$(\gamma_+^2 + \gamma_-^2 - 1)/\gamma_r^+$", "derived", "algebra (gamma_r^+ gamma_r^-)", file="checks"),
    E("ann.rest_primitives", r"$N(k) = C[1/k + (B - 2/D) L - 1/(D - k) - 2k]$", "derived",
      "primitives of eq. 41", SVENSSON, file="checks"),
    E("ann.three_photon", r"$4(\pi^2 - 9)\alpha/(3\pi) \approx 1/371$", "derived",
      "Berestetskii, Lifshitz & Pitaevskii, eqs. 89.1, 89.6", BLP, file="checks"),
    E("ann.coppi", r"$3/(8\gamma)\thinspace[\gamma^{-1/2} + \ln\gamma]$", "transcribed",
      "Mastichiadis & Kirk 1995, eq. 59 (Coppi & Blandford 1990, eq. 3.7)",
      MK95),
    E("ann.coppi_error", r"This rate is $-14\text{％}$ at $\gamma = 2.2$ and $+6\text{％}$ at $\gamma \approx 100$", "derived",
      "Mastichiadis & Kirk 1995, eq. 59 against Svensson 1982, eq. 18", MK95, file="checks"),
    E("ann.bs96", r"has $3\text{–}15\text{％}$ too many photons for Lorentz factors from 5 to $10^4$", "derived",
      "Boettcher & Schlickeiser 1996, eqs. 4-12, 16 and 33 against Svensson 1982, eq. 18", "TeX " + L.BS96,
      file="checks"),

    # triplet pair production
    E("tr.lnk", r"grows as $(28/9)\alpha r_e^2 \ln 2k$", "derived", "Lee et al. 2021, eq. 4.12 (leading term)",
      LEE21),
    E("tr.kn_ic", r"$-\dot\gamma$ $= (3/8) c\sigma_T n\varepsilon^{-1}$ $[\ln(4\gamma\varepsilon) - 11/6]$",
      "transcribed", "Blumenthal & Gould 1970, eq. 2.57; Mastichiadis 1991, eq. 28",
      "scan blumenthal1970.pdf, MNRAS-253-235_Mastichiadis-1991.pdf",
      "am3.ic_kn_limit (the exact loss tends to it)", file="checks"),
    E("tr.bg257", r"$-dE/d{t}$ $= \pi r_0^2 m^2c^5 \int d\varepsilon\ n(\varepsilon)$", "transcribed",
      "Blumenthal & Gould 1970, eq. 2.57", "scan blumenthal1970.pdf", file="checks"),
    E("tr.mast28", r"Mastichiadis writes as $\pi r_e^2 c n\varepsilon^{-1}$", "transcribed",
      "Mastichiadis 1991, eq. 28", "scan MNRAS-253-235_Mastichiadis-1991.pdf", file="checks"),
    E("tr.mast26", r"so the pair takes $5(\gamma\varepsilon)^{-3/4}$", "derived", "Mastichiadis 1991, eq. 26",
      "scan MNRAS-253-235_Mastichiadis-1991.pdf", "triplet/inelasticity.py", file="checks"),
    E("tr.lee25", r"$\eta = 1.768(s/m_e^2)^{-3/4}$", "transcribed", "Lee 1998, eq. 25",
      "TeX " + L.LEE98, "triplet/inelasticity.py", file="checks"),
    E("tr.anguelov", r"$0.195 \ln^2(2k)/k$", "transcribed", "Anguelov et al. 1999, eq. 9",
      "TeX " + L.ANGUELOV, "triplet/inelasticity.py", file="checks"),
    E("tr.ds", r"$= (4/3)\alpha_f c\sigma_T n\ (\gamma/\varepsilon)^{1/2}$", "transcribed",
      "Dermer & Schlickeiser 1991, eq. 7", "scan AA-252-414_Dermer-Schlickeiser-1991.pdf", "triplet/inelasticity.py", file="checks"),
    E("tr.k", r"$k = \gamma\varepsilon(1 - \cos\theta)$", "standard", "kinematics (gamma >> 1); the threshold k = 4"),
    E("tr.threshold", r"\frac{\sigma}{\alpha r_e^2} = \frac{\pi}{3\sqrt3}", "transcribed", "Lee et al. 2021, eq. 4.5",
      LEE21, "triplet/compare_exact.py (against the tables at k = 4.02)"),
    E("tr.high", r"\frac{\sigma}{\alpha r_e^2} = \frac{28}{9}\ln 2k - \frac{218}{27}", "transcribed",
      "Lee et al. 2021, eq. 4.13 (rounds eq. 4.12)", LEE21, "triplet/compare_exact.py (against the tables for k >= 1e3)"),
    E("tr.lee_J11", r"lacks a factor 2 in $J_{11}$. Their own differential system (eq. 4.2) requires this factor",
      "corrected", "Lee et al. 2021, eqs. 4.2-4.4", LEE21,
      file="checks"),
    E("tr.lee_exact", r"Without the factor, eq. 4.4 is $-28$ times the threshold expansion at $k = 4.001$", "derived",
      "Lee et al. 2021, eqs. 4.3-4.5, 4.12 (eq. 4.4 evaluated, with and without the factor 2 in J_11)", LEE21, file="checks"),
    E("tr.expansions", r"The threshold expansion is $10^{-3}$ below the exact cross section at $k = 4.2$", "derived",
      "Lee et al. 2021, eqs. 4.4, 4.5, 4.13 (the README's expansions against eq. 4.4)", LEE21, file="checks"),
    E("tr.haug", r"Haug's values are $2.4 \times 10^{-4}$ low on average", "derived",
      "Haug 1975, Table 1, against Lee et al. 2021, eq. 4.4", "PDF 10.1515_zna-1975-0901.pdf, " + LEE21,
      "triplet/compare_exact.py (the code against Haug's values)", file="checks"),
    E("tr.kn_cross", r"exceeds the Klein–Nishina cross section above $k = 272$", "derived",
      "Lee et al. 2021, eq. 4.4, against the Klein-Nishina cross section", LEE21, file="checks"),
    E("tr.x", r"$$x = \frac{p\cdot k}{p_1\cdot k}$$", "standard",
      "light-cone fraction; the lab energy share for gamma >> 1, eps << gamma",
      checks="triplet/x_scaling.py (independent: the QED events boosted)"),
    E("tr.x_rest", r"$x = E - p_z$", "derived", "kinematics"),
    E("tr.xi", r"$\xi = 1 - x_\mathrm{high} = x_+ + x_\mathrm{low}$", "derived", "momentum conservation"),
    E("tr.R", r"R(P) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k", "derived", "isotropic average of the rate (R and G_r)",
      checks="tests/unit/test_added_processes.py (code vs Python integration)"),
    E("tr.rate", r"interacts at the rate $c\sigma_T \int d\ln\varepsilon\ n_\gamma(\varepsilon)$", "standard",
      "the collisions with isotropic photons counted (R(P): tr.R)"),
    E("tr.pairs", r"\dot n_{+}(\gamma) = c\thinspace \sigma_T\int d\ln\gamma'", "derived",
      "the products of the collisions (charge conjugates for e+ targets)",
      checks="tests/unit/test_added_processes.py"),
    E("tr.loss", r"-\frac{\dot\gamma}{\gamma} = c\thinspace \sigma_T\int d\ln\varepsilon", "derived",
      "the mean energy xi gamma the collisions cost (treated as continuous: am3_mojo's choice)",
      checks="tests/unit/test_added_processes.py"),
    E("tr.sink", r"\dot n_\gamma(\varepsilon) = -\thinspace n_\gamma(\varepsilon)\thinspace c\thinspace \sigma_T",
      "derived", "one photon per collision", checks="tests/unit/test_added_processes.py"),
    E("tr.cc952", r"$(1 - e^{-\Delta})/\Delta = 0.952$", "derived", "AM3's Chang & Cooper regime",
      checks="tests/unit/test_added_processes.py (one short step)", file="checks"),
    E("tr.mast_cross", r"$(4\alpha_f)^{-4} = 1.4 \times 10^6$", "transcribed", "Mastichiadis 1991",
      "scan MNRAS-253-235_Mastichiadis-1991.pdf", file="checks"),

    # neutron decay
    E("nd.inputs", r"the code uses the masses and the mean life of the PDG 2026 listings", "transcribed",
      "PDG 2026, n, p and e listings and Table 1.1 (alpha); src/am3_mojo/utils/constants.mojo", PDG_N, file="checks"),
    E("nd.E0",r"E_0 = \frac{m_n^2 - m_p^2 + m_e^2}{2m_n}", "derived",
      "kinematics of n -> p e nu (the electron's largest energy); PDG 2026 masses", PDG_N),
    E("nd.spectrum", r"\frac{dN}{dE} \propto F(E)\thinspace p\thinspace E\thinspace (E_0 - E)^2", "derived",
      "Hayen et al. 2018, eq. 4 without its corrections, F their eq. 6 for gamma -> 1", HAYEN),
    E("nd.box", r"x \in \left[\frac{E^\ast - \beta_n p^\ast}{m_n}", "derived",
      "kinematics (isotropic decay, Lorentz boost of any beta)"),
    E("nd.density", r"with density $m_n/(2\beta_n p^\ast)$", "derived", "kinematics"),
    E("nd.source", r"Q_X(E) = \int \frac{dE_n}{E_n}", "derived",
      "the decays counted (time-dilated rate, x = E/E_n, densities per unit energy)"),
    E("nd.xp", r"$x_p = (m_n - E_0)/m_n$", "derived", "energy conservation (no recoil in the lepton spectra)",
      checks="tests/unit/test_added_processes.py (energy per decay)"),
    E("nd.endpoint", r"$x = 2(E_0 - m_e)/m_n = 1.664 \times 10^{-3}$", "derived", "kinematics (beta -> 1) with E_0",
      file="checks"),
    E("nd.lipari", r"(Lipari et al.: $\simeq 5.1 \times 10^{-4}$)", "transcribed", "Lipari et al. 2007, Sec. VII.C",
      LIPARI, file="checks"),
    E("nd.lipari_endpoint", r"$\simeq 2(m_n - m_p - m_e)/m_n$", "transcribed", "Lipari et al. 2007, Sec. VII.C",
      LIPARI, file="checks"),
    E("nd.huemmer", r"$Q_{\bar\nu}(E) = Q_n(E/\chi)/\chi$", "transcribed", "Hümmer et al. 2010, eq. 50", HUEMMER, file="checks"),
    E("nd.a_A", r"$A = -2\lambda(\lambda + 1)/(1 + 3\lambda^2)$", "transcribed",
      "PDG 2026, n listing (zero-recoil a0, A, B); derived from the V-A trace", PDG_N, file="checks"),
    E("nd.means", r"the electron's mean kinetic energy is 0.3011 MeV", "derived",
      "the README's spectrum (quadrature), PDG 2026 masses; mean lab fractions", PDG_N, file="checks"),
    E("nd.fermi", r"The Fermi function raises the normalised spectrum by $37\text{％}$ at 1 keV", "derived",
      "the README's spectrum with F against F = 1 (quadrature)", file="checks"),
    E("nd.fermi_rel", r"change the spectrum by at most $2.4 \times 10^{-4}$", "derived",
      "Hayen et al. 2018, eq. 6 with gamma_1 and R = sqrt(5/3) r_p (PDG 2026)", HAYEN, file="checks"),
    E("nd.trace", r"This matrix element is the V−A trace with explicit Dirac matrices, exactly.", "derived",
      "V-A trace with the proton recoiling (PDG's sign of lambda)", file="checks"),
    E("nd.recoil", r"the normalised spectra differ by up to $2 \times 10^{-3}$ below", "derived",
      "the exact three-body decay (Dalitz plot, |M|^2 of nd.trace, lambda of PDG 2026) against the README's spectrum",
      PDG_N, file="checks"),
    E("nd.sirlin", r"The change would range from $+0.4\text{％}$ at 10 keV", "derived",
      "Sirlin's outer correction, Hayen et al. 2018, eqs. 48, 50-51, on the README's spectrum", HAYEN, file="checks"),
    E("nd.bound", r"Bound decay into a hydrogen atom, $4.2 \times 10^{-6}$ of the decays", "transcribed",
      "Hayen et al. 2018 (ratio of bound to free decay)", HAYEN, file="checks"),
    E("nd.athena", r"take $m_p/m_n$ of its energy, $8 \times 10^{-7}$ less than in the code", "derived",
      "Dimitrakoudis et al. 2012, Sec. 2.2, eq. 23 and tau, against the README's x_p and tau_n",
      "TeX " + L.DIMITRAKOUDIS, file="checks"),
    E("nd.line", r"gives the antineutrinos of decays $\propto E_n^{-2}$ (per unit energy) to $3 \times 10^{-3}$", "derived",
      "Hümmer et al. 2010, eq. 50, against the README's Q_X and spectrum (power-law decays)", HUEMMER, file="checks"),

    # kaon production in photo-hadronic interactions
    E("kp.y", r"With $y = E_N \epsilon/(m_p c^2)$", "transcribed", "Hümmer et al. 2010, eq. 8", HUEMMER),
    E("kp.rate", r"t_{c,N}^{-1}(E_N) = c\int dy", "rearranged", "Hümmer et al. 2010, eqs. 3-4 with eq. 27", HUEMMER,
      "README: checked against the TeX source (no script)"),
    E("kp.source", r"Q_K(E_K) = \sum_c \sum_{N=p,n}", "rearranged", "Hümmer et al. 2010, eq. 28 (from eqs. 7, 27)",
      HUEMMER, "tests/unit/test_kaon_channels.py (code vs channels)"),
    E("kp.fc", r"f_c(y) = \sigma_c [\min(4y^2, \epsilon_\mathrm{max}^2) - \epsilon_\mathrm{min}^2]/(4y^2)",
      "derived", "Hümmer et al. 2010, eq. 27 with a constant cross section", HUEMMER),
    E("kp.eq42", r"Eq. 27 gives 3.7 μb $[1 - 1.20\ \mathrm{GeV}^2/(2y)^2]$", "corrected",
      "Hümmer et al. 2010, eqs. 27, 41-42", HUEMMER, file="checks"),
    E("kp.chi", r"\chi = \frac{s - m_Y^2 + m_K^2}{2s}\thinspace \big(1 - \beta^\ast \cos\theta\big)", "derived",
      "Hümmer et al. 2010, eq. 16 (with eqs. 5, 14-15; masses replaced)", HUEMMER,
      "tests/unit/test_kaon_channels.py (the code's chi against the data)"),
    E("kp.kstar_decay", r"together with the third of their $\mathrm{K}^{\ast 0}\Sigma^+$ that decays into", "derived",
      "isospin of K*0 -> K pi (Clebsch-Gordan)", checks="kaon_production/multibody_strangeness.py", file="checks"),
    E("kp.kstar", r"Pure $\mathrm{K}^\ast$ exchange would make it 0.5 at high energies", "derived",
      "isospin of the N K-bar* -> Sigma vertex (Clebsch-Gordan)", checks="kaon_production/two_body.py"),
    E("kp.mpi_sigma", r"The $\mathrm{K}^+ + \mathrm{K}^0$ cross section of a row is the type's cross section times its",
      "derived", "the products of the table's columns"),
    E("kp.s_tr", r"s_{t,r} = 1 - \frac{\sum_{c \in t,\thinspace K}", "model", "am3_mojo (energy bookkeeping)"),
    E("kp.chi_M4", r"share of the nucleon energy is $\chi = (E^\ast - p^\ast_L)/\sqrt{s}$", "derived", "kinematics",
      checks="kaon_production/m4_steps.py", file="checks"),
    E("kp.thresholds", r"The threshold of $\gamma\mathrm{N} \to \mathrm{K}\Lambda\pi$ is $1.152\text{–}1.168$ GeV",
      "derived", "kinematics (the README's s, PDG 2026 masses)", file="checks"),

    # kaon production in pp collisions
    E("pp.source", r"Q_K(E_K) = c\thinspace n_H\int \frac{dE_p}{E_p}", "transcribed",
      "Kelner et al. 2006, eq. 71 with F_K (AM3's pion source): the collisions counted", KELNER06,
      checks="tests/unit/test_pp_kaons.py"),
    E("pp.koers_n", r"n(\epsilon) = n_0\thinspace \epsilon^{-1}", "transcribed", "Koers et al. 2006, eqs. 16, 17a-b",
      KOERS),
    E("pp.koers_beta", r"\beta = 10^{-(\epsilon/\epsilon_3)^{p_{30}}}", "transcribed",
      "Koers et al. 2006, eqs. 17c-e", KOERS),
    E("pp.koers_params", r"with $\epsilon_3 = 10^{p_2+p_4}$, $\epsilon_5 = 10^{p_2+p_6}$", "transcribed",
      "Koers et al. 2006, eqs. 18a-e, the definitions after eq. 17; Table III", KOERS),
    E("pp.neutral", r"$= \tfrac12(F_{\mathrm{K}^0} + F_{\bar{\mathrm{K}}^0})$", "standard",
      "K0 = (KS + KL)/sqrt 2; CP violation: PDG 2026, A_L of the K0_L listing", "PDF PDG-2026_listing_K-zero-L.pdf"),
    E("pp.extension", r"F_K(x,E_p) = R(x)\thinspace \hat F_\pi(x,E_p)", "model", "am3_mojo extension",
      checks="tests/unit/test_pp_kaons.py"),
    E("pp.low_x", r"$\to R(10^{-4})$ $\rho(\min(xE_p, 10^{-4}E^\ast))/\rho(10^{-4}E^\ast)$", "model",
      "am3_mojo extension", checks="tests/unit/test_pp_kaons.py"),
    E("pp.gamma_cm", r"$\gamma_\mathrm{cm} = \sqrt{(E_p + m_p)/(2m_p)}$ $= (E_p + m_p)/\sqrt{s}$", "derived",
      "kinematics"),
    E("pp.delta", r"\tilde n_K(E_p) = \langle n_K\rangle_{\rm KPW}(E_t)", "model",
      "am3_mojo (fit to NA49's Table 11: pp.fit); the delta approximation of Kelner et al. 2006, eqs. 75-77", KELNER06,
      checks="tests/unit/test_pp_kaons.py"),
    E("pp.loss", r"$\dot E/E = -n_H c \sigma_\mathrm{inel} \sum_\pi \kappa_\pi$", "model",
      "AM3; the kaons' term is the energy of the kaons of pp.source", checks="tests/unit/test_pp_kaons.py"),
    E("pp.kernel", r"$= F_K(E_i/E_j, E_j)(E_i/E_j) c \sigma_\mathrm{inel}(E_j)$", "model", "am3_mojo (numerics)",
      checks="tests/unit/test_pp_kaons.py (indirectly: the kaons and the extra proton cooling in a source)", file="checks"),
    E("pp.kafexhiu", r"Kafexhiu et al.'s eq. 1 and the code have $[1 - (E_\mathrm{th}/T_p)^{1.9}]^3$", "corrected",
      "Kafexhiu et al. 2014, eq. 1; Klinger et al. 2024, eq. A52 (misprint)", KAFEXHIU + ", " + KLINGER, file="reference"),
    E("pp.fit", r"| Kaon | $A$ | $a$ | $b$ | $\sqrt{s_\mathrm{th}}$ [GeV] | Threshold channel |", "derived",
      "least squares to Anticic et al. 2010 (NA49), Table 11; the thresholds: kinematics", "TeX " + L.NA49),
    E("pp.code_tables", r"| | $p_{00}$ | $p_{01}$ | $p_{10}$ | $p_{11}$ |", "transcribed",
      "Koers et al. 2006, Table III (pp.koers_params), and the fit table, as src/am3_mojo/physics/proton_proton.mojo has them",
      KOERS),
    E("pp.table",r"Kaons per inelastic $\mathrm{pp}$ collision and the fraction of the proton energy they carry", "derived",
      "Koers et al. 2006, eqs. 16-18, Table III; Kelner et al. 2006, eqs. 6, 12 (extension)",
      KOERS + ", " + KELNER06, file="checks"),
    E("pp.kpw_range", r"At 158 GeV their spectra give", "derived",
      "Koers et al. 2006, eq. 15, Tables II-III; Anticic et al. 2010 (NA49), Table 11",
      KOERS + ", TeX " + L.NA49, file="checks"),
    E("pp.scaling", r"(3.5 at $10^{11}$ GeV)", "derived", "Koers et al. 2006, eqs. 16-18, Table III; the extension at E*",
      KOERS, file="checks"),
    E("pp.below", r"The normalisation to Koers et al. at 100 GeV scales the fitted", "derived",
      "the README's multiplicity fit and Anticic et al. 2010 (NA49), Table 11", "TeX " + L.NA49,
      file="checks"),
    E("pp.energy", r"*Energy:* from $E_p = 100$ GeV to $10^{12}$ GeV, all kaons together carry", "derived",
      "Koers et al. 2006, eqs. 16-18, Tables I, III; Kelner et al. 2006, eqs. 6, 12",
      KOERS + ", " + KELNER06, file="checks"),
    E("pp.sigma_pions", r"is $0.06\text{–}0.27\text{％}$ below Kafexhiu et al.'s from 6.7 GeV up", "derived",
      "Kafexhiu et al. 2014, eq. 1 at the pi+- and pi0 thresholds", KAFEXHIU, file="checks"),

    # kaon evolution
    E("ev.kinetic", r"\frac{\partial N_s}{\partial t} = -\frac{\partial}{\partial E}", "model",
      "AM3's solver; with cooling and decay alone its steady state is Lipari et al. 2007, eq. 38", LIPARI),
    E("ev.syn", r"$= (4/3)\ \sigma_T c U_B (m_e/m_K)^2 \gamma^2$", "standard",
      "synchrotron loss (Larmor), sigma_T of the kaon; Lipari et al. 2007, eq. 32", LIPARI,
      checks="tests/unit/test_added_processes.py ((m_K/m_pi)^4 scaling)"),
    E("ev.ic", r"$= (4/3)\ \sigma_T c U_\mathrm{rad} (m_e/m_K)^2 \gamma^2$", "standard",
      "Thomson loss in isotropic photons, sigma_T of the kaon", checks="tests/unit/test_added_processes.py"),

    # kaon decays
    E("kd.inputs", r"with the PDG 2026 masses and mean lives", "transcribed",
      "PDG 2026, K+-, K0, K0_S and K0_L listings (OUR FIT); src/am3_mojo/utils/constants.mojo",
      "PDF PDG-2026_listing_K-plus-minus.pdf, _K-zero.pdf, _K-zero-S.pdf, _K-zero-L.pdf"),
    E("kd.source", r"Q_X(E) = \sum_s \int_E^\infty", "standard", "decay source: the decays counted (as nd.source)"),
    E("kd.flat", r"$x \in [(E^\ast - p^\ast)/M, (E^\ast + p^\ast)/M]$ with density $M/(2p^\ast)$", "derived",
      "kinematics (isotropic decay, ultra-relativistic parent)"),
    E("kd.F", r"F(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}^{E_{\max}} \frac{g(E^\ast)}{p^\ast}", "derived",
      "kinematics"),
    E("kd.two_body", r"$E^\ast = (M^2 + m^2 - m_b^2)/(2M)$", "standard", "two-body kinematics"),
    E("kd.htilde", r"$\tilde h = (E^\ast M x - m^2)/(p^\ast M x)$", "derived", "kinematics (decay-plane average)"),
    E("kd.FRL", r"F_{R,L}(x) = \frac{M}{2}\int", "derived", "kinematics"),
    E("kd.lipari", r"the reduction of $F_{R,L}$ to Lipari et al.'s two-body formulas", "derived",
      "Lipari et al. 2007, eqs. 15-16, 18-19", LIPARI, file="checks"),
    E("kd.substitution", r"$E^\ast = m + (E_{\max} - m)\sin^2(\pi u/2)$", "model", "am3_mojo (quadrature)",
      checks="tests/unit/test_kaon_decays.py", file="checks"),
    E("kd.current", r"$H = f_+(t)(p_K + p_\pi) + f_-(t)(p_K - p_\pi)$", "standard",
      "K_l3 hadronic current: Chounet, Gaillard & Gaillard 1972, eqs. 6.7-6.8", "scan chounet1972.pdf"),
    E("kd.W", r"\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s)", "derived",
      "V-A trace; W_0: Chounet, Gaillard & Gaillard 1972, eqs. 6.7-6.9 (pp. 249-250)", "scan chounet1972.pdf"),
    E("kd.cgg", r"$\rho \propto A f_+^2 + B f_+f_- + C f_-^2$", "transcribed",
      "Chounet, Gaillard & Gaillard 1972, eq. 6.9", "scan chounet1972.pdf", file="citations"),
    E("kd.fminus", r"$f_- = (f_0 - f_+)(M^2 - m_\pi^2)/t$", "rearranged", "PDG 2026: f_0 = f_+ + f_- t/(m_K^2 - m_pi^2)",
      PDG),
    E("kd.formfactors", r"f_+(t) = 1 + \lambda'_ +\frac{t}{m_{\pi^+}^2}", "transcribed",
      "PDG 2026, K± listing p. 37 (K0_L p. 23)", PDG,
      "README: values checked against PDG"),
    E("kd.dalitz", r"\vert \mathcal M\vert^2 \propto 1 + g\thinspace u + h\thinspace u^2 + k\thinspace v^2",
      "transcribed", "PDG 2026, K± listing p. 32 (note, eq. 1)", PDG, "README: values checked against PDG"),
    E("kd.KL3pi0", r"$1 + h(u^2 + v^2/3)$", "derived", "PDG 2026 (K0_L listing pp. 20, 22) with Bose symmetry",
      "PDF PDG-2026_listing_K-zero-L.pdf"),
    E("kd.asym", r"probability $\left(1 + A_{L}\right)/2$", "standard",
      "charge asymmetry: PDG 2026, K0_L listing pp. 34-35", "PDF PDG-2026_listing_K-zero-L.pdf"),
    E("kd.bin_N", r"N = \frac{M}{2p^\ast}\left[w_g\thinspace (x_b - x_a)", "derived", "integral of F over a bin",
      file="checks"),
    E("kd.bin_E", r"E = \frac{M}{2p^\ast}\left[w_g\thinspace \frac{x_b^2 - x_a^2}{2}", "derived",
      "integral of x F over a bin", file="checks"),
    E("kd.share", r"$x_k$ receives $\left(E - x_{k+1}N\right)/(x_k - x_{k+1})$", "derived",
      "number and energy kept", checks="tests/unit/test_kaon_decays.py", file="checks"),

    # AM3: solver, synchrotron, inverse Compton, the AM3 paper's misprints
    E("am3.chang_cooper", r"$\delta = 1/w - 1/(e^w - 1)$", "transcribed",
      "Chang & Cooper 1970, eq. 18 (with eqs. 1, 15-17, 19)",
      "scan chang1970.pdf", file="citations"),
    E("am3.green", r"instead of the decaying $\exp(-\int \alpha/\vert A\vert\thinspace dx)$", "derived",
      "Green's function of the cooling equation", checks="tests/unit/test_fixes.py (regression)", file="reference"),
    E("am3.source_integral", r"$\int_x^y q(x') \exp(-\int_x^{x'} \alpha/\vert A\vert\ dx'')$", "derived",
      "Green's function of the cooling equation", checks="tests/unit/test_fixes.py, tests/solver_checks", file="reference"),
    E("am3.zfactor", r"about $z/(e^z - 1)$ times too low", "derived", "AM3's quadrature, cells of equal z", file="reference"),
    E("am3.regime4", r"are $n(y)\thinspace A(y)/A(x)$", "derived", "characteristics of the cooling equation",
      checks="tests/unit/test_fixes.py (regression)", file="reference"),
    E("am3.qs_chi", r"$\chi = \gamma B/B_\mathrm{cr}$ with $B_\mathrm{cr} = m^2c^3/(e\hbar) \propto m^2$",
      "standard", "quantum parameter", file="reference"),
    E("am3.qs_limits", r"it tends to $\chi^{-4/3}/2$, their eq. 90.27 to $0.556\ \chi^{-4/3}$", "derived",
      "Berestetskii, Lifshitz & Pitaevskii, eqs. 90.25-90.27", BLP, file="reference"),
    E("am3.qs_small", r"$1 - 2\sqrt{2}\thinspace \chi^{2/3}$ instead of $1 - (55\sqrt{3}/16)\thinspace \chi$",
      "derived", "Berestetskii, Lifshitz & Pitaevskii, eq. 90.26", BLP, file="reference"),
    E("am3.qs_const", r"$m_e c^2/B_\mathrm{cr} = 1.158 \times 10^{-8}$", "derived",
      "B_cr = m^2 c^3/(e hbar) (BLP eq. 90.3); CODATA 2022", BLP, file="reference"),
    E("am3.ic_F", r"$= \gamma$ $[f_1(\varepsilon\bar\gamma) - f_1(\varepsilon/\bar\gamma)]$", "transcribed",
      "Jones 1965, eqs. 13-15; Klinger et al. 2024, eqs. A20-A21 (the same)", JONES, file="reference"),
    E("am3.ic_loss", r"$= (3/16)\ \sigma_T c \int d\varepsilon\ n(\varepsilon)$", "rearranged",
      "Jones 1965, eq. 13 (AM3 without 1/beta)", JONES, file="reference"),
    E("am3.ic_klein_nishina", r"It is checked against a direct integration of the Klein–Nishina cross section to $10^{-13}$",
      "derived", "the Klein-Nishina cross section (Jones 1965, eq. 13)", JONES, file="reference"),
    E("am3.ic_kn_limit", r"is within $0.22\text{％}$ of its Klein–Nishina limit", "derived",
      "Jones 1965, eqs. 13-15 against Blumenthal & Gould 1970, eq. 2.57", JONES + ", scan blumenthal1970.pdf", file="checks"),
    E("am3.ic_jones13", r"the loss rate $\tfrac12\pi r_0^2 c \int d\alpha\ n(\alpha)$", "transcribed",
      "Jones 1965, eq. 13", JONES, file="citations"),
    E("am3.ic_negative", r"$\varepsilon \gtrsim (4/3)(\gamma^2 - 1)$ as $\gamma \to 1$", "derived",
      "Jones 1965, eqs. 13-15 (series)", JONES, file="reference"),
    E("am3.jones_series", r"am3_mojo uses Jones' series to $z^{10}$", "derived", "Jones 1965, eqs. 14-15, Table I",
      JONES, file="reference"),
    E("am3.f12_misprint", r"have $-2\ \mathrm{Li}_ 2(-2z)$ instead of $+2\ \mathrm{Li}_ 2(-2z)$", "corrected",
      "Klinger et al. 2024, eqs. A22-A23 (misprint); Jones 1965, eqs. 14-15", KLINGER + ", " + JONES, file="reference"),
    E("am3.ic_density_misprint", r"needs $n_\gamma(\varepsilon)/(\gamma^2\varepsilon^2)$", "corrected",
      "Klinger et al. 2024, eq. A19 (misprint); Jones 1965, eq. 13", KLINGER + ", " + JONES, file="reference"),
    E("am3.sigmaT", r"writes $\sigma_T \propto m_X^{-4}$ for the $m_X^{-2}$", "corrected",
      "Klinger et al. 2024, eq. A1 against Sec. A.2 (misprint); Jones 1965, eq. 21", KLINGER + ", " + JONES, file="reference"),
    E("am3.qs_kernel", r"has $\xi = 3/(2F_B\gamma)$ for $(3/2)F_B\gamma$", "corrected",
      "Klinger et al. 2024 (misprints); BLP eqs. 90.23-90.24", KLINGER + ", " + BLP, file="reference"),
    E("am3.syn_mX", r"has a factor $(m_X/m_e)^2$ too many", "corrected", "Klinger et al. 2024 (misprint)", KLINGER, file="reference"),
    E("am3.Rhat", r"has $99/{165}$ for $99/{162}$", "corrected", "Klinger et al. 2024 (misprint)", KLINGER, file="reference"),
    E("am3.Rtilde", r"\tilde R = [6(1 - v)(v/u) \ln(v/[2u(1 - v)])", "corrected",
      "Klinger et al. 2024 and Gao et al. 2017 (misprints); Blumenthal & Gould 1970, eq. 2.48",
      KLINGER + ", TeX " + L.GAO + ", scan blumenthal1970.pdf", file="reference"),
    E("am3.hadron_ic", r"$H(1 - \tfrac{1}{2}\gamma\varepsilon m_e/m_X)$", "corrected",
      "Klinger et al. 2024 (misprint)", KLINGER, file="reference"),
]


def display_equations(text):
    """(line number, line) of every display equation: a line holding only $$...$$."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            out.append((i, s))
    return out


def locate(entry, texts):
    """(number of occurrences of the pin in its file, line of the first), or None without the file."""
    text = texts.get(entry.file)
    if text is None:
        return None
    count = text.count(entry.pin)
    return count, (text[:text.index(entry.pin)].count("\n") + 1 if count else 0)


def derivation_checks():
    """Entry ids that have a derivation check in this folder (the registry of verify.py)."""
    import verify
    for module in verify.MODULES:
        __import__(module)
    return verify.CHECKS


def main():
    texts = {name: L.text(name) for name, path in L.FILES.items() if path.exists()}
    ids = [e.id for e in INVENTORY]
    problems = [f"duplicate id {i}" for i in sorted(set(ids)) if ids.count(i) > 1]
    problems += [f"{e.id}: unknown relation {e.relation}" for e in INVENTORY if e.relation not in RELATIONS]
    lines = {}
    for e in INVENTORY:
        found = locate(e, texts)
        if found is None:
            continue
        if found[0] != 1:
            problems.append(f"{e.id}: pin found {found[0]} times in {e.file}: {e.pin}")
        else:
            lines[e.id] = found[1]
    for i, eq in display_equations(texts["README"]):
        if not any(e.file == "README" and e.pin in eq for e in INVENTORY):
            problems.append(f"README line {i}: display equation without an inventory entry: {eq[:80]}")
    checks = derivation_checks()
    problems += [f"derivation check for unknown id {i}" for i in checks if i not in ids]
    problems += [f"{e.id}: an equation of README.md without a derivation check" for e in INVENTORY
                 if e.file == "README" and e.id not in checks]

    gaps_only = "--gaps" in sys.argv
    print("| id | line | relation | source | local copy | existing checks | derivation check |")
    print("|---|---|---|---|---|---|---|")
    for e in INVENTORY:
        derived = checks.get(e.id)
        if gaps_only and (e.relation not in TO_DERIVE or derived):
            continue
        where = f"{e.file} {lines[e.id]}" if e.id in lines else f"{e.file} (absent)"
        print(f"| {e.id} | {where} | {e.relation} | {e.source} | {e.paper or '—'} | {e.checks or '—'} | "
              f"{derived.__module__ + '.py' if derived else '—'} |")
    n = {r: sum(e.relation == r for e in INVENTORY) for r in RELATIONS}
    todo = [e for e in INVENTORY if e.relation in TO_DERIVE]
    done = [e for e in todo if e.id in checks]
    bare = [e for e in todo if e.id not in checks and not e.checks]
    print(f"\n{len(INVENTORY)} entries: " + ", ".join(f"{n[r]} {r}" for r in RELATIONS))
    print(f"{len(todo)} relate to a source; {len(done)} have a derivation check, "
          f"{len(todo) - len(done)} not yet ({len(bare)} of them without any check: "
          f"{', '.join(e.id for e in bare) or 'none'})")
    readme = [e for e in INVENTORY if e.file == "README"]
    rest = [e for e in INVENTORY if e.file != "README" and e.id not in checks]
    print(f"{sum(e.id in checks for e in readme)} of the {len(readme)} equations of README.md have a derivation check; "
          f"entries of the other files without one: {', '.join(e.id for e in rest) or 'none'}")
    if "citations" not in texts:
        print("knowledge/citations.md not found: its entries were not located")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
