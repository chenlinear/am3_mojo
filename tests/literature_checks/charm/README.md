# Charm hadrons

am3_mojo makes the charm hadrons $\mathrm D^\pm$, $\mathrm D^0$, $\bar{\mathrm D}^0$, $\mathrm D_s^\pm$, $\Lambda_c^+$ and $\bar\Lambda_c^-$ in photo-hadronic interactions and $\mathrm{pp}$ collisions. It evolves them and follows their [decays](#decays). This page starts from three kinds of sources. The first are the multi-pion interaction types of [Hümmer et al. (2010)](https://arxiv.org/abs/1002.1310). The second are measured and next-to-leading-order (NLO) charm cross sections and spectra. The third are the listings of the Particle Data Group, [PDG 2026](https://pdg.lbl.gov/2026/). The modules [charm_photoproduction.py](../../support/measured/charm_photoproduction.py) and [charm_pp.py](../../support/measured/charm_pp.py) hold the measured and NLO data, and their docstrings name each source. From these sources, the page derives step by step the equations and tables that the Mojo code uses. It also names the struct or function that uses each. The modules [charm_production.py](charm_production.py) and [charm_decays.py](charm_decays.py) check every step: `.venv/bin/python tests/literature_checks/common/run.py charm_production charm_decays`.

## Production in photo-hadronic interactions

In photo-hadronic interactions, am3_mojo adds charm hadrons to the multi-pion interaction types of [Hümmer et al.](https://arxiv.org/abs/1002.1310), in steps of the photon energy. The first subsection gives the interaction rate and the hadron source of a step. From measured cross sections and spectra, the next three subsections derive the charm pairs per interaction and the hadrons that the pairs become. They also derive the parts of the interaction that the pairs go to and the energy fractions of the hadrons. The subsection [The steps](#the-steps) collects these numbers for every step. The last subsection shows how each interaction keeps its energy.

### Steps of the multi-pion types

[Hümmer et al. (2010)](https://arxiv.org/abs/1002.1310) describe multi-pion production by interaction types (their Table 6). A type has a constant cross section over a range of the photon energy $\epsilon_r$ in the nucleon rest frame. Four of them reach the charm threshold: M4 covers 5–50 GeV, M5 50–500 GeV, M6 500–5000 GeV and M7 everything above, up to $5 \times 10^6$ GeV in AM³. Each type $t$ has the cross section $\sigma_t$ = 120 μb. am3_mojo divides their ranges above the charm threshold into ten steps, labeled $r$. A step runs from $\epsilon_{\rm min}$ to $\epsilon_{\rm max}$, with its type's cross section.

A nucleon of energy $E_{N}$ meets photons of energy $\epsilon$ in the lab frame. [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s eq. 8 defines $y = E_N\epsilon/(m_pc^2)$, with the speed of light $c$ and the proton mass $m_p$. AM³ uses this mass for both nucleons. In a head-on collision, the photon energy in the nucleon rest frame reaches its maximum, $\epsilon_r = 2y$. The response function of their eq. 27 averages the cross section up to that maximum, with the weight $\epsilon_r$. A step has the cross section $\sigma_r = \sigma_t$ from $\epsilon_{\rm min}$ to $\epsilon_{\rm max}$ and zero elsewhere, so its response function $f_r$ is a closed form:

$$
f_r(y) = \frac{1}{2y^2}\int_0^{2y} \epsilon_r\thinspace \sigma_r(\epsilon_r)\thinspace d\epsilon_r = \frac{\sigma_t}{4y^2}\left[\min(4y^2, \epsilon_{\rm max}^2) - \epsilon_{\rm min}^2\right]
$$

for $2y \ge \epsilon_{\rm min}$, and zero below. With $f_r$, their eq. 4 gives the rate $t_r^{-1}$ at which a nucleon interacts in step $r$, the inverse of a time. Their eq. 28, summed over the steps, parts and nucleons, gives the source of the hadron $h$ per unit energy:

$$
t_r^{-1}(E_N) = c\int dy\thinspace \frac{m_p c^2}{E_N}\thinspace n_\gamma\negthinspace \left(\frac{m_p c^2\thinspace y}{E_N}\right) f_r(y),\qquad Q_h(E) = \sum_r \sum_k \sum_{N=p,n} \frac{M_{k,N}^h}{\chi_k^h}\thinspace \Big[N_N\thinspace t_r^{-1}\Big]_ {E_N = E/\chi_k^h}.
$$

Here $n_\gamma$ is the number density of the photons per unit energy, and $N_{N}$ is that of the nucleons $N$, the protons and neutrons. A step splits its interactions into parts $k$: the photon side, the nucleon side and the two hadrons of associated production (see [The steps](#the-steps)). Part $k$ makes $M_{k,N}^h$ hadrons $h$ per interaction of the nucleon $N$, each with the fraction $\chi_k^h$ of the nucleon energy. The bracket is taken at $E_N = E/\chi_k^h$, the nucleon energy that gives these hadrons the energy $E$.

In the code, the methods `CStep.f`, `PhotoHadron.initialize_kernels` and `PhotoHadron.Calc_PhotoHadron_PhotonConvolution` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) compute the rate $t_r^{-1}$. The code keeps $E\thinspace dN/dE$, the densities per unit $\ln E$, on a grid of step $\Delta$ in $\ln E$. In these units, the factor $1/\chi_k^h$ cancels:

$$
E\thinspace Q_h(E) = \sum_r \sum_k \sum_{N=p,n} M_{k,N}^h\thinspace \Big[E_N N_N\thinspace t_r^{-1}\Big]_ {E_N = E/\chi_k^h}
$$

Each term is thus the nucleons' term, shifted by $\ln\chi_k^h$ in $\ln E$. Let $\ln\chi_k^h/\Delta = q + f$, with an integer $q$ and $f \in [0, 1)$. The hadrons of the nucleon bin at $E_i$ then go to the bins at $E_{i+q}$ and $E_{i+q+1}$. This fraction $f$ is a local symbol, neither the response function $f_r$ nor a hadron fraction $f_h$ (below). The upper bin gets the share $w = (e^{f\Delta} - 1)/(e^\Delta - 1)$ of the hadrons, and the lower bin the rest. This is the share of the function `number_split` in [specfun.mojo](../../../src/am3_mojo/utils/specfun.mojo), which splits the pions too. It keeps the number of hadrons and their energy. AM³ splits its pions with $f + f(1 - f)\Delta/2$, which keeps their number but not their energy. In the code, `PhotoHadron.Calc_PhotoHadron_Charm` and `pgc_work` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) add up these terms. The fraction $\chi_k^h$ holds for nucleons with the velocity $\beta \to 1$ (see [Energy fractions](#energy-fractions)). With target photons above the default limit of `set_optimize_photopion_target_photon_max`, slower nucleons interact too. With the energy fraction alone, some of their hadrons come out below their mass $m_h$. The code puts such a hadron at $m_h$ and splits it between the two bins around this energy with the same share. This split keeps the number of hadrons and their energy.

### Charm pairs per interaction

A step makes $R$ charm pairs per interaction. This number is the mean of the ratio of two cross sections over the step, with equal weight per $\ln\epsilon_r$:

$$
R(\epsilon_r) = \frac{\sigma_{c\bar c}(W)}{\sigma_{\gamma p}(W)},\qquad \sigma_{c\bar c} = A\left(1 - \frac{W_{\rm th}}{W}\right)^{n}\left(\frac{W}{10\thinspace\mathrm{GeV}}\right)^{\delta},\qquad W^2 = m_p^2 + 2m_p\epsilon_r.
$$

Here $W$ is the energy in the center-of-momentum frame, and $s = W^2$ is the squared energy of [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s eq. 5. These relations use units of $c = 1$. The cross section of charm pairs per nucleon is $\sigma_{c\bar c}$, for $\gamma\mathrm{N} \to \mathrm{c}\bar{\mathrm{c}}X$ with anything else $X$. The cross section of all photon–proton interactions is $\sigma_{\gamma p}$. The threshold $W_{\rm th}$ and the fitted parameters $A$, $n$ and $\delta$ follow below.

The threshold: The lightest final state with charm is $\Lambda_c^+\bar{\mathrm{D}}^0$. The threshold is thus $W_{\rm th} = m_{\Lambda_c} + m_{\mathrm{D}^0}$ = 4.151 GeV, with the masses of the table under [Evolution](#evolution). With the relation for $W^2$ above, charm is made above $\epsilon_r$ = 8.71 GeV, the threshold of $\gamma\mathrm{p} \to \Lambda_c^+\bar{\mathrm{D}}^0$. The first step starts there.

The total cross section $\sigma_{\gamma p}$ is the HPR1R2 fit of the COMPAS group in the review of [PDG 2016](https://doi.org/10.1088/1674-1137/40/10/100001) (Sec. 51, pp. 8–9). In that fit, the cross section of a collision of the particles $a$ and $b$ is $H\ln^2(s/s_M) + P^{ab}$ plus two Regge terms $R_i^{ab}(s_M/s)^{\eta_i}$ with $i = 1, 2$. The masses of $a$ and $b$ and a fitted mass $M$ set the scale $s_M$, and $H$, $P^{ab}$, $R_i^{ab}$ and $\eta_i$ are fitted too. For $\gamma\mathrm{p}$, the first two terms are those of $\mathrm{pp}$ times a factor that the review calls $\delta$. This page calls this factor $\delta_\gamma$ to keep it apart from the exponent $\delta$ of the charm cross section. The review's table has only the first Regge term for $\gamma\mathrm{p}$, so

$$
\sigma_{\gamma p} = \delta_\gamma\Big[H\ln^2(s/s_M) + P^{\rm pp}\Big] + R_1^{\gamma p}\left(\frac{s_M}{s}\right)^{\eta_1},\qquad s_M = (m_\rho + m_p + M)^2.
$$

The review's table gives $M$ = 2.1206 GeV, $H$ = 0.2720 mb, $\eta_1 = 0.4473$, $\delta_\gamma = 3.063 \times 10^{-3}$, $P^{\rm pp}$ = 34.41 mb and $R_1^{\gamma p}$ = 0.0139 mb. In $s_M$, the photon counts with the mass of the $\rho(770)$, $m_\rho$ = 775.26 MeV. The squared energies $s$ and $s_M$ are in GeV². The [PDG 2016](https://doi.org/10.1088/1674-1137/40/10/100001) parameters reproduce the curve that the [PDG 2026](https://pdg.lbl.gov/2026/) web page serves within 0.14%, from $\sqrt{s}$ = 25 GeV to 28 TeV.

The charm cross section $\sigma_{c\bar c}$ is fitted to the measurements from 20 GeV to HERA: $A$ = 0.590 μb, $n = 1.54$ and $\delta = 1.047$. The lowest measurement is at the photon energy of 20 GeV, and the highest are those at the electron–proton collider HERA (table below). The fit minimizes $\chi^2$, the sum of the squared deviations of the points from the curve, each in units of its error. A point has an error up and an error down, and the fit takes the one on the side of the curve. On each side, the statistical and systematic errors add in quadrature, and for H1 the error from the parton densities adds too. The module [charm_production.py](charm_production.py) refits $A$, $n$ and $\delta$, with $\chi^2 = 5.6$ for 13 degrees of freedom.

| Experiment | Target | Energy [GeV] | $\sigma_{c\bar c}$ per nucleon [μb] | Errors up, down [μb] | Data |
|---|---|---|---|---|---|
| SLAC HFP ([Abe et al. 1986](https://inspirehep.net/literature/214735)) | p | $\epsilon_r = 20$ | 0.062 | 0.017, 0.013 | [HEPData](https://www.hepdata.net/record/ins214735), Table 1 |
| PEC ([Adamovich et al. 1987](https://inspirehep.net/literature/230940)) | emulsion | $\epsilon_r = 45$ | 0.230 | 0.057, 0.057 | abstract |
| NA14/2 ([Alvarez et al. 1993](https://inspirehep.net/literature/333271)) | Si | $\epsilon_r = 100$ | 0.540 | 0.117, 0.117 | [HEPData](https://www.hepdata.net/record/ins333271), Table 3 |
| E691 ([Anjos et al. 1989](https://inspirehep.net/literature/289318)) | Be | $\epsilon_r = 145$ | 0.580 | 0.061, 0.061 | [HEPData](https://www.hepdata.net/record/ins289318), Table 9 |
| BFP ([Clark et al. 1980](https://doi.org/10.1103/PhysRevLett.45.682)) | Fe | $\epsilon_r = 100$ | 0.560 | 0.200, 0.120 | abstract |
| BFP | Fe | $\epsilon_r = 178$ | 0.750 | 0.180, 0.130 | abstract |
| E687, preliminary ([FN-555](https://doi.org/10.2172/1855270)) | Be | $\epsilon_r = 140$ | 1.260 | 0.423, 0.423 | Table 1 |
| E687, preliminary | Be | $\epsilon_r = 175$ | 0.924 | 0.301, 0.301 | Table 1 |
| E687, preliminary | Be | $\epsilon_r = 209$ | 0.889 | 0.315, 0.315 | Table 1 |
| E687, preliminary | Be | $\epsilon_r = 261$ | 0.919 | 0.386, 0.386 | Table 1 |
| E687, preliminary | Be | $\epsilon_r = 292$ | 1.440 | 0.706, 0.706 | Table 1 |
| ZEUS ([Derrick et al. 1995](https://arxiv.org/abs/hep-ex/9502002)) | p | $W = 163$ | 6.3 | 6.67, 3.72 | Table 2 |
| ZEUS | p | $W = 243$ | 16.9 | 14.84, 9.96 | Table 2 |
| H1 ([Aid et al. 1996](https://arxiv.org/abs/hep-ex/9604005)), tagged | p | $W = 200$ | 13.2 | 10.36, 5.55 | Table 3 |
| H1, untagged | p | $W = 142$ | 9.3 | 7.46, 4.23 | Table 3 |
| H1, untagged | p | $W = 230$ | 20.6 | 16.91, 9.86 | Table 3 |

The energy column gives the photon energy $\epsilon_r$ for the fixed-target beams and $W$ for the HERA bins. For PEC, NA14/2 and E691, this energy is the middle of the range. E691 assume a cross section that grows as the power 0.93 of the nucleon number of the target. BFP's values are those of virtual photons, extrapolated to zero virtuality. E687 measured $\mathrm{D}^{\ast\pm}$ on beryllium. [FN-555](https://doi.org/10.2172/1855270) (pp. 15–16) converts this to charm pairs per nucleon, dividing by 0.604 $\mathrm{D}^{\ast\pm}$ per pair and by $9.01^{0.93}$, with a further error of 5%. Here 9.01 stands for the nucleon number of beryllium, and 0.93 is E691's power. The column Data names the source of each value: a table of its HEPData record, the abstract or a table of the paper. The table's values are those of [charm_photoproduction.py](../../support/measured/charm_photoproduction.py). The module [charm_production.py](charm_production.py) compares them with their HEPData records and with the TeX sources of ZEUS and H1.

Above $W$ = 275 GeV, where the HERA data end, $\sigma_{c\bar c}$ follows the IP-SAT dipole-model curve of [Gonçalves, Machado & Meneses (2009)](https://arxiv.org/abs/0905.2067) (their Fig. 1). IP-SAT stands for impact-parameter saturation. The curve is scaled to the fit at 275 GeV. Beyond the figure's end at 1 TeV, it continues as the power law of its last 0.3 decade in $W$. Of the three curves of Gonçalves et al. as published, IP-SAT describes the HERA points best.

### Species and shares

This section derives how the charm pairs of a step split. Their quarks become the hadrons $h$ with the fractions $f_h$. The pairs go to the parts of the interaction. Associated production takes the share $h_a$, the nucleon side takes $h_N$ and the photon side takes the rest.

Species: A charm quark becomes $\mathrm{D}^0$, $\mathrm{D}^+$, $\mathrm D_s^+$ or $\Lambda_c^+$ with the fractions $f_h$ of [Lisovyi et al. (2016)](https://arxiv.org/abs/1509.01061). An anti-quark becomes the antiparticles with the same fractions. Their Table 12, column "Constrained S", averages $\mathrm{e}^+\mathrm{e}^-$, ep and pp data. That column constrains the sum $S$ of all fractions to one and gives 0.6086, 0.2404, 0.0802 and 0.0623. These add up to 0.9915, and the charm-strange baryons make the rest, 0.136 per $\Lambda_c^+$. Leaving those out, am3_mojo renormalizes to one hadron: 0.614, 0.242, 0.081 and 0.063. The constant `FRAG_D` in [constants.mojo](../../../src/am3_mojo/utils/constants.mojo) and the three after it hold these fractions.

Associated production, $\gamma\mathrm{N} \to \Lambda_c^+\bar{\mathrm{D}}X$, makes a share $h_a$ of the pairs. [Abe et al. (1986)](https://inspirehep.net/literature/214735) found $h_a = 0.71$ at 20 GeV, from the excess of $\bar{\mathrm{D}}$ over $\mathrm{D}$ decays. At 175 GeV, FOCUS measured the asymmetry $\mathcal{A} = 0.111$ of $\Lambda_c^+$ over $\bar\Lambda_c^-$ ([Link et al. 2004](https://arxiv.org/abs/hep-ex/0311022)). Both energies are photon energies $\epsilon_r$. Per pair, associated production makes $h_a$ $\Lambda_c^+$ and no $\bar\Lambda_c^-$, and the other pairs make $(1 - h_a)f_{\Lambda_c}$ of each. In all, a pair makes $N_+ = h_a + (1 - h_a)f_{\Lambda_c}$ $\Lambda_c^+$ and $N_- = (1 - h_a)f_{\Lambda_c}$ $\bar\Lambda_c^-$, so

$$
\mathcal{A} = \frac{N_+ - N_-}{N_+ + N_-} = \frac{h_a}{h_a + 2(1 - h_a)f_{\Lambda_c}},\qquad h_a = \frac{2\mathcal{A}f_{\Lambda_c}}{1 - \mathcal{A} + 2\mathcal{A}f_{\Lambda_c}},
$$

with the fraction $f_{\Lambda_c} = 0.0623/0.9915 = 0.0628$ above, before rounding. This gives $h_a = 0.0154$ at 175 GeV. A power law in $\epsilon_r$ through both values gives $h_a = 0.71\thinspace(\epsilon_r/\epsilon_a)^{-1.765}$ with $\epsilon_a$ = 20 GeV, at most 1. It reaches 1 below 16.5 GeV.

Resolved photons interact through their quarks and gluons. They put a share $h_N$ of the pairs on the nucleon side, where the hadrons move with the nucleon. [Frixione et al. (1995)](https://arxiv.org/abs/hep-ph/9412348) give the NLO cross sections of the point-like photon (their Table 1) and of the resolved photon (their Table 2). Their fifth column has the charm mass 1.5 GeV and the renormalization scale $\mu = m_c$, where $m_c$ is that mass. That column uses the parton densities MRSA for the proton and GRV-HO for the photon:

| $W$ [GeV] | 30 | 100 | 280 |
|---|---|---|---|
| Point-like [μb] | 1.090 | 2.826 | 5.858 |
| Resolved [μb] | 0.099 | 0.673 | 2.435 |
| Resolved share | 0.0833 | 0.1923 | 0.2936 |

The resolved share is the resolved cross section over the sum of the two cross sections. The share $h_N$ is half the resolved share, interpolated in $\ln W$. It is held above $W$ = 280 GeV. At low energies, it falls to zero at 12 GeV, on the line through its values at 30 and 100 GeV. This zero lies at the photon energy $\epsilon_r$ = 76 GeV. The photon side has the other pairs, the share $1 - h_a - h_N$.

### Energy fractions

Each part gives its hadrons a fraction $\chi$ of the nucleon energy. The photon side takes it from measured spectra, and the nucleon side takes it from $\mathrm{pp}$ collisions. Associated production uses two sources. Its $\Lambda_c^+$ takes the fraction from the multi-pion types, and its anti-D takes it from the photon side.

Photon side: In the center-of-momentum frame, a hadron has the energy $E^\ast$ and the momentum $p_L^\ast$ along the photon. The nucleon has the energy $E_N^\ast$ and the momentum $p_N^\ast$. Take a frame where the nucleon moves with the velocity $\beta \to 1$ (in units of $c$), as in the lab. The nucleon's Lorentz factor is $\gamma$. In this frame, the hadron has the energy $\gamma(E^\ast - \beta p_L^\ast)$, and the nucleon has $\gamma(E_N^\ast + \beta p_N^\ast)$, where $E_N^\ast + p_N^\ast = \sqrt{s}$. The ratio of the two energies, the hadron's energy fraction, thus tends to $(E^\ast - p_L^\ast)/\sqrt{s}$. On the photon side, $\chi = \langle E^\ast - p_L^\ast\rangle/\sqrt{s}$, with $s = W^2$. The angle brackets mark the mean over the hadrons. The mean is over the D-meson spectra of E691 at 145 GeV ([Anjos et al. 1989](https://inspirehep.net/literature/289318)). These spectra are their $d\sigma/dx_F$ and $d\sigma/dp_T^2$ ([HEPData](https://www.hepdata.net/record/ins289318) Tables 11 and 10). Here $p_T$ is the transverse momentum, and $x_F$ is Feynman's $x$, which E691 define as $p_L^\ast$ over its maximum. For that maximum, am3_mojo takes half the center-of-momentum energy: $x_F = 2p_L^\ast/\sqrt{s}$. An invariant cross section $F(x_F)\thinspace h(p_T^2)$ reproduces both spectra exactly, with $F$ constant in each $x_F$ bin and $h$ in each eighth of a $p_T^2$ bin. Such a cross section is $E^\ast d^3\sigma/dp^3$, the same in every frame. Its factor $h$ is a function here, not a hadron label. For a hadron of mass $m$ and transverse mass $m_T$, with $m_T^2 = m^2 + p_T^2$, this gives

$$
d\sigma = \pi F(x_F)\thinspace h(p_T^2)\thinspace \frac{dx_F\thinspace dp_T^2}{q},\qquad q = \frac{2E^\ast}{\sqrt{s}} = \sqrt{x_F^2 + a^2},\qquad a = \frac{2m_T}{\sqrt{s}},\qquad \chi = \frac{q - x_F}{2}.
$$

The shorthands $q$ and $a$ hold only in this paragraph. The $x_F$ integrals of each bin are thus closed forms, with the inverse hyperbolic sine arsinh:

$$
\int \frac{dx_F}{q} = \mathrm{arsinh}\frac{x_F}{a},\qquad \int \chi\thinspace \frac{dx_F}{q} = \frac{x_F - q}{2}.
$$

At other energies, $F$ and $h$ stay the same (Feynman scaling). Every species takes the D mesons' $F$ and $h$, with its own mass. The $x_F$ bins end where the hadron reaches its largest energy, $E^\ast = (s + m^2 - m_X^2)/(2\sqrt{s})$, against its lightest recoil of mass $m_X$. That recoil is the lightest set of particles that completes the charge, baryon number, charm and strangeness of $\gamma\mathrm{p}$:

| Hadron | $\mathrm{D}^0$ | $\mathrm{D}^+$ | $\mathrm D_s^+$ | $\Lambda_c^+$ | $\bar{\mathrm{D}}^0$ | $\mathrm{D}^-$ | $\mathrm D_s^-$ | $\bar\Lambda_c^-$ |
|---|---|---|---|---|---|---|---|---|
| Lightest recoil | $\bar{\mathrm{D}}^0\mathrm{p}$ | $\bar{\mathrm{D}}^0\mathrm{n}$ | $\mathrm D_s^-\mathrm{p}$ | $\bar{\mathrm{D}}^0$ | $\Lambda_c^+$ | $\Lambda_c^+\pi^+$ | $\Lambda_c^+\mathrm{K}^+$ | $\Lambda_c^+\mathrm{p}$ |

Nucleon side: Its $\chi$ comes from $\mathrm{pp}$ collisions at the proton energy $E_p = \epsilon_r$, with the quark spectra and fragmentation of [Production in pp collisions](#production-in-pp-collisions). It is the mean energy fraction $x$, in units of $E_p$, of the hadrons whose quarks have $y^\ast \gt 0$. Here $y^\ast$ is the rapidity in the center-of-momentum frame, and a quark of energy fraction $x$ has $y^\ast \gt 0$ for $x \gt x_c$ (below). A hadron takes the share $z$ of its quark's energy, with the distribution $D_h(z)$ of its species. This distribution does not depend on the quark's $x$. With the quarks' distribution $g$ (below), the mean is thus

$$
\chi_N = \bar z\thinspace \frac{\int_{x_c}^1 g(x)\thinspace dx}{\int_{x_c}^1 g(x)\thinspace dx/x},\qquad \bar z = \sum_h f_h\langle z\rangle_h,
$$

where $\langle z\rangle_h$ is the mean of $z$ over $D_h$, and $\bar z$ is the mean over the species. The published spectra start at the proton energy $E_t$. For lower $\epsilon_r$, $\chi_N$ takes the form of the $\delta$-functional approximation of $\mathrm{pp}$ collisions (below). This form uses the mean energy fraction of the forward hadrons at $E_t$.

Associated production: The associated $\Lambda_c^+$ takes the energy fraction of the type's nucleon. The nucleon keeps one minus the inelasticity $K$ of [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s Table 6. This fraction is 0.51 for M4, 0.55 for M5 and 0.56 for M6 and M7. The associated anti-D takes the mean of the photon side's $\chi$ of $\bar{\mathrm{D}}^0$ and $\mathrm{D}^-$, weighted by their fractions.

### The steps

The ten steps fill M4–M7 from the threshold to $5 \times 10^6$ GeV. For each step, the table lists the means of the numbers above. These are the pairs $R$, the shares $h_a$ and $h_N$ and the energy fractions $\chi$ of the parts.

| Type, $\epsilon_r$ [GeV] | $R$ | $h_a$ | $h_N$ | $\chi$ of $\mathrm{D}$, $\mathrm D_s$, $\Lambda_c$ (photon side) | $\chi$ of the associated anti-D | $\chi$ (nucleon side) |
|---|---|---|---|---|---|---|
| M4, 8.71–15 | $1.049 \times 10^{-4}$ | 1 | 0 | — | 0.3027 | — |
| M4, 15–30 | $6.556 \times 10^{-4}$ | 0.5884 | 0 | 0.1961, 0.2093, 0.2486 | 0.2166 | — |
| M4, 30–50 | $1.600 \times 10^{-3}$ | 0.2182 | 0 | 0.1423, 0.1511, 0.1788 | 0.1468 | — |
| M5, 50–150 | $3.710 \times 10^{-3}$ | 0.05347 | 0.006119 | 0.08039, 0.08518, 0.1007 | 0.09086 | 0.1266 |
| M5, 150–500 | $8.741 \times 10^{-3}$ | 0.007343 | 0.03075 | 0.03723, 0.03941, 0.04655 | 0.04273 | 0.08143 |
| M6, 500–1500 | 0.01733 | $9.727 \times 10^{-4}$ | 0.05628 | 0.01733, 0.01831, 0.02153 | 0.01938 | 0.07122 |
| M6, 1500–5000 | 0.03176 | $1.304 \times 10^{-4}$ | 0.08246 | 0.008173, 0.00862, 0.01008 | 0.009288 | 0.05159 |
| M7, $5000\text{–}5 \times 10^4$ | 0.07543 | $7.42 \times 10^{-6}$ | 0.1275 | 0.002672, 0.002812, 0.003263 | 0.003956 | 0.03065 |
| M7, $5 \times 10^4\text{–}5 \times 10^5$ | 0.1349 | $1.588 \times 10^{-7}$ | 0.1468 | $7.218 \times 10^{-4}$, $7.578 \times 10^{-4}$, $8.739 \times 10^{-4}$ | 0.00102 | 0.02012 |
| M7, $5 \times 10^5\text{–}5 \times 10^6$ | 0.1799 | $2.75 \times 10^{-9}$ | 0.1468 | $1.895 \times 10^{-4}$, $1.987 \times 10^{-4}$, $2.282 \times 10^{-4}$ | $2.654 \times 10^{-4}$ | 0.01434 |

Each entry is a mean over the step, with equal weight per $\ln\epsilon_r$. The means of $h_a$ and $h_N$ use $R$ as the weight. The means of $\chi$ use the pairs of their part as the weight, that is, $R$ times the share of the part. The $\chi$ of $\mathrm{D}$ averages $\mathrm{D}^0$, $\mathrm{D}^+$ and their antiparticles, weighted by their fractions. That of the associated anti-D weights $\bar{\mathrm{D}}^0$ and $\mathrm{D}^-$ by their fractions too. The $\chi$ of $\mathrm D_s$ and that of $\Lambda_c$ average the particle and its antiparticle.

A dash marks a part without pairs. The module [charm_production.py](charm_production.py) recomputes every entry from these formulas and the data.

Per interaction of a proton, the parts of a step make the hadrons below. These are the multiplicities $M_{k,N}^h$ and energy fractions $\chi_k^h$ of the source for $N = \mathrm p$:

| Part | Hadrons per interaction | $\chi$ |
|---|---|---|
| Photon side | $R(1 - h_a - h_N)f_h$ of each hadron $h$ and of its antiparticle | its column of the photon side |
| Nucleon side | $R h_N f_h$ of each hadron $h$ and of its antiparticle | the nucleon side's |
| Associated anti-D | $Rh_af_{\mathrm{D}^0}/(f_{\mathrm{D}^0} + f_{\mathrm{D}^+})$ $\bar{\mathrm{D}}^0$, $Rh_af_{\mathrm{D}^+}/(f_{\mathrm{D}^0} + f_{\mathrm{D}^+})$ $\mathrm{D}^-$ | the associated anti-D's |
| Associated $\Lambda_c^+$ | $R h_a$ | the type's nucleon fraction |

From neutrons, the associated anti-D is a $\mathrm{D}^-$ where protons give a $\bar{\mathrm{D}}^0$, and vice versa, so that $\gamma\mathrm{n} \to \Lambda_c^+\mathrm{D}^-$ keeps the charge. The other parts are the same as for protons. Each interaction makes $R$ charm quarks and $R$ anti-quarks. The associated $\Lambda_c^+$ comes in addition to the type's nucleon. It is an extra baryon, which adds the baryon number $R h_a$ per interaction, at most $4 \times 10^{-4}$. In the code, `charm_production_steps`, `CStep`, `CPart` and `_pairs` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) build the steps.

### Energy bookkeeping

[Hümmer et al.](https://arxiv.org/abs/1002.1310)'s multi-pion types put the whole inelastic energy into pions. As for the kaons, the charm hadrons take their energy from the type's pions. In the photon energies of step $r$, am3_mojo scales the pion multiplicities of the step's type $t$ by the factor

$$
s_{t,r} = 1 - \frac{\sum_k \sum_h \chi_k^h\thinspace M_{k,N}^h}{\sum_{c \in t} \chi_c\thinspace M_{c,N}^\pi}
$$

for the nucleon $N$. The numerator is the energy fraction that the charm hadrons of the step take per interaction. In the denominator, the index $c$ (not the speed of light) runs over the type's two entries in Table 6, L and H. These entries are the type's pions of lower and of higher energy. Each entry has the energy fraction $\chi_c$ and makes $M_{c,N}^\pi$ pions of all charges. When kaon production is on, the kaons of the same photon energies add to the numerator. The code subtracts $\kappa_r f_r$ from the type's response function $f_t$, with $\kappa_r = 1 - s_{t,r}$. Because the step has its type's cross section, $f_t - \kappa_r f_r$ is the response function of the cross section $s_{t,r}\sigma_t$ in the step and $\sigma_t$ elsewhere. Each interaction thus keeps its energy. The methods `PhotoHadron._set_pion_cuts` and `PhotoHadron.Calc_PhotoHadron_Rad` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) do this bookkeeping.

## Production in pp collisions

In $\mathrm{pp}$ collisions, a closed form fitted to the published NLO spectra gives the energy distribution of the charm quarks. Fragmentation functions turn these quarks into hadrons.

The charm hadrons $h$ of $\mathrm{pp}$ collisions have the source

$$
Q_h(E) = c\thinspace n_H\int \frac{dE_p}{E_p}\thinspace N_p(E_p)\left[\frac{d\sigma_h}{dx}\right]_ {x = E/E_p},\qquad x\frac{d\sigma_h}{dx} = \sigma_{c\bar c}\int_x^1 dz\thinspace D_h(z)\thinspace g\negthinspace\left(\frac{x}{z}\right),
$$

with the number density $n_H$ of the target protons and the density $N_p$ of the protons per unit energy. The first relation counts the hadrons of the collisions, as for the kaons. In the second relation, $\sigma_{c\bar c}$ is the cross section of charm pairs, this time in $\mathrm{pp}$ collisions. Here $x$ is an energy fraction in the lab, in units of $E_p$. In $d\sigma_h/dx$, it is the hadron's fraction $E/E_p$, and in $g$, it is a charm quark's fraction $E_c/E_p$. A hadron takes the share $z$ of its quark's energy, so its quark has the fraction $x/z$. Per pair, a quark (and alike an anti-quark) has the distribution $g(x)\thinspace dx/x$. This distribution is a closed form, fitted to the NLO spectra of [Bhattacharya et al. (2015)](https://arxiv.org/abs/1502.01076) (Fig. 2) and [Carpio et al. (2020)](https://arxiv.org/abs/2007.07945) (Fig. 7):

$$
g(x) = A\thinspace P(x)\thinspace P\negthinspace\left(\frac{x_c^2}{x}\right),\qquad P(u) = \frac{(1-u)^n}{1 + u/x_0},\qquad x_c = \frac{m_T}{\sqrt{s}},\qquad \int_{x_c^2}^1 g(x)\thinspace\frac{dx}{x} = 1,
$$

$$
\log_{10} x_0 = b_0 + b_1 l + b_2 l^2,\qquad n = n_0 + n_1 l,\qquad l = \min\negthinspace\left(\log_{10}\frac{E_p}{10^6\thinspace\mathrm{GeV}}, 5\right),
$$

with $s = 2m_p(m_p + E_p)$, $m_T$ = 1.66 GeV, $b_0 = -1.2836$, $b_1 = 0.0354$, $b_2 = -0.0171$, $n_0 = 2.957$ and $n_1 = 0.126$. Here $s$ is the squared center-of-momentum energy, $m_T$ is a fitted transverse mass of the charm quark, and $A$ normalizes the distribution. This $A$ and the exponent $n$ are new symbols, not those of the photo-hadronic charm cross section. The function $P$ takes $x$ or $x_c^2/x$ as its argument $u$. Its shape parameters $x_0$ and $n$ vary with the proton energy through $l$, the decimal logarithm of $E_p/10^6$ GeV, capped at 5. For $E_p \gg m_p$, a quark of rapidity $y^\ast$ in the center-of-momentum frame has $x = x_c e^{y^\ast}$. Under $x \to x_c^2/x$, that is $y^\ast \to -y^\ast$, the distribution $g\thinspace dx/x$ is unchanged. The spectrum is thus symmetric in rapidity, as it must be in $\mathrm{pp}$ collisions. A quark becomes the hadron $h$ with the Peterson fragmentation function of [Kniehl & Kramer (2006)](https://arxiv.org/abs/hep-ph/0607306). This function is their eq. 1, with the leading-order values of their Table I,

$$
D_h(z) = N_h\thinspace\frac{z(1-z)^2}{[(1-z)^2 + \epsilon_h z]^2},\qquad \int_0^1 D_h(z)\thinspace dz = f_h,
$$

with Peterson's parameter $\epsilon_h$ = 0.104 ($\mathrm{D}^+$), 0.101 ($\mathrm{D}^0$), 0.0322 ($\mathrm D_s^+$) and 0.00418 ($\Lambda_c^+$), and the same fractions $f_h$ as in photo-hadronic interactions. This $\epsilon_h$ is not a photon energy. The constant $N_h$ normalizes $D_h$. An anti-quark makes the antiparticles in the same way.

The fit: [Bhattacharya et al. (2015)](https://arxiv.org/abs/1502.01076)'s Fig. 2 lies a factor $1/{0.697}$ above the same calculation as replotted in 2016. The replot is Fig. 12 of [Bhattacharya et al. (2016)](https://arxiv.org/abs/1607.00193). The fit uses the level of the replot, and it uses [Carpio et al. (2020)](https://arxiv.org/abs/2007.07945)'s $\mathrm{D}^0$ curve only for its shape, up to $x = 0.35$. Above that value, their curve has a shoulder that they attribute to the extrapolated parton densities. The module [charm_production.py](charm_production.py) refits $g$ to these spectra and finds the parameters above. For the $\mathrm{D}^0$ curve, the fit folds the quark distribution with $D_h$.

The cross section is the NLO $\sigma(\mathrm{pp} \to \mathrm{c}\bar{\mathrm{c}}X)$ of [Bhattacharya et al. (2016)](https://arxiv.org/abs/1607.00193) (Table 1, $\mathrm{pp}$, scales $\propto m_T$). In that column, the factorization and renormalization scales are proportional to $m_T$, with $m_T^2 = p_T^2 + m_c^2$. This $m_T$ is the quark's transverse mass, not the fitted one above. am3_mojo interpolates the table linearly in log–log between their energies:

| $E_p$ [GeV] | $10^2$ | $10^3$ | $10^4$ | $10^5$ | $10^6$ | $10^7$ | $10^8$ | $10^9$ | $10^{10}$ |
|---|---|---|---|---|---|---|---|---|---|
| $\sigma_{c\bar c}$ [μb] | 1.51 | 38.4 | 252 | 858 | 2250 | 5360 | 12100 | 26700 | 56600 |

Up to $10^{11}$ GeV, the energy of [Carpio et al.](https://arxiv.org/abs/2007.07945)'s spectra, the slope of the last decade continues. There, $\sigma_{c\bar c}$ is 0.91 pairs per inelastic collision. Above that energy, its ratio to AM³'s inelastic cross section $\sigma_{\rm inel}$ stays at this value. Below 100 GeV,

$$
\sigma_{c\bar c} = \sigma_2\left[\frac{1 - s_{\rm th}/s}{1 - s_{\rm th}/s_2}\right]^{a},\qquad \sqrt{s_{\rm th}} = m_p + m_{\Lambda_c} + m_{\mathrm{D}^0},
$$

where $\sigma_2$ and $s_2$ are the cross section and $s$ at 100 GeV, and the exponent $a = 8.954$ keeps the slope there. The threshold is that of $\mathrm{pp} \to \mathrm{p}\Lambda_c^+\bar{\mathrm{D}}^0$.

$\boldsymbol{E_p \lt E_t = 10^3}$ **GeV (below the published spectra).** As for the kaons, a $\delta$-functional approximation applies ([Kelner et al. 2006](https://arxiv.org/abs/astro-ph/0606058), eqs. 75–77). Each collision makes $f_h\sigma_{c\bar c}/\sigma_{\rm inel}$ hadrons $h$, and as many antiparticles, of energy $\max(K_hT_p, \gamma_{\rm cm}m_h)$. The approximation thus puts all hadrons $h$ of a collision at one energy, a $\delta$ function in energy. Here $T_p = E_p - m_p$ is the proton's kinetic energy. The factor $K_h$ is the mean energy of a hadron $h$ in the closed form at $E_t$, in units of $T_p$. This $K_h$ is a new symbol, not the inelasticity $K$ of the types. The floor $\gamma_{\rm cm}m_h$ is the energy of a hadron of mass $m_h$ at rest in the center-of-momentum frame. That frame moves with the Lorentz factor $\gamma_{\rm cm}$. At $E_t$, the floor $\gamma_{\rm cm}m_h$ takes over for $\mathrm{D}^0$ and $\mathrm{D}^+$, whose mean energy rises there by 3.4% and 4.1%. The code splits each hadron between the two grid bins around its energy $E_h$. These bins lie at $E_l$ and $E_{l+1}$, where $l$ numbers the bins. The upper bin gets the share $w = (E_h - E_l)/(E_{l+1} - E_l)$ of the hadron, and the lower bin gets the rest. This split keeps the number and the energy of the hadrons.

On the grid: In the units of the grid, $E\thinspace dN/dE$ per unit $\ln E$, the source and the fragmentation become

$$
E\thinspace Q_h(E) = c\thinspace n_H\int d\ln E_p\thinspace E_pN_p(E_p)\left[x\frac{d\sigma_h}{dx}\right]_ {x = E/E_p},\qquad x\frac{d\sigma_h}{dx} = \sigma_{c\bar c}\int d\ln z\thinspace zD_h(z)\thinspace g\negthinspace\left(\frac{x}{z}\right).
$$

The code computes both integrals with the rectangle rule. The first runs over the proton bins $j$, at the energies $E_j$. The second runs over the points $z_l = e^{-ld}$ of a finer grid, with the step $d = \Delta/4$:

$$
E_iQ_h(E_i) \approx n_H\Delta\sum_j K_{ij}\thinspace E_jN_p(E_j),\qquad K_{ij} = c\thinspace\sigma_{c\bar c}\thinspace d\sum_l z_lD_h(z_l)\thinspace g\negthinspace\left(\frac{E_i}{E_jz_l}\right).
$$

Here $E_i$ is the energy of the hadron bin $i$, and the kernel $K_{ij}$ links it to the proton bin $j$. The kernel takes the cross section $\sigma_{c\bar c}$ at $E_j$, and the factor $d$ in front of a sum is the step, not a differential. On the finer grid, $d\sum_m g(e^{-md}) = 1$ and $d\sum_l z_lD_h(z_l) = f_h$ normalize $g$ and $D_h$, as $\int g\thinspace d\ln x = 1$ and $\int zD_h\thinspace d\ln z = f_h$ do. The integers $m$ and $l$ number the points of the finer grid, and this $l$ is not that of the closed form. The functions `pp_charm_shape`, `pp_charm_fragmentation` and `pp_charm_sigma_cc` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) give $g$, $z D_h$ and $\sigma_{c\bar c}$. In the code, `ProtonProton._init_charm_kernels`, `ProtonProton.Calc_PP_CharmSrc` and `ppc_work` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) build and apply the kernels.

Energy bookkeeping: am3_mojo makes the charm hadrons in addition to AM³'s pions. As for the kaons, the protons thus also lose the energy that these hadrons take. AM³'s energy-loss rate $\dot E/E$ of $\mathrm{pp}$ collisions gains $-n_H c\thinspace\sigma\kappa_C$, with $\sigma\kappa_C = 2\sum_h\int_0^1 x\thinspace d\sigma_h$. Here $\dot E$ is the time derivative of the proton energy. The product $\sigma\kappa_C$ is the cross section times the fraction $\kappa_C$ of $E_p$ that the hadrons and their antiparticles carry. The factor 2 counts the antiparticles. Below $E_t$, the integral is $f_h\sigma_{c\bar c}$ times the hadron's energy over $E_p$. The method `ProtonProton.Calc_PP_ProtonLoss` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) adds this term.

## Evolution

am3_mojo evolves the charm hadrons like the kaons ([Kaons](../kaons/README.md)), with AM³'s solver and the masses and mean lives of [PDG 2026](https://pdg.lbl.gov/2026/):

| Hadron | Mass [MeV] | Mean life [s] | Evolved as | [PDG 2026](https://pdg.lbl.gov/2026/) listing |
|---|---|---|---|---|
| $\mathrm{D}^\pm$ | 1869.66 | $1.033 \times 10^{-12}$ | the charged kaons | $\mathrm{D}^\pm$, p. 1 |
| $\mathrm{D}^0$, $\bar{\mathrm{D}}^0$ | 1864.84 | $4.103 \times 10^{-13}$ | the neutral kaons | $\mathrm{D}^0$, p. 1 |
| $\mathrm D_s^\pm$ | 1968.35 | $5.012 \times 10^{-13}$ | the charged kaons | $\mathrm D_s^\pm$, pp. 1–2 |
| $\Lambda_c^+$, $\bar\Lambda_c^-$ | 2286.46 | $2.026 \times 10^{-13}$ | the charged kaons | $\Lambda_c^+$, pp. 1–2 |

The masses are the listings' fits, and the mean lives are their averages. The charged hadrons lose energy to synchrotron and inverse-Compton radiation, as the kaons do but with their own mass. They also cool adiabatically and escape at the charged pions' rate, and the neutral $\mathrm{D}^0$ and $\bar{\mathrm{D}}^0$ escape at the neutrons' rate. Expansion and AM³'s user-set extra cooling act on all of them, as on the kaons. All of them decay: a hadron of mean life $\tau$ and Lorentz factor $\gamma$ lives $\gamma\tau$ on average in the lab. The constant `N_CHARM` in [particles.mojo](../../../src/am3_mojo/model/particles.mojo) and the indices after it list the eight species. The methods `SimulationManager._mk_charm`, `SimulationManager.Calc_Coeff_C` and `SimulationManager.Calc_C` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) run the evolution. The functions `charm_mass`, `charm_lifetime` and `charm_charged` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) give the masses, mean lives and charges.

## Decays

The charm hadrons $\mathrm D^+$, $\mathrm D^0$, $\mathrm D_s^+$ and $\Lambda_c^+$ decay by the modes of the [PDG 2026](https://pdg.lbl.gov/2026/) listings. Their antiparticles decay by the CP-conjugate modes, with C for charge conjugation and P for parity. This section derives the code's decay tables from these listings and from the V−A (vector minus axial-vector) matrix elements of the weak interaction. It first writes the source of the products, with one table per hadron and product. It then builds the tables from the modes and their matrix elements. The energy distributions in the parent frame follow, then the boost to the lab and the code's grids.

The code is in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) and [lambda_c_semileptonic.mojo](../../../src/am3_mojo/physics/lambda_c_semileptonic.mojo). The module [charm_decays.py](charm_decays.py) checks the steps of this section. Its checks use the listings as [pdg_charm.py](pdg_charm.py) transcribes them. That module lists each mode with its [PDG](https://pdg.lbl.gov/2026/) entry, page and rule.

### Source of the decay products

A charm hadron of species $s$, energy $E_{C}$, Lorentz factor $\gamma_C$ and mean life $\tau_s$ decays at the rate $1/(\gamma_C\tau_s)$. Here and below, $s$ labels the species, not a squared energy. Each decay gives $F_{s\to X}(x)$ products $X$ per unit of their energy fraction $x = E/E_C$. Their source is that of the kaon decays ([Kaons](../kaons/README.md)), with $s$ a charm species. The code works per unit $\ln E$: with the density $n_s$ per unit $\ln E_C$, the source per unit $\ln E$ is

$$
E\thinspace Q_X(E) = \sum_s \int_E^\infty \frac{dE_C}{E_C}\thinspace\frac{n_s(E_C)}{\gamma_C\thinspace\tau_s}\thinspace T_{s\to X}\negthinspace\left(\frac{E}{E_C}\right),\qquad n_s = E_CN_s,\qquad T_{s\to X}(x) = x\thinspace F_{s\to X}(x),
$$

where $N_s$ is the density per unit energy and $T_{s\to X}$ counts the products per decay per unit $\ln x$. The hadron grid and the product grids are logarithmic with the same step $\Delta$. The ratio of a product's grid energy $E_i$ to a hadron's $E_j$ is thus a point $x_k = e^{-k\Delta}$ of the tables. Here $k$ counts the grid steps between the two energies. The integral becomes

$$
E_i\thinspace Q_X(E_i) = \Delta\sum_s\sum_j T_{s\to X}\negthinspace\left(\frac{E_i}{E_j}\right)\frac{n_s(E_j)}{\gamma_j\thinspace\tau_s},
$$

with $\gamma_j$ the Lorentz factor at $E_j$. The methods `CharmDecay.initialize_kernels` and `CharmDecay.Calc_CtoX` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluate this sum. With $T$ short for $T_{s\to X}$, the sum counts $\Delta\sum_k T(x_k)$ products per decay. They carry the share $\Delta\sum_k x_kT(x_k)$ of the parent's energy. The tables hold the exact number and energy of the products, so the source keeps both.

### Decay modes

The modes are those of the [PDG 2026](https://pdg.lbl.gov/2026/) listings, chosen and normalized by these rules:

- Semileptonic: every exclusive mode above 0.1% that no other mode contains, with its non-resonant complement where [PDG](https://pdg.lbl.gov/2026/) lists one. They are scaled so that the $\mathrm{e}^+$ modes add up to PDG's inclusive $\mathrm{e}^+$ rate. The $\mu^+$ modes get the same factor. Here and below, $\ell$ stands for $\mathrm e$ or $\mu$. The mode $\Lambda_c^+ \to \mathrm{n}\ell^+\nu$, which PDG does not list, has the rates of [Meinel (2018)](https://arxiv.org/abs/1712.05783), eqs. 20–21. If PDG lacks a $\mu^+$ mode of the $\mathrm D_s^+$, that mode equals the matching $\mathrm{e}^+$ mode.
- Leptonic: $\mathrm D^+$ and $\mathrm D_s^+$ $\to \mu^+\nu$, $\tau^+\nu$. The $\tau^+$ is left-handed in the parent frame. It decays by its eleven modes above 0.5%, renormalized.
- Hadronic: every mode above 0.5%, renormalized to the rest of the width. A $\mathrm{K}^0_\mathrm{S}$ counts twice, as a $\mathrm{K}^0$ or $\bar{\mathrm{K}}^0$. From a multi-hadron total, the part that another listed mode makes through an $\eta$, $\eta'$, $\omega$, $\phi$ or $\mathrm{K}^\ast$ is subtracted.

The $\mathrm K^0_\mathrm S$ rule makes the assumption $\Gamma(\mathrm K^0) = 2\Gamma(\mathrm K^0_\mathrm S)$ that the $\mathrm D^\pm$ listing names (p. 2): a neutral kaon is half a $\mathrm K^0_\mathrm S$ and half a $\mathrm K^0_\mathrm L$. Here $\Gamma$ is the partial width of the decays into a final state with that kaon. The function `charm_decay_modes` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) holds the modes. The table shows what the rules make of them, with the branching ratios as fractions and Γn for entry n of a listing. The scale factor brings the exclusive semileptonic modes up to [PDG](https://pdg.lbl.gov/2026/)'s inclusive rate, and the hadronic modes get the rest of the width. The hadronic modes cover 60–78% of the rates before the renormalization.

| | $\mathrm D^+$ | $\mathrm D^0$ | $\mathrm D_s^+$ | $\Lambda_c^+$ |
|---|---|---|---|---|
| Semileptonic modes, $\mathrm e^+$ and $\mu^+$ | 10 and 10 | 5 and 6 | 6 and 6 | 2 and 2 |
| Sum of their $\mathrm e^+$ branching ratios | 0.155826 | 0.06219 | 0.06156 | 0.0397 |
| [PDG](https://pdg.lbl.gov/2026/)'s inclusive $\mathrm e^+$ rate | 0.1607 (Γ1, p. 2) | 0.0649 (Γ5, p. 15) | 0.0633 (Γ1, p. 3) | 0.0406 (Γ116, p. 5) |
| Scale factor of the semileptonic modes | 1.0313 | 1.0436 | 1.0283 | 1.0227 |
| Semileptonic branching ratio, scaled | 0.3178 | 0.1269 | 0.1252 | 0.0803 |
| Leptonic branching ratio | 0.001392 (Γ15 + Γ16) | 0 | 0.05927 (Γ21 + Γ22) | 0 |
| Hadronic modes | 21 | 29 | 30 | 32 |
| Sum of their branching ratios | 0.5973 | 0.7782 | 0.7067 | 0.7076 |
| Renormalization factor of the hadronic modes | 1.1397 | 1.1218 | 1.1541 | 1.2998 |

### Semileptonic modes and their matrix elements

The table lists the semileptonic modes with [PDG](https://pdg.lbl.gov/2026/)'s branching ratios before the scaling and with the inputs of their matrix elements (below). Its columns $\mathcal B_\mathrm e$ and $\mathcal B_\mu$ hold the branching ratios of the $\mathrm e^+$ and $\mu^+$ modes. Parentheses mark the S-wave or non-resonant complement of a resonant mode. The $\mathrm D^+$ entries are on pp. 2–3 of its listing. Those of the $\mathrm D^0$ are on p. 16, of the $\mathrm D_s^+$ on p. 4 and of the $\Lambda_c^+$ on p. 5. The mode $\Lambda_c^+ \to \Lambda(1520)\mathrm e^+\nu$, at 0.1% exactly, is not above the threshold.

| Parent | Mode | $\mathcal B_\mathrm e$ | $\mathcal B_\mu$ | [PDG 2026](https://pdg.lbl.gov/2026/) | Matrix element |
|---|---|---|---|---|---|
| $\mathrm D^+$ | $\bar{\mathrm K}^0\ell^+\nu$ | 0.0881 | 0.0868 | Γ17, Γ18 | $\alpha = 0.309$ |
| | $\bar{\mathrm K}^{\ast 0}\ell^+\nu$ | 0.0540 | 0.0526 | Γ29, Γ36 | $r_V = 1.47$, $r_2 = 0.786$ |
| | $(\mathrm K^-\pi^+)\ell^+\nu$ | 0.00228 | 0.0019 | Γ25, Γ35 | phase space |
| | $(\bar{\mathrm K}^0\pi^0)\ell^+\nu$ | 0.001206 | 0.00128 | 2Γ20, 2Γ31 | phase space |
| | $\pi^0\ell^+\nu$ | 0.00372 | 0.00350 | Γ44, Γ45 | $\alpha = 0.279$ |
| | $\eta\ell^+\nu$ | 0.00111 | 0.00104 | Γ46, Γ47 | $\alpha = 0$ |
| | $\rho^0\ell^+\nu$ | 0.00187 | 0.00164 | Γ50, Γ53 | $r_V = 1.53$, $r_2 = 0.82$ |
| | $(\pi^+\pi^-)\ell^+\nu$ | 0.00062 | 0.00072 | Γ49, Γ52 | phase space |
| | $\omega\ell^+\nu$ | 0.00169 | 0.00177 | Γ54, Γ55 | $r_V = 1.24$, $r_2 = 1.06$ |
| | $\mathrm K^-\pi^+\pi^0\ell^+\nu$ | 0.00123 | 0.00109 | Γ37, Γ40 | phase space |
| $\mathrm D^0$ | $\mathrm K^-\ell^+\nu$ | 0.03537 | 0.03418 | Γ20, Γ21 | $\alpha = 0.309$ |
| | $\mathrm K^{\ast -}\ell^+\nu$ | 0.0216 | 0.0201 | Γ22, Γ23 | $r_V = 1.45$, $r_2 = 0.720$ |
| | $(\bar{\mathrm K}^0\pi^-)\ell^+\nu$ | 0.00085 | 0.00075 | Γ26, Γ30 | phase space |
| | $(\mathrm K^-\pi^0)\ell^+\nu$ | — | 0.000409 | Γ28 | phase space |
| | $\pi^-\ell^+\nu$ | 0.00291 | 0.00267 | Γ39, Γ40 | $\alpha = 0.279$ |
| | $\rho^-\ell^+\nu$ | 0.00146 | 0.00135 | Γ42, Γ43 | $r_V = 1.53$, $r_2 = 0.82$ |
| $\mathrm D_s^+$ | $\phi\ell^+\nu$ | 0.0234 | 0.0224 | Γ26, Γ29 | $r_V = 1.76$, $r_2 = 0.83$ |
| | $\eta\ell^+\nu$ | 0.0227 | 0.0224 | Γ30, Γ32 | $\alpha = 0$ |
| | $\eta'\ell^+\nu$ | 0.0081 | 0.0080 | Γ31, Γ33 | $\alpha = 0$ |
| | $\mathrm K^0\ell^+\nu$ | 0.00288 | 0.00288 | Γ35 | $\alpha = 0$ |
| | $\mathrm K^{\ast 0}\ell^+\nu$ | 0.00205 | 0.00205 | Γ36 | $r_V = 1.67$, $r_2 = 0.77$ |
| | $f_0(980)\ell^+\nu$ | 0.00243 | 0.00243 | Γ39 + Γ40 | $\alpha = 0$ |
| $\Lambda_c^+$ | $\Lambda\ell^+\nu$ | 0.0356 | 0.0348 | Γ109, Γ115 | lattice, [Meinel 2017](https://arxiv.org/abs/1611.09696) |
| | $\mathrm n\ell^+\nu$ | 0.00410 | 0.00400 | [Meinel 2018](https://arxiv.org/abs/1712.05783) | lattice, Meinel 2018 |

Within `charm_decay_modes`, the helpers `_sl_pp`, `_sl_pv`, `_sl_ps` and `x_bl3` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) make the modes of each row. Three-body semileptonic modes use the V−A matrix element $\mathcal M$. For a massless neutrino, the lepton current of $\ell^+\nu$ and a hadronic current $J$ give the $\vert\mathcal M\vert^2$ below. The subscript "sum" marks the sum over the lepton's spin, and the equation leaves out the factor $G_F^2\vert V_{cq}\vert^2/2$. Here the current $J = J_r + iJ_i$ has the real four-vectors $J_r$ and $J_i$ as its parts. Fermi's constant is $G_{F}$, and $V_{cq}$ is the element of the Cabibbo–Kobayashi–Maskawa (CKM) matrix for $\mathrm c \to \mathrm d, \mathrm s$:

$$
\vert\mathcal M\vert^2_{\rm sum} = 8\left[2(p_\nu\cdot J_r)(p_\ell\cdot J_r) + 2(p_\nu\cdot J_i)(p_\ell\cdot J_i) - (p_\nu\cdot p_\ell)(J_r^2 + J_i^2)\right] + 16\thinspace\epsilon(J_r, J_i, p_\nu, p_\ell).
$$

Here $p_\nu$ and $p_\ell$ are the four-momenta of the neutrino and the charged lepton, and the dots are Minkowski products. The function $\epsilon(a, b, c, d)$ $= \epsilon_{\mu\nu\rho\sigma}a^\mu b^\nu c^\rho d^\sigma$ contracts four four-vectors with the totally antisymmetric (Levi-Civita) tensor, with $\epsilon_{0123} = +1$. This $\epsilon$ is not a photon energy. A lepton of mass $m_\ell$ and spin four-vector $s$ (here not a species label) then has the squared matrix element

$$
\vert\mathcal M\vert^2(s) = \tfrac12\thinspace\vert\mathcal M\vert^2_{\rm sum}(p_\ell \to p_\ell + m_\ell s),
$$

where the arrow marks a substitution in $\vert\mathcal M\vert^2_{\rm sum}$. This spin rule gives the helicities of the muons. Because $\vert\mathcal M\vert^2_{\rm sum}$ is linear in $p_\ell$, the two helicities differ by $\vert\mathcal M\vert^2_{\rm sum}(p_\ell \to m_\ell s)$, with $s$ along the lepton's momentum.

Pseudoscalar or scalar meson: In a mode $\mathrm{P} \to \mathrm{P}'\ell^+\nu$, both mesons are pseudoscalar, or the second is the scalar $f_0(980)$. [Richman & Burchat (1995)](https://arxiv.org/abs/hep-ph/9508250) write the hadronic current as $H = f_+(t)(p + p') + f_-(t)(p - p')$ (their eq. 82). Here $p$ and $p'$ are the four-momenta of the two mesons, and $M$ and $m$ are their masses. The squared four-momentum of the lepton pair is $t = (p - p')^2$, a new use of that letter. Their eq. 83 defines the scalar form factor $f_0$, which gives $f_- = (f_0 - f_+)(M^2 - m^2)/t$. This $f_0$ is a form factor, not the meson $f_0(980)$. The current is real: it is $J_r$ of the trace, with $J_i = 0$. The bracket above is then the density $W_0$ of the kaons' $\mathrm K_{\ell 3}$ decays, $\mathrm K \to \pi\ell\nu$ ([Kaons](../kaons/README.md)). The functions `pl3_density` and `l3_density` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluate it. The form factors $f_+$ and $f_0$ are pole forms, with the pole masses $m_{\rm p}$ and $m_0$. The upright p stands for pole, not for the proton. The modified pole's $\alpha$ is from [Ablikim et al. 2015](https://arxiv.org/abs/1508.07560) for $\mathrm{D} \to \mathrm{K}$ and $\mathrm{D} \to \pi$, and 0 otherwise:

$$
f_+(t) = \frac{1}{(1 - t/m_{\rm p}^2)(1 - \alpha t/m_{\rm p}^2)},\qquad f_0(t) = \frac{1}{1 - t/m_0^2}.
$$

The form factor $f_+$ is the modified pole of [Ablikim et al.](https://arxiv.org/abs/1508.07560) (their eq. II.4), divided by its value at $t = 0$. The form factor $f_0$ is [Richman & Burchat](https://arxiv.org/abs/hep-ph/9508250)'s single pole (eq. 88 with the power $n = 1$). Both are 1 at $t = 0$, as their eq. 81 requires, so $f_{-}$ stays finite there. For $\mathrm D \to \mathrm K$, the tables of Ablikim et al. give $\alpha = 0.309$ (Tables XIII and XV), and their text 0.310 (eq. VI.25). The code uses the tables' value. For $\mathrm D \to \pi$ both give 0.279.

Vector meson: In a mode $\mathrm{P} \to \mathrm{V}\ell^+\nu$, the vector meson $\mathrm V$ has the mass $m$, four-momentum $p'$ and polarization vector $\varepsilon$. [Richman & Burchat](https://arxiv.org/abs/hep-ph/9508250)'s eqs. 85–86 give the current

$$
\langle\mathrm V\vert V^\mu - A^\mu\vert\mathrm P\rangle = \frac{2i\epsilon^{\mu\nu\alpha\beta}}{M + m}\varepsilon^\ast_\nu p'_ \alpha p_\beta V - (M + m)\varepsilon^{\ast\mu}A_1 + \frac{\varepsilon^\ast\cdot q}{M + m}(p + p')^\mu A_2 + \frac{2m\thinspace\varepsilon^\ast\cdot q}{q^2}q^\mu(A_3 - A_0),\qquad A_3 = \frac{(M + m)A_1 - (M - m)A_2}{2m},
$$

with $q = p - p'$. Here $V^\mu$ and $A^\mu$ are the vector and axial-vector currents, and $i$ is the imaginary unit. The star marks the complex conjugate, and repeated Lorentz indices are summed. The polarization vector $\varepsilon$ is not the tensor $\epsilon$. [Richman & Burchat](https://arxiv.org/abs/hep-ph/9508250) state no convention for $\epsilon^{0123}$. With the convention above, the trace gives their eq. 113 for D decays. There, V−A makes the helicity amplitude $H_{-}$ (below) dominate. With the other sign, the trace exchanges $H_+$ and $H_{-}$. The hadronic current has the form factors $V$, $A_1$, $A_2$ and $A_0$, where $V$ is not the meson $\mathrm V$. These are single poles with the ratios $r_V = V(0)/A_1(0)$ and $r_2 = A_2(0)/A_1(0)$ of [PDG 2026](https://pdg.lbl.gov/2026/):

$$
V(t) = \frac{r_V}{1 - t/m_V^2},\qquad A_1(t) = \frac{1}{1 - t/m_A^2},\qquad A_2(t) = \frac{r_2}{1 - t/m_A^2},\qquad A_0(t) = \frac{A_0(0)}{1 - t/m_0^2}.
$$

Here $m_V$, $m_A$ and $m_0$ are the pole masses of $V$, of $A_1$ and $A_2$, and of $A_0$. This $m_0$ is not the pole mass of $f_0$. As above, $t = q^2$. Eq. 86 comes with $A_0(0) = A_3(0)$, which keeps the last term of the current finite at $q^2 = 0$. Hence $A_0(0) = [(M + m) - (M - m)r_2]/(2m)$. The ratios are [PDG](https://pdg.lbl.gov/2026/)'s averages, or single measurements where there is no average ($\mathrm D^\pm$ listing pp. 65–67, $\mathrm D^0$ p. 129, $\mathrm D_s^\pm$ pp. 59–62). The decay $\mathrm D^0 \to \rho^-$ takes the average of the $\mathrm D^\pm$ listing. The function `vl3_density` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) sums the trace over the polarizations of $\mathrm V$. In the rest frame of $\ell^+\nu$, with $\theta_\ell$ the lepton's angle to the direction opposite to $\mathrm V$, this gives

$$
\sum_\varepsilon\vert\mathcal M\vert^2_{\rm sum} = 2(q^2 - m_\ell^2)\left[(1 + \cos\theta_\ell)^2H_+^2 + (1 - \cos\theta_\ell)^2H_-^2 + 2\sin^2\theta_\ell\thinspace H_0^2 + \frac{m_\ell^2}{q^2}\left(\sin^2\theta_\ell\thinspace(H_+^2 + H_-^2) + 2(H_0\cos\theta_\ell - H_t)^2\right)\right],\qquad H_t = \frac{2Mp_\mathrm VA_0}{\sqrt{q^2}}.
$$

The helicity amplitudes $H_\pm$ and $H_0$ are those of [Richman & Burchat](https://arxiv.org/abs/hep-ph/9508250)'s eqs. 115–116, and $H_t$ carries $A_0$. Here $p_\mathrm V$ is the momentum of $\mathrm V$ in the parent frame. Integrated over the angles of the $\mathrm V$ decay, their eq. 113 for D decays is the bracket without the lepton mass, up to a constant. The terms in $m_\ell^2$ follow from the trace.

The poles are charmed mesons with the spin and parity $J^P$ of each part of the current. [Richman & Burchat](https://arxiv.org/abs/hep-ph/9508250) assign $J^P = 1^-$ to $f_+$ and $V$, $0^+$ to $f_0$ and $1^+$ to $A_1$ and $A_2$ (pp. 73–74). The form factor $A_0$ is the time-like part of the axial current. It takes $0^-$, as the baryon form factor $g_0$ (below) does in Table IV of [Meinel (2018)](https://arxiv.org/abs/1712.05783). The code takes $\mathrm D^\ast(2010)$, $\mathrm D_0^\ast(2300)$, $\mathrm D_1(2420)$ and $\mathrm D$ for $\mathrm c \to \mathrm d$, and $\mathrm D_s^\ast$, $\mathrm D_{s0}^\ast(2317)$, $\mathrm D_{s1}(2460)$ and $\mathrm D_s$ for $\mathrm c \to \mathrm s$, with their [PDG 2026](https://pdg.lbl.gov/2026/) masses.

Baryon: The modes $\Lambda_c^+ \to \mathrm B\ell^+\nu$ have the baryon $\mathrm B = \Lambda$ or $\mathrm n$. They use the form factors that [Meinel (2017)](https://arxiv.org/abs/1611.09696) and [Meinel (2018)](https://arxiv.org/abs/1712.05783) computed in lattice quantum chromodynamics (lattice QCD). Meinel defines the helicity form factors $f_+$, $f_\perp$, $f_0$ and $g_+$, $g_\perp$, $g_0$ by the vector and axial currents of his 2018 eqs. 1–2. His $f_+$ and $f_0$ are not those of the mesons. With $p$ and $p'$ the four-momenta and $M$ and $m_\mathrm B$ the masses of $\Lambda_c^+$ and $\mathrm B$, the V−A current regroups into

$$
J^\mu = \bar u_\mathrm B\left[F\gamma^\mu + aP^\mu + bq^\mu + \gamma_5\left(G\gamma^\mu + cP^\mu + dq^\mu\right)\right]u_{\Lambda_c},\qquad P = p + p',\qquad q = p - p',\qquad s_\pm = (M \pm m_\mathrm B)^2 - q^2,
$$

where $u_{\Lambda_c}$ and $\bar u_\mathrm B$ are the Dirac spinors of the two baryons. The bar marks the Dirac adjoint. The Dirac matrices are $\gamma^\mu$ and $\gamma_5$. The coefficients $F, G, a, b, c, d$ are new symbols. They collect the form factors:

$$
F = f_\perp,\qquad G = g_\perp,\qquad a = \frac{(M + m_\mathrm B)(f_+ - f_\perp)}{s_+},\qquad b = (M - m_\mathrm B)\left[\frac{f_0 - f_+}{q^2} - \frac{a}{M + m_\mathrm B}\right],\qquad c = \frac{(M - m_\mathrm B)(g_+ - g_\perp)}{s_-},\qquad d = (M + m_\mathrm B)\left[\frac{g_0 - g_+}{q^2} - \frac{c}{M - m_\mathrm B}\right].
$$

The struct `BL3FormFactors` in [lambda_c_semileptonic.mojo](../../../src/am3_mojo/physics/lambda_c_semileptonic.mojo) evaluates Meinel's nominal fits of the $z$ expansion ([2017](https://arxiv.org/abs/1611.09696) eq. 3, [2018](https://arxiv.org/abs/1712.05783) eq. 10):

$$
f(q^2) = \frac{a_0^f + a_1^fz + a_2^fz^2}{1 - q^2/(m_{\rm pole}^f)^2},\qquad z = \frac{\sqrt{t_+ - q^2} - \sqrt{t_+ - t_0}}{\sqrt{t_+ - q^2} + \sqrt{t_+ - t_0}},\qquad t_0 = (M - m_\mathrm B)^2.
$$

Here $f$ stands for each of the six form factors, and $t_0$ is the largest $q^2$ of the decay. The expansion variable $z$ is not the share $z$ of the $\mathrm{pp}$ section. The constant $t_+$ is $(m_\mathrm D + m_\mathrm K)^2$ for $\Lambda$ and $(m_\mathrm D + m_\pi)^2$ for $\mathrm n$, with the masses of the D meson, the kaon and the pion. The coefficients $a_n^f$ of the powers $z^n$ are those of Table IV ([2017](https://arxiv.org/abs/1611.09696)) and Table V ([2018](https://arxiv.org/abs/1712.05783)), with the digits of the supplemental files. The pole masses $m_{\rm pole}^f$ are those of the 2017 text and of Table IV (2018). The form factors $g_+$ and $g_\perp$ share $a_0$ and their pole, so that $g_+ = g_\perp$ at $q^2 = t_0$ (2018 eq. 7), where $s_{-}$ vanishes too. The function `bl3_density` in [lambda_c_semileptonic.mojo](../../../src/am3_mojo/physics/lambda_c_semileptonic.mojo) therefore writes $c$ without the 0/0 at $t_0$, with the shorthand $w$, a new use of the letter:

$$
c = \frac{(M - m_\mathrm B)\thinspace w\left[a_1^{g_+} - a_1^{g_\perp} + (a_2^{g_+} - a_2^{g_\perp})z\right]}{1 - q^2/(m_{\rm pole}^{g_+})^2},\qquad w = \frac{z}{s_-} = \frac{1}{\left(\sqrt{t_+ - q^2} + \sqrt{t_+ - t_0}\right)^2}.
$$

Summed over all spins, the squared matrix element is a bilinear form $W$ in the momenta of the leptons, a new use of the letter. Its coefficients $K$ are made of those of the current:

$$
\sum\vert\mathcal M\vert^2 = \tfrac12 G_F^2\vert V_{cq}\vert^2\thinspace W,\qquad W = K_g\thinspace p_\ell\cdot p_\nu + K_{PP}(P\cdot p_\ell)(P\cdot p_\nu) + K_{qq}(q\cdot p_\ell)(q\cdot p_\nu) + \tfrac12 K_S\left[(P\cdot p_\ell)(q\cdot p_\nu) + (q\cdot p_\ell)(P\cdot p_\nu)\right] + \tfrac12 K_A\left[(P\cdot p_\ell)(q\cdot p_\nu) - (q\cdot p_\ell)(P\cdot p_\nu)\right],
$$

$$
K_{PP} = 32\left[F^2 + G^2 + 2Fa(M + m_\mathrm B) + 2Gc(M - m_\mathrm B) + a^2s_+ + c^2s_-\right],\qquad K_{qq} = -32\left[F^2 + G^2 + 2Fb(M - m_\mathrm B) + 2Gd(M + m_\mathrm B) - b^2s_+ - d^2s_-\right],
$$

$$
K_S = -64\left[F\left(a(M - m_\mathrm B) - b(M + m_\mathrm B)\right) + G\left(c(M + m_\mathrm B) - d(M - m_\mathrm B)\right) - ab\thinspace s_+ - cd\thinspace s_-\right],\qquad K_A = -128\thinspace FG,
$$

$$
K_g = -16\left[4Mm_\mathrm B(F^2 - G^2) + s_+\left(2F\left(a(M + m_\mathrm B) + b(M - m_\mathrm B)\right) + (aP + bq)^2\right) + s_-\left(2G\left(c(M - m_\mathrm B) + d(M + m_\mathrm B)\right) + (cP + dq)^2\right)\right].
$$

In $K_g$, the square $(aP + bq)^2$ is $a^2P^2 + 2ab\thinspace P\cdot q + b^2q^2$, with $P^2 = 2(M^2 + m_\mathrm B^2) - q^2$ and $P\cdot q = M^2 - m_\mathrm B^2$. The square $(cP + dq)^2$ expands in the same way. No $\epsilon$ tensor survives. Setting all form factors to 1 gives the free-quark current, $W = 256\thinspace(p\cdot p_\ell)(p'\cdot p_\nu)$. The form $W$ is linear in $p_\ell$, so the spin rule gives the helicity difference by $p_\ell \to m_\ell s$. Integrated over the lepton energy at fixed $q^2$, $W$ gives [Meinel](https://arxiv.org/abs/1712.05783)'s differential rate $d\Gamma/dq^2$ (2018 eq. 17).

### Energy distributions at rest

Each mode gives its products an energy distribution in the parent frame, normalized to one product per decay. The branching ratio thus sets the rate, and the matrix elements only shape the distributions. In the code, `XMode`, `_nodes` and `XRules` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) give these distributions. A product of a two-body mode has the single energy $E^\ast = (M^2 + m^2 - m_b^2)/(2M)$. Here $M$, $m$ and $m_b$ are the masses of the parent, the product and the other product. In a three-body semileptonic mode, a product's distribution follows from the Dalitz-plot density $\vert\mathcal M\vert^2$. As for the kaons, it is the integral of this density along the line of fixed product energy. For the charged lepton, the same integral of the helicity difference gives its polarization weight. The function `_three_body_line` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluates both integrals.

The hadronic modes, the semileptonic complements and four of the $\tau$ modes are flat in phase space, with a constant matrix element. The decays of the short-lived products into three or more particles are flat too. In a mode of $n$ products, a product of mass $m$, energy $E$ and momentum $p$ recoils against the other products. The other products together have the invariant mass $m_{\rm rec}$. The function `_phase_space` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) builds its distribution $g(E)$, a new $g$, by the recursion over two-body phase spaces:

$$
g(E) \propto p\thinspace R_{n-1}(m_{\rm rec}),\qquad m_{\rm rec}^2 = M^2 + m^2 - 2ME,\qquad R_2(m) = \frac{\lambda^{1/2}(m^2, m_1^2, m_2^2)}{m^2},\qquad R_k(m) = \int dm'\thinspace 2m'\thinspace R_2(m; m_1, m')\thinspace R_{k-1}(m').
$$

Here $R_k(m)$ is the phase-space volume of $k$ products of invariant mass $m$, up to a constant, and $\lambda(a, b, c) = a^2 + b^2 + c^2 - 2ab - 2bc - 2ca$ is Källén's function. In $R_2$, $m_1$ and $m_2$ are the masses of the two products. In the recursion, $R_2(m; m_1, m')$ is $R_2$ for the first product, of mass $m_1$, and the other products, of invariant mass $m'$. For three products, $p\thinspace R_2(m_{\rm rec})$ is the length of the line of fixed $E$ across the flat Dalitz plot.

The charged lepton of a semileptonic complement keeps the helicity of the V−A current with a massless neutrino. The $\ell^+$ is right-handed in chirality, so its longitudinal polarization in the parent frame is its velocity $p/E$ (`_nodes` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo)). The $\mu^+$ of the complements have the mean polarization 0.93 in $\mathrm D^+$ and $\mathrm D^0$ decays. Those of the three-body modes have 0.955.

### The τ lepton

The $\tau^+$ of the decays $\mathrm D^+$ and $\mathrm D_s^+ \to \tau^+\nu_\tau$ is left-handed, with the longitudinal polarization $\mathcal P = -1$. The spin-0 parent and the left-handed neutrino fix its helicity. The function `tau_decay_modes` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) holds its eleven modes, the charge conjugates of [PDG](https://pdg.lbl.gov/2026/)'s $\tau^-$ basis modes (pp. 6–11). They are renormalized to their sum, 0.97045:

| Mode | Branching ratio | [PDG 2026](https://pdg.lbl.gov/2026/) | Angular distribution |
|---|---|---|---|
| $\mu^+\nu_\mu\bar\nu_\tau$ | 0.1737 | Γ3 | Michel spectra |
| $\mathrm e^+\nu_\mathrm e\bar\nu_\tau$ | 0.1785 | Γ5 | Michel spectra |
| $\pi^+\bar\nu_\tau$ | 0.1082 | Γ9 | $a = -1$ |
| $\mathrm K^+\bar\nu_\tau$ | 0.00697 | Γ10 | $a = -1$ |
| $\pi^+\pi^0\bar\nu_\tau$ | 0.2549 | Γ14 | as $\rho^+\bar\nu_\tau$, $a = a_\rho$ |
| $\pi^+\pi^0\pi^0\bar\nu_\tau$ | 0.0926 | Γ20 | as $a_1^+\bar\nu_\tau$, $a = a_{a_1}$ |
| $\pi^+\pi^+\pi^-\bar\nu_\tau$ | 0.0899 | Γ70 | as $a_1^+\bar\nu_\tau$, $a = a_{a_1}$ |
| $\pi^+\pi^0\pi^0\pi^0\bar\nu_\tau$ | 0.0104 | Γ27 | phase space |
| $\pi^+\mathrm K^0\bar\nu_\tau$ | 0.00838 | Γ36 | phase space |
| $\pi^+\pi^+\pi^-\pi^0\bar\nu_\tau$ | 0.0274 | Γ78 | phase space |
| $\pi^+\omega\bar\nu_\tau$ | 0.0195 | Γ178 | phase space |

In the $\tau$ frame, a product emitted at the angle $\theta$ to the $\tau^+$ flight direction has the distribution

$$
\frac{dN}{d\cos\theta} = \frac{1 + a\thinspace\mathcal P\cos\theta}{2},\qquad a_\pi = a_\mathrm K = -1,\qquad a_\rho = -\frac{m_\tau^2 - 2m_\rho^2}{m_\tau^2 + 2m_\rho^2}.
$$

Here $N$ counts the products, and $a$ is the product's asymmetry about the $\tau$ spin. The masses of the $\tau$ and the $\rho$ are $m_\tau$ and $m_\rho$. The V−A traces give $a = -1$ for a spin-0 hadron and $a_\rho$ for a spin-1 hadron summed over its spin. The asymmetry $a_{a_1}$ is $a_\rho$ with the mass of the $a_1$. The $\bar\nu_\tau$ of a two-body mode has the opposite asymmetry. [PDG](https://pdg.lbl.gov/2026/)'s fits of the chirality parameters $\xi(\pi)$, $\xi(\rho)$ and $\xi(a_1)$ agree with the V−A value 1 (pp. 88–89). In $\tau^+ \to \ell^+\nu_\ell\bar\nu_\tau$, the code treats the products as massless. A product of energy $E'$ in the $\tau$ frame has the energy fraction $x = 2E'/m_\tau$. The function `michel_nodes` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) uses

$$
\frac{dN_\ell}{dx} = 2x^2(3 - 2x),\qquad a_\ell = \frac{2x - 1}{3 - 2x},\qquad \frac{dN_\nu}{dx} = 12x^2(1 - x),\qquad a_\nu = -1,
$$

for the $\ell^+$ and the $\nu_\ell$. The $\bar\nu_\tau$ has the spectrum and asymmetry of the $\ell^+$. These are Michel's spectra with the V−A parameters $\rho = \delta = 3/4$, $\xi = 1$ and $\eta = 0$, which [PDG](https://pdg.lbl.gov/2026/)'s fits match (pp. 83–87). These four letters name Michel's parameters, not the mesons $\rho$ and $\eta$ or the fitted exponent $\delta$. The $\ell^+$ is right-handed.

The decay of the $\tau$ is not isotropic, so its products cannot go through the cascade of the other short-lived products (below). That cascade uses energy fractions alone. The $\tau^+$ decays in its parent's frame instead, with its polarization. The function `_emit_tau` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) places each product in the parent frame. A product of energy $E'$ and momentum $p'$ in the $\tau$ frame has there

$$
E = \gamma_\tau\left(E' + \beta_\tau p'\cos\theta\right),\qquad \gamma_\tau = \frac{M^2 + m_\tau^2}{2Mm_\tau},
$$

with $\gamma_\tau$ and $\beta_\tau$ the Lorentz factor and velocity of the $\tau$ in the parent frame, and $M$ the parent's mass. The parent decays isotropically, so the code then boosts the product to the lab with the energy $E$. A muon keeps its helicity, which is the same in every frame for a massless lepton.

### Tables on the grid

The function `charm_decay_tables` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) builds the tables. Two-body and phase-space modes are boosted to the lab as the kaon decays are, and so are the other modes. A product has the energy $E^\ast$ and momentum $p^\ast$ in the frame of a parent of mass $M$. The parent is ultra-relativistic and decays isotropically, so the product is spread uniformly over $x$ between $(E^\ast - p^\ast)/M$ and $(E^\ast + p^\ast)/M$ in the lab. A muon's helicity in the lab follows the kaons' rule. The products in the bin $x_{k+1} \lt x \lt x_k$ have the number $\mathcal N_k$ and the energy $\mathcal E_k$, in units of the parent's energy. The functions `_bin_node` and `_bins_to_table` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) compute these and share each bin between its two grid points, linearly in $x$:

$$
\Delta\thinspace T(x_k) = \frac{\mathcal E_k - x_{k+1}\mathcal N_k}{x_k - x_{k+1}} + \frac{x_{k-1}\mathcal N_{k-1} - \mathcal E_{k-1}}{x_{k-1} - x_k}.
$$

The first term is the upper point's share of bin $k$, and the second term is the lower point's share of bin $k-1$. The two shares of a bin add up to its number, and their moments in $x$ add up to its energy. Both shares are positive because the bin's mean $x$, $\mathcal E_k/\mathcal N_k$, lies in the bin.

### Short-lived products

The short-lived products are the $\tau^+$, $\eta$, $\eta'$, $\omega$, $\phi$, $\rho$, $\mathrm{K}^\ast$, $f_0(980)$, $a_1$ and the hyperons. The function `short_lived_modes` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) gives each its [PDG 2026](https://pdg.lbl.gov/2026/) modes of 0.5% and above, renormalized, and [pdg_charm.py](pdg_charm.py) names the pages. The $\rho$ decays into $\pi\pi$ only, which PDG gives as about 100%, and the "neutrals" of the $\omega$ are not a final state. PDG gives no charge split of the $\mathrm K^\ast$ and $f_0(980)$ decays into $\mathrm K\pi$ and $\pi\pi$. There, isospin gives 2/3 and 1/3, and the larger share goes to the final states with charged pions. The $a_1^+$ decays as in the two three-pion modes of the $\tau$. A short-lived product $Y$ decays at once, so the direct products add to those of each $Y$:

$$
F_{s\to X}(x) = F^{\rm direct}_ {s\to X}(x) + \sum_Y \int_x^1 \frac{dy}{y}\thinspace F_{s\to Y}(y)\thinspace F_{Y\to X}\negthinspace\left(\frac{x}{y}\right).
$$

Here $y$, a new use of the letter, is the energy fraction of $Y$. The light-cone fraction of a particle is its energy plus its momentum along the parent's flight, over the same sum for the parent. These fractions multiply exactly along a decay chain, and they are the energy fractions of ultra-relativistic particles. A product $X$ that takes the fraction $x/y$ of the energy of $Y$ thus takes the fraction $x$ of the parent's. With $T = xF$ the products per unit $\ln x$, the integrand times $x$ is $T_{s\to Y}(y)\thinspace T_{Y\to X}(x/y)$ per unit $\ln y$: a convolution in $\ln x$. On the grid $x_k = e^{-k\Delta}$, the function `_conv` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluates it as

$$
T_{s\to X}(x_k) = T^{\rm direct}_ {s\to X}(x_k) + \Delta\sum_Y\sum_{j=0}^k T_{s\to Y}(x_j)\thinspace T_{Y\to X}(x_{k-j}),\qquad x_jx_{k-j} = x_k,
$$

where $j$ numbers the grid points of the fraction $y$. The sum keeps the number and the energy of the products exactly. The functions `_short_lived_order` and `_resolve` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) apply this cascade to every table. The convolution needs an isotropic decay of $Y$. The vector mesons of the semileptonic modes are polarized, but their decays into spinless products are even in the polar angle. The number and energy of their products are thus exact, and only their spread is not. The code neglects the polarization of the hyperons.

### Products

The products are the pions, the muons of each helicity in the lab, the kaons and the nucleons. They also include $\mathrm e^\pm$, photons, and the neutrinos and antineutrinos of the three flavors. A $\mathrm K^0$ or $\bar{\mathrm K}^0$ counts half as a $\mathrm K^0_\mathrm S$ and half as a $\mathrm K^0_\mathrm L$. For an antiparticle, the products are the CP conjugates: a left-handed $\mu^+$ becomes a right-handed $\mu^-$, and antinucleons count as nucleons. The functions `charm_product_grid` and `charm_product_conjugate` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) place the products on the grids and conjugate them. The method `SimulationManager.Calc_C` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) computes the sources of these products.
