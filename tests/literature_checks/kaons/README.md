# Kaons

am3_mojo makes the kaons $\mathrm K^\pm$, $\mathrm K^0_\mathrm S$ and $\mathrm K^0_\mathrm L$ in photo-hadronic interactions (of nucleons with photons) and in proton–proton ($\mathrm{pp}$) collisions. It then evolves them until they [decay](#decays) or escape. This page starts from the arXiv versions of the papers it cites. [Hümmer et al. (2010)](https://arxiv.org/abs/1002.1310) give the photo-hadronic interactions. [Kelner et al. (2006)](https://arxiv.org/abs/astro-ph/0606058) and [Koers et al. (2006)](https://arxiv.org/abs/hep-ph/0611219) give the $\mathrm{pp}$ collisions, and [Lipari et al. (2007)](https://arxiv.org/abs/0704.0718) give the evolution and decays. The page also uses the listings of the Particle Data Group ([PDG 2026](https://pdg.lbl.gov/2026/)) and the measured data in [tests/support/measured/](../../support/measured/). From these sources, the page derives the equations and tables that the Mojo code uses, with the corresponding struct and function. The modules [kaon_production.py](kaon_production.py) and [kaon_decays.py](kaon_decays.py) check every step: `.venv/bin/python tests/literature_checks/common/run.py kaon_production kaon_decays`.

## Production in photo-hadronic interactions

### The interaction rate

[Hümmer et al. (2010)](https://arxiv.org/abs/1002.1310) split the photo-hadronic interactions into interaction types IT. AM³ makes its pions with their framework, and am3_mojo extends this framework to all four kaon species. In an isotropic photon field, their eq. 4 gives the rate $\Gamma^{\rm IT}$ at which a proton of energy $E_p$ interacts in type IT:

$$
\Gamma^{\rm IT}(E_p) = \frac{1}{2}\frac{m_p^2}{E_p^2}\int_{\epsilon_{\rm th}m_p/(2E_p)}^\infty d\varepsilon\thinspace \frac{n_\gamma(\varepsilon)}{\varepsilon^2}\int_{\epsilon_{\rm th}}^{2E_p\varepsilon/m_p} d\epsilon_r\thinspace \epsilon_r\thinspace \sigma^{\rm IT}(\epsilon_r)
$$

Here $n_\gamma(\varepsilon)$ is the density of photons of lab energy $\varepsilon$ per unit energy. [Hümmer et al.](https://arxiv.org/abs/1002.1310) write this energy $\varepsilon$, but the page's own equations write it $\epsilon$. The cross section $\sigma^{\rm IT}$ depends on the photon energy $\epsilon_r$ in the rest frame of the proton and is zero below the threshold $\epsilon_{\rm th}$. Hümmer et al. set $c = 1$, so $m_p$ is the proton's rest energy.

Eq. 3 of [Hümmer et al.](https://arxiv.org/abs/1002.1310) averages the rate over the angle between the proton and the photon. Eq. 4 is eq. 3 for an isotropic field. The inner integral runs over that angle. For an ultra-relativistic proton, $\epsilon_r$ goes from 0, when the photon and the proton are parallel, to its upper limit in a head-on collision. Their eqs. 8 and 27 define the response function $f^{\rm IT}$ and the variable $y$, which is half that limit:

$$
y = \frac{E_p\thinspace\varepsilon}{m_p},\qquad f^{\rm IT}(y) = \frac{1}{2y^2}\int_{\epsilon_{\rm th}}^{2y} d\epsilon_r\thinspace \epsilon_r\thinspace \sigma^{\rm IT}(\epsilon_r)
$$

The inner integral of eq. 4 is thus $2y^2 f^{\rm IT}(y)$. With $\varepsilon = m_p y/E_p$, the rate becomes one integral over $y$.

This page writes the same rate per interaction channel $c$ with the cross section $\sigma_c(\epsilon_r)$ at the photon energy $\epsilon_r$ in the nucleon rest frame. The channel's secondaries carry the fraction $\chi_c$ of the nucleon energy. Per interaction, the channel makes $M_{c,p}^K$ kaons of species $K$ from a proton and $M_{c,n}^K$ from a neutron. In kaon channels, the cross section $\sigma_{c,N}$ and the energy fraction $\chi_{c,N}^K$ can also depend on the nucleon $N$ and the kaon species $K$. A nucleon of energy $E_{N}$ and density $N_{N}$ per unit energy meets photons of lab energy $\epsilon$ and density $n_\gamma(\epsilon)$ per unit energy. AM³ uses the proton's rest energy $m_p c^2$ for both nucleons. With $y = E_N \epsilon/(m_p c^2)$, the interaction rate is

$$
t_{c,N}^{-1}(E_N) = c\int dy\thinspace \frac{m_p c^2}{E_N}\thinspace n_\gamma\negthinspace \left(\frac{m_p c^2\thinspace y}{E_N}\right) f_{c,N}(y),\qquad f_{c,N}(y) = \frac{1}{2y^2}\int_0^{2y} \epsilon_r\thinspace \sigma_{c,N}(\epsilon_r)\thinspace d\epsilon_r.
$$

Here $c$ in front of the integral is the speed of light, not a channel. The second relation, the channel's response function, integrates from $\epsilon_r = 0$, because the cross sections vanish below their thresholds.

On the code's grid: AM³'s energy grids are logarithmic. They step by $\Delta$ in $\ln E$. The step is the same for the nucleons and for the photons that they meet. The grids hold densities per unit $\ln E$, such as $\epsilon\thinspace n_\gamma(\epsilon)$. With $\epsilon = m_p c^2 y/E_N$, the measure $dy\thinspace m_p c^2/E_N$ is $d\epsilon$, which is $\epsilon\thinspace d\ln\epsilon$. The rectangle rule in $\ln\epsilon$ then gives the rate at the grid energy $E_i$ as a sum over the photon energies $\epsilon_j$ of the grid:

$$
t_{c,N}^{-1}(E_i) \simeq c\thinspace\Delta\sum_j f_{c,N}(y_{ij})\thinspace \epsilon_j n_\gamma(\epsilon_j),\qquad y_{ij} = E_i\epsilon_j/(m_p c^2)
$$

The product $E_i\epsilon_j$ grows by the factor $e^\Delta$ with each step of $i + j$, so $y_{ij}$ depends on $i + j$ alone. The methods `KChannel.f`, `PhotoHadron._fill_channel` and `PhotoHadron.Calc_PhotoHadron_PhotonConvolution` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) evaluate this sum.

### The kaon source

[Hümmer et al.](https://arxiv.org/abs/1002.1310)'s eq. 28 gives the source of the secondaries $b$ of type IT, per unit of their energy $E_b$. In their factorized response (eq. 27), a $\delta$ function puts every secondary at the fraction $\chi^{\rm IT}$ of the proton energy. One integral then remains:

$$
Q_b^{\rm IT} = N_p(E_b/\chi^{\rm IT})\thinspace \frac{m_p}{E_b}\int_{\epsilon_{\rm th}/2}^\infty dy\thinspace n_\gamma(m_p y\chi^{\rm IT}/E_b)\thinspace M_b^{\rm IT}\thinspace f^{\rm IT}(y)
$$

Here $N_p$ is the density of protons per unit energy, and $M_b^{\rm IT}$ is the number of secondaries per interaction. At the proton energy $E_p = E_b/\chi^{\rm IT}$, the argument of $n_\gamma$ is the photon energy $m_p y/E_p$, and the factor in front is $(m_p/E_p)/\chi^{\rm IT}$. The integral is then the rate of eq. 4, written as an integral over $y$, because $f^{\rm IT}$ vanishes below $y = \epsilon_{\rm th}/2$. The kaon source sums eq. 28 over the channels and the two nucleons and puts back the speed of light $c$. Per unit energy and time, the source of kaons of species $K$ and energy $E_{K}$ is

$$
Q_K(E_K) = \sum_c \sum_{N=p,n} \frac{M_{c,N}^K}{\chi_{c,N}^K}\thinspace \Big[N_N\thinspace t_{c,N}^{-1}\Big]_ {E_N = E_K/\chi_{c,N}^K}.
$$

On the code's grid: Per unit $\ln E_K$, the source has an extra factor $E_K = \chi_{c,N}^K E_N$, which cancels its factor $1/\chi$:

$$
E_K\thinspace Q_K(E_K) = \sum_{c,N} M_{c,N}^K\thinspace \Big[E_N N_N\thinspace t_{c,N}^{-1}\Big]_ {E_N = E_K/\chi_{c,N}^K}
$$

So the kaons of the nucleon bin $i$ land at $\ln E_i + \ln\chi$, which lies $q + r$ grid steps from $\ln E_i$, with $q$ an integer and $0 \le r \lt 1$. The method `PhotoHadron.Calc_PhotoHadron_Kaons` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) shares these kaons between the bins $i + q$ and $i + q + 1$. The method uses the weights of the pions (`number_split` in [specfun.mojo](../../../src/am3_mojo/utils/specfun.mojo)): the upper bin takes $r' = (e^{r\Delta} - 1)/(e^\Delta - 1)$, and the lower bin takes the rest. These weights keep the number of kaons and their energy, the energy at which they land. AM³ shares its pions with the weight $r + r(1 - r)\Delta/2$ instead, the weight of an interpolation. Up to terms of order $\Delta^3$, the particles it shares then have $1 + r(1 - r)\Delta^2$ times the energy at which they land. That is at most $1 + \Delta^2/4$ times that energy.

### Channels of constant cross section

Every channel has a constant cross section between the photon energies $\epsilon_\mathrm{min}$ and $\epsilon_\mathrm{max}$, and a zero cross section outside them. The response function of the channel then has a closed form:

$$
f_c(y) = \sigma_c [\min(4y^2, \epsilon_\mathrm{max}^2) - \epsilon_\mathrm{min}^2]/(4y^2)
$$

for $2y \ge \epsilon_\mathrm{min}$, and zero below. This response function is $1/(2y^2)$ times the integral of $\epsilon_r\sigma_c$ from $\epsilon_\mathrm{min}$ up to $2{y}$ or $\epsilon_\mathrm{max}$, whichever is lower. The method `KChannel.f` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) evaluates it.

### The energy fraction of a two-body kaon

[Hümmer et al.](https://arxiv.org/abs/1002.1310) take the energy fraction of a pion made with a nucleon in a two-body reaction from the kinematics. Their eq. 16, with the $s$ of their eq. 5, reads

$$
\chi(\epsilon_r) = \frac{s(\epsilon_r) - m_p^2 + m_\pi^2}{2s(\epsilon_r)}\thinspace (1 + \beta_\pi^{\rm cm}\cos\theta_\pi),\qquad s(\epsilon_r) = m_p^2 + 2m_p\epsilon_r
$$

Here $s$ is the squared energy in the center-of-momentum frame, and $m_\pi$ is the pion mass. In that frame, $\beta_\pi^{\rm cm}$ and $\theta_\pi$ are the pion's velocity and its angle to the nucleon, so $\cos\theta_\pi = -1$ means that the pion scatters backward. Eq. 16 boosts the pion's energy in that frame (their eq. 15) to the frame where the nucleon is fast (their eq. 14). It then divides the boosted energy by the proton energy $E_p$ in the new frame. For kaons, [Hümmer et al.](https://arxiv.org/abs/1002.1310) replace $m_p$ by the $\Lambda$ mass $m_\Lambda$ and $m_\pi$ by the kaon mass $m_K$. They also take the mean $\langle\cos\theta_\pi\rangle \simeq 0$ (their Sec. 3.1).

am3_mojo measures the angle from the photon instead, $\theta = \pi - \theta_\pi$, and uses the masses of each reaction. In the center-of-momentum frame, a kaon has the velocity $\beta^\ast$. A kaon emitted in that frame at the angle $\theta$ to the photon carries the fraction

$$
\chi = \frac{s - m_Y^2 + m_K^2}{2s}\thinspace \big(1 - \beta^\ast \cos\theta\big),\qquad s = m_N^2 + 2m_N\epsilon_r,
$$

of the nucleon energy, where $m_N, m_Y, m_K$ are the masses of the nucleon, the hyperon and the kaon. The fraction $\chi$ is linear in $\cos\theta$, so its mean over the angular distribution is $\chi(\langle\cos\theta\rangle)$. For every reaction and energy bin, the cross section $\sigma$ and $\langle\cos\theta\rangle$ come from the measured differential cross section $d\sigma/d\cos\theta$. The tool [ky_angular_extract.py](../../../tools/ky_angular_extract.py) computes these values from the HEPData records and the CLAS tables of [Compton et al.](https://arxiv.org/abs/1706.04748). The data module [ky_angular.py](../../support/measured/ky_angular.py) holds the results, reads this $\chi$ from this page and evaluates it for the measured angles. A step of the table below takes the mean $\chi$ over its photon energies and the channel's reactions, weighted by their cross sections.

The kaons of these reactions go mainly forward, along the photon, and more so at higher energies. This forward peaking lowers $\chi$ below the isotropic value that [Hümmer et al.](https://arxiv.org/abs/1002.1310) assume ($\langle\cos\theta\rangle = 0$). For $\mathrm K^+\Lambda$, $\langle\cos\theta\rangle$ rises from 0.1 near threshold to 0.7 at 3.5 GeV. So the $\chi$ of the channel $\mathrm p \to \mathrm K^+$, in which a proton makes a $\mathrm K^+$, lies below its isotropic value. The shortfall is 5% near threshold and more than half at 3–5 GeV ([two_body.py](two_body.py)). Near threshold, the reactions that make a $\mathrm K^0$ with a hyperon peak backward instead. The backward peaking is clear for $\gamma\mathrm n \to \mathrm K^0\Lambda$ and holds within the errors for $\mathrm K^0\Sigma^0$ and $\mathrm K^0\Sigma^+$, as for the lowest CLAS bin of $\mathrm K^+\Sigma^0$. This backward peaking raises the $\chi$ of $\mathrm n \to \mathrm K^0$ by 7–8% below 1.25 GeV.

### Two-body production near threshold

The code takes two-body production near threshold, $\gamma\mathrm N \to \mathrm K\Lambda, \mathrm K\Sigma$, in six steps of $\epsilon_r$. [Hümmer et al.](https://arxiv.org/abs/1002.1310) have one kaon channel, $\mathrm K^+$ production with 2.0 μb at 1.0–1.2 GeV and 3.7 μb at 1.2–1.65 GeV (their eqs. 41–42). The first three steps of the table below refine this channel with the measured energy dependence. Above 1.65 GeV, the measured fall-off replaces their constant 2.7 μb, which they keep up to the highest energies. In each step, each of the four channels has its own cross section $\sigma$ and energy fraction $\chi$. Six reactions make up the four channels, as the parentheses in the table header show. In a step, a channel's cross section is the mean over the step of the measured cross sections, summed over the channel's reactions. The paragraphs after the table describe the data and their extrapolation. The table is the list `ky` of `kaon_production_channels` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo).

| $\epsilon_r$ [GeV] | $\mathrm p \to \mathrm K^+$ ($\mathrm K^+\Lambda$, $\mathrm K^+\Sigma^0$) | $\mathrm p \to \mathrm K^0$ ($\mathrm K^0\Sigma^+$) | $\mathrm n \to \mathrm K^+$ ($\mathrm K^+\Sigma^-$) | $\mathrm n \to \mathrm K^0$ ($\mathrm K^0\Lambda$, $\mathrm K^0\Sigma^0$) |
|---|---|---|---|---|
| 0.91–1.05 | 1.31 μb, $\chi = 0.30$ | — | — | 0.81 μb, 0.34 |
| 1.05–1.25 | 3.00 μb, 0.29 | 0.31 μb, 0.32 | 0.49 μb, 0.29 | 1.78 μb, 0.35 |
| 1.25–1.65 | 4.09 μb, 0.29 | 0.55 μb, 0.30 | 1.28 μb, 0.30 | 1.89 μb, 0.33 |
| 1.65–2.2 | 2.44 μb, 0.26 | 0.26 μb, 0.29 | 1.03 μb, 0.29 | 0.80 μb, 0.29 |
| 2.2–3.0 | 1.25 μb, 0.23 | 0.083 μb, 0.30 | 0.90 μb, 0.29 | 0.28 μb, 0.27 |
| 3.0–5.0 | 0.44 μb, 0.19 | 0.030 μb, 0.23 | 0.39 μb, 0.22 | 0.10 μb, 0.19 |

The data: Three experiments measured the six reactions. Each experiment measured a different subset, up to a different photon energy:

| Reaction | SAPHIR | CLAS | A2 | Above its data, follows |
|---|---|---|---|---|
| $\gamma\mathrm p \to \mathrm K^+\Lambda$ | 0.91–2.58 ([Glander et al. 2004](https://arxiv.org/abs/nucl-ex/0308025)) | 1.25–3.82 ([McCracken et al. 2010](https://arxiv.org/abs/0912.4274)) | — | a power law |
| $\gamma\mathrm p \to \mathrm K^+\Sigma^0$ | 1.06–2.58 ([Glander et al. 2004](https://arxiv.org/abs/nucl-ex/0308025)) | 1.08–3.82 ([Dey et al. 2010](https://arxiv.org/abs/1006.0374)) | — | a power law |
| $\gamma\mathrm p \to \mathrm K^0\Sigma^+$ | 1.10–2.53 ([Lawall et al. 2005](https://arxiv.org/abs/nucl-ex/0504014)) | — | not used (protons bound in deuterium) | $\mathrm K^+\Sigma^0$ (0.15x) |
| $\gamma\mathrm n \to \mathrm K^+\Sigma^-$ | — | 1.15–3.55 ([Anefalos Pereira et al. 2010](https://arxiv.org/abs/0912.4833)) | — | $\mathrm K^+\Sigma^0$ (2.0x) |
| $\gamma\mathrm n \to \mathrm K^0\Lambda$ | — | 0.97–2.90 ([Compton et al. 2017](https://arxiv.org/abs/1706.04748)) | 0.94–1.36 ([Akondi et al. 2019](https://arxiv.org/abs/1811.05547)) | $\mathrm K^+\Lambda$ (0.21x) |
| $\gamma\mathrm n \to \mathrm K^0\Sigma^0$ | — | — | 1.05–1.36 ([Akondi et al. 2019](https://arxiv.org/abs/1811.05547)) | $\mathrm K^0\Sigma^+$ (1.7x) |

The ranges are $\epsilon_r$ in GeV, from the center of the first bin used to the center of the last ([two_body.py](two_body.py) lists them). If the measured angles of an energy bin reach from $\cos\theta \le -0.6$ to $\cos\theta \ge 0.6$, the bin is used. The A2 bins must reach $\pm 0.7$, but the CLAS bins of the $\mathrm K^0\Lambda$ tables need only reach backward to $-0.5$. For $\mathrm K^+\Sigma^-$, the CLAS bins need only reach backward to $-0.1$. This limit keeps the CLAS bins of $\mathrm K^+\Sigma^-$ above 2.85 GeV, whose data reach backward only to $\cos\theta = -0.4 \ldots {-0.1}$. The neutron data were taken on deuterium.

Between and beyond the data: For each reaction, the cross section and the kaon's mean angle, which gives $\chi$, follow the first rule that applies:

1. Within the data, each data set is interpolated linearly between its bins. If two experiments cover the same energy, their values are averaged.
2. Between the threshold and the first bin, the cross section rises linearly from zero, and the mean angle is that of the first bin. A2's first $\gamma\mathrm n \to \mathrm K^0\Sigma^0$ bin is at the photon–nucleon invariant mass $W$ = 1675–1705 MeV. In photon energy, the center of this bin lies just below the threshold. So the cross section of this reaction starts at 0.15 μb, the value of that bin.
3. Above its own data, a reaction follows its analogue, the $\gamma\mathrm p$ reaction of the last column, with the analogue's mean angle at the same energy. The cross section of the reaction is the analogue's cross section times a constant factor, which is not chosen by hand. The factor is the ratio of the two reactions' mean measured cross sections over the last 0.6 GeV of the reaction's data. For example, the CLAS data of $\mathrm K^0\Lambda$ end at 2.9 GeV, and its mean cross section is 0.21 times that of $\mathrm K^+\Lambda$ over 2.3–2.9 GeV. The data of $\mathrm K^0\Sigma^0$ span only 0.3 GeV, so its ratio is taken over all of them. An analogue can itself be extrapolated: $\mathrm K^0\Sigma^0$ follows $\mathrm K^0\Sigma^+$, which above 2.5 GeV follows $\mathrm K^+\Sigma^0$.
4. *Above 3.8 GeV*, where all data end, $\mathrm K^+\Lambda$ and $\mathrm K^+\Sigma^0$ fall as power laws fitted to their last GeV of data, $\propto E^{-2.7}$ and $E^{-2.9}$. Here $E$ is the photon energy. The mean angles of these two reactions stay at their values averaged over the last 0.4 GeV of data, weighted by the cross section. Through rule 3, the other four reactions follow them. This rule matters only from 3.8 to 5 GeV. The last step ends at 5 GeV, and the code has no two-body channel above this energy.

Why these analogues and factors: A constant factor carries the analogue's energy dependence over to the reaction. This transfer is reasonable only if the two reactions proceed in the same way. The analogue is therefore a $\gamma\mathrm p$ reaction, measured to higher energy, with a hyperon of the same isospin. A $\mathrm K\Lambda$ pair has isospin 1/2 only, but a $\mathrm K\Sigma$ pair can also have isospin 3/2. So the $\Delta$ resonances, baryons of isospin 3/2, make $\mathrm K\Sigma$ but not $\mathrm K\Lambda$. At higher energies, the exchange of a meson between the photon and the nucleon takes over from the resonances. This exchange is called the $t$ channel, after the squared four-momentum $t$ that the exchanged meson carries. The fitted factors reflect this:

- $\mathrm K^+\Sigma^-$ follows $\mathrm K^+\Sigma^0$ with 2.0. In $\mathrm K^+$ exchange, isospin symmetry makes a neutron couple $\sqrt{2}$ times as strongly to $\Sigma^-\mathrm K^+$ as a proton couples to $\Sigma^0\mathrm K^+$. The cross section goes with the square of the coupling, so it is twice as large. The fit gives the same factor, 2.0.
- $\mathrm K^0\Lambda$ follows $\mathrm K^+\Lambda$ with 0.21, and $\mathrm K^0\Sigma^+$ follows $\mathrm K^+\Sigma^0$ with 0.15. The data module [ky_angular.py](../../support/measured/ky_angular.py) computes both factors. The photon does not couple to a neutral $\mathrm K^0$, so these reactions have no kaon exchange in the $t$ channel. Their $\mathrm K^+$ analogues have this exchange, which dominates them at higher energies. So the $\mathrm K^0$ reactions are much smaller there.
- $\mathrm K^0\Sigma^0$ follows $\mathrm K^0\Sigma^+$ rather than $\mathrm K^+\Sigma^0$, because it lacks kaon exchange too. Its factor, 1.7, comes from A2's data near threshold, the only data on this reaction.

### Kaons in multi-pion production

[Hümmer et al.](https://arxiv.org/abs/1002.1310) (Table 6) describe multi-pion production by seven interaction types, M1–M7. Each type covers one range of the photon energy $\epsilon_r$ in the nucleon rest frame. In that range, the type has a constant cross section and its own pion multiplicities and energy fractions. The ranges are M1 0.5–0.9 GeV, M2 0.9–1.5 GeV, M3 1.5–5 GeV and M4 5–50 GeV. The ranges of the higher types are M5 50–500 GeV, M6 500–5000 GeV and M7 above, up to $5 \times 10^6$ GeV as in AM³.

In am3_mojo, M2–M7 also make the kaons that come with further particles, at the same cross sections. These kaons come from reactions such as $\gamma\mathrm N \to \mathrm K\mathrm Y\pi$, where Y stands for a hyperon. A kaon can be made with a third particle only above 1.15 GeV. The threshold of $\gamma\mathrm N \to \mathrm K\Lambda\pi$ is 1.152–1.168 GeV for the four charge states, and $\mathrm K\Sigma\pi$ starts at 1.30 GeV. So M1 makes no kaons, and M2 makes kaons only above 1.15 GeV. An antikaon $\bar{\mathrm K}$ comes only in a kaon pair, from 1.51 GeV on ($\gamma\mathrm N \to \mathrm N\mathrm K\bar{\mathrm K}$), so M2 makes no antikaons.

Like the pions, the kaons of a type have a low-energy part and a leading part. The leading part holds the share $h_+$ of the $\mathrm K^+$ and $\mathrm K^0$ by number, and the share $h_{-}$ of the antikaons, $\mathrm K^-$ and $\bar{\mathrm K}^0$. Each part puts its kaons at one energy fraction $\chi$, and the leading part has the larger fraction. A kaon's share of the nucleon energy is $\chi = (E^\ast - p^\ast_L)/\sqrt{s}$. Here $E^\ast$ is the kaon's energy and $p^\ast_L$ is its momentum along the photon, both in the center-of-momentum frame. In that frame, $\sqrt{s}$ is the total energy.

The table gives the kaons of each type per interaction of a proton. From a neutron, a type makes the isospin mirror of these kaons: $\mathrm K^+$ and $\mathrm K^0$ trade places, as do $\mathrm K^-$ and $\bar{\mathrm K}^0$. The yields rise most across M3 and M4. These two types have two steps of $\epsilon_r$ each:

| Type, photon energy $\epsilon_r$ | Cross section of the type [μb] | Kaons per interaction of a proton: $\mathrm K^+, \mathrm K^0, \mathrm K^-, \bar{\mathrm K}^0$ | $\mathrm K^+ + \mathrm K^0$ cross section [μb] | Leading shares $h_+$, $h_{-}$ | $\chi$ of the low part, of the leading part |
|---|---|---|---|---|---|
| M2, 1.15–1.5 GeV | 85 | 0.0051, 0.0060, 0, 0 | 0.94 | 0.25, 0.15 | 0.24, 0.35 |
| M3, 1.5–2.5 GeV | 120 | 0.0278, 0.0206, 0.0030, 0.0034 | 5.8 | 0.25, 0.15 | 0.24, 0.35 |
| M3, 2.5–5 GeV | 120 | 0.0585, 0.0449, 0.0214, 0.0201 | 12.4 | 0.25, 0.15 | 0.24, 0.35 |
| M4, 5–15 GeV | 120 | 0.0935, 0.0687, 0.0468, 0.0461 | 19.5 | 0.15, 0.09 | 0.126, 0.35 |
| M4, 15–50 GeV | 120 | 0.1316, 0.0988, 0.0794, 0.0796 | 27.6 | 0.15, 0.09 | 0.065, 0.35 |
| M5, 50–500 GeV | 120 | 0.185, 0.170, 0.140, 0.135 | 42.6 | 0.07, 0.04 | 0.032, 0.5 |
| M6, 500–5000 GeV | 120 | 0.350, 0.335, 0.300, 0.295 | 82.2 | 0.05, 0.03 | 0.011, 0.5 |
| M7, $5000\text{–}5 \times 10^6$ GeV | 120 | 0.575, 0.560, 0.520, 0.515 | 136 | 0.035, 0.02 | 0.0032, 0.6 |

The $\mathrm K^+ + \mathrm K^0$ cross section of a row is the type's cross section times its $\mathrm K^+$ and $\mathrm K^0$ per interaction, without the two-body reactions. This inclusive cross section counts every kaon. So it exceeds the type's cross section where an interaction makes more than one such kaon. For example, M7 makes 1.135 per interaction. The table is the list `mpi` of `kaon_production_channels` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo).

From [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s Table 6: The types' photon ranges and cross sections are those of their Table 6. The one exception is M7, which ends where AM³ ends it. In Table 6, each type makes a low-energy sample of pions (L) and a leading one (H), each with its own $\chi$. The leading part of the kaons takes the $\chi$ of the leading pions. The low part takes 1.6 times the $\chi$ of the low-energy pions, and 1.8 and 0.93 times in the two steps of M4. The factor 1.6 follows from kinematics. At the same rapidity, that is, the same velocity along the collision axis, a particle's energy is proportional to its transverse mass $m_T = (m^2 + p_T^2)^{1/2}$. Here $m$ is the particle's mass, and $p_T$ is its momentum across the collision axis. A kaon and a pion of the same rapidity and $p_T$ thus have the energy ratio $m_{T,K}/m_{T,\pi} = 1.6$ at $p_T$ = 0.35 GeV. The factors of M4 instead come from the $\mathrm K^0_\mathrm S$ measured at 20 GeV ([m4_steps.py](m4_steps.py)). Their distribution was measured in Feynman's $x = 2p^\ast_L/\sqrt{s}$. This variable is the momentum along the photon divided by about its largest value. This distribution, kept the same at other energies, gives the mean $\chi$ in each step.

From measured data: Strangeness balance and measured strange-particle photoproduction give the multiplicities of M2–M4. The kaons per pion of $\mathrm{pp}$ and proton–antiproton ($\bar{\mathrm p}\mathrm p$) collisions give those of M5–M7. The docstrings of [multibody_strangeness.py](multibody_strangeness.py) (M2 and M3), [m4_steps.py](m4_steps.py) (M4) and [higher_types.py](higher_types.py) (M5–M7 and the leading shares) derive these multiplicities, and the scripts compute them. Thresholds and kinematics enter in three places. The cross section of $\gamma\mathrm p \to \mathrm K^0\Lambda\pi^+$ behind M2 was measured as one value for 1.2–1.7 GeV. This cross section is counted either evenly over that range or only above 1.42 GeV, the threshold of $\Sigma(1385)\mathrm K$. M3's lower step counts only the kaon pairs made without a pion, as those with one start at 1.8 GeV. The kaons of M4 rise in the same way as $\mathrm K^0_\mathrm S$ production in $\pi^-\mathrm p$ collisions. Both rise with the available energy $E_A = \sqrt{s} - m_a - m_b$, where $m_a$ and $m_b$ are the masses of the colliding particles. So $E_{A}$ is their kinetic energy in the center-of-momentum frame.

The leading shares: The share $h_+$ is close to the leading share of the type's pions. This share is 0.25, 0.15, 0.07, 0.05 and 0.035 in M3–M7, and M2 takes the value of M3. The antikaons share no quark with the nucleon, so fewer of them are in the leading part. The leading part takes $h_- = 0.6\thinspace h_+$ of them, rounded as the table prints it.

### Energy bookkeeping

[Hümmer et al.](https://arxiv.org/abs/1002.1310)'s multi-pion types put the whole inelastic energy into pions, with pion energy fractions that reproduce those of the event generator SOPHIA. The code therefore rescales the pion multiplicities of each type $t$ for the kaons. Here $t$ labels a type, not a time. Each row $r$ of the table above is a range of photon energies in which the type makes kaons. In that range, the factor is

$$
s_{t,r} = 1 - \frac{\sum_{c \in t,\thinspace K} \chi_c\thinspace M_{c,r}^K}{\sum_{c \in t,\thinspace \pi} \chi_c\thinspace M_c^\pi},
$$

The numerator sums over the kaon channels $c$ of row $r$ and the kaon species $K$. The denominator sums over the pion channels of type $t$ and the pion species $\pi$. The factor thus keeps the energy of the interaction at every photon energy and leaves the nucleons unchanged. The two-body channels $\gamma\mathrm N \to \mathrm K\Lambda, \mathrm K\Sigma$ are not in AM³'s kernels for the nucleons' energy loss and re-injection. So the energy of their kaons is not taken from the nucleons. In Sec. 4.5, [Hümmer et al.](https://arxiv.org/abs/1002.1310) also neglect the effect of their $\mathrm K^+$ channel on the cooling and re-injection of the nucleons.

The methods `PhotoHadron._set_pion_cuts` and `PhotoHadron.Calc_PhotoHadron_Rad` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) apply the factor to the responses of the type's pions. The response is linear in the cross section. So these pions have the cross section $s_{t,r}\sigma_t$ in the photon range of row $r$, where $\sigma_t$ is the type's cross section.

### Neutral kaons

A $\mathrm K^0$ or $\bar{\mathrm K}^0$ is half $\mathrm K^0_\mathrm S$ and half $\mathrm K^0_\mathrm L$. CP violation and $\mathrm K^0\text{–}\bar{\mathrm K}^0$ interference ($\lesssim$ 1%) are neglected. Here CP is the combined symmetry of charge conjugation and parity. The function `_species` in [photohadron.mojo](../../../src/am3_mojo/physics/photohadron.mojo) applies this split to the multiplicities of the channels.

## Production in $\mathrm{pp}$ collisions

Above the transition energy $E_t$ = 100 GeV, AM³ makes pions in $\mathrm{pp}$ collisions with the distribution $F_\pi(x, E_p)$ of [Kelner, Aharonian & Bugayov (2006)](https://arxiv.org/abs/astro-ph/0606058). This pion energy distribution is the number of pions per inelastic collision and per unit $x$. Here $x = E_\pi/E_p$ is the share of the proton energy $E_p$ that a pion of energy $E_\pi$ takes. By default, AM³ uses Kelner et al.'s eq. 12, from the event generator SIBYLL, and otherwise their eq. 6, from QGSJET. Below $E_t$, AM³ uses a $\delta$-functional approximation for the pions. It takes the inelastic cross section from eq. 1 of [Kafexhiu et al. (2014)](https://arxiv.org/abs/1406.7369). Kelner et al. give no kaon distributions.

am3_mojo uses the kaon spectra of [Koers, Pe'er & Wijers (2006, arXiv:hep-ph/0611219)](https://arxiv.org/abs/hep-ph/0611219). These spectra are fits to the event generator PYTHIA 6.324 for $\mathrm K^\pm$, $\mathrm K^0$ and $\bar{\mathrm K}^0$ separately, per inelastic collision including diffractive ones. Koers et al. switched off the pion and kaon decays in their simulations, so the fits count the kaons before they decay.

AM³'s pions come from [Kelner et al.](https://arxiv.org/abs/astro-ph/0606058), who do not state whether their $F_\pi$ includes pions from kaon decays. They note that their leptons from all channels exceed those from their pion distribution (by about 10% at $x \sim 0.1$), mainly because of kaon decays. So at least the kaons' leptonic decays are not in $F_\pi$. To count nothing twice, am3_mojo assumes that no kaon decay is in $F_\pi$.

### The source

Eq. 71 of [Kelner et al.](https://arxiv.org/abs/astro-ph/0606058) gives the rate at which protons make $\gamma$-rays in a gas of hydrogen of density $n_H$. With the density $J_p(E_p)$ of the protons per unit energy (their eq. 70), the rate reads

$$
\Phi_\gamma(E_\gamma) = c\thinspace n_H\int_{E_\gamma}^\infty \sigma_{\rm inel}(E_p)\thinspace J_p(E_p)\thinspace F_\gamma\negthinspace\left(\frac{E_\gamma}{E_p}, E_p\right)\frac{dE_p}{E_p}
$$

Here $\sigma_\mathrm{inel}$ is the inelastic $\mathrm{pp}$ cross section, and $F_\gamma(x, E_p)$ is the number of photons per collision and per unit $x = E_\gamma/E_p$. A collision makes $F_\gamma\thinspace dx$ photons, which is $F_\gamma/E_p$ per unit $E_\gamma$. AM³ builds its pion source in the same way. With $F_{K}$ for $F_\gamma$ and $N_p$ for $J_p$, the kaon source is

$$
Q_K(E_K) = c\thinspace n_H\int \frac{dE_p}{E_p}\thinspace \sigma_{\rm inel}(E_p)\thinspace N_p(E_p)\thinspace F_K\negthinspace \left(\frac{E_K}{E_p},E_p\right),
$$

where $n_H$ is the density of the target protons, $N_p$ that of the protons per unit energy and $Q_{K}$ the kaon source per unit energy. AM³ takes $\sigma_\mathrm{inel}$ from [Kafexhiu et al.](https://arxiv.org/abs/1406.7369) (their eq. 1), in its version for the charged pions. The threshold of that version uses the $\pi^\pm$ mass (`ProtonProton.sigma_inel_Kafexhiu` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo)). The integral can run over all $E_p$, because $F_{K}$ vanishes for $x \gt 1$.

On the code's grid: With densities per unit $\ln E$, the rectangle rule in $\ln E_p$ gives the source at the grid energy $E_i$. The source is a sum over the proton energies $E_j$ of the grid:

$$
E_i\thinspace Q_K(E_i) \simeq c\thinspace n_H\Delta\sum_j \sigma_{\rm inel}(E_j)\thinspace E_j N_p(E_j)\thinspace x_{ij}F_K(x_{ij}, E_j),\qquad x_{ij} = E_i/E_j
$$

The methods `ProtonProton._init_kaon_kernels` and `ProtonProton.Calc_PP_KaonSrc` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) evaluate this sum. The next section gives $F_{K}$ in three ranges of the proton energy.

### The kaon spectra

**100 GeV $\le E_p \le 10^6$ GeV: [Koers et al. (2006)](https://arxiv.org/abs/hep-ph/0611219), eqs. 16–18 and their Table III.** With energies in GeV and $\log = \log_{10}$, the kaon distribution is $F_K(x, E_p) = E_p n(xE_p)$. Here $n(\epsilon)$ is the number of kaons per collision and per unit kaon energy $\epsilon$. As in Koers et al., this $\epsilon$ is the energy of the kaon, not of a photon. In their notation, and with $m_K$ for their $mc^2$, their eqs. 16–17 read:

$$
n(\epsilon) = n_0\thinspace \epsilon^{-1}\thinspace \alpha(\epsilon)\thinspace \beta(\epsilon)\thinspace \gamma_1(\epsilon)\thinspace \gamma_2(\epsilon),\qquad n_0 = 1.21\times 10^{\thinspace p_0 + p_1 p_2^2},\qquad \alpha = \epsilon^{\thinspace p_1(\log\epsilon - 2p_2)},
$$

$$
\beta = 10^{-(\epsilon/\epsilon_3)^{p_{30}}}\thinspace 10^{-(\epsilon/\epsilon_5)^{p_{50}}},\qquad \gamma_1 = \tanh\negthinspace \big(p_{70}\log(\epsilon/m_K)\big),\qquad \gamma_2 = \tanh\negthinspace \big(p_{80}\log(E_p/\epsilon)\big),
$$

with $\epsilon_3 = 10^{p_2+p_4}$, $\epsilon_5 = 10^{p_2+p_6}$, $p_0 = p_{00} + p_{01} \log E_p$, $p_1 = p_{10} + p_{11} \log E_p$, $p_2 = p_{20} + p_{21} \log E_p$, $p_4 = p_{40} - p_2$, $p_6 = p_{60} + p_2$, and $n = 0$ outside $m_K \lt \epsilon \lt E_p$. Here $m_K$ is the mass of the meson: $m_{\mathrm K^0}$ for $\mathrm K^0$ and $\bar{\mathrm K}^0$, and $m_{\pi^0}$ for the $\pi^0$ row. The factors $\alpha, \beta, \gamma_1, \gamma_2$ keep the names that [Koers et al.](https://arxiv.org/abs/hep-ph/0611219) give them. They do not stand for a velocity or a Lorentz factor here. The product $n_0\alpha$ completes the square only with the base-10 logarithm: it is $1.21 \times 10^{p_0 + p_1(\log\epsilon - p_2)^2}$, so its logarithm is a parabola in $\log\epsilon$. The parameters are:

| | $p_{00}$ | $p_{01}$ | $p_{10}$ | $p_{11}$ | $p_{20}$ | $p_{21}$ | $p_{30}$ | $p_{40}$ | $p_{50}$ | $p_{60}$ | $p_{70}$ | $p_{80}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $\mathrm K^+$ | −1.23 | 0.0657 | −0.147 | 0.0161 | −0.00411 | 0.493 | −0.989 | −0.345 | 0.777 | −0.235 | 42.7 | 12.0 |
| $\mathrm K^-$ | −1.46 | 0.101 | −0.109 | 0.00865 | −0.0577 | 0.491 | −1.22 | −0.164 | 1.04 | −0.279 | 18.6 | 4.23 |
| $\mathrm K^0$ | −1.29 | 0.0690 | −0.142 | 0.0154 | −0.00717 | 0.493 | −1.03 | −0.294 | 0.839 | −0.272 | 33.8 | 10.1 |
| $\bar{\mathrm K}^0$ | −1.50 | 0.101 | −0.118 | 0.0101 | −0.0567 | 0.491 | −1.25 | −0.169 | 1.05 | −0.272 | 21.2 | 4.07 |
| $\pi^0$ | −0.456 | 0.0950 | −0.142 | 0.0135 | −0.457 | 0.494 | −1.49 | −0.981 | 1.01 | 0.723 | 22.1 | 8.53 |

The $\pi^0$ row is used only by the extension above $10^6$ GeV. As in photo-hadronic interactions, a $\mathrm K^0$ or $\bar{\mathrm K}^0$ is half $\mathrm K^0_\mathrm S$ and half $\mathrm K^0_\mathrm L$: $F_{\mathrm K^0_\mathrm S}$ $= F_{\mathrm K^0_\mathrm L}$ $= \tfrac12(F_{\mathrm K^0} + F_{\bar{\mathrm K}^0})$. The functions `kpw_n` and `pp_kaon_F` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) evaluate $n(\epsilon)$ and give the $F_{K}$ of the four species. Below 1 TeV, a factor $c_K$ joins these spectra to the multiplicities measured below 100 GeV (see below).

$\boldsymbol{E_p \gt 10^6}$ **GeV (am3_mojo extension, not from a publication).** [Koers et al.](https://arxiv.org/abs/hep-ph/0611219)'s fit cannot be taken far beyond its range, because its parameters drift and the energy fraction of its pions grows toward 1. But within its range, the $\mathrm K^+/\pi^0$ and $\mathrm K^0/\pi^0$ ratios at fixed $x \gtrsim 0.03$ are nearly independent of $E_p$ (Feynman scaling of the ratio). For example, $\mathrm K^+/\pi^0 = 0.123\text{–}0.126$ at $x = 0.1$ and $0.26\text{–}0.28$ at $x = 0.9$ from $10^3$ to $10^6$ GeV. The $\mathrm K^-/\pi^0$ and $\bar{\mathrm K}^0/\pi^0$ ratios still rise by 13–14% at $x = 0.1$ and fall by 36–37% at $x = 0.9$ over that range. The module [kaon_production.py](kaon_production.py) computes these figures from the equations and Table III above. Above $E^\ast = 10^6$ GeV, am3_mojo keeps each kaon species' ratio to the $\pi^0$ at $E^\ast$ and multiplies it by the pion distribution that AM³ uses:

$$
F_K(x,E_p) = R(x)\thinspace \hat F_\pi(x,E_p),\qquad R(x) = \frac{E^\ast \thinspace n_K(x_s E^\ast;E^\ast)}{\hat F_\pi(x_s,E^\ast)},\qquad x_s = \max(x, 10^{-4}),
$$

where $n_K(\epsilon; E_p)$ is [Koers et al.](https://arxiv.org/abs/hep-ph/0611219)'s $n(\epsilon)$ of the kaon species at the proton energy $E_p$. The pion distribution $\hat F_\pi$ is [Kelner et al.](https://arxiv.org/abs/astro-ph/0606058)'s $F_\pi$ without its pion threshold factor $(1 - m_\pi/(xE_p))^{1/2}$, where $m_\pi$ is the pion mass. For $x \lt 10^{-4}$ (kaons below 100 GeV at $E^\ast$), the kaon mass shapes the ratio. In that range, $R$ follows Koers et al.'s $\mathrm K/\pi^0$ ratio $\rho(\epsilon) = n_K(\epsilon)/n_{\pi^0}(\epsilon)$ at $E^\ast$ as a function of the kaon energy. Here $n_{\pi^0}$ is their $n(\epsilon)$ of the $\pi^0$, from eqs. 16–18 and its row of Table III. So $R(x)$ $\to R(10^{-4})$ $\rho(\min(xE_p, 10^{-4}E^\ast))/\rho(10^{-4}E^\ast)$. The functions `kab_pion_F` and `_kpw_F` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) give $\hat F_\pi$ and the extension.

$\boldsymbol{E_p \lt E_t}$ **= 100 GeV (am3_mojo extension, not from a publication).** Below the transition energy $E_t$, the kaons use a $\delta$-functional approximation, like AM³'s pions. In this approximation of [Kelner et al. 2006](https://arxiv.org/abs/astro-ph/0606058) (eqs. 75–77), a collision makes $\tilde n$ pions. Together, these pions carry the fraction $\kappa$ of the proton's kinetic energy $E_{\rm kin}$. Each pion has the same energy, $(\kappa/\tilde n)E_{\rm kin}$:

$$
\tilde F_\pi = \tilde n\thinspace\delta\negthinspace\left(E_\pi - \frac{\kappa}{\tilde n}E_{\rm kin}\right),\qquad q_\pi(E_\pi) = \tilde n\thinspace\frac{c\thinspace n_H}{K_\pi}\thinspace\sigma_{\rm inel}\negthinspace\left(m_p + \frac{E_\pi}{K_\pi}\right)J_p\negthinspace\left(m_p + \frac{E_\pi}{K_\pi}\right),\qquad K_\pi = \kappa/\tilde n
$$

Here the pion distribution $\tilde F_\pi$ of this approximation is a Dirac $\delta$ function in the pion energy. In [Kelner et al.](https://arxiv.org/abs/astro-ph/0606058), $q_\pi$ is the pion source per unit energy, and $J_p$ is the density of the protons per unit energy. Their eq. 76 states that there are $\tilde n$ such pions and that they carry $\kappa E_{\rm kin}$, but it prints $k$ for $\kappa$. The kaons follow the same scheme. Each collision makes $\tilde n_K$ kaons of energy $E_K = \max(K_K T_p, \gamma_\mathrm{cm} m_K)$, where $m_K$ is the kaon mass. Here $T_p = E_p - m_p$ is the proton's kinetic energy, the $E_{\rm kin}$ of Kelner et al. The Lorentz factor of the center-of-momentum frame is $\gamma_\mathrm{cm} = \sqrt{(E_p + m_p)/(2m_p)}$ $= (E_p + m_p)/\sqrt{s}$, where $s$ is the squared energy in that frame. So $\gamma_\mathrm{cm} m_K$ is the lab energy of a kaon at rest in that frame, the lowest mean kaon energy. With $\tilde n_K$ for $\tilde n$ and $K_{K}$ for $K_\pi$, eq. 77 gives the kaon source of these collisions. The factor $K_{K}$ is the mean fraction of $T_p$ per kaon of the [Koers et al.](https://arxiv.org/abs/hep-ph/0611219) spectra at $E_t$: $K_K = E_t\langle x\rangle/(E_t - m_p)$, with $\langle x\rangle$ the mean $x$ of $F_K(x, E_t)$. The number of kaons per collision, $\tilde n_K$, is a fit to the kaon multiplicities of the NA49 compilation ([Anticic et al. 2010](https://arxiv.org/abs/1004.1889), Table 11):

$$
\tilde n_K(E_p) = f(s),\qquad f(s) = A\left(1-\frac{s_{\rm th}}{s}\right)^{a}\left(\frac{s}{s_{\rm th}}\right)^{b} + C\left(1-\frac{s_{\rm th}}{s}\right)^{2}\left(\frac{s_{\rm th}}{s}\right)^{d},\qquad s = 2m_p^2 + 2m_pE_p,
$$

where $s$ is again the squared center-of-momentum energy, and $s_{\rm th}$ is its value at the threshold in the table below. The parameters $A, a, b, C, d$ come from a least-squares fit in $\ln n$, the logarithm of the multiplicity. The fit takes the points of [Anticic et al.](https://arxiv.org/abs/1004.1889) with $\sqrt{s} \le$ 63 GeV, but it leaves out the charged-kaon points of [Fesefeldt et al. (1979)](https://doi.org/10.1016/0550-3213%2879%2990318-3). Fesefeldt et al. derived these points from their cross sections for channels with two strange particles. In this derivation, they set the unmeasured cross section of $\mathrm{pp} \to \mathrm K^+\mathrm K^-\mathrm X$ equal to that of $\mathrm{pp} \to \mathrm K^0\bar{\mathrm K}^0\mathrm X$, where $\mathrm X$ is anything else. They obtained the latter from their $\mathrm K^0_\mathrm S\thinspace\mathrm K^0_\mathrm S$ events, assuming equal cross sections for the two charge-conjugation states of $\mathrm K^0\bar{\mathrm K}^0$ (their p. 324). The threshold is that of the channel with the lowest threshold that makes the kaon:

| Kaon | $A$ | $a$ | $b$ | $C$ | $d$ | $\sqrt{s_\mathrm{th}}$ [GeV] | Threshold channel |
|---|---|---|---|---|---|---|---|
| $\mathrm K^+$ | 0.1132 | 4.099 | 0.2244 | 0.3354 | 5.29 | 2.548 | $\mathrm{pp} \to \mathrm p\Lambda\mathrm K^+$ |
| $\mathrm K^-$ | 0.0476 | 6.175 | 0.3324 | 0 | — | 2.864 | $\mathrm{pp} \to \mathrm{ppK}^+\mathrm K^-$ |
| $\mathrm K^0_\mathrm S$ and $\mathrm K^0_\mathrm L$ (fitted to $\langle \mathrm K^0_\mathrm S\rangle$) | 0.02128 | 2.115 | 0.5398 | 0 | — | 2.625 | $\mathrm{pp} \to \mathrm p\Sigma^+\mathrm K^0$ |

Only the $\mathrm K^+$ has the second term. It follows the two $\mathrm K^+$ points at $\sqrt{s}$ = 2.9 GeV, 0.35 GeV above the threshold. Without it, the first term, fitted to the other nine points, lies 7.3 and 7.6 times below them. Near threshold, $1 - s_{\rm th}/s$ is proportional to the energy $Q = \sqrt{s} - \sqrt{s_{\rm th}}$ above it, so the second term grows as $Q^2$. The phase space of the three particles of $\mathrm{pp} \to \mathrm p\Lambda\mathrm K^+$ grows in the same way. The factor $(s_{\rm th}/s)^d$ makes the term fade: from $\sqrt{s}$ = 3.8 GeV on, it is less than 10% of the fit. The eleven $\mathrm K^+$ points lie at 0.93 to 1.12 times the fit.

At $E_t$, the fit and the spectra of [Koers, Pe'er & Wijers](https://arxiv.org/abs/hep-ph/0611219) (KPW) differ. The fit's $\mathrm K^+$ are 1.30 times theirs, its $\mathrm K^-$ 0.97 times and its neutral kaons 0.985 times. Their $\mathrm K^+$ also fall short of the measured ones above $E_t$: 0.87 times at $\sqrt{s}$ = 17.2 GeV and 0.91 times at 23 GeV. Between $E_t$ and $E_1$ = 1 TeV, the lowest proton energy of their simulations (their abstract), the code therefore multiplies their $F_{K}$ by

$$
c_K(E_p) = c_t + (1 - c_t)\thinspace\frac{\ln(E_p/E_t)}{\ln(E_1/E_t)},\qquad c_t = \frac{f(s_t)}{\langle n_K\rangle_{\rm KPW}(E_t)}.
$$

Here $\langle n_K\rangle_{\rm KPW}(E_t)$ is the number of kaons per collision in their spectra at $E_t$, the integral of their $n(\epsilon)$. Also, $s_t$ is the value of $s$ at $E_t$. So the number of kaons per collision is continuous at $E_t$. From $E_1$ on, the number of kaons per collision is that of [KPW](https://arxiv.org/abs/hep-ph/0611219). There their $\mathrm K^+$ are 0.94 to 0.95 times the measured ones, at $\sqrt{s}$ = 45 to 200 GeV. The function `pp_kaon_fade` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) gives $c_K$.

The functions `pp_kaon_moments`, `pp_kaon_mult_fit` and `pp_kaon_sqrt_s_th` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) give the number and the energy of the kaons at $E_t$, the fit $f(s)$ and the thresholds. On the grid, the method `ProtonProton._init_kaon_kernels` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) shares the kaons of a proton bin between the two grid energies around $E_{K}$, $E_l \le E_K \lt E_{l+1}$. The upper one takes the weight $w = (E_K - E_l)/(E_{l+1} - E_l)$, which keeps the number and the energy of the kaons.

### Energy bookkeeping of $\mathrm{pp}$ collisions

am3_mojo makes the kaons in addition to AM³'s pions (see above), so the protons also lose the energy of the kaons. AM³'s $\mathrm{pp}$ energy-loss rate $\dot E/E = -n_H c \sigma_\mathrm{inel} \sum_\pi \kappa_\pi$ gains the term $-n_H c \sigma_\mathrm{inel} \sum_K \kappa_K$, with $\kappa_K$ the fraction of $E_p$ the kaons carry. In the same way, $\kappa_\pi$ is the fraction that the pions carry, and the sums run over the pion and kaon species. On the grid, $\kappa_K = \Delta\sum_i x_{ij}^2 F_K(x_{ij}, E_j)$, the rectangle rule of $\int xF_K\thinspace dx$ in $\ln x$, so that the protons lose exactly the energy of the kaon source. Below $E_t$, $\kappa_K = \tilde n_K E_K/E_p$. The function `ProtonProton.Calc_PP_ProtonLoss` in [proton_proton.mojo](../../../src/am3_mojo/physics/proton_proton.mojo) adds the term.

## Evolution

am3_mojo evolves each kaon species with AM³'s solver. The evolution uses the PDG 2026 masses and mean lives ([Particle Data Group, Takahashi et al. 2026, Int. J. Mod. Phys. A 41, 2630011](https://pdg.lbl.gov/2026/)). The charged kaons have $m_{\mathrm K^\pm}$ = 493.677 MeV and $\tau = 1.2380 \times 10^{-8}$ s. The neutral kaons have $m_{\mathrm K^0}$ = 497.611 MeV, $\tau_\mathrm S = 8.954 \times 10^{-11}$ s and $\tau_\mathrm L = 5.116 \times 10^{-8}$ s. These are the PDG fit values. For $\tau_\mathrm S$, the value comes from the fit that assumes CPT invariance.

The terms follow AM³'s treatment of the other hadrons:

| | $\mathrm K^\pm$ (as AM³'s charged pions) | $\mathrm K^0_\mathrm S$ and $\mathrm K^0_\mathrm L$ (as AM³'s neutrons) |
|---|---|---|
| synchrotron cooling and emission | ✓ | — (neutral) |
| inverse-Compton cooling and emission | ✓ | — (neutral) |
| quantum-synchrotron cooling | ✓ (when on, as for the pions) | — |
| adiabatic cooling | ✓ | — (not tied to the magnetized plasma) |
| escape | the pions' escape rate | the neutrons' escape rate |
| expansion | ✓ | ✓ |
| extra cooling | ✓ | ✓ |
| decay | ✓ | ✓ |

The methods `SimulationManager._mk_kaon`, `SimulationManager._kaon_charm_escape_cooling` and `SimulationManager.Calc_K` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) set up and evolve the kaons as the table says. These methods take the masses and mean lives from `kaon_mass` and `kaon_lifetime` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo). A kaon has the decay rate $1/{(\gamma\tau)}$, where $\gamma$ is its Lorentz factor.

The charged kaons lose energy to synchrotron and inverse-Compton radiation like AM³'s charged pions. The Thomson cross section of a charged particle scales as the inverse square of its mass. AM³'s solver holds the synchrotron loss as a rate (`SolverAlgorithm.compute_syn_ic` in [solver.mojo](../../../src/am3_mojo/solver/solver.mojo)). Above the Thomson regime, AM³'s Klein–Nishina treatment of inverse Compton applies. The emission of $\mathrm K^+$ and $\mathrm K^-$ uses AM³'s hadron synchrotron kernel (`SynP.Calc_SY_K` in [synchrotron.mojo](../../../src/am3_mojo/physics/synchrotron.mojo)). The emission also uses AM³'s step approximation of inverse Compton (`HadIC.Calc_KIC` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo)).

## Decays

The kaons $\mathrm K^\pm$, $\mathrm K^0_\mathrm S$ and $\mathrm K^0_\mathrm L$ decay into pions, muons, electrons, positrons and neutrinos. This section derives the sources of these products. It starts from the branching ratios and matrix elements of the [PDG 2026](https://pdg.lbl.gov/2026/) listings and ends with the tables on the code's energy grid. The semileptonic decays $\mathrm K_ {l3}$ are the decays $\mathrm K \to \pi\ell\nu$, where $\ell$ is an electron or a muon. Their density comes from [Chounet, Gaillard & Gaillard (1972)](https://doi.org/10.1016/0370-1573%2872%2990018-X). [Lipari, Lusignoli & Meloni (2007)](https://arxiv.org/abs/0704.0718) give the muon helicity.

### Modes and branching ratios

Modes: The code follows every non-radiative mode (one without an extra photon) with a branching ratio above 0.1% in the [PDG 2026](https://pdg.lbl.gov/2026/) listings. It also follows the smaller $\mathrm K^0_\mathrm S$ semileptonic and $\mathrm K^0_\mathrm L \to \pi^0\pi^0$ modes. Together they cover 99.995% of $\mathrm K^\pm$ decays, all of $\mathrm K^0_\mathrm S$ and 99.93% of $\mathrm K^0_\mathrm L$. In each radiative mode above 0.1%, most decays have a low-energy photon. PDG counts the decays with a low-energy photon in the parent mode. The $\mathrm K^-$ decays are the CP conjugates of $\mathrm K^+$ decays (a left-handed $\mu^+$ becomes a right-handed $\mu^-$).

The branching ratios are [PDG](https://pdg.lbl.gov/2026/)'s fits. The last column gives the page of the kaon's listing and PDG's label $\Gamma_i$ of the mode. This label is the symbol of the mode's partial width. The ratio $\Gamma_i/\Gamma$ of this width to the total width is the branching ratio. The function `kaon_decay_modes` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) holds these modes and branching ratios.

| Kaon | Mode | Branching ratio | [PDG 2026](https://pdg.lbl.gov/2026/) listing |
|---|---|---|---|
| $\mathrm K^+$ | $\mu^+\nu_\mu$ | 0.6356 | p. 12, $\Gamma_2$ |
| | $\pi^+\pi^0$ | 0.2067 | p. 16, $\Gamma_{10}$ |
| | $\pi^+\pi^+\pi^-$ | 0.05583 | p. 17, $\Gamma_{12}$ |
| | $\pi^0\mathrm e^+\nu_e$ ($\mathrm K^+_ {e3}$) | 0.0507 | p. 12, $\Gamma_3$ |
| | $\pi^0\mu^+\nu_\mu$ ($\mathrm K^+_ {\mu 3}$) | 0.03352 | p. 13, $\Gamma_4$ |
| | $\pi^+\pi^0\pi^0$ | 0.01760 | p. 17, $\Gamma_{11}$ |
| $\mathrm K^0_\mathrm S$ | $\pi^+\pi^-$ | 0.6920 | p. 4, $\Gamma_2$ |
| | $\pi^0\pi^0$ | 0.3069 | p. 4, $\Gamma_1$ |
| | $\pi^\mp\mathrm e^\pm\nu$ ($\mathrm K_ {e3}$) | $7.14 \times 10^{-4}$ | p. 7, $\Gamma_9$ |
| | $\pi^\mp\mu^\pm\nu$ ($\mathrm K_ {\mu 3}$) | $4.56 \times 10^{-4}$ | p. 7, $\Gamma_{10}$ |
| $\mathrm K^0_\mathrm L$ | $\pi^\mp\mathrm e^\pm\nu$ ($\mathrm K_ {e3}$) | 0.4055 | p. 6, $\Gamma_1$ |
| | $\pi^\mp\mu^\pm\nu$ ($\mathrm K_ {\mu 3}$) | 0.2704 | p. 6, $\Gamma_2$ |
| | $3\pi^0$ | 0.1952 | p. 8, $\Gamma_6$ |
| | $\pi^+\pi^-\pi^0$ | 0.1254 | p. 9, $\Gamma_7$ |
| | $\pi^+\pi^-$ | $1.967 \times 10^{-3}$ | p. 9, $\Gamma_8$ |
| | $\pi^0\pi^0$ | $8.64 \times 10^{-4}$ | p. 11, $\Gamma_9$ |

The $\mathrm K^0_\mathrm L$ semileptonic decays give $\pi^-\ell^+\nu$ with probability $\left(1 + A_{L}\right)/2$ and $\pi^+\ell^-\bar\nu$ with $\left(1 - A_{L}\right)/2$. [PDG](https://pdg.lbl.gov/2026/)'s charge asymmetry $A_{L}$ is the difference of the two rates over their sum ($\mathrm K^0_\mathrm L$ listing, pp. 34–35). Its averages are $A_L = 3.34 \times 10^{-3}$ ($\mathrm e$, p. 35) and $3.04 \times 10^{-3}$ ($\mu$, p. 34), for electrons and muons. The code splits the $\mathrm K^0_\mathrm S$ semileptonic decays evenly between the two final states.

Products per decay: Let $B_m$ be the branching ratio of mode $m$. A kaon of species $s$ then makes the products $X$ with the energy distribution

$$
F_{s\to X}(x) = \frac{\sum_m B_m\thinspace \nu_{mX}\thinspace F_{mX}(x)}{\sum_m B_m}
$$

in the energy fraction $x$, the product's share of the kaon energy, where both sums run over the modes of the table. Here $\nu_{mX}$ is the number of $X$ in a decay of mode $m$. It is 2 for the $\pi^+$ of $\pi^+\pi^+\pi^-$ and 3 for the $\pi^0$ of $3\pi^0$, and the probabilities above for the $\mathrm K^0_\mathrm L$ semileptonic modes. The distribution $F_{mX}$ of one such product is normalized to 1. The denominator divides the branching ratios by their sum, so the products of a decay carry the whole kaon energy. The functions `kaon_decay_terms` and `charge_conjugate` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) give the terms $B_m\nu_{mX}$ divided by the sum of the branching ratios, for each species.

### Matrix elements

Two-body decays need no matrix element, because each product has a single energy in the kaon frame (see below). Only the muon of $\mathrm K^+ \to \mu^+\nu_\mu$ carries a polarization. Angular momentum fixes this polarization (see [Muon helicity](#muon-helicity)).

$\mathbf{K}_ {\boldsymbol{l3}}$ **decays** ($\mathrm K \to \pi\ell\nu$) use the V−A (vector minus axial-vector) matrix element of the weak interaction, with the hadronic current $H = f_+(t)(p_K + p_\pi) + f_-(t)(p_K - p_\pi)$, where $t = (p_K - p_\pi)^2$. Here $p_K$ and $p_\pi$ are the four-momenta of the kaon and the pion, and $t$ is the squared four-momentum transferred to the leptons. The form factors $f_+$ and $f_{-}$ are given below. For an $\ell^+$ with the spin four-vector $s$ (not a kaon species here), the squared matrix element is

$$
\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s),\qquad W_0 = 2\thinspace (p_\nu\negthinspace \cdot\negthinspace H)(p_\ell\negthinspace \cdot\negthinspace H) - H^2\thinspace (p_\nu\negthinspace \cdot\negthinspace p_\ell),\qquad W_{\rm pol}(s) = m_\ell\left[2\thinspace (p_\nu\negthinspace \cdot\negthinspace H)(s\negthinspace \cdot\negthinspace H) - H^2\thinspace (p_\nu\negthinspace \cdot\negthinspace s)\right],
$$

and, summed over its spin, $\vert\mathcal M\vert^2 \propto 2W_0$. Here $p_\nu$ and $p_\ell$ are the four-momenta of the neutrino and the charged lepton, $m_\ell$ is the lepton's mass, and the dots are Minkowski products. For helicity $+1/{2}$, the spin vector $s$ points along the lepton's momentum in the kaon frame. For helicity $-1/{2}$, the spin vector is $-s$. The trace of the V−A amplitude with this hadronic current gives $W_0 \pm W_{\rm pol}(s)$ for these two helicities, so $W_\mathrm{pol}$ gives the muon polarization. In the amplitude, the $f_{-}$ term reduces to a scalar current proportional to $m_\ell$.

The term $W_0$ is the density of [Chounet, Gaillard & Gaillard](https://doi.org/10.1016/0370-1573%2872%2990018-X) (1972, Phys. Rep. 4, 199). Their eq. 6.7 gives $t$ in the kaon frame. Eq. 6.8 is the current $H$, in which they write $k$ and $q$ for $p_K$ and $p_\pi$. The Dalitz plot is the plane of two of the products' energies. Eq. 6.9 (pp. 249–250) is the density $\rho$ over this plot in the kaon frame:

$$
\rho \propto Af_+^2 + Bf_+f_- + Cf_-^2,\qquad A = M(2E_\mu E_\nu - ME'_ \pi) + m^2\left(\tfrac14 E'_ \pi - E_\nu\right),\qquad B = m^2\left(E_\nu - \tfrac12 E'_ \pi\right),\qquad C = \tfrac14 m^2E'_ \pi,
$$

in their notation: the kaon mass $M$, the pion mass $\mu$ and the lepton mass $m$. Their $\mu$ is not a muon, and their $A, B, C$ are not the parameters of the multiplicity fit of the $\mathrm{pp}$ collisions. Their $E_\pi$, $E_\mu$ and $E_\nu$ are the energies of the pion, the lepton and the neutrino, and $E'_ \pi$ is $E_\pi^{\max} - E_\pi$, with $E_\pi^{\max}$ $= (M^2 + \mu^2 - m^2)/(2M)$. They call the lepton's energy $E_\mu$ even for $\mathrm K_ {e3}$, the decay with an electron. With these energies, $W_0 = 4M\thinspace\rho$ at every point of the plot.

The form factors are those of the [PDG 2026](https://pdg.lbl.gov/2026/) fits, with $f_- = (f_0 - f_+)(M^2 - m_\pi^2)/t$ from the vector form factor $f_+$ and the scalar form factor $f_0$. Here $m_\pi$ is the mass of the mode's pion: the $\pi^0$ for $\mathrm K^\pm$ and the $\pi^\pm$ for $\mathrm K^0$. The form factors themselves are PDG's expansions in $t$ ($\mathrm K^\pm$ listing, p. 37, and $\mathrm K^0_\mathrm L$ listing, p. 23), normalized to $f_+(0)$:

$$
f_+(t) = 1 + \lambda'_ +\frac{t}{m_{\pi^+}^2} + \frac{\lambda''_ +}{2}\left(\frac{t}{m_{\pi^+}^2}\right)^2,\qquad f_0(t) = 1 + \lambda_0\frac{t}{m_{\pi^+}^2},
$$

Here $\lambda'_ {+}$ and $\lambda''_ {+}$ are the linear and quadratic slopes of $f_+$, and $\lambda_0$ is the slope of $f_0$. The charged pion's mass $m_{\pi^+}$ scales $t$ in all expansions, and the fits assume $\mu\text{–}\mathrm e$ universality, the same form factors for muons and electrons. The $\mathrm K^0$ values are $\lambda'_ + = 0.0240$, $\lambda''_ + = 0.0020$ and $\lambda_0 = 0.0116$ (quadratic fit, $\mathrm K^0_\mathrm L$ listing, pp. 25–27). The $\mathrm K^\pm$ values are $\lambda'_ + = 0.02959$, $\lambda''_ + = 0$ and $\lambda_0 = 0.0176$ ($\mathrm K^\pm$ listing, pp. 38–39). They come from the linear fit, because [PDG](https://pdg.lbl.gov/2026/) has no $\mathrm K^\pm$ quadratic fit with universality. The relation for $f_{-}$ is PDG's definition $f_0 = f_+ + f_-\thinspace t/(m_K^2 - m_\pi^2)$ solved for $f_{-}$, with $M$ for the kaon mass $m_K$. With the expansions, $f_{-}$ is finite where $t$ vanishes:

$$
f_-(t) = (M^2 - m_\pi^2)\left[\frac{\lambda_0 - \lambda'_ +}{m_{\pi^+}^2} - \frac{\lambda''_ +}{2}\thinspace\frac{t}{m_{\pi^+}^4}\right].
$$

The functions `KDecayMode._kl3` and `l3_density` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluate $f_+$, this $f_{-}$ and the densities $W_0$ and $W_{\rm pol}$.

$\boldsymbol{\mathbf{K} \to 3\pi}$ uses the [PDG](https://pdg.lbl.gov/2026/) Dalitz-plot expansion

$$
\vert \mathcal M\vert^2 \propto 1 + g\thinspace u + h\thinspace u^2 + k\thinspace v^2,\qquad u = \frac{s_3 - s_0}{m_{\pi^+}^2},\quad v = \frac{s_2 - s_1}{m_{\pi^+}^2},\quad s_i = (p_K - p_i)^2,\quad s_0 = \tfrac13\textstyle\sum_i s_i,
$$

where pion 3 is the odd one (the one whose charge differs), $p_i$ are the pions' four-momenta, and $g, h, k$ are [PDG](https://pdg.lbl.gov/2026/)'s slope parameters. This is eq. 1 of PDG's note on the Dalitz-plot parameters ($\mathrm K^\pm$ listing, p. 32) without its CP-violating terms. Here $s_i$ is the squared invariant mass of the two pions other than $i$, and $s_0$ is the mean of the three. The indexed $s$ is neither a species nor a spin vector. The coefficients ($\mathrm K^\pm$ listing, pp. 33–35, and $\mathrm K^0_\mathrm L$ listing, pp. 21–22) are $\left(g, h, k\right)$ $= (-0.21134, 0.01848, -0.00463)$ for $\mathrm K^+ \to \pi^+\pi^+\pi^-$, $\left(0.626, 0.052, 0.0054\right)$ for $\mathrm K^+ \to \pi^+\pi^0\pi^0$ and $\left(0.678, 0.076, 0.0099\right)$ for $\mathrm K^0_\mathrm L \to \pi^+\pi^-\pi^0$. For this last mode, the CP-violating terms $j\thinspace v + f\thinspace u\thinspace v$ have coefficients consistent with zero and are left out. The decay $\mathrm K^0_\mathrm L \to 3\pi^0$ has $1 + h(u^2 + v^2/3)$, PDG's expansion made symmetric in the three $\pi^0$, with $h = 0.00059$ ($\mathrm K^0_\mathrm L$ listing, p. 22). The method `KDecayMode._k3pi` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluates these expansions.

### Decay at rest

Phase space: In the kaon frame, the products of a three-body decay have the energies $E_1, E_2, E_3$. These energies add up to the kaon mass $M$. Any two of them, $E_i$ and $E_j$, fix the decay up to its orientation. The phase space is uniform in these two energies, so the decay rate is $d\Gamma \propto \rho\thinspace dE_i\thinspace dE_j$, with the density $\rho \propto \vert\mathcal M\vert^2$ over this Dalitz plot.

Kaon-frame products: The function `l3_density` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluates $W_0$ and $W_{\rm pol}$ from the energies. The kaon is at rest. Two decay products $a$ and $b$ together have the four-momentum $p_K$ minus that of the third product, $c$. So

$$
p_K\negthinspace\cdot\negthinspace p_a = ME_a,\qquad 2\thinspace p_a\negthinspace\cdot\negthinspace p_b = M^2 - 2ME_c + m_c^2 - m_a^2 - m_b^2,
$$

where $a$, $b$ and $c$ are the pion, the lepton and the neutrino in any order. Their energies are $E_a, E_b, E_c$, and their masses are $m_a, m_b, m_c$. The mass of the neutrino is zero. For helicity $+1/{2}$, the lepton's spin four-vector and its dot products are

$$
s = \frac{1}{m_\ell}\left(\vert\vec p_\ell\vert,\thinspace E_\ell\thinspace\hat p_\ell\right),\qquad p_a\negthinspace\cdot\negthinspace s = \frac{E_a\vert\vec p_\ell\vert - E_\ell\thinspace\vec p_a\negthinspace\cdot\negthinspace\hat p_\ell}{m_\ell},\qquad 2\thinspace\vec p_a\negthinspace\cdot\negthinspace\vec p_\ell = \vert\vec p_b\vert^2 - \vert\vec p_a\vert^2 - \vert\vec p_\ell\vert^2.
$$

Here $E_\ell$ and $\vec p_\ell$ are the lepton's energy and three-momentum, and $\hat p_\ell$ is the direction of that momentum. In the second relation, $a$ is the kaon, the pion or the neutrino. The kaon has no three-momentum. In the third relation, $a$ and $b$ are the pion and the neutrino, and the relation comes from the square of $\vec p_b = -\vec p_a - \vec p_\ell$. Each momentum follows from its energy and mass.

Dalitz lines: Product $i$ has the mass $m$, the kaon-frame energy $E^\ast$ and the momentum $p^\ast$. This product recoils against the pair of the other two products, $j$ and $k$, with the masses $m_j, m_k$. The pair has the invariant mass $m_{jk}$. In the rest frame of the pair, product $j$ has the energy $E_j'$ and the momentum $p_j'$. A Dalitz line is a line across the plot at fixed $E^\ast$. The boost of the pair to the kaon frame spreads $E_j$ between the ends of this line:

$$
E_j^{\rm lo,hi} = \frac{(M - E^\ast)\thinspace E_j' \mp p^\ast p_j'}{m_{jk}},\qquad m_{jk}^2 = M^2 + m^2 - 2ME^\ast,\qquad E_j' = \frac{m_{jk}^2 + m_j^2 - m_k^2}{2m_{jk}},\qquad E_{\max} = \frac{M^2 + m^2 - (m_j + m_k)^2}{2M}.
$$

At the ends of the line, the three momenta are collinear. The line shrinks to a point at the largest energy $E_{\max}$ of product $i$, where $m_{jk}$ is $m_j + m_k$ and $p_j'$ vanishes. The methods `KDecayMode.energy_density` and `KDecayMode.emax` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluate these ends and $E_{\max}$.

Energy distributions: The kaon-frame energy distribution $g$ of product $i$ is the integral of the density along the line of $E^\ast$. In the same way, $G$ integrates the polarization of a muon, weighted with the rate:

$$
g(E^\ast) = \frac{1}{Z}\int_{E_j^{\rm lo}}^{E_j^{\rm hi}} \rho\thinspace dE_j,\qquad G(E^\ast) = \frac{1}{Z}\int_{E_j^{\rm lo}}^{E_j^{\rm hi}} \rho\thinspace\mathcal P\thinspace dE_j,\qquad Z = \iint\rho\thinspace dE_i\thinspace dE_j.
$$

Here $\mathcal P$ is the muon's longitudinal polarization in the kaon frame. The integral $Z$ over the whole plot is the same for each product. For $\mathrm K_ {l3}$, $\rho$ is $W_0$ and $\rho\mathcal P$ is $W_{\rm pol}(s)$, with the spin vector $s$ of helicity $+1/{2}$. For $\mathrm K \to 3\pi$, $\rho$ is the expansion above.

Quadrature: Along a Dalitz line, the densities are polynomials in $E_j$: of degree 2 for $\mathrm K \to 3\pi$ and at most 5 for $\mathrm K_ {l3}$. The code's 24-point Gauss–Legendre rule along the line is therefore exact. The density $W_{\rm pol}$ is needed only on the muon's line, where $\vert\vec p_\ell\vert$ is fixed. As a function of $E^\ast$, the line's length $E_j^{\rm hi} - E_j^{\rm lo}$ is proportional to $p^\ast p_j'$ and vanishes at both ends of $[m, E_{\max}]$. The length vanishes like a square root at the bottom if product $i$ is massive, and at the top if the other two are. At the other edges, it vanishes linearly. A new variable $u$, not the Dalitz $u$ above, smooths these edges. The substitution $E^\ast = m + (E_{\max} - m)\sin^2(\pi u/2)$, with $0 \le u \le 1$, turns the square roots into sines and cosines of $\pi u/2$, and the linear edges into their squares. The methods `GLRules.rest_node` and `KDecayMode.normalise` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) apply this substitution, and the second method uses it to compute $Z$.

### From the kaon frame to the lab frame

Consider an ultra-relativistic kaon of mass $M$ that decays isotropically in its rest frame. A product of mass $m$ has the energy $E^\ast$ and the momentum $p^\ast$ in the kaon frame. In the lab, the product takes the share $x$ of the kaon energy. This share is spread uniformly over $x \in [(E^\ast - p^\ast)/M, (E^\ast + p^\ast)/M]$ with density $M/(2p^\ast)$. With $g(E^\ast)$ as the product's normalized energy distribution in the kaon frame, up to the largest energy $E_{\max}$, the distribution in $x$ is

$$
F(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}^{E_{\max}} \frac{g(E^\ast)}{p^\ast}\thinspace dE^\ast,\qquad E_{\rm lo}(x) = \frac{M^2x^2 + m^2}{2Mx}\ \ (\ge m).
$$

Here $E_{\rm lo}(x)$ is the lowest $E^\ast$ whose range contains $x$. Two-body decays have one $E^\ast = (M^2 + m^2 - m_b^2)/(2M)$, so $F$ is flat. In this energy, $m_b$ is the mass of the other product. For a three-body decay, $g$ is the distribution above. The code integrates $F$ over the bins of its grid (see [Source terms on the grid](#source-terms-on-the-grid)).

### Muon helicity

Take a muon with the longitudinal polarization $\mathcal P$ in the kaon frame. Averaged over the orientation of the decay plane, its lab helicity is $\mathcal P\thinspace \tilde h(x, E^\ast)$, with $\tilde h = (E^\ast M x - m^2)/(p^\ast M x)$. The right-handed ($R$) and left-handed ($L$) muons in the lab therefore have the distributions

$$
F_{R,L}(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}^{E_{\max}} \frac{g(E^\ast) \pm G(E^\ast)\thinspace \tilde h(x,E^\ast)}{2\thinspace p^\ast}\thinspace dE^\ast,
$$

where $G$ is the polarization weighted with the rate ($\rho\mathcal P$ integrated along the Dalitz line), and the upper sign is for $R$. The helicity matters because a muon's decay spectrum depends on it.

For $\mathrm K \to \mu\nu$, $\mathcal P = -1$ for the $\mu^+$. That is, this muon is left-handed in the kaon frame. The kaon has no spin and the neutrino is left-handed, so the muon must be left-handed too. Let $r = m_\mu^2/M^2$, where $m_\mu$ is the muon mass. For $r \le x \le 1$, $F_{R,L}$ then gives

$$
F_{\mu_R^+}(x) = \frac{r\thinspace (1 - x)}{(1 - r)^2\thinspace x},\qquad F_{\mu_L^+}(x) = \frac{x - r}{(1 - r)^2\thinspace x}.
$$

These are eqs. 18–19 of [Lipari, Lusignoli & Meloni (2007)](https://arxiv.org/abs/0704.0718) for $\pi^+ \to \mu^+\nu_\mu$. As the authors note, these equations also hold for $\mathrm K^+ \to \mu^+\nu_\mu$ with the kaon mass. Their eq. 16 gives the helicity $h$ of these muons (not the Dalitz parameter $h$), with $h = -1$ at $x = 1$. This helicity equals $\mathcal P\thinspace\tilde h$. Their eq. 19 also gives the conjugate decay, with $\mu^+_ R$ printed for $\mu^-_ R$.

### Source terms on the grid

Source: Let an ultra-relativistic particle $a$ of energy $E_a$ decay into a product $b$ of energy $E_b$. The spectrum of $b$ is its number per decay and per unit energy. In their eq. 13, [Lipari, Lusignoli & Meloni (2007)](https://arxiv.org/abs/0704.0718) write this spectrum in the scaling form

$$
\frac{dn_{a \to b}}{dE}(E_b; E_a) = \frac{1}{E_a}\thinspace F_{a \to b}\negthinspace\left(\frac{E_b}{E_a}\right).
$$

A kaon of species $s$ has the mass $m_s$, the energy $E_{K}$ and the Lorentz factor $\gamma_K = E_K/(m_s c^2)$. It decays at the rate $1/(\gamma_K\tau_s)$, where $\tau_s$ is its mean life. Its products $X$ have the distribution $F_{s\to X}(x) = dn/dx$ in the energy fraction $x = E/E_K$. Let $N_s$ be the density of the kaons per unit energy. The source of the products, per unit energy and time, is then

$$
Q_X(E) = \sum_s \int_E^\infty \frac{dE_K}{E_K}\thinspace \frac{N_s(E_K)}{\gamma_K\thinspace \tau_s}\thinspace F_{s\to X}\negthinspace \left(\frac{E}{E_K}\right).
$$

Per unit $\ln E$: The code holds densities and sources per unit $\ln E$, such as $E_K N_s$ and $E\thinspace Q_X$. As an integral over $\ln E_K$, the source is

$$
E\thinspace Q_X(E) = \sum_s \int_{\ln E}^\infty \frac{E_KN_s(E_K)}{\gamma_K\thinspace \tau_s}\thinspace T_{s\to X}\negthinspace \left(\frac{E}{E_K}\right) d\ln E_K,\qquad T(x) = x\thinspace F(x),
$$

where $T$ is the number of products per decay and per unit $\ln x$.

On the grid: The code's grids are logarithmic with the step $\Delta$. The hadron grid has the energies $E_j = e^{j\Delta}$ GeV. Every product grid has its points among the energies $e^{n\Delta}$ GeV, where $n$ is an integer. So a product at $E_i$ from a kaon at $E_j$ has the energy fraction $x_k = e^{-k\Delta}$, where $k$ is an integer. On the hadron grid, the integral becomes the sum

$$
E_iQ_X(E_i) = \Delta\sum_s\sum_j \frac{E_jN_s(E_j)}{\gamma_j\thinspace \tau_s}\thinspace T_{s\to X,\thinspace k},\qquad x_k = e^{-k\Delta} = \frac{E_i}{E_j},
$$

where $\gamma_j$ is the kaon's Lorentz factor at $E_j$, and $T_{s\to X,\thinspace k}$ is the table of $T$ on the grid (see below). The methods `KaonDecay.initialize_kernels` and `KaonDecay.Calc_KtoX` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) store the table and evaluate this sum. The methods `SimulationManager.Calc_K` and `SimulationManager.Calc_C` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) use these sources. The second method uses them for the kaons from charm decays.

The table: Let $n_q$ be the number of products between two grid points, $x_{q+1} \lt x \lt x_q$. Let $e_q$ be their energy, in units of $E_{K}$. The function `kaon_decay_tables` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) builds $T_k$ by sharing these products between the two points, linearly in $x$:

$$
T_k = \frac{e_k - x_{k+1}n_k}{\Delta\thinspace (x_k - x_{k+1})} + \frac{x_{k-1}n_{k-1} - e_{k-1}}{\Delta\thinspace (x_{k-1} - x_k)},\qquad n_q = \int_{x_{q+1}}^{x_q} F\thinspace dx,\qquad e_q = \int_{x_{q+1}}^{x_q} xF\thinspace dx.
$$

Summed over $k$, $\Delta\thinspace T_k$ gives the number $\sum_q n_q$ of products per decay. In the same way, $\Delta\thinspace x_kT_k$ gives their energy $\sum_q e_q$. No entry of the table is negative, because the mean $e_q/n_q$ of $x$ in a bin lies between the two points of the bin. Products below the lowest point of their grid are lost.

Bin integrals: Each node $E^\ast$ of the quadrature of $g$ has a flat spectrum of weight $w_g$. The single energy of a two-body decay also has such a spectrum. A muon of either helicity adds $w_G\thinspace\tilde h$, with a second weight $w_G$. Over the overlap $[x_a, x_b]$ of the node's range with a bin, these spectra have integrals in closed form. The function `_bin_node` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) adds them to $n_q$ and $e_q$:

$$
\int_{x_a}^{x_b} \frac{M}{2p^\ast}\left(w_g + w_G\thinspace\tilde h\right) dx = \frac{M}{2p^\ast}\left[\left(w_g + \frac{E^\ast}{p^\ast}w_G\right)(x_b - x_a) - \frac{m^2w_G}{p^\ast M}\ln\frac{x_b}{x_a}\right],
$$

$$
\int_{x_a}^{x_b} \frac{M}{2p^\ast}\left(w_g + w_G\thinspace\tilde h\right) x\thinspace dx = \frac{M}{2p^\ast}\left[\left(w_g + \frac{E^\ast}{p^\ast}w_G\right)\frac{x_b^2 - x_a^2}{2} - \frac{m^2w_G}{p^\ast M}(x_b - x_a)\right].
$$

Let $\omega_n$ be the quadrature weight of the node $E^\ast_n$. Let $\mathcal B$ be the product's term $B_m\nu_{mX}$ divided by the sum of the branching ratios. A right- or left-handed muon then has

$$
w_g = \tfrac12\mathcal B\thinspace g(E^\ast_n)\thinspace\omega_n,\qquad w_G = \pm\tfrac12\mathcal B\thinspace G(E^\ast_n)\thinspace\omega_n,
$$

with the upper sign for $R$. These weights give the quadrature of $F_{R,L}$. Another product has twice this $w_g$ and no $w_G$. In a two-body decay, $g\thinspace\omega_n$ is 1 and $G$ is $\mathcal P g$.

