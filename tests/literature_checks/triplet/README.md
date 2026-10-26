# Triplet pair production

An electron or positron that hits a photon makes an $\mathrm{e}^+\mathrm{e}^-$ pair, $\mathrm{e}^\pm\gamma$ $\to \mathrm{e}^\pm\mathrm{e}^+\mathrm{e}^-$. This happens if the photon's energy in the lepton's rest frame exceeds $4m_e c^2$, four times the electron's rest energy. This page starts from the equations and tables of the sources below and from the quantum electrodynamics (QED) Monte Carlo of [tools/triplet](../../../tools/triplet/README.md). From them it derives the equations and tables that the Mojo code uses, and it names the struct or function that uses each. The module [triplet.py](triplet.py) verifies every derivation: `.venv/bin/python tests/literature_checks/common/run.py triplet`. The scripts named at the steps check the Monte Carlo's tables and measure the code's approximations.

**Sources.**
- [Haug (1975)](https://doi.org/10.1515/zna-1975-0901) computes the lowest-order cross section from all eight Feynman diagrams. His Table 1 gives the total cross section at 58 photon energies, and his eq. 9.1 gives Votruba's threshold form.
- [Lee, Lyubyakin & Stotsky (2021)](https://doi.org/10.1007/JHEP01%282021%29144) give the exact total cross section (their eqs. 4.2–4.4). They also give its threshold expansion (eq. 4.5) and its high-energy expansions (eqs. 4.11–4.13). This page cites the equation numbers of the open-access journal version.
- [Blumenthal & Gould (1970)](https://doi.org/10.1103/RevModPhys.42.237) give the inverse-Compton energy loss in the Klein–Nishina limit (their eq. 2.57).

## Kinematics

Energies are in units of the electron's rest energy $m_e c^2$, where $m_e$ is the electron mass and $c$ the speed of light. Four-vectors are in units with $m_e = c = 1$. In the lab frame, a lepton of Lorentz factor $\gamma$ and speed $\beta c$ meets a photon of energy $\varepsilon$. The angle between their directions is $\theta$. The four-momenta of the lepton and the photon are $p_1$ and $k$. The total cross section depends only on the invariant $s = (p_1 + k)^2$, the square of the total energy in the centre-of-momentum frame. [Haug](https://doi.org/10.1515/zna-1975-0901) (p. 1109) and [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144) write it as a function of the photon's energy in the lepton's rest frame. They also call this energy $k$. This page does the same: $k$ is the photon's four-momentum in four-vector expressions, such as $p_1\cdot k$, and this energy everywhere else:

$$
s = (p_1 + k)^2 = 1 + 2k,\qquad k = p_1\cdot k = \gamma\varepsilon\thinspace (1 - \beta\cos\theta).
$$

For $\gamma \gg 1$, the photon's energy in the lepton's rest frame is $k = \gamma\varepsilon(1 - \cos\theta)$. At threshold, the three leptons of the final state are at rest in the centre-of-momentum frame. Their total energy there is then 3, which gives $s = 9$, so $k = 4$.

## Cross section

### The exact result

[Lee, Lyubyakin & Stotsky](https://doi.org/10.1007/JHEP01%282021%29144) compute the total cross section through the optical theorem. Their eq. 4.4 writes it with twelve functions $J_1, \ldots, J_{12}$ of $s$. The first two, $J_1$ and $J_2$, are complete elliptic integrals. Eq. 4.3 gives the other ten as one-fold integrals over $J_1$. These integrals solve the differential system of eq. 4.2 and vanish at $s = 9$. Like [Haug](https://doi.org/10.1515/zna-1975-0901)'s, their result is of lowest order in QED. The code therefore neglects the radiative corrections, which are of relative order $\alpha$, the fine-structure constant.

As printed, the $J_{11}$ of eq. 4.3 does not solve eq. 4.2, which requires $J_{11}' = zJ_9/(s - 5)$. Here the prime is the derivative in $s$, and $z = \sqrt{(s-5)/(s-1)}$. The integrand of $J_{11}$ depends on $s$ through $\ln(\sqrt{s-5} + \sqrt{s-1})$, whose derivative in $s$ is $1/[2\sqrt{(s-5)(s-1)}]$. That is half of $z/{(s - 5)}$, so $J_{11}$ needs a factor 2. Both the journal version and arXiv v1 print $J_{11}$ without it. The other nine integrals of eq. 4.3 solve eq. 4.2 as printed. The script [compare_exact.py](compare_exact.py) evaluates eq. 4.4 with the factor 2, in 30-digit arithmetic. The result agrees with the threshold expansion below to $3 \times 10^{-10}$ at $k = 4.001$.

### Near threshold

Eq. 4.5 of [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144) expands the cross section in powers of $r = (s-9)/(s-1)$, a series that converges for every $s \gt 9$. It gives the terms up to $r^4$, and $O(r^5)$ stands for the higher ones:

$$
\sigma = \frac{\pi}{3\sqrt3}\left[\frac{4r^2}{27} + \frac{602\thinspace r^3}{243} + \frac{4217\thinspace r^4}{4374} + O(r^5)\right].
$$

In terms of $k$, $r = 1 - 4/k$. As printed, eq. 4.5 omits the factor $\alpha^3$ that eqs. 4.4 and 4.11–4.13 carry. [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144) use units that set the reduced Planck constant, the speed of light and the electron mass to one: $\hbar = c = m_e = 1$. A cross section $\alpha^3$ is then $\alpha^3\hbar^2/(m_e c)^2 = \alpha r_e^2$ in ordinary units, as the classical electron radius is $r_e = \alpha\hbar/(m_e c)$. The code's threshold expansion is thus eq. 4.5 in units of $\alpha r_e^2$, used in the range given in brackets:

$$
\frac{\sigma}{\alpha r_e^2} = \frac{\pi}{3\sqrt3}\left(\frac{4r^2}{27} + \frac{602\thinspace r^3}{243} + \frac{4217\thinspace r^4}{4374}\right),\qquad r = \frac{s-9}{s-1},\quad s = 1 + 2k\qquad (k \le 4.02).
$$

Its leading term for $k \to 4$, $(\sqrt3\pi/972)\thinspace \alpha r_e^2 (k-4)^2$, is Votruba's threshold approximation. [Haug](https://doi.org/10.1515/zna-1975-0901) gives it as his eq. 9.1, with $r_0$ for $r_e$. His Table 1 lists it next to the exact values near threshold.

### At high energy

Eq. 4.11 of [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144) expands the cross section in powers of $1/{s}$, to first order:

$$
\sigma = \alpha^3\left[\frac{28\ln s}{9} - \frac{218}{27} - \frac{1}{s}\left(\frac{8}{3}\ln^3 s - \frac{49}{6}\ln^2 s - \left(2\pi^2 - \frac{409}{9}\right)\ln s + 12\zeta_3 + \frac{31\pi^2}{18} - \frac{1081}{12}\right)\right],
$$

where $\zeta_3$ is the Riemann zeta function at 3. Their eq. 4.12 writes this expansion in $k$. It follows from eq. 4.11 with $\ln s = \ln 2k + \ln[1 + 1/(2k)]$, expanded again to order $1/{k}$:

$$
\sigma = \alpha^3\left[\frac{28}{9}\ln 2k - \frac{218}{27} + \frac{1}{k}\left(-\frac{4}{3}\ln^3 2k + \frac{49}{12}\ln^2 2k + \left(\pi^2 - \frac{409}{18}\right)\ln 2k - 6\zeta_3 - \frac{31\pi^2}{36} + \frac{3355}{72}\right)\right].
$$

Their eq. 4.13 rounds the coefficients of eq. 4.12 to two decimals. In units of $\alpha r_e^2$, eq. 4.13 is the code's high-energy expansion, again used in the range given in brackets:

$$
\frac{\sigma}{\alpha r_e^2} = \frac{28}{9}\ln 2k - \frac{218}{27} - \frac{1}{k}\left(\frac{4}{3}\ln^3 2k - 4.08\ln^2 2k + 12.85\ln 2k - 30.89\right)\qquad (k \ge 10^3).
$$

### In the code

The Monte Carlo of [tools/triplet](../../../tools/triplet/README.md) integrates the exact lowest-order QED matrix element of [Haug](https://doi.org/10.1515/zna-1975-0901)'s eight diagrams over all final states. The Monte Carlo's tables give $\sigma(k)$ at 60 values of $k$, from 4.02 to $10^{10}$, and the energy distributions of the products there. These values of $k$ are the tables' nodes. The tables are `TPP_SIG` and `TPP_DIST` in [triplet_tables.mojo](../../../src/am3_mojo/physics/triplet_tables.mojo). Below $k = 4.02$ and above $k = 10^3$, the code uses the expansions of [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144) for $\sigma(k)$. From $k = 10^3$ on, it thus takes only the products' distributions from the tables. The method `TripletPairProduction.sigma` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) puts these pieces together:

| Photon energy | The code's $\sigma(k)$ | Code |
|---|---|---|
| $k \le 4$ | zero (below threshold) | `TripletPairProduction.sigma` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) |
| $4 \lt k \le 4.02$ | eq. 4.5 | `sigma_threshold` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) |
| $4.02 \lt k \lt 10^3$ | a cubic spline through the table's nodes | `TripletPairProduction._fit_sigma_spline` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) |
| $k \ge 10^3$ | eq. 4.13 | `sigma_asym` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) |

The spline interpolates $v = \ln\sigma$ in $u = \ln(k-4)$, in which the cross section's rise as $(k - 4)^2$ at threshold is a straight line. Its nodes $k_n$, from 4.02 to $10^3$, are numbered by $n$ from 0. At the node $n$, $u$ and $v$ take the values $u_n$ and $v_n$. Between the nodes $n$ and $n + 1$, with the step $h_n = u_{n+1} - u_n$ and $t = (u - u_n)/h_n$, the spline is

$$
v = (1-t)\thinspace v_n + t\thinspace v_{n+1} + \frac{h_n^2}{6}\Bigl[\bigl((1-t)^3 - (1-t)\bigr)M_n + (t^3 - t)\thinspace M_{n+1}\Bigr].
$$

Here $M_n$ is the spline's second derivative in $u$ at the node $n$. The second derivatives follow from one linear equation per node. The equation of each inner node makes the spline's slope continuous there:

$$
h_{n-1}M_{n-1} + 2\thinspace (h_{n-1} + h_n)\thinspace M_n + h_n M_{n+1} = 6\left(\frac{v_{n+1} - v_n}{h_n} - \frac{v_n - v_{n-1}}{h_{n-1}}\right).
$$

At the end nodes, $k_0 = 4.02$ and $k_N = 10^3$, the spline takes the slopes $v_0'$ and $v_N'$ of the two expansions. Here $N$ is the index of the last node, and a prime is the derivative in $u$:

$$
2h_0M_0 + h_0M_1 = 6\left(\frac{v_1 - v_0}{h_0} - v_0'\right),\qquad h_{N-1}M_{N-1} + 2h_{N-1}M_N = 6\left(v_N' - \frac{v_N - v_{N-1}}{h_{N-1}}\right).
$$

The function `sigma_threshold_slope` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) gives the slope of eq. 4.5. This slope follows from $\sigma \propto r^2 p(r)$, where $p(r) = 4/27 + 602r/243 + 4217r^2/4374$ is the bracket of eq. 4.5 divided by $r^2$, and from $dr/dk = 4/k^2$. With $p'$ the derivative of $p$, the slope is

$$
\frac{d\ln\sigma}{d\ln(k-4)} = \frac{4\thinspace (k-4)}{k^2}\left(\frac{2}{r} + \frac{p'(r)}{p(r)}\right).
$$

The function `sigma_asym_slope` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) gives the slope of eq. 4.13. Eq. 4.13 reads $\sigma/(\alpha r_e^2) = 28L/9 - 218/27 - Q(L)/k$, with $L = \ln 2k$ and $Q$ the cubic polynomial in its bracket. As $dL/dk = 1/k$, the slope is

$$
\frac{d\ln\sigma}{d\ln(k-4)} = \frac{k-4}{\sigma}\left(\frac{28}{9k} - \frac{Q'(L) - Q(L)}{k^2}\right),
$$

with $\sigma$ in units of $\alpha r_e^2$ and $Q'$ the derivative of $Q$. The code's rates are in units of the Thomson cross section, $\sigma_T = 8\pi r_e^2/3$, so they use $\alpha r_e^2/\sigma_T = 3\alpha/(8\pi)$, the constant `ALPHA_RE2` in [constants.mojo](../../../src/am3_mojo/utils/constants.mojo).

The script [compare_exact.py](compare_exact.py) checks the tables and the code's $\sigma(k)$ against eq. 4.4 of [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144), with the factor 2 of $J_{11}$. All the differences quoted here are relative. The Monte Carlo's nodes agree with eq. 4.4 to $1.8 \times 10^{-4}$, with a root-mean-square difference of $8 \times 10^{-5}$ above $k = 10^3$. The code's $\sigma(k)$ agrees with eq. 4.4 to $2 \times 10^{-6}$ in the threshold expansion, $1.8 \times 10^{-4}$ along the spline and $2.3 \times 10^{-5}$ in the high-energy expansion. At the nodes where the code uses an expansion, the expansion agrees with the table to $4 \times 10^{-5}$ for $k = 4.02$ and $2 \times 10^{-4}$ for $k \ge 10^3$. Against [Haug](https://doi.org/10.1515/zna-1975-0901)'s Table 1, given to three or four digits, the code's $\sigma(k)$ is within $8 \times 10^{-4}$ at all 58 energies. Fourteen of these energies are nodes, where the Monte Carlo is within $7 \times 10^{-4}$ of Haug's values. The script also runs the Monte Carlo at five energies between the nodes, $k = 4.03, 7, 10, 12$ and 45. There it is within $5.2 \times 10^{-4}$ of Haug's values.

## Energies of the products

For $\gamma \gg 1$ and $\varepsilon \ll \gamma$, each product carries the fraction

$$
x = \frac{p\cdot k}{p_1\cdot k}
$$

of the lepton's lab energy, with $p$, $p_1$ and $k$ the four-momenta of the product, the lepton and the photon. This fraction does not depend on the photon's direction in the lab. In the lepton's rest frame, with the photon along the $z$ axis, we have $x = E - p_z$. Here $E$ is the product's energy and $p_z$ its momentum along the photon. The fractions of the three products add up to 1. They are $x_+$ for the positron, and $x_\mathrm{low}$ and $x_\mathrm{high}$ for the electrons of lower and higher lab energy. The two electrons are identical, so the one of higher energy counts as the incident lepton after the collision: the scattered lepton. This lepton loses the fraction $\xi = 1 - x_\mathrm{high} = x_+ + x_\mathrm{low}$ of its energy.

Boosting a product from the lepton's rest frame back to the lab gives its exact lab energy:

$$
E_\mathrm{lab} = \gamma x + \frac{\varepsilon}{k}\thinspace p_z + \frac{\beta\sin\theta}{1 - \beta\cos\theta}\thinspace p_\perp\cos\phi.
$$

Here $p_\perp$ is the product's momentum across the photon in the lepton's rest frame. Its azimuth $\phi$ about the photon is counted from the half-plane that holds the lepton's direction of motion. The second term is at most $\varepsilon$, as $\vert p_z\vert \lt k$. The third is at most $p_\perp\sqrt{2\gamma\varepsilon/k}$, as $\sin^2\theta \le 2(1 - \beta\cos\theta)$. Relative to the lepton's energy $\gamma$, both terms vanish for $\gamma \gg 1$ and $\varepsilon \ll \gamma$. The code neglects them and takes the lab energy $\gamma x$ for photons up to $\varepsilon = 1$. For products of small $x$ on photons near $m_e c^2$, the two terms still matter.

The script [x_scaling.py](x_scaling.py) boosts the Monte Carlo's QED events to the lab, for leptons of $\gamma = 10^4$ and $10^6$ in isotropic photons. It sorts the products' lab energies into bins as wide as those of the code's grid. The change of a spectrum is half the summed absolute change of its bins, over their total. For $\varepsilon \le 0.1$, the code's lab energies change the positron spectrum by 0.1–0.9%. This is about the noise of the comparison: two exact spectra that differ only in their random azimuths differ by 0.1–0.8%. At $\varepsilon = 0.3$ and 1, the code's lab energies change the positron spectrum by 2.5–2.8% and 19–23%. They change its mean energy by 0.1–0.4% and 1–3%. The spectrum of the scattered lepton changes by less than 0.03%.

For every node $k$, the Monte Carlo's tables hold the distribution $f_r(k, y) = dN/d\ln y$ of each product $r$. Here $d{N}$ is the number of such products per interaction in the interval $d\ln y$, and the distribution is normalised to one. This subscript $r$ labels the product, as in $x_+$ and $x_\mathrm{low}$, and is not the $r$ of the threshold expansion. For the positron and the electron of lower energy, $y = xk = p\cdot k$, that is $y = x_+ k$ and $y = x_\mathrm{low} k$. For the scattered lepton, $y = \xi k$ is defined through the energy it loses. The tables hold the products of an electron target, that is, of an incident electron. A positron target gives their charge conjugates.

The table `TPP_DIST` in [triplet_tables.mojo](../../../src/am3_mojo/physics/triplet_tables.mojo) holds these distributions in bins of $\ln y$. The functions `tpp_nbins` and `TripletPairProduction.distributions` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) give them at any $k$, interpolated between the nodes. Above the last node, $k = 10^{10}$, the code extrapolates them with two laws. The first law, at fixed $y$, has two functions of $y$, $g_0$ and $g_1$. The second, at fixed $x$, has two functions of $x$, $A$ and $B$. The method `TripletPairProduction._fit_asymptotics` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) fits these four functions to the nodes from $k = 10^6$ on. Each law applies in the range given in brackets:

$$
f_r = g_0(y) + \frac{g_1(y)}{\ln k}\quad (y \le 100),\qquad k f_r = A(x)\ln k + B(x)\quad (x \ge 10^{-5}).
$$

Between the two ranges, the code uses $f_r = (a_0\ln y + b_0)/y$, with $a_0 = 10^{-5}A(10^{-5})$ and $b_0 = 10^{-5}B(10^{-5}) - a_0\ln 10^{-5}$. With these two constants, the middle law joins the second law at $x = 10^{-5}$.

The tables stop at $k = 10^{10}$, because above it the gauge cancellations between the diagrams approach the limit of double precision. The script [machinery.py](machinery.py) measures these cancellations through the Ward identity. This identity requires the squared matrix element $\vert\mathcal M\vert^2$ to vanish when the photon's polarisation is replaced by its momentum. The identity holds to $10^{-11}$ of $\vert\mathcal M\vert^2$ at $k = 10^{10}$, $2 \times 10^{-8}$ at $10^{12}$ and $10^{-4}$ at $10^{14}$. At $10^{14}$, the mean energies of the products come out up to $2 \times 10^6$ times too large.

The script [extrapolation.py](extrapolation.py) makes the code fit its laws to fewer nodes and compares the extrapolation with the nodes left out. One decade beyond the fit, the laws reproduce the distributions within 1%, in the summed absolute difference of the normalised distributions. Weighted with energy, the difference is 1–3%, and the mean energies agree within 0.1–2%. Three decades beyond, the mean energies agree within 2.5–6.4%, the least well for the scattered lepton. Above $k = 10^{10}$, nothing can test the code's own fit, as the Monte Carlo fails there.

## Rates

A lepton interacts with photons of density $n$ coming from the angle $\theta$ at the rate $c\thinspace (1 - \beta\cos\theta)\sigma(k)\thinspace n$. In an isotropic photon field, with $\mu = \cos\theta$ and $\beta \to 1$, the mean of this rate over the photons' directions is $c\sigma_T n$ times

$$
R(P) = \frac{1}{2}\int_{-1}^{1} d\mu\thinspace (1 - \mu)\thinspace \frac{\sigma\bigl(P\thinspace (1 - \mu)\bigr)}{\sigma_T},\qquad P = \gamma\varepsilon,
$$

With $k = P(1 - \mu)$, and $\sigma = 0$ below $k = 4$, the interaction rate of a lepton and the spectra $G_r$ of its products depend only on $P = \gamma\varepsilon$:

$$
R(P) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k\thinspace \frac{\sigma(k)}{\sigma_T},\qquad G_r(P, x) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k\thinspace \frac{\sigma(k)}{\sigma_T}\thinspace f_r(k, xk).
$$

The spectrum $G_r$ counts the products per unit $\ln x$, as $d\ln y = d\ln x$ at fixed $k$. A lepton therefore interacts at the rate $c\sigma_T \int d\ln\varepsilon\ n_\gamma(\varepsilon)$ $R(\gamma\varepsilon)$, with $n_\gamma(\varepsilon)$ the photon density per $\ln\varepsilon$.

## Kinetic terms

The process adds three terms to AM³'s kinetic equations. They are a source of pairs, a continuous energy loss of the leptons and a sink of photons. The positron and electron densities per $\ln\gamma$ are $n_\pm(\gamma)$. A dot marks the derivative in time.

The pairs are sources, one positron and one electron per interaction. The labels of the fractions above are those of an $\mathrm{e}^-$ target. An $\mathrm{e}^+$ target gives an $\mathrm{e}^-$ with $x_+$ and an $\mathrm{e}^+$ with $x_\mathrm{low}$. A product of Lorentz factor $\gamma$ made by a target of Lorentz factor $\gamma'$ has $x = \gamma/\gamma'$. The positron source is then

$$
\dot n_{+}(\gamma) = c\thinspace \sigma_T\int d\ln\gamma'\int d\ln\varepsilon\thinspace n_\gamma(\varepsilon)\left[n_-(\gamma')\thinspace G_{+}\negthinspace \left(\gamma'\varepsilon,\ \frac{\gamma}{\gamma'}\right) + n_+(\gamma')\thinspace G_{\rm low}\negthinspace \left(\gamma'\varepsilon,\ \frac{\gamma}{\gamma'}\right)\right],
$$

and likewise for the electrons, with $n_{-}$ and $n_+$ exchanged ($n_- G_\mathrm{low} + n_+ G_+$).

The scattered lepton loses the energy that the pair takes. The code treats this energy loss as continuous:

$$
-\frac{\dot\gamma}{\gamma} = c\thinspace \sigma_T\int d\ln\varepsilon\thinspace n_\gamma(\varepsilon)\thinspace R(\gamma\varepsilon)\thinspace \langle x_+ + x_{\rm low}\rangle(\gamma\varepsilon),
$$

with $\langle\ldots\rangle$ the mean over $k$ with the weight $k\sigma(k)$, as in $R$. This energy balance leaves out the photon's energy, which is small: $\varepsilon \le 1$ against the lepton's $\gamma$. At fixed $k$, the mean $\langle x_+ + x_\mathrm{low}\rangle$ comes from the Monte Carlo's tables. Table 2 of [Anguelov et al. (1999)](https://arxiv.org/abs/astro-ph/0001252) tabulates the mean energy $E_m$ of a produced particle as $E_m s/E_0$. Here $E_0$ is the lepton's energy, and their $s$ is our $k$, not our $s$. Twice their value, for the two particles of the pair, corresponds to the tables' $k\langle x_+ + x_\mathrm{low}\rangle$. At $k = 10^2$, $10^3$ and $10^4$, the two agree within 1% ([inelasticity.py](inelasticity.py)).

A continuous loss fits where the process dominates the lepton's losses, because the lepton then loses little per interaction. The script [inelasticity.py](inelasticity.py) compares the loss with the inverse-Compton loss in its Klein–Nishina limit, $-\dot\gamma$ $= (3/8) c\sigma_T n\varepsilon^{-1}$ $[\ln(4\gamma\varepsilon) - 11/6]$, for $\gamma\varepsilon \gg 1$. Here $n$ is the density of photons of energy $\varepsilon$. The limit comes from eq. 2.57 of [Blumenthal & Gould (1970)](https://doi.org/10.1103/RevModPhys.42.237). This equation gives the energy-loss rate $-dE/d{t}$ $= \pi r_0^2 m^2c^5 \int d\varepsilon\ n(\varepsilon)$ $\varepsilon^{-1}$ $[\ln(4\varepsilon\gamma/mc^2) - 11/6]$, with $r_0$ the classical electron radius $r_e$ and $m$ the electron mass. Their $n(\varepsilon)$ is the photon density per unit $\varepsilon$, and $t$ is the time. For photons of one energy, with the electron's energy $E = \gamma mc^2$ and $\varepsilon$ in units of $mc^2$, eq. 2.57 gives the form above, as $\pi r_e^2 = (3/8)\sigma_T$. The triplet loss exceeds the inverse-Compton loss above $\gamma\varepsilon = 2.6 \times 10^8$. In that range, the lepton loses on average at most $5.6 \times 10^{-7}$ of its energy per interaction. Where it loses more, as with $\langle\xi\rangle = 0.018$ at $\gamma\varepsilon = 10^3$ and 0.085 at $10^2$, the inverse-Compton loss dominates. AM³ treats that loss as continuous too.

The continuous loss neglects the spread of $\xi$ between interactions, which [inelasticity.py](inelasticity.py) measures in the tables. A lepton that loses more than a grid bin keeps less than $e^{-\Delta}$ of its energy, so $\xi \gt 1 - e^{-\Delta}$. Here $\Delta$ is the logarithmic step of AM³'s energy grids ([On the grid](#on-the-grid)). Interactions where the lepton loses more than a grid bin ($\xi \gt 0.095$) carry 41% of the loss at $k = 10^3$, 24% at $10^5$ and 20% at $10^6$. Interactions where it loses more than half of its energy carry 3–6% of the loss.

Each interaction removes its photon, so the photons have the sink

$$
\dot n_\gamma(\varepsilon) = -\thinspace n_\gamma(\varepsilon)\thinspace c\thinspace \sigma_T\int d\ln\gamma\thinspace \bigl(n_-(\gamma) + n_+(\gamma)\bigr)\thinspace R(\gamma\varepsilon).
$$

## On the grid

On the grid, the code evaluates the kinetic terms in two stages. It builds kernel rows once and sums them over the bins in each time step. AM³'s grids have the step $\Delta = 0.1$ in $\ln E$, the natural logarithm of the energy. The lepton bins, numbered by $i$ from 0, are $\gamma_i = e^{i\Delta}$. The photon bins, numbered by $j$, are $\varepsilon_j = e^{(j - j_0)\Delta}$, with $\varepsilon_{j_0} = 1$. A lepton of bin $i$ and a photon of bin $j$ have $P = \gamma_i\varepsilon_j$ on the grid

$$
P_q = e^{(m_0 + q)\Delta},\qquad m_0 = \left\lfloor \frac{\ln 2}{\Delta}\right\rfloor + 1,\qquad \gamma_i\varepsilon_j = P_q\quad \text{for}\quad q = i + j - j_0 - m_0.
$$

Here $\lfloor\thinspace\rfloor$ rounds down to an integer, so $P_0$ is the first grid value above the threshold $P = 2$, where head-on photons have $k = 2P = 4$.

**Kernel rows.** The method `TripletPairProduction.initialize_kernels` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) computes the rate and the products of a lepton in a unit photon density. At the grid value $P_q$, the rate is $R(P_q)$, written as an integral over $\ln k$:

$$
R_q = \frac{1}{2P_q^2}\int_{\ln 4}^{\ln 2P_q} d\ln k\thinspace k^2\thinspace \frac{\sigma(k)}{\sigma_T}.
$$

At each $k$ of this integral, the products of the table bin $y$ have $x = y/k$. The method `TripletPairProduction._deposit` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) shares them between the grid fractions $x_o = e^{-o\Delta}$ and $x_{o+1}$ next to $x$, with $o = \lfloor -(\ln x)/\Delta\rfloor$. A product at $x_o$ lands $o$ bins below its target, so $o$ is the product's offset. The method puts the share $w$ at $x_{o+1}$ and the rest at $x_o$:

$$
w = \frac{x_o - x}{x_o - x_{o+1}}.
$$

This keeps their number and energy: $w + (1 - w) = 1$ and $(1 - w)\thinspace x_o + w\thinspace x_{o+1} = x$. Integrated over $k$ and summed over the table bins, the shares make the positron's rows $W_{+,q,o}$ and the lower-energy electron's rows $W_{\mathrm{low},q,o}$. In the continuum limit, $W_{r,q,o}$ is $\Delta\thinspace G_r(P_q, x_o)$, the products in an offset bin of width $\Delta$ in $\ln x$. The energy that the pair takes is that of its deposits:

$$
L_q = \sum_o \bigl(W_{+,q,o} + W_{\mathrm{low},q,o}\bigr)\thinspace x_o = R_q\thinspace \langle x_+ + x_{\rm low}\rangle(P_q).
$$

**Each time step.** The integrals over $\ln\gamma'$ and $\ln\varepsilon$ become sums over the bins times $\Delta$. The kernel rows take the place of the continuous rates. The functions `TripletPairProduction.Calc_Triplet`, `triplet_rows_work` and `TripletPairProduction.photon_sink` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) evaluate these sums. The first sum gives $Z_{r,i,o}$, the rate at which a lepton of bin $i$ deposits the product $r$ at the offset $o$. In all three sums, $q$ is the row of the lepton bin $i$ and the photon bin $j$:

$$
Z_{r,i,o} = c\thinspace \sigma_T\Delta\sum_j n_\gamma(\varepsilon_j)\thinspace W_{r,q,o},\qquad -\frac{\dot\gamma_i}{\gamma_i} = c\thinspace \sigma_T\Delta\sum_j n_\gamma(\varepsilon_j)\thinspace L_q,\qquad \dot n_\gamma(\varepsilon_j) = -\thinspace n_\gamma(\varepsilon_j)\thinspace c\thinspace \sigma_T\Delta\sum_i \bigl(n_-(\gamma_i) + n_+(\gamma_i)\bigr)\thinspace R_q.
$$

The pair sources follow from $Z_{r,i,o}$: a product at the offset $o = i - l$ below its target's bin $i$ lands in bin $l$. The method `TripletPairProduction.add_pair_sources` in [triplet.mojo](../../../src/am3_mojo/physics/triplet.mojo) adds them up:

$$
\dot n_+(\gamma_l) = \sum_i \bigl[n_-(\gamma_i)\thinspace Z_{+,i,i-l} + n_+(\gamma_i)\thinspace Z_{\mathrm{low},i,i-l}\bigr],
$$

and the electrons' source exchanges $n_{-}$ and $n_+$.

The lab energy $\gamma x$ neglects the photon's energy. On photons above about $0.25\ m_e c^2$, this puts some products below the grid, into bin 0. Of the positrons made on photons of 0.3, 0.5 and $1\ m_e c^2$, 1%, 9–10% and 30–34% land in bins 0 and 1. The products put there carry up to 0.1%, 1.4% and 6% of the pairs' energy. AM³'s solver empties these two bins every step. Yet the photons of these products are absorbed, and the lepton loses their energy. In the scenario `leptohadronic_all`, at the end of its run, 69% of the triplet interactions are on photons above $0.25\ m_e c^2$. This way, 15% of its triplet positrons are lost ([below_grid.py](below_grid.py)).

AM³'s code units make $c\sigma_T = 1$, with lengths in $\sqrt{\sigma_T}$ and times in $\sqrt{\sigma_T}/c$, the unit `T_SSC` in [constants.mojo](../../../src/am3_mojo/utils/constants.mojo). In each step, the method `SimulationManager.Calc_Coeff_Lep` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) computes the terms. The methods `SimulationManager.Pass_Coeff_Lep`, `SimulationManager._triplet_sources` and `SimulationManager.Pass_Coeff_Phot` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) add the terms to the kinetic equations.
