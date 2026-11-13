# Electron–positron annihilation

Electrons and positrons annihilate into two photons, $\mathrm e^+\mathrm e^- \to \gamma\gamma$. This page starts from the cross section, rate and photon spectrum of [Svensson (1982)](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S). These results are exact for isotropic electrons and positrons of any Lorentz factors. From them, the page derives the equations and tables that the Mojo code uses. It also names the corresponding struct and function. The Python module [annihilation.py](annihilation.py) checks every step: `.venv/bin/python tests/literature_checks/common/run.py annihilation`.

## Notation

This page uses [Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s symbols and quotes his equations with his journal page and equation numbers. Energies are in units of $mc^2$, with $m$ the electron mass and $c$ the speed of light. A photon has the energy $k$. In the laboratory frame, the positron and the electron have the Lorentz factors $\gamma_\pm$ and the velocities $\beta_\pm$, in units of $c$. The two particles have the four-velocities $v_\pm$, and $\mu$ is the cosine of the angle between their velocities. In the center-of-momentum (cm) frame of the pair, each has the Lorentz factor $\gamma_{cm}$ and the velocity $\beta_{cm}$. In the rest frame of the electron, the positron has the Lorentz factor $\gamma_r$ and the velocity $\beta_r$. The positrons and electrons have the densities $n_\pm$ and the distributions $f_\pm(\gamma)$ in Lorentz factor, normalized to 1. A dot over a symbol marks a rate per unit time. An overline marks an average over the directions of isotropic particles, and angle brackets mark an average over the distributions. The classical electron radius is $r_e$, and $\alpha$ is the fine-structure constant. The code's rates are in units of $\sigma_T c$, with the Thomson cross section $\sigma_T = 8\pi r_e^2/3$.

## Cross section

[Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s results are the Born approximation of quantum electrodynamics. They hold for particle velocities well above $\alpha c$ (his p. 322). He leaves out Coulomb corrections, the formation of positronium and annihilation into three photons.

The cross section is simplest in the cm frame. [Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s eq. 6 (p. 323) prints Dirac's cross section as

$$
\sigma(\gamma_{cm}) = r_e^2\thinspace\frac{\pi}{4\beta_{cm}\gamma_{cm}^2}\left[\frac{1}{\beta_{cm}}\left(2 + \frac{2}{\gamma_{cm}^2} - \frac{1}{\gamma_{cm}^4}\right)\ln\negthinspace\left(\frac{1 + \beta_{cm}}{1 + \beta_{cm}}\right) - 2 - \frac{2}{\gamma_{cm}^2}\right].
$$

The argument of the logarithm in this equation is a misprint. As printed, the logarithm vanishes and the cross section is negative. The argument is $(1 + \beta_{cm})/(1 - \beta_{cm})$. Its logarithm is [Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s $L(\beta_{cm})$ of eq. 18. With this argument, his eq. 18 is the integral of eq. 16 (see [Rate](#rate)).

[Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s eq. 1a (p. 322) relates the laboratory frame, the cm frame and the electron's rest frame through the product of the four-velocities:

$$
v_+\cdot v_- = \gamma_+\gamma_-(1 - \beta_+\beta_-\mu) = 2\gamma_{cm}^2 - 1 = \gamma_r.
$$

So $\gamma_r$ alone fixes $\gamma_{cm}$ and $\beta_{cm}$. The code writes the cross section and the rate in $\gamma_r$. It takes $\gamma_{cm}$ and $\beta_{cm}$ from eq. 1a ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S), p. 328) and computes $L(\beta_{cm})$ from them:

$$
\gamma_{cm} = \left[\frac{\gamma_r + 1}{2}\right]^{1/2},\qquad \beta_{cm} = \left[\frac{\gamma_r - 1}{\gamma_r + 1}\right]^{1/2},\qquad L(\beta_{cm}) = \ln[\gamma_r(1 + \beta_r)].
$$

## Rate

[Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s eq. 13 (p. 324) defines the angle-averaged reaction rate per pair of particles:

$$
\overline{v\sigma}(\gamma_+,\gamma_-) \equiv \frac{1}{\gamma_+\gamma_-}\int_{-1}^{1}\frac{d\mu}{2}\thinspace 2\beta_{cm}\gamma_{cm}^2\thinspace c\thinspace\sigma(\gamma_{cm}).
$$

This rate is the cross section times the relative velocity, averaged over the directions of isotropic particles. In its integrand, $\gamma_{cm}$ and $\beta_{cm}$ depend on $\mu$ through eq. 1a, and the factor $2\beta_{cm}\gamma_{cm}^2$ is $\beta_r\gamma_r$. Divided by $\gamma_+\gamma_-$, the integrand is the rate of his eq. 8 (p. 323) per pair of particles. Eq. 8 holds for two beams that cross at the angle of cosine $\mu$. His eq. 16 (p. 324) takes $\gamma_{cm}$ as the integration variable instead:

$$
\overline{v\sigma}(\gamma_+,\gamma_-) = \frac{1}{\beta_+\gamma_+^2\beta_-\gamma_-^2}\int_{\gamma_{cm}^{\min}}^{\gamma_{cm}^{\max}}d\gamma_{cm}\thinspace 4\beta_{cm}\gamma_{cm}^3\thinspace c\thinspace\sigma(\gamma_{cm}).
$$

Its limits, his eq. 17, are the values of $\gamma_{cm}$ at $\mu = -1$ and $\mu = 1$, by eq. 1a:

$$
\gamma_{cm}^{\max} = \left[\tfrac{1}{2}(1 + \gamma_+\gamma_- + \gamma_+\gamma_-\beta_+\beta_-)\right]^{1/2},\qquad \gamma_{cm}^{\min} = \left[\tfrac{1}{2}(1 + \gamma_+\gamma_- - \gamma_+\gamma_-\beta_+\beta_-)\right]^{1/2}.
$$

His result is eq. 18 (p. 324). The bracket stands for its value at the upper limit minus its value at the lower limit:

$$
\overline{v\sigma}(\gamma_+,\gamma_-) = cr_e^2\thinspace\frac{\pi}{\beta_+\gamma_+^2\beta_-\gamma_-^2}\Big[\beta_{cm}^3\gamma_{cm}^2L(\beta_{cm}) - 2\gamma_{cm}^2 + \tfrac{3}{4}L^2(\beta_{cm})\Big]_ {\gamma_{cm}^{\min}}^{\gamma_{cm}^{\max}},\qquad L(\beta) = \ln\frac{1 + \beta}{1 - \beta}.
$$

With the corrected eq. 6, the derivative of the bracket in $\gamma_{cm}$ is $4\beta_{cm}\gamma_{cm}^3\sigma(\gamma_{cm})/(\pi r_e^2)$, so eq. 18 is the integral of eq. 16. The rate is symmetric in $\gamma_+$ and $\gamma_{-}$. For slow particles, the rate tends to $\pi r_e^2 c$ (his eq. 19a).

### The rate in the code

The function `ann_rate` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) computes $\overline{v\sigma}/(\sigma_T c)$ from eq. 18 or eq. 13, both written in $\gamma_r$. The range of $\gamma_r$ runs between the values of eq. 1a at $\mu = 1$ and $\mu = -1$:

$$
\gamma_r^{\max} = \gamma_+\gamma_-(1 + \beta_+\beta_-),\qquad \gamma_r^{\min} = \gamma_+\gamma_-(1 - \beta_+\beta_-) = \frac{\gamma_+^2 + \gamma_-^2 - 1}{\gamma_r^{\max}}.
$$

The code takes the last form of $\gamma_r^{\min}$. This form follows from $\gamma_r^{\max}\gamma_r^{\min} = \gamma_+^2 + \gamma_-^2 - 1$. The function `gcm_limits` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) computes the limits of eq. 17 from these two values, with eq. 1a. The functions `_eq18_bracket` and `L_cm` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) evaluate the bracket of eq. 18 and $L(\beta_{cm})$ with the relations of [Cross section](#cross-section).

Where $\beta_+\gamma_+\beta_-\gamma_-$ is below a hundredth of $\gamma_r^{\max}$, the brackets at the two limits nearly cancel. There the code takes the integral of eq. 13 with the 8-node Gauss–Legendre rule in $\mu$, whose error is below $10^{-30}$ there. At its nodes, $\gamma_r$ $= [\gamma_r^{\max}(1 - \mu) + \gamma_r^{\min}(1 + \mu)]/2$ by eq. 1a. The function `_eq13_integrand` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) evaluates the integrand in $\gamma_r$:

$$
2\beta_{cm}\gamma_{cm}^2\thinspace\sigma(\gamma_{cm}) = \frac{\pi r_e^2}{1 + \gamma_r}\left[(\gamma_r^2 + 4\gamma_r + 1)\thinspace\frac{L(\beta_{cm})}{\beta_r\gamma_r} - (\gamma_r + 3)\right].
$$

For $\gamma_r$ within $10^{-12}$ of 1, the code takes the first two terms of the series of the ratio in the bracket,

$$
\frac{L(\beta_{cm})}{\beta_r\gamma_r} = 1 - \frac{\gamma_r - 1}{3} + \frac{2\thinspace (\gamma_r - 1)^2}{15} - \ldots,
$$

and the first term it leaves out is below $10^{-24}$.

## Sinks and photon source

[Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s eqs. 14–15 (p. 324) give $\dot n_+$, the number of annihilations per unit volume and time:

$$
\dot n_+ = n_+n_-\langle\overline{v\sigma}\rangle,\qquad \langle\overline{v\sigma}\rangle \equiv \int_1^\infty d\gamma_+\thinspace f_+(\gamma_+)\int_1^\infty d\gamma_-\thinspace f_-(\gamma_-)\thinspace\overline{v\sigma}(\gamma_+,\gamma_-).
$$

Each annihilation removes one positron and one electron. Per unit Lorentz factor, the positrons of $\gamma_+$ and the electrons of $\gamma_-$ therefore annihilate at the rates

$$
\frac{d\dot n_+}{d\gamma_+} = n_+n_-f_+(\gamma_+)\int_1^\infty d\gamma_-\thinspace f_-(\gamma_-)\thinspace\overline{v\sigma}(\gamma_+,\gamma_-),\qquad \frac{d\dot n_+}{d\gamma_-} = n_+n_-f_-(\gamma_-)\int_1^\infty d\gamma_+\thinspace f_+(\gamma_+)\thinspace\overline{v\sigma}(\gamma_+,\gamma_-).
$$

These two rates are the integrands of the outer integral of eq. 14, taken in either order. Integrated over $\gamma_+$ or over $\gamma_-$, both give $\dot n_+$.

In the same way, [Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s eqs. 23–25 (p. 325) give $\dot n(k)$, the number of photons per unit volume, time and photon energy. They average his angle-averaged emissivity per pair over the distributions. This emissivity, $\overline{v\thinspace d\sigma/dk}$ of eq. 24, is the relative velocity times the cross section per unit photon energy, averaged over directions:

$$
\dot n(k) = n_+n_-\Big\langle\overline{v\frac{d\sigma}{dk}}(k)\Big\rangle,\qquad \Big\langle\overline{v\frac{d\sigma}{dk}}(k)\Big\rangle \equiv \int_{\gamma_+^{(L)}}^{\gamma_+^{(U)}}d\gamma_+\thinspace f_+(\gamma_+)\int_{\gamma_-^{(L)}}^{\gamma_-^{(U)}}d\gamma_-\thinspace f_-(\gamma_-)\thinspace\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-).
$$

The limits $\gamma_\pm^{(L)}$ and $\gamma_\pm^{(U)}$ of his eq. 37 (p. 327) bound the Lorentz factors of the pairs that make photons of energy $k$. Outside these limits, the emissivity vanishes, so the integrals can run over all Lorentz factors. The emissivity counts both photons of an annihilation:

$$
\int\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)\thinspace dk = 2\thinspace\overline{v\sigma}(\gamma_+,\gamma_-),\qquad \int k\thinspace\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-)\thinspace dk = (\gamma_+ + \gamma_-)\thinspace\overline{v\sigma}(\gamma_+,\gamma_-).
$$

Each annihilation thus makes two photons with the energy of the pair, as in [Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s cooling rate (eq. 21). He integrates his spectrum over $k$ and divides by 2 to recover eq. 18 (p. 331). On the code's grids, the annihilations and the photons become the sums of [Sinks on the grid](#sinks-on-the-grid) and [Photon source on the grid](#photon-source-on-the-grid).

## Photon spectrum

### Kinematics

In the cm frame, both photons have the energy $k_{cm} = \gamma_{cm}$. The cm frame moves through the laboratory frame with the Lorentz factor $\gamma_c = (\gamma_+ + \gamma_-)/(2\gamma_{cm})$ (eq. 2a, p. 322) and the velocity $\beta_c$. In the laboratory frame, a photon therefore has an energy between $k_{\min}$ and $k_{\max}$, which are $\gamma_{cm}\gamma_c(1 \mp \beta_c)$ (eq. 29, p. 326). By eq. 30,

$$
k_{\max}k_{\min} = \gamma_{cm}^2,\qquad k_{\max} + k_{\min} = \gamma_+ + \gamma_-.
$$

If a photon's energy $k$ is $k_{\min}$ or $k_{\max}$, elimination of the other energy gives the critical Lorentz factor $\gamma_{cm}^\ast$ (eq. 31, p. 326). Pairs with a larger $\gamma_{cm}$ cannot make the photon:

$$
\gamma_{cm}^{\ast 2} = k\thinspace(\gamma_+ + \gamma_- - k).
$$

The integrals over $\gamma_{cm}$ below therefore run between the lower and upper limits of eq. 32 (p. 326):

$$
\gamma_{cm}^{(U)} = \min\left[\gamma_{cm}^{\max},\ \gamma_{cm}^\ast\right],\qquad \gamma_{cm}^{(L)} = \gamma_{cm}^{\min}.
$$

All photons of the pair lie between the kinematic limits $k_L = \tfrac12[\gamma_+(1-\beta_+) + \gamma_-(1-\beta_-)]$ and $k_U = \tfrac12[\gamma_+(1+\beta_+) + \gamma_-(1+\beta_-)]$ (eqs. 33d and 33a). At these two energies, $\gamma_{cm}^\ast$ equals $\gamma_{cm}^{\min}$. The spectrum peaks where $\gamma_{cm}^\ast$ equals $\gamma_{cm}^{\max}$ (eqs. 33b–c), at

$$
k_\pm = \tfrac{1}{2}\left[\gamma_\pm(1+\beta_\pm) + \gamma_\mp(1-\beta_\mp)\right].
$$

The spectrum is symmetric about $k_T = (\gamma_+ + \gamma_-)/2$ (his pp. 326 and 331). The function `ann_k_limits` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) computes $k_L$, $k_\pm$ and $k_U$ with $\gamma(1 - \beta) = 1/[\gamma(1 + \beta)]$, which avoids the cancellation in $1 - \beta$ for fast particles.

### Spectrum of a pair

[Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s eq. 26 (p. 325) writes the emissivity of a pair as an integral over $\gamma_{cm}$, as eq. 16 does for the rate:

$$
\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-) = \frac{1}{\beta_+\gamma_+^2\beta_-\gamma_-^2}\int_{\gamma_{cm}^{(L)}}^{\gamma_{cm}^{(U)}}d\gamma_{cm}\thinspace 4\beta_{cm}\gamma_{cm}^3\thinspace c\thinspace\frac{d\sigma}{dk}(k,\gamma_+,\gamma_-,\gamma_{cm}).
$$

In its integrand, $d\sigma/dk$ is the cross section per unit photon energy at the given $\gamma_{cm}$. Integrated over the photon's directions in the cm frame (eqs. 46–49), this cross section is that of eq. 50 (p. 330):

$$
\frac{d\sigma}{dk} = r_e^2\thinspace\frac{1}{4\beta_{cm}\gamma_{cm}^3}\thinspace\pi\left\lbrace\frac{-4\gamma_{cm}}{[(\gamma_+ + \gamma_-)^2 - 4\gamma_{cm}^2]^{1/2}} + G_+ + G_-\right\rbrace,
$$

with the $G_\pm$ of eqs. 51–54:

$$
G_\pm = \left(2\gamma_{cm}^2 + 2 - \frac{1}{\gamma_{cm}^2}\right)\frac{1}{u_\pm} + \tfrac{1}{2}\left(2\gamma_{cm}^2 - d_\pm\right)\frac{1}{u_\pm^3},
$$

$$
d_\pm = \gamma_\mp(\gamma_+ + \gamma_-) \pm k(\gamma_+ - \gamma_-),\qquad u_\pm = (c_\pm\gamma_{cm}^2 + \gamma_{cm}^{\ast 2})^{1/2},\qquad c_\pm = (\gamma_\mp - k)^2 - 1.
$$

The integral over $\gamma_{cm}$ is elementary. Eqs. 55–57 (p. 330) give it. Like the bracket of eq. 18, the brace is taken between the limits:

$$
\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-) = cr_e^2\thinspace\frac{\pi}{\beta_+\gamma_+^2\beta_-\gamma_-^2}\Big\lbrace[(\gamma_+ + \gamma_-)^2 - 4\gamma_{cm}^2]^{1/2} + H_+ + H_-\Big\rbrace_ {\gamma_{cm}^{(L)}}^{\gamma_{cm}^{(U)}},
$$

$$
H_\pm = \left(2 + \frac{1 - \gamma_{cm}^{\ast 2}}{c_\pm}\right)I_\pm + \left[\frac{1}{\gamma_{cm}} - \frac{\gamma_{cm}}{c_\pm} + \frac{\gamma_{cm}}{2\gamma_{cm}^{\ast 2}}(2c_\pm - d_\pm)\right]\frac{1}{u_\pm} + \frac{\gamma_{cm}}{c_\pm}u_\pm,
$$

$$
I_\pm = \begin{cases}\dfrac{1}{(c_\pm)^{1/2}}\ln[\gamma_{cm}(c_\pm)^{1/2} + u_\pm] & c_\pm \gt 0\cr \dfrac{1}{(-c_\pm)^{1/2}}\sin^{-1}\negthinspace\left[\dfrac{\gamma_{cm}}{\gamma_{cm}^\ast}(-c_\pm)^{1/2}\right] & c_\pm \lt 0\end{cases}
$$

Here $\sin^{-1}$ is the arcsine. Where $c_\pm$ vanishes, eq. 58 (p. 331) replaces eq. 56:

$$
H_\pm = \left(\tfrac{2}{3}\gamma_{cm}^3 + 2\gamma_{cm} + \frac{1}{\gamma_{cm}}\right)\frac{1}{\gamma_{cm}^\ast} + \tfrac{1}{2}\left(\tfrac{2}{3}\gamma_{cm}^3 - d_\pm\gamma_{cm}\right)\frac{1}{\gamma_{cm}^{\ast 3}}.
$$

Eq. 55 is the integral of eq. 26 with eq. 50. The derivative of its brace in $\gamma_{cm}$ is the brace of eq. 50, because $dH_\pm/d\gamma_{cm} = G_\pm$. Eq. 58 is the limit of eq. 56 for vanishing $c_\pm$, up to a constant that cancels between the limits. The integrals of eq. 55 over $k$ give the photon number and energy above, to $3.9 \times 10^{-31}$ in a 30-digit quadrature. Eq. 55 passes the tests of [Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s p. 331. Like the rate, it is symmetric in $\gamma_+$ and $\gamma_{-}$, and it tends to eq. 41 as $\gamma_-$ tends to 1.

### The spectrum in the code

The functions `_emissivity`, `_eq55_brace` and `_H` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) evaluate eq. 55 between the limits of eq. 32, its brace and $H_\pm$ of eqs. 56–58. On the range of the integral, $c_\pm$ is at least $-1$ and $\gamma_{cm}$ is at most $\gamma_{cm}^\ast$. So $u_\pm$ is real, and the argument of the arcsine in eq. 57 is at most 1. The square root $u_\pm$ vanishes only where $\gamma_{cm} = \gamma_{cm}^\ast$ and $k = \gamma_\mp$. There the second term of $H_\pm$ tends to zero ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S), p. 331).

### One particle at rest

[Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S) separately derives the spectrum of a positron that annihilates with an electron at rest (eqs. 38–41, p. 328). His eq. 41 is

$$
\overline{v\frac{d\sigma}{dk}}(k,\gamma_r,1) = cr_e^2\thinspace\frac{\pi}{\beta_r\gamma_r^2}\left\lbrace\frac{-(3 + \gamma_r)/(1 + \gamma_r) + (3 + \gamma_r)/k - 1/k^2}{[1 - k/(1 + \gamma_r)]^2} - 2\right\rbrace,\qquad [\gamma_r(1 - \beta_r) + 1]/2 \le k \le [\gamma_r(1 + \beta_r) + 1]/2.
$$

Its range runs from $k_L$ to $k_U$ of eqs. 33 with $\gamma_- = 1$. The edges of the code's photon bins split this range into segments. Over each segment, `PairAnnihilation._fill_pair` and `ann_at_rest_segment` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) integrate eq. 41 in closed form. The photon number and energy have the primitives

$$
P(k) = cr_e^2\thinspace\frac{\pi}{\beta_r\gamma_r^2}\left[\frac{1}{k} + \left(3 + \gamma_r - \frac{2}{1 + \gamma_r}\right)\ln\frac{k}{1 + \gamma_r - k} - \frac{1}{1 + \gamma_r - k} - 2k\right],
$$

$$
E(k) = cr_e^2\thinspace\frac{\pi}{\beta_r\gamma_r^2}\left[-(3 + \gamma_r)(1 + \gamma_r)\ln(1 + \gamma_r - k) - \ln\frac{k}{1 + \gamma_r - k} - \frac{1 + \gamma_r}{1 + \gamma_r - k} - k^2\right].
$$

Their derivatives are eq. 41 and $k$ times eq. 41. Over the whole range, they give $2\thinspace\overline{v\sigma}(\gamma_r, 1)$ photons with the energy $(1 + \gamma_r)\thinspace\overline{v\sigma}(\gamma_r, 1)$. Here $\overline{v\sigma}(\gamma_r, 1)$ $= c\beta_r\sigma(\gamma_{cm})$ is the rate of a positron on an electron at rest ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S), p. 328). The photon number and energy of a segment, from $k_{\rm lo}$ to $k_{\rm hi}$, are the differences of the primitives between its ends. Their $\ln(1 + \gamma_r - k)$ terms become $\ln[(1 + \gamma_r - k_{\rm hi})/(1 + \gamma_r - k_{\rm lo})]$. At the top of the range, $1 + \gamma_r - k_U = k_L$ holds exactly.

### Both particles relativistic

[Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s eq. 59 (p. 331) approximates the spectrum of relativistic particles by one photon at each peak, with $\delta$ the Dirac delta function:

$$
\overline{v\frac{d\sigma}{dk}}(k,\gamma_+,\gamma_-) \approx \overline{v\sigma}(\gamma_+,\gamma_-)[\delta(k - k_+) + \delta(k - k_-)].
$$

Because $k_+ + k_- = \gamma_+ + \gamma_{-}$, the approximation keeps the energy of the photons as well as their number. For the pairs whose particles both have Lorentz factors above 100, the method `PairAnnihilation.initialize_kernels` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) sets up this approximation (see [Photon kernel](#photon-kernel)).

## On the code's grids

### Grids and units

The grids are those of AM³, the C++ code that am3_mojo ports to Mojo. The lepton grid points $\gamma_i$ have the spacing $\Delta$ in $\ln\gamma$, from $\gamma_0 = 1$, a particle at rest. The photon grid points $k_l$ have the same spacing $\Delta$ in $\ln k$. Photon bin $l$ holds the energies $k$ whose logarithm lies within $\Delta/2$ of $\ln k_l$. The code's arrays hold the lepton densities per unit $\ln\gamma$, $N_\pm(\gamma) = \gamma\thinspace n_\pm f_\pm(\gamma)$, and the photon densities per unit $\ln k$.

### Sinks on the grid

The methods `PairAnnihilation.initialize_kernels` and `PairAnnihilation.Calc_PairAnn` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) put the sinks on the grid. The first method tabulates the rate $R_{ij}$ for every pair of lepton grid points $i$ and $j$. The second method puts the particles of each bin at its grid point, with the density $N_\pm(\gamma_i)\Delta$. The annihilation rate of each electron and each positron then becomes a sum over the grid:

$$
\Gamma_\mp(\gamma_j) = \sigma_T c\thinspace\Delta\sum_i R_{ij}\thinspace N_\pm(\gamma_i),\qquad R_{ij} = \frac{\overline{v\sigma}(\gamma_i,\gamma_j)}{\sigma_T c}.
$$

Here $\Gamma_-$ is the rate of an electron, and $\Gamma_+$ is the rate of a positron. The particles of grid point $j$ thus decrease as $\dot N_\mp(\gamma_j)$ $= -\Gamma_\mp(\gamma_j)\thinspace N_\mp(\gamma_j)$, and the timescales $1/\Gamma_\mp$ are their annihilation times. The methods `SimulationManager.Pass_Coeff_Lep` and `SimulationManager.Calc_Lep` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) add $\Gamma_{-}$ and $\Gamma_+$ to the escape rates of the electrons and positrons.

### Photon kernel

The kernel $K_{ij}(l)$ counts the photons that the particles of grid points $i$ and $j$ put into photon bin $l$. It counts them per unit time and unit density of each particle, in units of $\sigma_T c$. The method `PairAnnihilation.initialize_kernels` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) sets up this kernel.

For the pairs with a particle of Lorentz factor 100 or less, `PairAnnihilation._fill_pair` and `PairAnnihilation._deposit` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) fill the kernel in two steps. The first step splits the photon energies into segments. The second step shares the photons of a segment, of mean energy $\bar k$, between the two photon grid points around $\bar k$. The point $k_l$ gets the fraction $1 - {w}$ of them, and $k_{l+1}$ gets the fraction $w$, where

$$
(1 - w)\thinspace k_l + w\thinspace k_{l+1} = \bar k,\qquad w = \frac{\bar k - k_l}{k_{l+1} - k_l}.
$$

This sharing keeps the number and the energy of the photons. How the code finds the photons of a pair depends on the pair's Lorentz factors:

- A particle and one at rest: the primitives $P$ and $E$ of eq. 41 give the photons of each segment.
- Other pairs with a particle of Lorentz factor 100 or less: Gauss–Legendre rules integrate eq. 55 over each segment.
- Pairs with both Lorentz factors above 100: the code shares out the two photons of eq. 59 in the same way (`PairAnnihilation._split` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo)).

The method `PairAnnihilation.__init__` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) sets up a map of the quadrature nodes:

$$
t = \frac{1 - \cos\pi s}{2} = \sin^2\frac{\pi s}{2},\qquad dt = \frac{\pi}{2}\sin(\pi s)\thinspace ds.
$$

Here $t$ is the position in a segment, from 0 at one end to 1 at the other. The Gauss–Legendre nodes lie in $s$. The map smooths a square-root edge at an end: $\sqrt{t}$ becomes $\sin(\pi s/2)$. Where the whole spectrum lies on the grid, the method `PairAnnihilation._store` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo) scales the kernel to the photon number of the normalization, $\sum_l K_{ij}(l) = 2R_{ij}$.

### Photon source on the grid

The code sums the kernel over the pairs of grid points with `PairAnnihilation.Calc_PairAnn` and `ann_step_work` in [pair_annihilation.mojo](../../../src/am3_mojo/physics/pair_annihilation.mojo). With the particles of each bin at its grid point, the photon source per unit $\ln k$, averaged over bin $l$, is

$$
Q_\gamma(k_l) = \sigma_T c\thinspace\Delta\sum_{i,j} N_+(\gamma_i)\thinspace N_-(\gamma_j)\thinspace K_{ij}(l).
$$

The emissivity is symmetric in $\gamma_+$ and $\gamma_{-}$, so $K_{ji} = K_{ij}$. One kernel thus serves both charge orders of a pair of grid points. The method `SimulationManager.Pass_Coeff_Phot` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) makes $Q_\gamma$ the source of the annihilation photons.
