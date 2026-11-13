# Neutron decay

A free neutron has one decay mode, $\mathrm{n} \to \mathrm{p}\thinspace\mathrm{e}^-\bar\nu_e$. Its mean life is $\tau_n$ = 878.3 s, the value of the Particle Data Group ([PDG 2026](https://pdg.lbl.gov/2026/)). This page starts from the allowed β spectrum and Fermi function of [Hayen et al. (2018)](https://arxiv.org/abs/1709.07530), the constants of PDG 2026 and the decay's kinematics. From them, the page derives the spectra and kernels of am3_mojo, and the corresponding struct and function. The module [neutron_decay.py](neutron_decay.py) checks every step. Run it with `.venv/bin/python tests/literature_checks/common/run.py neutron_decay`.

## The decay at rest

### End point

In the neutron's rest frame, the electron (total energy $E$, momentum $p$) and the antineutrino share the energy

$$
E_0 = \frac{m_n^2 - m_p^2 + m_e^2}{2m_n} = 1.29258\ \mathrm{MeV},
$$

with the masses of the neutron, proton and electron. On this page, masses and momenta are given as energies, in units with $c = 1$. Take the proton and the antineutrino together as one system, whose mass $m_{p\bar\nu}$ is at least the sum of the two masses. Four-momentum conservation then gives the electron's energy $E = (m_n^2 + m_e^2 - m_{p\bar\nu}^2)/(2m_n)$, and this energy falls as $m_{p\bar\nu}$ grows. The antineutrino has no mass, so the system's smallest mass is $m_{p\bar\nu} = m_p$, and the electron's largest total energy is $E_0$. Its largest kinetic energy, $E_0 - m_e$, is 781.6 keV.

The masses and the fine-structure constant $\alpha$ are the values of [PDG 2026](https://pdg.lbl.gov/2026/). These masses are the [CODATA 2022](https://doi.org/10.1103/RevModPhys.97.025002) recommended values of the fundamental constants. In the code, the four values are `MASS_N`, `MASS_P`, `MASS_E` and `ALPHA` in [constants.mojo](../../../src/am3_mojo/utils/constants.mojo). The function `neutron_beta_endpoint` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) computes $E_0$.

The proton takes the remaining $m_n - E_0$, so energy is conserved exactly. This is the proton's largest energy, $m_n - E_0 = (m_n^2 + m_p^2 - m_e^2)/(2m_n)$. The proton reaches it when the electron and the antineutrino together have their smallest mass, $m_e$. Its kinetic part, $m_n - m_p - E_0$ = 751.2 eV, is the proton's largest recoil energy. [Dubbers & Schmidt (2011)](https://arxiv.org/abs/1105.3694) give 750 eV. The electron and antineutrino spectra below leave out this recoil.

### Allowed spectrum

[Hayen et al. (2018)](https://arxiv.org/abs/1709.07530) give the allowed β spectrum with all its corrections (their eq. 4):

$$
N(W)\thinspace dW = \frac{G_V^2 V_{ud}^2}{2\pi^3}\thinspace F_0(Z, W)\thinspace L_0(Z, W)\thinspace U(Z, W)\thinspace D_\mathrm{FS}(Z, W, \beta_2)\thinspace R(W, W_0)\thinspace R_N(W, W_0, M)\thinspace Q(Z, W)\thinspace S(Z, W)\thinspace X(Z, W)\thinspace r(Z, W)\thinspace C(Z, W)\thinspace D_C(Z, W, \beta_2)\thinspace p W (W_0 - W)^2\thinspace dW.
$$

This section and the next use [Hayen et al.](https://arxiv.org/abs/1709.07530)'s units. Energies are in units of the electron's rest energy $m_e c^2$, and momenta in units of $m_e c$. Here $N(W)\thinspace dW$ is the rate of the decays whose electron has a total energy between $W$ and $W + dW$. The end point is $W_0$, and $p = \sqrt{W^2 - 1}$ is the electron's momentum. The vector coupling constant is $G_{V}$, and $V_{ud}$ is the up–down element of the Cabibbo–Kobayashi–Maskawa quark-mixing matrix. The daughter nucleus has the charge number $Z$, the mass $M$ and the quadrupole deformation $\beta_2$. The Fermi function of a point charge, $F_0$, accounts for the Coulomb interaction between the electron and the daughter nucleus. The eleven other factors correct the spectrum. They account for the size and shape of the nucleus ($L_0$, $U$, $D_\mathrm{FS}$), radiative effects ($R$) and the nucleus's recoil ($R_{N}$, $Q$). The rest account for the atomic electrons ($S$, $X$, $r$) and nuclear structure ($C$, with weak magnetism, and $D_{C}$). This $X$ is not the label $X$ of a decay product in the [lab frame](#lab-frame). A free neutron has no atomic electrons, so $S$, $X$ and $r$ are 1.

am3_mojo sets all eleven factors to 1. [Hayen et al.](https://arxiv.org/abs/1709.07530) find that the Fermi function holds most of the Coulomb effect and that its further corrections are typically a few percent at most. The code leaves out the photons of the radiative decays as well. It also leaves out the bound decay into a hydrogen atom, $4.2 \times 10^{-6}$ of the decays (Hayen et al., Sec. VII.E).

### Fermi function

[Hayen et al.](https://arxiv.org/abs/1709.07530)'s eqs. 6 and 7 give the Fermi function of a point charge:

$$
F_0(Z, W) = 4 (2pR)^{2(\gamma - 1)} e^{\pi y} \frac{\vert\Gamma(\gamma + i y)\vert^2}{\left( \Gamma(1 + 2\gamma) \right)^2},\qquad \gamma = \sqrt{1 - (\alpha Z)^2},\quad y = \pm \alpha Z W / p.
$$

Here $\Gamma$ is the gamma function, and $i$ is the imaginary unit. Eq. 7 gives the exponent $\gamma$ and the Coulomb parameter $y$. Both depend on the daughter's charge number $Z$. The upper sign of $y$ holds for electrons. The cut-off radius $R$ stands for the radius of the daughter nucleus. This $R$ is not the factor $R(W, W_0)$ of eq. 4.

The daughter of the neutron is the proton. So am3_mojo takes the Fermi function of a point charge $Z = 1$: [Hayen et al.](https://arxiv.org/abs/1709.07530)'s eq. 6 with their $\gamma = \sqrt{1 - \alpha^2}$ (written $\gamma_1$ here) set to 1. This approximation neglects $\alpha^2 = 5.3 \times 10^{-5}$ against 1, and the end of the [Spectrum](#spectrum) section bounds its effect. With $\gamma_1$ set to 1, the factor $(2pR)^{2(\gamma - 1)}$ is 1, which removes the cut-off radius, and the denominator $\Gamma(3)^2 = 4$ cancels the prefactor 4. For a real $y$, $\vert\Gamma(1 + iy)\vert^2$ $= \Gamma(1 + iy)\thinspace\Gamma(1 - iy)$, because the two factors are complex conjugates. The recurrence $\Gamma(1 + z) = z\thinspace\Gamma(z)$, valid for any complex $z$, and the reflection formula of the gamma function then give

$$
\Gamma(1 + iy)\thinspace\Gamma(1 - iy) = \frac{\pi y}{\sinh \pi y},\qquad F_0 \to e^{\pi y}\thinspace\frac{\pi y}{\sinh \pi y} = \frac{2\pi y}{1 - e^{-2\pi y}}.
$$

The arrow shows what $F_0$ becomes with $\gamma_1$ set to 1. At $Z = 1$ their $y = \alpha W/p$ is the Sommerfeld parameter of the electron. This parameter depends on $W$ and $p$ only through their ratio. The ratio is the same in any units.

### Spectrum

In this page's units, [Hayen et al.](https://arxiv.org/abs/1709.07530)'s energies are $W = E/m_e$ and $W_0 = E_0/m_e$, and their $p$ is the electron's momentum over $m_e$. Their eq. 4 without its corrections and with this form of $F_0$ is then the spectrum that am3_mojo uses:

$$
\frac{dN}{dE} \propto F(E)\thinspace p\thinspace E\thinspace (E_0 - E)^2,\qquad E_{\bar\nu} = E_0 - E,\qquad F(E) = \frac{2\pi\eta}{1 - e^{-2\pi\eta}},\quad \eta = \frac{\alpha E}{p}.
$$

Here $dN/{dE}$ is the electron's energy distribution, given up to a constant factor. In it, $F$ is the Fermi function, and $\eta$ is the Sommerfeld parameter. The spectrum leaves out the proton's recoil, so the antineutrino takes the rest of $E_0$, the energy $E_{\bar\nu}$.

Setting $\gamma_1$ to 1 in the [Fermi function](#fermi-function) also removes the proton's size from $F$. Keeping $\gamma_1$ and the proton's size changes $F$ by less than $10^{-3}$ for kinetic energies above 0.1 keV. For this comparison, the cut-off radius $R$ of eq. 6 is the radius of a uniformly charged sphere. That sphere's root-mean-square radius is the proton's charge radius, $r_p = 0.84075$ fm ([PDG 2026](https://pdg.lbl.gov/2026/)), so its radius is $R = \sqrt{5/3}\thinspace r_p$.

### Spectrum on the code's nodes

The Fermi function grows like $p^{-1}$ at the bottom of the spectrum, where the electron comes to rest. But the product $F\thinspace p$ stays finite:

$$
F(E)\thinspace p = \frac{2\pi\alpha E}{1 - e^{-2\pi\alpha E/p}},\qquad \lim_{p \to 0} F(E)\thinspace p = 2\pi\alpha m_e,
$$

and the function `neutron_beta_density` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) evaluates $dN/{dE}$ with $F\thinspace p$ in this form. The code then puts the spectrum onto the nodes of a Gauss–Legendre quadrature rule, after the substitution

$$
E = m_e + (E_0 - m_e)\sin^2\frac{\pi u}{2},\qquad E_0 - E = (E_0 - m_e)\cos^2\frac{\pi u}{2},\qquad \frac{dE}{du} = \pi\thinspace (E_0 - m_e)\sin\frac{\pi u}{2}\cos\frac{\pi u}{2},
$$

with the new variable $u$ from 0 to 1. This substitution gives the kinetic energy $T = E - m_e$ and the antineutrino's energy $E_0 - E$ as products, free of cancellation. With the rule's nodes $\xi_k$ and weights $a_k$ for $-1 \le \xi \le 1$, node $k$ sits at $u_k = (1 + \xi_k)/2$ and gets the weight

$$
\omega_k \propto \frac{a_k}{2}\thinspace \frac{dE}{du}\thinspace \frac{dN}{dE},\qquad \sum_k \omega_k = 1,
$$

with both derivatives taken at $u_k$. The weights $a_k$ are halved because $u$ covers half the length of the rule's interval. The weights $\omega_k$ add up to 1, one electron per decay. The structs `GLRules` and `NDecBuild` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) compute these nodes and weights. Sums over the nodes replace the integrals over the rest-frame spectrum.

## Lab frame

### Decay rate and source

A neutron of energy $E_n$ moves with the Lorentz factor $\gamma_n = E_n/m_n$ and the velocity $\beta_n$, in units of the speed of light. Time dilation makes it decay at the rate $1/(\gamma_n\tau_n)$. The neutrons' decays make each product $X$ (a proton, an electron or an antineutrino) at the rate

$$
Q_X(E) = \int \frac{dE_n}{E_n}\thinspace \frac{N_n(E_n)}{\gamma_n\thinspace \tau_n}\thinspace F_X\negthinspace \left(\frac{E}{E_n};\beta_n\right),
$$

where $N_n$ is the neutron density per unit energy, $Q_{X}$ the source of product $X$ per unit energy, and $E$ the product's lab-frame energy. This $E$ is not the electron's rest-frame energy of the sections above. The product's share of the neutron's energy is $x = E/E_n$. Its distribution per decay, $F_X(x;\beta_n) = dn/dx$, also depends on $\beta_n$. Here $n$ is the number of products per decay.

The integral counts the decays. The $N_n\thinspace dE_n$ neutrons with energies from $E_n$ to $E_n + dE_n$ decay at the time-dilated rate. Each decay puts $F_X\thinspace dx$ products between $x$ and $x + dx$, where $dx = dE/E_n$ at a fixed $E_n$. This relation gives the integral its factor $1/E_n$.

### One product

Take a product with the energy $E^\ast$ and the momentum $p^\ast$ in the neutron's rest frame, emitted at the angle $\theta^\ast$ to the neutron's direction. A star marks the quantities of that frame. The Lorentz boost to the lab gives the product the energy $E = \gamma_n(E^\ast + \beta_n p^\ast\cos\theta^\ast)$. Division by the neutron's energy gives the share $x = (E^\ast + \beta_n p^\ast\cos\theta^\ast)/m_n$. This share is linear in $\cos\theta^\ast$, and the cosine is uniform between −1 and 1 for an isotropic emission. So the product's share is spread uniformly over

$$
x \in \left[\frac{E^\ast - \beta_n p^\ast}{m_n},\ \frac{E^\ast + \beta_n p^\ast}{m_n}\right]
$$

with density $m_n/(2\beta_n p^\ast)$, the inverse of the range's width, for any $\beta_n$. For the electron and the antineutrino, $F_{X}$ sums these uniform ranges over the rest-frame spectrum, so it is not uniform itself:

$$
F_X(x;\beta_n) = \int dE^\ast\thinspace f_X(E^\ast)\thinspace \frac{m_n}{2\beta_n p^\ast}\thinspace H\negthinspace\left(\beta_n p^\ast - \vert m_n x - E^\ast\vert\right).
$$

Here $f_X$ is the product's energy distribution in the rest frame, normalized to 1. The unit step function $H$ is 1 for a positive argument and 0 otherwise. This function keeps the products of each energy $E^\ast$ within their range. The antineutrino has no mass, so $p^\ast = E^\ast$ for it. With its rest-frame energy $E_{\bar\nu}$ for $E^\ast$, its range is

$$
x \in \left[\frac{(1 - \beta_n)\thinspace E_{\bar\nu}}{m_n},\ \frac{(1 + \beta_n)\thinspace E_{\bar\nu}}{m_n}\right].
$$

am3_mojo neglects the proton's recoil momentum. So the proton's range shrinks to the share of its energy $m_n - E_0$, the single value $x_p = (m_n - E_0)/m_n$.

The function `_ndec_column` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) computes the lower ends of the ranges and $1 - \beta_n$ as

$$
E^\ast - \beta_n p^\ast = \frac{m_e^2}{E^\ast + p^\ast} + (1 - \beta_n)\thinspace p^\ast,\qquad 1 - \beta_n = \frac{1}{\gamma_n^2\thinspace (1 + \beta_n)}.
$$

The first form holds because the electron's energy and momentum satisfy $E^{\ast 2} - p^{\ast 2} = m_e^2$, and the second follows from the definition of $\gamma_n$. The code computes $\beta_n = \sqrt{(\gamma_n - 1)(\gamma_n + 1)}/\gamma_n$ from the Lorentz factor.

### Means and end points

Each range has its middle at $E^\ast/m_n$. Averaged over the emission angle and the spectrum, a product therefore has $\langle x\rangle = \langle E^\ast\rangle/m_n$ for any $\beta_n$, where the angle brackets denote the average. The rest-frame spectrum gives the electron the mean energy $\langle E\rangle$ = 0.8121 MeV, and the antineutrino $E_0 - \langle E\rangle$ = 0.4805 MeV. So on average the antineutrinos take $5.11 \times 10^{-4}$ of the neutron's energy and the electrons $8.64 \times 10^{-4}$. The proton takes $x_p = 0.99862$, and the three shares add up to 1. [Lipari et al. (2007)](https://arxiv.org/abs/0704.0718) give the antineutrinos' mean share as $\langle E_{\bar\nu_e}\rangle/E_n \simeq 5.1 \times 10^{-4}$ (their Sec. VII.C).

For an ultra-relativistic neutron the antineutrino takes up to $x = 2(E_0 - m_e)/m_n = 1.664 \times 10^{-3}$ of the energy. This is the upper end of its range in the limit $\beta_n \to 1$. The antineutrino reaches it at its largest energy, $E_0 - m_e$, when the electron is at rest. [Lipari et al.](https://arxiv.org/abs/0704.0718) give the largest share as $E_{\bar\nu}^{\max}/E_n$ $\simeq 2(m_n - m_p - m_e)/m_n$ $\simeq 1.66 \times 10^{-3}$, with $m_n - m_p$ for $E_0$, which leaves out the recoil.

## Sources on the grids

The code puts these distributions on its energy grids (`AM3Arrays` in [arrays.mojo](../../../src/am3_mojo/model/arrays.mojo)). All grids are logarithmic, with the same step $\Delta$ in $\ln E$ (0.1), and each grid point stands for a bin of that width. The hadron grid starts at 1 GeV, the lepton grid at the electron's rest energy $m_e c^2$, and the neutrino grid lower still. On these grids, AM³ stores densities per unit $\ln E$. Such a density is the density per unit energy times $E$, because $d\ln E = dE/E$. An integral over $\ln E$ is then the sum of the values at the grid points times $\Delta$.

The neutron solver, a `SolverAlgorithm` in [solver.mojo](../../../src/am3_mojo/solver/solver.mojo), removes the neutrons of hadron point $i$ at the rate $1/(\gamma_i\tau_n)$. Here $\gamma_i = E_i/m_n$ is the Lorentz factor at the point's energy $E_i$.

### Decays per grid point

Per unit $\ln E$, the source of product $X$ is $E\thinspace Q_X(E)$, the source per unit energy times $E$. The measure of its integral is $dE_n/E_n = d\ln E_n$. Writing $E = xE_n$ turns the neutron density per unit energy, $N_n$, into the density per unit $\ln E$, $n_n$. This substitution also leaves a factor $x$:

$$
E\thinspace Q_X(E) = \int d\ln E_n\thinspace \frac{n_n(E_n)}{\gamma_n\tau_n}\thinspace x F_X(x;\beta_n),\qquad n_n = E_n N_n.
$$

On the grids, this integral becomes a sum over the hadron points times $\Delta$. The neutrons of point $i$ decay at the rate $w_i$ per unit volume and $\ln E$. Their decays make the source $q_X$ at point $j$ of the product's grid:

$$
w_i = \frac{n_n(E_i)}{\gamma_i\tau_n},\qquad q_X(E_j) = \sum_i K_{ji}\thinspace w_i.
$$

Here the kernel $K_{ji}$ is the number of products that one decay at hadron point $i$ puts at point $j$. The next section derives the kernel. Per unit volume and time, the $\Delta\thinspace w_i$ decays of point $i$ thus put $\Delta\thinspace K_{ji} w_i$ products at point $j$. That point's bin has the same width $\Delta$. Summed over $i$ and divided by that width, these products give $q_X(E_j)$, the source per unit $\ln E$ at the point's energy $E_j$. The method `NeutronDecay.Calc_NtoX` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) computes these sums for the electrons and the antineutrinos, $q_e$ and $q_{\bar\nu}$.

### Kernels

The method `NeutronDecay.initialize_kernels` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) tabulates the kernels as the integrals

$$
K_{ji} = \int dx\thinspace F_X(x;\beta_i)\thinspace \Lambda_j(x),\qquad x_j = \frac{E_j}{E_i}.
$$

Here $\beta_i$ is the velocity of a neutron of energy $E_i$. A product at point $j$ has the share $x_j$ of that energy. The hat function $\Lambda_j$ of point $j$ rises linearly from 0 at the point below to 1 at point $j$. It falls back to 0 at the point above:

$$
\Lambda_j(x) = \frac{x - x_{j-1}}{x_j - x_{j-1}}\quad (x_{j-1} \le x \le x_j),\qquad \Lambda_j(x) = \frac{x_{j+1} - x}{x_{j+1} - x_j}\quad (x_j \le x \le x_{j+1}),
$$

and it is 0 elsewhere. Between the grid's end points, the hat functions add up to 1, and $\sum_j x_j\Lambda_j(x) = x$. So each column of $K$ keeps the number and the energy of one decay's products that land on the grid. Because the ranges of $x$ depend on $\beta_i$, the kernels depend on $E_i$ itself, not only on the ratio $E_j/E_i$.

The functions `ndec_build_work`, `_ndec_column`, `_bin_box` and `_share_to_points` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) compute the columns. For each rest-frame node $k$, the product's range of $x$ is a uniform box from $a$ to $b$. The box holds the node's weight $\omega_k$. On the interval from $x_j$ to $x_{j+1}$, the box's integrals against the hat functions need only the number and energy of its products there. The function `_bin_box` computes them as

$$
\mathcal N_j = \omega_k\thinspace\frac{x_b - x_a}{b - a},\qquad \mathcal E_j = \mathcal N_j\thinspace\frac{x_a + x_b}{2},
$$

where the box overlaps the interval from $x_a = \max(a, x_j)$ to $x_b = \min(b, x_{j+1})$. So $\mathcal N_j$ is the number of products in the overlap. Their energy in units of $E_i$ is $\mathcal E_j$, the number times the overlap's middle. The function `_share_to_points` then gives the interval's products to its two points. The interval's lower point $j$ gets the share $K_j^\mathrm{lo}$, and its upper point $j + 1$ gets the share $K_j^\mathrm{hi}$:

$$
K_j^\mathrm{lo} = \frac{x_{j+1}\mathcal N_j - \mathcal E_j}{x_{j+1} - x_j},\qquad K_j^\mathrm{hi} = \frac{\mathcal E_j - x_j\mathcal N_j}{x_{j+1} - x_j}.
$$

These are the integrals of the box over the interval against $\Lambda_j$ and $\Lambda_{j+1}$. Together they keep the interval's number and energy. Summed over the intervals and the nodes, they make the column $K_{ji}$. The code's kernels are therefore the integrals $K_{ji}$ above, up to the quadrature of the rest-frame spectrum.

### Protons

For a neutron at hadron point $i$, the proton's single share $x_p$ lies between the points $i - 1$ and $i$. In units of $E_i$, these points sit at $E_{i-1}/E_i$ and 1. The hat functions split each decay's proton between them. The fraction $s_i = \Lambda_{i-1}(x_p)$ goes one point down, and $1 - s_i = \Lambda_i(x_p)$ stays at point $i$. With $1 - x_p = E_0/m_n$, this gives

$$
s_i = \frac{E_0/m_n}{1 - E_{i-1}/E_i},\qquad q_p(E_i) = (1 - s_i)\thinspace w_i + s_{i+1}\thinspace w_{i+1}.
$$

Here $q_p$ is the protons' source per unit $\ln E$. Point $i$ keeps $1 - s_i$ of its own decays' protons and receives $s_{i+1}$ of those from point $i + 1$. The struct `NeutronDecay` in [decays.mojo](../../../src/am3_mojo/physics/decays.mojo) computes $s_i$ and $q_p$. The equal steps make every $s_i$ the same, 0.0145. The protons that point 0 sends down fall below the grid and are lost.

### In a time step

In each time step, the methods `SimulationManager.evolve_step` and `SimulationManager.Calc_Neutron_Decay` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) compute these sources. The methods `SimulationManager.Pass_Coeff_PRO`, `SimulationManager.Pass_Coeff_Lep` and `SimulationManager.Calc_Coeff_NU` in [manager.mojo](../../../src/am3_mojo/simulation/manager.mojo) then add $q_p$, $q_e$ and $q_{\bar\nu}$ to the sources of the protons, electrons and electron antineutrinos. The lepton solver takes the electrons' $q_e$ divided by `RATE_SSC` in [constants.mojo](../../../src/am3_mojo/utils/constants.mojo).
