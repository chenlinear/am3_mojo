# am3_mojo

**am3_mojo** is the [AM³](https://gitlab.desy.de/am3/am3) lepto-hadronic interactions code written in [Mojo](https://www.modular.com/mojo). It keeps the Python interface of AM³ v1.2.0. On 4 threads and for the same physics, it runs $\boldsymbol{4.1\text{–}6.0\times}$ **faster** per step than the C++ AM³ (serial), and uses **up to $22\times$ less memory**, and its kernel set-up is up to $\boldsymbol{3.7\times}$ **faster**. With the same fixes applied to the C++ code (see [Differences](#differences-from-the-c-original)), the simulated fluxes of the two codes agree to $10^{-7}\text{–} 10^{-5}$ or better in most energy bins. They agree to better than $1\text{％}$ everywhere.

In addition, am3_mojo adds four processes that AM³ lacks, chosen from Table 1 of [Cerruti et al. (2026, ApJS 282, 22)](https://arxiv.org/abs/2411.14218). They are **electron–positron annihilation** ($\mathrm{e}^+\mathrm{e}^- \to \gamma\gamma$), **triplet pair production** ($\mathrm{e}^\pm\gamma$ $\to \mathrm{e}^\pm\mathrm{e}^+\mathrm{e}^-$), **neutron decay** ($\mathrm{n} \to \mathrm{p}\thinspace\mathrm{e}^-\bar\nu_e$) and **kaon synchrotron and inverse Compton radiation** (including the production and decay of $\mathrm{K}^+$, $\mathrm{K}^-$, $\mathrm{K}^0_\mathrm{S}$ and $\mathrm{K}^0_\mathrm{L}$). AM³ already implements proton-proton pion production. All four added processes are on by default. With them, a step takes $27\text{–}71\text{％}$ more time. Switching them off gives AM³'s physics, and all comparisons with the C++ AM³ run this way (see [Processes am3_mojo adds to AM³](#added-processes)).

```python
import am3_mojo
a = am3_mojo.AM3()
a.set_process_pp(True)          # switch processes on/off
a.init_kernels()                # build the interaction kernels (once)
a.set_mag_field(10.0)           # source parameters
a.set_escape_timescale(3.3e4)
a.set_powerlaw_injection_parameters_protons(4.2e45, 1e46, 1.9e9, 1.9e9, 9.4e16, 2.0, 2.0, 1.0)
for _ in range(800):
    a.evolve_step()
E, photons = a.get_egrid_photons(), a.get_photons()   # E [eV], E dN/dE dV [cm^-3]
```

## Repository layout

```
src/
  am3_mojo_module.mojo     the Python extension module am3_mojo (class AM3), generated
                           by tools/gen_bindings.py
  am3_mojo/                the Mojo package
    utils/                 numerics: constants, libm wrappers, buffers, SIMD linear
                           algebra (Toeplitz/Hankel products), interpolation, tridiagonal
                           solver, Bessel functions, worker-pool parallelism
    model/                 run parameters, energy grids, particle densities
    solver/                the implicit PDE solver (Chang-Cooper + semi-analytic regimes)
    physics/               one module per process (synchrotron, inverse_compton,
                           pair_production, pair_annihilation, triplet (+ its QED
                           tables, triplet_tables), bethe_heitler, photopion (+ kaon
                           production, photopion_tables), proton_proton, decays (pions,
                           muons, neutrons, kaons; also the hadrons' inverse Compton),
                           acceleration, escape, expansion, extra_cooling, injection)
    simulation/            the time step: coefficients and parallel species updates
    api/                   the Python-facing AM3 object
tests/
  unit/                    tests of am3_mojo on its own
  reference/               comparison with the C++ AM³ (+ setup_reference.sh that
                           downloads and builds it, and am3_fixes.patch, the fixes of
                           the AM³ bugs that am3_mojo fixes)
  harness/                 Mojo programs that print the code's kernels and tables, for
                           the unit tests and the checks
  literature_checks/       checks of the equations of the added processes and of AM³'s
                           inverse-Compton loss and quantum-synchrotron cooling against
                           the papers: derivations/ (every equation of this README
                           against its source, and the equations the other checks
                           use) and the Monte Carlo and data comparisons
  solver_checks/           checks of AM³'s solver
  run_checks.py            runs literature_checks/ and solver_checks/ (make checks;
                           not part of make test)
  benchmark/               timings and per-stage profiles vs the C++ AM³
  plots/                   comparison, kernel-reuse and added-process figures
  support/                 what the tests, checks and figures share: the scenarios,
                           the runs of each implementation in its own process
                           (runs.py and the scripts it runs), the comparison
                           criterion (compare.py), the units and set-ups of the unit
                           tests (units.py, setups.py), the harness builds and
                           measured/ (the measured values behind the kaon channels)
results/                   tables and figures of the last full run (make all)
tools/                     what generates code and data: the binding generator, the
                           AM³ v1.2.0 interface specification (am3_api.json), the
                           methods am3_mojo adds (am3_mojo_api.json), the build backend
                           used by `pip install` (mojo_build_backend.py), triplet/, the
                           QED Monte Carlo that computes the triplet pair production
                           tables, and ky_angular_extract.py, which regenerates the
                           measured data of the two-body kaon channels
Makefile                   build / install / test / reference / benchmark / plots /
                           checks / all
CITATION.cff               how to cite am3_mojo (and AM³)
pyproject.toml, uv.lock    the Python environment (Mojo 1.1.0, numpy, and the packages
                           of the tests, the checks and the generators in tools/)
```

## Installation

There are two ways to install am3_mojo. Most users only need the first.

| | Install am3_mojo only | Development set-up |
|---|---|---|
| What you get | `import am3_mojo` in your Python environment | the module, plus the tests, the C++ reference, benchmarks, plots and the checks against the literature |
| Python | 3.14 with pip (or uv) | uv, which installs Python 3.14 |
| `make`, `git`, `curl` | not needed | needed |
| Mojo | needed, but installed automatically by pip (see below) | installed by `uv sync` |
| C/C++ compiler | a C compiler (Mojo uses it to link) | needed (linking, and building the C++ reference) |

**Operating system:** macOS on Apple Silicon (tested on an Apple M5, macOS 26), or Linux on x86-64 or ARM64. Mojo supports Linux, but am3_mojo has not been tested on it yet.

### Install am3_mojo only

`pip` compiles am3_mojo on your machine with the Mojo compiler, for that machine's CPU. This takes about 10 s. The only other requirement is a C compiler, which Mojo uses as its linker. Use the Xcode Command Line Tools on macOS (`xcode-select --install`), or gcc/clang on Linux.

```bash
# an unpacked copy of the repository
pip install ./am3_mojo
```

What pip does with Mojo:

1. **While installing**, pip creates a temporary build environment and installs `mojo-compiler==1.1.0` into it (declared under `[build-system]` in `pyproject.toml`). It compiles am3_mojo with this compiler and then deletes the build environment.
2. **In your environment**, pip installs am3_mojo together with its two dependencies, `numpy` and `mojo-compiler==1.1.0`. The module loads the Mojo runtime libraries from that package (`site-packages/modular/lib`). Do not uninstall `mojo-compiler`: without it, `import am3_mojo` fails. The package also adds a `mojo` command to the environment, but am3_mojo itself does not use it.

`uv pip install` works the same way.

Check the installation:

```bash
python -c "import am3_mojo; print(am3_mojo.__version__, am3_mojo.AM3())"
```

### Development set-up

This set-up needs uv, `make`, `git`, `curl` and a C/C++ compiler. `git` and `curl` download the original C++ AM³ and its dependencies for the reference tests. The C++ compiler builds them.

```bash
git clone https://github.com/chenlinear/am3_mojo && cd am3_mojo
uv sync            # .venv with Python 3.14, Mojo 1.1.0, numpy, and the packages of the tests
                   # and of the checks (see below)
make build         # compile the module -> build/am3_mojo.so  (about 10 s)
make install       # make `import am3_mojo` work anywhere in .venv (the tests and checks
                   # find build/ without it)
```

Check the installation (after `make install`):

```bash
.venv/bin/python -c "import am3_mojo; print(am3_mojo.__version__, am3_mojo.AM3())"
```

To run all the tests and the checks against the literature, see [Tests, reference comparison and benchmarks](#tests-reference-comparison-and-benchmarks).

## Using am3_mojo

The `AM3` class provides the complete AM³ v1.2.0 Python interface: 383 methods with the same names, arguments and docstrings. Where am3_mojo differs, 65 docstrings are adapted (e.g. `True`/`False` for the switches). A further 76 methods cover the [processes am3_mojo adds](#added-processes). The [AM³ documentation](https://am3.readthedocs.io/en/latest/) and its example notebooks therefore apply, with `import am3` replaced by `import am3_mojo`. `help(am3_mojo.AM3.set_process_pp)` shows each method's documentation.

- **Threads.** The parallel sections use the Mojo runtime's worker pool. The default thread count is the pool's worker count, which equals the number of performance cores (4 on an M5). `AM3_MOJO_THREADS=N` overrides it, and `AM3_MOJO_THREADS=1` runs serially. Results are bitwise identical for any $N$.
- **Switches** are Python booleans: `set_process_pp(True)`, `get_process_escape() -> True`. The setters accept only `True` and `False`. Any other value, including `0`, `1` and numpy booleans, raises an exception. As in AM³, set them before `init_kernels()`, which builds only the kernels of the processes that are on.
- **Reusing kernels.** `init_kernels()` is the expensive part. To run several simulations with one set of kernels:
  ```python
  a = am3_mojo.AM3(); <switches>; a.init_kernels()      # once
  for p in parameter_sets:
      a.clear_am3()                    # particles, injection, escape and extra-cooling arrays -> 0
      a.set_mag_field(p.B)             # rebuilds only the B-dependent synchrotron kernels
      a.set_escape_timescale(p.t_esc)  # escape rates must be set again after clear_am3()
      a.set_powerlaw_injection_parameters_electrons(...)   # injection too
      for _ in range(n): a.evolve_step()
  ```

  A reused object gives bitwise identical results to a fresh one. `tests/unit/test_reuse.py` checks this, and `results/plots/reuse_summary.png` shows it. On the left, correctly reused kernels (blue) are bitwise identical to kernels built again before every run. Skipping `clear_am3()`, or not setting the escape timescale again, gives wrong spectra from the second run on. The grey squares show how far the C++ AM³ is from am3_mojo. On the right, the kernel set-up time for four runs drops by a factor of 4.

  ![Kernel reuse: deviation from re-initialising the kernels, and set-up time saved](results/plots/reuse_summary.png)

  The figures below show the individual runs for the leptohadronic and the blazar setup. Each shows the photon spectra (top) and, per energy bin, the difference from re-initialising in the photon, $\mathrm{e}^\pm$, proton and neutrino spectra (bottom). Every getter is bitwise identical to the run with re-initialised kernels.

  ![Photon spectra and differences of four runs with reused kernels, leptohadronic setup](results/plots/reuse_leptohadronic.png)

  ![Photon spectra and differences of four runs with reused kernels, blazar setup](results/plots/reuse_blazar.png)

  `clear_am3()` sets the particle densities, the injection, escape and extra-cooling rates and all coefficient arrays to zero. As in AM³, it keeps the kernels, the switches and the scalar parameters ($B$, timescales, $\mathrm{pp}$ density, time step, power-law parameters). $B$, the expansion timescale, the $\mathrm{pp}$ density and the time step therefore act in the next run without being set again. The escape and injection rates, however, stay zero until `set_escape_timescale()` and `set_powerlaw_injection_parameters_*()` (or `set_injection_rate_*()`) are called again. Yet `get_escape_timescale()` and `get_powerlaw_injection_parameters_*()` keep reporting the old values. (With `set_estimate_max_energies(True)`, the power-law injection is computed again from the stored parameters, starting with the second step.)

<a id="added-processes"></a>

## Processes am3_mojo adds to AM³

Table 1 of the hadronic code comparison of [Cerruti et al. (2026, ApJS 282, 22)](https://arxiv.org/abs/2411.14218) lists the physical processes of five codes. Four of the processes that other codes have and AM³ lacks are added to am3_mojo:

| Process (Table 1) | AM³ | ATHEνA | [Böttcher et al. 2013](https://arxiv.org/abs/1304.0605) | LeHa-Paris | LeHaMoC | am3_mojo |
|---|---|---|---|---|---|---|
| electron–positron annihilation | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ ([Svensson 1982](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S): exact; peak approximation where both $\gamma \gt 100$) |
| triplet pair production | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ (exact lowest-order QED, all three products) |
| neutron decay | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ (allowed $\beta$ spectrum, exact boost; protons, electrons and antineutrinos) |
| kaon synchrotron radiation | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ ($\mathrm{K}^\pm$, $\mathrm{K}^0_\mathrm{S}$, $\mathrm{K}^0_\mathrm{L}$ from $\mathrm{p}\gamma$, $\mathrm{n}\gamma$ and $\mathrm{pp}$, evolved as species) |

All four are **on by default**. Every comparison with the C++ AM³ switches them off (the test implementation `mojo-am3` in `tests/support/paths.py`). With them off, none of their terms acts, so am3_mojo solves AM³'s equations.

| Process | Switches (default `True`) | Outputs and timescales |
|---|---|---|
| $\mathrm{e}^+\mathrm{e}^- \to \gamma\gamma$ | `set_process_pair_annihilation` (master switch); `_emission` / `_cooling`: the photons / the $\mathrm{e}^\pm$ loss | `get_photons_pair_annihilation()`; `get_t_electron_annihilation()`, `get_t_positron_annihilation()` |
| $\mathrm{e}^\pm\gamma$ $\to \mathrm{e}^\pm\mathrm{e}^+\mathrm{e}^-$ | `set_process_triplet_pair_production` (master switch; also the photon absorption); `_emission` / `_cooling`: the produced pairs / the $\mathrm{e}^\pm$ energy loss | `get_pairs_triplet()` (the $\mathrm{e}^\pm$ component) with its photons `get_photons_triplet_pairs_syn()`, `get_photons_triplet_pairs_compton()` and `get_photons_triplet_pairs_syn_compton()`; `get_t_pair_triplet()` ($\mathrm{e}^\pm$ energy loss), `get_t_photon_triplet()` (photon absorption) |
| $\mathrm{n} \to \mathrm{p}\thinspace\mathrm{e}^-\bar\nu_e$ | `set_process_neutron_decay`: the decay of the neutrons and its protons, electrons and electron antineutrinos | `get_electrons_neutron_decay()` (the $\mathrm{e}^\pm$ component) with its photons `get_photons_neutron_decay_electrons_syn()`, `get_photons_neutron_decay_electrons_compton()` and `get_photons_neutron_decay_electrons_syn_compton()`; `get_electron_anti_neutrinos_neutron_decay()` (the neutrino component); `get_t_neutron_decay()` |
| kaons | `set_process_kaons` (master switch): $\mathrm{K}^+$, $\mathrm{K}^-$, $\mathrm{K}^0_\mathrm{S}$, $\mathrm{K}^0_\mathrm{L}$ production in photo-hadronic interactions and $\mathrm{pp}$ collisions, their evolution and decay | `get_kaon_plus()`, `get_kaon_minus()`, `get_kaons()` ($\mathrm{K}^+ + \mathrm{K}^-$), `get_kaon_0S()`, `get_kaon_0L()`; `get_t_kaon_escape()` (charged), `get_t_kaon_neutral_escape()`; a user-set extra cooling, as for AM³'s other species: `set_t_kaon_extra_cooling()` / `get_t_kaon_extra_cooling()` |
| kaon decays | `set_process_kaon_decay`: every non-radiative mode above $0.1\text{％}$ and the smaller $\mathrm{K}^0_\mathrm{S}$ semileptonic and $\mathrm{K}^0_\mathrm{L} \to \pi^0\pi^0$ modes (see [Decays](#decays)) | `get_t_kaon_decay()` (charged), `get_t_kaon_0S_decay()`, `get_t_kaon_0L_decay()`; `get_neutrinos_kaon_decay()`, the neutrinos of the kaon decays and of the pions and muons the kaons decay into (per flavour `get_electron_neutrinos_kaon_decay()`, `get_electron_anti_neutrinos_kaon_decay()`, `get_muon_neutrinos_kaon_decay()`, `get_muon_anti_neutrinos_kaon_decay()`); the rest of AM³'s photo-hadronic and $\mathrm{pp}$ neutrino components, the neutrinos of the pions made in the interactions and of their muons: `get_neutrinos_pion_muon_decay_photopion()`, `get_neutrinos_pion_muon_decay_proton_proton()` (and per flavour, e.g. `get_muon_neutrinos_pion_muon_decay_photopion()`) |
| kaon synchrotron | `set_process_kaon_syn` (master switch); `_emission` / `_cooling`: the photons / the kaon energy loss | `get_photons_kaons_syn()`; `get_t_kaon_syn()` |
| kaon inverse Compton | `set_process_kaon_compton` (master switch); `_emission` / `_cooling`: the photons / the kaon energy loss | `get_photons_kaons_compton()`; `get_t_kaon_compton()` |

Each switch has a getter (`get_process_…`). (AM³'s own `set_process_annihilation`, `get_pairs_annihilation()` and `get_t_photon_annihilation()` belong to $\gamma\gamma$ pair production.) The printed switch table and the profiler (`get_profiled_time_labels()`) list the added processes too. Their methods are specified in `tools/am3_mojo_api.json`.

### Electron–positron annihilation

#### Choice of method

**The other codes.** Two codes of Table 1 have the process:
- **ATHEνA** ([Mastichiadis & Kirk 1995](https://arxiv.org/abs/astro-ph/9408077), section 3.9): relativistic leptons annihilate only with a pool of cold pairs. They annihilate at the rate fit of [Coppi & Blandford (1990)](https://doi.org/10.1093/mnras/245.3.453), $3/(8\gamma)\thinspace[\gamma^{-1/2} + \ln\gamma]$ (in $\sigma_T c$; Mastichiadis & Kirk's eq. 59). Their photons are put at $\gamma/2$. The cold pairs annihilate into a 511 keV line.
- **[Böttcher et al. (2013)](https://arxiv.org/abs/1304.0605)**: Table 1 marks the process, but neither the code's paper nor [Cerruti et al. (2026)](https://arxiv.org/abs/2411.14218) describes how it is treated. The papers that the code's leptonic part builds on leave annihilation out ([Böttcher, Mause & Schlickeiser 1997](https://arxiv.org/abs/astro-ph/9604003); [Böttcher & Bloom 2000](https://arxiv.org/abs/astro-ph/9806249); [Böttcher & Chiang 2002](https://arxiv.org/abs/astro-ph/0208238)). Elsewhere the group used an ultra-relativistic approximation of [Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S)'s result ([Böttcher & Schlickeiser 1996](https://arxiv.org/abs/astro-ph/9506112)). For transrelativistic pairs, it used Svensson's exact rate with the spectrum of a thermal plasma ([Böttcher, Pohl & Schlickeiser 1999](https://arxiv.org/abs/astro-ph/9802271), appendix A.4).

**The exact result.** [Svensson (1982)](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S) gives the angle-averaged annihilation rate and photon spectrum of isotropic electrons and positrons of any Lorentz factors in closed form. [Vurm & Poutanen (2009)](https://arxiv.org/abs/0807.2540) use the same rate. AM³ already uses their $\gamma\gamma$ pair-production treatment (through [Gao et al. 2017](https://arxiv.org/abs/1610.05306)). am3_mojo uses the exact result (`src/am3_mojo/physics/pair_annihilation.mojo`), with Svensson's peak approximation for the photons of pairs whose leptons both have $\gamma \gt 100$.

#### Equations

Energies are in units of $m_e c^2$ (photon energy $k$, lepton Lorentz factors $\gamma_\pm$, momenta $p = \sqrt{\gamma^2 - 1}$, velocities $\beta = p/\gamma$). With $n_\pm(\gamma)$ the $\mathrm{e}^\pm$ densities, annihilation removes electrons and positrons at the rates

$$\dot n_\mp(\gamma) = -\thinspace n_\mp(\gamma)\int d\gamma'\thinspace n_\pm(\gamma')\thinspace \langle\sigma v\rangle(\gamma',\gamma),$$

and creates photons with the source

$$\dot n_\gamma(k) = \iint d\gamma_+\thinspace d\gamma_-\thinspace n_+(\gamma_+)\thinspace n_-(\gamma_-)\thinspace v\frac{d\sigma}{dk}(k;\gamma_+,\gamma_-),$$

which counts both photons: $\int v\ \frac{d\sigma}{dk}\ dk = 2\langle\sigma v\rangle$ and $\int k\ v\ \frac{d\sigma}{dk}\ dk$ $= (\gamma_+ + \gamma_-)\langle\sigma v\rangle$.

**Rate** ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S) eq. 18). The angle average reduces to one integral over the relative Lorentz factor, whose primitive is closed-form:

$$\langle\sigma v\rangle(\gamma_+,\gamma_-) = c\thinspace \frac{G(\gamma_r^+) - G(\gamma_r^-)}{2\thinspace p_+p_-\thinspace \gamma_+\gamma_-},\qquad \gamma_r^\pm = \gamma_+\gamma_- \pm p_+p_-,$$

$$G(x) = \int_1^{x}\sqrt{t^2-1}\thinspace \sigma_{\rm rest}(t)\thinspace dt = \frac{3\sigma_T}{4}\left[\frac{(x-1)^{3/2}}{2\thinspace (x+1)^{1/2}}\thinspace \ell - (x+1) + \frac{3}{4}\thinspace \ell^2\right] + \frac{3\sigma_T}{2},\qquad \ell = \mathrm{arccosh}\thinspace x,$$

with the Dirac cross section of a lepton of Lorentz factor $t$ annihilating on one at rest,

$$\sigma_{\rm rest}(t) = \frac{3\sigma_T}{8\thinspace (t+1)}\left[\frac{t^2+4t+1}{t^2-1}\thinspace \ln\negthinspace \left(t+\sqrt{t^2-1}\right) - \frac{t+3}{\sqrt{t^2-1}}\right].$$

**Photon spectrum** ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S) eqs. 55–58). The spectrum is an integral over $\gamma_\ast$, the Lorentz factor of each lepton in the centre-of-momentum frame ($\gamma_\ast^2 = (1 + \gamma_r)/2$ for the relative Lorentz factor $\gamma_r$). The integral is done in closed form ($\gamma_\ast$ and $\gamma_s$ are Svensson's $\gamma_\mathrm{cm}$ and $\gamma^\ast_\mathrm{cm}$):

$$v\frac{d\sigma}{dk}(k;\gamma_+,\gamma_-) = \frac{3\sigma_T\thinspace c}{8\thinspace p_+\gamma_+\thinspace p_-\gamma_-}\Big[\sqrt{S^2 - 4\gamma_\ast^2} + H_+(\gamma_\ast) + H_-(\gamma_\ast)\Big]_ {\gamma_\ast^{\min}}^{\min(\gamma_\ast^{\max},\ \gamma_s)},$$

$$S = \gamma_+ + \gamma_-,\qquad \gamma_s^2 = k\thinspace (S-k),\qquad \gamma_\ast^{\max,\min} = \sqrt{\tfrac{1}{2}\left(1+\gamma_r^\pm\right)},$$

$$H_\pm(\gamma_\ast) = \left(2 + \frac{1-\gamma_s^2}{c_\pm}\right) I_\pm + \frac{1}{u_\pm}\left[\frac{1}{\gamma_\ast} - \frac{\gamma_\ast}{c_\pm} + \frac{\gamma_\ast \thinspace (2c_\pm - d_\pm)}{2\gamma_s^2}\right] + \frac{\gamma_\ast \thinspace u_\pm}{c_\pm},$$

$$c_\pm = (\gamma_\mp - k)^2 - 1,\quad d_\pm = \gamma_\mp S \pm k\thinspace (\gamma_+ - \gamma_-),\quad u_\pm = \sqrt{c_\pm\gamma_\ast^2 + \gamma_s^2},\quad I_\pm = \begin{cases}\ln\negthinspace \left(\gamma_\ast \sqrt{c_\pm} + u_\pm\right)/\sqrt{c_\pm} & c_\pm \gt 0\cr \arcsin\negthinspace \left(\gamma_\ast \sqrt{-c_\pm}/\gamma_s\right)/\sqrt{-c_\pm} & c_\pm \lt 0\end{cases}$$

The photons lie ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S) eqs. 33a–d) between the kinematic limits $k_L = \tfrac12[\gamma_+(1-\beta_+) + \gamma_-(1-\beta_-)]$ and $k_U = \tfrac12[\gamma_+(1+\beta_+) + \gamma_-(1+\beta_-)]$, and the spectrum peaks at

$$k_\pm = \tfrac{1}{2}\left[\gamma_\pm(1+\beta_\pm) + \gamma_\mp(1-\beta_\mp)\right].$$

**One particle at rest** ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S) eq. 41), for the other one with Lorentz factor $\gamma$:

$$v\frac{d\sigma}{dk}(k;\gamma) = \frac{3\sigma_T\thinspace c}{8\thinspace p\thinspace \gamma}\left[\frac{-\frac{3+\gamma}{1+\gamma} + \frac{3+\gamma}{k} - \frac{1}{k^2}}{\left(1 - \frac{k}{1+\gamma}\right)^2} - 2\right],\qquad \tfrac12\left(1 + \tfrac{1}{\gamma+p}\right) \le k \le \tfrac12\thinspace (1+\gamma+p).$$

**Both particles relativistic** ([Svensson](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S) eq. 59): the spectrum is approximated by one photon at each peak $k_\pm$.

### Triplet pair production

An electron or positron that hits a photon makes an $\mathrm{e}^+\mathrm{e}^-$ pair, $\mathrm{e}^\pm\gamma$ $\to \mathrm{e}^\pm\mathrm{e}^+\mathrm{e}^-$, if the photon's energy in the lepton's rest frame exceeds $4m_e c^2$. With $k$ this energy in units of $m_e c^2$, the cross section grows as $(28/9)\alpha r_e^2 \ln 2k$. At high energies it therefore exceeds the Klein–Nishina cross section, and the energy loss it causes exceeds that of inverse Compton. The code is in `physics/triplet.mojo`, with the tables of `physics/triplet_tables.mojo`, which the scripts in [`tools/triplet/`](tools/triplet/README.md) compute.

#### Choice of method

- **ATHEνA**, the only code of Table 1 with the process, follows [Mastichiadis (1991)](https://doi.org/10.1093/mnras/253.2.235) as [Petropoulou et al. (2019)](https://arxiv.org/abs/1907.03175) describe it. The leptons lose energy continuously at Mastichiadis' approximate rate. The pairs are injected with a power law (following [Lee 1998](https://arxiv.org/abs/astro-ph/9604098)), and no photons are removed. The published approximations of the loss rate and the pairs' energy differ from the exact result by up to a factor of 10 (see [Checks against the literature](#checks-against-the-literature)).
- **The exact result** at lowest order in QED has eight Feynman diagrams ([Haug 1975](https://doi.org/10.1515/zna-1975-0901)). [Lee, Lyubyakin & Stotsky (2021)](https://doi.org/10.1007/JHEP01%282021%29144) give the total cross section as one-fold integrals at any energy, with expansions near threshold and at high energy. The equation numbers cited here are those of the open-access journal version of their paper. The energy distributions of the products are known only numerically.

For each photon energy $k$, the Monte Carlo of `tools/triplet` integrates the exact QED matrix element over all final states. This gives the total cross section and the energy distributions of the three products. am3_mojo checks the cross section against [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144)'s exact result.

#### Equations

Energies are in units of $m_e c^2$. A lepton of Lorentz factor $\gamma \gg 1$ meets a photon of energy $\varepsilon$ at the angle $\theta$. The photon's energy in the lepton's rest frame is then $k = \gamma\varepsilon(1 - \cos\theta)$.

**Cross section.** The Monte Carlo's tables give $\sigma(k)$ at 60 values of $k$ from 4.02 to $10^{10}$. The code interpolates between them up to $k = 10^3$. Above this value, it takes only the energy distributions of the products from the tables. Below $k = 4.02$ and above $k = 10^3$, the code uses the expansions of [Lee et al.](https://doi.org/10.1007/JHEP01%282021%29144) instead. Near threshold it uses their eq. 4.5, and at high energy their eq. 4.13:

$$\frac{\sigma}{\alpha r_e^2} = \frac{\pi}{3\sqrt3}\left(\frac{4r^2}{27} + \frac{602\thinspace r^3}{243} + \frac{4217\thinspace r^4}{4374}\right),\qquad r = \frac{s-9}{s-1},\quad s = 1 + 2k\qquad (k \le 4.02),$$

$$\frac{\sigma}{\alpha r_e^2} = \frac{28}{9}\ln 2k - \frac{218}{27} - \frac{1}{k}\left(\frac{4}{3}\ln^3 2k - 4.08\ln^2 2k + 12.85\ln 2k - 30.89\right)\qquad (k \ge 10^3),$$

**Energies of the products.** For $\gamma \gg 1$ and $\varepsilon \ll \gamma$, each product carries the fraction

$$x = \frac{p\cdot k}{p_1\cdot k}$$

of the lepton's lab energy, where $p$, $p_1$ and $k$ are the four-momenta of the product, the lepton and the photon. This fraction does not depend on the direction of the photon. In the lepton's rest frame, with the photon along $z$, we have $x = E - p_z$. The fractions of the three products add up to 1: $x_+$ for the positron, and $x_\mathrm{low}$ and $x_\mathrm{high}$ for the electrons of lower and higher lab energy. The two electrons are identical, so the one of higher energy is counted as the lepton that comes back. This lepton loses the fraction $\xi = 1 - x_\mathrm{high} = x_+ + x_\mathrm{low}$ of its energy.

For every $k$, the Monte Carlo's tables hold the distribution of each product, $f_r(k, y) = dN/d\ln y$. This is the number of the product per interaction and per unit $\ln y$, normalised to one. For the positron and the electron of lower energy, $y = xk = p\cdot k$ ($y = x_+ k$ and $y = x_\mathrm{low} k$). For the lepton that comes back, $y = \xi k$ is defined through the energy it loses. The code takes the mean loss of this lepton from the energies of the deposited pair. These are the products of an electron. A positron gives their charge conjugates.

**Rates.** In an isotropic photon field, the interaction rate of a lepton and the spectra of its products depend only on $P = \gamma\varepsilon$:

$$R(P) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k\thinspace \frac{\sigma(k)}{\sigma_T},\qquad G_r(P, x) = \frac{1}{2P^2}\int_4^{2P} dk\thinspace k\thinspace \frac{\sigma(k)}{\sigma_T}\thinspace f_r(k, xk).$$

A lepton therefore interacts at the rate $c\sigma_T \int d\ln\varepsilon\ n_\gamma(\varepsilon)$ $R(\gamma\varepsilon)$, with $n_\gamma(\varepsilon)$ the photon density per $\ln\varepsilon$.

**Kinetic terms.** With the densities $n(\gamma)$ per $\ln\gamma$ ($\mathrm{e}^-$ and $\mathrm{e}^+$) and $n_\gamma(\varepsilon)$ per $\ln\varepsilon$ (photons):
- the produced pairs are sources, one positron and one electron per interaction. The labels above hold for an $\mathrm{e}^-$ target. An $\mathrm{e}^+$ target gives an $\mathrm{e}^-$ with $x_+$ and an $\mathrm{e}^+$ with $x_\mathrm{low}$. The positron source is

  $$\dot n_{+}(\gamma) = c\thinspace \sigma_T\int d\ln\gamma'\int d\ln\varepsilon\thinspace n_\gamma(\varepsilon)\left[n_-(\gamma')\thinspace G_{+}\negthinspace \left(\gamma'\varepsilon,\ \frac{\gamma}{\gamma'}\right) + n_+(\gamma')\thinspace G_{\rm low}\negthinspace \left(\gamma'\varepsilon,\ \frac{\gamma}{\gamma'}\right)\right],$$

  and likewise for the electrons, with $n_{-}$ and $n_+$ exchanged ($n_- G_\mathrm{low} + n_+ G_+$);
- the lepton that comes back loses the energy the pair takes, which the code treats as a continuous loss:

  $$-\frac{\dot\gamma}{\gamma} = c\thinspace \sigma_T\int d\ln\varepsilon\thinspace n_\gamma(\varepsilon)\thinspace R(\gamma\varepsilon)\thinspace \langle x_+ + x_{\rm low}\rangle(\gamma\varepsilon),$$

  with $\langle\ldots\rangle$ the mean over $k$ with the weight $k\sigma(k)$, as in $R$;
- the photons disappear:

  $$\dot n_\gamma(\varepsilon) = -\thinspace n_\gamma(\varepsilon)\thinspace c\thinspace \sigma_T\int d\ln\gamma\thinspace \bigl(n_-(\gamma) + n_+(\gamma)\bigr)\thinspace R(\gamma\varepsilon).$$

### Neutron decay

A free neutron decays, $\mathrm{n} \to \mathrm{p}\thinspace\mathrm{e}^-\bar\nu_e$, with the mean life $\tau_n = 878.3$ s ([PDG 2026](https://pdg.lbl.gov/2026/)). A neutron of Lorentz factor $\gamma$ decays at the rate $1/(\gamma\tau_n)$. AM³ has neutrons, made in photo-hadronic interactions or injected. They interact and escape but do not decay. In am3_mojo the neutrons decay. The code is in `physics/decays.mojo` (`NeutronDecay`).

#### Choice of method

- **ATHEνA**, the only code of Table 1 with the process ([Dimitrakoudis et al. 2012](https://arxiv.org/abs/1209.0413)), turns a decaying neutron into a proton of the same Lorentz factor. It leaves out the electron and the antineutrino.
- **The photo-hadronic model AM³ uses** ([Hümmer et al. 2010](https://arxiv.org/abs/1002.1310)) adds the $\bar\nu_e$ of the neutrons that decay outside the source. It adds them as a line at the antineutrino's mean share of the neutron's energy. AM³ does not implement this.
- **The decay itself** has one mode, $\mathrm{p}\thinspace\mathrm{e}^-\bar\nu_e$ ([PDG 2026](https://pdg.lbl.gov/2026/)). Its electron has the allowed $\beta$ spectrum with the Coulomb attraction of the proton (the Fermi function). Recoil-order and radiative corrections add to this spectrum ([Hayen et al. 2018](https://arxiv.org/abs/1709.07530)).

In the neutron's rest frame, am3_mojo takes the allowed spectrum with the Fermi function. It boosts every product exactly to the energy of the neutron and follows all three products.

#### Equations

**Rest frame.** In the neutron's rest frame, the electron (total energy $E$, momentum $p$) and the antineutrino share the energy

$$E_0 = \frac{m_n^2 - m_p^2 + m_e^2}{2m_n} = 1.29258\ \mathrm{MeV},$$

which is also the largest energy the electron can have (781.6 keV kinetic). The electron follows the allowed spectrum ([Hayen et al.](https://arxiv.org/abs/1709.07530), eq. 4 without its corrections)

$$\frac{dN}{dE} \propto F(E)\thinspace p\thinspace E\thinspace (E_0 - E)^2,\qquad E_{\bar\nu} = E_0 - E,\qquad F(E) = \frac{2\pi\eta}{1 - e^{-2\pi\eta}},\quad \eta = \frac{\alpha E}{p},$$

where $F$ is the Fermi function of a point charge $Z = 1$: Hayen et al.'s eq. 6 with their $\gamma = \sqrt{1 - \alpha^2}$ (written $\gamma_1$ here) set to 1. The proton takes the remaining $m_n - E_0$, so energy is conserved exactly. The electron and antineutrino spectra leave out the proton's recoil (at most 751 eV). The masses ([CODATA 2022](https://doi.org/10.1103/RevModPhys.97.025002)) and $\alpha$ are those of [PDG 2026](https://pdg.lbl.gov/2026/).

**Lab frame.** A neutron of energy $E_n$ moves with the Lorentz factor $\gamma_n = E_n/(m_n c^2)$ and the velocity $\beta_n$. It decays at the rate $1/(\gamma_n\tau_n)$. These decays make each product $X$ at the rate

$$Q_X(E) = \int \frac{dE_n}{E_n}\thinspace \frac{N_n(E_n)}{\gamma_n\thinspace \tau_n}\thinspace F_X\negthinspace \left(\frac{E}{E_n};\beta_n\right),$$

where $N_n$ is the neutron density per unit energy, $Q_{X}$ the source of product $X$ per unit energy, and $F_X(x;\beta_n) = dn/dx$ its distribution per decay. Here $x = E/E_n$ is the lab energy fraction. A product of a given energy $E^\ast$ and momentum $p^\ast$ in the neutron's rest frame, emitted isotropically there, has $x$ spread uniformly over

$$x \in \left[\frac{E^\ast - \beta_n p^\ast}{m_n},\ \frac{E^\ast + \beta_n p^\ast}{m_n}\right]$$

with density $m_n/(2\beta_n p^\ast)$ for any $\beta_n$. This holds because $x$ is linear in the cosine of the emission angle, which isotropy makes uniform. The [kaon decays](#decays) use the limit $\beta_n \to 1$. For the electron and the antineutrino, $F_{X}$ is these uniform ranges weighted by the rest-frame spectrum, so it is not uniform itself. The proton, whose momentum is neglected, has the single value $x_p = (m_n - E_0)/m_n$.

### Kaons

am3_mojo produces $\boldsymbol{\mathbf{K}^+}$, $\boldsymbol{\mathbf{K}^-}$, $\boldsymbol{\mathbf{K}^0_\mathbf{S}}$ and $\boldsymbol{\mathbf{K}^0_\mathbf{L}}$ in photo-hadronic interactions of protons and neutrons and in $\mathrm{pp}$ collisions. The code is in the modules of the corresponding pion processes. Production is in `physics/photopion.mojo` and `physics/proton_proton.mojo`, and decays are in `physics/decays.mojo`. Synchrotron and inverse-Compton radiation are in `physics/synchrotron.mojo` and `physics/decays.mojo`. The species are listed in `model/particles.mojo`. The measured data behind the photo-hadronic tables are in `tests/support/measured/ky_angular.py` and `tests/support/measured/strange_20gev.py`.

#### Choice of method

- **ATHEνA** computes the production of $\pi^0$, $\pi^\pm$, $\mathrm{K}^\pm$, $\mathrm{K}^0_\mathrm{S}$ and $\mathrm{K}^0_\mathrm{L}$ in photo-meson interactions with the [SOPHIA](https://arxiv.org/abs/astro-ph/9903478) event generator ([Petropoulou, Giannios & Dimitrakoudis 2014](https://arxiv.org/abs/1405.2091)). In one step, a charged kaon loses its synchrotron energy and then decays. The neutral kaons decay at once.
- **The simplified photo-hadronic model AM³ uses**, [Hümmer, Rüger, Spanier & Winter (2010)](https://arxiv.org/abs/1002.1310), has only a $\mathrm{K}^+$ channel. The $\mathrm{K}^-$ and $\mathrm{K}^0$ contributions are "about a factor of two suppressed" (their Sec. 3.1, citing [Lipari et al. 2007](https://arxiv.org/abs/0704.0718)). The Winter group's follow-up work evolves the $\mathrm{K}^+$ with the steady-state kinetic equation ([Hümmer et al. 2010b](https://arxiv.org/abs/1007.0006); [Baerwald, Hümmer & Winter 2011](https://arxiv.org/abs/1009.4010)). This equation includes synchrotron and adiabatic cooling and decay.

am3_mojo extends [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s framework, which AM³ already uses for the pions, to all four kaon species, and evolves them with the kinetic equation. With synchrotron cooling and decay alone, this equation gives the energy distribution at decay of particles that cool before they decay ([Lipari, Lusignoli & Meloni 2007](https://arxiv.org/abs/0704.0718), eq. 38). ATHEνA's loss-before-decay step approximates this distribution. The equation also handles time dependence, escape and adiabatic cooling.

#### Production in photo-hadronic interactions

In [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s framework, an interaction channel $c$ has a cross section $\sigma_c(\epsilon_r)$, where $\epsilon_r$ is the photon energy in the nucleon rest frame. Its secondaries carry the fraction $\chi_c$ of the nucleon energy. Per interaction of a proton (neutron), the channel makes $M_{c,p}^K$ ($M_{c,n}^K$) kaons of species $K$. In the kaon channels below, the cross section and the energy fraction can also depend on the nucleon $N$ and the kaon species $K$: $\sigma_{c,N}$ and $\chi_{c,N}^K$. The interaction rate of a nucleon uses Hümmer et al.'s response function (their eq. 27). With $y = E_N \epsilon/(m_p c^2)$ ($m_p$ for both nucleons, as in AM³), this rate and the kaon source are

$$t_{c,N}^{-1}(E_N) = c\int dy\thinspace \frac{m_p c^2}{E_N}\thinspace n_\gamma\negthinspace \left(\frac{m_p c^2\thinspace y}{E_N}\right) f_{c,N}(y),\qquad f_{c,N}(y) = \frac{1}{2y^2}\int_0^{2y} \epsilon_r\thinspace \sigma_{c,N}(\epsilon_r)\thinspace d\epsilon_r,$$

$$Q_K(E_K) = \sum_c \sum_{N=p,n} \frac{M_{c,N}^K}{\chi_{c,N}^K}\thinspace \Big[N_N\thinspace t_{c,N}^{-1}\Big]_ {E_N = E_K/\chi_{c,N}^K}.$$

Every channel has a constant cross section between $\epsilon_\mathrm{min}$ and $\epsilon_\mathrm{max}$, so its response function is closed-form:

$$f_c(y) = \sigma_c [\min(4y^2, \epsilon_\mathrm{max}^2) - \epsilon_\mathrm{min}^2]/(4y^2)$$

for $2y \ge \epsilon_\mathrm{min}$.

**Two-body production near threshold**, $\gamma\mathrm{N} \to \mathrm{K}\Lambda, \mathrm{K}\Sigma$, in six steps of $\epsilon_r$, each with its own cross section $\sigma$ and energy fraction $\chi$ for the four channels. Six reactions make up the four channels (in parentheses in the table header). A step's cross section is the mean over the step of the measured cross sections, summed over the channel's reactions. These cross sections are angle integrals of the same differential cross sections that give $\chi$ (below). Below the table, the data and their extrapolation are described.

| $\epsilon_r$ [GeV] | $\mathrm{p} \to \mathrm{K}^+$ ($\mathrm{K}^+\Lambda$, $\mathrm{K}^+\Sigma^0$) | $\mathrm{p} \to \mathrm{K}^0$ ($\mathrm{K}^0\Sigma^+$) | $\mathrm{n} \to \mathrm{K}^+$ ($\mathrm{K}^+\Sigma^-$) | $\mathrm{n} \to \mathrm{K}^0$ ($\mathrm{K}^0\Lambda$, $\mathrm{K}^0\Sigma^0$) |
|---|---|---|---|---|
| $0.91\text{–}1.05$ | 1.31 μb, $\chi = 0.30$ | — | — | 0.81 μb, 0.34 |
| $1.05\text{–}1.25$ | 3.00 μb, 0.29 | 0.31 μb, 0.32 | 0.49 μb, 0.29 | 1.78 μb, 0.35 |
| $1.25\text{–}1.65$ | 4.09 μb, 0.29 | 0.55 μb, 0.30 | 1.28 μb, 0.30 | 1.89 μb, 0.33 |
| $1.65\text{–}2.2$ | 2.44 μb, 0.26 | 0.26 μb, 0.29 | 1.03 μb, 0.29 | 0.80 μb, 0.29 |
| $2.2\text{–}3.0$ | 1.25 μb, 0.23 | 0.083 μb, 0.30 | 0.90 μb, 0.29 | 0.28 μb, 0.27 |
| $3.0\text{–}5.0$ | 0.44 μb, 0.19 | 0.030 μb, 0.23 | 0.39 μb, 0.22 | 0.10 μb, 0.19 |

**The data.** Three experiments measured the six reactions. Each measured a different subset, up to a different photon energy:

| Reaction | SAPHIR | CLAS | A2 | Above its data, follows |
|---|---|---|---|---|
| $\gamma\mathrm{p} \to \mathrm{K}^+\Lambda$ | $0.91\text{–}2.58$ ([Glander et al. 2004](https://arxiv.org/abs/nucl-ex/0308025)) | $1.25\text{–}3.82$ ([McCracken et al. 2010](https://arxiv.org/abs/0912.4274)) | — | a power law |
| $\gamma\mathrm{p} \to \mathrm{K}^+\Sigma^0$ | $1.06\text{–}2.58$ ([Glander et al. 2004](https://arxiv.org/abs/nucl-ex/0308025)) | $1.08\text{–}3.82$ ([Dey et al. 2010](https://arxiv.org/abs/1006.0374)) | — | a power law |
| $\gamma\mathrm{p} \to \mathrm{K}^0\Sigma^+$ | $1.10\text{–}2.53$ ([Lawall et al. 2005](https://arxiv.org/abs/nucl-ex/0504014)) | — | not used (protons bound in deuterium) | $\mathrm{K}^+\Sigma^0$ ($\times 0.15$) |
| $\gamma\mathrm{n} \to \mathrm{K}^+\Sigma^-$ | — | $1.15\text{–}3.55$ ([Anefalos Pereira et al. 2010](https://arxiv.org/abs/0912.4833)) | — | $\mathrm{K}^+\Sigma^0$ ($\times 2.0$) |
| $\gamma\mathrm{n} \to \mathrm{K}^0\Lambda$ | — | $0.97\text{–}2.90$ ([Compton et al. 2017](https://arxiv.org/abs/1706.04748)) | $0.94\text{–}1.36$ ([Akondi et al. 2019](https://arxiv.org/abs/1811.05547)) | $\mathrm{K}^+\Lambda$ ($\times 0.21$) |
| $\gamma\mathrm{n} \to \mathrm{K}^0\Sigma^0$ | — | — | $1.05\text{–}1.36$ ([Akondi et al. 2019](https://arxiv.org/abs/1811.05547)) | $\mathrm{K}^0\Sigma^+$ ($\times 1.7$) |

The ranges are $\epsilon_r$ in GeV, from the centre of the first bin used to that of the last. Only bins measured over most of the angular range are used (details under [Results](tests/literature_checks/README.md#kaon-production)). The neutron data were taken on deuterium.

**Between and beyond the data.** For each reaction, the cross section and the mean angle of the kaon (which gives $\chi$, below) follow the first of these rules that applies:

1. *Within the data*, each data set is interpolated linearly between its bins. Where two experiments cover the same energy, their values are averaged.
2. *Between the threshold and the first bin*, the cross section rises linearly from zero. The mean angle is that of the first bin.
3. *Above the reaction's own data*, it follows the analogous $\gamma\mathrm{p}$ reaction of the last column, with that reaction's mean angle at the same energy. Its cross section is the analogue's times a constant factor. The factor is not chosen by hand. It is the ratio of the two reactions' measured mean cross sections over the last 0.6 GeV of the reaction's data. For example, the CLAS data of $\mathrm{K}^0\Lambda$ end at 2.9 GeV, and over $2.3\text{–}2.9$ GeV its mean cross section is 0.21 times that of $\mathrm{K}^+\Lambda$. The data of $\mathrm{K}^0\Sigma^0$ span only 0.3 GeV, so its ratio is taken over all of them. An analogue can itself be extrapolated: $\mathrm{K}^0\Sigma^0$ follows $\mathrm{K}^0\Sigma^+$, which above 2.5 GeV follows $\mathrm{K}^+\Sigma^0$.
4. *Above 3.8 GeV*, where all data end, $\mathrm{K}^+\Lambda$ and $\mathrm{K}^+\Sigma^0$ fall as power laws fitted to their last GeV of data, $\propto E^{-2.7}$ and $E^{-2.9}$. Their mean angles stay at the cross-section-weighted mean of their last 0.4 GeV. Through rule 3, the other four reactions follow them. This rule covers only 3.8 to 5 GeV, as the last step ends at 5 GeV. Above 5 GeV the code has no two-body channel.

**Why these analogues and factors.** A constant factor carries the analogue's energy dependence over to the reaction. This is reasonable only if the two are made the same way. The analogue is therefore a $\gamma\mathrm{p}$ reaction, measured to higher energy, with a hyperon of the same isospin. A $\mathrm{K}\Lambda$ pair has isospin 1/2 only, while a $\mathrm{K}\Sigma$ pair also has 3/2, so $\Delta$ resonances make $\mathrm{K}\Sigma$ but not $\mathrm{K}\Lambda$. At higher energies, exchanges in the $t$ channel take over from the resonances, and the fitted factors reflect them:

- $\mathrm{K}^+\Sigma^-$ follows $\mathrm{K}^+\Sigma^0$ with 2.0. In $\mathrm{K}^+$ exchange, isospin symmetry makes a neutron couple $\sqrt{2}$ times as strongly to $\Sigma^-\mathrm{K}^+$ as a proton to $\Sigma^0\mathrm{K}^+$. The cross section is then twice as large, and the fit gives 2.0.
- $\mathrm{K}^0\Lambda$ follows $\mathrm{K}^+\Lambda$ with 0.21, and $\mathrm{K}^0\Sigma^+$ follows $\mathrm{K}^+\Sigma^0$ with 0.15. Both factors are computed in [`tests/support/measured/ky_angular.py`](tests/support/measured/ky_angular.py) and printed, with their fit windows, by [`tests/literature_checks/kaon_production/two_body.py`](tests/literature_checks/kaon_production/two_body.py). The photon does not couple to a neutral $\mathrm{K}^0$, so these reactions have no kaon exchange in the $t$ channel. Their $\mathrm{K}^+$ analogues have it, and it dominates them at higher energies. The $\mathrm{K}^0$ reactions are therefore much smaller there.
- $\mathrm{K}^0\Sigma^0$ follows $\mathrm{K}^0\Sigma^+$ rather than $\mathrm{K}^+\Sigma^0$, because it lacks kaon exchange too. Its factor, 1.7, comes from A2's data near threshold, the only data there are. Pure $\mathrm{K}^\ast$ exchange would make it 0.5 at high energies ([Results](tests/literature_checks/README.md#kaon-production) gives the effect of this choice).

**Energy fractions from the measured angular distributions.** In the centre-of-momentum frame, a kaon has the velocity $\beta^\ast$. A kaon emitted there at the angle $\theta$ to the photon carries the fraction

$$\chi = \frac{s - m_Y^2 + m_K^2}{2s}\thinspace \big(1 - \beta^\ast \cos\theta\big),\qquad s = m_N^2 + 2m_N\epsilon_r,$$

of the nucleon energy. The formula is [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s eq. 16 with the mass of the outgoing nucleon replaced by $m_Y$ and $m_\pi$ by $m_K$. Hümmer et al. measure the angle from the nucleon. The fraction $\chi$ is linear in $\cos\theta$, so its mean over the angular distribution is $\chi(\langle\cos\theta\rangle)$. For every reaction and energy bin, $\sigma$ and $\langle\cos\theta\rangle$ come from the measured $d\sigma/d\cos\theta$ (via HEPData and the CLAS tables of [Compton et al.](https://arxiv.org/abs/1706.04748)). A step's $\chi$ is the mean over its photon energies and the reactions of the channel, weighted by their cross sections. The reactions are forward-peaked along the photon, increasingly so with energy. The forward peaking lowers $\chi$ below the isotropic value that Hümmer et al. assume ($\langle\cos\theta\rangle = 0$).

**Kaons in multi-pion production.** [Hümmer et al.](https://arxiv.org/abs/1002.1310) (Table 6) describe multi-pion production by seven interaction types, M1–M7. Each type covers one range of the photon energy $\epsilon_r$ in the nucleon rest frame, with a constant cross section and its own pion multiplicities and energy fractions: M1 $0.5\text{–}0.9$ GeV, M2 $0.9\text{–}1.5$ GeV, M3 $1.5\text{–}5$ GeV, M4 $5\text{–}50$ GeV, M5 $50\text{–}500$ GeV, M6 $500\text{–}5000$ GeV and M7 above, up to $5 \times 10^6$ GeV as in AM³. With kaons on, M2–M7 also make the kaons that come with further particles ($\gamma\mathrm{N} \to \mathrm{K}\mathrm{Y}\pi$, $\mathrm{K}\bar{\mathrm{K}}\mathrm{N}$ and the like), with the same cross sections. M1 lies below the kaon threshold, and M2 makes kaons only above 1.15 GeV, the threshold of $\gamma\mathrm{N} \to \mathrm{K}\Lambda\pi$. The table gives the kaons of each type. M3 and M4, across which the yields rise most, have two steps of $\epsilon_r$ each:

| Type, photon energy $\epsilon_r$ | Cross section of the type [μb] | Kaons per interaction from protons: $\mathrm{K}^+, \mathrm{K}^0, \mathrm{K}^-, \bar{\mathrm{K}}^0$ (from neutrons: $\mathrm{K}^+ \leftrightarrow \mathrm{K}^0$, $\mathrm{K}^- \leftrightarrow \bar{\mathrm{K}}^0$) | $\mathrm{K}^+ + \mathrm{K}^0$ cross section [μb] | Share of the kaons in the leading part, $h_+$ ($\mathrm{K}^+, \mathrm{K}^0$), $h_{-}$ ($\mathrm{K}^-, \bar{\mathrm{K}}^0$) | $\chi$ of the low part, of the leading part |
|---|---|---|---|---|---|
| M2, $1.15\text{–}1.5$ GeV | 85 | 0.0051, 0.0060, 0, 0 | 0.94 | 0.25, 0.15 | 0.24, 0.35 |
| M3, $1.5\text{–}2.5$ GeV | 120 | 0.0278, 0.0206, 0.0030, 0.0034 | 5.8 | 0.25, 0.15 | 0.24, 0.35 |
| M3, $2.5\text{–}5$ GeV | 120 | 0.0585, 0.0449, 0.0214, 0.0201 | 12.4 | 0.25, 0.15 | 0.24, 0.35 |
| M4, $5\text{–}15$ GeV | 120 | 0.0935, 0.0687, 0.0468, 0.0461 | 19.5 | 0.15, 0.09 | 0.126, 0.35 |
| M4, $15\text{–}50$ GeV | 120 | 0.1316, 0.0988, 0.0794, 0.0796 | 27.6 | 0.15, 0.09 | 0.065, 0.35 |
| M5, $50\text{–}500$ GeV | 120 | 0.185, 0.170, 0.140, 0.135 | 42.6 | 0.07, 0.04 | 0.032, 0.5 |
| M6, $500\text{–}5000$ GeV | 120 | 0.350, 0.335, 0.300, 0.295 | 82.2 | 0.05, 0.03 | 0.011, 0.5 |
| M7, $5000\text{–}5 \times 10^6$ GeV | 120 | 0.575, 0.560, 0.520, 0.515 | 136 | 0.035, 0.02 | 0.0032, 0.6 |

The $\mathrm{K}^+ + \mathrm{K}^0$ cross section of a row is the type's cross section times its $\mathrm{K}^+$ and $\mathrm{K}^0$ per interaction, without the two-body reactions. It is an inclusive cross section, so it exceeds the type's cross section where an interaction makes more than one such kaon: M7 makes 1.135 per interaction.

How the multiplicities follow from strangeness balance and measured strange-particle photoproduction is documented in `kaon_production_channels` of `src/am3_mojo/physics/photopion.mojo` and, with the data and their checks, under [Results](tests/literature_checks/README.md#kaon-production) (`multibody_strangeness.py`, `m4_steps.py`).

**Energy bookkeeping.** [Hümmer et al.](https://arxiv.org/abs/1002.1310)'s multi-pion types put the whole inelastic energy into pions (their pion energy fractions reproduce SOPHIA's). With kaons on, the pion multiplicities of type $t$ are rescaled. In each range $r$ of photon energies in which the type makes kaons (a row of the table above), the factor is

$$s_{t,r} = 1 - \frac{\sum_{c \in t,\thinspace K} \chi_c\thinspace M_{c,r}^K}{\sum_{c \in t,\thinspace \pi} \chi_c\thinspace M_c^\pi},$$

(the kaons of the row, the pions of the type), so that the energy of the interaction is kept at every photon energy. The nucleons are unchanged. The two-body channels $\gamma\mathrm{N} \to \mathrm{K}\Lambda, \mathrm{K}\Sigma$ are not in AM³'s nucleon loss and re-injection kernels, so the energy of their kaons is not taken from the nucleons. [Hümmer et al.](https://arxiv.org/abs/1002.1310) (Sec. 4.5) likewise neglect the effect of their $\mathrm{K}^+$ channel on the cooling and re-injection of the nucleons.

**Neutral kaons.** A $\mathrm{K}^0$ or $\bar{\mathrm{K}}^0$ is half $\mathrm{K}^0_\mathrm{S}$ and half $\mathrm{K}^0_\mathrm{L}$. CP violation and $\mathrm{K}^0\text{–}\bar{\mathrm{K}}^0$ interference ($\lesssim 1\text{％}$) are neglected.

#### Production in $\mathrm{pp}$ collisions

Above $E_\mathrm{trans} = 100$ GeV, AM³ makes pions in $\mathrm{pp}$ collisions with the pion energy distribution $F_\pi(x, E_p)$ ($x = E_\pi/E_p$, per inelastic collision) of [Kelner, Aharonian & Bugayov (2006)](https://arxiv.org/abs/astro-ph/0606058). It uses their eq. 12 (SIBYLL) or eq. 6 (QGSJET). Below this energy, it uses a $\delta$-functional approximation. AM³ takes the inelastic cross section from [Kafexhiu et al. (2014)](https://arxiv.org/abs/1406.7369), eq. 1. Kelner et al. give no kaon distributions. Kafexhiu et al. parametrise only $\pi^0$ production and its $\gamma$-rays. The other parametrisations of $\mathrm{pp}$ secondaries ([Kamae et al. 2006](https://arxiv.org/abs/astro-ph/0605581), [AAfrag](https://arxiv.org/abs/1904.05129)) give only the spectra of stable secondaries, not those of the kaons. These stable secondaries are photons, $\mathrm{e}^\pm$, neutrinos and, in AAfrag, nucleons and antinucleons. The fits of measured spectra are limited to $E_p \lesssim 2 \times 10^3$ GeV ([Koers et al.](https://arxiv.org/abs/hep-ph/0611219), Sec. I). [Badhwar et al. 1977](https://doi.org/10.1103/PhysRevD.15.820) fit $\pi^\pm$ and $\mathrm{K}^\pm$. Theirs is the only fit with kaons. [Stephens & Badhwar 1981](https://doi.org/10.1007/BF00651256) fit $\pi^0$. [Blattnig et al. 2000](https://arxiv.org/abs/hep-ph/0010170) also fit $\pi^0$ and use Badhwar et al.'s $\pi^\pm$.

am3_mojo uses the kaon spectra of [Koers, Pe'er & Wijers (2006, arXiv:hep-ph/0611219)](https://arxiv.org/abs/hep-ph/0611219). They are fits to PYTHIA 6.324 for $\mathrm{K}^+$, $\mathrm{K}^-$, $\mathrm{K}^0$ and $\bar{\mathrm{K}}^0$ separately, per inelastic collision including diffraction. In these fits, the pion and kaon decays are switched off, so the kaons are counted before they decay. AM³'s pions are [Kelner et al.](https://arxiv.org/abs/astro-ph/0606058)'s. Kelner et al. do not state whether their $F_\pi$ includes pions from kaon decays. They note that their leptons from all channels exceed those from their pion distribution (by about $10\text{％}$ at $x \sim 0.1$), mainly because of kaon decays. At least the kaons' leptonic decays are therefore not in $F_\pi$. am3_mojo assumes that no kaon decays are, so that nothing is counted twice. The kaon source is built like AM³'s pion source,

$$Q_K(E_K) = c\thinspace n_H\int \frac{dE_p}{E_p}\thinspace \sigma_{\rm inel}(E_p)\thinspace N_p(E_p)\thinspace F_K\negthinspace \left(\frac{E_K}{E_p},E_p\right),$$

with $\sigma_\mathrm{inel}$ the inelastic $\mathrm{pp}$ cross section of [Kafexhiu et al.](https://arxiv.org/abs/1406.7369) (their eq. 1, in AM³'s version for the charged pions, whose threshold uses the $\pi^\pm$ mass), the proton density per unit energy $N_p$ and the kaon source per unit energy $Q_{K}$. The source is built in three ranges of the proton energy:

**100 GeV $\le E_p \le 10^6$ GeV: [Koers et al. (2006)](https://arxiv.org/abs/hep-ph/0611219), eqs. 16–18 and their Table III.** With energies in GeV and $\log = \log_{10}$, $F_K(x, E_p) = E_p n(xE_p)$, where

$$n(\epsilon) = n_0\thinspace \epsilon^{-1}\thinspace \alpha(\epsilon)\thinspace \beta(\epsilon)\thinspace \gamma_1(\epsilon)\thinspace \gamma_2(\epsilon),\qquad n_0 = 1.21\times 10^{\thinspace p_0 + p_1 p_2^2},\qquad \alpha = \epsilon^{\thinspace p_1(\log\epsilon - 2p_2)},$$

$$\beta = 10^{-(\epsilon/\epsilon_3)^{p_{30}}}\thinspace 10^{-(\epsilon/\epsilon_5)^{p_{50}}},\qquad \gamma_1 = \tanh\negthinspace \big(p_{70}\log(\epsilon/m_K)\big),\qquad \gamma_2 = \tanh\negthinspace \big(p_{80}\log(E_p/\epsilon)\big),$$

with $\epsilon_3 = 10^{p_2+p_4}$, $\epsilon_5 = 10^{p_2+p_6}$, $p_0 = p_{00} + p_{01} \log E_p$, $p_1 = p_{10} + p_{11} \log E_p$, $p_2 = p_{20} + p_{21} \log E_p$, $p_4 = p_{40} - p_2$, $p_6 = p_{60} + p_2$, and $n = 0$ outside $m_K \lt \epsilon \lt E_p$. Here $m_K$ stands for the mass of the meson: $m_{\mathrm{K}^0}$ for $\mathrm{K}^0$ and $\bar{\mathrm{K}}^0$, and $m_{\pi^0}$ for the $\pi^0$ row. The parameters are:

| | $p_{00}$ | $p_{01}$ | $p_{10}$ | $p_{11}$ | $p_{20}$ | $p_{21}$ | $p_{30}$ | $p_{40}$ | $p_{50}$ | $p_{60}$ | $p_{70}$ | $p_{80}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $\mathrm{K}^+$ | −1.23 | 0.0657 | −0.147 | 0.0161 | −0.00411 | 0.493 | −0.989 | −0.345 | 0.777 | −0.235 | 42.7 | 12.0 |
| $\mathrm{K}^-$ | −1.46 | 0.101 | −0.109 | 0.00865 | −0.0577 | 0.491 | −1.22 | −0.164 | 1.04 | −0.279 | 18.6 | 4.23 |
| $\mathrm{K}^0$ | −1.29 | 0.0690 | −0.142 | 0.0154 | −0.00717 | 0.493 | −1.03 | −0.294 | 0.839 | −0.272 | 33.8 | 10.1 |
| $\bar{\mathrm{K}}^0$ | −1.50 | 0.101 | −0.118 | 0.0101 | −0.0567 | 0.491 | −1.25 | −0.169 | 1.05 | −0.272 | 21.2 | 4.07 |
| $\pi^0$ | −0.456 | 0.0950 | −0.142 | 0.0135 | −0.457 | 0.494 | −1.49 | −0.981 | 1.01 | 0.723 | 22.1 | 8.53 |

(The $\pi^0$ row is used only by the extension above $10^6$ GeV.) As in photo-hadronic interactions, a $\mathrm{K}^0$ or $\bar{\mathrm{K}}^0$ is half $\mathrm{K}^0_\mathrm{S}$ and half $\mathrm{K}^0_\mathrm{L}$: $F_{\mathrm{K}^0_\mathrm{S}}$ $= F_{\mathrm{K}^0_\mathrm{L}}$ $= \tfrac12(F_{\mathrm{K}^0} + F_{\bar{\mathrm{K}}^0})$.

$\boldsymbol{E_p \gt 10^6}$ **GeV (am3_mojo extension, not from a publication).** [Koers et al.](https://arxiv.org/abs/hep-ph/0611219)'s fit cannot be taken far beyond its range. Its parameters drift so that the energy fraction of its pions reaches 1 at about $2.5 \times 10^9$ GeV. Within its range, however, the $\mathrm{K}^+/\pi^0$ and $\mathrm{K}^0/\pi^0$ ratios at fixed $x \gtrsim 0.03$ are nearly independent of $E_p$ (Feynman scaling of the ratio). am3_mojo keeps this ratio, taken at $E^\ast = 10^6$ GeV, on top of the pion distribution that AM³ uses:

$$F_K(x,E_p) = R(x)\thinspace \hat F_\pi(x,E_p),\qquad R(x) = \frac{E^\ast \thinspace n_K(x_s E^\ast;E^\ast)}{\hat F_\pi(x_s,E^\ast)},\qquad x_s = \max(x, 10^{-4}),$$

where $\hat F_\pi$ is [Kelner et al.](https://arxiv.org/abs/astro-ph/0606058)'s $F_\pi$ (their eq. 12, from SIBYLL, AM³'s default, or eq. 6, from QGSJET) without its pion threshold factor $(1 - m_\pi/(xE_p))^{1/2}$. For $x \lt 10^{-4}$ (kaons below 100 GeV at $E^\ast$), the kaon mass shapes the ratio. There, $R$ follows Koers et al.'s $\mathrm{K}/\pi^0$ ratio $\rho(\epsilon) = n_K(\epsilon)/n_{\pi^0}(\epsilon)$ at $E^\ast$ as a function of the kaon energy, with $n_K$ and $n_{\pi^0}$ from their eqs. 16–18 and Table III (the rows of the kaon and of the $\pi^0$): $R(x)$ $\to R(10^{-4})$ $\rho(\min(xE_p, 10^{-4}E^\ast))/\rho(10^{-4}E^\ast)$.

$\boldsymbol{E_p \lt E_t = 100}$ **GeV (am3_mojo extension, not from a publication).** Like AM³'s pions ([Kelner et al. 2006](https://arxiv.org/abs/astro-ph/0606058), eqs. 75–77), the kaons use a $\delta$-functional approximation. Each collision makes $\tilde n_K$ kaons of energy $E_K = \max(K_K T_p, \gamma_\mathrm{cm} m_K)$ ($T_p = E_p - m_p$; $\gamma_\mathrm{cm} = \sqrt{(E_p + m_p)/(2m_p)}$ $= (E_p + m_p)/\sqrt{s}$ is the Lorentz factor of the centre-of-momentum frame). Hence $\gamma_\mathrm{cm} m_K$ is the lab energy of a kaon at rest in that frame, the lowest mean kaon energy. The factor $K_{K}$ is the mean fraction of $T_p$ per kaon of the [Koers et al.](https://arxiv.org/abs/hep-ph/0611219) spectra at $E_t$. The multiplicity follows a fit to the kaon multiplicities of the NA49 compilation ([Anticic et al. 2010](https://arxiv.org/abs/1004.1889), Table 11), normalised to Koers et al. at $E_t$:

$$\tilde n_K(E_p) = \langle n_K\rangle_{\rm KPW}(E_t)\thinspace \frac{f(s)}{f(s_t)},\qquad f(s) = A\left(1-\frac{s_{\rm th}}{s}\right)^{a}\left(\frac{s}{s_{\rm th}}\right)^{b},\qquad s = 2m_p^2 + 2m_pE_p,$$

where $\langle n_K\rangle_{\rm KPW}(E_t)$ is the number of kaons per inelastic collision in the spectra of [Koers, Pe'er & Wijers](https://arxiv.org/abs/hep-ph/0611219) (KPW) at $E_t$, the integral of their $n(\epsilon)$. It makes the multiplicity continuous at $E_t$. The fit is a least-squares fit in $\ln n$ to the points of the compilation of [Anticic et al.](https://arxiv.org/abs/1004.1889) with $\sqrt{s} \le 63$ GeV. The threshold is that of the lowest-threshold channel that makes the kaon:

| Kaon | $A$ | $a$ | $b$ | $\sqrt{s_\mathrm{th}}$ [GeV] | Threshold channel |
|---|---|---|---|---|---|
| $\mathrm{K}^+$ | 0.0942 | 3.435 | 0.2574 | 2.548 | $\mathrm{pp} \to \mathrm{p}\Lambda\mathrm{K}^+$ |
| $\mathrm{K}^-$ | 0.0476 | 6.175 | 0.3324 | 2.864 | $\mathrm{pp} \to \mathrm{ppK}^+\mathrm{K}^-$ |
| $\mathrm{K}^0_\mathrm{S}$, $\mathrm{K}^0_\mathrm{L}$ (fitted to $\langle \mathrm{K}^0_\mathrm{S}\rangle$) | 0.02128 | 2.115 | 0.5398 | 2.625 | $\mathrm{pp} \to \mathrm{p}\Sigma^+\mathrm{K}^0$ |

**Energy bookkeeping.** The kaons are made in addition to AM³'s pions (see above), and the protons lose the kaons' energy too. AM³'s $\mathrm{pp}$ energy-loss rate $\dot E/E = -n_H c \sigma_\mathrm{inel} \sum_\pi \kappa_\pi$ gains the term $-n_H c \sigma_\mathrm{inel} \sum_K \kappa_K$, with $\kappa_K$ the fraction of $E_p$ the kaons carry.

#### Evolution

Every species $s$ is evolved with AM³'s solver,

$$\frac{\partial N_s}{\partial t} = -\frac{\partial}{\partial E}\Big(\dot E\thinspace N_s\Big) - \frac{N_s}{t_{\rm esc}} - \frac{N_s}{\gamma\thinspace \tau_s} + Q_s,\qquad \gamma = \frac{E}{m_s c^2},$$

with the PDG 2026 masses and mean lives ([Particle Data Group, Takahashi et al. 2026, Int. J. Mod. Phys. A 41, 2630011](https://pdg.lbl.gov/2026/)): $m_{\mathrm{K}^\pm} = 493.677$ MeV, $\tau = 1.2380 \times 10^{-8}$ s; $m_{\mathrm{K}^0} = 497.611$ MeV, $\tau_\mathrm{S} = 8.954 \times 10^{-11}$ s, $\tau_\mathrm{L} = 5.116 \times 10^{-8}$ s (the PDG fit values; for $\tau_\mathrm{S}$ the fit assuming CPT). The terms follow AM³'s treatment of the other hadrons:

| | $\mathrm{K}^+$, $\mathrm{K}^-$ (as AM³'s charged pions) | $\mathrm{K}^0_\mathrm{S}$, $\mathrm{K}^0_\mathrm{L}$ (as AM³'s neutrons) |
|---|---|---|
| synchrotron cooling and emission | ✓ | — (neutral) |
| inverse-Compton cooling and emission | ✓ | — (neutral) |
| quantum-synchrotron cooling | ✓ (when on, as for the pions) | — |
| adiabatic cooling | ✓ | — (not tied to the magnetised plasma) |
| escape | the pions' escape rate | the neutrons' escape rate |
| expansion | ✓ | ✓ |
| extra cooling | ✓ | ✓ |
| decay | ✓ | ✓ |

**Synchrotron:** cooling $-\dot E$ $= (4/3)\ \sigma_T c U_B (m_e/m_K)^2 \gamma^2$, and AM³'s hadron synchrotron emission kernel with the kaon mass, for $\mathrm{K}^+$ and $\mathrm{K}^-$, as a separate photon component.

**Inverse Compton:** AM³'s treatment of charged pions with the kaon mass. The cooling on the target photons is continuous (in the Thomson regime $-\dot E$ $= (4/3)\ \sigma_T c U_\mathrm{rad} (m_e/m_K)^2 \gamma^2$, with AM³'s Klein–Nishina treatment above). The emission uses AM³'s step approximation, as a separate photon component.

#### Decays

A kaon of energy $E_{K}$ decays at the rate $1/(\gamma\tau_s)$ into products $X$ with the distribution $F_{s\to X}(x) = dn/dx$ in the energy fraction $x = E/E_K$, which gives the source

$$Q_X(E) = \sum_s \int_E^\infty \frac{dE_K}{E_K}\thinspace \frac{N_s(E_K)}{\gamma_K\thinspace \tau_s}\thinspace F_{s\to X}\negthinspace \left(\frac{E}{E_K}\right).$$

**Modes.** Every non-radiative mode with a branching ratio above $0.1\text{％}$ in the [PDG 2026](https://pdg.lbl.gov/2026/) listings is followed, and so are the smaller $\mathrm{K}^0_\mathrm{S}$ semileptonic and $\mathrm{K}^0_\mathrm{L} \to \pi^0\pi^0$ modes. Together they cover $99.995\text{％}$ of $\mathrm{K}^\pm$ decays, all of $\mathrm{K}^0_\mathrm{S}$ and $99.93\text{％}$ of $\mathrm{K}^0_\mathrm{L}$. PDG counts most of each radiative mode above $0.1\text{％}$, its low-energy-photon part, in the parent mode. The branching ratios are renormalised to their sum. $\mathrm{K}^-$ decays are the CP conjugates of $\mathrm{K}^+$ decays (a left-handed $\mu^+$ becomes a right-handed $\mu^-$).

| Kaon | Mode | Branching ratio |
|---|---|---|
| $\mathrm{K}^+$ | $\mu^+\nu_\mu$ | 0.6356 |
| | $\pi^+\pi^0$ | 0.2067 |
| | $\pi^+\pi^+\pi^-$ | 0.05583 |
| | $\pi^0\mathrm{e}^+\nu_e$ ($\mathrm{K}^+_ {e3}$) | 0.0507 |
| | $\pi^0\mu^+\nu_\mu$ ($\mathrm{K}^+_ {\mu 3}$) | 0.03352 |
| | $\pi^+\pi^0\pi^0$ | 0.01760 |
| $\mathrm{K}^0_\mathrm{S}$ | $\pi^+\pi^-$ | 0.6920 |
| | $\pi^0\pi^0$ | 0.3069 |
| | $\pi^\mp\mathrm{e}^\pm\nu$ ($\mathrm{K}_ {e3}$) | $7.14 \times 10^{-4}$ |
| | $\pi^\mp\mu^\pm\nu$ ($\mathrm{K}_ {\mu 3}$) | $4.56 \times 10^{-4}$ |
| $\mathrm{K}^0_\mathrm{L}$ | $\pi^\mp\mathrm{e}^\pm\nu$ ($\mathrm{K}_ {e3}$) | 0.4055 |
| | $\pi^\mp\mu^\pm\nu$ ($\mathrm{K}_ {\mu 3}$) | 0.2704 |
| | $3\pi^0$ | 0.1952 |
| | $\pi^+\pi^-\pi^0$ | 0.1254 |
| | $\pi^+\pi^-$ | $1.967 \times 10^{-3}$ |
| | $\pi^0\pi^0$ | $8.64 \times 10^{-4}$ |

The $\mathrm{K}^0_\mathrm{L}$ semileptonic decays give $\pi^-\ell^+\nu$ with probability $\left(1 + A_{L}\right)/2$ and $\pi^+\ell^-\bar\nu$ with $\left(1 - A_{L}\right)/2$. The charge asymmetries are $A_L = 3.34 \times 10^{-3}$ ($\mathrm{e}$) and $3.04 \times 10^{-3}$ ($\mu$). The $\mathrm{K}^0_\mathrm{S}$ semileptonic decays are split evenly.

**From the kaon frame to the lab.** Consider an ultra-relativistic kaon of mass $M$ that decays isotropically. A product of mass $m$ with energy $E^\ast$ and momentum $p^\ast$ in the kaon frame is spread uniformly over $x \in [(E^\ast - p^\ast)/M, (E^\ast + p^\ast)/M]$ with density $M/(2p^\ast)$. With $g(E^\ast)$ its normalised energy distribution in the kaon frame,

$$F(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}^{E_{\max}} \frac{g(E^\ast)}{p^\ast}\thinspace dE^\ast,\qquad E_{\rm lo}(x) = \frac{M^2x^2 + m^2}{2Mx}\ \ (\ge m).$$

Two-body decays have one $E^\ast = (M^2 + m^2 - m_b^2)/(2M)$, so $F$ is flat. For a three-body decay with Dalitz-plot density $\rho(E_1, E_2, E_3)$ (phase space $\times \vert\mathcal M\vert^2$), $g$ is the integral of $\rho$ along the line $E_i = E^\ast$ of the Dalitz plot.

**Muon helicity.** A muon with longitudinal polarisation $\mathcal P$ in the kaon frame has the lab helicity $\mathcal P\thinspace \tilde h(x, E^\ast)$, with $\tilde h = (E^\ast M x - m^2)/(p^\ast M x)$ after averaging over the orientation of the decay plane. The two lab helicities therefore have

$$F_{R,L}(x) = \frac{M}{2}\int_{E_{\rm lo}(x)}^{E_{\max}} \frac{g(E^\ast) \pm G(E^\ast)\thinspace \tilde h(x,E^\ast)}{2\thinspace p^\ast}\thinspace dE^\ast,$$

where $G$ is the rate-weighted polarisation ($\rho\mathcal P$ integrated along the Dalitz line). For $\mathrm{K} \to \mu\nu$, $\mathcal P = -1$ for the $\mu^+$, which is left-handed in the kaon frame. The same holds for pions ([Lipari, Lusignoli & Meloni 2007](https://arxiv.org/abs/0704.0718): helicity $h = -1$ at $x = 1$, eq. 16; the spectra of the two helicities, eqs. 18–19). The helicity matters because a muon's decay spectrum depends on it.

$\mathbf{K}_ {\boldsymbol{l3}}$ **decays** ($\mathrm{K} \to \pi\ell\nu$) use the V−A matrix element with the hadronic current $H = f_+(t)(p_K + p_\pi) + f_-(t)(p_K - p_\pi)$, where $t = (p_K - p_\pi)^2$. For an $\ell^+$ with spin 4-vector $s$ ($s$ along its momentum in the kaon frame for helicity $+1/2$, $-s$ for $-1/2$),

$$\vert \mathcal M\vert^2 \propto W_0 + W_{\rm pol}(s),\qquad W_0 = 2\thinspace (p_\nu\negthinspace \cdot\negthinspace H)(p_\ell\negthinspace \cdot\negthinspace H) - H^2\thinspace (p_\nu\negthinspace \cdot\negthinspace p_\ell),\qquad W_{\rm pol}(s) = m_\ell\left[2\thinspace (p_\nu\negthinspace \cdot\negthinspace H)(s\negthinspace \cdot\negthinspace H) - H^2\thinspace (p_\nu\negthinspace \cdot\negthinspace s)\right],$$

and, summed over its spin, $\vert\mathcal M\vert^2 \propto 2W_0$. $W_0$ is the classic density of [Chounet, Gaillard & Gaillard](https://doi.org/10.1016/0370-1573%2872%2990018-X) (1972, Phys. Rep. 4, 199, their eq. 6.9). $W_\mathrm{pol}$ gives the muon polarisation. The form factors are those of the [PDG 2026](https://pdg.lbl.gov/2026/) fits, with $f_- = (f_0 - f_+)(M^2 - m_\pi^2)/t$:

$$f_+(t) = 1 + \lambda'_ +\frac{t}{m_{\pi^+}^2} + \frac{\lambda''_ +}{2}\left(\frac{t}{m_{\pi^+}^2}\right)^2,\qquad f_0(t) = 1 + \lambda_0\frac{t}{m_{\pi^+}^2},$$

from PDG's fits assuming $\mu\text{–}\mathrm{e}$ universality. The $\mathrm{K}^0$ values are $\lambda'_ + = 0.0240$, $\lambda''_ + = 0.0020$ and $\lambda_0 = 0.0116$ (quadratic fit). The $\mathrm{K}^\pm$ values are $\lambda'_ + = 0.02959$, $\lambda''_ + = 0$ and $\lambda_0 = 0.0176$. PDG has no $\mathrm{K}^\pm$ quadratic fit with universality, so its linear fit is used.

$\boldsymbol{\mathbf{K} \to 3\pi}$ uses the [PDG](https://pdg.lbl.gov/2026/) Dalitz-plot expansion

$$\vert \mathcal M\vert^2 \propto 1 + g\thinspace u + h\thinspace u^2 + k\thinspace v^2,\qquad u = \frac{s_3 - s_0}{m_{\pi^+}^2},\quad v = \frac{s_2 - s_1}{m_{\pi^+}^2},\quad s_i = (p_K - p_i)^2,\quad s_0 = \tfrac13\textstyle\sum_i s_i,$$

where pion 3 is the odd one. The coefficients are $(g, h, k)$ $= (-0.21134, 0.01848, -0.00463)$ for $\mathrm{K}^+ \to \pi^+\pi^+\pi^-$, $(0.626, 0.052, 0.0054)$ for $\mathrm{K}^+ \to \pi^+\pi^0\pi^0$ and $(0.678, 0.076, 0.0099)$ for $\mathrm{K}^0_\mathrm{L} \to \pi^+\pi^-\pi^0$. For this mode, the CP-violating terms $j\thinspace v + f\thinspace u\thinspace v$ are consistent with zero and left out. $\mathrm{K}^0_\mathrm{L} \to 3\pi^0$ has $1 + h(u^2 + v^2/3)$, PDG's expansion made symmetric in the three $\pi^0$, with $h = 0.00059$.

### Checks against the literature

[`tests/literature_checks/derivations/`](tests/literature_checks/README.md#derivation-checks) re-derives with sympy every equation of this README. An equation that a source gives, or that follows from a source or from kinematics, is derived again. For a modelling choice, the check derives what follows from it, such as a conservation law. Each check reads the equation as written here and the source's form from the paper's TeX source or a transcription of its scan. It also computes the numbers the READMEs quote from these equations, and checks the code's inputs and parameter tables against this README. The misprints it found in the sources, which this README corrects, are listed there. Those of the [AM³ paper](https://arxiv.org/abs/2312.13371) and of [Gao et al.](https://arxiv.org/abs/1610.05306) are also under [Differences](#differences-from-the-c-original).

The other scripts in [`tests/literature_checks/`](tests/literature_checks/README.md) check what a derivation cannot, outside the test suite. They compare the code's tables and kernels with Monte Carlo simulations. They also compare the tables with the measured data and their HEPData records. Their [results](tests/literature_checks/README.md#results), including the size of the approximations listed above, are in the folder's README.

### Effect on a simulation

`results/plots/added_processes_leptohadronic_all.png` (`make plots`) compares am3_mojo with and without the added processes for the `leptohadronic_all` scenario. Its dashed curves are components of the default run: those of the added processes and, for the neutrinos, their parents.

![Muon neutrino, electron neutrino, photon and e± spectra of leptohadronic_all with and without e+e- annihilation, triplet pair production, neutron decay and kaons](results/plots/added_processes_leptohadronic_all.png)

**Neutrinos.** Kaon decays extend the $\nu_\mu$ and $\nu_e$ spectra beyond AM³'s cutoff. At the highest energies, pions and muons lose their energy to synchrotron radiation before they decay. The heavier, shorter-lived charged kaons decay first, and the neutral kaons do not radiate at all. Above $10^{18}$ eV, kaon decays therefore make nearly all the neutrinos, and the spectra are up to $2 \times 10^5$ times higher. The total neutrino energy rises by only $1.2\text{％}$, because the kaons take their energy from the pions of multi-pion production. This is the known kaon effect on high-energy neutrino spectra ([Baerwald, Hümmer & Winter 2011](https://arxiv.org/abs/1009.4010); for the $\nu_e$ from $\mathrm{K}^0$ decays, [Petropoulou et al. 2014](https://arxiv.org/abs/1405.2091)).

**Other fluxes.** The effect of the added processes on the other fluxes is small. Below $10^{17}$ eV, the photon and $\mathrm{e}^\pm$ spectra change by at most $1.6\text{％}$ in any bin. Near their cutoffs they are up to $7.5\text{％}$ (photons) and $1.4\text{％}$ ($\mathrm{e}^\pm$) lower, as the kaons take energy from the pions of multi-pion production. Triplet pair production and neutron decay change every total spectrum by less than $10^{-4}$.

## Tests, reference comparison and benchmarks

Everything runs through `make`. To run all the tests and all the checks:

```bash
make reference     # once: download and build the C++ AM³ for the reference tests
make test          # all tests
make checks        # the literature and solver checks, outputs in build/checks/
```

All the targets:

| Command | What it does | Needs | Time (on Apple M5) |
|---|---|---|---|
| `make test-unit` | tests of am3_mojo on its own (167) | `make build` | 3.5 min |
| `make reference` | downloads and builds the C++ AM³, original and patched (see below) | internet, C++ compiler | $1\text{–}2$ min (once) |
| `make test-reference` | compares am3_mojo with the C++ AM³ (20) | `make reference` | 2 min |
| `make test` | all tests (187); reference tests are skipped if the reference is not built | | 5.5 min |
| `make test-asserts` | the unit tests on a build with Mojo's bounds checks and assertions (`build-asserts/`) | | 4 min (+ the compile) |
| `make benchmark` | timings, memory and per-stage profiles vs the C++ AM³ | `make reference` | 5.5 min |
| `make plots` | comparison, kernel-reuse and added-process figures | | 0.5 min |
| `make checks` | every script of `tests/literature_checks/` and `tests/solver_checks/` (`tests/run_checks.py`), with the arguments of this README's figures; outputs in `build/checks/` | the C++ AM³ of `make reference` for two of them (skipped without it) | 5 min |
| `make all` | build + reference + test + benchmark + plots + checks | | 16.5 min (+ reference build) |

Results are written to `results/`. The tables and figures are part of the repository. The results of the last full run (`make all` on an Apple M5) can therefore be seen without running anything:
- `reference/summary.md`: the deviation table;
- `benchmark/benchmark.md` and `benchmark/benchmark.json`: step and set-up times, and memory;
- `benchmark/stages_<scenario>.md`: per-stage profiles;
- `plots/*.png`: the figures shown in this README.

The raw simulation outputs of the tests are not part of the repository. The tests write them to `results/runs/` and `results/reuse/`: about 32 MB of `.npz` files, rewritten by every test run, plus 7 MB of reuse variants that `make plots` adds. `make test-asserts` writes its own to `results/runs-build-asserts/` and `results/reuse-build-asserts/`. `make plots` draws the figures from these outputs. It runs whatever is missing or older than the module that made it (a rebuild invalidates them), so the figures always show the current build.

pytest can also be run directly, for example `.venv/bin/python -m pytest tests/unit/test_api.py -v` or `-m "not reference"`. Each test file's docstring says what it checks.

<a id="reference-comparison"></a>
### Comparison with the original C++ AM³ (`tests/reference`)

`make reference` (`tests/reference/setup_reference.sh`) does the following:
1. clones AM³ from `gitlab.desy.de/am3/am3` at commit `04d63d9`: the tag v1.2.0 plus 7 commits that change only its documentation and examples (its package metadata and `am3.__version__` still say 1.0.1);
2. downloads and builds GSL 2.8, Eigen 3.4.0 and, on macOS, LLVM libomp 19.1.7 into `tests/reference/work/deps`;
3. compiles two serial builds of AM³'s Python module:
   - the original;
   - the original with the fixes of the AM³ bugs that am3_mojo fixes (`am3_fixes.patch`: 40 hunks in 6 source files, each marked with a comment; see [`tests/reference/README.md`](tests/reference/README.md)). It is rebuilt whenever the patch changes.

It can be re-run safely, since completed steps are skipped. `AM3_REFERENCE_DIR` changes the work directory, and `AM3_SOURCE_DIR` uses an existing AM³ checkout instead of cloning.

The reference tests run am3_mojo with the [processes it adds](#added-processes) switched off (`mojo-am3`), so that both codes have the same physics. The simulations are the four scenarios of [`tests/support/scenarios.py`](tests/support/scenarios.py), which drive the public API only. They are compared with the patched build, so that both codes solve the same equations. The benchmark uses the patched build too, so that both codes do the same work. `test_api_defaults.py` uses the original build. Only getters that both codes have are compared.

**Criterion** (`tests/support/compare.py`):
- **Fluxes:** particle densities, component spectra and injection rates must agree to $10^{-2}$ relative. The check covers every energy bin above $10^{-12}$ of the array's peak in particle number ($E\thinspace dN/dE$) or in energy ($E^2\thinspace dN/dE$, the quantity flux plots show). On a flux plot, which spans many orders of magnitude, a $1\text{％}$ difference is below the line width. The bins below both floors are not visible on it.
- **Grids:** these must be identical.
- **Timescales:** these are reported only, on the bins where the rate $1/{t}$ is above $10^{-12}$ of its largest. The longest timescales, up to those that stand for an absent process, carry no information.
- Every array must exist in both codes with the same shape (except the two neutrino timescales described in [`tests/reference/README.md`](tests/reference/README.md), compared on AM³'s bins). A flux that is not finite in a bin the criterion checks counts as a failure.

Current results:

| Scenario | Flux arrays | Worst flux deviation | Worst timescale deviation |
|---|---|---|---|
| blazar_simple | 768 | 1.1e-07 | 1.2e-07 |
| exotic | 384 | 1.9e-03 | 7.7e-04 |
| leptohadronic_all | 576 | 1.0e-03 | 2.3e-04 |
| ssc_external | 384 | 4.8e-08 | 1.7e-07 |

`make plots` shows the comparison as figures. `deviation_vs_time.png` plots the worst deviation over all fluxes at every checkpoint of the four scenarios. The dashed red line is the $10^{-2}$ criterion:

![Worst deviation between am3_mojo and the C++ AM³ at every checkpoint of the four scenarios](results/plots/deviation_vs_time.png)

`spectra_<scenario>.png` overlays the spectra of both codes at the first and the last checkpoint (top). It also shows the deviation of every energy bin the criterion checks at the last checkpoint (bottom). These bins cover the whole plotted range. The C++ spectra are drawn as thick grey lines, and the am3_mojo spectra as dashed blue lines on top of them. The AM³ example model:

![Photon, e±, proton and neutrino spectra of blazar_simple from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_blazar_simple.png)

Every process on:

![Photon, e±, proton and neutrino spectra of leptohadronic_all from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_leptohadronic_all.png)

A leptonic synchrotron self-Compton model with an external photon field:

![Photon and e± spectra of ssc_external (no hadrons) from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_ssc_external.png)

Exotic:

![Photon, e±, proton and neutrino spectra of exotic from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_exotic.png)

## Performance: am3_mojo (4 threads) vs the C++ original (serial)

`make benchmark` runs every scenario three times, in one fresh process per code and scenario. It reports the fastest run and the peak memory of one simulation. The benchmarks ran on an Apple M5 (4 performance and 6 efficiency cores, 16 GB). The C++ AM³ is the serial (single-threaded) Release build made by `make reference` (clang, `-O3`). It includes the fixes of `am3_fixes.patch`, so both codes solve the same equations.

**Time per evolution step** (lower is better). Both codes compute the same physics here: am3_mojo runs with the [processes it adds](#added-processes) switched off.

| Scenario | C++ AM³ (1 thread) | am3_mojo (1 thread) | am3_mojo (4 threads) | Speed-up over C++ (1 thread / 4 threads) |
|---|---|---|---|---|
| `blazar_simple` | 11.63 ms | 2.84 ms | **1.95 ms** | $4.1\times$ / $\boldsymbol{6.0\times}$ |
| `leptohadronic_all` | 17.03 ms | 4.56 ms | **3.00 ms** | $3.7\times$ / $\boldsymbol{5.7\times}$ |
| `ssc_external` | 3.04 ms | 1.02 ms | **0.52 ms** | $3.0\times$ / $\boldsymbol{5.8\times}$ |
| `exotic` | 11.67 ms | 4.23 ms | **2.86 ms** | $2.8\times$ / $\boldsymbol{4.1\times}$ |

**Cost of the added processes** (am3_mojo on 4 threads, time per evolution step):

| Scenario | Without them | With them (default) | Extra time |
|---|---|---|---|
| `blazar_simple` | 1.95 ms | 3.33 ms | $+71\text{％}$ |
| `leptohadronic_all` | 3.00 ms | 5.06 ms | $+69\text{％}$ |
| `ssc_external` | 0.52 ms | 0.69 ms | $+31\text{％}$ (triplet pair production only: merged $\mathrm{e}^\pm$, so no annihilation; no hadrons, so no neutron decay and no kaons) |
| `exotic` | 2.86 ms | 3.62 ms | $+27\text{％}$ |

**Kernel set-up and peak memory** (am3_mojo on 4 threads; the peak memory varies by up to about 10 MB from run to run):

| Scenario | Kernel set-up (`init_kernels`), C++ → am3_mojo | Peak memory, C++ → am3_mojo | Added processes: set-up, memory |
|---|---|---|---|
| `blazar_simple` | 6.91 s → **1.94 s** ($3.6\times$) | 2.88 GB → **0.25 GB** ($11\times$) | $+0.18$ s, $+28$ MB |
| `leptohadronic_all` | 11.07 s → **2.98 s** ($3.7\times$) | 2.88 GB → **0.24 GB** ($12\times$) | $+0.20$ s, $+42$ MB |
| `ssc_external` | 0.14 s → 0.17 s ($0.85\times$) | 2.07 GB → **0.094 GB** ($22\times$) | $+0.02$ s, $+2$ MB |
| `exotic` | 1.49 s → **0.46 s** ($3.3\times$) | 1.00 GB → **0.35 GB** ($2.8\times$) | $+0.08$ s, $+10$ MB |

Where the extra time of the added processes goes, from the profiler:
- $\boldsymbol{\mathbf{e}^+\mathbf{e}^-}$ **annihilation:** its photon source reads its large photon kernel once per step, and the $\mathrm{e}^\pm$ components are evolved as separate $\mathrm{e}^-$ and $\mathrm{e}^+$ halves, which doubles their solves.
- **Triplet pair production:** the convolution of its kernel rows with the photons, which is compute-bound.
- **Neutron decay:** two small kernel products per step, a negligible cost.
- **Kaons:** the production, evolution, radiation and decays of four more species. With kaons from both photo-hadronic interactions and $\mathrm{pp}$ collisions, a second set of kaon populations is evolved.
- **Neutrinos by parent:** the pions, muons and neutrinos of the kaon decays are evolved apart from those of the pions made in the interactions, which adds components.

The costs do not simply add up: with annihilation on, the triplet pairs are also evolved as $\mathrm{e}^-$ and $\mathrm{e}^+$ halves. The set-up and memory of the added processes are small next to AM³'s own kernels. Most of both is the annihilation photon kernel.

`exotic` uses the full $\gamma\gamma$ pair-production kernel (`nbins_lep × nbins_pho²` entries in am3_mojo, `nbins_pho³` in AM³). This kernel dominates its step and its memory in both codes. In `ssc_external` (a small leptonic model on a reduced grid), kernel set-up takes a fraction of a second in both codes. Here am3_mojo is 0.02 s slower.

Per-stage time of one step (`results/benchmark/stages_*.md`), from the profiler built into both codes (`set_profile_timing`):

`blazar_simple` (the AM³ example model):

| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.328 ms | 0.882 ms | 7.2x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.069 ms | 0.530 ms | 9.6x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.853 ms | 0.153 ms | 5.6x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.125 ms | 0.044 ms | 2.8x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.094 ms | 0.028 ms | 3.3x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.055 ms | 0.034 ms | 1.6x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.048 ms | 0.051 ms | 0.9x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.043 ms | 0.015 ms | 2.9x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.042 ms | 0.025 ms | 1.7x |
| calc. coeff. pairs | 2.821 ms | 0.087 ms | 32.4x |
| evolve pairs | 1.054 ms | 0.176 ms | 6.0x |
| calc. coeff. proton/neutron | 0.553 ms | 0.213 ms | 2.6x |
| calc. coeff. neutrino | 0.330 ms | 0.205 ms | 1.6x |
| evolve proton | 0.162 ms | 0.093 ms | 1.7x |
| calc. coeff. muon | 0.090 ms | 0.014 ms | 6.4x |
| evolve photon | 0.059 ms | 0.056 ms | 1.0x |
| calc. coeff. pion | 0.051 ms | 0.062 ms | 0.8x |
| photons -> had. grid | 0.040 ms | 0.001 ms | 39.1x |
| evolve neutrino | 0.022 ms | 0.023 ms | 1.0x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **11.56 ms** | **1.91 ms** | **6.1x** |

`leptohadronic_all` (every process on):

| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.752 ms | 1.151 ms | 5.9x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.066 ms | 0.634 ms | 8.0x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.855 ms | 0.181 ms | 4.7x |
| &nbsp;&nbsp;&nbsp;↳ muon compton em. | 0.135 ms | 0.014 ms | 10.0x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.125 ms | 0.030 ms | 4.1x |
| &nbsp;&nbsp;&nbsp;↳ pion compton em. | 0.100 ms | 0.014 ms | 7.1x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.099 ms | 0.028 ms | 3.6x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.091 ms | 0.039 ms | 2.3x |
| &nbsp;&nbsp;&nbsp;↳ pair annihil. | 0.080 ms | 0.065 ms | 1.2x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.047 ms | 0.054 ms | 0.9x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.042 ms | 0.015 ms | 2.8x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.040 ms | 0.031 ms | 1.3x |
| &nbsp;&nbsp;&nbsp;↳ pion syn. em. | 0.037 ms | 0.022 ms | 1.7x |
| &nbsp;&nbsp;&nbsp;↳ muon syn. em. | 0.036 ms | 0.021 ms | 1.7x |
| calc. coeff. pairs | 2.839 ms | 0.171 ms | 16.6x |
| evolve muon | 2.133 ms | 0.182 ms | 11.7x |
| evolve pairs | 1.599 ms | 0.224 ms | 7.1x |
| evolve pion | 1.190 ms | 0.144 ms | 8.3x |
| calc. coeff. proton/neutron | 0.553 ms | 0.223 ms | 2.5x |
| calc. coeff. neutrino | 0.437 ms | 0.256 ms | 1.7x |
| calc. coeff. pion | 0.249 ms | 0.103 ms | 2.4x |
| evolve proton | 0.166 ms | 0.132 ms | 1.3x |
| calc. coeff. muon | 0.120 ms | 0.020 ms | 6.1x |
| evolve photon | 0.083 ms | 0.073 ms | 1.1x |
| photons -> had. grid | 0.040 ms | 0.001 ms | 36.7x |
| evolve neutrino | 0.033 ms | 0.043 ms | 0.8x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **16.21 ms** | **2.77 ms** | **5.9x** |

The indented rows (↳) are parts of "calc. coeff. photon" and are included in its time. AM³'s profiler label "pair annihil." stands for the $\gamma\gamma$ absorption of the photons. The total is the sum of the stages. How it relates to the benchmark's time per step is described in `tests/benchmark/profile_stages.py`.

### What makes am3_mojo faster

Most of the gain comes from better algorithms, not only from threads. [`tests/benchmark/README.md`](tests/benchmark/README.md) explains each improvement:
- **Inverse-Compton emission** from a 4.36 MB table instead of AM³'s 236.1 MB dense kernel, using the kernel's structure on logarithmic grids.
- **Decay and photo-pion kernels** as Toeplitz/Hankel products. On the logarithmic grids these matrices are constant along their diagonals (Toeplitz) or anti-diagonals (Hankel), so a product with them is a convolution of a few hundred values instead of a dense matrix product.
- **Bethe–Heitler emission** from packed windows of its kernel, summed only over the proton bins that hold particles.
- **Cooling rates** computed once per species instead of once per component.
- **SIMD and parallel sections** on the Mojo worker pool, with bitwise identical results for any thread count.
- **Less memory**, from these forms and from allocating the full $\gamma\gamma$ pair kernel only when it is used.
- **Hadron inverse-Compton emission and the interpolation between grids** without AM³'s loops over every bin pair and its per-bin copies.

## Differences from the C++ original

am3_mojo keeps the physics and the interface of AM³ v1.2.0 and adds four processes. [`tests/reference/README.md`](tests/reference/README.md) describes each difference in detail.

**Added processes.** am3_mojo adds $\mathrm{e}^+\mathrm{e}^-$ annihilation, triplet pair production, neutron decay and kaons ($\mathrm{K}^\pm$, $\mathrm{K}^0_\mathrm{S}$, $\mathrm{K}^0_\mathrm{L}$). The kaons come from photo-hadronic interactions and $\mathrm{pp}$ collisions, and the $\mathrm{K}^\pm$ emit synchrotron and inverse-Compton radiation (see [Processes am3_mojo adds to AM³](#added-processes)). These processes are on by default. With them switched off, am3_mojo has AM³'s physics (with the fixes below).

**Bugs of AM³ v1.2.0 that am3_mojo fixes.** am3_mojo fixes bugs in AM³'s solver: the sign of its Green's function, its source integral, the boundaries of its windows, regime 4 and the top bins of a grid. It also fixes bugs in the simulation: the quantum-synchrotron cooling of heavy particles, the series of the inverse-Compton cooling functions, the maximum-energy estimates, the decay products and injections of some particles, the photons on the hadron grid after `update_energy_grid`, and stale decay sources. Further fixes remove reads and writes past the end of arrays and an OpenMP data race, and calls that AM³ does not check raise an exception. Each fix has a regression test in `tests/unit/test_fixes.py`. The C++ build used in the comparisons has the same fixes (`tests/reference/am3_fixes.patch`), except those of the interface and of the OpenMP race. The solver fixes change AM³'s results in common set-ups, by up to tens of per cent in the $\mathrm{e}^\pm$ spectra.

**Interface.** The module is `am3_mojo`. Switches are Python booleans, and errors are raised as Python exceptions.

**Numerics.** am3_mojo evaluates the same equations as AM³, but not with the same floating-point rounding. Most results agree with the C++ code to $10^{-7}$ or better. Three spots where AM³'s result depends on the last bit of a floating-point number differ more.

**Misprints of the AM³ paper.** Where the AM³ paper has misprints, am3_mojo follows the C++ code.

**Kept as in AM³.** am3_mojo keeps some of AM³'s approximations, among them the quantum suppression of synchrotron cooling and the ultra-relativistic forms of the continuous losses.

## License, credits and citation

am3_mojo is distributed under the BSD 3-Clause license of AM³ (`LICENSE`), which retains the copyright notice of the original authors. The physics (apart from the four added processes), the numerical scheme and the interface are those of AM³. The Mojo implementation, the added processes, the optimisations and the test suite are this repository's.

If you use am3_mojo, please cite **both** the AM³ paper and this repository:

- M. Klinger, A. Rudolph, X. Rodrigues, C. Yuan, G. Fichet de Clairfontaine, A. Fedynitch, W. Winter, M. Pohl, S. Gao, *AM³: An Open-Source Tool for Time-Dependent Lepto-Hadronic Modeling of Astrophysical Sources*, ApJS 275, 4 (2024), [arXiv:2312.13371](https://arxiv.org/abs/2312.13371)
- Chen Li, *am3_mojo: the AM³ lepto-hadronic interactions code in Mojo*, version 0.2.0 (2026), https://github.com/chenlinear/am3_mojo

```bibtex
@article{klinger2023am3,
  author        = {Klinger, Marc and Rudolph, Annika and Rodrigues, Xavier and Yuan, Chengchao
                   and Fichet de Clairfontaine, Gaëtan and Fedynitch, Anatoli and Winter, Walter
                   and Pohl, Martin and Gao, Shan},
  title         = {{AM$^3$: An Open-Source Tool for Time-Dependent Lepto-Hadronic Modeling
                   of Astrophysical Sources}},
  journal       = {Astrophys. J. Suppl.},
  volume        = {275},
  pages         = {4},
  year          = {2024},
  doi           = {10.3847/1538-4365/ad725c},
  eprint        = {2312.13371},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.HE}
}

@software{am3_mojo,
  author  = {Chen Li},
  title   = {{am3\_mojo: the AM$^3$ lepto-hadronic interactions code in Mojo}},
  version = {0.2.0},
  year    = {2026},
  url     = {https://github.com/chenlinear/am3_mojo}
}
```

The same information is in [`CITATION.cff`](CITATION.cff). From this file, GitHub offers a "Cite this repository" entry in APA and BibTeX format.

The numerical treatment goes back to:

- S. Gao, M. Pohl, W. Winter, *On the direct correlation between gamma-rays and PeV neutrinos from blazars*, ApJ 843, 109 (2017), [arXiv:1610.05306](https://arxiv.org/abs/1610.05306)

The original AM³ code is at https://gitlab.desy.de/am3/am3 and is documented at https://am3.readthedocs.io/en/latest/.
