# am3_mojo

**am3_mojo** is the [AM³](https://gitlab.desy.de/am3/am3) lepto-hadronic interactions code written in [Mojo](https://www.modular.com/mojo), with the Python interface of AM³ v1.2.0. For the same physics, it runs **9.7–16.0x faster** per step on 4 threads than the serial C++ AM³. It also uses **up to 23.7x less memory** and sets up its kernels up to **127.3x faster**. With the same fixes and constants in the C++ code, the simulated fluxes of the two codes agree to $10^{-6}$ or better.

am3_mojo also adds five processes that AM³ lacks: **electron–positron annihilation** ($\mathrm e^+\mathrm e^- \to \gamma\gamma$), **triplet pair production** ($\mathrm e^\pm\gamma$ $\to \mathrm e^\pm\mathrm e^+\mathrm e^-$), **neutron decay** ($\mathrm n \to \mathrm p\thinspace\mathrm e^-\bar\nu_e$), **kaons** (the production and decay of $\mathrm K^\pm$, $\mathrm K^0_\mathrm S$ and $\mathrm K^0_\mathrm L$), and **charm hadrons** (the production and decay of $\mathrm D^\pm$, $\mathrm D^0$, $\bar{\mathrm D}^0$, $\mathrm D_s^\pm$, $\Lambda_c^+$ and $\bar\Lambda_c^-$). All five processes are on by default and make a step cost 39–158% more time. With them switched off, am3_mojo has AM³'s physics, and the comparison with C++ AM³ are set up this way.

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

## Installation

There are two ways to install am3_mojo. Most users only need the first.

| | Install am3_mojo only | Development set-up |
|---|---|---|
| What you get | `import am3_mojo` in your Python environment | the module, plus the tests, the C++ reference, benchmarks, plots and the checks against the literature |
| Python | 3.14 with pip (or uv) | uv, which installs Python 3.14 |
| `make`, `git`, `curl` | not needed | needed |
| Mojo | needed, but installed automatically by pip (see below) | installed by `uv sync` |
| C/C++ compiler | a C compiler (Mojo uses it to link) | needed (linking, and building the C++ reference) |

**Operating system:** macOS on Apple Silicon (tested on an Apple M5, macOS 27), or Linux on x86-64 or ARM64. Mojo supports Linux, but am3_mojo has not been tested on it yet.

### Install am3_mojo only

`pip` compiles am3_mojo with the Mojo compiler for your machine's CPU, which takes about 10 s.

```bash
# an unpacked copy of the repository, `uv pip install` works the same way.
pip install ./am3_mojo
```

What pip does with Mojo:

1. **While installing**, pip creates a temporary build environment and installs `mojo-compiler==1.2.0` into it (as declared under `[build-system]` in `pyproject.toml`). It compiles am3_mojo with this compiler and then deletes the build environment.
2. **In your environment**, pip installs am3_mojo with its two dependencies, `numpy` and `mojo-compiler==1.2.0`. The module loads the Mojo runtime libraries from `mojo-compiler` (`site-packages/modular/lib`), so do not uninstall this package.

Check the installation:

```bash
python -c "import am3_mojo; print(am3_mojo.__version__, am3_mojo.AM3())"
```

### Development set-up

This set-up needs uv, `make`, `git`, `curl` and a C/C++ compiler. For the reference tests, `git` and `curl` download the original C++ AM³ and its dependencies, and the C++ compiler builds them.

```bash
git clone https://github.com/chenlinear/am3_mojo && cd am3_mojo
uv sync            # .venv with Python 3.14, Mojo 1.2.0, numpy, and the packages of the tests
                   # and of the checks (see below)
make build         # compile the module -> build/am3_mojo.so  (about 10 s)
make install       # make `import am3_mojo` work anywhere in .venv (the tests and checks
                   # find build/ without it)
```

`make clean` deletes `.venv` too. The next `make` target runs `uv sync` again when `.venv` is missing.

Check the installation (after `make install`):

```bash
.venv/bin/python -c "import am3_mojo; print(am3_mojo.__version__, am3_mojo.AM3())"
```

To run all the tests and the checks against the literature, see [Tests, reference comparison and benchmarks](#tests-reference-comparison-and-benchmarks).

## Using am3_mojo

The `AM3` class provides the complete AM³ v1.2.0 Python interface: 383 methods with the same names, arguments and docstrings (corrected where AM³'s misstate the code). There are another 140 methods that control the [processes am3_mojo adds](#added-processes). The [AM³ documentation](https://am3.readthedocs.io/en/latest/) therefore applies. Its examples need `import am3` replaced by `import am3_mojo`, and `True` and `False` in place of `1` and `0` in the switch calls. `help` shows each method's documentation, e.g. `help(am3_mojo.AM3.set_process_pp)`.

- **Threads.** The parallel sections run on the calling thread and on the Mojo runtime's worker pool. By default, the number of threads is the pool's number of workers, which equals the number of performance cores (4 on an M5). Environment variable `AM3_MOJO_THREADS=N` sets another number, and `AM3_MOJO_THREADS=1` runs serially. A value that is not a whole number, or below 1, is reported, and the default is used. One above the number of logical CPUs is reported and capped there. The results are bitwise identical for any $N$. The workers sleep as soon as a section ends: on import, am3_mojo sets the runtime's `MODULAR_THREAD_BUSY_WAIT_US` to 0 unless it is already set.
- **Switches** are Python booleans: `set_process_pp(True)`, `get_process_escape() -> True`. The setters accept only `True` and `False`, and any other value, including `0`, `1` and numpy booleans, raises an exception.
- **Numbers and arrays** passed to setters must be finite and in range. Numbers and the values of arrays must be integers or floats. The field, densities, injection rates and the pp target density must not be negative. Timescales must be at least $10^{-200}$ s. Acceleration factors must be above 0. A value out of range raises an exception and leaves the object as it was. [tests/reference/README.md](tests/reference/README.md) lists every check.
- **Set-up after `init_kernels()`.** `init_kernels()` ends with `clear_am3()`, as in AM³. The escape timescale, the injection and the escape and extra-cooling arrays are therefore set after it. A fresh object has no escape until `set_escape_timescale()` is called.
- **Reusing kernels.** `init_kernels()` is the expensive part of a run. To run several simulations with one set of kernels:
  ```python
  a = am3_mojo.AM3(); <switches>; a.init_kernels()      # once
  for p in parameter_sets:
      a.clear_am3()                    # particles, injection, escape and extra-cooling arrays -> 0
      a.set_mag_field(p.B)             # the next step rebuilds only the field-dependent synchrotron kernels
      a.set_escape_timescale(p.t_esc)  # escape rates must be set again after clear_am3()
      a.set_powerlaw_injection_parameters_electrons(...)   # injection
      for _ in range(n):
          a.evolve_step()
  ```

  A reused object gives results bitwise identical to those of a fresh one. `tests/unit/test_reuse.py` checks this, and `results/plots/reuse_summary.png` shows it.

  ![Kernel reuse: deviation from re-initialising the kernels, and set-up time saved](results/plots/reuse_summary.png)

  The figures below show the individual runs of the blazar and the leptohadronic set-ups. Each shows the photon spectra (top) and, per energy bin, how the photon, $\mathrm{e}^\pm$, proton and neutrino spectra differ from those with re-initialised kernels (bottom). Every getter returns bitwise identical values as in the run with re-initialised kernels.

  ![Photon spectra and differences of four runs with reused kernels, blazar setup](results/plots/reuse_blazar.png)

  ![Photon spectra and differences of four runs with reused kernels, leptohadronic setup](results/plots/reuse_leptohadronic.png)

  `clear_am3()` sets the particle densities, the injection, escape and extra-cooling rates and all coefficient arrays to zero, and keeps the kernels, the switches and the scalar parameters. The estimated maximum energies return to their defaults. Magnetic field $B$, the expansion timescale, the $\mathrm{pp}$ density and the time step therefore carry over to the next run without being set again. The escape and injection rates, however, stay zero until `set_escape_timescale()` and `set_powerlaw_injection_parameters_*()` (or `set_injection_rate_*()`) are called again. Meanwhile, `get_escape_timescale()` and `get_powerlaw_injection_parameters_*()` still report the old values. (With `set_estimate_max_energies(True)`, the power-law injection is recomputed from the stored parameters from the second step on.)

<a id="added-processes"></a>
## Processes am3_mojo adds to AM³

Table 1 of the hadronic code comparison of [Cerruti et al. (2026, ApJS 282, 22)](https://arxiv.org/abs/2411.14218) lists the physical processes of five codes. am3_mojo adds four processes that other codes have and AM³ lacks: electron–positron annihilation, triplet pair production, neutron decay and kaons. am3_mojo also adds charm hadrons, which Table 1 does not list. All five processes are **on by default**. Every comparison with the C++ AM³ switches them off (the test implementation `mojo-am3` in `tests/support/paths.py`), so am3_mojo solves AM³'s physics.

| Process | Switches (default `True`) | Outputs and timescales |
|---|---|---|
| $\mathrm e^+\mathrm e^- \to \gamma\gamma$ | `set_process_pair_annihilation` (master switch); `_emission` / `_cooling`: the photons / the $\mathrm e^\pm$ loss | `get_photons_pair_annihilation()`; `get_t_electron_annihilation()`, `get_t_positron_annihilation()` |
| $\mathrm e^\pm\gamma$ $\to \mathrm e^\pm\mathrm e^+\mathrm e^-$ | `set_process_triplet_pair_production` (master switch; also the photon absorption); `_emission` / `_cooling`: the produced pairs / the $\mathrm e^\pm$ energy loss | `get_pairs_triplet()` (the $\mathrm e^\pm$ component) with its photons `get_photons_triplet_pairs_syn()`, `get_photons_triplet_pairs_compton()` and `get_photons_triplet_pairs_syn_compton()`; `get_t_pair_triplet()` ($\mathrm e^\pm$ energy loss), `get_t_photon_triplet()` (photon absorption) |
| $\mathrm n \to \mathrm p\thinspace\mathrm e^-\bar\nu_e$ | `set_process_neutron_decay`: the decay of the neutrons and its protons, electrons and electron antineutrinos | `get_electrons_neutron_decay()` (the $\mathrm e^\pm$ component) with its photons `get_photons_neutron_decay_electrons_syn()`, `get_photons_neutron_decay_electrons_compton()` and `get_photons_neutron_decay_electrons_syn_compton()`; `get_electron_anti_neutrinos_neutron_decay()` (the neutrino component); `get_t_neutron_decay()` |
| kaons | `set_process_kaons` (master switch): $\mathrm K^\pm$, $\mathrm K^0_\mathrm S$ and $\mathrm K^0_\mathrm L$ production in photo-hadronic interactions and $\mathrm{pp}$ collisions, their evolution and decay | `get_kaon_plus()`, `get_kaon_minus()`, `get_kaons()` ($\mathrm K^+ + \mathrm K^-$), `get_kaon_0S()`, `get_kaon_0L()`; `get_t_kaon_escape()` (charged), `get_t_kaon_neutral_escape()`; a user-set extra cooling, as for AM³'s other species: `set_t_kaon_extra_cooling()` / `get_t_kaon_extra_cooling()` |
| kaon decays | `set_process_kaon_decay`: every non-radiative mode above 0.1% and the smaller $\mathrm K^0_\mathrm S$ semileptonic and $\mathrm K^0_\mathrm L \to \pi^0\pi^0$ modes | `get_t_kaon_decay()` (charged), `get_t_kaon_0S_decay()`, `get_t_kaon_0L_decay()`; AM³'s photo-hadronic and $\mathrm{pp}$ neutrino components by parent: the neutrinos of the decays of the pions made in the interactions, `get_neutrinos_pion_decay_photopion()` and `get_neutrinos_pion_decay_proton_proton()` ($\nu_\mu$ and $\bar\nu_\mu$ only); those of the decays of these pions' muons, `get_neutrinos_muon_decay_photopion()` and `get_neutrinos_muon_decay_proton_proton()`; and those of the kaon decays and of the pions and muons the kaons decay into, `get_neutrinos_kaon_decay_photopion()` and `get_neutrinos_kaon_decay_proton_proton()` (each also per flavour, e.g. `get_muon_neutrinos_pion_decay_photopion()`) |
| kaon synchrotron | `set_process_kaon_syn` (master switch); `_emission` / `_cooling`: the photons / the kaon energy loss | `get_photons_kaons_syn()`; `get_t_kaon_syn()` |
| kaon inverse Compton | `set_process_kaon_compton` (master switch); `_emission` / `_cooling`: the photons / the kaon energy loss | `get_photons_kaons_compton()`; `get_t_kaon_compton()` |
| charm hadrons | `set_process_charm` (master switch): $\mathrm D^\pm$, $\mathrm D^0$, $\bar{\mathrm D}^0$, $\mathrm D_s^\pm$, $\Lambda_c^+$ and $\bar\Lambda_c^-$ production in photo-hadronic interactions and $\mathrm{pp}$ collisions, their evolution and decay | `get_charm_hadrons()` (all eight), `get_D_plus()`, `get_D_minus()`, `get_D_zero()`, `get_anti_D_zero()`, `get_Ds_plus()`, `get_Ds_minus()`, `get_Lambda_c_plus()`, `get_anti_Lambda_c_minus()`; `get_t_charm_escape()` (charged), `get_t_charm_neutral_escape()`; `set_t_charm_extra_cooling()` / `get_t_charm_extra_cooling()` |
| charm decays | `set_process_charm_decay`: the [PDG](https://pdg.lbl.gov/2026/) modes above 0.5% (semileptonic above 0.1%) | `get_t_D_plus_decay()`, `get_t_D_zero_decay()`, `get_t_Ds_decay()`, `get_t_Lambda_c_decay()`; AM³'s photo-hadronic and $\mathrm{pp}$ neutrino components by parent: the neutrinos of the charm decays and of the kaons, pions and muons they decay into, `get_neutrinos_charm_decay_photopion()` and `get_neutrinos_charm_decay_proton_proton()` (each also per flavour); their $\nu_\tau$, `get_tau_neutrinos()` and `get_tau_anti_neutrinos()` (the sums of both origins), `get_tau_neutrinos_charm_decay_photopion()`, `get_tau_anti_neutrinos_charm_decay_photopion()` and the same for `proton_proton`; `get_neutrinos()`, `get_neutrinos_photopion()`, `get_neutrinos_proton_proton()` and both `get_neutrinos_charm_decay_…()` include the $\nu_\tau$ and $\bar\nu_\tau$ |
| charm synchrotron | `set_process_charm_syn` (master switch); `_emission` / `_cooling`: the photons / the energy loss of the charged charm hadrons | `get_photons_charm_syn()`; `get_t_charm_syn()` |
| charm inverse Compton | `set_process_charm_compton` (master switch); `_emission` / `_cooling`: the photons / the energy loss | `get_photons_charm_compton()`; `get_t_charm_compton()` |

Each switch has a getter (`get_process_…`). Switch the added processes on before `init_kernels()`, which builds their kernels: one switched on after it stays inactive. So do $\mathrm{pp}$ collisions, Bethe–Heitler and $\gamma\gamma$ pair production, and $\mathrm{e}^\pm$ inverse Compton, as in AM³. Photo-pion production or the hadronic processes switched on after it make `evolve_step()` raise. AM³'s own `set_process_annihilation`, `get_pairs_annihilation()` and `get_t_photon_annihilation()` belong to $\gamma\gamma$ pair production, not to electron–positron annihilation. The printed switch table and the profiler (`get_profiled_time_labels()`) list the added processes too. Their methods are listed in `tools/am3_mojo_api.json`.

### Derivations and checks

Each added process has a folder in [tests/literature_checks/](tests/literature_checks/README.md): [electron–positron annihilation](tests/literature_checks/annihilation/README.md), [triplet pair production](tests/literature_checks/triplet/README.md), [neutron decay](tests/literature_checks/neutron_decay/README.md), [kaons](tests/literature_checks/kaons/README.md) and [charm hadrons](tests/literature_checks/charm/README.md). The README starts with the equations and tables of the cited papers, derives the equations and tables that the Mojo code uses, and cites the struct or function in the Mojo code. The folder's Python checks verify that each derivation is correct, exactly with sympy or to 50 digits with mpmath. Where a table comes from measured data or a Monte Carlo, a script of the folder recomputes or checks it.

### Effect on a simulation

`results/plots/added_processes_leptohadronic_all.png` (`make plots`) compares am3_mojo with and without the added processes in the `leptohadronic_all` scenario.

![Muon, electron and tau neutrino spectra, the neutrino flavour ratio, photon and e± spectra of leptohadronic_all with and without e+e- annihilation, triplet pair production, neutron decay, kaons and charm hadrons](results/plots/added_processes_leptohadronic_all.png)

Kaon decays extend the $\nu_\mu$ and $\nu_e$ spectra beyond AM³'s cutoff. At the highest energies, pions and muons lose their energy to synchrotron radiation before they decay. The charged kaons, heavier and shorter-lived, decay before they lose their energy. This is the known kaon effect on high-energy neutrino spectra ([Baerwald, Hümmer & Winter 2011](https://arxiv.org/abs/1009.4010); for the $\nu_e$ from $\mathrm{K}^0$ decays, [Petropoulou et al. 2014](https://arxiv.org/abs/1405.2091)).

<a id="tests-reference-comparison-and-benchmarks"></a>
## Tests, reference comparison and benchmarks

Everything runs through `make`. To run all the tests and all the literature checks:

```bash
make reference          # once: download and build the C++ AM³ for the reference tests
make test               # all tests
make literature_checks  # the literature checks, outputs in results/literature_checks/
```

All the targets:

| Command | What it does | Needs | Time (on Apple M5) |
|---|---|---|---|
| `make format` | formats the Mojo code (`mojo format`) and the Python code of `tests/` and `tools/` (`ruff format`, settings in `pyproject.toml`) | | 30 s the first time, then under 1 s |
| `make reference` | downloads and builds the C++ AM³, original and patched (see below) | internet, C++ compiler | 1–2 min (once) |
| `make test-unit` | tests of am3_mojo on its own (401) | `make build` | 4 min |
| `make test-reference` | compares am3_mojo with the C++ AM³ (25) | `make reference` | 2 min |
| `make test-asserts` | the unit tests on a build with Mojo's bounds checks and assertions (`build-asserts/`) | | 4.5 min (+ the compile) |
| `make test` | all tests (426); reference tests are skipped if the reference is not built, or was built from another `am3_fixes.patch` | | 6 min |
| `make benchmark` | timings, memory and per-stage profiles vs the C++ AM³ | `make reference` | 5 min |
| `make plots` | comparison, kernel-reuse and added-process figures | `make reference` | 1 min |
| `make literature_checks` | the derivation checks, the inventory and every script of `tests/literature_checks/` (`tests/literature_checks/run_literature_checks.py`), with the arguments of the figures their READMEs quote, and `tools/ky_angular_extract.py`; outputs in `results/literature_checks/`, the tool's in `results/tools/` | | 5 min |
| `make all` | format + build + reference + test + benchmark + plots + literature_checks | | 18 min (+ reference build) |

Results are written to `results/`. The tables and plots are kept in this repository:
- `reference/summary.md`: the deviation table, written only by a run of all four scenarios on `build/`;
- `benchmark/benchmark.md` and `benchmark/benchmark.json`: step and set-up times, and memory;
- `benchmark/stages_<scenario>.md`: per-stage profiles;
- `plots/*.png`: the figures shown in this README.

### Comparison with the original C++ AM³ (`tests/reference`)

`make reference` (`tests/reference/setup_reference.sh`) does the following:
1. clones AM³ from `gitlab.desy.de/am3/am3` at commit `04d63d9`: the tag v1.2.0 plus 7 commits that change only its documentation and examples (its package metadata and `am3.__version__` still say 1.0.1);
2. downloads and builds GSL 2.8, Eigen 3.4.0 and, on macOS, LLVM libomp 19.1.7 into `tests/reference/work/deps`;
3. compiles two serial builds of AM³'s Python module:
   - the original;
   - the original with the fixes of the AM³ bugs that am3_mojo fixes and with am3_mojo's constants (`am3_fixes.patch`: 144 hunks in 21 source files, see [tests/reference/README.md](tests/reference/README.md)). It is rebuilt whenever the patch changes.

It can be re-run safely, since completed steps are skipped. `AM3_REFERENCE_DIR` changes the work directory, and `AM3_SOURCE_DIR` uses an existing AM³ checkout instead of cloning. That checkout is only read, also through a symlink. The patch is applied without fuzz, so a hunk whose context no longer fits stops the script.

The reference tests and the benchmark run am3_mojo with [processes it adds](#added-processes) switched off (`mojo-am3`) against the patched C++ build (`cpp-fixed`), so that both codes have the same physics.

**Criterion** (`tests/support/compare.py`):
- **Fluxes:** particle densities, component spectra and injection rates must agree to $10^{-6}$ relative: with $r$ the ratio of the two codes, $\max(r, 1/r) - 1$ must stay within it. The check covers every energy bin above $10^{-12}$ of the peak of either code's array in particle number ($E\thinspace dN/dE$) or in energy ($E^2\thinspace dN/dE$, the quantity flux plots show). The bins below all the floors are not visible on a flux plot, which spans many orders of magnitude.
- **Grids:** these must be identical, as must the scalars and the power-law injection parameters. The optimisation getters, which AM³ returns in the grid's units, are compared in eV.
- **Timescales:** these are reported, not checked, on the bins where the rate $1/{t}$ is above $10^{-12}$ of its largest. The longest timescales, including those that stand for an absent process, carry no information.
- Every array must exist in both codes with the same shape (except the two neutrino timescales described in [tests/reference/README.md](tests/reference/README.md), compared on AM³'s bins). A flux bin where either code's value is not finite counts as a failure.

`make plots` shows the comparison as figures, with the original C++ build beside the patched one. `deviation_vs_time.png` plots am3_mojo's worst deviation from each over all fluxes, at every checkpoint of the four scenarios.
- **Patched build:** the dashed red line marks the criterion of $10^{-6}$.
- **Original:** it has no criterion, as am3_mojo fixes its bugs ([tests/reference/README.md](tests/reference/README.md)).

![Worst deviation of am3_mojo from the patched and the original C++ AM³ at every checkpoint of the four scenarios](results/plots/deviation_vs_time.png)

`spectra_<scenario>.png` overlays the spectra of am3_mojo and both C++ builds at the first and the last checkpoint (top). It also shows am3_mojo's deviation from each build at the last checkpoint, in every energy bin the criterion checks (bottom).

![Photon, e±, proton and neutrino spectra of blazar_simple from am3_mojo and the patched and original C++ AM³, with the per-bin deviations](results/plots/spectra_blazar_simple.png)

Every process on:

![Photon, e±, proton and neutrino spectra of leptohadronic_all from am3_mojo and the patched and original C++ AM³, with the per-bin deviations](results/plots/spectra_leptohadronic_all.png)

A leptonic synchrotron self-Compton model with an external photon field:

![Photon and e± spectra of ssc_external (no hadrons) from am3_mojo and the patched and original C++ AM³, with the per-bin deviations](results/plots/spectra_ssc_external.png)

Exotic:

![Photon, e±, proton and neutrino spectra of exotic from am3_mojo and the patched and original C++ AM³, with the per-bin deviations](results/plots/spectra_exotic.png)

## Performance: am3_mojo (4 threads) vs the C++ original (serial)

`make benchmark` runs every scenario three times, in one fresh process per code and scenario. It reports the fastest run and the peak memory of one simulation. The set-up counts `init_kernels()`, `set_mag_field()` before the steps, and the first step. The C++ builds the kernels that depend on the field in `set_mag_field()`, and am3_mojo builds them in the first step. Every ratio below is computed from the unrounded values. The benchmarks ran on an Apple M5 (4 performance and 6 efficiency cores, 16 GB). [tests/benchmark/README.md](tests/benchmark/README.md) explains the improvements in am3_mojo.

The C++ AM³ is the serial (single-threaded) Release build made by `make reference` (clang, `-O3`). It includes the fixes of `am3_fixes.patch`, so both codes solve the same equations. [tests/reference/README.md](tests/reference/README.md) describes each difference in detail.

**Time per evolution step** (lower is better). Both codes compute the same physics here: am3_mojo runs with the [processes it adds](#added-processes) switched off.

| Scenario | C++ AM³ (1 thread) | am3_mojo (1 thread) | am3_mojo (4 threads) | Speed-up over C++ (1 thread / 4 threads) |
|---|---|---|---|---|
| `blazar_simple` | 11.36 ms | 1.58 ms | **0.71 ms** | 7.2x / **16.0x** |
| `leptohadronic_all` | 15.89 ms | 2.31 ms | **1.04 ms** | 6.9x / **15.3x** |
| `ssc_external` | 3.00 ms | 0.54 ms | **0.22 ms** | 5.5x / **13.7x** |
| `exotic` | 11.15 ms | 1.76 ms | **1.15 ms** | 6.3x / **9.7x** |

**Cost of the added processes** (am3_mojo on 4 threads, time per evolution step):

| Scenario | Without them | With them (default) | Extra time |
|---|---|---|---|
| `blazar_simple` | 0.71 ms | 1.83 ms | +158% |
| `leptohadronic_all` | 1.04 ms | 2.68 ms | +157% |
| `ssc_external` | 0.22 ms | 0.31 ms | +39% (triplet pair production only: merged $\mathrm{e}^\pm$, so no annihilation; no hadrons, so no neutron decay, kaons or charm) |
| `exotic` | 1.15 ms | 1.84 ms | +59% |

**Kernel set-up and peak memory** (am3_mojo on 4 threads; the peak memory varies by up to about 10 MB from run to run):

| Scenario | Kernel set-up (`init_kernels`, `set_mag_field` and the first step), C++ → am3_mojo | Peak memory, C++ → am3_mojo | Added processes: set-up, memory |
|---|---|---|---|
| `blazar_simple` | 17.54 s → **0.14 s** (127.3x) | 2.88 GB → **0.12 GB** (23.7x) | +0.36 s, +40 MB |
| `leptohadronic_all` | 18.18 s → **0.19 s** (95.7x) | 2.88 GB → **0.13 GB** (22.5x) | +0.42 s, +46 MB |
| `ssc_external` | 0.16 s → **0.08 s** (1.9x) | 2.07 GB → **0.10 GB** (21.2x) | +0.11 s, +13 MB |
| `exotic` | 2.39 s → **0.12 s** (19.9x) | 1.00 GB → **0.18 GB** (5.6x) | +0.25 s, +13 MB |

## License, credits and citation

am3_mojo is distributed under the BSD 3-Clause license of AM³ (`LICENSE`), which retains the copyright notice of the original authors. The physics (apart from the five added processes), the numerical scheme and the interface are those of AM³. The Mojo implementation, the added processes, the optimisations and the test suite are this repository's.

If you use am3_mojo, please cite **both** the [AM³ paper](https://arxiv.org/abs/2312.13371) and this repository:

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

The same information is in [CITATION.cff](CITATION.cff).

The numerical treatment is from:

- S. Gao, M. Pohl, W. Winter, *On the direct correlation between gamma-rays and PeV neutrinos from blazars*, ApJ 843, 109 (2017), [arXiv:1610.05306](https://arxiv.org/abs/1610.05306)

The original AM³ code is at https://gitlab.desy.de/am3/am3 and its documentation is at https://am3.readthedocs.io/en/latest/.
