# am3_mojo

**am3_mojo** is the [AM³](https://gitlab.desy.de/am3/am3) lepto-hadronic interactions code written in [Mojo](https://www.modular.com/mojo). am3_mojo keeps the Python interface of AM³ v1.0.1. On 4 threads it runs **3.8–5.8× faster** than the original C++ AM³ (serial) and uses **up to 27× less memory**; its kernel set-up is up to 3.4× faster. Its simulated fluxes agree with the original to better than 10⁻⁵ relative (typically 10⁻⁷).

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
                           pair_production, bethe_heitler, photopion, proton_proton,
                           decays, acceleration, escape, expansion, extra_cooling, injection)
    simulation/            the time step: coefficients and parallel species updates
    api/                   the Python-facing AM3 object
tests/
  unit/                    tests of am3_mojo on its own
  reference/               comparison with the original C++ AM³ (+ setup_reference.sh
                           that downloads and builds it)
  benchmark/               timings and per-stage profiles vs the C++ AM³
  plots/                   comparison and kernel-reuse figures
  support/                 shared scenarios, runners and the comparison criterion
results/                   tables and figures of the last full run (make all)
tools/                     binding generator, the AM³ v1.0.1 interface specification and
                           the build backend used by `pip install` (mojo_build_backend.py)
Makefile                   build / install / test / reference / benchmark / plots / all
CITATION.cff               how to cite am3_mojo (and AM³)
pyproject.toml, uv.lock    the Python environment (Mojo 1.1.0, numpy, pytest, ...)
```

## Installation

There are two ways to install am3_mojo. Most users only need the first.

| | Install am3_mojo only | Development set-up |
|---|---|---|
| What you get | `import am3_mojo` in your Python environment | the module, plus the tests, the C++ reference, benchmarks and plots |
| Python | 3.12 with pip (or uv) | uv, which installs Python 3.12 |
| `make`, `git`, `curl` | not needed | needed |
| Mojo | needed, but installed automatically by pip (see below) | installed by `uv sync` |
| C/C++ compiler | a C compiler (Mojo uses it to link) | needed (linking, and building the C++ reference) |

**Operating system:** macOS on Apple Silicon (tested on an Apple M5, macOS 26), or Linux on x86-64 or ARM64. Linux is supported by Mojo, but am3_mojo has not been tested on it yet.

**Mojo.** am3_mojo needs Mojo 1.1.0 twice: the Mojo *compiler* to compile the module when it is installed, and the Mojo *runtime libraries* whenever the module is imported. Both come from the `mojo-compiler` package on PyPI (about 230 MB), which pip or uv downloads automatically. You do not install Mojo yourself, and a Mojo installation that is already on the machine (e.g. through pixi or conda) is neither needed nor used.

### Install am3_mojo only

`pip` compiles am3_mojo with the Mojo compiler on your machine, for your machine's CPU; this takes about 10 s. The only other thing required is a C compiler, which Mojo uses as its linker: the Xcode Command Line Tools on macOS (`xcode-select --install`), or gcc/clang on Linux. `make`, `git` and `curl` are not used.

```bash
# an unpacked copy of the repository
pip install ./am3_mojo
```

What pip does with Mojo:

1. **While installing**, pip creates a temporary build environment, installs `mojo-compiler==1.1.0` into it (declared under `[build-system]` in `pyproject.toml`) and compiles am3_mojo with it. The build environment is deleted afterwards.
2. **In your environment**, pip installs am3_mojo together with its two dependencies, `numpy` and `mojo-compiler==1.1.0`. The module loads the Mojo runtime libraries from that package (`site-packages/modular/lib`), so do not uninstall `mojo-compiler`: without it, `import am3_mojo` fails. The package also puts a `mojo` command into the environment, which am3_mojo itself does not use.

`uv pip install` works the same way. Check the installation:

```bash
python -c "import am3_mojo; print(am3_mojo.__version__, am3_mojo.AM3())"
```

### Development set-up (tests, reference comparison, benchmarks)

This set-up needs **uv**, **`make`**, **`git`**, **`curl`** and a **C/C++ compiler**. `git` and `curl` download the original C++ AM³ and its dependencies for the reference tests, and the C++ compiler builds them.

```bash
git clone https://github.com/chenlinear/am3_mojo && cd am3_mojo
uv sync            # .venv with Python 3.12, Mojo 1.1.0, numpy, pytest, matplotlib, cmake, ninja
make build         # compile the module -> build/am3_mojo.so  (about 10 s)
make install       # optional: make `import am3_mojo` work anywhere in .venv
```

Check the installation:

```bash
.venv/bin/python -c "import am3_mojo; print(am3_mojo.__version__, am3_mojo.AM3())"
```

Without `make install`, add `build/` to the import path instead: `sys.path.insert(0, "build")` or `PYTHONPATH=build`.

## Using am3_mojo

The `AM3` class provides the complete AM³ v1.0.1 Python interface: 383 methods with the same names, arguments and docstrings. The [AM³ documentation](https://am3.readthedocs.io/en/latest/) and its example notebooks therefore apply; replace `import am3` by `import am3_mojo`. `help(am3_mojo.AM3.set_process_pp)` shows each method's documentation.

- **Threads.** The parallel sections use the Mojo runtime's worker pool. The default is its worker count, which is the number of performance cores (4 on an M5). `AM3_MOJO_THREADS=N` overrides it, and `AM3_MOJO_THREADS=1` runs serially. Results are bitwise identical for any N.
- **Switches** are Python booleans: `set_process_pp(True)`, `get_process_escape() -> True`. The integers 0 and 1 are accepted too.
- **Reusing kernels.** `init_kernels()` is the expensive part (0.2–3 s). To run several simulations with one set of kernels:
  ```python
  a = am3_mojo.AM3(); <switches>; a.init_kernels()      # once
  for p in parameter_sets:
      a.clear_am3()                    # particles, injection, escape and extra-cooling arrays -> 0
      a.set_mag_field(p.B)             # rebuilds only the B-dependent synchrotron kernels
      a.set_escape_timescale(p.t_esc)  # escape rates must be set again after clear_am3()
      a.set_powerlaw_injection_parameters_electrons(...)   # injection too
      for _ in range(n): a.evolve_step()
  ```

  A reused object gives bitwise the same results as a fresh one; `tests/unit/test_reuse.py` checks this, and `results/plots/reuse_summary.png` shows it. On the left, correctly reused kernels (blue) are bitwise identical to kernels built again before every run, while skipping `clear_am3()` or not setting the escape timescale again gives wrong spectra. On the right, the kernel set-up time for four runs drops by a factor of 4.

  ![Kernel reuse: deviation from re-initialising the kernels, and set-up time saved](results/plots/reuse_summary.png)

  The spectra of the individual runs, for the leptohadronic and the blazar setup:

  ![Spectra of four runs with reused kernels, leptohadronic setup](results/plots/reuse_leptohadronic.png)

  ![Spectra of four runs with reused kernels, blazar setup](results/plots/reuse_blazar.png) `clear_am3()` keeps the kernels, the switches and the scalar parameters (B, timescales, pp density, time step, power-law parameters). Note that `get_escape_timescale()` keeps reporting the old value, even though the escape *rates* are zero until `set_escape_timescale()` is called again.

## Tests, reference comparison and benchmarks

Everything is run through `make`:

| Command | What it does | Needs | Time (on Apple M5) |
|---|---|---|---|
| `make test-unit` | tests of am3_mojo on its own | `make build` | 1.5 min |
| `make reference` | downloads and builds the original C++ AM³ (see below) | internet, C++ compiler | 1–2 min (once) |
| `make test-reference` | compares am3_mojo with the C++ AM³ | `make reference` | 4 min |
| `make test` | all tests (692); reference tests are skipped if the reference is not built | | 5 min |
| `make benchmark` | timings, memory and per-stage profiles vs the C++ AM³ | `make reference` | 4 min |
| `make plots` | comparison and kernel-reuse figures | | 1 min |
| `make all` | build + reference + test + benchmark + plots | | 11 min (+ reference build) |

Results are written to `results/`. The tables and figures are part of the repository, so the results of the last full run (`make all` on an Apple M5) can be seen without running anything:
- `reference/summary.md`: the deviation table;
- `benchmark/benchmark.md` and `benchmark/benchmark.json`: step and set-up times, and memory;
- `benchmark/stages_<scenario>.md`: per-stage profiles;
- `plots/*.png`: the figures shown in this README.

The raw simulation outputs that the tests write to `results/runs/` and `results/reuse/` (about 36 MB of `.npz` files, rewritten by every test run) are not part of the repository.

pytest can also be run directly, for example `.venv/bin/python -m pytest tests/unit/test_api.py -v` or `-m "not reference"`.

### Unit tests (`tests/unit`)

| File | Checks |
|---|---|
| `test_api.py` | The interface matches AM³ v1.0.1: all 383 methods with their signatures and docstrings, and the version attributes. The 59 switches are real booleans: they round-trip `True`/`False`, reject non-boolean values, and show `True`/`False` in docstrings and in the printed switch table. Setting switches with `True`/`False` or with `1`/`0` gives bitwise identical simulations. |
| `test_simulation_sanity.py` | All four scenarios run. Every flux is finite and non-negative, and the spectra are populated and evolving. |
| `test_reuse.py` | Reusing kernels: identical to fresh objects bitwise, repeated configurations reproduced bitwise, the configurations really differ, and `init_kernels` runs only once. |
| `test_threads.py` | 1 and 4 worker threads give bitwise identical results. |
| `test_environment.py` | Importing am3_mojo leaves the process environment unchanged, so Python subprocesses and `multiprocessing` workers still run in the virtual environment. |

The four scenarios (`tests/support/scenarios.py`) drive the public API only:

| Scenario | Content |
|---|---|
| `blazar_simple` | The proton-synchrotron blazar model of the AM³ example notebook, 800 steps. |
| `leptohadronic_all` | Every process on, including pp, quantum synchrotron, pion and muon synchrotron and IC, and expansion; 300 steps. |
| `ssc_external` | A leptonic synchrotron self-Compton model with an external black-body field, on a reduced energy grid, with merged e±, maximum-energy estimation and user-set arrays. |
| `exotic` | Rarely used paths: the full γγ pair kernel, pp with QGSJET/Geant4, RK4 source integration, user densities, injection and escape for all species, and a magnetic-field change during the run. |

### Comparison with the original C++ AM³ (`tests/reference`)

`make reference` (`tests/reference/setup_reference.sh`) does the following:
1. clones AM³ from `gitlab.desy.de/am3/am3` at commit `04d63d9` (v1.0.1);
2. downloads and builds GSL 2.8, Eigen 3.4.0 and, on macOS, LLVM libomp 19.1.7 into `tests/reference/work/deps`;
3. compiles two serial builds of AM³'s Python module:
   - the original;
   - the original with its two out-of-bounds reads fixed (`oob_fix.patch`; see [Differences](#differences-from-the-c-original)).

It can be re-run safely; completed steps are skipped. `AM3_REFERENCE_DIR` changes the work directory, and `AM3_SOURCE_DIR` uses an existing AM³ checkout instead of cloning.

| File                        | Checks                                                                                                                                                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_scenarios_vs_cpp.py`  | Every getter at every checkpoint of the four scenarios, 5,126 arrays, against the C++ original (criterion below).                                                                                                                  |
| `test_api_defaults.py`      | All 233 getters are identical after `init_kernels`, and the same calls succeed or fail.                                                                                                                                            |
| `test_reuse_vs_cpp.py`      | The C++ AM³ supports the same kernel reuse (reused equals fresh bitwise), and am3_mojo's reused runs match it.                                                                                                                     |
| `test_special_functions.py` | Compiled harnesses compare `am3_mojo`'s inverse-Compton cooling kernel function, the Bessel functions `K_ν` (ν = 1/3, 2/3, 4/3, 5/3) and the synchrotron integral against the original C++/GSL code, element by element, to 10⁻¹³. |

**Criterion** (`tests/support/compare.py`):
- **Fluxes:** particle densities, component spectra and injection rates must agree to 10⁻⁵ relative on every energy bin above 10⁻¹² of the array's peak. On a flux plot, which spans many orders of magnitude, a 10⁻⁵ difference is far below the line width.
- **Grids:** these must be identical.
- **Timescales:** these are reported only.

Current results:

| Scenario | Flux arrays | Worst flux deviation | Worst timescale deviation |
|---|---|---|---|
| blazar_simple | 768 | 1.0e-07 | 3.0e-08 |
| exotic | 384 | 1.2e-07 | 2.1e-08 |
| leptohadronic_all | 576 | 2.2e-06 | 5.7e-08 |
| ssc_external | 384 | 1.5e-08 | 1.3e-07 |

The remaining differences come from am3_mojo computing in double precision where AM³ uses single-precision (float32) kernels and float32 matrix-vector products. am3_mojo is therefore the more accurate of the two. Almost every bin agrees to about 10⁻⁷. The largest deviation, 2.2 × 10⁻⁶ in `leptohadronic_all`, sits in a single bin of the pp-produced e± spectrum where the solver creates a one-bin dip. The dip amplifies any 10⁻⁷ change about twentyfold, and the neighbouring bins agree to 10⁻⁸.

`make plots` shows the comparison as figures. `deviation_vs_time.png` plots the worst deviation over all fluxes at every checkpoint of the four scenarios; the dashed red line is the 10⁻⁵ criterion:

![Worst deviation between am3_mojo and the C++ AM³ at every checkpoint of the four scenarios](results/plots/deviation_vs_time.png)

`spectra_<scenario>.png` overlays the spectra of both codes at the first and the last checkpoint (top) and shows the deviation of every energy bin at the last one (bottom). The C++ spectra are drawn as thick grey lines and the am3_mojo spectra as dashed blue lines on top of them. The AM³ example model:

![Photon, e±, proton and neutrino spectra of blazar_simple from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_blazar_simple.png)

Every process on:

![Photon, e±, proton and neutrino spectra of leptohadronic_all from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_leptohadronic_all.png)

A leptonic synchrotron self-Compton model with an external photon field:

![Photon, e±, proton and neutrino spectra of ssc_external from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_ssc_external.png)

Rarely used code paths:

![Photon, e±, proton and neutrino spectra of exotic from am3_mojo and the C++ AM³, with the per-bin deviation](results/plots/spectra_exotic.png)

## Performance: am3_mojo (4 threads) vs the C++ original (serial)

`make benchmark` measures every scenario three times, each in a fresh process, and reports the minimum. We used an Apple M5 (4 performance and 6 efficiency cores, 16 GB). The C++ AM³ is the serial Release build made by `make reference` (clang, `-O3`).

| Scenario | C++ AM³ (serial) | **am3_mojo, 4 threads** | Speed-up | am3_mojo, 1 thread |
|---|---|---|---|---|
| `blazar_simple` | 11.52 ms / step | **1.98 ms / step** | **5.8×** | 2.89 ms / step |
| `leptohadronic_all` | 16.72 ms / step | **3.13 ms / step** | **5.3×** | 4.35 ms / step |
| `ssc_external` | 3.06 ms / step | **0.62 ms / step** | **4.9×** | 1.38 ms / step |
| `exotic` | 11.54 ms / step | **3.07 ms / step** | **3.8×** | 7.08 ms / step |

| Scenario | Kernel set-up (`init_kernels`), C++ → am3_mojo | Peak memory, C++ → am3_mojo |
|---|---|---|
| `blazar_simple` | 6.96 s → **2.03 s** (3.4×) | 3.38 GB → **0.34 GB** (9.9×) |
| `leptohadronic_all` | 11.11 s → **3.24 s** (3.4×) | 3.38 GB → **0.35 GB** (9.7×) |
| `ssc_external` | 0.14 s → 0.17 s (0.8×) | 2.47 GB → **0.09 GB** (27×) |
| `exotic` | 1.49 s → **0.46 s** (3.2×) | 1.46 GB → **0.55 GB** (2.7×) |

`exotic` uses the full γγ pair-production kernel (`nbins_pho³` entries), which dominates its step and its memory in both codes. In `ssc_external` (a small leptonic model on a reduced grid) kernel set-up takes a fraction of a second in both codes, and am3_mojo is 0.03 s slower.

Per-stage time of one step (`results/benchmark/stages_*.md`), from the profiler built into both codes (`set_profile_timing`):

`blazar_simple` (the AM³ example model):

| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.143 ms | 0.938 ms | 6.5x |
| photon coeff: pair compton em. | 5.044 ms | 0.540 ms | 9.3x |
| calc. coeff. pairs | 2.805 ms | 0.077 ms | 36.5x |
| evolve pairs | 1.001 ms | 0.171 ms | 5.9x |
| photon coeff: pair syn. em. | 0.702 ms | 0.144 ms | 4.9x |
| calc. coeff. proton/neutron | 0.559 ms | 0.223 ms | 2.5x |
| calc. coeff. neutrino | 0.330 ms | 0.203 ms | 1.6x |
| evolve proton | 0.167 ms | 0.094 ms | 1.8x |
| photon coeff: pair syn. abs. | 0.126 ms | 0.031 ms | 4.0x |
| photon coeff: proton compton em. | 0.095 ms | 0.124 ms | 0.8x |
| calc. coeff. muon | 0.089 ms | 0.011 ms | 7.7x |
| evolve photon | 0.058 ms | 0.052 ms | 1.1x |
| calc. coeff. pion | 0.052 ms | 0.055 ms | 0.9x |
| photon coeff: pair compton abs. | 0.050 ms | 0.050 ms | 1.0x |
| photon coeff: photo-pion loss | 0.044 ms | 0.015 ms | 3.0x |
| photon coeff: proton syn. em. | 0.043 ms | 0.023 ms | 1.8x |
| photons -> had. grid | 0.041 ms | 0.001 ms | 38.9x |
| photon coeff: pi0 decay | 0.039 ms | 0.009 ms | 4.5x |
| evolve neutrino | 0.022 ms | 0.026 ms | 0.8x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of stages** | **17.45 ms** | **2.85 ms** | **6.1x** |

`leptohadronic_all` (every process on):

| Stage                                        | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
| -------------------------------------------- | ---------------- | ------------------- | -------- |
| calc. coeff. photon                          | 6.570 ms         | 1.339 ms            | 4.9x     |
| photon coeff: pair compton em.               | 5.070 ms         | 0.616 ms            | 8.2x     |
| calc. coeff. pairs                           | 2.844 ms         | 0.154 ms            | 18.5x    |
| evolve muon                                  | 2.144 ms         | 0.186 ms            | 11.5x    |
| evolve pairs                                 | 1.509 ms         | 0.222 ms            | 6.8x     |
| evolve pion                                  | 1.144 ms         | 0.132 ms            | 8.7x     |
| photon coeff: pair syn. em.                  | 0.681 ms         | 0.201 ms            | 3.4x     |
| calc. coeff. proton/neutron                  | 0.557 ms         | 0.240 ms            | 2.3x     |
| calc. coeff. neutrino                        | 0.441 ms         | 0.252 ms            | 1.8x     |
| calc. coeff. pion                            | 0.250 ms         | 0.102 ms            | 2.5x     |
| evolve proton                                | 0.171 ms         | 0.143 ms            | 1.2x     |
| photon coeff: pair syn. abs.                 | 0.125 ms         | 0.029 ms            | 4.4x     |
| photon coeff: muon compton em.               | 0.124 ms         | 0.080 ms            | 1.5x     |
| calc. coeff. muon                            | 0.121 ms         | 0.017 ms            | 7.1x     |
| photon coeff: proton compton em.             | 0.101 ms         | 0.098 ms            | 1.0x     |
| photon coeff: pion compton em.               | 0.092 ms         | 0.079 ms            | 1.2x     |
| photon coeff: pair annihil.                  | 0.087 ms         | 0.064 ms            | 1.4x     |
| evolve photon                                | 0.081 ms         | 0.071 ms            | 1.1x     |
| photon coeff: pi0 decay                      | 0.077 ms         | 0.032 ms            | 2.4x     |
| photon coeff: pair compton abs.              | 0.051 ms         | 0.053 ms            | 1.0x     |
| photon coeff: photo-pion loss                | 0.044 ms         | 0.015 ms            | 2.9x     |
| photon coeff: proton syn. em.                | 0.042 ms         | 0.029 ms            | 1.4x     |
| photons -> had. grid                         | 0.040 ms         | 0.001 ms            | 37.6x    |
| photon coeff: pion syn. em.                  | 0.038 ms         | 0.021 ms            | 1.8x     |
| photon coeff: muon syn. em.                  | 0.037 ms         | 0.021 ms            | 1.8x     |
| evolve neutrino                              | 0.032 ms         | 0.043 ms            | 0.7x     |
| (stages below 0.02 ms in both codes omitted) |                  |                     |          |
| **total of stages**                          | **22.50 ms**     | **4.26 ms**         | **5.3x** |

The stage times don't add up exactly to the benchmark numbers, because the profiler itself adds a little overhead. A few very small stages (at or below about 0.1 ms) are slightly slower in am3_mojo; that is the cost of handing work to the thread pool.

### What makes am3_mojo faster

Most of the gain comes from better algorithms, not only from threads. On one thread am3_mojo is already 1.6–4.0× faster than the C++ AM³. The improvements, ordered by their effect:

1. **Inverse-Compton emission, from a 100 MB kernel to a 4 MB table** (the "photon coeff: pair compton em." stage).
   - **C++ AM³:** stores the emission kernel `K[k](j, i)` (outgoing photon × electron × target photon) as ~26 million float32 values, about 100 MB, and streams all of it through a matrix-vector product every step. It is limited by memory bandwidth.
   - **What am3_mojo exploits:** on AM³'s logarithmic energy grids every branch of the cross section (Thomson, full, Klein–Nishina and their bin averages) has the form `1/(γ_j² ε_i) · Φ(i + j, k − j)`. The 3D kernel therefore factorises into a 2D table Φ of about 4 MB, which stays in cache.
   - **Self-check:** at initialisation am3_mojo compares the factorised form with every entry of the original kernel. The few entries it doesn't reproduce to 10⁻¹² are stored exactly, as per-bin exceptions: the i = k diagonal, whose inclusion depends on round-off in AM³, and the Klein–Nishina edge terms.
   - **Computation:** for each `d = k − j` the emission is a Hankel product, computed by a register-blocked SIMD kernel.
2. **Decay and photo-pion kernels as Toeplitz/Hankel products** (the "calc. coeff. neutrino", "calc. coeff. muon" and "calc. coeff. proton/neutron" stages).
   - **What am3_mojo exploits:** pion and muon decay kernels depend only on an energy ratio, and photo-pion kernels only on an energy product (γ_p ε). On a logarithmic grid these matrices are Toeplitz (constant along diagonals) or Hankel (constant along anti-diagonals).
   - **Detection:** am3_mojo detects this structure once per matrix, verified to 10⁻¹², and stores the ~rows + cols diagonal values instead of rows × cols entries.
   - **Computation:** every product becomes a SIMD convolution, 4 × 8 output rows accumulated in registers. The 51 photo-pion channels are computed as independent Hankel products in parallel.
3. **Bethe–Heitler pair emission and the pair sources** (the "calc. coeff. pairs" stage, 18–37× faster).
   - **C++ AM³:** streams its float32 Bethe–Heitler kernel every step.
   - **am3_mojo:** stores the kernel as packed windows of its non-zero rows, so it is read sequentially; skips the proton bins that are empty; and splits the emission over the worker threads. The pair-production and muon-decay sources use the Toeplitz products of point 2.
4. **Cooling rates computed once per species group** (the "evolve pion" and "evolve muon" stages).
   - **C++ AM³:** runs one solver per component: 6 pion solvers and 12 muon solvers. Every call recomputes the synchrotron and inverse-Compton cooling rates of the particle, although they are the same for all components of a species.
   - **am3_mojo:** computes them once per step and species and shares them. The species updates are described as tables and run in parallel.
5. **SIMD everywhere.** The matrix-vector products, the IC and photo-pion kernels and the IC cooling rates are written with explicit SIMD vectors (8 × float64). Output rows are accumulated in registers in blocks of 32 rows, with 8-wide tail blocks, and all-zero rows and columns of every kernel are skipped.
6. **Parallelism on the Mojo worker pool.** These run in parallel:
   - IC and Bethe–Heitler emission;
   - the photo-pion channels;
   - the full pair-production source;
   - all independent species updates;
   - kernel construction (Bethe–Heitler, quantum synchrotron, pair production).

   Each output element is computed by exactly one worker in a fixed order, so results are bitwise identical for any thread count. Kernel set-up is 3.2–3.4× faster than in C++ for the hadronic models.
7. **Memory: 0.34 GB instead of 3.38 GB** (blazar model).
   - The IC table and the Toeplitz/Hankel forms replace the large dense kernels.
   - The full γγ pair kernel (1.9 GB) is allocated only when that treatment is switched on.
   - Densities are updated in place; there is no per-step allocation.
   - Mojo's ownership model (borrowed `read`/`mut` arguments, `^` moves) avoids copies. The worker-pool helper borrows its job, so the compiler guarantees the job outlives the parallel section.
8. **Interpolation weights.** The interpolation weights between the hadron and lepton grids are computed once per grid instead of recomputing a `log` and a `floor` per bin and call. This makes the "photons -> had. grid" stage 38× faster.
9. **Numerics.** Transcendental functions use the platform libm, as the C++ build does. In the few ill-conditioned expressions of AM³ the fused multiply-adds of the C++ build are reproduced explicitly, so results stay well within 10⁻⁵. These are the IC cooling function at γ ≈ 1, the quantum-synchrotron kernel and the full γγ pair kernel.

**Array lengths and SIMD.** The grid sizes (314 hadron, 340 lepton, 464 neutrino and 620 photon bins at 10 bins per e-fold) come from the physics, not from SIMD. With 8-wide tail blocks, a 314 × 314 product runs within 20 % of a padded 320 × 320 one per element, so the arrays are not padded.

## Differences from the C++ original

am3_mojo keeps the physics and the interface of AM³ v1.0.1. It differs in these points.

**Bugs of AM³ v1.0.1 that am3_mojo fixes:**
- **Nondeterminism.** `PionDecay::initialize_kernels` reads `egrid_had[j+1]` past the end of the grid for the top hadron bin, and hadron IC cooling reads photon bins past the photon grid. With protons at the top of the energy grid, about 1 in 8 fresh `init_kernels()` calls of the C++ AM³ gives corrupted spectra (up to 100 % off). am3_mojo uses the next grid point and empty bins instead. The reference tests use a C++ build with these two reads fixed wherever this regime is reached.
- **OpenMP data race.** The OpenMP build of AM³ races on `+=` in the photo-pion source terms. am3_mojo's parallel sections are race-free and deterministic.
- **Neutrino timescales.** `get_t_neutrino_escape` and `get_t_neutrino_extra_cooling` return the whole neutrino grid; AM³ returns only the hadron-grid length.
- **Neutrino sources.** With pion decay off and muon decay on, AM³ never resets the total ν_μ sources and adds to them every step. am3_mojo recomputes them.
- **Uninitialised flag.** `profile_timing` is uninitialised in AM³; in am3_mojo it defaults to `False`.

**Interface:**
- **Module name.** The module is `am3_mojo` (AM³: `am3`). `am3_mojo.AM3_API_VERSION` is `"1.0.1"`.
- **Booleans.** Switch getters return `bool`, and so does `get_verbosity_level()` (AM³: 0/1). Switch setters accept `True`/`False` and 0/1 and reject other values. Docstrings and the printed switch table say `True`/`False`.
- **Exceptions.** Errors are raised as Python `Exception` with AM³'s message text.
- **GSL errors.** GSL domain errors abort the AM³ process; am3_mojo reports them from the solver.

**Kept as in AM³, because they shape the results:**
- All pion components are evolved with identically configured π⁺ solvers.
- The muon and neutrino injection arrays are paired exactly as in AM³.
- The `optimize_*_grid` switches are C++ `bool`s that always read as `True`. Their "every 2nd grid point" code paths are unreachable in AM³ and are not implemented.

## License, credits and citation

am3_mojo is distributed under the BSD 3-Clause license of AM³ (`LICENSE`), which retains the copyright notice of the original authors. The physics, numerical scheme and interface are those of AM³; the Mojo implementation, its optimisations and the test suite are this repository's.

If you use am3_mojo, please cite **both** the AM³ paper and this repository:

- M. Klinger, A. Rudolph, X. Rodrigues, C. Yuan, G. Fichet de Clairfontaine, A. Fedynitch, W. Winter, M. Pohl, S. Gao, *AM³: An Open-Source Tool for Time-Dependent Lepto-Hadronic Modeling of Astrophysical Sources* (2023), [arXiv:2312.13371](https://arxiv.org/abs/2312.13371)
- Chen Li, *am3_mojo: the AM³ lepto-hadronic interactions code in Mojo*, version 0.1.0 (2026), https://github.com/chenlinear/am3_mojo

```bibtex
@misc{klinger2023am3,
  author        = {Klinger, Marc and Rudolph, Annika and Rodrigues, Xavier and Yuan, Chengchao
                   and Fichet de Clairfontaine, Gaëtan and Fedynitch, Anatoli and Winter, Walter
                   and Pohl, Martin and Gao, Shan},
  title         = {{AM$^3$: An Open-Source Tool for Time-Dependent Lepto-Hadronic Modeling
                   of Astrophysical Sources}},
  year          = {2023},
  eprint        = {2312.13371},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.HE}
}

@software{am3_mojo,
  author  = {Chen Li},
  title   = {{am3\_mojo: the AM$^3$ lepto-hadronic interactions code in Mojo}},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/chenlinear/am3_mojo}
}
```

The same information is in [`CITATION.cff`](CITATION.cff), from which GitHub and GitLab offer a "Cite this repository" entry in APA and BibTeX format.

The numerical treatment goes back to:

- S. Gao, M. Pohl, W. Winter, *On the direct correlation between gamma-rays and PeV neutrinos from blazars* (2016), [arXiv:1610.05306](https://arxiv.org/abs/1610.05306)

The original AM³ code is at https://gitlab.desy.de/am3/am3 and is documented at https://am3.readthedocs.io/en/latest/.
