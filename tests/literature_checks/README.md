# Literature checks

This folder derives the five processes that am3_mojo adds to AM³, from the papers they come from to the Mojo code. The main README lists them, with their switches, under [Processes am3_mojo adds to AM³](../../README.md#added-processes). Nothing here is part of the test suite (`make test`).

## What it holds

Each process has a folder with its README, the derivation checks for that README and the scripts that its tables need. The README starts with the equations and tables of the cited papers. From them, the README derives the equations and tables that the Mojo code uses. It also names the struct or function that uses each of them.

- [annihilation/](annihilation/README.md): electron–positron annihilation, checked by `annihilation.py`.
- [triplet/](triplet/README.md): triplet pair production, checked by `triplet.py`. The folder's scripts check the tables of the QED Monte Carlo against the exact cross section and measure the code's approximations.
- [neutron_decay/](neutron_decay/README.md): neutron decay, checked by `neutron_decay.py`.
- [kaons/](kaons/README.md): the production, evolution and [decays](kaons/README.md#decays) of the kaons, checked by `kaon_production.py` and `kaon_decays.py`. The folder's scripts derive the code's tables from measured data.
- [charm/](charm/README.md): the production, evolution and [decays](charm/README.md#decays) of the charm hadrons, checked by `charm_production.py` and `charm_decays.py`. `pdg_charm.py` transcribes the [PDG](https://pdg.lbl.gov/2026/) listings of the decays.
- `common/`: the code that the derivation checks share. `latex.py` reads the pages and the sources' TeX, and parses their LaTeX with sympy. `verify.py` registers the checks and holds their tests. `run.py` runs the checks. `inventory.py` lists the equations. `mojo_constants.py` reads the constants of `src/am3_mojo/utils/constants.mojo`.

## How the derivation checks work

A derivation check finds its equation on the process README by a pin, a piece of its LaTeX that occurs exactly once on that page. It proves the equation from its source: exactly with sympy where it can, otherwise with mpmath to 50 digits at random points in every branch. A step that the code chooses, such as a quadrature or a grid, has no source to derive it from. Its check derives what follows from the step instead: a count, a conservation law, a limit or a continuity. Each check also has a control, a broken variant of the equation that must fail, so that a check that cannot fail is noticed. An edited equation thus fails its check.

The source's side comes from the paper's TeX in `knowledge/papers/`, a local folder that git ignores. Without that folder, the comparison with the TeX passes as "not re-checked". With the folder, a source that is missing from it fails its check. A source without TeX is transcribed in the module, with its page and equation numbers. The masses and constants come from the Mojo code. The checks do not compare the pages with the code's values: the unit tests and the scripts do.

`inventory.py` lists every equation of the pages with its source, its relation to the source and its check. If a relation of a display equation has no pin on its own page, the inventory fails. If a pin is not found exactly once, or if an equation has no check, the inventory also fails.

The checks found misprints in these sources. The pages correct the misprints:
- [Svensson (1982)](https://ui.adsabs.harvard.edu/abs/1982ApJ...258..321S), eq. 6 ([annihilation](annihilation/README.md#cross-section)).
- [Lee, Lyubyakin & Stotsky (2021)](https://doi.org/10.1007/JHEP01%282021%29144), eqs. 4.3 and 4.5 ([triplet pair production](triplet/README.md#cross-section)).
- [Kelner, Aharonian & Bugayov (2006)](https://arxiv.org/abs/astro-ph/0606058), eq. 76 ([kaons](kaons/README.md#the-kaon-spectra)).
- [Lipari, Lusignoli & Meloni (2007)](https://arxiv.org/abs/0704.0718), eq. 19 ([kaon decays](kaons/README.md#muon-helicity)).

## Running them

To run the derivation checks, the inventory, every script and `tools/ky_angular_extract.py`, run `make literature_checks` (`run_literature_checks.py`). The command writes their outputs to `results/literature_checks/`. It writes the output of the tool to `results/tools/`. The checks and scripts need scipy, mpmath, sympy and vegas. `uv sync` installs these packages with the rest of the [development set-up](../../README.md#development-set-up). Run the commands below from the repository root:

```bash
.venv/bin/python tests/literature_checks/common/run.py                  # every derivation check
.venv/bin/python tests/literature_checks/common/run.py kaon_decays      # the checks of one module
.venv/bin/python tests/literature_checks/common/inventory.py
.venv/bin/python tests/literature_checks/triplet/compare_exact.py       # a script
```

A module's docstring names its sources. Each check's docstring says what the check proves. A script's docstring says what it checks, from which data and with which options.
