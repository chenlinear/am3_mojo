# Triplet pair production tables

These scripts compute the tables of triplet pair production, $\mathrm{e}^\pm\gamma$ $\to \mathrm{e}^\pm\mathrm{e}^+\mathrm{e}^-$, in `src/am3_mojo/physics/triplet_tables.mojo`, which `src/am3_mojo/physics/triplet.mojo` uses. The main README's section [Triplet pair production](../../README.md#triplet-pair-production) describes the method, and each script's docstring says how it works:
- `qed_triplet.py`: the lowest-order QED matrix element, from all eight Feynman diagrams;
- `kinematics.py`: the three-body phase space in the rest frame of the target electron;
- `produce.py`: the cross section and the distributions of the products' energy fractions at 60 photon energies, with VEGAS;
- `make_tables.py`: the Mojo tables from the output of `produce.py`.

To regenerate the tables, after `uv sync` (which installs `vegas` and `gvar` with the rest of the [development set-up](../../README.md#development-set-up)):

```bash
.venv/bin/python tools/triplet/produce.py tables
.venv/bin/python tools/triplet/make_tables.py tables src/am3_mojo/physics/triplet_tables.mojo
```

All 60 photon energies take about 40 minutes when 8 processes share them. A run is seeded, but the tables in `src/` come from an unseeded one, so a new run reproduces them within their Monte Carlo errors, not bit for bit. The scripts of [`tests/literature_checks/triplet/`](../../tests/literature_checks/README.md#independent-checks) check these tables and the QED code.
