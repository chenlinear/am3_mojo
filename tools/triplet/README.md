# Triplet pair production tables

These scripts compute the tables of triplet pair production, $\mathrm{e}^\pm\gamma$ $\to \mathrm{e}^\pm\mathrm{e}^+\mathrm{e}^-$, in `src/am3_mojo/physics/triplet_tables.mojo`, which `src/am3_mojo/physics/triplet.mojo` uses. The [triplet pair production page](../../tests/literature_checks/triplet/README.md) of the literature checks describes the method and derives the equations that use the tables. Each script's docstring says how it works:
- `qed_triplet.py`: the lowest-order QED matrix element, from all eight Feynman diagrams;
- `kinematics.py`: the three-body phase space in the rest frame of the target electron;
- `produce.py`: the cross section and the distributions of the products' energy fractions at 60 photon energies, integrated with VEGAS;
- `make_tables.py`: the Mojo tables from the output of `produce.py`.

To regenerate the tables after `uv sync`, which installs `vegas` and `gvar` with the rest of the [development set-up](../../README.md#development-set-up):

```bash
.venv/bin/python tools/triplet/produce.py results/tools/triplet
.venv/bin/python tools/triplet/make_tables.py results/tools/triplet src/am3_mojo/physics/triplet_tables.mojo
```

Runs are seeded, but the tables in `src/` come from an unseeded run. A new run therefore reproduces them within their Monte Carlo errors, not bit for bit. The scripts of [tests/literature_checks/triplet/](../../tests/literature_checks/triplet/README.md) check these tables and the QED code, at the steps of the page that use them.
