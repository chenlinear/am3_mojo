# Solver checks

These scripts check AM³'s solver beyond the test suite. They print numbers to be read, not pass/fail results. [`tests/reference/README.md`](../reference/README.md) and the [Results](../literature_checks/README.md#results) of the literature checks quote them. Each script's docstring gives its set-up, what it prints and its options.
- `regime_boundary.py`: particles that cool into a solver window from far above. With `--impl`, it runs the C++ builds of `make reference`.
- `window_top_undershoot.py`: the Crank–Nicolson undershoot at the top cell of a Chang & Cooper window after a sharp pulse.
- `triplet_below_grid.py`: triplet pairs whose lab energy falls below the lepton grid.
- `triplet_steady_states.py`: time stepping where triplet pair production dominates the cooling.

`make checks` runs them with the [literature checks](../literature_checks/README.md) and writes their outputs to `build/checks/`. To run one, from the repository root after `make build`:

```bash
.venv/bin/python tests/solver_checks/regime_boundary.py
```
