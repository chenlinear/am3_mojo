"""Time stepping of triplet pair production: steady states in which it dominates the cooling, against the exact
balances (the limitations of src/am3_mojo/physics/triplet.mojo: time stepping; a figure in tests/reference/README.md,
"Particles that cool into a window from far above").

Electrons are injected per ln E as 1e9/E from 1e9 to 1e12 eV, photons at 0.03 m_e c^2, n_gamma per t_esc
(densities of about n_gamma = 3e15, 1e16 and 1e17 cm^-3). Both escape in t_esc = 1e4 s; triplet pair production
is the only other process. Time steps: 0.03, 0.1, 0.3, 1 and 10 t_esc. Prints the deviations from the balances:
  - "number": each interaction absorbs one photon and makes one e+, so the e+ escape at the rate photons are absorbed;
  - "energy": the leptons lose energy only to escape, so they carry off the injected power (the process neglects
    the target photons' energy, below m_e c^2).
AM3's solver causes the deviations: its time steps, its Chang & Cooper regime (the leptons lose
(1 - e^-dlnE)/dlnE = 0.952 of the energy the pairs take) and its regimes 3 and 4 at large steps.

usage: steady_states.py   (a few seconds)"""
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.setups import energy, number, per_ln_e, triplet_steady

T_ESC = 1e4


def steady_state(n_gamma, dtf):
    """The deviations (number, energy) of the steady state at dt = dtf t_esc, and its largest triplet loss rate over
    the escape rate."""
    a = triplet_steady(n_gamma, dtf * T_ESC, int(max(60, 30 / dtf)), T_ESC)
    E = a.get_egrid_lep()
    q = per_ln_e(E)
    npos = number(a.get_positrons())
    absorbed = number(np.array(a.get_photons()) / np.array(a.get_t_photon_triplet()))
    escaping = (energy(a.get_electrons(), E) + energy(a.get_positrons(), E)) / T_ESC
    loss = (T_ESC / np.array(a.get_t_pair_triplet())).max()
    return npos / T_ESC / absorbed - 1, escaping / energy(q, E) - 1, loss


for n_gamma in [3e15, 1e16, 1e17]:
    for dtf in [0.03, 0.1, 0.3, 1.0, 10.0]:
        dn, de, loss = steady_state(n_gamma, dtf)
        print(f'n_gamma = {n_gamma:.0e} cm^-3, dt = {dtf:g} t_esc: number {dn:+.1e}, energy {de:+.2e} '
              f'(triplet loss up to {loss:.3g} times the escape rate)', flush=True)
