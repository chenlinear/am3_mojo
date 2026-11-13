"""Time stepping of triplet pair production: steady states in which it dominates
the cooling, against the exact balances (the time-stepping limitations of
src/am3_mojo/physics/triplet.mojo, and a figure in tests/reference/README.md,
"Particles that cool into a window from far above").

Electrons and photons escape in t_esc = 1e4 s. Triplet pair production is the
only other process. Prints the deviations from the balances:
  - "number": each interaction absorbs one photon and makes one e+, so the e+
    escape at the rate at which photons are absorbed.
  - "energy": the leptons lose energy only to escape, so they carry off the
    injected power (the process neglects the target photons' energy, below
    m_e c^2).
AM3's solver causes the deviations: its time steps, its Chang & Cooper regime
and its regimes 3 and 4 at large steps. In the Chang & Cooper regime, the
leptons lose (1 - e^-dlnE)/dlnE = 0.952 of the energy that the pairs take.

usage: steady_states.py   (a few seconds)"""

import os, sys

import numpy as np

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
from support.setups import energy, number, per_ln_e, triplet_steady

T_ESC = 1e4


def steady_state(n_gamma, dtf):
    a = triplet_steady(n_gamma, dtf * T_ESC, int(max(60, 30 / dtf)), T_ESC)
    E = a.get_egrid_lep()
    q = per_ln_e(E)
    npos = number(a.get_positrons())
    absorbed = number(
        np.array(a.get_photons()) / np.array(a.get_t_photon_triplet())
    )
    escaping = (
        energy(a.get_electrons(), E) + energy(a.get_positrons(), E)
    ) / T_ESC
    loss = (T_ESC / np.array(a.get_t_pair_triplet())).max()
    return npos / T_ESC / absorbed - 1, escaping / energy(q, E) - 1, loss


for n_gamma in [3e15, 1e16, 1e17]:
    for dtf in [0.03, 0.1, 0.3, 1.0, 10.0]:
        dn, de, loss = steady_state(n_gamma, dtf)
        print(
            f"n_gamma = {n_gamma:.0e} cm^-3, dt = {dtf:g} t_esc: number "
            f"{dn:+.1e}, energy {de:+.2e} (triplet loss up to {loss:.3g} "
            "times the escape rate)",
            flush=True,
        )
