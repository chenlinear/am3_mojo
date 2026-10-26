"""Triplet pairs made below the lepton grid (the limitations of src/am3_mojo/physics/triplet.mojo: soft photons
only; figures in tests/literature_checks/triplet/README.md, "On the grid").

In the soft-photon approximation a product takes the lab energy x gamma, a fraction x of the lepton's. On target
photons above about 0.25 m_e c^2, some products fall below the lepton grid's lowest bins, whatever gamma. They go
into bin 0, which AM3's solver empties every step, like bin 1 (gamma < e^0.15 = 1.16). Their photons are absorbed
and the lepton loses their energy all the same. Prints:
  1. one short step (1e-3 s) of electrons at gamma = 1e2 and 1e4 on photons at eps = 0.03 ... 1 m_e c^2 (one bin
     each), with only triplet pair production: per interaction (the code's rate R(gamma eps)), the positrons made
     and the photons absorbed, and the shares of the positrons and of the pair energy put in bins 0 and 1;
  2. leptohadronic_all at the end of its run (the cached default run of `make plots`): the share of the triplet
     interactions on photons above 0.25 m_e c^2, and of the positrons put in bins 0 and 1.
Rates and deposits are the code's kernel rows of the default grid, read through tests/harness/triplet_mojo.mojo.

usage: below_grid.py   (about a minute; part 2 runs the scenario first if it is not cached)"""
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.harness import Triplet
from support.runs import cached_scenario, checkpoints
from support.setups import line, number, triplet_box
from support.units import DLNE, MEC2_EV, SIGMA_T_C

ROWS = Triplet().rows()   # the code's kernel rows: P = gamma eps = exp((m0 + q) dlnE)


def row(P):
    """The kernel row of P = gamma eps on the grid (negative below the first)."""
    return np.rint(np.log(P) / ROWS['dl']).astype(int) - ROWS['m0']


def rate(P):
    """The code's rate R(gamma eps) [sigma_T], 0 below the threshold."""
    q = row(np.asarray(P, float))
    return np.where(q >= 0, ROWS['R'][np.maximum(q, 0)], 0.0)


def below(gamma, eps):
    """Shares of the positrons (number) and of the pair energy that the code deposits in bins 0 and 1 (offsets
    o >= i - 1), for leptons of Lorentz factor gamma (bin i) on isotropic photons of energy eps."""
    q = int(row(gamma * eps))
    if q < 0:
        return 0.0, 0.0
    i = int(np.rint(np.log(gamma) / ROWS['dl']))
    pos, low = ROWS['W'][0][q], ROWS['W'][1][q]
    x = np.exp(-np.arange(len(pos)) * ROWS['dl'])
    lost = np.arange(len(pos)) >= i - 1
    return pos[lost].sum() / pos.sum(), ((pos + low) * x)[lost].sum() / ((pos + low) * x).sum()


def one_step(gamma, eps, ne=1e8, nph=1e14, dt=1e-3):
    a, (e0, ph0) = triplet_box(line(ne, gamma * MEC2_EV), pho=line(nph, eps * MEC2_EV), dt=dt)
    ie, ip = np.argmax(e0), np.argmax(ph0)
    g, e = a.get_egrid_lep()[ie] / MEC2_EV, a.get_egrid_photons()[ip] / MEC2_EV
    interactions = ne * DLNE * nph * DLNE * rate(g * e) * SIGMA_T_C * dt
    absorbed = (nph - a.get_photons()[ip]) * DLNE
    return g, e, number(a.get_positrons()) / interactions, absorbed / interactions


print('1. one short step: per interaction (the code\'s rate), the positrons made and the photons absorbed;'
      ' the shares of the positrons and of the pair energy the code deposits in bins 0 and 1')
for eps in [0.03, 0.1, 0.3, 0.5, 1.0]:
    for gamma in [1e2, 1e4]:
        g, e, made, absorbed = one_step(gamma, eps)
        n_low, e_low = below(g, e)
        print(f'  eps = {e:.3f}, gamma = {g:.4g}: positrons made {made:.4f}, photons absorbed {absorbed:.4f};'
              f' below the grid {n_low:.4f} of the positrons, {e_low:.2e} of the pair energy', flush=True)

D = cached_scenario('mojo', 'leptohadronic_all')
last = checkpoints(D)[-1]
get = lambda name: D[f'{last:06d}/{name}']
g, e = get('get_egrid_lep') / MEC2_EV, get('get_egrid_photons') / MEC2_EV
lep, ph = get('get_electrons') + get('get_positrons'), get('get_photons')
J = np.flatnonzero((e <= 1.0 + 1e-6) & (ph > 0))
I = np.flatnonzero(lep > 0)
tot = hard = lost = 0.0
for j in J:
    r = lep[I] * rate(g[I] * e[j]) * ph[j]
    tot += r.sum()
    if e[j] > 0.25:
        hard += r.sum()
    lost += sum(ri * below(gi, e[j])[0] for ri, gi in zip(r, g[I]) if ri > 0)
print(f'2. leptohadronic_all at step {last}: {hard / tot:.2f} of the triplet interactions on photons above'
      f' 0.25 m_e c^2; {lost / tot:.3f} of the positrons deposited in bins 0 and 1')
