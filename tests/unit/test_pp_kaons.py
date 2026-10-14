"""Kaons from pp collisions (src/am3_mojo/physics/proton_proton.mojo) in a pp-dominated source: they are made, decay
into the pp components and carry energy away from the protons (smoothly across 100 GeV; nothing from protons whose
kaons would fall below the grid). With kaons off, none are made.
tests/literature_checks/derivations/kaon_production.py checks the code's parameters against the README's
(pp.code_tables), and the README's spectra against Koers, Pe'er & Wijers 2006 (arXiv:hep-ph/0611219) and NA49.
"""
import numpy as np
import pytest

from support.setups import GRID_TO_1E21_EV, PROTONS_TO_1E20_EV, new


def pp_source(kaons=True, steps=20):
    a = new(grid=GRID_TO_1E21_EV, photopion=False, pp=True, kaons=kaons)
    a.set_pp_target_proton_density(1e10)
    a.set_mag_field(10.0)
    a.set_escape_timescale(1e5)
    a.set_solver_time_step(1e2)
    a.set_powerlaw_injection_parameters_protons(*PROTONS_TO_1E20_EV)
    for _ in range(steps):
        a.evolve_step()
    return a


@pytest.fixture(scope='module')
def runs():
    return {'on': pp_source(True), 'off': pp_source(False)}


def test_pp_kaons_in_a_source(runs):
    on, off = runs['on'], runs['off']
    E = on.get_egrid_had()
    for g in ('get_kaon_plus', 'get_kaon_minus', 'get_kaon_0S', 'get_kaon_0L'):
        assert getattr(on, g)().max() > 0, g
        assert not getattr(off, g)().any(), g
    # kaons a few per cent of the charged pions (energy weighted; 0.03 in this source)
    frac = np.sum(on.get_kaons() * E) / np.sum(on.get_pions() * E)
    assert 0.02 < frac < 0.3, frac
    # their decay products are in the pp components: more pp neutrinos at the highest energies, where the pions
    # cool before they decay
    nu_on, nu_off = on.get_muon_neutrinos_proton_proton(), off.get_muon_neutrinos_proton_proton()
    En = on.get_egrid_neutrinos()
    top = En > 10 * En[np.argmax(nu_off * En)]
    assert np.max(nu_on[top] / np.maximum(nu_off[top], 1e-300)) > 1.5
    # protons lose the kaon energy too: faster pp cooling, by 4-20 % (14-16 % in this source)
    r = off.get_t_proton_proton_proton() / on.get_t_proton_proton_proton()
    fin = np.isfinite(r) & (on.get_t_proton_proton_proton() < 1e199) & (E > 200e9)
    assert fin.sum() > 50
    assert np.all((r[fin] > 1.04) & (r[fin] < 1.2)), r[fin].min()
    # below 100 GeV (the delta-functional approximation), kaons below the lowest hadron bin (1 GeV), from protons
    # below about 7 GeV, are neither made nor charged to the protons. Above, the kaons' share rises smoothly into
    # that of the spectra above 100 GeV.
    assert np.all(r[E < 6e9] == 1.0)
    mid = (E > 6.5e9) & (E < 1.2e11)
    assert r[mid][0] > 1.0 and np.all(np.diff(r[mid]) > 0), r[mid]
