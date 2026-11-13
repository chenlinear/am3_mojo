"""The size of approximations that am3_mojo keeps from AM3: the figures of
tests/reference/README.md ("Kept as in AM3") for the pp photons beyond the range
of their parameterization, the decay products of a step, the pairs of the
eight-bin gamma-gamma approximation and the hadrons' inverse-Compton
emission. The rows of the full gamma-gamma kernel also test a fix: the energy
of its pairs (tests/reference/README.md)."""

import numpy as np
import pytest

from support.output import quiet
from support.setups import (
    ADDED_LEPTONIC,
    CLOSED_BOX,
    energy,
    evolve,
    new,
    number,
    power_law,
)

# a process that is switched on acts alone beside the decays
BOX = (
    "photopion",
    "pp",
    "bethe_heitler",
    "escape",
    "expansion",
    "adiabatic_cooling",
    "electron_syn",
    "electron_compton",
    "annihilation",
    "proton_syn",
    "proton_compton",
) + ADDED_LEPTONIC


@pytest.mark.parametrize(
    "model, e_min, ratio",
    [
        (0, 1e12, 0.49),
        (0, 1e14, 0.53),
        (0, 1e16, 0.71),
        (0, 1e17, 1.00),
        (0, 1e18, 1.61),
        (2, 1e14, 0.48),
        (2, 1e16, 1.11),
        (2, 1e18, 4.42),
    ],
)
def test_pp_photons_beyond_the_range_of_their_parameterization(
    model, e_min, ratio
):
    """The pi0 photons of pp collisions (the direct treatment, the default)
    against the energy that the protons lose in them
    (get_t_proton_proton_proton). AM3 takes the photon spectrum of Kafexhiu et
    al. (2014), given up to 1 PeV, on the whole grid. The photons alone have
    more than the energy lost from 1e17 eV with SIBYLL (model 0), and from
    1e16 eV with Geant4 (model 2)."""
    a = new(off=BOX, pp=True, dt=1e-3, muon_decay=False, init=False)
    a.set_pp_model_neutral_pions(model)
    quiet(a.init_kernels)
    a.set_pp_target_proton_density(1e6)
    Eh, Eg = a.get_egrid_had(), a.get_egrid_photons()
    protons = power_law(1.0, e_min, 10 * e_min)(Eh)
    a.set_current_densities_protons(protons)
    a.evolve_step()
    lost = 1e-3 * energy(protons / a.get_t_proton_proton_proton(), Eh)
    assert energy(a.get_photons(), Eg) / lost == pytest.approx(ratio, abs=0.01)


def test_decay_products_come_from_the_parents_left_after_a_step():
    """The decay source of a step is n dt / tau of the parents at its end. One
    step of 1 s leaves no pions and makes no muons. 1000 steps of 1e-3 s give a
    muon to 2.35 % of the pions, and 1000 steps of 1e-5 s give one to 88 %. If
    the pions are injected at a steady rate, each pion gives a muon, in the
    first step of 1 s and over 10 steps."""
    pions = power_law(1.0, 1e11, 1e13)

    def muons_per_pion(dt, steps, injected=False):
        a = new(off=BOX, dt=dt, muon_decay=False)
        n = pions(a.get_egrid_had())
        if injected:
            a.set_injection_rate_pion_plus(n)
        else:
            a.set_current_densities_pion_plus(n)
        evolve(a, steps)
        if not injected:
            assert number(a.get_pions()) < 1e-4 * number(n)
        return (
            number(a.get_muons()) / number(n) / (dt * steps if injected else 1)
        )

    assert muons_per_pion(1.0, 1) == 0
    assert muons_per_pion(1e-3, 1000) == pytest.approx(0.0235, rel=0.01)
    assert muons_per_pion(1e-5, 1000) == pytest.approx(0.878, rel=0.01)
    assert muons_per_pion(1.0, 1, injected=True) == pytest.approx(1, rel=1e-9)
    assert muons_per_pion(1.0, 10, injected=True) == pytest.approx(1, rel=1e-9)


@pytest.mark.parametrize("x", [0.1, 1.0, 10.0])
def test_decay_products_of_an_injection(x):
    """Pions are injected at a steady rate, and the time step is x times their
    decay time. The source of a step takes the pions at its end. So after the
    first step, the pions and muons hold (1 - e^-x)(1 + 1/x) times the pions
    injected (1.26 at x = 1). Once the injection stops after 20 steps and the
    pions are gone, the muons are the pions injected."""
    a = new(off=BOX, dt=1.0, muon_decay=False)
    E = a.get_egrid_had()
    k = int(np.argmin(np.abs(np.log(E / 5e12))))
    dt = x * a.get_t_pion_decay()[k]
    a.set_solver_time_step(dt)
    q = np.zeros(E.size)
    q[k] = 1.0
    a.set_injection_rate_pion_plus(q)
    a.evolve_step()
    held = number(a.get_pions() + a.get_muons()) / (number(q) * dt)
    assert held == pytest.approx((1 - np.exp(-x)) * (1 + 1 / x), rel=1e-6)
    evolve(a, 19)
    a.set_injection_rate_pion_plus(np.zeros(E.size))
    evolve(a, int(60 / x) + 1)
    assert number(a.get_pions()) < 1e-12 * number(q) * dt
    assert number(a.get_muons()) / (number(q) * 20 * dt) == pytest.approx(
        1, rel=1e-6
    )


@pytest.mark.parametrize(
    "eight_bin, e_min, e_max, in_energy, in_number",
    [
        (True, 1e5, 1e7, 0.726, 0.524),
        (True, 1e3, 1e9, 0.991, 0.811),
        (False, 1e5, 1e7, 0.988, 0.955),
        (False, 1e3, 1e9, 1.000, 0.843),
    ],
)
def test_gamma_gamma_pairs_against_the_absorbed_photons(
    eight_bin, e_min, e_max, in_energy, in_number
):
    """The e+- of gamma-gamma pair production against the photons absorbed, in
    energy and in number (e+ and e- per photon). The eight-bin approximation
    (the default) makes no pairs of absorbed photons below 1.9 MeV. The full
    kernel takes the energy of all absorbed photons, less that of the lowest
    lepton bin, which the pair source leaves empty. Neither keeps the number.
    (AM3 gave the pairs of the full kernel the energy of the absorbed photons
    above m_e c^2 only: 0.939 instead of 0.988 for photons from 0.1 to 10 MeV,
    an AM3 bug.)"""
    a = new(
        off=CLOSED_BOX + ADDED_LEPTONIC, annihilation=True, dt=1.0, init=False
    )
    a.set_optimize_annihilation_pair_emission(eight_bin)
    quiet(a.init_kernels)
    El, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    photons = power_law(1e9, e_min, e_max)(Eg)
    for _ in range(2):
        a.set_current_densities_photons(photons)
        before = a.get_pairs()
        a.evolve_step()
    pairs, lost = a.get_pairs() - before, photons - a.get_photons()
    assert energy(pairs, El) / energy(lost, Eg) == pytest.approx(
        in_energy, abs=1e-3
    )
    assert number(pairs) / number(lost) == pytest.approx(in_number, abs=1e-3)


@pytest.mark.parametrize(
    "e_min, e_max, ratio",
    [(1e-3, 1e-1, 1.256), (1.0, 1e3, 0.705), (1e-4, 1e5, 1.002)],
)
def test_proton_compton_emission_against_cooling(e_min, e_max, ratio):
    """The energy of the photons that protons emit by inverse Compton
    (get_photons_protons_compton) against the energy that they lose by it
    (get_t_proton_compton). The emission has AM3's step cross section. The
    cooling has Jones' exact cooling function."""
    off = tuple(p for p in BOX if p != "proton_compton")
    a = new(off=off + ("pion_decay", "muon_decay"), dt=1.0)
    Eh, Eg = a.get_egrid_had(), a.get_egrid_photons()
    protons = power_law(1.0, 1e10, 1e20)(Eh)
    a.set_current_densities_protons(protons)
    a.set_current_densities_photons(power_law(1e10, e_min, e_max)(Eg))
    a.evolve_step()
    lost = energy(protons / a.get_t_proton_compton(), Eh)
    assert energy(a.get_photons_protons_compton(), Eg) / lost == pytest.approx(
        ratio, abs=2e-3
    )
