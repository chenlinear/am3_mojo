"""Processes am3_mojo adds to AM3 (README "Processes am3_mojo adds to AM³"; tools/am3_mojo_api.json), checked
without the C++ code (their Python interface: tests/unit/test_api.py). The Mojo harnesses of tests/harness/ print the
code's kernels and tables, which the tests compare with the pages of tests/literature_checks/ that derive them:

* e+e- annihilation (Svensson 1982): each turns an e- and an e+ into two photons with their energy; the switches;
* triplet pair production (e+- gamma -> e+- e+ e-): kernels, the tabulated cross section against Lee et al.'s
  expansions, a step's products, steady-state balances, switches, the pair component;
* neutron decay (n -> p e- anti-nu_e): kernels, the decay rate and a step's products in a closed box, the switch,
  the components;
* kaons (K+-, K0_S and K0_L): the photo-hadronic channels against the measurements they come from, the kaons of pp
  collisions against the kaon page's fit and Koers et al.'s spectra; escape and cooling (K+- as charged pions, K0_S
  and K0_L as neutrons), decay, switches;
* charm hadrons (D+-, D0, anti-D0, Ds+-, Lambda_c+ and anti-Lambda_c-): the photo-hadronic steps against the charm
  page's table, the pp cross section, spectrum and kernels against its fits; escape, cooling and decay, switches;
* the decay tables of the kaons and the charm hadrons: energy, positivity and, for the charm hadrons, conserved
  numbers, CP conjugates and the semileptonic and tau rates;
* the neutrinos by parent.
The switches are on by default; `mojo-am3` (tests/support/paths.py) turns them off.
"""
import numpy as np
import pytest
import sympy as sp

from support import harness
from support.code_decays import DECAYS_MOJO, code_charm_modes, code_tau_modes
from support.measured import charm_pp
from support.measured import ky_angular as ky
from support.measured import strange_20gev as g20
from support.paths import load_derivation
from support.setups import (CLOSED_BOX, GRID_TO_1E21_EV, PAIR_COMPONENTS, PROTONS_TO_1E20_EV, energy, evolve,
                            hadronic_source, line, new, number, power_law, pp_source, pulse_source, triplet_box,
                            triplet_steady as steady_state)
from support.units import DLNE, MASS_EV, MASS_GEV, MEC2_EV, TAU_S


# ---------------------------------------------------------------- e+e- annihilation
def pair_box(ele, pos, steps=1, **switches):
    """A box where only e+e- annihilation acts; `ele`, `pos`: EdN/dEdV as functions of the lepton grid energy [eV];
    `steps` steps of 1 s."""
    a = new(off=CLOSED_BOX + ('triplet_pair_production',), dt=1.0, **switches)
    E = a.get_egrid_lep()
    a.set_current_densities_electrons(ele(E))
    a.set_current_densities_positrons(pos(E))
    n0 = (a.get_electrons(), a.get_positrons())
    return evolve(a, steps), n0


def slow(n):
    """Pairs in lepton bin 2 (gamma = e^0.2 = 1.221), the lowest bin AM3's solver evolves (it zeroes bins 0 and 1
    every step)."""
    return lambda E: np.where(np.arange(len(E)) == 2, n, 0.0)


@pytest.mark.parametrize('ele, pos', [
    (slow(1e12), slow(1e12)),
    (slow(1e12), power_law(1e10, 1e6, 1e10)),
    (power_law(1e11, 1e6, 1e9), power_law(1e11, 1e6, 1e12)),
    (power_law(1e11, 6.3e5, 1e7, 3.0), power_law(5e10, 6.3e5, 1e8)),
], ids=['slow-slow', 'slow-power law', 'power laws', 'steep power law'])
def test_conservation(ele, pos):
    """Each annihilation turns one electron and one positron into two photons with their energy (to 4e-4, from the
    solver's time stepping). Slow pairs (gamma = 1.221) make a line around 511 keV."""
    a, (e0, p0) = pair_box(ele, pos, steps=3)
    El, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    de = number(e0) - number(a.get_electrons())
    dp = number(p0) - number(a.get_positrons())
    g = a.get_photons_pair_annihilation()
    assert de > 0
    assert dp == pytest.approx(de, rel=1e-3)
    assert number(g) == pytest.approx(de + dp, rel=1e-3)
    lost = energy(e0, El) - energy(a.get_electrons(), El) + energy(p0, El) - energy(a.get_positrons(), El)
    assert energy(g, Eg) == pytest.approx(lost, rel=1e-3)
    np.testing.assert_array_equal(g, a.get_photons())  # the only photons
    if np.count_nonzero(e0) == np.count_nonzero(p0) == 1:  # slow pairs only
        inside = (Eg > 0.3 * MEC2_EV) & (Eg < 2.5 * MEC2_EV)
        assert np.sum(g[inside]) == pytest.approx(np.sum(g), rel=1e-6)


def test_annihilation_switches():
    box = lambda **switches: pair_box(slow(1e12), slow(1e12), **switches)
    a, (e0, _) = box(pair_annihilation_emission=False)
    assert not a.get_photons_pair_annihilation().any() and number(a.get_electrons()) < number(e0)
    a, (e0, _) = box(pair_annihilation_cooling=False)
    assert a.get_photons_pair_annihilation().any()
    np.testing.assert_array_equal(a.get_electrons(), e0)
    a, (e0, _) = box(pair_annihilation=False)
    assert not a.get_photons_pair_annihilation().any()
    np.testing.assert_array_equal(a.get_electrons(), e0)
    # positrons merged into the electrons: the charges are unknown, so no annihilation
    a, _ = box(merge_positrons_into_electrons=True)
    assert not a.get_photons_pair_annihilation().any()


# ---------------------------------------------------------------- triplet pair production
# the threshold and high-energy expansions of the cross section in tests/literature_checks/triplet/README.md, and
# where the latter starts (k >= 1e3)
sigma_threshold, _, sigma_asym, _, (_, K_ASYM) = load_derivation('triplet').numeric_expansions()
UPWIND = (1 - np.exp(-DLNE)) / DLNE  # energy a continuous loss removes in AM3's Chang & Cooper regime


@pytest.fixture(scope='module')
def triplet(tmp_path_factory):
    """The code's triplet pair production (tests/harness/triplet_mojo.mojo): its table nodes and kernel rows."""
    return harness.Triplet(tmp_path_factory.mktemp('harness'))


def test_triplet_cross_section_tables(triplet):
    """The tabulated QED cross section (tools/triplet) agrees with Lee et al.'s (2021) exact threshold and
    high-energy expansions (those of tests/literature_checks/triplet/README.md, as the code uses them) where these
    hold."""
    k, s = triplet.k, triplet.sig
    hi = k >= K_ASYM
    np.testing.assert_allclose(s[hi], sigma_asym(k[hi]), rtol=5e-4)
    lo = k <= 4.2
    np.testing.assert_allclose(s[lo], sigma_threshold(k[lo]), rtol=3e-3)


def test_triplet_kernels(triplet):
    """Each interaction deposits one positron and one lower-energy electron; the lepton loses exactly the pair's
    energy. No weight is negative. Above the last table node (2P = 1e10), the products' mean energies continue into
    the extrapolation without a jump."""
    rows = triplet.rows()
    P, R, L, (pos, low) = rows['P'], rows['R'], rows['L'], rows['W']
    x = [np.exp(-np.arange(len(w)) * rows['dl']) for w in pos]   # the energy fraction of each offset
    np.testing.assert_allclose([w.sum() for w in pos], R, rtol=1e-12)
    np.testing.assert_allclose([w.sum() for w in low], R, rtol=1e-12)
    np.testing.assert_allclose([(a + b) @ xo for a, b, xo in zip(pos, low, x)], L, rtol=1e-12)
    assert min(min(a.min(), b.min()) for a, b in zip(pos, low)) >= 0.0
    # k <x> / ln^2(2k) varies slowly with k (0.35 ... 0.39)
    mean = np.array([(a + b) @ xo / R[q] for q, (a, b, xo) in enumerate(zip(pos, low, x))])   # <x+> + <x_low>
    f = mean * P / np.log(4 * P) ** 2
    j = np.flatnonzero(2 * P > triplet.k[-1])[0]
    assert np.max(np.abs(np.diff(np.log(f[j - 10:j + 10])))) < 2e-3


@pytest.mark.parametrize('gamma', [1e3, 1e5, 3e7])
def test_triplet_products(gamma):
    """Monoenergetic electrons on triplet_box's photons, one short step: as many electrons appear as positrons are
    made, and the pair component holds both. The electrons lose the pair's energy as a continuous loss, of which
    AM3's Chang & Cooper scheme removes (1 - e^-dlnE)/dlnE, as it does of synchrotron and inverse-Compton losses."""
    a, (e0, _) = triplet_box(line(1e8, gamma * MEC2_EV), dt=1e-3)
    El = a.get_egrid_lep()
    npos = number(a.get_positrons())
    assert npos > 0
    assert number(a.get_electrons()) - number(e0) == pytest.approx(npos, rel=1e-8)
    assert number(a.get_pairs_triplet()) == pytest.approx(2 * npos, rel=1e-12)
    pairs = energy(a.get_pairs_triplet(), El)
    lep = energy(a.get_electrons(), El) + energy(a.get_positrons(), El)
    assert lep - energy(e0, El) == pytest.approx((1 - UPWIND) * pairs, rel=1e-4)


def triplet_steady(on, merged=False):
    """setups.triplet_steady after 8 steps of dt = 1e4 t_esc (t_esc = 1e4 s). At 3e11 photons cm^-3 the triplet loss
    is below 1 % of the escape rate, so AM3's solver evolves the leptons in its regime 1 (escape only, exact for a
    steady state)."""
    return steady_state(3e11, 1e8, 8, 1e4, triplet_pair_production=on, merge_positrons_into_electrons=merged)


def test_triplet_steady_state():
    """In a steady state each interaction absorbs one photon and makes one e+ and one e-. Against the run without the
    process, the extra e- and the missing photons both equal the e+, and the pair component holds twice as many. The
    e+ escape as fast as the photon solver absorbs photons. The missing photons (1.7e-8 of all) carry round-off,
    hence rel 1e-6."""
    a, b = triplet_steady(True), triplet_steady(False)
    npos = number(a.get_positrons())
    assert npos > 0
    assert number(a.get_electrons()) - number(b.get_electrons()) == pytest.approx(npos, rel=1e-12)
    assert number(b.get_photons()) - number(a.get_photons()) == pytest.approx(npos, rel=1e-6)
    assert number(a.get_pairs_triplet()) == pytest.approx(2 * npos, rel=1e-12)
    absorbed = number(a.get_photons() / a.get_t_photon_triplet())  # per second
    assert npos / 1e4 == pytest.approx(absorbed, rel=1e-9)
    assert (1e4 / a.get_t_pair_triplet()).max() < 1e-2  # (escape-dominated, as intended)


def test_triplet_merged_pairs():
    """With positrons merged into the electrons, each interaction adds two leptons (steady state)."""
    a, b = triplet_steady(True, merged=True), triplet_steady(False, merged=True)
    extra = number(a.get_pairs()) - number(b.get_pairs())
    assert extra > 0
    assert extra == pytest.approx(2 * (number(b.get_photons()) - number(a.get_photons())), rel=1e-6)
    assert number(a.get_pairs_triplet()) == pytest.approx(extra, rel=1e-12)


def test_triplet_switches():
    ele = line(1e8, 1e5 * MEC2_EV)
    a, (e0, _) = triplet_box(ele, triplet_pair_production_emission=False)
    assert not a.get_positrons().any() and not a.get_pairs_triplet().any()
    El = a.get_egrid_lep()
    assert number(a.get_electrons()) == pytest.approx(number(e0), rel=1e-12)
    assert energy(a.get_electrons(), El) < energy(e0, El)
    a, (e0, _) = triplet_box(ele, triplet_pair_production_cooling=False)
    assert a.get_positrons().any()
    np.testing.assert_array_equal(a.get_electrons()[np.argmax(e0)], e0[np.argmax(e0)])
    a, (e0, ph0) = triplet_box(ele, triplet_pair_production=False)
    np.testing.assert_array_equal(a.get_electrons(), e0)
    np.testing.assert_array_equal(a.get_photons(), ph0)
    assert not a.get_positrons().any()


@pytest.mark.parametrize('annihilation', [True, False])
def test_pair_components_add_up_with_triplet(annihilation):
    """In a source with injection, radiation and escape, the pair components (including the triplet pairs) add up
    to all pairs, with and without the e-/e+ halves of annihilation."""
    # The sum is exact unless a solve zeroes a negative density (AM3 does after each step). At dt = 1e4 s, a pulse of
    # pairs cooled by inverse Compton on the first synchrotron photons flows into the Crank-Nicolson window below,
    # whose top cell undershoots in some components but not in all pairs: the sum is 9e-2 off in that bin
    # (tests/reference/test_window_top_undershoot.py). At 3000 s nothing goes negative.
    a = evolve(pulse_source(3e3, pair_annihilation=annihilation), 60)
    comp = sum(getattr(a, getter)() for getter in PAIR_COMPONENTS)
    np.testing.assert_allclose(comp, a.get_pairs(), rtol=1e-6, atol=1e-12 * a.get_pairs().max())
    assert a.get_pairs_triplet().any()


# ---------------------------------------------------------------- neutron decay
M_N_EV, TAU_N = MASS_EV['MASS_N'], TAU_S['TAU_N']   # the code's neutron mass and mean life


@pytest.fixture(scope='module')
def neutron_decay_kernels(tmp_path_factory):
    """The default grid's kernels (tests/harness/neutron_decay_mojo.mojo, via harness.neutron_decay_kernels): the
    grids [GeV], K_e, K_nu and the proton shares."""
    return harness.neutron_decay_kernels(tmp_path_factory.mktemp('harness'))


def test_neutron_decay_kernels(neutron_decay_kernels):
    """Per decay: one electron, one antineutrino and one proton, carrying the neutron's energy less what falls below
    the grids (electrons below 0.511 MeV, the lepton grid's start; antineutrinos below 306 eV). No weight is
    negative. For ultra-relativistic neutrons the antineutrinos take 5.11e-4 of the energy (Lipari et al. 2007:
    5.1e-4)."""
    K = neutron_decay_kernels
    Eh, El, En = K['Eh'], K['El'], K['En']
    assert K['Ke'].min() >= 0.0 and K['Knu'].min() >= 0.0 and K['ps0'].min() > 0.0
    np.testing.assert_allclose(K['ps0'] + K['ps1'], 1.0, rtol=1e-14)   # the proton, shared by two bins
    r = np.concatenate([[np.exp(-DLNE)], Eh[:-1] / Eh[1:]])
    xp = K['ps0'] + K['ps1'] * r  # the proton's energy fraction
    ne, nnu = K['Ke'].sum(0), K['Knu'].sum(0)
    assert np.all(ne <= 1 + 1e-14) and ne.min() > 1 - 2e-8
    assert np.all(nnu <= 1 + 1e-14) and nnu.min() > 1 - 2e-7
    e = (K['Ke'] * El[:, None]).sum(0) / Eh + (K['Knu'] * En[:, None]).sum(0) / Eh + xp
    np.testing.assert_allclose(e, 1.0, atol=1e-11)
    assert (K['Knu'][:, -1] * En).sum() / Eh[-1] == pytest.approx(5.1e-4, rel=0.01)


def neutron_box(n_of_E, **switches):
    """Only neutron decay acts (no escape or other process): neutrons of density n_of_E(E [eV]), one step of 1 s."""
    a = new(off=('escape', 'electron_syn', 'electron_compton', 'annihilation', 'ssa', 'proton_syn', 'proton_compton',
                 'bethe_heitler', 'photopion', 'pp', 'expansion', 'adiabatic_cooling', 'pair_annihilation',
                 'triplet_pair_production', 'kaons'), dt=1.0, **switches)
    n0 = n_of_E(a.get_egrid_had())
    a.set_current_densities_neutrons(n0)
    a.evolve_step()
    return a, n0


def test_neutron_decay_box():
    """Neutrons of 0.1-100 TeV in a closed box decay at the rate 1/(gamma tau_n): one step removes
    n (1 - exp(-dt/(gamma tau_n))). As in AM3's decays, the products come from the evolved neutrons: one proton,
    electron and antineutrino per decay, with the neutron's energy, and no positron. The components equal them."""
    a, n0 = neutron_box(lambda E: np.where((E >= 1e11) & (E <= 1e14), 1e11 / E, 0.0))
    E, Enu = a.get_egrid_had(), a.get_egrid_neutrinos()
    El = a.get_egrid_lep()
    t = a.get_t_neutron_decay()
    np.testing.assert_allclose(t, E / M_N_EV * TAU_N, rtol=1e-13)
    n = a.get_neutrons()
    np.testing.assert_allclose(n, n0 * np.exp(-1.0 / t), rtol=1e-12)
    decays = number(n / t)  # per second
    p, e, nu = a.get_protons(), a.get_electrons(), a.get_electron_anti_neutrinos()
    assert number(p) == pytest.approx(decays, rel=1e-13)
    assert number(e) == pytest.approx(decays, rel=1e-12)
    assert number(nu) == pytest.approx(decays, rel=1e-6)  # (below the neutrino grid: 1.5e-7 at most)
    assert energy(p, E) + energy(e, El) + energy(nu, Enu) == pytest.approx(energy(n / t, E), rel=1e-12)
    assert not a.get_positrons().any()
    np.testing.assert_array_equal(a.get_electrons_neutron_decay(), e)
    np.testing.assert_array_equal(a.get_electron_anti_neutrinos_neutron_decay(), nu)


def test_neutron_decay_switch():
    """Switched off, the neutrons do not decay and nothing is made."""
    a, n0 = neutron_box(lambda E: np.where((E >= 1e9) & (E <= 1e12), 1e9 / E, 0.0), neutron_decay=False)
    np.testing.assert_array_equal(a.get_neutrons(), n0)
    for g in ['get_protons', 'get_electrons', 'get_electron_anti_neutrinos', 'get_electrons_neutron_decay',
              'get_electron_anti_neutrinos_neutron_decay']:
        assert not getattr(a, g)().any(), g


@pytest.fixture(scope='module')
def neutron_source():
    """A photo-hadronic source with a long escape time (1e6 s), so that neutrons up to about 1 TeV decay before they
    escape; 40 steps of 1e4 s."""
    a = new()
    a.set_mag_field(1.0)
    a.set_escape_timescale(1e6)
    a.set_solver_time_step(1e4)
    a.set_powerlaw_injection_parameters_protons(1e48, 1e46, 1e10, 1e10, 1e18, 2.0, 2.0, 1.0)
    a.set_powerlaw_injection_parameters_electrons(1e48, 1e43, 1e7, 1e7, 1e11, 2.0, 2.0, 1.0)
    return evolve(a, 40)


def test_neutron_decay_components(neutron_source):
    """In a source the e+- components, including the electrons of neutron decay, add up to all pairs. Of the electron
    antineutrinos, neutron decay makes most below 1 GeV and less than 1e-3 above 1 TeV."""
    a = neutron_source
    comp = sum(getattr(a, g)() for g in PAIR_COMPONENTS) + a.get_electrons_neutron_decay()
    np.testing.assert_allclose(comp, a.get_pairs(), rtol=1e-6, atol=1e-12 * a.get_pairs().max())
    nu = a.get_electron_anti_neutrinos()
    nd_nu = a.get_electron_anti_neutrinos_neutron_decay()
    assert a.get_electrons_neutron_decay().any() and a.get_photons_neutron_decay_electrons_syn_compton().any()
    Enu = a.get_egrid_neutrinos()
    low, high = (Enu < 1e9) & (nu > 0), (Enu > 1e12) & (nu > 0)
    assert low.sum() > 30 and np.all(nd_nu[low] > 0.9 * nu[low])
    assert np.all(nd_nu[high] < 1e-3 * nu[high])


# ---------------------------------------------------------------- kaons: photo-hadronic production
# the kaon page (tests/literature_checks/kaons/README.md) as kaons/kaon_production.py reads it: its thresholds, the
# multi-pion types and steps, and its fit of the kaons of pp collisions
kaon_production = load_derivation('kaon_production')
KP, KM, KS, KL = range(4)
# the multi-pion types M2 ... M7 of Hümmer et al. (2010, Table 6) as tests/literature_checks/kaons/README.md gives
# them (photon energies [GeV]), and the steps of its table, (t, eps_min, eps_max) of type M(t + 1)
TYPE_RANGE = {t: r for t, r in kaon_production.readme_type_ranges().items() if t >= 1}
MPI_STEPS = kaon_production.readme_mpi_steps()
K0S = kaon_production.numeric_neutral()   # the K0_S among K0 and anti-K0


def steps_of(t):
    """The steps of type M(t + 1) in the table of kaons/README.md, (eps_min, eps_max) in order."""
    return [(lo, hi) for tt, lo, hi in MPI_STEPS if tt == t]


@pytest.fixture(scope='module')
def kaon_channel_lines(tmp_path_factory):
    """The lines of tests/harness/kaon_channels_mojo.mojo: the kaon production channels of physics/photohadron.mojo
    (`kaon_production_channels`) and AM3's multi-pion entries ("pi ...")."""
    return harness.run('kaon_channels_mojo', tmp_path_factory.mktemp('harness'), text=True)


@pytest.fixture(scope='module')
def kaon_channels(kaon_channel_lines):
    """Rows emin, emax, chi_p[4], chi_n[4], mp[4], mn[4], sigma, mtype of every kaon channel."""
    return np.array([ln.split() for ln in kaon_channel_lines if not ln.startswith('pi')], dtype=float)


@pytest.fixture(scope='module')
def multi_pion(kaon_channel_lines):
    """Rows sigma, chi of AM3's 14 multi-pion entries (two per type: the low-energy and the leading pions)."""
    rows = np.array([ln.split()[1:] for ln in kaon_channel_lines if ln.startswith('pi')], dtype=float)
    assert rows.shape == (14, 2)
    return rows


@pytest.fixture(scope='module')
def two_body_channels(kaon_channels):
    """The two-body steps."""
    return kaon_channels[kaon_channels[:, -1] == -1]


def columns(row):
    """chi and cross section [mubarn] of the four channel columns of a step."""
    chi_p, chi_n, mp, mn = row[2:6], row[6:10], row[10:14], row[14:18]
    return {'p->K+': (chi_p[KP], mp[KP]), 'p->K0': (chi_p[KS], mp[KS] + mp[KL]),
            'n->K+': (chi_n[KP], mn[KP]), 'n->K0': (chi_n[KS], mn[KS] + mn[KL])}


@pytest.mark.parametrize('column', list(ky.COLUMNS))
def test_kaon_chi_from_the_measured_angular_distributions(two_body_channels, column):
    """The two-body channels gamma N -> K Y: the energy fraction chi of every step and channel column (p -> K+,
    p -> K0, n -> K+, n -> K0) from the measured angular distributions (tests/support/measured/ky_angular.py),
    rounded to two decimals; the same chi for K0_S and K0_L."""
    for step, row in enumerate(two_body_channels):
        chi, sigma = columns(row)[column]
        if sigma == 0.0:
            continue
        # rounded to two decimals
        assert chi == pytest.approx(ky.step_chi(column, step), abs=0.005 * 1.001), (column, ky.STEPS[step])
        # neutral kaons: the same chi for K0_S and K0_L
        chi_p, chi_n = row[2:6], row[6:10]
        assert chi_p[KS] == chi_p[KL] and chi_n[KS] == chi_n[KL]


@pytest.mark.parametrize('column', list(ky.COLUMNS))
def test_kaon_cross_sections_from_the_measurements(two_body_channels, column):
    """The steps of the measured cross sections and their means, integrated over angle, rounded to two decimals (two
    significant digits below 0.1 mubarn)."""
    assert [tuple(r[:2]) for r in two_body_channels] == ky.STEPS
    for step, row in enumerate(two_body_channels):
        sigma = columns(row)[column][1]
        want = ky.column_sigma(column, step)
        tol = 0.005 if want >= 0.1 else 0.0005
        assert sigma == pytest.approx(want, abs=tol * 1.001), (column, ky.STEPS[step])


def test_multi_pion_kaon_steps(kaon_channels, multi_pion):
    """The multi-pion types' kaons are those made with further particles (K Y pi, K anti-K N, ...); the two-body
    reactions gamma N -> K Lambda, K Sigma have their own, measured channels. Below the gamma N -> K Lambda pi
    threshold (1.15 GeV) only two-body reactions make kaons, so the kaons of type M2 (0.9-1.5 GeV) start there.
    Strange production with further particles rises most across M3 (1.5-5 GeV) and M4 (5-50 GeV), so each has two
    steps (1.5-2.5 and 2.5-5 GeV; 5-15 and 15-50 GeV). From step to step, protons make more of every kaon species
    per interaction."""
    multi = kaon_channels[kaon_channels[:, -1] >= 0]
    assert sorted(set(multi[:, -1])) == sorted(TYPE_RANGE)
    # gamma N -> K Lambda pi in its four charge states (the s of kaons/README.md, PDG 2026 masses)
    three_body = kaon_production.threshold_span(kaon_production.NUCLEON, kaon_production.KAON, ('Lambda',),
                                                kaon_production.PION)
    start = multi[:, 0].min()
    assert start <= three_body[0] < 1.01 * start  # 1.15 GeV; 1.152-1.168 GeV by charge state
    for t, (lo, hi) in TYPE_RANGE.items():
        mine = multi[multi[:, -1] == t]
        # the type's cross section: AM3's (Hümmer et al., Table 6)
        assert np.all(mine[:, 18] == multi_pion[2 * t, 0])
        # the type's kaons: consecutive steps from its start (or the three-body threshold) to its end, each with a
        # low-energy and a leading part
        steps = sorted(set(map(tuple, mine[:, :2])))
        assert steps[0][0] == max(lo, start) and steps[-1][1] == hi
        assert all(a[1] == b[0] for a, b in zip(steps[:-1], steps[1:]))
        assert all(np.sum(np.all(mine[:, :2] == st, axis=1)) == 2 for st in steps)
    for row in multi:
        # antikaons need a kaon as well: not below the threshold of gamma p -> p K+ K- (1.507 GeV)
        if row[10 + KM] > 0 or row[14 + KM] > 0:
            assert row[0] >= 0.99 * kaon_production.threshold(ky.MP, ky.MP, ky.MKP, ky.MKP)
    # the steps rise in every species
    steps = sorted(set(map(tuple, multi[:, :2])))
    per_step = [multi[np.all(multi[:, :2] == st, axis=1)][:, 10:14].sum(axis=0) for st in steps]
    for a, b in zip(per_step[:-1], per_step[1:]):
        assert np.all(b > a), (a, b)
    # at least one kaon per antikaon (strangeness), and K0_S and K0_L in equal parts
    for m in per_step:
        assert m[KP] + m[KS] + m[KL] >= 2 * m[KM] and m[KS] == m[KL]


def proton_kaons(row):
    """A channel's cross sections [mubarn] of (K+, K0, K-, anti-K0) for protons. A multi-pion channel makes the
    isospin mirror from neutrons (K+ <-> K0, K- <-> anti-K0), which tells its K0 from its anti-K0; a two-body step
    makes no antikaons."""
    sigma, mp, mn = row[18], row[10:14], row[14:18]
    if row[-1] < 0:
        return sigma * np.array([mp[KP], mp[KS] + mp[KL], 0.0, 0.0])
    assert mp[KS] + mp[KL] == pytest.approx(mn[KP] + mn[KM], rel=1e-12)
    return sigma * np.array([mp[KP], mn[KP], mp[KM], mn[KM]])


def test_m4_kaon_steps_from_the_strange_particles_at_20_gev(kaon_channels, multi_pion):
    """Type M4's kaons (5-50 GeV), in two steps (5-15 and 15-50 GeV), against the strange particles measured at
    20 GeV (Abe et al. 1985 and 1984; tests/support/measured/strange_20gev.py).

    Numbers: each species of a step lies, to the multiplicities' rounding, on a line in the logarithm of the photon
    energy, taken at the step's middle. The line runs through the code's kaons at 2.5-5 GeV (M3's upper step, with
    its two-body kaons) and the species that balance the strange particles measured at 20 GeV.

    Energy: a step's K0_S have the mean energy fraction chi of the x and pT^2 distribution measured at 20 GeV, taken
    to the step's middle by Feynman scaling. Their leading part has about the measured share and energy of the K0_S
    going towards the nucleon (to 0.03 and 15 %)."""
    rows = [(r[0], r[1], proton_kaons(r)) for r in kaon_channels]
    m3_upper = steps_of(2)[1]
    low, e_low = g20.ln_mean(rows, *m3_upper), np.sqrt(m3_upper[0] * m3_upper[1])
    sigma, chi_pions = multi_pion[6, 0], multi_pion[6:8, 1]  # of M4; chi of its low-energy and leading pions
    for lo, hi in steps_of(3):
        mine = kaon_channels[(kaon_channels[:, 0] == lo) & (kaon_channels[:, 1] == hi)]
        centre = np.sqrt(lo * hi)
        code = sum(proton_kaons(r) for r in mine)
        want = g20.trend(centre, e_low, low)
        # the multiplicities per interaction are rounded to four decimals
        assert np.all(np.abs(code - want) <= 0.5e-4 * 1.001 * sigma), (lo, hi, code, want)
        # the mean chi of the K0_S (from protons), and their leading part (the channel with the leading pions' chi)
        ks, chi = mine[:, 10 + KS], mine[:, 2 + KS]
        assert np.dot(ks, chi) / ks.sum() == pytest.approx(g20.mean_chi(centre)[0], rel=0.02)
        lead = chi == chi_pions[1]
        assert lead.sum() == 1 and chi[~lead][0] < chi[lead][0]
        lead_chi, lead_share = g20.mean_chi(centre, x_below=-0.2)
        assert ks[lead][0] / ks.sum() == pytest.approx(lead_share, abs=0.03)
        assert chi[lead][0] == pytest.approx(lead_chi, rel=0.15)
    # at 20 GeV itself (in the upper step, which stands for 15-50 GeV, over which the yields still rise)
    at20 = sum(proton_kaons(r) for r in kaon_channels if r[0] <= g20.E_GAMMA < r[1])
    assert K0S(at20[1], at20[3]) == pytest.approx(g20.K0S[0], rel=0.15)  # 10.7 against 9.66 mubarn
    assert at20[0] + at20[1] - at20[2] - at20[3] == pytest.approx(g20.hyperon_strangeness(), rel=0.06)  # 8.6, 8.5
    # the next type's kaons (M5, 50-500 GeV, set independently) follow the same line: kaons, antikaons and K0_S
    # within 12 % (6, 6 and 11 %), each species within 20 % (K0: 18 %, as the line keeps the K+ excess over K0
    # rising and M5 has less)
    m5 = sum(proton_kaons(r) for r in kaon_channels if r[-1] == 4)
    on_line = g20.trend(np.sqrt(TYPE_RANGE[4][0] * TYPE_RANGE[4][1]), e_low, low)
    for part in ([0, 1], [2, 3], [1, 3]):
        assert m5[part].sum() == pytest.approx(on_line[part].sum(), rel=0.12)
    assert np.all(np.abs(m5 / on_line - 1.0) < 0.20)


def test_no_kaons_without_photopion_emission():
    a = new(grid=GRID_TO_1E21_EV, photopion_emission=False)
    a.set_powerlaw_injection_parameters_protons(*PROTONS_TO_1E20_EV)
    evolve(a, 5)
    assert not a.get_kaons().any()


# ---------------------------------------------------------------- kaons: pp collisions
MP = MASS_GEV['MASS_P']   # GeV, the code's


@pytest.fixture(scope='module')
def pp_kaons_exe(tmp_path_factory):
    """tests/harness/pp_kaons_mojo.mojo: the kaons of pp collisions of src/am3_mojo/physics/proton_proton.mojo."""
    return harness.build('pp_kaons_mojo', tmp_path_factory.mktemp('harness'))


def test_pp_kaons_multiplicity_fit(pp_kaons_exe):
    """Below E_t = 100 GeV the code's kaons per collision follow the kaon page's fit to NA49's Table 11 (which
    kaons/kaon_production.py checks against NA49, pp.fit): its f(s) with the parameters of its table for each kaon
    (the K+ with its second, near-threshold term), zero up to each threshold, and rising above it."""
    fit, f = kaon_production.readme_fit(), kaon_production.f_numeric()
    E = np.geomspace(2.0, 100.0, 80)
    got = harness.output(pp_kaons_exe, 'fit', *[repr(float(e)) for e in E]).reshape(-1, 3)
    s = 2 * MP**2 + 2 * MP * E
    for col, kind in enumerate(('K+', 'K-', 'K0S')):
        A, a, b, C, d, rth = fit[kind]
        above = s > rth**2
        np.testing.assert_allclose(got[above, col], f(A, a, b, C, d, s[above], rth**2), rtol=1e-12, err_msg=kind)
        assert not got[~above, col].any(), kind
        assert np.all(np.diff(got[above, col]) > 0), kind


def test_pp_kaons_join_to_koers(pp_kaons_exe):
    """The factor on Koers et al.'s spectra (the page's c_K, "The kaon spectra"; kaons/kaon_production.py checks the
    spectra against Koers, Pe'er & Wijers 2006): the code's c_t is the page's, the fit over their kaons per collision
    at E_t (1.30, 0.97, 0.985); its fade is the page's formula; and its kernels make the fit's kaons per collision
    below E_t, c_K times Koers et al.'s from E_t to E_1 = 1 TeV, and Koers et al.'s from E_1 on."""
    fit, f = kaon_production.readme_fit(), kaon_production.f_numeric()
    c_k, e_t, e_1 = kaon_production.readme_fade()
    s_of = lambda E: 2 * MP**2 + 2 * MP * E
    kinds = ('K+', 'K-', 'K0S')
    c_t = harness.output(pp_kaons_exe, 'ct')
    page = [float(f(*fit[k][:5], s_of(e_t), fit[k][5]**2)) / kaon_production.kpw_multiplicity(k, e_t) for k in kinds]
    np.testing.assert_allclose(c_t, page, rtol=1e-4)
    E = np.geomspace(e_t, 0.999 * e_1, 25)
    for c in c_t:
        np.testing.assert_allclose(harness.output(pp_kaons_exe, 'fade', repr(float(c)), *[repr(float(e)) for e in E]),
                                   c_k(c, E), rtol=1e-12)
        assert np.all(harness.output(pp_kaons_exe, 'fade', repr(float(c)), repr(e_1), repr(1e5)) == 1.0)
    rows = harness.output(pp_kaons_exe, 'kernel').reshape(-1, 1 + 2 * len(kinds))
    Ej = rows[:, 0]
    for col, (kind, c) in enumerate(zip(kinds, c_t)):
        n, bare = rows[:, 1 + 2 * col], rows[:, 2 + 2 * col]
        low = (Ej < e_t) & (n > 0)
        assert low.sum() > 10, kind
        A, a, b, C, d, rth = fit[kind]
        np.testing.assert_allclose(n[low], f(A, a, b, C, d, s_of(Ej[low]), rth**2), rtol=1e-12, err_msg=kind)
        mid, top = (Ej >= e_t) & (Ej < e_1), Ej >= e_1
        np.testing.assert_allclose(n[mid] / bare[mid], c_k(c, Ej[mid]), rtol=1e-12, err_msg=kind)
        np.testing.assert_allclose(n[top], bare[top], rtol=1e-12, err_msg=kind)


def test_pp_kaons_in_a_source():
    """In a pp-dominated source (setups.pp_source) kaons are made, decay into the pp components and take energy from
    the protons: smoothly across 100 GeV, and none from protons whose kaons would fall below the grid. With kaons
    off, none are made."""
    # the source without charm (whose decays make kaons); `on` is made last, for test_pp_charm_in_a_source
    off, on = pp_source(kaons=False, charm=False), pp_source(charm=False)
    E = on.get_egrid_had()
    for g in ('get_kaon_plus', 'get_kaon_minus', 'get_kaon_0S', 'get_kaon_0L'):
        assert getattr(on, g)().max() > 0, g
        assert not getattr(off, g)().any(), g
    # kaons: a few per cent of the charged pions (energy weighted; 0.03 in this source)
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
    # below about 7 GeV, are neither made nor charged to the protons; above, the kaons' share rises smoothly into
    # that of the spectra above 100 GeV
    assert np.all(r[E < 6e9] == 1.0)
    mid = (E > 6.5e9) & (E < 1.2e11)
    assert r[mid][0] > 1.0 and np.all(np.diff(r[mid]) > 0), r[mid]


# ---------------------------------------------------------------- kaons: evolution
# the code's masses [GeV] and mean lives [s]: PDG 2026 for the kaons, AM3's charged pion
M_K, TAU_K, M_PI = MASS_GEV['MASS_K'], TAU_S['TAU_K'], MASS_GEV['MASS_PI']
M_K0, TAU_KS, TAU_KL = MASS_GEV['MASS_K0'], TAU_S['TAU_KS'], TAU_S['TAU_KL']


def hadronic(kaons=True, kaon_syn=True, kaon_ic=True, off=(), pp_density=0.0, charm=False):
    """A photo-hadronic source in 100 G, where kaons matter, after 30 steps; `off`: further processes switched off;
    `pp_density` [cm^-3]: target protons for pp collisions (none by default); charm hadrons, whose decays make
    kaons, off unless `charm`."""
    return hadronic_source(100.0, 30, pp_density, off=off, kaons=kaons, kaon_syn=kaon_syn, kaon_compton=kaon_ic,
                           charm=charm, pion_syn=True, muon_syn=True)


@pytest.fixture(scope='module')
def kaon_runs():
    return {'on': hadronic(True), 'off': hadronic(False),
            'no_radiation': hadronic(True, kaon_syn=False, kaon_ic=False)}


def test_kaon_timescales(kaon_runs):
    a = kaon_runs['on']
    E = a.get_egrid_had() / 1e9  # GeV
    np.testing.assert_allclose(a.get_t_kaon_decay(), E / M_K * TAU_K, rtol=1e-12)
    # synchrotron: t ∝ m^4 / E, so at equal energy t_K / t_pi = (m_K / m_pi)^4
    ratio = a.get_t_kaon_syn() / a.get_t_pion_syn()
    np.testing.assert_allclose(ratio, (M_K / M_PI) ** 4, rtol=1e-12)
    assert np.all(np.isfinite(a.get_t_kaon_compton()))


def kaon_step(frac_neutrons=1.0, frac_pions=1.0):
    """One step of t_esc = 1e3 s from no kaons, with kaon decay and cooling off: nucleons on soft photons make
    kaons, which only escape (charm, whose decays make kaons, is off)."""
    a = new(off=('kaon_decay', 'kaon_syn', 'kaon_compton', 'charm'), grid=GRID_TO_1E21_EV)
    a.set_escape_timescale(1e3)
    a.set_solver_time_step(1e3)
    a.set_escape_fraction_neutrons(frac_neutrons)
    a.set_escape_fraction_pions(frac_pions)
    E, eg = a.get_egrid_had(), a.get_egrid_photons()
    a.set_current_densities_protons(np.where((E > 1e16) & (E < 1e20), 1e-10 * 1e16 / E, 0.0))
    a.set_current_densities_photons(np.where((eg > 1) & (eg < 10), 1e10, 0.0))
    a.evolve_step()
    return a


def test_kaon_escape_in_the_evolution():
    """With escape the only loss, one step of length dt from n = 0 gives n = q/a (1 - exp(-a dt)) (the solver's
    sink regime). K+- take the escape fraction of pions, K0_S and K0_L that of neutrons."""
    base, neutral, charged = kaon_step(), kaon_step(frac_neutrons=0.25), kaon_step(frac_pions=0.5)
    np.testing.assert_allclose(neutral.get_t_kaon_neutral_escape(), 4e3, rtol=1e-12)
    np.testing.assert_allclose(neutral.get_t_kaon_escape(), 1e3, rtol=1e-12)
    np.testing.assert_allclose(charged.get_t_kaon_escape(), 2e3, rtol=1e-12)
    step = lambda f: (1 - np.exp(-f)) / f  # n / (q dt) for a dt = f
    for g, ch in (('get_kaon_plus', True), ('get_kaon_minus', True), ('get_kaon_0S', False),
                  ('get_kaon_0L', False)):
        n0, nn, nc = getattr(base, g)(), getattr(neutral, g)(), getattr(charged, g)()
        sig = n0 > 1e-30 * n0.max()
        assert sig.sum() > 50, g
        # (the source changes by < 1e-4 with the neutrons' escape: the step's kaons come from the evolved
        #  nucleons, as AM3's pions do)
        np.testing.assert_allclose(nn[sig] / n0[sig], 1.0 if ch else step(0.25) / step(1.0),
                                   rtol=1e-3, err_msg=g)
        np.testing.assert_allclose(nc[sig] / n0[sig], step(0.5) / step(1.0) if ch else 1.0,
                                   rtol=1e-12, err_msg=g)


@pytest.mark.parametrize('source', ['photo-hadronic', 'pp'])
def test_kaon_compton_cooling_thomson_scaling(source):
    """K+- cool by inverse Compton as charged pions do, with the kaon mass: on soft photons (Thomson regime)
    t_K / t_pi = (m_K / m_pi)^4 at equal energy. With kaons from pp collisions only, the kaon getters give the pp
    population's rates."""
    a = new(**(dict(photopion=False, pp=True) if source == 'pp' else {}))
    if source == 'pp':
        a.set_pp_target_proton_density(1e10)
    a.set_powerlaw_injection_parameters_protons(*PROTONS_TO_1E20_EV)
    eg = a.get_egrid_photons()
    a.set_current_densities_photons(np.where((eg > 1e-4) & (eg < 1e-3), 1e10, 0.0))
    a.evolve_step()
    E = a.get_egrid_had()
    # gamma eps << m_e c^2 for the pions, and gamma_K >> 1
    thomson = (E / (M_PI * 1e9) * 1e-3 < 1e-2 * MEC2_EV) & (E / (M_K * 1e9) > 30)
    assert thomson.sum() > 50
    ratio = (a.get_t_kaon_compton() / a.get_t_pion_compton())[thomson]
    np.testing.assert_allclose(ratio, (M_K / M_PI) ** 4, rtol=3e-3)


def test_kaon_extra_cooling_roundtrip():
    a = new(init=False)
    t = np.geomspace(1e3, 1e6, len(a.get_egrid_had()))
    a.set_t_kaon_extra_cooling(t)
    np.testing.assert_allclose(a.get_t_kaon_extra_cooling(), t, rtol=1e-12)


def test_kaons_produced_and_decaying(kaon_runs):
    on, off = kaon_runs['on'], kaon_runs['off']
    k = on.get_kaons()
    assert np.all(k >= 0) and k.max() > 0
    # kaons are 1-2 orders of magnitude rarer than charged pions
    E = on.get_egrid_had()
    frac = np.sum(k * E) / np.sum(on.get_pions() * E)
    assert 1e-4 < frac < 0.1, frac
    # where pions cool before they decay, kaon decays dominate the nu_mu flux
    nu_on, nu_off = on.get_muon_neutrinos(), off.get_muon_neutrinos()
    En = on.get_egrid_neutrinos()
    top = En > 10 * En[np.argmax(nu_off * En)]
    assert np.max(nu_on[top] / np.maximum(nu_off[top], 1e-300)) > 2
    # and the totals change only a little
    assert abs(np.sum(nu_on * En) / np.sum(nu_off * En) - 1) < 0.1


def test_kaon_species(kaon_runs):
    """K-, K0_S and K0_L are produced and decay as they should."""
    a = kaon_runs['on']
    E = a.get_egrid_had()
    kp, km, ks, kl = a.get_kaon_plus(), a.get_kaon_minus(), a.get_kaon_0S(), a.get_kaon_0L()
    assert min(km.max(), ks.max(), kl.max()) > 0
    np.testing.assert_array_equal(a.get_kaons(), kp + km)
    # strangeness goes with the K+ into hyperons, and near threshold only K+ and K0 are made: fewer K- than K+
    assert 0.02 < np.sum(km * E) / np.sum(kp * E) < 1
    np.testing.assert_allclose(a.get_t_kaon_0S_decay(), E / 1e9 / M_K0 * TAU_KS, rtol=1e-12)
    np.testing.assert_allclose(a.get_t_kaon_0L_decay(), E / 1e9 / M_K0 * TAU_KL, rtol=1e-12)
    # K0_S and K0_L are made equally: where both decay much faster than they escape and than the time step
    # (steady state), their densities are in the ratio of their lifetimes
    fast = ((a.get_t_kaon_0L_decay() < 1e-4 * a.get_t_kaon_escape())
            & (ks > 1e-6 * ks.max()) & (kl > 1e-6 * kl.max()))
    assert fast.sum() > 20
    np.testing.assert_allclose(kl[fast] / ks[fast], TAU_KL / TAU_KS, rtol=2e-3)


def test_kaons_off_leaves_no_trace(kaon_runs):
    off = kaon_runs['off']
    for getter in ['get_kaons', 'get_kaon_0S', 'get_kaon_0L']:
        assert not getattr(off, getter)().any()
    assert not off.get_photons_kaons_syn().any()
    assert not off.get_photons_kaons_compton().any()


def test_kaon_radiation_switches(kaon_runs):
    on, off = kaon_runs['on'], kaon_runs['no_radiation']
    assert on.get_photons_kaons_syn().max() > 0 and on.get_photons_kaons_compton().max() > 0
    assert not off.get_photons_kaons_syn().any() and not off.get_photons_kaons_compton().any()
    assert np.all(off.get_t_kaon_syn() >= 1e199) and np.all(off.get_t_kaon_compton() >= 1e199)  # no cooling


# ---------------------------------------------------------------- charm hadrons: photo-hadronic production
# the charm page (tests/literature_checks/charm/README.md) as charm/charm_production.py reads it: its step table
# ("Production in photo-hadronic interactions", which charm_production.py derives from the measured data, cp.*),
# its fragmentation fractions f_h and its pp parameters
charm_production = load_derivation('charm_production')
CHARM_STEPS = charm_production.readme_charm_steps()
FRAC = charm_production.readme_fractions()
# the species of the harness's columns, and the particle of each pair (D- -> D+, ...)
STEP_SPECIES = ['D+', 'D-', 'D0', 'anti-D0', 'Ds+', 'Ds-', 'Lc+', 'anti-Lc-']
KIND = {s: STEP_SPECIES[i - i % 2] for i, s in enumerate(STEP_SPECIES)}
LC, AD0, DM = (STEP_SPECIES.index(s) for s in ('Lc+', 'anti-D0', 'D-'))
TYPE_SIGMA = 120.0   # mubarn, the cross section of the multi-pion types M3-M7 (Huemmer et al. 2010, Table 6)


@pytest.fixture(scope='module')
def charm_steps(tmp_path_factory):
    """The photo-hadronic charm steps of physics/photohadron.mojo (`charm_production_steps`), through
    tests/harness/charm_steps_mojo.mojo: [(emin, emax, sigma, mtype, [part dicts])]; a part: chi_p, chi_n, mp, mn
    (arrays over STEP_SPECIES)."""
    out = []
    for ln in harness.run('charm_steps_mojo', tmp_path_factory.mktemp('harness'), text=True):
        w = ln.split()
        if w[0] == 'step':
            out.append((*map(float, w[1:4]), int(w[4]), []))
        else:
            v = np.array(w[1:], dtype=float)
            out[-1][4].append(dict(chi_p=v[0:8], chi_n=v[8:16], mp=v[16:24], mn=v[24:32]))
    return out


def classify(parts):
    """The parts of a step by role: photon side, nucleon side (both: pairs of every species), associated anti-D,
    associated Lambda_c+."""
    roles = {}
    for p in parts:
        nz = [STEP_SPECIES[s] for s in range(8) if p['mp'][s] > 0 or p['mn'][s] > 0]
        if nz == ['Lc+']:
            roles['assoc_Lc'] = p
        elif set(nz) <= {'D-', 'anti-D0'}:
            roles['assoc_D'] = p
        elif 'nucleon' not in roles and 'photon' in roles:
            roles['nucleon'] = p
        else:
            roles.setdefault('photon', p)
    return roles


def test_charm_steps_against_the_readme(charm_steps):
    """The steps of the charm page's table: each lies inside its multi-pion type, with the type's cross section, and
    makes R charm pairs per interaction: a share h_a with an associated Lambda_c+ (R h_a per interaction, at most
    4e-4), h_N on the nucleon side, the rest on the photon side. The photon and nucleon sides fragment with the
    page's fractions, particles and antiparticles alike, at its energy fractions; the associated anti-D has its own,
    the Lambda_c+ the type's nucleon fraction."""
    assert len(charm_steps) == len(CHARM_STEPS)
    worst = 0.0   # the most associated Lambda_c+ per interaction
    for (emin, emax, sigma, t, parts), row in zip(charm_steps, CHARM_STEPS):
        where = f"{row['eps_min']}-{row['eps_max']} GeV"
        assert (emin, emax, t) == (row['eps_min'], row['eps_max'], row['t']), where
        assert sigma == TYPE_SIGMA, where
        R, ha, hN = row['R'], row['h_a'], row['h_N']
        pairs = sum(p['mp'].sum() for p in parts) / 2
        assert pairs == pytest.approx(R, rel=1e-12), where
        roles = classify(parts)
        if 'assoc_Lc' in roles:
            worst = max(worst, roles['assoc_Lc']['mp'].sum())
        # the shares of the pairs and the fragmentation of each side
        for side, share in {'photon': 1 - ha - hN, 'nucleon': hN}.items():
            if share == 0:
                assert side not in roles, (where, side)
                continue
            p = roles[side]
            for s, name in enumerate(STEP_SPECIES):
                for m in (p['mp'], p['mn']):
                    assert m[s] == pytest.approx(R * share * FRAC[KIND[name]], rel=1e-12), (where, side, name)
        if 'photon' in roles:
            chi = roles['photon']['chi_p']
            want = {'D+': row['chi_D'], 'D0': row['chi_D'], 'Ds+': row['chi_Ds'], 'Lc+': row['chi_Lc']}
            for s, name in enumerate(STEP_SPECIES):
                assert chi[s] == pytest.approx(want[KIND[name]], rel=1e-12), (where, name)
        if hN > 0:
            assert np.allclose(roles['nucleon']['chi_p'], row['chi_N'], rtol=1e-12), where
        # associated production: the Lambda_c+ and its anti-D
        if ha > 0:
            lc, ad = roles['assoc_Lc'], roles['assoc_D']
            assert lc['mp'][LC] == pytest.approx(R * ha, rel=1e-12), where
            assert ad['mp'].sum() == pytest.approx(R * ha, rel=1e-12), where
            d0 = FRAC['D0'] / (FRAC['D0'] + FRAC['D+'])
            assert ad['mp'][AD0] == pytest.approx(R * ha * d0, rel=1e-12), where
            for s in (AD0, DM):
                assert ad['chi_p'][s] == pytest.approx(row['chi_aD'], rel=1e-12), where
            assert 0.5 < lc['chi_p'][LC] < 0.6, where   # the type's nucleon fraction, 0.51-0.56 for M4-M7
    assert 0 < worst < 4e-4, worst


def test_charm_steps_from_neutrons(charm_steps):
    """From neutrons, the associated anti-D0 and D- are exchanged; everything else is as from protons."""
    for emin, emax, sigma, t, parts in charm_steps:
        for p in parts:
            assert np.array_equal(p['chi_p'], p['chi_n'])
        for role, p in classify(parts).items():
            if role == 'assoc_D':
                assert p['mn'][AD0] == p['mp'][DM] and p['mn'][DM] == p['mp'][AD0], (emin, emax)
            else:
                assert np.array_equal(p['mp'], p['mn']), (emin, emax, role)


# ---------------------------------------------------------------- charm hadrons: pp collisions
# the charm page's "Production in pp collisions", whose parameters, table and extensions charm_production.py derives
# (cpp.*)
PP = charm_production.readme_pp_charm()
CHARM_PAIRS = ['D+', 'D0', 'Ds+', 'Lc+']   # the kernel's pairs (each of a hadron and its antiparticle)
MUB = 1e-30          # cm^2
SUB = 4              # the code's convolution steps per grid step (CPP_SUB)


@pytest.fixture(scope='module')
def pp_charm_exe(tmp_path_factory):
    """tests/harness/pp_charm_mojo.mojo: the charm hadrons of pp collisions of
    src/am3_mojo/physics/proton_proton.mojo."""
    return harness.build('pp_charm_mojo', tmp_path_factory.mktemp('harness'))


@pytest.fixture(scope='module')
def pp_charm_kernel(pp_charm_exe):
    """Rows E_j [GeV], sigma kappa [cm^2], then per pair sigma_h [cm^2] and <x>."""
    return harness.output(pp_charm_exe, 'kernel').reshape(-1, 2 + 2 * len(CHARM_PAIRS))


def pp_charm_sigma(exe, *E, mode='sigma'):
    """pp_charm_sigma_cc, or with mode='inel' AM3's sigma_inel, at the proton energies E [GeV] (in cm^2)."""
    return harness.output(exe, mode, *[repr(float(e)) for e in E])


def pp_charm_shape(exe, E):
    """The code's g at x_m = exp(-m d), d = DLNE / SUB, normalised to d sum g = 1."""
    g = harness.output(exe, 'shape', repr(DLNE / SUB), repr(float(E)))
    return np.exp(-np.arange(g.size) * DLNE / SUB), g


def test_pp_charm_cross_section_table(pp_charm_exe):
    """The cross section sigma_cc: at Bhattacharya et al.'s energies the code has their Table 1 values (the charm
    page's table, which cpp.sigma_table proves theirs); it is continuous at 100 GeV, where the threshold form takes
    over, and zero below the threshold of pp -> p Lambda_c+ anti-D0."""
    E = np.array(sorted(PP['sigma']))
    np.testing.assert_allclose(pp_charm_sigma(pp_charm_exe, *E), [PP['sigma'][e] * MUB for e in E], rtol=1e-12)
    lo, at, hi = pp_charm_sigma(pp_charm_exe, 100.0 * (1 - 1e-7), 100.0, 100.0 * (1 + 1e-7))
    slope_lo, slope_hi = np.log(at / lo) / -np.log1p(-1e-7), np.log(hi / at) / np.log1p(1e-7)
    assert abs(slope_lo / slope_hi - 1) < 1e-4, (slope_lo, slope_hi)
    m_th = MP + MASS_GEV['MASS_LC'] + MASS_GEV['MASS_D0']
    E_th = (m_th ** 2 - 2 * MP ** 2) / (2 * MP)
    s = pp_charm_sigma(pp_charm_exe, E_th * (1 - 1e-9), E_th * (1 + 1e-3))
    assert s[0] == 0.0 and s[1] > 0.0


def test_pp_charm_spectrum_against_nlo(pp_charm_exe):
    """x dsigma/dx of a charm quark against the NLO spectra it was fitted to (Bhattacharya et al. 2015, Fig. 2, at
    the level of their 2016 replot, R_BERSS = 0.697; tests/support/measured/charm_pp.py) at 1e3, 1e6 and 1e9 GeV,
    0.01 <= x <= 0.5: within 15 % everywhere, 6 % rms."""
    for E in (1e3, 1e6, 1e9):
        x, g = pp_charm_shape(pp_charm_exe, E)
        pts = np.array(charm_pp.BERSS15_FIG2[E])
        sel = (pts[:, 0] >= 0.01) & (pts[:, 0] <= 0.5)
        xd, ydata = pts[sel, 0], 0.697 * pts[sel, 1] * 1e-6 * pts[sel, 0]   # pb -> mub, times x
        model = np.exp(np.interp(-np.log(xd), -np.log(x[1:]), np.log(np.maximum(g[1:], 1e-300)))) \
            * pp_charm_sigma(pp_charm_exe, E)[0] / MUB
        r = np.log(model / ydata)
        assert np.max(np.abs(np.expm1(r))) < 0.15 and np.sqrt(np.mean(r ** 2)) < 0.06, (E, np.max(np.abs(r)))


def test_pp_charm_rapidity_symmetry(pp_charm_exe):
    """Half of the charm quarks lie above x_c = m_T/sqrt(s), the mirror point of the rapidity-symmetric spectrum."""
    for E in (1e3, 1e6, 1e9, 1e12):
        x, g = pp_charm_shape(pp_charm_exe, E)
        xc = PP['mT'] / np.sqrt(2 * MP ** 2 + 2 * MP * E)
        d = DLNE / SUB
        above = d * np.sum(g[x > xc])
        assert abs(d * np.sum(g) - 1) < 1e-12
        assert abs(above - 0.5) < d * g.max(), (E, above)   # within the fine grid's resolution


def test_pp_charm_kernel_numbers_and_energy(pp_charm_kernel, pp_charm_exe):
    """Each pair's kernel makes f_h sigma_cc hadrons (and as many antiparticles): exactly below E_t = 1 TeV (one
    energy per proton energy), within 1.5 % above (the hadrons below their own mass are left out); above
    E_top = 1e11 GeV, their number per inelastic collision stays at its value there. Their mean energy fraction
    changes by less than 10 % across E_t (one grid step lowers it by about 3.5 %), and the protons lose what the
    hadrons take."""
    kernel = pp_charm_kernel
    E = kernel[:, 0]
    sig = pp_charm_sigma(pp_charm_exe, *E)
    inel = pp_charm_sigma(pp_charm_exe, *E, mode='inel')
    made = sig > 0
    top = E > PP['E_top']
    per_inel = (pp_charm_sigma(pp_charm_exe, PP['E_top'])[0]
                / pp_charm_sigma(pp_charm_exe, PP['E_top'], mode='inel')[0])
    for k, h in enumerate(CHARM_PAIRS):
        n, xm = kernel[:, 2 + 2 * k], kernel[:, 3 + 2 * k]
        below, above = made & (E < PP['E_t']), (E >= PP['E_t']) & ~top
        np.testing.assert_allclose(n[below], FRAC[h] * sig[below], rtol=1e-12, err_msg=h)
        ratio = n[above] / (FRAC[h] * sig[above])
        assert np.all((ratio > 0.985) & (ratio <= 1 + 1e-9)), (h, ratio.min())
        ratio = n[top] / (FRAC[h] * per_inel * inel[top])
        assert top.sum() > 20 and np.all((ratio > 0.999) & (ratio <= 1 + 1e-9)), (h, ratio.min())
        j = np.flatnonzero(above)[0]
        assert abs(xm[j] / xm[j - 1] - 1) < 0.1, (h, xm[j - 1], xm[j])
    kappa = sum(2 * kernel[:, 2 + 2 * k] * kernel[:, 3 + 2 * k] for k in range(4))
    np.testing.assert_allclose(kernel[:, 1], kappa, rtol=1e-12, atol=0)


def test_pp_charm_in_a_source():
    """In a pp-dominated source (setups.pp_source) charm hadrons are made and their decays feed the pp neutrino
    component; the protons cool a little faster. With charm off, none of this."""
    off, on = pp_source(charm=False), pp_source(charm=True)   # off: test_pp_kaons_in_a_source's run with kaons
    assert on.get_charm_hadrons().max() > 0 and not off.get_charm_hadrons().any()
    nu = on.get_neutrinos_charm_decay_proton_proton()
    assert nu.max() > 0 and not off.get_neutrinos_charm_decay_proton_proton().any()
    assert on.get_tau_neutrinos_charm_decay_proton_proton().max() > 0
    assert not on.get_neutrinos_charm_decay_photopion().any()
    # the protons lose the charm hadrons' energy too: pp cooling faster by 0.03-1 % (charm takes 1e-4 to 4e-3 of E_p
    # per collision, against AM3's pions' ~0.5)
    E = on.get_egrid_had()
    t_on, t_off = on.get_t_proton_proton_proton(), off.get_t_proton_proton_proton()
    fin = (t_on < 1e199) & (E > 2e12)
    r = t_off[fin] / t_on[fin]
    assert fin.sum() > 50 and np.all((r > 1.0001) & (r < 1.02)), (r.min(), r.max())


# ---------------------------------------------------------------- charm hadrons: evolution
# the code's masses [GeV] and mean lives [s] (PDG 2026): of D+-, of D0 and anti-D0, of Ds+-, of Lambda_c+ and
# anti-Lambda_c-
CHARM = {'D_plus': ('MASS_D', 'TAU_D'), 'D_zero': ('MASS_D0', 'TAU_D0'), 'Ds': ('MASS_DS', 'TAU_DS'),
         'Lambda_c': ('MASS_LC', 'TAU_LC')}
CHARM_SPECIES = ['get_D_plus', 'get_D_minus', 'get_D_zero', 'get_anti_D_zero', 'get_Ds_plus', 'get_Ds_minus',
                 'get_Lambda_c_plus', 'get_anti_Lambda_c_minus']


def magnetised(charm=True, **switches):
    """A photo-hadronic source in a field where pions and kaons cool before they decay above about 1e6 and 4e7 GeV
    (1e5 G), but charm hadrons not below 1e11 GeV, after 30 steps; as `hadronic` otherwise."""
    return hadronic_source(1e5, 30, charm=charm, pion_syn=True, muon_syn=True, **switches)


@pytest.fixture(scope='module')
def charm_runs():
    return {'on': magnetised(True), 'off': magnetised(False),
            'no_radiation': magnetised(True, charm_syn=False, charm_compton=False)}


def test_charm_timescales(charm_runs):
    a = charm_runs['on']
    E = a.get_egrid_had() / 1e9  # GeV
    for name, (m, tau) in CHARM.items():
        np.testing.assert_allclose(getattr(a, f'get_t_{name}_decay')(), E / MASS_GEV[m] * TAU_S[tau], rtol=1e-12)
    # synchrotron: t_D / t_pi = (m_D / m_pi)^4 at equal energy, as for the kaons
    ratio = a.get_t_charm_syn() / a.get_t_pion_syn()
    np.testing.assert_allclose(ratio, (MASS_GEV['MASS_D'] / M_PI) ** 4, rtol=1e-12)
    assert np.all(np.isfinite(a.get_t_charm_compton()))
    # escape: the charged ones as charged pions, D0 and anti-D0 as neutrons
    np.testing.assert_array_equal(a.get_t_charm_escape(), a.get_t_kaon_escape())
    np.testing.assert_array_equal(a.get_t_charm_neutral_escape(), a.get_t_kaon_neutral_escape())


def test_charm_extra_cooling_roundtrip():
    a = new(init=False)
    t = np.geomspace(1e3, 1e6, len(a.get_egrid_had()))
    a.set_t_charm_extra_cooling(t)
    np.testing.assert_allclose(a.get_t_charm_extra_cooling(), t, rtol=1e-12)


def test_charm_species(charm_runs):
    """All eight species are made; the charm hadrons, made with about 0.2 % of the pions' energy and decaying
    1e4-1e5 times faster, hold about 1e-7 of the pions' energy. They decay before they cool: their share of the
    nu_mu grows from about 1e-5 at the spectrum's peak to about 0.1 near its end, where the K0_L (which do not
    radiate) make most of the rest."""
    on = charm_runs['on']
    E = on.get_egrid_had()
    total = 0
    for g in CHARM_SPECIES:
        n = getattr(on, g)()
        assert n.max() > 0 and np.all(n >= 0), g
        total = total + n
    np.testing.assert_allclose(on.get_charm_hadrons(), total, rtol=1e-15)
    frac = np.sum(total * E) / np.sum(on.get_pions() * E)
    assert 1e-8 < frac < 1e-6, frac
    nu = on.get_muon_neutrinos()
    charm = sum(getattr(on, f'get_muon_neutrinos_charm_decay_{o}')() for o in ORIGINS)
    En = on.get_egrid_neutrinos()
    peak = np.argmax(nu * En)
    end = np.flatnonzero(nu * En > 1e-8 * np.max(nu * En))[-1]
    assert charm[peak] < 1e-4 * nu[peak] and charm[end] > 0.05 * nu[end], (charm[peak] / nu[peak], charm[end] / nu[end])


def test_charm_off_leaves_no_trace(charm_runs):
    off = charm_runs['off']   # no pp target protons (the pp case: test_pp_charm_in_a_source)
    for g in CHARM_SPECIES + ['get_photons_charm_syn', 'get_photons_charm_compton',
                              'get_neutrinos_charm_decay_photopion', 'get_tau_neutrinos_charm_decay_photopion',
                              'get_tau_anti_neutrinos_charm_decay_photopion']:
        assert not getattr(off, g)().any(), g


def test_charm_radiation_switches(charm_runs):
    on, off = charm_runs['on'], charm_runs['no_radiation']
    assert on.get_photons_charm_syn().max() > 0 and on.get_photons_charm_compton().max() > 0
    assert not off.get_photons_charm_syn().any() and not off.get_photons_charm_compton().any()
    assert np.all(off.get_t_charm_syn() >= 1e199) and np.all(off.get_t_charm_compton() >= 1e199)


# ---------------------------------------------------------------- decays of the kaons and the charm hadrons
latex = load_derivation('latex')   # the reader of the code's docstrings
TABLES_DOC = 'kaon species s at the energy fraction x_k = exp(-k dlnE)'   # kaon_decay_tables' docstring (decays.mojo)
KAON_KMAX, KMAX = 200, 400
KAONS = ['K+', 'K-', 'KS', 'KL']
DECAY_SPECIES = ['D+', 'D-', 'D0', 'anti-D0', 'Ds+', 'Ds-', 'Lambda_c+', 'anti-Lambda_c-']
PRODUCTS = ['pi+', 'pi-', 'pi0', 'mu+L', 'mu+R', 'mu-L', 'mu-R', 'K+', 'K-', 'K0S', 'K0L', 'p', 'n', 'e-', 'e+',
            'nu_e', 'anti-nu_e', 'nu_mu', 'anti-nu_mu', 'nu_tau', 'anti-nu_tau', 'gamma']
CHARGE = {'pi+': 1, 'pi-': -1, 'mu+L': 1, 'mu+R': 1, 'mu-L': -1, 'mu-R': -1, 'K+': 1, 'K-': -1, 'p': 1, 'e-': -1,
          'e+': 1}
LEPTON = {'e': {'e-': 1, 'nu_e': 1, 'e+': -1, 'anti-nu_e': -1},
          'mu': {'mu-L': 1, 'mu-R': 1, 'nu_mu': 1, 'mu+L': -1, 'mu+R': -1, 'anti-nu_mu': -1},
          'tau': {'nu_tau': 1, 'anti-nu_tau': -1}}
# the charges (antiparticles' antinucleons count as nucleons)
CHARM_PARTICLES = {'D+': 1, 'D0': 0, 'Ds+': 1, 'Lambda_c+': 1}


def numeric_grid():
    """The code's grid points x_k = exp(-k dlnE) (kaon_decay_tables' docstring), a numpy function of (k, dlnE)."""
    kk, D = sp.symbols('k Delta', real=True)
    latex.line(TABLES_DOC, DECAYS_MOJO)
    return sp.lambdify((kk, D), latex.plain(TABLES_DOC.split('x_k = ')[1], {'k': kk, 'dlnE': D}), 'numpy')


XK = numeric_grid()(np.arange(KMAX + 1), DLNE)  # grid points x_k


@pytest.fixture(scope='module')
def decay_tables_exe(tmp_path_factory):
    """tests/harness/decay_tables_mojo.mojo: the decay tables of physics/decays.mojo (`kaon_decay_tables`,
    `charm_decay_tables`)."""
    return harness.build('decay_tables_mojo', tmp_path_factory.mktemp('harness'))


@pytest.fixture(scope='module')
def kaon_decay_tables(decay_tables_exe):
    """Products per decay at x_k, T[kaon, product, k], k = 0 .. KAON_KMAX."""
    return harness.output(decay_tables_exe, 'kaon', repr(DLNE), KAON_KMAX).reshape(4, 13, KAON_KMAX + 1) * DLNE


@pytest.fixture(scope='module')
def charm_decay_tables(decay_tables_exe):
    """Products per decay at x_k, T[species, product, k], k = 0 .. KMAX, in the order of decays.mojo (DECAY_SPECIES,
    PRODUCTS: CP_PIP ... CP_GAM)."""
    return harness.output(decay_tables_exe, 'charm', repr(DLNE), KMAX).reshape(8, 22, KMAX + 1) * DLNE


def per_decay(T, s, p):
    """The products p per decay of the charm species s."""
    return np.sum(T[DECAY_SPECIES.index(s), PRODUCTS.index(p)])


@pytest.mark.parametrize('parent', KAONS + DECAY_SPECIES)
def test_decays_energy_and_positivity(kaon_decay_tables, charm_decay_tables, parent):
    """Per decay, the products carry the parent's energy (to 1e-11 for the kaons, 1e-8 for the charm hadrons), and
    no entry is negative."""
    if parent in KAONS:
        T, tol = kaon_decay_tables[KAONS.index(parent)], 1e-11
    else:
        T, tol = charm_decay_tables[DECAY_SPECIES.index(parent)], 1e-8
    assert np.sum(T * XK[:T.shape[-1]]) == pytest.approx(1.0, abs=tol)
    assert np.all(T >= 0.0)


@pytest.mark.parametrize('species', list(CHARM_PARTICLES))
def test_charm_decays_conserve_charge_baryons_and_leptons(charm_decay_tables, species):
    """The charm hadrons' decay products conserve charge, baryon number and the three lepton numbers."""
    charge = sum(q * per_decay(charm_decay_tables, species, p) for p, q in CHARGE.items())
    assert charge == pytest.approx(CHARM_PARTICLES[species], abs=1e-9)
    baryons = per_decay(charm_decay_tables, species, 'p') + per_decay(charm_decay_tables, species, 'n')
    assert baryons == pytest.approx(1.0 if species == 'Lambda_c+' else 0.0, abs=1e-9)
    for flavour, lep in LEPTON.items():
        lepton_number = sum(l * per_decay(charm_decay_tables, species, p) for p, l in lep.items())
        assert lepton_number == pytest.approx(0.0, abs=1e-9), flavour


def test_charm_antiparticles_decay_as_cp_conjugates(charm_decay_tables):
    """An antiparticle's table is its particle's with every product replaced by its CP conjugate."""
    conj = ['pi-', 'pi+', 'pi0', 'mu-R', 'mu-L', 'mu+R', 'mu+L', 'K-', 'K+', 'K0S', 'K0L', 'p', 'n', 'e+', 'e-',
            'anti-nu_e', 'nu_e', 'anti-nu_mu', 'nu_mu', 'anti-nu_tau', 'nu_tau', 'gamma']
    for s in range(0, 8, 2):
        for p, c in enumerate(conj):
            assert np.array_equal(charm_decay_tables[s, p], charm_decay_tables[s + 1, PRODUCTS.index(c)]), \
                (DECAY_SPECIES[s], PRODUCTS[p])


def test_charm_semileptonic_mu_plus_right_handed(charm_decay_tables):
    """D0 makes muons only in its semileptonic decays, whose mu+ is mostly right-handed in V-A."""
    assert per_decay(charm_decay_tables, 'D0', 'mu+R') > 3.0 * per_decay(charm_decay_tables, 'D0', 'mu+L')


@pytest.mark.parametrize('species', list(CHARM_PARTICLES))
def test_charm_semileptonic_and_tau_rates(charm_decay_tables, species):
    """The e+ per decay, less the e- (the e+e- pairs of Dalitz decays) and the e+ of the tau decays, is the code's
    inclusive e+ rate (incl of charm_decay_modes); one nu_tau per tau and one anti-nu_tau per tau decay, at the
    code's B(D(s)+ -> tau+ nu). The rates are read from decays.mojo (support/code_decays.py)."""
    modes, tau = code_charm_modes()[species], code_tau_modes()
    tau_nu = float(sum(md['br'] for md in modes['lep'] if 'tau+' in md['fs']))
    tau_to_e = float(sum(md['br'] for md in tau if 'e+' in md['fs'])) / float(sum(md['br'] for md in tau))
    e_sl = (per_decay(charm_decay_tables, species, 'e+') - per_decay(charm_decay_tables, species, 'e-')
            - tau_nu * tau_to_e)
    assert e_sl == pytest.approx(float(modes['incl']), rel=1e-9)
    assert per_decay(charm_decay_tables, species, 'nu_tau') == pytest.approx(tau_nu, abs=1e-12)
    assert per_decay(charm_decay_tables, species, 'anti-nu_tau') == pytest.approx(tau_nu, abs=1e-12)


# ---------------------------------------------------------------- neutrinos by parent
NU_FLAVOURS = ['electron_neutrinos', 'electron_anti_neutrinos', 'muon_neutrinos', 'muon_anti_neutrinos']
ORIGINS = ['photopion', 'proton_proton']


def parents(f):
    """The parents of flavour f's neutrinos in each origin (the interactions' pions make only nu_mu and anti-nu_mu)."""
    return ['pion', 'muon', 'kaon', 'charm'] if f.startswith('muon') else ['muon', 'kaon', 'charm']


def test_neutrinos_by_parent():
    """Neutrinos by parent, components of each flavour. For each origin (photo-hadronic interactions and pp
    collisions, at 1e10 target protons cm^-3 where both matter): the decays of the pions made there, of their muons,
    of the kaons (with their pion and muon products) and of the charm hadrons (with their kaon, pion and muon
    products). With pion, muon and neutron decay off, the kaon and charm parts are all neutrinos. With all decays on,
    the parts (and, of the anti-nu_e, those of neutron decay) add up to all neutrinos, and an origin's parts to AM3's
    component; the charm decays also make tau neutrinos.
    Beyond AM3's cutoff, where pions and muons cool before decaying, kaons make nearly all neutrinos. Of the nu_mu
    and anti-nu_mu from the interactions' pions, pion decays make less than 0.8 at the spectrum's bottom and nearly
    all at its top, where the longer-lived muons cool first. Without kaons there are no kaon parts, and the pion and
    muon parts make up AM3's components."""
    only = hadronic(off=('pion_decay', 'muon_decay', 'neutron_decay'), pp_density=1e10, charm=True)
    on, off = hadronic(True, pp_density=1e10, charm=True), hadronic(False, pp_density=1e10, charm=True)
    part = lambda a, f, parent, o: getattr(a, f'get_{f}_{parent}_decay_{o}')()
    for f in NU_FLAVOURS:
        total = getattr(only, f'get_{f}')()
        assert total.max() > 0
        np.testing.assert_allclose(sum(part(only, f, p, o) for p in ('kaon', 'charm') for o in ORIGINS), total,
                                   rtol=1e-12, atol=0)
        assert not any(part(only, f, p, o).any() for p in parents(f) if p not in ('kaon', 'charm')
                       for o in ORIGINS), f
        total, parts = getattr(on, f'get_{f}')(), 0
        for o in ORIGINS:
            for p in parents(f):
                x = part(on, f, p, o)
                assert x.max() > 0 and np.all(x >= 0) and np.all(x <= total * (1 + 1e-12)), (f, p, o)
                parts = parts + x
            component = getattr(on, f'get_{f}_{o}')()
            np.testing.assert_allclose(sum(part(on, f, p, o) for p in parents(f)), component, rtol=0,
                                       atol=1e-14 * component.max())
            assert not part(off, f, 'kaon', o).any(), (f, o)
            component = getattr(off, f'get_{f}_{o}')()
            np.testing.assert_allclose(sum(part(off, f, p, o) for p in parents(f) if p != 'kaon'), component, rtol=0,
                                       atol=1e-14 * component.max())
        if f == 'electron_anti_neutrinos':
            parts = parts + on.get_electron_anti_neutrinos_neutron_decay()
        np.testing.assert_allclose(parts, total, rtol=1e-12, atol=1e-12 * total.max())
    for p in ('pion', 'muon', 'kaon', 'charm'):
        for o in ORIGINS:
            np.testing.assert_allclose(getattr(on, f'get_neutrinos_{p}_decay_{o}')(),
                                       sum(part(on, f, p, o) for f in NU_FLAVOURS if p in parents(f)), rtol=1e-15)
    assert on.get_tau_neutrinos_charm_decay_photopion().max() > 0
    for o in ORIGINS:
        for g in (f'get_tau_neutrinos_charm_decay_{o}', f'get_tau_anti_neutrinos_charm_decay_{o}'):
            assert np.all(getattr(on, g)() >= 0), g
    nu_on, nu_off = on.get_muon_neutrinos(), off.get_muon_neutrinos()
    beyond = nu_off < 1e-3 * nu_on  # beyond AM3's cutoff
    assert beyond.sum() > 10
    kaons = sum(part(on, 'muon_neutrinos', 'kaon', o) for o in ORIGINS)
    assert np.all(kaons[beyond] > 0.99 * nu_on[beyond])
    E = on.get_egrid_neutrinos()
    for o in ORIGINS:  # muon damping
        pions = part(on, 'muon_neutrinos', 'pion', o) + part(on, 'muon_anti_neutrinos', 'pion', o)
        made = pions + part(on, 'muon_neutrinos', 'muon', o) + part(on, 'muon_anti_neutrinos', 'muon', o)
        sig = np.flatnonzero(E * made > 1e-4 * np.max(E * made))
        share = pions[sig] / made[sig]
        assert share[0] < 0.8 and share[-1] > 0.99, (o, share[0], share[-1])
