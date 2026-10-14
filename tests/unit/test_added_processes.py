"""Processes am3_mojo adds to AM3 (README "Processes am3_mojo adds to AM³"; tools/am3_mojo_api.json), checked
without the C++ code (their Python interface: tests/unit/test_api.py):

* e+e- annihilation (Svensson 1982): each annihilation removes one e- and one e+ and makes two photons with the
  pair's energy; the switches;
* triplet pair production (e+- gamma -> e+- e+ e-): its kernels, the tabulated cross section against Lee et al.'s
  expansions, the products of a step, the balances of a steady state, the switches and the pair component;
* neutron decay (n -> p e- anti-nu_e): its kernels, the decay rate and the products of a step in a closed box, the
  switch and the components;
* kaons: K+, K-, K0_S and K0_L from photo-hadronic interactions (pp collisions: tests/unit/test_pp_kaons.py), their
  escape and cooling (K+- as charged pions, K0_S and K0_L as neutrons), decay, switches and neutrinos by parent.
The switches are on by default; `mojo-am3` (tests/support/paths.py) turns them off.
"""
import numpy as np
import pytest

from support import harness
from support.paths import load_derivation
from support.setups import (CLOSED_BOX, GRID_TO_1E21_EV, PAIR_COMPONENTS, PROTONS_TO_1E20_EV, energy, line, new,
                            number, power_law, pulse_source, triplet_box, triplet_steady as steady_state)
from support.units import DLNE, LEP_UNITS, MASS_EV, MASS_GEV, MEC2_AM3_EV, MEC2_GRID_EV, TDEC_S


# ---------------------------------------------------------------- e+e- annihilation
def pair_box(ele, pos, dt=1.0, steps=1, **switches):
    """Only e+e- annihilation acts (no escape, radiation or other process);
    `ele`, `pos`: functions of the lepton grid energy [eV] -> EdN/dEdV."""
    a = new(off=CLOSED_BOX + ('triplet_pair_production',), dt=dt, **switches)   # (dt taken over by init_kernels)
    E = a.get_egrid_lep()
    a.set_current_densities_electrons(ele(E))
    a.set_current_densities_positrons(pos(E))
    n0 = (a.get_electrons(), a.get_positrons())
    for _ in range(steps):
        a.evolve_step()
    return a, n0


def slow(n):
    """Pairs in lepton bin 2 (gamma = e^0.2 = 1.221), the lowest bin AM3's
    solver evolves (bins 0 and 1 are set to zero every step)."""
    return lambda E: np.where(np.arange(len(E)) == 2, n, 0.0)


@pytest.mark.parametrize('ele, pos', [
    (slow(1e12), slow(1e12)),
    (slow(1e12), power_law(1e10, 1e6, 1e10)),
    (power_law(1e11, 1e6, 1e9), power_law(1e11, 1e6, 1e12)),
    (power_law(1e11, 6.3e5, 1e7, 3.0), power_law(5e10, 6.3e5, 1e8)),
], ids=['slow-slow', 'slow-power law', 'power laws', 'steep power law'])
def test_conservation(ele, pos):
    """Each annihilation removes one electron and one positron and makes two photons with the pair's total energy
    (to 4e-4, the solver's time stepping). Slow pairs (gamma = 1.221, see slow) make a line around 511 keV."""
    a, (e0, p0) = pair_box(ele, pos, dt=1.0, steps=3)
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
        inside = (Eg > 0.3 * MEC2_GRID_EV) & (Eg < 2.5 * MEC2_GRID_EV)
        assert np.sum(g[inside]) == pytest.approx(np.sum(g), rel=1e-6)


def test_annihilation_switches():
    n = 1e12
    a, (e0, _) = pair_box(slow(n), slow(n), pair_annihilation_emission=False)
    assert not a.get_photons_pair_annihilation().any() and number(a.get_electrons()) < number(e0)
    a, (e0, _) = pair_box(slow(n), slow(n), pair_annihilation_cooling=False)
    assert a.get_photons_pair_annihilation().any()
    np.testing.assert_array_equal(a.get_electrons(), e0)
    a, (e0, _) = pair_box(slow(n), slow(n), pair_annihilation=False)
    assert not a.get_photons_pair_annihilation().any()
    np.testing.assert_array_equal(a.get_electrons(), e0)
    # with positrons merged into the electrons the charges are unknown: no annihilation
    a, _ = pair_box(slow(n), slow(n), merge_positrons_into_electrons=True)
    assert not a.get_photons_pair_annihilation().any()


# ---------------------------------------------------------------- triplet pair production
# the README's threshold and high-energy expansions of the cross section, and the start of the latter's range (k >= 1e3)
sigma_threshold, _, sigma_asym, _, (_, K_ASYM) = load_derivation('triplet').numeric_expansions()
EPS_SOFT = 0.03  # target photons at 0.03 m_e c^2 (15 keV)
UPWIND = (1 - np.exp(-DLNE)) / DLNE  # energy a continuous loss removes in AM3's Chang & Cooper regime


def soft_photons(n, eps=EPS_SOFT):
    """Photons in the single bin at eps m_e c^2."""
    return line(n, eps * MEC2_GRID_EV)


def one_bin(n, gamma):
    """Leptons in the single bin at gamma."""
    return line(n, gamma * MEC2_GRID_EV)


@pytest.fixture(scope='module')
def triplet(tmp_path_factory):
    """The code's triplet pair production (tests/harness/triplet_mojo.mojo): its table nodes and kernel rows."""
    return harness.Triplet(tmp_path_factory.mktemp('harness'))


def test_triplet_cross_section_tables(triplet):
    """The tabulated QED cross section (tools/triplet) agrees with the exact threshold and
    high-energy expansions of Lee et al. (2021) where these hold (the README's, as the code uses them)."""
    k, s = triplet.k, triplet.sig
    hi = k >= K_ASYM
    np.testing.assert_allclose(s[hi], sigma_asym(k[hi]), rtol=5e-4)
    lo = k <= 4.2
    np.testing.assert_allclose(s[lo], sigma_threshold(k[lo]), rtol=3e-3)


def test_triplet_kernels(triplet):
    """Every interaction deposits one positron and one lower-energy electron, and the lepton loses exactly the
    pair's energy. No weight is negative. Above the last table node (2P = 1e10), the products' mean energies follow
    the extrapolated distributions without a jump."""
    rows = triplet.rows()
    P, R, L, (pos, low) = rows['P'], rows['R'], rows['L'], rows['W']
    x = [np.exp(-np.arange(len(w)) * rows['dl']) for w in pos]   # the energy fraction of each offset
    np.testing.assert_allclose([w.sum() for w in pos], R, rtol=1e-12)
    np.testing.assert_allclose([w.sum() for w in low], R, rtol=1e-12)
    np.testing.assert_allclose([(a + b) @ xo for a, b, xo in zip(pos, low, x)], L, rtol=1e-12)
    assert min(min(a.min(), b.min()) for a, b in zip(pos, low)) >= 0.0
    # k <x> / ln^2(2k) changes slowly with k (0.35 ... 0.39): no jump where the extrapolation starts
    mean = np.array([(a + b) @ xo / R[q] for q, (a, b, xo) in enumerate(zip(pos, low, x))])   # <x+> + <x_low>
    f = mean * P / np.log(4 * P) ** 2
    j = np.flatnonzero(2 * P > triplet.k[-1])[0]
    assert np.max(np.abs(np.diff(np.log(f[j - 10:j + 10])))) < 2e-3


@pytest.mark.parametrize('gamma', [1e3, 1e5, 3e7])
def test_triplet_products(gamma):
    """A monoenergetic electron beam on monoenergetic photons: in a short step, as many electrons appear as positrons
    are made, and the pair component holds both. The electrons lose the pair's energy as a continuous loss, of which
    AM3's Chang & Cooper scheme removes (1 - e^-dlnE)/dlnE, as of the synchrotron and inverse-Compton losses."""
    ne, nph = 1e8, 1e14
    dt = 1e-3
    a, (e0, p0, ph0) = triplet_box(one_bin(ne, gamma), pho=soft_photons(nph), dt=dt)
    El = a.get_egrid_lep()
    npos = number(a.get_positrons())
    assert npos > 0
    assert number(a.get_electrons()) - number(e0) == pytest.approx(npos, rel=1e-8)
    assert number(a.get_pairs_triplet()) == pytest.approx(2 * npos, rel=1e-12)
    pairs = energy(a.get_pairs_triplet(), El)
    lep = energy(a.get_electrons(), El) + energy(a.get_positrons(), El)
    assert lep - energy(e0, El) == pytest.approx((1 - UPWIND) * pairs, rel=1e-4)


def triplet_steady(on, merged=False, nph=3e11, t_esc=1e4):
    """A steady state: setups.triplet_steady after 8 steps of dt = 1e4 t_esc. At nph = 3e11 cm^-3 the triplet energy
    loss stays below 1 % of the escape rate, so AM3's solver evolves the leptons with escape only (its regime 1,
    exact for a steady state)."""
    return steady_state(nph, 1e4 * t_esc, 8, t_esc, triplet_pair_production=on, merge_positrons_into_electrons=merged)


def test_triplet_steady_state():
    """In a steady state every interaction absorbs one photon and makes one e+ and one e-. Between the runs with and
    without the process, the extra e- and the missing photons both equal the e+. The pair component holds twice as
    many, and the e+ escape as fast as the photon solver absorbs photons. The missing photons, a difference of 1.7e-8
    of all photons, carry round-off (hence the tolerance 1e-6)."""
    a, b = triplet_steady(True), triplet_steady(False)
    assert number(b.get_positrons()) == 0.0
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
    ele = one_bin(1e8, 1e5)
    a, (e0, _, ph0) = triplet_box(ele, triplet_pair_production_emission=False)
    assert not a.get_positrons().any() and not a.get_pairs_triplet().any()
    El = a.get_egrid_lep()
    assert number(a.get_electrons()) == pytest.approx(number(e0), rel=1e-12)
    assert energy(a.get_electrons(), El) < energy(e0, El)
    a, (e0, _, ph0) = triplet_box(ele, triplet_pair_production_cooling=False)
    assert a.get_positrons().any()
    np.testing.assert_array_equal(a.get_electrons()[np.argmax(e0)], e0[np.argmax(e0)])
    a, (e0, _, ph0) = triplet_box(ele, triplet_pair_production=False)
    np.testing.assert_array_equal(a.get_electrons(), e0)
    np.testing.assert_array_equal(a.get_photons(), ph0)
    assert not a.get_positrons().any()


@pytest.mark.parametrize('annihilation', [True, False])
def test_pair_components_add_up_with_triplet(annihilation):
    """In a source with injection, radiation and escape, the pair components (including the triplet pairs) add up
    to all pairs, with and without the e-/e+ halves of annihilation."""
    # The components add up exactly unless a solve sets a negative density to zero, as AM3 does after each step.
    # At dt = 1e4 s, a pulse of pairs cooled by inverse Compton on the first synchrotron photons flows into the
    # Crank-Nicolson window below, which undershoots its top cell below zero in some components but not in all
    # pairs: the sum differs by 9e-2 in that bin (tests/solver_checks/window_top_undershoot.py). At 3000 s no
    # density goes negative.
    a = pulse_source(3e3, pair_annihilation=annihilation)
    for _ in range(60):
        a.evolve_step()
    comp = sum(getattr(a, getter)() for getter in PAIR_COMPONENTS)
    np.testing.assert_allclose(comp, a.get_pairs(), rtol=1e-6, atol=1e-12 * a.get_pairs().max())
    assert a.get_pairs_triplet().any()


# ---------------------------------------------------------------- neutron decay
M_N_EV, TAU_N = MASS_EV['massN_PDG'], TDEC_S['tdec_n']   # the code's neutron mass and mean life


@pytest.fixture(scope='module')
def neutron_decay_kernels(tmp_path_factory):
    """The kernels of the default grid (tests/harness/neutron_decay_mojo.mojo), as harness.neutron_decay_kernels
    gives them: the grids [GeV], K_e, K_nu and the proton shares."""
    return harness.neutron_decay_kernels(tmp_path_factory.mktemp('harness'))


def test_neutron_decay_kernels(neutron_decay_kernels):
    """Per decay: one electron, one antineutrino and one proton, which carry the neutron's energy less what falls
    below the grids (electrons below 0.511 MeV, where AM3's lepton grid starts; antineutrinos below 306 eV). No
    weight is negative. For ultra-relativistic neutrons the antineutrinos take 5.11e-4 of the energy (Lipari et al.
    2007: 5.1e-4)."""
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


def neutron_box(n_of_E, dt=1.0, **switches):
    """Only neutron decay acts (no escape, no other process): neutrons with the density n_of_E(E [eV]),
    one step of dt."""
    a = new(off=('escape', 'electron_syn', 'electron_compton', 'annihilation', 'ssa', 'proton_syn', 'proton_compton',
                 'bethe_heitler', 'photopion', 'pp', 'expansion', 'adiabatic_cooling', 'pair_annihilation',
                 'triplet_pair_production', 'kaons'), dt=dt, **switches)
    n0 = n_of_E(a.get_egrid_had())
    a.set_current_densities_neutrons(n0)
    a.evolve_step()
    return a, n0


def test_neutron_decay_box():
    """Neutrons of 0.1-100 TeV in a closed box decay at the rate 1/(gamma tau_n): one step removes
    n (1 - exp(-dt/(gamma tau_n))). As in AM3's decays, the products come from the evolved neutrons: one proton,
    one electron and one antineutrino per decay, with the neutron's energy, and no positron. They are also the
    components."""
    a, n0 = neutron_box(lambda E: np.where((E >= 1e11) & (E <= 1e14), 1e11 / E, 0.0))
    E, Enu = a.get_egrid_had(), a.get_egrid_neutrinos()
    # the lepton grid in AM3's unit of the hadronic e+- sources, 0.511 MeV (get_egrid_lep() reports
    # its energies with m_e c^2 = 510998.2 eV, 3.5e-6 lower)
    El = MEC2_AM3_EV * np.exp(DLNE * np.arange(a.get_grid_size_lep()))
    t = a.get_t_neutron_decay()
    np.testing.assert_allclose(t, E / M_N_EV * TAU_N, rtol=1e-13)
    n = a.get_neutrons()
    np.testing.assert_allclose(n, n0 * np.exp(-1.0 / t), rtol=1e-12)
    decays = number(n / t)  # per second
    p, e, nu = a.get_protons(), a.get_electrons(), a.get_electron_anti_neutrinos()
    assert number(p) == pytest.approx(decays, rel=1e-13)
    assert number(e) == pytest.approx(decays * LEP_UNITS, rel=1e-12)
    assert number(nu) == pytest.approx(decays, rel=1e-6)  # (below the neutrino grid: 1.5e-7 at most)
    assert energy(p, E) + energy(e, El) / LEP_UNITS + energy(nu, Enu) == pytest.approx(energy(n / t, E),
                                                                                       rel=1e-12)
    assert energy(nu, Enu) / energy(n / t, E) == pytest.approx(5.1e-4, rel=0.01)
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
    """A photo-hadronic source with a long escape time (1e6 s), where neutrons up to about 1 TeV
    decay before they escape (40 steps of 1e4 s)."""
    a = new()
    a.set_mag_field(1.0)
    a.set_escape_timescale(1e6)
    a.set_solver_time_step(1e4)
    a.set_powerlaw_injection_parameters_protons(1e48, 1e46, 1e10, 1e10, 1e18, 2.0, 2.0, 1.0)
    a.set_powerlaw_injection_parameters_electrons(1e48, 1e43, 1e7, 1e7, 1e11, 2.0, 2.0, 1.0)
    for _ in range(40):
        a.evolve_step()
    return a


def test_neutron_decay_components(neutron_source):
    """In a source the e+- components, with the electrons from neutron decay, add up to all pairs, and the electron
    antineutrinos of photo-hadronic interactions, pp collisions and neutron decay add up to all of them. Those of
    neutron decay make most of them below 1 GeV, and less than 1e-3 above 1 TeV."""
    a = neutron_source
    comp = (a.get_pairs_injected() + a.get_pairs_bethe_heitler() + a.get_pairs_annihilation()
            + a.get_pairs_photopion() + a.get_pairs_proton_proton() + a.get_pairs_triplet()
            + a.get_electrons_neutron_decay())
    np.testing.assert_allclose(comp, a.get_pairs(), rtol=1e-6, atol=1e-12 * a.get_pairs().max())
    nu = a.get_electron_anti_neutrinos()
    nd_nu = a.get_electron_anti_neutrinos_neutron_decay()
    comp = a.get_electron_anti_neutrinos_photopion() + a.get_electron_anti_neutrinos_proton_proton() + nd_nu
    np.testing.assert_allclose(comp, nu, rtol=1e-9, atol=1e-12 * nu.max())
    assert a.get_electrons_neutron_decay().any() and a.get_photons_neutron_decay_electrons_syn_compton().any()
    Enu = a.get_egrid_neutrinos()
    low, high = (Enu < 1e9) & (nu > 0), (Enu > 1e12) & (nu > 0)
    assert low.sum() > 30 and np.all(nd_nu[low] > 0.9 * nu[low])
    assert np.all(nd_nu[high] < 1e-3 * nu[high])


# ---------------------------------------------------------------- kaons
# the code's masses [GeV] and mean lives [s]: PDG 2026 for the kaons, AM3's charged pion
M_K, TAU_K, M_PI = MASS_GEV['massK'], TDEC_S['tdec_K'], MASS_GEV['masspip']
M_K0, TAU_KS, TAU_KL = MASS_GEV['massK0'], TDEC_S['tdec_KS'], TDEC_S['tdec_KL']


def hadronic(kaons=True, kaon_syn=True, kaon_ic=True, b=100.0, steps=30, off=()):
    """A photo-hadronic source with a strong field, where kaons matter; `off`: further processes to switch off."""
    a = new(off=off, grid=GRID_TO_1E21_EV, kaons=kaons, kaon_syn=kaon_syn, kaon_compton=kaon_ic, pion_syn=True,
            muon_syn=True)
    a.set_mag_field(b)
    a.set_escape_timescale(1e5)
    a.set_solver_time_step(1e2)  # stable for these fast-cooling pions
    a.set_powerlaw_injection_parameters_protons(*PROTONS_TO_1E20_EV)
    a.set_powerlaw_injection_parameters_electrons(1e47, 1e46, 1e7, 1e7, 1e12, 2.0, 2.0, 1.0)
    for _ in range(steps):
        a.evolve_step()
    return a


@pytest.fixture(scope='module')
def kaon_runs():
    return {'on': hadronic(True), 'off': hadronic(False),
            'no_radiation': hadronic(True, kaon_syn=False, kaon_ic=False)}


def test_kaon_timescales(kaon_runs):
    a = kaon_runs['on']
    E = a.get_egrid_had() / 1e9  # GeV
    # decay: E / m_K * tau_K
    np.testing.assert_allclose(a.get_t_kaon_decay(), E / M_K * TAU_K, rtol=1e-12)
    # synchrotron: t ∝ m^4 / E, so at equal energy t_K / t_pi = (m_K / m_pi)^4
    ratio = a.get_t_kaon_syn() / a.get_t_pion_syn()
    np.testing.assert_allclose(ratio, (M_K / M_PI) ** 4, rtol=1e-12)
    assert np.all(np.isfinite(a.get_t_kaon_compton()))


def kaon_step(t_esc=1e3, frac_neutrons=1.0, frac_pions=1.0):
    """One step of length t_esc from no kaons, with kaon decay and cooling
    off: nucleons on soft photons make kaons, which only escape."""
    a = new(off=('kaon_decay', 'kaon_syn', 'kaon_compton'), grid=GRID_TO_1E21_EV)
    a.set_escape_timescale(t_esc)
    a.set_solver_time_step(t_esc)
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
        # (the source changes by < 1e-4 with the neutrons' escape: the step's
        #  kaons come from the evolved nucleons, as AM3's pions do)
        np.testing.assert_allclose(nn[sig] / n0[sig], 1.0 if ch else step(0.25) / step(1.0),
                                   rtol=1e-3, err_msg=g)
        np.testing.assert_allclose(nc[sig] / n0[sig], step(0.5) / step(1.0) if ch else 1.0,
                                   rtol=1e-12, err_msg=g)


@pytest.mark.parametrize('source', ['photo-hadronic', 'pp'])
def test_kaon_compton_cooling_thomson_scaling(source):
    """K+- cool by inverse Compton as charged pions do, with the kaon mass: on soft photons (Thomson regime)
    t_K / t_pi = (m_K / m_pi)^4 at equal energy. With kaons from pp collisions only, the kaon getters give the
    rates of the pp population."""
    a = new(**(dict(photopion=False, pp=True) if source == 'pp' else {}))
    if source == 'pp':
        a.set_pp_target_proton_density(1e10)
    a.set_powerlaw_injection_parameters_protons(*PROTONS_TO_1E20_EV)
    eg = a.get_egrid_photons()
    a.set_current_densities_photons(np.where((eg > 1e-4) & (eg < 1e-3), 1e10, 0.0))
    a.evolve_step()
    E = a.get_egrid_had()
    # gamma eps << m_e c^2 for the pions, and gamma_K >> 1
    thomson = (E / (M_PI * 1e9) * 1e-3 < 1e-2 * MEC2_AM3_EV) & (E / (M_K * 1e9) > 30)
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
    # strangeness goes with the K+ into the hyperons, and near threshold only
    # K+ and K0 are made: fewer K- than K+
    assert 0.02 < np.sum(km * E) / np.sum(kp * E) < 1
    np.testing.assert_allclose(a.get_t_kaon_0S_decay(), E / 1e9 / M_K0 * TAU_KS, rtol=1e-12)
    np.testing.assert_allclose(a.get_t_kaon_0L_decay(), E / 1e9 / M_K0 * TAU_KL, rtol=1e-12)
    # K0_S and K0_L are produced equally. Where both decay much faster than they escape and than the time
    # step (steady state), their densities are in the ratio of their lifetimes.
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


NU_FLAVOURS = ['electron_neutrinos', 'electron_anti_neutrinos', 'muon_neutrinos', 'muon_anti_neutrinos']


def test_kaon_decay_neutrino_components(kaon_runs):
    """The neutrinos by parent, components of each flavour: kaon decays (with the pions and muons they decay into),
    and the pions of photo-hadronic interactions and pp collisions (with their muons). With pion, muon and neutron
    decay off, the kaon part is all neutrinos. With every decay on, the parts add up to all neutrinos and to AM3's
    photo-hadronic and pp components (which hold the kaons by origin), and the kaons make nearly all neutrinos
    beyond AM3's cutoff, where pions and muons cool before they decay. Without kaons there is no kaon part, and the
    pion parts are AM3's components."""
    only = hadronic(off=('pion_decay', 'muon_decay', 'neutron_decay'))
    on, off = kaon_runs['on'], kaon_runs['off']
    for f in NU_FLAVOURS:
        total = getattr(only, f'get_{f}')()
        assert total.max() > 0
        np.testing.assert_allclose(getattr(only, f'get_{f}_kaon_decay')(), total, rtol=1e-12, atol=0)
        k, total = getattr(on, f'get_{f}_kaon_decay')(), getattr(on, f'get_{f}')()
        assert k.max() > 0 and np.all(k >= 0) and np.all(k <= total * (1 + 1e-12)), f
        pions = (getattr(on, f'get_{f}_pion_muon_decay_photopion')()
                 + getattr(on, f'get_{f}_pion_muon_decay_proton_proton')())
        parts = pions + k
        if f == 'electron_anti_neutrinos':
            parts = parts + on.get_electron_anti_neutrinos_neutron_decay()
        np.testing.assert_allclose(parts, total, rtol=1e-12, atol=1e-12 * total.max())
        by_origin = getattr(on, f'get_{f}_photopion')() + getattr(on, f'get_{f}_proton_proton')()
        np.testing.assert_allclose(pions + k, by_origin, rtol=0, atol=1e-14 * by_origin.max())
        assert not getattr(off, f'get_{f}_kaon_decay')().any(), f
        np.testing.assert_array_equal(getattr(off, f'get_{f}_pion_muon_decay_photopion')(),
                                      getattr(off, f'get_{f}_photopion')())
    for c in ('kaon_decay', 'pion_muon_decay_photopion', 'pion_muon_decay_proton_proton'):
        np.testing.assert_allclose(getattr(on, f'get_neutrinos_{c}')(),
                                   sum(getattr(on, f'get_{f}_{c}')() for f in NU_FLAVOURS), rtol=1e-15)
    nu_on, nu_off = on.get_muon_neutrinos(), off.get_muon_neutrinos()
    beyond = nu_off < 1e-3 * nu_on  # beyond AM3's cutoff
    assert beyond.sum() > 10
    assert np.all(on.get_muon_neutrinos_kaon_decay()[beyond] > 0.99 * nu_on[beyond])


def test_no_kaons_without_photopion_emission():
    a = new(grid=GRID_TO_1E21_EV, photopion_emission=False)
    a.set_powerlaw_injection_parameters_protons(*PROTONS_TO_1E20_EV)
    for _ in range(5):
        a.evolve_step()
    assert not a.get_kaons().any()
