"""Set-ups of am3_mojo shared by the tests (tests/unit/, the solver tests of tests/reference/) and the triplet
scripts (tests/literature_checks/triplet/below_grid.py, steady_states.py): objects with some processes off, the
populations they put in, a population's number and energy, and the sources both run."""
import functools

import numpy as np

from support.output import quiet
from support.paths import new_am3
from support.units import DLNE, MEC2_EV

# processes off so that the leptons' own synchrotron cooling and escape act alone (with the switches of a test)
LEPTONS_ALONE = ('hadronic', 'electron_compton', 'annihilation', 'ssa', 'proton_syn', 'expansion', 'adiabatic_cooling')
# also without radiation and escape: one process acting in a closed box
CLOSED_BOX = LEPTONS_ALONE + ('electron_syn', 'escape')
# the processes am3_mojo adds that act on leptons
ADDED_LEPTONIC = ('pair_annihilation', 'triplet_pair_production')
# update_energy_grid to 1e21 eV for the kaons of the highest proton energies (photons from 1e-6, neutrinos from 1e2 eV)
GRID_TO_1E21_EV = (1e-6, 1e2, 1e21)
# set_powerlaw_injection_parameters_protons: 1e46 erg/s into 1e48 cm^3 as E^-2 from 1e10 to 1e20 eV
PROTONS_TO_1E20_EV = (1e48, 1e46, 1e10, 1e10, 1e20, 2.0, 2.0, 1.0)


def new(impl='mojo', off=(), dt=None, grid=None, init=True, **switches):
    """A fresh, quiet AM3 object of `impl`: the processes `off` switched off, the switches set (name=on), the energy
    grid (update_energy_grid) and time step dt set if given, the kernels initialised silently unless init=False."""
    a = new_am3(impl)
    a.set_verbosity_level(False)
    for name in off:
        getattr(a, 'set_process_' + name)(False)
    for name, on in switches.items():
        getattr(a, 'set_process_' + name)(on)
    if grid is not None:
        a.update_energy_grid(*grid)
    if dt is not None:
        a.set_solver_time_step(dt)
    if init:
        quiet(a.init_kernels)
    return a


def evolve(a, n):
    """a after n more steps."""
    for _ in range(n):
        a.evolve_step()
    return a


def number(n):
    """The number of particles of a population per ln E on a grid (sum n dlnE)."""
    return np.sum(n) * DLNE


def energy(n, E):
    """Their energy (sum n E dlnE)."""
    return np.sum(n * E) * DLNE


def line(n, E0):
    """n in the single bin at the energy E0 [eV], as a function of the grid's energies."""
    return lambda E: np.where(np.abs(np.log(E / E0)) < 0.05, n, 0.0)


def power_law(n0, emin, emax, s=2.0):
    """n0 (E/emin)^(1-s) per ln E from emin to emax [eV], as a function of the grid's energies."""
    return lambda E: np.where((E >= emin) & (E <= emax), n0 * (E / emin) ** (1 - s), 0.0)


def per_ln_e(E):
    """The injection 1e9 eV / E per ln E from 1e9 to 1e12 eV (E^-2 per unit energy) on the grid's energies E."""
    return np.where((E >= 1e9) & (E <= 1e12), 1e9 / E, 0.0)


def triplet_box(ele, pho=None, dt=1.0, **switches):
    """Only triplet pair production acts (no escape, radiation, gamma-gamma pair production or e+e- annihilation);
    `ele`: a function of the lepton energy [eV], `pho` of the photon energy [eV] -> E dN/dE dV (by default 1e14
    photons at 0.03 m_e c^2). Returns the object after one step and the initial (electrons, photons)."""
    a = new(off=CLOSED_BOX + ('pair_annihilation',), dt=dt, **switches)
    E, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    a.set_current_densities_electrons(ele(E))
    a.set_current_densities_photons(pho(Eg) if pho else line(1e14, 0.03 * MEC2_EV)(Eg))
    n0 = (a.get_electrons(), a.get_photons())
    a.evolve_step()
    return a, n0


def triplet_steady(n_gamma, dt, steps, t_esc, **switches):
    """Electrons injected as per_ln_e(E) and photons at 0.03 m_e c^2 (n_gamma per t_esc), both escaping in t_esc,
    with triplet pair production the only other process (no synchrotron, no inverse Compton): the object after
    `steps` steps of dt."""
    a = new(off=LEPTONS_ALONE + ('electron_syn', 'pair_annihilation'), dt=dt, **switches)
    a.set_escape_timescale(t_esc)
    E, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    a.set_injection_rate_electrons(per_ln_e(E))
    a.set_injection_rate_photons(line(n_gamma / t_esc, 0.03 * MEC2_EV)(Eg))
    return evolve(a, steps)


def pulse_source(dt, **switches):
    """Electrons injected with 1e46 erg/s into 1e48 cm^3 as E^-2 from 1e9 to 1e14 eV, in 1 G, escaping in 1e6 s,
    with all leptonic processes (hadronic ones off) and time step dt; not yet evolved. Over the first steps the
    cooling grows hundreds of times (inverse Compton on the first synchrotron photons)."""
    a = new(off=('hadronic',), **switches)
    a.set_mag_field(1.0)
    a.set_escape_timescale(1e6)
    a.set_solver_time_step(dt)
    a.set_powerlaw_injection_parameters_electrons(1e48, 1e46, 1e9, 1e9, 1e14, 2.0, 2.0, 1.0)
    return a


def hadronic_source(bfield, steps, pp_density=0.0, electrons=True, **kw):
    """new(**kw) on the grid to 1e21 eV, in bfield [G], with pp_density [cm^-3] target protons, escape in 1e5 s, the
    protons PROTONS_TO_1E20_EV and, unless electrons=False, 1e46 erg/s of electrons into 1e47 cm^3 as E^-2 from 1e7 to
    1e12 eV: the object after `steps` steps of 1e2 s."""
    a = new(grid=GRID_TO_1E21_EV, **kw)
    a.set_mag_field(bfield)
    a.set_escape_timescale(1e5)
    a.set_solver_time_step(1e2)  # stable for fast-cooling pions
    a.set_pp_target_proton_density(pp_density)
    a.set_powerlaw_injection_parameters_protons(*PROTONS_TO_1E20_EV)
    if electrons:
        a.set_powerlaw_injection_parameters_electrons(1e47, 1e46, 1e7, 1e7, 1e12, 2.0, 2.0, 1.0)
    return evolve(a, steps)


@functools.lru_cache(maxsize=1)
def pp_source(**switches):
    """The pp-dominated source of test_pp_kaons_in_a_source and test_pp_charm_in_a_source
    (tests/unit/test_added_processes.py): no photo-pion production, 1e10 target protons per cm^3, 10 G, no electrons;
    after 20 steps. The last one made is kept, so that both tests use the one without charm."""
    return hadronic_source(10.0, 20, 1e10, electrons=False, photopion=False, pp=True, **switches)


# the pair components, which add up to all pairs
PAIR_COMPONENTS = ('get_pairs_injected', 'get_pairs_bethe_heitler', 'get_pairs_annihilation', 'get_pairs_photopion',
                   'get_pairs_proton_proton', 'get_pairs_triplet')
