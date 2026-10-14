"""Set-ups of am3_mojo that the unit tests and the solver checks share: objects with some of the processes off, the
populations they put in, a population's number and energy, and the sources both run (triplet pair production alone
and in a steady state, a pulse of electrons with radiation and escape)."""
import numpy as np

from support.output import quiet
from support.paths import new_am3
from support.units import DLNE, MEC2_GRID_EV

# the processes off when the leptons' own synchrotron cooling and escape act alone (with the switches of a test)
LEPTONS_ALONE = ('hadronic', 'electron_compton', 'annihilation', 'ssa', 'proton_syn', 'expansion', 'adiabatic_cooling')
# and also without radiation and escape: one process acting in a closed box
CLOSED_BOX = LEPTONS_ALONE + ('electron_syn', 'escape')
# the processes am3_mojo adds that act on leptons
ADDED_LEPTONIC = ('pair_annihilation', 'triplet_pair_production')
# update_energy_grid for the kaons of the highest proton energies: photons from 1e-6 eV, neutrinos from 1e2 eV, every
# particle up to 1e21 eV
GRID_TO_1E21_EV = (1e-6, 1e2, 1e21)
# set_powerlaw_injection_parameters_protons: 1e46 erg/s into 1e48 cm^3 as E^-2 from 1e10 to 1e20 eV
PROTONS_TO_1E20_EV = (1e48, 1e46, 1e10, 1e10, 1e20, 2.0, 2.0, 1.0)


def new(impl='mojo', off=(), dt=None, grid=None, init=True, **switches):
    """A fresh, quiet AM3 object of `impl`: the processes `off` switched off, the given switches set (name=on), the
    energy grid (update_energy_grid) and time step dt set if given, and the kernels initialised silently unless
    init=False."""
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


def per_ln_e(E, lo=1e9, hi=1e12):
    """The injection lo/E per ln E from lo to hi [eV] (1 at lo; E^-2 per unit energy) on the grid's energies E."""
    return np.where((E >= lo) & (E <= hi), lo / E, 0.0)


def triplet_box(ele, pos=None, pho=None, dt=1.0, steps=1, impl='mojo', **switches):
    """Only triplet pair production acts (no escape, radiation, gamma-gamma pair production or e+e- annihilation);
    `ele`, `pos`: functions of the lepton energy [eV], `pho` of the photon energy [eV] -> E dN/dE dV (by default
    1e14 photons at 0.03 m_e c^2). Returns the object after `steps` steps and the initial (e-, e+, photons)."""
    a = new(impl, CLOSED_BOX + ('pair_annihilation',), dt, **switches)
    E, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    a.set_current_densities_electrons(ele(E))
    a.set_current_densities_positrons(pos(E) if pos else np.zeros(E.size))
    a.set_current_densities_photons(pho(Eg) if pho else line(1e14, 0.03 * MEC2_GRID_EV)(Eg))
    n0 = (a.get_electrons(), a.get_positrons(), a.get_photons())
    for _ in range(steps):
        a.evolve_step()
    return a, n0


def triplet_steady(n_gamma, dt, steps, t_esc=1e4, impl='mojo', **switches):
    """Electrons injected as per_ln_e(E) from 1e9 to 1e12 eV and photons at 0.03 m_e c^2 (n_gamma per t_esc
    injected), both escaping in t_esc, with triplet pair production the only other process (no synchrotron, no
    inverse Compton), after `steps` steps of dt. Returns the object."""
    a = new(impl, LEPTONS_ALONE + ('electron_syn', 'pair_annihilation'), dt, **switches)
    a.set_escape_timescale(t_esc)
    E, Eg = a.get_egrid_lep(), a.get_egrid_photons()
    a.set_injection_rate_electrons(per_ln_e(E))
    a.set_injection_rate_photons(line(n_gamma / t_esc, 0.03 * MEC2_GRID_EV)(Eg))
    for _ in range(steps):
        a.evolve_step()
    return a


def pulse_source(dt, impl='mojo', **switches):
    """Electrons injected with 1e46 erg/s into 1e48 cm^3 as a power law E^-2 from 1e9 to 1e14 eV, in 1 G, escaping
    in 1e6 s, with all leptonic processes and the time step dt; the hadronic ones off. Over its first steps the
    cooling grows hundreds of times (inverse Compton on the first synchrotron photons). Not yet evolved."""
    a = new(impl, ('hadronic',), **switches)
    a.set_mag_field(1.0)
    a.set_escape_timescale(1e6)
    a.set_solver_time_step(dt)
    a.set_powerlaw_injection_parameters_electrons(1e48, 1e46, 1e9, 1e9, 1e14, 2.0, 2.0, 1.0)
    return a


# the components of the pairs, which add up to all of them
PAIR_COMPONENTS = ('get_pairs_injected', 'get_pairs_bethe_heitler', 'get_pairs_annihilation', 'get_pairs_photopion',
                   'get_pairs_proton_proton', 'get_pairs_triplet')
