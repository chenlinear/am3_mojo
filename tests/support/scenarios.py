"""Simulation scenarios shared by the verification and benchmark scripts.

Every scenario drives an `AM3()` object exclusively through the public
Python API, so the very same code runs with am3_mojo and with the original
C++ AM3 module.

A scenario function configures the object, initialises kernels and returns
a generator of "checkpoints": after each yielded step count the caller may
dump the state.
"""
import numpy as np

C_CGS = 2.99792458e10
ME_C2_EV = 510998.95
MP_C2_EV = 938272088.16
SIGMA_T = 6.6524587321e-25
ME_G = 9.1093837015e-28
MP_G = 1.67262192369e-24


def _blazar_switches(am3):
    am3.set_verbosity_level(False)
    am3.set_process_parse_sed(True)
    am3.set_process_escape(True)
    am3.set_process_adiabatic_cooling(False)
    am3.set_process_expansion(False)
    am3.set_process_electron_syn(True)
    am3.set_process_ssa(True)
    am3.set_process_proton_syn(True)
    am3.set_process_pion_syn(False)
    am3.set_process_muon_syn(False)
    am3.set_process_electron_compton(True)
    am3.set_process_proton_compton(True)
    am3.set_process_pion_compton(False)
    am3.set_process_muon_compton(False)
    am3.set_process_pion_decay(True)
    am3.set_process_muon_decay(True)
    am3.set_process_annihilation(False)
    am3.set_process_photopion(True)
    am3.set_process_pp(False)
    am3.set_process_bethe_heitler(True)


def _blazar_params(am3, time_step_frac=1e-2, bfield=10.0):
    radius = 1e15
    volume = 4 / 3.0 * np.pi * radius ** 3
    injpower_e = 10.0 ** (-7.47) * 4 * np.pi * radius * (ME_G * C_CGS ** 3 / SIGMA_T)
    injpower_p = 10 ** (-4.93) * 4 * np.pi * radius * (MP_G * C_CGS ** 3 / SIGMA_T)
    am3.set_mag_field(bfield)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_solver_time_step(time_step_frac * radius / C_CGS)
    am3.set_powerlaw_injection_parameters_electrons(
        volume, injpower_e, 2.0 * ME_C2_EV, 2.0 * ME_C2_EV, 1e3 * ME_C2_EV, 1.9, 1.9, 1.0)
    am3.set_powerlaw_injection_parameters_protons(
        volume, injpower_p, 2.0 * MP_C2_EV, 2.0 * MP_C2_EV, 1e8 * MP_C2_EV, 1.9, 1.9, 1.0)


def blazar_simple(am3, n_steps=800, every=100):
    """docs/examples/blazar_simple_example.ipynb (proton-synchrotron blazar),
    8 light-crossing times with dt = R/c/100."""
    _blazar_switches(am3)
    am3.init_kernels()
    _blazar_params(am3)
    done = 0
    while done < n_steps:
        k = min(every, n_steps - done)
        for _ in range(k):
            am3.evolve_step()
        done += k
        yield done


def leptohadronic_all(am3, n_steps=300, every=50):
    """Every process switched on (pp with thermal target protons, pion/muon
    synchrotron and IC, gamma-gamma annihilation, expansion + adiabatic
    cooling, quantum synchrotron) with a strong field."""
    am3.set_verbosity_level(False)
    am3.set_process_parse_sed(True)
    am3.set_process_escape(True)
    am3.set_process_expansion(True)
    am3.set_process_adiabatic_cooling(True)
    am3.set_process_quantum_syn(True)
    am3.set_process_quantum_syn_cooling(True)
    am3.set_process_pp(True)
    am3.set_process_annihilation(True)
    am3.init_kernels()
    radius = 3e14
    volume = 4 / 3.0 * np.pi * radius ** 3
    am3.set_mag_field(300.0)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_expansion_timescale(3 * radius / C_CGS)
    am3.set_pp_target_proton_density(1e8)
    am3.set_solver_time_step(3e-3 * radius / C_CGS)
    am3.set_powerlaw_injection_parameters_electrons(
        volume, 3e41, 1e2 * ME_C2_EV, 1e3 * ME_C2_EV, 1e5 * ME_C2_EV, 1.8, 2.8, 1.0)
    am3.set_powerlaw_injection_parameters_protons(
        volume, 1e46, 1e1 * MP_C2_EV, 1e1 * MP_C2_EV, 1e9 * MP_C2_EV, 2.0, 2.0, 1.0)
    done = 0
    while done < n_steps:
        k = min(every, n_steps - done)
        for _ in range(k):
            am3.evolve_step()
        done += k
        yield done


def ssc_external(am3, n_steps=400, every=100):
    """Leptonic SSC + external black-body photon field (as in the TDE example):
    no hadronic processes, merged e+/-, no SED parsing, max-energy estimate,
    user escape/extra-cooling arrays, reduced energy grid."""
    am3.set_verbosity_level(False)
    am3.update_energy_grid(1e-6, 1e2, 1e17)
    am3.set_process_hadronic(False)
    am3.set_process_parse_sed(False)
    am3.set_process_merge_positrons_into_electrons(True)
    am3.set_process_annihilation(True)
    am3.set_estimate_max_energies(True)
    am3.set_lep_optimizer_on()
    am3.init_kernels()
    radius = 1e16
    volume = 4 / 3.0 * np.pi * radius ** 3
    am3.set_mag_field(0.1)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_solver_time_step(1e-2 * radius / C_CGS)
    am3.set_electron_accel_factor(1e4)
    am3.set_powerlaw_injection_parameters_electrons(
        volume, 1e43, 1e2 * ME_C2_EV, 1e2 * ME_C2_EV, 1e6 * ME_C2_EV, 2.2, 2.2, 1.0)
    eg = am3.get_egrid_photons()
    kT = 10.0  # eV
    bb = eg ** 3 / np.expm1(np.minimum(eg / kT, 700.0)) / 1e30
    am3.set_injection_rate_photons(bb * 1e-8)
    ne = am3.get_grid_size_lep()
    am3.set_t_pair_extra_cooling(np.full(ne, 1e9))
    am3.set_t_photon_escape(np.full(am3.get_grid_size_photons(), 2 * radius / C_CGS))
    done = 0
    while done < n_steps:
        k = min(every, n_steps - done)
        for _ in range(k):
            am3.evolve_step()
        done += k
        yield done


def exotic(am3, n_steps=200, every=50):
    """Rarely used code paths: full gamma-gamma pair kernel, pp QGSJET/Geant4
    models with pi0 into the cascade, RK4 source integration in the solver,
    no escape for photons, user-set densities / injection / escape /
    extra-cooling arrays for every species and a field change mid-run."""
    am3.set_verbosity_level(False)
    am3.update_energy_grid(1e-2, 1e3, 3e14)
    am3.set_process_parse_sed(True)
    am3.set_process_annihilation(True)
    am3.set_optimize_annihilation_pair_emission(False)
    am3.set_process_pp(True)
    am3.set_pp_model_charged_pions(1)
    am3.set_pp_model_neutral_pions(2)
    am3.set_pp_transition_energy_to_delta_approx(3e11)
    am3.set_process_pp_emission_pi0_into_cascade(True)
    am3.set_solver_simpleqintegration(True)
    am3.set_process_bethe_heitler_cooling(True)
    am3.set_had_optimizer_on()
    am3.set_process_bethe_heitler_cooling(True)
    am3.init_kernels()
    radius = 1e14
    volume = 4 / 3.0 * np.pi * radius ** 3
    am3.set_mag_field(30.0)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_escape_fraction_neutral_particles(0.5)
    am3.set_escape_fraction_charged_particles(0.1)
    am3.set_pp_target_proton_density(1e9)
    am3.set_solver_time_step(1e-2 * radius / C_CGS)
    am3.set_powerlaw_injection_parameters_electrons(
        volume, 1e41, 3 * ME_C2_EV, 3e2 * ME_C2_EV, 1e4 * ME_C2_EV, 1.5, 3.0, 2.0)
    am3.set_powerlaw_injection_parameters_protons(
        volume, 1e44, 2 * MP_C2_EV, 1e2 * MP_C2_EV, 1e5 * MP_C2_EV, 1.5, 2.5, 1.0)
    nh, nl, ng, nn = (am3.get_grid_size_had(), am3.get_grid_size_lep(),
                      am3.get_grid_size_photons(), am3.get_grid_size_neutrinos())
    eh = am3.get_egrid_had()
    bump = np.exp(-0.5 * ((np.log(eh) - np.log(1e12)) / 1.0) ** 2)
    am3.set_injection_rate_pion_plus(1e-8 * bump)
    am3.set_injection_rate_mu_minus_left(2e-8 * bump)
    am3.set_injection_rate_electron_neutrinos(np.full(nn, 1e-12))
    am3.set_current_densities_protons(1e-3 * bump)
    am3.set_current_densities_photons(np.full(ng, 1e-2))
    am3.set_t_proton_extra_cooling(np.full(nh, 3e5))
    am3.set_t_muon_escape(np.full(nh, 1e4))
    am3.set_t_photon_extra_cooling(np.full(ng, 1e7))
    done = 0
    while done < n_steps:
        k = min(every, n_steps - done)
        for _ in range(k):
            am3.evolve_step()
        done += k
        if done == every:
            am3.set_mag_field(3.0)
        yield done


def dump_state(am3):
    """Every public array/scalar getter that does not modify state."""
    out = {}
    for name in sorted(dir(am3)):
        if not name.startswith('get_'):
            continue
        if name in ('get_profiled_times', 'get_profiled_time_labels', 'get_profile_timing'):
            continue
        try:
            v = getattr(am3, name)()
        except TypeError:
            continue
        if isinstance(v, tuple):
            v = np.array(v, dtype=float)
        out[name] = np.asarray(v, dtype=float)
    return out


SCENARIOS = {
    'blazar_simple': blazar_simple,
    'leptohadronic_all': leptohadronic_all,
    'ssc_external': ssc_external,
    'exotic': exotic,
}
