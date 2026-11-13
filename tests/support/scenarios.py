"""Simulation scenarios shared by the tests, checks, figures and benchmark.

Each drives an `AM3()` object through the public Python API only, so the same
code runs with am3_mojo and with the C++ AM3.
"""

import numpy as np

from support.codata import C_CGS, ME_C2_EV, ME_G, MP_C2_EV, MP_G, SIGMA_T


def sphere_volume(radius):
    return 4 / 3.0 * np.pi * radius**3


def evolve(am3, n_steps, every, after=None):
    if every < 1:
        raise ValueError(f"every = {every}: must be 1 or more")
    done = 0
    while done < n_steps:
        k = min(every, n_steps - done)
        for _ in range(k):
            am3.evolve_step()
        done += k
        if after is not None:
            after(done)
        yield done


def set_processes(am3, **switches):
    """set_process_<name>(on) for every name=on, in this order."""
    for name, on in switches.items():
        getattr(am3, "set_process_" + name)(on)


def blazar_switches(am3):
    """The switches of the blazar model of AM3's example notebook."""
    am3.set_verbosity_level(False)
    set_processes(
        am3,
        parse_sed=True,
        escape=True,
        adiabatic_cooling=False,
        expansion=False,
        electron_syn=True,
        ssa=True,
        proton_syn=True,
        pion_syn=False,
        muon_syn=False,
        electron_compton=True,
        proton_compton=True,
        pion_compton=False,
        muon_compton=False,
        pion_decay=True,
        muon_decay=True,
        annihilation=False,
        photopion=True,
        pp=False,
        bethe_heitler=True,
    )


def blazar_simple(am3, n_steps=800, every=100):
    """The proton-synchrotron blazar model of AM3's example notebook
    (docs/examples/blazar_simple_example.ipynb)."""
    blazar_switches(am3)
    am3.init_kernels()
    radius = 1e15
    volume = sphere_volume(radius)
    injpower_e = (
        10.0 ** (-7.47) * 4 * np.pi * radius * (ME_G * C_CGS**3 / SIGMA_T)
    )
    injpower_p = (
        10 ** (-4.93) * 4 * np.pi * radius * (MP_G * C_CGS**3 / SIGMA_T)
    )
    am3.set_mag_field(10.0)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_solver_time_step(1e-2 * radius / C_CGS)
    am3.set_powerlaw_injection_parameters_electrons(
        volume,
        injpower_e,
        2.0 * ME_C2_EV,
        2.0 * ME_C2_EV,
        1e3 * ME_C2_EV,
        1.9,
        1.9,
        1.0,
    )
    am3.set_powerlaw_injection_parameters_protons(
        volume,
        injpower_p,
        2.0 * MP_C2_EV,
        2.0 * MP_C2_EV,
        1e8 * MP_C2_EV,
        1.9,
        1.9,
        1.0,
    )
    yield from evolve(am3, n_steps, every)


def leptohadronic_switches(am3):
    """Every process on. The e+/- kernel of the quantum synchrotron depends on
    B, and the next step rebuilds it."""
    am3.set_verbosity_level(False)
    set_processes(
        am3,
        parse_sed=True,
        escape=True,
        expansion=True,
        adiabatic_cooling=True,
        quantum_syn=True,
        quantum_syn_cooling=True,
        pp=True,
        annihilation=True,
    )


def leptohadronic_params(am3, bfield=300.0, time_step_frac=3e-3):
    """A dense source in a strong field, in which every hadronic
    process matters."""
    radius = 3e14
    volume = sphere_volume(radius)
    am3.set_mag_field(bfield)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_expansion_timescale(3 * radius / C_CGS)
    am3.set_pp_target_proton_density(1e8)
    am3.set_solver_time_step(time_step_frac * radius / C_CGS)
    am3.set_powerlaw_injection_parameters_electrons(
        volume,
        3e41,
        1e2 * ME_C2_EV,
        1e3 * ME_C2_EV,
        1e5 * ME_C2_EV,
        1.8,
        2.8,
        1.0,
    )
    am3.set_powerlaw_injection_parameters_protons(
        volume,
        1e46,
        1e1 * MP_C2_EV,
        1e1 * MP_C2_EV,
        1e9 * MP_C2_EV,
        2.0,
        2.0,
        1.0,
    )
    return time_step_frac * radius / C_CGS


def leptohadronic_all(am3, n_steps=300, every=50):
    """Every process on. The time step (3e-3 R/c) is too long for converged
    spectra. A tenth of that time step gives 1.8 times the e+- at 0.6 MeV and
    17 % fewer photons at 1e16 eV. The comparison with the C++ AM3 only checks
    that both codes solve the same equations."""
    leptohadronic_switches(am3)
    am3.init_kernels()
    leptohadronic_params(am3)
    yield from evolve(am3, n_steps, every)


def ssc_external(am3, n_steps=400, every=100):
    """Leptonic SSC + external black-body photon field (as in the TDE example):
    no hadronic processes, merged e+/-, no SED parsing, max-energy estimate,
    user escape/extra-cooling arrays, reduced energy grid."""
    am3.set_verbosity_level(False)
    am3.update_energy_grid(1e-6, 1e2, 1e17)
    set_processes(
        am3,
        hadronic=False,
        parse_sed=False,
        merge_positrons_into_electrons=True,
        annihilation=True,
    )
    am3.set_estimate_max_energies(True)
    am3.set_lep_optimizer_on()
    am3.init_kernels()
    radius = 1e16
    volume = sphere_volume(radius)
    am3.set_mag_field(0.1)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_solver_time_step(1e-2 * radius / C_CGS)
    am3.set_electron_accel_factor(1e4)
    am3.set_powerlaw_injection_parameters_electrons(
        volume,
        1e43,
        1e2 * ME_C2_EV,
        1e2 * ME_C2_EV,
        1e6 * ME_C2_EV,
        2.2,
        2.2,
        1.0,
    )
    eg = am3.get_egrid_photons()
    kT = 10.0  # eV
    bb = eg**3 / np.expm1(np.minimum(eg / kT, 700.0)) / 1e30
    am3.set_injection_rate_photons(bb * 1e-8)
    am3.set_t_pair_extra_cooling(np.full(am3.get_grid_size_lep(), 1e9))
    am3.set_t_photon_escape(
        np.full(am3.get_grid_size_photons(), 2 * radius / C_CGS)
    )
    yield from evolve(am3, n_steps, every)


def exotic(am3, n_steps=200, every=50):
    """Rarely used code paths: full gamma-gamma pair kernel, pp QGSJET/Geant4
    models with pi0 into the cascade, RK4 source integration in the solver,
    user-set arrays and a field change mid-run."""
    am3.set_verbosity_level(False)
    am3.update_energy_grid(1e-2, 1e3, 3e14)
    set_processes(am3, parse_sed=True, annihilation=True)
    am3.set_optimize_annihilation_pair_emission(False)
    am3.set_process_pp(True)
    am3.set_pp_model_charged_pions(1)
    am3.set_pp_model_neutral_pions(2)
    am3.set_pp_transition_energy_to_delta_approx(3e11)
    am3.set_process_pp_emission_pi0_into_cascade(True)
    am3.set_solver_simpleqintegration(True)
    am3.set_had_optimizer_on()
    # (set_had_optimizer_on switches it off)
    am3.set_process_bethe_heitler_cooling(True)
    am3.init_kernels()
    radius = 1e14
    volume = sphere_volume(radius)
    am3.set_mag_field(30.0)
    am3.set_escape_timescale(radius / C_CGS)
    am3.set_escape_fraction_neutral_particles(0.5)
    am3.set_escape_fraction_charged_particles(0.1)
    am3.set_pp_target_proton_density(1e9)
    am3.set_solver_time_step(1e-2 * radius / C_CGS)
    am3.set_powerlaw_injection_parameters_electrons(
        volume,
        1e41,
        3 * ME_C2_EV,
        3e2 * ME_C2_EV,
        1e4 * ME_C2_EV,
        1.5,
        3.0,
        2.0,
    )
    am3.set_powerlaw_injection_parameters_protons(
        volume,
        1e44,
        2 * MP_C2_EV,
        1e2 * MP_C2_EV,
        1e5 * MP_C2_EV,
        1.5,
        2.5,
        1.0,
    )
    nh, ng, nn = (
        am3.get_grid_size_had(),
        am3.get_grid_size_photons(),
        am3.get_grid_size_neutrinos(),
    )
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

    def field_change(done):
        if done == every:
            am3.set_mag_field(3.0)

    yield from evolve(am3, n_steps, every, after=field_change)


def dump_state(am3):
    out = {}
    for name in sorted(dir(am3)):
        if not name.startswith("get_") or name.startswith("get_profile"):
            continue
        try:
            v = getattr(am3, name)()
        except TypeError:
            continue
        out[name] = np.asarray(v, dtype=float)
    return out


SCENARIOS = {
    f.__name__: f
    for f in (blazar_simple, leptohadronic_all, ssc_external, exotic)
}
