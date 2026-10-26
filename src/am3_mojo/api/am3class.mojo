"""Port of the AM3 user-interface class (AM3.cc)."""

from am3_mojo.api.convert import (
    scaled,
    summed,
    timescales,
    constant_timescales,
    emission_timescales,
    set_rates_from_timescales,
    assign_divided,
)
from am3_mojo.model.particles import *
from am3_mojo.physics.expansion import Expansion
from am3_mojo.simulation.coeffs import *
from am3_mojo.simulation.manager import SimulationManager
from am3_mojo.solver.ids import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *


struct AM3(Defaultable, Movable, Writable):
    var sim: SimulationManager

    def __init__(out self):
        self.sim = SimulationManager()

    def _pair_solver(self) -> Int:
        """Solver of the e+/- array the pair getters report on."""
        return (
            SA_ep if self.sim.process_merge_positrons_into_electrons else SA_e
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write("<am3_mojo.AM3>")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("<am3_mojo.AM3>")

    # ------------------------------------------------------------------
    # general functions
    # ------------------------------------------------------------------
    def init_kernels(mut self) raises:
        self.sim.init_kernels()

    def clear_particle_densities(mut self):
        self.sim.clear_particle_densities()

    def clear_am3(mut self):
        self.sim.clear_am3()

    def evolve_step(mut self) raises:
        self.sim.evolve_step()

    def get_verbosity_level(mut self) -> Bool:
        return self.sim.verbose

    def set_verbosity_level(mut self, v: Bool):
        self.sim.update_verbosity(v)

    def print_process_switches(mut self):
        self.sim.print_process_switches()

    def set_lep_optimizer_on(mut self):
        print("turned on lep_optimzer switches ")
        self.set_process_compton_photon_energy_loss(False)
        self.set_optimize_compton_emission_emin(1e-3)
        self.set_optimize_compton_emission_grid(True)
        self.set_optimize_compton_target_emax(1e6)
        self.set_optimize_compton_target_grid(True)
        self.set_optimize_annihilation_pair_emission(True)

    def set_had_optimizer_on(mut self):
        print("turned on had_optimzer switches")
        self.set_optimize_bethe_heitler_outgoing_pairs_grid(True)
        self.set_optimize_bethe_heitler_incoming_protons_min(1e12)
        self.set_optimize_bethe_heitler_target_photon_max(1e6)
        self.set_process_bethe_heitler_cooling(False)
        self.set_optimize_photopion_target_photon_grid(True)
        self.set_optimize_photopion_target_photon_max(1e6)
        self.set_process_photopion_photon_loss(False)

    # ------------------------------------------------------------------
    # optimisation parameters (stored as grid indices)
    # ------------------------------------------------------------------
    def get_optimize_compton_emission_emin(mut self) -> Float64:
        return self.sim.dat.egrid_pho[self.sim.optimize_compton_emission_emin]

    def get_optimize_compton_target_emax(mut self) -> Float64:
        return self.sim.dat.egrid_pho[self.sim.optimize_compton_target_emax]

    def get_optimize_bethe_heitler_incoming_protons_min(mut self) -> Float64:
        return self.sim.dat.egrid_had[
            self.sim.optimize_bethe_heitler_incoming_protons_min
        ]

    def get_optimize_bethe_heitler_target_photon_max(mut self) -> Float64:
        return self.sim.dat.egrid_pho[
            self.sim.optimize_bethe_heitler_target_photon_max
        ]

    def get_optimize_photopion_target_photon_max(mut self) -> Float64:
        return self.sim.dat.egrid_pho[
            self.sim.optimize_photopion_target_photon_max
        ]

    def get_pp_transition_energy_to_delta_approx(mut self) -> Float64:
        return self.sim.pp_transition_energy_to_delta_approx * 1e9

    @staticmethod
    def _even_index(index: Int, n: Int) -> Int:
        """`index` rounded up to an even number, as in AM3, but kept on the
        grid of n bins (AM3 can return n, one past the last bin)."""
        var k = index + 1 if index % 2 != 0 else index
        while k > n - 1:
            k -= 2
        return max(k, 0)

    def _photon_index_even(mut self, value: Float64) -> Int:
        var E_max_eV = (
            self.sim.dat.egrid_pho[self.sim.dat.nbins_pho - 1] * MEC2_EV
        )
        var Egamma_min_eV = self.sim.dat.egrid_pho[0] * MEC2_EV
        var v = max(min(value, E_max_eV), Egamma_min_eV)
        return Self._even_index(
            self.sim.dat.eV_to_photon_grid_index(v), self.sim.dat.nbins_pho
        )

    def set_optimize_compton_emission_emin(mut self, new_value: Float64):
        self.sim.optimize_compton_emission_emin = self._photon_index_even(
            new_value
        )

    def set_optimize_compton_target_emax(mut self, new_value: Float64):
        self.sim.optimize_compton_target_emax = self._photon_index_even(
            new_value
        )

    def set_optimize_compton_emission_grid(mut self, new_value: Bool):
        self.sim.optimize_compton_emission_grid = (
            True  # bool = int(new_value) + 1
        )

    def set_optimize_compton_target_grid(mut self, new_value: Bool):
        self.sim.optimize_compton_target_grid = True

    def set_optimize_bethe_heitler_outgoing_pairs_grid(
        mut self, new_value: Bool
    ):
        self.sim.optimize_bethe_heitler_outgoing_pairs_grid = True

    def set_optimize_bethe_heitler_incoming_protons_min(
        mut self, new_value: Float64
    ):
        var E_max_eV = self.sim.dat.egrid_had[self.sim.dat.nbins_had - 1] * 1e9
        var E_min_eV = 1e9
        var v = max(min(new_value, E_max_eV), E_min_eV)
        self.sim.optimize_bethe_heitler_incoming_protons_min = Self._even_index(
            self.sim.dat.eV_to_had_grid_index(v), self.sim.dat.nbins_had
        )

    def set_optimize_bethe_heitler_target_photon_max(
        mut self, new_value: Float64
    ):
        self.sim.optimize_bethe_heitler_target_photon_max = (
            self._photon_index_even(new_value)
        )

    def set_optimize_photopion_target_photon_grid(mut self, new_value: Bool):
        self.sim.optimize_photopion_target_photon_grid = True

    def set_optimize_photopion_target_photon_max(mut self, new_value: Float64):
        self.sim.optimize_photopion_target_photon_max = self._photon_index_even(
            new_value
        )

    def set_pp_transition_energy_to_delta_approx(mut self, new_value: Float64):
        self.sim.pp_transition_energy_to_delta_approx = 1e-9 * new_value

    # ------------------------------------------------------------------
    # physical parameters needing extra work
    # ------------------------------------------------------------------
    def get_powerlaw_injection_parameters_electrons(
        mut self,
    ) -> Array[Float64, 8]:
        var rp = self.sim.rp.copy()
        var _r: Array[Float64, 8] = [
            rp.powerlaw_injection_electrons_volume,
            rp.powerlaw_injection_electrons_luminosity,
            rp.powerlaw_injection_electrons_emin,
            rp.powerlaw_injection_electrons_ebreak,
            rp.powerlaw_injection_electrons_emax,
            rp.powerlaw_injection_electrons_index,
            rp.powerlaw_injection_electrons_index_above_break,
            rp.powerlaw_injection_electrons_cutoff_steepness,
        ]
        return _r^

    def get_powerlaw_injection_parameters_protons(
        mut self,
    ) -> Array[Float64, 8]:
        var rp = self.sim.rp.copy()
        var _r: Array[Float64, 8] = [
            rp.powerlaw_injection_protons_volume,
            rp.powerlaw_injection_protons_luminosity,
            rp.powerlaw_injection_protons_emin,
            rp.powerlaw_injection_protons_ebreak,
            rp.powerlaw_injection_protons_emax,
            rp.powerlaw_injection_protons_index,
            rp.powerlaw_injection_protons_index_above_break,
            rp.powerlaw_injection_protons_cutoff_steepness,
        ]
        return _r^

    def set_powerlaw_injection_parameters_electrons(
        mut self,
        volume: Float64,
        luminosity: Float64,
        emin: Float64,
        ebreak: Float64,
        emax: Float64,
        index: Float64,
        index_above_break: Float64,
        cutoff_steepness: Float64,
    ):
        self.sim.rp.powerlaw_injection_electrons_volume = volume
        self.sim.rp.powerlaw_injection_electrons_luminosity = luminosity
        self.sim.rp.powerlaw_injection_electrons_emin = emin
        self.sim.rp.powerlaw_injection_electrons_ebreak = ebreak
        self.sim.rp.powerlaw_injection_electrons_emax = emax
        self.sim.rp.powerlaw_injection_electrons_index = index
        self.sim.rp.powerlaw_injection_electrons_index_above_break = (
            index_above_break
        )
        self.sim.rp.powerlaw_injection_electrons_cutoff_steepness = (
            cutoff_steepness
        )
        self.sim.oInj.Calc_IN_Electron_powerlaw(self.sim.rp, self.sim.dat)

    def set_powerlaw_injection_parameters_protons(
        mut self,
        volume: Float64,
        luminosity: Float64,
        emin: Float64,
        ebreak: Float64,
        emax: Float64,
        index: Float64,
        index_above_break: Float64,
        cutoff_steepness: Float64,
    ):
        self.sim.rp.powerlaw_injection_protons_volume = volume
        self.sim.rp.powerlaw_injection_protons_luminosity = luminosity
        self.sim.rp.powerlaw_injection_protons_emin = emin
        self.sim.rp.powerlaw_injection_protons_ebreak = ebreak
        self.sim.rp.powerlaw_injection_protons_emax = emax
        self.sim.rp.powerlaw_injection_protons_index = index
        self.sim.rp.powerlaw_injection_protons_index_above_break = (
            index_above_break
        )
        self.sim.rp.powerlaw_injection_protons_cutoff_steepness = (
            cutoff_steepness
        )
        self.sim.oInj.Calc_IN_Proton_powerlaw(self.sim.rp, self.sim.dat)

    def set_mag_field(mut self, b: Float64):
        self.sim.Update_Mag_Field(b)

    def set_solver_time_step(mut self, t: Float64):
        self.sim.rp.dt = t

    def update_energy_grid(
        mut self, Egamma_min_eV: Float64, Enu_min_eV: Float64, E_max_eV: Float64
    ) raises:
        # the grids start at the electron rest energy (leptons) and 1 GeV
        # (hadrons): the photon and neutrino grids must reach below them, and
        # the hadron grid must span at least a decade
        if not (
            Egamma_min_eV > 0.0
            and Egamma_min_eV < MEC2_EV
            and Enu_min_eV > 0.0
            and Enu_min_eV < 1e9
            and E_max_eV >= 1e10
            and E_max_eV < 1e300
        ):
            raise Error(
                String(
                    "update_energy_grid: need 0 < Egamma_min < 511 keV,",
                    " 0 < Enu_min < 1 GeV and 10 GeV <= E_max < 1e300 eV; got ",
                    Egamma_min_eV,
                    ", ",
                    Enu_min_eV,
                    ", ",
                    E_max_eV,
                )
            )
        var d = self.sim.dat.egrid_pho.copy()
        var h = self.sim.dat.egrid_had.copy()
        var opt_E_ic_ph_in_max = max(
            min(d[self.sim.optimize_compton_target_emax] * MEC2_EV, E_max_eV),
            Egamma_min_eV,
        )
        var opt_E_ic_ph_out_min = max(
            min(d[self.sim.optimize_compton_emission_emin] * MEC2_EV, E_max_eV),
            Egamma_min_eV,
        )
        var opt_E_bh_pr_in_min = max(
            min(
                h[self.sim.optimize_bethe_heitler_incoming_protons_min] * 1e9,
                E_max_eV,
            ),
            2e9,
        )
        var opt_E_bh_ph_in_max = max(
            min(
                d[self.sim.optimize_bethe_heitler_target_photon_max] * MEC2_EV,
                E_max_eV,
            ),
            Egamma_min_eV,
        )
        var opt_E_pg_ph_in_max = max(
            min(
                d[self.sim.optimize_photopion_target_photon_max] * MEC2_EV,
                E_max_eV,
            ),
            Egamma_min_eV,
        )
        self.sim.dat.update_energy_grid_parameters(
            Egamma_min_eV, Enu_min_eV, E_max_eV
        )
        self.sim.dat.update_energy_grids()
        # every array on the new grids; the kernels follow at init_kernels,
        # which evolve_step requires again
        self.sim.clear_am3()
        self.sim.kernels_ready = False
        self.set_optimize_compton_target_emax(opt_E_ic_ph_in_max)
        self.set_optimize_compton_emission_emin(opt_E_ic_ph_out_min)
        self.set_optimize_bethe_heitler_incoming_protons_min(opt_E_bh_pr_in_min)
        self.set_optimize_bethe_heitler_target_photon_max(opt_E_bh_ph_in_max)
        self.set_optimize_photopion_target_photon_max(opt_E_pg_ph_in_max)

    # ------------------------------------------------------------------
    # photon emission timescales (bool products as in the C++ code)
    # ------------------------------------------------------------------
    def get_t_photon_emission_from_syn(mut self) -> Vec:
        var s = (
            self.sim.process_electron_syn
            and self.sim.process_electron_syn_emission
        )
        var p = (
            self.sim.process_proton_syn and self.sim.process_proton_syn_emission
        )
        var pi_ = (
            self.sim.process_pion_syn and self.sim.process_pion_syn_emission
        )
        var mu = (
            self.sim.process_muon_syn and self.sim.process_muon_syn_emission
        )
        var res = zeros(self.sim.dat.nbins_pho)
        for i in range(self.sim.dat.nbins_pho):
            res[i] = (
                T_SSC
                * (self.sim.dat.n[P_photons][i])
                / (
                    fbool(s) * self.sim.c[C_qgma_sy][i]
                    + fbool(p) * self.sim.c[C_qgma_psy][i]
                    + fbool(pi_) * self.sim.c[C_qgma_pisy][i]
                    + fbool(mu) * self.sim.c[C_qgma_musy][i]
                    + 1e-200
                )
            )
        return res^

    def get_t_photon_emission_from_ic(mut self) -> Vec:
        var s = (
            self.sim.process_electron_compton
            and self.sim.process_electron_compton_emission
        )
        var p = (
            self.sim.process_proton_compton
            and self.sim.process_proton_compton_emission
        )
        var pi_ = (
            self.sim.process_pion_compton
            and self.sim.process_pion_compton_emission
        )
        var mu = (
            self.sim.process_muon_compton
            and self.sim.process_muon_compton_emission
        )
        var res = zeros(self.sim.dat.nbins_pho)
        for i in range(self.sim.dat.nbins_pho):
            res[i] = (
                T_SSC
                * (self.sim.dat.n[P_photons][i])
                / (
                    fbool(s) * self.sim.c[C_qgma_ic][i]
                    + fbool(p) * self.sim.c[C_qgma_pic][i]
                    + fbool(pi_) * self.sim.c[C_qgma_piic][i]
                    + fbool(mu) * self.sim.c[C_qgma_muic][i]
                    + 1e-200
                )
            )
        return res^

    def get_t_photon_emission_from_pi0(mut self) -> Vec:
        var res = zeros(self.sim.dat.nbins_pho)
        for i in range(self.sim.dat.nbins_pho):
            res[i] = (
                T_SSC
                * (self.sim.dat.n[P_photons][i])
                / (
                    fbool(self.sim.process_pion_decay)
                    * (
                        self.sim.c[C_qgma_pi0_pg][i]
                        + self.sim.c[C_qgma_pi0_pp][i]
                    )
                    + 1e-200
                )
            )
        return res^

    def estimate_max_electron_energy(mut self):
        self.sim.estimate_maximum_lepton_energy()

    def estimate_max_proton_energy(mut self):
        self.sim.estimate_maximum_proton_energy()

    # ------------------------------------------------------------------
    # profiling
    # ------------------------------------------------------------------
    def get_profile_timing(mut self) -> Bool:
        return self.sim.profile_timing

    def set_profile_timing(mut self, new_value: Bool):
        self.sim.profile_timing = new_value

    def get_profiled_times(mut self) -> Vec:
        return self.sim.timing.copy()

    def get_profiled_time_labels(mut self) -> List[String]:
        var labels = List[String]()
        for s in [
            "photons -> had. grid",
            "calc. coeff. proton/neutron",
            "pass coeff. proton",
            "evolve proton",
            "pass coeff. neutron",
            "evolve neutron",
            "calc. coeff. pion",
            "pass coeff. pion",
            "evolve pion",
            "calc. coeff. muon",
            "pass coeff. muon",
            "evolve muon",
            "calc. coeff. neutrino",
            "pass coeff. neutrino",
            "evolve neutrino",
            "calc. coeff. pairs",
            "pass coeff. pairs",
            "evolve pairs",
            "calc. coeff. photon",
            "pass coeff. photon",
            "evolve photon",
            "photon coeff: injection",
            "photon coeff: pair syn. em.",
            "photon coeff: pair syn. abs.",
            "photon coeff: pair compton em.",
            "photon coeff: pair compton abs.",
            "photon coeff: pair annihil.",
            "photon coeff: photo-pion loss",
            "photon coeff: proton syn. em.",
            "photon coeff: proton compton em.",
            "photon coeff: pion syn. em.",
            "photon coeff: pion compton em.",
            "photon coeff: muon syn. em.",
            "photon coeff: muon compton em.",
            "photon coeff: pi0 decay",
            "photon coeff: escape",
            # stages of the processes am3_mojo adds to AM3
            "e+e- annihilation",
            "triplet pair production",
            "neutron decay",
            "calc. coeff. kaon",
            "evolve kaon",
            "photon coeff: kaon syn. em.",
            "photon coeff: kaon compton em.",
            "calc. coeff. charm",
            "evolve charm",
            "photon coeff: charm syn. em.",
            "photon coeff: charm compton em.",
        ]:
            labels.append(String(s))
        return labels^

    # ------------------------------------------------------------------
    # plain getters and setters of the AM3 interface
    # ------------------------------------------------------------------
    def get_estimate_max_energies(mut self) -> Bool:
        return self.sim.estimate_max_energies

    def get_process_parse_sed(mut self) -> Bool:
        return self.sim.process_parse_sed

    def get_process_hadronic(mut self) -> Bool:
        return self.sim.process_hadronic

    def get_process_merge_positrons_into_electrons(mut self) -> Bool:
        return self.sim.process_merge_positrons_into_electrons

    def get_process_escape(mut self) -> Bool:
        return self.sim.process_escape

    def get_process_adiabatic_cooling(mut self) -> Bool:
        return self.sim.process_adiabatic_cooling

    def get_process_expansion(mut self) -> Bool:
        return self.sim.process_expansion

    def get_process_electron_syn(mut self) -> Bool:
        return self.sim.process_electron_syn

    def get_process_electron_syn_emission(mut self) -> Bool:
        return self.sim.process_electron_syn_emission

    def get_process_electron_syn_cooling(mut self) -> Bool:
        return self.sim.process_electron_syn_cooling

    def get_process_quantum_syn(mut self) -> Bool:
        return self.sim.process_quantum_syn

    def get_process_quantum_syn_cooling(mut self) -> Bool:
        return self.sim.process_quantum_syn_cooling

    def get_process_ssa(mut self) -> Bool:
        return self.sim.process_ssa

    def get_process_proton_syn(mut self) -> Bool:
        return self.sim.process_proton_syn

    def get_process_proton_syn_emission(mut self) -> Bool:
        return self.sim.process_proton_syn_emission

    def get_process_proton_syn_cooling(mut self) -> Bool:
        return self.sim.process_proton_syn_cooling

    def get_process_pion_syn(mut self) -> Bool:
        return self.sim.process_pion_syn

    def get_process_pion_syn_emission(mut self) -> Bool:
        return self.sim.process_pion_syn_emission

    def get_process_pion_syn_cooling(mut self) -> Bool:
        return self.sim.process_pion_syn_cooling

    def get_process_muon_syn(mut self) -> Bool:
        return self.sim.process_muon_syn

    def get_process_muon_syn_emission(mut self) -> Bool:
        return self.sim.process_muon_syn_emission

    def get_process_muon_syn_cooling(mut self) -> Bool:
        return self.sim.process_muon_syn_cooling

    def get_process_electron_compton(mut self) -> Bool:
        return self.sim.process_electron_compton

    def get_process_electron_compton_emission(mut self) -> Bool:
        return self.sim.process_electron_compton_emission

    def get_process_electron_compton_cooling(mut self) -> Bool:
        return self.sim.process_electron_compton_cooling

    def get_process_compton_photon_energy_loss(mut self) -> Bool:
        return self.sim.process_compton_photon_energy_loss

    def get_process_proton_compton(mut self) -> Bool:
        return self.sim.process_proton_compton

    def get_process_proton_compton_emission(mut self) -> Bool:
        return self.sim.process_proton_compton_emission

    def get_process_proton_compton_cooling(mut self) -> Bool:
        return self.sim.process_proton_compton_cooling

    def get_process_pion_compton(mut self) -> Bool:
        return self.sim.process_pion_compton

    def get_process_pion_compton_emission(mut self) -> Bool:
        return self.sim.process_pion_compton_emission

    def get_process_pion_compton_cooling(mut self) -> Bool:
        return self.sim.process_pion_compton_cooling

    def get_process_muon_compton(mut self) -> Bool:
        return self.sim.process_muon_compton

    def get_process_muon_compton_emission(mut self) -> Bool:
        return self.sim.process_muon_compton_emission

    def get_process_muon_compton_cooling(mut self) -> Bool:
        return self.sim.process_muon_compton_cooling

    def get_process_pion_decay(mut self) -> Bool:
        return self.sim.process_pion_decay

    def get_process_muon_decay(mut self) -> Bool:
        return self.sim.process_muon_decay

    def get_process_annihilation(mut self) -> Bool:
        return self.sim.process_annihilation

    def get_process_annihilation_cooling(mut self) -> Bool:
        return self.sim.process_annihilation_cooling

    def get_process_annihilation_pair_emission(mut self) -> Bool:
        return self.sim.process_annihilation_emission

    def get_process_bethe_heitler(mut self) -> Bool:
        return self.sim.process_bethe_heitler

    def get_process_bethe_heitler_emission(mut self) -> Bool:
        return self.sim.process_bethe_heitler_emission

    def get_process_bethe_heitler_cooling(mut self) -> Bool:
        return self.sim.process_bethe_heitler_cooling

    def get_process_photopion(mut self) -> Bool:
        return self.sim.process_photopion

    def get_process_photopion_emission(mut self) -> Bool:
        return self.sim.process_photopion_emission

    def get_process_photopion_cooling(mut self) -> Bool:
        return self.sim.process_photopion_cooling

    def get_process_photopion_photon_loss(mut self) -> Bool:
        return self.sim.process_photopion_photon_loss

    def get_process_pp(mut self) -> Bool:
        return self.sim.process_pp

    def get_process_pp_emission(mut self) -> Bool:
        return self.sim.process_pp_emission

    def get_process_pp_emission_pi0_into_cascade(mut self) -> Bool:
        return self.sim.process_pp_emission_pi0_into_cascade

    def get_process_pp_cooling(mut self) -> Bool:
        return self.sim.process_pp_cooling

    def get_optimize_compton_emission_grid(mut self) -> Bool:
        return self.sim.optimize_compton_emission_grid

    def get_optimize_compton_target_grid(mut self) -> Bool:
        return self.sim.optimize_compton_target_grid

    def get_optimize_annihilation_pair_emission(mut self) -> Bool:
        return self.sim.optimize_annihilation_pair_emission

    def get_optimize_bethe_heitler_outgoing_pairs_grid(mut self) -> Bool:
        return self.sim.optimize_bethe_heitler_outgoing_pairs_grid

    def get_optimize_photopion_target_photon_grid(mut self) -> Bool:
        return self.sim.optimize_photopion_target_photon_grid

    def get_pp_model_charged_pions(mut self) -> Int:
        return self.sim.pp_model_charged_pions

    def get_pp_model_neutral_pions(mut self) -> Int:
        return self.sim.pp_model_neutral_pions

    def get_solver_simpleqintegration(mut self) -> Bool:
        return self.sim.solver_simpleqintegration

    def get_solver_threshold_cool_dom(mut self) -> Float64:
        return self.sim.solver_threshold_cool_dom

    def get_solver_threshold_esc_dom(mut self) -> Float64:
        return self.sim.solver_threshold_esc_dom

    def get_solver_threshold_matrix(mut self) -> Float64:
        return self.sim.solver_threshold_matrix

    def set_estimate_max_energies(mut self, new_value: Bool):
        self.sim.estimate_max_energies = new_value

    def set_process_parse_sed(mut self, new_value: Bool):
        self.sim.process_parse_sed = new_value

    def set_process_hadronic(mut self, new_value: Bool):
        self.sim.process_hadronic = new_value

    def set_process_merge_positrons_into_electrons(mut self, new_value: Bool):
        self.sim.process_merge_positrons_into_electrons = new_value

    def set_process_escape(mut self, new_value: Bool):
        self.sim.process_escape = new_value

    def set_process_adiabatic_cooling(mut self, new_value: Bool):
        self.sim.process_adiabatic_cooling = new_value

    def set_process_expansion(mut self, new_value: Bool):
        self.sim.process_expansion = new_value

    def set_process_electron_syn(mut self, new_value: Bool):
        self.sim.process_electron_syn = new_value

    def set_process_electron_syn_emission(mut self, new_value: Bool):
        self.sim.process_electron_syn_emission = new_value

    def set_process_electron_syn_cooling(mut self, new_value: Bool):
        self.sim.process_electron_syn_cooling = new_value

    def set_process_quantum_syn(mut self, new_value: Bool):
        self.sim.process_quantum_syn = new_value

    def set_process_quantum_syn_cooling(mut self, new_value: Bool):
        self.sim.process_quantum_syn_cooling = new_value

    def set_process_ssa(mut self, new_value: Bool):
        self.sim.process_ssa = new_value

    def set_process_proton_syn(mut self, new_value: Bool):
        self.sim.process_proton_syn = new_value

    def set_process_proton_syn_emission(mut self, new_value: Bool):
        self.sim.process_proton_syn_emission = new_value

    def set_process_proton_syn_cooling(mut self, new_value: Bool):
        self.sim.process_proton_syn_cooling = new_value

    def set_process_pion_syn(mut self, new_value: Bool):
        self.sim.process_pion_syn = new_value

    def set_process_pion_syn_emission(mut self, new_value: Bool):
        self.sim.process_pion_syn_emission = new_value

    def set_process_pion_syn_cooling(mut self, new_value: Bool):
        self.sim.process_pion_syn_cooling = new_value

    def set_process_muon_syn(mut self, new_value: Bool):
        self.sim.process_muon_syn = new_value

    def set_process_muon_syn_emission(mut self, new_value: Bool):
        self.sim.process_muon_syn_emission = new_value

    def set_process_muon_syn_cooling(mut self, new_value: Bool):
        self.sim.process_muon_syn_cooling = new_value

    def set_process_electron_compton(mut self, new_value: Bool):
        self.sim.process_electron_compton = new_value

    def set_process_electron_compton_emission(mut self, new_value: Bool):
        self.sim.process_electron_compton_emission = new_value

    def set_process_electron_compton_cooling(mut self, new_value: Bool):
        self.sim.process_electron_compton_cooling = new_value

    def set_process_compton_photon_energy_loss(mut self, new_value: Bool):
        self.sim.process_compton_photon_energy_loss = new_value

    def set_process_proton_compton(mut self, new_value: Bool):
        self.sim.process_proton_compton = new_value

    def set_process_proton_compton_emission(mut self, new_value: Bool):
        self.sim.process_proton_compton_emission = new_value

    def set_process_proton_compton_cooling(mut self, new_value: Bool):
        self.sim.process_proton_compton_cooling = new_value

    def set_process_pion_compton(mut self, new_value: Bool):
        self.sim.process_pion_compton = new_value

    def set_process_pion_compton_emission(mut self, new_value: Bool):
        self.sim.process_pion_compton_emission = new_value

    def set_process_pion_compton_cooling(mut self, new_value: Bool):
        self.sim.process_pion_compton_cooling = new_value

    def set_process_muon_compton(mut self, new_value: Bool):
        self.sim.process_muon_compton = new_value

    def set_process_muon_compton_emission(mut self, new_value: Bool):
        self.sim.process_muon_compton_emission = new_value

    def set_process_muon_compton_cooling(mut self, new_value: Bool):
        self.sim.process_muon_compton_cooling = new_value

    def set_process_pion_decay(mut self, new_value: Bool):
        self.sim.process_pion_decay = new_value

    def set_process_muon_decay(mut self, new_value: Bool):
        self.sim.process_muon_decay = new_value

    def set_process_annihilation(mut self, new_value: Bool):
        self.sim.process_annihilation = new_value

    def set_process_annihilation_cooling(mut self, new_value: Bool):
        self.sim.process_annihilation_cooling = new_value

    def set_process_annihilation_pair_emission(mut self, new_value: Bool):
        self.sim.process_annihilation_emission = new_value

    def set_process_bethe_heitler(mut self, new_value: Bool):
        self.sim.process_bethe_heitler = new_value

    def set_process_bethe_heitler_emission(mut self, new_value: Bool):
        self.sim.process_bethe_heitler_emission = new_value

    def set_process_bethe_heitler_cooling(mut self, new_value: Bool):
        self.sim.process_bethe_heitler_cooling = new_value

    def set_process_photopion(mut self, new_value: Bool):
        self.sim.process_photopion = new_value

    def set_process_photopion_emission(mut self, new_value: Bool):
        self.sim.process_photopion_emission = new_value

    def set_process_photopion_cooling(mut self, new_value: Bool):
        self.sim.process_photopion_cooling = new_value

    def set_process_photopion_photon_loss(mut self, new_value: Bool):
        self.sim.process_photopion_photon_loss = new_value

    def set_process_pp(mut self, new_value: Bool):
        self.sim.process_pp = new_value

    def set_process_pp_emission(mut self, new_value: Bool):
        self.sim.process_pp_emission = new_value

    def set_process_pp_emission_pi0_into_cascade(mut self, new_value: Bool):
        self.sim.process_pp_emission_pi0_into_cascade = new_value

    def set_process_pp_cooling(mut self, new_value: Bool):
        self.sim.process_pp_cooling = new_value

    def set_optimize_annihilation_pair_emission(mut self, new_value: Bool):
        self.sim.optimize_annihilation_pair_emission = new_value

    def set_pp_model_charged_pions(mut self, new_value: Int):
        self.sim.pp_model_charged_pions = new_value

    def set_pp_model_neutral_pions(mut self, new_value: Int):
        self.sim.pp_model_neutral_pions = new_value

    def set_solver_simpleqintegration(mut self, new_value: Bool):
        self.sim.solver_simpleqintegration = new_value

    def set_solver_threshold_cool_dom(mut self, new_value: Float64):
        self.sim.solver_threshold_cool_dom = new_value

    def set_solver_threshold_esc_dom(mut self, new_value: Float64):
        self.sim.solver_threshold_esc_dom = new_value

    def set_solver_threshold_matrix(mut self, new_value: Float64):
        self.sim.solver_threshold_matrix = new_value

    def get_mag_field(mut self) -> Float64:
        return self.sim.rp.mag_field

    def get_escape_timescale(mut self) -> Float64:
        return self.sim.rp.escape_timescale

    def get_expansion_timescale(mut self) -> Float64:
        return self.sim.rp.expansion_timescale

    def get_pp_target_proton_density(mut self) -> Float64:
        return self.sim.rp.pp_target_proton_density

    def get_escape_fraction_pairs(mut self) -> Float64:
        return self.sim.rp.escape_fraction_pairs

    def get_escape_fraction_photons(mut self) -> Float64:
        return self.sim.rp.escape_fraction_photons

    def get_escape_fraction_protons(mut self) -> Float64:
        return self.sim.rp.escape_fraction_protons

    def get_escape_fraction_neutrons(mut self) -> Float64:
        return self.sim.rp.escape_fraction_neutrons

    def get_escape_fraction_pions(mut self) -> Float64:
        return self.sim.rp.escape_fraction_pions

    def get_escape_fraction_muons(mut self) -> Float64:
        return self.sim.rp.escape_fraction_muons

    def get_escape_fraction_neutrinos(mut self) -> Float64:
        return self.sim.rp.escape_fraction_neutrinos

    def get_solver_time_step(mut self) -> Float64:
        return self.sim.rp.dt

    def get_electron_accel_factor(mut self) -> Float64:
        return self.sim.rp.electron_accel_factor_for_emax_estimate

    def get_proton_accel_factor(mut self) -> Float64:
        return self.sim.rp.proton_accel_factor_for_emax_estimate

    def set_expansion_timescale(mut self, t: Float64):
        self.sim.rp.expansion_timescale = t

    def set_pp_target_proton_density(mut self, rho: Float64):
        self.sim.rp.pp_target_proton_density = rho

    def set_escape_timescale(mut self, t: Float64):
        self.sim.rp.escape_timescale = t
        self.sim.oEscape.Calc_Escape_Proton(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Neutron(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Pipm(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Mu(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Nu(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Lep(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Phot(self.sim.rp, self.sim.dat)

    def set_escape_fraction_pairs(mut self, frac: Float64):
        self.sim.rp.escape_fraction_pairs = frac
        self.sim.oEscape.Calc_Escape_Lep(self.sim.rp, self.sim.dat)

    def set_escape_fraction_photons(mut self, frac: Float64):
        self.sim.rp.escape_fraction_photons = frac
        self.sim.oEscape.Calc_Escape_Phot(self.sim.rp, self.sim.dat)

    def set_escape_fraction_protons(mut self, frac: Float64):
        self.sim.rp.escape_fraction_protons = frac
        self.sim.oEscape.Calc_Escape_Proton(self.sim.rp, self.sim.dat)

    def set_escape_fraction_neutrons(mut self, frac: Float64):
        self.sim.rp.escape_fraction_neutrons = frac
        self.sim.oEscape.Calc_Escape_Neutron(self.sim.rp, self.sim.dat)

    def set_escape_fraction_pions(mut self, frac: Float64):
        self.sim.rp.escape_fraction_pions = frac
        self.sim.oEscape.Calc_Escape_Pipm(self.sim.rp, self.sim.dat)

    def set_escape_fraction_muons(mut self, frac: Float64):
        self.sim.rp.escape_fraction_muons = frac
        self.sim.oEscape.Calc_Escape_Mu(self.sim.rp, self.sim.dat)

    def set_escape_fraction_neutrinos(mut self, frac: Float64):
        self.sim.rp.escape_fraction_neutrinos = frac
        self.sim.oEscape.Calc_Escape_Nu(self.sim.rp, self.sim.dat)

    def set_escape_fraction_charged_particles(mut self, frac: Float64):
        self.sim.rp.escape_fraction_pairs = frac
        self.sim.rp.escape_fraction_protons = frac
        self.sim.rp.escape_fraction_pions = frac
        self.sim.rp.escape_fraction_muons = frac
        self.sim.oEscape.Calc_Escape_Proton(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Pipm(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Mu(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Lep(self.sim.rp, self.sim.dat)

    def set_escape_fraction_neutral_particles(mut self, frac: Float64):
        self.sim.rp.escape_fraction_photons = frac
        self.sim.rp.escape_fraction_neutrons = frac
        self.sim.rp.escape_fraction_neutrinos = frac
        self.sim.oEscape.Calc_Escape_Neutron(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Nu(self.sim.rp, self.sim.dat)
        self.sim.oEscape.Calc_Escape_Phot(self.sim.rp, self.sim.dat)

    def set_electron_accel_factor(mut self, eta: Float64):
        self.sim.rp.electron_accel_factor_for_emax_estimate = eta

    def set_proton_accel_factor(mut self, eta: Float64):
        self.sim.rp.proton_accel_factor_for_emax_estimate = eta

    def get_egrid_lep(mut self) -> Vec:
        return scaled(self.sim.dat.egrid_lep, MEC2_EV)

    def get_egrid_photons(mut self) -> Vec:
        return scaled(self.sim.dat.egrid_pho, MEC2_EV)

    def get_egrid_had(mut self) -> Vec:
        return scaled(self.sim.dat.egrid_had, 1e9)

    def get_egrid_neutrinos(mut self) -> Vec:
        return scaled(self.sim.dat.egrid_nu, 1e9)

    def get_grid_size_lep(mut self) -> Int:
        return self.sim.dat.nbins_lep

    def get_grid_size_photons(mut self) -> Int:
        return self.sim.dat.nbins_pho

    def get_grid_size_had(mut self) -> Int:
        return self.sim.dat.nbins_had

    def get_grid_size_neutrinos(mut self) -> Int:
        return self.sim.dat.nbins_nu

    def eV_to_lep_grid_index(mut self, E_eV: Float64) -> Int:
        return self.sim.dat.eV_to_lep_grid_index(E_eV)

    def eV_to_photon_grid_index(mut self, E_eV: Float64) -> Int:
        return self.sim.dat.eV_to_photon_grid_index(E_eV)

    def eV_to_had_grid_index(mut self, E_eV: Float64) -> Int:
        return self.sim.dat.eV_to_had_grid_index(E_eV)

    def eV_to_neutrino_grid_index(mut self, E_eV: Float64) -> Int:
        return self.sim.dat.eV_to_neutrino_grid_index(E_eV)

    def get_log_energy_spacing(mut self) -> Float64:
        return self.sim.dat.dlnE

    def get_protons(mut self) -> Vec:
        return self.sim.dat.n[P_protons].copy()

    def get_neutrons(mut self) -> Vec:
        return self.sim.dat.n[P_neutrons].copy()

    def get_pions(mut self) -> Vec:
        return summed(
            1.0, self.sim.dat.n[P_pion_plus], self.sim.dat.n[P_pion_minus]
        )

    def get_pion_plus(mut self) -> Vec:
        return self.sim.dat.n[P_pion_plus].copy()

    def get_pion_minus(mut self) -> Vec:
        return self.sim.dat.n[P_pion_minus].copy()

    def _with_chains(self, made: Int, kaon: Int, charm: Int) -> Vec:
        """Photo-hadronic or pp component of the pions, muons or neutrinos: the
        part from the pions made in the interactions (n[made]) plus the parts
        from the kaon decays (n[kaon]) and from the charm decays (n[charm])."""
        return summed(
            1.0,
            self.sim.dat.n[made],
            self.sim.dat.n[kaon],
            self.sim.dat.n[charm],
        )

    def get_pions_photopion(mut self) -> Vec:
        return summed(
            1.0,
            self._with_chains(P_pion_plus_pg, P_pion_plus_Kpg, P_pion_plus_Cpg),
            self._with_chains(
                P_pion_minus_pg, P_pion_minus_Kpg, P_pion_plus_Cpg + 1
            ),
        )

    def get_pion_plus_photopion(mut self) -> Vec:
        return self._with_chains(
            P_pion_plus_pg, P_pion_plus_Kpg, P_pion_plus_Cpg
        )

    def get_pion_minus_photopion(mut self) -> Vec:
        return self._with_chains(
            P_pion_minus_pg, P_pion_minus_Kpg, P_pion_plus_Cpg + 1
        )

    def get_pions_proton_proton(mut self) -> Vec:
        return summed(
            1.0,
            self._with_chains(P_pion_plus_pp, P_pion_plus_Kpp, P_pion_plus_Cpp),
            self._with_chains(
                P_pion_minus_pp, P_pion_minus_Kpp, P_pion_plus_Cpp + 1
            ),
        )

    def get_pion_plus_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_pion_plus_pp, P_pion_plus_Kpp, P_pion_plus_Cpp
        )

    def get_pion_minus_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_pion_minus_pp, P_pion_minus_Kpp, P_pion_plus_Cpp + 1
        )

    def get_mu_minus_left(mut self) -> Vec:
        return self.sim.dat.n[P_muon_minus_left].copy()

    def get_mu_minus_right(mut self) -> Vec:
        return self.sim.dat.n[P_muon_minus_right].copy()

    def get_mu_plus_left(mut self) -> Vec:
        return self.sim.dat.n[P_muon_plus_left].copy()

    def get_mu_plus_right(mut self) -> Vec:
        return self.sim.dat.n[P_muon_plus_right].copy()

    def get_muons(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_muon_minus_left],
            self.sim.dat.n[P_muon_minus_right],
            self.sim.dat.n[P_muon_plus_left],
            self.sim.dat.n[P_muon_plus_right],
        )

    def get_mu_minus_left_photopion(mut self) -> Vec:
        return self._with_chains(
            P_muon_minus_left_pg,
            P_muon_plus_left_Kpg + 2,
            P_muon_plus_left_Cpg + 2,
        )

    def get_mu_minus_right_photopion(mut self) -> Vec:
        return self._with_chains(
            P_muon_minus_right_pg,
            P_muon_plus_left_Kpg + 3,
            P_muon_plus_left_Cpg + 3,
        )

    def get_mu_plus_left_photopion(mut self) -> Vec:
        return self._with_chains(
            P_muon_plus_left_pg, P_muon_plus_left_Kpg, P_muon_plus_left_Cpg
        )

    def get_mu_plus_right_photopion(mut self) -> Vec:
        return self._with_chains(
            P_muon_plus_right_pg,
            P_muon_plus_left_Kpg + 1,
            P_muon_plus_left_Cpg + 1,
        )

    def get_muons_photopion(mut self) -> Vec:
        return summed(
            1.0,
            self._with_chains(
                P_muon_minus_left_pg,
                P_muon_plus_left_Kpg + 2,
                P_muon_plus_left_Cpg + 2,
            ),
            self._with_chains(
                P_muon_minus_right_pg,
                P_muon_plus_left_Kpg + 3,
                P_muon_plus_left_Cpg + 3,
            ),
            self._with_chains(
                P_muon_plus_left_pg, P_muon_plus_left_Kpg, P_muon_plus_left_Cpg
            ),
            self._with_chains(
                P_muon_plus_right_pg,
                P_muon_plus_left_Kpg + 1,
                P_muon_plus_left_Cpg + 1,
            ),
        )

    def get_mu_minus_left_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_muon_minus_left_pp,
            P_muon_plus_left_Kpp + 2,
            P_muon_plus_left_Cpp + 2,
        )

    def get_mu_minus_right_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_muon_minus_right_pp,
            P_muon_plus_left_Kpp + 3,
            P_muon_plus_left_Cpp + 3,
        )

    def get_mu_plus_left_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_muon_plus_left_pp, P_muon_plus_left_Kpp, P_muon_plus_left_Cpp
        )

    def get_mu_plus_right_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_muon_plus_right_pp,
            P_muon_plus_left_Kpp + 1,
            P_muon_plus_left_Cpp + 1,
        )

    def get_muons_proton_proton(mut self) -> Vec:
        return summed(
            1.0,
            self._with_chains(
                P_muon_minus_left_pp,
                P_muon_plus_left_Kpp + 2,
                P_muon_plus_left_Cpp + 2,
            ),
            self._with_chains(
                P_muon_minus_right_pp,
                P_muon_plus_left_Kpp + 3,
                P_muon_plus_left_Cpp + 3,
            ),
            self._with_chains(
                P_muon_plus_left_pp, P_muon_plus_left_Kpp, P_muon_plus_left_Cpp
            ),
            self._with_chains(
                P_muon_plus_right_pp,
                P_muon_plus_left_Kpp + 1,
                P_muon_plus_left_Cpp + 1,
            ),
        )

    def get_electron_neutrinos(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el].copy()

    def get_electron_anti_neutrinos(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_anti].copy()

    def get_muon_neutrinos(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu].copy()

    def get_muon_anti_neutrinos(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_anti].copy()

    def get_neutrinos(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_el],
            self.sim.dat.n[P_neutrino_el_anti],
            self.sim.dat.n[P_neutrino_mu],
            self.sim.dat.n[P_neutrino_mu_anti],
        )

    def _nu_mu_component(self, mu: Int, pi: Int, kaon: Int, charm: Int) -> Vec:
        """Photo-hadronic or pp component of the nu_mu or anti-nu_mu: the parts
        from the decays of the muons (n[mu]) and of the pions (n[pi]) made in
        the interactions, and from the kaon and charm decays (n[kaon],
        n[charm])."""
        return summed(
            1.0,
            self.sim.dat.n[mu],
            self.sim.dat.n[pi],
            self.sim.dat.n[kaon],
            self.sim.dat.n[charm],
        )

    def get_electron_neutrinos_photopion(mut self) -> Vec:
        return self._with_chains(
            P_neutrino_el_pg, P_neutrino_el_Kpg, P_neutrino_el_Cpg
        )

    def get_electron_anti_neutrinos_photopion(mut self) -> Vec:
        return self._with_chains(
            P_neutrino_el_anti_pg, P_neutrino_el_Kpg + 1, P_neutrino_el_Cpg + 1
        )

    def get_muon_neutrinos_photopion(mut self) -> Vec:
        return self._nu_mu_component(
            P_neutrino_mu_pg,
            P_neutrino_mu_pidec_pg,
            P_neutrino_el_Kpg + 2,
            P_neutrino_el_Cpg + 2,
        )

    def get_muon_anti_neutrinos_photopion(mut self) -> Vec:
        return self._nu_mu_component(
            P_neutrino_mu_anti_pg,
            P_neutrino_mu_pidec_pg + 1,
            P_neutrino_el_Kpg + 3,
            P_neutrino_el_Cpg + 3,
        )

    def get_neutrinos_photopion(mut self) -> Vec:
        return summed(
            1.0,
            self.get_electron_neutrinos_photopion(),
            self.get_electron_anti_neutrinos_photopion(),
            self.get_muon_neutrinos_photopion(),
            self.get_muon_anti_neutrinos_photopion(),
        )

    def get_electron_neutrinos_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_neutrino_el_pp, P_neutrino_el_Kpp, P_neutrino_el_Cpp
        )

    def get_electron_anti_neutrinos_proton_proton(mut self) -> Vec:
        return self._with_chains(
            P_neutrino_el_anti_pp, P_neutrino_el_Kpp + 1, P_neutrino_el_Cpp + 1
        )

    def get_muon_neutrinos_proton_proton(mut self) -> Vec:
        return self._nu_mu_component(
            P_neutrino_mu_pp,
            P_neutrino_mu_pidec_pp,
            P_neutrino_el_Kpp + 2,
            P_neutrino_el_Cpp + 2,
        )

    def get_muon_anti_neutrinos_proton_proton(mut self) -> Vec:
        return self._nu_mu_component(
            P_neutrino_mu_anti_pp,
            P_neutrino_mu_pidec_pp + 1,
            P_neutrino_el_Kpp + 3,
            P_neutrino_el_Cpp + 3,
        )

    def get_neutrinos_proton_proton(mut self) -> Vec:
        return summed(
            1.0,
            self.get_electron_neutrinos_proton_proton(),
            self.get_electron_anti_neutrinos_proton_proton(),
            self.get_muon_neutrinos_proton_proton(),
            self.get_muon_anti_neutrinos_proton_proton(),
        )

    def get_electrons(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_electrons], N_SSC)

    def get_positrons(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_positrons], N_SSC)

    def get_pairs(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs], N_SSC)

    def get_pairs_injected(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs_in], N_SSC)

    def get_pairs_annihilation(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs_pair], N_SSC)

    def get_pairs_bethe_heitler(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs_bh], N_SSC)

    def get_pairs_photopion(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs_pg], N_SSC)

    def get_pairs_proton_proton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs_pp], N_SSC)

    def get_photons(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons], N_SSC)

    def get_photons_except_injected(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_except_injected], N_SSC)

    def get_photons_injected_electrons_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_insy], N_SSC)

    def get_photons_injected_electrons_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_inic], N_SSC)

    def get_photons_injected_electrons_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_insy],
            self.sim.dat.n[P_photons_inic],
        )

    def get_photons_annihilation_pairs_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pairsy], N_SSC)

    def get_photons_annihilation_pairs_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pairic], N_SSC)

    def get_photons_annihilation_pairs_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_pairsy],
            self.sim.dat.n[P_photons_pairic],
        )

    def get_photons_bethe_heitler_pairs_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_bhsy], N_SSC)

    def get_photons_bethe_heitler_pairs_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_bhic], N_SSC)

    def get_photons_bethe_heitler_pairs_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_bhsy],
            self.sim.dat.n[P_photons_bhic],
        )

    def get_photons_photo_pion_pairs_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pgsy], N_SSC)

    def get_photons_photo_pion_pairs_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pgic], N_SSC)

    def get_photons_photo_pion_pairs_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_pgsy],
            self.sim.dat.n[P_photons_pgic],
        )

    def get_photons_proton_proton_pairs_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_ppsy], N_SSC)

    def get_photons_proton_proton_pairs_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_ppic], N_SSC)

    def get_photons_proton_proton_pairs_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_ppsy],
            self.sim.dat.n[P_photons_ppic],
        )

    def get_photons_protons_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_psy], N_SSC)

    def get_photons_protons_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pic], N_SSC)

    def get_photons_protons_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_psy],
            self.sim.dat.n[P_photons_pic],
        )

    def get_photons_pions_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pisy], N_SSC)

    def get_photons_pions_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_piic], N_SSC)

    def get_photons_pions_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_pisy],
            self.sim.dat.n[P_photons_piic],
        )

    def get_photons_muons_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_musy], N_SSC)

    def get_photons_muons_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_muic], N_SSC)

    def get_photons_muons_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_musy],
            self.sim.dat.n[P_photons_muic],
        )

    def get_photons_pi0_decay(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_pi0_pg],
            self.sim.dat.n[P_photons_pi0_pp],
        )

    def get_photons_pi0_decay_photopion(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pi0_pg], N_SSC)

    def get_photons_pi0_decay_proton_proton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pi0_pp], N_SSC)

    def set_current_densities_protons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_protons], InputArray, 1.0)

    def set_current_densities_neutrons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_neutrons], InputArray, 1.0)

    def set_current_densities_pion_plus(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_pion_plus], InputArray, 1.0)

    def set_current_densities_pion_minus(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_pion_minus], InputArray, 1.0)

    def set_current_densities_mu_minus_left(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_muon_minus_left], InputArray, 1.0)

    def set_current_densities_mu_plus_left(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_muon_plus_left], InputArray, 1.0)

    def set_current_densities_mu_minus_right(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_muon_minus_right], InputArray, 1.0)

    def set_current_densities_mu_plus_right(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_muon_plus_right], InputArray, 1.0)

    def set_current_densities_electron_neutrinos(
        mut self, InputArray: Vec
    ) raises:
        assign_divided(self.sim.dat.n[P_neutrino_el], InputArray, 1.0)

    def set_current_densities_electron_anti_neutrinos(
        mut self, InputArray: Vec
    ) raises:
        assign_divided(self.sim.dat.n[P_neutrino_el_anti], InputArray, 1.0)

    def set_current_densities_muon_neutrinos(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_neutrino_mu], InputArray, 1.0)

    def set_current_densities_muon_anti_neutrinos(
        mut self, InputArray: Vec
    ) raises:
        assign_divided(self.sim.dat.n[P_neutrino_mu_anti], InputArray, 1.0)

    def set_current_densities_electrons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_electrons], InputArray, N_SSC)

    def set_current_densities_positrons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_positrons], InputArray, N_SSC)

    def set_current_densities_photons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.dat.n[P_photons], InputArray, N_SSC)

    def get_injection_rate_protons(mut self) -> Vec:
        return self.sim.oInj.inj_proton.copy()

    def get_injection_rate_neutrons(mut self) -> Vec:
        return self.sim.oInj.inj_neutron.copy()

    def get_injection_rate_pion_plus(mut self) -> Vec:
        return self.sim.oInj.inj_piplus.copy()

    def get_injection_rate_pion_minus(mut self) -> Vec:
        return self.sim.oInj.inj_piminus.copy()

    def get_injection_rate_pi0(mut self) -> Vec:
        return self.sim.oInj.inj_pi0.copy()

    def get_injection_rate_mu_minus_left(mut self) -> Vec:
        return self.sim.oInj.inj_muminusleft.copy()

    def get_injection_rate_mu_plus_left(mut self) -> Vec:
        return self.sim.oInj.inj_muplusleft.copy()

    def get_injection_rate_mu_minus_right(mut self) -> Vec:
        return self.sim.oInj.inj_muminusright.copy()

    def get_injection_rate_mu_plus_right(mut self) -> Vec:
        return self.sim.oInj.inj_muplusright.copy()

    def get_injection_rate_electron_neutrinos(mut self) -> Vec:
        return self.sim.oInj.inj_nue.copy()

    def get_injection_rate_electron_anti_neutrinos(mut self) -> Vec:
        return self.sim.oInj.inj_antinue.copy()

    def get_injection_rate_muon_neutrinos(mut self) -> Vec:
        return self.sim.oInj.inj_numu.copy()

    def get_injection_rate_muon_anti_neutrinos(mut self) -> Vec:
        return self.sim.oInj.inj_antinumu.copy()

    def get_injection_rate_electrons(mut self) -> Vec:
        return scaled(self.sim.oInj.inj_electron, RATE_SSC)

    def get_injection_rate_positrons(mut self) -> Vec:
        return scaled(self.sim.oInj.inj_positron, RATE_SSC)

    def get_injection_rate_photons(mut self) -> Vec:
        return scaled(self.sim.oInj.inj_photon, RATE_SSC)

    def set_injection_rate_protons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_proton, InputArray, 1.0)

    def set_injection_rate_neutrons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_neutron, InputArray, 1.0)

    def set_injection_rate_pion_plus(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_piplus, InputArray, 1.0)

    def set_injection_rate_pion_minus(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_piminus, InputArray, 1.0)

    def set_injection_rate_pi0(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_pi0, InputArray, 1.0)

    def set_injection_rate_mu_minus_left(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_muminusleft, InputArray, 1.0)

    def set_injection_rate_mu_plus_left(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_muplusleft, InputArray, 1.0)

    def set_injection_rate_mu_minus_right(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_muminusright, InputArray, 1.0)

    def set_injection_rate_mu_plus_right(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_muplusright, InputArray, 1.0)

    def set_injection_rate_electron_neutrinos(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_nue, InputArray, 1.0)

    def set_injection_rate_electron_anti_neutrinos(
        mut self, InputArray: Vec
    ) raises:
        assign_divided(self.sim.oInj.inj_antinue, InputArray, 1.0)

    def set_injection_rate_muon_neutrinos(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_numu, InputArray, 1.0)

    def set_injection_rate_muon_anti_neutrinos(
        mut self, InputArray: Vec
    ) raises:
        assign_divided(self.sim.oInj.inj_antinumu, InputArray, 1.0)

    def set_injection_rate_electrons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_electron, InputArray, RATE_SSC)

    def set_injection_rate_positrons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_positron, InputArray, RATE_SSC)

    def set_injection_rate_photons(mut self, InputArray: Vec) raises:
        assign_divided(self.sim.oInj.inj_photon, InputArray, RATE_SSC)

    def set_t_proton_escape(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(self.sim.oEscape.afescPRO, InputArray, 1.0)

    def set_t_neutron_escape(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(self.sim.oEscape.afescNTR, InputArray, 1.0)

    def set_t_pion_escape(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(self.sim.oEscape.afescPipm, InputArray, 1.0)

    def set_t_muon_escape(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(self.sim.oEscape.afescMu, InputArray, 1.0)

    def set_t_neutrino_escape(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(self.sim.oEscape.afescNU, InputArray, 1.0)

    def set_t_pair_escape(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(self.sim.oEscape.afescE, InputArray, T_SSC)

    def set_t_photon_escape(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(self.sim.oEscape.afescG, InputArray, T_SSC)

    def set_t_proton_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolPRO, InputArray, -1.0
        )

    def set_t_neutron_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolNTR, InputArray, -1.0
        )

    def set_t_pion_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolPipm, InputArray, -1.0
        )

    def set_t_muon_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolMu, InputArray, -1.0
        )

    def set_t_neutrino_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolNU, InputArray, -1.0
        )

    def set_t_pair_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolE, InputArray, -T_SSC
        )

    def set_t_photon_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolG, InputArray, -T_SSC
        )

    def get_t_proton_syn(mut self) -> Vec:
        return timescales(self.sim.S[SA_proton].Asy, 1.0, -1.0)

    def get_t_proton_compton(mut self) -> Vec:
        return timescales(self.sim.S[SA_proton].Aic, 1.0, -1.0)

    def get_t_proton_acceleration(mut self) -> Vec:
        var res = zeros(self.sim.dat.nbins_had)
        for i in range(self.sim.dat.nbins_had):
            res[i] = 1.0 / (
                self.sim.oAcc.AAChadBohm(self.sim.rp, self.sim.dat, i) + 1e-200
            )
        return res^

    def get_t_proton_adiabatic(mut self) -> Vec:
        return constant_timescales(
            self.sim.dat.nbins_had, 1.0, -Expansion.AAdiHad(self.sim.rp)
        )

    def get_t_proton_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescPRO, 1.0, 1.0)

    def get_t_proton_bethe_heitler(mut self) -> Vec:
        return timescales(self.sim.oBH.ABHHadP, 1.0, -1.0)

    def get_t_proton_photopion(mut self) -> Vec:
        return timescales(self.sim.oPhotoHadron.afPG_PRO, 1.0, 1.0)

    def get_t_proton_proton_proton(mut self) -> Vec:
        return timescales(self.sim.oProtonProton.APP_PRO, 1.0, -1.0)

    def get_t_proton_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolPRO, 1.0, -1.0)

    def get_t_neutron_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescNTR, 1.0, 1.0)

    def get_t_neutron_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolNTR, 1.0, -1.0)

    def get_t_pion_syn(mut self) -> Vec:
        return timescales(self.sim.S[SA_Pip].Asy, 1.0, -1.0)

    def get_t_pion_compton(mut self) -> Vec:
        return timescales(self.sim.S[SA_Pip].Aic, 1.0, -1.0)

    def get_t_pion_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescPipm, 1.0, 1.0)

    def get_t_pion_decay(mut self) -> Vec:
        return timescales(self.sim.S[SA_Pip].alpha_decay, 1.0, 1.0)

    def get_t_pion_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolPipm, 1.0, -1.0)

    def get_t_muon_syn(mut self) -> Vec:
        return timescales(self.sim.S[SA_MmL].Asy, 1.0, -1.0)

    def get_t_muon_compton(mut self) -> Vec:
        return timescales(self.sim.S[SA_MmL].Aic, 1.0, -1.0)

    def get_t_muon_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescMu, 1.0, 1.0)

    def get_t_muon_decay(mut self) -> Vec:
        return timescales(self.sim.S[SA_MmL].alpha_decay, 1.0, 1.0)

    def get_t_muon_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolMu, 1.0, -1.0)

    def get_t_neutrino_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescNU, 1.0, 1.0)

    def get_t_neutrino_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolNU, 1.0, -1.0)

    def get_t_pair_syn(mut self) -> Vec:
        return timescales(self.sim.S[self._pair_solver()].Asy, T_SSC, -1.0)

    def get_t_pair_compton(mut self) -> Vec:
        return timescales(self.sim.S[self._pair_solver()].Aic, T_SSC, -1.0)

    def get_t_pair_acceleration(mut self) -> Vec:
        var res = zeros(self.sim.dat.nbins_lep)
        for i in range(self.sim.dat.nbins_lep):
            res[i] = T_SSC / (
                self.sim.oAcc.AAClepBohm(self.sim.rp, self.sim.dat, i) + 1e-200
            )
        return res^

    def get_t_pair_adiabatic(mut self) -> Vec:
        return constant_timescales(
            self.sim.dat.nbins_lep, T_SSC, -Expansion.AAdiLep(self.sim.rp)
        )

    def get_t_pair_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescE, T_SSC, 1.0)

    def get_t_pair_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolE, T_SSC, -1.0)

    def get_t_pair_emission_from_annihilation(mut self) -> Vec:
        return emission_timescales(
            self.sim.dat.n[P_electrons], self.sim.c[C_qepair], T_SSC
        )

    def get_t_pair_emission_from_injection(mut self) -> Vec:
        return emission_timescales(
            self.sim.dat.n[P_electrons], self.sim.c[C_qein], T_SSC
        )

    def get_t_pair_emission_from_bethe_heitler(mut self) -> Vec:
        return emission_timescales(
            self.sim.dat.n[P_electrons], self.sim.c[C_qebh], T_SSC
        )

    def get_t_pair_emission_from_photopion(mut self) -> Vec:
        return emission_timescales(
            self.sim.dat.n[P_electrons], self.sim.c[C_qepg], T_SSC
        )

    def get_t_pair_emission_from_proton_proton(mut self) -> Vec:
        return emission_timescales(
            self.sim.dat.n[P_electrons], self.sim.c[C_qepp], T_SSC
        )

    def get_t_photon_ssa(mut self) -> Vec:
        return timescales(self.sim.oSyn.afsyG, T_SSC, 1.0)

    def get_t_photon_compton(mut self) -> Vec:
        return timescales(self.sim.oComp.aficG, T_SSC, 1.0)

    def get_t_photon_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescG, T_SSC, 1.0)

    def get_t_photon_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolG, T_SSC, -1.0)

    def get_t_photon_annihilation(mut self) -> Vec:
        return timescales(self.sim.oPair.afpairG, T_SSC, 1.0)

    def get_t_photon_photopion(mut self) -> Vec:
        return timescales(self.sim.oPhotoHadron.afPG_Rad, 1.0, 1.0)

    def get_estimated_electron_emax(mut self) -> Float64:
        return self.sim.rp.estimated_electron_emax

    def get_estimated_proton_emax(mut self) -> Float64:
        return self.sim.rp.estimated_proton_emax

    # ------------------------------------------------------------------
    # processes am3_mojo adds to AM3 (tools/am3_mojo_api.json)
    # ------------------------------------------------------------------
    def get_process_pair_annihilation(mut self) -> Bool:
        return self.sim.process_pair_annihilation

    def set_process_pair_annihilation(mut self, new_value: Bool):
        self.sim.process_pair_annihilation = new_value

    def get_process_pair_annihilation_emission(mut self) -> Bool:
        return self.sim.process_pair_annihilation_emission

    def set_process_pair_annihilation_emission(mut self, new_value: Bool):
        self.sim.process_pair_annihilation_emission = new_value

    def get_process_pair_annihilation_cooling(mut self) -> Bool:
        return self.sim.process_pair_annihilation_cooling

    def set_process_pair_annihilation_cooling(mut self, new_value: Bool):
        self.sim.process_pair_annihilation_cooling = new_value

    def get_photons_pair_annihilation(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_pair_ann], N_SSC)

    def get_t_electron_annihilation(mut self) -> Vec:
        return timescales(self.sim.oPairAnn.Gamma_ele, T_SSC, 1.0)

    def get_t_positron_annihilation(mut self) -> Vec:
        return timescales(self.sim.oPairAnn.Gamma_pos, T_SSC, 1.0)

    def get_process_triplet_pair_production(mut self) -> Bool:
        return self.sim.process_triplet_pair_production

    def set_process_triplet_pair_production(mut self, new_value: Bool):
        self.sim.process_triplet_pair_production = new_value

    def get_process_triplet_pair_production_emission(mut self) -> Bool:
        return self.sim.process_triplet_pair_production_emission

    def set_process_triplet_pair_production_emission(mut self, new_value: Bool):
        self.sim.process_triplet_pair_production_emission = new_value

    def get_process_triplet_pair_production_cooling(mut self) -> Bool:
        return self.sim.process_triplet_pair_production_cooling

    def set_process_triplet_pair_production_cooling(mut self, new_value: Bool):
        self.sim.process_triplet_pair_production_cooling = new_value

    def get_pairs_triplet(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs_tpp], N_SSC)

    def get_photons_triplet_pairs_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_tppsy], N_SSC)

    def get_photons_triplet_pairs_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_tppic], N_SSC)

    def get_photons_triplet_pairs_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_tppsy],
            self.sim.dat.n[P_photons_tppic],
        )

    def get_t_pair_triplet(mut self) -> Vec:
        return timescales(self.sim.oTPP.eloss, T_SSC, 1.0)

    def get_t_photon_triplet(mut self) -> Vec:
        return timescales(self.sim.oTPP.alpha_g, T_SSC, 1.0)

    def get_process_neutron_decay(mut self) -> Bool:
        return self.sim.process_neutron_decay

    def set_process_neutron_decay(mut self, new_value: Bool):
        self.sim.process_neutron_decay = new_value

    def get_electrons_neutron_decay(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_pairs_ndec], N_SSC)

    def get_photons_neutron_decay_electrons_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_ndecsy], N_SSC)

    def get_photons_neutron_decay_electrons_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_ndecic], N_SSC)

    def get_photons_neutron_decay_electrons_syn_compton(mut self) -> Vec:
        return summed(
            N_SSC,
            self.sim.dat.n[P_photons_ndecsy],
            self.sim.dat.n[P_photons_ndecic],
        )

    def get_electron_anti_neutrinos_neutron_decay(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_anti_ndec].copy()

    def get_t_neutron_decay(mut self) -> Vec:
        return timescales(self.sim.S[SA_neutron].alpha_decay, 1.0, 1.0)

    # AM3's photo-hadronic and pp neutrinos by parent: the pion and the muon
    # decays (am3_mojo addition)
    def get_neutrinos_pion_decay_photopion(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_mu_pidec_pg],
            self.sim.dat.n[P_neutrino_mu_pidec_pg + 1],
        )

    def get_muon_neutrinos_pion_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_pidec_pg].copy()

    def get_muon_anti_neutrinos_pion_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_pidec_pg + 1].copy()

    def get_neutrinos_pion_decay_proton_proton(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_mu_pidec_pp],
            self.sim.dat.n[P_neutrino_mu_pidec_pp + 1],
        )

    def get_muon_neutrinos_pion_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_pidec_pp].copy()

    def get_muon_anti_neutrinos_pion_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_pidec_pp + 1].copy()

    def get_neutrinos_muon_decay_photopion(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_el_pg],
            self.sim.dat.n[P_neutrino_el_anti_pg],
            self.sim.dat.n[P_neutrino_mu_pg],
            self.sim.dat.n[P_neutrino_mu_anti_pg],
        )

    def get_electron_neutrinos_muon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_pg].copy()

    def get_electron_anti_neutrinos_muon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_anti_pg].copy()

    def get_muon_neutrinos_muon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_pg].copy()

    def get_muon_anti_neutrinos_muon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_anti_pg].copy()

    def get_neutrinos_muon_decay_proton_proton(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_el_pp],
            self.sim.dat.n[P_neutrino_el_anti_pp],
            self.sim.dat.n[P_neutrino_mu_pp],
            self.sim.dat.n[P_neutrino_mu_anti_pp],
        )

    def get_electron_neutrinos_muon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_pp].copy()

    def get_electron_anti_neutrinos_muon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_anti_pp].copy()

    def get_muon_neutrinos_muon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_pp].copy()

    def get_muon_anti_neutrinos_muon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_mu_anti_pp].copy()

    def get_process_kaons(mut self) -> Bool:
        return self.sim.process_kaons

    def set_process_kaons(mut self, new_value: Bool):
        self.sim.process_kaons = new_value

    def get_process_kaon_decay(mut self) -> Bool:
        return self.sim.process_kaon_decay

    def set_process_kaon_decay(mut self, new_value: Bool):
        self.sim.process_kaon_decay = new_value

    def get_process_kaon_syn(mut self) -> Bool:
        return self.sim.process_kaon_syn

    def set_process_kaon_syn(mut self, new_value: Bool):
        self.sim.process_kaon_syn = new_value

    def get_process_kaon_syn_emission(mut self) -> Bool:
        return self.sim.process_kaon_syn_emission

    def set_process_kaon_syn_emission(mut self, new_value: Bool):
        self.sim.process_kaon_syn_emission = new_value

    def get_process_kaon_syn_cooling(mut self) -> Bool:
        return self.sim.process_kaon_syn_cooling

    def set_process_kaon_syn_cooling(mut self, new_value: Bool):
        self.sim.process_kaon_syn_cooling = new_value

    def get_kaons(mut self) -> Vec:
        var res = self.sim.dat.n[P_kaon_plus].copy()
        for i in range(len(res)):
            res[i] += self.sim.dat.n[P_kaon_minus][i]
        return res^

    def get_kaon_plus(mut self) -> Vec:
        return self.sim.dat.n[P_kaon_plus].copy()

    def get_kaon_minus(mut self) -> Vec:
        return self.sim.dat.n[P_kaon_minus].copy()

    def get_kaon_0S(mut self) -> Vec:
        return self.sim.dat.n[P_kaon_0S].copy()

    def get_kaon_0L(mut self) -> Vec:
        return self.sim.dat.n[P_kaon_0L].copy()

    def get_t_kaon_0S_decay(mut self) -> Vec:
        return timescales(self.sim.S[SA_KS].alpha_decay, 1.0, 1.0)

    def get_t_kaon_0L_decay(mut self) -> Vec:
        return timescales(self.sim.S[SA_KL].alpha_decay, 1.0, 1.0)

    def get_photons_kaons_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_kasy], N_SSC)

    def _kaon_cooling_solver(self) -> Int:
        """The K+ solver whose cooling rates are current: the photo-hadronic
        population's, or the pp population's when only pp makes kaons (the
        two are configured identically)."""
        return kaon_solver(KP, not self.sim.kaons_pg_on())

    def get_t_kaon_syn(mut self) -> Vec:
        return timescales(
            self.sim.S[self._kaon_cooling_solver()].Asy, 1.0, -1.0
        )

    def get_t_kaon_decay(mut self) -> Vec:
        return timescales(self.sim.S[SA_K].alpha_decay, 1.0, 1.0)

    def get_t_kaon_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescPipm, 1.0, 1.0)

    def get_t_kaon_neutral_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescNTR, 1.0, 1.0)

    def get_process_kaon_compton(mut self) -> Bool:
        return self.sim.process_kaon_compton

    def set_process_kaon_compton(mut self, new_value: Bool):
        self.sim.process_kaon_compton = new_value

    def get_process_kaon_compton_emission(mut self) -> Bool:
        return self.sim.process_kaon_compton_emission

    def set_process_kaon_compton_emission(mut self, new_value: Bool):
        self.sim.process_kaon_compton_emission = new_value

    def get_process_kaon_compton_cooling(mut self) -> Bool:
        return self.sim.process_kaon_compton_cooling

    def set_process_kaon_compton_cooling(mut self, new_value: Bool):
        self.sim.process_kaon_compton_cooling = new_value

    def get_photons_kaons_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_kaic], N_SSC)

    def get_neutrinos_kaon_decay_photopion(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_el_Kpg],
            self.sim.dat.n[P_neutrino_el_Kpg + 1],
            self.sim.dat.n[P_neutrino_el_Kpg + 2],
            self.sim.dat.n[P_neutrino_el_Kpg + 3],
        )

    def get_electron_neutrinos_kaon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpg].copy()

    def get_electron_anti_neutrinos_kaon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpg + 1].copy()

    def get_muon_neutrinos_kaon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpg + 2].copy()

    def get_muon_anti_neutrinos_kaon_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpg + 3].copy()

    def get_neutrinos_kaon_decay_proton_proton(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_el_Kpp],
            self.sim.dat.n[P_neutrino_el_Kpp + 1],
            self.sim.dat.n[P_neutrino_el_Kpp + 2],
            self.sim.dat.n[P_neutrino_el_Kpp + 3],
        )

    def get_electron_neutrinos_kaon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpp].copy()

    def get_electron_anti_neutrinos_kaon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpp + 1].copy()

    def get_muon_neutrinos_kaon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpp + 2].copy()

    def get_muon_anti_neutrinos_kaon_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Kpp + 3].copy()

    def get_t_kaon_compton(mut self) -> Vec:
        return timescales(
            self.sim.S[self._kaon_cooling_solver()].Aic, 1.0, -1.0
        )

    def set_t_kaon_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolK, InputArray, -1.0
        )

    def get_t_kaon_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolK, 1.0, -1.0)

    # ------------------------------------------------------------------
    # charm hadrons (am3_mojo addition)
    # ------------------------------------------------------------------
    def get_process_charm(mut self) -> Bool:
        return self.sim.process_charm

    def set_process_charm(mut self, new_value: Bool):
        self.sim.process_charm = new_value

    def get_process_charm_decay(mut self) -> Bool:
        return self.sim.process_charm_decay

    def set_process_charm_decay(mut self, new_value: Bool):
        self.sim.process_charm_decay = new_value

    def get_process_charm_syn(mut self) -> Bool:
        return self.sim.process_charm_syn

    def set_process_charm_syn(mut self, new_value: Bool):
        self.sim.process_charm_syn = new_value

    def get_process_charm_syn_emission(mut self) -> Bool:
        return self.sim.process_charm_syn_emission

    def set_process_charm_syn_emission(mut self, new_value: Bool):
        self.sim.process_charm_syn_emission = new_value

    def get_process_charm_syn_cooling(mut self) -> Bool:
        return self.sim.process_charm_syn_cooling

    def set_process_charm_syn_cooling(mut self, new_value: Bool):
        self.sim.process_charm_syn_cooling = new_value

    def get_process_charm_compton(mut self) -> Bool:
        return self.sim.process_charm_compton

    def set_process_charm_compton(mut self, new_value: Bool):
        self.sim.process_charm_compton = new_value

    def get_process_charm_compton_emission(mut self) -> Bool:
        return self.sim.process_charm_compton_emission

    def set_process_charm_compton_emission(mut self, new_value: Bool):
        self.sim.process_charm_compton_emission = new_value

    def get_process_charm_compton_cooling(mut self) -> Bool:
        return self.sim.process_charm_compton_cooling

    def set_process_charm_compton_cooling(mut self, new_value: Bool):
        self.sim.process_charm_compton_cooling = new_value

    def get_charm_hadrons(mut self) -> Vec:
        var res = self.sim.dat.n[P_charm].copy()
        for s in range(1, N_CHARM):
            for i in range(len(res)):
                res[i] += self.sim.dat.n[P_charm + s][i]
        return res^

    def get_D_plus(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CDP].copy()

    def get_D_minus(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CDM].copy()

    def get_D_zero(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CD0].copy()

    def get_anti_D_zero(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CD0B].copy()

    def get_Ds_plus(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CDSP].copy()

    def get_Ds_minus(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CDSM].copy()

    def get_Lambda_c_plus(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CLC].copy()

    def get_anti_Lambda_c_minus(mut self) -> Vec:
        return self.sim.dat.n[P_charm + CLCB].copy()

    def get_t_D_plus_decay(mut self) -> Vec:
        return timescales(self.sim.S[charm_solver(CDP)].alpha_decay, 1.0, 1.0)

    def get_t_D_zero_decay(mut self) -> Vec:
        return timescales(self.sim.S[charm_solver(CD0)].alpha_decay, 1.0, 1.0)

    def get_t_Ds_decay(mut self) -> Vec:
        return timescales(self.sim.S[charm_solver(CDSP)].alpha_decay, 1.0, 1.0)

    def get_t_Lambda_c_decay(mut self) -> Vec:
        return timescales(self.sim.S[charm_solver(CLC)].alpha_decay, 1.0, 1.0)

    def _charm_cooling_solver(self) -> Int:
        """The D+ solver whose cooling rates are current (as
        _kaon_cooling_solver)."""
        return charm_solver(CDP, not self.sim.charm_pg_on())

    def get_t_charm_syn(mut self) -> Vec:
        return timescales(
            self.sim.S[self._charm_cooling_solver()].Asy, 1.0, -1.0
        )

    def get_t_charm_compton(mut self) -> Vec:
        return timescales(
            self.sim.S[self._charm_cooling_solver()].Aic, 1.0, -1.0
        )

    def get_t_charm_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescPipm, 1.0, 1.0)

    def get_t_charm_neutral_escape(mut self) -> Vec:
        return timescales(self.sim.oEscape.afescNTR, 1.0, 1.0)

    def set_t_charm_extra_cooling(mut self, InputArray: Vec) raises:
        set_rates_from_timescales(
            self.sim.oExtraCool.AExtraCoolC, InputArray, -1.0
        )

    def get_t_charm_extra_cooling(mut self) -> Vec:
        return timescales(self.sim.oExtraCool.AExtraCoolC, 1.0, -1.0)

    def get_photons_charm_syn(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_chsy], N_SSC)

    def get_photons_charm_compton(mut self) -> Vec:
        return scaled(self.sim.dat.n[P_photons_chic], N_SSC)

    def get_neutrinos_charm_decay_photopion(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_el_Cpg],
            self.sim.dat.n[P_neutrino_el_Cpg + 1],
            self.sim.dat.n[P_neutrino_el_Cpg + 2],
            self.sim.dat.n[P_neutrino_el_Cpg + 3],
        )

    def get_electron_neutrinos_charm_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpg].copy()

    def get_electron_anti_neutrinos_charm_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpg + 1].copy()

    def get_muon_neutrinos_charm_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpg + 2].copy()

    def get_muon_anti_neutrinos_charm_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpg + 3].copy()

    def get_neutrinos_charm_decay_proton_proton(mut self) -> Vec:
        return summed(
            1.0,
            self.sim.dat.n[P_neutrino_el_Cpp],
            self.sim.dat.n[P_neutrino_el_Cpp + 1],
            self.sim.dat.n[P_neutrino_el_Cpp + 2],
            self.sim.dat.n[P_neutrino_el_Cpp + 3],
        )

    def get_electron_neutrinos_charm_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpp].copy()

    def get_electron_anti_neutrinos_charm_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpp + 1].copy()

    def get_muon_neutrinos_charm_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpp + 2].copy()

    def get_muon_anti_neutrinos_charm_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_el_Cpp + 3].copy()

    def get_tau_neutrinos_charm_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_tau_Cpg].copy()

    def get_tau_anti_neutrinos_charm_decay_photopion(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_tau_Cpg + 1].copy()

    def get_tau_neutrinos_charm_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_tau_Cpp].copy()

    def get_tau_anti_neutrinos_charm_decay_proton_proton(mut self) -> Vec:
        return self.sim.dat.n[P_neutrino_tau_Cpp + 1].copy()
