"""Port of SimulationManager (SimulationManager.cc): switches, per-step
coefficient calculation and evolution of all species."""

from std.time import perf_counter_ns
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.solver.solver import SolverAlgorithm
from am3_mojo.solver.ids import *
from am3_mojo.model.particles import *
from am3_mojo.simulation.coeffs import *
from am3_mojo.physics.acceleration import Acc
from am3_mojo.physics.escape import Escape
from am3_mojo.physics.expansion import Expansion
from am3_mojo.physics.extra_cooling import ExtraCooling
from am3_mojo.physics.injection import Injection
from am3_mojo.physics.synchrotron import Syn, SynP
from am3_mojo.physics.inverse_compton import Comp
from am3_mojo.physics.pair_production import Pair
from am3_mojo.physics.bethe_heitler import BetheHeitler
from am3_mojo.physics.photopion import PhotoPion
from am3_mojo.physics.proton_proton import ProtonProton
from am3_mojo.physics.decays import PionDecay, MuonDecay, HadIC
from am3_mojo.utils.parallel import default_num_threads
from am3_mojo.simulation.evolve import Evolution, EvolveJob, run_jobs, vp


struct SimulationManager(Movable):
    var rp: RunParams
    var dat: AM3Arrays

    # PhysicsHandler
    var oAcc: Acc
    var oPair: Pair
    var oSyn: Syn
    var oComp: Comp
    var oSynP: SynP
    var oHadIC: HadIC
    var oBH: BetheHeitler
    var oPhotoPion: PhotoPion
    var oProtonProton: ProtonProton
    var oPionDec: PionDecay
    var oMuonDec: MuonDecay
    var oExpansion: Expansion
    var oEscape: Escape
    var oInj: Injection
    var oExtraCool: ExtraCooling

    # switches
    var estimate_max_energies: Bool
    var process_parse_sed: Bool
    var process_hadronic: Bool
    var process_merge_positrons_into_electrons: Bool
    var process_escape: Bool
    var process_adiabatic_cooling: Bool
    var process_expansion: Bool
    var process_electron_syn: Bool
    var process_electron_syn_emission: Bool
    var process_electron_syn_cooling: Bool
    var process_quantum_syn: Bool
    var process_quantum_syn_cooling: Bool
    var process_ssa: Bool
    var process_proton_syn: Bool
    var process_proton_syn_emission: Bool
    var process_proton_syn_cooling: Bool
    var process_muon_syn: Bool
    var process_muon_syn_emission: Bool
    var process_muon_syn_cooling: Bool
    var process_pion_syn: Bool
    var process_pion_syn_emission: Bool
    var process_pion_syn_cooling: Bool
    var process_electron_compton: Bool
    var process_electron_compton_emission: Bool
    var process_electron_compton_cooling: Bool
    var process_compton_photon_energy_loss: Bool
    var process_proton_compton: Bool
    var process_proton_compton_emission: Bool
    var process_proton_compton_cooling: Bool
    var process_pion_compton: Bool
    var process_pion_compton_emission: Bool
    var process_pion_compton_cooling: Bool
    var process_muon_compton: Bool
    var process_muon_compton_emission: Bool
    var process_muon_compton_cooling: Bool
    var process_pion_decay: Bool
    var process_muon_decay: Bool
    var process_annihilation: Bool
    var process_annihilation_cooling: Bool
    var process_annihilation_emission: Bool
    var process_bethe_heitler: Bool
    var process_bethe_heitler_emission: Bool
    var process_bethe_heitler_cooling: Bool
    var process_photopion: Bool
    var process_photopion_emission: Bool
    var process_photopion_cooling: Bool
    var process_photopion_photon_loss: Bool
    var process_pp: Bool
    var process_pp_emission: Bool
    var process_pp_emission_pi0_into_cascade: Bool
    var process_pp_cooling: Bool
    var verbose: Bool

    var optimize_compton_emission_emin: Int
    var optimize_compton_target_emax: Int
    # declared `bool` in the C++ code: any non-zero value is stored as true (=1)
    var optimize_compton_emission_grid: Bool
    var optimize_compton_target_grid: Bool
    var optimize_annihilation_pair_emission: Bool
    var annihilation_pair_emission_max_com_energy: Int
    var optimize_bethe_heitler_outgoing_pairs_grid: Bool
    var optimize_bethe_heitler_incoming_protons_min: Int
    var optimize_bethe_heitler_target_photon_max: Int
    var optimize_photopion_target_photon_grid: Bool
    var optimize_photopion_target_photon_max: Int
    var pp_model_charged_pions: Int
    var pp_model_neutral_pions: Int
    var pp_transition_energy_to_delta_approx: Float64
    var step_counter: Int
    var solver_simpleqintegration: Bool
    var solver_threshold_cool_dom: Float64
    var solver_threshold_esc_dom: Float64
    var solver_threshold_matrix: Float64

    var S: List[SolverAlgorithm]

    # coefficient arrays, indexed by the C_* constants of simulation/coeffs.mojo
    var c: List[Vec]

    var profile_timing: Bool
    var timing: List[Float64]
    var nthreads: Int

    def __init__(out self):
        self.rp = RunParams()
        self.dat = AM3Arrays()
        self.oAcc = Acc(False)
        self.oPair = Pair(self.rp, self.dat, False)
        self.oSyn = Syn(self.rp, self.dat, False)
        self.oComp = Comp(self.dat, False)
        self.oSynP = SynP(self.rp, self.dat, False)
        self.oHadIC = HadIC(self.dat, False)
        self.oBH = BetheHeitler(self.dat, False)
        self.oPhotoPion = PhotoPion(self.dat, False)
        self.oProtonProton = ProtonProton(self.dat, False)
        self.oPionDec = PionDecay(self.dat, False)
        self.oMuonDec = MuonDecay(self.dat, False)
        self.oExpansion = Expansion(False)
        self.oEscape = Escape(self.dat, False)
        self.oInj = Injection(self.dat, False)
        self.oExtraCool = ExtraCooling(self.dat, False)
        self.nthreads = default_num_threads()
        self.oComp.nthreads = self.nthreads
        self.oBH.nthreads = self.nthreads
        self.oSyn.nthreads = self.nthreads
        self.oPair.nthreads = self.nthreads
        self.oPhotoPion.nthreads = self.nthreads

        self.solver_threshold_matrix = 10.0
        self.solver_threshold_cool_dom = 1e-2
        self.solver_threshold_esc_dom = 1e-2
        self.solver_simpleqintegration = False
        self.estimate_max_energies = False
        self.process_parse_sed = True
        self.process_hadronic = True
        self.process_merge_positrons_into_electrons = False
        self.process_escape = True
        self.process_expansion = False
        self.process_adiabatic_cooling = False
        self.process_electron_syn = True
        self.process_electron_syn_emission = True
        self.process_electron_syn_cooling = True
        self.process_quantum_syn = False
        self.process_quantum_syn_cooling = False
        self.process_proton_syn = True
        self.process_proton_syn_emission = True
        self.process_proton_syn_cooling = True
        self.process_muon_syn = True
        self.process_muon_syn_emission = True
        self.process_muon_syn_cooling = True
        self.process_pion_syn = True
        self.process_pion_syn_emission = True
        self.process_pion_syn_cooling = True
        self.process_ssa = True
        self.process_electron_compton = True
        self.process_electron_compton_emission = True
        self.process_electron_compton_cooling = True
        self.process_compton_photon_energy_loss = True
        self.process_proton_compton = True
        self.process_proton_compton_emission = True
        self.process_proton_compton_cooling = True
        self.process_muon_compton = True
        self.process_muon_compton_emission = True
        self.process_muon_compton_cooling = True
        self.process_pion_compton = True
        self.process_pion_compton_emission = True
        self.process_pion_compton_cooling = True
        self.optimize_compton_emission_emin = 100
        self.optimize_compton_emission_grid = True
        self.optimize_compton_target_emax = 280
        self.optimize_compton_target_grid = True
        self.process_annihilation = True
        self.process_annihilation_cooling = True
        self.process_annihilation_emission = True
        self.optimize_annihilation_pair_emission = True
        self.annihilation_pair_emission_max_com_energy = 138
        self.process_bethe_heitler = True
        self.process_bethe_heitler_cooling = True
        self.process_bethe_heitler_emission = True
        self.optimize_bethe_heitler_outgoing_pairs_grid = True
        self.optimize_bethe_heitler_incoming_protons_min = 70
        self.optimize_bethe_heitler_target_photon_max = 280
        self.process_photopion = True
        self.process_photopion_emission = True
        self.process_photopion_cooling = True
        self.process_photopion_photon_loss = True
        self.optimize_photopion_target_photon_grid = True
        self.optimize_photopion_target_photon_max = 280
        self.process_pp = True
        self.process_pp_emission = True
        self.process_pp_emission_pi0_into_cascade = False
        self.process_pp_cooling = True
        self.pp_transition_energy_to_delta_approx = 100.0
        self.pp_model_charged_pions = 0
        self.pp_model_neutral_pions = 0
        self.process_pion_decay = True
        self.process_muon_decay = True
        self.verbose = False

        self.c = List[Vec](capacity=C_COUNT)
        for _ in range(C_COUNT):
            self.c.append(Vec())
        self.S = List[SolverAlgorithm]()
        self.step_counter = 0
        self.profile_timing = False
        self.timing = zeros(36)
        self.Initialize_Solver_Algorithms()

    # ------------------------------------------------------------------
    def update_verbosity(mut self, v: Bool):
        if v:
            print("updated verbosity to ", 1 if v else 0, sep="")
        self.verbose = v
        self.oAcc.verbose = v
        self.oBH.verbose = v
        self.oComp.verbose = v
        self.oEscape.verbose = v
        self.oExpansion.verbose = v
        self.oHadIC.verbose = v
        self.oInj.verbose = v
        self.oMuonDec.verbose = v
        self.oPair.verbose = v
        self.oPhotoPion.verbose = v
        self.oProtonProton.verbose = v
        self.oPionDec.verbose = v
        self.oSyn.verbose = v
        self.oSynP.verbose = v
        for i in range(len(self.S)):
            self.S[i].verbose = v

    def init_kernels(mut self) raises:
        print("init. AM3 kernels:")
        self.print_process_switches()
        print()
        self.rp.print_params()

        self.oSyn.reinitialize_empty_kernels(self.dat)
        self.oComp.reinitialize_empty_kernels(self.dat)
        if self.process_electron_compton:
            # (the C++ "every 2nd grid point" options are unreachable: the
            #  optimize_*_grid switches are stored as C++ bools, i.e. step 1)
            self.oComp.initialize_kernels(
                self.dat,
                self.process_electron_compton_emission,
                self.process_compton_photon_energy_loss,
                self.optimize_compton_emission_emin,
                self.optimize_compton_target_emax,
            )
        self.oPair.reinitialize_empty_kernels(
            self.dat, self.optimize_annihilation_pair_emission
        )
        if self.process_annihilation:
            self.oPair.initialize_kernels(
                self.rp,
                self.dat,
                self.process_annihilation_cooling,
                self.process_annihilation_emission,
                self.optimize_annihilation_pair_emission,
            )
        if self.process_hadronic:
            self.oSynP.reinitialize_empty_kernels(self.dat)
            self.oHadIC.reinitialize_empty_kernels(self.dat, True, True, True)
            self.oHadIC.initialize_kernels(
                self.dat,
                self.process_proton_compton
                and self.process_proton_compton_emission,
                self.process_pion_compton
                and self.process_pion_compton_emission,
                self.process_muon_compton
                and self.process_muon_compton_emission,
            )
            self.oBH.reinitialize_empty_kernels(
                self.dat,
                1,
                Int(self.optimize_bethe_heitler_outgoing_pairs_grid),
                self.optimize_bethe_heitler_incoming_protons_min,
                self.optimize_bethe_heitler_target_photon_max,
            )
            if self.process_bethe_heitler:
                self.oBH.initialize_kernels(
                    self.dat,
                    Int(self.process_bethe_heitler_emission),
                    Int(self.process_bethe_heitler_cooling),
                    1,
                    Int(self.optimize_bethe_heitler_outgoing_pairs_grid),
                    self.optimize_bethe_heitler_incoming_protons_min,
                    self.optimize_bethe_heitler_target_photon_max,
                )
            self.oPionDec.reinitialize_empty_kernels(self.dat)
            self.oPionDec.initialize_kernels(self.dat)
            self.oMuonDec.reinitialize_empty_kernels(self.dat)
            self.oMuonDec.initialize_kernels(self.dat)
            if self.process_photopion:
                # (the "every n-th target photon" option is unreachable: the
                #  optimize_photopion_target_photon_grid switch is a C++ bool)
                self.oPhotoPion.reinitialize_empty_kernels(
                    self.dat, self.optimize_photopion_target_photon_max
                )
                self.oPhotoPion.initialize_kernels(
                    self.dat,
                    Int(self.process_photopion_emission),
                    Int(self.process_photopion_cooling),
                    Int(self.process_photopion_photon_loss),
                    self.optimize_photopion_target_photon_max,
                )
            if self.process_pp:
                self.oProtonProton.reinitialize_empty_kernels(self.dat)
                self.oProtonProton.initialize_kernels(
                    self.dat,
                    self.process_pp_emission_pi0_into_cascade,
                    self.pp_transition_energy_to_delta_approx,
                    self.pp_model_charged_pions,
                    self.pp_model_neutral_pions,
                )
        self.Update_Mag_Field(self.rp.mag_field)
        self.clear_am3()

    def ReInit_Physics_Modules(mut self):
        if self.verbose:
            print("Sim.Manager: ReInit_Physics_Modules")
        self.oSyn.clear_internal_arrays(self.dat)
        self.oSynP.clear_internal_arrays(self.dat)
        self.oComp.clear_internal_arrays(self.dat)
        self.oHadIC.clear_internal_arrays(self.dat)
        self.oPair.clear_internal_arrays(self.dat)
        self.oBH.clear_internal_arrays(self.dat)
        self.oPhotoPion.clear_internal_arrays(self.dat)
        self.oProtonProton.clear_internal_arrays(self.dat)
        self.oPionDec.clear_internal_arrays(self.dat)
        self.oMuonDec.clear_internal_arrays(self.dat)
        self.oEscape.clear_internal_arrays(self.dat)
        self.oInj.clear_internal_arrays(self.dat)
        self.oExtraCool.clear_internal_arrays(self.dat)
        for i in range(self.dat.nbins_pho):
            self.dat.n[P_photons_on_had_grid][i] = 0.0

    def clear_particle_densities(mut self):
        if self.verbose:
            print("Sim.Manager: clear_particle_densities")
        self.dat.clear_all_particle_arrays()
        self.step_counter = 0

    def clear_am3(mut self):
        if self.verbose:
            print("Sim.Manager: clear_particle_densities")
        self.clear_particle_densities()
        self.reset_coefficients()
        self.ReInit_Physics_Modules()
        self.Initialize_Solver_Algorithms()

    def _reset_coeffs(mut self, begin: Int, end: Int, n: Int):
        for id in range(begin, end):
            resize_zero(self.c[id], n)

    def reset_coefficients(mut self):
        """Resize every coefficient array to its grid and set it to zero."""
        if self.verbose:
            print("Sim.Manager: reset coefficient arrays")
        self._reset_coeffs(C_HAD_BEGIN, C_HAD_END, self.dat.nbins_had)
        self._reset_coeffs(C_NU_BEGIN, C_NU_END, self.dat.nbins_nu)
        self._reset_coeffs(C_LEP_BEGIN, C_LEP_END, self.dat.nbins_lep)
        self._reset_coeffs(C_PHO_BEGIN, C_PHO_END, self.dat.nbins_pho)
        resize_zero(self.dat.n[P_photons_on_had_grid], self.dat.nbins_pho)

    def print_process_switches(self):
        # Switches are shown as True/False (the C++ original prints 1/0).
        @always_inline
        def b(x: Bool) -> String:
            return "True" if x else "False"

        print(
            "AM3 has the following switches (at step: ",
            self.step_counter,
            ")",
            sep="",
        )
        print(
            "estimate maximum energies: ", b(self.estimate_max_energies), sep=""
        )
        print("parse sed components: ", b(self.process_parse_sed), sep="")
        print("escape: ", b(self.process_escape), sep="")
        print("expansion: ", b(self.process_expansion), sep="")
        print("adiabatic: ", b(self.process_adiabatic_cooling), sep="")
        print(
            "synchrotron:\n  e+/-: ",
            b(self.process_electron_syn),
            "  (em.: ",
            b(self.process_electron_syn_emission),
            ", cool.: ",
            b(self.process_electron_syn_cooling),
            ")",
            sep="",
        )
        print(
            "  protons: ",
            b(self.process_proton_syn),
            "  (em.: ",
            b(self.process_proton_syn_emission),
            ", cool.: ",
            b(self.process_proton_syn_cooling),
            ")",
            sep="",
        )
        print(
            "  pions: ",
            b(self.process_pion_syn),
            "  (em.: ",
            b(self.process_pion_syn_emission),
            ", cool.: ",
            b(self.process_pion_syn_cooling),
            ")",
            sep="",
        )
        print(
            "  muons: ",
            b(self.process_muon_syn),
            "  (em.: ",
            b(self.process_muon_syn_emission),
            ", cool.: ",
            b(self.process_muon_syn_cooling),
            ")",
            sep="",
        )
        print("  syn-self-abs.: ", b(self.process_ssa), sep="")
        print(
            "  quantum-syn.: (em. (e+/-) ",
            b(self.process_quantum_syn),
            ", cool. (all charged): ",
            b(self.process_quantum_syn_cooling),
            ")",
            sep="",
        )
        print(
            "inv. Compton:\n  e+/-: ",
            b(self.process_electron_compton),
            "  (em. : ",
            b(self.process_electron_compton_emission),
            ", cool.: ",
            b(self.process_electron_compton_cooling),
            " (continuous))",
            sep="",
        )
        print(
            "  photon loss due to upscattering: ",
            b(self.process_compton_photon_energy_loss),
            sep="",
        )
        print(
            "  protons: ",
            b(self.process_proton_compton),
            "  (em. (step approx.): ",
            b(self.process_proton_compton_emission),
            ", no cooling)",
            sep="",
        )
        print(
            "  pions: ",
            b(self.process_pion_compton),
            "  (em. (step approx.): ",
            b(self.process_pion_compton_emission),
            ", cont. cool.: ",
            b(self.process_pion_compton_cooling),
            ")",
            sep="",
        )
        print(
            "  muons: ",
            b(self.process_muon_compton),
            "  (em. (step approx.): ",
            b(self.process_muon_compton_emission),
            ", cont. cool.: ",
            b(self.process_muon_compton_cooling),
            ")",
            sep="",
        )
        var opt = (
            "(opt. 14-bin kernel)" if self.optimize_annihilation_pair_emission else "(full kernel)"
        )
        print(
            "pair prod. (gamma+gamma->e+e): ",
            b(self.process_annihilation),
            "  (photon loss.: ",
            b(self.process_annihilation_cooling),
            ", e+/- source (feedback): ",
            b(self.process_annihilation_emission),
            " ",
            opt,
            ")",
            sep="",
        )
        print(
            "'hadronic' processes (below): ", b(self.process_hadronic), sep=""
        )
        print(
            " Pion decay: ",
            b(self.process_pion_decay),
            " Muon decay: ",
            b(self.process_muon_decay),
            ")",
            sep="",
        )
        print(
            "proton Bethe-Heitler: ",
            b(self.process_bethe_heitler),
            "  (em.: ",
            b(self.process_bethe_heitler_emission),
            ", cool.: ",
            b(self.process_bethe_heitler_cooling),
            ")",
            sep="",
        )
        print(
            "proton photo-pion: ",
            b(self.process_photopion),
            "  (em.: ",
            b(self.process_photopion_emission),
            ", cool.: ",
            b(self.process_photopion_cooling),
            ", photon loss: ",
            b(self.process_photopion_photon_loss),
            ")",
            sep="",
        )
        print(
            "proton p-p: ",
            b(self.process_pp),
            "  (em.: ",
            b(self.process_pp_emission),
            " , cool.: ",
            b(self.process_pp_cooling),
            ")",
            sep="",
        )

    # ------------------------------------------------------------------
    # solver algorithms
    # ------------------------------------------------------------------
    def _mk(
        self,
        name: String,
        NB: Int,
        PGO: Int,
        XUG: Float64,
        LUC: Float64,
        TUS: Float64,
        HLS: Float64,
        MR: Float64,
        Dt: Float64,
        sy: Bool,
        qsc: Bool,
        ic: Bool,
        dec: Bool,
    ) -> SolverAlgorithm:
        return SolverAlgorithm(
            name,
            NB,
            PGO,
            self.dat.dlnE,
            XUG,
            LUC,
            TUS,
            HLS,
            MR,
            Dt,
            sy,
            qsc,
            ic,
            dec,
            self.solver_threshold_cool_dom,
            self.solver_threshold_esc_dom,
            self.solver_threshold_matrix,
            self.solver_simpleqintegration,
            self.verbose,
        )

    def Initialize_Solver_Algorithms(mut self):
        if self.verbose:
            print("Initialize_Solver_Algorithms")
        var t_nodecay_s = 1.0e7
        var t_neutron_decay_s = 8.795e2
        var t_pipm_decay_s = 2.6033e-8
        var t_muon_decay_s = 2.1969811e-6
        var mass_nu = 1e-100
        var mass_g = 1e-100
        var dt_sec = self.rp.dt
        var dt_SSC = dt_sec / T_SSC_CGS
        var nh = self.dat.nbins_had
        var nn = self.dat.nbins_nu
        var nl = self.dat.nbins_lep
        var ng = self.dat.nbins_pho
        var pgo_h = -self.dat.ind_min_energy_pho_hadgrid
        var pgo_l = -self.dat.ind_min_energy_pho
        var qsc = self.process_quantum_syn_cooling
        var p_sy = self.process_proton_syn and self.process_proton_syn_cooling
        var p_ic = (
            self.process_proton_compton and self.process_proton_compton_cooling
        )
        var pi_sy = self.process_pion_syn and self.process_pion_syn_cooling
        var pi_ic = (
            self.process_pion_compton and self.process_pion_compton_cooling
        )
        var mu_sy = self.process_muon_syn and self.process_muon_syn_cooling
        var mu_ic = (
            self.process_muon_compton and self.process_muon_compton_cooling
        )
        var e_sy = (
            self.process_electron_syn and self.process_electron_syn_cooling
        )
        var e_ic = (
            self.process_electron_compton
            and self.process_electron_compton_cooling
        )
        var pid = self.process_pion_decay
        var mud = self.process_muon_decay

        var S = List[SolverAlgorithm](capacity=N_SOLVERS)
        S.append(
            self._mk(
                "proton",
                nh,
                pgo_h,
                1.0,
                1.0,
                1.0,
                t_nodecay_s,
                massp / masse,
                dt_sec,
                p_sy,
                qsc,
                p_ic,
                False,
            )
        )
        S.append(
            self._mk(
                "neutron",
                nh,
                pgo_h,
                1.0,
                1.0,
                1.0,
                t_neutron_decay_s,
                massn / masse,
                dt_sec,
                False,
                False,
                False,
                False,
            )
        )
        var pion_names = List[String]()
        pion_names.append("pion_plus")
        pion_names.append("pion_minus")
        pion_names.append("pgamma_produced_pion_plus")
        pion_names.append("pgamma_produced_pion_minus")
        pion_names.append("pp_produced_pion_plus")
        pion_names.append("pp_produced_pion_minus")
        for k in range(6):
            var mr = (masspip if k % 2 == 0 else masspim) / masse
            S.append(
                self._mk(
                    pion_names[k],
                    nh,
                    pgo_h,
                    1.0,
                    1.0,
                    1.0,
                    t_pipm_decay_s,
                    mr,
                    dt_sec,
                    pi_sy,
                    qsc,
                    pi_ic,
                    pid,
                )
            )
        var mu_names = List[String]()
        for pre in ["", "pgamma_produced_", "pp_produced_"]:
            for n in [
                "muon_plus_left",
                "muon_plus_right",
                "muon_minus_left",
                "muon_minus_right",
            ]:
                mu_names.append(String(pre) + String(n))
        for k in range(12):
            S.append(
                self._mk(
                    mu_names[k],
                    nh,
                    pgo_h,
                    1.0,
                    1.0,
                    1.0,
                    t_muon_decay_s,
                    massmu / masse,
                    dt_sec,
                    mu_sy,
                    qsc,
                    mu_ic,
                    mud,
                )
            )
        var nu_names = List[String]()
        for pre in ["", "pgamma_produced_", "pp_produced_"]:
            for n in [
                "electron_neutrino",
                "electron_anti_neutrino",
                "muon_neutrino",
                "muon_anti_neutrino",
            ]:
                nu_names.append(String(pre) + String(n))
        for k in range(12):
            S.append(
                self._mk(
                    nu_names[k],
                    nn,
                    pgo_h,
                    1.0,
                    1.0,
                    1.0,
                    t_nodecay_s,
                    mass_nu / masse,
                    dt_sec,
                    False,
                    False,
                    False,
                    False,
                )
            )
        var lep_names = List[String]()
        for n in [
            "electron",
            "positron",
            "electron+positron",
            "injected_pairs",
            "photon_annihilated_pairs",
            "bethe_heitler_pairs",
            "pgamma_produced_pairs",
            "pp_produced_pairs",
        ]:
            lep_names.append(String(n))
        for k in range(8):
            S.append(
                self._mk(
                    lep_names[k],
                    nl,
                    pgo_l,
                    masse,
                    L_SSC_CGS,
                    T_SSC_CGS,
                    t_nodecay_s,
                    1.0,
                    dt_SSC,
                    e_sy,
                    qsc,
                    e_ic,
                    False,
                )
            )
        var g_names = List[String]()
        for n in [
            "photon",
            "photons_without_ext_inj",
            "photon_injected_syn",
            "photon_injected_IC",
            "photon_pair-produced_syn",
            "photon_pair-produced_IC",
            "photon_BH-produced_syn",
            "photon_BH-produced_IC",
            "photon_p-gamma-produced_syn",
            "photon_p-gamma-produced_IC",
            "photon_p-p-produced_sy",
            "photon_p-p-produced_IC",
            "photon_proton_syn",
            "photon_proton_IC",
            "photon_pion_syn",
            "photon_pion_IC",
            "photon_muon_syn",
            "photon_muon_IC",
            "photon_pi0_pg",
            "photon_pi0_pp",
            "photon_injected",
        ]:
            g_names.append(String(n))
        for k in range(21):
            S.append(
                self._mk(
                    g_names[k],
                    ng,
                    0,
                    mass_g,
                    L_SSC_CGS,
                    T_SSC_CGS,
                    t_nodecay_s,
                    1.0,
                    dt_SSC,
                    False,
                    False,
                    False,
                    False,
                )
            )
        self.S = S^

    # ------------------------------------------------------------------
    # species evolution
    # ------------------------------------------------------------------
    def _ensure_ic_kernel(mut self, k: Int) raises:
        """IC cooling kernels are identical for solvers with the same grid,
        units and mass: compute once and copy."""
        if not self.S[k].process_ic or self.S[k].ker_status != 0:
            return
        for o in range(len(self.S)):
            if (
                o != k
                and self.S[o].ker_status == 1
                and self.S[k].same_ic_kernel(self.S[o])
            ):
                self.S[k].ker_ic_t = self.S[o].ker_ic_t.copy()
                self.S[k].ker_status = 1
                return
        self.S[k].compute_ic_ker()

    def evolve_group(mut self, rows: List[Evolution]) raises:
        """Advance a group of independent species by one time step.

        Cooling rates depend only on the species and the target photons, so
        they are computed by the first solver of each configuration and
        shared with the others; the solves then run in parallel."""
        var b = self.rp.mag_field
        var distinct = True
        for r in range(len(rows)):
            var k = rows[r].solver
            self.S[k].updateDeltaT(self.rp.dt)
            var lead = -1
            for o in range(r):
                if rows[o].solver == k or rows[o].target == rows[r].target:
                    distinct = False
                if (
                    lead < 0
                    and rows[o].nphot == rows[r].nphot
                    and self.S[rows[o].solver].same_cooling(self.S[k])
                ):
                    lead = rows[o].solver
            if lead < 0:
                self._ensure_ic_kernel(k)
                self.S[k].update_cooling(b, self.dat.n[rows[r].nphot])
            elif lead != k:
                var asy = self.S[lead].Asy.copy()
                var aic = self.S[lead].Aic.copy()
                self.S[k].share_cooling(asy, aic, self.S[lead].cooling_on)
        var jobs = List[EvolveJob](capacity=len(rows))
        for r in range(len(rows)):
            ref R = rows[r]
            jobs.append(
                EvolveJob(
                    Pointer(to=self.S[R.solver]).unsafe_origin_cast[
                        MutUntrackedOrigin
                    ](),
                    vp(self.c[R.a]),
                    vp(self.c[R.A]),
                    vp(self.c[R.q]),
                    vp(self.dat.n[R.target]),
                )
            )
        run_jobs(jobs, self.nthreads if distinct else 1)

    def _block_rows(
        self,
        mut rows: List[Evolution],
        variant: Int,
        solver0: Int,
        target0: Int,
        q0: Int,
        a: Int,
        A: Int,
    ):
        """Four rows for the four states of a species (muons or neutrinos) of
        one variant (0 all, 1 p-gamma, 2 p-p): solver, density and source ids of
        a variant are consecutive blocks of four."""
        var o = 4 * variant
        for k in range(4):
            rows.append(
                Evolution(
                    solver0 + o + k,
                    target0 + o + k,
                    a,
                    A,
                    q0 + o + k,
                    P_photons_on_had_grid,
                )
            )

    # ------------------------------------------------------------------
    # evolve one step
    # ------------------------------------------------------------------
    @always_inline
    def _tic(self) -> Int:
        return perf_counter_ns()

    @always_inline
    def _toc(mut self, idx: Int, t0: Int):
        if self.profile_timing:
            self.timing[idx] = Float64(perf_counter_ns() - t0)

    def evolve_step(mut self) raises:
        if self.verbose:
            print("evolve for 1 step")
        if self.estimate_max_energies and self.step_counter > 0:
            self.estimate_maximum_lepton_energy()
            self.estimate_maximum_proton_energy()
            self.rp.powerlaw_injection_electrons_emax = (
                self.rp.estimated_electron_emax
            )
            self.rp.powerlaw_injection_protons_emax = (
                self.rp.estimated_proton_emax
            )
            self.oInj.Calc_IN_Electron_powerlaw(self.rp, self.dat)
            self.oInj.Calc_IN_Proton_powerlaw(self.rp, self.dat)
        var t0 = self._tic()
        if self.process_hadronic:
            t0 = self._tic()
            self.Pass_LepG2HadG()
            self._toc(0, t0)
            t0 = self._tic()
            self.Calc_Coeff_proton_NTR()
            self._toc(1, t0)
            t0 = self._tic()
            self.Pass_Coeff_PRO()
            self._toc(2, t0)
            t0 = self._tic()
            self.Calc_PRO()
            self._toc(3, t0)
            t0 = self._tic()
            self.Pass_Coeff_NTR()
            self._toc(4, t0)
            t0 = self._tic()
            self.Calc_NTR()
            self._toc(5, t0)
            t0 = self._tic()
            self.Calc_Coeff_PI()
            self._toc(6, t0)
            t0 = self._tic()
            self.Pass_Coeff_PI()
            self._toc(7, t0)
            t0 = self._tic()
            self.Calc_PI()
            self._toc(8, t0)
            t0 = self._tic()
            self.Calc_Coeff_MU()
            self._toc(9, t0)
            t0 = self._tic()
            self.Pass_Coeff_MU()
            self._toc(10, t0)
            t0 = self._tic()
            self.Calc_MU()
            self._toc(11, t0)
            t0 = self._tic()
            self.Calc_Coeff_NU()
            self._toc(12, t0)
            t0 = self._tic()
            self.Pass_Coeff_NU()
            self._toc(13, t0)
            t0 = self._tic()
            self.Calc_NU()
            self._toc(14, t0)
        t0 = self._tic()
        self.Calc_Coeff_Lep()
        self._toc(15, t0)
        t0 = self._tic()
        self.Pass_Coeff_Lep()
        self._toc(16, t0)
        t0 = self._tic()
        self.Calc_Lep()
        self._toc(17, t0)
        t0 = self._tic()
        self.Calc_Coeff_Phot()
        self._toc(18, t0)
        t0 = self._tic()
        self.Pass_Coeff_Phot()
        self._toc(19, t0)
        t0 = self._tic()
        self.Calc_Phot()
        self._toc(20, t0)
        self.step_counter += 1

    def Pass_LepG2HadG(mut self):
        if self.verbose:
            print("Pass photons to had. grid")
        var ph = pd(self.dat.n[P_photons])
        for i in range(self.dat.nbins_pho):
            self.dat.n[P_photons_on_had_grid][
                i
            ] = n_SSC_CGS * self.dat.grid_lep2had(i, ph, 0, self.dat.nbins_pho)

    def Calc_Coeff_proton_NTR(mut self):
        if self.verbose:
            print(" Calculate proton + neutron coefficient")
        if self.process_bethe_heitler and self.process_bethe_heitler_cooling:
            self.oBH.Calc_BH_Cool(self.dat)
        if self.process_photopion and self.process_photopion_cooling:
            self.oPhotoPion.Calc_PhotoPion_PhotonConvolution(self.dat)
            self.oPhotoPion.Calc_PhotoPion_ProtonNeutronReinj(self.dat)
            self.oPhotoPion.Calc_PhotoPion_ProtonNeutronLoss(self.dat)
        if self.process_pp and self.process_pp_cooling:
            self.oProtonProton.Calc_PP_ProtonLoss(self.rp, self.dat)

    def Pass_Coeff_PRO(mut self):
        if self.verbose:
            print("Pass proton coefficient")
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiHad(
            self.rp
        )
        var bh = fbool(
            self.process_bethe_heitler and self.process_bethe_heitler_cooling
        )
        var ppc = fbool(self.process_pp and self.process_pp_cooling)
        var pgc = fbool(
            self.process_photopion and self.process_photopion_cooling
        )
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        for i in range(self.dat.nbins_had):
            self.c[C_Apro][i] = (
                adi
                + bh * self.oBH.ABHHadP[i]
                + ppc * self.oProtonProton.APP_PRO[i]
            ) + self.oExtraCool.AExtraCoolPRO[i]
            self.c[C_apro][i] = (
                pgc * self.oPhotoPion.afPG_PRO[i]
                + esc * self.oEscape.afescPRO[i]
                + exh
            )
            self.c[C_qpro][i] = (
                pgc * self.oPhotoPion.QPG_PROGen[i] + self.oInj.inj_proton[i]
            )

    def Calc_PRO(mut self) raises:
        if self.verbose:
            print(" Calculate protons")
        var rows = List[Evolution]()
        rows.append(
            Evolution(
                SA_proton,
                P_protons,
                C_apro,
                C_Apro,
                C_qpro,
                P_photons_on_had_grid,
            )
        )
        self.evolve_group(rows)

    def Pass_Coeff_NTR(mut self):
        if self.verbose:
            print("Pass neutrons")
        var pgc = fbool(
            self.process_photopion and self.process_photopion_cooling
        )
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        for i in range(self.dat.nbins_had):
            self.c[C_Antr][i] = self.oExtraCool.AExtraCoolNTR[i]
            self.c[C_antr][i] = (
                pgc * self.oPhotoPion.afPG_NTR[i]
                + esc * self.oEscape.afescNTR[i]
                + exh
            )
            self.c[C_qntr][i] = (
                pgc * self.oPhotoPion.QPG_NTRGen[i] + self.oInj.inj_neutron[i]
            )

    def Calc_NTR(mut self) raises:
        if self.verbose:
            print(" Calculate neutrons")
        var rows = List[Evolution]()
        rows.append(
            Evolution(
                SA_neutron,
                P_neutrons,
                C_antr,
                C_Antr,
                C_qntr,
                P_photons_on_had_grid,
            )
        )
        self.evolve_group(rows)

    def Calc_Coeff_PI(mut self):
        if self.verbose:
            print(" Calculate pion coefficient")
        if self.process_photopion and self.process_photopion_emission:
            if not (self.process_photopion and self.process_photopion_cooling):
                self.oPhotoPion.Calc_PhotoPion_PhotonConvolution(self.dat)
            self.oPhotoPion.Calc_PhotoPion_Rad(self.dat)
        if self.process_pp and self.process_pp_emission:
            self.oProtonProton.Calc_PP_PionSrc(
                self.rp, self.dat, self.process_pp_emission_pi0_into_cascade
            )

    def Pass_Coeff_PI(mut self):
        if self.verbose:
            print("Pass pion coefficient")
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiHad(
            self.rp
        )
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        var pge = fbool(
            self.process_photopion and self.process_photopion_emission
        )
        var ppe = fbool(self.process_pp and self.process_pp_emission)
        var ppe0 = fbool(
            self.process_pp
            and self.process_pp_emission
            and self.process_pp_emission_pi0_into_cascade
        )
        for i in range(self.dat.nbins_had):
            self.c[C_Apip][i] = adi + self.oExtraCool.AExtraCoolPipm[i]
            self.c[C_Apim][i] = adi + self.oExtraCool.AExtraCoolPipm[i]
            self.c[C_apip][i] = esc * self.oEscape.afescPipm[i] + exh
            self.c[C_apim][i] = esc * self.oEscape.afescPipm[i] + exh
            self.c[C_qpip_pg][i] = pge * self.oPhotoPion.qpip[i]
            self.c[C_qpip_pp][i] = ppe * self.oProtonProton.qpip[i]
            self.c[C_qpip][i] = (
                self.c[C_qpip_pg][i]
                + self.c[C_qpip_pp][i]
                + self.oInj.inj_piplus[i]
            )
            self.c[C_qpim_pg][i] = pge * self.oPhotoPion.qpim[i]
            self.c[C_qpim_pp][i] = ppe * self.oProtonProton.qpim[i]
            self.c[C_qpim][i] = (
                self.c[C_qpim_pg][i]
                + self.c[C_qpim_pp][i]
                + self.oInj.inj_piminus[i]
            )
            self.c[C_qpi0_pg][i] = pge * self.oPhotoPion.qpi0[i]
            self.c[C_qpi0_pp][i] = ppe0 * self.oProtonProton.qpi0[i]
            self.c[C_qpi0][i] = (
                self.c[C_qpi0_pg][i]
                + self.c[C_qpi0_pp][i]
                + self.oInj.inj_pi0[i]
            )

    def Calc_PI(mut self) raises:
        if self.verbose:
            print(" Calculate pions")
        # The C++ code evolves every pion component with the pi+ solver; the
        # pi- and component solvers are configured identically (same mass,
        # decay and cooling), so using them gives the same result in parallel.
        var rows = List[Evolution]()
        rows.append(
            Evolution(
                SA_Pip,
                P_pion_plus,
                C_apip,
                C_Apip,
                C_qpip,
                P_photons_on_had_grid,
            )
        )
        rows.append(
            Evolution(
                SA_Pim,
                P_pion_minus,
                C_apim,
                C_Apim,
                C_qpim,
                P_photons_on_had_grid,
            )
        )
        if (
            self.process_parse_sed
            and self.process_photopion
            and self.process_photopion_emission
        ):
            rows.append(
                Evolution(
                    SA_Pip_pg,
                    P_pion_plus_pg,
                    C_apip,
                    C_Apip,
                    C_qpip_pg,
                    P_photons_on_had_grid,
                )
            )
            rows.append(
                Evolution(
                    SA_Pim_pg,
                    P_pion_minus_pg,
                    C_apim,
                    C_Apim,
                    C_qpim_pg,
                    P_photons_on_had_grid,
                )
            )
        if (
            self.process_parse_sed
            and self.process_pp
            and self.process_pp_emission
        ):
            rows.append(
                Evolution(
                    SA_Pip_pp,
                    P_pion_plus_pp,
                    C_apip,
                    C_Apip,
                    C_qpip_pp,
                    P_photons_on_had_grid,
                )
            )
            rows.append(
                Evolution(
                    SA_Pim_pp,
                    P_pion_minus_pp,
                    C_apim,
                    C_Apim,
                    C_qpim_pp,
                    P_photons_on_had_grid,
                )
            )
        self.evolve_group(rows)

    def _pion_to_muon_sources(mut self, pip: Int, pim: Int, q0: Int):
        """Muon sources from the decay of the pions n[pip], n[pim] into the
        block c[q0 .. q0+3] = (mu+ left, mu+ right, mu- left, mu- right)."""
        self.oPionDec.Calc_PItoMU(self.dat, self.dat.n[pip], self.dat.n[pim])
        for i in range(self.dat.nbins_had):
            self.c[q0][i] = self.oPionDec.MpLGen[i]
            self.c[q0 + 1][i] = self.oPionDec.MpRGen[i]
            self.c[q0 + 2][i] = self.oPionDec.MmLGen[i]
            self.c[q0 + 3][i] = self.oPionDec.MmRGen[i]

    def Calc_Coeff_MU(mut self):
        if self.verbose:
            print(" Calculate muon coefficient")
        if not self.process_pion_decay:
            return
        self._pion_to_muon_sources(P_pion_plus, P_pion_minus, C_qmupl_tot_dec)
        if (
            self.process_parse_sed
            and self.process_photopion
            and self.process_photopion_emission
        ):
            self._pion_to_muon_sources(
                P_pion_plus_pg, P_pion_minus_pg, C_qmupl_pg
            )
        if (
            self.process_parse_sed
            and self.process_pp
            and self.process_pp_emission
        ):
            self._pion_to_muon_sources(
                P_pion_plus_pp, P_pion_minus_pp, C_qmupl_pp
            )

    def Pass_Coeff_MU(mut self):
        if self.verbose:
            print("Pass muon coefficient")
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiHad(
            self.rp
        )
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        for i in range(self.dat.nbins_had):
            self.c[C_Amu][i] = adi + self.oExtraCool.AExtraCoolMu[i]
            self.c[C_amu][i] = esc * self.oEscape.afescMu[i] + exh
            # (injection arrays paired exactly as in the C++ code)
            self.c[C_qmuml][i] = (
                self.oInj.inj_muminusleft[i] + self.c[C_qmuml_tot_dec][i]
            )
            self.c[C_qmupl][i] = (
                self.oInj.inj_muminusright[i] + self.c[C_qmupl_tot_dec][i]
            )
            self.c[C_qmumr][i] = (
                self.oInj.inj_muplusleft[i] + self.c[C_qmumr_tot_dec][i]
            )
            self.c[C_qmupr][i] = (
                self.oInj.inj_muplusright[i] + self.c[C_qmupr_tot_dec][i]
            )

    def Calc_MU(mut self) raises:
        if self.verbose:
            print(" Calculate muons")
        comptime assert SA_MpL_pg == SA_MpL + 4 and SA_MpL_pp == SA_MpL + 8
        comptime assert (
            P_muon_plus_left_pg == P_muon_plus_left + 4
            and C_qmupl_pg == C_qmupl + 4
        )
        var rows = List[Evolution]()
        self._block_rows(
            rows, 0, SA_MpL, P_muon_plus_left, C_qmupl, C_amu, C_Amu
        )
        if (
            self.process_parse_sed
            and self.process_photopion
            and self.process_photopion_emission
        ):
            self._block_rows(
                rows, 1, SA_MpL, P_muon_plus_left, C_qmupl, C_amu, C_Amu
            )
        if (
            self.process_parse_sed
            and self.process_pp
            and self.process_pp_emission
        ):
            self._block_rows(
                rows, 2, SA_MpL, P_muon_plus_left, C_qmupl, C_amu, C_Amu
            )
        self.evolve_group(rows)

    def _decay_neutrino_sources(mut self, pip: Int, mu0: Int, q0: Int):
        """Neutrino sources from the decay of the pions n[pip], n[pip + 1] and
        the muons n[mu0 .. mu0+3] into c[q0 .. q0+3] = (nu_e, anti-nu_e, nu_mu,
        anti-nu_mu). (For pion decay off and muon decay on, the C++ code never
        resets the total nu_mu sources and adds to them every step; here they
        are recomputed.)"""
        var pid = fbool(self.process_pion_decay)
        var mud = fbool(self.process_muon_decay)
        if self.process_pion_decay:
            self.oPionDec.Calc_PItoNU(
                self.dat, self.dat.n[pip], self.dat.n[pip + 1]
            )
        if self.process_muon_decay:
            self.oMuonDec.Calc_MUtoNU(
                self.dat,
                self.dat.n[mu0 + 2],
                self.dat.n[mu0],
                self.dat.n[mu0 + 3],
                self.dat.n[mu0 + 1],
            )
        for i in range(self.dat.nbins_nu):
            self.c[q0][i] = mud * self.oMuonDec.NUeGen_mu[i]
            self.c[q0 + 1][i] = mud * self.oMuonDec.NUeAGen_mu[i]
            self.c[q0 + 2][i] = (
                pid * self.oPionDec.NUmGen_pi[i]
                + mud * self.oMuonDec.NUmGen_mu[i]
            )
            self.c[q0 + 3][i] = (
                pid * self.oPionDec.NUmAGen_pi[i]
                + mud * self.oMuonDec.NUmAGen_mu[i]
            )

    def Calc_Coeff_NU(mut self):
        if self.verbose:
            print(" Calculate neutrino coefficient")
        comptime assert (
            P_pion_minus == P_pion_plus + 1
            and P_muon_plus_right == P_muon_plus_left + 1
        )
        comptime assert (
            P_muon_minus_left == P_muon_plus_left + 2
            and P_muon_minus_right == P_muon_plus_left + 3
        )
        self._decay_neutrino_sources(
            P_pion_plus, P_muon_plus_left, C_qnue_tot_dec
        )
        if (
            self.process_parse_sed
            and self.process_photopion
            and self.process_photopion_emission
        ):
            self._decay_neutrino_sources(
                P_pion_plus_pg, P_muon_plus_left_pg, C_qnue_pg
            )
        if (
            self.process_parse_sed
            and self.process_pp
            and self.process_pp_emission
        ):
            self._decay_neutrino_sources(
                P_pion_plus_pp, P_muon_plus_left_pp, C_qnue_pp
            )

    def Pass_Coeff_NU(mut self):
        if self.verbose:
            print("Pass neutrino coefficient")
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        for i in range(self.dat.nbins_nu):
            self.c[C_Anu][i] = self.oExtraCool.AExtraCoolNU[i]
            self.c[C_anu][i] = esc * self.oEscape.afescNU[i] + exh
            # (all four flavours use inj_nue, as in the C++ code)
            self.c[C_qnue][i] = self.c[C_qnue_tot_dec][i] + self.oInj.inj_nue[i]
            self.c[C_qnuea][i] = (
                self.c[C_qnuea_tot_dec][i] + self.oInj.inj_nue[i]
            )
            self.c[C_qnum][i] = self.c[C_qnum_tot_dec][i] + self.oInj.inj_nue[i]
            self.c[C_qnuma][i] = (
                self.c[C_qnuma_tot_dec][i] + self.oInj.inj_nue[i]
            )

    def Calc_NU(mut self) raises:
        if self.verbose:
            print(" Calculate neutrinos")
        comptime assert SA_nue_pg == SA_nue + 4 and SA_nue_pp == SA_nue + 8
        comptime assert (
            P_neutrino_el_pg == P_neutrino_el + 4 and C_qnue_pg == C_qnue + 4
        )
        var rows = List[Evolution]()
        self._block_rows(rows, 0, SA_nue, P_neutrino_el, C_qnue, C_anu, C_Anu)
        if (
            self.process_parse_sed
            and self.process_photopion
            and self.process_photopion_emission
        ):
            self._block_rows(
                rows, 1, SA_nue, P_neutrino_el, C_qnue, C_anu, C_Anu
            )
        if (
            self.process_parse_sed
            and self.process_pp
            and self.process_pp_emission
        ):
            self._block_rows(
                rows, 2, SA_nue, P_neutrino_el, C_qnue, C_anu, C_Anu
            )
        self.evolve_group(rows)

    def Calc_Coeff_Lep(mut self):
        if self.verbose:
            print(" Calculate electron coefficient")
        if self.process_annihilation and self.process_annihilation_emission:
            if self.optimize_annihilation_pair_emission:
                self.oPair.Calc_Pair_Lep_OPT(self.dat)
            else:
                self.oPair.Calc_Pair_Lep(
                    self.dat, self.annihilation_pair_emission_max_com_energy
                )
        if self.process_hadronic:
            if (
                self.process_bethe_heitler
                and self.process_bethe_heitler_emission
            ):
                self.oBH.Calc_BH_Rad(
                    self.dat,
                    Int(self.optimize_bethe_heitler_outgoing_pairs_grid),
                    self.optimize_bethe_heitler_incoming_protons_min,
                    self.optimize_bethe_heitler_target_photon_max,
                )
            var both_decays = (
                self.process_pion_decay and self.process_muon_decay
            )
            if self.process_parse_sed:
                if self.verbose:
                    print("computing muon decay to pairs terms for components")
                if (
                    self.process_photopion
                    and self.process_photopion_emission
                    and both_decays
                ):
                    self._muon_to_pair_sources(
                        P_muon_plus_left_pg, C_qepg, C_qppg, 1.0
                    )
                if self.process_pp and self.process_pp_emission and both_decays:
                    self._muon_to_pair_sources(
                        P_muon_plus_left_pp, C_qepp, C_qppp, 1.0
                    )
            else:
                if self.verbose:
                    print("computing muon decay to pairs terms total")
                self._muon_to_pair_sources(
                    P_muon_plus_left,
                    C_qe_tot_dec,
                    C_qp_tot_dec,
                    fbool(self.process_muon_decay),
                )

    def _muon_to_pair_sources(
        mut self, mu0: Int, qe: Int, qp: Int, factor: Float64
    ):
        """e-/e+ sources on the lepton grid from the decay of the muons
        n[mu0 .. mu0+3] (skipped, leaving the last result, if muon decay is off).
        """
        if self.process_muon_decay:
            self.oMuonDec.Calc_MUtoE(
                self.dat,
                self.dat.n[mu0 + 2],
                self.dat.n[mu0],
                self.dat.n[mu0 + 3],
                self.dat.n[mu0 + 1],
            )
        var nl = self.dat.nbins_lep
        var e = pd(self.oMuonDec.QMU2EGen_mu)
        var p = pd(self.oMuonDec.QMU2PGen_mu)
        for i in range(nl):
            self.c[qe][i] = (
                factor / Rate_SSC_CGS * self.dat.grid_had2lep(i, e, 0, nl)
            )
            self.c[qp][i] = (
                factor / Rate_SSC_CGS * self.dat.grid_had2lep(i, p, 0, nl)
            )

    def Pass_Coeff_Lep(mut self):
        if self.verbose:
            print("Pass electron/positron coefficient")
        var nl = self.dat.nbins_lep
        var bh_on = (
            self.process_hadronic
            and self.process_bethe_heitler
            and self.process_bethe_heitler_emission
        )
        var pg_on = (
            self.process_hadronic
            and self.process_photopion
            and self.process_photopion_emission
            and self.process_pion_decay
            and self.process_muon_decay
        )
        var pp_on = (
            self.process_hadronic
            and self.process_pp
            and self.process_pp_emission
            and self.process_pion_decay
            and self.process_muon_decay
        )
        var qbh = pd(self.oBH.QBH_LGen)
        for i in range(nl):
            if bh_on:
                self.c[C_qebh][i] = (
                    1.0 / Rate_SSC_CGS * self.dat.grid_had2lep(i, qbh, 0, nl)
                )
                self.c[C_qpbh][i] = (
                    1.0 / Rate_SSC_CGS * self.dat.grid_had2lep(i, qbh, 0, nl)
                )
            else:
                self.c[C_qebh][i] = 0.0
                self.c[C_qpbh][i] = 0.0
            if not pg_on:
                self.c[C_qepg][i] = 0.0
                self.c[C_qppg][i] = 0.0
            if not pp_on:
                self.c[C_qepp][i] = 0.0
                self.c[C_qppp][i] = 0.0
            if self.process_parse_sed:
                self.c[C_qe_tot_dec][i] = self.c[C_qepg][i] + self.c[C_qepp][i]
                self.c[C_qp_tot_dec][i] = self.c[C_qppg][i] + self.c[C_qppp][i]
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiLep(
            self.rp
        )
        var esc = fbool(self.process_escape)
        var exl = fbool(self.process_expansion) * Expansion.afExpLep(self.rp)
        var ann = fbool(
            self.process_annihilation and self.process_annihilation_emission
        )
        for i in range(nl):
            self.c[C_Ae][i] = adi + self.oExtraCool.AExtraCoolE[i]
            self.c[C_ae][i] = esc * self.oEscape.afescE[i] + exl
            self.c[C_qein][i] = self.oInj.inj_electron[i]
            self.c[C_qpin][i] = self.oInj.inj_positron[i]
            self.c[C_qepair][i] = ann * self.oPair.epspairL[i]
            self.c[C_qppair][i] = ann * self.oPair.epspairL[i]
            self.c[C_qele][i] = (
                self.c[C_qepair][i]
                + self.c[C_qebh][i]
                + self.c[C_qe_tot_dec][i]
                + self.c[C_qein][i]
            )
            self.c[C_qpos][i] = (
                self.c[C_qppair][i]
                + self.c[C_qpbh][i]
                + self.c[C_qp_tot_dec][i]
                + self.c[C_qpin][i]
            )
            if self.process_merge_positrons_into_electrons:
                self.c[C_qep][i] = self.c[C_qele][i] + self.c[C_qpos][i]
        if self.process_parse_sed:
            for i in range(nl):
                self.c[C_qepin][i] = self.c[C_qein][i] + self.c[C_qpin][i]
                self.c[C_qepbh][i] = self.c[C_qebh][i] + self.c[C_qpbh][i]
                self.c[C_qeppair][i] = self.c[C_qepair][i] + self.c[C_qppair][i]
                self.c[C_qeppg][i] = self.c[C_qepg][i] + self.c[C_qppg][i]
                self.c[C_qeppp][i] = self.c[C_qepp][i] + self.c[C_qppp][i]

    def Calc_Lep(mut self) raises:
        if self.verbose:
            print(" Calculate electrons/positrons")
        var rows = List[Evolution]()

        @parameter
        def lep(solver: Int, target: Int, q: Int):
            rows.append(Evolution(solver, target, C_ae, C_Ae, q, P_photons))

        if self.process_merge_positrons_into_electrons:
            lep(SA_ep, P_pairs, C_qep)
        else:
            lep(SA_e, P_electrons, C_qele)
            lep(SA_p, P_positrons, C_qpos)
        if self.process_parse_sed:
            lep(SA_epin, P_pairs_in, C_qepin)
            if self.process_annihilation and self.process_annihilation_emission:
                lep(SA_eppair, P_pairs_pair, C_qeppair)
            if (
                self.process_bethe_heitler
                and self.process_bethe_heitler_emission
            ):
                lep(SA_epbh, P_pairs_bh, C_qepbh)
            if self.process_muon_decay:
                lep(SA_eppg, P_pairs_pg, C_qeppg)
            if (
                self.process_pp
                and self.process_pp_emission
                and self.process_muon_decay
            ):
                lep(SA_eppp, P_pairs_pp, C_qeppp)
        self.evolve_group(rows)
        if not self.process_merge_positrons_into_electrons:
            for i in range(self.dat.nbins_lep):
                self.dat.n[P_pairs][i] = (
                    self.dat.n[P_electrons][i] + self.dat.n[P_positrons][i]
                )

    def _lep_ins(self) -> Array[FPtr, 6]:
        var _r: Array[FPtr, 6] = [
            pd(self.dat.n[P_pairs]),
            pd(self.dat.n[P_pairs_in]),
            pd(self.dat.n[P_pairs_bh]),
            pd(self.dat.n[P_pairs_pair]),
            pd(self.dat.n[P_pairs_pg]),
            pd(self.dat.n[P_pairs_pp]),
        ]
        return _r^

    def Calc_Coeff_Phot(mut self):
        if self.verbose:
            print(" Calculate photon coefficient")
        var t0 = self._tic()
        for i in range(self.dat.nbins_pho):
            self.c[C_qgma_inj][i] = self.oInj.inj_photon[i]
        self._toc(21, t0)
        t0 = self._tic()
        if self.process_electron_syn and self.process_electron_syn_emission:
            var outs: Array[FPtr, 6] = [
                pd(self.c[C_qgma_sy]),
                pd(self.c[C_qgma_insy]),
                pd(self.c[C_qgma_bhsy]),
                pd(self.c[C_qgma_pairsy]),
                pd(self.c[C_qgma_pgsy]),
                pd(self.c[C_qgma_ppsy]),
            ]
            self.oSyn.Calc_SY_Phot_Emit(
                self.dat, outs, self._lep_ins(), not self.process_parse_sed
            )
        self._toc(22, t0)
        t0 = self._tic()
        if self.process_ssa:
            self.oSyn.Calc_SY_Phot_Sink(self.dat)
        self._toc(23, t0)
        t0 = self._tic()
        if (
            self.process_electron_compton
            and self.process_electron_compton_emission
        ):
            var outs: Array[FPtr, 6] = [
                pd(self.c[C_qgma_ic]),
                pd(self.c[C_qgma_inic]),
                pd(self.c[C_qgma_bhic]),
                pd(self.c[C_qgma_pairic]),
                pd(self.c[C_qgma_pgic]),
                pd(self.c[C_qgma_ppic]),
            ]
            self.oComp.Calc_IC_Phot_Rad(
                self.dat, outs, self._lep_ins(), not self.process_parse_sed
            )
        self._toc(24, t0)
        t0 = self._tic()
        if (
            self.process_electron_compton
            and self.process_compton_photon_energy_loss
        ):
            self.oComp.Calc_IC_Phot_Loss(self.dat)
        self._toc(25, t0)
        t0 = self._tic()
        if self.process_annihilation and self.process_annihilation_cooling:
            self.oPair.Calc_Pair_Phot(self.dat)
        self._toc(26, t0)
        if self.process_hadronic:
            t0 = self._tic()
            if self.process_photopion and self.process_photopion_photon_loss:
                self.oPhotoPion.Calc_PhotoPion_PhotonLoss(self.dat)
            self._toc(27, t0)
            t0 = self._tic()
            if self.process_proton_syn and self.process_proton_syn_emission:
                self.oSynP.Calc_SY_PRO(self.dat)
            self._toc(28, t0)
            t0 = self._tic()
            if (
                self.process_proton_compton
                and self.process_proton_compton_emission
            ):
                self.oHadIC.Calc_PIC(self.dat)
            self._toc(29, t0)
            t0 = self._tic()
            if self.process_pion_syn and self.process_pion_syn_emission:
                self.oSynP.Calc_SY_Pi(self.dat)
            self._toc(30, t0)
            t0 = self._tic()
            if self.process_pion_compton and self.process_pion_compton_emission:
                self.oHadIC.Calc_PiIC(self.dat)
            self._toc(31, t0)
            t0 = self._tic()
            if self.process_muon_syn and self.process_muon_syn_emission:
                self.oSynP.Calc_SY_Mu(self.dat)
            self._toc(32, t0)
            t0 = self._tic()
            if self.process_muon_compton and self.process_muon_compton_emission:
                self.oHadIC.Calc_MuIC(self.dat)
            self._toc(33, t0)
            t0 = self._tic()
            var ng = self.dat.nbins_pho
            for i in range(ng):
                self.c[C_qgma_pi0][i] = 0.0
                self.c[C_qgma_pi0_pg][i] = 0.0
                self.c[C_qgma_pi0_pp][i] = 0.0
            if self.process_pion_decay:
                if (
                    self.process_pp
                    and self.process_pp_emission
                    and not self.process_pp_emission_pi0_into_cascade
                ):
                    self.oProtonProton.Calc_PP_P2Pi02GG(self.rp, self.dat)
                    var eg = pd(self.oProtonProton.epsilon_gamma_p2pi02GG)
                    for i in range(ng):
                        self.c[C_qgma_pi0_pp][i] = (
                            1.0
                            / Rate_SSC_CGS
                            * self.dat.grid_had2lep(i, eg, 0, ng)
                        )
                else:
                    for i in range(ng):
                        self.c[C_qgma_pi0_pp][i] = 0.0
                if self.process_parse_sed:
                    if (
                        self.process_photopion
                        and self.process_photopion_emission
                    ):
                        self.oPionDec.Calc_PItoG(self.dat, self.c[C_qpi0_pg])
                        var qg = pd(self.oPionDec.QPI2GGen)
                        for i in range(ng):
                            self.c[C_qgma_pi0_pg][i] = (
                                1.0
                                / Rate_SSC_CGS
                                * self.dat.grid_had2lep(i, qg, 0, ng)
                            )
                    if self.process_pp and self.process_pp_emission:
                        if self.process_pp_emission_pi0_into_cascade:
                            self.oPionDec.Calc_PItoG(
                                self.dat, self.c[C_qpi0_pp]
                            )
                            var qg = pd(self.oPionDec.QPI2GGen)
                            for i in range(ng):
                                self.c[C_qgma_pi0_pp][i] += (
                                    1.0
                                    / Rate_SSC_CGS
                                    * self.dat.grid_had2lep(i, qg, 0, ng)
                                )
                    for i in range(ng):
                        self.c[C_qgma_pi0][i] = (
                            self.c[C_qgma_pi0_pg][i] + self.c[C_qgma_pi0_pp][i]
                        )
                else:
                    self.oPionDec.Calc_PItoG(self.dat, self.c[C_qpi0])
                    var qg = pd(self.oPionDec.QPI2GGen)
                    for i in range(ng):
                        self.c[C_qgma_pi0][i] = (
                            1.0
                            / Rate_SSC_CGS
                            * self.dat.grid_had2lep(i, qg, 0, ng)
                        )
            self._toc(34, t0)
        t0 = self._tic()
        self._toc(35, t0)

    def Pass_Coeff_Phot(mut self):
        if self.verbose:
            print("Pass photon coefficient")
        var ng = self.dat.nbins_pho
        if self.process_hadronic:
            var r = 1.0 / Rate_SSC_CGS
            var psy = (
                fbool(
                    self.process_proton_syn and self.process_proton_syn_emission
                )
                / Rate_SSC_CGS
            )
            var pic = (
                fbool(
                    self.process_proton_compton
                    and self.process_proton_compton_emission
                )
                / Rate_SSC_CGS
            )
            var pisy = (
                fbool(self.process_pion_syn and self.process_pion_syn_emission)
                / Rate_SSC_CGS
            )
            var piic = (
                fbool(
                    self.process_pion_compton
                    and self.process_pion_compton_emission
                )
                / Rate_SSC_CGS
            )
            var musy = (
                fbool(self.process_muon_syn and self.process_muon_syn_emission)
                / Rate_SSC_CGS
            )
            var muic = (
                fbool(
                    self.process_muon_compton
                    and self.process_muon_compton_emission
                )
                / Rate_SSC_CGS
            )
            var pgl = (
                fbool(
                    self.process_photopion
                    and self.process_photopion_photon_loss
                )
                * T_SSC_CGS
            )
            _ = r
            var e_psy = pd(self.oSynP.epsilonsyP)
            var e_pic = pd(self.oHadIC.epsilonPIC)
            var e_pisy = pd(self.oSynP.epsilonsyPi)
            var e_piic = pd(self.oHadIC.epsilonPiIC)
            var e_musy = pd(self.oSynP.epsilonsyMu)
            var e_muic = pd(self.oHadIC.epsilonMuIC)
            var e_pg = pd(self.oPhotoPion.afPG_Rad)
            for i in range(ng):
                self.c[C_qgma_psy][i] = psy * self.dat.grid_had2lep(
                    i, e_psy, 0, ng
                )
                self.c[C_qgma_pic][i] = pic * self.dat.grid_had2lep(
                    i, e_pic, 0, ng
                )
                self.c[C_qgma_pisy][i] = pisy * self.dat.grid_had2lep(
                    i, e_pisy, 0, ng
                )
                self.c[C_qgma_piic][i] = piic * self.dat.grid_had2lep(
                    i, e_piic, 0, ng
                )
                self.c[C_qgma_musy][i] = musy * self.dat.grid_had2lep(
                    i, e_musy, 0, ng
                )
                self.c[C_qgma_muic][i] = muic * self.dat.grid_had2lep(
                    i, e_muic, 0, ng
                )
                self.c[C_qgma_had][i] = (
                    self.c[C_qgma_pi0][i]
                    + self.c[C_qgma_psy][i]
                    + self.c[C_qgma_pic][i]
                    + self.c[C_qgma_pisy][i]
                    + self.c[C_qgma_piic][i]
                    + self.c[C_qgma_musy][i]
                    + self.c[C_qgma_muic][i]
                )
                self.c[C_agma_pg][i] = pgl * self.dat.grid_had2lep(
                    i, e_pg, 0, ng
                )
        else:
            for i in range(ng):
                self.c[C_qgma_had][i] = 0.0
                self.c[C_agma_pg][i] = 0.0
        var ann = fbool(
            self.process_annihilation and self.process_annihilation_cooling
        )
        var ssa = fbool(self.process_electron_syn and self.process_ssa)
        var icl = fbool(
            self.process_electron_compton
            and self.process_compton_photon_energy_loss
        )
        var exl = fbool(self.process_expansion) * Expansion.afExpLep(self.rp)
        var esc = fbool(self.process_escape)
        for i in range(ng):
            self.c[C_Agma][i] = self.oExtraCool.AExtraCoolG[i]
            self.c[C_agma][i] = (
                ann * self.oPair.afpairG[i]
                + ssa * self.oSyn.afsyG[i]
                + icl * self.oComp.aficG[i]
                + exl
                + esc * self.oEscape.afescG[i]
                + self.c[C_agma_pg][i]
            )
            self.c[C_qgma][i] = (
                self.c[C_qgma_sy][i]
                + self.c[C_qgma_ic][i]
                + self.c[C_qgma_had][i]
                + self.c[C_qgma_inj][i]
            )
            self.c[C_qgma1][i] = (
                self.c[C_qgma_sy][i]
                + self.c[C_qgma_ic][i]
                + self.c[C_qgma_had][i]
            )

    def Calc_Phot(mut self) raises:
        if self.verbose:
            print(" Calculate photons")
        # the total photon density first: the components are evolved in the
        # updated photon field
        var total = List[Evolution]()
        total.append(
            Evolution(SA_g, P_photons, C_agma, C_Agma, C_qgma, P_photons)
        )
        self.evolve_group(total)
        var rows = List[Evolution]()

        @parameter
        def comp(solver: Int, target: Int, q: Int):
            rows.append(Evolution(solver, target, C_agma, C_Agma, q, P_photons))

        if self.process_parse_sed:
            comp(SA_ginsy, P_photons_insy, C_qgma_insy)
            comp(SA_ginic, P_photons_inic, C_qgma_inic)
            comp(SA_gbhsy, P_photons_bhsy, C_qgma_bhsy)
            comp(SA_gbhic, P_photons_bhic, C_qgma_bhic)
            comp(SA_gpairsy, P_photons_pairsy, C_qgma_pairsy)
            comp(SA_gpairic, P_photons_pairic, C_qgma_pairic)
            comp(SA_gpgsy, P_photons_pgsy, C_qgma_pgsy)
            comp(SA_gpgic, P_photons_pgic, C_qgma_pgic)
            comp(SA_gppsy, P_photons_ppsy, C_qgma_ppsy)
            comp(SA_gppic, P_photons_ppic, C_qgma_ppic)
            comp(SA_gpi0_pg, P_photons_pi0_pg, C_qgma_pi0_pg)
            comp(SA_gpi0_pp, P_photons_pi0_pp, C_qgma_pi0_pp)
            comp(SA_gpsy, P_photons_psy, C_qgma_psy)
            comp(SA_gpic, P_photons_pic, C_qgma_pic)
            comp(SA_gpisy, P_photons_pisy, C_qgma_pisy)
            comp(SA_gpiic, P_photons_piic, C_qgma_piic)
            comp(SA_gmusy, P_photons_musy, C_qgma_musy)
            comp(SA_gmuic, P_photons_muic, C_qgma_muic)
            comp(SA_ginj, P_photons_inj, C_qgma_inj)
        comp(SA_g1, P_photons_except_injected, C_qgma1)
        self.evolve_group(rows)

    def Update_Mag_Field(mut self, b_Gauss: Float64):
        if self.verbose:
            print("Magnetic field strength updated to ", b_Gauss)
        self.rp.mag_field = b_Gauss
        self.oSyn.b_gauss = b_Gauss
        self.oSynP.b_gauss = b_Gauss
        self.oSyn.reinitialize_empty_kernels(self.dat)
        if (
            self.process_electron_syn and self.process_electron_syn_emission
        ) or (self.process_electron_syn and self.process_ssa):
            self.oSyn.initialize_kernels(self.dat, self.process_quantum_syn)
        if self.verbose:
            print("for e+/-")
        if self.process_hadronic:
            self.oSynP.reinitialize_empty_kernels(self.dat)
            self.oSynP.initialize_kernels(
                self.dat,
                self.process_proton_syn and self.process_proton_syn_emission,
                self.process_pion_syn and self.process_pion_syn_emission,
                self.process_muon_syn and self.process_muon_syn_emission,
            )

    def estimate_maximum_lepton_energy(mut self):
        if self.verbose:
            print("estimating maximum lepton energy")
        var nl = self.dat.nbins_lep
        var total_loss = zeros(nl)
        var sy = fbool(
            self.process_electron_syn and self.process_electron_syn_cooling
        )
        var ic = fbool(
            self.process_electron_compton
            and self.process_electron_compton_cooling
        )
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiLep(
            self.rp
        )
        for i in range(nl):
            total_loss[i] = (
                -sy * self.S[SA_e].Asy[i] - ic * self.S[SA_e].Aic[i] - adi
            )
        var imax = 0
        while (
            imax < nl
            and self.oAcc.AAClepBohm(self.rp, self.dat, imax)
            >= total_loss[imax]
        ):
            imax += 1
        # when acceleration wins on the whole grid the C++ code reads one element
        # past the end of egrid_lep; use the next grid point exp(n*dlnE) instead
        var e = self.dat.egrid_lep[imax] if imax < nl else c_exp(
            Float64(imax) * self.dat.dlnE
        )
        self.rp.estimated_electron_emax = e * masse * 1e9

    def estimate_maximum_proton_energy(mut self):
        if self.verbose:
            print("estimating maximum proton energy")
        var nh = self.dat.nbins_had
        var total_loss = zeros(nh)
        var sy = fbool(
            self.process_proton_syn and self.process_proton_syn_cooling
        )
        var ic = fbool(
            self.process_proton_compton and self.process_proton_compton_cooling
        )
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiHad(
            self.rp
        )
        var pg = fbool(
            self.process_photopion and self.process_photopion_cooling
        )
        var pp = fbool(self.process_pp and self.process_pp_cooling)
        var bh = fbool(
            self.process_bethe_heitler and self.process_bethe_heitler_cooling
        )
        for i in range(nh):
            total_loss[i] = (
                -sy * self.S[SA_proton].Asy[i]
                - ic * self.S[SA_proton].Aic[i]
                - adi
                + pg * self.oPhotoPion.afPG_PRO[i]
                - pp * self.oProtonProton.APP_PRO[i]
                - bh * self.oBH.ABHHadP[i]
            )
        var imax = 0
        while (
            imax < nh
            and self.oAcc.AAChadBohm(self.rp, self.dat, imax)
            >= total_loss[imax]
        ):
            imax += 1
        var e = self.dat.egrid_had[imax] if imax < nh else c_exp(
            Float64(imax) * self.dat.dlnE
        )
        self.rp.estimated_proton_emax = e * 1e9
