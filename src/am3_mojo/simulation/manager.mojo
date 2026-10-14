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
from am3_mojo.physics.pair_annihilation import PairAnnihilation
from am3_mojo.physics.triplet import TripletPairProduction
from am3_mojo.physics.bethe_heitler import BetheHeitler
from am3_mojo.physics.photopion import PhotoPion
from am3_mojo.physics.proton_proton import ProtonProton
from am3_mojo.physics.decays import (
    PionDecay,
    MuonDecay,
    HadIC,
    NeutronDecay,
    KaonDecay,
    N_PROD,
    kaon_mass,
    kaon_lifetime,
    kaon_name,
)
from am3_mojo.utils.parallel import default_num_threads
from am3_mojo.simulation.evolve import Evolution, EvolveJob, run_jobs, vp


# profiler slots: 0-35 are AM3's stages (get_profiled_time_labels), the rest
# the stages of the processes am3_mojo adds
comptime T_PAIR_ANN = 36
comptime T_TRIPLET = 37
comptime T_NEUTRON_DECAY = 38
comptime T_KAON_COEFF = 39
comptime T_KAON_EVOLVE = 40
comptime T_KAON_SYN = 41
comptime T_KAON_IC = 42
comptime N_TIMING = 43


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
    var oPairAnn: PairAnnihilation
    var oTPP: TripletPairProduction
    var oNeutronDec: NeutronDecay
    var oKaonDec: KaonDecay
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
    # processes am3_mojo adds to AM3 (on by default)
    var process_pair_annihilation: Bool
    var process_pair_annihilation_emission: Bool
    var process_pair_annihilation_cooling: Bool
    var process_triplet_pair_production: Bool
    var process_triplet_pair_production_emission: Bool
    var process_triplet_pair_production_cooling: Bool
    var process_neutron_decay: Bool
    var process_kaons: Bool
    var process_kaon_decay: Bool
    var process_kaon_syn: Bool
    var process_kaon_syn_emission: Bool
    var process_kaon_syn_cooling: Bool
    var process_kaon_compton: Bool
    var process_kaon_compton_emission: Bool
    var process_kaon_compton_cooling: Bool
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
    var pg_conv_fresh: Bool  # kaon stage did this step's photo-pion convolution
    var kernels_ready: Bool  # init_kernels ran for the current grid
    var pair_halves_valid: Bool  # e-/e+ parts of the pair components current
    # per kaon origin (photo-hadronic, pp): the pions, muons and neutrinos of
    # the kaon decays are evolved in this step / may be non-zero
    var kchain: List[Bool]
    var kchain_live: List[Bool]
    var hadronic_built: Bool  # init_kernels ran with the hadronic processes
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
        self.oPairAnn = PairAnnihilation(self.dat, False)
        self.oTPP = TripletPairProduction(self.dat, False)
        self.oNeutronDec = NeutronDecay(self.dat, False)
        self.oKaonDec = KaonDecay(self.dat, False)
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
        self.oPairAnn.nthreads = self.nthreads
        self.oTPP.nthreads = self.nthreads
        self.oNeutronDec.nthreads = self.nthreads
        self.oKaonDec.nthreads = self.nthreads

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
        self.process_pair_annihilation = True
        self.process_pair_annihilation_emission = True
        self.process_pair_annihilation_cooling = True
        self.process_triplet_pair_production = True
        self.process_triplet_pair_production_emission = True
        self.process_triplet_pair_production_cooling = True
        self.process_neutron_decay = True
        self.process_kaons = True
        self.process_kaon_decay = True
        self.process_kaon_syn = True
        self.process_kaon_syn_emission = True
        self.process_kaon_syn_cooling = True
        self.process_kaon_compton = True
        self.process_kaon_compton_emission = True
        self.process_kaon_compton_cooling = True
        self.verbose = False

        self.c = List[Vec](capacity=C_COUNT)
        for _ in range(C_COUNT):
            self.c.append(Vec())
        self.S = List[SolverAlgorithm]()
        self.step_counter = 0
        self.profile_timing = False
        self.pg_conv_fresh = False
        self.kernels_ready = False
        self.hadronic_built = False
        self.pair_halves_valid = True
        self.kchain = [False, False]
        self.kchain_live = [False, False]
        self.timing = zeros(N_TIMING)
        self.Initialize_Solver_Algorithms()
        self.reset_coefficients()  # (getters work before init_kernels)

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
        self.oPairAnn.verbose = v
        self.oTPP.verbose = v
        self.oNeutronDec.verbose = v
        self.oKaonDec.verbose = v
        self.oExtraCool.verbose = v
        for i in range(len(self.S)):
            self.S[i].verbose = v

    def init_kernels(mut self) raises:
        # the pp delta approximation's transition energy must lie on the hadron
        # grid: above its top AM3 crashes, and unchecked the pions get no energy
        var etop = self.dat.egrid_had[self.dat.nbins_had - 1]
        if (
            self.process_hadronic
            and self.process_pp
            and self.pp_transition_energy_to_delta_approx > etop
        ):
            raise Error(
                String(
                    "the transition energy of the pp delta approximation (",
                    self.pp_transition_energy_to_delta_approx * 1e9,
                    " eV) lies above the hadron grid (top: ",
                    etop * 1e9,
                    " eV): set_pp_transition_energy_to_delta_approx() a lower",
                    " value or update_energy_grid() a larger grid",
                )
            )
        print("init. AM3 kernels:")
        self.print_process_switches()
        print()
        self.rp.print_params()

        self.oSyn.reinitialize_empty_kernels(self.dat)
        self.oComp.reinitialize_empty_kernels(self.dat)
        if self.process_electron_compton:
            # (the C++ "every 2nd grid point" options are unreachable: the
            #  optimize_*_grid switches are C++ bools, i.e. step 1)
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
        if self.pair_annihilation_active():
            self.oPairAnn.initialize_kernels(self.dat)
        else:
            self.oPairAnn.free_kernels()  # (no stale kernels of another grid)
        if self.process_triplet_pair_production:
            self.oTPP.initialize_kernels(self.dat)
        else:
            self.oTPP.free_kernels()
        if self.process_hadronic:
            self.oSynP.reinitialize_empty_kernels(self.dat)
            self.oHadIC.reinitialize_empty_kernels(
                self.dat, True, True, True, True
            )
            # (the step tables are cheap and always built, so that an emission
            #  switched on after init_kernels gets the right one)
            self.oHadIC.initialize_kernels(self.dat, True, True, True, True)
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
            if self.process_neutron_decay:
                self.oNeutronDec.initialize_kernels(self.dat)
            else:
                self.oNeutronDec.free_kernels()
            if self.process_kaons:
                self.oKaonDec.initialize_kernels(self.dat)
            else:
                self.oKaonDec.free_kernels()
            # (the modules are always reset to the current grid, so that no
            #  kernel of an earlier grid survives)
            self.oPhotoPion.reinitialize_empty_kernels(
                self.dat, self.optimize_photopion_target_photon_max
            )
            if self.process_photopion:
                # (the "every n-th target photon" option is unreachable: the
                #  optimize_photopion_target_photon_grid switch is a C++ bool)
                self.oPhotoPion.initialize_kernels(
                    self.dat,
                    Int(self.process_photopion_emission),
                    Int(self.process_photopion_cooling),
                    Int(self.process_photopion_photon_loss),
                    self.optimize_photopion_target_photon_max,
                    self.process_kaons,
                )
            self.oProtonProton.reinitialize_empty_kernels(self.dat)
            if self.process_pp:
                self.oProtonProton.initialize_kernels(
                    self.dat,
                    self.process_pp_emission_pi0_into_cascade,
                    self.pp_transition_energy_to_delta_approx,
                    self.pp_model_charged_pions,
                    self.pp_model_neutral_pions,
                    self.process_kaons,
                )
        self.Update_Mag_Field(self.rp.mag_field)
        self.clear_am3()
        self.kernels_ready = True
        self.hadronic_built = self.process_hadronic

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
        self.oPairAnn.clear_internal_arrays(self.dat)
        self.oTPP.clear_internal_arrays(self.dat)
        self.oNeutronDec.clear_internal_arrays(self.dat)
        self.oKaonDec.clear_internal_arrays(self.dat)
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
        self.pair_halves_valid = True  # (all zero)
        self.kchain_live = [False, False]

    def clear_am3(mut self):
        if self.verbose:
            print("Sim.Manager: clear_particle_densities")
        self.clear_particle_densities()
        self.reset_coefficients()
        self.ReInit_Physics_Modules()
        self.Initialize_Solver_Algorithms()

    def _nonzero(self, id: Int) -> Bool:
        for v in self.dat.n[id]:
            if v != 0.0:
                return True
        return False

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
        self._reset_coeffs(C_EXT_LEP_BEGIN, C_EXT_LEP_END, self.dat.nbins_lep)
        self._reset_coeffs(C_EXT_PHO_BEGIN, C_EXT_PHO_END, self.dat.nbins_pho)
        self._reset_coeffs(C_EXT_HAD_BEGIN, C_EXT_HAD_END, self.dat.nbins_had)
        self._reset_coeffs(C_EXT_NU_BEGIN, C_EXT_NU_END, self.dat.nbins_nu)
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
        print("processes am3_mojo adds to AM3:")
        print(
            "  e+e- annihilation: ",
            b(self.process_pair_annihilation),
            "  (photon em.: ",
            b(self.process_pair_annihilation_emission),
            ", e+/- loss: ",
            b(self.process_pair_annihilation_cooling),
            ")",
            sep="",
        )
        print(
            "  triplet pair production (e gamma -> e e+ e-): ",
            b(self.process_triplet_pair_production),
            "  (pair em.: ",
            b(self.process_triplet_pair_production_emission),
            ", e+/- loss: ",
            b(self.process_triplet_pair_production_cooling),
            ")",
            sep="",
        )
        print(
            "  neutron decay (n -> p e- anti-nu_e): ",
            b(self.process_neutron_decay),
            sep="",
        )
        print(
            "  kaons (N-gamma, pp -> K+-, K0_S, K0_L): ",
            b(self.process_kaons),
            "  (decay: ",
            b(self.process_kaon_decay),
            "; synchrotron: ",
            b(self.process_kaon_syn),
            " (em.: ",
            b(self.process_kaon_syn_emission),
            ", cool.: ",
            b(self.process_kaon_syn_cooling),
            "); compton: ",
            b(self.process_kaon_compton),
            " (em.: ",
            b(self.process_kaon_compton_emission),
            ", cool.: ",
            b(self.process_kaon_compton_cooling),
            "))",
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

    def _mk_kaon(
        self,
        s: Int,
        nh: Int,
        pgo_h: Int,
        dt_sec: Float64,
        sy: Bool,
        qsc: Bool,
        ic: Bool,
    ) -> SolverAlgorithm:
        """Solver of kaon species s: synchrotron and inverse-Compton cooling
        for K+- (as for charged pions), decay."""
        var charged = s == KP or s == KM
        return self._mk(
            kaon_name(s),
            nh,
            pgo_h,
            1.0,
            1.0,
            1.0,
            kaon_lifetime(s),
            kaon_mass(s) / masse,
            dt_sec,
            sy and charged,
            qsc and charged,
            ic and charged,
            self.process_kaon_decay,
        )

    def Initialize_Solver_Algorithms(mut self):
        if self.verbose:
            print("Initialize_Solver_Algorithms")
        var t_nodecay_s = 1.0e7
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
        # neutron decay (am3_mojo addition): PDG 2026 mean life and mass. AM3
        # gives the neutron solver 879.5 s and m_p, with the decay off.
        S.append(
            self._mk(
                "neutron",
                nh,
                pgo_h,
                1.0,
                1.0,
                1.0,
                tdec_n,
                massN_PDG / masse,
                dt_sec,
                False,
                False,
                False,
                self.process_neutron_decay,
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
        # species am3_mojo adds to AM3
        S.append(
            self._mk(
                "photon_pair_annihilation",
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
        S.append(
            self._mk(
                "triplet_produced_pairs",
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
        for n in ["photon_triplet-produced_syn", "photon_triplet-produced_IC"]:
            S.append(
                self._mk(
                    String(n),
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
        S.append(
            self._mk(
                "neutron_decay_electrons",
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
        for n in [
            "photon_neutron-decay-electrons_syn",
            "photon_neutron-decay-electrons_IC",
        ]:
            S.append(
                self._mk(
                    String(n),
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
        S.append(
            self._mk(
                "neutron_decay_electron_anti_neutrino",
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
        var k_sy = self.process_kaon_syn and self.process_kaon_syn_cooling
        var k_ic = (
            self.process_kaon_compton and self.process_kaon_compton_cooling
        )
        for s in range(N_KAON):  # kaons from photo-hadronic interactions
            S.append(self._mk_kaon(s, nh, pgo_h, dt_sec, k_sy, qsc, k_ic))
        S.append(
            self._mk(
                "photon_kaon_syn",
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
        S.append(
            self._mk(
                "photon_kaon_IC",
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
        for s in range(N_KAON):  # kaons from pp collisions
            S.append(self._mk_kaon(s, nh, pgo_h, dt_sec, k_sy, qsc, k_ic))
        # the pions, muons and neutrinos of the kaon decays, per kaon origin:
        # configured as AM3's pion, muon and neutrino components
        var kpre: List[String] = ["pgamma_kaon_decay_", "pp_kaon_decay_"]
        for o in range(2):
            for k in range(2):
                S.append(
                    self._mk(
                        kpre[o] + pion_names[k],
                        nh,
                        pgo_h,
                        1.0,
                        1.0,
                        1.0,
                        t_pipm_decay_s,
                        (masspip if k == 0 else masspim) / masse,
                        dt_sec,
                        pi_sy,
                        qsc,
                        pi_ic,
                        pid,
                    )
                )
        for o in range(2):
            for k in range(4):
                S.append(
                    self._mk(
                        kpre[o] + mu_names[k],
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
        for o in range(2):
            for k in range(4):
                S.append(
                    self._mk(
                        kpre[o] + nu_names[k],
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
        comptime assert (
            SA_gpann == 61
            and SA_eptpp == 62
            and SA_gtppsy == 63
            and SA_gtppic == 64
            and SA_endec == 65
            and SA_gndecsy == 66
            and SA_gndecic == 67
            and SA_nueA_ndec == 68
            and SA_K == 69
            and SA_Km == 70
            and SA_KS == 71
            and SA_KL == 72
            and SA_gksy == 73
            and SA_gkic == 74
            and SA_Kpp == 75
            and SA_Pip_Kpg == 79
            and SA_Pip_Kpp == 81
            and SA_MpL_Kpg == 83
            and SA_MpL_Kpp == 87
            and SA_nue_Kpg == 91
            and SA_nue_Kpp == 95
            and N_SOLVERS == 99
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

        Cooling rates depend only on the species and the target photons: the
        first solver of each configuration computes them and shares them with
        the others. The solves then run in parallel."""
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

    def _toc_add(mut self, idx: Int, t0: Int):
        """As _toc, for the later parts of a sub-step timed in several parts."""
        if self.profile_timing:
            self.timing[idx] += Float64(perf_counter_ns() - t0)

    def evolve_step(mut self) raises:
        if self.verbose:
            print("evolve for 1 step")
        if not self.kernels_ready:
            raise Error(
                "init_kernels() must be called before evolve_step() (and"
                " again after update_energy_grid())"
            )
        if self.process_hadronic and not self.hadronic_built:
            raise Error(
                "the hadronic processes were switched on after init_kernels():"
                " call init_kernels() again"
            )
        # kaons are made in this step where they are switched on and their
        # kernels were built at init_kernels
        var had = self.process_hadronic
        self.oPhotoPion.set_kaons(had and self.kaons_pg_active())
        self.oProtonProton.kaons_src = (
            had and self.kaons_pp_active() and self.oProtonProton.kaons_built
        )
        # (as for AM3's pions, the protons lose the energy that goes into
        #  kaons whether or not the emission is followed)
        self.oProtonProton.kaons_loss = (
            had
            and self.process_kaons
            and self.process_pp
            and self.oProtonProton.kaons_built
        )
        self._update_kaon_chain()
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
        var t0: Int
        if self.process_hadronic:
            t0 = self._tic()
            self.Pass_LepG2HadG()
            self._toc(0, t0)
            t0 = self._tic()
            self.Calc_Coeff_proton_NTR()
            self._toc(1, t0)
            # the neutrons before the protons, so that their decays feed this
            # step's protons, e- and anti-nu_e. AM3, where the two are
            # independent within a step, evolves the protons first.
            t0 = self._tic()
            self.Pass_Coeff_NTR()
            self._toc(4, t0)
            t0 = self._tic()
            self.Calc_NTR()
            self._toc(5, t0)
            t0 = self._tic()
            self.Calc_Neutron_Decay()
            self._toc(T_NEUTRON_DECAY, t0)
            t0 = self._tic()
            self.Pass_Coeff_PRO()
            self._toc(2, t0)
            t0 = self._tic()
            self.Calc_PRO()
            self._toc(3, t0)
            t0 = self._tic()
            self.Calc_Coeff_K()
            self._toc(T_KAON_COEFF, t0)
            t0 = self._tic()
            self.Calc_K()
            self._toc(T_KAON_EVOLVE, t0)
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
            self.oPhotoPion.Calc_PhotoPion_PhotonConvolution(
                self.dat, self.optimize_photopion_target_photon_max
            )
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
        if self.neutron_decay_on():
            for i in range(self.dat.nbins_had):
                self.c[C_qpro][i] += self.c[C_qpro_ndec][i]

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

    # ------------------------------------------------------------------
    # neutron decay: n -> p e- anti-nu_e (am3_mojo addition)
    # ------------------------------------------------------------------
    def neutron_decay_on(self) -> Bool:
        """Neutron decay acts in the step: the hadronic processes are on, and
        init_kernels set up both the neutron solver's decay sink and the
        product kernels (they must agree)."""
        return (
            self.process_hadronic
            and self.S[SA_neutron].process_decay
            and self.oNeutronDec.ready(self.dat)
        )

    def Calc_Neutron_Decay(mut self):
        """Products of the neutrons that decay in this step, from the evolved
        neutrons as in AM3's decays: sources of the protons, the electrons
        (lepton solver units) and the electron antineutrinos."""
        if self.verbose:
            print(" Calculate neutron decay")
        if self.neutron_decay_on():
            self.oNeutronDec.Calc_NtoX(self.dat, self.S[SA_neutron].alpha_decay)
        else:
            self.oNeutronDec.clear_internal_arrays(self.dat)
        for i in range(self.dat.nbins_had):
            self.c[C_qpro_ndec][i] = self.oNeutronDec.q_p[i]
        for i in range(self.dat.nbins_lep):
            self.c[C_qe_ndec][i] = self.oNeutronDec.q_e[i] / Rate_SSC_CGS
        for i in range(self.dat.nbins_nu):
            self.c[C_qnuea_ndec][i] = self.oNeutronDec.q_nu[i]

    # ------------------------------------------------------------------
    # kaons: K+, K-, K0_S, K0_L (am3_mojo addition)
    # ------------------------------------------------------------------
    def kaons_pg_active(self) -> Bool:
        """Kaons are produced in photo-hadronic interactions: with the
        photo-pion process and its emission on."""
        return (
            self.process_kaons
            and self.process_hadronic
            and self.process_photopion
            and self.process_photopion_emission
        )

    def kaons_pp_active(self) -> Bool:
        """Kaons are produced in pp collisions: with the pp process and its
        emission on."""
        return (
            self.process_kaons
            and self.process_hadronic
            and self.process_pp
            and self.process_pp_emission
        )

    def kaons_active(self) -> Bool:
        return self.kaons_pg_active() or self.kaons_pp_active()

    # the same in the step: switched on and kernels built (set at the start
    # of evolve_step)
    def kaons_pg_on(self) -> Bool:
        return self.oPhotoPion.kaons

    def kaons_pp_on(self) -> Bool:
        return self.oProtonProton.kaons_src

    def kaons_on(self) -> Bool:
        return self.kaons_pg_on() or self.kaons_pp_on()

    def kaon_chain_on(self, origin: Int) -> Bool:
        """The kaon decay chain of an origin (0 photo-hadronic, 1 pp) is
        evolved as components in this step (_update_kaon_chain)."""
        return self.kchain[origin]

    def _update_kaon_chain(mut self):
        """Once per step, with the components on: evolve the kaon decay
        products of each origin while those kaons decay, otherwise while any are
        left, so that they escape like their species. kchain_live is false where
        they are known to be zero, sparing the scan without kaons."""
        for o in range(2):
            var on = self.kaons_pg_on() if o == 0 else self.kaons_pp_on()
            var chain = False
            if self.process_parse_sed:
                if on and self.kaon_decay_on():
                    chain = True
                elif self.kchain_live[o]:
                    chain = self._kaon_chain_nonzero(o)
                self.kchain_live[o] = chain
            self.kchain[o] = chain

    def _kaon_chain_nonzero(self, origin: Int) -> Bool:
        var p0 = P_pion_plus_Kpg if origin == 0 else P_pion_plus_Kpp
        var m0 = P_muon_plus_left_Kpg if origin == 0 else P_muon_plus_left_Kpp
        var n0 = P_neutrino_el_Kpg if origin == 0 else P_neutrino_el_Kpp
        for k in range(4):
            if (
                (k < 2 and self._nonzero(p0 + k))
                or self._nonzero(m0 + k)
                or self._nonzero(n0 + k)
            ):
                return True
        return False

    def kaon_decay_on(self) -> Bool:
        """Kaon decay as configured in the kaon solvers at init_kernels (their
        decay sink and the decay products must agree)."""
        return self.kaons_on() and self.S[SA_K].process_decay

    def Calc_Coeff_K(mut self):
        """Kaon sources (photo-hadronic interactions and pp collisions),
        escape and cooling."""
        if self.verbose:
            print(" Calculate kaon coefficient")
        self.pg_conv_fresh = False
        if not self.kaons_on():
            return
        var nh = self.dat.nbins_had
        if self.kaons_pg_on():
            if not (self.process_photopion and self.process_photopion_cooling):
                self.oPhotoPion.Calc_PhotoPion_PhotonConvolution(
                    self.dat, self.optimize_photopion_target_photon_max
                )
                self.pg_conv_fresh = True
            self.oPhotoPion.Calc_PhotoPion_Kaons(self.dat)
            for s in range(N_KAON):
                for i in range(nh):
                    self.c[C_qK + s][i] = self.oPhotoPion.qK[s][i]
        if self.kaons_pp_on():
            self.oProtonProton.Calc_PP_KaonSrc(self.rp, self.dat)
            for s in range(N_KAON):
                for i in range(nh):
                    self.c[C_qKpp + s][i] = self.oProtonProton.qK[s][i]
        # escape and cooling: charged kaons as charged pions
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiHad(
            self.rp
        )
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        # neutral kaons as neutrons: no adiabatic cooling, neutral escape
        for i in range(self.dat.nbins_had):
            self.c[C_AK][i] = adi + self.oExtraCool.AExtraCoolK[i]
            self.c[C_aK][i] = esc * self.oEscape.afescPipm[i] + exh
            self.c[C_AK0][i] = self.oExtraCool.AExtraCoolK[i]
            self.c[C_aK0][i] = esc * self.oEscape.afescNTR[i] + exh

    def Calc_K(mut self) raises:
        """Evolve the kaons of both origins and compute their decay products
        (pions, muons, neutrinos, e+-), sources for the rest of this step."""
        if self.verbose:
            print(" Calculate kaons")
        comptime assert (
            P_kaon_minus == P_kaon_plus + KM
            and P_kaon_0S == P_kaon_plus + KS
            and P_kaon_0L == P_kaon_plus + KL
        )
        var pg = self.kaons_pg_on()
        var pp = self.kaons_pp_on()
        var rows = List[Evolution]()
        for s in range(N_KAON):
            var a = C_aK if s <= KM else C_aK0
            var A = C_AK if s <= KM else C_AK0
            if pg:
                rows.append(
                    Evolution(
                        kaon_solver(s),
                        P_kaon_plus_pg + s,
                        a,
                        A,
                        C_qK + s,
                        P_photons_on_had_grid,
                    )
                )
            if pp:
                rows.append(
                    Evolution(
                        kaon_solver(s, True),
                        P_kaon_plus_pp + s,
                        a,
                        A,
                        C_qKpp + s,
                        P_photons_on_had_grid,
                    )
                )
        if len(rows) > 0:
            self.evolve_group(rows)
        var fg = fbool(pg)
        var fp = fbool(pp)
        for s in range(N_KAON):
            for i in range(self.dat.nbins_had):
                self.dat.n[P_kaon_plus + s][i] = (
                    fg * self.dat.n[P_kaon_plus_pg + s][i]
                    + fp * self.dat.n[P_kaon_plus_pp + s][i]
                )
        # products (physics/decays.mojo order) -> coefficient arrays
        var dest: Array[Int, N_PROD] = [
            C_qpip_K,
            C_qpim_K,
            C_qpi0_K,
            C_qmupl_K,
            C_qmupr_K,
            C_qmuml_K,
            C_qmumr_K,
            C_qe_K,
            C_qp_K,
            C_qnue_K,
            C_qnuea_K,
            C_qnum_K,
            C_qnuma_K,
        ]
        var dest_pp: Array[Int, N_PROD] = [
            C_qpip_Kpp,
            C_qpim_Kpp,
            C_qpi0_Kpp,
            C_qmupl_Kpp,
            C_qmupl_Kpp + 1,
            C_qmupl_Kpp + 2,
            C_qmupl_Kpp + 3,
            C_qe_Kpp,
            C_qp_Kpp,
            C_qnue_Kpp,
            C_qnue_Kpp + 1,
            C_qnue_Kpp + 2,
            C_qnue_Kpp + 3,
        ]
        for origin in range(2):
            var on = pg if origin == 0 else pp
            if on and self.kaon_decay_on():
                self.oKaonDec.Calc_KtoX(
                    self.dat, P_kaon_plus_pg if origin == 0 else P_kaon_plus_pp
                )
            else:
                self.oKaonDec.clear_internal_arrays(self.dat)
            for p in range(N_PROD):
                var d = dest[p] if origin == 0 else dest_pp[p]
                for i in range(len(self.oKaonDec.out[p])):
                    self.c[d][i] = self.oKaonDec.out[p][i]

    def Calc_Coeff_PI(mut self):
        if self.verbose:
            print(" Calculate pion coefficient")
        if self.process_photopion and self.process_photopion_emission:
            if (
                not (self.process_photopion and self.process_photopion_cooling)
                and not self.pg_conv_fresh
            ):
                self.oPhotoPion.Calc_PhotoPion_PhotonConvolution(
                    self.dat, self.optimize_photopion_target_photon_max
                )
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
            self.c[C_qpip_pp][i] = (
                ppe * self.oProtonProton.qpip[i] + self.c[C_qpip_Kpp][i]
            )
            self.c[C_qpip_pg][i] += self.c[C_qpip_K][i]
            self.c[C_qpip][i] = (
                self.c[C_qpip_pg][i]
                + self.c[C_qpip_pp][i]
                + self.oInj.inj_piplus[i]
            )
            self.c[C_qpim_pg][i] = (
                pge * self.oPhotoPion.qpim[i] + self.c[C_qpim_K][i]
            )
            self.c[C_qpim_pp][i] = (
                ppe * self.oProtonProton.qpim[i] + self.c[C_qpim_Kpp][i]
            )
            self.c[C_qpim][i] = (
                self.c[C_qpim_pg][i]
                + self.c[C_qpim_pp][i]
                + self.oInj.inj_piminus[i]
            )
            # the photo-hadronic and pp components hold the pions made in the
            # interactions; those of the kaon decays are P_pion_*_Kpg / _Kpp
            self.c[C_qpip_pg][i] = pge * self.oPhotoPion.qpip[i]
            self.c[C_qpip_pp][i] = ppe * self.oProtonProton.qpip[i]
            self.c[C_qpim_pg][i] = pge * self.oPhotoPion.qpim[i]
            self.c[C_qpim_pp][i] = ppe * self.oProtonProton.qpim[i]
            self.c[C_qpi0_pg][i] = (
                pge * self.oPhotoPion.qpi0[i] + self.c[C_qpi0_K][i]
            )
            # (pi0 from kaon decay always go through pi0 -> gamma gamma)
            self.c[C_qpi0_pp][i] = (
                ppe0 * self.oProtonProton.qpi0[i] + self.c[C_qpi0_Kpp][i]
            )
            self.c[C_qpi0][i] = (
                self.c[C_qpi0_pg][i]
                + self.c[C_qpi0_pp][i]
                + self.oInj.inj_pi0[i]
            )

    def Calc_PI(mut self) raises:
        if self.verbose:
            print(" Calculate pions")
        # AM3 evolves all pion components with the pi+ solver; the identical
        # pi- and component solvers give the same result, in parallel
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
        # the pions of the kaon decays, per kaon origin
        for o in range(2):
            if self.kaon_chain_on(o):
                var p0 = P_pion_plus_Kpg if o == 0 else P_pion_plus_Kpp
                rows.append(
                    Evolution(
                        SA_Pip_Kpg + 2 * o,
                        p0,
                        C_apip,
                        C_Apip,
                        C_qpip_K if o == 0 else C_qpip_Kpp,
                        P_photons_on_had_grid,
                    )
                )
                rows.append(
                    Evolution(
                        SA_Pip_Kpg + 2 * o + 1,
                        p0 + 1,
                        C_apim,
                        C_Apim,
                        C_qpim_K if o == 0 else C_qpim_Kpp,
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

    def _add_kaon_muons(mut self, q0: Int, src: Int):
        """Add the muons from kaon decay c[src .. src + 3] to the block
        c[q0 .. q0 + 3] = (mu+ left, mu+ right, mu- left, mu- right)."""
        for k in range(4):
            for i in range(self.dat.nbins_had):
                self.c[q0 + k][i] += self.c[src + k][i]

    def _zero_block(mut self, q0: Int):
        for k in range(4):
            fill_zero(self.c[q0 + k])

    def Calc_Coeff_MU(mut self):
        """Muon sources from pion decay and (am3_mojo) kaon decay. Without
        pion decay the pion part is zero (AM3 kept the last sources)."""
        if self.verbose:
            print(" Calculate muon coefficient")
        var pid = self.process_pion_decay
        if pid:
            self._pion_to_muon_sources(
                P_pion_plus, P_pion_minus, C_qmupl_tot_dec
            )
        else:
            self._zero_block(C_qmupl_tot_dec)
        self._add_kaon_muons(C_qmupl_tot_dec, C_qmupl_K)
        self._add_kaon_muons(C_qmupl_tot_dec, C_qmupl_Kpp)
        if (
            self.process_parse_sed
            and self.process_photopion
            and self.process_photopion_emission
        ):
            if pid:
                self._pion_to_muon_sources(
                    P_pion_plus_pg, P_pion_minus_pg, C_qmupl_pg
                )
            else:
                self._zero_block(C_qmupl_pg)
        if (
            self.process_parse_sed
            and self.process_pp
            and self.process_pp_emission
        ):
            if pid:
                self._pion_to_muon_sources(
                    P_pion_plus_pp, P_pion_minus_pp, C_qmupl_pp
                )
            else:
                self._zero_block(C_qmupl_pp)
        # the muons of the kaon decays and of the pions they decay into
        for o in range(2):
            if self.kaon_chain_on(o):
                var p0 = P_pion_plus_Kpg if o == 0 else P_pion_plus_Kpp
                var q0 = C_qmu_Kpg if o == 0 else C_qmu_Kpp
                if pid:
                    self._pion_to_muon_sources(p0, p0 + 1, q0)
                else:
                    self._zero_block(q0)
                self._add_kaon_muons(q0, C_qmupl_K if o == 0 else C_qmupl_Kpp)

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
            # each species takes its own external injection (AM3 swapped those
            # of mu+_L and mu-_R: AM3 bug)
            self.c[C_qmuml][i] = (
                self.oInj.inj_muminusleft[i] + self.c[C_qmuml_tot_dec][i]
            )
            self.c[C_qmupl][i] = (
                self.oInj.inj_muplusleft[i] + self.c[C_qmupl_tot_dec][i]
            )
            self.c[C_qmumr][i] = (
                self.oInj.inj_muminusright[i] + self.c[C_qmumr_tot_dec][i]
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
        for o in range(2):
            if self.kaon_chain_on(o):
                for k in range(4):
                    rows.append(
                        Evolution(
                            (SA_MpL_Kpg if o == 0 else SA_MpL_Kpp) + k,
                            (
                                P_muon_plus_left_Kpg if o
                                == 0 else P_muon_plus_left_Kpp
                            )
                            + k,
                            C_amu,
                            C_Amu,
                            (C_qmu_Kpg if o == 0 else C_qmu_Kpp) + k,
                            P_photons_on_had_grid,
                        )
                    )
        self.evolve_group(rows)

    def _decay_neutrino_sources(mut self, pip: Int, mu0: Int, q0: Int):
        """Neutrino sources from the decay of the pions n[pip], n[pip + 1] and
        the muons n[mu0 .. mu0+3] into c[q0 .. q0+3] = (nu_e, anti-nu_e, nu_mu,
        anti-nu_mu), recomputed every step (AM3 never resets the total nu_mu
        sources with pion decay off and muon decay on)."""
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

    def _add_kaon_neutrinos(mut self, q0: Int, src: Int):
        """Add the neutrinos from kaon decay c[src .. src + 3] to the block
        c[q0 .. q0 + 3] = (nu_e, anti-nu_e, nu_mu, anti-nu_mu)."""
        for k in range(4):
            for i in range(self.dat.nbins_nu):
                self.c[q0 + k][i] += self.c[src + k][i]

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
        if self.neutron_decay_on():
            for i in range(self.dat.nbins_nu):
                self.c[C_qnuea_tot_dec][i] += self.c[C_qnuea_ndec][i]
        self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_K)
        self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_Kpp)
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
        # the neutrinos of the kaon decays and of their pions and muons
        for o in range(2):
            if self.kaon_chain_on(o):
                var q0 = C_qnu_Kpg if o == 0 else C_qnu_Kpp
                self._decay_neutrino_sources(
                    P_pion_plus_Kpg if o == 0 else P_pion_plus_Kpp,
                    P_muon_plus_left_Kpg if o == 0 else P_muon_plus_left_Kpp,
                    q0,
                )
                self._add_kaon_neutrinos(q0, C_qnue_K if o == 0 else C_qnue_Kpp)

    def Pass_Coeff_NU(mut self):
        if self.verbose:
            print("Pass neutrino coefficient")
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        for i in range(self.dat.nbins_nu):
            self.c[C_Anu][i] = self.oExtraCool.AExtraCoolNU[i]
            self.c[C_anu][i] = esc * self.oEscape.afescNU[i] + exh
            # each flavour takes its own external injection (AM3 added that of
            # nu_e to all four and ignored the other three: AM3 bug)
            self.c[C_qnue][i] = self.c[C_qnue_tot_dec][i] + self.oInj.inj_nue[i]
            self.c[C_qnuea][i] = (
                self.c[C_qnuea_tot_dec][i] + self.oInj.inj_antinue[i]
            )
            self.c[C_qnum][i] = (
                self.c[C_qnum_tot_dec][i] + self.oInj.inj_numu[i]
            )
            self.c[C_qnuma][i] = (
                self.c[C_qnuma_tot_dec][i] + self.oInj.inj_antinumu[i]
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
        # the anti-nu_e from neutron decay (also after the process is switched
        # off, so that the component escapes like the neutrinos it is part of)
        if self.process_parse_sed and (
            self.neutron_decay_on() or self._nonzero(P_neutrino_el_anti_ndec)
        ):
            rows.append(
                Evolution(
                    SA_nueA_ndec,
                    P_neutrino_el_anti_ndec,
                    C_anu,
                    C_Anu,
                    C_qnuea_ndec,
                    P_photons_on_had_grid,
                )
            )
        # the neutrinos of the kaon decays, per kaon origin
        for o in range(2):
            if self.kaon_chain_on(o):
                for k in range(4):
                    rows.append(
                        Evolution(
                            (SA_nue_Kpg if o == 0 else SA_nue_Kpp) + k,
                            (P_neutrino_el_Kpg if o == 0 else P_neutrino_el_Kpp)
                            + k,
                            C_anu,
                            C_Anu,
                            (C_qnu_Kpg if o == 0 else C_qnu_Kpp) + k,
                            P_photons_on_had_grid,
                        )
                    )
        self.evolve_group(rows)

    def pair_annihilation_active(self) -> Bool:
        """e+ e- annihilation needs electrons and positrons tracked apart."""
        return (
            self.process_pair_annihilation
            and not self.process_merge_positrons_into_electrons
        )

    def triplet_active(self) -> Bool:
        """Triplet pair production acts: switched on and its kernels built at
        init_kernels."""
        return self.process_triplet_pair_production and self.oTPP.ready(
            self.dat
        )

    def Calc_Coeff_Lep(mut self):
        if self.verbose:
            print(" Calculate electron coefficient")
        var t0 = self._tic()
        if self.pair_annihilation_active():
            self.oPairAnn.Calc_PairAnn(
                self.dat, self.dat.n[P_electrons], self.dat.n[P_positrons]
            )
        else:
            self.oPairAnn.clear_internal_arrays(self.dat)
        self._toc(T_PAIR_ANN, t0)
        t0 = self._tic()
        if self.triplet_active():
            # e+- energy-loss rates and photon-convolved product rows for the
            # photons and the e+- at the start of the step
            if self.process_merge_positrons_into_electrons:
                self.oTPP.Calc_Triplet(
                    self.dat, self.dat.n[P_photons], self.dat.n[P_pairs], Vec()
                )
            else:
                self.oTPP.Calc_Triplet(
                    self.dat,
                    self.dat.n[P_photons],
                    self.dat.n[P_electrons],
                    self.dat.n[P_positrons],
                )
        else:
            self.oTPP.clear_internal_arrays(self.dat)
        self._toc(T_TRIPLET, t0)
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
            # (muons come from pion decay and, with kaons, from kaon decay)
            var both_decays = self.process_muon_decay and (
                self.process_pion_decay or self.kaon_decay_on()
            )
            if self.process_parse_sed:
                if self.verbose:
                    print("computing muon decay to pairs terms for components")
                # (the pg and pp pair components take the pairs of the muons
                #  made in the interactions and of the kaon-decay muons)
                if (
                    self.process_photopion
                    and self.process_photopion_emission
                    and both_decays
                ):
                    self._muon_to_pair_sources(
                        P_muon_plus_left_pg, C_qepg, C_qppg, 1.0
                    )
                    if self.kaon_chain_on(0):
                        self._muon_to_pair_sources(
                            P_muon_plus_left_Kpg, C_qepg, C_qppg, 1.0, True
                        )
                if self.process_pp and self.process_pp_emission and both_decays:
                    self._muon_to_pair_sources(
                        P_muon_plus_left_pp, C_qepp, C_qppp, 1.0
                    )
                    if self.kaon_chain_on(1):
                        self._muon_to_pair_sources(
                            P_muon_plus_left_Kpp, C_qepp, C_qppp, 1.0, True
                        )
            # the totals from all muons, also with the components on (AM3 summed
            # the components, which miss injected and set particles: AM3 bug)
            if self.verbose:
                print("computing muon decay to pairs terms total")
            self._muon_to_pair_sources(
                P_muon_plus_left,
                C_qe_tot_dec,
                C_qp_tot_dec,
                fbool(self.process_muon_decay),
            )

    def _muon_to_pair_sources(
        mut self,
        mu0: Int,
        qe: Int,
        qp: Int,
        factor: Float64,
        accumulate: Bool = False,
    ):
        """e-/e+ sources on the lepton grid from the decay of the muons
        n[mu0 .. mu0+3], stored in c[qe], c[qp] or (accumulate) added to them.
        If muon decay is off, the decay is skipped and its last result used."""
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
            var ve = factor / Rate_SSC_CGS * self.dat.grid_had2lep(i, e, 0, nl)
            var vp = factor / Rate_SSC_CGS * self.dat.grid_had2lep(i, p, 0, nl)
            self.c[qe][i] = ve if not accumulate else self.c[qe][i] + ve
            self.c[qp][i] = vp if not accumulate else self.c[qp][i] + vp

    def Pass_Coeff_Lep(mut self):
        if self.verbose:
            print("Pass electron/positron coefficient")
        var nl = self.dat.nbins_lep
        var bh_on = (
            self.process_hadronic
            and self.process_bethe_heitler
            and self.process_bethe_heitler_emission
        )
        var mu_dec = self.process_muon_decay and (
            self.process_pion_decay or self.kaon_decay_on()
        )
        var pg_on = (
            self.process_hadronic
            and self.process_photopion
            and self.process_photopion_emission
            and mu_dec
        )
        var pp_on = (
            self.process_hadronic
            and self.process_pp
            and self.process_pp_emission
            and mu_dec
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
            # e+- from kaon decay (photo-hadronic and pp origin)
            var kon = fbool(self.kaon_decay_on())
            var r = kon / Rate_SSC_CGS
            var ke_g = r * self.dat.grid_had2lep(i, pd(self.c[C_qe_K]), 0, nl)
            var kp_g = r * self.dat.grid_had2lep(i, pd(self.c[C_qp_K]), 0, nl)
            var ke_p = r * self.dat.grid_had2lep(i, pd(self.c[C_qe_Kpp]), 0, nl)
            var kp_p = r * self.dat.grid_had2lep(i, pd(self.c[C_qp_Kpp]), 0, nl)
            var ke = ke_g + ke_p
            var kp = kp_g + kp_p
            if self.process_parse_sed:
                self.c[C_qepg][i] += ke_g
                self.c[C_qppg][i] += kp_g
                self.c[C_qepp][i] += ke_p
                self.c[C_qppp][i] += kp_p
            self.c[C_qe_tot_dec][i] += ke
            self.c[C_qp_tot_dec][i] += kp
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiLep(
            self.rp
        )
        var esc = fbool(self.process_escape)
        var exl = fbool(self.process_expansion) * Expansion.afExpLep(self.rp)
        var ann = fbool(
            self.process_annihilation and self.process_annihilation_emission
        )
        var pac = fbool(
            self.pair_annihilation_active()
            and self.process_pair_annihilation_cooling
        )
        # triplet pair production (am3_mojo addition): the e+- lose the energy
        # the pairs take as a continuous loss (cooling switch)
        var tpp = self.triplet_active()
        var tco = fbool(tpp and self.process_triplet_pair_production_cooling)
        # neutron decay (am3_mojo addition): its electrons
        var ndec = self.neutron_decay_on()
        for i in range(nl):
            self.c[C_Ae][i] = adi + self.oExtraCool.AExtraCoolE[i]
            if tpp:
                self.c[C_Ae][i] -= tco * self.oTPP.eloss[i]
            self.c[C_ae][i] = esc * self.oEscape.afescE[i] + exl
            # e+e- annihilation (am3_mojo addition): electrons disappear at
            # the rate set by the positrons and vice versa
            self.c[C_ae_ele][i] = self.c[C_ae][i] + pac * self.oPairAnn.a_ele[i]
            self.c[C_ae_pos][i] = self.c[C_ae][i] + pac * self.oPairAnn.a_pos[i]
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
            if ndec:
                self.c[C_qele][i] += self.c[C_qe_ndec][i]
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
        var t0 = self._tic()
        self._triplet_sources(tpp)
        self._toc_add(T_TRIPLET, t0)

    def _triplet_sources(mut self, tpp: Bool):
        """The pairs that triplet pair production makes, from the leptons at
        the start of the step: sources of the e- and e+ (or merged e+-) totals
        and of the triplet pair component."""
        var nl = self.dat.nbins_lep
        fill_zero(self.c[C_qetpp])
        fill_zero(self.c[C_qptpp])
        fill_zero(self.c[C_qeptpp])
        if not tpp:
            return
        var tem = fbool(self.process_triplet_pair_production_emission)
        if self.process_merge_positrons_into_electrons:
            self.oTPP.add_pair_sources(
                pd(self.c[C_qeptpp]),
                pd(self.c[C_qptpp]),
                self.dat.n[P_pairs],
                Vec(),
                tem,
            )
            for i in range(nl):
                self.c[C_qep][i] += self.c[C_qeptpp][i]
            return
        self.oTPP.add_pair_sources(
            pd(self.c[C_qetpp]),
            pd(self.c[C_qptpp]),
            self.dat.n[P_electrons],
            self.dat.n[P_positrons],
            tem,
        )
        for i in range(nl):
            self.c[C_qeptpp][i] = self.c[C_qetpp][i] + self.c[C_qptpp][i]
            self.c[C_qele][i] += self.c[C_qetpp][i]
            self.c[C_qpos][i] += self.c[C_qptpp][i]

    def Calc_Lep(mut self) raises:
        if self.verbose:
            print(" Calculate electrons/positrons")
        var rows = List[Evolution]()
        # with e+e- annihilation, e- and e+ have their own sinks (C_ae_ele,
        # C_ae_pos), and the pair components are evolved in e-/e+ halves
        var ann = self.pair_annihilation_active()
        var rows_e = List[Evolution]()
        var rows_p = List[Evolution]()
        var comps = List[Int]()  # pair components evolved in halves

        def lep(solver: Int, target: Int, q: Int) capturing:
            rows.append(Evolution(solver, target, C_ae, C_Ae, q, P_photons))

        var tpp = self.triplet_active()

        def comp(
            solver: Int, k: Int, target: Int, q: Int, qe: Int, qp: Int
        ) capturing:
            if not ann:
                lep(solver, target, q)
                return
            comps.append(k)
            rows_e.append(
                Evolution(
                    solver, P_pairs_in_ele + k, C_ae_ele, C_Ae, qe, P_photons
                )
            )
            rows_p.append(
                Evolution(
                    solver, P_pairs_in_pos + k, C_ae_pos, C_Ae, qp, P_photons
                )
            )

        if self.process_merge_positrons_into_electrons:
            lep(SA_ep, P_pairs, C_qep)
        else:
            rows.append(
                Evolution(
                    SA_e,
                    P_electrons,
                    C_ae_ele if ann else C_ae,
                    C_Ae,
                    C_qele,
                    P_photons,
                )
            )
            rows.append(
                Evolution(
                    SA_p,
                    P_positrons,
                    C_ae_pos if ann else C_ae,
                    C_Ae,
                    C_qpos,
                    P_photons,
                )
            )
        if self.process_parse_sed:
            # (the photo-hadronic and pp pairs come from muon decay and, with
            #  kaons, also directly from kaon decay)
            var dec = self.process_muon_decay or self.kaon_decay_on()
            comp(SA_epin, 0, P_pairs_in, C_qepin, C_qein, C_qpin)
            if self.process_annihilation and self.process_annihilation_emission:
                comp(SA_eppair, 2, P_pairs_pair, C_qeppair, C_qepair, C_qppair)
            if (
                self.process_bethe_heitler
                and self.process_bethe_heitler_emission
            ):
                comp(SA_epbh, 1, P_pairs_bh, C_qepbh, C_qebh, C_qpbh)
            if dec:
                comp(SA_eppg, 3, P_pairs_pg, C_qeppg, C_qepg, C_qppg)
            if self.process_pp and self.process_pp_emission and dec:
                comp(SA_eppp, 4, P_pairs_pp, C_qeppp, C_qepp, C_qppp)
            # (also after the process is switched off, so that the component
            #  escapes and cools like the pairs it is part of)
            if (
                tpp and self.process_triplet_pair_production_emission
            ) or self._nonzero(P_pairs_tpp):
                comp(SA_eptpp, 5, P_pairs_tpp, C_qeptpp, C_qetpp, C_qptpp)
            # the electrons from neutron decay: electrons only, so evolved
            # whole, with the electrons' sink
            if self.neutron_decay_on() or self._nonzero(P_pairs_ndec):
                rows.append(
                    Evolution(
                        SA_endec,
                        P_pairs_ndec,
                        C_ae_ele if ann else C_ae,
                        C_Ae,
                        C_qe_ndec,
                        P_photons,
                    )
                )
        if len(comps) > 0 and not self.pair_halves_valid:
            self._split_pair_components()
        self.evolve_group(rows)
        if len(comps) > 0:
            self.evolve_group(rows_e)
            self.evolve_group(rows_p)
        if len(comps) > 0:
            for k in comps:
                var t = self._pair_component(k)
                for i in range(self.dat.nbins_lep):
                    self.dat.n[t][i] = (
                        self.dat.n[P_pairs_in_ele + k][i]
                        + self.dat.n[P_pairs_in_pos + k][i]
                    )
        # the halves follow the components only while they evolve apart
        self.pair_halves_valid = ann
        if not self.process_merge_positrons_into_electrons:
            for i in range(self.dat.nbins_lep):
                self.dat.n[P_pairs][i] = (
                    self.dat.n[P_electrons][i] + self.dat.n[P_positrons][i]
                )

    @staticmethod
    def _pair_component(k: Int) -> Int:
        """Density array of pair component k (P_pairs_in_ele order)."""
        if k == 0:
            return P_pairs_in
        elif k == 1:
            return P_pairs_bh
        elif k == 2:
            return P_pairs_pair
        elif k == 5:
            return P_pairs_tpp
        return P_pairs_pg if k == 3 else P_pairs_pp

    def _split_pair_components(mut self):
        """Split the pair components into e- and e+ parts with the charge
        fractions of all pairs in each bin (when annihilation starts acting)."""
        for k in range(N_PAIR_COMP):
            var t = Self._pair_component(k)
            for i in range(self.dat.nbins_lep):
                var ne = self.dat.n[P_electrons][i]
                var npo = self.dat.n[P_positrons][i]
                var fe = ne / (ne + npo) if ne + npo > 0.0 else 0.5
                self.dat.n[P_pairs_in_ele + k][i] = fe * self.dat.n[t][i]
                self.dat.n[P_pairs_in_pos + k][i] = (1.0 - fe) * self.dat.n[t][
                    i
                ]
        self.pair_halves_valid = True

    def _lep_ins(self) -> Array[FPtr, N_LEP_IN]:
        var _r: Array[FPtr, N_LEP_IN] = [
            pd(self.dat.n[P_pairs]),
            pd(self.dat.n[P_pairs_in]),
            pd(self.dat.n[P_pairs_bh]),
            pd(self.dat.n[P_pairs_pair]),
            pd(self.dat.n[P_pairs_pg]),
            pd(self.dat.n[P_pairs_pp]),
            pd(self.dat.n[P_pairs_tpp]),
            pd(self.dat.n[P_pairs_ndec]),
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
            var outs: Array[FPtr, N_LEP_IN] = [
                pd(self.c[C_qgma_sy]),
                pd(self.c[C_qgma_insy]),
                pd(self.c[C_qgma_bhsy]),
                pd(self.c[C_qgma_pairsy]),
                pd(self.c[C_qgma_pgsy]),
                pd(self.c[C_qgma_ppsy]),
                pd(self.c[C_qgma_tppsy]),
                pd(self.c[C_qgma_ndecsy]),
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
            var outs: Array[FPtr, N_LEP_IN] = [
                pd(self.c[C_qgma_ic]),
                pd(self.c[C_qgma_inic]),
                pd(self.c[C_qgma_bhic]),
                pd(self.c[C_qgma_pairic]),
                pd(self.c[C_qgma_pgic]),
                pd(self.c[C_qgma_ppic]),
                pd(self.c[C_qgma_tppic]),
                pd(self.c[C_qgma_ndecic]),
            ]
            self.oComp.Calc_IC_Phot_Rad(
                self.dat,
                outs,
                self._lep_ins(),
                not self.process_parse_sed,
                self.optimize_compton_emission_emin,
                self.optimize_compton_target_emax,
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
            if (
                self.kaons_on()
                and self.process_kaon_syn
                and self.process_kaon_syn_emission
            ):
                self.oSynP.Calc_SY_K(self.dat)
            else:
                fill_zero(self.oSynP.epsilonsyK)
            self._toc(T_KAON_SYN, t0)
            t0 = self._tic()
            if (
                self.kaons_on()
                and self.process_kaon_compton
                and self.process_kaon_compton_emission
            ):
                self.oHadIC.Calc_KIC(self.dat)
            else:
                fill_zero(self.oHadIC.epsilonKIC)
            self._toc(T_KAON_IC, t0)
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
                # the direct pp photons (non-zero only when the pp pi0 do not
                # go into the cascade), for the total
                var direct_pp = self.c[C_qgma_pi0_pp].copy()
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
                        # C_qpi0_pp holds the pp pi0 (into the cascade) and
                        # the pi0 from the decay of kaons from pp collisions
                        if (
                            self.process_pp_emission_pi0_into_cascade
                            or self.kaons_pp_on()
                        ):
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
                # all pi0 (pg, pp and injected) plus the direct pp photons, with
                # or without the components (AM3 summed the components, missing
                # injected pi0, and dropped the direct pp photons without them)
                self.oPionDec.Calc_PItoG(self.dat, self.c[C_qpi0])
                var qg = pd(self.oPionDec.QPI2GGen)
                for i in range(ng):
                    self.c[C_qgma_pi0][i] = (
                        1.0 / Rate_SSC_CGS * self.dat.grid_had2lep(i, qg, 0, ng)
                    ) + direct_pp[i]
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
            var e_psy = pd(self.oSynP.epsilonsyP)
            var e_pic = pd(self.oHadIC.epsilonPIC)
            var e_pisy = pd(self.oSynP.epsilonsyPi)
            var e_piic = pd(self.oHadIC.epsilonPiIC)
            var e_musy = pd(self.oSynP.epsilonsyMu)
            var e_muic = pd(self.oHadIC.epsilonMuIC)
            var e_ksy = pd(self.oSynP.epsilonsyK)
            var e_kic = pd(self.oHadIC.epsilonKIC)
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
                self.c[C_qgma_ksy][i] = r * self.dat.grid_had2lep(
                    i, e_ksy, 0, ng
                )
                self.c[C_qgma_kic][i] = r * self.dat.grid_had2lep(
                    i, e_kic, 0, ng
                )
                self.c[C_qgma_had][i] = (
                    self.c[C_qgma_pi0][i]
                    + self.c[C_qgma_psy][i]
                    + self.c[C_qgma_pic][i]
                    + self.c[C_qgma_pisy][i]
                    + self.c[C_qgma_piic][i]
                    + self.c[C_qgma_musy][i]
                    + self.c[C_qgma_muic][i]
                    + self.c[C_qgma_ksy][i]
                    + self.c[C_qgma_kic][i]
                )
                self.c[C_agma_pg][i] = pgl * self.dat.grid_had2lep(
                    i, e_pg, 0, ng
                )
        else:
            for i in range(ng):
                self.c[C_qgma_had][i] = 0.0
                self.c[C_agma_pg][i] = 0.0
                self.c[C_qgma_ksy][i] = 0.0
                self.c[C_qgma_kic][i] = 0.0
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
        var pae = fbool(
            self.pair_annihilation_active()
            and self.process_pair_annihilation_emission
        )
        for i in range(ng):
            self.c[C_qgma_pann][i] = pae * self.oPairAnn.q_pho[i]
        # triplet pair production (am3_mojo addition): the photons absorbed by
        # the e+- as they are now, like the inverse-Compton photon loss
        var tpa = fbool(self.triplet_active())
        if tpa != 0.0:
            var t0 = self._tic()
            if self.process_merge_positrons_into_electrons:
                self.oTPP.photon_sink(self.dat, self.dat.n[P_pairs], Vec())
            else:
                self.oTPP.photon_sink(
                    self.dat, self.dat.n[P_electrons], self.dat.n[P_positrons]
                )
            self._toc_add(T_TRIPLET, t0)
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
            if tpa != 0.0:
                self.c[C_agma][i] += tpa * self.oTPP.alpha_g[i]
            self.c[C_qgma][i] = (
                self.c[C_qgma_sy][i]
                + self.c[C_qgma_ic][i]
                + self.c[C_qgma_had][i]
                + self.c[C_qgma_inj][i]
                + self.c[C_qgma_pann][i]
            )
            self.c[C_qgma1][i] = (
                self.c[C_qgma_sy][i]
                + self.c[C_qgma_ic][i]
                + self.c[C_qgma_had][i]
                + self.c[C_qgma_pann][i]
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

        def comp(solver: Int, target: Int, q: Int) capturing:
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
            # (also after the process is switched off, so that the component
            #  escapes like the photons it is part of)
            if self.pair_annihilation_active() or self._nonzero(
                P_photons_pair_ann
            ):
                comp(SA_gpann, P_photons_pair_ann, C_qgma_pann)
            if (
                self.triplet_active()
                or self._nonzero(P_photons_tppsy)
                or self._nonzero(P_photons_tppic)
            ):
                comp(SA_gtppsy, P_photons_tppsy, C_qgma_tppsy)
                comp(SA_gtppic, P_photons_tppic, C_qgma_tppic)
            if (
                self.neutron_decay_on()
                or self._nonzero(P_photons_ndecsy)
                or self._nonzero(P_photons_ndecic)
            ):
                comp(SA_gndecsy, P_photons_ndecsy, C_qgma_ndecsy)
                comp(SA_gndecic, P_photons_ndecic, C_qgma_ndecic)
            if self.kaons_on() or self._nonzero(P_photons_kasy):
                comp(SA_gksy, P_photons_kasy, C_qgma_ksy)
            if self.kaons_on() or self._nonzero(P_photons_kaic):
                comp(SA_gkic, P_photons_kaic, C_qgma_kic)
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
                self.process_kaons
                and self.process_kaon_syn
                and self.process_kaon_syn_emission,
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
        # the solver that evolves the e+-: SA_ep when the positrons are merged
        # (AM3 used SA_e, which is then never evolved: no cooling, AM3 bug)
        var k = SA_ep if self.process_merge_positrons_into_electrons else SA_e
        for i in range(nl):
            total_loss[i] = -sy * self.S[k].Asy[i] - ic * self.S[k].Aic[i] - adi
        var imax = 0
        while (
            imax < nl
            and self.oAcc.AAClepBohm(self.rp, self.dat, imax)
            >= total_loss[imax]
        ):
            imax += 1
        # if acceleration wins on the whole grid, the C++ code reads one element
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
