"""Port of SimulationManager (SimulationManager.cc)."""

from std.time import perf_counter_ns
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.physics.acceleration import Acc
from am3_mojo.physics.bethe_heitler import BetheHeitler
from am3_mojo.physics.decays import (
    PionDecay,
    MuonDecay,
    HadIC,
    NeutronDecay,
    kaon_mass,
    kaon_lifetime,
    kaon_name,
    PR_PIP,
    PR_PIM,
    PR_PI0,
    PR_MUPL,
    PR_MUPR,
    PR_MUML,
    PR_MUMR,
    PR_EM,
    PR_EP,
    PR_NUE,
    PR_NUEB,
    PR_NUM,
    PR_NUMB,
    N_PROD,
    KaonDecay,
    charm_mass,
    charm_lifetime,
    charm_charged,
    charm_name,
    CP_PIP,
    CP_PIM,
    CP_PI0,
    CP_MUPL,
    CP_MUPR,
    CP_MUML,
    CP_MUMR,
    CP_KP,
    CP_P,
    CP_N,
    CP_EM,
    CP_EP,
    CP_NUE,
    CP_NUEB,
    CP_NUM,
    CP_NUMB,
    CP_NUT,
    CP_NUTB,
    CP_GAM,
    N_CPROD,
    CharmDecay,
)
from am3_mojo.physics.escape import Escape
from am3_mojo.physics.expansion import Expansion
from am3_mojo.physics.extra_cooling import ExtraCooling
from am3_mojo.physics.injection import Injection
from am3_mojo.physics.inverse_compton import Comp
from am3_mojo.physics.pair_annihilation import PairAnnihilation
from am3_mojo.physics.pair_production import Pair
from am3_mojo.physics.photohadron import PhotoHadron
from am3_mojo.physics.proton_proton import PP_E_TRANS_NEGATIVE, ProtonProton
from am3_mojo.physics.synchrotron import Syn, SynP
from am3_mojo.physics.triplet import TripletPairProduction
from am3_mojo.simulation.coeffs import *
from am3_mojo.simulation.evolve import (
    vp,
    Evolution,
    EvolveJob,
    run_jobs,
    CoolJob,
    run_cooling,
)
from am3_mojo.solver.ids import *
from am3_mojo.solver.solver import SolverAlgorithm
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.parallel import default_num_threads, parallel_tasks

# a species that does not decay gets this mean life [s] (AM3's t_nodecay_s),
# and the neutrinos and photons this mass [GeV] (AM3's mass_nu and mass_g)
comptime T_NO_DECAY = 1.0e7
comptime MASS_NONE = 1e-100
# the largest center-of-momentum energy bin of the full gamma-gamma pair
# kernel (AM3's member annihilation_pair_emission_max_com_energy, which nothing
# changes)
comptime ANNIHILATION_PAIR_EMISSION_MAX_COM_ENERGY = 138

# profiler slots: 0-35 are AM3's stages (get_profiled_time_labels), and the
# rest are the stages of the processes that am3_mojo adds
comptime T_PAIR_ANN = 36
comptime T_TRIPLET = 37
comptime T_NEUTRON_DECAY = 38
comptime T_KAON_COEFF = 39
comptime T_KAON_EVOLVE = 40
comptime T_KAON_SYN = 41
comptime T_KAON_IC = 42
comptime T_CHARM_COEFF = 43
comptime T_CHARM_EVOLVE = 44
comptime T_CHARM_SYN = 45
comptime T_CHARM_IC = 46
comptime N_TIMING = 47

# The numbers of decay products in the coefficient blocks of coeffs.mojo, in the
# order of physics/decays.mojo. Calc_C asserts the layout of the blocks.
comptime N_KPROD_HAD = PR_MUMR - PR_PIP + 1
comptime N_KPROD_NU = PR_NUMB - PR_NUE + 1
comptime N_CPROD_NU = CP_NUTB - CP_NUE + 1


def _phot_part_work(m: Pointer[SimulationManager, MutUntrackedOrigin], k: Int):
    m[]._phot_part(m[].phot_parts[k])


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
    var oPhotoHadron: PhotoHadron
    var oProtonProton: ProtonProton
    var oPionDec: PionDecay
    var oMuonDec: MuonDecay
    var oPairAnn: PairAnnihilation
    var oTPP: TripletPairProduction
    var oNeutronDec: NeutronDecay
    var oKaonDec: KaonDecay
    var oCharmDec: CharmDecay
    var oEscape: Escape
    var oInj: Injection
    var oExtraCool: ExtraCooling

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
    var process_charm: Bool
    var process_charm_decay: Bool
    var process_charm_syn: Bool
    var process_charm_syn_emission: Bool
    var process_charm_syn_cooling: Bool
    var process_charm_compton: Bool
    var process_charm_compton_emission: Bool
    var process_charm_compton_cooling: Bool
    var verbose: Bool

    var optimize_compton_emission_emin: Int
    var optimize_compton_target_emax: Int
    # declared `bool` in the C++ code: any non-zero value is stored as true (=1)
    var optimize_compton_emission_grid: Bool
    var optimize_compton_target_grid: Bool
    var optimize_annihilation_pair_emission: Bool
    var optimize_bethe_heitler_outgoing_pairs_grid: Bool
    var optimize_bethe_heitler_incoming_protons_min: Int
    var optimize_bethe_heitler_target_photon_max: Int
    var optimize_photopion_target_photon_grid: Bool
    var optimize_photopion_target_photon_max: Int
    # The energies [eV] that the setters of the five limits above asked for.
    # update_energy_grid takes the limits of the new grid from them.
    var asked_compton_emission_emin: Float64
    var asked_compton_target_emax: Float64
    var asked_bethe_heitler_incoming_protons_min: Float64
    var asked_bethe_heitler_target_photon_max: Float64
    var asked_photopion_target_photon_max: Float64
    var pp_model_charged_pions: Int
    var pp_model_neutral_pions: Int
    var pp_transition_energy_to_delta_approx: Float64
    var step_counter: Int
    var solver_simpleqintegration: Bool
    var solver_threshold_cool_dom: Float64
    var solver_threshold_esc_dom: Float64
    var solver_threshold_matrix: Float64

    var S: List[SolverAlgorithm]

    var c: List[Vec]

    var profile_timing: Bool
    var pg_conv_fresh: Bool
    var kc_idle: Bool
    var kc_zeroed: Bool
    var pi0_pp_direct: Vec
    var kernels_ready: Bool
    var pair_halves_valid: Bool
    var components_stale: Bool
    var last_step_merged: Bool
    var last_step_hadronic: Bool
    var kchain: List[Bool]
    var kchain_live: List[Bool]
    var cchain: List[Bool]
    var cchain_live: List[Bool]
    var ckaons: List[Bool]
    var ckaons_live: List[Bool]
    var kaons_ev: List[Bool]
    var kaons_live: List[Bool]
    var charm_ev: List[Bool]
    var charm_live: List[Bool]
    var pchain: List[Bool]
    var pchain_live: List[Bool]
    var syn_built: Bool
    var syn_field: Float64
    var syn_qsyn: Bool
    var synp_built: Bool
    var synp_field: Float64
    var hadronic_built: Bool
    var ic_built: Bool
    var bh_built: Bool
    # the version that the photo-hadronic and pp kernels were built on
    # (-1: none)
    var grid_gen: Int
    var pg_grid: Int
    var pp_grid: Int
    var pp_built_e_trans: Float64
    var pp_built_charged: Int
    var pp_built_neutral: Int
    var pg_built_photon_max: Int
    var bh_built_protons_min: Int
    var bh_built_photon_max: Int
    var ic_built_emin: Int
    var ic_built_photon_max: Int
    var timing: List[Float64]
    var nthreads: Int
    var fresh: List[Int]
    var fresh_nphot: List[Int]
    var phot_parts: List[Int]

    def __init__(out self):
        self.rp = RunParams()
        self.dat = AM3Arrays()
        self.oAcc = Acc()
        self.oPair = Pair(self.rp, self.dat, False)
        self.oSyn = Syn(self.rp, self.dat, False)
        self.oComp = Comp(self.dat, False)
        self.oSynP = SynP(self.rp, self.dat, False)
        self.oHadIC = HadIC(self.dat, False)
        self.oBH = BetheHeitler(self.dat, False)
        self.oPhotoHadron = PhotoHadron(self.dat, False)
        self.oProtonProton = ProtonProton(self.dat, False)
        self.oPionDec = PionDecay(self.dat, False)
        self.oMuonDec = MuonDecay(self.dat, False)
        self.oPairAnn = PairAnnihilation(self.dat, False)
        self.oTPP = TripletPairProduction(self.dat, False)
        self.oNeutronDec = NeutronDecay(self.dat, False)
        self.oKaonDec = KaonDecay(self.dat, False)
        self.oCharmDec = CharmDecay(self.dat, False)
        self.oEscape = Escape(self.dat, False)
        self.oInj = Injection(self.dat, False)
        self.oExtraCool = ExtraCooling(self.dat)
        self.nthreads = default_num_threads()
        self.fresh = List[Int]()
        self.phot_parts = List[Int]()
        self.fresh_nphot = List[Int]()
        self.oComp.nthreads = self.nthreads
        self.oBH.nthreads = self.nthreads
        self.oSyn.nthreads = self.nthreads
        self.oPair.nthreads = self.nthreads
        self.oPhotoHadron.nthreads = self.nthreads
        self.oPairAnn.nthreads = self.nthreads
        self.oTPP.nthreads = self.nthreads
        self.oNeutronDec.nthreads = self.nthreads
        self.oKaonDec.nthreads = self.nthreads
        self.oCharmDec.nthreads = self.nthreads
        self.oProtonProton.nthreads = self.nthreads
        self.oSynP.nthreads = self.nthreads
        self.oHadIC.nthreads = self.nthreads
        self.oPionDec.nthreads = self.nthreads
        self.oMuonDec.nthreads = self.nthreads

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
        self.asked_compton_emission_emin = (
            self.dat.egrid_pho[self.optimize_compton_emission_emin] * MEC2_EV
        )
        self.asked_compton_target_emax = (
            self.dat.egrid_pho[self.optimize_compton_target_emax] * MEC2_EV
        )
        self.asked_bethe_heitler_incoming_protons_min = (
            self.dat.egrid_had[self.optimize_bethe_heitler_incoming_protons_min]
            * 1e9
        )
        self.asked_bethe_heitler_target_photon_max = (
            self.dat.egrid_pho[self.optimize_bethe_heitler_target_photon_max]
            * MEC2_EV
        )
        self.asked_photopion_target_photon_max = (
            self.dat.egrid_pho[self.optimize_photopion_target_photon_max]
            * MEC2_EV
        )
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
        self.process_charm = True
        self.process_charm_decay = True
        self.process_charm_syn = True
        self.process_charm_syn_emission = True
        self.process_charm_syn_cooling = True
        self.process_charm_compton = True
        self.process_charm_compton_emission = True
        self.process_charm_compton_cooling = True
        self.verbose = False

        self.c = List[Vec](capacity=C_COUNT)
        for _ in range(C_COUNT):
            self.c.append(Vec())
        self.S = List[SolverAlgorithm]()
        self.step_counter = 0
        self.profile_timing = False
        self.pg_conv_fresh = False
        self.kc_idle = False
        self.kc_zeroed = False
        self.pi0_pp_direct = Vec()
        self.kernels_ready = False
        self.hadronic_built = False
        self.ic_built = False
        self.bh_built = False
        self.grid_gen = 0
        self.pg_grid = -1
        self.pp_grid = -1
        self.pp_built_e_trans = 0.0
        self.pp_built_charged = 0
        self.pp_built_neutral = 0
        self.pg_built_photon_max = 0
        self.bh_built_protons_min = 0
        self.bh_built_photon_max = 0
        self.ic_built_emin = 0
        self.ic_built_photon_max = 0
        self.pair_halves_valid = True
        self.components_stale = False
        self.last_step_merged = False
        self.last_step_hadronic = False
        self.kchain = [False, False]
        self.kchain_live = [False, False]
        self.cchain = [False, False]
        self.cchain_live = [False, False]
        self.ckaons = [False, False]
        self.ckaons_live = [False, False]
        self.kaons_ev = [False, False]
        self.kaons_live = [False, False]
        self.charm_ev = [False, False]
        self.charm_live = [False, False]
        self.pchain = [False, False]
        self.pchain_live = [False, False]
        self.syn_built = False
        self.syn_field = 0.0
        self.syn_qsyn = False
        self.synp_built = False
        self.synp_field = 0.0
        self.timing = zeros(N_TIMING)
        self.Initialize_Solver_Algorithms()
        self.reset_coefficients()  # (getters work before init_kernels)

    # --------------------------------------------------------------------------
    def update_verbosity(mut self, v: Bool):
        if v:
            print("updated verbosity to ", 1 if v else 0, sep="")
        self.verbose = v
        self.oBH.verbose = v
        self.oComp.verbose = v
        self.oEscape.verbose = v
        self.oHadIC.verbose = v
        self.oInj.verbose = v
        self.oMuonDec.verbose = v
        self.oPair.verbose = v
        self.oPhotoHadron.verbose = v
        self.oProtonProton.verbose = v
        self.oPionDec.verbose = v
        self.oSyn.verbose = v
        self.oSynP.verbose = v
        self.oPairAnn.verbose = v
        self.oTPP.verbose = v
        self.oNeutronDec.verbose = v
        self.oKaonDec.verbose = v
        self.oCharmDec.verbose = v
        for i in range(len(self.S)):
            self.S[i].verbose = v

    def _check_pp_transition_energy(self) raises:
        """The transition energy of the pp delta approximation must lie on the
        hadron grid. Above the top of the grid, AM3 crashes. Without this
        check, the pions get no energy. The check reads the energy as
        ProtonProton.initialize_kernels takes it: a negative energy stands
        for PP_E_TRANS_NEGATIVE, which can lie above a small grid. NaN also
        fails the check."""
        var etop = self.dat.egrid_had[self.dat.nbins_had - 1]
        var e_trans = self.pp_transition_energy_to_delta_approx
        if e_trans < 0.0:
            e_trans = PP_E_TRANS_NEGATIVE
        if not (e_trans <= etop):
            raise Error(
                String(
                    "the transition energy of the pp delta approximation (",
                    e_trans * 1e9,
                    " eV) is not a number or lies above the hadron grid",
                    " (top: ",
                    etop * 1e9,
                    " eV): set_pp_transition_energy_to_delta_approx() a lower",
                    " value or update_energy_grid() a larger grid",
                )
            )

    def init_kernels(mut self) raises:
        if self.process_hadronic and self.process_pp:
            self._check_pp_transition_energy()
        print("init. AM3 kernels:")
        self.print_process_switches()
        print()
        self.rp.print_params()

        # (the synchrotron kernels are built for the field before their first
        #  use: _build_syn_kernels)
        self.syn_built = False
        self.synp_built = False
        self.oComp.reinitialize_empty_kernels(self.dat)
        if self.process_electron_compton:
            # (The C++ "every 2nd grid point" options are unreachable: the
            #  optimize_*_grid switches are C++ bools, that is, step 1. Both
            #  kernels are built, whatever the emission and photon-loss
            #  switches.)
            self.oComp.initialize_kernels(
                self.dat,
                True,
                self.optimize_compton_emission_emin,
                self.optimize_compton_target_emax,
            )
            self.ic_built_emin = self.optimize_compton_emission_emin
            self.ic_built_photon_max = self.optimize_compton_target_emax
        self.ic_built = self.process_electron_compton
        self.oPair.reinitialize_empty_kernels(
            self.dat, self.optimize_annihilation_pair_emission
        )
        if self.process_annihilation:
            # (both kernels, whatever the cooling and emission switches)
            self.oPair.initialize_kernels(
                self.rp, self.dat, self.optimize_annihilation_pair_emission
            )
        # (whatever the positron merge, which can be switched off after it)
        if self.process_pair_annihilation:
            self.oPairAnn.initialize_kernels(self.dat)
        else:
            self.oPairAnn.free_kernels()  # (no stale kernels of another grid)
        if self.process_triplet_pair_production:
            self.oTPP.initialize_kernels(self.dat)
        else:
            self.oTPP.free_kernels()
        self.bh_built = False
        if self.process_hadronic:
            self.oHadIC.reinitialize_empty_kernels(self.dat)
            # (the step tables are cheap and always built, so that an emission
            #  switched on after init_kernels gets the right one)
            self.oHadIC.initialize_kernels(self.dat)
            if self.process_bethe_heitler:
                self._build_bh_kernels()
            else:
                self.oBH.reinitialize_empty_kernels(self.dat)
            self.oPionDec.initialize_kernels(self.dat)
            self.oMuonDec.initialize_kernels(self.dat)
            if self.process_neutron_decay:
                self.oNeutronDec.initialize_kernels(self.dat)
            else:
                self.oNeutronDec.free_kernels()
            # (the charm decays make kaons: their kernels are built with charm)
            if self.process_kaons or self.process_charm:
                self.oKaonDec.initialize_kernels(self.dat)
            else:
                self.oKaonDec.free_kernels()
            if self.process_charm:
                self.oCharmDec.initialize_kernels(self.dat)
            else:
                self.oCharmDec.free_kernels()
            # (The modules are reset to the current grid, so that no kernel of
            #  an earlier grid survives. With the process off, the kernels of
            #  this grid stay, as in AM3, for the process switched on again.)
            if self.process_photopion:
                self._build_pg_kernels(self.process_kaons, self.process_charm)
            elif self.pg_grid != self.grid_gen:
                self.oPhotoHadron.reinitialize_empty_kernels(self.dat)
                self.pg_grid = -1
            if self.process_pp:
                self._build_pp_kernels(self.process_kaons, self.process_charm)
            elif self.pp_grid != self.grid_gen:
                self.oProtonProton.reinitialize_empty_kernels(self.dat)
                self.pp_grid = -1
        self.clear_am3()
        self.kernels_ready = True
        self.hadronic_built = self.process_hadronic

    def _build_bh_kernels(mut self) raises:
        """Both Bethe-Heitler kernels, whatever the emission and cooling
        switches."""
        self.oBH.initialize_kernels(
            self.dat,
            self.optimize_bethe_heitler_incoming_protons_min,
            self.optimize_bethe_heitler_target_photon_max,
        )
        self.bh_built = True
        self.bh_built_protons_min = (
            self.optimize_bethe_heitler_incoming_protons_min
        )
        self.bh_built_photon_max = self.optimize_bethe_heitler_target_photon_max

    def _build_pg_kernels(mut self, kaons: Bool, charm: Bool):
        """All photo-hadronic kernels, whatever the emission, cooling and
        photon-loss switches. (The "every n-th target photon" option is
        unreachable: the optimize_photopion_target_photon_grid switch is a C++
        bool.)"""
        self.oPhotoHadron.reinitialize_empty_kernels(self.dat)
        self.oPhotoHadron.initialize_kernels(
            self.dat, self.optimize_photopion_target_photon_max, kaons, charm
        )
        self.pg_grid = self.grid_gen
        self.pg_built_photon_max = self.optimize_photopion_target_photon_max

    def _build_pp_kernels(mut self, kaons: Bool, charm: Bool):
        self.oProtonProton.reinitialize_empty_kernels(self.dat)
        self.oProtonProton.initialize_kernels(
            self.dat,
            self.pp_transition_energy_to_delta_approx,
            self.pp_model_charged_pions,
            self.pp_model_neutral_pions,
            kaons,
            charm,
        )
        self.pp_grid = self.grid_gen
        self.pp_built_e_trans = self.pp_transition_energy_to_delta_approx
        self.pp_built_charged = self.pp_model_charged_pions
        self.pp_built_neutral = self.pp_model_neutral_pions

    def _rebuild_changed_kernels(mut self) raises:
        """A setting changed after init_kernels acts as set before it (AM3's
        kernels keep those of init_kernels, while the getters report the new
        ones). Kernels not built stay so. Their process, if switched on after an
        init_kernels without it, makes nothing (pp, Bethe-Heitler, inverse
        Compton, as in AM3). For photo-pion production, evolve_step raises
        instead. A rebuild keeps the kaons and charm hadrons of the kernels
        (kaons_can_act)."""
        if self.process_hadronic:
            if (
                self.process_pp
                and self.pp_grid == self.grid_gen
                and (
                    self.pp_transition_energy_to_delta_approx
                    != self.pp_built_e_trans
                    or self.pp_model_charged_pions != self.pp_built_charged
                    or self.pp_model_neutral_pions != self.pp_built_neutral
                )
            ):
                self._check_pp_transition_energy()
                var kaons = self.oProtonProton.kaons_built
                var charm = self.oProtonProton.charm_built
                self._build_pp_kernels(kaons, charm)
            if (
                self.process_photopion
                and self.pg_grid == self.grid_gen
                and self.optimize_photopion_target_photon_max
                != self.pg_built_photon_max
            ):
                var kaons = self.oPhotoHadron.kaons_built
                var charm = self.oPhotoHadron.charm_built
                self._build_pg_kernels(kaons, charm)
            # (the proton-loss kernel takes neither limit)
            if (
                self.process_bethe_heitler
                and self.process_bethe_heitler_emission
                and self.bh_built
                and (
                    self.optimize_bethe_heitler_incoming_protons_min
                    != self.bh_built_protons_min
                    or self.optimize_bethe_heitler_target_photon_max
                    != self.bh_built_photon_max
                )
            ):
                self._build_bh_kernels()
        # (the emission kernel: the photon-loss kernel takes neither limit)
        if (
            self.process_electron_compton
            and self.process_electron_compton_emission
            and self.ic_built
            and (
                self.optimize_compton_emission_emin != self.ic_built_emin
                or self.optimize_compton_target_emax != self.ic_built_photon_max
            )
        ):
            self.oComp.initialize_kernels(
                self.dat,
                False,
                self.optimize_compton_emission_emin,
                self.optimize_compton_target_emax,
            )
            self.ic_built_emin = self.optimize_compton_emission_emin
            self.ic_built_photon_max = self.optimize_compton_target_emax

    def ReInit_Physics_Modules(mut self):
        if self.verbose:
            print("Sim.Manager: ReInit_Physics_Modules")
        self.oSyn.clear_internal_arrays(self.dat)
        self.oSynP.clear_internal_arrays(self.dat)
        self.oComp.clear_internal_arrays(self.dat)
        self.oHadIC.clear_internal_arrays(self.dat)
        self.oPair.clear_internal_arrays(self.dat)
        self.oBH.clear_internal_arrays(self.dat)
        self.oPhotoHadron.clear_internal_arrays(self.dat)
        self.oProtonProton.clear_internal_arrays(self.dat)
        self.oPionDec.clear_internal_arrays(self.dat)
        self.oMuonDec.clear_internal_arrays(self.dat)
        self.oPairAnn.clear_internal_arrays(self.dat)
        self.oTPP.clear_internal_arrays(self.dat)
        self.oNeutronDec.clear_internal_arrays(self.dat)
        self.oKaonDec.clear_internal_arrays(self.dat)
        self.oCharmDec.clear_internal_arrays(self.dat)
        self.oEscape.clear_internal_arrays(self.dat)
        self.oInj.clear_internal_arrays(self.dat)
        self.oExtraCool.clear_internal_arrays(self.dat)

    def clear_particle_densities(mut self):
        if self.verbose:
            print("Sim.Manager: clear_particle_densities")
        self.dat.clear_all_particle_arrays()
        # the nucleons of the last step's charm decays, a source of the next
        fill_zero(self.c[C_qpro_C])
        fill_zero(self.c[C_qntr_C])
        self.step_counter = 0
        self.pair_halves_valid = True  # (all zero)
        self.components_stale = False
        self.last_step_merged = False
        self.last_step_hadronic = False
        self.kchain_live = [False, False]
        self.cchain_live = [False, False]
        self.ckaons_live = [False, False]
        self.kaons_live = [False, False]
        self.charm_live = [False, False]
        self.pchain_live = [False, False]
        # and the populations the last step evolved, which the kaon and charm
        # cooling getters follow until the next step
        self.ckaons = [False, False]
        self.kaons_ev = [False, False]
        self.charm_ev = [False, False]
        self.kc_zeroed = False
        # The estimated maximum energies of a fresh object, not the last run's.
        # Otherwise, the first step (without an estimate) reports the old ones.
        var fresh = RunParams()
        self.rp.estimated_electron_emax = fresh.estimated_electron_emax
        self.rp.estimated_proton_emax = fresh.estimated_proton_emax

    def clear_am3(mut self):
        if self.verbose:
            print("Sim.Manager: clear_am3")
        self.clear_particle_densities()
        self.reset_coefficients()
        self.ReInit_Physics_Modules()
        self.Initialize_Solver_Algorithms()

    def zero_components(mut self):
        """The kaons and charm hadrons of each origin, the kaons of the charm
        decays and the nu_tau of the charm decay chains are not components. They
        are evolved without parse_sed too."""
        comptime assert (
            P_muon_plus_left_Kpp + 4 - P_pion_plus_Kpg == 2 * (2 + 4)
            and P_muon_plus_left_Cpp + 4 - P_pion_plus_Cpg == 2 * (2 + 4)
            and P_neutrino_el_Cpp + 4 - P_neutrino_el_anti_ndec
            == 1 + 4 * 4 + 2 * 2
            and P_pairs_ndec + 1 - P_pairs_in_ele == 2 * N_PAIR_COMP + 2
        )
        var first: List[Int] = [
            P_pion_plus_pg,
            P_muon_plus_left_pg,
            P_pion_plus_Kpg,
            P_pion_plus_Cpg,
            P_neutrino_el_pg,
            P_neutrino_el_anti_ndec,
            P_pairs_in,
            P_pairs_in_ele,
            P_photons_insy,
            P_photons_pair_ann,
        ]
        var last: List[Int] = [
            P_pion_0_pp,
            P_muon_minus_right_pp,
            P_muon_plus_left_Kpp + 3,
            P_muon_plus_left_Cpp + 3,
            P_neutrino_mu_anti_pp,
            P_neutrino_el_Cpp + 3,
            P_pairs_pp,
            P_pairs_ndec,
            P_photons_inj,
            P_photons_chic,
        ]
        for r in range(len(first)):
            for id in range(first[r], last[r] + 1):
                fill_zero(self.dat.n[id])
        self.pair_halves_valid = True
        self.pchain_live = [False, False]
        self.kchain_live = [False, False]
        self.cchain_live = [False, False]
        self.components_stale = False

    def _nonzero(self, id: Int) -> Bool:
        for v in self.dat.n[id]:
            if v != 0.0:
                return True
        return False

    def _block_nonzero(self, id0: Int, n: Int) -> Bool:
        for k in range(n):
            if self._nonzero(id0 + k):
                return True
        return False

    def _live(self, id: Int, q: Int) -> Bool:
        """A component that holds particles or gets some is evolved, also after
        its process is switched off. So it escapes and cools like the total that
        it is part of. (AM3 froze it while the total evolved.)"""
        if self._nonzero(id):
            return True
        for v in self.c[q]:
            if v != 0.0:
                return True
        return False

    def _reset_coeffs(mut self, begin: Int, end: Int, n: Int):
        for id in range(begin, end):
            resize_zero(self.c[id], n)

    def reset_coefficients(mut self):
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
        @inline(.always)
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
        var opt = "(opt. eight-bin kernel)" if (
            self.optimize_annihilation_pair_emission
        ) else "(full kernel)"
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
            "  kaons (N-gamma, pp -> K+-, K0_S and K0_L): ",
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
        print(
            (
                "  charm hadrons (N-gamma, pp -> D+-, D0, anti-D0, Ds+-,"
                " Lambda_c+ and anti-Lambda_c-): "
            ),
            b(self.process_charm),
            "  (decay: ",
            b(self.process_charm_decay),
            "; synchrotron: ",
            b(self.process_charm_syn),
            " (em.: ",
            b(self.process_charm_syn_emission),
            ", cool.: ",
            b(self.process_charm_syn_cooling),
            "); compton: ",
            b(self.process_charm_compton),
            " (em.: ",
            b(self.process_charm_compton_emission),
            ", cool.: ",
            b(self.process_charm_compton_cooling),
            "))",
            sep="",
        )

    # --------------------------------------------------------------------------
    # solver algorithms
    # --------------------------------------------------------------------------
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
    ) -> SolverAlgorithm:
        """The solver's time step and its cooling and decay switches are set
        before every use (evolve_group, the _sync_* functions of evolve_step,
        AM3._current_cooling). So the solver starts without them. (AM3 gives
        them here, and nowhere else.)"""
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
            0.0,
            False,
            False,
            False,
            False,
            self.solver_threshold_cool_dom,
            self.solver_threshold_esc_dom,
            self.solver_threshold_matrix,
            self.solver_simpleqintegration,
            self.verbose,
        )

    def _mk_hadron(
        self, name: String, tau: Float64, mass: Float64
    ) -> SolverAlgorithm:
        """Solver of a hadron or muon of mean life tau [s] and mass [GeV], on
        the hadron grid in seconds and GeV."""
        return self._mk(
            name,
            self.dat.nbins_had,
            -self.dat.ind_min_energy_pho_hadgrid,
            1.0,
            1.0,
            1.0,
            tau,
            mass / MASS_E,
        )

    def _mk_neutrino(self, name: String) -> SolverAlgorithm:
        return self._mk(
            name,
            self.dat.nbins_nu,
            -self.dat.ind_min_energy_pho_hadgrid,
            1.0,
            1.0,
            1.0,
            T_NO_DECAY,
            MASS_NONE / MASS_E,
        )

    def _mk_lepton(self, name: String) -> SolverAlgorithm:
        return self._mk(
            name,
            self.dat.nbins_lep,
            -self.dat.ind_min_energy_pho,
            MASS_E,
            L_SSC,
            T_SSC,
            T_NO_DECAY,
            1.0,
        )

    def _mk_photon(self, name: String) -> SolverAlgorithm:
        return self._mk(
            name,
            self.dat.nbins_pho,
            0,
            MASS_NONE,
            L_SSC,
            T_SSC,
            T_NO_DECAY,
            1.0,
        )

    def _mk_kaon(self, s: Int) -> SolverAlgorithm:
        return self._mk_hadron(kaon_name(s), kaon_lifetime(s), kaon_mass(s))

    def _mk_charm(self, s: Int) -> SolverAlgorithm:
        return self._mk_hadron(charm_name(s), charm_lifetime(s), charm_mass(s))

    def Initialize_Solver_Algorithms(mut self):
        if self.verbose:
            print("Initialize_Solver_Algorithms")

        var S = List[SolverAlgorithm](capacity=N_SOLVERS)
        S.append(self._mk_hadron("proton", T_NO_DECAY, MASS_P))
        # neutron decay (am3_mojo addition): PDG 2026 mean life and mass. AM3
        # gives the neutron solver 879.5 s and m_p, with the decay off.
        S.append(self._mk_hadron("neutron", TAU_N, MASS_N))
        var pion_names = List[String]()
        pion_names.append("pion_plus")
        pion_names.append("pion_minus")
        pion_names.append("pgamma_produced_pion_plus")
        pion_names.append("pgamma_produced_pion_minus")
        pion_names.append("pp_produced_pion_plus")
        pion_names.append("pp_produced_pion_minus")
        for k in range(6):
            S.append(self._mk_hadron(pion_names[k], TAU_PI, MASS_PI))
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
            S.append(self._mk_hadron(mu_names[k], TAU_MU, MASS_MU))
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
            S.append(self._mk_neutrino(nu_names[k]))
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
            S.append(self._mk_lepton(lep_names[k]))
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
            S.append(self._mk_photon(g_names[k]))
        # species am3_mojo adds to AM3
        S.append(self._mk_photon("photon_pair_annihilation"))
        S.append(self._mk_lepton("triplet_produced_pairs"))
        for n in ["photon_triplet-produced_syn", "photon_triplet-produced_IC"]:
            S.append(self._mk_photon(String(n)))
        S.append(self._mk_lepton("neutron_decay_electrons"))
        for n in [
            "photon_neutron-decay-electrons_syn",
            "photon_neutron-decay-electrons_IC",
        ]:
            S.append(self._mk_photon(String(n)))
        S.append(self._mk_neutrino("neutron_decay_electron_anti_neutrino"))
        for s in range(N_KAON):  # kaons from photo-hadronic interactions
            S.append(self._mk_kaon(s))
        S.append(self._mk_photon("photon_kaon_syn"))
        S.append(self._mk_photon("photon_kaon_IC"))
        for s in range(N_KAON):  # kaons from pp collisions
            S.append(self._mk_kaon(s))
        # the pions, muons and neutrinos of the kaon decays are configured as
        # AM3's pion, muon and neutrino components
        var kpre: List[String] = ["pgamma_kaon_decay_", "pp_kaon_decay_"]
        for o in range(2):
            for k in range(2):
                S.append(
                    self._mk_hadron(kpre[o] + pion_names[k], TAU_PI, MASS_PI)
                )
        for o in range(2):
            for k in range(4):
                S.append(
                    self._mk_hadron(kpre[o] + mu_names[k], TAU_MU, MASS_MU)
                )
        for o in range(2):
            for k in range(4):
                S.append(self._mk_neutrino(kpre[o] + nu_names[k]))
        # the nu_mu and anti-nu_mu of the decays of the pions made in the
        # interactions are configured as AM3's neutrino components
        var ppre: List[String] = [
            "pgamma_produced_pion_decay_",
            "pp_produced_pion_decay_",
        ]
        for o in range(2):
            for k in range(2, 4):
                S.append(self._mk_neutrino(ppre[o] + nu_names[k]))
        # charm hadrons (photo-hadronic, their photon components, pp) and the
        # kaons, pions, muons and neutrinos of their decays, per origin
        for s in range(N_CHARM):
            S.append(self._mk_charm(s))
        for n in ["photon_charm_syn", "photon_charm_IC"]:
            S.append(self._mk_photon(String(n)))
        for s in range(N_CHARM):
            S.append(self._mk_charm(s))
        for _ in range(2):
            for s in range(N_KAON):
                S.append(self._mk_kaon(s))
        var cpre: List[String] = ["pgamma_charm_decay_", "pp_charm_decay_"]
        for o in range(2):
            for k in range(2):
                S.append(
                    self._mk_hadron(cpre[o] + pion_names[k], TAU_PI, MASS_PI)
                )
        for o in range(2):
            for k in range(4):
                S.append(
                    self._mk_hadron(cpre[o] + mu_names[k], TAU_MU, MASS_MU)
                )
        for o in range(2):
            for k in range(4):
                S.append(self._mk_neutrino(cpre[o] + nu_names[k]))
        var tau_names: List[String] = ["tau_neutrino", "tau_anti_neutrino"]
        for o in range(2):
            for k in range(2):
                S.append(self._mk_neutrino(cpre[o] + tau_names[k]))
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
            and SA_num_pidec_pg == 99
            and SA_num_pidec_pp == 101
            and SA_C == 103
            and SA_gcsy == 111
            and SA_gcic == 112
            and SA_Cpp == 113
            and SA_K_Cpg == 121
            and SA_K_Cpp == 125
            and SA_Pip_Cpg == 129
            and SA_Pip_Cpp == 131
            and SA_MpL_Cpg == 133
            and SA_MpL_Cpp == 137
            and SA_nue_Cpg == 141
            and SA_nue_Cpp == 145
            and SA_nutau_Cpg == 149
            and SA_nutau_Cpp == 151
            and N_SOLVERS == 153
        )
        # Keep the inverse-Compton cooling kernels of the solvers before. They
        # depend only on the grid, the units and the mass (same_ic_kernel). So
        # the step after a clear_am3 does not compute them again (AM3 does, and
        # so does a step on a new grid).
        if len(self.S) == len(S):
            for k in range(len(S)):
                if self.S[k].ker_status == 1 and S[k].same_ic_kernel(self.S[k]):
                    swap(S[k].ker_ic_t, self.S[k].ker_ic_t)
                    S[k].ker_status = 1
        self.S = S^

    # --------------------------------------------------------------------------
    # species evolution
    # --------------------------------------------------------------------------
    def _ensure_ic_kernel(mut self, k: Int) raises:
        """IC cooling kernels are identical for solvers with the same grid,
        units and mass."""
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
        self.S[k].compute_ic_ker(self.nthreads)

    def evolve_group(mut self, rows: List[Evolution]) raises:
        """Cooling rates depend only on the species, the field and the target
        photons, which are all fixed during a step. Solvers of the same
        configuration share them."""
        var b = self.rp.mag_field
        var distinct = True
        var lead = List[Int](length=len(rows), fill=-1)
        var cjobs = List[CoolJob]()
        for r in range(len(rows)):
            var k = rows[r].solver
            self.S[k].updateDeltaT(self.rp.dt)
            var ld = -1
            for o in range(r):
                if rows[o].solver == k or rows[o].target == rows[r].target:
                    distinct = False
                if (
                    ld < 0
                    and rows[o].nphot == rows[r].nphot
                    and self.S[rows[o].solver].same_cooling(self.S[k])
                ):
                    ld = rows[o].solver
            if ld < 0:
                for f in range(len(self.fresh)):
                    var j = self.fresh[f]
                    if self.fresh_nphot[f] == rows[r].nphot and self.S[
                        j
                    ].same_cooling(self.S[k]):
                        ld = j
                        break
            if ld < 0:
                self._ensure_ic_kernel(k)
                cjobs.append(
                    CoolJob(
                        Pointer(to=self.S[k]).unsafe_origin_cast[
                            MutUntrackedOrigin
                        ](),
                        vp(self.dat.n[rows[r].nphot]),
                        b,
                    )
                )
                ld = k
            lead[r] = ld
        run_cooling(cjobs, self.nthreads)
        for r in range(len(rows)):
            var k = rows[r].solver
            if lead[r] == k:
                var seen = False
                for f in range(len(self.fresh)):
                    if self.fresh[f] == k:
                        seen = True
                if not seen:
                    self.fresh.append(k)
                    self.fresh_nphot.append(rows[r].nphot)
            else:
                # (through pointers: passed as arguments of a call on self.S[k],
                #  the lead's arrays need copies, one allocation each)
                var asy = pd(self.S[lead[r]].Asy)
                var aic = pd(self.S[lead[r]].Aic)
                var con = self.S[lead[r]].cooling_on
                self.S[k].share_cooling(asy, aic, con)
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
        """Variant 0 is the block at the ids given, 1 (p-gamma) and 2 (p-p) the
        blocks after it. The solver, density and source ids of a variant are
        consecutive blocks of four."""
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

    def _pion_decay_rows(
        self, mut rows: List[Evolution], solver0: Int, target0: Int, q0: Int
    ):
        """The nu_mu and anti-nu_mu of the decays of the pions made in the
        interactions are evolved as the other neutrinos."""
        for k in range(2):
            rows.append(
                Evolution(
                    solver0 + k,
                    target0 + k,
                    C_anu,
                    C_Anu,
                    q0 + k,
                    P_photons_on_had_grid,
                )
            )

    # --------------------------------------------------------------------------
    # evolve one step
    # --------------------------------------------------------------------------
    @inline(.always)
    def _tic(self) -> Int:
        return perf_counter_ns()

    @inline(.always)
    def _toc(mut self, idx: Int, t0: Int):
        if self.profile_timing:
            self.timing[idx] = Float64(perf_counter_ns() - t0)

    def _toc_add(mut self, idx: Int, t0: Int):
        """As _toc, for the later parts of a sub-step timed in several parts."""
        if self.profile_timing:
            self.timing[idx] += Float64(perf_counter_ns() - t0)

    def _sync_decay_sinks(mut self):
        """Each solver's decay sink as the decay products of the step. The
        solvers take the switches at init_kernels. So a decay switched off after
        init_kernels kept removing the particles without products. A decay
        switched on after init_kernels made products without removing the
        particles (AM3 bug for pion and muon decay)."""
        var pid = self.process_pion_decay
        var mud = self.process_muon_decay
        var kad = self.kaon_decay_sink()
        var chd = self.charm_decay_sink()
        for k in range(SA_Pip, SA_Pim_pp + 1):
            self.S[k].process_decay = pid
        for k in range(SA_MpL, SA_MmR_pp + 1):
            self.S[k].process_decay = mud
        for o in range(2):
            var kpi = SA_Pip_Kpg if o == 0 else SA_Pip_Kpp
            var cpi = SA_Pip_Cpg if o == 0 else SA_Pip_Cpp
            var kmu = SA_MpL_Kpg if o == 0 else SA_MpL_Kpp
            var cmu = SA_MpL_Cpg if o == 0 else SA_MpL_Cpp
            var ck = SA_K_Cpg if o == 0 else SA_K_Cpp
            for k in range(2):
                self.S[kpi + k].process_decay = pid
                self.S[cpi + k].process_decay = pid
            for k in range(4):
                self.S[kmu + k].process_decay = mud
                self.S[cmu + k].process_decay = mud
            for s in range(N_KAON):
                self.S[kaon_solver(s, o == 1)].process_decay = kad
                self.S[ck + s].process_decay = kad
            for s in range(N_CHARM):
                self.S[charm_solver(s, o == 1)].process_decay = chd
        self.S[SA_neutron].process_decay = self.neutron_decay_sink()

    def kaon_decay_sink(self) -> Bool:
        return self.process_kaon_decay and self.oKaonDec.ready(self.dat)

    def charm_decay_sink(self) -> Bool:
        return self.process_charm_decay and self.oCharmDec.ready(self.dat)

    def neutron_decay_sink(self) -> Bool:
        return self.process_neutron_decay and self.oNeutronDec.ready(self.dat)

    def _set_cooling(mut self, k0: Int, n: Int, sy: Bool, qsc: Bool, ic: Bool):
        for k in range(k0, k0 + n):
            self.S[k].process_sy = sy
            self.S[k].process_quantum_syn_cooling = qsc
            self.S[k].process_ic = ic

    def _sync_cooling_switches(mut self):
        """Each solver's synchrotron, inverse-Compton and quantum-synchrotron
        cooling with the switches of its species, as
        Initialize_Solver_Algorithms sets them. The solvers take the switches
        at init_kernels. So a cooling switched off after init_kernels kept
        cooling the particles whose emission stopped. A cooling switched on
        after init_kernels did not cool them (AM3 bug)."""
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
        # (Inverse Compton cools the e+- only with the kernels of their
        #  emission and photon loss. Otherwise, inverse Compton switched on
        #  after an init_kernels without it cools them without photons.)
        var e_ic = (
            self.process_electron_compton
            and self.process_electron_compton_cooling
            and self.ic_built
        )
        var k_sy = self.process_kaon_syn and self.process_kaon_syn_cooling
        var k_ic = (
            self.process_kaon_compton and self.process_kaon_compton_cooling
        )
        var c_sy = self.process_charm_syn and self.process_charm_syn_cooling
        var c_ic = (
            self.process_charm_compton and self.process_charm_compton_cooling
        )
        self._set_cooling(SA_proton, 1, p_sy, qsc, p_ic)
        self._set_cooling(SA_Pip, SA_Pim_pp + 1 - SA_Pip, pi_sy, qsc, pi_ic)
        self._set_cooling(SA_MpL, SA_MmR_pp + 1 - SA_MpL, mu_sy, qsc, mu_ic)
        self._set_cooling(SA_e, SA_eppp + 1 - SA_e, e_sy, qsc, e_ic)
        self._set_cooling(SA_eptpp, 1, e_sy, qsc, e_ic)
        self._set_cooling(SA_endec, 1, e_sy, qsc, e_ic)
        for o in range(2):
            var kpi = SA_Pip_Kpg if o == 0 else SA_Pip_Kpp
            var cpi = SA_Pip_Cpg if o == 0 else SA_Pip_Cpp
            var kmu = SA_MpL_Kpg if o == 0 else SA_MpL_Kpp
            var cmu = SA_MpL_Cpg if o == 0 else SA_MpL_Cpp
            self._set_cooling(kpi, 2, pi_sy, qsc, pi_ic)
            self._set_cooling(cpi, 2, pi_sy, qsc, pi_ic)
            self._set_cooling(kmu, 4, mu_sy, qsc, mu_ic)
            self._set_cooling(cmu, 4, mu_sy, qsc, mu_ic)
            var ck = SA_K_Cpg if o == 0 else SA_K_Cpp
            for s in range(N_KAON):
                var ch = s == KP or s == KM
                var sy = k_sy and ch
                var q = qsc and ch
                var ic = k_ic and ch
                self._set_cooling(kaon_solver(s, o == 1), 1, sy, q, ic)
                self._set_cooling(ck + s, 1, sy, q, ic)
            for s in range(N_CHARM):
                var ch = charm_charged(s)
                self._set_cooling(
                    charm_solver(s, o == 1),
                    1,
                    c_sy and ch,
                    qsc and ch,
                    c_ic and ch,
                )

    def _sync_solver_settings(mut self):
        """Each solver's regime thresholds and source integral as their
        setters now have them. The solvers take them when they are built
        (clear_am3). So a setting changed after the build was ignored, while
        its getter reported it (AM3 bug)."""
        for k in range(len(self.S)):
            self.S[k].threshold_cool_dom = self.solver_threshold_cool_dom
            self.S[k].threshold_esc_dom = self.solver_threshold_esc_dom
            self.S[k].threshold_matrix = self.solver_threshold_matrix
            self.S[k].simpleqintegration = self.solver_simpleqintegration

    def evolve_step(mut self) raises:
        if self.verbose:
            print("evolve for 1 step")
        # the cooling rates of earlier steps are stale
        self.fresh.clear()
        self.fresh_nphot.clear()
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
        # (Without its kernels, photo-pion production does nothing while
        #  switched on. AM3 crashes. pp collisions, Bethe-Heitler and
        #  gamma-gamma pair production and inverse Compton without theirs make
        #  nothing, as in AM3, and their docstrings say so.)
        if (
            self.process_hadronic
            and self.process_photopion
            and self.pg_grid != self.grid_gen
        ):
            raise Error(
                "photo-pion production was switched on after an init_kernels()"
                " without it: call init_kernels() again"
            )
        self._rebuild_changed_kernels()
        self._build_syn_kernels()
        self._sync_decay_sinks()
        self._sync_cooling_switches()
        self._sync_solver_settings()
        if not self.process_parse_sed:
            self.components_stale = True
        var had = self.process_hadronic
        self.last_step_hadronic = had
        self.oPhotoHadron.set_kaons(had and self.kaons_pg_active())
        self.oProtonProton.kaons_src = had and self.kaons_pp_active()
        # (as for AM3's pions, the protons lose the energy that goes into
        #  kaons whether or not the emission is followed)
        self.oProtonProton.kaons_loss = (
            had and self.process_pp and self.kaons_can_act(1)
        )
        self.oPhotoHadron.set_charm(had and self.charm_pg_active())
        self.oProtonProton.charm_src = had and self.charm_pp_active()
        self.oProtonProton.charm_loss = (
            had and self.process_pp and self.charm_can_act(1)
        )
        self._update_populations(had)
        self._update_pion_chains(had)
        self._update_kaon_chain()
        self._update_charm_chain(had)
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
            # The kaons and charm hadrons before the pions: their decays feed
            # this step's pions, muons and neutrinos. Without any (kc_idle), the
            # stage only sets their products and totals to zero. It is skipped
            # after a step without any (kc_zeroed), and the sums of the later
            # stages leave them out.
            self.kc_idle = not (
                self.kaons_ev[0]
                or self.kaons_ev[1]
                or self.ckaons[0]
                or self.ckaons[1]
                or self.charm_ev[0]
                or self.charm_ev[1]
                or self.kchain[0]
                or self.kchain[1]
                or self.cchain[0]
                or self.cchain[1]
            )
            var kc = not (self.kc_idle and self.kc_zeroed)
            self.kc_zeroed = False  # (until the stage runs)
            t0 = self._tic()
            if kc:
                self.Calc_Coeff_K()
            else:
                self.pg_conv_fresh = False
            self._toc(T_KAON_COEFF, t0)
            t0 = self._tic()
            if kc:
                self.Calc_K()
            self._toc(T_KAON_EVOLVE, t0)
            t0 = self._tic()
            if kc:
                self.Calc_Coeff_C()
            self._toc(T_CHARM_COEFF, t0)
            t0 = self._tic()
            if kc:
                self.Calc_C()
            self._toc(T_CHARM_EVOLVE, t0)
            self.kc_zeroed = self.kc_idle
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
            ] = N_SSC * self.dat.grid_lep2had(i, ph, self.dat.nbins_pho)

    def Calc_Coeff_proton_NTR(mut self):
        if self.verbose:
            print(" Calculate proton + neutron coefficient")
        # (The loss rates of a process that is off are zero, as the getters and
        #  the proton maximum-energy estimate read them. AM3 kept those of its
        #  last step with the process on.)
        if self.process_bethe_heitler and self.process_bethe_heitler_cooling:
            self.oBH.Calc_BH_Cool(self.dat)
        else:
            fill_zero(self.oBH.ABHHadP)
        if self.process_photopion and self.process_photopion_cooling:
            # (before the nucleons are evolved: with the bins of their
            #  injection and of the nucleons of the last step's charm decays)
            self.oPhotoHadron.Calc_PhotoHadron_PhotonConvolution(
                self.dat,
                max(
                    PhotoHadron.nonzero_top(self.oInj.inj_proton),
                    PhotoHadron.nonzero_top(self.oInj.inj_neutron),
                    PhotoHadron.nonzero_top(self.c[C_qpro_C]),
                    PhotoHadron.nonzero_top(self.c[C_qntr_C]),
                ),
            )
            self.oPhotoHadron.Calc_PhotoHadron_ProtonNeutronReinj(self.dat)
            self.oPhotoHadron.Calc_PhotoHadron_ProtonNeutronLoss(self.dat)
        else:
            fill_zero(self.oPhotoHadron.afPG_PRO)
        if self.process_pp and self.process_pp_cooling:
            self.oProtonProton.Calc_PP_ProtonLoss(self.rp)
        else:
            fill_zero(self.oProtonProton.APP_PRO)

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
                pgc * self.oPhotoHadron.afPG_PRO[i]
                + esc * self.oEscape.afescPRO[i]
                + exh
            )
            self.c[C_qpro][i] = (
                pgc * self.oPhotoHadron.QPG_PROGen[i] + self.oInj.inj_proton[i]
            )
        if self.neutron_decay_on():
            for i in range(self.dat.nbins_had):
                self.c[C_qpro][i] += self.c[C_qpro_ndec][i]
        # the protons of the charm decays of the last step (am3_mojo addition,
        # zero without them)
        for i in range(self.dat.nbins_had):
            self.c[C_qpro][i] += self.c[C_qpro_C][i]

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
            # (the neutrons' photo-hadronic loss rate is the protons')
            self.c[C_antr][i] = (
                pgc * self.oPhotoHadron.afPG_PRO[i]
                + esc * self.oEscape.afescNTR[i]
                + exh
            )
            self.c[C_qntr][i] = (
                pgc * self.oPhotoHadron.QPG_NTRGen[i] + self.oInj.inj_neutron[i]
            )
        # the neutrons of the charm decays of the last step (am3_mojo addition,
        # zero without them)
        for i in range(self.dat.nbins_had):
            self.c[C_qntr][i] += self.c[C_qntr_C][i]

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

    # --------------------------------------------------------------------------
    # neutron decay: n -> p e- anti-nu_e (am3_mojo addition)
    # --------------------------------------------------------------------------
    def neutron_decay_on(self) -> Bool:
        """The neutron solver's decay sink needs the product kernels of
        init_kernels (_sync_decay_sinks), and the two must agree."""
        return (
            self.process_hadronic
            and self.S[SA_neutron].process_decay
            and self.oNeutronDec.ready(self.dat)
        )

    def Calc_Neutron_Decay(mut self):
        """Products of the neutrons that decay in this step, from the evolved
        neutrons as in AM3's decays."""
        if self.verbose:
            print(" Calculate neutron decay")
        if self.neutron_decay_on():
            self.oNeutronDec.Calc_NtoX(self.dat, self.S[SA_neutron].alpha_decay)
        else:
            self.oNeutronDec.clear_internal_arrays(self.dat)
        copy_into(self.c[C_qpro_ndec], self.oNeutronDec.q_p)
        for i in range(self.dat.nbins_lep):
            self.c[C_qe_ndec][i] = self.oNeutronDec.q_e[i] / RATE_SSC
        copy_into(self.c[C_qnuea_ndec], self.oNeutronDec.q_nu)

    # --------------------------------------------------------------------------
    # pions: pi+, pi-, pi0 (AM3)
    # --------------------------------------------------------------------------
    def Calc_Coeff_PI(mut self):
        if self.verbose:
            print(" Calculate pion coefficient")
        if self.process_photopion and self.process_photopion_emission:
            if (
                not (self.process_photopion and self.process_photopion_cooling)
                and not self.pg_conv_fresh
            ):
                self.oPhotoHadron.Calc_PhotoHadron_PhotonConvolution(self.dat)
            self.oPhotoHadron.Calc_PhotoHadron_Rad(self.dat)
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
            # pge and ppe are 0 or 1, so the products are exact. The totals also
            # take the pions of the kaon and charm decay chains of each origin,
            # added in this order.
            var pip_pg = pge * self.oPhotoHadron.qpip[i]
            var pip_pp = ppe * self.oProtonProton.qpip[i]
            var pim_pg = pge * self.oPhotoHadron.qpim[i]
            var pim_pp = ppe * self.oProtonProton.qpim[i]
            self.c[C_qpip][i] = (
                (pip_pg + self.c[C_qpip_K][i] + self.c[C_qpi_Cpg][i])
                + (pip_pp + self.c[C_qpip_Kpp][i] + self.c[C_qpi_Cpp][i])
                + self.oInj.inj_piplus[i]
            )
            self.c[C_qpim][i] = (
                (pim_pg + self.c[C_qpim_K][i] + self.c[C_qpi_Cpg + 1][i])
                + (pim_pp + self.c[C_qpim_Kpp][i] + self.c[C_qpi_Cpp + 1][i])
                + self.oInj.inj_piminus[i]
            )
            # The photo-hadronic and pp components hold the pions made in the
            # interactions. Those of the kaon decays are P_pion_*_Kpg / _Kpp.
            self.c[C_qpip_pg][i] = pip_pg
            self.c[C_qpip_pp][i] = pip_pp
            self.c[C_qpim_pg][i] = pim_pg
            self.c[C_qpim_pp][i] = pim_pp
            self.c[C_qpi0_pg][i] = (
                pge * self.oPhotoHadron.qpi0[i] + self.c[C_qpi0_K][i]
            )
            # (pi0 from kaon and charm decays always go through pi0 -> gamma
            #  gamma)
            self.c[C_qpi0_pp][i] = (
                ppe0 * self.oProtonProton.qpi0[i] + self.c[C_qpi0_Kpp][i]
            )
            self.c[C_qpi0_pg][i] += (
                self.c[C_qpip_C + (CP_PI0 - CP_PIP)][i]
                + self.c[C_qpip_KC + (PR_PI0 - PR_PIP)][i]
            )
            self.c[C_qpi0_pp][i] += (
                self.c[C_qpip_Cpp + (CP_PI0 - CP_PIP)][i]
                + self.c[C_qpip_KCpp + (PR_PI0 - PR_PIP)][i]
            )
            self.c[C_qpi0][i] = (
                self.c[C_qpi0_pg][i]
                + self.c[C_qpi0_pp][i]
                + self.oInj.inj_pi0[i]
            )

    def Calc_PI(mut self) raises:
        if self.verbose:
            print(" Calculate pions")
        # AM3 evolves all pion components with the pi+ solver. The identical
        # pi- and component solvers give the same result, in parallel.
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
        if self.pchain[0]:
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
        if self.pchain[1]:
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
        for o in range(2):
            if self.charm_chain_on(o):
                for k in range(2):
                    rows.append(
                        Evolution(
                            (SA_Pip_Cpg if o == 0 else SA_Pip_Cpp) + k,
                            (P_pion_plus_Cpg if o == 0 else P_pion_plus_Cpp)
                            + k,
                            C_apip if k == 0 else C_apim,
                            C_Apip if k == 0 else C_Apim,
                            (C_qpi_Cpg if o == 0 else C_qpi_Cpp) + k,
                            P_photons_on_had_grid,
                        )
                    )
        self.evolve_group(rows)

    # --------------------------------------------------------------------------
    # kaons: K+-, K0_S and K0_L (am3_mojo addition)
    # --------------------------------------------------------------------------
    def kaons_can_act(self, origin: Int) -> Bool:
        """Kaons of an origin (0 photo-hadronic, 1 pp) can be made and decay. An
        init_kernels with kaons off frees the kaon decay tables, unless charm is
        on (the charm decays make kaons). But where their process is off, it
        keeps the production kernels that an earlier init_kernels built with
        kaons. Kaons switched on again then stay off until the next
        init_kernels, as after any init_kernels without them. With charm on, an
        origin whose process was off and whose kept kernels have kaons is the
        exception. The production, loss, decay sink (_sync_decay_sinks) and
        decay products of the kaons all follow kaons_can_act, so that kaons are
        never removed by decay without products."""
        var built = self.oProtonProton.kaons_built
        if origin == 0:
            built = self.oPhotoHadron.kaons_built
        return self.process_kaons and built and self.oKaonDec.ready(self.dat)

    def kaons_pg_active(self) -> Bool:
        return (
            self.kaons_can_act(0)
            and self.process_hadronic
            and self.process_photopion
            and self.process_photopion_emission
        )

    def kaons_pp_active(self) -> Bool:
        return (
            self.kaons_can_act(1)
            and self.process_hadronic
            and self.process_pp
            and self.process_pp_emission
        )

    # the same in the step (set at the start of evolve_step)
    def kaons_pg_on(self) -> Bool:
        return self.oPhotoHadron.kaons

    def kaons_pp_on(self) -> Bool:
        return self.oProtonProton.kaons_src

    def kaons_on(self) -> Bool:
        return self.kaons_ev[0] or self.kaons_ev[1]

    def _update_populations(mut self, had: Bool):
        """Evolve the kaons and the charm hadrons of each origin's interactions
        while they are made, otherwise while any are left. So they decay and
        escape like AM3's pions, which are evolved whatever the switches of
        their production. The live flags are false where the populations are
        known to be zero. This spares the scans."""
        for o in range(2):
            var k = self.kaons_pg_on() if o == 0 else self.kaons_pp_on()
            if had and not k and self.kaons_live[o]:
                k = self._block_nonzero(
                    P_kaon_plus_pg if o == 0 else P_kaon_plus_pp, N_KAON
                )
                self.kaons_live[o] = k
            if k:
                self.kaons_live[o] = True
            self.kaons_ev[o] = k
            var c = self.charm_pg_on() if o == 0 else self.charm_pp_on()
            if had and not c and self.charm_live[o]:
                c = self._block_nonzero(
                    P_charm_pg if o == 0 else P_charm_pp, N_CHARM
                )
                self.charm_live[o] = c
            if c:
                self.charm_live[o] = True
            self.charm_ev[o] = c

    def any_kaons_on(self) -> Bool:
        return self.kaons_on() or self.ckaons[0] or self.ckaons[1]

    def _update_pion_chains(mut self, had: Bool):
        """Evolve the pion, muon and neutrino components of each origin's
        interactions while the interactions make pions, otherwise while any are
        left. So they decay and escape like the totals they are part of (AM3
        froze them once the emission was switched off). pchain_live is false
        where the components are known to be zero. This spares the scan."""
        for o in range(2):
            var chain = False
            if self.process_parse_sed and had:
                if o == 0:
                    chain = (
                        self.process_photopion
                        and self.process_photopion_emission
                    )
                else:
                    chain = self.process_pp and self.process_pp_emission
                if not chain and self.pchain_live[o]:
                    chain = self._pion_chain_nonzero(o)
                self.pchain_live[o] = chain
            self.pchain[o] = chain

    def _pion_chain_nonzero(self, o: Int) -> Bool:
        var p0 = P_pion_plus_pg if o == 0 else P_pion_plus_pp
        var m0 = P_muon_plus_left_pg if o == 0 else P_muon_plus_left_pp
        var n0 = P_neutrino_el_pg if o == 0 else P_neutrino_el_pp
        var d0 = P_neutrino_mu_pidec_pg if o == 0 else P_neutrino_mu_pidec_pp
        return (
            self._block_nonzero(p0, 2)
            or self._block_nonzero(m0, 4)
            or self._block_nonzero(n0, 4)
            or self._block_nonzero(d0, 2)
        )

    def kaon_chain_on(self, origin: Int) -> Bool:
        return self.kchain[origin]

    def _update_kaon_chain(mut self):
        """Evolve the kaon decay products of each origin while those kaons
        decay, otherwise while any are left. So they escape like their species.
        kchain_live is false where the products are known to be zero. This
        spares the scan without kaons."""
        for o in range(2):
            var on = self.kaons_ev[o]
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
        """The decay sink of the kaon solvers (_sync_decay_sinks) and the decay
        products must agree."""
        return self.kaons_on() and self.S[SA_K].process_decay

    def Calc_Coeff_K(mut self):
        if self.verbose:
            print(" Calculate kaon coefficient")
        self.pg_conv_fresh = False
        if not self.kaons_on():
            return
        if self.kaons_pg_on():
            if not (self.process_photopion and self.process_photopion_cooling):
                self.oPhotoHadron.Calc_PhotoHadron_PhotonConvolution(self.dat)
                self.pg_conv_fresh = True
            self.oPhotoHadron.Calc_PhotoHadron_Kaons(self.dat)
            for s in range(N_KAON):
                copy_into(self.c[C_qK + s], self.oPhotoHadron.qK[s])
        else:  # (kaons left from earlier steps only)
            for s in range(N_KAON):
                fill_zero(self.c[C_qK + s])
        if self.kaons_pp_on():
            self.oProtonProton.Calc_PP_KaonSrc(self.rp, self.dat)
            for s in range(N_KAON):
                copy_into(self.c[C_qKpp + s], self.oProtonProton.qK[s])
        else:
            for s in range(N_KAON):
                fill_zero(self.c[C_qKpp + s])
        self._kaon_charm_escape_cooling(False)

    def _kaon_charm_escape_cooling(mut self, charm: Bool):
        """The charged kaons and charm hadrons escape and cool as charged pions.
        The neutral ones (K0_S, K0_L, D0 and anti-D0) escape and cool as
        neutrons (no adiabatic cooling, neutral escape)."""
        var adi = fbool(self.process_adiabatic_cooling) * Expansion.AAdiHad(
            self.rp
        )
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        var A = C_AC if charm else C_AK
        var a = C_aC if charm else C_aK
        var A0 = C_AC0 if charm else C_AK0
        var a0 = C_aC0 if charm else C_aK0
        for i in range(self.dat.nbins_had):
            var x = self.oExtraCool.AExtraCoolC[
                i
            ] if charm else self.oExtraCool.AExtraCoolK[i]
            self.c[A][i] = adi + x
            self.c[a][i] = esc * self.oEscape.afescPipm[i] + exh
            self.c[A0][i] = x
            self.c[a0][i] = esc * self.oEscape.afescNTR[i] + exh

    def Calc_K(mut self) raises:
        if self.verbose:
            print(" Calculate kaons")
        comptime assert (
            P_kaon_minus == P_kaon_plus + KM
            and P_kaon_0S == P_kaon_plus + KS
            and P_kaon_0L == P_kaon_plus + KL
        )
        var pg = self.kaons_ev[0]
        var pp = self.kaons_ev[1]
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
        var dest = Array[Int, N_PROD](fill=0)
        dest[PR_PIP] = C_qpip_K
        dest[PR_PIM] = C_qpim_K
        dest[PR_PI0] = C_qpi0_K
        dest[PR_MUPL] = C_qmupl_K
        dest[PR_MUPR] = C_qmupr_K
        dest[PR_MUML] = C_qmuml_K
        dest[PR_MUMR] = C_qmumr_K
        dest[PR_EM] = C_qe_K
        dest[PR_EP] = C_qp_K
        dest[PR_NUE] = C_qnue_K
        dest[PR_NUEB] = C_qnuea_K
        dest[PR_NUM] = C_qnum_K
        dest[PR_NUMB] = C_qnuma_K
        # (the pp muons and neutrinos: blocks in the order of the products)
        comptime assert (
            PR_MUPR == PR_MUPL + 1
            and PR_MUML == PR_MUPL + 2
            and PR_MUMR == PR_MUPL + 3
            and PR_NUEB == PR_NUE + 1
            and PR_NUM == PR_NUE + 2
            and PR_NUMB == PR_NUE + 3
        )
        var dest_pp = Array[Int, N_PROD](fill=0)
        dest_pp[PR_PIP] = C_qpip_Kpp
        dest_pp[PR_PIM] = C_qpim_Kpp
        dest_pp[PR_PI0] = C_qpi0_Kpp
        for p in range(PR_MUPL, PR_MUMR + 1):
            dest_pp[p] = C_qmupl_Kpp + (p - PR_MUPL)
        dest_pp[PR_EM] = C_qe_Kpp
        dest_pp[PR_EP] = C_qp_Kpp
        for p in range(PR_NUE, PR_NUMB + 1):
            dest_pp[p] = C_qnue_Kpp + (p - PR_NUE)
        if self.kaon_decay_on() and (pg or pp):
            self.oKaonDec.Calc_KtoX(
                self.dat, P_kaon_plus_pg, P_kaon_plus_pp, pg, pp
            )
        else:
            self.oKaonDec.clear_internal_arrays(self.dat)
        ref out = self.oKaonDec.out
        for origin in range(2):
            for p in range(N_PROD):
                var d = dest[p] if origin == 0 else dest_pp[p]
                copy_into(self.c[d], out[origin * N_PROD + p])

    # --------------------------------------------------------------------------
    # charm hadrons: D+-, D0, anti-D0, Ds+-, Lambda_c+ and anti-Lambda_c-
    # (am3_mojo addition)
    # --------------------------------------------------------------------------
    def charm_can_act(self, origin: Int) -> Bool:
        """Charm hadrons of an origin can be made and decay, as kaons_can_act.
        This needs charm switched on and the charm and kaon decay tables of the
        last init_kernels (the charm decays make kaons)."""
        var built = self.oProtonProton.charm_built
        if origin == 0:
            built = self.oPhotoHadron.charm_built
        return (
            self.process_charm
            and built
            and self.oCharmDec.ready(self.dat)
            and self.oKaonDec.ready(self.dat)
        )

    def charm_pg_active(self) -> Bool:
        return (
            self.charm_can_act(0)
            and self.process_hadronic
            and self.process_photopion
            and self.process_photopion_emission
        )

    def charm_pp_active(self) -> Bool:
        return (
            self.charm_can_act(1)
            and self.process_hadronic
            and self.process_pp
            and self.process_pp_emission
        )

    # the same in the step (set at the start of evolve_step)
    def charm_pg_on(self) -> Bool:
        return self.oPhotoHadron.charm

    def charm_pp_on(self) -> Bool:
        return self.oProtonProton.charm_src

    def charm_on(self) -> Bool:
        return self.charm_ev[0] or self.charm_ev[1]

    def charm_decay_on(self) -> Bool:
        return self.charm_on() and self.S[SA_C].process_decay

    def charm_kaons_on(self, origin: Int) -> Bool:
        return self.ckaons[origin]

    def charm_chain_on(self, origin: Int) -> Bool:
        return self.cchain[origin]

    def _update_charm_chain(mut self, had: Bool):
        """Evolve the kaons of each origin's charm decays while those charm
        hadrons decay, otherwise while any are left. So they escape like their
        species. The decays of these kaons also feed the chain's pions, muons
        and neutrinos (as Calc_C decays the kaons). With the hadronic processes
        off, no kaons are evolved (as in _update_populations). In that case,
        ckaons_live keeps whether any are left for when the hadronic processes
        are on again."""
        for o in range(2):
            var on = self.charm_ev[o]
            var dec = on and self.charm_decay_on()
            var kaons = dec
            if had and not kaons and self.ckaons_live[o]:
                kaons = self._charm_kaons_nonzero(o)
                self.ckaons_live[o] = kaons
            if kaons:
                self.ckaons_live[o] = True
            self.ckaons[o] = kaons
            var kdec = kaons and self.S[SA_K_Cpg].process_decay
            var chain = False
            if self.process_parse_sed:
                if dec or kdec:
                    chain = True
                elif self.cchain_live[o]:
                    chain = self._charm_chain_nonzero(o)
                self.cchain_live[o] = chain
            self.cchain[o] = chain

    def _charm_kaons_nonzero(self, origin: Int) -> Bool:
        # (each pp block follows its pg block)
        comptime assert P_kaon_plus_Cpp == P_kaon_plus_Cpg + 4
        for s in range(N_KAON):
            if self._nonzero(P_kaon_plus_Cpg + 4 * origin + s):
                return True
        return False

    def _charm_chain_nonzero(self, origin: Int) -> Bool:
        # (each pp block follows its pg block)
        comptime assert (
            P_pion_plus_Cpp == P_pion_plus_Cpg + 2
            and P_muon_plus_left_Cpp == P_muon_plus_left_Cpg + 4
            and P_neutrino_el_Cpp == P_neutrino_el_Cpg + 4
            and P_neutrino_tau_Cpp == P_neutrino_tau_Cpg + 2
        )
        for k in range(4):
            if (
                (k < 2 and self._nonzero(P_pion_plus_Cpg + 2 * origin + k))
                or self._nonzero(P_muon_plus_left_Cpg + 4 * origin + k)
                or self._nonzero(P_neutrino_el_Cpg + 4 * origin + k)
                or (
                    k < 2 and self._nonzero(P_neutrino_tau_Cpg + 2 * origin + k)
                )
            ):
                return True
        return False

    def Calc_Coeff_C(mut self):
        if self.verbose:
            print(" Calculate charm coefficient")
        if not self.charm_on():
            return
        if self.charm_pg_on():
            if (
                not (self.process_photopion and self.process_photopion_cooling)
                and not self.pg_conv_fresh
            ):
                self.oPhotoHadron.Calc_PhotoHadron_PhotonConvolution(self.dat)
                self.pg_conv_fresh = True
            self.oPhotoHadron.Calc_PhotoHadron_Charm(self.dat)
            for s in range(N_CHARM):
                copy_into(self.c[C_qC + s], self.oPhotoHadron.qC[s])
        else:  # (charm hadrons left from earlier steps only)
            for s in range(N_CHARM):
                fill_zero(self.c[C_qC + s])
        if self.charm_pp_on():
            self.oProtonProton.Calc_PP_CharmSrc(self.rp, self.dat)
            for s in range(N_CHARM):
                copy_into(self.c[C_qCpp + s], self.oProtonProton.qC[s])
        else:
            for s in range(N_CHARM):
                fill_zero(self.c[C_qCpp + s])
        self._kaon_charm_escape_cooling(True)

    def _zero_charm_products(mut self, origin: Int):
        var hb = C_qpip_C if origin == 0 else C_qpip_Cpp
        var kb = C_qpip_KC if origin == 0 else C_qpip_KCpp
        for p in range(N_CPROD_HAD):
            fill_zero(self.c[hb + p])
        for p in range(N_KPROD_HAD):
            fill_zero(self.c[kb + p])
        fill_zero(self.c[C_qe_C if origin == 0 else C_qe_Cpp])
        fill_zero(self.c[C_qp_C if origin == 0 else C_qp_Cpp])
        for k in range(N_CPROD_NU):
            fill_zero(self.c[(C_qnue_C if origin == 0 else C_qnue_Cpp) + k])
        for k in range(N_KPROD_NU):
            fill_zero(self.c[(C_qnue_KC if origin == 0 else C_qnue_KCpp) + k])
        fill_zero(self.c[C_qgma_C if origin == 0 else C_qgma_Cpp])
        for k in range(2):
            fill_zero(self.c[(C_qpi_Cpg if origin == 0 else C_qpi_Cpp) + k])

    def Calc_C(mut self) raises:
        """The decay products of the charm hadrons and their kaons feed this
        step's charm decay chain. Lagged by a step, they also feed the
        nucleons."""
        if self.verbose:
            print(" Calculate charm hadrons")
        var pg = self.charm_ev[0]
        var pp = self.charm_ev[1]
        var rows = List[Evolution]()
        for s in range(N_CHARM):
            var ch = charm_charged(s)
            var a = C_aC if ch else C_aC0
            var A = C_AC if ch else C_AC0
            if pg:
                rows.append(
                    Evolution(
                        charm_solver(s),
                        P_charm_pg + s,
                        a,
                        A,
                        C_qC + s,
                        P_photons_on_had_grid,
                    )
                )
            if pp:
                rows.append(
                    Evolution(
                        charm_solver(s, True),
                        P_charm_pp + s,
                        a,
                        A,
                        C_qCpp + s,
                        P_photons_on_had_grid,
                    )
                )
        if len(rows) > 0:
            self.evolve_group(rows)
        var fg = fbool(pg)
        var fp = fbool(pp)
        for s in range(N_CHARM):
            for i in range(self.dat.nbins_had):
                self.dat.n[P_charm + s][i] = (
                    fg * self.dat.n[P_charm_pg + s][i]
                    + fp * self.dat.n[P_charm_pp + s][i]
                )
        fill_zero(self.c[C_qpro_C])
        fill_zero(self.c[C_qntr_C])
        # the products' blocks (coeffs.mojo) hold them in the order of
        # physics/decays.mojo, each origin's after the other's
        comptime assert (
            N_CPROD_HAD == CP_N - CP_PIP + 1
            and C_qpip_Cpp == C_qpip_C + N_CPROD_HAD
            and C_qpip_KC == C_qpip_Cpp + N_CPROD_HAD
            and C_qpip_KCpp == C_qpip_KC + N_KPROD_HAD
            and C_qpi_Cpg == C_qpip_KCpp + N_KPROD_HAD
            and C_qnue_Cpp == C_qnue_C + N_CPROD_NU
            and C_qnue_KC == C_qnue_Cpp + N_CPROD_NU
            and C_qnue_KCpp == C_qnue_KC + N_KPROD_NU
            and C_qnu_Cpg == C_qnue_KCpp + N_KPROD_NU
        )
        for origin in range(2):
            self._zero_charm_products(origin)
        var dec = self.charm_decay_on()
        if dec and (pg or pp):
            self.oCharmDec.Calc_CtoX(self.dat, P_charm_pg, P_charm_pp, pg, pp)
        # (the products are copied into the coefficient arrays, which keep
        #  their storage)
        for origin in range(2):
            var on = pg if origin == 0 else pp
            if not (on and dec):
                continue
            ref out = self.oCharmDec.out
            var ob = origin * N_CPROD
            var hb = C_qpip_C if origin == 0 else C_qpip_Cpp
            for p in range(N_CPROD_HAD):
                copy_into(self.c[hb + p], out[ob + CP_PIP + p])
            var qe = C_qe_C if origin == 0 else C_qe_Cpp
            var qp = C_qp_C if origin == 0 else C_qp_Cpp
            copy_into(self.c[qe], out[ob + CP_EM])
            copy_into(self.c[qp], out[ob + CP_EP])
            var qn = C_qnue_C if origin == 0 else C_qnue_Cpp
            for k in range(N_CPROD_NU):
                copy_into(self.c[qn + k], out[ob + CP_NUE + k])
            var qg = C_qgma_C if origin == 0 else C_qgma_Cpp
            copy_into(self.c[qg], out[ob + CP_GAM])
            for i in range(self.dat.nbins_had):
                self.c[C_qpro_C][i] += self.oCharmDec.out[ob + CP_P][i]
                self.c[C_qntr_C][i] += self.oCharmDec.out[ob + CP_N][i]
        var krows = List[Evolution]()
        for o in range(2):
            if not self.charm_kaons_on(o):
                continue
            var hb = C_qpip_C if o == 0 else C_qpip_Cpp
            for s in range(N_KAON):
                krows.append(
                    Evolution(
                        (SA_K_Cpg if o == 0 else SA_K_Cpp) + s,
                        (P_kaon_plus_Cpg if o == 0 else P_kaon_plus_Cpp) + s,
                        C_aK if s <= KM else C_aK0,
                        C_AK if s <= KM else C_AK0,
                        hb + (CP_KP - CP_PIP) + s,
                        P_photons_on_had_grid,
                    )
                )
        if len(krows) > 0:
            self._kaon_charm_escape_cooling(False)
            self.evolve_group(krows)
        var kdec = List[Bool]()
        for o in range(2):
            kdec.append(
                self.charm_kaons_on(o) and self.S[SA_K_Cpg].process_decay
            )
            if not self.charm_kaons_on(o):
                continue
            var kbase = P_kaon_plus_Cpg if o == 0 else P_kaon_plus_Cpp
            for s in range(N_KAON):
                for i in range(self.dat.nbins_had):
                    self.dat.n[P_kaon_plus + s][i] += self.dat.n[kbase + s][i]
        if kdec[0] or kdec[1]:
            self.oKaonDec.Calc_KtoX(
                self.dat, P_kaon_plus_Cpg, P_kaon_plus_Cpp, kdec[0], kdec[1]
            )
        for o in range(2):
            if not kdec[o]:
                continue
            ref kout = self.oKaonDec.out
            var ob = o * N_PROD
            var kb = C_qpip_KC if o == 0 else C_qpip_KCpp
            for p in range(N_KPROD_HAD):
                copy_into(self.c[kb + p], kout[ob + PR_PIP + p])
            var qe = C_qe_C if o == 0 else C_qe_Cpp
            var qp = C_qp_C if o == 0 else C_qp_Cpp
            for i in range(len(kout[ob + PR_EM])):
                self.c[qe][i] += kout[ob + PR_EM][i]
                self.c[qp][i] += kout[ob + PR_EP][i]
            var qn = C_qnue_KC if o == 0 else C_qnue_KCpp
            for k in range(N_KPROD_NU):
                copy_into(self.c[qn + k], kout[ob + PR_NUE + k])
        comptime assert CP_PIM == CP_PIP + 1 and PR_PIM == PR_PIP + 1
        for o in range(2):
            var hb = C_qpip_C if o == 0 else C_qpip_Cpp
            var kb = C_qpip_KC if o == 0 else C_qpip_KCpp
            var q0 = C_qpi_Cpg if o == 0 else C_qpi_Cpp
            for k in range(2):
                for i in range(self.dat.nbins_had):
                    self.c[q0 + k][i] = self.c[hb + k][i] + self.c[kb + k][i]

    def _pion_to_muon_sources(mut self, pions: List[Int], blocks: List[Int]):
        self.oPionDec.Calc_PItoMU(self.dat, pions)
        for k in range(len(pions)):
            var q0 = blocks[k]
            copy_into(self.c[q0], self.oPionDec.MpLGen[k])
            copy_into(self.c[q0 + 1], self.oPionDec.MpRGen[k])
            copy_into(self.c[q0 + 2], self.oPionDec.MmLGen[k])
            copy_into(self.c[q0 + 3], self.oPionDec.MmRGen[k])

    def _add_kaon_muons(mut self, q0: Int, src: Int):
        for k in range(4):
            for i in range(self.dat.nbins_had):
                self.c[q0 + k][i] += self.c[src + k][i]

    def _zero_block(mut self, q0: Int):
        for k in range(4):
            fill_zero(self.c[q0 + k])

    def Calc_Coeff_MU(mut self):
        """Without pion decay, the pion part of the muon sources is zero (AM3
        kept the last sources)."""
        if self.verbose:
            print(" Calculate muon coefficient")
        comptime assert (
            P_pion_minus == P_pion_plus + 1
            and P_pion_minus_pg == P_pion_plus_pg + 1
            and P_pion_minus_pp == P_pion_plus_pp + 1
        )
        var pions: List[Int] = [P_pion_plus]
        var blocks: List[Int] = [C_qmupl_tot_dec]
        if self.pchain[0]:
            pions.append(P_pion_plus_pg)
            blocks.append(C_qmupl_pg)
        if self.pchain[1]:
            pions.append(P_pion_plus_pp)
            blocks.append(C_qmupl_pp)
        for o in range(2):
            if self.kaon_chain_on(o):
                pions.append(P_pion_plus_Kpg if o == 0 else P_pion_plus_Kpp)
                blocks.append(C_qmu_Kpg if o == 0 else C_qmu_Kpp)
        for o in range(2):
            if self.charm_chain_on(o):
                pions.append(P_pion_plus_Cpg if o == 0 else P_pion_plus_Cpp)
                blocks.append(C_qmu_Cpg if o == 0 else C_qmu_Cpp)
        if self.process_pion_decay:
            self._pion_to_muon_sources(pions, blocks)
        else:
            for q0 in blocks:
                self._zero_block(q0)
        # The muons of the kaon decays, and of the charm decays and their
        # kaons' decays (zero without them). They are those of the products'
        # blocks from mu+ left (cmu, kmu) on, in the order of _add_kaon_muons.
        comptime assert (
            PR_MUPR == PR_MUPL + 1
            and PR_MUML == PR_MUPL + 2
            and PR_MUMR == PR_MUPL + 3
            and CP_MUPR == CP_MUPL + 1
            and CP_MUML == CP_MUPL + 2
            and CP_MUMR == CP_MUPL + 3
        )
        comptime cmu = CP_MUPL - CP_PIP
        comptime kmu = PR_MUPL - PR_PIP
        if not self.kc_idle:
            self._add_kaon_muons(C_qmupl_tot_dec, C_qmupl_K)
            self._add_kaon_muons(C_qmupl_tot_dec, C_qmupl_Kpp)
            self._add_kaon_muons(C_qmupl_tot_dec, C_qpip_C + cmu)
            self._add_kaon_muons(C_qmupl_tot_dec, C_qpip_Cpp + cmu)
            self._add_kaon_muons(C_qmupl_tot_dec, C_qpip_KC + kmu)
            self._add_kaon_muons(C_qmupl_tot_dec, C_qpip_KCpp + kmu)
        for o in range(2):
            if self.kaon_chain_on(o):
                self._add_kaon_muons(
                    C_qmu_Kpg if o == 0 else C_qmu_Kpp,
                    C_qmupl_K if o == 0 else C_qmupl_Kpp,
                )
        for o in range(2):
            if self.charm_chain_on(o):
                var q0 = C_qmu_Cpg if o == 0 else C_qmu_Cpp
                self._add_kaon_muons(
                    q0, (C_qpip_C if o == 0 else C_qpip_Cpp) + cmu
                )
                self._add_kaon_muons(
                    q0, (C_qpip_KC if o == 0 else C_qpip_KCpp) + kmu
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
        if self.pchain[0]:
            self._block_rows(
                rows, 1, SA_MpL, P_muon_plus_left, C_qmupl, C_amu, C_Amu
            )
        if self.pchain[1]:
            self._block_rows(
                rows, 2, SA_MpL, P_muon_plus_left, C_qmupl, C_amu, C_Amu
            )
        for o in range(2):
            if self.kaon_chain_on(o):
                self._block_rows(
                    rows,
                    0,
                    SA_MpL_Kpg if o == 0 else SA_MpL_Kpp,
                    P_muon_plus_left_Kpg if o == 0 else P_muon_plus_left_Kpp,
                    C_qmu_Kpg if o == 0 else C_qmu_Kpp,
                    C_amu,
                    C_Amu,
                )
        for o in range(2):
            if self.charm_chain_on(o):
                self._block_rows(
                    rows,
                    0,
                    SA_MpL_Cpg if o == 0 else SA_MpL_Cpp,
                    P_muon_plus_left_Cpg if o == 0 else P_muon_plus_left_Cpp,
                    C_qmu_Cpg if o == 0 else C_qmu_Cpp,
                    C_amu,
                    C_Amu,
                )
        self.evolve_group(rows)

    def _decay_neutrino_sources(
        mut self,
        pions: List[Int],
        muons: List[Int],
        blocks: List[Int],
        pblocks: List[Int],
    ):
        """The neutrino sources are recomputed every step (AM3 never resets the
        total nu_mu sources with pion decay off and muon decay on)."""
        var on_p = self.process_pion_decay
        var on_m = self.process_muon_decay
        var pid = fbool(on_p)
        var mud = fbool(on_m)
        if on_p:
            self.oPionDec.Calc_PItoNU(self.dat, pions)
        if on_m:
            self.oMuonDec.Calc_MUtoNU(self.dat, muons)
        var zero = zeros(self.dat.nbins_nu)
        ref P = self.oPionDec
        ref M = self.oMuonDec
        for k in range(len(blocks)):
            var q0 = blocks[k]
            var qpi = pblocks[k]
            var pm = pd(P.NUmGen_pi[k]) if on_p else pd(zero)
            var pma = pd(P.NUmAGen_pi[k]) if on_p else pd(zero)
            var me = pd(M.NUeGen_mu[k]) if on_m else pd(zero)
            var mea = pd(M.NUeAGen_mu[k]) if on_m else pd(zero)
            var mm = pd(M.NUmGen_mu[k]) if on_m else pd(zero)
            var mma = pd(M.NUmAGen_mu[k]) if on_m else pd(zero)
            if qpi >= 0:
                for i in range(self.dat.nbins_nu):
                    self.c[q0][i] = mud * me[unsafe_offset=i]
                    self.c[q0 + 1][i] = mud * mea[unsafe_offset=i]
                    self.c[q0 + 2][i] = mud * mm[unsafe_offset=i]
                    self.c[q0 + 3][i] = mud * mma[unsafe_offset=i]
                    self.c[qpi][i] = pid * pm[unsafe_offset=i]
                    self.c[qpi + 1][i] = pid * pma[unsafe_offset=i]
                continue
            for i in range(self.dat.nbins_nu):
                self.c[q0][i] = mud * me[unsafe_offset=i]
                self.c[q0 + 1][i] = mud * mea[unsafe_offset=i]
                self.c[q0 + 2][i] = (
                    pid * pm[unsafe_offset=i] + mud * mm[unsafe_offset=i]
                )
                self.c[q0 + 3][i] = (
                    pid * pma[unsafe_offset=i] + mud * mma[unsafe_offset=i]
                )

    def _add_kaon_neutrinos(mut self, q0: Int, src: Int):
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
        var pions: List[Int] = [P_pion_plus]
        var muons: List[Int] = [P_muon_plus_left]
        var blocks: List[Int] = [C_qnue_tot_dec]
        var pblocks: List[Int] = [-1]
        if self.pchain[0]:
            pions.append(P_pion_plus_pg)
            muons.append(P_muon_plus_left_pg)
            blocks.append(C_qnue_pg)
            pblocks.append(C_qnum_pidec_pg)
        if self.pchain[1]:
            pions.append(P_pion_plus_pp)
            muons.append(P_muon_plus_left_pp)
            blocks.append(C_qnue_pp)
            pblocks.append(C_qnum_pidec_pp)
        for o in range(2):
            if self.kaon_chain_on(o):
                pions.append(P_pion_plus_Kpg if o == 0 else P_pion_plus_Kpp)
                muons.append(
                    P_muon_plus_left_Kpg if o == 0 else P_muon_plus_left_Kpp
                )
                blocks.append(C_qnu_Kpg if o == 0 else C_qnu_Kpp)
                pblocks.append(-1)
        for o in range(2):
            if self.charm_chain_on(o):
                pions.append(P_pion_plus_Cpg if o == 0 else P_pion_plus_Cpp)
                muons.append(
                    P_muon_plus_left_Cpg if o == 0 else P_muon_plus_left_Cpp
                )
                blocks.append(C_qnu_Cpg if o == 0 else C_qnu_Cpp)
                pblocks.append(-1)
        self._decay_neutrino_sources(pions, muons, blocks, pblocks)
        if self.neutron_decay_on():
            for i in range(self.dat.nbins_nu):
                self.c[C_qnuea_tot_dec][i] += self.c[C_qnuea_ndec][i]
        if not self.kc_idle:
            self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_K)
            self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_Kpp)
        # (and of the charm decays and their kaons' decays, zero without them.
        #  They are those of the products' blocks from nu_e on, in the order of
        #  _add_kaon_neutrinos.)
        comptime assert (
            PR_NUEB == PR_NUE + 1
            and PR_NUM == PR_NUE + 2
            and PR_NUMB == PR_NUE + 3
            and CP_NUEB == CP_NUE + 1
            and CP_NUM == CP_NUE + 2
            and CP_NUMB == CP_NUE + 3
        )
        if not self.kc_idle:
            self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_C)
            self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_Cpp)
            self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_KC)
            self._add_kaon_neutrinos(C_qnue_tot_dec, C_qnue_KCpp)
        for o in range(2):
            if self.kaon_chain_on(o):
                self._add_kaon_neutrinos(
                    C_qnu_Kpg if o == 0 else C_qnu_Kpp,
                    C_qnue_K if o == 0 else C_qnue_Kpp,
                )
        for o in range(2):
            if self.charm_chain_on(o):
                var q0 = C_qnu_Cpg if o == 0 else C_qnu_Cpp
                self._add_kaon_neutrinos(q0, C_qnue_C if o == 0 else C_qnue_Cpp)
                self._add_kaon_neutrinos(
                    q0, C_qnue_KC if o == 0 else C_qnue_KCpp
                )

    def Pass_Coeff_NU(mut self):
        if self.verbose:
            print("Pass neutrino coefficient")
        var esc = fbool(self.process_escape)
        var exh = fbool(self.process_expansion) * Expansion.afExpHad(self.rp)
        for i in range(self.dat.nbins_nu):
            self.c[C_Anu][i] = self.oExtraCool.AExtraCoolNU[i]
            self.c[C_anu][i] = esc * self.oEscape.afescNU[i] + exh
            # each flavor takes its own external injection (AM3 added that of
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
        if self.pchain[0]:
            self._block_rows(
                rows, 1, SA_nue, P_neutrino_el, C_qnue, C_anu, C_Anu
            )
            self._pion_decay_rows(
                rows, SA_num_pidec_pg, P_neutrino_mu_pidec_pg, C_qnum_pidec_pg
            )
        if self.pchain[1]:
            self._block_rows(
                rows, 2, SA_nue, P_neutrino_el, C_qnue, C_anu, C_Anu
            )
            self._pion_decay_rows(
                rows, SA_num_pidec_pp, P_neutrino_mu_pidec_pp, C_qnum_pidec_pp
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
        for o in range(2):
            if self.kaon_chain_on(o):
                self._block_rows(
                    rows,
                    0,
                    SA_nue_Kpg if o == 0 else SA_nue_Kpp,
                    P_neutrino_el_Kpg if o == 0 else P_neutrino_el_Kpp,
                    C_qnu_Kpg if o == 0 else C_qnu_Kpp,
                    C_anu,
                    C_Anu,
                )
        # The nu_tau and anti-nu_tau of the charm decay chains are evolved with
        # or without the components: AM3 has no nu_tau.
        comptime assert CP_NUTB == CP_NUT + 1
        comptime nut = CP_NUT - CP_NUE
        for o in range(2):
            if self.charm_chain_on(o):
                self._block_rows(
                    rows,
                    0,
                    SA_nue_Cpg if o == 0 else SA_nue_Cpp,
                    P_neutrino_el_Cpg if o == 0 else P_neutrino_el_Cpp,
                    C_qnu_Cpg if o == 0 else C_qnu_Cpp,
                    C_anu,
                    C_Anu,
                )
            if self.charm_kaons_on(o) or self._charm_tau_nonzero(o):
                for k in range(2):
                    rows.append(
                        Evolution(
                            (SA_nutau_Cpg if o == 0 else SA_nutau_Cpp) + k,
                            (
                                P_neutrino_tau_Cpg if o
                                == 0 else P_neutrino_tau_Cpp
                            )
                            + k,
                            C_anu,
                            C_Anu,
                            (C_qnue_C if o == 0 else C_qnue_Cpp) + nut + k,
                            P_photons_on_had_grid,
                        )
                    )
        self.evolve_group(rows)

    def _charm_tau_nonzero(self, origin: Int) -> Bool:
        var b = P_neutrino_tau_Cpg if origin == 0 else P_neutrino_tau_Cpp
        return self._nonzero(b) or self._nonzero(b + 1)

    def pair_annihilation_active(self) -> Bool:
        return (
            self.process_pair_annihilation
            and not self.process_merge_positrons_into_electrons
            and self.oPairAnn.ready(self.dat)
        )

    def triplet_active(self) -> Bool:
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
                    self.dat, ANNIHILATION_PAIR_EMISSION_MAX_COM_ENERGY
                )
        if self.process_hadronic:
            if (
                self.process_bethe_heitler
                and self.process_bethe_heitler_emission
            ):
                self.oBH.Calc_BH_Rad(self.dat)
            # The totals take all muons, also with the components on (AM3 summed
            # the components, which miss injected and set particles: AM3 bug).
            # All muon sets decay with the muon decay switch. (AM3 gave the
            # components the e+- of muon decay only with pion decay on too. So
            # after pion decay was switched off, they missed those of the muons
            # left.)
            var muons: List[Int] = [P_muon_plus_left]
            var qe: List[Int] = [C_qe_tot_dec]
            var qp: List[Int] = [C_qp_tot_dec]
            for o in range(2):
                if not (self.process_muon_decay and self._pair_muons(o)):
                    continue
                var first = len(muons)
                if self.pchain[o]:
                    muons.append(
                        P_muon_plus_left_pg if o == 0 else P_muon_plus_left_pp
                    )
                if self.kaon_chain_on(o):
                    muons.append(
                        P_muon_plus_left_Kpg if o == 0 else P_muon_plus_left_Kpp
                    )
                if self.charm_chain_on(o):
                    muons.append(
                        P_muon_plus_left_Cpg if o == 0 else P_muon_plus_left_Cpp
                    )
                for _ in range(first, len(muons)):
                    qe.append(C_qepg if o == 0 else C_qepp)
                    qp.append(C_qppg if o == 0 else C_qppp)
            if self.verbose:
                print("computing muon decay to pairs terms")
            self._muon_to_pair_sources(
                muons, qe, qp, fbool(self.process_muon_decay)
            )
        else:
            # no muons decay: the totals are zero, not the last step's (AM3
            # with parse_sed on summed them from the zeroed components)
            fill_zero(self.c[C_qe_tot_dec])
            fill_zero(self.c[C_qp_tot_dec])

    def _pair_muons(self, origin: Int) -> Bool:
        return (
            self.pchain[origin]
            or self.kaon_chain_on(origin)
            or self.charm_chain_on(origin)
        )

    # (not inlined: with f known to the compiler, it fuses the additions of the
    # later sets otherwise, which moves their sums at the rounding level)
    @inline(.never)
    def _muon_to_pair_sources(
        mut self, muons: List[Int], qe: List[Int], qp: List[Int], f: Float64
    ):
        """If muon decay is off, the decays are skipped: only set 0 is there,
        with f = 0."""
        if self.process_muon_decay:
            self.oMuonDec.Calc_MUtoE(self.dat, muons)
        var nl = self.dat.nbins_lep
        for k in range(len(muons)):
            var e = pd(self.oMuonDec.QMU2EGen_mu[k])
            var p = pd(self.oMuonDec.QMU2PGen_mu[k])
            var a = k > 0 and qe[k] == qe[k - 1]
            ref ce = self.c[qe[k]]
            ref cp = self.c[qp[k]]
            for i in range(nl):
                var ve = f / RATE_SSC * self.dat.grid_had2lep(i, e, nl)
                var vp = f / RATE_SSC * self.dat.grid_had2lep(i, p, nl)
                ce[i] = ve if not a else ce[i] + ve
                cp[i] = vp if not a else cp[i] + vp

    def Pass_Coeff_Lep(mut self):
        if self.verbose:
            print("Pass electron/positron coefficient")
        var nl = self.dat.nbins_lep
        var bh_on = (
            self.process_hadronic
            and self.process_bethe_heitler
            and self.process_bethe_heitler_emission
        )
        # (the muon e+- of the pg and pp components: as Calc_Coeff_Lep)
        var pg_on = (
            self.process_hadronic
            and self.process_muon_decay
            and self._pair_muons(0)
        )
        var pp_on = (
            self.process_hadronic
            and self.process_muon_decay
            and self._pair_muons(1)
        )
        var qbh = pd(self.oBH.QBH_LGen)
        for i in range(nl):
            if bh_on:
                self.c[C_qebh][i] = (
                    1.0 / RATE_SSC * self.dat.grid_had2lep(i, qbh, nl)
                )
                self.c[C_qpbh][i] = (
                    1.0 / RATE_SSC * self.dat.grid_had2lep(i, qbh, nl)
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
            # (the kaon and charm parts below are zero)
            if self.process_hadronic and self.kc_idle:
                continue
            var kon = fbool(self.kaon_decay_on())
            var r = kon / RATE_SSC
            var ke_g = r * self.dat.grid_had2lep(i, pd(self.c[C_qe_K]), nl)
            var kp_g = r * self.dat.grid_had2lep(i, pd(self.c[C_qp_K]), nl)
            var ke_p = r * self.dat.grid_had2lep(i, pd(self.c[C_qe_Kpp]), nl)
            var kp_p = r * self.dat.grid_had2lep(i, pd(self.c[C_qp_Kpp]), nl)
            var ke = ke_g + ke_p
            var kp = kp_g + kp_p
            if self.process_parse_sed:
                self.c[C_qepg][i] += ke_g
                self.c[C_qppg][i] += kp_g
                self.c[C_qepp][i] += ke_p
                self.c[C_qppp][i] += kp_p
            self.c[C_qe_tot_dec][i] += ke
            self.c[C_qp_tot_dec][i] += kp
            # e+- from the charm decays and their kaons' decays (zero without
            # them). Calc_C, which resets them, runs only with the hadronic
            # processes.
            var r_c = fbool(self.process_hadronic) / RATE_SSC
            var ce_g = r_c * self.dat.grid_had2lep(i, pd(self.c[C_qe_C]), nl)
            var cp_g = r_c * self.dat.grid_had2lep(i, pd(self.c[C_qp_C]), nl)
            var ce_p = r_c * self.dat.grid_had2lep(i, pd(self.c[C_qe_Cpp]), nl)
            var cp_p = r_c * self.dat.grid_had2lep(i, pd(self.c[C_qp_Cpp]), nl)
            if self.process_parse_sed:
                self.c[C_qepg][i] += ce_g
                self.c[C_qppg][i] += cp_g
                self.c[C_qepp][i] += ce_p
                self.c[C_qppp][i] += cp_p
            self.c[C_qe_tot_dec][i] += ce_g + ce_p
            self.c[C_qp_tot_dec][i] += cp_g + cp_p
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
        # neutron decay (am3_mojo addition): its electrons (none after a
        # switch-off, while those left evolve)
        var ndec = self.neutron_decay_on()
        if not ndec:
            fill_zero(self.c[C_qe_ndec])
        for i in range(nl):
            self.c[C_Ae][i] = adi + self.oExtraCool.AExtraCoolE[i]
            if tpp:
                self.c[C_Ae][i] -= tco * self.oTPP.eloss[i]
            self.c[C_ae][i] = esc * self.oEscape.afescE[i] + exl
            # e+e- annihilation (am3_mojo addition): electrons disappear at
            # the rate set by the positrons and vice versa
            self.c[C_ae_ele][i] = (
                self.c[C_ae][i] + pac * self.oPairAnn.Gamma_ele[i]
            )
            self.c[C_ae_pos][i] = (
                self.c[C_ae][i] + pac * self.oPairAnn.Gamma_pos[i]
            )
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
        # The loss rates with their cooling off are zero, as they do not act
        # (for the getters and the next step's estimate). The emission does
        # not take them.
        if pac == 0.0:
            fill_zero(self.oPairAnn.Gamma_ele)
            fill_zero(self.oPairAnn.Gamma_pos)
        if tco == 0.0:
            fill_zero(self.oTPP.eloss)
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
        var comps = List[Int]()

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

        self.last_step_merged = self.process_merge_positrons_into_electrons
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
            # The components of AM3's processes while they hold or get pairs
            # (also after their process is switched off: _live).
            comp(SA_epin, 0, P_pairs_in, C_qepin, C_qein, C_qpin)
            if self._live(P_pairs_pair, C_qeppair):
                comp(SA_eppair, 2, P_pairs_pair, C_qeppair, C_qepair, C_qppair)
            if self._live(P_pairs_bh, C_qepbh):
                comp(SA_epbh, 1, P_pairs_bh, C_qepbh, C_qebh, C_qpbh)
            if self._live(P_pairs_pg, C_qeppg):
                comp(SA_eppg, 3, P_pairs_pg, C_qeppg, C_qepg, C_qppg)
            if self._live(P_pairs_pp, C_qeppp):
                comp(SA_eppp, 4, P_pairs_pp, C_qeppp, C_qepp, C_qppp)
            # (also after the process is switched off, so that the component
            #  escapes and cools like the pairs it is part of)
            if (
                tpp and self.process_triplet_pair_production_emission
            ) or self._nonzero(P_pairs_tpp):
                comp(SA_eptpp, 5, P_pairs_tpp, C_qeptpp, C_qetpp, C_qptpp)
            # the electrons from neutron decay: electrons only, so they are
            # evolved whole, with the electrons' sink
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
        # The halves follow the components only while they evolve apart. In a
        # step without the components, neither changes.
        if self.process_parse_sed:
            self.pair_halves_valid = ann
        if self.process_merge_positrons_into_electrons:
            self._split_merged_pairs()
        else:
            for i in range(self.dat.nbins_lep):
                self.dat.n[P_pairs][i] = (
                    self.dat.n[P_electrons][i] + self.dat.n[P_positrons][i]
                )

    def _split_merged_pairs(mut self):
        """In a bin without e- and e+ before the step, all pairs become e-
        (merged, the pairs count as electrons, as AM3's switch puts it). The e-
        and e+ getters follow the merged pairs, and the steps after the merge is
        switched off go on from them. (AM3 kept the e- and e+ of the last step
        before the merge, and the first step after it summed them into the
        pairs. So the pairs lost what the merged steps changed.)"""
        for i in range(self.dat.nbins_lep):
            var ne = self.dat.n[P_electrons][i]
            var npo = self.dat.n[P_positrons][i]
            var fe = ne / (ne + npo) if ne + npo > 0.0 else 1.0
            self.dat.n[P_electrons][i] = fe * self.dat.n[P_pairs][i]
            self.dat.n[P_positrons][i] = (1.0 - fe) * self.dat.n[P_pairs][i]

    @staticmethod
    def _pair_component(k: Int) -> Int:
        """Density array of pair component k (P_pairs_in_ele order)."""
        match k:
            case 0:
                return P_pairs_in
            case 1:
                return P_pairs_bh
            case 2:
                return P_pairs_pair
            case 3:
                return P_pairs_pg
            case 5:
                return P_pairs_tpp
            case _:
                return P_pairs_pp

    def _split_pair_components(mut self):
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
        """The inverse-Compton emission and the charm-hadron emission run in
        parallel themselves. The other parts are independent of each other (each
        writes its own arrays). They run as the tasks of one parallel section,
        the larger first (_phot_part). Their profiler slots time each part, so
        with several threads the parts can add up to more than the stage."""
        if self.verbose:
            print(" Calculate photon coefficient")
        var t0 = self._tic()
        copy_into(self.c[C_qgma_inj], self.oInj.inj_photon)
        self._toc(21, t0)
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
                self.dat, outs, self._lep_ins(), not self.process_parse_sed
            )
        else:
            # (not the last step's, after a switch-off: AM3 bug)
            for id in [
                C_qgma_ic,
                C_qgma_inic,
                C_qgma_bhic,
                C_qgma_pairic,
                C_qgma_pgic,
                C_qgma_ppic,
                C_qgma_tppic,
                C_qgma_ndecic,
            ]:
                fill_zero(self.c[id])
        self._toc(24, t0)
        self.phot_parts.clear()
        self.phot_parts.append(22)
        if self.process_hadronic:
            self.phot_parts.append(28)
            self.phot_parts.append(29)
        self.phot_parts.append(26)
        if self.process_hadronic:
            self.phot_parts.append(34)
        self.phot_parts.append(25)
        self.phot_parts.append(23)
        if self.process_hadronic:
            self.phot_parts.append(27)
        var n = len(self.phot_parts)
        var nt = self.nthreads
        parallel_tasks[_phot_part_work](self, n, nt)
        if self.process_hadronic:
            t0 = self._tic()
            if (
                self.charm_on()
                and self.process_charm_syn
                and self.process_charm_syn_emission
            ):
                self.oSynP.Calc_SY_C(self.dat)
            else:
                fill_zero(self.oSynP.epsilonsyC)
            self._toc(T_CHARM_SYN, t0)
            t0 = self._tic()
            if (
                self.charm_on()
                and self.process_charm_compton
                and self.process_charm_compton_emission
            ):
                self.oHadIC.Calc_CIC(self.dat)
            else:
                fill_zero(self.oHadIC.epsilonCIC)
            self._toc(T_CHARM_IC, t0)
        # (slot 35, "photon coeff: escape", stays 0: AM3 times there its
        #  commented-out photon escape, which only the escape setters change)

    def _phot_part(mut self, part: Int):
        """Part `part` of Calc_Coeff_Phot (its profiler slot). Parts 28 and 29
        compute the synchrotron and inverse-Compton emission of protons, pions,
        muons and kaons. They compute each species in turn, with a shared work
        array."""
        var t0 = self._tic()
        if part == 22:
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
            else:
                # (not the last step's, after a switch-off: AM3 bug)
                for id in [
                    C_qgma_sy,
                    C_qgma_insy,
                    C_qgma_bhsy,
                    C_qgma_pairsy,
                    C_qgma_pgsy,
                    C_qgma_ppsy,
                    C_qgma_tppsy,
                    C_qgma_ndecsy,
                ]:
                    fill_zero(self.c[id])
            self._toc(22, t0)
        elif part == 23:
            # (with the gate of Pass_Coeff_Phot, and zero while off, as the
            #  getter reads the rates. AM3 computed them with the e+-
            #  synchrotron off, and kept those of its last step with SSA.)
            if self.process_electron_syn and self.process_ssa:
                self.oSyn.Calc_SY_Phot_Sink(self.dat)
            else:
                fill_zero(self.oSyn.afsyG)
            self._toc(23, t0)
        elif part == 25:
            if (
                self.process_electron_compton
                and self.process_compton_photon_energy_loss
            ):
                self.oComp.Calc_IC_Phot_Loss(self.dat)
            else:
                fill_zero(self.oComp.aficG)
            self._toc(25, t0)
        elif part == 26:
            # (The pair emission takes its number from these rates, also with
            #  the photon loss off. AM3 refreshed them only with it on, so
            #  pairs came from the rates of its last step, or none at all. The
            #  rates are zero while the process is off. So the step after it is
            #  switched on again makes no pairs of rates before the switch-off.)
            if self.process_annihilation and (
                self.process_annihilation_cooling
                or self.process_annihilation_emission
            ):
                self.oPair.Calc_Pair_Phot(self.dat)
            else:
                fill_zero(self.oPair.afpairG)
            self._toc(26, t0)
        elif part == 27:
            if self.process_photopion and self.process_photopion_photon_loss:
                self.oPhotoHadron.Calc_PhotoHadron_PhotonLoss(self.dat)
            else:
                fill_zero(self.oPhotoHadron.afPG_Rad)
            self._toc(27, t0)
        elif part == 28:
            if self.process_proton_syn and self.process_proton_syn_emission:
                self.oSynP.Calc_SY_PRO(self.dat)
            self._toc(28, t0)
            t0 = self._tic()
            if self.process_pion_syn and self.process_pion_syn_emission:
                self.oSynP.Calc_SY_Pi(self.dat)
            self._toc(30, t0)
            t0 = self._tic()
            if self.process_muon_syn and self.process_muon_syn_emission:
                self.oSynP.Calc_SY_Mu(self.dat)
            self._toc(32, t0)
            t0 = self._tic()
            if (
                self.any_kaons_on()
                and self.process_kaon_syn
                and self.process_kaon_syn_emission
            ):
                self.oSynP.Calc_SY_K(self.dat)
            else:
                fill_zero(self.oSynP.epsilonsyK)
            self._toc(T_KAON_SYN, t0)
        elif part == 29:
            if (
                self.process_proton_compton
                and self.process_proton_compton_emission
            ):
                self.oHadIC.Calc_PIC(self.dat)
            self._toc(29, t0)
            t0 = self._tic()
            if self.process_pion_compton and self.process_pion_compton_emission:
                self.oHadIC.Calc_PiIC(self.dat)
            self._toc(31, t0)
            t0 = self._tic()
            if self.process_muon_compton and self.process_muon_compton_emission:
                self.oHadIC.Calc_MuIC(self.dat)
            self._toc(33, t0)
            t0 = self._tic()
            if (
                self.any_kaons_on()
                and self.process_kaon_compton
                and self.process_kaon_compton_emission
            ):
                self.oHadIC.Calc_KIC(self.dat)
            else:
                fill_zero(self.oHadIC.epsilonKIC)
            self._toc(T_KAON_IC, t0)
        elif part == 34:
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
                            1.0 / RATE_SSC * self.dat.grid_had2lep(i, eg, ng)
                        )
                # the direct pp photons, for the total
                copy_into(self.pi0_pp_direct, self.c[C_qgma_pi0_pp])
                if self.process_parse_sed:
                    # C_qpi0_pg holds the pi0 of the photo-hadronic
                    # interactions and of the decays of their kaons and charm
                    # hadrons (also those left after their production stops)
                    if (
                        (
                            self.process_photopion
                            and self.process_photopion_emission
                        )
                        or self.kaons_ev[0]
                        or self.charm_ev[0]
                        or self.charm_kaons_on(0)
                    ):
                        self.oPionDec.Calc_PItoG(self.dat, self.c[C_qpi0_pg])
                        var qg = pd(self.oPionDec.QPI2GGen)
                        for i in range(ng):
                            self.c[C_qgma_pi0_pg][i] = (
                                1.0
                                / RATE_SSC
                                * self.dat.grid_had2lep(i, qg, ng)
                            )
                    # C_qpi0_pp holds the pp pi0 (into the cascade) and the pi0
                    # from the decays of pp kaons and charm hadrons (as for the
                    # photo-hadronic ones)
                    if (
                        (
                            self.process_pp
                            and self.process_pp_emission
                            and self.process_pp_emission_pi0_into_cascade
                        )
                        or self.kaons_ev[1]
                        or self.charm_ev[1]
                        or self.charm_kaons_on(1)
                    ):
                        self.oPionDec.Calc_PItoG(self.dat, self.c[C_qpi0_pp])
                        var qg = pd(self.oPionDec.QPI2GGen)
                        for i in range(ng):
                            self.c[C_qgma_pi0_pp][i] += (
                                1.0
                                / RATE_SSC
                                * self.dat.grid_had2lep(i, qg, ng)
                            )
                # All pi0 (pg, pp and injected) plus the direct pp photons, with
                # or without the components. (AM3 summed the components, which
                # miss injected pi0, and dropped the direct pp photons without
                # them.)
                self.oPionDec.Calc_PItoG(self.dat, self.c[C_qpi0])
                var qg = pd(self.oPionDec.QPI2GGen)
                for i in range(ng):
                    self.c[C_qgma_pi0][i] = (
                        1.0 / RATE_SSC * self.dat.grid_had2lep(i, qg, ng)
                    ) + self.pi0_pp_direct[i]
            # the photons of the charm decays (for example eta -> gamma gamma):
            # to the pi0-decay photons of their origin (am3_mojo addition)
            if self.charm_decay_on():
                var r = 1.0 / RATE_SSC
                var qgc = pd(self.c[C_qgma_C])
                var qgp = pd(self.c[C_qgma_Cpp])
                for i in range(ng):
                    var vg = r * self.dat.grid_had2lep(i, qgc, ng)
                    var vp = r * self.dat.grid_had2lep(i, qgp, ng)
                    if self.process_parse_sed:
                        self.c[C_qgma_pi0_pg][i] += vg
                        self.c[C_qgma_pi0_pp][i] += vp
                    self.c[C_qgma_pi0][i] += vg + vp
            self._toc(34, t0)

    def Pass_Coeff_Phot(mut self):
        if self.verbose:
            print("Pass photon coefficient")
        var ng = self.dat.nbins_pho
        if self.process_hadronic:
            var r = 1.0 / RATE_SSC
            var psy = (
                fbool(
                    self.process_proton_syn and self.process_proton_syn_emission
                )
                / RATE_SSC
            )
            var pic = (
                fbool(
                    self.process_proton_compton
                    and self.process_proton_compton_emission
                )
                / RATE_SSC
            )
            var pisy = (
                fbool(self.process_pion_syn and self.process_pion_syn_emission)
                / RATE_SSC
            )
            var piic = (
                fbool(
                    self.process_pion_compton
                    and self.process_pion_compton_emission
                )
                / RATE_SSC
            )
            var musy = (
                fbool(self.process_muon_syn and self.process_muon_syn_emission)
                / RATE_SSC
            )
            var muic = (
                fbool(
                    self.process_muon_compton
                    and self.process_muon_compton_emission
                )
                / RATE_SSC
            )
            var pgl = (
                fbool(
                    self.process_photopion
                    and self.process_photopion_photon_loss
                )
                * T_SSC
            )
            var e_psy = pd(self.oSynP.epsilonsyP)
            var e_pic = pd(self.oHadIC.epsilonPIC)
            var e_pisy = pd(self.oSynP.epsilonsyPi)
            var e_piic = pd(self.oHadIC.epsilonPiIC)
            var e_musy = pd(self.oSynP.epsilonsyMu)
            var e_muic = pd(self.oHadIC.epsilonMuIC)
            var e_ksy = pd(self.oSynP.epsilonsyK)
            var e_kic = pd(self.oHadIC.epsilonKIC)
            var e_csy = pd(self.oSynP.epsilonsyC)
            var e_cic = pd(self.oHadIC.epsilonCIC)
            var e_pg = pd(self.oPhotoHadron.afPG_Rad)
            for i in range(ng):
                self.c[C_qgma_psy][i] = psy * self.dat.grid_had2lep(
                    i, e_psy, ng
                )
                self.c[C_qgma_pic][i] = pic * self.dat.grid_had2lep(
                    i, e_pic, ng
                )
                self.c[C_qgma_pisy][i] = pisy * self.dat.grid_had2lep(
                    i, e_pisy, ng
                )
                self.c[C_qgma_piic][i] = piic * self.dat.grid_had2lep(
                    i, e_piic, ng
                )
                self.c[C_qgma_musy][i] = musy * self.dat.grid_had2lep(
                    i, e_musy, ng
                )
                self.c[C_qgma_muic][i] = muic * self.dat.grid_had2lep(
                    i, e_muic, ng
                )
                if self.kc_idle:  # (their emission is zero)
                    self.c[C_qgma_ksy][i] = 0.0
                    self.c[C_qgma_kic][i] = 0.0
                    self.c[C_qgma_csy][i] = 0.0
                    self.c[C_qgma_cic][i] = 0.0
                else:
                    self.c[C_qgma_ksy][i] = r * self.dat.grid_had2lep(
                        i, e_ksy, ng
                    )
                    self.c[C_qgma_kic][i] = r * self.dat.grid_had2lep(
                        i, e_kic, ng
                    )
                    self.c[C_qgma_csy][i] = r * self.dat.grid_had2lep(
                        i, e_csy, ng
                    )
                    self.c[C_qgma_cic][i] = r * self.dat.grid_had2lep(
                        i, e_cic, ng
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
                    + self.c[C_qgma_csy][i]
                    + self.c[C_qgma_cic][i]
                )
                self.c[C_agma_pg][i] = pgl * self.dat.grid_had2lep(i, e_pg, ng)
        else:
            # (the photon components of hadronic origin too: AM3 evolved them
            #  with the sources of the last step with the hadronic processes)
            for id in [
                C_qgma_had,
                C_agma_pg,
                C_qgma_pi0,
                C_qgma_pi0_pg,
                C_qgma_pi0_pp,
                C_qgma_psy,
                C_qgma_pic,
                C_qgma_pisy,
                C_qgma_piic,
                C_qgma_musy,
                C_qgma_muic,
                C_qgma_ksy,
                C_qgma_kic,
                C_qgma_csy,
                C_qgma_cic,
            ]:
                fill_zero(self.c[id])
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
            self.c[C_qgma_pann][i] = pae * self.oPairAnn.Q_gamma[i]
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
            # The photons of the triplet pairs and of the neutron-decay
            # electrons: their e+- are evolved also after the process is
            # switched off, and radiate.
            if self._live(P_photons_tppsy, C_qgma_tppsy):
                comp(SA_gtppsy, P_photons_tppsy, C_qgma_tppsy)
            if self._live(P_photons_tppic, C_qgma_tppic):
                comp(SA_gtppic, P_photons_tppic, C_qgma_tppic)
            if self._live(P_photons_ndecsy, C_qgma_ndecsy):
                comp(SA_gndecsy, P_photons_ndecsy, C_qgma_ndecsy)
            if self._live(P_photons_ndecic, C_qgma_ndecic):
                comp(SA_gndecic, P_photons_ndecic, C_qgma_ndecic)
            if self.any_kaons_on() or self._nonzero(P_photons_kasy):
                comp(SA_gksy, P_photons_kasy, C_qgma_ksy)
            if self.any_kaons_on() or self._nonzero(P_photons_kaic):
                comp(SA_gkic, P_photons_kaic, C_qgma_kic)
            if self.charm_on() or self._nonzero(P_photons_chsy):
                comp(SA_gcsy, P_photons_chsy, C_qgma_csy)
            if self.charm_on() or self._nonzero(P_photons_chic):
                comp(SA_gcic, P_photons_chic, C_qgma_cic)
        comp(SA_g1, P_photons_except_injected, C_qgma1)
        self.evolve_group(rows)

    def Update_Mag_Field(mut self, b_Gauss: Float64):
        """The synchrotron kernels of the field are built before their first use
        (_build_syn_kernels)."""
        if self.verbose:
            print("Magnetic field strength updated to ", b_Gauss)
        self.rp.mag_field = b_Gauss

    def _build_syn_kernels(mut self):
        """The synchrotron kernels of the field in effect, built before a step
        unless they are built for it. (AM3 builds them at every set_mag_field
        and at init_kernels, for the field then. So it builds them twice for an
        init_kernels and a set_mag_field.) Every kernel of the field is built,
        whatever the emission and SSA switches, so that one switched on later
        finds it. (AM3 built only those of the switches on. The e+- one, emptied
        otherwise, then gave no emission and no SSA, and the hadrons' kept an
        earlier field.)"""
        var b = self.rp.mag_field
        if not (
            self.syn_built
            and self.syn_field == b
            and self.syn_qsyn == self.process_quantum_syn
        ):
            if self.verbose:
                print("synchrotron kernels for e+/- in ", b, " G")
            self.oSyn.b_gauss = b
            self.oSyn.reinitialize_empty_kernels(self.dat)
            self.oSyn.initialize_kernels(self.dat, self.process_quantum_syn)
            self.syn_built = True
            self.syn_field = b
            self.syn_qsyn = self.process_quantum_syn
        if self.process_hadronic and not (
            self.synp_built and self.synp_field == b
        ):
            # (Those of the kaons and charm hadrons where they can be made:
            #  their kernels are built at init_kernels. Charm decays make
            #  kaons.)
            var charm = (
                self.oPhotoHadron.charm_built or self.oProtonProton.charm_built
            )
            var kaons = (
                charm
                or self.oPhotoHadron.kaons_built
                or self.oProtonProton.kaons_built
            )
            if self.verbose:
                print("synchrotron kernels for hadrons in ", b, " G")
            self.oSynP.b_gauss = b
            self.oSynP.initialize_kernels(self.dat, kaons, charm)
            self.synp_built = True
            self.synp_field = b

    def pair_solver(self) -> Int:
        """The solver that evolved the e+- in the last step: SA_ep if it merged
        the positrons, whatever the merge switch is now."""
        return SA_ep if self.last_step_merged else SA_e

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
        # the triplet pair-production loss (am3_mojo addition), which the e+-
        # cooling of Pass_Coeff_Lep has too
        var tpp = self.triplet_active()
        var tco = fbool(tpp and self.process_triplet_pair_production_cooling)
        # (AM3 used SA_e, which is not evolved with the positrons merged: no
        # cooling, AM3 bug)
        var k = self.pair_solver()
        # (with the extra cooling, which the e+- cooling has too: AM3 left it
        #  out)
        for i in range(nl):
            total_loss[i] = (
                -sy * self.S[k].Asy[i]
                - ic * self.S[k].Aic[i]
                - adi
                - self.oExtraCool.AExtraCoolE[i]
            )
            if tpp:
                total_loss[i] += tco * self.oTPP.eloss[i]
        var imax = 0
        while (
            imax < nl
            and self.oAcc.AAClepBohm(self.rp, self.dat, imax)
            >= total_loss[imax]
        ):
            imax += 1
        # If acceleration wins on the whole grid, the C++ code reads one element
        # past the end of egrid_lep. Use the next grid point exp(n*dlnE)
        # instead.
        var e = self.dat.egrid_lep[imax] if imax < nl else c_exp(
            Float64(imax) * self.dat.dlnE
        )
        self.rp.estimated_electron_emax = e * MASS_E * 1e9

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
        # (with the extra cooling, which the proton cooling has too: AM3 left
        #  it out)
        for i in range(nh):
            total_loss[i] = (
                -sy * self.S[SA_proton].Asy[i]
                - ic * self.S[SA_proton].Aic[i]
                - adi
                + pg * self.oPhotoHadron.afPG_PRO[i]
                - pp * self.oProtonProton.APP_PRO[i]
                - bh * self.oBH.ABHHadP[i]
                - self.oExtraCool.AExtraCoolPRO[i]
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
