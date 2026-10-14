"""Indices of the SolverAlgorithm instances held by SimulationManager
(one per species / SED component, as in SimulationManager.h)."""

comptime SA_proton = 0
comptime SA_neutron = 1
comptime SA_Pip = 2
comptime SA_Pim = 3
comptime SA_Pip_pg = 4
comptime SA_Pim_pg = 5
comptime SA_Pip_pp = 6
comptime SA_Pim_pp = 7
comptime SA_MpL = 8
comptime SA_MpR = 9
comptime SA_MmL = 10
comptime SA_MmR = 11
comptime SA_MpL_pg = 12
comptime SA_MpR_pg = 13
comptime SA_MmL_pg = 14
comptime SA_MmR_pg = 15
comptime SA_MpL_pp = 16
comptime SA_MpR_pp = 17
comptime SA_MmL_pp = 18
comptime SA_MmR_pp = 19
comptime SA_nue = 20
comptime SA_nueA = 21
comptime SA_num = 22
comptime SA_numA = 23
comptime SA_nue_pg = 24
comptime SA_nueA_pg = 25
comptime SA_num_pg = 26
comptime SA_numA_pg = 27
comptime SA_nue_pp = 28
comptime SA_nueA_pp = 29
comptime SA_num_pp = 30
comptime SA_numA_pp = 31
comptime SA_e = 32
comptime SA_p = 33
comptime SA_ep = 34
comptime SA_epin = 35
comptime SA_eppair = 36
comptime SA_epbh = 37
comptime SA_eppg = 38
comptime SA_eppp = 39
comptime SA_g = 40
comptime SA_g1 = 41
comptime SA_ginsy = 42
comptime SA_ginic = 43
comptime SA_gpairsy = 44
comptime SA_gpairic = 45
comptime SA_gbhsy = 46
comptime SA_gbhic = 47
comptime SA_gpgsy = 48
comptime SA_gpgic = 49
comptime SA_gppsy = 50
comptime SA_gppic = 51
comptime SA_gpsy = 52
comptime SA_gpic = 53
comptime SA_gpisy = 54
comptime SA_gpiic = 55
comptime SA_gmusy = 56
comptime SA_gmuic = 57
comptime SA_gpi0_pg = 58
comptime SA_gpi0_pp = 59
comptime SA_ginj = 60
# solvers of the species am3_mojo adds to AM3
comptime SA_gpann = 61  # e+e- annihilation photon component
comptime SA_eptpp = 62  # pairs from triplet pair production
comptime SA_gtppsy = 63  # their synchrotron photons
comptime SA_gtppic = 64  # their inverse-Compton photons
comptime SA_endec = 65  # electrons from neutron decay
comptime SA_gndecsy = 66  # their synchrotron photons
comptime SA_gndecic = 67  # their inverse-Compton photons
comptime SA_nueA_ndec = 68  # anti-nu_e from neutron decay
comptime SA_K = 69  # kaons from photo-hadronic interactions, SA_K + s
comptime SA_Km = 70
comptime SA_KS = 71
comptime SA_KL = 72
comptime SA_gksy = 73  # kaon synchrotron photon component
comptime SA_gkic = 74  # kaon inverse-Compton photon component
comptime SA_Kpp = 75  # kaons from pp collisions, SA_Kpp + s
# pions, muons and neutrinos from the decays of the photo-hadronic and the pp
# kaons (model/particles.mojo, P_pion_plus_Kpg ...)
comptime SA_Pip_Kpg = 79  # pi+, pi-
comptime SA_Pip_Kpp = 81
comptime SA_MpL_Kpg = 83  # + 0 .. 3
comptime SA_MpL_Kpp = 87
comptime SA_nue_Kpg = 91  # + 0 .. 3
comptime SA_nue_Kpp = 95
comptime N_SOLVERS = 99


@always_inline
def kaon_solver(s: Int, pp: Bool = False) -> Int:
    """Solver of kaon species s (model/particles.mojo: K+, K-, K0_S, K0_L), of
    the photo-hadronic or (pp) the pp component; both are configured the same.
    """
    return (SA_Kpp if pp else SA_K) + s
