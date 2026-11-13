comptime P_HAD_BEGIN = 0
comptime P_protons = 0
comptime P_neutrons = 1
comptime P_pion_plus = 2
comptime P_pion_minus = 3
comptime P_pion_0 = 4
comptime P_pion_plus_pg = 5
comptime P_pion_minus_pg = 6
comptime P_pion_0_pg = 7
comptime P_pion_plus_pp = 8
comptime P_pion_minus_pp = 9
comptime P_pion_0_pp = 10
comptime P_muon_plus_left = 11
comptime P_muon_plus_right = 12
comptime P_muon_minus_left = 13
comptime P_muon_minus_right = 14
comptime P_muon_plus_left_pg = 15
comptime P_muon_plus_right_pg = 16
comptime P_muon_minus_left_pg = 17
comptime P_muon_minus_right_pg = 18
comptime P_muon_plus_left_pp = 19
comptime P_muon_plus_right_pp = 20
comptime P_muon_minus_left_pp = 21
comptime P_muon_minus_right_pp = 22
comptime P_HAD_END = 23

comptime P_NU_BEGIN = 23
comptime P_neutrino_el = 23
comptime P_neutrino_el_anti = 24
comptime P_neutrino_mu = 25
comptime P_neutrino_mu_anti = 26
comptime P_neutrino_el_pg = 27
comptime P_neutrino_el_anti_pg = 28
comptime P_neutrino_mu_pg = 29
comptime P_neutrino_mu_anti_pg = 30
comptime P_neutrino_el_pp = 31
comptime P_neutrino_el_anti_pp = 32
comptime P_neutrino_mu_pp = 33
comptime P_neutrino_mu_anti_pp = 34
comptime P_NU_END = 35

comptime P_LEP_BEGIN = 35
comptime P_electrons = 35
comptime P_positrons = 36
comptime P_pairs = 37
comptime P_pairs_in = 38
comptime P_pairs_bh = 39
comptime P_pairs_pair = 40
comptime P_pairs_pg = 41
comptime P_pairs_pp = 42
comptime P_LEP_END = 43

comptime P_PHO_BEGIN = 43
comptime P_photons = 43
comptime P_photons_except_injected = 44
comptime P_photons_insy = 45
comptime P_photons_inic = 46
comptime P_photons_bhsy = 47
comptime P_photons_bhic = 48
comptime P_photons_pairsy = 49
comptime P_photons_pairic = 50
comptime P_photons_pgsy = 51
comptime P_photons_pgic = 52
comptime P_photons_ppsy = 53
comptime P_photons_ppic = 54
comptime P_photons_pi0 = 55
comptime P_photons_pi0_pg = 56
comptime P_photons_pi0_pp = 57
comptime P_photons_psy = 58
comptime P_photons_pic = 59
comptime P_photons_pisy = 60
comptime P_photons_piic = 61
comptime P_photons_musy = 62
comptime P_photons_muic = 63
comptime P_photons_inj = 64
comptime P_photons_on_had_grid = 65
comptime P_PHO_END = 66

# Arrays of the processes am3_mojo adds to AM3 follow AM3's arrays, so that
# those keep their indices.
comptime P_EXT_LEP_BEGIN = 66
# e- and e+ parts of the pair components (P_pairs_in_ele + k,
# P_pairs_in_pos + k), in the order injected, Bethe-Heitler, gamma-gamma pair
# production, photo-hadronic, pp, triplet pair production. They are evolved
# apart while e+e- annihilation acts, whose sink differs for e- and e+.
comptime P_pairs_in_ele = 66
comptime P_pairs_in_pos = 72
comptime N_PAIR_COMP = 6
comptime P_pairs_tpp = 78
# electrons from neutron decay: a component of the e+-, electrons only, so it
# is evolved with the electrons' sink and has no e-/e+ parts
comptime P_pairs_ndec = 79
comptime P_EXT_LEP_END = 80
comptime P_EXT_PHO_BEGIN = 80
comptime P_photons_pair_ann = 80
comptime P_photons_tppsy = 81
comptime P_photons_tppic = 82
comptime P_photons_ndecsy = 83
comptime P_photons_ndecic = 84
comptime P_photons_kasy = 85
comptime P_photons_kaic = 86
comptime P_photons_chsy = 87
comptime P_photons_chic = 88
comptime P_EXT_PHO_END = 89
comptime P_EXT_HAD_BEGIN = 89
# all kaons: the sum of the two origins and of the kaons of the charm decays
comptime P_kaon_plus = 89
comptime P_kaon_minus = 90
comptime P_kaon_0S = 91
comptime P_kaon_0L = 92
comptime P_kaon_plus_pg = 93
comptime P_kaon_plus_pp = 97
# Per kaon origin, photo-hadronic (_Kpg) or pp (_Kpp): the pions of the kaon
# decays (pi+, pi-) and the muons of the kaon decays and of these pions
# (mu+ left, mu+ right, mu- left, mu- right). Each of AM3's photo-hadronic and
# pp components of the pions and muons is the sum of three parts. These are
# this part, the part from the pions made in the interactions (P_*_pg,
# P_*_pp) and that of the charm decays (below).
comptime P_pion_plus_Kpg = 101
comptime P_pion_minus_Kpg = 102
comptime P_pion_plus_Kpp = 103
comptime P_pion_minus_Kpp = 104
comptime P_muon_plus_left_Kpg = 105
comptime P_muon_plus_left_Kpp = 109
comptime P_charm = 113
comptime P_charm_pg = 121
comptime P_charm_pp = 129
# per charm origin, photo-hadronic (_Cpg) or pp (_Cpp), the decay chain of the
# charm hadrons: the kaons of the charm decays (K+, K-, K0_S, K0_L), the pions
# of the charm decays and of these kaons' decays (pi+, pi-), and the muons of
# all their decays (mu+ left, mu+ right, mu- left, mu- right)
comptime P_kaon_plus_Cpg = 137
comptime P_kaon_plus_Cpp = 141
comptime P_pion_plus_Cpg = 145
comptime P_pion_plus_Cpp = 147
comptime P_muon_plus_left_Cpg = 149
comptime P_muon_plus_left_Cpp = 153
comptime P_EXT_HAD_END = 157
comptime P_EXT_NU_BEGIN = 157
comptime P_neutrino_el_anti_ndec = 157
# per kaon origin: the neutrinos of the kaon decays and of the decays of the
# pions and muons above (nu_e, anti-nu_e, nu_mu, anti-nu_mu)
comptime P_neutrino_el_Kpg = 158
comptime P_neutrino_el_Kpp = 162
# Per origin, photo-hadronic (_pg) or pp (_pp): the nu_mu and anti-nu_mu of the
# decays of the pions made in the interactions. The neutrino components
# P_neutrino_*_pg and _pp hold those of the decays of these pions' muons. AM3's
# photo-hadronic and pp neutrino components are the sums of the three parts and
# the charm decay chain's (below).
comptime P_neutrino_mu_pidec_pg = 166
comptime P_neutrino_mu_pidec_pp = 168
# Per charm origin: the neutrinos of the charm decay chain (nu_e, anti-nu_e,
# nu_mu, anti-nu_mu), and its nu_tau and anti-nu_tau (of D(s) -> tau nu and
# the tau decays). AM3 has no species for these tau neutrinos, so they are not
# part of the totals P_neutrino_el ... P_neutrino_mu_anti.
comptime P_neutrino_el_Cpg = 170
comptime P_neutrino_el_Cpp = 174
comptime P_neutrino_tau_Cpg = 178
comptime P_neutrino_tau_Cpp = 180
comptime P_EXT_NU_END = 182

comptime P_COUNT = 182

# e+- arrays whose synchrotron and inverse-Compton photons are computed: all
# pairs, the N_PAIR_COMP components and the neutron-decay electrons
comptime N_LEP_IN = 2 + N_PAIR_COMP

# kaon species s: arrays P_kaon_plus + s, P_kaon_plus_pg + s and
# P_kaon_plus_pp + s, solvers ids.kaon_solver(s, pp)
comptime N_KAON = 4
comptime KP = 0
comptime KM = 1
comptime KS = 2
comptime KL = 3

# charm species s (PDG order, each particle before its antiparticle): arrays
# P_charm + s, P_charm_pg + s and P_charm_pp + s,
# solvers ids.charm_solver(s, pp)
comptime N_CHARM = 8
comptime CDP = 0
comptime CDM = 1
comptime CD0 = 2
comptime CD0B = 3
comptime CDSP = 4
comptime CDSM = 5
comptime CLC = 6
comptime CLCB = 7


@inline(.always)
def charged_charm_species(k: Int) -> Int:
    """The charged charm hadrons k = 0, 1, 2 (D+, Ds+, Lambda_c+), in the order
    of their synchrotron and inverse-Compton kernels. Each antiparticle is at
    the species + 1."""
    return CDP if k == 0 else (CDSP if k == 1 else CLC)
