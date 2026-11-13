comptime C_HAD_BEGIN = 0
comptime C_apro = 0
comptime C_qpro = 1
comptime C_Apro = 2
comptime C_antr = 3
comptime C_qntr = 4
comptime C_Antr = 5
comptime C_apip = 6
comptime C_qpip = 7
comptime C_Apip = 8
comptime C_apim = 9
comptime C_qpim = 10
comptime C_Apim = 11
comptime C_qpi0 = 12
comptime C_qpip_pg = 13
comptime C_qpip_pp = 14
comptime C_qpip_ext = 15
comptime C_qpim_pg = 16
comptime C_qpim_pp = 17
comptime C_qpim_ext = 18
comptime C_qpi0_pg = 19
comptime C_qpi0_pp = 20
comptime C_qpi0_ext = 21
comptime C_amu = 22
comptime C_Amu = 23
comptime C_qmupl = 24
comptime C_qmupr = 25
comptime C_qmuml = 26
comptime C_qmumr = 27
comptime C_qmupl_pg = 28
comptime C_qmupr_pg = 29
comptime C_qmuml_pg = 30
comptime C_qmumr_pg = 31
comptime C_qmupl_pp = 32
comptime C_qmupr_pp = 33
comptime C_qmuml_pp = 34
comptime C_qmumr_pp = 35
comptime C_qmupl_tot_dec = 36
comptime C_qmupr_tot_dec = 37
comptime C_qmuml_tot_dec = 38
comptime C_qmumr_tot_dec = 39
comptime C_qmupl_ext = 40
comptime C_qmupr_ext = 41
comptime C_qmuml_ext = 42
comptime C_qmumr_ext = 43
comptime C_HAD_END = 44

comptime C_NU_BEGIN = 44
comptime C_anu = 44
comptime C_Anu = 45
comptime C_qnue = 46
comptime C_qnuea = 47
comptime C_qnum = 48
comptime C_qnuma = 49
comptime C_qnue_pg = 50
comptime C_qnuea_pg = 51
comptime C_qnum_pg = 52
comptime C_qnuma_pg = 53
comptime C_qnue_pp = 54
comptime C_qnuea_pp = 55
comptime C_qnum_pp = 56
comptime C_qnuma_pp = 57
comptime C_qnue_tot_dec = 58
comptime C_qnuea_tot_dec = 59
comptime C_qnum_tot_dec = 60
comptime C_qnuma_tot_dec = 61
comptime C_qnue_ext = 62
comptime C_qnuea_ext = 63
comptime C_qnum_ext = 64
comptime C_qnuma_ext = 65
comptime C_NU_END = 66

comptime C_LEP_BEGIN = 66
comptime C_ae = 66
comptime C_Ae = 67
comptime C_qele = 68
comptime C_qpos = 69
comptime C_qein = 70
comptime C_qebh = 71
comptime C_qepair = 72
comptime C_qepg = 73
comptime C_qepp = 74
comptime C_qpin = 75
comptime C_qpbh = 76
comptime C_qppair = 77
comptime C_qppg = 78
comptime C_qppp = 79
comptime C_qe_tot_dec = 80
comptime C_qp_tot_dec = 81
comptime C_qep = 82
comptime C_qepin = 83
comptime C_qepbh = 84
comptime C_qeppair = 85
comptime C_qeppg = 86
comptime C_qeppp = 87
comptime C_LEP_END = 88

comptime C_PHO_BEGIN = 88
comptime C_agma = 88
comptime C_Agma = 89
comptime C_qgma = 90
comptime C_qgma1 = 91
comptime C_qgma_save = 92
comptime C_qgma_inj = 93
comptime C_agma_pg = 94
comptime C_qgma_sy = 95
comptime C_qgma_ic = 96
comptime C_qgma_insy = 97
comptime C_qgma_inic = 98
comptime C_qgma_bhsy = 99
comptime C_qgma_bhic = 100
comptime C_qgma_pairsy = 101
comptime C_qgma_pairic = 102
comptime C_qgma_pgsy = 103
comptime C_qgma_pgic = 104
comptime C_qgma_ppsy = 105
comptime C_qgma_ppic = 106
comptime C_qgma_pi0 = 107
comptime C_qgma_pi0_pg = 108
comptime C_qgma_pi0_pp = 109
comptime C_qgma_psy = 110
comptime C_qgma_pic = 111
comptime C_qgma_pisy = 112
comptime C_qgma_piic = 113
comptime C_qgma_musy = 114
comptime C_qgma_muic = 115
comptime C_qgma_had = 116
comptime C_PHO_END = 117

# The coefficients of the processes that am3_mojo adds to AM3 come after AM3's,
# which keep their indices.
comptime C_EXT_LEP_BEGIN = 117
comptime C_ae_ele = 117
comptime C_ae_pos = 118
comptime C_qetpp = 119
comptime C_qptpp = 120
comptime C_qeptpp = 121
comptime C_qe_ndec = 122  # e- from neutron decay (lepton solver units)
comptime C_qe_K = 123  # e-, e+ from kaon decay (hadron-anchored lepton grid)
comptime C_qp_K = 124
comptime C_qe_Kpp = 125
comptime C_qp_Kpp = 126
# e-, e+ from the decays of the charm hadrons and of their kaons
# (hadron-anchored lepton grid)
comptime C_qe_C = 127
comptime C_qp_C = 128
comptime C_qe_Cpp = 129
comptime C_qp_Cpp = 130
comptime C_EXT_LEP_END = 131
comptime C_EXT_PHO_BEGIN = 131
comptime C_qgma_pann = 131
comptime C_qgma_tppsy = 132
comptime C_qgma_tppic = 133
comptime C_qgma_ndecsy = 134
comptime C_qgma_ndecic = 135
comptime C_qgma_ksy = 136
comptime C_qgma_kic = 137
comptime C_qgma_csy = 138
comptime C_qgma_cic = 139
# photons of the decays of the photo-hadronic / pp charm hadrons, on the
# hadron-anchored photon grid
comptime C_qgma_C = 140
comptime C_qgma_Cpp = 141
comptime C_EXT_PHO_END = 142
comptime C_EXT_HAD_BEGIN = 142
comptime C_qpro_ndec = 142
comptime C_aK = 143
comptime C_AK = 144
# kaon sources of species s: C_qK + s from photo-hadronic interactions,
# C_qKpp + s from pp collisions
comptime C_qK = 145
comptime C_qpip_K = 149
comptime C_qpim_K = 150
comptime C_qpi0_K = 151
comptime C_qmupl_K = 152
comptime C_qmupr_K = 153
comptime C_qmuml_K = 154
comptime C_qmumr_K = 155
comptime C_aK0 = 156
comptime C_AK0 = 157
comptime C_qKpp = 158
comptime C_qpip_Kpp = 162
comptime C_qpim_Kpp = 163
comptime C_qpi0_Kpp = 164
comptime C_qmupl_Kpp = 165
# the sources of the muons P_muon_plus_left_Kpg / _Kpp + 0 .. 3
comptime C_qmu_Kpg = 169
comptime C_qmu_Kpp = 173
# charm-hadron sources of species s: C_qC + s from photo-hadronic interactions,
# C_qCpp + s from pp collisions
comptime C_aC = 177
comptime C_AC = 178
comptime C_aC0 = 179
comptime C_AC0 = 180
comptime C_qC = 181
comptime C_qCpp = 189
# Per charm origin (photo-hadronic, then pp): the hadron-grid products of the
# charm decays, in the order of physics/decays.mojo.
comptime C_qpip_C = 197  # + 0 .. 12
comptime C_qpip_Cpp = 210  # + 0 .. 12
comptime N_CPROD_HAD = 13
comptime C_qpip_KC = 223  # + 0 .. 6
comptime C_qpip_KCpp = 230  # + 0 .. 6
comptime C_qpi_Cpg = 237  # + 0 .. 1
comptime C_qpi_Cpp = 239
comptime C_qmu_Cpg = 241  # + 0 .. 3
comptime C_qmu_Cpp = 245
# protons and neutrons of the charm decays of both origins, sources of the
# nucleons in the next step
comptime C_qpro_C = 249
comptime C_qntr_C = 250
comptime C_EXT_HAD_END = 251
comptime C_EXT_NU_BEGIN = 251
comptime C_qnuea_ndec = 251
comptime C_qnue_K = 252
comptime C_qnuea_K = 253
comptime C_qnum_K = 254
comptime C_qnuma_K = 255
comptime C_qnue_Kpp = 256
# the sources of the neutrinos P_neutrino_el_Kpg / _Kpp + 0 .. 3
comptime C_qnu_Kpg = 260
comptime C_qnu_Kpp = 264
# the sources of P_neutrino_mu_pidec_pg / _pp + 0 .. 1
comptime C_qnum_pidec_pg = 268
comptime C_qnum_pidec_pp = 270
comptime C_qnue_C = 272  # + 0 .. 5
comptime C_qnue_Cpp = 278  # + 0 .. 5
comptime C_qnue_KC = 284  # + 0 .. 3
comptime C_qnue_KCpp = 288  # + 0 .. 3
# the sources of P_neutrino_el_Cpg / _Cpp + 0 .. 3
comptime C_qnu_Cpg = 292
comptime C_qnu_Cpp = 296
comptime C_EXT_NU_END = 300

comptime C_COUNT = 300
