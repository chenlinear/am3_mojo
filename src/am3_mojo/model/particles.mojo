"""Indices of the particle density arrays, grouped by the energy grid they live on
(hadron, neutrino, lepton, photon)."""

# had grid
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

# nu grid
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

# lep grid
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

# pho grid
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

comptime P_COUNT = 66
