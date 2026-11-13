"""The values of PDG 2026 (its Table 1.1, from CODATA 2022, and its particle
listings), in cgs units with masses in GeV.

PDG 2026 is the Review of Particle Physics of the Particle Data Group, F.
Takahashi et al., Int. J. Mod. Phys. A 41, 2630011 (2026). The comment above
each constant cites its entry. "Table 1.1" is the table of Section 1,
"Physical Constants" (p. 1). "<particle> listing, p. N" is the printed page N
of that particle's Particle Listing (pdg.lbl.gov/2026/listings). Such a
citation also gives the data block, the value as printed and its label
(OUR FIT, OUR AVERAGE, OUR EVALUATION, OUR ESTIMATE) or the measurement that
the listing gives. A value derived here names the constants it is derived
from.

The mathematical constants come first. The physical quantities follow the order
in which the pages of the added processes introduce them
(tests/literature_checks/<process>/README.md, in the order of README.md). A
quantity that no page names follows the quantities it is derived from or used
with. In the section Hadrons (muon, pions, kaons) and in the section
Charm hadrons (tau, D, D_s, Lambda_c), the masses and mean lives go from the
lepton up in mass. The other products of the charm decays follow the list of
the charm page (its section Decays). Within a family, charged particles come
before neutral ones, and each mass comes before its mean life."""

from std.math import sqrt

# ==============================================================================
# Mathematical constants
# ==============================================================================
# pi = 3.141 592 653 589 793 238...: PDG 2026, Table 1.1
comptime PI: Float64 = 3.141592653589793
# ln 10, the Float64 nearest to it (not in PDG 2026)
comptime LN10: Float64 = 2.302585092994046

# ==============================================================================
# Electron-positron annihilation
# ==============================================================================
# Thomson cross section [cm^2]: PDG 2026, Table 1.1, sigma_T = 8 pi r_e^2/3 =
# 0.665 245 870 51(62) barn, with 1 barn = 1e-28 m^2 (Table 1.1)
comptime SIGMA_T: Float64 = 0.66524587051e-24
# speed of light [cm/s]: PDG 2026, Table 1.1, c = 299 792 458 m/s (exact)
comptime C_LIGHT: Float64 = 2.99792458e10
# electron mass [GeV]: PDG 2026, Table 1.1, m_e = 0.510 998 950 69(16) MeV;
# e listing, p. 1, e MASS: 0.51099895069 +- 0.00000000016 MeV (MOHR 25, the
# 2022 CODATA value)
comptime MASS_E: Float64 = 0.51099895069e-3
# The code units of the leptons and photons (AM3's SSC units, sigma_T c = 1):
# length sqrt(sigma_T) [cm], time sqrt(sigma_T) / c [s], density [cm^-3] and
# rate [cm^-3 s^-1]. Their energies are in m_e c^2.
comptime L_SSC: Float64 = sqrt(SIGMA_T)
comptime T_SSC: Float64 = L_SSC / C_LIGHT
comptime N_SSC: Float64 = 1.0 / (L_SSC * L_SSC * L_SSC)
comptime RATE_SSC: Float64 = N_SSC / T_SSC
# GeV per erg: PDG 2026, Table 1.1, 1 eV = 1.602 176 634 x 10^-19 J (exact)
# and 1 erg = 1e-7 J
comptime GEV_PER_ERG: Float64 = 1.0 / 1.602176634e-3
comptime MEC2_EV: Float64 = MASS_E * 1.0e9
comptime MEC2_ERG: Float64 = MASS_E / GEV_PER_ERG
# Planck constant [GeV s]: PDG 2026, Table 1.1, h = 6.626 070 15 x 10^-34 J s
# (exact), with 1 GeV = 1.602 176 634 x 10^-10 J (1 eV, Table 1.1)
comptime H_PLANCK: Float64 = 6.62607015e-34 / 1.602176634e-10
# elementary charge [esu]: PDG 2026, Table 1.1, e = 1.602 176 634 x 10^-19 C
# (exact) and 1 C = 299 792 458 x 10 x r_mu0 esu, with (its footnote)
# r_mu0 = sqrt(mu_0 / (4 pi x 10^-7 N A^-2)) = 0.999 999 999 93(8)
comptime E_CHARGE: Float64 = 1.602176634e-19 * 2.99792458e9 * 0.99999999993
# critical magnetic field B_cr = m_e^2 c^3 / (e hbar) [G], with hbar = h/2 pi
# (PDG 2026, Table 1.1), for AM3's synchrotron and acceleration
comptime B_CRIT: Float64 = MEC2_ERG * MEC2_ERG / (
    E_CHARGE * (H_PLANCK / GEV_PER_ERG / (2.0 * PI)) * C_LIGHT
)
# pi r_e^2 [sigma_T] = 3/8, from sigma_T = 8 pi r_e^2/3 (PDG 2026, Table 1.1):
# the unit of Svensson's annihilation rate and spectrum
comptime PI_RE2: Float64 = 0.375

# ==============================================================================
# Triplet pair production
# ==============================================================================
# Fine-structure constant: PDG 2026, Table 1.1, alpha = e^2/(4 pi epsilon_0
# hbar c) = 1/137.035 999 177(21).
comptime ALPHA: Float64 = 1.0 / 137.035999177
# alpha r_e^2 [sigma_T] = 3 alpha/(8 pi), from sigma_T = 8 pi r_e^2/3: the
# unit of the triplet tables' cross sections
comptime ALPHA_RE2: Float64 = 3.0 * ALPHA / (8.0 * PI)

# ==============================================================================
# Neutron decay
# ==============================================================================
# neutron mean life [s]: PDG 2026, n listing, p. 3, n MEAN LIFE: 878.3 +- 0.4
# s, OUR AVERAGE
comptime TAU_N: Float64 = 878.3
# neutron mass [GeV]: PDG 2026, Table 1.1, m_n = 939.565 421 94(48) MeV;
# n listing, p. 1, n MASS: 939.56542194 +- 0.00000048 MeV (MOHR 25, the 2022
# CODATA value)
comptime MASS_N: Float64 = 0.93956542194
# proton mass [GeV]: PDG 2026, Table 1.1, m_p = 938.272 089 43(29) MeV;
# p listing, p. 1, p MASS: 938.27208943 +- 0.00000029 MeV, OUR EVALUATION
comptime MASS_P: Float64 = 0.93827208943

# ==============================================================================
# Hadrons
# ==============================================================================
# microbarn and millibarn [cm^2]: PDG 2026, Table 1.1, 1 barn = 1e-28 m^2
comptime MICROBARN: Float64 = 1.0e-30
comptime MILLIBARN: Float64 = 1.0e-27
# masses [GeV] and mean lives [s] of the muon, of pi+- and pi0 (the pi0 decays
# at once), and of K+- and K0 (the mean lives of K0_S and K0_L)
# PDG 2026, mu listing, p. 1, mu MASS: 105.6583755 +- 0.0000023 MeV (MOHR 25,
# the 2022 CODATA value)
comptime MASS_MU: Float64 = 0.1056583755
# PDG 2026, mu listing, p. 2, mu MEAN LIFE: (2.1969811 +- 0.0000022) x 10^-6
# s, OUR AVERAGE
comptime TAU_MU: Float64 = 2.1969811e-6
# PDG 2026, pi+- listing, p. 1, pi+- MASS: 139.57039 +- 0.00018 MeV, OUR FIT
comptime MASS_PI: Float64 = 0.13957039
# PDG 2026, pi+- listing, p. 3, pi+- MEAN LIFE: (2.6033 +- 0.0005) x 10^-8 s,
# OUR AVERAGE
comptime TAU_PI: Float64 = 2.6033e-8
# PDG 2026, pi0 listing, p. 1, pi0 MASS: 134.9768 +- 0.0005 MeV, OUR FIT
comptime MASS_PI0: Float64 = 0.1349768
# PDG 2026, K+- listing, p. 6, K+- MASS: 493.677 +- 0.015 MeV, OUR FIT
comptime MASS_K: Float64 = 0.493677
# PDG 2026, K+- listing, p. 7, K+- MEAN LIFE: (1.2380 +- 0.0020) x 10^-8 s,
# OUR FIT
comptime TAU_K: Float64 = 1.2380e-8
# PDG 2026, K0 listing, p. 1, K0 MASS: 497.611 +- 0.013 MeV, OUR FIT
comptime MASS_K0: Float64 = 0.497611
# PDG 2026, K0_S listing, p. 1, K0_S MEAN LIFE: (0.8954 +- 0.0004) x 10^-10 s,
# OUR FIT assuming CPT
comptime TAU_KS: Float64 = 8.954e-11
# PDG 2026, K0_L listing, p. 2, K0_L MEAN LIFE: (5.116 +- 0.021) x 10^-8 s,
# OUR FIT
comptime TAU_KL: Float64 = 5.116e-8

# ==============================================================================
# Charm hadrons
# ==============================================================================
# the mass [GeV] of the tau (D(s) -> tau nu), and the masses [GeV] and mean
# lives [s] of D+- and D0, of Ds+- and of Lambda_c+
# PDG 2026, tau listing, p. 1, tau MASS: 1776.93 +- 0.09 MeV, OUR AVERAGE
comptime MASS_TAU: Float64 = 1.77693
# PDG 2026, D+- listing, p. 1, D+- MASS: 1869.66 +- 0.05 MeV, OUR FIT
comptime MASS_D: Float64 = 1.86966
# PDG 2026, D+- listing, p. 1, D+- MEAN LIFE: (1033 +- 5) x 10^-15 s, OUR
# AVERAGE
comptime TAU_D: Float64 = 1.033e-12
# PDG 2026, D0 listing, p. 1, D0 MASS: 1864.84 +- 0.04 MeV, OUR FIT
comptime MASS_D0: Float64 = 1.86484
# PDG 2026, D0 listing, p. 1, D0 MEAN LIFE: (410.3 +- 1.0) x 10^-15 s, OUR
# AVERAGE
comptime TAU_D0: Float64 = 4.103e-13
# PDG 2026, Ds+- listing, p. 1, Ds+- MASS: 1968.35 +- 0.07 MeV, OUR FIT
comptime MASS_DS: Float64 = 1.96835
# PDG 2026, Ds+- listing, p. 2, Ds+- MEAN LIFE: (501.2 +- 2.2) x 10^-15 s, OUR
# AVERAGE
comptime TAU_DS: Float64 = 5.012e-13
# PDG 2026, Lambda_c+ listing, p. 1, Lambda_c+ MASS: 2286.46 +- 0.14 MeV, OUR
# FIT
comptime MASS_LC: Float64 = 2.28646
# PDG 2026, Lambda_c+ listing, p. 2, Lambda_c+ MEAN LIFE: (202.6 +- 1.0) x
# 10^-15 s, OUR AVERAGE
comptime TAU_LC: Float64 = 2.026e-13
# Fractions f(c -> D+, D0, Ds+, Lambda_c+) of charm quarks (and, alike, of
# anti-quarks) that hadronize into each species, for the pgamma and the pp
# charm hadrons. They are the global average of e+e-, ep and pp data
# (Lisovyi et al. 2016, Table 12, "constrained S"), renormalized to one
# hadron per quark. The charm-strange baryons (0.9 %) are left out.
comptime _FRAG_SUM: Float64 = 0.2404 + 0.6086 + 0.0802 + 0.0623
comptime FRAG_D: Float64 = 0.2404 / _FRAG_SUM
comptime FRAG_D0: Float64 = 0.6086 / _FRAG_SUM
comptime FRAG_DS: Float64 = 0.0802 / _FRAG_SUM
comptime FRAG_LC: Float64 = 0.0623 / _FRAG_SUM
# Masses [GeV] of the other short-lived products of the charm decays, which
# decay at once in the code (physics/decays.mojo). They are the listings'
# averages (rho+- and rho0, K*+- hadroproduced and K*0 apart), and the
# estimates of f0(980) and a1(1260).
# PDG 2026, eta listing, p. 1, eta MASS: 547.862 +- 0.017 MeV, OUR AVERAGE
comptime MASS_ETA: Float64 = 0.547862
# PDG 2026, eta'(958) listing, p. 1, eta'(958) MASS: 957.78 +- 0.06 MeV, OUR
# AVERAGE
comptime MASS_ETAP: Float64 = 0.95778
# PDG 2026, omega(782) listing, p. 1, omega(782) MASS: 782.66 +- 0.13 MeV, OUR
# AVERAGE
comptime MASS_OMEGA: Float64 = 0.78266
# PDG 2026, phi(1020) listing, p. 1, phi(1020) MASS: 1019.460 +- 0.016 MeV,
# OUR AVERAGE
comptime MASS_PHI: Float64 = 1.01946
# PDG 2026, rho(770) listing, p. 2, rho(770) MASS, CHARGED ONLY, tau DECAYS
# and e+ e-: 775.11 +- 0.34 MeV, OUR AVERAGE
comptime MASS_RHO: Float64 = 0.77511
# PDG 2026, rho(770) listing, p. 1, rho(770) MASS, NEUTRAL ONLY, e+ e-:
# 775.26 +- 0.23 MeV, OUR AVERAGE
comptime MASS_RHO0: Float64 = 0.77526
# PDG 2026, K*(892) listing, p. 1, K*(892) MASS, CHARGED ONLY, HADROPRODUCED:
# 891.88 +- 0.23 MeV, OUR AVERAGE
comptime MASS_KSTAR: Float64 = 0.89188
# PDG 2026, K*(892) listing, p. 2, K*(892) MASS, NEUTRAL ONLY: 895.56 +- 0.20
# MeV, OUR AVERAGE
comptime MASS_KSTAR0: Float64 = 0.89556
# PDG 2026, f0(980) listing, p. 1, f0(980) MASS: 990 +- 20 MeV, OUR ESTIMATE
comptime MASS_F0: Float64 = 0.990
# PDG 2026, a1(1260) listing, p. 1, a1(1260) MASS: 1230 +- 40 MeV, OUR
# ESTIMATE
comptime MASS_A1: Float64 = 1.230
# PDG 2026, Lambda listing, p. 1, Lambda MASS: 1115.683 +- 0.006 MeV, OUR FIT
comptime MASS_LAMBDA: Float64 = 1.115683
# PDG 2026, Sigma+ listing, p. 1, Sigma+ MASS: 1189.37 +- 0.07 MeV, OUR FIT
comptime MASS_SIGMA_PLUS: Float64 = 1.18937
# PDG 2026, Sigma- listing, p. 1, Sigma- MASS: 1197.449 +- 0.029 MeV, OUR FIT
comptime MASS_SIGMA_MINUS: Float64 = 1.197449
# PDG 2026, Sigma0 listing, p. 1, Sigma0 MASS: 1192.642 +- 0.024 MeV, OUR FIT
comptime MASS_SIGMA0: Float64 = 1.192642
# PDG 2026, Xi- listing, p. 1, Xi- MASS: 1321.71 +- 0.07 MeV, OUR FIT
comptime MASS_XI_MINUS: Float64 = 1.32171
# PDG 2026, Xi0 listing, p. 1, Xi0 MASS: 1314.86 +- 0.20 MeV, OUR FIT
comptime MASS_XI0: Float64 = 1.31486
# masses [GeV] of the form-factor poles of the semileptonic charm decays, the
# lightest charm mesons with the quantum numbers of the current: vector
# D_s*+ (c -> s) and D*(2010)+ (c -> d), scalar D_s0*(2317)+ and D_0*(2300),
# axial-vector D_s1(2460)+ and D_1(2420)
# PDG 2026, D_s*+- listing, p. 1, D_s*+- MASS: 2112.2 +- 0.4 MeV, OUR FIT
comptime MASS_DS_STAR: Float64 = 2.1122
# PDG 2026, D*(2010)+- listing, p. 1, D*(2010)+- MASS: 2010.27 +- 0.04 MeV,
# OUR FIT
comptime MASS_D_STAR: Float64 = 2.01027
# PDG 2026, D_s0*(2317)+- listing, p. 1, D_s0*(2317)+- MASS: 2317.8 +- 0.5
# MeV, OUR FIT
comptime MASS_DS0_STAR: Float64 = 2.3178
# PDG 2026, D_0*(2300) listing, p. 1, D_0*(2300) MASS: 2343 +- 10 MeV, OUR
# AVERAGE
comptime MASS_D0_STAR: Float64 = 2.343
# PDG 2026, D_s1(2460)+- listing, p. 1, D_s1(2460)+- MASS: 2459.5 +- 0.6 MeV,
# OUR FIT
comptime MASS_DS1: Float64 = 2.4595
# PDG 2026, D_1(2420) listing, p. 1, D_1(2420) MASS: 2422.1 +- 0.6 MeV, OUR
# FIT
comptime MASS_D1: Float64 = 2.4221
