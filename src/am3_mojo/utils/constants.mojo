"""Physical and numerical constants of AM3 (consts.h)."""

# ============================================================================
# Physics & math constants (consts.h)
# ============================================================================
comptime mathpi: Float64 = 3.141592654
comptime lambdaC_over_8pi: Float64 = 11.8332
comptime L_SSC_CGS: Float64 = 8.156e-13
comptime T_SSC_CGS: Float64 = 2.719e-23
comptime n_SSC_CGS: Float64 = 1.843e36
comptime Rate_SSC_CGS: Float64 = 6.784e58
comptime erg_GeV: Float64 = 624.0
comptime statcol: Float64 = 4.803e-10
comptime sigmaT: Float64 = 0.665e-24
comptime massp: Float64 = 0.938
comptime massn: Float64 = 0.938
comptime masspi0: Float64 = 0.1350
comptime masspip: Float64 = 0.1396
comptime masspim: Float64 = 0.1396
comptime masseta: Float64 = 0.54786
comptime massmu: Float64 = 0.10566
comptime masse: Float64 = 0.511e-3
comptime tdec_pipm: Float64 = 2.6033e-8
comptime c0: Float64 = 3.0e10
comptime mubarn: Float64 = 1.0e-30

# ============================================================================
# Constants of the processes am3_mojo adds to AM3
# ============================================================================
# neutron decay (PDG 2026, with its CODATA 2022 masses): neutron and proton
# masses [GeV], neutron mean life [s] and the fine-structure constant (for
# the Coulomb correction of the beta spectrum). The proton mass also enters
# s of the pp kaons, and alpha the unit alpha r_e^2 of the triplet cross
# section. AM3's rounded nucleon masses massp, massn stay for its processes.
comptime massN_PDG: Float64 = 0.93956542194
comptime massP_PDG: Float64 = 0.93827208943
comptime tdec_n: Float64 = 878.3
comptime alpha_PDG: Float64 = 1.0 / 137.035999177
# kaons (PDG 2026 fit values): masses [GeV] and mean lives [s] of K+- and
# K0_S, K0_L.
comptime massK: Float64 = 0.493677
comptime tdec_K: Float64 = 1.2380e-8
comptime massK0: Float64 = 0.497611
comptime tdec_KS: Float64 = 8.954e-11
comptime tdec_KL: Float64 = 5.116e-8
# PDG 2026 masses [GeV] of the kaon decay products (the electron's also for
# neutron decay), for the decay kinematics (physics/decays.mojo; AM3's
# rounded masses above stay for its processes)
comptime massPi_PDG: Float64 = 0.13957039
comptime massPi0_PDG: Float64 = 0.1349768
comptime massMu_PDG: Float64 = 0.1056583755
comptime massE_PDG: Float64 = 0.51099895069e-3
