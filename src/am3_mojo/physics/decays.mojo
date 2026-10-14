"""Ports of PionDecay (PionDecay.cc), MuonDecay (MuonDecay.cc) and HadIC
(HadronicInverseCompton.cc), and the neutron and kaon decays (am3_mojo
additions)."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.linalg import MatD, gemv_d
from am3_mojo.utils.specfun import gauss_legendre, M_PI
from am3_mojo.utils.parallel import parallel_run


def FMu1(x: Float64) -> Float64:
    var rpi = massmu * massmu / masspip / masspip
    if x >= 1.0 or x < rpi:
        return 0.0
    return rpi * (1.0 - x) / (1.0 - rpi) / (1.0 - rpi) / x


def FMu2(x: Float64) -> Float64:
    var rpi = massmu * massmu / masspip / masspip
    if x >= 1.0 or x < rpi:
        return 0.0
    return (x - rpi) / (1.0 - rpi) / (1.0 - rpi) / x


def FPi2NU(x: Float64) -> Float64:
    var rpi = massmu * massmu / masspip / masspip
    if x < 0.0 or x > 1.0 - rpi:
        return 0.0
    return 1.0 / (1.0 - rpi)


def FPi2G(x: Float64) -> Float64:
    if x <= 0.0 or x > 1.00001:
        return 0.0
    return 2.0


struct PionDecay(Copyable, Movable):
    var verbose: Bool
    var charged_pion_decay_s: Float64
    var neutral_pion_decay_s: Float64
    var MpLGen: Vec
    var MpRGen: Vec
    var MmLGen: Vec
    var MmRGen: Vec
    var NUmGen_pi: Vec
    var NUmAGen_pi: Vec
    var QPI2GGen: Vec
    var KER_FMU1: MatD
    var KER_FMU2: MatD
    var KER_FPi2NU: MatD
    var KER_FPi2G: MatD
    var w_p: Vec
    var w_m: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.charged_pion_decay_s = 2.6033e-8
        self.neutral_pion_decay_s = 8.52e-17
        self.MpLGen = Vec()
        self.MpRGen = Vec()
        self.MmLGen = Vec()
        self.MmRGen = Vec()
        self.NUmGen_pi = Vec()
        self.NUmAGen_pi = Vec()
        self.QPI2GGen = Vec()
        self.KER_FMU1 = MatD()
        self.KER_FMU2 = MatD()
        self.KER_FPi2NU = MatD()
        self.KER_FPi2G = MatD()
        self.w_p = Vec()
        self.w_m = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.MpLGen, dat.nbins_had)
        resize_zero(self.MpRGen, dat.nbins_had)
        resize_zero(self.MmLGen, dat.nbins_had)
        resize_zero(self.MmRGen, dat.nbins_had)
        # (on the neutrino grid they are read on, also when pion decay is off)
        resize_zero(self.NUmGen_pi, dat.nbins_nu)
        resize_zero(self.NUmAGen_pi, dat.nbins_nu)
        resize_zero(self.QPI2GGen, dat.nbins_pho)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        self.KER_FMU1 = MatD(dat.nbins_had, dat.nbins_had)
        self.KER_FMU2 = MatD(dat.nbins_had, dat.nbins_had)
        self.KER_FPi2NU = MatD(dat.nbins_nu, dat.nbins_had)
        self.KER_FPi2G = MatD(dat.nbins_pho, dat.nbins_had)

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing charged pion decay to muon + neutrino kernels")
        var nh = dat.nbins_had
        for i in range(nh):
            for j in range(nh):
                var r = dat.egrid_had[i] / dat.egrid_had[j]
                self.KER_FMU1.set(i, j, r * FMu1(r))
                self.KER_FMU2.set(i, j, r * FMu2(r))
        for i in range(dat.nbins_nu):
            for j in range(nh):
                var r = dat.egrid_nu[i] / dat.egrid_had[j]
                self.KER_FPi2NU.set(i, j, r * FPi2NU(r))
        if self.verbose:
            print("computing pi0 decay to gamma kernel")
        for i in range(dat.nbins_pho):
            for j in range(nh):
                # egrid_had[j+1] reads past the end of the C++ vector for the
                # last bin; the next grid point exp((j+1) dlnE) is used instead.
                var e_next = dat.egrid_had[j + 1] if j + 1 < nh else c_exp(
                    Float64(j + 1) * dat.dlnE
                )
                var r = (
                    dat.egrid_had_photons[i] * 2.0 / (dat.egrid_had[j] + e_next)
                )
                self.KER_FPi2G.set(i, j, r * FPi2G(r))

    def _rates(mut self, dat: AM3Arrays, Pip: Vec, Pim: Vec):
        var nh = dat.nbins_had
        resize_zero(self.w_p, nh)
        resize_zero(self.w_m, nh)
        for i in range(nh):
            var tdec = dat.egrid_had[i] / masspip * self.charged_pion_decay_s
            self.w_p[i] = Pip[i] / tdec
            self.w_m[i] = Pim[i] / tdec

    def Calc_PItoMU(mut self, dat: AM3Arrays, Pip: Vec, Pim: Vec):
        if self.verbose:
            print("computing pion decay to muon term")
        self._rates(dat, Pip, Pim)
        gemv_d(self.KER_FMU1, pd(self.w_p), dat.dlnE, pd(self.MpRGen))
        gemv_d(self.KER_FMU1, pd(self.w_m), dat.dlnE, pd(self.MmLGen))
        gemv_d(self.KER_FMU2, pd(self.w_p), dat.dlnE, pd(self.MpLGen))
        gemv_d(self.KER_FMU2, pd(self.w_m), dat.dlnE, pd(self.MmRGen))

    def Calc_PItoNU(mut self, dat: AM3Arrays, Pip: Vec, Pim: Vec):
        if self.verbose:
            print("computing pion decay to neutrino term")
        self._rates(dat, Pip, Pim)
        resize_zero(self.NUmGen_pi, self.KER_FPi2NU.rows)
        resize_zero(self.NUmAGen_pi, self.KER_FPi2NU.rows)
        gemv_d(self.KER_FPi2NU, pd(self.w_p), dat.dlnE, pd(self.NUmGen_pi))
        gemv_d(self.KER_FPi2NU, pd(self.w_m), dat.dlnE, pd(self.NUmAGen_pi))

    def Calc_PItoG(mut self, dat: AM3Arrays, QPI0: Vec):
        if self.verbose:
            print("computing pion_0 decay to gamma term")
        resize_zero(self.QPI2GGen, self.KER_FPi2G.rows)
        gemv_d(self.KER_FPi2G, pd(QPI0), dat.dlnE, pd(self.QPI2GGen))


def FNU1(x: Float64, h: Float64) -> Float64:
    if x > 1.0 or x < 0.0:
        return 0.0
    return (5.0 / 3.0 - 3.0 * x * x + 4.0 * x * x * x / 3.0) + h * (
        -1.0 / 3.0 + 3.0 * x * x - 8.0 * x * x * x / 3.0
    )


def FNU2(x: Float64, h: Float64) -> Float64:
    if x > 1.0 or x < 0.0:
        return 0.0
    return (2.0 - 6.0 * x * x + 4.0 * x * x * x) + h * (
        2.0 - 12.0 * x + 18.0 * x * x - 8.0 * x * x * x
    )


def FMU2E(x: Float64) -> Float64:
    if x <= 0.0 or x > 1.00001:
        return 0.0
    return 1.3333 * (1.0 - x * x * x)


struct MuonDecay(Copyable, Movable):
    var verbose: Bool
    var mu_decay_s: Float64
    var NUeGen_mu: Vec
    var NUmGen_mu: Vec
    var NUeAGen_mu: Vec
    var NUmAGen_mu: Vec
    var QMU2EGen_mu: Vec
    var QMU2PGen_mu: Vec
    var KER_FMU2E: MatD
    var KER_FNU1p: MatD
    var KER_FNU1m: MatD
    var KER_FNU2p: MatD
    var KER_FNU2m: MatD
    var w_ml: Vec
    var w_pl: Vec
    var w_mr: Vec
    var w_pr: Vec
    var t1: Vec
    var t2: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.mu_decay_s = 2.1969811e-6
        self.NUeGen_mu = Vec()
        self.NUmGen_mu = Vec()
        self.NUeAGen_mu = Vec()
        self.NUmAGen_mu = Vec()
        self.QMU2EGen_mu = Vec()
        self.QMU2PGen_mu = Vec()
        self.KER_FMU2E = MatD()
        self.KER_FNU1p = MatD()
        self.KER_FNU1m = MatD()
        self.KER_FNU2p = MatD()
        self.KER_FNU2m = MatD()
        self.w_ml = Vec()
        self.w_pl = Vec()
        self.w_mr = Vec()
        self.w_pr = Vec()
        self.t1 = Vec()
        self.t2 = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.NUeGen_mu, dat.nbins_nu)
        resize_zero(self.NUeAGen_mu, dat.nbins_nu)
        resize_zero(self.NUmGen_mu, dat.nbins_nu)
        resize_zero(self.NUmAGen_mu, dat.nbins_nu)
        resize_zero(self.QMU2EGen_mu, dat.nbins_lep)
        resize_zero(self.QMU2PGen_mu, dat.nbins_lep)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        self.KER_FMU2E = MatD(dat.nbins_lep, dat.nbins_had)
        self.KER_FNU1p = MatD(dat.nbins_nu, dat.nbins_had)
        self.KER_FNU1m = MatD(dat.nbins_nu, dat.nbins_had)
        self.KER_FNU2p = MatD(dat.nbins_nu, dat.nbins_had)
        self.KER_FNU2m = MatD(dat.nbins_nu, dat.nbins_had)

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing muon decay to neutrino kernels")
        for i in range(dat.nbins_nu):
            for j in range(dat.nbins_had):
                var r = dat.egrid_nu[i] / dat.egrid_had[j]
                self.KER_FNU1p.set(i, j, r * FNU1(r, 1.0))
                self.KER_FNU1m.set(i, j, r * FNU1(r, -1.0))
                self.KER_FNU2p.set(i, j, r * FNU2(r, 1.0))
                self.KER_FNU2m.set(i, j, r * FNU2(r, -1.0))
        if self.verbose:
            print("computing muon decay to e+/- kernels")
        for i in range(dat.nbins_lep):
            for j in range(dat.nbins_had):
                var r = dat.egrid_had_leptons[i] / dat.egrid_had[j]
                self.KER_FMU2E.set(i, j, r * FMU2E(r))

    def Calc_MUtoNU(
        mut self, dat: AM3Arrays, MmL: Vec, MpL: Vec, MmR: Vec, MpR: Vec
    ):
        if self.verbose:
            print("computing muon decay to neutrinos terms")
        var nh = dat.nbins_had
        resize_zero(self.w_ml, nh)
        resize_zero(self.w_pl, nh)
        resize_zero(self.w_mr, nh)
        resize_zero(self.w_pr, nh)
        for i in range(nh):
            var tdec = dat.egrid_had[i] / massmu * self.mu_decay_s
            self.w_ml[i] = MmL[i] / tdec
            self.w_pl[i] = MpL[i] / tdec
            self.w_mr[i] = MmR[i] / tdec
            self.w_pr[i] = MpR[i] / tdec
        var nr = self.KER_FNU1p.rows
        resize_zero(self.t1, nr)
        resize_zero(self.t2, nr)
        resize_zero(self.NUeAGen_mu, nr)
        resize_zero(self.NUeGen_mu, nr)
        resize_zero(self.NUmAGen_mu, nr)
        resize_zero(self.NUmGen_mu, nr)
        # dlnE * (A x + B y): both products into temporaries, then combined
        gemv_d(self.KER_FNU2p, pd(self.w_ml), 1.0, pd(self.t1))
        gemv_d(self.KER_FNU2m, pd(self.w_mr), 1.0, pd(self.t2))
        for i in range(nr):
            self.NUeAGen_mu[i] = dat.dlnE * (self.t1[i] + self.t2[i])
        gemv_d(self.KER_FNU2m, pd(self.w_pl), 1.0, pd(self.t1))
        gemv_d(self.KER_FNU2p, pd(self.w_pr), 1.0, pd(self.t2))
        for i in range(nr):
            self.NUeGen_mu[i] = dat.dlnE * (self.t1[i] + self.t2[i])
        gemv_d(self.KER_FNU1m, pd(self.w_pl), 1.0, pd(self.t1))
        gemv_d(self.KER_FNU1p, pd(self.w_pr), 1.0, pd(self.t2))
        for i in range(nr):
            self.NUmAGen_mu[i] = dat.dlnE * (self.t1[i] + self.t2[i])
        gemv_d(self.KER_FNU1p, pd(self.w_ml), 1.0, pd(self.t1))
        gemv_d(self.KER_FNU1m, pd(self.w_mr), 1.0, pd(self.t2))
        for i in range(nr):
            self.NUmGen_mu[i] = dat.dlnE * (self.t1[i] + self.t2[i])

    def Calc_MUtoE(
        mut self, dat: AM3Arrays, MmL: Vec, MpL: Vec, MmR: Vec, MpR: Vec
    ):
        if self.verbose:
            print("computing muon decay to pairs terms")
        var nh = dat.nbins_had
        resize_zero(self.w_pl, nh)
        resize_zero(self.w_ml, nh)
        for i in range(nh):
            var tdec = dat.egrid_had[i] / massmu * self.mu_decay_s
            self.w_pl[i] = (MpL[i] + MpR[i]) / tdec
            self.w_ml[i] = (MmL[i] + MmR[i]) / tdec
        resize_zero(self.QMU2PGen_mu, self.KER_FMU2E.rows)
        resize_zero(self.QMU2EGen_mu, self.KER_FMU2E.rows)
        gemv_d(self.KER_FMU2E, pd(self.w_pl), dat.dlnE, pd(self.QMU2PGen_mu))
        gemv_d(self.KER_FMU2E, pd(self.w_ml), dat.dlnE, pd(self.QMU2EGen_mu))


struct HadIC(Copyable, Movable):
    var verbose: Bool
    var epsilonPIC: Vec
    var epsilonPiIC: Vec
    var epsilonMuIC: Vec
    var epsilonKIC: Vec  # charged kaons (am3_mojo addition)
    var deltaE_p: VecI
    var jmax_p: VecI
    var deltaE_pi: VecI
    var jmax_pi: VecI
    var deltaE_mu: VecI
    var jmax_mu: VecI
    var deltaE_K: VecI
    var jmax_K: VecI
    var w_in: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.epsilonPIC = Vec()
        self.epsilonPiIC = Vec()
        self.epsilonMuIC = Vec()
        self.epsilonKIC = Vec()
        self.deltaE_p = VecI()
        self.jmax_p = VecI()
        self.deltaE_pi = VecI()
        self.jmax_pi = VecI()
        self.deltaE_mu = VecI()
        self.jmax_mu = VecI()
        self.deltaE_K = VecI()
        self.jmax_K = VecI()
        self.w_in = Vec()
        self.clear_internal_arrays(dat)
        self.reinitialize_empty_kernels(dat, True, True, True, True)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.epsilonPIC, dat.nbins_pho)
        resize_zero(self.epsilonPiIC, dat.nbins_pho)
        resize_zero(self.epsilonMuIC, dat.nbins_pho)
        resize_zero(self.epsilonKIC, dat.nbins_pho)

    def reinitialize_empty_kernels(
        mut self,
        dat: AM3Arrays,
        rad_P: Bool,
        rad_Pi: Bool,
        rad_Mu: Bool,
        rad_K: Bool,
    ):
        if rad_P:
            self.deltaE_p = zerosi(dat.nbins_pho)
            self.jmax_p = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)
        if rad_Pi:
            self.deltaE_pi = zerosi(dat.nbins_pho)
            self.jmax_pi = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)
        if rad_Mu:
            self.deltaE_mu = zerosi(dat.nbins_pho)
            self.jmax_mu = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)
        if rad_K:
            self.deltaE_K = zerosi(dat.nbins_pho)
            self.jmax_K = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)

    @staticmethod
    def _steps(dat: AM3Arrays, mass: Float64, mut dE: VecI, mut jmax: VecI):
        for i in range(dat.nbins_had):
            var g = c_exp(Float64(i) * dat.dlnE) / mass
            var d = to_int(c_log(g * g) / dat.dlnE)
            dE[i] = d
            var jmax1 = (
                to_int(c_log(0.5 * mass / g) / dat.dlnE)
                + dat.ind_min_energy_pho_hadgrid
            )
            jmax[i] = dat.nbins_pho if (jmax1 + d) > dat.nbins_pho else (
                jmax1 + d
            )

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        rad_P: Bool,
        rad_Pi: Bool,
        rad_Mu: Bool,
        rad_K: Bool,
    ):
        if rad_P:
            if self.verbose:
                print("computing proton IC rad. kernel (step approx.)")
            HadIC._steps(dat, massp, self.deltaE_p, self.jmax_p)
        if rad_Pi:
            if self.verbose:
                print("computing pion IC rad. kernel (step approx.)")
            HadIC._steps(dat, masspim, self.deltaE_pi, self.jmax_pi)
        if rad_Mu:
            if self.verbose:
                print("computing muon IC rad. kernel (step approx.)")
            HadIC._steps(dat, massmu, self.deltaE_mu, self.jmax_mu)
        if rad_K:
            if self.verbose:
                print("computing kaon IC rad. kernel (step approx.)")
            HadIC._steps(dat, massK, self.deltaE_K, self.jmax_K)

    @staticmethod
    def _emit(
        dat: AM3Arrays,
        dE: VecI,
        jmax: VecI,
        part: Vec,
        coeff: Float64,
        mut out_: Vec,
    ):
        """out[j] = dlnE coeff sum_i ph[j-dE[i]] part[i], dE[i] <= j <= jmax[i].

        The step approximation puts the photons of hadron bin i at the fixed
        shift dE[i] from their target photon bin. Each hadron bin that is not
        empty adds them over its contiguous photon range (an axpy). The C++
        code loops over every pair (j, i) and tests whether bin i reaches j."""
        var ph = pd(dat.n[P_photons_on_had_grid])
        var pp = pd(part)
        var npho = dat.nbins_pho
        var nh = dat.nbins_had
        var o = pd(out_)
        for j in range(npho):
            o[unsafe_offset=j] = 0.0
        for i in range(nh):
            var w = pp[unsafe_offset=i]
            if w == 0.0:
                continue
            var d = dE[i]
            var lo = max(d, 0)
            var hi = min(min(jmax[i] + 1, npho), npho + d)
            var src = ph.unsafe_offset(lo - d)
            var dst = o.unsafe_offset(lo)
            for m in range(hi - lo):
                dst[unsafe_offset=m] += src[unsafe_offset=m] * w
        var f = dat.dlnE * coeff
        for j in range(npho):
            o[unsafe_offset=j] *= f

    def Calc_PIC(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing proton IC emission")
        var coeff_p = c0 * sigmaT / (massp / masse) / (massp / masse)
        resize_zero(self.epsilonPIC, dat.nbins_pho)
        HadIC._emit(
            dat,
            self.deltaE_p,
            self.jmax_p,
            dat.n[P_protons],
            coeff_p,
            self.epsilonPIC,
        )

    def Calc_PiIC(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing pion IC emission")
        var coeff_pi = c0 * sigmaT / (masspim / masse) / (masspim / masse)
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = dat.n[P_pion_plus][i] + dat.n[P_pion_minus][i]
        resize_zero(self.epsilonPiIC, dat.nbins_pho)
        HadIC._emit(
            dat,
            self.deltaE_pi,
            self.jmax_pi,
            self.w_in,
            coeff_pi,
            self.epsilonPiIC,
        )

    def Calc_MuIC(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing muon IC emission")
        var coeff_mu = c0 * sigmaT / (massmu / masse) / (massmu / masse)
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = (
                dat.n[P_muon_minus_left][i]
                + dat.n[P_muon_minus_right][i]
                + dat.n[P_muon_plus_left][i]
                + dat.n[P_muon_plus_right][i]
            )
        resize_zero(self.epsilonMuIC, dat.nbins_pho)
        HadIC._emit(
            dat,
            self.deltaE_mu,
            self.jmax_mu,
            self.w_in,
            coeff_mu,
            self.epsilonMuIC,
        )

    def Calc_KIC(mut self, dat: AM3Arrays):
        """IC emission of the charged kaons (am3_mojo addition), as for pions.
        """
        if self.verbose:
            print("computing kaon IC emission")
        var coeff_K = c0 * sigmaT / (massK / masse) / (massK / masse)
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = dat.n[P_kaon_plus][i] + dat.n[P_kaon_minus][i]
        resize_zero(self.epsilonKIC, dat.nbins_pho)
        HadIC._emit(
            dat,
            self.deltaE_K,
            self.jmax_K,
            self.w_in,
            coeff_K,
            self.epsilonKIC,
        )


# ============================================================================
# Neutron decay (am3_mojo addition)
# ============================================================================
# n -> p e- anti-nu_e (100 %, PDG 2026) at the neutron solver's rate
# 1 / (gamma tau_n) (README, "Neutron decay"). In the rest frame, the electron
# and antineutrino share E_0 by the allowed spectrum (neutron_beta_density),
# and the proton takes m_n - E_0. Each product is boosted exactly to every
# hadron bin's energy, also for slow neutrons (beta < 1), so the kernels are
# not Toeplitz. `NeutronDecay.initialize_kernels` integrates them exactly over
# the grids.
# Limitations (sizes: tests/literature_checks/README.md, Results): no recoil
# in the lepton spectra (the proton's is at most 751 eV), no weak magnetism,
# no radiative corrections (Sirlin's outer correction) or radiative-decay
# photons, a point-charge Fermi function (no relativistic or finite-size
# terms) and no bound decay into hydrogen.
# AM3's solver keeps the two lowest bins of every grid empty: neutrons below
# 1.2 GeV are removed there, and some electrons of the slowest decaying
# neutrons fall into the two lowest lepton bins and are lost. As for AM3's
# decays, a step's products come from the neutrons at its end (exact in a
# steady state, first order in the time step otherwise).


def neutron_beta_endpoint() -> Float64:
    """Total energy E_0 [GeV] of the electron at the end of its spectrum in the
    neutron's rest frame: (m_n^2 - m_p^2 + m_e^2) / (2 m_n)."""
    var mn = massN_PDG
    var mp = massP_PDG
    var me = massE_PDG
    return ((mn - mp) * (mn + mp) + me * me) / (2.0 * mn)


def neutron_beta_density(T: Float64, Enu: Float64) -> Float64:
    """Unnormalised electron energy distribution in the neutron's rest frame:
    the allowed spectrum F(E) p E Enu^2, with the electron's kinetic energy T,
    E = m_e + T, momentum p and the antineutrino energy Enu = E_0 - m_e - T
    [GeV]. F = 2 pi eta / (1 - exp(-2 pi eta)), eta = alpha E / p, is the
    Fermi function of a charge Z = 1 (F p is finite at p = 0)."""
    var me = massE_PDG
    var E = me + T
    var p = c_sqrt(T * (T + 2.0 * me))
    var c = 2.0 * M_PI * alpha_PDG * E  # 2 pi eta p
    var fp = c if p <= 0.0 else c / (-c_expm1(-c / p))
    return fp * E * Enu * Enu


def _bin_box(
    a: Float64,
    b: Float64,
    w: Float64,
    x: FPtr,
    n: Int,
    N: FPtr,
    En: FPtr,
):
    """Adds products of weight w, uniform in x on [a, b], to the number N[j]
    and energy En[j] (in units of the parent's energy) between the grid
    points x[j] < x < x[j + 1] (n points, logarithmic, ascending). Products
    outside [x[0], x[n - 1]] are left out. A box narrower than the rounding
    of its ends is a point at its middle."""
    var dl = c_log(x[unsafe_offset=1] / x[unsafe_offset=0])
    if not (b - a > 0.0):
        var m = 0.5 * (a + b)
        if m < x[unsafe_offset=0] or m >= x[unsafe_offset=n - 1]:
            return
        var j = min(n - 2, max(0, to_int(c_floor(c_log(m / x[unsafe_offset=0]) / dl))))
        while j > 0 and x[unsafe_offset=j] > m:
            j -= 1
        while j < n - 2 and x[unsafe_offset=j + 1] <= m:
            j += 1
        N[unsafe_offset=j] += w
        En[unsafe_offset=j] += w * m
        return
    var jlo = 0
    if a > 0.0:
        jlo = max(0, to_int(c_floor(c_log(a / x[unsafe_offset=0]) / dl)) - 1)
    var jhi = min(n - 2, to_int(c_floor(c_log(b / x[unsafe_offset=0]) / dl)) + 1)
    var dens = w / (b - a)
    for j in range(jlo, jhi + 1):
        var x1 = max(a, x[unsafe_offset=j])
        var x2 = min(b, x[unsafe_offset=j + 1])
        if x2 <= x1:
            continue
        var d = dens * (x2 - x1)
        N[unsafe_offset=j] += d
        En[unsafe_offset=j] += d * 0.5 * (x1 + x2)


def _share_to_points(x: FPtr, n: Int, N: FPtr, En: FPtr, out_: FPtr):
    """The products between x[j] and x[j + 1] (number N[j], energy En[j]) go
    to these two grid points linearly in x, which keeps their number and
    energy: out_[j] products per decay at grid point j."""
    for j in range(n):
        out_[unsafe_offset=j] = 0.0
    for j in range(n - 1):
        var nj = N[unsafe_offset=j]
        var ej = En[unsafe_offset=j]
        if nj == 0.0 and ej == 0.0:
            continue
        var x0 = x[unsafe_offset=j]
        var x1 = x[unsafe_offset=j + 1]
        var dx = x1 - x0
        out_[unsafe_offset=j] += (x1 * nj - ej) / dx
        out_[unsafe_offset=j + 1] += (ej - x0 * nj) / dx


struct NDecBuild(Movable):
    """What the workers of `NeutronDecay.initialize_kernels` share: the grids
    [GeV], the nodes of the rest-frame spectrum (electron kinetic energy T and
    momentum pe, antineutrino energy Ev, normalised weight w) and the kernels
    being filled (column-major, one column per hadron bin)."""

    var nh: Int
    var nl: Int
    var nn: Int
    var Eh: Vec
    var El: Vec
    var Enu: Vec
    var T: Vec
    var pe: Vec
    var Ev: Vec
    var w: Vec
    var Ke: Vec
    var Knu: Vec

    def __init__(out self, dat: AM3Arrays):
        self.nh = dat.nbins_had
        self.nl = dat.nbins_lep
        self.nn = dat.nbins_nu
        self.Eh = dat.egrid_had.copy()
        self.El = zeros(self.nl)
        for j in range(self.nl):
            self.El[j] = masse * dat.egrid_lep[j]  # (AM3's lepton grid unit)
        self.Enu = dat.egrid_nu.copy()
        self.T = zeros(N_REST)
        self.pe = zeros(N_REST)
        self.Ev = zeros(N_REST)
        self.w = zeros(N_REST)
        self.Ke = zeros(self.nl * self.nh)
        self.Knu = zeros(self.nn * self.nh)
        # the rest-frame spectrum on the N_REST nodes of the substitution
        # E = m_e + (E_0 - m_e) sin^2(pi u / 2) (as for the kaon decays), with
        # both distances from the ends, T and Ev = E_0 - E, to full precision
        var gl = GLRules()
        var me = massE_PDG
        var L = neutron_beta_endpoint() - me
        var tot = 0.0
        for n in range(N_REST):
            var u = 0.25 * M_PI * (gl.xo[n] + 1.0)  # pi u / 2
            var s = c_sin(u)
            var c = c_cos(u)
            self.T[n] = L * s * s
            self.Ev[n] = L * c * c
            self.pe[n] = c_sqrt(self.T[n] * (self.T[n] + 2.0 * me))
            self.w[n] = (
                0.5
                * M_PI
                * L
                * s
                * c
                * gl.wo[n]
                * neutron_beta_density(self.T[n], self.Ev[n])
            )
            tot += self.w[n]
        for n in range(N_REST):
            self.w[n] /= tot


def _ndec_column(ref J: NDecBuild, i: Int, xl: FPtr, xn: FPtr, N: FPtr, En: FPtr):
    """Column i of both kernels: electrons and antineutrinos per decay of a
    neutron of hadron bin i. A product of rest-frame energy E* and momentum p*
    is emitted isotropically, so its lab energy fraction x = E / E_n is
    uniform on [(E* - beta p*) / m_n, (E* + beta p*) / m_n]."""
    var mn = massN_PDG
    var me = massE_PDG
    var En_n = J.Eh[i]
    var g = En_n / mn
    var beta = c_sqrt((g - 1.0) * (g + 1.0)) / g
    var omb = 1.0 / (g * g * (1.0 + beta))  # 1 - beta
    for j in range(J.nl):
        xl[unsafe_offset=j] = J.El[j] / En_n
    for j in range(J.nl):
        N[unsafe_offset=j] = 0.0
        En[unsafe_offset=j] = 0.0
    for n in range(N_REST):
        var p = J.pe[n]
        var E = me + J.T[n]
        # E - beta p = me^2 / (E + p) + (1 - beta) p, without cancellation
        var lo = (me * me / (E + p) + omb * p) / mn
        var hi = (E + beta * p) / mn
        _bin_box(lo, hi, J.w[n], xl, J.nl, N, En)
    _share_to_points(xl, J.nl, N, En, pd(J.Ke).unsafe_offset(i * J.nl))
    for j in range(J.nn):
        xn[unsafe_offset=j] = J.Enu[j] / En_n
    for j in range(J.nn):
        N[unsafe_offset=j] = 0.0
        En[unsafe_offset=j] = 0.0
    for n in range(N_REST):
        var Ev = J.Ev[n]
        _bin_box(omb * Ev / mn, (1.0 + beta) * Ev / mn, J.w[n], xn, J.nn, N, En)
    _share_to_points(xn, J.nn, N, En, pd(J.Knu).unsafe_offset(i * J.nn))


def ndec_build_work(job: Pointer[NDecBuild, MutUntrackedOrigin], t: Int, nt: Int):
    """Columns t, t + nt, ... of the neutron decay kernels."""
    ref J = job[]
    var m = max(J.nl, J.nn)
    var xl = zeros(J.nl)
    var xn = zeros(J.nn)
    var N = zeros(m)
    var En = zeros(m)
    var i = t
    while i < J.nh:
        _ndec_column(J, i, pd(xl), pd(xn), pd(N), pd(En))
        i += nt


struct NeutronDecay(Copyable, Movable):
    """Sources of protons (hadron grid), electrons (lepton grid) and electron
    antineutrinos (neutrino grid) from neutron decay. K_e[j, i] and K_nu[j, i]
    hold the products at grid point j per decay of a neutron of hadron bin i.
    A decay's proton goes to bins i (share ps0[i]) and i - 1 (ps1[i])."""

    var verbose: Bool
    var K_e: MatD
    var K_nu: MatD
    var ps0: Vec
    var ps1: Vec
    # sources per ln E [cm^-3 s^-1] on the hadron, lepton and neutrino grids
    var q_p: Vec
    var q_e: Vec
    var q_nu: Vec
    var w: Vec  # decays per ln E and second, n_n / (gamma tau_n)
    var nthreads: Int
    var built_nh: Int  # hadron bins of the grid the kernels were built for

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.K_e = MatD()
        self.K_nu = MatD()
        self.ps0 = Vec()
        self.ps1 = Vec()
        self.q_p = Vec()
        self.q_e = Vec()
        self.q_nu = Vec()
        self.w = Vec()
        self.nthreads = 1
        self.built_nh = -1
        self.clear_internal_arrays(dat)

    def free_kernels(mut self):
        self.K_e = MatD()
        self.K_nu = MatD()
        self.ps0 = Vec()
        self.ps1 = Vec()
        self.built_nh = -1

    def ready(self, dat: AM3Arrays) -> Bool:
        """Kernels built for the current grid."""
        return self.built_nh == dat.nbins_had

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.q_p, dat.nbins_had)
        resize_zero(self.q_e, dat.nbins_lep)
        resize_zero(self.q_nu, dat.nbins_nu)

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing neutron decay kernels")
        var job = NDecBuild(dat)
        parallel_run[ndec_build_work](job, min(self.nthreads, job.nh))
        var nh = dat.nbins_had
        self.K_e = MatD(dat.nbins_lep, nh)
        self.K_e.data = job.Ke.copy()
        self.K_e.analyze()
        self.K_nu = MatD(dat.nbins_nu, nh)
        self.K_nu.data = job.Knu.copy()
        self.K_nu.analyze()
        # the proton, at the energy fraction x_p = (m_n - E_0) / m_n, between
        # the grid points 1 (bin i) and E_{i-1} / E_i (bin i - 1)
        var f = neutron_beta_endpoint() / massN_PDG  # 1 - x_p
        self.ps0 = zeros(nh)
        self.ps1 = zeros(nh)
        for i in range(nh):
            var r = dat.egrid_had[i - 1] / dat.egrid_had[
                i
            ] if i > 0 else c_exp(-dat.dlnE)
            self.ps1[i] = f / (1.0 - r)
            self.ps0[i] = 1.0 - self.ps1[i]
        self.built_nh = nh

    def Calc_NtoX(mut self, dat: AM3Arrays, rate: Vec):
        """Decay products of the neutrons dat.n[P_neutrons] (EdN/dEdV on the
        hadron grid) decaying at the rates `rate` [1/s] (the neutron solver's
        1 / (gamma tau_n))."""
        if self.verbose:
            print("computing neutron decay terms")
        self.clear_internal_arrays(dat)
        if not self.ready(dat):
            return
        var nh = dat.nbins_had
        resize_zero(self.w, nh)
        var any_decay = False
        for i in range(nh):
            self.w[i] = dat.n[P_neutrons][i] * rate[i]
            any_decay = any_decay or self.w[i] != 0.0
        if not any_decay:
            return
        for k in range(nh):
            var v = self.ps0[k] * self.w[k]
            if k + 1 < nh:
                v += self.ps1[k + 1] * self.w[k + 1]
            self.q_p[k] = v
        gemv_d(self.K_e, pd(self.w), 1.0, pd(self.q_e))
        gemv_d(self.K_nu, pd(self.w), 1.0, pd(self.q_nu))


# ============================================================================
# Kaon decays (am3_mojo addition)
# ============================================================================
# Sources of pions, muons, e+- and neutrinos from the decays of K+, K-, K0_S
# and K0_L (model/particles.mojo), treated like AM3's pion and muon decays:
#   Q(E_i) = dlnE * sum_j T(E_i / E_j) n_K(E_j) / t_dec(E_j).
# Modes: every non-radiative one with a branching ratio above 0.1 %
# (PDG 2026), plus the smaller K0_S semileptonic and K0_L -> pi0 pi0 modes.
# Their lab-frame spectra follow from the matrix elements of KDecay (README,
# "Kaons", "Decays"). `kaon_decay_tables` integrates the kernels T exactly
# over the grid at kernel initialisation.


@always_inline
def kaon_mass(s: Int) -> Float64:
    return massK if s <= KM else massK0


@always_inline
def kaon_lifetime(s: Int) -> Float64:
    if s <= KM:
        return tdec_K
    return tdec_KS if s == KS else tdec_KL


@always_inline
def kaon_name(s: Int) -> String:
    if s == KP:
        return "kaon_plus"
    elif s == KM:
        return "kaon_minus"
    return "kaon_0S" if s == KS else "kaon_0L"


# decay products and the grids they live on
comptime PR_PIP = 0
comptime PR_PIM = 1
comptime PR_PI0 = 2
comptime PR_MUPL = 3  # mu+, left-handed (lab helicity)
comptime PR_MUPR = 4
comptime PR_MUML = 5
comptime PR_MUMR = 6
comptime PR_EM = 7  # e- (hadron-anchored lepton grid)
comptime PR_EP = 8  # e+
comptime PR_NUE = 9  # neutrino grid
comptime PR_NUEB = 10
comptime PR_NUM = 11
comptime PR_NUMB = 12
comptime N_PROD = 13


@always_inline
def product_grid(p: Int) -> Int:
    """0: hadron grid, 1: hadron-anchored lepton grid, 2: neutrino grid."""
    if p <= PR_MUMR:
        return 0
    return 1 if p <= PR_EP else 2


@always_inline
def charge_conjugate(p: Int) -> Int:
    """Product of the charge-conjugate decay (CP: a mu+ of helicity L becomes
    a mu- of helicity R)."""
    var cc: Array[Int, N_PROD] = [
        PR_PIM,
        PR_PIP,
        PR_PI0,
        PR_MUMR,
        PR_MUML,
        PR_MUPR,
        PR_MUPL,
        PR_EP,
        PR_EM,
        PR_NUEB,
        PR_NUE,
        PR_NUMB,
        PR_NUM,
    ]
    return cc[p]


# Gauss-Legendre points along a Dalitz line (the densities are low-order
# polynomials there, so this is exact) and for the rest-frame energy
comptime N_DALITZ = 24
comptime N_REST = 256

comptime _TWO_BODY = 0
comptime _K3PI = 1
comptime _KL3 = 2


struct GLRules(Copyable, Movable):
    var xi: Vec  # N_DALITZ nodes / weights on [-1, 1]
    var wi: Vec
    var xo: Vec  # N_REST nodes / weights on [-1, 1]
    var wo: Vec

    def __init__(out self):
        self.xi = Vec()
        self.wi = Vec()
        self.xo = Vec()
        self.wo = Vec()
        gauss_legendre(N_DALITZ, self.xi, self.wi)
        gauss_legendre(N_REST, self.xo, self.wo)

    def rest_node(
        self, n: Int, m: Float64, Emax: Float64
    ) -> Tuple[Float64, Float64]:
        """Node n of the rule for a rest-frame energy E on [m, Emax], after
        the substitution E = m + (Emax - m) sin^2(pi u / 2), u in [0, 1],
        which absorbs the square-root edges of the energy distributions at
        both ends: (E, weight)."""
        var a = 0.25 * M_PI * (self.xo[n] + 1.0)  # pi u / 2
        var s = c_sin(a)
        var c = c_cos(a)
        var L = Emax - m
        # from the nearer end, so that E - m and Emax - E keep their precision
        var E = m + L * s * s if s < c else Emax - L * c * c
        return (E, 0.5 * M_PI * L * s * c * self.wo[n])


struct KDecayMode(Copyable, Movable):
    """Decay K(M) -> 0 + 1 (+ 2) with product masses m0, m1, m2.

    * two-body: monochromatic in the kaon frame; P0 is the longitudinal
      polarisation of product 0 there (-1 for the mu+ of K+ -> mu+ nu).
    * K -> 3 pi (products 0, 1: even pions, 2: odd pion): PDG Dalitz-plot
      expansion |M|^2 = 1 + g u + h u^2 + k v^2 (sym3: 1 + h (u^2 + v^2 / 3)).
    * K_l3 (products pi, l+, nu): V-A matrix element with the form factors
      f+(t) = 1 + l1 t / m_pi^2 + l2 / 2 (t / m_pi^2)^2 and the scalar f0
      with slope l0; also the density of the l+ longitudinal polarisation.
    """

    var kind: Int
    var M: Float64
    var m0: Float64
    var m1: Float64
    var m2: Float64
    var P0: Float64
    var g: Float64
    var h: Float64
    var k: Float64
    var sym3: Bool
    var l1: Float64
    var l2: Float64
    var l0: Float64
    var norm: Float64

    def __init__(
        out self,
        kind: Int,
        M: Float64,
        m0: Float64,
        m1: Float64,
        m2: Float64,
        P0: Float64 = 0.0,
        g: Float64 = 0.0,
        h: Float64 = 0.0,
        k: Float64 = 0.0,
        sym3: Bool = False,
        l1: Float64 = 0.0,
        l2: Float64 = 0.0,
        l0: Float64 = 0.0,
    ):
        self.kind = kind
        self.M = M
        self.m0 = m0
        self.m1 = m1
        self.m2 = m2
        self.P0 = P0
        self.g = g
        self.h = h
        self.k = k
        self.sym3 = sym3
        self.l1 = l1
        self.l2 = l2
        self.l0 = l0
        self.norm = 1.0

    @always_inline
    def mass(self, i: Int) -> Float64:
        if i == 0:
            return self.m0
        return self.m1 if i == 1 else self.m2

    def _k3pi(self, E0: Float64, E1: Float64, E2: Float64) -> Float64:
        var M2 = self.M * self.M
        var mp2 = massPi_PDG * massPi_PDG
        var s0 = (
            M2 + self.m0 * self.m0 + self.m1 * self.m1 + self.m2 * self.m2
        ) / 3.0
        var s1 = M2 + self.m0 * self.m0 - 2.0 * self.M * E0
        var s2 = M2 + self.m1 * self.m1 - 2.0 * self.M * E1
        var s3 = M2 + self.m2 * self.m2 - 2.0 * self.M * E2
        var u = (s3 - s0) / mp2
        var v = (s2 - s1) / mp2
        if self.sym3:
            return 1.0 + self.h * (u * u + v * v / 3.0)
        return 1.0 + self.g * u + self.h * u * u + self.k * v * v

    def _kl3(self, Epi: Float64, El: Float64, pol: Bool) -> Float64:
        """|M|^2 summed over the l+ spin (pol: the part proportional to its
        longitudinal polarisation in the kaon frame), for
        H = f+ (p_K + p_pi) + f- (p_K - p_pi):
        W0 = 2 (nu.H)(l.H) - H^2 (nu.l),
        Wpol = m_l [2 (nu.H)(s.H) - H^2 (nu.s)].
        """
        var M = self.M
        var mpi = self.m0
        var ml = self.m1
        var Enu = M - Epi - El
        var t = M * M + mpi * mpi - 2.0 * M * Epi
        var mp2 = massPi_PDG * massPi_PDG
        var Fp = 1.0 + self.l1 * t / mp2 + 0.5 * self.l2 * (t / mp2) * (t / mp2)
        # f- = (f0 - f+) (M^2 - m_pi^2) / t
        var Fm = (
            (self.l0 - self.l1) / mp2 - 0.5 * self.l2 * t / (mp2 * mp2)
        ) * (M * M - mpi * mpi)
        var Kl = M * El
        var Knu = M * Enu
        var pil = 0.5 * (M * M - 2.0 * M * Enu - mpi * mpi - ml * ml)
        var pinu = 0.5 * (M * M + ml * ml - 2.0 * M * El - mpi * mpi)
        var lnu = 0.5 * (t - ml * ml)
        var nuH = Fp * (Knu + pinu) + Fm * (Knu - pinu)
        var lH = Fp * (Kl + pil) + Fm * (Kl - pil)
        var H2 = (
            Fp * Fp * (M * M + mpi * mpi + 2.0 * M * Epi)
            + 2.0 * Fp * Fm * (M * M - mpi * mpi)
            + Fm * Fm * t
        )
        if not pol:
            return 2.0 * nuH * lH - H2 * lnu
        # spin 4-vector s of the l+ along its momentum in the kaon frame
        var pl = c_sqrt(max(El * El - ml * ml, 1e-300))
        var ppi2 = Epi * Epi - mpi * mpi
        var pnu2 = Enu * Enu
        var ppi_pl = 0.5 * (pnu2 - ppi2 - pl * pl)
        var pnu_pl = 0.5 * (ppi2 - pnu2 - pl * pl)
        var Ks = M * pl / ml
        var pis = (Epi * pl - El * ppi_pl / pl) / ml
        var nus = (Enu * pl - El * pnu_pl / pl) / ml
        var Hs = Fp * (Ks + pis) + Fm * (Ks - pis)
        return ml * (2.0 * nuH * Hs - H2 * nus)

    def density(
        self, E0: Float64, E1: Float64, E2: Float64, pol: Bool
    ) -> Float64:
        if self.kind == _KL3:
            return self._kl3(E0, E1, pol)
        return self._k3pi(E0, E1, E2)

    def emax(self, i: Int) -> Float64:
        var j = 1 if i == 0 else 0
        var k = 1 if i == 2 else 2
        var mjk = self.mass(j) + self.mass(k)
        var mi = self.mass(i)
        return (self.M * self.M + mi * mi - mjk * mjk) / (2.0 * self.M)

    def energy_density(
        self, i: Int, E: Float64, pol: Bool, gl: GLRules
    ) -> Float64:
        """Rest-frame energy distribution of product i (normalised to 1), or
        (pol) its longitudinal polarisation times that distribution: the
        Dalitz density integrated along the line E_i = E."""
        var j = 1 if i == 0 else 0
        var k = 1 if i == 2 else 2
        var M = self.M
        var mi = self.mass(i)
        var mj = self.mass(j)
        var mk = self.mass(k)
        var P = c_sqrt(max(E * E - mi * mi, 0.0))
        var mjk2 = M * M + mi * mi - 2.0 * M * E
        var mjk = c_sqrt(max(mjk2, 1e-300))
        var Ejs = (mjk2 + mj * mj - mk * mk) / (2.0 * mjk)
        var pjs = c_sqrt(max(Ejs * Ejs - mj * mj, 0.0))
        var lo = ((M - E) * Ejs - P * pjs) / mjk
        var hi = ((M - E) * Ejs + P * pjs) / mjk
        var s = 0.0
        var Es: Array[Float64, 3] = [0.0, 0.0, 0.0]
        for n in range(N_DALITZ):
            var Ej = 0.5 * (hi - lo) * gl.xi[n] + 0.5 * (hi + lo)
            Es[i] = E
            Es[j] = Ej
            Es[k] = M - E - Ej
            s += (
                self.density(Es[0], Es[1], Es[2], pol)
                * 0.5
                * (hi - lo)
                * gl.wi[n]
            )
        return s / self.norm

    def normalise(mut self, gl: GLRules):
        if self.kind == _TWO_BODY:
            return
        self.norm = 1.0
        var Emax = self.emax(0)
        var s = 0.0
        for n in range(N_REST):
            var E: Float64
            var w: Float64
            E, w = gl.rest_node(n, self.m0, Emax)
            s += self.energy_density(0, E, False, gl) * w
        self.norm = s


# branching ratios (PDG 2026)
comptime BR_K_MUNU = 0.6356
comptime BR_K_PIPI0 = 0.2067
comptime BR_K_3PI = 0.05583  # pi+ pi+ pi-
comptime BR_K_E3 = 0.0507
comptime BR_K_MU3 = 0.03352
comptime BR_K_PIPI0PI0 = 0.01760
comptime BR_KS_PIPI = 0.6920
comptime BR_KS_PI0PI0 = 0.3069
comptime BR_KS_E3 = 7.14e-4  # both charge states
comptime BR_KS_MU3 = 4.56e-4
comptime BR_KL_E3 = 0.4055
comptime BR_KL_MU3 = 0.2704
comptime BR_KL_3PI0 = 0.1952
comptime BR_KL_PIPIPI0 = 0.1254
comptime BR_KL_PIPI = 1.967e-3
comptime BR_KL_PI0PI0 = 8.64e-4
# K_L charge asymmetries [G(pi- l+ nu) - G(pi+ l- nu-bar)] / sum
comptime AL_E = 0.00334
comptime AL_MU = 0.00304

# decay modes (index into kaon_decay_modes)
comptime _M_KMUNU = 0  # K+ -> mu+ nu_mu
comptime _M_KPIPI0 = 1  # K+ -> pi+ pi0
comptime _M_K0PIPI = 2  # K0 -> pi+ pi-
comptime _M_K0PI0PI0 = 3
comptime _M_KE3 = 4  # K+ -> pi0 e+ nu_e
comptime _M_KMU3 = 5
comptime _M_K0E3 = 6  # K0 -> pi- e+ nu_e
comptime _M_K0MU3 = 7
comptime _M_K3PI = 8  # K+ -> pi+ pi+ pi-
comptime _M_KPIPI0PI0 = 9  # K+ -> pi0 pi0 pi+
comptime _M_KLPIPIPI0 = 10  # K_L -> pi+ pi- pi0
comptime _M_KL3PI0 = 11


def kaon_decay_modes(gl: GLRules) -> List[KDecayMode]:
    """The decay modes, with the K_l3 form factors and K -> 3 pi Dalitz slopes
    of PDG 2026. Form factors: PDG's fits assuming mu-e universality (K0:
    quadratic f+ and linear f0; K+-: linear f+ and f0, as PDG has no K+-
    quadratic fit with universality)."""
    var mK = massK
    var mK0 = massK0
    var mpi = massPi_PDG
    var mpi0 = massPi0_PDG
    var md: List[KDecayMode] = [
        KDecayMode(_TWO_BODY, mK, massMu_PDG, 0.0, 0.0, P0=-1.0),
        KDecayMode(_TWO_BODY, mK, mpi, mpi0, 0.0),
        KDecayMode(_TWO_BODY, mK0, mpi, mpi, 0.0),
        KDecayMode(_TWO_BODY, mK0, mpi0, mpi0, 0.0),
        KDecayMode(
            _KL3, mK, mpi0, massE_PDG, 0.0, l1=0.02959, l2=0.0, l0=0.0176
        ),
        KDecayMode(
            _KL3, mK, mpi0, massMu_PDG, 0.0, l1=0.02959, l2=0.0, l0=0.0176
        ),
        KDecayMode(
            _KL3, mK0, mpi, massE_PDG, 0.0, l1=0.0240, l2=0.0020, l0=0.0116
        ),
        KDecayMode(
            _KL3, mK0, mpi, massMu_PDG, 0.0, l1=0.0240, l2=0.0020, l0=0.0116
        ),
        KDecayMode(_K3PI, mK, mpi, mpi, mpi, g=-0.21134, h=0.01848, k=-0.00463),
        KDecayMode(_K3PI, mK, mpi0, mpi0, mpi, g=0.626, h=0.052, k=0.0054),
        KDecayMode(_K3PI, mK0, mpi, mpi, mpi0, g=0.678, h=0.076, k=0.0099),
        KDecayMode(_K3PI, mK0, mpi0, mpi0, mpi0, h=0.00059, sym3=True),
    ]
    for i in range(len(md)):
        md[i].normalise(gl)
    return md^


@fieldwise_init
struct KTerm(Copyable, Movable):
    """br products p per decay from product i of mode `mode`; hel: lab
    helicity of a muon (+1 R, -1 L, 0 both)."""

    var br: Float64
    var p: Int
    var mode: Int
    var i: Int
    var hel: Int


def _semileptonic_terms(
    mut T: List[KTerm],
    b_e3: Float64,
    b_mu3: Float64,
    a_e: Float64,
    a_mu: Float64,
):
    """K0-type semileptonic decays: pi- l+ nu with probability (1 + A) / 2,
    pi+ l- nu-bar with (1 - A) / 2 (the l- with the opposite helicity)."""
    var fp = 0.5 * (1.0 + a_e) * b_e3
    var fm = 0.5 * (1.0 - a_e) * b_e3
    T.append(KTerm(fp, PR_PIM, _M_K0E3, 0, 0))
    T.append(KTerm(fp, PR_EP, _M_K0E3, 1, 0))
    T.append(KTerm(fp, PR_NUE, _M_K0E3, 2, 0))
    T.append(KTerm(fm, PR_PIP, _M_K0E3, 0, 0))
    T.append(KTerm(fm, PR_EM, _M_K0E3, 1, 0))
    T.append(KTerm(fm, PR_NUEB, _M_K0E3, 2, 0))
    fp = 0.5 * (1.0 + a_mu) * b_mu3
    fm = 0.5 * (1.0 - a_mu) * b_mu3
    T.append(KTerm(fp, PR_PIM, _M_K0MU3, 0, 0))
    T.append(KTerm(fp, PR_MUPL, _M_K0MU3, 1, -1))
    T.append(KTerm(fp, PR_MUPR, _M_K0MU3, 1, 1))
    T.append(KTerm(fp, PR_NUM, _M_K0MU3, 2, 0))
    T.append(KTerm(fm, PR_PIP, _M_K0MU3, 0, 0))
    T.append(KTerm(fm, PR_MUMR, _M_K0MU3, 1, -1))
    T.append(KTerm(fm, PR_MUML, _M_K0MU3, 1, 1))
    T.append(KTerm(fm, PR_NUMB, _M_K0MU3, 2, 0))


def kaon_decay_terms(s: Int) -> List[KTerm]:
    """Products of K+, K0_S and K0_L, with the multiplicities per decay
    renormalised to the modes that are followed (K-: charge conjugates of K+).
    """
    var T = List[KTerm]()
    var total: Float64
    if s == KS:
        T.append(KTerm(BR_KS_PIPI, PR_PIP, _M_K0PIPI, 0, 0))
        T.append(KTerm(BR_KS_PIPI, PR_PIM, _M_K0PIPI, 0, 0))
        T.append(KTerm(2.0 * BR_KS_PI0PI0, PR_PI0, _M_K0PI0PI0, 0, 0))
        _semileptonic_terms(T, BR_KS_E3, BR_KS_MU3, 0.0, 0.0)
        total = BR_KS_PIPI + BR_KS_PI0PI0 + BR_KS_E3 + BR_KS_MU3
    elif s == KL:
        _semileptonic_terms(T, BR_KL_E3, BR_KL_MU3, AL_E, AL_MU)
        T.append(KTerm(3.0 * BR_KL_3PI0, PR_PI0, _M_KL3PI0, 0, 0))
        T.append(KTerm(BR_KL_PIPIPI0, PR_PIP, _M_KLPIPIPI0, 0, 0))
        T.append(KTerm(BR_KL_PIPIPI0, PR_PIM, _M_KLPIPIPI0, 1, 0))
        T.append(KTerm(BR_KL_PIPIPI0, PR_PI0, _M_KLPIPIPI0, 2, 0))
        T.append(KTerm(BR_KL_PIPI, PR_PIP, _M_K0PIPI, 0, 0))
        T.append(KTerm(BR_KL_PIPI, PR_PIM, _M_K0PIPI, 0, 0))
        T.append(KTerm(2.0 * BR_KL_PI0PI0, PR_PI0, _M_K0PI0PI0, 0, 0))
        total = (
            BR_KL_E3
            + BR_KL_MU3
            + BR_KL_3PI0
            + BR_KL_PIPIPI0
            + BR_KL_PIPI
            + BR_KL_PI0PI0
        )
    else:
        T.append(KTerm(BR_K_MUNU, PR_MUPL, _M_KMUNU, 0, -1))
        T.append(KTerm(BR_K_MUNU, PR_MUPR, _M_KMUNU, 0, 1))
        T.append(KTerm(BR_K_MUNU, PR_NUM, _M_KMUNU, 1, 0))
        T.append(KTerm(BR_K_PIPI0, PR_PIP, _M_KPIPI0, 0, 0))
        T.append(KTerm(BR_K_PIPI0, PR_PI0, _M_KPIPI0, 1, 0))
        T.append(KTerm(2.0 * BR_K_3PI, PR_PIP, _M_K3PI, 0, 0))
        T.append(KTerm(BR_K_3PI, PR_PIM, _M_K3PI, 2, 0))
        T.append(KTerm(2.0 * BR_K_PIPI0PI0, PR_PI0, _M_KPIPI0PI0, 0, 0))
        T.append(KTerm(BR_K_PIPI0PI0, PR_PIP, _M_KPIPI0PI0, 2, 0))
        T.append(KTerm(BR_K_E3, PR_PI0, _M_KE3, 0, 0))
        T.append(KTerm(BR_K_E3, PR_EP, _M_KE3, 1, 0))
        T.append(KTerm(BR_K_E3, PR_NUE, _M_KE3, 2, 0))
        T.append(KTerm(BR_K_MU3, PR_PI0, _M_KMU3, 0, 0))
        T.append(KTerm(BR_K_MU3, PR_MUPL, _M_KMU3, 1, -1))
        T.append(KTerm(BR_K_MU3, PR_MUPR, _M_KMU3, 1, 1))
        T.append(KTerm(BR_K_MU3, PR_NUM, _M_KMU3, 2, 0))
        total = (
            BR_K_MUNU
            + BR_K_PIPI0
            + BR_K_3PI
            + BR_K_E3
            + BR_K_MU3
            + BR_K_PIPI0PI0
        )
        if s == KM:
            for t in range(len(T)):
                T[t].p = charge_conjugate(T[t].p)
    for t in range(len(T)):
        T[t].br /= total
    return T^


def _bin_node(
    M: Float64,
    m: Float64,
    E: Float64,
    wg: Float64,
    wG: Float64,
    xc: Vec,
    dlnE: Float64,
    mut N: FPtr,
    mut En: FPtr,
):
    """Number N[q] and energy En[q] (in units of the kaon energy) between
    the grid points x_{q+1} < x < x_q of the products with rest-frame energy
    E: uniform in x on [(E - p)/M, (E + p)/M] with density M / (2 p), times
    wg + wG htilde(x), htilde = (E M x - m^2) / (p M x) (lab helicity of a
    muon with longitudinal polarisation 1 in the kaon frame)."""
    var K = len(xc) - 1
    var p = c_sqrt(max(E * E - m * m, 0.0))
    if p <= 0.0:
        return
    var a = (E - p) / M
    var b = (E + p) / M
    var dens = M / (2.0 * p)
    var qlo = max(0, to_int(c_floor(-c_log(b) / dlnE)) - 1)
    var qhi = K - 1
    if a > 0.0:
        qhi = min(K - 1, to_int(c_floor(-c_log(a) / dlnE)) + 1)
    var c1 = E / p
    var c2 = m * m / (p * M)
    for q in range(qlo, qhi + 1):
        var x2 = min(b, xc[q])
        var x1 = max(a, xc[q + 1])
        if x2 <= x1:
            continue
        var L1 = x2 - x1
        var L2 = 0.5 * (x2 - x1) * (x2 + x1)
        var n = wg * L1
        var e = wg * L2
        if wG != 0.0:
            n += wG * (c1 * L1 - c2 * c_log(x2 / x1))
            e += wG * (c1 * L2 - c2 * L1)
        N[unsafe_offset=q] += dens * n
        En[unsafe_offset=q] += dens * e


def kaon_decay_tables(dlnE: Float64, K: Int) -> List[Vec]:
    """Decay kernels T[s * N_PROD + p][k], k = 0 .. K: products p per decay of
    kaon species s at the energy fraction x_k = exp(-k dlnE), per unit ln x.

    Exact: for each rest-frame energy E of a product (one for two-body decays,
    the N_REST Gauss-Legendre nodes of its distribution for three-body ones),
    `_bin_node` integrates the number and energy of the products between two
    grid points in closed form. They go to these two points linearly in x,
    which keeps both."""
    var gl = GLRules()
    var modes = kaon_decay_modes(gl)
    var xc = zeros(K + 1)
    for k in range(K + 1):
        xc[k] = c_exp(-Float64(k) * dlnE)
    var tabs = List[Vec]()
    for _ in range(N_KAON * N_PROD):
        tabs.append(Vec())
    for s in range(N_KAON):
        var N = zeros(N_PROD * K)
        var En = zeros(N_PROD * K)
        var terms = kaon_decay_terms(s)
        for t in terms:
            ref md = modes[t.mode]
            var m = md.mass(t.i)
            var Np = pd(N).unsafe_offset(t.p * K)
            var Ep = pd(En).unsafe_offset(t.p * K)
            if md.kind == _TWO_BODY:
                var mo = md.mass(1 - t.i)
                var E = (md.M * md.M + m * m - mo * mo) / (2.0 * md.M)
                var P = md.P0 if t.i == 0 else 0.0
                if t.hel == 0:
                    _bin_node(md.M, m, E, t.br, 0.0, xc, dlnE, Np, Ep)
                else:
                    var h = 0.5 * Float64(t.hel) * P * t.br
                    _bin_node(md.M, m, E, 0.5 * t.br, h, xc, dlnE, Np, Ep)
                continue
            var Emax = md.emax(t.i)
            for n in range(N_REST):
                var E: Float64
                var w: Float64
                E, w = gl.rest_node(n, m, Emax)
                w *= t.br
                var g = md.energy_density(t.i, E, False, gl)
                if t.hel == 0:
                    _bin_node(md.M, m, E, g * w, 0.0, xc, dlnE, Np, Ep)
                else:
                    var G = md.energy_density(t.i, E, True, gl)
                    var h = 0.5 * Float64(t.hel) * G * w
                    _bin_node(md.M, m, E, 0.5 * g * w, h, xc, dlnE, Np, Ep)
        for p in range(N_PROD):
            var C = zeros(K + 1)
            for q in range(K):
                var n = N[p * K + q]
                var e = En[p * K + q]
                var dx = xc[q] - xc[q + 1]
                C[q] += (e - xc[q + 1] * n) / dx / dlnE
                C[q + 1] += (xc[q] * n - e) / dx / dlnE
            tabs[s * N_PROD + p] = C^
    return tabs^


struct KDecJob(Movable):
    var D: Pointer[KaonDecay, MutUntrackedOrigin]
    var dlnE: Float64

    def __init__(
        out self, D: Pointer[KaonDecay, MutUntrackedOrigin], dlnE: Float64
    ):
        self.D = D
        self.dlnE = dlnE


def kdec_work(job: Pointer[KDecJob, MutUntrackedOrigin], t: Int, nt: Int):
    """Products t, t + nt, ...: sum over the kaon species in a fixed order."""
    ref D = job[].D[]
    var p = t
    while p < N_PROD:
        var o = pd(D.out[p])
        var tmp = pd(D.tmp[p])
        for s in range(N_KAON):
            if not D.has[s * N_PROD + p] or not D.active[s]:
                continue
            gemv_d(D.K[s * N_PROD + p], pd(D.w[s]), job[].dlnE, tmp)
            for i in range(len(D.out[p])):
                o[unsafe_offset=i] += tmp[unsafe_offset=i]
        p += nt


struct KaonDecay(Copyable, Movable):
    """Sources of pions, muons, e+- and neutrinos from the decay of the four
    kaon species. T(E_i / E_j) depends only on i - j, so each (species,
    product) kernel, summed over the decay modes, is a Toeplitz matrix stored
    by its diagonal values."""

    var verbose: Bool
    var K: List[MatD]  # kernel of (species s, product p) at s * N_PROD + p
    var has: List[Bool]
    var out: List[Vec]  # source of product p (on its grid)
    var w: List[Vec]  # n_K / t_dec per species
    var active: List[Bool]  # species with a non-zero density
    var tmp: List[Vec]  # work space per product
    var nthreads: Int
    var built_nh: Int  # hadron bins of the grid the kernels were built for

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.K = List[MatD]()
        self.has = List[Bool]()
        for _ in range(N_KAON * N_PROD):
            self.K.append(MatD())
            self.has.append(False)
        self.out = List[Vec]()
        self.tmp = List[Vec]()
        for _ in range(N_PROD):
            self.out.append(Vec())
            self.tmp.append(Vec())
        self.w = List[Vec]()
        self.active = List[Bool]()
        for _ in range(N_KAON):
            self.w.append(Vec())
            self.active.append(False)
        self.nthreads = 1
        self.built_nh = -1
        self.clear_internal_arrays(dat)

    def free_kernels(mut self):
        for k in range(len(self.K)):
            self.K[k] = MatD()
            self.has[k] = False
        self.built_nh = -1

    def ready(self, dat: AM3Arrays) -> Bool:
        """Kernels built for the current grid."""
        return self.built_nh == dat.nbins_had

    @staticmethod
    def _rows(dat: AM3Arrays, p: Int) -> Int:
        var g = product_grid(p)
        if g == 0:
            return dat.nbins_had
        return dat.nbins_lep_hadgrid if g == 1 else dat.nbins_nu

    @staticmethod
    def _energy(dat: AM3Arrays, p: Int, i: Int) -> Float64:
        var g = product_grid(p)
        if g == 0:
            return dat.egrid_had[i]
        return dat.egrid_had_leptons[i] if g == 1 else dat.egrid_nu[i]

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        for p in range(N_PROD):
            resize_zero(self.out[p], KaonDecay._rows(dat, p))

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing kaon decay kernels")
        var nh = dat.nbins_had
        # the grids are logarithmic with step dlnE, so E_i / E_j = x_k with
        # integer k; K: the k of the lowest x on any product grid
        var K = 0
        for p in range(N_PROD):
            var x = KaonDecay._energy(dat, p, 0) / dat.egrid_had[nh - 1]
            K = max(K, to_int(c_floor(-c_log(x) / dat.dlnE + 0.5)))
        var T = kaon_decay_tables(dat.dlnE, K)
        for s in range(N_KAON):
            for p in range(N_PROD):
                var rows = KaonDecay._rows(dat, p)
                var nt = rows + nh - 1
                var t = zeros(nt)
                var used = False
                for m in range(nt):
                    # A[i, j] = t[i - j + nh - 1]: pick a valid (i, j)
                    var j = max(0, nh - 1 - m)
                    var i = m - (nh - 1) + j
                    var k = to_int(
                        c_floor(
                            -c_log(
                                KaonDecay._energy(dat, p, i) / dat.egrid_had[j]
                            )
                            / dat.dlnE
                            + 0.5
                        )
                    )
                    if k >= 0 and k <= K:
                        t[m] = T[s * N_PROD + p][k]
                        used = used or t[m] != 0.0
                self.has[s * N_PROD + p] = used
                self.K[s * N_PROD + p] = MatD.toeplitz(
                    rows, nh, t^
                ) if used else MatD()
        self.built_nh = nh

    def Calc_KtoX(mut self, dat: AM3Arrays, base: Int):
        """Decay products of the kaon densities dat.n[base + s] (EdN/dEdV on
        the hadron grid; base: the first species of a kaon population)."""
        if self.verbose:
            print("computing kaon decay terms")
        self.clear_internal_arrays(dat)
        if not self.ready(dat):
            return
        var nh = dat.nbins_had
        var any_active = False
        for s in range(N_KAON):
            resize_zero(self.w[s], nh)
            self.active[s] = False
            var r = kaon_mass(s) / kaon_lifetime(s)
            for i in range(nh):
                self.w[s][i] = dat.n[base + s][i] * r / dat.egrid_had[i]
                self.active[s] = self.active[s] or self.w[s][i] != 0.0
            any_active = any_active or self.active[s]
        if not any_active:
            return
        for p in range(N_PROD):
            resize_zero(self.tmp[p], len(self.out[p]))
        var job = KDecJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](), dat.dlnE
        )
        parallel_run[kdec_work](job, min(self.nthreads, N_PROD))
