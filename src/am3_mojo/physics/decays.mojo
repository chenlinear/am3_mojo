"""Ports of PionDecay (PionDecay.cc), MuonDecay (MuonDecay.cc) and HadIC
(HadronicInverseCompton.cc)."""

from std.math import fma
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.linalg import MatD, gemv_d


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
        resize_zero(self.NUmGen_pi, dat.nbins_had)
        resize_zero(self.NUmAGen_pi, dat.nbins_had)
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
    var deltaE_p: VecI
    var jmax_p: VecI
    var deltaE_pi: VecI
    var jmax_pi: VecI
    var deltaE_mu: VecI
    var jmax_mu: VecI
    var w_in: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.epsilonPIC = Vec()
        self.epsilonPiIC = Vec()
        self.epsilonMuIC = Vec()
        self.deltaE_p = VecI()
        self.jmax_p = VecI()
        self.deltaE_pi = VecI()
        self.jmax_pi = VecI()
        self.deltaE_mu = VecI()
        self.jmax_mu = VecI()
        self.w_in = Vec()
        self.clear_internal_arrays(dat)
        self.reinitialize_empty_kernels(dat, True, True, True)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.epsilonPIC, dat.nbins_pho)
        resize_zero(self.epsilonPiIC, dat.nbins_pho)
        resize_zero(self.epsilonMuIC, dat.nbins_pho)

    def reinitialize_empty_kernels(
        mut self, dat: AM3Arrays, rad_P: Bool, rad_Pi: Bool, rad_Mu: Bool
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
        mut self, dat: AM3Arrays, rad_P: Bool, rad_Pi: Bool, rad_Mu: Bool
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

    @staticmethod
    def _emit(
        dat: AM3Arrays,
        dE: VecI,
        jmax: VecI,
        part: Vec,
        coeff: Float64,
        mut out_: Vec,
    ):
        """out[j] = dlnE*coeff * sum_i ph[j - dE[i]] * part[i]  over dE[i] <= j <= jmax[i].
        """
        var ph = pd(dat.n[P_photons_on_had_grid])
        var pp = pd(part)
        var npho = dat.nbins_pho
        var nh = dat.nbins_had
        for j in range(npho):
            var z = 0.0
            for i in range(nh):
                var d = dE[i]
                if j < d or j > jmax[i]:
                    continue
                z = fma(ph[unsafe_offset=j - d], pp[unsafe_offset=i], z)
            out_[j] = z * dat.dlnE * coeff

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
