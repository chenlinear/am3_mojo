"""Ports of PionDecay (PionDecay.cc), MuonDecay (MuonDecay.cc) and HadIC
(HadronicInverseCompton.cc), and the neutron, kaon and charm hadron decays
(am3_mojo additions)."""

from std.math import floor, sqrt
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.physics.lambda_c_semileptonic import (
    BL3FormFactors,
    bl3_density,
)
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import (
    MatD,
    toeplitz_window_zero,
    toeplitz_gemv,
    gemv_d,
    GemvPair,
    run_gemv_pairs,
)
from am3_mojo.utils.parallel import parallel_run, parallel_tasks
from am3_mojo.utils.specfun import gauss_legendre


def PMu1(x: Float64) -> Float64:
    """A primitive of the energy fraction distribution of the mu+_R (mu-_L) of
    pi+ (pi-) decay, r (1 - x) / ((1 - r)^2 x) on [r, 1] with r = (m_mu/m_pi)^2
    (AM3 paper, Lipari et al. 2007)."""
    var rpi = MASS_MU * MASS_MU / MASS_PI / MASS_PI
    return rpi * (c_log(x) - x) / (1.0 - rpi) / (1.0 - rpi)


def PMu2(x: Float64) -> Float64:
    """A primitive of that of the mu+_L (mu-_R), (x - r) / ((1 - r)^2 x)."""
    var rpi = MASS_MU * MASS_MU / MASS_PI / MASS_PI
    return (x - rpi * c_log(x)) / (1.0 - rpi) / (1.0 - rpi)


def FPi2NU(x: Float64) -> Float64:
    var rpi = MASS_MU * MASS_MU / MASS_PI / MASS_PI
    if x < 0.0 or x > 1.0 - rpi:
        return 0.0
    return 1.0 / (1.0 - rpi)


def _sized(mut sets: List[Vec], n: Int, size: Int):
    while len(sets) < n:
        sets.append(zeros(size))
    for k in range(len(sets)):
        if len(sets[k]) != size:
            sets[k] = zeros(size)


struct PionDecay(Copyable, Movable):
    """The decays of the charged pions of several pion sets (all pions, the
    components, the decay chains) run together."""

    var nthreads: Int
    var verbose: Bool
    var MpLGen: List[Vec]
    var MpRGen: List[Vec]
    var MmLGen: List[Vec]
    var MmRGen: List[Vec]
    var NUmGen_pi: List[Vec]
    var NUmAGen_pi: List[Vec]
    var QPI2GGen: Vec
    var KER_FMU1: MatD
    var KER_FMU2: MatD
    var KER_FPi2NU: MatD
    var KER_FPi2G: MatD
    var w_p: List[Vec]
    var w_m: List[Vec]

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.nthreads = 1
        self.verbose = verbose
        self.MpLGen = List[Vec]()
        self.MpRGen = List[Vec]()
        self.MmLGen = List[Vec]()
        self.MmRGen = List[Vec]()
        self.NUmGen_pi = List[Vec]()
        self.NUmAGen_pi = List[Vec]()
        self.QPI2GGen = Vec()
        self.KER_FMU1 = MatD()
        self.KER_FMU2 = MatD()
        self.KER_FPi2NU = MatD()
        self.KER_FPi2G = MatD()
        self.w_p = List[Vec]()
        self.w_m = List[Vec]()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        self.MpLGen = [zeros(dat.nbins_had)]
        self.MpRGen = [zeros(dat.nbins_had)]
        self.MmLGen = [zeros(dat.nbins_had)]
        self.MmRGen = [zeros(dat.nbins_had)]
        self.NUmGen_pi = [zeros(dat.nbins_nu)]
        self.NUmAGen_pi = [zeros(dat.nbins_nu)]
        resize_zero(self.QPI2GGen, dat.nbins_pho)

    def initialize_kernels(mut self, dat: AM3Arrays):
        """The kernels are functions of the ratio of the energies. So on the
        grids of one dlnE, they are Toeplitz matrices. KER_FMU2 is kept dense.
        Its differences of PMu2 cancel, so its values vary along a diagonal by
        1.2e-11. This is more than the tolerance of analyze() for a Toeplitz
        matrix (1e-12)."""
        if self.verbose:
            print("computing charged pion decay to muon + neutrino kernels")
        var nh = dat.nbins_had
        # The muons of a decay whose energy fraction lies in the bin of grid
        # point i. With these bins, every decay makes one muon. AM3 took the
        # distributions at the grid points and dropped x = 1, where the
        # distribution of the mu+_L is largest. In AM3, a decay made
        # 0.876 muons with 15 % too little energy (AM3 bug).
        var rpi = MASS_MU * MASS_MU / MASS_PI / MASS_PI
        var half = c_exp(0.5 * dat.dlnE)
        var t1 = zeros(2 * nh - 1)
        for m in range(2 * nh - 1):
            var i: Int
            var j: Int
            i, j = MatD.toeplitz_entry(m, nh)
            var r = dat.egrid_had[i] / dat.egrid_had[j]
            var lo = min(max(r / half, rpi), 1.0)
            var hi = min(max(r * half, rpi), 1.0)
            t1[m] = (PMu1(hi) - PMu1(lo)) / dat.dlnE
        self.KER_FMU1 = MatD.toeplitz(nh, nh, t1^)
        self.KER_FMU2 = MatD(nh, nh)
        for i in range(nh):
            for j in range(nh):
                var r = dat.egrid_had[i] / dat.egrid_had[j]
                var lo = min(max(r / half, rpi), 1.0)
                var hi = min(max(r * half, rpi), 1.0)
                self.KER_FMU2.set(i, j, (PMu2(hi) - PMu2(lo)) / dat.dlnE)
        var nn = dat.nbins_nu
        var tn = zeros(nn + nh - 1)
        for m in range(nn + nh - 1):
            var i: Int
            var j: Int
            i, j = MatD.toeplitz_entry(m, nh)
            var r = dat.egrid_nu[i] / dat.egrid_had[j]
            tn[m] = r * FPi2NU(r)
        self.KER_FPi2NU = MatD.toeplitz(nn, nh, tn^)
        if self.verbose:
            print("computing pi0 decay to gamma kernel")
        # The photons of a decay whose energy fraction lies in the bin of grid
        # point i, as for the muons above. So every decay makes two photons,
        # with 1.001 of the pion's energy. AM3 took 2 x at
        # x = 2 E_gamma / (E_j + E_j+1), the pion between its grid point and
        # the next. In AM3, the photons of a decay carried 1.048 times its
        # energy (AM3 bug).
        var npho = dat.nbins_pho
        var tg = zeros(npho + nh - 1)
        for m in range(npho + nh - 1):
            var i: Int
            var j: Int
            i, j = MatD.toeplitz_entry(m, nh)
            var r = dat.egrid_had_photons[i] / dat.egrid_had[j]
            tg[m] = 2.0 * (min(r * half, 1.0) - min(r / half, 1.0)) / dat.dlnE
        self.KER_FPi2G = MatD.toeplitz(npho, nh, tg^)

    def _rates(mut self, dat: AM3Arrays, pions: List[Int]):
        var nh = dat.nbins_had
        _sized(self.w_p, len(pions), nh)
        _sized(self.w_m, len(pions), nh)
        for k in range(len(pions)):
            ref Pip = dat.n[pions[k]]
            ref Pim = dat.n[pions[k] + 1]
            for i in range(nh):
                var tdec = dat.egrid_had[i] / MASS_PI * TAU_PI
                self.w_p[k][i] = Pip[i] / tdec
                self.w_m[k][i] = Pim[i] / tdec

    def Calc_PItoMU(mut self, dat: AM3Arrays, pions: List[Int]):
        if self.verbose:
            print("computing pion decay to muon term")
        var n = len(pions)
        _sized(self.MpLGen, n, self.KER_FMU2.rows)
        _sized(self.MpRGen, n, self.KER_FMU1.rows)
        _sized(self.MmLGen, n, self.KER_FMU1.rows)
        _sized(self.MmRGen, n, self.KER_FMU2.rows)
        self._rates(dat, pions)
        var jobs = List[GemvPair]()
        for k in range(n):
            var wp = pd(self.w_p[k])
            var wm = pd(self.w_m[k])
            jobs.append(
                GemvPair(
                    self.KER_FMU1,
                    wp,
                    wm,
                    dat.dlnE,
                    pd(self.MpRGen[k]),
                    pd(self.MmLGen[k]),
                )
            )
            jobs.append(
                GemvPair(
                    self.KER_FMU2,
                    wp,
                    wm,
                    dat.dlnE,
                    pd(self.MpLGen[k]),
                    pd(self.MmRGen[k]),
                )
            )
        run_gemv_pairs(jobs, self.nthreads)

    def Calc_PItoNU(mut self, dat: AM3Arrays, pions: List[Int]):
        if self.verbose:
            print("computing pion decay to neutrino term")
        var n = len(pions)
        _sized(self.NUmGen_pi, n, self.KER_FPi2NU.rows)
        _sized(self.NUmAGen_pi, n, self.KER_FPi2NU.rows)
        self._rates(dat, pions)
        var jobs = List[GemvPair]()
        for k in range(n):
            jobs.append(
                GemvPair(
                    self.KER_FPi2NU,
                    pd(self.w_p[k]),
                    pd(self.w_m[k]),
                    dat.dlnE,
                    pd(self.NUmGen_pi[k]),
                    pd(self.NUmAGen_pi[k]),
                )
            )
        run_gemv_pairs(jobs, self.nthreads)

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


struct MuonDecay(Copyable, Movable):
    """The decays of the muons of several muon sets run together (as
    PionDecay)."""

    var nthreads: Int
    var verbose: Bool
    var NUeGen_mu: List[Vec]
    var NUmGen_mu: List[Vec]
    var NUeAGen_mu: List[Vec]
    var NUmAGen_mu: List[Vec]
    var QMU2EGen_mu: List[Vec]
    var QMU2PGen_mu: List[Vec]
    var KER_FMU2Ep: MatD
    var KER_FMU2Em: MatD
    var KER_FNU1p: MatD
    var KER_FNU1m: MatD
    var KER_FNU2p: MatD
    var KER_FNU2m: MatD
    var w_ml: List[Vec]
    var w_pl: List[Vec]
    var w_mr: List[Vec]
    var w_pr: List[Vec]
    var t_nu: List[Vec]
    var t_e: List[Vec]

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.nthreads = 1
        self.verbose = verbose
        self.NUeGen_mu = List[Vec]()
        self.NUmGen_mu = List[Vec]()
        self.NUeAGen_mu = List[Vec]()
        self.NUmAGen_mu = List[Vec]()
        self.QMU2EGen_mu = List[Vec]()
        self.QMU2PGen_mu = List[Vec]()
        self.KER_FMU2Ep = MatD()
        self.KER_FMU2Em = MatD()
        self.KER_FNU1p = MatD()
        self.KER_FNU1m = MatD()
        self.KER_FNU2p = MatD()
        self.KER_FNU2m = MatD()
        self.w_ml = List[Vec]()
        self.w_pl = List[Vec]()
        self.w_mr = List[Vec]()
        self.w_pr = List[Vec]()
        self.t_nu = List[Vec]()
        self.t_e = List[Vec]()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        self.NUeGen_mu = [zeros(dat.nbins_nu)]
        self.NUeAGen_mu = [zeros(dat.nbins_nu)]
        self.NUmGen_mu = [zeros(dat.nbins_nu)]
        self.NUmAGen_mu = [zeros(dat.nbins_nu)]
        self.QMU2EGen_mu = [zeros(dat.nbins_lep)]
        self.QMU2PGen_mu = [zeros(dat.nbins_lep)]

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing muon decay to neutrino kernels")
        var nh = dat.nbins_had
        var nn = dat.nbins_nu
        var t1p = zeros(nn + nh - 1)
        var t1m = zeros(nn + nh - 1)
        var t2p = zeros(nn + nh - 1)
        var t2m = zeros(nn + nh - 1)
        for m in range(nn + nh - 1):
            var i: Int
            var j: Int
            i, j = MatD.toeplitz_entry(m, nh)
            var r = dat.egrid_nu[i] / dat.egrid_had[j]
            t1p[m] = r * FNU1(r, 1.0)
            t1m[m] = r * FNU1(r, -1.0)
            t2p[m] = r * FNU2(r, 1.0)
            t2m[m] = r * FNU2(r, -1.0)
        self.KER_FNU1p = MatD.toeplitz(nn, nh, t1p^)
        self.KER_FNU1m = MatD.toeplitz(nn, nh, t1m^)
        self.KER_FNU2p = MatD.toeplitz(nn, nh, t2p^)
        self.KER_FNU2m = MatD.toeplitz(nn, nh, t2m^)
        if self.verbose:
            print("computing muon decay to e+/- kernels")
        # The e+- of a muon of helicity h have the spectrum of its nu_mu-type
        # neutrino (V-A): FNU1(x, h) for a mu+ and FNU1(x, -h) for a mu-. AM3
        # gave the e+- of both helicities 4/3 (1 - x^3), the spectrum of
        # h = +1. So the e+- of h = -1 got 0.40 of the muon energy, not 0.30
        # (AM3 bug).
        var nl = dat.nbins_lep
        var tep = zeros(nl + nh - 1)
        var tem = zeros(nl + nh - 1)
        for m in range(nl + nh - 1):
            var i: Int
            var j: Int
            i, j = MatD.toeplitz_entry(m, nh)
            var r = dat.egrid_had_leptons[i] / dat.egrid_had[j]
            tep[m] = r * FNU1(r, 1.0)
            tem[m] = r * FNU1(r, -1.0)
        self.KER_FMU2Ep = MatD.toeplitz(nl, nh, tep^)
        self.KER_FMU2Em = MatD.toeplitz(nl, nh, tem^)

    def _rates(mut self, dat: AM3Arrays, muons: List[Int]):
        var nh = dat.nbins_had
        var n = len(muons)
        _sized(self.w_ml, n, nh)
        _sized(self.w_pl, n, nh)
        _sized(self.w_mr, n, nh)
        _sized(self.w_pr, n, nh)
        for k in range(n):
            ref MpL = dat.n[muons[k]]
            ref MpR = dat.n[muons[k] + 1]
            ref MmL = dat.n[muons[k] + 2]
            ref MmR = dat.n[muons[k] + 3]
            for i in range(nh):
                var tdec = dat.egrid_had[i] / MASS_MU * TAU_MU
                self.w_ml[k][i] = MmL[i] / tdec
                self.w_pl[k][i] = MpL[i] / tdec
                self.w_mr[k][i] = MmR[i] / tdec
                self.w_pr[k][i] = MpR[i] / tdec

    def Calc_MUtoNU(mut self, dat: AM3Arrays, muons: List[Int]):
        if self.verbose:
            print("computing muon decay to neutrinos terms")
        var n = len(muons)
        var nr = self.KER_FNU1p.rows
        _sized(self.t_nu, 8 * n, nr)
        _sized(self.NUeAGen_mu, n, nr)
        _sized(self.NUeGen_mu, n, nr)
        _sized(self.NUmAGen_mu, n, nr)
        _sized(self.NUmGen_mu, n, nr)
        self._rates(dat, muons)
        var jobs = List[GemvPair]()
        for k in range(n):
            var ml = pd(self.w_ml[k])
            var pl = pd(self.w_pl[k])
            var mr = pd(self.w_mr[k])
            var pr = pd(self.w_pr[k])
            ref t = self.t_nu
            var b = 8 * k
            jobs.append(
                GemvPair(self.KER_FNU2p, ml, pr, 1.0, pd(t[b]), pd(t[b + 3]))
            )
            jobs.append(
                GemvPair(
                    self.KER_FNU2m, mr, pl, 1.0, pd(t[b + 1]), pd(t[b + 2])
                )
            )
            jobs.append(
                GemvPair(
                    self.KER_FNU1m, pl, mr, 1.0, pd(t[b + 4]), pd(t[b + 7])
                )
            )
            jobs.append(
                GemvPair(
                    self.KER_FNU1p, pr, ml, 1.0, pd(t[b + 5]), pd(t[b + 6])
                )
            )
        run_gemv_pairs(jobs, self.nthreads)
        for k in range(n):
            ref t = self.t_nu
            var b = 8 * k
            for i in range(nr):
                self.NUeAGen_mu[k][i] = dat.dlnE * (t[b][i] + t[b + 1][i])
                self.NUeGen_mu[k][i] = dat.dlnE * (t[b + 2][i] + t[b + 3][i])
                self.NUmAGen_mu[k][i] = dat.dlnE * (t[b + 4][i] + t[b + 5][i])
                self.NUmGen_mu[k][i] = dat.dlnE * (t[b + 6][i] + t[b + 7][i])

    def Calc_MUtoE(mut self, dat: AM3Arrays, muons: List[Int]):
        if self.verbose:
            print("computing muon decay to pairs terms")
        var n = len(muons)
        var nr = self.KER_FMU2Ep.rows
        _sized(self.t_e, 4 * n, nr)
        _sized(self.QMU2PGen_mu, n, nr)
        _sized(self.QMU2EGen_mu, n, nr)
        self._rates(dat, muons)
        var jobs = List[GemvPair]()
        for k in range(n):
            ref t = self.t_e
            var b = 4 * k
            jobs.append(
                GemvPair(
                    self.KER_FMU2Em,
                    pd(self.w_pl[k]),
                    pd(self.w_mr[k]),
                    1.0,
                    pd(t[b]),
                    pd(t[b + 3]),
                )
            )
            jobs.append(
                GemvPair(
                    self.KER_FMU2Ep,
                    pd(self.w_pr[k]),
                    pd(self.w_ml[k]),
                    1.0,
                    pd(t[b + 1]),
                    pd(t[b + 2]),
                )
            )
        run_gemv_pairs(jobs, self.nthreads)
        for k in range(n):
            ref t = self.t_e
            var b = 4 * k
            for i in range(nr):
                self.QMU2PGen_mu[k][i] = dat.dlnE * (t[b][i] + t[b + 1][i])
                self.QMU2EGen_mu[k][i] = dat.dlnE * (t[b + 2][i] + t[b + 3][i])


struct CICJob(Movable):
    var H: Pointer[HadIC, MutUntrackedOrigin]
    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]
    var coeff: List[Float64]

    def __init__(
        out self,
        H: Pointer[HadIC, MutUntrackedOrigin],
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
        var coeff: List[Float64],
    ):
        self.H = H
        self.dat = dat
        self.coeff = coeff^


def cic_work(job: Pointer[CICJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref H = job[].H[]
    var k = t
    while k < 3:
        HadIC._emit(
            job[].dat[],
            H.deltaE_C[k],
            H.jmax_C[k],
            H.w_inC[k],
            job[].coeff[k],
            H.tmpC[k],
        )
        k += nt


struct HadIC(Copyable, Movable):
    var nthreads: Int
    var verbose: Bool
    var epsilonPIC: Vec
    var epsilonPiIC: Vec
    var epsilonMuIC: Vec
    var epsilonKIC: Vec  # am3_mojo addition
    var deltaE_p: VecI
    var jmax_p: VecI
    var deltaE_pi: VecI
    var jmax_pi: VecI
    var deltaE_mu: VecI
    var jmax_mu: VecI
    var deltaE_K: VecI
    var jmax_K: VecI
    # charged charm hadrons (am3_mojo addition): the steps of D+-, Ds+-,
    # Lambda_c+ and anti-Lambda_c- (one per mass)
    var epsilonCIC: Vec
    var deltaE_C: List[VecI]
    var jmax_C: List[VecI]
    var w_in: Vec
    var w_inC: List[Vec]
    var tmpC: List[Vec]

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.nthreads = 1
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
        self.epsilonCIC = Vec()
        self.deltaE_C = [VecI(), VecI(), VecI()]
        self.jmax_C = [VecI(), VecI(), VecI()]
        self.w_in = Vec()
        self.w_inC = [Vec(), Vec(), Vec()]
        self.tmpC = [Vec(), Vec(), Vec()]
        self.clear_internal_arrays(dat)
        self.reinitialize_empty_kernels(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.epsilonPIC, dat.nbins_pho)
        resize_zero(self.epsilonPiIC, dat.nbins_pho)
        resize_zero(self.epsilonMuIC, dat.nbins_pho)
        resize_zero(self.epsilonKIC, dat.nbins_pho)
        resize_zero(self.epsilonCIC, dat.nbins_pho)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        self.deltaE_p = zerosi(dat.nbins_pho)
        self.jmax_p = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)
        self.deltaE_pi = zerosi(dat.nbins_pho)
        self.jmax_pi = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)
        self.deltaE_mu = zerosi(dat.nbins_pho)
        self.jmax_mu = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)
        self.deltaE_K = zerosi(dat.nbins_pho)
        self.jmax_K = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)
        for k in range(3):
            self.deltaE_C[k] = zerosi(dat.nbins_pho)
            self.jmax_C[k] = VecI(length=dat.nbins_pho, fill=dat.nbins_pho)

    @staticmethod
    def _steps(dat: AM3Arrays, mass: Float64, mut dE: VecI, mut jmax: VecI):
        for i in range(dat.nbins_had):
            var g = c_exp(Float64(i) * dat.dlnE) / mass
            var d = Int(c_log(g * g) / dat.dlnE)
            dE[i] = d
            var jmax1 = (
                Int(c_log(0.5 * mass / g) / dat.dlnE)
                + dat.ind_min_energy_pho_hadgrid
            )
            jmax[i] = dat.nbins_pho if (jmax1 + d) > dat.nbins_pho else (
                jmax1 + d
            )

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing proton IC rad. kernel (step approx.)")
        HadIC._steps(dat, MASS_P, self.deltaE_p, self.jmax_p)
        if self.verbose:
            print("computing pion IC rad. kernel (step approx.)")
        HadIC._steps(dat, MASS_PI, self.deltaE_pi, self.jmax_pi)
        if self.verbose:
            print("computing muon IC rad. kernel (step approx.)")
        HadIC._steps(dat, MASS_MU, self.deltaE_mu, self.jmax_mu)
        if self.verbose:
            print("computing kaon IC rad. kernel (step approx.)")
        HadIC._steps(dat, MASS_K, self.deltaE_K, self.jmax_K)
        if self.verbose:
            print("computing charm-hadron IC rad. kernels (step approx.)")
        for k in range(3):
            var m = charm_mass(charged_charm_species(k))
            HadIC._steps(dat, m, self.deltaE_C[k], self.jmax_C[k])

    @staticmethod
    def _emit(
        dat: AM3Arrays,
        dE: VecI,
        jmax: VecI,
        part: Vec,
        coeff: Float64,
        mut out_: Vec,
    ):
        """The step approximation puts the photons of hadron bin i at the fixed
        shift dE[i] from their target photon bin. The C++ code loops over every
        pair (j, i) and tests whether bin i reaches j."""
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
        var coeff_p = C_LIGHT * SIGMA_T / (MASS_P / MASS_E) / (MASS_P / MASS_E)
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
        var coeff_pi = (
            C_LIGHT * SIGMA_T / (MASS_PI / MASS_E) / (MASS_PI / MASS_E)
        )
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
        var coeff_mu = (
            C_LIGHT * SIGMA_T / (MASS_MU / MASS_E) / (MASS_MU / MASS_E)
        )
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
        if self.verbose:
            print("computing kaon IC emission")
        var coeff_K = C_LIGHT * SIGMA_T / (MASS_K / MASS_E) / (MASS_K / MASS_E)
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

    def Calc_CIC(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing charm-hadron IC emission")
        resize_zero(self.epsilonCIC, dat.nbins_pho)
        var coeff = List[Float64]()
        for k in range(3):
            var s0 = charged_charm_species(k)
            var m = charm_mass(s0)
            coeff.append(C_LIGHT * SIGMA_T / (m / MASS_E) / (m / MASS_E))
            resize_zero(self.w_inC[k], dat.nbins_had)
            resize_zero(self.tmpC[k], dat.nbins_pho)
            for i in range(dat.nbins_had):
                self.w_inC[k][i] = (
                    dat.n[P_charm + s0][i] + dat.n[P_charm + s0 + 1][i]
                )
        var job = CICJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=dat)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            coeff^,
        )
        parallel_run[cic_work](job, min(self.nthreads, 3))
        for k in range(3):
            for i in range(dat.nbins_pho):
                self.epsilonCIC[i] += self.tmpC[k][i]


# ==============================================================================
# Neutron decay (am3_mojo addition)
# ==============================================================================
# n -> p e- anti-nu_e (100 %, PDG 2026) at the neutron solver's rate
# 1 / (gamma tau_n) (tests/literature_checks/neutron_decay/README.md). Each
# product is boosted exactly to the energy of every hadron bin, also for slow
# neutrons (beta < 1). So the kernels are not Toeplitz. Limitations: no
# recoil in the lepton spectra (the proton's is at most 751 eV), no weak
# magnetism, no radiative corrections (Sirlin's outer correction) or
# radiative-decay photons, a point-charge Fermi function (no relativistic or
# finite-size terms) and no bound decay into hydrogen. AM3's solver keeps the
# two lowest bins of every grid empty. So neutrons below 1.2 GeV are removed
# there. Some electrons of the slowest decaying neutrons fall into the two
# lowest lepton bins and are lost. As for AM3's decays, a step's products
# come from the neutrons at its end. This is exact in a steady state, and
# first order in the time step otherwise.


def neutron_beta_endpoint() -> Float64:
    """Total energy E_0 [GeV] of the electron at the end of its spectrum in the
    neutron's rest frame."""
    var mn = MASS_N
    var mp = MASS_P
    var me = MASS_E
    return ((mn - mp) * (mn + mp) + me * me) / (2.0 * mn)


def neutron_beta_density(T: Float64, Enu: Float64) -> Float64:
    """Unnormalized electron energy distribution in the neutron's rest frame:
    the allowed spectrum F(E) p E Enu^2, with the electron's kinetic energy T,
    E = m_e + T, momentum p and the antineutrino energy
    Enu = E_0 - m_e - T [GeV]. F is the Fermi function of a charge Z = 1
    (F p is finite at p = 0)."""
    var me = MASS_E
    var E = me + T
    var p = sqrt(T * (T + 2.0 * me))
    var c = 2.0 * PI * ALPHA * E
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
    """En[j] is in units of the parent's energy. The grid has n points,
    logarithmic and ascending. A box narrower than the rounding of its ends
    is a point at its middle."""
    var dl = c_log(x[unsafe_offset=1] / x[unsafe_offset=0])
    if not (b - a > 0.0):
        var m = 0.5 * (a + b)
        if m < x[unsafe_offset=0] or m >= x[unsafe_offset=n - 1]:
            return
        var j = min(
            n - 2, max(0, Int(floor(c_log(m / x[unsafe_offset=0]) / dl)))
        )
        while j > 0 and x[unsafe_offset=j] > m:
            j -= 1
        while j < n - 2 and x[unsafe_offset=j + 1] <= m:
            j += 1
        N[unsafe_offset=j] += w
        En[unsafe_offset=j] += w * m
        return
    var jlo = 0
    if a > 0.0:
        jlo = max(0, Int(floor(c_log(a / x[unsafe_offset=0]) / dl)) - 1)
    var jhi = min(n - 2, Int(floor(c_log(b / x[unsafe_offset=0]) / dl)) + 1)
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
    """The products between x[j] and x[j + 1] go to these two grid points
    linearly in x. This keeps their number and energy."""
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
            self.El[j] = MASS_E * dat.egrid_lep[j]  # [GeV] (unit m_e c^2)
        self.Enu = dat.egrid_nu.copy()
        self.T = zeros(N_REST)
        self.pe = zeros(N_REST)
        self.Ev = zeros(N_REST)
        self.w = zeros(N_REST)
        self.Ke = zeros(self.nl * self.nh)
        self.Knu = zeros(self.nn * self.nh)
        # The rest-frame spectrum on the N_REST nodes of the substitution
        # E = m_e + (E_0 - m_e) sin^2(pi u / 2) (as for the kaon decays). Both
        # distances from the ends, T and Ev = E_0 - E, have full precision.
        var gl = GLRules()
        var me = MASS_E
        var L = neutron_beta_endpoint() - me
        var tot = 0.0
        for n in range(N_REST):
            var u = 0.25 * PI * (gl.xo[n] + 1.0)  # pi u / 2
            var s = c_sin(u)
            var c = c_cos(u)
            self.T[n] = L * s * s
            self.Ev[n] = L * c * c
            self.pe[n] = sqrt(self.T[n] * (self.T[n] + 2.0 * me))
            self.w[n] = (
                0.5
                * PI
                * L
                * s
                * c
                * gl.wo[n]
                * neutron_beta_density(self.T[n], self.Ev[n])
            )
            tot += self.w[n]
        for n in range(N_REST):
            self.w[n] /= tot


def _ndec_column(
    ref J: NDecBuild, i: Int, xl: FPtr, xn: FPtr, N: FPtr, En: FPtr
):
    """A product of rest-frame energy E* and momentum p* is emitted
    isotropically. So its lab energy fraction x = E / E_n is uniform on
    [(E* - beta p*) / m_n, (E* + beta p*) / m_n]."""
    var mn = MASS_N
    var me = MASS_E
    var En_n = J.Eh[i]
    var g = En_n / mn
    var beta = sqrt((g - 1.0) * (g + 1.0)) / g
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


def ndec_build_work(
    job: Pointer[NDecBuild, MutUntrackedOrigin], t: Int, nt: Int
):
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
    var verbose: Bool
    var K_e: MatD
    var K_nu: MatD
    var ps0: Vec
    var ps1: Vec
    # sources per ln E [cm^-3 s^-1]
    var q_p: Vec
    var q_e: Vec
    var q_nu: Vec
    var w: Vec  # decays per ln E and second
    var nthreads: Int
    var built_nh: Int

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
        var f = neutron_beta_endpoint() / MASS_N
        self.ps0 = zeros(nh)
        self.ps1 = zeros(nh)
        for i in range(nh):
            var r = dat.egrid_had[i - 1] / dat.egrid_had[i] if i > 0 else c_exp(
                -dat.dlnE
            )
            self.ps1[i] = f / (1.0 - r)
            self.ps0[i] = 1.0 - self.ps1[i]
        self.built_nh = nh

    def Calc_NtoX(mut self, dat: AM3Arrays, rate: Vec):
        """The rates `rate` [1/s] are the neutron solver's 1 / (gamma tau_n)."""
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


# ==============================================================================
# Kaon decays (am3_mojo addition)
# ==============================================================================
# The decays of K+-, K0_S and K0_L are treated like AM3's pion and muon
# decays. Modes: every non-radiative one with a branching ratio above
# 0.1 % (PDG 2026), plus the smaller K0_S semileptonic and K0_L -> pi0 pi0
# modes. Their lab-frame spectra follow from the matrix elements of KDecay
# (tests/literature_checks/kaons/README.md, "Decays").


@inline(.always)
def kaon_mass(s: Int) -> Float64:
    return MASS_K if s <= KM else MASS_K0


@inline(.always)
def kaon_lifetime(s: Int) -> Float64:
    if s <= KM:
        return TAU_K
    return TAU_KS if s == KS else TAU_KL


@inline(.always)
def kaon_name(s: Int) -> String:
    if s == KP:
        return "kaon_plus"
    elif s == KM:
        return "kaon_minus"
    return "kaon_0S" if s == KS else "kaon_0L"


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


@inline(.always)
def product_grid(p: Int) -> Int:
    """0: hadron grid, 1: hadron-anchored lepton grid, 2: neutrino grid."""
    if p <= PR_MUMR:
        return 0
    return 1 if p <= PR_EP else 2


@inline(.always)
def charge_conjugate(p: Int) -> Int:
    """CP: a mu+ of helicity L becomes a mu- of helicity R."""
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
# polynomials there, so this is exact)
comptime N_DALITZ = 24
comptime N_REST = 256

comptime _TWO_BODY = 0
comptime _K3PI = 1
comptime _KL3 = 2


def _sin2_node(
    x: Float64, w: Float64, lo: Float64, hi: Float64
) -> Tuple[Float64, Float64]:
    """The substitution E = lo + (hi - lo) sin^2(pi u / 2), u in [0, 1],
    absorbs square-root edges at both ends."""
    var a = 0.25 * PI * (x + 1.0)  # pi u / 2
    var s = c_sin(a)
    var c = c_cos(a)
    var L = hi - lo
    # from the nearer end, so that E - lo and hi - E keep their precision
    var E = lo + L * s * s if s < c else hi - L * c * c
    return (E, 0.5 * PI * L * s * c * w)


struct GLRules(Copyable, Movable):
    var xi: Vec
    var wi: Vec
    var xo: Vec
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
        return _sin2_node(self.xo[n], self.wo[n], m, Emax)


@inline(.always)
def l3_density(
    M: Float64,
    mpi: Float64,
    ml: Float64,
    Epi: Float64,
    El: Float64,
    Fp: Float64,
    Fm: Float64,
    pol: Bool,
) -> Float64:
    """|M|^2 of P -> P' l+ nu (K_l3, D -> K l nu, ...) at the parent-frame
    energies E_pi of P' and E_l, summed over the l+ spin. With pol, it is the
    part proportional to the longitudinal polarization of the l+ in the parent
    frame. For the hadronic current H = f+ (p_P + p_P') + f- (p_P - p_P'),
    Fp = f+(t) and Fm = f-(t), t = (p_P - p_P')^2.
    """
    var Enu = M - Epi - El
    var t = M * M + mpi * mpi - 2.0 * M * Epi
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
    var pl = sqrt(max(El * El - ml * ml, 1e-300))
    var ppi2 = Epi * Epi - mpi * mpi
    var pnu2 = Enu * Enu
    var ppi_pl = 0.5 * (pnu2 - ppi2 - pl * pl)
    var pnu_pl = 0.5 * (ppi2 - pnu2 - pl * pl)
    var Ks = M * pl / ml
    var pis = (Epi * pl - El * ppi_pl / pl) / ml
    var nus = (Enu * pl - El * pnu_pl / pl) / ml
    var Hs = Fp * (Ks + pis) + Fm * (Ks - pis)
    return ml * (2.0 * nuH * Hs - H2 * nus)


def _emax(M: Float64, mi: Float64, others: Float64) -> Float64:
    return (M * M + mi * mi - others * others) / (2.0 * M)


def _dalitz_line(
    M: Float64, mi: Float64, mj: Float64, mk: Float64, E: Float64
) -> Tuple[Float64, Float64]:
    var P = sqrt(max(E * E - mi * mi, 0.0))
    var mjk2 = M * M + mi * mi - 2.0 * M * E
    var mjk = sqrt(max(mjk2, 1e-300))
    var Ejs = (mjk2 + mj * mj - mk * mk) / (2.0 * mjk)
    var pjs = sqrt(max(Ejs * Ejs - mj * mj, 0.0))
    var lo = ((M - E) * Ejs - P * pjs) / mjk
    var hi = ((M - E) * Ejs + P * pjs) / mjk
    return (lo, hi)


struct KDecayMode(Copyable, Movable):
    """Decay K(M) -> 0 + 1 (+ 2) with product masses m0, m1, m2.

    * two-body: P0 is the longitudinal polarization of product 0 in the kaon
      frame (-1 for the mu+ of K+ -> mu+ nu).
    * K -> 3 pi (products 0, 1: even pions, 2: odd pion): PDG Dalitz-plot
      expansion.
    * K_l3 (products pi, l+, nu): V-A matrix element.
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

    @inline(.always)
    def mass(self, i: Int) -> Float64:
        if i == 0:
            return self.m0
        return self.m1 if i == 1 else self.m2

    def _k3pi(self, E0: Float64, E1: Float64, E2: Float64) -> Float64:
        var M2 = self.M * self.M
        var mp2 = MASS_PI * MASS_PI
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
        """Kaon form factors f+ (quadratic in t) and
        f- = (f0 - f+) (M^2 - m_pi^2) / t (f0 linear)."""
        var M = self.M
        var mpi = self.m0
        var t = M * M + mpi * mpi - 2.0 * M * Epi
        var mp2 = MASS_PI * MASS_PI
        var Fp = 1.0 + self.l1 * t / mp2 + 0.5 * self.l2 * (t / mp2) * (t / mp2)
        var Fm = (
            (self.l0 - self.l1) / mp2 - 0.5 * self.l2 * t / (mp2 * mp2)
        ) * (M * M - mpi * mpi)
        return l3_density(M, mpi, self.m1, Epi, El, Fp, Fm, pol)

    def density(
        self, E0: Float64, E1: Float64, E2: Float64, pol: Bool
    ) -> Float64:
        if self.kind == _KL3:
            return self._kl3(E0, E1, pol)
        return self._k3pi(E0, E1, E2)

    def emax(self, i: Int) -> Float64:
        var j = 1 if i == 0 else 0
        var k = 1 if i == 2 else 2
        return _emax(self.M, self.mass(i), self.mass(j) + self.mass(k))

    def energy_density(
        self, i: Int, E: Float64, pol: Bool, gl: GLRules
    ) -> Float64:
        """Rest-frame energy distribution of product i (normalized to 1), or
        (pol) its longitudinal polarization times that distribution."""
        var j = 1 if i == 0 else 0
        var k = 1 if i == 2 else 2
        var M = self.M
        var lo: Float64
        var hi: Float64
        lo, hi = _dalitz_line(M, self.mass(i), self.mass(j), self.mass(k), E)
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


# branching ratios: PDG 2026 (F. Takahashi et al., Int. J. Mod. Phys. A 41,
# 2630011), the OUR FIT values of the BRANCHING RATIOS of its Particle
# Listings (pdg.lbl.gov/2026/listings): the listing, its printed page and
# PDG's name Gamma_i/Gamma of the mode
comptime BR_K_MUNU = 0.6356  # K+- listing, p. 12, Gamma_2
comptime BR_K_PIPI0 = 0.2067  # K+- listing, p. 16, Gamma_10
comptime BR_K_3PI = 0.05583  # pi+ pi+ pi-: K+- listing, p. 17, Gamma_12
comptime BR_K_E3 = 0.0507  # K+- listing, p. 12, Gamma_3
comptime BR_K_MU3 = 0.03352  # K+- listing, p. 13, Gamma_4
comptime BR_K_PIPI0PI0 = 0.01760  # K+- listing, p. 17, Gamma_11
comptime BR_KS_PIPI = 0.6920  # K0_S listing, p. 4, Gamma_2
comptime BR_KS_PI0PI0 = 0.3069  # K0_S listing, p. 4, Gamma_1
comptime BR_KS_E3 = 7.14e-4  # both charge states: K0_S listing, p. 7, Gamma_9
comptime BR_KS_MU3 = 4.56e-4  # K0_S listing, p. 7, Gamma_10
comptime BR_KL_E3 = 0.4055  # K0_L listing, p. 6, Gamma_1
comptime BR_KL_MU3 = 0.2704  # K0_L listing, p. 6, Gamma_2
comptime BR_KL_3PI0 = 0.1952  # K0_L listing, p. 8, Gamma_6
comptime BR_KL_PIPIPI0 = 0.1254  # K0_L listing, p. 9, Gamma_7
comptime BR_KL_PIPI = 1.967e-3  # K0_L listing, p. 9, Gamma_8
comptime BR_KL_PI0PI0 = 8.64e-4  # K0_L listing, p. 11, Gamma_9
# K_L charge asymmetries [G(pi- l+ nu) - G(pi+ l- nu-bar)] / sum: PDG 2026,
# K0_L listing, A_L(e) = 0.334 +- 0.007 % (p. 35) and
# A_L(mu) = 0.304 +- 0.025 % (p. 34), OUR AVERAGE
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
    of PDG 2026. Form factors: PDG's fits assuming mu-e universality. The K0
    fit has a quadratic f+ and a linear f0. The K+- fit has a linear f+ and
    f0, because PDG has no K+- quadratic fit with universality."""
    var mK = MASS_K
    var mK0 = MASS_K0
    var mpi = MASS_PI
    var mpi0 = MASS_PI0
    var md: List[KDecayMode] = [
        KDecayMode(_TWO_BODY, mK, MASS_MU, 0.0, 0.0, P0=-1.0),
        KDecayMode(_TWO_BODY, mK, mpi, mpi0, 0.0),
        KDecayMode(_TWO_BODY, mK0, mpi, mpi, 0.0),
        KDecayMode(_TWO_BODY, mK0, mpi0, mpi0, 0.0),
        # K+- l3: PDG 2026, K+- listing, OUR FIT assuming mu-e universality,
        # linear f+: lambda_+ = (2.959 +- 0.025) x 10^-2 (p. 38) and
        # lambda_0 = (1.76 +- 0.25) x 10^-2 (p. 39)
        KDecayMode(_KL3, mK, mpi0, MASS_E, 0.0, l1=0.02959, l2=0.0, l0=0.0176),
        KDecayMode(_KL3, mK, mpi0, MASS_MU, 0.0, l1=0.02959, l2=0.0, l0=0.0176),
        # K0 l3: PDG 2026, K0_L listing, OUR FIT assuming mu-e universality,
        # quadratic f+: lambda'_+ = (2.40 +- 0.12) x 10^-2 (p. 25),
        # lambda''_+ = (0.20 +- 0.05) x 10^-2 (p. 26),
        # lambda_0 = (1.16 +- 0.09) x 10^-2 (p. 27)
        KDecayMode(
            _KL3, mK0, mpi, MASS_E, 0.0, l1=0.0240, l2=0.0020, l0=0.0116
        ),
        KDecayMode(
            _KL3, mK0, mpi, MASS_MU, 0.0, l1=0.0240, l2=0.0020, l0=0.0116
        ),
        # Dalitz-plot coefficients, PDG 2026. K+- -> pi+- pi+ pi-, K+- listing
        # (BATLEY 07B): g = -0.21134 +- 0.00017 and
        # h = (1.848 +- 0.040) x 10^-2 (p. 33), k = (-4.63 +- 0.14) x 10^-3
        # (p. 34)
        KDecayMode(_K3PI, mK, mpi, mpi, mpi, g=-0.21134, h=0.01848, k=-0.00463),
        # K+- -> pi+- pi0 pi0, K+- listing, OUR AVERAGE: g = 0.626 +- 0.007
        # (p. 34), h = 0.052 +- 0.008 and k = 0.0054 +- 0.0035 (p. 35)
        KDecayMode(_K3PI, mK, mpi0, mpi0, mpi, g=0.626, h=0.052, k=0.0054),
        # K0_L -> pi+ pi- pi0, K0_L listing, OUR AVERAGE: g = 0.678 +- 0.008
        # (p. 21), h = 0.076 +- 0.006 and k = 0.0099 +- 0.0015 (p. 22)
        KDecayMode(_K3PI, mK0, mpi, mpi, mpi0, g=0.678, h=0.076, k=0.0099),
        # K0_L -> 3 pi0, K0_L listing, p. 22 (ABOUZAID 08A):
        # h = (0.59 +- 0.20 +- 1.16) x 10^-3
        KDecayMode(_K3PI, mK0, mpi0, mpi0, mpi0, h=0.00059, sym3=True),
    ]
    for i in range(len(md)):
        md[i].normalise(gl)
    return md^


@fieldwise_init
struct KTerm(Copyable, Movable):
    """hel: lab helicity of the mu+ whose energy distribution the term takes
    (+1 R, -1 L, 0 both). A mu- term takes that of the mu+ with the opposite
    helicity (CP)."""

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
    """En[q] is in units of the kaon energy. The products are uniform in x,
    times wg + wG htilde(x). Here htilde = (E M x - m^2) / (p M x) is the
    lab helicity of a muon with longitudinal polarization 1 in the kaon
    frame."""
    var K = len(xc) - 1
    var p = sqrt(max(E * E - m * m, 0.0))
    if p <= 0.0:
        return
    var a = (E - p) / M
    var b = (E + p) / M
    var dens = M / (2.0 * p)
    var qlo = max(0, Int(floor(-c_log(b) / dlnE)) - 1)
    var qhi = K - 1
    if a > 0.0:
        qhi = min(K - 1, Int(floor(-c_log(a) / dlnE)) + 1)
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


def _bins_to_table(N: Vec, En: Vec, off: Int, xc: Vec, dlnE: Float64) -> Vec:
    """Products per decay per unit ln x. The products between x_(q+1) and x_q
    are shared linearly in x, which keeps their number and energy."""
    var K = len(xc) - 1
    var C = zeros(K + 1)
    for q in range(K):
        var n = N[off + q]
        var e = En[off + q]
        var dx = xc[q] - xc[q + 1]
        C[q] += (e - xc[q + 1] * n) / dx / dlnE
        C[q + 1] += (xc[q] * n - e) / dx / dlnE
    return C^


def kaon_decay_tables(dlnE: Float64, K: Int) -> List[Vec]:
    """Decay kernels T[s * N_PROD + p][k], k = 0 .. K: products p per decay of
    kaon species s at the energy fraction x_k = exp(-k dlnE), per unit ln x.

    The kernels are exact."""
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
                var E = _emax(md.M, m, mo)
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
            tabs[s * N_PROD + p] = _bins_to_table(N, En, p * K, xc, dlnE)
    return tabs^


struct KDecJob(Movable):
    var D: Pointer[KaonDecay, MutUntrackedOrigin]
    var dlnE: Float64

    def __init__(
        out self, D: Pointer[KaonDecay, MutUntrackedOrigin], dlnE: Float64
    ):
        self.D = D
        self.dlnE = dlnE


comptime DECAY_ROWS = 32  # output rows per task of the kaon and charm decays


def _row_tasks(mut tasks: List[Int], p: Int, n: Int):
    """Each task sums its rows over the species, so that the threads share the
    work evenly."""
    var r0 = 0
    while r0 < n:
        tasks.append(p)
        tasks.append(r0)
        tasks.append(min(r0 + DECAY_ROWS, n))
        r0 += DECAY_ROWS


def _restricted_toeplitz_add[
    two: Bool
](
    ref A: MatD,
    x0: FPtr,
    x1: FPtr,
    j0: Int,
    j1: Int,
    r0: Int,
    r1: Int,
    alpha: Float64,
    tmp0: FPtr,
    tmp1: FPtr,
    res0: FPtr,
    res1: FPtr,
):
    """A must be Toeplitz, and x0 and x1 zero outside the columns [j0, j1)."""
    if toeplitz_window_zero(A, j0, j1, r0, r1):
        return
    toeplitz_gemv[two](A, x0, x1, j0, j1, r0, r1, alpha, tmp0, tmp1)
    for i in range(r0, r1):
        res0[unsafe_offset=i] += tmp0[unsafe_offset=i]
        comptime if two:
            res1[unsafe_offset=i] += tmp1[unsafe_offset=i]


def _species_products(
    ref A: MatD,
    ref w: List[Vec],
    active: List[Bool],
    jlo: List[Int],
    jhi: List[Int],
    k0: Int,
    k1: Int,
    ref tmp: List[Vec],
    ref res: List[Vec],
    p0: Int,
    p1: Int,
    r0: Int,
    r1: Int,
    alpha: Float64,
):
    """A species of both origins takes one pass over A, over both origins' bins
    (the others add zeros)."""
    var a0 = active[k0]
    var a1 = active[k1]
    if a0 and a1:
        _restricted_toeplitz_add[True](
            A,
            pd(w[k0]),
            pd(w[k1]),
            min(jlo[k0], jlo[k1]),
            max(jhi[k0], jhi[k1]),
            r0,
            r1,
            alpha,
            pd(tmp[p0]),
            pd(tmp[p1]),
            pd(res[p0]),
            pd(res[p1]),
        )
    elif a0 or a1:
        # (of one origin: the second vector's pointers are not read)
        var k = k0 if a0 else k1
        var p = p0 if a0 else p1
        var x = pd(w[k])
        var t = pd(tmp[p])
        var y = pd(res[p])
        _restricted_toeplitz_add[False](
            A, x, x, jlo[k], jhi[k], r0, r1, alpha, t, t, y, y
        )


def _decay_task(
    ref K: List[MatD],
    ref has: List[Bool],
    ref w: List[Vec],
    ref active: List[Bool],
    ref jlo: List[Int],
    ref jhi: List[Int],
    ref tmp: List[Vec],
    ref out: List[Vec],
    ref tasks: List[Int],
    q: Int,
    dlnE: Float64,
):
    """The rows are summed over the species in a fixed order."""
    var n_prod = len(out) // 2
    var n_sp = len(w) // 2
    var p = tasks[3 * q]
    for s in range(n_sp):
        if has[s * n_prod + p]:
            _species_products(
                K[s * n_prod + p],
                w,
                active,
                jlo,
                jhi,
                s,
                n_sp + s,
                tmp,
                out,
                p,
                n_prod + p,
                tasks[3 * q + 1],
                tasks[3 * q + 2],
                dlnE,
            )


def _decay_kernels[
    rows: def(AM3Arrays, Int) thin -> Int,
    energy: def(AM3Arrays, Int, Int) thin -> Float64,
](
    dat: AM3Arrays,
    T: List[Vec],
    n_prod: Int,
    K: Int,
    mut kernels: List[MatD],
    mut has: List[Bool],
):
    var nh = dat.nbins_had
    for s in range(len(kernels) // n_prod):
        for p in range(n_prod):
            var nr = rows(dat, p)
            var nt = nr + nh - 1
            var t = zeros(nt)
            var used = False
            for m in range(nt):
                var i: Int
                var j: Int
                i, j = MatD.toeplitz_entry(m, nh)
                var k = Int(
                    floor(
                        -c_log(energy(dat, p, i) / dat.egrid_had[j]) / dat.dlnE
                        + 0.5
                    )
                )
                if k >= 0 and k <= K:
                    t[m] = T[s * n_prod + p][k]
                    used = used or t[m] != 0.0
            has[s * n_prod + p] = used
            kernels[s * n_prod + p] = MatD.toeplitz(
                nr, nh, t^
            ) if used else MatD()


def _decay_rates[
    mass: def(Int) thin -> Float64, lifetime: def(Int) thin -> Float64
](
    dat: AM3Arrays,
    base0: Int,
    base1: Int,
    on0: Bool,
    on1: Bool,
    mut w: List[Vec],
    mut active: List[Bool],
    mut jlo: List[Int],
    mut jhi: List[Int],
) -> Bool:
    var nh = dat.nbins_had
    var n_sp = len(w) // 2
    var any_active = False
    for o in range(2):
        var on = on0 if o == 0 else on1
        var base = base0 if o == 0 else base1
        for s in range(n_sp):
            var k = o * n_sp + s
            resize_zero(w[k], nh)
            active[k] = False
            jlo[k] = nh
            jhi[k] = 0
            if not on:
                continue
            var r = mass(s) / lifetime(s)
            for i in range(nh):
                w[k][i] = dat.n[base + s][i] * r / dat.egrid_had[i]
                if w[k][i] != 0.0:
                    active[k] = True
                    jlo[k] = min(jlo[k], i)
                    jhi[k] = i + 1
            any_active = any_active or active[k]
    return any_active


def kdec_work(job: Pointer[KDecJob, MutUntrackedOrigin], q: Int):
    ref D = job[].D[]
    _decay_task(
        D.K,
        D.has,
        D.w,
        D.active,
        D.jlo,
        D.jhi,
        D.tmp,
        D.out,
        D.tasks,
        q,
        job[].dlnE,
    )


struct KaonDecay(Copyable, Movable):
    """T(E_i / E_j) depends only on i - j. So each (species, product) kernel,
    summed over the decay modes, is a Toeplitz matrix stored by its diagonal
    values."""

    var verbose: Bool
    var K: List[MatD]
    var has: List[Bool]
    # per origin o (0: photo-hadronic, 1: pp)
    var out: List[Vec]
    var w: List[Vec]
    var active: List[Bool]
    var jlo: List[Int]
    var jhi: List[Int]
    var tmp: List[Vec]
    var tasks: List[Int]
    var nthreads: Int
    var built_nh: Int

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.K = List[MatD]()
        self.has = List[Bool]()
        for _ in range(N_KAON * N_PROD):
            self.K.append(MatD())
            self.has.append(False)
        self.out = List[Vec]()
        self.tmp = List[Vec]()
        for _ in range(2 * N_PROD):
            self.out.append(Vec())
            self.tmp.append(Vec())
        self.w = List[Vec]()
        self.active = List[Bool]()
        self.jlo = List[Int]()
        self.jhi = List[Int]()
        for _ in range(2 * N_KAON):
            self.w.append(Vec())
            self.active.append(False)
            self.jlo.append(0)
            self.jhi.append(0)
        self.tasks = List[Int]()
        self.nthreads = 1
        self.built_nh = -1
        self.clear_internal_arrays(dat)

    def free_kernels(mut self):
        for k in range(len(self.K)):
            self.K[k] = MatD()
            self.has[k] = False
        self.built_nh = -1

    def ready(self, dat: AM3Arrays) -> Bool:
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
        for o in range(2):
            for p in range(N_PROD):
                resize_zero(self.out[o * N_PROD + p], KaonDecay._rows(dat, p))

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing kaon decay kernels")
        var nh = dat.nbins_had
        # The grids are logarithmic with step dlnE, so E_i / E_j = x_k with
        # integer k.
        var K = 0
        for p in range(N_PROD):
            var x = KaonDecay._energy(dat, p, 0) / dat.egrid_had[nh - 1]
            K = max(K, Int(floor(-c_log(x) / dat.dlnE + 0.5)))
        var T = kaon_decay_tables(dat.dlnE, K)
        _decay_kernels[KaonDecay._rows, KaonDecay._energy](
            dat, T, N_PROD, K, self.K, self.has
        )
        self.built_nh = nh

    def Calc_KtoX(
        mut self, dat: AM3Arrays, base0: Int, base1: Int, on0: Bool, on1: Bool
    ):
        if self.verbose:
            print("computing kaon decay terms")
        self.clear_internal_arrays(dat)
        if not self.ready(dat):
            return
        if not _decay_rates[kaon_mass, kaon_lifetime](
            dat,
            base0,
            base1,
            on0,
            on1,
            self.w,
            self.active,
            self.jlo,
            self.jhi,
        ):
            return
        self.tasks.clear()
        for k in range(2 * N_PROD):
            resize_zero(self.tmp[k], len(self.out[k]))
            if k < N_PROD:
                _row_tasks(self.tasks, k, len(self.out[k]))
        var job = KDecJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](), dat.dlnE
        )
        parallel_tasks[kdec_work](job, len(self.tasks) // 3, self.nthreads)


# ==============================================================================
# Charm-hadron decays (am3_mojo addition)
# ==============================================================================
# Modes (`charm_decay_modes`): PDG 2026, every mode above 0.5 % and every
# exclusive semileptonic mode above 0.1 %
# (tests/literature_checks/charm/README.md, "Decays"). The short-lived
# particles of the modes (tau, eta, eta', omega, phi, rho, K*, f0, a1,
# Lambda, Sigma, Xi) decay at once. So their products count as products of
# the charm decay. The antiparticles decay as the CP conjugates.


@inline(.always)
def charm_mass(s: Int) -> Float64:
    if s <= CDM:
        return MASS_D
    if s <= CD0B:
        return MASS_D0
    return MASS_DS if s <= CDSM else MASS_LC


@inline(.always)
def charm_lifetime(s: Int) -> Float64:
    if s <= CDM:
        return TAU_D
    if s <= CD0B:
        return TAU_D0
    return TAU_DS if s <= CDSM else TAU_LC


@inline(.always)
def charm_charged(s: Int) -> Bool:
    return s != CD0 and s != CD0B


def charm_name(s: Int) -> String:
    if s == CDP:
        return "D_plus"
    elif s == CDM:
        return "D_minus"
    elif s == CD0:
        return "D0"
    elif s == CD0B:
        return "anti_D0"
    elif s == CDSP:
        return "Ds_plus"
    elif s == CDSM:
        return "Ds_minus"
    return "Lambda_c_plus" if s == CLC else "anti_Lambda_c_minus"


# products of the charm decays, in the order of the coefficient blocks of
# simulation/coeffs.mojo
comptime CP_PIP = 0
comptime CP_PIM = 1
comptime CP_PI0 = 2
comptime CP_MUPL = 3  # mu+, left-handed (lab helicity)
comptime CP_MUPR = 4
comptime CP_MUML = 5
comptime CP_MUMR = 6
comptime CP_KP = 7
comptime CP_KM = 8
comptime CP_KS = 9
comptime CP_KL = 10
comptime CP_P = 11  # protons (and the antiprotons of the antiparticles)
comptime CP_N = 12  # neutrons (and antineutrons)
comptime CP_EM = 13  # e- (hadron-anchored lepton grid)
comptime CP_EP = 14
comptime CP_NUE = 15  # neutrino grid
comptime CP_NUEB = 16
comptime CP_NUM = 17
comptime CP_NUMB = 18
comptime CP_NUT = 19
comptime CP_NUTB = 20
comptime CP_GAM = 21  # photons (hadron-anchored photon grid)
comptime N_CPROD = 22


@inline(.always)
def charm_product_grid(p: Int) -> Int:
    """0: hadron grid, 1: hadron-anchored lepton grid, 2: neutrino grid,
    3: hadron-anchored photon grid."""
    if p <= CP_N:
        return 0
    if p <= CP_EP:
        return 1
    return 2 if p <= CP_NUTB else 3


@inline(.always)
def charm_product_conjugate(p: Int) -> Int:
    """Product of the CP-conjugate decay (a mu+ of helicity L becomes a mu-
    of helicity R, and antinucleons count as nucleons)."""
    var cc: Array[Int, N_CPROD] = [
        CP_PIM,
        CP_PIP,
        CP_PI0,
        CP_MUMR,
        CP_MUML,
        CP_MUPR,
        CP_MUPL,
        CP_KM,
        CP_KP,
        CP_KS,
        CP_KL,
        CP_P,
        CP_N,
        CP_EP,
        CP_EM,
        CP_NUEB,
        CP_NUE,
        CP_NUMB,
        CP_NUM,
        CP_NUTB,
        CP_NUT,
        CP_GAM,
    ]
    return cc[p]


# Particles of the decay network: the products above (a muon with its
# polarization, K0 and anti-K0 as half K0_S, half K0_L) and the short-lived
# particles that decay at once (X_TAUP and later).
comptime X_PIP = 0
comptime X_PIM = 1
comptime X_PI0 = 2
comptime X_MUP = 3
comptime X_MUM = 4
comptime X_KP = 5
comptime X_KM = 6
comptime X_K0 = 7  # K0 or anti-K0
comptime X_P = 8
comptime X_N = 9
comptime X_EP = 10
comptime X_EM = 11
comptime X_NUE = 12
comptime X_NUEB = 13
comptime X_NUM = 14
comptime X_NUMB = 15
comptime X_NUT = 16
comptime X_NUTB = 17
comptime X_GAM = 18
comptime N_XFIN = 19
comptime X_TAUP = 19  # tau+ (decays in the frame of its parent)
comptime X_ETA = 20
comptime X_ETAP = 21  # eta'(958)
comptime X_OMEGA = 22
comptime X_PHI = 23
comptime X_RHOP = 24
comptime X_RHO0 = 25
comptime X_RHOM = 26
comptime X_KSTP = 27  # K*(892)+
comptime X_KSTM = 28
comptime X_KST0 = 29  # K*(892)0
comptime X_KST0B = 30  # anti-K*(892)0
comptime X_F0 = 31  # f0(980)
comptime X_A1P = 32  # a1(1260)+
comptime X_LAM = 33
comptime X_SIGP = 34
comptime X_SIG0 = 35
comptime X_SIGM = 36
comptime X_XI0 = 37
comptime X_XIM = 38
comptime N_X = 39


def x_mass(x: Int) -> Float64:
    var m: Array[Float64, N_X] = [
        MASS_PI,
        MASS_PI,
        MASS_PI0,
        MASS_MU,
        MASS_MU,
        MASS_K,
        MASS_K,
        MASS_K0,
        MASS_P,
        MASS_N,
        MASS_E,
        MASS_E,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        MASS_TAU,
        MASS_ETA,
        MASS_ETAP,
        MASS_OMEGA,
        MASS_PHI,
        MASS_RHO,
        MASS_RHO0,
        MASS_RHO,
        MASS_KSTAR,
        MASS_KSTAR,
        MASS_KSTAR0,
        MASS_KSTAR0,
        MASS_F0,
        MASS_A1,
        MASS_LAMBDA,
        MASS_SIGMA_PLUS,
        MASS_SIGMA0,
        MASS_SIGMA_MINUS,
        MASS_XI0,
        MASS_XI_MINUS,
    ]
    return m[x]


comptime XK_TWO = 0  # two-body
comptime XK_PS = 1  # n-body phase space
comptime XK_PL3 = 2  # P -> P' l+ nu
comptime XK_VL3 = 3  # P -> V l+ nu
comptime XK_BL3 = 4  # Lambda_c+ -> B l+ nu
comptime XK_MICHEL = 5  # tau+ -> l+ nu_l anti-nu_tau


struct XMode(Copyable, Movable):
    """Decay mode: branching ratio br into the network particles prod (X_*),
    with the kinematics `kind`:
    * TWO: product 0 has the longitudinal polarization P0 in the parent frame
      (for a lepton). It also has the asymmetry a0 about the spin of a
      polarized parent (the tau): (1 + a0 P cos theta) / 2. Product 1 has -a0.
    * PS: in a semileptonic mode (P0 = 1, _sl_ps), the l+ has the
      longitudinal polarization p / E (_nodes).
    * PL3 (products P', l+, nu): the V-A matrix element (l3_density) with
      the pole form factors.
    * VL3 (products V, l+, nu): the V-A matrix element (vl3_density).
    * BL3 (products B, l+, nu): Lambda_c+ -> Lambda (bl = 0) or n (bl = 1)
      with the lattice form factors of physics/lambda_c_semileptonic.mojo.
    * MICHEL (products l+, nu_l, anti-nu_tau): the V-A spectra of a
      polarized tau+ with massless products (michel_nodes)."""

    var kind: Int
    var br: Float64
    var prod: List[Int]
    var P0: Float64
    var a0: Float64
    var mp: Float64
    var alpha: Float64
    var m0: Float64
    var rV: Float64
    var r2: Float64
    var mV: Float64
    var mA: Float64
    var bl: Int

    def __init__(out self, kind: Int, br: Float64, var prod: List[Int]):
        self.kind = kind
        self.br = br
        self.prod = prod^
        self.P0 = 0.0
        self.a0 = 0.0
        self.mp = 0.0
        self.alpha = 0.0
        self.m0 = 0.0
        self.rV = 0.0
        self.r2 = 0.0
        self.mV = 0.0
        self.mA = 0.0
        self.bl = 0


def x_two(
    br: Float64, a: Int, b: Int, P0: Float64 = 0.0, a0: Float64 = 0.0
) -> XMode:
    var m = XMode(XK_TWO, br, [a, b])
    m.P0 = P0
    m.a0 = a0
    return m^


def x_ps(br: Float64, var prod: List[Int]) -> XMode:
    if len(prod) == 2:
        return x_two(br, prod[0], prod[1])
    return XMode(XK_PS, br, prod^)


def x_pl3(
    br: Float64,
    h: Int,
    l: Int,
    nu: Int,
    mp: Float64,
    alpha: Float64,
    m0: Float64,
) -> XMode:
    var m = XMode(XK_PL3, br, [h, l, nu])
    m.mp = mp
    m.alpha = alpha
    m.m0 = m0
    return m^


def x_vl3(
    br: Float64,
    v: Int,
    l: Int,
    nu: Int,
    rV: Float64,
    r2: Float64,
    mV: Float64,
    mA: Float64,
    m0: Float64,
) -> XMode:
    var m = XMode(XK_VL3, br, [v, l, nu])
    m.rV = rV
    m.r2 = r2
    m.mV = mV
    m.mA = mA
    m.m0 = m0
    return m^


def x_bl3(br: Float64, b: Int, l: Int, nu: Int, bl: Int) -> XMode:
    var m = XMode(XK_BL3, br, [b, l, nu])
    m.bl = bl
    return m^


def x_michel(br: Float64, l: Int, nul: Int, nut: Int) -> XMode:
    return XMode(XK_MICHEL, br, [l, nul, nut])


comptime V4 = SIMD[DType.float64, 4]


@inline(.always)
def _mdot(a: V4, b: V4) -> Float64:
    """Minkowski product a.b (metric +, -, -, -)."""
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


@inline(.always)
def _det3(
    a0: Float64,
    a1: Float64,
    a2: Float64,
    b0: Float64,
    b1: Float64,
    b2: Float64,
    c0: Float64,
    c1: Float64,
    c2: Float64,
) -> Float64:
    return (
        a0 * (b1 * c2 - b2 * c1)
        - a1 * (b0 * c2 - b2 * c0)
        + a2 * (b0 * c1 - b1 * c0)
    )


def _eps_low(a: V4, b: V4, c: V4) -> V4:
    """e_mu = eps_{mu nu rho sigma} a^nu b^rho c^sigma, eps_{0123} = +1."""
    return V4(
        _det3(a[1], a[2], a[3], b[1], b[2], b[3], c[1], c[2], c[3]),
        -_det3(a[0], a[2], a[3], b[0], b[2], b[3], c[0], c[2], c[3]),
        _det3(a[0], a[1], a[3], b[0], b[1], b[3], c[0], c[1], c[3]),
        -_det3(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]),
    )


@inline(.always)
def _raise(e: V4) -> V4:
    return V4(e[0], -e[1], -e[2], -e[3])


@inline(.always)
def _eps4(a: V4, b: V4, c: V4, d: V4) -> Float64:
    var e = _eps_low(b, c, d)
    return a[0] * e[0] + a[1] * e[1] + a[2] * e[2] + a[3] * e[3]


def vl3_density(
    md: XMode,
    M: Float64,
    mV: Float64,
    ml: Float64,
    EV: Float64,
    El: Float64,
    pol: Bool,
) -> Float64:
    """|M|^2 of P -> V l+ nu at the parent-frame energies E_V and E_l, summed
    over the V polarizations and the l+ spin. With pol, it is the part
    proportional to the l+ longitudinal polarization in the parent frame. It
    uses the hadronic current of the form factors of `md` (XMode, VL3),
    contracted with the V-A lepton tensor of P = p_l (pol: P = m_l s, the
    rule |M|^2(s) = 1/2 |M|^2(p_l -> p_l + m_l s) for a massless neutrino)."""
    var Enu = M - EV - El
    var pV = sqrt(max(EV * EV - mV * mV, 0.0))
    var pl = sqrt(max(El * El - ml * ml, 0.0))
    if pV <= 0.0 or pl <= 0.0 or Enu <= 0.0:
        return 0.0
    var c = (Enu * Enu - pV * pV - pl * pl) / (2.0 * pV * pl)
    c = min(1.0, max(-1.0, c))
    var s = sqrt(max(1.0 - c * c, 0.0))
    var p = V4(M, 0.0, 0.0, 0.0)
    var k = V4(EV, 0.0, 0.0, pV)
    var l = V4(El, pl * s, 0.0, pl * c)
    var nu = p - k - l
    var q = p - k
    var t = _mdot(q, q)
    var P: V4
    if pol:
        P = V4(pl, El * s, 0.0, El * c)
    else:
        P = l
    var Vt = md.rV / (1.0 - t / (md.mV * md.mV))
    var A1 = 1.0 / (1.0 - t / (md.mA * md.mA))
    var A2 = md.r2 / (1.0 - t / (md.mA * md.mA))
    var A00 = ((M + mV) - (M - mV) * md.r2) / (2.0 * mV)
    var A0 = A00 / (1.0 - t / (md.m0 * md.m0))
    var A3 = ((M + mV) * A1 - (M - mV) * A2) / (2.0 * mV)
    var cv = 2.0 * Vt / (M + mV)
    var pk = p + k
    var res = 0.0
    for pi in range(3):
        var e: V4
        if pi == 0:
            e = V4(0.0, 1.0, 0.0, 0.0)
        elif pi == 1:
            e = V4(0.0, 0.0, 1.0, 0.0)
        else:
            e = V4(pV / mV, 0.0, 0.0, EV / mV)
        var Jr = cv * _raise(_eps_low(e, p, k))
        var eq = _mdot(e, q)
        var Ji = -(
            (M + mV) * A1 * e
            - (A2 / (M + mV) * eq) * pk
            - (2.0 * mV / t * (A3 - A0) * eq) * q
        )
        res += 8.0 * (
            2.0 * (_mdot(nu, Jr) * _mdot(P, Jr) + _mdot(nu, Ji) * _mdot(P, Ji))
            - _mdot(nu, P) * (_mdot(Jr, Jr) + _mdot(Ji, Ji))
        ) + 16.0 * _eps4(Jr, Ji, nu, P)
    return res


def pl3_density(
    md: XMode,
    M: Float64,
    mh: Float64,
    ml: Float64,
    Eh: Float64,
    El: Float64,
    pol: Bool,
) -> Float64:
    var t = M * M + mh * mh - 2.0 * M * Eh
    var Fp = 1.0 / (
        (1.0 - t / (md.mp * md.mp)) * (1.0 - md.alpha * t / (md.mp * md.mp))
    )
    var F0 = 1.0 / (1.0 - t / (md.m0 * md.m0))
    var Fm = (F0 - Fp) * (M * M - mh * mh) / t
    return l3_density(M, mh, ml, Eh, El, Fp, Fm, pol)


def _dalitz_density(
    md: XMode, M: Float64, E0: Float64, E1: Float64, pol: Bool
) -> Float64:
    var mh = x_mass(md.prod[0])
    var ml = x_mass(md.prod[1])
    if md.kind == XK_PL3:
        return pl3_density(md, M, mh, ml, E0, E1, pol)
    if md.kind == XK_VL3:
        return vl3_density(md, M, mh, ml, E0, E1, pol)
    var ff = (
        BL3FormFactors.lambdac_lambda() if md.bl
        == 0 else BL3FormFactors.lambdac_neutron()
    )
    return bl3_density(M, mh, ml, E0, E1, pol, ff)


def _phase_space(m: Float64, masses: List[Float64], gl: XRules) -> Float64:
    """Volume (up to a constant) of the phase space of a system of mass m
    decaying into particles of the given masses."""
    var n = len(masses)
    if n == 1:
        return 1.0
    if n == 2:
        var a = masses[0]
        var b = masses[1]
        var m2 = m * m
        var lam = (m2 - (a + b) * (a + b)) * (m2 - (a - b) * (a - b))
        return sqrt(max(lam, 0.0)) / m2
    var rest = List[Float64]()
    var lo = 0.0
    for j in range(1, n):
        rest.append(masses[j])
        lo += masses[j]
    var hi = m - masses[0]
    if hi <= lo:
        return 0.0
    var two = List[Float64]()
    two.append(masses[0])
    two.append(0.0)
    var s = 0.0
    for k in range(N_PSG):
        var mr: Float64
        var w: Float64
        mr, w = gl.ps_node(k, lo, hi)
        two[1] = mr
        s += (
            w * 2.0 * mr * _phase_space(m, two, gl) * _phase_space(mr, rest, gl)
        )
    return s


comptime N_CREST = 128
comptime N_PSG = 16
comptime N_BOX = 16
comptime N_MICHEL = 32


struct XRules(Copyable, Movable):
    var xi: Vec
    var wi: Vec
    var xo: Vec
    var wo: Vec
    var xp: Vec
    var wp: Vec
    var xb: Vec
    var wb: Vec
    var xm: Vec
    var wm: Vec

    def __init__(out self):
        self.xi = Vec()
        self.wi = Vec()
        self.xo = Vec()
        self.wo = Vec()
        self.xp = Vec()
        self.wp = Vec()
        self.xb = Vec()
        self.wb = Vec()
        self.xm = Vec()
        self.wm = Vec()
        gauss_legendre(N_DALITZ, self.xi, self.wi)
        gauss_legendre(N_CREST, self.xo, self.wo)
        gauss_legendre(N_PSG, self.xp, self.wp)
        gauss_legendre(N_BOX, self.xb, self.wb)
        gauss_legendre(N_MICHEL, self.xm, self.wm)

    def rest_node(
        self, n: Int, lo: Float64, hi: Float64
    ) -> Tuple[Float64, Float64]:
        return _sin2_node(self.xo[n], self.wo[n], lo, hi)

    def ps_node(
        self, n: Int, lo: Float64, hi: Float64
    ) -> Tuple[Float64, Float64]:
        return _sin2_node(self.xp[n], self.wp[n], lo, hi)


struct XNodes(Movable):
    var E: Vec
    var w: Vec
    var G: Vec
    var a: Vec

    def __init__(out self):
        self.E = Vec()
        self.w = Vec()
        self.G = Vec()
        self.a = Vec()

    def add(mut self, E: Float64, w: Float64, G: Float64, a: Float64):
        self.E.append(E)
        self.w.append(w)
        self.G.append(G)
        self.a.append(a)


def _three_body_line(
    md: XMode, M: Float64, i: Int, E: Float64, pol: Bool, gl: XRules
) -> Float64:
    var j = 1 if i == 0 else 0
    var k = 1 if i == 2 else 2
    var mi = x_mass(md.prod[i])
    var mj = x_mass(md.prod[j])
    var mk = x_mass(md.prod[k])
    var lo: Float64
    var hi: Float64
    lo, hi = _dalitz_line(M, mi, mj, mk, E)
    var s = 0.0
    var Es: Array[Float64, 3] = [0.0, 0.0, 0.0]
    for n in range(N_DALITZ):
        var Ej = 0.5 * (hi - lo) * gl.xi[n] + 0.5 * (hi + lo)
        Es[i] = E
        Es[j] = Ej
        Es[k] = M - E - Ej
        s += (
            _dalitz_density(md, M, Es[0], Es[1], pol)
            * 0.5
            * (hi - lo)
            * gl.wi[n]
        )
    return s


def michel_nodes(M: Float64, i: Int, gl: XRules) -> XNodes:
    """tau+ -> l+ nu_l anti-nu_tau (V-A, massless products). The l+ is
    right-handed. The products are boosted as massless (p = E), which keeps
    energy and momentum. A muon below its mass then goes to the bin of its
    energy."""
    var r = XNodes()
    for n in range(N_MICHEL):
        var x = 0.5 * (gl.xm[n] + 1.0)
        var dx = 0.5 * gl.wm[n]
        var w: Float64
        var a: Float64
        if i == 1:
            w = 12.0 * x * x * (1.0 - x) * dx
            a = -1.0
        else:
            w = 2.0 * x * x * (3.0 - 2.0 * x) * dx
            a = (2.0 * x - 1.0) / (3.0 - 2.0 * x)
        var G = w if i == 0 else 0.0
        r.add(0.5 * x * M, w, G, a)
    return r^


def _nodes(md: XMode, M: Float64, i: Int, gl: XRules) -> XNodes:
    """Parent-frame energy distribution of product i of the mode (one per
    decay, before the branching ratio)."""
    var r = XNodes()
    var n = len(md.prod)
    var mi = x_mass(md.prod[i])
    if md.kind == XK_TWO:
        var mo = x_mass(md.prod[1 - i])
        var E = _emax(M, mi, mo)
        var P0 = md.P0 if i == 0 else 0.0
        var a = md.a0 if i == 0 else -md.a0
        r.add(E, 1.0, P0, a)
        return r^
    if md.kind == XK_MICHEL:
        return michel_nodes(M, i, gl)
    var others = 0.0
    var rest = List[Float64]()
    for j in range(n):
        if j != i:
            others += x_mass(md.prod[j])
            rest.append(x_mass(md.prod[j]))
    var Emax = _emax(M, mi, others)
    var total = 0.0
    for k in range(N_CREST):
        var E: Float64
        var w: Float64
        E, w = gl.rest_node(k, mi, Emax)
        var g: Float64
        var G = 0.0
        if md.kind == XK_PS:
            var p = sqrt(max(E * E - mi * mi, 0.0))
            var mrec = sqrt(max(M * M + mi * mi - 2.0 * M * E, 0.0))
            g = p * _phase_space(mrec, rest, gl)
            # V-A with a massless neutrino makes the l+ right-handed in
            # chirality. Its helicity is then +1 with the probability
            # (1 + p / E) / 2. (The matrix elements of the three-body modes
            # give 0.4-0.6 % less on average.)
            if md.P0 != 0.0 and (md.prod[i] == X_EP or md.prod[i] == X_MUP):
                G = md.P0 * g * p / E
        else:
            g = _three_body_line(md, M, i, E, False, gl)
            if i == 1:
                G = _three_body_line(md, M, i, E, True, gl)
        total += g * w
        r.add(E, g * w, G * w, 0.0)
    for k in range(len(r.w)):
        r.w[k] /= total
        r.G[k] /= total
    return r^


struct XAcc(Movable):
    """The energy is in units of the parent's energy."""

    var K: Int
    var N: Vec
    var En: Vec

    def __init__(out self, K: Int):
        self.K = K
        self.N = zeros((N_CPROD + N_X) * K)
        self.En = zeros((N_CPROD + N_X) * K)

    def slot_nonzero(self, slot: Int) -> Bool:
        for q in range(self.K):
            if self.N[slot * self.K + q] != 0.0:
                return True
        return False


@inline(.always)
def _x_to_cp(x: Int) -> Int:
    """Product slot of a final network particle (not a muon or K0)."""
    var cp: Array[Int, N_XFIN] = [
        CP_PIP,
        CP_PIM,
        CP_PI0,
        -1,
        -1,
        CP_KP,
        CP_KM,
        -1,
        CP_P,
        CP_N,
        CP_EP,
        CP_EM,
        CP_NUE,
        CP_NUEB,
        CP_NUM,
        CP_NUMB,
        CP_NUT,
        CP_NUTB,
        CP_GAM,
    ]
    return cp[x]


def _bin_any(
    M: Float64,
    m: Float64,
    E: Float64,
    wg: Float64,
    wG: Float64,
    xc: Vec,
    dlnE: Float64,
    mut acc: XAcc,
    slot: Int,
):
    """A product at rest in the parent frame (x = E / M exactly) goes to the
    bin that contains it."""
    var N = pd(acc.N).unsafe_offset(slot * acc.K)
    var En = pd(acc.En).unsafe_offset(slot * acc.K)
    var p = sqrt(max(E * E - m * m, 0.0))
    if p > 1e-12 * E:
        _bin_node(M, m, E, wg, wG, xc, dlnE, N, En)
        return
    var x = E / M
    var q = Int(floor(-c_log(x) / dlnE))
    if q >= 0 and q < acc.K:
        N[unsafe_offset=q] += wg
        En[unsafe_offset=q] += wg * x


def _emit(
    mut acc: XAcc,
    x: Int,
    M: Float64,
    E: Float64,
    w: Float64,
    G: Float64,
    xc: Vec,
    dlnE: Float64,
):
    var m = x_mass(x)
    if x == X_MUP or x == X_MUM:
        var l = CP_MUPL if x == X_MUP else CP_MUML
        _bin_any(M, m, E, 0.5 * w, -0.5 * G, xc, dlnE, acc, l)
        _bin_any(M, m, E, 0.5 * w, 0.5 * G, xc, dlnE, acc, l + 1)
    elif x == X_K0:
        _bin_any(M, m, E, 0.5 * w, 0.0, xc, dlnE, acc, CP_KS)
        _bin_any(M, m, E, 0.5 * w, 0.0, xc, dlnE, acc, CP_KL)
    elif x < N_XFIN:
        _bin_any(M, m, E, w, 0.0, xc, dlnE, acc, _x_to_cp(x))
    else:
        _bin_any(M, m, E, w, 0.0, xc, dlnE, acc, N_CPROD + x)


def _emit_tau(
    mut acc: XAcc,
    M: Float64,
    Etau: Float64,
    Ptau: Float64,
    wtau: Float64,
    tau_modes: List[XMode],
    xc: Vec,
    dlnE: Float64,
    gl: XRules,
):
    """The tau of a parent decay decays at once. A muon keeps its
    polarization."""
    var mt = MASS_TAU
    var g = Etau / mt
    var b = sqrt(max(1.0 - 1.0 / (g * g), 0.0))
    for t in range(len(tau_modes)):
        ref md = tau_modes[t]
        for j in range(len(md.prod)):
            var nd = _nodes(md, mt, j, gl)
            var m = x_mass(md.prod[j])
            for k in range(len(nd.E)):
                var Ep = nd.E[k]
                var pp = Ep if md.kind == XK_MICHEL else sqrt(
                    max(Ep * Ep - m * m, 0.0)
                )
                var w0 = wtau * md.br * nd.w[k]
                var gfrac = nd.G[k] / nd.w[k] if nd.w[k] != 0.0 else 0.0
                for n in range(N_BOX):
                    var c = gl.xb[n]
                    var w = w0 * 0.5 * gl.wb[n] * (1.0 + nd.a[k] * Ptau * c)
                    var ED = g * (Ep + b * pp * c)
                    _emit(acc, md.prod[j], M, ED, w, gfrac * w, xc, dlnE)


def _conv(A: Vec, B: Vec, dlnE: Float64) -> Vec:
    """Products per unit ln x of a cascade: A (the short-lived particle per
    unit ln x) convolved in ln x with B (its products per decay per unit
    ln y). Number and energy multiply exactly on the grid x_k = e^(-k dlnE).
    """
    var n = len(A)
    var C = zeros(n)
    var a_lo = n
    for j in range(n):
        if A[j] != 0.0:
            a_lo = j
            break
    for j in range(a_lo, n):
        var a = A[j] * dlnE
        if a == 0.0:
            continue
        for l in range(n - j):
            C[j + l] += a * B[l]
    return C^


def _is_zero(v: Vec) -> Bool:
    for x in v:
        if x != 0.0:
            return False
    return True


def _resolve(
    M: Float64,
    modes: List[XMode],
    tau_modes: List[XMode],
    F: List[List[Vec]],
    xc: Vec,
    dlnE: Float64,
    K: Int,
    gl: XRules,
) -> List[Vec]:
    """Products per decay per unit ln x of a parent of mass M with these modes.
    The products of its short-lived products x are F[x]."""
    var acc = XAcc(K)
    for d in range(len(modes)):
        ref md = modes[d]
        for i in range(len(md.prod)):
            var x = md.prod[i]
            var nd = _nodes(md, M, i, gl)
            for k in range(len(nd.E)):
                var w = md.br * nd.w[k]
                if x == X_TAUP:
                    _emit_tau(
                        acc,
                        M,
                        nd.E[k],
                        nd.G[k] / nd.w[k],
                        w,
                        tau_modes,
                        xc,
                        dlnE,
                        gl,
                    )
                else:
                    _emit(acc, x, M, nd.E[k], w, md.br * nd.G[k], xc, dlnE)
    var T = List[Vec]()
    for p in range(N_CPROD):
        T.append(_bins_to_table(acc.N, acc.En, p * K, xc, dlnE))
    for x in range(N_XFIN + 1, N_X):
        var slot = N_CPROD + x
        if not acc.slot_nonzero(slot):
            continue
        var TR = _bins_to_table(acc.N, acc.En, slot * K, xc, dlnE)
        for p in range(N_CPROD):
            if _is_zero(F[x][p]):
                continue
            var C = _conv(TR, F[x][p], dlnE)
            for k in range(K + 1):
                T[p][k] += C[k]
    return T^


def _short_lived_order() -> List[Int]:
    """The short-lived particles, each after those it decays into."""
    var o: List[Int] = [
        X_ETA,
        X_RHOP,
        X_RHO0,
        X_RHOM,
        X_KSTP,
        X_KSTM,
        X_KST0,
        X_KST0B,
        X_F0,
        X_OMEGA,
        X_PHI,
        X_ETAP,
        X_A1P,
        X_LAM,
        X_SIGP,
        X_SIGM,
        X_SIG0,
        X_XI0,
        X_XIM,
    ]
    return o^


# ------------------------------------------------------------------------------
# decay modes (PDG 2026)
# ------------------------------------------------------------------------------
def _renormalised(var t: List[XMode]) -> List[XMode]:
    var s = 0.0
    for k in range(len(t)):
        s += t[k].br
    for k in range(len(t)):
        t[k].br /= s
    return t^


def tau_decay_modes() -> List[XMode]:
    """tau+ decays: the eleven modes above 0.5 % of PDG 2026 (charge conjugates
    of its tau- modes), renormalized to their sum (97.045 %). l+ nu nu: the V-A
    spectra of a polarized tau with massless products. pi+ (K+) anti-nu_tau:
    the asymmetry -1 about the tau spin. pi+ pi0 as rho+ and the two
    three-pion modes as a1+, with the asymmetry of a spin-1 hadron. The a1+
    decays into pi+ pi0 pi0 and pi+ pi+ pi- in the ratio of the two modes. The
    others: unpolarized phase space."""
    var mt2 = MASS_TAU * MASS_TAU
    var mr2 = MASS_RHO * MASS_RHO
    var ma2 = MASS_A1 * MASS_A1
    var a_rho = -(mt2 - 2.0 * mr2) / (mt2 + 2.0 * mr2)
    var a_a1 = -(mt2 - 2.0 * ma2) / (mt2 + 2.0 * ma2)
    var t = List[XMode]()
    t.append(x_michel(0.1737, X_MUP, X_NUM, X_NUTB))  # G3
    t.append(x_michel(0.1785, X_EP, X_NUE, X_NUTB))  # G5
    t.append(x_two(0.1082, X_PIP, X_NUTB, a0=-1.0))  # G9
    t.append(x_two(0.00697, X_KP, X_NUTB, a0=-1.0))  # G10
    t.append(x_two(0.2549, X_RHOP, X_NUTB, a0=a_rho))  # G14
    t.append(x_two(0.0926 + 0.0899, X_A1P, X_NUTB, a0=a_a1))  # G20, G70
    t.append(x_ps(0.0104, [X_PIP, X_PI0, X_PI0, X_PI0, X_NUTB]))  # G27
    t.append(x_ps(0.00838, [X_PIP, X_K0, X_NUTB]))  # G36
    t.append(x_ps(0.0274, [X_PIP, X_PIP, X_PIM, X_PI0, X_NUTB]))  # G78
    t.append(x_ps(0.0195, [X_PIP, X_OMEGA, X_NUTB]))  # G178
    return _renormalised(t^)


def short_lived_modes(x: Int) -> List[XMode]:
    """Decays of the short-lived particles: every mode of 0.5 % and above of
    PDG 2026, renormalized to their sum. The modes are two-body modes or
    phase space. K* and f0 decay into their isospin shares (PDG gives none).
    The a1+ decays as the three-pion modes of the tau."""
    var t = List[XMode]()
    if x == X_ETA:
        t.append(x_two(0.3936, X_GAM, X_GAM))
        t.append(x_ps(0.3256, [X_PI0, X_PI0, X_PI0]))
        t.append(x_ps(0.2302, [X_PIP, X_PIM, X_PI0]))
        t.append(x_ps(0.0428, [X_PIP, X_PIM, X_GAM]))
        t.append(x_ps(0.00696, [X_EP, X_EM, X_GAM]))
    elif x == X_ETAP:
        t.append(x_ps(0.425, [X_PIP, X_PIM, X_ETA]))
        t.append(x_two(0.2948, X_RHO0, X_GAM))
        t.append(x_ps(0.224, [X_PI0, X_PI0, X_ETA]))
        t.append(x_two(0.0252, X_OMEGA, X_GAM))
        t.append(x_two(0.02307, X_GAM, X_GAM))
    elif x == X_OMEGA:
        t.append(x_ps(0.892, [X_PIP, X_PIM, X_PI0]))
        t.append(x_two(0.0833, X_PI0, X_GAM))
        t.append(x_two(0.0153, X_PIP, X_PIM))
    elif x == X_PHI:
        t.append(x_two(0.501, X_KP, X_KM))
        t.append(x_two(0.336, X_K0, X_K0))
        t.append(x_ps(0.147, [X_PIP, X_PIM, X_PI0]))
        t.append(x_two(0.01306, X_ETA, X_GAM))
    elif x == X_RHOP:
        t.append(x_two(1.0, X_PIP, X_PI0))
    elif x == X_RHO0:
        t.append(x_two(1.0, X_PIP, X_PIM))
    elif x == X_RHOM:
        t.append(x_two(1.0, X_PIM, X_PI0))
    elif x == X_KSTP:
        t.append(x_two(2.0 / 3.0, X_K0, X_PIP))
        t.append(x_two(1.0 / 3.0, X_KP, X_PI0))
    elif x == X_KSTM:
        t.append(x_two(2.0 / 3.0, X_K0, X_PIM))
        t.append(x_two(1.0 / 3.0, X_KM, X_PI0))
    elif x == X_KST0:
        t.append(x_two(2.0 / 3.0, X_KP, X_PIM))
        t.append(x_two(1.0 / 3.0, X_K0, X_PI0))
    elif x == X_KST0B:
        t.append(x_two(2.0 / 3.0, X_KM, X_PIP))
        t.append(x_two(1.0 / 3.0, X_K0, X_PI0))
    elif x == X_F0:
        t.append(x_two(2.0 / 3.0, X_PIP, X_PIM))
        t.append(x_two(1.0 / 3.0, X_PI0, X_PI0))
    elif x == X_A1P:
        t.append(x_ps(0.0926, [X_PIP, X_PI0, X_PI0]))
        t.append(x_ps(0.0899, [X_PIP, X_PIP, X_PIM]))
    elif x == X_LAM:
        t.append(x_two(0.641, X_P, X_PIM))
        t.append(x_two(0.359, X_N, X_PI0))
    elif x == X_SIGP:
        t.append(x_two(0.5147, X_P, X_PI0))
        t.append(x_two(0.4843, X_N, X_PIP))
    elif x == X_SIG0:
        t.append(x_two(1.0, X_LAM, X_GAM))
        t.append(x_ps(0.005, [X_LAM, X_EP, X_EM]))
    elif x == X_SIGM:
        t.append(x_two(0.99848, X_N, X_PIM))
    elif x == X_XI0:
        t.append(x_two(0.99517, X_LAM, X_PI0))
    elif x == X_XIM:
        t.append(x_two(0.99887, X_LAM, X_PIM))
    return _renormalised(t^)


def _sl_pp(
    mut t: List[XMode],
    be: Float64,
    bmu: Float64,
    h: Int,
    cs: Bool,
    alpha: Float64,
):
    var mp = MASS_DS_STAR if cs else MASS_D_STAR
    var m0 = MASS_DS0_STAR if cs else MASS_D0_STAR
    t.append(x_pl3(be, h, X_EP, X_NUE, mp, alpha, m0))
    t.append(x_pl3(bmu, h, X_MUP, X_NUM, mp, alpha, m0))


def _sl_pv(
    mut t: List[XMode],
    be: Float64,
    bmu: Float64,
    v: Int,
    cs: Bool,
    rV: Float64,
    r2: Float64,
):
    var mV = MASS_DS_STAR if cs else MASS_D_STAR
    var mA = MASS_DS1 if cs else MASS_D1
    var m0 = MASS_DS if cs else MASS_D
    t.append(x_vl3(be, v, X_EP, X_NUE, rV, r2, mV, mA, m0))
    t.append(x_vl3(bmu, v, X_MUP, X_NUM, rV, r2, mV, mA, m0))


def _sl_ps(
    mut t: List[XMode],
    be: Float64,
    bmu: Float64,
    var hadrons: List[Int],
):
    """A non-resonant semileptonic mode with more than one hadron, for e and
    mu, as phase space with the l+ of V-A (P0 = 1)."""
    if be > 0.0:
        var pe = hadrons.copy()
        pe.append(X_EP)
        pe.append(X_NUE)
        var me = x_ps(be, pe^)
        me.P0 = 1.0
        t.append(me^)
    if bmu > 0.0:
        var pm = hadrons^
        pm.append(X_MUP)
        pm.append(X_NUM)
        var mm = x_ps(bmu, pm^)
        mm.P0 = 1.0
        t.append(mm^)


def _scaled(var t: List[XMode], s: Float64) -> List[XMode]:
    for k in range(len(t)):
        t[k].br *= s
    return t^


def charm_decay_modes(s: Int) -> List[XMode]:
    """Decays of D+, D0, Ds+ and Lambda_c+ (PDG 2026,
    tests/literature_checks/charm/README.md, "Decays"). The comments give the
    PDG names.

    Semileptonic: every exclusive mode above 0.1 % that does not overlap
    another, with its non-resonant complement (S-wave) where PDG lists one.
    P -> P' (scalar f0 alike) and P -> V have pole form factors. The alpha of
    the modified pole is from BESIII 2015. r_V = V(0)/A1(0) and
    r_2 = A2(0)/A1(0) are from PDG 2026, the SEMILEPTONIC FORM FACTORS of the
    parent's listing, cited above each mode. Lambda_c+ -> Lambda, n have the
    lattice form factors, and the others are phase space. The mu modes PDG
    does not list (Ds+ -> K0, K*0, f0) are taken equal to the e modes.
    Lambda_c+ -> n l+ nu (not in PDG 2026) is from Meinel (2018,
    eqs. 20-21). The modes are scaled so that the e+ modes add up to PDG's
    inclusive e+ rate (the mu+ modes by the same factor). Hadronic: every mode
    above 0.5 %, less the parts of their final states that come from the eta,
    eta', omega or phi of another listed mode. A K0S x 2 counts as a K0 or
    anti-K0 (two K0S x 4). The hadronic modes are renormalized to the rest of
    the width."""
    var sl = List[XMode]()
    var lep = List[XMode]()
    var had = List[XMode]()
    var incl: Float64
    if s == CDP:
        _sl_pp(sl, 0.0881, 0.0868, X_K0, True, 0.309)  # G17, G18
        # r_V = 1.47 +- 0.04 (p. 66), r_2 = 0.786 +- 0.018 (p. 67), OUR AVERAGE
        _sl_pv(sl, 0.0540, 0.0526, X_KST0B, True, 1.47, 0.786)  # G29, G36
        _sl_ps(sl, 0.00228, 0.0019, [X_KM, X_PIP])  # G25, G35
        _sl_ps(sl, 0.001206, 0.00128, [X_K0, X_PI0])  # 2 G20, 2 G31
        _sl_pp(sl, 0.00372, 0.00350, X_PI0, False, 0.279)  # G44, G45
        _sl_pp(sl, 0.00111, 0.00104, X_ETA, False, 0.0)  # G46, G47
        # D+,D0 -> rho e+ nu_e: r_V = 1.53 +- 0.08, r_2 = 0.82 +- 0.05 (p. 66),
        # OUR AVERAGE
        _sl_pv(sl, 0.00187, 0.00164, X_RHO0, False, 1.53, 0.82)  # G50, G53
        _sl_ps(sl, 0.00062, 0.00072, [X_PIP, X_PIM])  # G49, G52 (f0(500))
        # r_V = 1.24 +- 0.09 +- 0.06, r_2 = 1.06 +- 0.15 +- 0.05 (p. 65,
        # ABLIKIM 15W)
        _sl_pv(sl, 0.00169, 0.00177, X_OMEGA, False, 1.24, 1.06)  # G54, G55
        _sl_ps(sl, 0.00123, 0.00109, [X_KM, X_PIP, X_PI0])  # G37, G40 (K1)
        lep.append(x_two(0.000402, X_MUP, X_NUM, P0=-1.0))  # G15
        lep.append(x_two(0.00099, X_TAUP, X_NUT, P0=-1.0))  # G16
        incl = 0.1607  # G1
    elif s == CD0:
        _sl_pp(sl, 0.03537, 0.03418, X_KM, True, 0.309)  # G20, G21
        # r_V = 1.45 +- 0.04, r_2 = 0.720 +- 0.033 (p. 129), OUR AVERAGE
        _sl_pv(sl, 0.0216, 0.0201, X_KSTM, True, 1.45, 0.720)  # G22, G23
        _sl_ps(sl, 0.00085, 0.00075, [X_K0, X_PIM])  # G26, G30
        _sl_ps(sl, 0.0, 0.000409, [X_KM, X_PI0])  # G28 (mu only)
        _sl_pp(sl, 0.00291, 0.00267, X_PIM, False, 0.279)  # G39, G40
        # the D+,D0 -> rho e+ nu_e averages of the D+- listing (p. 66), to
        # which the D0 listing refers
        _sl_pv(sl, 0.00146, 0.00135, X_RHOM, False, 1.53, 0.82)  # G42, G43
        incl = 0.0649  # G5
    elif s == CDSP:
        # r_V = 1.76 +- 0.07 (p. 60), r_2 = 0.83 +- 0.08 (p. 59), OUR AVERAGE
        _sl_pv(sl, 0.0234, 0.0224, X_PHI, True, 1.76, 0.83)  # G26, G29
        _sl_pp(sl, 0.0227, 0.0224, X_ETA, True, 0.0)  # G30, G32
        _sl_pp(sl, 0.0081, 0.0080, X_ETAP, True, 0.0)  # G31, G33
        _sl_pp(sl, 0.00288, 0.00288, X_K0, False, 0.0)  # G35
        # r_V = 1.67 +- 0.34 +- 0.16, r_2 = 0.77 +- 0.28 +- 0.07 (p. 62,
        # ABLIKIM 19D)
        _sl_pv(sl, 0.00205, 0.00205, X_KST0, False, 1.67, 0.77)  # G36
        _sl_pp(sl, 0.00243, 0.00243, X_F0, True, 0.0)  # G39 + G40
        lep.append(x_two(0.00537, X_MUP, X_NUM, P0=-1.0))  # G21
        lep.append(x_two(0.0539, X_TAUP, X_NUT, P0=-1.0))  # G22
        incl = 0.0633  # G1
    else:
        sl.append(x_bl3(0.0356, X_LAM, X_EP, X_NUE, 0))  # G109
        sl.append(x_bl3(0.0348, X_LAM, X_MUP, X_NUM, 0))  # G115
        sl.append(x_bl3(0.00410, X_N, X_EP, X_NUE, 1))  # Meinel 2018
        sl.append(x_bl3(0.00400, X_N, X_MUP, X_NUM, 1))
        incl = 0.0406  # G116
    if s == CDP:
        had.append(x_ps(0.03108, [X_K0, X_PIP]))  # K0bar pi+ (G66, K0S x 2)
        had.append(x_ps(0.0931, [X_KM, X_PIP, X_PIP]))  # K- pi+ pi+ (G68)
        had.append(
            x_ps(0.1462, [X_K0, X_PIP, X_PI0])
        )  # K0bar pi+ pi0 (G78, K0S x 2)
        had.append(
            x_ps(0.0142, [X_K0, X_PIP, X_OMEGA])
        )  # K0bar pi+ omega (G88, K0S x 2)
        had.append(
            x_ps(0.0254, [X_K0, X_PIP, X_ETA])
        )  # K0bar pi+ eta (G89, K0S x 2)
        had.append(
            x_ps(0.0635, [X_KM, X_PIP, X_PIP, X_PI0])
        )  # K- pi+ pi+ pi0 (G93)
        had.append(
            x_ps(0.061383, [X_K0, X_PIP, X_PIP, X_PIM])
        )  # K0bar pi+ pi+ pi- (G107, remainder)
        had.append(
            x_ps(0.0578, [X_K0, X_PIP, X_PI0, X_PI0])
        )  # K0bar pi+ pi0 pi0 (G108, K0S x 2)
        had.append(
            x_ps(0.0057, [X_KM, X_PIP, X_PIP, X_PIP, X_PIM])
        )  # K- pi+ pi+ pi+ pi- (G117)
        had.append(
            x_ps(0.012087, [X_K0, X_PIP, X_PIP, X_PIM, X_PI0])
        )  # K0bar pi+ pi+ pi- pi0 (G124, remainder)
        had.append(
            x_ps(0.0027298, [X_K0, X_PIP, X_PI0, X_PI0, X_PI0])
        )  # K0bar pi+ pi0 pi0 pi0 (G125, remainder)
        had.append(
            x_ps(0.010812, [X_PIP, X_PIP, X_PIM, X_PI0])
        )  # pi+ pi+ pi- pi0 (G147, remainder)
        had.append(
            x_ps(0.0107, [X_PIP, X_PIP, X_PIM, X_PI0, X_PI0])
        )  # pi+ pi+ pi- pi0 pi0 (G150)
        had.append(x_ps(0.00604, [X_K0, X_KP]))  # K0bar K+ (G170, K0S x 2)
        had.append(
            x_ps(0.01014, [X_K0, X_KP, X_PI0])
        )  # K0bar K+ pi0 (G172, K0S x 2)
        had.append(
            x_ps(0.0067543, [X_KP, X_KM, X_PIP])
        )  # K+ K- pi+ (G177, remainder)
        had.append(x_ps(0.0057, [X_PHI, X_PIP]))  # phi pi+ (G185)
        had.append(
            x_ps(0.00662, [X_KP, X_KM, X_PIP, X_PI0])
        )  # K+ K- pi+ pi0 (G186)
        had.append(
            x_ps(0.01188, [X_K0, X_K0, X_PIP])
        )  # K0 K0bar pi+ (G187, K0S K0S x 4)
        had.append(
            x_ps(0.00536, [X_K0, X_K0, X_PIP, X_PI0])
        )  # K0 K0bar pi+ pi0 (G190, K0S K0S x 4)
        had.append(
            x_ps(0.01016, [X_KP, X_K0, X_K0])
        )  # K+ K0bar K0bar (G128, K0S K0S x 4)
    elif s == CD0:
        had.append(x_ps(0.03936, [X_KM, X_PIP]))  # K- pi+ (G47)
        had.append(x_ps(0.02478, [X_K0, X_PI0]))  # K0bar pi0 (G48, K0S x 2)
        had.append(
            x_ps(0.05646, [X_K0, X_PIP, X_PIM])
        )  # K0bar pi+ pi- (G53, remainder)
        had.append(x_ps(0.144, [X_KM, X_PIP, X_PI0]))  # K- pi+ pi0 (G68)
        had.append(
            x_ps(0.0182, [X_K0, X_PI0, X_PI0])
        )  # K0bar pi0 pi0 (G77, K0S x 2)
        had.append(
            x_ps(0.011892, [X_K0, X_PI0, X_PI0, X_PI0])
        )  # K0bar pi0 pi0 pi0 (G86, remainder)
        had.append(
            x_ps(0.081481, [X_KM, X_PIP, X_PIP, X_PIM])
        )  # K- pi+ pi+ pi- (G87, remainder)
        had.append(
            x_ps(0.08039, [X_K0, X_PIP, X_PIM, X_PI0])
        )  # K0bar pi+ pi- pi0 (G98, remainder)
        had.append(
            x_ps(0.0886, [X_KM, X_PIP, X_PI0, X_PI0])
        )  # K- pi+ pi0 pi0 (G101)
        had.append(
            x_ps(0.0033787, [X_KM, X_PIP, X_PI0, X_PI0, X_PI0])
        )  # K- pi+ pi0 pi0 pi0 (G102, remainder)
        had.append(
            x_ps(0.005586, [X_K0, X_PIP, X_PIM, X_PI0, X_PI0])
        )  # K0bar pi+ pi- pi0 pi0 (G103, remainder)
        had.append(
            x_ps(0.0084334, [X_KM, X_PIP, X_PIP, X_PIM, X_PI0])
        )  # K- pi+ pi+ pi- pi0 (G104, remainder)
        had.append(x_ps(0.0339, [X_KM, X_PIP, X_OMEGA]))  # K- pi+ omega (G107)
        had.append(
            x_ps(0.017, [X_K0, X_PI0, X_OMEGA])
        )  # K0bar pi0 omega (G109, K0S x 2)
        had.append(
            x_ps(0.0202, [X_K0, X_ETA, X_PI0])
        )  # K0bar eta pi0 (G110, K0S x 2)
        had.append(x_ps(0.0188, [X_KM, X_PIP, X_ETA]))  # K- pi+ eta (G113)
        had.append(
            x_ps(0.0056, [X_K0, X_PIP, X_PIM, X_ETA])
        )  # K0bar pi+ pi- eta (G118, K0S x 2)
        had.append(
            x_ps(0.0054, [X_K0, X_PIP, X_PIP, X_PIM, X_PIM])
        )  # K0bar pi+ pi+ pi- pi- (G120, K0S x 2)
        had.append(x_ps(0.01016, [X_K0, X_ETA]))  # K0bar eta (G128, K0S x 2)
        had.append(x_ps(0.0222, [X_K0, X_OMEGA]))  # K0bar omega (G129, K0S x 2)
        had.append(x_ps(0.01898, [X_K0, X_ETAP]))  # K0bar etap (G130, K0S x 2)
        had.append(x_ps(0.00643, [X_KM, X_PIP, X_ETAP]))  # K- pi+ etap (G133)
        had.append(
            x_ps(0.00504, [X_K0, X_ETAP, X_PI0])
        )  # K0bar etap pi0 (G134, K0S x 2)
        had.append(
            x_ps(0.00486, [X_K0, X_KP, X_KM])
        )  # K0bar K+ K- (G136, remainder)
        had.append(x_ps(0.0082236, [X_K0, X_PHI]))  # K0bar phi (G141, derived)
        had.append(x_ps(0.0149, [X_PIP, X_PIM, X_PI0]))  # pi+ pi- pi0 (G153)
        had.append(
            x_ps(0.00715, [X_PIP, X_PIP, X_PIM, X_PIM])
        )  # pi+ pi+ pi- pi- (G171)
        had.append(
            x_ps(0.01002, [X_PIP, X_PIM, X_PI0, X_PI0])
        )  # pi+ pi- pi0 pi0 (G212)
        had.append(
            x_ps(0.0068, [X_K0, X_KM, X_PIP])
        )  # K0 K- pi+ (G268, K0S x 2)
    elif s == CDSP:
        had.append(x_ps(0.0295, [X_KP, X_K0]))  # K+ K0bar (G51)
        had.append(
            x_ps(0.031955, [X_KP, X_KM, X_PIP])
        )  # K+ K- pi+ (G52, remainder)
        had.append(x_ps(0.045, [X_PHI, X_PIP]))  # phi pi+ (G53)
        had.append(
            x_ps(0.02942, [X_KP, X_K0, X_PI0])
        )  # K+ K0bar pi0 (G65, K0S x 2)
        had.append(
            x_ps(0.01812, [X_K0, X_K0, X_PIP])
        )  # K0 K0bar pi+ (G46, derived)
        had.append(x_ps(0.049, [X_PHI, X_RHOP]))  # phi rho+ (G73, resonant)
        had.append(
            x_ps(0.018218, [X_KP, X_KM, X_PIP, X_PI0])
        )  # K+ K- pi+ pi0 (G74, remainder)
        had.append(
            x_ps(0.0063133, [X_K0, X_KM, X_PIP, X_PIP])
        )  # K0 K- pi+ pi+ (G82, remainder)
        had.append(
            x_ps(0.01868, [X_KP, X_K0, X_PIP, X_PIM])
        )  # K+ K0bar pi+ pi- (G89, K0S x 2)
        had.append(
            x_ps(0.0564, [X_KSTP, X_KST0B])
        )  # Kstar+ Kstar0bar (G84, resonant)
        had.append(
            x_ps(0.0005379, [X_KP, X_KM, X_PIP, X_PIP, X_PIM])
        )  # K+ K- pi+ pi+ pi- (G90, remainder)
        had.append(
            x_ps(0.0121, [X_PHI, X_PIP, X_PIP, X_PIM])
        )  # phi pi+ pi+ pi- (G91)
        had.append(x_ps(0.0109, [X_PIP, X_PIP, X_PIM]))  # pi+ pi+ pi- (G99)
        had.append(x_ps(0.0052, [X_PIP, X_PI0, X_PI0]))  # pi+ pi0 pi0 (G110)
        had.append(
            x_ps(0.013214, [X_PIP, X_PIP, X_PIM, X_PI0])
        )  # pi+ pi+ pi- pi0 (G117, remainder)
        had.append(x_ps(0.01686, [X_ETA, X_PIP]))  # eta pi+ (G116)
        had.append(
            x_ps(0.0077552, [X_PIP, X_PIP, X_PIP, X_PIM, X_PIM])
        )  # pi+ pi+ pi+ pi- pi- (G129, remainder)
        had.append(
            x_ps(0.011699, [X_PIP, X_PIP, X_PIM, X_PI0, X_PI0])
        )  # pi+ pi+ pi- pi0 pi0 (G130, remainder)
        had.append(x_ps(0.091, [X_ETA, X_PIP, X_PI0]))  # eta pi+ pi0 (G140)
        had.append(x_ps(0.028, [X_OMEGA, X_PIP, X_PI0]))  # omega pi+ pi0 (G143)
        had.append(
            x_ps(0.0308, [X_ETA, X_PIP, X_PIP, X_PIM])
        )  # eta pi+ pi+ pi- (G144)
        had.append(
            x_ps(0.021995, [X_PIP, X_PIP, X_PIP, X_PIM, X_PIM, X_PI0])
        )  # pi+ pi+ pi+ pi- pi- pi0 (G152, remainder)
        had.append(
            x_ps(0.016, [X_OMEGA, X_PIP, X_PIP, X_PIM])
        )  # omega pi+ pi+ pi- (G153)
        had.append(x_ps(0.0395, [X_ETAP, X_PIP]))  # etap pi+ (G154)
        had.append(
            x_ps(0.0054, [X_OMEGA, X_ETA, X_PIP])
        )  # omega eta pi+ (G156)
        had.append(x_ps(0.0614, [X_ETAP, X_PIP, X_PI0]))  # etap pi+ pi0 (G158)
        had.append(x_ps(0.00623, [X_KP, X_PIP, X_PIM]))  # K+ pi+ pi- (G165)
        had.append(
            x_ps(0.01018, [X_K0, X_PIP, X_PI0])
        )  # K0 pi+ pi0 (G176, K0S x 2)
        had.append(
            x_ps(0.0056, [X_K0, X_PIP, X_PIP, X_PIM])
        )  # K0 pi+ pi+ pi- (G182, K0S x 2)
        had.append(
            x_ps(0.0097, [X_KP, X_PIP, X_PIM, X_PI0])
        )  # K+ pi+ pi- pi0 (G183)
    else:
        had.append(x_ps(0.0326, [X_P, X_K0]))  # p K0bar (G1, K0S x 2)
        had.append(x_ps(0.0637, [X_P, X_KM, X_PIP]))  # p K- pi+ (G3)
        had.append(
            x_ps(0.0424, [X_P, X_K0, X_PI0])
        )  # p K0bar pi0 (G17, K0S x 2)
        had.append(
            x_ps(0.0372, [X_N, X_K0, X_PIP])
        )  # n K0bar pi+ (G19, K0S x 2)
        had.append(
            x_ps(0.017, [X_N, X_K0, X_PIP, X_PI0])
        )  # n K0bar pi+ pi0 (G21, K0S x 2)
        had.append(x_ps(0.019, [X_N, X_KM, X_PIP, X_PIP]))  # n K- pi+ pi+ (G22)
        had.append(x_ps(0.009, [X_P, X_K0, X_ETA]))  # p K0bar eta (G23)
        had.append(
            x_ps(0.0326, [X_P, X_K0, X_PIP, X_PIM])
        )  # p K0bar pi+ pi- (G24, K0S x 2)
        had.append(
            x_ps(0.0454, [X_P, X_KM, X_PIP, X_PI0])
        )  # p K- pi+ pi0 (G26)
        had.append(
            x_ps(0.01, [X_P, X_KM, X_PIP, X_PI0, X_PI0])
        )  # p K- pi+ pi0 pi0 (G31)
        had.append(x_ps(0.0064, [X_N, X_PIP, X_PI0]))  # n pi+ pi0 (G40)
        had.append(x_ps(0.0132, [X_LAM, X_PIP]))  # Lambda pi+ (G50)
        had.append(x_ps(0.0716, [X_LAM, X_PIP, X_PI0]))  # Lambda pi+ pi0 (G52)
        had.append(
            x_ps(0.036671, [X_LAM, X_PIP, X_PIP, X_PIM])
        )  # Lambda pi+ pi+ pi- (G56, remainder)
        had.append(
            x_ps(0.0052002, [X_LAM, X_PIP, X_PIP, X_PIM, X_PI0])
        )  # Lambda pi+ pi+ pi- pi0 (G62, remainder)
        had.append(
            x_ps(0.015, [X_LAM, X_PIP, X_OMEGA])
        )  # Lambda pi+ omega (G63)
        had.append(x_ps(0.0192, [X_LAM, X_PIP, X_ETA]))  # Lambda pi+ eta (G65)
        had.append(x_ps(0.0059, [X_LAM, X_KP, X_K0]))  # Lambda K+ K0bar (G69)
        had.append(x_ps(0.0129, [X_SIG0, X_PIP]))  # Sigma0 pi+ (G71)
        had.append(x_ps(0.0076, [X_SIG0, X_PIP, X_ETA]))  # Sigma0 pi+ eta (G72)
        had.append(x_ps(0.0127, [X_SIGP, X_PI0]))  # Sigma+ pi0 (G73)
        had.append(
            x_ps(0.045437, [X_SIGP, X_PIP, X_PIM])
        )  # Sigma+ pi+ pi- (G76, remainder)
        had.append(x_ps(0.0187, [X_SIGM, X_PIP, X_PIP]))  # Sigma- pi+ pi+ (G78)
        had.append(x_ps(0.036, [X_SIG0, X_PIP, X_PI0]))  # Sigma0 pi+ pi0 (G79)
        had.append(x_ps(0.0157, [X_SIGP, X_PI0, X_PI0]))  # Sigma+ pi0 pi0 (G80)
        had.append(
            x_ps(0.0113, [X_SIG0, X_PIP, X_PIP, X_PIM])
        )  # Sigma0 pi+ pi+ pi- (G81)
        had.append(x_ps(0.0172, [X_SIGP, X_OMEGA]))  # Sigma+ omega (G82)
        had.append(
            x_ps(0.021, [X_SIGM, X_PIP, X_PIP, X_PI0])
        )  # Sigma- pi+ pi+ pi0 (G83)
        had.append(x_ps(0.0055, [X_XI0, X_KP]))  # Xi0 K+ (G90)
        had.append(x_ps(0.0063, [X_XIM, X_KP, X_PIP]))  # Xi- K+ pi+ (G91)
        had.append(x_ps(0.0078, [X_XI0, X_KP, X_PI0]))  # Xi0 K+ pi0 (G92)
        had.append(
            x_ps(0.0074, [X_XI0, X_K0, X_PIP])
        )  # Xi0 K0 pi+ (G93, K0S x 2)
    var se = 0.0
    for k in range(len(sl)):
        if sl[k].prod[len(sl[k].prod) - 2] == X_EP:
            se += sl[k].br
    var t = _scaled(sl^, incl / se)
    var b_sl = 0.0
    for k in range(len(t)):
        b_sl += t[k].br
    var b_lep = 0.0
    for k in range(len(lep)):
        b_lep += lep[k].br
        t.append(lep[k].copy())
    var b_had = 0.0
    for k in range(len(had)):
        b_had += had[k].br
    var h = _scaled(had^, (1.0 - b_sl - b_lep) / b_had)
    for k in range(len(h)):
        t.append(h[k].copy())
    return t^


def charm_decay_tables(dlnE: Float64, K: Int) -> List[Vec]:
    """Products p per decay of charm species s at the energy fraction
    x_k = exp(-k dlnE), per unit ln x (antiparticles: the CP conjugates of D+,
    D0, Ds+, Lambda_c+). Products below x_K are dropped."""
    var gl = XRules()
    var xc = zeros(K + 1)
    for k in range(K + 1):
        xc[k] = c_exp(-Float64(k) * dlnE)
    var tau = tau_decay_modes()
    var F = List[List[Vec]]()
    for _ in range(N_X):
        var z = List[Vec]()
        for _ in range(N_CPROD):
            z.append(zeros(K + 1))
        F.append(z^)
    var order = _short_lived_order()
    for x in order:
        F[x] = _resolve(
            x_mass(x), short_lived_modes(x), tau, F, xc, dlnE, K, gl
        )
    var tabs = List[Vec]()
    for _ in range(N_CHARM * N_CPROD):
        tabs.append(Vec())
    var parts: List[Int] = [CDP, CD0, CDSP, CLC]
    for s in parts:
        var T = _resolve(
            charm_mass(s), charm_decay_modes(s), tau, F, xc, dlnE, K, gl
        )
        for p in range(N_CPROD):
            tabs[(s + 1) * N_CPROD + charm_product_conjugate(p)] = T[p].copy()
            tabs[s * N_CPROD + p] = T[p].copy()
    return tabs^


struct CDecJob(Movable):
    var D: Pointer[CharmDecay, MutUntrackedOrigin]
    var dlnE: Float64

    def __init__(
        out self, D: Pointer[CharmDecay, MutUntrackedOrigin], dlnE: Float64
    ):
        self.D = D
        self.dlnE = dlnE


def cdec_work(job: Pointer[CDecJob, MutUntrackedOrigin], q: Int):
    ref D = job[].D[]
    _decay_task(
        D.K,
        D.has,
        D.w,
        D.active,
        D.jlo,
        D.jhi,
        D.tmp,
        D.out,
        D.tasks,
        q,
        job[].dlnE,
    )


struct CharmDecay(Copyable, Movable):
    var verbose: Bool
    var K: List[MatD]
    var has: List[Bool]
    # per origin o (0: photo-hadronic, 1: pp)
    var out: List[Vec]
    var w: List[Vec]
    var active: List[Bool]
    var jlo: List[Int]
    var jhi: List[Int]
    var tmp: List[Vec]
    var tasks: List[Int]
    var nthreads: Int
    var built_nh: Int

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.K = List[MatD]()
        self.has = List[Bool]()
        for _ in range(N_CHARM * N_CPROD):
            self.K.append(MatD())
            self.has.append(False)
        self.out = List[Vec]()
        self.tmp = List[Vec]()
        for _ in range(2 * N_CPROD):
            self.out.append(Vec())
            self.tmp.append(Vec())
        self.w = List[Vec]()
        self.active = List[Bool]()
        self.jlo = List[Int]()
        self.jhi = List[Int]()
        for _ in range(2 * N_CHARM):
            self.w.append(Vec())
            self.active.append(False)
            self.jlo.append(0)
            self.jhi.append(0)
        self.tasks = List[Int]()
        self.nthreads = 1
        self.built_nh = -1
        self.clear_internal_arrays(dat)

    def free_kernels(mut self):
        for k in range(len(self.K)):
            self.K[k] = MatD()
            self.has[k] = False
        self.built_nh = -1

    def ready(self, dat: AM3Arrays) -> Bool:
        return self.built_nh == dat.nbins_had

    @staticmethod
    def _rows(dat: AM3Arrays, p: Int) -> Int:
        var g = charm_product_grid(p)
        if g == 0:
            return dat.nbins_had
        if g == 1:
            return dat.nbins_lep_hadgrid
        return dat.nbins_nu if g == 2 else dat.nbins_pho_hadgrid

    @staticmethod
    def _energy(dat: AM3Arrays, p: Int, i: Int) -> Float64:
        var g = charm_product_grid(p)
        if g == 0:
            return dat.egrid_had[i]
        if g == 1:
            return dat.egrid_had_leptons[i]
        return dat.egrid_nu[i] if g == 2 else dat.egrid_had_photons[i]

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        for o in range(2):
            for p in range(N_CPROD):
                resize_zero(self.out[o * N_CPROD + p], CharmDecay._rows(dat, p))

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing charm decay kernels")
        var nh = dat.nbins_had
        # The grids are logarithmic with step dlnE, so E_i / E_j = x_k with
        # integer k. K is the k of the lowest x on the hadron, lepton and
        # neutrino grids (photons below it are dropped).
        var K = 0
        for p in range(CP_GAM):
            var x = CharmDecay._energy(dat, p, 0) / dat.egrid_had[nh - 1]
            K = max(K, Int(floor(-c_log(x) / dat.dlnE + 0.5)))
        var T = charm_decay_tables(dat.dlnE, K)
        _decay_kernels[CharmDecay._rows, CharmDecay._energy](
            dat, T, N_CPROD, K, self.K, self.has
        )
        self.built_nh = nh

    def Calc_CtoX(
        mut self, dat: AM3Arrays, base0: Int, base1: Int, on0: Bool, on1: Bool
    ):
        if self.verbose:
            print("computing charm decay terms")
        self.clear_internal_arrays(dat)
        if not self.ready(dat):
            return
        if not _decay_rates[charm_mass, charm_lifetime](
            dat,
            base0,
            base1,
            on0,
            on1,
            self.w,
            self.active,
            self.jlo,
            self.jhi,
        ):
            return
        self.tasks.clear()
        for k in range(2 * N_CPROD):
            resize_zero(self.tmp[k], len(self.out[k]))
            if k < N_CPROD:
                _row_tasks(self.tasks, k, len(self.out[k]))
        var job = CDecJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](), dat.dlnE
        )
        parallel_tasks[cdec_work](job, len(self.tasks) // 3, self.nthreads)
