"""Port of Pair (PairProduction.cc): gamma-gamma annihilation into e+/-."""

from std.math import sqrt
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import MatD, MatI, DOT_BLOCK, dot_nonzero, gemv_d
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.specfun import beta, signf


struct PairInitJob(Movable):
    var P: Pointer[Pair, MutUntrackedOrigin]
    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]

    def __init__(
        out self,
        P: Pointer[Pair, MutUntrackedOrigin],
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
    ):
        self.P = P
        self.dat = dat


def pair_init_work(
    job: Pointer[PairInitJob, MutUntrackedOrigin], t: Int, nt: Int
):
    """The rows of KER_RggD1 that pair_lep_work reads. Each row is cut to the
    blocks of `dot_nonzero` that reach its non-zero values."""
    ref P = job[].P[]
    ref dat = job[].dat[]
    var n = dat.nbins_pho
    var row = zeros(n)
    var i = t
    while i < dat.nbins_lep:
        var seg = Vec()
        for j in range(P.KER_outerPP[i], n):
            var k0 = P.KER_innerPP.get(i, j)
            var lo = n
            var hi = 0
            for k in range(k0, n):
                var v = P.RggD1(
                    dat.egrid_lep[i], dat.egrid_pho[j], dat.egrid_pho[k]
                )
                row[k] = v
                if v != 0.0:
                    lo = min(lo, k)
                    hi = k + 1
            var r = i * n + j
            P.KER_nzlo[r] = lo
            P.KER_nzhi[r] = hi
            if hi > lo:
                var ks = k0 + (lo - k0) // DOT_BLOCK * DOT_BLOCK
                var ke = min(
                    n, k0 + (hi - k0 + DOT_BLOCK - 1) // DOT_BLOCK * DOT_BLOCK
                )
                P.KER_start[r] = len(seg) - ks
                for k in range(ks, ke):
                    seg.append(row[k])
        P.KER_RggD1[i] = seg^
        i += nt


struct PairLepJob(Movable):
    var P: Pointer[Pair, MutUntrackedOrigin]
    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]
    var factor: Int

    def __init__(
        out self,
        P: Pointer[Pair, MutUntrackedOrigin],
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
        factor: Int,
    ):
        self.P = P
        self.dat = dat
        self.factor = factor


def pair_lep_work(
    job: Pointer[PairLepJob, MutUntrackedOrigin], t: Int, nt: Int
):
    """The sums over k skip the blocks of `dot` that hold only zeros of the
    kernel (outside the kinematic limits, most of the range)."""
    ref P = job[].P[]
    ref dat = job[].dat[]
    var png = pd(P.PP_NG)
    var n = dat.nbins_pho
    var i = 1 + t
    while i < dat.nbins_lep - 1:
        var x = 0.0
        var Kp = pd(P.KER_RggD1[i])
        for j in range(P.KER_outerPP[i], n):
            var k0 = P.KER_innerPP.get(i, j)
            var kmax = min(-j + 2 * dat.ind_min_energy_pho + job[].factor, n)
            var r = i * n + j
            if kmax > k0:
                x += png[unsafe_offset=j] * dot_nonzero(
                    Kp.unsafe_offset(P.KER_start[r] + k0),
                    png.unsafe_offset(k0),
                    kmax - k0,
                    P.KER_nzlo[r] - k0,
                    P.KER_nzhi[r] - k0,
                )
        P.epspairL[i] = x * dat.dlnE * dat.dlnE * 0.75 * dat.egrid_lep[i]
        i += nt


struct Pair(Copyable, Movable):
    var nthreads: Int
    var verbose: Bool
    var afpairG: Vec
    var epspairL: Vec
    var PP_NG: Vec
    var KER_phibarD: MatD
    # The C++ code allocates one dense matrix per photon bin,
    # nbins_pho^3 doubles (~1.9 GB). Only the first nbins_lep = 340 matrices
    # are used. The C++ code allocates them even when the kernel is never
    # used: braces are missing after `if(!feedback_8bin)` in PairProduction.cc.
    var KER_RggD1: List[Vec]
    var KER_start: VecI
    var KER_nzlo: VecI
    var KER_nzhi: VecI
    var KER_outerPP: VecI
    var KER_innerPP: MatI
    var full_built: Bool
    var A0V: Vec
    var A_A0_hV: Vec
    var REG: Float64

    def __init__(out self, rp: RunParams, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.nthreads = 1
        self.afpairG = Vec()
        self.epspairL = Vec()
        self.PP_NG = Vec()
        self.KER_phibarD = MatD()
        self.KER_RggD1 = List[Vec]()
        self.KER_start = VecI()
        self.KER_nzlo = VecI()
        self.KER_nzhi = VecI()
        self.KER_outerPP = VecI()
        self.KER_innerPP = MatI()
        self.full_built = False
        self.REG = rp.REG
        self.A0V = zeros(2101)
        self.A_A0_hV = zeros(2101)
        self.clear_internal_arrays(dat)
        for i in range(2101):
            var h = -1.0 + 0.01 * Float64(i)
            self.A0V[i] = self.A0f(h)
            self.A_A0_hV[i] = self.A_A0_hf(h)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.epspairL, dat.nbins_lep)
        resize_zero(self.afpairG, dat.nbins_pho)
        resize_zero(self.PP_NG, dat.nbins_pho)

    def reinitialize_empty_kernels(
        mut self, dat: AM3Arrays, feedback_8bin: Bool
    ):
        # (no photon loss kernel until initialize_kernels: with the process
        #  off at init_kernels, it takes no photons and makes no pairs, as
        #  with AM3's zero kernel)
        self.KER_phibarD = MatD()
        if not feedback_8bin:
            self.KER_outerPP = zerosi(dat.nbins_lep)
        self.KER_innerPP = MatI(dat.nbins_lep, dat.nbins_pho)
        self.KER_RggD1 = List[Vec]()
        self.full_built = False

    def _alloc_full_kernel(mut self, dat: AM3Arrays):
        if len(self.KER_RggD1) != dat.nbins_lep or len(self.KER_nzlo) != (
            dat.nbins_lep * dat.nbins_pho
        ):
            self.KER_RggD1 = List[Vec](length=dat.nbins_lep, fill=Vec())
            self.KER_start = zerosi(dat.nbins_lep * dat.nbins_pho)
            self.KER_nzlo = zerosi(dat.nbins_lep * dat.nbins_pho)
            self.KER_nzhi = zerosi(dat.nbins_lep * dat.nbins_pho)

    def initialize_kernels(
        mut self, rp: RunParams, dat: AM3Arrays, feedback_8bin: Bool
    ):
        """Both kernels, whatever the cooling (photon loss) and emission
        switches: the pairs take their number from the photon loss rates.
        Also, a switch turned on after init_kernels finds its kernel. (AM3
        built each kernel only with its switch on, so with the photon loss
        off no pairs were made.)"""
        self.REG = rp.REG
        if self.verbose:
            print("computing photon to pair annihilation kernel")
        self.KER_phibarD = MatD(dat.nbins_pho, dat.nbins_pho)
        for i in range(dat.nbins_pho):
            for j in range(dat.nbins_pho):
                var ei = dat.egrid_pho[i]
                var ej = dat.egrid_pho[j]
                self.KER_phibarD.set(
                    i, j, 0.375 * self.phibar(ei, ej) / ei / ei / ej / ej
                )
        if not feedback_8bin:
            self.build_full_kernel(dat)
        elif self.verbose:
            print(
                "using eight-bin approx. for e+- distribution from pair"
                " production"
            )

    def build_full_kernel(mut self, dat: AM3Arrays):
        """The full pair-emission kernel is built at init_kernels without the
        eight-bin approximation. Otherwise, it is built at the first step after
        the approximation is switched off (Calc_Pair_Lep), if init_kernels built
        the photon loss kernel. (AM3 then read a kernel that it did not
        allocate.)"""
        if self.verbose:
            print("computing full e+/- pair production kernel")
        self._alloc_full_kernel(dat)
        if len(self.KER_outerPP) < dat.nbins_lep:
            self.KER_outerPP = zerosi(dat.nbins_lep)
        for i in range(dat.nbins_lep):
            self.KER_outerPP[i] = self.outerPP(dat, i)
            for j in range(dat.nbins_pho):
                self.KER_innerPP.set(
                    i,
                    j,
                    min(
                        self.innerPP(dat, dat.egrid_pho[j], dat.egrid_lep[i]),
                        dat.nbins_pho - 1,
                    ),
                )
        var job = PairInitJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=dat)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
        )
        parallel_run[pair_init_work](job, self.nthreads)
        self.full_built = True
        if self.verbose:
            var size = 0
            for i in range(dat.nbins_lep):
                size += len(self.KER_RggD1[i])
            print("full pair kernel:", size, "values")

    def Calc_Pair_Phot(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing pair annihilation photon losses")
        if self.KER_phibarD.rows == 0:
            fill_zero(self.afpairG)
            return
        gemv_d(
            self.KER_phibarD, pd(dat.n[P_photons]), dat.dlnE, pd(self.afpairG)
        )

    def phibar(self, a: Float64, b: Float64) -> Float64:
        var x = a * b
        if x <= 1.0:
            return 0.0
        elif x > 1.0 and x <= 1.1:
            return self.phibar1(x)
        elif x > 1.1 and x < 5.0:
            return self.phibar2(x, -1.0 + 2.0 * x * (1.0 + sqrt(1.0 - 1.0 / x)))
        else:
            return self.phibar3(x)

    def phibar1(self, s: Float64) -> Float64:
        var s1 = s - 1.0
        var s2 = s1 * sqrt(s1)
        return s2 * (4.0 / 3.0 + 1.2 * s1 - 253.0 / 70.0 * s1 * s1)

    def phibar2(self, s: Float64, w: Float64) -> Float64:
        var v = c_log(w)
        var u = 1.0 - 1.0 / s
        return (
            (2.0 - 4.0 * s) * sqrt(u)
            + v * (4.0 * c_log(1.0 + w) - 3.0 * v + s * (1.0 + u * u))
            - PI * PI / 3.0
            + self.phisum(w)
        )

    def phibar3(self, s: Float64) -> Float64:
        var w = c_log(4.0 * s)
        return (
            (2.0 * s + w) * (w - 2.0) + (w + 1.125) / s - (PI * PI / 3.0 - 3.0)
        )

    def phisum(self, w: Float64) -> Float64:
        var k = 0.0
        for i in range(1, 15):
            k += Float64(signf(i)) / (
                c_pow(w, Float64(i)) * Float64(i) * Float64(i)
            )
        return -4.0 * k

    def Calc_Pair_Lep(mut self, dat: AM3Arrays, max_com_energy_factor: Int):
        if self.verbose:
            print(" Calculate pair feedback full")
        for i in range(dat.nbins_lep):
            self.epspairL[i] = 0.0
        # The pairs take their number from the photon loss rates. Without the
        # photon loss kernel (the process off at init_kernels), there are no
        # pairs. So the full kernel is not built either (1.2 s and 330 MB on
        # the default grids).
        if self.KER_phibarD.rows == 0:
            return
        if not self.full_built:
            self.build_full_kernel(dat)
        for i in range(dat.nbins_pho):
            self.PP_NG[i] = (
                dat.n[P_photons][i] / dat.egrid_pho[i] / dat.egrid_pho[i]
            )
        var job = PairLepJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=dat)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            max_com_energy_factor,
        )
        parallel_run[pair_lep_work](job, self.nthreads)
        # The energy of all absorbed photons. (AM3 summed those above m_e c^2
        # only. So the pairs missed the energy of the softer partners, 6 % of
        # it for photons of 0.1 to 10 MeV: AM3 bug.)
        var r_alpha_phot = 0.0
        for i in range(dat.nbins_pho):
            r_alpha_phot += (
                dat.egrid_pho[i] * self.afpairG[i] * dat.n[P_photons][i]
            )
        var r_epspairl_raw = 0.0
        for i in range(dat.nbins_lep):
            r_epspairl_raw += dat.egrid_lep[i] * self.epspairL[i]
        if r_epspairl_raw > 1.0e-100:
            for i in range(dat.nbins_lep):
                self.epspairL[i] *= 0.5 * r_alpha_phot / r_epspairl_raw
        self.epspairL[0] = 0.0

    def Calc_Pair_Lep_OPT(mut self, dat: AM3Arrays):
        if self.verbose:
            print(" Calculate pair feedback 8bin")
        for i in range(dat.nbins_lep):
            self.epspairL[i] = 0.0
        var shift_neg: Array[Int, 7] = [0, 1, 3, 4, 7, 10, 17]
        for j in range(dat.ind_min_energy_pho, dat.nbins_pho):
            if j > dat.ind_min_energy_pho + 13:
                var N_phot = self.afpairG[j] * dat.n[P_photons][j]
                var N_span = 8.0
                var ind_mec2_half = j - dat.ind_min_energy_pho - 7
                for w in range(8):
                    var index_eplus = ind_mec2_half + w
                    if index_eplus >= 0 and index_eplus < dat.nbins_lep:
                        self.epspairL[index_eplus] += 0.5 * N_phot / N_span
                for w in range(7):
                    var index_eminus = ind_mec2_half - shift_neg[w]
                    if index_eminus >= 0 and index_eminus < dat.nbins_lep:
                        self.epspairL[index_eminus] += 0.5 * N_phot / N_span
        self.epspairL[0] = 0.0

    def innerPP(self, dat: AM3Arrays, x: Float64, gm: Float64) -> Int:
        if x <= 0.5:
            var f = self.Fpp_m(x, gm)
            # x below xL(gm), the threshold of the pairs: the first bin of
            # KER_outerPP can lie there. (AM3 takes the logarithm of a negative
            # number and converts the NaN to an int: undefined, 0 on arm64.)
            if f <= 0.0:
                return 0
            var arg = gm - x + 0.5 * (f + 1.0 / f)
            debug_assert(arg > 0.0, "innerPP: logarithm of arg <= 0")
            var q = Int(
                c_log(arg) / dat.dlnE + Float64(dat.ind_min_energy_pho) + 1.0
            )
            return min(max(q, 0), dat.nbins_pho)
        elif x > 0.5 and x < 1.0 and gm < self.gmB(x):
            var f = self.Fpp_m(x, gm)
            var q = Int(
                c_log(gm - x + 0.5 * (f + 1.0 / f)) / dat.dlnE
                + Float64(dat.ind_min_energy_pho)
                + 1.0
            )
            return min(max(q, 0), dat.nbins_pho)
        elif x > 1.0 and gm < self.gmB(x):
            # gm - x + (f + 1/f) / 2 as the same xL(gm) + 1 / (2 f). AM3's
            # difference cancels where x and gm are large, to a value of the
            # rounding, 0 or below (AM3 bug).
            var f = self.Fpp_p(x, gm)
            var q = Int(
                c_log(self.xL(gm) + 0.5 / f) / dat.dlnE
                + Float64(dat.ind_min_energy_pho)
                + 1.0
            )
            return min(max(q, 0), dat.nbins_pho)
        return 0

    def outerPP(self, dat: AM3Arrays, i: Int) -> Int:
        var q = Int(
            c_log(self.xL(dat.egrid_lep[i])) / dat.dlnE
            + Float64(dat.ind_min_energy_pho)
        )
        return min(max(q, 0), dat.nbins_pho - 1)

    def xL(self, gm: Float64) -> Float64:
        # gm (1 - beta) as 1 / (gm (1 + beta)). AM3's difference 1 - beta
        # cancels at large gm. It is 74 % too large at gm = 8.8e7, and 0 from
        # 9.5e7 on, where log(xL) is -inf (AM3 bug).
        return 0.5 / gm / (1.0 + beta(gm, self.REG))

    def gmB(self, x: Float64) -> Float64:
        return x - (x - 1.0) / (2.0 * x - 1.0)

    def Fpp_m(self, x: Float64, gp: Float64) -> Float64:
        # (gp (1 - beta) as in xL)
        return 2.0 * x - 1.0 / gp / (1.0 + beta(gp, self.REG))

    def Fpp_p(self, x: Float64, gp: Float64) -> Float64:
        return 2.0 * x - gp * (1.0 + beta(gp, self.REG))

    def RggD1(self, gm: Float64, x: Float64, x1: Float64) -> Float64:
        var gp = x + x1 - gm
        var x_hi: Float64
        var x_lo: Float64
        if gp > 10.0 and gm > 10.0:
            x_lo = (
                0.25 / gp
                + 0.09375 / gp / gp / gp
                + 0.25 / gm
                + 0.09375 / gm / gm / gm
            )
            x_hi = 0.5 * (gp + sqrt(gp * gp - 1.0) + gm + sqrt(gm * gm - 1.0))
        elif gp > 10.0 and gm <= 10.0:
            var sgm = sqrt(gm * gm - 1.0)
            x_lo = 0.5 * (0.5 / gp + gm - sgm)
            x_hi = gp + 0.5 * (gm + sgm)
        elif gp <= 10.0 and gm > 10.0:
            var sgp = sqrt(gp * gp - 1.0)
            x_lo = 0.5 * (gp - sgp + 0.5 / gm)
            x_hi = gm + 0.5 * (gp + sgp)
        else:
            var sgp = sqrt(gp * gp - 1.0)
            var sgm = sqrt(gm * gm - 1.0)
            x_lo = 0.5 * (gp - sgp + gm - sgm)
            x_hi = 0.5 * (gp + sgp + gm + sgm)
        if gp <= 1.000 or gm <= 1.000:
            return 0.0
        elif x < x_lo or x1 < x_lo:
            return 0.0
        elif x > x_hi or x1 > x_hi:
            return 0.0
        var T = x + x1
        var XUpper = self.XcmU(gm, x, x1)
        var XLower = self.XcmL(gm, x, x1)
        var u = 4.0 * XUpper * XUpper / T / T
        var v = 4.0 * XLower * XLower / T / T
        var a_nh = (gm - x) * (gm - x) - 1.0
        var a_ih = (gm - x1) * (gm - x1) - 1.0
        var h_NH_U = a_nh * XUpper * XUpper / (x * x1)
        var h_NH_L = a_nh * XLower * XLower / (x * x1)
        var h_IH_U = a_ih * XUpper * XUpper / (x * x1)
        var h_IH_L = a_ih * XLower * XLower / (x * x1)
        var TD2A = self.TD2(gm, x, x1, XUpper)
        var TD2B = self.TD2(gm, x, x1, XLower)
        var TD2C = self.TD2(gm, x1, x, XUpper)
        var TD2D = self.TD2(gm, x1, x, XLower)
        if h_NH_U > 1000.0 and h_NH_L > 1000.0:
            TD2A = -(XUpper * XUpper + 2.0 * c_log(XUpper)) / sqrt(a_nh)
            TD2B = -(XLower * XLower + 2.0 * c_log(XLower)) / sqrt(a_nh)
        if h_IH_U > 1000.0 and h_IH_L > 1000.0:
            TD2C = -(XUpper * XUpper + 2.0 * c_log(XUpper)) / sqrt(a_ih)
            TD2D = -(XLower * XLower + 2.0 * c_log(XLower)) / sqrt(a_ih)
        var td_sum = TD2A - TD2B + TD2C - TD2D
        if u < 0.01 and v < 0.01:
            return -0.25 * (0.5 * T * (u - v) + td_sum)
        return -0.25 * (T * (sqrt(1.0 - v) - sqrt(1.0 - u + 1.0e-9)) + td_sum)

    def XcmU(self, gm: Float64, x: Float64, x1: Float64) -> Float64:
        var gp = x + x1 - gm
        if gp > 15.0 and gm > 15.0:
            return min(sqrt(x * x1), sqrt(gp * gm))
        var t = gm * gp + 1.0 + sqrt(gm * gm - 1.0) * sqrt(gp * gp - 1.0)
        return min(sqrt(x * x1), sqrt(0.5 * t))

    def XcmL(self, gm: Float64, x: Float64, x1: Float64) -> Float64:
        var gp = x + x1 - gm
        if gp > 10.0:
            if gm > 10.0:
                return 0.5 * sqrt(2.0 + gp / gm + gm / gp)
            elif gm < 1.001:
                return 1.001
            else:
                var d = gm - sqrt(gm * gm - 1.0 + self.REG)
                return sqrt(0.5 * (1.0 + d * (gp - 0.5 / gp)))
        elif gp < 1.001:
            return 1.001
        else:
            if gm > 10.0:
                var d = gp - sqrt(gp * gp - 1.0 + self.REG)
                return sqrt(0.5 * (1.0 + d * (gm - 0.5 / gm)))
            elif gm < 1.001:
                return 1.001
            else:
                var r = sqrt((gm * gm - 1.0) * (gp * gp - 1.0))
                return sqrt(0.5 * (gp * gm + 1.0 - r))

    def TD2(
        self, gm: Float64, x: Float64, x1: Float64, xcm: Float64
    ) -> Float64:
        var y = sqrt(x * x1)
        var y2 = y * y
        var q = xcm / y
        var h = ((gm - x) * (gm - x) - 1.0) * q * q
        var b = (x * x + gm * (x1 - x)) / y2
        if h > 1000.0:
            var s = (
                c_log(h)
                + xcm * xcm
                + h / (xcm * xcm)
                + (2.0 * c_log(2.0) - 0.5)
                - 0.5 * b
            )
            return -s * q / sqrt(h)
        elif h < -0.999:
            return q * (
                0.5 * ((x1 - x) / y2 - 2.0 * PI) - 0.5 * PI * q * q * (y2 - 1.0)
            )
        var Ah = sqrt(1.0 + h)
        var t1 = q * q * q * (y2 - 1.0) * self.A_A0_h(h) - Ah / (xcm * y)
        var t2 = (1.0 - 2.0 * q * q + b) / Ah
        return t1 + 0.5 * q * (t2 - 4.0 * self.A0(h))

    def A_A0_h(self, h: Float64) -> Float64:
        if h <= 20.0:
            var i = 100 + Int(h / 0.01)
            return self.A_A0_hV[i]
        return -(1.0 - (c_log(2.0) + 0.5 * c_log(h)) / h) / sqrt(h)

    def A0(self, h: Float64) -> Float64:
        if h < 20.0:
            var i = 100 + Int(h / 0.01)
            return self.A0V[i]
        return (c_log(2.0) + 0.5 * c_log(h)) / sqrt(h)

    def A0f(self, h: Float64) -> Float64:
        if h > 1.0e-6:
            return c_log(sqrt(h) + sqrt(1.0 + h)) / sqrt(h)
        elif h < -1.0e-6:
            return c_asin(sqrt(-h) - 1.0e-8) / sqrt(-h)
        return 1.0 - h / 6.0

    def A_A0_hf(self, h: Float64) -> Float64:
        if h >= -0.1 and h <= 0.1:
            return -2.0 / 3.0 + h * (0.2 + h * (-3.0 / 28.0 + 5.0 / 72.0 * h))
        return (self.A0f(h) - sqrt(1.0 + h + 1.0e-8)) / h
