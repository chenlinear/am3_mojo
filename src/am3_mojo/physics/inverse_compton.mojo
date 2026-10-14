"""Port of Comp (InverseCompton.cc): IC photon emission/loss by e+/-."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.specfun import polylog
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.linalg import MatD, gemv_d, diagonal_gemv, dot


struct ICKernel2D(Copyable, Movable):
    """The IC emission kernel K[k](j, i) (outgoing photon k, electron j, target
    photon i) in factorised form.

    On AM3's logarithmic grids every cross-section branch is
    (1/gamma_j^2)(1/eps_i) * Phi(L, K) with L = 2 gamma eps and K = eps1/gamma,
    a function of s = i + j and d = k - j only. So
        K[k](j, i) = w[j] * u[i] * phi(d, s)   for i < min(k, ihi[j])
    (i < k: the photon gains energy; ihi[j]: the C++ zeroing of low-gamma
    columns). Entries that deviate are stored exactly as per-k exceptions.
    The small table makes the emission a cache-resident SIMD computation
    instead of a memory-bandwidth-bound stream.

    On the default grids phi has 860 x 620 Float64 values (4.27 MB). With the
    exceptions and index arrays, the kernel takes 4,357,888 bytes, against
    236,096,000 for the C++ kernel (one dense 340 x 280 float32 matrix per
    outgoing photon bin). Of the 49.5 million entries of the 520 matrices of
    the emitted bins, 2,745 are exceptions: 2,487 on the i == k diagonal,
    whose inclusion the C++ decides by round-off; 181 at i == k - 1 of the top
    electron bin, whose phi no other entry gives; and 77 just below the
    kinematic limit, which differ from the table by round-off (1-11e-12).
    init_kernels() with set_verbosity_level(True) prints the table's size and
    the number of exceptions."""

    var valid: Bool
    var npho: Int
    var nl: Int
    var ipho: Int
    var inmax: Int
    var outmin: Int
    var ns: Int  # s = i + j in [0, ns)
    var d0: Int  # d = k - j in [d0, d0 + nd)
    var nd: Int
    var phi: Vec  # phi[(d - d0) * ns + s]
    var slo: VecI  # phi(d, s) is zero outside s in [slo[d - d0], shi[d - d0])
    var shi: VecI
    var w: Vec  # 1 / gamma_j^2
    var u: Vec  # 1 / eps_i
    var ihi: VecI  # columns i >= ihi[j] are zero
    var ex_start: VecI  # exceptions of bin k: [ex_start[k], ex_start[k + 1])
    var ex_j: VecI
    var ex_i: VecI
    var ex_v: Vec

    def __init__(out self):
        self.valid = False
        self.npho = 0
        self.nl = 0
        self.ipho = 0
        self.inmax = 0
        self.outmin = 0
        self.ns = 0
        self.d0 = 0
        self.nd = 0
        self.phi = Vec()
        self.slo = VecI()
        self.shi = VecI()
        self.w = Vec()
        self.u = Vec()
        self.ihi = VecI()
        self.ex_start = VecI()
        self.ex_j = VecI()
        self.ex_i = VecI()
        self.ex_v = Vec()


struct ICJob(Movable):
    """Per-step IC emission, in two parallel phases:

    1. rows: for every d = k - j (outgoing photon minus electron bin)
           RT[d][j] = sum_i U[i] * phi(d, i + j)       (U_i = v_i / eps_i)
       over the allowed i. Where neither kinematic mask (i < k, i < ihi[j])
       cuts into the non-zero band, this is a Hankel product over j
       (diagonal_gemv, register-blocked); the few other rows use masked dot
       products.
    2. outputs: out_m[k] = pref_k * (sum_j RT[k - j][j] * w_j * E_m[j]
                                     + exception terms)."""

    var K: Pointer[ICKernel2D, MutUntrackedOrigin]
    var U: FPtr  # v_i * u_i
    var v: FPtr  # normalised target photons
    var RT: FPtr  # [(d - d0) * nl + j]
    var outs: Array[FPtr, N_LEP_IN]
    var ins: Array[FPtr, N_LEP_IN]
    var N_calc: Int
    var HaRN: Float64
    var dl2: Float64
    var egrid_pho: FPtr
    var elo: Int
    var ehi: Int
    var ulo: Int
    var uhi: Int
    var outmin: Int  # first emitted photon bin (current setting)

    def __init__(
        out self,
        K: Pointer[ICKernel2D, MutUntrackedOrigin],
        U: FPtr,
        v: FPtr,
        RT: FPtr,
        outs: Array[FPtr, N_LEP_IN],
        ins: Array[FPtr, N_LEP_IN],
        N_calc: Int,
        HaRN: Float64,
        dl2: Float64,
        egrid_pho: FPtr,
        elo: Int,
        ehi: Int,
        ulo: Int,
        uhi: Int,
        outmin: Int,
    ):
        self.K = K
        self.U = U
        self.v = v
        self.RT = RT
        self.outs = outs.copy()
        self.ins = ins.copy()
        self.N_calc = N_calc
        self.HaRN = HaRN
        self.dl2 = dl2
        self.egrid_pho = egrid_pho
        self.elo = elo
        self.ehi = ehi
        self.ulo = ulo
        self.uhi = uhi
        self.outmin = outmin


def ic_rows_work(job: Pointer[ICJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref J = job[]
    ref K = J.K[]
    var phi = pd(K.phi)
    var dd = t
    while dd < K.nd:
        var d = dd + K.d0
        # electrons whose outgoing bin k = d + j is emitted
        var jA = max(max(J.elo, J.outmin - d), 0)
        var jB = min(J.ehi, K.npho - d)
        var row = phi.unsafe_offset(dd * K.ns)
        var rt = J.RT.unsafe_offset(dd * K.nl)
        if d > K.ipho or jB <= jA:
            dd += nt
            continue
        var slo = K.slo[dd]
        var shi = K.shi[dd]
        # first electron from which neither mask cuts into the band
        var jstar = jA
        while jstar < jB:
            var iu = min(J.uhi, shi - jstar)
            if d + jstar >= iu and K.ihi[jstar] >= iu:
                break
            jstar += 1
        for j in range(jA, jstar):
            var i0 = max(J.ulo, slo - j)
            var i1 = min(min(min(d + j, K.ihi[j]), J.uhi), shi - j)
            rt[unsafe_offset=j] = 0.0
            if i1 > i0:
                rt[unsafe_offset=j] = dot(
                    J.U.unsafe_offset(i0), row.unsafe_offset(j + i0), i1 - i0
                )
        if jB > jstar and J.uhi > J.ulo:
            # rows r = j - jstar, columns c = i - ulo:
            # A[r, c] = phi(d, r + c + jstar + ulo)
            var off = jstar + J.ulo
            diagonal_gemv[True](
                row.unsafe_offset(off),
                slo - off,
                shi - off,
                jB - jstar,
                J.uhi - J.ulo,
                J.U.unsafe_offset(J.ulo),
                1.0,
                rt.unsafe_offset(jstar),
            )
        else:
            for j in range(jstar, jB):
                rt[unsafe_offset=j] = 0.0
        dd += nt


def ic_out_work(job: Pointer[ICJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref J = job[]
    ref K = J.K[]
    var k = J.outmin + t
    while k < K.npho:
        var jlo = max(max(k - K.ipho, 0), J.elo)
        var pref = J.HaRN * J.dl2 * J.egrid_pho[unsafe_offset=k]
        var s = Array[Float64, N_LEP_IN](fill=0.0)
        for j in range(jlo, J.ehi):
            var r = J.RT[unsafe_offset=(k - j - K.d0) * K.nl + j] * K.w[j]
            for m in range(J.N_calc):
                s[m] += r * J.ins[m][unsafe_offset=j]
        for x in range(K.ex_start[k], K.ex_start[k + 1]):
            var j = K.ex_j[x]
            if j >= jlo and j < J.ehi:
                var r = K.ex_v[x] * J.v[unsafe_offset=K.ex_i[x]]
                for m in range(J.N_calc):
                    s[m] += r * J.ins[m][unsafe_offset=j]
        for m in range(J.N_calc):
            J.outs[m][unsafe_offset=k] = s[m] * pref
        k += nt


struct Comp(Copyable, Movable):
    var verbose: Bool
    var aficG: Vec
    var K2: ICKernel2D
    var nthreads: Int
    var KER_IC_s_tot: MatD  # (nbins_pho x nbins_lep)
    var w_U: Vec
    var w_v: Vec
    var w_RT: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.aficG = Vec()
        self.K2 = ICKernel2D()
        self.nthreads = 1
        self.KER_IC_s_tot = MatD()
        self.w_U = Vec()
        self.w_v = Vec()
        self.w_RT = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.aficG, dat.nbins_pho)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        self.K2 = ICKernel2D()
        self.KER_IC_s_tot = MatD(dat.nbins_pho, dat.nbins_lep)

    def kernel_slice(
        self, dat: AM3Arrays, j: Int, inmax: Int, outmin: Int, mut sl: Vec
    ):
        """Electron column j of the original (C++) kernel, sl[k * inmax + i],
        evaluated in double precision exactly as InverseCompton.cc builds it."""
        var npho = dat.nbins_pho
        var ipho = dat.ind_min_energy_pho
        for x in range(npho * inmax):
            sl[x] = 0.0
        var mp = Array[Float64, 100](fill=0.0)
        for m in range(100):
            mp[m] = c_exp((-0.5 + 0.01 * Float64(m)) * dat.dlnE)
        var elec_in = dat.egrid_lep[j]
        for i in range(inmax):
            var phot_in = dat.egrid_pho[i]
            var L = 2.0 * phot_in * elec_in
            var kappa_max = 2.0 * L / (1.0 + 2.0 * L)
            var kappa_min = 0.5 * L / elec_in / elec_in
            if L < 10.0:
                for k in range(outmin, npho):
                    var kappa = dat.egrid_pho[k] / elec_in
                    if kappa_min < kappa and kappa < kappa_max:
                        if L < 0.001:
                            sl[k * inmax + i] = self.s_diff_th(
                                phot_in, dat.egrid_pho[k], elec_in
                            )
                        else:
                            sl[k * inmax + i] = self.s_diff_full(
                                phot_in, dat.egrid_pho[k], elec_in
                            )
            elif L < 1000.0:
                for k in range(outmin, npho):
                    var kappa = dat.egrid_pho[k] / elec_in
                    if kappa_min < kappa:
                        if -ipho + k <= j - 5:
                            sl[k * inmax + i] = self.s_diff_full(
                                phot_in, dat.egrid_pho[k], elec_in
                            )
                        elif -ipho + k <= j:
                            var y = 0.0
                            for m in range(100):
                                y += self.s_diff_full(
                                    phot_in, dat.egrid_pho[k] * mp[m], elec_in
                                )
                            sl[k * inmax + i] = 0.01 * y
            else:
                for k in range(max(1, outmin), npho):
                    var kappa = dat.egrid_pho[k] / elec_in
                    if kappa_min < kappa:
                        if -ipho + k < j:
                            sl[k * inmax + i] = self.s_diff_full(
                                phot_in, dat.egrid_pho[k], elec_in
                            )
                        elif -ipho + k == j:
                            sl[(k - 1) * inmax + i] += self.s_diff_KN_L(
                                dat, phot_in, elec_in
                            )
                            sl[k * inmax + i] += self.s_diff_KN_R(
                                dat, phot_in, elec_in
                            )
            if L > 2.0 * elec_in * elec_in - 0.5:
                for k in range(npho):
                    sl[k * inmax + i] = 0.0

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        rad: Bool,
        photonLoss: Bool,
        outmin: Int,
        inmax: Int,
    ):
        if rad:
            if self.verbose:
                print(
                    "computing electron IC photon generation kernel"
                    " (factorised)"
                )
            self.build_factorised_kernel(dat, inmax, outmin)
        if photonLoss:
            if self.verbose:
                print("computing IC photon loss kernel")
            for j in range(dat.nbins_lep):
                for i in range(dat.nbins_pho):
                    var L = 2.0 * dat.egrid_pho[i] * dat.egrid_lep[j]
                    if L < 0.001:
                        self.KER_IC_s_tot.set(i, j, self.s_tot_th(L))
                    elif 0.001 <= L and L <= 1000.0:
                        self.KER_IC_s_tot.set(i, j, self.s_tot_full(L))
                    else:
                        self.KER_IC_s_tot.set(i, j, self.s_tot_KN(L))

    def build_factorised_kernel(
        mut self, dat: AM3Arrays, inmax_in: Int, outmin: Int
    ):
        var npho = dat.nbins_pho
        var nl = dat.nbins_lep
        var inmax = min(inmax_in, npho)
        var K = ICKernel2D()
        K.npho = npho
        K.nl = nl
        K.ipho = dat.ind_min_energy_pho
        K.inmax = inmax
        K.outmin = outmin
        K.ns = inmax + nl
        K.d0 = outmin - nl
        K.nd = npho - K.d0
        K.phi = zeros(K.nd * K.ns)
        var have = zerosi(K.nd * K.ns)
        K.w = zeros(nl)
        K.u = zeros(inmax)
        K.ihi = zerosi(nl)
        for j in range(nl):
            K.w[j] = 1.0 / (dat.egrid_lep[j] * dat.egrid_lep[j])
        for i in range(inmax):
            K.u[i] = 1.0 / dat.egrid_pho[i]
        for j in range(nl):
            var g = dat.egrid_lep[j]
            var n = inmax
            for i in range(inmax):
                if 2.0 * dat.egrid_pho[i] * g > 2.0 * g * g - 0.5:
                    n = i
                    break
            K.ihi[j] = n
        var sl = zeros(npho * inmax)
        # pass 1: phi(d, s) from the first regular entry (i <= k - 2, below the
        # zeroed columns) of every (d, s)
        for j in range(nl):
            self.kernel_slice(dat, j, inmax, outmin, sl)
            for k in range(outmin, npho):
                var d = k - j
                for i in range(min(min(k - 1, K.ihi[j]), inmax)):
                    var v = sl[k * inmax + i]
                    var x = (d - K.d0) * K.ns + (i + j)
                    if v != 0.0 and have[x] == 0:
                        K.phi[x] = v / (K.w[j] * K.u[i])
                        have[x] = 1
        # pass 2: every entry the factorised form does not reproduce to 1e-12
        # becomes an exception
        var ek = VecI()
        for j in range(nl):
            self.kernel_slice(dat, j, inmax, outmin, sl)
            for k in range(outmin, npho):
                var d = k - j
                var n = min(min(k, K.ihi[j]), inmax)
                for i in range(inmax):
                    var orig = sl[k * inmax + i]
                    var pred = 0.0
                    if i < n:
                        pred = (
                            K.w[j] * K.u[i] * K.phi[(d - K.d0) * K.ns + (i + j)]
                        )
                    var diff = orig - pred
                    if c_fabs(diff) > 1e-12 * max(c_fabs(orig), c_fabs(pred)):
                        ek.append(k)
                        K.ex_j.append(j)
                        K.ex_i.append(i)
                        K.ex_v.append(diff)
        # non-zero band of every phi(d, .)
        K.slo = zerosi(K.nd)
        K.shi = zerosi(K.nd)
        for dd in range(K.nd):
            var lo = K.ns
            var hi = 0
            for x in range(K.ns):
                if K.phi[dd * K.ns + x] != 0.0:
                    lo = min(lo, x)
                    hi = x + 1
            K.slo[dd] = lo
            K.shi[dd] = hi
        # bucket the exceptions by k
        var cnt = zerosi(npho + 1)
        for x in range(len(ek)):
            cnt[ek[x] + 1] += 1
        for k in range(npho):
            cnt[k + 1] += cnt[k]
        K.ex_start = cnt.copy()
        var pos = cnt^
        var ej = zerosi(len(ek))
        var ei = zerosi(len(ek))
        var ev = zeros(len(ek))
        for x in range(len(ek)):
            var p = pos[ek[x]]
            ej[p] = K.ex_j[x]
            ei[p] = K.ex_i[x]
            ev[p] = K.ex_v[x]
            pos[ek[x]] += 1
        K.ex_j = ej^
        K.ex_i = ei^
        K.ex_v = ev^
        K.valid = True
        if self.verbose:
            print(
                "IC kernel: factorised table ",
                K.nd,
                " x ",
                K.ns,
                ", ",
                len(K.ex_v),
                " exceptions",
                sep="",
            )
        self.K2 = K^

    def Calc_IC_Phot_Rad(
        mut self,
        dat: AM3Arrays,
        outs: Array[FPtr, N_LEP_IN],
        ins: Array[FPtr, N_LEP_IN],
        only_first: Bool,
        outmin_now: Int,
        inmax_now: Int,
    ):
        """IC emission. The kernel covers the emitted photon bins from the
        minimum, and the target photon bins up to the maximum, of
        init_kernels. The current settings narrow them further, as in the
        C++, which loops over the current ranges."""
        if self.verbose:
            print(" Calculate IC Photon Emission term")
        var N_calc = 1 if only_first else N_LEP_IN
        var npho = dat.nbins_pho
        var nl = dat.nbins_lep
        for m in range(N_calc):
            for i in range(npho):
                outs[m][unsafe_offset=i] = 0.0
        if not self.K2.valid:
            return  # no kernel: no emission
        var RN = 1.0e-50
        for i in range(npho):
            RN += dat.n[P_photons][i]
        var inmax = self.K2.inmax
        var in_now = min(inmax, inmax_now)
        resize_zero(self.w_U, inmax)
        resize_zero(self.w_v, inmax)
        for i in range(in_now):
            self.w_v[i] = dat.n[P_photons][i] / RN
            self.w_U[i] = self.w_v[i] * self.K2.u[i]
        # electron bins where every input spectrum is zero contribute nothing,
        # and inputs that are zero everywhere are left out (their outputs stay
        # zero): the workers take the first n_in of ins_nz / outs_nz
        var elo = nl
        var ehi = 0
        var ins_nz = ins.copy()
        var outs_nz = outs.copy()
        var n_in = 0
        for m in range(N_calc):
            var e = ins[m]
            var nonzero = False
            for j in range(nl):
                if e[unsafe_offset=j] != 0.0:
                    elo = min(elo, j)
                    ehi = max(ehi, j + 1)
                    nonzero = True
            if nonzero:
                ins_nz[n_in] = ins[m]
                outs_nz[n_in] = outs[m]
                n_in += 1
        var ulo = inmax
        var uhi = 0
        for i in range(inmax):
            if self.w_U[i] != 0.0:
                ulo = min(ulo, i)
                uhi = i + 1
        resize_zero(self.w_RT, self.K2.nd * nl)
        var job = ICJob(
            Pointer(to=self.K2).unsafe_origin_cast[MutUntrackedOrigin](),
            pd(self.w_U),
            pd(self.w_v),
            pd(self.w_RT),
            outs_nz,
            ins_nz,
            n_in,
            RN,
            dat.dlnE * dat.dlnE,
            pd(dat.egrid_pho),
            elo,
            ehi,
            ulo,
            uhi,
            max(self.K2.outmin, outmin_now),
        )
        parallel_run[ic_rows_work](job, self.nthreads)
        parallel_run[ic_out_work](job, self.nthreads)

    def s_diff_th(self, e: Float64, z: Float64, g: Float64) -> Float64:
        var L = 2.0 * g * e
        var K = z / g
        var R = K / L
        var a1 = (R - 2.0) * (R + 1.0)
        var a2 = -2.0 * R * c_log(0.5 * R)
        var a3 = -0.75 / g / L
        var Kmax = 2.0 * L / (1.0 + 2.0 * L)
        if K < Kmax:
            return a3 * (a1 + a2)
        return 0.0

    def s_diff_full(self, e: Float64, z: Float64, g: Float64) -> Float64:
        var L = 2.0 * g * e
        var K = z / g
        var K_ = 1.0 - K
        var R = K / L
        var a1 = (R - 2.0 + 2.0 * K) * (R + 1.0 + 0.5 * K * K - K)
        var a2 = -2.0 * K_ * R * c_log(0.5 * R / K_)
        var a3 = -0.75 / g / L / K_ / K_
        var Kmax = 2.0 * L / (1.0 + 2.0 * L)
        if K < Kmax:
            return a3 * (a1 + a2)
        return 0.0

    def s_diff_KN_L(self, dat: AM3Arrays, e: Float64, g: Float64) -> Float64:
        var a1 = 0.375 / dat.dlnE / g / g / e
        var a2 = 1.0 / (1.0 - dat.dlnE)
        return a1 * a2

    def s_diff_KN_R(self, dat: AM3Arrays, e: Float64, g: Float64) -> Float64:
        var a1 = 0.375 / dat.dlnE / g / g / e
        var a2 = c_log(dat.dlnE * g * e) + 2.0 * c_log(2.0) - 2.0
        return a1 * a2

    def Calc_IC_Phot_Loss(mut self, dat: AM3Arrays):
        if self.verbose:
            print(" Calculate IC photon loss")
        if self.KER_IC_s_tot.rows == 0:
            fill_zero(self.aficG)
            return
        gemv_d(self.KER_IC_s_tot, pd(dat.n[P_pairs]), dat.dlnE, pd(self.aficG))

    def s_tot_th(self, L: Float64) -> Float64:
        return 1.0 - 1.333333 * L

    def s_tot_full(self, L: Float64) -> Float64:
        var a1 = -2.0 * L * (4.0 + 9.0 * L + L * L)
        var a2 = (
            (4.0 + L) * (1.0 + 2.0 * L) * (1.0 + 2.0 * L) * c_log(1.0 + 2.0 * L)
        )
        var a3 = 4.0 * L * (1.0 + 2.0 * L) * polylog(-2.0 * L)
        var a4 = 0.375 / L / L / L / (1.0 + 2.0 * L)
        return a4 * (a1 + a2 + a3)

    def s_tot_KN(self, L: Float64) -> Float64:
        return 0.75 * (c_log(2.0 * L) - 0.5) / L
