"""Port of Comp (InverseCompton.cc): IC photon emission/loss by e+/-."""

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import MatD, gemv_d
from am3_mojo.utils.parallel import parallel_run, parallel_tasks
from am3_mojo.utils.specfun import polylog

comptime QD = 8  # rows d of an emission block (the SIMD lanes)
comptime LJ = 4  # electron bins of an emission block
comptime QK = 4  # emitted photon bins of an output block (the SIMD lanes)
comptime UPAD = 8  # zeros on either side of the reversed target photons
comptime RAMP0 = 32  # Comp.ramp[x] = 1 for x > RAMP0, else 0


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

    The table keeps the rows d <= ipho, where bin k is not above the energy
    of bin j (the others are zero). They are stored in groups of 8 rows
    interleaved by s, the order in which ic_rows_work reads them. On the
    default grids that is 528 x 620 Float64 values (2.62 MB). With the
    exceptions and index arrays, the kernel takes 2,705,744 bytes, against
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
    var d0: Int  # d = k - j in [d0, d0 + nd), up to ipho
    var nd: Int
    var phi: Vec  # phi(d0 + 8 g + q, s) at [(g * ns + s) * 8 + q]
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
       over the allowed i, in increasing i. With s = i + j this is
       sum_s phi(d, s) U[s - j], a matrix product: ic_rows_work takes blocks
       of 8 rows d and 4 electron bins j, which reuse each loaded value of
       phi for 4 products and each of U for 8.
    2. outputs: out_m[k] = pref_k * (sum_j RT[k - j][j] * w_j * E_m[j]
                                     + exception terms), in increasing j."""

    var K: Pointer[ICKernel2D, MutUntrackedOrigin]
    var Ur: FPtr  # U[i] at ur0 - i, zero outside [ulo, uhi)
    var ur0: Int
    var ramp: FPtr
    var v: FPtr  # normalised target photons
    var RT: FPtr  # RT[d][j] at j * ndp + (d - d0)
    var ndp: Int
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
        Ur: FPtr,
        ur0: Int,
        ramp: FPtr,
        v: FPtr,
        RT: FPtr,
        ndp: Int,
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
        self.Ur = Ur
        self.ur0 = ur0
        self.ramp = ramp
        self.v = v
        self.RT = RT
        self.ndp = ndp
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


def ic_rows_work(job: Pointer[ICJob, MutUntrackedOrigin], g: Int):
    """RT of the rows d = d0 + 8 g + q (q < 8), in blocks of 4 electron bins.

    The 8 x 4 sums of a block run over the union s of their ranges; outside
    its own range a sum adds exact zeros (phi or U is zero there) except
    where the masks i < k and i < ihi[j] cut, at the end of the range. There
    the ramp table zeroes the cut lanes."""
    ref J = job[]
    ref K = J.K[]
    var dd0 = g * QD
    var jA = Array[Int, QD](fill=0)
    var jB = Array[Int, QD](fill=0)
    var nrow = min(QD, K.nd - dd0)
    # electrons whose outgoing bin k = d + j is emitted
    var jlo = K.nl
    var jhi = 0
    for q in range(nrow):
        var d = K.d0 + dd0 + q
        jA[q] = max(max(J.elo, J.outmin - d), 0)
        jB[q] = min(J.ehi, K.npho - d)
        if jB[q] > jA[q]:
            jlo = min(jlo, jA[q])
            jhi = max(jhi, jB[q])
    var zero = SIMD[DType.float64, QD](0)
    var lane = SIMD[DType.int64, QD](0, 1, 2, 3, 4, 5, 6, 7)
    # rows past the table stay zero: the outputs read them as zeros
    var inrow = lane.lt(SIMD[DType.int64, QD](nrow))
    var pg = pd(K.phi).unsafe_offset(g * K.ns * QD)
    var j0 = jlo
    while j0 < jhi:
        # the s ranges of the block's sums; from s_m on, masks cut
        var s_lo = K.ns
        var s_hi = 0
        var s_m = K.ns
        for q in range(nrow):
            var d = K.d0 + dd0 + q
            for j in range(max(j0, jA[q]), min(j0 + LJ, jB[q])):
                var i0 = max(J.ulo, K.slo[dd0 + q] - j)
                var mb = min(d + j, K.ihi[j])
                var i1 = min(min(J.uhi, K.shi[dd0 + q] - j), mb)
                s_m = min(s_m, mb + j)
                if i1 > i0:
                    s_lo = min(s_lo, i0 + j)
                    s_hi = max(s_hi, i1 + j)
        var a = Array[SIMD[DType.float64, QD], LJ](fill=zero)
        # U[s - j0 - l] at ub[l - s]
        var ub = J.Ur.unsafe_offset(J.ur0 + j0)
        var s = s_lo
        var s_end = min(s_m, s_hi)
        while s < s_end:
            var p = pg.unsafe_offset(s * QD).unsafe_load[width=QD]()
            var u = ub.unsafe_offset(-s)
            comptime for l in range(LJ):
                a[l] = p.fma(SIMD[DType.float64, QD](u[unsafe_offset=l]), a[l])
            s += 1
        if s < s_hi:
            # lane q of bin j = j0 + l takes phi while s < d + 2 j (i < k),
            # the ramp at d0 + dd0 + q + 2 j - s + RAMP0, and U while
            # s < ihi[j] + j
            var ihs = Array[Int, LJ](fill=0)
            comptime for l in range(LJ):
                ihs[l] = K.ihi[min(j0 + l, K.nl - 1)] + j0 + l
            var r0 = K.d0 + dd0 + 2 * j0 + RAMP0
            while s < s_hi:
                var p = pg.unsafe_offset(s * QD).unsafe_load[width=QD]()
                var u = ub.unsafe_offset(-s)
                comptime for l in range(LJ):
                    var x = min(max(r0 + 2 * l - s, 0), 2 * RAMP0 - QD)
                    var m = J.ramp.unsafe_offset(x).unsafe_load[width=QD]()
                    var ul = u[unsafe_offset=l] if s < ihs[l] else 0.0
                    a[l] = (p * m).fma(SIMD[DType.float64, QD](ul), a[l])
                s += 1
        var rt = J.RT.unsafe_offset(j0 * J.ndp + dd0)
        comptime for l in range(LJ):
            if j0 + l < jhi:
                rt.unsafe_offset(l * J.ndp).unsafe_store(
                    inrow.select(a[l], zero)
                )
        j0 += LJ


def _ic_out_block[n: Int](job: Pointer[ICJob, MutUntrackedOrigin], kb: Int):
    """Outputs k0 .. k0 + 3 of the first n inputs, k0 = outmin + 4 kb. The
    lanes start at the first electron bin of k0; for a later lane those
    below its own first bin read rows d > ipho, which are zero."""
    ref J = job[]
    ref K = J.K[]
    var k0 = J.outmin + kb * QK
    var jlo = max(max(k0 - K.ipho, 0), J.elo)
    var acc = Array[SIMD[DType.float64, QK], n](fill=SIMD[DType.float64, QK](0))
    # RT[k0 - j][j] at rk + j * (ndp - 1)
    var rk = J.RT.unsafe_offset(k0 - K.d0)
    for j in range(jlo, J.ehi):
        var r = (
            rk.unsafe_offset(j * (J.ndp - 1)).unsafe_load[width=QK]() * K.w[j]
        )
        comptime for m in range(n):
            acc[m] = r.fma(
                SIMD[DType.float64, QK](J.ins[m][unsafe_offset=j]), acc[m]
            )
    for l in range(min(QK, K.npho - k0)):
        var k = k0 + l
        var jl = max(max(k - K.ipho, 0), J.elo)
        var pref = J.HaRN * J.dl2 * J.egrid_pho[unsafe_offset=k]
        var s = Array[Float64, n](fill=0.0)
        comptime for m in range(n):
            s[m] = acc[m][l]
        for x in range(K.ex_start[k], K.ex_start[k + 1]):
            var j = K.ex_j[x]
            if j >= jl and j < J.ehi:
                var r = K.ex_v[x] * J.v[unsafe_offset=K.ex_i[x]]
                comptime for m in range(n):
                    s[m] += r * J.ins[m][unsafe_offset=j]
        comptime for m in range(n):
            J.outs[m][unsafe_offset=k] = s[m] * pref


def ic_out_work(job: Pointer[ICJob, MutUntrackedOrigin], t: Int, nt: Int):
    comptime assert N_LEP_IN <= 8, "the register blocks have up to 8 inputs"
    var nkb = (job[].K[].npho - job[].outmin + QK - 1) // QK
    var kb = t
    while kb < nkb:
        # the inputs' register block, sized at compile time
        var n = job[].N_calc
        if n == 1:
            _ic_out_block[1](job, kb)
        elif n == 2:
            _ic_out_block[2](job, kb)
        elif n == 3:
            _ic_out_block[3](job, kb)
        elif n == 4:
            _ic_out_block[4](job, kb)
        elif n == 5:
            _ic_out_block[5](job, kb)
        elif n == 6:
            _ic_out_block[6](job, kb)
        elif n == 7:
            _ic_out_block[7](job, kb)
        else:
            _ic_out_block[8](job, kb)
        kb += nt


struct ICBuildJob(Movable):
    """build_factorised_kernel's sweeps over the electron bins j = t, t + nt,
    ...: per thread t, the first of its bins with a regular value of every
    table entry (first: entry x at t * nx + x, -1 for none); the table
    (written by that first bin of all threads, first[x]); per bin its
    exceptions."""

    var C: Pointer[Comp, MutUntrackedOrigin]
    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]
    var K: Pointer[ICKernel2D, MutUntrackedOrigin]
    var inmax: Int
    var outmin: Int
    var first: List[Int32]
    var table: FPtr
    var ex_k: List[VecI]
    var ex_i: List[VecI]
    var ex_v: List[Vec]

    def __init__(
        out self,
        C: Pointer[Comp, MutUntrackedOrigin],
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
        K: Pointer[ICKernel2D, MutUntrackedOrigin],
        inmax: Int,
        outmin: Int,
        nt: Int,
        nl: Int,
    ):
        self.C = C
        self.dat = dat
        self.K = K
        self.inmax = inmax
        self.outmin = outmin
        var nx = K[].nd * K[].ns
        self.first = List[Int32](length=nt * nx, fill=-1)
        self.table = pd(K[].w)  # (the table, set before it is filled)
        self.ex_k = List[VecI](length=nl, fill=VecI())
        self.ex_i = List[VecI](length=nl, fill=VecI())
        self.ex_v = List[Vec](length=nl, fill=Vec())


def ic_build_first_work(
    job: Pointer[ICBuildJob, MutUntrackedOrigin], t: Int, nt: Int
):
    """Sweep 1: thread t's first bin of every table entry; on one thread,
    that bin writes the entry too (sweep 2 is then not needed)."""
    ref J = job[]
    ref K = J.K[]
    ref dat = J.dat[]
    var npho = K.npho
    var inmax = J.inmax
    var first = J.first.unsafe_ptr().unsafe_offset(t * K.nd * K.ns)
    var sl = zeros(npho * inmax)
    var j = t
    while j < K.nl:
        J.C[].kernel_slice(dat, j, inmax, J.outmin, sl)
        for k in range(J.outmin, min(npho, j + K.ipho + 1)):
            var d = k - j
            for i in range(min(min(k - 1, K.ihi[j]), inmax)):
                var x = (d - K.d0) * K.ns + (i + j)
                var v = sl[k * inmax + i]
                if v != 0.0 and first[unsafe_offset=x] < 0:
                    first[unsafe_offset=x] = Int32(j)
                    if nt == 1:
                        J.table[unsafe_offset=x] = v / (K.w[j] * K.u[i])
        j += nt


def ic_build_table_work(
    job: Pointer[ICBuildJob, MutUntrackedOrigin], t: Int, nt: Int
):
    """Sweep 2: each table entry from its first bin, first[x] (one bin
    writes it)."""
    ref J = job[]
    ref K = J.K[]
    ref dat = J.dat[]
    var npho = K.npho
    var inmax = J.inmax
    var sl = zeros(npho * inmax)
    var j = t
    while j < K.nl:
        J.C[].kernel_slice(dat, j, inmax, J.outmin, sl)
        for k in range(J.outmin, min(npho, j + K.ipho + 1)):
            var d = k - j
            for i in range(min(min(k - 1, K.ihi[j]), inmax)):
                var x = (d - K.d0) * K.ns + (i + j)
                if Int(J.first[x]) == j:
                    J.table[unsafe_offset=x] = sl[k * inmax + i] / (
                        K.w[j] * K.u[i]
                    )
        j += nt


def ic_build_exceptions_work(
    job: Pointer[ICBuildJob, MutUntrackedOrigin], t: Int, nt: Int
):
    ref J = job[]
    ref K = J.K[]
    ref dat = J.dat[]
    var npho = K.npho
    var inmax = J.inmax
    var sl = zeros(npho * inmax)
    var j = t
    while j < K.nl:
        J.C[].kernel_slice(dat, j, inmax, J.outmin, sl)
        for k in range(J.outmin, npho):
            var d = k - j
            var n = min(min(k, K.ihi[j]), inmax) if d <= K.ipho else 0
            for i in range(inmax):
                var orig = sl[k * inmax + i]
                var pred = 0.0
                if i < n:
                    pred = (
                        K.w[j]
                        * K.u[i]
                        * J.table[unsafe_offset=(d - K.d0) * K.ns + (i + j)]
                    )
                var diff = orig - pred
                if abs(diff) > 1e-12 * max(abs(orig), abs(pred)):
                    J.ex_k[j].append(k)
                    J.ex_i[j].append(i)
                    J.ex_v[j].append(diff)
        j += nt


struct Comp(Copyable, Movable):
    var verbose: Bool
    var aficG: Vec
    var K2: ICKernel2D
    var nthreads: Int
    var KER_IC_s_tot: MatD  # (nbins_pho x nbins_lep)
    var ramp: Vec  # 0 up to RAMP0, then 1 (the masks of ic_rows_work)
    var w_Ur: Vec
    var w_v: Vec
    var w_RT: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.aficG = Vec()
        self.K2 = ICKernel2D()
        self.nthreads = 1
        self.KER_IC_s_tot = MatD()
        self.ramp = zeros(2 * RAMP0)
        for x in range(RAMP0 + 1, 2 * RAMP0):
            self.ramp[x] = 1.0
        self.w_Ur = Vec()
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
        K.nd = K.ipho - K.d0 + 1
        # phi[(d - d0) * ns + s] while it is built
        var phi = zeros(K.nd * K.ns)
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
        # pass 1: phi(d, s) from the first regular entry (i <= k - 2, below the
        # zeroed columns) of every (d, s). The electron bins run in parallel:
        # each thread notes the first of its bins that has the entry, and the
        # first of all bins then writes it
        var nt = max(1, min(self.nthreads, nl))
        var nx = K.nd * K.ns
        var job = ICBuildJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=dat)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=K).unsafe_origin_cast[MutUntrackedOrigin](),
            inmax,
            outmin,
            nt,
            nl,
        )
        job.table = pd(phi)
        parallel_run[ic_build_first_work](job, nt)
        if nt > 1:
            for x in range(nx):
                var first = -1
                for t in range(nt):
                    var j = Int(job.first[t * nx + x])
                    if j >= 0 and (first < 0 or j < first):
                        first = j
                job.first[x] = Int32(first)
            job.first.resize(nx, 0)
            parallel_run[ic_build_table_work](job, nt)
        job.first = List[Int32]()
        # pass 2: every entry the factorised form does not reproduce to 1e-12
        # becomes an exception (in parallel, kept in the order of the bins)
        parallel_run[ic_build_exceptions_work](job, nt)
        var ek = VecI()
        for j in range(nl):
            for x in range(len(job.ex_k[j])):
                ek.append(job.ex_k[j][x])
                K.ex_j.append(j)
                K.ex_i.append(job.ex_i[j][x])
                K.ex_v.append(job.ex_v[j][x])
        # non-zero band of every phi(d, .)
        K.slo = zerosi(K.nd)
        K.shi = zerosi(K.nd)
        for dd in range(K.nd):
            var lo = K.ns
            var hi = 0
            for x in range(K.ns):
                if phi[dd * K.ns + x] != 0.0:
                    lo = min(lo, x)
                    hi = x + 1
            K.slo[dd] = lo
            K.shi[dd] = hi
        # groups of 8 rows interleaved by s (ic_rows_work), zero-padded
        var ng = (K.nd + QD - 1) // QD
        K.phi = zeros(ng * K.ns * QD)
        for dd in range(K.nd):
            for x in range(K.ns):
                K.phi[((dd // QD) * K.ns + x) * QD + dd % QD] = phi[
                    dd * K.ns + x
                ]
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
        self.w_RT = Vec()  # (rows past the table must start at zero)
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
        resize_zero(self.w_v, inmax)
        # U_i = v_i / eps_i, reversed (U[i] at ur0 - i) and padded with zeros
        var ur0 = inmax + UPAD - 1
        resize_zero(self.w_Ur, inmax + 2 * UPAD)
        var ulo = inmax
        var uhi = 0
        for i in range(in_now):
            self.w_v[i] = dat.n[P_photons][i] / RN
            var U = self.w_v[i] * self.K2.u[i]
            self.w_Ur[ur0 - i] = U
            if U != 0.0:
                ulo = min(ulo, i)
                uhi = i + 1
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
        # RT of the rows d0 .. d0 + ndp - 1 (those past the table are zero)
        var ng = (self.K2.nd + QD - 1) // QD
        var ndp = (ng + 1) * QD
        if len(self.w_RT) != nl * ndp:
            self.w_RT = zeros(nl * ndp)
        var job = ICJob(
            Pointer(to=self.K2).unsafe_origin_cast[MutUntrackedOrigin](),
            pd(self.w_Ur),
            ur0,
            pd(self.ramp),
            pd(self.w_v),
            pd(self.w_RT),
            ndp,
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
        parallel_tasks[ic_rows_work](job, ng, self.nthreads)
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
