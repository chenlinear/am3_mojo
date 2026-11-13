"""Port of BetheHeitler (BetheHeitler.cc): p + gamma -> p + e+ e-.

init_kernels packs the pair-emission kernel straight into its non-zero row
windows (PackedGemvF), which the emission reads sequentially. The C++ code
streams the dense kernel (about 120 MB on the default grids) every step. Where
the photon energy in the proton's frame reaches omega = 600, the kernel
factorizes (the saturated part). On the default grids, this part holds four
fifths of the non-zero entries of the kernel, so the emission never reads these
entries."""

from std.math import sqrt
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import MatD, gemv_d, PackedGemvF
from am3_mojo.utils.parallel import parallel_run, parallel_tasks


@inline(.always)
def bh_sigma_W(omega: Float64, E_: Float64, ksi: Float64) -> Float64:
    if omega < 2.0001 or omega > 600.0:
        return 0.0
    var REG = 1.0e-30
    var k = omega
    var g = E_
    var u = ksi
    var p = sqrt(g * g - 1.0 + REG)
    var g1 = k - g
    var p1 = sqrt(g1 * g1 - 1.0 + REG)
    var D = g - u * p
    var T = sqrt(k * k + p * p - 2.0 * k * p * u)
    var d1T = c_log((T + p1) / (T - p1))
    var y1 = c_log((g1 + p1) / (g1 - p1)) / p1
    var Y = 2.0 * c_log((g * g1 + p * p1 + 1.0) / k) / (p * p)
    var A0 = p * p1 / (k * k * k)
    var A1 = (
        -4.0 * (1.0 - u * u) * (2.0 * g * g + 1.0) / (p * p * D * D * D * D)
    )
    var A2 = (5.0 * g * g - 2.0 * g * g1 + 3.0) / (p * p * D * D)
    var A3 = (p * p - k * k) / (T * T * D * D)
    var A4 = 2.0 * g1 / (p * p * D)
    var A50 = Y / (p * p1)
    var A51 = 2.0 * g * (1.0 - u * u) * (3.0 * k + p * p * g1) / (D * D * D * D)
    var A52 = (
        (
            2.0 * g * g * (g * g + g1 * g1)
            - 7.0 * g * g
            - 3.0 * g * g1
            - g1 * g1
            + 1.0
        )
        / D
        / D
    )
    var A53 = k * (g * g - g * g1 - 1.0) / D
    var A6 = -(d1T / p1 / T) * (
        2.0 / D / D - 3.0 * k / D - k * (p * p - k * k) / T / T / D
    )
    var A7 = -2.0 * y1 / D
    return A0 * (A1 + A2 + A3 + A4 + A50 * (A51 + A52 + A53) + A6 + A7)


@inline(.always)
def bh_inner(gp: Float64, Ee: Float64, omega: Float64, E_: Float64) -> Float64:
    var REG = 1.0e-20
    var p_ = sqrt(E_ * E_ - 1.0 + REG)
    var ksi = (gp * E_ - Ee) / (gp * p_)
    if ksi < -1.0 or ksi > 1.0 or E_ < 1.0 or E_ > omega - 1.0:
        return 0.0
    return omega * bh_sigma_W(omega, E_, ksi) / p_


def bh_outer(gp: Float64, Ee: Float64, omega: Float64) -> Float64:
    var REG = 1.0e-5
    var lower = (gp * gp + Ee * Ee) / (2.0 * gp * Ee) + REG
    var upper = omega - 1.0 - REG
    var diff = (upper - lower) / lower
    var tolerance = 1.0e-4
    # (omega >= (gp + Ee)^2 / (2 gp Ee), the lower limit of the integral
    #  over it. So diff >= -2 REG, and AM3's error for diff < -tolerance
    #  cannot occur.)
    if -tolerance < diff and diff < tolerance:
        return 0.0
    # RK4_4(inner, lower, upper, gp, Ee, omega, 5)
    comptime N_BINS = 5
    var DX = (upper - lower) / Float64(N_BINS)
    var x = lower
    var y = 0.0
    for _ in range(N_BINS):
        var k1 = DX * bh_inner(gp, Ee, omega, x)
        var k2 = DX * bh_inner(gp, Ee, omega, x + 0.5 * DX)
        var k4 = DX * bh_inner(gp, Ee, omega, x + DX)
        y += (1.0 / 6.0) * (k1 + k4) + (2.0 / 3.0) * k2
        x += DX
    return y


def bh_kernal_elec_gen(Ee: Float64, gp: Float64, Eph: Float64) -> Float64:
    # The photon energy in the proton's frame, omega, runs from the pair
    # threshold to head-on collisions. But omega does not go beyond 600, where
    # bh_sigma_W ends (AM3's Born limit). AM3 spread its 5 steps over the
    # whole range. Above 2 gp Eph ~ 600, these steps sampled the part below
    # 600 ever more coarsely, and above ~8000 not at all. So the pairs
    # vanished while the protons kept losing the energy.
    var upper = min(2.0 * gp * Eph, 600.0)
    var lower = (gp + Ee) * (gp + Ee) / (2.0 * gp * Ee)
    var diff = (upper - lower) / lower
    var tolerance = 1.0e-8
    if diff < tolerance:
        return 0.0
    # RK4_3(outer, lower, upper, gp, Ee, 5)
    comptime N_BINS = 5
    var DX = (upper - lower) / Float64(N_BINS)
    var x = lower
    var y = 0.0
    for n in range(N_BINS):
        var k1 = DX * bh_outer(gp, Ee, x)
        var k2 = DX * bh_outer(gp, Ee, x + 0.5 * DX)
        # (Summed over the steps, x + DX can round to slightly above 600, where
        #  bh_sigma_W is zero.)
        var xe = upper if n == N_BINS - 1 else x + DX
        var k4 = DX * bh_outer(gp, Ee, xe)
        y += (1.0 / 6.0) * (k1 + k4) + (2.0 / 3.0) * k2
        x += DX
    return y * Ee / 2.0 / gp / gp / gp / Eph / Eph


def bh_eloss_kernal_semiana_phi(x: Float64) -> Float64:
    var y = 0.0
    if x < 2.0 + 1.0e-6:
        pass
    elif x < 25.0:
        var z = x - 2.0
        y = (
            (PI / 12.0)
            * z
            * z
            * z
            * z
            / (
                1.0
                + 0.8048 * z
                + 0.1459 * z * z
                + 1.137e-3 * z * z * z
                - 3.879e-6 * z * z * z * z
            )
        )
    else:
        var z = c_log(x)
        y = (
            x
            * (-86.07 + 50.96 * z - 14.45 * z * z + 8.0 / 3.0 * z * z * z)
            / (1.0 - 2.910 / x - 78.35 / x / x - 1837.0 / x / x / x)
        )
    return y


def bh_kernal_proton_loss(gp: Float64, Eph: Float64) -> Float64:
    return (
        0.5 * bh_eloss_kernal_semiana_phi(2.0 * gp * Eph) / gp / gp / Eph / Eph
    )


struct BHInitJob(Movable):
    """The saturated diagonals s >= sat_s stay zero (they are the factorized
    part)."""

    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]
    var T: F32Ptr
    var proton_in_min: Int
    var kmax: Int
    var d0: Int
    var s0: Int
    var ns: Int
    var sat_s: Int

    def __init__(
        out self,
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
        T: F32Ptr,
        proton_in_min: Int,
        kmax: Int,
        d0: Int,
        s0: Int,
        ns: Int,
        sat_s: Int,
    ):
        self.dat = dat
        self.T = T
        self.proton_in_min = proton_in_min
        self.kmax = kmax
        self.d0 = d0
        self.s0 = s0
        self.ns = ns
        self.sat_s = sat_s


def bh_table_work(job: Pointer[BHInitJob, MutUntrackedOrigin], r: Int):
    ref J = job[]
    ref dat = J.dat[]
    var d = J.d0 + r
    var jlo = max(J.proton_in_min, -d)
    var jhi = min(dat.nbins_had, dat.nbins_lep - d)
    if jhi <= jlo:
        return
    for s in range(jlo, min(jhi + J.kmax - 1, J.sat_s)):
        var j = max(jlo, s - J.kmax + 1)
        J.T[unsafe_offset=r * J.ns + s - J.s0] = Float32(
            bh_kernal_elec_gen(
                dat.egrid_had_leptons[d + j] / MASS_E,
                dat.egrid_had[j] / MASS_P,
                dat.egrid_had_photons[s - j] / MASS_E,
            )
        )


@inline(.always)
def bh_table_entry(
    job: Pointer[BHInitJob, MutUntrackedOrigin], i: Int, j: Int, k: Int
) -> Float32:
    ref J = job[]
    if j < J.proton_in_min or k >= J.kmax:
        return 0.0
    return J.T[unsafe_offset=(i - j - J.d0) * J.ns + j + k - J.s0]


struct BHJob(Movable):
    var K: Pointer[PackedGemvF, MutUntrackedOrigin]
    var v: F32Ptr
    var protons: FPtr
    var sat: FPtr
    var out_: FPtr
    var prinmin: Int
    var nh: Int
    var nout: Int
    var HaRN: Float64
    var dlnE: Float64
    var k_inj: Float64

    def __init__(
        out self,
        K: Pointer[PackedGemvF, MutUntrackedOrigin],
        v: F32Ptr,
        protons: FPtr,
        sat: FPtr,
        out_: FPtr,
        prinmin: Int,
        nh: Int,
        nout: Int,
        HaRN: Float64,
        dlnE: Float64,
        k_inj: Float64,
    ):
        self.K = K
        self.v = v
        self.protons = protons
        self.sat = sat
        self.out_ = out_
        self.prinmin = prinmin
        self.nh = nh
        self.nout = nout
        self.HaRN = HaRN
        self.dlnE = dlnE
        self.k_inj = k_inj


def bh_rad_work(job: Pointer[BHJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref J = job[]
    var buf = zerosf(J.K[].res_len())
    var res = pf(buf)
    # proton bins with exactly zero density only add exact zeros: skip them
    var plo: Int
    var phi: Int
    plo, phi = nonzero_range(J.protons, J.nh, J.prinmin)
    var i = t
    while i < J.nout:
        var x = 0.0
        if phi > plo:
            J.K[].apply(i, J.v, Float32(1.0), res, plo, phi)
        for j in range(plo, phi):
            x += Float64(res[unsafe_offset=j]) * J.protons[unsafe_offset=j]
        x += J.sat[unsafe_offset=i]
        J.out_[unsafe_offset=i] = J.HaRN * x * J.dlnE * J.dlnE * J.k_inj
        i += nt
    _ = buf^


struct BHCoolJob(Movable):
    var B: Pointer[BetheHeitler, MutUntrackedOrigin]
    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]

    def __init__(
        out self,
        B: Pointer[BetheHeitler, MutUntrackedOrigin],
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
    ):
        self.B = B
        self.dat = dat


def bh_cool_work(job: Pointer[BHCoolJob, MutUntrackedOrigin], t: Int, nt: Int):
    """SIMD runs over the protons j, and the photons i are accumulated in the
    original order. A masked lane skips photons below threshold (Eph < 1/gp),
    exactly as the scalar `continue` does."""
    ref B = job[].B[]
    ref dat = job[].dat[]
    var k_rate_semiana = (
        (3.0 / 8.0 / PI) * ALPHA * SIGMA_T * C_LIGHT * (MASS_E / MASS_P)
    )
    var K = B.KER_BH_LOSS.ptr()
    var nh = B.KER_BH_LOSS.rows
    var ph = pd(dat.n[P_photons_on_had_grid])
    var nb = dat.nbins_had
    comptime W = WD
    var b = t
    while b <= nb // W:
        var j = b * W
        if j + W <= nb:
            var ginv = SIMD[DType.float64, W](0)
            for l in range(W):
                ginv[l] = 1.0 / (dat.egrid_had[j + l] / MASS_P)
            var rj = SIMD[DType.float64, W](0)
            for i in range(dat.nbins_pho):
                var Eph = dat.egrid_had_photons[i] / MASS_E
                var keep = SIMD[DType.float64, W](Eph).ge(ginv)
                var k = K.unsafe_offset(i * nh + j).unsafe_load[width=W]()
                rj = keep.select(rj + k * ph[unsafe_offset=i], rj)
            for l in range(W):
                B.ABHHadP[j + l] = -rj[l] * k_rate_semiana * dat.dlnE
        else:
            for jj in range(j, nb):
                var rj = 0.0
                var gp = dat.egrid_had[jj] / MASS_P
                for i in range(dat.nbins_pho):
                    var Eph = dat.egrid_had_photons[i] / MASS_E
                    if Eph < 1.0 / gp:
                        continue
                    rj += K[unsafe_offset=i * nh + jj] * ph[unsafe_offset=i]
                B.ABHHadP[jj] = -rj * k_rate_semiana * dat.dlnE
        b += nt


struct BetheHeitler(Copyable, Movable):
    var verbose: Bool
    var QBH_LGen: Vec
    var ABHHadP: Vec
    var KER_BH_LOSS: MatD
    # voM_KER_BH_LGen, packed
    var packed: PackedGemvF
    var nthreads: Int
    var sparse_prinmin: Int
    var w_vphot: VecF
    var sat_B: MatD
    var sat_s: Int
    var sat_kmax: Int
    var inv_gp2: Vec
    var inv_eph2: Vec
    var u_sat: Vec
    var w_sat: Vec
    var q_sat: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.QBH_LGen = Vec()
        self.ABHHadP = Vec()
        self.KER_BH_LOSS = MatD()
        self.packed = PackedGemvF()
        self.nthreads = 1
        self.sparse_prinmin = -1
        self.w_vphot = VecF()
        self.sat_B = MatD()
        self.sat_s = 0
        self.sat_kmax = 0
        self.inv_gp2 = Vec()
        self.inv_eph2 = Vec()
        self.u_sat = Vec()
        self.w_sat = Vec()
        self.q_sat = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.QBH_LGen, dat.nbins_lep)
        resize_zero(self.ABHHadP, dat.nbins_had)
        resize_zero(self.w_sat, dat.nbins_had)
        resize_zero(self.q_sat, dat.nbins_lep)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        # (With the process off at init_kernels, the emission and the cooling
        #  switched on later make no pairs and no loss. AM3's zero kernels do
        #  the same but take 120 MB on the default grids.)
        self.packed = PackedGemvF()
        self.sat_B = MatD()
        self.sat_kmax = 0
        self.KER_BH_LOSS = MatD()

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        proton_in_min: Int,
        photon_in_max: Int,
    ):
        """Both kernels, whatever the emission and cooling switches, so that a
        switch turned on after init_kernels finds its kernel. (AM3 built each
        kernel only with its switch on, so cooling switched on later made pairs
        with no proton loss.)

        The pair kernel's entry of e+- bin i, proton bin j and photon bin k
        depends on the energies only through Ee / gp and gp Eph. The three grids
        share dlnE, so the entry is a function of i - j and j + k. Its values
        are computed once per pair (i - j, j + k). Where they saturate (below),
        they are computed once per i - j. On the default grids, this gives
        0.13 million values for the 23 million entries, which AM3 computes one
        by one. The threads take the rows of this table in turn, as their cost
        varies. The kernel's non-zero row windows are packed straight from the
        table (PackedGemvF.pack), so the entries are never stored densely.

        Where 2 gp Eph >= 600, the upper limit of the integral over omega is 600
        itself. So the entry is B(i - j) / (gp Eph)^2, with B = y Ee / (2 gp) of
        the integral y."""
        self.sparse_prinmin = max(proton_in_min, 0)
        if self.verbose:
            print(
                "computing BH e+/- generation kernel. Using proton_in_min: ",
                proton_in_min,
                ", photon_in_max: ",
                photon_in_max,
                sep="",
            )
        self.packed = PackedGemvF()
        self.sat_B = MatD()
        self.sat_kmax = 0
        var pmin = self.sparse_prinmin
        var kmax = min(photon_in_max, dat.nbins_pho_hadgrid)
        if pmin < dat.nbins_had and kmax > 0:
            var d0 = 1 - dat.nbins_had
            var nd = dat.nbins_lep - pmin - d0
            var ns = dat.nbins_had - pmin + kmax - 1
            var sat_s = self._saturated_diagonal(dat, pmin, kmax, ns)
            var T = zerosf(nd * ns)
            var job = BHInitJob(
                Pointer(to=dat)
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutUntrackedOrigin](),
                pf(T),
                pmin,
                kmax,
                d0,
                pmin,
                ns,
                sat_s,
            )
            var nt = self.nthreads
            parallel_tasks[bh_table_work](job, nd, nt)
            # every row from proton_in_min: the minimum of each emission call
            # only bounds its sum below (as in the C++)
            var nmat = dat.nbins_lep
            var row_min = zerosi(nmat)
            for i in range(nmat):
                row_min[i] = pmin
            self.packed = PackedGemvF.pack[bh_table_entry](
                job, nmat, dat.nbins_had, photon_in_max, row_min, nt
            )
            _ = T^  # (read by the workers through a pointer)
            self._saturated_part(dat, pmin, kmax, d0, nd, sat_s)
        if self.verbose:
            print("computing BH proton loss kernel")
        self.KER_BH_LOSS = MatD(dat.nbins_had, dat.nbins_pho)
        for i in range(dat.nbins_had):
            for j in range(dat.nbins_pho):
                self.KER_BH_LOSS.set(
                    i,
                    j,
                    bh_kernal_proton_loss(
                        dat.egrid_had[i] / MASS_P,
                        dat.egrid_had_photons[j] / MASS_E,
                    ),
                )

    def _saturated_diagonal(
        self, dat: AM3Arrays, pmin: Int, kmax: Int, ns: Int
    ) -> Int:
        """gp Eph grows along j + k."""
        for s in range(pmin, pmin + ns):
            var saturated = True
            for j in range(max(pmin, s - kmax + 1), min(dat.nbins_had, s + 1)):
                var gp = dat.egrid_had[j] / MASS_P
                if 2.0 * gp * (dat.egrid_had_photons[s - j] / MASS_E) < 600.0:
                    saturated = False
                    break
            if saturated:
                return s
        return pmin + ns

    def _saturated_part(
        mut self,
        dat: AM3Arrays,
        pmin: Int,
        kmax: Int,
        d0: Int,
        nd: Int,
        sat_s: Int,
    ):
        """Every 2 gp Eph >= 600 gives the same integral over omega."""
        var nh = dat.nbins_had
        var nl = dat.nbins_lep
        var t = zeros(nl + nh - 1)
        for r in range(nd):
            var d = d0 + r
            var jlo = max(pmin, -d)
            if jlo >= min(nh, nl - d):
                continue
            var gp = dat.egrid_had[jlo] / MASS_P
            var gpe = 600.0  # gp Eph
            var Ee = dat.egrid_had_leptons[d + jlo] / MASS_E
            t[r] = bh_kernal_elec_gen(Ee, gp, gpe / gp) * gpe * gpe
        self.sat_B = MatD.toeplitz(nl, nh, t^)
        self.sat_s = sat_s
        self.sat_kmax = kmax
        resize_zero(self.inv_gp2, nh)
        for j in range(nh):
            var g = MASS_P / dat.egrid_had[j]
            self.inv_gp2[j] = g * g
        resize_zero(self.inv_eph2, kmax)
        for k in range(kmax):
            var e = MASS_E / dat.egrid_had_photons[k]
            self.inv_eph2[k] = e * e
        resize_zero(self.u_sat, kmax + 1)

    def _saturated_sums(mut self, dat: AM3Arrays):
        var nh = dat.nbins_had
        for j in range(nh):
            self.w_sat[j] = 0.0
        if self.sat_B.rows == 0:
            for i in range(len(self.q_sat)):
                self.q_sat[i] = 0.0
            return
        var kk = self.sat_kmax
        self.u_sat[kk] = 0.0
        for k in range(kk - 1, -1, -1):
            self.u_sat[k] = (
                self.u_sat[k + 1] + Float64(self.w_vphot[k]) * self.inv_eph2[k]
            )
        for j in range(self.sparse_prinmin, nh):
            var k0 = max(self.sat_s - j, 0)
            if k0 < kk:
                self.w_sat[j] = (
                    dat.n[P_protons][j] * self.inv_gp2[j] * self.u_sat[k0]
                )
        gemv_d(self.sat_B, pd(self.w_sat), 1.0, pd(self.q_sat))

    def Calc_BH_Rad(mut self, dat: AM3Arrays):
        """The pair emission of the protons and target photons within the
        optimization limits that the kernel was built for."""
        if self.verbose:
            print(" computing BH radiation term")
        for i in range(dat.nbins_lep):
            self.QBH_LGen[i] = 0.0
        var k_inj_rate = C_LIGHT * 3.0 / (16.0 * PI) * SIGMA_T * ALPHA
        var ncols = self.packed.cols
        var photon_norm = 1.0e-50
        for k in range(ncols):
            photon_norm += dat.n[P_photons_on_had_grid][k]
        self.w_vphot = zerosf(ncols)
        for k in range(ncols):
            self.w_vphot[k] = Float32(
                dat.n[P_photons_on_had_grid][k] / photon_norm
            )
        if self.packed.nmat == 0:
            return
        # the packed kernel reads the photon vector through an unchecked pointer
        debug_assert(
            len(self.w_vphot) >= self.packed.cols,
            "Bethe-Heitler: photon vector shorter than the kernel",
        )
        self._saturated_sums(dat)
        var job = BHJob(
            Pointer(to=self.packed).unsafe_origin_cast[MutUntrackedOrigin](),
            pf(self.w_vphot),
            pd(dat.n[P_protons]),
            pd(self.q_sat),
            pd(self.QBH_LGen),
            self.sparse_prinmin,
            dat.nbins_had,
            dat.nbins_lep,
            photon_norm,
            dat.dlnE,
            k_inj_rate,
        )
        var nt = self.nthreads
        parallel_run[bh_rad_work](job, nt)

    def Calc_BH_Cool(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing BH proton cooling term")
        for i in range(dat.nbins_had):
            self.ABHHadP[i] = 0.0
        if self.KER_BH_LOSS.rows == 0:
            return
        var job = BHCoolJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=dat)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
        )
        var nblk = dat.nbins_had // WD + 1
        parallel_run[bh_cool_work](job, min(self.nthreads, nblk))
