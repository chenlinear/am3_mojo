"""Port of BetheHeitler (BetheHeitler.cc): p + gamma -> p + e+ e-.

init_kernels builds the float32 pair-emission kernel densely (about 120 MB on
the default grids). The first step packs it into its non-zero row windows
(PackedGemvF), which the emission reads sequentially; the C++ code streams the
dense kernel every step. The sums run only over the proton bins between the
lowest and the highest that hold particles, and the worker threads share the
outgoing lepton bins."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.linalg import MatD, Mat3F, PackedGemvF


@always_inline
def bh_sigma_W(omega: Float64, E_: Float64, ksi: Float64) -> Float64:
    if omega < 2.0001 or omega > 600.0:
        return 0.0
    var REG = 1.0e-30
    var k = omega
    var g = E_
    var u = ksi
    var p = c_sqrt(g * g - 1.0 + REG)
    var g1 = k - g
    var p1 = c_sqrt(g1 * g1 - 1.0 + REG)
    var D = g - u * p
    var T = c_sqrt(k * k + p * p - 2.0 * k * p * u)
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


@always_inline
def bh_inner(gp: Float64, Ee: Float64, omega: Float64, E_: Float64) -> Float64:
    var REG = 1.0e-20
    var p_ = c_sqrt(E_ * E_ - 1.0 + REG)
    var ksi = (gp * E_ - Ee) / (gp * p_)
    if ksi < -1.0 or ksi > 1.0 or E_ < 1.0 or E_ > omega - 1.0:
        return 0.0
    return omega * bh_sigma_W(omega, E_, ksi) / p_


def bh_outer(gp: Float64, Ee: Float64, omega: Float64) raises -> Float64:
    var REG = 1.0e-5
    var lower = (gp * gp + Ee * Ee) / (2.0 * gp * Ee) + REG
    var upper = omega - 1.0 - REG
    var diff = (upper - lower) / lower
    var tolerance = 1.0e-4
    if diff < -tolerance:
        print("gp = ", gp)
        print("Ee = ", Ee)
        print("om = ", omega)
        print("lo = ", lower)
        print("up = ", upper)
        raise Error(
            "in function BetheHeitler::outer, upper/lower - 1 > tolerance\n"
        )
    elif -tolerance < diff and diff < tolerance:
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


def bh_kernal_elec_gen(
    Ee: Float64, gp: Float64, Eph: Float64
) raises -> Float64:
    var upper = 2.0 * gp * Eph
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
    for _ in range(N_BINS):
        var k1 = DX * bh_outer(gp, Ee, x)
        var k2 = DX * bh_outer(gp, Ee, x + 0.5 * DX)
        var k4 = DX * bh_outer(gp, Ee, x + DX)
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
            (3.1416 / 12.0)
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
    var K: Pointer[Mat3F, MutUntrackedOrigin]
    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]
    var n_el_out: Int
    var proton_in_min: Int
    var photon_in_max: Int
    var failed: Bool

    def __init__(
        out self,
        K: Pointer[Mat3F, MutUntrackedOrigin],
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
        n_el_out: Int,
        proton_in_min: Int,
        photon_in_max: Int,
    ):
        self.K = K
        self.dat = dat
        self.n_el_out = n_el_out
        self.proton_in_min = proton_in_min
        self.photon_in_max = photon_in_max
        self.failed = False


def bh_init_work(job: Pointer[BHInitJob, MutUntrackedOrigin], t: Int, nt: Int):
    """Electron-generation kernel for the outgoing lepton bins
    i = t*n_el_out, (t+nt)*n_el_out, ..."""
    ref J = job[]
    ref dat = J.dat[]
    var i = t * J.n_el_out
    try:
        while i < dat.nbins_lep:
            for j in range(J.proton_in_min, dat.nbins_had):
                for k in range(J.photon_in_max):
                    if k < dat.nbins_pho_hadgrid:
                        J.K[].set(
                            i,
                            j,
                            k,
                            Float32(
                                bh_kernal_elec_gen(
                                    dat.egrid_had_leptons[i] / masse,
                                    dat.egrid_had[j] / massp,
                                    dat.egrid_had_photons[k] / masse,
                                )
                            ),
                        )
            i += nt * J.n_el_out
    except:
        J.failed = True


struct BHJob(Movable):
    var K: Pointer[PackedGemvF, MutUntrackedOrigin]
    var v: F32Ptr
    var protons: FPtr
    var out_: FPtr
    var prinmin: Int
    var nh: Int
    var nout: Int
    var step: Int
    var HaRN: Float64
    var dlnE: Float64
    var k_inj: Float64

    def __init__(
        out self,
        K: Pointer[PackedGemvF, MutUntrackedOrigin],
        v: F32Ptr,
        protons: FPtr,
        out_: FPtr,
        prinmin: Int,
        nh: Int,
        nout: Int,
        step: Int,
        HaRN: Float64,
        dlnE: Float64,
        k_inj: Float64,
    ):
        self.K = K
        self.v = v
        self.protons = protons
        self.out_ = out_
        self.prinmin = prinmin
        self.nh = nh
        self.nout = nout
        self.step = step
        self.HaRN = HaRN
        self.dlnE = dlnE
        self.k_inj = k_inj


def bh_rad_work(job: Pointer[BHJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref J = job[]
    var buf = zerosf(J.K[].res_len())
    var res = pf(buf)
    # proton bins with exactly zero density only add exact zeros: skip them
    var plo = J.nh
    var phi = 0
    for j in range(J.prinmin, J.nh):
        if J.protons[unsafe_offset=j] != 0.0:
            plo = min(plo, j)
            phi = max(phi, j + 1)
    var i = t * J.step
    while i < J.nout:
        var x = 0.0
        if phi > plo:
            J.K[].apply(i, J.v, Float32(1.0), res, plo, phi)
        for j in range(plo, phi):
            x += Float64(res[unsafe_offset=j]) * J.protons[unsafe_offset=j]
        J.out_[unsafe_offset=i] = J.HaRN * x * J.dlnE * J.dlnE * J.k_inj
        i += nt * J.step
    _ = buf^


struct BetheHeitler(Copyable, Movable):
    var verbose: Bool
    var QBH_LGen: Vec
    var ABHHadP: Vec
    var KER_BH_LOSS: MatD  # (nbins_had x nbins_pho)
    var KER: Mat3F  # voM_KER_BH_LGen: (e_out_dim) x (nbins_had x photon_in_dim)
    var packed: PackedGemvF
    var sparse_valid: Bool
    var nthreads: Int
    var sparse_prinmin: Int  # first proton row of the kernel (init_kernels)
    var w_res: VecF
    var w_vphot: VecF

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.QBH_LGen = Vec()
        self.ABHHadP = Vec()
        self.KER_BH_LOSS = MatD()
        self.KER = Mat3F()
        self.packed = PackedGemvF()
        self.sparse_valid = False
        self.nthreads = 1
        self.sparse_prinmin = -1
        self.w_res = VecF()
        self.w_vphot = VecF()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.QBH_LGen, dat.nbins_lep)
        resize_zero(self.ABHHadP, dat.nbins_had)

    def update_dim_KERBH(
        mut self, dat: AM3Arrays, e_out_dim: Int, photon_in_dim: Int
    ):
        self.KER = Mat3F(e_out_dim, dat.nbins_had, photon_in_dim)
        self.sparse_valid = False

    def reinitialize_empty_kernels(
        mut self,
        dat: AM3Arrays,
        rad_fast: Int,
        n_el_out: Int,
        proton_in_min: Int,
        photon_in_max: Int,
    ):
        var _n_el_out = 1
        var _photon_in_max = dat.nbins_pho
        if rad_fast != 0:
            _n_el_out = n_el_out
            _photon_in_max = photon_in_max
        self.update_dim_KERBH(dat, dat.nbins_lep // _n_el_out, _photon_in_max)
        self.KER_BH_LOSS = MatD(dat.nbins_had, dat.nbins_pho)

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        rad: Int,
        cool: Int,
        rad_fast: Int,
        n_el_out: Int,
        proton_in_min: Int,
        photon_in_max: Int,
    ) raises:
        if rad != 0:
            self.sparse_prinmin = max(proton_in_min, 0)
            if self.verbose:
                print(
                    "computing BH e+/- generation kernel. Using n_el_out: ",
                    n_el_out,
                    ", proton_in_min: ",
                    proton_in_min,
                    ", photon_in_max: ",
                    photon_in_max,
                    sep="",
                )
            var job = BHInitJob(
                Pointer(to=self.KER).unsafe_origin_cast[MutUntrackedOrigin](),
                Pointer(to=dat)
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutUntrackedOrigin](),
                n_el_out,
                proton_in_min,
                photon_in_max,
            )
            var nt = self.nthreads
            parallel_run[bh_init_work](job, nt)
            if job.failed:
                raise Error(
                    "in function BetheHeitler::outer, upper/lower - 1 >"
                    " tolerance\n"
                )
        if cool != 0:
            if self.verbose:
                print("computing BH proton loss kernel")
            for i in range(dat.nbins_had):
                for j in range(dat.nbins_pho):
                    self.KER_BH_LOSS.set(
                        i,
                        j,
                        bh_kernal_proton_loss(
                            dat.egrid_had[i] / massp,
                            dat.egrid_had_photons[j] / masse,
                        ),
                    )

    def Calc_BH_Rad(
        mut self, dat: AM3Arrays, nelout: Int, prinmin: Int, phinmax: Int
    ):
        if self.verbose:
            print(" computing BH radiation term")
        for i in range(dat.nbins_lep):
            self.QBH_LGen[i] = 0.0
        var k_inj_rate = c0 * 3.0 / (16.0 * mathpi) * sigmaT / 137.0
        # the kernel keeps the target photons of init_kernels: photons above a
        # limit lowered since then count as zero (as for inverse Compton), and
        # the photon vector covers every column the kernel reads
        var ncols = self.packed.cols if self.sparse_valid else self.KER.cols
        var n_now = min(ncols, phinmax)
        var photon_norm = 1.0e-50
        for k in range(n_now):
            photon_norm += dat.n[P_photons_on_had_grid][k]
        self.w_vphot = zerosf(ncols)
        for k in range(n_now):
            self.w_vphot[k] = Float32(
                dat.n[P_photons_on_had_grid][k] / photon_norm
            )
        if self.KER.nmat == 0:
            return
        if not self.sparse_valid and self.KER.rows > 0:
            # pack every row from the minimum proton energy of init_kernels,
            # not of the first call, which would lose rows if the minimum were
            # lowered later. The current minimum only bounds the sum below, as
            # in the C++.
            var row_min = zerosi(self.KER.nmat)
            for k in range(self.KER.nmat):
                row_min[k] = self.sparse_prinmin
            self.packed = PackedGemvF(self.KER, row_min)
            self.sparse_valid = True
            self.KER = Mat3F(self.KER.nmat, 0, 0)
        if not self.sparse_valid:
            return
        # the packed kernel reads the photon vector through an unchecked pointer
        debug_assert(
            len(self.w_vphot) >= self.packed.cols,
            "Bethe-Heitler: photon vector shorter than the kernel",
        )
        var job = BHJob(
            Pointer(to=self.packed).unsafe_origin_cast[MutUntrackedOrigin](),
            pf(self.w_vphot),
            pd(dat.n[P_protons]),
            pd(self.QBH_LGen),
            prinmin,
            dat.nbins_had,
            dat.nbins_lep // nelout,
            nelout,
            photon_norm,
            dat.dlnE,
            k_inj_rate,
        )
        var nt = self.nthreads
        parallel_run[bh_rad_work](job, nt)
        if nelout == 2:
            for ii in range(dat.nbins_lep // 2 - 2):
                self.QBH_LGen[2 * ii + 1] = c_exp(
                    0.5
                    * (
                        c_log(self.QBH_LGen[2 * ii])
                        + c_log(self.QBH_LGen[2 * ii + 2])
                    )
                )

    def Calc_BH_Cool(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing BH proton cooling term")
        for i in range(dat.nbins_had):
            self.ABHHadP[i] = 0.0
        if self.KER_BH_LOSS.rows == 0:
            return
        var k_rate_semiana = (
            (3.0 / 8.0 / mathpi) * (1.0 / 137.0) * sigmaT * c0 * (masse / massp)
        )
        var K = self.KER_BH_LOSS.ptr()
        var nh = self.KER_BH_LOSS.rows
        var ph = pd(dat.n[P_photons_on_had_grid])
        var nb = dat.nbins_had
        # SIMD over protons j, photons i accumulated in the original order. A
        # masked lane skips photons below threshold (Eph < 1/gp), exactly as
        # the scalar `continue` does
        var j = 0
        while j < nb:
            comptime W = WD
            if j + W <= nb:
                var ginv = SIMD[DType.float64, W](0)
                for l in range(W):
                    ginv[l] = 1.0 / (dat.egrid_had[j + l] / massp)
                var rj = SIMD[DType.float64, W](0)
                for i in range(dat.nbins_pho):
                    var Eph = dat.egrid_had_photons[i] / masse
                    var keep = SIMD[DType.float64, W](Eph).ge(ginv)
                    var k = K.unsafe_offset(i * nh + j).unsafe_load[width=W]()
                    rj = keep.select(rj + k * ph[unsafe_offset=i], rj)
                for l in range(W):
                    self.ABHHadP[j + l] = -rj[l] * k_rate_semiana * dat.dlnE
                j += W
            else:
                var rj = 0.0
                var gp = dat.egrid_had[j] / massp
                for i in range(dat.nbins_pho):
                    var Eph = dat.egrid_had_photons[i] / masse
                    if Eph < 1.0 / gp:
                        continue
                    rj += K[unsafe_offset=i * nh + j] * ph[unsafe_offset=i]
                self.ABHHadP[j] = -rj * k_rate_semiana * dat.dlnE
                j += 1
