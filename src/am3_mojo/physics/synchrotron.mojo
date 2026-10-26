"""Ports of Syn (electron/positron synchrotron, Synchrotron.cc) and SynP
(proton/pion/muon synchrotron, HadronicSynchrotron.cc)."""

from std.math import sqrt
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import MatD, gemv_d
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.specfun import gsl_bessel_Knu, int_bessel_k53


def _syn_fdb_approx(norm: Float64, xbar: Float64, low_cut: Float64) -> Float64:
    """Finke, Dermer & Boettcher (2008) approximation shared by both kernels."""
    if 2.0 * xbar < low_cut:
        return norm * 1.80842 * c_pow(xbar, 0.33333) * c_pow(2.0, -0.6666667)
    elif 2.0 * xbar >= low_cut and 2.0 * xbar < 1.0:
        var y = c_log10(2.0 * xbar)
        var A0 = -0.35775237
        var A1 = -0.83695385 * y
        var A2 = -1.1449608 * (y * y)
        var A3 = -0.68137283 * c_pow(y, 3.0)
        var A4 = -0.22754737 * c_pow(y, 4.0)
        var A5 = -0.031967334 * c_pow(y, 5.0)
        return norm * c_pow(10.0, A0 + A1 + A2 + A3 + A4 + A5) / 2.0
    elif 2.0 * xbar >= 1.0 and 2.0 * xbar < 10.0:
        var y = c_log10(2.0 * xbar)
        var A0 = -0.35842494
        var A1 = -0.79652041 * y
        var A2 = -1.6113032 * (y * y)
        var A3 = 0.26055213 * c_pow(y, 3.0)
        var A4 = -1.6979017 * c_pow(y, 4.0)
        var A5 = 0.032955035 * c_pow(y, 5.0)
        return norm * c_pow(10.0, A0 + A1 + A2 + A3 + A4 + A5) / 2.0
    elif 2.0 * xbar >= 10.0 and 2.0 * xbar < 100.0:
        return (
            norm
            * PI
            / 4.0
            * c_exp(-2.0 * xbar)
            * (1.0 - 99.0 / 162.0 / 2.0 / xbar)
        )
    else:
        return 0.0


def _I_Qsyn_mu_1(
    mu: Float64, b_ratio: Float64, gamma_e: Float64, epsilon: Float64
) -> Float64:
    var ksi0 = 1.5 * b_ratio * gamma_e
    var epsilon_c = 1.5 * b_ratio * gamma_e * gamma_e
    var x0 = epsilon / epsilon_c
    var s = sqrt(1.0 - mu * mu)
    var x = x0 / s
    var ksi = ksi0 * s
    var z = x / (1.0 - x * ksi)
    if z <= 0.0:
        return 0.0
    var y1 = 0.0 if z > 1.0e3 else int_bessel_k53(z)
    return x0 * y1


def _I_Qsyn_mu_2(
    mu: Float64, b_ratio: Float64, gamma_e: Float64, epsilon: Float64
) -> Float64:
    var ksi0 = 1.5 * b_ratio * gamma_e
    var epsilon_c = 1.5 * b_ratio * gamma_e * gamma_e
    var x0 = epsilon / epsilon_c
    var s = sqrt(1.0 - mu * mu)
    var x = x0 / s
    var ksi = ksi0 * s
    var z = x / (1.0 - x * ksi)
    if z <= 0.0:
        return 0.0
    var y2 = 0.0 if z > 1.0e2 else x * ksi * ksi * z * gsl_bessel_Knu(
        2.0 / 3.0, z
    )
    return x0 * y2


def _rk4_mu[
    which: Int
](
    x_lower: Float64,
    x_upper: Float64,
    N_INTG: Int,
    b: Float64,
    g: Float64,
    e: Float64,
) -> Float64:
    var diff_x = (x_upper - x_lower) / Float64(N_INTG)
    var x = x_lower
    var y = 0.0
    for _ in range(N_INTG):
        var v1: Float64
        var v2: Float64
        var v4: Float64
        comptime if which == 1:
            v1 = _I_Qsyn_mu_1(x, b, g, e)
            v2 = _I_Qsyn_mu_1(x + 0.5 * diff_x, b, g, e)
            v4 = _I_Qsyn_mu_1(x + diff_x, b, g, e)
        else:
            v1 = _I_Qsyn_mu_2(x, b, g, e)
            v2 = _I_Qsyn_mu_2(x + 0.5 * diff_x, b, g, e)
            v4 = _I_Qsyn_mu_2(x + diff_x, b, g, e)
        y += (v1 + 4.0 * v2 + v4) / 6.0
        x += diff_x
    y *= diff_x
    return y


struct QSynJob(Movable):
    var syn: Pointer[Syn, MutUntrackedOrigin]
    var dat: Pointer[AM3Arrays, MutUntrackedOrigin]

    def __init__(
        out self,
        syn: Pointer[Syn, MutUntrackedOrigin],
        dat: Pointer[AM3Arrays, MutUntrackedOrigin],
    ):
        self.syn = syn
        self.dat = dat


def qsyn_init_work(job: Pointer[QSynJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref S = job[].syn[]
    ref dat = job[].dat[]
    var i = t
    while i < dat.nbins_pho:
        for j in range(dat.nbins_lep):
            S.KER_PsyU.set(
                i,
                j,
                S.Q_Qsyn_avg(dat.egrid_lep[j], dat.egrid_pho[i], dat.dlnE, 6)
                * T_SSC,
            )
        i += nt


struct Syn(Copyable, Movable):
    var nthreads: Int
    var verbose: Bool
    var afsyG: Vec
    var KER_PsyU: MatD  # (nbins_pho x nbins_lep), photon index contiguous
    var b_gauss: Float64
    var coeff_abs: Float64
    var w_epx: Vec

    def __init__(out self, rp: RunParams, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.nthreads = 1
        self.afsyG = Vec()
        self.KER_PsyU = MatD()
        self.w_epx = Vec()
        self.b_gauss = rp.mag_field
        var lambC = H_PLANCK * C_LIGHT / MASS_E / L_SSC
        self.coeff_abs = c_pow(lambC, 3.0) / 8.0 / PI
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.afsyG, dat.nbins_pho)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        self.KER_PsyU = MatD(dat.nbins_pho, dat.nbins_lep)

    def initialize_kernels(mut self, dat: AM3Arrays, qsyn: Bool):
        if qsyn:
            if self.verbose:
                print("computing qsyn e+/- kernel")
            var job = QSynJob(
                Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
                Pointer(to=dat)
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutUntrackedOrigin](),
            )
            parallel_run[qsyn_init_work](job, self.nthreads)
        else:
            if self.verbose:
                print("computing classical e+/- kernel")
            for i in range(dat.nbins_pho):
                for j in range(dat.nbins_lep):
                    self.KER_PsyU.set(
                        i,
                        j,
                        self.syn_kernel_ultrarel(
                            dat.egrid_pho[i], dat.egrid_lep[j]
                        ),
                    )

    def syn_kernel_ultrarel(self, x: Float64, g: Float64) -> Float64:
        var b_dimless = self.b_gauss / B_CRIT
        # the magnetic energy density B^2 / (8 pi) in the code units
        var b_ub_nat = (
            self.b_gauss * self.b_gauss / (8.0 * PI) / (MEC2_ERG * N_SSC)
        )
        var norm = (1.65399 / b_dimless) * b_ub_nat
        var xbar = x / (3.0 * g * g * b_dimless)
        return _syn_fdb_approx(norm, xbar, 4e-3)

    def I_Qsyn_avg(
        self,
        b_ratio: Float64,
        gamma_e: Float64,
        epsilon: Float64,
        bin_width: Float64,
        N_INTG: Int,
    ) -> Float64:
        var eta_0 = 1.5 * b_ratio * gamma_e
        var I_K53 = 0.5 * _rk4_mu[1](
            -0.99, 0.99, N_INTG, b_ratio, gamma_e, epsilon
        )
        var I_K23 = 0.5 * _rk4_mu[2](
            -0.99, 0.99, N_INTG, b_ratio, gamma_e, epsilon
        )
        var last_bin_avg = (
            (2.712 * c_pow(eta_0 * bin_width, 2.0 / 3.0) - 4.947)
            / bin_width
            / (2.0 * eta_0)
        )
        var alpha_e = 1.0 - epsilon / gamma_e
        if (
            eta_0 > 5.0e2
            and alpha_e > -0.2 * bin_width
            and alpha_e < 0.8 * bin_width
        ):
            return I_K53 + last_bin_avg
        return I_K53 + I_K23

    def Q_Qsyn_avg(
        self,
        gamma_e: Float64,
        epsilon: Float64,
        bin_width: Float64,
        N_INTG: Int,
    ) -> Float64:
        var b_dimless = self.b_gauss / B_CRIT
        var nu_0 = MASS_E / H_PLANCK  # m_e c^2 / h [Hz]
        return (
            sqrt(3.0)
            * nu_0
            * ALPHA
            * b_dimless
            * self.I_Qsyn_avg(b_dimless, gamma_e, epsilon, bin_width, N_INTG)
        )

    def Calc_SY_Phot_Sink(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing pair syn self-abs")
        var nl = dat.nbins_lep
        var npho = dat.nbins_pho
        resize_zero(self.w_epx, nl)
        var EPX = pd(self.w_epx)
        var pairs = pd(dat.n[P_pairs])
        var eg = pd(dat.egrid_lep)
        for j in range(1, nl - 1):
            EPX[unsafe_offset=j] = (
                (
                    6.0 * pairs[unsafe_offset=j]
                    - (pairs[unsafe_offset=j + 1] - pairs[unsafe_offset=j - 1])
                    / dat.dlnE
                )
                / eg[unsafe_offset=j]
                / 2.0
            )
        EPX[unsafe_offset=0] = 0.0
        EPX[unsafe_offset=nl - 1] = 0.0
        var K = self.KER_PsyU.ptr()
        var out_ = pd(self.afsyG)
        var ep = pd(dat.egrid_pho)
        var coef = self.coeff_abs * dat.dlnE
        var i = 0
        while i + WD <= npho:
            var x = SIMD[DType.float64, WD](0)
            for j in range(1, nl - 1):
                var k = K.unsafe_offset(j * npho + i).unsafe_load[width=WD]()
                x += k * EPX[unsafe_offset=j]
            var e = ep.unsafe_offset(i).unsafe_load[width=WD]()
            var v = coef * x / e / e
            out_.unsafe_offset(i).unsafe_store(
                v.gt(0).select(v, SIMD[DType.float64, WD](0))
            )
            i += WD
        while i < npho:
            var x = 0.0
            for j in range(1, nl - 1):
                x += K[unsafe_offset=j * npho + i] * EPX[unsafe_offset=j]
            var v = coef * x / ep[unsafe_offset=i] / ep[unsafe_offset=i]
            out_[unsafe_offset=i] = v if v > 0.0 else 0.0
            i += 1

    def Calc_SY_Phot_Emit(
        self,
        dat: AM3Arrays,
        outs: Array[FPtr, N_LEP_IN],
        ins: Array[FPtr, N_LEP_IN],
        only_first: Bool,
    ):
        """V_phot_out_rate[m][i] = dlnE * sum_j K(i,j) V_elec_in[m][j]."""
        comptime assert N_LEP_IN <= 8, "the register blocks have up to 8 inputs"
        if self.verbose:
            print("computing pair syn emission")
        var npho = dat.nbins_pho
        var nl = dat.nbins_lep
        var dl = dat.dlnE
        for m in range(N_LEP_IN):
            for i in range(npho):
                outs[m][unsafe_offset=i] = 0.0
        var K = self.KER_PsyU.ptr()
        if only_first:
            var e0 = ins[0]
            var o0 = outs[0]
            var i = 0
            while i + WD <= npho:
                var y = SIMD[DType.float64, WD](0)
                for j in range(1, nl - 1):
                    var z = K.unsafe_offset(j * npho + i).unsafe_load[
                        width=WD
                    ]()
                    y += z * e0[unsafe_offset=j]
                o0.unsafe_offset(i).unsafe_store(dl * y)
                i += WD
            while i < npho:
                var y = 0.0
                for j in range(1, nl - 1):
                    y += K[unsafe_offset=j * npho + i] * e0[unsafe_offset=j]
                o0[unsafe_offset=i] = dl * y
                i += 1
            return
        # inputs that are zero everywhere, and electron bins where every input
        # is zero, only add exact zeros: they are left out (the outputs of
        # those inputs stay zero)
        var nz_ins = ins.copy()
        var nz_outs = outs.copy()
        var n = 0
        var jlo = nl - 1
        var jhi = 1
        for m in range(N_LEP_IN):
            var nonzero = False
            for j in range(1, nl - 1):
                if ins[m][unsafe_offset=j] != 0.0:
                    jlo = min(jlo, j)
                    jhi = max(jhi, j + 1)
                    nonzero = True
            if nonzero:
                nz_ins[n] = ins[m]
                nz_outs[n] = outs[m]
                n += 1
        if n == 1:
            _sy_emit[1](K, npho, jlo, jhi, dl, nz_ins, nz_outs)
        elif n == 2:
            _sy_emit[2](K, npho, jlo, jhi, dl, nz_ins, nz_outs)
        elif n == 3:
            _sy_emit[3](K, npho, jlo, jhi, dl, nz_ins, nz_outs)
        elif n == 4:
            _sy_emit[4](K, npho, jlo, jhi, dl, nz_ins, nz_outs)
        elif n == 5:
            _sy_emit[5](K, npho, jlo, jhi, dl, nz_ins, nz_outs)
        elif n == 6:
            _sy_emit[6](K, npho, jlo, jhi, dl, nz_ins, nz_outs)
        elif n == 7:
            _sy_emit[7](K, npho, jlo, jhi, dl, nz_ins, nz_outs)
        elif n == 8:
            _sy_emit[8](K, npho, jlo, jhi, dl, nz_ins, nz_outs)


def _sy_emit[
    n: Int
](
    K: FPtr,
    npho: Int,
    jlo: Int,
    jhi: Int,
    dl: Float64,
    ins: Array[FPtr, N_LEP_IN],
    outs: Array[FPtr, N_LEP_IN],
):
    """outs[m][i] = dl sum_j K[j * npho + i] ins[m][j] over the electron bins
    [jlo, jhi) for the first n inputs: 8 photon bins at a time (a cache line
    of each kernel row; 4 for more than 6 inputs), the inputs in registers.
    """
    comptime W = 8 if n <= 6 else 4
    var i = 0
    while i + W <= npho:
        var y = Array[SIMD[DType.float64, W], n](fill=SIMD[DType.float64, W](0))
        for j in range(jlo, jhi):
            var z = K.unsafe_offset(j * npho + i).unsafe_load[width=W]()
            comptime for m in range(n):
                y[m] += z * ins[m][unsafe_offset=j]
        comptime for m in range(n):
            outs[m].unsafe_offset(i).unsafe_store(dl * y[m])
        i += W
    while i < npho:
        comptime for m in range(n):
            var y = 0.0
            for j in range(jlo, jhi):
                y += K[unsafe_offset=j * npho + i] * ins[m][unsafe_offset=j]
            outs[m][unsafe_offset=i] = dl * y
        i += 1


struct SynCJob(Movable):
    """The charm-hadron synchrotron emission of the masses t, t + nt, ...
    (sync_work)."""

    var S: Pointer[SynP, MutUntrackedOrigin]
    var dlnE: Float64

    def __init__(out self, S: Pointer[SynP, MutUntrackedOrigin], dlnE: Float64):
        self.S = S
        self.dlnE = dlnE


def sync_work(job: Pointer[SynCJob, MutUntrackedOrigin], t: Int, nt: Int):
    """Mass k = t, t + nt, ...: its emission into its own buffer."""
    ref S = job[].S[]
    var k = t
    while k < 3:
        gemv_d(S.KER_PsyC[k], pd(S.w_inC[k]), job[].dlnE, pd(S.tmpC[k]))
        k += nt


struct SynP(Copyable, Movable):
    var nthreads: Int
    var verbose: Bool
    var b_gauss: Float64
    var epsilonsyP: Vec
    var epsilonsyPi: Vec
    var epsilonsyMu: Vec
    var epsilonsyK: Vec
    var KER_PsyP: MatD
    var KER_PsyPi: MatD
    var KER_PsyMu: MatD
    var KER_PsyK: MatD  # kaons (am3_mojo addition), allocated when needed
    # charged charm hadrons (am3_mojo addition): one kernel per mass (D+-;
    # Ds+-; Lambda_c+ and anti-Lambda_c-), allocated when needed, and their
    # emission
    var KER_PsyC: List[MatD]
    var epsilonsyC: Vec
    var w_in: Vec
    var w_inC: List[Vec]  # per mass: the summed charges, and the emission
    var tmpC: List[Vec]

    def __init__(out self, rp: RunParams, dat: AM3Arrays, verbose: Bool):
        self.nthreads = 1
        self.verbose = verbose
        self.epsilonsyP = Vec()
        self.epsilonsyPi = Vec()
        self.epsilonsyMu = Vec()
        self.epsilonsyK = Vec()
        self.KER_PsyP = MatD()
        self.KER_PsyPi = MatD()
        self.KER_PsyMu = MatD()
        self.KER_PsyK = MatD()
        self.KER_PsyC = [MatD(), MatD(), MatD()]
        self.epsilonsyC = Vec()
        self.w_in = Vec()
        self.w_inC = [Vec(), Vec(), Vec()]
        self.tmpC = [Vec(), Vec(), Vec()]
        self.b_gauss = rp.mag_field
        self.clear_internal_arrays(dat)
        self.reinitialize_empty_kernels(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.epsilonsyP, dat.nbins_pho)
        resize_zero(self.epsilonsyPi, dat.nbins_pho)
        resize_zero(self.epsilonsyMu, dat.nbins_pho)
        resize_zero(self.epsilonsyK, dat.nbins_pho)
        resize_zero(self.epsilonsyC, dat.nbins_pho)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        self.KER_PsyP = MatD(dat.nbins_pho, dat.nbins_had)
        self.KER_PsyPi = MatD(dat.nbins_pho, dat.nbins_had)
        self.KER_PsyMu = MatD(dat.nbins_pho, dat.nbins_had)
        self.KER_PsyK = MatD()
        for k in range(3):
            self.KER_PsyC[k] = MatD()

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        rad_P: Bool,
        rad_Pi: Bool,
        rad_Mu: Bool,
        rad_K: Bool,
        rad_C: Bool,
    ):
        if rad_P:
            if self.verbose:
                print("computing proton syn. rad. kernel")
            for i in range(dat.nbins_pho):
                for j in range(dat.nbins_had):
                    self.KER_PsyP.set(
                        i,
                        j,
                        self.syn_kernel_ultrarel(
                            dat.egrid_had_photons[i], dat.egrid_had[j], MASS_P
                        ),
                    )
        if rad_Pi:
            if self.verbose:
                print("computing pion syn. rad. kernel")
            for i in range(dat.nbins_pho):
                for j in range(dat.nbins_had):
                    self.KER_PsyPi.set(
                        i,
                        j,
                        self.syn_kernel_ultrarel(
                            dat.egrid_had_photons[i], dat.egrid_had[j], MASS_PI
                        ),
                    )
        if rad_Mu:
            if self.verbose:
                print("computing muon syn. rad. kernel")
            for i in range(dat.nbins_pho):
                for j in range(dat.nbins_had):
                    self.KER_PsyMu.set(
                        i,
                        j,
                        self.syn_kernel_ultrarel(
                            dat.egrid_had_photons[i], dat.egrid_had[j], MASS_MU
                        ),
                    )
        if rad_K:
            if self.verbose:
                print("computing kaon syn. rad. kernel")
            self.KER_PsyK = MatD(dat.nbins_pho, dat.nbins_had)
            for i in range(dat.nbins_pho):
                for j in range(dat.nbins_had):
                    self.KER_PsyK.set(
                        i,
                        j,
                        self.syn_kernel_ultrarel(
                            dat.egrid_had_photons[i], dat.egrid_had[j], MASS_K
                        ),
                    )
        if rad_C:
            if self.verbose:
                print("computing charm-hadron syn. rad. kernels")
            var masses: List[Float64] = [MASS_D, MASS_DS, MASS_LC]
            for k in range(3):
                self.KER_PsyC[k] = MatD(dat.nbins_pho, dat.nbins_had)
                for i in range(dat.nbins_pho):
                    for j in range(dat.nbins_had):
                        self.KER_PsyC[k].set(
                            i,
                            j,
                            self.syn_kernel_ultrarel(
                                dat.egrid_had_photons[i],
                                dat.egrid_had[j],
                                masses[k],
                            ),
                        )

    def syn_kernel_ultrarel(
        self, x: Float64, pe: Float64, mass: Float64
    ) -> Float64:
        var norm = (
            (3.0 * 1.732 / PI)
            * SIGMA_T
            * C_LIGHT
            * self.b_gauss
            * B_CRIT
            / (8.0 * PI)
            * GEV_PER_ERG
            / mass
        )
        var b_dimless = self.b_gauss / B_CRIT
        var xbar = (
            x
            * mass
            / (3.0 * pe * pe * b_dimless * MASS_E / mass * MASS_E / mass)
        )
        return _syn_fdb_approx(norm, xbar, 1e-2)

    def Calc_SY_PRO(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing proton syn emission")
        gemv_d(
            self.KER_PsyP, pd(dat.n[P_protons]), dat.dlnE, pd(self.epsilonsyP)
        )

    def Calc_SY_Pi(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing pion IC emission")
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = dat.n[P_pion_plus][i] + dat.n[P_pion_minus][i]
        gemv_d(self.KER_PsyPi, pd(self.w_in), dat.dlnE, pd(self.epsilonsyPi))

    def Calc_SY_Mu(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing muon IC emission")
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = (
                dat.n[P_muon_plus_left][i]
                + dat.n[P_muon_plus_right][i]
                + dat.n[P_muon_minus_left][i]
                + dat.n[P_muon_minus_right][i]
            )
        gemv_d(self.KER_PsyMu, pd(self.w_in), dat.dlnE, pd(self.epsilonsyMu))

    def Calc_SY_K(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing kaon syn emission")
        if self.KER_PsyK.rows == 0:
            fill_zero(self.epsilonsyK)
            return
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = dat.n[P_kaon_plus][i] + dat.n[P_kaon_minus][i]
        gemv_d(self.KER_PsyK, pd(self.w_in), dat.dlnE, pd(self.epsilonsyK))

    def Calc_SY_C(mut self, dat: AM3Arrays):
        """Synchrotron emission of the charged charm hadrons (am3_mojo
        addition): D+-, Ds+-, Lambda_c+ and anti-Lambda_c-, each with its mass.
        The masses run in parallel, each into its own buffer; the sum keeps the
        order D, Ds, Lambda_c, so the result does not depend on the threads."""
        if self.verbose:
            print("computing charm-hadron syn emission")
        fill_zero(self.epsilonsyC)
        if self.KER_PsyC[0].rows == 0:
            return
        for k in range(3):
            var s0 = CDP if k == 0 else (CDSP if k == 1 else CLC)
            resize_zero(self.w_inC[k], dat.nbins_had)
            resize_zero(self.tmpC[k], dat.nbins_pho)
            for i in range(dat.nbins_had):
                self.w_inC[k][i] = (
                    dat.n[P_charm + s0][i] + dat.n[P_charm + s0 + 1][i]
                )
        var job = SynCJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](), dat.dlnE
        )
        parallel_run[sync_work](job, min(self.nthreads, 3))
        for k in range(3):
            for i in range(dat.nbins_pho):
                self.epsilonsyC[i] += self.tmpC[k][i]
