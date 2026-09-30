"""Port of PhotoPion (PhotoPion.cc): p/n + gamma -> pions (Hümmer et al. 2010)."""

from std.math import fma
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.specfun import number_split
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.linalg import MatD, gemv_d, diagonal_gemv
from am3_mojo.utils.parallel import parallel_run

comptime N_PG_CHANNELS = 28 + 9 + 14
from am3_mojo.physics.photopion_tables import *


@always_inline
def pg_Theta(x: Float64) -> Float64:
    return 1.0 if x >= 0.0 else 0.0


@always_inline
def pg_fres(it: Int, y: Float64) -> Float64:
    return pg_Theta(2.0 * y - IT1Er(it)) / (2.0 * y * y)


def pg_fmult(it: Int, y: Float64) -> Float64:
    if y < 0.5 * IT3Em(it):
        return 0.0
    elif y < 0.5 * IT3Ex(it) and y >= 0.5 * IT3Em(it):
        return 1.0 - IT3Em(it) * IT3Em(it) / (4.0 * y * y)
    return (IT3Ex(it) * IT3Ex(it) - IT3Em(it) * IT3Em(it)) / (4.0 * y * y)


def pg_Idir(it: Int, z: Float64) -> Float64:
    var x = c_log10(0.5 * z)
    if it < 3:
        if z < 0.17:
            return 0.0
        elif z >= 0.17 and z < 0.96:
            return (
                35.95
                + 84.08 * x
                + 110.76 * x * x
                + 102.73 * x * x * x
                + 40.47 * x * x * x * x
            )
        return (
            30.20
            + 40.55 * x
            + 2.031 * x * x
            - 0.3879 * x * x * x
            + 0.02504 * x * x * x * x
        )
    if z < 0.4:
        return 0.0
    return -3.4083 + 16.2864 / z + 40.7160 * c_log(z)


def pg_fdir(it: Int, y: Float64) -> Float64:
    if y < 0.5 * IT2Em(it):
        return 0.0
    elif y < 0.5 * IT2Ex(it) and y >= 0.5 * IT2Em(it):
        return (pg_Idir(it, 2.0 * y) - pg_Idir(it, IT2Em(it))) / (2.0 * y * y)
    return (pg_Idir(it, IT2Ex(it)) - pg_Idir(it, IT2Em(it))) / (2.0 * y * y)


struct PGConvJob(Movable):
    """res[c * nh + i] = sum_j H_c[i + vlo + j] v[j] for the photo-pion channels.
    """

    var H: FPtr
    var res: FPtr
    var v: FPtr
    var nhk: Int
    var nh: Int
    var vlo: Int
    var nv: Int
    var hlo: IPtr
    var hhi: IPtr

    def __init__(
        out self,
        H: FPtr,
        res: FPtr,
        v: FPtr,
        nhk: Int,
        nh: Int,
        vlo: Int,
        nv: Int,
        hlo: IPtr,
        hhi: IPtr,
    ):
        self.H = H
        self.res = res
        self.v = v
        self.nhk = nhk
        self.nh = nh
        self.vlo = vlo
        self.nv = nv
        self.hlo = hlo
        self.hhi = hhi


def pg_conv_work(job: Pointer[PGConvJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref J = job[]
    var c = t
    while c < N_PG_CHANNELS:
        # the table row shifted by vlo, so that column 0 is target bin vlo
        diagonal_gemv[True](
            J.H.unsafe_offset(c * J.nhk + J.vlo),
            J.hlo[unsafe_offset=c] - J.vlo,
            J.hhi[unsafe_offset=c] - J.vlo,
            J.nh,
            J.nv,
            J.v,
            1.0,
            J.res.unsafe_offset(c * J.nh),
        )
        c += nt


struct PhotoPion(Copyable, Movable):
    var verbose: Bool
    var QPG_PROGen: Vec
    var QPG_NTRGen: Vec
    var afPG_PRO: Vec
    var afPG_NTR: Vec
    var qpi0: Vec
    var qpip: Vec
    var qpim: Vec
    var afPG_Rad: Vec
    # Pion-production channels (28 resonance, 9 direct, 14 multi-pion). Every
    # kernel entry depends on gamma_p * eps = exp((i + j) dlnE) / m_p, i.e. on
    # i + j only (Hankel form), so channel c is stored as one vector
    # H[c * nhk + m], m = i + j (proton bin i, target photon bin j).
    var ncols: Int
    var nhk: Int
    var H: Vec
    var hlo: VecI
    var hhi: VecI
    var kern_valid: Bool
    var nthreads: Int
    # channel results res[c * nbins_had + i] = sum_j K_c(i, j) v_j
    var res: Vec
    var KER_FPGapn: MatD  # (nbins_had x nbins_pho)
    var KER_FPGara: MatD  # (nbins_pho x nbins_had)
    var EigenRN: Float64
    var w_vphot: Vec
    var w_in: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.QPG_PROGen = Vec()
        self.QPG_NTRGen = Vec()
        self.afPG_PRO = Vec()
        self.afPG_NTR = Vec()
        self.qpi0 = Vec()
        self.qpip = Vec()
        self.qpim = Vec()
        self.afPG_Rad = Vec()
        self.ncols = 0
        self.nhk = 0
        self.H = Vec()
        self.hlo = VecI()
        self.hhi = VecI()
        self.kern_valid = False
        self.nthreads = 1
        self.res = zeros(dat.nbins_had * N_PG_CHANNELS)
        self.KER_FPGapn = MatD()
        self.KER_FPGara = MatD()
        self.EigenRN = 0.0
        self.w_vphot = Vec()
        self.w_in = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.qpi0, dat.nbins_had)
        resize_zero(self.qpip, dat.nbins_had)
        resize_zero(self.qpim, dat.nbins_had)
        resize_zero(self.QPG_PROGen, dat.nbins_had)
        resize_zero(self.QPG_NTRGen, dat.nbins_had)
        resize_zero(self.afPG_PRO, dat.nbins_had)
        resize_zero(self.afPG_NTR, dat.nbins_had)
        resize_zero(self.afPG_Rad, dat.nbins_pho)

    def reinitialize_empty_kernels(
        mut self, dat: AM3Arrays, photon_in_max: Int
    ):
        self.ncols = photon_in_max
        self.kern_valid = False
        if len(self.res) != dat.nbins_had * N_PG_CHANNELS:
            self.res = zeros(dat.nbins_had * N_PG_CHANNELS)
        self.KER_FPGapn = MatD(dat.nbins_had, dat.nbins_pho)
        self.KER_FPGara = MatD(dat.nbins_pho, dat.nbins_had)

    @staticmethod
    def channel_value(c: Int, y: Float64) -> Float64:
        """Kernel value of channel c (0-27 resonances, 28-36 direct, 37-50
        multi-pion) at gamma_p * eps = y."""
        if c < 28:
            return IT1Er(c) * IT1Bout(c) * IT1G(c) * pg_fres(c, y)
        elif c < 37:
            return pg_fdir(c - 28, y)
        return IT3sig(c - 37) * pg_fmult(c - 37, y)

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        rad: Int,
        cool: Int,
        photonLoss: Int,
        photon_in_max: Int,
    ):
        if rad != 0:
            if self.verbose:
                print(
                    (
                        "computing p-gamma to pion generation kernel (Hankel"
                        " form), photon_in_max: "
                    ),
                    photon_in_max,
                    sep="",
                )
            var nh = dat.nbins_had
            var nc = min(photon_in_max, dat.nbins_pho_hadgrid)
            self.ncols = nc
            self.nhk = nh + nc - 1
            self.H = zeros(N_PG_CHANNELS * self.nhk)
            self.hlo = zerosi(N_PG_CHANNELS)
            self.hhi = zerosi(N_PG_CHANNELS)
            for c in range(N_PG_CHANNELS):
                var lo = self.nhk
                var hi = 0
                for m in range(self.nhk):
                    var i = min(m, nh - 1)
                    var y = (dat.egrid_had[i] / massp) * dat.egrid_had_photons[
                        m - i
                    ]
                    var v = PhotoPion.channel_value(c, y)
                    self.H[c * self.nhk + m] = v
                    if v != 0.0:
                        lo = min(lo, m)
                        hi = m + 1
                self.hlo[c] = lo
                self.hhi[c] = hi
            self.kern_valid = True
        var co = c0 * dat.dlnE * mubarn
        if cool != 0:
            if self.verbose:
                print("computing p-gamma proton/neutron loss kernel")
            for i in range(dat.nbins_had):
                for j in range(dat.nbins_pho):
                    var yg = (dat.egrid_had[i] / massp) * dat.egrid_had_photons[
                        j
                    ]
                    self.KER_FPGapn.set(i, j, self._loss_kernel(yg, co))
        if photonLoss != 0:
            if self.verbose:
                print("computing p-gamma photon loss kernel")
            for j in range(dat.nbins_pho):
                for i in range(dat.nbins_had):
                    var yg = (dat.egrid_had[i] / massp) * dat.egrid_had_photons[
                        j
                    ]
                    self.KER_FPGara.set(
                        j,
                        i,
                        self.KER_FPGara.get(j, i) + self._loss_kernel(yg, co),
                    )

    def _loss_kernel(self, yg: Float64, co: Float64) -> Float64:
        """Sum over the channels that change the nucleon (loss rate kernel)."""
        var v = 0.0
        for it in range(28):
            if IT1MpPRO(it) < 0.0001 and IT1MpNTR(it) < 0.0001:
                continue
            v += IT1Er(it) * IT1Bout(it) * IT1G(it) * pg_fres(it, yg) * co
        for it in range(9):
            if IT2MpPRO(it) < 0.0001 and IT2MpNTR(it) < 0.0001:
                continue
            v += pg_fdir(it, yg) * co
        for it in range(14):
            if IT3MpPRO(it) < 0.0001 and IT3MpNTR(it) < 0.0001:
                continue
            v += IT3sig(it) * pg_fmult(it, yg) * co
        return v

    def Calc_PhotoPion_ProtonNeutronLoss(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing photo-pion proton/neutron cooling term")
        if self.KER_FPGapn.rows == 0:
            fill_zero(self.afPG_PRO)
        else:
            gemv_d(
                self.KER_FPGapn,
                pd(dat.n[P_photons_on_had_grid]),
                1.0,
                pd(self.afPG_PRO),
            )
        self.afPG_NTR = self.afPG_PRO.copy()

    def Calc_PhotoPion_PhotonLoss(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing photo-pion photon loss term")
        if self.KER_FPGara.rows == 0:
            fill_zero(self.afPG_Rad)
            return
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = dat.n[P_protons][i] + dat.n[P_neutrons][i]
        gemv_d(self.KER_FPGara, pd(self.w_in), 1.0, pd(self.afPG_Rad))

    def Calc_PhotoPion_PhotonConvolution(mut self, dat: AM3Arrays):
        """Channel responses res[c, i] = sum_j K_c(i, j) v_j to the normalised
        target photon field v (Hankel products)."""
        if self.verbose:
            print("computing photo-pion photon-convolution")
        var nc = self.ncols
        self.EigenRN = 1.0e-50
        for k in range(nc):
            self.EigenRN += dat.n[P_photons_on_had_grid][k]
        resize_zero(self.w_vphot, nc)
        for k in range(nc):
            self.w_vphot[k] = dat.n[P_photons_on_had_grid][k] / self.EigenRN
        if not self.kern_valid:
            fill_zero(self.res)
            return
        # only target bins [vlo, vhi) hold photons
        var vlo = nc
        var vhi = 0
        for k in range(nc):
            if self.w_vphot[k] != 0.0:
                vlo = min(vlo, k)
                vhi = k + 1
        if vhi <= vlo:
            fill_zero(self.res)
            return
        var job = PGConvJob(
            pd(self.H),
            pd(self.res),
            pd(self.w_vphot).unsafe_offset(vlo),
            self.nhk,
            dat.nbins_had,
            vlo,
            vhi - vlo,
            pi_(self.hlo),
            pi_(self.hhi),
        )
        parallel_run[pg_conv_work](job, self.nthreads)

    def _reinj_channel(
        mut self,
        dat: AM3Arrays,
        kai: Float64,
        kaiPN: Float64,
        res_c: FPtr,
        MpPRO: Float64,
        MnPRO: Float64,
        MpNTR: Float64,
        MnNTR: Float64,
        co: Float64,
    ):
        var qq = c_log(kai) / dat.dlnE
        var pp = c_log(kaiPN) / dat.dlnE
        var p: Int
        var pr: Float64
        p, pr = number_split(pp, dat.dlnE)
        _ = qq
        var nh = dat.nbins_had
        var QP = pd(self.QPG_PROGen)
        var QN = pd(self.QPG_NTRGen)
        var prot = pd(dat.n[P_protons])
        var neut = pd(dat.n[P_neutrons])
        for i in range(nh):
            var rco = co * self.EigenRN * res_c[unsafe_offset=i]
            var p_i = p + i
            var p_i1 = p_i + 1
            var npr = pr * rco
            var vpr = (1.0 - pr) * rco
            if p_i >= 0 and p_i1 < nh:
                var pi_ = prot[unsafe_offset=i]
                var ni = neut[unsafe_offset=i]
                QP[unsafe_offset=p_i] += vpr * (pi_ * MpPRO + ni * MnPRO)
                QP[unsafe_offset=p_i1] += npr * (pi_ * MpPRO + ni * MnPRO)
                QN[unsafe_offset=p_i] += vpr * (pi_ * MpNTR + ni * MnNTR)
                QN[unsafe_offset=p_i1] += npr * (pi_ * MpNTR + ni * MnNTR)

    def Calc_PhotoPion_ProtonNeutronReinj(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing photo-pion proton/neutron reinjection term")
        var co = c0 * dat.dlnE * mubarn
        for i in range(dat.nbins_had):
            self.QPG_PROGen[i] = 0.0
            self.QPG_NTRGen[i] = 0.0
        var nh = dat.nbins_had
        var r = pd(self.res)
        # (serial channel order; the C++ OpenMP loop races on these sums)
        for it in range(28):
            self._reinj_channel(
                dat,
                IT1kai(it),
                IT1kaiPN(it),
                r.unsafe_offset(it * nh),
                IT1MpPRO(it),
                IT1MnPRO(it),
                IT1MpNTR(it),
                IT1MnNTR(it),
                co,
            )
        for it in range(9):
            self._reinj_channel(
                dat,
                IT2kai(it),
                IT2kaiPN(it),
                r.unsafe_offset((28 + it) * nh),
                IT2MpPRO(it),
                IT2MnPRO(it),
                IT2MpNTR(it),
                IT2MnNTR(it),
                co,
            )
        for it in range(14):
            self._reinj_channel(
                dat,
                IT3kai(it),
                IT3kaiPN(it),
                r.unsafe_offset((37 + it) * nh),
                IT3MpPRO(it),
                IT3MnPRO(it),
                IT3MpNTR(it),
                IT3MnNTR(it),
                co,
            )

    def _rad_channel(
        mut self,
        dat: AM3Arrays,
        kai: Float64,
        res_c: FPtr,
        MpPi0: Float64,
        MnPi0: Float64,
        MpPip: Float64,
        MnPip: Float64,
        MpPim: Float64,
        MnPim: Float64,
        co: Float64,
    ):
        var qq = c_log(kai) / dat.dlnE
        var q: Int
        var qr: Float64
        q, qr = number_split(qq, dat.dlnE)
        var nh = dat.nbins_had
        var Q0 = pd(self.qpi0)
        var QP = pd(self.qpip)
        var QM = pd(self.qpim)
        var prot = pd(dat.n[P_protons])
        var neut = pd(dat.n[P_neutrons])
        for i in range(nh):
            var rco = co * self.EigenRN * res_c[unsafe_offset=i]
            var qi = q + i
            var qi1 = qi + 1
            var nqr = qr * rco
            var vqr = (1.0 - qr) * rco
            if qi >= 0 and qi1 < nh:
                var pi_ = prot[unsafe_offset=i]
                var ni = neut[unsafe_offset=i]
                Q0[unsafe_offset=qi] += vqr * (pi_ * MpPi0 + ni * MnPi0)
                Q0[unsafe_offset=qi1] += nqr * (pi_ * MpPi0 + ni * MnPi0)
                QP[unsafe_offset=qi] += vqr * (pi_ * MpPip + ni * MnPip)
                QP[unsafe_offset=qi1] += nqr * (pi_ * MpPip + ni * MnPip)
                QM[unsafe_offset=qi] += vqr * (pi_ * MpPim + ni * MnPim)
                QM[unsafe_offset=qi1] += nqr * (pi_ * MpPim + ni * MnPim)

    def Calc_PhotoPion_Rad(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing photo-pion pion-generation term")
        var co = c0 * dat.dlnE * mubarn
        for i in range(dat.nbins_had):
            self.qpi0[i] = 0.0
            self.qpip[i] = 0.0
            self.qpim[i] = 0.0
        var nh = dat.nbins_had
        var r = pd(self.res)
        for it in range(28):
            self._rad_channel(
                dat,
                IT1kai(it),
                r.unsafe_offset(it * nh),
                IT1MpPi0(it),
                IT1MnPi0(it),
                IT1MpPip(it),
                IT1MnPip(it),
                IT1MpPim(it),
                IT1MnPim(it),
                co,
            )
        for it in range(9):
            self._rad_channel(
                dat,
                IT2kai(it),
                r.unsafe_offset((28 + it) * nh),
                IT2MpPi0(it),
                IT2MnPi0(it),
                IT2MpPip(it),
                IT2MnPip(it),
                IT2MpPim(it),
                IT2MnPim(it),
                co,
            )
        for it in range(14):
            self._rad_channel(
                dat,
                IT3kai(it),
                r.unsafe_offset((37 + it) * nh),
                IT3MpPi0(it),
                IT3MnPi0(it),
                IT3MpPip(it),
                IT3MnPip(it),
                IT3MpPim(it),
                IT3MnPim(it),
                co,
            )
