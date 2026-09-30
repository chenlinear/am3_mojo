"""Port of ProtonProton (ProtonProton.cc): p + p -> pions (Kelner et al. 2006,
Kafexhiu et al. 2014)."""

from std.math import fma
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.linalg import MatD, gemv_d


def getIndexBelow(arr: Vec, n: Int, target: Float64) -> Int:
    if target <= arr[0]:
        return 0
    if target >= arr[n - 1]:
        return n - 1
    var count = 1
    while arr[count] < target:
        count += 1
    return count


def InterpolateOnNewGrid(
    oldgrid: Vec, arr: Vec, n: Int, new_position: Float64
) -> Float64:
    var idx_below = getIndexBelow(oldgrid, n, new_position)
    if idx_below == n - 1:
        idx_below = n - 2
    return arr[idx_below] + (
        (arr[idx_below + 1] - arr[idx_below])
        / (oldgrid[idx_below + 1] - oldgrid[idx_below])
    ) * (new_position - oldgrid[idx_below])


struct ProtonProton(Copyable, Movable):
    var verbose: Bool
    var E_trans: Float64
    var K_pi: Float64
    var Tp_th: Float64
    var sigma_inel_pipm: Vec
    var sigma_inel_pi0: Vec
    var egrid_shifted: Vec
    var sigma_inel_pipm_shifted: Vec
    var sigma_inel_pi0_shifted: Vec
    var kappa_prot_pipm: Vec
    var kappa_prot_pi0: Vec
    var F_pipm: MatD
    var F_pi0: MatD
    var KER_PP2Pipm: MatD
    var KER_PP2Pi0: MatD
    var KER_FP2Pi02GG: MatD  # (nbins_pho x nbins_had)
    var qpi0: Vec
    var qpip: Vec
    var qpim: Vec
    var APP_PRO: Vec
    var epsilon_gamma_p2pi02GG: Vec
    var w_prot: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.E_trans = 100.0
        self.K_pi = 0.112478
        self.Tp_th = 2.0 * masspi0 + (masspi0 * masspi0) / (2.0 * massp)
        self.sigma_inel_pipm = Vec()
        self.sigma_inel_pi0 = Vec()
        self.egrid_shifted = Vec()
        self.sigma_inel_pipm_shifted = Vec()
        self.sigma_inel_pi0_shifted = Vec()
        self.kappa_prot_pipm = Vec()
        self.kappa_prot_pi0 = Vec()
        self.F_pipm = MatD()
        self.F_pi0 = MatD()
        self.KER_PP2Pipm = MatD()
        self.KER_PP2Pi0 = MatD()
        self.KER_FP2Pi02GG = MatD()
        self.qpi0 = Vec()
        self.qpip = Vec()
        self.qpim = Vec()
        self.APP_PRO = Vec()
        self.epsilon_gamma_p2pi02GG = Vec()
        self.w_prot = Vec()
        self.reinitialize_empty_kernels(dat)
        self.clear_internal_arrays(dat)
        self.egrid_shifted = zeros(dat.nbins_had)
        for i in range(dat.nbins_had):
            self.egrid_shifted[i] = dat.egrid_had[i] / self.K_pi + massp

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.qpi0, dat.nbins_had)
        resize_zero(self.qpip, dat.nbins_had)
        resize_zero(self.qpim, dat.nbins_had)
        resize_zero(self.APP_PRO, dat.nbins_had)
        resize_zero(self.epsilon_gamma_p2pi02GG, dat.nbins_had)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        var nh = dat.nbins_had
        self.sigma_inel_pipm = zeros(nh)
        self.sigma_inel_pi0 = zeros(nh)
        self.sigma_inel_pipm_shifted = zeros(nh)
        self.sigma_inel_pi0_shifted = zeros(nh)
        self.kappa_prot_pipm = zeros(nh)
        self.kappa_prot_pi0 = zeros(nh)
        self.F_pipm = MatD(nh, nh)
        self.F_pi0 = MatD(nh, nh)
        self.KER_PP2Pipm = MatD(nh, nh)
        self.KER_PP2Pi0 = MatD(nh, nh)
        self.KER_FP2Pi02GG = MatD(dat.nbins_pho, nh)

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        pi0_into_cascade: Bool,
        EtransGeV: Float64,
        pp_model_charged_pions_in: Int,
        pp_model_neutral_pions_in: Int,
    ):
        if EtransGeV < 0.0:
            print(
                (
                    "transition energy [GeV] for pp delta approx. is set to a"
                    " value below 0! EtransGeV = "
                ),
                EtransGeV,
                ". Using 100 GeV instead.",
                sep="",
            )
            self.E_trans = 100.0
        else:
            self.E_trans = EtransGeV
        var pp_model_charged_pions = pp_model_charged_pions_in
        if not (pp_model_charged_pions >= 0 and pp_model_charged_pions <= 1):
            print(
                (
                    "cascade for pp can be either 0 or 1!"
                    " pp_model_charged_pions = "
                ),
                pp_model_charged_pions,
                ". Using 0=SIBYLL instead.",
                sep="",
            )
            pp_model_charged_pions = 0
        var pp_model_neutral_pions = pp_model_neutral_pions_in
        if not (pp_model_neutral_pions >= 0 and pp_model_neutral_pions <= 3):
            print(
                (
                    "cascade for pp can be either 0, 1, 2 or 3!"
                    " pp_model_neutral_pions = "
                ),
                pp_model_neutral_pions,
                ". Using 0=SIBYLL instead.",
                sep="",
            )
            pp_model_neutral_pions = 0
        var nh = dat.nbins_had
        for i in range(nh):
            self.sigma_inel_pipm[i] = self.sigma_inel_Kafexhiu(
                dat.egrid_had[i], masspip
            )
            self.sigma_inel_pi0[i] = self.sigma_inel_Kafexhiu(
                dat.egrid_had[i], masspi0
            )
        if self.verbose:
            print("computing proton proton emission kernel")
        for j in range(nh):
            var Ej = dat.egrid_had[j]
            if Ej > self.E_trans / 5.0:
                for i in range(nh):
                    var Ei = dat.egrid_had[i]
                    if Ei < Ej:
                        var F_over_cutoff: Float64
                        if pp_model_charged_pions == 0:
                            var L = c_log(0.001 * Ej)
                            var a = 3.67 + 0.83 * L + 0.075 * L * L
                            var B = a + 0.25
                            var r = 2.6 / c_sqrt(a)
                            var alpha = 0.98 / c_sqrt(a)
                            var x = Ei / Ej
                            var xtoalphaminusone = c_pow(x, alpha - 1.0)
                            var xtoalpha = c_pow(x, alpha)
                            var to_exponate = (1.0 - xtoalpha) / (
                                1.0 + r * xtoalpha * (1.0 - xtoalpha)
                            )
                            F_over_cutoff = (
                                4.0
                                * alpha
                                * B
                                * xtoalphaminusone
                                * c_pow(to_exponate, 4.0)
                                * (
                                    1.0 / (1.0 - xtoalpha)
                                    + r
                                    * (1.0 - 2.0 * xtoalpha)
                                    / (1.0 + r * xtoalpha * (1.0 - xtoalpha))
                                )
                            )
                        else:
                            var L = c_log(0.001 * Ej)
                            var B = 5.58 + 0.78 * L + 0.1 * L * L
                            var r = 3.1 / c_pow(B, 1.5)
                            var alpha = (
                                0.89 / c_sqrt(B) / (1.0 - c_exp(-0.33 * B))
                            )
                            var x = Ei / Ej
                            var xtoalphaminusone = c_pow(x, alpha - 1.0)
                            var xtoalpha = c_pow(x, alpha)
                            var to_exponate = (1.0 - xtoalpha) / c_pow(
                                1.0 + r * xtoalpha, 3.0
                            )
                            F_over_cutoff = (
                                4.0
                                * alpha
                                * B
                                * xtoalphaminusone
                                * c_pow(to_exponate, 4.0)
                                * (
                                    1.0 / (1.0 - xtoalpha)
                                    + 3.0 * r / (1.0 + r * xtoalpha)
                                )
                            )
                        if Ei > masspi0:
                            var cutoff = c_sqrt(1.0 - masspi0 / Ei)
                            self.F_pi0.set(i, j, F_over_cutoff * cutoff)
                            self.KER_PP2Pi0.set(
                                i,
                                j,
                                F_over_cutoff
                                * cutoff
                                * Ei
                                / Ej
                                * c0
                                * self.sigma_inel_pi0[j],
                            )
                        if Ei > masspip:
                            var cutoff = c_sqrt(1.0 - masspip / Ei)
                            self.F_pipm.set(i, j, F_over_cutoff * cutoff)
                            self.KER_PP2Pipm.set(
                                i,
                                j,
                                F_over_cutoff
                                * cutoff
                                * Ei
                                / Ej
                                * c0
                                * self.sigma_inel_pipm[j],
                            )
        if self.verbose:
            print("computing proton proton multiplicites")
        for j in range(nh):
            var Ej = dat.egrid_had[j]
            if Ej >= self.E_trans:
                var Ekin = Ej - massp
                for i in range(1, j):
                    var Ei = dat.egrid_had[i]
                    self.kappa_prot_pi0[j] += (
                        dat.dlnE * self.F_pi0.get(i, j) * Ei * Ei / Ej / Ekin
                    )
                    self.kappa_prot_pipm[j] += (
                        dat.dlnE * self.F_pipm.get(i, j) * Ei * Ei / Ej / Ekin
                    )
        if self.verbose:
            print("computing proton proton delta-approx. kernels")
        var idx_etrans = getIndexBelow(dat.egrid_had, nh, self.E_trans) + 1
        self.K_pi = self.kappa_prot_pi0[idx_etrans]
        for j in range(nh):
            if dat.egrid_had[j] < self.E_trans:
                self.kappa_prot_pipm[j] = self.K_pi
                self.kappa_prot_pi0[j] = self.K_pi
        for i in range(nh):
            self.egrid_shifted[i] = dat.egrid_had[i] / self.K_pi + massp
            self.sigma_inel_pipm_shifted[i] = self.sigma_inel_Kafexhiu(
                self.egrid_shifted[i], masspip
            )
            self.sigma_inel_pi0_shifted[i] = self.sigma_inel_Kafexhiu(
                self.egrid_shifted[i], masspi0
            )
        if self.verbose:
            print("computing proton proton direct pp->pi0->gamma kernel")
        for i in range(dat.nbins_pho):
            var Eg = dat.egrid_had_photons[i]
            for j in range(nh):
                var Ep = dat.egrid_had[j]
                var ds: Float64
                if pp_model_neutral_pions == 0:
                    ds = self.dsigma_dEgamma_SIBYLL(Ep, Eg)
                elif pp_model_neutral_pions == 1:
                    ds = self.dsigma_dEgamma_QGSJET(Ep, Eg)
                elif pp_model_neutral_pions == 2:
                    ds = self.dsigma_dEgamma_Geant4(Ep, Eg)
                else:
                    ds = self.dsigma_dEgamma_Pythia(Ep, Eg)
                self.KER_FP2Pi02GG.set(i, j, dat.dlnE * Eg * ds * c0)

    def sigma_inel_Kafexhiu(self, Eproton: Float64, mpi: Float64) -> Float64:
        var T_th = 2.0 * mpi + mpi * mpi / (2.0 * massp)
        var T_p = Eproton - massp
        var L = c_log(T_p / T_th)
        var cutoff = c_pow(T_th / T_p, 1.9)
        var mb2cm = 1e-27
        if T_p > T_th:
            return (
                (30.7 - 0.96 * L + 0.18 * L * L)
                * c_pow(1.0 - cutoff, 3.0)
                * mb2cm
            )
        return 0.0

    def Calc_PP_ProtonLoss(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing proton-proton proton-cooling term")
        var nh = len(self.kappa_prot_pipm)
        resize_zero(self.APP_PRO, nh)
        var f = -rp.pp_target_proton_density * c0
        for i in range(nh):
            self.APP_PRO[i] = f * (
                2.0 * (self.kappa_prot_pipm[i] * self.sigma_inel_pipm[i])
                + self.kappa_prot_pi0[i] * self.sigma_inel_pi0[i]
            )

    def Calc_PP_PionSrc(
        mut self, rp: RunParams, dat: AM3Arrays, pi0_into_cascade: Bool
    ):
        if self.verbose:
            print("computing proton-proton pion-generation term")
        var nh = dat.nbins_had
        var i_trans = to_int(c_log(self.E_trans * self.K_pi) / dat.dlnE)
        for i in range(nh):
            self.qpip[i] = 0.0
            self.qpim[i] = 0.0
            self.qpi0[i] = 0.0
        var dens = rp.pp_target_proton_density
        for i in range(i_trans):
            var es = self.egrid_shifted[i]
            var N_P_shifted = (
                InterpolateOnNewGrid(dat.egrid_had, dat.n[P_protons], nh, es)
                / es
            )
            var t = (
                N_P_shifted
                * self.sigma_inel_pipm_shifted[i]
                * dens
                * c0
                * dat.egrid_had[i]
                / self.K_pi
            )
            self.qpip[i] += t
            self.qpim[i] += t
            if pi0_into_cascade:
                self.qpi0[i] += (
                    N_P_shifted
                    * self.sigma_inel_pi0_shifted[i]
                    * dens
                    * c0
                    * dat.egrid_had[i]
                    / self.K_pi
                )
        var Kpm = self.KER_PP2Pipm.ptr()
        var K0 = self.KER_PP2Pi0.ptr()
        var prot = pd(dat.n[P_protons])
        var nr = self.KER_PP2Pipm.rows
        for i in range(max(i_trans, 0), nh):
            var sp = self.qpip[i]
            var sm = self.qpim[i]
            var s0 = self.qpi0[i]
            for j in range(nh):
                var base = dat.dlnE * prot[unsafe_offset=j] * dens
                var kpm = Kpm[unsafe_offset=j * nr + i]
                sp += base * kpm
                sm += base * kpm
                if pi0_into_cascade:
                    s0 += base * K0[unsafe_offset=j * nr + i]
            self.qpip[i] = sp
            self.qpim[i] = sm
            self.qpi0[i] = s0

    def Calc_PP_P2Pi02GG(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing proton-proton direct pi0->gamma+gamma term")
        self.w_prot = dat.n[P_protons].copy()
        for j in range(3):
            self.w_prot[j] = 0.0
        resize_zero(self.epsilon_gamma_p2pi02GG, dat.nbins_pho)
        gemv_d(
            self.KER_FP2Pi02GG,
            pd(self.w_prot),
            rp.pp_target_proton_density,
            pd(self.epsilon_gamma_p2pi02GG),
        )

    # ---------------- Kafexhiu et al. 2014 pi0 -> gamma parameterisation --------
    def dsigma_dEgamma_SIBYLL(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var mb2cm = mubarn * 1e3
        var Tp = Eproton - massp
        return mb2cm * self.Amax_SIBYLL(Tp) * self.F_SIBYLL(Tp, Egamma)

    def dsigma_dEgamma_QGSJET(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var mb2cm = mubarn * 1e3
        var Tp = Eproton - massp
        return mb2cm * self.Amax_QGSJET(Tp) * self.F_QGSJET(Tp, Egamma)

    def dsigma_dEgamma_Geant4(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var mb2cm = mubarn * 1e3
        var Tp = Eproton - massp
        return mb2cm * self.Amax_Geant4(Tp) * self.F_Geant4(Tp, Egamma)

    def dsigma_dEgamma_Pythia(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var mb2cm = mubarn * 1e3
        var Tp = Eproton - massp
        return mb2cm * self.Amax_Pythia(Tp) * self.F_Pythia(Tp, Egamma)

    def Amax_SIBYLL(self, Tp: Float64) -> Float64:
        var theta_p = Tp / massp
        var Ltheta_p = c_log(theta_p)
        var Amax = 0.0
        if Tp > self.Tp_th and Tp <= 100.0:
            Amax = self.Amax_Geant4(Tp)
        if Tp > 100.0:
            Amax = (
                10.77
                * c_pow(theta_p, -0.412)
                * c_exp(0.01264 * (Ltheta_p * Ltheta_p))
                * self.sigma_pi_SIBYLL(Tp)
                / massp
            )
        return Amax

    def Amax_QGSJET(self, Tp: Float64) -> Float64:
        var theta_p = Tp / massp
        var Ltheta_p = c_log(theta_p)
        var Amax = 0.0
        if Tp > self.Tp_th and Tp <= 100.0:
            Amax = self.Amax_Geant4(Tp)
        if Tp > 100.0:
            Amax = (
                13.16
                * c_pow(theta_p, -0.4419)
                * c_exp(0.01439 * (Ltheta_p * Ltheta_p))
                * self.sigma_pi_QGSJET(Tp)
                / massp
            )
        return Amax

    def Amax_Geant4(self, Tp: Float64) -> Float64:
        var theta_p = Tp / massp
        var Ltheta_p = c_log(theta_p)
        var Amax = 0.0
        if Tp > self.Tp_th and Tp < 1.0:
            Amax = 5.9 * self.sigma_pi_Geant4(Tp) / self.E_pi_max_LAB(Tp)
        elif Tp >= 1.0 and Tp < 5.0:
            Amax = (
                9.53
                * c_pow(theta_p, -0.52)
                * c_exp(0.054 * (Ltheta_p * Ltheta_p))
                * self.sigma_pi_Geant4(Tp)
                / massp
            )
        else:
            Amax = (
                9.13
                * c_pow(theta_p, -0.35)
                * c_exp(9.7e-3 * (Ltheta_p * Ltheta_p))
                * self.sigma_pi_Geant4(Tp)
                / massp
            )
        return Amax

    def Amax_Pythia(self, Tp: Float64) -> Float64:
        var theta_p = Tp / massp
        var Ltheta_p = c_log(theta_p)
        var Amax = 0.0
        if Tp > self.Tp_th and Tp <= 50.0:
            Amax = self.Amax_Geant4(Tp)
        if Tp > 50.0:
            Amax = (
                9.06
                * c_pow(theta_p, -0.3795)
                * c_exp(0.01105 * (Ltheta_p * Ltheta_p))
                * self.sigma_pi_Pythia(Tp)
                / massp
            )
        return Amax

    def _XY(self, Tp: Float64, Egamma: Float64) -> Tuple[Float64, Float64]:
        """(X, Y0) as in the F_* functions (pow(masspi0, 2.0) == masspi0^2)."""
        var mpi2 = masspi0 * masspi0
        var Y = Egamma + mpi2 / (4.0 * Egamma)
        var Egm = self.Egamma_max(Tp)
        var Y0 = Egm + mpi2 / (4.0 * Egm)
        var X = (Y - masspi0) / (Y0 - masspi0)
        return (X, Y0)

    def F_SIBYLL(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var C = 3.55 * masspi0 / Y0
        if X >= 0.0 and X < 1.0 and Tp > 50.0:
            return c_pow(1.0 - c_sqrt(X), 3.6) / (1.0 + X / C)
        return self.F_Geant4(Tp, Egamma)

    def F_QGSJET(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var C = 3.55 * masspi0 / Y0
        if X >= 0.0 and X < 1.0 and Tp > 50.0:
            return c_pow(1.0 - c_sqrt(X), 4.5) / (1.0 + X / C)
        return self.F_Geant4(Tp, Egamma)

    def F_Geant4(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var theta = Tp / massp
        var kappa = 3.29 - 0.2 * c_pow(theta, -1.5)
        var q = (Tp - 1.0) / massp
        var C = 3.0 * masspi0 / Y0
        if X >= 0.0 and X < 1.0:
            if Tp > self.Tp_th and Tp < 1.0:
                return c_pow(1.0 - X, kappa)
            elif Tp >= 1.0 and Tp <= 4.0:
                var mu = 1.25 * c_pow(q, 1.25) * c_exp(-1.25 * q)
                var beta_ = mu + 2.45
                var gamma = mu + 1.45
                return c_pow(1.0 - X, beta_) / c_pow(1.0 + X / C, gamma)
            elif Tp > 4.0 and Tp <= 20.0:
                var mu = 1.25 * c_pow(q, 1.25) * c_exp(-1.25 * q)
                var beta_ = 1.5 * mu + 4.95
                var gamma = mu + 1.5
                return c_pow(1.0 - X, beta_) / c_pow(1.0 + X / C, gamma)
            elif Tp > 20.0 and Tp <= 100.0:
                return c_pow(1.0 - c_sqrt(X), 4.2) / (1.0 + X / C)
            elif Tp > 100.0:
                return c_pow(1.0 - c_sqrt(X), 4.9) / (1.0 + X / C)
            return 0.0
        return 0.0

    def F_Pythia(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var C = 3.5 * masspi0 / Y0
        if X >= 0.0 and X < 1.0 and Tp > 50.0:
            return c_pow(1.0 - c_sqrt(X), 4.0) / (1.0 + X / C)
        return self.F_Geant4(Tp, Egamma)

    def sigma_pi_SIBYLL(self, Tp: Float64) -> Float64:
        return (
            self.sigma_1_pi(Tp)
            + self.sigma_2_pi(Tp)
            + self.sigma_inel(Tp) * self.multip_pi0_SIBYLL(Tp)
        )

    def sigma_pi_QGSJET(self, Tp: Float64) -> Float64:
        return (
            self.sigma_1_pi(Tp)
            + self.sigma_2_pi(Tp)
            + self.sigma_inel(Tp) * self.multip_pi0_QGSJET(Tp)
        )

    def sigma_pi_Geant4(self, Tp: Float64) -> Float64:
        return (
            self.sigma_1_pi(Tp)
            + self.sigma_2_pi(Tp)
            + self.sigma_inel(Tp) * self.multip_pi0_Geant4(Tp)
        )

    def sigma_pi_Pythia(self, Tp: Float64) -> Float64:
        return (
            self.sigma_1_pi(Tp)
            + self.sigma_2_pi(Tp)
            + self.sigma_inel(Tp) * self.multip_pi0_Pythia(Tp)
        )

    def sigma_1_pi(self, Tp: Float64) -> Float64:
        var M_RES = 1.1883
        var Gamma_RES = 0.2264
        var sigma_0 = 7.66e-3
        var XS_1_pi = 0.0
        if Tp > self.Tp_th and Tp <= 2.0:
            var s = 2.0 * massp * (Tp + 2.0 * massp)
            var X = c_sqrt(s) - massp
            var t = s - masspi0 * masspi0 - 4.0 * massp * massp
            var eta = c_sqrt(
                t * t - 16.0 * masspi0 * masspi0 * massp * massp
            ) / (2.0 * masspi0 * c_sqrt(s))
            var g_RES = c_sqrt(
                M_RES * M_RES * (M_RES * M_RES + Gamma_RES * Gamma_RES)
            )
            var K_RES = (
                c_sqrt(8.0)
                * M_RES
                * Gamma_RES
                * g_RES
                / (mathpi * c_sqrt(M_RES * M_RES + g_RES))
            )
            var u = X * X - M_RES * M_RES
            var w = M_RES * Gamma_RES
            var f_BW = massp * K_RES / (u * u + w * w)
            XS_1_pi = (
                sigma_0
                * c_pow(eta, 1.95)
                * (1.0 + eta + c_pow(eta, 5.0))
                * c_pow(f_BW, 1.86)
            )
        return XS_1_pi

    def sigma_2_pi(self, Tp: Float64) -> Float64:
        var XS_2_pi = 0.0
        if Tp >= 0.56 and Tp <= 2.0:
            XS_2_pi = 5.7 / (1.0 + c_exp(-9.3 * (Tp - 1.4)))
        return XS_2_pi

    def sigma_inel(self, Tp: Float64) -> Float64:
        var LX = c_log(Tp / self.Tp_th)
        var Threshold = max(0.0, (1.0 - c_pow(self.Tp_th / Tp, 1.9)))
        return (30.7 - 0.96 * LX + 0.18 * (LX * LX)) * c_pow(Threshold, 3.0)

    def multip_pi0_SIBYLL(self, Tp: Float64) -> Float64:
        var m = 0.0
        if Tp <= 100.0:
            m = self.multip_pi0_Geant4(Tp)
        if Tp > 100.0:
            var xi_p = (Tp - 3.0) / massp
            m = (
                5.436
                * c_pow(xi_p, 0.075)
                * (1.0 + c_exp(-0.254 * c_pow(xi_p, 0.166)))
                * (1.0 - c_exp(-0.072 * c_pow(xi_p, 0.25)))
            )
        return m

    def multip_pi0_QGSJET(self, Tp: Float64) -> Float64:
        if Tp <= 100.0:
            return self.multip_pi0_Geant4(Tp)
        var xi_p = (Tp - 3.0) / massp
        return (
            0.908
            * c_pow(xi_p, 0.176)
            * (1.0 + c_exp(-0.0009 * c_pow(xi_p, 0.448)))
            * (1.0 - c_exp(-6.089 * c_pow(xi_p, 0.25)))
        )

    def multip_pi0_Geant4(self, Tp: Float64) -> Float64:
        var m = 0.0
        if Tp > 2.0 and Tp < 5.0:
            var Qp = (Tp - self.Tp_th) / massp
            m = -6.0e-3 + 0.237 * Qp - 0.023 * (Qp * Qp)
        if Tp >= 5.0:
            var xi_p = (Tp - 3.0) / massp
            m = (
                0.728
                * c_pow(xi_p, 0.2503)
                * (1.0 + c_exp(-0.596 * c_pow(xi_p, 0.117)))
                * (1.0 - c_exp(-0.491 * c_pow(xi_p, 0.25)))
            )
        return m

    def multip_pi0_Pythia(self, Tp: Float64) -> Float64:
        if Tp <= 50.0:
            return self.multip_pi0_Geant4(Tp)
        var xi_p = (Tp - 3.0) / massp
        return (
            0.652
            * c_pow(xi_p, 0.1928)
            * (1.0 + c_exp(-0.0016 * c_pow(xi_p, 0.483)))
            * (1.0 - c_exp(-0.488 * c_pow(xi_p, 0.25)))
        )

    def Egamma_max(self, Tp: Float64) -> Float64:
        var gamma_pi_LAB = self.E_pi_max_LAB(Tp) / masspi0
        var Beta_pi_LAB = c_sqrt(1.0 - c_pow(gamma_pi_LAB, -2.0))
        return (masspi0 / 2.0) * gamma_pi_LAB * (1.0 + Beta_pi_LAB)

    def E_pi_max_LAB(self, Tp: Float64) -> Float64:
        var s = 2.0 * massp * (Tp + 2.0 * massp)
        var gamma_CM = (Tp + 2.0 * massp) / c_sqrt(s)
        var E_pi_CM = (s - 4.0 * massp * massp + masspi0 * masspi0) / (
            2.0 * c_sqrt(s)
        )
        var P_pi_CM = c_sqrt(E_pi_CM * E_pi_CM - masspi0 * masspi0)
        var Beta_CM = c_sqrt(1.0 - c_pow(gamma_CM, -2.0))
        return gamma_CM * (E_pi_CM + P_pi_CM * Beta_CM)
