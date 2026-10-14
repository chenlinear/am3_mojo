"""Port of ProtonProton (ProtonProton.cc): p + p -> pions (Kelner et al. 2006,
Kafexhiu et al. 2014)."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from std.builtin.globals import global_constant
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


def kab_pion_F(x: Float64, Ep: Float64, model: Int) -> Float64:
    """Pion energy distribution F_pi(x, E_p) of Kelner, Aharonian & Bugayov
    (2006) without the threshold factor (1 - m_pi / (x E_p))^(1/2): eq. 12
    (SIBYLL, model 0) or eq. 6 (QGSJET, model 1); E_p in GeV."""
    if model == 0:
        var L = c_log(0.001 * Ep)
        var a = 3.67 + 0.83 * L + 0.075 * L * L
        var B = a + 0.25
        var r = 2.6 / c_sqrt(a)
        var alpha = 0.98 / c_sqrt(a)
        var xtoalphaminusone = c_pow(x, alpha - 1.0)
        var xtoalpha = c_pow(x, alpha)
        var to_exponate = (1.0 - xtoalpha) / (
            1.0 + r * xtoalpha * (1.0 - xtoalpha)
        )
        return (
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
    var L = c_log(0.001 * Ep)
    var B = 5.58 + 0.78 * L + 0.1 * L * L
    var r = 3.1 / c_pow(B, 1.5)
    var alpha = 0.89 / c_sqrt(B) / (1.0 - c_exp(-0.33 * B))
    var xtoalphaminusone = c_pow(x, alpha - 1.0)
    var xtoalpha = c_pow(x, alpha)
    var to_exponate = (1.0 - xtoalpha) / c_pow(1.0 + r * xtoalpha, 3.0)
    return (
        4.0
        * alpha
        * B
        * xtoalphaminusone
        * c_pow(to_exponate, 4.0)
        * (1.0 / (1.0 - xtoalpha) + 3.0 * r / (1.0 + r * xtoalpha))
    )


# ---------------------------------------------------------------------------
# kaon production in pp collisions (am3_mojo addition)
# ---------------------------------------------------------------------------
# Kaon energy distribution F_K(x, E_p) = E_p n(x E_p) per inelastic collision,
# x = E_K / E_p, in three ranges of E_p (README, "Production in pp
# collisions"):
#   * KPW_E_MIN <= E_p <= KPW_E_MAX: Koers, Pe'er & Wijers (2006,
#     arXiv:hep-ph/0611219), eqs. 16-18 with Table 3 (PYTHIA 6.324, fitted
#     for 1e3 - 1e6 GeV, checked against kaon data down to sqrt(s) = 14 GeV,
#     E_p ~ 100 GeV);
#   * E_p > KPW_E_MAX (am3_mojo extension, not from a publication): the fit's
#     kaon/pion ratio at E_p = KPW_E_MAX, kept at fixed x (nearly independent
#     of E_p in the fit range), times AM3's pions (Kelner et al. 2006);
#   * E_p < KPW_E_MIN (am3_mojo extension): a delta-functional approximation
#     like Kelner et al.'s for pions (2006, eq. 77), with kaon multiplicities
#     fitted to the NA49 compilation (Anticic et al. 2010, Table 11).
comptime KPW_E_MIN = 100.0  # GeV
comptime KPW_E_MAX = 1.0e6  # GeV
comptime KPW_X_MIN = 1.0e-4  # the ratio is taken at x >= KPW_X_MIN

# Koers et al. (2006) Table 3: p00 p01 p10 p11 p20 p21 p30 p40 p50 p60 p70 p80
# for K+, K-, K0, anti-K0 and pi0
comptime _KPW_K = 0
comptime _KPW_KBAR = 1
comptime _KPW_K0 = 2
comptime _KPW_K0BAR = 3
comptime _KPW_PI0 = 4
comptime _KPW: Array[Float64, 60] = [
    -1.23,
    0.0657,
    -0.147,
    0.0161,
    -0.00411,
    0.493,
    -0.989,
    -0.345,
    0.777,
    -0.235,
    42.7,
    12.0,
    -1.46,
    0.101,
    -0.109,
    0.00865,
    -0.0577,
    0.491,
    -1.22,
    -0.164,
    1.04,
    -0.279,
    18.6,
    4.23,
    -1.29,
    0.0690,
    -0.142,
    0.0154,
    -0.00717,
    0.493,
    -1.03,
    -0.294,
    0.839,
    -0.272,
    33.8,
    10.1,
    -1.50,
    0.101,
    -0.118,
    0.0101,
    -0.0567,
    0.491,
    -1.25,
    -0.169,
    1.05,
    -0.272,
    21.2,
    4.07,
    -0.456,
    0.0950,
    -0.142,
    0.0135,
    -0.457,
    0.494,
    -1.49,
    -0.981,
    1.01,
    0.723,
    22.1,
    8.53,
]


@always_inline
def _kpw(sp: Int, k: Int) -> Float64:
    return global_constant[_KPW]()[sp * 12 + k]


@always_inline
def _log10(x: Float64) -> Float64:
    return c_log(x) / 2.302585092994046


@always_inline
def _pow10(x: Float64) -> Float64:
    return c_exp(2.302585092994046 * x)


@always_inline
def _tanh_pos(x: Float64) -> Float64:
    """tanh(x) for x >= 0."""
    var t = c_exp(-2.0 * x)
    return (1.0 - t) / (1.0 + t)


@always_inline
def kpw_mass(sp: Int) -> Float64:
    if sp == _KPW_PI0:
        return massPi0_PDG
    return massK if sp <= _KPW_KBAR else massK0


def kpw_n(sp: Int, eps: Float64, Ep: Float64) -> Float64:
    """Kaon energy spectrum dN/d eps [1/GeV] per inelastic pp collision of a
    proton of energy Ep on a proton at rest (Koers et al. 2006, eqs. 16-18;
    energies in GeV, log = log10)."""
    var m = kpw_mass(sp)
    if eps <= m or eps >= Ep:
        return 0.0
    var lE = _log10(Ep)
    var le = _log10(eps)
    var p0 = _kpw(sp, 0) + _kpw(sp, 1) * lE
    var p1 = _kpw(sp, 2) + _kpw(sp, 3) * lE
    var p2 = _kpw(sp, 4) + _kpw(sp, 5) * lE
    var p4 = _kpw(sp, 7) - p2
    var p6 = _kpw(sp, 9) + p2
    var n0 = 1.21 * _pow10(p0 + p1 * p2 * p2)
    var alpha = _pow10(p1 * (le - 2.0 * p2) * le)  # eps^(p1 (log eps - 2 p2))
    var e3 = _pow10(p2 + p4)
    var e5 = _pow10(p2 + p6)
    var beta = _pow10(-c_pow(eps / e3, _kpw(sp, 6))) * _pow10(
        -c_pow(eps / e5, _kpw(sp, 8))
    )
    var g1 = _tanh_pos(_kpw(sp, 10) * _log10(eps / m))
    var g2 = _tanh_pos(_kpw(sp, 11) * _log10(Ep / eps))
    return n0 / eps * alpha * beta * g1 * g2


def _kpw_F(sp: Int, x: Float64, Ep: Float64, model: Int) -> Float64:
    """F(x, Ep) = Ep n(x Ep) of species sp, extended above KPW_E_MAX as
    F = R(x) F_pi(x, Ep), with Kelner et al.'s pion distribution F_pi (without
    its threshold factor) and the kaon/pion ratio at KPW_E_MAX
    R(x) = KPW_E_MAX n_K(x_s KPW_E_MAX) / F_pi(x_s, KPW_E_MAX),
    x_s = max(x, KPW_X_MIN). Below KPW_X_MIN (kaon energies below
    e_s = KPW_X_MIN KPW_E_MAX = 100 GeV), R follows Koers et al.'s K/pi0 ratio
    rho(eps) = n_K / n_pi0 at KPW_E_MAX (eps: kaon energy), which holds the
    kaon mass threshold: R(x) = R(KPW_X_MIN) rho(min(x Ep, e_s)) / rho(e_s).
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0
    if Ep <= KPW_E_MAX:
        return Ep * kpw_n(sp, x * Ep, Ep)
    var xs = max(x, KPW_X_MIN)
    var phi_star = kab_pion_F(xs, KPW_E_MAX, model)
    if phi_star <= 0.0:
        return 0.0
    var ratio = KPW_E_MAX * kpw_n(sp, xs * KPW_E_MAX, KPW_E_MAX) / phi_star
    if x < KPW_X_MIN:
        var es = KPW_X_MIN * KPW_E_MAX
        var e2 = min(x * Ep, es)
        var n_pi = kpw_n(_KPW_PI0, e2, KPW_E_MAX)
        if n_pi <= 0.0:
            return 0.0
        var rho = kpw_n(sp, e2, KPW_E_MAX) / n_pi
        var rho_s = kpw_n(sp, es, KPW_E_MAX) / kpw_n(_KPW_PI0, es, KPW_E_MAX)
        ratio *= rho / rho_s
    return ratio * kab_pion_F(x, Ep, model)


def pp_kaon_F(s: Int, x: Float64, Ep: Float64, model: Int) -> Float64:
    """Energy distribution F_K(x, Ep) of kaon species s (K+, K-, K0_S, K0_L)
    for E_p >= KPW_E_MIN; a K0 or anti-K0 is half K0_S, half K0_L."""
    if s == KP:
        return _kpw_F(_KPW_K, x, Ep, model)
    if s == KM:
        return _kpw_F(_KPW_KBAR, x, Ep, model)
    return 0.5 * (
        _kpw_F(_KPW_K0, x, Ep, model) + _kpw_F(_KPW_K0BAR, x, Ep, model)
    )


def pp_kaon_moments(s: Int, Ep: Float64, model: Int) -> Tuple[Float64, Float64]:
    """Multiplicity integral of F dx and energy fraction integral of x F dx
    of species s at Ep (trapezoidal rule in ln x, 4000 intervals)."""
    comptime N = 4000
    var lx0 = c_log(massK / Ep)
    var h = -lx0 / Float64(N)
    var n = 0.0
    var e = 0.0
    for k in range(N + 1):
        var x = c_exp(lx0 + Float64(k) * h)
        var w = h * (0.5 if (k == 0 or k == N) else 1.0)
        var f = pp_kaon_F(s, x, Ep, model) * x  # F dx = F x d ln x
        n += w * f
        e += w * f * x
    return (n, e)


@always_inline
def pp_kaon_sqrt_s_th(s: Int) -> Float64:
    """Threshold sqrt(s) [GeV] of species s: p p -> p Lambda K+ (K+),
    p p -> p p K+ K- (K-), p p -> p Sigma+ K0 (the neutral kaons)."""
    if s == KP:
        return 2.548
    return 2.864 if s == KM else 2.625


def pp_kaon_mult_fit(s: Int, Ep: Float64) -> Float64:
    """Fit to the kaon multiplicities per inelastic pp collision of the NA49
    compilation (Anticic et al. 2010, Table 11; am3_mojo fit):
    n = A (1 - s_th / s)^a (s / s_th)^b, s = 2 m_p^2 + 2 m_p E_p. For the
    neutral kaons it fits <K0_S> = (K0 + anti-K0) / 2."""
    var A: Float64
    var a: Float64
    var b: Float64
    if s == KP:
        A = 0.0942
        a = 3.435
        b = 0.2574
    elif s == KM:
        A = 0.0476
        a = 6.175
        b = 0.3324
    else:
        A = 0.02128
        a = 2.115
        b = 0.5398
    var mp = massP_PDG
    var ss = 2.0 * mp * mp + 2.0 * mp * Ep
    var th = pp_kaon_sqrt_s_th(s)
    var st = th * th
    if ss <= st:
        return 0.0
    return A * c_pow(1.0 - st / ss, a) * c_pow(ss / st, b)


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
    # kaons (am3_mojo addition): kernels built at init_kernels when kaons are
    # switched on (kaons_built). The manager sets per step whether kaons are
    # made (kaons_src) and whether the protons lose their energy (kaons_loss).
    var kaons_built: Bool
    var kaons_src: Bool
    var kaons_loss: Bool
    var qK: List[Vec]  # source of kaon species s (model/particles.mojo)
    var KER_PP2K: List[MatD]  # E_p >= the kaon transition: F_K kernels
    var kd_l: List[VecI]  # E_p below: kaons of proton bin j go to bins
    var kd_w: List[Vec]  # kd_l, kd_l + 1 (weight kd_w of the upper one)
    var kd_n: List[Vec]  # kaons per unit proton density and time
    # kappa_K: fraction of E_p carried by all kaons. AM3's kappa_prot_* are
    # fractions of T_p (as in its C++ code); kappa_K is one of E_p, so that the
    # protons lose exactly the energy of the kaon source.
    var kappa_K: Vec

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
        self.kaons_built = False
        self.kaons_src = False
        self.kaons_loss = False
        self.qK = List[Vec]()
        self.KER_PP2K = List[MatD]()
        self.kd_l = List[VecI]()
        self.kd_w = List[Vec]()
        self.kd_n = List[Vec]()
        for _ in range(N_KAON):
            self.qK.append(Vec())
            self.KER_PP2K.append(MatD())
            self.kd_l.append(VecI())
            self.kd_w.append(Vec())
            self.kd_n.append(Vec())
        self.kappa_K = Vec()
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
        for s in range(N_KAON):
            resize_zero(self.qK[s], dat.nbins_had)

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
        self.egrid_shifted = zeros(nh)  # (on the current grid)
        # no kaon kernels until initialize_kernels builds them for this grid
        self.kappa_K = zeros(nh)
        for s in range(N_KAON):
            self.KER_PP2K[s] = MatD()
            self.kd_l[s] = VecI()
            self.kd_w[s] = Vec()
            self.kd_n[s] = Vec()
        self.kaons_built = False
        self.kaons_src = False
        self.kaons_loss = False

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        pi0_into_cascade: Bool,
        EtransGeV: Float64,
        pp_model_charged_pions_in: Int,
        pp_model_neutral_pions_in: Int,
        build_kaons: Bool,
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
                        var F_over_cutoff = kab_pion_F(
                            Ei / Ej, Ej, pp_model_charged_pions
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
        # (kept on the grid when E_trans lies at or above its top)
        var idx_etrans = min(
            getIndexBelow(dat.egrid_had, nh, self.E_trans) + 1, nh - 1
        )
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
        self.kappa_K = zeros(nh)
        if build_kaons:
            self._init_kaon_kernels(dat, pp_model_charged_pions)
            self.kaons_built = True

    def _init_kaon_kernels(mut self, dat: AM3Arrays, model: Int):
        """Kaon production kernels (am3_mojo addition, see pp_kaon_F)."""
        if self.verbose:
            print("computing proton proton kaon kernels")
        var nh = dat.nbins_had
        # Koers et al. from 100 GeV on, whatever the transition energy of the
        # pions (the multiplicity fit below covers only sqrt(s) <= 63 GeV)
        var Et = KPW_E_MIN
        var mp = massP_PDG
        for s in range(N_KAON):
            self.KER_PP2K[s] = MatD(nh, nh)
            self.kd_l[s] = VecI(length=nh, fill=-1)
            self.kd_w[s] = zeros(nh)
            self.kd_n[s] = zeros(nh)
            # proton energies from Et on: the distribution F_K
            for j in range(nh):
                var Ej = dat.egrid_had[j]
                if Ej < Et:
                    continue
                for i in range(j):
                    var Ei = dat.egrid_had[i]
                    var F = pp_kaon_F(s, Ei / Ej, Ej, model)
                    self.KER_PP2K[s].set(
                        i, j, F * Ei / Ej * c0 * self.sigma_inel_pipm[j]
                    )
                    # fraction of E_p in the kaons of the kernel (AM3 applies
                    # the pp loss as dE/dt = A E)
                    self.kappa_K[j] += dat.dlnE * F * Ei * Ei / Ej / Ej
            # below: one kaon energy per proton energy, ntilde kaons per
            # collision, matched to F_K at Et in number and mean energy
            var nt: Float64
            var et: Float64
            nt, et = pp_kaon_moments(s, Et, model)
            if nt <= 0.0:
                continue
            var K_s = et * Et / nt / (Et - mp)  # mean fraction of T_p
            var fit_t = pp_kaon_mult_fit(s, Et)
            for j in range(nh):
                var Ej = dat.egrid_had[j]
                if Ej >= Et:
                    continue
                var fit = pp_kaon_mult_fit(s, Ej)
                if fit <= 0.0:
                    continue
                var nj = nt * fit / fit_t
                var Tj = Ej - mp
                var g_cm = c_sqrt((Ej + mp) / (2.0 * mp))
                var mK = kpw_mass(_KPW_K if s <= KM else _KPW_K0)
                var EK = max(K_s * Tj, g_cm * mK)
                # number- and energy-keeping split between two bins
                var l = to_int(c_floor(c_log(EK) / dat.dlnE))
                if l < 0 or l + 1 >= nh:
                    continue  # below the hadron grid: neither made nor lost
                self.kappa_K[j] += nj * EK / Ej
                self.kd_l[s][j] = l
                self.kd_w[s][j] = (EK - dat.egrid_had[l]) / (
                    dat.egrid_had[l + 1] - dat.egrid_had[l]
                )
                self.kd_n[s][j] = nj * c0 * self.sigma_inel_pipm[j]

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
        if self.kaons_loss and len(self.kappa_K) == nh:
            # the energy the kaons carry away (am3_mojo addition)
            for i in range(nh):
                self.APP_PRO[i] += f * self.kappa_K[i] * self.sigma_inel_pipm[i]

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

    def Calc_PP_KaonSrc(mut self, rp: RunParams, dat: AM3Arrays):
        """Kaon sources qK[s] (per ln E) from pp collisions:
        q_K(E_i) = dlnE n_H sum_j N_p(E_j) K[i, j] above the kaon transition
        energy, and the delta-functional term below."""
        var nh = dat.nbins_had
        var dens = rp.pp_target_proton_density
        for s in range(N_KAON):
            resize_zero(self.qK[s], nh)
            if not self.kaons_src or self.KER_PP2K[s].rows != nh:
                continue
            gemv_d(
                self.KER_PP2K[s],
                pd(dat.n[P_protons]),
                dat.dlnE * dens,
                pd(self.qK[s]),
            )
            for j in range(nh):
                var l = self.kd_l[s][j]
                if l < 0:
                    continue
                # number per volume and time, per ln E of the kaon bins
                var r = self.kd_n[s][j] * dens * dat.n[P_protons][j]
                self.qK[s][l] += (1.0 - self.kd_w[s][j]) * r
                self.qK[s][l + 1] += self.kd_w[s][j] * r

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

    # ---------------- Kafexhiu et al. 2014 pi0 -> gamma parameterisation ------
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
        var Amax: Float64
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
