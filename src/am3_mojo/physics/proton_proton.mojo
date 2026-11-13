"""Port of ProtonProton (ProtonProton.cc). The port also makes kaons and charm
hadrons (am3_mojo addition)."""

from std.builtin.globals import global_constant
from std.math import ceil, floor, sqrt
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.physics.decays import charm_mass
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import MatD, gemv_d
from am3_mojo.utils.parallel import parallel_run


def getIndexBelow(arr: Vec, n: Int, target: Float64) -> Int:
    """The last index with arr <= target, within [0, n - 1] (AM3 returned the
    first with arr >= target, the point above: AM3 bug)."""
    if target <= arr[0]:
        return 0
    if target >= arr[n - 1]:
        return n - 1
    var count = 1
    while arr[count] <= target:
        count += 1
    return count - 1


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
    (SIBYLL, model 0) or eq. 6 (QGSJET, model 1), with E_p in GeV."""
    if model == 0:
        var L = c_log(0.001 * Ep)
        var a = 3.67 + 0.83 * L + 0.075 * L * L
        var B = a + 0.25
        var r = 2.6 / sqrt(a)
        var alpha = 0.98 / sqrt(a)
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
    var alpha = 0.89 / sqrt(B) / (1.0 - c_exp(-0.33 * B))
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


# ------------------------------------------------------------------------------
# kaon production in pp collisions (am3_mojo addition)
# ------------------------------------------------------------------------------
# Kaon energy distribution F_K(x, E_p) = E_p n(x E_p) per inelastic collision,
# x = E_K / E_p, in three ranges of E_p
# (tests/literature_checks/kaons/README.md, "Production in pp collisions"):
#   * KPW_E_MIN <= E_p <= KPW_E_MAX: Koers, Pe'er & Wijers (2006,
#     arXiv:hep-ph/0611219), eqs. 16-18 with Table 3 (PYTHIA 6.324, fitted
#     for 1e3 - 1e6 GeV, checked against kaon data down to sqrt(s) = 14 GeV,
#     E_p ~ 100 GeV). Below KPW_FIT_E_MIN, the factor pp_kaon_fade joins
#     these spectra to the multiplicities measured below KPW_E_MIN.
#   * E_p > KPW_E_MAX (am3_mojo extension, not from a publication): the fit's
#     kaon/pion ratio at E_p = KPW_E_MAX, kept at fixed x (nearly independent
#     of E_p in the fit range), times AM3's pions (Kelner et al. 2006).
#   * E_p < KPW_E_MIN (am3_mojo extension): a delta-functional approximation
#     like Kelner et al.'s for pions (2006, eq. 77), with kaon multiplicities
#     fitted to the NA49 compilation (Anticic et al. 2010, Table 11).
comptime KPW_E_MIN = 100.0  # GeV
comptime KPW_FIT_E_MIN = 1.0e3  # GeV, where Koers et al.'s fit range begins
comptime KPW_E_MAX = 1.0e6  # GeV
comptime KPW_X_MIN = 1.0e-4

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


@inline(.always)
def _kpw(sp: Int, k: Int) -> Float64:
    return global_constant[_KPW]()[sp * 12 + k]


@inline(.always)
def _log10(x: Float64) -> Float64:
    return c_log(x) / LN10


@inline(.always)
def _pow10(x: Float64) -> Float64:
    return c_exp(LN10 * x)


@inline(.always)
def _tanh_pos(x: Float64) -> Float64:
    var t = c_exp(-2.0 * x)
    return (1.0 - t) / (1.0 + t)


@inline(.always)
def kpw_mass(sp: Int) -> Float64:
    if sp == _KPW_PI0:
        return MASS_PI0
    return MASS_K if sp <= _KPW_KBAR else MASS_K0


def kpw_n(sp: Int, eps: Float64, Ep: Float64) -> Float64:
    """Kaon energy spectrum dN/d eps [1/GeV] per inelastic pp collision of a
    proton of energy Ep on a proton at rest (Koers et al. 2006, eqs. 16-18).
    Energies are in GeV, and log = log10."""
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
    var alpha = _pow10(p1 * (le - 2.0 * p2) * le)
    var e3 = _pow10(p2 + p4)
    var e5 = _pow10(p2 + p6)
    var beta = _pow10(-c_pow(eps / e3, _kpw(sp, 6))) * _pow10(
        -c_pow(eps / e5, _kpw(sp, 8))
    )
    var g1 = _tanh_pos(_kpw(sp, 10) * _log10(eps / m))
    var g2 = _tanh_pos(_kpw(sp, 11) * _log10(Ep / eps))
    return n0 / eps * alpha * beta * g1 * g2


def _kpw_F(sp: Int, x: Float64, Ep: Float64, model: Int) -> Float64:
    """Below KPW_X_MIN, the kaon/pion ratio follows Koers et al.'s K/pi0 ratio
    rho(eps) = n_K / n_pi0 at KPW_E_MAX (eps: kaon energy). The ratio rho holds
    the kaon mass threshold.
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
    """Energy distribution F_K(x, Ep) of kaon species s for E_p >= KPW_E_MIN.
    A K0 or anti-K0 is half K0_S, half K0_L."""
    if s == KP:
        return _kpw_F(_KPW_K, x, Ep, model)
    if s == KM:
        return _kpw_F(_KPW_KBAR, x, Ep, model)
    return 0.5 * (
        _kpw_F(_KPW_K0, x, Ep, model) + _kpw_F(_KPW_K0BAR, x, Ep, model)
    )


def pp_kaon_moments(s: Int, Ep: Float64, model: Int) -> Tuple[Float64, Float64]:
    comptime N = 4000
    var lx0 = c_log(MASS_K / Ep)
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


def pp_kaon_fade(c_t: Float64, Ep: Float64) -> Float64:
    """At KPW_E_MIN, the factor is c_t, the multiplicity fit over Koers et al.'s
    multiplicity there (pp_kaon_mult_fit, pp_kaon_moments). It goes linearly in
    ln E_p to 1 at KPW_FIT_E_MIN, and it is 1 above. So the kaons per collision
    are continuous at KPW_E_MIN, and they are those of Koers et al. in their own
    fit range (tests/literature_checks/kaons/README.md, "The kaon spectra")."""
    if Ep >= KPW_FIT_E_MIN:
        return 1.0
    var w = c_log(Ep / KPW_E_MIN) / c_log(KPW_FIT_E_MIN / KPW_E_MIN)
    return c_t + (1.0 - c_t) * w


@inline(.always)
def pp_kaon_sqrt_s_th(s: Int) -> Float64:
    """Threshold sqrt(s) [GeV] of species s: p p -> p Lambda K+ (K+),
    p p -> p p K+ K- (K-), p p -> p Sigma+ K0 (the neutral kaons)."""
    if s == KP:
        return 2.548
    return 2.864 if s == KM else 2.625


def pp_kaon_mult_fit(s: Int, Ep: Float64) -> Float64:
    """am3_mojo fit to the kaon multiplicities per inelastic pp collision of the
    NA49 compilation (Anticic et al. 2010, Table 11). The second term grows like
    the phase space of p p -> p Lambda K+ from the threshold. It follows the
    table's two points at sqrt(s) = 2.9 GeV and fades with energy
    (tests/literature_checks/kaons/README.md, "The kaon spectra"). For the
    neutral kaons, the fit is to <K0_S> = (K0 + anti-K0) / 2."""
    var A: Float64
    var a: Float64
    var b: Float64
    var C = 0.0
    var d = 0.0
    if s == KP:
        A = 0.1132
        a = 4.099
        b = 0.2244
        C = 0.3354
        d = 5.29
    elif s == KM:
        A = 0.0476
        a = 6.175
        b = 0.3324
    else:
        A = 0.02128
        a = 2.115
        b = 0.5398
    var mp = MASS_P
    var ss = 2.0 * mp * mp + 2.0 * mp * Ep
    var th = pp_kaon_sqrt_s_th(s)
    var st = th * th
    if ss <= st:
        return 0.0
    var x = 1.0 - st / ss
    var n = A * c_pow(x, a) * c_pow(ss / st, b)
    if C > 0.0:
        n += C * x * x * c_pow(st / ss, d)
    return n


# ------------------------------------------------------------------------------
# charm-hadron production in pp collisions (am3_mojo addition)
# ------------------------------------------------------------------------------
# Charm quarks of a proton of energy E_p on a proton at rest, x = E_c / E_p in
# the lab frame (tests/literature_checks/charm/README.md,
# "Production in pp collisions"): a closed form fitted to the published NLO
# spectra (tests/literature_checks/charm/charm_production.py).
comptime CHARM_E_FIT = 1.0e3  # GeV: the closed form from here on
comptime CHARM_E_TOP = 1.0e11  # GeV: sigma_cc / sigma_inel held above
comptime CPP_MT = 1.66  # m_T [GeV]
comptime CPP_B0 = -1.2836
comptime CPP_B1 = 0.0354
comptime CPP_B2 = -0.0171
comptime CPP_N0 = 2.957
comptime CPP_N1 = 0.126
comptime CPP_L_MAX = 5.0
comptime CPP_SUB = 4
comptime CPP_LN_ZMIN = 4.6  # z of the fragmentation down to exp(-4.6) = 0.01


@inline(.always)
def pp_charm_fraction(h: Int) -> Float64:
    if h == 0:
        return FRAG_D
    if h == 1:
        return FRAG_D0
    return FRAG_DS if h == 2 else FRAG_LC


@inline(.always)
def pp_charm_peterson_eps(h: Int) -> Float64:
    """eps_h of the fragmentation functions of Kniehl & Kramer (2006,
    Table I, LO, c -> hadron)."""
    if h == 0:
        return 0.104
    if h == 1:
        return 0.101
    return 0.0322 if h == 2 else 0.00418


def pp_charm_sigma_cc(Ep: Float64) -> Float64:
    """sigma(pp -> c c-bar X) [cm^2] at a proton energy Ep [GeV] up to
    CHARM_E_TOP. It takes the NLO values of Bhattacharya et al. (2016, Table 1,
    pp, scales m_T) at 1e2, 1e3, ..., 1e10 GeV. The exponent a matches the slope
    of the first decade at 1e2 GeV."""
    var tab: Array[Float64, 9] = [
        1.51,
        38.4,
        252.0,
        858.0,
        2250.0,
        5360.0,
        12100.0,
        26700.0,
        56600.0,
    ]
    var l = c_log10(Ep)
    if l >= 2.0:
        var k = min(Int(floor(l - 2.0)), 7)
        var a = c_log10(tab[k])
        var b = c_log10(tab[k + 1])
        return c_pow(10.0, a + (b - a) * (l - 2.0 - Float64(k))) * MICROBARN
    var mp = MASS_P
    var s = 2.0 * mp * mp + 2.0 * mp * Ep
    var m_th = mp + MASS_LC + MASS_D0
    var s_th = m_th * m_th
    if s <= s_th:
        return 0.0
    var s2 = 2.0 * mp * mp + 2.0 * mp * 100.0
    var a = (
        c_log10(tab[1] / tab[0]) * (s2 - s_th) * s2 / (s_th * 2.0 * mp * 100.0)
    )
    return tab[0] * MICROBARN * c_pow((1.0 - s_th / s) / (1.0 - s_th / s2), a)


def pp_charm_shape(Ep: Float64, d: Float64) -> List[Float64]:
    var mp = MASS_P
    var l = min(c_log10(Ep) - 6.0, CPP_L_MAX)
    var xc = CPP_MT / sqrt(2.0 * mp * mp + 2.0 * mp * Ep)
    var xc2 = xc * xc
    var x0 = c_pow(10.0, CPP_B0 + CPP_B1 * l + CPP_B2 * l * l)
    var n = CPP_N0 + CPP_N1 * l
    var nm = Int(ceil(-c_log(xc2) / d))
    var g = List[Float64](length=nm, fill=0.0)
    var sum = 0.0
    for m in range(1, nm):
        var x = c_exp(-Float64(m) * d)
        if x <= xc2:
            break
        var u = xc2 / x
        g[m] = (
            c_pow(1.0 - x, n)
            / (1.0 + x / x0)
            * c_pow(1.0 - u, n)
            / (1.0 + u / x0)
        )
        sum += g[m]
    for m in range(nm):
        g[m] /= d * sum
    return g^


def pp_charm_fragmentation(h: Int, d: Float64, nz: Int) -> List[Float64]:
    var eps = pp_charm_peterson_eps(h)
    var zD = List[Float64](length=nz, fill=0.0)
    var sum = 0.0
    for k in range(1, nz):
        var z = c_exp(-Float64(k) * d)
        var u = (1.0 - z) * (1.0 - z)
        var den = u + eps * z
        zD[k] = z * z * u / (den * den)
        sum += zD[k]
    for k in range(nz):
        zD[k] *= pp_charm_fraction(h) / (d * sum)
    return zD^


@inline(.always)
def _split_bins(E: Float64, dat: AM3Arrays) -> Tuple[Int, Float64]:
    """The split keeps the number and energy of the hadron. Off the hadron grid,
    l = -1."""
    var l = Int(floor(c_log(E) / dat.dlnE))
    if l < 0 or l + 1 >= dat.nbins_had:
        return (-1, 0.0)
    return (
        l,
        (E - dat.egrid_had[l]) / (dat.egrid_had[l + 1] - dat.egrid_had[l]),
    )


@inline(.always)
def _add_split(
    q: FPtr,
    ref l: VecI,
    ref w: Vec,
    ref n: Vec,
    dens: Float64,
    prot: FPtr,
    nh: Int,
):
    """Add the delta-functional term to the source q (number per volume and
    time, per ln E of the hadron bins)."""
    for j in range(nh):
        var lj = l[j]
        if lj < 0:
            continue
        var r = n[j] * dens * prot[unsafe_offset=j]
        q[unsafe_offset=lj] += (1.0 - w[j]) * r
        q[unsafe_offset=lj + 1] += w[j] * r


struct PPCharmJob(Movable):
    var P: Pointer[ProtonProton, MutUntrackedOrigin]
    var dens: Float64
    var dlnE: Float64

    def __init__(
        out self,
        P: Pointer[ProtonProton, MutUntrackedOrigin],
        dens: Float64,
        dlnE: Float64,
    ):
        self.P = P
        self.dens = dens
        self.dlnE = dlnE


def ppc_work(job: Pointer[PPCharmJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref P = job[].P[]
    var h = t
    while h < 4:
        var s = 2 * h
        ref prot = P.prot_view
        gemv_d(
            P.KER_PP2C[h],
            pd(prot),
            job[].dlnE * job[].dens,
            pd(P.qC[s]),
        )
        _add_split(
            pd(P.qC[s]),
            P.cd_l[h],
            P.cd_w[h],
            P.cd_n[h],
            job[].dens,
            pd(prot),
            len(prot),
        )
        for i in range(len(P.qC[s])):
            P.qC[s + 1][i] = P.qC[s][i]
        h += nt


# GeV: the transition energy of the delta approximation that a negative one
# stands for (as in AM3)
comptime PP_E_TRANS_NEGATIVE = 100.0


struct ProtonProton(Copyable, Movable):
    var verbose: Bool
    var nthreads: Int
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
    var KER_PP2Pipm: MatD
    var KER_PP2Pi0: MatD
    var KER_FP2Pi02GG: MatD
    var qpi0: Vec
    var qpip: Vec
    var qpim: Vec
    var APP_PRO: Vec
    var epsilon_gamma_p2pi02GG: Vec
    var w_prot: Vec
    # Kaons (am3_mojo addition)
    var kaons_built: Bool
    var kaons_src: Bool
    var kaons_loss: Bool
    var qK: List[Vec]
    var KER_PP2K: List[MatD]
    var kd_l: List[VecI]
    var kd_w: List[Vec]
    var kd_n: List[Vec]
    # AM3's kappa_prot_* are fractions of T_p (as in its C++ code). kappa_K is a
    # fraction of E_p, so that the protons lose exactly the energy of the kaon
    # source.
    var kappa_K: Vec
    # Charm hadrons (am3_mojo addition). sig_kappa_C: the cross section times
    # the fraction of E_p carried by all charm hadrons [cm^2].
    var prot_view: Vec  # the protons, for the charm jobs (ppc_work)
    var charm_built: Bool
    var charm_src: Bool
    var charm_loss: Bool
    var qC: List[Vec]
    var KER_PP2C: List[MatD]
    var cd_l: List[VecI]
    var cd_w: List[Vec]
    var cd_n: List[Vec]
    var sig_kappa_C: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.nthreads = 1
        self.E_trans = 100.0
        self.K_pi = 0.112478
        self.Tp_th = 2.0 * MASS_PI0 + (MASS_PI0 * MASS_PI0) / (2.0 * MASS_P)
        self.sigma_inel_pipm = Vec()
        self.sigma_inel_pi0 = Vec()
        self.egrid_shifted = Vec()
        self.sigma_inel_pipm_shifted = Vec()
        self.sigma_inel_pi0_shifted = Vec()
        self.kappa_prot_pipm = Vec()
        self.kappa_prot_pi0 = Vec()
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
        self.prot_view = Vec()
        self.charm_built = False
        self.charm_src = False
        self.charm_loss = False
        self.qC = List[Vec]()
        for _ in range(N_CHARM):
            self.qC.append(Vec())
        self.KER_PP2C = List[MatD]()
        self.cd_l = List[VecI]()
        self.cd_w = List[Vec]()
        self.cd_n = List[Vec]()
        for _ in range(4):
            self.KER_PP2C.append(MatD())
            self.cd_l.append(VecI())
            self.cd_w.append(Vec())
            self.cd_n.append(Vec())
        self.sig_kappa_C = Vec()
        self.reinitialize_empty_kernels(dat)
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.qpi0, dat.nbins_had)
        resize_zero(self.qpip, dat.nbins_had)
        resize_zero(self.qpim, dat.nbins_had)
        resize_zero(self.APP_PRO, dat.nbins_had)
        resize_zero(self.epsilon_gamma_p2pi02GG, dat.nbins_had)
        for s in range(N_KAON):
            resize_zero(self.qK[s], dat.nbins_had)
        for s in range(N_CHARM):
            resize_zero(self.qC[s], dat.nbins_had)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        var nh = dat.nbins_had
        self.sigma_inel_pipm = zeros(nh)
        self.sigma_inel_pi0 = zeros(nh)
        self.sigma_inel_pipm_shifted = zeros(nh)
        self.sigma_inel_pi0_shifted = zeros(nh)
        self.kappa_prot_pipm = zeros(nh)
        self.kappa_prot_pi0 = zeros(nh)
        # (no kernels until initialize_kernels: without them, Calc_PP_PionSrc
        #  and Calc_PP_P2Pi02GG add nothing, as the zero kernels of AM3 do)
        self.KER_PP2Pipm = MatD()
        self.KER_PP2Pi0 = MatD()
        self.KER_FP2Pi02GG = MatD()
        # On the current grid, as the constructor sets it. If pp is switched
        # on after an init_kernels without pp, pp gives no pions (zero cross
        # sections), as in AM3, not 0/0.
        self.egrid_shifted = zeros(nh)
        for i in range(nh):
            self.egrid_shifted[i] = dat.egrid_had[i] / self.K_pi + MASS_P
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
        self.sig_kappa_C = zeros(nh)
        for h in range(4):
            self.KER_PP2C[h] = MatD()
            self.cd_l[h] = VecI()
            self.cd_w[h] = Vec()
            self.cd_n[h] = Vec()
        self.charm_built = False
        self.charm_src = False
        self.charm_loss = False

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        EtransGeV: Float64,
        pp_model_charged_pions: Int,
        pp_model_neutral_pions: Int,
        build_kaons: Bool,
        build_charm: Bool,
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
            self.E_trans = PP_E_TRANS_NEGATIVE
        else:
            self.E_trans = EtransGeV
        # (the setters take only the models there are: 0 or 1 for the
        #  charged pions, 0 to 3 for the neutral ones)
        debug_assert(
            pp_model_charged_pions >= 0 and pp_model_charged_pions <= 1,
            "pp_model_charged_pions must be 0 or 1",
        )
        debug_assert(
            pp_model_neutral_pions >= 0 and pp_model_neutral_pions <= 3,
            "pp_model_neutral_pions must be 0 to 3",
        )
        var nh = dat.nbins_had
        self.KER_PP2Pipm = MatD(nh, nh)
        self.KER_PP2Pi0 = MatD(nh, nh)
        self.KER_FP2Pi02GG = MatD(dat.nbins_pho, nh)
        for i in range(nh):
            self.sigma_inel_pipm[i] = self.sigma_inel_Kafexhiu(
                dat.egrid_had[i], MASS_PI
            )
            self.sigma_inel_pi0[i] = self.sigma_inel_Kafexhiu(
                dat.egrid_had[i], MASS_PI0
            )
        if self.verbose:
            print("computing proton proton emission kernel")
        # A transition energy within 1e-9 of a grid point's energy is that
        # point's energy, as in AM3Arrays._grid_index. The setter's value,
        # times 1e-9, comes out an ulp above the point for a third of the
        # points. This put the point below the transition: AM3 bug.
        var e_at = self.E_trans * (1.0 - 1e-9)
        for j in range(nh):
            var Ej = dat.egrid_had[j]
            if Ej > self.E_trans / 5.0:
                var kap = Ej >= e_at
                var Ekin = Ej - MASS_P
                for i in range(nh):
                    var Ei = dat.egrid_had[i]
                    if Ei < Ej:
                        var F_over_cutoff = kab_pion_F(
                            Ei / Ej, Ej, pp_model_charged_pions
                        )
                        var f0 = 0.0
                        var f1 = 0.0
                        if Ei > MASS_PI0:
                            var cutoff = sqrt(1.0 - MASS_PI0 / Ei)
                            f0 = F_over_cutoff * cutoff
                            self.KER_PP2Pi0.set(
                                i,
                                j,
                                F_over_cutoff
                                * cutoff
                                * Ei
                                / Ej
                                * C_LIGHT
                                * self.sigma_inel_pi0[j],
                            )
                        if Ei > MASS_PI:
                            var cutoff = sqrt(1.0 - MASS_PI / Ei)
                            f1 = F_over_cutoff * cutoff
                            self.KER_PP2Pipm.set(
                                i,
                                j,
                                F_over_cutoff
                                * cutoff
                                * Ei
                                / Ej
                                * C_LIGHT
                                * self.sigma_inel_pipm[j],
                            )
                        if kap and i >= 1:
                            self.kappa_prot_pi0[j] += (
                                dat.dlnE * f0 * Ei * Ei / Ej / Ekin
                            )
                            self.kappa_prot_pipm[j] += (
                                dat.dlnE * f1 * Ei * Ei / Ej / Ekin
                            )
        if self.verbose:
            print("computing proton proton delta-approx. kernels")
        # (kept on the grid when E_trans lies at or above its top)
        var idx_etrans = min(
            getIndexBelow(dat.egrid_had, nh, self.E_trans * (1.0 + 1e-9)) + 1,
            nh - 1,
        )
        self.K_pi = self.kappa_prot_pi0[idx_etrans]
        for j in range(nh):
            if dat.egrid_had[j] < e_at:
                self.kappa_prot_pipm[j] = self.K_pi
                self.kappa_prot_pi0[j] = self.K_pi
        for i in range(nh):
            self.egrid_shifted[i] = dat.egrid_had[i] / self.K_pi + MASS_P
            self.sigma_inel_pipm_shifted[i] = self.sigma_inel_Kafexhiu(
                self.egrid_shifted[i], MASS_PI
            )
            self.sigma_inel_pi0_shifted[i] = self.sigma_inel_Kafexhiu(
                self.egrid_shifted[i], MASS_PI0
            )
        if self.verbose:
            print("computing proton proton direct pp->pi0->gamma kernel")
        for i in range(dat.nbins_pho):
            var Eg = dat.egrid_had_photons[i]
            for j in range(nh):
                var Ep = dat.egrid_had[j]
                var ds: Float64
                match pp_model_neutral_pions:
                    case 0:
                        ds = self.dsigma_dEgamma_SIBYLL(Ep, Eg)
                    case 1:
                        ds = self.dsigma_dEgamma_QGSJET(Ep, Eg)
                    case 2:
                        ds = self.dsigma_dEgamma_Geant4(Ep, Eg)
                    case _:
                        ds = self.dsigma_dEgamma_Pythia(Ep, Eg)
                self.KER_FP2Pi02GG.set(i, j, dat.dlnE * Eg * ds * C_LIGHT)
        self.kappa_K = zeros(nh)
        if build_kaons:
            self._init_kaon_kernels(dat, pp_model_charged_pions)
            self.kaons_built = True
        self.sig_kappa_C = zeros(nh)
        if build_charm:
            self._init_charm_kernels(dat)
            self.charm_built = True

    def sigma_inel_Kafexhiu(self, Eproton: Float64, mpi: Float64) -> Float64:
        var T_th = 2.0 * mpi + mpi * mpi / (2.0 * MASS_P)
        var T_p = Eproton - MASS_P
        var L = c_log(T_p / T_th)
        var cutoff = c_pow(T_th / T_p, 1.9)
        if T_p > T_th:
            return (
                (30.7 - 0.96 * L + 0.18 * L * L)
                * c_pow(1.0 - cutoff, 3.0)
                * MILLIBARN
            )
        return 0.0

    def Calc_PP_ProtonLoss(mut self, rp: RunParams):
        if self.verbose:
            print("computing proton-proton proton-cooling term")
        var nh = len(self.kappa_prot_pipm)
        resize_zero(self.APP_PRO, nh)
        var f = -rp.pp_target_proton_density * C_LIGHT
        for i in range(nh):
            self.APP_PRO[i] = f * (
                2.0 * (self.kappa_prot_pipm[i] * self.sigma_inel_pipm[i])
                + self.kappa_prot_pi0[i] * self.sigma_inel_pi0[i]
            )
        if self.kaons_loss and len(self.kappa_K) == nh:
            # the energy the kaons carry away (am3_mojo addition)
            for i in range(nh):
                self.APP_PRO[i] += f * self.kappa_K[i] * self.sigma_inel_pipm[i]
        if self.charm_loss and len(self.sig_kappa_C) == nh:
            # the energy the charm hadrons carry away (am3_mojo addition)
            for i in range(nh):
                self.APP_PRO[i] += f * self.sig_kappa_C[i]

    def Calc_PP_PionSrc(
        mut self, rp: RunParams, dat: AM3Arrays, pi0_into_cascade: Bool
    ):
        if self.verbose:
            print("computing proton-proton pion-generation term")
        var nh = dat.nbins_had
        # (Within the grid: E_trans and K_pi come from the last
        #  initialize_kernels. If it ran on a larger grid, AM3 read and wrote
        #  past the end of the arrays. Below the first bin, at
        #  E_trans K_pi <= 1 GeV, that is E_trans below about 12 GeV, the delta
        #  approximation has no bins. AM3 started the kernel's pions at a
        #  negative bin, before the arrays, and at K_pi = 0 at Int(-inf).)
        var e_pi = self.E_trans * self.K_pi
        var i_trans = min(Int(c_log(e_pi) / dat.dlnE), nh) if e_pi > 1.0 else 0
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
                * C_LIGHT
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
                    * C_LIGHT
                    * dat.egrid_had[i]
                    / self.K_pi
                )
        if (
            self.KER_PP2Pipm.rows == 0
        ):  # (no kernels: pp after an init without it)
            return
        var Kpm = self.KER_PP2Pipm.ptr()
        var K0 = self.KER_PP2Pi0.ptr()
        var prot = pd(dat.n[P_protons])
        var nr = self.KER_PP2Pipm.rows
        var qp = pd(self.qpip)
        var qm = pd(self.qpim)
        var q0 = pd(self.qpi0)
        # The sums run over the protons j > i (the kernels are zero for
        # E_j <= E_i), between the lowest and the highest bin that hold any
        # protons. The terms of the other bins are exact zeros.
        var plo: Int
        var phi: Int
        plo, phi = nonzero_range(prot, nh)
        comptime W = WD
        var i = i_trans
        while i + W <= nh:
            var sp = qp.unsafe_offset(i).unsafe_load[width=W]()
            var sm = qm.unsafe_offset(i).unsafe_load[width=W]()
            var s0 = q0.unsafe_offset(i).unsafe_load[width=W]()
            for j in range(max(i + 1, plo), phi):
                var base = SIMD[DType.float64, W](
                    dat.dlnE * prot[unsafe_offset=j] * dens
                )
                var kpm = Kpm.unsafe_offset(j * nr + i).unsafe_load[width=W]()
                var t = base * kpm
                sp += t
                sm += t
                if pi0_into_cascade:
                    s0 = base.fma(
                        K0.unsafe_offset(j * nr + i).unsafe_load[width=W](), s0
                    )
            qp.unsafe_offset(i).unsafe_store(sp)
            qm.unsafe_offset(i).unsafe_store(sm)
            q0.unsafe_offset(i).unsafe_store(s0)
            i += W
        for i2 in range(i, nh):
            var sp = self.qpip[i2]
            var sm = self.qpim[i2]
            var s0 = self.qpi0[i2]
            for j in range(max(i2 + 1, plo), phi):
                var base = dat.dlnE * prot[unsafe_offset=j] * dens
                var kpm = Kpm[unsafe_offset=j * nr + i2]
                sp += base * kpm
                sm += base * kpm
                if pi0_into_cascade:
                    s0 += base * K0[unsafe_offset=j * nr + i2]
            self.qpip[i2] = sp
            self.qpim[i2] = sm
            self.qpi0[i2] = s0

    def Calc_PP_P2Pi02GG(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing proton-proton direct pi0->gamma+gamma term")
        copy_into(self.w_prot, dat.n[P_protons])
        for j in range(3):
            self.w_prot[j] = 0.0
        resize_zero(self.epsilon_gamma_p2pi02GG, dat.nbins_pho)
        if self.KER_FP2Pi02GG.rows == 0:  # (as in Calc_PP_PionSrc)
            return
        gemv_d(
            self.KER_FP2Pi02GG,
            pd(self.w_prot),
            rp.pp_target_proton_density,
            pd(self.epsilon_gamma_p2pi02GG),
        )

    # ---------------- Kafexhiu et al. 2014 pi0 -> gamma parametrization -------

    def dsigma_dEgamma_SIBYLL(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var Tp = Eproton - MASS_P
        return MILLIBARN * self.Amax_SIBYLL(Tp) * self.F_SIBYLL(Tp, Egamma)

    def dsigma_dEgamma_QGSJET(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var Tp = Eproton - MASS_P
        return MILLIBARN * self.Amax_QGSJET(Tp) * self.F_QGSJET(Tp, Egamma)

    def dsigma_dEgamma_Geant4(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var Tp = Eproton - MASS_P
        return MILLIBARN * self.Amax_Geant4(Tp) * self.F_Geant4(Tp, Egamma)

    def dsigma_dEgamma_Pythia(
        self, Eproton: Float64, Egamma: Float64
    ) -> Float64:
        var Tp = Eproton - MASS_P
        return MILLIBARN * self.Amax_Pythia(Tp) * self.F_Pythia(Tp, Egamma)

    def Amax_SIBYLL(self, Tp: Float64) -> Float64:
        var theta_p = Tp / MASS_P
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
                / MASS_P
            )
        return Amax

    def Amax_QGSJET(self, Tp: Float64) -> Float64:
        var theta_p = Tp / MASS_P
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
                / MASS_P
            )
        return Amax

    def Amax_Geant4(self, Tp: Float64) -> Float64:
        var theta_p = Tp / MASS_P
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
                / MASS_P
            )
        else:
            Amax = (
                9.13
                * c_pow(theta_p, -0.35)
                * c_exp(9.7e-3 * (Ltheta_p * Ltheta_p))
                * self.sigma_pi_Geant4(Tp)
                / MASS_P
            )
        return Amax

    def Amax_Pythia(self, Tp: Float64) -> Float64:
        var theta_p = Tp / MASS_P
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
                / MASS_P
            )
        return Amax

    def _XY(self, Tp: Float64, Egamma: Float64) -> Tuple[Float64, Float64]:
        """pow(m, 2.0) == m^2 for MASS_PI0."""
        var mpi2 = MASS_PI0 * MASS_PI0
        var Y = Egamma + mpi2 / (4.0 * Egamma)
        var Egm = self.Egamma_max(Tp)
        var Y0 = Egm + mpi2 / (4.0 * Egm)
        var X = (Y - MASS_PI0) / (Y0 - MASS_PI0)
        return (X, Y0)

    def F_SIBYLL(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var C = 3.55 * MASS_PI0 / Y0
        # From 100 GeV, as Amax_SIBYLL (Kafexhiu et al. 2014, Table V). AM3
        # took this shape from 50 GeV, with Geant4's peak value up to 100 GeV:
        # AM3 bug.
        if X >= 0.0 and X < 1.0 and Tp > 100.0:
            return c_pow(1.0 - sqrt(X), 3.6) / (1.0 + X / C)
        return self.F_Geant4(Tp, Egamma)

    def F_QGSJET(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var C = 3.55 * MASS_PI0 / Y0
        # (from 100 GeV, as F_SIBYLL)
        if X >= 0.0 and X < 1.0 and Tp > 100.0:
            return c_pow(1.0 - sqrt(X), 4.5) / (1.0 + X / C)
        return self.F_Geant4(Tp, Egamma)

    def F_Geant4(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var theta = Tp / MASS_P
        var kappa = 3.29 - 0.2 * c_pow(theta, -1.5)
        var q = (Tp - 1.0) / MASS_P
        var C = 3.0 * MASS_PI0 / Y0
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
                return c_pow(1.0 - sqrt(X), 4.2) / (1.0 + X / C)
            elif Tp > 100.0:
                return c_pow(1.0 - sqrt(X), 4.9) / (1.0 + X / C)
            return 0.0
        return 0.0

    def F_Pythia(self, Tp: Float64, Egamma: Float64) -> Float64:
        var X: Float64
        var Y0: Float64
        X, Y0 = self._XY(Tp, Egamma)
        var C = 3.5 * MASS_PI0 / Y0
        if X >= 0.0 and X < 1.0 and Tp > 50.0:
            return c_pow(1.0 - sqrt(X), 4.0) / (1.0 + X / C)
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
            var s = 2.0 * MASS_P * (Tp + 2.0 * MASS_P)
            var X = sqrt(s) - MASS_P
            var t = s - MASS_PI0 * MASS_PI0 - 4.0 * MASS_P * MASS_P
            var eta = sqrt(
                t * t - 16.0 * MASS_PI0 * MASS_PI0 * MASS_P * MASS_P
            ) / (2.0 * MASS_PI0 * sqrt(s))
            var g_RES = sqrt(
                M_RES * M_RES * (M_RES * M_RES + Gamma_RES * Gamma_RES)
            )
            var K_RES = (
                sqrt(8.0)
                * M_RES
                * Gamma_RES
                * g_RES
                / (PI * sqrt(M_RES * M_RES + g_RES))
            )
            var u = X * X - M_RES * M_RES
            var w = M_RES * Gamma_RES
            var f_BW = MASS_P * K_RES / (u * u + w * w)
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
            var xi_p = (Tp - 3.0) / MASS_P
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
        var xi_p = (Tp - 3.0) / MASS_P
        return (
            0.908
            * c_pow(xi_p, 0.176)
            * (1.0 + c_exp(-0.0009 * c_pow(xi_p, 0.448)))
            * (1.0 - c_exp(-6.089 * c_pow(xi_p, 0.25)))
        )

    def multip_pi0_Geant4(self, Tp: Float64) -> Float64:
        var m = 0.0
        if Tp > 2.0 and Tp < 5.0:
            var Qp = (Tp - self.Tp_th) / MASS_P
            m = -6.0e-3 + 0.237 * Qp - 0.023 * (Qp * Qp)
        if Tp >= 5.0:
            var xi_p = (Tp - 3.0) / MASS_P
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
        var xi_p = (Tp - 3.0) / MASS_P
        return (
            0.652
            * c_pow(xi_p, 0.1928)
            * (1.0 + c_exp(-0.0016 * c_pow(xi_p, 0.483)))
            * (1.0 - c_exp(-0.488 * c_pow(xi_p, 0.25)))
        )

    def Egamma_max(self, Tp: Float64) -> Float64:
        var gamma_pi_LAB = self.E_pi_max_LAB(Tp) / MASS_PI0
        var Beta_pi_LAB = sqrt(1.0 - c_pow(gamma_pi_LAB, -2.0))
        return (MASS_PI0 / 2.0) * gamma_pi_LAB * (1.0 + Beta_pi_LAB)

    def E_pi_max_LAB(self, Tp: Float64) -> Float64:
        var s = 2.0 * MASS_P * (Tp + 2.0 * MASS_P)
        var gamma_CM = (Tp + 2.0 * MASS_P) / sqrt(s)
        var E_pi_CM = (s - 4.0 * MASS_P * MASS_P + MASS_PI0 * MASS_PI0) / (
            2.0 * sqrt(s)
        )
        var P_pi_CM = sqrt(E_pi_CM * E_pi_CM - MASS_PI0 * MASS_PI0)
        var Beta_CM = sqrt(1.0 - c_pow(gamma_CM, -2.0))
        return gamma_CM * (E_pi_CM + P_pi_CM * Beta_CM)

    # kaons (am3_mojo addition)
    def _init_kaon_kernels(mut self, dat: AM3Arrays, model: Int):
        """Kaon production kernels (am3_mojo addition, see pp_kaon_F).

        The factor (E_i/E_j) comes from AM3's densities, which are per ln E.
        kappa_K, the sum of x c F_K dx over the same grid, is the fraction of
        E_p that the kaons take. So the protons lose the energy of the kaon
        source. Below E_t, the multiplicity fit gives the kaons per collision.
        They are shared between the two hadron bins around E_K, so that their
        number and energy are kept. Kaons below the lowest hadron bin (1 GeV),
        from protons below about 7 GeV, are neither made nor charged to the
        protons."""
        if self.verbose:
            print("computing proton proton kaon kernels")
        var nh = dat.nbins_had
        # Koers et al. from 100 GeV on, whatever the transition energy of the
        # pions (the multiplicity fit below covers only sqrt(s) <= 63 GeV)
        var Et = KPW_E_MIN
        var mp = MASS_P
        for s in range(N_KAON):
            self.KER_PP2K[s] = MatD(nh, nh)
            self.kd_l[s] = VecI(length=nh, fill=-1)
            self.kd_w[s] = zeros(nh)
            self.kd_n[s] = zeros(nh)
            var nt: Float64
            var et: Float64
            nt, et = pp_kaon_moments(s, Et, model)
            var c_t = pp_kaon_mult_fit(s, Et) / nt if nt > 0.0 else 1.0
            for j in range(nh):
                var Ej = dat.egrid_had[j]
                if Ej < Et:
                    continue
                var c = pp_kaon_fade(c_t, Ej)
                for i in range(j):
                    var Ei = dat.egrid_had[i]
                    var F = c * pp_kaon_F(s, Ei / Ej, Ej, model)
                    self.KER_PP2K[s].set(
                        i, j, F * Ei / Ej * C_LIGHT * self.sigma_inel_pipm[j]
                    )
                    # fraction of E_p in the kaons of the kernel (AM3 applies
                    # the pp loss as dE/dt = A E)
                    self.kappa_K[j] += dat.dlnE * F * Ei * Ei / Ej / Ej
            if nt <= 0.0:
                continue
            var K_s = et * Et / nt / (Et - mp)  # mean fraction of T_p
            for j in range(nh):
                var Ej = dat.egrid_had[j]
                if Ej >= Et:
                    continue
                var nj = pp_kaon_mult_fit(s, Ej)
                if nj <= 0.0:
                    continue
                var Tj = Ej - mp
                var g_cm = sqrt((Ej + mp) / (2.0 * mp))
                var mK = kpw_mass(_KPW_K if s <= KM else _KPW_K0)
                var EK = max(K_s * Tj, g_cm * mK)
                var l: Int
                var w: Float64
                l, w = _split_bins(EK, dat)
                if l < 0:
                    continue  # below the hadron grid: neither made nor lost
                self.kappa_K[j] += nj * EK / Ej
                self.kd_l[s][j] = l
                self.kd_w[s][j] = w
                self.kd_n[s][j] = nj * C_LIGHT * self.sigma_inel_pipm[j]

    def Calc_PP_KaonSrc(mut self, rp: RunParams, dat: AM3Arrays):
        """Kaon sources qK[s] (per ln E) from pp collisions."""
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
            _add_split(
                pd(self.qK[s]),
                self.kd_l[s],
                self.kd_w[s],
                self.kd_n[s],
                dens,
                pd(dat.n[P_protons]),
                nh,
            )

    # charm hadrons (am3_mojo addition)
    def _init_charm_kernels(mut self, dat: AM3Arrays):
        """Charm production kernels (am3_mojo addition). K_h comes from the mean
        energy of the closed form at CHARM_E_FIT."""
        if self.verbose:
            print("computing proton proton charm kernels")
        var nh = dat.nbins_had
        var d = dat.dlnE / Float64(CPP_SUB)
        # (+ 1e-9: 4.6 / 0.025 is 183.99999999999997. Without the 1e-9, the
        #  fragmentation ends one point above z = 0.01.)
        var nz = Int(CPP_LN_ZMIN / d + 1e-9) + 1
        var zD = List[List[Float64]]()
        for h in range(4):
            zD.append(pp_charm_fragmentation(h, d, nz))
            self.KER_PP2C[h] = MatD(nh, nh)
            self.cd_l[h] = VecI(length=nh, fill=-1)
            self.cd_w[h] = zeros(nh)
            self.cd_n[h] = zeros(nh)
        var per_inel = pp_charm_sigma_cc(
            CHARM_E_TOP
        ) / self.sigma_inel_Kafexhiu(CHARM_E_TOP, MASS_PI)
        for j in range(nh):
            var Ej = dat.egrid_had[j]
            if Ej < CHARM_E_FIT:
                continue
            var sig = (
                pp_charm_sigma_cc(Ej) if Ej
                <= CHARM_E_TOP else per_inel * self.sigma_inel_pipm[j]
            )
            var g = pp_charm_shape(Ej, d)
            var ng = len(g)
            for k in range(1, j + 1):
                var i = j - k
                var c = k * CPP_SUB
                var lo = max(0, c - (ng - 1))
                var hi = min(nz - 1, c)
                if lo > hi:
                    continue
                var Ei = dat.egrid_had[i]
                for h in range(4):
                    if Ei < charm_mass(2 * h):
                        continue
                    var s = 0.0
                    for l in range(lo, hi + 1):
                        s += zD[h][l] * g[c - l]
                    var xs = sig * d * s  # x dsigma_h/dx
                    self.KER_PP2C[h].set(i, j, xs * C_LIGHT)
                    # a hadron and its antiparticle
                    self.sig_kappa_C[j] += 2.0 * dat.dlnE * xs * Ei / Ej
        var mp = MASS_P
        var gT = pp_charm_shape(CHARM_E_FIT, d)
        var xq = 0.0
        for m in range(len(gT)):
            xq += d * c_exp(-Float64(m) * d) * gT[m]
        for h in range(4):
            var zm = 0.0
            for l in range(nz):
                zm += d * c_exp(-Float64(l) * d) * zD[h][l]
            zm /= pp_charm_fraction(h)
            var K_h = xq * zm * CHARM_E_FIT / (CHARM_E_FIT - mp)
            for j in range(nh):
                var Ej = dat.egrid_had[j]
                if Ej >= CHARM_E_FIT:
                    continue
                var sig = pp_charm_sigma_cc(Ej)
                if sig <= 0.0:
                    continue
                var g_cm = sqrt((Ej + mp) / (2.0 * mp))
                var Eh = max(K_h * (Ej - mp), g_cm * charm_mass(2 * h))
                var l: Int
                var w: Float64
                l, w = _split_bins(Eh, dat)
                if l < 0:
                    continue
                var nhad = pp_charm_fraction(h) * sig
                self.sig_kappa_C[j] += 2.0 * nhad * Eh / Ej
                self.cd_l[h][j] = l
                self.cd_w[h][j] = w
                self.cd_n[h][j] = nhad * C_LIGHT

    def Calc_PP_CharmSrc(mut self, rp: RunParams, dat: AM3Arrays):
        """Charm sources qC[s] (per ln E) from pp collisions. A hadron and its
        antiparticle have the source of their pair, computed once (ppc_work)."""
        var nh = dat.nbins_had
        for s in range(N_CHARM):
            resize_zero(self.qC[s], nh)
        if not self.charm_src or self.KER_PP2C[0].rows != nh:
            return
        copy_into(self.prot_view, dat.n[P_protons])
        var job = PPCharmJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            rp.pp_target_proton_density,
            dat.dlnE,
        )
        parallel_run[ppc_work](job, min(self.nthreads, 4))
