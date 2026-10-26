"""Electron-positron annihilation, e+ e- -> gamma gamma (am3_mojo addition;
AM3 has no pair annihilation).

Exact treatment of Svensson (1982, ApJ 258, 321) for isotropic positrons and
electrons of Lorentz factors gamma_+ and gamma_-, in his notation (the
derivations: tests/literature_checks/annihilation/README.md):
  * the angle-averaged rate per pair vsigma(gamma_+, gamma_-) (eqs. 13 and
    18), written in gamma_r, the positron's Lorentz factor in the rest frame
    of the electron (eq. 1a: gamma_r = 2 gamma_cm^2 - 1);
  * the emissivity v dsigma/dk (k, gamma_+, gamma_-) of a pair, counting both
    photons (eqs. 55-58; eq. 41 if one particle is at rest).
Vurm & Poutanen (2009), whose treatment AM3's gamma gamma pair production
builds on, use the same rate and an equivalent form of the spectrum. Rates
are in sigma_T c (= 1 in the code units of the leptons and photons), energies
in m c^2. Valid for all Lorentz factors while the relative velocity is
>> alpha c (no Coulomb focusing or positronium formation).

On the grids (lepton grid points gamma_i and photon grid points k_l, Delta
apart in ln gamma and ln k):
  * rate table R_ij = vsigma(gamma_i, gamma_j) / (sigma_T c) for every pair of
    lepton grid points (grid point 0 is gamma = 1, at rest);
  * photon kernel K_ij(l) (photons of the pair (i, j) in photon bin l), exact
    for min(gamma_i, gamma_j) <= GAMMA_EXACT: Gauss-Legendre quadrature
    keeping photon number and energy segment by segment, or with one particle
    at rest, eq. 41 integrated over each bin in closed form;
  * pairs with both Lorentz factors above GAMMA_EXACT (rate <~1e-3 of that of
    cold pairs) use Svensson's eq. 59: two photons at
    k_+- = [gamma_+-(1 + beta_+-) + gamma_-+(1 - beta_-+)] / 2, each shared
    between the two neighbouring photon grid points (number and energy kept).
Per time step, from the densities per ln gamma N_+-: the annihilation rates
per particle Gamma_-+(gamma_j) = Delta sum_i R_ij N_+-(gamma_i) and the photon
source Q_gamma(k_l) = Delta sum_ij N_+(gamma_i) N_-(gamma_j) K_ij(l). The pair
sums run in fixed chunks, added in a fixed order, so results do not depend on
the thread count.

Limitations:
  * AM3's solver keeps lepton bins 0 and 1 empty (gamma < 1.22): pairs that
    cool to rest leave the system there, as in AM3, and the 511 keV line is
    not produced;
  * the sinks and photon source of a step come from the densities at its
    start, so a step makes about Gamma dt / 2 more photons than two per pair
    lost. Keep the time step below about 0.05 of the shortest annihilation
    time 1 / Gamma;
  * where AM3's solver uses its regime 4 (cooling only) and annihilation is
    below 1 % of the cooling, the solver neglects the sink, while the photon
    source still counts those leptons;
  * Svensson's formulas are the Born approximation (no Coulomb corrections).
    Three-photon annihilation, 4 (pi^2 - 9) alpha / (3 pi) ~ 1/371 of the
    two-photon rate for slow pairs (Berestetskii, Lifshitz & Pitaevskii,
    sec. 89), is neglected.
"""

from std.math import floor, sqrt
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import MatD, gemv_d
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.specfun import gauss_legendre

comptime GAMMA_EXACT: Float64 = 100.0
comptime NQ = 16  # quadrature points per segment next to a limit or peak
comptime NQ_SMOOTH = 4  # quadrature points per segment away from them
comptime N_CHUNKS = 32  # fixed partition of the per-step pair sum


# ----------------------------------------------------------------------------
# Svensson 1982: gamma_+-, beta_+- of the positron and the electron, gamma_cm,
# beta_cm in their cm frame, gamma_r, beta_r in the electron's rest frame
# ----------------------------------------------------------------------------
@inline(.always)
def beta_gamma(g: Float64) -> Float64:
    """beta gamma = (gamma^2 - 1)^(1/2) at gamma = g, without cancellation."""
    return sqrt(max((g - 1.0) * (g + 1.0), 0.0))


@inline(.always)
def L_cm(gr: Float64) -> Float64:
    """L(beta_cm) = ln[(1 + beta_cm) / (1 - beta_cm)] of eq. 18 at the gamma_cm
    of gamma_r = gr: ln[gamma_r (1 + beta_r)]."""
    return c_log(gr + beta_gamma(gr))


def gcm_limits(gp: Float64, gm: Float64) -> Tuple[Float64, Float64]:
    """(gamma_cm^min, gamma_cm^max) of eq. 17: [(1 + gamma_r) / 2]^(1/2)
    (eq. 1a) at gamma_r^min and gamma_r^max (see ann_rate)."""
    var grmax = gp * gm + beta_gamma(gp) * beta_gamma(gm)
    var grmin = (gp * gp + gm * gm - 1.0) / grmax
    return (sqrt(0.5 * (1.0 + grmin)), sqrt(0.5 * (1.0 + grmax)))


def _eq18_bracket(gr: Float64) -> Float64:
    """The bracket of eq. 18, beta_cm^3 gamma_cm^2 L(beta_cm) - 2 gamma_cm^2
    + (3/4) L^2(beta_cm), at the gamma_cm of gamma_r = gr: gamma_cm^2 =
    (gr + 1) / 2 and beta_cm = [(gr - 1) / (gr + 1)]^(1/2)."""
    var gcm2 = 0.5 * (gr + 1.0)
    var bcm = sqrt(max(gr - 1.0, 0.0) / (gr + 1.0))
    var L = L_cm(gr)
    return bcm * bcm * bcm * gcm2 * L - 2.0 * gcm2 + 0.75 * L * L


def _eq13_integrand(gr: Float64) -> Float64:
    """2 beta_cm gamma_cm^2 sigma(gamma_cm) / sigma_T, the integrand of eq. 13
    without c, at the gamma_cm of gamma_r = gr (eq. 6 with its logarithm
    corrected):
      pi r_e^2 / (1 + gr) [(gr^2 + 4 gr + 1) L(beta_cm) / (beta_r gr)
                           - (gr + 3)],
    with L(beta_cm) / (beta_r gr) = 1 - (gr - 1) / 3 within 1e-12 of gr = 1.
    """
    var gr_m1 = max(gr - 1.0, 0.0)
    var bg_r = sqrt(gr_m1 * (gr + 1.0))  # beta_r gamma_r
    var L_over_bg = (
        c_log1p(gr_m1 + bg_r) / bg_r if gr_m1 > 1e-12 else 1.0 - gr_m1 / 3.0
    )
    return (
        PI_RE2
        / (1.0 + gr)
        * ((gr * gr + 4.0 * gr + 1.0) * L_over_bg - (gr + 3.0))
    )


def ann_rate(gp: Float64, gm: Float64, gl_mu: Vec, gl_w: Vec) -> Float64:
    """vsigma(gamma_+, gamma_-) / (sigma_T c), written in gamma_r. Its range
    runs between eq. 1a at mu = -1 and mu = 1,
      gamma_r^max = gamma_+ gamma_- (1 + beta_+ beta_-),
      gamma_r^min = (gamma_+^2 + gamma_-^2 - 1) / gamma_r^max
    (without the cancellation in 1 - beta_+ beta_-). Eq. 18 is then
      pi r_e^2 [bracket(gamma_r^max) - bracket(gamma_r^min)]
      / (beta_+ gamma_+^2 beta_- gamma_-^2).
    Where beta_+ gamma_+ beta_- gamma_- < 1e-2 gamma_r^max, the two brackets
    nearly cancel: the rate is then eq. 13, with the 8-node Gauss-Legendre
    rule in mu (gl_mu, gl_w) at the nodes
      gamma_r = [gamma_r^max (1 - mu) + gamma_r^min (1 + mu)] / 2."""
    var bgp = beta_gamma(gp)
    var bgm = beta_gamma(gm)
    var grmax = gp * gm + bgp * bgm
    var grmin = max(1.0, (gp * gp + gm * gm - 1.0) / grmax)
    if bgp * bgm > 1e-2 * grmax:
        return (
            PI_RE2
            * (_eq18_bracket(grmax) - _eq18_bracket(grmin))
            / (bgp * gp * bgm * gm)
        )
    var total = 0.0
    for q in range(len(gl_mu)):
        var mu = gl_mu[q]
        var gr = 0.5 * (grmax * (1.0 - mu) + grmin * (1.0 + mu))
        total += gl_w[q] * _eq13_integrand(gr)
    return 0.5 * total / (gp * gm)


def _H(
    k: Float64,
    gp: Float64,
    gm: Float64,
    gcm: Float64,
    sign: Float64,
    gstar2: Float64,
) -> Float64:
    """H_+ (sign = 1) or H_- (sign = -1) of eqs. 56-58 at gamma_cm = gcm, with
    c, d, u of eqs. 52-54 and gstar2 = gamma_cm*^2 of eq. 31."""
    var g_mp = gm if sign > 0.0 else gp  # gamma_-+
    var c = (g_mp - k) * (g_mp - k) - 1.0
    var d = g_mp * (gp + gm) + sign * k * (gp - gm)
    var gstar = sqrt(gstar2)
    if abs(c) < 1e-6:
        # eq. 58 (c = 0), accurate to <~ 0.3 |c|; eq. 56 loses digits for
        # |c| << 1e-6
        return (
            (2.0 / 3.0) * gcm * gcm * gcm + 2.0 * gcm + 1.0 / gcm
        ) / gstar + 0.5 * ((2.0 / 3.0) * gcm * gcm * gcm - d * gcm) / (
            gstar * gstar * gstar
        )
    # c >= -1 and gcm <= gstar on the range of the integral: the clamps
    # catch rounding errors
    var u = sqrt(max(c * gcm * gcm + gstar2, 0.0))
    var sqrt_c = sqrt(abs(c))  # (c)^(1/2) or (-c)^(1/2)
    var I: Float64
    if c > 0.0:
        I = c_log(gcm * sqrt_c + u) / sqrt_c
    else:
        I = c_asin(min(gcm * sqrt_c / gstar, 1.0)) / sqrt_c
    # the second term, which tends to zero where u vanishes
    # (gamma_cm = gamma_cm* and k = gamma_-+)
    var second = 0.0
    if u > 0.0:
        second = (
            1.0 / gcm - gcm / c + gcm / (2.0 * gstar2) * (2.0 * c - d)
        ) / u
    return (2.0 + (1.0 - gstar2) / c) * I + second + gcm / c * u


@inline(.always)
def _emissivity(
    k: Float64,
    gp: Float64,
    gm: Float64,
    gcm_min: Float64,
    gcm_max: Float64,
    pref: Float64,
) -> Float64:
    """v dsigma/dk (k, gamma_+, gamma_-) / (sigma_T c), eq. 55 between the
    limits of eq. 32 (both photons: integral dk = 2 vsigma, integral k dk =
    (gamma_+ + gamma_-) vsigma), with gcm_min, gcm_max of eq. 17 and
    pref = pi r_e^2 / (beta_+ gamma_+^2 beta_- gamma_-^2)."""
    var gstar2 = k * (gp + gm - k)  # eq. 31
    if gstar2 <= 0.0:
        return 0.0
    var gcm_U = min(gcm_max, sqrt(gstar2))
    if gcm_U <= gcm_min:
        return 0.0
    return pref * (
        _eq55_brace(k, gp, gm, gstar2, gcm_U)
        - _eq55_brace(k, gp, gm, gstar2, gcm_min)
    )


@inline(.always)
def _eq55_brace(
    k: Float64, gp: Float64, gm: Float64, gstar2: Float64, gcm: Float64
) -> Float64:
    """The brace of eq. 55 at gamma_cm = gcm,
    [(gamma_+ + gamma_-)^2 - 4 gamma_cm^2]^(1/2) + H_+ + H_-."""
    return (
        sqrt(max((gp + gm) * (gp + gm) - 4.0 * gcm * gcm, 0.0))
        + _H(k, gp, gm, gcm, 1.0, gstar2)
        + _H(k, gp, gm, gcm, -1.0, gstar2)
    )


def ann_at_rest_segment(
    k_lo: Float64, k_hi: Float64, gr: Float64, d_lo: Float64, d_hi: Float64
) -> Tuple[Float64, Float64]:
    """Photon number and energy of eq. 41 (one particle at rest, the other of
    gamma_r = gr) between k = k_lo and k_hi: the differences of the primitives
      P(k) = pref [1/k + (3 + gr - 2/(1 + gr)) ln(k/(1 + gr - k))
                   - 1/(1 + gr - k) - 2k],
      E(k) = pref [-(3 + gr)(1 + gr) ln(1 + gr - k) - ln(k/(1 + gr - k))
                   - (1 + gr)/(1 + gr - k) - k^2],
    pref = pi r_e^2 / (beta_r gamma_r^2), taken term by term. d_lo, d_hi =
    1 + gr - k_lo, 1 + gr - k_hi."""
    var pref = PI_RE2 / (beta_gamma(gr) * gr)
    var gr1 = 1.0 + gr
    var gr3 = 3.0 + gr
    # ln[(1 + gr - k_hi) / (1 + gr - k_lo)]: for a segment short against d_lo,
    # from log1p (at large gr, d_lo and d_hi round alike, so their ratio loses
    # all digits). Near the top, from the exact distances (k_hi = k_U is
    # rounded, 1 + gr - k_U = k_L is not)
    var short = k_hi - k_lo < 0.5 * d_lo
    var ln_d = c_log1p(-(k_hi - k_lo) / d_lo) if short else c_log(d_hi / d_lo)
    var d_ln = c_log(k_hi / k_lo) - ln_d  # the difference of ln(k/(1 + gr - k))
    var d_inv = (k_hi - k_lo) / (d_lo * d_hi)  # that of 1/(1 + gr - k)
    var dP = pref * (
        (k_lo - k_hi) / (k_lo * k_hi)
        + (gr3 - 2.0 / gr1) * d_ln
        - d_inv
        - 2.0 * (k_hi - k_lo)
    )
    var dE = pref * (
        -gr3 * gr1 * ln_d - d_ln - gr1 * d_inv - (k_hi - k_lo) * (k_hi + k_lo)
    )
    return (dP, dE)


def ann_k_limits(
    gp: Float64, gm: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Photon energies (k_L, k_-, k_+, k_U) of eqs. 33: the kinematic limits
    and the two peaks of the spectrum, with gamma (1 - beta) =
    1 / [gamma (1 + beta)] (no cancellation in 1 - beta)."""
    var up_p = gp + beta_gamma(gp)  # gamma_+ (1 + beta_+)
    var up_m = gm + beta_gamma(gm)
    var down_p = 1.0 / up_p  # gamma_+ (1 - beta_+)
    var down_m = 1.0 / up_m
    return (
        0.5 * (down_p + down_m),
        0.5 * (down_p + up_m),
        0.5 * (up_p + down_m),
        0.5 * (up_p + up_m),
    )


# ----------------------------------------------------------------------------
# kernels on the grids
# ----------------------------------------------------------------------------
struct AnnKernelJob(Movable):
    """Builds the exact photon kernel of pairs t, t + nt, ..."""

    var A: Pointer[PairAnnihilation, MutUntrackedOrigin]
    var gam: FPtr
    var kl: FPtr  # photon bin centres
    var ln_k0: Float64
    var dlnE: Float64
    var ng: Int

    def __init__(
        out self,
        A: Pointer[PairAnnihilation, MutUntrackedOrigin],
        gam: FPtr,
        kl: FPtr,
        ln_k0: Float64,
        dlnE: Float64,
        ng: Int,
    ):
        self.A = A
        self.gam = gam
        self.kl = kl
        self.ln_k0 = ln_k0
        self.dlnE = dlnE
        self.ng = ng


def ann_kernel_work(
    job: Pointer[AnnKernelJob, MutUntrackedOrigin], t: Int, nt: Int
):
    ref J = job[]
    ref A = J.A[]
    var buf = zeros(J.ng)
    var nodes = Vec()
    var p = t
    while p < len(A.pa):
        A._fill_pair(p, J.gam, J.kl, J.ln_k0, J.dlnE, J.ng, buf, nodes)
        p += nt


struct AnnStepJob(Movable):
    """Photon source of the exact pairs, in N_CHUNKS fixed chunks."""

    var A: Pointer[PairAnnihilation, MutUntrackedOrigin]
    var wp: FPtr  # N_+ Delta, the positrons of each grid point
    var wm: FPtr  # N_- Delta
    var part: FPtr  # N_CHUNKS x ng partial sums
    var ng: Int

    def __init__(
        out self,
        A: Pointer[PairAnnihilation, MutUntrackedOrigin],
        wp: FPtr,
        wm: FPtr,
        part: FPtr,
        ng: Int,
    ):
        self.A = A
        self.wp = wp
        self.wm = wm
        self.part = part
        self.ng = ng


def ann_step_work(
    job: Pointer[AnnStepJob, MutUntrackedOrigin], t: Int, nt: Int
):
    ref J = job[]
    ref A = J.A[]
    var c = t
    while c < N_CHUNKS:
        var out = J.part.unsafe_offset(c * J.ng)
        for l in range(J.ng):
            out[unsafe_offset=l] = 0.0
        var p0 = A.chunk0[c]
        var p1 = A.chunk0[c + 1]
        var vals = pf(A.kval)
        comptime W = 8
        for p in range(p0, p1):
            var i = A.pa[p]
            var j = A.pb[p]
            var dens = J.wp[unsafe_offset=i] * J.wm[unsafe_offset=j]
            if i != j:
                dens += J.wp[unsafe_offset=j] * J.wm[unsafe_offset=i]
            if dens == 0.0:
                continue
            var n = A.pn[p]
            var o = out.unsafe_offset(A.pl0[p])
            var v = vals.unsafe_offset(A.poff[p])
            var m = 0
            while m + W <= n:
                var x = (
                    v.unsafe_offset(m)
                    .unsafe_load[width=W]()
                    .cast[DType.float64]()
                )
                var op = o.unsafe_offset(m)
                op.unsafe_store(op.unsafe_load[width=W]() + dens * x)
                m += W
            while m < n:
                o[unsafe_offset=m] += dens * Float64(v[unsafe_offset=m])
                m += 1
        # peak approximation for pairs with both Lorentz factors high
        var nd = len(A.da)
        var d0 = c * nd // N_CHUNKS
        var d1 = (c + 1) * nd // N_CHUNKS
        for d in range(d0, d1):
            var i = A.da[d]
            var j = A.db[d]
            var dens = J.wp[unsafe_offset=i] * J.wm[unsafe_offset=j]
            if i != j:
                dens += J.wp[unsafe_offset=j] * J.wm[unsafe_offset=i]
            if dens == 0.0:
                continue
            var n_ph = dens * A.dr[d]  # the photons at each peak
            var lm = A.dl_m[d]
            if lm >= 0:
                out[unsafe_offset=lm] += n_ph * (1.0 - A.dw_m[d])
                out[unsafe_offset=lm + 1] += n_ph * A.dw_m[d]
            var lp = A.dl_p[d]
            if lp >= 0:
                out[unsafe_offset=lp] += n_ph * (1.0 - A.dw_p[d])
                out[unsafe_offset=lp + 1] += n_ph * A.dw_p[d]
        c += nt


struct PairAnnihilation(Movable):
    var verbose: Bool
    var nthreads: Int
    var R: MatD  # rate table R_ij (nbins_lep x nbins_lep, symmetric)
    # exact photon kernel: pair p = (pa[p] <= pb[p]) has photon bins
    # pl0[p] .. pl0[p] + pn[p] - 1 with values kval[poff[p] ..]
    var pa: VecI
    var pb: VecI
    var pl0: VecI
    var pn: VecI
    var poff: VecI
    var kval: VecF
    var chunk0: VecI  # first exact pair of each chunk (equal shares of entries)
    var i_exact: Int  # pairs with both bins > i_exact: peak approximation
    # peak approximation: pair d = (da[d] <= db[d]) puts R_ij photons at k_-
    # and at k_+, shared between bins dl_m[d], dl_m[d] + 1 (the share dw_m[d]
    # of the upper one) and dl_p[d], dl_p[d] + 1 (dl < 0: off the grid)
    var da: VecI
    var db: VecI
    var dr: Vec
    var dl_m: VecI
    var dw_m: Vec
    var dl_p: VecI
    var dw_p: Vec
    var gl_mu: Vec  # the 8-node Gauss-Legendre rule in mu (eq. 13)
    var gl_w: Vec
    var cl: Vec  # quadrature nodes t on [0, 1], clustered at the ends
    var cw: Vec
    var sl: Vec  # NQ_SMOOTH Gauss-Legendre nodes on [0, 1]
    var sw: Vec
    # per-step results
    var Gamma_ele: Vec  # annihilation rate per electron, Gamma_- (code units)
    var Gamma_pos: Vec  # per positron, Gamma_+
    var Q_gamma: Vec  # photon source per ln k
    var wp: Vec  # N_+ Delta
    var wm: Vec  # N_- Delta
    var part: Vec
    var valid: Bool

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.nthreads = 1
        self.R = MatD()
        self.pa = VecI()
        self.pb = VecI()
        self.pl0 = VecI()
        self.pn = VecI()
        self.poff = VecI()
        self.kval = VecF()
        self.chunk0 = VecI()
        self.i_exact = 0
        self.da = VecI()
        self.db = VecI()
        self.dr = Vec()
        self.dl_m = VecI()
        self.dw_m = Vec()
        self.dl_p = VecI()
        self.dw_p = Vec()
        self.gl_mu = Vec()
        self.gl_w = Vec()
        gauss_legendre(8, self.gl_mu, self.gl_w)
        var xq = Vec()
        var wq = Vec()
        gauss_legendre(NQ, xq, wq)
        self.cl = zeros(NQ)
        self.cw = zeros(NQ)
        for q in range(NQ):
            # s in [0, 1] mapped to t = (1 - cos(pi s)) / 2: clusters the nodes
            # at the segment ends (the spectrum has kinks at its limits, where
            # it vanishes linearly, and at its peaks)
            var s = 0.5 * (xq[q] + 1.0)
            self.cl[q] = 0.5 * (1.0 - c_cos(PI * s))
            self.cw[q] = 0.25 * PI * c_sin(PI * s) * wq[q]
        gauss_legendre(NQ_SMOOTH, xq, wq)
        self.sl = zeros(NQ_SMOOTH)
        self.sw = zeros(NQ_SMOOTH)
        for q in range(NQ_SMOOTH):
            self.sl[q] = 0.5 * (xq[q] + 1.0)
            self.sw[q] = 0.5 * wq[q]
        self.Gamma_ele = Vec()
        self.Gamma_pos = Vec()
        self.Q_gamma = Vec()
        self.wp = Vec()
        self.wm = Vec()
        self.part = Vec()
        self.valid = False
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.Gamma_ele, dat.nbins_lep)
        resize_zero(self.Gamma_pos, dat.nbins_lep)
        resize_zero(self.Q_gamma, dat.nbins_pho)

    def free_kernels(mut self):
        """Release the kernels (annihilation off at init_kernels)."""
        self.R = MatD()
        self.pa = VecI()
        self.pb = VecI()
        self.pl0 = VecI()
        self.pn = VecI()
        self.poff = VecI()
        self.kval = VecF()
        self.chunk0 = VecI()
        self.da = VecI()
        self.db = VecI()
        self.dr = Vec()
        self.dl_m = VecI()
        self.dw_m = Vec()
        self.dl_p = VecI()
        self.dw_p = Vec()
        self.part = Vec()
        self.valid = False

    def ready(self, dat: AM3Arrays) -> Bool:
        """Kernels built for the current grid."""
        return (
            self.valid
            and self.R.rows == dat.nbins_lep
            and len(self.part) == N_CHUNKS * dat.nbins_pho
        )

    def _gamma(self, dat: AM3Arrays, i: Int) -> Float64:
        return 1.0 if i == 0 else dat.egrid_lep[i]

    def initialize_kernels(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing e+e- annihilation kernels")
        var nl = dat.nbins_lep
        var ng = dat.nbins_pho
        var dlnE = dat.dlnE
        var gam = zeros(nl)
        for i in range(nl):
            gam[i] = self._gamma(dat, i)
        # rate table
        self.R = MatD(nl, nl)
        for i in range(nl):
            for j in range(i, nl):
                var r = ann_rate(gam[i], gam[j], self.gl_mu, self.gl_w)
                self.R.set(i, j, r)
                self.R.set(j, i, r)
        # exact pairs and their photon-bin ranges
        self.i_exact = 0
        while self.i_exact + 1 < nl and gam[self.i_exact + 1] <= GAMMA_EXACT:
            self.i_exact += 1
        var kl = zeros(ng)
        for l in range(ng):
            kl[l] = dat.egrid_pho[l]
        var ln_k0 = c_log(kl[0])
        self.pa = VecI()
        self.pb = VecI()
        self.pl0 = VecI()
        self.pn = VecI()
        self.poff = VecI()
        var total = 0
        for i in range(self.i_exact + 1):
            for j in range(i, nl):
                var lo: Int
                var hi: Int
                lo, hi = self._bin_range(gam[i], gam[j], ln_k0, dlnE, ng)
                self.pa.append(i)
                self.pb.append(j)
                self.pl0.append(lo)
                self.pn.append(max(hi - lo, 0))
                self.poff.append(total)
                total += max(hi - lo, 0)
        self.kval = zerosf(total)
        # chunks of the per-step sum with about equal numbers of entries
        self.chunk0 = zerosi(N_CHUNKS + 1)
        var p = 0
        for c in range(1, N_CHUNKS):
            while p < len(self.pa) and self.poff[p] < c * total // N_CHUNKS:
                p += 1
            self.chunk0[c] = p
        self.chunk0[N_CHUNKS] = len(self.pa)
        var job = AnnKernelJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            pd(gam),
            pd(kl),
            ln_k0,
            dlnE,
            ng,
        )
        parallel_run[ann_kernel_work](job, self.nthreads)
        # peak approximation for the other pairs
        self.da = VecI()
        self.db = VecI()
        self.dr = Vec()
        self.dl_m = VecI()
        self.dw_m = Vec()
        self.dl_p = VecI()
        self.dw_p = Vec()
        for i in range(self.i_exact + 1, nl):
            for j in range(i, nl):
                var km: Float64
                var kp: Float64
                _, km, kp, _ = ann_k_limits(gam[i], gam[j])
                self.da.append(i)
                self.db.append(j)
                self.dr.append(self.R.get(i, j))
                var l: Int
                var w: Float64
                l, w = self._split(km, kl, ln_k0, dlnE, ng)
                self.dl_m.append(l)
                self.dw_m.append(w)
                l, w = self._split(kp, kl, ln_k0, dlnE, ng)
                self.dl_p.append(l)
                self.dw_p.append(w)
        resize_zero(self.part, N_CHUNKS * ng)
        self.valid = True

    @staticmethod
    def _bin_range(
        gp: Float64, gm: Float64, ln_k0: Float64, dlnE: Float64, ng: Int
    ) -> Tuple[Int, Int]:
        """Photon bins [lo, hi) the spectrum of the pair reaches."""
        var kL: Float64
        var kU: Float64
        kL, _, _, kU = ann_k_limits(gp, gm)
        if gp == 1.0 and gm == 1.0:
            kL = 1.0
            kU = 1.0
        # bins whose centres bracket the photon energies [kL, kU]
        var lo = Int(floor((c_log(kL) - ln_k0) / dlnE))
        var hi = Int(floor((c_log(kU) - ln_k0) / dlnE)) + 2
        return (max(lo, 0), min(hi, ng))

    @staticmethod
    def _split(
        k: Float64, kl: Vec, ln_k0: Float64, dlnE: Float64, ng: Int
    ) -> Tuple[Int, Float64]:
        """Bin l with kl[l] <= k < kl[l + 1] and the weight w of bin l + 1
        that keeps number and energy: (1 - w) kl[l] + w kl[l + 1] = k."""
        var l = Int(floor((c_log(k) - ln_k0) / dlnE))
        if l < 0 or l + 1 >= ng:
            return (-1, 0.0)
        return (l, (k - kl[l]) / (kl[l + 1] - kl[l]))

    def _deposit(
        self,
        n_ph: Float64,
        k_bar: Float64,
        kl: FPtr,
        ln_k0: Float64,
        dlnE: Float64,
        ng: Int,
        l0: Int,
        n: Int,
        mut buf: Vec,
    ):
        """Add n_ph photons of mean energy k_bar to the grid points
        k_l <= k_bar < k_l+1, the fractions 1 - w and w of them with
        (1 - w) k_l + w k_l+1 = k_bar: number and energy kept."""
        var l = Int(floor((c_log(k_bar) - ln_k0) / dlnE))
        if l < 0 or l + 1 >= ng:
            return
        var w = (k_bar - kl[unsafe_offset=l]) / (
            kl[unsafe_offset=l + 1] - kl[unsafe_offset=l]
        )
        if l >= l0 and l < l0 + n:
            buf[l - l0] += (1.0 - w) * n_ph
        if l + 1 >= l0 and l + 1 < l0 + n:
            buf[l + 1 - l0] += w * n_ph

    def _fill_pair(
        self,
        p: Int,
        gam: FPtr,
        kl: FPtr,
        ln_k0: Float64,
        dlnE: Float64,
        ng: Int,
        mut buf: Vec,
        mut nodes: Vec,
    ):
        """Exact photon kernel of pair p: photons per bin, number and energy
        kept segment by segment."""
        var n = self.pn[p]
        if n == 0:
            return
        var i = self.pa[p]
        var j = self.pb[p]
        var gp = gam[unsafe_offset=i]
        var gm = gam[unsafe_offset=j]
        var r = self.R.get(i, j)
        var l0 = self.pl0[p]
        var out = pf(self.kval).unsafe_offset(self.poff[p])
        for l in range(n):
            buf[l] = 0.0
        if i == 0 and j == 0:
            # both at rest: the 511 keV line, 2 photons at k = 1
            self._deposit(2.0 * r, 1.0, kl, ln_k0, dlnE, ng, l0, n, buf)
            PairAnnihilation._store(out, buf, n, r, l0, ng)
            return
        var kL: Float64
        var km: Float64
        var kp: Float64
        var kU: Float64
        kL, km, kp, kU = ann_k_limits(gp, gm)
        if i == 0:
            # one particle at rest (gamma_r = gm): eq. 41 in closed form
            # between the bin edges; 1 + gamma_r - k_U = k_L, exactly
            var gr1 = 1.0 + gm
            var k_lo = kL
            var d_lo = gr1 - kL
            var l = Int(floor((c_log(kL) - ln_k0) / dlnE + 0.5))
            while k_lo < kU:
                var k_hi = c_exp(ln_k0 + (Float64(l) + 0.5) * dlnE)
                var d_hi: Float64
                if k_hi >= kU:
                    k_hi = kU
                    d_hi = kL
                else:
                    d_hi = gr1 - k_hi
                var n_seg: Float64
                var e_seg: Float64
                n_seg, e_seg = ann_at_rest_segment(k_lo, k_hi, gm, d_lo, d_hi)
                if n_seg > 0.0:
                    self._deposit(
                        n_seg, e_seg / n_seg, kl, ln_k0, dlnE, ng, l0, n, buf
                    )
                k_lo = k_hi
                d_lo = d_hi
                l += 1
            PairAnnihilation._store(out, buf, n, r, l0, ng)
            return
        # integration nodes: bin edges inside (k_L, k_U), the limits, the
        # peaks, the centre k_T = (gamma_+ + gamma_-) / 2 and points
        # approaching the peaks and the limits geometrically (the features
        # there are ~1 wide in k)
        var kT = 0.5 * (gp + gm)
        nodes.clear()
        nodes.append(kL)
        nodes.append(kU)
        for kf in [kL, km, kp, kT, kU]:
            if kf >= kL and kf <= kU:
                if kf > kL and kf < kU:
                    nodes.append(kf)
                var step = 1.0
                while step < kf:
                    for sgn in [-1.0, 1.0]:
                        var k_node = kf + sgn * step
                        if k_node > kL and k_node < kU:
                            nodes.append(k_node)
                    step *= 10.0
        for l in range(l0, l0 + n + 1):
            var k_edge = c_exp(ln_k0 + (Float64(l) - 0.5) * dlnE)
            if k_edge > kL and k_edge < kU:
                nodes.append(k_edge)
        sort(nodes)
        var feat: Array[Float64, 5] = [kL, km, kp, kT, kU]
        var gcm_min: Float64
        var gcm_max: Float64
        gcm_min, gcm_max = gcm_limits(gp, gm)
        var pref = PI_RE2 / (beta_gamma(gp) * gp * beta_gamma(gm) * gm)
        for seg in range(len(nodes) - 1):
            var k_lo = nodes[seg]
            var k_hi = nodes[seg + 1]
            if k_hi <= k_lo:
                continue
            # segments at least twice their length away from the limits and
            # peaks (the kinks of the spectrum, which vanishes linearly at the
            # limits) are smooth: plain Gauss-Legendre with fewer points
            var dist = 1.0e300
            for kf in feat:
                if kf >= kL and kf <= kU:
                    dist = min(dist, max(k_lo - kf, kf - k_hi, 0.0))
            var smooth = dist >= 2.0 * (k_hi - k_lo)
            var nq = NQ_SMOOTH if smooth else NQ
            var n_seg = 0.0
            var e_seg = 0.0
            for q in range(nq):
                var t = self.sl[q] if smooth else self.cl[q]
                var wt = self.sw[q] if smooth else self.cw[q]
                var k = k_lo + (k_hi - k_lo) * t
                var emis = _emissivity(k, gp, gm, gcm_min, gcm_max, pref)
                var dn = emis * (k_hi - k_lo) * wt
                n_seg += dn
                e_seg += dn * k
            if n_seg > 0.0:
                self._deposit(
                    n_seg, e_seg / n_seg, kl, ln_k0, dlnE, ng, l0, n, buf
                )
        PairAnnihilation._store(out, buf, n, r, l0, ng)

    @staticmethod
    def _store(dst: F32Ptr, buf: Vec, n: Int, r: Float64, l0: Int, ng: Int):
        """Store the photons per bin of one pair. The closed forms lose
        digits for gamma >~ 1e10: normalise to the photon number 2 R_ij of the
        rate table when the whole spectrum lies on the grid."""
        var tot = 0.0
        for m in range(n):
            tot += buf[m]
        var on_grid = l0 > 0 and l0 + n < ng
        var scale = 2.0 * r / tot if (on_grid and tot > 0.0) else 1.0
        for m in range(n):
            dst[unsafe_offset=m] = Float32(buf[m] * scale)

    def Calc_PairAnn(mut self, dat: AM3Arrays, ele: Vec, pos: Vec):
        """The annihilation rates per particle Gamma_-+ and the photon source
        Q_gamma per ln k, from the densities per ln gamma ele = N_- and
        pos = N_+."""
        if self.verbose:
            print("computing e+e- annihilation")
        var nl = dat.nbins_lep
        var ng = dat.nbins_pho
        var dlnE = dat.dlnE
        resize_zero(self.wp, nl)
        resize_zero(self.wm, nl)
        var any_p = False
        var any_m = False
        for i in range(nl):
            self.wp[i] = pos[i] * dlnE
            self.wm[i] = ele[i] * dlnE
            any_p = any_p or self.wp[i] != 0.0
            any_m = any_m or self.wm[i] != 0.0
        fill_zero(self.Q_gamma)
        if not self.ready(dat):
            fill_zero(self.Gamma_ele)
            fill_zero(self.Gamma_pos)
            return
        # each charge annihilates on the other one, also where it has no
        # particles yet at the start of the step (they may be injected in it)
        if any_p:
            gemv_d(self.R, pd(self.wp), 1.0, pd(self.Gamma_ele))
        else:
            fill_zero(self.Gamma_ele)
        if any_m:
            gemv_d(self.R, pd(self.wm), 1.0, pd(self.Gamma_pos))
        else:
            fill_zero(self.Gamma_pos)
        if not any_p or not any_m:
            return
        var job = AnnStepJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            pd(self.wp),
            pd(self.wm),
            pd(self.part),
            ng,
        )
        parallel_run[ann_step_work](job, min(self.nthreads, N_CHUNKS))
        var q = pd(self.Q_gamma)
        for c in range(N_CHUNKS):
            var src = pd(self.part).unsafe_offset(c * ng)
            for l in range(ng):
                q[unsafe_offset=l] += src[unsafe_offset=l]
        # number per bin and time -> source per ln k
        for l in range(ng):
            q[unsafe_offset=l] /= dlnE
