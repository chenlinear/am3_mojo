"""Electron-positron annihilation, e+ e- -> gamma gamma (am3_mojo addition;
AM3 has no pair annihilation).

Exact treatment of Svensson (1982, ApJ 258, 321) for isotropic e+ and e- of
Lorentz factors g+ and g-:
  * the angle-averaged rate <sigma v>(g+, g-) (Svensson eq. 18),
    [G(g_r+) - G(g_r-)] / (2 p+ p- g+ g-) with the relative Lorentz factors
    g_r+- = g+ g- +- p+ p-, p = sqrt(g^2 - 1);
  * the photon spectrum v dsigma/dk (k; g+, g-) (eqs. 55-58; eq. 41 if one
    particle is at rest), counting both photons.
Vurm & Poutanen (2009), whose treatment AM3's gamma gamma pair production
builds on, use the same rate and an equivalent form of the spectrum. Rates
are in sigma_T c (= 1 in the code units of the leptons and photons), energies
in m_e c^2. Valid for all Lorentz factors while the relative velocity is
>> alpha c (no Coulomb focusing or positronium formation).

On the grids:
  * rate table R[i, j] for every pair of lepton bins (bin 0 is g = 1, at rest);
  * photon kernel K[i, j, l] (photons of pair (i, j) in bin l), exact for
    min(g_i, g_j) <= GAMMA_EXACT: Gauss-Legendre quadrature keeping photon
    number and energy segment by segment, or with one particle at rest,
    eq. 41 integrated over each bin in closed form;
  * pairs with both Lorentz factors above GAMMA_EXACT (rate <~1e-3 of that of
    cold pairs) use Svensson's eq. 59: two photons at
    k+- = [g+-(1 + b+-) + g-+(1 - b-+)] / 2, each shared between the two
    neighbouring bins (number and energy kept).
Per time step: the sinks alpha_e-(g_j) = sum_i n_e+(i) dlnE R[i, j] (and the
other way round) and the photon source. The pair sums run in fixed chunks,
added in a fixed order, so results do not depend on the thread count.

Limitations:
  * AM3's solver keeps lepton bins 0 and 1 empty (g < 1.22): pairs that cool
    to rest leave the system there, as in AM3, and the 511 keV line is not
    produced;
  * the sinks and photon source of a step come from the densities at its
    start, so a step makes about alpha dt / 2 more photons than two per pair
    lost (alpha: the annihilation rate). Keep the time step below about 0.05
    of the shortest annihilation time;
  * Svensson's formulas are the Born approximation (no Coulomb corrections).
    Three-photon annihilation, 4 (pi^2 - 9) alpha / (3 pi) ~ 1/371 of the
    two-photon rate for slow pairs (Berestetskii, Lifshitz & Pitaevskii,
    sec. 89), is neglected.
"""

from std.math import sqrt as _sqrt
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.utils.linalg import MatD, gemv_d
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.specfun import gauss_legendre

comptime PI_RE2: Float64 = 0.375  # pi r_e^2 / sigma_T
comptime GAMMA_EXACT: Float64 = 100.0
comptime NQ = 16  # quadrature points per segment next to a limit or peak
comptime NQ_SMOOTH = 4  # quadrature points per segment away from them
comptime N_CHUNKS = 32  # fixed partition of the per-step pair sum


# ----------------------------------------------------------------------------
# Svensson 1982
# ----------------------------------------------------------------------------
@always_inline
def mom(g: Float64) -> Float64:
    """p = sqrt(g^2 - 1), without cancellation."""
    return c_sqrt(max((g - 1.0) * (g + 1.0), 0.0))


@always_inline
def acosh(x: Float64) -> Float64:
    return c_log(x + mom(x))


def gcm_limits(gp: Float64, gm: Float64) -> Tuple[Float64, Float64]:
    """Range of the centre-of-momentum Lorentz factor (min, max)."""
    var pp = mom(gp)
    var pm = mom(gm)
    var gmax = c_sqrt(0.5 * (1.0 + gp * gm + pp * pm))
    var gmin = c_sqrt(
        0.5 * (1.0 + (gp * gp + gm * gm - 1.0) / (gp * gm + pp * pm))
    )
    return (gmin, gmax)


def _G(gr: Float64) -> Float64:
    """G(g_r) = integral_1^g_r sqrt(x^2 - 1) sigma_rest(x) dx [sigma_T]."""
    var ell = acosh(gr)
    return (
        0.75
        * (
            (gr - 1.0) * c_sqrt(gr - 1.0) / (2.0 * c_sqrt(gr + 1.0)) * ell
            - (gr + 1.0)
            + 0.75 * ell * ell
        )
        + 1.5
    )


def _G_prime(x: Float64) -> Float64:
    """dG/dg_r = sqrt(x^2 - 1) sigma_rest(x)."""
    var xm1 = max(x - 1.0, 0.0)
    var p = c_sqrt(xm1 * (x + 1.0))
    var acosh_over_p = c_log1p(xm1 + p) / p if xm1 > 1e-12 else 1.0 - xm1 / 3.0
    return (
        PI_RE2
        / (x + 1.0)
        * ((x * x + 4.0 * x + 1.0) * acosh_over_p - (x + 3.0))
    )


def ann_rate(gp: Float64, gm: Float64, gl_x: Vec, gl_w: Vec) -> Float64:
    """<sigma v>(g+, g-) / (sigma_T c), Svensson eq. 18 in a stable form
    (mean of G' over [g_r-, g_r+] where the difference G(g_r+) - G(g_r-)
    would cancel)."""
    var pp = mom(gp)
    var pm = mom(gm)
    var grp = gp * gm + pp * pm
    var grm = max(1.0, (gp * gp + gm * gm - 1.0) / grp)
    if pp * pm > 1e-2 * grp:
        return (_G(grp) - _G(grm)) / (2.0 * pp * pm * gp * gm)
    var s = 0.0
    for q in range(len(gl_x)):
        var x = 0.5 * (grp - grm) * gl_x[q] + 0.5 * (grp + grm)
        s += gl_w[q] * _G_prime(x)
    return 0.5 * s / (gp * gm)


def _H(
    k: Float64,
    gp: Float64,
    gm: Float64,
    gcm: Float64,
    sign: Float64,
    gs2: Float64,
) -> Float64:
    """H_+ (sign = 1) or H_- (sign = -1) of Svensson eqs. 56-58."""
    var gopp = gm if sign > 0.0 else gp
    var c = (gopp - k) * (gopp - k) - 1.0
    var d = gopp * (gp + gm) + sign * k * (gp - gm)
    var gs = c_sqrt(gs2)
    if c_fabs(c) < 1e-6:
        # Svensson eq. 58 (c = 0), accurate to <~ 0.3 |c|; the general form
        # below loses digits for |c| << 1e-6
        return (
            (2.0 / 3.0) * gcm * gcm * gcm + 2.0 * gcm + 1.0 / gcm
        ) / gs + 0.5 * ((2.0 / 3.0) * gcm * gcm * gcm - d * gcm) / (
            gs * gs * gs
        )
    var u = c_sqrt(max(c * gcm * gcm + gs2, 0.0))
    var sc = c_sqrt(c_fabs(c))
    var I: Float64
    if c > 0.0:
        I = c_log(gcm * sc + u) / sc
    else:
        I = c_asin(min(gcm * sc / gs, 1.0)) / sc
    var t2 = 0.0
    if u > 0.0:
        t2 = (1.0 / gcm - gcm / c + gcm / (2.0 * gs2) * (2.0 * c - d)) / u
    return (2.0 + (1.0 - gs2) / c) * I + t2 + gcm / c * u


def ann_spectrum(k: Float64, gp: Float64, gm: Float64) -> Float64:
    """v dsigma/dk (k; g+, g-) / (sigma_T c), Svensson eq. 55 (both photons:
    integral dk = 2 <sigma v>, integral k dk = (g+ + g-) <sigma v>)."""
    var gmin: Float64
    var gmax: Float64
    gmin, gmax = gcm_limits(gp, gm)
    var pref = PI_RE2 / (mom(gp) * gp * mom(gm) * gm)
    return _ann_spectrum(k, gp, gm, gmin, gmax, pref)


@always_inline
def _ann_spectrum(
    k: Float64,
    gp: Float64,
    gm: Float64,
    gmin: Float64,
    gmax: Float64,
    pref: Float64,
) -> Float64:
    """ann_spectrum with the pair's constants: the limits of g_cm and
    pref = pi r_e^2 / (p+ g+ p- g-)."""
    var S = gp + gm
    var gs2 = k * (S - k)
    if gs2 <= 0.0:
        return 0.0
    var gU = min(gmax, c_sqrt(gs2))
    if gU <= gmin:
        return 0.0
    return pref * (
        _prim(k, gp, gm, S, gs2, gU) - _prim(k, gp, gm, S, gs2, gmin)
    )


@always_inline
def _prim(
    k: Float64, gp: Float64, gm: Float64, S: Float64, gs2: Float64, g: Float64
) -> Float64:
    """Primitive (in g_cm) of Svensson eq. 55."""
    return (
        c_sqrt(max(S * S - 4.0 * g * g, 0.0))
        + _H(k, gp, gm, g, 1.0, gs2)
        + _H(k, gp, gm, g, -1.0, gs2)
    )


def ann_spectrum_at_rest(k: Float64, gr: Float64) -> Float64:
    """Spectrum when one particle is at rest and the other has Lorentz factor
    gr (Svensson eq. 41)."""
    var p = mom(gr)
    var lo = 0.5 * (1.0 / (gr + p) + 1.0)
    var hi = 0.5 * (gr + p + 1.0)
    if k < lo or k > hi:
        return 0.0
    var num = -(3.0 + gr) / (1.0 + gr) + (3.0 + gr) / k - 1.0 / (k * k)
    var t = 1.0 - k / (1.0 + gr)
    return PI_RE2 / (p * gr) * (num / (t * t) - 2.0)


def ann_at_rest_segment(
    a: Float64, b: Float64, gr: Float64, dDa: Float64, dDb: Float64
) -> Tuple[Float64, Float64]:
    """Photon number and energy of the at-rest spectrum (eq. 41) between
    k = a and b, in closed form. With D = 1 + gr, B = 3 + gr, A = -B / D,
    C = pi r_e^2 / (p gr) and L = ln k - ln(D - k), the primitives are
      N(k) = C [1/k + (B - 2/D) L - 1/(D - k) - 2k],
      E(k) = C [A D^2 ln(D - k) - L - D/(D - k) - k^2]
    (A D + B = 0 removes the large terms). dDa, dDb = D - a, D - b."""
    var p = mom(gr)
    var C = PI_RE2 / (p * gr)
    var D = 1.0 + gr
    var B = 3.0 + gr
    var A = -B / D
    # ln((D - b) / (D - a)): for a segment short against D - a, from log1p
    # (at large gr, D - a and D - b round alike, so their ratio loses all
    # digits). Near the top, from the exact distances (b = k_U is rounded,
    # D - k_U is not)
    var l1 = c_log1p(-(b - a) / dDa) if b - a < 0.5 * dDa else c_log(dDb / dDa)
    var dL = c_log(b / a) - l1
    var inv = (b - a) / (dDa * dDb)  # 1/(D - b) - 1/(D - a)
    var n = C * ((a - b) / (a * b) + (B - 2.0 / D) * dL - inv - 2.0 * (b - a))
    var e = C * (A * D * D * l1 - dL - D * inv - (b - a) * (b + a))
    return (n, e)


def ann_k_limits(
    gp: Float64, gm: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    """Photon energies (k_L, k_-, k_+, k_U): the kinematic limits and the
    two peaks of the spectrum."""
    var pp = mom(gp)
    var pm = mom(gm)
    var ap = 1.0 / (gp + pp)  # g (1 - b)
    var am = 1.0 / (gm + pm)
    return (
        0.5 * (ap + am),
        0.5 * (ap + gm + pm),
        0.5 * (gp + pp + am),
        0.5 * (gp + pp + gm + pm),
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
    var wp: FPtr  # n_e+ dlnE
    var wm: FPtr  # n_e- dlnE
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
            var w = J.wp[unsafe_offset=i] * J.wm[unsafe_offset=j]
            if i != j:
                w += J.wp[unsafe_offset=j] * J.wm[unsafe_offset=i]
            if w == 0.0:
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
                op.unsafe_store(op.unsafe_load[width=W]() + w * x)
                m += W
            while m < n:
                o[unsafe_offset=m] += w * Float64(v[unsafe_offset=m])
                m += 1
        # peak approximation for pairs with both Lorentz factors high
        var nd = len(A.da)
        var d0 = c * nd // N_CHUNKS
        var d1 = (c + 1) * nd // N_CHUNKS
        for d in range(d0, d1):
            var i = A.da[d]
            var j = A.db[d]
            var w = J.wp[unsafe_offset=i] * J.wm[unsafe_offset=j]
            if i != j:
                w += J.wp[unsafe_offset=j] * J.wm[unsafe_offset=i]
            if w == 0.0:
                continue
            var rw = w * A.dr[d]
            var lm = A.dl_m[d]
            if lm >= 0:
                out[unsafe_offset=lm] += rw * (1.0 - A.dw_m[d])
                out[unsafe_offset=lm + 1] += rw * A.dw_m[d]
            var lp = A.dl_p[d]
            if lp >= 0:
                out[unsafe_offset=lp] += rw * (1.0 - A.dw_p[d])
                out[unsafe_offset=lp + 1] += rw * A.dw_p[d]
        c += nt


struct PairAnnihilation(Movable):
    var verbose: Bool
    var nthreads: Int
    var R: MatD  # rate table (nbins_lep x nbins_lep, symmetric)
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
    # peak approximation: pair d = (da[d] <= db[d]) puts R photons at k- and
    # at k+, shared between bins dl_m[d], dl_m[d] + 1 (weight dw_m[d] of the
    # upper one) and dl_p[d], dl_p[d] + 1 (dl < 0: off the grid)
    var da: VecI
    var db: VecI
    var dr: Vec
    var dl_m: VecI
    var dw_m: Vec
    var dl_p: VecI
    var dw_p: Vec
    var gl_x: Vec
    var gl_w: Vec
    var cl: Vec  # quadrature nodes on [0, 1], clustered at the ends
    var cw: Vec
    var sl: Vec  # NQ_SMOOTH Gauss-Legendre nodes on [0, 1]
    var sw: Vec
    # per-step results
    var a_ele: Vec  # annihilation rate of electrons (code units)
    var a_pos: Vec
    var q_pho: Vec  # photon source per ln k
    var wp: Vec
    var wm: Vec
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
        self.gl_x = Vec()
        self.gl_w = Vec()
        gauss_legendre(8, self.gl_x, self.gl_w)
        var xq = Vec()
        var wq = Vec()
        gauss_legendre(NQ, xq, wq)
        self.cl = zeros(NQ)
        self.cw = zeros(NQ)
        for q in range(NQ):
            # t in [0, 1] mapped by (1 - cos(pi t)) / 2: clusters the nodes at
            # the segment ends, where the spectrum has square-root edges
            var t = 0.5 * (xq[q] + 1.0)
            self.cl[q] = 0.5 * (1.0 - c_cos(mathpi * t))
            self.cw[q] = 0.25 * mathpi * c_sin(mathpi * t) * wq[q]
        gauss_legendre(NQ_SMOOTH, xq, wq)
        self.sl = zeros(NQ_SMOOTH)
        self.sw = zeros(NQ_SMOOTH)
        for q in range(NQ_SMOOTH):
            self.sl[q] = 0.5 * (xq[q] + 1.0)
            self.sw[q] = 0.5 * wq[q]
        self.a_ele = Vec()
        self.a_pos = Vec()
        self.q_pho = Vec()
        self.wp = Vec()
        self.wm = Vec()
        self.part = Vec()
        self.valid = False
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.a_ele, dat.nbins_lep)
        resize_zero(self.a_pos, dat.nbins_lep)
        resize_zero(self.q_pho, dat.nbins_pho)

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
                var r = ann_rate(gam[i], gam[j], self.gl_x, self.gl_w)
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
        var lo = to_int(c_floor((c_log(kL) - ln_k0) / dlnE))
        var hi = to_int(c_floor((c_log(kU) - ln_k0) / dlnE)) + 2
        return (max(lo, 0), min(hi, ng))

    @staticmethod
    def _split(
        k: Float64, kl: Vec, ln_k0: Float64, dlnE: Float64, ng: Int
    ) -> Tuple[Int, Float64]:
        """Bin l with kl[l] <= k < kl[l + 1] and the weight w of bin l + 1
        that keeps number and energy: (1 - w) kl[l] + w kl[l + 1] = k."""
        var l = to_int(c_floor((c_log(k) - ln_k0) / dlnE))
        if l < 0 or l + 1 >= ng:
            return (-1, 0.0)
        return (l, (k - kl[l]) / (kl[l + 1] - kl[l]))

    def _deposit(
        self,
        n_ph: Float64,
        k: Float64,
        kl: FPtr,
        ln_k0: Float64,
        dlnE: Float64,
        ng: Int,
        l0: Int,
        n: Int,
        mut buf: Vec,
    ):
        """Add n_ph photons of mean energy k to the two bins whose centres
        bracket k, keeping number and energy."""
        var l = to_int(c_floor((c_log(k) - ln_k0) / dlnE))
        if l < 0 or l + 1 >= ng:
            return
        var x = (k - kl[unsafe_offset=l]) / (
            kl[unsafe_offset=l + 1] - kl[unsafe_offset=l]
        )
        if l >= l0 and l < l0 + n:
            buf[l - l0] += (1.0 - x) * n_ph
        if l + 1 >= l0 and l + 1 < l0 + n:
            buf[l + 1 - l0] += x * n_ph

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
            # one particle at rest: closed form between the bin edges
            var D = 1.0 + gm
            var dD_top = 0.5 * (1.0 + 1.0 / (gm + mom(gm)))  # D - kU, exact
            var a = kL
            var dDa = D - kL
            var l = to_int(c_floor((c_log(kL) - ln_k0) / dlnE + 0.5))
            while a < kU:
                var b = c_exp(ln_k0 + (Float64(l) + 0.5) * dlnE)
                var dDb: Float64
                if b >= kU:
                    b = kU
                    dDb = dD_top
                else:
                    dDb = D - b
                var n_seg: Float64
                var e_seg: Float64
                n_seg, e_seg = ann_at_rest_segment(a, b, gm, dDa, dDb)
                if n_seg > 0.0:
                    self._deposit(
                        n_seg, e_seg / n_seg, kl, ln_k0, dlnE, ng, l0, n, buf
                    )
                a = b
                dDa = dDb
                l += 1
            PairAnnihilation._store(out, buf, n, r, l0, ng)
            return
        # integration nodes: bin edges inside (kL, kU), the limits, the peaks,
        # the centre (g+ + g-) / 2 and points approaching the peaks and the
        # limits geometrically (the features there are ~1 wide in k)
        nodes.clear()
        nodes.append(kL)
        nodes.append(kU)
        for b in [kL, km, kp, 0.5 * (gp + gm), kU]:
            if b >= kL and b <= kU:
                if b > kL and b < kU:
                    nodes.append(b)
                var d = 1.0
                while d < b:
                    for s in [-1.0, 1.0]:
                        var e = b + s * d
                        if e > kL and e < kU:
                            nodes.append(e)
                    d *= 10.0
        for l in range(l0, l0 + n + 1):
            var e = c_exp(ln_k0 + (Float64(l) - 0.5) * dlnE)
            if e > kL and e < kU:
                nodes.append(e)
        sort(nodes)
        var feat: Array[Float64, 5] = [kL, km, kp, 0.5 * (gp + gm), kU]
        var gmin: Float64
        var gmax: Float64
        gmin, gmax = gcm_limits(gp, gm)
        var pref = PI_RE2 / (mom(gp) * gp * mom(gm) * gm)
        for s in range(len(nodes) - 1):
            var a = nodes[s]
            var b = nodes[s + 1]
            if b <= a:
                continue
            # segments at least twice their length away from the limits and
            # peaks (where the spectrum has square-root edges and kinks) are
            # smooth: plain Gauss-Legendre with fewer points
            var dist = 1.0e300
            for f in feat:
                if f >= kL and f <= kU:
                    dist = min(dist, max(a - f, f - b, 0.0))
            var smooth = dist >= 2.0 * (b - a)
            var nq = NQ_SMOOTH if smooth else NQ
            var n_seg = 0.0
            var e_seg = 0.0
            for q in range(nq):
                var t = self.sl[q] if smooth else self.cl[q]
                var wt = self.sw[q] if smooth else self.cw[q]
                var k = a + (b - a) * t
                var f = _ann_spectrum(k, gp, gm, gmin, gmax, pref)
                var dn = f * (b - a) * wt
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
        digits for g >~ 1e10: normalise to the photon number 2R of the rate
        table when the whole spectrum lies on the grid."""
        var tot = 0.0
        for m in range(n):
            tot += buf[m]
        var on_grid = l0 > 0 and l0 + n < ng
        var scale = 2.0 * r / tot if (on_grid and tot > 0.0) else 1.0
        for m in range(n):
            dst[unsafe_offset=m] = Float32(buf[m] * scale)

    def Calc_PairAnn(mut self, dat: AM3Arrays, ele: Vec, pos: Vec):
        """Annihilation rates of electrons and positrons and the photon
        source per ln k, from the densities ele, pos (n(x) per ln gamma)."""
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
        fill_zero(self.q_pho)
        if not self.ready(dat):
            fill_zero(self.a_ele)
            fill_zero(self.a_pos)
            return
        # each charge annihilates on the other one, also where it has no
        # particles yet at the start of the step (they may be injected in it)
        if any_p:
            gemv_d(self.R, pd(self.wp), 1.0, pd(self.a_ele))
        else:
            fill_zero(self.a_ele)
        if any_m:
            gemv_d(self.R, pd(self.wm), 1.0, pd(self.a_pos))
        else:
            fill_zero(self.a_pos)
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
        var q = pd(self.q_pho)
        for c in range(N_CHUNKS):
            var src = pd(self.part).unsafe_offset(c * ng)
            for l in range(ng):
                q[unsafe_offset=l] += src[unsafe_offset=l]
        # number per bin and time -> source per ln k
        for l in range(ng):
            q[unsafe_offset=l] /= dlnE
