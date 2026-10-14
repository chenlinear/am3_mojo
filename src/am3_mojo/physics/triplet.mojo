"""Triplet pair production, e+- + gamma -> e+- + e+ + e- (am3_mojo addition;
AM3 does not have the process).

The total cross section sigma(k) and the products' energy distributions come
from tables computed with the exact lowest-order QED matrix element, all
eight Feynman diagrams (tools/triplet, physics/triplet_tables.mojo). k is the
photon energy in the rest frame of the target lepton [m_e c^2].

For gamma >> 1 and a soft photon (eps << gamma, here eps <= m_e c^2), a
product's lab energy is the fraction x = (p.k)/(p1.k) of the lepton's, which
does not depend on the photon direction. In an isotropic photon field, the
rate and the products then depend only on P = gamma eps:
    rate     R(P)    = 1/(2 P^2) int_4^{2P} dk k sigma(k)
    products G(P, x) = 1/(2 P^2) int_4^{2P} dk k sigma(k) f(k, x)
with f(k, x) dx the products of one interaction: the positron, the electron
of lower energy and that of higher energy. The latter counts as the target
after the interaction: it keeps the fraction 1 - xi, with xi = x+ + x_low
the energy the pair takes.

Kinetic terms: the pairs are sources, shared between grid points to keep
each product's number and energy. The photon disappears (sink of the photon
solver). The returning lepton loses the energy the pair takes as a
continuous loss (a cooling term of the lepton solvers) at the rate
    sum_eps n(eps) dlnE R(gamma eps) <x+ + x_low>(gamma eps).
Rates are in sigma_T c (= 1 in the code units of the leptons and photons).

The loss is continuous because where the triplet loss exceeds the
inverse-Compton loss (gamma eps >~ 3e8), the lepton loses <~6e-7 of its
energy per interaction (7e-5 at gamma eps = 1e6, where the process takes half
as much as inverse Compton). Where the lepton loses much (<xi> = 0.018 at
gamma eps = 1e3, 0.085 at 1e2), the inverse-Compton loss dominates, which AM3
treats as continuous too. Re-adding the lepton as a source at the energy it
keeps would move it down by at most one interaction per step, as sources come
from the densities at the step's start. Yet the process often cools the
leptons through more than one bin per step, even at short steps: in
leptohadronic_all at the end of its run (step 300), the triplet energy-loss
time of the pairs from 1e14 to 1e16 eV is 3-7 s, against a step of 30 s
(3e-3 R/c). AM3's solver integrates a continuous loss within the step, also
where it is fast.

Limitations (their size: tests/literature_checks/README.md, Results):
  * soft photons only (eps <= m_e c^2, gamma >> 1): the lab energies x gamma
    neglect the photon's energy and direction. On photons above about
    0.25 m_e c^2, some products fall below the lepton grid into bin 0, which
    AM3's solver empties every step; their photons are still absorbed and
    the lepton still loses their energy;
  * the spread of the energy the returning lepton loses per interaction is
    neglected;
  * the pairs of a step come from the leptons at its start (as AM3's other
    sources), so each generation of a cascade takes one step;
  * above the last table node (k = 1e10) the distributions are extrapolated;
  * lowest order (Born): radiative corrections (relative order alpha) and
    the photon's energy in the energy balance (eps <= m_e c^2 against
    gamma m_e c^2) are neglected.
"""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.specfun import gauss_legendre
from am3_mojo.physics.triplet_tables import (
    TPP_LY0,
    TPP_DLY,
    TPP_K,
    TPP_SIG,
    TPP_DIST,
)

comptime SIG_UNIT: Float64 = 3.0 * alpha_PDG / (
    8.0 * 3.141592653589793
)  # alpha r_e^2 / sigma_T
# above K_ASYM: the high-energy expansion of sigma(k) (Lee et al., rounded
# coefficients)
comptime K_ASYM: Float64 = 1000.0
# above the last node, the asymptotic laws fitted to the nodes k >= K_FIT:
# f = g0(y) + g1(y)/ln k at fixed y <= Y_SW, k f = A(x) ln k + B(x) at fixed
# x >= X_LO, and (a0 ln y + b0)/y between
comptime K_FIT: Float64 = 1e6
comptime Y_SW: Float64 = 100.0
comptime X_LO: Float64 = 1e-5
comptime NGL = 4  # Gauss-Legendre points per slab of the k integral
comptime N_ROLES = 3  # tables: e+, lower-energy e-, higher-energy e- (recoil)
comptime ROLE_POS = 0
comptime ROLE_LOW = 1
comptime ROLE_HIGH = 2
comptime N_KR = 2  # kernels: the produced pair (e+, lower-energy e-)
comptime BLK = 4 * WD  # offsets per register block of the per-step convolution


def _parse(s: StringSlice) raises -> Vec:
    var out = Vec()
    for tok in s.split(" "):
        if tok.byte_length() > 0:
            out.append(atof(tok))
    return out^


@always_inline
def tpp_nbins(k: Float64) -> Int:
    """Number of ln y bins of the tables for photon energy k (y = x k)."""
    return to_int(c_floor((c_log(k) + 0.1 - TPP_LY0) / TPP_DLY)) + 1


def sigma_threshold(k: Float64) -> Float64:
    """sigma / (alpha r_e^2) near threshold (Lee et al. 2021, arXiv:2010.15430),
    r = (s-9)/(s-1)."""
    var s = 1.0 + 2.0 * k
    var r = (s - 9.0) / (s - 1.0)
    return (
        3.141592653589793
        / (3.0 * c_sqrt(3.0))
        * (r * r * (4.0 / 27.0 + r * (602.0 / 243.0 + r * 4217.0 / 4374.0)))
    )


def sigma_asym(k: Float64) -> Float64:
    """sigma / (alpha r_e^2) at high energy, with the 1/k terms
    (arXiv:2010.15430)."""
    var L = c_log(2.0 * k)
    return (
        28.0 / 9.0 * L
        - 218.0 / 27.0
        - (4.0 / 3.0 * L * L * L - 4.08 * L * L + 12.85 * L - 30.89) / k
    )


def sigma_threshold_slope(k: Float64) -> Float64:
    """d ln sigma / d ln(k - 4) of sigma_threshold (r = 1 - 4/k, dr/dk = 4/k^2).
    """
    var r = 1.0 - 4.0 / k
    var p = 4.0 / 27.0 + r * (602.0 / 243.0 + r * 4217.0 / 4374.0)
    var dp = 602.0 / 243.0 + 2.0 * r * 4217.0 / 4374.0
    return (k - 4.0) * 4.0 / (k * k) * (2.0 / r + dp / p)


def sigma_asym_slope(k: Float64) -> Float64:
    """d ln sigma / d ln(k - 4) of sigma_asym."""
    var L = c_log(2.0 * k)
    var poly = 4.0 / 3.0 * L * L * L - 4.08 * L * L + 12.85 * L - 30.89
    var dpoly = 4.0 * L * L - 8.16 * L + 12.85
    return (
        (k - 4.0) * (28.0 / 9.0 / k - (dpoly - poly) / (k * k)) / sigma_asym(k)
    )


struct TripletJob(Movable):
    """Per-step photon convolution of the kernels: for every target bin i
    that holds leptons (ele or pos nonzero),
        Z_r[i][o] = sum_j n_ph[j] dlnE W_r[m(i, j)][o]
    (o = offset of the product bin below i), and for every bin the
    energy-loss rate.
    """

    var T: Pointer[TripletPairProduction, MutUntrackedOrigin]
    var nph: FPtr
    var ele: FPtr
    var pos: FPtr
    var two: Bool  # pos given (not merged e+-)
    var nl: Int
    var j0: Int
    var jmax: Int
    var dl: Float64

    def __init__(
        out self,
        T: Pointer[TripletPairProduction, MutUntrackedOrigin],
        nph: FPtr,
        ele: FPtr,
        pos: FPtr,
        two: Bool,
        nl: Int,
        j0: Int,
        jmax: Int,
        dl: Float64,
    ):
        self.T = T
        self.nph = nph
        self.ele = ele
        self.pos = pos
        self.two = two
        self.nl = nl
        self.j0 = j0
        self.jmax = jmax
        self.dl = dl


def triplet_rows_work(
    job: Pointer[TripletJob, MutUntrackedOrigin], t: Int, nt: Int
):
    ref J = job[]
    ref T = J.T[]
    var i = t
    while i < J.nl:
        var jmin = max(0, T.m0 - i + J.j0)
        var e = 0.0
        for j in range(jmin, J.jmax + 1):
            var n = J.nph[unsafe_offset=j]
            if n != 0.0:
                e += n * J.dl * T.Lxi[i + j - J.j0 - T.m0]
        T.eloss[i] = e
        # (rows of empty bins are unused: the e- and e+ totals make the pairs)
        var full = J.ele[unsafe_offset=i] != 0.0 or (
            J.two and J.pos[unsafe_offset=i] != 0.0
        )
        var nz = T.nz[i]
        for r in range(N_KR):
            var z = pd(T.Z[r]).unsafe_offset(T.zoff[i])
            var w0 = pd(T.W[r])
            # blocks of BLK offsets, accumulated in registers over the
            # photons. The rows that reach offset ob (nd[q] > ob) are those
            # from q = qblk[ob / BLK] on
            var ob = 0
            while ob < nz:
                var a0 = SIMD[DType.float64, WD](0.0)
                var a1 = SIMD[DType.float64, WD](0.0)
                var a2 = SIMD[DType.float64, WD](0.0)
                var a3 = SIMD[DType.float64, WD](0.0)
                if full:
                    var js = max(jmin, T.qblk[ob // BLK] - i + J.j0 + T.m0)
                    for j in range(js, J.jmax + 1):
                        var n = J.nph[unsafe_offset=j]
                        if n == 0.0:
                            continue
                        var w = SIMD[DType.float64, WD](n * J.dl)
                        var src = w0.unsafe_offset(
                            T.woff[i + j - J.j0 - T.m0] + ob
                        )
                        a0 = src.unsafe_load[width=WD]().fma(w, a0)
                        a1 = (
                            src.unsafe_offset(WD)
                            .unsafe_load[width=WD]()
                            .fma(w, a1)
                        )
                        a2 = (
                            src.unsafe_offset(2 * WD)
                            .unsafe_load[width=WD]()
                            .fma(w, a2)
                        )
                        a3 = (
                            src.unsafe_offset(3 * WD)
                            .unsafe_load[width=WD]()
                            .fma(w, a3)
                        )
                var zp = z.unsafe_offset(ob)
                zp.unsafe_store(a0)
                zp.unsafe_offset(WD).unsafe_store(a1)
                zp.unsafe_offset(2 * WD).unsafe_store(a2)
                zp.unsafe_offset(3 * WD).unsafe_store(a3)
                ob += BLK
        i += nt


struct TripletPairProduction(Movable):
    var verbose: Bool
    var nthreads: Int
    # tables (parsed once)
    var parsed: Bool
    var tk: Vec  # photon energies k of the nodes
    var tsig: Vec  # sigma / (alpha r_e^2) at the nodes
    # spline of ln sigma up to K_ASYM (_fit_sigma_spline): knots (tsu, tsv),
    # second derivatives tsm
    var nsp: Int
    var tsu: Vec
    var tsv: Vec
    var tsm: Vec
    var toff: VecI  # start of node n in tdist (three roles of tpp_nbins(k_n))
    var tdist: Vec  # dN/dln y per interaction
    # extrapolation above the last node (per role r): g0/g1 on the ln y bins
    # up to Y_SW [r * nby + b], A/B on ln x = ln X_LO + j TPP_DLY
    # [r * nlx + j], a0/b0 [r]
    var nby: Int
    var nlx: Int
    var eg0: Vec
    var eg1: Vec
    var eA: Vec
    var eB: Vec
    var ea0: Vec
    var eb0: Vec
    # kernels on the current grid, for P = gamma eps = exp((m0 + q) dlnE)
    var m0: Int
    var R: Vec  # rate per photon [sigma_T]
    var Lxi: Vec  # R <x+ + x_low>: lepton energy fraction lost per unit time
    var nd: VecI  # product offsets 0 .. nd[q] - 1 below the target bin
    var woff: VecI  # (rows padded with zeros to multiples of BLK)
    var qblk: VecI  # first row q with nd[q] > b BLK
    var W: List[Vec]  # per product of the pair: deposit rates [woff[q] + o]
    var valid: Bool
    # per-step results
    var eloss: Vec  # energy-loss rate of an e+- (energy fraction per unit time)
    var alpha_g: Vec  # photon sink
    var Z: List[
        Vec
    ]  # per role: photon-convolved rows [zoff[i] + o] (padded as W)
    var zoff: VecI
    var nz: VecI
    var nlep: Vec
    var step_valid: Bool

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.nthreads = 1
        self.parsed = False
        self.tk = Vec()
        self.tsig = Vec()
        self.nsp = 0
        self.tsu = Vec()
        self.tsv = Vec()
        self.tsm = Vec()
        self.toff = VecI()
        self.tdist = Vec()
        self.nby = 0
        self.nlx = 0
        self.eg0 = Vec()
        self.eg1 = Vec()
        self.eA = Vec()
        self.eB = Vec()
        self.ea0 = Vec()
        self.eb0 = Vec()
        self.m0 = 0
        self.R = Vec()
        self.Lxi = Vec()
        self.nd = VecI()
        self.woff = VecI()
        self.qblk = VecI()
        self.W = List[Vec]()
        self.Z = List[Vec]()
        for _ in range(N_KR):
            self.W.append(Vec())
            self.Z.append(Vec())
        self.valid = False
        self.eloss = Vec()
        self.alpha_g = Vec()
        self.zoff = VecI()
        self.nz = VecI()
        self.nlep = Vec()
        self.step_valid = False
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.eloss, dat.nbins_lep)
        resize_zero(self.alpha_g, dat.nbins_pho)
        self.step_valid = False

    def free_kernels(mut self):
        self.R = Vec()
        self.Lxi = Vec()
        self.nd = VecI()
        self.woff = VecI()
        self.qblk = VecI()
        for r in range(N_KR):
            self.W[r] = Vec()
            self.Z[r] = Vec()
        self.zoff = VecI()
        self.nz = VecI()
        self.valid = False
        self.step_valid = False

    def ready(self, dat: AM3Arrays) -> Bool:
        return self.valid and len(self.nz) == dat.nbins_lep

    # ------------------------------------------------------------------
    # tables
    # ------------------------------------------------------------------
    def _parse_tables(mut self) raises:
        if self.parsed:
            return
        self.tk = _parse(TPP_K)
        self.tsig = _parse(TPP_SIG)
        self.tdist = _parse(TPP_DIST)
        self.toff = VecI()
        var o = 0
        for n in range(len(self.tk)):
            self.toff.append(o)
            o += N_ROLES * tpp_nbins(self.tk[n])
        if o != len(self.tdist) or len(self.tsig) != len(self.tk):
            raise Error("triplet tables are inconsistent")
        self._fit_sigma_spline()
        self._fit_asymptotics()
        self.parsed = True

    def _fit_sigma_spline(mut self):
        """Cubic spline of ln sigma in u = ln(k - 4) through the nodes up to
        K_ASYM, with the slopes of the threshold and high-energy expansions at
        its ends. The second derivatives tsm solve a symmetric tridiagonal
        system (Thomas algorithm)."""
        var n = 0
        while n < len(self.tk) and self.tk[n] <= K_ASYM:
            n += 1
        self.nsp = n
        self.tsu = Vec()
        self.tsv = Vec()
        for i in range(n):
            self.tsu.append(c_log(self.tk[i] - 4.0))
            self.tsv.append(c_log(self.tsig[i]))
        var h = zeros(n - 1)
        for i in range(n - 1):
            h[i] = self.tsu[i + 1] - self.tsu[i]
        # rows i: h[i-1] M[i-1] + a[i] M[i] + h[i] M[i+1] = d[i]
        var a = zeros(n)
        var d = zeros(n)
        a[0] = 2.0 * h[0]
        d[0] = 6.0 * (
            (self.tsv[1] - self.tsv[0]) / h[0]
            - sigma_threshold_slope(self.tk[0])
        )
        for i in range(1, n - 1):
            a[i] = 2.0 * (h[i - 1] + h[i])
            d[i] = 6.0 * (
                (self.tsv[i + 1] - self.tsv[i]) / h[i]
                - (self.tsv[i] - self.tsv[i - 1]) / h[i - 1]
            )
        a[n - 1] = 2.0 * h[n - 2]
        d[n - 1] = 6.0 * (
            sigma_asym_slope(self.tk[n - 1])
            - (self.tsv[n - 1] - self.tsv[n - 2]) / h[n - 2]
        )
        for i in range(1, n):
            var w = h[i - 1] / a[i - 1]
            a[i] -= w * h[i - 1]
            d[i] -= w * d[i - 1]
        self.tsm = zeros(n)
        self.tsm[n - 1] = d[n - 1] / a[n - 1]
        var i = n - 2
        while i >= 0:
            self.tsm[i] = (d[i] - h[i] * self.tsm[i + 1]) / a[i]
            i -= 1

    def _node_at(self, n: Int, r: Int, ly: Float64) -> Float64:
        """Table of node n, role r at ln y, linear between the bin centres
        (0 outside)."""
        var u = (ly - TPP_LY0) / TPP_DLY - 0.5
        var b = to_int(c_floor(u))
        var t = u - Float64(b)
        return (1.0 - t) * self._node(n, r, b) + t * self._node(n, r, b + 1)

    def _fit_asymptotics(mut self, k_fit: Float64 = K_FIT):
        """Least-squares fits of the asymptotic laws over the nodes k >= k_fit
        (see K_FIT). tests/literature_checks/triplet/extrapolation.py uses
        other values, through tests/harness/triplet_mojo.mojo."""
        var n0 = 0
        while n0 < len(self.tk) and self.tk[n0] < k_fit:
            n0 += 1
        var N = len(self.tk)
        self.nby = to_int(c_floor((c_log(Y_SW) - TPP_LY0) / TPP_DLY)) + 1
        self.nlx = to_int(c_floor(-c_log(X_LO) / TPP_DLY)) + 1
        self.eg0 = zeros(N_ROLES * self.nby)
        self.eg1 = zeros(N_ROLES * self.nby)
        self.eA = zeros(N_ROLES * self.nlx)
        self.eB = zeros(N_ROLES * self.nlx)
        self.ea0 = zeros(N_ROLES)
        self.eb0 = zeros(N_ROLES)

        # straight-line fits v = c0 + c1 z over the nodes
        @always_inline
        def line(z: Vec, v: Vec) -> Tuple[Float64, Float64]:
            var m = Float64(len(z))
            var sz = 0.0
            var sv = 0.0
            var szz = 0.0
            var szv = 0.0
            for i in range(len(z)):
                sz += z[i]
                sv += v[i]
                szz += z[i] * z[i]
                szv += z[i] * v[i]
            var c1 = (m * szv - sz * sv) / (m * szz - sz * sz)
            return ((sv - c1 * sz) / m, c1)

        for r in range(N_ROLES):
            for b in range(self.nby):
                var z = Vec()
                var v = Vec()
                for n in range(n0, N):
                    z.append(1.0 / c_log(self.tk[n]))
                    v.append(self._node(n, r, b))
                var c: Tuple[Float64, Float64] = line(z, v)
                self.eg0[r * self.nby + b] = c[0]
                self.eg1[r * self.nby + b] = c[1]
            for j in range(self.nlx):
                var lx = c_log(X_LO) + Float64(j) * TPP_DLY
                var z = Vec()
                var v = Vec()
                for n in range(n0, N):
                    var lk = c_log(self.tk[n])
                    z.append(lk)
                    v.append(self.tk[n] * self._node_at(n, r, lx + lk))
                var c: Tuple[Float64, Float64] = line(z, v)
                self.eB[r * self.nlx + j] = c[0]
                self.eA[r * self.nlx + j] = c[1]
            var lx0 = c_log(X_LO)
            self.ea0[r] = self.eA[r * self.nlx] * X_LO
            self.eb0[r] = self.eB[r * self.nlx] * X_LO - self.ea0[r] * lx0

    def sigma(self, k: Float64) -> Float64:
        """sigma(k) / (alpha r_e^2)."""
        if k <= 4.0:
            return 0.0
        if k >= K_ASYM:
            return sigma_asym(k)
        if k <= self.tk[0]:
            return sigma_threshold(k)
        var n = 0
        while n + 2 < self.nsp and self.tk[n + 1] < k:
            n += 1
        # cubic spline of ln sigma in ln(k - 4) (sigma ~ (k - 4)^2 at threshold)
        var h = self.tsu[n + 1] - self.tsu[n]
        var t = (c_log(k - 4.0) - self.tsu[n]) / h
        var s = 1.0 - t
        return c_exp(
            s * self.tsv[n]
            + t * self.tsv[n + 1]
            + h
            * h
            / 6.0
            * (
                (s * s * s - s) * self.tsm[n]
                + (t * t * t - t) * self.tsm[n + 1]
            )
        )

    def _node(self, n: Int, r: Int, b: Int) -> Float64:
        var nb = tpp_nbins(self.tk[n])
        if b < 0 or b >= nb:
            return 0.0
        return self.tdist[self.toff[n] + r * nb + b]

    def distributions(self, k: Float64, mut out: Vec):
        """dN/dln y of the three products for photon energy k, normalised to
        one per interaction: out[r * nb + b], nb = tpp_nbins(k). y = x k for
        the positron and the lower-energy electron, y = xi k for the recoil.
        """
        var nb = tpp_nbins(k)
        var N = len(self.tk)
        out = zeros(N_ROLES * nb)
        if k <= self.tk[N - 1]:
            var n = 0
            while n + 2 < N and self.tk[n + 1] < k:
                n += 1
            var t = 0.0
            if k > self.tk[n]:
                t = c_log(k / self.tk[n]) / c_log(self.tk[n + 1] / self.tk[n])
            t = min(t, 1.0)
            for r in range(N_ROLES):
                for b in range(nb):
                    out[r * nb + b] = (1.0 - t) * self._node(
                        n, r, b
                    ) + t * self._node(n + 1, r, b)
        else:
            # above the last node: the fitted asymptotic laws (_fit_asymptotics)
            var lk = c_log(k)
            for r in range(N_ROLES):
                for b in range(nb):
                    var ly = TPP_LY0 + (Float64(b) + 0.5) * TPP_DLY
                    var v: Float64
                    if b < self.nby:
                        v = (
                            self.eg0[r * self.nby + b]
                            + self.eg1[r * self.nby + b] / lk
                        )
                    else:
                        var u = (ly - lk - c_log(X_LO)) / TPP_DLY
                        if u >= 0.0:
                            var j = min(to_int(c_floor(u)), self.nlx - 2)
                            var t = u - Float64(j)
                            var o = r * self.nlx + j
                            var A = (1.0 - t) * self.eA[o] + t * self.eA[o + 1]
                            var B = (1.0 - t) * self.eB[o] + t * self.eB[o + 1]
                            v = (A * lk + B) / k
                        else:
                            v = (self.ea0[r] * ly + self.eb0[r]) * c_exp(-ly)
                    out[r * nb + b] = max(v, 0.0)
        # interpolation between nodes can leave a little weight beyond the
        # kinematic range of this k (x > 1, or xi >= 1 for the recoil): remove
        # it, then normalise
        var lk = c_log(k)
        for r in range(N_ROLES):
            var s = 0.0
            for b in range(nb):
                var ly = TPP_LY0 + (Float64(b) + 0.5) * TPP_DLY
                if ly >= lk:
                    out[r * nb + b] = 0.0
                s += out[r * nb + b]
            if s > 0.0:
                for b in range(nb):
                    out[r * nb + b] /= s * TPP_DLY

    # ------------------------------------------------------------------
    # kernels
    # ------------------------------------------------------------------
    @staticmethod
    def _deposit(
        mut I: Vec, amount: Float64, x: Float64, dl: Float64, e1: Float64
    ):
        """Share `amount` products at the fraction x between the two
        neighbouring grid offsets, keeping their number and energy."""
        var u = -c_log(x) / dl
        var o = to_int(c_floor(u))
        if o < 0:
            o = 0
        if o + 1 >= len(I):
            # not reached (the rows cover the kinematic range): keep the
            # number at the last offset
            I[len(I) - 1] += amount
            return
        var xh = c_exp(-Float64(o) * dl)
        var xl = xh * e1
        var w = (xh - x) / (xh - xl)
        I[o] += amount * (1.0 - w)
        I[o + 1] += amount * w

    def initialize_kernels(mut self, dat: AM3Arrays) raises:
        if self.verbose:
            print("computing triplet pair production kernels")
        self._parse_tables()
        var dl = dat.dlnE
        var e1 = c_exp(-dl)
        var nl = dat.nbins_lep
        # P = exp(m dl): threshold P = 2 (k = 2P = 4)
        self.m0 = to_int(c_floor(c_log(2.0) / dl)) + 1
        var nq = max(nl - self.m0, 0)
        self.R = zeros(nq)
        self.Lxi = zeros(nq)
        self.nd = zerosi(nq)
        self.woff = zerosi(nq + 1)
        var tot = 0
        for q in range(nq):
            # offsets down to the lowest table bin at k = 2P,
            # x = e^TPP_LY0 / (2P)
            var P = c_exp(Float64(self.m0 + q) * dl)
            self.nd[q] = to_int((c_log(2.0 * P) - TPP_LY0) / dl) + 3
            self.woff[q] = tot
            tot += (self.nd[q] + BLK - 1) // BLK * BLK
        self.woff[nq] = tot
        # (nd grows with q: the rows reaching offset b BLK are q >= qblk[b])
        var ndmax0 = self.nd[nq - 1] if nq > 0 else 0
        self.qblk = zerosi((ndmax0 + BLK - 1) // BLK + 1)
        var qb = 0
        for b in range(len(self.qblk)):
            while qb < nq and self.nd[qb] <= b * BLK:
                qb += 1
            self.qblk[b] = qb
        for r in range(N_KR):
            self.W[r] = zeros(tot)
        var ndmax = self.nd[nq - 1] if nq > 0 else 1
        var I = List[Vec]()
        for _ in range(N_KR):
            I.append(zeros(ndmax + 2))
        var gx = Vec()
        var gw = Vec()
        gauss_legendre(NGL, gx, gw)
        var rc = 0.0
        var dist = Vec()
        # breakpoints of the integrand in ln k: the table nodes (kinks of the
        # distributions, which are linear between nodes) and K_ASYM
        var brk = Vec()
        for n in range(len(self.tk)):
            brk.append(c_log(self.tk[n]))
        brk.append(c_log(K_ASYM))
        sort(brk)
        var lnk_lo = c_log(4.0)
        for q in range(nq):
            var m = self.m0 + q
            var lnk_hi = c_log(2.0) + Float64(m) * dl
            # slab [lnk_lo, lnk_hi] of the k integral, split at the breakpoints
            # inside it, with NGL Gauss-Legendre points in ln k per piece
            var cuts = Vec()
            cuts.append(lnk_lo)
            for b in brk:
                if b > lnk_lo and b < lnk_hi:
                    cuts.append(b)
            cuts.append(lnk_hi)
            for p in range(len(cuts) - 1):
                var a0 = cuts[p]
                var a1 = cuts[p + 1]
                for g in range(NGL):
                    var lnk = 0.5 * (a1 + a0) + 0.5 * (a1 - a0) * gx[g]
                    var k = c_exp(lnk)
                    var c = (
                        0.5
                        * (a1 - a0)
                        * gw[g]
                        * k
                        * k
                        * self.sigma(k)
                        * SIG_UNIT
                    )
                    if c <= 0.0:
                        continue
                    rc += c
                    self.distributions(k, dist)
                    var nb = tpp_nbins(k)
                    for r in range(N_KR):
                        for b in range(nb):
                            var f = dist[r * nb + b]
                            if f <= 0.0:
                                continue
                            var y = c_exp(
                                TPP_LY0 + (Float64(b) + 0.5) * TPP_DLY
                            )
                            Self._deposit(I[r], c * f * TPP_DLY, y / k, dl, e1)
            lnk_lo = lnk_hi
            var norm = 1.0 / (2.0 * c_exp(2.0 * Float64(m) * dl))
            self.R[q] = rc * norm
            var wo = self.woff[q]
            var nd = self.nd[q]
            for r in range(N_KR):
                for o in range(nd):
                    self.W[r][wo + o] = I[r][o] * norm
            # the energy the pair takes, which the lepton loses: exactly that
            # of the deposits
            var epair = 0.0
            for o in range(nd):
                var xo = c_exp(-Float64(o) * dl)
                epair += (
                    self.W[ROLE_POS][wo + o] + self.W[ROLE_LOW][wo + o]
                ) * xo
            self.Lxi[q] = epair
        # per-step rows: target bin i has offsets up to those of the largest
        # P (eps <= 1: P <= gamma)
        self.zoff = zerosi(nl + 1)
        self.nz = zerosi(nl)
        var zt = 0
        for i in range(nl):
            var q = i - self.m0
            self.nz[i] = self.nd[min(q, nq - 1)] if q >= 0 and nq > 0 else 0
            self.zoff[i] = zt
            zt += (self.nz[i] + BLK - 1) // BLK * BLK
        self.zoff[nl] = zt
        for r in range(N_KR):
            self.Z[r] = zeros(zt)
        self.nlep = zeros(nl)
        self.valid = True
        self.step_valid = False

    # ------------------------------------------------------------------
    # per step
    # ------------------------------------------------------------------
    def Calc_Triplet(mut self, dat: AM3Arrays, pho: Vec, ele: Vec, pos: Vec):
        """Energy-loss rates of the e+- and photon-convolved pair rows
        (TripletJob) for the photon field `pho` at the start of the step.
        Rows are computed only where the e- or e+ totals `ele`, `pos` are not
        zero (`pos` empty with merged e+-)."""
        var nl = dat.nbins_lep
        var ng = dat.nbins_pho
        # photons up to eps = 1 (m_e c^2): the soft-photon limit
        var j0 = dat.ind_min_energy_pho
        var two = len(pos) == nl
        var job = TripletJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            pd(pho),
            pd(ele),
            pd(pos) if two else pd(ele),
            two,
            nl,
            j0,
            min(j0, ng - 1),
            dat.dlnE,
        )
        parallel_run[triplet_rows_work](job, self.nthreads)
        self.step_valid = True

    def photon_sink(mut self, dat: AM3Arrays, ele: Vec, pos: Vec):
        """Photon sink for eps <= m_e c^2 (`pos` empty with merged e+-):
            alpha_g(eps_j) = sum_i (n_e- + n_e+)(i) dlnE R(gamma_i eps_j)."""
        var nl = dat.nbins_lep
        var ng = dat.nbins_pho
        var j0 = dat.ind_min_energy_pho
        var dl = dat.dlnE
        for i in range(nl):
            self.nlep[i] = ele[i] + (pos[i] if len(pos) == nl else 0.0)
        for j in range(ng):
            var a = 0.0
            if j <= j0:
                for i in range(max(0, self.m0 - j + j0), nl):
                    var n = self.nlep[i]
                    if n != 0.0:
                        a += n * dl * self.R[i + j - j0 - self.m0]
            self.alpha_g[j] = a

    def add_pair_sources(
        self, de: FPtr, dp: FPtr, ne: Vec, np_: Vec, factor: Float64
    ):
        """Add the production rates per ln E of the pairs made by the e- (ne)
        and e+ (np_). An e- gives an e+ (ROLE_POS) and a lower-energy e-
        (ROLE_LOW), an e+ the reverse:
        de[l] += factor sum_i (ne[i] Z_low[i][i - l] + np_[i] Z_pos[i][i - l]),
        and dp likewise with the roles exchanged. With np_ empty (merged e+-),
        de += factor sum_i ne[i] (Z_low + Z_pos) and dp is unused. Products
        below the grid go into bin 0."""
        if not self.step_valid or factor == 0.0:
            return
        var two = len(np_) == len(ne)
        var zl0 = pd(self.Z[ROLE_LOW])
        var zp0 = pd(self.Z[ROLE_POS])
        for i in range(len(ne)):
            var ve = factor * ne[i]
            var vp = factor * np_[i] if two else 0.0
            if ve == 0.0 and vp == 0.0:
                continue
            var zl = zl0.unsafe_offset(self.zoff[i])
            var zp = zp0.unsafe_offset(self.zoff[i])
            var nz = self.nz[i]
            # offsets o <= i land on the bins l = i - o (WD at a time,
            # reversed), the rest in bin 0
            var no = min(nz, i + 1)
            var o = 0
            var sve = SIMD[DType.float64, WD](ve)
            var svp = SIMD[DType.float64, WD](vp)
            while o + WD <= no:
                var a = zl.unsafe_offset(o).unsafe_load[width=WD]().reversed()
                var b = zp.unsafe_offset(o).unsafe_load[width=WD]().reversed()
                var d = de.unsafe_offset(i - o - WD + 1)
                if two:
                    d.unsafe_store(
                        d.unsafe_load[width=WD]() + sve * a + svp * b
                    )
                    var dq = dp.unsafe_offset(i - o - WD + 1)
                    dq.unsafe_store(
                        dq.unsafe_load[width=WD]() + sve * b + svp * a
                    )
                else:
                    d.unsafe_store(d.unsafe_load[width=WD]() + sve * (a + b))
                o += WD
            for oo in range(o, nz):
                var l = max(i - oo, 0)
                var a = zl[unsafe_offset=oo]
                var b = zp[unsafe_offset=oo]
                if two:
                    de[unsafe_offset=l] += ve * a + vp * b
                    dp[unsafe_offset=l] += ve * b + vp * a
                else:
                    de[unsafe_offset=l] += ve * (a + b)
