"""Port of SolverAlgorithm (Solver.cc): time-step solver for one species.

Differences from the C++ code that do not change results:
  * work arrays are allocated once per solver;
  * GSL splines and tridiagonal solves are ported in interp.mojo;
  * the IC cooling kernel is photon-major, for SIMD over particles;
  * photon bins beyond the photon array (read out of bounds by the C++ code
    for hadrons) count as empty.
"""

from std.builtin.globals import global_constant
from std.math import ceil, fma, sqrt
from std.utils.numerics import nextafter
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.interp import LinearSpline, solve_tridiag
from am3_mojo.utils.libm import *
from am3_mojo.utils.parallel import parallel_run

# Taylor coefficients of the inverse-Compton cooling functions (Jones 1965,
# Table I, as exact fractions): f1 from z^3, f2 from z^2, both to z^10
comptime _JONES_F1: Array[Float64, 8] = [
    8.0 / 9.0,
    -14.0 / 5.0,
    196.0 / 25.0,
    -6464.0 / 315.0,
    7520.0 / 147.0,
    -2584.0 / 21.0,
    116704.0 / 405.0,
    -1636352.0 / 2475.0,
]
comptime _JONES_F2: Array[Float64, 9] = [
    4.0 / 3.0,
    -128.0 / 45.0,
    7.0,
    -8812.0 / 525.0,
    4112.0 / 105.0,
    -4384.0 / 49.0,
    63356.0 / 315.0,
    -397792.0 / 891.0,
    2427904.0 / 2475.0,
]

# 4-point Gauss-Legendre rule on [0, 1]: nodes and weights (the inflow of the
# regime-1 and regime-2 windows, SolverAlgorithm.crossing_count)
comptime _GL4_T: Array[Float64, 4] = [
    0.06943184420297371,
    0.33000947820757187,
    0.66999052179242813,
    0.93056815579702629,
]
comptime _GL4_W: Array[Float64, 4] = [
    0.17392742256872693,
    0.32607257743127307,
    0.32607257743127307,
    0.17392742256872693,
]


@inline(.always)
def _phi1(u: Float64) -> Float64:
    """(e^u - 1) / u, 1 at u = 0."""
    return 1.0 if abs(u) < 1e-12 else c_expm1(u) / u


@inline(.never)
def _zero(p: FPtr, n: Int):
    for i in range(n):
        p[unsafe_offset=i] = 0.0


# regime 3's source integral (simple_integral_q): moments of e^(-D v) on [0, 1]
@inline(.always)
def _expmom0(D: Float64) -> Float64:
    """int_0^1 e^(-D v) dv = (1 - e^-D) / D."""
    return 1.0 if D < 1e-300 else -c_expm1(-D) / D


@inline(.always)
def _expmom(D: Float64, k: Int) -> Float64:
    """int_0^1 v^(k - 1) e^(-D v) dv for k = 2, 3 (series below D = 0.5)."""
    if D < 0.5:
        var s = 0.0
        var f = 1.0
        for n in range(14):
            s += f / Float64(n + k)
            f *= -D / Float64(n + 1)
        return s
    var e = c_exp(-D)
    if k == 2:
        return (1.0 - (1.0 + D) * e) / (D * D)
    return (2.0 - (D * D + 2.0 * D + 2.0) * e) / (D * D * D)


@inline(.always)
def _source_piece(
    K: Float64,
    h: Float64,
    l0: Float64,
    l1: Float64,
    q0: Float64,
    q1: Float64,
    mut D: Float64,
) -> Float64:
    """e^-K int_0^h q e^(-int_0^x' k) dx' over a piece of width h on which ln k
    goes linearly from l0 to l1 and q from q0 to q1 (k = alpha / |A|); sets
    D = int_0^h k. In t = int k, the integrand is g(t) e^-t, with g = q / k
    approximated by the quadratic through the ends and the middle (exact for
    constant k and linear q).
    """
    var dd = l1 - l0
    var ra = 1.0  # kbar / k(0), kbar = D / h
    var rb = 1.0  # kbar / k(h)
    var rm = 1.0  # kbar / k(h / 2)
    var vm = 0.5  # t(h / 2) / D
    if abs(dd) >= 1e-12:
        var w = c_expm1(0.5 * dd)
        ra = w * (w + 2.0) / dd
        rb = ra / ((1.0 + w) * (1.0 + w))
        rm = ra / (1.0 + w)
        vm = w / (dd * ra)
    D = h * c_exp(l0) * ra
    var g0 = q0 * ra
    var g1 = q1 * rb
    var gm = 0.5 * (q0 + q1) * rm
    var c = (gm - g0 - (g1 - g0) * vm) / (vm * (vm - 1.0))
    var m1 = _expmom(D, 2)
    return (
        c_exp(-K)
        * h
        * (g0 * _expmom0(D) + (g1 - g0) * m1 + c * (_expmom(D, 3) - m1))
    )


@inline(.always)
def _source_cell(
    h: Float64,
    lka: Float64,
    lkb: Float64,
    qa: Float64,
    qb: Float64,
    mut D: Float64,
) -> Float64:
    """_source_piece over a whole cell of width h (ln k from lka to lkb, q from
    qa to qb), summed over equal pieces (at most 400) in which ln k changes by
    at most 0.25 and the attenuation exponent by at most 2 on average."""
    var d = lkb - lka
    var dc = h * c_exp(lka) * (1.0 if d == 0.0 else c_expm1(d) / d)
    var mm = ceil(max(abs(d) / 0.25, dc / 2.0))
    var m = 1 if not (mm >= 1.0) else (400 if mm > 400.0 else Int(mm))
    var K = 0.0
    var integral = 0.0
    for p in range(m):
        if K >= 745.0:
            break
        var f0 = Float64(p) / Float64(m)
        var f1 = Float64(p + 1) / Float64(m)
        var Dp = 0.0
        integral += _source_piece(
            K,
            h / Float64(m),
            lka + d * f0,
            lka + d * f1,
            qa + (qb - qa) * f0,
            qa + (qb - qa) * f1,
            Dp,
        )
        K += Dp
    D = K
    return integral


struct ICKerJob(Movable):
    var S: Pointer[SolverAlgorithm, MutUntrackedOrigin]
    var eph: FPtr  # photon energies [GeV]
    var failed: Bool

    def __init__(
        out self, S: Pointer[SolverAlgorithm, MutUntrackedOrigin], eph: FPtr
    ):
        self.S = S
        self.eph = eph
        self.failed = False


def ic_ker_work(job: Pointer[ICKerJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref S = job[].S[]
    var mass = S.MASS_R * MASS_E
    var NB = S.N_BINS
    var c_light = C_LIGHT * S.T_UNIT_S / S.L_UNIT_CM
    var sigma_t = SIGMA_T / S.L_UNIT_CM / S.L_UNIT_CM
    var coeff = 3.0 / 16.0 * c_light * sigma_t * MASS_E * MASS_E
    var kp = pd(S.ker_ic_t)
    var i = t
    while i < NB:
        var gm = c_exp(S.DELTA_X * Float64(i)) * S.X_UNIT_GEV / mass
        for j in range(S.N_PHOT):
            var eph = job[].eph[unsafe_offset=j]
            try:
                kp[unsafe_offset=j * NB + i] = (
                    coeff * S.Fic(gm, eph) / eph / eph / gm / gm / gm
                )
            except:
                job[].failed = True
        i += nt


struct SolverAlgorithm(Copyable, Movable):
    var particle_name: String
    var N_BINS: Int
    var PHOT_GRID_ORIGIN: Int
    var DELTA_X: Float64
    var X_UNIT_GEV: Float64
    var L_UNIT_CM: Float64
    var T_UNIT_S: Float64
    var MASS_R: Float64
    var delta_t: Float64
    var process_sy: Bool
    var process_quantum_syn_cooling: Bool
    var process_ic: Bool
    var process_decay: Bool
    var threshold_cool_dom: Float64
    var threshold_esc_dom: Float64
    var threshold_matrix: Float64
    var simpleqintegration: Bool
    var verbose: Bool

    var alpha_decay: Vec
    var gam: Vec  # Lorentz factor of every bin
    var Asy: Vec
    var Aic: Vec
    var cooling_on: Bool
    var ker_status: Int
    # IC cooling kernel [photon j][particle i] (particle index contiguous)
    var ker_ic_t: Vec
    var N_PHOT: Int

    # work buffers (sized N_BINS + 40)
    var w_a: Vec
    var w_A: Vec
    var w_q: Vec
    var w_n: Vec
    var w_out: Vec
    var w_x: Vec
    var w_mivA: Vec
    var w_aln: Vec
    var w_q1: Vec
    var w_q2: Vec
    var w_q3: Vec
    var w_b: Vec
    var w_y: Vec
    var w_K: Vec
    var m_DA: Vec
    var m_DB: Vec
    var m_DD: Vec
    var m_PL: Vec
    var m_PD: Vec
    var m_PR: Vec
    var m_TS: Vec
    var m_TD: Vec
    var m_TB: Vec
    var m_TQ: Vec
    var m_TN: Vec
    var m_alpha: Vec
    var m_z: Vec
    var e_a: Vec
    var e_A: Vec
    var e_res: Vec

    def __init__(
        out self,
        p_name: String,
        NB: Int,
        PGO: Int,
        DX: Float64,
        XUG: Float64,
        LUC: Float64,
        TUS: Float64,
        HLS: Float64,
        MR: Float64,
        Dt: Float64,
        process_sy: Bool,
        process_quantum_syn_cooling: Bool,
        process_ic: Bool,
        process_decay: Bool,
        threshold_cool_dom: Float64,
        threshold_esc_dom: Float64,
        threshold_matrix: Float64,
        simpleqintegration: Bool,
        verbose: Bool,
    ):
        self.particle_name = p_name
        self.N_BINS = NB
        self.PHOT_GRID_ORIGIN = PGO
        self.DELTA_X = DX
        self.X_UNIT_GEV = XUG
        self.L_UNIT_CM = LUC
        self.T_UNIT_S = TUS
        self.MASS_R = MR
        self.delta_t = Dt
        self.process_sy = process_sy
        self.process_quantum_syn_cooling = process_quantum_syn_cooling
        self.process_ic = process_ic
        self.process_decay = process_decay
        self.threshold_cool_dom = threshold_cool_dom
        self.threshold_esc_dom = threshold_esc_dom
        self.threshold_matrix = threshold_matrix
        self.simpleqintegration = simpleqintegration
        self.verbose = verbose
        self.alpha_decay = zeros(NB)
        self.gam = zeros(NB)
        self.Asy = zeros(NB)
        self.Aic = zeros(NB)
        self.cooling_on = False
        self.ker_status = 0
        self.N_PHOT = NB - PGO
        self.ker_ic_t = Vec()
        var mass = MR * MASS_E
        for i in range(NB):
            var gamma = c_exp(DX * Float64(i)) * XUG / mass
            self.gam[i] = gamma
            self.alpha_decay[i] = TUS / HLS / gamma
        var nw = NB + 40
        self.w_a = zeros(nw)
        self.w_A = zeros(nw)
        self.w_q = zeros(nw)
        self.w_n = zeros(nw)
        self.w_out = zeros(nw)
        self.w_x = zeros(nw)
        self.w_mivA = zeros(nw)
        self.w_aln = zeros(nw)
        self.w_q1 = zeros(nw)
        self.w_q2 = zeros(nw)
        self.w_q3 = zeros(nw)
        self.w_b = zeros(nw)
        self.w_y = zeros(nw)
        self.w_K = zeros(nw)
        self.m_DA = zeros(nw)
        self.m_DB = zeros(nw)
        self.m_DD = zeros(nw)
        self.m_PL = zeros(nw)
        self.m_PD = zeros(nw)
        self.m_PR = zeros(nw)
        self.m_TS = zeros(nw)
        self.m_TD = zeros(nw)
        self.m_TB = zeros(nw)
        self.m_TQ = zeros(nw)
        self.m_TN = zeros(nw)
        self.m_alpha = zeros(nw)
        self.m_z = zeros(nw)
        self.e_a = zeros(NB)
        self.e_A = zeros(NB)
        self.e_res = zeros(NB)

    def updateDeltaT(mut self, dt_sec: Float64):
        self.delta_t = dt_sec / self.T_UNIT_S

    def same_ic_kernel(self, other: SolverAlgorithm) -> Bool:
        """True if `other` computes a bit-identical IC cooling kernel."""
        return (
            self.N_BINS == other.N_BINS
            and self.PHOT_GRID_ORIGIN == other.PHOT_GRID_ORIGIN
            and self.DELTA_X == other.DELTA_X
            and self.X_UNIT_GEV == other.X_UNIT_GEV
            and self.L_UNIT_CM == other.L_UNIT_CM
            and self.T_UNIT_S == other.T_UNIT_S
            and self.MASS_R == other.MASS_R
        )

    def same_cooling(self, other: SolverAlgorithm) -> Bool:
        """True if `other` computes bit-identical synchrotron + IC cooling rates.
        """
        return (
            self.same_ic_kernel(other)
            and self.process_sy == other.process_sy
            and self.process_ic == other.process_ic
            and self.process_quantum_syn_cooling
            == other.process_quantum_syn_cooling
        )

    # ------------------------------------------------------------------
    # cooling rates (synchrotron + inverse Compton)
    # ------------------------------------------------------------------
    def start_cooling(
        mut self, b_gauss: Float64, n_phot: Vec, nthreads: Int
    ) raises -> Bool:
        """Whether this step has cooling (`cooling_on`), and the IC cooling
        kernel if it is still missing: what compute_syn_ic needs before it
        computes Asy and Aic (run_cooling). The rates depend only on the
        field, the target photons and the species, so solvers of one species
        share them (`share_cooling`)."""
        var sum_nphot = 0.0
        var npp = pd(n_phot)
        for i in range(len(n_phot)):
            sum_nphot += npp[unsafe_offset=i]
        self.cooling_on = (self.process_sy and b_gauss > 1.0e-12) or (
            self.process_ic and sum_nphot > 1e-100
        )
        if self.cooling_on and self.process_ic and self.ker_status == 0:
            self.compute_ic_ker(nthreads)
        return self.cooling_on

    def share_cooling(mut self, Asy: Vec, Aic: Vec, cooling_on: Bool):
        """Take the cooling rates computed by a solver of the same species
        (AM3 recomputes them in each of its 6 pion and 12 muon solver calls).
        """
        self.cooling_on = cooling_on
        if cooling_on:
            for i in range(self.N_BINS):
                self.Asy[i] = Asy[i]
                self.Aic[i] = Aic[i]

    # ------------------------------------------------------------------
    # evolve
    # ------------------------------------------------------------------
    def evolve(
        mut self, a_esc_in: Vec, A_cool_in: Vec, q_inj: Vec, mut n: Vec
    ) raises:
        """Advance density `n` by one time step in place, using the escape
        rates `a_esc_in`, cooling rates `A_cool_in` (plus the rates of the last
        `compute_syn_ic`/`share_cooling`) and the source `q_inj`."""
        var NB = self.N_BINS
        var a_esc = pd(self.e_a)
        var A_cool = pd(self.e_A)
        var ain = pd(a_esc_in)
        var Ain = pd(A_cool_in)
        var Asy = pd(self.Asy)
        var Aic = pd(self.Aic)
        for i in range(NB):
            a_esc[unsafe_offset=i] = ain[unsafe_offset=i]
            A_cool[unsafe_offset=i] = Ain[unsafe_offset=i]
        if self.cooling_on:
            for i in range(NB):
                A_cool[unsafe_offset=i] += (
                    Asy[unsafe_offset=i] + Aic[unsafe_offset=i]
                )

        if self.process_decay:
            if self.verbose:
                print("added decay term")
            for i in range(NB):
                a_esc[unsafe_offset=i] += self.alpha_decay[i]

        # bin 0 loses its rates, as in AM3. The top bin keeps them, though the
        # solver leaves it out: they decide where the particles of the highest
        # evolved bin come from (AM3 zeroed them too, so its semi-analytic
        # regimes evolved that bin without cooling: an AM3 bug, hidden while
        # the top bins were emptied)
        a_esc[unsafe_offset=0] = 0.0
        A_cool[unsafe_offset=0] = 0.0

        var qp = pd(q_inj)
        var n0p = pd(n)
        var Lm3 = c_pow(self.L_UNIT_CM, -3.0)
        var sum_qinj = 0.0
        for i in range(NB):
            sum_qinj += qp[unsafe_offset=i] * Lm3 / self.T_UNIT_S
        var sum_esc = 0.0
        for i in range(NB):
            sum_esc += a_esc[unsafe_offset=i] / self.T_UNIT_S
        var sum_A = 0.0
        for i in range(NB):
            sum_A += abs(A_cool[unsafe_offset=i]) / self.T_UNIT_S
        var sum_n = 0.0
        for i in range(NB):
            sum_n += n0p[unsafe_offset=i] * Lm3

        var nothing = 1e-100
        if (sum_qinj > nothing) or (
            sum_qinj < nothing and sum_n > nothing and sum_A + sum_esc > nothing
        ):
            var res = pd(self.e_res)
            try:
                self.solve_next_step(a_esc, A_cool, qp, n0p, res)
            except e:
                print(
                    (
                        "Something went wrong in the solver - don't trust the"
                        " results!"
                    ),
                    " Rerun your code with verbosity=1 to debug",
                )
                print(e)
                _zero(res, NB)
            for i in range(NB):
                var v = res[unsafe_offset=i]
                n0p[unsafe_offset=i] = v if v > 0.0 else 0.0
        elif self.verbose:
            print(
                "In SolverAlgorithm ",
                self.particle_name,
                " : sum_A = ",
                sum_A,
                ", sum_esc = ",
                sum_esc,
                ", sum_qinj = ",
                sum_qinj,
                ", sum_n = ",
                sum_n,
                " does not fulfill condition for evolving. Leaving ",
                self.particle_name,
                " unchanged.",
                sep="",
            )

    def compute_syn_ic(
        mut self, b_gauss: Float64, n_phot: Vec, i0: Int, i1: Int
    ):
        """Asy and Aic of the bins [i0, i1), after start_cooling. Ranges that
        start at a multiple of WD (and end at one, or at N_BINS) give every
        bin the code path of the whole range, so they may be computed apart.
        """
        if self.verbose:
            print("add syn and ic")
        var T = self.T_UNIT_S
        var L = self.L_UNIT_CM
        var c_light = C_LIGHT * T / L
        var sigma_t = SIGMA_T / L / L
        var mass = self.MASS_R * MASS_E
        var NB = self.N_BINS
        var ub_Edimless = (
            b_gauss * b_gauss / (8.0 * PI * MASS_E / GEV_PER_ERG) * L * L * L
        )
        var k_synch = (
            (4.0 / 3.0)
            * c_light
            * sigma_t
            / self.MASS_R
            / self.MASS_R
            / self.MASS_R
            * ub_Edimless
        )
        var sy = fbool(self.process_sy)
        var Asy = pd(self.Asy)
        var Aic = pd(self.Aic)
        # quantum parameter chi = gamma B / B_cr(m), B_cr ∝ m^2: AM3's value /
        # MASS_R (AM3 divides by m only, overstating chi by m / m_e for the
        # hadrons and muons: an AM3 bug)
        var chi_g = 1.2 * c_pow(10.0, -8.0) * b_gauss / mass / c_pow(10.0, 9.0)
        for i in range(i0, i1):
            var gamma = self.gam[i]
            Asy[unsafe_offset=i] = -sy * k_synch * gamma
            if self.process_quantum_syn_cooling:
                var x = (chi_g * gamma) / self.MASS_R
                Asy[unsafe_offset=i] *= 1.0 / c_pow(
                    1.0 + sqrt(2.0) * c_pow(x, 2.0 / 3.0), 2.0
                )
            Aic[unsafe_offset=i] = 0.0
        if self.process_ic:
            # Aic[i] = -DX * sum_j ker[i][j] * n_phot[j], SIMD over i
            var nj = min(self.N_PHOT, len(n_phot))
            var kp = pd(self.ker_ic_t)
            var npp = pd(n_phot)
            var DX = self.DELTA_X
            var i = i0
            while i + WD <= i1:
                var acc = SIMD[DType.float64, WD](0)
                for j in range(nj):
                    var k = kp.unsafe_offset(j * NB + i).unsafe_load[width=WD]()
                    acc += k * npp[unsafe_offset=j]
                Aic.unsafe_offset(i).unsafe_store(-DX * acc)
                i += WD
            while i < i1:
                var acc = 0.0
                for j in range(nj):
                    acc += kp[unsafe_offset=j * NB + i] * npp[unsafe_offset=j]
                Aic[unsafe_offset=i] = -DX * acc
                i += 1

    def compute_ic_ker(mut self, nthreads: Int) raises:
        """The IC cooling kernel, on up to nthreads threads (the particle bins
        i = t, t + nt, ...)."""
        if self.verbose:
            print("calc ic kernel")
        self.ker_ic_t = zeros(self.N_BINS * self.N_PHOT)
        var eph = zeros(self.N_PHOT)
        for j in range(self.N_PHOT):
            eph[j] = (
                c_exp(Float64(self.PHOT_GRID_ORIGIN + j) * self.DELTA_X)
                * self.X_UNIT_GEV
            )
        var job = ICKerJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](), pd(eph)
        )
        parallel_run[ic_ker_work](job, max(1, min(nthreads, self.N_BINS)))
        _ = eph^  # (read by the workers through a pointer)
        if job.failed:
            raise Error("error in SolverAlgorithm.compute_ic_ker")
        self.ker_status = 1

    def Fic(self, gamma: Float64, Eph_GeV: Float64) raises -> Float64:
        """Inverse-Compton cooling kernel of a particle with Lorentz factor
        gamma on photons of energy Eph_GeV, F(z+) - F(z-) with
        z+- = (Eph / m) * (gamma + sqrt(gamma^2 - 1))^(+-1)."""
        var w = Eph_GeV / (self.MASS_R * MASS_E)
        var gamma_bar = gamma + sqrt(gamma * gamma - 1.0 + 1.0e-10)
        var za = w * gamma_bar
        var zb = w / gamma_bar
        var d1 = self.Fic_1(za) - self.Fic_1(zb)
        var d2 = self.Fic_2(za) - self.Fic_2(zb)
        var y = gamma * d1 - w * d2
        return y if y > 0.0 else 0.0

    def Fic_1(self, z: Float64) raises -> Float64:
        if z < 0.1:
            # Jones' series to z^10, 2e-5 at z = 0.1 (AM3 keeps three terms,
            # 2.4 % off there - an AM3 bug)
            var s = 0.0
            var k = 7
            while k >= 0:
                s = s * z + global_constant[_JONES_F1]()[k]
                k -= 1
            return z * z * z * s
        var p = 1.0 + 2.0 * z
        var num = 22.0 / 3.0 * z * z * z + 24.0 * z * z + 18.0 * z + 4.0
        return (
            (z + 6.0 + 3.0 / z) * c_log(p)
            - num / (p * p)
            - 2.0
            + 2.0 * self.polylog(-2.0 * z)
        )

    def Fic_2(self, z: Float64) raises -> Float64:
        if z < 0.1:
            # Jones' series to z^10, 2e-6 at z = 0.1 (AM3 keeps three terms,
            # 1.2 % off there - an AM3 bug)
            var s = 0.0
            var k = 8
            while k >= 0:
                s = s * z + global_constant[_JONES_F2]()[k]
                k -= 1
            return z * z * s
        var p = 1.0 + 2.0 * z
        var num = (
            22.0 / 3.0 * z * z * z
            + 28.0 * z * z
            + 103.0 / 3.0 * z
            + 17.0
            + 3.0 / z
        )
        return (
            (z + 31.0 / 6.0 + 5.0 / z + 1.5 / (z * z)) * c_log(p)
            - num / (p * p)
            - 2.0
            + self.polylog(-2.0 * z)
        )

    def polylog(self, z: Float64) raises -> Float64:
        if z >= 0.99:
            raise Error(
                "in function SolverAlgorithm::polylog_2, z>0.9 out of range\n"
            )
        elif z >= -0.1 and z < 0.9:
            return self.polylog_smallz(z)
        elif z < -0.1 and z > -20.0:
            return self.polylog_rk4(z)
        else:
            return self.polylog_largez(z)

    def polylog_smallz(self, z: Float64) -> Float64:
        """Li_2(z) = sum_k z^k / k^2, first six terms."""
        return z * (
            1.0
            + z
            * (
                1.0 / 4.0
                + z
                * (1.0 / 9.0 + z * (1.0 / 16.0 + z * (1.0 / 25.0 + z / 36.0)))
            )
        )

    def polylog_rk4(self, z: Float64) -> Float64:
        var x = 0.1
        var y = 0.0
        comptime N_INTG = 20
        var diff_x = (-z - 0.1) / Float64(N_INTG)
        # RK4 for dLi_2/dx = -ln(1 + x) / x (k2 = k3, the slope does not
        # depend on y)
        for _ in range(N_INTG):
            var xm = x + 0.5 * diff_x
            var k1 = -diff_x * c_log(1.0 + x) / x
            var k2 = -diff_x * c_log(1.0 + xm) / xm
            var k4 = -diff_x * c_log(1.0 + x + diff_x) / (x + diff_x)
            y += (k1 + 4.0 * k2 + k4) / 6.0
            x += diff_x
        return -0.097605235 + y

    def polylog_largez(self, z: Float64) -> Float64:
        """Asymptotic expansion of Li_2(z) for z -> -infinity."""
        var x = -z
        var lx = c_log(x)
        return (
            -0.5 * lx * lx
            - PI * PI / 6.0
            + 1.0 / x
            - 1.0 / (4.0 * x * x)
            + 1.0 / (9.0 * x * x * x)
        )

    # ------------------------------------------------------------------
    # regime selection
    # ------------------------------------------------------------------
    def solve_next_step(
        mut self, a_esc: FPtr, A_cool: FPtr, q_inj: FPtr, n0: FPtr, n1: FPtr
    ) raises:
        if self.verbose:
            print("solving next step for ", self.particle_name)
        var NB = self.N_BINS
        var n_min_regime = 5
        var imax = NB - 2
        var nqint = 0.0
        var threshold_inj_n0 = 1e-100 * c_pow(self.L_UNIT_CM, 3.0)
        var regime = 0
        var new_regime: Int
        # regime of the window above, solved before this one (0: none, or no
        # particles)
        var regime_above = 0
        # the highest bin with particles: the cells above it (regime 0) stay
        # empty, so the padding of semi-analytic windows and the inflow count
        # stop just above it
        var itop = -1
        var dt = self.delta_t
        for t in range(NB):
            n1[unsafe_offset=t] = 0.0
        var wa = pd(self.w_a)
        var wA = pd(self.w_A)
        var wq = pd(self.w_q)
        var wn = pd(self.w_n)
        var wo = pd(self.w_out)
        var i = imax
        while i > 0:
            nqint += n0[unsafe_offset=i] + q_inj[unsafe_offset=i] * dt
            if nqint > threshold_inj_n0:
                if itop < 0:
                    itop = i
                var Ai = A_cool[unsafe_offset=i]
                var ai = a_esc[unsafe_offset=i]
                if -Ai < self.threshold_esc_dom * ai:
                    new_regime = 1
                else:
                    if dt < -self.threshold_matrix * self.DELTA_X / Ai:
                        new_regime = 2
                    else:
                        if ai < -self.threshold_cool_dom * Ai:
                            new_regime = 4
                        else:
                            new_regime = 3
            else:
                new_regime = 0
            if self.verbose:
                print(
                    "regime at i=",
                    i,
                    " : ",
                    regime,
                    " ,new_regime :",
                    new_regime,
                    ", A[i]=",
                    A_cool[unsafe_offset=i],
                    " , a[i]=",
                    a_esc[unsafe_offset=i],
                    ", dt=",
                    dt,
                    ", q[i]=",
                    q_inj[unsafe_offset=i],
                    " , n0[i]=",
                    n0[unsafe_offset=i],
                    ", nqint=",
                    nqint,
                    sep="",
                )
            # the regime-0 window (empty bins at the top) ends at the first bin
            # with particles (AM3 held it to the minimum window length, at
            # least the top 6 bins, and dropped their particles: AM3 bug)
            if (
                regime != new_regime
                and (imax - i > n_min_regime or i < n_min_regime or regime == 0)
            ) or i == 1:
                var N = imax - i
                if self.verbose:
                    print(
                        "i=",
                        i,
                        ": solving ",
                        N,
                        " indices using regime ",
                        regime,
                        sep="",
                    )
                if regime == 1:
                    for k in range(N):
                        wa[unsafe_offset=k] = a_esc[unsafe_offset=i + 1 + k]
                        wq[unsafe_offset=k] = q_inj[unsafe_offset=i + 1 + k]
                        wn[unsafe_offset=k] = n0[unsafe_offset=i + 1 + k]
                    # the inflow within the step from a window of regime 2, 3
                    # or 4 above stays in the top cell, as regime 1 neglects
                    # cooling (AM3 dropped it: AM3 bug)
                    if regime_above >= 2:
                        wq[unsafe_offset=N - 1] += self.crossing_count(
                            a_esc, A_cool, q_inj, n0, imax, itop
                        ) / (self.DELTA_X * dt)
                    self.solve_steadystate_esc_inj(wa, wq, wn, wo, N)
                    for k in range(N):
                        n1[unsafe_offset=i + 1 + k] = wo[unsafe_offset=k]
                elif regime == 2:
                    # the window and Nb cells below it, with no flux through
                    # its top; the inflow from a window of regime 3 or 4 above
                    # goes to its top cell (regime 1 neglects cooling). (AM3
                    # padded the window with 10 cells of old densities and no
                    # inflow at their top, dropping the particles that cool
                    # in from further up: AM3 bug)
                    var Nb = 10
                    var M = N + Nb
                    for k in range(M):
                        wa[unsafe_offset=k] = 0.0
                        wA[unsafe_offset=k] = 0.0
                        wq[unsafe_offset=k] = 0.0
                        wn[unsafe_offset=k] = 0.0
                    for k in range(M):
                        var g = i + 1 - Nb + k
                        if g >= 0 and g < NB:
                            wa[unsafe_offset=k] = (
                                a_esc[unsafe_offset=g] + 1e-150
                            )
                            wA[unsafe_offset=k] = (
                                A_cool[unsafe_offset=g] - 1e-150
                            )
                            wq[unsafe_offset=k] = (
                                q_inj[unsafe_offset=g] + 1e-150
                            )
                            wn[unsafe_offset=k] = n0[unsafe_offset=g] + 1e-150
                    if regime_above >= 2:
                        wq[unsafe_offset=M - 1] += self.crossing_count(
                            a_esc, A_cool, q_inj, n0, imax, itop
                        ) / (self.DELTA_X * dt)
                    self.solve_matrix(wa, wA, wq, wn, wo, M)
                    for k in range(N):
                        var g = i + 1 + k
                        if g >= 0 and g < NB:
                            n1[unsafe_offset=g] = wo[unsafe_offset=k + Nb]
                elif regime == 3 or regime == 4:
                    # the padding above holds every cell from which particles
                    # cool into the window within the step (AM3 always took
                    # 10, dropping the particles from further up: AM3 bug)
                    var Nb = 10
                    var Na = self.cells_reached(A_cool, imax, itop)
                    var M = N + Nb + Na
                    for k in range(M):
                        wa[unsafe_offset=k] = 0.0
                        wA[unsafe_offset=k] = 0.0
                        wq[unsafe_offset=k] = 0.0
                        wn[unsafe_offset=k] = 0.0
                    for k in range(M):
                        var g = i + 1 - Nb + k
                        if g >= 0 and g < NB:
                            if regime == 3:
                                wa[unsafe_offset=k] = (
                                    a_esc[unsafe_offset=g] + 1e-150
                                )
                            wA[unsafe_offset=k] = (
                                A_cool[unsafe_offset=g] - 1e-150
                            )
                            wq[unsafe_offset=k] = (
                                q_inj[unsafe_offset=g] + 1e-150
                            )
                            wn[unsafe_offset=k] = n0[unsafe_offset=g] + 1e-150
                    # the cells below the window (other regimes, below the
                    # grid, bin 0) take the cooling rate of its first bin. A
                    # bin's solution needs only the bins above it, but the
                    # cooling time is summed from the lowest cell: AM3's A = 0
                    # below the grid (or a much slower cooling) made the sum
                    # swamp dt, and the window did not cool (AM3 bug)
                    var kfirst = -1
                    for k in range(Nb, M):
                        var g = i + 1 - Nb + k
                        if g >= 1 and g < NB:
                            kfirst = k
                            break
                    if kfirst > 0:
                        for k in range(kfirst):
                            wA[unsafe_offset=k] = wA[unsafe_offset=kfirst]
                    if regime == 3:
                        self.solve_steadystate_esc_cool_inj(
                            wa, wA, wq, wn, wo, M, i - Nb
                        )
                    else:
                        self.solve_steadystate_cool_inj(
                            wA, wq, wn, wo, M, i - Nb
                        )
                    for k in range(N):
                        var g = i + 1 + k
                        if g >= 0 and g < NB:
                            n1[unsafe_offset=g] = wo[unsafe_offset=k + Nb]
                imax = i
                regime_above = regime
                regime = new_regime
            i -= 1
        n1[unsafe_offset=0] = 0.0
        n1[unsafe_offset=1] = 0.0
        if self.verbose:
            print("finished evolving ", self.particle_name)

    @inline(.never)
    def cells_reached(self, A_cool: FPtr, imax: Int, itop: Int) -> Int:
        """Number of cells above imax (at least 10) from which particles
        cooling at A_cool reach imax within the step (one more as margin), up
        to the cell above itop, the highest bin with particles."""
        var tc = 0.0
        var g = imax + 1
        while g < self.N_BINS and g <= itop + 1 and tc <= self.delta_t:
            tc += self.DELTA_X / (-A_cool[unsafe_offset=g] + 1e-300)
            g += 1
        return max(10, g - imax + 1)

    @inline(.never)
    def crossing_count(
        self,
        a_esc: FPtr,
        A_cool: FPtr,
        q_inj: FPtr,
        n0: FPtr,
        ib: Int,
        itop: Int,
    ) -> Float64:
        """The particles that cool across the upper edge x_b = x_ib + dx/2 of
        bin ib within the step, as n dx (n per ln E and volume, dx the bin
        width):

            int_{x_b}^{y_b} [n0(x) + q(x) (dt - tau(x))] e^(-S(x)) dx,

        with tau(x) = int_{x_b}^x dx'/|A|, S(x) = int_{x_b}^x a/|A| dx' and
        tau(y_b) = dt (exact along the characteristics for A, a and q constant
        in the step).

        As in regimes 3 and 4, n0, q, ln(1/|A|) and ln(a/|A|) are linear
        between the bin centres above the edge. From the edge to the centre of
        bin ib + 1, they keep that bin's values, as the window above does there
        (a regime-3/4 window gives the cells below it its first bin's cooling
        rate; the Chang & Cooper flux through the edge is the upper cell's).
        4-point Gauss-Legendre on pieces over which S grows by at most 1, up to
        the cell above itop, the highest bin with particles."""
        var NB = self.N_BINS
        var dt = self.delta_t
        if ib + 1 >= NB:
            return 0.0
        # the interval [x_j, x_j + h]: from x_b to the centre of bin ib + 1,
        # then between the bin centres
        var Lj = c_log(1.0 / (-A_cool[unsafe_offset=ib + 1] + 1e-200))
        var Mj = c_log(
            (a_esc[unsafe_offset=ib + 1] + 1e-200)
            / (-A_cool[unsafe_offset=ib + 1] + 1e-200)
        )
        var nj = n0[unsafe_offset=ib + 1]
        var qj = q_inj[unsafe_offset=ib + 1]
        var h = 0.5 * self.DELTA_X
        var tau = 0.0
        var S = 0.0
        var total = 0.0
        var k = ib + 1
        while k < NB and k <= itop + 1:
            var Lk = c_log(1.0 / (-A_cool[unsafe_offset=k] + 1e-200))
            var Mk = c_log(
                (a_esc[unsafe_offset=k] + 1e-200)
                / (-A_cool[unsafe_offset=k] + 1e-200)
            )
            var nk = n0[unsafe_offset=k]
            var qk = q_inj[unsafe_offset=k]
            var dL = Lk - Lj
            var dM = Mk - Mj
            var hL = h * c_exp(Lj)  # tau(s) = tau + hL phi1(dL s/h) s/h
            var hM = h * c_exp(Mj)  # S(s) = S + hM phi1(dM s/h) s/h
            var dtau = hL * _phi1(dL)
            var send = h
            var last = False
            if tau + dtau >= dt:
                # the part of the interval within reach: tau(send) = dt
                var r = (dt - tau) / hL
                send = h * (
                    r if abs(dL * r) < 1e-12 else c_log1p(dL * r) / dL
                )
                last = True
            var dS = hM * _phi1(dM * send / h) * send / h
            var npiece = 1 if dS <= 1.0 else min(64, Int(ceil(dS)))
            var ds = send / Float64(npiece)
            for p in range(npiece):
                var s0 = ds * Float64(p)
                for g in range(4):
                    var sv = s0 + ds * global_constant[_GL4_T]()[g]
                    var u = sv / h
                    var tau_s = tau + hL * _phi1(dL * u) * u
                    var S_s = S + hM * _phi1(dM * u) * u
                    var n_s = nj + (nk - nj) * u
                    var q_s = qj + (qk - qj) * u
                    total += (
                        global_constant[_GL4_W]()[g]
                        * ds
                        * (n_s + q_s * max(dt - tau_s, 0.0))
                        * c_exp(-S_s)
                    )
            if last:
                break
            tau += dtau
            S += hM * _phi1(dM)
            if S > 60.0:
                break
            Lj = Lk
            Mj = Mk
            nj = nk
            qj = qk
            h = self.DELTA_X
            k += 1
        return total

    # ------------------------------------------------------------------
    # regime 1: escape + injection only
    # ------------------------------------------------------------------
    def solve_steadystate_esc_inj(
        self, a: FPtr, q: FPtr, n0: FPtr, res: FPtr, N: Int
    ):
        if self.verbose:
            print(
                self.particle_name,
                " using evolve_steadystate_esc_inj for ",
                N,
                " bins.",
                sep="",
            )
        var dt = self.delta_t
        for i in range(N):
            var ai = a[unsafe_offset=i]
            var at = ai * dt
            if at < 0.1:
                res[unsafe_offset=i] = (1.0 - at + 0.5 * at * at) * n0[
                    unsafe_offset=i
                ] + (dt - 0.5 * dt * dt * ai) * q[unsafe_offset=i]
            elif at >= 0.1 and at < 200.0:
                res[unsafe_offset=i] = c_exp(-at) * n0[unsafe_offset=i] + (
                    q[unsafe_offset=i] / ai
                ) * (1.0 - c_exp(-at))
            else:
                res[unsafe_offset=i] = q[unsafe_offset=i] / ai

    # ------------------------------------------------------------------
    # regime 2: Chang & Cooper tridiagonal matrix solver
    # ------------------------------------------------------------------
    @inline(.always)
    def chc_delta(self, k: Int, p: Float64, q: Float64) -> Float64:
        if k == 0:
            return 0.5
        var w = -(p / q) * self.DELTA_X
        if w > -0.01 and w < 0.01:
            return 0.5
        elif w > 50.0:
            return 1.0 / w
        elif w < -50.0:
            return 1.0
        else:
            return 1.0 / w - 1.0 / (c_exp(w) - 1.0)

    @inline(.always)
    def A_w(
        self, i: Int, j: Int, p: Float64, q: Float64, A_total: FPtr
    ) -> Float64:
        var d0 = self.chc_delta(j, p, q)
        return (
            d0 * A_total[unsafe_offset=i]
            + (1.0 - d0) * A_total[unsafe_offset=i + 1]
        )

    def solve_matrix(
        mut self,
        a_esc: FPtr,
        A_cool: FPtr,
        q_inj: FPtr,
        n0: FPtr,
        out_: FPtr,
        N: Int,
    ) raises:
        if self.verbose:
            print(
                self.particle_name,
                " using evolve_matrix for ",
                N,
                " bins.",
                sep="",
            )
        var DA = pd(self.m_DA)
        var DB = pd(self.m_DB)
        var DD = pd(self.m_DD)
        var PL = pd(self.m_PL)
        var PD = pd(self.m_PD)
        var PR = pd(self.m_PR)
        var TS = pd(self.m_TS)
        var TD = pd(self.m_TD)
        var TB = pd(self.m_TB)
        var TQ = pd(self.m_TQ)
        var TN = pd(self.m_TN)
        _zero(out_, N)
        _zero(DA, N)
        _zero(DB, N)
        _zero(DD, N)
        _zero(TN, N)
        _zero(PR, N)
        _zero(PD, N)
        _zero(PL, N)
        _zero(TS, N)
        _zero(TD, N)
        _zero(TB, N)
        _zero(TQ, N)
        # construct_trid_mat
        var DX = self.DELTA_X
        var r = 0.5 * self.delta_t / DX
        var size = N
        var B = 1e-100
        for i in range(size - 1):
            var rA = self.A_w(i, 0, 1.0, 1.0, A_cool)
            var rB = B
            # two Chang & Cooper iterations of A (B stays constant), delta from
            # the last
            for _ in range(2):
                rA = self.A_w(i, 1, rA, rB, A_cool)
            DA[unsafe_offset=i] = rA
            DB[unsafe_offset=i] = rB
            DD[unsafe_offset=i] = self.chc_delta(1, rA, rB)
        DA[unsafe_offset=0] = 0.0
        DA[unsafe_offset=size - 1] = 0.0
        DB[unsafe_offset=0] = 0.0
        DB[unsafe_offset=size - 1] = 0.0
        DD[unsafe_offset=0] = 0.5
        DD[unsafe_offset=size - 1] = 0.5
        for i in range(1, size):
            var p1 = (1.0 - DD[unsafe_offset=i]) * DA[unsafe_offset=i] - DB[
                unsafe_offset=i
            ] / DX
            var p3 = (
                DD[unsafe_offset=i] * DA[unsafe_offset=i]
                - (1.0 - DD[unsafe_offset=i - 1]) * DA[unsafe_offset=i - 1]
                + (DB[unsafe_offset=i] + DB[unsafe_offset=i - 1]) / DX
            )
            var p4 = DX
            var p5 = (
                -DD[unsafe_offset=i - 1] * DA[unsafe_offset=i - 1]
                - DB[unsafe_offset=i - 1] / DX
            )
            PR[unsafe_offset=i] = r * p1
            PD[unsafe_offset=i] = r * (p3 + p4 * a_esc[unsafe_offset=i])
            PL[unsafe_offset=i] = r * p5
        PR[unsafe_offset=0] = 0.0
        PD[unsafe_offset=0] = 0.0
        PL[unsafe_offset=0] = 0.0
        for i in range(1, size - 1):
            TS[unsafe_offset=i] = PR[unsafe_offset=i]
            TD[unsafe_offset=i] = 1.0 + PD[unsafe_offset=i]
            TB[unsafe_offset=i - 1] = PL[unsafe_offset=i]
            TQ[unsafe_offset=i] = q_inj[unsafe_offset=i] * self.delta_t
        TS[unsafe_offset=0] = PR[unsafe_offset=0]
        TD[unsafe_offset=0] = 1.0 + PD[unsafe_offset=0]
        TS[unsafe_offset=size - 1] = 0.0
        TD[unsafe_offset=size - 1] = 1.0 + PD[unsafe_offset=size - 1]
        TB[unsafe_offset=size - 2] = PL[unsafe_offset=size - 1]
        TB[unsafe_offset=size - 1] = 0.0
        # the last cell is the top of the window and keeps its injection,
        # which includes the inflow from above (AM3: an unused padding cell)
        TQ[unsafe_offset=size - 1] = (
            q_inj[unsafe_offset=size - 1] * self.delta_t
        )
        # TempNVE = M * n0
        for i in range(1, size - 1):
            TN[unsafe_offset=i] = (
                PL[unsafe_offset=i] * n0[unsafe_offset=i - 1]
                + PD[unsafe_offset=i] * n0[unsafe_offset=i]
                + PR[unsafe_offset=i] * n0[unsafe_offset=i + 1]
            )
        TN[unsafe_offset=0] = (
            PD[unsafe_offset=0] * n0[unsafe_offset=0]
            + PR[unsafe_offset=0] * n0[unsafe_offset=1]
        )
        TN[unsafe_offset=size - 1] = (
            PL[unsafe_offset=size - 1] * n0[unsafe_offset=size - 2]
            + PD[unsafe_offset=size - 1] * n0[unsafe_offset=size - 1]
        )
        for i in range(size):
            TN[unsafe_offset=i] = (
                -TN[unsafe_offset=i] + TQ[unsafe_offset=i] + n0[unsafe_offset=i]
            )
        var ok = solve_tridiag(
            TD, TS, TB, TN, out_, size, pd(self.m_alpha), pd(self.m_z)
        )
        if not ok:
            raise Error("matrix must be positive definite")

    # ------------------------------------------------------------------
    # regimes 3/4: semi-analytical steady state for dt
    # ------------------------------------------------------------------
    def rk4_gslspl_b(
        self,
        lower: Float64,
        upper: Float64,
        N_INTG: Int,
        x_min: Float64,
        x_max: Float64,
        mut spl_lnmivA: LinearSpline,
    ) -> Float64:
        var x = lower
        var y = 0.0
        var diff_x = (upper - lower) / Float64(N_INTG)
        # a step that leaves [x_min, x_max] ends the sum (x stops); the
        # integrand at the end of a step is the one at the start of the next
        # (the same x). The sum (k1 + k4) / 6 + 2 k2 / 3 is written with the
        # fused multiply-adds the compiler makes of it.
        if x < x_min or x + 1.0 * diff_x > x_max:
            return y
        var f1 = c_exp(spl_lnmivA.eval(x))
        for _ in range(N_INTG):
            if x < x_min or x + 1.0 * diff_x > x_max:
                break
            var k2 = diff_x * c_exp(spl_lnmivA.eval(x + 0.5 * diff_x))
            var f4 = c_exp(spl_lnmivA.eval(x + 1.0 * diff_x))
            var k14 = fma(diff_x, f4, diff_x * f1)
            y += fma(k2, 2.0 / 3.0, (1.0 / 6.0) * k14)
            x += diff_x
            f1 = f4
        return y

    def compute_y(
        mut self, N: Int, x_ln: FPtr, y_ln: FPtr, mut spl_lnmivA: LinearSpline
    ) raises:
        if self.verbose:
            print("computing y - filling b_lin")
        var b_lin = pd(self.w_b)
        b_lin[unsafe_offset=0] = 0.0
        var x0 = x_ln[unsafe_offset=0]
        var xN = x_ln[unsafe_offset=N - 1]
        for i in range(1, N):
            var delta_b = self.rk4_gslspl_b(
                x_ln[unsafe_offset=i - 1],
                x_ln[unsafe_offset=i],
                10,
                x0,
                xN,
                spl_lnmivA,
            )
            var set_delta_b_min = b_lin[unsafe_offset=i - 1] * 1.0e-15
            delta_b = set_delta_b_min if delta_b < set_delta_b_min else delta_b
            b_lin[unsafe_offset=i] = b_lin[unsafe_offset=i - 1] + delta_b
            if not (b_lin[unsafe_offset=i] > b_lin[unsafe_offset=i - 1]):
                var prev = b_lin[unsafe_offset=i - 1]
                var dir = max(prev + 1.0, prev * 1.0)
                b_lin[unsafe_offset=i] = nextafter(prev, dir)
        if spl_lnmivA.error:
            raise Error("interpolation error (gsl_spline_eval out of range)")
        for i in range(1, N):
            if not (b_lin[unsafe_offset=i - 1] < b_lin[unsafe_offset=i]):
                raise Error("x values must be strictly increasing")
        var spl_ivB = LinearSpline(b_lin, x_ln, N)
        var b_max = b_lin[unsafe_offset=N - 1]
        var x_max = xN
        if self.verbose:
            print("computing y - filling y")
        var dt = self.delta_t
        for i in range(N):
            y_ln[unsafe_offset=i] = x_max - 1.0e-3 * self.DELTA_X
            if b_lin[unsafe_offset=i] + dt < b_max:
                y_ln[unsafe_offset=i] = spl_ivB.eval(
                    b_lin[unsafe_offset=i] + dt
                )
        if spl_ivB.error:
            raise Error("interpolation error (gsl_spline_eval out of range)")

    def rk4_gslspl_a(
        self,
        lower: Float64,
        upper: Float64,
        N_INTG: Int,
        x_min: Float64,
        x_max: Float64,
        mut spl_lna: LinearSpline,
    ) raises -> Float64:
        if lower < x_min or upper > x_max:
            print("in rk4_gslspl_a, ", upper, " > ", x_max)
            raise Error(
                "in function rk4_gslspl_a, integration range out of bound\n"
            )
        var diff_x = (upper - lower) / Float64(N_INTG)
        var x = lower
        var y = 0.0
        # the integrand at the end of a step is the one at the start of the
        # next (the same x). The sum (k1 + k4) / 6 + 2 k2 / 3, with
        # k = -diff_x * loss, is written with the fused multiply-adds the
        # compiler makes of it.
        var k1_lnloss = spl_lna.eval(x)
        var k1_loss = 0.0 if k1_lnloss < -200.0 else c_exp(k1_lnloss)
        for _ in range(N_INTG):
            var k2_lnloss = spl_lna.eval(x + 0.5 * diff_x)
            var k4_lnloss = spl_lna.eval(x + 1.0 * diff_x)
            var k2_loss = 0.0 if k2_lnloss < -200.0 else c_exp(k2_lnloss)
            var k4_loss = 0.0 if k4_lnloss < -200.0 else c_exp(k4_lnloss)
            var k2 = -diff_x * k2_loss
            var k4 = -diff_x * k4_loss
            y += fma(fma(-diff_x, k1_loss, k4), 1.0 / 6.0, (2.0 / 3.0) * k2)
            x += diff_x
            k1_loss = k4_loss
        return y

    def rk4_gslspl_q(
        self,
        lower: Float64,
        upper: Float64,
        x_min: Float64,
        x_max: Float64,
        mut spl_lna: LinearSpline,
        mut spl_q: LinearSpline,
        mut spl_lnmivA: LinearSpline,
    ) raises -> Float64:
        if self.verbose:
            print("calling rk4_gslspl_q")
        if lower < x_min or upper > x_max:
            raise Error(
                "in function rk4_gslspl_q, integration range out of bound\n"
            )
        if lower >= upper:
            raise Error(
                "in function rk4_gslspl_q, integration bound lower >= upper"
                " occurred \n"
            )
        var y = 0.0
        var abs_alpha_ivA = c_exp(spl_lna.eval(lower))
        var I_alpha_ivA = (upper - lower) * abs_alpha_ivA
        if I_alpha_ivA < 1.0e-3:
            var N_min = 10
            var N_eff = Int((upper - lower) / 0.1)
            var N_pts = N_min if N_min > N_eff else N_eff
            var diff_x = (upper - lower) / Float64(N_pts)
            var x_prime = lower
            for _ in range(N_pts):
                var k1 = diff_x * spl_q.eval(x_prime)
                var k2 = diff_x * spl_q.eval(x_prime + 0.5 * diff_x)
                var k4 = diff_x * spl_q.eval(x_prime + 1.0 * diff_x)
                y += (1.0 / 6.0) * (k1 + k4) + (2.0 / 3.0) * k2
                x_prime += diff_x
            y *= c_exp(spl_lnmivA.eval(lower))
        else:
            var upper_eff = upper
            var range_ = upper - lower
            comptime N_OUTER_INIT = 10
            comptime N_INNER_INIT = 10
            var diff_x = range_ / Float64(N_OUTER_INIT)
            if diff_x / lower < 1.0e-10:
                print("lower bound = ", lower)
                print("upper bound = ", upper_eff)
                raise Error(
                    "in function rk4_gslspl_q, integration range too narrow \n"
                )
            var stat_reduced_range = False
            while (
                self.rk4_gslspl_a(
                    lower, lower + diff_x, N_INNER_INIT, x_min, x_max, spl_lna
                )
                < -2.0
            ):
                upper_eff -= 0.5 * range_
                range_ = upper_eff - lower
                diff_x = range_ / Float64(N_OUTER_INIT)
                stat_reduced_range = True
            # (the injection comes from below y = upper only: AM3 extended the
            #  range to 0.99 x_max, the top of the window, and did not update
            #  diff_x here, so the loop never ended once entered - AM3 bugs)
            if stat_reduced_range:
                while (
                    self.rk4_gslspl_a(
                        lower,
                        lower + diff_x,
                        N_INNER_INIT,
                        x_min,
                        x_max,
                        spl_lna,
                    )
                    > -0.8
                    and upper_eff < upper
                ):
                    upper_eff = min(upper_eff + 0.5 * range_, upper)
                    range_ = upper_eff - lower
                    diff_x = range_ / Float64(N_OUTER_INIT)
            var N_min = 10
            var N_eff = Int((upper_eff - lower) / 0.1)
            var N_pts = N_min if N_min > N_eff else N_eff
            diff_x = (upper_eff - lower) / Float64(N_pts)
            var x_prime = lower
            for _ in range(N_pts):
                var k1_lnloss = self.rk4_gslspl_a(
                    lower, x_prime, N_INNER_INIT, x_min, x_max, spl_lna
                )
                var k2_lnloss = self.rk4_gslspl_a(
                    lower,
                    x_prime + 0.5 * diff_x,
                    N_INNER_INIT,
                    x_min,
                    x_max,
                    spl_lna,
                )
                var k4_lnloss = self.rk4_gslspl_a(
                    lower,
                    x_prime + 1.0 * diff_x,
                    N_INNER_INIT,
                    x_min,
                    x_max,
                    spl_lna,
                )
                var k1_loss = 0.0 if k1_lnloss < -300.0 else c_exp(k1_lnloss)
                var k2_loss = 0.0 if k2_lnloss < -300.0 else c_exp(k2_lnloss)
                var k4_loss = 0.0 if k4_lnloss < -300.0 else c_exp(k4_lnloss)
                var k1 = diff_x * spl_q.eval(x_prime) * k1_loss
                var k2 = diff_x * spl_q.eval(x_prime + 0.5 * diff_x) * k2_loss
                var k4 = diff_x * spl_q.eval(x_prime + 1.0 * diff_x) * k4_loss
                y += (1.0 / 6.0) * (k1 + k4) + (2.0 / 3.0) * k2
                x_prime += diff_x
            y *= c_exp(spl_lnmivA.eval(lower))
        return y

    def simple_integral_q(
        self,
        N: Int,
        x_ln: FPtr,
        y_ln: FPtr,
        a_ln: FPtr,
        q: FPtr,
        mivA_ln: FPtr,
        mut spl_lna: LinearSpline,
        mut spl_q: LinearSpline,
        inj: FPtr,
    ):
        """inj[i] = int_x^y q(x') exp(-int_x^x' alpha / |A| dx'') dx' / |A(x)|
        (x = x_ln[i], y = y_ln[i]) for the bins 1 .. N - 2 of the window.

        _source_cell integrates each cell of the splines (ln(alpha / |A|) and q
        linear in x), exactly where the sink or the cooling dominates. Chained
        from the top, J_k = P_k + e^-D_k J_(k+1) (P_k, D_k: integral and
        exponent of cell k), the integral is J_i - e^-(K_m - K_i) (J_m - the
        part of cell m below y), with K the cumulative exponent and m the cell
        that holds y. Where that loses more than 4 digits (the injection above
        y dominating), the cells from x to y are summed instead.

        (AM3 summed q(x') exp(-int_x^(x' + dx) alpha / |A|) dx over the grid
        points below y, with exp(+...), and dropped the injection where the
        sink is negligible: AM3 bugs. Its densities were too low by
        z / (e^z - 1), z = dx alpha / |A|, where the sink dominates, and off by
        about dx / 2 times the logarithmic slope of q where cooling dominates.)
        """
        var DX = self.DELTA_X
        var P = inj  # P_k, overwritten by inj[k] once J is known
        var Dc = pd(self.w_q2)
        var J = pd(self.w_q3)
        var K = pd(self.w_K)
        for k in range(N - 1):
            var Dk = 0.0
            P[unsafe_offset=k] = _source_cell(
                DX,
                a_ln[unsafe_offset=k],
                a_ln[unsafe_offset=k + 1],
                q[unsafe_offset=k],
                q[unsafe_offset=k + 1],
                Dk,
            )
            Dc[unsafe_offset=k] = Dk
        J[unsafe_offset=N - 1] = 0.0
        var k = N - 2
        while k >= 0:
            J[unsafe_offset=k] = (
                P[unsafe_offset=k]
                + c_exp(-Dc[unsafe_offset=k]) * J[unsafe_offset=k + 1]
            )
            k -= 1
        K[unsafe_offset=0] = 0.0
        for k2 in range(1, N):
            K[unsafe_offset=k2] = (
                K[unsafe_offset=k2 - 1] + Dc[unsafe_offset=k2 - 1]
            )
        inj[unsafe_offset=0] = 0.0
        for i in range(1, N - 1):
            var xi = x_ln[unsafe_offset=i]
            var yi = y_ln[unsafe_offset=i]
            # the cell that holds y: x_ln[m] <= y < x_ln[m + 1]
            var m = min(N - 2, i + max(0, Int((yi - xi) / DX)))
            while m > i and x_ln[unsafe_offset=m] > yi:
                m -= 1
            while m < N - 2 and x_ln[unsafe_offset=m + 1] <= yi:
                m += 1
            var Dp = 0.0
            var part = _source_cell(
                yi - x_ln[unsafe_offset=m],
                a_ln[unsafe_offset=m],
                spl_lna.eval(yi),
                q[unsafe_offset=m],
                spl_q.eval(yi),
                Dp,
            )
            var I = part
            if m > i:
                I = J[unsafe_offset=i] - c_exp(
                    -(K[unsafe_offset=m] - K[unsafe_offset=i])
                ) * (J[unsafe_offset=m] - part)
                # over 4 digits lost: sum the cells from x to y instead (P_k,
                # k >= i, are not overwritten yet)
                if I * 1e4 < J[unsafe_offset=i]:
                    I = (
                        c_exp(-(K[unsafe_offset=m] - K[unsafe_offset=i]))
                        * part
                    )
                    for kk in range(i, m):
                        I += P[unsafe_offset=kk] * c_exp(
                            -(K[unsafe_offset=kk] - K[unsafe_offset=i])
                        )
            inj[unsafe_offset=i] = I * c_exp(mivA_ln[unsafe_offset=i])

    def solve_steadystate_esc_cool_inj(
        mut self,
        a_esc: FPtr,
        A_cool: FPtr,
        q_inj: FPtr,
        n0: FPtr,
        n1: FPtr,
        N: Int,
        i0: Int,
    ) raises:
        if self.verbose:
            print(
                self.particle_name,
                " using solve_steadystate_esc_cool_inj for ",
                N,
                " bins.",
                sep="",
            )
        var x_ln = pd(self.w_x)
        var mivA_ln = pd(self.w_mivA)
        var a_ln = pd(self.w_aln)
        var y_ln = pd(self.w_y)
        var DX = self.DELTA_X
        for i in range(N):
            x_ln[unsafe_offset=i] = DX * Float64(i + i0)
            mivA_ln[unsafe_offset=i] = c_log(
                1.0 / (-A_cool[unsafe_offset=i] + 1e-200)
            )
            a_ln[unsafe_offset=i] = c_log(
                (a_esc[unsafe_offset=i] + 1e-200)
                / (-A_cool[unsafe_offset=i] + 1e-200)
            )
            y_ln[unsafe_offset=i] = 0.0
        # (the q-integral arrays of the C++ version are not used by this
        # regime; w_q1 .. w_q3 hold simple_integral_q's cells instead)
        var spl_lna = LinearSpline(x_ln, a_ln, N)
        var spl_lnmivA = LinearSpline(x_ln, mivA_ln, N)
        var spl_q = LinearSpline(x_ln, q_inj, N)
        var spl_n = LinearSpline(x_ln, n0, N)
        self.compute_y(N, x_ln, y_ln, spl_lnmivA)
        var inj = pd(self.w_q1)
        if not self.simpleqintegration:
            self.simple_integral_q(
                N, x_ln, y_ln, a_ln, q_inj, mivA_ln, spl_lna, spl_q, inj
            )
        # compute_n1_a_A
        if self.verbose:
            print("computing n1")
        var dt = self.delta_t
        var x0 = x_ln[unsafe_offset=0]
        var xN = x_ln[unsafe_offset=N - 1]
        for i in range(N):
            n1[unsafe_offset=i] = 0.0
        for i in range(1, N - 1):
            var yi = y_ln[unsafe_offset=i]
            var xi = x_ln[unsafe_offset=i]
            var ai = a_esc[unsafe_offset=i]
            if yi - xi < 1.0e-4:
                var effective_duration = (
                    c_exp(-ai * dt) - 1.0
                ) / ai if ai * dt > 1.0e-4 else -dt
                n1[unsafe_offset=i] = (
                    n0[unsafe_offset=i] * c_exp(-ai * dt)
                    - q_inj[unsafe_offset=i] * effective_duration
                )
            else:
                var ny = spl_n.eval(yi)
                var ivAy = -c_exp(spl_lnmivA.eval(yi))
                var Aratio = 1.0 / A_cool[unsafe_offset=i] / ivAy
                spl_lna.hint(i)  # (x_i is grid point i)
                var lnloss = self.rk4_gslspl_a(xi, yi, 5, x0, xN, spl_lna)
                var loss = 0.0 if lnloss < -100.0 else c_exp(lnloss)
                var injection: Float64
                if self.simpleqintegration:
                    injection = self.rk4_gslspl_q(
                        xi, yi, x0, xN, spl_lna, spl_q, spl_lnmivA
                    )
                else:
                    injection = inj[unsafe_offset=i]
                n1[unsafe_offset=i] = ny * Aratio * loss + injection
        n1[unsafe_offset=N - 1] = 0.0
        if spl_lna.error or spl_lnmivA.error or spl_q.error or spl_n.error:
            raise Error("interpolation error (gsl_spline_eval out of range)")

    def solve_steadystate_cool_inj(
        mut self, A_cool: FPtr, q_inj: FPtr, n0: FPtr, n1: FPtr, N: Int, i0: Int
    ) raises:
        if self.verbose:
            print(
                self.particle_name,
                " using solve_steadystate_cool_inj for ",
                N,
                " bins.",
                sep="",
            )
        var x_ln = pd(self.w_x)
        var mivA_ln = pd(self.w_mivA)
        var q_intg = pd(self.w_q1)
        var y_ln = pd(self.w_y)
        var DX = self.DELTA_X
        for i in range(N):
            x_ln[unsafe_offset=i] = DX * Float64(i + i0)
            mivA_ln[unsafe_offset=i] = c_log(
                1.0 / (-A_cool[unsafe_offset=i] + 1e-200)
            )
            y_ln[unsafe_offset=i] = 0.0
        # the injection integral -int_x^top q dx (trapezoids), summed from the
        # top, keeps the digits of int_x^y q dx where q falls steeply to high
        # energies: a bin's particles come from above it. (AM3 summed from the
        # bottom, restarting twice where q dx / sum < 1e-15. The bins below
        # each restart kept almost no significant digits of their injection,
        # and those above a third such point (no restart left) none: AM3 bug)
        q_intg[unsafe_offset=N - 1] = 0.0
        var k = N - 2
        while k >= 0:
            q_intg[unsafe_offset=k] = (
                q_intg[unsafe_offset=k + 1]
                - 0.5
                * (q_inj[unsafe_offset=k] + q_inj[unsafe_offset=k + 1])
                * DX
            )
            k -= 1
        var spl_lnmivA = LinearSpline(x_ln, mivA_ln, N)
        var spl_q_intg = LinearSpline(x_ln, q_intg, N)
        # (q_intg at x and at y: one spline for each, so that both searches
        # move up the grid)
        var spl_q_intg_x = LinearSpline(x_ln, q_intg, N)
        var spl_n = LinearSpline(x_ln, n0, N)
        self.compute_y(N, x_ln, y_ln, spl_lnmivA)
        if self.verbose:
            print("computing n1")
        var dt = self.delta_t
        for i in range(N):
            n1[unsafe_offset=i] = 0.0
        for i in range(1, N - 1):
            var yi = y_ln[unsafe_offset=i]
            var xi = x_ln[unsafe_offset=i]
            if yi - xi < 1.0e-4:
                n1[unsafe_offset=i] = (
                    n0[unsafe_offset=i] + q_inj[unsafe_offset=i] * dt
                )
            else:
                var ny = spl_n.eval(yi)
                var mivAy = -c_exp(spl_lnmivA.eval(yi))
                var intgd_injection = spl_q_intg.eval(yi) - spl_q_intg_x.eval(
                    xi
                )
                var Ai = A_cool[unsafe_offset=i]
                # A(y)/A(x) n(y) = (1/A(x)) n(y) / (1/A(y)) (AM3 had the
                # opposite sign of this term)
                n1[unsafe_offset=i] = ny / Ai / mivAy - intgd_injection / Ai
        n1[unsafe_offset=N - 1] = 0.0
        if (
            spl_lnmivA.error
            or spl_q_intg.error
            or spl_q_intg_x.error
            or spl_n.error
        ):
            raise Error("interpolation error (gsl_spline_eval out of range)")
