"""Port of SolverAlgorithm (Solver.cc): time-step solver for one species.

Differences from the C++ implementation that do not change results:
  * work arrays are allocated once per solver instead of per call;
  * GSL splines/tridiagonal solves are replaced by the ports in interp.mojo;
  * IC cooling uses a kernel stored photon-major so it runs on SIMD lanes;
  * target photon bins beyond the end of the photon array (read out of bounds
    by the C++ code for hadrons) are treated as empty.
"""

from std.math import fma
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.interp import LinearSpline, solve_tridiag


@no_inline
def _zero(p: FPtr, n: Int):
    for i in range(n):
        p[unsafe_offset=i] = 0.0


struct SolverAlgorithm(Copyable, Movable):
    var particle_name: String
    var N_BINS: Int
    var PHOT_GRID_ORIGIN: Int
    var DELTA_X: Float64
    var X_UNIT_GEV: Float64
    var L_UNIT_CM: Float64
    var T_UNIT_S: Float64
    var HALF_LIFE_S: Float64
    var MASS_R: Float64
    var delta_t: Float64
    var step_counter: Int
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
    var Asy: Vec
    var Aic: Vec
    var cooling_on: Bool
    var ker_status: Int
    # IC cooling kernel, stored [photon j][particle i] (particle index contiguous)
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
        self.HALF_LIFE_S = HLS
        self.MASS_R = MR
        self.delta_t = Dt
        self.step_counter = 0
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
        self.Asy = zeros(NB)
        self.Aic = zeros(NB)
        self.cooling_on = False
        self.ker_status = 0
        self.N_PHOT = NB - PGO
        self.ker_ic_t = Vec()
        var mass_e = 0.511e-3
        var mass = MR * mass_e
        for i in range(NB):
            var gamma = c_exp(DX * Float64(i)) * XUG / mass
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
    def update_cooling(mut self, b_gauss: Float64, n_phot: Vec) raises:
        """Compute Asy and Aic for this step. The rates depend only on the
        field, the target photons and the solver's species, so solvers of the
        same species can share them (see `share_cooling`)."""
        var sum_nphot = 0.0
        var npp = pd(n_phot)
        for i in range(len(n_phot)):
            sum_nphot += npp[unsafe_offset=i]
        self.cooling_on = (self.process_sy and b_gauss > 1.0e-12) or (
            self.process_ic and sum_nphot > 1e-100
        )
        if self.cooling_on:
            self.compute_syn_ic(b_gauss, n_phot)

    def share_cooling(mut self, Asy: Vec, Aic: Vec, cooling_on: Bool):
        """Take the cooling rates computed by a solver of the same species."""
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
        `update_cooling`/`share_cooling`) and the source `q_inj`."""
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

        a_esc[unsafe_offset=0] = 0.0
        a_esc[unsafe_offset=NB - 1] = 0.0
        A_cool[unsafe_offset=0] = 0.0
        A_cool[unsafe_offset=NB - 1] = 0.0

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
            sum_A += c_fabs(A_cool[unsafe_offset=i]) / self.T_UNIT_S
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

    def compute_syn_ic(mut self, b_gauss: Float64, n_phot: Vec) raises:
        if self.verbose:
            print("add syn and ic")
        if self.process_ic and self.ker_status == 0:
            self.compute_ic_ker()
        var T = self.T_UNIT_S
        var L = self.L_UNIT_CM
        var c_light = 3.0e10 * T / L
        var sigma_t = 0.665e-24 / L / L
        var mass_e = 0.511e-3
        var erg_to_gev = 624.0
        var mass = self.MASS_R * mass_e
        var NB = self.N_BINS
        var ub_Edimless = (
            b_gauss * b_gauss / (8.0 * 3.1416 * mass_e / erg_to_gev) * L * L * L
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
        for i in range(NB):
            var gamma = (
                c_exp(self.DELTA_X * Float64(i)) * self.X_UNIT_GEV / mass
            )
            Asy[unsafe_offset=i] = -sy * k_synch * gamma
            if self.process_quantum_syn_cooling:
                var x = (
                    1.2
                    * c_pow(10.0, -8.0)
                    * b_gauss
                    / mass
                    / c_pow(10.0, 9.0)
                    * gamma
                )
                Asy[unsafe_offset=i] *= 1.0 / c_pow(
                    1.0 + c_sqrt(2.0) * c_pow(x, 2.0 / 3.0), 2.0
                )
            Aic[unsafe_offset=i] = 0.0
        if self.process_ic:
            # Aic[i] += -ker[i][j] * n_phot[j] * DX, j ascending, SIMD over i
            var nj = min(self.N_PHOT, len(n_phot))
            var kp = pd(self.ker_ic_t)
            var npp = pd(n_phot)
            var DX = self.DELTA_X
            var i = 0
            while i + WD <= NB:
                var acc = SIMD[DType.float64, WD](0)
                for j in range(nj):
                    var k = kp.unsafe_offset(j * NB + i).unsafe_load[width=WD]()
                    acc = (-k * npp[unsafe_offset=j]).fma(
                        SIMD[DType.float64, WD](DX), acc
                    )
                Aic.unsafe_offset(i).unsafe_store(acc)
                i += WD
            while i < NB:
                var acc = 0.0
                for j in range(nj):
                    acc = fma(
                        -kp[unsafe_offset=j * NB + i] * npp[unsafe_offset=j],
                        DX,
                        acc,
                    )
                Aic[unsafe_offset=i] = acc
                i += 1

    def compute_ic_ker(mut self) raises:
        if self.verbose:
            print("calc ic kernel")
        var mass_e = 0.511e-3
        var mass = self.MASS_R * mass_e
        var NB = self.N_BINS
        var NP = self.N_PHOT
        var T = self.T_UNIT_S
        var L = self.L_UNIT_CM
        var c_light = 3.0e10 * T / L
        var sigma_t = 0.665e-24 / L / L
        var coeff = 3.0 / 16.0 * c_light * sigma_t * mass_e * mass_e
        self.ker_ic_t = zeros(NB * NP)
        var kp = pd(self.ker_ic_t)
        for i in range(NB):
            var gm = c_exp(self.DELTA_X * Float64(i)) * self.X_UNIT_GEV / mass
            for j in range(NP):
                var eph = (
                    c_exp(Float64(self.PHOT_GRID_ORIGIN + j) * self.DELTA_X)
                    * self.X_UNIT_GEV
                )
                kp[unsafe_offset=j * NB + i] = (
                    coeff * self.Fic(gm, eph) / eph / eph / gm / gm / gm
                )
        self.ker_status = 1

    # The expressions below mirror clang's -ffp-contract=on fusion of the
    # C++ source (one fused multiply-add per `a*b +/- c` in a statement);
    # this matters for gamma ~ 1 where Fic suffers heavy cancellation.
    def Fic(self, gamma: Float64, Eph_GeV: Float64) raises -> Float64:
        var mass_mu = self.MASS_R * 0.511e-3
        var gamma_bar = gamma + c_sqrt(fma(gamma, gamma, -1.0) + 1.0e-10)
        var za = Eph_GeV / mass_mu * gamma_bar
        var zb = Eph_GeV / mass_mu / gamma_bar
        var d1 = self.Fic_1(za) - self.Fic_1(zb)
        var d2 = self.Fic_2(za) - self.Fic_2(zb)
        var y = fma(gamma, d1, -((Eph_GeV / mass_mu) * d2))
        return y if y > 0.0 else 0.0

    def Fic_1(self, z: Float64) raises -> Float64:
        if z < 0.1:
            var t1 = 8.0 / 9.0 * z * z
            var t2 = 2.8 * z * z * z * z
            var t3 = 7.84 * z * z * z * z
            return fma(t3, z, fma(t1, z, -t2))
        var A = z + 6.0 + 3.0 / z
        var L = c_log(fma(2.0, z, 1.0))
        var num = 22.0 * z * z * z / 3.0
        num = fma(24.0 * z, z, num)
        num = fma(18.0, z, num)
        num = num + 4.0
        var q = num / fma(2.0, z, 1.0) / fma(2.0, z, 1.0)
        var r = fma(A, L, -q) - 2.0
        return fma(2.0, self.polylog(-2.0 * z), r)

    def Fic_2(self, z: Float64) raises -> Float64:
        if z < 0.1:
            var d = 4.0 * z * z / 3.0 - 128.0 * z * z * z / 45.0
            return fma(7.0 * z * z * z, z, d)
        var A = z + 31.0 / 6.0 + 5.0 / z + 1.5 / z / z
        var L = c_log(fma(2.0, z, 1.0))
        var num = 22.0 * z * z * z / 3.0
        num = fma(28.0 * z, z, num)
        num = num + 103.0 * z / 3.0
        num = num + 17.0
        num = num + 3.0 / z
        var q = num / fma(2.0, z, 1.0) / fma(2.0, z, 1.0)
        var r = fma(A, L, -q) - 2.0
        return r + self.polylog(-2.0 * z)

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
        var t = 0.04 + z / 36.0
        t = fma(z, t, 1.0 / 16.0)
        t = fma(z, t, 1.0 / 9.0)
        t = fma(z, t, 0.25)
        t = fma(z, t, 1.0)
        return z * t

    def polylog_rk4(self, z: Float64) -> Float64:
        var x = 0.1
        var y = 0.0
        comptime N_INTG = 20
        var diff_x = (-z - 0.1) / Float64(N_INTG)
        for _ in range(N_INTG):
            var k1 = -diff_x * c_log(1.0 + x) / x
            var k2 = (
                -diff_x * c_log(fma(0.5, diff_x, 1.0 + x)) / fma(0.5, diff_x, x)
            )
            var k4 = -diff_x * c_log((1.0 + x) + diff_x) / (x + diff_x)
            y = y + fma(1.0 / 6.0, k1 + k4, (2.0 / 3.0) * k2)
            x += diff_x
        return -0.097605235 + y

    def polylog_largez(self, z: Float64) -> Float64:
        var x = -z
        var lx = c_log(x)
        var r = fma(
            -(0.5 * lx), lx, -3.1415926 * 3.1415927 / 6.0
        )  # constant folded as in C++
        r = r + 1.0 / x
        r = r - 1.0 / 4.0 / x / x
        return r + 1.0 / 9.0 / x / x / x

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
        var new_regime = 0
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
            if (
                regime != new_regime
                and (imax - i > n_min_regime or i < n_min_regime)
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
                    self.solve_steadystate_esc_inj(wa, wq, wn, wo, N)
                    for k in range(N):
                        n1[unsafe_offset=i + 1 + k] = wo[unsafe_offset=k]
                elif regime == 2:
                    var Nb = 10
                    var Na = 10
                    var M = N + Nb + Na + 3
                    for k in range(M):
                        wa[unsafe_offset=k] = 0.0
                        wA[unsafe_offset=k] = 0.0
                        wq[unsafe_offset=k] = 0.0
                        wn[unsafe_offset=k] = 0.0
                    for k in range(N + Nb + Na):
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
                    self.solve_matrix(wa, wA, wq, wn, wo, M)
                    for k in range(N):
                        var g = i + 1 + k
                        if g >= 0 and g < NB:
                            n1[unsafe_offset=g] = wo[unsafe_offset=k + Nb]
                elif regime == 3 or regime == 4:
                    var Nb = 10
                    var Na = 10
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
                regime = new_regime
            i -= 1
        n1[unsafe_offset=0] = 0.0
        n1[unsafe_offset=1] = 0.0
        self.step_counter += 1
        if self.verbose:
            print("finished evolving ", self.particle_name)

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
    @always_inline
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

    @always_inline
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
            var rD = 0.5
            # two Chang & Cooper iterations (written out: a `for _ in range(2)`
            # loop here sends the Mojo 1.1 optimizer into a very long compile)
            rA = self.A_w(i, 1, rA, rB, A_cool)
            rB = B
            rD = self.chc_delta(1, rA, rB)
            rA = self.A_w(i, 1, rA, rB, A_cool)
            rB = B
            rD = self.chc_delta(1, rA, rB)
            DA[unsafe_offset=i] = rA
            DB[unsafe_offset=i] = rB
            DD[unsafe_offset=i] = rD
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
        for _ in range(N_INTG):
            if x < x_min or x + 1.0 * diff_x > x_max:
                continue
            var k1 = diff_x * c_exp(spl_lnmivA.eval(x))
            var k2 = diff_x * c_exp(spl_lnmivA.eval(x + 0.5 * diff_x))
            var k4 = diff_x * c_exp(spl_lnmivA.eval(x + 1.0 * diff_x))
            y += (1.0 / 6.0) * (k1 + k4) + (2.0 / 3.0) * k2
            x += diff_x
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
                b_lin[unsafe_offset=i] = c_nextafter(prev, dir)
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
        for _ in range(N_INTG):
            var k1_lnloss = spl_lna.eval(x)
            var k2_lnloss = spl_lna.eval(x + 0.5 * diff_x)
            var k4_lnloss = spl_lna.eval(x + 1.0 * diff_x)
            var k1_loss = 0.0 if k1_lnloss < -200.0 else c_exp(k1_lnloss)
            var k2_loss = 0.0 if k2_lnloss < -200.0 else c_exp(k2_lnloss)
            var k4_loss = 0.0 if k4_lnloss < -200.0 else c_exp(k4_lnloss)
            var k1 = -diff_x * k1_loss
            var k2 = -diff_x * k2_loss
            var k4 = -diff_x * k4_loss
            y += (1.0 / 6.0) * (k1 + k4) + (2.0 / 3.0) * k2
            x += diff_x
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
            var N_eff = to_int((upper - lower) / 0.1)
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
                ):
                    upper_eff += 0.5 * range_
            else:
                upper_eff = 0.99 * x_max
            var N_min = 10
            var N_eff = to_int((upper_eff - lower) / 0.1)
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
        lower: Float64,
        upper: Float64,
        mut spl_lna: LinearSpline,
        mut spl_q: LinearSpline,
        mut spl_lnmivA: LinearSpline,
    ) -> Float64:
        var DX = self.DELTA_X
        var x = lower
        var y = upper
        var N = to_int((y - x) / DX)
        var a_vec = 0.0
        var adx_cum_vec = 0.0
        var integral = 0.0
        for i in range(N):
            var xprime = x + Float64(i) * DX
            var lna = spl_lna.eval(xprime)
            if lna * DX > -30.0:
                a_vec = c_exp(lna)
                adx_cum_vec = adx_cum_vec + DX * a_vec
                var q = spl_q.eval(xprime)
                integral += DX * q * c_exp(adx_cum_vec)
        return integral * c_exp(spl_lnmivA.eval(x))

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
        # (the q-integral arrays of the C++ version are not used by this regime)
        var spl_lna = LinearSpline(x_ln, a_ln, N)
        var spl_lnmivA = LinearSpline(x_ln, mivA_ln, N)
        var spl_q = LinearSpline(x_ln, q_inj, N)
        var spl_n = LinearSpline(x_ln, n0, N)
        self.compute_y(N, x_ln, y_ln, spl_lnmivA)
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
                var lnloss = self.rk4_gslspl_a(xi, yi, 5, x0, xN, spl_lna)
                var loss = 0.0 if lnloss < -100.0 else c_exp(lnloss)
                var injection: Float64
                if self.simpleqintegration:
                    injection = self.rk4_gslspl_q(
                        xi, yi, x0, xN, spl_lna, spl_q, spl_lnmivA
                    )
                else:
                    injection = self.simple_integral_q(
                        xi, yi, spl_lna, spl_q, spl_lnmivA
                    )
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
        var q2_intg = pd(self.w_q2)
        var q3_intg = pd(self.w_q3)
        var y_ln = pd(self.w_y)
        var DX = self.DELTA_X
        for i in range(N):
            x_ln[unsafe_offset=i] = DX * Float64(i + i0)
            mivA_ln[unsafe_offset=i] = c_log(
                1.0 / (-A_cool[unsafe_offset=i] + 1e-200)
            )
            q_intg[unsafe_offset=i] = 0.0
            q2_intg[unsafe_offset=i] = 0.0
            q3_intg[unsafe_offset=i] = 0.0
            y_ln[unsafe_offset=i] = 0.0
        for i in range(1, N):
            q_intg[unsafe_offset=i] = (
                q_intg[unsafe_offset=i - 1]
                + 0.5
                * (q_inj[unsafe_offset=i] + q_inj[unsafe_offset=i - 1])
                * DX
            )
        var q1_q2_sep = 1
        for i in range(1, N - 1):
            if q_inj[unsafe_offset=i] * DX / q_intg[unsafe_offset=i] < 1.0e-15:
                break
            q1_q2_sep += 1
        for i in range(q1_q2_sep, N):
            q2_intg[unsafe_offset=i] = (
                q2_intg[unsafe_offset=i - 1]
                + 0.5
                * (q_inj[unsafe_offset=i] + q_inj[unsafe_offset=i - 1])
                * DX
            )
        var q2_q3_sep = q1_q2_sep
        for i in range(q1_q2_sep, N - 1):
            if q_inj[unsafe_offset=i] * DX / q2_intg[unsafe_offset=i] < 1.0e-15:
                break
            q2_q3_sep += 1
        for i in range(q2_q3_sep, N):
            q3_intg[unsafe_offset=i] = (
                q3_intg[unsafe_offset=i - 1]
                + 0.5
                * (q_inj[unsafe_offset=i] + q_inj[unsafe_offset=i - 1])
                * DX
            )
        var spl_lnmivA = LinearSpline(x_ln, mivA_ln, N)
        var spl_q_intg = LinearSpline(x_ln, q_intg, N)
        var spl_q2_intg = LinearSpline(x_ln, q2_intg, N)
        var spl_q3_intg = LinearSpline(x_ln, q3_intg, N)
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
                var intgd_injection: Float64
                if i < q1_q2_sep:
                    intgd_injection = spl_q_intg.eval(yi) - spl_q_intg.eval(xi)
                elif i < q2_q3_sep:
                    intgd_injection = spl_q2_intg.eval(yi) - spl_q2_intg.eval(
                        xi
                    )
                else:
                    intgd_injection = spl_q3_intg.eval(yi) - spl_q3_intg.eval(
                        xi
                    )
                var Ai = A_cool[unsafe_offset=i]
                n1[unsafe_offset=i] = -ny / Ai / mivAy - intgd_injection / Ai
        n1[unsafe_offset=N - 1] = 0.0
        if (
            spl_lnmivA.error
            or spl_q_intg.error
            or spl_q2_intg.error
            or spl_q3_intg.error
            or spl_n.error
        ):
            raise Error("interpolation error (gsl_spline_eval out of range)")
