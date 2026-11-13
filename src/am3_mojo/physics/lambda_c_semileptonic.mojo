"""Semileptonic Lambda_c+ -> B l+ nu_l decays (B = Lambda, n; l = e, mu) of
the charm decays (physics/decays.mojo, an am3_mojo addition). Their
Dalitz-plot density uses the lattice QCD form factors of Meinel,
PRL 118, 082001 (2017), arXiv:1611.09696 (Lambda_c -> Lambda) and
PRD 97, 034511 (2018), arXiv:1712.05783 (Lambda_c -> N).

Kinematics: Lambda_c(p, M) -> B(p', m_B) l+(k1, m_l) nu_l(k2, 0). The form
factors are the helicity form factors of Feldmann & Yip, defined as in
Detmold, Lehner & Meinel, PRD 92, 034503 (2015), arXiv:1503.01421, Sec. II,
Eqs. (3)-(5), b -> c, q -> s, d (the same as Eqs. (1)-(2) of Meinel 2018).
H_eff = G_F / sqrt(2) V_cq [qbar g^mu (1 - g5) c][nubar g_mu (1 - g5) l],
with the same g5 = i g^0 g^1 g^2 g^3 in both currents.

bl3_density returns W(k1) / 4, the |M|^2 averaged over the Lambda_c spin and
summed over the others, in units of G_F^2 |V_cq|^2:
d^2 Gamma / (dE_B dE_l) = G_F^2 |V_cq|^2 / (64 pi^3 M) * bl3_density.
With pol, it returns W(m_l s) / 4 instead. Here s is the spin vector of
helicity +1/2. (For V - A with a massless neutrino, the projector
(k1/ - m_l)(1 + g5 s/)/2 of the l+ v-spinor gives
|M|^2(s) = 1/2 |M|^2_summed(k1 -> k1 + m_l s). W is linear in k1.)
"""

from std.math import sqrt


@fieldwise_init
struct BL3ZFit(Copyable, Movable):
    """One helicity form factor in the physical limit of a lattice fit
    (mpole [GeV])."""

    var a0: Float64
    var a1: Float64
    var a2: Float64
    var mpole: Float64

    @inline(.always)
    def pole(self, q2: Float64) -> Float64:
        return 1.0 / (1.0 - q2 / (self.mpole * self.mpole))

    @inline(.always)
    def at(self, q2: Float64, z: Float64) -> Float64:
        return (self.a0 + z * (self.a1 + z * self.a2)) * self.pole(q2)


struct BL3FormFactors(Copyable, Movable):
    """tplus = t_+ [GeV^2] of the z expansion. t_0 = q^2_max = (M - m_B)^2 comes
    with the decay's masses. g_+ and g_perp must share a0 and mpole (as in both
    lattice fits)."""

    var fplus: BL3ZFit
    var fperp: BL3ZFit
    var f0: BL3ZFit
    var gplus: BL3ZFit
    var gperp: BL3ZFit
    var g0: BL3ZFit
    var tplus: Float64

    def __init__(
        out self,
        fplus: BL3ZFit,
        fperp: BL3ZFit,
        f0: BL3ZFit,
        gplus: BL3ZFit,
        gperp: BL3ZFit,
        g0: BL3ZFit,
        tplus: Float64,
    ):
        self.fplus = fplus.copy()
        self.fperp = fperp.copy()
        self.f0 = f0.copy()
        self.gplus = gplus.copy()
        self.gperp = gperp.copy()
        self.g0 = g0.copy()
        self.tplus = tplus

    @staticmethod
    def lambdac_lambda() -> BL3FormFactors:
        """Lambda_c -> Lambda, Meinel PRL 118, 082001 (2017): nominal fit of
        Table IV (a_0^{g_perp, g_+} shared) with the digits of the
        supplemental file LambdacLambda_results.dat. Eq. (3) and the text
        after it: D_s pole masses m_pole^{f_+, f_perp} = 2.112,
        m_pole^{f_0} = 2.318, m_pole^{g_+, g_perp} = 2.460,
        m_pole^{g_0} = 1.968 GeV, t_+ = (m_D + m_K)^2 with m_D = 1.870 and
        m_K = 0.494 GeV, t_0 = (m_Lc - m_Lambda)^2."""
        var a0_gpp = 0.67568362532183  # Table IV: 0.68 +- 0.02
        return BL3FormFactors(
            # Table IV: 0.81, -2.89, 7.82
            BL3ZFit(0.80681305417928, -2.8862401794234, 7.8173654487099, 2.112),
            # Table IV: 1.30, -3.27, 7.16
            BL3ZFit(1.2955733552102, -3.2664775710397, 7.1587415496786, 2.112),
            # Table IV: 0.77, -2.24, 5.38
            BL3ZFit(0.77313413420789, -2.2356600619022, 5.3821053913979, 2.318),
            # Table IV: (0.68), -2.44, 13.7
            BL3ZFit(a0_gpp, -2.4371671687515, 13.70127301529, 2.460),
            # Table IV: (0.68), -1.91, 6.24
            BL3ZFit(a0_gpp, -1.9073035414579, 6.2444334166104, 2.460),
            # Table IV: 0.71, -2.86, 11.8
            BL3ZFit(0.7146069790198, -2.8582829568773, 11.80480784834, 1.968),
            (1.870 + 0.494) * (1.870 + 0.494),
        )

    @staticmethod
    def lambdac_neutron() -> BL3FormFactors:
        """Lambda_c -> n, Meinel PRD 97, 034511 (2018) (isospin limit,
        Lambda_c -> n = Lambda_c -> p): nominal fit of Table V
        (a_0^{g_perp, g_+} shared) with the digits of the supplemental file
        Lambdacproton_results.dat. The z expansion of Eqs. (10)-(13):
        t_+ = (m_D + m_pi)^2 with m_D = 1.870 and m_pi = 0.135 GeV (Eq. (12),
        Table IV caption), t_0 = q^2_max = (m_Lc - m_N)^2 (Eq. (13), here
        with m_n). D pole masses of Table IV: m_pole^{f_+, f_perp} = 2.010,
        m_pole^{f_0} = 2.351, m_pole^{g_+, g_perp} = 2.423,
        m_pole^{g_0} = 1.870 GeV."""
        var a0_gpp = 0.68653490341925  # Table V: 0.69 +- 0.02
        return BL3FormFactors(
            # Table V: 0.83, -2.33, 8.41
            BL3ZFit(0.83253055920653, -2.3309542260103, 8.4088373584501, 2.010),
            # Table V: 1.36, -1.70, 0.71
            BL3ZFit(1.3629101111921, -1.6996176062383, 0.70894108564687, 2.010),
            # Table V: 0.84, -2.57, 9.87
            BL3ZFit(0.83558637190456, -2.5698251471323, 9.8729125261567, 2.351),
            # Table V: (0.69), -0.90, 2.25
            BL3ZFit(a0_gpp, -0.90303820170714, 2.249274327598, 2.423),
            # Table V: (0.69), -0.68, 0.70
            BL3ZFit(a0_gpp, -0.68345507203321, 0.69589476713259, 2.423),
            # Table V: 0.73, -0.97, 0.83
            BL3ZFit(
                0.72755668638733, -0.96674724145661, 0.82644748408508, 1.870
            ),
            (1.870 + 0.135) * (1.870 + 0.135),
        )

    @inline(.always)
    def zw(self, q2: Float64, t0: Float64) -> Float64:
        var r = sqrt(self.tplus - q2) + sqrt(self.tplus - t0)
        return 1.0 / (r * r)


def bl3_density(
    M: Float64,
    mB: Float64,
    ml: Float64,
    EB: Float64,
    El: Float64,
    pol: Bool,
    ff: BL3FormFactors,
) -> Float64:
    """|M|^2 / (G_F^2 |V_cq|^2) at the Lambda_c-frame energies EB, El [GeV]
    (Lambda_c spin averaged, the others summed). With pol, it returns the
    density of l+ helicity +1/2 minus that of -1/2 in the Lambda_c frame. ml
    must be above 0, and the point must lie in the Dalitz plot."""
    var Mp = M + mB
    var Mm = M - mB
    var Enu = M - EB - El
    var sp = 2.0 * M * (EB + mB)
    var sm = 2.0 * M * (EB - mB)  # s_- = q^2_max - q^2
    var q2 = Mm * Mm - sm
    var w = ff.zw(q2, Mm * Mm)
    var z = sm * w
    var fp = ff.fplus.at(q2, z)
    var F = ff.fperp.at(q2, z)
    var f0 = ff.f0.at(q2, z)
    var gp = ff.gplus.at(q2, z)
    var G = ff.gperp.at(q2, z)
    var g0 = ff.g0.at(q2, z)
    # (g_+ - g_perp) / s_- without the 0/0 at q^2_max
    var a = Mp * (fp - F) / sp
    var b = Mm * ((f0 - fp) / q2 - a / Mp)
    var c = (
        Mm
        * ff.gplus.pole(q2)
        * ((ff.gplus.a1 - ff.gperp.a1) + z * (ff.gplus.a2 - ff.gperp.a2))
        * w
    )
    var d = Mp * ((g0 - gp) / q2 - c / Mm)
    var P2 = 2.0 * (M * M + mB * mB) - q2
    var Pq = Mp * Mm
    var FF2 = F * F
    var GG2 = G * G
    var Kg = -4.0 * (
        4.0 * M * mB * (FF2 - GG2)
        + sp
        * (
            2.0 * F * (a * Mp + b * Mm)
            + a * a * P2
            + 2.0 * a * b * Pq
            + b * b * q2
        )
        + sm
        * (
            2.0 * G * (c * Mm + d * Mp)
            + c * c * P2
            + 2.0 * c * d * Pq
            + d * d * q2
        )
    )
    var KPP = 8.0 * (
        FF2
        + GG2
        + 2.0 * F * a * Mp
        + 2.0 * G * c * Mm
        + a * a * sp
        + c * c * sm
    )
    var Kqq = -8.0 * (
        FF2
        + GG2
        + 2.0 * F * b * Mm
        + 2.0 * G * d * Mp
        - b * b * sp
        - d * d * sm
    )
    var KS = -16.0 * (
        F * (a * Mm - b * Mp) + G * (c * Mp - d * Mm) - a * b * sp - c * d * sm
    )
    var KA = -32.0 * F * G
    var k1k2 = 0.5 * (q2 - ml * ml)
    var Pk2 = 2.0 * M * Enu - k1k2
    var qk2 = k1k2
    var Pl: Float64
    var ql: Float64
    var lk2: Float64
    if not pol:
        ql = 0.5 * (q2 + ml * ml)
        Pl = 2.0 * M * El - ql
        lk2 = k1k2
    else:
        var pk = sqrt(max((El - ml) * (El + ml), 0.0))
        if pk <= 0.0:
            return 0.0
        lk2 = (El * k1k2 - ml * ml * Enu) / pk
        ql = lk2
        Pl = 2.0 * M * pk - lk2
    return (
        Kg * lk2
        + KPP * Pl * Pk2
        + Kqq * ql * qk2
        + 0.5 * KS * (Pl * qk2 + ql * Pk2)
        + 0.5 * KA * (Pl * qk2 - ql * Pk2)
    )
