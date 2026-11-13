"""PhotoHadron, the port of AM3's PhotoPion (PhotoPion.cc). PhotoHadron makes
the pions as AM3 does, then the kaons and the charm hadrons (am3_mojo
addition)."""

from std.math import ceil, fma
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import *
from am3_mojo.physics.decays import charm_mass
from am3_mojo.physics.photopion_tables import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.linalg import MatD, diagonal_gemv, gemv_d, gemv_d_par
from am3_mojo.utils.parallel import parallel_run
from am3_mojo.utils.specfun import number_split

comptime N_PG_CHANNELS = 28 + 9 + 14


# ------------------------------------------------------------------------------
# pion production (AM3): Hümmer et al.'s (2010) channels
# ------------------------------------------------------------------------------
@inline(.always)
def pg_Theta(x: Float64) -> Float64:
    return 1.0 if x >= 0.0 else 0.0


@inline(.always)
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
    var H: FPtr
    var res: FPtr
    var v: FPtr
    var nhk: Int
    var nh: Int
    var top: Int
    var vlo: Int
    var nv: Int
    var hlo: IPtr
    var hhi: IPtr
    var nch: Int

    def __init__(
        out self,
        H: FPtr,
        res: FPtr,
        v: FPtr,
        nhk: Int,
        nh: Int,
        top: Int,
        vlo: Int,
        nv: Int,
        hlo: IPtr,
        hhi: IPtr,
        nch: Int,
    ):
        self.H = H
        self.res = res
        self.v = v
        self.nhk = nhk
        self.nh = nh
        self.top = top
        self.vlo = vlo
        self.nv = nv
        self.hlo = hlo
        self.hhi = hhi
        self.nch = nch


def pg_conv_work(job: Pointer[PGConvJob, MutUntrackedOrigin], t: Int, nt: Int):
    ref J = job[]
    var c = t
    while c < J.nch:
        # the table row shifted by vlo, so that column 0 is target bin vlo
        diagonal_gemv[True](
            J.H.unsafe_offset(c * J.nhk + J.vlo),
            J.hlo[unsafe_offset=c] - J.vlo,
            J.hhi[unsafe_offset=c] - J.vlo,
            J.top,
            J.nv,
            J.v,
            1.0,
            J.res.unsafe_offset(c * J.nh),
        )
        for i in range(J.top, J.nh):
            J.res[unsafe_offset=c * J.nh + i] = 0.0
        c += nt


@inline(.always)
def _split_add[
    n: Int
](
    Q: Array[FPtr, n],
    Mp: Array[Float64, n],
    Mn: Array[Float64, n],
    q: Int,
    qr: Float64,
    f: Float64,
    res_c: FPtr,
    prot: FPtr,
    neut: FPtr,
    nh: Int,
):
    """AM3's split of the products of bin i between the two grid bins next to
    their energy, with the shares of number_split.

    The SIMD code over 8 bins gives bitwise the same result as a loop over i.
    Each bin q + i takes the share qr of bin i - 1 before the share 1 - qr of
    bin i. The code uses the same fused multiply-adds as the compiled scalar
    loop."""
    comptime W = WD
    var vqr = 1.0 - qr
    var i = max(0, -q)
    var iend = min(nh, nh - 1 - q)
    while i + W <= iend:
        var r = f * res_c.unsafe_offset(i).unsafe_load[width=W]()
        var lo = vqr * r
        var hi = qr * r
        var pv = prot.unsafe_offset(i).unsafe_load[width=W]()
        var nv = neut.unsafe_offset(i).unsafe_load[width=W]()
        comptime for o in range(n):
            var t = pv.fma(SIMD[DType.float64, W](Mp[o]), nv * Mn[o])
            var up = Q[o].unsafe_offset(q + i + 1)
            up.unsafe_store(hi.fma(t, up.unsafe_load[width=W]()))
            var dn = Q[o].unsafe_offset(q + i)
            dn.unsafe_store(lo.fma(t, dn.unsafe_load[width=W]()))
        i += W
    while i < iend:
        var r = f * res_c[unsafe_offset=i]
        var lo = vqr * r
        var hi = qr * r
        var pi_ = prot[unsafe_offset=i]
        var ni = neut[unsafe_offset=i]
        comptime for o in range(n):
            var t = fma(pi_, Mp[o], ni * Mn[o])
            Q[o][unsafe_offset=q + i] = fma(lo, t, Q[o][unsafe_offset=q + i])
            Q[o][unsafe_offset=q + i + 1] = fma(
                hi, t, Q[o][unsafe_offset=q + i + 1]
            )
        i += 1


# ------------------------------------------------------------------------------
# kaon production (am3_mojo addition)
# ------------------------------------------------------------------------------
struct KChannel(Copyable, Movable):
    """Kaon production channel in the framework of Hümmer et al. (2010).

    sigma [mubarn]: constant cross section for the photon energies
    emin - emax [GeV] in the nucleon rest frame. mtype >= 0: part of multi-pion
    type M(mtype + 1) of Hümmer et al.'s Table 6 (photopion_tables IT3 entries
    2 mtype and 2 mtype + 1), over all or part of the type's photon range."""

    var sigma: Float64
    var emin: Float64
    var emax: Float64
    var chi_p: Array[Float64, N_KAON]
    var chi_n: Array[Float64, N_KAON]
    var mp: Array[Float64, N_KAON]
    var mn: Array[Float64, N_KAON]
    var mtype: Int

    def __init__(
        out self,
        sigma: Float64,
        emin: Float64,
        emax: Float64,
        chi_p: Array[Float64, N_KAON],
        chi_n: Array[Float64, N_KAON],
        mp: Array[Float64, N_KAON],
        mn: Array[Float64, N_KAON],
        mtype: Int,
    ):
        self.sigma = sigma
        self.emin = emin
        self.emax = emax
        self.chi_p = chi_p.copy()
        self.chi_n = chi_n.copy()
        self.mp = mp.copy()
        self.mn = mn.copy()
        self.mtype = mtype

    def f(self, y: Float64) -> Float64:
        """Response function f(y) = 1/(2 y^2) * integral of eps sigma(eps)
        d eps up to eps = 2y (Hümmer et al. 2010, Eq. 27)."""
        var e2 = 4.0 * y * y
        if e2 < self.emin * self.emin:
            return 0.0
        var top = min(e2, self.emax * self.emax)
        return self.sigma * (top - self.emin * self.emin) / e2


def _species(
    kp: Float64, km: Float64, k0: Float64, k0bar: Float64
) -> Array[Float64, N_KAON]:
    """A K0 or anti-K0 is half K0_S, half K0_L."""
    var m: Array[Float64, N_KAON] = [
        kp,
        km,
        0.5 * (k0 + k0bar),
        0.5 * (k0 + k0bar),
    ]
    return m^


def _per_charge(
    k_charged: Float64, k_neutral: Float64
) -> Array[Float64, N_KAON]:
    var v: Array[Float64, N_KAON] = [
        k_charged,
        k_charged,
        k_neutral,
        k_neutral,
    ]
    return v^


def kaon_production_channels() -> List[KChannel]:
    """Kaon production in photo-hadronic interactions (derived in
    tests/literature_checks/kaons/README.md):

    * Two-body associated production N gamma -> K Y (Y = Lambda, Sigma) near
      threshold, in six steps of the photon energy eps_r in the nucleon rest
      frame: p -> K+ (K+ Lambda, K+ Sigma0) and K0 (K0 Sigma+), n -> K+
      (K+ Sigma-) and K0 (K0 Lambda, K0 Sigma0). Cross sections [mubarn]: step
      means of the data of SAPHIR (Glander et al. 2004; Lawall et al. 2005),
      CLAS (McCracken et al. 2010; Dey et al. 2010; Anefalos Pereira et al.
      2010; Compton et al. 2017) and A2 (Akondi et al. 2019). Beyond the data,
      K0 Sigma0 above 1.36 GeV follows K0 Sigma+, scaled to the A2 data. The
      first three steps refine the K+ channel of Hümmer et al. (2010,
      Sec. 4.5). The measured fall-off replaces its constant 2.7 mubarn up to
      the highest energies. chi: two-body kinematics (Hümmer et al. Eq. 16
      with m_p -> m_Y and m_pi -> m_K, s with m_N) at the kaon's measured
      mean CM angle. tests/support/measured/ky_angular.py computes both.
    * Kaons made with further particles (for example K Y pi and K anti-K N)
      in the multi-pion types M2 - M7 of Hümmer et al. (Table 6, same cross
      sections and energy ranges). Their multiplicities follow from
      strangeness balance and inclusive photo- and hadro-production data. A
      low part has chi = 1.6 x that of the type's low-energy pions. At the
      same rapidity, the energy of a kaon is the kaon/pion transverse-mass
      ratio at pT = 0.35 GeV times that of a pion. A leading part has the chi
      of the type's leading pions. The leading part takes a share h+ of the
      K+ and K0, close to that of the type's pions. It takes the share
      h- = 0.6 h+ of the K- and anti-K0, which share no quark with the
      nucleon (NA49, Anticic et al. 2010, Table 5: K+ and K- at x_F > 0.2 in
      pp collisions at 17.3 GeV). Neutrons: K+ <-> K0, K- <-> anti-K0.
      M2 (0.9 - 1.5 GeV) makes kaons from 1.15 GeV, the threshold of
      gamma N -> K Lambda pi (1.152 - 1.168 GeV for the four charge states).
      Their cross section lies at the middle of the measured three-body cross
      sections (SAPHIR, CBELSA/TAPS and ABBHHM, see
      tests/literature_checks/kaons/multibody_strangeness.py).
      M3 (1.5 - 5 GeV), where the yields rise most, has two steps
      (1.5 - 2.5, 2.5 - 5 GeV). The yields of the steps are the middle of the
      ranges from two sources, times 1.21 and 1.32. The sources are the
      strange-particle events of the ABBHHM bubble chamber (Erbe et al. 1969)
      and the cross sections measured with a hyperon and further particles at
      1.5 - 2.5 GeV (ABBHHM; SAPHIR, Schulday et al. 2010; CBELSA/TAPS,
      Nanova et al. 2008). The SLAC bubble chamber and DESY count more visible
      strange-particle decays than ABBHHM. The factors are the middle of the
      range of the three experiments, raised to the least that DESY allows
      (multibody_strangeness.py). M2 and M3 take h+ = 0.25, the share of M3's
      pions.
      M4 (5 - 50 GeV) also has two steps (5 - 15, 15 - 50 GeV). Their yields
      lie on the line in log eps_r through two points. These points are M3 at
      2.5 - 5 GeV and the kaons that balance the strange particles measured at
      20 GeV (Abe et al. 1985). The K0 share of the kaons with a Lambda comes
      from the K0_S Lambda pairs of that measurement. The chi of the low part
      is 1.8 and 0.93 x that of the type's low-energy pions. This gives the
      K0_S the mean energy fraction measured there (Abe et al. 1984), taken to
      the middle of each step (m4_steps.py).
      M5 - M7 (50 GeV - 5e6 GeV) have 0.070, 0.085 and 0.100 kaons per pion
      of the type before the energy bookkeeping scales its pions. These
      values are close to the kaons per pion of pp and p-pbar collisions at
      their energies (NA49; NA27, Aguilar-Benitez et al. 1991; UA5,
      Alner et al. 1985, Ansorge et al. 1987). The kaons exceed the antikaons
      by the hyperons per interaction, 0.08 - 0.10 (0.074 at 20 GeV). K+
      exceeds K0 by 0.015, about as much as the kaons that come with a hyperon
      at 20 GeV. K- exceeds anti-K0 by 0.005 (higher_types.py).
    """
    var c = List[KChannel]()
    # (emin, emax [GeV], then the chi of p -> K+, p -> K0, n -> K+, n -> K0,
    # then the cross sections [mubarn] of p -> K+, p -> K0, n -> K+, n -> K0)
    var ky: List[Array[Float64, 10]] = [
        [0.91, 1.05, 0.30, 0.30, 0.30, 0.34, 1.31, 0.0, 0.0, 0.81],
        [1.05, 1.25, 0.29, 0.32, 0.29, 0.35, 3.00, 0.31, 0.49, 1.78],
        [1.25, 1.65, 0.29, 0.30, 0.30, 0.33, 4.09, 0.55, 1.28, 1.89],
        [1.65, 2.2, 0.26, 0.29, 0.29, 0.29, 2.44, 0.26, 1.03, 0.80],
        [2.2, 3.0, 0.23, 0.30, 0.29, 0.27, 1.25, 0.083, 0.90, 0.28],
        [3.0, 5.0, 0.19, 0.23, 0.22, 0.19, 0.44, 0.030, 0.39, 0.10],
    ]
    for r in ky:
        # unit cross section, the per-species cross sections as multiplicities
        c.append(
            KChannel(
                1.0,
                r[0],
                r[1],
                _per_charge(r[2], r[3]),
                _per_charge(r[4], r[5]),
                _species(r[6], 0.0, r[7], 0.0),
                _species(r[8], 0.0, r[9], 0.0),
                -1,
            )
        )
    # multi-pion types (index t: M(t + 1)) in steps eps_min - eps_max [GeV] of
    # the photon energy: kaons per interaction from protons (K+, K0, K-,
    # anti-K0), leading fractions h+ (K+, K0) and h- (K-, anti-K0), and the
    # other kaons' energy fraction in units of the type's low-energy pions'
    var mpi: List[Array[Float64, 10]] = [
        [1.0, 1.15, 1.5, 0.0051, 0.0060, 0.0, 0.0, 0.25, 0.15, 1.6],
        [2.0, 1.5, 2.5, 0.0278, 0.0206, 0.0030, 0.0034, 0.25, 0.15, 1.6],
        [2.0, 2.5, 5.0, 0.0585, 0.0449, 0.0214, 0.0201, 0.25, 0.15, 1.6],
        [3.0, 5.0, 15.0, 0.0935, 0.0687, 0.0468, 0.0461, 0.15, 0.09, 1.8],
        [3.0, 15.0, 50.0, 0.1316, 0.0988, 0.0794, 0.0796, 0.15, 0.09, 0.93],
        [4.0, 50.0, 500.0, 0.185, 0.170, 0.140, 0.135, 0.07, 0.04, 1.6],
        [5.0, 500.0, 5000.0, 0.350, 0.335, 0.300, 0.295, 0.05, 0.03, 1.6],
        [6.0, 5000.0, 5.0e6, 0.575, 0.560, 0.520, 0.515, 0.035, 0.02, 1.6],
    ]
    for r in mpi:
        var t = Int(r[0])
        var kp = r[3]
        var k0 = r[4]
        var km = r[5]
        var k0b = r[6]
        for lead in range(2):
            var it = 2 * t + lead  # IT3 entry: L (lead = 0) or H (lead = 1)
            var hp = r[7] if lead == 1 else 1.0 - r[7]
            var hm = r[8] if lead == 1 else 1.0 - r[8]
            var chi = IT3kai(it) if lead == 1 else r[9] * IT3kai(it)
            c.append(
                KChannel(
                    IT3sig(it),
                    r[1],
                    r[2],
                    _per_charge(chi, chi),
                    _per_charge(chi, chi),
                    _species(hp * kp, hm * km, hp * k0, hm * k0b),
                    _species(hp * k0, hm * k0b, hp * kp, hm * km),
                    t,
                )
            )
    return c^


# ------------------------------------------------------------------------------
# charm-hadron production (am3_mojo addition)
# ------------------------------------------------------------------------------
struct CPart(Copyable, Movable):
    var chi_p: Array[Float64, N_CHARM]
    var chi_n: Array[Float64, N_CHARM]
    var mp: Array[Float64, N_CHARM]
    var mn: Array[Float64, N_CHARM]

    def __init__(
        out self,
        chi_p: Array[Float64, N_CHARM],
        chi_n: Array[Float64, N_CHARM],
        mp: Array[Float64, N_CHARM],
        mn: Array[Float64, N_CHARM],
    ):
        self.chi_p = chi_p.copy()
        self.chi_n = chi_n.copy()
        self.mp = mp.copy()
        self.mn = mn.copy()


struct CStep(Copyable, Movable):
    """A step emin - emax [GeV] of the photon energy eps_r in the nucleon rest
    frame, inside multi-pion type M(mtype + 1) of Hümmer et al.'s Table 6.
    The step has the cross section sigma [mubarn] of that type. The charm
    hadrons of its interactions come in parts with their own energy
    fractions. The parts share the response function of the step (one photon
    convolution per step)."""

    var sigma: Float64
    var emin: Float64
    var emax: Float64
    var mtype: Int
    var parts: List[CPart]

    def __init__(
        out self, sigma: Float64, emin: Float64, emax: Float64, mtype: Int
    ):
        self.sigma = sigma
        self.emin = emin
        self.emax = emax
        self.mtype = mtype
        self.parts = List[CPart]()

    def f(self, y: Float64) -> Float64:
        """Response function (Hümmer et al. 2010, Eq. 27) of the step's constant
        cross section."""
        var e2 = 4.0 * y * y
        if e2 < self.emin * self.emin:
            return 0.0
        var top = min(e2, self.emax * self.emax)
        return self.sigma * (top - self.emin * self.emin) / e2


def _pairs(n: Float64) -> Array[Float64, N_CHARM]:
    """Hadrons of n c c-bar pairs that fragment alike: each species and its
    antiparticle with its fraction."""
    var m: Array[Float64, N_CHARM] = [
        n * FRAG_D,
        n * FRAG_D,
        n * FRAG_D0,
        n * FRAG_D0,
        n * FRAG_DS,
        n * FRAG_DS,
        n * FRAG_LC,
        n * FRAG_LC,
    ]
    return m^


def _by_kind(
    chi_d: Float64, chi_ds: Float64, chi_lc: Float64
) -> Array[Float64, N_CHARM]:
    var v: Array[Float64, N_CHARM] = [
        chi_d,
        chi_d,
        chi_d,
        chi_d,
        chi_ds,
        chi_ds,
        chi_lc,
        chi_lc,
    ]
    return v^


def charm_production_steps() -> List[CStep]:
    """Charm-hadron production in photo-hadronic interactions (derived in
    tests/literature_checks/charm/README.md). The production comes in steps of
    the photon energy eps_r inside the multi-pion types M4 - M7 of
    Hümmer et al. (2010, Table 6), with their cross sections. A row of `cs`:
    [t, eps_min, eps_max, R, h_a, h_N, chi of the D, Ds and Lambda_c on the
    photon side, chi of the anti-D of associated production, chi on the
    nucleon side]. R: c c-bar pairs per interaction. A share h_a of these
    pairs is made with a Lambda_c+ (associated production,
    gamma N -> Lambda_c+ anti-D X). A share h_N is made on the nucleon side
    (resolved photons), and the rest on the photon side. The hadrons of the
    photon and nucleon sides fragment as in `_pairs`, alike for protons and
    neutrons. The associated Lambda_c+ has the energy fraction of the type's
    outgoing nucleon. It comes in addition to that nucleon (at most 4e-4 per
    interaction). Like the energy of the other charm hadrons, its energy is
    taken from the pions. Its anti-D is an anti-D0 or a D-. From protons, the
    two come in the ratio of their fragmentation fractions, and from neutrons
    the other way round.
    tests/literature_checks/charm/charm_production.py recomputes the rows
    from the measured data (tests/support/measured/charm_photoproduction.py).
    """
    var cs: List[Array[Float64, 11]] = [
        [3.0, 8.71, 15.0, 0.0001049, 1.0, 0.0, 0.0, 0.0, 0.0, 0.3027, 0.0],
        [
            3.0,
            15.0,
            30.0,
            0.0006556,
            0.5884,
            0.0,
            0.1961,
            0.2093,
            0.2486,
            0.2166,
            0.0,
        ],
        [
            3.0,
            30.0,
            50.0,
            0.0016,
            0.2182,
            0.0,
            0.1423,
            0.1511,
            0.1788,
            0.1468,
            0.0,
        ],
        [
            4.0,
            50.0,
            150.0,
            0.00371,
            0.05347,
            0.006119,
            0.08039,
            0.08518,
            0.1007,
            0.09086,
            0.1266,
        ],
        [
            4.0,
            150.0,
            500.0,
            0.008741,
            0.007343,
            0.03075,
            0.03723,
            0.03941,
            0.04655,
            0.04273,
            0.08143,
        ],
        [
            5.0,
            500.0,
            1500.0,
            0.01733,
            0.0009727,
            0.05628,
            0.01733,
            0.01831,
            0.02153,
            0.01938,
            0.07122,
        ],
        [
            5.0,
            1500.0,
            5000.0,
            0.03176,
            0.0001304,
            0.08246,
            0.008173,
            0.00862,
            0.01008,
            0.009288,
            0.05159,
        ],
        [
            6.0,
            5000.0,
            5.0e4,
            0.07543,
            7.42e-6,
            0.1275,
            0.002672,
            0.002812,
            0.003263,
            0.003956,
            0.03065,
        ],
        [
            6.0,
            5.0e4,
            5.0e5,
            0.1349,
            1.588e-7,
            0.1468,
            0.0007218,
            0.0007578,
            0.0008739,
            0.00102,
            0.02012,
        ],
        [
            6.0,
            5.0e5,
            5.0e6,
            0.1799,
            2.75e-9,
            0.1468,
            0.0001895,
            0.0001987,
            0.0002282,
            0.0002654,
            0.01434,
        ],
    ]
    var aD0 = FRAG_D0 / (FRAG_D0 + FRAG_D)
    var steps = List[CStep]()
    for r in cs:
        var t = Int(r[0])
        var R = r[3]
        var h_a = r[4]
        var h_N = r[5]
        var st = CStep(IT3sig(2 * t), r[1], r[2], t)
        if 1.0 - h_a - h_N > 0.0:
            var photon = _pairs(R * (1.0 - h_a - h_N))
            var chi_ph = _by_kind(r[6], r[7], r[8])
            st.parts.append(CPart(chi_ph, chi_ph, photon, photon))
        if h_N > 0.0:
            var nucleon = _pairs(R * h_N)
            var chi_nu = _by_kind(r[10], r[10], r[10])
            st.parts.append(CPart(chi_nu, chi_nu, nucleon, nucleon))
        if h_a > 0.0:
            var a = R * h_a
            var zero = _pairs(0.0)
            var mp = zero.copy()
            var mn = zero.copy()
            mp[CD0B] = a * aD0
            mp[CDM] = a * (1.0 - aD0)
            mn[CD0B] = a * (1.0 - aD0)
            mn[CDM] = a * aD0
            var chi_ad = _by_kind(r[9], r[9], r[9])
            st.parts.append(CPart(chi_ad, chi_ad, mp, mn))
            var lc = zero.copy()
            lc[CLC] = a
            var chi_lc = _by_kind(IT3kaiPN(2 * t), 0.0, IT3kaiPN(2 * t))
            st.parts.append(CPart(chi_lc, chi_lc, lc, lc))
        steps.append(st^)
    return steps^


struct PGCharmJob(Movable):
    var P: Pointer[PhotoHadron, MutUntrackedOrigin]
    var dp: FPtr
    var dn: FPtr
    var nh: Int
    var dlnE: Float64
    var f: Float64

    def __init__(
        out self,
        P: Pointer[PhotoHadron, MutUntrackedOrigin],
        dp: FPtr,
        dn: FPtr,
        nh: Int,
        dlnE: Float64,
        f: Float64,
    ):
        self.P = P
        self.dp = dp
        self.dn = dn
        self.nh = nh
        self.dlnE = dlnE
        self.f = f


def pgc_work(job: Pointer[PGCharmJob, MutUntrackedOrigin], t: Int, nt: Int):
    """The sum over the steps, their parts and the two nucleons, in that order
    (as in a single thread). A hadron takes the energy chi E_N, as from a
    nucleon of beta -> 1. If that energy is below the mass m of the hadron, the
    hadron goes to m instead. Such hadrons come from the slow nucleons that
    target photons above the default limit of
    set_optimize_photopion_target_photon_max reach. number_split shares them
    between the bins around m and keeps their number and energy."""
    ref P = job[].P[]
    var nh = job[].nh
    var f = job[].f
    var dl = job[].dlnE
    var s = t
    while s < N_CHARM:
        var QC0 = pd(P.qC[s])
        # the mass, in hadron bins (egrid_had[i] = exp(i dlnE) GeV)
        var xm = c_log(charm_mass(s)) / dl
        var l: Int
        var lr: Float64
        l, lr = number_split(xm, dl)
        for c in range(len(P.csteps)):
            var r = pd(P.resC).unsafe_offset(c * nh)
            for k in range(len(P.csteps[c].parts)):
                ref part = P.csteps[c].parts[k]
                for nucleon in range(2):
                    var m = part.mp[s] if nucleon == 0 else part.mn[s]
                    if m == 0.0:
                        continue
                    var dens = job[].dp if nucleon == 0 else job[].dn
                    var chi = part.chi_p[s] if nucleon == 0 else part.chi_n[s]
                    var q: Int
                    var qr: Float64
                    q, qr = number_split(c_log(chi) / dl, dl)
                    var im = min(Int(ceil(xm - c_log(chi) / dl)), nh)
                    if l >= 0 and l + 1 < nh:
                        var mlo = (1.0 - lr) * m * f
                        var mhi = lr * m * f
                        for i in range(im):
                            var a = r[unsafe_offset=i] * dens[unsafe_offset=i]
                            QC0[unsafe_offset=l] += mlo * a
                            QC0[unsafe_offset=l + 1] += mhi * a
                    var i0 = max(0, -q, im)
                    var i1 = min(nh, nh - 1 - q)
                    var lo = (1.0 - qr) * m * f
                    var hi = qr * m * f
                    var QC = QC0.unsafe_offset(q)
                    for i in range(i0, i1):
                        var a = r[unsafe_offset=i] * dens[unsafe_offset=i]
                        QC[unsafe_offset=i] += lo * a
                        QC[unsafe_offset=i + 1] += hi * a
        s += nt


struct PhotoHadron(Copyable, Movable):
    var verbose: Bool
    var QPG_PROGen: Vec
    var QPG_NTRGen: Vec
    var afPG_PRO: Vec
    var qpi0: Vec
    var qpip: Vec
    var qpim: Vec
    var afPG_Rad: Vec
    # Pion-production channels (28 resonance, 9 direct, 14 multi-pion). A kernel
    # entry depends only on m = i + j (proton bin i, target photon bin j),
    # through gamma_p * eps = exp(m dlnE) / m_p (Hankel form).
    var ncols: Int
    var nhk: Int
    var H: Vec
    var hlo: VecI
    var hhi: VecI
    var nthreads: Int
    var res: Vec
    var KER_FPGapn: MatD
    var KER_FPGara: MatD
    var photon_norm: Float64
    var w_vphot: Vec
    var w_in: Vec
    var kaons: Bool
    var kaons_built: Bool
    var kchan: List[KChannel]
    var HK: Vec
    var hloK: VecI
    var hhiK: VecI
    var resK: Vec
    var qK: List[Vec]
    var pion_cut_p: Vec
    var pion_cut_n: Vec
    var w_res_p: Vec
    var w_res_n: Vec
    var charm: Bool
    var charm_built: Bool
    var csteps: List[CStep]
    var HC: Vec
    var hloC: VecI
    var hhiC: VecI
    var resC: Vec
    var qC: List[Vec]
    var cut_C_p: Vec
    var cut_C_n: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.QPG_PROGen = Vec()
        self.QPG_NTRGen = Vec()
        self.afPG_PRO = Vec()
        self.qpi0 = Vec()
        self.qpip = Vec()
        self.qpim = Vec()
        self.afPG_Rad = Vec()
        self.ncols = 0
        self.nhk = 0
        self.H = Vec()
        self.hlo = VecI()
        self.hhi = VecI()
        self.nthreads = 1
        self.res = zeros(dat.nbins_had * N_PG_CHANNELS)
        self.KER_FPGapn = MatD()
        self.KER_FPGara = MatD()
        self.photon_norm = 0.0
        self.w_vphot = Vec()
        self.w_in = Vec()
        self.kaons = False
        self.kaons_built = False
        self.kchan = kaon_production_channels()
        self.HK = Vec()
        self.hloK = VecI()
        self.hhiK = VecI()
        self.resK = Vec()
        self.qK = List[Vec]()
        for _ in range(N_KAON):
            self.qK.append(Vec())
        self.pion_cut_p = Vec()
        self.pion_cut_n = Vec()
        self.w_res_p = Vec()
        self.w_res_n = Vec()
        self.charm = False
        self.charm_built = False
        self.csteps = charm_production_steps()
        self.HC = Vec()
        self.hloC = VecI()
        self.hhiC = VecI()
        self.resC = Vec()
        self.qC = List[Vec]()
        for _ in range(N_CHARM):
            self.qC.append(Vec())
        self.cut_C_p = Vec()
        self.cut_C_n = Vec()
        self._set_pion_cuts()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.QPG_PROGen, dat.nbins_had)
        resize_zero(self.QPG_NTRGen, dat.nbins_had)
        resize_zero(self.afPG_PRO, dat.nbins_had)
        resize_zero(self.qpi0, dat.nbins_had)
        resize_zero(self.qpip, dat.nbins_had)
        resize_zero(self.qpim, dat.nbins_had)
        resize_zero(self.afPG_Rad, dat.nbins_pho)
        for s in range(N_KAON):
            resize_zero(self.qK[s], dat.nbins_had)
        for s in range(N_CHARM):
            resize_zero(self.qC[s], dat.nbins_had)

    def reinitialize_empty_kernels(mut self, dat: AM3Arrays):
        self.ncols = 0  # (no target photons until initialize_kernels)
        if len(self.res) != dat.nbins_had * N_PG_CHANNELS:
            self.res = zeros(dat.nbins_had * N_PG_CHANNELS)
        # (no loss kernels until initialize_kernels: the losses are zero
        #  without them)
        self.KER_FPGapn = MatD()
        self.KER_FPGara = MatD()
        # no kaon kernels until initialize_kernels builds them for this grid
        self.HK = Vec()
        self.hloK = VecI()
        self.hhiK = VecI()
        self.resK = Vec()
        self.kaons_built = False
        self.set_kaons(False)
        self.HC = Vec()
        self.hloC = VecI()
        self.hhiC = VecI()
        self.resC = Vec()
        self.charm_built = False
        self.set_charm(False)

    def initialize_kernels(
        mut self,
        dat: AM3Arrays,
        photon_in_max: Int,
        build_kaons: Bool,
        build_charm: Bool,
    ):
        """All the kernels, whatever the emission, cooling and photon-loss
        switches. The re-injection of the nucleons, part of their cooling,
        takes the channels of the products. A switch turned on after
        init_kernels finds its kernel. (AM3 built each kernel only with its
        switch on. If emission was off at init_kernels, the cooling removed the
        interacting nucleons. If cooling was switched on later, the
        re-injection created them.) The nucleon and photon losses take the
        target photons of the products, below photon_in_max. (AM3 took them
        all. So the nucleons that interact with photons above the limit
        vanished without products.)"""
        var nh = dat.nbins_had
        var nc = min(photon_in_max, dat.nbins_pho_hadgrid)
        if self.verbose:
            print(
                (
                    "computing p-gamma to pion generation kernel (Hankel"
                    " form), photon_in_max: "
                ),
                photon_in_max,
                sep="",
            )
        self.ncols = nc
        self.nhk = nh + nc - 1
        self.H = zeros(N_PG_CHANNELS * self.nhk)
        self.hlo = zerosi(N_PG_CHANNELS)
        self.hhi = zerosi(N_PG_CHANNELS)
        for c in range(N_PG_CHANNELS):
            PhotoHadron._fill_channel(
                dat, self.nhk, c, -1, self.kchan, self.H, self.hlo, self.hhi
            )
        self.KER_FPGapn = MatD(dat.nbins_had, dat.nbins_pho)
        self.KER_FPGara = MatD(dat.nbins_pho, dat.nbins_had)
        var co = C_LIGHT * dat.dlnE * MICROBARN
        if self.verbose:
            print("computing p-gamma proton/neutron loss kernel")
        for i in range(dat.nbins_had):
            for j in range(nc):
                var yg = (dat.egrid_had[i] / MASS_P) * dat.egrid_had_photons[j]
                self.KER_FPGapn.set(i, j, self._loss_kernel(yg, co))
        if self.verbose:
            print("computing p-gamma photon loss kernel")
        for j in range(nc):
            for i in range(dat.nbins_had):
                var yg = (dat.egrid_had[i] / MASS_P) * dat.egrid_had_photons[j]
                self.KER_FPGara.set(
                    j,
                    i,
                    self.KER_FPGara.get(j, i) + self._loss_kernel(yg, co),
                )
        if build_kaons:
            var nk = len(self.kchan)
            self.HK = zeros(nk * self.nhk)
            self.hloK = zerosi(nk)
            self.hhiK = zerosi(nk)
            self.resK = zeros(nk * nh)
            for c in range(nk):
                PhotoHadron._fill_channel(
                    dat,
                    self.nhk,
                    c,
                    c,
                    self.kchan,
                    self.HK,
                    self.hloK,
                    self.hhiK,
                )
            self.kaons_built = True
        if build_charm:
            var nst = len(self.csteps)
            self.HC = zeros(nst * self.nhk)
            self.hloC = zerosi(nst)
            self.hhiC = zerosi(nst)
            self.resC = zeros(nst * nh)
            for c in range(nst):
                var lo = self.nhk
                var hi = 0
                for m in range(self.nhk):
                    var i = min(m, nh - 1)
                    var y = (dat.egrid_had[i] / MASS_P) * dat.egrid_had_photons[
                        m - i
                    ]
                    var v = self.csteps[c].f(y)
                    self.HC[c * self.nhk + m] = v
                    if v != 0.0:
                        lo = min(lo, m)
                        hi = m + 1
                self.hloC[c] = lo
                self.hhiC[c] = hi
            self.charm_built = True

    @staticmethod
    def channel_value(c: Int, y: Float64) -> Float64:
        if c < 28:
            return IT1Er(c) * IT1Bout(c) * IT1G(c) * pg_fres(c, y)
        elif c < 37:
            return pg_fdir(c - 28, y)
        return IT3sig(c - 37) * pg_fmult(c - 37, y)

    @staticmethod
    def _fill_channel(
        dat: AM3Arrays,
        nhk: Int,
        c: Int,
        kaon: Int,
        kchan: List[KChannel],
        mut H: Vec,
        mut hlo: VecI,
        mut hhi: VecI,
    ):
        var nh = dat.nbins_had
        var lo = nhk
        var hi = 0
        for m in range(nhk):
            var i = min(m, nh - 1)
            var y = (dat.egrid_had[i] / MASS_P) * dat.egrid_had_photons[m - i]
            var v = kchan[kaon].f(
                y
            ) if kaon >= 0 else PhotoHadron.channel_value(c, y)
            H[c * nhk + m] = v
            if v != 0.0:
                lo = min(lo, m)
                hi = m + 1
        hlo[c] = lo
        hhi[c] = hi

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

    def Calc_PhotoHadron_ProtonNeutronLoss(mut self, dat: AM3Arrays):
        """The nucleons' loss rate on the target photons of the products, those
        below the target-photon limit of the kernels."""
        if self.verbose:
            print("computing photo-hadron proton/neutron cooling term")
        if self.KER_FPGapn.rows == 0:
            fill_zero(self.afPG_PRO)
        else:
            gemv_d_par(
                self.KER_FPGapn,
                pd(dat.n[P_photons_on_had_grid]),
                1.0,
                pd(self.afPG_PRO),
                self.nthreads,
            )

    def Calc_PhotoHadron_PhotonLoss(mut self, dat: AM3Arrays):
        """The photon loss rate, on the target photons of the products (as
        Calc_PhotoHadron_ProtonNeutronLoss)."""
        if self.verbose:
            print("computing photo-hadron photon loss term")
        if self.KER_FPGara.rows == 0:
            fill_zero(self.afPG_Rad)
            return
        resize_zero(self.w_in, dat.nbins_had)
        for i in range(dat.nbins_had):
            self.w_in[i] = dat.n[P_protons][i] + dat.n[P_neutrons][i]
        gemv_d(self.KER_FPGara, pd(self.w_in), 1.0, pd(self.afPG_Rad))

    @staticmethod
    def nonzero_top(v: Vec) -> Int:
        var top = len(v)
        while top > 0 and v[top - 1] == 0.0:
            top -= 1
        return top

    def Calc_PhotoHadron_PhotonConvolution(
        mut self, dat: AM3Arrays, src_top: Int = 0
    ):
        """The responses cover the nucleon bins up to the highest that holds
        protons or neutrons. They also cover the bins below src_top, where the
        sources of the step put nucleons later. Above these bins, res is zero,
        because the products take res[c, i] times the nucleons of bin i."""
        if self.verbose:
            print("computing photo-hadron photon-convolution")
        var nc = self.ncols
        self.photon_norm = 1.0e-50
        for k in range(nc):
            self.photon_norm += dat.n[P_photons_on_had_grid][k]
        resize_zero(self.w_vphot, nc)
        for k in range(nc):
            self.w_vphot[k] = dat.n[P_photons_on_had_grid][k] / self.photon_norm
        var vlo: Int
        var vhi: Int
        vlo, vhi = nonzero_range(pd(self.w_vphot), nc)
        if vhi <= vlo:
            fill_zero(self.res)
            fill_zero(self.resK)
            fill_zero(self.resC)
            return
        var top = max(
            src_top,
            PhotoHadron.nonzero_top(dat.n[P_protons]),
            PhotoHadron.nonzero_top(dat.n[P_neutrons]),
        )
        var job = PGConvJob(
            pd(self.H),
            pd(self.res),
            pd(self.w_vphot).unsafe_offset(vlo),
            self.nhk,
            dat.nbins_had,
            top,
            vlo,
            vhi - vlo,
            pi_(self.hlo),
            pi_(self.hhi),
            N_PG_CHANNELS,
        )
        parallel_run[pg_conv_work](job, self.nthreads)
        if self.kaons and len(self.resK) == len(self.kchan) * dat.nbins_had:
            var jobK = PGConvJob(
                pd(self.HK),
                pd(self.resK),
                pd(self.w_vphot).unsafe_offset(vlo),
                self.nhk,
                dat.nbins_had,
                top,
                vlo,
                vhi - vlo,
                pi_(self.hloK),
                pi_(self.hhiK),
                len(self.kchan),
            )
            parallel_run[pg_conv_work](
                jobK, min(self.nthreads, len(self.kchan))
            )
        var nst = len(self.csteps)
        if self.charm and len(self.resC) == nst * dat.nbins_had:
            var jobC = PGConvJob(
                pd(self.HC),
                pd(self.resC),
                pd(self.w_vphot).unsafe_offset(vlo),
                self.nhk,
                dat.nbins_had,
                top,
                vlo,
                vhi - vlo,
                pi_(self.hloC),
                pi_(self.hhiC),
                nst,
            )
            parallel_run[pg_conv_work](jobC, min(self.nthreads, nst))

    def _reinj_channel(
        mut self,
        dat: AM3Arrays,
        kaiPN: Float64,
        res_c: FPtr,
        MpPRO: Float64,
        MnPRO: Float64,
        MpNTR: Float64,
        MnNTR: Float64,
        co: Float64,
    ):
        var pp = c_log(kaiPN) / dat.dlnE
        var p: Int
        var pr: Float64
        p, pr = number_split(pp, dat.dlnE)
        var Q: Array[FPtr, 2] = [pd(self.QPG_PROGen), pd(self.QPG_NTRGen)]
        var Mp: Array[Float64, 2] = [MpPRO, MpNTR]
        var Mn: Array[Float64, 2] = [MnPRO, MnNTR]
        _split_add[2](
            Q,
            Mp,
            Mn,
            p,
            pr,
            co * self.photon_norm,
            res_c,
            pd(dat.n[P_protons]),
            pd(dat.n[P_neutrons]),
            dat.nbins_had,
        )

    def Calc_PhotoHadron_ProtonNeutronReinj(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing photo-hadron proton/neutron reinjection term")
        var co = C_LIGHT * dat.dlnE * MICROBARN
        for i in range(dat.nbins_had):
            self.QPG_PROGen[i] = 0.0
            self.QPG_NTRGen[i] = 0.0
        var nh = dat.nbins_had
        var r = pd(self.res)
        # (serial channel order: the C++ OpenMP loop races on these sums)
        for it in range(28):
            self._reinj_channel(
                dat,
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
        var Q: Array[FPtr, 3] = [pd(self.qpi0), pd(self.qpip), pd(self.qpim)]
        var Mp: Array[Float64, 3] = [MpPi0, MpPip, MpPim]
        var Mn: Array[Float64, 3] = [MnPi0, MnPip, MnPim]
        _split_add[3](
            Q,
            Mp,
            Mn,
            q,
            qr,
            co * self.photon_norm,
            res_c,
            pd(dat.n[P_protons]),
            pd(dat.n[P_neutrons]),
            dat.nbins_had,
        )

    def _rad_channel_pn(
        mut self,
        dat: AM3Arrays,
        kai: Float64,
        res_p: FPtr,
        res_n: FPtr,
        MpPi0: Float64,
        MnPi0: Float64,
        MpPip: Float64,
        MnPip: Float64,
        MpPim: Float64,
        MnPim: Float64,
        co: Float64,
    ):
        var q: Int
        var qr: Float64
        q, qr = number_split(c_log(kai) / dat.dlnE, dat.dlnE)
        var nh = dat.nbins_had
        var Q: Array[FPtr, 3] = [pd(self.qpi0), pd(self.qpip), pd(self.qpim)]
        var Mp: Array[Float64, 3] = [MpPi0, MpPip, MpPim]
        var Mn: Array[Float64, 3] = [MnPi0, MnPip, MnPim]
        var prot = pd(dat.n[P_protons])
        var neut = pd(dat.n[P_neutrons])
        var f = co * self.photon_norm
        var vqr = 1.0 - qr
        # SIMD over 8 bins, as in _split_add. Bin q + i takes the share qr of
        # bin i - 1 before the share 1 - qr of bin i, as in a loop over i.
        comptime W = WD
        var i = max(0, -q)
        var iend = min(nh, nh - 1 - q)
        while i + W <= iend:
            var wp = (
                f * res_p.unsafe_offset(i).unsafe_load[width=W]()
            ) * prot.unsafe_offset(i).unsafe_load[width=W]()
            var wn = (
                f * res_n.unsafe_offset(i).unsafe_load[width=W]()
            ) * neut.unsafe_offset(i).unsafe_load[width=W]()
            comptime for o in range(3):
                var t = wp.fma(SIMD[DType.float64, W](Mp[o]), wn * Mn[o])
                var up = Q[o].unsafe_offset(q + i + 1)
                up.unsafe_store(
                    SIMD[DType.float64, W](qr).fma(t, up.unsafe_load[width=W]())
                )
                var dn = Q[o].unsafe_offset(q + i)
                dn.unsafe_store(
                    SIMD[DType.float64, W](vqr).fma(
                        t, dn.unsafe_load[width=W]()
                    )
                )
            i += W
        while i < iend:
            var wp = f * res_p[unsafe_offset=i] * prot[unsafe_offset=i]
            var wn = f * res_n[unsafe_offset=i] * neut[unsafe_offset=i]
            comptime for o in range(3):
                var t = fma(wp, Mp[o], wn * Mn[o])
                Q[o][unsafe_offset=q + i] = fma(
                    vqr, t, Q[o][unsafe_offset=q + i]
                )
                Q[o][unsafe_offset=q + i + 1] = fma(
                    qr, t, Q[o][unsafe_offset=q + i + 1]
                )
            i += 1

    def Calc_PhotoHadron_Rad(mut self, dat: AM3Arrays):
        if self.verbose:
            print("computing photo-hadron pion-generation term")
        var co = C_LIGHT * dat.dlnE * MICROBARN
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
        var with_kaons = self.kaons and len(self.resK) == len(self.kchan) * nh
        var with_charm = self.charm and len(self.resC) == len(self.csteps) * nh
        if with_kaons or with_charm:
            resize_zero(self.w_res_p, nh)
            resize_zero(self.w_res_n, nh)
        for it in range(14):
            var res_it = r.unsafe_offset((37 + it) * nh)
            if not with_kaons and not with_charm:
                self._rad_channel(
                    dat,
                    IT3kai(it),
                    res_it,
                    IT3MpPi0(it),
                    IT3MnPi0(it),
                    IT3MpPip(it),
                    IT3MnPip(it),
                    IT3MpPim(it),
                    IT3MnPim(it),
                    co,
                )
                continue
            # the type's kaons take their energy from its pions: fewer pions in
            # each kaon channel's photon range
            var rp = pd(self.w_res_p)
            var rn = pd(self.w_res_n)
            for i in range(nh):
                rp[unsafe_offset=i] = res_it[unsafe_offset=i]
                rn[unsafe_offset=i] = res_it[unsafe_offset=i]
            for c in range(len(self.kchan)):
                if not with_kaons or self.kchan[c].mtype != it // 2:
                    continue
                var rK = pd(self.resK).unsafe_offset(c * nh)
                var cp = self.pion_cut_p[c]
                var cn = self.pion_cut_n[c]
                for i in range(nh):
                    rp[unsafe_offset=i] -= cp * rK[unsafe_offset=i]
                    rn[unsafe_offset=i] -= cn * rK[unsafe_offset=i]
            # (and the charm hadrons, likewise per charm step)
            for c in range(len(self.csteps)):
                if not with_charm or self.csteps[c].mtype != it // 2:
                    continue
                var rC = pd(self.resC).unsafe_offset(c * nh)
                var cp = self.cut_C_p[c]
                var cn = self.cut_C_n[c]
                for i in range(nh):
                    rp[unsafe_offset=i] -= cp * rC[unsafe_offset=i]
                    rn[unsafe_offset=i] -= cn * rC[unsafe_offset=i]
            self._rad_channel_pn(
                dat,
                IT3kai(it),
                rp,
                rn,
                IT3MpPi0(it),
                IT3MnPi0(it),
                IT3MpPip(it),
                IT3MnPip(it),
                IT3MpPim(it),
                IT3MnPim(it),
                co,
            )

    def _set_pion_cuts(mut self):
        """Hümmer et al.'s multi-pion types put the whole inelastic energy into
        pions (their fractions reproduce SOPHIA's). The kaons of channel c of
        type t take E_K = sum chi * M_K of the nucleon energy. The pions of the
        type take E_pi = sum chi * M_pi. To keep the energy,
        Calc_PhotoHadron_Rad leaves out the share E_K / E_pi of the type's
        pions in the channel's photon energies."""
        var n = len(self.kchan)
        self.pion_cut_p = zeros(n)
        self.pion_cut_n = zeros(n)
        for c in range(n):
            var t = self.kchan[c].mtype
            if t < 0:
                continue
            var e_pi_p = 0.0
            var e_pi_n = 0.0
            for it in range(2 * t, 2 * t + 2):
                e_pi_p += IT3kai(it) * (
                    IT3MpPi0(it) + IT3MpPip(it) + IT3MpPim(it)
                )
                e_pi_n += IT3kai(it) * (
                    IT3MnPi0(it) + IT3MnPip(it) + IT3MnPim(it)
                )
            var e_k_p = 0.0
            var e_k_n = 0.0
            for s in range(N_KAON):
                e_k_p += self.kchan[c].chi_p[s] * self.kchan[c].mp[s]
                e_k_n += self.kchan[c].chi_n[s] * self.kchan[c].mn[s]
            self.pion_cut_p[c] = e_k_p / e_pi_p
            self.pion_cut_n[c] = e_k_n / e_pi_n
        var nst = len(self.csteps)
        self.cut_C_p = zeros(nst)
        self.cut_C_n = zeros(nst)
        for c in range(nst):
            var t = self.csteps[c].mtype
            var e_pi_p = 0.0
            var e_pi_n = 0.0
            for it in range(2 * t, 2 * t + 2):
                e_pi_p += IT3kai(it) * (
                    IT3MpPi0(it) + IT3MpPip(it) + IT3MpPim(it)
                )
                e_pi_n += IT3kai(it) * (
                    IT3MnPi0(it) + IT3MnPip(it) + IT3MnPim(it)
                )
            var e_c_p = 0.0
            var e_c_n = 0.0
            for k in range(len(self.csteps[c].parts)):
                ref part = self.csteps[c].parts[k]
                for s in range(N_CHARM):
                    e_c_p += part.chi_p[s] * part.mp[s]
                    e_c_n += part.chi_n[s] * part.mn[s]
            self.cut_C_p[c] = e_c_p / e_pi_p
            self.cut_C_n[c] = e_c_n / e_pi_n

    # kaons (am3_mojo addition)
    def set_kaons(mut self, active: Bool):
        """Kaons are made in this step if `active` and their kernels are
        built. The pion yields of the multi-pion types follow
        (Calc_PhotoHadron_Rad)."""
        self.kaons = active and self.kaons_built

    def Calc_PhotoHadron_Kaons(mut self, dat: AM3Arrays):
        """Kaon sources qK[s] on the hadron grid from the last
        Calc_PhotoHadron_PhotonConvolution. As for the pions, number_split
        shares the kaons between the two neighboring bins and keeps their number
        and energy."""
        var co = C_LIGHT * dat.dlnE * MICROBARN
        for s in range(N_KAON):
            fill_zero(self.qK[s])
        if not self.kaons or len(self.resK) != len(self.kchan) * dat.nbins_had:
            return
        var nh = dat.nbins_had
        var f = co * self.photon_norm
        for c in range(len(self.kchan)):
            ref ch = self.kchan[c]
            var r = pd(self.resK).unsafe_offset(c * nh)
            for nucleon in range(2):
                var dens = pd(dat.n[P_protons if nucleon == 0 else P_neutrons])
                for s in range(N_KAON):
                    var m = ch.mp[s] if nucleon == 0 else ch.mn[s]
                    if m == 0.0:
                        continue
                    var chi = ch.chi_p[s] if nucleon == 0 else ch.chi_n[s]
                    var q: Int
                    var qr: Float64
                    q, qr = number_split(c_log(chi) / dat.dlnE, dat.dlnE)
                    var i0 = max(0, -q)
                    var i1 = min(nh, nh - 1 - q)
                    var lo = (1.0 - qr) * m * f
                    var hi = qr * m * f
                    var QK = pd(self.qK[s]).unsafe_offset(q)
                    # SIMD over 8 bins. Bin i takes the share hi of bin i - 1
                    # before the share lo of bin i, as in a loop over i.
                    comptime W = WD
                    var i = i0
                    while i + W <= i1:
                        var a = r.unsafe_offset(i).unsafe_load[width=W]() * (
                            dens.unsafe_offset(i).unsafe_load[width=W]()
                        )
                        var up = QK.unsafe_offset(i + 1)
                        up.unsafe_store(
                            SIMD[DType.float64, W](hi).fma(
                                a, up.unsafe_load[width=W]()
                            )
                        )
                        var dn = QK.unsafe_offset(i)
                        dn.unsafe_store(
                            SIMD[DType.float64, W](lo).fma(
                                a, dn.unsafe_load[width=W]()
                            )
                        )
                        i += W
                    while i < i1:
                        var a = r[unsafe_offset=i] * dens[unsafe_offset=i]
                        QK[unsafe_offset=i] = fma(lo, a, QK[unsafe_offset=i])
                        QK[unsafe_offset=i + 1] = fma(
                            hi, a, QK[unsafe_offset=i + 1]
                        )
                        i += 1

    # charm hadrons (am3_mojo addition)
    def set_charm(mut self, active: Bool):
        """Charm hadrons are made in this step if `active` and their kernels
        are built. The pion yields of the multi-pion types follow."""
        self.charm = active and self.charm_built

    def Calc_PhotoHadron_Charm(mut self, dat: AM3Arrays):
        """Charm sources qC[s] on the hadron grid from the last
        Calc_PhotoHadron_PhotonConvolution. number_split shares the charm
        hadrons between the two neighboring bins and keeps their number and
        energy."""
        for s in range(N_CHARM):
            fill_zero(self.qC[s])
        if not self.charm or len(self.resC) != len(self.csteps) * dat.nbins_had:
            return
        var job = PGCharmJob(
            Pointer(to=self).unsafe_origin_cast[MutUntrackedOrigin](),
            pd(dat.n[P_protons]),
            pd(dat.n[P_neutrons]),
            dat.nbins_had,
            dat.dlnE,
            C_LIGHT * dat.dlnE * MICROBARN * self.photon_norm,
        )
        parallel_run[pgc_work](job, min(self.nthreads, N_CHARM))
