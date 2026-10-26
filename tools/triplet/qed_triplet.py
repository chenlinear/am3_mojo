"""Triplet pair production gamma + e- -> e- + e- + e+ at lowest order in QED, from the full matrix element
(8 Feynman diagrams: the photon attached to the target line or to the produced pair line, and the exchange of the two
outgoing electrons).

Frame: rest frame of the incoming electron (ERF), units m_e = c = 1, photon of energy k along +z. Four-vectors are kept
in light-cone components (p+, p-, px, py), p+- = E +- pz, so that small invariants (p.k = p- k+/2, momentum transfers)
never come from cancellations. The spinors are Lepage-Brodsky light-front spinors in the chiral representation, whose
components are bounded by sqrt(p+) or m/sqrt(p+).

For an ultra-relativistic incoming electron in the lab, a product's lab energy fraction is the
invariant x = (p.k)/(p1.k) = p-/m (independent of the photon direction).
"""
import numpy as np

# ---------------------------------------------------------------------------
# light-cone four-vectors: arrays (..., 4) = (p+, p-, px, py)
# ---------------------------------------------------------------------------
def dot(a, b):
    return 0.5 * (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]) - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3]


def onshell(plus, px, py, m=1.0):
    """(p+, p-, px, py) of a particle of mass m from p+ and p_perp (p- from the mass shell)."""
    return np.stack([plus, (m * m + px * px + py * py) / plus, px, py], axis=-1)


# ---------------------------------------------------------------------------
# Dirac algebra, chiral representation, spinor index order (L1, L2, R1, R2)
# ---------------------------------------------------------------------------
def slash(a):
    """a_mu gamma^mu for (complex) light-cone vectors a (..., 4) -> (..., 4, 4).
    a.sigma = [[a-, -(ax - i ay)], [-(ax + i ay), a+]], a.sigmabar = [[a+, ax - i ay], [ax + i ay, a-]]."""
    ap, am, ax, ay = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    cm = ax - 1j * ay
    cp = ax + 1j * ay
    z = np.zeros_like(ap, dtype=complex)
    S = np.zeros(a.shape[:-1] + (4, 4), dtype=complex)
    # upper right block: a.sigma (acts on R components), lower left: a.sigmabar (acts on L)
    S[..., 0, 2], S[..., 0, 3], S[..., 1, 2], S[..., 1, 3] = am, -cm, -cp, ap
    S[..., 2, 0], S[..., 2, 1], S[..., 3, 0], S[..., 3, 1] = ap, cm, cp, am
    del z
    return S


def u_spinors(p, m=1.0):
    """Light-front spinors u(p, +-1/2), shape (..., 2, 4): u+ = (m, 0, p+, pc)/sqrt(p+), u- = (-conj(pc), p+, 0,
    m)/sqrt(p+), pc = px + i py."""
    sp = np.sqrt(p[..., 0])
    pc = p[..., 2] + 1j * p[..., 3]
    zero = np.zeros_like(pc)
    up = np.stack([m + zero, zero, p[..., 0] + zero, pc], axis=-1) / sp[..., None]
    um = np.stack([-np.conj(pc), p[..., 0] + zero, zero, m + zero], axis=-1) / sp[..., None]
    return np.stack([up, um], axis=-2)


def v_spinors(p, m=1.0):
    """v(p, s) = u(p, s) with m -> -m (satisfies (pslash + m) v = 0)."""
    return u_spinors(p, -m)


def bar(u):
    """Row spinor ubar = u^dagger gamma0; gamma0 swaps the L and R blocks."""
    c = np.conj(u)
    return np.concatenate([c[..., 2:4], c[..., 0:2]], axis=-1)


def current(ub, w):
    """J^mu = ubar gamma^mu w in light-cone components (J+, J-, Jx, Jy); ub is the bar spinor (index order of bar():
    (R1*, R2*, L1*, L2*)) and w a column spinor (L1, L2, R1, R2)."""
    # recover u_L*, u_R* from the bar ordering
    uR1, uR2, uL1, uL2 = ub[..., 0], ub[..., 1], ub[..., 2], ub[..., 3]
    wL1, wL2, wR1, wR2 = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    Jp = 2.0 * (uL2 * wL2 + uR1 * wR1)
    Jm = 2.0 * (uL1 * wL1 + uR2 * wR2)
    Jx = -(uL1 * wL2 + uL2 * wL1) + (uR1 * wR2 + uR2 * wR1)
    Jy = -(-1j * uL1 * wL2 + 1j * uL2 * wL1) + (-1j * uR1 * wR2 + 1j * uR2 * wR1)
    return np.stack([Jp, Jm, Jx, Jy], axis=-1)


def sandwich(ub, X, w):
    return np.einsum('...i,...ij,...j->...', ub, X, w)


ID4 = np.eye(4, dtype=complex)


# ---------------------------------------------------------------------------
# matrix element
# ---------------------------------------------------------------------------
def amplitude_sq(p1, k, p2, p3, p4, d, check_gauge=False):
    """Sum over all spins and photon polarisations of |M|^2 (couplings e^3 removed), for events given as light-cone
    arrays (N, 4). `d` holds accurate invariants:
      t23 = (p1-p2)^2, t32 = (p1-p3)^2, s34 = (p3+p4)^2, s24 = (p2+p4)^2,
      kp1 = k.p1, kp2 = k.p2, kp3 = k.p3, kp4 = k.p4,
    and the light-cone differences used in the propagator numerators:
      p3mk = p3 - k, kmp4 = k - p4, p2mk = p2 - k  (computed with momentum conservation).
    With check_gauge, the polarisation vectors are replaced by k (Ward identity test)."""
    N = p1.shape[0]
    u1 = u_spinors(p1)
    u2 = u_spinors(p2)
    u3 = u_spinors(p3)
    v4 = v_spinors(p4)
    if check_gauge:
        pols = [k]
    else:
        e1 = np.zeros((N, 4)); e1[:, 2] = 1.0
        e2 = np.zeros((N, 4)); e2[:, 3] = 1.0
        pols = [e1, e2]
    P1k = slash(p1 + k) + ID4  # (p1 + k)slash + m
    P3k = slash(d['p3mk']) + ID4
    K4 = slash(d['kmp4']) + ID4
    P2k = slash(d['p2mk']) + ID4
    D1 = 2.0 * d['kp1']           # (p1+k)^2 - m^2
    D3 = -2.0 * d['kp3']          # (p3-k)^2 - m^2
    D4 = -2.0 * d['kp4']          # (k-p4)^2 - m^2
    D2 = -2.0 * d['kp2']          # (p2-k)^2 - m^2
    total = np.zeros(N)
    for eps in pols:
        E = slash(eps)
        # direct (2 = target-line electron, 3 = pair electron) and exchanged (2 <-> 3) amplitudes
        amp = np.zeros((N, 2, 2, 2, 2), dtype=complex)  # [s1, s2, s3, s4]
        for (ia, ta, sab) in ((2, d['t23'], d['s34']), (3, d['t32'], d['s24'])):
            sign = 1.0 if ia == 2 else -1.0
            ua = u2 if ia == 2 else u3      # target-line outgoing electron
            ub_ = u3 if ia == 2 else u2      # pair-line outgoing electron
            ba, bb = bar(ua), bar(ub_)
            # propagator numerators/denominators that depend on the labelling
            if ia == 2:
                PB, DB = P3k, D3           # pair electron absorbs the photon: (pb - k)
                PA2, DA2 = P2k, D2         # target line, photon after the virtual photon: (pa - k)
            else:
                PB, DB = P2k, D2
                PA2, DA2 = P3k, D3
            for s1 in range(2):
                for sa in range(2):
                    # target current J_a1 = ubar_a gamma u1, B-type diagrams (photon on pair line)
                    J = current(ba[:, sa], u1[:, s1])
                    Js = slash(J)
                    XB = (E @ PB @ Js) / DB[:, None, None] + (Js @ K4 @ E) / D4[:, None, None]
                    for sb in range(2):
                        for s4 in range(2):
                            mB = sandwich(bb[:, sb], XB, v4[:, s4]) / ta
                            if ia == 2:
                                amp[:, s1, sa, sb, s4] += sign * mB
                            else:
                                amp[:, s1, sb, sa, s4] += sign * mB
            for sb in range(2):
                for s4 in range(2):
                    # pair current J_b4 = ubar_b gamma v4, A-type diagrams (photon on target line)
                    J = current(bb[:, sb], v4[:, s4])
                    Js = slash(J)
                    XA = (Js @ P1k @ E) / D1[:, None, None] + (E @ PA2 @ Js) / DA2[:, None, None]
                    for s1 in range(2):
                        for sa in range(2):
                            mA = sandwich(ba[:, sa], XA, u1[:, s1]) / sab
                            if ia == 2:
                                amp[:, s1, sa, sb, s4] += sign * mA
                            else:
                                amp[:, s1, sb, sa, s4] += sign * mA
        total += np.sum(np.abs(amp) ** 2, axis=(1, 2, 3, 4))
    return total


def amplitude_sq_fast(p1, k, p2, p3, p4, d):
    """Same as amplitude_sq, with the spin sums vectorised (all helicity combinations at once)."""
    N = p1.shape[0]
    u1 = u_spinors(p1)            # (N, 2, 4)
    u2 = u_spinors(p2)
    u3 = u_spinors(p3)
    v4 = v_spinors(p4)
    b2, b3 = bar(u2), bar(u3)
    e1 = np.zeros((N, 4)); e1[:, 2] = 1.0
    e2 = np.zeros((N, 4)); e2[:, 3] = 1.0
    P1k = slash(p1 + k) + ID4
    P3k = slash(d['p3mk']) + ID4
    K4 = slash(d['kmp4']) + ID4
    P2k = slash(d['p2mk']) + ID4
    D1 = (2.0 * d['kp1'])[:, None, None]
    D3 = (-2.0 * d['kp3'])[:, None, None]
    D4 = (-2.0 * d['kp4'])[:, None, None]
    D2 = (-2.0 * d['kp2'])[:, None, None]
    # currents for all spin pairs: target-line J_a1 (a = 2 or 3) and pair-line J_b4
    def cur(bA, w):   # bA (N,2,4) bar spinors, w (N,2,4) -> (N, 2(sa), 2(sw), 4)
        return current(bA[:, :, None, :], w[:, None, :, :])
    J21, J31 = cur(b2, u1), cur(b3, u1)
    J34, J24 = cur(b3, v4), cur(b2, v4)
    S21, S31, S34, S24 = slash(J21), slash(J31), slash(J34), slash(J24)   # (N,2,2,4,4)
    total = np.zeros(N)
    for eps in (e1, e2):
        E = slash(eps)
        EP3, EP2 = E @ P3k / D3, E @ P2k / D2          # photon on the outgoing electron (3 or 2)
        K4E, P1E = K4 @ E / D4, P1k @ E / D1
        # direct: B (pair line 3-4, target current J21), A (target line 2, pair current J34)
        XB23 = EP3[:, None, None] @ S21 + S21 @ K4E[:, None, None]       # [s2, s1]
        XA23 = S34 @ P1E[:, None, None] + EP2[:, None, None] @ S34       # [s3, s4]
        # exchanged: B (pair line 2-4, target current J31), A (target line 3, pair current J24)
        XB32 = EP2[:, None, None] @ S31 + S31 @ K4E[:, None, None]       # [s3, s1]
        XA32 = S24 @ P1E[:, None, None] + EP3[:, None, None] @ S24       # [s2, s4]
        # amplitudes amp[n, s1, s2, s3, s4]
        mB23 = np.einsum('nci,nbaij,ndj->nabcd', b3, XB23, v4) / d['t23'][:, None, None, None, None]
        mA23 = np.einsum('nbi,ncdij,naj->nabcd', b2, XA23, u1) / d['s34'][:, None, None, None, None]
        mB32 = np.einsum('nbi,ncaij,ndj->nabcd', b2, XB32, v4) / d['t32'][:, None, None, None, None]
        mA32 = np.einsum('nci,nbdij,naj->nabcd', b3, XA32, u1) / d['s24'][:, None, None, None, None]
        amp = mB23 + mA23 - mB32 - mA32
        total += np.sum(amp.real ** 2 + amp.imag ** 2, axis=(1, 2, 3, 4))
    return total
