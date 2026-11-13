"""Phase space of gamma(k) + e(p1) -> e(p2) + e(p3) + e+(p4) in the rest frame
of e(p1).

Variables:
  T2  recoil kinetic energy of p2 (|t| = 2 T2)
  M2  pair mass squared (p3 + p4)^2
  c*  cosine of the angle between p3 and the photon in the pair rest frame
      (tanh map that absorbs the collinear peaks 1/((1 - b c)(1 + b c)))
  phi* azimuth of p3 around the photon direction in the pair rest frame
The azimuth of p2 around the photon is integrated analytically (2 pi).

Phase space: dPhi3 = beta*/(512 pi^4 k) dT2 dM2 dc* dphi*. Cross section:
  sigma / (alpha r_e^2) = beta*/(128 pi k^2) * Int sum|M|^2 dT2 dM2 dc* dphi*
over the region E2 <= E3. The two outgoing electrons are identical, so the
integral covers this region with the full |M|^2, without a factor 1/2.
"""

import numpy as np

from qed_triplet import onshell, dot


def t2_limits(k):
    r = k * np.sqrt(k * (k - 4.0))
    Ep = ((k * k - 1.0) + r) / (1.0 + 2.0 * k)
    # T- = E- - 1 = (a - r)/(1 + 2k) with a = k(k - 2) - 2.
    # a^2 - r^2 = 4 (1 + 2k), so T- = 4/(a + r).
    a = k * (k - 2.0) - 2.0
    return 4.0 / (a + r), Ep - 1.0


def generate(u, k):
    """jac is the phase-space Jacobian (including beta* and the constant), and
    inv holds the accurate invariants for qed_triplet.amplitude_sq."""
    N = u.shape[0]
    Tm, Tp = t2_limits(k)
    lT = np.log(Tm) + u[:, 0] * (np.log(Tp) - np.log(Tm))
    T2 = np.exp(lT)
    jac = T2 * (np.log(Tp) - np.log(Tm))
    p2a = np.sqrt(T2 * (T2 + 2.0))
    # M2 max for this T2 (cos theta2 = 1): 2k (p2 - T2) - 2 T2, with
    # p2 - T2 = 2 T2 / (p2 + T2)
    M2max = 2.0 * k * (2.0 * T2 / (p2a + T2)) - 2.0 * T2
    ok = M2max > 4.0
    M2max = np.where(ok, M2max, 4.0 * (1 + 1e-12))
    lM = np.log(4.0) + u[:, 1] * (np.log(M2max) - np.log(4.0))
    M2 = np.exp(lM)
    jac = jac * M2 * (np.log(M2max) - np.log(4.0))
    # cos theta2 of p2 relative to the photon: M2 = 2k (p2 c - T2) - 2 T2
    c2 = (M2 + 2.0 * T2 * (1.0 + k)) / (2.0 * k * p2a)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(np.maximum(1.0 - c2 * c2, 0.0))
    E2 = 1.0 + T2
    p2z = p2a * c2
    p2x = p2a * s2
    # Backward (c2 < 0), E2 + p2z cancels, and p+ = (1 + px^2) / (E2 - p2z)
    # instead. Only there: forward, E2 - p2z rounds to 0 at large T2 (k from
    # about 1e10), and the division would warn although its value is not used.
    p2plus = E2 + p2z
    back = c2 < 0.0
    p2plus[back] = (1.0 + p2x[back] * p2x[back]) / (E2[back] - p2z[back])
    zero = np.zeros(N)
    p2 = onshell(p2plus, p2x, zero)
    p1 = onshell(np.ones(N), zero, zero)
    kv = np.stack([np.full(N, 2.0 * k), zero, zero, zero], axis=-1)
    Pplus = 1.0 + 2.0 * k - p2plus
    Px = -p2x
    M = np.sqrt(M2)
    vx = Px / Pplus
    kps = (M / Pplus) * 2.0 * k
    kms = (Pplus / M) * vx * vx * 2.0 * k
    kxs = -vx * 2.0 * k
    kE = 0.5 * (kps + kms)
    kz = 0.5 * (kps - kms)
    nk = np.stack([kxs, zero, kz], axis=-1) / kE[:, None]
    # the y axis is perpendicular (k* is in the x-z plane)
    e2v = np.stack([zero, np.ones(N), zero], axis=-1)
    e1v = np.cross(e2v, nk)
    bs = np.sqrt(np.maximum(1.0 - 4.0 / M2, 0.0))
    ps = 0.5 * M * bs
    Es = 0.5 * M
    bs_ = np.maximum(bs, 1e-300)
    wmax = np.arctanh(np.minimum(bs_, 1.0 - 1e-16))
    w = (2.0 * u[:, 2] - 1.0) * wmax
    cs = np.tanh(w) / bs_
    jac = jac * 2.0 * wmax * (1.0 - bs_ * bs_ * cs * cs) / bs_
    phs = 2.0 * np.pi * u[:, 3]
    jac = jac * 2.0 * np.pi
    ss = np.sqrt(np.maximum(1.0 - cs * cs, 0.0))
    dvec = cs[:, None] * nk + ss[:, None] * (
        np.cos(phs)[:, None] * e1v + np.sin(phs)[:, None] * e2v
    )

    # p3*, p4* in the pair frame, light-cone with the stable branch
    def lc_pair(sign):
        dz = sign * dvec[:, 2]
        dx = sign * dvec[:, 0]
        dy = sign * dvec[:, 1]
        qx = ps * dx
        qy = ps * dy
        qplus = np.where(
            dz >= 0.0,
            Es + ps * dz,
            (1.0 + qx * qx + qy * qy) / np.maximum(Es - ps * dz, 1e-300),
        )
        return qplus, qx, qy

    def to_erf(qplus, qx, qy):
        plus = (Pplus / M) * qplus
        x = qx + vx * plus
        y = qy
        return onshell(plus, x, y)

    p3 = to_erf(*lc_pair(1.0))
    p4 = to_erf(*lc_pair(-1.0))
    inv = {}
    inv["t23"] = -2.0 * T2
    T3 = ((p3[:, 0] - 1.0) ** 2 + p3[:, 2] ** 2 + p3[:, 3] ** 2) / (
        2.0 * p3[:, 0]
    )
    inv["t32"] = -2.0 * T3
    inv["s34"] = M2
    inv["s24"] = 2.0 + 2.0 * dot(p2, p4)
    inv["kp1"] = np.full(N, k)
    inv["kp2"] = 0.5 * p2[:, 1] * 2.0 * k

    # p3.k and p4.k from the pair frame: k* (E* -+ p* c*), without cancellation
    def kdot(c):
        return np.where(
            c >= 0.0,
            kE * (1.0 + ps * ps * (1.0 - c * c)) / (Es + ps * c),
            kE * (Es - ps * c),
        )

    inv["kp3"] = kdot(cs)
    inv["kp4"] = np.where(
        -cs >= 0.0,
        kE * (1.0 + ps * ps * (1.0 - cs * cs)) / (Es - ps * cs),
        kE * (Es + ps * cs),
    )
    # light-cone differences via momentum conservation (p1 + k = p2 + p3 + p4)
    inv["p3mk"] = np.stack(
        [-(p2[:, 0] + p4[:, 0] - 1.0), p3[:, 1], p3[:, 2], p3[:, 3]], axis=-1
    )
    inv["kmp4"] = np.stack(
        [p2[:, 0] + p3[:, 0] - 1.0, -p4[:, 1], -p4[:, 2], -p4[:, 3]], axis=-1
    )
    inv["p2mk"] = np.stack(
        [-(p3[:, 0] + p4[:, 0] - 1.0), p2[:, 1], p2[:, 2], p2[:, 3]], axis=-1
    )
    E3 = 1.0 + T3
    ok = ok & (E2 <= E3) & (bs > 0.0)
    jac = np.where(ok, jac * bs / (128.0 * np.pi * k * k), 0.0)
    return ok, jac, p1, kv, p2, p3, p4, inv
