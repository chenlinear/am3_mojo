"""Production of the triplet tables. For each photon energy k (electron rest
frame), the script computes the total cross section and the per-interaction
distributions of the products' lab energy fractions:
  role 0: positron, y = x_+ k
  role 1: lower-energy electron, y = x_low k
  role 2: higher-energy (recoil) electron through its energy loss
  xi = 1 - x_high = x_+ + x_low,
          y = xi k
Each k has four VEGAS maps (Lepage 1978, J. Comput. Phys. 27, 192; the package
vegas, Lepage 2021, arXiv:2009.05112), adapted to the rate and to the rate
weighted by x_+, x_low and xi.

The photon energies are the 60 of K_NODES, from 4.02 to 1e10 (m_e c^2), or those
given. Every k and map seeds gvar's generator, which vegas draws from, so a run
is reproducible. The tables in src/ come from an unseeded run. A new run
therefore reproduces them within their Monte Carlo errors (the cross sections
to 3e-4), not bit for bit.

usage: produce.py OUT_DIR [k ...]   (one file OUT_DIR/k_<k>.npz per k)
As the README runs it, OUT_DIR is results/tools/triplet: the tools' outputs go
to results/tools/. The script skips each k whose file exists. So a run can be
resumed, and processes given different k values can share the work. All 60 take
about 40 minutes with 8 processes.
"""

import os, sys, time

import numpy as np
import gvar
import vegas

from qed_triplet import amplitude_sq_fast
from kinematics import generate

# (the bins of the tables that make_tables.py writes and physics/triplet.mojo
#  reads, so that the Monte Carlo fills exactly these)
from make_tables import DLY, LY0, tpp_nbins


# events per amplitude call (larger arrays are slower because of memory traffic)
CHUNK = 4096


def events(u, k):
    if len(u) > CHUNK:
        parts = [events(u[i : i + CHUNK], k) for i in range(0, len(u), CHUNK)]
        return (
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts], axis=1),
        )
    ok, jac, p1, kv, p2, p3, p4, inv = generate(u, k)
    w = np.zeros(u.shape[0])
    x = np.zeros((3, u.shape[0]))
    if ok.any():
        d = {key: val[ok] for key, val in inv.items()}
        w[ok] = jac[ok] * amplitude_sq_fast(
            p1[ok], kv[ok], p2[ok], p3[ok], p4[ok], d
        )
        x2, x3, x4 = p2[ok, 1], p3[ok, 1], p4[ok, 1]
        xl = np.minimum(x2, x3)
        x[0, ok], x[1, ok], x[2, ok] = x4, xl, x4 + xl
    return w, x


def make_integrand(k, role):
    @vegas.batchintegrand
    def f(u):
        w, x = events(np.asarray(u), k)
        return w if role < 0 else w * x[role]

    return f


def run_k(k, neval_adapt=100000, nitn_adapt=10, neval=250000, npass=16, seed=0):
    t0 = time.time()
    rng_seed = seed + int(1000 * np.log(k))
    NB = tpp_nbins(k)
    roles = [-1, 0, 1, 2]
    est = {}
    for role in roles:
        # vegas draws from gvar's generator
        gvar.ranseed(rng_seed + 7 * (role + 1))
        np.random.seed(rng_seed + 7 * (role + 1))
        integ = vegas.Integrator(4 * [[0.0, 1.0]])
        integ(make_integrand(k, role), nitn=nitn_adapt, neval=neval_adapt)
        integ.set(neval=neval)
        S = np.zeros(npass)
        H = np.zeros((npass, 3, NB))
        M = np.zeros((npass, 3))
        for p in range(npass):
            for u, wgt in integ.random_batch():
                w, x = events(u, k)
                w = w * wgt
                m = w > 0
                if not m.any():
                    continue
                w, x = w[m], x[:, m]
                S[p] += w.sum()
                for r in range(3):
                    M[p, r] += np.sum(w * x[r])
                    b = np.floor((np.log(x[r] * k) - LY0) / DLY).astype(int)
                    ok = (b >= 0) & (b < NB)
                    if not np.all(ok):
                        raise RuntimeError(
                            f"k={k}: y outside the table "
                            f"({np.min(np.log(x[r] * k)):.3f}, "
                            f"{np.max(np.log(x[r] * k)):.3f})"
                        )
                    H[p, r] += np.bincount(b, weights=w, minlength=NB)
        est[role] = (S, H, M)

    def comb(vals):
        v = np.array([a.mean() for a in vals])
        e = np.array([a.std(ddof=1) / np.sqrt(len(a)) for a in vals])
        wv = 1.0 / np.maximum(e, 1e-300) ** 2
        return np.sum(wv * v) / np.sum(wv), 1.0 / np.sqrt(np.sum(wv))

    sig, dsig = comb([est[r][0] for r in roles])
    means, dmeans = np.zeros(3), np.zeros(3)
    dist, derr = np.zeros((3, NB)), np.zeros((3, NB))
    win = np.ones(11) / 11.0
    for r in range(3):
        means[r], dmeans[r] = comb(
            [est[q][2][:, r] / est[q][0] for q in (-1, r)]
        )
        # Weights from each bin's own variance estimate favor downward
        # fluctuations.
        vals, var = [], []
        for q in (-1, r):
            S, H, M = est[q]
            h = H[:, r, :] / S[:, None]
            vals.append(h.mean(0))
            var.append(np.convolve(h.var(0, ddof=1) / npass, win, mode="same"))
        vals, var = np.array(vals), np.array(var)
        wv = np.where(var > 0, 1.0 / np.where(var > 0, var, 1.0), 0.0)
        tot = wv.sum(0)
        comb_v = np.where(
            tot > 0,
            np.sum(wv * vals, 0) / np.where(tot > 0, tot, 1.0),
            vals.mean(0),
        )
        comb_e = np.where(
            tot > 0,
            np.sqrt(np.sum(wv**2 * var, 0)) / np.where(tot > 0, tot, 1.0),
            0.0,
        )
        norm = comb_v.sum()
        dist[r] = comb_v / norm / DLY
        derr[r] = comb_e / norm / DLY
    return dict(
        k=k,
        sig=sig,
        dsig=dsig,
        means=means,
        dmeans=dmeans,
        dist=dist,
        derr=derr,
        ly0=LY0,
        dly=DLY,
        seconds=time.time() - t0,
    )


K_NODES = [
    4.02,
    4.05,
    4.1,
    4.2,
    4.35,
    4.6,
    5.0,
    5.6,
    6.4,
    7.5,
    9.0,
    11.0,
    14.0,
    18.0,
    23.0,
    30.0,
    40.0,
    55.0,
    75.0,
    100.0,
] + [float(f"{100.0 * 10 ** (0.2 * n):.6g}") for n in range(1, 41)]

if __name__ == "__main__":
    out = sys.argv[1]
    ks = [float(a) for arg in sys.argv[2:] for a in arg.split()] or K_NODES
    os.makedirs(out, exist_ok=True)
    for k in ks:
        fn = os.path.join(out, f"k_{k:.6g}.npz")
        if os.path.exists(fn):
            continue
        r = run_k(k)
        np.savez(fn, **r)
        print(
            f"k={k:g}: sigma={r['sig']:.6g}({r['dsig']:.1e}) "
            f"means={r['means']} [{r['seconds']:.0f} s]",
            flush=True,
        )
