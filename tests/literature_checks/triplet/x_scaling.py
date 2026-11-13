"""Triplet pair production (tests/literature_checks/triplet/README.md, "Energies
of the products"): the kinetic code's soft-photon approximation. For a lepton
of Lorentz factor gamma and a photon of energy eps (m_e c^2 units), a product's
lab energy is exactly the README's E_lab (with eps (y - x/k) = eps p_z/k).
Here x = p-/m and y = E'/k in the target lepton's rest frame. The code uses
E = gamma x. The script boosts the exact QED events (tools/triplet) to the lab
for isotropic photons. It compares both forms. It prints the change of the
positron and recoil spectra against the noise of the comparison. The change is
half the L1 distance, in the code's bins of 0.1 in ln E, over five bin offsets.
The noise is the change between two exact spectra that differ only in the
random azimuth. It also prints the share of products that change bin (in either
direction) and the mean positron energy.
usage: x_scaling.py (a few minutes)."""

import os, sys

import numpy as np
import gvar
import vegas

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
from support.paths import use_tool
from support.units import DLNE

use_tool("triplet")
from kinematics import generate
from produce import events
from qed_triplet import amplitude_sq_fast

rng = np.random.default_rng(3)
gvar.ranseed(3)  # vegas draws from gvar's generator: reproducible results


def weighted_events(k):
    integ = vegas.Integrator(4 * [[0.0, 1.0]])
    f = vegas.batchintegrand(lambda u: events(np.asarray(u), k)[0])
    integ(f, nitn=6, neval=40000)
    integ.set(neval=60000)
    out = []
    for u, w in integ.random_batch():
        for c in range(0, len(u), 4096):
            uc, wc = u[c : c + 4096], w[c : c + 4096]
            ok, jac, p1, kv, p2, p3, p4, inv = generate(uc, k)
            d = {key: val[ok] for key, val in inv.items()}
            wt = (
                wc[ok]
                * jac[ok]
                * amplitude_sq_fast(p1[ok], kv[ok], p2[ok], p3[ok], p4[ok], d)
            )
            out.append((wt, p2[ok], p3[ok], p4[ok]))
    return [np.concatenate([o[i] for o in out]) for i in range(4)]


DLN = DLNE
OFFSETS = DLN * np.arange(5) / 5
IOFF = range(len(OFFSETS))
FORMS = ("x", "e", "e2")


def lab_spectra(gamma, eps):
    beta = np.sqrt(1 - 1 / gamma**2)
    lk = np.linspace(np.log(4.4), np.log(gamma * eps * (1 + beta)), 13)
    lo = -2.0
    nb = int(np.ceil((np.log(gamma) + 0.6 - lo) / DLN)) + 1
    H = {
        (name, form, o): np.zeros(nb)
        for name in ("pos", "rec")
        for form in FORMS
        for o in IOFF
    }
    moved = {(name, o): 0.0 for name in ("pos", "rec") for o in IOFF}
    wsum = E_tot_x = E_tot_e = 0.0
    for a, b in zip(lk[:-1], lk[1:]):
        k = np.exp(0.5 * (a + b))
        W, p2, p3, p4 = weighted_events(k)
        # weight of this k: int dmu (1 - beta mu) sigma ~ k sigma(k) dk:
        # sigma(k) is sum(W) here
        wk = W * k * k * (b - a)
        wsum += np.sum(wk)
        mu = (1 - k / (gamma * eps)) / beta
        uperp = beta * np.sqrt(max(1 - mu * mu, 0.0)) / (1 - beta * mu)
        for name, p in (
            ("pos", p4),
            ("rec", np.where((p2[:, 1] >= p3[:, 1])[:, None], p2, p3)),
        ):
            x = p[:, 1]
            y = 0.5 * (p[:, 0] + p[:, 1]) / k
            pperp = np.hypot(p[:, 2], p[:, 3])
            Ex = gamma * x
            base = Ex + eps * (y - x / k)
            Ee, Ee2 = (
                base - pperp * uperp * np.cos(rng.uniform(0, 2 * np.pi, len(x)))
                for _ in range(2)
            )
            logs = [np.log(np.maximum(v, 1e-300)) for v in (Ex, Ee, Ee2)]
            for o, off in enumerate(OFFSETS):
                idx = [
                    np.clip(
                        np.floor((v - lo - off) / DLN).astype(int), 0, nb - 1
                    )
                    for v in logs
                ]
                for form, i in zip(FORMS, idx):
                    H[(name, form, o)] += np.bincount(
                        i, weights=wk, minlength=nb
                    )
                moved[(name, o)] += np.sum(wk[idx[0] != idx[1]])
            if name == "pos":
                E_tot_x += np.sum(wk * Ex)
                E_tot_e += np.sum(wk * Ee)
    return H, moved, wsum, E_tot_x / wsum, E_tot_e / wsum


if __name__ == "__main__":
    half_l1 = lambda h1, h2: 0.5 * np.abs(h1 - h2).sum() / h2.sum()
    for gamma in [1e4, 1e6]:
        for eps in [0.01, 0.1, 0.3, 1.0]:
            H, moved, wsum, ex, ee = lab_spectra(gamma, eps)
            for name in ("pos", "rec"):
                change, noise = (
                    [half_l1(H[(name, f, o)], H[(name, "e", o)]) for o in IOFF]
                    for f in ("x", "e2")
                )
                mv = [moved[(name, o)] / wsum for o in IOFF]
                print(
                    f"gamma={gamma:.0e} eps={eps:4.2f} {name}: spectrum change "
                    "(half L1, bins of 0.1) "
                    f"{min(change):6.2%}-{max(change):6.2%} against the noise "
                    f"of the comparison {min(noise):6.2%}-{max(noise):6.2%}; "
                    f"products in another bin {min(mv):6.2%}-{max(mv):6.2%}"
                    + (
                        f"; mean positron energy exact/x-scaling {ee / ex:.5f}"
                        if name == "pos"
                        else ""
                    ),
                    flush=True,
                )
