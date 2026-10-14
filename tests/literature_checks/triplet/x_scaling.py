"""Triplet pair production, README "Triplet pair production": the soft-photon approximation of the
kinetic code. For a lepton of Lorentz factor gamma and a photon of energy eps (m_e c^2 units) at the
lab angle theta, the lab energy of a product is exactly
    E = gamma x + eps (y - x/k) - |p_perp| beta sin(theta)/(1 - beta cos(theta)) cos(dphi)
(x = p-/m and y = E'/k in the target lepton's rest frame, p_perp the product's momentum transverse
to the photon there, dphi a random azimuth). The code uses E = gamma x. The script boosts the exact
QED events (tools/triplet) to the lab for isotropic photons (gamma = 1e4 and 1e6; eps = 0.01, 0.1,
0.3 and 1) and compares both forms: the change of the positron and recoil spectra (half the L1
distance, in the code's bins of 0.1 in ln E, over five bin offsets) against the noise of that
comparison (two exact spectra that differ only in the random azimuth), the share of products that
change bin (in both directions), and the mean positron energy.
usage: x_scaling.py (a few minutes)."""
import os, sys
import numpy as np
import gvar
import vegas
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.paths import load_derivation, use_tool  # noqa: E402

use_tool('triplet')
from kinematics import generate  # noqa: E402
from produce import events  # noqa: E402
from qed_triplet import amplitude_sq_fast  # noqa: E402
derivation = load_derivation('triplet')   # the grid step

rng = np.random.default_rng(3)
gvar.ranseed(3)  # vegas draws from gvar's generator: reproducible results


def weighted_events(k, n=60000):
    integ = vegas.Integrator(4 * [[0.0, 1.0]])
    f = vegas.batchintegrand(lambda u: events(np.asarray(u), k)[0])
    integ(f, nitn=6, neval=40000)
    integ.set(neval=n)
    out = []
    for u, w in integ.random_batch():
        for c in range(0, len(u), 4096):
            uc, wc = u[c:c + 4096], w[c:c + 4096]
            ok, jac, p1, kv, p2, p3, p4, inv = generate(uc, k)
            d = {key: val[ok] for key, val in inv.items()}
            wt = wc[ok] * jac[ok] * amplitude_sq_fast(p1[ok], kv[ok], p2[ok], p3[ok], p4[ok], None, d)
            out.append((wt, p2[ok], p3[ok], p4[ok]))
    W = np.concatenate([o[0] for o in out])
    return W, [np.concatenate([o[i] for o in out]) for i in (1, 2, 3)]


DLN = float(derivation.grid_step())          # the bin width of the code's grids in ln E
OFFSETS = DLN * np.arange(5) / 5            # five offsets of the bins


def lab_spectra(gamma, eps, nk=12):
    """Binned lab spectra of the positron and the recoil (the leading electron) for isotropic photons, per
    bin offset: with E = gamma x ('x'), exact ('e') and exact with another random azimuth ('e2'). Also the
    weighted share of products whose bin differs between 'x' and 'e', and the mean positron energies of both."""
    beta = np.sqrt(1 - 1 / gamma**2)
    kmax = gamma * eps * (1 + beta)
    if kmax <= 4.4:
        return None
    lk = np.linspace(np.log(4.4), np.log(kmax), nk + 1)
    lo = -2.0
    nb = int(np.ceil((np.log(gamma) + 0.6 - lo) / DLN)) + 1
    H = {(name, form, o): np.zeros(nb) for name in ('pos', 'rec') for form in ('x', 'e', 'e2') for o in range(len(OFFSETS))}
    moved = {(name, o): 0.0 for name in ('pos', 'rec') for o in range(len(OFFSETS))}
    wsum = {'pos': 0.0, 'rec': 0.0}
    E_tot_x = E_tot_e = N = 0.0
    for a, b in zip(lk[:-1], lk[1:]):
        k = np.exp(0.5 * (a + b))
        W, (p2, p3, p4) = weighted_events(k)
        # weight of this k: int dmu (1 - beta mu) sigma ~ k sigma(k) dk: sigma(k) is sum(W) here
        wk = W / W.sum() * W.sum() * k * k * (b - a)
        mu = (1 - k / (gamma * eps)) / beta
        uperp = beta * np.sqrt(max(1 - mu * mu, 0.0)) / (1 - beta * mu)
        x2, x3 = p2[:, 1], p3[:, 1]
        lead = x2 >= x3
        for name, p in (('pos', p4), ('rec', np.where(lead[:, None], p2, p3))):
            x = p[:, 1]
            y = 0.5 * (p[:, 0] + p[:, 1]) / k
            pperp = np.hypot(p[:, 2], p[:, 3])
            Ex = gamma * x
            base = gamma * x + eps * (y - x / k)
            Ee = base - pperp * uperp * np.cos(rng.uniform(0, 2 * np.pi, len(x)))
            Ee2 = base - pperp * uperp * np.cos(rng.uniform(0, 2 * np.pi, len(x)))
            lx, le, le2 = (np.log(np.maximum(v, 1e-300)) for v in (Ex, Ee, Ee2))
            for o, off in enumerate(OFFSETS):
                ix, ie, ie2 = (np.clip(np.floor((v - lo - off) / DLN).astype(int), 0, nb - 1) for v in (lx, le, le2))
                H[(name, 'x', o)] += np.bincount(ix, weights=wk, minlength=nb)
                H[(name, 'e', o)] += np.bincount(ie, weights=wk, minlength=nb)
                H[(name, 'e2', o)] += np.bincount(ie2, weights=wk, minlength=nb)
                moved[(name, o)] += np.sum(wk[ix != ie])
            wsum[name] += np.sum(wk)
            if name == 'pos':
                E_tot_x += np.sum(wk * Ex)
                E_tot_e += np.sum(wk * Ee)
                N += np.sum(wk)
    return H, moved, wsum, E_tot_x / N, E_tot_e / N


if __name__ == '__main__':
    half_l1 = lambda h1, h2: 0.5 * np.abs(h1 - h2).sum() / h2.sum()
    for gamma in [1e4, 1e6]:
        for eps in [0.01, 0.1, 0.3, 1.0]:
            r = lab_spectra(gamma, eps)
            if r is None:
                continue
            H, moved, wsum, ex, ee = r
            for name in ('pos', 'rec'):
                change = [half_l1(H[(name, 'x', o)], H[(name, 'e', o)]) for o in range(len(OFFSETS))]
                noise = [half_l1(H[(name, 'e2', o)], H[(name, 'e', o)]) for o in range(len(OFFSETS))]
                mv = [moved[(name, o)] / wsum[name] for o in range(len(OFFSETS))]
                print(f'gamma={gamma:.0e} eps={eps:4.2f} {name}: spectrum change (half L1, bins of 0.1) {min(change):6.2%}-{max(change):6.2%}'
                      f' against the noise of the comparison {min(noise):6.2%}-{max(noise):6.2%}; products in another bin'
                      f' {min(mv):6.2%}-{max(mv):6.2%}' + (f'; mean positron energy exact/x-scaling {ee / ex:.5f}' if name == 'pos' else ''),
                      flush=True)
