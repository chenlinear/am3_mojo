"""Plot am3_mojo against the C++ AM3 (the original with the fixes of
tests/reference/am3_fixes.patch, as in the reference tests): photon, e+/-,
proton and neutrino spectra of every scenario with the per-bin relative
deviation, and the worst flux deviation at every checkpoint.

usage: python tests/plots/plot_spectra.py [--rerun]
Uses the runs cached in results/runs/ by the reference tests (runs whatever
is missing). Figures: results/plots/spectra_<scenario>.png, deviation_vs_time.png
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.compare import FLOOR, RTOL, deviations, grid_of, kind  # noqa: E402
from support.paths import results_dir  # noqa: E402
from support.runs import cached_scenario, checkpoints  # noqa: E402
from support.scenarios import SCENARIOS  # noqa: E402
from deviation import IDENT, rel  # noqa: E402

SPECIES = [('get_photons', 'get_egrid_photons', 'photons'), ('get_pairs', 'get_egrid_lep', 'electrons + positrons'),
           ('get_protons', 'get_egrid_had', 'protons'), ('get_neutrinos', 'get_egrid_neutrinos', 'neutrinos (all flavours)')]
CRIT = f'1e{round(np.log10(RTOL))}'  # '1e-2'


def worst_flux(R, T, step):
    """Worst deviation over all fluxes, as checked by the reference tests
    (tests/support/compare.py)."""
    w = 0.0
    for k in R.files:
        if k.startswith(f'{step:06d}/') and kind(k) == 'flux' and R[k].shape == T[k].shape:
            E = grid_of(R, k, R[k].size) if R[k].ndim == 1 else None
            d = deviations(R[k], T[k], FLOOR, E)
            if d:
                w = max(w, d[0])
    return max(w, IDENT)


def plot_scenario(sc, R, M, outdir):
    steps = checkpoints(R)
    first, last = steps[0], steps[-1]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex='col',
                             gridspec_kw=dict(height_ratios=[3, 2], hspace=0.06, wspace=0.28))
    for c, (getter, grid, label) in enumerate(SPECIES):
        ax, axd = axes[0, c], axes[1, c]
        E = R[f'{last:06d}/{grid}']
        ref = R[f'{last:06d}/{getter}']
        ax.set_title(label)
        ax.set_xscale('log')
        axd.set_xlabel('E [eV]')
        if not np.any(ref > 0):
            ax.text(0.5, 0.5, 'not simulated\nin this scenario', ha='center', va='center', transform=ax.transAxes)
            continue
        for step, alpha in ((first, 0.45), (last, 1.0)):
            ax.loglog(E, E * R[f'{step:06d}/{getter}'], color='k', lw=3.2, alpha=0.3 * alpha + 0.05,
                      label=f'C++ AM3, step {step}')
            ax.loglog(E, E * M[f'{step:06d}/{getter}'], color='#1f77b4', lw=1.2, ls='--', alpha=alpha,
                      label=f'am3_mojo, step {step}')
        pk = np.max(E * ref)
        ax.set_ylim(pk * FLOOR, pk * 10)  # the range the reference tests check
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc='lower left')
        axd.semilogy(E, rel(ref, M[f'{last:06d}/{getter}'], E), ls='none', marker='.', ms=3.5, color='#1f77b4')
        axd.axhline(RTOL, color='r', lw=0.8, ls='--')
        axd.axhline(np.finfo(float).eps, color='k', lw=0.5, ls=':')
        axd.set_ylim(IDENT / 5, 1)
        axd.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(r'$E^2\,dN/dE\,dV$  [eV cm$^{-3}$]')
            axd.set_ylabel(f'|am3_mojo / C++ - 1| at step {last}\n(red: {CRIT} criterion)')
    fig.suptitle(f'{sc}: am3_mojo vs C++ AM3 (with am3_fixes.patch) at step {last} - '
                 f'worst deviation over all fluxes: {worst_flux(R, M, last):.1e}')
    path = os.path.join(outdir, f'spectra_{sc}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rerun', action='store_true')
    a = ap.parse_args()
    outdir = results_dir('plots')
    written = []
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.5 * len(SCENARIOS), 4.2), sharey=True)
    for ax, sc in zip(axes, sorted(SCENARIOS)):
        R, M = cached_scenario('cpp-fixed', sc, a.rerun), cached_scenario('mojo-am3', sc, a.rerun)
        written.append(plot_scenario(sc, R, M, outdir))
        steps = checkpoints(R)
        ax.semilogy(steps, [worst_flux(R, M, s) for s in steps], 'o-', color='#1f77b4')
        ax.axhline(RTOL, color='r', lw=0.8, ls='--')
        ax.set_title(sc)
        ax.set_xlabel('time step')
        ax.set_ylim(1e-12, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('worst |am3_mojo / C++ - 1| over all fluxes')
    fig.suptitle(f'am3_mojo vs C++ AM3 (with am3_fixes.patch) at every checkpoint (red: {CRIT} criterion)')
    fig.tight_layout()
    path = os.path.join(outdir, 'deviation_vs_time.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    written.append(path)
    print('\n'.join(written))


if __name__ == '__main__':
    main()
