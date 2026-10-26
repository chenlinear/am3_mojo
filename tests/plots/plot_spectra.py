"""Plot am3_mojo against the C++ AM3 (with the fixes of tests/reference/am3_fixes.patch, as in the reference tests):
photon, e+/-, proton and neutrino spectra of every scenario with the per-bin relative deviation, and the worst flux
deviation at every checkpoint.

usage: python tests/plots/plot_spectra.py
Uses the runs cached in results/runs/ by the reference tests (runs whatever is missing).
Figures: results/plots/spectra_<scenario>.png, deviation_vs_time.png
"""
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.compare import FLOOR, RTOL, deviations, grid_of, kind
from support.paths import results_dir
from support.runs import cached_scenario, checkpoints
from support.scenarios import SCENARIOS
from deviation import IDENT, plt, rel

# (get_neutrinos has no nu_tau, as in AM3)
SPECIES = [('get_photons', 'get_egrid_photons', r'$\gamma$'),
           ('get_pairs', 'get_egrid_lep', r'$\mathrm{e}^{-} + \mathrm{e}^{+}$'),
           ('get_protons', 'get_egrid_had', r'$\mathrm{p}$'),
           ('get_neutrinos', 'get_egrid_neutrinos', r'$\nu_\mu + \bar{\nu}_\mu + \nu_e + \bar{\nu}_e$')]
CRIT = rf'$10^{{{round(np.log10(RTOL))}}}$'  # '$10^{-2}$'
DEVIATION = '|am3_mojo / C++ − 1|'


def worst_flux(R, T, step):
    """Worst deviation over all fluxes, as the reference tests check it (tests/support/compare.py)."""
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
    fig, axes = plt.subplots(2, 4, figsize=(16.6, 7.2), sharex='col', layout='constrained',
                             gridspec_kw=dict(height_ratios=[3, 2.4]))
    for c, (getter, grid, label) in enumerate(SPECIES):
        ax, axd = axes[0, c], axes[1, c]
        E = R[f'{last:06d}/{grid}']
        ref = R[f'{last:06d}/{getter}']
        ax.set_title(label)
        ax.set_xscale('log')
        axd.set_xlabel(r'$E$ [eV]')
        if not np.any(ref > 0):
            ax.text(0.5, 0.5, 'not simulated\nin this scenario', ha='center', va='center', transform=ax.transAxes)
            for a in (ax, axd):
                a.set(xticks=[], yticks=[])
            continue
        for step, alpha in ((first, 0.45), (last, 1.0)):
            ax.loglog(E, E * R[f'{step:06d}/{getter}'], color='k', lw=5, alpha=0.3 * alpha + 0.05,
                      label=f'C++ AM3, step {step}')
            ax.loglog(E, E * M[f'{step:06d}/{getter}'], color='#1f77b4', lw=2.2, ls='--', alpha=alpha,
                      label=f'am3_mojo, step {step}')
        pk = np.max(E * ref)
        ax.set_ylim(pk * FLOOR, pk * 10)  # the range the reference tests check
        ax.grid(alpha=0.25)
        axd.semilogy(E, rel(ref, M[f'{last:06d}/{getter}'], E), ls='none', marker='.', ms=5, color='#1f77b4')
        axd.axhline(RTOL, color='r', lw=1.6, ls='--', label=f'{CRIT} criterion')
        axd.axhline(np.finfo(float).eps, color='k', lw=1.0, ls=':')
        axd.set_ylim(IDENT / 5, 1)
        axd.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(r'$E^2\,dN/dE\,dV$' + '\n' + r'[eV cm$^{-3}$]')
            axd.set_ylabel(f'deviation\nat step {last}')
    # the same lines in every panel: one legend below them
    c = next(c for c in range(4) if axes[0, c].get_legend_handles_labels()[0])
    (h, l), (hd, ld) = axes[0, c].get_legend_handles_labels(), axes[1, c].get_legend_handles_labels()
    fig.legend(h + hd, l + ld, loc='outside lower center', ncol=5, fontsize=12)
    fig.get_layout_engine().set(h_pad=0.1)  # (inches: some room above the legend)
    fig.suptitle(f'{sc}: am3_mojo vs C++ AM3 (with am3_fixes.patch) at step {last}\n'
                 f'worst deviation {DEVIATION} over all fluxes: {worst_flux(R, M, last):.1e}')
    path = os.path.join(outdir, f'spectra_{sc}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    outdir = results_dir('plots')
    written = []
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4.15 * len(SCENARIOS), 3.9), sharey=True,
                             layout='constrained')
    for ax, sc in zip(axes, sorted(SCENARIOS)):
        R, M = cached_scenario('cpp-fixed', sc), cached_scenario('mojo-am3', sc)
        written.append(plot_scenario(sc, R, M, outdir))
        steps = checkpoints(R)
        ax.semilogy(steps, [worst_flux(R, M, s) for s in steps], 'o-', color='#1f77b4')
        ax.axhline(RTOL, color='r', lw=1.6, ls='--')
        ax.set_title(sc)
        ax.set_xlabel('time step')
        ax.set_ylim(1e-12, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_yticks(10.0 ** np.arange(-12, 1, 2))  # (every second decade, with 1 and the criterion)
    axes[0].set_ylabel('worst deviation\nover all fluxes')
    fig.suptitle(f'am3_mojo vs C++ AM3 (with am3_fixes.patch) at every checkpoint (red: {CRIT} criterion)')
    path = os.path.join(outdir, 'deviation_vs_time.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    written.append(path)
    print('\n'.join(written))


if __name__ == '__main__':
    main()
