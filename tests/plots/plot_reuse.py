"""Plot the difference between re-initialising the kernels for every
simulation and reusing them (am3_mojo).

usage: python tests/plots/plot_reuse.py [--rerun]

Uses the runs of tests/support/reuse_runner.py that the unit and reference
tests cache in results/reuse/, and runs the missing ones:
  mojo        fresh             new AM3 + init_kernels() per configuration
  mojo        reused            init_kernels() once, clear_am3() in between
  mojo        no_clear          wrong on purpose: clear_am3() never called
  mojo        no_escape_reset   wrong on purpose: escape timescale not set again
  mojo-am3    reused            the same with only AM3's processes, the C++ run's baseline
  cpp-fixed   reused            the C++ AM3 doing the correct reuse (if built)
The configurations are CONFIGS in tests/support/reuse_runner.py; the 4th
repeats the 1st.

Figures in results/plots/:
  reuse_<setup>.png   photon spectra with and without re-initialisation, and
                      their per-bin relative difference, per configuration
  reuse_summary.png   worst difference in the four spectra per configuration,
                      and the time spent in init_kernels
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.compare import FLOOR, grid_of, significant  # noqa: E402
from support.paths import available, results_dir  # noqa: E402
from support.reuse_runner import CONFIGS, N_STEPS, SETUPS  # noqa: E402
from support.runs import run_reuse  # noqa: E402
from deviation import IDENT, rel  # noqa: E402

FIGS = results_dir('plots')
RUNS = [('mojo', 'fresh'), ('mojo', 'reused'), ('mojo', 'no_clear'),
        ('mojo', 'no_escape_reset'), ('mojo-am3', 'reused'), ('cpp', 'reused')]
N_CONFIGS = len(CONFIGS)
SPECIES = [('get_photons', 'get_egrid_photons', 'photons', '#1f77b4'),
           ('get_pairs', 'get_egrid_lep', 'e±', '#2ca02c'),
           ('get_protons', 'get_egrid_had', 'protons', '#9467bd'),
           ('get_neutrinos', 'get_egrid_neutrinos', 'neutrinos', '#8c564b')]
C_NOCLEAR, C_NOESC, C_CPP = '#d62728', '#ff7f0e', '#7f7f7f'


def ensure_runs(rerun):
    """{(setup): {(impl, mode): npz}}; the C++ run is used only if built."""
    D = {}
    for s in SETUPS:
        for impl, mode in RUNS:
            real = 'cpp-fixed' if impl == 'cpp' else impl
            if available(real):
                D.setdefault(s, {})[impl, mode] = run_reuse(real, s, mode, rerun)
    return D


def worst(R, T, k, names=None):
    """Largest relative difference over the getters `names` (default: all)
    of configuration k."""
    w = 0.0
    for n in R.files:
        if not n.startswith(f'{k}/') or (names and n.split('/')[1] not in names) or n not in T:
            continue  # (getters of am3_mojo's added processes have no C++ counterpart)
        r, t = R[n], T[n]
        if r.shape != t.shape or r.size == 0 or not np.all(np.isfinite(r)) or not np.any(r != 0):
            continue
        E = grid_of(R, n, r.size) if (r.ndim == 1 and not n.split('/')[1].startswith('get_t_')) else None
        sig = significant(r, FLOOR, E)
        w = max(w, np.max(np.abs(t[sig] / r[sig] - 1.0), initial=0.0))
    return max(w, IDENT)


def label(c):
    return (f"B = {c['B']:g} G, t_esc = {c['t_esc']:g} R/c\n"
            f"t_exp = {c['t_exp']:g} R/c, n_pp = {c['n_pp']:g}, dt = {c['dt']:g} R/c")


def plot_setup(setup, D):
    F, U = D['mojo', 'fresh'], D['mojo', 'reused']
    NC, NE, CP = D['mojo', 'no_clear'], D['mojo', 'no_escape_reset'], D.get(('cpp', 'reused'))
    fig, axes = plt.subplots(2, N_CONFIGS, figsize=(19, 8.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 2.4], hspace=0.07, wspace=0.22))
    for k in range(N_CONFIGS):
        ax, axd = axes[0, k], axes[1, k]
        E = F[f'{k}/get_egrid_photons']
        f, u = F[f'{k}/get_photons'], U[f'{k}/get_photons']
        ax.loglog(E, E * f, color='k', lw=4, alpha=0.25, label='init_kernels() for this run')
        ax.loglog(E, E * u, color='#1f77b4', lw=1.3, ls='--', label='kernels reused, clear_am3()')
        if k > 0:
            ax.loglog(E, E * NC[f'{k}/get_photons'], color=C_NOCLEAR, lw=0.8, alpha=0.8,
                      label='wrong: no clear_am3()')
            ax.loglog(E, E * NE[f'{k}/get_photons'], color=C_NOESC, lw=0.8, alpha=0.8,
                      label='wrong: escape not set again')
        pk = np.max(E * f)
        ax.set_ylim(pk * 1e-12, pk * 30)
        title = f'run {k + 1}' + (' (repeat of run 1)' if k == N_CONFIGS - 1 else '')
        ax.set_title(f'{title}\n{label(CONFIGS[k])}', fontsize=9)
        ax.grid(alpha=0.25)
        if k == 0:
            ax.set_ylabel(r'photons  $E^2\,dN/dE\,dV$  [eV cm$^{-3}$]')
        ax.legend(fontsize=7, loc='lower center')

        for getter, grid, name, col in SPECIES:
            Eg = F[f'{k}/{grid}']
            d = rel(F[f'{k}/{getter}'], U[f'{k}/{getter}'], Eg)
            axd.semilogy(Eg, d, ls='none', marker='o', ms=2.2, color=col,
                         label=f'reused vs re-init: {name}')
        # the C++ code has only AM3's processes: compare it with mojo-am3
        A = D.get(('mojo-am3', 'reused'))
        d = rel(A[f'{k}/get_photons'], CP[f'{k}/get_photons'], E) if CP is not None else None
        if CP is not None:
            axd.semilogy(E, d, color=C_CPP, lw=0.9, label='for scale: C++ AM3 vs am3_mojo (photons)')
        if k > 0:
            axd.semilogy(E, rel(f, NC[f'{k}/get_photons'], E), color=C_NOCLEAR, lw=0.9,
                         label='wrong: no clear_am3() (photons)')
            axd.semilogy(E, rel(f, NE[f'{k}/get_photons'], E), color=C_NOESC, lw=0.9,
                         label='wrong: escape not set again (photons)')
        axd.axhline(IDENT, color='k', lw=0.6)
        axd.text(E[0], IDENT * 3, 'bitwise identical', fontsize=7.5)
        axd.axhline(np.finfo(float).eps, color='k', lw=0.5, ls=':')
        axd.set_ylim(IDENT / 10, 1e4)
        axd.set_xlabel('E [eV]')
        axd.grid(alpha=0.25)
        if k == 0:
            axd.set_ylabel('|x / re-initialised − 1|\n(bins > 1e-12 of peak)')
        same = all(np.array_equal(F[f'{k}/{g}'], U[f'{k}/{g}']) for g, *_ in SPECIES)
        n_same = sum(np.array_equal(F[n], U[n], equal_nan=True) for n in F.files if n.startswith(f'{k}/'))
        n_all = sum(1 for n in F.files if n.startswith(f'{k}/'))
        axd.text(0.5, 0.2, ('reused = re-initialised, bitwise,\n' if same else 'DIFFERENT\n') +
                 f'{n_same} of {n_all} getters identical',
                 transform=axd.transAxes, ha='center', fontsize=8.5, color='#1f77b4',
                 bbox=dict(fc='white', ec='#1f77b4', lw=0.6))
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f'{setup} setup, am3_mojo: kernels initialised once and reused (clear_am3() '
                 f'between runs) vs init_kernels() before every run — {N_STEPS} steps per run',
                 fontsize=12)
    out = os.path.join(FIGS, f'reuse_{setup}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_summary(D_all):
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    runs = np.arange(1, N_CONFIGS + 1)
    styles = [('reused', 'mojo', '#1f77b4', 'o-', 'kernels reused (correct)'),
              ('reused', 'cpp', C_CPP, 's--', 'for scale: C++ AM3 reused vs am3_mojo'),
              ('no_clear', 'mojo', C_NOCLEAR, '^:', 'wrong: no clear_am3()'),
              ('no_escape_reset', 'mojo', C_NOESC, 'v:', 'wrong: escape not set again')]
    for si, setup in enumerate(SETUPS):
        D = D_all[setup]
        F = D['mojo', 'fresh']
        for mode, impl, col, sty, lab in styles:
            if (impl, mode) not in D:
                continue
            ref = D['mojo-am3', 'reused'] if impl == 'cpp' else F
            w = [worst(ref, D[impl, mode], k, [g for g, *_ in SPECIES]) for k in range(N_CONFIGS)]
            ax.semilogy(runs + (si - 0.5) * 0.08, w, sty, color=col, ms=6,
                        mfc=col if si == 0 else 'white', label=f'{lab}, {setup}')
    ax.axhline(IDENT, color='k', lw=0.6)
    ax.text(0.9, IDENT * 3, 'bitwise identical', fontsize=8)
    ax.set_xticks(runs, [f'run {k}' + (' (= run 1)' if k == N_CONFIGS else '') for k in runs])
    ax.set_ylim(IDENT / 10, 1e32)
    ax.set_ylabel('worst |x / re-initialised − 1|\nphoton, e±, proton and neutrino spectra')
    ax.set_title('Difference from re-initialising the kernels before every run\n'
                 '(filled: leptohadronic setup, open: blazar setup)', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc='upper left', ncol=2)

    x = np.arange(len(SETUPS))
    ini_f = [float(D_all[s]['mojo', 'fresh']['__init_s']) for s in SETUPS]
    ini_r = [float(D_all[s]['mojo', 'reused']['__init_s']) for s in SETUPS]
    ev = [float(D_all[s]['mojo', 'reused']['__evolve_s']) for s in SETUPS]
    axt.bar(x - 0.2, ini_f, 0.4, color='0.6', label=f'init_kernels() before each of {N_CONFIGS} runs')
    axt.bar(x + 0.2, ini_r, 0.4, color='#1f77b4', label='init_kernels() once, reused')
    for xi, (a, b, e) in enumerate(zip(ini_f, ini_r, ev)):
        axt.text(xi - 0.2, a, f'{a:.1f} s', ha='center', va='bottom', fontsize=8)
        axt.text(xi + 0.2, b, f'{b:.1f} s', ha='center', va='bottom', fontsize=8)
        axt.text(xi, max(a, b) * 1.12, f'(evolving: {e:.1f} s for {N_CONFIGS}×{N_STEPS} steps)',
                 ha='center', fontsize=7.5, color='0.3')
    axt.set_xticks(x, SETUPS)
    axt.set_ylim(0, max(ini_f) * 1.3)
    axt.set_ylabel('time in init_kernels() [s]')
    axt.set_title(f'Kernel set-up time for {N_CONFIGS} runs (am3_mojo)', fontsize=10)
    axt.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    out = os.path.join(FIGS, 'reuse_summary.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    D_all = ensure_runs('--rerun' in sys.argv)
    written = [plot_setup(s, D_all[s]) for s in SETUPS]
    written.append(plot_summary(D_all))
    print('worst relative difference from re-initialising (all getters; the C++ AM3,\n'
          'which has only AM3\'s processes, against am3_mojo without the added ones):')
    for s in SETUPS:
        F = D_all[s]['mojo', 'fresh']
        for i, m in RUNS[1:]:
            if (i, m) not in D_all[s] or i == 'mojo-am3':
                continue
            ref = D_all[s]['mojo-am3', 'reused'] if i == 'cpp' else F
            w = [worst(ref, D_all[s][i, m], k) for k in range(N_CONFIGS)]
            print(f'  {s:14s} {i:4s} {m:16s} ' + '  '.join(
                'identical' if x == IDENT else f'{x:9.1e}' for x in w))
    print('figures:\n  ' + '\n  '.join(written))


if __name__ == '__main__':
    main()
