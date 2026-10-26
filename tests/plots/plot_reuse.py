"""Plot the difference between re-initialising the kernels for every simulation and reusing them (am3_mojo).

usage: python tests/plots/plot_reuse.py

Uses the runs of tests/support/reuse_runner.py (its modes and CONFIGS; the 4th configuration repeats the 1st) that
the unit and reference tests cache in results/reuse/, and runs the missing ones. am3_mojo runs fresh, reused and in
the two modes wrong on purpose (no_clear, no_escape_reset); mojo-am3 (only AM3's processes, the C++ run's baseline)
and the C++ AM3 (cpp-fixed, if built) run reused.

Figures in results/plots/:
  reuse_<setup>.png   per configuration, photon spectra with and without re-initialisation and their per-bin
                      relative difference
  reuse_summary.png   worst difference in the four spectra per configuration, and the time spent in init_kernels
"""
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.compare import FLOOR, grid_of, significant
from support.paths import available, results_dir
from support.reuse_runner import CONFIGS, N_STEPS, SETUPS
from support.runs import run_reuse
from deviation import IDENT, plt, rel

FIGS = results_dir('plots')
RUNS = [('mojo', 'fresh'), ('mojo', 'reused'), ('mojo', 'no_clear'),
        ('mojo', 'no_escape_reset'), ('mojo-am3', 'reused'), ('cpp', 'reused')]
N_CONFIGS = len(CONFIGS)
# (get_neutrinos has no nu_tau, as in AM3)
SPECIES = [('get_photons', 'get_egrid_photons', r'$\gamma$', '#1f77b4'),
           ('get_pairs', 'get_egrid_lep', r'$\mathrm{e}^{\pm}$', '#2ca02c'),
           ('get_protons', 'get_egrid_had', r'$\mathrm{p}$', '#9467bd'),
           ('get_neutrinos', 'get_egrid_neutrinos', r'$\nu_\mu + \bar{\nu}_\mu + \nu_e + \bar{\nu}_e$', '#8c564b')]
DIFFERENCE = '|x / re-initialised − 1|'
C_CPP = '#7f7f7f'
WRONG = [('mojo', 'no_clear', '#d62728', '^:', 'wrong: no clear_am3()'),  # impl, mode, colour, marker and line, label
         ('mojo', 'no_escape_reset', '#ff7f0e', 'v:', 'wrong: escape not set again')]


def ensure_runs():
    """{(setup): {(impl, mode): npz}}; the C++ run is used only if built."""
    D = {}
    for s in SETUPS:
        for impl, mode in RUNS:
            real = 'cpp-fixed' if impl == 'cpp' else impl
            if available(real):
                D.setdefault(s, {})[impl, mode] = run_reuse(real, s, mode)
    return D


def worst(R, T, k, names=None):
    """Largest relative difference over the getters `names` (default: all) of configuration k."""
    w = 0.0
    for n in R.files:
        if not n.startswith(f'{k}/') or (names and n.split('/')[1] not in names) or n not in T:
            continue  # (getters of am3_mojo's added processes have no C++ counterpart)
        r, t = R[n], T[n]
        if r.shape != t.shape or not np.all(np.isfinite(r)) or not np.any(r):
            continue
        E = grid_of(R, n, r.size) if (r.ndim == 1 and not n.split('/')[1].startswith('get_t_')) else None
        sig = significant(r, FLOOR, E)
        w = max(w, np.max(np.abs(t[sig] / r[sig] - 1.0), initial=0.0))
    return max(w, IDENT)


def worsts(D, impl, mode, names=None):
    """worst() of each configuration against the re-initialised run (the C++ run: against mojo-am3)."""
    ref = D['mojo-am3', 'reused'] if impl == 'cpp' else D['mojo', 'fresh']
    return [worst(ref, D[impl, mode], k, names) for k in range(N_CONFIGS)]


def label(c):
    return (f"B = {c['B']:g} G, t_esc = {c['t_esc']:g} R/c\n"
            f"t_exp = {c['t_exp']:g} R/c, n_pp = {c['n_pp']:g}\ndt = {c['dt']:g} R/c")


def plot_setup(setup, D):
    F, U, CP = D['mojo', 'fresh'], D['mojo', 'reused'], D.get(('cpp', 'reused'))
    fig, axes = plt.subplots(2, N_CONFIGS, figsize=(17, 10), sharex=True, layout='constrained',
                             gridspec_kw=dict(height_ratios=[3, 2.6]))
    for k in range(N_CONFIGS):
        ax, axd = axes[0, k], axes[1, k]
        E = F[f'{k}/get_egrid_photons']
        f, u = F[f'{k}/get_photons'], U[f'{k}/get_photons']
        ax.loglog(E, E * f, color='k', lw=6, alpha=0.25, label='init_kernels() for this run')
        ax.loglog(E, E * u, color='#1f77b4', lw=2.4, ls='--', label='kernels reused, clear_am3()')
        for impl, mode, col, _, lab in (WRONG if k > 0 else ()):
            ax.loglog(E, E * D[impl, mode][f'{k}/get_photons'], color=col, lw=1.6, alpha=0.8, label=lab)
        pk = np.max(E * f)
        ax.set_ylim(pk * 1e-12, pk * 30)
        title = f'run {k + 1}' + (' (repeat of run 1)' if k == N_CONFIGS - 1 else '')
        ax.set_title(f'{title}\n{label(CONFIGS[k])}', fontsize=15)
        ax.grid(alpha=0.25)
        if k == 0:
            ax.set_ylabel(r'$E^2\,dN/dE\,dV$ of $\gamma$' + '\n' + r'[eV cm$^{-3}$]')

        for getter, grid, name, col in SPECIES:
            Eg = F[f'{k}/{grid}']
            d = rel(F[f'{k}/{getter}'], U[f'{k}/{getter}'], Eg)
            axd.semilogy(Eg, d, ls='none', marker='o', ms=3.5, color=col,
                         label=f'reused vs re-init: {name}')
        if CP is not None:  # (the C++ code has only AM3's processes: against mojo-am3)
            axd.semilogy(E, rel(D['mojo-am3', 'reused'][f'{k}/get_photons'], CP[f'{k}/get_photons'], E), color=C_CPP,
                         lw=1.6, label=r'for scale: C++ AM3 vs am3_mojo ($\gamma$)')
        for impl, mode, col, _, lab in (WRONG if k > 0 else ()):
            axd.semilogy(E, rel(f, D[impl, mode][f'{k}/get_photons'], E), color=col, lw=1.6,
                         label=lab + r' ($\gamma$)')
        axd.axhline(IDENT, color='k', lw=1.2)
        axd.text(E[0], IDENT * 3, 'bitwise identical', fontsize=12)
        axd.axhline(np.finfo(float).eps, color='k', lw=1.0, ls=':')
        axd.set_ylim(IDENT / 10, 1e4)
        axd.set_xlabel(r'$E$ [eV]')
        axd.grid(alpha=0.25)
        if k == 0:
            axd.set_ylabel(f'{DIFFERENCE}\n' + r'(bins > $10^{-12}$ of peak)')
    # one legend for all panels: the spectra of the runs (top), the differences (bottom; the wrong runs once)
    top, bottom = axes[0, 1].get_legend_handles_labels(), axes[1, 1].get_legend_handles_labels()
    keep = [i for i, lab in enumerate(bottom[1]) if not lab.startswith('wrong:')]
    fig.legend(top[0] + [bottom[0][i] for i in keep], top[1] + [bottom[1][i] for i in keep],
               loc='outside lower center', ncol=3, fontsize=15)
    fig.get_layout_engine().set(h_pad=0.1)  # (inches: some room above the legend)
    fig.suptitle(f'{setup} setup, am3_mojo: kernels initialised once and reused (clear_am3() between runs)\n'
                 f'vs init_kernels() before every run — {N_STEPS} steps per run', fontsize=18)
    out = os.path.join(FIGS, f'reuse_{setup}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_summary(D_all):
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(14, 7), layout='constrained', gridspec_kw=dict(width_ratios=[1.6, 1]))
    runs = np.arange(1, N_CONFIGS + 1)
    styles = [('mojo', 'reused', '#1f77b4', 'o-', 'kernels reused (correct)'),
              ('cpp', 'reused', C_CPP, 's--', 'for scale: C++ AM3 reused vs am3_mojo')] + WRONG
    for si, setup in enumerate(SETUPS):
        D = D_all[setup]
        for impl, mode, col, sty, lab in styles:
            if (impl, mode) not in D:
                continue
            w = worsts(D, impl, mode, [g for g, *_ in SPECIES])
            ax.semilogy(runs + (si - 0.5) * 0.08, w, sty, color=col, ms=8,
                        mfc=col if si == 0 else 'white', label=lab if si == 0 else '_nolegend_')
    ax.axhline(IDENT, color='k', lw=1.2)
    ax.text(0.9, IDENT * 3, 'bitwise identical', fontsize=12)
    ax.set_xticks(runs, [f'run {k}' + ('\n(= run 1)' if k == N_CONFIGS else '') for k in runs])
    ax.set_ylim(IDENT / 10, 1e32)
    ax.set_ylabel(f'worst {DIFFERENCE}\n' + r'in the $\gamma$, $\mathrm{e}^{\pm}$, $\mathrm{p}$, $\nu$ spectra')
    ax.set_title('Difference from re-initialising the kernels before every run\n'
                 '(filled: leptohadronic setup, open: blazar setup)', fontsize=15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.5)

    x = np.arange(len(SETUPS))
    ini_f = [float(D_all[s]['mojo', 'fresh']['__init_s']) for s in SETUPS]
    ini_r = [float(D_all[s]['mojo', 'reused']['__init_s']) for s in SETUPS]
    ev = [float(D_all[s]['mojo', 'reused']['__evolve_s']) for s in SETUPS]
    axt.bar(x - 0.2, ini_f, 0.4, color='0.6', label=f'init_kernels() before\neach of {N_CONFIGS} runs')
    axt.bar(x + 0.2, ini_r, 0.4, color='#1f77b4', label='init_kernels() once,\nreused')
    for xi, (a, b) in enumerate(zip(ini_f, ini_r)):
        axt.text(xi - 0.2, a, f'{a:.1f} s', ha='center', va='bottom', fontsize=12)
        axt.text(xi + 0.2, b, f'{b:.1f} s', ha='center', va='bottom', fontsize=12)
    axt.set_xticks(x, [f'{s}\n(evolving:\n{e:.1f} s)' for s, e in zip(SETUPS, ev)])
    axt.set_ylim(0, max(ini_f) * 1.5)
    axt.set_ylabel('time in init_kernels() [s]')
    axt.set_title(f'Kernel set-up time for {N_CONFIGS} runs\nof {N_STEPS} steps (am3_mojo)', fontsize=15)
    axt.legend(fontsize=12, loc='upper center')
    out = os.path.join(FIGS, 'reuse_summary.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    D_all = ensure_runs()
    written = [plot_setup(s, D_all[s]) for s in SETUPS]
    written.append(plot_summary(D_all))
    print('worst relative difference from re-initialising (all getters; the C++ AM3,\n'
          'which has only AM3\'s processes, against am3_mojo without the added ones):')
    for s in SETUPS:
        for i, m in RUNS[1:]:
            if (i, m) in D_all[s] and i != 'mojo-am3':
                print(f'  {s:14s} {i:4s} {m:16s} ' + '  '.join(
                    'identical' if x == IDENT else f'{x:9.1e}' for x in worsts(D_all[s], i, m)))
    print('figures:\n  ' + '\n  '.join(written))


if __name__ == '__main__':
    main()
