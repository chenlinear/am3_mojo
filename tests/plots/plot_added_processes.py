"""Effect of the processes am3_mojo adds to AM3 (e+e- annihilation, triplet pair production, neutron decay,
kaons) on one scenario: am3_mojo with them (the default) against am3_mojo without them (`mojo-am3`, AM3's
physics). Dashed: the default run's components from the added processes and, for the neutrinos, from their
parents: photo-hadronic and pp pions (with their muons) and kaons (with the pions and muons of their decays).

usage: python tests/plots/plot_added_processes.py [--scenario leptohadronic_all] [--rerun]
Uses the runs cached in results/runs/ (runs whatever is missing).
Figure: results/plots/added_processes_<scenario>.png
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.compare import FLOOR, significant  # noqa: E402
from support.paths import results_dir  # noqa: E402
from support.runs import cached_scenario, checkpoints  # noqa: E402

# one colour per process in every panel (the totals are blue and grey)
COLOURS = {'p-gamma pions': 'tab:olive', 'pp pions': 'tab:cyan',
           'annihilation': 'tab:orange', 'triplet': 'tab:green', 'neutron decay': 'tab:red',
           'kaon decays': 'tab:purple', 'kaon synchrotron': 'tab:pink', 'kaon inverse Compton': 'tab:brown'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='leptohadronic_all')
    ap.add_argument('--rerun', action='store_true')
    a = ap.parse_args()
    F = cached_scenario('mojo', a.scenario, a.rerun)  # with the added processes
    A = cached_scenario('mojo-am3', a.scenario, a.rerun)  # AM3's processes only
    st = f'{checkpoints(F)[-1]:06d}'
    g = lambda d, name: d[f'{st}/{name}']
    nu_mu = lambda d, c='': g(d, f'get_muon_neutrinos{c}') + g(d, f'get_muon_anti_neutrinos{c}')  # noqa: E731
    nu_e = lambda d, c='': g(d, f'get_electron_neutrinos{c}') + g(d, f'get_electron_anti_neutrinos{c}')  # noqa: E731
    def parents(nu):  # the neutrinos by parent
        return [('from p-gamma pions and muons', 'p-gamma pions', lambda: nu(F, '_pion_muon_decay_photopion')),
                ('from pp pions and muons', 'pp pions', lambda: nu(F, '_pion_muon_decay_proton_proton')),
                ('from kaon decays (with their pions, muons)', 'kaon decays', lambda: nu(F, '_kaon_decay'))]

    panels = [  # components (label, process, a getter of the default run or a function giving the array)
        ('muon neutrinos + anti-neutrinos', 'get_egrid_neutrinos', nu_mu, parents(nu_mu)),
        ('electron neutrinos + anti-neutrinos', 'get_egrid_neutrinos', nu_e,
         parents(nu_e)[:2] + [('from neutron decay (anti-nu_e)', 'neutron decay',
                               'get_electron_anti_neutrinos_neutron_decay')] + parents(nu_e)[2:]),
        ('photons', 'get_egrid_photons', lambda d: g(d, 'get_photons'),
         [('from e+e- annihilation', 'annihilation', 'get_photons_pair_annihilation'),
          ('from triplet pairs (syn. + IC)', 'triplet', 'get_photons_triplet_pairs_syn_compton'),
          ('from neutron-decay electrons (syn. + IC)', 'neutron decay',
           'get_photons_neutron_decay_electrons_syn_compton'),
          ('from kaon synchrotron', 'kaon synchrotron', 'get_photons_kaons_syn'),
          ('from kaon inverse Compton', 'kaon inverse Compton', 'get_photons_kaons_compton')]),
        ('electrons + positrons', 'get_egrid_lep', lambda d: g(d, 'get_pairs'),
         [('from triplet pair production', 'triplet', 'get_pairs_triplet'),
          ('from neutron decay (e-)', 'neutron decay', 'get_electrons_neutron_decay')]),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(22, 7.5), sharex='col',
                             gridspec_kw=dict(height_ratios=[3, 1.6], hspace=0.06, wspace=0.25))
    for c, (title, grid, total, comps) in enumerate(panels):
        ax, axr = axes[0, c], axes[1, c]
        E = g(F, grid)
        f, am3 = total(F), total(A)
        pk = np.max(E * f)
        ax.loglog(E, E * am3, color='k', lw=3, alpha=0.3, label='without the added processes (as AM3)')
        ax.loglog(E, E * f, color='#1f77b4', lw=1.3, label='am3_mojo (default)')
        for label, process, comp in comps:
            ax.loglog(E, E * (g(F, comp) if isinstance(comp, str) else comp()), lw=1.0, ls='--',
                      color=COLOURS[process], label=label)
        ax.set_ylim(pk * 1e-12, pk * 10)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='lower left')
        sig = significant(am3, FLOOR, E) & (am3 > 0)  # number or energy
        ratio = np.where(sig, f / np.where(sig, am3, 1.0), np.nan)
        axr.semilogx(E, ratio, color='#1f77b4', lw=1.2)
        axr.axhline(1, color='k', lw=0.6)
        if np.nanmax(ratio) / np.nanmin(ratio) > 3:
            axr.set_yscale('log')
        else:
            axr.ticklabel_format(axis='y', useOffset=False)
        axr.grid(alpha=0.25)
        axr.set_xlabel('E [eV]')
        if c == 0:
            ax.set_ylabel(r'$E^2\,dN/dE\,dV$  [eV cm$^{-3}$]')
            axr.set_ylabel('with / without')
    fig.suptitle(f'{a.scenario}, step {int(st)}: am3_mojo with and without e+e- annihilation, triplet pair '
                 'production, neutron decay and kaons')
    path = os.path.join(results_dir('plots'), f'added_processes_{a.scenario}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(path)


if __name__ == '__main__':
    main()
