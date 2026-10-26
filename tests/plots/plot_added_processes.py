"""Effect of the processes am3_mojo adds to AM3 (e+e- annihilation, triplet pair production, neutron decay, kaons,
charm hadrons) on leptohadronic_all: am3_mojo with them (the default) against without them (`mojo-am3`, AM3's
physics). Six panels: the muon, electron and tau neutrinos (top row), the three neutrino flavours together, the photons
and the e+- (bottom row). Below each spectrum is the ratio of the two runs, below the three flavours their flavour
ratio. Dashed: the default run's components from the added processes and, for the neutrinos, from their parents in
photo-hadronic interactions: the pions made there, their muons, the kaons (with the pions and muons they decay into) and
the charm hadrons (with the kaons, pions and muons they decay into). Dotted: the neutrinos from the same parents in pp
collisions. Only the charm decays make nu_tau (D_s -> tau nu_tau and the tau decays), so AM3's physics has none. The
flavour ratio is each flavour's share of the neutrinos plus anti-neutrinos of all three flavours as they leave the
source (no oscillations).

usage: python tests/plots/plot_added_processes.py
Uses the runs cached in results/runs/ (runs whatever is missing).
Figure: results/plots/added_processes_leptohadronic_all.png
"""
import os, sys
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.compare import FLOOR, significant
from support.paths import results_dir
from support.runs import cached_scenario, checkpoints
from deviation import plt

# one colour per process or parent in every panel (the totals are blue and grey)
COLOURS = {'p-gamma pions': 'tab:olive', 'pp pions': 'tab:olive', 'p-gamma muons': 'tab:cyan',
           'pp muons': 'tab:cyan', 'p-gamma kaons': 'tab:purple', 'pp kaons': 'tab:purple',
           'annihilation': 'tab:orange', 'triplet': 'tab:green', 'neutron decay': 'tab:red',
           'kaon synchrotron': 'tab:pink', 'kaon inverse Compton': 'tab:brown',
           # (tab10 is used up)
           'p-gamma charm': 'goldenrod', 'pp charm': 'goldenrod', 'charm radiation': 'goldenrod'}
# the flavours of the flavour panel: colour (the nu_tau in that of their parents, the charm hadrons) and index
FLAVOURS = {'muon': ('#1f77b4', r'\mu'), 'electron': ('tab:red', 'e'), 'tau': ('goldenrod', r'\tau')}
WITHOUT = dict(color='k', lw=4.5, alpha=0.3)  # the run without the added processes, in every panel
E2_LABEL = r'$E^2\,dN/dE\,dV$' + '\n' + r'[eV cm$^{-3}$]'
E_LABEL = r'$E$ [eV]'
NU = lambda a: rf'$\nu_{a} + \bar{{\nu}}_{a}$'  # the neutrinos and anti-neutrinos of flavour index a
WITHOUT_LABEL = r'without the added processes (as AM$^3$)'
PG, PP = r'$\mathrm{p}\gamma$', r'$\mathrm{pp}$'
# the kaons, the charm hadrons and the charged ones, listed as in the README (the {} after a bar keeps the charge at
# the height of the others: mathtext raises a script after an accent)
KAONS = r'$\mathrm{K}^{\pm}$, $\mathrm{K}^0_\mathrm{S}$ and $\mathrm{K}^0_\mathrm{L}$'
CHARM = (r'$\mathrm{D}^{\pm}$, $\mathrm{D}^0$, $\bar{\mathrm{D}}{}^0$, $\mathrm{D}_s^{\pm}$, $\Lambda_c^{+}$ and '
         r'$\bar{\Lambda}{}_c^{-}$')
CHARGED_CHARM = r'$\mathrm{D}^{\pm}$, $\mathrm{D}_s^{\pm}$, $\Lambda_c^{+}$ and $\bar{\Lambda}{}_c^{-}$'
N_DECAY = r'$\mathrm{n} \to \mathrm{p}\,\mathrm{e}^{-}\bar{\nu}_e$'
TRIPLET = r'$\mathrm{e}^{\pm}\gamma \to \mathrm{e}^{\pm}\mathrm{e}^{+}\mathrm{e}^{-}$'
# the ranges of the four neutrino panels (the three flavours and all of them): E [eV] and E^2 dN/dE dV [eV cm^-3]
NU_XLIM, NU_YLIM = (1e3, 1e21), (1e4, 1e16)
NU_XTICKS = np.logspace(3, 21, 7)  # (every third decade, both ends labelled)
GAMMA_YLIM = (1e4, 1e16)  # the E^2 dN/dE dV range of the photon panel [eV cm^-3]

SCENARIO = 'leptohadronic_all'


def style(process):
    """The line of a component: the colour of its process or parent, dotted for the parents in pp collisions."""
    return dict(color=COLOURS[process], lw=1.8, ls=':' if process.startswith('pp ') else '--')


def legend_in(axl, entries, loc='center', **kw):
    """A legend of the entries (handle, label) at loc of the axis axl, which has nothing else. It may be wider than
    axl: the layout leaves it out (a wider legend would narrow the panels), and its row of the grid is sized to fit it.
    (axl spans one column: axes across columns would widen the space between all columns.)"""
    axl.axis('off')
    axl.legend(*zip(*entries), loc=loc, bbox_to_anchor=dict(center=(0.5, 0.5), right=(1, 0.5))[loc.split()[-1]],
               fontsize=12, handlelength=1.5, handletextpad=0.5, **kw).set_in_layout(False)


def spectrum_panel(ax, axr, title, E, f, am3, comps):
    """E^2 dN/dE of both runs (f: the default, am3: without the added processes) and of the default run's components
    (process, array), and their ratio below, on a log scale if it spans more than a factor 3."""
    pk = np.max(E * f)
    if np.any(am3 > 0):
        ax.loglog(E, E * am3, **WITHOUT)
    ax.loglog(E, E * f, color='#1f77b4', lw=2.4)
    for process, comp in comps:
        ax.loglog(E, E * comp, **style(process))
    ax.set_ylim(pk * 1e-12, pk * 10)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    sig = significant(am3, FLOOR, E) & (am3 > 0)  # number or energy
    if sig.any():
        r = np.where(sig, f / np.where(sig, am3, 1.0), np.nan)
        axr.axhline(1, color='k', lw=1.2)
        axr.semilogx(E, r, color='#1f77b4', lw=2.2)
        if np.nanmax(r) / np.nanmin(r) > 3:
            axr.set_yscale('log')
        else:
            axr.ticklabel_format(axis='y', useOffset=False)
    else:
        axr.text(0.5, 0.5, 'no ratio: no ' + r'$\nu_\tau$' + ' without\nthe added processes', transform=axr.transAxes,
                 ha='center', va='center')
        axr.set_yticks([])
    axr.grid(alpha=0.25)
    axr.set_xlabel(E_LABEL)


def shares(phis, E):
    """The flavour ratio: each flavour's share of the sum of the fluxes phis (flavour -> array), so that the shares add
    up to 1, in the bins above FLOOR of the peak of the sum (in number or energy)."""
    tot = sum(phis.values())
    sig = significant(tot, FLOOR, E) & (tot > 0)
    return {fl: np.where(sig, phi / np.where(sig, tot, 1.0), np.nan) for fl, phi in phis.items()}


def flavour_panel(ax, axr, E, f, am3):
    """E^2 dN/dE of the three flavours (flavour -> array) of both runs and their flavour ratio below (`shares`);
    returns the legend entries (handle, label).
    The run without the added processes has the line style of the other panels in the colour of each flavour."""
    pk = max(np.max(E * phi) for phi in f.values())
    fs, am3s = shares(f, E), shares(am3, E)
    without = []  # (one legend entry for the flavours without the added processes)
    for fl, (colour, a) in FLAVOURS.items():
        has = np.any(am3[fl] > 0)
        if has:
            line = dict(WITHOUT, color=colour)
            without += ax.loglog(E, E * am3[fl], **line)
            axr.semilogx(E, am3s[fl], **line)
        ax.loglog(E, E * f[fl], color=colour, lw=2.4,
                  label=NU(a) + (' (default)' if has else r' (default; AM$^3$ has none)'))
        axr.semilogx(E, fs[fl], color=colour, lw=2.2)
    ax.set_ylim(pk * 1e-12, pk * 10)
    ax.set_title(NU(r'\alpha') + r', $\alpha = \mu, e, \tau$')
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    axr.set_yscale('log')  # (to show the nu_tau)
    lo = np.nanmin([np.nanmin(s[s > 0]) for s in fs.values()])
    axr.set_ylim(10 ** np.floor(np.log10(lo)), 2)
    axr.text(0.97, 0.45, r'$f_\mu + f_e + f_\tau = 1$', transform=axr.transAxes, ha='right', va='center')
    axr.grid(alpha=0.25)
    axr.set_xlabel(E_LABEL)
    return [(tuple(without), WITHOUT_LABEL)] + list(zip(handles, labels))


def main():
    F = cached_scenario('mojo', SCENARIO)  # with the added processes
    A = cached_scenario('mojo-am3', SCENARIO)  # AM3's processes only
    st = f'{checkpoints(F)[-1]:06d}'
    g = lambda d, name: d[f'{st}/{name}']
    nu_mu = lambda d, c='': g(d, f'get_muon_neutrinos{c}') + g(d, f'get_muon_anti_neutrinos{c}')
    nu_e = lambda d, c='': g(d, f'get_electron_neutrinos{c}') + g(d, f'get_electron_anti_neutrinos{c}')
    nu_tau_of = lambda d, c: g(d, f'get_tau_neutrinos{c}') + g(d, f'get_tau_anti_neutrinos{c}')  # (no total getter)
    nu_tau = lambda d: nu_tau_of(d, '_charm_decay_photopion') + nu_tau_of(d, '_charm_decay_proton_proton')
    def parents(nu, *names):  # the neutrinos of the default run by parent (pions, muons, kaons, charm) and origin
        return [(f'{o} {n}', nu(F, f"_{n.removesuffix('s')}_decay_{origin}")) for n in names
                for o, origin in (('p-gamma', 'photopion'), ('pp', 'proton_proton'))]

    spectra = {  # (row, column): title, energy grid, total, components (process, getter or array)
        (0, 0): (NU(r'\mu'), 'get_egrid_neutrinos', nu_mu, parents(nu_mu, 'pions', 'muons', 'kaons', 'charm')),
        (0, 1): (NU('e'), 'get_egrid_neutrinos', nu_e,  # (the pion decays make no nu_e)
                 parents(nu_e, 'muons', 'kaons', 'charm')
                 + [('neutron decay', 'get_electron_anti_neutrinos_neutron_decay')]),
        (0, 2): (NU(r'\tau'), 'get_egrid_neutrinos', nu_tau, parents(nu_tau_of, 'charm')),
        (1, 1): (r'$\gamma$', 'get_egrid_photons', lambda d: g(d, 'get_photons'),
                 [('annihilation', 'get_photons_pair_annihilation'),
                  ('triplet', 'get_photons_triplet_pairs_syn_compton'),
                  ('neutron decay', 'get_photons_neutron_decay_electrons_syn_compton'),
                  ('kaon synchrotron', 'get_photons_kaons_syn'),
                  ('kaon inverse Compton', 'get_photons_kaons_compton'),
                  ('charm radiation', g(F, 'get_photons_charm_syn') + g(F, 'get_photons_charm_compton'))]),
        (1, 2): (r'$\mathrm{e}^{-} + \mathrm{e}^{+}$', 'get_egrid_lep', lambda d: g(d, 'get_pairs'),
                 [('triplet', 'get_pairs_triplet'), ('neutron decay', 'get_electrons_neutron_decay')]),
    }
    # the legends below the rows: the neutrinos (the colour of each parent, the line of each origin), the flavours, and
    # the photons and e+- (the photons of a process that makes e+- are their synchrotron and inverse Compton emission)
    key = lambda label, **line: (Line2D([], [], **line), label)
    colour = lambda process: dict(color=COLOURS[process], lw=3.5)
    totals = [key(WITHOUT_LABEL, **WITHOUT), key('am3_mojo (default)', color='#1f77b4', lw=2.4)]
    nu_key = totals + [  # (3 columns: the totals and the neutron decay, the mesons, the charm hadrons and the origins)
        key('from ' + N_DECAY, **colour('neutron decay')),
        key(r'from $\pi^{\pm}$ decays', **colour('p-gamma pions')),
        key(r'from $\mu^{\pm}$ decays', **colour('p-gamma muons')),
        key('from ' + KAONS + ' decay chains', **colour('p-gamma kaons')),
        key('from ' + CHARM + '\ndecay chains', **colour('p-gamma charm')),
        key('in ' + PG + ' interactions', color='k', lw=1.8, ls='--'),
        key('in ' + PP + ' collisions', color='k', lw=1.8, ls=':')]
    photon_key = totals + [
        key(r'from $\mathrm{e}^{+}\mathrm{e}^{-} \to \gamma\gamma$', **style('annihilation')),
        key('from ' + TRIPLET + '\n' + r'($\gamma$: syn. + IC of the pairs)', **style('triplet')),
        key('from ' + N_DECAY + '\n' + r'($\gamma$: syn. + IC of the $\mathrm{e}^{-}$)', **style('neutron decay')),
        key(r'from $\mathrm{K}^{\pm}$ synchrotron', **style('kaon synchrotron')),
        key(r'from $\mathrm{K}^{\pm}$ inverse Compton', **style('kaon inverse Compton')),
        key('from ' + CHARGED_CHARM + '\nsynchrotron + IC', **style('charm radiation'))]

    fig = plt.figure(figsize=(15.5, 15.6), layout='constrained')
    rows = fig.add_gridspec(6, 3, height_ratios=[2.8, 1.3, 1.15, 2.8, 1.3, 1.6])  # (two rows of panels and legends)
    def panel(r, c):  # a panel's main axis and the ratio axis below it, in row r (0 or 1) of the panels
        ax = fig.add_subplot(rows[3 * r, c])
        axr = fig.add_subplot(rows[3 * r + 1, c], sharex=ax)
        ax.tick_params(labelbottom=False)
        return ax, axr
    for (r, c), (title, grid, total, comps) in spectra.items():
        ax, axr = panel(r, c)
        spectrum_panel(ax, axr, title, g(F, grid), total(F), total(A),
                       [(process, g(F, comp) if isinstance(comp, str) else comp) for process, comp in comps])
        if grid == 'get_egrid_neutrinos':
            ax.set(xlim=NU_XLIM, ylim=NU_YLIM, xticks=NU_XTICKS)  # (the ratio below shares the x axis)
        elif grid == 'get_egrid_photons':
            ax.set_ylim(GAMMA_YLIM)
        if c == 0:
            ax.set_ylabel(E2_LABEL)
        if (r, c) != (0, 2):  # (the nu_tau have no ratio)
            axr.set_ylabel('with /\nwithout')
    ax, axr = panel(1, 0)
    flavours = lambda d: {'muon': nu_mu(d), 'electron': nu_e(d), 'tau': nu_tau(d)}
    flavour_key = flavour_panel(ax, axr, g(F, 'get_egrid_neutrinos'), flavours(F), flavours(A))
    ax.set(xlim=NU_XLIM, ylim=NU_YLIM, xticks=NU_XTICKS)
    ax.set_ylabel(E2_LABEL)
    axr.set_ylabel('flavour\n' + r'ratio $f_\alpha$')
    legend_in(fig.add_subplot(rows[2, 1]), nu_key, ncol=3)  # (centred on the panels)
    legend_in(fig.add_subplot(rows[5, 0]), flavour_key, handler_map={tuple: HandlerTuple(ndivide=None)})
    legend_in(fig.add_subplot(rows[5, 2]), photon_key, 'center right', ncol=2)  # (below the photons and e+-)
    fig.suptitle(f'{SCENARIO}, step {int(st)}: am3_mojo with and without '
                 r'$\mathrm{e}^{+}\mathrm{e}^{-} \to \gamma\gamma$, ' + TRIPLET + ',\n' + N_DECAY + ', kaons ('
                 + KAONS + ') and charm hadrons (' + CHARM + ')')
    path = os.path.join(results_dir('plots'), f'added_processes_{SCENARIO}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(path)


if __name__ == '__main__':
    main()
