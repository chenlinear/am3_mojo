"""Effect of the processes that am3_mojo adds to AM3 (e+e- annihilation,
triplet pair production, neutron decay, kaons, charm hadrons) on
leptohadronic_all. The figure compares am3_mojo with them (the default) and
without them (`mojo-am3`, AM3's physics). Six panels: the muon, electron and
tau neutrinos (top row), the three neutrino flavors together, the photons and
the e+- (bottom row). Below each spectrum is the ratio of the two runs. Below
the three flavors is their flavor ratio. Dashed: the components of the default
run from the added processes and, for the neutrinos, from their parents in
photo-hadronic interactions. These parents are the pions made there, their
muons, the kaons (with the pions and muons they decay into) and the charm
hadrons (with the kaons, pions and muons they decay into). Dotted: the neutrinos
from the same parents in pp collisions. Only the charm decays make nu_tau
(D_s -> tau nu_tau and the tau decays), so AM3's physics has none. The flavor
ratio is each flavor's share of the neutrinos plus anti-neutrinos of all three
flavors as they leave the source (no oscillations).

usage: python tests/plots/plot_added_processes.py
Uses the runs cached in results/runs/ (runs whatever is missing).
Figure: results/plots/added_processes_leptohadronic_all.png
"""

import os, sys

import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
from support.compare import FLOOR, significant
from support.paths import published_dir
from support.runs import cached_scenario, checkpoints
from deviation import plt

# one color per process or parent in every panel (the totals are blue and gray)
COLOURS = {
    "p-gamma pions": "tab:olive",
    "pp pions": "tab:olive",
    "p-gamma muons": "tab:cyan",
    "pp muons": "tab:cyan",
    "p-gamma kaons": "tab:purple",
    "pp kaons": "tab:purple",
    "annihilation": "tab:orange",
    "triplet": "tab:green",
    "neutron decay": "tab:red",
    "kaon synchrotron": "tab:pink",
    "kaon inverse Compton": "tab:brown",
    # (tab10 is used up)
    "p-gamma charm": "goldenrod",
    "pp charm": "goldenrod",
    "charm radiation": "goldenrod",
}
# (the nu_tau in the color of their parents, the charm hadrons)
FLAVOURS = {
    "muon": ("#1f77b4", r"\mu"),
    "electron": ("tab:red", "e"),
    "tau": ("goldenrod", r"\tau"),
}
WITHOUT = dict(color="k", lw=4.5, alpha=0.3)
E2_LABEL = r"$E^2\,dN/dE\,dV$" + "\n" + r"[eV cm$^{-3}$]"
E_LABEL = r"$E$ [eV]"
NU = lambda a: rf"$\nu_{a} + \bar{{\nu}}_{a}$"
WITHOUT_LABEL = r"without the added processes (as AM$^3$)"
PG, PP = r"$\mathrm{p}\gamma$", r"$\mathrm{pp}$"
# The kaons, the charm hadrons and the charged ones, listed as in the README.
# The {} after a bar keeps the charge at the height of the others, because
# mathtext raises a script after an accent.
KAONS = (
    r"$\mathrm{K}^{\pm}$, $\mathrm{K}^0_\mathrm{S}$ and "
    r"$\mathrm{K}^0_\mathrm{L}$"
)
CHARM = (
    r"$\mathrm{D}^{\pm}$, $\mathrm{D}^0$, $\bar{\mathrm{D}}{}^0$, "
    r"$\mathrm{D}_s^{\pm}$, $\Lambda_c^{+}$ and $\bar{\Lambda}{}_c^{-}$"
)
CHARGED_CHARM = (
    r"$\mathrm{D}^{\pm}$, $\mathrm{D}_s^{\pm}$, $\Lambda_c^{+}$ "
    r"and $\bar{\Lambda}{}_c^{-}$"
)
N_DECAY = r"$\mathrm{n} \to \mathrm{p}\,\mathrm{e}^{-}\bar{\nu}_e$"
TRIPLET = (
    r"$\mathrm{e}^{\pm}\gamma \to "
    r"\mathrm{e}^{\pm}\mathrm{e}^{+}\mathrm{e}^{-}$"
)
NU_XLIM, NU_YLIM = (1e3, 1e21), (1e4, 1e16)
NU_XTICKS = np.logspace(3, 21, 7)  # (every third decade, both ends labeled)
GAMMA_YLIM = (1e4, 1e16)

SCENARIO = "leptohadronic_all"


def style(process):
    return dict(
        color=COLOURS[process],
        lw=1.8,
        ls=":" if process.startswith("pp ") else "--",
    )


def legend_in(axl, entries, loc="center", **kw):
    """The legend can be wider than axl. The layout leaves it out (in the
    layout, a wider legend narrows the panels), and its row of the grid is sized
    to fit it. (axl spans one column: axes across columns widen the space
    between all columns.)"""
    axl.axis("off")
    axl.legend(
        *zip(*entries),
        loc=loc,
        bbox_to_anchor=dict(center=(0.5, 0.5), right=(1, 0.5))[loc.split()[-1]],
        fontsize=12,
        handlelength=1.5,
        handletextpad=0.5,
        **kw,
    ).set_in_layout(False)


def spectrum_panel(ax, axr, title, E, f, am3, comps):
    pk = np.max(E * f)
    if np.any(am3 > 0):
        ax.loglog(E, E * am3, **WITHOUT)
    ax.loglog(E, E * f, color="#1f77b4", lw=2.4)
    for process, comp in comps:
        ax.loglog(E, E * comp, **style(process))
    ax.set_ylim(pk * 1e-12, pk * 10)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    sig = significant(am3, FLOOR, E) & (am3 > 0)
    if sig.any():
        r = np.where(sig, f / np.where(sig, am3, 1.0), np.nan)
        axr.axhline(1, color="k", lw=1.2)
        axr.semilogx(E, r, color="#1f77b4", lw=2.2)
        if np.nanmax(r) / np.nanmin(r) > 3:
            axr.set_yscale("log")
        else:
            axr.ticklabel_format(axis="y", useOffset=False)
    else:
        axr.text(
            0.5,
            0.5,
            "no ratio: no " + r"$\nu_\tau$" + " without\nthe added processes",
            transform=axr.transAxes,
            ha="center",
            va="center",
        )
        axr.set_yticks([])
    axr.grid(alpha=0.25)
    axr.set_xlabel(E_LABEL)


def shares(phis, E):
    tot = sum(phis.values())
    sig = significant(tot, FLOOR, E) & (tot > 0)
    return {
        fl: np.where(sig, phi / np.where(sig, tot, 1.0), np.nan)
        for fl, phi in phis.items()
    }


def flavour_panel(ax, axr, E, f, am3):
    pk = max(np.max(E * phi) for phi in f.values())
    fs, am3s = shares(f, E), shares(am3, E)
    # (one legend entry for the flavors without the added processes)
    without = []
    for fl, (colour, a) in FLAVOURS.items():
        has = np.any(am3[fl] > 0)
        if has:
            line = dict(WITHOUT, color=colour)
            without += ax.loglog(E, E * am3[fl], **line)
            axr.semilogx(E, am3s[fl], **line)
        ax.loglog(
            E,
            E * f[fl],
            color=colour,
            lw=2.4,
            label=NU(a)
            + (" (default)" if has else r" (default; AM$^3$ has none)"),
        )
        axr.semilogx(E, fs[fl], color=colour, lw=2.2)
    ax.set_ylim(pk * 1e-12, pk * 10)
    ax.set_title(NU(r"\alpha") + r", $\alpha = \mu, e, \tau$")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    axr.set_yscale("log")  # (to show the nu_tau)
    lo = np.nanmin([np.nanmin(s[s > 0]) for s in fs.values()])
    axr.set_ylim(10 ** np.floor(np.log10(lo)), 2)
    axr.text(
        0.97,
        0.45,
        r"$f_\mu + f_e + f_\tau = 1$",
        transform=axr.transAxes,
        ha="right",
        va="center",
    )
    axr.grid(alpha=0.25)
    axr.set_xlabel(E_LABEL)
    return [(tuple(without), WITHOUT_LABEL)] + list(zip(handles, labels))


def main():
    out = published_dir("plots")  # (before the runs: stops a variant)
    F = cached_scenario("mojo", SCENARIO)
    A = cached_scenario("mojo-am3", SCENARIO)
    st = f"{checkpoints(F)[-1]:06d}"
    g = lambda d, name: d[f"{st}/{name}"]
    nu_mu = lambda d, c="": (
        g(d, f"get_muon_neutrinos{c}") + g(d, f"get_muon_anti_neutrinos{c}")
    )
    nu_e = lambda d, c="": (
        g(d, f"get_electron_neutrinos{c}")
        + g(d, f"get_electron_anti_neutrinos{c}")
    )
    nu_tau_of = lambda d, c: (
        g(d, f"get_tau_neutrinos{c}") + g(d, f"get_tau_anti_neutrinos{c}")
    )  # (no total getter)
    nu_tau = lambda d: (
        nu_tau_of(d, "_charm_decay_photopion")
        + nu_tau_of(d, "_charm_decay_proton_proton")
    )

    def parents(nu, *names):
        return [
            (f"{o} {n}", nu(F, f"_{n.removesuffix('s')}_decay_{origin}"))
            for n in names
            for o, origin in (("p-gamma", "photopion"), ("pp", "proton_proton"))
        ]

    spectra = {
        (0, 0): (
            NU(r"\mu"),
            "get_egrid_neutrinos",
            nu_mu,
            parents(nu_mu, "pions", "muons", "kaons", "charm"),
        ),
        # (the pion decays make no nu_e)
        (0, 1): (
            NU("e"),
            "get_egrid_neutrinos",
            nu_e,
            parents(nu_e, "muons", "kaons", "charm")
            + [("neutron decay", "get_electron_anti_neutrinos_neutron_decay")],
        ),
        (0, 2): (
            NU(r"\tau"),
            "get_egrid_neutrinos",
            nu_tau,
            parents(nu_tau_of, "charm"),
        ),
        (1, 1): (
            r"$\gamma$",
            "get_egrid_photons",
            lambda d: g(d, "get_photons"),
            [
                ("annihilation", "get_photons_pair_annihilation"),
                ("triplet", "get_photons_triplet_pairs_syn_compton"),
                (
                    "neutron decay",
                    "get_photons_neutron_decay_electrons_syn_compton",
                ),
                ("kaon synchrotron", "get_photons_kaons_syn"),
                ("kaon inverse Compton", "get_photons_kaons_compton"),
                (
                    "charm radiation",
                    g(F, "get_photons_charm_syn")
                    + g(F, "get_photons_charm_compton"),
                ),
            ],
        ),
        (1, 2): (
            r"$\mathrm{e}^{-} + \mathrm{e}^{+}$",
            "get_egrid_lep",
            lambda d: g(d, "get_pairs"),
            [
                ("triplet", "get_pairs_triplet"),
                ("neutron decay", "get_electrons_neutron_decay"),
            ],
        ),
    }
    key = lambda label, **line: (Line2D([], [], **line), label)
    colour = lambda process: dict(color=COLOURS[process], lw=3.5)
    totals = [
        key(WITHOUT_LABEL, **WITHOUT),
        key("am3_mojo (default)", color="#1f77b4", lw=2.4),
    ]
    # (3 columns: the totals and the neutron decay, the mesons, the charm
    # hadrons and the origins)
    nu_key = totals + [
        key("from " + N_DECAY, **colour("neutron decay")),
        key(r"from $\pi^{\pm}$ decays", **colour("p-gamma pions")),
        key(r"from $\mu^{\pm}$ decays", **colour("p-gamma muons")),
        key("from " + KAONS + " decay chains", **colour("p-gamma kaons")),
        key("from " + CHARM + "\ndecay chains", **colour("p-gamma charm")),
        key("in " + PG + " interactions", color="k", lw=1.8, ls="--"),
        key("in " + PP + " collisions", color="k", lw=1.8, ls=":"),
    ]
    photon_key = totals + [
        key(
            r"from $\mathrm{e}^{+}\mathrm{e}^{-} \to \gamma\gamma$",
            **style("annihilation"),
        ),
        key(
            "from " + TRIPLET + "\n" + r"($\gamma$: syn. + IC of the pairs)",
            **style("triplet"),
        ),
        key(
            "from "
            + N_DECAY
            + "\n"
            + r"($\gamma$: syn. + IC of the $\mathrm{e}^{-}$)",
            **style("neutron decay"),
        ),
        key(
            r"from $\mathrm{K}^{\pm}$ synchrotron", **style("kaon synchrotron")
        ),
        key(
            r"from $\mathrm{K}^{\pm}$ inverse Compton",
            **style("kaon inverse Compton"),
        ),
        key(
            "from " + CHARGED_CHARM + "\nsynchrotron + IC",
            **style("charm radiation"),
        ),
    ]

    fig = plt.figure(figsize=(15.5, 15.6), layout="constrained")
    rows = fig.add_gridspec(6, 3, height_ratios=[2.8, 1.3, 1.15, 2.8, 1.3, 1.6])

    def panel(r, c):
        ax = fig.add_subplot(rows[3 * r, c])
        axr = fig.add_subplot(rows[3 * r + 1, c], sharex=ax)
        ax.tick_params(labelbottom=False)
        return ax, axr

    for (r, c), (title, grid, total, comps) in spectra.items():
        ax, axr = panel(r, c)
        spectrum_panel(
            ax,
            axr,
            title,
            g(F, grid),
            total(F),
            total(A),
            [
                (process, g(F, comp) if isinstance(comp, str) else comp)
                for process, comp in comps
            ],
        )
        if grid == "get_egrid_neutrinos":
            # (the ratio below shares the x axis)
            ax.set(xlim=NU_XLIM, ylim=NU_YLIM, xticks=NU_XTICKS)
        elif grid == "get_egrid_photons":
            ax.set_ylim(GAMMA_YLIM)
        if c == 0:
            ax.set_ylabel(E2_LABEL)
        if (r, c) != (0, 2):  # (the nu_tau have no ratio)
            axr.set_ylabel("with /\nwithout")
    ax, axr = panel(1, 0)
    flavours = lambda d: {
        "muon": nu_mu(d),
        "electron": nu_e(d),
        "tau": nu_tau(d),
    }
    flavour_key = flavour_panel(
        ax, axr, g(F, "get_egrid_neutrinos"), flavours(F), flavours(A)
    )
    ax.set(xlim=NU_XLIM, ylim=NU_YLIM, xticks=NU_XTICKS)
    ax.set_ylabel(E2_LABEL)
    axr.set_ylabel("flavour\n" + r"ratio $f_\alpha$")
    # (centered on the panels)
    legend_in(fig.add_subplot(rows[2, 1]), nu_key, ncol=3)
    legend_in(
        fig.add_subplot(rows[5, 0]),
        flavour_key,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    # (below the photons and e+-)
    legend_in(fig.add_subplot(rows[5, 2]), photon_key, "center right", ncol=2)
    fig.suptitle(
        f"{SCENARIO}, step {int(st)}: am3_mojo with and without "
        r"$\mathrm{e}^{+}\mathrm{e}^{-} \to \gamma\gamma$, "
        + TRIPLET
        + ",\n"
        + N_DECAY
        + ", kaons ("
        + KAONS
        + ") and charm hadrons ("
        + CHARM
        + ")"
    )
    path = os.path.join(out, f"added_processes_{SCENARIO}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
