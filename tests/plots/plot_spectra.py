"""Plot am3_mojo against the C++ AM3, patched (with the fixes of
tests/reference/am3_fixes.patch, as in the reference tests) and original. The
figures show the photon, e+/-, proton and neutrino spectra of every scenario
with the per-bin relative deviation from each. They also show the worst flux
deviation from each at every checkpoint. The deviation is that of
tests/support/compare.py, max(r, 1/r) - 1 with r = am3_mojo / C++. The red line
marks the criterion of the patched C++ (RTOL there).
results/reference/summary.md has the worst deviations. The original has no
criterion, because am3_mojo fixes its bugs (tests/reference/README.md). Where
only one of the two has particles in a bin, the deviation is infinite:
deviations from OFF_SCALE up are drawn at the dotted line.

usage: python tests/plots/plot_spectra.py
Uses the runs cached in results/runs/ by the reference tests (runs whatever is
missing).
Figures: results/plots/spectra_<scenario>.png, deviation_vs_time.png
"""

import os, sys

import numpy as np

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
from support.compare import FLOOR, RTOL, deviations, grid_of, kind, merged
from support.paths import published_dir
from support.runs import cached_scenario, checkpoints
from support.scenarios import SCENARIOS
from deviation import IDENT, plt, rel

# (mojo-am3 and the C++ AM3 have no nu_tau)
SPECIES = [
    ("get_photons", "get_egrid_photons", r"$\gamma$"),
    ("get_pairs", "get_egrid_lep", r"$\mathrm{e}^{-} + \mathrm{e}^{+}$"),
    ("get_protons", "get_egrid_had", r"$\mathrm{p}$"),
    (
        "get_neutrinos",
        "get_egrid_neutrinos",
        r"$\nu_\mu + \bar{\nu}_\mu + \nu_e + \bar{\nu}_e$",
    ),
]
CRITERION = rf"criterion, C++ patched: $10^{{{round(np.log10(RTOL))}}}$"
# where larger deviations and infinite ones are drawn
OFF_SCALE = 1e2
OFF_SCALE_LABEL = r"$10^{2}$ or more, or $\infty$"
C_MOJO, C_ORIGINAL = "#1f77b4", "#009e73"
DEVIATION = r"$\max(r, 1/r) - 1$"
VS_PATCHED, VS_ORIGINAL = "am3_mojo vs C++ patched", "am3_mojo vs C++ original"
# What the original leaves empty where the positrons are merged into the
# electrons. The original evolves only all pairs, which am3_mojo and the patched
# C++ split into these.
EMPTY_IN_ORIGINAL_IF_MERGED = ("get_electrons", "get_positrons")


def mark_limits(ax):
    ax.axhline(RTOL, color="r", lw=1.6, ls="--", label=CRITERION)
    ax.axhline(OFF_SCALE, color="k", lw=1.0, ls=":", label=OFF_SCALE_LABEL)


def shown(x):
    return r"$\infty$" if np.isinf(x) else f"{x:.1e}"


def worst_flux(R, T, step, original=False):
    """Worst deviation over all fluxes of T from R, as the reference tests check
    it (tests/support/compare.py). original: R is the original C++, without
    EMPTY_IN_ORIGINAL_IF_MERGED where the positrons are merged, because
    get_pairs holds their particles. (Without this exception, they deviate
    infinitely in every bin.) A flux that T lacks or holds in another shape is
    an infinite deviation, because compare() lists it as a mismatch."""
    w = 0.0
    for k in R.files:
        if (
            k.startswith(f"{step:06d}/")
            and kind(k) == "flux"
            and not (
                original
                and k.split("/")[-1] in EMPTY_IN_ORIGINAL_IF_MERGED
                and merged(R, k)
            )
        ):
            if k not in T.files or R[k].shape != T[k].shape:
                return np.inf
            E = grid_of(R, k, R[k].size) if R[k].ndim == 1 else None
            d = deviations(R[k], T[k], FLOOR, E)
            if d:
                w = max(w, d[0])
    return max(w, IDENT)


def plot_scenario(sc, R, O, M, outdir):
    steps = checkpoints(R)
    first, last = steps[0], steps[-1]
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16.6, 7.2),
        sharex="col",
        layout="constrained",
        gridspec_kw=dict(height_ratios=[3, 2.4]),
    )
    for c, (getter, grid, label) in enumerate(SPECIES):
        ax, axd = axes[0, c], axes[1, c]
        # (the original's photon and e+/- grids differ by its constants)
        E, EO = R[f"{last:06d}/{grid}"], O[f"{last:06d}/{grid}"]
        ref = R[f"{last:06d}/{getter}"]
        ax.set_title(label)
        ax.set_xscale("log")
        axd.set_xlabel(r"$E$ [eV]")
        if not np.any(ref > 0):
            ax.text(
                0.5,
                0.5,
                "not simulated\nin this scenario",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            for a in (ax, axd):
                a.set(xticks=[], yticks=[])
            continue
        for step, alpha in ((first, 0.45), (last, 1.0)):
            ax.loglog(
                E,
                E * R[f"{step:06d}/{getter}"],
                color="k",
                lw=5,
                alpha=0.3 * alpha + 0.05,
                label=f"C++ patched, step {step}",
            )
            ax.loglog(
                EO,
                EO * O[f"{step:06d}/{getter}"],
                color=C_ORIGINAL,
                lw=1.6,
                alpha=alpha,
                label=f"C++ original, step {step}",
            )
            ax.loglog(
                E,
                E * M[f"{step:06d}/{getter}"],
                color=C_MOJO,
                lw=2.2,
                ls="--",
                alpha=alpha,
                label=f"am3_mojo, step {step}",
            )
        pk = np.max(E * ref)
        ax.set_ylim(pk * FLOOR, pk * 10)  # the range the reference tests check
        ax.grid(alpha=0.25)
        mojo = M[f"{last:06d}/{getter}"]
        axd.semilogy(
            E,
            np.minimum(rel(ref, mojo, E), OFF_SCALE),
            ls="none",
            marker=".",
            ms=5,
            color=C_MOJO,
            label=VS_PATCHED,
        )
        axd.semilogy(
            EO,
            np.minimum(rel(O[f"{last:06d}/{getter}"], mojo, EO), OFF_SCALE),
            ls="none",
            marker="x",
            ms=3.5,
            mew=1.0,
            color=C_ORIGINAL,
            label=VS_ORIGINAL,
        )
        mark_limits(axd)
        axd.set_ylim(IDENT / 5, OFF_SCALE * 5)
        axd.set_yticks(10.0 ** np.arange(-18, 3, 4))
        axd.grid(alpha=0.25)
        if c == 0:
            ax.set_ylabel(r"$E^2\,dN/dE\,dV$" + "\n" + r"[eV cm$^{-3}$]")
            axd.set_ylabel(f"deviation\nat step {last}")
    # the same lines in every panel: one legend below them, by column the lines
    # of each code, then the deviations
    c = next(c for c in range(4) if axes[0, c].get_legend_handles_labels()[0])
    (h, l), (hd, ld) = (
        axes[0, c].get_legend_handles_labels(),
        axes[1, c].get_legend_handles_labels(),
    )
    by_code = (0, 3, 1, 4, 2, 5)
    fig.legend(
        [h[i] for i in by_code] + hd,
        [l[i] for i in by_code] + ld,
        loc="outside lower center",
        ncol=5,
        fontsize=12,
    )
    # (inches: some room above the legend)
    fig.get_layout_engine().set(h_pad=0.1)
    fig.suptitle(
        f"{sc}: am3_mojo vs C++ AM3, patched (am3_fixes.patch) and original, "
        f"at step {last}\nworst deviation {DEVIATION} over all fluxes, with "
        f"$r$ = am3_mojo / C++: {shown(worst_flux(R, M, last))} from patched, "
        f"{shown(worst_flux(O, M, last, original=True))} from original"
    )
    path = os.path.join(outdir, f"spectra_{sc}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    outdir = published_dir("plots")
    written = []
    fig, axes = plt.subplots(
        1,
        len(SCENARIOS),
        figsize=(4.15 * len(SCENARIOS), 4.3),
        sharey=True,
        layout="constrained",
    )
    for ax, sc in zip(axes, SCENARIOS):
        R, O, M = (
            cached_scenario("cpp-fixed", sc),
            cached_scenario("cpp", sc),
            cached_scenario("mojo-am3", sc),
        )
        written.append(plot_scenario(sc, R, O, M, outdir))
        steps = checkpoints(R)
        ax.semilogy(
            steps,
            [min(worst_flux(R, M, s), OFF_SCALE) for s in steps],
            "o-",
            color=C_MOJO,
            label=VS_PATCHED,
        )
        ax.semilogy(
            steps,
            [min(worst_flux(O, M, s, original=True), OFF_SCALE) for s in steps],
            "x-",
            color=C_ORIGINAL,
            label=VS_ORIGINAL,
        )
        mark_limits(ax)
        ax.set_title(sc)
        ax.set_xlabel("time step")
        ax.set_ylim(1e-9, OFF_SCALE * 5)
        ax.grid(alpha=0.25)
    axes[0].set_yticks(10.0 ** np.arange(-8, 3, 2))
    axes[0].set_ylabel("worst deviation\nover all fluxes")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        fontsize=12,
    )
    # (inches: some room above the legend)
    fig.get_layout_engine().set(h_pad=0.1)
    fig.suptitle(
        f"am3_mojo vs C++ AM3, patched and original, at every checkpoint: "
        f"{DEVIATION} with $r$ = am3_mojo / C++"
    )
    path = os.path.join(outdir, "deviation_vs_time.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    written.append(path)
    print("\n".join(written))


if __name__ == "__main__":
    main()
