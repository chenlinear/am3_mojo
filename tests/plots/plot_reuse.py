"""Plot the difference between re-initializing the kernels for every simulation
and reusing them (am3_mojo).

usage: python tests/plots/plot_reuse.py

Uses the runs of tests/support/reuse_runner.py (its modes and CONFIGS) that the
unit and reference tests cache in results/reuse/, and runs the missing ones.
The 4th configuration repeats the 1st. am3_mojo runs fresh, reused and in the
two modes wrong on purpose (no_clear, no_escape_reset). mojo-am3 (only AM3's
processes, the C++ run's baseline) and the C++ AM3 (cpp-fixed, if built) run
reused. The differences are those of tests/support/compare.py, on the bins that
the reference tests check. The C++ run is the reference of mojo-am3 (as
tests/reference/test_reuse_vs_cpp.py compares them).

Figures in results/plots/:
  reuse_<setup>.png   per configuration, photon spectra with and without
                      re-initialization and their per-bin relative difference
  reuse_summary.png   worst difference in the four spectra per configuration,
                      and the time spent in init_kernels
"""

import os, sys

import numpy as np

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
from support.compare import FLOOR, compare, mismatches
from support.paths import available, published_dir
from support.reuse_runner import CONFIGS, N_STEPS, SETUPS
from support.runs import config, run_reuse
from deviation import IDENT, plt, rel

FIGS = published_dir("plots")
RUNS = [
    ("mojo", "fresh"),
    ("mojo", "reused"),
    ("mojo", "no_clear"),
    ("mojo", "no_escape_reset"),
    ("mojo-am3", "reused"),
    ("cpp", "reused"),
]
N_CONFIGS = len(CONFIGS)
SPECIES = [
    ("get_photons", "get_egrid_photons", r"$\gamma$", "#1f77b4"),
    ("get_pairs", "get_egrid_lep", r"$\mathrm{e}^{\pm}$", "#2ca02c"),
    ("get_protons", "get_egrid_had", r"$\mathrm{p}$", "#9467bd"),
    (
        "get_neutrinos",
        "get_egrid_neutrinos",
        r"$\nu + \bar{\nu}$",
        "#8c564b",
    ),
]
DIFFERENCE, RATIO = r"$\max(r, 1/r) - 1$", r"$r$ = x / re-initialised"
# where the summary draws larger differences, and infinite ones (particles in a
# bin where only one of the two runs has them)
OFF_SCALE = 1e30
C_CPP = "#7f7f7f"
WRONG = [
    ("mojo", "no_clear", "#d62728", "^:", "wrong: no clear_am3()"),
    ("mojo", "no_escape_reset", "#ff7f0e", "v:", "wrong: escape not set again"),
]


def ensure_runs():
    D = {}
    for s in SETUPS:
        for impl, mode in RUNS:
            real = "cpp-fixed" if impl == "cpp" else impl
            if available(real):
                D.setdefault(s, {})[impl, mode] = run_reuse(real, s, mode)
    return D


def worst(R, T, k, names=None, cpp=False):
    o = compare(config(R, k), config(T, k), FLOOR, cpp)

    def picked(n):
        return names is None or n.split()[0] in names

    if any(picked(n) for n in mismatches(o)):
        return np.inf
    return max(
        [IDENT] + [d for d, n in o["flux"] + o["timescale"] if picked(n)]
    )


def worsts(D, impl, mode, names=None):
    if impl == "cpp":
        return [
            worst(D[impl, mode], D["mojo-am3", "reused"], k, names, cpp=True)
            for k in range(N_CONFIGS)
        ]
    return [
        worst(D["mojo", "fresh"], D[impl, mode], k, names)
        for k in range(N_CONFIGS)
    ]


def label(c):
    return (
        f"B = {c['B']:g} G, t_esc = {c['t_esc']:g} R/c\n"
        f"t_exp = {c['t_exp']:g} R/c, n_pp = {c['n_pp']:g}\n"
        f"dt = {c['dt']:g} R/c"
    )


def plot_setup(setup, D):
    F, U, CP = D["mojo", "fresh"], D["mojo", "reused"], D.get(("cpp", "reused"))
    fig, axes = plt.subplots(
        2,
        N_CONFIGS,
        figsize=(17, 10),
        sharex=True,
        layout="constrained",
        gridspec_kw=dict(height_ratios=[3, 2.6]),
    )
    for k in range(N_CONFIGS):
        ax, axd = axes[0, k], axes[1, k]
        E = F[f"{k}/get_egrid_photons"]
        f, u = F[f"{k}/get_photons"], U[f"{k}/get_photons"]
        ax.loglog(
            E,
            E * f,
            color="k",
            lw=6,
            alpha=0.25,
            label="init_kernels() for this run",
        )
        ax.loglog(
            E,
            E * u,
            color="#1f77b4",
            lw=2.4,
            ls="--",
            label="kernels reused, clear_am3()",
        )
        for impl, mode, col, _, lab in WRONG if k > 0 else ():
            ax.loglog(
                E,
                E * D[impl, mode][f"{k}/get_photons"],
                color=col,
                lw=1.6,
                alpha=0.8,
                label=lab,
            )
        pk = np.max(E * f)
        ax.set_ylim(pk * 1e-12, pk * 30)
        title = f"run {k + 1}" + (
            " (repeat of run 1)" if k == N_CONFIGS - 1 else ""
        )
        ax.set_title(f"{title}\n{label(CONFIGS[k])}", fontsize=15)
        ax.grid(alpha=0.25)
        if k == 0:
            ax.set_ylabel(
                r"$E^2\,dN/dE\,dV$ of $\gamma$" + "\n" + r"[eV cm$^{-3}$]"
            )

        for getter, grid, name, col in SPECIES:
            Eg = F[f"{k}/{grid}"]
            d = rel(F[f"{k}/{getter}"], U[f"{k}/{getter}"], Eg)
            axd.semilogy(
                Eg,
                d,
                ls="none",
                marker="o",
                ms=3.5,
                color=col,
                label=f"reused vs re-init: {name}",
            )
        # (the C++ code has only AM3's processes: mojo-am3 against it)
        if CP is not None:
            axd.semilogy(
                E,
                rel(
                    CP[f"{k}/get_photons"],
                    D["mojo-am3", "reused"][f"{k}/get_photons"],
                    E,
                ),
                color=C_CPP,
                lw=1.6,
                label=r"for scale: C++ AM3 vs am3_mojo ($\gamma$)",
            )
        for impl, mode, col, _, lab in WRONG if k > 0 else ():
            axd.semilogy(
                E,
                rel(f, D[impl, mode][f"{k}/get_photons"], E),
                color=col,
                lw=1.6,
                label=lab + r" ($\gamma$)",
            )
        axd.axhline(IDENT, color="k", lw=1.2)
        axd.text(E[0], IDENT * 3, "bitwise identical", fontsize=12)
        axd.set_ylim(IDENT / 10, 1e4)
        axd.set_xlabel(r"$E$ [eV]")
        axd.grid(alpha=0.25)
        if k == 0:
            axd.set_ylabel(
                f"{DIFFERENCE}\n{RATIO}\n" + r"(bins > $10^{-12}$ of peak)"
            )
    # one legend for all panels, with the wrong runs once
    top, bottom = (
        axes[0, 1].get_legend_handles_labels(),
        axes[1, 1].get_legend_handles_labels(),
    )
    keep = [
        i for i, lab in enumerate(bottom[1]) if not lab.startswith("wrong:")
    ]
    fig.legend(
        top[0] + [bottom[0][i] for i in keep],
        top[1] + [bottom[1][i] for i in keep],
        loc="outside lower center",
        ncol=3,
        fontsize=15,
    )
    # (inches: some room above the legend)
    fig.get_layout_engine().set(h_pad=0.1)
    fig.suptitle(
        f"{setup} setup, am3_mojo: kernels initialised once and "
        "reused (clear_am3() between runs)\nvs init_kernels() before "
        f"every run — {N_STEPS} steps per run",
        fontsize=18,
    )
    out = os.path.join(FIGS, f"reuse_{setup}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary(D_all):
    fig, (ax, axt) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        layout="constrained",
        gridspec_kw=dict(width_ratios=[1.6, 1]),
    )
    runs = np.arange(1, N_CONFIGS + 1)
    styles = [
        ("mojo", "reused", "#1f77b4", "o-", "kernels reused (correct)"),
        (
            "cpp",
            "reused",
            C_CPP,
            "s--",
            "for scale: C++ AM3 reused vs am3_mojo",
        ),
    ] + WRONG
    for si, setup in enumerate(SETUPS):
        D = D_all[setup]
        for impl, mode, col, sty, lab in styles:
            if (impl, mode) not in D:
                continue
            w = np.minimum(
                worsts(D, impl, mode, [g for g, *_ in SPECIES]), OFF_SCALE
            )
            ax.semilogy(
                runs + (si - 0.5) * 0.08,
                w,
                sty,
                color=col,
                ms=8,
                mfc=col if si == 0 else "white",
                label=lab if si == 0 else "_nolegend_",
            )
    ax.axhline(IDENT, color="k", lw=1.2)
    ax.text(0.9, IDENT * 3, "bitwise identical", fontsize=12)
    ax.axhline(OFF_SCALE, color="k", lw=1.0, ls=":")
    ax.text(
        N_CONFIGS + 0.1,
        OFF_SCALE * 3,
        r"$10^{30}$ or more, or $\infty$",
        ha="right",
        fontsize=12,
    )
    ax.set_xticks(
        runs,
        [f"run {k}" + ("\n(= run 1)" if k == N_CONFIGS else "") for k in runs],
    )
    ax.set_ylim(IDENT / 10, 1e32)
    ax.set_ylabel(
        f"worst {DIFFERENCE}, {RATIO}\n"
        + r"in the $\gamma$, $\mathrm{e}^{\pm}$, "
        r"$\mathrm{p}$, $\nu$ spectra"
    )
    ax.set_title(
        "Difference from re-initialising the kernels before every run\n"
        "(filled: blazar setup, open: leptohadronic setup)",
        fontsize=15,
    )
    ax.grid(alpha=0.25)
    # (below the off-scale line, above the differences of order 1)
    ax.legend(
        fontsize=12,
        loc="center right",
        bbox_to_anchor=(1.0, 0.68),
        framealpha=0.5,
    )

    x = np.arange(len(SETUPS))
    ini_f = [float(D_all[s]["mojo", "fresh"]["__init_s"]) for s in SETUPS]
    ini_r = [float(D_all[s]["mojo", "reused"]["__init_s"]) for s in SETUPS]
    ev = [float(D_all[s]["mojo", "reused"]["__evolve_s"]) for s in SETUPS]
    axt.bar(
        x - 0.2,
        ini_f,
        0.4,
        color="0.6",
        label=f"init_kernels() before\neach of {N_CONFIGS} runs",
    )
    axt.bar(
        x + 0.2,
        ini_r,
        0.4,
        color="#1f77b4",
        label="init_kernels() once,\nreused",
    )
    for xi, (a, b) in enumerate(zip(ini_f, ini_r)):
        axt.text(
            xi - 0.2, a, f"{a:.1f} s", ha="center", va="bottom", fontsize=12
        )
        axt.text(
            xi + 0.2, b, f"{b:.1f} s", ha="center", va="bottom", fontsize=12
        )
    axt.set_xticks(
        x, [f"{s}\n(evolving:\n{e:.1f} s)" for s, e in zip(SETUPS, ev)]
    )
    axt.set_ylim(0, max(ini_f) * 1.5)
    axt.set_ylabel("time in init_kernels() [s]")
    axt.set_title(
        f"Kernel set-up time for {N_CONFIGS} runs\nof {N_STEPS} "
        "steps (am3_mojo)",
        fontsize=15,
    )
    axt.legend(fontsize=12, loc="upper center")
    out = os.path.join(FIGS, "reuse_summary.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    D_all = ensure_runs()
    written = [plot_setup(s, D_all[s]) for s in SETUPS]
    written.append(plot_summary(D_all))
    print(
        "worst relative difference from re-initialising "
        "(all getters; am3_mojo without\nthe added processes against the C++ "
        "AM3, which has only AM3's):"
    )
    for s in SETUPS:
        for i, m in RUNS[1:]:
            if (i, m) in D_all[s] and i != "mojo-am3":
                print(
                    f"  {s:14s} {i:4s} {m:16s} "
                    + "  ".join(
                        "identical" if x == IDENT else f"{x:9.1e}"
                        for x in worsts(D_all[s], i, m)
                    )
                )
    print("figures:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
