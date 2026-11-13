"""Benchmark am3_mojo against the C++ AM3. The C++ has the fixes of
tests/reference/am3_fixes.patch (`cpp-fixed`). Each (implementation, scenario)
pair runs in a fresh process.

The processes read Python's bytecode from a cache of their own, which an untimed
run per implementation fills first. So no process compiles Python sources, and
the peak memory depends neither on the bytecode caches of the environment nor on
PYTHONDONTWRITEBYTECODE.

usage: python tests/benchmark/benchmark.py
Writes results/benchmark/benchmark.json and benchmark.md. Without the C++
reference, it prints the timings of am3_mojo and leaves these files, which
hold the comparison, as they are.
"""

import json, os, sys, tempfile

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
from support.paths import available, published_dir
from support.runs import time_scenario
from support.scenarios import SCENARIOS

RUNS = [
    ("C++ AM3 (serial)", "cpp-fixed", {}),
    ("am3_mojo, 4 threads", "mojo-am3", {"AM3_MOJO_THREADS": "4"}),
    ("am3_mojo, 1 thread", "mojo-am3", {"AM3_MOJO_THREADS": "1"}),
    (
        "am3_mojo + added processes, 4 threads",
        "mojo",
        {"AM3_MOJO_THREADS": "4"},
    ),
]
REPEATS = 3
WARM_UP = "ssc_external"  # the shortest scenario


def main():
    out = published_dir("benchmark")  # (before the runs: stops a variant)
    runs = [r for r in RUNS if available(r[1])]
    if not any(r[1] == "cpp-fixed" for r in runs):
        print(
            "C++ reference not built (make reference): benchmarking "
            "am3_mojo only"
        )
    cache = tempfile.TemporaryDirectory()  # removed when main() returns
    bytecode = {
        "PYTHONDONTWRITEBYTECODE": "",
        "PYTHONPYCACHEPREFIX": cache.name,
    }
    for impl in dict.fromkeys(r[1] for r in runs):
        time_scenario(impl, WARM_UP, 1, bytecode)
    res = {}
    for sc in SCENARIOS:
        for label, impl, env in runs:
            d = time_scenario(impl, sc, REPEATS, {**env, **bytecode})
            x = res.setdefault(sc, {})[label] = dict(
                init_s=min(d["init"]),
                ms_per_step=1e3 * min(d["evolve"]) / d["steps"],
                steps=d["steps"],
                max_rss_gb=d["max_rss"] / 1e9,
            )
            print(
                f"{sc:18s} {label:38s} init {x['init_s']:7.3f} s   "
                f"{x['ms_per_step']:8.3f} ms/step   peak memory "
                f"{x['max_rss_gb']:.2f} GB",
                flush=True,
            )
    if len(runs) < len(RUNS):
        print(f"\n{out}/benchmark.md (the comparison with C++) not written")
        return
    json.dump(res, open(os.path.join(out, "benchmark.json"), "w"), indent=1)
    labels = [r[0] for r in runs]
    with open(os.path.join(out, "benchmark.md"), "w") as f:
        f.write(
            "| Scenario | "
            + " | ".join(labels)
            + " | Speed-up (4 threads vs C++) |\n"
        )
        f.write("|---" * (len(labels) + 2) + "|\n")
        for sc, r in res.items():
            cells = [
                (
                    "{ms_per_step:.2f} ms ({init_s:.2f} s, {max_rss_gb:.2f} GB)"
                ).format(**r[l])
                for l in labels
            ]
            sp = (
                r["C++ AM3 (serial)"]["ms_per_step"]
                / r["am3_mojo, 4 threads"]["ms_per_step"]
            )
            f.write(f"| {sc} | " + " | ".join(cells) + f" | {sp:.1f}x |\n")
        f.write(
            "\nms per evolution step (set-up: init_kernels, set_mag_field "
            "before the steps and the first step; peak memory of one "
            f"simulation); minimum of {REPEATS} runs in one fresh process per "
            "implementation and scenario.\n"
        )
    print(f"\nwritten {out}/benchmark.md")


if __name__ == "__main__":
    main()
