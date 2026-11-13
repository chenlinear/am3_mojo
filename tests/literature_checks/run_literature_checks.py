"""Run the literature checks (`make literature_checks`): the derivation checks,
the inventory and every script of tests/literature_checks/, and
tools/ky_angular_extract.py. Each runs with the arguments that give the figures
quoted in its README.

usage: .venv/bin/python tests/literature_checks/run_literature_checks.py
       [-j N] [--out DIR] [name ...]
Each script's output goes to DIR/literature_checks/<name>.txt, and a tool's
output goes to DIR/tools/<name>.txt (DIR: results/ by default). The exit code
and run time of each are printed and written to
DIR/literature_checks/summary.txt. With -j 6 (the default), a run takes about
5 minutes on an Apple M5, the time of its longest script,
triplet/machinery.py --high-k.
Exit status 1 if a script failed.
"""

import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(
    1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)  # tests/
from support.paths import ROOT

L = "tests/literature_checks/"
# longest first, for the parallel runs
SCRIPTS = {
    "triplet_machinery": [L + "triplet/machinery.py", "--high-k"],
    "triplet_x_scaling": [L + "triplet/x_scaling.py"],
    "triplet_compare_exact": [
        L + "triplet/compare_exact.py",
        "--mc",
        "4.03",
        "7",
        "10",
        "12",
        "45",
    ],
    "triplet_below_grid": [L + "triplet/below_grid.py"],
    "derivations": [L + "common/run.py"],
    "triplet_inelasticity": [L + "triplet/inelasticity.py"],
    "triplet_extrapolation": [L + "triplet/extrapolation.py"],
    "kaon_production_two_body": [L + "kaons/two_body.py"],
    "kaon_production_multibody_strangeness": [
        L + "kaons/multibody_strangeness.py"
    ],
    "kaon_production_m4_steps": [L + "kaons/m4_steps.py"],
    "kaon_production_higher_types": [L + "kaons/higher_types.py"],
    "ky_angular_extract": ["tools/ky_angular_extract.py"],
    "derivations_inventory": [L + "common/inventory.py"],
    "triplet_steady_states": [L + "triplet/steady_states.py"],
}


def output(out, name, args):
    return os.path.join(
        out,
        "tools" if args[0].startswith("tools/") else "literature_checks",
        name + ".txt",
    )


def run(name, args, out):
    t0 = time.time()
    with open(output(out, name, args), "w") as f:
        f.write("$ " + " ".join(args) + "\n")
        f.flush()
        rc = subprocess.run(
            [sys.executable] + args,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
        ).returncode
    line = (
        f"{name:40s} {'ok' if rc == 0 else f'FAILED (exit {rc})':16s} "
        f"{time.time() - t0:7.1f} s"
    )
    print(line, flush=True)
    return line, rc


def workers(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not 1 or more")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-j", type=workers, default=6, help="scripts run in parallel"
    )
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "results"),
        help="the folder of literature_checks/ and tools/",
    )
    ap.add_argument("names", nargs="*", help="run only these (default: all)")
    a = ap.parse_args()
    unknown = [name for name in a.names if name not in SCRIPTS]
    if unknown:
        ap.error(f"unknown script(s): {', '.join(unknown)}")
    for folder in ("literature_checks", "tools"):
        os.makedirs(os.path.join(a.out, folder), exist_ok=True)
    todo = [
        (name, args)
        for name, args in SCRIPTS.items()
        if not a.names or name in a.names
    ]
    with ThreadPoolExecutor(a.j) as ex:
        res = list(ex.map(lambda t: run(*t, a.out), todo))
    lines = [line for line, _ in res]
    with open(
        os.path.join(a.out, "literature_checks", "summary.txt"), "w"
    ) as f:
        f.write("\n".join(lines) + "\n")
    failed = [line for line, rc in res if rc != 0]
    print(
        f"\n{len(res) - len(failed)} of {len(res)} scripts ran without "
        f"error; outputs in {os.path.join(a.out, 'literature_checks')}/ and "
        f"{os.path.join(a.out, 'tools')}/"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
