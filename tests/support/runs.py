import hashlib, json, os, subprocess, sys

import numpy as np

from support.paths import (
    CACHE_SUFFIX,
    LABELS,
    SUPPORT,
    built_from_another_patch,
    module_fingerprint,
    module_mtime,
    results_dir,
)

PY = sys.executable


def _current(impl):
    """The results of a patched C++ built from another am3_fixes.patch are not
    those of the current fixes."""
    if impl == "cpp-fixed" and built_from_another_patch():
        raise RuntimeError(
            f"{LABELS[impl]} built from another patch: run `make reference`"
        )


def _run(script, env=None, capture=False, **opts):
    _current(opts.get("impl"))
    args = [
        x for k, v in opts.items() if v is not None for x in ("--" + k, str(v))
    ]
    r = subprocess.run(
        [PY, os.path.join(SUPPORT, script)] + args,
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"{script} {' '.join(args)} failed:\n{r.stderr[-3000:]}"
        )
    return json.loads(r.stdout.strip().splitlines()[-1]) if capture else None


def save_npz(out, **arrays):
    """A reader never sees a part of the file, and a run that is stopped, or two
    runs at once, leave a whole file. (If the file is written in place, a
    stopped run leaves a part of the file, which _stale takes for current.)"""
    tmp = f"{out}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def run_scenario(impl, scenario, out, steps=None, every=None, env=None):
    _run(
        "run_scenario.py",
        env,
        impl=impl,
        scenario=scenario,
        out=out,
        steps=steps,
        every=every,
    )
    return np.load(out)


# the files that define the cached runs (scenarios, set-ups, their units)
DEFINITIONS = [
    os.path.join(SUPPORT, f)
    for f in (
        "scenarios.py",
        "run_scenario.py",
        "reuse_runner.py",
        "codata.py",
        "paths.py",
    )
]


def _made_by(impl):
    """The fingerprint and the digest are taken before a run. If a module is
    rebuilt or a definition is edited (or restored with an older time) during
    the run, the output is stale."""
    h = hashlib.sha256()
    for f in DEFINITIONS:
        with open(f, "rb") as fh:
            h.update(fh.read())
    return module_fingerprint(impl) + "\ndefinitions " + h.hexdigest()


def _record(out, made_by):
    with open(out + ".module", "w") as f:
        f.write(made_by)


def _stale(out, impl):
    newest = max(
        [module_mtime(impl)] + [os.path.getmtime(f) for f in DEFINITIONS]
    )
    if not os.path.exists(out) or os.path.getmtime(out) < newest:
        return True
    if not os.path.exists(out + ".module"):
        return True
    with open(out + ".module") as f:
        return f.read() != _made_by(impl)


def cached_scenario(impl, scenario, rerun=False):
    out = os.path.join(
        results_dir("runs" + CACHE_SUFFIX), f"{impl}_{scenario}.npz"
    )
    _current(impl)
    if rerun or _stale(out, impl):
        made_by = _made_by(impl)
        run_scenario(impl, scenario, out)
        _record(out, made_by)
    return np.load(out)


def run_reuse(impl, setup, mode, rerun=False):
    out = os.path.join(
        results_dir("reuse" + CACHE_SUFFIX), f"{impl}_{setup}_{mode}.npz"
    )
    _current(impl)
    if rerun or _stale(out, impl):
        made_by = _made_by(impl)
        _run("reuse_runner.py", impl=impl, setup=setup, mode=mode, out=out)
        _record(out, made_by)
    return np.load(out)


def time_scenario(impl, scenario, repeats, env):
    """{'init': [s...], 'evolve': [s...], 'steps': n, 'max_rss': bytes} from a
    fresh process."""
    return _run(
        "time_scenario.py",
        env,
        capture=True,
        impl=impl,
        scenario=scenario,
        repeats=repeats,
    )


def profile_scenario(impl, scenario, steps, env=None):
    """{'labels': [...], 'ms': [...]}: AM3's profiler slots, ms per step
    (tests/support/profile_scenario.py)."""
    return _run(
        "profile_scenario.py",
        env,
        capture=True,
        impl=impl,
        scenario=scenario,
        steps=steps,
    )


def api_defaults(impl, out):
    _run("api_defaults.py", impl=impl, out=out)
    with open(out) as f:
        return json.load(f)


def checkpoints(npz):
    return sorted({int(k.split("/")[0]) for k in npz.files})


class Getters(dict):
    """Arrays by getter, with the `files` of an npz (what compare() takes)."""

    files = property(lambda self: list(self))


def config(npz, k):
    pre = f"{k}/"
    return Getters(
        {n[len(pre) :]: npz[n] for n in npz.files if n.startswith(pre)}
    )
