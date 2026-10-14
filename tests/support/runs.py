"""Run implementations in subprocesses (one implementation per process),
cache their outputs under results/, and read them."""
import json, os, subprocess, sys
import numpy as np

from support.paths import CACHE_SUFFIX, SUPPORT, module_mtime, results_dir

PY = sys.executable


def _run(script, args, env=None, capture=False):
    e = dict(os.environ)
    e.update(env or {})
    r = subprocess.run([PY, os.path.join(SUPPORT, script)] + args, env=e, text=True,
                       stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f'{script} {" ".join(args)} failed:\n{r.stderr[-3000:]}')
    return r.stdout


def run_scenario(impl, scenario, out, steps=None, every=None, env=None):
    """Checkpoints of every getter -> `out` (.npz); returns the loaded file."""
    args = ['--impl', impl, '--scenario', scenario, '--out', str(out)]
    if steps:
        args += ['--steps', str(steps), '--every', str(every or steps)]
    _run('run_scenario.py', args, env)
    return np.load(out)


def _stale(out, impl):
    """True if the cached output is missing or older than the module that made it
    (a rebuild invalidates the caches)."""
    return not os.path.exists(out) or os.path.getmtime(out) < module_mtime(impl)


def cached_scenario(impl, scenario, rerun=False):
    """Full scenario run, cached in results/runs/<impl>_<scenario>.npz."""
    out = os.path.join(results_dir('runs' + CACHE_SUFFIX), f'{impl}_{scenario}.npz')
    if rerun or _stale(out, impl):
        run_scenario(impl, scenario, out)
    return np.load(out)


def run_reuse(impl, setup, mode, rerun=False):
    """tests/support/reuse_runner.py output, cached in results/reuse/."""
    out = os.path.join(results_dir('reuse' + CACHE_SUFFIX), f'{impl}_{setup}_{mode}.npz')
    if rerun or _stale(out, impl):
        _run('reuse_runner.py', ['--impl', impl, '--setup', setup, '--mode', mode, '--out', out])
    return np.load(out)


def time_scenario(impl, scenario, repeats=1, env=None):
    """{'init': [s...], 'evolve': [s...], 'steps': n, 'max_rss': bytes} from a fresh process."""
    out = _run('time_scenario.py', ['--impl', impl, '--scenario', scenario, '--repeats', str(repeats)],
               env, capture=True)
    return json.loads(out.strip().splitlines()[-1])


def profile_scenario(impl, scenario, steps, env=None):
    """{'labels': [...], 'ms': [...]}: AM3's profiler slots, ms per step (tests/support/profile_scenario.py)."""
    out = _run('profile_scenario.py', ['--impl', impl, '--scenario', scenario, '--steps', str(steps)],
               env, capture=True)
    return json.loads(out.strip().splitlines()[-1])


def api_defaults(impl, out):
    """The state after init_kernels and the outcome of a few calls (tests/support/api_defaults.py), via `out`."""
    _run('api_defaults.py', ['--impl', impl, '--out', str(out)])
    with open(out) as f:
        return json.load(f)


def checkpoints(npz):
    """The step numbers of the checkpoints of a scenario run."""
    return sorted({int(k.split('/')[0]) for k in npz.files})


class Getters(dict):
    """Arrays by getter, with the `files` of an npz (what compare() takes)."""
    files = property(lambda self: list(self))


def config(npz, k):
    """The getters of configuration k of a reuse run."""
    pre = f'{k}/'
    return Getters({n[len(pre):]: npz[n] for n in npz.files if n.startswith(pre)})
