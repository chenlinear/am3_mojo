"""Run implementations in subprocesses (one implementation per process) and
cache their outputs under results/."""
import json, os, subprocess, sys
import numpy as np

from support.paths import SUPPORT, results_dir

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


def cached_scenario(impl, scenario, rerun=False):
    """Full scenario run, cached in results/runs/<impl>_<scenario>.npz."""
    out = os.path.join(results_dir('runs'), f'{impl}_{scenario}.npz')
    if rerun or not os.path.exists(out):
        run_scenario(impl, scenario, out)
    return np.load(out)


def run_reuse(impl, setup, mode, rerun=False):
    """tests/support/reuse_runner.py output, cached in results/reuse/."""
    out = os.path.join(results_dir('reuse'), f'{impl}_{setup}_{mode}.npz')
    if rerun or not os.path.exists(out):
        _run('reuse_runner.py', ['--impl', impl, '--setup', setup, '--mode', mode, '--out', out])
    return np.load(out)


def time_scenario(impl, scenario, repeats=1, env=None):
    """{'init': [s...], 'evolve': [s...], 'steps': n} from a fresh process."""
    out = _run('time_scenario.py', ['--impl', impl, '--scenario', scenario, '--repeats', str(repeats)],
               env, capture=True)
    return json.loads(out.strip().splitlines()[-1])
