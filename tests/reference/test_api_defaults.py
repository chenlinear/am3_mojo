"""After init_kernels, every getter of am3_mojo returns the same value as the
original C++ AM3, and the same calls succeed or fail.

Known differences (see README): get_t_neutrino_escape and
get_t_neutrino_extra_cooling cover the whole neutrino grid (the original
returns only the hadron-grid length), and errors are raised as `Exception`
rather than `TypeError`/`RuntimeError`.
"""
import json, os, subprocess, sys
import numpy as np
import pytest

from support.paths import SUPPORT

NEUTRINO_GRID_GETTERS = {'get_t_neutrino_escape', 'get_t_neutrino_extra_cooling'}


def defaults(impl, tmp_path):
    out = tmp_path / f'{impl}.json'
    subprocess.run([sys.executable, os.path.join(SUPPORT, 'api_defaults.py'), '--impl', impl, '--out', str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out))


@pytest.fixture(scope='module')
def both(tmp_path_factory):
    d = tmp_path_factory.mktemp('api')
    return defaults('cpp', d), defaults('mojo', d)


def test_getter_defaults_identical(both):
    cpp, mojo = both
    getters = [k for k in cpp if not k.startswith('__')]
    assert len(getters) == 233
    diff = []
    for k in getters:
        a, b = np.asarray(cpp[k], dtype=float), np.asarray(mojo[k], dtype=float)
        if k in NEUTRINO_GRID_GETTERS:
            a, b = a, b[:a.size]  # the original stops at the hadron-grid length
        if not np.array_equal(a, b):
            diff.append(k)
    assert diff == []


def test_same_calls_succeed_and_fail(both):
    cpp, mojo = both
    assert {k: v[0] for k, v in cpp['__calls__'].items()} == {k: v[0] for k, v in mojo['__calls__'].items()}
    assert cpp['__calls__']['eV_to_had_grid_index1'] == mojo['__calls__']['eV_to_had_grid_index1']
