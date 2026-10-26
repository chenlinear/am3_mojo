"""After init_kernels, every getter of am3_mojo returns the same value as the patched C++ AM3 (cpp-fixed:
am3_fixes.patch, which also gives it am3_mojo's PDG 2026 constants), and the same calls succeed or fail.

Known differences (tests/reference/README.md): get_t_neutrino_escape and get_t_neutrino_extra_cooling cover the whole
neutrino grid (AM3: the hadron-grid length), and errors are raised as `Exception`, not `TypeError`/`RuntimeError`.
"""
import numpy as np
import pytest

from support.compare import SHORTER_IN_AM3
from support.runs import api_defaults


@pytest.fixture(scope='module')
def both(tmp_path_factory):
    d = tmp_path_factory.mktemp('api')
    return api_defaults('cpp-fixed', d / 'cpp.json'), api_defaults('mojo-am3', d / 'mojo-am3.json')


def test_getter_defaults_identical(both):
    cpp, mojo = both
    diff = []
    for k in [k for k in cpp if not k.startswith('__')]:
        a, b = np.asarray(cpp[k], dtype=float), np.asarray(mojo[k], dtype=float)
        if k in SHORTER_IN_AM3:
            b = b[:a.size]  # AM3 stops at the hadron-grid length
        if not np.array_equal(a, b):
            diff.append(k)
    assert diff == []


def test_same_calls_succeed_and_fail(both):
    cpp, mojo = both
    assert {k: v[0] for k, v in cpp['__calls__'].items()} == {k: v[0] for k, v in mojo['__calls__'].items()}
    assert cpp['__calls__']['eV_to_had_grid_index1'] == mojo['__calls__']['eV_to_had_grid_index1']
