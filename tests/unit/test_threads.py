"""Results do not depend on the number of worker threads.

In every parallel section of am3_mojo, exactly one worker computes each output element, in a fixed order, so runs
with AM3_MOJO_THREADS=1 and 4 must be bitwise identical.
"""
import numpy as np
import pytest

from support.runs import run_scenario


@pytest.mark.parametrize('scenario', ['blazar_simple', 'leptohadronic_all', 'exotic'])
def test_one_thread_equals_four(scenario, tmp_path):
    a, b = (run_scenario('mojo', scenario, tmp_path / f'{scenario}_{nt}.npz', steps=100, every=50,
                         env={'AM3_MOJO_THREADS': nt}) for nt in ('1', '4'))
    assert a.files == b.files
    assert [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)] == []
