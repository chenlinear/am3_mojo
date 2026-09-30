"""Results do not depend on the number of worker threads.

Every parallel section of am3_mojo partitions the work so that each output
element is computed by exactly one worker in a fixed order, so a run with
AM3_MOJO_THREADS=1 and one with 4 threads must be bitwise identical.
"""
import numpy as np
import pytest

from support.runs import run_scenario


@pytest.mark.parametrize('scenario', ['blazar_simple', 'leptohadronic_all', 'exotic'])
def test_one_thread_equals_four(scenario, tmp_path):
    out = {}
    for nt in ('1', '4'):
        out[nt] = run_scenario('mojo', scenario, tmp_path / f'{scenario}_{nt}.npz', steps=100, every=50,
                               env={'AM3_MOJO_THREADS': nt})
    a, b = out['1'], out['4']
    assert a.files == b.files
    assert [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)] == []
