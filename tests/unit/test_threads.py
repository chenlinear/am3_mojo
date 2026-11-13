"""Results do not depend on the number of worker threads.

In every parallel section of am3_mojo, exactly one worker computes each output
element, in a fixed order. So runs with AM3_MOJO_THREADS=1 and 4 must be bitwise
identical.
"""

import os, re, subprocess, sys

import numpy as np
import pytest

from support.paths import IMPLEMENTATIONS
from support.runs import run_scenario
from support.scenarios import SCENARIOS

MOJO = IMPLEMENTATIONS["mojo"][0]


# (every scenario: ssc_external alone has merged positrons, the maximum-energy
#  estimates and user escape and extra-cooling arrays)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_one_thread_equals_four(scenario, tmp_path):
    a, b = (
        run_scenario(
            "mojo",
            scenario,
            tmp_path / f"{scenario}_{nt}.npz",
            steps=100,
            every=50,
            env={"AM3_MOJO_THREADS": nt},
        )
        for nt in ("1", "4")
    )
    assert a.files == b.files
    assert [
        k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)
    ] == []


def made_with_threads(value):
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {MOJO!r}); import am3_mojo;"
            " am3_mojo.AM3()",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "AM3_MOJO_THREADS": value},
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_malformed_thread_count_is_reported():
    """An AM3_MOJO_THREADS that is not a whole number is reported when an AM3
    object is made, which then runs on the default number of threads. (It was
    ignored without a word: AM3_MOJO_THREADS=1.0 ran on every core.)"""
    out = made_with_threads("1.0")
    assert "AM3_MOJO_THREADS=1.0 is not a whole number" in out


@pytest.mark.parametrize("value", ["0", "-2", "1000"])
def test_thread_count_out_of_range_is_reported(value):
    """An AM3_MOJO_THREADS below 1 is reported, and the default number of
    threads is used. A value above the number of logical CPUs is reported and
    capped there. (Values below 1 ran serially without a word, and 1000 ran 1000
    threads: peak memory 149 -> 500 MB, steps 22 % slower.)"""
    out = made_with_threads(value)
    if value == "1000":
        m = re.search(
            "AM3_MOJO_THREADS=1000 is more than the ([0-9]+) logical CPUs;"
            " using ([0-9]+) threads",
            out,
        )
        assert m and m[1] == m[2] and int(m[2]) <= os.cpu_count(), out
    else:
        assert re.search(
            f"AM3_MOJO_THREADS={value} is below 1; using [1-9]", out
        )
