"""Element-wise comparison of C++ special functions with their am3_mojo ports:
AM3's IC cooling kernel function Fic (Solver.cc, copied verbatim into
harness/fic_cpp.cc with the series and dilogarithm fixes of am3_fixes.patch),
GSL's gsl_sf_bessel_Knu for nu = 1/3, 2/3, 4/3, 5/3, and AM3's int_bessel_k53
with the fix of am3_fixes.patch (math.cc of work/src-fixed).

Tolerances: 1e-13 for the Bessel functions and int_bessel_k53 (the same
algorithms in both codes). 1e-6 for Fic, which cancels heavily right below the
photon energy, where it changes sign (and is cut to zero). There, the C++ build,
in which clang fuses some multiply-adds, differs in the last digits.
"""

import os, subprocess

import numpy as np
import pytest

from support import harness
from support.paths import REFERENCE_WORK

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness")
PREFIX = os.path.join(REFERENCE_WORK, "deps", "install")
FIXED = os.path.join(REFERENCE_WORK, "src-fixed")


@pytest.fixture(scope="module")
def values(tmp_path_factory):
    d = tmp_path_factory.mktemp("harness")
    cc = ["c++", "-O3", "-DNDEBUG", "-std=c++17"]
    subprocess.run(
        cc + [os.path.join(HARNESS, "fic_cpp.cc"), "-o", str(d / "fic_cpp")],
        check=True,
    )
    subprocess.run(
        cc
        + [
            f"-I{PREFIX}/include",
            f"-I{FIXED}/include",
            os.path.join(HARNESS, "specfun_cpp.cc"),
            os.path.join(FIXED, "src", "math.cc"),
            f"{PREFIX}/lib/libgsl.a",
            f"{PREFIX}/lib/libgslcblas.a",
            "-o",
            str(d / "specfun_cpp"),
        ],
        check=True,
    )
    out = {}
    for name in ("fic", "specfun"):
        subprocess.run(
            [str(d / f"{name}_cpp"), str(d / f"{name}_cpp.bin")], check=True
        )
        mojo = harness.build(os.path.join(HARNESS, f"{name}_mojo.mojo"), d)
        out[name] = (np.fromfile(d / f"{name}_cpp.bin"), harness.output(mojo))
    return out


def rel(c, m):
    nz = c != 0
    assert np.array_equal(c == 0, m == 0)
    return np.max(np.abs(m[nz] / c[nz] - 1))


def test_fic(values):
    assert rel(*values["fic"]) < 1e-6


@pytest.mark.parametrize(
    "k, name",
    list(enumerate(["K_1/3", "K_2/3", "K_4/3", "K_5/3", "int_bessel_k53"])),
)
def test_bessel(values, k, name):
    c, m = (v.reshape(5, -1)[k] for v in values["specfun"])
    assert rel(c, m) < 1e-13, name
