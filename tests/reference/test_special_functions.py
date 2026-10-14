"""Element-wise comparison of special functions with the C++ code: the IC
cooling kernel function Fic (copied verbatim from AM3's Solver.cc into
harness/fic_cpp.cc, with the series fix of am3_fixes.patch), GSL's
gsl_sf_bessel_Knu for nu = 1/3, 2/3, 4/3, 5/3 and AM3's int_bessel_k53
(math.cc), against their am3_mojo ports. The harnesses are compiled with the
reference's GSL (tests/reference/setup_reference.sh).

Tolerances: 1e-13 for the Bessel functions, the same algorithm in both codes.
1e-6 for Fic, which cancels heavily just below the photon energy, where it
changes sign (and is cut to zero): there the C++ build, which fuses
multiply-adds where clang chooses to, differs in the last digits. 1e-2 for
int_bessel_k53, a 100-step sum in am3_mojo: because of rounding, the C++ loop
`while (y < yf)` takes a 101st step for about half of all x, which changes the
result by up to 0.4 % (tests/reference/README.md, "Numerics").
"""
import os, subprocess
import numpy as np
import pytest

from support import harness
from support.paths import REFERENCE_WORK

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness')
PREFIX = os.path.join(REFERENCE_WORK, 'deps', 'install')
AM3_SRC = os.path.join(REFERENCE_WORK, 'am3')


def build_and_run(tmp_path):
    cc = ['c++', '-O3', '-DNDEBUG', '-std=c++17']
    subprocess.run(cc + [os.path.join(HARNESS, 'fic_cpp.cc'), '-o', str(tmp_path / 'fic_cpp')], check=True)
    subprocess.run(cc + [f'-I{PREFIX}/include', f'-I{AM3_SRC}/include', os.path.join(HARNESS, 'specfun_cpp.cc'),
                         os.path.join(AM3_SRC, 'src', 'math.cc'), f'{PREFIX}/lib/libgsl.a',
                         f'{PREFIX}/lib/libgslcblas.a', '-o', str(tmp_path / 'specfun_cpp')], check=True)
    out = {}
    for name in ('fic', 'specfun'):
        subprocess.run([str(tmp_path / f'{name}_cpp'), str(tmp_path / f'{name}_cpp.bin')], check=True)
        mojo = harness.build(os.path.join(HARNESS, f'{name}_mojo.mojo'), tmp_path)
        out[name] = (np.fromfile(tmp_path / f'{name}_cpp.bin'), harness.output(mojo))
    return out


@pytest.fixture(scope='module')
def values(tmp_path_factory):
    return build_and_run(tmp_path_factory.mktemp('harness'))


def rel(c, m):
    nz = c != 0
    assert np.array_equal(c == 0, m == 0)
    return np.max(np.abs(m[nz] / c[nz] - 1))


def test_fic(values):
    c, m = values['fic']
    assert rel(c, m) < 1e-6


@pytest.mark.parametrize('k, name', list(enumerate(['K_1/3', 'K_2/3', 'K_4/3', 'K_5/3', 'int_bessel_k53'])))
def test_bessel(values, k, name):
    c, m = values['specfun']
    c, m = c.reshape(5, -1)[k], m.reshape(5, -1)[k]
    assert rel(c, m) < (1e-2 if name == 'int_bessel_k53' else 1e-13), name
