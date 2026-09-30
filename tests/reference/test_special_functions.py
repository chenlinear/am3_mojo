"""Element-wise comparison of special functions with the original code:
the IC cooling kernel function Fic (copied verbatim from AM3's Solver.cc into
harness/fic_cpp.cc), GSL's gsl_sf_bessel_Knu for nu = 1/3, 2/3, 4/3, 5/3 and
AM3's int_bessel_k53 (math.cc), against their am3_mojo ports. The harnesses
are compiled with the reference's GSL (tests/reference/setup_reference.sh).
"""
import os, subprocess
import numpy as np
import pytest

from support.paths import ROOT, REFERENCE_WORK

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'harness')
PREFIX = os.path.join(REFERENCE_WORK, 'deps', 'install')
AM3_SRC = os.path.join(REFERENCE_WORK, 'am3')
MOJO = os.path.join(ROOT, '.venv', 'bin', 'mojo')


def build_and_run(tmp_path):
    cc = ['c++', '-O3', '-DNDEBUG', '-std=c++17']
    subprocess.run(cc + [os.path.join(HARNESS, 'fic_cpp.cc'), '-o', str(tmp_path / 'fic_cpp')], check=True)
    subprocess.run(cc + [f'-I{PREFIX}/include', f'-I{AM3_SRC}/include', os.path.join(HARNESS, 'specfun_cpp.cc'),
                         os.path.join(AM3_SRC, 'src', 'math.cc'), f'{PREFIX}/lib/libgsl.a',
                         f'{PREFIX}/lib/libgslcblas.a', '-o', str(tmp_path / 'specfun_cpp')], check=True)
    for name in ('fic_mojo', 'specfun_mojo'):
        subprocess.run([MOJO, 'build', '-D', 'ASSERT=none', '--fp-mode', 'contract=off', '-I',
                        os.path.join(ROOT, 'src'), os.path.join(HARNESS, name + '.mojo'), '-o',
                        str(tmp_path / name)], check=True, capture_output=True)
    out = {}
    for name in ('fic', 'specfun'):
        subprocess.run([str(tmp_path / f'{name}_cpp'), str(tmp_path / f'{name}_cpp.bin')], check=True)
        txt = subprocess.run([str(tmp_path / f'{name}_mojo')], check=True, capture_output=True, text=True).stdout
        out[name] = (np.fromfile(tmp_path / f'{name}_cpp.bin'), np.array(txt.split(), dtype=float))
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
    assert rel(c, m) < 1e-14


@pytest.mark.parametrize('k, name', list(enumerate(['K_1/3', 'K_2/3', 'K_4/3', 'K_5/3', 'int_bessel_k53'])))
def test_bessel(values, k, name):
    c, m = values['specfun']
    c, m = c.reshape(5, -1)[k], m.reshape(5, -1)[k]
    assert rel(c, m) < 1e-13, name
