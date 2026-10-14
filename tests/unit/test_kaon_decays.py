"""am3_mojo's kaon decay kernels (physics/decays.mojo, `kaon_decay_tables`): the products of the decays carry
the kaon's energy, and no entry is negative.
"""
import numpy as np
import pytest

from support import harness
from support.paths import load_derivation
from support.units import DLNE

derivation = load_derivation('kaon_decays')   # the README's grid points
KMAX = 200
KAONS = ['K+', 'K-', 'KS', 'KL']
X = derivation.numeric_grid()(np.arange(KMAX + 1), DLNE)  # grid points x_k


@pytest.fixture(scope='module')
def mojo_tables(tmp_path_factory):
    return harness.kaon_decay_tables(tmp_path_factory.mktemp('harness'), DLNE, KMAX)


@pytest.mark.parametrize('kaon', KAONS)
def test_energy_and_positivity(mojo_tables, kaon):
    T = mojo_tables[KAONS.index(kaon)] * DLNE  # products per decay at x_k
    assert np.sum(T * X) == pytest.approx(1.0, abs=1e-11)
    assert np.all(T >= 0.0)
