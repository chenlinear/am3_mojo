"""The inverse-Compton emission of the e+- in a photon field that spans more
than 38 orders of magnitude.

AM3 divides the photons by their sum and multiplies them with the emission
kernel in single precision (the Eigen float types of InverseCompton.cc). The
normal numbers of single precision end at 1.2e-38, and its smallest number is
1.4e-45. am3_mojo takes double precision. The photon field also holds a line,
above the targets that the emission takes (511 keV). Without the electrons'
synchrotron radiation, gamma-gamma pair production and synchrotron
self-absorption, the photons from 1 keV to 10 MeV are those that the electrons
scatter.
"""

import json, os, subprocess, sys

import numpy as np
import pytest

from support.paths import TESTS
from support.setups import energy, line, new, power_law

OFF = (
    "hadronic",
    "electron_syn",
    "annihilation",
    "ssa",
    "escape",
    "expansion",
    "adiabatic_cooling",
)
# densities of the photon line [cm^-3 per ln E]: the targets are 1e-37, 1e-40,
# 1e-43 and 1e-46 of the photon sum
LINES = [1e27, 1e30, 1e33, 1e36]


def scattered(impl):
    out = []
    for density in LINES:
        a = new(impl, OFF, dt=1.0)
        El, Eg = a.get_egrid_lep(), a.get_egrid_photons()
        a.set_current_densities_photons(
            power_law(1e-10, 1e-3, 1.0, s=1.0)(Eg) + line(density, 3e7)(Eg)
        )
        a.set_current_densities_electrons(line(1.0, 1e9)(El))
        a.evolve_step()
        band = (Eg > 1e3) & (Eg < 1e7)
        out.append(energy(a.get_photons()[band], Eg[band]))
    return out


@pytest.fixture(scope="module")
def runs():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for impl in ("mojo-am3", "cpp-fixed"):
        code = (
            f"import json, sys; sys.path[:0] = [{TESTS!r}, {here!r}]\n"
            "from test_single_precision import scattered\n"
            f"print(json.dumps(scattered({impl!r})))"
        )
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-3000:]
        out[impl] = np.array(json.loads(r.stdout.splitlines()[-1]))
    return out


def test_am3_mojo_scatters_the_same_photons_beside_any_line(runs):
    """In am3_mojo, the scattered photons do not depend on the line: the same
    energy to 1e-12 for all four densities."""
    np.testing.assert_allclose(
        runs["mojo-am3"], runs["mojo-am3"][0], rtol=1e-12
    )


def test_am3_loses_targets_below_single_precision(runs):
    """In the C++, targets at 1e-37 of the photon sum give the scattered energy
    of am3_mojo to 1e-6. Targets at 1e-40 and 1e-43, below the normal numbers
    of single precision, give 5e-6 and 0.5 % less. Targets at 1e-46, below the
    smallest number of single precision, give none."""
    r = runs["cpp-fixed"] / runs["mojo-am3"]
    assert r[0] == pytest.approx(1.0, abs=1e-6)
    assert 1 - r[1] == pytest.approx(5.4e-6, rel=0.1)
    assert 1 - r[2] == pytest.approx(5.1e-3, rel=0.1)
    assert r[3] == 0.0
