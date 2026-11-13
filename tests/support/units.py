"""The CODATA values in which the scenarios and test sources are written, and
the code's own constants, read from src/."""

import re
from pathlib import Path

import sympy as sp

from support.codata import C_CGS, ME_C2_EV, ME_G, MP_C2_EV, MP_G, SIGMA_T
from support.paths import ROOT, load_derivation

_code = load_derivation("mojo_constants")

_n, _t = (float(v) for v in _code.read(("N_SSC", "T_SSC")).values())
# sigma_T c [cm^3/s], the inverse of the code's units of density times time
SIGMA_T_C = 1.0 / (_n * _t)

# the code's masses [GeV] and mean lives [s], PDG 2026 (exact decimals, so that
# 0.13957039e9 eV, for example, stays that float)
MASSES = _code.read(
    (
        "MASS_E",
        "MASS_P",
        "MASS_N",
        "MASS_PI0",
        "MASS_PI",
        "MASS_MU",
        "MASS_K",
        "MASS_K0",
        "MASS_D",
        "MASS_D0",
        "MASS_DS",
        "MASS_LC",
    )
)
TAUS = _code.read(
    (
        "TAU_K",
        "TAU_KS",
        "TAU_KL",
        "TAU_N",
        "TAU_D",
        "TAU_D0",
        "TAU_DS",
        "TAU_LC",
    )
)
MASS_GEV = {n: float(v) for n, v in MASSES.items()}
MASS_EV = {n: float(v * 10**9) for n, v in MASSES.items()}
TAU_S = {n: float(v) for n, v in TAUS.items()}
# m_e c^2: the unit of the lepton and photon grids
MEC2_EV = float(_code.read(("MEC2_EV",))["MEC2_EV"])


def _grid_step():
    """Exact, as model/arrays.mojo sets it (`self.dlnE = <literal>`)."""
    src = Path(ROOT, "src", "am3_mojo", "model", "arrays.mojo").read_text()
    return sp.Rational(re.search(r"self\.dlnE = ([^\s#]+)", src).group(1))


DLNE = float(_grid_step())  # the step of every grid in ln E
