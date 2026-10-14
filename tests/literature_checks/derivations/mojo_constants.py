"""Reads the code's constants (src/am3_mojo/utils/constants.mojo). The derivation checks compare them with the README
and the PDG listings; the scripts take the code's inputs from here."""
import re

import sympy as sp

import latex as L

PATH = L.ROOT / "src" / "am3_mojo" / "utils" / "constants.mojo"
ARRAYS = L.ROOT / "src" / "am3_mojo" / "model" / "arrays.mojo"


def read(names, src=None):
    """{name: value} of these constants, exact: decimal literals as rationals, an expression such as 1.0 / 137.035999177
    as written. src: the file's text (read from the file by default)."""
    src = PATH.read_text() if src is None else src
    return {n: sp.sympify(re.search(n + r": Float64 = ([^\n#]+)", src).group(1).strip(), rational=True) for n in names}


def read_grid(names=("dlnE", "mec2_erg", "erg2eV")):
    """{name: value} of the grids' constants, exact, as model/arrays.mojo sets them (`self.<name> = <literal>`): the
    step dlnE of every grid in ln E, m_e c^2 (the unit of the lepton and photon grids) in erg, and erg in eV."""
    src = ARRAYS.read_text()
    return {n: sp.Rational(re.search(r"self\." + n + r" = ([^\s#]+)", src).group(1)) for n in names}
