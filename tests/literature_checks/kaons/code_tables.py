"""This module reads the code's kaon tables from src/am3_mojo/physics/ for
multibody_strangeness.py, m4_steps.py and higher_types.py (which put tests/ on
sys.path)."""

import re
from pathlib import Path

from support.paths import ROOT

PHYSICS = Path(ROOT, "src", "am3_mojo", "physics")


def _rows(name):
    src = (PHYSICS / "photohadron.mojo").read_text()
    block = re.search(
        rf"var {name}: List\[Array\[Float64, 10\]\] = "
        r"\[(.*?)\n    \]",
        src,
        re.S,
    ).group(1)
    return [
        [float(v) for v in row.split(",")]
        for row in re.findall(r"\[([^\[\]]+)\]", block)
    ]


def two_body_steps():
    """Rows [eps_min, eps_max [GeV], chi of p -> K+, p -> K0, n -> K+, n -> K0,
    cross sections [mubarn] in the same order]."""
    return _rows("ky")


def code_mpi():
    """The multi-pion kaon rows [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+,
    h-, the low part's chi over that of the type's low-energy pions], type
    M(t + 1)."""
    return _rows("mpi")


def code_it3(name):
    """AM3's table `name` of the types M1-M7. Entry 2t is the low-energy part of
    type M(t + 1), and entry 2t + 1 is its leading part."""
    text = (PHYSICS / "photopion_tables.mojo").read_text()
    body = re.search(
        rf"comptime _{name}: Array\[Float64, 14\] = \[(.*?)\]", text, re.S
    ).group(1)
    return [float(v) for v in body.replace("\n", " ").split(",") if v.strip()]
