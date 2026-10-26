"""The code's kaon tables, read from src/am3_mojo/physics/ for multibody_strangeness.py, m4_steps.py and
higher_types.py (which put tests/ on sys.path): the two-body steps and the multi-pion kaons (kaon_production_channels
in photohadron.mojo), and AM3's tables of the multi-pion types (photopion_tables.mojo)."""
import re
from pathlib import Path

from support.paths import ROOT

PHYSICS = Path(ROOT, 'src', 'am3_mojo', 'physics')


def _rows(name):
    """The rows of the table `var name: List[Array[Float64, 10]]` of kaon_production_channels (photohadron.mojo)."""
    src = (PHYSICS / 'photohadron.mojo').read_text()
    block = re.search(rf'var {name}: List\[Array\[Float64, 10\]\] = \[(.*?)\n    \]', src, re.S).group(1)
    return [[float(v) for v in row.split(',')] for row in re.findall(r'\[([^\[\]]+)\]', block)]


def two_body_steps():
    """Rows [eps_min, eps_max [GeV], chi of p -> K+, p -> K0, n -> K+, n -> K0, cross sections [mubarn] in the same
    order]."""
    return _rows('ky')


def code_mpi():
    """The multi-pion kaon rows [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+, h-, the low part's chi over that of the
    type's low-energy pions], type M(t + 1)."""
    return _rows('mpi')


def code_it3(name):
    """AM3's table `name` of the 14 entries of the types M1-M7: entry 2t is the low-energy part of type M(t + 1),
    2t + 1 the leading one."""
    text = (PHYSICS / 'photopion_tables.mojo').read_text()
    body = re.search(rf'comptime _{name}: Array\[Float64, 14\] = \[(.*?)\]', text, re.S).group(1)
    return [float(v) for v in body.replace('\n', ' ').split(',') if v.strip()]
