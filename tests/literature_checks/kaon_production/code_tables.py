"""The code's tables for multibody_strangeness.py and m4_steps.py, read from am3_mojo's source (src/am3_mojo/physics/):
AM3's photo-pion interaction tables (photopion_tables.mojo), and the two-body kaon steps and the kaons of the
multi-pion types (photopion.mojo)."""
import os
import re
import sys

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.paths import ROOT  # noqa: E402

PHYSICS = os.path.join(ROOT, 'src', 'am3_mojo', 'physics')


def photopion_tables():
    """{'IT1Er': array, ..., 'IT3MpPim': array} from photopion_tables.mojo."""
    src = open(os.path.join(PHYSICS, 'photopion_tables.mojo')).read()
    T = {}
    for m in re.finditer(r'comptime _(IT\w+): Array\[Float64, \d+\] = \[(.*?)\]', src, re.S):
        T[m.group(1)] = np.array([float(v) for v in m.group(2).replace('\n', ' ').split(',') if v.strip()])
    return T


def multipion_kaons():
    """The kaons of the multi-pion types M(t + 1), t = 1 .. 6, from kaon_production_channels in photopion.mojo:
    rows [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+, h-, f] per photon-energy step [GeV]. Multiplicities per
    interaction from protons, leading fractions, and f, the energy fraction of the other kaons over that of the
    type's low-energy pions."""
    src = open(os.path.join(PHYSICS, 'photopion.mojo')).read()
    block = re.search(r'var mpi: List\[Array\[Float64, 10\]\] = \[(.*?)\n    \]', src, re.S).group(1)
    return [[float(v) for v in row.split(',')] for row in re.findall(r'\[([^\[\]]+)\]', block)]


def multi_body_threshold():
    """The photon energy [GeV] from which the kaons of the multi-pion types are made."""
    return min(row[1] for row in multipion_kaons())


def two_body_steps():
    """The two-body steps, from kaon_production_channels in photopion.mojo: rows [eps_min, eps_max [GeV], chi of
    p -> K+, p -> K0, n -> K+, n -> K0, cross sections [mubarn] in the same order]."""
    src = open(os.path.join(PHYSICS, 'photopion.mojo')).read()
    block = re.search(r'var ky: List\[Array\[Float64, 10\]\] = \[(.*?)\n    \]', src, re.S).group(1)
    return [[float(v) for v in row.split(',')] for row in re.findall(r'\[([^\[\]]+)\]', block)]
