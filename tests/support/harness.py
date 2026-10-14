"""The Mojo harnesses of tests/harness/: small Mojo programs that print what the code computes (kernels and tables).
They are built against src/ with the project's Mojo (at the assertion level of `make test-asserts` when it is set)
and run by the unit tests and the scripts of tests/literature_checks and tests/solver_checks. This module also
builds the Mojo halves of the reference harnesses (tests/reference/harness/)."""
import os
import subprocess
import tempfile

import numpy as np

from support.paths import MOJO_ASSERT, ROOT, TESTS

MOJO = os.path.join(ROOT, '.venv', 'bin', 'mojo')
HARNESSES = os.path.join(TESTS, 'harness')


def build(source, out_dir):
    """Builds a harness (its name in tests/harness/, or the path of a .mojo file) into out_dir; returns the executable."""
    path = source if source.endswith('.mojo') else os.path.join(HARNESSES, source + '.mojo')
    exe = os.path.join(str(out_dir), os.path.basename(path)[:-len('.mojo')])
    subprocess.run([MOJO, 'build', '-D', 'ASSERT=' + MOJO_ASSERT, '-I', os.path.join(ROOT, 'src'), path, '-o', exe],
                   check=True, capture_output=True)
    return exe


def output(exe, *args, text=False):
    """What a built harness prints with these arguments: its numbers, or with text=True its lines."""
    out = subprocess.run([exe, *map(str, args)], check=True, capture_output=True, text=True).stdout
    return out.splitlines() if text else np.array(out.split(), dtype=float)


def run(name, out_dir, *args, text=False):
    """Builds the harness `name` into out_dir and returns what it prints with these arguments (output)."""
    return output(build(name, out_dir), *args, text=text)


def neutron_decay_kernels(out_dir):
    """The neutron decay kernels of the default grid (neutron_decay_mojo): the grids [GeV] Eh, El, En, the kernels
    Ke[j, i], Knu[j, i] (products at grid point j per decay of a neutron of hadron bin i) and the proton shares ps0, ps1."""
    v = run('neutron_decay_mojo', out_dir)
    nh, nl, nn = (int(x) for x in v[:3])
    parts, o = {}, 3
    for name, n in [('Eh', nh), ('El', nl), ('En', nn), ('Ke', nl * nh), ('Knu', nn * nh), ('ps0', nh), ('ps1', nh)]:
        parts[name] = v[o:o + n]
        o += n
    assert o == len(v)
    parts['Ke'] = parts['Ke'].reshape(nh, nl).T
    parts['Knu'] = parts['Knu'].reshape(nh, nn).T
    return parts


def kaon_decay_tables(out_dir, dlne, kmax):
    """The kaon decay tables T[kaon, product, k] (kaon_decay_mojo) on the grid x_k = e^(-k dlne), k = 0 .. kmax."""
    return run('kaon_decay_mojo', out_dir, repr(dlne), kmax).reshape(4, 13, kmax + 1)


class Triplet:
    """am3_mojo's triplet pair production (triplet_mojo), built once into out_dir (a temporary directory if None).
    It holds the table nodes k, sig (sigma / (alpha r_e^2)) and dist[n]: dN/dln y of the positron, the lower-energy
    electron and the recoil, shape (3, nb), on the ln y bins ly0 + (b + 1/2) dly."""

    def __init__(self, out_dir=None):
        if out_dir is None:
            self._tmp = tempfile.TemporaryDirectory()
            out_dir = self._tmp.name
        self.exe = build('triplet_mojo', out_dir)
        v = self._run('tables')
        self.ly0, self.dly, n = v[0], v[1], int(v[2])
        self.k, self.sig, self.dist, o = np.zeros(n), np.zeros(n), [], 3
        for i in range(n):
            self.k[i], self.sig[i], nb = v[o], v[o + 1], int(v[o + 2])
            self.dist.append(v[o + 3:o + 3 + 3 * nb].reshape(3, nb))
            o += 3 + 3 * nb
        assert o == len(v)

    def _run(self, mode, *values):
        return output(self.exe, mode, *(repr(float(x)) for x in values))

    def sigma(self, ks):
        """sigma(k) / (alpha r_e^2) at the photon energies ks [m_e c^2] (rest frame of the target lepton)."""
        return self._run('sigma', *ks)

    @staticmethod
    def _per_k(v, n):
        out, o = [], 0
        for _ in range(n):
            nb = int(v[o])
            out.append(v[o + 1:o + 1 + 3 * nb].reshape(3, nb))
            o += 1 + 3 * nb
        assert o == len(v)
        return out

    def distributions(self, ks):
        """[the code's dN/dln y of the three products at each k, shape (3, nb)], normalised to one per interaction."""
        return self._per_k(self._run('dist', *ks), len(ks))

    def extrapolated(self, k_lo, k_hi, ks):
        """distributions(ks) of the tables cut to the nodes k <= k_hi, with the asymptotic laws fitted to the nodes from
        k_lo on (as the code fits them from 1e6 on)."""
        return self._per_k(self._run('extrap', k_lo, k_hi, *ks), len(ks))

    def rows(self):
        """The kernel rows of the default grid, P = gamma eps = exp((m0 + q) dl): a dict of P, R (the rate, sigma_T),
        L (the energy-loss rate R <x+ + x_low>) and W (the deposits of the positron, W[0][q], and of the lower-energy
        electron, W[1][q], at the offsets o = 0, 1, ... below the target bin, x = e^(-o dl)), m0 and dl."""
        v = self._run('rows')
        m0, dl, nq = int(v[0]), v[1], int(v[2])
        P, R, L, W, o = np.zeros(nq), np.zeros(nq), np.zeros(nq), ([], []), 3
        for q in range(nq):
            P[q], R[q], L[q], nd = v[o], v[o + 1], v[o + 2], int(v[o + 3])
            o += 4
            for r in range(2):
                W[r].append(v[o:o + nd])
                o += nd
        assert o == len(v)
        return dict(P=P, R=R, L=L, W=W, m0=m0, dl=dl)
