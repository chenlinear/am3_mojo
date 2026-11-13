"""Prints am3_mojo's kaon or charm decay kernels T[s * N_PROD + p][k] (charm:
N_CPROD), k = 0 .. K, for the grid step dlnE. Arguments: kaon|charm dlnE K
(tests/unit/test_added_processes.py: kaon 0.1 200, charm 0.1 400). With the
argument mu, it prints a line for each mu+ of the semileptonic modes of D+,
D0, Ds+ and Lambda_c+, except the two-body ones. The line holds the species
(CDP, ...), 0 for a mode taken as phase space or 1 for a three-body one, its
branching ratio and the number of nodes of its parent-frame energy distribution
(_nodes). Then it prints one line per node: its energy E [GeV], number w and
polarization weight G."""

from std.sys import argv

from am3_mojo.model.particles import CDP, CD0, CDSP, CLC
from am3_mojo.physics.decays import (
    kaon_decay_tables,
    charm_mass,
    X_MUP,
    XK_TWO,
    XK_PS,
    XRules,
    _nodes,
    charm_decay_modes,
    charm_decay_tables,
)
from am3_mojo.utils.buffers import Vec


def main() raises:
    var args = argv()
    if args[1] == "mu":
        var gl = XRules()
        var species: List[Int] = [CDP, CD0, CDSP, CLC]
        for s in species:
            var modes = charm_decay_modes(s)
            for d in range(len(modes)):
                ref md = modes[d]
                if md.kind == XK_TWO:
                    continue
                for i in range(len(md.prod)):
                    if md.prod[i] != X_MUP:
                        continue
                    var nd = _nodes(md, charm_mass(s), i, gl)
                    print(s, 0 if md.kind == XK_PS else 1, md.br, len(nd.E))
                    for k in range(len(nd.E)):
                        print(nd.E[k], nd.w[k], nd.G[k])
        return
    var dlnE = Float64(String(args[2]))
    var kmax = Int(String(args[3]))
    var T: List[Vec]
    if args[1] == "kaon":
        T = kaon_decay_tables(dlnE, kmax)
    else:
        T = charm_decay_tables(dlnE, kmax)
    for v in T:
        for x in v:
            print(x)
