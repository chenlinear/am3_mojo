"""Prints am3_mojo's kaon or charm decay kernels T[s * N_PROD + p][k] (charm:
N_CPROD), k = 0 .. K, for the grid step dlnE; arguments kaon|charm dlnE K
(tests/unit/test_added_processes.py: kaon 0.1 200, charm 0.1 400)."""

from std.sys import argv

from am3_mojo.physics.decays import kaon_decay_tables, charm_decay_tables
from am3_mojo.utils.buffers import Vec


def main() raises:
    var args = argv()
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
