"""Prints am3_mojo's kaon decay kernels T[s * N_PROD + p][k], k = 0 .. K, for
the grid step dlnE; arguments dlnE K (default 0.1 200, as
tests/unit/test_kaon_decays.py uses them)."""

from std.sys import argv

from am3_mojo.physics.decays import kaon_decay_tables


def main() raises:
    var args = argv()
    var dlnE = Float64(String(args[1])) if len(args) > 1 else 0.1
    var kmax = Int(String(args[2])) if len(args) > 2 else 200
    var T = kaon_decay_tables(dlnE, kmax)
    for v in T:
        for x in v:
            print(x)
