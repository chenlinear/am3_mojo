"""Prints am3_mojo's kaon production channels (the two-body steps gamma N -> K Y
and the multi-pion types' kaons). Each channel takes one line: emin emax,
chi_p[K+, K-, K0_S, K0_L], chi_n[...], mp[...], mn[...], sigma, mtype (-1: a
two-body step). Then it prints AM3's 14 multi-pion entries, one line each:
"pi", sigma, chi (tests/unit/test_added_processes.py)."""

from am3_mojo.physics.photohadron import kaon_production_channels
from am3_mojo.physics.photopion_tables import IT3sig, IT3kai


def main():
    for ch in kaon_production_channels():
        var line = String(ch.emin, " ", ch.emax)
        for s in range(4):
            line += String(" ", ch.chi_p[s])
        for s in range(4):
            line += String(" ", ch.chi_n[s])
        for s in range(4):
            line += String(" ", ch.mp[s])
        for s in range(4):
            line += String(" ", ch.mn[s])
        line += String(" ", ch.sigma, " ", ch.mtype)
        print(line)
    for it in range(14):
        print("pi", IT3sig(it), IT3kai(it))
