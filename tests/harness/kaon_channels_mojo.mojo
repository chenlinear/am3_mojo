"""Prints am3_mojo's kaon production channels, one line per channel (the
two-body steps gamma N -> K Y and the kaons of the multi-pion types):
emin emax, chi_p[K+, K-, K0_S, K0_L], chi_n[...], mp[...], mn[...], sigma,
mtype (-1: a two-body step). Then the 14 multi-pion entries of AM3's table,
one line each: "pi", index, emin, emax, sigma, chi, and the multiplicities of
pi0, pi+, pi- from protons and from neutrons
(tests/unit/test_kaon_channels.py)."""

from am3_mojo.physics.photopion import kaon_production_channels
from am3_mojo.physics.photopion_tables import *


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
        print(
            "pi",
            it,
            IT3Em(it),
            IT3Ex(it),
            IT3sig(it),
            IT3kai(it),
            IT3MpPi0(it),
            IT3MpPip(it),
            IT3MpPim(it),
            IT3MnPi0(it),
            IT3MnPip(it),
            IT3MnPim(it),
        )
