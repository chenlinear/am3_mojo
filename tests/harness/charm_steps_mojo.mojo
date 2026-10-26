"""Prints am3_mojo's photo-hadronic charm production steps: per step a line
"step" emin emax sigma mtype, then per part a line "part" with, for D+, D-,
D0, anti-D0, Ds+, Ds-, Lambda_c+, anti-Lambda_c-, the energy fractions from
protons, those from neutrons, and the multiplicities per interaction from
protons and from neutrons (tests/unit/test_added_processes.py).
"""

from am3_mojo.physics.photohadron import charm_production_steps


def main():
    for st in charm_production_steps():
        print("step", st.emin, st.emax, st.sigma, st.mtype)
        for k in range(len(st.parts)):
            ref p = st.parts[k]
            var line = String("part")
            for s in range(8):
                line += String(" ", p.chi_p[s])
            for s in range(8):
                line += String(" ", p.chi_n[s])
            for s in range(8):
                line += String(" ", p.mp[s])
            for s in range(8):
                line += String(" ", p.mn[s])
            print(line)
