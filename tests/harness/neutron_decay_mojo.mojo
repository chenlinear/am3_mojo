"""Prints am3_mojo's neutron decay kernels on the default grid. It prints the
numbers of hadron, lepton and neutrino bins, and the three grids [GeV] (the
lepton grid converted from its unit, m_e c^2). Then it prints the kernels K_e
and K_nu (column-major: per hadron bin, the products at each grid point per
decay) and the proton shares ps0, ps1 (tests/unit/test_added_processes.py)."""

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.physics.decays import NeutronDecay
from am3_mojo.utils.constants import MASS_E


def main() raises:
    var dat = AM3Arrays()
    var nd = NeutronDecay(dat, False)
    nd.initialize_kernels(dat)
    print(dat.nbins_had, dat.nbins_lep, dat.nbins_nu)
    for v in dat.egrid_had:
        print(v)
    for v in dat.egrid_lep:
        print(MASS_E * v)
    for v in dat.egrid_nu:
        print(v)
    for v in nd.K_e.data:
        print(v)
    for v in nd.K_nu.data:
        print(v)
    for v in nd.ps0:
        print(v)
    for v in nd.ps1:
        print(v)
