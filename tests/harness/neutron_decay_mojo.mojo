"""Prints am3_mojo's neutron decay kernels on the default grid: the numbers of
hadron, lepton and neutrino bins, the three grids [GeV] (the lepton grid
converted from AM3's unit, 0.511 MeV), the kernels K_e and K_nu (column-major:
for every hadron bin, the products at every grid point per decay) and the
proton shares ps0, ps1 (tests/unit/test_added_processes.py)."""

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.physics.decays import NeutronDecay
from am3_mojo.utils.constants import masse


def main() raises:
    var dat = AM3Arrays()
    var nd = NeutronDecay(dat, False)
    nd.initialize_kernels(dat)
    print(dat.nbins_had, dat.nbins_lep, dat.nbins_nu)
    for v in dat.egrid_had:
        print(v)
    for v in dat.egrid_lep:
        print(masse * v)
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
