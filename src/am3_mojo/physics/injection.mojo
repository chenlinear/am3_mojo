"""Port of the AM3 C++ class Injection."""

from std.utils.numerics import isinf
from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.runparams import RunParams
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *


struct Injection(Copyable, Movable):
    var verbose: Bool
    var inj_proton: Vec
    var inj_neutron: Vec
    var inj_piplus: Vec
    var inj_piminus: Vec
    var inj_pi0: Vec
    var inj_muminusleft: Vec
    var inj_muminusright: Vec
    var inj_muplusleft: Vec
    var inj_muplusright: Vec
    var inj_nue: Vec
    var inj_antinue: Vec
    var inj_numu: Vec
    var inj_antinumu: Vec
    var inj_electron: Vec
    var inj_positron: Vec
    var inj_photon: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.inj_proton = Vec()
        self.inj_neutron = Vec()
        self.inj_piplus = Vec()
        self.inj_piminus = Vec()
        self.inj_pi0 = Vec()
        self.inj_muminusleft = Vec()
        self.inj_muminusright = Vec()
        self.inj_muplusleft = Vec()
        self.inj_muplusright = Vec()
        self.inj_nue = Vec()
        self.inj_antinue = Vec()
        self.inj_numu = Vec()
        self.inj_antinumu = Vec()
        self.inj_electron = Vec()
        self.inj_positron = Vec()
        self.inj_photon = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.inj_proton, dat.nbins_had)
        resize_zero(self.inj_neutron, dat.nbins_had)
        resize_zero(self.inj_piplus, dat.nbins_had)
        resize_zero(self.inj_piminus, dat.nbins_had)
        resize_zero(self.inj_pi0, dat.nbins_had)
        resize_zero(self.inj_muminusleft, dat.nbins_had)
        resize_zero(self.inj_muplusleft, dat.nbins_had)
        resize_zero(self.inj_muminusright, dat.nbins_had)
        resize_zero(self.inj_muplusright, dat.nbins_had)
        resize_zero(self.inj_nue, dat.nbins_nu)
        resize_zero(self.inj_antinue, dat.nbins_nu)
        resize_zero(self.inj_numu, dat.nbins_nu)
        resize_zero(self.inj_antinumu, dat.nbins_nu)
        resize_zero(self.inj_electron, dat.nbins_lep)
        resize_zero(self.inj_positron, dat.nbins_lep)
        resize_zero(self.inj_photon, dat.nbins_pho)

    def Calc_IN_Electron_powerlaw(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing electron injection spectrum")
        _broken_powerlaw(
            self.inj_electron,
            dat.egrid_lep,
            dat.nbins_lep,
            dat.dlnE,
            dat.eV_to_lep_grid_index(rp.powerlaw_injection_electrons_emin),
            dat.eV_to_lep_grid_index(rp.powerlaw_injection_electrons_ebreak),
            rp.powerlaw_injection_electrons_ebreak / MEC2_EV,
            rp.powerlaw_injection_electrons_emax / MEC2_EV,
            rp.powerlaw_injection_electrons_index,
            rp.powerlaw_injection_electrons_index_above_break,
            rp.powerlaw_injection_electrons_cutoff_steepness,
            rp.powerlaw_injection_electrons_luminosity
            / rp.powerlaw_injection_electrons_volume
            * GEV_PER_ERG
            / MASS_E
            / RATE_SSC,
        )

    def Calc_IN_Proton_powerlaw(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing proton injection spectrum")
        _broken_powerlaw(
            self.inj_proton,
            dat.egrid_had,
            dat.nbins_had,
            dat.dlnE,
            dat.eV_to_had_grid_index(rp.powerlaw_injection_protons_emin),
            dat.eV_to_had_grid_index(rp.powerlaw_injection_protons_ebreak),
            rp.powerlaw_injection_protons_ebreak / 1e9,
            rp.powerlaw_injection_protons_emax / 1e9,
            rp.powerlaw_injection_protons_index,
            rp.powerlaw_injection_protons_index_above_break,
            rp.powerlaw_injection_protons_cutoff_steepness,
            rp.powerlaw_injection_protons_luminosity
            / rp.powerlaw_injection_protons_volume
            * GEV_PER_ERG,
        )


def _broken_powerlaw(
    mut q: Vec,
    egrid: Vec,
    n: Int,
    dlnE: Float64,
    imin: Int,
    imid: Int,
    e_mid: Float64,
    e_max: Float64,
    index: Float64,
    index_above_break: Float64,
    cutoff_steepness: Float64,
    L_density: Float64,
):
    """q per unit ln E on the energies of a grid (in its units). The shape is
    taken from its logarithm, relative to its largest bin. So neither the shape
    nor the scale factor leaves the range of the numbers. (If the spectrum
    misses the grid, there is no injection. AM3 divided by the zero integral,
    which made every bin NaN. AM3 took the shape as it is and scaled it by
    L_density / integral. For a hard or steep spectrum, the shape overflowed
    and made bins infinite or NaN (index -25 on the default lepton grid). Or
    the factor underflowed to 0 and left no injection (index -20 without a
    cutoff). AM3 bug.)"""
    var i0 = max(imin, 1)
    var lmax = 0.0
    var found = False
    for i in range(n):
        if i < i0:
            q[i] = 0.0
            continue
        var p = (index if i <= imid else index_above_break) - 1.0
        q[i] = -p * c_log(egrid[i] / e_mid) - c_pow(
            egrid[i] / e_max, cutoff_steepness
        )
        if not found or q[i] > lmax:
            lmax = q[i]
            found = True
    var integral = 0.0
    if found and not (isinf(lmax) and lmax < 0.0):
        for i in range(i0, n):
            # (1 in the largest bin, also where its logarithm is infinite)
            q[i] = 1.0 if q[i] == lmax else c_exp(q[i] - lmax)
            integral += dlnE * egrid[i] * q[i]
    if integral > 0.0:
        for i in range(n):
            q[i] *= L_density / integral
    else:
        for i in range(n):
            q[i] = 0.0
