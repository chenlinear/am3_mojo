"""Injection physics module (port of the AM3 C++ class)."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays


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
        var volume = rp.powerlaw_injection_electrons_volume
        var imin = dat.eV_to_lep_grid_index(
            rp.powerlaw_injection_electrons_emin
        )
        var imid = dat.eV_to_lep_grid_index(
            rp.powerlaw_injection_electrons_ebreak
        )
        var pl_idx = rp.powerlaw_injection_electrons_index
        var ph_idx = rp.powerlaw_injection_electrons_index_above_break
        var exp_steep = rp.powerlaw_injection_electrons_cutoff_steepness
        var lum = rp.powerlaw_injection_electrons_luminosity
        var pidxm1_low = pl_idx - 1.0
        var pidxm1_high = ph_idx - 1.0
        var integral = 0.0
        var mec2_eV = 0.510998e6
        var Gamma_mid = rp.powerlaw_injection_electrons_ebreak / mec2_eV
        var Gamma_max = rp.powerlaw_injection_electrons_emax / mec2_eV
        for i in range(dat.nbins_lep):
            var log_cutoff = c_pow(dat.egrid_lep[i] / Gamma_max, exp_steep)
            if i < max(imin, 1):
                self.inj_electron[i] = 0.0
            elif i <= imid:
                self.inj_electron[i] = c_pow(
                    dat.egrid_lep[i] / Gamma_mid, -pidxm1_low
                ) * c_exp(-log_cutoff)
            else:
                self.inj_electron[i] = c_pow(
                    dat.egrid_lep[i] / Gamma_mid, -pidxm1_high
                ) * c_exp(-log_cutoff)
            integral += dat.dlnE * dat.egrid_lep[i] * self.inj_electron[i]
        var L_density = lum / volume * erg_GeV / masse / Rate_SSC_CGS
        for i in range(dat.nbins_lep):
            self.inj_electron[i] *= L_density / integral

    def Calc_IN_Proton_powerlaw(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing proton injection spectrum")
        var volume = rp.powerlaw_injection_protons_volume
        var imin = dat.eV_to_had_grid_index(rp.powerlaw_injection_protons_emin)
        var imid = dat.eV_to_had_grid_index(
            rp.powerlaw_injection_protons_ebreak
        )
        var pl_idx = rp.powerlaw_injection_protons_index
        var ph_idx = rp.powerlaw_injection_protons_index_above_break
        var exp_steep = rp.powerlaw_injection_protons_cutoff_steepness
        var lum = rp.powerlaw_injection_protons_luminosity
        var pidxm1_low = pl_idx - 1.0
        var pidxm1_high = ph_idx - 1.0
        var integral = 0.0
        var Emid_GeV = rp.powerlaw_injection_protons_ebreak / 1e9
        var Emax_GeV = rp.powerlaw_injection_protons_emax / 1e9
        for i in range(dat.nbins_had):
            var log_cutoff = c_pow(dat.egrid_had[i] / Emax_GeV, exp_steep)
            if i < max(imin, 1):
                self.inj_proton[i] = 0.0
            elif i <= imid:
                self.inj_proton[i] = c_pow(
                    dat.egrid_had[i] / Emid_GeV, -pidxm1_low
                ) * c_exp(-log_cutoff)
            else:
                self.inj_proton[i] = c_pow(
                    dat.egrid_had[i] / Emid_GeV, -pidxm1_high
                ) * c_exp(-log_cutoff)
            integral += dat.dlnE * dat.egrid_had[i] * self.inj_proton[i]
        var L_density = lum / volume * erg_GeV
        for i in range(dat.nbins_had):
            self.inj_proton[i] *= L_density / integral
