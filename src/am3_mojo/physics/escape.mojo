"""Escape physics module (port of the AM3 C++ class)."""

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.runparams import RunParams
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *


struct Escape(Copyable, Movable):
    var verbose: Bool
    var afescE: Vec
    var afescG: Vec
    var afescPRO: Vec
    var afescNTR: Vec
    var afescNU: Vec
    var afescPipm: Vec
    var afescMu: Vec

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.afescE = Vec()
        self.afescG = Vec()
        self.afescPRO = Vec()
        self.afescNTR = Vec()
        self.afescNU = Vec()
        self.afescPipm = Vec()
        self.afescMu = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.afescE, dat.nbins_lep)
        resize_zero(self.afescG, dat.nbins_pho)
        resize_zero(self.afescPRO, dat.nbins_had)
        resize_zero(self.afescNTR, dat.nbins_had)
        resize_zero(self.afescPipm, dat.nbins_had)
        resize_zero(self.afescMu, dat.nbins_had)
        resize_zero(self.afescNU, dat.nbins_nu)

    def Calc_Escape_Lep(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing pair escape term")
        for i in range(dat.nbins_lep):
            self.afescE[i] = (
                T_SSC * rp.escape_fraction_pairs / rp.escape_timescale
            )

    def Calc_Escape_Phot(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing photon escape term")
        for i in range(dat.nbins_pho):
            self.afescG[i] = (
                T_SSC * rp.escape_fraction_photons / rp.escape_timescale
            )

    def Calc_Escape_Proton(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing proton escape term")
        for i in range(dat.nbins_had):
            self.afescPRO[i] = rp.escape_fraction_protons / rp.escape_timescale

    def Calc_Escape_Neutron(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing neutron escape term")
        for i in range(dat.nbins_had):
            self.afescNTR[i] = rp.escape_fraction_neutrons / rp.escape_timescale

    def Calc_Escape_Pipm(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing pion escape term")
        for i in range(dat.nbins_had):
            self.afescPipm[i] = rp.escape_fraction_pions / rp.escape_timescale

    def Calc_Escape_Mu(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing muon escape term")
        for i in range(dat.nbins_had):
            self.afescMu[i] = rp.escape_fraction_muons / rp.escape_timescale

    def Calc_Escape_Nu(mut self, rp: RunParams, dat: AM3Arrays):
        if self.verbose:
            print("computing neutrino escape term")
        for i in range(dat.nbins_nu):
            self.afescNU[i] = rp.escape_fraction_neutrinos / rp.escape_timescale
