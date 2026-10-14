"""ExtraCooling physics module (port of the AM3 C++ class)."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays


struct ExtraCooling(Copyable, Movable):
    var verbose: Bool
    var AExtraCoolE: Vec
    var AExtraCoolG: Vec
    var AExtraCoolPRO: Vec
    var AExtraCoolNTR: Vec
    var AExtraCoolNU: Vec
    var AExtraCoolPipm: Vec
    var AExtraCoolMu: Vec
    var AExtraCoolK: Vec  # kaons (am3_mojo addition)

    def __init__(out self, dat: AM3Arrays, verbose: Bool):
        self.verbose = verbose
        self.AExtraCoolE = Vec()
        self.AExtraCoolG = Vec()
        self.AExtraCoolPRO = Vec()
        self.AExtraCoolNTR = Vec()
        self.AExtraCoolNU = Vec()
        self.AExtraCoolPipm = Vec()
        self.AExtraCoolMu = Vec()
        self.AExtraCoolK = Vec()
        self.clear_internal_arrays(dat)

    def clear_internal_arrays(mut self, dat: AM3Arrays):
        resize_zero(self.AExtraCoolE, dat.nbins_lep)
        resize_zero(self.AExtraCoolG, dat.nbins_pho)
        resize_zero(self.AExtraCoolPRO, dat.nbins_had)
        resize_zero(self.AExtraCoolNTR, dat.nbins_had)
        resize_zero(self.AExtraCoolPipm, dat.nbins_had)
        resize_zero(self.AExtraCoolMu, dat.nbins_had)
        resize_zero(self.AExtraCoolK, dat.nbins_had)
        resize_zero(self.AExtraCoolNU, dat.nbins_nu)
