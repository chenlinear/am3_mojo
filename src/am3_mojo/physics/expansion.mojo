"""Expansion physics module (port of the AM3 C++ class)."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *
from am3_mojo.model.runparams import RunParams
from am3_mojo.model.arrays import AM3Arrays


struct Expansion(Copyable, Movable):
    var verbose: Bool

    def __init__(out self, verbose: Bool):
        self.verbose = verbose

    @staticmethod
    def get_t_adiabatic(rp: RunParams) -> Float64:
        return 3.0 * rp.expansion_timescale

    @staticmethod
    def afExpLep(rp: RunParams) -> Float64:
        return 1.0 / (rp.expansion_timescale / T_SSC_CGS)

    @staticmethod
    def afExpHad(rp: RunParams) -> Float64:
        return 1.0 / rp.expansion_timescale

    @staticmethod
    def AAdiLep(rp: RunParams) -> Float64:
        return -1.0 / (3.0 * rp.expansion_timescale / T_SSC_CGS)

    @staticmethod
    def AAdiHad(rp: RunParams) -> Float64:
        return -1.0 / (3.0 * rp.expansion_timescale)
