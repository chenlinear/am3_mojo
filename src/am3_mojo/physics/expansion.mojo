"""Port of the AM3 C++ class Expansion."""

from am3_mojo.model.runparams import RunParams
from am3_mojo.utils.constants import *


struct Expansion:
    @staticmethod
    def afExpLep(rp: RunParams) -> Float64:
        return 1.0 / (rp.expansion_timescale / T_SSC)

    @staticmethod
    def afExpHad(rp: RunParams) -> Float64:
        return 1.0 / rp.expansion_timescale

    @staticmethod
    def AAdiLep(rp: RunParams) -> Float64:
        return -1.0 / (3.0 * rp.expansion_timescale / T_SSC)

    @staticmethod
    def AAdiHad(rp: RunParams) -> Float64:
        return -1.0 / (3.0 * rp.expansion_timescale)
