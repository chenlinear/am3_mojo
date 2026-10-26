"""Acc physics module (port of the AM3 C++ class)."""

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.runparams import RunParams
from am3_mojo.utils.constants import *


struct Acc(Copyable, Movable):
    var k_acc: Float64

    def __init__(out self):
        self.k_acc = E_CHARGE * C_LIGHT * GEV_PER_ERG

    def AAClepBohm(self, rp: RunParams, dat: AM3Arrays, i: Int) -> Float64:
        return (
            self.k_acc
            * rp.mag_field
            / (
                rp.electron_accel_factor_for_emax_estimate
                * dat.egrid_lep[i]
                * MASS_E
            )
            * T_SSC
        )

    def AAChadBohm(self, rp: RunParams, dat: AM3Arrays, i: Int) -> Float64:
        return (
            self.k_acc
            * rp.mag_field
            / rp.proton_accel_factor_for_emax_estimate
            / dat.egrid_had[i]
        )
