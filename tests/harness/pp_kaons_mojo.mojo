"""Prints am3_mojo's kaon production in pp collisions
(tests/unit/test_added_processes.py), by mode, for the K+, the K- and the
neutral kaons (one column each):
  fit E...       pp_kaon_mult_fit, the multiplicity fit below E_t, at each
                 proton energy E [GeV];
  ct             c_t, the fit over Koers et al.'s kaons per collision at
                 E_t = KPW_E_MIN (pp_kaon_moments);
  fade c E...    pp_kaon_fade(c, E) at each E [GeV] (one value per line);
  kernel         per proton energy E_j of the default hadron grid a line with
                 E_j [GeV], then per kaon the kaons per collision on the grid
                 from the kernels (from E_t on the sum of dlnE K[i, j] over i,
                 below it the delta-functional term; divided by c sigma_inel),
                 and from E_t on the same sum of Koers et al.'s F_K without
                 the factor (0 below E_t).
"""

from std.sys import argv

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.model.particles import KP, KM, KS
from am3_mojo.physics.proton_proton import (
    KPW_E_MIN,
    pp_kaon_F,
    pp_kaon_moments,
    pp_kaon_fade,
    pp_kaon_mult_fit,
    ProtonProton,
)
from am3_mojo.utils.constants import C_LIGHT


def kernel():
    var dat = AM3Arrays()
    var pp = ProtonProton(dat, False)
    pp.initialize_kernels(dat, 100.0, 0, 0, True, False)
    var nh = dat.nbins_had
    var species: List[Int] = [KP, KM, KS]
    for j in range(nh):
        var Ej = dat.egrid_had[j]
        var line = String(Ej)
        var csig = C_LIGHT * pp.sigma_inel_pipm[j]
        for s in species:
            var n = 0.0
            var bare = 0.0
            if Ej >= KPW_E_MIN:
                for i in range(j):
                    var Ei = dat.egrid_had[i]
                    n += dat.dlnE * pp.KER_PP2K[s].get(i, j) / csig
                    bare += dat.dlnE * pp_kaon_F(s, Ei / Ej, Ej, 0) * Ei / Ej
            elif pp.kd_l[s][j] >= 0:
                n = pp.kd_n[s][j] / csig
            line += String(" ", n, " ", bare)
        print(line)


def main() raises:
    var args = argv()
    var mode = String(args[1]) if len(args) > 1 else String("kernel")
    var species: List[Int] = [KP, KM, KS]
    if mode == "fit":
        for k in range(2, len(args)):
            var E = Float64(String(args[k]))
            print(
                pp_kaon_mult_fit(KP, E),
                pp_kaon_mult_fit(KM, E),
                pp_kaon_mult_fit(KS, E),
            )
    elif mode == "ct":
        for s in species:
            var nt: Float64
            nt, _ = pp_kaon_moments(s, KPW_E_MIN, 0)
            print(pp_kaon_mult_fit(s, KPW_E_MIN) / nt)
    elif mode == "fade":
        var c = Float64(String(args[2]))
        for k in range(3, len(args)):
            print(pp_kaon_fade(c, Float64(String(args[k]))))
    elif mode == "kernel":
        kernel()
    else:
        raise Error("unknown mode " + mode)
