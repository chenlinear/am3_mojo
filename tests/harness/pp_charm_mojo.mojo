"""Prints am3_mojo's charm production in pp collisions
(tests/unit/test_added_processes.py), by mode:
  kernel           per proton energy E_j of the default hadron grid a line with
                   E_j [GeV], the cross section times the fraction of E_j in
                   all charm hadrons (sig_kappa_C) [cm^2], and per pair of a
                   hadron and its antiparticle (D+-; D0 and anti-D0; Ds+-;
                   Lambda_c+ and anti-Lambda_c-) the
                   cross section sigma_h [cm^2] of one of them and its mean
                   fraction <x> of E_j: from the kernel from 1 TeV on, from the
                   delta-functional term below;
  sigma E...       pp_charm_sigma_cc [cm^2] at each proton energy E [GeV];
  inel E...        AM3's inelastic cross section sigma_inel [cm^2] (Kafexhiu et
                   al. 2014, eq. 1, with the pi+- threshold) at each E [GeV];
  shape d E        the closed form's g at x_m = exp(-m d), m = 0, 1, ...
                   (pp_charm_shape), one value per line, at E [GeV].
"""

from std.sys import argv

from am3_mojo.model.arrays import AM3Arrays
from am3_mojo.physics.proton_proton import (
    pp_charm_sigma_cc,
    pp_charm_shape,
    ProtonProton,
)
from am3_mojo.utils.constants import C_LIGHT, MASS_PI


def kernel():
    var dat = AM3Arrays()
    var pp = ProtonProton(dat, False)
    pp.initialize_kernels(dat, 100.0, 0, 0, False, True)
    var nh = dat.nbins_had
    for j in range(nh):
        var Ej = dat.egrid_had[j]
        var line = String(Ej, " ", pp.sig_kappa_C[j])
        for h in range(4):
            var n = 0.0
            var e = 0.0
            for i in range(nh):
                var k = pp.KER_PP2C[h].get(i, j) / C_LIGHT
                n += dat.dlnE * k
                e += dat.dlnE * k * dat.egrid_had[i] / Ej
            var l = pp.cd_l[h][j]
            if l >= 0:
                var w = pp.cd_w[h][j]
                n = pp.cd_n[h][j] / C_LIGHT
                e = (
                    n
                    * ((1.0 - w) * dat.egrid_had[l] + w * dat.egrid_had[l + 1])
                    / Ej
                )
            line += String(" ", n, " ", e / n if n > 0.0 else 0.0)
        print(line)


def main() raises:
    var args = argv()
    var mode = String(args[1]) if len(args) > 1 else String("kernel")
    if mode == "kernel":
        kernel()
    elif mode == "sigma":
        for k in range(2, len(args)):
            print(pp_charm_sigma_cc(Float64(String(args[k]))))
    elif mode == "inel":
        var dat = AM3Arrays()
        var pp = ProtonProton(dat, False)
        for k in range(2, len(args)):
            print(pp.sigma_inel_Kafexhiu(Float64(String(args[k])), MASS_PI))
    elif mode == "shape":
        var g = pp_charm_shape(
            Float64(String(args[3])), Float64(String(args[2]))
        )
        for v in g:
            print(v)
    else:
        raise Error("unknown mode " + mode)
