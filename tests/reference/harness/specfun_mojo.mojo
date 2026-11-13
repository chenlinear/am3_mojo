# Harness: am3_mojo's gsl_bessel_Knu and int_bessel_k53 on the grid of
# specfun_cpp.cc
from std.math import fma

from am3_mojo.utils.libm import c_exp
from am3_mojo.utils.specfun import gsl_bessel_Knu, int_bessel_k53


def main() raises:
    var out = String()
    var nus = [1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 5.0 / 3.0]
    for k in range(4):
        for i in range(2000):
            out += (
                String(
                    gsl_bessel_Knu(nus[k], c_exp(fma(0.0075, Float64(i), -9.0)))
                )
                + "\n"
            )
    for i in range(2000):
        out += (
            String(int_bessel_k53(c_exp(fma(0.0075, Float64(i), -9.0)))) + "\n"
        )
    print(out, end="")
