"""Prints am3_mojo's int_bessel_k53, the integral of K_5/3 from x to infinity,
at each argument x (tests/unit/test_fixes.py)."""

from std.sys import argv

from am3_mojo.utils.specfun import int_bessel_k53


def main() raises:
    var args = argv()
    for k in range(1, len(args)):
        print(int_bessel_k53(Float64(String(args[k]))))
