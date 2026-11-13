"""Prints am3_mojo's polylog, the dilogarithm Li_2(z), at each argument z
(tests/unit/test_fixes.py)."""

from std.sys import argv

from am3_mojo.utils.specfun import polylog


def main() raises:
    var args = argv()
    for k in range(1, len(args)):
        print(polylog(Float64(String(args[k]))))
