"""Prints am3_mojo's number_split for the bin width dlnE (first argument) at
each position x (in bins). Each line holds the grid point below x and the
share of the point above it (tests/unit/test_fixes.py)."""

from std.sys import argv

from am3_mojo.utils.specfun import number_split


def main() raises:
    var args = argv()
    var dlnE = Float64(String(args[1]))
    for k in range(2, len(args)):
        var s = number_split(Float64(String(args[k])), dlnE)
        print(s[0], s[1])
