"""The code's constants (src/am3_mojo/utils/constants.mojo, which cites PDG 2026 for each): the derivation checks, the
scripts and the tests read the code's inputs from here."""
import re

import sympy as sp

import latex as L

PATH = L.ROOT / "src" / "am3_mojo" / "utils" / "constants.mojo"


def definitions():
    """{name: expression} of every `comptime NAME: Float64 = ...` of the file, as written (an expression continued on
    the next lines inside parentheses joined, comments cut)."""
    lines = PATH.read_text().splitlines()
    out = {}
    for i, line in enumerate(lines):
        m = re.match(r"comptime (\w+): Float64 = (.*)", line)
        if m:
            expr, j = m.group(2).split("#")[0], i
            while expr.count("(") > expr.count(")"):
                j += 1
                expr += " " + lines[j].split("#")[0]
            out[m.group(1)] = " ".join(expr.split())
    return out


def read(names):
    """{name: value} of these constants, exact: decimal literals as rationals, expressions (1.0 / 137.035999177,
    sqrt(SIGMA_T)) as written, with the constants they use resolved."""
    defs, values = definitions(), {}

    def value(n):
        if n not in values:
            used = set(re.findall(r"(?<![\w.])[A-Za-z_]\w*", defs[n])) - {"sqrt"}
            values[n] = sp.sympify(defs[n], locals={**{u: value(u) for u in used}, "sqrt": sp.sqrt}, rational=True)
        return values[n]

    return {n: value(n) for n in names}
