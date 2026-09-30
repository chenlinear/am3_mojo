# Harness: evaluates am3_mojo's SolverAlgorithm.Fic on the same grid as fic_cpp.cc
from am3_mojo.solver.solver import SolverAlgorithm
from am3_mojo.utils.libm import c_exp
from am3_mojo.utils.constants import massp, masse


def main() raises:
    var s = SolverAlgorithm(
        "proton", 10, 0, 0.1, 1.0, 1.0, 1.0, 1e100, massp / masse, 1.0,
        False, False, True, False, 1.0, 1.0, 1.0, False, False,
    )
    var out = String()
    for i in range(200):
        var g = c_exp(0.1 * Float64(i))
        for j in range(400):
            var e = c_exp(0.1 * Float64(j)) * 1e-9
            out += String(s.Fic(g, e)) + "\n"
    print(out, end="")
