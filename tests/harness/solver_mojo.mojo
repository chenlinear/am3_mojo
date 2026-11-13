"""Prints what three guards of am3_mojo's solver do, for
tests/unit/test_fixes.py.
Argument: spline, rk4, rk4q or rk4q_no_sink.

spline: whether a linear spline on [0, 0.1] evaluated at NaN reports an error
(1 or 0), and its value.
rk4: SolverAlgorithm.rk4_gslspl_a in 5 steps from 0.8 to 1.1 over a spline of
ln(a/|A|) = -1 with the nodes 0, 0.1, ..., 1.1: whether the spline reported an
error, and the integral. The fifth step ends at 0.8 plus five steps of 0.06,
summed, which rounds past 1.1, the spline's top.
rk4q, rk4q_no_sink: SolverAlgorithm.rk4_gslspl_q from 0.8 to 1.1 with q = 2,
1/|A| = 1 and ln(a/|A|) = -1 (rk4q_no_sink: -20, a negligible sink) on the same
nodes: whether a spline reported an error, and the integral. Its tenth step ends
at 0.8 plus ten steps of 0.03, summed, which rounds past 1.1."""

from std.sys import argv
from std.utils.numerics import nan

from am3_mojo.solver.solver import SolverAlgorithm
from am3_mojo.utils.buffers import zeros, pd
from am3_mojo.utils.interp import LinearSpline


def main() raises:
    var xa = zeros(12)
    var ya = zeros(12)
    for k in range(12):
        xa[k] = 0.1 * Float64(k)
        ya[k] = -1.0
    if argv()[1] == "spline":
        var spl = LinearSpline(pd(xa), pd(ya), 2)
        var v = spl.eval(nan[DType.float64]())
        print(1 if spl.error else 0, v)
    else:
        var mode = argv()[1]
        if mode == "rk4q_no_sink":
            for k in range(12):
                ya[k] = -20.0
        var qa = zeros(12)
        var la = zeros(12)
        for k in range(12):
            qa[k] = 2.0
        var spl = LinearSpline(pd(xa), pd(ya), 12)
        var spl_q = LinearSpline(pd(xa), pd(qa), 12)
        var spl_l = LinearSpline(pd(xa), pd(la), 12)
        var s = SolverAlgorithm(
            "electron",
            12,
            0,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            False,
            False,
            False,
            False,
            1.0,
            1.0,
            1.0,
            False,
            False,
        )
        var v = 0.0
        if mode == "rk4":
            v = s.rk4_gslspl_a(xa[8], xa[11], 5, xa[0], xa[11], spl)
        else:
            v = s.rk4_gslspl_q(xa[8], xa[11], xa[0], xa[11], spl, spl_q, spl_l)
        print(1 if spl.error or spl_q.error or spl_l.error else 0, v)
        _ = qa^
        _ = la^
    _ = xa^  # (read by the splines through pointers)
    _ = ya^
