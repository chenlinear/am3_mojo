"""Port of the GSL pieces AM3 uses from `gsl_spline` (linear interpolation
with an acceleration cache) and `gsl_linalg_solve_tridiag`."""

from std.utils.numerics import nan
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *


struct LinearSpline(Copyable, Movable):
    """`gsl_spline` with `gsl_interp_linear` + `gsl_interp_accel`.

    Unlike GSL, which copies the arrays on `gsl_spline_init`, the spline keeps
    raw pointers to them: the caller guarantees that they outlive it and do
    not change while it is used, as in AM3.
    """

    var xa: FPtr
    var ya: FPtr
    var size: Int
    var xmin: Float64
    var xmax: Float64
    var cache: Int
    var error: Bool

    def __init__(out self, xa: FPtr, ya: FPtr, size: Int):
        self.xa = xa
        self.ya = ya
        self.size = size
        self.xmin = xa[unsafe_offset=0]
        self.xmax = xa[unsafe_offset=size - 1]
        self.cache = 0
        self.error = False

    @always_inline
    def _bsearch(self, x: Float64, index_lo: Int, index_hi: Int) -> Int:
        var ilo = index_lo
        var ihi = index_hi
        while ihi > ilo + 1:
            var i = (ihi + ilo) // 2
            if self.xa[unsafe_offset=i] > x:
                ihi = i
            else:
                ilo = i
        return ilo

    @always_inline
    def _find(mut self, x: Float64) -> Int:
        var x_index = self.cache
        if x < self.xa[unsafe_offset=x_index]:
            self.cache = self._bsearch(x, 0, x_index)
        elif x >= self.xa[unsafe_offset=x_index + 1]:
            self.cache = self._bsearch(x, x_index, self.size - 1)
        return self.cache

    @always_inline
    def eval(mut self, x: Float64) -> Float64:
        """gsl_spline_eval. Out-of-range input is a GSL domain error (which
        aborts the C++ program); here it sets `error` and returns NaN."""
        if x < self.xmin or x > self.xmax:
            self.error = True
            return nan[DType.float64]()
        var index = self._find(x)
        var x_lo = self.xa[unsafe_offset=index]
        var x_hi = self.xa[unsafe_offset=index + 1]
        var y_lo = self.ya[unsafe_offset=index]
        var y_hi = self.ya[unsafe_offset=index + 1]
        var dx = x_hi - x_lo
        if dx > 0.0:
            return y_lo + (x - x_lo) / dx * (y_hi - y_lo)
        return 0.0


def solve_tridiag(
    diag: FPtr,
    abovediag: FPtr,
    belowdiag: FPtr,
    rhs: FPtr,
    x: FPtr,
    N: Int,
    alpha: FPtr,
    z: FPtr,
) -> Bool:
    """gsl_linalg_solve_tridiag (non-symmetric Thomas algorithm).

    `alpha` and `z` are caller-provided work arrays of length N.
    Returns False when a zero pivot was encountered (GSL_EZERODIV).
    """
    var ok = True
    alpha[unsafe_offset=0] = diag[unsafe_offset=0]
    z[unsafe_offset=0] = rhs[unsafe_offset=0]
    if alpha[unsafe_offset=0] == 0.0:
        ok = False
    for i in range(1, N):
        var t = belowdiag[unsafe_offset=i - 1] / alpha[unsafe_offset=i - 1]
        alpha[unsafe_offset=i] = (
            diag[unsafe_offset=i] - t * abovediag[unsafe_offset=i - 1]
        )
        z[unsafe_offset=i] = rhs[unsafe_offset=i] - t * z[unsafe_offset=i - 1]
        if alpha[unsafe_offset=i] == 0.0:
            ok = False
    x[unsafe_offset=N - 1] = z[unsafe_offset=N - 1] / alpha[unsafe_offset=N - 1]
    if N >= 2:
        var i = N - 2
        while i >= 0:
            x[unsafe_offset=i] = (
                z[unsafe_offset=i]
                - abovediag[unsafe_offset=i] * x[unsafe_offset=i + 1]
            ) / alpha[unsafe_offset=i]
            i -= 1
    return ok
