from std.math import fma
from am3_mojo.utils.buffers import *
from am3_mojo.utils.parallel import parallel_run, parallel_tasks


struct MatD(Copyable, Movable):
    """`analyze()` records the non-zero row range of every column block of the
    GEMV kernel, so gemv_d skips rows that add only exact zeros. It also
    detects Toeplitz matrices (A[i, j] a function of i - j) and Hankel
    matrices (of i + j). Every kernel of an energy ratio on AM3's logarithmic
    grids is Toeplitz, and kernels of an energy product are Hankel. gemv_d
    then uses the ~rows + cols diagonal values instead of streaming the
    matrix.
    """

    var rows: Int
    var cols: Int
    var data: Vec
    var sp_lo: VecI
    var sp_hi: VecI
    var sp_valid: Bool
    var is_toeplitz: Bool
    var is_hankel: Bool
    var toe: Vec
    var toe_lo: Int
    var toe_hi: Int

    def __init__(out self):
        self.rows = 0
        self.cols = 0
        self.data = Vec()
        self.sp_lo = VecI()
        self.sp_hi = VecI()
        self.sp_valid = False
        self.is_toeplitz = False
        self.is_hankel = False
        self.toe = Vec()
        self.toe_lo = 0
        self.toe_hi = 0

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = zeros(rows * cols)
        self.sp_lo = VecI()
        self.sp_hi = VecI()
        self.sp_valid = False
        self.is_toeplitz = False
        self.is_hankel = False
        self.toe = Vec()
        self.toe_lo = 0
        self.toe_hi = 0

    @staticmethod
    def toeplitz(rows: Int, cols: Int, var t: Vec) -> MatD:
        """Toeplitz matrix A[i, j] = t[i - j + cols - 1] given by its diagonal
        values only (no dense storage)."""
        var A = MatD()
        A.rows = rows
        A.cols = cols
        A.toe_lo, A.toe_hi = nonzero_range(pd(t), len(t))
        A.toe = t^
        A.is_toeplitz = True
        A.sp_valid = True
        return A^

    @staticmethod
    def hankel(rows: Int, cols: Int, var t: Vec) -> MatD:
        """Hankel matrix A[i, j] = t[i + j] given by its diagonal values only
        (no dense storage)."""
        var A = MatD.toeplitz(rows, cols, t^)
        A.is_toeplitz = False
        A.is_hankel = True
        return A^

    @staticmethod
    def toeplitz_entry(m: Int, cols: Int) -> Tuple[Int, Int]:
        """The entry (i, j) of diagonal m of a Toeplitz matrix whose value
        `analyze()` takes for it: the first in column-major order."""
        var j = max(0, cols - 1 - m)
        return (m - (cols - 1) + j, j)

    def _detect_structure(mut self):
        self.is_toeplitz = False
        self.is_hankel = False
        if self.rows == 0 or self.cols == 0:
            return
        for kind in range(2):
            var nt = self.rows + self.cols - 1
            var t = zeros(nt)
            var seen = zerosi(nt)
            var p = pd(self.data)
            var ok = True
            for j in range(self.cols):
                for i in range(self.rows):
                    var v = p[unsafe_offset=j * self.rows + i]
                    var m = (i - j + self.cols - 1) if kind == 0 else (i + j)
                    if seen[m] == 0:
                        t[m] = v
                        seen[m] = 1
                    elif abs(v - t[m]) > 1e-12 * max(abs(v), abs(t[m])):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            self.toe_lo, self.toe_hi = nonzero_range(pd(t), nt)
            self.toe = t^
            self.is_toeplitz = kind == 0
            self.is_hankel = kind == 1
            return

    def analyze(mut self):
        var block = gemv_block_cols[8](self.rows, self.cols)
        var nb = (self.cols + block - 1) // block if self.cols > 0 else 0
        self.sp_lo = zerosi(nb)
        self.sp_hi = zerosi(nb)
        var p = pd(self.data)
        for b in range(nb):
            var lo = self.rows
            var hi = 0
            for j in range(b * block, min(b * block + block, self.cols)):
                for i in range(self.rows):
                    if p[unsafe_offset=j * self.rows + i] != 0.0:
                        lo = min(lo, i)
                        hi = max(hi, i + 1)
            if hi <= lo:
                lo = 0
                hi = 0
            self.sp_lo[b] = lo
            self.sp_hi[b] = hi
        self._detect_structure()
        self.sp_valid = True

    @inline(.always)
    def ptr(ref self) -> FPtr:
        return pd(self.data)

    @inline(.always)
    def get(self, i: Int, j: Int) -> Float64:
        return self.data[j * self.rows + i]

    @inline(.always)
    def set(mut self, i: Int, j: Int, v: Float64):
        self.data[j * self.rows + i] = v
        self.sp_valid = False


struct MatI(Copyable, Movable):
    var rows: Int
    var cols: Int
    var data: VecI

    def __init__(out self):
        self.rows = 0
        self.cols = 0
        self.data = VecI()

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = zerosi(rows * cols)

    @inline(.always)
    def get(self, i: Int, j: Int) -> Int:
        return self.data[j * self.rows + i]

    @inline(.always)
    def set(mut self, i: Int, j: Int, v: Int):
        self.data[j * self.rows + i] = v


comptime DOT_BLOCK = 4 * WD  # elements per step of `dot`'s main loop


def dot(a: FPtr, b: FPtr, n: Int) -> Float64:
    """A single running sum is limited by the latency of each add."""
    comptime W = WD
    var s0 = SIMD[DType.float64, W](0)
    var s1 = SIMD[DType.float64, W](0)
    var s2 = SIMD[DType.float64, W](0)
    var s3 = SIMD[DType.float64, W](0)
    var i = 0
    while i + 4 * W <= n:
        var pa = a.unsafe_offset(i)
        var pb = b.unsafe_offset(i)
        s0 = pa.unsafe_load[width=W]().fma(pb.unsafe_load[width=W](), s0)
        s1 = (
            pa.unsafe_offset(W)
            .unsafe_load[width=W]()
            .fma(pb.unsafe_offset(W).unsafe_load[width=W](), s1)
        )
        s2 = (
            pa.unsafe_offset(2 * W)
            .unsafe_load[width=W]()
            .fma(pb.unsafe_offset(2 * W).unsafe_load[width=W](), s2)
        )
        s3 = (
            pa.unsafe_offset(3 * W)
            .unsafe_load[width=W]()
            .fma(pb.unsafe_offset(3 * W).unsafe_load[width=W](), s3)
        )
        i += 4 * W
    while i + W <= n:
        s0 = (
            a.unsafe_offset(i)
            .unsafe_load[width=W]()
            .fma(b.unsafe_offset(i).unsafe_load[width=W](), s0)
        )
        i += W
    var s = ((s0 + s1) + (s2 + s3)).reduce_add()
    while i < n:
        s += a[unsafe_offset=i] * b[unsafe_offset=i]
        i += 1
    return s


def dot_nonzero(a: FPtr, b: FPtr, n: Int, lo: Int, hi: Int) -> Float64:
    """`dot(a, b, n)` for an `a` that is zero outside [lo, hi), bitwise. The
    blocks of `dot` (32 elements, then 8, then single ones) that lie outside
    add exact zeros to its partial sums, so they are skipped."""
    comptime W = WD
    var s0 = SIMD[DType.float64, W](0)
    var s1 = SIMD[DType.float64, W](0)
    var s2 = SIMD[DType.float64, W](0)
    var s3 = SIMD[DType.float64, W](0)
    var nb = n - n % (4 * W)
    var i = max(lo, 0) // (4 * W) * (4 * W)
    var iend = min(nb, hi)
    while i < iend:
        var pa = a.unsafe_offset(i)
        var pb = b.unsafe_offset(i)
        s0 = pa.unsafe_load[width=W]().fma(pb.unsafe_load[width=W](), s0)
        s1 = (
            pa.unsafe_offset(W)
            .unsafe_load[width=W]()
            .fma(pb.unsafe_offset(W).unsafe_load[width=W](), s1)
        )
        s2 = (
            pa.unsafe_offset(2 * W)
            .unsafe_load[width=W]()
            .fma(pb.unsafe_offset(2 * W).unsafe_load[width=W](), s2)
        )
        s3 = (
            pa.unsafe_offset(3 * W)
            .unsafe_load[width=W]()
            .fma(pb.unsafe_offset(3 * W).unsafe_load[width=W](), s3)
        )
        i += 4 * W
    i = nb
    while i + W <= n:
        if i + W > lo and i < hi:
            s0 = (
                a.unsafe_offset(i)
                .unsafe_load[width=W]()
                .fma(b.unsafe_offset(i).unsafe_load[width=W](), s0)
            )
        i += W
    var s = ((s0 + s1) + (s2 + s3)).reduce_add()
    for k in range(max(i, lo), min(n, hi)):
        s += a[unsafe_offset=k] * b[unsafe_offset=k]
    return s


@inline(.always)
def gemv_block_cols[elt_size: Int](rows: Int, cols: Int) -> Int:
    """Narrow matrices form one block. Others take 16 columns, or 4 when a block
    of 16 does not stay in cache."""
    if cols < 128:
        return cols
    return 16 if rows * elt_size < 32000 else 4


def gemv_block_rows[
    dtype: DType, W: Int
](
    A: Pointer[Scalar[dtype], MutUntrackedOrigin],
    lda: Int,
    j2: Int,
    jend: Int,
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    alpha: Scalar[dtype],
    res: Pointer[Scalar[dtype], MutUntrackedOrigin],
    r0: Int,
    r1: Int,
):
    var i = r0
    var valpha = SIMD[dtype, W](alpha)
    while i + W <= r1:
        var c = SIMD[dtype, W](0)
        for j in range(j2, jend):
            var a = A.unsafe_offset(j * lda + i).unsafe_load[width=W]()
            c = a.fma(SIMD[dtype, W](x[unsafe_offset=j]), c)
        var rp = res.unsafe_offset(i)
        rp.unsafe_store(c.fma(valpha, rp.unsafe_load[width=W]()))
        i += W
    comptime W4 = 4 if W > 4 else W
    while i + W4 <= r1:
        var c = SIMD[dtype, W4](0)
        for j in range(j2, jend):
            var a = A.unsafe_offset(j * lda + i).unsafe_load[width=W4]()
            c = a.fma(SIMD[dtype, W4](x[unsafe_offset=j]), c)
        var rp = res.unsafe_offset(i)
        rp.unsafe_store(
            c.fma(SIMD[dtype, W4](alpha), rp.unsafe_load[width=W4]())
        )
        i += W4
    while i < r1:
        var c = Scalar[dtype](0)
        for j in range(j2, jend):
            c = fma(A[unsafe_offset=j * lda + i], x[unsafe_offset=j], c)
        res[unsafe_offset=i] = fma(c, alpha, res[unsafe_offset=i])
        i += 1


@inline(.always)
def _diag_cols[
    hankel: Bool
](tlo: Int, thi: Int, C: Int, c1: Int, i0: Int, nb: Int) -> Tuple[Int, Int]:
    comptime if hankel:
        return (max(0, tlo - (i0 + nb - 1)), min(C, thi - i0))
    else:
        return (max(0, i0 + c1 - thi + 1), min(C, i0 + nb - 1 + c1 - tlo + 1))


def diagonal_gemv[
    hankel: Bool
](
    t: FPtr,
    tlo: Int,
    thi: Int,
    R: Int,
    C: Int,
    x: FPtr,
    alpha: Float64,
    res: FPtr,
):
    """res = alpha * A x for the R x C matrix A[i, j] = t[i - j + C - 1]
    (Toeplitz) or t[i + j] (Hankel), with t zero outside [tlo, thi)."""
    comptime W = WD
    comptime sgn = 1 if hankel else -1
    var c1 = 0 if hankel else C - 1
    var av = SIMD[DType.float64, W](alpha)
    var i0 = 0
    while i0 + 4 * W <= R:
        var jlo: Int
        var jhi: Int
        jlo, jhi = _diag_cols[hankel](tlo, thi, C, c1, i0, 4 * W)
        var a0 = SIMD[DType.float64, W](0)
        var a1 = SIMD[DType.float64, W](0)
        var a2 = SIMD[DType.float64, W](0)
        var a3 = SIMD[DType.float64, W](0)
        for j in range(jlo, jhi):
            var xv = SIMD[DType.float64, W](x[unsafe_offset=j])
            var base = t.unsafe_offset(i0 + sgn * j + c1)
            a0 = base.unsafe_load[width=W]().fma(xv, a0)
            a1 = base.unsafe_offset(W).unsafe_load[width=W]().fma(xv, a1)
            a2 = base.unsafe_offset(2 * W).unsafe_load[width=W]().fma(xv, a2)
            a3 = base.unsafe_offset(3 * W).unsafe_load[width=W]().fma(xv, a3)
        res.unsafe_offset(i0).unsafe_store(a0 * av)
        res.unsafe_offset(i0 + W).unsafe_store(a1 * av)
        res.unsafe_offset(i0 + 2 * W).unsafe_store(a2 * av)
        res.unsafe_offset(i0 + 3 * W).unsafe_store(a3 * av)
        i0 += 4 * W
    while i0 + W <= R:
        var jlo: Int
        var jhi: Int
        jlo, jhi = _diag_cols[hankel](tlo, thi, C, c1, i0, W)
        var a0 = SIMD[DType.float64, W](0)
        for j in range(jlo, jhi):
            var xv = SIMD[DType.float64, W](x[unsafe_offset=j])
            a0 = (
                t.unsafe_offset(i0 + sgn * j + c1)
                .unsafe_load[width=W]()
                .fma(xv, a0)
            )
        res.unsafe_offset(i0).unsafe_store(a0 * av)
        i0 += W
    for i in range(i0, R):
        var jlo: Int
        var jhi: Int
        jlo, jhi = _diag_cols[hankel](tlo, thi, C, c1, i, 1)
        var acc = 0.0
        for j in range(jlo, jhi):
            acc = fma(
                t[unsafe_offset=i + sgn * j + c1], x[unsafe_offset=j], acc
            )
        res[unsafe_offset=i] = acc * alpha


def diagonal_gemv2(
    t: FPtr,
    tlo: Int,
    thi: Int,
    R: Int,
    C: Int,
    x0: FPtr,
    x1: FPtr,
    alpha: Float64,
    res0: FPtr,
    res1: FPtr,
):
    """`diagonal_gemv` for two vectors in one pass over t. Each result is
    bitwise that of `diagonal_gemv`: every row adds the same products in the
    same order. (The columns that a row block reaches beyond a row's own band
    add exact zeros, so the blocks can differ.)"""
    comptime W = WD
    var c1 = C - 1
    var av = SIMD[DType.float64, W](alpha)
    var i0 = 0
    while i0 + 2 * W <= R:
        var jlo: Int
        var jhi: Int
        jlo, jhi = _diag_cols[False](tlo, thi, C, c1, i0, 2 * W)
        var a0 = SIMD[DType.float64, W](0)
        var a1 = SIMD[DType.float64, W](0)
        var b0 = SIMD[DType.float64, W](0)
        var b1 = SIMD[DType.float64, W](0)
        for j in range(jlo, jhi):
            var xv = SIMD[DType.float64, W](x0[unsafe_offset=j])
            var yv = SIMD[DType.float64, W](x1[unsafe_offset=j])
            var base = t.unsafe_offset(i0 - j + c1)
            var t0 = base.unsafe_load[width=W]()
            var t1 = base.unsafe_offset(W).unsafe_load[width=W]()
            a0 = t0.fma(xv, a0)
            a1 = t1.fma(xv, a1)
            b0 = t0.fma(yv, b0)
            b1 = t1.fma(yv, b1)
        res0.unsafe_offset(i0).unsafe_store(a0 * av)
        res0.unsafe_offset(i0 + W).unsafe_store(a1 * av)
        res1.unsafe_offset(i0).unsafe_store(b0 * av)
        res1.unsafe_offset(i0 + W).unsafe_store(b1 * av)
        i0 += 2 * W
    while i0 + W <= R:
        var jlo: Int
        var jhi: Int
        jlo, jhi = _diag_cols[False](tlo, thi, C, c1, i0, W)
        var a0 = SIMD[DType.float64, W](0)
        var b0 = SIMD[DType.float64, W](0)
        for j in range(jlo, jhi):
            var t0 = t.unsafe_offset(i0 - j + c1).unsafe_load[width=W]()
            a0 = t0.fma(SIMD[DType.float64, W](x0[unsafe_offset=j]), a0)
            b0 = t0.fma(SIMD[DType.float64, W](x1[unsafe_offset=j]), b0)
        res0.unsafe_offset(i0).unsafe_store(a0 * av)
        res1.unsafe_offset(i0).unsafe_store(b0 * av)
        i0 += W
    for i in range(i0, R):
        var jlo: Int
        var jhi: Int
        jlo, jhi = _diag_cols[False](tlo, thi, C, c1, i, 1)
        var acc0 = 0.0
        var acc1 = 0.0
        for j in range(jlo, jhi):
            var tv = t[unsafe_offset=i - j + c1]
            acc0 = fma(tv, x0[unsafe_offset=j], acc0)
            acc1 = fma(tv, x1[unsafe_offset=j], acc1)
        res0[unsafe_offset=i] = acc0 * alpha
        res1[unsafe_offset=i] = acc1 * alpha


@inline(.always)
def _toeplitz_window(A: MatD, j1: Int, r0: Int) -> Tuple[Int, Int, Int]:
    """A on its rows from r0 and its columns up to j1 is the Toeplitz matrix of
    the diagonal values of A from off = r0 + cols - j1 on."""
    var off = r0 + A.cols - j1
    return (off, max(A.toe_lo - off, 0), A.toe_hi - off)


def toeplitz_window_zero(A: MatD, j0: Int, j1: Int, r0: Int, r1: Int) -> Bool:
    var w = _toeplitz_window(A, j1, r0)
    return w[2] <= 0 or w[1] >= (r1 - r0) + (j1 - j0) - 1


def toeplitz_gemv[
    two: Bool
](
    A: MatD,
    x0: FPtr,
    x1: FPtr,
    j0: Int,
    j1: Int,
    r0: Int,
    r1: Int,
    alpha: Float64,
    res0: FPtr,
    res1: FPtr,
):
    """x0 must be zero outside the columns [j0, j1): the other columns add
    exact zeros."""
    var off: Int
    var tlo: Int
    var thi: Int
    off, tlo, thi = _toeplitz_window(A, j1, r0)
    var t = pd(A.toe).unsafe_offset(off)
    comptime if two:
        diagonal_gemv2(
            t,
            tlo,
            thi,
            r1 - r0,
            j1 - j0,
            x0.unsafe_offset(j0),
            x1.unsafe_offset(j0),
            alpha,
            res0.unsafe_offset(r0),
            res1.unsafe_offset(r0),
        )
    else:
        diagonal_gemv[False](
            t,
            tlo,
            thi,
            r1 - r0,
            j1 - j0,
            x0.unsafe_offset(j0),
            alpha,
            res0.unsafe_offset(r0),
        )


def gemv_d(mut A: MatD, x: FPtr, alpha: Float64, res: FPtr):
    if not A.sp_valid:
        A.analyze()
    if A.is_toeplitz or A.is_hankel:
        var j0: Int
        var j1: Int
        j0, j1 = nonzero_range(x, A.cols)
        if A.is_toeplitz:
            toeplitz_gemv[False](A, x, x, j0, j1, 0, A.rows, alpha, res, res)
        else:
            # A on the columns [j0, j1): the Hankel matrix of the diagonal
            # values from j0 on
            diagonal_gemv[True](
                pd(A.toe).unsafe_offset(j0),
                max(A.toe_lo - j0, 0),
                A.toe_hi - j0,
                A.rows,
                j1 - j0,
                x.unsafe_offset(j0),
                alpha,
                res,
            )
        return
    for i in range(A.rows):
        res[unsafe_offset=i] = 0
    var block = gemv_block_cols[8](A.rows, A.cols)
    var Ap = A.ptr()
    for b in range(len(A.sp_lo)):
        var lo = A.sp_lo[b]
        var hi = A.sp_hi[b]
        if hi <= lo:
            continue
        var j2 = b * block
        gemv_block_rows[DType.float64, WD](
            Ap, A.rows, j2, min(j2 + block, A.cols), x, alpha, res, lo, hi
        )


def gemv_rows(A: MatD, x: FPtr, alpha: Float64, res: FPtr, r0: Int, r1: Int):
    """Rows [r0, r1) of `gemv_d` (A analyzed), bitwise. Every row adds the
    same products in the same order, in whichever block of rows it falls.
    (The columns that a block reaches beyond a row's own band add exact
    zeros.)"""
    if r1 <= r0:
        return
    if A.is_toeplitz or A.is_hankel:
        # A[r0 + i, j] = t'[i -+ j ...] with t' = t + r0
        var t = pd(A.toe).unsafe_offset(r0)
        var lo = A.toe_lo - r0
        var hi = A.toe_hi - r0
        if A.is_toeplitz:
            diagonal_gemv[False](
                t, lo, hi, r1 - r0, A.cols, x, alpha, res.unsafe_offset(r0)
            )
        else:
            diagonal_gemv[True](
                t, lo, hi, r1 - r0, A.cols, x, alpha, res.unsafe_offset(r0)
            )
        return
    for i in range(r0, r1):
        res[unsafe_offset=i] = 0
    var block = gemv_block_cols[8](A.rows, A.cols)
    var Ap = pd(A.data)
    for b in range(len(A.sp_lo)):
        var lo = max(A.sp_lo[b], r0)
        var hi = min(A.sp_hi[b], r1)
        if hi <= lo:
            continue
        var j2 = b * block
        gemv_block_rows[DType.float64, WD](
            Ap, A.rows, j2, min(j2 + block, A.cols), x, alpha, res, lo, hi
        )


struct _GemvRowsJob(Movable):
    var A: Pointer[MatD, MutUntrackedOrigin]
    var x: FPtr
    var alpha: Float64
    var res: FPtr

    def __init__(
        out self,
        A: Pointer[MatD, MutUntrackedOrigin],
        x: FPtr,
        alpha: Float64,
        res: FPtr,
    ):
        self.A = A
        self.x = x
        self.alpha = alpha
        self.res = res


def _gemv_rows_work(
    job: Pointer[_GemvRowsJob, MutUntrackedOrigin], t: Int, nt: Int
):
    ref J = job[]
    var nb = (J.A[].rows + 4 * WD - 1) // (4 * WD)
    var r0 = t * nb // nt * 4 * WD
    var r1 = min((t + 1) * nb // nt * 4 * WD, J.A[].rows)
    gemv_rows(J.A[], J.x, J.alpha, J.res, r0, r1)


def gemv_d_par(mut A: MatD, x: FPtr, alpha: Float64, res: FPtr, nthreads: Int):
    """`gemv_d` on up to nthreads threads: the same values (gemv_rows)."""
    if not A.sp_valid:
        A.analyze()
    var job = _GemvRowsJob(
        Pointer(to=A).unsafe_origin_cast[MutUntrackedOrigin](), x, alpha, res
    )
    var nb = (A.rows + 4 * WD - 1) // (4 * WD)
    parallel_run[_gemv_rows_work](job, max(1, min(nthreads, nb)))


def gemv_pair(
    mut A: MatD, x0: FPtr, x1: FPtr, alpha: Float64, res0: FPtr, res1: FPtr
):
    """Each result is bitwise that of gemv_d. Toeplitz matrices take one pass
    over their diagonals for both vectors. A is analyzed (run_gemv_pairs)."""
    if A.is_toeplitz:
        var lo0: Int
        var hi0: Int
        var lo1: Int
        var hi1: Int
        lo0, hi0 = nonzero_range(x0, A.cols)
        lo1, hi1 = nonzero_range(x1, A.cols)
        var j0 = lo1 if hi0 == 0 else (lo0 if hi1 == 0 else min(lo0, lo1))
        var j1 = max(hi0, hi1)
        toeplitz_gemv[True](A, x0, x1, j0, j1, 0, A.rows, alpha, res0, res1)
    else:
        gemv_d(A, x0, alpha, res0)
        gemv_d(A, x1, alpha, res1)


struct GemvPair(Copyable, Movable):
    var A: Pointer[MatD, MutUntrackedOrigin]
    var x0: FPtr
    var x1: FPtr
    var alpha: Float64
    var res0: FPtr
    var res1: FPtr

    def __init__(
        out self,
        ref A: MatD,
        x0: FPtr,
        x1: FPtr,
        alpha: Float64,
        res0: FPtr,
        res1: FPtr,
    ):
        self.A = (
            Pointer(to=A)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        self.x0 = x0
        self.x1 = x1
        self.alpha = alpha
        self.res0 = res0
        self.res1 = res1


def _gemv_pairs_work(jobs: Pointer[List[GemvPair], MutUntrackedOrigin], k: Int):
    ref J = jobs[][k]
    gemv_pair(J.A[], J.x0, J.x1, J.alpha, J.res0, J.res1)


def run_gemv_pairs(mut jobs: List[GemvPair], nthreads: Int):
    """The matrices are analyzed first, so that the workers only read them.
    Each product writes its own outputs: the results do not depend on the
    threads."""
    for k in range(len(jobs)):
        if not jobs[k].A[].sp_valid:
            jobs[k].A[].analyze()
    parallel_tasks[_gemv_pairs_work](jobs, len(jobs), nthreads)


struct SparseGemvF(Copyable, Movable):
    """PackedGemvF packs the matrices so that the blocked GEMV skips the
    all-zero rows of every column block. Skipped rows add only exact zeros:
    the result equals the dense product.
    """

    var nmat: Int
    var rows: Int
    var cols: Int
    var block: Int
    var nb: Int
    var lo: VecI
    var hi: VecI

    def __init__(out self, nmat: Int, rows: Int, cols: Int):
        self.nmat = nmat
        self.rows = rows
        self.cols = cols
        self.block = gemv_block_cols[4](rows, cols)
        self.nb = (cols + self.block - 1) // self.block if cols > 0 else 0
        self.lo = zerosi(nmat * self.nb)
        self.hi = zerosi(nmat * self.nb)


struct _PackJob[T: AnyType](Movable):
    var src: Pointer[Self.T, MutUntrackedOrigin]
    var row_min: Pointer[VecI, MutUntrackedOrigin]
    var sp: Pointer[SparseGemvF, MutUntrackedOrigin]
    var P: Pointer[PackedGemvF, MutUntrackedOrigin]

    def __init__(
        out self,
        src: Pointer[Self.T, MutUntrackedOrigin],
        row_min: Pointer[VecI, MutUntrackedOrigin],
        sp: Pointer[SparseGemvF, MutUntrackedOrigin],
        P: Pointer[PackedGemvF, MutUntrackedOrigin],
    ):
        self.src = src
        self.row_min = row_min
        self.sp = sp
        self.P = P


def _sparse_scan_work[
    T: AnyType,
    entry: def(Pointer[T, MutUntrackedOrigin], Int, Int, Int) thin -> Float32,
](job: Pointer[_PackJob[T], MutUntrackedOrigin], t: Int, nt: Int):
    """Row by row, so that a source stored by rows reads its memory in order."""
    ref S = job[].sp[]
    var src = job[].src
    var k = t
    while k < S.nmat:
        var rmin = job[].row_min[][k]
        for b in range(S.nb):
            var j2 = b * S.block
            var jend = min(j2 + S.block, S.cols)
            var lo = S.rows
            var hi = 0
            for i in range(rmin, S.rows):
                for j in range(j2, jend):
                    if entry(src, k, i, j) != 0.0:
                        lo = min(lo, i)
                        hi = i + 1
                        break
            if hi <= lo:
                lo = 0
                hi = 0
            S.lo[k * S.nb + b] = lo
            S.hi[k * S.nb + b] = hi
        k += nt


comptime PW = 16


def _packed_copy_work[
    T: AnyType,
    entry: def(Pointer[T, MutUntrackedOrigin], Int, Int, Int) thin -> Float32,
](job: Pointer[_PackJob[T], MutUntrackedOrigin], t: Int, nt: Int):
    """Each matrix has its own range of the data."""
    ref sp = job[].sp[]
    ref P = job[].P[]
    var src = job[].src
    var dp = pf(P.data)
    var k = t
    while k < P.nmat:
        for b in range(P.nb):
            var t2 = k * P.nb + b
            var j2 = b * P.block
            var bw = min(P.block, P.cols - j2)
            var hi = sp.hi[t2]
            for c in range(P.nch[t2]):
                for j in range(bw):
                    var dst = dp.unsafe_offset(P.off[t2] + (c * bw + j) * PW)
                    for l in range(PW):
                        var r = P.lo[t2] + c * PW + l
                        dst[unsafe_offset=l] = entry(
                            src, k, r, j2 + j
                        ) if r < hi else Float32(0.0)
        k += nt


struct PackedGemvF(Copyable, Movable):
    """Only the non-zero row range [lo, hi) of every column block is kept, in
    chunks of PW rows. Within a chunk, each of the block's columns holds PW
    contiguous rows, so the kernel reads memory strictly sequentially.
    Rows >= hi in the last chunk are zero padding and receive exact zeros.
    """

    var nmat: Int
    var rows: Int
    var cols: Int
    var block: Int
    var nb: Int
    var lo: VecI
    var nch: VecI
    var off: VecI
    var data: VecF

    def __init__(out self):
        self.nmat = 0
        self.rows = 0
        self.cols = 0
        self.block = 1
        self.nb = 0
        self.lo = VecI()
        self.nch = VecI()
        self.off = VecI()
        self.data = VecF()

    @staticmethod
    def pack[
        T: AnyType,
        //,
        entry: def(
            Pointer[T, MutUntrackedOrigin], Int, Int, Int
        ) thin -> Float32,
    ](
        src: T, nmat: Int, rows: Int, cols: Int, row_min: VecI, nthreads: Int
    ) -> PackedGemvF:
        """The matrices K[k][i][j] = entry(src, k, i, j) are packed without
        storing them densely. `row_min[k]`: rows below it are never read for
        matrix k."""
        var sp = SparseGemvF(nmat, rows, cols)
        # (a source can read the rows through an unchecked pointer)
        for k in range(nmat):
            debug_assert(row_min[k] >= 0, "PackedGemvF: row_min below 0")
        var P = PackedGemvF()
        P.nmat = nmat
        P.rows = rows
        P.cols = cols
        P.block = sp.block
        P.nb = sp.nb
        var job = _PackJob[T](
            Pointer(to=src)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=row_min)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=sp).unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=P).unsafe_origin_cast[MutUntrackedOrigin](),
        )
        var nt = max(1, min(nthreads, nmat))
        parallel_run[_sparse_scan_work[T, entry]](job, nt)
        P.lo = zerosi(nmat * P.nb)
        P.nch = zerosi(nmat * P.nb)
        P.off = zerosi(nmat * P.nb)
        var total = 0
        for t in range(nmat * P.nb):
            var b = t % P.nb if P.nb > 0 else 0
            var bw = min(P.block, cols - b * P.block)
            var n = sp.hi[t] - sp.lo[t]
            var nc = (n + PW - 1) // PW if n > 0 else 0
            P.lo[t] = sp.lo[t]
            P.nch[t] = nc
            P.off[t] = total
            total += nc * bw * PW
        P.data = zerosf(total)
        parallel_run[_packed_copy_work[T, entry]](job, nt)
        _ = sp^  # (read by the workers through a pointer)
        return P^

    def get(self, k: Int, i: Int, j: Int) -> Float32:
        var b = j // self.block
        var t = k * self.nb + b
        var r = i - self.lo[t]
        if r < 0 or r >= self.nch[t] * PW:
            return 0.0
        var j2 = b * self.block
        var bw = min(self.block, self.cols - j2)
        return self.data[self.off[t] + ((r // PW) * bw + j - j2) * PW + r % PW]

    @inline(.always)
    def res_len(self) -> Int:
        return self.rows + PW

    def apply(
        self,
        k: Int,
        x: F32Ptr,
        alpha: Float32,
        res: F32Ptr,
        row_lo: Int = 0,
        row_hi: Int = Int.MAX,
    ):
        """res must have the length res_len(). Only rows in [row_lo, row_hi) are
        computed, for callers that read no others. Only these rows start from
        zero, and whole PW-row chunks outside are skipped."""
        for i in range(max(row_lo, 0), min(row_hi, self.rows + PW)):
            res[unsafe_offset=i] = 0
        var dp = pf(self.data)
        var valpha = SIMD[DType.float32, PW](alpha)
        for b in range(self.nb):
            var t = k * self.nb + b
            var nc = self.nch[t]
            if nc == 0:
                continue
            var j2 = b * self.block
            var bw = min(self.block, self.cols - j2)
            var base = dp.unsafe_offset(self.off[t])
            var r = self.lo[t]
            var c = max(0, (row_lo - r) // PW)
            if row_hi < Int.MAX:
                nc = min(nc, (row_hi - r + PW - 1) // PW)
            # four independent row chunks in flight to hide FMA latency
            while c + 4 <= nc:
                var a0 = SIMD[DType.float32, PW](0)
                var a1 = SIMD[DType.float32, PW](0)
                var a2 = SIMD[DType.float32, PW](0)
                var a3 = SIMD[DType.float32, PW](0)
                var cp = base.unsafe_offset(c * bw * PW)
                var st = bw * PW
                for j in range(bw):
                    var xv = SIMD[DType.float32, PW](x[unsafe_offset=j2 + j])
                    var q = cp.unsafe_offset(j * PW)
                    a0 = q.unsafe_load[width=PW]().fma(xv, a0)
                    a1 = q.unsafe_offset(st).unsafe_load[width=PW]().fma(xv, a1)
                    a2 = (
                        q.unsafe_offset(2 * st)
                        .unsafe_load[width=PW]()
                        .fma(xv, a2)
                    )
                    a3 = (
                        q.unsafe_offset(3 * st)
                        .unsafe_load[width=PW]()
                        .fma(xv, a3)
                    )
                var rp = res.unsafe_offset(r + c * PW)
                rp.unsafe_store(a0.fma(valpha, rp.unsafe_load[width=PW]()))
                rp = rp.unsafe_offset(PW)
                rp.unsafe_store(a1.fma(valpha, rp.unsafe_load[width=PW]()))
                rp = rp.unsafe_offset(PW)
                rp.unsafe_store(a2.fma(valpha, rp.unsafe_load[width=PW]()))
                rp = rp.unsafe_offset(PW)
                rp.unsafe_store(a3.fma(valpha, rp.unsafe_load[width=PW]()))
                c += 4
            while c < nc:
                var acc = SIMD[DType.float32, PW](0)
                var cp = base.unsafe_offset(c * bw * PW)
                for j in range(bw):
                    var a = cp.unsafe_offset(j * PW).unsafe_load[width=PW]()
                    acc = a.fma(
                        SIMD[DType.float32, PW](x[unsafe_offset=j2 + j]), acc
                    )
                var rp = res.unsafe_offset(r + c * PW)
                rp.unsafe_store(acc.fma(valpha, rp.unsafe_load[width=PW]()))
                c += 1
