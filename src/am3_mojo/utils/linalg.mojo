"""Matrices and SIMD matrix-vector products.

* `MatD` / `gemv_d`: column-major float64 matrices. Toeplitz and Hankel
  matrices are multiplied as a SIMD convolution (`diagonal_gemv`), others by a
  product vectorised over rows that skips all-zero row ranges.
* `Mat3F` / `PackedGemvF`: float32 3D kernels stored as packed non-zero row
  windows (used by the Bethe-Heitler pair emission).
"""

from std.math import fma
from std.sys import size_of
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *


struct MatD(Copyable, Movable):
    """Column-major Float64 matrix.

    `analyze()` records the non-zero row range of every column block of the
    GEMV kernel, so gemv_d skips rows that would only add exact zeros. It also
    detects Toeplitz matrices (A[i, j] a function of i - j, as for every kernel
    of an energy ratio on AM3's logarithmic grids) and Hankel matrices (of
    i + j, for kernels of an energy product). gemv_d then uses the ~rows + cols
    diagonal values instead of streaming the matrix. The dense data is kept as
    well (about 10 MB for AM3's decay matrices).
    """

    var rows: Int
    var cols: Int
    var data: Vec
    var sp_lo: VecI
    var sp_hi: VecI
    var sp_valid: Bool
    # diagonal form: Toeplitz A[i, j] = toe[i - j + cols - 1] or Hankel
    # A[i, j] = toe[i + j], non-zero for indices in [toe_lo, toe_hi)
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
        """rows x cols Toeplitz matrix A[i, j] = t[i - j + cols - 1] given by
        its diagonal values only (no dense storage)."""
        var A = MatD()
        A.rows = rows
        A.cols = cols
        var lo = len(t)
        var hi = 0
        for m in range(len(t)):
            if t[m] != 0.0:
                lo = min(lo, m)
                hi = m + 1
        A.toe = t^
        A.toe_lo = lo if hi > 0 else 0
        A.toe_hi = hi
        A.is_toeplitz = True
        A.sp_valid = True
        return A^

    def _detect_structure(mut self):
        """Toeplitz (A[i, j] = t[i - j + cols - 1]) or Hankel (A[i, j] =
        t[i + j]) if every such diagonal is constant to 1e-12 (relative)."""
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
                    elif c_fabs(v - t[m]) > 1e-12 * max(
                        c_fabs(v), c_fabs(t[m])
                    ):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            var lo = nt
            var hi = 0
            for m in range(nt):
                if t[m] != 0.0:
                    lo = min(lo, m)
                    hi = m + 1
            self.toe = t^
            self.toe_lo = lo
            self.toe_hi = hi
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

    @always_inline
    def ptr(ref self) -> FPtr:
        return pd(self.data)

    @always_inline
    def col(ref self, j: Int) -> FPtr:
        return pd(self.data).unsafe_offset(j * self.rows)

    @always_inline
    def get(self, i: Int, j: Int) -> Float64:
        return self.data[j * self.rows + i]

    @always_inline
    def set(mut self, i: Int, j: Int, v: Float64):
        self.data[j * self.rows + i] = v
        self.sp_valid = False

    @always_inline
    def add(mut self, i: Int, j: Int, v: Float64):
        self.data[j * self.rows + i] += v
        self.sp_valid = False


struct MatI(Copyable, Movable):
    """Column-major Int matrix."""

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

    @always_inline
    def get(self, i: Int, j: Int) -> Int:
        return self.data[j * self.rows + i]

    @always_inline
    def set(mut self, i: Int, j: Int, v: Int):
        self.data[j * self.rows + i] = v


struct Mat3F(Copyable, Movable):
    """A stack of `nmat` column-major float matrices of
    identical shape stored back to back."""

    var nmat: Int
    var rows: Int
    var cols: Int
    var data: VecF

    def __init__(out self):
        self.nmat = 0
        self.rows = 0
        self.cols = 0
        self.data = VecF()

    def __init__(out self, nmat: Int, rows: Int, cols: Int):
        self.nmat = nmat
        self.rows = rows
        self.cols = cols
        self.data = zerosf(nmat * rows * cols)

    @always_inline
    def mat(ref self, k: Int) -> F32Ptr:
        return pf(self.data).unsafe_offset(k * self.rows * self.cols)

    @always_inline
    def get(self, k: Int, i: Int, j: Int) -> Float32:
        return self.data[(k * self.cols + j) * self.rows + i]

    @always_inline
    def set(mut self, k: Int, i: Int, j: Int, v: Float32):
        self.data[(k * self.cols + j) * self.rows + i] = v

    @always_inline
    def add(mut self, k: Int, i: Int, j: Int, v: Float32):
        self.data[(k * self.cols + j) * self.rows + i] += v


def dot(a: FPtr, b: FPtr, n: Int) -> Float64:
    """sum_i a[i] * b[i] over i < n, with four independent SIMD partial sums
    (a single running sum would be limited by the latency of each add)."""
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


@always_inline
def gemv_block_cols[elt_size: Int](rows: Int, cols: Int) -> Int:
    """Column-block width of the blocked GEMV: narrow matrices form one block;
    otherwise 16 columns, or 4 when a block of 16 would not stay in cache."""
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
    """One column block [j2, jend) applied to rows [r0, r1)."""
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


@always_inline
def _diag_cols[
    hankel: Bool
](tlo: Int, thi: Int, C: Int, c1: Int, i0: Int, nb: Int) -> Tuple[Int, Int]:
    """Columns j whose diagonal index i +- j + c1 lies in [tlo, thi) for some
    row i in [i0, i0 + nb)."""
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
    (Toeplitz) or t[i + j] (Hankel), with t zero outside [tlo, thi).

    Output rows go in blocks of 4 x W, then W, then single rows, each
    accumulated in registers: every column that reaches a block adds x[j]
    times a contiguous slice of t (loads and FMAs only)."""
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


def gemv_d(mut A: MatD, x: FPtr, alpha: Float64, res: FPtr):
    """res = alpha * A x, skipping all-zero row ranges of each column block
    (the structure is analysed lazily and cached); Toeplitz and Hankel
    matrices use `diagonal_gemv`."""
    if not A.sp_valid:
        A.analyze()
    if A.is_toeplitz:
        diagonal_gemv[False](
            pd(A.toe), A.toe_lo, A.toe_hi, A.rows, A.cols, x, alpha, res
        )
        return
    if A.is_hankel:
        diagonal_gemv[True](
            pd(A.toe), A.toe_lo, A.toe_hi, A.rows, A.cols, x, alpha, res
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


struct SparseGemvF(Copyable, Movable):
    """Non-zero structure of a stack of float matrices, so that the blocked
    GEMV skips the all-zero rows of every column block.

    `lo[k*nb+b]`..`hi[k*nb+b]` is the row range holding all non-zeros of
    column block b of matrix k (within the rows the caller needs). Skipped
    rows would only add exact zeros: the result equals the dense product.
    """

    var nmat: Int
    var rows: Int
    var cols: Int
    var block: Int
    var nb: Int
    var lo: VecI
    var hi: VecI

    def __init__(out self):
        self.nmat = 0
        self.rows = 0
        self.cols = 0
        self.block = 1
        self.nb = 0
        self.lo = VecI()
        self.hi = VecI()

    def __init__(out self, K: Mat3F, row_min: VecI):
        """`row_min[k]`: rows below it are never read for matrix k."""
        self.nmat = K.nmat
        self.rows = K.rows
        self.cols = K.cols
        self.block = gemv_block_cols[4](K.rows, K.cols)
        self.nb = (K.cols + self.block - 1) // self.block if K.cols > 0 else 0
        self.lo = zerosi(self.nmat * self.nb)
        self.hi = zerosi(self.nmat * self.nb)
        for k in range(self.nmat):
            var A = K.mat(k)
            for b in range(self.nb):
                var j2 = b * self.block
                var jend = min(j2 + self.block, self.cols)
                var lo = self.rows
                var hi = 0
                for j in range(j2, jend):
                    var colp = A.unsafe_offset(j * self.rows)
                    for i in range(row_min[k], self.rows):
                        if colp[unsafe_offset=i] != 0.0:
                            if i < lo:
                                lo = i
                            if i + 1 > hi:
                                hi = i + 1
                if hi <= lo:
                    lo = 0
                    hi = 0
                self.lo[k * self.nb + b] = lo
                self.hi[k * self.nb + b] = hi

    def nnz_fraction(self) -> Float64:
        var tot = 0
        for t in range(len(self.lo)):
            var b = t % self.nb if self.nb > 0 else 0
            var w = min(self.block, self.cols - b * self.block)
            tot += (self.hi[t] - self.lo[t]) * w
        var full = self.nmat * self.rows * self.cols
        return Float64(tot) / Float64(full) if full > 0 else 0.0

    def apply(self, K: Mat3F, k: Int, x: F32Ptr, alpha: Float32, res: F32Ptr):
        """res[0:rows] = alpha * K[k] x (rows outside the recorded ranges are 0).
        """
        for i in range(self.rows):
            res[unsafe_offset=i] = 0
        var A = K.mat(k)
        for b in range(self.nb):
            var lo = self.lo[k * self.nb + b]
            var hi = self.hi[k * self.nb + b]
            if hi <= lo:
                continue
            var j2 = b * self.block
            var jend = min(j2 + self.block, self.cols)
            gemv_block_rows[DType.float32, WF](
                A, self.rows, j2, jend, x, alpha, res, lo, hi
            )


comptime PW = 16  # rows per packed SIMD chunk


struct PackedGemvF(Copyable, Movable):
    """Stack of float matrices re-packed for a streaming blocked GEMV.

    Only the non-zero row range [lo, hi) of every column block is kept, in
    chunks of PW rows; within a chunk, each of the block's columns holds PW
    contiguous rows, so the kernel reads memory strictly sequentially:
        data[off + (c * bw + j) * PW + l] = K[k][lo + c*PW + l][j2 + j]
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

    def __init__(out self, K: Mat3F, row_min: VecI):
        var sp = SparseGemvF(K, row_min)
        self.nmat = K.nmat
        self.rows = K.rows
        self.cols = K.cols
        self.block = sp.block
        self.nb = sp.nb
        self.lo = zerosi(self.nmat * self.nb)
        self.nch = zerosi(self.nmat * self.nb)
        self.off = zerosi(self.nmat * self.nb)
        var total = 0
        for t in range(self.nmat * self.nb):
            var b = t % self.nb if self.nb > 0 else 0
            var bw = min(self.block, self.cols - b * self.block)
            var n = sp.hi[t] - sp.lo[t]
            var nc = (n + PW - 1) // PW if n > 0 else 0
            self.lo[t] = sp.lo[t]
            self.nch[t] = nc
            self.off[t] = total
            total += nc * bw * PW
        self.data = zerosf(total)
        var dp = pf(self.data)
        for k in range(self.nmat):
            var A = K.mat(k)
            for b in range(self.nb):
                var t = k * self.nb + b
                var j2 = b * self.block
                var bw = min(self.block, self.cols - j2)
                var hi = sp.hi[t]
                for c in range(self.nch[t]):
                    for j in range(bw):
                        var colp = A.unsafe_offset((j2 + j) * self.rows)
                        var dst = dp.unsafe_offset(
                            self.off[t] + (c * bw + j) * PW
                        )
                        for l in range(PW):
                            var r = self.lo[t] + c * PW + l
                            dst[unsafe_offset=l] = colp[
                                unsafe_offset=r
                            ] if r < hi else Float32(0.0)

    @always_inline
    def res_len(self) -> Int:
        """Length the result buffer must have (rows + padding)."""
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
        """res[0:rows] = alpha * K[k] x; res needs res_len().

        Only rows in [row_lo, row_hi) are guaranteed: whole PW-row chunks
        outside are skipped, for callers that multiply those rows by exact
        zeros afterwards."""
        for i in range(self.rows + PW):
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
