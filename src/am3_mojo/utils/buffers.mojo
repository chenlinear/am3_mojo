comptime FPtr = Pointer[Float64, MutUntrackedOrigin]
comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime IPtr = Pointer[Int, MutUntrackedOrigin]

comptime Vec = List[Float64]
comptime VecF = List[Float32]
comptime VecI = List[Int]

# SIMD width of the hot loops (NEON: 2 x f64 per register). LLVM splits wider
# vectors, which gives extra instruction-level parallelism.
comptime WD = 8


@inline(.always)
def zeros(n: Int) -> Vec:
    return Vec(length=n, fill=0.0)


@inline(.always)
def zerosf(n: Int) -> VecF:
    return VecF(length=n, fill=Float32(0.0))


@inline(.always)
def zerosi(n: Int) -> VecI:
    return VecI(length=n, fill=0)


@inline(.always)
def pd(ref v: Vec) -> FPtr:
    return (
        v.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


@inline(.always)
def pf(ref v: VecF) -> F32Ptr:
    return (
        v.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


@inline(.always)
def pi_(ref v: VecI) -> IPtr:
    return (
        v.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def fill_zero(mut v: Vec):
    var p = pd(v)
    for i in range(len(v)):
        p[unsafe_offset=i] = 0.0


def resize_zero(mut v: Vec, n: Int):
    if len(v) != n:
        v = zeros(n)
    else:
        fill_zero(v)


@inline(.always)
def nonzero_range(p: FPtr, n: Int, start: Int = 0) -> Tuple[Int, Int]:
    var hi = n
    while hi > start and p[unsafe_offset=hi - 1] == 0.0:
        hi -= 1
    if hi <= start:
        return (0, 0)
    var lo = start
    while p[unsafe_offset=lo] == 0.0:
        lo += 1
    return (lo, hi)


def copy_into(mut dst: Vec, src: Vec):
    """`dst = src.copy()` without an allocation, where the lengths agree
    (otherwise a copy: the first time, or on another grid)."""
    if len(dst) != len(src):
        dst = src.copy()
        return
    var d = pd(dst)
    var q = pd(src)
    for i in range(len(src)):
        d[unsafe_offset=i] = q[unsafe_offset=i]
