"""Array/pointer aliases, the SIMD width and small buffer helpers."""

comptime FPtr = Pointer[Float64, MutUntrackedOrigin]
comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime IPtr = Pointer[Int, MutUntrackedOrigin]

comptime Vec = List[Float64]
comptime VecF = List[Float32]
comptime VecI = List[Int]

# SIMD width of the hot loops (NEON: 2 x f64 per register; LLVM splits wider
# vectors, which gives extra instruction-level parallelism).
comptime WD = 8

# ============================================================================
# small array helpers
# ============================================================================


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
    """Raw (unchecked, untracked) pointer to the storage of a Float64 list."""
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
    """`v.resize(n); std::fill(v, 0)`."""
    if len(v) != n:
        v = zeros(n)
    else:
        fill_zero(v)
