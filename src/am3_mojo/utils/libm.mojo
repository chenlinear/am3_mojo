"""Thin wrappers around the platform libm and C conversion semantics.

Transcendental functions go through libm (the library the C++ AM3 links
against) rather than Mojo's std.math approximations."""

from std.ffi import external_call
from std.math import sqrt as _sqrt, floor as _floor

# ============================================================================
# libm wrappers
# ============================================================================


@always_inline
def c_exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


@always_inline
def c_log(x: Float64) -> Float64:
    return external_call["log", Float64](x)


@always_inline
def c_log10(x: Float64) -> Float64:
    return external_call["log10", Float64](x)


@always_inline
def c_pow(x: Float64, y: Float64) -> Float64:
    return external_call["pow", Float64](x, y)


@always_inline
def c_exp10(x: Float64) -> Float64:
    """clang lowers `pow(10, x)` to `__exp10(x)` on Darwin."""
    return external_call["__exp10", Float64](x)


@always_inline
def c_exp2(x: Float64) -> Float64:
    """clang lowers `pow(2, x)` to `exp2(x)`."""
    return external_call["exp2", Float64](x)


@always_inline
def c_sin(x: Float64) -> Float64:
    return external_call["sin", Float64](x)


@always_inline
def c_sinh(x: Float64) -> Float64:
    return external_call["sinh", Float64](x)


@always_inline
def c_cosh(x: Float64) -> Float64:
    return external_call["cosh", Float64](x)


@always_inline
def c_asin(x: Float64) -> Float64:
    return external_call["asin", Float64](x)


@always_inline
def c_nextafter(x: Float64, y: Float64) -> Float64:
    return external_call["nextafter", Float64](x, y)


@always_inline
def c_sqrt(x: Float64) -> Float64:
    # IEEE-754 square root is correctly rounded everywhere.
    return _sqrt(x)


@always_inline
def c_floor(x: Float64) -> Float64:
    return _floor(x)


@always_inline
def c_fabs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


@always_inline
def to_int(x: Float64) -> Int:
    """C-style truncating double -> int conversion."""
    return Int(x)


@always_inline
def c_mod(a: Int, b: Int) -> Int:
    """C-style `%` (sign follows the dividend)."""
    var r = a - (a // b) * b
    if r != 0 and ((r < 0) != (a < 0)):
        r -= b
    return r


@always_inline
def fbool(b: Bool) -> Float64:
    return 1.0 if b else 0.0
