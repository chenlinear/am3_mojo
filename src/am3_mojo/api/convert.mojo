"""Conversions between the internal arrays and the values the Python API
returns or accepts (unit scaling, component sums, rate <-> timescale)."""

from am3_mojo.utils.buffers import Vec, zeros


def scaled(v: Vec, factor: Float64) -> Vec:
    """Copy of `v` multiplied by `factor` (internal units -> physical units)."""
    var res = v.copy()
    for i in range(len(res)):
        res[i] *= factor
    return res^


# fixed arities: Mojo 1.1.0 miscompiles a variadic pack of list references in
# builds with assertions enabled
def summed(factor: Float64, a: Vec, b: Vec) -> Vec:
    """Element-wise sum of the arrays, multiplied by `factor`."""
    var res = a.copy()
    for i in range(len(res)):
        res[i] += b[i]
    if factor != 1.0:
        for i in range(len(res)):
            res[i] *= factor
    return res^


def summed(factor: Float64, a: Vec, b: Vec, c: Vec, d: Vec) -> Vec:
    """Element-wise sum of the arrays, multiplied by `factor`."""
    var res = a.copy()
    for i in range(len(res)):
        res[i] += b[i]
    for i in range(len(res)):
        res[i] += c[i]
    for i in range(len(res)):
        res[i] += d[i]
    if factor != 1.0:
        for i in range(len(res)):
            res[i] *= factor
    return res^


def timescales(rates: Vec, numerator: Float64, sign: Float64) -> Vec:
    """`numerator / (sign * rate + 1e-200)`: timescale of a (signed) rate array.
    """
    var res = zeros(len(rates))
    for i in range(len(rates)):
        res[i] = numerator / (sign * rates[i] + 1e-200)
    return res^


def constant_timescales(n: Int, numerator: Float64, rate: Float64) -> Vec:
    """Timescale array of an energy-independent rate."""
    var res = zeros(n)
    for i in range(n):
        res[i] = numerator / (rate + 1e-200)
    return res^


def emission_timescales(density: Vec, source: Vec, numerator: Float64) -> Vec:
    """Time for a source term to build up the current density,
    `numerator * (n + 1e-100) / (q + 1e-200)`."""
    var res = zeros(len(density))
    for i in range(len(density)):
        res[i] = numerator * (density[i] + 1e-100) / (source[i] + 1e-200)
    return res^


def check_length(src: Vec, n: Int) raises:
    """An input array must have one value per bin of its energy grid."""
    if len(src) != n:
        raise Error(
            String("array has ", len(src), " elements, the energy grid has ", n)
        )


def set_rates_from_timescales(
    mut rates: Vec, times: Vec, numerator: Float64
) raises:
    """`rates[i] = numerator / times[i]` (numerator carries sign and units)."""
    check_length(times, len(rates))
    for i in range(len(rates)):
        rates[i] = numerator / times[i]


def assign_divided(mut dst: Vec, src: Vec, divisor: Float64) raises:
    """`dst[i] = src[i] / divisor` (physical units -> internal units)."""
    check_length(src, len(dst))
    for i in range(len(dst)):
        dst[i] = src[i] / divisor
