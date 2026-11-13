from std.utils.numerics import isfinite
from am3_mojo.utils.buffers import Vec, zeros
from am3_mojo.utils.libm import c_log


def scaled(v: Vec, factor: Float64) -> Vec:
    var res = v.copy()
    for i in range(len(res)):
        res[i] *= factor
    return res^


def summed(factor: Float64, a: Vec, b: Vec) -> Vec:
    var res = a.copy()
    for i in range(len(res)):
        res[i] += b[i]
    if factor != 1.0:
        for i in range(len(res)):
            res[i] *= factor
    return res^


def summed(factor: Float64, a: Vec, b: Vec, c: Vec) -> Vec:
    var res = a.copy()
    for i in range(len(res)):
        res[i] += b[i]
    for i in range(len(res)):
        res[i] += c[i]
    if factor != 1.0:
        for i in range(len(res)):
            res[i] *= factor
    return res^


def summed(factor: Float64, a: Vec, b: Vec, c: Vec, d: Vec) -> Vec:
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
    var res = zeros(len(rates))
    for i in range(len(rates)):
        res[i] = numerator / (sign * rates[i] + 1e-200)
    return res^


def constant_timescales(n: Int, numerator: Float64, rate: Float64) -> Vec:
    var res = zeros(n)
    for i in range(n):
        res[i] = numerator / (rate + 1e-200)
    return res^


def emission_timescales(density: Vec, source: Vec, numerator: Float64) -> Vec:
    var res = zeros(len(density))
    for i in range(len(density)):
        res[i] = numerator * (density[i] + 1e-100) / (source[i] + 1e-200)
    return res^


def check_finite(name: StaticString, value: Float64) raises:
    """Without this check, a NaN or an infinity reaches the steps. At a source
    that is not finite, the solver stops a step after it advanced some
    species."""
    if not isfinite(value):
        raise Error(String(name, "(): expected a finite number, got ", value))


def check_positive(name: StaticString, value: Float64) raises:
    """The acceleration factor divides the acceleration rate of the
    maximum-energy estimates. The logarithm of an energy gives its grid index.
    That of 0 is -inf, and that of a negative number is NaN."""
    if not (value > 0.0 and isfinite(value)):
        raise Error(
            String(name, "(): expected a positive finite number, got ", value)
        )


def check_on_grid(
    name: StaticString, E_eV: Float64, first_eV: Float64, dlnE: Float64
) raises:
    """An energy below the first point of a grid has no index on it. AM3 gives
    it 0 up to one bin below that point and a negative index further below,
    which numpy takes as one counted from the end of an array: 1e8 eV then
    reads the proton bin at 4e21 eV. Within 1e-9 of a bin of the first point,
    an energy is that point's, as in AM3Arrays._grid_index."""
    if c_log(E_eV / first_eV) / dlnE < -1e-9:
        raise Error(
            String(
                name,
                "(): ",
                E_eV,
                " eV lies below the grid, which starts at ",
                first_eV,
                " eV",
            )
        )


def is_timescale(t: Float64) -> Bool:
    """A timescale of at least 1e-200 s has a rate 1/t of at most 1e200 per
    second. With an escape fraction of at most 1e100 (check_fraction), an escape
    rate is at most 1e300. So the sums of the rates over a grid stay finite. At
    a sum that is not finite, the solver stops a step after it advanced some
    species. With a negative timescale, particles multiply instead of escaping,
    or gain energy."""
    return t >= 1e-200 and isfinite(t)


def check_timescale(name: StaticString, t: Float64) raises:
    check_finite(name, t)
    if not is_timescale(t):
        raise Error(
            String(name, "(): expected a timescale >= 1e-200 s, got ", t)
        )


def check_fraction(name: StaticString, frac: Float64) raises:
    """An escape fraction must be at most 1e100 (is_timescale). A negative
    fraction makes the particles grow instead of escaping."""
    check_finite(name, frac)
    if frac < 0.0:
        raise Error(String(name, "(): expected a fraction >= 0, got ", frac))
    if frac > 1e100:
        raise Error(
            String(name, "(): expected a fraction <= 1e100, got ", frac)
        )


def check_powerlaw(
    name: StaticString,
    volume: Float64,
    luminosity: Float64,
    emin: Float64,
    ebreak: Float64,
    emax: Float64,
    cutoff_steepness: Float64,
) raises:
    """A negative volume or luminosity flips the sign of the injection. At a
    volume of 0, the injection is not finite. A negative steepness s turns the
    cutoff exp(-(E/emax)^s) into one below emax."""
    for v in [volume, emin, ebreak, emax]:
        check_positive(name, v)
    if luminosity < 0.0:
        raise Error(
            String(name, "(): expected a luminosity >= 0, got ", luminosity)
        )
    if cutoff_steepness < 0.0:
        raise Error(
            String(
                name,
                "(): expected a cutoff steepness >= 0, got ",
                cutoff_steepness,
            )
        )


def check_input(src: Vec, n: Int) raises:
    if len(src) != n:
        raise Error(
            String("array has ", len(src), " elements, the energy grid has ", n)
        )
    for i in range(n):
        if not isfinite(src[i]):
            raise Error(
                String("array element ", i, " is ", src[i], ", not finite")
            )


def set_rates_from_timescales(
    mut rates: Vec, times: Vec, numerator: Float64
) raises:
    """The numerator carries sign and units. If not all timescales pass
    check_input and is_timescale, it raises before any rate is set. Each rate is
    reduced by the 1e-200 that `timescales` adds to a rate, so that the
    timescale a getter returns gives its rate back. For a rate of zero, the
    getter returns 1e200 s (in the rate's units). This timescale gives zero, not
    a rate of 1e-200, which the getter then reports as half of 1e200 s. Any rate
    above 1e-184 stays as it is to the last bit."""
    check_input(times, len(rates))
    for i in range(len(rates)):
        if not is_timescale(times[i]):
            raise Error(
                String(
                    "array element ",
                    i,
                    " is ",
                    times[i],
                    ", not a timescale >= 1e-200 s",
                )
            )
    var sign = -1.0 if numerator < 0.0 else 1.0
    for i in range(len(rates)):
        rates[i] = sign * max(abs(numerator) / times[i] - 1e-200, 0.0)


def assign_divided(mut dst: Vec, src: Vec, divisor: Float64) raises:
    """A negative value of a density or an injection rate is a negative number
    of particles. For an array that does not pass check_input or has a negative
    value, it raises before any value is set."""
    check_input(src, len(dst))
    for i in range(len(dst)):
        if src[i] < 0.0:
            raise Error(
                String("array element ", i, " is ", src[i], ", not >= 0")
            )
    for i in range(len(dst)):
        dst[i] = src[i] / divisor
