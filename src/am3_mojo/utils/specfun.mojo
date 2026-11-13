"""Port of AM3/math.cc and the parts of GSL that AM3 calls:
`gsl_sf_bessel_Knu`."""

from std.builtin.globals import global_constant
from std.math import floor, sqrt
from std.utils.numerics import nan
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *

comptime GSL_DBL_EPSILON: Float64 = 2.2204460492503131e-16
comptime GSL_SQRT_DBL_MAX: Float64 = 1.3407807929942596e154
comptime GSL_SQRT_DBL_MIN: Float64 = 1.4916681462400413e-154
comptime GSL_LOG_DBL_MAX: Float64 = 7.0978271289338397e02
comptime GSL_LOG_DBL_MIN: Float64 = -7.0839641853226408e02

# ------------------------------------------------------------------------------
# GSL: Chebyshev series for the Temme gamma functions (specfunc/bessel_temme.c)
# ------------------------------------------------------------------------------
comptime _G1: Array[Float64, 14] = [
    -1.14516408366268311786898152867,
    0.00636085311347084238122955495,
    0.00186245193007206848934643657,
    0.000152833085873453507081227824,
    0.000017017464011802038795324732,
    -6.4597502923347254354668326451e-07,
    -5.1819848432519380894104312968e-08,
    4.5189092894858183051123180797e-10,
    3.2433227371020873043666259180e-11,
    6.8309434024947522875432400828e-13,
    2.8353502755172101513119628130e-14,
    -7.9883905769323592875638087541e-16,
    -3.3726677300771949833341213457e-17,
    -3.6586334809210520744054437104e-20,
]


@inline(.always)
def _g1_coef(j: Int) -> Float64:
    return global_constant[_G1]()[j]


comptime _G2: Array[Float64, 15] = [
    1.882645524949671835019616975350,
    -0.077490658396167518329547945212,
    -0.018256714847324929419579340950,
    0.0006338030209074895795923971731,
    0.0000762290543508729021194461175,
    -9.5501647561720443519853993526e-07,
    -8.8927268107886351912431512955e-08,
    -1.9521334772319613740511880132e-09,
    -9.4003052735885162111769579771e-11,
    4.6875133849532393179290879101e-12,
    2.2658535746925759582447545145e-13,
    -1.1725509698488015111878735251e-15,
    -7.0441338200245222530843155877e-17,
    -2.4377878310107693650659740228e-18,
    -7.5225243218253901727164675011e-20,
]


@inline(.always)
def _g2_coef(j: Int) -> Float64:
    return global_constant[_G2]()[j]


def _cheb_eval_g1(x: Float64) -> Float64:
    # cheb_series g1_cs = {g1_dat, 13, -1, 1, 7}
    var d = 0.0
    var dd = 0.0
    var y = (2.0 * x - (-1.0) - 1.0) / (1.0 - (-1.0))
    var y2 = 2.0 * y
    var j = 13
    while j >= 1:
        var temp = d
        d = y2 * d - dd + _g1_coef(j)
        dd = temp
        j -= 1
    d = y * d - dd + 0.5 * _g1_coef(0)
    return d


def _cheb_eval_g2(x: Float64) -> Float64:
    # cheb_series g2_cs = {g2_dat, 14, -1, 1, 8}
    var d = 0.0
    var dd = 0.0
    var y = (2.0 * x - (-1.0) - 1.0) / (1.0 - (-1.0))
    var y2 = 2.0 * y
    var j = 14
    while j >= 1:
        var temp = d
        d = y2 * d - dd + _g2_coef(j)
        dd = temp
        j -= 1
    d = y * d - dd + 0.5 * _g2_coef(0)
    return d


def _bessel_K_scaled_temme(nu: Float64, x: Float64) -> Tuple[Float64, Float64]:
    """GSL gsl_sf_bessel_K_scaled_temme."""
    comptime max_iter = 15000
    var half_x = 0.5 * x
    var ln_half_x = c_log(half_x)
    var half_x_nu = c_exp(nu * ln_half_x)
    var pi_nu = PI * nu
    var sigma = -nu * ln_half_x
    var sinrat = 1.0 if abs(pi_nu) < GSL_DBL_EPSILON else pi_nu / c_sin(pi_nu)
    var sinhrat = 1.0 if abs(sigma) < GSL_DBL_EPSILON else c_sinh(sigma) / sigma
    var ex = c_exp(x)

    # gsl_sf_temme_gamma
    var anu = abs(nu)
    var xx = 4.0 * anu - 1.0
    var g1 = _cheb_eval_g1(xx)
    var g2 = _cheb_eval_g2(xx)
    var g_1mnu = 1.0 / (g2 + nu * g1)
    var g_1pnu = 1.0 / (g2 - nu * g1)

    var fk = sinrat * (c_cosh(sigma) * g1 - sinhrat * ln_half_x * g2)
    var pk = 0.5 / half_x_nu * g_1pnu
    var qk = 0.5 * half_x_nu * g_1mnu
    var hk = pk
    var ck = 1.0
    var sum0 = fk
    var sum1 = hk
    var k = 0
    while k < max_iter:
        k += 1
        var kf = Float64(k)
        fk = (kf * fk + pk + qk) / (kf * kf - nu * nu)
        ck *= half_x * half_x / kf
        pk /= kf - nu
        qk /= kf + nu
        hk = -kf * fk + pk
        var del0 = ck * fk
        var del1 = ck * hk
        sum0 += del0
        sum1 += del1
        if abs(del0) < 0.5 * abs(sum0) * GSL_DBL_EPSILON:
            break
    var K_nu = sum0 * ex
    var K_nup1 = sum1 * 2.0 / x * ex
    return (K_nu, K_nup1)


def _bessel_K_scaled_steed_temme_CF2(
    nu: Float64, x: Float64
) -> Tuple[Float64, Float64]:
    """GSL gsl_sf_bessel_K_scaled_steed_temme_CF2 (x >= 2)."""
    comptime maxiter = 10000
    var bi = 2.0 * (1.0 + x)
    var di = 1.0 / bi
    var delhi = di
    var hi = di
    var qi = 0.0
    var qip1 = 1.0
    var ai = -(0.25 - nu * nu)
    var a1 = ai
    var ci = -ai
    var Qi = -ai
    var s = 1.0 + Qi * delhi
    var i = 2
    while i <= maxiter:
        var fi = Float64(i)
        ai -= 2.0 * Float64(i - 1)
        ci = -ai * ci / fi
        var tmp = (qi - bi * qip1) / ai
        qi = qip1
        qip1 = tmp
        Qi += ci * qip1
        bi += 2.0
        di = 1.0 / (bi + ai * di)
        delhi = (bi * di - 1.0) * delhi
        hi += delhi
        var dels = Qi * delhi
        s += dels
        if abs(dels / s) < GSL_DBL_EPSILON:
            break
        i += 1
    hi *= -a1
    var K_nu = sqrt(PI / (2.0 * x)) / s
    var K_nup1 = K_nu * (nu + x + 0.5 - hi) / x
    return (K_nu, K_nup1)


def _exp_mult(x: Float64, y: Float64) -> Float64:
    """gsl_sf_exp_mult_err_e of GSL, value only."""
    var ay = abs(y)
    if y == 0.0:
        return 0.0
    elif (x < 0.5 * GSL_LOG_DBL_MAX and x > 0.5 * GSL_LOG_DBL_MIN) and (
        ay < 0.8 * GSL_SQRT_DBL_MAX and ay > 1.2 * GSL_SQRT_DBL_MIN
    ):
        var ex = c_exp(x)
        return y * ex
    else:
        var ly = c_log(ay)
        var lnr = x + ly
        if lnr > GSL_LOG_DBL_MAX - 0.01:
            return Float64.MAX  # GSL signals an overflow error here
        elif lnr < GSL_LOG_DBL_MIN + 0.01:
            return 0.0  # GSL signals an underflow error here
        else:
            var sy = 1.0 if y >= 0.0 else -1.0
            var M = floor(x)
            var N = floor(ly)
            var a = x - M
            var b = ly - N
            var eMN = c_exp(M + N)
            var eab = c_exp(a + b)
            return sy * eMN * eab


def gsl_bessel_Knu(nu: Float64, x: Float64) -> Float64:
    """Port of gsl_sf_bessel_Knu(nu, x) for x > 0, nu >= 0."""
    if x <= 0.0 or nu < 0.0:
        # GSL signals a domain error and returns NaN. AM3 never gets here.
        return Float64.MAX
    var N = Int(nu + 0.5)
    var mu = nu - Float64(N)
    var K_mu: Float64
    var K_mup1: Float64
    if x < 2.0:
        K_mu, K_mup1 = _bessel_K_scaled_temme(mu, x)
    else:
        K_mu, K_mup1 = _bessel_K_scaled_steed_temme_CF2(mu, x)
    var K_nu = K_mu
    var K_nup1 = K_mup1
    var e10 = 0
    for n in range(N):
        var K_num1 = K_nu
        K_nu = K_nup1
        if abs(K_nu) > GSL_SQRT_DBL_MAX:
            var p = floor(c_log(abs(K_nu)) / LN10)
            var factor = c_pow(10.0, p)
            K_num1 /= factor
            K_nu /= factor
            e10 += Int(p)
        K_nup1 = 2.0 * (mu + Float64(n) + 1.0) / x * K_nu + K_num1
    var val = K_nu
    # gsl_sf_result_smash_e
    if e10 != 0:
        var av = abs(val)
        if (
            GSL_SQRT_DBL_MIN < av
            and av < GSL_SQRT_DBL_MAX
            and 0.49 * GSL_LOG_DBL_MIN < Float64(e10)
            and Float64(e10) < 0.49 * GSL_LOG_DBL_MAX
        ):
            val = val * c_exp(Float64(e10) * LN10)
        else:
            val = _exp_mult(Float64(e10) * LN10, val)
    return _exp_mult(-x, val)


@inline(.always)
def beta(g: Float64, REG: Float64) -> Float64:
    if g < 10.0:
        return sqrt(1.0 - (1.0 / (g * g)) + REG)
    var ig2 = 1.0 / (g * g)
    return 1.0 - 0.5 * ig2 * (1.0 + 0.25 * ig2)


@inline(.always)
def signf(a: Int) -> Int:
    if a % 2 == 0:
        return 1
    else:
        return -1


def int_bk53m(x: Float64) -> Float64:
    """The integral of K_5/3 from x to infinity for x > 1e-3, as int_0^inf
    exp(-x cosh u) cosh(5u/3) / cosh(u) du (from K_nu(t) = int_0^inf
    exp(-t cosh u) cosh(nu u) du) by the trapezoidal rule. The integrand is
    even and analytic for |Im u| < pi/2, so the rule converges exponentially.
    With the step h = min(1/4, 0.4 / sqrt(x)) (the integrand's width is
    1/sqrt(x) at large x), it converges to 2e-15 for 1e-3 <= x <= 1e3, in at
    most 46 nodes.
    (AM3 summed 100 rectangles in ln t from x to 10, 1.9 to 3.1 % too much.
    Above x = 10, AM3 took (1.253314 + 0.139338 / x) e^-x / sqrt(x), 5.4 %
    too little right above x = 10, where AM3's value jumped. The 1/x term of
    the asymptotic series is sqrt(pi/2) 55/72 = 0.957 / x.)"""
    var h = min(0.25, 0.4 / sqrt(x))
    var s = 0.5  # (the node u = 0, at the end of the range)
    for k in range(1, 100):
        var u = Float64(k) * h
        var sh = c_sinh(0.5 * u)
        var v = (
            c_exp(-2.0 * x * sh * sh)
            * c_cosh(u * (5.0 / 3.0))
            / (1.0 + 2.0 * sh * sh)
        )
        s += v
        if v < 1e-17 * s:
            break
    return h * s * c_exp(-x)


def int_bk53s(x: Float64) -> Float64:
    """The integral of K_5/3 from x to infinity for x <= 1e-3: the series of
    K_5/3 = (pi/sqrt3) (I_-5/3 - I_5/3) integrated term by term, within 3e-14.
    The constant term: 2 K_2/3(x) - int_x^inf K_1/3 at small x,
    int_0^inf K_1/3 = pi/sqrt3. (AM3 had 2.14953, 1.814 and 0.142393 and no
    x^(4/3) term: 1.9e-7 too little at x = 1e-3.)"""
    var c = c_pow(2.0, 2.0 / 3.0) * 1.3541179394264004169  # Gamma(2/3)
    var a = PI / sqrt(3.0)
    return (
        c * c_pow(x, -2.0 / 3.0)
        - a
        + 0.1875 * c * c_pow(x, 4.0 / 3.0)
        - 27.0 / 160.0 * a / c * c_pow(x, 8.0 / 3.0)
    )


def int_bessel_k53(x: Float64) -> Float64:
    """The integral of K_5/3 from x to infinity, x > 0."""
    if x > 0.001:
        return int_bk53m(x)
    return int_bk53s(x)


# B_2k / (2k + 1)!, k = 1 .. 9 (B_2k: the Bernoulli numbers): the series of
# Li_2 in u = -ln(1 - z)
comptime _LI2_SERIES: Array[Float64, 9] = [
    1.0 / 36.0,
    -1.0 / 3600.0,
    1.0 / 211680.0,
    -1.0 / 10886400.0,
    1.0 / 526901760.0,
    -691.0 / 16999766784000.0,
    1.0 / 1120863744000.0,
    -3617.0 / 181400588328960000.0,
    43867.0 / 97072790126247936000.0,
]


def _li2_series(z: Float64) -> Float64:
    """Li_2(z) for -1 <= z <= 1/2 by its series in u = -ln(1 - z), where
    |u| <= ln 2. The terms after k = 9 add less than 1e-20 of the sum."""
    var u = -c_log1p(-z)
    var u2 = u * u
    var s = 0.0
    var k = 8
    while k >= 0:
        s = s * u2 + global_constant[_LI2_SERIES]()[k]
        k -= 1
    return u - 0.25 * u2 + u * u2 * s


def polylog(z: Float64) -> Float64:
    """The dilogarithm Li_2(z) = -int_0^z ln(1 - t) / t dt for real z <= 1, to
    rounding. NaN for z > 1, where Li_2 is complex, and for a NaN, so that the
    solver's checks of its rates see such an argument, which a finite value
    would hide. (AM3's math.cc sums 49 terms of the power series for z > -1. It
    takes 99 rectangles of the integral from -1 for -10 < z < -1, and
    -ln^2(-z)/2 - 1.6 for the others. This is up to 1 % off at the arguments of
    the inverse-Compton photon loss, 2.2 % at z = -10, and -1.6 at z = -1, where
    Li_2 = -pi^2/12.)"""
    if z < -1.0:
        var l = c_log(-z)
        return -PI * PI / 6.0 - 0.5 * l * l - _li2_series(1.0 / z)
    if z <= 0.5:
        return _li2_series(z)
    if z < 1.0:
        return PI * PI / 6.0 - c_log(z) * c_log1p(-z) - _li2_series(1.0 - z)
    if z == 1.0:
        return PI * PI / 6.0
    return nan[DType.float64]()


@inline(.always)
def number_split(x: Float64, dlnE: Float64) -> Tuple[Int, Float64]:
    """The split of a product at the position x (in bins) between the grid
    points i and i + 1 around it. The share r for point i + 1 and 1 - r for
    point i keep its number and its energy, E_i e^(f dlnE) with f the
    fraction of x. (AM3's math.cc takes r = f + f (1 - f) dlnE / 2, the weight
    of an interpolation between two points. This books the product
    f (1 - f) dlnE^2 too high in energy, up to 0.25 %: AM3 bug.)"""
    if x > 0.0 or x < 0.0:
        var i = Int(floor(x))
        var f = x - 1.0 * Float64(i)
        return (i, c_expm1(f * dlnE) / c_expm1(dlnE))
    return (0, 0.0)


def gauss_legendre(n: Int, mut x: Vec, mut w: Vec):
    x = zeros(n)
    w = zeros(n)
    for i in range(n):
        var z = c_cos(PI * (Float64(i) + 0.75) / (Float64(n) + 0.5))
        var dp = 0.0
        for _ in range(100):
            var p0 = 1.0
            var p1 = 0.0
            for k in range(1, n + 1):
                var p2 = p1
                p1 = p0
                p0 = (
                    (2.0 * Float64(k) - 1.0) * z * p1 - (Float64(k) - 1.0) * p2
                ) / Float64(k)
            dp = Float64(n) * (z * p0 - p1) / (z * z - 1.0)
            var z1 = z
            z = z1 - p0 / dp
            if abs(z - z1) < 1e-15:
                break
        x[i] = z
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp)
