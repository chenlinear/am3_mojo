"""Special functions used by AM3 (port of AM3/math.cc and the parts of GSL
that AM3 calls: `gsl_sf_bessel_Knu`)."""

from std.builtin.globals import global_constant
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *
from am3_mojo.utils.buffers import *

comptime GSL_DBL_EPSILON: Float64 = 2.2204460492503131e-16
comptime GSL_SQRT_DBL_MAX: Float64 = 1.3407807929942596e154
comptime GSL_SQRT_DBL_MIN: Float64 = 1.4916681462400413e-154
comptime GSL_LOG_DBL_MAX: Float64 = 7.0978271289338397e02
comptime GSL_LOG_DBL_MIN: Float64 = -7.0839641853226408e02
comptime M_PI: Float64 = 3.14159265358979323846264338328
comptime M_LN10: Float64 = 2.30258509299404568401799145468


# ----------------------------------------------------------------------------
# GSL: Chebyshev series for the Temme gamma functions (specfunc/bessel_temme.c)
# ----------------------------------------------------------------------------
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


@always_inline
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


@always_inline
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
    """GSL gsl_sf_bessel_K_scaled_temme: returns (K_nu, K_nup1) scaled by e^x.
    """
    comptime max_iter = 15000
    var half_x = 0.5 * x
    var ln_half_x = c_log(half_x)
    var half_x_nu = c_exp(nu * ln_half_x)
    var pi_nu = M_PI * nu
    var sigma = -nu * ln_half_x
    var sinrat = 1.0 if c_fabs(pi_nu) < GSL_DBL_EPSILON else pi_nu / c_sin(
        pi_nu
    )
    var sinhrat = (
        1.0 if c_fabs(sigma) < GSL_DBL_EPSILON else c_sinh(sigma) / sigma
    )
    var ex = c_exp(x)

    # gsl_sf_temme_gamma
    var anu = c_fabs(nu)
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
        if c_fabs(del0) < 0.5 * c_fabs(sum0) * GSL_DBL_EPSILON:
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
        if c_fabs(dels / s) < GSL_DBL_EPSILON:
            break
        i += 1
    hi *= -a1
    var K_nu = c_sqrt(M_PI / (2.0 * x)) / s
    var K_nup1 = K_nu * (nu + x + 0.5 - hi) / x
    return (K_nu, K_nup1)


def _exp_mult(x: Float64, y: Float64) -> Float64:
    """gsl_sf_exp_mult_err_e of GSL, value only (underflow 0, overflow MAX)."""
    var ay = c_fabs(y)
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
            return Float64.MAX  # GSL would signal an overflow error
        elif lnr < GSL_LOG_DBL_MIN + 0.01:
            return 0.0  # GSL would signal an underflow error
        else:
            var sy = 1.0 if y >= 0.0 else -1.0
            var M = c_floor(x)
            var N = c_floor(ly)
            var a = x - M
            var b = ly - N
            var eMN = c_exp(M + N)
            var eab = c_exp(a + b)
            return sy * eMN * eab


def gsl_bessel_Knu(nu: Float64, x: Float64) -> Float64:
    """Port of gsl_sf_bessel_Knu(nu, x) for x > 0, nu >= 0."""
    if x <= 0.0 or nu < 0.0:
        return Float64.MAX  # GSL domain error -> NaN; not reachable in AM3
    var N = to_int(nu + 0.5)
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
        if c_fabs(K_nu) > GSL_SQRT_DBL_MAX:
            var p = c_floor(c_log(c_fabs(K_nu)) / M_LN10)
            var factor = c_pow(10.0, p)
            K_num1 /= factor
            K_nu /= factor
            e10 += to_int(p)
        K_nup1 = 2.0 * (mu + Float64(n) + 1.0) / x * K_nu + K_num1
    var val = K_nu
    # gsl_sf_result_smash_e
    if e10 != 0:
        var av = c_fabs(val)
        if (
            GSL_SQRT_DBL_MIN < av
            and av < GSL_SQRT_DBL_MAX
            and 0.49 * GSL_LOG_DBL_MIN < Float64(e10)
            and Float64(e10) < 0.49 * GSL_LOG_DBL_MAX
        ):
            val = val * c_exp(Float64(e10) * M_LN10)
        else:
            val = _exp_mult(Float64(e10) * M_LN10, val)
    return _exp_mult(-x, val)


@always_inline
def beta(g: Float64, REG: Float64) -> Float64:
    if g < 10.0:
        return c_sqrt(1.0 - (1.0 / (g * g)) + REG)
    # second-order expansion of sqrt(1 - 1/g^2) for large g
    var ig2 = 1.0 / (g * g)
    return 1.0 - 0.5 * ig2 * (1.0 + 0.25 * ig2)


@always_inline
def signf(a: Int) -> Int:
    if c_mod(a, 2) == 0:
        return 1
    else:
        return -1


def bessel_k13(x: Float64) -> Float64:
    if x > 700.0:
        return 0.0
    return gsl_bessel_Knu(1.0 / 3.0, x)


def bessel_k43(x: Float64) -> Float64:
    if x > 700.0:
        return 0.0
    return gsl_bessel_Knu(4.0 / 3.0, x)


def int_bk53l(x: Float64) -> Float64:
    return (1.253314 + 0.139338 / x) / (c_exp(x) * c_sqrt(x))


def int_bk53m(x: Float64) -> Float64:
    """Integral of K_5/3 from x to 10 as a 100-step left Riemann sum in
    ln t, plus the tail beyond t = 10."""
    comptime N = 100
    var yi = c_log(x)
    var dy = (c_log(10.0) - yi) / Float64(N)
    var result = 0.0
    for i in range(N):
        var t = c_exp(yi + Float64(i) * dy)
        result += t * gsl_bessel_Knu(5.0 / 3.0, t)
    return result * dy + 0.0000192


def int_bk53s(x: Float64) -> Float64:
    return (
        2.14953 * c_pow(x, -0.66666667)
        - 0.142393 * c_pow(x, 2.66666667)
        - 1.814
    )


def int_bessel_k53(x: Float64) -> Float64:
    if x > 10.0:
        return int_bk53l(x)
    elif x <= 10.0 and x > 0.001:
        return int_bk53m(x)
    else:
        return int_bk53s(x)


def polylog(z_in: Float64) -> Float64:
    """AM3 math.cc polylog: returns Li_2(-z)."""
    var z = -z_in
    var z1 = z - 1.0
    if z1 < 0.0:
        var a = 0.0
        for n in range(1, 50):
            a += Float64(signf(n)) * c_pow(z, Float64(n)) / Float64(n * n)
        return a
    elif z1 > 0.0 and z < 10.0:
        var b = mathpi * mathpi / 12.0
        var z2 = 0.01 * z1
        for m in range(1, 100):
            var mf = Float64(m)
            b += z2 * c_log(2.0 + mf * z2) / (1.0 + mf * z2)
        return -b
    else:
        return -0.5 * c_log(z) * c_log(z) - 1.6


@always_inline
def number_split(x: Float64, dlnE: Float64) -> Tuple[Int, Float64]:
    """Split x into integer part i and weight r (AM3 math.cc number_split)."""
    if x > 0.0 or x < 0.0:
        var i = to_int(c_floor(x))
        var r = x - 1.0 * Float64(i)
        r += 0.5 * r * (1.0 - r) * dlnE
        return (i, r)
    return (0, 0.0)


def gauss_legendre(n: Int, mut x: Vec, mut w: Vec):
    """Nodes x and weights w of the n-point Gauss-Legendre rule on [-1, 1]."""
    x = zeros(n)
    w = zeros(n)
    for i in range(n):
        var z = c_cos(mathpi * (Float64(i) + 0.75) / (Float64(n) + 0.5))
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
            if c_fabs(z - z1) < 1e-15:
                break
        x[i] = z
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp)
