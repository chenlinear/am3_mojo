from std.ffi import external_call


@inline(.always)
def c_exp(x: Float64) -> Float64:
    return external_call["exp", Float64](x)


@inline(.always)
def c_log(x: Float64) -> Float64:
    return external_call["log", Float64](x)


@inline(.always)
def c_log10(x: Float64) -> Float64:
    return external_call["log10", Float64](x)


@inline(.always)
def c_pow(x: Float64, y: Float64) -> Float64:
    return external_call["pow", Float64](x, y)


@inline(.always)
def c_log1p(x: Float64) -> Float64:
    return external_call["log1p", Float64](x)


@inline(.always)
def c_expm1(x: Float64) -> Float64:
    return external_call["expm1", Float64](x)


@inline(.always)
def c_cos(x: Float64) -> Float64:
    return external_call["cos", Float64](x)


@inline(.always)
def c_sin(x: Float64) -> Float64:
    return external_call["sin", Float64](x)


@inline(.always)
def c_sinh(x: Float64) -> Float64:
    return external_call["sinh", Float64](x)


@inline(.always)
def c_cosh(x: Float64) -> Float64:
    return external_call["cosh", Float64](x)


@inline(.always)
def c_asin(x: Float64) -> Float64:
    return external_call["asin", Float64](x)


@inline(.always)
def fbool(b: Bool) -> Float64:
    return 1.0 if b else 0.0
