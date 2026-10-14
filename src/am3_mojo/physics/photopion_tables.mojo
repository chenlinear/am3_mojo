"""Hümmer et al. (2010/2012) tables for the photo-pion model (PhotoPion.cc)."""

from std.builtin.globals import global_constant

comptime _IT1Er: Array[Float64, 28] = [
    0.34,
    0.64,
    0.64,
    0.64,
    0.75,
    0.75,
    0.75,
    0.75,
    0.77,
    1.03,
    1.03,
    1.03,
    1.04,
    1.04,
    1.04,
    1.04,
    1.05,
    1.05,
    1.05,
    1.05,
    1.45,
    1.45,
    1.45,
    1.45,
    1.56,
    1.56,
    1.56,
    1.56,
]

comptime _IT1Bout: Array[Float64, 28] = [
    1.00,
    0.67,
    0.33,
    0.33,
    0.52,
    0.42,
    0.42,
    0.06,
    0.45,
    0.75,
    0.14,
    0.14,
    0.64,
    0.22,
    0.22,
    0.14,
    0.14,
    0.55,
    0.55,
    0.31,
    0.14,
    0.13,
    0.13,
    0.73,
    0.37,
    0.39,
    0.39,
    0.24,
]

comptime _IT1G: Array[Float64, 28] = [
    66.69,
    4.260,
    4.260,
    4.260,
    27.01,
    27.01,
    27.01,
    27.01,
    6.590,
    3.190,
    3.190,
    3.190,
    15.72,
    15.72,
    15.72,
    15.72,
    21.68,
    21.68,
    21.68,
    21.68,
    3.030,
    3.030,
    3.030,
    3.030,
    16.46,
    16.46,
    16.46,
    16.46,
]

comptime _IT1kai: Array[Float64, 28] = [
    0.22,
    0.29,
    0.14,
    0.19,
    0.31,
    0.17,
    0.18,
    0.22,
    0.32,
    0.35,
    0.23,
    0.17,
    0.35,
    0.24,
    0.17,
    0.23,
    0.35,
    0.24,
    0.16,
    0.23,
    0.38,
    0.29,
    0.15,
    0.23,
    0.39,
    0.30,
    0.15,
    0.23,
]

comptime _IT1kaiPN: Array[Float64, 28] = [
    0.78,
    0.71,
    1.00,
    0.67,
    0.69,
    1.00,
    0.65,
    0.56,
    0.68,
    0.65,
    1.00,
    0.60,
    0.65,
    1.00,
    0.59,
    0.54,
    0.65,
    1.00,
    0.60,
    0.54,
    0.62,
    1.00,
    0.56,
    0.54,
    0.61,
    1.00,
    0.55,
    0.54,
]

comptime _IT1MpPi0: Array[Float64, 28] = [
    0.667,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.667,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.667,
    0.667,
    0.067,
    0.400,
    0.333,
    0.667,
    0.067,
    0.400,
    0.333,
    0.667,
    0.067,
    0.400,
    0.333,
]

comptime _IT1MpPip: Array[Float64, 28] = [
    0.333,
    0.667,
    0.167,
    0.611,
    0.667,
    0.167,
    0.611,
    1.000,
    0.667,
    0.667,
    0.167,
    0.611,
    0.667,
    0.167,
    0.611,
    1.000,
    0.333,
    0.533,
    0.422,
    1.000,
    0.333,
    0.533,
    0.422,
    1.000,
    0.333,
    0.533,
    0.422,
    1.000,
]

comptime _IT1MpPim: Array[Float64, 28] = [
    0.000,
    0.000,
    0.500,
    0.056,
    0.000,
    0.500,
    0.056,
    0.333,
    0.000,
    0.000,
    0.500,
    0.056,
    0.000,
    0.500,
    0.056,
    0.333,
    0.000,
    0.400,
    0.178,
    0.667,
    0.000,
    0.400,
    0.178,
    0.667,
    0.000,
    0.400,
    0.178,
    0.667,
]

comptime _IT1MnPi0: Array[Float64, 28] = [
    0.667,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.667,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.333,
    0.667,
    0.667,
    0.067,
    0.400,
    0.333,
    0.667,
    0.067,
    0.400,
    0.333,
    0.667,
    0.067,
    0.400,
    0.333,
]

comptime _IT1MnPip: Array[Float64, 28] = [
    0.000,
    0.000,
    0.500,
    0.056,
    0.000,
    0.500,
    0.056,
    0.333,
    0.000,
    0.000,
    0.500,
    0.056,
    0.000,
    0.500,
    0.056,
    0.333,
    0.000,
    0.400,
    0.178,
    0.667,
    0.000,
    0.400,
    0.178,
    0.667,
    0.000,
    0.400,
    0.178,
    0.667,
]

comptime _IT1MnPim: Array[Float64, 28] = [
    0.333,
    0.667,
    0.167,
    0.611,
    0.667,
    0.167,
    0.611,
    1.000,
    0.667,
    0.667,
    0.167,
    0.611,
    0.667,
    0.167,
    0.611,
    1.000,
    0.333,
    0.533,
    0.422,
    1.000,
    0.333,
    0.533,
    0.422,
    1.000,
    0.333,
    0.533,
    0.422,
    1.000,
]

comptime _IT1MpPRO: Array[Float64, 28] = [
    0.667,
    0.333,
    0.000,
    0.778,
    0.333,
    0.000,
    0.778,
    0.333,
    0.333,
    0.333,
    0.000,
    0.778,
    0.333,
    0.000,
    0.778,
    0.333,
    0.667,
    0.000,
    0.622,
    0.667,
    0.667,
    0.000,
    0.622,
    0.667,
    0.667,
    0.000,
    0.622,
    0.667,
]

comptime _IT1MpNTR: Array[Float64, 28] = [
    0.333,
    0.667,
    0.000,
    0.222,
    0.667,
    0.000,
    0.222,
    0.667,
    0.667,
    0.667,
    0.000,
    0.222,
    0.667,
    0.000,
    0.222,
    0.667,
    0.333,
    0.000,
    0.378,
    0.333,
    0.333,
    0.000,
    0.378,
    0.333,
    0.333,
    0.000,
    0.378,
    0.333,
]

comptime _IT1MnPRO: Array[Float64, 28] = [
    0.333,
    0.667,
    0.000,
    0.222,
    0.667,
    0.000,
    0.222,
    0.667,
    0.667,
    0.667,
    0.000,
    0.222,
    0.667,
    0.000,
    0.222,
    0.667,
    0.333,
    0.000,
    0.378,
    0.333,
    0.333,
    0.000,
    0.378,
    0.333,
    0.333,
    0.000,
    0.378,
    0.333,
]

comptime _IT1MnNTR: Array[Float64, 28] = [
    0.667,
    0.333,
    0.000,
    0.778,
    0.333,
    0.000,
    0.778,
    0.333,
    0.333,
    0.333,
    0.000,
    0.778,
    0.333,
    0.000,
    0.778,
    0.333,
    0.667,
    0.000,
    0.622,
    0.667,
    0.667,
    0.000,
    0.622,
    0.667,
    0.667,
    0.000,
    0.622,
    0.667,
]

comptime _IT2Em: Array[Float64, 9] = [
    0.17,
    0.56,
    10.0,
    0.4,
    1.58,
    10.0,
    0.4,
    1.58,
    10.0,
]

comptime _IT2Ex: Array[Float64, 9] = [
    0.56,
    10.0,
    1.0e6,
    1.58,
    10.0,
    1.0e6,
    1.58,
    10.0,
    1.0e6,
]

comptime _IT2kai: Array[Float64, 9] = [
    0.13,
    0.05,
    0.001,
    0.08,
    0.02,
    0.001,
    0.2,
    0.2,
    0.2,
]

comptime _IT2kaiPN: Array[Float64, 9] = [
    0.87,
    0.95,
    0.999,
    0.72,
    0.78,
    0.799,
    1.00,
    1.00,
    1.00,
]

comptime _IT2MpPi0: Array[Float64, 9] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1667,
    0.1667,
    0.1667,
]

comptime _IT2MpPip: Array[Float64, 9] = [
    1.0,
    1.0,
    1.0,
    0.25,
    0.25,
    0.25,
    0.75,
    0.75,
    0.75,
]

comptime _IT2MpPim: Array[Float64, 9] = [
    0.0,
    0.0,
    0.0,
    0.75,
    0.75,
    0.75,
    0.0833,
    0.0833,
    0.0833,
]

comptime _IT2MnPi0: Array[Float64, 9] = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1667,
    0.1667,
    0.1667,
]

comptime _IT2MnPip: Array[Float64, 9] = [
    0.0,
    0.0,
    0.0,
    0.75,
    0.75,
    0.75,
    0.0833,
    0.0833,
    0.0833,
]

comptime _IT2MnPim: Array[Float64, 9] = [
    1.0,
    1.0,
    1.0,
    0.25,
    0.25,
    0.25,
    0.75,
    0.75,
    0.75,
]

comptime _IT2MpPRO: Array[Float64, 9] = [
    0.000,
    0.000,
    0.000,
    0.833,
    0.833,
    0.833,
    0.0,
    0.0,
    0.0,
]

comptime _IT2MpNTR: Array[Float64, 9] = [
    1.000,
    1.000,
    1.000,
    0.167,
    0.167,
    0.167,
    0.0,
    0.0,
    0.0,
]

comptime _IT2MnPRO: Array[Float64, 9] = [
    1.000,
    1.000,
    1.000,
    0.167,
    0.167,
    0.167,
    0.0,
    0.0,
    0.0,
]

comptime _IT2MnNTR: Array[Float64, 9] = [
    0.000,
    0.000,
    0.000,
    0.833,
    0.833,
    0.833,
    0.0,
    0.0,
    0.0,
]

comptime _IT3sig: Array[Float64, 14] = [
    60.0,
    60.0,
    85.0,
    85.0,
    120.0,
    120.0,
    120.0,
    120.0,
    120.0,
    120.0,
    120.0,
    120.0,
    120.0,
    120.0,
]

comptime _IT3Em: Array[Float64, 14] = [
    0.5,
    0.5,
    0.9,
    0.9,
    1.5,
    1.5,
    5.0,
    5.0,
    50.0,
    50.0,
    500.0,
    500.0,
    5000.0,
    5000.0,
]

comptime _IT3Ex: Array[Float64, 14] = [
    0.9,
    0.9,
    1.5,
    1.5,
    5.0,
    5.0,
    50.0,
    50.0,
    500.0,
    500.0,
    5000.0,
    5000.0,
    5.0e6,
    5.0e6,
]

comptime _IT3kai: Array[Float64, 14] = [
    0.1,
    0.4,
    0.15,
    0.35,
    0.15,
    0.35,
    0.07,
    0.35,
    0.02,
    0.5,
    0.007,
    0.5,
    0.002,
    0.6,
]

comptime _IT3kaiPN: Array[Float64, 14] = [
    0.73,
    1.00,
    0.66,
    1.00,
    0.61,
    1.00,
    0.51,
    1.00,
    0.55,
    1.00,
    0.56,
    1.00,
    0.56,
    1.00,
]

comptime _IT3MpPi0: Array[Float64, 14] = [
    0.32,
    0.17,
    0.42,
    0.19,
    0.59,
    0.16,
    1.38,
    0.16,
    3.01,
    0.20,
    5.13,
    0.27,
    7.59,
    0.26,
]

comptime _IT3MpPip: Array[Float64, 14] = [
    0.34,
    0.29,
    0.31,
    0.35,
    0.57,
    0.21,
    1.37,
    0.25,
    2.86,
    0.21,
    4.68,
    0.29,
    6.80,
    0.27,
]

comptime _IT3MpPim: Array[Float64, 14] = [
    0.04,
    0.05,
    0.07,
    0.08,
    0.30,
    0.13,
    1.11,
    0.23,
    2.64,
    0.14,
    4.57,
    0.12,
    6.65,
    0.13,
]

comptime _IT3MnPi0: Array[Float64, 14] = [
    0.32,
    0.17,
    0.42,
    0.19,
    0.59,
    0.16,
    1.38,
    0.16,
    3.01,
    0.20,
    5.13,
    0.27,
    7.59,
    0.26,
]

comptime _IT3MnPip: Array[Float64, 14] = [
    0.04,
    0.05,
    0.07,
    0.08,
    0.30,
    0.13,
    1.11,
    0.23,
    2.64,
    0.14,
    4.57,
    0.12,
    6.65,
    0.13,
]

comptime _IT3MnPim: Array[Float64, 14] = [
    0.34,
    0.29,
    0.31,
    0.35,
    0.57,
    0.21,
    1.37,
    0.25,
    2.86,
    0.21,
    4.68,
    0.29,
    6.80,
    0.27,
]

comptime _IT3MpPRO: Array[Float64, 14] = [
    0.46,
    0.00,
    0.49,
    0.00,
    0.65,
    0.00,
    0.72,
    0.00,
    0.71,
    0.00,
    0.72,
    0.00,
    0.71,
    0.00,
]

comptime _IT3MpNTR: Array[Float64, 14] = [
    0.54,
    0.00,
    0.51,
    0.00,
    0.35,
    0.00,
    0.28,
    0.00,
    0.29,
    0.00,
    0.28,
    0.00,
    0.29,
    0.00,
]

comptime _IT3MnPRO: Array[Float64, 14] = [
    0.54,
    0.00,
    0.51,
    0.00,
    0.35,
    0.00,
    0.28,
    0.00,
    0.29,
    0.00,
    0.28,
    0.00,
    0.29,
    0.00,
]

comptime _IT3MnNTR: Array[Float64, 14] = [
    0.46,
    0.00,
    0.49,
    0.00,
    0.65,
    0.00,
    0.72,
    0.00,
    0.71,
    0.00,
    0.72,
    0.00,
    0.71,
    0.00,
]


@always_inline
def IT1Er(i: Int) -> Float64:
    return global_constant[_IT1Er]()[i]


@always_inline
def IT1Bout(i: Int) -> Float64:
    return global_constant[_IT1Bout]()[i]


@always_inline
def IT1G(i: Int) -> Float64:
    return global_constant[_IT1G]()[i]


@always_inline
def IT1kai(i: Int) -> Float64:
    return global_constant[_IT1kai]()[i]


@always_inline
def IT1kaiPN(i: Int) -> Float64:
    return global_constant[_IT1kaiPN]()[i]


@always_inline
def IT1MpPi0(i: Int) -> Float64:
    return global_constant[_IT1MpPi0]()[i]


@always_inline
def IT1MpPip(i: Int) -> Float64:
    return global_constant[_IT1MpPip]()[i]


@always_inline
def IT1MpPim(i: Int) -> Float64:
    return global_constant[_IT1MpPim]()[i]


@always_inline
def IT1MnPi0(i: Int) -> Float64:
    return global_constant[_IT1MnPi0]()[i]


@always_inline
def IT1MnPip(i: Int) -> Float64:
    return global_constant[_IT1MnPip]()[i]


@always_inline
def IT1MnPim(i: Int) -> Float64:
    return global_constant[_IT1MnPim]()[i]


@always_inline
def IT1MpPRO(i: Int) -> Float64:
    return global_constant[_IT1MpPRO]()[i]


@always_inline
def IT1MpNTR(i: Int) -> Float64:
    return global_constant[_IT1MpNTR]()[i]


@always_inline
def IT1MnPRO(i: Int) -> Float64:
    return global_constant[_IT1MnPRO]()[i]


@always_inline
def IT1MnNTR(i: Int) -> Float64:
    return global_constant[_IT1MnNTR]()[i]


@always_inline
def IT2Em(i: Int) -> Float64:
    return global_constant[_IT2Em]()[i]


@always_inline
def IT2Ex(i: Int) -> Float64:
    return global_constant[_IT2Ex]()[i]


@always_inline
def IT2kai(i: Int) -> Float64:
    return global_constant[_IT2kai]()[i]


@always_inline
def IT2kaiPN(i: Int) -> Float64:
    return global_constant[_IT2kaiPN]()[i]


@always_inline
def IT2MpPi0(i: Int) -> Float64:
    return global_constant[_IT2MpPi0]()[i]


@always_inline
def IT2MpPip(i: Int) -> Float64:
    return global_constant[_IT2MpPip]()[i]


@always_inline
def IT2MpPim(i: Int) -> Float64:
    return global_constant[_IT2MpPim]()[i]


@always_inline
def IT2MnPi0(i: Int) -> Float64:
    return global_constant[_IT2MnPi0]()[i]


@always_inline
def IT2MnPip(i: Int) -> Float64:
    return global_constant[_IT2MnPip]()[i]


@always_inline
def IT2MnPim(i: Int) -> Float64:
    return global_constant[_IT2MnPim]()[i]


@always_inline
def IT2MpPRO(i: Int) -> Float64:
    return global_constant[_IT2MpPRO]()[i]


@always_inline
def IT2MpNTR(i: Int) -> Float64:
    return global_constant[_IT2MpNTR]()[i]


@always_inline
def IT2MnPRO(i: Int) -> Float64:
    return global_constant[_IT2MnPRO]()[i]


@always_inline
def IT2MnNTR(i: Int) -> Float64:
    return global_constant[_IT2MnNTR]()[i]


@always_inline
def IT3sig(i: Int) -> Float64:
    return global_constant[_IT3sig]()[i]


@always_inline
def IT3Em(i: Int) -> Float64:
    return global_constant[_IT3Em]()[i]


@always_inline
def IT3Ex(i: Int) -> Float64:
    return global_constant[_IT3Ex]()[i]


@always_inline
def IT3kai(i: Int) -> Float64:
    return global_constant[_IT3kai]()[i]


@always_inline
def IT3kaiPN(i: Int) -> Float64:
    return global_constant[_IT3kaiPN]()[i]


@always_inline
def IT3MpPi0(i: Int) -> Float64:
    return global_constant[_IT3MpPi0]()[i]


@always_inline
def IT3MpPip(i: Int) -> Float64:
    return global_constant[_IT3MpPip]()[i]


@always_inline
def IT3MpPim(i: Int) -> Float64:
    return global_constant[_IT3MpPim]()[i]


@always_inline
def IT3MnPi0(i: Int) -> Float64:
    return global_constant[_IT3MnPi0]()[i]


@always_inline
def IT3MnPip(i: Int) -> Float64:
    return global_constant[_IT3MnPip]()[i]


@always_inline
def IT3MnPim(i: Int) -> Float64:
    return global_constant[_IT3MnPim]()[i]


@always_inline
def IT3MpPRO(i: Int) -> Float64:
    return global_constant[_IT3MpPRO]()[i]


@always_inline
def IT3MpNTR(i: Int) -> Float64:
    return global_constant[_IT3MpNTR]()[i]


@always_inline
def IT3MnPRO(i: Int) -> Float64:
    return global_constant[_IT3MnPRO]()[i]


@always_inline
def IT3MnNTR(i: Int) -> Float64:
    return global_constant[_IT3MnNTR]()[i]
