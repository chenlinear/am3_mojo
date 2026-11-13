"""The CODATA 2018 values (cgs and eV) in which the scenarios and test sources
are written. They are in a separate module because support.units reads the
code's constants with sympy. A benchmark process (time_scenario.py) must not
load sympy: its peak memory is that of the simulation."""

C_CGS = 2.99792458e10
ME_C2_EV = 510998.95
MP_C2_EV = 938272088.16
SIGMA_T = 6.6524587321e-25
ME_G = 9.1093837015e-28
MP_G = 1.67262192369e-24
