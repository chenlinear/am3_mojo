"""Units and constants of the tests: the CODATA values in which the scenarios and test sources are written, and the
code's own constants, read from src/: AM3's sigma_T c and lepton-number unit (from its conversion constants), its
rounded masses and m_e c^2 = 0.511 MeV, the PDG 2026 kaon masses and mean lives of the processes am3_mojo adds, and
the grids' step and m_e c^2 (510998.2 eV)."""
from support.paths import load_derivation

_code = load_derivation('mojo_constants')

# CODATA 2018 (cgs and eV), as the scenarios and test sources are written
C_CGS = 2.99792458e10
ME_C2_EV = 510998.95
MP_C2_EV = 938272088.16
SIGMA_T = 6.6524587321e-25
ME_G = 9.1093837015e-28
MP_G = 1.67262192369e-24

_c = {n: float(v) for n, v in _code.read(('n_SSC_CGS', 'T_SSC_CGS', 'Rate_SSC_CGS')).items()}
# sigma_T c [cm^3/s] as the code's unit conversions imply (2.8e-4 above 0.665e-24 cm^2 * 3e10 cm/s)
SIGMA_T_C = 1.0 / (_c['n_SSC_CGS'] * _c['T_SSC_CGS'])
# lepton number per unit of a hadronic e+- source: AM3 converts these sources with Rate_SSC_CGS, 8.5e-4 above
# n_SSC_CGS / T_SSC_CGS (as for the e+- of muon decay): 0.99915 e- per decay
LEP_UNITS = _c['n_SSC_CGS'] / (_c['Rate_SSC_CGS'] * _c['T_SSC_CGS'])

# the code's masses [GeV] and mean lives [s] (exact decimal values, so that e.g. 0.1396e9 eV stays that float)
MASSES = _code.read(('masse', 'massp', 'massn', 'masspi0', 'masspip', 'massmu', 'massK', 'massK0', 'massN_PDG'))
TDEC = _code.read(('tdec_K', 'tdec_KS', 'tdec_KL', 'tdec_n'))
MASS_GEV = {n: float(v) for n, v in MASSES.items()}
MASS_EV = {n: float(v * 10**9) for n, v in MASSES.items()}
TDEC_S = {n: float(v) for n, v in TDEC.items()}
MEC2_AM3_EV = MASS_EV['masse']   # 0.511 MeV: AM3's m_e c^2 of its hadronic e+- sources

_g = _code.read_grid()
DLNE = float(_g['dlnE'])                                       # the step of every grid in ln E
MEC2_GRID_EV = float(_g['mec2_erg']) * float(_g['erg2eV'])     # m_e c^2 [eV] of the lepton and photon grids
