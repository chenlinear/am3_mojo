"""Photo-hadronic kaons, README "Kaons in multi-pion production": the kaons of the multi-pion type M4 (5-50 GeV),
in two steps (5-15 and 15-50 GeV), in number and energy, against the strange particles measured at 20 GeV by the
SLAC Hybrid Facility Photon Collaboration (Abe et al. 1985, Phys. Rev. D 32, 2869: cross sections; Abe et al. 1984,
Phys. Rev. D 29, 1877: K0_S distributions). The values are in tests/support/measured/strange_20gev.py, checked here
against the HEPData records ins213727 and ins190724 where local copies exist.

  * Numbers. K0_S, Lambda, anti-Lambda, Sigma0 and Xi are measured; the four kaon species follow from strangeness
    balance. The charged Sigma are taken as Sigma+ + Sigma- = 2 Sigma0. This holds if the three are made alike, for
    production linear in I3, and for a strange quark replacing one of the proton's, which needs the Sigma0 to be a
    quarter of the Lambda and Sigma0 (as in SU(6); measured: 1.65 of 5.60 mubarn). The kaon that comes with a
    Lambda is a K0 in 0.40 of the events, from the K0_S Lambda pairs (Table 2 of the preprint SLAC-PUB-3634). The
    other shares of K0 and anti-K0 are ABBHHM's (multibody_strangeness.py). The K0_S K0_S pairs are a check, a
    quarter of the K0 anti-K0 pairs if these are made without coherence. Varied, alone and together: the Sigma0
    within its error, the charged Sigma, the Xi0, extra K anti-K pairs with the Lambda, and ABBHHM's shares in the
    pairs.
    Each species is then linear in ln E between the code's kaons at 2.5-5 GeV (the upper step of M3,
    multibody_strangeness.py, with the two-body kaons there) and its value at 20 GeV. A step has the value at its
    middle (in ln E), its mean for a photon field n(eps) ~ 1/eps. The line reaches the kaons of M5 (50-500 GeV)
    within 18 % per species (6-11 % for the kaons, the antikaons and the K0_S). It is also checked against the
    energy dependence of K0_S production in pi- p collisions (Abe et al. 1984, Fig. 3, with the values of Abe et
    al. 1985, Table 3).
  * Energy. The mean fraction chi of the nucleon energy that a K0_S takes (in the frame where the nucleon is fast):
    at 20 GeV from the distributions in x and pT^2 (Tables III and VI; checked with that in z, Table V), and at the
    middle of each step with the same invariant distribution (Feynman scaling). A step's kaons have a leading part
    (at the chi of the type's leading pions, 0.35); the others are at f x the chi of its low-energy pions (0.07),
    and f gives the K0_S this mean. The higher types, with f = 1.6, against the same scaling.
  * The alternative, one step for all of M4 with the values measured at 20 GeV (Results, "One step or two for M4"):
    its numbers and energy against the line and the distribution. multibody_strangeness.py compares it with the
    bubble-chamber data.

The code's tables are read from src/ (code_tables.py)."""
import csv, glob, io, os, sys, tarfile

import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from code_tables import photopion_tables, multipion_kaons, two_body_steps  # noqa: E402
from support.measured import strange_20gev as g20  # noqa: E402
from support.paths import ROOT, load_derivation  # noqa: E402

derivation = load_derivation('kaon_production')   # the README's s, neutral kaons, masses

T = photopion_tables()
SIG3, KAI = T['IT3sig'], T['IT3kai']
MPI = multipion_kaons()   # [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+, h-, f]
NAMES = ('K+', 'K0', 'K-', 'anti-K0')
K0S = derivation.numeric_neutral()   # the K0_S among K0 and anti-K0 (the README: half of each)
_, M3_UPPER = sorted((r for r in MPI if int(r[0]) == 2), key=lambda r: r[1])   # the code's two steps of M3
M4_LOWER, M4_UPPER = sorted((r for r in MPI if int(r[0]) == 3), key=lambda r: r[1])   # and of M4
M4 = (M4_LOWER[1], M4_UPPER[2])
F_HIGHER, = {r[9] for r in MPI if int(r[0]) >= 4}   # the factor on the chi of the low-energy pions of M5-M7
fmt = lambda v, d=2: ', '.join(f'{x:.{d}f}' for x in v)

# the code's kaons for protons [mub]: (eps_min, eps_max, (K+, K0, K-, anti-K0))
channels = [(r[0], r[1], np.array([r[6], r[7], 0.0, 0.0])) for r in two_body_steps()]
channels += [(row[1], row[2], SIG3[2 * int(row[0])] * np.array(row[3:7])) for row in MPI]

# the values of tests/support/measured/strange_20gev.py against the HEPData records, where local copies are present
HEPDATA = os.path.join(ROOT, 'knowledge', 'HEPData')


def hepdata_tables(prefix):
    """{table name: list of (reaction, rows of floats)} from the CSV archive of a HEPData record, or None."""
    files = glob.glob(os.path.join(HEPDATA, prefix + '*_csv.tgz'))
    if not files:
        return None
    out = {}
    with tarfile.open(files[0]) as tar:
        for m in tar.getmembers():
            if not m.name.endswith('.csv'):
                continue
            blocks, reaction, rows = [], None, None
            for line in io.TextIOWrapper(tar.extractfile(m), encoding='utf-8'):
                line = line.strip()
                if line.startswith('#: RE,'):
                    reaction, rows = line.split(',')[-1], []
                    blocks.append((reaction, rows))
                elif line and not line.startswith('#') and rows is not None and not line.startswith('"'):
                    vals = next(csv.reader([line]))
                    rows.append([float(v) if v.replace('.', '', 1).replace('-', '', 1).replace('e', '', 1).isdigit() else v for v in vals])
            out[os.path.basename(m.name)[:-4]] = blocks
    return out


print('The values of tests/support/measured/strange_20gev.py against the HEPData records:')
rec84, rec85 = hepdata_tables('HEPData-ins190724'), hepdata_tables('HEPData-ins213727')
if rec84 is None or rec85 is None:
    print('  (not checked: knowledge/HEPData does not have the records ins190724 and ins213727)')
else:
    def same(name, table, values, edges=None):
        rows = rec84[table][0][1]
        got = np.array([r[3] for r in rows])
        ok = np.allclose(got, values, rtol=1e-9, atol=1e-12)
        if edges is not None:
            ok = ok and np.allclose([r[1] for r in rows] + [rows[-1][2]], edges)
        print(f'  Abe et al. 1984, {name}: {"the same" if ok else "DIFFERENT"} ({len(rows)} values)')
        return ok
    ok = same('K0_S in x (Table III; HEPData Table 1)', 'Table1', g20.F_K0S, g20.X_EDGES)
    ok &= same('K0_S in z (Table V; HEPData Table 4)', 'Table4', g20.DNDZ_K0S, g20.Z_EDGES)
    ok &= same('K0_S in pT^2 (Table VI; HEPData Table 7)', 'Table7', g20.DSDPT2_K0S, g20.PT2_EDGES)
    want = {'GAMMA P --> KS X': g20.K0S, 'GAMMA P --> LAMBDA X': g20.LAMBDA, 'GAMMA P --> LAMBDABAR X': g20.ANTI_LAMBDA,
            'GAMMA P --> XI- X': g20.XI_MINUS, 'GAMMA P --> XIBAR+ X': g20.ANTI_XI_PLUS, 'GAMMA P --> SIGMA0 X': g20.SIGMA0,
            'GAMMA P --> SIGMA(1385P13)+ X': g20.SIGMA1385_PLUS, 'GAMMA P --> SIGMA(1385P13)- X': g20.SIGMA1385_MINUS}
    got = {r: rows[0] for r, rows in rec85['Table1']}
    ok85 = all(abs(got[r][3] - v) < 1e-9 and abs(got[r][4] - e) < 1e-9 for r, (v, e) in want.items())
    ok85 &= got['GAMMA P --> OMEGA- X'][3] == '<%g' % g20.OMEGA_MINUS_LIMIT
    print(f'  Abe et al. 1985, the {len(got)} cross sections (HEPData Table 1): {"the same" if ok85 else "DIFFERENT"}')
    assert ok and ok85

print('\nMeasured at 20 GeV [mub] (Abe et al. 1985; of %g +- %g):' % g20.SIGMA_TOT)
print('  K0_S %.3f +- %.3f, Lambda %.3f +- %.3f, anti-Lambda %.3f +- %.3f, Sigma0 %.2f +- %.2f, Xi- %.3f +- %.3f, anti-Xi+ %.3f +- %.3f,'
      ' Sigma*+(1385) %.2f +- %.2f, Sigma*-(1385) %.2f +- %.2f, Omega- below %.3f'
      % (g20.K0S + g20.LAMBDA + g20.ANTI_LAMBDA + g20.SIGMA0 + g20.XI_MINUS + g20.ANTI_XI_PLUS + g20.SIGMA1385_PLUS + g20.SIGMA1385_MINUS
         + (g20.OMEGA_MINUS_LIMIT,)))
print('  (from an earlier third of the events, Abe et al. 1984: K0_S %.3f +- %.3f, Lambda %.3f +- %.3f, anti-Lambda %.3f +- %.3f)'
      % (g20.K0S_1984 + g20.LAMBDA_1984 + g20.ANTI_LAMBDA_1984))
_, _, w84 = g20.chi_samples()
print(f'  the distributions of 1984 against the inclusive cross section of their sample: K0_S {w84.sum():.2f} mub in pT^2'
      f' ({w84.sum() / g20.K0S_1984[0] - 1:+.1%}), {g20.mean_chi_from_z()[1] * g20.SIGMA_TOT[0]:.2f} mub in z'
      f' ({g20.mean_chi_from_z()[1] * g20.SIGMA_TOT[0] / g20.K0S_1984[0] - 1:+.1%})')
sp = g20.species()
hyperons = g20.hyperon_strangeness()
print(f'  the balance of strangeness: hyperons less antihyperons {hyperons:.2f} mub (charged Sigma {g20.SIGMA_PM:g} x Sigma0, Xi0 = Xi-;'
      f' Lambda or Sigma0 then {g20.lambda_share():.2f} of the hyperons, against {g20.LAMBDA_SHARE_ABBHHM} in ABBHHM\'s events);')
print(f'  pairs of neutral strange particles in one event [mub] (Abe et al. 1985, Table 2 of the preprint): K0_S K0_S {g20.K0S_K0S[0]:.3f} +- {g20.K0S_K0S[1]:.3f},'
      f' K0_S Lambda {g20.K0S_LAMBDA[0]:.3f} +- {g20.K0S_LAMBDA[1]:.3f}, K0_S anti-Lambda {g20.K0S_ANTI_LAMBDA[0]:.3f} +- {g20.K0S_ANTI_LAMBDA[1]:.3f},'
      f' Lambda anti-Lambda {g20.LAMBDA_ANTI_LAMBDA[0]:.3f} +- {g20.LAMBDA_ANTI_LAMBDA[1]:.3f}, Lambda Lambda {g20.LAMBDA_LAMBDA[0]:.3f} +- {g20.LAMBDA_LAMBDA[1]:.3f}')
a_lam = g20.k0_share_with_lambda()
a_lam_err = a_lam * g20.K0S_LAMBDA[1] / g20.K0S_LAMBDA[0]
print(f'    a K0_S comes with {g20.K0S_LAMBDA[0] / g20.LAMBDA[0]:.3f} of the Lambda: the K0 among the kaons that come with a Lambda {a_lam:.2f} +- {a_lam_err:.2f}'
      f' (ABBHHM: {g20.K0_SHARE_LAMBDA} in its events with further particles), {g20.k0_share_with_lambda(0.1):.2f} with 0.1 extra K anti-K pairs per Lambda;'
      f' with the charged Sigma as in ABBHHM\'s events ({g20.K0_SHARE_SIGMA_PLUS}, {g20.K0_SHARE_SIGMA_MINUS}), {g20.k0_share_with_hyperons():.2f} among the kaons that come with a hyperon')
print(f'    with K0 {g20.K0_SHARE} of the kaons of K anti-K pairs and anti-K0 {g20.ANTI_K0_SHARE} of the antikaons (ABBHHM): K+, K0, K-, anti-K0 = {fmt(sp)};'
      f' kaons / antikaons = {(sp[0] + sp[1]) / (sp[2] + sp[3]):.2f} (measured: K*+(890) / K*-(890) = {g20.kstar_ratio()[0]:.2f} +- {g20.kstar_ratio()[1]:.2f})')
anti = sp[2] + sp[3]
print(f'    check: K0_S K0_S pairs, if the K0 anti-K0 pairs are made without coherence (a C-even pair gives K0_S K0_S twice as often, a C-odd one'
      f' never): {g20.K0_SHARE} x {g20.ANTI_K0_SHARE} x {anti:.2f} / 4 = {g20.K0_SHARE * g20.ANTI_K0_SHARE * anti / 4:.3f} mub, measured {g20.K0S_K0S[0]:.3f} +- {g20.K0S_K0S[1]:.3f}')
for label, kwargs in (('0.1 extra K anti-K pairs per Lambda', dict(k0_with_hyperons=g20.k0_share_with_hyperons(0.1))),
                      (f"ABBHHM's K0 share for the kaons with a hyperon too ({g20.K0_SHARE})", dict(k0_with_hyperons=g20.K0_SHARE)),
                      (f'charged Sigma {g20.SIGMA_PM_ALT} x Sigma0 (as the Sigma*(1385) with Sigma+ = Sigma0)', dict(sigma_pm=g20.SIGMA_PM_ALT)),
                      ('Xi0 = 2 Xi-', dict(xi0_over_xim=2.0)),
                      ('K0 = K+ in the K anti-K pairs (share 0.5)', dict(k0_in_pairs=0.5)),
                      *((f'anti-K0 share {a}', dict(anti_k0_share=a)) for a in g20.ANTI_K0_SHARE_RANGE),
                      (f'K0 share {g20.K0_SHARE_ALL} in the K anti-K pairs', dict(k0_in_pairs=g20.K0_SHARE_ALL)),
                      ("both of ABBHHM's shares with the reactions seen only when a K+- decays (K0 "
                       f"{g20.K0_SHARE_ALL} in the pairs, anti-K0 {g20.ANTI_K0_SHARE_RANGE[0]})",
                       dict(k0_in_pairs=g20.K0_SHARE_ALL, anti_k0_share=g20.ANTI_K0_SHARE_RANGE[0]))):
    print(f'      with {label}: {fmt(g20.species(**kwargs))}')
for sign in (-1, 1):   # (the charged Sigma, sigma_pm x Sigma0, are the only use of the Sigma0 cross section)
    s0 = g20.SIGMA0[0] + sign * g20.SIGMA0[1]
    print(f'      with Sigma0 = {s0:.2f}: {fmt(g20.species(sigma_pm=g20.SIGMA_PM * s0 / g20.SIGMA0[0]))}')

# ------------------------------------------------------------------------------------------------- numbers
M3_UP = tuple(M3_UPPER[1:3])
low, e_low = g20.ln_mean(channels, *M3_UP), float(np.sqrt(M3_UP[0] * M3_UP[1]))
print(f'\nThe code at {M3_UP[0]:g}-{M3_UP[1]:g} GeV (upper step of M3 with the two-body kaons; middle {e_low:.2f} GeV) [mub]: {fmt(low)}; K0_S {K0S(low[1], low[3]):.2f}')
two_body_lambda = g20.ln_mean([(r[0], r[1], r[6]) for r in two_body_steps()], *M3_UP)   # K+ Lambda and K+ Sigma0
m3_hyperons = low[0] + low[1] - low[2] - low[3] - g20.ln_mean([(r[0], r[1], r[6] + r[7]) for r in two_body_steps()], *M3_UP)
fl = g20.LAMBDA_SHARE_ABBHHM
print(f'  Lambda (with those from Sigma0) there, from ABBHHM\'s events: {fl} x {m3_hyperons:.2f} (with further particles) + {two_body_lambda:.2f} (two-body)'
      f' = {fl * m3_hyperons + two_body_lambda:.2f} mub; measured at 20 GeV: {g20.LAMBDA[0]:.2f} +- {g20.LAMBDA[1]:.2f}')
print('The line in ln E through both, at the middle of each step; per interaction of the type (%g mub):' % SIG3[6])
for row in MPI:
    t, lo, hi = int(row[0]), row[1], row[2]
    if t not in (3, 4):
        continue
    centre = float(np.sqrt(lo * hi))
    line = g20.trend(centre, e_low, low) / SIG3[2 * t]
    code = np.array(row[3:7])
    note = '' if t == 3 else ('  (M5, not changed: ' + ', '.join(f'{c / l - 1:+.0%}' for c, l in zip(code, line))
                              + f'; kaons {(code[0] + code[1]) / (line[0] + line[1]) - 1:+.0%}, antikaons {(code[2] + code[3]) / (line[2] + line[3]) - 1:+.0%},'
                              f' K0_S {K0S(code[1], code[3]) / K0S(line[1], line[3]) - 1:+.0%})')
    print(f'  {lo:g}-{hi:g} GeV (middle {centre:.2f} GeV): {fmt(line, 4)}; the code {fmt(code, 4)}{note}')
    if t == 3:
        print(f'      [mub] K0_S {SIG3[6] * K0S(code[1], code[3]):.2f}, events with a hyperon {SIG3[6] * (code[0] + code[1] - code[2] - code[3]):.2f},'
              f' with K anti-K {SIG3[6] * (code[2] + code[3]):.2f}')
at20 = sum(v for a, b, v in channels if a <= g20.E_GAMMA < b)
print(f'  at 20 GeV: K0_S of the code {K0S(at20[1], at20[3]):.2f} mub ({K0S(at20[1], at20[3]) / g20.K0S[0] - 1:+.0%}; the line itself goes through the measured'
      f' {g20.K0S[0]:.2f})')

# K0_S per inelastic pi- p interaction against the available energy E_A = sqrt(s) - m_a - m_b (Abe et al. 1985,
# Table 3; the points of Abe et al. 1984, Fig. 3): (beam momentum [GeV/c], K0_S per inelastic interaction, error)
PIMP = [(6.0, 0.043, 0.003), (15.0, 0.091, 0.004), (18.5, 0.075, 0.005)]
M_PI = derivation.MASS['pi+']
ea_pi = [derivation.available_energy(M_PI, g20.M_P, float(np.hypot(p, M_PI))) for p, _, _ in PIMP]
# total gamma p cross sections [mub] of the same experiments: (photon energy [GeV], sigma)
SIGMA_TOT = [t for t in g20.SIGMA_TOT_SLAC if t[0] > 4] + [(g20.E_GAMMA, g20.SIGMA_TOT[0])]


def sigma_tot(e):
    """Linear in ln E between the measured values."""
    return float(np.interp(np.log(e), np.log([x for x, _ in SIGMA_TOT]), [y for _, y in SIGMA_TOT]))


def photon_energy(ea):
    """Photon energy [GeV] on a proton at rest with the available energy ea = sqrt(s) - m_p (the README's s)."""
    return derivation.threshold(g20.M_P, ea + g20.M_P)


k0s_line = lambda e: float(K0S(*g20.trend(e, e_low, low)[[1, 3]]))
e1, e2 = photon_energy(ea_pi[0]), photon_energy(0.5 * (ea_pi[1] + ea_pi[2]))
r_pi = PIMP[0][1] / (0.5 * (PIMP[1][1] + PIMP[2][1]))
r_pi_err = r_pi * np.hypot(PIMP[0][2] / PIMP[0][1], 0.5 * np.hypot(PIMP[1][2], PIMP[2][2]) / (0.5 * (PIMP[1][1] + PIMP[2][1])))
r_line = (k0s_line(e1) / sigma_tot(e1)) / (k0s_line(e2) / sigma_tot(e2))
print(f'  energy dependence, pi- p (Abe et al. 1985, Table 3): K0_S per inelastic interaction {PIMP[0][1]} +- {PIMP[0][2]} at E_A = {ea_pi[0]:.2f} GeV,'
      f' {PIMP[1][1]} +- {PIMP[1][2]} and {PIMP[2][1]} +- {PIMP[2][2]} at {ea_pi[1]:.2f} and {ea_pi[2]:.2f} GeV: ratio {r_pi:.2f} +- {r_pi_err:.2f}')
print(f'    the line at the same E_A (photons of {e1:.1f} and {e2:.1f} GeV): K0_S {k0s_line(e1):.2f} of {sigma_tot(e1):.0f} mub and {k0s_line(e2):.2f} of'
      f' {sigma_tot(e2):.0f} mub: ratio {r_line:.2f}')

# ------------------------------------------------------------------------------------------------- energy
print('\nEnergy fraction chi of the K0_S (fraction of the nucleon energy in the frame in which the nucleon is fast)')
chi20, _ = g20.mean_chi()
chi_z = g20.mean_chi_from_z()[0]
lead20, others20 = g20.mean_chi(x_below=-0.2), g20.mean_chi(x_above=-0.2)
print(f'  at 20 GeV: mean {chi20:.4f} from the distributions in x and pT^2 (Tables III, VI); {chi_z:.4f} from those in z and pT^2 (Tables V, VI)')
print(f'    towards the nucleon (x < -0.2): {lead20[1]:.1%} of them, mean chi {lead20[0]:.3f}; the others {others20[0]:.4f}')
print('  at the middle of the steps, with the same invariant distribution, and the code:')
for row in MPI:
    t, lo, hi, (kp, k0, km, k0b, hp, hm, f) = int(row[0]), row[1], row[2], row[3:]
    if t != 3:
        continue
    centre = float(np.sqrt(lo * hi))
    data, lead = g20.mean_chi(centre)[0], g20.mean_chi(centre, x_below=-0.2)
    chi_l, chi_h = f * KAI[2 * t], KAI[2 * t + 1]
    h_mix = (hp * k0 + hm * k0b) / (k0 + k0b)                    # leading part of the K0 and anti-K0 together
    code = (1 - h_mix) * chi_l + h_mix * chi_h
    f_data = (data - h_mix * chi_h) / (1 - h_mix) / KAI[2 * t]
    with_16 = (1 - h_mix) * F_HIGHER * KAI[2 * t] + h_mix * chi_h
    e_kaons = sum(m * ((1 - h) * chi_l + h * chi_h) for m, h in ((kp, hp), (k0, hp), (km, hm), (k0b, hm)))
    print(f'    {lo:g}-{hi:g} GeV ({centre:.2f} GeV): mean {data:.4f} (x < -0.2: {lead[1]:.1%} at {lead[0]:.3f});')
    print(f'        the code {code:.4f} ({code / data - 1:+.1%}): {h_mix:.1%} at {chi_h:g} and the others at {f:g} x {KAI[2 * t]:g} = {chi_l:.4f} (the mean needs'
          f' {f_data:.2f} x; with {F_HIGHER:g} x, as in the other types: {with_16:.4f}, {with_16 / data - 1:+.0%})')
    print(f'        energy of the kaons per interaction: {e_kaons:.4f} of the nucleon energy ({kp + k0 + km + k0b:.3f} kaons)')
print(f'  the higher types ({F_HIGHER:g} x the chi of their low-energy pions, not changed), against the same distribution at their middles:')
for row in MPI:
    t, lo, hi, (kp, k0, km, k0b, hp, hm, f) = int(row[0]), row[1], row[2], row[3:]
    if t < 4:
        continue
    centre = float(np.sqrt(lo * hi))
    h_mix = (hp * k0 + hm * k0b) / (k0 + k0b)
    code = (1 - h_mix) * f * KAI[2 * t] + h_mix * KAI[2 * t + 1]
    data = g20.mean_chi(centre)[0]
    print(f'    M{t + 1}, {lo:g}-{hi:g} GeV ({centre:.3g} GeV): mean {data:.4f}; the code {code:.4f} ({code / data - 1:+.0%})')
all20 = float(np.sum(sp)) / SIG3[6]
print(f'  from the measurement at 20 GeV: {np.sum(sp):.1f} mub of kaons, {all20:.3f} per interaction of the type ({SIG3[6]:g} mub),'
      f' x {chi20:.4f} = {all20 * chi20:.4f}')

# ------------------------------------------------------------------- the alternative: one step for M4 with the 20 GeV values
print(f'\nThe alternative: one step for M4 ({M4[0]:g}-{M4[1]:g} GeV) with the values measured at 20 GeV')
alt = g20.one_step(SIG3[6])
hp4, hm4 = M4_LOWER[7:9]   # the leading fractions of M4
h_alt = (hp4 * alt[1] + hm4 * alt[3]) / (alt[1] + alt[3])
f_alt = (chi20 - h_alt * KAI[7]) / (1 - h_alt) / KAI[6]
print(f'  per interaction: {fmt(alt, 3)}; the K0_S then have the mean chi of 20 GeV, {chi20:.4f}, with {f_alt:.2f} x the chi of the low-energy pions'
      f' (taken as {g20.ONE_STEP_F}); K0_S at 20 GeV {SIG3[6] * K0S(alt[1], alt[3]):.2f} mub (measured {g20.K0S[0]:.2f})')
mid = float(np.sqrt(M4[0] * M4[1]))
line_mid = g20.trend(mid, e_low, low) / SIG3[6]
print(f'  the line at the middle of {M4[0]:g}-{M4[1]:g} GeV ({mid:.1f} GeV), the mean over the type for a photon field ~ 1/eps: {fmt(line_mid, 4)}; one step:'
      f' {", ".join(f"{a / l - 1:+.0%}" for a, l in zip(alt, line_mid))}')
chi_alt = (1 - h_alt) * g20.ONE_STEP_F * KAI[6] + h_alt * KAI[7]
print(f'  mean chi of the K0_S at that middle, with the same invariant distribution: {g20.mean_chi(mid)[0]:.4f}; one step {chi_alt:.4f}'
      f' ({chi_alt / g20.mean_chi(mid)[0] - 1:+.0%}; exact at 20 GeV)')
lower = M4_LOWER
print(f'  K0_S from {M3_UP[0]:g}-{M3_UP[1]:g} GeV ({K0S(low[1], low[3]):.2f} mub) to the start of M4: one step {SIG3[6] * K0S(alt[1], alt[3]):.2f} mub, the code\'s'
      f' lower step {SIG3[6] * K0S(lower[4], lower[6]):.2f} mub')
print(f'  at {lower[1]:g}-{lower[2]:g} GeV, one step against the code\'s lower step (photons of a narrow line there): '
      + ', '.join(f'{n} x{a / c:.2f}' for n, a, c in zip(NAMES, alt, lower[3:7])))
