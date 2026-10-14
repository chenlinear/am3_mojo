"""Photo-hadronic kaons, README "Kaons in multi-pion production": the kaons of the multi-pion types M2 (0.9-1.5 GeV),
M3 (1.5-5 GeV) and M4 (5-50 GeV) against measured strange-particle photoproduction on protons, without the two-body
reactions gamma N -> K Lambda, K Sigma (separate channels in the code).

  * Thresholds. Below that of gamma N -> K Lambda pi (1.15 GeV), where the kaons of M2 start, only the two-body
    reactions make a kaon. Their thresholds are printed; derivations/kaon_production.py derives the others
    (kp.thresholds).
  * Three-body production up to 1.5 GeV, against M2: gamma p -> K+ Sigma+- pi-+ (SAPHIR, HEPData ins875421),
    K0 pi0 Sigma+ (CBELSA/TAPS, HEPData ins781382) and K0 Lambda pi+ (ABBHHM, one value for 1.2-1.7 GeV). The
    unmeasured K+ Lambda pi0 is taken as 0.5 to 1 times K0 Lambda pi+.
  * ABBHHM (Aachen-Berlin-Bonn-Hamburg-Heidelberg-Muenchen collaboration, Erbe et al. 1969, Phys. Rev. 188, 2060;
    hydrogen bubble chamber, bremsstrahlung up to 5.8 GeV): the events per reaction (Table VI: 1312 events, 1185 of
    them with a visible strange-particle decay), corrected for the decays not seen; the cross section of
    gamma p -> p K+ K- (Table V) and those of four reactions with further particles (read off Fig. 29). With a
    photon flux ~ 1/E, a reaction's events are ~ int sigma dE/E, so those with further particles are normalised to
    the two-body ones.
  * The two steps of M3 (1.5-2.5 and 2.5-5 GeV), across which the yields rise, from ABBHHM: events with a hyperon
    and further particles, with K anti-K, and per kaon species. The code takes these times a factor per step from
    ABBHHM, SLAC and DESY (computed after their comparisons).
  * SLAC 82-inch bubble chamber, photons of one energy: events with a visible strange-particle decay at 1.44, 2.8,
    4.7 GeV (Ballam et al. 1972, Phys. Rev. D 5, 545, Table V) and 9.3 GeV (Bingham et al. 1973, Phys. Rev. D 8,
    1277, Table IV), against the visible decays of the code's kaons and of the hyperons that balance their
    strangeness. ABBHHM and SLAC against each other: in visible decays, in gamma p -> p K+ K- (Ballam et al.,
    Table VII; Bingham et al., Table V), and in the Lambda that each needs, against the Lambda measured at 20 GeV
    (Abe et al. 1985; tests/support/measured/strange_20gev.py). SLAC's normalisation: its channels without strange
    particles against those of the DESY streamer chamber with tagged photons (Struczinski et al. 1976,
    Nucl. Phys. B 108, 45; HEPData ins100089, Table 1, from Table 3 of the paper), which is normalised to a fit of
    the published total cross sections (+-5 %).
  * DESY's events with a visible strange-particle decay (Struczinski et al. 1976, Table 2, not in the HEPData
    record) in 8 bins of 1.6-6.3 GeV, lower limits as they lack the decays the chamber could not see: against the
    code's, with ABBHHM's visibilities and with no decay lost, and from 2 to 5.8 GeV against ABBHHM and SLAC.

The two steps of M4 (5-15 and 15-50 GeV) are set from the measurement at 20 GeV (m4_steps.py). At the end, the same
comparisons for one step for all of M4 with the values measured at 20 GeV (Results, "One step or two for M4"). The
code's tables are read from src/ (code_tables.py)."""
import math
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from code_tables import photopion_tables, multipion_kaons, two_body_steps, multi_body_threshold  # noqa: E402
from support.measured import strange_20gev as g20  # noqa: E402
from support.paths import load_derivation  # noqa: E402

derivation = load_derivation('kaon_production')   # PDG 2026 masses, the README's s
kaon_decays = load_derivation('kaon_decays')       # the README's branching ratios
pdg = load_derivation('pdg')                       # PDG 2026's Lambda -> p pi-

# masses [GeV] (PDG 2026, derivations/kaon_production.py)
MP, MN, MKP, MK0, MLAM, MSIGP, MSIG0, MSIGM, MSIG1385 = (derivation.MASS[q] for q in (
    "p", "n", "K+", "K0", "Lambda", "Sigma+", "Sigma0", "Sigma-", "Sigma*+"))
threshold = derivation.threshold   # photon energy [GeV] at the threshold of gamma N -> these masses (the README's s)


# --------------------------------------------------------------------------------------------- the code's kaons
T = photopion_tables()
SIG3, EM3, EX3 = T['IT3sig'], T['IT3Em'], T['IT3Ex']
KY = two_body_steps()            # [emin, emax, chi x 4, sigma of p -> K+, p -> K0, n -> K+, n -> K0]
MPI = multipion_kaons()          # [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+, h-, f]: steps of type M(t + 1), protons
START = multi_body_threshold()
M2, M3, M4 = ((EM3[2 * t], EX3[2 * t]) for t in (1, 2, 3))     # the photon energies [GeV] of the types
M3_STEPS = sorted(tuple(r[1:3]) for r in MPI if int(r[0]) == 2)   # the code's two steps of M3
M4_ROW = min((r for r in MPI if int(r[0]) == 3), key=lambda r: r[1])
ABB_TOP = 5.8                    # ABBHHM's photons reach 5.8 GeV


def two_body(e):
    """(K+, K0) cross sections [mub] of the two-body steps for protons at the photon energy e [GeV]."""
    for r in KY:
        if r[0] <= e < r[1]:
            return r[6], r[7]
    return 0.0, 0.0


def multi(e):
    """(K+, K0, K-, anti-K0) cross sections [mub] of the multi-pion types for protons at the photon energy e."""
    for row in MPI:
        if row[1] <= e < row[2]:
            return tuple(SIG3[2 * int(row[0])] * m for m in row[3:7])
    return 0.0, 0.0, 0.0, 0.0


def mean(f, a, b, alpha, n=6000):
    """Mean of f over the photon energies a ... b with the weight e^-alpha: the weight of the photon energy e (in
    the nucleon rest frame) in the interaction rate of a nucleon in a photon field n(eps) ~ eps^-alpha."""
    s = w = 0.0
    for i in range(n):
        e = a + (b - a) * (i + 0.5) / n
        s += f(e) * e ** (-alpha)
        w += e ** (-alpha)
    return s / w


def lnint(f, a, b, n=6000):
    """int f dE/E from a to b."""
    return mean(f, a, b, 1, n) * math.log(b / a)


def steps_lnint(table):
    """int sigma dE/E of a table of steps (E_low, E_high, sigma, ...)."""
    return sum(r[2] * math.log(r[1] / r[0]) for r in table)


def step(table, e, col=2):
    for r in table:
        if r[0] <= e < r[1]:
            return r[col]
    return 0.0


print('Thresholds [GeV] (photon energy in the nucleon rest frame) of the two-body reactions')
for name, mt, ms in (('gamma p -> K+ Lambda', MP, (MKP, MLAM)), ('gamma n -> K0 Lambda', MN, (MK0, MLAM)),
                     ('gamma p -> K+ Sigma0', MP, (MKP, MSIG0)), ('gamma p -> K0 Sigma+', MP, (MK0, MSIGP)),
                     ('gamma n -> K+ Sigma-', MN, (MKP, MSIGM))):
    print(f'  {name:28s} {threshold(mt, *ms):.3f}')
# gamma N -> K Lambda pi in its four charge states, and gamma p -> K0 Sigma+(1385)
E3 = derivation.threshold_span(derivation.NUCLEON, derivation.KAON, ("Lambda",), derivation.PION)[0]
E_Y1385 = threshold(MP, MK0, MSIG1385)
print(f'  the kaons of the multi-pion types start at {START:g} GeV in the code')

print('\nThe code: kaons of the multi-pion types (protons), cross section x multiplicity [mub]')
for row in MPI:
    t = int(row[0])
    s = SIG3[2 * t]
    assert EM3[2 * t] <= row[1] < row[2] <= EX3[2 * t]
    kp, k0, km, k0b = (s * m for m in row[3:7])
    print(f'  M{t + 1}, {row[1]:g}-{row[2]:g} GeV ({s:g} mub): K+ {kp:.3f}  K0 {k0:.3f}  K- {km:.3f}  anti-K0 {k0b:.3f};'
          f' events with a hyperon {kp + k0 - km - k0b:.2f}, with K anti-K {km + k0b:.2f}')

# ---------------------------------------------------------------------------------------------------- M2
print(f'\n== M2 ({M2[0]:g}-{M2[1]:g} GeV). Below {E3:.3f} GeV only the two-body reactions make a kaon.')
m2 = multi(0.5 * (START + M2[1]))

# measured cross sections of reactions with further particles [mub]: (E_low, E_high [GeV], sigma ..., error ...)
# SAPHIR (Schulday et al. 2010, HEPData ins875421); its errors contain the uncertainty of the background subtraction.
# gamma p -> K+ Sigma- pi+ (HEPData Table 1, the paper's Table 2):
SAPHIR_SM = [(1.3, 1.4, 0.0028, 0.0026), (1.4, 1.5, 0.029, 0.0061), (1.5, 1.6, 0.1121, 0.0095), (1.6, 1.7, 0.1322, 0.0137),
             (1.7, 1.8, 0.1597, 0.0113), (1.8, 1.9, 0.1894, 0.0152), (1.9, 2.0, 0.179, 0.0152), (2.0, 2.1, 0.2322, 0.0141),
             (2.1, 2.2, 0.2761, 0.0177), (2.2, 2.3, 0.2727, 0.0179), (2.3, 2.4, 0.2865, 0.0194), (2.4, 2.5, 0.2595, 0.03),
             (2.5, 2.6, 0.3339, 0.0243)]
# gamma p -> K+ Sigma+ pi-, from Sigma+ -> n pi+ and from Sigma+ -> p pi0, each the whole cross section, and their errors
# (HEPData Table 2, the paper's Tables 3-4; the error of the last p pi0 bin is printed as 0, a bin not used here):
SAPHIR_SP = [(1.3, 1.4, 0.0116, 0.0067, 0.0076, 0.0113), (1.4, 1.5, 0.072, 0.0367, 0.0191, 0.0347),
             (1.5, 1.6, 0.1644, 0.1075, 0.0275, 0.0538), (1.6, 1.7, 0.2314, 0.3395, 0.0419, 0.0496),
             (1.7, 1.8, 0.4005, 0.3817, 0.0359, 0.0519), (1.8, 1.9, 0.5377, 0.5983, 0.0448, 0.0605),
             (1.9, 2.0, 0.7802, 0.7209, 0.0559, 0.0662), (2.0, 2.1, 0.6789, 0.7605, 0.0641, 0.0794),
             (2.1, 2.2, 0.6861, 0.9128, 0.065, 0.0662), (2.2, 2.3, 0.8147, 0.9323, 0.0726, 0.0816),
             (2.3, 2.4, 0.9029, 0.8905, 0.0894, 0.1151), (2.4, 2.5, 0.6359, 0.7702, 0.0806, 0.1115),
             (2.5, 2.6, 0.7302, 0.9037, 0.0868, 0.0)]
# CBELSA/TAPS (Nanova et al. 2008, HEPData ins781382; statistical errors): gamma p -> K0 pi0 Sigma+ (Table 13, the
# paper's Fig. 13), which above 1.85 GeV leaves out K*0 Sigma+ (sigma_Sigma - sigma_SigmaK*, the paper's Sec. 6.2;
# the counts of its Table 1), and gamma p -> K*0 Sigma+ (Table 12), a third of which decays into K0 pi0:
CBELSA_K0PI0 = [(1.4, 1.5, 0.085, 0.034), (1.5, 1.7, 0.0975, 0.0224), (1.7, 1.85, 0.1511, 0.0294),
                (1.85, 2.0, 0.1734, 0.0445), (2.0, 2.15, 0.2743, 0.0457), (2.15, 2.3, 0.2774, 0.0703),
                (2.3, 2.5, 0.2496, 0.051)]
CBELSA_KSTAR = [(1.85, 2.0, 0.2648, 0.0576), (2.0, 2.15, 0.4414, 0.0883), (2.15, 2.3, 0.5301, 0.0846),
                (2.3, 2.5, 0.5192, 0.0762)]
# the whole final state K0 pi0 Sigma+ (E_low, E_high, sigma, error), the two errors added as independent. Above
# 1.85 GeV, where Table 13 is anti-correlated with Table 12, this overstates the error, which is not used.
K0PI0 = float(derivation.readme_kstar_k0pi0())   # the share of the K*0 that decays into K0 pi0
CBELSA_K0 = [(lo, hi, s + sum(k * K0PI0 for l2, h2, k, _ in CBELSA_KSTAR if (l2, h2) == (lo, hi)),
              math.hypot(e, sum(ek * K0PI0 for l2, h2, _, ek in CBELSA_KSTAR if (l2, h2) == (lo, hi))))
             for lo, hi, s, e in CBELSA_K0PI0]
# ABBHHM (Erbe et al. 1969), Fig. 29, read off the figure to about 0.02 mub: (E_low, E_high, sigma, error). The
# Lambda K0 pi+ panel, read on its own tick marks, puts its threshold arrow at 1.17-1.19 GeV (true: 1.168 GeV).
ABB_LK0PIP = [(1.2, 1.7, 0.74, 0.34), (1.7, 2.0, 1.43, 0.59), (2.0, 3.5, 0.80, 0.26), (3.5, 5.8, 0.22, 0.14)]
ABB_SPKPPIM = [(1.3, 1.5, 0.07, 0.055), (1.5, 2.0, 0.22, 0.085), (2.0, 2.5, 0.73, 0.185), (2.5, 3.0, 0.36, 0.14),
               (3.0, 3.5, 0.26, 0.135), (3.5, 5.8, 0.19, 0.09)]
ABB_SMKPPIP = [(1.3, 1.5, 0.05, 0.05), (1.5, 2.0, 0.13, 0.06), (2.0, 3.5, 0.24, 0.05), (3.5, 5.8, 0.17, 0.055)]
ABB_LKPPIPPIM = [(1.45, 2.0, 0.03, 0.035), (2.0, 2.5, 0.36, 0.14), (2.5, 3.0, 0.29, 0.15), (3.0, 3.5, 0.47, 0.125),
                 (3.5, 5.8, 0.27, 0.09)]
# ABBHHM, Table V: gamma p -> p K+ K-
ABB_PKK = [(1.58, 2.0, 0.21, 0.07), (2.0, 2.5, 0.47, 0.12), (2.5, 3.0, 0.54, 0.14), (3.0, 3.5, 1.06, 0.23),
           (3.5, 4.0, 0.64, 0.19), (4.0, 4.5, 0.97, 0.26), (4.5, 5.0, 1.11, 0.30), (5.0, 5.8, 0.92, 0.26)]


def three_body(e, r, lam):
    """Measured three-body kaon cross sections (K+, K0) [mub] of protons at the photon energy e < 1.5 GeV:
    K+ Sigma+- pi-+ (SAPHIR), K0 pi0 Sigma+ (CBELSA/TAPS; K+ Sigma0 pi0 and K0 Sigma0 pi+, not measured, each taken
    equal to it), K0 Lambda pi+ = lam(e) (from ABBHHM's value for 1.2-1.7 GeV) and K+ Lambda pi0 = r x that."""
    sp = 0.5 * (step(SAPHIR_SP, e, 2) + step(SAPHIR_SP, e, 3))
    k0s = step(CBELSA_K0, e)
    return sp + step(SAPHIR_SM, e) + k0s + r * lam(e), 2 * k0s + lam(e)


lo, hi, sig, err = ABB_LK0PIP[0]
# the same int sigma dE/E, constant over the bin or only above the threshold of Sigma+(1385) K0, through which most
# of the reaction goes there (the paper's Table VII: 0.68 +- 0.48 mub for 1.42-2.0 GeV)
flat = lambda e: sig if lo <= e < hi else 0.0
late = lambda e: sig * math.log(hi / lo) / math.log(hi / E_Y1385) if E_Y1385 <= e < hi else 0.0
print(f'  measured three-body production on protons, mean over {START:g}-{M2[1]:g} GeV [mub]. K0 Lambda pi+ = {sig:.2f} +- {err:.2f} mub for')
print(f'  {lo}-{hi} GeV (ABBHHM), taken as constant over that bin or as zero below the Sigma(1385) K threshold, {E_Y1385:.2f} GeV;')
print('  K+ Lambda pi0 = 0.5 ... 1 x K0 Lambda pi+:')


def three_body_errors(r, lam, alpha):
    """Errors of the mean of sum(three_body) over START-1.5 GeV with the weight e^-alpha: in all, and of SAPHIR's
    and CBELSA/TAPS's reactions alone. Their bins are independent, K0 pi0 Sigma+ counting three times (for itself
    and the two reactions equal to it). K0 Lambda pi+ and K+ Lambda pi0 = r x that have ABBHHM's relative error,
    fully correlated."""
    w = lambda lo, hi: mean(lambda e: 1.0 if lo <= e < hi else 0.0, START, M2[1], alpha)
    var = sum((w(lo, hi) * de) ** 2 for lo, hi, _, de in SAPHIR_SM)
    var += sum((w(lo, hi) * 0.5 * math.hypot(e1, e2)) ** 2 for lo, hi, _, _, e1, e2 in SAPHIR_SP)
    var += sum((3 * w(lo, hi) * de) ** 2 for lo, hi, _, de in CBELSA_K0)
    abb = (1 + r) * mean(lam, START, M2[1], alpha) * err / sig
    return math.sqrt(var + abb ** 2), math.sqrt(var)


for alpha in (0, 1, 2):
    vals, rel, others = [], [], []
    for lam in (late, flat):
        for r in (0.5, 1.0):
            v = mean(lambda e: sum(three_body(e, r, lam)), START, M2[1], alpha)
            e_all, e_others = three_body_errors(r, lam, alpha)
            vals.append(v)
            rel.append(e_all / v)
            others.append(e_others)
    print(f"    weight e^-{alpha}: {min(vals):.2f}-{max(vals):.2f}, error {min(rel):.0%}-{max(rel):.0%} (ABBHHM's K0 Lambda pi+ alone"
          f" {err / sig:.0%}; SAPHIR's and CBELSA/TAPS's reactions {max(others):.2f} mub); the code (M2): K+ {m2[0]:.3f},"
          f" K0 {m2[1]:.3f}, together {m2[0] + m2[1]:.3f}")
# the code's M2 is set at the middle of these cross sections, per species, with the weight 1/E (as the steps of M3)
kp_vals = [mean(lambda e: three_body(e, r, lam)[0], START, M2[1], 1) for lam in (late, flat) for r in (0.5, 1.0)]
k0_vals = [mean(lambda e: three_body(e, r, lam)[1], START, M2[1], 1) for lam in (late, flat) for r in (0.5, 1.0)]
m2_mid = (0.5 * (min(kp_vals) + max(kp_vals)), 0.5 * (min(k0_vals) + max(k0_vals)))
print(f'    the middle with the weight 1/E, which the code takes: K+ {min(kp_vals):.3f}-{max(kp_vals):.3f}, middle {m2_mid[0]:.3f}'
      f' ({m2_mid[0] / SIG3[2]:.4f} per interaction of {SIG3[2]:g} mub); K0 {min(k0_vals):.3f}-{max(k0_vals):.3f}, middle'
      f' {m2_mid[1]:.3f} ({m2_mid[1] / SIG3[2]:.4f}); together {sum(m2_mid):.3f}')

# --------------------------------------------------------------------------------------------- M3 and M4
print(f'\n== M3 ({M3[0]:g}-{M3[1]:g} GeV) and M4 ({M4[0]:g}-{M4[1]:g} GeV)')
# probability that a strange particle decays visibly in the bubble chamber: the branching ratio into charged particles
# (Lambda -> p pi-, PDG 2026; a K0 is a K0_S half the time and K0_S -> pi+ pi-, both as in the README) over ABBHHM's
# correction for decays outside the chamber or too close to the vertex. Few K+- decay in the chamber (an estimate).
BR_LAMBDA = float(pdg.BR_LAMBDA_P_PIM)
BR_K0_PIPI = float(derivation.numeric_neutral()(1.0, 0.0)) * kaon_decays.branching_ratios()['K0S'][r'$\pi^+\pi^-$']
V_LAM = BR_LAMBDA / 1.08
V_K0 = BR_K0_PIPI / 1.11
V_SIGP, V_SIGM, V_KCH = 1 / 1.24, 1 / 1.11, 0.05


def vis(*p):
    """Probability that at least one of the particles with the visibilities p decays visibly."""
    q = 1.0
    for x in p:
        q *= 1.0 - x
    return 1.0 - q


# ABBHHM, Table VI: (reaction, events with a visible decay (column B, ambiguous ones apportioned), kaons (K+, K0, K-,
# anti-K0), probability of a visible decay, class: 2 two-body, Y with a hyperon and further particles, KK with
# K anti-K). p K+ K- and p K+ K- pi+ pi- include the events without a visible decay (107 and 20), found by their
# kinematics and ionisation.
TABLE_VI = [
    ('Lambda K+', 186, (1, 0, 0, 0), vis(V_LAM), '2'), ('Sigma+ K0', 18, (0, 1, 0, 0), vis(V_SIGP, V_K0), '2'),
    ('Sigma0 K+', 102, (1, 0, 0, 0), vis(V_LAM), '2'),
    ('p K+ K-', 5 + 107, (1, 0, 1, 0), 1.0, 'KK'),
    ('p K0 anti-K0 (pi0 ...)', 31, (0, 1, 0, 1), vis(V_K0, V_K0), 'KK'),
    ('Lambda K0 pi+', 122, (0, 1, 0, 0), vis(V_LAM, V_K0), 'Y'), ('Sigma+ K+ pi-', 54, (1, 0, 0, 0), vis(V_SIGP), 'Y'),
    ('Sigma0 K0 pi+', 16, (0, 1, 0, 0), vis(V_LAM, V_K0), 'Y'), ('Sigma- K+ pi+', 34, (1, 0, 0, 0), vis(V_SIGM), 'Y'),
    ('n K+ anti-K0 (pi0 ...)', 23, (1, 0, 0, 1), vis(V_K0, V_KCH), 'KK'),
    ('Sigma+ K0 pi0 (pi0 ...)', 18, (0, 1, 0, 0), vis(V_SIGP, V_K0), 'Y'),
    ('(Sigma0, Lambda) K+ pi0 (pi0 ...)', 173, (1, 0, 0, 0), vis(V_LAM), 'Y'),
    ('p K+ anti-K0 pi-', 8, (1, 0, 0, 1), vis(V_K0, V_KCH), 'KK'), ('p K0 K- pi+', 19, (0, 1, 1, 0), vis(V_K0, V_KCH), 'KK'),
    ('Lambda K+ pi+ pi-', 38, (1, 0, 0, 0), vis(V_LAM), 'Y'), ('Sigma+ K0 pi+ pi-', 13, (0, 1, 0, 0), vis(V_SIGP, V_K0), 'Y'),
    ('Sigma0 K+ pi+ pi-', 6, (1, 0, 0, 0), vis(V_LAM), 'Y'), ('Sigma- K0 pi+ pi+', 10, (0, 1, 0, 0), vis(V_SIGM, V_K0), 'Y'),
    ('p K0 anti-K0 pi0 (pi0 ...)', 5, (0, 1, 0, 1), vis(V_K0, V_K0), 'KK'),
    ('n K+ K- pi+ (pi0 ...)', 7, (1, 0, 1, 0), vis(V_KCH, V_KCH), 'KK?'),   # seen only when a K+- decays in the chamber
    ('n K0 anti-K0 pi+ (pi0 ...)', 12, (0, 1, 0, 1), vis(V_K0, V_K0), 'KK'),
    ('Sigma+ K+ pi- pi0 (pi0 ...)', 31, (1, 0, 0, 0), vis(V_SIGP), 'Y'),
    ('(Sigma0, Lambda) K0 pi+ pi0 (pi0 ...)', 107, (0, 1, 0, 0), vis(V_LAM, V_K0), 'Y'),
    ('Sigma- K+ pi+ pi0 (pi0 ...)', 13, (1, 0, 0, 0), vis(V_SIGM), 'Y'),
    ('Xi- K+ K0 pi+ (pi0 ...)', 1, (1, 1, 0, 0), 0.95, 'Y'),   # printed Xi- K+ anti-K0 pi+, which violates strangeness
    ('p K+ K- pi+ pi-', 5 + 20, (1, 0, 1, 0), 1.0, 'KK'),
    ('Lambda K0 pi+ pi+ pi-', 7, (0, 1, 0, 0), vis(V_LAM, V_K0), 'Y'), ('Sigma- K+ pi+ pi+ pi-', 1, (1, 0, 0, 0), vis(V_SIGM), 'Y'),
    ('p K+ anti-K0 pi- pi0 (pi0 ...)', 4, (1, 0, 0, 1), vis(V_K0, V_KCH), 'KK'),
    ('p K0 anti-K0 pi+ pi- (pi0 ...)', 18, (0, 1, 0, 1), vis(V_K0, V_K0), 'KK'),
    ('p K0 K- pi+ pi0 (pi0 ...)', 5, (0, 1, 1, 0), vis(V_K0, V_KCH), 'KK'),
    ('n K+ anti-K0 pi+ pi- (pi0 ...)', 6, (1, 0, 0, 1), vis(V_K0, V_KCH), 'KK'),
    ('n K0 K- pi+ pi+ (pi0 ...)', 3, (0, 1, 1, 0), vis(V_K0, V_KCH), 'KK'),
    ('Sigma+ K0 pi+ pi- pi0 (pi0 ...)', 19, (0, 1, 0, 0), vis(V_SIGP, V_K0), 'Y'),
    ('(Sigma0, Lambda) K+ pi+ pi- pi0 (pi0 ...)', 23, (1, 0, 0, 0), vis(V_LAM), 'Y'),
    ('(Sigma0, Lambda) K0 pi+ pi+ pi- pi0 (pi0 ...)', 21, (0, 1, 0, 0), vis(V_LAM, V_K0), 'Y'),
    ('Sigma- K0 pi+ pi+ pi0 (pi0 ...)', 13, (0, 1, 0, 0), vis(V_SIGM, V_K0), 'Y'),
    ('p K+ K- pi+ pi- pi0 (pi0 ...)', 2, (1, 0, 1, 0), vis(V_KCH, V_KCH), 'KK?'),
    ('n K0 anti-K0 pi+ pi+ pi- (pi0 ...)', 1, (0, 1, 0, 1), vis(V_K0, V_K0), 'KK'),
    ('Sigma+ K0 pi+ pi+ pi- pi- (pi0 ...)', 5, (0, 1, 0, 0), vis(V_SIGP, V_K0), 'Y'),
]
assert sum(n for _, n, _, _, _ in TABLE_VI) == 1185 + 107 + 20      # the sums of the paper's table
CLASSES = ('2', 'Y', 'KK', 'KK?')
seen = {c: sum(n for _, n, _, _, cls in TABLE_VI if cls == c) for c in CLASSES}
made = {c: sum(n / v for _, n, _, v, cls in TABLE_VI if cls == c) for c in CLASSES}
kaons = {c: [sum(n / v * k[i] for _, n, k, v, cls in TABLE_VI if cls == c) for i in range(4)] for c in CLASSES}
print(f'  ABBHHM, Table VI: {sum(seen.values())} events: two-body {seen["2"]}, with a hyperon and further particles {seen["Y"]},'
      f' with K anti-K {seen["KK"] + seen["KK?"]} ({seen["KK?"]} of them in two reactions seen only when a K+- decays)')
print(f'    corrected for the decays that are not seen: {made["2"]:.0f}, {made["Y"]:.0f} and {made["KK"]:.0f} (+ {made["KK?"]:.0f} for those two reactions)')
ln2 = sum((r[6] + r[7]) * math.log(r[1] / r[0]) for r in KY)
per_event = ln2 / made['2']
print(f'    int sigma dE/E of the code\'s two-body steps (protons): {ln2:.2f} mub, so {1 / per_event:.0f} events per mub')
own = []
for name, table in [('Lambda K0 pi+', ABB_LK0PIP), ('Sigma+ K+ pi-', ABB_SPKPPIM), ('Sigma- K+ pi+', ABB_SMKPPIP),
                    ('Lambda K+ pi+ pi-', ABB_LKPPIPPIM), ('p K+ K-', ABB_PKK)]:
    n, v = next((n, v) for nm, n, _, v, _ in TABLE_VI if nm == name)
    own.append(n / v / steps_lnint(table))
    print(f'      (the paper\'s own cross section of {name}: int sigma dE/E = {steps_lnint(table):.2f} mub for {n / v:.0f} events: {own[-1]:.0f} per mub)')
data_Y = per_event * made['Y']
data_KK = (per_event * made['KK'], per_event * (made['KK'] + made['KK?']))
m_Y = lambda e: multi(e)[0] + multi(e)[1] - multi(e)[2] - multi(e)[3]
m_KK = lambda e: multi(e)[2] + multi(e)[3]
code_Y, code_KK = lnint(m_Y, M2[0], ABB_TOP), lnint(m_KK, M2[0], ABB_TOP)
print(f'    int sigma dE/E up to {ABB_TOP:g} GeV, with a hyperon and further particles: {data_Y:.2f} mub; the code {code_Y:.2f} ({code_Y / data_Y - 1:+.0%})'
      f' (M2 {lnint(m_Y, *M2):.2f}, M3 {lnint(m_Y, *M3):.2f}, M4 up to {ABB_TOP:g} GeV {lnint(m_Y, M4[0], ABB_TOP):.2f})')
print(f'    with K anti-K: {data_KK[0]:.2f}-{data_KK[1]:.2f} mub; the code {code_KK:.2f} ({code_KK / data_KK[0] - 1:+.0%} ... {code_KK / data_KK[1] - 1:+.0%})'
      f' (M3 {lnint(m_KK, *M3):.2f}, M4 up to {ABB_TOP:g} GeV {lnint(m_KK, M4[0], ABB_TOP):.2f})')
for i, name in enumerate(('K+', 'K0', 'K-', 'anti-K0')):
    d = per_event * (kaons['Y'][i] + kaons['KK'][i] + kaons['KK?'][i])
    c = lnint(lambda e: multi(e)[i], M2[0], ABB_TOP)
    print(f'      {name:8s} {d:.2f} mub; the code {c:.2f} ({c / d - 1:+.0%})')
lam = sum(n / v for name, n, k, v, cls in TABLE_VI if cls == 'Y' and ('Lambda' in name or 'Sigma0' in name))
sgp = sum(n / v for name, n, k, v, cls in TABLE_VI if cls == 'Y' and name.startswith('Sigma+'))
sgm = sum(n / v for name, n, k, v, cls in TABLE_VI if cls == 'Y' and name.startswith('Sigma-'))
fl, fp, fm = (x / (lam + sgp + sgm) for x in (lam, sgp, sgm))
print(f'    hyperons of the events with further particles: Lambda or Sigma0 {fl:.2f}, Sigma+ {fp:.2f}, Sigma- {fm:.2f}')
V_Y = fl * V_LAM + fp * V_SIGP + fm * V_SIGM

# the energy dependence: the reactions whose cross sections the paper gives (Fig. 29, Table V), together
print('    the five reactions with further particles whose cross sections the paper gives, together [mub]:')
for lo, hi in [(1.25, 1.5), (1.5, 1.75), (1.75, 2.05), (2.05, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 5.8)]:
    e = 0.5 * (lo + hi)
    s5 = sum(step(t, e) for t in (ABB_LK0PIP, ABB_SPKPPIM, ABB_SMKPPIP, ABB_LKPPIPPIM, ABB_PKK))
    print(f'      {lo:.2f}-{hi:.2f} GeV: {s5:.2f}')
five = sum(n / v for nm, n, _, v, _ in TABLE_VI if nm in ('Lambda K0 pi+', 'Sigma+ K+ pi-', 'Sigma- K+ pi+', 'Lambda K+ pi+ pi-', 'p K+ K-'))
print(f'      (they are {five / (made["Y"] + made["KK"] + made["KK?"]):.0%} of the events with further particles)')

# the two steps of M3 (1.5-2.5 and 2.5-5 GeV): what these data give for each
print('  the two steps of M3 from ABBHHM [mub]:')
(LO1, LO2), (HI1, HI2) = M3_STEPS
L_LO, L_HI = math.log(LO2 / LO1), math.log(ABB_TOP / HI1)
step_mean = lambda table, a, b: lnint(lambda e: step(table, e), a, b) / math.log(b / a)
corrected = lambda *names: sum(n / v for nm, n, _, v, _ in TABLE_VI if nm in names)
# with K anti-K: p K+ K- (Table V) times the K anti-K events per p K+ K- event of Table VI; in the lower step only
# those without further particles (p K+ K-, p K0 anti-K0, n K+ anti-K0): with a pion they start at 1.8 GeV
THREE = ('p K+ K-', 'p K0 anti-K0 (pi0 ...)', 'n K+ anti-K0 (pi0 ...)')
kk_lo = step_mean(ABB_PKK, LO1, LO2) * corrected(*THREE) / corrected('p K+ K-')
kk_hi = [step_mean(ABB_PKK, HI1, HI2) * x / corrected('p K+ K-') for x in (made['KK'], made['KK'] + made['KK?'])]
# with a hyperon, lower step: (a) the reactions with measured cross sections (Lambda K0 pi+ and Lambda K+ pi+ pi- of
# ABBHHM's Fig. 29, Sigma+- K+ pi-+ of SAPHIR, K0 pi0 Sigma+ of CBELSA/TAPS); (b) their unmeasured three-body partners
# (K+ Lambda pi0 = 0.5 ... 1 x K0 Lambda pi+; K+ Sigma0 pi0 and K0 Sigma0 pi+ as K0 pi0 Sigma+); (c) the other
# reactions with two pions, as Lambda K+ pi+ pi- by their events in Table VI (those with three pions, from 1.7-1.9 GeV
# on, are left out). Upper limit: the measured reactions scaled to all events with a hyperon, as if every reaction
# rose as early. SAPHIR's Sigma+- K+ pi-+ cover the step in ten bins (2-3 % errors for the means over it), ABBHHM's
# in two (21-23 %). K+ Sigma+ pi- is the mean of its two Sigma+ decays.
SAPHIR_SP_MEAN = [(lo, hi, 0.5 * (a + b), 0.5 * math.hypot(ea, eb)) for lo, hi, a, b, ea, eb in SAPHIR_SP]
MEASURED = (ABB_LK0PIP, SAPHIR_SP_MEAN, SAPHIR_SM, ABB_LKPPIPPIM, CBELSA_K0)
MEASURED_ABBHHM = (ABB_LK0PIP, ABB_SPKPPIM, ABB_SMKPPIP, ABB_LKPPIPPIM, CBELSA_K0)
TWO_PIONS = ('Sigma+ K0 pi+ pi-', 'Sigma0 K+ pi+ pi-', 'Sigma- K0 pi+ pi+', 'Sigma+ K+ pi- pi0 (pi0 ...)',
             '(Sigma0, Lambda) K0 pi+ pi0 (pi0 ...)', 'Sigma- K+ pi+ pi0 (pi0 ...)')
assert all(any(nm == name and cls == 'Y' for nm, _, _, _, cls in TABLE_VI) for name in THREE + TWO_PIONS if name not in THREE)
y_meas = sum(step_mean(t, LO1, LO2) for t in MEASURED)
y_meas_abb = sum(step_mean(t, LO1, LO2) for t in MEASURED_ABBHHM)
STEP_LO, STEP_HI = f'{LO1:g}-{LO2:g} GeV', f'{HI1:g}-{HI2:g} GeV'
print(f'    {STEP_LO}, Sigma+ K+ pi- and Sigma- K+ pi+ (mean with the weight 1/E): SAPHIR {step_mean(SAPHIR_SP_MEAN, LO1, LO2):.3f}'
      f' and {step_mean(SAPHIR_SM, LO1, LO2):.3f} mub, ABBHHM (Fig. 29) {step_mean(ABB_SPKPPIM, LO1, LO2):.3f} and {step_mean(ABB_SMKPPIP, LO1, LO2):.3f}')
partners = [r * step_mean(ABB_LK0PIP, LO1, LO2) + 2 * step_mean(CBELSA_K0, LO1, LO2) for r in (0.5, 1.0)]
two_pions = step_mean(ABB_LKPPIPPIM, LO1, LO2) * corrected(*TWO_PIONS) / corrected('Lambda K+ pi+ pi-')
y_lo = [y_meas + x + two_pions for x in partners]
measured_events = corrected('Lambda K0 pi+', 'Sigma+ K+ pi-', 'Sigma- K+ pi+', 'Lambda K+ pi+ pi-', 'Sigma+ K0 pi0 (pi0 ...)')
y_lo_max = y_meas * made['Y'] / measured_events
# upper step: what is left of the events with a hyperon and further particles (up to 5.8 GeV)
y_hi = lambda lower: (data_Y - lnint(m_Y, *M2) - lower * L_LO) / L_HI
code_lo, code_hi = multi(math.sqrt(LO1 * LO2)), multi(math.sqrt(HI1 * HI2))
hyp = lambda c: c[0] + c[1] - c[2] - c[3]
print(f'    {STEP_LO}, with a hyperon and further particles: {y_meas:.1f} in the measured reactions ({measured_events / made["Y"]:.0%} of the events),'
      f' + {partners[0]:.1f}-{partners[1]:.1f} for their three-body partners, + {two_pions:.1f} for the other reactions with two pions:'
      f' {y_lo[0]:.1f}-{y_lo[1]:.1f} (at most {y_lo_max:.1f}); the code {hyp(code_lo):.2f}')
print(f'    {STEP_LO}, with K anti-K: {kk_lo:.2f}; the code {code_lo[2] + code_lo[3]:.2f}')
print(f'    {STEP_HI}, with a hyperon and further particles (the rest of the events up to {ABB_TOP:g} GeV): {y_hi(y_lo[1]):.1f}-{y_hi(y_lo[0]):.1f}'
      f' (at least {y_hi(y_lo_max):.1f}); the code {hyp(code_hi):.2f}')
print(f'    {STEP_HI}, with K anti-K: {kk_hi[0]:.1f}-{kk_hi[1]:.1f}; the code {code_hi[2] + code_hi[3]:.2f}')
y_lo_abb = [y_meas_abb + x + two_pions for x in partners]
print(f'    (with ABBHHM\'s cross sections of Sigma+- K+ pi-+ instead of SAPHIR\'s: {y_meas_abb:.1f} in the measured reactions,'
      f' {y_lo_abb[0]:.1f}-{y_lo_abb[1]:.1f} at {STEP_LO} and {y_hi(y_lo_abb[1]):.1f}-{y_hi(y_lo_abb[0]):.1f} at {STEP_HI})')
# kaons per event of Table VI: with a hyperon; with K anti-K (without further particles; all, without and with the
# two reactions seen only when a K+- decays)
k_y = [kaons['Y'][i] / made['Y'] for i in range(4)]
k_three = [sum(n / v * k[i] for nm, n, k, v, _ in TABLE_VI if nm in THREE) / corrected(*THREE) for i in range(4)]
k_kk = [[kaons['KK'][i] / made['KK'] for i in range(4)],
        [(kaons['KK'][i] + kaons['KK?'][i]) / (made['KK'] + made['KK?']) for i in range(4)]]
print(f'    kaons per event: with a hyperon K+ {k_y[0]:.2f}, K0 {k_y[1]:.2f}; with K anti-K: K+ {k_kk[0][0]:.2f}, K0 {k_kk[0][1]:.2f},'
      f' K- {k_kk[0][2]:.2f}, anti-K0 {k_kk[0][3]:.2f} (without further particles: {k_three[0]:.2f}, {k_three[1]:.2f}, {k_three[2]:.2f}, {k_three[3]:.2f})')
# the K0 among the kaons of the events with a hyperon and further particles, by hyperon (Lambda or Sigma0, Sigma+,
# Sigma-), as tests/support/measured/strange_20gev.py takes them for 20 GeV
def k0_share_of(prefixes):
    k0 = sum(n / v * k[1] for nm, n, k, v, cls in TABLE_VI if cls == 'Y' and nm.startswith(prefixes))
    kk = sum(n / v * (k[0] + k[1]) for nm, n, k, v, cls in TABLE_VI if cls == 'Y' and nm.startswith(prefixes))
    return k0 / kk


k0_by_hyperon = {'Lambda or Sigma0': k0_share_of(('Lambda', 'Sigma0', '(Sigma0, Lambda)')), 'Sigma+': k0_share_of('Sigma+'),
                 'Sigma-': k0_share_of('Sigma-')}
print('    K0 among the kaons of the events with a hyperon and further particles: ' + ', '.join(f'with {h} {v:.2f}' for h, v in k0_by_hyperon.items())
      + f'; with the two-body events too, with Lambda or Sigma0 {sum(n / v * k[1] for nm, n, k, v, cls in TABLE_VI if cls in ("Y", "2") and nm.startswith(("Lambda", "Sigma0", "(Sigma0, Lambda)"))) / sum(n / v * (k[0] + k[1]) for nm, n, k, v, cls in TABLE_VI if cls in ("Y", "2") and nm.startswith(("Lambda", "Sigma0", "(Sigma0, Lambda)"))):.2f}')
# the shares of these events that tests/support/measured/strange_20gev.py takes, to their two decimals
pair_share = lambda k, i: k[i] / (k[i - 1] + k[i])   # K0 among the kaons (i = 1), anti-K0 among the antikaons (i = 3)
for name, value in (('LAMBDA_SHARE_ABBHHM', fl), ('K0_SHARE_LAMBDA', k0_by_hyperon['Lambda or Sigma0']),
                    ('K0_SHARE_SIGMA_PLUS', k0_by_hyperon['Sigma+']), ('K0_SHARE_SIGMA_MINUS', k0_by_hyperon['Sigma-']),
                    ('K0_SHARE', pair_share(k_kk[0], 1)), ('K0_SHARE_ALL', pair_share(k_kk[1], 1)),
                    ('ANTI_K0_SHARE_RANGE', sorted((pair_share(k_kk[0], 3), pair_share(k_kk[1], 3))))):
    assert f'{getattr(g20, name)}' == f'{value:.2f}' if name != 'ANTI_K0_SHARE_RANGE' else \
        [f'{v:.2f}' for v in value] == [f'{v}' for v in g20.ANTI_K0_SHARE_RANGE], (name, value, getattr(g20, name))
lo_species = [[y * k_y[i] + kk_lo * k_three[i] for i in range(4)] for y in y_lo]
hi_species = [[y * k_y[i] + kk * k[i] for i in range(4)] for y, kk, k in ((y_hi(y_lo[1]), kk_hi[0], k_kk[0]), (y_hi(y_lo[0]), kk_hi[1], k_kk[1]))]
for label, d, c in ((STEP_LO, lo_species, code_lo), (STEP_HI, hi_species, code_hi)):
    span = lambda x, y: f'{x:.2f}' if f'{x:.2f}' == f'{y:.2f}' else f'{min(x, y):.2f}-{max(x, y):.2f}'
    print(f'    {label}: ' + ', '.join(f'{name} {span(d[0][i], d[1][i])}' for i, name in enumerate(('K+', 'K0', 'K-', 'anti-K0')))
          + f'; the code {c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f}')
# ABBHHM's two steps: the lower at the middle of its range, the upper with the rest of the events with a hyperon and
# the middle of its K anti-K range. The code takes them times the factors of the three experiments (below).
y_mid = 0.5 * (y_lo[0] + y_lo[1])
mid_lo = [y_mid * k_y[i] + kk_lo * k_three[i] for i in range(4)]
mid_hi = [y_hi(y_mid) * k_y[i] + 0.5 * (kk_hi[0] * k_kk[0][i] + kk_hi[1] * k_kk[1][i]) for i in range(4)]
print(f'    ABBHHM\'s steps, the middle of the ranges [mub] (per interaction, of {SIG3[4]:g} mub):')
for label, d in ((STEP_LO, mid_lo), (STEP_HI, mid_hi)):
    print(f'      {label}: ' + ', '.join(f'{name} {x:.2f} ({x / SIG3[4]:.4f})' for x, name in zip(d, ('K+', 'K0', 'K-', 'anti-K0')))
          + f'; with a hyperon {hyp(d):.2f}, with K anti-K {d[2] + d[3]:.2f}')

# SLAC: (E [GeV], events with a visible strange-particle decay [mub], error); the total cross sections at these energies
# are those of tests/support/measured/strange_20gev.py (SIGMA_TOT_SLAC)
SLAC = [(1.44, 4.4, 0.9), (2.8, 8.1, 0.5), (4.7, 8.7, 0.4), (9.3, 9.8, 0.4)]
print('  SLAC: events with a visible strange-particle decay [mub], measured and from the code\'s kaons')
print(f'    (visible: Lambda -> p pi- {V_LAM:.2f}, K0 -> pi+ pi- {V_K0:.2f}, Sigma+ {V_SIGP:.2f}, Sigma- {V_SIGM:.2f}, K+- {V_KCH}; a hyperon of the events with'
      f' further particles {V_Y:.2f})')


def code_visible(e, mf=multi, v=None):
    """Visible two-body events, events with a hyperon and further particles, and K anti-K events of the code [mub]
    (mf: the kaons of the multi-pion types at the photon energy e; v: the visibilities of Lambda, K0, Sigma+,
    Sigma- and K+-, by default ABBHHM's)."""
    v_lam, v_k0, v_sigp, v_sigm, v_kch = v or (V_LAM, V_K0, V_SIGP, V_SIGM, V_KCH)
    v_y = fl * v_lam + fp * v_sigp + fm * v_sigm
    tb = two_body(e)
    kp, k0, km, k0b = mf(e)
    f0 = k0 / (kp + k0) if kp + k0 else 0.0            # share of K0 among the kaons
    v2 = tb[0] * v_lam + tb[1] * vis(v_sigp, v_k0)
    vy = (kp + k0 - km - k0b) * ((1 - f0) * vis(v_y, v_kch) + f0 * vis(v_y, v_k0))
    # an antikaon with a K+ or a K0 in the proportion of the type's kaons
    vkk = km * ((1 - f0) * vis(v_kch, v_kch) + f0 * vis(v_k0, v_kch)) + k0b * ((1 - f0) * vis(v_k0, v_kch) + f0 * vis(v_k0, v_k0))
    return v2, vy, vkk


for e, visible, err in SLAC:
    v2, vy, vkk = code_visible(e)
    print(f'    {e:5.2f} GeV: measured {visible:.1f} +- {err:.1f}; the code {v2 + vy + vkk:.1f} (two-body {v2:.2f}, with a hyperon and further particles {vy:.2f},'
          f' with K anti-K {vkk:.2f}): {(v2 + vy + vkk) / visible - 1:+.0%}')

# ABBHHM and SLAC against each other, without the code's multiplicities: ABBHHM's events with a visible decay (1185),
# normalised to the visible decays of its two-body events, and the SLAC cross sections integrated over the same 1/E
# spectrum (interpolated linearly in ln E between the measured energies, from zero at the K+ Lambda threshold)
E_THR = threshold(MP, MKP, MLAM)
PTS = [(E_THR, 0.0)] + [(e, v) for e, v, _ in SLAC]


def slac_visible(e):
    for (e1, v1), (e2, v2) in zip(PTS[:-1], PTS[1:]):
        if e1 <= e <= e2:
            return v1 + (v2 - v1) * math.log(e / e1) / math.log(e2 / e1)
    return 0.0


ln2_visible = sum((r[6] * V_LAM + r[7] * vis(V_SIGP, V_K0)) * math.log(r[1] / r[0]) for r in KY)
abbhhm = 1185 * ln2_visible / seen['2']
slac = lnint(slac_visible, E_THR, ABB_TOP)
code = lnint(lambda e: sum(code_visible(e)), E_THR, ABB_TOP)
print(f'  events with a visible decay up to {ABB_TOP:g} GeV, int sigma dE/E: ABBHHM {abbhhm:.1f} mub (1185 events, {seen["2"]} of them two-body for'
      f' {ln2_visible:.2f} mub), SLAC {slac:.1f} mub ({slac / abbhhm:.2f} times as many), the code {code:.1f} mub ({code / abbhhm - 1:+.0%} and {code / slac - 1:+.0%})')
# above 2 GeV: without the two-body events and the events with further particles below 2 GeV (from the code)
E2 = 2.0
below = lnint(lambda e: sum(code_visible(e)), E_THR, E2)
L2 = math.log(ABB_TOP / E2)
print(f'    above {E2:g} GeV (the code\'s {below:.1f} mub below {E2:g} GeV taken off): ABBHHM {abbhhm - below:.1f}, SLAC {lnint(slac_visible, E2, ABB_TOP):.1f},'
      f' the code {lnint(lambda e: sum(code_visible(e)), E2, ABB_TOP):.1f} mub; per unit ln E: {(abbhhm - below) / L2:.1f}, {lnint(slac_visible, E2, ABB_TOP) / L2:.1f}'
      f' and {lnint(lambda e: sum(code_visible(e)), E2, ABB_TOP) / L2:.1f} mub')

# the cross section of gamma p -> p K+ K-, which both experiments give: SLAC (Ballam et al., Table VII: photons of
# 2.4-3.3 and 4.1-5.3 GeV, their note 24) against ABBHHM's Table V over the same photon energies
print('  gamma p -> p K+ K- [mub], SLAC against ABBHHM over the same photon energies:')
SLAC_PKK = [(2.8, 2.4, 3.3, 1.0, 0.1), (4.7, 4.1, 5.3, 0.7, 0.1)]
for e, lo, hi, sig, err in SLAC_PKK:
    parts = [(min(b, hi) - max(a, lo), v, dv) for a, b, v, dv in ABB_PKK if min(b, hi) > max(a, lo)]
    width = sum(w for w, _, _ in parts)
    abb = sum(w * v for w, v, _ in parts) / width
    abb_err = math.sqrt(sum((w * dv) ** 2 for w, _, dv in parts)) / width
    ratio = sig / abb
    print(f'    {e} GeV: SLAC {sig} +- {err}, ABBHHM {abb:.2f} +- {abb_err:.2f}: ratio {ratio:.2f} +- {ratio * math.hypot(err / sig, abb_err / abb):.2f}')
# with pions as well (SLAC; Bingham et al., Table V, at 9.3 GeV): p K+ K-, p K+ K- pi+ pi-, p K+ K- 2pi+ 2pi-
SLAC_CHARGED = [(2.8, (1.0, 0.01, 0.0)), (4.7, (0.7, 0.3, 0.0)), (9.3, (0.58, 0.46, 0.06))]
print('    K+ K- with a proton and charged pions only (SLAC), and the K anti-K events of the code:')
for e, parts in SLAC_CHARGED:
    kk = multi(e)[2] + multi(e)[3]
    print(f'      {e} GeV: {" + ".join(f"{x:g}" for x in parts)} = {sum(parts):.2f}; the code {kk:.2f} ({sum(parts) / kk:.0%} of them)')

# the Lambda that each experiment needs. ABBHHM: its Lambda share of the hyperons with further particles, plus the
# two-body K+ Lambda, K+ Sigma0 (Sigma0 -> Lambda gamma). SLAC: the hyperons its visible decays need with the code's
# K anti-K events. Measured at 20 GeV: Abe et al. 1985 (tests/support/measured/strange_20gev.py).
print(f'  Lambda (with those from Sigma0) [mub]; measured at 20 GeV: {g20.LAMBDA[0]:.2f} +- {g20.LAMBDA[1]:.2f} (Abe et al. 1985;'
      f' {g20.LAMBDA[0] - g20.ANTI_LAMBDA[0]:.2f} less the anti-Lambda; the code {g20.lambda_share() * m_Y(20.0):.2f} with the share of Lambda among the'
      f' hyperons measured there, {g20.lambda_share():.2f}); below 10 GeV with the share of ABBHHM, {fl:.2f}:')
for e, visible, err in SLAC[1:]:
    v2, vy, vkk = code_visible(e)
    need = m_Y(e) * (visible - v2 - vkk) / vy
    print(f'    {e} GeV: the code {fl * m_Y(e) + two_body(e)[0]:.1f}; for the visible decays of SLAC {fl * need + two_body(e)[0]:.1f}'
          f' ({need:.1f} mub of events with a hyperon and further particles instead of {m_Y(e):.1f})')


# SLAC's normalisation against the DESY streamer chamber (tagged photons, 1.6-6.3 GeV; Struczinski et al. 1976,
# HEPData ins100089, Table 1) in the channels without strange particles [mub]. DESY: linear in ln E between its bin
# middles, with its normalisation error (5 %, independent of energy) added. SLAC: Ballam et al. 1972, Table VII.
DESY_NORM = 0.05
DESY_BINS = [(1.6, 1.9), (1.9, 2.1), (2.1, 2.3), (2.3, 2.6), (2.6, 3.25), (3.25, 4.0), (4.0, 5.0), (5.0, 6.3)]
DESY = {  # (values per bin, errors per bin)
    'p pi+ pi-': ([41.7, 39.7, 38.5, 33.8, 28.1, 23.4, 22.5, 19.5], [2.3, 2.4, 2.4, 1.7, 1.6, 1.2, 2.0, 1.8]),
    'p pi+ pi- pi0': ([25.9, 25.0, 26.0, 23.6, 23.2, 18.8, 18.0, 14.4], [1.7, 1.8, 2.1, 1.3, 1.3, 1.4, 1.6, 1.5]),
    'n 2pi+ pi-': ([8.0, 11.4, 9.1, 9.5, 11.3, 9.8, 7.9, 7.7], [0.8, 1.1, 1.0, 0.7, 0.8, 0.9, 1.3, 1.3]),
    '3 prongs, more neutrals': ([17.9, 19.0, 23.6, 27.2, 30.2, 35.0, 29.8, 31.8], [1.1, 1.3, 1.7, 1.3, 1.4, 1.9, 2.3, 2.4]),
    'p 2pi+ 2pi-': ([3.0, 2.9, 2.0, 3.4, 4.5, 4.4, 5.5, 5.4], [1.0, 0.8, 0.6, 0.6, 0.6, 0.5, 0.5, 0.4]),
    'p 2pi+ 2pi- pi0': ([0.0, 0.3, 0.6, 1.4, 2.7, 4.0, 5.7, 7.4], [0.0, 0.2, 0.2, 0.3, 0.5, 0.5, 0.6, 0.6]),
    'n 3pi+ 2pi-': ([0.0, 0.1, 0.2, 0.3, 0.5, 1.5, 2.3, 2.2], [0.0, 0.1, 0.2, 0.1, 0.2, 0.3, 0.3, 0.3]),
    '5 prongs, more neutrals': ([0.3, 0.3, 0.4, 1.3, 2.3, 4.1, 7.1, 10.5], [0.3, 0.2, 0.2, 0.4, 0.5, 0.6, 0.9, 1.1]),
}
SLAC_CHANNELS = {  # 2.8 GeV, 4.7 GeV: (value, error); more neutrals: those with a proton plus those with a neutron
    'p pi+ pi-': ((30.9, 1.2), (20.5, 0.8)),
    'p pi+ pi- pi0': ((24.9, 1.5), (15.1, 1.5)),
    'n 2pi+ pi-': ((10.1, 1.3), (7.2, 2.0)),
    '3 prongs, more neutrals': ((14.0 + 11.2, math.hypot(2.0, 0.9)), (20.8 + 16.3, math.hypot(3.9, 2.3))),
    'p 2pi+ 2pi-': ((4.1, 0.3), (5.1, 0.3)),
    'p 2pi+ 2pi- pi0': ((3.2, 0.3), (7.0, 0.6)),
    'n 3pi+ 2pi-': ((0.4, 0.07), (1.6, 0.5)),
    '5 prongs, more neutrals': ((0.2 + 0.2, math.hypot(0.15, 0.08)), (3.2 + 1.6, math.hypot(0.7, 0.3))),
}


def desy_at(name, e):
    """DESY's cross section [mub] and error at the photon energy e, linear in ln E between the middles of its bins."""
    mid = [math.sqrt(a * b) for a, b in DESY_BINS]
    vals, errs = DESY[name]
    for i in range(len(mid) - 1):
        if mid[i] <= e <= mid[i + 1]:
            w = math.log(e / mid[i]) / math.log(mid[i + 1] / mid[i])
            return (1 - w) * vals[i] + w * vals[i + 1], math.hypot((1 - w) * errs[i], w * errs[i + 1])
    raise ValueError(e)


print('  the normalisation of SLAC: channels without strange particles, SLAC (Ballam et al., Table VII) against the DESY streamer chamber'
      ' (Struczinski et al. 1976, normalised to the published total cross sections, +-5 % included) [mub]:')
for j, e in enumerate((2.8, 4.7)):
    groups = {'3 prongs': [n for n in DESY if n in ('p pi+ pi-', 'p pi+ pi- pi0', 'n 2pi+ pi-', '3 prongs, more neutrals')],
              '5 prongs': [n for n in DESY if n.startswith(('p 2pi', 'n 3pi', '5 prongs'))]}
    parts = []
    for label, names in [('p pi+ pi-', ['p pi+ pi-'])] + list(groups.items()):
        sv = sum(SLAC_CHANNELS[n][j][0] for n in names)
        se = math.sqrt(sum(SLAC_CHANNELS[n][j][1] ** 2 for n in names))
        dv = sum(desy_at(n, e)[0] for n in names)
        de = math.hypot(math.sqrt(sum(desy_at(n, e)[1] ** 2 for n in names)), DESY_NORM * sum(desy_at(n, e)[0] for n in names))
        r = sv / dv
        parts.append(f'{label} {sv:.1f} against {dv:.1f}: {r:.2f} +- {r * math.hypot(se / sv, de / dv):.2f}')
    print(f'    {e} GeV: ' + '; '.join(parts))

# the DESY streamer chamber's events with a visible decay (V or kink) of a strange particle (Struczinski et al. 1976,
# Table 2, p. 52; not in the HEPData record), per bin and normalised as its channels. Lower limits: not corrected for
# the decays outside the visible volume or within the vertex resolution (p. 51), and, as a vertex needs two tracks,
# perhaps without the events whose only charged particle from the vertex comes with a V (most two-body ones). The
# code: averaged over each bin with the weight 1/E, with ABBHHM's visibilities and with no decay lost (branching
# ratios only, K+- still 0.05, and with K+- visible in 0.25).
DESY_VISIBLE = [(1.6, 1.9, 5.2, 0.6), (1.9, 2.1, 5.9, 0.7), (2.1, 2.3, 5.9, 0.7), (2.3, 2.6, 6.9, 0.6),
                (2.6, 3.25, 6.5, 0.5), (3.25, 4.0, 9.1, 0.8), (4.0, 5.0, 8.4, 0.8), (5.0, 6.3, 8.7, 0.9)]
V_NO_LOSS = (BR_LAMBDA, BR_K0_PIPI, 1.0, 1.0, V_KCH)
V_NO_LOSS_K = V_NO_LOSS[:4] + (0.25,)
print('  DESY streamer chamber: events with a visible strange-particle decay [mub], measured (not corrected for the decays'
      ' it could not see) and the code with ABBHHM\'s visibilities / with no decay lost / and K+- visible in 0.25 (two-body part):')
for lo, hi, val, err in DESY_VISIBLE:
    c, c0, ck = (mean(lambda e: sum(code_visible(e, v=v)), lo, hi, 1) for v in (None, V_NO_LOSS, V_NO_LOSS_K))
    tb = mean(lambda e: code_visible(e)[0], lo, hi, 1)
    print(f'    {lo:.2f}-{hi:.2f} GeV: {val:.1f} +- {err:.1f}; the code {c:.1f} ({c / val - 1:+.0%}) / {c0:.1f} ({c0 / val - 1:+.0%})'
          f' / {ck:.1f} ({ck / val - 1:+.0%}) ({tb:.1f})')


def desy_visible(e):
    return next((val for lo, hi, val, _ in DESY_VISIBLE if lo <= e < hi), 0.0)


desy = lnint(desy_visible, E2, ABB_TOP)
desy_err = math.sqrt(sum((err * math.log(min(hi, ABB_TOP) / max(lo, E2))) ** 2 for lo, hi, _, err in DESY_VISIBLE if hi > E2))
c, c0 = lnint(lambda e: sum(code_visible(e)), E2, ABB_TOP), lnint(lambda e: sum(code_visible(e, v=V_NO_LOSS)), E2, ABB_TOP)
print(f'    int sigma dE/E from {E2:g} to {ABB_TOP:g} GeV: DESY {desy:.2f} +- {desy_err:.2f} (stat.) +- {DESY_NORM * desy:.2f} (norm.) mub,'
      f' {desy / (abbhhm - below):.2f} times ABBHHM ({abbhhm - below:.2f}) and {desy / lnint(slac_visible, E2, ABB_TOP):.2f} times SLAC'
      f' ({lnint(slac_visible, E2, ABB_TOP):.2f}); the code {c:.2f} ({c / desy - 1:+.0%}), with no decay lost {c0:.2f} ({c0 / desy - 1:+.0%})')

# the yields of M3 the code takes: ABBHHM's steps (above) times a factor per step. An experiment's factor is its
# events with a visible decay, less the code's two-body ones, over the visible decays of the multi-pion kaons of
# ABBHHM's step (int dE/E over the step, or over the part DESY's bins cover): 1 for ABBHHM, SLAC interpolated in
# ln E, DESY with ABBHHM's visibilities and, as a floor, with no decay lost. The code takes the middle of the range
# of the three factors, or DESY's floor where that is higher, rounded to 0.01.
def abbhhm_multi(e):
    """The kaons of the multi-pion types [mub] with ABBHHM's steps in M3."""
    return tuple(mid_lo) if LO1 <= e < LO2 else tuple(mid_hi) if HI1 <= e < HI2 else multi(e)


def step_factor(measured, parts, v=None):
    two = sum(lnint(lambda e: code_visible(e, abbhhm_multi, v)[0], lo, hi) for lo, hi in parts)
    many = sum(lnint(lambda e: sum(code_visible(e, abbhhm_multi, v)[1:]), lo, hi) for lo, hi in parts)
    return (measured - two) / many


print('  the factors on the multi-pion kaons of ABBHHM\'s steps of M3 from the three experiments, and the code\'s steps:')
M3_FACTOR = {}
for (a, b), mid in zip(M3_STEPS, (mid_lo, mid_hi)):
    row = next(i for i, r in enumerate(MPI) if int(r[0]) == 2 and r[1] == a)
    f_slac = step_factor(lnint(slac_visible, a, b), [(a, b)])
    cover = [(max(lo, a), min(hi, b), val) for lo, hi, val, _ in DESY_VISIBLE if min(hi, b) > max(lo, a)]
    measured = sum(val * math.log(hi / lo) for lo, hi, val in cover)
    f_desy, f_floor = (step_factor(measured, [(lo, hi) for lo, hi, _ in cover], v) for v in (None, V_NO_LOSS))
    span = (min(1.0, f_slac, f_desy), max(1.0, f_slac, f_desy))
    f = M3_FACTOR[a] = round(max(0.5 * sum(span), f_floor), 2)
    want = [round(x * f / SIG3[4], 4) for x in mid]
    have = MPI[row][3:7]
    print(f'    {a}-{b} GeV: ABBHHM 1, SLAC {f_slac:.2f}, DESY {f_desy:.2f} (at least {f_floor:.2f}); the middle {0.5 * sum(span):.2f}:'
          f' the code takes {f:.2f}: ' + ', '.join(f'{w:.4f}' for w in want) + ' per interaction; the code ' + ', '.join(f'{h:.4f}' for h in have)
          + (' (the same)' if all(abs(w - h) < 5e-5 for w, h in zip(want, have)) else ' (DIFFERENT)'))

# the alternative to the two steps of M4, one step for 5-50 GeV with the kaons measured at 20 GeV, against the same data
alt = list(g20.one_step(SIG3[6]))
MPI_ALT = [r for r in MPI if int(r[0]) != 3] + [[3.0, *M4] + alt + list(M4_ROW[7:9]) + [g20.ONE_STEP_F]]


def multi_alt(e):
    """multi(e) with that one step for M4."""
    for row in MPI_ALT:
        if row[1] <= e < row[2]:
            return tuple(SIG3[2 * int(row[0])] * m for m in row[3:7])
    return 0.0, 0.0, 0.0, 0.0


print(f'\nThe alternative: one step for M4 ({M4[0]:g}-{M4[1]:g} GeV) with the values measured at 20 GeV, {", ".join(f"{v:.3f}" for v in alt)} per interaction,'
      ' against the same data (the code\'s two steps in brackets):')
for label, f_code, f_alt, data in (
        ('events with a hyperon and further particles', lambda e: m_Y(e), lambda e: multi_alt(e)[0] + multi_alt(e)[1] - multi_alt(e)[2] - multi_alt(e)[3], (data_Y, data_Y)),
        ('events with K anti-K', lambda e: m_KK(e), lambda e: multi_alt(e)[2] + multi_alt(e)[3], data_KK)):
    c, a = lnint(f_code, 0.9, 5.8), lnint(f_alt, 0.9, 5.8)
    print(f'  ABBHHM, int sigma dE/E up to 5.8 GeV, {label}: {data[0]:.2f}' + (f'-{data[1]:.2f}' if data[1] != data[0] else '')
          + f' mub; one step {a:.2f} ({a / data[1] - 1:+.0%} ... {a / data[0] - 1:+.0%}) [{c:.2f}: {c / data[1] - 1:+.0%} ... {c / data[0] - 1:+.0%}]')
for i, name in enumerate(('K+', 'K0', 'K-', 'anti-K0')):
    d = per_event * (kaons['Y'][i] + kaons['KK'][i] + kaons['KK?'][i])
    a, c = lnint(lambda e: multi_alt(e)[i], 0.9, 5.8), lnint(lambda e: multi(e)[i], 0.9, 5.8)
    print(f'    {name:8s} {d:.2f} mub; one step {a:.2f} ({a / d - 1:+.0%}) [{c:.2f} ({c / d - 1:+.0%})]')
for e, visible, err in SLAC:
    a, c = sum(code_visible(e, multi_alt)), sum(code_visible(e))
    if a != c:
        print(f'  SLAC, events with a visible decay at {e} GeV: measured {visible} +- {err}; one step {a:.1f} ({a / visible - 1:+.0%}) [{c:.1f} ({c / visible - 1:+.0%})]')
a = lnint(lambda e: sum(code_visible(e, multi_alt)), E_THR, 5.8)
print(f'  events with a visible decay up to 5.8 GeV: one step {a:.1f} mub ({a / abbhhm - 1:+.0%} against ABBHHM, {a / slac - 1:+.0%} against SLAC)'
      f' [{code:.1f}: {code / abbhhm - 1:+.0%}, {code / slac - 1:+.0%}]')
