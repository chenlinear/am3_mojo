"""Photo-hadronic kaons, kaons/README.md "Kaons in multi-pion production": the kaons of the multi-pion types M2
(0.9-1.5 GeV) and M3 (1.5-5 GeV) from measured strange-particle photoproduction on protons, and the code's kaons of
M2-M4 against these measurements (M4's come from the measurement at 20 GeV: m4_steps.py). The two-body reactions gamma N
-> K Lambda, K Sigma are channels of their own in the code (two_body.py), so the multi-pion kaons are those made with
further particles. Strangeness balance and the measured strange particles give the multiplicities. eps_r is the photon
energy in the rest frame of the proton.

  * M2 from the three-body reactions. Below the threshold of gamma N -> K Lambda pi (1.15 GeV) only the two-body
    reactions make a kaon, and an antikaon comes only with a kaon pair, from 1.51 GeV on, so M2 makes none
    (kaons/README.md gives these thresholds from its s; kaon_production.threshold_span computes them). Of the three-body
    reactions of protons below 1.5 GeV, SAPHIR measured gamma p -> K+ Sigma+- pi-+ (Schulday et al. 2010,
    arXiv:1011.0823, Tables 2-4; HEPData ins875421). CBELSA/TAPS measured K0 pi0 Sigma+ (Nanova et al. 2008,
    arXiv:0803.2146, Fig. 13; HEPData ins781382), and ABBHHM (the Aachen-Berlin-Bonn-Hamburg-Heidelberg-Muenchen
    collaboration, Erbe et al. 1969, Phys. Rev. 188, 2060) measured K0 Lambda pi+ (Fig. 29). The unmeasured partners of
    these reactions are estimated from them: K+ Lambda pi0 as 0.5 to 1 times K0 Lambda pi+, and K+ Sigma0 pi0 and K0
    Sigma0 pi+ each as K0 pi0 Sigma+. ABBHHM give a single value, 0.74 +- 0.34 mubarn for 1.2-1.7 GeV. It is counted
    either evenly over that range or only above 1.42 GeV, the threshold of Sigma(1385) K. The three-body cross sections
    of protons are then averaged over the photon energies from 1.15 to 1.5 GeV, with the weight 1, 1/eps_r or 1/eps_r^2.
    These averages add up to 0.5-1.4 mubarn. The sum has an error of about 40 %, nearly all of it the statistical error
    of ABBHHM's K0 Lambda pi+. For each species, the code takes the average with the weight 1/eps_r, that is, over ln
    eps_r. It then takes the middle of the range that the two choices above give: 0.43 mubarn of K+ and 0.51 mubarn of
    K0. Divided by the cross section of M2, 85 mubarn, these give the row of M2: 0.0051 K+ and 0.0060 K0 per
    interaction.
  * M3 from ABBHHM's events. ABBHHM's hydrogen bubble chamber was exposed to a bremsstrahlung beam of photons up to
    5.8 GeV. It found 1312 events with strange particles (Table VI), 1185 of them with a visible strange-particle decay.
    The beam's photon spectrum falls about as 1/E with the photon energy E, the eps_r of a proton at rest. So the events
    of a reaction count the integral int sigma dE/E of its cross section sigma. Corrected for the decays not seen
    (p. 2074), the two-body events match the code's two-body steps at 145 events per mubarn of int sigma dE/E. The
    paper's own cross sections of five reactions give 129-182 events per mubarn. The events with a hyperon and further
    particles then correspond to int sigma dE/E = 7.1 mubarn up to 5.8 GeV. Those with a kaon pair correspond to
    3.2-3.8 mubarn. One multiplicity for all of M3 could give these totals, but not their rise across the type, so M3
    has two steps, 1.5-2.5 and 2.5-5 GeV.
  * M3's two steps. As for M2, a step's value is the mean over its photon energies with the weight 1/eps_r. The events
    with a kaon pair follow the cross section of gamma p -> p K+ K- (ABBHHM's Table V). This cross section is scaled by
    the number of all such events per p K+ K- event. The lower step counts only the reactions without a pion, as those
    with one start at 1.8 GeV. So these events give 0.6 mubarn in the lower step and 3.4-4.1 mubarn in the upper one.
    For the events with a hyperon and further particles at 1.5-2.5 GeV, the reactions with measured cross sections
    (ABBHHM's read off its Fig. 29) make 2.2 mubarn. Here SAPHIR's Sigma+- K+ pi-+ replaces ABBHHM's. In ABBHHM's
    events, these reactions are 35 % of all events with a hyperon and further particles. The CBELSA/TAPS cross section
    is counted together with the third of their K*0 Sigma+ that decays into K0 pi0 (K0PI0). Their paper gives that
    channel separately above 1.85 GeV. The K*0 is an excited neutral kaon. With the unmeasured partners and the other
    reactions with two pions, these events give 3.9-4.4 mubarn in the lower step. The rest of them, from 2.5 up to
    5.8 GeV, give 5.5-5.8 mubarn in the upper step. With ABBHHM's kaons per event of each kind (Table VI), the middles
    of these ranges give ABBHHM's steps. At 1.5-2.5 GeV, these are 2.76, 2.05, 0.30 and 0.34 mubarn of K+, K0, K- and
    anti-K0. At 2.5-5 GeV, they are 5.32, 4.08, 1.95 and 1.83 mubarn.
  * The factors of M3 from SLAC and DESY. Two other experiments saw more events with strange particles than ABBHHM. The
    SLAC 82-inch bubble chamber, with photons of one energy, measured the cross section of the events with a visible
    decay of a strange particle. It is 4.4 +- 0.9, 8.1 +- 0.5 and 8.7 +- 0.4 mubarn at 1.44, 2.8 and 4.7 GeV
    (Ballam et al. 1972, Phys. Rev. D 5, 545, Table V). At 9.3 GeV, it is 9.8 +- 0.4 mubarn (Bingham et al. 1973, Phys.
    Rev. D 8, 1277, Table IV). Integrated over ABBHHM's photon spectrum, these cross sections give 1.36 times the 1185
    events with a visible decay that ABBHHM found. Above 2 GeV, they give about 1.6 times as many. The DESY streamer
    chamber, with tagged photons, gives lower limits in eight bins from 1.6 to 6.3 GeV (Struczinski et al. 1976, Nucl.
    Phys. B 108, 45, Table 2, p. 52, not in the HEPData record). From 2 to 5.8 GeV, its events give int sigma dE/E =
    8.2 +- 0.3 (statistical) +- 0.4 (normalisation) mubarn. This is 1.5 times ABBHHM's value and 0.95 times SLAC's. The
    overall normalisations of the chambers differ less. In gamma p -> p K+ K-, SLAC has 1.42 +- 0.26 and 0.69 +- 0.15
    times ABBHHM's cross section at 2.8 and 4.7 GeV (Ballam et al., Table VII; at 9.3 GeV, Bingham et al., Table V).
    SLAC's channels without strange particles agree with DESY's (HEPData ins100089, Table 1, the paper's Table 3, p. 53,
    normalised to a fit of the published total cross sections, +-5 %) to 0.98 +- 0.06 and 1.04 +- 0.09. So the chambers
    differ in how they found and counted the events with strange particles. The Lambda measured at 20 GeV speak for
    ABBHHM, because they come from the fragmentation of the target. Their rate changes little with energy (Abe et al.
    1984, Phys. Rev. D 29, 1877, Sec. 3). ABBHHM's steps give no more of them at 2.8 and 4.7 GeV than the
    5.6 +- 0.2 mubarn measured at 20 GeV (Abe et al. 1985; tests/support/measured/strange_20gev.py). SLAC's visible
    decays would need 7.4, 7.8 and 6.8 mubarn of them at 2.8, 4.7 and 9.3 GeV.
  * The code's factors. The code takes ABBHHM's mix of reactions and kaon species, and the number of kaons from all
    three experiments. An experiment's factor on ABBHHM's multi-pion kaons is a ratio of visible decays, each integrated
    over the step with the weight 1/eps_r. Its numerator is the experiment's visible decays less those of the code's
    two-body kaons. Its denominator is the visible decays of ABBHHM's multi-pion kaons and of the hyperons that come
    with them. In the two steps, the factor is 1.38 and 1.53 for SLAC and 1 for ABBHHM. For DESY it is 1.37 and 1.45,
    and at least 1.21 and 1.32 if no decay is lost. The code takes the middle of the range of these three factors, 1.19
    and 1.27, raised to the least that DESY allows: 1.21 and 1.32. Multiplied by these factors and divided by M3's cross
    section of 120 mubarn, ABBHHM's steps give the rows of M3. Up to 5.8 GeV, the code then has 33 %, 37 %, 32 % and 60
    % more K+, K0, K- and anti-K0 than ABBHHM's events give. These kaons give 25 % more visible decays than ABBHHM found
    and 9 % fewer than the SLAC cross sections. At SLAC's four energies, they and the hyperons that balance their
    strangeness give 3.6, 7.6, 7.1 and 9.4 mubarn of visible decays. These are 19 %, 6 %, 18 % and 4 % below SLAC. From
    2 to 5.8 GeV they give 7.5 mubarn, 8 % below DESY.

The code's tables are read from src/ by code_tables.py and its Lambda decays by tests/support/code_decays.py."""
import math, os, sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from code_tables import code_it3, code_mpi, two_body_steps
from support.code_decays import code_short_lived, key
from support.measured import strange_20gev as g20
from support.paths import load_derivation

derivation = load_derivation('kaon_production')   # the code's masses (PDG 2026), the s of kaons/README.md
kaon_decays = load_derivation('kaon_decays')       # the branching ratios of the kaon decays

# masses [GeV]
MP, MKP, MK0, MLAM, MSIG1385 = (derivation.MASS[q] for q in ("p", "K+", "K0", "Lambda", "Sigma*+"))
threshold = derivation.threshold   # threshold photon energy [GeV] of gamma N -> the given masses

SIG3, EM3, EX3 = (code_it3(n) for n in ('IT3sig', 'IT3Em', 'IT3Ex'))
KY = two_body_steps()            # [emin, emax, chi x 4, sigma of p -> K+, p -> K0, n -> K+, n -> K0]
MPI = code_mpi()                 # [t, eps_min, eps_max, K+, K0, K-, anti-K0, h+, h-, f]:
                                 # steps of type M(t + 1), protons
START = min(r[1] for r in MPI)   # the photon energy [GeV] from which the multi-pion types make kaons
M2, M3, M4 = ((EM3[2 * t], EX3[2 * t]) for t in (1, 2, 3))     # photon energy ranges [GeV]
M3_STEPS = sorted(tuple(r[1:3]) for r in MPI if int(r[0]) == 2)   # the code's two steps of M3
ABB_TOP = 5.8                    # top of ABBHHM's photon spectrum [GeV]
SPECIES = ('K+', 'K0', 'K-', 'anti-K0')
fmt = lambda v, d=2: ', '.join(f'{x:.{d}f}' for x in v)


def step(table, e, col=2):
    """Column col of the step (E_low, E_high, ...) of table that holds e, 0 outside them."""
    for r in table:
        if r[0] <= e < r[1]:
            return r[col]
    return 0.0


def two_body(e):
    """(K+, K0) cross sections [mub] of the two-body steps for protons at the photon energy e [GeV]."""
    return step(KY, e, 6), step(KY, e, 7)


def multi(e):
    """(K+, K0, K-, anti-K0) cross sections [mub] of the multi-pion types for protons at the photon energy e."""
    for row in MPI:
        if row[1] <= e < row[2]:
            return tuple(SIG3[2 * int(row[0])] * m for m in row[3:7])
    return 0.0, 0.0, 0.0, 0.0


hyp = lambda c: c[0] + c[1] - c[2] - c[3]   # of (K+, K0, K-, anti-K0): the kaons with a hyperon
kkbar = lambda c: c[2] + c[3]                # and those with K anti-K


def mean(f, a, b, alpha):
    """Mean of f over the photon energies a ... b (nucleon rest frame) with the weight e^-alpha, that of e in the
    interaction rate of a nucleon in a photon field n(eps) ~ eps^-alpha."""
    s = w = 0.0
    for i in range(6000):
        e = a + (b - a) * (i + 0.5) / 6000
        s += f(e) * e ** (-alpha)
        w += e ** (-alpha)
    return s / w


def lnint(f, a, b):
    """int f dE/E from a to b."""
    return mean(f, a, b, 1) * math.log(b / a)


def steps_lnint(table):
    """int sigma dE/E of a table of steps (E_low, E_high, sigma, ...)."""
    return sum(r[2] * math.log(r[1] / r[0]) for r in table)


# gamma N -> K Lambda pi in its four charge states, and gamma p -> K0 Sigma+(1385)
E3 = derivation.threshold_span(derivation.NUCLEON, derivation.KAON, ("Lambda",), derivation.PION)[0]
E_Y1385 = threshold(MP, MK0, MSIG1385)

print(f'== M2 ({M2[0]:g}-{M2[1]:g} GeV). Below {E3:.3f} GeV only the two-body reactions make a kaon.')
m2 = multi(0.5 * (START + M2[1]))

# cross sections [mub] of reactions with further particles: (E_low, E_high [GeV], sigma ..., error ...)
# SAPHIR (Schulday et al. 2010, HEPData ins875421; the errors include that of the background subtraction).
# gamma p -> K+ Sigma- pi+ (HEPData Table 1, the paper's Table 2):
SAPHIR_SM = [(1.3, 1.4, 0.0028, 0.0026), (1.4, 1.5, 0.029, 0.0061), (1.5, 1.6, 0.1121, 0.0095),
             (1.6, 1.7, 0.1322, 0.0137), (1.7, 1.8, 0.1597, 0.0113), (1.8, 1.9, 0.1894, 0.0152),
             (1.9, 2.0, 0.179, 0.0152), (2.0, 2.1, 0.2322, 0.0141), (2.1, 2.2, 0.2761, 0.0177),
             (2.2, 2.3, 0.2727, 0.0179), (2.3, 2.4, 0.2865, 0.0194), (2.4, 2.5, 0.2595, 0.03),
             (2.5, 2.6, 0.3339, 0.0243)]
# gamma p -> K+ Sigma+ pi-, the whole cross section from Sigma+ -> n pi+ and from Sigma+ -> p pi0, then their errors
# (HEPData Table 2, the paper's Tables 3-4; the last p pi0 error is printed as 0, a bin not used here):
SAPHIR_SP = [(1.3, 1.4, 0.0116, 0.0067, 0.0076, 0.0113), (1.4, 1.5, 0.072, 0.0367, 0.0191, 0.0347),
             (1.5, 1.6, 0.1644, 0.1075, 0.0275, 0.0538), (1.6, 1.7, 0.2314, 0.3395, 0.0419, 0.0496),
             (1.7, 1.8, 0.4005, 0.3817, 0.0359, 0.0519), (1.8, 1.9, 0.5377, 0.5983, 0.0448, 0.0605),
             (1.9, 2.0, 0.7802, 0.7209, 0.0559, 0.0662), (2.0, 2.1, 0.6789, 0.7605, 0.0641, 0.0794),
             (2.1, 2.2, 0.6861, 0.9128, 0.065, 0.0662), (2.2, 2.3, 0.8147, 0.9323, 0.0726, 0.0816),
             (2.3, 2.4, 0.9029, 0.8905, 0.0894, 0.1151), (2.4, 2.5, 0.6359, 0.7702, 0.0806, 0.1115),
             (2.5, 2.6, 0.7302, 0.9037, 0.0868, 0.0)]
# its mean over the two Sigma+ decays
SAPHIR_SP_MEAN = [(lo, hi, 0.5 * (a + b), 0.5 * math.hypot(ea, eb)) for lo, hi, a, b, ea, eb in SAPHIR_SP]
# CBELSA/TAPS (Nanova et al. 2008, HEPData ins781382; statistical errors): gamma p -> K0 pi0 Sigma+ (Table 13, the
# paper's Fig. 13), without K*0 Sigma+ above 1.85 GeV (sigma_Sigma - sigma_SigmaK*, the paper's Sec. 6.2 and the
# counts of its Table 1), and gamma p -> K*0 Sigma+ (Table 12), a third of which decays into K0 pi0:
CBELSA_K0PI0 = [(1.4, 1.5, 0.085, 0.034), (1.5, 1.7, 0.0975, 0.0224), (1.7, 1.85, 0.1511, 0.0294),
                (1.85, 2.0, 0.1734, 0.0445), (2.0, 2.15, 0.2743, 0.0457), (2.15, 2.3, 0.2774, 0.0703),
                (2.3, 2.5, 0.2496, 0.051)]
CBELSA_KSTAR = [(1.85, 2.0, 0.2648, 0.0576), (2.0, 2.15, 0.4414, 0.0883), (2.15, 2.3, 0.5301, 0.0846),
                (2.3, 2.5, 0.5192, 0.0762)]
# the whole final state K0 pi0 Sigma+ (E_low, E_high, sigma, error), errors added as independent: above 1.85 GeV,
# where Tables 13 and 12 are anti-correlated, this overstates the error (not used there)
K0PI0 = 1 / 3   # share of the K*0 decaying into K0 pi0 (isospin: the rest go into K+ pi-)
CBELSA_K0 = [(lo, hi, s + sum(k * K0PI0 for l2, h2, k, _ in CBELSA_KSTAR if (l2, h2) == (lo, hi)),
              math.hypot(e, sum(ek * K0PI0 for l2, h2, _, ek in CBELSA_KSTAR if (l2, h2) == (lo, hi))))
             for lo, hi, s, e in CBELSA_K0PI0]
# ABBHHM (Erbe et al. 1969), Fig. 29, read off to about 0.02 mub: (E_low, E_high, sigma, error). Read on its own
# tick marks, the Lambda K0 pi+ panel puts its threshold arrow at 1.17-1.19 GeV (true: 1.168 GeV).
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
    K+ Sigma+- pi-+ (SAPHIR), K0 pi0 Sigma+ (CBELSA/TAPS; the unmeasured K+ Sigma0 pi0 and K0 Sigma0 pi+ each taken
    equal to it), K0 Lambda pi+ = lam(e) (from ABBHHM's value for 1.2-1.7 GeV) and K+ Lambda pi0 = r x that."""
    k0s = step(CBELSA_K0, e)
    return step(SAPHIR_SP_MEAN, e) + step(SAPHIR_SM, e) + k0s + r * lam(e), 2 * k0s + lam(e)


lo, hi, sig, err = ABB_LK0PIP[0]
# the same int sigma dE/E, flat or only above the Sigma+(1385) K0 threshold: most of the reaction goes through that
# channel there (the paper's Table VII: 0.68 +- 0.48 mub for 1.42-2.0 GeV)
flat = lambda e: sig if lo <= e < hi else 0.0
late = lambda e: sig * math.log(hi / lo) / math.log(hi / E_Y1385) if E_Y1385 <= e < hi else 0.0
print(f'  measured three-body production on protons, mean over {START:g}-{M2[1]:g} GeV [mub].'
      f' K0 Lambda pi+ = {sig:.2f} +- {err:.2f} mub for')
print(f'  {lo}-{hi} GeV (ABBHHM), taken as constant over that bin or as zero below the Sigma(1385) K threshold,'
      f' {E_Y1385:.2f} GeV;')
print('  K+ Lambda pi0 = 0.5 ... 1 x K0 Lambda pi+:')


def three_body_errors(r, lam, alpha):
    """Errors of the mean of sum(three_body) over START-1.5 GeV with the weight e^-alpha: in all, and of SAPHIR's
    and CBELSA/TAPS's reactions alone. Their bins are independent; K0 pi0 Sigma+ counts three times (with the two
    reactions taken equal to it). K0 Lambda pi+ and K+ Lambda pi0 = r x that have ABBHHM's relative error, fully
    correlated."""
    w = lambda lo, hi: mean(lambda e: 1.0 if lo <= e < hi else 0.0, START, M2[1], alpha)
    var = sum((w(lo, hi) * de) ** 2 for lo, hi, _, de in SAPHIR_SM)
    var += sum((w(lo, hi) * de) ** 2 for lo, hi, _, de in SAPHIR_SP_MEAN)
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
    print(f"    weight e^-{alpha}: {min(vals):.2f}-{max(vals):.2f}, error {min(rel):.0%}-{max(rel):.0%}"
          f" (ABBHHM's K0 Lambda pi+ alone {err / sig:.0%}; SAPHIR's and CBELSA/TAPS's reactions {max(others):.2f}"
          f" mub); the code (M2): K+ {m2[0]:.3f}, K0 {m2[1]:.3f}, together {m2[0] + m2[1]:.3f}")
# the code's M2 (weight 1/E, as for the steps of M3)
kp_vals, k0_vals = ([mean(lambda e: three_body(e, r, lam)[i], START, M2[1], 1) for lam in (late, flat)
                     for r in (0.5, 1.0)] for i in (0, 1))
m2_mid = [0.5 * (min(v) + max(v)) for v in (kp_vals, k0_vals)]
print(f'    the middle with the weight 1/E, which the code takes: K+ {min(kp_vals):.3f}-{max(kp_vals):.3f}, middle'
      f' {m2_mid[0]:.3f} ({m2_mid[0] / SIG3[2]:.4f} per interaction of {SIG3[2]:g} mub); K0'
      f' {min(k0_vals):.3f}-{max(k0_vals):.3f}, middle {m2_mid[1]:.3f} ({m2_mid[1] / SIG3[2]:.4f}); together'
      f' {sum(m2_mid):.3f}')

print(f'\n== M3 ({M3[0]:g}-{M3[1]:g} GeV) and M4 ({M4[0]:g}-{M4[1]:g} GeV)')
# visibility of a decay in the bubble chamber: the branching ratio into charged particles (Lambda -> p pi-, the
# code's, physics/decays.mojo; a K0 is a K0_S half the time, K0_S -> pi+ pi-, as in kaons/README.md) over ABBHHM's
# correction for decays outside the chamber or too close to the vertex. V_KCH: few K+- decay in the chamber (an
# estimate).
LAMBDA_MODES = code_short_lived()["Lambda"]
BR_LAMBDA = float(sum(md["br"] for md in LAMBDA_MODES if md["fs"] == key(["p", "pi-"]))
                  / sum(md["br"] for md in LAMBDA_MODES))   # the code renormalises the table to its sum
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
# K anti-K). p K+ K- and p K+ K- pi+ pi- include the events without a visible decay (107 and 20), found by
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
    ('p K+ anti-K0 pi-', 8, (1, 0, 0, 1), vis(V_K0, V_KCH), 'KK'),
    ('p K0 K- pi+', 19, (0, 1, 1, 0), vis(V_K0, V_KCH), 'KK'),
    ('Lambda K+ pi+ pi-', 38, (1, 0, 0, 0), vis(V_LAM), 'Y'),
    ('Sigma+ K0 pi+ pi-', 13, (0, 1, 0, 0), vis(V_SIGP, V_K0), 'Y'),
    ('Sigma0 K+ pi+ pi-', 6, (1, 0, 0, 0), vis(V_LAM), 'Y'),
    ('Sigma- K0 pi+ pi+', 10, (0, 1, 0, 0), vis(V_SIGM, V_K0), 'Y'),
    ('p K0 anti-K0 pi0 (pi0 ...)', 5, (0, 1, 0, 1), vis(V_K0, V_K0), 'KK'),
    ('n K+ K- pi+ (pi0 ...)', 7, (1, 0, 1, 0), vis(V_KCH, V_KCH), 'KK?'),   # seen only when a K+- decays in the chamber
    ('n K0 anti-K0 pi+ (pi0 ...)', 12, (0, 1, 0, 1), vis(V_K0, V_K0), 'KK'),
    ('Sigma+ K+ pi- pi0 (pi0 ...)', 31, (1, 0, 0, 0), vis(V_SIGP), 'Y'),
    ('(Sigma0, Lambda) K0 pi+ pi0 (pi0 ...)', 107, (0, 1, 0, 0), vis(V_LAM, V_K0), 'Y'),
    ('Sigma- K+ pi+ pi0 (pi0 ...)', 13, (1, 0, 0, 0), vis(V_SIGM), 'Y'),
    ('Xi- K+ K0 pi+ (pi0 ...)', 1, (1, 1, 0, 0), 0.95, 'Y'),   # printed Xi- K+ anti-K0 pi+, which violates strangeness
    ('p K+ K- pi+ pi-', 5 + 20, (1, 0, 1, 0), 1.0, 'KK'),
    ('Lambda K0 pi+ pi+ pi-', 7, (0, 1, 0, 0), vis(V_LAM, V_K0), 'Y'),
    ('Sigma- K+ pi+ pi+ pi-', 1, (1, 0, 0, 0), vis(V_SIGM), 'Y'),
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
corrected = lambda *names: sum(n / v for nm, n, _, v, _ in TABLE_VI if nm in names)
print(f'  ABBHHM, Table VI: {sum(seen.values())} events: two-body {seen["2"]}, with a hyperon and further particles'
      f' {seen["Y"]}, with K anti-K {seen["KK"] + seen["KK?"]} ({seen["KK?"]} of them in two reactions seen only'
      ' when a K+- decays)')
print(f'    corrected for the decays that are not seen: {made["2"]:.0f}, {made["Y"]:.0f} and {made["KK"]:.0f}'
      f' (+ {made["KK?"]:.0f} for those two reactions)')
ln2 = sum((r[6] + r[7]) * math.log(r[1] / r[0]) for r in KY)
per_event = ln2 / made['2']
print(f'    int sigma dE/E of the code\'s two-body steps (protons): {ln2:.2f} mub,'
      f' so {1 / per_event:.0f} events per mub')
ABB_FIG = [('Lambda K0 pi+', ABB_LK0PIP), ('Sigma+ K+ pi-', ABB_SPKPPIM), ('Sigma- K+ pi+', ABB_SMKPPIP),
           ('Lambda K+ pi+ pi-', ABB_LKPPIPPIM), ('p K+ K-', ABB_PKK)]   # those with cross sections (Fig. 29, Table V)
for name, table in ABB_FIG:
    ev = corrected(name)
    print(f"      (the paper's own cross section of {name}: int sigma dE/E = {steps_lnint(table):.2f} mub for {ev:.0f}"
          f' events: {ev / steps_lnint(table):.0f} per mub)')
data_Y = per_event * made['Y']
data_KK = (per_event * made['KK'], per_event * (made['KK'] + made['KK?']))
m_Y = lambda e: hyp(multi(e))
m_KK = lambda e: kkbar(multi(e))
code_Y, code_KK = lnint(m_Y, M2[0], ABB_TOP), lnint(m_KK, M2[0], ABB_TOP)
print(f'    int sigma dE/E up to {ABB_TOP:g} GeV, with a hyperon and further particles: {data_Y:.2f} mub; the code'
      f' {code_Y:.2f} ({code_Y / data_Y - 1:+.0%}) (M2 {lnint(m_Y, *M2):.2f}, M3 {lnint(m_Y, *M3):.2f}, M4 up to'
      f' {ABB_TOP:g} GeV {lnint(m_Y, M4[0], ABB_TOP):.2f})')
print(f'    with K anti-K: {data_KK[0]:.2f}-{data_KK[1]:.2f} mub; the code {code_KK:.2f}'
      f' ({code_KK / data_KK[0] - 1:+.0%} ... {code_KK / data_KK[1] - 1:+.0%})'
      f' (M3 {lnint(m_KK, *M3):.2f}, M4 up to {ABB_TOP:g} GeV {lnint(m_KK, M4[0], ABB_TOP):.2f})')
for i, name in enumerate(SPECIES):
    d = per_event * (kaons['Y'][i] + kaons['KK'][i] + kaons['KK?'][i])
    c = lnint(lambda e: multi(e)[i], M2[0], ABB_TOP)
    print(f'      {name:8s} {d:.2f} mub; the code {c:.2f} ({c / d - 1:+.0%})')
LAMBDA_OR_SIGMA0 = ('Lambda', 'Sigma0', '(Sigma0, Lambda)')
lam, sgp, sgm = (sum(n / v for nm, n, _, v, cls in TABLE_VI if cls == 'Y' and nm.startswith(h))
                 for h in (LAMBDA_OR_SIGMA0, 'Sigma+', 'Sigma-'))
fl, fp, fm = (x / (lam + sgp + sgm) for x in (lam, sgp, sgm))
print(f'    hyperons of the events with further particles: Lambda or Sigma0 {fl:.2f}, Sigma+ {fp:.2f}, Sigma- {fm:.2f}')
V_Y = fl * V_LAM + fp * V_SIGP + fm * V_SIGM

# the energy dependence (Fig. 29, Table V)
print('    the five reactions with further particles whose cross sections the paper gives, together [mub]:')
for lo, hi in [(1.25, 1.5), (1.5, 1.75), (1.75, 2.05), (2.05, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 5.8)]:
    e = 0.5 * (lo + hi)
    s5 = sum(step(t, e) for _, t in ABB_FIG)
    print(f'      {lo:.2f}-{hi:.2f} GeV: {s5:.2f}')
five = corrected(*(nm for nm, _ in ABB_FIG))
print(f'      (they are {five / (made["Y"] + made["KK"] + made["KK?"]):.0%} of the events with further particles)')

print('  the two steps of M3 from ABBHHM [mub]:')
(LO1, LO2), (HI1, HI2) = M3_STEPS
L_LO, L_HI = math.log(LO2 / LO1), math.log(ABB_TOP / HI1)
step_mean = lambda table, a, b: lnint(lambda e: step(table, e), a, b) / math.log(b / a)
# with K anti-K: p K+ K- (Table V) times the K anti-K events per p K+ K- event of Table VI; in the lower step only
# those without further particles (p K+ K-, p K0 anti-K0, n K+ anti-K0), as those with a pion start at 1.8 GeV
THREE = ('p K+ K-', 'p K0 anti-K0 (pi0 ...)', 'n K+ anti-K0 (pi0 ...)')
kk_lo = step_mean(ABB_PKK, LO1, LO2) * corrected(*THREE) / corrected('p K+ K-')
kk_hi = [step_mean(ABB_PKK, HI1, HI2) * x / corrected('p K+ K-') for x in (made['KK'], made['KK'] + made['KK?'])]
# with a hyperon, lower step: (a) the measured reactions (Lambda K0 pi+ and Lambda K+ pi+ pi- of ABBHHM's Fig. 29,
# Sigma+- K+ pi-+ of SAPHIR, K0 pi0 Sigma+ of CBELSA/TAPS); (b) their unmeasured three-body partners
# (K+ Lambda pi0 = 0.5 ... 1 x K0 Lambda pi+; K+ Sigma0 pi0 and K0 Sigma0 pi+ as K0 pi0 Sigma+); (c) the other
# reactions with two pions, as Lambda K+ pi+ pi- times their events in Table VI over its own (those with three, from
# 1.7-1.9 GeV on, left out). Upper limit: the measured reactions scaled to all events with a hyperon, as if all
# rose as early. SAPHIR's Sigma+- K+ pi-+ cover the step in ten bins (2-3 % errors on the step means), ABBHHM's
# in two (21-23 %).
MEASURED = (ABB_LK0PIP, SAPHIR_SP_MEAN, SAPHIR_SM, ABB_LKPPIPPIM, CBELSA_K0)
MEASURED_ABBHHM = (ABB_LK0PIP, ABB_SPKPPIM, ABB_SMKPPIP, ABB_LKPPIPPIM, CBELSA_K0)
TWO_PIONS = ('Sigma+ K0 pi+ pi-', 'Sigma0 K+ pi+ pi-', 'Sigma- K0 pi+ pi+', 'Sigma+ K+ pi- pi0 (pi0 ...)',
             '(Sigma0, Lambda) K0 pi+ pi0 (pi0 ...)', 'Sigma- K+ pi+ pi0 (pi0 ...)')
assert all(any(nm == name and cls == 'Y' for nm, _, _, _, cls in TABLE_VI) for name in TWO_PIONS)
y_meas = sum(step_mean(t, LO1, LO2) for t in MEASURED)
y_meas_abb = sum(step_mean(t, LO1, LO2) for t in MEASURED_ABBHHM)
STEP_LO, STEP_HI = f'{LO1:g}-{LO2:g} GeV', f'{HI1:g}-{HI2:g} GeV'
print(f'    {STEP_LO}, Sigma+ K+ pi- and Sigma- K+ pi+ (mean with the weight 1/E): SAPHIR'
      f' {step_mean(SAPHIR_SP_MEAN, LO1, LO2):.3f} and {step_mean(SAPHIR_SM, LO1, LO2):.3f} mub, ABBHHM (Fig. 29)'
      f' {step_mean(ABB_SPKPPIM, LO1, LO2):.3f} and {step_mean(ABB_SMKPPIP, LO1, LO2):.3f}')
partners = [r * step_mean(ABB_LK0PIP, LO1, LO2) + 2 * step_mean(CBELSA_K0, LO1, LO2) for r in (0.5, 1.0)]
two_pions = step_mean(ABB_LKPPIPPIM, LO1, LO2) * corrected(*TWO_PIONS) / corrected('Lambda K+ pi+ pi-')
y_lo = [y_meas + x + two_pions for x in partners]
measured_events = corrected('Lambda K0 pi+', 'Sigma+ K+ pi-', 'Sigma- K+ pi+', 'Lambda K+ pi+ pi-',
                            'Sigma+ K0 pi0 (pi0 ...)')
y_lo_max = y_meas * made['Y'] / measured_events
# upper step: the rest of the events with a hyperon and further particles (up to 5.8 GeV)
y_hi = lambda lower: (data_Y - lnint(m_Y, *M2) - lower * L_LO) / L_HI
code_lo, code_hi = multi(math.sqrt(LO1 * LO2)), multi(math.sqrt(HI1 * HI2))
print(f'    {STEP_LO}, with a hyperon and further particles: {y_meas:.1f} in the measured reactions'
      f' ({measured_events / made["Y"]:.0%} of the events), + {partners[0]:.1f}-{partners[1]:.1f} for their three-body'
      f' partners, + {two_pions:.1f} for the other reactions with two pions: {y_lo[0]:.1f}-{y_lo[1]:.1f}'
      f' (at most {y_lo_max:.1f}); the code {hyp(code_lo):.2f}')
print(f'    {STEP_LO}, with K anti-K: {kk_lo:.2f}; the code {kkbar(code_lo):.2f}')
print(f'    {STEP_HI}, with a hyperon and further particles (the rest of the events up to {ABB_TOP:g} GeV):'
      f' {y_hi(y_lo[1]):.1f}-{y_hi(y_lo[0]):.1f} (at least {y_hi(y_lo_max):.1f}); the code {hyp(code_hi):.2f}')
print(f'    {STEP_HI}, with K anti-K: {kk_hi[0]:.1f}-{kk_hi[1]:.1f}; the code {kkbar(code_hi):.2f}')
y_lo_abb = [y_meas_abb + x + two_pions for x in partners]
print(f'    (with ABBHHM\'s cross sections of Sigma+- K+ pi-+ instead of SAPHIR\'s: {y_meas_abb:.1f} in the measured'
      f' reactions, {y_lo_abb[0]:.1f}-{y_lo_abb[1]:.1f} at {STEP_LO} and'
      f' {y_hi(y_lo_abb[1]):.1f}-{y_hi(y_lo_abb[0]):.1f} at {STEP_HI})')
# kaons per event of Table VI: with a hyperon; with K anti-K (without further particles; all, without and with the
# two reactions seen only when a K+- decays)
k_y = [kaons['Y'][i] / made['Y'] for i in range(4)]
k_three = [sum(n / v * k[i] for nm, n, k, v, _ in TABLE_VI if nm in THREE) / corrected(*THREE) for i in range(4)]
k_kk = [[kaons['KK'][i] / made['KK'] for i in range(4)],
        [(kaons['KK'][i] + kaons['KK?'][i]) / (made['KK'] + made['KK?']) for i in range(4)]]
print(f'    kaons per event: with a hyperon K+ {k_y[0]:.2f}, K0 {k_y[1]:.2f}; with K anti-K: K+ {k_kk[0][0]:.2f}, K0'
      f' {k_kk[0][1]:.2f}, K- {k_kk[0][2]:.2f}, anti-K0 {k_kk[0][3]:.2f} (without further particles: {fmt(k_three)})')
# the K0 shares by hyperon, as strange_20gev.py takes them for 20 GeV
def k0_share_of(prefixes, classes=('Y',)):
    k0 = sum(n / v * k[1] for nm, n, k, v, cls in TABLE_VI if cls in classes and nm.startswith(prefixes))
    kk = sum(n / v * (k[0] + k[1]) for nm, n, k, v, cls in TABLE_VI if cls in classes and nm.startswith(prefixes))
    return k0 / kk


k0_by_hyperon = {'Lambda or Sigma0': k0_share_of(LAMBDA_OR_SIGMA0), 'Sigma+': k0_share_of('Sigma+'),
                 'Sigma-': k0_share_of('Sigma-')}
print('    K0 among the kaons of the events with a hyperon and further particles: '
      + ', '.join(f'with {h} {v:.2f}' for h, v in k0_by_hyperon.items())
      + f'; with the two-body events too, with Lambda or Sigma0 {k0_share_of(LAMBDA_OR_SIGMA0, ("Y", "2")):.2f}')
# strange_20gev.py's shares of these events, to their two decimals
pair_share = lambda k, i: k[i] / (k[i - 1] + k[i])   # K0 among the kaons (i = 1), anti-K0 among the antikaons (i = 3)
for name, value in (('LAMBDA_SHARE_ABBHHM', fl), ('K0_SHARE_LAMBDA', k0_by_hyperon['Lambda or Sigma0']),
                    ('K0_SHARE_SIGMA_PLUS', k0_by_hyperon['Sigma+']), ('K0_SHARE_SIGMA_MINUS', k0_by_hyperon['Sigma-']),
                    ('K0_SHARE', pair_share(k_kk[0], 1)), ('K0_SHARE_ALL', pair_share(k_kk[1], 1))):
    assert f'{getattr(g20, name)}' == f'{value:.2f}', (name, value, getattr(g20, name))
assert [f'{v}' for v in g20.ANTI_K0_SHARE_RANGE] == [f'{v:.2f}' for v in sorted(pair_share(k, 3) for k in k_kk)]
lo_species = [[y * k_y[i] + kk_lo * k_three[i] for i in range(4)] for y in y_lo]
hi_species = [[y * k_y[i] + kk * k[i] for i in range(4)] for y, kk, k in ((y_hi(y_lo[1]), kk_hi[0], k_kk[0]),
                                                                          (y_hi(y_lo[0]), kk_hi[1], k_kk[1]))]
for label, d, c in ((STEP_LO, lo_species, code_lo), (STEP_HI, hi_species, code_hi)):
    span = lambda x, y: f'{x:.2f}' if f'{x:.2f}' == f'{y:.2f}' else f'{min(x, y):.2f}-{max(x, y):.2f}'
    print(f'    {label}: ' + ', '.join(f'{name} {span(d[0][i], d[1][i])}' for i, name in enumerate(SPECIES))
          + f'; the code {fmt(c)}')
# ABBHHM's two steps: the lower at the middle of its range, the upper with the rest of the events with a hyperon and
# the middle of its K anti-K range
y_mid = 0.5 * (y_lo[0] + y_lo[1])
mid_lo = [y_mid * k_y[i] + kk_lo * k_three[i] for i in range(4)]
mid_hi = [y_hi(y_mid) * k_y[i] + 0.5 * (kk_hi[0] * k_kk[0][i] + kk_hi[1] * k_kk[1][i]) for i in range(4)]
print(f'    ABBHHM\'s steps, the middle of the ranges [mub] (per interaction, of {SIG3[4]:g} mub):')
for label, d in ((STEP_LO, mid_lo), (STEP_HI, mid_hi)):
    print(f'      {label}: ' + ', '.join(f'{name} {x:.2f} ({x / SIG3[4]:.4f})' for x, name in zip(d, SPECIES))
          + f'; with a hyperon {hyp(d):.2f}, with K anti-K {kkbar(d):.2f}')

# SLAC: (E [GeV], events with a visible strange-particle decay [mub], error)
SLAC = [(1.44, 4.4, 0.9), (2.8, 8.1, 0.5), (4.7, 8.7, 0.4), (9.3, 9.8, 0.4)]
print('  SLAC: events with a visible strange-particle decay [mub], measured and from the code\'s kaons')
print(f'    (visible: Lambda -> p pi- {V_LAM:.2f}, K0 -> pi+ pi- {V_K0:.2f}, Sigma+ {V_SIGP:.2f}, Sigma- {V_SIGM:.2f},'
      f' K+- {V_KCH}; a hyperon of the events with further particles {V_Y:.2f})')


def code_visible(e, mf=multi, v=None):
    """The code's events with a visible decay [mub] at the photon energy e: two-body, with a hyperon and further
    particles, and with K anti-K (mf(e): the kaons of the multi-pion types; v: the visibilities of Lambda, K0,
    Sigma+, Sigma- and K+-, by default ABBHHM's)."""
    v_lam, v_k0, v_sigp, v_sigm, v_kch = v or (V_LAM, V_K0, V_SIGP, V_SIGM, V_KCH)
    v_y = fl * v_lam + fp * v_sigp + fm * v_sigm
    tb = two_body(e)
    kp, k0, km, k0b = mf(e)
    f0 = k0 / (kp + k0) if kp + k0 else 0.0            # share of K0 among the kaons
    v2 = tb[0] * v_lam + tb[1] * vis(v_sigp, v_k0)
    vy = (kp + k0 - km - k0b) * ((1 - f0) * vis(v_y, v_kch) + f0 * vis(v_y, v_k0))
    # an antikaon with a K+ or a K0 in the proportion of the type's kaons
    vkk = (km * ((1 - f0) * vis(v_kch, v_kch) + f0 * vis(v_k0, v_kch))
           + k0b * ((1 - f0) * vis(v_k0, v_kch) + f0 * vis(v_k0, v_k0)))
    return v2, vy, vkk


for e, visible, err in SLAC:
    v2, vy, vkk = code_visible(e)
    print(f'    {e:5.2f} GeV: measured {visible:.1f} +- {err:.1f}; the code {v2 + vy + vkk:.1f}'
          f' (two-body {v2:.2f}, with a hyperon and further particles {vy:.2f}, with K anti-K {vkk:.2f}):'
          f' {(v2 + vy + vkk) / visible - 1:+.0%}')

# ABBHHM against SLAC, without the code's multiplicities: ABBHHM's events with a visible decay, normalised by its
# two-body ones to the visible decays of the code's two-body steps; SLAC over the same 1/E spectrum, linear in ln E
# between its energies and from zero at the K+ Lambda threshold
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
code_vis = lambda e: sum(code_visible(e))
code = lnint(code_vis, E_THR, ABB_TOP)
print(f'  events with a visible decay up to {ABB_TOP:g} GeV, int sigma dE/E: ABBHHM {abbhhm:.1f} mub'
      f' (1185 events, {seen["2"]} of them two-body for {ln2_visible:.2f} mub), SLAC {slac:.1f} mub'
      f' ({slac / abbhhm:.2f} times as many), the code {code:.1f} mub'
      f' ({code / abbhhm - 1:+.0%} and {code / slac - 1:+.0%})')
E2 = 2.0
below = lnint(code_vis, E_THR, E2)
slac2, code2, L2 = lnint(slac_visible, E2, ABB_TOP), lnint(code_vis, E2, ABB_TOP), math.log(ABB_TOP / E2)
print(f"    above {E2:g} GeV (the code's {below:.1f} mub below {E2:g} GeV taken off): ABBHHM {abbhhm - below:.1f}, SLAC"
      f' {slac2:.1f}, the code {code2:.1f} mub; per unit ln E: {(abbhhm - below) / L2:.1f}, {slac2 / L2:.1f} and'
      f' {code2 / L2:.1f} mub')

# gamma p -> p K+ K-, given by both: SLAC (Ballam et al., Table VII: photons of 2.4-3.3 and 4.1-5.3 GeV, their
# note 24) against ABBHHM's Table V over the same photon energies
print('  gamma p -> p K+ K- [mub], SLAC against ABBHHM over the same photon energies:')
SLAC_PKK = [(2.8, 2.4, 3.3, 1.0, 0.1), (4.7, 4.1, 5.3, 0.7, 0.1)]
for e, lo, hi, sig, err in SLAC_PKK:
    parts = [(min(b, hi) - max(a, lo), v, dv) for a, b, v, dv in ABB_PKK if min(b, hi) > max(a, lo)]
    width = sum(w for w, _, _ in parts)
    abb = sum(w * v for w, v, _ in parts) / width
    abb_err = math.sqrt(sum((w * dv) ** 2 for w, _, dv in parts)) / width
    ratio = sig / abb
    print(f'    {e} GeV: SLAC {sig} +- {err}, ABBHHM {abb:.2f} +- {abb_err:.2f}: ratio'
          f' {ratio:.2f} +- {ratio * math.hypot(err / sig, abb_err / abb):.2f}')
# with pions as well (SLAC; Bingham et al., Table V, at 9.3 GeV): p K+ K-, p K+ K- pi+ pi-, p K+ K- 2pi+ 2pi-
SLAC_CHARGED = [(2.8, (1.0, 0.01, 0.0)), (4.7, (0.7, 0.3, 0.0)), (9.3, (0.58, 0.46, 0.06))]
print('    K+ K- with a proton and charged pions only (SLAC), and the K anti-K events of the code:')
for e, parts in SLAC_CHARGED:
    kk = kkbar(multi(e))
    print(f'      {e} GeV: {" + ".join(f"{x:g}" for x in parts)} = {sum(parts):.2f}; the code {kk:.2f}'
          f' ({sum(parts) / kk:.0%} of them)')

# the Lambda each experiment needs. ABBHHM: its Lambda share of the hyperons with further particles, plus the
# two-body K+ Lambda, K+ Sigma0 (Sigma0 -> Lambda gamma). SLAC: the hyperons its visible decays need with the code's
# K anti-K events.
print(f'  Lambda (with those from Sigma0) [mub]; measured at 20 GeV: {g20.LAMBDA[0]:.2f} +- {g20.LAMBDA[1]:.2f}'
      f' (Abe et al. 1985; {g20.LAMBDA[0] - g20.ANTI_LAMBDA[0]:.2f} less the anti-Lambda; the code'
      f' {g20.lambda_share() * m_Y(20.0):.2f} with the share of Lambda among the hyperons measured there,'
      f' {g20.lambda_share():.2f}); below 10 GeV with the share of ABBHHM, {fl:.2f}:')
for e, visible, err in SLAC[1:]:
    v2, vy, vkk = code_visible(e)
    need = m_Y(e) * (visible - v2 - vkk) / vy
    print(f'    {e} GeV: the code {fl * m_Y(e) + two_body(e)[0]:.1f}; for the visible decays of SLAC'
          f' {fl * need + two_body(e)[0]:.1f} ({need:.1f} mub of events with a hyperon and further particles'
          f' instead of {m_Y(e):.1f})')


# SLAC's normalisation, channels without strange particles [mub]. DESY (Struczinski et al. 1976, HEPData ins100089,
# Table 1): linear in ln E between its bin middles, plus its 5 % normalisation error (independent of energy). SLAC:
# Ballam et al. 1972, Table VII.
DESY_NORM = 0.05
DESY_BINS = [(1.6, 1.9), (1.9, 2.1), (2.1, 2.3), (2.3, 2.6), (2.6, 3.25), (3.25, 4.0), (4.0, 5.0), (5.0, 6.3)]
DESY = {  # (values per bin, errors per bin)
    'p pi+ pi-': ([41.7, 39.7, 38.5, 33.8, 28.1, 23.4, 22.5, 19.5], [2.3, 2.4, 2.4, 1.7, 1.6, 1.2, 2.0, 1.8]),
    'p pi+ pi- pi0': ([25.9, 25.0, 26.0, 23.6, 23.2, 18.8, 18.0, 14.4], [1.7, 1.8, 2.1, 1.3, 1.3, 1.4, 1.6, 1.5]),
    'n 2pi+ pi-': ([8.0, 11.4, 9.1, 9.5, 11.3, 9.8, 7.9, 7.7], [0.8, 1.1, 1.0, 0.7, 0.8, 0.9, 1.3, 1.3]),
    '3 prongs, more neutrals': ([17.9, 19.0, 23.6, 27.2, 30.2, 35.0, 29.8, 31.8],
                                [1.1, 1.3, 1.7, 1.3, 1.4, 1.9, 2.3, 2.4]),
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


print('  the normalisation of SLAC: channels without strange particles, SLAC (Ballam et al., Table VII) against the'
      ' DESY streamer chamber (Struczinski et al. 1976, normalised to the published total cross sections, +-5 %'
      ' included) [mub]:')
CHANNELS = list(DESY)   # four with three prongs, then four with five
for j, e in enumerate((2.8, 4.7)):
    parts = []
    for label, names in [('p pi+ pi-', ['p pi+ pi-']), ('3 prongs', CHANNELS[:4]), ('5 prongs', CHANNELS[4:])]:
        sv = sum(SLAC_CHANNELS[n][j][0] for n in names)
        se = math.sqrt(sum(SLAC_CHANNELS[n][j][1] ** 2 for n in names))
        dv = sum(desy_at(n, e)[0] for n in names)
        de = math.hypot(math.sqrt(sum(desy_at(n, e)[1] ** 2 for n in names)), DESY_NORM * dv)
        r = sv / dv
        parts.append(f'{label} {sv:.1f} against {dv:.1f}: {r:.2f} +- {r * math.hypot(se / sv, de / dv):.2f}')
    print(f'    {e} GeV: ' + '; '.join(parts))

# DESY's events with a visible decay (V or kink) of a strange particle (Struczinski et al. 1976, Table 2, p. 52),
# normalised as its channels. Lower limits: not corrected for decays outside the visible volume or within the vertex
# resolution (p. 51), and, as a vertex needs two tracks, perhaps without the events whose only charged particle from
# the vertex comes with a V (most two-body ones). The code: bin means with the weight 1/E; no decay lost = the
# branching ratios alone (K+- still 0.05).
DESY_VISIBLE = [(1.6, 1.9, 5.2, 0.6), (1.9, 2.1, 5.9, 0.7), (2.1, 2.3, 5.9, 0.7), (2.3, 2.6, 6.9, 0.6),
                (2.6, 3.25, 6.5, 0.5), (3.25, 4.0, 9.1, 0.8), (4.0, 5.0, 8.4, 0.8), (5.0, 6.3, 8.7, 0.9)]
V_NO_LOSS = (BR_LAMBDA, BR_K0_PIPI, 1.0, 1.0, V_KCH)
V_NO_LOSS_K = V_NO_LOSS[:4] + (0.25,)
print('  DESY streamer chamber: events with a visible strange-particle decay [mub], measured'
      ' (not corrected for the decays it could not see) and the code with ABBHHM\'s visibilities / with no decay lost /'
      ' and K+- visible in 0.25 (two-body part):')
for lo, hi, val, err in DESY_VISIBLE:
    c, c0, ck = (mean(lambda e: sum(code_visible(e, v=v)), lo, hi, 1) for v in (None, V_NO_LOSS, V_NO_LOSS_K))
    tb = mean(lambda e: code_visible(e)[0], lo, hi, 1)
    print(f'    {lo:.2f}-{hi:.2f} GeV: {val:.1f} +- {err:.1f}; the code {c:.1f} ({c / val - 1:+.0%}) / {c0:.1f}'
          f' ({c0 / val - 1:+.0%}) / {ck:.1f} ({ck / val - 1:+.0%}) ({tb:.1f})')

desy = lnint(lambda e: step(DESY_VISIBLE, e), E2, ABB_TOP)
desy_err = math.sqrt(sum((err * math.log(min(hi, ABB_TOP) / max(lo, E2))) ** 2
                         for lo, hi, _, err in DESY_VISIBLE if hi > E2))
c0 = lnint(lambda e: sum(code_visible(e, v=V_NO_LOSS)), E2, ABB_TOP)
print(f'    int sigma dE/E from {E2:g} to {ABB_TOP:g} GeV: DESY {desy:.2f} +- {desy_err:.2f} (stat.) +-'
      f' {DESY_NORM * desy:.2f} (norm.) mub, {desy / (abbhhm - below):.2f} times ABBHHM ({abbhhm - below:.2f}) and'
      f' {desy / slac2:.2f} times SLAC ({slac2:.2f}); the code {code2:.2f} ({code2 / desy - 1:+.0%}), with no decay'
      f' lost {c0:.2f} ({c0 / desy - 1:+.0%})')

# the code's yields of M3: ABBHHM's steps (above) times a factor per step. An experiment's factor: its events with a
# visible decay, less the code's two-body ones, over the visible decays of the multi-pion kaons of ABBHHM's step
# (int dE/E over the step, or over the part DESY's bins cover). 1 for ABBHHM, SLAC interpolated in ln E, DESY with
# ABBHHM's visibilities and, as a floor, with no decay lost. The code takes the middle of the three factors' range,
# or DESY's floor where that is higher, rounded to 0.01.
def abbhhm_multi(e):
    """The kaons of the multi-pion types [mub] with ABBHHM's steps in M3."""
    return tuple(mid_lo) if LO1 <= e < LO2 else tuple(mid_hi) if HI1 <= e < HI2 else multi(e)


def step_factor(measured, parts, v=None):
    two = sum(lnint(lambda e: code_visible(e, abbhhm_multi, v)[0], lo, hi) for lo, hi in parts)
    many = sum(lnint(lambda e: sum(code_visible(e, abbhhm_multi, v)[1:]), lo, hi) for lo, hi in parts)
    return (measured - two) / many


print('  the factors on the multi-pion kaons of ABBHHM\'s steps of M3 from the three experiments,'
      ' and the code\'s steps:')
for (a, b), mid in zip(M3_STEPS, (mid_lo, mid_hi)):
    f_slac = step_factor(lnint(slac_visible, a, b), [(a, b)])
    cover = [(max(lo, a), min(hi, b), val) for lo, hi, val, _ in DESY_VISIBLE if min(hi, b) > max(lo, a)]
    measured = sum(val * math.log(hi / lo) for lo, hi, val in cover)
    f_desy, f_floor = (step_factor(measured, [(lo, hi) for lo, hi, _ in cover], v) for v in (None, V_NO_LOSS))
    span = (min(1.0, f_slac, f_desy), max(1.0, f_slac, f_desy))
    f = round(max(0.5 * sum(span), f_floor), 2)
    want = [round(x * f / SIG3[4], 4) for x in mid]
    have = next(r[3:7] for r in MPI if int(r[0]) == 2 and r[1] == a)
    print(f'    {a}-{b} GeV: ABBHHM 1, SLAC {f_slac:.2f}, DESY {f_desy:.2f} (at least {f_floor:.2f}); the middle'
          f' {0.5 * sum(span):.2f}: the code takes {f:.2f}: {fmt(want, 4)} per interaction; the code {fmt(have, 4)}'
          + (' (the same)' if all(abs(w - h) < 5e-5 for w, h in zip(want, have)) else ' (DIFFERENT)'))
