"""PDG 2026 values: the derivation modules check the README and the code against them, and the check scripts use them.
Transcribed from the listings and Table 1.1 (CODATA 2022) in knowledge/papers/ (the Lambda's from pdgLive). Decimal
strings, exact as sympy Rationals and mpmath numbers alike."""

INV_ALPHA = "137.035999177"     # 1/alpha, Table 1.1 (CODATA 2022)
R_E_CM = "2.8179403205e-13"     # classical electron radius [cm], Table 1.1 (CODATA 2022)

# masses [MeV]: n (MOHR 25), p (OUR EVALUATION), e (MOHR 25), K+- (OUR FIT, p. 6), K0 (OUR FIT, p. 1)
MASS_MEV = {"n": "939.56542194", "p": "938.27208943", "e": "0.51099895069", "K+": "493.677", "K0": "497.611",
            "pi+": "139.57039", "pi0": "134.9768", "Lambda": "1115.683", "Sigma+": "1189.37", "Sigma0": "1192.642",
            "Sigma-": "1197.449", "Sigma*+": "1382.8"}
BR_LAMBDA_P_PIM = "0.641"       # Lambda listing (pdgLive, 2026 edition): Gamma(p pi-)/Gamma = (64.1 +- 0.5) %
