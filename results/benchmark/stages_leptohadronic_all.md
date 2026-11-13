| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.645 ms | 0.405 ms | 16.4x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.034 ms | 0.310 ms | 16.3x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.797 ms | 0.055 ms | 14.6x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.126 ms | 0.018 ms | 7.0x |
| &nbsp;&nbsp;&nbsp;↳ muon compton em. | 0.124 ms | 0.015 ms | 8.5x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.096 ms | 0.017 ms | 5.6x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.092 ms | 0.041 ms | 2.2x |
| &nbsp;&nbsp;&nbsp;↳ pion compton em. | 0.091 ms | 0.015 ms | 6.1x |
| &nbsp;&nbsp;&nbsp;↳ pair annihil. | 0.082 ms | 0.056 ms | 1.5x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.047 ms | 0.010 ms | 4.6x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.042 ms | 0.010 ms | 4.1x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.040 ms | 0.021 ms | 1.9x |
| &nbsp;&nbsp;&nbsp;↳ pion syn. em. | 0.037 ms | 0.027 ms | 1.4x |
| &nbsp;&nbsp;&nbsp;↳ muon syn. em. | 0.036 ms | 0.027 ms | 1.3x |
| calc. coeff. pairs | 2.903 ms | 0.108 ms | 26.8x |
| evolve muon | 2.079 ms | 0.068 ms | 30.5x |
| evolve pairs | 1.414 ms | 0.070 ms | 20.2x |
| evolve pion | 1.110 ms | 0.075 ms | 14.8x |
| calc. coeff. proton/neutron | 0.555 ms | 0.080 ms | 7.0x |
| calc. coeff. neutrino | 0.440 ms | 0.051 ms | 8.6x |
| calc. coeff. pion | 0.234 ms | 0.019 ms | 12.5x |
| evolve proton | 0.167 ms | 0.030 ms | 5.5x |
| calc. coeff. muon | 0.122 ms | 0.016 ms | 7.5x |
| evolve photon | 0.088 ms | 0.060 ms | 1.5x |
| photons -> had. grid | 0.039 ms | 0.000 ms | 90.7x |
| evolve neutrino | 0.036 ms | 0.036 ms | 1.0x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **15.85 ms** | **1.05 ms** | **15.1x** |
