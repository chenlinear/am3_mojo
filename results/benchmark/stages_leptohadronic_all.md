| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.736 ms | 0.429 ms | 15.7x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.043 ms | 0.308 ms | 16.4x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.855 ms | 0.088 ms | 9.7x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.125 ms | 0.026 ms | 4.9x |
| &nbsp;&nbsp;&nbsp;↳ pion compton em. | 0.124 ms | 0.015 ms | 8.4x |
| &nbsp;&nbsp;&nbsp;↳ muon compton em. | 0.120 ms | 0.014 ms | 8.4x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.096 ms | 0.017 ms | 5.7x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.091 ms | 0.060 ms | 1.5x |
| &nbsp;&nbsp;&nbsp;↳ pair annihil. | 0.081 ms | 0.062 ms | 1.3x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.047 ms | 0.046 ms | 1.0x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.041 ms | 0.008 ms | 5.4x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.040 ms | 0.024 ms | 1.7x |
| &nbsp;&nbsp;&nbsp;↳ pion syn. em. | 0.037 ms | 0.032 ms | 1.2x |
| &nbsp;&nbsp;&nbsp;↳ muon syn. em. | 0.036 ms | 0.039 ms | 0.9x |
| calc. coeff. pairs | 2.915 ms | 0.144 ms | 20.2x |
| evolve muon | 2.139 ms | 0.085 ms | 25.3x |
| evolve pairs | 1.610 ms | 0.100 ms | 16.2x |
| evolve pion | 1.195 ms | 0.091 ms | 13.1x |
| calc. coeff. proton/neutron | 0.552 ms | 0.089 ms | 6.2x |
| calc. coeff. neutrino | 0.436 ms | 0.063 ms | 7.0x |
| calc. coeff. pion | 0.227 ms | 0.031 ms | 7.3x |
| evolve proton | 0.166 ms | 0.035 ms | 4.7x |
| calc. coeff. muon | 0.120 ms | 0.020 ms | 6.0x |
| evolve photon | 0.080 ms | 0.062 ms | 1.3x |
| photons -> had. grid | 0.040 ms | 0.000 ms | 87.5x |
| evolve neutrino | 0.033 ms | 0.034 ms | 0.9x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **16.27 ms** | **1.24 ms** | **13.2x** |
