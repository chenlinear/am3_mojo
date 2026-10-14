| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.752 ms | 1.151 ms | 5.9x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.066 ms | 0.634 ms | 8.0x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.855 ms | 0.181 ms | 4.7x |
| &nbsp;&nbsp;&nbsp;↳ muon compton em. | 0.135 ms | 0.014 ms | 10.0x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.125 ms | 0.030 ms | 4.1x |
| &nbsp;&nbsp;&nbsp;↳ pion compton em. | 0.100 ms | 0.014 ms | 7.1x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.099 ms | 0.028 ms | 3.6x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.091 ms | 0.039 ms | 2.3x |
| &nbsp;&nbsp;&nbsp;↳ pair annihil. | 0.080 ms | 0.065 ms | 1.2x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.047 ms | 0.054 ms | 0.9x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.042 ms | 0.015 ms | 2.8x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.040 ms | 0.031 ms | 1.3x |
| &nbsp;&nbsp;&nbsp;↳ pion syn. em. | 0.037 ms | 0.022 ms | 1.7x |
| &nbsp;&nbsp;&nbsp;↳ muon syn. em. | 0.036 ms | 0.021 ms | 1.7x |
| calc. coeff. pairs | 2.839 ms | 0.171 ms | 16.6x |
| evolve muon | 2.133 ms | 0.182 ms | 11.7x |
| evolve pairs | 1.599 ms | 0.224 ms | 7.1x |
| evolve pion | 1.190 ms | 0.144 ms | 8.3x |
| calc. coeff. proton/neutron | 0.553 ms | 0.223 ms | 2.5x |
| calc. coeff. neutrino | 0.437 ms | 0.256 ms | 1.7x |
| calc. coeff. pion | 0.249 ms | 0.103 ms | 2.4x |
| evolve proton | 0.166 ms | 0.132 ms | 1.3x |
| calc. coeff. muon | 0.120 ms | 0.020 ms | 6.1x |
| evolve photon | 0.083 ms | 0.073 ms | 1.1x |
| photons -> had. grid | 0.040 ms | 0.001 ms | 36.7x |
| evolve neutrino | 0.033 ms | 0.043 ms | 0.8x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **16.21 ms** | **2.77 ms** | **5.9x** |
