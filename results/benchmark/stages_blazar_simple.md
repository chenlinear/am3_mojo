| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.143 ms | 0.938 ms | 6.5x |
| photon coeff: pair compton em. | 5.044 ms | 0.540 ms | 9.3x |
| calc. coeff. pairs | 2.805 ms | 0.077 ms | 36.5x |
| evolve pairs | 1.001 ms | 0.171 ms | 5.9x |
| photon coeff: pair syn. em. | 0.702 ms | 0.144 ms | 4.9x |
| calc. coeff. proton/neutron | 0.559 ms | 0.223 ms | 2.5x |
| calc. coeff. neutrino | 0.330 ms | 0.203 ms | 1.6x |
| evolve proton | 0.167 ms | 0.094 ms | 1.8x |
| photon coeff: pair syn. abs. | 0.126 ms | 0.031 ms | 4.0x |
| photon coeff: proton compton em. | 0.095 ms | 0.124 ms | 0.8x |
| calc. coeff. muon | 0.089 ms | 0.011 ms | 7.7x |
| evolve photon | 0.058 ms | 0.052 ms | 1.1x |
| calc. coeff. pion | 0.052 ms | 0.055 ms | 0.9x |
| photon coeff: pair compton abs. | 0.050 ms | 0.050 ms | 1.0x |
| photon coeff: photo-pion loss | 0.044 ms | 0.015 ms | 3.0x |
| photon coeff: proton syn. em. | 0.043 ms | 0.023 ms | 1.8x |
| photons -> had. grid | 0.041 ms | 0.001 ms | 38.9x |
| photon coeff: pi0 decay | 0.039 ms | 0.009 ms | 4.5x |
| evolve neutrino | 0.022 ms | 0.026 ms | 0.8x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of stages** | **17.45 ms** | **2.85 ms** | **6.1x** |
