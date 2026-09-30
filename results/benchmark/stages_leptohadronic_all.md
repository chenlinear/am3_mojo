| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.570 ms | 1.339 ms | 4.9x |
| photon coeff: pair compton em. | 5.070 ms | 0.616 ms | 8.2x |
| calc. coeff. pairs | 2.844 ms | 0.154 ms | 18.5x |
| evolve muon | 2.144 ms | 0.186 ms | 11.5x |
| evolve pairs | 1.509 ms | 0.222 ms | 6.8x |
| evolve pion | 1.144 ms | 0.132 ms | 8.7x |
| photon coeff: pair syn. em. | 0.681 ms | 0.201 ms | 3.4x |
| calc. coeff. proton/neutron | 0.557 ms | 0.240 ms | 2.3x |
| calc. coeff. neutrino | 0.441 ms | 0.252 ms | 1.8x |
| calc. coeff. pion | 0.250 ms | 0.102 ms | 2.5x |
| evolve proton | 0.171 ms | 0.143 ms | 1.2x |
| photon coeff: pair syn. abs. | 0.125 ms | 0.029 ms | 4.4x |
| photon coeff: muon compton em. | 0.124 ms | 0.080 ms | 1.5x |
| calc. coeff. muon | 0.121 ms | 0.017 ms | 7.1x |
| photon coeff: proton compton em. | 0.101 ms | 0.098 ms | 1.0x |
| photon coeff: pion compton em. | 0.092 ms | 0.079 ms | 1.2x |
| photon coeff: pair annihil. | 0.087 ms | 0.064 ms | 1.4x |
| evolve photon | 0.081 ms | 0.071 ms | 1.1x |
| photon coeff: pi0 decay | 0.077 ms | 0.032 ms | 2.4x |
| photon coeff: pair compton abs. | 0.051 ms | 0.053 ms | 1.0x |
| photon coeff: photo-pion loss | 0.044 ms | 0.015 ms | 2.9x |
| photon coeff: proton syn. em. | 0.042 ms | 0.029 ms | 1.4x |
| photons -> had. grid | 0.040 ms | 0.001 ms | 37.6x |
| photon coeff: pion syn. em. | 0.038 ms | 0.021 ms | 1.8x |
| photon coeff: muon syn. em. | 0.037 ms | 0.021 ms | 1.8x |
| evolve neutrino | 0.032 ms | 0.043 ms | 0.7x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of stages** | **22.50 ms** | **4.26 ms** | **5.3x** |
