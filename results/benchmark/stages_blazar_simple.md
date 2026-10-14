| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.328 ms | 0.882 ms | 7.2x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.069 ms | 0.530 ms | 9.6x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.853 ms | 0.153 ms | 5.6x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.125 ms | 0.044 ms | 2.8x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.094 ms | 0.028 ms | 3.3x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.055 ms | 0.034 ms | 1.6x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.048 ms | 0.051 ms | 0.9x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.043 ms | 0.015 ms | 2.9x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.042 ms | 0.025 ms | 1.7x |
| calc. coeff. pairs | 2.821 ms | 0.087 ms | 32.4x |
| evolve pairs | 1.054 ms | 0.176 ms | 6.0x |
| calc. coeff. proton/neutron | 0.553 ms | 0.213 ms | 2.6x |
| calc. coeff. neutrino | 0.330 ms | 0.205 ms | 1.6x |
| evolve proton | 0.162 ms | 0.093 ms | 1.7x |
| calc. coeff. muon | 0.090 ms | 0.014 ms | 6.4x |
| evolve photon | 0.059 ms | 0.056 ms | 1.0x |
| calc. coeff. pion | 0.051 ms | 0.062 ms | 0.8x |
| photons -> had. grid | 0.040 ms | 0.001 ms | 39.1x |
| evolve neutrino | 0.022 ms | 0.023 ms | 1.0x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **11.56 ms** | **1.91 ms** | **6.1x** |
