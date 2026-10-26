| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.297 ms | 0.325 ms | 19.4x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.035 ms | 0.270 ms | 18.7x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.856 ms | 0.043 ms | 20.1x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.126 ms | 0.030 ms | 4.2x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.095 ms | 0.016 ms | 6.1x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.055 ms | 0.017 ms | 3.3x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.046 ms | 0.025 ms | 1.8x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.042 ms | 0.007 ms | 5.6x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.041 ms | 0.011 ms | 3.7x |
| calc. coeff. pairs | 2.854 ms | 0.093 ms | 30.8x |
| evolve pairs | 1.057 ms | 0.065 ms | 16.3x |
| calc. coeff. proton/neutron | 0.554 ms | 0.087 ms | 6.3x |
| calc. coeff. neutrino | 0.326 ms | 0.053 ms | 6.1x |
| evolve proton | 0.161 ms | 0.024 ms | 6.6x |
| calc. coeff. muon | 0.090 ms | 0.016 ms | 5.5x |
| evolve photon | 0.058 ms | 0.050 ms | 1.2x |
| calc. coeff. pion | 0.051 ms | 0.012 ms | 4.4x |
| photons -> had. grid | 0.040 ms | 0.001 ms | 71.1x |
| evolve neutrino | 0.022 ms | 0.028 ms | 0.8x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **11.56 ms** | **0.83 ms** | **13.8x** |
