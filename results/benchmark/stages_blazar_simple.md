| Stage | C++ AM3 (serial) | am3_mojo, 4 threads | Speed-up |
|---|---|---|---|
| calc. coeff. photon | 6.224 ms | 0.309 ms | 20.1x |
| &nbsp;&nbsp;&nbsp;↳ pair compton em. | 5.025 ms | 0.268 ms | 18.8x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. em. | 0.796 ms | 0.026 ms | 31.2x |
| &nbsp;&nbsp;&nbsp;↳ pair syn. abs. | 0.125 ms | 0.019 ms | 6.5x |
| &nbsp;&nbsp;&nbsp;↳ proton compton em. | 0.095 ms | 0.015 ms | 6.1x |
| &nbsp;&nbsp;&nbsp;↳ pi0 decay | 0.055 ms | 0.011 ms | 5.0x |
| &nbsp;&nbsp;&nbsp;↳ pair compton abs. | 0.046 ms | 0.009 ms | 5.2x |
| &nbsp;&nbsp;&nbsp;↳ photo-pion loss | 0.042 ms | 0.005 ms | 9.2x |
| &nbsp;&nbsp;&nbsp;↳ proton syn. em. | 0.041 ms | 0.010 ms | 4.1x |
| calc. coeff. pairs | 2.835 ms | 0.067 ms | 42.1x |
| evolve pairs | 0.980 ms | 0.055 ms | 17.9x |
| calc. coeff. proton/neutron | 0.555 ms | 0.074 ms | 7.5x |
| calc. coeff. neutrino | 0.330 ms | 0.038 ms | 8.8x |
| evolve proton | 0.162 ms | 0.023 ms | 7.2x |
| calc. coeff. muon | 0.089 ms | 0.010 ms | 8.7x |
| evolve photon | 0.065 ms | 0.046 ms | 1.4x |
| calc. coeff. pion | 0.050 ms | 0.012 ms | 4.2x |
| photons -> had. grid | 0.040 ms | 0.000 ms | 90.8x |
| evolve neutrino | 0.025 ms | 0.023 ms | 1.1x |
| (stages below 0.02 ms in both codes omitted) | | | |
| **total of a step** | **11.40 ms** | **0.72 ms** | **15.9x** |
