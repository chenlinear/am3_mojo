| Scenario | C++ AM3 (serial) | am3_mojo, 4 threads | am3_mojo, 1 thread | am3_mojo + added processes, 4 threads | Speed-up (4 threads vs C++) |
|---|---|---|---|---|---|
| blazar_simple | 11.36 ms (17.54 s, 2.88 GB) | 0.71 ms (0.14 s, 0.12 GB) | 1.58 ms (0.36 s, 0.12 GB) | 1.83 ms (0.50 s, 0.16 GB) | 16.0x |
| leptohadronic_all | 15.89 ms (18.18 s, 2.88 GB) | 1.04 ms (0.19 s, 0.13 GB) | 2.31 ms (0.49 s, 0.13 GB) | 2.68 ms (0.61 s, 0.17 GB) | 15.3x |
| ssc_external | 3.00 ms (0.16 s, 2.07 GB) | 0.22 ms (0.08 s, 0.10 GB) | 0.54 ms (0.21 s, 0.10 GB) | 0.31 ms (0.19 s, 0.11 GB) | 13.7x |
| exotic | 11.15 ms (2.39 s, 1.00 GB) | 1.15 ms (0.12 s, 0.18 GB) | 1.76 ms (0.38 s, 0.17 GB) | 1.84 ms (0.37 s, 0.19 GB) | 9.7x |

ms per evolution step (set-up: init_kernels, set_mag_field before the steps and the first step; peak memory of one simulation); minimum of 3 runs in one fresh process per implementation and scenario.
