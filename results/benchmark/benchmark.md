| Scenario | C++ AM3 (serial) | am3_mojo, 4 threads | am3_mojo, 1 thread | am3_mojo + added processes, 4 threads | Speed-up (4 threads vs C++) |
|---|---|---|---|---|---|
| blazar_simple | 11.63 ms (6.91 s, 2.88 GB) | 1.95 ms (1.94 s, 0.25 GB) | 2.84 ms (6.57 s, 0.25 GB) | 3.33 ms (2.13 s, 0.28 GB) | 6.0x |
| leptohadronic_all | 17.03 ms (11.07 s, 2.88 GB) | 3.00 ms (2.98 s, 0.24 GB) | 4.56 ms (11.33 s, 0.24 GB) | 5.06 ms (3.18 s, 0.28 GB) | 5.7x |
| ssc_external | 3.04 ms (0.14 s, 2.07 GB) | 0.52 ms (0.17 s, 0.09 GB) | 1.02 ms (0.17 s, 0.09 GB) | 0.69 ms (0.18 s, 0.10 GB) | 5.8x |
| exotic | 11.67 ms (1.49 s, 1.00 GB) | 2.86 ms (0.46 s, 0.35 GB) | 4.23 ms (1.43 s, 0.34 GB) | 3.62 ms (0.53 s, 0.36 GB) | 4.1x |

ms per evolution step (kernel initialisation, peak memory of one simulation); minimum of 3 runs in one fresh process per implementation and scenario.
