| Scenario | C++ AM3 (serial) | am3_mojo, 4 threads | am3_mojo, 1 thread | am3_mojo + added processes, 4 threads | Speed-up (4 threads vs C++) |
|---|---|---|---|---|---|
| blazar_simple | 11.54 ms (6.91 s, 2.92 GB) | 0.85 ms (1.79 s, 0.30 GB) | 1.89 ms (6.68 s, 0.30 GB) | 2.02 ms (2.17 s, 0.34 GB) | 13.5x |
| leptohadronic_all | 17.01 ms (11.05 s, 2.92 GB) | 1.30 ms (2.85 s, 0.30 GB) | 2.90 ms (10.69 s, 0.30 GB) | 3.05 ms (3.36 s, 0.35 GB) | 13.1x |
| ssc_external | 3.03 ms (0.14 s, 2.11 GB) | 0.23 ms (0.09 s, 0.15 GB) | 0.58 ms (0.21 s, 0.15 GB) | 0.31 ms (0.10 s, 0.15 GB) | 13.1x |
| exotic | 11.67 ms (1.48 s, 1.04 GB) | 1.23 ms (0.41 s, 0.25 GB) | 1.87 ms (1.43 s, 0.26 GB) | 1.96 ms (0.66 s, 0.26 GB) | 9.5x |

ms per evolution step (kernel initialisation, peak memory of one simulation); minimum of 3 runs in one fresh process per implementation and scenario.
