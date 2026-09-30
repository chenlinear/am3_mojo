| Scenario | C++ AM3 (serial) | am3_mojo, 4 threads | am3_mojo, 1 thread | Speed-up (4 threads vs C++) |
|---|---|---|---|---|
| blazar_simple | 11.52 ms (6.96 s, 3.38 GB) | 1.98 ms (2.03 s, 0.34 GB) | 2.89 ms (6.93 s, 0.34 GB) | 5.8x |
| leptohadronic_all | 16.72 ms (11.11 s, 3.38 GB) | 3.13 ms (3.24 s, 0.35 GB) | 4.35 ms (11.73 s, 0.35 GB) | 5.3x |
| ssc_external | 3.06 ms (0.14 s, 2.47 GB) | 0.62 ms (0.17 s, 0.09 GB) | 1.38 ms (0.17 s, 0.09 GB) | 4.9x |
| exotic | 11.54 ms (1.49 s, 1.46 GB) | 3.07 ms (0.46 s, 0.55 GB) | 7.08 ms (1.49 s, 0.55 GB) | 3.8x |

ms per evolution step (kernel initialisation, peak memory); minimum of 3 runs, each in a fresh process.
