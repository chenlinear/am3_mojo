// Harness: GSL gsl_sf_bessel_Knu and AM3 int_bessel_k53 (patched math.cc)
#include <cmath>
#include <cstdio>
#include <gsl/gsl_sf_bessel.h>
#include "AM3/math.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "wb");
  const double nus[4] = {1.0/3.0, 2.0/3.0, 4.0/3.0, 5.0/3.0};
  for (int k = 0; k < 4; k++)
    for (int i = 0; i < 2000; i++) { double x = exp(-9.0 + 0.0075 * i); double y = gsl_sf_bessel_Knu(nus[k], x); fwrite(&y, 8, 1, f); }
  for (int i = 0; i < 2000; i++) { double x = exp(-9.0 + 0.0075 * i); double y = int_bessel_k53(x); fwrite(&y, 8, 1, f); }
  fclose(f);
}
