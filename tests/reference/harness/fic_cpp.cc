// Harness: AM3's Fic code, verbatim from src/Solver.cc with the fixes of
// am3_fixes.patch (Fic_1, Fic_2: Jones' Taylor series to z^10 and the dilogarithm
// of math.cc) and its PDG 2026 constants (masse, massp and mathpi of consts.h)
#include <cmath>
#include <cstdio>
using namespace std;
static double MASS_R;
double Fic(double,double); double Fic_1(double); double Fic_2(double);
double polylog(double);
double Fic(double gamma, double Eph_GeV)
{
	double mass_mu = MASS_R * 0.51099895069e-3;
	double gamma_bar = gamma + sqrt(gamma * gamma - 1.0 + 1.0e-10);
	double za = Eph_GeV / mass_mu * gamma_bar;
	double zb = Eph_GeV / mass_mu / gamma_bar;

	double y = gamma * (Fic_1(za) - Fic_1(zb)) - (Eph_GeV / mass_mu) * (Fic_2(za) - Fic_2(zb));

	return y > 0.0 ? y : 0.0;
}

double Fic_1(double z)
{
	// am3_mojo fix: Jones' Taylor series to z^10 (his Table I): three terms are 2.4 % off at z = 0.1
	if (z < 0.1)
		return z * z * z * (8.0 / 9.0 + z * (-14.0 / 5.0 + z * (196.0 / 25.0 + z * (-6464.0 / 315.0 + z * (7520.0 / 147.0 + z * (-2584.0 / 21.0 + z * (116704.0 / 405.0 + z * (-1636352.0 / 2475.0))))))));
	return (z + 6.0 + 3.0 / z) * log(1.0 + 2.0 * z) - (22.0 * z * z * z / 3.0 + 24.0 * z * z + 18.0 * z + 4.0) / (1.0 + 2.0 * z) / (1.0 + 2.0 * z) - 2.0 + 2.0 * polylog(-2.0 * z);
}

double Fic_2(double z)
{
	// am3_mojo fix: Jones' Taylor series to z^10 (his Table I): three terms are 1.2 % off at z = 0.1
	if (z < 0.1)
		return z * z * (4.0 / 3.0 + z * (-128.0 / 45.0 + z * (7.0 + z * (-8812.0 / 525.0 + z * (4112.0 / 105.0 + z * (-4384.0 / 49.0 + z * (63356.0 / 315.0 + z * (-397792.0 / 891.0 + z * (2427904.0 / 2475.0)))))))));
	return (z + 31.0 / 6.0 + 5.0 / z + 3.0 / 2.0 / z / z) * log(1.0 + 2.0 * z) - (22.0 * z * z * z / 3.0 + 28.0 * z * z + 103.0 * z / 3.0 + 17.0 + 3.0 / z) / (1.0 + 2.0 * z) / (1.0 + 2.0 * z) - 2.0 + polylog(-2.0 * z);
}

// the dilogarithm of math.cc in am3_fixes.patch (Fic_1, Fic_2 call it)
static double li2_series(double z)
{
    static const double c[9] = {
        1.0 / 36.0, -1.0 / 3600.0, 1.0 / 211680.0, -1.0 / 10886400.0,
        1.0 / 526901760.0, -691.0 / 16999766784000.0, 1.0 / 1120863744000.0,
        -3617.0 / 181400588328960000.0, 43867.0 / 97072790126247936000.0};
    double u = -log1p(-z);
    double u2 = u * u;
    double s = 0.0;
    for (int k = 8; k >= 0; k--)
        s = s * u2 + c[k];
    return u - 0.25 * u2 + u * u2 * s;
}

double polylog(double z)
{
    const double mathpi = 3.141592653589793;
    if (z < -1.0)
    {
        double l = log(-z);
        return -mathpi * mathpi / 6.0 - 0.5 * l * l - li2_series(1.0 / z);
    }
    if (z <= 0.5)
        return li2_series(z);
    if (z < 1.0)
        return mathpi * mathpi / 6.0 - log(z) * log1p(-z) - li2_series(1.0 - z);
    return mathpi * mathpi / 6.0;
}

int main(int argc,char**argv){
  MASS_R = 0.93827208943/0.51099895069e-3;   // proton
  FILE*f=fopen(argv[1],"wb");
  for(int i=0;i<200;i++){ double g=exp(0.1*i);
    for(int j=0;j<400;j++){ double e=exp(0.1*j)*1e-9; double y=Fic(g,e); fwrite(&y,8,1,f);} }
  fclose(f);
}
