// Harness: the AM3 Fic/polylog code, copied verbatim from src/Solver.cc with the
// series fix of am3_fixes.patch (Fic_1, Fic_2: Jones' Taylor series to z^10)
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace std;
static double MASS_R;
double Fic(double,double); double Fic_1(double); double Fic_2(double);
double polylog(double); double polylog_smallz(double); double polylog_rk4(double); double polylog_largez(double);
double Fic(double gamma, double Eph_GeV)
{
	double mass_mu = MASS_R * 0.511e-3;
	double gamma_bar = gamma + sqrt(gamma * gamma - 1.0 + 1.0e-10);
	double za = Eph_GeV / mass_mu * gamma_bar;
	double zb = Eph_GeV / mass_mu / gamma_bar;

	double y = gamma * (Fic_1(za) - Fic_1(zb)) - (Eph_GeV / mass_mu) * (Fic_2(za) - Fic_2(zb));

	return y > 0.0 ? y : 0.0;
}

double Fic_1(double z)
{
	// am3_mojo fix: Jones' Taylor series to z^10 (his Table I); three terms are 2.4 % off at z = 0.1
	if (z < 0.1)
		return z * z * z * (8.0 / 9.0 + z * (-14.0 / 5.0 + z * (196.0 / 25.0 + z * (-6464.0 / 315.0 + z * (7520.0 / 147.0 + z * (-2584.0 / 21.0 + z * (116704.0 / 405.0 + z * (-1636352.0 / 2475.0))))))));
	return (z + 6.0 + 3.0 / z) * log(1.0 + 2.0 * z) - (22.0 * z * z * z / 3.0 + 24.0 * z * z + 18.0 * z + 4.0) / (1.0 + 2.0 * z) / (1.0 + 2.0 * z) - 2.0 + 2.0 * polylog(-2.0 * z);
}

double Fic_2(double z)
{
	// am3_mojo fix: Jones' Taylor series to z^10 (his Table I); three terms are 1.2 % off at z = 0.1
	if (z < 0.1)
		return z * z * (4.0 / 3.0 + z * (-128.0 / 45.0 + z * (7.0 + z * (-8812.0 / 525.0 + z * (4112.0 / 105.0 + z * (-4384.0 / 49.0 + z * (63356.0 / 315.0 + z * (-397792.0 / 891.0 + z * (2427904.0 / 2475.0)))))))));
	return (z + 31.0 / 6.0 + 5.0 / z + 3.0 / 2.0 / z / z) * log(1.0 + 2.0 * z) - (22.0 * z * z * z / 3.0 + 28.0 * z * z + 103.0 * z / 3.0 + 17.0 + 3.0 / z) / (1.0 + 2.0 * z) / (1.0 + 2.0 * z) - 2.0 + polylog(-2.0 * z);
}

double polylog(double z)
{
	double y = 0.0;

	if (z >= 0.99)
		throw std::invalid_argument("in function polylog_2, z>0.9 out of range\n");
	else if (z >= -0.1 and z < 0.9)
		y = polylog_smallz(z);
	else if (z < -0.1 and z > -20.0)
		y = polylog_rk4(z);
	else
		y = polylog_largez(z);
	return y;
}

double polylog_smallz(double z)
{
	if (z < -0.1 or z > 0.9)
	{
		throw std::invalid_argument("in function polylog_smallz, argument z too large - requires -0.1<z<0.9 \n");
	}

	return z * (1.0 + z * (1.0 / 4.0 + z * (1.0 / 9.0 + z * (1.0 / 16.0 + z * (1.0 / 25.0 + z / 36.0)))));
}

double polylog_rk4(double z) // applies when z<-0.1
{
	double x = 0.1;
	double y = 0.0;
	const int N_INTG = 20;
	const double diff_x = (-z - 0.1) / N_INTG;

	if (z > -0.1)
	{
		throw std::invalid_argument("in function polylog_rk4, argument z should be z<-0.1\n");
	}

	for (int i = 0; i < N_INTG; i++)
	{
		double k1 = -diff_x * log(1.0 + x) / x;
		double k2 = -diff_x * log(1.0 + x + 0.5 * diff_x) / (x + 0.5 * diff_x);
		double k4 = -diff_x * log(1.0 + x + 1.0 * diff_x) / (x + 1.0 * diff_x);
		y += (1.0 / 6.0) * (k1 + k4) + (2.0 / 3.0) * k2;
		x += diff_x;
	}

	return -0.097605235 + y; // the number = Li2[-0.1]
}

double polylog_largez(double z)
{
	double x = -z;
	return -3.1415926 * 3.1415927 / 6.0 - 0.5 * log(x) * log(x) + 1.0 / x - 1.0 / 4.0 / x / x + 1.0 / 9.0 / x / x / x;
}


int main(int argc,char**argv){
  MASS_R = 0.938/0.511e-3;               // proton
  FILE*f=fopen(argv[1],"wb");
  for(int i=0;i<200;i++){ double g=exp(0.1*i);
    for(int j=0;j<400;j++){ double e=exp(0.1*j)*1e-9; double y=Fic(g,e); fwrite(&y,8,1,f);} }
  fclose(f);
}
