#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Libraries/mt19937ar.h"

const double EXACT_VALUE = 0.3789446916;

// Function to integration
double f(double x) { return x * x * exp(-x * x); }

int main(int argc, char* argv[]) {
  int n = 100000;
  if (argc > 2) {
    printf("Usage: %s <num_samples>\n", argv[0]);
    return 1;
  }

  if (argc == 2) {
    n = atoi(argv[1]);
  }

  if (n < 2) {
    printf("Number of samples must be >= 2\n");
    return 1;
  }

  // Random number generator
  init_genrand(12345);

  const double a = -1.0;
  const double b = 1.0;

  double sum = 0.0;
  double sum2 = 0.0;
  for (int i = 0; i < n; i++) {
    // Uniform random number in [-1, 1]
    double x = a + (b - a) * genrand_real1();

    double v = f(x);
    sum += v;
    sum2 += (v - EXACT_VALUE) * (v - EXACT_VALUE);
  }
  double mean = (sum / n);

  // Monte Carlo integration
  double integral = (b - a) * mean;

  // Mean squared deviation from exact integral
  double mse = sum2 / n;

  // Root mean squared deviation
  double rmsd = sqrt(mse);

  // Absolute error of integral estimate
  double abs_error = fabs(integral - EXACT_VALUE);

  printf("Samples: %d\n", n);
  printf("Integral: %.10f\n", integral);
  printf("MSE (reference): %.10e\n", mse);
  printf("RMSD (reference): %.10e\n", rmsd);
  printf("Absolute error: %.10e\n", abs_error);

  return 0;
}