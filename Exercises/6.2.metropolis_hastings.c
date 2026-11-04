#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Libraries/mt19937ar.h"

const double Z = 1.493648265624854;
const double EXACT_VALUE = 0.3789446916;

// Probability to sample (unnormalized)
double probability_density(double x) { return exp(-x * x); }

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

  const double proposal_width = 0.1;
  const int thermalization = 10000;
  double current_x = 0.0;
  int accepted_moves = 0;

  double sum = 0.0;
  double sum2 = 0.0;
  for (int i = 0; i < n + thermalization; i++) {
    // Proposal a new position
    double proposed_x =
        current_x + proposal_width * (2.0 * genrand_real1() - 1.0);
    // Compute acceptance probability Pi(x')/Pi(x)
    double A = 0.0;
    if (proposed_x >= a && proposed_x <= b) {
      A = probability_density(proposed_x) / probability_density(current_x);
    }

    // Accept or reject based on the Metropolis criterion
    if (A >= 1.0 || A >= genrand_real1()) {
      current_x = proposed_x;
      if (i > thermalization) {
        accepted_moves++;
      }
    }

    // Discard samples during thermalization
    if (i < thermalization) continue;

    // Importance sampling: f(x) / Pi(x) = Z * x^2
    double v = Z * current_x * current_x;
    sum += v;
    sum2 += (v - EXACT_VALUE) * (v - EXACT_VALUE);
  }
  double mean = (sum / n);

  // Monte Carlo integration
  double integral = mean;

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
  printf("Acceptance Rate: %.2f%%\n", (double)accepted_moves / n * 100.0);

  return 0;
}