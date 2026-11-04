#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "Libraries/mt19937ar.h"

const double Z = 1.493648265624854;
const double EXACT_VALUE = 0.3789446916;

// Probability to sample (unnormalized)
double probability_density(double x) { return exp(-x * x); }

// Function to integrate
double f(double x) { return x * x * exp(-x * x); }

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int num_of_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);

  int n = 100000;
  if (argc > 2) {
    if (rank == 0) {
      printf("Usage: %s <num_samples>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }
  if (argc == 2) {
    n = atoi(argv[1]);
  }

  if (n < 2) {
    if (rank == 0) {
      printf("Number of samples must be >= 2\n");
    }
    MPI_Finalize();
    return 1;
  }

  // Divide work load
  int local_n = n / num_of_ranks;
  if (rank == 0) {
    local_n += n % num_of_ranks;
  }

  init_genrand(12345 + rank);

  const double a = -1.0;
  const double b = 1.0;

  const double proposal_width = 0.1;
  const int thermalization = 1000;
  double current_x = 0.0;
  int accepted_moves = 0;

  double local_sum[2] = {0.0, 0.0};
  for (int i = 0; i < local_n + thermalization; i++) {
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
    local_sum[0] += v;
    local_sum[1] += (v - EXACT_VALUE) * (v - EXACT_VALUE);
  }

  double* g_sum = NULL;
  if (rank == 0) {
    g_sum = malloc(2 * num_of_ranks * sizeof(double));
  }

  MPI_Gather(local_sum, 2, MPI_DOUBLE, g_sum, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double sum = 0.0;
  double sum2 = 0.0;
  if (rank == 0) {
    for (int i = 0; i < num_of_ranks; i++) {
      sum += g_sum[i * 2];
      sum2 += g_sum[i * 2 + 1];
    }
    free(g_sum);
    double mean = sum / n;
    // Get integration
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
  }

  MPI_Finalize();
  return 0;
}