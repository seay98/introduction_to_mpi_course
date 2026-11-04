#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "Libraries/mt19937ar.h"

const double EXACT_VALUE = 0.3789446916;

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

  double local_sum[2] = {0.0, 0.0};
  for (int i = 0; i < local_n; i++) {
    double x = a + (b - a) * genrand_real1();
    double v = f(x);
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
  }

  MPI_Finalize();
  return 0;
}