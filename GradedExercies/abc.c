#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Libraries/arrays.h"
#include "Libraries/linearalgebra.h"

int main(int argc, char* argv[]) {
  // Init arguments
  int n, m, k;
  int seed_a, seed_b, seed_c;
  if (argc == 1) {
    n = 128000;
    m = 500;
    k = 1000;
    seed_a = 10;
    seed_b = 20;
    seed_c = 30;
  } else if (argc == 7) {
    n = atoi(argv[1]);
    m = atoi(argv[2]);
    k = atoi(argv[3]);
    seed_a = atoi(argv[4]);
    seed_b = atoi(argv[5]);
    seed_c = atoi(argv[6]);
  } else {
    printf(
        "Usage: %s <N> <M> <K> <random seed a> <random seed b> <random seed "
        "c>\n",
        argv[0]);
    return 1;
  }

  double** matrix_a = allocate_2d_packed_double(&n, &m);
  double** matrix_b = allocate_2d_packed_double(&m, &k);
  double* vector_c = allocate_1d_double(&k);
  double* vector_mid = allocate_1d_double(&m);
  double* vector_out = allocate_1d_double(&n);

  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);

  init_random_matrix(seed_a, &n, &m, matrix_a);
  init_random_matrix(seed_b, &m, &k, matrix_b);
  init_random_vector(seed_c, &k, vector_c);

  // Calculate d = ABc
  dgemv(&m, &k, matrix_b, vector_c, vector_mid);
  dgemv(&n, &m, matrix_a, vector_mid, vector_out);

  // Calculte norm
  double norm = sqrt(ddot(&n, vector_out, vector_out));

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

  // print_2d_double(&m, &k, matrix_b);
  // print_1d_double(&k, vector_c);
  // print_1d_double(&n, vector_out);
  printf("Norm of the output vector is : %4.10f\n", norm);
  printf("Elapsed time: %.6f seconds\n", elapsed);
  return 0;
}