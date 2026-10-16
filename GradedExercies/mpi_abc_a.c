#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Libraries/arrays.h"
#include "Libraries/linearalgebra.h"
#include "mpi.h"

int main(int argc, char* argv[]) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  int num_of_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Measure initial time
  double t0 = MPI_Wtime();

  // Check arguments
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
    if (mpi_rank == 0) {
      printf(
          "Usage: %s <N> <M> <K> <random seed a> <random seed b> <random seed "
          "c>\n",
          argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  double** matrix_a;
  double** matrix_b;
  double* vector_c;
  double* vector_mid;
  double* vector_out;

  // Initialize matrices and vector by rank 0
  vector_mid = allocate_1d_double(&m);
  if (mpi_rank == 0) {
    matrix_a = allocate_2d_packed_double(&n, &m);
    matrix_b = allocate_2d_packed_double(&m, &k);
    vector_c = allocate_1d_double(&k);
    init_random_matrix(seed_a, &n, &m, matrix_a);
    init_random_matrix(seed_b, &m, &k, matrix_b);
    init_random_vector(seed_c, &k, vector_c);

    // Rank 0 calculate Bc
    dgemv(&m, &k, matrix_b, vector_c, vector_mid);
  }

  // Bcast vector_mid
  MPI_Bcast(&vector_mid[0], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Split work load
  int num_of_rows_per_rank = (int)n / num_of_ranks;
  int num_of_residual = (int)n % num_of_ranks;

  // Distribute matrix A to all ranks
  if (mpi_rank == 0) {
    for (int i = 1; i < num_of_ranks; i++) {
      MPI_Send(&matrix_a[i * num_of_rows_per_rank + num_of_residual][0],
               num_of_rows_per_rank * m, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
    }
  } else {
    matrix_a = allocate_2d_packed_double(&num_of_rows_per_rank, &m);
    MPI_Recv(&matrix_a[0][0], num_of_rows_per_rank * m, MPI_DOUBLE, 0, mpi_rank,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (mpi_rank == 0) {
    num_of_rows_per_rank += num_of_residual;
  }

  vector_out = allocate_1d_double(&n);

  // Measure time before operations
  double t1 = MPI_Wtime();

  // Compute local ABc
  dgemv(&num_of_rows_per_rank, &m, matrix_a, vector_mid, vector_out);

  // Compute local ddot for norm
  double norm = ddot(&num_of_rows_per_rank, vector_out, vector_out);

  // Measure time after operations
  double t2 = MPI_Wtime();

  // Reduce local result to calculate norm
  if (mpi_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&norm, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  if (mpi_rank == 0) {
    norm = sqrt(norm);
    printf("Norm of the output vector is : %4.10f\n", norm);
  }

  // End time
  double t3 = MPI_Wtime();

  // Compute time differences and print them
  double dt0 = t1 - t0;  // Time taken for initialization
  double dt1 = t2 - t1;  // Time taken for calculation
  double dt2 = t3 - t2;  // Time taken for reduction
  double dt3 = t3 - t0;  // Total time
  if (mpi_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &dt0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &dt1, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &dt2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &dt3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&dt0, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dt1, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dt2, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dt3, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (mpi_rank == 0) {
    printf("Average intiialization time per rank: %f seconds\n",
           dt0 / ((double)num_of_ranks));
    printf("Average calculation time per rank   : %f seconds\n",
           dt1 / ((double)num_of_ranks));
    printf("Average reduction time per rank     : %f seconds\n",
           dt2 / ((double)num_of_ranks));
    printf("Average Total time per rank         : %f seconds\n",
           dt3 / ((double)num_of_ranks));
  }

  MPI_Finalize();

  return 0;
}