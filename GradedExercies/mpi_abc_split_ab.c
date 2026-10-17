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

  // Measure Bc initial time
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
    if (n <= 0 || m <= 0 || k <= 0) {
      if (mpi_rank == 0) {
        printf("<N> <M> <K> should be greater than 0.\n");
      }
      MPI_Finalize();
      return 1;
    }
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

  double** matrix_a = NULL;
  double** matrix_b = NULL;
  double* vector_c;
  double* vector_mid;
  double* vector_out;

  // Compute work load for Bc
  int num_of_rows_per_rank = (int)m / num_of_ranks;
  int num_of_residual = (int)m % num_of_ranks;

  // Initialize A B c
  if (mpi_rank == 0) {
    matrix_a = allocate_2d_packed_double(&n, &m);
    matrix_b = allocate_2d_packed_double(&m, &k);
    vector_c = allocate_1d_double(&k);
    init_random_matrix(seed_a, &n, &m, matrix_a);
    init_random_matrix(seed_b, &m, &k, matrix_b);
    init_random_vector(seed_c, &k, vector_c);
  } else {
    vector_c = allocate_1d_double(&k);
    if (num_of_rows_per_rank > 0) {
      matrix_b = allocate_2d_packed_double(&num_of_rows_per_rank, &k);
    }
  }

  // Bcast vector c
  MPI_Bcast(&vector_c[0], k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Distribute matrix b
  if (num_of_rows_per_rank > 0) {
    if (mpi_rank == 0) {
      for (int i = 1; i < num_of_ranks; i++) {
        MPI_Send(&matrix_b[i * num_of_rows_per_rank + num_of_residual][0],
                 num_of_rows_per_rank * k, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
      }
    } else {
      MPI_Recv(&matrix_b[0][0], num_of_rows_per_rank * k, MPI_DOUBLE, 0,
               mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Rank 0 handles remaining work
  if (mpi_rank == 0) {
    num_of_rows_per_rank += num_of_residual;
  }

  vector_mid = allocate_1d_double(&m);

  // Measure Bc time
  double t1 = MPI_Wtime();

  // dgemv operation for Bc, place result at right position in the vector
  int offset =
      mpi_rank == 0 ? 0 : mpi_rank * num_of_rows_per_rank + num_of_residual;
  dgemv(&num_of_rows_per_rank, &k, matrix_b, vector_c, &vector_mid[offset]);

  // Measure reduce time
  double t2 = MPI_Wtime();

  // Reduce  to get whole vector_mid for every rank
  MPI_Allreduce(MPI_IN_PLACE, &vector_mid[0], m, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  // Time to initialize Ax
  double t3 = MPI_Wtime();

  // Split matrix A
  num_of_rows_per_rank = (int)n / num_of_ranks;
  num_of_residual = (int)n % num_of_ranks;

  // Distribute matrix A
  if (num_of_rows_per_rank > 0) {
    if (mpi_rank == 0) {
      for (int i = 1; i < num_of_ranks; i++) {
        MPI_Send(&matrix_a[i * num_of_rows_per_rank + num_of_residual][0],
                 num_of_rows_per_rank * m, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
      }
    } else {
      matrix_a = allocate_2d_packed_double(&num_of_rows_per_rank, &m);
      MPI_Recv(&matrix_a[0][0], num_of_rows_per_rank * m, MPI_DOUBLE, 0,
               mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Adjust work load for rank 0
  if (mpi_rank == 0) {
    num_of_rows_per_rank += num_of_residual;
  }

  // Init vector out
  vector_out = allocate_1d_double(&num_of_rows_per_rank);

  // Time to calculate Ax and ddot
  double t4 = MPI_Wtime();

  // dgemv operation
  dgemv(&num_of_rows_per_rank, &m, matrix_a, vector_mid, vector_out);

  // ddot operation
  double norm = ddot(&num_of_rows_per_rank, vector_out, vector_out);

  // Time to reduce for norm
  double t5 = MPI_Wtime();

  // Reduce and compute norm
  if (mpi_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&norm, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (mpi_rank == 0) {
    norm = sqrt(norm);
    printf("Norm of the output vector is : %4.10f\n", norm);
  }

  // Final time
  double t6 = MPI_Wtime();

  // Compute time differences and print them
  double dt0 = t1 - t0;    // Time taken for initialization Bc
  double dt1 = t2 - t1;    // Time taken for calculation Bc
  double dt2 = t3 - t2;    // Time taken for reduction to get vector_mid
  double dt3 = t4 - t3;    // Time to initialize Ax
  double dt4 = t5 - t4;    // Time taken for calculation Ax
  double dt5 = t6 - t5;    // Time taken for reduction to get norm
  double dt6 = t6 - t0;    // Total time
  double dt7 = dt0 + dt3;  // Total time for initialization
  double dt8 = dt1 + dt4;  // Total time for calculation
  double dt9 = dt2 + dt5;  // Total time for reduction

  if (mpi_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &dt7, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &dt8, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &dt9, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &dt6, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&dt7, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dt8, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dt9, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dt6, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }
  if (mpi_rank == 0) {
    printf("Average intiialization time per rank: %f seconds\n",
           dt7 / ((double)num_of_ranks));
    printf("Average calculation time per rank   : %f seconds\n",
           dt8 / ((double)num_of_ranks));
    printf("Average reduction time per rank     : %f seconds\n",
           dt9 / ((double)num_of_ranks));
    printf("Total time                          : %f seconds\n", dt6);
  }

  MPI_Finalize();
  return 0;
}