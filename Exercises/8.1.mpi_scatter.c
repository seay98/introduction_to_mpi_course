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

  // Measure initial system time
  double t0 = MPI_Wtime();

  // Check number of arguments
  if (argc != 4) {
    if (mpi_rank == 0) {
      printf("Usage: %s <rows> <columns> <random seed>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  // Convert string arguments to integers using atoi()
  int rows = atoi(argv[1]);
  int cols = atoi(argv[2]);
  int seed = atoi(argv[3]);

  double* vector_in;
  double** matrix_in;
  double* vector_out;

  // Compute the number of rows that must be distributed on each rank.
  int num_of_matrix_rows_per_rank = (int)rows / num_of_ranks;
  int num_of_residual_elements = (int)rows % num_of_ranks;

  if (num_of_residual_elements != 0) {
    if (mpi_rank == 0) {
      printf("The number of rows is not divisible by the number of ranks.\n");
      printf("The program will be terminated.\n");
    }
    MPI_Finalize();
    return 1;
  } else {
    if (mpi_rank == 0) {
      matrix_in = allocate_2d_packed_double(&rows, &cols);
      vector_in = allocate_1d_double(&cols);
      init_random_matrix(seed, &rows, &cols, matrix_in);
      // print_2d_double(&rows, &cols, matrix_in);
      init_random_vector(seed, &cols, vector_in);
    } else {
      matrix_in =
          allocate_2d_packed_double(&num_of_matrix_rows_per_rank, &cols);
      vector_in = allocate_1d_double(&cols);
    }
    MPI_Scatter(&matrix_in[0][0], num_of_matrix_rows_per_rank * cols,
                MPI_DOUBLE, &matrix_in[0][0],
                num_of_matrix_rows_per_rank * cols, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
  }

  // print_2d_double(&num_of_matrix_rows_per_rank, &cols, matrix_in);

  // Bcast all vector b to all ranks
  MPI_Bcast(&vector_in[0], cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // print_1d_double(&cols, vector_in);

  // Allocate "local" output vector
  vector_out = allocate_1d_double(&num_of_matrix_rows_per_rank);

  // Measure time before mathematical operations
  double t1 = MPI_Wtime();

  // Compute "local" dgemv operation
  dgemv(&num_of_matrix_rows_per_rank, &cols, matrix_in, vector_in, vector_out);
  // print_1d_double(&num_of_matrix_rows_per_rank, vector_out);

  // Compute "local dot product"
  double norm = ddot(&num_of_matrix_rows_per_rank, vector_out, vector_out);

  // Measure time after mathematical operations
  double t2 = MPI_Wtime();

  // Reduce valuse of the "norm" they contain the local dot products
  if (mpi_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&norm, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  // Rank 0 computes norm (sqrt of the dot product)
  if (mpi_rank == 0) {
    norm = sqrt(norm);
    printf("Norm of the output vector is : %4.10f\n", norm);
  }

  // Final time
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
