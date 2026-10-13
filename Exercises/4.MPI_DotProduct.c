#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to compute dot productbetween vectors
double ddot(double* a, double* b, int* num_of_elements) {
  double c = 0.0;
  if (*num_of_elements > 0) {
    for (int i = 0; i < *num_of_elements; i++) {
      c += a[i] * b[i];
    }
  }
  return c;
}

int main(int argc, char* argv[]) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  int num_of_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int num_of_elements = 100;
  double* a;
  double* b;

  if (mpi_rank == 0) {
    a = (double*)calloc(num_of_elements, sizeof(double));
    b = (double*)calloc(num_of_elements, sizeof(double));
    for (int i = 0; i < num_of_elements; i++) {
      a[i] = (double)i;
      b[i] = (double)i;
    }
  }

  int num_of_elements_per_rank = (int)num_of_elements / num_of_ranks;
  int num_of_residual_elements = (int)num_of_elements % num_of_ranks;

  if (num_of_elements_per_rank > 0) {
    if (mpi_rank == 0) {
      int first_element = 0;
      for (int i = 1; i < num_of_ranks; i++) {
        first_element = i * num_of_elements_per_rank + num_of_residual_elements;
        MPI_Send(&a[first_element], num_of_elements_per_rank, MPI_DOUBLE, i, i,
                 MPI_COMM_WORLD);
        MPI_Send(&b[first_element], num_of_elements_per_rank, MPI_DOUBLE, i,
                 i + num_of_ranks, MPI_COMM_WORLD);
      }
    } else {
      a = (double*)calloc(num_of_elements_per_rank, sizeof(double));
      b = (double*)calloc(num_of_elements_per_rank, sizeof(double));
      MPI_Recv(&a[0], num_of_elements_per_rank, MPI_DOUBLE, 0, mpi_rank,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&b[0], num_of_elements_per_rank, MPI_DOUBLE, 0,
               mpi_rank + num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Add the residual elements to the number of elements of rank 0
  if (mpi_rank == 0) {
    num_of_elements_per_rank += num_of_residual_elements;
  }

  // Each Rank computes the dot product if it has points to compute
  double result = ddot(&a[0], &b[0], &num_of_elements_per_rank);

  // Communicate the results of the integrations to rank 0 for final summation
  // if (mpi_rank != 0)
  // {
  //    MPI_Send(&result, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);
  // }
  // else
  // {
  //    double result_tmp = 0.0;
  //    for (int i = 1; i < num_of_ranks; i++)
  //    {
  //       MPI_Recv(&result_tmp, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD,
  //       MPI_STATUS_IGNORE); result += result_tmp;
  //    }

  //    // Print result
  //    printf("Value of the dot product: %.10E\n", result);
  // }

  // Use MPI_Reduce to sum up the integrals from all ranks to rank 0
  double total_result = 0.0;
  MPI_Reduce(&result, &total_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Print result
  if (mpi_rank == 0) {
    printf("Value of the dot product: %.10E\n", total_result);
  }

  // Use MPI_Allreduce to sum up the integrals from all ranks to all ranks
  double total_result_all = 0.0;
  MPI_Allreduce(&result, &total_result_all, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  printf("Rank %d: Value of the dot product (Allreduce): %.10E\n", mpi_rank,
         total_result_all);

  // Finalize the MPI environment
  MPI_Finalize();

  return 0;
}