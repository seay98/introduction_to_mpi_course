#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
  // Prepare vectors
  int num_of_elements = 100;
  double *a;
  double *b;
  double c;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  int num_of_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Rank 0 seperates the work and sends to other ranks
  int elements_per_rank;
  if (mpi_rank == 0) {
    a = (double *)calloc(num_of_elements, sizeof(double));
    b = (double *)calloc(num_of_elements, sizeof(double));
    for (int i = 0; i < num_of_elements; i++)
    {
      a[i] = (double)i;
      b[i] = (double)i;
    }
    elements_per_rank = num_of_elements / (num_of_ranks - 1);
    int remainder = num_of_elements % (num_of_ranks - 1);
    int offset = 0;
    for (int i = 1; i < num_of_ranks; i++) {
      elements_per_rank = i < remainder ? elements_per_rank + 1 : elements_per_rank;
      MPI_Send(&elements_per_rank, 1, MPI_INT, i, i, MPI_COMM_WORLD);
      MPI_Send(&a[offset], elements_per_rank, MPI_DOUBLE, i, i + num_of_ranks, MPI_COMM_WORLD);
      MPI_Send(&b[offset], elements_per_rank, MPI_DOUBLE, i, i + 2 * num_of_ranks, MPI_COMM_WORLD);
      offset += elements_per_rank;
    }
  }

  if (mpi_rank != 0) {
    // Each rank receives the number of elements and the vectors from rank 0
    MPI_Recv(&elements_per_rank, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    a = (double *)calloc(elements_per_rank, sizeof(double));
    b = (double *)calloc(elements_per_rank, sizeof(double));
    MPI_Recv(&a[0], elements_per_rank, MPI_DOUBLE, 0, mpi_rank + num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&b[0], elements_per_rank, MPI_DOUBLE, 0, mpi_rank + 2 * num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Each rank computes its partial dot product
    c = 0.0;
    for (int i = 0; i < elements_per_rank; i++)
    {
      c += a[i] * b[i];
    }
    // Each rank sends its partial dot product to rank 0
    MPI_Send(&c, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);
  }

  // Rank 0 receives the partial dot products and sums them up
  if (mpi_rank == 0) {
    double total_c = 0.0;
    for (int i = 1; i < num_of_ranks; i++) {
      MPI_Recv(&c, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_c += c;
    }
    c = total_c;
    printf(" %4.2f \n", c);
  }

   // Finalize the MPI environment
   MPI_Finalize();
   
  return 0;
}