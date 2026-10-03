#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

double ddot(double *a, double *b, int *num_of_elements)
{
  double c = 0.0;
  if (*num_of_elements > 0)
  {
    for (int i = 0; i < *num_of_elements; i++)
    {
      c += a[i] * b[i];
    }
  }
  return c;
}

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
  int base = num_of_elements / num_of_ranks;
  int remainder = num_of_elements % num_of_ranks;
  int elements_per_rank = 0;
  if (mpi_rank == 0)
  {
    a = (double *)calloc(num_of_elements, sizeof(double));
    b = (double *)calloc(num_of_elements, sizeof(double));
    for (int i = 0; i < num_of_elements; i++)
    {
      a[i] = (double)i;
      b[i] = (double)i;
    }
    int offset = base + (remainder > 0 ? 1 : 0);
    for (int i = 1; i < num_of_ranks; i++)
    {
      elements_per_rank = i < remainder ? base + 1 : base;
      MPI_Send(&elements_per_rank, 1, MPI_INT, i, i, MPI_COMM_WORLD);
      MPI_Send(&a[offset], elements_per_rank, MPI_DOUBLE, i, i + num_of_ranks, MPI_COMM_WORLD);
      MPI_Send(&b[offset], elements_per_rank, MPI_DOUBLE, i, i + 2 * num_of_ranks, MPI_COMM_WORLD);
      offset += elements_per_rank;
    }
  }
  else
  {
    // Each rank receives the number of elements and the vectors from rank 0
    MPI_Recv(&elements_per_rank, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    a = (double *)calloc(elements_per_rank, sizeof(double));
    b = (double *)calloc(elements_per_rank, sizeof(double));
    MPI_Recv(a, elements_per_rank, MPI_DOUBLE, 0, mpi_rank + num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(b, elements_per_rank, MPI_DOUBLE, 0, mpi_rank + 2 * num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (mpi_rank != 0)
  {
    // Each rank computes its partial dot product
    c = ddot(a, b, &elements_per_rank);
    // Each rank sends its partial dot product to rank 0
    MPI_Send(&c, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);
  }
  else
  {
    // Rank 0 computes its partial dot product
    elements_per_rank = base + (remainder > 0 ? 1 : 0);
    c = ddot(a, b, &elements_per_rank);

    double c_tmp = 0.0;
    for (int i = 1; i < num_of_ranks; i++)
    {
      MPI_Recv(&c_tmp, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      c += c_tmp;
      printf("Rank %d received partial dot product %4.2f from rank %d\n", mpi_rank, c_tmp, i);
    }
    printf(" %4.2f \n", c);
  }

  free(a);
  free(b);
  // Finalize the MPI environment
  MPI_Finalize();

  return 0;
}