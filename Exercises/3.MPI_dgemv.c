#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void print_matrix(double *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}

void print_vector(double *vector, int size)
{
  for (int i = 0; i < size; i++)
  {
    printf("%f ", vector[i]);
  }
  printf("\n");
}

double *alloc_matrix(int rows, int cols)
{
  if (rows <= 0 || cols <= 0)
  {
    return NULL;
  }
  double *matrix = (double *)malloc(rows * cols * sizeof(double));
  return matrix;
}

void init_matrix(double *matrix, int N, int M)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < M; j++)
    {
      matrix[i * M + j] = (double)(i * M + j);
    }
  }
}

double *alloc_vector(int size)
{
  if (size <= 0)
  {
    return NULL;
  }
  double *vector = (double *)malloc(size * sizeof(double));
  return vector;
}

void init_vector(double *vector, int size)
{
  for (int i = 0; i < size; i++)
  {
    vector[i] = (double)i; // Initialize the vector elements to zero
  }
}

void free_matrix(double *matrix)
{
  if (matrix != NULL)
  {
    free(matrix);
  }
}

void free_vector(double *vector)
{
  if (vector != NULL)
  {
    free(vector);
  }
}

void matrix_vector_multiply(double *A, int N, int M, double *b, double *c)
{
  for (int i = 0; i < N; i++)
  {
    c[i] = 0.0;
    for (int j = 0; j < M; j++)
    {
      c[i] += A[i * M + j] * b[j];
    }
  }
}

int main(int argc, char *argv[])
{
  int num_of_ranks;
  int mpi_rank;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // N x M matrix A
  int N = 4;
  int M = 3;
  double *A = NULL;

  // Vector b and c
  double *b = NULL;
  double *c = NULL;

  // Determine the number of ranks to use for the computation
  if (num_of_ranks > N)
  {
    num_of_ranks = N;
  }
  int rows_per_rank = N / num_of_ranks;
  int remaining_rows = N % num_of_ranks;

  // Rank 0 initializes the matrix and vector, and distributes the work to other ranks
  if (mpi_rank == 0)
  {
    A = alloc_matrix(N, M);
    init_matrix(A, N, M);
    b = alloc_vector(M);
    init_vector(b, M);
    c = alloc_vector(N);
    
    for (int i = 1; i < num_of_ranks; i++)
    {
      MPI_Send(A + i * rows_per_rank * M + remaining_rows * M, rows_per_rank * M, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
      MPI_Send(b, M, MPI_DOUBLE, i, i + num_of_ranks, MPI_COMM_WORLD);
    }
  }
  // Other ranks receive their portion of the matrix and vector from rank 0
  else
  {
    if (mpi_rank < num_of_ranks)
    {
      A = alloc_matrix(rows_per_rank, M);
      b = alloc_vector(M);
      MPI_Recv(A, rows_per_rank * M, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(b, M, MPI_DOUBLE, 0, mpi_rank + num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Each rank performs its portion of the matrix-vector multiplication
  if (mpi_rank != 0)
  {
    if (mpi_rank < num_of_ranks)
    {
      c = alloc_vector(rows_per_rank);
      matrix_vector_multiply(A, rows_per_rank, M, b, c);
      MPI_Send(c, rows_per_rank, MPI_DOUBLE, 0, mpi_rank + 2 * num_of_ranks, MPI_COMM_WORLD);
    }
  }
  // Rank 0 collects the results from other ranks and prints the final result
  else
  {
    matrix_vector_multiply(A, rows_per_rank + remaining_rows, M, b, c);
    for (int i = 1; i < num_of_ranks; i++)
    {
      MPI_Recv(c + i * rows_per_rank + remaining_rows, rows_per_rank, MPI_DOUBLE, i, i + 2 * num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    print_vector(c, N);
  }

  // Free allocated memory
  free_matrix(A);
  free_vector(b);
  free_vector(c);

  // Finalize the MPI environment
  MPI_Finalize();

  return 0;
}