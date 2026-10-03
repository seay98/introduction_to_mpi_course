#include <stdio.h>
#include <math.h>
#include <mpi.h>

// Define a function to be integrated
double my_function(double *x)
{
  double result = 0.0;
  result = sin(*x);
  return result;
}

double trapezoidal_rule(double *a, double *b, double *h, int *num_of_intervals)
{
  double integral = 0.0;
  if (*a > *b)
  {
    double c = *b;
    *b = *a;
    *a = c;
  }
  if (*num_of_intervals > 0)
  {
    integral = my_function(a) / 2.0;
    for (int i = 1; i < *num_of_intervals; i++)
    {
      double x = *a + *h * (double)(i);
      integral += my_function(&x);
    }
    integral += my_function(b) / 2.0;
    integral *= *h;
  }
  return integral;
}

int main(int argc, char *argv[])
{
  double a, b, h, integral;
  int num_of_intervals = 10000;
  a = 0.0;
  b = 0.5 * M_PI;

  int num_of_ranks;
  int mpi_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  h = (b - a) / ((double)(num_of_intervals));

  int num_of_intervals_per_rank = num_of_intervals / num_of_ranks;
  int remainder = num_of_intervals % num_of_ranks;

  if (mpi_rank == 0)
  {
    num_of_intervals_per_rank += remainder;
    b = a + h * num_of_intervals_per_rank;
  }
  else
  {
    a = a + h * (num_of_intervals_per_rank * mpi_rank + remainder);
    b = a + h * num_of_intervals_per_rank;
  }

  // Compute partial integral
  integral = trapezoidal_rule(&a, &b, &h, &num_of_intervals_per_rank);

  if (mpi_rank != 0)
  {
    MPI_Send(&integral, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);
  }
  else
  {
    double integral_receive = 0.0;
    for (int i = 1; i < num_of_ranks; i++)
    {
      MPI_Recv(&integral_receive, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      integral += integral_receive;
    }
    // Print result
    printf("Approximate Integral: %.10E\n", integral);
  }

  MPI_Finalize();

  return 0;
}