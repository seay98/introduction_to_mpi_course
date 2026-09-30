#include <stdio.h>
#include <math.h>
#include <mpi.h>

// Define a function to be integrated
double my_function(double *x)
{
   double result = 0.0;
   result = cos(*x);
   return result;
}

// Function that integrats my_function in a finite integral [a,b]
double integral_of_my_function(double *a, double *b, int *num_of_intervals)
{
   // Invert oreder of a and b if a > b
   if (*a > *b)
   {
      double c = *b;
      *b = *a;
      *a = c;
   }
   double integral = 0.0;
   if (*num_of_intervals > 0)
   {
      // Compute the width of each trapezoid
      double h = (*b - *a) / ((double)(*num_of_intervals));
      // Compute Integral
      integral = my_function(a) / 2.0;
      for (int i = 1; i < *num_of_intervals; i++)
      {
         double x = *a + h * (double)(i);
         integral += my_function(&x);
      }
      integral += my_function(b) / 2.0;
      integral = h * integral;
   }
   return integral;
}

int main(int argc, char *argv[])
{

   // Initialize the MPI environment
   MPI_Init(&argc, &argv);
   int num_of_ranks;
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   int mpi_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   // Integration intervals
   int num_of_intervals = 10008;

   // Integration limits for cos(x) from -pi/2 to pi/2
   double a = -0.5 * M_PI;
   double b = 0.5 * M_PI;

   // Invert oreder of a and b if a > b
   if (a > b)
   {
      double c = b;
      b = a;
      a = c;
   }

   // Compute the width of each trapezoid
   double h = (b - a) / ((double)(num_of_intervals));

   int num_of_intervals_per_rank = (int)num_of_intervals / num_of_ranks;
   int num_of_residual_points = (int)num_of_intervals % num_of_ranks;

   // Compute integration points and intervals per rank.
   if (mpi_rank == 0)
   {
      num_of_intervals_per_rank += num_of_residual_points;
      b = a + h * num_of_intervals_per_rank;
   }
   else
   {
      a = a + h * (mpi_rank * num_of_intervals_per_rank + num_of_residual_points);
      b = a + h * num_of_intervals_per_rank;
   }

   // Compute local integrals
   double integral = integral_of_my_function(&a, &b, &num_of_intervals_per_rank);
   
   // Communicate the results of the integrations to rank 0 for final summation
   if (mpi_rank != 0)
   {
      MPI_Send(&integral, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);
   }
   else
   {
      double integral_tmp = 0.0;
      for (int i = 1; i < num_of_ranks; i++)
      {
         MPI_Recv(&integral_tmp, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         integral += integral_tmp;
      }
      // Print result
      printf("Approximate Integral: %.10E\n", integral);
   }

   // Finalize the MPI environment
   MPI_Finalize();

   return 0;
}