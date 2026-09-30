#include <stdio.h>
#include <math.h>
#include <mpi.h>

// Define a function to be integrated
double my_function(double *x) {
    double result = 0.0;
    result = sin(*x);
    return result;
}

int main(int argc, char *argv[]) {
   double a, b, h, x, y, integral;
   int num_of_intervals = 500;
   a=0.0; 
   b=0.5*M_PI;

   int num_of_ranks;
   int mpi_rank;
   double mpi_rank_receive;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   if (mpi_rank == 0) {
      // Define interval length
      if (a<b) {
         h = (b-a)/((double)num_of_ranks);
      } else if ( a>b) {
         h = (a-b)/((double)num_of_ranks);
      }
      for (int i=0; i < num_of_ranks; i++) {
         MPI_Send(&h, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
      }
   }

   MPI_Recv(&h, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   
   // Compute partial integral 
   x = a + h*(double)(mpi_rank);
   y = x + h;
   h /= (double)num_of_intervals;

   integral = my_function(&x)/2.0;
   for (int i = 1; i < num_of_intervals; i++){
      x += h;
      integral+=my_function(&x);
   }
   integral+= my_function(&y)/2.0;
   integral=h*integral;

   MPI_Send(&integral, 1, MPI_DOUBLE, 0, mpi_rank, MPI_COMM_WORLD);

   if (mpi_rank == 0) {
      integral = 0.0;
      for (int i = 0; i < num_of_ranks; i++) {
         MPI_Recv(&mpi_rank_receive, 1, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         integral += mpi_rank_receive;
      }
      // Print result
      printf("Approximate Integral: %.10E\n", integral);
   }

   MPI_Finalize();

   
   return 0;
}