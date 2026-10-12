#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{

   // Initialize the MPI environment
   MPI_Init(&argc, &argv);
   int num_of_ranks;
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   int mpi_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   // All ranks initialize the values that contribute to the sum.
   double local_values_to_sum = (double)mpi_rank;

   // All ranks initialize the values of the sum
   double sum = 0.0;

   // Perform the reduction operation to sum values across all ranks
   MPI_Allreduce(&local_values_to_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   // Every rank prints the sum
   printf("Sum from rank %d : %4.2f \n", mpi_rank, sum);

   // Perform the reduction operation on the variables to reduce
   MPI_Allreduce(MPI_IN_PLACE,&local_values_to_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   // Every rank prints the sum
   printf("Sum from rank %d : %4.2f \n", mpi_rank, local_values_to_sum);

   // Finalize the MPI environment
   MPI_Finalize();

   return 0;
}
