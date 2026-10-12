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

   if (mpi_rank == 0)
   {
      // Rank 0 passes MPI_IN_PLACE for sendbuffer; result is stored directly in 'local_values_to_sum'
      MPI_Reduce(MPI_IN_PLACE, &local_values_to_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   }
   else
   {
      // Other ranks pass their local buffer to sendbuffer; recvbuffer is ignored (NULL)
      MPI_Reduce(&local_values_to_sum, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   }

   // Every rank prints the sum
   printf("Sum from rank %d : %4.2f \n", mpi_rank, local_values_to_sum);

   // Finalize the MPI environment
   MPI_Finalize();

   return 0;
}
