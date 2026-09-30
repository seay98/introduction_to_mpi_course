#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
   int number_of_elements;
   double *vector;

   double starting_time, ending_time, accuacy_time;

   // Initialize the MPI environment
   MPI_Init(&argc, &argv);
   int num_of_ranks;
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   int mpi_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   accuacy_time = MPI_Wtick();
   if (mpi_rank == 0) {
      printf("MPI timings are computed with an accuracy of : %10.3e seconds\n", accuacy_time);
   }
   starting_time = MPI_Wtime();

   // Initialize vector
   if (mpi_rank == 0)
   {
      number_of_elements = 50000000;
      vector = (double *)calloc(number_of_elements, sizeof(double));
      for (int i = 0; i < number_of_elements; i++)
      {
         vector[i] = (double)i;
      }
   }
   // Process/Rank 0 send the vector to all other ranks
   if (mpi_rank == 0)
   {
      for (int i = 1; i < num_of_ranks; i++)
      {
         MPI_Send(&number_of_elements, 1, MPI_INT, i, i, MPI_COMM_WORLD);
         MPI_Send(&vector[0], number_of_elements, MPI_DOUBLE, i, i + num_of_ranks, MPI_COMM_WORLD);
      }
   }

   // Every other process receives the rank
   if (mpi_rank != 0)
   {
      MPI_Recv(&number_of_elements, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      vector = (double *)calloc(number_of_elements, sizeof(double));
      MPI_Recv(&vector[0], number_of_elements, MPI_DOUBLE, 0, mpi_rank + num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   }

   ending_time = MPI_Wtime();
   printf("MPI rank %d, Time taken to send and receive vector : %f seconds\n", mpi_rank, ending_time - starting_time);

   // Finalize the MPI environment
   MPI_Finalize();

   return 0;
}
