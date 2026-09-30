#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
   int number_of_elements;
   double *vector;

   // Initialize the MPI environment
   MPI_Init(&argc, &argv);
   int num_of_ranks;
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   int mpi_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   // Initialize vector
   if (mpi_rank == 0)
   {
      number_of_elements = 5;
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
         MPI_Ssend(&number_of_elements, 1, MPI_INT, i, i, MPI_COMM_WORLD);
         MPI_Ssend(&vector[0], number_of_elements, MPI_DOUBLE, i, i + num_of_ranks, MPI_COMM_WORLD);
      }
   }

   // Every other process receives the rank
   if (mpi_rank != 0)
   {
      MPI_Recv(&number_of_elements, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      vector = (double *)calloc(number_of_elements, sizeof(double));
      printf("MPI rank %d Received from %d, the number of elements : %d\n", mpi_rank, 0, number_of_elements);
      MPI_Recv(&vector[0], number_of_elements, MPI_DOUBLE, 0, mpi_rank + num_of_ranks, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   }

   // MPI Barrier to synchronize the processes
   MPI_Barrier(MPI_COMM_WORLD);

   // Every rank prints vectors
   printf("Vector from rank %d : ", mpi_rank);
   for (int i = 0; i < number_of_elements; i++)
   {
      printf(" %4.2f ", vector[i]);
   }
   printf("\n");
   
   // Finalize the MPI environment
   MPI_Finalize();

   return 0;

}
