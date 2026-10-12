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

   int number_of_elements;
   double *vector;

   // Rank 0 defines the number of elements of the vector
   if (mpi_rank == 0)
   {
      number_of_elements = 100;
   }
   // The number of elements is broadcasted to all ranks
   MPI_Bcast(&number_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

   // All ranks allocate memory for the vector
   vector = (double *)calloc(number_of_elements, sizeof(double));

   // Rank 0 initializes the vector
   if (mpi_rank == 0)
   {
      for (int i = 0; i < number_of_elements; i++)
      {
         vector[i] = (double)i;
      }
   }
   // Rank 0 broadcasts the vector to all other ranks
   MPI_Bcast(vector, number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   // Every rank prints vectors
   printf("Vector from rank %d : ", mpi_rank);
   for (int i = 0; i < number_of_elements; i++)
   {
      printf(" %4.2f ", vector[i]);
   }
   printf("\n");
   
   MPI_Finalize();

   return 0;
}