#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{

   // Initialize the MPI environment
   MPI_Init(&argc, &argv);

   // Get the number of processes
   int num_of_ranks;
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);

   // Get the rank of the process
   int mpi_rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   // Print hello world from all MPI tasks
   printf("Hello world from MPI process %d out of %d processes\n", mpi_rank, num_of_ranks);

   // Finalize the MPI environment
   MPI_Finalize();

   return 0;
}
