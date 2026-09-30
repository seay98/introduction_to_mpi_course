
#if defined _MPI
#include <mpi.h>
#endif
#include <stdio.h>

int main(int argc, char *argv[])
{
   int num_of_ranks = 1;
   int mpi_rank = 0;

#if defined _MPI
   // Initialize the MPI environment
   MPI_Init(&argc, &argv);
   // Get the number of processes
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   // Get the rank of the process
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

   // Print hello world from all MPI tasks
   printf("Hello world from MPI process %d out of %d processes\n", mpi_rank, num_of_ranks);

#if defined _MPI
   // Finalize the MPI environment
   MPI_Finalize();
#endif

   return 0;
}
