
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
   int num_of_ranks;
   int mpi_rank, mpi_rank_receive;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   if (mpi_rank != 0)
   {
      MPI_Send(&mpi_rank, 1, MPI_INT, 0, mpi_rank, MPI_COMM_WORLD);
   }
   else
   {
      printf("Hello from process %d of %d\n", mpi_rank, num_of_ranks);
      for (int i = 1; i < num_of_ranks; i++)
      {
         MPI_Recv(&mpi_rank_receive, 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         printf("Hello from process %d of %d\n", mpi_rank_receive, num_of_ranks);
      }
   }

   // Finalize the MPI environment
   MPI_Finalize();

   return 0;
}
