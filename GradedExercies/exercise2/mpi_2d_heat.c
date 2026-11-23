#include <math.h>
#include <mpi.h>
#include <stdio.h>

#include "Libraries/arrays.h"

#define N_GRID_POINTSX 512  // Grid interior points along x
#define N_GRID_POINTSY 512  // Grid interior points along y
#define N_TEMP_STEPS 10000  // Number of time steps
#define D 1.0               // Thermal diffusion constant
#define L 1.0               // Length of the side of the plane

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int num_of_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_of_ranks);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // Create 2D Cartesian Topology
  int dims[2] = {0, 0};
  MPI_Dims_create(num_of_ranks, 2, dims);
  int periods[2] = {0, 0};
  int reorder = 1;
  MPI_Comm MPI_COMM_2D;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &MPI_COMM_2D);

  // Get rank under MPI_Comm_2D
  int cart_rank;
  MPI_Comm_rank(MPI_COMM_2D, &cart_rank);
  int coords[2];
  MPI_Cart_coords(MPI_COMM_2D, cart_rank, 2, coords);

  // Find up, down, left and right neighbors
  int up, down, left, right;
  MPI_Cart_shift(MPI_COMM_2D, 0, 1, &up, &down);
  MPI_Cart_shift(MPI_COMM_2D, 1, 1, &left, &right);

  // Local grid setup
  int local_rows = N_GRID_POINTSY / dims[0];
  int local_cols = N_GRID_POINTSX / dims[1];
  int num_elements_row = local_rows + 2;
  int num_elements_col = local_cols + 2;
  double** u = allocate_2d_packed_double(&num_elements_row, &num_elements_col);
  double** u_new =
      allocate_2d_packed_double(&num_elements_row, &num_elements_col);

  // Initial conditions
  const double dx = L / (N_GRID_POINTSX + 1);
  const double dy = L / (N_GRID_POINTSY + 1);
  const double dt = 0.25 * dx * dx * dy * dy / (D * (dx * dx + dy * dy));
  const double rx = D * dt / (dx * dx);
  const double ry = D * dt / (dy * dy);
  double local_sum = 0.0;
  for (int i = 0; i < num_elements_col; i++) {
    u[0][i] = 0;
    u_new[0][i] = 0.0;
    u[num_elements_row - 1][i] = 0;
    u_new[num_elements_row - 1][i] = 0;
  }
  for (int i = 0; i < num_elements_row; i++) {
    u[i][0] = 0;
    u_new[i][0] = 0;
    u[i][num_elements_col - 1] = 0;
    u_new[i][num_elements_col - 1] = 0;
  }
  for (int i = 1; i <= local_rows; i++) {
    int global_row = coords[0] * local_rows + i;
    double y = global_row * dy;
    for (int j = 1; j <= local_cols; j++) {
      int global_col = coords[1] * local_cols + j;
      double x = global_col * dx;
      double d2 = (x - L / 2) * (x - L / 2) + (y - L / 2) * (y - L / 2);
      u[i][j] = exp(-d2 / 0.01);
      local_sum += u[i][j];
    }
  }

  double global_sum = 0.0;
  double initial_average_temperature = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_2D);
  if (cart_rank == 0) {
    initial_average_temperature = global_sum * dx * dy / (L * L);
  }

  // Make contiguous columns type for sendrecv operation
  MPI_Datatype col_type;
  MPI_Type_vector(local_rows, 1, num_elements_col, MPI_DOUBLE, &col_type);
  MPI_Type_commit(&col_type);

  // Start to compute
  double comm_time = 0.0;
  double comp_time = 0.0;
  double start_time = MPI_Wtime();
  for (int step = 0; step < N_TEMP_STEPS; step++) {
    // Measure communication time
    double t0 = MPI_Wtime();

    // Send up boundary to UP neighbor, receive into down ghost cell from DOWN
    // neighbor
    MPI_Sendrecv(&u[1][1], local_cols, MPI_DOUBLE, up, 0, &u[local_rows + 1][1],
                 local_cols, MPI_DOUBLE, down, 0, MPI_COMM_2D,
                 MPI_STATUS_IGNORE);
    // Send down boundary to DOWN neighbor, receive into up ghost cell from UP
    // neighbor
    MPI_Sendrecv(&u[local_rows][1], local_cols, MPI_DOUBLE, down, 1, &u[0][1],
                 local_cols, MPI_DOUBLE, up, 1, MPI_COMM_2D, MPI_STATUS_IGNORE);

    // Send left boundary to LEFT neighbor, receive into right ghost cell from
    // RIGHT neighbor
    MPI_Sendrecv(&u[1][1], 1, col_type, left, 2, &u[1][local_cols + 1], 1,
                 col_type, right, 2, MPI_COMM_2D, MPI_STATUS_IGNORE);

    // Send righ boundary to RIGHT neighbor, receive into left ghost cell from
    // LEFT neighbor
    MPI_Sendrecv(&u[1][local_cols], 1, col_type, right, 3, &u[1][0], 1,
                 col_type, left, 3, MPI_COMM_2D, MPI_STATUS_IGNORE);

    // Measure compute time
    double t1 = MPI_Wtime();

    // Five-point finite difference
    for (int i = 1; i <= local_rows; i++) {
      for (int j = 1; j <= local_cols; j++) {
        u_new[i][j] = u[i][j] +
                      rx * (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) +
                      ry * (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]);
      }
    }
    // Swap u and u_new
    double** temp = u;
    u = u_new;
    u_new = temp;

    // End point for compute time
    double t2 = MPI_Wtime();
    comm_time += t1 - t0;
    comp_time += t2 - t1;

    // Print average temperature for checking
    // if (step % 1000 == 0 || step == 9999) {
    //   double local_sum = 0.0;
    //   for (int i = 1; i <= local_rows; i++) {
    //     for (int j = 1; j <= local_cols; j++) {
    //       local_sum += u[i][j];
    //     }
    //   }
    //   double global_sum = 0.0;
    //   MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
    //              MPI_COMM_2D);
    //   if (cart_rank == 0) {
    //     final_average_temperature = global_sum * dx * dy / (L * L);
    //     printf("Average temperature: %.12f\n", final_average_temperature);
    //   }
    // }
  }
  double end_time = MPI_Wtime();

  // Compute final average temperature
  local_sum = 0.0;
  for (int i = 1; i <= local_rows; i++) {
    for (int j = 1; j <= local_cols; j++) {
      local_sum += u[i][j];
    }
  }
  global_sum = 0.0;
  double final_average_temperature = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_2D);
  if (cart_rank == 0) {
    final_average_temperature = global_sum * dx * dy / (L * L);
  }

  // Compute time
  double local_total_time = end_time - start_time;
  double max_total_time = 0.0;
  double total_comm_time = 0.0;
  double total_comp_time = 0.0;
  MPI_Reduce(&comm_time, &total_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_2D);
  MPI_Reduce(&comp_time, &total_comp_time, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_2D);
  MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_2D);

  if (cart_rank == 0) {
    printf("Temperature change: %.12f\n",
           final_average_temperature - initial_average_temperature);
    printf("Average communication time per rank: %f seconds\n",
           total_comm_time / ((double)num_of_ranks));
    printf("Average calculation time per rank: %f seconds\n",
           total_comp_time / ((double)num_of_ranks));
    printf("Average total time: %f seconds\n",
           (total_comm_time + total_comp_time) / ((double)num_of_ranks));
    printf("Total execution time: %.6f seconds\n", max_total_time);
  }

  MPI_Type_free(&col_type);
  MPI_Comm_free(&MPI_COMM_2D);
  MPI_Finalize();

  return 0;
}