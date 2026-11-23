#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_GRID_POINTSX 512
#define N_GRID_POINTSY 512
#define N_TEMP_STEPS 10000
#define D 1.0
#define L 1.0

int main(void) {
  const int rows = N_GRID_POINTSY;
  const int cols = N_GRID_POINTSX;
  const int stride = cols + 2;

  const size_t n = (size_t)(rows + 2) * (cols + 2);

  double* u = calloc(n, sizeof(double));
  double* u_new = calloc(n, sizeof(double));

  if (!u || !u_new) {
    fprintf(stderr, "Memory allocation failed\n");
    free(u);
    free(u_new);
    return EXIT_FAILURE;
  }

  const double dx = L / (cols + 1);
  const double dy = L / (rows + 1);
  const double dt = 0.25 * dx * dx * dy * dy / (D * (dx * dx + dy * dy));
  const double rx = D * dt / (dx * dx);
  const double ry = D * dt / (dy * dy);

  // Initialize the Gaussian temperature distribution.
  double sum = 0.0;
  double initial_average_temperature = 0.0;
  double final_average_temperature = 0.0;
  for (int i = 1; i <= rows; i++) {
    double y = i * dy;

    for (int j = 1; j <= cols; j++) {
      double x = j * dx;
      double d2 = (x - L / 2) * (x - L / 2) + (y - L / 2) * (y - L / 2);

      u[i * stride + j] = exp(-d2 / 0.01);
      sum += u[i * stride + j];
    }
  }

  initial_average_temperature = sum * dx * dy / (L * L);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // Time stepping.
  for (int step = 0; step < N_TEMP_STEPS; step++) {
    for (int i = 1; i <= rows; i++) {
      for (int j = 1; j <= cols; j++) {
        int k = i * stride + j;

        u_new[k] = u[k] + rx * (u[k + 1] - 2.0 * u[k] + u[k - 1]) +
                   ry * (u[k + stride] - 2.0 * u[k] + u[k - stride]);
      }
    }

    double* temp = u;
    u = u_new;
    u_new = temp;

    // if (step % 1000 == 0 || step == 9999) {
    //   sum = 0.0;
    //   for (int i = 1; i <= rows; i++) {
    //     for (int j = 1; j <= cols; j++) {
    //       sum += u[i * stride + j];
    //     }
    //   }
    //   double avg_temp = sum * dx * dy / (L * L);
    //   final_average_temperature = avg_temp;
    //   printf("Average temperature: %.12f\n", final_average_temperature);
    // }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  double elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;

  sum = 0.0;
  for (int i = 1; i <= rows; i++) {
    for (int j = 1; j <= cols; j++) {
      sum += u[i * stride + j];
    }
  }
  final_average_temperature = sum * dx * dy / (L * L);
  printf("Serial computation time: %.6f seconds\n", elapsed);
  printf("Temperature change: %.12f\n",
         final_average_temperature - initial_average_temperature);

  // Save only interior points, in row-major order.
  // FILE* fp = fopen("serial_result.bin", "wb");
  // if (!fp) {
  //   perror("fopen");
  //   free(u);
  //   free(u_new);
  //   return EXIT_FAILURE;
  // }

  // for (int i = 1; i <= rows; i++) {
  //   if (fwrite(&u[i * stride + 1], sizeof(double), cols, fp) != (size_t)cols)
  //   {
  //     fprintf(stderr, "Error writing serial result\n");
  //     fclose(fp);
  //     free(u);
  //     free(u_new);
  //     return EXIT_FAILURE;
  //   }
  // }

  // fclose(fp);
  free(u);
  free(u_new);

  return EXIT_SUCCESS;
}
