#include <stdio.h>
#include <math.h>

// Define a function to be integrated
double my_function(double *x)
{
   double result = 0.0;
   result = sin(*x);
   return result;
}

int main(int argc, char *argv[])
{
   double a, b, h, x, integral;
   int num_of_intervals;
   num_of_intervals = 100;
   a = 0.0;
   b = 0.5 * M_PI;

   // Define interval length
   if (a < b)
   {
      h = (b - a) / ((double)num_of_intervals);
   }
   else if (a > b)
   {
      h = (a - b) / ((double)num_of_intervals);
   }

   // Compute Integral
   integral = my_function(&a) / 2.0;
   for (int i = 1; i < num_of_intervals; i++)
   {
      x = a + h * (double)(i);
      integral += my_function(&x);
   }
   integral += my_function(&b) / 2.0;
   integral = h * integral;

   // Print result
   printf("Approximate Integral: %.10E\n", integral);

   return 0;
}