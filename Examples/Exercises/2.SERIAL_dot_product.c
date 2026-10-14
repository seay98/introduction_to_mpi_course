#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
   int num_of_elements=100;
   double *a; 
   double *b; 
   double c;
   a = (double *)calloc(num_of_elements, sizeof(double));
   b = (double *)calloc(num_of_elements, sizeof(double));
   for (int i = 0; i < num_of_elements; i++)
   {
      a[i] = (double)i;
      b[i] = (double)i;
   }
   c=0.0;
   for (int i = 0; i < num_of_elements; i++)
   {
      c+=a[i]*b[i];
   }
   printf(" %4.2f ", c);

   return 0;
}