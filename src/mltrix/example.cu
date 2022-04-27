#include "mltrix.cu"
#include "stdio.h"
#include <cuda_runtime.h>

int main()
{
    double power;
    power = 4.0;
    mltrix::sigmoid(&power);
    printf("%f\n", power);
    return 0;
}