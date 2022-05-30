/*
An example for a perceptron with feed-forward operation capacities
*/

#include "mltrix.cu"
#include "stdio.h"
#include <cuda_runtime.h>
#include "../cudatrix/cudatrix.cu"

class FeedForwardPerceptron
{
public:
    double weight;
    double bias;
};

double forwardPass(double dataX, double weight, double bias)
{
    cudatrix::scalarMult(&dataX, &weight); // multiply weight by the input data
    cudatrix::scalarSum(&dataX, &bias);    // sum preactivated data and bias
    mltrix::sigmoid(&dataX);  // activate our data!
    return dataX;
}

int main()
{
    FeedForwardPerceptron myPerceptron;
    myPerceptron.bias = ((double)rand() / (RAND_MAX)); // random numbers between 1 and 0
    myPerceptron.weight = ((double)rand() / (RAND_MAX));
    double dataToPass = ((double)rand() / (RAND_MAX)); // random X data
    printf("Bias: %f\n", myPerceptron.bias);
    printf("Weight: %f\n", myPerceptron.weight);
    printf("Passed data: %f\n", dataToPass);
    double perceptronOutput = forwardPass(dataToPass, myPerceptron.weight, myPerceptron.bias); // forward pass!
    printf("Perceptron output: %f\n", perceptronOutput);
}