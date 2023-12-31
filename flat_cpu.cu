
// TESTING Implementations of convolutions in CPU


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>




void conv2D_cpu(int *input, int *output, int *filter, int inputHeight, int inputWidth, int filterHeight, int filterWidth, int numChannels) {
 
    int filterHeightRadius = filterHeight / 2;
    int filterWidthRadius = filterWidth / 2;
    int outputHeight = inputHeight - filterHeight + 1;
    int outputWidth = inputWidth - filterWidth + 1;

    for (int channel = 0; channel < numChannels; ++channel) {
        for (int row = 0; row < outputHeight; ++row) {
            for (int col = 0; col < outputWidth; ++col) {
                int sum = 0;
                for (int i = -filterHeightRadius; i <= filterHeightRadius; ++i) {
                    for (int j = -filterWidthRadius; j <= filterWidthRadius; ++j) {
                        int y = row + i + filterHeightRadius;
                        int x = col + j + filterWidthRadius;

                        if (x >= 0 && x < inputWidth && y >= 0 && y < inputHeight) {
                            int inputIndex = (channel * inputHeight * inputWidth) + (y * inputWidth) + x;
                            int filterIndex = (channel * filterHeight * filterWidth) + ((i + filterHeightRadius) * filterWidth) + (j + filterWidthRadius);
                            sum += filter[filterIndex] * input[inputIndex];
                        }
                    }
                }
                int outputIndex = (channel * outputHeight * outputWidth) + (row * outputWidth) + col;
                output[outputIndex] = sum;
            }
        }
    }
}


__global__ void conv2D_kernel(int *input, int *output, int *filter, int inputHeight, int inputWidth, int filterHeight, int filterWidth, int numChannels) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    int filterHeightRadius = filterHeight / 2;
    int filterWidthRadius = filterWidth / 2;

    int outputHeight = inputHeight - filterHeight + 1;
    int outputWidth = inputWidth - filterWidth + 1;

    if (row < outputHeight && col < outputWidth && channel < numChannels) {
        int sum = 0;
        for (int i = -filterHeightRadius; i <= filterHeightRadius; ++i) {
                    for (int j = -filterWidthRadius; j <= filterWidthRadius; ++j) {
                        int y = row + i + filterHeightRadius;
                        int x = col + j + filterWidthRadius;
                        
                        if (x >= 0 && x < inputWidth && y >= 0 && y < inputHeight) {
                            int inputIndex = (channel * inputHeight * inputWidth) + (y * inputWidth) + x;
                            int filterIndex = (channel * filterHeight * filterWidth) + ((i + filterHeightRadius) * filterWidth) + (j + filterWidthRadius);
                            sum += filter[filterIndex] * input[inputIndex];
                        }
                    }
                }
        int outputIndex = (channel * outputHeight * outputWidth) + (row * outputWidth) + col;
        output[outputIndex] = sum;
    }
}


void print2DArray(int *array, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", array[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}



int main() {
    
    int inputHeight = 5, inputWidth = 5;
    int filterHeight = 3, filterWidth = 3;
    int numChannels = 1;

   // int outputSize = inputHeight - filterHeight + 1; 
    int outputSize = numChannels *(inputHeight- filterHeight + 1)* (inputWidth- filterWidth + 1) ;
    int inputSize = numChannels * inputHeight* inputWidth;
    int *input = (int *)malloc(inputSize * sizeof(int));
    int outputHeight = inputHeight - filterHeight + 1;
    int outputWidth = inputWidth - filterWidth + 1;
    int *output = (int *)malloc(numChannels * outputHeight * outputWidth * sizeof(int));
    int filterSize = numChannels * filterHeight * filterWidth;
    int *filter = (int *)malloc(filterSize * sizeof(int));

    for (int i = 0; i < inputSize; i++) {
        input[i] = i;  // example input
    }

    for (int i = 0; i < filterSize; i++) {
        filter[i] = 1;  // example filter
    }

    dim3 blockDim(5, 5, 1);
    dim3 gridDim((inputWidth) / blockDim.x, (inputHeight) / blockDim.y, numChannels); 
    
    int *d_input, *d_output, *d_filter;
    
    cudaMalloc((void**)&d_input, inputSize * sizeof(int));
    cudaMalloc((void**)&d_output, outputSize * sizeof(int));
    cudaMalloc((void**)&d_filter, filterSize * sizeof(int));

    cudaMemcpy(d_input, input, inputSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, filterSize * sizeof(int), cudaMemcpyHostToDevice);

    
    //conv2D_cpu(input, output, filter, inputHeight, inputWidth, filterHeight, filterWidth, numChannels);

    conv2D_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_filter, inputHeight, inputWidth, filterHeight, filterWidth, numChannels);
    cudaMemcpy(output, d_output, outputSize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    printf("Output:\n");
    print2DArray(output, outputHeight, outputWidth);

    
    free(input);
    free(output);
    free(filter);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    return 0;
}
