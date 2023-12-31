#include <iostream>
#include <cmath>
#include <math.h>
#include "kernels.cuh"


// Function to check the results
bool checkResults(float* output, float* expected, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (std::abs(output[i] - expected[i]) > tolerance) {
            return false;
        }
    }
    return true;
}


int main() {
    // Example parameters
    int width = 7;  // Width of the input matrix
    int height = 7; // Height of the input matrix
    int filterWidth = 3;
    int inputChannels = 1;
    int outputChannels = 1;
    int imageSize = width * height * inputChannels;
    int outputWidth = width - filterWidth + 1;
    int outputHeight = height - filterWidth + 1;
    int outputSize = outputWidth * outputHeight * outputChannels;
    int filterSize = filterWidth * filterWidth * inputChannels * outputChannels;

    //pooling
    int outputWidth_pooling = floor(width/2);
    int outputHeight_pooling = floor(height/2);

    // Allocate host memory and initialize test data
    float h_input[49] = {
        0,1,1,1,0,0,0,
        0,0,1,1,1,0,0,
        0,0,0,1,1,1,0,
        0,0,0,1,1,0,0,
        0,0,1,1,0,0,0,
        0,1,1,0,0,0,0,
        1,1,0,0,0,0,0
    };

    float h_filter[9] = {
        1,0,1,
        0,1,0,
        1,0,1
    };

    float h_output_same[imageSize];
    float h_output[outputSize];
    float h_output_pooling[outputWidth_pooling*outputHeight_pooling*inputChannels];

    float h_expectedOutput[25] = {
        1,4,3,4,1,
        1,2,4,3,3,
        1,2,3,4,1,
        1,3,3,1,1,
        3,3,1,1,0
    };

    float h_expectedOutput_pooling[outputWidth_pooling*outputHeight_pooling*inputChannels] = {
        0.25, 1, 0.25,
        0, 0.5, 0.75,
        0.25, 0.75,0
    };

    float h_expectedOutput_same[49] = {
        0, 2, 2, 3, 1, 1, 0,
        1, 1, 4, 3, 4, 1, 1,
        0, 1, 2, 4, 3, 3, 0,
        0, 1, 2, 3, 4, 1, 1,
        1, 1, 3, 3, 1, 1, 0,
        1, 3, 3, 1, 1, 0, 0,
        2, 2, 1, 1, 0, 0, 0
    };

    // Allocate device memory
    float *d_input, *d_output, *d_filter, *d_output_pooling,*d_output_same;
    cudaMalloc(&d_input, imageSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
    cudaMalloc(&d_output_same, imageSize * sizeof(float));
    cudaMalloc(&d_filter, filterSize * sizeof(float));
    cudaMalloc(&d_output_pooling, outputWidth_pooling*outputHeight_pooling*inputChannels * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up kernel launch parameters and run the kernel

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, 
               (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y, 
               outputChannels);
    //convolution3D<<<numBlocks, threadsPerBlock>>>(d_input, inputChannels, width, filterWidth, 1, d_filter, d_output, false);           
    conv2DKernelCombined<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_filter, width, height, filterWidth, inputChannels, outputChannels);

    dim3 threadsPerBlock_same(16, 16, 1);
    dim3 numBlocks_same((width + threadsPerBlock_same.x - 1) / threadsPerBlock_same.x, 
               (height + threadsPerBlock_same.y - 1) / threadsPerBlock_same.y, 
               outputChannels);

    //convolution3D<<<numBlocks, threadsPerBlock>>>(d_input, inputChannels, width, filterWidth, 1, d_filter, d_output_same, true);
    conv2DKernelCombined<<<numBlocks_same, threadsPerBlock_same>>>(d_input, d_output_same, d_filter, width, height, filterWidth, inputChannels, outputChannels,true);


    dim3 threadsPerBlock_pool(16, 16, 1);
    dim3 numBlocks_pool((outputWidth_pooling + threadsPerBlock_pool.x - 1) / threadsPerBlock_pool.x, 
               (outputHeight_pooling + threadsPerBlock_pool.y - 1) / threadsPerBlock_pool.y, 
               inputChannels);

    averagePoolingKernel<<<numBlocks_pool, threadsPerBlock_pool>>>(d_input, d_output_pooling, width, height, outputWidth_pooling, outputHeight_pooling, inputChannels, 2);


    // Copy result back to host
    cudaMemcpy(h_output_same, d_output_same, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_pooling, d_output_pooling, outputWidth_pooling*outputHeight_pooling*inputChannels * sizeof(float), cudaMemcpyDeviceToHost);

    // After copying the output back to the host
    std::cout << "Kernel output:" << std::endl;
    for (int i = 0; i < outputSize; ++i) {  
        std::cout << h_output[i] << " ";
        if ((i + 1) % outputWidth == 0) std::cout << std::endl;
    }
    std::cout << "Expected output Conv:" << std::endl;
    for (int i = 0; i < 25; ++i) {  
        std::cout << h_expectedOutput[i] << " ";
        if ((i + 1) % 5 == 0) std::cout << std::endl;
    }

    // Check the results
    if (checkResults(h_output, h_expectedOutput, outputSize)) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    std::cout << "Kernel output Pooling:" << std::endl;
    for (int i = 0; i < outputWidth_pooling*outputHeight_pooling*inputChannels; ++i) {  
        std::cout << h_output_pooling[i] << " ";
        if ((i + 1) % outputWidth_pooling == 0) std::cout << std::endl;
    }

    std::cout << "Expected output Pooling:" << std::endl;
    for (int i = 0; i < outputWidth_pooling*outputHeight_pooling*inputChannels; ++i) {  
        std::cout << h_expectedOutput_pooling[i] << " ";
        if ((i + 1) % outputWidth_pooling == 0) std::cout << std::endl;
    }

      if (checkResults(h_output_pooling, h_expectedOutput_pooling, outputWidth_pooling*outputHeight_pooling*inputChannels)) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    std::cout << "Kernel output same:" << std::endl;
    for (int i = 0; i < imageSize; ++i) {  
        std::cout << h_output_same[i] << " ";
        if ((i + 1) % width == 0) std::cout << std::endl;
    }

    if (checkResults(h_output_same, h_expectedOutput_same, imageSize)) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}
