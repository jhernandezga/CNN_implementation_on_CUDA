// main.cpp or main.cu
#include <iostream>
#include "kernels.cuh"
#include "utils.cuh"
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>


int main() {

    //Read image from dataset
    int image_index = 3;
    int **img_2d;
    

    int width = 28;
    int height = 28;
    int inputChannels_C1 = 1;

    img_2d = (int **)malloc(height * sizeof(int *));
    for(int i = 0; i < height; i++) {
        img_2d[i] = (int *)malloc(width * sizeof(int));
    }

    readImageAtIndex(img_2d, image_index, "train-images/train-images.idx3-ubyte");
    imgGrayPrint(height, width, img_2d);

    //Normalize 0-1 -> values between 0 and 1. 1D-ARRAY
    float *raw_data = normalizeAndFlattenImage(img_2d,height,width) ; // Host input array
    //printMatrix(raw_data,width,height,1);

    int filterSize = 5;
    int outputChannels_C1 = 6;

    int width_C1 = width;  //PADDING SAME: 28x28
    int height_C1 = height;

    int width_S1 = floor(width_C1/2);
    int height_S1 = floor(height_C1/2);

    int width_C2 = width_S1-filterSize+1;  //Padding VALID: (14-5+1)x(14-5+1)
    int height_C2 = height_S1-filterSize+1;
    int outputChannels_C2 = 16;

    int width_S2 = floor(width_C2/2);
    int height_S2 = floor(height_C2/2);

    int input_dense1 = width_S2*height_S2*outputChannels_C2;
    int output_dense1 = 120;
    int output_dense2 = 84;
    int output_dense3 = 10;

    //float *raw_data;
    float *C1_data,*C2_data; // host convolution
    float *S1_data, *S2_data; // Host pooling
    float *C1_kernel, *C2_kernel; // Host kernel/filter
    float *dense_1, *weights_1,*dense_2, *weights_2,*dense_3, *weights_3,*sofmax_layer;
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel,*d_C2_data, *d_S2_data, *d_C2_kernel, *d_dense_1,*d_weights_1; // Device arrays
    float *d_dense_2,*d_weights_2, *d_dense_3,*d_weights_3;
    

    //Host Allocation
    //raw_data = (float *)malloc(width * height * inputChannels_C1 * sizeof(float));
    C1_data = (float *)malloc(width_C1 * height_C1 * outputChannels_C1 * sizeof(float));
    S1_data = (float *)malloc(width_S1 * height_S1 * outputChannels_C1 * sizeof(float));
    C1_kernel = (float *)malloc(filterSize * filterSize * outputChannels_C1 * sizeof(float));

    C2_data = (float *)malloc(width_C2 * height_C2 * outputChannels_C2 * sizeof(float));
    S2_data = (float *)malloc(width_S2 * height_S2 * outputChannels_C2 * sizeof(float));
    C2_kernel = (float *)malloc(filterSize * filterSize* outputChannels_C1* outputChannels_C2 * sizeof(float));

    dense_1 = (float *)malloc(output_dense1* sizeof(float));
    weights_1 = (float *)malloc(input_dense1*output_dense1*sizeof(float));
        
    dense_2 = (float *)malloc(output_dense2* sizeof(float));
    weights_2 = (float *)malloc(output_dense1*output_dense2*sizeof(float));

    dense_3 = (float *)malloc(output_dense3* sizeof(float));
    weights_3 = (float *)malloc(output_dense2*output_dense3*sizeof(float));

    sofmax_layer = (float *)malloc(output_dense3* sizeof(float));
    //Initializations


    //initializeWithRandomValues(raw_data, width, height, inputChannels_C1,1);
    initializeWithZero(C1_data,width_C1,height_C1,outputChannels_C1);
    initializeWithZero(S1_data,width_S1,height_S1,outputChannels_C1);
    

    //initializeWithRandomValues(C1_kernel,filterSize,filterSize,outputChannels_C1,1);
    initializeWeights("weights/layer_weights_0.bin", C1_kernel, filterSize,filterSize,outputChannels_C1);
    //printMatrix(C1_kernel, filterSize, filterSize, outputChannels_C1);
    
    initializeWithZero(C2_data,width_C2,height_C2,outputChannels_C2);
    initializeWithZero(S2_data,width_S2,height_S2,outputChannels_C2);
    //initializeWithRandomValues(C2_kernel, filterSize, filterSize, outputChannels_C1,outputChannels_C2);
    initializeWeights("weights/layer_weights_2.bin", C2_kernel, filterSize,filterSize,outputChannels_C2*outputChannels_C1);
    //printTensor(C2_kernel, filterSize, filterSize, outputChannels_C1, outputChannels_C2);

    //initializeWithRandomValues(weights_1, input_dense1, output_dense1, 1,1);
    initializeWeights("weights/layer_weights_5.bin", weights_1,output_dense1,input_dense1,1);
    initializeWithZero(dense_1,output_dense1,1,1);

    initializeWeights("weights/layer_weights_6.bin", weights_2,output_dense2,output_dense1,1);
    //initializeWithRandomValues(weights_2, output_dense1, output_dense2, 1,1);
    initializeWithZero(dense_2,output_dense2,1,1);

    initializeWeights("weights/layer_weights_7.bin", weights_3,output_dense3,output_dense2,1);
    //initializeWithRandomValues(weights_3, output_dense2, output_dense3, 1,1);
    initializeWithZero(dense_3,output_dense3,1,1);
    initializeWithZero(sofmax_layer,output_dense3,1,1);
    //printMatrix(weights_3, output_dense2, output_dense3, 1);
    //printMatrix(S1_data, width_S1, height_S1, outputChannels_C1);

    // Allocate device memory and copy data from host to device
    cudaMalloc(&d_raw_data, width * height * inputChannels_C1 * sizeof(float));
    cudaMemcpy(d_raw_data, raw_data, width * height * inputChannels_C1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_C1_data, width_C1 * height_C1 * outputChannels_C1 * sizeof(float));
    cudaMemcpy(d_C1_data, C1_data, width_C1 * height_C1 * outputChannels_C1   * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_S1_data, width_S1 * height_S1 * outputChannels_C1 * sizeof(float));
    cudaMemcpy(d_S1_data, S1_data, width_S1 * height_S1 * outputChannels_C1   * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_C1_kernel, filterSize * filterSize * outputChannels_C1 * sizeof(float));
    cudaMemcpy(d_C1_kernel, C1_kernel, filterSize * filterSize * outputChannels_C1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_C2_kernel, filterSize * filterSize* outputChannels_C1 * outputChannels_C2 * sizeof(float));
    cudaMemcpy(d_C2_kernel, C2_kernel, filterSize * filterSize* outputChannels_C1 * outputChannels_C2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_C2_data, width_C2 * height_C2 * outputChannels_C2 * sizeof(float));
    cudaMemcpy(d_C2_data, C2_data, width_C2 * height_C2 * outputChannels_C2 * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_S2_data, width_S2 * height_S2 * outputChannels_C2 * sizeof(float));
    cudaMemcpy(d_S2_data, S2_data, width_S2 * height_S2 * outputChannels_C2* sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_weights_1, input_dense1*output_dense1* sizeof(float));
    cudaMemcpy(d_weights_1, weights_1, input_dense1*output_dense1* sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_dense_1, output_dense1* sizeof(float));
    cudaMemcpy(d_dense_1, dense_1, output_dense1* sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_weights_2, output_dense1*output_dense2* sizeof(float));
    cudaMemcpy(d_weights_2, weights_2, output_dense1*output_dense2* sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_dense_2, output_dense2* sizeof(float));
    cudaMemcpy(d_dense_2, dense_2, output_dense2* sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_weights_3, output_dense2*output_dense3* sizeof(float));
    cudaMemcpy(d_weights_3, weights_3, output_dense2*output_dense3* sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_dense_3, output_dense3* sizeof(float));
    cudaMemcpy(d_dense_3, dense_3, output_dense3* sizeof(float), cudaMemcpyHostToDevice);


    //////// STACKING LAYERS

    /// 1st Layer
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width_C1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (height_C1 + threadsPerBlock.y - 1) / threadsPerBlock.y, outputChannels_C1);

    dim3 threadsPerBlock_pool(16, 16, 1);
    dim3 numBlocks_pool((width_S1 + threadsPerBlock_pool.x - 1) / threadsPerBlock_pool.x, (height_S1 + threadsPerBlock_pool.y - 1) / threadsPerBlock_pool.y, 
               outputChannels_C1);

    convolution3D<<<numBlocks, threadsPerBlock>>>(d_raw_data, d_C1_data, d_C1_kernel, width, filterSize, inputChannels_C1, outputChannels_C1,true,true);
    //conv2DKernelCombined<<<numBlocks, threadsPerBlock>>>(d_raw_data, d_C1_data, d_C1_kernel, width, height, filterSize, inputChannels_C1, outputChannels_C1,true,true);
    cudaDeviceSynchronize();
    averagePoolingKernel<<<numBlocks_pool,threadsPerBlock_pool>>>(d_C1_data,d_S1_data,width_C1,height_C1,width_S1,height_S1,outputChannels_C1,2);
    cudaDeviceSynchronize();

    //2nd Layer
    dim3 threadsPerBlock_C2(16, 16, 1);
    dim3 numBlocks_C2((width_C2 + threadsPerBlock_C2.x - 1) / threadsPerBlock_C2.x, (height_C2 + threadsPerBlock_C2.y - 1) / threadsPerBlock_C2.y, outputChannels_C2);

    dim3 threadsPerBlock_pool_S2(16, 16, 1);
    dim3 numBlocks_pool_S2((width_S2 + threadsPerBlock_pool_S2.x - 1) / threadsPerBlock_pool_S2.x, (height_S2 + threadsPerBlock_pool_S2.y - 1) / threadsPerBlock_pool_S2.y, 
               outputChannels_C2);

    convolution3D<<<numBlocks_C2, threadsPerBlock_C2>>>(d_S1_data, d_C2_data, d_C2_kernel, width_S1, filterSize, outputChannels_C1, outputChannels_C2,false,true);
    //conv2DKernelCombined<<<numBlocks_C2, threadsPerBlock_C2>>>(d_S1_data, d_C2_data, d_C2_kernel, width_S1, height_S1, filterSize, outputChannels_C1, outputChannels_C2,false,true);
    cudaDeviceSynchronize();
    averagePoolingKernel<<<numBlocks_pool_S2,threadsPerBlock_pool_S2>>>(d_C2_data,d_S2_data,width_C2,height_C2,width_S2,height_S2,outputChannels_C2,2);
    cudaDeviceSynchronize();

    //1st Dense layer

    dim3 threadsPerBlock_d1(256);  
    dim3 numBlocks_d1((output_dense1 + threadsPerBlock_d1.x - 1) / threadsPerBlock_d1.x);

    denseLayerKernel<<<numBlocks_d1, threadsPerBlock_d1>>>(d_S2_data, d_dense_1, d_weights_1,nullptr, input_dense1, output_dense1, true);

    cudaDeviceSynchronize();
    //2nd Dense layer

    dim3 threadsPerBlock_d2(256);
    dim3 numBlocks_d2((output_dense2 + threadsPerBlock_d2.x - 1) / threadsPerBlock_d2.x);

    denseLayerKernel<<<numBlocks_d2, threadsPerBlock_d2>>>(d_dense_1, d_dense_2, d_weights_2,nullptr, output_dense1, output_dense2, true);
    cudaDeviceSynchronize();
    //3rd Dense layer

    dim3 threadsPerBlock_d3(256);  
    dim3 numBlocks_d3((output_dense3 + threadsPerBlock_d3.x - 1) / threadsPerBlock_d3.x);
    
    denseLayerKernel<<<numBlocks_d3, threadsPerBlock_d3>>>(d_dense_2, d_dense_3, d_weights_3,nullptr, output_dense2, output_dense3, false);
    cudaDeviceSynchronize();


    cudaMemcpy(dense_3, d_dense_3, output_dense3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    
    //cudaMemcpy(C1_data, d_C1_data, height_C1 * width_C1 * outputChannels_C1* sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(C1_kernel, d_C1_kernel, height_C2 * width_C2 * outputChannels_C2 * sizeof(float), cudaMemcpyDeviceToHost);
    softmax(dense_3,  output_dense3);

    printMatrix(dense_3, output_dense3,1, 1);
   // printMatrix(C_data, width_C1, height_C1, outputChannels_C1);

    // Free device memory
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_S1_data);
    cudaFree(d_C2_data);
    cudaFree(d_C2_kernel);
    cudaFree(d_S2_data);
    cudaFree(d_weights_1);
    cudaFree(d_dense_1);
    cudaFree(d_weights_2);
    cudaFree(d_dense_2);
    cudaFree(d_weights_3);
    cudaFree(d_dense_3);

    // Free host memory
    free(raw_data);
    for(int i = 0; i < height; i++) {
        free(img_2d[i]);
    }
    free(img_2d);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(C2_data);
    free(S2_data);
    free(C2_kernel);
    free(dense_1);
    free(weights_1);
    free(dense_2);
    free(weights_2);
    free(dense_3);
    free(weights_3);
    free(sofmax_layer);

    return 0;
}



