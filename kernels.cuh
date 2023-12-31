
#ifndef KERNEL_H
#define KERNEL_H

__global__ void conv2DKernel(float *d_input, float *d_output, float *d_filter, int width, int height, int filterWidth, int inputChannels, int outputChannels, bool activation = false);
__global__ void conv2DKernelSamePadding(float *d_input, float *d_output, float *d_filter, int width, int height, int filterWidth, int inputChannels, int outputChannels, bool activation = false);
__global__ void conv2DKernelCombined(float *d_input, float *d_output, float *d_filter, int width, int height, int filterWidth, int inputChannels, int outputChannels, bool useSamePadding=false, bool activation=false);
__global__ void averagePoolingKernel(float *d_input, float *d_output, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, int poolSize);
__global__ void denseLayerKernel(float *d_input, float *d_output, float *d_weights, float *d_bias, int inputSize, int outputSize, bool applyActivation);
__global__ void convolution3D(float *input, float *output, float *kernel, int input_size, int kernel_size,int input_filters, int number_of_filters, bool useSamePadding, bool activation = false);
void softmax(float* input, int size);

#endif // KERNEL_H
