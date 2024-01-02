#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>


__device__ float activation_tanh(float M);

// conv2d padding valid
__global__ void conv2DKernel(float *d_input, float *d_output, float *d_filter, int width, int height, int filterWidth, int inputChannels, int outputChannels, bool activation = false) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    int outputWidth = width - filterWidth + 1;
    int outputHeight = height - filterWidth + 1;

    if (ix < outputWidth && iy < outputHeight && kz < outputChannels) {
        float value = 0.0f;

        for (int channel = 0; channel < inputChannels; ++channel) {
            for (int fx = 0; fx < filterWidth; ++fx) {
                for (int fy = 0; fy < filterWidth; ++fy) {
                    int imgX = ix + fx;
                    int imgY = iy + fy;

                    int inputIdx = (imgY * width + imgX) * inputChannels + channel;
                    int filterIdx = ((kz * filterWidth + fy) * filterWidth + fx) * inputChannels + channel;

                    // Multiply and accumulate
                    value += d_filter[filterIdx] * d_input[inputIdx];
                }
            }
        }

        int outputIdx = (iy * outputWidth + ix) * outputChannels + kz;
        if(activation){
            d_output[outputIdx] = activation_tanh(value);
        }
        else{
            d_output[outputIdx] = value;
        }
        
    }
}


//conv2 padding same

__global__ void conv2DKernelSamePadding(float *d_input, float *d_output, float *d_filter, int width, int height, int filterWidth, int inputChannels, int outputChannels, bool activation = false) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    int filterRadius = filterWidth / 2;

    if (ix < width && iy < height && kz < outputChannels) {
        float value = 0.0f;

        for (int channel = 0; channel < inputChannels; ++channel) {
            for (int fx = 0; fx < filterWidth; ++fx) {
                for (int fy = 0; fy < filterWidth; ++fy) {
                    int imgX = ix + fx - filterRadius;
                    int imgY = iy + fy - filterRadius;

                    if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                        int inputIdx = (imgY * width + imgX) * inputChannels + channel;
                        int filterIdx = ((kz * filterWidth + fy) * filterWidth + fx) * inputChannels + channel;
                        value += d_filter[filterIdx] * d_input[inputIdx];
                    }
                }
            }
        }

        int outputIdx = (iy * width + ix) * outputChannels + kz;
        d_output[outputIdx] = value;
    }
}


//conv2d combination of last 2 functions in one

__global__ void conv2DKernelCombined(float *d_input, float *d_output, float *d_filter, int width, int height, int filterWidth, int inputChannels, int outputChannels, bool useSamePadding=false, bool activation =false) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int kz = blockIdx.z * blockDim.z + threadIdx.z;

    int filterRadius = filterWidth / 2;
    int outputWidth = useSamePadding ? width : (width - filterWidth + 1);
    int outputHeight = useSamePadding ? height : (height - filterWidth + 1);

    if (ix < outputWidth && iy < outputHeight && kz < outputChannels) {
        float value = 0.0f;

        for (int channel = 0; channel < inputChannels; ++channel) {
            for (int fx = 0; fx < filterWidth; ++fx) {
                for (int fy = 0; fy < filterWidth; ++fy) {
                    int imgX = useSamePadding ? (ix + fx - filterRadius) : (ix + fx);
                    int imgY = useSamePadding ? (iy + fy - filterRadius) : (iy + fy);

                    if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                        int inputIdx = (imgY * width + imgX) * inputChannels + channel;
                        int filterIdx = ((kz * filterWidth + fy) * filterWidth + fx) * inputChannels + channel;
                        value += d_filter[filterIdx] * d_input[inputIdx];
                    }
                }
            }
        }

        int outputIdx = (iy * outputWidth + ix) * outputChannels + kz;
        if (activation) {
            d_output[outputIdx] = activation_tanh(value);
        } else {
            d_output[outputIdx] = value;
        }
    }
}

// Fixed version to deal with several filters
__global__ void convolution3D(float *input, float *output, float *kernel, int input_size, int kernel_size,int input_filters, int number_of_filters, bool useSamePadding, bool activation = false) {
    int output_size = useSamePadding ? input_size : input_size - kernel_size + 1;
    int pad = (kernel_size - 1) / 2;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;  
    int oy = blockIdx.y * blockDim.y + threadIdx.y;  
    int f_out = blockIdx.z;  // Filter index

    if (ox < output_size && oy < output_size && f_out < number_of_filters) {
        float sum = 0.0;

        for (int f_in = 0; f_in < input_filters; ++f_in) {
            for (int x = 0; x < kernel_size; ++x) {
                for (int y = 0; y < kernel_size; ++y) {
                    int input_x = useSamePadding ? ox + x - pad : ox + x;
                    int input_y = useSamePadding ? oy + y - pad : oy + y;

                    if (input_x >= 0 && input_x < input_size && input_y >= 0 && input_y < input_size) {
                        int input_offset = (f_in * input_size + input_x) * input_size + input_y;
                        int kernel_offset = (f_out * input_filters * kernel_size * kernel_size) + (f_in * kernel_size * kernel_size) + (x * kernel_size) + y;
                        sum += input[input_offset] * kernel[kernel_offset];
                    }
                }
            }
        }

        int output_offset = (f_out * output_size + ox) * output_size + oy;
        if(activation){
            output[output_offset] = activation_tanh(sum);
        }
        else{
            output[output_offset] = sum;
        }
        
    }
}




__global__ void averagePoolingKernel(float *d_input, float *d_output, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, int poolSize) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < outputWidth && iy < outputHeight && iz < channels) {
        float sum = 0.0f;
        int count = 0;

        // Calculate the start position for the pooling window
        int startX = ix * poolSize;
        int startY = iy * poolSize;

        for (int x = startX; x < startX + poolSize; ++x) {
            for (int y = startY; y < startY + poolSize; ++y) {
                if (x < inputWidth && y < inputHeight) {
                    sum += d_input[(y * inputWidth + x) * channels + iz];
                    count++;
                }
            }
        }

        if (count != 0) {
            d_output[(iy * outputWidth + ix) * channels + iz] = sum / count;
        } else {
            d_output[(iy * outputWidth + ix) * channels + iz] = 0;
        }
    }
}


__global__ void denseLayerKernel(float *d_input, float *d_output, float *d_weights, float *d_bias, int inputSize, int outputSize, bool applyActivation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < outputSize) {
        float value = 0.0f;
        for (int i = 0; i < inputSize; ++i) {
            value += d_weights[idx * inputSize + i] * d_input[i];
        }

        // Adding the bias if provided
        if (d_bias != nullptr) {
            value += d_bias[idx];
        }

        
        if (applyActivation) {
            
            value = tanhf(value);
        }

        d_output[idx] = value;
    }
}

void softmax(float* input, int size) {
    float maxElement = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > maxElement) {
            maxElement = input[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        //input[i] = exp(input[i] - maxElement);
        input[i] = exp(input[i]);
        sum += input[i];
    }

    for (int i = 0; i < size; ++i) {
        input[i] /= sum;
    }
}




__device__ float activation_tanh(float M) {
    return tanhf(M); 
}


