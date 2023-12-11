#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h> 


void MatrixInit(float *M, int n, int p, int l, bool init_zero){

    srand(40);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            for(int k = 0; k < l;k++){
                if(init_zero){
                    M[i*p + j*l + k] = 0;    
                }
                else{
                    M[i*p + j*l + k] = (float)(rand()/(float)(RAND_MAX));    
                }
            }
                       }
    }

}

void MatrixPrint(float *M, int n, int p,int l){
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            for(int k = 0; k < l; k++){
                printf("%f ", M[i*p + j*l + k]);
            }
            
        }
        printf("\n");
    }
}

void Matrix_kernel_init(float *M, int n, int p,int l){
    
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            for(int k = 0; k < l; k++){
                printf("%f ", M[i*p + j*l + k]);
            }
            
        }
        printf("\n");
    }
}

void conv2D(float *M1, float *kernel, float *Mout, int input_size, int kernel_size, int n_channel){

    //Stride = 1

    
    int n_points = input_size-(kernel_size-1)+1;
    
    float sum = 0;
    for(int k = 0; k < n_channel; k++){
        for(int i =0; i < n_points;i++){
            for(int j =0; j < n_points;j++){
                    for(int c = 0; c < kernel_size;c++){
                        sum = kernel[c]*M1[n_points*i+n_points*n_points*k+j+c];
                    }    
                }        
            }
    }

    float sum = 0;

    if(i <= n_points){

        for(int k = 0; k < kernel_size; k++){
            for(int c = 0; c < kernel_size; c++){
                sum += kernel[k*kernel_size + c ]*M1[i+j+k*kernel_size+c];
            }
    }
        }    
        

    Mout[i*n_points + j] = sum; 
    
}


__global__ void cudaConvolution(float *M1, float *kernel, float *Mout, int input_size, int kernel_size){

    //Stride = 1

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int n_points = input_size-(kernel_size-1)+1;
    
    float sum = 0;

    if(i <= n_points){

        for(int k = 0; k < kernel_size; k++){
            for(int c = 0; c < kernel_size; c++){
                sum += kernel[k*kernel_size + c ]*M1[i+j+k*kernel_size+c];
            }
    }
        }    
        

    Mout[i*n_points + j] = sum; 
    
}

__global__ void cudaMatrixMult_dot(float *M1, float *M2, float *Mout, int n){

    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    
    *(Mout+i) = *(M1+i)*(*(M2+i));
}




int main(int argc, char *argv[]){

    float *raw_data, *C1_data, *S1_data, *C1_kernel, *Mout;

    // Allocate device memory
    cudaMallocManaged((void**)&raw_data, sizeof(float)*32*32);
    cudaMallocManaged((void**)&C1_data, sizeof(float)*7);
    cudaMallocManaged((void**)&S1_data, sizeof(float)*14*14*6);
    cudaMallocManaged((void**)&C1_kernel, sizeof(float)*3);
    cudaMallocManaged((void**)&Mout, sizeof(float)*5);


    // Copy data from host to device
    float host_C1_data[] = {-1, 4, 11, 14, 21, 25, 30};
    float host_C1_kernel[] = {0.33, 0.33, 0.33};
    float host_Mout[] = {0, 0, 0, 0, 0};

    cudaMemcpy(C1_data, host_C1_data, sizeof(float)*7, cudaMemcpyHostToDevice);
    cudaMemcpy(C1_kernel, host_C1_kernel, sizeof(float)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(Mout, host_Mout, sizeof(float)*5, cudaMemcpyHostToDevice);
    
    cudaConvolution<<<1, 5>>>(C1_data, C1_kernel, Mout, 7, 3);
    
    cudaDeviceSynchronize();

    for(int i = 0; i < 5; i++){
        printf("%f ", Mout[i]);    
    }
    //MatrixPrint(C1_kernel,5,5,6);
    
     // Free device memory
     cudaFree(raw_data);
     cudaFree(C1_data);
     cudaFree(S1_data);
     cudaFree(C1_kernel);
     cudaFree(Mout);
}