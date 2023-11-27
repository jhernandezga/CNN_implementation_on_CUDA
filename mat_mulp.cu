#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
//#include <time.h> 



#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

void MatrixInit(float *M, int n, int p){

    srand(40);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            M[i*p + j] = (float)(2*(rand()/(float)(RAND_MAX)-0.5));
        }
    }

}

void MatrixPrint(float *M, int n, int p){
    
        for(int i = 0; i < n; i++){
            for(int j = 0; j < p; j++){
                printf("%f ", M[i*p + j]);
            }
            printf("\n");
        }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j++){
            Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
        }
    }
}

__global__ void cudaMatrixAdd_1(float *M1, float *M2, float *Mout, int n, int p){

    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    
    *(Mout+i) = *(M1+i) + *(M2+i);
        
    
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            Mout[i*n + j] = 0;
            for(int k = 0; k < n; k++){
                Mout[i*n + j] += M1[i*n + k] * M2[k*n + j];
            }
        }
    }
}

int main(int argc, char *argv[]){

    int n = 3;
    int p = 4;
    float *M1, *M2, *Mout, *Mout_mult;
    M1   = (float*)malloc(sizeof(float) * n*p);
    M2   = (float*)malloc(sizeof(float) * n*p);
    Mout   = (float*)malloc(sizeof(float) * n*p);
    Mout_mult   = (float*)malloc(sizeof(float) * n*p);

    float *M1_c, *M2_c, *Mout_c;

    //cudaMalloc((void**)&M, sizeof(float)*n*p);
    cudaMallocManaged((void**)&M1_c, sizeof(float)*n*p);
    cudaMallocManaged((void**)&M2_c, sizeof(float)*n*p);
    cudaMallocManaged((void**)&Mout_c, sizeof(float)*n*p);

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    MatrixInit(Mout, n, p);

    MatrixInit(M1_c, n, p);
    MatrixInit(M2_c, n, p);
    MatrixInit(Mout_c, n, p);

    cudaDeviceSynchronize();

    MatrixAdd(M1, M2, Mout, n, p);
    MatrixMult(M1, M2, Mout_mult, n);

    cudaMatrixAdd_1<<<n,p>>>(M1_c, M2_c, Mout_c, n, p);
    cudaDeviceSynchronize();
    
    MatrixPrint(M1_c, n, p);
    MatrixPrint(M2_c, n, p);
    //MatrixPrint(Mout_c, n, p);

    MatrixPrint(Mout_mult , n, p);

    cudaFree(M1_c);
    cudaFree(M2_c);
    cudaFree(Mout_c);
    return 0;
}
