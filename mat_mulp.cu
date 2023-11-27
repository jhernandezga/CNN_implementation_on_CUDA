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

int main(int argc, char *argv[]){

    int n = 3;
    int p = 4;
    float *M1, *M2, *Mout;
    M1   = (float*)malloc(sizeof(float) * n*p);
    M2   = (float*)malloc(sizeof(float) * n*p);
    Mout   = (float*)malloc(sizeof(float) * n*p);

    //cudaMalloc((void**)&M, sizeof(float)*n*p);
    //cudaMallocManaged(&M, sizeof(float)*n*p);

    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    MatrixInit(Mout, n, p);

    //cudaDeviceSynchronize();

    MatrixPrint(M1, n, p);
    MatrixPrint(M2, n, p);

    MatrixAdd(M1, M2, Mout, n, p);
    
    MatrixPrint(Mout, n, p);


    //cudaFree(M);

    return 0;
}
