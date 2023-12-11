// BEGIN: be15d9bcejpp

_global__ void cudaConvolution(float *M1, float *kernel, float *Mout, int input_size, int kernel_size){

    //Stride = 1

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int j = blockIdx.y * blockDim.y + threadIdx.y;

    int n_points = input_size-(kernel_size-1) + 1;
    
    int sum = 0;

    if(i < input_size-kernel_size){
        for(int k = 0; k < kernel_size; k++){
            sum += kernel[k]*M1[i+k];
        }    
    }

    M1[i] = sum; 
    
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


int main(int argc, char *argv[]){

    float *raw_data,*C1_data,*S1_data,*C1_kernel, *conv_output;


    cudaMallocManaged((void**)&raw_data, sizeof(float)*32*32);
    cudaMallocManaged((void**)&C1_data, sizeof(float)*28*28*6);
    cudaMallocManaged((void**)&S1_data, sizeof(float)*14*14*6);
    cudaMallocManaged((void**)&C1_kernel, sizeof(float)*5*5*6);
    cudaMallocManaged((void**)&conv_output, sizeof(float)*28*28*6);

    //initializations
    MatrixInit(raw_data,32,32,1,false);
    MatrixInit(C1_data,28,28,6,true);
    MatrixInit(S1_data,14,14,6,true);
    MatrixInit(C1_kernel,5,5,6,false);
    cudaDeviceSynchronize();

    MatrixPrint(C1_kernel,5,5,6);

    // Test the cudaConvolution function
    int input_size = 28*28*6;
    int kernel_size = 5;

    // Set the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (input_size + threadsPerBlock - 1) / threadsPerBlock;

    // Call the cudaConvolution function
    cudaConvolution<<<blocksPerGrid, threadsPerBlock>>>(C1_data, C1_kernel, conv_output, input_size, kernel_size);
    cudaDeviceSynchronize();

    // Print the output of the convolution
    MatrixPrint(conv_output, 28, 28, 6);

    // Free the allocated memory
    cudaFree(raw_data);
    cudaFree(C1_data);
    cudaFree(S1_data);
    cudaFree(C1_kernel);
    cudaFree(conv_output);

    return 0;
}
// END: be15d9bcejpp