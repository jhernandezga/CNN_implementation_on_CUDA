# LeNet-5 CUDA Implementation

### Authors: Jorge Hernández and Kia Zegrati 
This project presents the implementation of the LeNet-5 convolutional neural network using CUDA, the parallel computing platform and API model created by NVIDIA.This implementation showcases the power of GPU-accelerated computing in processing neural networks. Developed primarily for educational purposes, this work forms a part of the practical coursework in "Hardware for Signal Processing", a subject that delves into the hardware aspects of processing signals and data.
<div align="center">
  <img src="https://github.com/jhernandezga/CNN_implementation_on_CUDA/assets/57245076/2da81942-caf3-486b-98e6-a5d6f52a1bcc" width="400">
  <img src="https://github.com/jhernandezga/CNN_implementation_on_CUDA/assets/57245076/19ebb169-1f41-482e-909b-b36fb629024e" width="800" alt="CNN Implementation on CUDA">
<br>Image from: [The Convolutional Network](https://pabloinsente.github.io/the-convolutional-network)
</div>







## Description

LeNet-5, one of the earliest convolutional neural networks, remains a foundational architecture for modern deep learning in image recognition and computer vision. While the network is typically trained in Python due to its simplicity and widespread use in the deep learning community, this project uniquely employs CUDA for the inference phase, leveraging the computational power of NVIDIA GPUs for enhanced performance.

The project serves as a practical example for those looking to transition from Python to CUDA in deep learning. It provides a hands-on opportunity to explore the intricacies of GPU programming and understand how conventional neural networks can be adapted to harness the power of parallel computing.

## Key Features

- CUDA-optimized inference implementation of the LeNet-5 neural network.
- A demonstration of the transition from Python-based training to GPU-accelerated inference.
- Exploration of key deep learning concepts in a hardware-accelerated context.
- An educational tool for understanding the practical deployment of trained neural networks on GPUs.

## Getting Started

### Dependencies

- **Python**: Version 3.8 or above
- **CUDA Toolkit**: Version 11.2
- **NVIDIA GPU**: With CUDA Compute Capability 6.1 or higher
- **Operating System**: Tested on Ubuntu 22.04

### Instructions

#### Step 1: Downloading the Dataset
- Download the MNIST training dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=train-images.idx3-ubyte). Look for the file named `train-images.idx3-ubyte`.
- Create a folder in your project directory named `train-images`.
- Place the downloaded `train-images.idx3-ubyte` file inside the `train-images` folder.

#### Step 2: Configuring the Test Image
- Open the `main.cu` file in a text editor.
- Locate the section of the code where the test image index is set (look for a variable or a comment indicating this).
- Modify the image index to select a specific test image from the MNIST dataset. For example, set the image index to `5` to select the sixth image (assuming the indexing starts from `0`).

#### Step 3: Compilation
- Compile the CUDA code by opening a terminal or command prompt in your project directory.
- Run the following command to compile the project:

  ```bash
  nvcc -o main main.cu kernels.cu utils.cu
#### Step 4: Running the Program
After compilation, run the program by executing:
    ```bash 

            ./main


## Script Descriptions

This section provides an overview of the key scripts in the project, detailing their purpose and functionality.

### `main.cu`

- **Purpose**: This is the main script that orchestrates the execution of the LeNet-5 inference process.
- **Functionality**: 
  - Initializes the CUDA environment.
  - Loads the MNIST dataset image specified by the user.
  - Calls the necessary CUDA kernels for the inference process, implementing the stack of LeNet5
  - Outputs the inference results.

### `kernels.cu`

- **Purpose**: Contains the CUDA kernels used in the inference process.
- **Functionality**: 
  - Includes kernels for various operations like convolution, pooling, dense layers and activation functions.
  - Optimized for performance on NVIDIA GPUs.

### `utils.cu`

- **Purpose**: Provides utility functions that support the main inference process.
- **Functionality**: 
  - Includes functions for data loading, preprocessing, initialization, visualization and other auxiliary tasks.

### Additional Scripts

- `tests.cu`: Contains some toy testing examples for verifying that the CUDA kernels are correctly implemented
- `flat_cpu.cu`: some initial scripts for understaing the implementation of Conv2D in CPU
- `mat_mulp.cu`: initial scripts for understanding and testing the implementation of simple operations (matrix addition/multiplication) in CUDA and CPU

-`weights\LeNet5.ipynb`: Training of LeNet5and storing of the weights of the model



## How to Navigate the Codebase

- Start by reading `main.cu`, followed by `kernels.cu` for understanding the core logic.



## Current state

- All kernel and auxiliary operations have been implemented.
- The LeNet-5 architecture stack is fully implemented.
- Image loading and visualization for inference are implemented.
- The current performance is not as expected; further debugging is required. We believe the main issue right now is related to loading the weights. Despite thorough debugging of this step, more detailed and careful attention is necessary in how the weigths are stored and loaded. Additionally, properly testing the conv3d kernel with a generalized test might be needed.

The network is able to perform the operations for inference with the loaded weigths from the trained model, however the outputs are most of the time not coherent with what is expected.

## Contact Information
jhernandezga@unal.edu.co

### Outputs
Preview of some of the outputs with the current network:

![image](https://github.com/jhernandezga/CNN_implementation_on_CUDA/assets/57245076/33bc47a6-2a70-4362-a68b-a73d63ee0e67)
![image](https://github.com/jhernandezga/CNN_implementation_on_CUDA/assets/57245076/c18eb82e-21be-4ad2-b688-5928520672c2)
![image](https://github.com/jhernandezga/CNN_implementation_on_CUDA/assets/57245076/0050f7f3-7cb3-4468-b499-dba2d621d179)
![image](https://github.com/jhernandezga/CNN_implementation_on_CUDA/assets/57245076/a0071c8c-f401-42ba-a73c-426a1899865c)




