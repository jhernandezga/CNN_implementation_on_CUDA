#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <c++/10/iomanip>

unsigned int bigEndianToLittleEndian(unsigned int value) {
    return ((value>>24)&0xff) | // Move byte 3 to byte 0
           ((value<<8)&0xff0000) | // Move byte 1 to byte 2
           ((value>>8)&0xff00) | // Move byte 2 to byte 1
           ((value<<24)&0xff000000); // byte 0 to byte 3
}

void charBckgrndPrint(char *str, int intensity) {
    printf("\033[48;2;%d;%d;%dm%s\033[0m", intensity, intensity, intensity, str);
}

void imgGrayPrint(int height, int width, int **img) {
    int row, col;
    char *str = "  "; // Two spaces for better visibility
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            charBckgrndPrint(str, img[row][col]);
        }
        printf("\n");
    }
}

void readImageAtIndex(int **img_gray, unsigned int selectedImageIndex, const char* filename) {
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;

    // Open File
    if((fptr = fopen(filename, "rb")) == NULL) {
        printf("Can't open file\n");
        exit(1);
    }

    // Read File Header
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    magic = bigEndianToLittleEndian(magic);
    nbImg = bigEndianToLittleEndian(nbImg);
    nbRows = bigEndianToLittleEndian(nbRows);
    nbCols = bigEndianToLittleEndian(nbCols);

    // Check if selectedImageIndex is valid
    if (selectedImageIndex >= nbImg) {
        printf("Selected image index is out of range\n");
        fclose(fptr);
        exit(1);
    }

    // Skip to the selected image
    fseek(fptr, 16 + selectedImageIndex * nbRows * nbCols, SEEK_SET);

    // Read the selected image
    for(int i = 0; i < nbRows; i++) {
        for(int j = 0; j < nbCols; j++) { 
            fread(&val, sizeof(unsigned char), 1, fptr);
            img_gray[i][j] = (int)val;
        }
    }

    fclose(fptr);
}


float* normalizeAndFlattenImage(int **img_2d, int height, int width) {
    // Find the maximum value in the image
    int maxVal = 255;
    /*for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (img_2d[i][j] > maxVal) {
                maxVal = img_2d[i][j];
            }
        }
    }

    // Check for division by zero
    if (maxVal == 0) {
        fprintf(stderr, "Maximum value in the image is zero\n");
        exit(1);
    }*/

    // Allocate memory for the normalized image
    float *normalized_img_1d = (float *)malloc(height * width * sizeof(float));
    if (normalized_img_1d == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Normalize and flatten the image
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            normalized_img_1d[i * width + j] = (float)img_2d[i][j] / maxVal;
        }
    }

    return normalized_img_1d;
}


void initializeWeights(const std::string& filename, float* weights, int width, int height, int channels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file.\n";
        throw std::runtime_error("File open failed");
    }

    int num_weights = width * height * channels;
    file.read(reinterpret_cast<char*>(weights), num_weights * sizeof(float));

    if (!file) {
        std::cerr << "Error occurred during file read. Read " << file.gcount() << " bytes.\n";
        if (file.eof()) {
            std::cerr << "End of file reached unexpectedly.\n";
        }
        throw std::runtime_error("File read failed");
    }

    file.close();
}

/*
void initializeWeights(const std::string& filename, float* weights, int num_weights) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file.\n";
        throw std::runtime_error("File open failed");
    }

    file.read(reinterpret_cast<char*>(weights), num_weights * sizeof(float));

    if (!file) {
        std::cerr << "Error occurred during file read. Read " << file.gcount() << " bytes.\n";
        if (file.eof()) {
            std::cerr << "End of file reached unexpectedly.\n";
        }
        throw std::runtime_error("File read failed");
    }

    file.close();
}*/


void printMatrix(const float* array, int width, int height, int channels) {
    for (int z = 0; z < channels; ++z) {
        std::cout << "Channel " << z + 1 << ":\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                std::cout << std::fixed << std::setprecision(2) << array[z * width * height + y * width + x] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void initializeWithRandomValues(float* array, int width, int height, int channels, int num_filters) {
    // Initialize the random number generator with a seed
    std::srand(std::time(0));

    int size = width * height * channels * num_filters;
    for (int i = 0; i < size; ++i) {
        array[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
}


void initializeWithZero(float* array, int width, int height, int channels) {
    int size = width * height * channels;
    for (int i = 0; i < size; ++i) {
        array[i] = 0;
    }
}

void printTensor(const float* array, int width, int height, int channels, int num_filters) {
    for (int n = 0; n < num_filters; ++n) {
        std::cout << "Filter " << n + 1 << ":\n";
        for (int z = 0; z < channels; ++z) {
            std::cout << "  Channel " << z + 1 << ":\n";
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = n * (width * height * channels) + z * (width * height) + y * width + x;
                    std::cout << std::fixed << std::setprecision(2) << array[index] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}



