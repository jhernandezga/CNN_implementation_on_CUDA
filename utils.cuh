#ifndef IMAGE_UTILS_CUH
#define IMAGE_UTILS_CUH

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

// Function to convert big endian to little endian
unsigned int bigEndianToLittleEndian(unsigned int value);

// Function to print a character with a grayscale background
void charBckgrndPrint(char *str, int intensity);

// Function to print the grayscale image
void imgGrayPrint(int height, int width, int **img);

// Function to read an image at a specified index from a file
void readImageAtIndex(int **img_gray, unsigned int selectedImageIndex, const char* filename);

// Function to flatten a 2D image into a 1D array
float* normalizeAndFlattenImage(int **img_2d, int height, int width);

// Function to initialize weights from a binary file
void initializeWeights(const std::string& filename, float* weights, int num_weights);

// Function to print a matrix (or array) in a formatted way
void printMatrix(const float* array, int width, int height, int channels);

void initializeWithRandomValues(float* array, int width, int height, int channels);

void initializeWithZero(float* array, int width, int height, int channels);

#endif // IMAGE_UTILS_CUH
