#include <fstream>
#include <iostream>
#include <c++/10/iomanip>

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
int main() {
    // Define the number of weights to read
    // This should match the number of weights in the layer

    int width = 28;
    int height = 28;
    int inputChannels_C1 = 1;

    int filterSize = 5;

    int outputChannels_C1 = 6;

    int width_C1 = width;  //PADDING SAME: 28x28
    int height_C1 = height;
    int num_weights = 400*120;
    //int num_weights = filterSize*filterSize*outputChannels_C1;

    std::ifstream file("layer_weights_5.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file.\n";
        return 1;
    }

    float* weights = new float[num_weights];

    file.read(reinterpret_cast<char*>(weights), num_weights * sizeof(float));

    if (!file) {
        std::cerr << "Error occurred during file read. Read " << file.gcount() << " bytes.\n";
        if (file.eof()) {
            std::cerr << "End of file reached unexpectedly.\n";
        }
        delete[] weights;
        return 1;
    }

    printMatrix(weights, 400, 120, 1);

    delete[] weights;
    file.close();
}
