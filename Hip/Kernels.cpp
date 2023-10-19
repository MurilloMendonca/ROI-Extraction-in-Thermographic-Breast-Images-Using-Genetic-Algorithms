#include "Kernels.hpp"

__device__ bool isInside(int c_x, int c_y, int c_size, int x, int y) {
	float r = c_size / 2;

	float distance = sqrt(pow(x - c_x, 2) + pow(y - c_y, 2));
	float theta = atan2(y - c_y, x - c_x);

	bool cardioidEquation = pow(distance, 2) <= pow(r,2)*(1 - sin(theta));

	return cardioidEquation;
}

__global__ void calculateFitnessKernel(unsigned char* img, int* cardioidXs, int* cardioidYs, int* cardioidSizes, float* fitnesses, int imgRows, int imgCols, int imageSize, int populationSize) {
    int individualIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (individualIndex >= populationSize) return; // Ensure we do not go out of bounds

    float fitness = 0.0f;
    int c_x = cardioidXs[individualIndex];
    int c_y = cardioidYs[individualIndex];
    int c_size = cardioidSizes[individualIndex];

    for (int i = 0; i < imgRows; ++i) {
        for (int j = 0; j < imgCols; ++j) {
            // Use the adjusted isInside method
            if (isInside(c_x, c_y, c_size, i, j)) {
                // Access the color using .val field
                int color = img[i * imgCols + j];
				//printf("color: %d\n", color);

                if (color <45) {
                    fitness += -90;
                } else if (color < 140) {
                    fitness += 15;
                } else if (color < 200) {
                    fitness += 25;
                } else {
                    fitness += -20.5;
                }
            }
        }
    }
	//printf("\nFound fitness on individual %i: %f\n", individualIndex, fitness);
    float totalWeight = -90 + 15 +25 -20.5;
    if (fitness < 0)
        fitness = 0;
    else
        fitness = fitness / abs(totalWeight);

    fitnesses[individualIndex] = fitness;
}


__global__ void calculateFitnessPerRowKernel(const uchar* flat_img, int* cardioidXs, int* cardioidYs, int* cardioidSizes, float* fitness_results, int imgRows, int imgCols, int numIndividuals) {
    int rowIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int individualIndex = blockIdx.y; 

    if (rowIndex < imgRows) {
        float rowFitness = 0.0f;
        for (int colIndex = 0; colIndex < imgCols; ++colIndex) {
            int imgIndex = rowIndex * imgCols + colIndex;
            uchar color = flat_img[imgIndex]; 

            int c_x = cardioidXs[individualIndex];
            int c_y = cardioidYs[individualIndex];
            int c_size = cardioidSizes[individualIndex];

            if (isInside(c_x, c_y, c_size, colIndex, rowIndex)) { 
                if (color <45) {
                    rowFitness += -90;
                } else if (color < 140) {
                    rowFitness += 15;
                } else if (color < 200) {
                    rowFitness += 25;
                } else {
                    rowFitness += -20.5;
                }
            }
        }
        fitness_results[individualIndex*imgRows + rowIndex] = rowFitness;
    }
}
