#pragma once
#include <hip/hip_runtime.h>

__device__ bool isInside(int c_x, int c_y, int c_size, int x, int y);

__global__ void calculateFitnessKernel(unsigned char* img, int* cardioidXs, int* cardioidYs, int* cardioidSizes, float* fitnesses, int imgRows, int imgCols, int imageSize, int populationSize);

__global__ void calculateFitnessPerRowKernel(const uchar* flat_img, int* cardioidXs, int* cardioidYs, int* cardioidSizes, float* fitness_results, int imgRows, int imgCols, int numIndividuals);
