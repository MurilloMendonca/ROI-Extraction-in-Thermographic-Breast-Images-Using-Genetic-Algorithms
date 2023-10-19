#pragma once

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "Individual.hpp"


class GA{
	private:
		std::vector<Individual> population;
		float mutationRate;
		float elitismRate;
		int populationSize;
		cv::Mat img;
		unsigned char* imgData;

		void initializePopulation();
		void evaluatePopulation();
		void evaluatePopulationInParallel();
		void evaluatePopulationInHip();
		void mutatePopulation();
		void crossoverPopulation();

		std::pair<Individual, Individual> selection();
		std::pair<Individual, Individual> crossover(Individual parent1, Individual parent2);
		void mutation(Individual& individual);
		bool isViable(Individual individual);
		float calculateFitness(Individual individual);
		float calculateFitnessInParallel(Individual individual);
		float calculateFitnessInHip(Individual individual);

		unsigned char* d_img;

	public:
		GA(cv::Mat img,int populationSize, float mutationRate, float elitismRate);
		GA(cv::Mat img);
		~GA();

		void run(int generations);
		void runWithTimings(int generations);
		Individual getBestIndividual();

		void showPopulation();

		void savePopulation(std::string path);


};