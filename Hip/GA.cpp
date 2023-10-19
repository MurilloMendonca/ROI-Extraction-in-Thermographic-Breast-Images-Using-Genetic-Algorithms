#include "GA.hpp"
#include "Helper.hpp"
#include "Kernels.hpp"
#include <random>
#include <iostream>
#include <thread>

#include <hip/hip_runtime.h>

float COLOR_WEIGHTS[4] = {-90, 15, 25, -20.5};
int COLOR_INTERVAL[5] = {0, 45, 140, 200, 255};





GA::GA(cv::Mat img,int populationSize, float mutationRate, float elitismRate)
{
	this->populationSize = populationSize;
	this->mutationRate = mutationRate;
	this->elitismRate = elitismRate;
	this->img = img;

	// Flatten the image data
    int imageSize = img.rows * img.cols;
    imgData = new unsigned char[imageSize]; // ensure this is the correct type and size for your data

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            imgData[i * img.cols + j] = img.at<cv::Vec3b>(i, j)[0];
        }
    }

	hipError_t err;
    err = hipMalloc(&d_img, imageSize * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for image: " << hipGetErrorString(err) << std::endl;
        return;
    }

	// Copy data to device memory
    err = hipMemcpy(d_img, imgData, imageSize * sizeof(unsigned char), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying image data to device: " << hipGetErrorString(err) << std::endl;
        // Free resources before returning
        hipFree(d_img);
        return;
    }

	initializePopulation();
	
}

GA::GA(cv::Mat img){
	this->populationSize = 50; 
	this->mutationRate = 0.01; 
	this->elitismRate = 0.1; 
	this->img = img;

	// Flatten the image data
    int imageSize = img.rows * img.cols;
    imgData = new unsigned char[imageSize]; // ensure this is the correct type and size for your data

    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            imgData[i * img.cols + j] = img.at<cv::Vec3b>(i, j)[0];
        }
    }

	hipError_t err;
    err = hipMalloc(&d_img, imageSize * sizeof(unsigned char));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for image: " << hipGetErrorString(err) << std::endl;
        return;
    }

	// Copy data to device memory
    err = hipMemcpy(d_img, imgData, imageSize * sizeof(unsigned char), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying image data to device: " << hipGetErrorString(err) << std::endl;
        // Free resources before returning
        hipFree(d_img);
        return;
    }

	initializePopulation();

}

void GA::initializePopulation()
{
	//set up random seed
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> disX(0, img.cols);
	std::uniform_int_distribution<> disY(0, img.rows);
	std::uniform_int_distribution<> disSize(0, img.rows/2);
	for(int i = 0; i < populationSize; i++)
	{
		Cardioid cardioid = Cardioid(disX(gen), disY(gen), disSize(gen));
		Individual individual = Individual(cardioid);
		population.push_back(individual);
	}
	evaluatePopulation();
	std::sort(population.begin(), population.end());

	

}

void GA::evaluatePopulationInParallel(){
	std::vector<std::thread> threads;
	for(auto& individual : population)
	{
		threads.push_back(std::thread([](Individual& individual, GA* ga){
			if(!ga->isViable(individual))
			{
				individual.setFitness(0);
				return;
			}
			
			individual.setFitness(ga->calculateFitness(individual));
		}, std::ref(individual), this));
	}
	for(auto& thread : threads)
	{
		thread.join();
	}
}

void GA::evaluatePopulationInHip() {
   	int populationSize = population.size();
	int imageSize = img.rows * img.cols;

	// Prepare the cardioid data
    int* cardioidXs = new int[populationSize];
    int* cardioidYs = new int[populationSize];
    int* cardioidSizes = new int[populationSize];

    for (int i = 0; i < populationSize; ++i) {
        cardioidXs[i] = population[i].getCardioid().getX();
        cardioidYs[i] = population[i].getCardioid().getY();
        cardioidSizes[i] = population[i].getCardioid().getSize();
    }

    // Allocate device memory
    int* d_cardioidXs;
    int* d_cardioidYs;
    int* d_cardioidSizes;
    float* d_fitnesses;


    hipError_t err;
    err = hipMalloc(&d_cardioidXs, populationSize * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for cardioids: " << hipGetErrorString(err) << std::endl;
        hipFree(d_img); // Free other resources before returning
        return;
    }

	err = hipMalloc(&d_cardioidYs, populationSize * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for cardioids: " << hipGetErrorString(err) << std::endl;
        hipFree(d_img); // Free other resources before returning
        hipFree(d_cardioidXs); // Free other resources before returning
        return;
    }

	err = hipMalloc(&d_cardioidSizes, populationSize * sizeof(int));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for cardioids: " << hipGetErrorString(err) << std::endl;
        hipFree(d_img); // Free other resources before returning
        hipFree(d_cardioidXs); // Free other resources before returning
        hipFree(d_cardioidYs); // Free other resources before returning
        return;
    }

    err = hipMalloc(&d_fitnesses, img.rows*populationSize * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating device memory for fitnesses: " << hipGetErrorString(err) << std::endl;
        hipFree(d_img); // Free other resources before returning
        hipFree(d_cardioidXs); // Free other resources before returning
        hipFree(d_cardioidYs); // Free other resources before returning
        hipFree(d_cardioidSizes); // Free other resources before returning
        return;
    }

    

    err = hipMemcpy(d_cardioidXs, cardioidXs, populationSize * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying cardioids data to device: " << hipGetErrorString(err) << std::endl;
        // Free resources before returning
        hipFree(d_img);
        hipFree(d_cardioidXs); // Free other resources before returning
        hipFree(d_cardioidYs); // Free other resources before returning
        hipFree(d_cardioidSizes); // Free other resources before returning
        hipFree(d_fitnesses);
        return;
    }

	err = hipMemcpy(d_cardioidYs, cardioidYs, populationSize * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying cardioids data to device: " << hipGetErrorString(err) << std::endl;
        // Free resources before returning
        hipFree(d_img);
        hipFree(d_cardioidXs); // Free other resources before returning
        hipFree(d_cardioidYs); // Free other resources before returning
        hipFree(d_cardioidSizes); // Free other resources before returning
        hipFree(d_fitnesses);
        return;
    }

	err = hipMemcpy(d_cardioidSizes, cardioidSizes, populationSize * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying cardioids data to device: " << hipGetErrorString(err) << std::endl;
        // Free resources before returning
        hipFree(d_img);
        hipFree(d_cardioidXs); 
        hipFree(d_cardioidYs); 
        hipFree(d_cardioidSizes); 
        hipFree(d_fitnesses);
        return;
    }

	dim3 threadsPerBlock(256, 1);
	dim3 numBlocks((img.rows + threadsPerBlock.x - 1) / threadsPerBlock.x, populationSize); 

	calculateFitnessPerRowKernel<<<numBlocks, threadsPerBlock>>>(d_img, d_cardioidXs, d_cardioidYs, d_cardioidSizes, d_fitnesses, img.rows, img.cols, populationSize);
    // Check for kernel launch errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(err) << std::endl;
        // Free resources before returning
        hipFree(d_img);
       hipFree(d_cardioidXs); 
        hipFree(d_cardioidYs); 
        hipFree(d_cardioidSizes); 
        hipFree(d_fitnesses);
        return;
    }

    // Copy results back to host memory
    std::vector<float> fitnesses(populationSize*img.rows);
    err = hipMemcpy(&fitnesses[0], d_fitnesses, img.rows*populationSize * sizeof(float), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::cerr << "Error copying fitness results from device to host: " << hipGetErrorString(err) << std::endl;
        // Free resources before returning
        hipFree(d_img);
        hipFree(d_cardioidXs); // Free other resources before returning
        hipFree(d_cardioidYs); // Free other resources before returning
        hipFree(d_cardioidSizes); // Free other resources before returning
        hipFree(d_fitnesses);
        return;
    }
	float maxScore = img.cols*25;
	for(int i=0;i<populationSize;i++){
		float sum = 0;
		for(int j=0;j<img.rows;j++){
			if(fitnesses[i*img.rows+j]>maxScore)
				std::cout<<"Fitness for individual "<<i<<" is too high: "<<fitnesses[i*img.rows+j]<<std::endl;
			sum += fitnesses[i*img.rows+j];
			//std::cout<<"Partial fit for individual"<<i<<": "<<fitnesses[i*img.rows+j]<<std::endl;
		}
		population[i].setFitness(sum>0?sum:0);
	}


    

    // Free device memory
    
    hipFree(d_cardioidXs); // Free other resources before returning
    hipFree(d_cardioidYs); // Free other resources before returning
    hipFree(d_cardioidSizes); // Free other resources before returning
    hipFree(d_fitnesses);
	delete[] cardioidXs;
	delete[] cardioidYs;
	delete[] cardioidSizes;
}


/*
void GA::evaluatePopulationInParallel(){
	int numThreads = 12;
	std::vector<std::thread> threads;
	for(int t=0;t<numThreads;t++){
		threads.push_back(std::thread([](int index, int numThreads, std::vector<Individual>& population, const cv::Mat& img, GA* ga){
			int initialIndividual = index*population.size()/numThreads;
			int finalIndividual = (index+1)*population.size()/numThreads;
			for(int i = initialIndividual; i < finalIndividual; i++)
			{
				if(!ga->isViable(population[i]))
				{
					population[i].setFitness(0);
					continue;
				}
				
				population[i].setFitness(ga->calculateFitness(population[i]));
			}
		},t, numThreads, std::ref(population), std::ref(img), this));
	}
	for(auto& thread : threads)
	{
		thread.join();
	}
}
*/
void GA::evaluatePopulation()
{

	for(auto& individual : population)
	{
		if(!isViable(individual))
		{
			individual.setFitness(0);
			continue;
		}
		
		individual.setFitness(calculateFitness(individual));
	}
}
float GA::calculateFitness(Individual individual){
	float fitness = 0;
	//loop over the pixels of the image
	for(int i = 0; i < img.rows; i++)
	{
		for(int j = 0; j < img.cols; j++)
		{
			//if the pixel is inside the cardioid
			if(individual.getCardioid().isInside(i, j))
			{
				//get the color of the pixel in a gray scale image
				int color = img.at<cv::Vec3b>(i, j)[0];
				//std::cout<<"color: "<<color<<std::endl;
				
				if(color< COLOR_INTERVAL[1])
				{
					fitness += COLOR_WEIGHTS[0];
				}
				else if(color < COLOR_INTERVAL[2])
				{
					fitness += COLOR_WEIGHTS[1];
				}
				else if(color < COLOR_INTERVAL[3])
				{
					fitness += COLOR_WEIGHTS[2];
				}
				else
				{
					fitness += COLOR_WEIGHTS[3];
				}

			}
		}
	}
	
	float totalWeight = COLOR_WEIGHTS[0] + COLOR_WEIGHTS[1] + COLOR_WEIGHTS[2] + COLOR_WEIGHTS[3];
	if(fitness<0)
		fitness = 0;
	else
		fitness = fitness/abs(totalWeight);
	return fitness;
}

float GA::calculateFitnessInParallel(Individual individual){
	
	int numThreads = 4;

	std::vector<std::thread> threads;
	std::vector<float> fitnesses(numThreads);

	for(int t=0;t<numThreads; t++){
		threads.push_back(std::thread([](int index, Individual individual, int numThreads, float& fitness, const cv::Mat& img){
			int initialRow = index*individual.getCardioid().getSize()/numThreads;
			int finalRow = (index+1)*individual.getCardioid().getSize()/numThreads;
			//loop over the pixels of the image
			for(int i = initialRow; i < finalRow; i++)
			{
				for(int j = 0; j < img.cols; j++)
				{
					//if the pixel is inside the cardioid
					if(individual.getCardioid().isInside(i, j))
					{
						//get the color of the pixel in a gray scale image
						int color = img.at<cv::Vec3b>(i, j)[0];
						//std::cout<<"color: "<<color<<std::endl;
						
						if(color< COLOR_INTERVAL[1])
						{
							fitness += COLOR_WEIGHTS[0];
						}
						else if(color < COLOR_INTERVAL[2])
						{
							fitness += COLOR_WEIGHTS[1];
						}
						else if(color < COLOR_INTERVAL[3])
						{
							fitness += COLOR_WEIGHTS[2];
						}
						else
						{
							fitness += COLOR_WEIGHTS[3];
						}

					}
				}
			}
		},t,individual, numThreads, std::ref(fitnesses[t]), std::ref(img)));
	}
	
	for (auto& thread : threads) {
		thread.join();
	}
	float fitness = 0;
	for(int i=0; i<numThreads; i++){
		fitness += fitnesses[i];
	}
	float totalWeight = COLOR_WEIGHTS[0] + COLOR_WEIGHTS[1] + COLOR_WEIGHTS[2] + COLOR_WEIGHTS[3];
	if(fitness<0)
		fitness = 0;
	else
		fitness = fitness/abs(totalWeight);
	return fitness;
}

void GA::mutatePopulation()
{
	for(auto& individual : population)
	{
		mutation(individual);
	}
}

void GA::mutation(Individual& i){
	std::string genes = i.getGenotype();
	//set up random seed
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> mut(0, 1);

	for(int j = 0; j < genes.size(); j++)
	{
		if(mut(gen) < mutationRate)
		{
			genes[j] = genes[j] == '0' ? '1' : '0';
		}
	}

	i.setGenotypeAndUpdateCardioid(genes);
	
}

std::pair<Individual,Individual> GA::selection(){
	//Get a pair of distinct Individuals using tournament selection
	std::pair<Individual, Individual> parents;
	std::vector<Individual> tournament;
	//set up random seed
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, populationSize - 1);

	tournament.clear();
	for(int j = 0; j < 10; j++)
	{
		tournament.push_back(population[dis(gen)]);
	}
	std::sort(tournament.begin(), tournament.end());
	parents.first = tournament[0];
	parents.second = tournament[1];
	int i = 2;
	while(parents.first == parents.second && i < 10){
		parents.second = tournament[i++];
	}

	return parents;
}

bool GA::isViable(Individual i){
	Cardioid cardioid = i.getCardioid();
	return cardioid.getX() >= 0 && 
			cardioid.getX()  <= img.cols &&
			cardioid.getY()  >= 0 &&
			cardioid.getY()  <= img.rows;
}

void GA::crossoverPopulation(){
	std::vector<Individual> newPopulation;
	while(newPopulation.size() < populationSize)
	{
		std::pair<Individual, Individual> parents = selection();
		std::pair<Individual, Individual> children = crossover(parents.first, parents.second);
		newPopulation.push_back(children.first);
		newPopulation.push_back(children.second);
	}
	population = newPopulation;
}

std::pair<Individual, Individual> GA::crossover(Individual ind1, Individual ind2){
	std::string genes1 = ind1.getGenotype();
	std::string genes2 = ind2.getGenotype();
	//set up random seed
	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_int_distribution<> dis(0, genes1.size() - 1);
	int crossoverPoint = dis(gen);

	std::string newGenes1 = genes1.substr(0, crossoverPoint) + genes2.substr(crossoverPoint, genes2.size() - crossoverPoint);
	std::string newGenes2 = genes2.substr(0, crossoverPoint) + genes1.substr(crossoverPoint, genes1.size() - crossoverPoint);
	


	
	return std::make_pair(Individual(newGenes1), Individual(newGenes2));
}

void GA::run(int g){ 
	for(int i = 0; i < g; i++)
	{
		std::vector<Individual> elite;
		for(int j = 0; j < populationSize * elitismRate; j++)
		{
			elite.push_back(population[j]);
		}
		crossoverPopulation();
		mutatePopulation();
		evaluatePopulationInHip();
		population.insert(population.end(), elite.begin(), elite.end());
		std::sort(population.begin(), population.end());
		
		population.erase(population.begin() + populationSize, population.end());
		//std::cout << "Generation " << i << " best fitness: " << population[0].getFitness() << std::endl;
	}
}

void GA::runWithTimings(int g){
	std::chrono::duration<double> eliteTotal;
	std::chrono::duration<double> crossoverTotal;
	std::chrono::duration<double> mutationTotal;
	std::chrono::duration<double> evaluationTotal;
	std::chrono::duration<double> sortingTotal;
	std::chrono::duration<double> total;

	std::chrono::_V2::system_clock::time_point start;
	std::chrono::_V2::system_clock::time_point stop;
	for(int i = 0; i < g; i++)
	{
		start = std::chrono::high_resolution_clock::now();
		std::vector<Individual> elite;
		for(int j = 0; j < populationSize * elitismRate; j++)
		{
			elite.push_back(population[j]);
		}
		stop = std::chrono::high_resolution_clock::now();
		eliteTotal += stop - start;

		start = std::chrono::high_resolution_clock::now();
		crossoverPopulation();
		stop = std::chrono::high_resolution_clock::now();
		crossoverTotal += stop - start;

		start = std::chrono::high_resolution_clock::now();
		mutatePopulation();
		stop = std::chrono::high_resolution_clock::now();
		mutationTotal += stop - start;

		start = std::chrono::high_resolution_clock::now();
		evaluatePopulationInHip();
		stop = std::chrono::high_resolution_clock::now();
		evaluationTotal += stop - start;

		start = std::chrono::high_resolution_clock::now();
		population.insert(population.end(), elite.begin(), elite.end());
		stop = std::chrono::high_resolution_clock::now();
		eliteTotal += stop - start;

		start = std::chrono::high_resolution_clock::now();
		std::sort(population.begin(), population.end());
		stop = std::chrono::high_resolution_clock::now();
		sortingTotal += stop - start;
		
		start = std::chrono::high_resolution_clock::now();
		population.erase(population.begin() + populationSize, population.end());
		stop = std::chrono::high_resolution_clock::now();
		eliteTotal += stop - start;
		//std::cout << "Generation " << i << " best fitness: " << population[0].getFitness() << std::endl;
	}

	total = eliteTotal + crossoverTotal + mutationTotal + evaluationTotal + sortingTotal;

	std::cout << "Elite time: " << eliteTotal.count() << " s; Percentage: "<< 100.0*eliteTotal.count()/total.count()<<"% \n";
	std::cout << "Crossover time: " << crossoverTotal.count() << " s; Percentage: "<< 100.0*crossoverTotal.count()/total.count()<<"% \n";
	std::cout << "Mutation time: " << mutationTotal.count() << " s; Percentage: "<< 100.0*mutationTotal.count()/total.count()<<"% \n";
	std::cout << "Evaluation time: " << evaluationTotal.count() << " s; Percentage: "<< 100.0*evaluationTotal.count()/total.count()<<"% \n";
	std::cout << "Sorting time: " << sortingTotal.count() << " s; Percentage: "<< 100.0*sortingTotal.count()/total.count()<<"% \n";
	std::cout << "Total time: " << total.count() << " s\n";
}

Individual GA::getBestIndividual(){
	return population[0];
}

void GA::showPopulation(){
	for(int i = 0; i < populationSize; i++)
	{
		std::cout<<i<<": "<<population[i].getFitness()<<"x: "<<population[i].getCardioid().getX()<<" y: "<<population[i].getCardioid().getY()<<" size: "<<population[i].getCardioid().getSize()<<std::endl;
	}
}

void GA::savePopulation(std::string path){
	for(int i = 0; i < populationSize; i++)
	{
		std::string name = path + "/ind_" + std::to_string(i) + ".jpg";
		population[i].saveCardioidImage(img, name);
	}
}

GA::~GA(){
	hipFree(d_img);
	delete[] imgData;
}