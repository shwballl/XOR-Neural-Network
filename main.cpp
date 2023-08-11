#include <iostream>
#include <cmath>

using namespace std;

//Simple nn that can learn ^ and || 

double initWeights(){
	return ((double)rand())/((double)RAND_MAX);
}

double sigmoidF(double x){return 1 / (1+exp(-x));};
double DsigmoidF(double x){return x*(1-x);};

void shuffle(int *array, size_t n){
	if(n > 1){
		for(size_t i = 0; i < n-1;i++){
			size_t j = i+rand() / (RAND_MAX / (n-i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
		}
	}
}



#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4



int main(void){
	const double lr = 0.2f; // Learn rate

	double hiddenL[numHiddenNodes];
	double outputL[numOutputs];

	double hiddenLayerBias[numHiddenNodes];
	double outputLayerBias[numOutputs];

	double hiddenW[numInputs][numHiddenNodes];
	double outputW[numHiddenNodes][numOutputs];


	double trainingInputs[numTrainingSets][numInputs] = {{0.0,0.0},{1.0, 0.0},{0.0,1.0},{1.0,1.0}};
	double trainingOutputs[numTrainingSets][numOutputs] = {{0.0},{1.0},{1.0},{1.0}};

	for(int i = 0; i < numInputs; i++){
		for(int j = 0; j < numHiddenNodes; j++){
			hiddenW[i][j] = initWeights();
		}
	}

	for(int i = 0; i < numHiddenNodes; i++){
		for(int j = 0; j < numOutputs; j++){
			outputW[i][j] = initWeights();
		}
	}

	for(int i = 0; i < numOutputs; i++){
		outputLayerBias[i] = initWeights();
	}

	int trainingSetOrder[] = {0,1,2,3};

	int numberOfEpochs = 10000;

	//Training nn
	for(int epoch = 0; epoch < numberOfEpochs; epoch++){
		shuffle(trainingSetOrder, numTrainingSets);
		
		for(int x =0; x < numTrainingSets;x++){
			int i = trainingSetOrder[x];
			for (int j = 0; j < numHiddenNodes; ++j)
			{
				double activation = hiddenLayerBias[j];

				for (int k = 0; k < numInputs; ++k)
				{
					activation += trainingInputs[i][k] *  hiddenW[k][j];
				}

				hiddenL[j] = sigmoidF(activation);
			}

			for (int j = 0; j < numOutputs; ++j)
			{
				double activation = outputLayerBias[j];

				for (int k = 0; k < numHiddenNodes; ++k)
				{
					activation += hiddenL[k] *  outputW[k][j];
				}

				outputL[j] = sigmoidF(activation);
			}

			cout << "Input: " << trainingInputs[i][0] << " " << trainingInputs[i][1]
					<< " Output: " << outputL[0] <<
					" Predicted: " << trainingOutputs[i][0] << endl;
			double deltaOut[numOutputs];

			for (int j = 0; j < numOutputs; ++j)
			{
				double error = (trainingOutputs[i][j] - outputL[j]);
				deltaOut[j] = error*DsigmoidF(outputL[j]);
			}

			double deltaHid[numHiddenNodes];

			for (int j = 0; j < numHiddenNodes; ++j)
			{
				double error = 0.0;
				for (int k = 0; k < numOutputs; ++k)
				{
					error += deltaOut[k]* outputW[j][k];
				}
				deltaHid[j] = error*DsigmoidF(hiddenL[j]);
			}

			for (int j = 0; j < numOutputs; ++j)
			{
				outputLayerBias[j] += deltaOut[j] * lr;
				for (int k = 0; k < numHiddenNodes; ++k)
				{
					outputW[k][j] += hiddenL[k]*deltaOut[j]*lr;
				}
			}

			for (int j = 0; j < numHiddenNodes; ++j)
			{
				hiddenLayerBias[j] += deltaHid[j] * lr;
				for (int k = 0; k < numInputs; ++k)
				{
					hiddenW[k][j] += trainingInputs[i][k]*deltaHid[j]*lr;
				}
			}

		}



	}
	cout << "Final Hidden Weights\n[";
			for (int j = 0; j < numHiddenNodes; ++j)
			{
				cout << "[";
				for (int k = 0; k < numInputs; ++k)
				{
					cout << " " << hiddenW[k][j];
				}
				cout << "]";
			}
			cout << "] \n";

			cout << "Final Output Weights\n[";
			for (int j = 0; j < numOutputs; ++j)
			{
				cout << "[";
				for (int k = 0; k < numHiddenNodes; ++k)
				{
					cout << " " << outputW[k][j];
				}
				cout << "]";
			}
			cout << "]\n";

			cout << "Final Hidden Biases\n[";
			for(const auto & el : hiddenLayerBias){
				cout << " " << el;
			}
			cout << "]\nFinal Output Biases\n[";
			for(const auto & el : outputLayerBias){
				cout << " " << el;
			}
			cout << "]\n";

	return 0;
}