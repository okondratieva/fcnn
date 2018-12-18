#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>

#define USE_MNIST_LOADER
#define MNIST_FLOAT

#include "mnist.h"
#include "network.h"

using namespace std;

int main(int argc, char **argv)
{
	if (argc < 4) {
		cout << "Arguments aren't enough. Launch format:" << endl;
		cout << "[learning_rate] [epoch] [hidden layer size]" << endl;
		return 1;
	}
	srand(time(0));

	mnist_data *train;
	mnist_data *test;
	unsigned int cntTrain;
	unsigned int cntTest;
	
	float learningRate = atof(argv[1]);
	unsigned int epochCount = atoi(argv[2]);
	unsigned int hiddenSize = atoi(argv[3]);
	unsigned int imageSize = 28 * 28;

	cout << "Parameters: " << endl;
	cout << "learning rate: " << learningRate<<endl;
	cout << "epochs: " << epochCount << endl;
	cout << "hidden neurons: " << hiddenSize << endl;

	//load dataset
	mnist_load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &train, &cntTrain);
	mnist_load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", &test, &cntTest);

	cout << "Train count: " << cntTrain << endl;
	cout << "Test count: " << cntTest << endl;
	cout << "Data was loaded" << endl;
	
	//configuring network
	FCNetwork network(imageSize, hiddenSize);
	cout << "Network was configured" << endl;

	//for suffle procedure
	vector<unsigned int> indexes; 
	indexes.resize(cntTrain);

	//train and test procedures 
	for (unsigned int i = 0; i < cntTrain; i++) {
		indexes[i] = i;
	}
	cout << "Start train" << endl;

	for (unsigned int epoch = 0; epoch < epochCount; epoch++)
	{
		cout << "Epoch: " << epoch << endl;
		random_shuffle(indexes.begin(), indexes.end());

		for (unsigned int i = 0; i < cntTrain; i++)
		{			
			network.set(train[indexes[i]].data);
			network.teach(learningRate, train[indexes[i]].label);
		}
	}
	cout << "Start test" << endl;
	unsigned int accuracy = 0;
	int res, gold;
	for (unsigned int i = 0; i < cntTest; i++)
	{
		network.set(test[i].data);
		res = network.test();
		gold = test[i].label;
		if (res == gold)
			accuracy++;
	}

	cout << "accuracy: " << (float)accuracy/cntTest*100 << endl;
	system("pause");
	return 0;
}