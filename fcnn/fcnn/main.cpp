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
	/*if (argc < 5) {
		cout << "Arguments aren't enough. Launch format:" << endl;
		cout << "[learning_rate] [epoch] [hidden layer size] [path to data]" << endl;
		return 1;
	}*/
	srand(time(0));

	mnist_data *train;
	mnist_data *test;
	unsigned int cntTrain;
	unsigned int cntTest;
	
	float learningRate = 0.1;// atof(argv[1]);
	unsigned int epochCount = 5;// atoi(argv[2]);
	unsigned int hiddenSize = 20;//atoi(argv[3]);
	string pathToData = "path";// (argv[4]);
	unsigned int imageSize = 28 * 28;

	cout << "Parameters: " << endl;
	cout << "learning rate: " << learningRate<<endl;
	cout << "epochs: " << epochCount << endl;
	cout << "hidden neurons: " << hiddenSize << endl;
	cout << "path to data: " << pathToData << endl;

	//TODO: string to char

	mnist_load("C:\\study\\data\\train-images.idx3-ubyte", "C:\\study\\data\\train-labels.idx1-ubyte", &train, &cntTrain);
	mnist_load("C:\\study\\data\\t10k-images.idx3-ubyte", "C:\\study\\data\\t10k-labels.idx1-ubyte", &test, &cntTest);


	cout << "Train count: " << cntTrain << endl;
	cout << "Test count: " << cntTest << endl;
	cout << "Data was loaded" << endl;
	FCNetwork network(imageSize, hiddenSize, 0.01);
	cout << "Network was configured" << endl;

	vector<unsigned int> indexes; //for suffle procedure
	indexes.resize(cntTrain);


	//train and test procedures 
	for (unsigned int i = 0; i < cntTrain; ++i) {
		indexes[i] = i;
	}
	cout << "Start train" << endl;

	for (unsigned int epoch = 0; epoch < epochCount; epoch++)
	{
		cout << "Epoch: " << epoch << endl;
		random_shuffle(indexes.begin(), indexes.end());

		for (unsigned int i = 0; i < cntTrain; ++i)
		{
			
			network.set(train[indexes[i]].data);
			network.teach(learningRate, train[indexes[i]].label);
			if (i%1000 == 0) cout << "Progress: " << ((float)i / cntTrain) * 100 << endl;
		}
	}
	cout << "Start test" << endl;
	unsigned int accuracy = 0;
	int res, gold;
	for (unsigned int i = 0; i < cntTrain; ++i)
	{
		network.set(train[i].data);
		res = network.test();
		gold = train[i].label;
		if (res == gold)
			accuracy++;
		if (i % 1000 == 0) cout << "Progress: " << ((float)i / cntTrain) * 100 << endl;
	}

	cout << "accuracy: " << accuracy << endl;
	system("pause");
	return 0;
}