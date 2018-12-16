
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
	if (argc < 5) {
		cout << "Arguments aren't enough. Launch format:" << endl;
		cout << "[learning_rate] [epoch] [hidden layer size] [path to data]" << endl;
		return 1;
	}

	mnist_data *train;
	mnist_data *test;
	unsigned int cntTrain;
	unsigned int cntTest;
	
	float learningRate = atof(argv[1]);
	unsigned int epochCount = atoi(argv[2]);
	unsigned int hiddenSize = atoi(argv[3]);
	string pathToData(argv[4]);
	unsigned int imageSize = 28 * 28;

	cout << "Parameters: " << endl;
	cout << "learning rate: " << learningRate<<endl;
	cout << "epochs: " << epochCount << endl;
	cout << "hidden neurons: " << hiddenSize << endl;
	cout << "path to data: " << pathToData << endl;

	//TODO: string to char

	mnist_load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &train, &cntTrain);
	mnist_load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", &test, &cntTest);


	FCNetwork network(imageSize, hiddenSize, 0.1);
	vector<unsigned int> indexes; //for suffle procedure
	indexes.resize(cntTrain);

	//train and test procedures 
	for (unsigned int i = 0; i < cntTrain; ++i)
		indexes[i] = i;

	for (unsigned int epoch = 0; epoch < epochCount; epoch++)
	{
		random_shuffle(indexes.begin(), indexes.end());
		for (unsigned int i = 0; i < cntTrain; ++i)
		{
			network.set(train[indexes[i]].data);
			network.teach(train[indexes[i]].label, learningRate);
			if (i % 100 == 0)
				cout << "\repoch: " << epoch << " progress: " << (i / double(cntTrain)) * 100 << "%         ";
		}
	}

	unsigned int accuracy = 0;
	int res;
	for (unsigned int i = 0; i < cntTest; ++i)
	{
		network.set(test[i].data);
		res = network.test();
		if (res == test[i].label)
			++accuracy;
	}

	cout << "\raccuracy: " << (double(accuracy) / cntTest) * 100 << "%          " << endl;
	return 0;
}