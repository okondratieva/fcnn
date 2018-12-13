
#include <iostream>
#include <ctime>

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
	unsigned int cnt_train;
	unsigned int cnt_test;

	mnist_load("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &train, &cnt_train);
	mnist_load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", &test, &cnt_test);

	float learningRate = atof(argv[1]);
	unsigned int epoch = atoi(argv[2]);

	FCNetwork network();

	//train procedures 

	return 0;
}