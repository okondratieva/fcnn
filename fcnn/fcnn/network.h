#pragma once
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

float random()
{
	return rand() / float(RAND_MAX);
}

class FCNetwork {
public:
	unsigned int sizes[3];//sizes of layers' output
	float *weights[2]; //weights of connections between 0-1 and 1-2 layers
	float *outs[3]; //outputs of layers, 0 for network's input
	int k = 10; //count of neurons in the second layer

public:
	FCNetwork(int size, int hiddenLayerSize) {
		
		sizes[0] = size;
		sizes[1] = hiddenLayerSize * size;
		sizes[2] = hiddenLayerSize * k;

		outs[0] = new float[size];
		weights[0] = new float[sizes[1]];
		weights[1] = new float[sizes[2]];

		for (int i = 0; i < sizes[1]; i++) {
			weights[0][i] = random();
		}

		for (int i = 0; i < sizes[2]; i++) {
			weights[1][i] = random();
		}
		
	}

	void set(float *inputs) {
		for (int i = 0; i < sizes[0]; i++) {
			outs[0][i] = inputs[i];
		}
	}

	float losses(int label) {
		return -logf(outs[2][label]); //cross-entropy
	}

	int predict() {
		float max = 0.f;
		int value = 0;
		for (int i = 0; i < sizes[2]; i++)
			if (outs[2][i] > max) {
				max = outs[2][i];//to find prediction class
				value = i;
			}
		return value;
	}

	void teach() {

	}

	//activation function for the first layer (hidden layer)
	void activation() {
		for (int i = 0; i < sizes[1]; i++)
			weights[0][i] = tanh(weights[0][i]);
	}

	//activation function for the second layer (softmax function)
	void softmax() {
		float sum = 0.f;
		for (int i = 0; i < sizes[2]; i++) 
			sum += expf(outs[2][i]);
		
		sum = 1.f / sum;

		for (int i = 0; i < sizes[2]; i++)
			outs[2][i] = expf(outs[2][i])*sum;
	}


};