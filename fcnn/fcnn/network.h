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
	unsigned int sizes[3];	//sizes of layers' output
	float *weights[2];		//weights of connections between 0-1 and 1-2 layers
	float *outs[3];			//outputs of layers, 0 for network's input
	float *derivatives[2];	//derivatives for updating weights
	int k = 10;				//count of neurons in the second layer
	float eps = 0.1;		//error 

public:
	FCNetwork(int size, int hiddenLayerSize, int accuracy) {

		eps = accuracy;

		sizes[0] = size; //input verctor
		sizes[1] = hiddenLayerSize;
		sizes[2] = k;

		outs[0] = new float[size];
		outs[1] = new float[sizes[1]];
		outs[2] = new float[sizes[2]];

		derivatives[0] = new float[sizes[1]]; //for output from first layer
		derivatives[1] = new float[sizes[2]]; //for output from softmax layer

		weights[0] = new float[sizes[0]*sizes[1]];
		weights[1] = new float[sizes[1]*sizes[2]];

		for (int i = 0; i < sizes[0] * sizes[1]; i++) {
			weights[0][i] = random();
		}

		for (int i = 0; i < sizes[1] * sizes[2]; i++) {
			weights[1][i] = random();
		}

		for (int i = 0; i < sizes[1]; i++) {
			outs[1][i] = 0.0f;
		}

		for (int i = 0; i < sizes[2]; i++) {
			outs[2][i] = 0.0f;
		}
		cout << "Configuring done." << endl;
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
		float max = outs[2][0];
		int value = 0;
		for (int i = 0; i < sizes[2]; i++)
			if (outs[2][i] > max) {
				max = outs[2][i];  //to find predicted class
				value = i;			//predicted class
			}
		return value;
	}

	//activation function for the first layer (hidden layer)
	void activation() {
		for (int i = 0; i < sizes[1]; i++)
			outs[1][i] = tanh(outs[1][i]);
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

	void updateDerivatives(int label)
	{
		//for softmax layer
		for (int i = 0; i < sizes[2]; i++)
		{
			for (unsigned int j = 0; j < sizes[2]; j++)
				derivatives[1][i] = outs[2][j];
			derivatives[1][label] -= 1.f;
		}
		
		//for hidden layer (with tanh activation function)
		for (int i = 0; i < sizes[1]; i++)
		{
			derivatives[0][i] = 0.f;
			for (unsigned int j = 0; j < sizes[2]; j++)
				derivatives[0][i] += derivatives[1][j] * weights[1][j*sizes[2] + i];
			derivatives[0][i] *= 1.f - outs[1][i] * outs[1][i];
		}		
	}

	void directStep() {
		//outs after first layer (hidden)
		for (int i = 0; i < sizes[1]; i++) {
			for (int j = 0; j < sizes[0]; j++) {
				outs[1][i] += weights[0][i*sizes[0] + j] * outs[0][j];
			}
			outs[1][i] = tanh(outs[1][i]);
		}

		//outs after second layer (softmax)
		float max = 0.f;
		for (unsigned int i = 0; i < sizes[2]; ++i)
			if (outs[2][i] > max)
				max = outs[2][i];
		float sum = 0.f;
		for (int i = 0; i < sizes[2]; i++) {
			for (int j = 0; j < sizes[1]; j++) {
				outs[2][i] += weights[1][i*sizes[1] + j] * outs[1][j];
			}
			sum += expf(outs[2][i]);
		}
		sum = 1.f / sum;
		for (int i = 0; i < sizes[2]; i++)
			outs[2][i] = sum * expf(outs[2][i]);
	}
	void backPropagation(float learningRate, int label) {

		//for softmax layer
		for (int i = 0; i < sizes[2]; i++){
			derivatives[1][i] = outs[2][i];
		}
		derivatives[1][label] -= 1.f;
		//update weights
		for (int i = 0; i < sizes[2]; i++) {
			for (unsigned int j = 0; j < sizes[1]; j++) {
				weights[1][i*sizes[1] + j] -= derivatives[1][i] * outs[1][j] * learningRate;
			}
		}

		//for hidden layer (with tanh activation function)
		for (int i = 0; i < sizes[1]; i++)
		{
			derivatives[0][i] = 0.f;
			for (unsigned int j = 0; j < sizes[2]; j++)
				derivatives[0][i] += derivatives[1][j] * weights[1][j*sizes[2] + i];
			derivatives[0][i] *= 1.f - outs[1][i] * outs[1][i];
		}
		//update weights
		for (int i = 0; i < sizes[1]; i++) {
			for (unsigned int j = 0; j < sizes[0]; j++) {
				weights[0][i*sizes[0] + j] -= derivatives[0][i] * outs[0][j] * learningRate;
			}
		}
	}

	void teach(float learningRate, int label) {
		directStep();
		backPropagation(learningRate, label);
	}

	int test() {
		directStep();
		return predict();
	}
};