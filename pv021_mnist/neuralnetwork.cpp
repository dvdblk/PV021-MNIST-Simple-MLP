//
//  neuralnetwork.cpp
//  pv021_mnist_project
//
//  Created by David Bielik on 06/11/2017.
//  Copyright © 2017 David Bielik. All rights reserved.
//

#include "neuralnetwork.hpp"
#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

using namespace std;

// File names
const string kMnistTrainVectorsFileName = "MNIST_DATA/mnist_train_vectors.csv";
const string kMnistTrainLabelsFileName = "MNIST_DATA/mnist_train_labels.csv";
const string kMnistTestVectorsFileName = "MNIST_DATA/mnist_test_vectors.csv";
const string kMnistTestLabelsFileName = "MNIST_DATA/mnist_test_labels.csv";
const string kTrainPredictionsFileName = "trainPredictions";
const string kActualPredictionsFileName = "actualTestPredictions";

// -- Constants
const int kOutputClasses = 10;      // y = { 0, ..., 9 }
const int kInputImageWidth = 28;
const int kInputImageHeight = 28;
const float kEpsilonInit = 0.12;    // random initialization bounds [-kEpsilonInit, kEpsilonInit]
const float kEpsilon = 0.0000001;   // Adagrad / RMSProp prevents division by zero

// - Hyperparameters
const int kEpochs = 15;              // number of epochs (nr of iterations over the entire dataset)
const int kBatchSize = 200;         // size of one batch - the weights (thetas) are updated after one batch of inputs
const float kLearningRate = 0.15;
const float kRho = 0.9;           // RMSProp parameter - how much of the previous cache affects the current one

// notation from Coursera ML by Stanford (Andrew Ng)
// 3 layers, S represents number of neurons / units in each layer
const int kS1 = (kInputImageWidth * kInputImageHeight);
const int kS2 = 600;
const int kS3 = kOutputClasses;
// layer sizes with bias unit, we don't need the kS3 to have a bias...
const int kS1WithBias = kS1 + 1;
const int kS2WithBias = kS2 + 1;

// weights for each layer
typedef vector<vector<float> > FMatrix2D;
FMatrix2D theta_1(kS2, vector<float>(kS1WithBias));
FMatrix2D theta_2(kS3, vector<float>(kS2WithBias));

typedef vector<vector<int> > IMatrix2D;
IMatrix2D input_vector;                   // vector which contains input vectors, aka the entire dataset
vector<int> label_vector;                 // correct label represented as a vector -> (0, 0, 1, 0, ..., 0)

vector<int> *current_input_vector;        // vector which contains kS1 values / all the inputs in current feedforward
int *expected_label_for_current_input_vector;   // correct label for current_input_vector

vector<float> input_layer_activation_unit(kS1WithBias);
vector<float> hidden_layer_activation_unit(kS2WithBias);
vector<float> output_layer_activation_unit(kS3);

vector<float> z_2(kS2);
vector<float> z_3(kS3);

FMatrix2D gradient_accumulator_hidden_layer(kS2, vector<float>(kS1WithBias));
FMatrix2D gradient_accumulator_output_layer(kS3, vector<float>(kS2WithBias));

FMatrix2D rmsprop_cache_1(kS2, vector<float>(kS1WithBias));
FMatrix2D rmsprop_cache_2(kS3, vector<float>(kS2WithBias));

float error;

// --- Neural Network Implementation

// Sigmoid, activation function
float LogisticSigmoid(float z) {
    return 1 / (1 + exp(-z));
}
// Derivative of the sigmoid function
float LogisticSigmoidPrime(float z) {
    float classic_sigmoid = LogisticSigmoid(z);
    return classic_sigmoid * (1 - classic_sigmoid);
}

// Random initialization of weights (theta)
void RandomlyInitialize(FMatrix2D &theta, int size, int size2) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size2; j++) {
            float random_value = ((float(rand()) / float(RAND_MAX)) * 2 * kEpsilonInit);
            theta[i][j] = -kEpsilonInit + random_value;
        }
    }
}

void Propagate(vector<int> input) {
    // clear these from previous iterations
    for (int i = 0; i < kS2; i++) { z_2[i] = 0; }
    for (int i = 0; i < kS3; i++) { z_3[i] = 0; }

    // First layer's activation unit vector is just the input vector
    for (int i = 0; i < kS1; i++) {
        input_layer_activation_unit[i+1] = input[i];
    }
    input_layer_activation_unit[0] = 1; // 1st layer bias

    // hidden layer
    for (int i = 0; i < kS2; i++) {
        for (int j = 0; j < kS1WithBias; j++) {
            z_2[i] += input_layer_activation_unit[j] * theta_1[i][j];
        }
        hidden_layer_activation_unit[i+1] = LogisticSigmoid(z_2[i]);
    }
    hidden_layer_activation_unit[0] = 1; // hidden layer bias

    // compute the output
    for (int i = 0; i < kS3; i++) {
        for (int j = 0; j < kS2WithBias; j++) {
            z_3[i] += hidden_layer_activation_unit[j] * theta_2[i][j];
        }
        output_layer_activation_unit[i] = LogisticSigmoid(z_3[i]);
    }
}

void Backpropagate() {
    vector<float> delta_hidden(kS2WithBias);
    vector<float> delta_output(kS3);
    vector<float> target_value(kS3);
    target_value[*expected_label_for_current_input_vector] = 1;

    for (int k = 0; k < kS3; k++) {
        delta_output[k] = LogisticSigmoidPrime(z_3[k]) * (target_value[k] - output_layer_activation_unit[k]);
    }

    for (int i = 0; i < kS2WithBias; i++) {
        for (int k = 0; k < kS3; k++) {
            delta_hidden[i] += delta_output[k] * theta_2[k][i];
        }
        delta_hidden[i] *= LogisticSigmoidPrime(z_2[i]);
    }

    for (int i = 0; i < kS2; i++) {
        for (int j = 0; j < kS3; j++) {
            gradient_accumulator_output_layer[j][i] += delta_output[j] * hidden_layer_activation_unit[i];
        }
    }

    for (int i = 0; i < kS1; i++) {
        for (int j = 0; j < kS2; j++) {
            gradient_accumulator_hidden_layer[j][i] += delta_hidden[j] * input_layer_activation_unit[i];
        }
    }

    // accumulates the MSE
    float sum = 0;
    for (int i = 0; i < kS3; i++) {
        sum += pow(target_value[i]-output_layer_activation_unit[i], 2)/kS3;
    }
    error += sum;
}

void UpdateWeights() {
    for (int i = 0; i < kS2WithBias; i++) {
        for (int j = 0; j < kS3; j++) {
            rmsprop_cache_2[j][i] = kRho * rmsprop_cache_2[j][i] + (1 - kRho) * pow(gradient_accumulator_output_layer[j][i], 2);
            theta_2[j][i] += kLearningRate * (gradient_accumulator_output_layer[j][i] / kBatchSize) / (sqrt(rmsprop_cache_2[j][i])+kEpsilon);
            gradient_accumulator_output_layer[j][i] = 0; // make sure to reset the accumulator of the batch gradient
        }
    }

    for (int i = 0; i < kS1WithBias; i++) {
        for (int j = 0; j < kS2; j++) {
            rmsprop_cache_1[j][i] = kRho * rmsprop_cache_1[j][i] + (1 - kRho) * pow(gradient_accumulator_hidden_layer[j][i], 2);
            theta_1[j][i] += kLearningRate * (gradient_accumulator_hidden_layer[j][i] / kBatchSize) / (sqrt(rmsprop_cache_1[j][i])+kEpsilon);
            gradient_accumulator_hidden_layer[j][i] = 0;
        }
    }
}

// Creates a vector<int> from comma-delimited string of integers
vector<int> InputVectorFromString(string str) {
    vector<int> vector;
    stringstream ss(str);
    int i;
    while (ss >> i) {
        vector.push_back(i);
        if (ss.peek() == ',')
            ss.ignore();
    }
    return vector;
}

int LoadVectorsAndLabelsFromFile(string vector_filename, string label_filename) {
    ifstream vectors_file(vector_filename);
    ifstream labels_file(label_filename);
    string vector_line;
    string label_line;
    input_vector.clear();
    label_vector.clear();

    cout << "Loading vectors from file " << vector_filename << endl;
    int n = 0;
    while (getline(vectors_file, vector_line)) {
        // save the n'th input vector, 28x28 image
        vector<int> current_input_vec = InputVectorFromString(vector_line);
        input_vector.push_back(current_input_vec);
        // and the corresponding label...
        getline(labels_file, label_line);
        int current_label = stoi(label_line);
        label_vector.push_back(current_label);
        n++;
    }
    vectors_file.close();
    labels_file.close();
    cout << "Loaded " << n << "vectors from file."  << endl;
    return n;
}

void PrintInformation() {
    cout << "Mini-batch RMSProp of a 2-layer NN..." << endl;
    cout << "(" << kS1 << " input neurons) x (" << kS2 << " hidden neurons) x (" << kS3 << " output neurons)" << endl;
    cout << "Learning rate: " << kLearningRate << endl;
    cout << "Gamma: " << kRho << endl;
    cout << "Epochs: " << kEpochs << endl;
    cout << "Batch size: " << kBatchSize << endl;
}

void TrainNeuralNetwork() {
    PrintInformation();
    int trainingSetSize = LoadVectorsAndLabelsFromFile(kMnistTrainVectorsFileName, kMnistTrainLabelsFileName);

    // random init of thetas (weights)
    RandomlyInitialize(theta_1, kS2, kS1WithBias);
    RandomlyInitialize(theta_2, kS3, kS2WithBias);
    cout << "Started training after randomly initializing weights..." << endl;

    for (int e = 0; e < kEpochs; e++) {
        random_device rd("/dev/random");
        mt19937 eng(rd());
        uniform_int_distribution<> distr(0, trainingSetSize-1);
        for (int d = 0; d < trainingSetSize/kBatchSize; d++) {
            for (int k = 0; k < kBatchSize; k++) {
                // select a random input/label pair from our input files
                int random_line_number = distr(eng);
                current_input_vector = &input_vector[random_line_number];
                expected_label_for_current_input_vector = &label_vector[random_line_number];

                Propagate(*current_input_vector);
                Backpropagate();
            }
            //cout << "Loss is " << error/(2*kBatchSize) << " after " << (d+1) << " batches and " << e << " epochs." << endl;
            //error = 0;
            UpdateWeights();
        }
    }
    cout << "Finished training!" << endl;
}

// returns the index of the predicted class
int GetPredictedClass() {
    int max_index = 0;
    float max_value = -1;
    for (int i = 0; i < kS3; i++) {
        if (output_layer_activation_unit[i] > max_value) {
            max_value = output_layer_activation_unit[i];
            max_index = i;
        }
    }
    return max_index;
}

void TestNeuralNetwork() {
    cout << "Testing Training Data" << endl;
    ofstream train_predictions_file(kTrainPredictionsFileName);
    int trainingSetSize = LoadVectorsAndLabelsFromFile(kMnistTrainVectorsFileName, kMnistTrainLabelsFileName);

    for (int i = 0; i < trainingSetSize; i++) {
        Propagate(input_vector[i]);
        train_predictions_file << GetPredictedClass() << endl;
    }
    train_predictions_file.close();

    cout << "Testing Test Data" << endl;
    ofstream actual_predictions_file(kActualPredictionsFileName);
    trainingSetSize = LoadVectorsAndLabelsFromFile(kMnistTestVectorsFileName, kMnistTestLabelsFileName);

    for (int i = 0; i < trainingSetSize; i++) {
        Propagate(input_vector[i]);
        actual_predictions_file << GetPredictedClass() << endl;
    }
    actual_predictions_file.close();
    PrintInformation();
}
