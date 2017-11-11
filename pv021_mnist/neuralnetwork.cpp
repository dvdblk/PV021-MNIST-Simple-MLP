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

// --- Constants

// File names
const std::string kMnistTrainVectorsFileName = "MNIST_DATA/mnist_train_vectors.csv";
const std::string kMnistTrainLabelsFileName = "MNIST_DATA/mnist_train_labels.csv";
const std::string kMnistTestVectorsFileName = "MNIST_DATA/mnist_test_vectors.csv";
const std::string kMnistTestLabelsFileName = "MNIST_DATA/mnist_test_labels.csv";
const std::string kTrainPredictionsFileName = "trainPredictions";
const std::string kActualPredictionsFileName = "actualTestPredictions";

using namespace std;


// -- Constants
const int kOutputClasses = 10; // y = { 0, ..., 9 }
const int kInputImageWidth = 28;
const int kInputImageHeight = 28;

const float epsilon_init = 0.12;

// TWEAK LATER
const int epochs = 6000;
const int batch_size = 10;
const float learning_rate = 0.1;

const float momentum = 0.8;

// 3 layers, s represents number of neurons / units in each layer
const int kS1 = (kInputImageWidth * kInputImageHeight);
const int kS2 = 700; // tweak later
const int kS3 = kOutputClasses;

const int kS1WithBias = kS1 + 1;
const int kS2WithBias = kS2 + 1;

// weights for each (except the output) layer
typedef vector<vector<float> > FMatrix2D;
FMatrix2D theta_1(kS2, vector<float>(kS1WithBias));
FMatrix2D theta_2(kS3, vector<float>(kS2WithBias));

typedef vector<vector<int> > IMatrix2D;
IMatrix2D input_vector;
vector<int> label_vector;

// reuse these for training
vector<int> *current_input_vector;
int *expected_label_for_current_input_vector;

vector<float> a_1(kS1WithBias);
vector<float> a_2(kS2WithBias);
vector<float> a_3(kS3); // h of Theta

vector<float> z_2(kS2);
vector<float> z_3(kS3);

vector<float> delta_hidden(kS2WithBias);
vector<float> delta_output(kS3);


// --- Implementation

// Sigmoid, activation function
float LogisticSigmoid(float z) {
    return 1 / (1 + exp(-z));
}

float LogisticSigmoidPrime(float z) {
    float classic_sigmoid = LogisticSigmoid(z);
    return classic_sigmoid * (1 - classic_sigmoid);
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

// Random initialization of weights (theta)
void RandomlyInitialize(FMatrix2D &theta, int size, int size2) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size2; j++) {
            float random_value = ((float(rand()) / float(RAND_MAX)) * 2 * epsilon_init);
            theta[i][j] = -epsilon_init + random_value;
        }
    }
}

// notation from Coursera ML by Stanford (Andrew Ng)
void Propagate(vector<int> input) {
    for (int i = 0; i < kS2; i++) {
        z_2[i] = 0;
    }
    
    for (int i = 0; i < kS3; i++) {
        z_3[i] = 0;
    }
    
    // First layer's activation vector is just the input vector
    a_1[0] = 1; // 1st layer bias
    for (int i = 0; i < kS1; i++) {
        a_1[i+1] = input[i];
    }
    
    for (int i = 0; i < kS2; i++) {
        for (int j = 0; j < kS1WithBias; j++) {
            z_2[i] += a_1[j] * theta_1[i][j];
        }
        a_2[i+1] = LogisticSigmoid(z_2[i]);
    }
    a_2[0] = 1; // 2nd layer bias
    
    for (int i = 0; i < kS3; i++) {
        for (int j = 0; j < kS2WithBias; j++) {
            z_3[i] += a_2[j] * theta_2[i][j];
        }
        a_3[i] = LogisticSigmoid(z_3[i]);
    }
}

float error;
const float epsilon = 0.000001; // 0.0000001 = 91.1%
const float kGamma = 0.9;

FMatrix2D gradient_acc_hidden(kS2, vector<float>(kS1WithBias));
FMatrix2D gradient_acc_output(kS3, vector<float>(kS2WithBias));

void Backpropagate() {
    vector<float> target_value(kS3);
    target_value[*expected_label_for_current_input_vector] = 1;
    
    vector<float> current_delta_output(kS3);
    
    for (int k = 0; k < kS3; k++) {
        current_delta_output[k] = LogisticSigmoidPrime(z_3[k]) * (target_value[k] - a_3[k]);
        delta_output[k] = current_delta_output[k];
    }
    
    for (int i = 0; i < kS2WithBias; i++) {
        float sum;
        for (int k = 0; k < kS3; k++) {
            sum += current_delta_output[k] * theta_2[k][i];
        }
        delta_hidden[i] = LogisticSigmoidPrime(z_2[i]) * sum;
    }
    
    for (int i = 0; i < kS2; i++) {
        for (int j = 0; i < kS3; i++) {
            gradient_acc_output[j][i+1] += delta_output[j] * a_2[i+1];
        }
    }
    
    for (int i = 0; i < kS3; i++) {
        gradient_acc_output[i][0] += delta_output[i];
    }
    
    for (int i = 0; i < kS1; i++) {
        for (int j = 0; i < kS2; i++) {
            gradient_acc_hidden[j][i+1] += delta_hidden[j] * a_1[i+1];
        }
    }
    
    for (int i = 0; i < kS2; i++) {
        gradient_acc_hidden[i][0] += delta_hidden[i];
    }
    
    float sum = 0;
    for (int i = 0; i < kS3; i++) {
        sum += pow(target_value[i]-a_3[i], 2)/kS3;
    }
    error += sum;
}

void UpdateWeights() {
    for (int i = 0; i < kS2WithBias; i++) {
        for (int j = 0; j < kS3; j++) {
            theta_2[j][i] += learning_rate * gradient_acc_output[j][i] / batch_size;
            gradient_acc_output[j][i] = 0;
            //theta_2[j][i+1] = learning_rate * delta_3[j] * a_2[i+1] + momentum * theta_2[j][i+1];
        }
    }
    
    for (int i = 0; i < kS1WithBias; i++) {
        for (int j = 0; j < kS2; j++) {
            theta_1[j][i] += learning_rate * gradient_acc_hidden[j][i] / batch_size;
            gradient_acc_hidden[j][i] = 0;
            //theta_1[j][i+1] += learning_rate * delta_2[j] * a_1[i+1];
            //theta_1[j][i+1] = learning_rate * delta_2[j] * a_1[i+1] + momentum * theta_1[j][i+1];
        }
    }
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
    cout << "Loaded vectors from file. Size: " << n << endl;
    return n;
}

void TrainNeuralNetwork() {
    int trainingSetSize = LoadVectorsAndLabelsFromFile(kMnistTrainVectorsFileName, kMnistTrainLabelsFileName);
    cout << "Starting mini-batch GD of a 3-layer NN..." << endl;
    cout << "(" << kS1 << " input neurons) x (" << kS2 << " hidden neurons) x (" << kS3 << " output neurons)" << endl;
    cout << "Learning rate: " << learning_rate << endl;
    cout << "Epochs: " << epochs << endl;
    cout << "Batch size: " << batch_size << endl;
    // random init of thetas (weights)
    RandomlyInitialize(theta_1, kS2, kS1WithBias);
    RandomlyInitialize(theta_2, kS3, kS2WithBias);
    cout << "Started training after randomly initializing weights..." << endl;
    
    random_device rd;
    mt19937 eng(rd());
    uniform_int_distribution<> distr(0, trainingSetSize-1);
    
    for (int e = 0; e < epochs; e++) {
        for (int k = 0; k < batch_size; k++) {
            // select a random input/label pair from our input files
            int random_line_number = distr(eng);
            current_input_vector = &input_vector[random_line_number];
            expected_label_for_current_input_vector = &label_vector[random_line_number];
            
            Propagate(*current_input_vector);
            Backpropagate();
        }
        cout << "Error is " << error/batch_size << " after " << e << " epochs." << endl;
        error = 0;
        UpdateWeights();
    }
    cout << "Finished training!" << endl;
}

int GetPredictedClass() {
    int max_index = 0;
    float max_value = -1;
    for (int i = 0; i < kS3; i++) {
        if (a_3[i] > max_value) {
            max_value = a_3[i];
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
}
