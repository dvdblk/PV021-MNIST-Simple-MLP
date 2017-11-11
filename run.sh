#!/bin/bash
echo "Compiling mnistNeuralNet..."
g++ pv021_mnist/*.cpp -o mnistNeuralNet
./mnistNeuralNet
