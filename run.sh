#!/bin/bash
date
echo "Compiling mnistNeuralNet..."
g++ pv021_mnist/*.cpp -o mnistNeuralNet
echo "Finished."
./mnistNeuralNet
date
