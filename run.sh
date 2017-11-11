#!/bin/bash

# start time
SECONDS=0
START_TIME=`date`
echo $START_TIME

# remove previous binary
rm mnistNeuralNet

# compile
echo "Compiling mnistNeuralNet..."
g++ pv021_mnist/*.cpp -O3 -std=c++11 -o mnistNeuralNet
echo -e "Finished compiling.\n"

# train the model
echo "Current time is `date`"
./mnistNeuralNet

# check predictions
echo -e "Train predictions:"
java -jar fi-muni-pv021-automatic-evaluator-1.0-SNAPSHOT-jar-with-dependencies.jar trainPredictions MNIST_DATA/mnist_train_labels.csv 10
cat Results

echo -e "\n\nActual test predictions:"
java -jar fi-muni-pv021-automatic-evaluator-1.0-SNAPSHOT-jar-with-dependencies.jar actualTestPredictions MNIST_DATA/mnist_test_labels.csv 10
cat Results

# print total time
echo "Start time: $START_TIME"
echo "Current time: `date`"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
