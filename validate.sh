#!/bin/bash
echo -e "Train predictions:"
java -jar fi-muni-pv021-automatic-evaluator-1.0-SNAPSHOT-jar-with-dependencies.jar trainPredictions MNIST_DATA/mnist_train_labels.csv 10
cat Results
echo -e "\n\nActual test predictions:"
java -jar fi-muni-pv021-automatic-evaluator-1.0-SNAPSHOT-jar-with-dependencies.jar actualTestPredictions MNIST_DATA/mnist_test_labels.csv 10
cat Results
echo ""
