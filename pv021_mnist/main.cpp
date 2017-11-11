//
//  main.cpp
//  pv021_mnist_project
//
//  Created by David Bielik on 05/11/2017.
//  Copyright © 2017 David Bielik. All rights reserved.
//

#include <iostream>
#include <string>
#include "neuralnetwork.hpp"

using namespace std;

// Arguments
const string kHelpArgument = "--help";

void PrintHelp() {
    cout << "This binary file can be used for training and testing a neural network with the MNIST [1] Dataset in .csv format." << endl;
}

int main(int argc, const char * argv[]) {
    
    if (argc == 1) {
        TrainNeuralNetwork();
        TestNeuralNetwork();
    } else if (argc >= 2) {
        string operation = argv[1];
        
        if (operation == kHelpArgument) {
            PrintHelp();
        } else {
            cout << "Invalid argument." << endl;
        }
    }
    return 0;
}

