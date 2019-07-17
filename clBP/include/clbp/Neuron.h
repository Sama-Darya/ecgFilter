#pragma once

#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace std;

class Neuron {
public:
    Neuron(int _nInputs);
    ~Neuron();
    enum biasInitMethod { B_NONE = 0, B_RANDOM = 1 };
    enum weightInitMethod { W_ZEROS = 0, W_ONES = 1, W_RANDOM = 2 };

    void initWeights(weightInitMethod _wim, biasInitMethod _bim);
    void setLearningRate(double _learningRate);

    void setInput(int _index, double _value);
    void propInputs(int _index, double _value);
    void calcOutput();
    void updateWeights();
    double doActivation(double _sum);
    double doActivationPrime(double _input);
    void setError(double _nextSum);  // for the output layer only
    void propError(double _nextSum); // used for all layers except the output

    double getOutput();
    double getSumOutput();
    double getWeights(int _inputIndex);
    double getInitWeights(int _inputIndex);
    double getError();
    double getWeightChange();
    double getWeightDistance();
    int getnInputs();
    void saveWeights(string _fileName);

    inline void setWeight(int _index, double _weight) {
        assert((_index >= 0) && (_index < nInputs));
        weights[_index] = _weight;
    }
    void printNeuron();

private:
    int nInputs = 0;
    double *inputs = 0;
    double *weights = 0;
    double *initialWeights = 0;
    double bias = 0;
    double error = 0;
    double output = 0;
    double learningRate = 0;
    double sum = 0;
    double weightChange=0;
};