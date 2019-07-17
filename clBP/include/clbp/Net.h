#pragma once

#include "Layer.h"

class Net {
public:
    Net(int _nLayers, int *_nNeurons, int _nInputs);
    ~Net();
    Layer *getLayer(int _layerIndex);
    void initWeights(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim);

    void setLearningRate(double _learningRate);
    void setInputs(const double *_inputs);
    void propInputs();
    void setError(double _leadError);
    void propError();
    void updateWeights();

    double getOutput(int _neuronIndex);
    double getSumOutput(int _neuronIndex);
    int getnLayers();
    int getnInputs();
    double getWeightDistance();
    double getWeights(int _layerIndex, int _neuronIndex, int _weightIndex);
    int getnNeurons();
    void saveWeights();
    void printNetwork();

private:
    int nLayers = 0;
    int nInputs = 0;
    int nOutputs = 0;
    const double *inputs = 0;
    Layer **layers = 0;
    double learningRate = 0;
    int nNeurons;
};