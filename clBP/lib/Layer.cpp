#include "clbp/Layer.h"
#include "clbp/Neuron.h"

#include <fstream>

Layer::Layer(int _nNeurons, int _nInputs)
{
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new Neuron*[nNeurons];
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
        neurons[i]=new Neuron(nInputs);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs*/
}

Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    //delete[] inputs;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}

void Layer::setInputs(const double* _inputs){
    inputs=_inputs;
    for (int j=0; j<nInputs; j++){
        Neuron** neuronsp = neurons;//point to the 1st neuron
        /* sets a temporarily pointer to neuron-pointers
         * within the scope of this function. this is inside
         * the loop, so that it is set to the first neuron
         * everytime a new value is distributed to neurons */
        double input= *inputs; //take this input value
        for (int i=0; i<nNeurons; i++){
            (*neuronsp)->setInput(j,input);
            //set this input value for this neuron
            neuronsp++; //point to the next neuron
        }
        inputs++; //point to the next input value
    }
}

void Layer::propInputs(int _index, double _value){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->propInputs(_index, _value);
    }
}

void Layer::calcOutputs(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->calcOutput();
    }
}

void Layer::setError(double _leadError){
    /* this is only for the final layer */
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setError(_leadError);
    }
}

void Layer::propError(int _neuronIndex, double _nextSum){
    neurons[_neuronIndex]->propError(_nextSum);
}

double Layer::getError(int _neuronIndex){
    return (neurons[_neuronIndex]->getError());
}

double Layer::getSumOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getSumOutput());
}

double Layer::getWeights(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getWeights(_weightIndex));
}

double Layer::getInitWeight(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getInitWeights(_weightIndex));
}

double Layer::getWeightChange(){
    double weightChange = 0;
    for (int i=0; i<nNeurons; i++){
        weightChange += neurons[i]->getWeightChange();
    }
    //cout<< "Layer: WeightChange is: " << weightChange << endl;
    return (weightChange);
}


double Layer::getOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getOutput());
}


void Layer::initWeights(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initWeights(_wim, _bim);
    }
}

void Layer::setlearningRate(double _learningRate){
    learningRate=_learningRate;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(learningRate);
    }
}

void Layer::updateWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateWeights();
    }
}

int Layer::getnNeurons(){
    return (nNeurons);
}

int Layer::saveWeights(int _layerIndex, int _neuronCount){
    char l = '0';
    char n = '0';
    l += _layerIndex + 1;
    for (int i=0; i<nNeurons; i++){
        _neuronCount += 1;
        string name = "neuronWeight";
        n += 1;
        name += l;
        name += n;
        name += ".txt";
        neurons[i]->saveWeights(name);
    }
    return (_neuronCount);
}

void Layer::snapWeights(int _layerIndex){
    std::ofstream wfile;
    char l = '0';
    l += _layerIndex + 1;
    string name = "layerWeight";
    name += l;
    name += ".txt";
    wfile.open(name);
    for (int i=0; i<nNeurons; i++){
        for (int j=0; j<nInputs; j++){
            wfile << neurons[i]->getWeights(j) << " ";
        }
        wfile << "\n";
    }
    wfile.close();
}

Neuron* Layer::getNeuron(int _neuronIndex){
    assert(_neuronIndex < nNeurons);
    return (neurons[_neuronIndex]);
}

void Layer::printLayer(){
    cout<< "\t This layer has " << nNeurons << " Neurons" <<endl;
    cout<< "\t There are " << nInputs << " inputs to this layer" <<endl;
    for (int i=0; i<nNeurons; i++){
        cout<< "\t Neuron number " << i << ":" <<endl;
        neurons[i]->printNeuron();
    }

}

