#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <boost/circular_buffer.hpp>

#include "clbp/Neuron.h"
#include "clbp/Layer.h"
#include "clbp/Net.h"

#include "Iir.h"

#include "parameters.h"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace std;

int nNeurons[NLAYERS]={N1,N2,N3};
int* nNeuronsp=nNeurons;
Net* net = new Net(NLAYERS, nNeuronsp, NINPUTS);
double* inputs = NULL;


// define gains:
double errorGain = 1;
double outputGain = 1;
double signalGain = 1;
double xAccGain = 1;
double yAccGain = 1;
double zAccGain = 1;


//creat circular buffers for plotting
boost::circular_buffer<double> xAccBuffer(1000);
boost::circular_buffer<double> yAccBuffer(1000);
boost::circular_buffer<double> zAccBuffer(1000);
boost::circular_buffer<double> errorBuffer(1000);
boost::circular_buffer<double> outputBuffer(1000);
boost::circular_buffer<double> signalBuffer(1000);
boost::circular_buffer<double> signalrawBuffer(1000);


int main(int argc, const char *argv[]) {
    cout << "ECG filtering!" << endl;

    //initialising a 50Hz removal filter
    const float fs = 1000;
    const float mains = 50;
    Iir::RBJ::IIRNotch iirnotch2;
    iirnotch2.setup(fs,mains);
    Iir::RBJ::IIRNotch iirnotch3;
    iirnotch3.setup(fs,mains);

    const float ecg_max_f = 100;
    Iir::Butterworth::LowPass<4> lp2;
    lp2.setup(fs,ecg_max_f);
    Iir::Butterworth::LowPass<4> lp3;
    lp3.setup(fs,ecg_max_f);



    cv::Mat frame = cv::Mat(cv::Size(1000, 610), CV_8UC3);
    // Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
    cvui::init(WINDOW_NAME, 20);


    //initialise the network
    net->initWeights(Neuron::W_RANDOM, Neuron::B_NONE);
    net->setLearningRate(LEARNINGRATE);

    //get the data from .tsv files: chest_strap_V2_V1   cables_Einth_II   cables_Einth_III   acc_x   acc_y   acc_z
    inputs = new double[NINPUTS];

    double control, signal2, signal3, signal2raw, signal3raw, signal2intermediate, xAcc, yAcc, zAcc;
    ifstream infile;
    infile.open("sub00walk.tsv");//open the text file
    if (!infile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }

    fstream errorlog;
    errorlog.open("errorECG2.tsv", fstream::out);

    fstream outputlog;
    outputlog.open("outputECG2.tsv", fstream::out);

    fstream signallog;
    signallog.open("signalECG2.tsv", fstream::out);

    fstream controllog;
    controllog.open("controlECG2.tsv", fstream::out);


    int i=0;
    while (!infile.eof())
    {
        //To make three arrays for each column (a for 1st column, b for 2nd....)
        infile >> control >> signal2raw >> signal3raw >> xAcc >> yAcc >> zAcc;
        i++;
        cout << "iteration " << i << endl;

        //filter the signals with the 50Hz removal filter
        signal2intermediate = lp2.filter(signal2raw);
        signal2 = iirnotch2.filter(signal2intermediate);

        signal3 = iirnotch3.filter(signal3raw);

        //set the acceleration as the inputs to the network
        inputs[0] = xAcc * xAccGain;
        inputs[1] = yAcc * yAccGain;
        inputs[2] = zAcc * zAccGain;

        //propagate the inputs
        net->setInputs(inputs);
        net->propInputs();
        //get the network's output
        double outPut = net->getOutput(0) * outputGain;
        double signal = signal2 * signalGain;
        //workout the error
        double leadError = (signal - outPut) * errorGain;
        //propagate the error
        net->setError(leadError);
        net->propError();
        //do learning on the weights
        net->updateWeights();

        cout << "error: " << leadError << " output: " << outPut << "" << endl;

        //save the data in files
        controllog << control << endl;
        outputlog << outPut << endl;
        signallog << signal << endl;
        errorlog << leadError << endl;

        //put the data in their buffers
        errorBuffer.push_back(leadError);
        outputBuffer.push_back(outPut);
        signalBuffer.push_back(signal2);
        signalrawBuffer.push_back(signal2raw);
        xAccBuffer.push_back(xAcc);
        yAccBuffer.push_back(yAcc);
        zAccBuffer.push_back(zAcc);

        /*Print and Plot on screen
         */

        frame = cv::Scalar(100, 100, 100);

        int graphW = 750;
        int graphH = 100;
        int graphOffset = 10;

        int barL = 200;
        int barOffset = graphW + 2 * graphOffset;
        int barCenter = graphH / 2;
        int textCenter = barCenter - graphOffset;

        std::vector<double> xAccPlot(xAccBuffer.begin(), xAccBuffer.end());
        std::vector<double> yAccPlot(yAccBuffer.begin(), yAccBuffer.end());
        std::vector<double> zAccPlot(zAccBuffer.begin(), zAccBuffer.end());

        cvui::sparkline(frame, xAccPlot, graphOffset, graphH * 0, graphW, graphH, 0xffffff);
        cvui::sparkline(frame, yAccPlot, graphOffset, graphH * 1, graphW, graphH, 0xffffff);
        cvui::sparkline(frame, zAccPlot, graphOffset, graphH * 2, graphW, graphH, 0xffffff);

        cvui::text(frame, barOffset, graphH * 0 + textCenter, "x_acceleration gain");
        cvui::trackbar(frame, barOffset, graphH * 0 + barCenter, barL, &xAccGain, (double)1., (double)10.);

        cvui::text(frame, barOffset, graphH * 1 + textCenter, "y_acceleration gain");
        cvui::trackbar(frame, barOffset, graphH * 1 + barCenter, barL, &yAccGain, (double)1., (double)10.);

        cvui::text(frame, barOffset, graphH * 2 + textCenter, "z_acceleration gain");
        cvui::trackbar(frame, barOffset, graphH * 2 + barCenter, barL, &zAccGain, (double)1., (double)10.);

        std::vector<double> signalPlot(signalBuffer.begin(), signalBuffer.end());
        std::vector<double> signalrawPlot(signalrawBuffer.begin(), signalrawBuffer.end());
        std::vector<double> outputPlot(outputBuffer.begin(), outputBuffer.end());
        std::vector<double> errorPlot(errorBuffer.begin(), errorBuffer.end());


        cvui::sparkline(frame, signalPlot, graphOffset, graphH * 3, graphW, graphH, 0x000000);
        cvui::sparkline(frame, outputPlot, graphOffset, graphH * 4, graphW, graphH, 0x000000);
        cvui::sparkline(frame, errorPlot , graphOffset, graphH * 5, graphW, graphH, 0x000000);

        cvui::text(frame, barOffset, graphH * 3 + textCenter, "signal gain");
        cvui::trackbar(frame, barOffset, graphH * 3 + barCenter, barL, &signalGain, (double)1., (double)10.);

        cvui::text(frame, barOffset, graphH * 4 + textCenter, "output gain");
        cvui::trackbar(frame, barOffset, graphH * 4 + barCenter, barL, &outputGain, (double)1., (double)10.);

        cvui::text(frame, barOffset, graphH * 5 + textCenter, "error gain");
        cvui::trackbar(frame, barOffset, graphH * 5 + barCenter, barL, &errorGain, (double)1., (double)10.);


        cvui::update();        // This function must be called *AFTER* all UI
        // components. It does all the behind the scenes magic to handle mouse clicks, etc.

        // Show everything on the screen
        cv::imshow(WINDOW_NAME, frame);

    }
    infile.close();
    cout << "The program has reahced the end of the input file" << endl;
}