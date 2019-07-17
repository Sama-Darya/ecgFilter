#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <boost/circular_buffer.hpp>
#include "clbp/Neuron.h"
#include "clbp/Layer.h"
#include "clbp/Net.h"
#include <Iir.h>
#include <Fir1.h>
#include "parameters.h"
#include "bandpass.h"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace std;

#ifdef doAccBP
int totalNINPUTS = NINPUTS * NbpFilters;
Bandpass*** bandpass = NULL;
double minT = MINT;
double maxT = MAXT;
double dampingCoeff = DAMPINGCOEFF;
#else
int totalNINPUTS = NINPUTS;
#endif
Fir1** lmsFilter = NULL;

//initialise network


int nNeurons[NLAYERS]={N1,N2,N3};
int* nNeuronsp=nNeurons;
Net* net = new Net(NLAYERS, nNeuronsp, totalNINPUTS);



// define gains
double errorGain = 1;
double outputGain = 1;
double xAccGain = 1;
double yAccGain = 1;
double zAccGain = 1;

//creat circular buffers for plotting
boost::circular_buffer<double> xAccBuffer(1000);
boost::circular_buffer<double> xAccBufferDelayMin(1000);
boost::circular_buffer<double> xAccBufferDelayMax(1000);
boost::circular_buffer<double> yAccBuffer(1000);
boost::circular_buffer<double> yAccBufferDelayMin(1000);
boost::circular_buffer<double> yAccBufferDelayMax(1000);
boost::circular_buffer<double> zAccBuffer(1000);
boost::circular_buffer<double> zAccBufferDelayMin(1000);
boost::circular_buffer<double> zAccBufferDelayMax(1000);
boost::circular_buffer<double> errorBuffer(1000);
boost::circular_buffer<double> outputBuffer(1000);
boost::circular_buffer<double> signalBuffer(1000);
boost::circular_buffer<double> signalrawBuffer(1000);
boost::circular_buffer<double> corrLMSBuffer(1000);
boost::circular_buffer<double> errorLMSBuffer(1000);



int main(int argc, const char *argv[]) {

    fstream errorlog;
    errorlog.open("errorECG.tsv", fstream::out);
    fstream outputlog;
    outputlog.open("outputECG.tsv", fstream::out);
    fstream signallog;
    signallog.open("signalECG.tsv", fstream::out);
    fstream controllog;
    controllog.open("controlECG.tsv", fstream::out);

// initialise filters
    lmsFilter = new Fir1*[totalNINPUTS];
    for(int i=0;i<totalNINPUTS;i++){
        lmsFilter[i] = new Fir1(LMS_COEFF);
        lmsFilter[i]->setLearningRate(LEARNING_RATE);
    }
    double corrLMS = 0;

//initialising 50Hz and 100+Hz removal filters
    const float fs = 1000;
    const float mains = 50;
    Iir::RBJ::IIRNotch iirnotch;
    iirnotch.setup(fs,mains);
    const float ecg_max_f = 100;
    Iir::Butterworth::LowPass<4> lp;
    lp.setup(fs,ecg_max_f);

//make highpass and bandpass filters for xyz accelerations
#ifdef doAccHP
    Iir::Butterworth::HighPass<2> hpAcc[3];
    for (int i=0; i<NINPUTS; i++){
        hpAcc[i].setup(fs,HP_CUTOFF);
    }
#endif

//make highpass and bandpass filters for xyz accelerations
#ifdef doAccLP
    Iir::Butterworth::LowPass<2> lpAcc[3];
    for (int i=0; i<NINPUTS; i++){
        lpAcc[i].setup(fs,LP_CUTOFF);
    }
#endif

#ifdef doAccBP
    bandpass = new Bandpass**[NINPUTS];
    for(int i=0;i<NINPUTS;i++) {
        if (bandpass != NULL) {
            bandpass[i] = new Bandpass*[NbpFilters];
            double ffs = 1;
            double fmin = ffs/maxT;
            double fmax = ffs/minT;
            double df = (fmax-fmin)/((double)(NbpFilters-1));
            double f = fmin;

            for(int j=0;j<NbpFilters;j++) {
                bandpass[i][j] = new Bandpass();
                bandpass[i][j]->setParameters(f,dampingCoeff);
                f = f + df;
                for(int k=0;k<maxT;k++) {
                    double a = 0;
                    if (k==minT) {
                        a = 1;
                    }
                    double b = bandpass[i][j]->filter(a);
                    assert(b != NAN);
                    assert(b != INFINITY);
                }
                bandpass[i][j]->reset();
            }
        }
    }
#endif

#ifdef doECGBP
    Bandpass ecgBP;
    double fsEcg=1;
    double fEcg= fsEcg / Tdelay;
    ecgBP.setParameters(fEcg, Damping);
    for(int k=0;k<Tdelay+10;k++) {
        double a = 0;
        if (k==Tdelay-10) {
            a = 1;
        }
        double b = ecgBP.filter(a);
        assert(b != NAN);
        assert(b != INFINITY);
    }
    ecgBP.reset();
#endif



//initialise plots
    cv::Mat Learningframe = cv::Mat(cv::Size(1000, 610), CV_8UC3);
    cvui::init(WINDOW_NAME1, 20);

//initialise the network
    net->initWeights(Neuron::W_RANDOM, Neuron::B_NONE);
    net->setLearningRate(LEARNINGRATE);
    double  acc[NINPUTS];
    double  accFiltered[NINPUTS];
    double  inputs[NINPUTS];
    double inputsDelayed[totalNINPUTS];
    double control, signal2raw, signal3raw, signalintermediate, xAcc, yAcc, zAcc;

//open the data file
    ifstream infile;
    infile.open("sub00walk.tsv");
    if (!infile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }

    while (!infile.eof())
    {
//get the data from .tsv files: chest_strap_V2_V1   cables_Einth_II   cables_Einth_III   acc_x   acc_y   acc_z
        infile >> control >> signal2raw >> signal3raw >> xAcc >> yAcc >> zAcc;
            //cout << "iteration " << i ;

//decide what signal to use II or III
        double signalraw = signal2raw;

//filter the signals with the 50Hz removal filter
        signalintermediate = lp.filter(signalraw);
        double signaltemp = iirnotch.filter(signalintermediate);

//filer signal with a bandpass
#ifdef doECGBP
        double signal = ecgBP.filter(signaltemp);
#else
        double signal = signaltemp;
#endif

//high-pass filter the xyz accelerations and set them as the inputs to the network and LMS filter
        acc[0]= xAcc;// -30;
        acc[1]= yAcc;// +18;
        acc[2]= zAcc;// +3;

#ifdef doAccHP
        for (int i=0; i<NINPUTS; i++){
            accFiltered[i]=hpAcc[i].filter(acc[i]);
        }
#else
        for (int i=0; i<NINPUTS; i++){
            accFiltered[i]=acc[i];
        }
#endif
#ifdef doAccLP
        for (int i=0; i<NINPUTS; i++){
            inputs[i]=lpAcc[i].filter(accFiltered[i]);
        }
#else
        for (int i=0; i<NINPUTS; i++){
            inputs[i]=accFiltered[i];
        }
#endif

//filtering xyz with bandpass filters to generate a range of time-advanced predictive signals
#ifdef doAccBP
        int k=0;
        for (int i=0; i<NINPUTS; i++){
            for (int j=0; j<NbpFilters; j++){
                inputsDelayed[k]=bandpass[i][j]->filter(inputs[i]);
                k++;
                }
            }
        double* inputsDelayedPointer = &inputsDelayed[0];
#else
        double* inputsDelayedPointer = &inputs[0];
#endif

        //propagate the inputs
        net->setInputs(inputsDelayedPointer);
        net->propInputs();
        //get the network's output
        double outPut = net->getOutput(0) * outputGain;
        //workout the error
        double leadError = (signal - outPut) * errorGain;
        //propagate the error
        net->setError(leadError);
        net->propError();
        //do learning on the weights
        net->updateWeights();

        //LMS filter
        for (int i=0; i<totalNINPUTS; i++){
            corrLMS += lmsFilter[i]->filter(inputsDelayed[i]);
        }
        double errorLMS  = signal - corrLMS;
        for (int i=0; i<totalNINPUTS; i++){
            lmsFilter[i]->lms_update(errorLMS);
        }

        //cout << ": (error: " << leadError << " output: " << outPut << ")" << endl;

//save the data in files

        controllog << control << endl;
        signallog << signal << endl;
        outputlog << outPut << endl;
        errorlog << leadError << endl;


//put the data in their buffers
        errorBuffer.push_back(leadError);
        outputBuffer.push_back(outPut);
        signalBuffer.push_back(signal);
        signalrawBuffer.push_back(signal2raw);
        xAccBuffer.push_back(inputs[0]); // this is xAcc
        xAccBufferDelayMin.push_back(inputsDelayed[0]); // this is xAcc min delay
        xAccBufferDelayMax.push_back(inputsDelayed[4]); // this is xAcc max delay
        yAccBuffer.push_back(inputs[1]); // this is yAcc
        yAccBufferDelayMin.push_back(inputsDelayed[5]); // this is yAcc min delay
        yAccBufferDelayMax.push_back(inputsDelayed[9]); // this is yAcc max delay
        zAccBuffer.push_back(inputs[2]); // this is zAcc
        zAccBufferDelayMin.push_back(inputsDelayed[10]); // this is zAcc min delay
        zAccBufferDelayMax.push_back(inputsDelayed[14]); // this is zAcc max delay
        corrLMSBuffer.push_back(corrLMS);
        errorLMSBuffer.push_back(errorLMS);
        
        
//making vectors for plotting
        std::vector<double> xAccPlot(xAccBuffer.begin(), xAccBuffer.end());
        std::vector<double> xAccPlotDelayMin(xAccBufferDelayMin.begin(), xAccBufferDelayMin.end());
        std::vector<double> xAccPlotDelayMax(xAccBufferDelayMax.begin(), xAccBufferDelayMax.end());
//        xAccPlot[0]=0;
//        xAccPlotDelayMin[0]=0;
//        xAccPlotDelayMax[0]=0;

        std::vector<double> yAccPlot(yAccBuffer.begin(), yAccBuffer.end());
        std::vector<double> yAccPlotDelayMin(yAccBufferDelayMin.begin(), yAccBufferDelayMin.end());
        std::vector<double> yAccPlotDelayMax(yAccBufferDelayMax.begin(), yAccBufferDelayMax.end());
//        yAccPlot[0]=0;
//        yAccPlotDelayMin[0]=0;
//        yAccPlotDelayMax[0]=0;

        std::vector<double> zAccPlot(zAccBuffer.begin(), zAccBuffer.end());
        std::vector<double> zAccPlotDelayMin(zAccBufferDelayMin.begin(), zAccBufferDelayMin.end());
        std::vector<double> zAccPlotDelayMax(zAccBufferDelayMax.begin(), zAccBufferDelayMax.end());
//        zAccPlot[0]=0;
//        zAccPlotDelayMin[0]=0;
//        zAccPlotDelayMax[0]=0;

        std::vector<double> signalPlot(signalBuffer.begin(), signalBuffer.end());
        std::vector<double> signalrawPlot(signalrawBuffer.begin(), signalrawBuffer.end());
        std::vector<double> outputPlot(outputBuffer.begin(), outputBuffer.end());
        std::vector<double> errorPlot(errorBuffer.begin(), errorBuffer.end());
//        signalPlot[0]=0;
//        signalrawPlot[0]=0;
//        outputPlot[0]=0;
//        errorPlot[0]=0;

        std::vector<double> corrLMSPlot(corrLMSBuffer.begin(), corrLMSBuffer.end());
        std::vector<double> errorLMSPlot(errorLMSBuffer.begin(), errorLMSBuffer.end());
//        corrLMSPlot[0]=0;
//        errorLMSPlot[0]=0;

        
//plot parameters
        int graphW = 750;
        int graphH = 100;
        int graphOffset = 10;

        int barL = 200;
        int barOffset = graphW + 2 * graphOffset;
        int barCenter = graphH / 2;
        int textCenter = barCenter - graphOffset;


//plotting
        Learningframe = cv::Scalar(180, 180, 180);

        cvui::sparkline(Learningframe, xAccPlot, graphOffset, graphH * 0, graphW, graphH, 0xffffff);
        cvui::sparkline(Learningframe, xAccPlotDelayMin, graphOffset, graphH * 0, graphW, graphH, 0xffff99);
        cvui::sparkline(Learningframe, xAccPlotDelayMax, graphOffset, graphH * 0, graphW, graphH, 0xffff00);

        cvui::sparkline(Learningframe, yAccPlot, graphOffset, graphH * 1, graphW, graphH, 0xffffff);
        cvui::sparkline(Learningframe, yAccPlotDelayMin, graphOffset, graphH * 1, graphW, graphH, 0xffff99);
        cvui::sparkline(Learningframe, yAccPlotDelayMax, graphOffset, graphH * 1, graphW, graphH, 0xffff00);

        cvui::sparkline(Learningframe, zAccPlot, graphOffset, graphH * 2, graphW, graphH, 0xffffff);
        cvui::sparkline(Learningframe, zAccPlotDelayMin, graphOffset, graphH * 2, graphW, graphH, 0xffff99);
        cvui::sparkline(Learningframe, zAccPlotDelayMax, graphOffset, graphH * 2, graphW, graphH, 0xffff00);

        cvui::text(Learningframe, barOffset, graphH * 0 + textCenter, "x_acceleration gain");
        cvui::trackbar(Learningframe, barOffset, graphH * 0 + barCenter, barL, &xAccGain, (double)1., (double)10.);

        cvui::text(Learningframe, barOffset, graphH * 1 + textCenter, "y_acceleration gain");
        cvui::trackbar(Learningframe, barOffset, graphH * 1 + barCenter, barL, &yAccGain, (double)1., (double)10.);

        cvui::text(Learningframe, barOffset, graphH * 2 + textCenter, "z_acceleration gain");
        cvui::trackbar(Learningframe, barOffset, graphH * 2 + barCenter, barL, &zAccGain, (double)1., (double)10.);

        cvui::sparkline(Learningframe, signalrawPlot, graphOffset, graphH * 3, graphW, graphH, 0xccff66);
        cvui::sparkline(Learningframe, signalPlot, graphOffset, graphH * 3, graphW, graphH, 0x000000);

        cvui::sparkline(Learningframe, outputPlot, graphOffset, graphH * 4, graphW, graphH, 0x000000);
        cvui::sparkline(Learningframe, corrLMSPlot, graphOffset, graphH * 4, graphW, graphH, 0xccff00);

        cvui::sparkline(Learningframe, errorPlot , graphOffset, graphH * 5, graphW, graphH, 0x000000);
        cvui::sparkline(Learningframe, errorLMSPlot , graphOffset, graphH * 5, graphW, graphH, 0xccff00);

        cvui::text(Learningframe, barOffset, graphH * 4 + textCenter, "output gain");
        cvui::trackbar(Learningframe, barOffset, graphH * 4 + barCenter, barL, &outputGain, (double)1., (double)10.);

        cvui::text(Learningframe, barOffset, graphH * 5 + textCenter, "error gain");
        cvui::trackbar(Learningframe, barOffset, graphH * 5 + barCenter, barL, &errorGain, (double)1., (double)10.);

        cvui::update();
        cv::imshow(WINDOW_NAME1, Learningframe);

    }
    infile.close();
    cout << "The program has reahced the end of the input file" << endl;
}