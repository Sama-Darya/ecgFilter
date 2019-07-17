//
// Created by sama on 25/06/19.
//

#ifndef ECGFILTER_PARAMETERS_H
#define ECGFILTER_PARAMETERS_H

#define NINPUTS 3 //this is the x-y-z accelerations inputs
#define NLAYERS 3
#define N1 9
#define N2 3
#define N3 1 //this has to always be 1
#define LEARNINGRATE 0.1


/* High-pass filter for the accelerations:
 */
//#define doAccHP
#ifdef doAccHP
    #define HP_CUTOFF 0.1
#endif


/* Low-pass filter for the accelerations:
 */
//#define doAccLP
#ifdef doAccLP
#define LP_CUTOFF 20
#endif

/*Band-pass filter for the accelerations:
 */
//#define doAccBP
#ifdef doAccBP
    #define NbpFilters 5
    #define MINT 50
    #define MAXT 100
    #define DAMPINGCOEFF 0.51
#endif

/*
 * Bandpass filter for the ECG signal to cause delay
 */

//#define doECGBP
#ifdef doECGBP
    #define Tdelay 100
    #define Damping 0.51
#endif


/*
 * LMS filter specifications
 */
#define LMS_COEFF (int)(10000)
#define LEARNING_RATE 0.00001

/*
 * plot related definitions
 */
#define WINDOW_NAME1 "Learning's plots"

#endif //ECGFILTER_PARAMETERS_H



