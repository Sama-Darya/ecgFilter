#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:43:52 2019

@author: sama
"""


import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd


plt.close("all")

def rmsValue(arr, n): 
    square = 0
    mean = 0.0
    root = 0.0
      
    #Calculate square 
    for i in range(0,n): 
        square += (arr[i]**2) 
      
    #Calculate Mean  
    mean = (square / (float)(n)) 
      
    #Calculate Root 
    root = math.sqrt(mean) 
      
    return mean

dataECG=np.loadtxt('./cmake-build-debug/sub00walk.tsv');
dataError=np.loadtxt('./cmake-build-debug/errorECG2.tsv');
dataOutput=np.loadtxt('./cmake-build-debug/outputECG2.tsv');
dataSignal=np.loadtxt('./cmake-build-debug/signalECG2.tsv');
dataControl=np.loadtxt('./cmake-build-debug/controlECG2.tsv');

control=dataECG[:,0]
signal2=dataECG[:,1]
signal3=dataECG[:,2]
xAcc=dataECG[:,3]
yAcc=dataECG[:,4]
zAcc=dataECG[:,5]
    
power3 = rmsValue(signal3, len(signal3))

plt.figure()
plt.subplot(311)
plt.plot(xAcc[4000::])
plt.subplot(312)
plt.plot(yAcc[4000::])
plt.subplot(313)
plt.plot(zAcc[4000::])

plt.figure()
plt.subplot(411)
plt.plot(dataSignal[4000::])
plt.subplot(412)
plt.plot(dataOutput[4000::])
plt.subplot(413)
plt.plot(dataError[4000::])
plt.subplot(414)
plt.plot(dataControl[4000::])

diff = dataSignal[4000::] - dataError[4000::]

plt.figure()
plt.plot(diff*100000)
