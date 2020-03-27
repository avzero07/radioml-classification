# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:43:43 2020

@author: akshay
"""

# Loading Data

pathToDataset = "C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Datasets/Standard/RML2016.10a_dict.pkl"

# Extract the pickle file
import pickle
import numpy as np
Xd = pickle.load(open(pathToDataset,'rb'),encoding="bytes")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Import Necessary Packages
import os
import random
#import tensorflow.keras.utils
#import tensorflow.keras.models as models
#from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten
#from tensorflow.keras.layers import GaussianNoise
#from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
#from tensorflow.keras.regularizers import *
#from tensorflow.keras.optimizers import *
import matplotlib.pyplot as plt
#import seaborn as sns
#import tensorflow.keras
import numpy as np
import scipy.stats as sp

np.random.seed(777)

index = np.arange(0,220000)
random.shuffle(index)

trainIdx = index[0:110000]
testIdx = index[110000:220000]

trainX = X[trainIdx]
#X_train = np.expand_dims(trainX, axis=-1) # Important

testX = X[testIdx]
#X_test = np.expand_dims(testX, axis=-1) # Important

# Feature Extraction Methods

test = np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]]) # Test Data
testAng = np.array([[[1,2],[1,2]],[[5,6],[5,6]],[[9,10],[9,10]]]) # Test Data Angular
testNeg = np.array([[[1,2],[-1,-20]],[[5,6],[-5,-60]],[[9,10],[-9,-100]]])

# Normalize By Mean
def normalizeMean(data):
    meanData = np.mean(data,axis=2)
    
    dataReal = data[:,0,:]
    meanReal = (np.array([meanData[:,0]])).T
    normalReal = dataReal/meanReal
    
    dataImag = data[:,1,:]
    meanImag = (np.array([meanData[:,1]])).T
    normalImag = dataImag/meanImag
    
    return normalReal, normalImag

# Compute Instantaneous Amplitude (or Magnitude)
def instAmp(data):
    return np.linalg.norm(data,axis=1)

# Function for Raw Moment
def rawMoment(data,n):
    # Calculate the nth Raw Moment of The Data
    dataRaised = np.power(data,n)
    nthMoment = np.array([np.mean(dataRaised,axis=1)])
    
    return nthMoment.T

# Function for (x+y)th Order Moment
def highOrdMoment(data,x,y):
    complexData = data[:,0,:]+(1j*data[:,1,:]) # Data In Complex Form
    complexDataConj = np.conj(complexData) # Complex Conjugate
    
    finDat = np.power(complexData,x-y)*np.power(complexDataConj,y)
    
    finDatMean = np.array([np.mean(finDat,axis=1)]).T
    
    return finDatMean

# Feature 1: Ratio of Real and Complex Power
def betaRatio(data):
    sumOfSquares = np.sum(np.square(data),axis=2)
    beta = sumOfSquares[:,1]/sumOfSquares[:,0] # Q / I
    beta = np.array([beta])
    beta = beta.T
    return beta

# Feature 2: Standard Deviation of Direct Instantaneous Phase
def sigmaDP(data,n):
    # n is a flag for normalization
    # n = 1 : Normalize
    
    if(n==0):
        dataReal = data[:,0,:]
        dataImag = data[:,1,:]
    
    if(n==1):
        dataReal, dataImag = normalizeMean(data)
    
    # Perform I / R
    tanArg = dataImag/dataReal
    
    # ArcTan | Instantaneous Phase
    phase = np.arctan(tanArg)
    
    # Standard Deviation of Phase
    sigDP = np.array([np.std(phase,axis=1)])
    
    return phase, sigDP.T # Also Returns Phase for Use Later

# Feature 3: Standard Deviation of Absolute Value of Non Linear Component of Instantaneous Phase
def sigmaAP(data,n):
    # n is a flag for normalization
    # n = 1 : Normalize
    
    if(n==0):
        dataReal = data[:,0,:]
        dataImag = data[:,1,:]
    
    if(n==1):
        dataReal, dataImag = normalizeMean(data)
    
    # Perform I / R
    tanArg = dataImag/dataReal
    
    # ArcTan | Instantaneous Phase
    phase = np.arctan(tanArg)
    phase = np.abs(phase)
    
    # Standard Deviation of Phase
    sigAP = np.array([np.std(phase,axis=1)])
    
    return sigAP.T

# Feature 4: Standard Deviation of Absolute Value of Normalized Instantaneous Amplitude

# Absolute of Magnitude Basically
    
def sigmaAA(data):
    instAmplitude = instAmp(data)
    
    # Sepcial Normalization
    meanInstAmplitude = (np.array([np.mean(instAmplitude,axis=1)])).T
    normInstAmplitude = (instAmplitude / meanInstAmplitude) - 1 # This is acn
    
    # Find Absolute of acn
    normInstAmplitude = np.abs(normInstAmplitude)
    
    sigAA = np.array([np.std(normInstAmplitude,axis=1)])
    
    return sigAA.T

# Feature 5: Standard Deviation of Absolute Normalized Centered Instantaneous Frequency

# Less Accurate Estimate Using First Differences
def sigmaAF(phase):
    
    # Compute Instantaneous Frequency
    instFrequency = np.diff(phase) # Approximation of Derivative
    
    # Normalize and Center 
    meanInstFrequency = (np.array([np.mean(instFrequency,axis=1)])).T
    normInstFrequency = (instFrequency / meanInstFrequency) - 1
    
    # Find Absolute
    normInstFrequency = np.abs(normInstFrequency)
    
    # Find Standard Deviation
    sigAF = np.array([np.std(normInstFrequency,axis=1)])
    
    return sigAF.T


# Feature 6: Standard Deviation of Absolute Value of Instantaneous Amplitude (Normalized w.r.t Variance)
def sigmaV(data):
    instAmplitude = instAmp(data)
    
    # Normalize w.r.t Variance
    varInstAmplitude = (np.array([np.var(instAmplitude,axis=1)])).T
    normInstAmplitude = np.sqrt(instAmplitude/varInstAmplitude) - 1
    
    # Find Absolute
    normInstAmplitude = np.abs(normInstAmplitude)
    
    # Find Standard Deviation
    sigV = np.array([np.std(normInstAmplitude,axis=1)])
    
    return sigV.T

# Feature 7: Mixed Order Moments [M(4,2) / M(2,1)]
def genV20(data):
    instAmplitude = instAmp(data)
    
    moment4 = rawMoment(instAmplitude,4)
    moment2 = rawMoment(instAmplitude,2) # Typo in Paper, Squared?
    
    return moment4 / moment2

# Feature 8: Mean of Signal Magnitude
def meanMag(data):
    instAmplitude = instAmp(data)
    meanMagnitude = np.array([np.mean(instAmplitude,axis=1)])
    
    return meanMagnitude.T

# Feature 9: Normalized Square Root of Sum of Amplitudes
def normRootSumAmp(data):
    instAmplitude = instAmp(data)
    sumAmpl = np.array([np.sum(instAmplitude,axis=1)]).T
    normRoot = np.sqrt(sumAmpl)/instAmplitude.shape[1]
    
    return normRoot

# Feature 10: Max PSD of Normalized Centered Amplitude
def maxPSD(data):
    instAmplitude = instAmp(data)
    # Compute DFT
    ampDFT = np.fft.fft(instAmplitude,axis=1)
    # Compute Magnitude Spectrum
    magAmpDFT = np.abs(ampDFT)
    # Power Spectrum
    powDFT = np.square(magAmpDFT)
    # Max Power
    maxPow = np.array([np.max(powDFT,axis=1)]).T
    
    return maxPow / maxPow.shape[0]

# Feature 11: Cumulant C20
def getC20(data):
    m20 = highOrdMoment(data,2,0)
    return np.abs(m20)

# Feature 12: Cumulant C21
def getC21(data):
    m21 = highOrdMoment(data,2,1)
    return np.abs(m21)

# Feature 13: Cumulant C40
def getC40(data):
    m40 = highOrdMoment(data,4,0)
    m20 = highOrdMoment(data,2,0)
    c40 = m40 - (3*np.square(m20))
    
    return np.abs(c40)
    
# Feature 14: Cumulant C41
def getC41(data):
    m41 = highOrdMoment(data,4,1)
    m21 = highOrdMoment(data,2,1)
    m20 = highOrdMoment(data,2,0)
    
    c41 = m41 - (3*m20*m21)
    
    return np.abs(c41)
    
# Feature 15: Cumulant C42
def getC42(data):
    m42 = highOrdMoment(data,4,2)
    m21 = highOrdMoment(data,4,2)
    m20 = highOrdMoment(data,2,0)
    
    c42 = m42 - np.square(m20) - (2*np.square(m21))
    
    return np.abs(c42)

# Feature 16: Cumulant C63
def getC63(data):
    m63 = highOrdMoment(data,6,3)
    m20 = highOrdMoment(data,2,0)
    m21 = highOrdMoment(data,2,1)
    m22 = highOrdMoment(data,2,2)
    m40 = highOrdMoment(data,4,0)
    m41 = highOrdMoment(data,4,1)
    m42 = highOrdMoment(data,4,2)
    
    t1 = m63 - (9*m21*m42) + (12*np.power(m21,3))
    #t2 = (-3*m20*m42)-(3*m22*m41)
    #t3 = 18*m20*m21*m22
    
    t2 = (-6*m20*m40) + (18*np.square(m20)*m21) 
    
    #c63 = t1+t2+t3
    c63 = t1+t2
    
    return np.abs(c63)

# Feature 17: Cumulant C80
def getC80(data):
    m80 = highOrdMoment(data,8,0)
    m60 = highOrdMoment(data,6,0)
    m40 = highOrdMoment(data,4,0)
    m20 = highOrdMoment(data,2,0)
    
    t1 = m80 - (35*np.square(m40))
    t2 = (-28*m60*m20) + (420*m40)
    t3 = (-630*np.power(m20,4))
    
    c80 = t1+t2+t3
    
    return np.abs(c80)

# Feature 18: Kurtosis
def getKurtosis(data):
    complexData = data[:,0,:]+(1j*data[:,1,:]) # Data In Complex Form
    meanComplexData = np.array([np.mean(complexData,axis=1)]).T
    
    # Find fourth central moment
    fourthPower = np.power(complexData - meanComplexData,4)
    centralMoment4 = (np.array([np.sum(fourthPower,axis=1)]).T)/fourthPower.shape[1]
    
    # Variance
    var = np.array([np.var(complexData,axis=1)]).T
    
    kurt = np.abs(centralMoment4)/(np.square(var)) # var already abs
    
    return kurt

# Feature 19: Skewness
def getSkewness(data):
    complexData = data[:,0,:]+(1j*data[:,1,:]) # Data In Complex Form
    meanComplexData = np.array([np.mean(complexData,axis=1)]).T
    
    # Find third central moment
    thirdPower = np.power(complexData - meanComplexData,3)
    centralMoment3 = (np.array([np.sum(thirdPower,axis=1)]).T)/thirdPower.shape[1]
    
    # Standard Deviation
    std = np.array([np.std(complexData,axis=1)]).T
    
    skew = np.abs(centralMoment3)/(np.power(std,3)) # std already abs
    
    return skew

# Feature 20: Peak to RMS Ratio
def getPR(data):
    complexData = data[:,0,:]+(1j*data[:,1,:]) # Data In Complex Form
    absSquare = np.square(np.abs(complexData))
    absSquareMax = np.array([np.max(absSquare,axis=1)]).T
    
    # Calculate RMS (without Root)
    rms = np.array([np.mean(absSquare,axis=1)]).T # Consider Abs of Mean?
    
    # Calculate PR
    PR = absSquareMax/rms
    
    return PR

# Feature 21: Peak to Average Ratio
def getPA(data):
    complexData = data[:,0,:]+(1j*data[:,1,:]) # Data In Complex Form
    absData = np.abs(complexData)
    absMax = np.array([np.max(absData,axis=1)]).T
    
    # Calculate Mean
    meanData = np.array([np.mean(absData,axis=1)]).T # Consider Abs of Mean?
    
    # Calculate PA
    PA = absMax / meanData
    
    return PA

# Init Features

def createIPVector(data):
    beta = betaRatio(data)
    sigPhase, sigDp = sigmaDP(data,1)
    sigAp = sigmaAP(data,1)
    sigAa = sigmaAA(data)
    sigAF = sigmaAF(sigPhase)
    sigV = sigmaV(data)
    v20 = genV20(data)
    meanMagX = meanMag(data)
    X2 = normRootSumAmp(data)
    gammaMax = maxPSD(data)
    
    cumulantC20 = getC20(data)
    cumulantC21 = getC21(data)
    cumulantC40 = getC40(data)
    cumulantC41 = getC41(data)
    cumulantC42 = getC42(data)
    cumulantC63 = getC63(data)
    cumulantC80 = getC80(data)
    
    kurtosis = getKurtosis(data)
    skewness = getSkewness(data)
    
    pr = getPR(data)
    pa = getPA(data)

    # Concat
    xtrainIP = np.concatenate((beta,sigDp,sigAp,sigAa,sigAF,sigV,v20,meanMagX,X2,gammaMax,cumulantC20,cumulantC21,cumulantC40,cumulantC41,cumulantC42,cumulantC63,cumulantC80,kurtosis,skewness,pr,pa),axis=1)
    return xtrainIP
