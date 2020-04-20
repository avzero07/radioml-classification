# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:31:27 2020

@author: akshay
"""
# Loading Data

pathToDataset = "C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Datasets/Standard/RML2016.10a_dict.pkl"

# Extract the pickle file
import pickle
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

Xd = pickle.load(open(pathToDataset,'rb'),encoding="bytes")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

#  into training and test sets of the form we can train/test on 
np.random.seed(777)

index = np.arange(0,220000)
random.shuffle(index)

trainIdx = index[0:110000]
testIdx = index[110000:220000]

trainX = X[trainIdx]
X_train = np.expand_dims(trainX, axis=-1) # Important

testX = X[testIdx]
X_test = np.expand_dims(testX, axis=-1) # Important

# One Hot Encode Labels
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(np.asarray(lbl)[:,0])
print(lb.classes_)
lbl_encoded=lb.transform(np.asarray(lbl)[:,0])
y_train=lbl_encoded[trainIdx]
y_test=lbl_encoded[testIdx]


# Function to Extract Test Data of Specific SNR
def extractTest(data,labels,labelsEncoded,testIndex,snr):
    testData = data[testIndex]
    labelArray = np.array([labels])
    testLabels = labelArray[:,testIdx,:]
    testLabelsEncoded = labelsEncoded[testIdx]
    
    idxOP = list()
    
    # Loop Through Label Array To Get Index of Specific SNR
    for i in range(0,testLabels.shape[1]):
        if testLabels[0,i,1].decode('ascii')==snr:
            idxOP.append(i)
    
    # Return Subset of Test Data and Corresponding Labels
    opTestData = np.expand_dims(testData[idxOP,:,:],axis=-1)
    opTestLabel = testLabelsEncoded[idxOP]
    
    return opTestData, opTestLabel
    
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
#test_Y_hat = model.predict(X_test, batch_size=1024)
    
# Confusion Matrix Function
def prepConfMat(testData,testLabel,predTestLabel,mods):
    modString = list()
    for i in range(0,len(mods)):
        modString.append(mods[i].decode('ascii'))
    
    conf = np.zeros([len(mods),len(mods)])
    confnorm = np.zeros([len(mods),len(mods)])
    for i in range(0,testData.shape[0]):
        j = list(testLabel[i,:]).index(1)
        k = int(np.argmax(predTestLabel[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(mods)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plot_confusion_matrix(confnorm, labels=modString)


# Other Utility Functions

# Get Label Array
lblArray = np.squeeze(np.array([lbl]))

# Count Number of Samples at Specific SNRs
def getSNRCount(labelArray,snr):
    bsnr = (str(snr)).encode()
    npLabelSpecific = labelArray[np.where(labelArray[:,1]==bsnr)]
    
    return npLabelSpecific.shape[0]

def getMODCount(labelArray,mod):
    npLabelSpecific = labelArray[np.where(labelArray[:,0]==mod)]
    
    return npLabelSpecific.shape[0]

def strataHist(labelArray,strata,strataList,dataName):
    cList = list()
    
    # strata = 1 | SNR
    if strata==1:
        x = np.arange(-20,20,2)
        y = np.arange(0,12000,1000)
        ptitle = 'SNR'
        for i in np.arange(0,len(strataList),1):
            cList.append(getSNRCount(labelArray,strataList[i]))
    # strata = 0 | Modulation
    if strata==0:
        modString = list()
        x = np.arange(0,11,1)
        y = np.arange(0,22000,2000)
        ptitle = 'Modulation'
        for i in np.arange(0,len(strataList),1):
            cList.append(getMODCount(labelArray,strataList[i]))
        
        for i in range(0,len(strataList)):
            modString.append(strataList[i].decode('ascii'))
        strataList = modString
        
    plt.figure(figsize=(8, 6))
    plt.bar(x,cList)
    plt.yticks(y)
    plt.ylabel('Number of Samples')
    plt.xticks(x,strataList)
    plt.xlabel('SNR')
    plt.title('Histogram of '+dataName+' Data Based on '+ptitle)
    
# Plot strataHist
strataHist(lblArray,1,snrs,'Input') # SNR
strataHist(lblArray,0,mods,'Input') # Mods

# strataHist of Train Data
trainLabelArray = lblArray[trainIdx,:]
strataHist(trainLabelArray,1,snrs,'Training') # SNR
strataHist(trainLabelArray,0,mods,'Training') # Mods

testLabelArray = lblArray[testIdx,:]
strataHist(testLabelArray,1,snrs,'Testing') # SNR
strataHist(testLabelArray,0,mods,'Testing') # Mods


# Create Validation Data Set
indexVal = np.arange(0,110000)
random.shuffle(indexVal)

realTrainIdx = indexVal[0:99000] 
valIdx = indexVal[99000:110000]

# Training Data
realTrainX = trainX[realTrainIdx]
realTrainY = y_train[realTrainIdx]

# Validation Data
validX = trainX[valIdx]
validY = y_train[valIdx]