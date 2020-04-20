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
random.seed(77)
np.random.seed(77)


index = np.arange(0,220000)
random.shuffle(index)

trainIdx = index[0:110000]
testIdx = index[110000:220000]

trainX = X[trainIdx]


# Create Validation Data Set
indexVal = np.arange(0,110000)
random.shuffle(indexVal)

realTrainIdx = indexVal[0:99000] 
valIdx = indexVal[99000:110000]

# Actual Training Data
realTrainX = trainX[realTrainIdx]
X_train = np.expand_dims(realTrainX, axis=-1) # Important

# Actual Validation Data
validX = trainX[valIdx]
X_valid = np.expand_dims(validX, axis=-1) # Important

# Actual Testing Data
testX = X[testIdx]
X_test = np.expand_dims(testX, axis=-1) # Important

# One Hot Encode Labels
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(np.asarray(lbl)[:,0])
print(lb.classes_)
lbl_encoded=lb.transform(np.asarray(lbl)[:,0])
ytrain=lbl_encoded[trainIdx]

y_train = ytrain[realTrainIdx]
y_valid = ytrain[valIdx]
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
        ptitle = 'Modulation Technique'
        for i in np.arange(0,len(strataList),1):
            cList.append(getMODCount(labelArray,strataList[i]))
        
        for i in range(0,len(strataList)):
            modString.append(strataList[i].decode('ascii'))
        strataList = modString
        
    plt.figure(figsize=(8, 6))
    plt.bar(x,cList,label=dataName+' Samples')
    plt.yticks(y)
    plt.ylabel('Number of Data Samples')
    plt.xticks(x,strataList)
    plt.xlabel(ptitle)
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Histogram of '+dataName+' Data Based on '+ptitle)
    
    return x,cList
    
# Plot strataHist
xInp, xcList = strataHist(lblArray,1,snrs,'Input') # SNR
xInpMod, xcListMod = strataHist(lblArray,0,mods,'Input') # Mods

# strataHist of Train Data
interLabelArray = lblArray[trainIdx,:]
trainLabelArray = interLabelArray[realTrainIdx]
xInpTrain, xcListTrain = strataHist(trainLabelArray,1,snrs,'Training') # SNR
xInpTrainMod, xcListTrainMod = strataHist(trainLabelArray,0,mods,'Training') # Mods

# strataHist of Valid Data
validLabelArray = interLabelArray[valIdx]
xInpValid, xcListValid = strataHist(validLabelArray,1,snrs,'Validation') # SNR
xInpValidMod, xcListValidMod =strataHist(validLabelArray,0,mods,'Validation') # Mods

# strataHist of Test Data
testLabelArray = lblArray[testIdx,:]
xInpTest, xcListTest = strataHist(testLabelArray,1,snrs,'Testing') # SNR
xInptTestMod, xcListTestMod = strataHist(testLabelArray,0,mods,'Testing') # Mods

# SNR Combo
plt.figure(figsize=(10, 4))
ax = plt.subplot(111)
ax.bar(np.arange(-20,20,2)-0.5,xcListValid,width=0.5,label='Validation Samples')
ax.bar(np.arange(-20,20,2),xcListTrain,width=0.5,label='Training Samples')
ax.bar(np.arange(-20,20,2)+0.5,xcListTest,width=0.5,label='Testing Samples')
plt.yticks(np.arange(0,12000,1000))
plt.ylabel('Number of Data Samples')
plt.ylim(0,9000)
plt.xticks(xInpTrain,snrs)
plt.xlabel('SNR')
plt.legend(loc='upper right')
plt.grid()
#plt.title('Histogram of Data Partitions Based on SNR')

modString = list()
for i in range(0,len(mods)):
    modString.append(mods[i].decode('ascii'))

# MOD Combo
plt.figure(figsize=(10, 4))
ax = plt.subplot(111)
ax.bar(np.arange(0,11,1)-0.2,xcListValidMod,width=0.2,label='Validation Samples')
ax.bar(np.arange(0,11,1),xcListTrainMod,width=0.2,label='Training Samples')
ax.bar(np.arange(0,11,1)+0.2,xcListTestMod,width=0.2,label='Testing Samples')
plt.yticks(np.arange(0,22000,2000))
plt.ylabel('Number of Data Samples')
plt.ylim(0,16000)
plt.xticks(np.arange(0,11,1),modString)
plt.xlabel('Modulation Technique')
plt.legend(loc='upper right')
plt.grid()
#plt.title('Histogram of Data Partitions Based on Modulation Technique')
# Create Validation Data Set
#indexVal = np.arange(0,110000)
#random.shuffle(indexVal)

#realTrainIdx = indexVal[0:99000] 
#valIdx = indexVal[99000:110000]

# Training Data
#realTrainX = trainX[realTrainIdx]
#realTrainY = y_train[realTrainIdx]

# Validation Data
#validX = trainX[valIdx]
#validY = y_train[valIdx]