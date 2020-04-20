# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 12:55:08 2020

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