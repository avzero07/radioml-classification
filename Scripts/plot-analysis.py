# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:13:51 2020

@author: akshay
"""

import numpy as np
import matplotlib.pyplot as plt


cnn2 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-2/accResSNR.npy')
cnn3 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-3/accResSNR.npy')
cnn4 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-4/accResSNR.npy')
cnn5 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-5/accResSNR.npy')
cnn6 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-6/accResSNR.npy')
cnn7 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-7/accResSNR.npy')
cnn8 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-8/accResSNR.npy')
cnn9 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-9/accResSNR.npy')
cnn10 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/CNN-10/accResSNR.npy')

plt.figure(figsize=(8, 6))
plt.title('Accuracy at Different SNR')
plt.plot(np.arange(-20,20,2), cnn2.T, label='CNN-2')
plt.plot(np.arange(-20,20,2), cnn3.T, label='CNN-3')
plt.plot(np.arange(-20,20,2), cnn4.T, label='CNN-4')
plt.plot(np.arange(-20,20,2), cnn5.T, label='CNN-5')
#plt.plot(np.arange(-20,20,2), cnn6.T, label='CNN-6')
#plt.plot(np.arange(-20,20,2), cnn7.T, label='CNN-7')
#plt.plot(np.arange(-20,20,2), cnn8.T, label='CNN-8')
#plt.plot(np.arange(-20,20,2), cnn9.T, label='CNN-9')
#plt.plot(np.arange(-20,20,2), cnn10.T, label='CNN-10')
plt.xlabel('SNR')
plt.xticks(np.arange(-20,20,2))
plt.ylabel('Class Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()


ann1 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/ANN-1/accResSNR.npy')
ann2 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/ANN-2/accResSNR.npy')
ann3 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/ANN-3/accResSNR.npy')
ann4 = np.load('C:/Users/aksha/Documents/Jupyter Notebooks/radioml-classification/Notebooks/Data/ANN-4/accResSNR.npy')

plt.figure(figsize=(8, 6))
plt.title('Accuracy at Different SNR')
plt.plot(np.arange(-20,20,2), ann1.T, label='ANN-1')
plt.plot(np.arange(-20,20,2), ann2.T, label='ANN-2')
plt.plot(np.arange(-20,20,2), ann3.T, label='ANN-3')
plt.plot(np.arange(-20,20,2), ann4.T, label='ANN-4')
#plt.plot(np.arange(-20,20,2), cnn5.T, label='CNN-5')
#plt.plot(np.arange(-20,20,2), cnn6.T, label='CNN-6')
#plt.plot(np.arange(-20,20,2), cnn7.T, label='CNN-7')
#plt.plot(np.arange(-20,20,2), cnn8.T, label='CNN-8')
#plt.plot(np.arange(-20,20,2), cnn9.T, label='CNN-9')
#plt.plot(np.arange(-20,20,2), cnn10.T, label='CNN-10')
plt.xlabel('SNR')
plt.xticks(np.arange(-20,20,2))
plt.ylabel('Class Accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()
