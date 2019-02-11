#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:37:31 2019

@author: user
"""

import numpy as np
import csv

#labelFile = "../data/twitter/twitter_label.txt"
#dataFile = '../data/twitter/twitter.txt'
#numClass = 3

def BoW(labelFile,dataFile):

    NoOfDocs = 0
    file = open(dataFile, 'r')
    #string = file.readlines()
    #string = file.split()
    
    string = file.readline()
    for line in file:
        stringTemp = file.readline()
        string += stringTemp
        NoOfDocs += 1
    file.close()
    
    string = string.replace ('\n', '')
    words = string.split(' ')
    
    uniqueWords = set(words)
    
    
    No = []
    for i in uniqueWords:
        word = i + ' '
        No.append(string.count(word))
        #print(No)
        
    thresh = 200
    dictWords = []
    n = 0
    for i in uniqueWords:
        
        if(No[n]>thresh):
            dictWords.append(i)
        n = n+1
    
    
    
    file = open(dataFile, 'r')
    featureInt = np.zeros((NoOfDocs,len(dictWords)))
    rowIndex = 0
    
    for line in file:
        stringTemp = file.readline()
        colIndex = 0
        for string in dictWords:
            featureInt[rowIndex,colIndex] = stringTemp.count(string)
            colIndex += 1
        rowIndex += 1
    file.close()
    
    featureDim = len(dictWords)
    
#    label = []
    labelInt = []
    i = 0
    with open(labelFile) as file:
        for line in file:
            value = file.readline().split()
            
            labelInt.append(int(value[0]))

    return featureInt,labelInt,featureDim

#featureTrain = []
#featureTest = []
#labelTrain = []
#labelTest = []
#split = 10
#NoInSplit = len(featureInt)/split
#testIndex = split-8
#for i in range(len(featureInt)):
#    if i > testIndex*NoInSplit and i < (testIndex+1)*NoInSplit:
#        featureTest.append(featureInt[i,])
#        labelTest.append(labelInt[i])
#    else:
#        featureTrain.append(featureInt[i,])
#        labelTrain.append(labelInt[i])
#        
#trainNo = len(featureTrain)
#testNo = len(featureTest)
#    
# #   numClass = 4
#mew = np.zeros((numClass, featureDim))
#variance = np.zeros((numClass, featureDim,featureDim))
#numInClass = np.zeros((numClass,1))
#priorProb = np.zeros((numClass,1))
#for i in range(trainNo):
#    numInClass[labelInt[i]] += 1
#    mew[labelTrain[i],] += featureTrain[i]
#    
#    
#for i in range(numClass):
#    priorProb[i] = numInClass[i]/len(featureTrain)
#    mew[i,] = mew[i,]/numInClass[i]
#    
#for i in range(trainNo):
#    variance[labelTrain[i],] += np.transpose(featureTrain[i]-mew[labelTrain[i],])*(featureTrain[i]-mew[labelTrain[i],])
#    
#for i in range(numClass):
#    variance[i,] = variance[i,]/(numInClass[i])
#    variance[i,] = variance[i,]+np.eye(featureDim)
#
#postProb = np.zeros((numClass))   
#inData = np.zeros(featureDim) 
#for i in range(testNo):
#    for j in range(numClass):
#        inData = featureTest[i]-mew[j]
#        var = variance[j]
#        #postProb[j] = (1/(np.sqrt(((2*np.pi)**featureDim)*np.linalg.det(var))))*np.exp(-0.5*np.linalg.pinv(var).dot(inData).dot(inData))*priorProb[j]
#        postProb[j] = np.exp(-0.5*inData.dot(np.linalg.pinv(var)).dot(inData))#*priorProb[j]
#    print(postProb)
#    print(labelTest[i],np.argmax(postProb))



#fileName = '../data/twitter/BoW.csv'
#with open(fileName, 'w') as f:
#    writer = csv.writer(f,delimiter=' ')
#    writer.writerows(featureInt)
#
#f.close()
#file = open(labelFile, 'r')
#for line in file:
#    labelNo = labelNo+1