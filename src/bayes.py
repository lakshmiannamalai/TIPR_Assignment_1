# Implement Bayes Classifier here!


import csv
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import BagOfWords as BW
from sklearn.preprocessing import StandardScaler
import util as ut

    
classifier = 'bayes' #'naive'

def bayesClassifier(dataFile,labelFile,dataset):
    if (dataset == 'dolphins'):
        numClass = 4
    if (dataset == 'pubmed'):
        numClass = 3
    if (dataset == 'twitter'):
            numClass = 3
    if dataset == 'twitter':
        labelInt = []
        i = 0
        with open(labelFile) as file:
            for line in file:
                value = file.readline().split()
        
                labelInt.append(int(value[0]))   
        if dataFile.find('.txt') == -1:
            with open(dataFile) as csv_file:
                file_id = csv.reader(csv_file, delimiter=',')
                index = 0
                feature = []
                for line in file_id:
                    feature.append(line)
                    index += 1     
        
        
        
            
        
            rowIndex = 0
            Initialize = 1
            for row in feature:
                for i in row:
                    x = i.split(" ")
                    if Initialize == 1:
                        featureDim = len(x)
                        featureInt = np.zeros((len(feature),featureDim))
                        Initialize = 0
                    for colIndex in range(len(x)):
                        featureInt[rowIndex,colIndex] = x[colIndex]
                    
                rowIndex += 1
        else:    
            [featureInt,labelInt,featureDim] = BW.BoW(labelFile,dataFile)
        
    else:
        
        with open(labelFile) as csv_file:
            
            file_id = csv.reader(csv_file, delimiter=';')
            index = 0
            label = []
            for line in file_id:
                label.append(line)
                index += 1
            labelInt = [int(i[0]) for i in label]   
            
                
                
        
        with open(dataFile) as csv_file:
            file_id = csv.reader(csv_file, delimiter=',')
            index = 0
            feature = []
            for line in file_id:
                feature.append(line)
                index += 1     
        
        
        
            
        
        rowIndex = 0
        Initialize = 1
        for row in feature:
            for i in row:
                x = i.split(" ")
                if Initialize == 1:
                    featureDim = len(x)
                    featureInt = np.zeros((len(feature),featureDim))
                    Initialize = 0
                for colIndex in range(len(x)):
                    featureInt[rowIndex,colIndex] = x[colIndex]
                
            rowIndex += 1
        
    Accuracy = 0
    f1_macro = 0
    f1_micro = 0
    Accuracy_py = 0
    f1_macro_py = 0
    f1_micro_py = 0
    split = 10
    NoInSplit = len(featureInt)/split
    for cValid in range(1,split):
    
        featureTrain = []
        featureTest = []
        labelTrain = []
        labelTest = []
        
        testIndex = split-cValid
        for i in range(len(featureInt)):
            if i > testIndex*NoInSplit and i < (testIndex+1)*NoInSplit:
                featureTest.append(featureInt[i,])
                labelTest.append(labelInt[i])
            else:
                featureTrain.append(featureInt[i,])
                labelTrain.append(labelInt[i])
                        
        
        
        scaler = StandardScaler()
        scaler.fit(featureTrain)
        featureTrain = scaler.transform(featureTrain)
        featureTest = scaler.transform(featureTest)       
        
            
        trainNo = len(featureTrain)
        testNo = len(featureTest)
            
         
        mew = np.zeros((numClass, featureDim))
        variance = np.zeros((numClass, featureDim,featureDim))
        numInClass = np.zeros((numClass,1))
        priorProb = np.zeros((numClass,1))
        for i in range(trainNo):
            numInClass[labelInt[i]] += 1
            mew[labelTrain[i],] += featureTrain[i]
            
            
        for i in range(numClass):
            priorProb[i] = numInClass[i]/len(featureTrain)
            mew[i,] = mew[i,]/numInClass[i]
            
        for i in range(trainNo):
            variance[labelTrain[i],] += np.transpose(featureTrain[i]-mew[labelTrain[i],])*(featureTrain[i]-mew[labelTrain[i],])
            
        for i in range(numClass):
            variance[i,] = variance[i,]/(numInClass[i])
            variance[i,] = variance[i,]+np.eye(featureDim)
        
        postProb = np.zeros((numClass))   
        inData = np.zeros(featureDim) 
        predicted = np.zeros(len(labelTest))
        for i in range(testNo):
            for j in range(numClass):
                inData = featureTest[i]-mew[j]
                var = variance[j]
                
                if (classifier == 'bayes'): 
                    postProb[j] = (1/(np.sqrt(((2*np.pi)**featureDim)*np.linalg.det(var))))*np.exp(-0.5*np.linalg.pinv(var).dot(inData).dot(inData))*priorProb[j]
                if (classifier == 'naive'):
                    postProb[j] = 1;
                    for k in range(featureDim):
                        postProb[j] *= (1/(np.sqrt(2*np.pi*var[k,k])))*np.exp(-0.5*inData[k]*var[k,k]*inData[k])#*priorProb[j]
                postProb[j] *= priorProb[j]
            
            
            predicted[i] = np.argmax(postProb)
            
        Accuracy += ut.getAccuracy(predicted, labelTest)
        [m1,m2] = ut.fi_macro_micro(predicted,labelTest,numClass)
        f1_macro += m1
        f1_micro += m2
        print('Bayes: Accuracy is %f, F1 macro is %f and F1 micro is %f at iteration No = %d' %(Accuracy/cValid, f1_macro/cValid, f1_micro/cValid, cValid))
        
#        gnb = GaussianNB()
#        gnb.fit(featureTrain, labelTrain)
#        predicted = gnb.predict(featureTest)
#        
#        Accuracy_py += ut.getAccuracy(predicted, labelTest)
#        [m1,m2] = ut.fi_macro_micro(predicted,labelTest,numClass)
#        f1_macro_py += m1
#        f1_micro_py += m2
        
    
    print('Bayes: Avergae Accuracy is %f, Average F1 macro is %f and Average F1 micro is %f' %(Accuracy/(split-1), f1_macro/(split-1), f1_micro/(split-1)))
          
#    print('Avergae Accuracy is %f, Average F1 macro is %f and Average F1 micro is %f' %(Accuracy_py/(split-1), f1_macro_py/(split-1), f1_micro_py/(split-1)))
          
    return Accuracy/(split-1),f1_macro/(split-1),f1_micro/(split-1)




