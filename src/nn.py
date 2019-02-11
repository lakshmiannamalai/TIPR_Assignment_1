# Implement Nearest Neighbour classifier here!
import csv
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import BagOfWords as BW
from sklearn.preprocessing import StandardScaler
import util as ut




def NearestNeighbor(dataFile,labelFile,dataset):
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
            #if labelFile.find('.txt') == -1:
            file_id = csv.reader(csv_file, delimiter=';')
            index = 0
            label = []
            for line in file_id:
                label.append(line)
                #print(f'\t{label[index]}.')
                index += 1
            labelInt = [int(i[0]) for i in label]   
            #else:
                
                
        
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
        
    
    
    
    
    numInClass = np.zeros((numClass,1))
    for i in range(len(featureInt)):
        numInClass[labelInt[i]] += 1
        
        
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
        
        predicted = np.zeros(len(labelTest))
        if (dataset == 'pubmed'):
            sampleSize = 10
        else:
            sampleSize = len(featureTest)
        for i in range(sampleSize):
            distance = []
            for j in range(trainNo):
                dist = 0
                for x in range(featureDim):
                    dist += pow((featureTest[i][x] - featureTrain[j][x]), 2)
                distance.append(math.sqrt(dist))
            #print(labelTest[i],labelTrain[np.argmin(distance)])
            predicted[i] = labelTrain[np.argmin(distance)]
            
        Accuracy += ut.getAccuracy(predicted, labelTest[0:sampleSize-1])
        [m1,m2] = ut.fi_macro_micro(predicted,labelTest[0:sampleSize-1],numClass)
        f1_macro += m1
        f1_micro += m2
        print('Nearest Neighbor: Accuracy is %f, F1 macro is %f and F1 micro is %f at iteration No = %d' %(Accuracy/cValid, f1_macro/cValid, f1_micro/cValid, cValid))
        
        

#        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(featureTrain)
#        distances, indices = nbrs.kneighbors(featureTest)
#        predicted = np.zeros(len(labelTest))
#        for i in range(testNo):
#            predicted[i] = labelTrain[indices[i][0]]
#            #print(labelTest[i],labelTrain[indices[i][0]])
#            
#        Accuracy_py += ut.getAccuracy(predicted, labelTest[0:sampleSize-1])
#        [m1,m2] = ut.fi_macro_micro(predicted,labelTest[0:sampleSize-1],numClass)
#        f1_macro_py += m1
#        f1_micro_py += m2

        
    print('Avergae Accuracy is %f, Average F1 macro is %f and Average F1 micro is %f' %(Accuracy/(split-1), f1_macro/(split-1), f1_micro/(split-1)))
          
#    print('Avergae Accuracy is %f, Average F1 macro is %f and Average F1 micro is %f' %(Accuracy_py/(split-1), f1_macro_py/(split-1), f1_micro_py/(split-1)))
          
    return Accuracy/(split-1),f1_macro/(split-1),f1_micro/(split-1)