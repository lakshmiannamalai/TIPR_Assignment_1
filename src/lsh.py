import csv
import numpy as np
import math
from scipy.linalg import qr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import BagOfWords as BW
import util as ut


def projMat(m,n):

    H = np.random.randn(m,n)
    Q, R = qr(H)
    print(Q)
    return (H)



def hamming(string_1, string_2):
   distance = 0
   for char1, char2 in zip(string_1, string_2):
       if char1 != char2:
           distance += 1
   return distance

def euDist(featureTest, featureTrain):
    dist = 0
    for x in range(len(featureTest)):
        dist += pow((featureTest[x] - featureTrain[x]), 2)
    dist = math.sqrt(dist)
    return dist
    


def LSH(dataFile,labelFile,dataset):
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
    
    reduceDim = int(featureDim/2)
    NoOfAnd = 2
    NoOfOr = 5
    NoOfTables = NoOfAnd * NoOfOr
    Rmat = np.zeros((NoOfTables,featureDim, reduceDim))
    for table in range(NoOfTables):
        Rmat[table] = np.random.normal(0,1,(featureDim, reduceDim))
    
    Accuracy = 0
    f1_macro = 0
    f1_micro = 0
    Accuracy_pca = 0
    f1_macro_pca = 0
    f1_micro_pca = 0
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
        
        hashFeature = list()
        
        
        
        reduceFeature = np.zeros((NoOfTables,len(featureTrain),reduceDim))
        for table in range(NoOfTables):
            
            
            np.matmul(featureTrain, Rmat[table], reduceFeature[table])
            
            hash = (np.dot(featureTrain, Rmat[table]) > 0).astype(int).astype(str)
            
            hashFeature.append(hash)
            
        
            
        hashDist = np.zeros((NoOfTables,len(featureTrain)))
        hashTest = np.zeros((reduceDim))
        index = np.zeros((NoOfTables))
        classIndex = np.zeros((numClass))
        predicted = np.zeros(len(labelTest))
        if (dataset == 'pubmed'):
            sampleSize = 10
        else:
            sampleSize = len(featureTest)
        for ftre in range(sampleSize):
            for table in range(NoOfTables):
                hashTrain = hashFeature[table]
                hashTest = (np.dot(featureTest[ftre], Rmat[table]) > 0).astype(int).astype(str)
                for train in range(len(hashTrain)):
                    hashDist[table,train] = hamming(hashTest,hashTrain[train])
                    
                index[table] = labelTrain[np.argmin(hashDist[table])]
            
            

            tempIndex = []
            for j in range(NoOfOr):
                res = all(ele == index[j*NoOfAnd] for ele in index[j*NoOfAnd:(j+1)*NoOfAnd]) 
                if(res):
                    tempIndex.append(index[j*NoOfAnd])
                    
            for j in range(numClass):
               classIndex[j] = sum(x == j for x in index)
            predicted[ftre] = np.argmax(classIndex)
            
            
        Accuracy += ut.getAccuracy(predicted, labelTest[0:sampleSize-1])
        [m1,m2] = ut.fi_macro_micro(predicted,labelTest[0:sampleSize-1],numClass)
        f1_macro += m1
        f1_micro += m2
        
        print('LSH: Accuracy is %f, F1 macro is %f and F1 micro is %f at iteration No = %d' %(Accuracy/cValid, f1_macro/cValid, f1_micro/cValid, cValid))
        
    
    
    
#        scaler = StandardScaler()
#        scaler.fit(featureTrain)
#        featureTrain = scaler.transform(featureTrain)
#        featureTest = scaler.transform(featureTest)
#        pca = PCA(.95)
#        pca.fit(featureTrain)
#        featureTrain = pca.transform(featureTrain)
#        featureTest = pca.transform(featureTest)
#        
#        
#        trainNo = len(featureTrain)
#        testNo = len(featureTest)
#        featureDim = len(featureTest.T)
#        for i in range(sampleSize):
#            distance = []
#            for j in range(trainNo):
#                dist = 0
#                for x in range(featureDim):
#                    dist += pow((featureTest[i][x] - featureTrain[j][x]), 2)
#                distance.append(math.sqrt(dist))
#                #print(labelTest[i],labelTrain[np.argmin(distance)])
#            predicted[i] = labelTrain[np.argmin(distance)]
#                
#        Accuracy_pca += ut.getAccuracy(predicted, labelTest[0:sampleSize-1])
#        [m1,m2] = ut.fi_macro_micro(predicted,labelTest[0:sampleSize-1],numClass)
#        f1_macro_pca += m1
#        f1_micro_pca += m2
        
#        print('LSH: Accuracy is %f, F1 macro is %f and F1 micro is %f at iteration No = %d' %(Accuracy_pca/cValid, f1_macro_pca/cValid, f1_micro_pca/cValid, cValid))
        
    
    
    print('LSH: Avergae Accuracy is %f, Average F1 macro is %f and Average F1 micro is %f' %(Accuracy/(split-1), f1_macro/(split-1), f1_micro/(split-1)))
          
#    print('Avergae Accuracy is %f, Average F1 macro is %f and Average F1 micro is %f' %(Accuracy_pca/(split-1), f1_macro_pca/(split-1), f1_micro_pca/(split-1)))
          
    #print('Avergae Accuracy is %f, Average F1 macro is %f and Average F1 micro is %f' %(Accuracy_py/(split-1), f1_macro_py/(split-1), f1_micro_py/(split-1)))
          
    return Accuracy/(split-1),f1_macro/(split-1),f1_micro/(split-1)