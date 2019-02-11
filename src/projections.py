# Implement code for random projections here!
import csv
import numpy as np
import BagOfWords as BW





def RandomProjection(dataFile,labelFile,dataset):
    
    if dataset == 'twitter':
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
    
        with open(dataFile) as csv_file:
            file_id = csv.reader(csv_file, delimiter=',')
            index = 0
            feature = []
            for line in file_id:
                feature.append(line)
                #print(f'\t{feature[index]}.')
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
            
    if dataset == 'twitter':
        tempStr = dataFile.split(".txt")
    else:
        tempStr = dataFile.split(".csv")
    
    
    upLimit = int(featureDim/2)
    Rfullmat = np.random.normal(0,1,(featureDim, upLimit))
    for reduceDim in range(2, upLimit, 2):
        reduceFeature = np.zeros((len(featureInt),reduceDim))
        #Rmat = np.random.normal(0,1,(featureDim, reduceDim))
        Rmat = Rfullmat[:,0:reduceDim]
        np.matmul(featureInt, Rmat, reduceFeature)
        fileName = tempStr[0] + '_{}.csv'.format(reduceDim) 
        with open(fileName, 'w') as f:
            writer = csv.writer(f,delimiter=' ')
            writer.writerows(reduceFeature)
        
        f.close()
    
    
