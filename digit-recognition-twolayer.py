import numpy as np
import csv
import nnet
import cnn
import softmax
import random
import math
import validation


def load_data(pathX):
    matrixX = [] 
    dataX = file(pathX)

    for line in dataX.readlines():
        row = []
        x = line.strip().split(',')
        for column in x:
            tmp = float(column)
            row.append(tmp)
        matrixX.append(row)
    
    return matrixX

def savedata(data,filename):
    data = str(data.tolist())
    save_file = open(filename,'w')
    save_file.write(data)
    save_file.flush()
    save_file.close()

def samplingFeatureMap(data):
    patchsize = 5
    patchlen = patchsize*patchsize
    nummaps = len(data)
    numpatches = 185000
    patches = np.zeros((patchlen*nummaps,numpatches))

    for pic in range(37000):
        for patchNum in range(5):
            xPos = random.randint(0, 12-patchsize)
            yPos = random.randint(0, 12-patchsize)

            index = pic*10+patchNum
            for i in range(nummaps):
                picture = data[i][:,pic]
                picture.shape = (12,12)

                patches[patchlen*i:patchlen*(i+1),index:index+1] = np.reshape(picture[xPos:xPos+patchsize,yPos:yPos+patchsize],(patchsize*patchsize,1))

    return patches

def samplingImg(data):
    patchsize = 5
    numpatches = 370000
    patches = np.zeros((patchsize*patchsize,numpatches))

    for pic in range(37000):
        picture = data[:,pic]
        picture.shape = (28,28)

        for patchNum in range(10):
            xPos = random.randint(0, 28-patchsize)
            yPos = random.randint(0, 28-patchsize)

            index = pic*10+patchNum
            patches[:,index:index+1] = np.reshape(picture[xPos:xPos+patchsize,yPos:yPos+patchsize],(patchsize*patchsize,1))

    return patches

def predict(theta, data, labels):
    predict = (theta.dot(data)).argmax(0)
    accuracy = (predict == labels.flatten())
    print 'Accuracy:',accuracy.mean()
    return predict

def loadTrainingData(path):
    l=[]
    with open(path) as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l = np.array(l, dtype = np.float)
    label=l[:,0]
    data=l[:,1:]
    return data, label

def loadTestingData(path):
    l=[]  
    with open(path) as file:  
         lines=csv.reader(file)  
         for line in lines:  
             l.append(line) #28001*784  
    l.remove(l[0])  
    data = np.array(l, dtype = np.float)
    return data

def loadTestResult():  
    l=[]  
    with open('dataset/rf_benchmark.csv') as file:  
         lines=csv.reader(file)  
         for line in lines:  
             l.append(line) #28001*2  

    l.remove(l[0])  
    label = np.array(l, dtype = np.int)
    label = label[:,1]
    return label

def prepareData():
    raw_data, raw_label = loadTrainingData('dataset/train.csv')
    test_data = loadTestingData('dataset/test.csv')

    #nomalize the train & test data
    raw_data = nnet.normalization(raw_data)
    test_data = nnet.normalization(test_data)
    
    #split training data to training set & validating set
    training_data = raw_data[:37000,:]
    training_label = raw_label[:37000,]
    validating_data = raw_data[37000:,:]
    validating_label = raw_label[37000:,]
    
    return training_data.transpose(),training_label,validating_data.transpose(),validating_label,test_data.transpose()

def saveResult(result):  
    with open('dataset/result.csv','wb') as myFile:      
        myWriter=csv.writer(myFile)  
        for i in result:  
            tmp=[]  
            tmp.append(i)  
            myWriter.writerow(tmp) 

if __name__=='__main__':
    #public variables
    patchsize = 5 
    inputSize = patchsize*patchsize
    numClasses = 10
    hiddenSizeL1 = 6
    hiddenSizeL2 = 16
    sparsityParam = 0.1
    beta = 3
    lmd = 0.001
    alpha = 0.07
    
    #step 1 load dataset
    training_data,training_label,validating_data,validating_label,test_data = prepareData()
    print 'load done'

    training_set = samplingImg(training_data)

    #step 2 L1 feature learning using sparse autoencoder
    #W = nnet.sparseAutoencoder(inputSize,hiddenSizeL1,sparsityParam,lmd,beta,alpha,training_set)
    #savedata(W,'weightL1')
    W = load_data('weightL1')
    W = np.array(W)
    W = W.transpose()
    W1 = np.reshape(W[:hiddenSizeL1*inputSize,], (hiddenSizeL1, inputSize))
    b1 = np.reshape(W[2*hiddenSizeL1*inputSize:2*hiddenSizeL1*inputSize+hiddenSizeL1,],(hiddenSizeL1,1))

    #step3 convolution layer, compute feature map
    #TODO extract imagesize
    step =1
    imagesize = 28
    convWeight = cnn.convolutionWeight(W1, patchsize, imagesize, step)
    featureMap = cnn.convolutionFeatureMap(training_data, b1, convWeight)

    #step4 pooling layer
    poolingSize = 2
    poolingCore = 1.0/math.pow(poolingSize, 2) * np.ones((1, poolingSize*poolingSize))
    featureSize = math.sqrt(featureMap[0].shape[0])
    poolingWeight = cnn.convolutionWeight(poolingCore, poolingSize, featureSize, poolingSize)
    poolingWeight = poolingWeight[0]
    convData = cnn.pooling(featureMap, poolingWeight)

    #step 5 L2 feature learning using sparse autoencoder
    #step 5.1 sampling L1 feature maps & train sae
    training_set = samplingFeatureMap(convData)
    inputSizeL2 = inputSize * hiddenSizeL1#len(convData)
    #W = nnet.sparseAutoencoder(inputSizeL2,hiddenSizeL2,sparsityParam,lmd,beta,alpha,training_set)
    #savedata(W,'weightL2')
    W = load_data('weightL2')
    W = np.array(W)
    W = W.transpose()
    W2 = np.reshape(W[:hiddenSizeL2*inputSizeL2,], (hiddenSizeL2, inputSizeL2))
    b2 = np.reshape(W[2*hiddenSizeL2*inputSizeL2:2*hiddenSizeL2*inputSizeL2+hiddenSizeL2,],(hiddenSizeL2,1))

    #step5.2 convolution layer, compute feature map
    #TODO extract imagesize
    step =1
    imagesize = 12
    convWeightL2 = []
    for feature in range(hiddenSizeL1):
        featWeight = W2[:,feature*inputSize:inputSize*(feature+1)] 
        tmp = cnn.convolutionWeight(featWeight, patchsize, imagesize, step)
        convWeightL2.append(tmp)
    featureMap = cnn.convolutionFeatureMapMulti(convData, b2, convWeightL2)

    #step5.3 pooling layer
    featureSize = math.sqrt(featureMap[0].shape[0])
    poolingWeightL2 = cnn.convolutionWeight(poolingCore, poolingSize, featureSize, poolingSize)
    poolingWeightL2= poolingWeightL2[0]
    convData = cnn.pooling(featureMap, poolingWeightL2)
    convData = cnn.mergeRow(convData)
    print 'done'

    #step6 softmax regression 
    inputSize = convData.shape[0]
    valid_featureMap = cnn.convolutionFeatureMap(validating_data, b1, convWeight)
    valid_convData = cnn.pooling(valid_featureMap, poolingWeight)
    valid_featureMap = cnn.convolutionFeatureMapMulti(valid_convData, b2, convWeightL2)
    valid_convData = cnn.pooling(valid_featureMap, poolingWeightL2)
    valid_convData = cnn.mergeRow(valid_convData)
    validator = validation.validator(valid_convData, validating_label, (numClasses, inputSize))

    W = softmax.softmax_regression(inputSize,numClasses,0,convData,training_label,7000,validator,a=1.9)

    #step7 testing
    print 'testing'
    featureMap = cnn.convolutionFeatureMap(test_data, b1, convWeight)
    convData = cnn.pooling(featureMap, poolingWeight)
    featureMap = cnn.convolutionFeatureMapMulti(convData, b2, convWeightL2)
    convData = cnn.pooling(featureMap, poolingWeightL2)
    convData = cnn.mergeRow(convData)

    theta = W.reshape((numClasses, -1))
    benchmark = loadTestResult()
    result = predict(theta, convData, benchmark)
    saveResult(result.tolist())
    print 'done'
