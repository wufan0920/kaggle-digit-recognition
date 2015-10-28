import scipy.io as sio
import numpy as np
import nnet
import softmax
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import math

def samplingImg(data):
    patchsize = 9
    numpatches = 24000
    patches = np.zeros((patchsize*patchsize,numpatches))

    for pic in range(800):
        picture = data[:,pic]
        picture.shape = (50,50)

        for patchNum in range(30):
            xPos = random.randint(0, 50-patchsize)
            yPos = random.randint(0, 50-patchsize)

            index = pic*30+patchNum
            patches[:,index:index+1] = np.reshape(picture[xPos:xPos+patchsize,yPos:yPos+patchsize],(patchsize*patchsize,1))

    return patches

def convolutionWeight(weight, patchsize, imagesize, step):
    result = []
    (n, size) = weight.shape
    for index in range(n):
        core = weight[index:index+1,:]
        xPos = 0
        yPos = 0
        weightMap = 'null'
        while yPos + patchsize <= imagesize:
            while xPos + patchsize <= imagesize:
                tmp = np.zeros((1, imagesize*imagesize))
                for i in range(patchsize):
                    insertPos = (yPos + i) * imagesize + xPos
                    tmp[:,insertPos:insertPos+patchsize] = core[:,patchsize*i:patchsize*(i+1)]
                if weightMap == 'null':
                    weightMap = tmp
                else:
                    weightMap = np.r_[weightMap, tmp]
                xPos += step
            yPos += step
            xPos = 0 
        result.append(weightMap)
    return result

def convolutionFeatureMap(data, bias, w):
    featureMap = []
    size = len(w)
    for index in range(size):
        result = w[index].dot(data) + bias[index,:]
        featureMap.append(nnet.sigmoid(result))
    return featureMap

def convolutionFeatureMapMulti(data, bias, w):
    featNum = len(data)  #how many L1 feature maps to be convoluted
    size = len(w[0]) #how many L2 features(covolution cores) dose each L1 feature map have
    #so variable w is the array that [L1-feat-nums * L2-feat-nums * each-convlution-weight]

    featureMap = []
    for index in range(size):
        for i in range(featNum):
            if i == 0:
                tmp = w[i][index].dot(data[i])
            else:
                tmp += w[i][index].dot(data[i])

        result = tmp + bias[index,:]
        featureMap.append(nnet.sigmoid(result))
    return featureMap

def pooling(featureMap, w):
    #TODO implement max pooling
    result = []
    for index in range(len(featureMap)):
        result.append(w.dot(featureMap[index]))
    return result

def mergeRow(data):
    for index in range(len(data)):
        if index == 0:
            result = data[index]
        else:
            result = np.r_[result, data[index]]
    return result

def mergeCol(data):
    for index in range(len(data)):
        if index == 0:
            result = data[index]
        else:
            result = np.c_[result, data[index]]
    return result

def predict(theta, data, labels):
    predict = (theta.dot(data)).argmax(0)
    accuracy = (predict == labels.flatten())
    print 'Accuracy:',accuracy.mean()


if __name__=='__main__':
    #public variables
    patchsize = 9 
    inputSize = patchsize*patchsize
    numClasses = 10
    hiddenSizeL1 = 5
    sparsityParam = 0.1
    beta = 3
    lmd = 0.001
    alpha = 0.07

    #step1 load training data
    #TODO move normalization to miscellaneous
    file_path = 'dataset/faces.mat'

    images = sio.loadmat(file_path)
    data = images['data']
    data = data.transpose()
    #normalize data
    data = nnet.normalization(data)

    #split data to training set and testing set
    testing_lables = []
    training_labels = []
    for index in range(10):
        train = data[:,90*index:90*index+80]
        test = data[:,90*index+80:90*(index+1)]
        if index == 0:
            training_set = train 
            testing_set = test
        else:
            training_set = np.c_[training_set, train]
            testing_set = np.c_[testing_set, test]

        labels = 10*[index]
        testing_lables += labels
        training_labels += 8*labels
    testing_lables = np.array(testing_lables)
    training_labels = np.array(training_labels)

    training_data = samplingImg(training_set)

    #step2 L1 feature learning using sparse autoencoder
    W = nnet.sparseAutoencoder(inputSize,hiddenSizeL1,sparsityParam,lmd,beta,alpha,training_data)
    W1 = np.reshape(W[:hiddenSizeL1*inputSize,], (hiddenSizeL1, inputSize))
    b1 = np.reshape(W[2*hiddenSizeL1*inputSize:2*hiddenSizeL1*inputSize+hiddenSizeL1,],(hiddenSizeL1,1))

    #step3 convolution layer, compute feature map
    #TODO extract imagesize
    step =1
    imagesize = 50
    convWeight = convolutionWeight(W1, patchsize, imagesize, step)
    featureMap = convolutionFeatureMap(training_set, b1, convWeight)
    #step4 pooling layer
    poolingSize = 2
    poolingCore = 1.0/math.pow(poolingSize, 2) * np.ones((1, poolingSize*poolingSize))
    featureSize = math.sqrt(featureMap[0].shape[0])
    poolingWeight = convolutionWeight(poolingCore, poolingSize, featureSize, poolingSize)
    poolingWeight = poolingWeight[0]
    convData = pooling(featureMap, poolingWeight)
    convData = mergeRow(convData)

    inputSize = convData.shape[0]

    #step5 softmax regression 
    W = softmax.softmax_regression(inputSize,numClasses,0.003,convData,training_labels,100)

    #step6 validation & testing
    inputSize = convData.shape[0]
    theta = W.reshape((numClasses, inputSize))

    print 'validation'
    predict(theta, convData, training_labels)

    print 'testing'
    featureMap = convolutionFeatureMap(testing_set, b1, convWeight)
    convData = pooling(featureMap, poolingWeight)
    convData = mergeRow(convData)
    #convData = mergeRow(featureMap)
    predict(theta, convData, testing_lables)
    print 'done'
