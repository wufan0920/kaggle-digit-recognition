import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import math
import functools
import minfunc.lbfgs as minfunc


def normalization(inputs):
    m = np.array(inputs)
    means = m.mean(axis=0)
    pstd = 3*m.std()

    m = m - means
    m = np.maximum(np.minimum(m,pstd), -pstd)/pstd
    m = (m + 1) * 0.4 + 0.1;
    return m

def samplingImg(image):
    patchsize = 8
    numpatches = 10000
    patches = np.zeros((patchsize*patchsize,numpatches))

    images = sio.loadmat(image)
    data = images['IMAGES']

    for pic in range(10):
        picture = data[:,:,pic]
        row,col = picture.shape

        for patchNum in range(1000):
            xPos = random.randint(0, row-patchsize)
            yPos = random.randint(0, col-patchsize)

            index = pic*1000+patchNum
            patches[:,index:index+1] = np.reshape(picture[xPos:xPos+8,yPos:yPos+8],(64,1))

    return patches

def initializeParameters(hiddenSize, visibleSize):
    r  = math.sqrt(6) / math.sqrt(hiddenSize+visibleSize+1);
    W1 = np.random.rand(hiddenSize, visibleSize) * 2 * r - r;
    W2 = np.random.rand(visibleSize, hiddenSize) * 2 * r - r;

    b1 = np.zeros((hiddenSize, 1));
    b2 = np.zeros((visibleSize, 1));
    return W1,W2,b1,b2

def sigmoid(inputs):
    return 1.0/(1.0+np.exp(-inputs))

def sigInv(inputs):
    return sigmoid(inputs)*(1-sigmoid(inputs))

def sparseAutoencoderGradient(data,theta,visibleSize,hiddenSize,sparsityParam,lmd,beta):
    (n,m) = data.shape
    W1 = np.reshape(theta[:hiddenSize*visibleSize,], (hiddenSize, visibleSize))
    W2 = np.reshape(theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize,], (visibleSize, hiddenSize))
    b1 = np.reshape(theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize,],(hiddenSize,1))
    b2 = np.reshape(theta[2*hiddenSize*visibleSize+hiddenSize:,],(visibleSize,1))

    #step1 compute activations for each layer
    z2 = W1.dot(data) + b1
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2
    a3 = sigmoid(z3)

    #step2 compute cost 
    Jcost = (0.5/m)*np.sum((a3-data)*(a3-data))
    Jweight = 0.5*(np.sum(W1*W1)+np.sum(W2*W2))
    rho = (1.0/m)*np.sum(a2,axis=1)
    Jsparse = np.sum(sparsityParam*np.log(sparsityParam/rho)+\
                     (1-sparsityParam)*np.log((1-sparsityParam)/(1-rho)))
    cost = Jcost + lmd*Jweight + beta * Jsparse

    #step3 compute grad for each layer using back propagation
    d3 =(a3-data)*sigInv(z3)
    dsparse = beta*(-sparsityParam/rho+(1-sparsityParam)/(1-rho))
    dsparse = dsparse.reshape((len(dsparse),1))
    d2 = (W2.T.dot(d3)+dsparse)*sigInv(z2)

    #step4 compute gradient
    W1grad = np.zeros(W1.shape)
    W2grad = np.zeros(W2.shape)
    b1grad = np.zeros(b1.shape)
    b2grad = np.zeros(b2.shape)

    W1grad = W1grad + d2.dot(data.T)
    W1grad = (1.0/m)*W1grad+lmd*W1 

    W2grad = W2grad + d3.dot(a2.T)
    W2grad = (1.0/m)*W2grad+lmd*W2 

    b1grad = b1grad + np.sum(d2,axis=1).reshape(b1.shape)
    b2grad = b2grad + np.sum(d3,axis=1).reshape(b2.shape)
    b1grad = (1.0/m)*b1grad
    b2grad = (1.0/m)*b2grad
    grad = np.r_[W1grad.flatten(),W2grad.flatten(),b1grad.flatten(),b2grad.flatten()]
    return grad

def sparseAutoencoder(vSize,hSize,spars,lamda,b,a,data,iters=400,batches=1):
    #step1 initialize parameters
    W1,W2,b1,b2 = initializeParameters(hSize,vSize)
    theta = np.r_[W1.flatten(),W2.flatten(),b1.flatten(),b2.flatten()]

    #step2 wrap grad function paramameters
    grad_func = functools.partial(sparseAutoencoderGradient,visibleSize=vSize,hiddenSize=hSize,sparsityParam=spars,lmd=lamda,beta=b)

    #step3 lbfgs
    W = minfunc.lbfgs_without_validate(data,theta,grad_func,alpha=a,max_iter=iters,batch=batches)
    return W

if __name__=='__main__':
    #public variables
    visibleSize = 8*8;
    hiddenSize = 25;
    sparsityParam = 0.01;
    lmd = 0.0001;
    beta = 3;     
    alpha = 0.3;     
    #step1 load&sampling images
    X = samplingImg('dataset/IMAGES.mat')
    X = normalization(X)

    #step2 extract features
    W = sparseAutoencoder(visibleSize,hiddenSize,sparsityParam,lmd,beta,alpha,X)
    print 'done'
    W1 = np.reshape(W[:hiddenSize*visibleSize,], (hiddenSize, visibleSize))

    #step3 plot the weight
    fig = plt.figure(2)

    for index in range(hiddenSize):
        weight = W1[index,:]
        weight = np.reshape(weight,(8,8))
        ax = fig.add_subplot(5,5,1+index)
        ax.imshow(weight,mpl.cm.gray)

    plt.show()
