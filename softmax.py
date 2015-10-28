import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import functools
import minfunc.lbfgs as minfunc

def softmax_gradient(data,theta,input_size,num_classes,input_labels,lmd):
    theta = theta.reshape((num_classes,input_size))
    numCases = data.shape[1]

    grad = np.zeros((num_classes,input_size))

    #construct label sparse matrix
    row = input_labels.flatten().tolist()
    col = range(numCases)
    elements = [1]*numCases

    grandTruth = sp.coo_matrix((elements,(row,col)),shape=(num_classes,numCases)).todense()
    grandTruth = np.array(grandTruth)

    #to avoid exponential overflow score need to minus max value of each column
    score = theta.dot(data)
    max_values = score.max(axis=0)
    score = score - max_values
    M = np.exp(score)
    p = M/M.sum(axis=0)

    grad = -1/numCases * (grandTruth - p).dot(data.transpose()) + lmd*theta

    return grad.flatten()


def softmax_regression(size,classes,lamda,data,labels,iters,validator,a=0.3):
    #initialize theta
    #TODO figure out why 0.005?
    theta = 0.005 * np.random.rand(size*classes,)

    #wrap initail parameters
    grad_func = functools.partial(softmax_gradient,input_size=size,num_classes=classes,input_labels=labels,lmd=lamda)

    #lbfgs
    W = minfunc.lbfgs(data,theta,grad_func,validator,iters,alpha=a)
    return W

if __name__=='__main__':
    #public variables
    inputSize = 28*28
    numClasses = 10
    lmd = 0.0001

    #step1 load mnist training data & change 0 to 10
    image = sio.loadmat('dataset/mnist_dataset/mnist_train.mat')
    label = sio.loadmat('dataset/mnist_dataset/mnist_train_labels.mat')
    data = image['mnist_train']
    labels = label['mnist_train_labels']

    data = data.transpose()

    #step2 training
    W = softmax_regression(inputSize,numClasses,lmd,data,labels,100)

    #step3 testing
    image = sio.loadmat('dataset/mnist_dataset/mnist_test.mat')
    label = sio.loadmat('dataset/mnist_dataset/mnist_test_labels.mat')
    data = image['mnist_test']
    labels = label['mnist_test_labels']
    data = data.transpose()

    theta = W.reshape((numClasses, inputSize))
    predict = (theta.dot(data)).argmax(0)
    print predict
    print labels.flatten()
    accuracy = (predict == labels.flatten())
    print 'Accuracy:',accuracy.mean()

    print 'done'
    
    
