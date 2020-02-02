'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def loadDataset():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt', 'r', encoding='utf-8')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat

def sigmoid(inx):
    return 1.0 / (1 + exp(-inx))

def gradient_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_matrix = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.5
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((500*m,n))
    for j in range(500):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
            weightsHistory[j*m + i,:] = weights
    return weightsHistory

def stocGradAscent1(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.4
    weights = ones(n)   #initialize to all ones
    weightsHistory=zeros((40*m,n))
    for j in range(40):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            #print error
            weights = weights + alpha * error * dataMatrix[randIndex]
            weightsHistory[j*m + i,:] = weights
            del(dataIndex[randIndex])
    print(weights)
    return weightsHistory
    

def plot_sd_error(func):
    dataMat,labelMat=loadDataset()
    dataArr = array(dataMat)
    
    myHist = func(dataArr,labelMat)


    n = shape(dataArr)[0] #number of points to create
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []

    markers =[]
    colors =[]


    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(311)
    type1 = ax.plot(myHist[:,0])
    plt.ylabel('X0')
    ax = fig.add_subplot(312)
    type1 = ax.plot(myHist[:,1])
    plt.ylabel('X1')
    ax = fig.add_subplot(313)
    type1 = ax.plot(myHist[:,2])
    plt.xlabel('iteration')
    plt.ylabel('X2')
    plt.show()