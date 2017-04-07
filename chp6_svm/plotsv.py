'''
Created on Nov 22, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plotsv(b_mat, w_mat, circles):

    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    markers =[]
    colors =[]
    fr = open('testSet.txt')#this file was generated by 2normalGen.py
    for line in fr.readlines():
        lineSplit = line.strip().split('\t')
        xPt = float(lineSplit[0])
        yPt = float(lineSplit[1])
        label = int(lineSplit[2])
        if (label == -1):
            xcord0.append(xPt)
            ycord0.append(yPt)
        else:
            xcord1.append(xPt)
            ycord1.append(yPt)

    fr.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord0,ycord0, marker='s', s=90)
    ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
    plt.title('Support Vectors Circled')
    for circle_p in circles:
        circle = Circle((circle_p[0], circle_p[1]), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    #plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane
    # b = -3.75567; w0=0.8065; w1=-0.2761
    b = b_mat[0, 0]
    w0 = w_mat[0][0]
    w1 = w_mat[1][0]
    x = arange(-2.0, 12.0, 0.1)
    y = (-w0*x - b)/w1
    ax.plot(x,y)
    ax.axis([-2,12,-8,6])
    plt.show()