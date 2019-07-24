from math import *
from numpy import *


# 加载数据集
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('./data-set/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid函数
def sigmoid(z):
    return 1.0 / (1 + exp(-z))


def gradAscent(dataIn, classLabels):
    dataMatrix = mat(dataIn)
    y = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    # m*n矩阵 * n*1矩阵 = m*1矩阵
    for k in range(maxCycles):
        # z = w*x
        # a = sigmoid(z)
        # w := w + alpha * gradient(w)
        a = sigmoid(dataMatrix * weights)
        error = y - a
        # n*m矩阵 * m*1矩阵 = n*1矩阵
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    gradAscent(dataMat, labelMat)
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights.getA())  # getA()将weights矩阵转换为数组，getA()函数与mat()函数的功能相反