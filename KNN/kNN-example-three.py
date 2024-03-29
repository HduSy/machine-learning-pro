from numpy import *


# 二进制图像文件转matrix
def img2matrix(filename):
    returnMat = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnMat[0, 32 * i + j] = int(lineStr[j])
    return returnMat
