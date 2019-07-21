from numpy import *


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOLines = len(arrayOLines)
    returnMat = zeros((numberOLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # index 数组的索引值
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


if __name__ == '__main__':
   datingDataMat,datingLabels = file2matrix('./train-set/datingTestSet.txt')
   print(datingDataMat)
   print(datingLabels)