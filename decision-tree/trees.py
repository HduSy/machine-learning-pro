from math import log


# 计算给定数据集香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts.keys():
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 关键计算公式
    return shannonEnt


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# TODO: 求最佳特征划分数据集 测试每个特征划分后的熵
if __name__ == '__main__':
    myMat, labels = createDataSet()
    print(calcShannonEnt(myMat))
    # 随着分类的增多 熵越来越大
    myMat[0][-1] = 'maybe'
    print(calcShannonEnt(myMat))
    myMat[2][-1] = 'habby'
    print(calcShannonEnt(myMat))
    myMat[3][-1] = 'clc'
    print(calcShannonEnt(myMat))
