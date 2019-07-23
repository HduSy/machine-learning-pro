from math import log
import operator
import matplotlib.pyplot as plt


def createDataSet():
    # 以第0个特征划分数据集效果最好
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    # 以第1个特征划分数据集效果最好
    dataSet2 = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'yes'],
        [0, 1, 'yes']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算给定数据集香农熵 shannonEnt越小,纯度数据集越高
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


# dataSet:划分数据集 axis:第axis个特征 value:特征值
# 当选择某个特征划分数据集时,抽取出符合特征的所有元素
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if (featVec[axis] == value):
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    # return featVec[axis] == value 的数据集
    return retDataSet


# 选择划分效果最好的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数 -1:最后一位是类标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    # 循环遍历所有特征 依次取数据集中第i个特征划分数据集
    for i in range(numFeatures):
        # 取第i个特征的特征列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 从分类标签列表中抽取不重复的标签组成集合,简单来说去重
        newEntropy = 0
        # 计算每种划分方式的信息熵,eg.['red','green','blue']
        for val in uniqueVals:
            # dataSet:划分数据集 i:第i个特征 val:特征值
            subDataSet = splitDataSet(dataSet, i, val)
            # 特征i对应特征值val形成的子集subDataSet占比
            prob = len(subDataSet) / float(len(dataSet))
            # TODO:7.23 为什么这样计算newEntropy?
            # 因为不同分支节点样本数不同,给分支节点赋予权重,即样本越多的分支节点影响越大
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 内层循环结束,本特征划分,信息增益情况
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # 外层循环结束,测试下一个特征
    return bestFeature


# 叶节点类标签不唯一时,'投票表决'
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    # dict.items() 以列表返回可遍历的(键,值)元组数组
    sortedClassCount = sorted(classCount.items(), key=operator.getitem(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]
    # TODO: why == 1
    # 因为这是只有(剩)一个特征的情况 仍不只一种类标签
    # eg.len(dataSet[0])、len(dataSet[1])、len(dataSet[2])...=1
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featVals = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featVals)
    for val in uniqueVals:
        sublabels = labels[:]  # 已删除bestFeat
        myTree[bestFeatLabel][val] = createTree(splitDataSet(dataSet, bestFeat, val), sublabels)
    return myTree


# 递归
def classify(inputTree, featLabels, testVect):
    firstStr = list(inputTree.keys())[0]
    secondTree = inputTree[firstStr]

    featIndex = featLabels.index(firstStr)  # 0/1/2/3/...
    key = testVect[featIndex]

    valueOfeat = secondTree[key]
    if isinstance(valueOfeat, dict):
        classLabel = classify(valueOfeat, featLabels, testVect)
    else:
        classLabel = valueOfeat
    return classLabel


def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


# 序列化对象到文件中去
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


# 反序列化文件
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


# TODO-7.23: 求最佳特征划分数据集 测试每个特征划分后的熵
if __name__ == '__main__':
    myMat, labels = createDataSet()
    # TEST1
    # 随着分类的增多 熵越来越大
    # myMat[0][-1] = 'maybe'
    # print(calcShannonEnt(myMat))
    # myMat[2][-1] = 'habby'
    # print(calcShannonEnt(myMat))
    # myMat[3][-1] = 'clc'
    # print(calcShannonEnt(myMat))
    # TEST2
    # print('第%s个特征是最好的用于划分数据集的特征' % chooseBestFeatureToSplit(myMat))
    # print(calcShannonEnt(myMat))
    # 决策树分类器测试
    myTree = retrieveTree(0)
    print(classify(myTree, labels, [1, 1]))
