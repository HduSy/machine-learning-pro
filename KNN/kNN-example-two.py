from numpy import *
import matplotlib.pyplot as plt
import operator

# 分类器
def classify0(inputX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 行数
    diffMat = tile(inputX, (dataSetSize, 1)) - dataSet  # 行方向重复dataSetSize次,列方向重复1次形成新数组
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 0 竖向相加 1 横向相加
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()  # axis = -1 按序返回值对应索引值,默认降序大->小
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndices[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1  # dict.get(key, default=None)
    # 排序函数(接受可迭代对象为第一个参数)-以列表返回可遍历的(键,值)元组数组-获取对象特定维-是否反转
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 默认升序
    return sortedClassCount[0][0]

# 准备数据 文件转矩阵
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


# 归一化特征值
# newVal = (oldVal-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # ():所有值中最小值 (0):按列,每列最小值 (1):按行,每行最小值
    maxVals = dataSet.max(0)
    range = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = dataSet / tile(range, (m, 1))
    return normDataSet, range, minVals

# 准确率
def datingClassTest():
    hoRatio = 0.1  # 取10%
    datingDataMat, datingLabels = file2matrix('./train-set/datingTestSet.txt')
    normMat, range, minVal = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errCount = 0
    for i in range(numTestVecs):
        classifierRet = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print('the classifier came back with: %d, the real answer is %d' % (classifierRet, datingLabels[i]))
        if classifierRet != datingLabels[i]: errCount += 1
    print('the total error rate is : %f' % (errCount / float(numTestVecs)))

# 输入测试
def classPerson():
    retList = ['not at all', 'in small doses', 'in large doses']
    pertcentTats = float(input('percent of time spend on video games?'))
    ffMiles = float(input('frequent filter miles earned per year?'))
    iceCream = float(input('liters of ice cream comsumed per year?'))
    datingDataMat, datingLabels = file2matrix('./train-set/datingTestSet.txt')
    normMat, range, minVal = autoNorm(datingDataMat)
    inArr = array([ffMiles, pertcentTats, iceCream])
    classifierRet = classify0((inArr - minVal) / range, normMat, datingLabels, 3)
    print('You will probably like this person:', retList[classifierRet - 1])


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('./train-set/datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)  # (nrows ,ncols ,index)defaults to (1, 1, 1)
    # (x,y,s=None,c=None,marker=None,...) 散点size 散点color
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15 * array(datingLabels), array(datingLabels))  # [:,x]取第x个值
    plt.show()
    # datingClassTest()  # 报错
    classPerson()
