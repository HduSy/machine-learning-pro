from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


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


if __name__ == '__main__':
    group, labels = createDataSet()
    type = classify0([0, 0], group, labels, 3)
    print(type)
