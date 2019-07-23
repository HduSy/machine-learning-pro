from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


# 所有不重复单词组成词s列表
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


# 输入参数:词汇表,测试文档
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    # 输出:测试文档词's 词向量
    return returnVec


# eg.词向量样本[[0,0,1],[0,1,0],[1,1,0],[1,0,1]],类别[1,0,1,0]
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  # 与词汇表单词个数相等
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 值为1的个数[滥用单词个数]/总数
    # 分子初始化为1 分母初始化为2 避免概率乘积为0
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    # 遍历所有post
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 每个元素除以该类别中的总词数
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testNB():
    listOposts, listClasses = loadDataSet()
    myVocablist = createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dog']
    thisDOC = array(setOfWords2Vec(myVocablist, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDOC, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDOC = array(setOfWords2Vec(myVocablist, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDOC, p0V, p1V, pAb))


if __name__ == '__main__':
    listOposts, listClasses = loadDataSet()
    # TEST1
    myVocablist = createVocabList(listOposts)
    # print(myVocablist)
    # print(setOfWords2Vec(myVocablist, listOposts[0]))
    # TEST2
    trainMat = []
    # ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
    # for postinDoc in listOposts:
    #     trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
    # print('postingList集合转词向量集合,trainMat: ', trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(p0V, p1V, pAb)
    # TEST3
    testNB()
