'''
构建决策树

感觉决策树是通用算法， 建立一个模型，可以处理各种数据
'''
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    # 每个元素出现的次数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算信息熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    '''
    :param dataSet: 数据集
    :param axis: 划分轴
    :param value: 需要返回的值
    :return: 返回数据集中在axis轴上的值为value的数据，这些数据已经去除了axis轴
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去除掉轴的值
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的轴对数据集进行划分
def chooseBestFeatureToSplit(dataSet):
    # 最后一列是标签
    numFeatures = len(dataSet[0]) - 1
    # 未划分时的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 通过信息增益，获得最好的划分特征
    bestInfoGain, bestFeature= 0.0, -1
    for i in range(numFeatures):
        # 在i特征上的取值
        uniqueVals = set([example[i] for example in dataSet])
        newEntropy = 0.0
        # 在i特征上每种可能取值进行划分
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息的增益
        infoGain = baseEntropy - newEntropy
        # 保存目前信息增益最大的特征
        if (infoGain > bestInfoGain):
            bestInfoGain, bestFeature = infoGain, i
    return bestFeature

# 计算当前集合出现最多的类别标签
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建决策树
def createTree(dataSet,labels):
    '''
    :param dataSet: 当前的数据集
    :param labels: 标签
    :return: 以当前最优划分轴为类标的决策树
    '''
    classList = [example[-1] for example in dataSet]
    # 类别完全相同，停止划分
    if classList.count(classList[0]) == len(classList): 
        return classList[0]

    # 当所有特征都进行了划分，停止划分，返回出现次数最大的类别标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 递归在每个特征上划分
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree                            

# 分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

# 存储决策树
def storeTree(inputTree, filename):
    import pickle
    with open(filename,'w') as f:
        pickle.dump(inputTree, f)
    pass

# 加载决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
if __name__ == '__main__':

    data, labels = createDataSet()
    tree = createTree(data, labels)

    print(tree)

    pass
