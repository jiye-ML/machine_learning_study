'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
import numpy as np

# 创建数据
def loadSimpData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

# 从文件中加载数据，但会特征和标签
def loadDataSet(fileName):
    # 特征树数目
    numFeat = len(open(fileName).readline().split('\t'))
    # 数据
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # 一条数据
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 在某一个轴上根据阈值将数据分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''

    :param dataMatrix:  特征
    :param dimen: 分割维度
    :param threshVal: 分割阈值
    :param threshIneq: 取值 ['lt', 'gt']，如果为'lt'，则在分割维度上小于等于阈值标签定位 -1.0
    :return:
    '''
    # 预测标签
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    
# 单层决策树
def buildStump(dataArr,classLabels,D):
    '''
    :param dataArr: 输入数据
    :param classLabels: 类别标签
    :param D: 每个数据的权重
    :return: 一颗单层的决策树
    注意：这里的数据分布的改变体现在权值，在对于每个分类中，是按照分类器的阈值进行分类的，只有在计算损失的时候才会利用到分布，
    然后根据分布来绝对这次分类器的效果，作出判断
    '''
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    # 每个特征上划分的数量
    numSteps = 10.0
    # 最好的划分维度以及划分的阈值和划分方式
    bestStump = {}
    # 最优的预测结果
    bestClasEst = np.mat(np.zeros((m,1)))
    # 误差
    minError = np.inf
    # 在所有特征上划分
    for i in range(n):
        # 划分特征的取值范围
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        # 每次划分的步长
        stepSize = (rangeMax-rangeMin)/numSteps
        # loop over all range in current dimension
        for j in range(-1, int(numSteps) + 1):
            # go over less than and greater than
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 数据在本次划分轴和阈值下的预测结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 预测错误的数量
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                # 当前分布下的加权误差
                weightedError = D.T * errArr
                # 得到最优分割结果
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''
    利用单层决策树为基学习器，构建adaboost
    :param dataArr: 
    :param classLabels: 
    :param numIt: 迭代次数
    :return: adaboost
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]
    # 初始化每个样本的权值矩阵
    D = np.mat(np.ones((m, 1)) / m)
    # 集成器分类的结果
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        # 基学习器
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # 基学习的权重
        alpha = float(0.5 * np.log((1.0-error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        # 加权每个样本
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr

# ada分类器
def adaClassify(datToClass, classifierArr):
    '''

    :param datToClass: 需要分类的数据
    :param classifierArr: 得到的分类器
    :return: ada分类器的加权分类结果
    '''
    dataMatrix = np.mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 迭代所有的分类器
    for i in range(len(classifierArr)):
        # 分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        # 加权结果，
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

# 绘制ROC曲线
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print( "the Area Under the Curve is: ", ySum * xStep)


if __name__ == '__main__':

    # 训练数据
    data_arr, label_arr = loadDataSet("horseColicTraining2.txt")
    # 分类器
    classifier_array = adaBoostTrainDS(data_arr, label_arr, 10)
    # 测试数据
    test_arr, test_label_arr = loadDataSet('horseColicTest2.txt')
    # 预测结果
    prediction_10 = adaClassify(test_arr, classifier_array)
    # 前67个数据的误差
    err_arr = np.mat(np.ones((67, 1)))
    print(err_arr[prediction_10 != np.mat(test_label_arr).T].sum())
    pass

