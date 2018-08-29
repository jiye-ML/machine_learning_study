'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = [float(i) for i in curLine]
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    '''
    将数据集在某个特征上根据某个值切分为两个集合
    :param dataSet: 数据集
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return: 
        mat0 小于等于 value 的数据集在左边
        mat1 大于 value 的数据集在右边
    '''

    # # 测试案例
    # print 'dataSet[:, feature]=', dataSet[:, feature]
    # print 'nonzero(dataSet[:, feature] > value)[0]=', nonzero(dataSet[:, feature] > value)[0]
    # print 'nonzero(dataSet[:, feature] <= value)[0]=', nonzero(dataSet[:, feature] <= value)[0]

    mat0 = dataSet[nonzero(dataSet[:, feature] >  value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:]
    return mat0,mat1

# returns the value used for each leaf
def regLeaf(dataSet):
    temp = dataSet[:, -1]
    return mean(dataSet[:, -1])

# 回归误差
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

# 1.用最佳方式切分数据集
# 2.生成相应的叶节点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    用最佳方式切分数据集 和 生成相应的叶节点
    :param dataSet: 加载的原始数据集
    :param leafType: 建立叶子点的函数
    :param errType: 误差计算函数(求总方差)
    :param ops: [容许误差下降值，切分的最少样本数]
    :return: 
        bestIndex feature的index坐标
        bestValue 切分的最优值
    '''
    # ops=(1,4)，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
    # 之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合 size 小于 tolN 时，选择停止继续划分。
    # 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
    tolS = ops[0]
    # 划分最小 size 小于，就不继续划分了
    tolN = ops[1]
    # 如果结果集(最后一列为1个变量)，就返回退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    # 循环处理每一列对应的feature值
    for featIndex in range(n-1):
        # [0]表示这一列的[所有行]，不要[0]就是一个array[[所有行]]
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 对该列进行分组，然后组内的成员的val值进行 二元切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 判断二元切分的方式的元素数量是否符合预期
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
            # 如果划分后误差小于 bestS，则说明找到了新的bestS
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 判断二元切分的方式的元素误差是否符合预期
    if (S - bestS) < tolS: 
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 对整体的成员进行判断，是否符合预期
    # 如果集合的 size 小于 tolN
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    '''
    createTree(获取回归树)
        Description：递归函数：如果构建的是回归树，该模型是一个常数，如果是模型树，其模型师一个线性方程。
    :param dataSet: 加载的原始数据集
    :param leafType: 建立叶子点的函数
    :param errType: 误差计算函数
    :param ops: [容许误差下降值，切分的最少样本数]
    :return: 
        retTree 决策树最后的结果
    '''
    # 选择最好的切分方式： feature索引值，最优切分值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果 splitting 达到一个停止条件，那么返回 val
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    # 大于在右边，小于在左边，分为2个数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归的进行调用，在左右子树中继续递归生成树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# 判断节点是否是一个字典
def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    '''
    从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。对 tree 进行塌陷处理，即返回树平均值。
    :param tree: 
    :return: 
        返回 tree 节点的平均值
    '''
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']) / 2.0

# 检查是否适合合并分枝
def prune(tree, testData):
    '''
    从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
    :param tree:  待剪枝的树
    :param testData: 剪枝所需要的测试数据 testData
    :return: 
        tree -- 剪枝完成的树
    '''
    if shape(testData)[0] == 0:
        return getMean(tree)

    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
    if isTree(tree['right']):
        tree['right'] =  prune(tree['right'], rSet)

    # 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点

    # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
    # 1. 如果正确
    #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
    #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
    # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge: 
            print( "merging")
            return treeMean
        else:
            return tree
    else:
        return tree
    
def regTreeEval(model, inDat):
    return float(model)

'''
模型树
'''

def linearSolve(dataSet): #helper function used in two places
    '''
    将数据集格式化成目标变量Y和自变量X，执行简单的线性回归，得到ws
    :param dataSet: 
    :return: 
        ws -- 执行线性回归的回归系数
        X -- 格式化自变量X
        Y -- 格式化目标变量Y
    '''
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    # X的0列为1，常数项，用于计算平衡误差
    X[:, 1 : n] = dataSet[:, 0 : n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

# 得到模型的ws系数： f(x) = x0 + x1*featrue1+ x3*featrue2 ..
def modelLeaf(dataSet):
    '''
    当数据不再需要切分的时候，生成叶节点的模型。
    :param dataSet: 输入数据集
    :return: 
        调用 linearSolve 函数，返回得到的 回归系数ws
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws

# 计算线性模型的误差值
def modelErr(dataSet):
    '''
    在给定数据集上计算误差。
    :param dataSet: 输入数据集
    :return: 
        调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':

    # 测试： 普通的CRAT树
    # data = mat(loadDataSet("ex00.txt"))
    # tree = createTree(data)
    # print(tree)

    # 测试： 树模型
    data2 = mat(loadDataSet('exp2.txt'))
    tree2 = createTree(data2, modelLeaf, modelErr)

    print(tree2)


