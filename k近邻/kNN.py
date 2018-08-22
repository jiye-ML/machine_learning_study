'''
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

'''
from numpy import *
import operator


'''
分类器
'''

def classify0(inX, dataSet, labels, k):
    '''
    对数据分类
    :param inX: 用于分类的输入
    :param dataSet: 输入的训练样本集合
    :param labels: 标签
    :param k: 选择最近的邻居的数目
    :return: 
    '''
    dataSetSize = dataSet.shape[0]
    # 计算距离：计算输入到数据集中每个点的欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序: 通过每个类别的值进行排序，找到最大的类别的数据来标签输入的数据
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
准备数据
'''
# 读取
def file2matrix(filename):
    '''
    解析文件，获得需要的数据特征和标签
    :param filename: 
    :return:  [特征，标签]
    '''
    # 得到文件行数
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    # 创建返回的numpy矩阵
    returnMat = zeros((numberOfLines, 3))
    # 解析文件数据到列表
    classLabelVector = []
    fr = open(filename)
    for index, line in enumerate(fr.readlines()):
        listFromLine = line.strip().split('\t')
        # 特征
        returnMat[index, :] = listFromLine[0:3]
        # 标签
        classLabelVector.append(int(listFromLine[-1]))
        pass
    return returnMat, classLabelVector

# 归一化: new_value = (old_value - min_value) /  (max_value - min_value)
def autoNorm(dataSet):
    minVals, maxVals = dataSet.min(0), dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


'''
分析数据
'''

def show_data():
    import matplotlib.pyplot as plt
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 不同颜色不同尺寸绘制数据
    ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], 15.0 * array(dating_labels), 15.0 * array(dating_labels))
    plt.show()
    pass



'''
测试算法
'''

def datingClassTest():
    # 测试集数据量
    hoRatio = 0.50
    # 读入数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 数据处理
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 测试
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0

    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)
    pass


'''
使用算法
'''



if __name__ == '__main__':

    datingClassTest()


    pass