# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets,model_selection

def load_classification_data():
    '''
    加载分类模型使用的数据集。

    :return: 一个元组，依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    '''
    digits=datasets.load_digits() # 使用 scikit-learn 自带的手写识别数据集 Digit Dataset
    return model_selection.train_test_split(digits.data, digits.target,test_size=0.25, random_state=0)


def test_KNeighborsClassifier(*data):
    '''
    功能：k近邻分类算法
    函数： sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p = 2, metric='minkowski'
                metric_params=None, n_jobs=1, **kwags)
    参数： n_neighbors：一个整数，k值
            weights： 一个字符串或者可调用对象，制定投票权重类型，
                    uniform: 本节点所有邻居节点投票权重相等
                    distance: 本节点所有邻居节点投票权重与距离成反比，
                    [callalbe]: 一个可调用对象，出入距离的数组，
            algorithm: 一个字符串， 指定计算最近距离的算法：
                    ball_tree: 使用 BallTree算法
                    kd_tree: 使用kdTree算法
                    brute： 使用暴力搜索算法，
                    auto： 自动决定最合适的算法
    '''
    X_train,X_test,y_train,y_test=data
    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    print("Training Score:%f"%clf.score(X_train,y_train))
    print("Testing Score:%f"%clf.score(X_test,y_test))

def test_KNeighborsClassifier_k_w(*data):
    '''
    测试 KNeighborsClassifier 中 n_neighbors 和 weights 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Ks=np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')
    weights=['uniform','distance']

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ### 绘制不同 weights 下， 预测得分随 n_neighbors 的曲线
    for weight in weights:
        training_scores=[]
        testing_scores=[]
        for K in Ks:
            clf=neighbors.KNeighborsClassifier(weights=weight,n_neighbors=K)
            clf.fit(X_train,y_train)
            testing_scores.append(clf.score(X_test,y_test))
            training_scores.append(clf.score(X_train,y_train))
        ax.plot(Ks,testing_scores,label="testing score:weight=%s"%weight)
        ax.plot(Ks,training_scores,label="training score:weight=%s"%weight)
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()

def test_KNeighborsClassifier_k_p(*data):
    '''
    测试 KNeighborsClassifier 中 n_neighbors 和 p 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    Ks=np.linspace(1,y_train.size,endpoint=False,dtype='int')
    Ps=[1,2,10]

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ### 绘制不同 p 下， 预测得分随 n_neighbors 的曲线
    for P in Ps:
        training_scores=[]
        testing_scores=[]
        for K in Ks:
            clf=neighbors.KNeighborsClassifier(p=P,n_neighbors=K)
            clf.fit(X_train,y_train)
            testing_scores.append(clf.score(X_test,y_test))
            training_scores.append(clf.score(X_train,y_train))
        ax.plot(Ks,testing_scores,label="testing score:p=%d"%P)
        ax.plot(Ks,training_scores,label="training score:p=%d"%P)
    ax.legend(loc='best')
    ax.set_xlabel("K")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.05)
    ax.set_title("KNeighborsClassifier")
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test = load_classification_data() # 获取分类模型的数据集
    #test_KNeighborsClassifier(X_train,X_test,y_train,y_test) # 调用 test_KNeighborsClassifier
    # test_KNeighborsClassifier_k_w(X_train,X_test,y_train,y_test)# 调用 test_KNeighborsClassifier_k_w
    test_KNeighborsClassifier_k_p(X_train,X_test,y_train,y_test)# 调用 test_KNeighborsClassifier_k_p