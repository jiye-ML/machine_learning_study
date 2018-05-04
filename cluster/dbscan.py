# -*- coding: utf-8 -*-
"""
    密度聚类
"""
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np


def test_DBSCAN(*data):
    '''
    测试 DBSCAN 的用法
    函数： sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
    参数： eps: 用于确定领域大小
            min_samples: MinPts参数，用于判断核心对象
            metric: 用于计算距离，
            algorithm: 用于计算两点间距离并找出最近邻点，
                * auto ; 有算法自适应决定
                * ball_tree：用ball树来搜索
                * kd_tree
                * brute
            leaf_size: 指定用树算法时， 改参数会影响构建树，搜索最近邻的速度，同事影响存储的内存
    属性：
        core_sample_indices_ 核心样本在原始训练集中的位置
        components_ 核心样本的副本
    '''
    X,labels_true=data
    clst = cluster.DBSCAN()
    predicted_labels = clst.fit_predict(X)
    print("ARI:%s" % adjusted_rand_score(labels_true,predicted_labels))
    print("Core sample num:%d" % len(clst.core_sample_indices_))

def test_DBSCAN_epsilon(*data):
    '''
    测试 DBSCAN 的聚类结果随  eps 参数的影响
    '''
    X,labels_true=data
    epsilons=np.logspace(-1,1.5)
    ARIs=[]
    Core_nums=[]
    for epsilon in epsilons:
        clst=cluster.DBSCAN(eps=epsilon)
        predicted_labels=clst.fit_predict(X)
        ARIs.append( adjusted_rand_score(labels_true,predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(epsilons,ARIs,marker='+')
    ax.set_xscale('log')
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylim(0,1)
    ax.set_ylabel('ARI')

    ax=fig.add_subplot(1,2,2)
    ax.plot(epsilons,Core_nums,marker='o')
    ax.set_xscale('log')
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel('Core_Nums')

    fig.suptitle("DBSCAN")
    plt.show()

def test_DBSCAN_min_samples(*data):
    '''
    测试 DBSCAN 的聚类结果随  min_samples 参数的影响
    '''
    X,labels_true=data
    min_samples=range(1,100)
    ARIs=[]
    Core_nums=[]
    for num in min_samples:
        clst=cluster.DBSCAN(min_samples=num)
        predicted_labels=clst.fit_predict(X)
        ARIs.append( adjusted_rand_score(labels_true,predicted_labels))
        Core_nums.append(len(clst.core_sample_indices_))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(min_samples,ARIs,marker='+')
    ax.set_xlabel( "min_samples")
    ax.set_ylim(0,1)
    ax.set_ylabel('ARI')

    ax=fig.add_subplot(1,2,2)
    ax.plot(min_samples,Core_nums,marker='o')
    ax.set_xlabel( "min_samples")
    ax.set_ylabel('Core_Nums')

    fig.suptitle("DBSCAN")
    plt.show()
