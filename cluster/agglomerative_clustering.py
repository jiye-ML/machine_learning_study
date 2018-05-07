# -*- coding: utf-8 -*-
"""
    层次聚类
"""
from sklearn import  cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

def test_AgglomerativeClustering(*data):
    '''
    测试 AgglomerativeClustering 的用法
    sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='euvlidean', memory=Memory(cachedir=NOne), connectivety=None, n_components=None, compute_full_tree='auto', linkage='ward', pooling_func=<funcition mean>)
    参数：
        n_clusters: 指定分类簇的数量
        connectivity: 用于指定连接矩阵，给出了每个样本的可连接样本
        affinity: 用于计算距离
    '''
    X,labels_true=data
    clst=cluster.AgglomerativeClustering()
    predicted_labels=clst.fit_predict(X)
    print("ARI:%s"% adjusted_rand_score(labels_true,predicted_labels))

def test_AgglomerativeClustering_nclusters(*data):
    '''
    测试 AgglomerativeClustering 的聚类结果随 n_clusters 参数的影响

    :param data:  可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    ARIs=[]
    for num in nums:
        clst=cluster.AgglomerativeClustering(n_clusters=num)
        predicted_labels=clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    fig.suptitle("AgglomerativeClustering")
    plt.show()

def test_AgglomerativeClustering_linkage(*data):
    '''
    测试 AgglomerativeClustering 的聚类结果随链接方式的影响

    :param data:  可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X,labels_true=data
    nums=range(1,50)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    linkages=['ward','complete','average']
    markers="+o*"
    for i, linkage in enumerate(linkages):
        ARIs=[]
        for num in nums:
            clst=cluster.AgglomerativeClustering(n_clusters=num,linkage=linkage)
            predicted_labels=clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        ax.plot(nums,ARIs,marker=markers[i],label="linkage:%s"%linkage)

    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()