# -*- coding: utf-8 -*-
"""
    k均值聚类
"""
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


def test_Kmeans(*data):
    '''
    测试 KMeans 的用法
    函数： sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances="auto", verbose=0)
    参数： n_clusters: 一个整数， 指定分类簇的数量
            init: 字符串， 指定初始均值向量策略，
                    * k-means++ ： 该初始化策略选择的初始均值向量相互之间都距离较远。
                    * random
            n_init: 指定了k均值算法运行的次数，每一次都会选择一组不同的初始化均值向量，最终算法会选择最佳的分类簇来作为最终结果
            max_iter： 指定了单轮k均值算法中，最大的迭代次数，
            tol: 指定收敛的阈值
    属性： cluster_centers_：给出了每个样本
     '''
    X, labels_true = data
    clst = cluster.KMeans()
    clst.fit(X)
    predicted_labels = clst.predict(X)
    print("ARI:%s" % adjusted_rand_score(labels_true, predicted_labels))
    print("Sum center distance %s" % clst.inertia_)

def test_Kmeans_nclusters(*data):
    '''
    测试 KMeans 的聚类结果随 n_clusters 参数的影响
    '''
    X,labels_true=data
    nums=range(1,50)
    ARIs=[]
    Distances=[]
    for num in nums:
        clst=cluster.KMeans(n_clusters=num)
        clst.fit(X)
        predicted_labels=clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true,predicted_labels))
        Distances.append(clst.inertia_)

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs,marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances,marker='o')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")
    fig.suptitle("KMeans")
    plt.show()

def test_Kmeans_n_init(*data):
    '''
    测试 KMeans 的聚类结果随 n_init 和 init  参数的影响
    '''
    X,labels_true=data
    nums=range(1,50)
    ## 绘图
    fig=plt.figure()

    ARIs_k=[]
    Distances_k=[]
    ARIs_r=[]
    Distances_r=[]
    for num in nums:
            clst=cluster.KMeans(n_init=num,init='k-means++')
            clst.fit(X)
            predicted_labels=clst.predict(X)
            ARIs_k.append(adjusted_rand_score(labels_true,predicted_labels))
            Distances_k.append(clst.inertia_)

            clst=cluster.KMeans(n_init=num,init='random')
            clst.fit(X)
            predicted_labels=clst.predict(X)
            ARIs_r.append(adjusted_rand_score(labels_true,predicted_labels))
            Distances_r.append(clst.inertia_)

    ax=fig.add_subplot(1,2,1)
    ax.plot(nums,ARIs_k,marker="+",label="k-means++")
    ax.plot(nums,ARIs_r,marker="+",label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    ax=fig.add_subplot(1,2,2)
    ax.plot(nums,Distances_k,marker='o',label="k-means++")
    ax.plot(nums,Distances_r,marker='o',label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("inertia_")
    ax.legend(loc='best')

    fig.suptitle("KMeans")
    plt.show()
