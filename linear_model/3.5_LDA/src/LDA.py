# -*- coding: utf-8 -*
from _operator import inv
import numpy as np
import matplotlib.pyplot as plt
from self_def import GetProjectivePoint_2D

# data
data_file = open('../data/watermelon_3a.csv')
dataset = np.loadtxt(data_file, delimiter=",")

# separate the data from the target attributes
X = dataset[:,1:3]
y = dataset[:,3]

# 绘制原始数据
f1 = plt.figure(1, figsize=(15, 10))
plt.subplot(221)
plt.title('watermelon_3a')  
plt.xlabel('density')  
plt.ylabel('ratio_sugar')  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')
plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
plt.legend(loc = 'upper right')

# 1. 获得均值
u = []  
for i in range(2):
    u.append(np.mean(X[y==i], axis=0))

# 2. 计算类内散度矩阵
m,n = np.shape(X)
Sw = np.zeros((n,n))
for i in range(m):
    x_tmp = X[i].reshape(n,1)  # row -> cloumn vector
    if y[i] == 0: u_tmp = u[0].reshape(n,1)
    if y[i] == 1: u_tmp = u[1].reshape(n,1)
    Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )
# 3.1 求类内矩阵的逆
Sw = np.mat(Sw)
# 分解
U, sigma, V= np.linalg.svd(Sw)
Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T

# 3. 计算参数 w
w = np.dot(Sw_inv, (u[0] - u[1]).reshape(n,1))
print("权值参数")
print(w)

# 4. 画图
plt.subplot(222)
plt.xlim( -0.2, 1 )
plt.ylim( -0.5, 0.7 )

p0_x0 = -X[:, 0].max()
p0_x1 = (w[1,0] / w[0,0]) * p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = (w[1,0] / w[0,0]) * p1_x0

plt.title('watermelon_3a - LDA')  
plt.xlabel('density')  
plt.ylabel('ratio_sugar')  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=10, label = 'bad')
plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=10, label = 'good')
plt.legend(loc = 'upper right')  

plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])
# 绘制投影
m,n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D( [X[i,0], X[i,1]], [w[1,0] / w[0,0] , 0] ) 
    if y[i] == 0: 
        plt.plot(x_p[0], x_p[1], 'ko', markersize = 5)
    if y[i] == 1: 
        plt.plot(x_p[0], x_p[1], 'go', markersize = 5)   
    plt.plot([x_p[0], X[i,0]], [x_p[1], X[i,1] ], 'c--', linewidth = 0.3)


'''
implementation of LDA again after delete outlier (X[14])
'''
# 1. 删除 (X[14])
X = np.delete(X, 14, 0)
y = np.delete(y, 14, 0)

u = []  
for i in range(2):
    u.append(np.mean(X[y==i], axis=0))
# 2
m,n = np.shape(X)
Sw = np.zeros((n,n))
for i in range(m):
    x_tmp = X[i].reshape(n,1)
    if y[i] == 0: u_tmp = u[0].reshape(n,1)
    if y[i] == 1: u_tmp = u[1].reshape(n,1)
    Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )
Sw = np.mat(Sw)
U, sigma, V= np.linalg.svd(Sw)
Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T

# 3
w = np.dot( Sw_inv, (u[0] - u[1]).reshape(n,1) )
print("删除 x[14] 后， 权值")
print(w)



# 画图
plt.subplot(223)
plt.xlim( -0.2, 1 )
plt.ylim( -0.5, 0.7 )

p0_x0 = -X[:, 0].max()
p0_x1 = ( w[1,0] / w[0,0] ) * p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = ( w[1,0] / w[0,0] ) * p1_x0

plt.title('watermelon_3a - LDA')  
plt.xlabel('density')  
plt.ylabel('ratio_sugar')  
plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=10, label = 'bad')
plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=10, label = 'good')
plt.legend(loc = 'upper right')  

plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])

# 绘制投影
m,n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D( [X[i,0], X[i,1]], [w[1,0] / w[0,0] , 0] ) 
    if y[i] == 0: 
        plt.plot(x_p[0], x_p[1], 'ko', markersize = 5)
    if y[i] == 1: 
        plt.plot(x_p[0], x_p[1], 'go', markersize = 5)   
    plt.plot([ x_p[0], X[i,0]], [x_p[1], X[i,1] ], 'c--', linewidth = 0.3)

plt.show()

