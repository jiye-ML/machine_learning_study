from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import email_preprocess

# 加载数据
features_train, features_test, labels_train, labels_test = email_preprocess.preprocess()
# features_train = features_train[: len(features_train) // 100]
# labels_train = labels_train[: len(labels_train) // 100]

########################## SVM #################################
def basic_frame():

    ### 创建SSVM模型用于分类
    clf = SVC(kernel="rbf", C = 10000.)
    #### 训练模型
    clf.fit(features_train, labels_train)

    #### 存储预测结果
    pred = clf.predict(features_test)

    print(sum(pred == 1))
    # 准确率
    # acc = accuracy_score(pred, labels_test)
    # print(acc)

    pass


### 测试c参数对SVM的影响
# def test_c():
#     for c in [10.0, 100., 1000., 10000.]:
#         model = SVC(C = c, kernel="rbf")
#         #### 训练模型
#         model.fit(features_train, labels_train)
#
#         #### 存储预测结果
#         pred = model.predict(features_test)
#
#         # 准确率
#         acc = accuracy_score(pred, labels_test)
#         print("C = {}, acc = {}".format(c, acc))
#     pass



if __name__ == '__main__':

    basic_frame()

    pass