# -*- coding: utf-8 -*


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import self_def


# draw scatter diagram to show the raw data
def show_data():
    f1 = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
    plt.legend(loc='upper right')
    plt.show()
    pass


'''
using sklearn lib for logistic regression
'''
class SklearnLogistic:

    def __init__(self, X, y):
        # data
        # generalization of test and train set
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.5,
                                                                                                random_state=0)
        self.y_pred = None

        # model training
        self.log_model = LogisticRegression()
        pass

    def run(self):
        self.log_model.fit(self.X_train, self.y_train)

        # model validation
        self.y_pred = self.log_model.predict(self.X_test)

        # summarize the fit of the model
        print(metrics.confusion_matrix(self.y_test, self.y_pred))
        print(metrics.classification_report(self.y_test, self.y_pred))

        precision, recall, thresholds = metrics.precision_recall_curve(self.y_test, self.y_pred)

        self.show()

        pass

    # 展示运行结果
    def show(self):
        # show decision boundary in plt
        f2 = plt.figure(2)
        h = 0.001
        x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))

        # here "model" is your model's prediction (classification) function
        z = self.log_model.predict(np.c_[x0.ravel(), x1.ravel()])

        # Put the result into a color plot
        z = z.reshape(x0.shape)
        plt.contourf(x0, x1, z, cmap=plt.cm.Paired)

        # Plot also the training pointsplt.title('watermelon_3a')
        plt.title('watermelon_3a')
        plt.xlabel('density')
        plt.ylabel('ratio_sugar')
        plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
        plt.show()
        pass

    pass



'''
coding to implement logistic regression
'''
class MyLogistic():

    def __init__(self, X, y, model_selection = model_selection):
        # data
        self.X = X
        self.y = y
        self.model_selection = model_selection

        # X_train, X_test, y_train, y_test
        m, n = np.shape(self.X)
        X_ex = np.c_[X, np.ones(m)]  # extend the variable matrix to [x, 1]
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X_ex, self.y, test_size=0.5,
                                                                                                random_state=0)
        pass

    def run(self):

        # using gradDescent to get the optimal parameter beta = [w, b] in page-59
        beta = self_def.gradDscent_2(self.X_train, self.y_train)

        # prediction, beta mapping to the model
        y_pred = self_def.predict(self.X_test, beta)

        m_test = np.shape(self.X_test)[0]
        # calculation of confusion_matrix and prediction accuracy
        cfmat = np.zeros((2, 2))
        for i in range(m_test):
            if y_pred[i] == self.y_test[i] == 0:
                cfmat[0, 0] += 1
            elif y_pred[i] == self.y_test[i] == 1:
                cfmat[1, 1] += 1
            elif y_pred[i] == 0:
                cfmat[1, 0] += 1
            elif y_pred[i] == 1:
                cfmat[0, 1] += 1

        print(cfmat)

        pass

    pass



if __name__ == '__main__':

    dataset = np.loadtxt('../data/watermelon_3a.csv', delimiter=",")

    # separate the data from the target attributes
    X = dataset[:, 1:3]
    y = dataset[:, 3]

    m, n = np.shape(X)

    # using sklearn lib for logistic regression
    # SklearnLogistic(X, y).run()

    MyLogistic(X, y).run()
    pass







