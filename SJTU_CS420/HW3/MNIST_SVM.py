# 导入必备的包
import numpy as np
import struct
import matplotlib.pyplot as plt
import os
##加载svm模型
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
###用于做数据预处理
from sklearn import preprocessing
import time

# 加载数据的路径
path = 'data/mnist/'


def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def linear_SVM():
    train_images, train_labels = load_mnist_train(path)
    test_images, test_labels = load_mnist_test(path)

    X = preprocessing.StandardScaler().fit_transform(train_images)
    X_train = X[0:60000]
    y_train = train_labels[0:60000]

    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    model_svc = svm.LinearSVC()
    # model_svc = svm.SVC()
    model_svc.fit(X_train, y_train)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

    ##显示前30个样本的真实标签和预测值，用图显示
    x = preprocessing.StandardScaler().fit_transform(test_images)
    x_test = x[0:10000]
    y_pred = test_labels[0:10000]
    print(model_svc.score(x_test, y_pred))
    y = model_svc.predict(x)

    fig1 = plt.figure(figsize=(8, 8))
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(100):
        ax = fig1.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(np.reshape(test_images[i], [28, 28]), cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 2, "pred:" + str(y[i]), color='red')
        # ax.text(0,32,"real:"+str(test_labels[i]),color='blue')
    plt.show()


def kernel_SVM():
    train_images, train_labels = load_mnist_train(path)
    test_images, test_labels = load_mnist_test(path)
    print(train_labels[0])
    print(test_labels[0])
    ss = preprocessing.StandardScaler()

    X = ss.fit_transform(train_images)

    # X=preprocessing.StandardScaler().fit_transform(train_images)
    X_train = X[0:10000]
    y_train = train_labels[0:10000]
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print(y_train)

    ##显示前30个样本的真实标签和预测值，用图显示
    x = preprocessing.StandardScaler().fit_transform(test_images)
    x_test = x[0:10000]
    y_test = test_labels[0:10000]
    # print(model_svc.score(x_test, y_test))
    # y = model_svc.predict(x)
    scores = ['precision', 'recall']
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.1, 10],
                         'C': [1, 10, 100, 1000]}]
    # print("# Tuning hyper-parameters for %s" % score)
    print()
    # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    clf = GridSearchCV(SVC(class_weight='balanced'), param_grid={"C": [0.1, 1, 100, 10], "gamma": [0.01, 10, 0.1]},
                       cv=4)
    # 用训练集训练这个学习器 clf
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()

    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf.best_params_)

    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y = y_test, clf.predict(x)

    # 打印在测试集上的预测结果与真实值的分数
    print(classification_report(y_true, y))

    print()

    # fig1=plt.figure(figsize=(8,8))
    # fig1.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    # for i in range(100):
    #     ax=fig1.add_subplot(10,10,i+1,xticks=[],yticks=[])
    #     ax.imshow(np.reshape(test_images[i], [28,28]),cmap=plt.cm.binary,interpolation='nearest')
    #     ax.text(0,2,"pred:"+str(y[i]),color='red')
    #     ax.text(0,32,"real:"+str(test_labels[i]),color='blue')
    # plt.show()

if __name__ == '__main__':
    kernel_SVM()