from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression

import numpy as np
import matplotlib.pyplot as plt

interval = 5
max_iter = 200

def pca(x):
    model = PCA(n_components=9)
    data = model.fit_transform(x)
    print(model.explained_variance_ratio_.sum())
    return data
    #return x

def select(x, y):
    selector = SelectKBest(score_func=f_classif, k=10)
    real_features = selector.fit_transform(x, y)
    #print(selector.scores_)

    return real_features

################################### SVM Part #####################################

def svm(data_name):
    # load data
    x_tr = np.load('data/' + data_name + '_feature.npy')
    y_tr = np.load('data/' + data_name + '_target.npy')
    x_t = np.load('data/' + data_name + '.t_feature.npy')
    y_t = np.load('data/' + data_name + '.t_target.npy')
    # if data_name == 'madelon':
    scaler = StandardScaler()
    scaler.fit(x_tr)
    x_tr = scaler.transform(x_tr)
    scaler.fit(x_t)
    x_t = scaler.transform(x_t)

    res_tr = []
    res_t = []

    # training stage
    for i in range(interval, max_iter + 1, interval):
        model = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=2, gamma='auto', kernel='rbf',
                max_iter=i, probability=False, shrinking=True,
                tol=0.001, verbose=False)
        model.fit(x_tr, y_tr)

        res_tr.append(round(model.score(x_tr, y_tr), 3))
        res_t.append(round(model.score(x_t, y_t), 3))
        # print(model.score(x_tr, y_tr))
        # print(model.score(x_t, y_t))
        # print(model.predict(x_tr))
    print('train: ', res_tr)
    print('test: ', res_t)

def plot_s_kernel_splice():
    # model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #             decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    #             max_iter=i, probability=False, shrinking=True,
    #             tol=0.001, verbose=False, random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_linear = [0.556, 0.642, 0.644, 0.61, 0.725, 0.745, 0.657, 0.76, 0.769, 0.728, 0.76, 0.736, 0.674, 0.7, 0.734, 0.792]
    y_tr_poly3 =  [0.642, 0.664, 0.658, 0.638, 0.651, 0.661, 0.664, 0.693, 0.68, 0.644, 0.646, 0.647, 0.645, 0.643, 0.634, 0.628]
    y_tr_poly2 = [0.509, 0.49, 0.538, 0.548, 0.555, 0.555, 0.573, 0.561, 0.553, 0.544, 0.573, 0.567, 0.577, 0.586, 0.597, 0.59]
    y_tr_rbf = [0.578, 0.698, 0.764, 0.832, 0.863, 0.862, 0.866, 0.867, 0.898, 0.903, 0.899, 0.896, 0.9, 0.899, 0.899, 0.899]
    y_tr_sigmoid = [0.389, 0.441, 0.346, 0.555, 0.525, 0.55, 0.556, 0.616, 0.657, 0.659, 0.672, 0.679, 0.683, 0.683, 0.683, 0.683]
    y_t_linear = [0.536, 0.588, 0.622, 0.588, 0.692, 0.738, 0.69, 0.714, 0.732, 0.706, 0.726, 0.672, 0.66, 0.666, 0.732, 0.749]
    y_t_poly3 = [0.632, 0.656, 0.652, 0.644, 0.652, 0.66, 0.666, 0.688, 0.674, 0.657, 0.653, 0.654, 0.653, 0.647, 0.644, 0.638]
    y_t_poly2 = [0.509, 0.49, 0.538, 0.548, 0.555, 0.555, 0.573, 0.561, 0.553, 0.544, 0.573, 0.567, 0.577, 0.586, 0.597, 0.59]
    y_t_rbf = [0.626, 0.699, 0.766, 0.798, 0.818, 0.821, 0.83, 0.824, 0.846, 0.846, 0.852, 0.842, 0.848, 0.846, 0.847, 0.848]
    y_t_sigmoid = [0.334, 0.438, 0.348, 0.498, 0.474, 0.5, 0.505, 0.562, 0.599, 0.612, 0.624, 0.636, 0.629, 0.628, 0.628, 0.628]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_linear, color='#90EE90', linewidth=1.7, label='linear')
    ax.plot(x, y_tr_poly3, color='#ffa07a', linewidth=1.7, label='3-polynomial')
    ax.plot(x, y_tr_poly2, color='#FA8072', linewidth=1.7, label='2-polynomial')
    ax.plot(x, y_tr_rbf, color='#9999ff', linewidth=1.7, label='rbf')
    ax.plot(x, y_tr_sigmoid, color='#F0E68C', linewidth=1.7, label='sigmoid')
    ax.scatter(x, y_tr_linear, s=13, c='#90EE90')
    ax.scatter(x, y_tr_poly3, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_poly2, s=13, c='#FA8072')
    ax.scatter(x, y_tr_rbf, s=13, c='#9999ff')
    ax.scatter(x, y_tr_sigmoid, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('img/svm_kernel_splice_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_linear, color='#90EE90', linewidth=1.7, label='linear')
    ax.plot(x, y_t_poly3, color='#ffa07a', linewidth=1.7, label='3-polynomial')
    ax.plot(x, y_t_poly2, color='#FA8072', linewidth=1.7, label='2-polynomial')
    ax.plot(x, y_t_rbf, color='#9999ff', linewidth=1.7, label='rbf')
    ax.plot(x, y_t_sigmoid, color='#F0E68C', linewidth=1.7, label='sigmoid')
    ax.scatter(x, y_t_linear, s=13, c='#90EE90')
    ax.scatter(x, y_t_poly3, s=13, c='#ffa07a')
    ax.scatter(x, y_t_poly2, s=13, c='#FA8072')
    ax.scatter(x, y_t_rbf, s=13, c='#9999ff')
    ax.scatter(x, y_t_sigmoid, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('img/svm_kernel_splice_t')
    plt.show()

def plot_s_kernel_sat():
    # model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #             decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    #             max_iter=i, probability=False, shrinking=True,
    #             tol=0.001, verbose=False, random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_linear = [0.633, 0.726, 0.801, 0.655, 0.665, 0.808, 0.833, 0.826, 0.853, 0.857, 0.857, 0.851, 0.858, 0.862, 0.872, 0.868]
    y_tr_poly3 =  [0.344, 0.403, 0.38, 0.437, 0.49, 0.532, 0.514, 0.64, 0.655, 0.617, 0.594, 0.773, 0.836, 0.874, 0.874, 0.872]
    y_tr_poly2 = [0.352, 0.364, 0.403, 0.375, 0.6, 0.459, 0.637, 0.661, 0.787, 0.822, 0.834, 0.841, 0.852, 0.856, 0.856, 0.856]
    y_tr_rbf =  [0.688, 0.685, 0.676, 0.715, 0.582, 0.624, 0.783, 0.823, 0.826, 0.828, 0.825, 0.855, 0.854, 0.856, 0.856, 0.854]
    y_tr_sigmoid =[0.657, 0.706, 0.647, 0.681, 0.679, 0.648, 0.683, 0.776, 0.78, 0.798, 0.81, 0.807, 0.823, 0.823, 0.824, 0.823]
    y_t_linear = [0.56, 0.673, 0.776, 0.662, 0.684, 0.78, 0.796, 0.796, 0.806, 0.825, 0.819, 0.812, 0.826, 0.816, 0.836, 0.821]
    y_t_poly3 =  [0.39, 0.44, 0.438, 0.488, 0.532, 0.574, 0.546, 0.635, 0.652, 0.612, 0.635, 0.764, 0.792, 0.826, 0.82, 0.818]
    y_t_poly2 = [0.352, 0.364, 0.403, 0.375, 0.6, 0.459, 0.637, 0.661, 0.787, 0.822, 0.834, 0.841, 0.852, 0.856, 0.856, 0.856]
    y_t_rbf =  [0.664, 0.682, 0.656, 0.707, 0.608, 0.63, 0.764, 0.795, 0.798, 0.804, 0.797, 0.818, 0.828, 0.828, 0.828, 0.828]
    y_t_sigmoid = [0.657, 0.706, 0.647, 0.681, 0.679, 0.648, 0.683, 0.776, 0.78, 0.798, 0.81, 0.807, 0.823, 0.823, 0.824, 0.823]


    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_linear, color='#90EE90', linewidth=1.7, label='no kernel')
    ax.plot(x, y_tr_poly3, color='#ffa07a', linewidth=1.7, label='3-polynomial')
    ax.plot(x, y_tr_poly2, color='#FA8072', linewidth=1.7, label='2-polynomial')
    ax.plot(x, y_tr_rbf, color='#9999ff', linewidth=1.7, label='rbf')
    ax.plot(x, y_tr_sigmoid, color='#F0E68C', linewidth=1.7, label='sigmoid')
    ax.scatter(x, y_tr_linear, s=13, c='#90EE90')
    ax.scatter(x, y_tr_poly3, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_poly2, s=13, c='#FA8072')
    ax.scatter(x, y_tr_rbf, s=13, c='#9999ff')
    ax.scatter(x, y_tr_sigmoid, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('img/svm_kernel_sat_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_linear, color='#90EE90', linewidth=1.7, label='no kernel')
    ax.plot(x, y_t_poly3, color='#ffa07a', linewidth=1.7, label='3-polynomial')
    ax.plot(x, y_t_poly2, color='#FA8072', linewidth=1.7, label='2-polynomial')
    ax.plot(x, y_t_rbf, color='#9999ff', linewidth=1.7, label='rbf')
    ax.plot(x, y_t_sigmoid, color='#F0E68C', linewidth=1.7, label='sigmoid')
    ax.scatter(x, y_t_linear, s=13, c='#90EE90')
    ax.scatter(x, y_t_poly3, s=13, c='#ffa07a')
    ax.scatter(x, y_t_poly2, s=13, c='#FA8072')
    ax.scatter(x, y_t_rbf, s=13, c='#9999ff')
    ax.scatter(x, y_t_sigmoid, s=13, c='#F0E68C')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('img/svm_kernel_sat_t')
    plt.show()

# poly kernel results: kernel_num (2-10)
#     splice: tr:[0.852, 0.947, 0.983, 0.99, 0.99, 0.993, 0.99, 0.989, 0.983]
#             t:[0.786, 0.857, 0.865, 0.871, 0.88, 0.874, 0.864, 0.867, 0.851]
#     sat: tr:[0.651, 0.672, 0.492, 0.553, 0.475, 0.511, 0.471, 0.524, 0.373]
#          t:[0.635, 0.666, 0.505, 0.554, 0.488, 0.512, 0.477, 0.528, 0.403]

def plot_s_penalty_splice():
    # model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #             max_iter=i, probability=False, shrinking=True,
    #             tol=0.001, verbose=False, random_state=666)
    x = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
    y_tr = [0.823, 0.819, 0.955, 0.968, 0.975, 0.974, 0.984, 1.0, 1.0]
    y_t = [0.747, 0.742, 0.864, 0.88, 0.897, 0.897, 0.9, 0.895, 0.897]

    x_ax = np.arange(9) * 0.9
    total_width, n = 0.8, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.bar(x_ax, y_tr, width=width, facecolor='#483D8B', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#DB7093', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.02, y1, '%.3f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.075, y2, '%.3f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0))
    plt.xlabel('Penalty parameter')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.245)
    plt.legend()
    plt.savefig('img/svm_penalty_splice')
    plt.show()

def plot_s_penalty_sat():
    # model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #             max_iter=i, probability=False, shrinking=True,
    #             tol=0.001, verbose=False, random_state=666)
    x = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
    y_tr = [0.692, 0.72, 0.78, 0.801, 0.815, 0.837, 0.854, 0.879, 0.889]
    y_t = [0.67, 0.701, 0.756, 0.778, 0.79, 0.814, 0.828, 0.85, 0.854]

    x_ax = np.arange(9) * 0.9
    total_width, n = 0.9, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.bar(x_ax, y_tr, width=width, facecolor='#483D8B', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#DB7093', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.04, y1, '%.3f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.075, y2, '%.3f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0))
    plt.xlabel('Penalty parameter')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig('img/svm_penalty_sat')
    plt.show()
################################### SVM Part #####################################




################################### MLP Part #####################################

def mlp(data_name):
    # load data
    x_tr = np.load('data/' + data_name + '_feature.npy')
    y_tr = np.load('data/' + data_name + '_target.npy')
    x_t = np.load('data/' + data_name + '.t_feature.npy')
    y_t = np.load('data/' + data_name + '.t_target.npy')

    # training stage
    res_tr = []
    res_t = []
    for i in range(interval, max_iter + 1, interval):
        model = MLPClassifier(solver='adam', alpha=1e-3,
                    learning_rate_init=0.01, max_iter=i,
                    activation='relu',
                    hidden_layer_sizes=(20,20, 20))
        model.fit(x_tr, y_tr)

        res_tr.append(round(model.score(x_tr, y_tr), 3))
        res_t.append(round(model.score(x_t, y_t), 3))

    print('train: ', res_tr)
    print('test: ', res_t)

def plot_activation_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_relu = [0.838, 0.88, 0.9, 0.945, 0.962, 0.981, 0.99, 0.995, 1.0, 1.0]
    y_tr_logi = [0.799, 0.822, 0.839, 0.844, 0.845, 0.851, 0.854, 0.86, 0.866, 0.876]
    y_tr_tanh = [0.852, 0.885, 0.914, 0.934, 0.951, 0.971, 0.982, 0.988, 0.997, 0.999]
    y_t_relu = [0.832, 0.852, 0.851, 0.871, 0.879, 0.886, 0.895, 0.892, 0.897, 0.896]
    y_t_logi =  [0.828, 0.841, 0.844, 0.85, 0.853, 0.854, 0.858, 0.863, 0.867, 0.872]
    y_t_tanh =  [0.839, 0.856, 0.867, 0.875, 0.885, 0.889, 0.894, 0.892, 0.9, 0.897]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_tr_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_tr_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_tr_relu, s=13, c='#90EE90')
    ax.scatter(x, y_tr_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../img/mlp_activation_splice_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_t_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_t_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_t_relu, s=13, c='#90EE90')
    ax.scatter(x, y_t_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_t_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../img/mlp_activation_splice_t')
    plt.show()

def plot_activation_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = list(range(interval, max_iter + 1, interval))
    y_tr_relu =  [0.851, 0.873, 0.892, 0.9, 0.91, 0.919, 0.926, 0.93, 0.937, 0.943]
    y_tr_logi = [0.771, 0.808, 0.83, 0.846, 0.852, 0.855, 0.858, 0.859, 0.861, 0.861]
    y_tr_tanh =  [0.837, 0.856, 0.865, 0.871, 0.88, 0.886, 0.892, 0.895, 0.902, 0.91]
    y_t_relu =  [0.826, 0.834, 0.854, 0.862, 0.87, 0.872, 0.875, 0.879, 0.878, 0.878]
    y_t_logi =  [0.735, 0.778, 0.804, 0.816, 0.822, 0.826, 0.828, 0.83, 0.83, 0.83]
    y_t_tanh = [0.812, 0.822, 0.827, 0.837, 0.84, 0.846, 0.851, 0.856, 0.862, 0.862]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_tr_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_tr_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_tr_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_tr_relu, s=13, c='#90EE90')
    ax.scatter(x, y_tr_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_tr_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../img/mlp_activation_sat_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_t_relu, color='#90EE90', linewidth=1.7, label='relu')
    ax.plot(x, y_t_logi, color='#ffa07a', linewidth=1.7, label='sigmoid')
    ax.plot(x, y_t_tanh, color='#9999ff', linewidth=1.7, label='tanh')
    ax.scatter(x, y_t_relu, s=13, c='#90EE90')
    ax.scatter(x, y_t_logi, s=13, c='#ffa07a')
    ax.scatter(x, y_t_tanh, s=13, c='#9999ff')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../img/mlp_activation_sat_t')
    plt.show()

def plot_lr_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=300,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = [0.0001, 0.001, 0.005, 0.01, 0.1]
    y_tr = [0.856, 1.0, 1.0, 1.0, 0.517]
    y_t = [0.844, 0.896, 0.894, 0.892, 0.52]

    x_ax = np.arange(5)
    total_width, n = 0.75, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.bar(x_ax, y_tr, width=width, facecolor='#483D8B', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#DB7093', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.02, y1, '%.3f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.05, y2, '%.3f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.0001, 0.001, 0.005, 0.01, 0.1))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../img/mlp_lr_splice')
    plt.show()

def plot_lr_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.01, max_iter=200,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)
    x = [0.0001, 0.001, 0.01, 0.1, 0.5]
    y_tr = [0.85, 0.943, 0.995, 0.935, 0.88]
    y_t = [0.827, 0.878, 0.858, 0.852, 0.811]

    x_ax = np.arange(5)
    total_width, n = 0.9, 2
    width = total_width / n
    x_ax = x_ax - (total_width - width) / 2

    plt.bar(x_ax, y_tr, width=width, facecolor='#483D8B', edgecolor='white', label='Training set')
    plt.bar(x_ax + width, y_t, width=width, facecolor='#DB7093', edgecolor='white', label='Testing set')
    for x, y1, y2 in zip(x_ax, y_tr, y_t):
        plt.text(x - 0.02, y1, '%.3f' % y1, ha='center', va='bottom')
        plt.text(x + width + 0.05, y2, '%.3f' % y2, ha='center', va='bottom')

    ax = plt.gca()
    ax.set_xticks(x_ax + width / 2)
    ax.set_xticklabels((0.0001, 0.001, 0.01, 0.1, 0.5))
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.06)
    plt.legend()
    plt.savefig('../img/mlp_lr_sat')
    plt.show()

def plot_archi_splice():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)

    # 1: (10,)
    # 2: (20,)
    # 3; (10, 10)
    # 4: (20, 20)
    # 5: (10, 10, 10)
    # 6: (20, 20, 20)

    x = list(range(interval, max_iter + 1, interval))
    y_1_tr = [0.864, 0.883, 0.888, 0.9, 0.897, 0.9, 0.91, 0.907, 0.905, 0.919]
    y_2_tr = [0.891, 0.915, 0.926, 0.93, 0.939, 0.945, 0.94, 0.95, 0.94, 0.949]
    y_3_tr = [0.874, 0.901, 0.907, 0.895, 0.919, 0.92, 0.928, 0.93, 0.919, 0.929]
    y_4_tr = [0.904, 0.924, 0.947, 0.952, 0.958, 0.974, 0.971, 0.965, 0.971, 0.973]
    y_5_tr = [0.871, 0.894, 0.914, 0.919, 0.918, 0.923, 0.929, 0.925, 0.916, 0.931]
    y_6_tr = [0.905, 0.942, 0.948, 0.958, 0.954, 0.97, 0.962, 0.96, 0.97, 0.952]
    y_1_t = [0.834, 0.848, 0.842, 0.854, 0.839, 0.848, 0.844, 0.846, 0.836, 0.848]
    y_2_t = [0.857, 0.864, 0.858, 0.864, 0.864, 0.864, 0.853, 0.868, 0.854, 0.862]
    y_3_t = [0.843, 0.854, 0.852, 0.832, 0.848, 0.854, 0.842, 0.848, 0.847, 0.854]
    y_4_t = [0.846, 0.862, 0.854, 0.854, 0.856, 0.86, 0.848, 0.85, 0.852, 0.864]
    y_5_t = [0.832, 0.85, 0.845, 0.846, 0.838, 0.842, 0.844, 0.844, 0.846, 0.846]
    y_6_t = [0.85, 0.868, 0.854, 0.856, 0.856, 0.858, 0.854, 0.856, 0.85, 0.846]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_tr, color='#90EE90', linewidth=1.7, label='(10)')
    ax.plot(x, y_2_tr, color='#ffa07a', linewidth=1.7, label='(20)')
    ax.plot(x, y_3_tr, color='#9999ff', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_4_tr, color='#FA8072', linewidth=1.7, label='(20, 20)')
    ax.plot(x, y_5_tr, color='#F0E68C', linewidth=1.7, label='(10, 10, 10)')
    ax.plot(x, y_6_tr, color='#DB7093', linewidth=1.7, label='(20, 20, 20)')
    ax.scatter(x, y_1_tr, s=13, c='#90EE90')
    ax.scatter(x, y_2_tr, s=13, c='#ffa07a')
    ax.scatter(x, y_3_tr, s=13, c='#9999ff')
    ax.scatter(x, y_4_tr, s=13, c='#FA8072')
    ax.scatter(x, y_5_tr, s=13, c='#F0E68C')
    ax.scatter(x, y_6_tr, s=13, c='#DB7093')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('../img/mlp_archi_splice_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_t, color='#90EE90', linewidth=1.7, label='(10)')
    ax.plot(x, y_2_t, color='#ffa07a', linewidth=1.7, label='(20)')
    ax.plot(x, y_3_t, color='#9999ff', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_4_t, color='#FA8072', linewidth=1.7, label='(20, 20)')
    ax.plot(x, y_5_t, color='#F0E68C', linewidth=1.7, label='(10, 10, 10)')
    ax.plot(x, y_6_t, color='#DB7093', linewidth=1.7, label='(20, 20, 20)')
    ax.scatter(x, y_1_t, s=13, c='#90EE90')
    ax.scatter(x, y_2_t, s=13, c='#ffa07a')
    ax.scatter(x, y_3_t, s=13, c='#9999ff')
    ax.scatter(x, y_4_t, s=13, c='#FA8072')
    ax.scatter(x, y_5_t, s=13, c='#F0E68C')
    ax.scatter(x, y_6_t, s=13, c='#DB7093')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('../img/mlp_archi_splice_t')
    plt.show()

def plot_archi_sat():
    # model = MLPClassifier(solver='adam', alpha=1e-3,
    #                       learning_rate_init=0.001, max_iter=i,
    #                       activation='relu',
    #                       hidden_layer_sizes=(10, 10, 2), random_state=666)

    # 1: (10,)
    # 2: (20,)
    # 3; (10, 10)
    # 4: (20, 20)
    # 5: (10, 10, 10)
    # 6: (20, 20, 20)

    x = list(range(interval, max_iter + 1, interval))
    y_1_tr = [0.864, 0.883, 0.888, 0.9, 0.897, 0.9, 0.91, 0.907, 0.905, 0.919]
    y_2_tr = [0.891, 0.915, 0.926, 0.93, 0.939, 0.945, 0.94, 0.95, 0.94, 0.949]
    y_3_tr = [0.874, 0.901, 0.907, 0.895, 0.919, 0.92, 0.928, 0.93, 0.919, 0.929]
    y_4_tr = [0.904, 0.924, 0.947, 0.952, 0.958, 0.974, 0.971, 0.965, 0.971, 0.973]
    y_5_tr = [0.871, 0.894, 0.914, 0.919, 0.918, 0.923, 0.929, 0.925, 0.916, 0.931]
    y_6_tr = [0.905, 0.942, 0.948, 0.958, 0.954, 0.97, 0.962, 0.96, 0.97, 0.952]
    y_1_t = [0.834, 0.848, 0.842, 0.854, 0.839, 0.848, 0.844, 0.846, 0.836, 0.848]
    y_2_t = [0.857, 0.864, 0.858, 0.864, 0.864, 0.864, 0.853, 0.868, 0.854, 0.862]
    y_3_t = [0.843, 0.854, 0.852, 0.832, 0.848, 0.854, 0.842, 0.848, 0.847, 0.854]
    y_4_t = [0.846, 0.862, 0.854, 0.854, 0.856, 0.86, 0.848, 0.85, 0.852, 0.864]
    y_5_t = [0.832, 0.85, 0.845, 0.846, 0.838, 0.842, 0.844, 0.844, 0.846, 0.846]
    y_6_t = [0.85, 0.868, 0.854, 0.856, 0.856, 0.858, 0.854, 0.856, 0.85, 0.846]

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_tr, color='#90EE90', linewidth=1.7, label='(10)')
    ax.plot(x, y_2_tr, color='#ffa07a', linewidth=1.7, label='(20)')
    ax.plot(x, y_3_tr, color='#9999ff', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_4_tr, color='#FA8072', linewidth=1.7, label='(20, 20)')
    ax.plot(x, y_5_tr, color='#F0E68C', linewidth=1.7, label='(10, 10, 10)')
    ax.plot(x, y_6_tr, color='#DB7093', linewidth=1.7, label='(20, 20, 20)')
    ax.scatter(x, y_1_tr, s=13, c='#90EE90')
    ax.scatter(x, y_2_tr, s=13, c='#ffa07a')
    ax.scatter(x, y_3_tr, s=13, c='#9999ff')
    ax.scatter(x, y_4_tr, s=13, c='#FA8072')
    ax.scatter(x, y_5_tr, s=13, c='#F0E68C')
    ax.scatter(x, y_6_tr, s=13, c='#DB7093')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('../img/mlp_archi_sat_tr')
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(x, y_1_t, color='#90EE90', linewidth=1.7, label='(10)')
    ax.plot(x, y_2_t, color='#ffa07a', linewidth=1.7, label='(20)')
    ax.plot(x, y_3_t, color='#9999ff', linewidth=1.7, label='(10, 10)')
    ax.plot(x, y_4_t, color='#FA8072', linewidth=1.7, label='(20, 20)')
    ax.plot(x, y_5_t, color='#F0E68C', linewidth=1.7, label='(10, 10, 10)')
    ax.plot(x, y_6_t, color='#DB7093', linewidth=1.7, label='(20, 20, 20)')
    ax.scatter(x, y_1_t, s=13, c='#90EE90')
    ax.scatter(x, y_2_t, s=13, c='#ffa07a')
    ax.scatter(x, y_3_t, s=13, c='#9999ff')
    ax.scatter(x, y_4_t, s=13, c='#FA8072')
    ax.scatter(x, y_5_t, s=13, c='#F0E68C')
    ax.scatter(x, y_6_t, s=13, c='#DB7093')
    ax.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig('../img/mlp_archi_sat_t')
    plt.show()

################################### MLP Part #####################################

if __name__ == '__main__':
    # splice satimage.scale

    ## MLP ##
    mlp('satimage.scale')
    #mlp('splice')

    ## SVM ##
    #svm('satimage.scale')
    #svm('splice')

    #plot_s_kernel_sat()
    #plot_s_penalty_splice()
    #plot_s_penalty_sat()
    #plot_activation_sat()
    #plot_lr_splice()
    #plot_lr_sat()
    #plot_archi_sat()