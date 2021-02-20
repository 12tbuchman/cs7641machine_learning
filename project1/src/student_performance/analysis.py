from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import recall_score

import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_dataset, TEST_SIZE, RANDOM_STATE

print("Analysis")
load_dataset(use_math=False, setup='A', passing='C')

###########################
fig = 1
###########################

if fig == 1:
    tree_testing_scores = []
    tree_training_times = []
    tree_testing_times = []
    adaboost_testing_scores = []
    adaboost_training_times = []
    adaboost_testing_times = []
    knn_testing_scores = []
    knn_training_times = []
    knn_testing_times = []
    mlp_testing_scores = []
    mlp_training_times = []
    mlp_testing_times = []
    svc_testing_scores = []
    svc_training_times = []
    svc_testing_times = []
    # Tree
    tree_criterion = 'gini'
    tree_ccp_alpha = 0.020851489481254898
    min_samples_leaf = 1
    min_samples_split = 2
    # Adaboost
    boost_criterion = 'gini'
    boost_ccp_alpha = 0.020851489481254898
    n_estimators = 40
    learning_rate = 0.2
    # KNN
    neighbors = 15
    algorithm = 'kd_tree'
    leaf_size = 15
    p = 1
    # MLP
    solver = 'lbfgs'
    activation = 'tanh'
    alpha = 0.00008
    mlp_tol = 0.00018
    layers = (30, 2)
    # SVC
    kernel = 'poly'
    degree = 2
    C = 1.0
    coef0 = 0.15
    svc_tol = 0.0005

    data_all, grades_all = sample_dataset('p')
    sample_size = [i * 50 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        num_folds = 3
        # Tree
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=tree_criterion, ccp_alpha=tree_ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        start = time.time()
        tree.fit(data_train, labels_train)
        stop = time.time()
        tree_training_times.append(stop - start)
        start = time.time()
        result = tree.score(data_test, labels_test)
        stop = time.time()
        tree_testing_times.append(stop - start)
        tree_testing_scores.append(result)
        # Boost
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=boost_criterion, ccp_alpha=boost_ccp_alpha)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        start = time.time()
        boost.fit(data_train, labels_train)
        stop = time.time()
        adaboost_training_times.append(stop - start)
        start = time.time()
        result = boost.score(data_test, labels_test)
        stop = time.time()
        adaboost_testing_times.append(stop - start)
        adaboost_testing_scores.append(result)
        # KNN
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p)
        start = time.time()
        knn.fit(data_train, labels_train)
        stop = time.time()
        knn_training_times.append(stop - start)
        start = time.time()
        result = knn.score(data_test, labels_test)
        stop = time.time()
        knn_testing_times.append(stop - start)
        knn_testing_scores.append(result)
        # MLP
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=2000,
                           alpha=alpha, tol=mlp_tol, hidden_layer_sizes=layers)
        start = time.time()
        nn.fit(data_train, labels_train)
        stop = time.time()
        mlp_training_times.append(stop - start)
        start = time.time()
        result = nn.score(data_test, labels_test)
        stop = time.time()
        mlp_testing_times.append(stop - start)
        mlp_testing_scores.append(result)
        # SVC
        svc = SVC(random_state=RANDOM_STATE, kernel=kernel, degree=degree, C=C, coef0=coef0, tol=svc_tol)
        start = time.time()
        svc.fit(data_train, labels_train)
        stop = time.time()
        svc_training_times.append(stop - start)
        start = time.time()
        result = svc.score(data_test, labels_test)
        stop = time.time()
        svc_testing_times.append(stop - start)
        svc_testing_scores.append(result)
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.6.1 Accuracy vs Sample Size for Various Algorithms")
    ax.plot(sample_size, tree_testing_scores, marker='s', label='Decision Tree')
    ax.plot(sample_size, adaboost_testing_scores, marker='s', label='Adaboost')
    ax.plot(sample_size, knn_testing_scores, marker='s', label='KNN')
    ax.plot(sample_size, mlp_testing_scores, marker='s', label='MLP')
    ax.plot(sample_size, svc_testing_scores, marker='s', label='SVC')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Fig 1.6.2 Training Time vs Sample Size for Various Algorithms")
    ax.plot(sample_size, tree_training_times, marker='s', label='Decision Tree')
    ax.plot(sample_size, adaboost_training_times, marker='s', label='Adaboost')
    ax.plot(sample_size, knn_training_times, marker='s', label='KNN')
    ax.plot(sample_size, mlp_training_times, marker='s', label='MLP')
    ax.plot(sample_size, svc_training_times, marker='s', label='SVC')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Testing Time (seconds)")
    ax.set_title("Fig 1.6.3 Testing Time vs Sample Size for Various Algorithms")
    ax.plot(sample_size, tree_testing_scores, marker='s', label='Decision Tree')
    ax.plot(sample_size, adaboost_testing_scores, marker='s', label='Adaboost')
    ax.plot(sample_size, knn_testing_scores, marker='s', label='KNN')
    ax.plot(sample_size, mlp_testing_times, marker='s', label='MLP')
    ax.plot(sample_size, svc_testing_times, marker='s', label='SVC')
    plt.legend()
    plt.show()
