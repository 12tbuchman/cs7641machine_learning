from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_dataset, TEST_SIZE, RANDOM_STATE

print("Support Vector Machines")
load_dataset(use_math=False, setup='A', passing='C')

###########################
fig = 0
###########################

if fig == 0:
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    kernels = ['poly', 'rbf', 'linear']
    for kernel in kernels:
        if kernel == 'poly':
            parameters = {'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                          'degree': [2, 3, 4, 5, 6, 7],
                          'coef0': [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2],
                          'tol': [0.0005, 0.001, 0.0015]}
        else:
            parameters = {'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
                          'tol': [0.0005, 0.001, 0.0015]}
        svm = SVC(random_state=RANDOM_STATE, kernel=kernel)
        search = GridSearchCV(svm, parameters).fit(data_train, labels_train)
        print(kernel)
        print(search.best_params_)
        print(search.best_score_)

"""
poly
{'C': 0.1, 'coef0': 0.0, 'degree': 4, 'tol': 0.0005}
0.9383882783882784
poly
{'C': 0.5, 'tol': 0.0005}
0.9207326007326009
poly
{'C': 0.01, 'tol': 0.0005}
0.9229304029304031
"""

# Figure 1.5.1
if fig == 1:
    kernel = 'poly'
    degree = 4
    C = 0.1
    coef0 = 0.0
    tol = 0.005
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_dataset('p')
    sample_size = [i * 50 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        svm = SVC(random_state=RANDOM_STATE, kernel=kernel, degree=degree, C=C, coef0=coef0, tol=tol)
        num_folds = 3
        cv_results = cross_validate(svm, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        svm.fit(data_train, labels_train)
        testing_scores.append(svm.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.1a Accuracy vs Sample Size for Polynomial Kernel")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.1b Test Scores Accuracy vs Sample Size for Polynomial Kernel")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.5.2
if fig == 2:
    kernel = 'poly'
    degree = 4
    C = 0.1
    coef0 = 0.0
    tol = 0.005
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 5 for i in range(8, 25)]
    for iterations in max_iterations:
        num_folds = 3
        svm = SVC(random_state=RANDOM_STATE, kernel=kernel, degree=degree, C=C, coef0=coef0, tol=tol,
                  max_iter=iterations)
        cv_results = cross_validate(svm, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.2 Accuracy vs Max Iterations for Polynomial Kernel")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.5.3
if fig == 3:
    kernel = 'rbf'
    C = 0.5
    tol = 0.0005
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_dataset('p')
    sample_size = [i * 50 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        svm = SVC(random_state=RANDOM_STATE, kernel=kernel, C=C, tol=tol)
        num_folds = 3
        cv_results = cross_validate(svm, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        svm.fit(data_train, labels_train)
        testing_scores.append(svm.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.3a Accuracy vs Sample Size for RBF Kernel")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.3b Test Scores Accuracy vs Sample Size for RBF Kernel")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.5.4
if fig == 4:
    kernel = 'rbf'
    C = 0.5
    tol = 0.0005
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 5 for i in range(8, 25)]
    for iterations in max_iterations:
        num_folds = 3
        svm = SVC(random_state=RANDOM_STATE, kernel=kernel, C=C, tol=tol, max_iter=iterations)
        cv_results = cross_validate(svm, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.4 Accuracy vs Max Iterations for RBF Kernel")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.5.5
if fig == 5:
    kernel = 'linear'
    C = 0.01
    tol = 0.0005
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_dataset('p')
    sample_size = [i * 50 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        svm = SVC(random_state=RANDOM_STATE, kernel=kernel, C=C, tol=tol)
        num_folds = 3
        cv_results = cross_validate(svm, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        svm.fit(data_train, labels_train)
        testing_scores.append(svm.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.5a Accuracy vs Sample Size for Linear Kernel")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.5b Test Scores Accuracy vs Sample Size for Linear Kernel")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.5.6
if fig == 6:
    kernel = 'linear'
    C = 0.01
    tol = 0.0005
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 5 for i in range(8, 25)]
    for iterations in max_iterations:
        num_folds = 3
        svm = SVC(random_state=RANDOM_STATE, kernel=kernel, C=C, tol=tol, max_iter=iterations)
        cv_results = cross_validate(svm, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.5.6 Accuracy vs Max Iterations for Linear Kernel")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
