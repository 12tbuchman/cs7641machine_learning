from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import recall_score

import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_unbalanced_dataset, sample_balanced_dataset, TEST_SIZE, RANDOM_STATE

print("Neural Networks")
load_dataset()

###########################
fig = 9
###########################

solver = 'lbfgs'
if fig == 0:
    data, grades = sample_balanced_dataset(samples=600)
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    activation = ['logistic', 'tanh']
    for a in activation:
        parameters = {'alpha': [0.00002 * i for i in range(1, 11)],
                      'tol': [0.00002 * i for i in range(1, 11)],
                      'hidden_layer_sizes': [(10, 1), (20, 1), (30, 1), (40, 1), (45, 1)]
                      }
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=a, max_iter=10000)
        search = GridSearchCV(nn, parameters).fit(data_train, labels_train)
        print(a)
        print(search.best_params_)
        print(search.best_score_)

"""
logistic
{'alpha': 0.00016, 'hidden_layer_sizes': (10, 1), 'tol': 2e-05}
0.5416666666666666
tanh
{'alpha': 0.0001, 'hidden_layer_sizes': (20, 1), 'tol': 6.000000000000001e-05}
0.5166666666666667
"""

if fig == 0.1:
    data, grades = sample_balanced_dataset(samples=300)
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    activation = ['logistic', 'tanh']
    for a in activation:
        parameters = {'alpha': [0.00002 * i for i in range(3, 8)],
                      'tol': [0.00002 * i for i in range(3, 8)],
                      'hidden_layer_sizes': [(10, 2), (30, 2), (50, 2), (70, 2)]
                      }
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=a, max_iter=10000)
        search = GridSearchCV(nn, parameters).fit(data_train, labels_train)
        print(a)
        print(search.best_params_)
        print(search.best_score_)

"""
logistic
{'alpha': 0.00014000000000000001, 'hidden_layer_sizes': (30, 2), 'tol': 0.0001}
0.5
tanh
{'alpha': 0.00012000000000000002, 'hidden_layer_sizes': (30, 2), 'tol': 8e-05}
0.5055555555555556
"""

# Figure 2.4.1
if fig == 1:
    activation = 'logistic'
    alpha = 0.00002
    tol = 0.00008
    layers = (10, 1)
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset(samples=6000)
    sample_size = [i * 500 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=1000,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        num_folds = 3
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        nn.fit(data_train, labels_train)
        testing_scores.append(nn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.1a Accuracy vs Sample Size for Logistic Activation")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.1b Test Scores Accuracy vs Sample Size for Logistic Activation")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.4.2
if fig == 2:
    activation = 'logistic'
    alpha = 0.00002
    tol = 0.00008
    layers = (10, 1)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 21)]
    for iterations in max_iterations:
        num_folds = 3
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=iterations,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.2 Accuracy vs Max Iterations for Logistic Activation")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.4.3
if fig == 3:
    activation = 'tanh'
    alpha = 0.0001
    tol = 0.00006
    layers = (20, 1)
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset(samples=6000)
    sample_size = [i * 500 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=1000,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        num_folds = 3
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        nn.fit(data_train, labels_train)
        testing_scores.append(nn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.3a Accuracy vs Sample Size for Hyperbolic Activation")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.3b Test Scores Accuracy vs Sample Size for Hyperbolic Activation")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.4.4
if fig == 4:
    activation = 'tanh'
    alpha = 0.0001
    tol = 0.00006
    layers = (20, 1)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 21)]
    for iterations in max_iterations:
        num_folds = 3
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=iterations,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.4 Accuracy vs Max Iterations for Hyperbolic Activation")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.4.5
if fig == 5:
    activation = 'logistic'
    alpha = 0.00014
    tol = 0.0001
    layers = (30, 2)
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset(samples=6000)
    sample_size = [i * 500 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=2000,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        num_folds = 3
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        nn.fit(data_train, labels_train)
        testing_scores.append(nn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.5a Accuracy vs Sample Size for Logistic Activation W/ 2 HL")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.5b Testing Accuracy vs Sample Size for Logistic Activation W/ 2 HL")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.4.6
if fig == 6:
    activation = 'logistic'
    alpha = 0.00014
    tol = 0.0001
    layers = (30, 2)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 21)]
    for iterations in max_iterations:
        num_folds = 3
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=iterations,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.6 Accuracy vs Max Iterations for Logistic Activation W/ 2 HL")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.4.7
if fig == 7:
    activation = 'logistic'
    alpha = 0.00012
    tol = 0.00006
    layers = (40, 2)
    cv_testing_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 21)]
    for iterations in max_iterations:
        num_folds = 3
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=iterations,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        nn.fit(data_train, labels_train)
        testing_scores.append(nn.score(data_test, labels_test))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.7 Accuracy vs Max Iterations for Logistic Activation W/ 2 HL")
    ax.plot(max_iterations, cv_testing_scores, marker='s', label='With Cross Validation')
    ax.plot(max_iterations, testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.4.8
if fig == 8:
    activation = 'tanh'
    alpha = 0.00012
    tol = 0.00008
    layers = (30, 2)
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset(samples=6000)
    sample_size = [i * 500 for i in range(1, 13)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=2000,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        num_folds = 3
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        nn.fit(data_train, labels_train)
        testing_scores.append(nn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.8a Accuracy vs Sample Size for Hyperbolic Activation W/ 2 HL")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.8b Testing Accuracy vs Sample Size for Hyperbolic Activation W/ 2 HL")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.4.9
if fig == 9:
    activation = 'tanh'
    alpha = 0.00012
    tol = 0.00008
    layers = (30, 2)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 21)]
    for iterations in max_iterations:
        num_folds = 3
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=activation, max_iter=iterations,
                           alpha=alpha, tol=tol, hidden_layer_sizes=layers)
        cv_results = cross_validate(nn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Iterations")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.4.9 Accuracy vs Max Iterations for Hyperbolic Activation W/ 2 HL")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
