from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_dataset, TEST_SIZE, RANDOM_STATE

print("Neural Networks")
load_dataset(use_math=False, setup='A', passing='C')

###########################
fig = 8
###########################

solver = 'lbfgs'
if fig == 0:
    data, grades = sample_dataset('p')
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
{'alpha': 0.00014000000000000001, 'hidden_layer_sizes': (30, 1), 'tol': 8e-05}
0.892014652014652
tanh
{'alpha': 0.00016, 'hidden_layer_sizes': (45, 1), 'tol': 2e-05}
0.8964835164835165
"""

if fig == 0.1:
    data, grades = sample_dataset('p', num_samples=200)
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    activation = ['logistic', 'tanh']
    for a in activation:
        parameters = {'alpha': [0.00002 * i for i in range(1, 11)],
                      'tol': [0.00002 * i for i in range(1, 11)],
                      'hidden_layer_sizes': [(10, 2), (20, 2), (30, 2), (40, 2), (45, 2)]
                      }
        nn = MLPClassifier(random_state=RANDOM_STATE, solver=solver, activation=a, max_iter=10000)
        search = GridSearchCV(nn, parameters).fit(data_train, labels_train)
        print(a)
        print(search.best_params_)
        print(search.best_score_)

"""
logistic
{'alpha': 8e-05, 'hidden_layer_sizes': (30, 2), 'tol': 0.00018}
0.9071428571428573
"""

# Figure 1.4.1
if fig == 1:
    activation = 'logistic'
    alpha = 0.00014
    tol = 0.00008
    layers = (30, 1)
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
    ax.set_title("Fig 1.4.1a Accuracy vs Sample Size for Logistic Activation")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.4.1b Test Scores Accuracy vs Sample Size for Logistic Activation")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.4.2
if fig == 2:
    activation = 'logistic'
    alpha = 0.00014
    tol = 0.00008
    layers = (30, 1)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 12)]
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
    ax.set_title("Fig 1.4.2 Accuracy vs Max Iterations for Logistic Activation")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.4.3
if fig == 3:
    activation = 'tanh'
    alpha = 0.00016
    tol = 0.00002
    layers = (45, 1)
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
    ax.set_title("Fig 1.4.3a Accuracy vs Sample Size for Hyperbolic Activation")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.4.3b Test Scores Accuracy vs Sample Size for Hyperbolic Activation")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.4.4
if fig == 4:
    activation = 'tanh'
    alpha = 0.00016
    tol = 0.00002
    layers = (45, 1)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 12)]
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
    ax.set_title("Fig 1.4.4 Accuracy vs Max Iterations for Hyperbolic Activation")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.4.5
if fig == 5:
    activation = 'logistic'
    alpha = 0.00008
    tol = 0.00018
    layers = (30, 2)
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
    ax.set_title("Fig 1.4.5a Accuracy vs Sample Size for Logistic Activation W/ 2HL")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.4.5b Test Scores Accuracy vs Sample Size for Logistic Activation W/ 2HL")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.4.6
if fig == 6:
    activation = 'logistic'
    alpha = 0.00008
    tol = 0.00018
    layers = (30, 2)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 12)]
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
    ax.set_title("Fig 1.4.6 Accuracy vs Max Iterations for Logistic Activation W/ 2HL")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.4.7
if fig == 7:
    activation = 'tanh'
    alpha = 0.00016
    tol = 0.00002
    layers = (45, 1)
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
    ax.set_title("Fig 1.4.7a Accuracy vs Sample Size for Hyperbolic Activation W/ 2HL")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.4.7b Test Scores Accuracy vs Sample Size for Hyperbolic Activation W/ 2HL")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.4.8
if fig == 8:
    activation = 'tanh'
    alpha = 0.00016
    tol = 0.00002
    layers = (45, 1)
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_iterations = [i * 20 for i in range(1, 12)]
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
    ax.set_title("Fig 1.4.8 Accuracy vs Max Iterations for Hyperbolic Activation W/ 2HL")
    ax.plot(max_iterations, training_scores, marker='s', label='Training Scores')
    ax.plot(max_iterations, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_iterations, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
