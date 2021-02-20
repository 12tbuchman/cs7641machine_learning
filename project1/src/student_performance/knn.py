from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_dataset, TEST_SIZE, RANDOM_STATE

print("K-Nearest Neighbor")
load_dataset(use_math=False, setup='A', passing='C')

###########################
fig = 7
###########################

if fig == 0:
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    neighbors = [5, 7, 9, 11, 13, 15]
    for n in neighbors:
        parameters = {'algorithm': ['kd_tree', 'ball_tree'],
                      'leaf_size': [5 * i for i in range(3, 21)],
                      'p': [1, 2, 3]
                      }
        knn = KNeighborsClassifier(n_neighbors=n)
        search = GridSearchCV(knn, parameters).fit(data_train, labels_train)
        print(str(n) + " neighbors")
        print(search.best_params_)
        print(search.best_score_)

"""
5 neighbors
{'algorithm': 'ball_tree', 'leaf_size': 50, 'p': 1}
0.9031501831501831
7 neighbors
{'algorithm': 'ball_tree', 'leaf_size': 15, 'p': 1}
0.9141636141636141
9 neighbors
{'algorithm': 'kd_tree', 'leaf_size': 50, 'p': 1}
0.9185836385836386
11 neighbors
{'algorithm': 'kd_tree', 'leaf_size': 15, 'p': 2}
0.9163858363858364
13 neighbors
{'algorithm': 'kd_tree', 'leaf_size': 15, 'p': 2}
0.9207570207570208
15 neighbors
{'algorithm': 'kd_tree', 'leaf_size': 15, 'p': 1}
0.9097680097680098
"""

# Figure 1.3.1
if fig == 1:
    neighbors = 5
    algorithm = 'ball_tree'
    leaf_size = 50
    p = 1
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
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p)
        num_folds = 3
        cv_results = cross_validate(knn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        knn.fit(data_train, labels_train)
        testing_scores.append(knn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.1a Accuracy vs Sample Size for 5 Nearest Neighbor")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.1b Test Scores Accuracy vs Sample Size for 5 Nearest Neighbor")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.3.2
if fig == 2:
    neighbors = 7
    algorithm = 'ball_tree'
    leaf_size = 15
    p = 1
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
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p)
        num_folds = 3
        cv_results = cross_validate(knn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        knn.fit(data_train, labels_train)
        testing_scores.append(knn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.2a Accuracy vs Sample Size for 7 Nearest Neighbor")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.2b Test Scores Accuracy vs Sample Size for 7 Nearest Neighbor")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.3.3
if fig == 3:
    neighbors = 9
    algorithm = 'kd_tree'
    leaf_size = 50
    p = 1
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
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p)
        num_folds = 3
        cv_results = cross_validate(knn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        knn.fit(data_train, labels_train)
        testing_scores.append(knn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.3a Accuracy vs Sample Size for 9 Nearest Neighbor")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.3b Test Scores Accuracy vs Sample Size for 9 Nearest Neighbor")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.3.4
if fig == 4:
    neighbors = 11
    algorithm = 'kd_tree'
    leaf_size = 15
    p = 2
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
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p)
        num_folds = 3
        cv_results = cross_validate(knn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        knn.fit(data_train, labels_train)
        testing_scores.append(knn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.4a Accuracy vs Sample Size for 11 Nearest Neighbor")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.4b Test Scores Accuracy vs Sample Size for 11 Nearest Neighbor")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.3.5
if fig == 5:
    neighbors = 13
    algorithm = 'kd_tree'
    leaf_size = 15
    p = 2
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
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p)
        num_folds = 3
        cv_results = cross_validate(knn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        knn.fit(data_train, labels_train)
        testing_scores.append(knn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.5a Accuracy vs Sample Size for 13 Nearest Neighbor")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.5b Test Scores Accuracy vs Sample Size for 13 Nearest Neighbor")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.3.6
if fig == 6:
    neighbors = 15
    algorithm = 'kd_tree'
    leaf_size = 15
    p = 1
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
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p)
        num_folds = 3
        cv_results = cross_validate(knn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        knn.fit(data_train, labels_train)
        testing_scores.append(knn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.6a Accuracy vs Sample Size for 15 Nearest Neighbor")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.6b Test Scores Accuracy vs Sample Size for 15 Nearest Neighbor")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.3.7
if fig == 7:
    neighbors = 5
    algorithm = 'ball_tree'
    leaf_size = 50
    p = 1
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
        knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm,
                                   leaf_size=leaf_size, p=p, weights='distance')
        num_folds = 3
        cv_results = cross_validate(knn, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        knn.fit(data_train, labels_train)
        testing_scores.append(knn.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.7a Accuracy vs Sample Size for 5 NN W/ Distance Weight")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.3.7b Test Scores Accuracy vs Sample Size for 5 NN W/ Distance Weight")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
