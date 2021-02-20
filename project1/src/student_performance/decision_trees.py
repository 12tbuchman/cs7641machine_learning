from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_dataset, TEST_SIZE, RANDOM_STATE

print("Decision Trees")
load_dataset(use_math=False, setup='A', passing='C')

###########################
fig = 0
###########################

if fig == 0:
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)

    criterion = ['gini', 'entropy']
    for c in criterion:
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=c)
        alphas = tree.cost_complexity_pruning_path(data_train, labels_train).ccp_alphas
        parameters = {'min_samples_split': [i for i in range(2, 10)],
                      'min_samples_leaf': [i for i in range(1, 10)],
                      'ccp_alpha': alphas
                      }
        search = GridSearchCV(tree, parameters).fit(data_train, labels_train)
        print(c)
        print(search.best_params_)
        print(search.best_score_)

"""
gini
{'ccp_alpha': 0.020851489481254898, 'min_samples_leaf': 1, 'min_samples_split': 2}
0.9426862026862027
entropy
{'ccp_alpha': 0.07513403294522009, 'min_samples_leaf': 1, 'min_samples_split': 2}
0.9426862026862027
"""

if fig == 0.1:
    load_dataset(use_math=False, setup='C', passing='C')
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)

    criterion = ['gini', 'entropy']
    for c in criterion:
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=c)
        alphas = tree.cost_complexity_pruning_path(data_train, labels_train).ccp_alphas
        parameters = {'min_samples_split': [i for i in range(2, 10)],
                      'min_samples_leaf': [i for i in range(1, 10)],
                      'ccp_alpha': alphas
                      }
        search = GridSearchCV(tree, parameters).fit(data_train, labels_train)
        print(c)
        print(search.best_params_)
        print(search.best_score_)

"""
gini
{'ccp_alpha': 0.009975213499442588, 'min_samples_leaf': 1, 'min_samples_split': 2}
0.781978021978022
entropy
{'ccp_alpha': 0.01347299923388915, 'min_samples_leaf': 8, 'min_samples_split': 2}
0.7665934065934066
"""

# Figure 1.1.1
if fig == 1:
    criterion = 'gini'
    ccp_alpha = 0.020851489481254898
    min_samples_leaf = 1
    min_samples_split = 2
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
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        num_folds = 3
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        tree.fit(data_train, labels_train)
        testing_scores.append(tree.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.1a Accuracy vs Sample Size for GINI Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.1b Test Scores Accuracy vs Sample Size for GINI Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.1.2
if fig == 2:
    criterion = 'gini'
    ccp_alpha = 0.020851489481254898
    min_samples_leaf = 1
    min_samples_split = 2
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 10)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                      max_depth=depth)
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
        print(cv_results['estimator'][0].feature_importances_)
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.2 Accuracy vs Max Depth for GINI Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.1.3
if fig == 3:
    criterion = 'entropy'
    ccp_alpha = 0.07513403294522009
    min_samples_leaf = 1
    min_samples_split = 2
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
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        num_folds = 3
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        tree.fit(data_train, labels_train)
        testing_scores.append(tree.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.3a Accuracy vs Sample Size for Entropy Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.3b Test Scores Accuracy vs Sample Size for Entropy Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.1.4
if fig == 4:
    criterion = 'entropy'
    ccp_alpha = 0.07513403294522009
    min_samples_leaf = 1
    min_samples_split = 2
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 10)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                      max_depth=depth)
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
        print(cv_results['estimator'][0].feature_importances_)
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.4 Accuracy vs Max Depth for Entropy Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.1.5
if fig == 5:
    criterion = 'entropy'
    ccp_alpha = 0.07513403294522009
    min_samples_leaf = 1
    min_samples_split = 2
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE+1)
    max_depth = [i for i in range(1, 20)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE+1, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                      max_depth=depth)
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.5 Accuracy vs Max Depth With Alternate Random State")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.1.6
if fig == 6:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'gini'
    ccp_alpha = 0.009975213499442588
    min_samples_leaf = 1
    min_samples_split = 2
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
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        num_folds = 3
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        tree.fit(data_train, labels_train)
        testing_scores.append(tree.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.6a Accuracy vs Sample Size For GINI With Dataset Setup C")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.6b Test Scores Accuracy vs Sample Size For GINI With Dataset Setup C")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.1.7
if fig == 7:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'gini'
    ccp_alpha = 0.009975213499442588
    min_samples_leaf = 1
    min_samples_split = 2
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 10)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                      max_depth=depth)
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
        print(cv_results['estimator'][0].feature_importances_)
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.7 Accuracy vs Max Depth For GINI With Dataset Setup C")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.1.8
if fig == 8:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'entropy'
    ccp_alpha = 0.01347299923388915
    min_samples_leaf = 8
    min_samples_split = 2
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
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        num_folds = 3
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        tree.fit(data_train, labels_train)
        testing_scores.append(tree.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.8a Accuracy vs Sample Size For Entropy With Dataset Setup C")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.8b Test Scores Accuracy vs Sample Size For Entropy With Dataset Setup C")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.1.9
if fig == 9:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'entropy'
    ccp_alpha = 0.01347299923388915
    min_samples_leaf = 8
    min_samples_split = 2
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 10)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                      max_depth=depth)
        cv_results = cross_validate(tree, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
        print(cv_results['estimator'][0].feature_importances_)
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.1.9 Accuracy vs Max Depth For Entropy With Dataset Setup C")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
