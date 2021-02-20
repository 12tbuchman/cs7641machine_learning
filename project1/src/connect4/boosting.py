from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score

import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_unbalanced_dataset, sample_balanced_dataset, TEST_SIZE, RANDOM_STATE

print("Boosting")
load_dataset()

###########################
fig = 0
###########################

if fig == 0:
    data, grades = sample_balanced_dataset(samples=600)
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)

    criterion = ['gini', 'entropy']
    for c in criterion:
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=c)
        alphas = tree.cost_complexity_pruning_path(data_train, labels_train).ccp_alphas[::13]
        boost = AdaBoostClassifier(random_state=RANDOM_STATE)
        estimators = [DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=c, ccp_alpha=alphas[i])
                      for i in range(len(alphas))]
        parameters = {'base_estimator': estimators,
                      'n_estimators': [i * 10 for i in range(3, 6)],
                      'learning_rate': [i * 0.1 for i in range(1, 11)]
                      }
        search = GridSearchCV(boost, parameters).fit(data_train, labels_train)
        print(c)
        print(search.best_params_)
        print(search.best_score_)

"""
gini
{'base_estimator': DecisionTreeClassifier(ccp_alpha=0.0037037037037037034, random_state=42),
 'learning_rate': 0.9, 'n_estimators': 30}
0.5472222222222223
entropy
{'base_estimator': DecisionTreeClassifier(ccp_alpha=0.015537397122943816, criterion='entropy', random_state=42),
 'learning_rate': 0.8, 'n_estimators': 40}
0.5444444444444444
"""

if fig == 0.1:
    data, grades = sample_balanced_dataset(samples=600)
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)

    criterion = ['gini', 'entropy']
    for c in criterion:
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=c)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree)
        parameters = {'n_estimators': [i * 10 for i in range(3, 6)],
                      'learning_rate': [i * 0.1 for i in range(1, 11)]
                      }
        search = GridSearchCV(boost, parameters).fit(data_train, labels_train)
        print(c)
        print(search.best_params_)
        print(search.best_score_)

"""
gini
{'learning_rate': 0.1, 'n_estimators': 30}
0.46944444444444444
entropy
{'learning_rate': 0.1, 'n_estimators': 30}
0.4416666666666667
"""

# Figure 2.2.1
if fig == 1:
    criterion = 'gini'
    ccp_alpha = 0.0037037037037037034
    n_estimators = 30
    learning_rate = 0.9
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset()
    sample_size = [i * 1000 for i in range(1, 19)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        num_folds = 3
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        boost.fit(data_train, labels_train)
        testing_scores.append(boost.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.1a Accuracy vs Sample Size for Boosted GINI Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.1b Test Scores Accuracy vs Sample Size for Boosted GINI Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.2.2
if fig == 2:
    criterion = 'gini'
    ccp_alpha = 0.0037037037037037034
    n_estimators = 30
    learning_rate = 0.9
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 11)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      max_depth=depth)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.2 Accuracy vs Max Depth for Boosted GINI Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.2.3
if fig == 3:
    criterion = 'entropy'
    ccp_alpha = 0.015537397122943816
    n_estimators = 40
    learning_rate = 0.8
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset()
    sample_size = [i * 1000 for i in range(1, 19)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        num_folds = 3
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        boost.fit(data_train, labels_train)
        testing_scores.append(boost.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.3a Accuracy vs Sample Size for Boosted Entropy Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.3b Test Scores Accuracy vs Sample Size for Boosted Entropy Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.2.4
if fig == 4:
    criterion = 'entropy'
    ccp_alpha = 0.015537397122943816
    n_estimators = 40
    learning_rate = 0.8
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 11)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      max_depth=depth)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.4 Accuracy vs Max Depth for Boosted Entropy Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.2.5
if fig == 5:
    criterion = 'gini'
    n_estimators = 30
    learning_rate = 0.1
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset()
    sample_size = [i * 1000 for i in range(1, 19)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        num_folds = 3
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        boost.fit(data_train, labels_train)
        testing_scores.append(boost.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.5a Accuracy vs Sample Size for Boosted GINI Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.5b Test Scores Accuracy vs Sample Size for Boosted GINI Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.2.6
if fig == 6:
    criterion = 'gini'
    n_estimators = 30
    learning_rate = 0.1
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 16)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion,
                                      max_depth=depth)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.6 Accuracy vs Max Depth for Boosted GINI Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.2.7
if fig == 7:
    criterion = 'entropy'
    n_estimators = 30
    learning_rate = 0.1
    training_scores = []
    validation_scores = []
    cv_testing_scores = []
    testing_scores = []
    data_all, grades_all = sample_balanced_dataset()
    sample_size = [i * 1000 for i in range(1, 19)]
    for samples in sample_size:
        data = data_all[:samples]
        grades = grades_all[:samples]
        data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                            test_size=TEST_SIZE,
                                                                            random_state=RANDOM_STATE)
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        num_folds = 3
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        cv_testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                          for i in range(num_folds)]))
        boost.fit(data_train, labels_train)
        testing_scores.append(boost.score(data_test, labels_test))
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.7a Accuracy vs Sample Size for Boosted Entropy Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.7b Test Scores Accuracy vs Sample Size for Boosted Entropy Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 2.2.8
if fig == 8:
    criterion = 'entropy'
    n_estimators = 30
    learning_rate = 0.1
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    max_depth = [i for i in range(1, 16)]
    for depth in max_depth:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion,
                                      max_depth=depth)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.8 Accuracy vs Max Depth for Boosted Entropy Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.2.9
if fig == 9:
    criterion = 'gini'
    ccp_alpha = 0.0037037037037037034
    learning_rate = 0.9
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    estimators = [i for i in range(1, 11)]
    for n in estimators:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      max_depth=10)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Num Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.9 Accuracy vs Num Estimators for Boosted GINI Decision Tree")
    ax.plot(estimators, training_scores, marker='s', label='Training Scores')
    ax.plot(estimators, validation_scores, marker='s', label='Validation Scores')
    ax.plot(estimators, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 2.2.10
if fig == 10:
    criterion = 'entropy'
    ccp_alpha = 0.015537397122943816
    learning_rate = 0.8
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_balanced_dataset()
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    estimators = [i for i in range(1, 11)]
    for n in estimators:
        num_folds = 3
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=criterion, ccp_alpha=ccp_alpha,
                                      max_depth=10)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
    fig, ax = plt.subplots()
    ax.set_xlabel("Num Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 2.2.10 Accuracy vs Num Estimators for Boosted Entropy Decision Tree")
    ax.plot(estimators, training_scores, marker='s', label='Training Scores')
    ax.plot(estimators, validation_scores, marker='s', label='Validation Scores')
    ax.plot(estimators, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
