from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import time

from preprocessing import load_dataset, sample_dataset, TEST_SIZE, RANDOM_STATE

print("Boosting")
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
        boost = AdaBoostClassifier(random_state=RANDOM_STATE)
        estimators = [DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=c, ccp_alpha=alphas[i])
                      for i in range(len(alphas))]
        parameters = {'base_estimator': estimators,
                      'n_estimators': [i * 10 for i in range(3, 8)],
                      'learning_rate': [i * 0.1 for i in range(1, 15)]
                      }
        search = GridSearchCV(boost, parameters).fit(data_train, labels_train)
        print(c)
        print(search.best_params_)
        print(search.best_score_)

"""
gini
{'base_estimator': DecisionTreeClassifier(ccp_alpha=0.020851489481254898, random_state=42),
 'learning_rate': 0.2, 'n_estimators': 40}
0.9537484737484737
entropy
{'base_estimator': DecisionTreeClassifier(ccp_alpha=0.013230869951538196, criterion='entropy', random_state=42),
 'learning_rate': 0.9, 'n_estimators': 40}
0.9515262515262515
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
        boost = AdaBoostClassifier(random_state=RANDOM_STATE)
        estimators = [DecisionTreeClassifier(random_state=RANDOM_STATE, criterion=c, ccp_alpha=alphas[i])
                      for i in range(len(alphas))]
        parameters = {'base_estimator': estimators,
                      'n_estimators': [i * 10 for i in range(3, 8)],
                      'learning_rate': [i * 0.1 for i in range(1, 15)]
                      }
        search = GridSearchCV(boost, parameters).fit(data_train, labels_train)
        print(c)
        print(search.best_params_)
        print(search.best_score_)

"""
{'base_estimator': DecisionTreeClassifier(ccp_alpha=0.009975213499442588, random_state=42),
 'learning_rate': 0.7, 'n_estimators': 60}
0.7641758241758241
entropy
{'base_estimator': DecisionTreeClassifier(ccp_alpha=0.006789424048877238, criterion='entropy', random_state=42),
 'learning_rate': 0.8, 'n_estimators': 30}
0.7688156288156287
"""

# Figure 1.2.1
if fig == 1:
    criterion = 'gini'
    ccp_alpha = 0.020851489481254898
    n_estimators = 40
    learning_rate = 0.2
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
    ax.set_title("Fig 1.2.1a Accuracy vs Sample Size for Boosted GINI Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.2.1b Test Scores Accuracy vs Sample Size for Boosted GINI Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.2.2
if fig == 2:
    criterion = 'gini'
    ccp_alpha = 0.020851489481254898
    n_estimators = 40
    learning_rate = 0.2
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
    ax.set_title("Fig 1.2.2 Accuracy vs Max Depth for Boosted GINI Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.2.3
if fig == 3:
    criterion = 'entropy'
    ccp_alpha = 0.013230869951538196
    n_estimators = 40
    learning_rate = 0.9
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
    ax.set_title("Fig 1.2.3a Accuracy vs Sample Size for Boosted Entropy Decision Tree")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.2.3b Test Scores Accuracy vs Sample Size for Boosted Entropy Decision Tree")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.2.4
if fig == 4:
    criterion = 'entropy'
    ccp_alpha = 0.013230869951538196
    n_estimators = 40
    learning_rate = 0.9
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
    ax.set_title("Fig 1.2.4 Accuracy vs Max Depth for Boosted Entropy Decision Tree")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.2.5
if fig == 5:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'gini'
    ccp_alpha = 0.009975213499442588
    n_estimators = 60
    learning_rate = 0.7
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
    ax.set_title("Fig 1.2.5a Accuracy vs Sample Size For Boosted GINI With Dataset Setup C")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.2.5b Test Scores Accuracy vs Sample Size For Boosted GINI With Dataset Setup C")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.2.6
if fig == 6:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'gini'
    ccp_alpha = 0.009975213499442588
    n_estimators = 60
    learning_rate = 0.7
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
                                      max_depth=depth)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
        print(cv_results['estimator'][0].feature_importances_)
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.2.6 Accuracy vs Max Depth For Boosted GINI With Dataset Setup C")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.2.7
if fig == 7:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'entropy'
    ccp_alpha = 0.006789424048877238
    n_estimators = 30
    learning_rate = 0.8
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
    ax.set_title("Fig 1.2.7a Accuracy vs Sample Size For Entropy With Dataset Setup C")
    ax.plot(sample_size, training_scores, marker='s', label='Training Scores')
    ax.plot(sample_size, validation_scores, marker='s', label='Validation Scores')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
    _, ax = plt.subplots()
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.2.7b Test Scores Accuracy vs Sample Size For Boosted Entropy With Dataset Setup C")
    ax.plot(sample_size, testing_scores, marker='s', label='With Cross Validation')
    ax.plot(sample_size, cv_testing_scores, marker='s', label='Without Cross Validation')
    plt.legend()
    plt.show()
# Figure 1.2.8
if fig == 8:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'entropy'
    ccp_alpha = 0.006789424048877238
    n_estimators = 30
    learning_rate = 0.8
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
                                      max_depth=depth)
        boost = AdaBoostClassifier(random_state=RANDOM_STATE, base_estimator=tree,
                                   n_estimators=n_estimators, learning_rate=learning_rate)
        cv_results = cross_validate(boost, data_train, labels_train, cv=num_folds,
                                    return_estimator=True, return_train_score=True)
        training_scores.append(np.mean(cv_results['train_score']))
        validation_scores.append(np.mean(cv_results['test_score']))
        testing_scores.append(np.mean([cv_results['estimator'][i].score(data_test, labels_test)
                                       for i in range(num_folds)]))
        print(cv_results['estimator'][0].feature_importances_)
    fig, ax = plt.subplots()
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.2.8 Accuracy vs Max Depth For Boosted Entropy With Dataset Setup C")
    ax.plot(max_depth, training_scores, marker='s', label='Training Scores')
    ax.plot(max_depth, validation_scores, marker='s', label='Validation Scores')
    ax.plot(max_depth, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.2.9
if fig == 9:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'gini'
    ccp_alpha = 0.009975213499442588
    n_estimators = 60
    learning_rate = 0.7
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    estimators = [i for i in range(1, 16)]
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
    ax.set_title("Fig 1.2.9 Accuracy vs Num Estimators for Boosted GINI Tree With Setup C")
    ax.plot(estimators, training_scores, marker='s', label='Training Scores')
    ax.plot(estimators, validation_scores, marker='s', label='Validation Scores')
    ax.plot(estimators, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
# Figure 1.2.10
if fig == 10:
    load_dataset(use_math=False, setup='C', passing='C')
    criterion = 'entropy'
    ccp_alpha = 0.006789424048877238
    learning_rate = 0.8
    training_scores = []
    validation_scores = []
    testing_scores = []
    data, grades = sample_dataset('p')
    data_train, data_test, labels_train, labels_test = train_test_split(data, grades,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)
    estimators = [i for i in range(1, 16)]
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
        print(cv_results['estimator'][0].feature_importances_)
    fig, ax = plt.subplots()
    ax.set_xlabel("Num Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig 1.2.10 Accuracy vs Num Estimators For Boosted Entropy With Setup C")
    ax.plot(estimators, training_scores, marker='s', label='Training Scores')
    ax.plot(estimators, validation_scores, marker='s', label='Validation Scores')
    ax.plot(estimators, testing_scores, marker='s', label='Test Scores')
    plt.legend()
    plt.show()
