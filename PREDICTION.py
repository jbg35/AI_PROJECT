import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from operator import itemgetter
from sklearn.model_selection import cross_val_score

def scores(model):
    
    model.fit(xtrain, ytrain.values.ravel())
    y_pred = model.predict(xtest)
    
    # print("Accuracy score: %.3f" % metrics.accuracy_score(ytest, y_pred))
    # print("Recall: %.3f" % metrics.recall_score(ytest, y_pred))
    # print("Precision: %.3f" % metrics.precision_score(ytest, y_pred))
    # print("F1: %.3f" % metrics.f1_score(ytest, y_pred))
    
    # proba = model.predict_proba(xtest)
    # print("Log loss: %.3f" % metrics.log_loss(ytest, proba))

    # pos_prob = proba[:, 1]
    # print("Area under ROC curve: %.3f" % metrics.roc_auc_score(ytest, pos_prob))
    
    # cv = cross_val_score(model, xtest, ytest.values.ravel(), cv = 3, scoring = 'accuracy')
    # print("Accuracy (cross validation score): %0.3f (+/- %0.3f)" % (cv.mean(), cv.std() * 2))
    
    return y_pred

def make_pred(model, testpredict, dfCurrentNames):

    proba = model.predict_proba(testpredict)
    pos_prob = proba[:, 1]
    
    combined_list = [[i, j] for i, j in zip(dfCurrentNames, pos_prob)]
    combined_list = sorted(combined_list, key = itemgetter(1), reverse = True)
    
    #for i in combined_list:
        #print(i)
        
    return pos_prob

    
if __name__ == "__main__":
    all_NBA_1979_2018 = pd.read_csv('ALL-NBA-1979-2018.csv')
    non_all_NBA_1979_2018 = pd.read_csv('Non-ALL-NBA-1979-2018.csv')

    all_NBA_1979_2018['ALL-NBA'] = 1
    non_all_NBA_1979_2018['ALL-NBA'] = 0

    ALL_PLAYERS = all_NBA_1979_2018.append(non_all_NBA_1979_2018)

    all_NBA_79_18 = ALL_PLAYERS.loc[ALL_PLAYERS['ALL-NBA'] == 1]
    non_all_nba_79_18 = ALL_PLAYERS.loc[ALL_PLAYERS['ALL-NBA'] == 0]

    train, test = train_test_split(ALL_PLAYERS, test_size = 0.05, random_state = 36)

    xtrain = train[['PTS', 'TRB', 'AST', 'VORP', 'WS', 'BPM']]
    ytrain = train[['ALL-NBA']]

    xtest = test[['PTS', 'TRB', 'AST', 'VORP', 'WS', 'BPM']]
    ytest = test[['ALL-NBA']]

    print("Training set size: %.0f" % len(xtrain))
    print("Testing set size: %.0f" % len(xtest))



    svc = SVC(kernel = 'rbf', gamma = 1e-3, C = 100, probability = True)
    print('SVC Scores')
    y_svc = scores(svc)

    rf = RandomForestClassifier(random_state = 999, n_estimators = 100, criterion = 'gini')
    print('RF Scores')
    y_rf = scores(rf)

    knn = neighbors.KNeighborsClassifier(n_neighbors = 12, weights = 'uniform')
    print('kNN Scores')
    y_knn = scores(knn)

    dnn = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = 100, random_state = 999, activation = 'relu')
    print('MLP Classifier')
    y_dnn = scores(dnn)

    dtc =  DecisionTreeClassifier(max_depth=5)
    print('DTC Scores')
    y_dtc = scores(dtc)

    print('Input File: ')
    filename = input()
    nba_players_INPUT = pd.read_csv(filename)
    nba_names_INPUT = nba_players_INPUT.iloc[:, 1]
    predict_INPUT = nba_players_INPUT[['PTS', 'TRB', 'AST', 'VORP', 'WS', 'BPM']]
    
    svc_prob = make_pred(svc, predict_INPUT, nba_names_INPUT)
    rf_prob = make_pred(rf, predict_INPUT, nba_names_INPUT)
    knn_prob = make_pred(knn,predict_INPUT, nba_names_INPUT)
    dnn_prob = make_pred(dnn,predict_INPUT, nba_names_INPUT)
    dtc_prob = make_pred(dtc,predict_INPUT, nba_names_INPUT)

    avg_prob = []

    for i, j, k, l, m in zip(svc_prob, rf_prob, knn_prob, dnn_prob,dtc_prob ):
        avg_prob.append((i + j + k + l + m) / 5)
        
    avg_list = [[i, j] for i, j in zip(nba_names_INPUT, avg_prob)]
    avg_list = sorted(avg_list, key = itemgetter(1), reverse = True)

    for i in avg_list:
        print(i)
