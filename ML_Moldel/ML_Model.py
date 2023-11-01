import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def createtable_bestparam(bestF):
    bestF_df = pd.DataFrame(bestF.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
    bestF_df = bestF_df.sort_values(by='mean_test_score', ascending=False)
    bestF_df = bestF_df.reset_index(drop=True)
    return bestF_df


def RandomForest(X_train, y_train):
    n_estimators = [400]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100] 
    forest = RandomForestClassifier(random_state = 1)
    hyperF ={'n_estimators' : n_estimators, 'max_depth' : max_depth, 'min_samples_split' : min_samples_split}
    # hyperF ={'max_depth' : max_depth, 'min_samples_split' : min_samples_split}
    gridF = GridSearchCV(forest, hyperF, cv = 10, verbose = 1, n_jobs = -1)
    bestF = gridF.fit(X_train, y_train)
    
    bestF_df = createtable_bestparam(bestF)
    # examine the first result
    print("**examine the first result","\n")

    print(bestF.cv_results_['params'][0])
    print(bestF.cv_results_['mean_test_score'][0])

    # print the array of mean scores only
    print("\n","**print the array of mean scores only","\n")

    grid_mean_scores = bestF.cv_results_['mean_test_score']
    print(grid_mean_scores)

    # examine the best model
    print("\n","**examine the best model","\n")

    print(bestF.best_score_)
    print(bestF.best_params_)
    print(bestF.best_estimator_)
    
    #Print the tured parameters and score
    print("Tuned Decision Tree Parameters: {}".format(bestF.best_params_))
    print("Best score is {}".format(bestF.best_score_))
    
    depth = bestF.best_params_['max_depth']
    samples_split = bestF.best_params_['min_samples_split']
    estimators = bestF.best_params_['n_estimators']
    
    #Fit Model and setting parameters

