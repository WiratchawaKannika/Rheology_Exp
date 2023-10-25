import os
import pandas as pd 
import numpy as np 


def loaddata(path, classname, fold)
    data_feature = pd.read_csv(path)
    print(data_feature.shape)
    print("-"*100)
    print(f"[INFO]: All Fold : {set(data_feature.fold)}")
    ## Split Train data Set
    feature_train = data_feature[data_feature["fold"]!=fold].reset_index(drop=True)
    print(f"Validated Train Set : Fold ==> {set(feature_train.fold)}")
    print("Train Set With Shape = ", feature_train.shape)

    X_train = feature_train[['MCV','MCH','Hb']]
    y_train = feature_train[classname]
    print("X_train shape: ", X_train.shape)
    print("X_train shape: ", y_train.shape)
    print(f"Classified Number: ", len(list(set(y_train))))
    print(set(y_train))
    print("="*100)
    
    return X_train, y_train

