#!/usr/bin/env python
# -*- coding: utf-8 -*-

import metrics

import numpy as np
import pandas as pd

__dataset_home += "./data/replicated_datasets/"

def remove_const_violations(X, y, y_avg):
    """ Remove all the constraints not consistent with the
        rule:
        if x[i] dominant on x[j] then y[i] > y[j]

    Paramters
    ---------
        X : list(float)
            Instances of accuracy
        y : list(float)
            Error
        y_avg : list(float)
            Average Error

    Returns
    -------
        X : list(float)
            Instances of accuracy without inversions
        y : list(float)
            Error without inversions
        y_avg : list(float)
            Average Error without the same element
            removed from y
        tot_couples : int
            Number of couples (dominant, dominated)
    """
    n = X.shape[0]
    to_delete = set()
    tot_couples = 0
    for i in range(n-1):
        x_1 = X[i]
        for j in range(i+1, n):
            x_2 = X[j]
            if metrics.is_dominant(x_1, x_2):
                tot_couples += 1
                if y[i] < y[j]:
                    to_delete.add(j)
    to_delete = list(to_delete)
    X = np.delete(X, to_delete, 0)
    y = np.delete(y, to_delete, 0)
    y_avg = np.delete(y_avg, to_delete, 0)
    assert metrics.const_violation(X, y)[0] == 0
    assert X.shape[0] == n - len(to_delete)
    assert y.shape[0] == X.shape[0]
    return X, y, y_avg, tot_couples


def read_replicable_dataset(n_train, n_test, bm, i):
    """ Reads replicable dataset (dataset fo testing porpouse)

    Returns
    -------
        Dataframe for train set and test set
    """
    base_path = __dataset_replicable_home + str(n_train) + 'tr_' \
                + str(n_test) + 'te_500v/' + str(bm) + '/'
    train_path = base_path + str(bm) + '_train_set_' + str (i) + '.csv'
    test_path =  base_path + str(bm) + '_test_set_' + str (i) + '.csv'
    val_path =  base_path + str(bm) + '_val_set_' + str (i) + '.csv'
    df_train = pd.read_csv(train_path, sep=';')
    df_test = pd.read_csv(test_path, sep=';')
    df_val = pd.read_csv(val_path, sep=';')
    return df_train, df_test, df_val

def read_dataset(bm):
    """ Reads dataset """
    # Initialize a pandas DataFrame from file
    data_file = 'exp_results_{}.csv'.format(bm)
    df = pd.read_csv(__dataset_home + data_file, sep=';')
    # Keep entries with all non-zero values
    df = df[(df != 0).all(1)]

    return df
