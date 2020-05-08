#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Script to launch experiments with replicable data sets 
    - autosklearn version
University of Bologna
2019-12-19
'''

import sys, os, time
import csv

import loss
import reader
import metrics

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import extract_dep_graphs_from_json as ed
from sklearn.exceptions import DataConversionWarning
from mkl_random import bench
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import autosklearn.regression

suff_label_target = 'err_ds_'

niter = 5

benchmarks = ['BlackScholes', 'convolution', 'correlation', 'dwt', 
    'Jacobi', 'saxpy']
train_set_size = [500]
test_set_size = 1000
val_set_size = 500

cap_error = True
cap = 0.95
improved_dataset = 1

n_layers = 4
neurons_per_layer = 10   # indicates 10 times the number of features

labels = list(map(str, list(range(niter))))

label_targets = [suff_label_target + str(i) for i in range(ndataset)]

def benchmark_test(n_train, n_test, n_val, bm):
    (mae_is0_is0, inversion_is0_is0, plateau_is0_is0, 
            dist_const_is0_is0) = [], [], [], []
    (mae_is0_avg, inversion_is0_avg, plateau_is0_avg, 
            dist_const_is0_avg) = [], [], [], []
    (mae_avg_is0, inversion_avg_is0, plateau_avg_is0, 
            dist_const_avg_is0) = [], [], [], []
    (mae_avg_avg, inversion_avg_avg, plateau_avg_avg, 
            dist_const_avg_avg) = [], [], [], []
    tot_couples_train, tot_couples_test = [], []

    for i in range(niter):
        print("Experiment n. {}".format(i))
        # reading i-th test and trainig set
        df_train, df_test, df_val = reader.read_replicable_dataset(
                n_train, n_test, bm, i)
        ninput = len(list(df_train.filter(regex='var_*')))
        width = ninput * 10

        # normalizing test and training set
        scaler = preprocessing.MinMaxScaler()
        label_target = suff_label_target + str(0)

        if improved_dataset == 1:
            add_features = ed.getAdditionalFeatures(bm)
            for add_feat in add_features:
                df_train['{}-{}'.format(
                    add_feat[0], add_feat[1])
                    ] = df_train[add_feat[0]] - df_train[add_feat[1]]
                df_test['{}-{}'.format(
                    add_feat[0], add_feat[1])
                    ] = df_test[add_feat[0]] - df_test[add_feat[1]]
                df_val['{}-{}'.format(
                    add_feat[0], add_feat[1])
                    ] = df_val[add_feat[0]] - df_val[add_feat[1]]
            ninput += len(add_features)
            #Arrange the df with err column at the end
            for _label_target in label_targets:
                df_temp = df_train.pop(_label_target)
                df_train[_label_target] = df_temp
                df_temp = df_test.pop(_label_target)
                df_test[_label_target] = df_temp
                df_temp = df_val.pop(_label_target)
                df_val[_label_target] = df_temp
 
        if cap_error:
            # capping error
            for _label_target in label_targets:
                df_train[_label_target] = [cap if x > cap else x for x in 
                        df_train[_label_target]]
                df_test[_label_target] = [cap if x > cap else x for x in 
                        df_test[_label_target]]
                df_val[_label_target] = [cap if x > cap else x for x in 
                        df_val[_label_target]]

        # -log10(err)
        for _label_target in label_targets:
            df_train[_label_target] = [sys.float_info.min if 0 == x else 
                    -np.log10(x) for x in df_train[_label_target]]
            df_test[_label_target] = [sys.float_info.min if 0 == x else 
                    -np.log10(x) for x in df_test[_label_target]]
            df_val[_label_target] = [sys.float_info.min if 0 == x else 
                    -np.log10(x) for x in df_val[_label_target]]

        y_avg_train = np.mean(np.array(df_train[label_targets]), axis=1, 
                dtype='float32').reshape((-1, 1))
        y_avg_test = np.mean(np.array(df_test[label_targets]), axis=1, 
                dtype='float32').reshape((-1, 1))
        y_avg_val = np.mean(np.array(df_val[label_targets]), axis=1, 
                dtype='float32').reshape((-1, 1))

        # splitting training and test set in features and targets
        x_train = scaler.fit_transform(df_train.iloc[:,0:ninput])
        y_train = scaler.fit_transform(np.array(df_train[label_target]
            ).reshape((-1, 1)))
        x_test = scaler.fit_transform(df_test.iloc[:,0:ninput])
        y_test = scaler.fit_transform(np.array(df_test[label_target]
            ).reshape((-1, 1)))
        x_val = scaler.fit_transform(df_val.iloc[:,0:ninput])
        y_val = scaler.fit_transform(np.array(df_val[label_target]
            ).reshape((-1, 1)))

        (_, _, _, couples_train) = reader.remove_const_violations(
                x_train, y_train, y_avg_train)
        (_, _, _, couples_test) = reader.remove_const_violations(
                x_test, y_test, y_avg_test)
        (_, _, _, couples_val) = reader.remove_const_violations(
                x_val, y_val, y_avg_val)

        tot_couples_train.append(couples_train)
        tot_couples_test.append(couples_test)

        # Train model on specific input set 
        automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120, per_run_time_limit=30,)
        automl.fit(x_train, y_train)
        # test it on average err among
        y_pred = np.array(automl.predict(x_test))
        mae_is0_avg.append(metrics.mae(y_avg_test, y_pred))
        tmp = metrics.const_violation(x_test, y_pred)
        inversion_is0_avg.append(tmp[0])
        plateau_is0_avg.append(tmp[1])
        dist_const_is0_avg.append(metrics.error_average_distance(y_pred))
        # test it on err of same input set as training
        mae_is0_is0.append(metrics.mae(y_test, y_pred))
        inversion_is0_is0.append(tmp[0])
        plateau_is0_is0.append(tmp[1])
        dist_const_is0_is0.append(metrics.error_average_distance(y_pred))

        # Train model on avg error among input set 
        automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120, per_run_time_limit=30,)
        automl.fit(x_train, y_avg_train)
        # test it on average error 
        y_pred = np.array(automl.predict(x_test))
        mae_avg_avg.append(metrics.mae(y_avg_test, y_pred))
        tmp = metrics.const_violation(x_test, y_pred)
        inversion_avg_avg.append(tmp[0])
        plateau_avg_avg.append(tmp[1])
        dist_const_avg_avg.append(metrics.error_average_distance(y_pred))
        # test it on error of specific input set 
        mae_avg_is0.append(metrics.mae(y_test, y_pred))
        inversion_avg_is0.append(tmp[0])
        plateau_avg_is0.append(tmp[1])
        dist_const_avg_is0.append(metrics.error_average_distance(y_pred))

    return (mae_is0_is0, inversion_is0_is0, plateau_is0_is0, dist_const_is0_is0,
            mae_is0_avg, inversion_is0_avg, plateau_is0_avg, dist_const_is0_avg,
            mae_avg_is0, inversion_avg_is0, plateau_avg_is0, dist_const_avg_is0,
            mae_avg_avg, inversion_avg_avg, plateau_avg_avg, dist_const_avg_avg,
            tot_couples_train, tot_couples_test)


def store(filename, mae, inversion, plateau, dist_const,
         tot_couples_train, tot_couples_test):
    """ Store in a CSV file the results """
    with open('csv/' + filename + '.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerows(np.array([mae, inversion, plateau, dist_const,
         tot_couples_train, tot_couples_test]).T)

def main(argv):
    bm = argv[0]
    n_train = int(argv[1])
    n_test = int(argv[2])
    n_val = int(argv[3])

    if cap_error:
        cap_str = '{}cap'.format(cap)
    else:
        cap_str = 'NOcap'

    print('{} {}tr {}te {}v {} {}ext autoskl'.format(bm, 
        n_train, n_test, n_val, cap_str, improved_dataset))
    filename = "{}_{}tr_{}te_{}v_{}_{}ext_autoskl".format(bm, n_train, n_test,
            n_val, cap_str, improved_dataset)
    fname_is0_is0 = "{}_is0_is0_{}tr_{}te_{}v_{}_{}ext_autoskl".format(bm,
            n_train, n_test, n_val, cap_str, improved_dataset)
    fname_is0_avg = "{}_is0_avg_{}tr_{}te_{}v_{}_{}ext_autoskl".format(bm,
            n_train, n_test, n_val, cap_str, improved_dataset)
    fname_avg_is0 = "{}_avg_is0_{}tr_{}te_{}v_{}_{}ext_autoskl".format(bm,
            n_train, n_test, n_val, cap_str, improved_dataset)
    fname_avg_avg = "{}_avg_avg_{}tr_{}te_{}v_{}_{}ext_autoskl".format(bm,
            n_train, n_test, n_val, cap_str, improved_dataset)

    (mae_is0_is0, inversion_is0_is0, plateau_is0_is0, dist_const_is0_is0,
            mae_is0_avg, inversion_is0_avg, plateau_is0_avg, dist_const_is0_avg,
            mae_avg_is0, inversion_avg_is0, plateau_avg_is0, dist_const_avg_is0,
            mae_avg_avg, inversion_avg_avg, plateau_avg_avg, dist_const_avg_avg,
            tot_couples_train, tot_couples_test) = benchmark_test(n_train,
                    n_test, n_val, bm)

    store(fname_is0_is0, mae_is0_is0, inversion_is0_is0, plateau_is0_is0, 
            dist_const_is0_is0, tot_couples_train, tot_couples_test)
    store(fname_is0_avg, mae_is0_avg, inversion_is0_avg, plateau_is0_avg, 
            dist_const_is0_avg, tot_couples_train, tot_couples_test)
    store(fname_avg_is0, mae_avg_is0, inversion_avg_is0, plateau_avg_is0, 
            dist_const_avg_is0, tot_couples_train, tot_couples_test)
    store(fname_avg_avg, mae_avg_avg, inversion_avg_avg, plateau_avg_avg, 
            dist_const_avg_avg, tot_couples_train, tot_couples_test)

if __name__ == '__main__':
    main(sys.argv[1:])
