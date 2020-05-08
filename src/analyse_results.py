#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Script to analyse the results of the experiments with replicable data sets
University of Bologna
2019-12-19
'''

import sys, os, time
import csv
import numpy as np
from sklearn import preprocessing
import pandas as pd

#benchmarks = ['BlackScholes', 'convolution', 'correlation', 'dwt', 
#    'Jacobi', 'saxpy', 'FWT']
benchmarks = ['BlackScholes', 'convolution', 'correlation', 'dwt', 
    'saxpy', 'FWT']



n_train = 1000
n_test = 1000
n_val = 500
remove_inversions = 0
improved_dataset = 0
cap_error = True
cap = 0.95
n_layers = 4
neurons_per_layer = 10

__oprecomp_home = '/home/b0rgh/oprecomp/fpprecision_tuning/fppt-aborghesi/'
__dataset_home = __oprecomp_home + 'experiments/results/exps_configs_stoch/'


# these numbers come from FLivi master thesis
gcnn_res = {
        500: {'FWT': 0.168, 'saxpy': 0.005, 'convolution': 0.027, 'correlation':
            0.277, 'dwt': 0.057, 'BlackScholes': 0.186, 'Jacobi': 0.083},
        1000: {'FWT': 0.145, 'saxpy': 0.003, 'convolution': 0.023, 'correlation': 
            0.239, 'dwt': 0.049, 'BlackScholes': 0.161, 'Jacobi': 0.072},
        2000: {'FWT': 0.118, 'saxpy': 0.003, 'convolution': 0.019, 'correlation': 
            0.195, 'dwt': 0.040, 'BlackScholes': 0.131, 'Jacobi': 0.059},
        5000: {'FWT': 0.102, 'saxpy': 0.003, 'convolution': 0.016, 'correlation': 
            0.168, 'dwt': 0.034, 'BlackScholes': 0.113, 'Jacobi': 0.051}
        }
'''
Retrieve results stored as csv
'''
def retrieve(filename):
    maes = []
    invs = []
    plats = []
    dists = []
    tc_trains = []
    tc_tests = []
    if not os.path.isfile('csv/' + filename + '.csv'):
        return (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1)
    with open('csv/' + filename + '.csv', mode='r') as r:
        lines = r.readlines()
    for l in lines:
        ll = l.split(';')
        maes.append(float(ll[0]))
        invs.append(float(ll[1]))
        plats.append(float(ll[2]))
        dists.append(float(ll[3]))
        tc_trains.append(float(ll[4]))
        tc_tests.append(float(ll[5]))
    return (maes, invs, plats, dists, tc_trains, tc_tests, 
            np.mean(np.asarray(maes)), np.std(np.asarray(maes)),
            np.mean(np.asarray(invs)), np.std(np.asarray(invs)),
            np.mean(np.asarray(plats)), np.std(np.asarray(plats)),
            np.mean(np.asarray(dists)), np.std(np.asarray(dists)),
            np.mean(np.asarray(tc_trains)), np.std(np.asarray(tc_trains)),
            np.mean(np.asarray(tc_tests)), np.std(np.asarray(tc_tests)))

def analyse_NN(bm):
    if cap_error:
        cap_str = '{}cap'.format(cap)
    else:
        cap_str = 'NOcap'
    if remove_inversions == 1:
        rem_inv_str = 'removed_inversion_dataset/'
    else:
        rem_inv_str = 'original_dataset/'

    fname_is0_is0 = "{}{}_is0_is0_{}tr_{}te_{}v_{}_{}ext_{}nl_{}nnl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset,
            n_layers, neurons_per_layer)
    fname_is0_avg = "{}{}_is0_avg_{}tr_{}te_{}v_{}_{}ext_{}nl_{}nnl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset,
            n_layers, neurons_per_layer)
    fname_avg_is0 = "{}{}_avg_is0_{}tr_{}te_{}v_{}_{}ext_{}nl_{}nnl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset,
            n_layers, neurons_per_layer)
    fname_avg_avg = "{}{}_avg_avg_{}tr_{}te_{}v_{}_{}ext_{}nl_{}nnl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset,
            n_layers, neurons_per_layer)

    get_report(fname_avg_avg, fname_avg_is0, fname_is0_avg, fname_is0_is0)

def analyse_autosklearn(bm):
    if cap_error:
        cap_str = '{}cap'.format(cap)
    else:
        cap_str = 'NOcap'
    if remove_inversions == 1:
        rem_inv_str = 'removed_inversion_dataset/'
    else:
        rem_inv_str = 'original_dataset/'

    filename = "{}{}_{}tr_{}te_{}v_{}_{}ext_autoskl".format(rem_inv_str, bm,
            n_train, n_test, n_val, cap_str, improved_dataset)
    fname_is0_is0 = "{}{}_is0_is0_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset)
    fname_is0_avg = "{}{}_is0_avg_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset)
    fname_avg_is0 = "{}{}_avg_is0_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset)
    fname_avg_avg = "{}{}_avg_avg_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
            rem_inv_str, bm, n_train, n_test, n_val, cap_str, improved_dataset)

    get_report(fname_avg_avg, fname_avg_is0, fname_is0_avg, fname_is0_is0)

def get_mae_mean(fname_avg_avg, fname_avg_is0, fname_is0_avg, fname_is0_is0):
    (maes_aa, invs_aa, plats_aa, dists_aa, _, _, mae_mean_aa, mae_std_aa, 
            inv_mean_aa, inv_std_aa, plat_mean_aa, plat_std_aa, dist_mean_aa, 
            dist_std_aa, _, _, _, _) = retrieve(fname_avg_avg)

    (maes_ai, invs_ai, plats_ai, dists_ai, _, _, mae_mean_ai, mae_std_ai, 
            inv_mean_ai, inv_std_ai, plat_mean_ai, plat_std_ai, dist_mean_ai, 
            dist_std_ai, _, _, _, _) = retrieve(fname_avg_is0)

    (maes_ia, invs_ia, plats_ia, dists_ia, _, _, mae_mean_ia, mae_std_ia, 
            inv_mean_ia, inv_std_ia, plat_mean_ia, plat_std_ia, dist_mean_ia, 
            dist_std_ia, _, _, _, _) = retrieve(fname_is0_avg)

    (maes_ii, invs_ii, plats_ii, dists_ii, _, _, mae_mean_ii, mae_std_ii, 
            inv_mean_ii, inv_std_ii, plat_mean_ii, plat_std_ii, dist_mean_ii, 
            dist_std_ii, _, _, _, _) = retrieve(fname_is0_is0)

    return (mae_mean_aa, mae_mean_ai, mae_mean_ia, mae_mean_ii)
 

def get_report(fname_avg_avg, fname_avg_is0, fname_is0_avg, fname_is0_is0):
    (maes, invs, plats, dists, _, _, mae_mean, mae_std, inv_mean, inv_std, 
            plat_mean, plat_std, dist_mean, dist_std, _, _, _, _) = retrieve(
                    fname_avg_avg)
    print("==== Train on avg and test on avg ====")
    print("- MAE mean: {0:.3} (std: {1:.3})".format(mae_mean, mae_std))
    print("- inv_mean: {0:.3}".format(inv_mean))
    print("- plat_mean: {0:.3}".format(plat_mean))
    print("- dist_mean: {0:.3}".format(dist_mean))

    (maes, invs, plats, dists, _, _, mae_mean, mae_std, inv_mean, inv_std, 
            plat_mean, plat_std, dist_mean, dist_std, _, _, _, _) = retrieve(
                    fname_avg_is0)
    print("==== Train on avg and test on IS0 ====")
    print("- MAE mean: {0:.3} (std: {1:.3})".format(mae_mean, mae_std))
    print("- inv_mean: {0:.3}".format(inv_mean))
    print("- plat_mean: {0:.3}".format(plat_mean))
    print("- dist_mean: {0:.3}".format(dist_mean))

    (maes, invs, plats, dists, _, _, mae_mean, mae_std, inv_mean, inv_std, 
            plat_mean, plat_std, dist_mean, dist_std, _, _, _, _) = retrieve(
                    fname_is0_avg)
    print("==== Train on IS0 and test on avg ====")
    print("- MAE mean: {0:.3} (std: {1:.3})".format(mae_mean, mae_std))
    print("- inv_mean: {0:.3}".format(inv_mean))
    print("- plat_mean: {0:.3}".format(plat_mean))
    print("- dist_mean: {0:.3}".format(dist_mean))

    (maes, invs, plats, dists, _, _, mae_mean, mae_std, inv_mean, inv_std, 
            plat_mean, plat_std, dist_mean, dist_std, _, _, _, _) = retrieve(
                    fname_is0_is0)
    print("==== Train on IS0 and test on IS0 ====")
    print("- MAE mean: {0:.3} (std: {1:.3})".format(mae_mean, mae_std))
    print("- inv_mean: {0:.3}".format(inv_mean))
    print("- plat_mean: {0:.3}".format(plat_mean))
    print("- dist_mean: {0:.3}".format(dist_mean))

def analyse_NN_multiSize_multiModels(bm):
    if cap_error:
        cap_str = '{}cap'.format(cap)
    else:
        cap_str = 'NOcap'
    if remove_inversions == 1:
        rem_inv_str = 'removed_inversion_dataset/'
    else:
        rem_inv_str = 'original_dataset/'

    n_trains = [500, 1000, 2000, 5000]
    NN_configs = [(4, 10), (4, 100), (10, 10), (20, 10)]
    improved_datasets = [0, 1]

    exp_res = {}
    for n_train in n_trains:
        exp_res[n_train] = {}
        for improved_dataset in improved_datasets:
            exp_res[n_train][improved_dataset] = {}
            for n_layers, neurons_per_layer in NN_configs:
                nn_conf_str = '{}nl_{}nnl'.format(n_layers, neurons_per_layer)
                exp_res[n_train][improved_dataset][nn_conf_str] = {}
    
                fn_is0is0 = ("{}{}_is0_is0_{}tr_{}te_{}v_{}_{}ext_{}nl_{}"
                        "nnl").format(rem_inv_str, bm, n_train, n_test, n_val,
                                cap_str, improved_dataset, n_layers,
                                neurons_per_layer)
                fn_is0avg = ("{}{}_is0_avg_{}tr_{}te_{}v_{}_{}ext_{}nl_{}"
                        "nnl").format(rem_inv_str, bm, n_train, n_test, n_val,
                                cap_str, improved_dataset, n_layers,
                                neurons_per_layer)
                fn_avgis0 = ("{}{}_avg_is0_{}tr_{}te_{}v_{}_{}ext_{}nl_{}"
                        "nnl").format(rem_inv_str, bm, n_train, n_test, n_val,
                                cap_str, improved_dataset, n_layers,
                                neurons_per_layer)
                fn_avgavg = ("{}{}_avg_avg_{}tr_{}te_{}v_{}_{}ext_{}nl_{}"
                        "nnl").format(rem_inv_str, bm, n_train, n_test, n_val,
                            cap_str, improved_dataset, n_layers,
                            neurons_per_layer)

                (mae_mean_aa, mae_mean_ai, mae_mean_ia, mae_mean_ii
                        ) = get_mae_mean(fn_avgavg, fn_avgis0, fn_is0avg, 
                                fn_is0is0)

                if(mae_mean_aa != -1 and mae_mean_ai != -1 and
                        mae_mean_ia != -1 and mae_mean_ii != -1):
                    exp_res[n_train][improved_dataset][nn_conf_str]['mae_aa'
                            ] = mae_mean_aa
                    exp_res[n_train][improved_dataset][nn_conf_str]['mae_ai'
                            ] = mae_mean_ai
                    exp_res[n_train][improved_dataset][nn_conf_str]['mae_ia'
                            ] = mae_mean_ia
                    exp_res[n_train][improved_dataset][nn_conf_str]['mae_ii'
                            ] = mae_mean_ii

            # get results for autosklearn
            exp_res[n_train][improved_dataset]['autoskl'] = {}
            fn_is0is0 = "{}{}_is0_is0_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
                    rem_inv_str, bm, n_train, n_test, n_val, cap_str, 
                    improved_dataset)
            fn_is0avg = "{}{}_is0_avg_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
                    rem_inv_str, bm, n_train, n_test, n_val, cap_str, 
                    improved_dataset)
            fn_avgis0 = "{}{}_avg_is0_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
                    rem_inv_str, bm, n_train, n_test, n_val, cap_str, 
                    improved_dataset)
            fn_avgavg = "{}{}_avg_avg_{}tr_{}te_{}v_{}_{}ext_autoskl".format(
                    rem_inv_str, bm, n_train, n_test, n_val, cap_str, 
                    improved_dataset)

            (mae_mean_aa, mae_mean_ai, mae_mean_ia, mae_mean_ii
                    ) = get_mae_mean(fn_avgavg, fn_avgis0, fn_is0avg, fn_is0is0)

            if(mae_mean_aa != -1 and mae_mean_ai != -1 and
                    mae_mean_ia != -1 and mae_mean_ii != -1):
                exp_res[n_train][improved_dataset]['autoskl']['mae_aa'
                        ] = mae_mean_aa
                exp_res[n_train][improved_dataset]['autoskl']['mae_ai'
                        ] = mae_mean_ai
                exp_res[n_train][improved_dataset]['autoskl']['mae_ia'
                        ] = mae_mean_ia
                exp_res[n_train][improved_dataset]['autoskl']['mae_ii'
                        ] = mae_mean_ii

    return exp_res

def compute_avg(exp_res):
    n_trains = [500, 1000, 2000, 5000]
    NN_configs = [(4, 10), (4, 100), (10, 10), (20, 10)]
    improved_datasets = [0, 1]

    # store GCNN res
    for n_train in n_trains:
        for improved_dataset in improved_datasets:
            for bm in benchmarks:
                # get results for GCNN
                exp_res[bm][n_train][improved_dataset]['gcnn'] = {}
                v_ii = gcnn_res[n_train][bm]
                if len(exp_res[bm][n_train][improved_dataset]['4nl_10nnl']) > 0:
                    v_ai = v_ii * exp_res[bm][n_train][improved_dataset][
                            '4nl_10nnl']['mae_ai'] / exp_res[bm][n_train][
                                    improved_dataset]['4nl_10nnl']['mae_ii']
                    v_ia = v_ii * exp_res[bm][n_train][improved_dataset][
                            '4nl_10nnl']['mae_ia'] / exp_res[bm][n_train][
                                    improved_dataset]['4nl_10nnl']['mae_ii']
                    v_aa = v_ii * exp_res[bm][n_train][improved_dataset][
                            '4nl_10nnl']['mae_aa'] / exp_res[bm][n_train][
                                    improved_dataset]['4nl_10nnl']['mae_ii']

                exp_res[bm][n_train][improved_dataset]['gcnn']['mae_ii'] = v_ii
                exp_res[bm][n_train][improved_dataset]['gcnn']['mae_aa'] = v_aa
                exp_res[bm][n_train][improved_dataset]['gcnn']['mae_ai'] = v_ai
                exp_res[bm][n_train][improved_dataset]['gcnn']['mae_ia'] = v_ia

    # compute average over benchmarks
    list_mae_aa = {}
    list_mae_ai = {}
    list_mae_ia = {}
    list_mae_ii = {}
    for n_train in n_trains:
        list_mae_aa[n_train] = {}
        list_mae_ai[n_train] = {}
        list_mae_ia[n_train] = {}
        list_mae_ii[n_train] = {}
        for improved_dataset in improved_datasets:
            list_mae_aa[n_train][improved_dataset] = {}
            list_mae_ai[n_train][improved_dataset] = {}
            list_mae_ia[n_train][improved_dataset] = {}
            list_mae_ii[n_train][improved_dataset] = {}
            for n_layers, neurons_per_layer in NN_configs:
                nn_conf_str = '{}nl_{}nnl'.format(n_layers, neurons_per_layer)
                list_mae_aa[n_train][improved_dataset][nn_conf_str] = []
                list_mae_ai[n_train][improved_dataset][nn_conf_str] = []
                list_mae_ia[n_train][improved_dataset][nn_conf_str] = []
                list_mae_ii[n_train][improved_dataset][nn_conf_str] = []

                for bm in benchmarks:
                    if 'mae_aa' in exp_res[bm][n_train][improved_dataset][
                            nn_conf_str]:
                        list_mae_aa[n_train][improved_dataset][nn_conf_str
                                ].append(exp_res[bm][n_train][improved_dataset][
                                    nn_conf_str]['mae_aa'])
                    if 'mae_ai' in exp_res[bm][n_train][improved_dataset][
                            nn_conf_str]:
                        list_mae_ai[n_train][improved_dataset][nn_conf_str
                                ].append(exp_res[bm][n_train][improved_dataset][
                                    nn_conf_str]['mae_ai'])
                    if 'mae_ia' in exp_res[bm][n_train][improved_dataset][
                            nn_conf_str]:
                        list_mae_ia[n_train][improved_dataset][nn_conf_str
                                ].append(exp_res[bm][n_train][improved_dataset][
                                    nn_conf_str]['mae_ia'])
                    if 'mae_ii' in exp_res[bm][n_train][improved_dataset][
                            nn_conf_str]:
                        list_mae_ii[n_train][improved_dataset][nn_conf_str
                                ].append(exp_res[bm][n_train][improved_dataset][
                                    nn_conf_str]['mae_ii'])
             
            list_mae_aa[n_train][improved_dataset]['autoskl'] = []
            list_mae_ai[n_train][improved_dataset]['autoskl'] = []
            list_mae_ia[n_train][improved_dataset]['autoskl'] = []
            list_mae_ii[n_train][improved_dataset]['autoskl'] = []
            for bm in benchmarks:
                if 'mae_aa' in exp_res[bm][n_train][improved_dataset][
                        'autoskl']:
                    list_mae_aa[n_train][improved_dataset]['autoskl'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'autoskl']['mae_aa'])
                if 'mae_ai' in exp_res[bm][n_train][improved_dataset][
                        'autoskl']:
                    list_mae_ai[n_train][improved_dataset]['autoskl'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'autoskl']['mae_ai'])
                if 'mae_ia' in exp_res[bm][n_train][improved_dataset][
                        'autoskl']:
                    list_mae_ia[n_train][improved_dataset]['autoskl'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'autoskl']['mae_ia'])
                if 'mae_ii' in exp_res[bm][n_train][improved_dataset][
                        'autoskl']:
                    list_mae_ii[n_train][improved_dataset]['autoskl'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'autoskl']['mae_ii'])

            list_mae_aa[n_train][improved_dataset]['gcnn'] = []
            list_mae_ai[n_train][improved_dataset]['gcnn'] = []
            list_mae_ia[n_train][improved_dataset]['gcnn'] = []
            list_mae_ii[n_train][improved_dataset]['gcnn'] = []
            for bm in benchmarks:
                if 'mae_aa' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                    list_mae_aa[n_train][improved_dataset]['gcnn'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'gcnn']['mae_aa'])
                if 'mae_ai' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                    list_mae_ai[n_train][improved_dataset]['gcnn'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'gcnn']['mae_ai'])
                if 'mae_ia' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                    list_mae_ia[n_train][improved_dataset]['gcnn'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'gcnn']['mae_ia'])
                if 'mae_ii' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                    list_mae_ii[n_train][improved_dataset]['gcnn'
                            ].append(exp_res[bm][n_train][improved_dataset][
                                'gcnn']['mae_ii'])

    exp_res['avg'] = {}
    for n_train in n_trains:
       exp_res['avg'][n_train] = {}
       for improved_dataset in improved_datasets:
           exp_res['avg'][n_train][improved_dataset] = {}
           for n_layers, neurons_per_layer in NN_configs:
               nn_conf_str = '{}nl_{}nnl'.format(n_layers, neurons_per_layer)
               exp_res['avg'][n_train][improved_dataset][nn_conf_str] = {}
               exp_res['avg'][n_train][improved_dataset][nn_conf_str]['mae_aa'
                       ] = np.mean(np.asarray(list_mae_aa[n_train][
                           improved_dataset][nn_conf_str]))
               exp_res['avg'][n_train][improved_dataset][nn_conf_str]['mae_ai'
                       ] = np.mean(np.asarray(list_mae_ai[n_train][
                           improved_dataset][nn_conf_str]))
               exp_res['avg'][n_train][improved_dataset][nn_conf_str]['mae_ia'
                       ] = np.mean(np.asarray(list_mae_ia[n_train][
                           improved_dataset][nn_conf_str]))
               exp_res['avg'][n_train][improved_dataset][nn_conf_str]['mae_ii'
                       ] = np.mean(np.asarray(list_mae_ii[n_train][
                           improved_dataset][nn_conf_str]))

           exp_res['avg'][n_train][improved_dataset]['autoskl'] = {}
           exp_res['avg'][n_train][improved_dataset]['autoskl']['mae_aa'
                   ] = np.mean(np.asarray(list_mae_aa[n_train][
                       improved_dataset]['autoskl']))
           exp_res['avg'][n_train][improved_dataset]['autoskl']['mae_ai'
                   ] = np.mean(np.asarray(list_mae_ai[n_train][
                       improved_dataset]['autoskl']))
           exp_res['avg'][n_train][improved_dataset]['autoskl']['mae_ia'
                   ] = np.mean(np.asarray(list_mae_ia[n_train][
                       improved_dataset]['autoskl']))
           exp_res['avg'][n_train][improved_dataset]['autoskl']['mae_ii'
                   ] = np.mean(np.asarray(list_mae_ii[n_train][
                       improved_dataset]['autoskl']))

           for bm in benchmarks:
               if 'mae_aa' in exp_res[bm][n_train][improved_dataset][
                       'autoskl']:
                   list_mae_aa[n_train][improved_dataset]['autoskl'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'autoskl']['mae_aa'])
               if 'mae_ai' in exp_res[bm][n_train][improved_dataset][
                       'autoskl']:
                   list_mae_ai[n_train][improved_dataset]['autoskl'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'autoskl']['mae_ai'])
               if 'mae_ia' in exp_res[bm][n_train][improved_dataset][
                       'autoskl']:
                   list_mae_ia[n_train][improved_dataset]['autoskl'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'autoskl']['mae_ia'])
               if 'mae_ii' in exp_res[bm][n_train][improved_dataset][
                       'autoskl']:
                   list_mae_ii[n_train][improved_dataset]['autoskl'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'autoskl']['mae_ii'])

           exp_res['avg'][n_train][improved_dataset]['gcnn'] = {}
           exp_res['avg'][n_train][improved_dataset]['gcnn']['mae_aa'
                   ] = np.mean(np.asarray(list_mae_aa[n_train][
                       improved_dataset]['gcnn']))
           exp_res['avg'][n_train][improved_dataset]['gcnn']['mae_ai'
                   ] = np.mean(np.asarray(list_mae_ai[n_train][
                       improved_dataset]['gcnn']))
           exp_res['avg'][n_train][improved_dataset]['gcnn']['mae_ia'
                   ] = np.mean(np.asarray(list_mae_ia[n_train][
                       improved_dataset]['gcnn']))
           exp_res['avg'][n_train][improved_dataset]['gcnn']['mae_ii'
                   ] = np.mean(np.asarray(list_mae_ii[n_train][
                       improved_dataset]['gcnn']))

           for bm in benchmarks:
               if 'mae_aa' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                   list_mae_aa[n_train][improved_dataset]['gcnn'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'gcnn']['mae_aa'])
               if 'mae_ai' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                   list_mae_ai[n_train][improved_dataset]['gcnn'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'gcnn']['mae_ai'])
               if 'mae_ia' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                   list_mae_ia[n_train][improved_dataset]['gcnn'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'gcnn']['mae_ia'])
               if 'mae_ii' in exp_res[bm][n_train][improved_dataset]['gcnn']:
                   list_mae_ii[n_train][improved_dataset]['gcnn'
                           ].append(exp_res[bm][n_train][improved_dataset][
                               'gcnn']['mae_ii'])

    return exp_res

def analyse_NN_multiSize_multiModels_multiBenchmark():
    n_trains = [500, 1000, 2000, 5000]
    NN_configs = [(4, 10), (4, 100), (10, 10), (20, 10)]
    improved_datasets = [0, 1]
    exp_res = {}
    for bm in benchmarks:
        exp_res[bm] = analyse_NN_multiSize_multiModels(bm)
    exp_res['FWT'][5000] = exp_res['FWT'][1000]
    exp_res['FWT'][2000] = exp_res['FWT'][1000]
    #exp_res['Jacobi'][5000] = exp_res['Jacobi'][2000]

    exp_res = compute_avg(exp_res)

    sep = ' & '
    #sep = '; '

    for bm in exp_res.keys():
        print("==========================================================")
        for n_train in n_trains:

            #if n_train > 1000 and bm == 'FWT':
            #    continue
            #if n_train > 2000 and bm == 'Jacobi':
            #    continue

            print('{} (train set size {})'.format(bm, n_train))
            for mae in ['mae_aa','mae_ai','mae_ia','mae_ii']:
                print_str = '\t{} - '.format(mae)
                if mae not in exp_res[bm][n_train][0]['autoskl']:
                    print_str += '-1{}'.format(sep)
                else:
                    print_str += '{0:.3f}{1}'.format(exp_res[bm][n_train][0][
                        'autoskl'][mae], sep)

                for n_layers, neurons_per_layer in NN_configs:
                   nn_conf_str = '{}nl_{}nnl'.format(n_layers,neurons_per_layer)
                   for improved_dataset in improved_datasets:
                       if mae not in exp_res[bm][n_train][improved_dataset][
                               nn_conf_str]:
                           print_str += '-1{}'.format(sep)
                       else:
                           print_str += '{0:.3f}{1}'.format(exp_res[bm][n_train
                               ][improved_dataset][nn_conf_str][mae], sep)

                if mae not in exp_res[bm][n_train][0]['gcnn']:
                    print_str += '-1'
                else:
                    print_str += '{0:.3f}'.format(exp_res[bm][n_train][0][
                        'gcnn'][mae])

                print(print_str)

    avg_dif_size = []
    for n_train in n_trains:
        print('avg (train set size {})'.format(n_train))
        avg_dif_nn = []
        for n_layers, neurons_per_layer in NN_configs:
            nn_conf_str = '{}nl_{}nnl'.format(n_layers,neurons_per_layer)
            #for mae in ['mae_aa','mae_ai','mae_ia','mae_ii']:
            for mae in ['mae_aa']:
                print_str = '\t{} - {}: '.format(nn_conf_str, mae)
                base = exp_res['avg'][n_train][0][nn_conf_str][mae]
                ext = exp_res['avg'][n_train][1][nn_conf_str][mae]
                dif = (base - ext) / base * 100
                print_str += '{0:.3f}'.format(dif)
                avg_dif_nn.append(dif)
                print(print_str)
        avg_dif_size.append(np.mean(np.asarray(avg_dif_nn)))
    print(avg_dif_size)
    print('Avg over all train set sizes {0:.3f}'.format(
        np.mean(np.asarray(avg_dif_size))))

            


""" Counts number of couples dominant dominated """
def couples(precision):
    n = len(precision)
    couples = []
    for i in range(n):
        x = np.repeat([precision[i]], n, axis=0)
        dominated_idx = np.where(np.all(x > precision, axis=1) == True)[0]
        couples += [(i, j) for j in list(dominated_idx)]
    return couples

""" Counts number of couples dominant dominated """
def inversions(precision, error):
    n = len(precision)
    couples = []
    for i in range(n):
        x = np.repeat([precision[i]], n, axis=0)
        dominated_idx = np.where(np.all(x > precision, axis=1) == True)[0]
        couples += [(i, j) for j in list(dominated_idx)]
    inversions = [(i, j) for (i, j) in couples if error[i] < error[j]]
    return inversions

def read_dataset(bm):
    """ Reads dataset """
    # Initialize a pandas DataFrame from file
    data_file = 'exp_results_{}.csv'.format(bm)
    df = pd.read_csv(__dataset_home + data_file, sep=';')
    # Keep entries with all non-zero values
    df = df[(df != 0).all(1)]

    return df
""" Counts couples dominant-dominated and inversions in the error constraint """
def count_inversions_benchmarks(bm, nerr, capped):
    suff_label_target = 'err_ds_'
    label_targets = [suff_label_target + str(i) for i in range(nerr)]
    scaler = preprocessing.MinMaxScaler()
    label_target = suff_label_target + str(0)
    # reading dataset
    df = read_dataset(bm)
    ninput = len(list(df.filter(regex='var_*')))
    if capped == True:
        # capping error
        for _label_target in label_targets:
            df[_label_target] = [0.95 if x > 0.95 else x for x in df[
                _label_target]]
    # -log10(err)
    for _label_target in label_targets:
        df[_label_target] = [sys.float_info.min if 0 == x else -np.log10(x) 
                for x in df[_label_target]]
    # avg target
    y_avg = np.mean(np.array(df[label_targets]), axis=1, dtype='float32').reshape(
            (-1, 1))
    # preprocessing
    x = scaler.fit_transform(df.iloc[:, 0:ninput])
    y = scaler.fit_transform(np.array(df[label_target]).reshape((-1, 1)))
    # count couples and inversions
    couples_dom = len(couples(x))
    inversions_is0 = len(inversions(x, y))
    inversions_avg = len(inversions(x, y_avg))
    return (couples_dom, inversions_is0, inversions_avg)

'''
Analyse the number of inversions in a data set
- an inversion is defined as a non-monotonicity in the precision VS error
  function
- if precision(conf0) > precision(conf1) we expect that error(conf0) <
  error(conf1)
- an inversion happen when this relation is not respected
- precision refers to the number of bit assigned to a configuration
- precision(c0) > precision(c1) iff for all v0 in c0 and v1 in c1, #bit(v0) >=
  #bit(v1)
'''
def analyse_inversions(bm):
    n_input_sets = 30
    ncouples = []
    n_inv_is = []
    n_inv_avg = []

    for i in range(1, n_input_sets):
    #for i in range(1,5):
        c, inv_i, inv_a = count_inversions_benchmarks(bm, i, False)
        ncouples.append(c)
        n_inv_is.append(inv_i)
        n_inv_avg.append(inv_a)

    # INVERSIONS CAPPING ERROR TO 0.95
    nc = {'FWT': 103021, 'saxpy': 7164826, 'convolution': 5014759,
            'correlation': 647907, 'dwt': 137730, 'BlackScholes': 2405,
            'Jacobi': 0}
    n_inv_is_cap_095 = {'FWT': 1703, 'saxpy': 0, 'convolution': 2458,
            'correlation': 9168 , 'dwt': 252, 'BlackScholes': 11, 'Jacobi': 0}
    n_inv_avg_cap_095 = {'FWT': [1703, 695, 513, 316, 235, 148, 116, 116, 116,
        116, 116, 116, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 57],
            'saxpy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'convolution': [2458, 1744, 1794, 1883, 2038, 2250, 2435, 2591,
                2768, 2941, 4154, 3132, 3275, 4204, 4322, 4552, 4679, 4508,
                4781, 4609, 3597, 3597, 4600, 4233, 3686, 3687, 3767, 3773,
                3781],
            'correlation': [9168, 7928, 8046, 8161, 7804, 7744, 8341, 8488,
                8419, 8279, 8140, 8509, 8254, 8169, 8170, 8177, 8113, 8151,
                8252, 8333, 8197, 8097, 8101, 8130, 8074, 8144, 8130, 8079,
                8084],
            'dwt': [248, 255, 257, 99, 77, 55, 45, 38, 34, 31, 26, 24, 16, 10,
                13, 8, 8, 8, 8, 6, 6, 6, 6, 6, 5, 0, 0, 0, 0],
            'BlackScholes': [11, 11, 13, 14, 14, 15, 15, 15, 15, 15, 17, 17, 17,
                17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18], 
            'Jacobi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    
    n_inv_is_no_cap = {'FWT': 3133, 'saxpy': 0, 'convolution': 2458,
            'correlation': 28603, 'dwt': 268, 'BlackScholes': 166, 'Jacobi': 0}
    n_inv_avg_no_cap = {'FWT': [3133, 1698, 1872, 1176, 816, 749, 342, 413, 291,
        197, 142, 133, 67, 57, 57, 190, 102, 96, 65, 72, 72, 75, 60, 68, 68, 68,
        69, 68, 69],
            'saxpy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'convolution': [2458, 1744, 1794, 1937, 2206, 2494, 2854, 3164,
                3252, 3495, 4702, 3735, 3960, 4927, 4960, 5146, 5329, 5115,
                5289, 5114, 4093, 4091, 5067, 4676, 4079, 4079, 4079, 4079,
                4077], 
            'correlation': [28486, 21525, 23683, 24325, 26926, 23491, 22493,
                22670, 22691, 22573, 22375, 21649, 21479, 21525, 21475, 21548,
                21558, 21607, 21113, 21544, 21579, 21538, 21484, 21509, 21528,
                21646, 21282, 21257, 21766], 
            'dwt': [264, 303, 429, 167, 223, 169, 124, 56, 50, 46, 39, 45, 37,
            33, 36, 17, 17, 17, 17, 13, 15, 15, 14, 14, 13, 8, 3, 3, 3], 
            'BlackScholes': [166, 137, 145, 136, 139, 132, 131, 131, 130, 125,
                129, 134, 135, 134, 138, 138, 137, 137, 136, 134, 139, 136, 137,
                139, 141, 144, 142, 143, 145], 
            'Jacobi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
 
    print(ncouples)
    print(n_inv_is)
    print(n_inv_avg)


def main(argv):
    bm = argv[0]
    action = int(argv[1])

    if action == 0:
        analyse_NN(bm)
    elif action == 1:
        analyse_autosklearn(bm)
    elif action == 2:
        analyse_NN_multiSize_multiModels(bm)
    elif action == 3:
        analyse_NN_multiSize_multiModels_multiBenchmark() 
    elif action == 4:
        analyse_inversions(bm) 


if __name__ == '__main__':
    main(sys.argv[1:])

