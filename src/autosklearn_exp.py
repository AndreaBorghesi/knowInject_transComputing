'''
    Script to experiment with autosklearn 
'''
import extract_dep_graphs_from_json as ed
import sys, os, time, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import scipy.sparse as sp
import seaborn as seabornInstance 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib.pyplot import ylim, plot
from mkl_random import bench
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import autosklearn.regression

N_INPUTSETS = 30
BATCH_SIZE = 32
BENCHMARKS = ["dwt", "BlackScholes", "convolution", "correlation","jacobi"]
res_dir = "./results/"

def log_error(error):
    return -np.log(error)

def preprocess_all_data(dataset, del_big_errors, improved):
    res = [] 
    for i in range(N_INPUTSETS):
        res.append(preprocess_input_set(dataset, del_big_errors, i, improved))
    res.append(preprocess_input_set(dataset, del_big_errors, "all", improved))
    return res

def extract_bench_name(dataset): 
    for benchmark in BENCHMARKS:
        if benchmark in dataset:
            return benchmark 
       
def preprocess_input_set(dataset, del_big_errors, input_set, improved):
    
    #Read csv file
    df = pd.read_csv(dataset, sep=";")
    
    #Remove useless err_mean and err_std columns
    df = df.drop('err_std', axis=1)
    df = df.drop('err_mean', axis=1)
    
    err_column=""
    
    #Two cases now:
    if input_set!="all": #1) Predict error in a given inputset   
        df = df.drop([
            col for col in df if col.startswith('err') 
            and col!="err_ds_{}".format(
                input_set)], axis=1)
        
        #Delete rows with big errors if del_big_errors == 1
        to_be_deleted = []
        if del_big_errors==1:
            for index, row in df.iterrows():
                if row['err_ds_{}'.format(input_set)]>0.95:
                    to_be_deleted.append(index)         
            df = df.drop(to_be_deleted, axis=0)
        err_column = 'err_ds_{}'.format(input_set)
        
    else: #2) Predict mean error in all the inputsets 
        to_be_deleted = []
        if del_big_errors==1:
            for index, row in df.iterrows():
                for i in range(N_INPUTSETS):
                    if row['err_ds_{}'.format(i)]>0.95:
                        to_be_deleted.append(index)
                        break
            df = df.drop(to_be_deleted, axis=0)    
        
        #Add row mean_err_all_ds
        err_columns = [col for col in df if col.startswith('err')]
        df['mean_err_all_ds'] = df[err_columns].mean(axis=1)      
        df = df.drop(err_columns, axis=1)   
        err_column = 'mean_err_all_ds'  

    if improved==1:
        add_features = ed.getAdditionalFeatures(extract_bench_name(dataset))
        for add_feat in add_features:
            df['{}-{}'.format(
                add_feat[0], add_feat[1])]=df[add_feat[0]]-df[add_feat[1]]
        #Arrange the df with err column at the end
        df_temp = df.pop(err_column)
        df[err_column]=df_temp
    
    #Transform errors in -log(error)
    var_columns=[]
    for col in df.columns:
        if "var" in col: 
            var_columns.append(col)
        else:
            df[col] = df[col].apply(log_error)  

    #Delete rows with nan, inf, -inf or 0.0
    df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]    
    if df.empty:
        return "", 0
    
    #Normalize the variables (input features)
    n_scaler = preprocessing.MinMaxScaler()
    n_scaler_df = n_scaler.fit_transform(df[var_columns])
    n_scaler_df = pd.DataFrame(n_scaler_df, columns=var_columns)
    
    #Standardize the errors
    s_scaler = preprocessing.StandardScaler()
    s_scaler_df = s_scaler.fit_transform(df[err_column].values.reshape(-1,1))
    s_scaler_df = pd.DataFrame(s_scaler_df, columns=[err_column])
     
    #Resulting preprocessed training set
    df = pd.concat([n_scaler_df, s_scaler_df], axis=1)
    
    #Output to new csv
    #df.to_csv(datasetPath+"test.csv",index=False, sep=";")
    return [df, s_scaler]

def evaluate(y_test, y_pred, s_scaler, errors):
    std_mae = metrics.mean_absolute_error(y_test, y_pred)
    abs_y_test = s_scaler.inverse_transform(y_test)
    abs_y_pred = s_scaler.inverse_transform(y_pred)
    abs_mae = metrics.mean_absolute_error(abs_y_test, abs_y_pred)
    abs_errors = s_scaler.inverse_transform(errors)
    abs_mean = np.mean(abs_errors)
    abs_var = np.var(abs_errors)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2_score = metrics.r2_score(y_test, y_pred)
    return std_mae, abs_mae, abs_mean, abs_var, rmse, r2_score

def autosk_regr(dataset, s_scaler, verbose=False):
    #Split the dataset into attributes (X) and labels (Y)

    var_columns = [col for col in dataset if col.startswith('var')]
    err_column = [col for col in dataset.columns if col not in var_columns]
    
    x = dataset[var_columns].values    
    y = dataset[err_column].values
    
    #80% of the dataset is our training set while 20% is our test set 
    x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=0)
    
    n_samples, n_attributes = x_train.shape

    automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            )
    automl.fit(x_train, y_train)

    model_train_time = time.time()
    
    y_pred = automl.predict(x_test)
    
    std_mae, abs_mae, abs_mean, abs_var, rmse, r2_score = evaluate(
            y_test, y_pred, s_scaler, dataset[err_column])

    return std_mae, abs_mae, abs_mean, abs_var, rmse, r2_score, automl

def perform_exp(dataset_path, benchmark):
    for j in range(2):
        improved = j
        print('Improved {}'.format(improved))
        for k in range(2):
            del_big_errors = k
            print('\tDeleting big errors {}'.format(del_big_errors))
            proc_data = preprocess_all_data(dataset_path, del_big_errors,
                    improved)
            for i in range(N_INPUTSETS+1):
                if i == N_INPUTSETS:
                    input_set == 'all'
                else:
                    input_set = i
            
                if(input_set=="all"):
                    proc_dataset = proc_data[N_INPUTSETS][0]
                    s_scaler = proc_data[N_INPUTSETS][1]
                else:
                    proc_dataset = proc_data[int(input_set)][0]
                    s_scaler = proc_data[int(input_set)][1]
                    
                (std_mae, abs_mae, abs_mean, abs_var, rmse, r2_score, 
                        _) = autosk_regr(proc_dataset, s_scaler, True)   
                
                print("\t\t{};{};{};{};{};{};{}".format(input_set, abs_mae, 
                    std_mae, abs_mean, abs_var, rmse, r2_score))

                res_file = res_dir + '{}_{}dataAug_{}delBigEr_autoskl'.format(
                        benchmark, improved, del_big_errors)
                with open(res_file, 'a') as rf:
                    rf.write("{};{};{};{};{};{};{}".format(input_set, abs_mae, 
                    std_mae, abs_mean, abs_var, rmse, r2_score))

def parse_result(benchmark):
    res_autosklearn = {}

    print('===================== {} ========================'.format(benchmark))
    print('------------- Skip errors > 0.95 -------------')
    res_file = res_dir + '{}_0dataAug_1delBigEr_autoskl'.format(
            benchmark)
    maes_skip = []
    maes_norm_skip = []
    with open(res_file, 'r') as rf:
        lines = rf.readlines()
    for l in lines:
        vals = l.split(';')
        if vals[0] == 'all':
            input_set = 'all'
        else:
            input_set = int(vals[0])
        abs_mae = float(vals[1])
        std_mae = float(vals[2])
        abs_mean = float(vals[3])
        abs_var = float(vals[4])
        rmse = float(vals[5])
        r2_score = float(vals[6])
        maes_skip.append(abs_mae)
        maes_norm_skip.append(abs_mae/abs_mean)
        #print("{0}: MAE {1:.3f} MAE norm {2:.3f} (abs mean {3:.3f})".format(
        #    input_set, abs_mae, abs_mae/abs_mean, abs_mean))
    print("Avg.: MAE {0:.4f} MAE norm {1:.3f}".format(
        np.mean(np.asarray(maes_skip)),
        abs(np.mean(np.asarray(maes_norm_skip)))))

    print('----------------- All errors -----------------')
    res_file = res_dir + '{}_0dataAug_0delBigEr_autoskl'.format(
            benchmark)
    maes_allErr = []
    maes_norm_allErr = []
    with open(res_file, 'r') as rf:
        lines = rf.readlines()
    for l in lines:
        vals = l.split(';')
        if vals[0] == 'all':
            input_set = 'all'
        else:
            input_set = int(vals[0])
        abs_mae = float(vals[1])
        std_mae = float(vals[2])
        abs_mean = float(vals[3])
        abs_var = float(vals[4])
        rmse = float(vals[5])
        r2_score = float(vals[6])
        maes_allErr.append(abs_mae)
        maes_norm_allErr.append(abs_mae/abs_mean)
        print("{0}: MAE {1:.3f} MAE norm {2:.3f} (abs mean {3:.3f})".format(
            input_set, abs_mae, abs_mae/abs_mean, abs_mean))
    print("Avg.: MAE {0:.4f} MAE norm {1:.3f}".format(
        np.mean(np.asarray(maes_allErr)),
        abs(np.mean(np.asarray(maes_norm_allErr)))))

    print('============================================================')



def main(argv):
    dataset_path = argv[0]
    if dataset_path == 'parse':
        benchmark = argv[1]
        parse_result(benchmark)
    else:
        benchmark = extract_bench_name(dataset_path)
        perform_exp(dataset_path, benchmark)

if __name__ == '__main__':
    main(sys.argv[1:])
