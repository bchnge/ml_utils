import pandas as pd
import os
import random
import numpy as np
from pprint import pprint
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc, make_scorer
from matplotlib import pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import requests
import json

data_path = '/vol2/competitions/home-credit-default-risk/'

def notify_ifttt(data = '', config_path = '/projects/ifttt/code_complete.json'):   
    with open(config_path, 'r') as f:
        config_jsn = json.load(f)
    ifttt_notification_url = 'https://maker.ifttt.com/trigger/' + config_jsn.get('name') + '/with/key/' + config_jsn.get('ifttt_key')
    payload = {'value1': data}
    requests.post(ifttt_notification_url, data=payload)
    
def get_feature_grouping(X):
    pca = PCA().fit(X.dropna())
    
    num_groups, num_features = pca.components_.shape
    print(str(num_groups) + ' groups...')
    print(str(num_features) + ' features...')
    
    # Calculate factor loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    abs_loadings = np.abs(loadings)
    variable_groupings = abs_loadings.argmax(axis = 1)

    groupings = {}
    for g in range(variable_groupings.min(), variable_groupings.max()):
        groupings[g] = [X.columns[i] for i,x in enumerate(variable_groupings) if x == g]
    return(groupings)

def agg_supplement_old(supp_df, func):
    supp = supp_df \
            .loc[:,(supp_df.dtypes != object) & (supp_df.columns != 'SK_ID_PREV')] \
            .groupby('SK_ID_CURR') \
            .agg(func) \
            .reset_index()
    return(supp)

def agg_supplement(supp_df):
    # Numerics
    supp_numeric_mean = supp_df \
            .loc[:,(supp_df.dtypes != object) & (supp_df.columns != 'SK_ID_PREV')] \
            .groupby('SK_ID_CURR') \
            .mean() \
            .reset_index()
    supp_numeric_min = supp_df \
            .loc[:,(supp_df.dtypes != object) & (supp_df.columns != 'SK_ID_PREV')] \
            .groupby('SK_ID_CURR') \
            .min() \
            .reset_index()
    supp_numeric_max = supp_df \
            .loc[:,(supp_df.dtypes != object) & (supp_df.columns != 'SK_ID_PREV')] \
            .groupby('SK_ID_CURR') \
            .max() \
            .reset_index()
    supp = supp_numeric_mean \
        .merge(supp_numeric_max, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_mean', '_max')) \
        .merge(supp_numeric_min, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_max', '_min'))

    # Categorical
    objects = supp_df.columns[(supp_df.dtypes == object) & (supp_df.columns != 'SK_ID_PREV')]
    if len(objects) > 0 :
        tmp = pd.concat([pd.get_dummies(supp_df[o], dummy_na = True, prefix = o) for o in objects], axis = 1)
        tmp['SK_ID_CURR'] = supp_df.SK_ID_CURR
        supp_object_max = tmp \
            .groupby('SK_ID_CURR') \
            .max().reset_index()
        supp_object_min = tmp \
            .groupby('SK_ID_CURR') \
            .min().reset_index()
        supp_object_mean = tmp \
            .groupby('SK_ID_CURR') \
            .mean().reset_index()

        supp = supp \
        .merge(supp_object_mean, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_min', '_mean')) \
        .merge(supp_object_max, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_mean', '_max')) \
        .merge(supp_object_min, how = 'inner', on = 'SK_ID_CURR', suffixes = ('_max', '_min'))
    
    return(supp)


def test_supplement(supplement_path):
    supp_df = load_supplement(supplement_path)
    results = {}
    for f in ['max', 'mean', 'min']:
        print(f + ':')
        supp = agg_supplement(supp_df, f)
        supp_cols = [x for x in supp.columns if x not in ['SK_ID_CURR', 'TARGET']]
        supp_x, supp_y = utils.get_design_matrix(supp, supp_cols)
        scores = eval_model(supp_x, supp_y)
        print(scores)
        print((scores.mean(), scores.std()))
        print('\n')
        results[f] = (scores.mean(), scores.std())
    return(results)

def load_data(train = True, supp_dict = None):
    ''' 
    Create applications dataframe. 

        Parameters:
        train (True/False): whether or not the applications are in the training set
        supp_dict: dictionary of (supplement_file_name : supplement_aggregation)
    '''
    
    if train:
        print('Loading training applications')
        df = pd.read_csv(data_path + 'application_train.csv.zip')
    else:
        print('Loading test applications')
        df = pd.read_csv(data_path + 'application_test.csv.zip')
    
    if supp_dict:
        supp_idx = 1
        for supp_name in supp_dict.keys():
            print('Loading ' + supp_name)
            supp = agg_supplement(pd.read_csv(data_path + supp_name))                      
            df = df.merge(supp, how = 'left', on = 'SK_ID_CURR', 
                          suffixes = ('', '_'+str(supp_idx)))
            supp_idx += 1
    
    return(df)   

def autoencode_dataframe(df):
    ''' given a dataframe with numerics and objects, convert objects into dummies'''
    objects = df.columns[(df.dtypes == object)]        
    # Convert categorical to dummy
    for o in objects:
        df = pd.concat([df, pd.get_dummies(df[o])], axis = 1)
        del df[o]
    return(df)

def combine_levels(o, covariates, min_rt = 0.05, return_assignment = False):
    ''' given a Series categorical o and covariate df, produce an alternate series p that is reduced '''
    val_cts = o.value_counts()
    val_rts = val_cts / float(len(o))
    # print(val_rts)
    # print('\n')
    vals = list(val_cts.index)
    core_levels = [vals[i] for i,x in enumerate(val_rts) if x >= min_rt]
    reduce_levels = [vals[i] for i,x in enumerate(val_rts) if x < min_rt]
    # print('Assign these levels: ' + '\n')
    # print(reduce_levels)
    # print(' \n to these levels: \n')
    # print(core_levels)
    df = autoencode_dataframe(covariates)
    df['x'] = o
    # Get the average value across all covariates for each group
    distributions = df.groupby(['x']).mean().reset_index()
    labs = distributions.x
    distributions = distributions.drop('x', axis = 1)
    cs = cosine_similarity(distributions)
    # print(cs)
    similarities = pd.DataFrame(cs, columns = labs, index = labs) \
        .loc[reduce_levels, core_levels]
    assignment = similarities.apply(lambda row: row.idxmax(), axis = 1).to_dict()
    # print(assignment)
    z = o.replace(assignment)
    if return_assignment:
        return(z, assignment)
    else:
        return(z)
    
def clean_numeric(x):
    is_missing = pd.isnull(x)
    num_missing = is_missing.sum()
    if num_missing > 1:
        #z = x.fillna(x.mean())
        z = x.fillna(0)
        cleaned = pd.DataFrame({x.name+'_IMP' : z, x.name + '_NAFLAG' : is_missing})
    else:
        cleaned = pd.DataFrame({x.name+'_ORI' : x.fillna(0)})        
    return(cleaned)
        
def enhance_numeric(df):
    ''' given a numeric Series, produce an alternate version with a missing encoding and imputation'''
    numerics = df.columns[df.dtypes != object]
    for n in numerics:
        df = pd.concat([df, clean_numeric(df[n])], axis = 1)
        del df[n]
    return(df)


def load_data_v1(train = True):
    # Initialize data paths
    data_path = '/vol2/competitions/home-credit-default-risk/'
    data_files =  [data_path + x for x in os.listdir(data_path)]

    # Read secondary files
    number_of_applications = pd.read_csv(data_path + 'previous_application.csv.zip') \
    .groupby('SK_ID_CURR') \
    .agg({'SK_ID_PREV':'count'}) \
    .sort_values('SK_ID_PREV', ascending = False) \
    .reset_index() \
    .rename(columns = {'SK_ID_PREV':'num_previous_applications'})

    installments_payments = pd.read_csv(data_path + 'installments_payments.csv.zip')
    installments_payments.assign(nopay = lambda x:  x.AMT_PAYMENT < x.AMT_INSTALMENT).nopay.mean()
    
    underpayments = installments_payments \
    .assign(underpayment = lambda x: x.AMT_PAYMENT < x.AMT_INSTALMENT) \
    .groupby('SK_ID_CURR').agg({'underpayment':'sum'}).reset_index()

    installments_balance = installments_payments \
    .groupby('SK_ID_CURR')['AMT_INSTALMENT', 'AMT_PAYMENT'].sum().reset_index() \
    .assign(NETBALANCE = lambda x: x.AMT_PAYMENT - x.AMT_INSTALMENT )

    cc_balance = pd.read_csv(data_path + 'credit_card_balance.csv.zip')
    cc_util = cc_balance \
    .assign(utilization = lambda x: np.where(x.AMT_CREDIT_LIMIT_ACTUAL > 0, x.AMT_BALANCE / x.AMT_CREDIT_LIMIT_ACTUAL, 1),
            payment_balance = lambda x: np.where(x.AMT_BALANCE > 0, x.AMT_PAYMENT_TOTAL_CURRENT / x.AMT_BALANCE, 1)) \
    .groupby('SK_ID_CURR') \
    .agg({'utilization': 'mean', 'payment_balance': 'mean'}) \
    .reset_index()

    bureau = pd.read_csv(data_path + 'bureau.csv.zip')
    bureau_balance = pd.read_csv(data_path + 'bureau_balance.csv.zip')

    bureau_balance_statuses = bureau_balance \
        .groupby(['SK_ID_BUREAU', 'STATUS']) \
        .count() \
        .reset_index() \
        .pivot(index = 'SK_ID_BUREAU', columns = 'STATUS') \
        .fillna(0) \
        .reset_index('SK_ID_BUREAU')

    bureau_balance_statuses.columns = bureau_balance_statuses.columns.to_series().str.join('_')

    status_vars = list(bureau_balance_statuses.columns[1:])

    bureau_2 = bureau.merge(bureau_balance_statuses, how = 'left', left_on = 'SK_ID_BUREAU', right_on = 'SK_ID_BUREAU_')
    bureau_2.loc[:, status_vars] = bureau_2.loc[:, status_vars].fillna(0)
    bureau_3 = bureau_2.groupby('SK_ID_CURR')[status_vars].mean().reset_index()

    POS_CASH = pd.read_csv(data_path + 'POS_CASH_balance.csv.zip')
    DaysPastDue = POS_CASH.groupby('SK_ID_CURR').SK_DPD.mean().reset_index()
    
    # Stitch them all together
    if train:
        application_file = 'application_train.csv.zip'
    else:
        application_file = 'application_test.csv.zip'
    
    df = pd.read_csv(data_path + application_file) \
    .merge(number_of_applications, how = 'left', on = 'SK_ID_CURR') \
    .merge(underpayments, how = 'left', on = 'SK_ID_CURR') \
    .merge(cc_util, how = 'left', on = 'SK_ID_CURR') \
    .merge(bureau_3, how = 'left', on = 'SK_ID_CURR') \
    .merge(DaysPastDue, how = 'left', on = 'SK_ID_CURR') \
    .merge(installments_balance, how = 'left', on = 'SK_ID_CURR') \
    .assign(NAME_INCOME_TYPE = lambda x: np.where(x.NAME_INCOME_TYPE.isin(['Maternity leave', 'Businessman', 'Unemployed', 'Student', 'Unknown']), 'Other', x.NAME_INCOME_TYPE))
    return(df)

def get_design_matrix(data, features, train_test_split = False, train_sample = 0.75):
    objects = data.columns[(data.dtypes == object)]
    objects = objects[objects.isin(features)]
        
    data = data[features + ['TARGET']]
    
    # Convert categorical to dummy
    for o in objects:
        if len(data[o].unique()) >2:
            data[o] = combine_levels(data[o], data.loc[:,['TARGET', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], min_rt = 0.05, return_assignment = False)
        data = pd.concat([data, pd.get_dummies(data[o], prefix = o, prefix_sep = '_', dummy_na = True)], axis = 1)
        del data[o]                
    
    train_sample_idx = np.array([random.random() <= train_sample for i in range(data.shape[0])])
    
    if train_test_split:
        X_train = data.loc[train_sample_idx].drop('TARGET', axis = 1)
        y_train = data.loc[train_sample_idx, 'TARGET']
        X_test = data.loc[~train_sample_idx].drop('TARGET', axis = 1)
        y_test = data.loc[~train_sample_idx, 'TARGET']
        return(X_train, y_train, X_test, y_test)
    else:
        X = data.drop('TARGET', axis = 1)
        y = data['TARGET']
        return(X, y)
    
def category_to_numeric(x):
    values = list(x.astype(str).unique())
    le = LabelEncoder().fit(values + ['nan'])
    a = x.values.astype(str)
    result = le.transform(x.values.astype(str))
    return(result)


def get_design_matrix_lbl(data, features, train = True, train_test_split = False, train_sample = 0.75, convert_categorical = True):
    
    if train:
        data = data[features + ['TARGET']]
    else:
        data = data[features]
        
    if convert_categorical:
        # convert categorical column to labeled values
        objects = data.columns[(data.dtypes == object)]
        objects = objects[objects.isin(features)]
   
        for o in objects:
            data.loc[:,o] = category_to_numeric(data[o])
    
    if train == False:
        return(data)
    else:
        
        if train_test_split:
            train_sample_idx = np.array([random.random() <= train_sample for i in range(data.shape[0])])
            X_train = data.loc[train_sample_idx].drop('TARGET', axis = 1)
            y_train = data.loc[train_sample_idx, 'TARGET']
            X_test = data.loc[~train_sample_idx].drop('TARGET', axis = 1)
            y_test = data.loc[~train_sample_idx, 'TARGET']
            return(X_train, y_train, X_test, y_test)
        else:
            X = data.drop('TARGET', axis = 1)
            y = data['TARGET']
            return(X, y)
    
def get_design_matrix_refined(data, features, train_test_split = False, train_sample = 0.75):
    objects = data.columns[(data.dtypes == object)]
    objects = objects[objects.isin(features)]

    numerics = data.columns[(data.dtypes != object)]
    numerics = numerics[numerics.isin(features)]
    
    data = data[features + ['TARGET']]
    
    # Convert categorical to dummy
    for o in objects:
        data = pd.concat([data, pd.get_dummies(data[o], prefix = o, prefix_sep = '_')], axis = 1)
        del data[o]                

    # Discretize numerics if needed
    for n in numerics:
        if pd.isnull(data[n]).sum() > 0:
            data[n] = data[n]
            disc = pd.qcut(data[n], q = 100, duplicates = 'drop')
            disc_dummies = pd.get_dummies(disc, prefix = n, prefix_sep = '_', dummy_na= True)
            data = pd.concat([data, disc_dummies], axis = 1)
            del data[n]
        
    train_sample_idx = np.array([random.random() <= train_sample for i in range(data.shape[0])])
    
    if train_test_split:
        X_train = data.loc[train_sample_idx].drop('TARGET', axis = 1)
        y_train = data.loc[train_sample_idx, 'TARGET']
        X_test = data.loc[~train_sample_idx].drop('TARGET', axis = 1)
        y_test = data.loc[~train_sample_idx, 'TARGET']
        return(X_train, y_train, X_test, y_test)
    else:
        X = data.drop('TARGET', axis = 1)
        y = data['TARGET']
        return(X, y)    
    
def get_design_matrix_submission(data, features):
    objects = data.columns[(data.dtypes == object)]
    objects = objects[objects.isin(features)]
        
    data = data[features]
    
    # Convert categorical to dummy
    for o in objects:
        data = pd.concat([data, pd.get_dummies(data[o])], axis = 1)
        del data[o]                    
    return(data)
    
def plot_validation_curve(train_scores, validation_scores):
    #Plot validation curve
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(validation_scores, axis=1)
    test_scores_std = np.std(validation_scores, axis=1)

    plt.title("Validation Curve")
    plt.ylabel("Score")
    plt.ylim(0.5, 1.1)
    lw = 2
    plt.legend(loc="best")
    plt.show()
