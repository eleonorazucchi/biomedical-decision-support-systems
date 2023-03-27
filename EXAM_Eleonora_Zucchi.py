# EXAM: BIOMEDICAL DECISION SUPPORT SYSTEMS 
# NAME: Eleonora Zucchi

import pandas as pd
import random
import numpy as np
from MultiTimeSeries import MultiTimeSeries
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import warnings
import os.path

warnings.filterwarnings("ignore")

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ PARAMETERS ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

freq = '6H'                      # frequency of the timeseries
sliding_window_length = 16       # length of the sliding window 
number_of_samples = 10           # number of values that will be generated
sample_length = 20               # number of sampled samples
end_date = "2020-03-30"          # end date to match all datasets
kmeans_n_clusters = 10
n_splits = 5                     # KFold cross-validation splits

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

def load_data(path):
    _, extension = os.path.splitext(path)
    if extension == '.json':
        dataframe = pd.read_json(path)
        dataframe = dataframe.set_index('dateTime')
    elif extension == '.csv':
        dataframe = pd.read_csv(path)
        dataframe = dataframe.set_index('timestamp')
    return dataframe


calories = load_data('./pmdata/p01/fitbit/calories.json')               # load calories dataset

distances = load_data('./pmdata/p01/fitbit/distance.json')              # load distances dataset

heart_rate = load_data('./pmdata/p01/fitbit/heart_rate.json')           # load heart_rate dataset       
heart_rate["value"] = heart_rate["value"].apply(lambda x: x['bpm'])     

steps = load_data('./pmdata/p01/fitbit/steps.json')                     # load steps dataset         

sleep_score = load_data('./pmdata/p01/fitbit/sleep_score.csv')          # load sleep_score dataset  
sleep_score = sleep_score['overall_score']
sleep_score.index = pd.to_datetime(sleep_score.index)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

# Exercise 1.

# Function that normalizes the data setting a common granularity and ending date. 
# Values are summed according to the choosen timestamp, 
# missing values are copied from the previous row

def preprocessing_sum(dataframe, freq, end_date):
    dataframe.index = dataframe.index.round(freq=freq)
    dataframe = dataframe.groupby(by=dataframe.index, as_index=True).sum()
    dataframe = dataframe.resample(freq).ffill()
    dataframe = dataframe.loc[dataframe.index <= end_date]
    return dataframe    

# Function that normalizes the data setting a common granularity and ending date. 
# Values are averaged according to the choosen timestamp, 
# missing values are copied from the previous row

def preprocessing_mean(dataframe, freq, end_date):
    dataframe.index = dataframe.index.round(freq=freq)
    dataframe = dataframe.groupby(by=dataframe.index, as_index=True).mean()
    dataframe = dataframe.resample(freq).ffill()
    dataframe = dataframe.loc[dataframe.index <= end_date]
    return dataframe   

calories = preprocessing_sum(calories, freq, end_date)
distances = preprocessing_sum(distances, freq, end_date)
heart_rate = preprocessing_mean(heart_rate, freq, end_date)    # if freq = 'T' you need to delete last row with [:-1]
steps = preprocessing_sum(steps, freq, end_date)
sleep_score = preprocessing_mean(sleep_score, 'D', end_date)

series_list = [calories, distances, heart_rate, steps]

# ~ ~ ~ ~ ~ ~ ~ ~ ~ MULTI-LAYER PERCEPTRON REGRESSOR ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

nn_calories = MLPRegressor(hidden_layer_sizes = (100, 100), max_iter = 1000)
nn_distances = MLPRegressor(hidden_layer_sizes = (100, 100), max_iter = 1000)
nn_heart_rate = MLPRegressor(hidden_layer_sizes = (100, 100), solver = 'lbfgs', max_iter = 1000,  activation = 'relu')
nn_steps = MLPRegressor(hidden_layer_sizes = (100, 100), max_iter = 1000)

regressor_list = [nn_calories, nn_distances, nn_heart_rate, nn_steps]

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

# Exercise 2.

# Univariate MultiTimeSeries class initialization
ts = MultiTimeSeries(
    series_list = series_list,             # list of time series
    regressor_list = regressor_list,       # list of regressors
    univariate=True
)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

# # Exercise 3.

# Multivariate MultiTimeSeries class initialization
ts_1 = MultiTimeSeries(
    series_list = series_list,           # list time series
    regressor_list = regressor_list,     # list of regressors
    univariate=False
)

ts_transformed = ts.fit(length = sliding_window_length)

y = ts.generate_samples(number_of_samples = number_of_samples)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ RANDOM FOREST CLASSIFIER ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

rf_calories = RandomForestClassifier()
rf_distances = RandomForestClassifier()
rf_heart_rate = RandomForestClassifier()
rf_steps = RandomForestClassifier()

classifier_list = [rf_calories, rf_distances, rf_heart_rate, rf_steps]

matrix = ts.build_matrix(classifier_list)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ BSCAN CLUSTERER ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

c_calories = DBSCAN()
c_distances = DBSCAN()
c_heart_rate = DBSCAN()
c_steps = DBSCAN()

cluster_obj_list = [c_calories, c_distances, c_heart_rate, c_steps]

# 'metric' : 'precomputed' --> DBSCAN uses distance matrix from build_matrix function
c_params_calories = {'min_samples' : 3, 'metric' : 'precomputed'}
c_params_distances = {'min_samples' : 3, 'metric' : 'precomputed'}
c_params_heart_rate = {'min_samples' : 3, 'metric' : 'precomputed'}
c_params_steps = {'min_samples' : 3, 'metric' : 'precomputed'}

kargs_list = [c_params_calories, c_params_distances, c_params_heart_rate, c_params_steps]

ts.fit_cluster(cluster_obj_list = cluster_obj_list, kargs_list = kargs_list)

sample_index = random.randint(0, len(calories) - sample_length)

samples = calories['value'].reset_index(drop=True).iloc[sample_index : sample_index + sample_length].index.values
print(samples)

labels = ts.predict_cluster(sample = samples, index = 0)
print(labels)

calories_clusters, calories_labels = ts.parse_into_cluster(timeseries = calories, timeseries_index = 0)
print(calories_clusters)

distances_clusters, distances_labels = ts.parse_into_cluster(timeseries = distances, timeseries_index = 1)
print(distances_clusters)

heart_rate_clusters, heart_rate_labels = ts.parse_into_cluster(timeseries = heart_rate, timeseries_index = 2)
print(heart_rate_clusters)

steps_clusters, staps_labels = ts.parse_into_cluster(timeseries = steps, timeseries_index = 3)
print(steps_clusters)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

# Exercise 4:

# Function thet given a frqeuncy returns a dataset where each row represents a 24h window.
# The number of feature depends on the granularity (freq) choosen for the dataset.
def sample_generator_hour(dataframe, freq):
    daily_dataframe = list()
    offset = freq[:-1]
    time_interval = int(round(24 / int(offset), 0))
    for i in range(0, dataframe.shape[0] - time_interval, time_interval):
        daily_dataframe.append(dataframe[i:i+time_interval])
    daily_dataframe = pd.DataFrame(daily_dataframe).reset_index(drop=True)
    return daily_dataframe

def feature_concat(dataframe1, dataframe2):
    dataframe = pd.concat([dataframe1, dataframe2], axis = 1)
    return dataframe

def add_response(dataframe):
    y = dataframe['overall_score'].shift(-1)
    return y

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ K-MEANS ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

kmeans_calories = KMeans(n_clusters = kmeans_n_clusters, random_state = 5)
kmeans_distances = KMeans(n_clusters = kmeans_n_clusters, random_state = 5)
kmeans_heart_rate = KMeans(n_clusters = kmeans_n_clusters, random_state = 5)
kmeans_steps = KMeans(n_clusters = kmeans_n_clusters, random_state = 5)

kmeans_clusterer = [kmeans_calories, kmeans_distances, kmeans_heart_rate, kmeans_steps]
kmeans_dataframe = list()

for i in range(len(series_list)):
    kmeans_dataframe.append(kmeans_clusterer[i].fit_predict(series_list[i]))

# Function that cut the dataset at a choosen timestamp in hour fromat.
def cut_dataframe_hour(dataframe, freq, cut_hour):
    offset = freq[:-1]
    num_columns = int(round(cut_hour / int(offset), 0) + 1)
    dataframe = dataframe.iloc[:, :num_columns]
    return dataframe

cut_hour_list = list()

# The dataset is cut in the at the following timestamps 10:00, 12:00, 14:00 16:00, 18:00 generating 5 different datasets. 
# The generated dataset are the concatenated to create a 5 different multivariate style datasets.
for i in range(10, 20, 2):

    daily_dataframe = pd.DataFrame()

    for j in range(len(series_list)):
        tmp = sample_generator_hour(dataframe = kmeans_dataframe[j], freq = freq)
       
        daily_dataframe = pd.concat(
            [daily_dataframe, cut_dataframe_hour(dataframe=tmp, freq=freq, cut_hour=i)], \
            axis = 1
        ).reset_index(drop=True
)

        del tmp

    cut_hour_list.append(daily_dataframe)
    del daily_dataframe

# y fearure contains woth both categorical and ordinale labels are greated
y_cut_hour_ordinal, y_cut_hour_categorical = list(), list()

for i in range(len(cut_hour_list)):
    cut_hour_list[i] = feature_concat(dataframe1 = cut_hour_list[i], dataframe2 = sleep_score.reset_index(drop=True))

    y_cut_hour_ordinal.append(add_response(dataframe = cut_hour_list[i]))

    # label meaning
    labels_dict = {0: 'poor', 1: 'fair', 2: 'good', 3: 'excellent'}
    y_cut_hour_categorical.append(pd.cut(y_cut_hour_ordinal[i], bins = [0, 60, 80, 90, 100], labels = [0, 1, 2, 3]))

    cut_hour_list[i] = cut_hour_list[i][:-1]
    y_cut_hour_ordinal[i] = y_cut_hour_ordinal[i][:-1]
    y_cut_hour_categorical[i] = y_cut_hour_categorical[i][:-1]

y_cut_hour = list()

# categroical feature is added to each dataframe
for i in range(len(y_cut_hour_ordinal)):
    y_cut_hour.append(pd.concat([y_cut_hour_ordinal[i], y_cut_hour_categorical[i]], axis = 1).reset_index(drop = True))


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ KFOLD ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

# we have a list containing different dataframes one for each cut hour.
# With KFold we split each dataframe in the list into k dataframes, where every time the test set changes.
# Using k = 5, result: n * k dataframes to train

X_train, X_test = list(), list()
y_train, y_test = list(), list()

X_train_i, X_test_i = list(), list()
y_train_i, y_test_i = list(), list()

kfold = KFold(n_splits = n_splits, shuffle = False)
for i in range(len(cut_hour_list)):
    for train_index, test_index in kfold.split(cut_hour_list[i]):
        X_train_ij, X_test_ij = cut_hour_list[i].iloc[train_index], cut_hour_list[i].iloc[test_index]
        y_train_ij, y_test_ij = y_cut_hour[i].iloc[train_index], y_cut_hour[i].iloc[test_index]

        X_train_i.append(X_train_ij)
        X_test_i.append(X_test_ij)
        y_train_i.append(y_train_ij)
        y_test_i.append(y_test_ij)

    X_train.append(X_train_i)
    X_test.append(X_test_i)
    y_train.append(y_train_i)
    y_test.append(y_test_i)


models_regression = list()
classification_models_to_fit = [MLPClassifier(), RandomForestClassifier(), KNeighborsClassifier()]

# function that predicts the value for the sleep_score and returns the average mape score for each cut_hour
def regression_problem(X_train):
    col_to_predict = 0

    for i in range(len(X_train)):
        models_regression.append(MLPRegressor())
        mape_train, mape_test = list(), list()

        for j in range(n_splits):
            models_regression[i].fit(X = X_train[i][j], y = y_train[i][j].iloc[:, col_to_predict])

            y_pred_train = models_regression[i].predict(X_train[i][j])
            y_pred_test = models_regression[i].predict(X_test[i][j])

            mape_train.append(mean_absolute_percentage_error(y_true = y_train[i][j].iloc[:, col_to_predict], y_pred = y_pred_train))
            mape_test.append(mean_absolute_percentage_error(y_true = y_test[i][j].iloc[:, col_to_predict], y_pred = y_pred_test))

            # print(mape_train[j], mape_test[j]) # mape for each model run

        cross_val_mape_train = sum(mape_train) / n_splits
        cross_val_mape_test = sum(mape_test) / n_splits

        print('average mape train for cut {}: {}'.format(i + 1, cross_val_mape_train))
        print('average mape test for cut {}: {}'.format(i + 1, cross_val_mape_test))
        del mape_train, mape_test

    return cross_val_mape_train, cross_val_mape_test


# function that predicts the sleep_class and returns the average accuracy score
def classification_problem(X_train, classifier):
    col_to_predict = 1
    models_classification = list()

    for i in range(len(X_train)):
        models_classification.append(classifier)
        accuracy_train, accuracy_test = list(), list()

        for j in range(n_splits):
            models_classification[i].fit(X = X_train[i][j], y = y_train[i][j].iloc[:, col_to_predict])

            y_pred_train = models_classification[i].predict(X_train[i][j]) 
            y_pred_test = models_classification[i].predict(X_test[i][j])

            accuracy_train.append(accuracy_score(y_true = y_train[i][j].iloc[:, col_to_predict], y_pred = y_pred_train))
            accuracy_test.append(accuracy_score(y_true = y_test[i][j].iloc[:, col_to_predict], y_pred = y_pred_test))
                
            # print(accuracy_train[j], accuracy_test[j]) # accuracy for each model run

        cross_val_accuracy_train = sum(accuracy_train) / n_splits
        cross_val_accuracy_test = sum(accuracy_test) / n_splits

        print('average accuracy train for classifier: {} and for cut {}: {}'.format(classifier, i + 1, cross_val_accuracy_train))
        print('average accuracy test for classifier: {} and for cut {}: {}'.format(classifier, i + 1, cross_val_accuracy_test))

        del accuracy_train, accuracy_test   
    return models_classification

cross_val_mape_train, cross_val_mape_test = regression_problem(X_train)

# probability=True --> 
classifiers = [MLPClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
classifiers_fitted = []
for classifier in classifiers:
    classifiers_fitted.append(classification_problem(X_train, classifier))

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

# Exercise 5:

models = [classifier_fitted[-1] for classifier_fitted in classifiers_fitted]
# del classifiers_fitted
dataframe = cut_hour_list[0]
y = y_cut_hour[0].iloc[:, -1:]

# Split the data into training and calibration sets using bagging
index_calibration = np.random.choice(len(dataframe), size=int(0.2 * len(dataframe)), replace=True)
X_calibration, y_calibration = dataframe.iloc[index_calibration], y.iloc[index_calibration]

# kfold initialization here
kf = KFold(n_splits=5, shuffle=True)

for model in models:
    y_pred_calibration = model.predict(X_calibration)
    errors = 1 - (y_pred_calibration == y_calibration['overall_score'])
    # generate prediction intervals
    lower, upper = np.percentile(errors, [5, 95])
    coverage = []

    # kfold splitting here, we test all the dataframe
    for train_index, test_index in kf.split(dataframe):
        X_train, X_test = dataframe.iloc[train_index, :], dataframe.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # predict the probabilities for test set
        y_pred = model.predict_proba(X_test)
        lower_bound = y_pred - lower
        upper_bound = y_pred + upper

        # coverage rate for each fold
        in_interval = np.logical_and(lower_bound >= y_test.values[:, np.newaxis], y_test.values[:, np.newaxis] <= upper_bound)
        coverage.append(np.mean(in_interval))
        print(np.mean(in_interval))

    average_coverage = np.mean(coverage)
    print('average coverage: {}'.format(average_coverage))
