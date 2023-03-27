# EXAM: BIOMEDICAL DECISION SUPPORT SYSTEMS 
# NAME: Eleonora Zucchi


import numpy as np
import pandas as pd
from scipy import stats
from itertools import groupby
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

class MultiTimeSeries:
    def __init__(
        self, 
        series_list: list, 
        regressor_list: list,  
        univariate: bool = True
    ): 
        
        self.regressor_list = regressor_list
        self.series_list = series_list
        self.univariate = univariate

        if (len(regressor_list) != len(series_list)):
            raise AssertionError("Length of the two lists must coincide.")

        return

    # function that returns the dataset shaped according to the choosen sliding window length
    def _time_series_generator(self, length):
        # make a copy to not alter the original data
        _series_list = self.series_list.copy()

        for i in range(len(_series_list)):
            cols, names = list(), list()
            for sw in range(length):
                cols.append(_series_list[i].shift(-sw))

            cols.append(_series_list[i].shift(-(sw+1)))
            names += ['x' + str(j) for j in range(length)]      # x columns names 
            names += ['y']                                      # y column name

            _series_list[i] = pd.concat(cols, axis=1)
            _series_list[i].columns = names
            _series_list[i] = _series_list[i][:-(sw+1)]         # deletes trailing nan values at end of dataframe

        return _series_list


    def fit(self, length: int = 2):
        if (length < 1):
            raise ValueError('Specify an integer number higher than zero (0).')

        self.length = length
        _series_list = self._time_series_generator(length = self.length)
       
        self.X, self.Y = list(), list()
        X_train, X_test = list(), list()
        Y_train, Y_test = list(), list()

        for i in range(len(_series_list)):
            self.Y.append(_series_list[i]['y'])   
            self.X.append(_series_list[i].drop(columns=['y']))

        for i in range(len(_series_list)):
            X_train_i, X_test_i, Y_train_i, Y_test_i = train_test_split(
                self.X[i], self.Y[i], test_size = 0.15, shuffle = False
            )

            X_train.append(X_train_i)
            X_test.append(X_test_i)
            Y_train.append(Y_train_i)
            Y_test.append(Y_test_i)

        if (self.univariate == True):
            for i in range(len(_series_list)):
                self.regressor_list[i].fit(X_train[i], Y_train[i])

                Y_pred_train = self.regressor_list[i].predict(X_train[i])
                Y_pred_test = self.regressor_list[i].predict(X_test[i])
                
                print("mape train: {}".format(mean_absolute_percentage_error(y_true = Y_train[i], y_pred = Y_pred_train)*100))
                print("mape test: {}".format(mean_absolute_percentage_error(y_true = Y_test[i], y_pred = Y_pred_test)*100))
        
        else:  
  
            names_X_concat, names_Y_concat = list(), list() 
            
            for i in range(len(_series_list)):          

                names_X_concat += ['x' + str(i) + '.' + str(j) for j in range(len(self.X_train[i].columns))]  
                names_Y_concat += ['y' + str(i)]

            X_train_concat = pd.concat(X_train, axis=1)   
            X_train_concat.columns = names_X_concat

            Y_train_concat = pd.concat(Y_train, axis=1)
            Y_train_concat.columns = names_Y_concat

            X_test_concat = pd.concat(X_test, axis=1)   
            X_test_concat.columns = names_X_concat

            Y_test_concat = pd.concat(Y_test, axis=1)
            Y_test_concat.columns = names_Y_concat

            for i, y_col in enumerate(Y_train_concat.columns):              
                self.regressor_list[i].fit(X_train_concat, Y_train_concat[y_col])

                Y_pred_train = self.regressor_list[i].predict(X_train_concat)
                Y_pred_test = self.regressor_list[i].predict(X_test_concat)

                print("mape train: {}".format(mean_absolute_percentage_error(y_true = Y_train_concat[y_col], y_pred = Y_pred_train)))
                print("mape test: {}".format(mean_absolute_percentage_error(y_true = Y_test_concat[y_col], y_pred = Y_pred_test)))
        
        return 

    # Function that generate a choosen number of samples with the choosen length 
    def generate_samples(self, number_of_samples: int):       
        self.y_pred = list()
        self.number_of_samples = number_of_samples

        if (self.univariate == True):

            for i in range(len(self.series_list)):
                # sample select a random row from the dataset
                current = self.X[i].sample(n=1)     
                y = list()

                for j in range(number_of_samples):
                    y.append(self.regressor_list[i].predict(current))
                    current['x_'+str(j + self.length)] = y[j]
                    current = current[current.columns[1:]]
                
                self.y_pred.append(pd.DataFrame(np.array(y)))
                del y
                    
        else:
            current = list()

            for i in range(len(self.series_list)):
                self.X[i].columns = self.X[i].columns + '.' + str(i)
                current.append(self.X[i].sample(n=1))
                current[i].reset_index(drop=True, inplace=True)
                   
            for j in range(number_of_samples):
                y = list()

                for i in range(len(self.series_list)):
                    current_concat = pd.concat(current, axis = 1)
                    y.append(self.regressor_list[i].predict(current_concat))
                    current[i]['x_' + str(j + self.length) + '.' + str(i)] = y[i]
                    current[i] = current[i][current[i].columns[1:]]
                    del current_concat

                self.y_pred.append(np.array(y))
                del y
            self.y_pred = [pd.DataFrame(col) for col in np.concatenate(self.y_pred, axis = 1)]

        return self.y_pred # 'fake dataset'

    # function that returns the distance between two samples 
    def distance(self, s1, s2, _classifier) : 
        s1_leaves = _classifier.apply(s1)
        s2_leaves = _classifier.apply(s2)
        
        return np.array([
            (s1_row.reshape(-1, 1) == s2_row.reshape(-1, 1)).any(axis = 1).sum() / _classifier.n_estimators \
                for s1_row in s1_leaves \
                for s2_row in s2_leaves
        ]).reshape(s1_leaves.shape[0], s2_leaves.shape[0])
    
    # function that returns the distance matrix 
    # generated dataset is given lable 1, while original dataset ig given label 0
    def build_matrix(self, classifier_list):
        _series_list = self.series_list.copy()
        _y_pred = self.y_pred.copy()
        self.classifier_list = classifier_list

        list_concat = list()
        self.distance_matrix = list()

        X_concat = list()

        for i in range(len(_series_list)):        
            _series_list[i]['label'] = 0                             # 'real dataset' --> label 0

            _y_pred[i] = _y_pred[i].rename(columns = {0:'value'})    # 'fake dataset' --> label 1
            _y_pred[i]['label'] = 1

            _series_list[i] = _series_list[i].reset_index(drop=True)
            _y_pred[i] = _y_pred[i].reset_index(drop=True)

            list_concat.append(pd.concat([_series_list[i], _y_pred[i]], axis=0))

            X = list_concat[i]['value'].values.reshape(-1, 1) # X.shape -> (n, 1)
            y = list_concat[i]['label'] # pd.Series.shape -> (n, )

            self.classifier_list[i].fit(X, y)

            X_concat.append(X)
            del X, y

        for i in range(len(classifier_list)):
            self.distance_matrix.append(self.distance(
                s1 = X_concat[i],  \
                s2 = X_concat[i],  \
                _classifier = self.classifier_list[i])
            )

        return self.distance_matrix
    
    # function that fits a distance cluster object list
    def fit_cluster(self, cluster_obj_list, kargs_list): 
        self.cluster_obj_list = cluster_obj_list

        for i in range(len(self.cluster_obj_list)):
            self.cluster_obj_list[i].set_params(**kargs_list[i])

        for i in range(len(cluster_obj_list)):
            self.cluster_obj_list[i].fit(self.distance_matrix[i])
            
        return self.cluster_obj_list
    
    
    # function that returns the label of a known sample present in the distance matrix
    def predict_cluster(self, sample, index):       
        return self.cluster_obj_list[index].labels_[sample]
    
    # function that returns the labels of the corresponding timeseries_index passed
    def parse_into_cluster(self, timeseries, timeseries_index): 
        labels_list = list()

        for j in range(len(timeseries) - self.length) :
            labels_list.append(
                stats.mode(
                    np.array([
                        self.cluster_obj_list[timeseries_index].labels_[i] \
                        for i in range(j, j + self.length)
                    ])
                ).mode[0]
            )

        labels_list_tuple = [(label, len(list(count))) for label, count in groupby(labels_list)]
        return labels_list_tuple, labels_list

