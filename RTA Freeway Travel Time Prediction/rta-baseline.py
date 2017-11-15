import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np


def get_gaussian_process_regressor():
    gp = GaussianProcessRegressor()
    return [gp],['Gaussian Process']


def get_mlp_regressor(num_hidden_units=51):
    mlp = MLPRegressor(hidden_layer_sizes=num_hidden_units)
    return [mlp],['Multi-Layer Perceptron']


def get_ensemble_models():
    rf = RandomForestRegressor(n_estimators=51,min_samples_leaf=5,min_samples_split=3,random_state=42)
    bag = BaggingRegressor(n_estimators=51,random_state=42)
    extra = ExtraTreesRegressor(n_estimators=71,random_state=42)
    ada = AdaBoostRegressor(random_state=42)
    grad = GradientBoostingRegressor(n_estimators=101,random_state=42)
    classifier_list = [rf,bag,extra,ada,grad]
    classifier_name_list = ['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost']
    return classifier_list, classifier_name_list


def get_linear_model():
    elastic_net = ElasticNet()
    return [elastic_net],['Elastic Net']


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name ,' ---------\n'
    predicted_values = trained_model.predict(X_test)
    print "Mean Absolute Error : ", metrics.mean_absolute_error(y_test,predicted_values)
    print "Median Absolute Error : ", metrics.median_absolute_error(y_test,predicted_values)
    print "Mean Squared Error : ", metrics.mean_squared_error(y_test,predicted_values)
    print "R2 Score : ", metrics.r2_score(y_test,predicted_values)
    print "---------------------------------------\n"


def remove_nan_rows(dataframe):
    for column in dataframe.columns:
        if dataframe[column].isnull().sum() > 10000:
            del dataframe[column]
    return dataframe


def spilt_date(list_of_date_string,date_separator='-',time_separator=':',format='yyyy-mm-dd hh:mm'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    hour_list = list([])
    minute_list = list([])
    for date_string in list_of_date_string:
        timestamp_list = date_string.strip().split(' ')
        date_list = timestamp_list[0].strip().split(date_separator)
        month_list.append(int(date_list[1]))
        day_list.append(int(date_list[2]))
        year_list.append(int(date_list[0]))
        time_list = timestamp_list[1].strip().split(time_separator)
        hour_list.append(int(time_list[0]))
        minute_list.append(int(time_list[1]))
    return month_list,day_list,year_list,hour_list,minute_list


def isfloat(num):
    try:
        num = float(num)
        return True
    except ValueError:
        return False


data_file = 'RTAHistorical.csv'
train_frame = pd.read_csv(data_file)
train_frame = remove_nan_rows(train_frame)
train_frame.dropna(inplace=True)

train_frame['month'],train_frame['day'],train_frame['year'],\
train_frame['hour'],train_frame['minute'] = spilt_date(list(train_frame['Unnamed: 0'].values))
del train_frame['Unnamed: 0']
columns = train_frame.columns
column_list = list([])

for c in columns:
    if isfloat(c):
        column_list.append(c)


for col in column_list:
    iteration_target_frame = train_frame[col].values
    column_list.remove(col)
    iteration_feature_frame = train_frame[column_list].values
    X_train,X_test,y_train,y_test = train_test_split(iteration_feature_frame,iteration_target_frame,test_size=0.2,random_state=42)
    regressor_list,regressor_name_list = get_ensemble_models()
    for regressor, regressor_name in zip(regressor_list,regressor_name_list):
        regressor.fit(X_train,y_train)
        print_evaluation_metrics(regressor,regressor_name,X_test,y_test)

