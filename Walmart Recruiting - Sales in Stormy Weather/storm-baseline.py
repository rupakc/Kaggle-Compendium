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


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is np.nan:
            for i in range(len(dataframe)):
                if i > 1000:
                    break
                if type(dataframe[column][i]) is str:
                    dataframe[column] = encoder.fit_transform(dataframe[column].values)
                    break
        elif type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def spilt_date(list_of_date_string,separator='-',format='yyyy-mm-dd'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    for date_string in list_of_date_string:
        date_list = date_string.strip().split(separator)
        month_list.append(date_list[1])
        day_list.append(date_list[2])
        year_list.append(date_list[0])
    return month_list,day_list,year_list


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def handle_mixed_data_types(dataframe):
    for column_name in dataframe.columns:
        column_data = list(dataframe[column_name].values)
        float_count = 0
        float_sum = 0.0
        string_count = 0
        for data in column_data:
            if isfloat(data):
                float_count += 1.0
                float_sum += float(data)
            else:
                string_count += 1
        if float_count >= string_count:
            mean = float_sum/float_count
            for index,value in enumerate(column_data):
                if not isfloat(value):
                    column_data[index] = mean
                else:
                    column_data[index] = float(value)
        dataframe[column_name] = column_data
    return dataframe


weather_filename = 'weather.csv'
train_filename = 'train.csv'
key_filename = 'key.csv'
weather_frame = pd.read_csv(weather_filename)
train_frame = pd.read_csv(train_filename)
key_frame = pd.read_csv(key_filename)
weather_frame.drop(['codesum','depart'],axis=1,inplace=True)
weather_frame = handle_mixed_data_types(weather_frame)
final_frame = pd.merge(train_frame,key_frame,how='inner',left_on='store_nbr',right_on='store_nbr')
final_frame = pd.merge(final_frame,weather_frame,how='inner',left_on=['station_nbr','date'],right_on=['station_nbr','date'])
target_values = list(final_frame['units'].values)
final_frame['month'], final_frame['day'], final_frame['year'] = spilt_date(list(final_frame['date'].values))
del final_frame['units']
del final_frame['date']
X_train,X_test,y_train,y_test = train_test_split(final_frame.values,target_values,test_size=0.2,random_state=42)
regressor_list,regressor_name_list = get_ensemble_models()
for regressor,regressor_name in zip(regressor_list,regressor_name_list):
    regressor.fit(X_train,y_train)
    print_evaluation_metrics(regressor,regressor_name,X_test,y_test)
