import pandas as pd
import datetime
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


def spilt_date(list_of_date_string,date_separator='-',time_separator=':',format='yyyy-mm-dd hh:mm:ss'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    hour_list = list([])
    minute_list = list([])
    second_list = list([])
    for date_string in list_of_date_string:
        timestamp_list = date_string.strip().split(' ')
        date_list = timestamp_list[0].strip().split(date_separator)
        month_list.append(int(date_list[1]))
        day_list.append(int(date_list[2]))
        year_list.append(int(date_list[0]))
        time_list = timestamp_list[1].strip().split(time_separator)
        hour_list.append(int(time_list[0]))
        minute_list.append(int(time_list[1]))
        second_list.append(int(time_list[2]))
    return month_list,day_list,year_list,hour_list,minute_list,second_list


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def get_start_and_stop_coordinates(poly_line_values):
    start_x_list = list([])
    start_y_list = list([])
    end_x_list = list([])
    end_y_list = list([])

    for list_of_coordinates in poly_line_values:
        list_of_coordinates = list_of_coordinates.replace('[','')
        list_of_coordinates = list_of_coordinates.replace(']','')
        list_of_coordinates = list_of_coordinates.split(',')

        start_coordinates_x = list_of_coordinates[0]
        start_coordinates_y = list_of_coordinates[1]
        end_coordinates_x = list_of_coordinates[len(list_of_coordinates)-2]
        end_coordinates_y = list_of_coordinates[len(list_of_coordinates) - 1]
        start_x_list.append(float(start_coordinates_x))
        start_y_list.append(float(start_coordinates_y))
        end_x_list.append(float(end_coordinates_x))
        end_y_list.append(float(end_coordinates_y))
    return start_x_list,start_y_list,end_x_list,end_y_list


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name ,' ---------\n'
    predicted_values = trained_model.predict(X_test)
    print "Mean Absolute Error : ", metrics.mean_absolute_error(y_test,predicted_values)
    print "Median Absolute Error : ", metrics.median_absolute_error(y_test,predicted_values)
    print "Mean Squared Error : ", metrics.mean_squared_error(y_test,predicted_values)
    print "R2 Score : ", metrics.r2_score(y_test,predicted_values)
    print "---------------------------------------\n"


filename = 'train.csv'
train_frame = pd.read_csv(filename,nrows=500000)
columns_to_delete = ['ORIGIN_CALL','ORIGIN_STAND','TRIP_ID']
train_frame.drop(columns_to_delete,axis=1,inplace=True)
train_frame = train_frame[train_frame['MISSING_DATA'] == False]
del train_frame['MISSING_DATA']
train_frame['Time'] = map(lambda x:(x.count('[')-1)*15.0,list(train_frame['POLYLINE'].values))
train_frame = train_frame[train_frame['Time'] > 1.0]
train_frame['TIMESTAMP'] = map(lambda x:datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'),train_frame['TIMESTAMP'].values)
train_frame['Start_X'],train_frame['Start_Y'],train_frame['End_X'],train_frame['End_Y'] = get_start_and_stop_coordinates(list(train_frame['POLYLINE'].values))
train_frame['Month'],train_frame['Day'],train_frame['Year'],train_frame['Hour'],train_frame['Minute'],train_frame['Second'] = spilt_date(list(train_frame['TIMESTAMP']))
del train_frame['TIMESTAMP']
del train_frame['POLYLINE']
train_frame = label_encode_frame(train_frame)
target_values = train_frame['Time']
del train_frame['Time']
X_train,X_test,y_train,y_test = train_test_split(train_frame.values,target_values,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
