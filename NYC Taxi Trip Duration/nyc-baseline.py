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
        month_list.append(date_list[1])
        day_list.append(date_list[2])
        year_list.append(date_list[0])
        time_list = timestamp_list[1].strip().split(time_separator)
        hour_list.append(time_list[0])
        minute_list.append(time_list[1])
        second_list.append(time_list[2])
    return month_list,day_list,year_list,hour_list,minute_list,second_list


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name ,' ---------\n'
    predicted_values = trained_model.predict(X_test)
    print "Mean Absolute Error : ", metrics.mean_absolute_error(y_test,predicted_values)
    print "Median Absolute Error : ", metrics.median_absolute_error(y_test,predicted_values)
    print "Mean Squared Error : ", metrics.mean_squared_error(y_test,predicted_values)
    print "R2 Score : ", metrics.r2_score(y_test,predicted_values)
    print "---------------------------------------\n"


filename = 'train.csv'
nyc_frame = pd.read_csv(filename)
del nyc_frame['store_and_fwd_flag']
month_list,day_list,year_list,hour_list,minute_list,second_list = spilt_date(list(nyc_frame['pickup_datetime'].values))
nyc_frame['Month_Pickup'] = month_list
nyc_frame['Day_Pickup'] = day_list
nyc_frame['Year_Pickup'] = year_list
nyc_frame['Hour_Pickup'] = hour_list
nyc_frame['Minute_Pickup'] = minute_list
nyc_frame['Second_Pickup'] = second_list
del nyc_frame['pickup_datetime']
month_list,day_list,year_list,hour_list,minute_list,second_list = spilt_date(list(nyc_frame['dropoff_datetime'].values))
nyc_frame['Month_Dropoff'] = month_list
nyc_frame['Day_Dropoff'] = day_list
nyc_frame['Year_Dropoff'] = year_list
nyc_frame['Hour_Dropoff'] = hour_list
nyc_frame['Minute_Dropoff'] = minute_list
nyc_frame['Second_Dropoff'] = second_list
del nyc_frame['dropoff_datetime']
predicted_values = list(nyc_frame['trip_duration'].values)
del nyc_frame['id']
del nyc_frame['trip_duration']

X_train,X_test,y_train,y_test = train_test_split(nyc_frame.values,predicted_values,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_mlp_regressor()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
