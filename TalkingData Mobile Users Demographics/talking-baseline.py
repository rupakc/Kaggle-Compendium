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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


def get_gaussian_process_regressor():
    gp = GaussianProcessRegressor()
    return [gp],['Gaussian Process']


def get_mlp_regressor(num_hidden_units=51):
    mlp = MLPRegressor(hidden_layer_sizes=num_hidden_units)
    return [mlp],['Multi-Layer Perceptron']


def get_ensemble_models_regressor():
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


def print_evaluation_metrics_regression(trained_model,trained_model_name,X_test,y_test):
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


def get_naive_bayes_models():
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    classifier_list = [gnb,mnb,bnb]
    classifier_name_list = ['Gaussian NB','Multinomial NB','Bernoulli NB']
    return classifier_list,classifier_name_list


def get_neural_network(hidden_layer_size=50):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size)
    return [mlp], ['MultiLayer Perceptron']


def get_ensemble_models_classification():
    rf = RandomForestClassifier(n_estimators=51,min_samples_leaf=5,min_samples_split=3)
    bagg = BaggingClassifier(n_estimators=71,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=57,random_state=42)
    ada = AdaBoostClassifier(n_estimators=51,random_state=42)
    grad = GradientBoostingClassifier(n_estimators=101,random_state=42)
    classifier_list = [rf,bagg,extra,ada,grad]
    classifier_name_list = ['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost']
    return classifier_list,classifier_name_list


def print_evaluation_metrics_classification(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name,' -----------------'
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


gender_frame = pd.read_csv('gender_age_train.csv', nrows=10000)
phone_brand_frame = pd.read_csv('phone_brand_device_model.csv', nrows=10000)
event_frame = pd.read_csv('events.csv', nrows=10000)
app_event_frame = pd.read_csv('app_events.csv', nrows=10000)
app_label_frame = pd.read_csv('app_labels.csv', nrows=10000)
label_category_frame = pd.read_csv('label_categories.csv', nrows=10000)

list_of_df = [gender_frame,phone_brand_frame,event_frame,app_event_frame,app_label_frame,label_category_frame]

app_label_concat_frame = pd.merge(app_label_frame,label_category_frame,how='right',left_on='label_id',right_on='label_id')
gender_device_concat_frame = pd.merge(gender_frame,phone_brand_frame,how='right',left_on='device_id',right_on='device_id')
app_event_label_frame = pd.merge(app_label_concat_frame,app_event_frame,how='right',left_on='app_id',right_on='app_id')
event_app_event_merge_frame = pd.merge(app_event_label_frame,event_frame,how='right',left_on='event_id',right_on='event_id')
final_merged_frame = pd.merge(gender_device_concat_frame,event_app_event_merge_frame,how='right',left_on='device_id',right_on='device_id')

for df in list_of_df:
    del df

columns_to_drop = ['device_id','group','app_id','label_id','category','is_installed','is_active']
final_merged_frame.drop(columns_to_drop,axis=1,inplace=True)
final_merged_frame['month'],final_merged_frame['day'],final_merged_frame['year'],final_merged_frame['hour'], final_merged_frame['minute'], final_merged_frame['second'] = spilt_date(list(final_merged_frame['timestamp'].values))
del final_merged_frame['timestamp']
final_merged_frame = label_encode_frame(final_merged_frame)
gender_labels = list(final_merged_frame['gender'].values)
del final_merged_frame['gender']
imputed_values = Imputer().fit_transform(final_merged_frame.values)
X_train,X_test,y_train,y_test = train_test_split(imputed_values,gender_labels,test_size=0.1,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models_classification()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics_classification(classifier,classifier_name,X_test,y_test)


