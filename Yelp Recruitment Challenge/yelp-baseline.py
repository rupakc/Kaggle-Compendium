import pandas as pd
import json
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
import hashfeatures
import preprocess
import numpy as np


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


def get_dataframe_from_file(filename):
    master_list = list([])
    with open(filename, 'r') as json_file:
        temp_list = json_file.read().splitlines()
        master_list = map(lambda temp_element:json.loads(temp_element),temp_list)
    return pd.DataFrame(master_list)


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is np.nan:
            for i in range(len(dataframe)):
                if i > 1000:
                    break
                if type(dataframe[column][i]) is unicode or type(dataframe[column][i]) is np.bool_:
                    dataframe[column] = encoder.fit_transform(dataframe[column].values)
                    break
        elif type(dataframe[column][0]) is unicode or type(dataframe[column][0]) is np.bool_:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def get_hash_features(dataframe,column_dim_dict):
    text_features_master_list = list([])
    for key in column_dim_dict.keys():
        text_list = list(dataframe[key].values)
        text_features = hashfeatures.FeatureHash(max_feature_num=column_dim_dict[key]).get_feature_set(text_list)
        text_features_master_list.append(text_features)
    return reduce(lambda x,y:np.hstack((x,y)),text_features_master_list)


def spilt_date(list_of_date_string,separator='-',format='yyyy-mm-dd'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    for date_string in list_of_date_string:
        date_list = date_string.strip().split(separator)
        month_list.append(int(date_list[1]))
        day_list.append(int(date_list[2]))
        year_list.append(int(date_list[0]))
    return month_list,day_list,year_list


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name ,' ---------\n'
    predicted_values = trained_model.predict(X_test)
    print "Mean Absolute Error : ", metrics.mean_absolute_error(y_test,predicted_values)
    print "Median Absolute Error : ", metrics.median_absolute_error(y_test,predicted_values)
    print "Mean Squared Error : ", metrics.mean_squared_error(y_test,predicted_values)
    print "R2 Score : ", metrics.r2_score(y_test,predicted_values)
    print "---------------------------------------\n"


filename_user = 'yelp_training_set_user.json'
filename_review = 'yelp_training_set_review.json'
filename_business = 'yelp_training_set_business.json'


user_frame = get_dataframe_from_file(filename_user)
review_frame = get_dataframe_from_file(filename_review)
business_frame = get_dataframe_from_file(filename_business)

user_review_frame = pd.merge(user_frame,review_frame,how='outer',left_on='user_id',right_on='user_id',suffixes=('_user','_review'))
master_frame = pd.merge(user_review_frame,business_frame,how='outer',left_on='business_id',right_on='business_id',suffixes=('_review','_business'))
master_frame.dropna(inplace=True)
master_frame['votes_user'] = map(lambda x:sum(x.values()),master_frame['votes_user'].values)
master_frame['votes_review'] = map(lambda x:sum(x.values()),master_frame['votes_review'].values)

target_value_label = 'stars_business'

del user_frame
del review_frame
del business_frame

columns_to_delete = ['type','neighborhoods','type_user','type_review','user_id','review_id','business_id','name_review']
columns_for_hash_dict = dict({'text':1000,'name_business':50,'full_address':150})
master_frame.drop(columns_to_delete,axis=1,inplace=True)
master_frame['categories'] = map(lambda x:["None"] if len(x) == 0 else x,master_frame['categories'].values)
text_features_hash = get_hash_features(master_frame,columns_for_hash_dict)
master_frame.drop(columns_for_hash_dict.keys(),axis=1,inplace=True)
master_frame['month'],master_frame['day'],master_frame['year'] = spilt_date(list(master_frame['date'].values))
del master_frame['date']
del master_frame['categories']
target_values = list(master_frame[target_value_label].values)
del master_frame[target_value_label]
master_frame = label_encode_frame(master_frame)

for column in master_frame.columns:
    print column," : ", master_frame[column][0], " : ", type(master_frame[column][0])

final_feature_set = np.hstack((text_features_hash,master_frame.values))
X_train,X_test,y_train,y_test = train_test_split(final_feature_set,target_values,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
