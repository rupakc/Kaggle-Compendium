import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.model_selection import train_test_split
import preprocess
import hashfeatures


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


def get_total_features(search_frame,first_col_name='content',second_col_name='title'):
    feature_hash_first_text = hashfeatures.FeatureHash()
    feature_hash_second_text = hashfeatures.FeatureHash(max_feature_num=100)
    first_text_list = preprocess.text_clean_pipeline_list(list(search_frame[first_col_name].values))
    second_text_list = preprocess.text_clean_pipeline_list(list(search_frame[second_col_name].values))
    first_feature_set = feature_hash_first_text.get_feature_set(first_text_list)
    second_feature_set = feature_hash_second_text.get_feature_set(second_text_list)
    final_consolidated_feature_list = np.hstack((first_feature_set,second_feature_set))
    return final_consolidated_feature_list


stack_filename_list = ["biology.csv","cooking.csv","crypto.csv","diy.csv","robotics.csv","travel.csv"]
df_list = list([])
for filename in stack_filename_list:
    data_frame = pd.read_csv(filename)
    df_list.append(data_frame)
full_frame = reduce(lambda first,second:pd.concat([first,second],ignore_index=True),df_list)
tag_length_list = list(map(lambda tags:len(tags.strip().split()),full_frame['tags'].values))
del full_frame['id']
del full_frame['tags']
consolidated_features = get_total_features(full_frame)
X_train,X_test,y_train,y_test = train_test_split(consolidated_features,tag_length_list,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
