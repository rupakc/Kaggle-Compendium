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


def get_total_features(search_frame,first_col_name='product_title',second_col_name='product_description',query_col_name='query'):
    feature_hash_first_text = hashfeatures.FeatureHash(max_feature_num=100)
    feature_hash_second_text = hashfeatures.FeatureHash(max_feature_num=1000)
    feature_hash_query = hashfeatures.FeatureHash(max_feature_num=50)

    first_text_list = preprocess.text_clean_pipeline_list(list(search_frame[first_col_name].values))
    second_text_list = preprocess.text_clean_pipeline_list(list(search_frame[second_col_name].values))
    query_text_list = preprocess.text_clean_pipeline_list(list(search_frame[query_col_name].values))

    first_feature_set = feature_hash_first_text.get_feature_set(first_text_list)
    second_feature_set = feature_hash_second_text.get_feature_set(second_text_list)
    query_feature_set = feature_hash_query.get_feature_set(query_text_list)

    final_consolidated_feature_list = np.hstack((first_feature_set,second_feature_set,query_feature_set))
    return final_consolidated_feature_list


filename = 'train.csv'
search_frame = pd.read_csv(filename)
search_frame.dropna(inplace=True)

del search_frame['id']
del search_frame['relevance_variance']
prediction_values = list(search_frame['median_relevance'].values)
del search_frame['median_relevance']
features = get_total_features(search_frame)
X_train,X_test,y_train,y_test = train_test_split(features,prediction_values,test_size=0.1,random_state=42)
classifier_list, classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)

