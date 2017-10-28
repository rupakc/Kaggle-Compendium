import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import preprocess
import hashfeatures
import numpy as np


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


def get_ensemble_models():
    rf = RandomForestClassifier(n_estimators=91,min_samples_leaf=5,min_samples_split=3)
    bagg = BaggingClassifier(n_estimators=71,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=57,random_state=42)
    ada = AdaBoostClassifier(n_estimators=51,random_state=42)
    grad = GradientBoostingClassifier(n_estimators=101,random_state=42)
    classifier_list = [rf,bagg,extra,ada,grad]
    classifier_name_list = ['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost']
    return classifier_list,classifier_name_list


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][0]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def split_date(list_of_date_string, separator=' ', format='mm/dd/yyyy hh:mm:ss'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    hour_list = list([])
    minute_list = list([])
    second_list = list([])
    for date_string in list_of_date_string:
        date_time_list = date_string.strip().split(separator)
        date_part = date_time_list[0].strip().split('/')
        month_list.append(int(date_part[0]))
        day_list.append(int(date_part[1]))
        year_list.append(int(date_part[2]))
        if len(date_time_list) == 2:
            time_part = date_time_list[1].strip().split(':')
            hour_list.append(int(time_part[0]))
            minute_list.append(int(time_part[1]))
            second_list.append(int(time_part[2]))
        else:
            hour_list.append(int(0))
            minute_list.append(int(0))
            second_list.append(int(0))
    return month_list,day_list,year_list,hour_list,minute_list,second_list


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name, ' ---------------\n'
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


def get_full_feature_set(dataframe):
    title_list = list(dataframe['Title'].values)
    body_list = list(dataframe['BodyMarkdown'].values)
    clean_title_list = preprocess.text_clean_pipeline_list(title_list)
    clean_body_list = preprocess.text_clean_pipeline_list(body_list)
    title_feature = hashfeatures.FeatureHash(max_feature_num=100)
    body_feature = hashfeatures.FeatureHash(max_feature_num=400)
    title_hash_features = title_feature.get_feature_set(clean_title_list)
    body_hash_features = body_feature.get_feature_set(clean_body_list)
    del dataframe['Title']
    del dataframe['BodyMarkdown']
    full_feature_set = np.hstack((title_hash_features,body_hash_features,dataframe.values))
    return full_feature_set


filename = 'train-sample.csv'
predict_frame = pd.read_csv(filename)
columns_to_drop = ['PostId','OwnerUserId','PostClosedDate','Tag2','Tag3','Tag4','Tag5']
predict_frame.drop(labels=columns_to_drop,axis=1,inplace=True)
open_label_list = list(predict_frame['OpenStatus'].values)
del predict_frame['OpenStatus']
month_list,day_list,year_list,hour_list,minute_list,second_list = split_date(list(predict_frame['PostCreationDate'].values))
predict_frame['post_month'] = month_list
predict_frame['post_day'] = day_list
predict_frame['post_year'] = year_list
predict_frame['post_hour'] = hour_list
predict_frame['post_minute'] = minute_list
predict_frame['post_second'] = second_list
del predict_frame['PostCreationDate']
del predict_frame['OwnerCreationDate']
predict_frame['Tag1'] = LabelEncoder().fit_transform(list(predict_frame['Tag1'].values))
full_features = get_full_feature_set(predict_frame)
print np.shape(full_features)
X_train,X_test,y_train,y_test = train_test_split(full_features,open_label_list,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier, classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
