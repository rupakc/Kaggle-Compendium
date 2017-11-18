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
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import numpy as np
import operator
import hashfeatures


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
    rf = RandomForestClassifier(n_estimators=51,min_samples_leaf=5,min_samples_split=3)
    bagg = BaggingClassifier(n_estimators=71,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=57,random_state=42)
    ada = AdaBoostClassifier(n_estimators=51,random_state=42)
    grad = GradientBoostingClassifier(n_estimators=101,random_state=42)
    classifier_list = [rf,bagg,extra,ada,grad]
    classifier_name_list = ['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost']
    return classifier_list,classifier_name_list


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name, ' ----------------'
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


def get_word_count(list_of_words):
    count_dict = dict({})
    for word in list_of_words:
        if word in count_dict.keys():
            count_dict[word] += 1
        else:
            count_dict[word] = 1
    return count_dict


def get_word_list_from_string(string_list):
    total_word_list = list([])
    for number_string in string_list:
        if number_string is not np.nan:
            count_list = number_string.split()
            total_word_list.extend(count_list)
    return total_word_list


def get_top_n_dict_keys(dictionary,top_n=1):
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    sorted_x = sorted_x[:top_n]
    key_list = list([])
    for sort_value in sorted_x:
        key_list.append(sort_value[0])
    return key_list


def fill_nan_in_string(dataframe):
    for column in dataframe.columns:
        string_list = dataframe[column].values
        total_word_list = get_word_list_from_string(string_list)
        word_dict = get_word_count(total_word_list)
        top_keys = get_top_n_dict_keys(word_dict,top_n=7)
        top_key_string = " ".join(top_keys)
        dataframe[column] = map(lambda x:top_key_string if x is np.nan else x,dataframe[column].values)
    return dataframe


filename = 'training.csv'
train_frame = pd.read_csv(filename)
name_desc_cap_frame = train_frame[['name','description','caption']]
target_class_labels = train_frame['good'].values
train_frame.drop(['name','description','caption','good'],axis=1,inplace=True)
name_desc_cap_frame = fill_nan_in_string(name_desc_cap_frame)
name_features = hashfeatures.FeatureHash(max_feature_num=100).get_feature_set(name_desc_cap_frame['name'].values)
desc_features = hashfeatures.FeatureHash(max_feature_num=500).get_feature_set(name_desc_cap_frame['description'].values)
caption_features = hashfeatures.FeatureHash(max_feature_num=200).get_feature_set(name_desc_cap_frame['caption'].values)
train_features = Imputer().fit_transform(train_frame.values)
final_features = np.hstack((name_features,desc_features,caption_features,train_features))
X_train,X_test,y_train,y_test = train_test_split(final_features,target_class_labels,test_size=0.1,random_state=42)
classifer_list, classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifer_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
