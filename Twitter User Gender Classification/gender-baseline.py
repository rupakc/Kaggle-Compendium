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
from sklearn.svm import LinearSVC
import numpy as np
import hashfeatures
import preprocess


def get_svm():
    svm = LinearSVC()
    return [svm], ['SVM']


def get_naive_bayes_models():
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    classifier_list = [gnb,mnb,bnb]
    classifier_name_list = ['Gaussian NB','Multinomial NB','Bernoulli NB']
    return classifier_list, classifier_name_list


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


def get_hashed_features(tweet_list):
    feat = hashfeatures.FeatureHash(max_feature_num=5000)
    hash_feature_set = feat.get_feature_set(tweet_list)
    return hash_feature_set


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name,'-----------------'
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ", metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


filename = 'gender.csv'
gender_frame = pd.read_csv(filename)
filtered_gender_frame = gender_frame[gender_frame['gender'].isin(['male','female'])]
tweet_features = get_hashed_features(list(filtered_gender_frame['text'].values))
gender_labels = list(filtered_gender_frame['gender'].values)
print len(tweet_features), " : ", len(gender_labels)
X_train,X_test,y_train,y_test = train_test_split(tweet_features,gender_labels,test_size=0.1,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train, y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)


