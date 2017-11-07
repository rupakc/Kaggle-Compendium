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


def label_encode_frame(dataframe):
    columns = dataframe.columns
    encoder = LabelEncoder()
    for column in columns:
        if type(dataframe[column][328]) is str:
            dataframe[column] = encoder.fit_transform(dataframe[column].values)
    return dataframe


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


def split_date(list_of_date_string, format='yyyy-mm-dd hh:mm:ss'):
    month_list = list([])
    day_list = list([])
    year_list = list([])
    hour_list = list([])
    minute_list = list([])
    second_list = list([])
    for date_string in list_of_date_string:
        date_part,time_part = date_string.strip().split(' ')
        date_list = date_part.split('-')
        time_list = time_part.strip().split(':')
        month_list.append(date_list[1])
        day_list.append(date_list[2])
        year_list.append(date_list[0])
        hour_list.append(time_list[0])
        minute_list.append(time_list[1])
        second_list.append(time_list[2])
    return month_list,day_list,year_list,hour_list,minute_list,second_list


filename = 'training.csv'
train_frame = pd.read_csv(filename)
train_frame.dropna(inplace=True)
class_label = list(train_frame['correct'].values)
print train_frame.columns
print train_frame.head(3)

columns_to_delete = ['outcome','correct','date_of_test']
date_columns_to_encode = ['round_started_at','answered_at','deactivated_at']
train_frame.drop(columns_to_delete,axis=1,inplace=True)

print len(train_frame)

for index,column_value in enumerate(date_columns_to_encode):
    train_frame[column_value +'_m'],train_frame[column_value + '_d'],train_frame[column_value + '_y'],\
    train_frame[column_value + '_h'], train_frame[column_value + '_m'], train_frame[column_value + '_s'] = split_date(list(train_frame[column_value].values))

train_frame.drop(date_columns_to_encode,axis=1,inplace=True)
train_frame = label_encode_frame(train_frame)
X_train,X_test,y_train,y_test = train_test_split(train_frame.values,class_label,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()

for classifier, classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
