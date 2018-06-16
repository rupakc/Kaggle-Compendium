import pandas as pd
import hashfeatures
import preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split


def get_classifiers():
    rf = RandomForestClassifier(n_estimators=51, random_state=42)
    linear_svm = LinearSVC()
    classifier_list = [rf,linear_svm]
    classifier_name_list = ['Random Forest','Linear SVM']
    return classifier_list,classifier_name_list


def report_classification_metrics(trained_model, X_test, y_test):
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print metrics.accuracy_score(y_test,predicted_values)


filename = 'C:\\Users\\rupachak\\Desktop\\Kaggle Data\\DonorsChoose\\train.csv'
train_frame = pd.read_csv(filename)
train_frame = train_frame.head(25000)
class_labels = list(train_frame['project_is_approved'].values)
summary_text = preprocess.text_clean_pipeline_list(list(train_frame['project_resource_summary'].values))
feature_set = hashfeatures.FeatureHash(max_feature_num=2000).get_feature_set(summary_text)

del train_frame
del summary_text
X_train, X_test, y_train, y_test = train_test_split(feature_set,class_labels,test_size=0.2,random_state=42)
del class_labels
del feature_set

classifier_list,classifier_name_list = get_classifiers()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print "---------- For Classifier: ", classifier_name, " --------------------\n"
    report_classification_metrics(classifier, X_test, y_test)
