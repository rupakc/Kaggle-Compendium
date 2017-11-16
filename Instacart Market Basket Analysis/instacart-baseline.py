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


def print_evaluation_metrics(trained_model,trained_model_name,X_test,y_test):
    print '--------- For Model : ', trained_model_name,' ---------------\n'
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


order_product_filename = 'order_products__train.csv'
orders_filename = 'orders.csv'

order_product_frame = pd.read_csv(order_product_filename)
order_frame = pd.read_csv(orders_filename)

order_master_frame = pd.merge(order_product_frame,order_frame,how='outer',left_on='order_id',right_on='order_id')
columns_to_drop = ['user_id','eval_set','order_id','reordered']
target_class_labels = order_master_frame['reordered'].values
order_master_frame.drop(columns_to_drop,axis=1,inplace=True)
del order_product_frame
del order_frame
order_master_frame_values = Imputer().fit_transform(order_master_frame.values)
target_class_labels = Imputer().fit_transform(target_class_labels.reshape(-1,1))
target_class_labels = map(lambda x:int(x),target_class_labels)
X_train,X_test,y_train,y_test = train_test_split(order_master_frame_values,target_class_labels,test_size=0.2,random_state=42)
classifier_list, classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    print_evaluation_metrics(classifier,classifier_name,X_test,y_test)
