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
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv1D,MaxPooling1D,Flatten
from keras.utils import np_utils
import numpy as np


def get_cnn_model(input_feature_dim):
    model = Sequential()
    model.add(Conv1D(64,3,input_shape=(input_feature_dim,1),padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv1D(32,3,padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv1D(32, 3, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


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
    print '--------- For Model : ', trained_model_name
    predicted_values = trained_model.predict(X_test)
    print metrics.classification_report(y_test,predicted_values)
    print "Accuracy Score : ",metrics.accuracy_score(y_test,predicted_values)
    print "---------------------------------------\n"


feature_filenames = ['YouTube_visual.csv','YouTube_vocal.csv','YouTube_acoustic.csv']
class_label_filename = 'YouTube_sentiment_label.csv'
class_labels = pd.read_csv(class_label_filename,header=None)
dataframe_list = list([])
for feature_filename in feature_filenames:
    df = pd.read_csv(feature_filename,header=None)
    dataframe_list.append(df.values)

combined_features = reduce(lambda x,y:np.hstack((x,y)),dataframe_list)
del dataframe_list
X = combined_features
y = class_labels.values

X = X.reshape(X.shape[0],X.shape[1],1)
y = np_utils.to_categorical(y,2)

model = get_cnn_model(X.shape[1])
model.fit(X,y,validation_split=0.1,batch_size=50,epochs=150,verbose=2)

