import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,MaxPooling1D,BatchNormalization,PReLU,LeakyReLU,Dropout,LSTM,GRU


def get_deep_rnn(timesteps,feature_dim,output_dimension):
    model = Sequential()
    model.add(LSTM(32,input_shape=(timesteps,feature_dim),dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(GRU(32,recurrent_dropout=0.2,dropout=0.2))
    model.add(Dense(output_dimension,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


def get_deep_nn(input_dimension,output_dimension):
    model = Sequential()
    model.add(Dropout(rate=0.3,input_shape=(input_dimension,)))
    model.add(Dense(25,activation='relu',activity_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Dense(10,activation='relu'))
    model.add(PReLU())
    model.add(Dense(5,activation='relu'))
    model.add(LeakyReLU())
    model.add(Dense(output_dimension,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


def get_deep_conv_nn(input_shape,output_dimension):
    model = Sequential()
    model.add(Conv1D(filters=32,kernel_size=3,activation='relu',input_shape=input_shape,padding='same'))
    model.add(Conv1D(filters=32,kernel_size=3,activation='relu',padding='same'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Conv1D(filters=16,kernel_size=3,activation='relu',padding='same'))
    model.add(Conv1D(filters=16,kernel_size=3,activation='relu',padding='same'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(output_dimension,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


def get_ensemble_models():
    rf = RandomForestClassifier(n_estimators=51,max_depth=5,min_samples_split=3,random_state=42)
    grad = GradientBoostingClassifier(random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    bag = BaggingClassifier(n_estimators=51,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=51,random_state=42,max_depth=5)
    classifier_list = [rf,grad,ada,bag,extra]
    classifier_name_list = ['Random Forests','Gradient Boosting','AdaBoost','Bagging','Extra Trees']
    return classifier_list,classifier_name_list


def evaluate_models(trained_model,trained_model_name,X_test,y_test):
    predicted_values = trained_model.predict(X_test)
    print ('---------- For Model Name : ', trained_model_name, ' ---------\n')
    print (metrics.classification_report(y_test,predicted_values))
    print (metrics.accuracy_score(y_test,predicted_values))
    print (metrics.matthews_corrcoef(y_test,predicted_values))
    print ('-------------------------------------\n')


def parse_date_custom(date_list,date_separator='-',time_separator=':',space_separator=' '):
    year_list = list([])
    month_list = list([])
    day_list = list([])
    hour_list = list([])
    minute_list = list([])
    second_list = list([])
    for date_string in date_list:
        separated_date_list = str(date_string).split(space_separator)
        date_part = separated_date_list[0]
        time_part = separated_date_list[1]
        date_split = date_part.split(date_separator)
        time_split = time_part.split(time_separator)
        year_list.append(date_split[0])
        month_list.append(date_split[1])
        day_list.append(date_split[2])
        hour_list.append(time_split[0])
        minute_list.append(time_split[1])
        second_list.append(time_split[2])
    return year_list,month_list,day_list,hour_list,minute_list,second_list


filename = 'train_sample.csv'
train_frame = pd.read_csv(filename)
del train_frame['attributed_time']
class_labels = train_frame['is_attributed'].values
del train_frame['is_attributed']
train_frame['year'],train_frame['month'],train_frame['day'], \
train_frame['hour'],train_frame['minute'],train_frame['second'] = parse_date_custom(list(train_frame['click_time'].values))
del train_frame['click_time']
X_train,X_test,y_train,y_test = train_test_split(train_frame.values,class_labels,test_size=0.2,random_state=42)
classifier_list,classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(X_train,y_train)
    evaluate_models(classifier,classifier_name,X_test,y_test)

