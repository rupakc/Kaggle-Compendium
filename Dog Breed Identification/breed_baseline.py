import pandas as pd
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_one_hot_encoded_classes(class_list,n_classes):
    label_encoded_classes = LabelEncoder().fit_transform(class_list)
    one_hot_encoded = np_utils.to_categorical(label_encoded_classes,num_classes=n_classes)
    return one_hot_encoded


def get_conv_model(input_shape=(3,250,250),num_classes=120):
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape,padding='same'))
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


folder_name = 'train/'
file_extension = '.jpg'
label_filename = 'labels.csv'
label_frame = pd.read_csv(label_filename)
breed_classes = label_frame['breed'].values
file_id_values = label_frame['id'].values
master_image_list = list([])
n_classes = len(label_frame['breed'].value_counts())
output_values = get_one_hot_encoded_classes(breed_classes,n_classes)
del label_frame
model = get_conv_model()
print model.summary()
print output_values.shape

for file_id in file_id_values:
    img_filename = folder_name + str(file_id) + file_extension
    img = image.load_img(img_filename,target_size=(250,250))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape(3,250,250)
    master_image_list.append(img_array)

print len(master_image_list)
image_data = np.array(master_image_list)
print image_data.shape
del master_image_list
X_train,X_test,y_train,y_test = train_test_split(image_data,output_values,test_size=0.2,random_state=42)
model.fit(X_train,y_train,batch_size=250,epochs=250)
print model.evaluate(X_test,y_test)
