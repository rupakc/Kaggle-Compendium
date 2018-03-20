import os
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,BatchNormalization,LeakyReLU
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


def get_conv_model(channels,image_height,image_width,num_classes):
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),input_shape=(channels,image_height,image_width),activation='relu',padding='same'))
    model.add(Conv2D(32,kernel_size=(3,3),padding='same'))
    model.add(Dropout(rate=0.3))
    model.add(MaxPooling2D(padding='same'))
    model.add(Conv2D(16,kernel_size=(3,3),padding='same'))
    model.add(Conv2D(16, kernel_size=(3, 3),padding='same'))
    model.add(Dropout(rate=0.3))
    model.add(MaxPooling2D(padding='same'))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


plant_name_list = ['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen',
            'Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse','Small-flowered Cranesbill','Sugar beet']
base_folder_name = 'train'
path = os.path.join(base_folder_name,plant_name_list[0])
master_image_list = list([])
master_image_label_list = list([])

for plant_name in plant_name_list:
    image_path = os.path.join(base_folder_name,plant_name)
    image_file_names = os.listdir(image_path)
    for image_file in image_file_names:
        img_array = image.img_to_array(image.load_img(os.path.join(image_path,image_file),target_size=(50,50))).reshape(3,50,50)
        master_image_list.append(img_array)
        master_image_label_list.append(plant_name)


encoded_values = to_categorical(LabelEncoder().fit_transform(master_image_label_list),num_classes=len(set(plant_name_list)))
print(encoded_values.shape)
model_checkpoint = ModelCheckpoint('best_plant_model.h5',save_best_only=True,monitor='val_acc')
early_stop = EarlyStopping(patience=15,monitor='val_acc')
model = get_conv_model(3,50,50,len(set(plant_name_list)))
print(model.summary())
image_data = np.array(master_image_list)
del master_image_list
del master_image_label_list
model.fit(image_data,encoded_values,batch_size=500,epochs=10,
          validation_split=0.2,callbacks=[model_checkpoint,early_stop])


