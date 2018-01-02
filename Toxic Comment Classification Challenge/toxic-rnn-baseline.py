from keras.layers import Dense,LSTM,Bidirectional,Embedding
from keras.models import Sequential
from hashfeatures import FeatureHash
import pandas as pd


filename = 'train.csv'
toxic_frame = pd.read_csv(filename)
feature_hash_extractor = FeatureHash(max_feature_num=1000)
text = toxic_frame['comment_text'].values
del toxic_frame['id']
del toxic_frame['comment_text']
output_labels = toxic_frame.values
text_features = feature_hash_extractor.get_feature_set(text)

model = Sequential()
model.add(Embedding(input_dim=1000,input_length=1000,output_dim=300))
model.add(LSTM(300,dropout=0.2,recurrent_dropout=0.3))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

text_features = text_features.reshape(text_features.shape[0],1,text_features.shape[1])

model.fit(text_features,output_labels,validation_split=0.2)
