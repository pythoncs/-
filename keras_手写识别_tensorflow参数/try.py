# import numpy as np
# # from keras.models import Sequential
# # from keras.layers import Dense,Dropout,Activation
# # import keras
# # from tf_utils import load_dataset
# #
# # model = Sequential()
# # model.add(Dense(25,activation='relu',input_dim=12288))
# # # model.add(Dropout(0.5))
# # model.add(Dense(12,activation='relu'))
# # # model.add(Dropout(0.3))
# # model.add(Dense(6,activation='softmax'))
# # sgd = keras.optimizers.Adam(lr=0.001,decay=1e-6)
# # model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
# # x_train_orig,y_train_orig,x_test_orig,y_test_orig,classes = load_dataset()
# # x_train_flatten = x_train_orig.reshape(x_train_orig.shape[0],-1)/255
# # x_test_flatten = x_test_orig.reshape(x_test_orig.shape[0],-1)/255
# # y_train_orig = y_train_orig.reshape(-1,1)
# # y_test_orig = y_test_orig.reshape(-1,1)
# # print('x_train_flatten.shape:',x_train_flatten.shape)
# # print('x_test_flatten.shape:',x_test_flatten.shape)
# #
# # one_hot_labels = keras.utils.to_categorical(y_train_orig,num_classes=6)
# # one_hot_labels2 = keras.utils.to_categorical(y_test_orig,num_classes=6)
# # model.fit(x_train_flatten,one_hot_labels,epochs=30,batch_size=64)
# # # model.fit(x_test_flatten,one_hot_labels2,epochs=300,batch_size=64)
# # w = model.get_weights()
# # print(w)
# # # score = model.evaluate(x_test_flatten, one_hot_labels2,sample_weight=w ,batch_size=32)

# import numpy as np
# a = np.fours((2,1))
# print(a)
from sklearn import preprocessing
a = ['中国','北京','牛逼']
b = ['河北','邯郸','中国','背景','北京']
# le =preprocessing.LabelEncoder()
# c = le.fit_transform(a)
# c2 = le.fit_transform(b)
# print(int('中国'))
# print(c2)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
c = [['中国','北京','牛逼'],['河北','邯郸','中国','背景','北京']]
for i in range(len(c)):
    c[i] = ' '.join(c[i])
vectorize = CountVectorizer(min_df=1e-5)  #去低频词
x = vectorize.fit_transform(c)
print(x.toarray())
print(x)
print(vectorize.vocabulary_)