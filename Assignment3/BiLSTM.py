import re
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

location = '/gdrive/My Drive/NLP CS 613/Assignment3/'

with open(location+"train_dict_small1045.pkl", "rb") as f:
	train_dict = pickle.load(f)

with open(location+"test_dict.pkl", "rb") as f:
	test_dict = pickle.load(f)

X_train = []
y_train = []

X_test = []
y_test = []

for key in train_dict:
	X_train.append(train_dict[key]["text"])
	y_train.append(train_dict[key]["senti"])

for key in test_dict:
	X_test.append(test_dict[key]["text"])
	y_test.append(test_dict[key]["senti"])

cvec = CountVectorizer()
Xcvec = cvec.fit_transform(X_train)
df = pd.DataFrame(Xcvec.toarray(), columns=cvec.get_feature_names())
tfidf_transf = TfidfTransformer()
X_tfidf_train = tfidf_transf.fit_transform(df)
y_train = np.array(y_train)

y_test = np.array(y_test)
Xcvec = cvec.transform(X_test)
df = pd.DataFrame(Xcvec.toarray(), columns=cvec.get_feature_names())
x_test = tfidf_transf.fit_transform(df)

max_features = 500
maxlen = 5030
batch_size = 32

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')

model.fit(X_tfidf_train[:-100], y_train[:-100], batch_size=batch_size, epochs=4, validation_data=[X_tfidf_train[-100:], y_train[-100:]])
score, acc = model.evaluate(x_test, y_test, batch_size=32)

print("Accuracy: ", acc)
