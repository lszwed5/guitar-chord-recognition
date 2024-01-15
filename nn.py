import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, LSTM
from keras.optimizers import Adam, RMSprop, Nadam, SGD, Lion
from keras.models import clone_model

label_map = {'c': 0, 'd': 1, 'g': 2}

c_data = pd.read_csv('data/c.csv', sep=',', header=None)
c_target = pd.DataFrame([[1, 0, 0]] * len(c_data), columns=None)

print(len(c_target))

msk = np.random.rand(len(c_data)) < 0.9
X_train, X_test = c_data[msk], c_data[~msk]
Y_train, Y_test = c_target[msk], c_target[~msk]

timesteps = 17220
features = 3
X_train = X_train.values.reshape(-1, timesteps, features)
X_test = X_test.values.reshape(-1, timesteps, features)

model = Sequential()
# model.add(Flatten(input_shape=(17220, 3)))
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(timesteps, features)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, Y_train, epochs=100)