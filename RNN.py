import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd

num_epochs = 100
num_batch = 128
time_step = 10
hidden_dim = 100

_file_path = "cleaned.csv"

def get_training_testing_data_from_csv(csv_file):
	"""
	return training feature matrix, dimensions = [number of weeks in 8 years, feature_dim]
	also return Labeling matrix Y, diemsnions = [number of weeks in 8 years, 1]
	also return X_test, Y_test
	type being numpy ndarray
	"""

	#some transformation here
	df = pd.read_csv(csv_file)
	#output a matrix of [number of weeks, feature_dim]
	filtered = df.dropna(axis=0, how='any')
	return filtered[["NUM. OF PROVIDERS", "PERCENT POSITIVE", "TOTAL PATIENTS"]].values, filtered[["TOTAL SPECIMENS"]].values

# get_training_testing_data_from_csv(_file_path)

X, Y = get_training_testing_data_from_csv(_file_path)

def transform_training_data(X):
	"""
	sliding window methods - takes in a giant matrix X, number_weeks * feature_dim
	return a 3D matrix, [num_example, time_step, feature_dim]
	"""
	#create a 3D numpy array
	x_train = np.zeros((X.shape[0] - time_step + 1, time_step ,X.shape[1]))
	for i in range(X.shape[0] - time_step + 1):
		x_train[i] = X[i: i + time_step, :, :]

	return x_train


# print(X.shape[1])
feature_dim = X.shape[1]
#X = np.reshape

#Instructions - remove first 9 y values,
# remove the null values in the y column need the same vertical length.

#need to divide X Y into x and y.

model = Sequential()
model.add(LSTM(hidden_dim, input_shape=(time_step, feature_dim), return_sequences = True))
model.add(Dense(100, activation = 'softmax'))
model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', merics = ['accuracy'])
model.fit(transform_training_data(X), Y, num_epochs, num_batch, validation_data = (x_test, y_test))
