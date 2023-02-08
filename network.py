import numpy as np
import pandas as pd


# retrieve testing and training data
test_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')


# testing data
test_data = np.array(test_data)

test_data = test_data.T # transpose the data
X_test = test_data
X_test = X_test / 255

# training data
train_data = np.array(train_data)
np.random.shuffle(train_data) # shuffle the data around

train_data = train_data.T # transpose the data
Y_train = train_data[0]
X_train = train_data[1:]
X_train = X_train / 255