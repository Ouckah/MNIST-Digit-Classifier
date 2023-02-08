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


# initialize weights and biases
def init_params():
    '''
    Randomly generates the weights and biases for the first generation neural network.
    '''

    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2