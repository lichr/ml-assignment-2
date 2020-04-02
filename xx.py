# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# ref:  https://gist.github.com/fxsjy/5574345
import numpy
import random
#from classification import *
import numpy as np
import time
from numpy import arange
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle



# %%

# mnist = fetch_openml('mnist_784', version=1, return_X_y=True)
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn import preprocessing
import pandas as pd


data, target = fetch_openml('mnist_784', version=1, return_X_y=True)

#mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
#print mnist.data.shape
# Trunk the data
n_train = 6000
n_test = 1000

# Define training and testing sets
indices = arange(len(data))
random.seed(0)

train_idx = arange(0,n_train)
test_idx = arange(n_train+1,n_train+n_test)


# X, Y1 = data[train_idx].transpose(), target[train_idx].transpose()
X, Y1 = data[train_idx].transpose(), target[train_idx]
# X_test, y_test = data[test_idx].transpose(), target[test_idx].transpose()

# Convert to "one-hot" vectors
s = pd.Series(Y1)
Y = pd.get_dummies(s).to_numpy().transpose()

print("X.shape: ", X.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Y shape: ", Y.shape)
print("First 5 training labels(one-hot): ", Y[:5]) # [5, 0, 4, 1, 9]


# %%



# %%
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


# %%
# This function initializes the parameters of our network
# We will have a network with two layers, thus we need parameters W1 and b1 for
# the first layer and W2 and b2 for the second layer
# Remember that the weights can't be initialized to 0 but rather small random values
# n_x represents the number of input features
# n_h represents the number of hidden units (in the hidden layer)
# n_y represents the number of output units
def init_params(n_x, n_h, n_y):
    np.random.seed(2) 
    W1 = np.random.randn(n_h, n_x)* 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)* 0.01
    b2 = np.zeros((n_y, 1))
    
    params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return params

# For ease of use we will store the parameters in a dictionary called params


# %%
# This function performs forward propagation
# it receives as parameters the matrix X containing the input features for the entire training set
# and the paramters of the network in the dictionary params
def forward_propagation(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    Z1 = np.dot(W1, X) + b1
    # Z1 = np.tanh(W1, X) + b1
    
    # A1 = sigmoid(Z1)
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    forwd = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, forwd

# We will store the Zi and Ai matrixes in a dictionary called forwd
# for ease of use we will also separately return A2 which corresponds with the output of the network


# %%
# Here we compute the cost function over the entire training set
# all we need is the predicted value by the network (Y_pred) 
# and the actual class of the training examples (Y)
def compute_cost(Y_pred, Y):
    # m = Y.shape[1] # number of example
    m = Y.shape[1] # number of example

    logprobs = np.multiply(np.log(Y_pred),Y) + np.multiply(np.log(1 - Y_pred), (1 - Y))
    cost = - (1/m) * np.sum(logprobs) 
    print ("cost is: ", cost)
    cost = float(np.squeeze(cost))  # makes sure cost is a real number.
    
    return cost


# %%
# This function performs backward propagation
# it calculates dW2, db2, dW1 and db1
def backward_propagation(params, forwd, X, Y):
    m = X.shape[1]
    W1 = params['W1']
    W2 = params['W2']

    A1 = forwd['A1']
    A2 = forwd['A2']
    
    dZ2 = A2 - Y  #dz in slide 10, result of applying the chain rule (see also slide 18)
    dW2 = 1/m*np.dot(dZ2, A1.T)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)
    # dZ1 = np.multiply(np.dot(W2.T, dZ2), A1 - np.power(A1, 2)) # derivative of the sigmoidal function is a(1-a) see slides 10 and 18
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    
    dW1 = 1/m*np.dot(dZ1, X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
# This function returns the gradients in a dictionary


# %%
# This function uses the gradients and the learning rate to update the parameters
def update_params(params, grads, learn_rate = 1.2):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # tami: gradients (theta = theta - alpha*dw (derivative of w1))
    W1 = W1 - learn_rate*dW1
    b1 = b1 - learn_rate*db1
    W2 = W2 - learn_rate*dW2
    b2 = b2 - learn_rate*db2

    params = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2}
    
    return params
# It returns the updated parameters


# %%
# Here we create and train the actual Neural Network model
# We receives the dataset features X and classes Y
# we receive the number of hidden units as a hyperparameter (n_h)
# and we also get as a hyperparameter how many iterations to train

# tami: num_iterations is the iteration of the gradient iteration step times
def nn_model(X, Y, n_h, num_iterations = 10, print_cost=False):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0] 

    params = init_params(n_x, n_h, n_y)

    # This loop is to perform the forward and backward iterations
    for i in range(0, num_iterations):
        # Inside the loop all the computations (forward and backward computations) are vectorized
        # Tami: A2 is the second layer result
        A2, forwd = forward_propagation(X, params) 
        cost = compute_cost(A2, Y)
        grads = backward_propagation(params, forwd, X, Y)
        params = update_params(params, grads)

        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return params
# The method returns the parameters of the network, which is what is learned
# ... and all we need to predict!


# %%
# This method uses the learned parameters and a set of input values to perform a prediction
def predict(params, X):
    Y_pred, forwd = forward_propagation(X, params)
    predictions = np.argmax(Y_pred, axis=0)  # in a binary classification problem we predict 1 if the output (y_pred) is larger than 0.5
    return predictions


# %%
# Let's train a neural network with 4 hidden units for 10.000 iterations (epochs)
params = nn_model(X, Y, n_h = 20, num_iterations = 10000, print_cost=True)

# And now let's use the prediction to plot the decision boundary
# plot_decision_boundary(lambda x: predict(params, x.T), X, Y.ravel())
# plt.title("Decision Boundary for hidden layer size " + str(4))


# %%
# And this is the accuracy we got
predictions = predict(params, X)
print ('Accuracy: %d' % float(np.sum(Y1.astype(np.int) == predictions) / Y1.size * 100) + '%')

