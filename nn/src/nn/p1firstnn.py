import numpy as np
from .zutils import *

def init_newtork():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    print(W1.shape)
    print(x.shape)
    print(b1.shape)

    a1 = np.dot(x, W1) + b1
    print(a1)
    z1 = sigmoid(a1)
    print(z1)
    a2 = np.dot(z1, W2) + b2
    print(a2)
    z2 = sigmoid(a2)
    print(z2)
    a3 = np.dot(z2, W3) + b3
    print(a3)
    y = identity_function(a3)
    print(y)
    return y

def main():
    network = init_newtork()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
