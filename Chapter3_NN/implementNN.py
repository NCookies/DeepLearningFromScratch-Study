import numpy as np
from Chapter3_NN.activation_function import *


def init_network():
    """
    1층 레이어의 노드 : 2개 (input)
    2층 레이어의 노드 : 3개
    3층 레이어의 노드 : 2개
    4층 레이어의 노드 : 2개 (output)
    
    W : 가중치(weight)
    b : 편향(bias)
    :return: 
    """
    network = dict()

    # 1층 레이어
    network['W1'] = np.array([[.1, .3, .5], [.2, .4, .6]])
    network['b1'] = np.array([.1, .2, .3])

    # 2층 레이어
    network['W2'] = np.array([[.1, .4], [.2, .5], [.3, .6]])
    network['b2'] = np.array([.1, .2])

    # 3층 레이어
    network['W3'] = np.array([[.1, .3], [.2, .4]])
    network['b3'] = np.array([.1, .2])

    return network


def identity_function(a):
    return a


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1     # XW + b
    z1 = sigmoid(a1)            # activation function 을 적용한 값
                                # 다음 레이어의 input(X) 값이 됨

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([.1, .5])
    y = forward(network, x)

    print(y)
