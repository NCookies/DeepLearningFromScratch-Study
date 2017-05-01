import numpy as np
import matplotlib.pyplot as plt


def step_function(input_x):
    y = input_x > 0
    return y.astype(np.int)
    # return np.array(input_x > 0, dtype=np.int)


def sigmoid_function(input_x):
    return 1 / (1 + np.exp(-input_x))


def relu_function(input_x):
    return np.maximum(0, input_x)


x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid_function(x)
y3 = relu_function(x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)

plt.ylim(-0.1, 1.1)
plt.show()
