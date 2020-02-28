# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

# シグモイド関数
def sigmoid(x):
    # np.exp(-x) == exp(-x)
    # 限界値が決まっているので、分母が分子に近づくほど1に近づく
    return 1 / (1 + np.exp(-x)) # 1 + e^-5.0

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()