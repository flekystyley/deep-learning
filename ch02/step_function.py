# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    # 0を境に、出力が0から1に切り替わる
    return np.array(x > 0, dtype=np.int)

# -5.0から5.0までの範囲を0.1刻みで作成する[-5.0...4.9] まで
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 図で描画するy軸の範囲を指定
plt.show()
