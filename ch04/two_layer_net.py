# coding: utf-8
sys.path.append(os.pairdir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    ## initialize params
    ## - input_size (入力のニューロン数)
    ## - hidden_size (隠れ層のニューロン数)
    ## - output_size (出力層のニューロン数)
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        # params (ニューラルネットワークのパラメータを保持)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    ## predict (認識、推論を行う)
    ## x は 画像データ
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) * b2
        y = softmax(a2)
    
        return y

    ## loss (損失関数の値を求める)
    ## xは画像のデータ、tは正解ラベル
    def loss(self, x, t):
        y = self.predict(x)
    
        return cross_entropy_error(y, t)

    ## accuray (認識精度を求める)
    ## xは画像データ、　tは正解ラベル
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
    
        return accuracy

    ## numerical_gradient (重みパラメーターに対する、勾配を求める)
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
    
        # grads (勾配を保持するディクショナリ)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    ## gradient (重みパラメーターに対する、勾配を求める) numerical_gradientの高速版
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
    
        batch_num = x.shape[0]
    
    # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads