import numpy as np
import matplotlib.pyplot as plt

class MLP():
    #標準正規分布に従い、重み行列の作成
    def __init__(self, input_units, hidden_units, output_units):
        self.W1 = np.random.randn(input_units, hidden_units)
        self.b1 = np.random.randn(1, hidden_units)
        self.W2 = np.random.randn(hidden_units, output_units)
        self.b2 = np.random.randn(1, output_units)

    #シグモイド関数
    def sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    #シグモイド関数の微分
    def sigmoid_derivative(self, x):
        y = self.sigmoid(x) * (1 - self.sigmoid(x))
        return y

    #training function
    def train(self, X_train, y_train, epochs, learning_rate):
        print('training data')
        print('[0,0]->[0]')
        print('[0,1]->[1]')
        print('[1,0]->[1]')
        print('[1,1]->[0]')
        print('-----------------------')
        for i in range(epochs):
            m = X_train.shape[0]

            #Forward-Propagation
            layer_xh = np.dot(X_train, self.W1) + self.b1
            layer_hid = self.sigmoid(layer_xh)
            layer_ho = np.dot(layer_hid, self.W2)  + self.b2
            layer_out = self.sigmoid(layer_ho)

            #Back-Propagation
            dlayer_z2 = (layer_out - y_train)/m
            dW2 = np.dot(layer_hid.T, dlayer_z2)
            db2 = np.sum(dlayer_z2, axis=0 ,keepdims=True)

            dlayer_z1 = np.dot(dlayer_z2, self.W2.T) * self.sigmoid_derivative(layer_xh)
            dW1 = np.dot(X_train.T, dlayer_z1)
            db1 = np.sum(dlayer_z1, axis=0 ,keepdims=True)

            #update weight
            self.W2 -= learning_rate * dW2
            self.b2 -=  learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        print('result')
        print('[0,0]->' + str(layer_out[0]))
        print('[0,1]->' + str(layer_out[1]))
        print('[1,0]->' + str(layer_out[2]))
        print('[1,1]->' + str(layer_out[3]))
        return layer_out

# unit size
input_units, hidden_units, output_units = (2,2,1)
# input data
X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
# teacher data
y_train = np.array([ [0], [1], [1], [0]])
# epoch
epochs = 30000
#学習率
learning_rate = 0.3

loss = [] #コストを記録
mlp = MLP(input_units, hidden_units, output_units) #インスタンスを生成
mlp.train(X_train, y_train, epochs, learning_rate) #trainメソッドの呼び出し
