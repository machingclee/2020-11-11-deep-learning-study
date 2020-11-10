import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris_data = datasets.load_iris()

# define X = Y[0], the input data in the 0th layer
input_data = iris_data.data.T
num_of_data = input_data.shape[1]

# convert the format of target
_correct_data = iris_data.target
correct_data = np.zeros((3, num_of_data))
for i in range(num_of_data):
    correct_data[_correct_data[i], i] = 1

# normalize data, by finding mean and standard deviation first:
means = np.average(input_data, axis=1).reshape(-1, 1)
stds = np.std(input_data, axis=1).reshape(-1, 1)
input_data = (input_data - means) / stds


def relu(x):
    return np.where(x <= 0, 0, x)


def d_relu(x):
    return np.where(x <= 0, 0, 1)


index = np.arange(num_of_data)
index_train = index[index % 2 == 0]
index_test = index[index % 2 == 1]

input_train = input_data[:, index_train]
correct_train = correct_data[:, index_train]

input_test = input_data[:, index_test]
correct_test = correct_data[:, index_test]

wb_width = 0.1
eta = 0.01
epoch = 1000
batch_size = 8
interval = 100

n_in = 4
n_mid = 25
n_out = 3

n_train = input_train.shape[1]
n_test = input_test.shape[1]
batch_size = 8


n_batch = np.floor(n_train/batch_size).astype(int)


A = np.random.rand(100, 100)
A


class BaseLayer:
    def __init__(self, num_prev, num_curr):
        self.W = wb_width * np.random.randn(num_curr, num_prev)
        self.b = wb_width * np.random.randn(num_curr, 1)

        self.h_w = np.zeros((num_curr, num_prev)) + 1e-8
        self.h_b = np.zeros(num_curr).reshape(-1, 1) + 1e-8

    def update(self, eta):
        self.h_w = self.h_w + self.dW * self.dW
        self.h_b = self.h_b + self.db * self.db

        self.W = self.W - eta * self.dW / np.sqrt(self.h_w)
        self.b = self.b - eta * self.db / np.sqrt(self.h_b)


class MiddleLayer(BaseLayer):
    def forward(self, Y_prev):
        self.Y_prev = Y_prev

        self.U = np.dot(self.W, self.Y_prev) + self.b

        self.Y = relu(self.U)

    def backward(self, W_next, delta_next):
        self.delta = d_relu(self.U) * np.dot(W_next.T, delta_next)

        self.dW = np.dot(self.delta, self.Y_prev.T)
        self.db = np.sum(self.delta, axis=1, keepdims=True).reshape(-1, 1)


class OutputLayer(BaseLayer):
    def forward(self, Y_prev):
        self.Y_prev = Y_prev

        self.U = np.dot(self.W, Y_prev) + self.b
        self.Y = np.exp(self.U)/np.sum(np.exp(self.U), axis=0)

    def backward(self, Target):
        self.delta = self.Y - Target

        self.dW = np.dot(self.delta, self.Y_prev.T)
        self.db = np.sum(self.delta, axis=1).reshape(-1, 1)


middle_layer_1 = MiddleLayer(n_in, n_mid)
middle_layer_2 = MiddleLayer(n_mid, n_mid)
output_layer = OutputLayer(n_mid, n_out)


def get_error(target, num_of_target):
    return -np.sum(target * np.log(output_layer.Y + 1e-7))/num_of_target


train_error_epoach = []
train_error = []

test_error_epoach = []
test_error = []


def forward_propagation(X):
    middle_layer_1.forward(X)
    middle_layer_2.forward(middle_layer_1.Y)
    output_layer.forward(middle_layer_2.Y)


def backward_propagation(T):
    output_layer.backward(T)
    middle_layer_2.backward(output_layer.W, output_layer.delta)
    middle_layer_1.backward(middle_layer_2.W, middle_layer_2.delta)


def update_wb(eta):
    middle_layer_1.update(eta)
    middle_layer_2.update(eta)
    output_layer.update(eta)


for i in range(epoch):
    index_random = np.arange(n_train)
    np.random.shuffle(index_random)

    forward_propagation(input_train)
    train_error_epoach.append(i)
    train_error.append(get_error(correct_train, n_train))

    forward_propagation(input_test)
    test_error_epoach.append(i)
    test_error.append(get_error(correct_test, n_test))

    for j in range(n_batch):
        b_index = index_random[j*batch_size: (j+1)*batch_size]
        X = input_train[:, b_index]
        T = correct_train[:, b_index]

        forward_propagation(X)

        error_train = get_error(T, batch_size)

        backward_propagation(T)
        update_wb(eta)

plt.plot(train_error_epoach, train_error, label="training")
plt.plot(test_error_epoach, test_error, label="testing")
plt.legend()

forward_propagation(input_train)
count_train = np.sum(np.argmax(output_layer.Y, axis=0)
                     == np.argmax(correct_train, axis=0))

forward_propagation(input_test)
count_test = np.sum(np.argmax(output_layer.Y, axis=0)
                    == np.argmax(correct_test, axis=0))

print("accuracy train:", str(count_train/n_train*100)+"%")
print("accuracy test:", str(count_test/n_train*100)+"%")
