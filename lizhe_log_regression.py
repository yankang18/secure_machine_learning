import sys
import time

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def glorot_normal(fan_in, fan_out):
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, stddev, (fan_in, fan_out))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def approx_sigmoid(x):
    global x3_max
    global x3_min
    if x3_max < abs(x).max():
        x3_max = abs(x).max()

    if x3_min > abs(x).min():
        x3_min = abs(x).min()
    return 0.5 + 0.5 * 0.1 * x


class LogisticRegression:
    def __init__(self, feature_dim, learning_rate):
        self.feature_dim = feature_dim
        # self.W should be column vetor
        self.W = self.__init_weights(self.feature_dim, 1)
        self.learning_rate = learning_rate
        self.stoped_training = False

    def __init_weights(self, feature_dim, dim):
        return np.random.normal(0, 0.1, (feature_dim, dim))

    def train(self, X, Y):
        batch_size, _ = X.shape
        forward = approx_sigmoid(np.dot(X, self.W))
        # Y.reshape(batch_size, 1)
        # print (Y)
        # print ('Y.shape=',Y.shape)
        # print ('subtract.shape=(%d, %d)' % ((forward-Y).shape))
        # print ('dot.shape=(%d,%d)' % (np.dot(X.transpose(), (forward-Y)).shape))
        backward = (self.learning_rate / batch_size) * np.dot(X.transpose(), forward - Y)
        # print ('backward.shape=(%d,%d)' % (backward.shape))
        self.W -= backward

    def predict(self, X):
        return sigmoid(np.dot(X, self.W))

    def start_training(self):
        self.stoped_training = False

    def stop_training(self):
        self.stoped_training = True

    def is_stop_training(self):
        return self.stoped_training


class StopCheck:
    def __init__(self, model, patience=60):
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.model = model

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        self.model.start_training()

    def on_iteration_end(self, epoch, batch, current):
        if current > self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training()


def get_binary_labels(Y, binary=(0, 1)):
    # Y is column vector
    Y_b = [0 if Y[i] == binary[0] else 1 for i in range(Y.shape[0])]
    return np.array(Y_b)


def to_binary_int(v):
    if v <= 0.5:
        return 0
    elif v > 0.5:
        return 1
    else:
        return 2


def compute_accuracy(Y_targets, Y_pred):
    # print (Y_targets.shape, Y_pred.shape)
    # Yb_pred = to_binary_int(np.array(Y_pred))
    dim = Y_targets.shape[0];
    cnt = 0
    for i in range(dim):
        if Y_targets[i][0] == to_binary_int(Y_pred[i][0]):
            cnt = cnt + 1
    return 1.0 * cnt / dim


def filter_data_and_convert(data, binary):
    converted = []
    for x in data:
        if x[0] == binary[0]:
            x[0] = 0
            converted.append(x)
        elif x[0] == binary[1]:
            x[0] = 1
            converted.append(x)
        else:
            pass
    return converted


def getKaggleMNIST(binary=(0, 1)):
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv('train.csv').values.astype(np.float32)
    # filter the data here
    # train = np.array([x for x in train if x[0]==binary[0] or x[0]==binary[1]])
    train = np.array(filter_data_and_convert(train, binary))

    train = shuffle(train)

    # why divides by 255?
    Xtrain = train[:-1000, 1:] / 255
    Ytrain = train[:-1000, 0:1].astype(np.int32)
    Xtest = train[-1000:, 1:] / 255
    Ytest = train[-1000:, 0:1].astype(np.int32)
    # change the value type
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    print('FinishReadingData')
    return Xtrain, Ytrain, Xtest, Ytest


def cur_time_str():
    return time.asctime(time.localtime())


x3_max = -1000
x3_min = 1000
if __name__ == '__main__':
    print(cur_time_str())
    l1 = 0;
    l2 = 1
    if len(sys.argv) == 1:
        # no binary label
        pass
    elif len(sys.argv) == 3:
        # pass
        l1 = int(sys.argv[1])
        l2 = int(sys.argv[2])
    else:
        sys.exit(-1)
    selected_label = (l1, l2)
    print('selected_label=(%d,%d)' % (selected_label))
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST(selected_label)
    learning_rate = 0.001
    # batch_size = 128
    batch_size = 32
    # N: size of dataset, D: feature dim
    N, D = Xtrain.shape
    n_batches = (N // batch_size) * 4
    epochs = 1
    global_step = 0
    learn_model = LogisticRegression(D, learning_rate)
    stop_check = StopCheck(learn_model)
    stop_check.on_train_begin()
    print('batch_size=%d, learning_rate=%.5f, data_size=%d, feature_dim=%d' % (
    batch_size, learning_rate, Xtrain.shape[0], Xtrain.shape[1]))
    # get data set X and Y
    # test set X_test, Y_targets
    for ep in range(epochs):
        for j in range(n_batches):
            i = j % (N // batch_size)
            global_step += 1
            X_batch = np.array(Xtrain[i * batch_size: i * batch_size + batch_size])
            Y_batch = np.array(Ytrain[i * batch_size: i * batch_size + batch_size])
            learn_model.train(X_batch, Y_batch)
            if i % 5 == 0:
                Y_pred = learn_model.predict(Xtest)
                acc = compute_accuracy(Ytest, Y_pred)
                print("global_step=%04d,epoch=%04d,batch=%04d,acc=%.5f" % (global_step, ep, i, acc))
                stop_check.on_iteration_end(ep, i, acc)
                # predict with test set
                # compute_accuracy
            if learn_model.is_stop_training():
                print('StopTraingning')
                break
        if learn_model.is_stop_training():
            print('StopTraingning')
            break
    print(cur_time_str())

    # print (x3_max)
    # print (x3_min)
