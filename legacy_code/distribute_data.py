import pandas as pd
import sys
from sklearn.utils import shuffle
from legacy_code.common_data import *


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
    train = pd.read_csv('train.csv').values.astype(np.float32)
    # filter the data here
    train = np.array(filter_data_and_convert(train, binary))

    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    # Xtrain = train[:-1000, 1:] / 255
    # Ytrain = train[:-1000, 0:1].astype(np.int32)
    Ytrain = train[:-1000, 0:1].astype(np.int32)
    Xtest = train[-1000:, 1:] / 255
    Ytest = train[-1000:, 0:1].astype(np.int32)
    # change the value type
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    print('FinishReadingData')
    return Xtrain, Ytrain, Xtest, Ytest


if __name__ == '__main__':
    # read_data_from file
    # split the data into two parts
    print(cur_time_str())
    l1 = 4
    l2 = 8
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

    # n is number of samples and d is dimension of feature
    n, d = Xtrain.shape

    # B is batch size
    B = 32

    # t is number of batch. TODO: why timing it with 4?
    t = (n // B) * 4

    print('n=%d,d=%d,t=%d,b=%d' % (n, d, t, B))
    X = Xtrain
    y = Ytrain

    X0 = generate_random_matrix(n, d)
    X1 = X - X0
    y0 = generate_random_matrix(n, 1)
    y1 = y - y0

    dim_para = DimensionPara(n, d, t, B)

    # send two shares of X, Y to P0 and P1 respectively

    # send to P0
    send_to_P0(DistributeData(dim_para, X0, y0, Xtest, Ytest))
    print('SentDataToP0')
    # send to P1
    send_to_P1(DistributeData(dim_para, X1, y1, Xtest, Ytest))
    print('SentDataToP1')
    print(cur_time_str())
