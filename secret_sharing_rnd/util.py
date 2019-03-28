import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def assert_matrix(M1, M2):
    assert M1.shape == M2.shape
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            # print(M1[i][j], M2[i][j])
            assert round(M1[i][j], 4) == round(M2[i][j], 4)


def read_data_2(file_path):
    df = pd.read_csv(file_path)
    data = df.values
    print("data shape:", data.shape)
    train = shuffle(data)

    Xtrain = train[:, 2:]
    Ytrain = train[:, 1].astype(np.int32)
    Ytrain = Ytrain.reshape(-1, 1)
    # Xtest = train[-1000:, 1:]
    # Ytest = train[-1000:, 0:1].astype(np.int32)
    # change the value type
    # Xtrain = Xtrain.astype(np.float32)
    # Xtest = Xtest.astype(np.float32)
    print("Xtrain shape", Xtrain.shape)
    print("Ytrain shape", Ytrain.shape)
    return Xtrain, Ytrain


def read_data(file_path, file_path_2):
    df = pd.read_csv(file_path)
    data_1 = df.values
    # print(df.head(5))
    print(data_1.shape)
    # print(data_1)
    df_2 = pd.read_csv(file_path_2)
    data_2 = df_2.values

    data = np.concatenate((data_1, data_2), axis=0)

    print("data shape:", data.shape)
    train = shuffle(data)

    Xtrain = train[:, 2:]
    Ytrain = train[:, 1].astype(np.int32)
    Ytrain = Ytrain.reshape(-1, 1)
    # Xtest = train[-1000:, 1:]
    # Ytest = train[-1000:, 0:1].astype(np.int32)
    # change the value type
    # Xtrain = Xtrain.astype(np.float32)
    # Xtest = Xtest.astype(np.float32)
    print("Xtrain shape", Xtrain.shape)
    print("Ytrain shape", Ytrain.shape)
    return Xtrain, Ytrain
