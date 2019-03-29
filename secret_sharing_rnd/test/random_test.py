import numpy as np
import pandas as pd
from sklearn.utils import shuffle

if __name__ == '__main__':

    df = pd.read_csv("../data/breast_homo_guest.csv")
    data = df.values
    print(df.head(5))
    print(data.shape)
    # print(data)

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
    # print(Xtrain)
    # print(Ytrain)
