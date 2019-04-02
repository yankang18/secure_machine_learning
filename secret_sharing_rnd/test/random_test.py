import numpy as np

if __name__ == '__main__':

    # df = pd.read_csv("../data/breast_homo_guest.csv")
    # data = df.values
    # print(df.head(5))
    # print(data.shape)
    # # print(data)
    #
    # train = shuffle(data)
    #
    # Xtrain = train[:, 2:]
    # Ytrain = train[:, 1].astype(np.int32)
    # Ytrain = Ytrain.reshape(-1, 1)
    # # Xtest = train[-1000:, 1:]
    # # Ytest = train[-1000:, 0:1].astype(np.int32)
    # # change the value type
    # # Xtrain = Xtrain.astype(np.float32)
    # # Xtest = Xtest.astype(np.float32)
    # print("Xtrain shape", Xtrain.shape)
    # print("Ytrain shape", Ytrain.shape)
    # # print(Xtrain)
    # # print(Ytrain)

    X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    X = X.reshape(3, 1, 3)
    Y = Y.reshape(3, 3, 3)

    print("X", X, X.shape)
    print("Y", Y, Y.shape)

    Z = np.matmul(X, Y)
    print("Z", Z, Z.shape)

    Z2 = np.dot(X, Y)
    print("Z2", Z2, Z2.shape)
