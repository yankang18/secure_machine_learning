import configparser
import logging
import pickle
import socket
import threading
import time

import numpy as np

learning_rate = 0.001
approx_rate = 0.01


# approx_rate = 0.05
def init_w(d):
    # column vector
    return np.random.normal(0, 0.01, (d, 1))


class DimensionPara:
    def __init__(self, n, d, t, B):
        self.n = n
        self.d = d
        self.t = t
        self.B = B


class P0ReconstructW:
    def __init__(self, w0):
        self.data_type = 'P0ReconstructW'
        self.w0 = w0


class P1ReconstructW:
    def __init__(self, w1):
        self.data_type = 'P1ReconstructW'
        self.w1 = w1


class P1ReconstructFStarj:
    def __init__(self, j, fstar1):
        self.data_type = 'P1ReconstructFStarj'
        self.j = j
        self.fstar1 = fstar1


class P0ReconstructFStarj:
    def __init__(self, j, fstar0):
        self.data_type = 'P0ReconstructFStarj'
        self.j = j
        self.fstar0 = fstar0


class P1ReconstructFj:
    def __init__(self, j, f1):
        self.data_type = 'P1ReconstructFj'
        self.j = j
        self.f1 = f1


class P0ReconstructFj:
    def __init__(self, j, f0):
        self.data_type = 'P0ReconstructFj'
        self.j = j
        self.f0 = f0


class P0BeaverData:
    def __init__(self, M0, M00, M01, N0, N00, N01, P0, P00, P01):
        self.M0 = M0
        self.M00 = M00
        self.M01 = M01
        self.N0 = N0
        self.N00 = N00
        self.N01 = N01
        self.P0 = P0
        self.P00 = P00
        self.P01 = P01


class P1BeaverData:
    def __init__(self, M1, M10, M11, N1, N10, N11, P1, P10, P11):
        self.M1 = M1
        self.M10 = M10
        self.M11 = M11
        self.N1 = N1
        self.N10 = N10
        self.N11 = N11
        self.P1 = P1
        self.P10 = P10
        self.P11 = P11


class P0StartTraining:
    def __init__(self):
        self.data_type = 'P0StartTraining'


class P1StartTraining:
    def __init__(self):
        self.data_type = 'P1StartTraining'


class P0ReconstructE:
    def __init__(self, E0):
        self.data_type = 'P0ReconstructE'
        self.E0 = E0


class P1ReconstructE:
    def __init__(self, E1):
        self.data_type = 'P1ReconstructE'
        self.E1 = E1


class DistributeData:
    def __init__(self, dim_para, X, y, Xtest, Ytest):
        self.data_type = 'DistributeData'
        self.dim_para = dim_para
        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.Ytest = Ytest


class P0ToP1:
    def __init__(self, M01, N01, P01):
        self.data_type = 'P0ToP1'
        self.M01 = M01
        self.N01 = N01
        self.P01 = P01


class P1ToP0:
    def __init__(self, M10, N10, P10):
        self.data_type = 'P1ToP0'
        self.M10 = M10
        self.N10 = N10
        self.P10 = P10


class P0ToCarlo:
    def __init__(self, M00, N00, P00, dim_para):
        self.data_type = 'P0ToCarlo'
        self.dim_para = dim_para
        self.M00 = M00
        self.N00 = N00
        self.P00 = P00


class P1ToCarlo:
    def __init__(self, M11, N11, P11, dim_para):
        self.data_type = 'P1ToCarlo'
        self.dim_para = dim_para
        self.M11 = M11
        self.N11 = N11
        self.P11 = P11


class CarloToP0:
    def __init__(self, O0, Q0):
        self.data_type = 'CarloToP0'
        self.O0 = O0
        self.Q0 = Q0


class CarloToP1:
    def __init__(self, O1, Q1):
        self.data_type = 'CarloToP1'
        self.O1 = O1
        self.Q1 = Q1


def generate_random_matrix(r, c):
    # TODO: why generate random number like this??
    a1 = np.random.rand(r, c)
    a2 = -np.random.rand(r, c)
    return a1 + a2


def generate_random_3_dim_matrix(r, c, z):
    a1 = np.random.rand(r, c, z)
    a2 = -np.random.rand(r, c, z)
    return a1 + a2


def dump_obj(obj):
    return pickle.dumps(obj, protocol=0)


def load_obj(data):
    return pickle.loads(data)


def client(ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        sock.sendall(message)


class IpConfig:
    def __init__(self, P0, P1, Carlo):
        self.P0_ip = P0[0]
        self.P0_port = P0[1]
        self.P0 = P0

        self.P1_ip = P1[0]
        self.P1_port = P1[1]
        self.P1 = P1

        self.Carlo_ip = Carlo[0]
        self.Carlo_port = Carlo[1]
        self.Carlo = Carlo


def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    default_section = config['DEFAULT']
    P0_address, P0_port = default_section['P0Address'], int(default_section['P0Port'])
    P1_address, P1_port = default_section['P1Address'], int(default_section['P1Port'])
    Carlo_address, Carlo_port = default_section['CarloAddress'], int(default_section['CarloPort'])
    return IpConfig((P0_address, P0_port), (P1_address, P1_port), (Carlo_address, Carlo_port))


def send_to_P0(data):
    P0_client = read_config().P0
    client(P0_client[0], P0_client[1], dump_obj(data))


def send_to_P1(data):
    P1_client = read_config().P1
    client(P1_client[0], P1_client[1], dump_obj(data))


def send_to_Carlo(data):
    Carlo_client = read_config().Carlo
    client(Carlo_client[0], Carlo_client[1], dump_obj(data))


def my_print(data):
    cur_thread = threading.current_thread()
    cur_str = "{}: {}".format(cur_thread.name, data)
    # print (cur_str)


def cur_time_str():
    return time.asctime(time.localtime())


def to_binary_int(v):
    if v <= 0.5:
        return 0
    elif v > 0.5:
        return 1
    else:
        return 2


def compute_accuracy(Xtest, Ytest, w):
    # print (Ytest)
    # print (Y_targets.shape, Y_pred.shape)
    # Yb_pred = to_binary_int(np.array(Y_pred))
    Y_pred = predict(Xtest, w)
    dim = Ytest.shape[0]
    cnt = 0
    for i in range(dim):
        if Ytest[i][0] == to_binary_int(Y_pred[i][0]):
            cnt = cnt + 1
    return 1.0 * cnt / dim


def sigmoid(x):
    # return 0.5+0.5*approx_rate*x
    return 1.0 / (1.0 + np.exp(-x))


def predict(X, w):
    return sigmoid(np.dot(X, w))


def write_to_file(file_name, arr):
    np.savetxt(file_name, arr, delimiter=',')


def start_all_threads(thread_lst):
    for thr in thread_lst:
        thr.start()


def init_logging(log_file_name):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    # DATE_FORMAT = "%Y%m%d %H:%M:%S %p"
    logging.basicConfig(filename=log_file_name, filemode='a', \
                        level=logging.INFO, format=LOG_FORMAT)
