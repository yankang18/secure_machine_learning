from secret_sharing_rnd.secret_sharing_operations import local_compute_alpha_beta_share, compute_matmul_share
import numpy as np


class PartyA(object):

    def __init__(self, approx_rate=0.01):
        self.approx_rate = approx_rate
        self.party_a_bt_map = None
        self.a_share_map = None

    def build(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def set_bt_map(self, party_a_bt_map):
        self.party_a_bt_map = party_a_bt_map

    def init_model(self, w0):
        self.w0 = np.copy(w0)

    def set_batch(self, X0, y0):
        self.X0 = X0
        self.y0 = y0
        self.batch_size = self.X0.shape[0]

    def compute_alpha_beta_share(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A0 = self.party_a_bt_map[global_index][op_id]["A0"]
        B0 = self.party_a_bt_map[global_index][op_id]["B0"]
        C0 = self.party_a_bt_map[global_index][op_id]["C0"]

        print("A0, B0, C0", A0.shape, B0.shape, C0.shape)

        self.a_share_map = dict()
        self.a_share_map["is_party_a"] = True
        self.a_share_map["As"] = A0
        self.a_share_map["Bs"] = B0
        self.a_share_map["Cs"] = C0

        # TODO: different operation come with different X and Y
        if op_id == "logit":
            self.a_share_map["Xs"] = self.X0
            self.a_share_map["Ys"] = self.w0
        elif op_id == "grad":
            self.a_share_map["Xs"] = self.X0.transpose()
            self.a_share_map["Ys"] = self.loss
        else:
            ValueError("Does not support", op_id)

        self.alpha_0, self.beta_0 = local_compute_alpha_beta_share(self.a_share_map)
        return self.alpha_0, self.beta_0

    def compute_matmul_share(self, alpha_1, beta_1):
        self.Z0 = compute_matmul_share(self.alpha_0, alpha_1, self.beta_0, beta_1, self.a_share_map)
        return self.Z0

    def compute_prediction(self):
        i = 0 if self.a_share_map["is_party_a"] else 1
        self.prediction0 = 0.5 * i + 0.5 * self.approx_rate * self.Z0
        return self.prediction0

    def compute_loss(self):
        self.loss = self.prediction0 - self.y0

    def update_weights(self):
        self.w0 = self.w0 - self.learning_rate / self.batch_size * self.Z0

    def get_weights(self):
        return self.w0.copy()

    def get_loss(self):
        return np.mean(np.square(self.loss))


class PartyB(object):

    def __init__(self, approx_rate=0.01):
        self.approx_rate = approx_rate
        self.party_b_bt_map = None
        self.b_share_map = None

    def build(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def set_bt_map(self, party_b_bt_map):
        self.party_b_bt_map = party_b_bt_map

    def init_model(self, w1):
        self.w1 = np.copy(w1)

    def set_batch(self, X1, y1):
        self.X1 = X1
        self.y1 = y1
        self.batch_size = self.X1.shape[0]

    def compute_alpha_beta_share(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A1 = self.party_b_bt_map[global_index][op_id]["A1"]
        B1 = self.party_b_bt_map[global_index][op_id]["B1"]
        C1 = self.party_b_bt_map[global_index][op_id]["C1"]

        self.b_share_map = dict()
        self.b_share_map["is_party_a"] = False
        self.b_share_map["As"] = A1
        self.b_share_map["Bs"] = B1
        self.b_share_map["Cs"] = C1
        # self.b_share_map["Xs"] = self.X1
        # self.b_share_map["Ys"] = self.w1

        # TODO: different operation come with different X and Y
        if op_id == "logit":
            self.b_share_map["Xs"] = self.X1
            self.b_share_map["Ys"] = self.w1
        elif op_id == "grad":
            self.b_share_map["Xs"] = self.X1.transpose()
            self.b_share_map["Ys"] = self.loss
        else:
            ValueError("Does not support", op_id)

        self.alpha_1, self.beta_1 = local_compute_alpha_beta_share(self.b_share_map)
        return self.alpha_1, self.beta_1

    def compute_matmul_share(self, alpha_0, beta_0):
        self.Z1 = compute_matmul_share(alpha_0, self.alpha_1, beta_0, self.beta_1, self.b_share_map)
        return self.Z1

    def compute_prediction(self):
        i = 0 if self.b_share_map["is_party_a"] else 1
        self.prediction1 = 0.5 * i + 0.5 * self.approx_rate * self.Z1
        return self.prediction1

    def compute_loss(self):
        self.loss = self.prediction1 - self.y1

    def update_weights(self):
        self.w1 = self.w1 - self.learning_rate / self.batch_size * self.Z1

    def get_weights(self):
        return self.w1.copy()

    def get_loss(self):
        return np.mean(np.square(self.loss))


class SimpleLogisticRegression(object):

    def __init__(self, approx_rate=0.01):
        self.approx_rate = approx_rate
        self.party_b_bt_map = None
        self.b_share_map = None

    def build(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def set_bt_map(self, party_b_bt_map):
        self.party_b_bt_map = party_b_bt_map

    def init_model(self, w):
        self.w = np.copy(w)

    def set_batch(self, X, y):
        self.X = X
        self.y = y
        self.batch_size = self.X.shape[0]

    def compute_logit(self):
        self.Xw = np.dot(self.X, self.w)

    def compute_prediction(self):
        self.prediction = 0.5 + 0.5 * self.approx_rate * self.Xw
        return self.prediction

    def compute_loss(self):
        self.loss = self.prediction - self.y

    def compute_grad(self):
        self.grad = np.dot(self.X.transpose(), self.loss)

    def update_weights(self):
        self.w = self.w - self.learning_rate / self.batch_size * self.grad

    def get_weights(self):
        return self.w.copy()

    def get_logit(self):
        return self.Xw.copy()

    def get_grad(self):
        return self.grad.copy()

    def get_loss(self):
        return np.mean(np.square(self.loss))
        # return np.mean(self.loss)
