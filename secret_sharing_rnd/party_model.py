from secret_sharing_rnd.secret_sharing_operations import local_compute_alpha_beta_share, compute_matmul_share
import numpy as np


class SecureSharingParty(object):

    def __init__(self, approx_rate=0.01):
        self.approx_rate = approx_rate
        self.beaver_triple_share_map = None

    def build(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def init_model(self, w_s):
        self.w_s = np.copy(w_s)

    def set_batch(self, X_s, y_s):
        self.X_s = X_s
        self.y_s = y_s
        self.batch_size = self.X_s.shape[0]

    def _prepare_beaver_triple(self, global_index, op_id):
        pass

    def _set_beaver_triple(self, As, Bs, Cs, is_party_a):
        print("As, Bs, Cs", As.shape, Bs.shape, Cs.shape)
        self.beaver_triple_share_map = dict()
        self.beaver_triple_share_map["is_party_a"] = is_party_a
        self.beaver_triple_share_map["As"] = As
        self.beaver_triple_share_map["Bs"] = Bs
        self.beaver_triple_share_map["Cs"] = Cs

    def compute_shares_for_alpha_beta_of_computing_logit(self, global_index, op_id=None):
        op_id = "logit"
        self._prepare_beaver_triple(global_index, op_id)
        self.beaver_triple_share_map["Xs"] = self.X_s
        self.beaver_triple_share_map["Ys"] = self.w_s
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_shares_for_alpha_beta_of_computing_grad(self, global_index, op_id=None):
        op_id = "grad"
        self._prepare_beaver_triple(global_index, op_id)
        self.beaver_triple_share_map["Xs"] = self.X_s.transpose()
        self.beaver_triple_share_map["Ys"] = self.loss_s
        self.alpha_s, self.beta_s = local_compute_alpha_beta_share(self.beaver_triple_share_map)
        return self.alpha_s, self.beta_s

    def compute_share_for_logit(self, alpha_t, beta_t):
        self.logit_s = compute_matmul_share(self.alpha_s, alpha_t, self.beta_s, beta_t, self.beaver_triple_share_map)
        return self.logit_s

    def compute_share_for_grad(self, alpha_t, beta_t):
        self.grad_s = compute_matmul_share(self.alpha_s, alpha_t, self.beta_s, beta_t, self.beaver_triple_share_map)
        return self.grad_s

    def compute_prediction(self):
        i = 0 if self.beaver_triple_share_map["is_party_a"] else 1
        self.prediction_s = 0.5 * i + 0.5 * self.approx_rate * self.logit_s
        return self.prediction_s

    def compute_loss(self):
        self.loss_s = self.prediction_s - self.y_s

    def update_weights(self):
        self.w_s = self.w_s - self.learning_rate / self.batch_size * self.grad_s

    def get_weights(self):
        return self.w_s.copy()

    def get_loss(self):
        return np.mean(np.square(self.loss_s))


class PartyA(SecureSharingParty):

    def __init__(self, approx_rate=0.01):
        super(PartyA, self).__init__(approx_rate)
        self.party_a_bt_map = None

    def set_bt_map(self, party_a_bt_map):
        self.party_a_bt_map = party_a_bt_map

    def _prepare_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A0 = self.party_a_bt_map[global_index][op_id]["A0"]
        B0 = self.party_a_bt_map[global_index][op_id]["B0"]
        C0 = self.party_a_bt_map[global_index][op_id]["C0"]
        self._set_beaver_triple(A0, B0, C0, True)


class PartyB(SecureSharingParty):

    def __init__(self, approx_rate=0.01):
        super(PartyB, self).__init__(approx_rate)
        self.party_b_bt_map = None

    def set_bt_map(self, party_b_bt_map):
        self.party_b_bt_map = party_b_bt_map

    def _prepare_beaver_triple(self, global_index, op_id):
        # get beaver triple for operation:op_id at iteration:global_index
        A1 = self.party_b_bt_map[global_index][op_id]["A1"]
        B1 = self.party_b_bt_map[global_index][op_id]["B1"]
        C1 = self.party_b_bt_map[global_index][op_id]["C1"]
        self._set_beaver_triple(A1, B1, C1, False)


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
