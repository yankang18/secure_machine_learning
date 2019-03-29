import unittest

import numpy as np

from secret_sharing_rnd.party_model import PartyA, PartyB, SimpleLogisticRegression
from secret_sharing_rnd.secret_sharing_operations import share
from secret_sharing_rnd.test.beaver_triple_test import create_beaver_triples
from secret_sharing_rnd.util import assert_matrix


class TestMulInLogisticRegression(unittest.TestCase):

    def test_matmul(self):

        X = np.array([[0.3, 0.4, 0.9],
                      [0.1, 0.5, 0.6],
                      [0.2, 0.7, 0.8],
                      [0.5, 0.3, 0.1],
                      [0.7, 0.5, 0.2]])

        y = np.array([[1], [-1], [1], [-1], [1]])

        w = np.array([[0.3], [0.2], [0.4]])

        print("X shape", X.shape)
        print("y shape", y.shape)
        print("w shape", w.shape)

        # X0, X1 = share(X)
        # w0, w1 = share(w)

        batch_size = 2
        # num_batch = X.shape[0] // batch_size + 1
        # residual = X.shape[0] % batch_size

        residual = X.shape[0] % batch_size
        if residual == 0:
            # if residual is 0, the number of samples is in multiples of batch_size.
            # Thus, we can directly use the real floor division operator "//" to compute
            # the num_batch
            num_batch = X.shape[0] // batch_size

            # if residual is 0, the last batch is a full batch.
            # In other words, all batches have the same number of samples
            last_batch_size = batch_size
        else:
            # if residual is not 0,
            num_batch = X.shape[0] // batch_size + 1

            # if residual is not 0, the last batch has 'residual' number of samples
            last_batch_size = residual

        op_id_1 = "logit"
        op_id_2 = "grad"

        mul_ops = dict()
        mul_ops[op_id_1] = dict()
        mul_ops[op_id_1]["mul_type"] = "matmul"
        mul_ops[op_id_1]["left_0"] = batch_size
        mul_ops[op_id_1]["left_1"] = X.shape[1]
        mul_ops[op_id_1]["right_0"] = X.shape[1]
        mul_ops[op_id_1]["right_1"] = w.shape[1]
        mul_ops[op_id_1]["last_left_0"] = last_batch_size
        mul_ops[op_id_1]["last_left_1"] = X.shape[1]
        mul_ops[op_id_1]["last_right_0"] = X.shape[1]
        mul_ops[op_id_1]["last_right_1"] = w.shape[1]

        mul_ops[op_id_2] = dict()
        mul_ops[op_id_2]["mul_type"] = "matmul"
        mul_ops[op_id_2]["left_0"] = X.shape[1]
        mul_ops[op_id_2]["left_1"] = batch_size
        mul_ops[op_id_2]["right_0"] = batch_size
        mul_ops[op_id_2]["right_1"] = w.shape[1]
        mul_ops[op_id_2]["last_left_0"] = X.shape[1]
        mul_ops[op_id_2]["last_left_1"] = last_batch_size
        mul_ops[op_id_2]["last_right_0"] = last_batch_size
        mul_ops[op_id_2]["last_right_1"] = w.shape[1]

        num_epoch = 2
        global_iters = num_batch * num_epoch

        print("num_batch:", num_batch)
        print("global_iters", global_iters)

        party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)

        print("party_a_bt_map:")
        print(party_a_bt_map)

        learning_rate = 0.01
        approx_rate = 0.28
        party_a = PartyA(approx_rate)
        party_a.set_bt_map(party_a_bt_map)
        party_a.build(learning_rate=learning_rate)

        print("party_b_bt_map:")
        print(party_b_bt_map)

        party_b = PartyB(approx_rate)
        party_b.set_bt_map(party_b_bt_map)
        party_b.build(learning_rate=learning_rate)

        slr = SimpleLogisticRegression(approx_rate)
        slr.build(learning_rate=learning_rate)

        slr.init_model(w)

        w0, w1 = share(w)
        party_a.init_model(w0)
        party_b.init_model(w1)

        print("-" * 20)
        for ep in range(num_epoch):
            for batch_i in range(num_batch):
                global_index = ep * num_batch + batch_i

                print("------ ep:", ep, ", batch: ", batch_i, ", global_index", global_index)

                X_batch = X[batch_i * batch_size: batch_i * batch_size + batch_size, :]
                y_batch = y[batch_i * batch_size: batch_i * batch_size + batch_size, :]

                print("X_batch shape", X_batch.shape)
                # print("X_batch: \n", X_batch)
                print("y_batch shape: \n", y_batch.shape)
                # print("y_batch: \n", y_batch)
                print("w shape", w.shape)

                slr.set_batch(X_batch, y_batch)

                # TODO: sends X1 to the other party
                X0, X1 = share(X_batch)
                y0, y1 = share(y_batch)

                party_a.set_batch(X0, y0)
                party_b.set_batch(X1, y1)

                # compute logit
                slr.compute_logit()
                alpha_0, beta_0 = party_a.compute_alpha_beta_share(global_index, op_id_1)
                alpha_1, beta_1 = party_b.compute_alpha_beta_share(global_index, op_id_1)

                Z0 = party_a.compute_matmul_share(alpha_1, beta_1)
                Z1 = party_b.compute_matmul_share(alpha_0, beta_0)

                print(">>> ###### Test logit ################")
                # print("Z0 \n", Z0)
                # print("Z1 \n", Z1)
                Z = Z0 + Z1

                actual_Xw = slr.get_logit()
                # print("Z: \n", Z)
                # print("Xw: \n", actual_Xw)
                assert Z.shape == actual_Xw.shape
                assert_matrix(Z, actual_Xw)
                print("test passed !")
                print("<<< ######################")

                slr.compute_prediction()
                slr.compute_loss()

                party_a.compute_prediction()
                party_a.compute_loss()

                party_b.compute_prediction()
                party_b.compute_loss()

                # compute grad
                slr.compute_grad()
                alpha_0, beta_0 = party_a.compute_alpha_beta_share(global_index, op_id_2)
                alpha_1, beta_1 = party_b.compute_alpha_beta_share(global_index, op_id_2)

                G0 = party_a.compute_matmul_share(alpha_1, beta_1)
                G1 = party_b.compute_matmul_share(alpha_0, beta_0)

                print(">>> ###### Test grad ################")
                # print("G0 \n", G0)
                # print("G1 \n", G1)
                G = G0 + G1

                actual_grad = slr.get_grad()
                # print("G: \n", G)
                # print("grad: \n", actual_grad)
                assert G.shape == actual_grad.shape
                assert_matrix(G, actual_grad)
                print("test passed !")
                print("<<< ######################")

                # update model
                slr.update_weights()
                party_a.update_weights()
                party_b.update_weights()

                print(">>> ###### Test weight ################")
                W0 = party_a.get_weights()
                W1 = party_b.get_weights()
                W = W0 + W1

                actual_w = slr.get_weights()
                # print("W: \n", W)
                # print("weights: \n", actual_w)
                assert W.shape == actual_w.shape
                assert_matrix(W, actual_w)
                print("test passed !")
                print("<<< ######################")


if __name__ == '__main__':
    unittest.main()
