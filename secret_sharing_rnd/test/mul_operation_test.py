import unittest

import numpy as np

from secret_sharing_rnd.beaver_triple import fill_beaver_triple_shape, create_beaver_triples
from secret_sharing_rnd.secret_sharing_operations import share, local_compute_alpha_beta_share, compute_matmul_share, \
    compute_multiply_share, compute_sum_of_matmul_share, compute_sum_of_multiply_share
from secret_sharing_rnd.util import assert_matrix, assert_array


def matmul(a_share_map, b_share_map):
    alpha_0, beta_0 = local_compute_alpha_beta_share(a_share_map)
    alpha_1, beta_1 = local_compute_alpha_beta_share(b_share_map)

    # print("alpha_0", alpha_0)
    # print("beta_0", beta_0)
    # print("alpha_1", alpha_1)
    # print("beta_1", beta_1)

    Z0 = compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, a_share_map)
    Z1 = compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, b_share_map)

    return Z0, Z1


def sum_matmul(a_share_map, b_share_map, axis=None):
    alpha_0, beta_0 = local_compute_alpha_beta_share(a_share_map)
    alpha_1, beta_1 = local_compute_alpha_beta_share(b_share_map)

    Z0 = compute_sum_of_matmul_share(alpha_0, alpha_1, beta_0, beta_1, a_share_map, axis)
    Z1 = compute_sum_of_matmul_share(alpha_0, alpha_1, beta_0, beta_1, b_share_map, axis)

    return Z0, Z1


def multiply(a_share_map, b_share_map):
    alpha_0, beta_0 = local_compute_alpha_beta_share(a_share_map)
    alpha_1, beta_1 = local_compute_alpha_beta_share(b_share_map)

    # print("alpha_0", alpha_0)
    # print("beta_0", beta_0)
    # print("alpha_1", alpha_1)
    # print("beta_1", beta_1)

    Z0 = compute_multiply_share(alpha_0, alpha_1, beta_0, beta_1, a_share_map)
    Z1 = compute_multiply_share(alpha_0, alpha_1, beta_0, beta_1, b_share_map)

    return Z0, Z1


def sum_multiply(a_share_map, b_share_map, axis=None):
    alpha_0, beta_0 = local_compute_alpha_beta_share(a_share_map)
    alpha_1, beta_1 = local_compute_alpha_beta_share(b_share_map)

    Z0 = compute_sum_of_multiply_share(alpha_0, alpha_1, beta_0, beta_1, a_share_map, axis)
    Z1 = compute_sum_of_multiply_share(alpha_0, alpha_1, beta_0, beta_1, b_share_map, axis)

    return Z0, Z1


def create_share_map_for_party_a(*, party_bt_map, global_index, op_id, X0, Y0):
    A0 = party_bt_map[global_index][op_id]["A0"]
    B0 = party_bt_map[global_index][op_id]["B0"]
    C0 = party_bt_map[global_index][op_id]["C0"]

    a_share_map = dict()
    a_share_map["As"] = A0
    a_share_map["Bs"] = B0
    a_share_map["Cs"] = C0
    a_share_map["Xs"] = X0
    a_share_map["Ys"] = Y0
    a_share_map["is_party_a"] = True
    return a_share_map


def create_share_map_for_party_b(*, party_bt_map, global_index, op_id, X1, Y1):
    A1 = party_bt_map[global_index][op_id]["A1"]
    B1 = party_bt_map[global_index][op_id]["B1"]
    C1 = party_bt_map[global_index][op_id]["C1"]

    b_share_map = dict()
    b_share_map["As"] = A1
    b_share_map["Bs"] = B1
    b_share_map["Cs"] = C1
    b_share_map["Xs"] = X1
    b_share_map["Ys"] = Y1
    b_share_map["is_party_a"] = False
    return b_share_map


class TestCarloBeaverTriplesBasedMul(unittest.TestCase):

    def test_matmul(self):
        X = np.array([[0.3, 0.4, 0.9],
                      [0.1, 0.5, 0.6],
                      [0.2, 0.7, 0.8],
                      [0.5, 0.3, 0.1],
                      [0.7, 0.5, 0.2]])

        w = np.array([[0.3], [0.2], [0.4]])

        print("X shape", X.shape)
        print("w shape", w.shape)

        op_id = "logit"
        mul_ops = dict()
        batch_size = 2
        num_epoch = 3

        num_batch = fill_beaver_triple_shape(mul_ops,
                                             X_shape=X.shape,
                                             Y_shape=w.shape,
                                             batch_size=batch_size,
                                             op_id=op_id,
                                             mul_type="matmul")
        global_iters = num_batch * num_epoch
        party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)

        print("-" * 20)
        for ep in range(num_epoch):
            for batch_i in range(num_batch):
                print("---- ep:", ep, ", batch: ", batch_i)

                global_index = ep * num_epoch + batch_i

                X_batch = X[batch_i * batch_size: batch_i * batch_size + batch_size, :]

                print("X_batch shape", X_batch.shape)
                # print("X_batch: \n", X_batch)
                print("w shape", w.shape)

                X0, X1 = share(X_batch)
                w0, w1 = share(w)

                a_share_map = create_share_map_for_party_a(party_bt_map=party_a_bt_map, global_index=global_index,
                                                           op_id=op_id, X0=X0,
                                                           Y0=w0)
                b_share_map = create_share_map_for_party_b(party_bt_map=party_b_bt_map, global_index=global_index,
                                                           op_id=op_id, X1=X1,
                                                           Y1=w1)

                Z0, Z1 = matmul(a_share_map, b_share_map)

                # print("Z0 \n", Z0)
                # print("Z1 \n", Z1)
                Z = Z0 + Z1

                actual_Z = np.dot(X_batch, w)
                # print("Z: \n", Z)
                # print("Xw: \n", actual_Z)
                assert Z.shape == actual_Z.shape
                assert_matrix(Z, actual_Z)
                print("test passed !")

    def test_sum_matmul(self):
        X = np.array([[0.3, 0.4, 0.9],
                      [0.1, 0.5, 0.6],
                      [0.2, 0.7, 0.8],
                      [0.5, 0.3, 0.1],
                      [0.7, 0.5, 0.2]])

        w = np.array([[0.3], [0.2], [0.4]])

        print("X shape", X.shape)
        print("w shape", w.shape)

        op_id = "logit"
        mul_ops = dict()
        batch_size = 2
        num_epoch = 3
        num_batch = fill_beaver_triple_shape(mul_ops,
                                             X_shape=X.shape,
                                             Y_shape=w.shape,
                                             batch_size=batch_size,
                                             op_id=op_id,
                                             mul_type="matmul")
        global_iters = num_batch * num_epoch
        party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)

        print("-" * 20)
        for ep in range(num_epoch):
            for batch_i in range(num_batch):
                print("---- ep:", ep, ", batch: ", batch_i)

                global_index = ep * num_epoch + batch_i

                X_batch = X[batch_i * batch_size: batch_i * batch_size + batch_size, :]

                print("X_batch shape", X_batch.shape)
                # print("X_batch: \n", X_batch)
                print("w shape", w.shape)

                X0, X1 = share(X_batch)
                w0, w1 = share(w)

                a_share_map = create_share_map_for_party_a(party_bt_map=party_a_bt_map, global_index=global_index,
                                                           op_id=op_id, X0=X0,
                                                           Y0=w0)
                b_share_map = create_share_map_for_party_b(party_bt_map=party_b_bt_map, global_index=global_index,
                                                           op_id=op_id, X1=X1,
                                                           Y1=w1)

                Z0, Z1 = sum_matmul(a_share_map, b_share_map, axis=0)

                # print("Z0 \n", Z0)
                # print("Z1 \n", Z1)
                Z = Z0 + Z1

                actual_Z = np.dot(X_batch, w).sum(axis=0)
                print("Z: \n", Z, Z.shape)
                print("Xw: \n", actual_Z, actual_Z.shape)
                assert Z.shape == actual_Z.shape
                assert_array(Z, actual_Z)
                print("test passed !")

    def test_multiply(self):
        X = np.array([[0.3, 0.4, 0.9],
                      [0.1, 0.5, 0.6],
                      [0.2, 0.7, 0.8],
                      [0.5, 0.3, 0.1],
                      [0.7, 0.5, 0.2]])

        Y = np.array([[0.4, 0.1, 0.9],
                      [0.2, 0.6, 0.2],
                      [0.3, 0.8, 0.3],
                      [0.6, 0.9, 0.4],
                      [0.2, 0.5, 0.2]])

        op_id = "logit"
        mul_ops = dict()
        batch_size = 2
        num_epoch = 3
        num_batch = fill_beaver_triple_shape(mul_ops,
                                             X_shape=X.shape,
                                             Y_shape=Y.shape,
                                             batch_size=batch_size,
                                             op_id=op_id,
                                             mul_type="multiply")
        global_iters = num_batch * num_epoch
        party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)

        print("-" * 20)
        for ep in range(num_epoch):
            for batch_i in range(num_batch):
                print("---- ep:", ep, ", batch: ", batch_i)

                global_index = ep * num_epoch + batch_i
                X_batch = X[batch_i * batch_size: batch_i * batch_size + batch_size, :]
                Y_batch = X[batch_i * batch_size: batch_i * batch_size + batch_size, :]

                print("X_batch shape", X_batch.shape)
                # print("X_batch: \n", X_batch)
                print("Y_batch shape", Y_batch.shape)

                X0, X1 = share(X_batch)
                Y0, Y1 = share(Y_batch)

                a_share_map = create_share_map_for_party_a(party_bt_map=party_a_bt_map, global_index=global_index,
                                                           op_id=op_id, X0=X0,
                                                           Y0=Y0)
                b_share_map = create_share_map_for_party_b(party_bt_map=party_b_bt_map, global_index=global_index,
                                                           op_id=op_id, X1=X1,
                                                           Y1=Y1)

                Z0, Z1 = multiply(a_share_map, b_share_map)

                # print("Z0 \n", Z0)
                # print("Z1 \n", Z1)
                Z = Z0 + Z1

                actual_Z = np.multiply(X_batch, Y_batch)
                # print("Z: \n", Z)
                # print("Xw: \n", actual_Z)
                print(Z.shape, actual_Z.shape)
                assert Z.shape == actual_Z.shape
                assert_matrix(Z, actual_Z)
                print("test passed !")

    def test_sum_multiply(self):
        X = np.array([[0.3, 0.4, 0.9],
                      [0.1, 0.5, 0.6],
                      [0.2, 0.7, 0.8],
                      [0.5, 0.3, 0.1],
                      [0.7, 0.5, 0.2]])

        Y = np.array([[0.4, 0.1, 0.9],
                      [0.2, 0.6, 0.2],
                      [0.3, 0.8, 0.3],
                      [0.6, 0.9, 0.4],
                      [0.2, 0.5, 0.2]])

        op_id = "logit"
        mul_ops = dict()
        batch_size = 2
        num_epoch = 3
        num_batch = fill_beaver_triple_shape(mul_ops,
                                             X_shape=X.shape,
                                             Y_shape=Y.shape,
                                             batch_size=batch_size,
                                             op_id=op_id,
                                             mul_type="multiply")
        global_iters = num_batch * num_epoch
        party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)

        print("-" * 20)
        for ep in range(num_epoch):
            for batch_i in range(num_batch):
                print("---- ep:", ep, ", batch: ", batch_i)

                global_index = ep * num_epoch + batch_i
                X_batch = X[batch_i * batch_size: batch_i * batch_size + batch_size, :]
                Y_batch = X[batch_i * batch_size: batch_i * batch_size + batch_size, :]

                print("X_batch shape", X_batch.shape)
                # print("X_batch: \n", X_batch)
                print("Y_batch shape", Y_batch.shape)

                X0, X1 = share(X_batch)
                Y0, Y1 = share(Y_batch)

                a_share_map = create_share_map_for_party_a(party_bt_map=party_a_bt_map, global_index=global_index,
                                                           op_id=op_id, X0=X0,
                                                           Y0=Y0)
                b_share_map = create_share_map_for_party_b(party_bt_map=party_b_bt_map, global_index=global_index,
                                                           op_id=op_id, X1=X1,
                                                           Y1=Y1)

                Z0, Z1 = sum_multiply(a_share_map, b_share_map, axis=0)

                # print("Z0 \n", Z0)
                # print("Z1 \n", Z1)
                Z = Z0 + Z1

                actual_Z = np.multiply(X_batch, Y_batch).sum(axis=0)
                print("Z: \n", Z, Z.shape)
                print("Xw: \n", actual_Z, actual_Z.shape)
                print(Z.shape, actual_Z.shape)
                assert Z.shape == actual_Z.shape
                assert_array(Z, actual_Z)
                print("test passed !")


if __name__ == '__main__':
    unittest.main()
