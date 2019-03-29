import numpy as np
from secret_sharing_rnd.secret_sharing_operations import share, matmul
from secret_sharing_rnd.carlo_test import create_beaver_triples
from secret_sharing_rnd.party_model import PartyA, PartyB
from secret_sharing_rnd.util import assert_matrix


if __name__ == '__main__':

    X = np.array([[0.3, 0.4, 0.9],
                  [0.1, 0.5, 0.6],
                  [0.2, 0.7, 0.8],
                  [0.5, 0.3, 0.1],
                  [0.7, 0.5, 0.2]])

    # X = np.array([[0.3, 0.4, 0.9],
    #               [0.1, 0.5, 0.6],
    #               [0.2, 0.7, 0.8],
    #               [0.5, 0.3, 0.1],
    #               [0.5, 0.6, 0.2],
    #               [0.7, 0.5, 0.2]])

    w = np.array([[0.3], [0.2], [0.4]])

    print("X shape", X.shape)
    print("w shape", w.shape)

    # X0, X1 = share(X)
    # w0, w1 = share(w)

    batch_size = 2
    residual = X.shape[0] % batch_size
    if residual == 0:
        num_batch = X.shape[0] // batch_size

        # if residual is 0, the last batch is a full batch. In other words, all batches have the same number of samples
        last_batch_size = batch_size
    else:
        num_batch = X.shape[0] // batch_size + 1

        # if residual is not 0, the last batch has 'residual' number of samples
        last_batch_size = residual

    print("# samples:", X.shape[0])
    print("batch_size:", batch_size)
    print("num_batch:", num_batch)
    print("residual:", residual)

    op_id = "logit"
    mul_ops = dict()
    mul_ops[op_id] = dict()
    mul_ops[op_id]["last_left"] = last_batch_size
    mul_ops[op_id]["left"] = batch_size
    mul_ops[op_id]["last_middle"] = X.shape[1]
    mul_ops[op_id]["middle"] = X.shape[1]
    mul_ops[op_id]["right"] = w.shape[1]

    num_epoch = 3
    global_iters = num_batch * num_epoch

    print("global_iters", global_iters)

    #
    # Start creating beaver triples
    #

    party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=global_iters, num_batch=num_batch)

    print("party_a_bt_map:")
    print(party_a_bt_map)

    party_a = PartyA()
    party_a.set_bt_map(party_a_bt_map)

    print("party_b_bt_map:")
    print(party_b_bt_map)

    party_b = PartyB()
    party_b.set_bt_map(party_b_bt_map)

    w0, w1 = share(w)
    party_a.init_model(w0)
    party_b.init_model(w1)

    print("-"*20)
    for ep in range(num_epoch):
        for batch_i in range(num_batch):
            print("---- ep:", ep, ", batch: ", batch_i)

            global_index = ep * num_batch + batch_i

            X_batch = X[batch_i*batch_size: batch_i*batch_size + batch_size, :]

            print("X_batch shape", X_batch.shape)
            print("X_batch: \n", X_batch)
            print("w shape", w.shape)

            # TODO: sends X1 to the other party
            X0, X1 = share(X_batch)

            party_a.set_batch(X0, None)
            party_b.set_batch(X1, None)

            alpha_0, beta_0 = party_a.compute_alpha_beta_share(global_index, op_id)
            alpha_1, beta_1 = party_b.compute_alpha_beta_share(global_index, op_id)

            Z0 = party_a.compute_matmul_share(alpha_1, beta_1)
            Z1 = party_b.compute_matmul_share(alpha_0, beta_0)

            print("Z0 \n", Z0)
            print("Z1 \n", Z1)
            Z = Z0 + Z1

            actual_Z = np.dot(X_batch, w)
            print("Z: \n", Z)
            print("Xw: \n", actual_Z)
            assert Z.shape == actual_Z.shape
            assert_matrix(Z, actual_Z)
            print("test passed !")
