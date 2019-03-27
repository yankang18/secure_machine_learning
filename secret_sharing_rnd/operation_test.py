import numpy as np
from secret_sharing_rnd.secret_sharing_operations import share, mul
from secret_sharing_rnd.carlo_test import create_beaver_triples


if __name__ == '__main__':

    op_id = "logit"

    X = np.array([[0.3, 0.4, 0.7],
                  [0.1, 0.5, 0.6],
                  [0.2, 0.9, 0.8]])

    w = np.array([[0.3], [0.2], [0.4]])

    print("X shape", X.shape)
    print("w shape", w.shape)

    X0, X1 = share(X)
    w0, w1 = share(w)

    k = 3
    p = 3
    q = 1

    party_a_bt_map, party_b_bt_map = create_beaver_triples(k, p, q, op_id, iters=1)

    print("party_a_bt_map \n")
    print(party_a_bt_map)

    print("party_b_bt_map \n")
    print(party_b_bt_map)

    A0 = party_a_bt_map[0][op_id]["A0"]
    B0 = party_a_bt_map[0][op_id]["B0"]
    C0 = party_a_bt_map[0][op_id]["C0"]

    A1 = party_b_bt_map[0][op_id]["A1"]
    B1 = party_b_bt_map[0][op_id]["B1"]
    C1 = party_b_bt_map[0][op_id]["C1"]

    a_share_map = dict()
    a_share_map["As"] = A0
    a_share_map["Bs"] = B0
    a_share_map["Cs"] = C0
    a_share_map["Xs"] = X0
    a_share_map["Ys"] = w0
    a_share_map["is_party_a"] = True

    b_share_map = dict()
    b_share_map["As"] = A1
    b_share_map["Bs"] = B1
    b_share_map["Cs"] = C1
    b_share_map["Xs"] = X1
    b_share_map["Ys"] = w1
    b_share_map["is_party_a"] = False

    Z0, Z1 = mul(a_share_map, b_share_map)

    Z = Z0 + Z1

    print("Z: \n", Z)
    print("Xw: \n", np.dot(X, w))
