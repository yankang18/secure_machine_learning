from secret_sharing_rnd.carlo import carlo_deal_data
from common_data import generate_random_matrix, generate_random_3_dim_matrix
import numpy as np


def create_beaver_triples(k, p, q, op_id, iters):

    party_a_bt_map = [dict() for _ in range(iters)]
    party_a_bt_map_to_carlo = [dict() for _ in range(iters)]
    party_a_bt_map_to_b = [dict() for _ in range(iters)]

    party_b_bt_map = [dict() for _ in range(iters)]
    party_b_bt_map_to_carlo = [dict() for _ in range(iters)]
    party_b_bt_map_to_a = [dict() for _ in range(iters)]

    for i in range(iters):
        # TODO: note here is creating beaver triple for just one multiplication operation

        # party A generate data
        A0 = generate_random_matrix(k, p)
        A00 = generate_random_matrix(k, p)
        A01 = A0 - A00

        B0 = generate_random_matrix(p, q)
        B00 = generate_random_matrix(p, q)
        B01 = B0 - B00

        # party B generate data
        A1 = generate_random_matrix(k, p)
        A11 = generate_random_matrix(k, p)
        A10 = A1 - A11

        B1 = generate_random_matrix(p, q)
        B11 = generate_random_matrix(p, q)
        B10 = B1 - B11

        # self
        party_a_bt_map[i][op_id] = dict()
        party_a_bt_map[i][op_id]["A0"] = A0
        party_a_bt_map[i][op_id]["B0"] = B0
        party_a_bt_map[i][op_id]["U0"] = A00
        party_a_bt_map[i][op_id]["U1"] = A01
        party_a_bt_map[i][op_id]["E0"] = B00
        party_a_bt_map[i][op_id]["E1"] = B01

        party_b_bt_map[i][op_id] = dict()
        party_b_bt_map[i][op_id]["A1"] = A1
        party_b_bt_map[i][op_id]["B1"] = B1
        party_b_bt_map[i][op_id]["V0"] = B10
        party_b_bt_map[i][op_id]["V1"] = B11
        party_b_bt_map[i][op_id]["F0"] = A10
        party_b_bt_map[i][op_id]["F1"] = A11

        # exchange
        party_a_bt_map_to_b[i][op_id] = dict()
        party_a_bt_map_to_b[i][op_id]["U1"] = A01
        party_a_bt_map_to_b[i][op_id]["E1"] = B01

        party_b_bt_map_to_a[i][op_id] = dict()
        party_b_bt_map_to_a[i][op_id]["F0"] = A10
        party_b_bt_map_to_a[i][op_id]["V0"] = B10

        # to carlo
        party_a_bt_map_to_carlo[i][op_id] = dict()
        party_a_bt_map_to_carlo[i][op_id]["U0"] = A00
        party_a_bt_map_to_carlo[i][op_id]["E0"] = B00

        party_b_bt_map_to_carlo[i][op_id] = dict()
        party_b_bt_map_to_carlo[i][op_id]["V1"] = B11
        party_b_bt_map_to_carlo[i][op_id]["F1"] = A11

    # print(party_a_bt_map_to_carlo)

    _party_a_bt_map, _party_b_bt_map = carlo_deal_data(party_a_bt_map_to_carlo, party_b_bt_map_to_carlo)

    # print(_party_a_bt_map)
    # print(_party_b_bt_map)

    for i in range(len(_party_a_bt_map)):
        A0 = party_a_bt_map[i][op_id]["A0"]
        B0 = party_a_bt_map[i][op_id]["B0"]

        A1 = party_b_bt_map[i][op_id]["A1"]
        B1 = party_b_bt_map[i][op_id]["B1"]

        A = A0 + A1
        B = B0 + B1

        # P0 compute W0, Z0 and C0
        U0 = party_a_bt_map[i][op_id]["U0"]
        U1 = party_a_bt_map[i][op_id]["U1"]
        V0 = party_b_bt_map_to_a[i][op_id]["V0"]
        S0 = _party_a_bt_map[i][op_id]["S0"]

        E0 = party_a_bt_map[i][op_id]["E0"]
        E1 = party_a_bt_map[i][op_id]["E1"]
        F0 = party_b_bt_map_to_a[i][op_id]["F0"]
        K0 = _party_a_bt_map[i][op_id]["K0"]

        W0 = np.dot(U0, V0) + np.dot(U1, V0) + S0
        Z0 = np.dot(F0, E0) + np.dot(F0, E1) + K0
        C0 = np.dot(A0, B0) + W0 + Z0

        party_a_bt_map[i][op_id]["C0"] = C0

        # P1 compute W1, Z1 and C1
        U1 = party_a_bt_map_to_b[i][op_id]["U1"]
        V1 = party_b_bt_map[i][op_id]["V1"]
        S1 = _party_b_bt_map[i][op_id]["S1"]

        E1 = party_a_bt_map_to_b[i][op_id]["E1"]
        F1 = party_b_bt_map[i][op_id]["F1"]
        K1 = _party_b_bt_map[i][op_id]["K1"]

        W1 = np.dot(U1, V1) + S1
        Z1 = np.dot(F1, E1) + K1
        C1 = np.dot(A1, B1) + W1 + Z1

        party_b_bt_map[i][op_id]["C1"] = C1

        # compute C (only for testing purpose)
        C = C0 + C1

        print("C: \n", C)
        print("AB: \n", np.dot(A, B))

    return party_a_bt_map, party_b_bt_map


if __name__ == '__main__':

    iters = 1

    op_id = "logit"

    k = 3
    p = 3
    q = 1

    party_a_bt_map, party_b_bt_map = create_beaver_triples(k, p, q, op_id, iters)

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

    A = A0 + A1
    B = B0 + B1
    C = C0 + C1

    print("C: \n", C)
    print("AB: \n", np.dot(A, B))






