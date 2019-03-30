import numpy as np

from secret_sharing_rnd.util import generate_random_matrix


def get_matrix_dims(val, global_iter_index, num_batch):
    if (global_iter_index + 1) % num_batch == 0:
        # if last batch
        k = val["last_left_0"]
        l = val["last_left_1"]
        p = val["last_right_0"]
        q = val["last_right_1"]
    else:
        k = val["left_0"]
        l = val["left_1"]
        p = val["right_0"]
        q = val["right_1"]

    return k, l, p, q


def create_beaver_triples(mul_ops, global_iters, num_batch):
    """

    :param mul_ops:
    :param global_iters:  start from 0
    :return:
    """

    party_a_bt_map = [dict() for _ in range(global_iters)]
    party_a_bt_map_to_carlo = [dict() for _ in range(global_iters)]
    party_a_bt_map_to_b = [dict() for _ in range(global_iters)]

    party_b_bt_map = [dict() for _ in range(global_iters)]
    party_b_bt_map_to_carlo = [dict() for _ in range(global_iters)]
    party_b_bt_map_to_a = [dict() for _ in range(global_iters)]

    for i in range(global_iters):

        for op_id, val in mul_ops.items():
            k, l, p, q = get_matrix_dims(val=val, global_iter_index=i, num_batch=num_batch)

            print("k, p, l, q", k, l, p, q)

            # party A generate data
            A0 = generate_random_matrix(k, l)
            A00 = generate_random_matrix(k, l)
            A01 = A0 - A00

            B0 = generate_random_matrix(p, q)
            B00 = generate_random_matrix(p, q)
            B01 = B0 - B00

            party_a_bt_map[i][op_id] = dict()
            party_a_bt_map[i][op_id]["A0"] = A0
            party_a_bt_map[i][op_id]["B0"] = B0
            party_a_bt_map[i][op_id]["U0"] = A00
            party_a_bt_map[i][op_id]["U1"] = A01
            party_a_bt_map[i][op_id]["E0"] = B00
            party_a_bt_map[i][op_id]["E1"] = B01

            # party B generate data
            A1 = generate_random_matrix(k, l)
            A11 = generate_random_matrix(k, l)
            A10 = A1 - A11

            B1 = generate_random_matrix(p, q)
            B11 = generate_random_matrix(p, q)
            B10 = B1 - B11

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

    _party_a_bt_map, _party_b_bt_map = carlo_deal_data(party_a_bt_map_to_carlo, party_b_bt_map_to_carlo, mul_ops)

    # print(_party_a_bt_map)
    # print(_party_b_bt_map)

    for i in range(len(_party_a_bt_map)):

        for op_id in mul_ops.keys():

            A0 = party_a_bt_map[i][op_id]["A0"]
            B0 = party_a_bt_map[i][op_id]["B0"]

            A1 = party_b_bt_map[i][op_id]["A1"]
            B1 = party_b_bt_map[i][op_id]["B1"]

            # P0 compute W0, Z0 and C0
            U0 = party_a_bt_map[i][op_id]["U0"]
            U1 = party_a_bt_map[i][op_id]["U1"]
            V0 = party_b_bt_map_to_a[i][op_id]["V0"]
            S0 = _party_a_bt_map[i][op_id]["S0"]

            E0 = party_a_bt_map[i][op_id]["E0"]
            E1 = party_a_bt_map[i][op_id]["E1"]
            F0 = party_b_bt_map_to_a[i][op_id]["F0"]
            K0 = _party_a_bt_map[i][op_id]["K0"]

            mul_type = mul_ops[op_id]["mul_type"]
            if mul_type == "matmul":
                W0 = np.dot(U0, V0) + np.dot(U1, V0) + S0
                Z0 = np.dot(F0, E0) + np.dot(F0, E1) + K0
                C0 = np.dot(A0, B0) + W0 + Z0
            elif mul_type == "multiply":
                W0 = np.multiply(U0, V0) + np.multiply(U1, V0) + S0
                Z0 = np.multiply(F0, E0) + np.multiply(F0, E1) + K0
                C0 = np.multiply(A0, B0) + W0 + Z0
            else:
                raise TypeError("does not support" + mul_type)

            party_a_bt_map[i][op_id]["C0"] = C0

            # P1 compute W1, Z1 and C1
            U1 = party_a_bt_map_to_b[i][op_id]["U1"]
            V1 = party_b_bt_map[i][op_id]["V1"]
            S1 = _party_b_bt_map[i][op_id]["S1"]

            E1 = party_a_bt_map_to_b[i][op_id]["E1"]
            F1 = party_b_bt_map[i][op_id]["F1"]
            K1 = _party_b_bt_map[i][op_id]["K1"]

            mul_type = mul_ops[op_id]["mul_type"]
            if mul_type == "matmul":
                W1 = np.dot(U1, V1) + S1
                Z1 = np.dot(F1, E1) + K1
                C1 = np.dot(A1, B1) + W1 + Z1
            elif mul_type == "multiply":
                W1 = np.multiply(U1, V1) + S1
                Z1 = np.multiply(F1, E1) + K1
                C1 = np.multiply(A1, B1) + W1 + Z1
            else:
                raise TypeError("does not support" + mul_type)

            party_b_bt_map[i][op_id]["C1"] = C1

    return party_a_bt_map, party_b_bt_map


def schema_copy(bt_map):
    _bt_map = [dict() for _ in range(len(bt_map))]
    for index, item in enumerate(bt_map):
        for key in item.keys():
            _bt_map[index][key] = dict()
    return _bt_map


def carlo_deal_data(party_a_bt_map, party_b_bt_map, mul_ops):
    # print("len(party_a_bt_map) len(party_b_bt_map)", len(party_a_bt_map), len(party_b_bt_map))
    assert len(party_a_bt_map) == len(party_b_bt_map)

    _party_a_bt_map = schema_copy(party_a_bt_map)
    _party_b_bt_map = schema_copy(party_b_bt_map)

    global_iters = len(party_a_bt_map)
    for i in range(global_iters):
        a_bt_i = party_a_bt_map[i]
        b_bt_i = party_b_bt_map[i]

        for op_id in mul_ops.keys():

            U0 = a_bt_i[op_id]["U0"]
            E0 = a_bt_i[op_id]["E0"]
            V1 = b_bt_i[op_id]["V1"]
            F1 = b_bt_i[op_id]["F1"]

            mul_type = mul_ops[op_id]["mul_type"]
            if mul_type == "matmul":
                S = np.dot(U0, V1)
                K = np.dot(F1, E0)
            elif mul_type == "multiply":
                S = np.multiply(U0, V1)
                K = np.multiply(F1, E0)
            else:
                raise TypeError("does not support" + mul_type)

            S0 = generate_random_matrix(S.shape[0], S.shape[1])
            K0 = generate_random_matrix(K.shape[0], K.shape[1])
            S1 = S - S0
            K1 = K - K0

            _party_a_bt_map[i][op_id]["S0"] = S0
            _party_b_bt_map[i][op_id]["S1"] = S1

            _party_a_bt_map[i][op_id]["K0"] = K0
            _party_b_bt_map[i][op_id]["K1"] = K1

    # TODO: sends _party_a_bt_map to party A and _party_b_bt_map to party B
    return _party_a_bt_map, _party_b_bt_map
