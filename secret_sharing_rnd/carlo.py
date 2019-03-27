import numpy as np

from common_data import generate_random_matrix


def schema_copy(bt_map):
    _bt_map = [dict() for _ in range(len(bt_map))]
    for index, item in enumerate(bt_map):
        for key in item.keys():
            _bt_map[index][key] = dict()
    return _bt_map


def carlo_deal_data(party_a_bt_map, party_b_bt_map):
    assert len(party_a_bt_map) is len(party_b_bt_map)

    # TODO: make op_id scalable
    op_id = "logit"

    _party_a_bt_map = schema_copy(party_a_bt_map)
    _party_b_bt_map = schema_copy(party_b_bt_map)

    global_iters = len(party_a_bt_map)
    for i in range(global_iters):
        a_bt_i = party_a_bt_map[i]
        b_bt_i = party_b_bt_map[i]

        U0 = a_bt_i[op_id]["U0"]
        E0 = a_bt_i[op_id]["E0"]
        V1 = b_bt_i[op_id]["V1"]
        F1 = b_bt_i[op_id]["F1"]

        S = np.dot(U0, V1)
        K = np.dot(F1, E0)
        S0 = generate_random_matrix(S.shape[0], S.shape[1])
        K0 = generate_random_matrix(K.shape[0], K.shape[1])
        S1 = S - S0
        K1 = K - K0

        _party_a_bt_map[i][op_id]["S0"] = S0
        _party_b_bt_map[i][op_id]["S1"] = S1

        _party_a_bt_map[i][op_id]["K0"] = K0
        _party_b_bt_map[i][op_id]["K1"] = K1

    # sends _party_a_bt_map to party A and _party_b_bt_map to party B
    return _party_a_bt_map, _party_b_bt_map
