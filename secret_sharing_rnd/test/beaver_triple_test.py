import unittest

import numpy as np

from secret_sharing_rnd.carlo import create_beaver_triples
from secret_sharing_rnd.util import assert_matrix


class TestCarloBeaverTriples(unittest.TestCase):

    def test_carlo_beaver_triple_for_matmul(self):

        iters = 1

        op_id_1 = "logit"
        op_id_2 = "grad"

        ops = [op_id_1, op_id_2]

        mul_ops = dict()
        mul_ops[op_id_1] = dict()
        mul_ops[op_id_1]["mul_type"] = "matmul"
        mul_ops[op_id_1]["left_0"] = 2
        mul_ops[op_id_1]["left_1"] = 3
        mul_ops[op_id_1]["right_0"] = 3
        mul_ops[op_id_1]["right_1"] = 1

        mul_ops[op_id_2] = dict()
        mul_ops[op_id_2]["mul_type"] = "matmul"
        mul_ops[op_id_2]["left_0"] = 3
        mul_ops[op_id_2]["left_1"] = 2
        mul_ops[op_id_2]["right_0"] = 2
        mul_ops[op_id_2]["right_1"] = 3

        party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=iters, num_batch=10)

        # print("party_a_bt_map \n")
        # print(party_a_bt_map)
        # print("party_b_bt_map \n")
        # print(party_b_bt_map)

        for op in ops:
            A0 = party_a_bt_map[0][op]["A0"]
            B0 = party_a_bt_map[0][op]["B0"]
            C0 = party_a_bt_map[0][op]["C0"]

            A1 = party_b_bt_map[0][op]["A1"]
            B1 = party_b_bt_map[0][op]["B1"]
            C1 = party_b_bt_map[0][op]["C1"]

            A = A0 + A1
            B = B0 + B1
            C = C0 + C1
            actual_C = np.dot(A, B)
            # print("C: \n", C)
            # print("AB: \n", actual_C)
            assert C.shape == actual_C.shape
            assert_matrix(C, actual_C)
        print("test passed !")

    def test_carlo_beaver_triple_for_multiply(self):

        iters = 1

        op_id_1 = "logit"
        op_id_2 = "grad"

        ops = [op_id_1, op_id_2]

        mul_ops = dict()
        mul_ops[op_id_1] = dict()
        mul_ops[op_id_1]["mul_type"] = "multiply"
        mul_ops[op_id_1]["left_0"] = 3
        mul_ops[op_id_1]["left_1"] = 2
        mul_ops[op_id_1]["right_0"] = 3
        mul_ops[op_id_1]["right_1"] = 2

        mul_ops[op_id_2] = dict()
        mul_ops[op_id_2]["mul_type"] = "multiply"
        mul_ops[op_id_2]["left_0"] = 4
        mul_ops[op_id_2]["left_1"] = 3
        mul_ops[op_id_2]["right_0"] = 4
        mul_ops[op_id_2]["right_1"] = 3

        party_a_bt_map, party_b_bt_map = create_beaver_triples(mul_ops, global_iters=iters, num_batch=10)

        # print("party_a_bt_map \n")
        # print(party_a_bt_map)
        # print("party_b_bt_map \n")
        # print(party_b_bt_map)

        for op in ops:
            A0 = party_a_bt_map[0][op]["A0"]
            B0 = party_a_bt_map[0][op]["B0"]
            C0 = party_a_bt_map[0][op]["C0"]

            A1 = party_b_bt_map[0][op]["A1"]
            B1 = party_b_bt_map[0][op]["B1"]
            C1 = party_b_bt_map[0][op]["C1"]

            A = A0 + A1
            B = B0 + B1
            C = C0 + C1
            actual_C = np.multiply(A, B)
            assert C.shape == actual_C.shape
            assert_matrix(C, actual_C)
        print("test passed !")


if __name__ == '__main__':
    unittest.main()
