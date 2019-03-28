import numpy as np
from secret_sharing_rnd.secret_sharing_operations import share, mul
from secret_sharing_rnd.carlo_test import create_beaver_triples


if __name__ == '__main__':

    n = 101 / 32
    m = 101 // 32
    _m = 101 % 32

    k = 3 % 0
    print(n)
    print(m)
    print(_m)
    print(k)