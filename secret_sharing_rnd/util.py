def assert_matrix(M1, M2):
    assert M1.shape == M2.shape
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            # print(M1[i][j], M2[i][j])
            assert round(M1[i][j], 4) == round(M2[i][j], 4)
