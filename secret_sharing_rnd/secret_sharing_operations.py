import numpy as np

from common_data import generate_random_matrix


def share(X):
    n, d = X.shape
    X0 = generate_random_matrix(n, d)
    X1 = X - X0
    return X0, X1


def reconstruct(share0, share1):
    return share0 + share1


def local_compute_alpha_beta_share(share_map):
    As = share_map["As"]
    Bs = share_map["Bs"]
    Xs = share_map["Xs"]
    Ys = share_map["Ys"]

    alpha_s = Xs - As
    beta_s = Ys - Bs

    return alpha_s, beta_s


def compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, share_map):
    alpha = reconstruct(alpha_0, alpha_1)
    beta = reconstruct(beta_0, beta_1)

    As = share_map["As"]
    Bs = share_map["Bs"]
    Cs = share_map["Cs"]
    i = 0 if share_map["is_party_a"] else 1

    # print("is_party_a", share_map["is_party_a"])
    # print("alpha", alpha)
    # print("beta", beta)
    # print("As", As)
    # print("Bs", Bs)
    # print("Cs", Cs)

    # print("is_party_a", share_map["is_party_a"])
    # print("alpha.shape", alpha.shape)
    # print("beta.shape", beta.shape)
    # print("As.shape", As.shape)
    # print("Bs.shape", Bs.shape)
    # print("Cs.shape", Cs.shape)

    Zs = i * np.dot(alpha, beta) + np.dot(As, beta) + np.dot(alpha, Bs) + Cs
    return Zs


def matmul(a_share_map, b_share_map):
    alpha_0, beta_0 = local_compute_alpha_beta_share(a_share_map)
    alpha_1, beta_1 = local_compute_alpha_beta_share(b_share_map)

    print("alpha_0", alpha_0)
    print("beta_0", beta_0)
    print("alpha_1", alpha_1)
    print("beta_1", beta_1)

    Z0 = compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, a_share_map)
    Z1 = compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, b_share_map)

    return Z0, Z1
