import numpy as np

from secret_sharing_rnd.util import generate_random_matrix


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


def compute_sum_of_matmul_share(alpha_0, alpha_1, beta_0, beta_1, share_map, axis=None):
    Zs = compute_matmul_share(alpha_0, alpha_1, beta_0, beta_1, share_map)
    return np.sum(Zs, axis=axis)


def compute_multiply_share(alpha_0, alpha_1, beta_0, beta_1, share_map):
    alpha = reconstruct(alpha_0, alpha_1)
    beta = reconstruct(beta_0, beta_1)

    As = share_map["As"]
    Bs = share_map["Bs"]
    Cs = share_map["Cs"]
    i = 0 if share_map["is_party_a"] else 1

    Zs = i * np.multiply(alpha, beta) + np.multiply(As, beta) + np.multiply(alpha, Bs) + Cs
    return Zs


def compute_sum_of_multiply_share(alpha_0, alpha_1, beta_0, beta_1, share_map, axis=None):
    Zs = compute_multiply_share(alpha_0, alpha_1, beta_0, beta_1, share_map)
    return np.sum(Zs, axis=axis)
