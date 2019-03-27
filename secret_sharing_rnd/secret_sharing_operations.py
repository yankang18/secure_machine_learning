import numpy as np

from common_data import generate_random_matrix


def share(X):
    n, d = X.shape
    X0 = generate_random_matrix(n, d)
    X1 = X - X0
    return X0, X1


def reconstruct(share0, share1):
    return share0 + share1


def compute_alpha_beta_share(share_map):
    As = share_map["As"]
    Bs = share_map["Bs"]
    Xs = share_map["Xs"]
    Ys = share_map["Ys"]

    alpha_s = Xs - As
    beta_s = Ys - Bs

    return alpha_s, beta_s


def compute_mul_share(alpha_0, alpha_1, beta_0, beta_1, share_map):
    alpha = reconstruct(alpha_0, alpha_1)
    beta = reconstruct(beta_0, beta_1)

    As = share_map["As"]
    Bs = share_map["Bs"]
    Cs = share_map["Cs"]
    i = 0 if share_map["is_party_a"] else 1

    Zs = i * np.dot(alpha, beta) + np.dot(As, beta) + np.dot(alpha, Bs) + Cs
    return Zs


def mul(a_share_map, b_share_map):
    alpha_0, beta_0 = compute_alpha_beta_share(a_share_map)
    alpha_1, beta_1 = compute_alpha_beta_share(b_share_map)

    Z0 = compute_mul_share(alpha_0, alpha_1, beta_0, beta_1, a_share_map)
    Z1 = compute_mul_share(alpha_0, alpha_1, beta_0, beta_1, b_share_map)

    return Z0, Z1
