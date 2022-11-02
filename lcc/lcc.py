import copy
import logging

import numpy as np
# import torch


def modular_inv(a, p):
    x, y, m = 1, 0, p
    while a > 1:
        q = a // m
        t = m

        m = np.mod(a, m)
        a = t
        t = y

        y, x = x - np.int64(q) * np.int64(y), t

        if x < 0:
            x = np.mod(x, p)
    return np.mod(x, p)


def divmodp(_num, _den, _p):
    # compute num / denom modulo prime p
    _num = np.mod(_num, _p)
    _den = np.mod(_den, _p)
    _inv = modular_inv(_den, _p)
    return np.mod(np.int64(_num) * np.int64(_inv), _p)


def PI(vals, p):  # upper-case PI -- product of inputs
    accum = 1
    for v in vals:
        tmp = np.mod(v, p)
        accum = np.mod(accum * tmp, p)
    return accum


def gen_Lagrange_coeffs(alpha_s, beta_s, p, is_K1=False):
    num_alpha = 1 if is_K1 else len(alpha_s)
    U = np.zeros((len(beta_s), num_alpha), dtype="int64")

    denom = np.zeros((len(beta_s)), dtype="int64") # prod (beta_i - beta_l)
    for i in range(len(beta_s)):
        cur_beta = beta_s[i]
        den = PI([cur_beta - o for o in beta_s if cur_beta != o], p)
        denom[i] = den

    enum = np.zeros((num_alpha), dtype="int64")  # prod (alpha_j - beta_l)
    for j in range(num_alpha):
        enum[j] = PI([alpha_s[j] - o for o in beta_s], p)

    for i in range(len(beta_s)):
        for j in range(num_alpha):
            current_denom = np.mod(np.mod(alpha_s[j] - beta_s[i], p) * denom[i], p)
            U[i][j] = divmodp(enum[j], current_denom, p) # enum / denom
    return U.astype("int64")


# def LCC_encoding_with_points(X, alpha_s, beta_s, p):
#     m, d = np.shape(X)
#     U = gen_Lagrange_coeffs(beta_s, alpha_s, p)
#     X_LCC = np.zeros((len(beta_s), d), dtype="int64")
#     for i in range(len(beta_s)):
#         X_LCC[i, :] = np.dot(np.reshape(U[i, :], (1, len(alpha_s))), X)
#     return np.mod(X_LCC, p)


# def LCC_decoding_with_points(f_eval, eval_points, target_points, p):
#     alpha_s_eval = eval_points
#     beta_s = target_points
#     U_dec = gen_Lagrange_coeffs(beta_s, alpha_s_eval, p)
#     f_recon = np.mod((U_dec).dot(f_eval), p)
#     return f_recon


def quantize(X, q_bit, p):
    X_int = np.round(X * (2 ** q_bit)).astype("int64")
    is_negative = (abs(np.sign(X_int)) - np.sign(X_int)) / 2
    out = X_int + p * is_negative
    return np.mod(out.astype("int64"), p)


def dequantize(X_q, q_bit, p):
    flag = X_q - (p - 1) / 2
    is_negative = (abs(np.sign(flag)) + np.sign(flag)) / 2
    X_q = X_q - p * is_negative
    return X_q.astype(float) / (2 ** q_bit)


# def transform_finite_to_tensor(model_params, p, q_bits):
#     for k in model_params.keys():
#         tmp = np.array(model_params[k])
#         tmp_real = dequantize(tmp, q_bits, p)
#         tmp_real = (
#             torch.Tensor([tmp_real])
#             if isinstance(tmp_real, np.floating)
#             else torch.Tensor(tmp_real)
#         )
#         model_params[k] = tmp_real
#     return model_params


# def transform_tensor_to_finite(model_params, p, q_bits):
#     for k in model_params.keys():
#         tmp = np.array(model_params[k])
#         tmp_finite = dequantize(tmp, q_bits, p)
#         model_params[k] = tmp_finite
#     return model_params


# def model_dimension(weights):
#     logging.info("Get model dimension")
#     dimensions = []
#     for k in weights.keys():
#         tmp = weights[k].cpu().detach().numpy()
#         cur_shape = tmp.shape
#         _d = int(np.prod(cur_shape))
#         dimensions.append(_d)
#     total_dimension = sum(dimensions)
#     logging.info("Dimension of model d is %d." % total_dimension)
#     return dimensions, total_dimension


# def aggregate_models_in_finite(weights_finite, prime_number):
#     """
#     weights_finite : array of state_dict() of length n_active_users
#     prime_number   : size of the finite field
#     """
#     w_sum = copy.deepcopy(weights_finite[0])

#     for key in w_sum.keys():

#         for i in range(1, len(weights_finite)):
#             w_sum[key] += weights_finite[i][key]
#             w_sum[key] = np.mod(w_sum[key], prime_number)

#     return w_sum