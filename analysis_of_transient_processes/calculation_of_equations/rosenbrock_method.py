import numpy as np
from scipy import linalg

from analysis_of_transient_processes.data import RODAS_COEFFICIENTS_PATH

alfa = np.load(f'{RODAS_COEFFICIENTS_PATH}alfa.npy')
gamma = np.load(f'{RODAS_COEFFICIENTS_PATH}gamma.npy')
b = np.load(f'{RODAS_COEFFICIENTS_PATH}b.npy')

alfa_sum = np.array([np.sum(alfa[i, :i]) for i in range(alfa.shape[0])])
gamma_sum = np.sum(gamma, axis=1)
assessed_b = alfa[-1]
stages_number = b.size


def calculate_g_and_f(smb_g, smb_f, current_args):
    current_g = np.zeros(len(smb_g))
    current_f = np.zeros(len(smb_f))

    for i in range(len(smb_g)):
        for g in smb_g[i]:
            current_g[i] += g(current_args[1:])

    for i in range(len(smb_f)):
        for f in smb_f[i]:
            current_f[i] += f(current_args[0])

    return current_g, current_f


def calculate_values(M, A, J, smb_g, smb_f, df_dt, h, args):
    number_of_equations = M.shape[0]
    current_t = args[0]
    current_y = args[1:]

    def calculate_k():
        k = np.zeros((stages_number, number_of_equations))

        for i in range(stages_number):
            args = np.concatenate(
                ([current_t + alfa_sum[i] * h],
                 current_y + np.sum(np.array([alfa[i, :i] * k[:i, j] for j in range(number_of_equations)]), axis=1))
            )
            g, f = calculate_g_and_f(smb_g, smb_f, args)
            phi = -(np.sum(A * args[1:], axis=1)) - g - f
            gamma_times_k_sum = np.dot(gamma[i], k)
            A_J = -(A + J)
            k[i] = linalg.solve(M - h * A_J * gamma[i][i], h * phi + gamma_sum[i] * (h ** 2) * df_dt +
                                h * np.sum(A_J * gamma_times_k_sum, axis=1))
        return k

    k = calculate_k()
    next_y = current_y + np.sum(np.array([b * k[:, i] for i in range(number_of_equations)]), axis=1)
    assessed_y = current_y + np.sum(np.array([assessed_b * k[:, i] for i in range(number_of_equations)]), axis=1)

    err = np.amax(np.abs(next_y - assessed_y))

    return err, next_y
