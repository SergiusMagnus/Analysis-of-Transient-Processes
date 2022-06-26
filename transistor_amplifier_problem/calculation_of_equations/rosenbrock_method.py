import numpy as np
from scipy import linalg

from transient_analysis.data import RODAS_COEFFICIENTS_PATH

alfa = np.load(f'{RODAS_COEFFICIENTS_PATH}alfa.npy')
gamma = np.load(f'{RODAS_COEFFICIENTS_PATH}gamma.npy')
b = np.load(f'{RODAS_COEFFICIENTS_PATH}b.npy')

alfa_sum = np.array([np.sum(alfa[i, :i]) for i in range(alfa.shape[0])])
gamma_sum = np.sum(gamma, axis=1)
assessed_b = alfa[-1]
stages_number = b.size


def calculate_values(M, DAE, J, h, args):
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
            phi = DAE(args)
            gamma_times_k_sum = np.dot(gamma[i], k)

            k[i] = linalg.solve(np.float64(M - h * J[:, 1:] * gamma[i][i]), np.float64(h * phi + gamma_sum[i] * (h ** 2) * np.concatenate(J[:, :1]) +
                                h * np.sum(J[:, 1:] * gamma_times_k_sum, axis=1)))
        return k

    k = calculate_k()
    next_y = current_y + np.sum(np.array([b * k[:, i] for i in range(number_of_equations)]), axis=1)
    assessed_y = current_y + np.sum(np.array([assessed_b * k[:, i] for i in range(number_of_equations)]), axis=1)

    err = np.amax(np.abs(next_y - assessed_y))

    return err, next_y
