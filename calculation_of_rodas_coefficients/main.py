import numpy as np
from scipy import linalg

gamma = 0.25
s = 6
b_i = np.array([np.nan for _ in range(s)])
alfa_i = np.array([np.nan for _ in range(s)])
alfa_ij = np.array([np.nan for _ in range(s * s)]).reshape(s, s)
beta_i = np.array([np.nan for _ in range(s)])
beta_ij = np.array([np.nan for _ in range(s * s)]).reshape(s, s)
omega_ij = np.array([np.nan for _ in range(s * s)]).reshape(s, s)
gamma_ij = np.array([np.nan for _ in range(s * s)]).reshape(s, s)


def set_initial_values() -> None:
    for i in range(s):
        for j in range(i, s):
            alfa_ij[i, j] = 0.
            if j != s:
                beta_ij[i, j] = 0.

    alfa_i[0] = 0.
    alfa_i[1] = 0.386
    alfa_i[2] = 0.21
    alfa_i[3] = 0.63
    alfa_i[4] = 1.
    alfa_i[5] = 1.

    alfa_ij[1, 0] = alfa_i[1]

    beta_i[0] = 0.
    beta_i[1] = 0.0317
    beta_i[2] = 0.0635
    beta_i[3] = 0.3438

    beta_ij[1, 0] = beta_i[1]
    for i in range(s):
        beta_ij[i, i] = gamma


def step_1() -> None:
    b_i[5] = gamma

    # Calculation: b_i[0], b_i[1], b_i[2], b_i[3], b_i[4]
    A = np.array([
        [1, 1, 1, 1, 1],
        [0, beta_i[1], beta_i[2], beta_i[3], 1 - gamma],
        [0, alfa_i[1] ** 2, alfa_i[2] ** 2, alfa_i[3] ** 2, 1],
        [0, alfa_i[1] ** 3, alfa_i[2] ** 3, alfa_i[3] ** 3, 1],
        [alfa_i[0] ** 4, alfa_i[1] ** 4, alfa_i[2] ** 4, alfa_i[3] ** 4, alfa_i[4] ** 4]
    ])
    b = np.array([1 - b_i[5], 1/2 - b_i[5] * (1 - gamma) - gamma, 1/3 - b_i[5],
                  1/4 - b_i[5], 1/5 - b_i[5] * alfa_i[5] ** 4])

    b_i[:-1] = linalg.solve(A, b)

    # Calculation: beta_ij[5, i]
    beta_ij[5] = b_i

    # Calculation: beta_i[5]
    beta_i[5] = beta_ij[5, :-1].sum()


def step_2() -> None:
    # Calculation: b_i[2] * beta_ij[2, 1] + b_i[3] * beta_ij[3, 1], b_i[3] * beta_ij[3, 2]
    A = np.array([
        [beta_i[1], beta_i[2]],
        [alfa_i[1] ** 2, alfa_i[2] ** 2]
    ])
    b = np.array([1/6 - (b_i[4] + b_i[5]) * (1/2 - 2 * gamma + gamma ** 2) - gamma + gamma ** 2,
                  1/12 - (b_i[4] + b_i[5]) * (1 / 3 - gamma) - gamma / 3])

    b2beta21b3beta31, b3beta32 = linalg.solve(A, b)

    # Calculation: beta_ij[2, 1]
    A = np.array([b3beta32 * beta_i[1]])
    b = np.array([1/24 - (b_i[4] + b_i[5]) * (1/6 - 3/2 * gamma + 3 * gamma ** 2 - gamma ** 3) -
                  gamma / 2 + 3/2 * gamma ** 2 - gamma ** 3])

    beta_ij[2, 1] = linalg.solve(A, b)

    # Calculation: beta_ij[3, 1], beta_ij[3, 2]
    beta_ij[3, 2] = b3beta32 / b_i[3]
    beta_ij[3, 1] = (b2beta21b3beta31 - b_i[2] * beta_ij[2, 1]) / b_i[3]

    # Calculation: beta_ij[2, 0], beta_ij[3, 0]
    beta_ij[2, 0] = beta_i[2] - beta_ij[2, 1]
    beta_ij[3, 0] = beta_i[3] - beta_ij[3, 1] - beta_ij[3, 2]


def step_3() -> None:
    alfa_ij[5, 4] = gamma

    # Calculation: alfa_ij[5, 1], alfa_ij[5, 2], alfa_ij[5, 3]
    A = np.array([
        [beta_i[1], beta_i[2], beta_i[3]],
        [alfa_i[1] ** 2, alfa_i[2] ** 2, alfa_i[3] ** 2],
        [0, beta_ij[2, 1] * beta_i[1], (beta_ij[3, 1:3] * beta_i[1:3]).sum()]
    ])
    b = np.array([1/2 - 2 * gamma + gamma ** 2, 1/3 - gamma, 1/6 - 3/2 * gamma + 3 * gamma ** 2 - gamma ** 3])

    alfa_ij[5, 1:4] = linalg.solve(A, b)

    # Calculation: alfa_ij[5, 0]
    alfa_ij[5, 0] = alfa_i[5] - alfa_ij[5, 1:5].sum()

    # Calculation: beta_ij[4, j], beta_i[4]
    beta_ij[4, :4] = alfa_ij[5, :4]
    beta_i[4] = beta_ij[4, :-2].sum()

    # Calculation: omega_ij
    global omega_ij
    omega_ij = linalg.inv(beta_ij)


def step_4() -> None:
    # Calculation: alfa_ij[4, 0], alfa_ij[4, 1], alfa_ij[4, 2], alfa_ij[4, 3]
    A = np.array([
        [0, beta_i[1], beta_i[2], beta_i[3]],
        [0, omega_ij[1, 1] * alfa_i[1] ** 2, omega_ij[2, 1] * alfa_i[1] ** 2 + omega_ij[2, 2] * alfa_i[2] ** 2,
         omega_ij[3, 1] * alfa_i[1] ** 2 + omega_ij[3, 2] * alfa_i[2] ** 2 + omega_ij[3, 3] * alfa_i[3] ** 2],
        [1, 1, 1, 1],
        [omega_ij[0, :1].sum(), omega_ij[1, :2].sum(), omega_ij[2, :3].sum(), omega_ij[3, :4].sum()]
    ])
    b = np.array([1/2 - gamma, 1, 1, 1])

    alfa_ij[4, 0:4] = linalg.solve(A, b)


def step_5() -> None:
    # Calculation: alfa_ij[2, 1], alfa_ij[3, 1], alfa_ij[3, 2]
    A = np.array([
        [b_i[2] * alfa_i[2] * beta_i[1], b_i[3] * alfa_i[3] * beta_i[1], b_i[3] * alfa_i[3] * beta_i[2]],
        [b_i[2] * alfa_i[2] * omega_ij[1, 1] * alfa_i[1] ** 2, b_i[3] * alfa_i[3] * omega_ij[1, 1] * alfa_i[1] ** 2,
         b_i[3] * alfa_i[3] * (omega_ij[2, 1] * alfa_i[1] ** 2 + omega_ij[2, 2] * alfa_i[2] ** 2)],
        [b_i[2] * alfa_i[2] ** 2 * beta_i[1], b_i[3] * alfa_i[3] ** 2 * beta_i[1], b_i[3] * alfa_i[3] ** 2 * beta_i[2]]
    ])
    b = np.array([1/8 - (b_i[4] + b_i[5]) * (1/2 - gamma) - gamma / 3, 1/4 - (b_i[4] + b_i[5]),
                  1/10 - b_i[4] * (alfa_ij[4] * beta_i).sum() - b_i[5] * (alfa_ij[5] * beta_i).sum() - gamma / 4])

    alfa_ij[2, 1], alfa_ij[3, 1], alfa_ij[3, 2] = linalg.solve(A, b)

    # Calculation: alfa_ij[2, 0], alfa_ij[3, 0]
    alfa_ij[2, 0] = alfa_i[2] - alfa_ij[2, 1]
    alfa_ij[3, 0] = alfa_i[3] - alfa_ij[3, 1] - alfa_ij[3, 2]

    # Calculation: gamma_ij
    global gamma_ij
    gamma_ij = beta_ij - alfa_ij


def save_rodas_coefficients() -> None:
    np.save('rodas_coefficients/alfa', alfa_ij)
    np.save('rodas_coefficients/gamma', gamma_ij)
    np.save('rodas_coefficients/b', b_i)

    np.savetxt('rodas_coefficients/alfa.csv', alfa_ij)
    np.savetxt('rodas_coefficients/gamma.csv', gamma_ij)
    np.savetxt('rodas_coefficients/b.csv', b_i)


def checking() -> None:
    inv_gamma_ij = np.linalg.inv(gamma_ij)

    new_alfa_ij = np.dot(alfa_ij, inv_gamma_ij)
    c_ij = np.diag(np.diag(gamma_ij) ** -1) - inv_gamma_ij

    print('inv_gamma_ij')
    print(inv_gamma_ij)

    print('new_alfa_ij')
    print(new_alfa_ij)

    print('c_ij')
    print(c_ij)


def calculate_rodas_coefficients() -> None:
    set_initial_values()
    step_1()
    step_2()
    step_3()
    step_4()
    step_5()


def calculate_and_save_rodas_coefficients() -> None:
    calculate_rodas_coefficients()
    save_rodas_coefficients()


if __name__ == '__main__':
    """
    This script calculates the defining coefficients of the Rosenbrock method, which are used in the RODAS program.
    """
    calculate_and_save_rodas_coefficients()
    print('RODAS coefficients have been calculated and saved.')
