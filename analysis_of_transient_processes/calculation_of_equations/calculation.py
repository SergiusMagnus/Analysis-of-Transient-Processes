import numpy as np

from .rosenbrock_method import calculate_values as rosenbrock


def calculate_J_and_df_dt(smb_J, smb_df_dt, current_args):
    current_J = np.zeros((len(smb_J), len(smb_J)))
    current_df_dt = np.zeros(len(smb_df_dt))

    for i in range(len(smb_J)):
        for j in range(len(smb_J)):
            for J in smb_J[i][j]:
                current_J[i, j] += J(current_args[1:])

    for i in range(len(smb_df_dt)):
        for df_dt in smb_df_dt[i]:
            current_df_dt[i] += df_dt(current_args[0])

    return current_J, current_df_dt


def calculate_solution(circuit, start_t, end_t, initial_values=None, tol=1e-6):
    number_of_equations = len(circuit.variables)

    t = np.array([0])
    y = np.zeros((number_of_equations, 1))
    if initial_values is not None:
        y[:, 0] = initial_values
    else:
        y[:, 0] = np.zeros(number_of_equations)

    B, A, smb_J, smb_g, smb_f, smb_df_dt = circuit.get_matrices_of_electric_circuit()

    current_t = start_t
    max_step = 1 / 100
    step_t = max_step / 100
    counter = 0

    while True:
        if current_t + step_t <= end_t:
            current_args = np.array([t[-1], *y[:, -1]])
            J, df_dt = calculate_J_and_df_dt(smb_J, smb_df_dt, current_args)
            err, next_y = rosenbrock(B, A, J, smb_g, smb_f, df_dt, step_t, current_args)

            if err <= tol:
                t = np.append(t, t[-1] + step_t)
                y = np.append(y, next_y.reshape((number_of_equations, 1)), axis=1)
                current_t = t[-1]
                counter = 0
            else:
                counter += 1

            if counter == 2:
                step_t /= 10
                counter = 0
            else:
                new_step = step_t * min(6., max(0.2, 0.9 * (tol / err) ** (1 / 4)))
                step_t = new_step if new_step < max_step else max_step
        else:
            step_t = end_t - current_t
            current_args = np.array([t[-1], *y[:, -1]])
            J, df_dt = calculate_J_and_df_dt(smb_J, smb_df_dt, current_args)
            err, next_y = rosenbrock(B, A, J, smb_g, smb_f, df_dt, step_t, current_args)

            if err <= tol:
                t = np.append(t, t[-1] + step_t)
                y = np.append(y, next_y.reshape((number_of_equations, 1)), axis=1)
                current_t = t[-1]
                counter = 0
            else:
                counter += 1

            if counter == 2:
                step_t /= 10
                counter = 0
            else:
                new_step = step_t * min(6., max(0.2, 0.9 * (tol / err) ** (1 / 4)))
                step_t = new_step if new_step < max_step else max_step
            if current_t >= end_t:
                break

    return t, y
