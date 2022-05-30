import numpy as np
import sympy as smp

from .rosenbrock_method import calculate_values as rosenbrock


def calculate_solution(M, DAE, start_t, end_t, initial_values=None, tol=1e-6):
    number_of_equations = len(M)

    t = np.array([0])
    y = np.zeros((number_of_equations, 1))
    if initial_values is not None:
        y[:, 0] = initial_values()
    else:
        y[:, 0] = np.zeros(number_of_equations)

    symb_args = np.array([smp.symbols('arg_' + str(i + 1)) for i in range(number_of_equations + 1)])
    symb_J = smp.Matrix(DAE(symb_args)).jacobian(symb_args)

    current_t = start_t
    max_step = 1 / 100
    step_t = max_step / 100
    counter = 0

    while True:
        if current_t + step_t <= end_t:
            current_args = np.array([t[-1], *y[:, -1]])
            J = np.array(symb_J.subs(zip(symb_args, current_args)))
            err, next_y = rosenbrock(M, DAE, J, step_t, current_args)

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
            J = np.array(symb_J.subs(zip(symb_args, current_args)))
            err, next_y = rosenbrock(M, DAE, J, step_t, current_args)

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
