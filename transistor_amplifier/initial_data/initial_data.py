import numpy as np
import sympy as smp

R = np.array([1000.] + [9000.] * 5)
R0, R1, R2, R3, R4, R5 = R

C = np.array([1., 2., 3.]) * 1e-6
C1, C2, C3 = C

Ub = 6

M = np.array([[-C1, C1, 0., 0., 0.],
              [C1, -C1, 0., 0., 0.],
              [0., 0., -C2, 0., 0.],
              [0., 0., 0., -C3, C3],
              [0., 0., 0., C3, -C3]])


def f(arg):
    U = arg
    return pow(10, -6) * (pow(np.e, U / 0.026) - 1)


def Ue(t):
    return 0.4 * smp.sin(200 * np.pi * t)


def DAE(args):
    t, U1, U2, U3, U4, U5 = args

    f1 = U1 / R0 - Ue(t) / R0
    f2 = U2 * (1. / R1 + 1. / R2) + 0.01 * f(U2 - U3) - Ub / R2
    f3 = U3 / R3 - f(U2 - U3)
    f4 = U4 / R4 + 0.99 * f(U2 - U3) - Ub / R4
    f5 = U5 / R5

    equations = np.array([f1, f2, f3, f4, f5])

    return equations


def get_initial_condition():
    U1 = 0
    U2 = U3 = (Ub * R1) / (R1 + R2)
    U4 = Ub
    U5 = 0
    return U1, U2, U3, U4, U5
