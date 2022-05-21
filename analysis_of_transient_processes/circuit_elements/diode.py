import numpy as np


class Diode:
    """Diode"""

    def __init__(self):
        self.IS = 1e-14
        self.ISR = 0
        self.N = 1
        self.NR = 2
        self.IKF = np.inf
        self.M = 0.5
        self.VJ = 1
        self.IBV = 1e-10
        self.BV = np.inf
        self.NBV = 1
        self.IBVL = 0
        self.NBVL = 1

        k = 1.38 * 1e-23
        T = 300
        q = 1.6 * 1e-19
        self.V_t = k * T / q

        self.output_node_index = -1
        self.input_node_index = -1

    def __repr__(self):
        return 'Diode'

    def get_u(self, args):
        if self.output_node_index != -1:
            u0 = args[self.output_node_index]
        else:
            u0 = 0

        if self.input_node_index != -1:
            u1 = args[self.input_node_index]
        else:
            u1 = 0

        return u0 - u1

    def current(self, args):
        u = self.get_u(args)

        I_nrm = self.IS * (np.exp(u / (self.N * self.V_t)) - 1)
        I_rec = self.ISR * (np.exp(u / (self.NR * self.V_t)) - 1)

        if self.IKF > 0 and self.IKF is not np.inf:
            K_inj = (self.IKF / (self.IKF + I_nrm)) ** (1 / 2)
        else:
            K_inj = 1

        K_gen = (((1 - u) / self.VJ) ** 2 + 0.005) ** (self.M / 2)

        I_fwd = I_nrm * K_inj + I_rec * K_gen

        I_rev_high = self.IBV * np.exp(-(u + self.BV) / (self.NBV * self.V_t))
        I_rev_low = self.IBVL * np.exp(-(u + self.BV) / (self.NBVL * self.V_t))

        I_rev = I_rev_high + I_rev_low

        I = I_fwd - I_rev

        return I

    def minus_current(self, args):
        return -self.current(args)

    def current_diff(self, args):
        u = self.get_u(args)

        if self.IKF > 0 and self.IKF is not np.inf:
            I_diff = self.IBV * np.exp((-self.BV - u) / (self.NBV * self.V_t)) / (self.NBV * self.V_t) + \
                     self.IBVL * np.exp((-self.BV - u) / (self.NBVL * self.V_t))/(self.NBVL * self.V_t) - 0.5 * \
                     self.IS ** 2 * (self.IKF / (self.IKF + self.IS * (np.exp(u / (self.N * self.V_t)) - 1))) ** 0.5 \
                     * (np.exp(u / (self.N * self.V_t)) - 1) * np.exp(u / (self.N * self.V_t)) / \
                     (self.N * self.V_t * (self.IKF + self.IS * (np.exp(u / (self.N * self.V_t)) - 1))) + self.IS * \
                     (self.IKF / (self.IKF + self.IS * (np.exp(u / (self.N * self.V_t)) - 1))) ** 0.5 * \
                     np.exp(u / (self.N * self.V_t)) / (self.N * self.V_t) + self.ISR * self.M * \
                     (0.005 + (1 - u) ** 2 / self.VJ ** 2) ** (self.M / 2) * (2 * u - 2) * \
                     (np.exp(u / (self.NR * self.V_t)) - 1) / \
                     (2 * self.VJ ** 2 * (0.005 + (1 - u) ** 2 / self.VJ ** 2)) + self.ISR * \
                     (0.005 + (1 - u) ** 2 / self.VJ ** 2) ** (self.M / 2) * np.exp(u / (self.NR * self.V_t)) / \
                     (self.NR * self.V_t)
        else:
            I_diff = self.IBV * np.exp((-self.BV - u) / (self.NBV * self.V_t)) / (self.NBV * self.V_t) + \
                     self.IBVL * np.exp((-self.BV - u) / (self.NBVL * self.V_t)) / (self.NBVL * self.V_t) + \
                     self.IS * np.exp(u / (self.N * self.V_t)) / (self.N * self.V_t) + self.ISR * self.M * \
                     (0.005 + (1 - u) ** 2 / self.VJ ** 2) ** (self.M / 2) * (2 * u - 2) * \
                     (np.exp(u / (self.NR * self.V_t)) - 1) / \
                     (2 * self.VJ ** 2 * (0.005 + (1 - u) ** 2 / self.VJ ** 2)) + self.ISR * \
                     (0.005 + (1 - u) ** 2 / self.VJ ** 2) ** (self.M / 2) * np.exp(u / (self.NR * self.V_t)) / \
                     (self.NR * self.V_t)
        return I_diff

    def minus_current_diff(self, args):
        return -self.current_diff(args)

    def form_matrices(self, B, A, J, g, f, df_dt, nodes_index, current_index):
        self.output_node_index, self.input_node_index = nodes_index

        # voltage
        if self.output_node_index != -1:
            g[self.output_node_index].append(self.current)
            J[self.output_node_index][self.output_node_index].append(self.current_diff)

        if self.input_node_index != -1:
            g[self.input_node_index].append(self.minus_current)
            J[self.input_node_index][self.input_node_index].append(self.current_diff)

        if self.output_node_index != -1 and self.input_node_index != -1:
            J[self.output_node_index][self.input_node_index].append(self.minus_current_diff)
            J[self.input_node_index][self.output_node_index].append(self.minus_current_diff)

        # current
        if current_index != -1:
            A[current_index][current_index] += -1
            g[current_index].append(self.current)

        if self.output_node_index != -1 and current_index != -1:
            J[current_index][self.output_node_index].append(self.current_diff)

        if self.input_node_index != -1 and current_index != -1:
            J[current_index][self.input_node_index].append(self.minus_current_diff)
