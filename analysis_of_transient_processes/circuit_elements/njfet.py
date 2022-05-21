import numpy as np


class NJFET:
    """NJFET"""

    def __init__(self):
        self.JFET_type = 1  # 1 - n-type, -1 - p-type

        self.IS = 1e-14
        self.N = 1
        self.ISR = 0
        self.NR = 2
        self.PB = 1
        self.M = 0.5
        self.VTO = -2
        # self.VTO = -2.617
        self.BETA = 1e-4
        # self.BETA = 1.578e-3
        self.LAMBDA = 0
        # self.LAMBDA = 1.89e-3
        self.ALPHA = 0
        self.VK = 0

        k = 1.38 * 1e-23
        T = 300
        q = 1.6 * 1e-19
        self.V_t = k * T / q

        self.source_node_index = -1
        self.drain_node_index = -1
        self.gate_node_index = -1

    def __repr__(self):
        return 'n-JFET'

    def get_u(self, args):
        if self.source_node_index != -1:
            u_s = args[self.source_node_index]
        else:
            u_s = 0

        if self.drain_node_index != -1:
            u_d = args[self.drain_node_index]
        else:
            u_d = 0

        if self.gate_node_index != -1:
            u_g = args[self.gate_node_index]
        else:
            u_g = 0

        u_gs = self.JFET_type * (u_g - u_s)
        u_gd = self.JFET_type * (u_g - u_d)
        u_ds = self.JFET_type * (u_d - u_s)

        return u_gs, u_gd, u_ds

    def I_gs(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)

        I_n = self.IS * (np.exp(u_gs / (self.N * self.V_t)) - 1)
        I_r = self.ISR * (np.exp(u_gs / (self.NR * self.V_t)) - 1)
        K_g = ((1 - u_gs / self.PB) ** 2 + 0.005) ** (self.M / 2)
        I_gs = I_n + I_r * K_g

        return self.JFET_type * I_gs

    def minus_I_gs(self, args):
        return -self.I_gs(args)

    def I_drain(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)
        I_drain = 0

        if u_ds >= 0:
            if u_gs - self.VTO <= 0:
                I_drain = 0
            elif u_ds <= u_gs - self.VTO:
                I_drain = self.BETA * (1 + self.LAMBDA * u_ds) * u_ds * (2 * (u_gs - self.VTO) - u_ds)
            elif 0 < u_gs - self.VTO < u_ds:
                I_drain = self.BETA * (1 + self.LAMBDA * u_ds) * (u_gs - self.VTO) ** 2
        else:
            if u_gd - self.VTO <= 0:
                I_drain = 0
            elif -u_ds <= u_gd - self.VTO:
                I_drain = self.BETA * (1 - self.LAMBDA * u_ds) * u_ds * (2 * (u_gd - self.VTO) + u_ds)
            elif 0 < u_gd - self.VTO < -u_ds:
                I_drain = -self.BETA * (1 - self.LAMBDA * u_ds) * (u_gd - self.VTO) ** 2

        return self.JFET_type * I_drain

    def minus_I_drain(self, args):
        return -self.I_drain(args)

    def I_gd(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)

        I_drain = self.I_drain(args)
        I_n = self.IS * (np.exp(u_gd / (self.N * self.V_t)) - 1)
        I_r = self.ISR * (np.exp(u_gd / (self.NR * self.V_t)) - 1)
        K_g = ((1 - u_gd / self.PB) ** 2 + 0.005) ** (self.M / 2)

        if 0 < u_gs - self.VTO < u_ds:
            V_dif = u_ds - (u_gs - self.VTO)
            I_i = I_drain * self.ALPHA * V_dif * np.exp(-self.VK / V_dif)
        else:
            I_i = 0

        I_gd = I_n + I_r * K_g + I_i

        return self.JFET_type * I_gd

    def minus_I_gd(self, args):
        return -self.I_gd(args)

    def I_gs_diff_u_s(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)

        I_diff = -self.IS * np.exp(u_gs / (self.N * self.V_t)) / (self.N * self.V_t) + self.ISR * self.M * \
            (1 - u_gs / self.PB) * ((1 - u_gs / self.PB) ** 2 + 0.005) ** (self.M / 2) * \
            (np.exp(u_gs / (self.NR * self.V_t)) - 1) / (self.PB * ((1 - u_gs / self.PB) ** 2 + 0.005)) - self.ISR * \
            ((1 - u_gs / self.PB) ** 2 + 0.005) ** (self.M / 2) * np.exp(u_gs / (self.NR * self.V_t)) / \
            (self.NR * self.V_t)
        return I_diff

    def minus_I_gs_diff_u_s(self, args):
        return -self.I_gs_diff_u_s(args)

    def I_gs_diff_u_g(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)

        I_diff = self.IS * np.exp(u_gs / (self.N * self.V_t)) / (self.N * self.V_t) - self.ISR * self.M * \
            (1 - u_gs / self.PB) * ((1 - u_gs / self.PB) ** 2 + 0.005) ** (self.M / 2) * \
            (np.exp(u_gs / (self.NR * self.V_t)) - 1) / (self.PB * ((1 - u_gs / self.PB) ** 2 + 0.005)) + self.ISR * \
            ((1 - u_gs / self.PB) ** 2 + 0.005) ** (self.M / 2) * np.exp(u_gs / (self.NR * self.V_t)) / \
            (self.NR * self.V_t)
        return I_diff

    def minus_I_gs_diff_u_g(self, args):
        return -self.I_gs_diff_u_g(args)

    def I_drain_diff_u_s(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)
        I_diff = 0

        if u_ds >= 0:
            if u_gs - self.VTO <= 0:
                I_diff = 0
            elif u_ds <= u_gs - self.VTO:
                I_diff = -self.BETA * self.LAMBDA * u_ds * (-2 * self.VTO + u_gd + u_gs) - \
                         self.BETA * u_ds * (self.LAMBDA * u_ds + 1) - self.BETA * \
                         (self.LAMBDA * u_ds + 1) * (-2 * self.VTO + u_gd + u_gs)
            elif 0 < u_gs - self.VTO < u_ds:
                I_diff = -self.BETA * self.LAMBDA * (-self.VTO + u_gs) ** 2 + self.BETA * \
                         (self.LAMBDA * u_ds + 1) * (2 * self.VTO - 2 * u_gs)
        else:
            if u_gd - self.VTO <= 0:
                I_diff = 0
            elif -u_ds <= u_gd - self.VTO:
                I_diff = self.BETA * self.LAMBDA * u_ds * (-2 * self.VTO + u_gd + u_gs) - \
                         self.BETA * u_ds * (-self.LAMBDA * u_ds + 1) - self.BETA * \
                         (-self.LAMBDA * u_ds + 1) * (-2 * self.VTO + u_gd + u_gs)
            elif 0 < u_gd - self.VTO < -u_ds:
                I_diff = -self.BETA * self.LAMBDA * (-self.VTO + u_gd) ** 2

        return I_diff

    def minus_I_drain_diff_u_s(self, args):
        return -self.I_drain_diff_u_s(args)

    def I_drain_diff_u_d(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)
        I_diff = 0

        if u_ds >= 0:
            if u_gs - self.VTO <= 0:
                I_diff = 0
            elif u_ds <= u_gs - self.VTO:
                I_diff = self.BETA * self.LAMBDA * u_ds * (-2 * self.VTO + u_gs + u_gd) - \
                         self.BETA * u_ds * (self.LAMBDA * u_ds + 1) + self.BETA * \
                         (self.LAMBDA * u_ds + 1) * (-2 * self.VTO + u_gs + u_gd)
            elif 0 < u_gs - self.VTO < u_ds:
                I_diff = self.BETA * self.LAMBDA * (-self.VTO + u_gs) ** 2
        else:
            if u_gd - self.VTO <= 0:
                I_diff = 0
            elif -u_ds <= u_gd - self.VTO:
                I_diff = -self.BETA * self.LAMBDA * u_ds * (-2 * self.VTO + u_gs + u_gd) - \
                         self.BETA * u_ds * (-self.LAMBDA * u_ds + 1) + self.BETA * \
                         (-self.LAMBDA * u_ds + 1) * (-2 * self.VTO + u_gs + u_gd)
            elif 0 < u_gd - self.VTO < -u_ds:
                I_diff = self.BETA * self.LAMBDA * (-self.VTO + u_gd) ** 2 - self.BETA * \
                         (-self.LAMBDA * u_ds + 1) * (2 * self.VTO - 2 * u_gd)

        return I_diff

    def minus_I_drain_diff_u_d(self, args):
        return -self.I_drain_diff_u_d(args)

    def I_drain_diff_u_g(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)
        I_diff = 0

        if u_ds >= 0:
            if u_gs - self.VTO <= 0:
                I_diff = 0
            elif u_ds <= u_gs - self.VTO:
                I_diff = 2 * self.BETA * u_ds * (self.LAMBDA * u_ds + 1)
            elif 0 < u_gs - self.VTO < u_ds:
                I_diff = self.BETA * (self.LAMBDA * u_ds + 1) * (-2 * self.VTO + 2 * u_gs)
        else:
            if u_gd - self.VTO <= 0:
                I_diff = 0
            elif -u_ds <= u_gd - self.VTO:
                I_diff = 2 * self.BETA * u_ds * (-self.LAMBDA * u_ds + 1)
            elif 0 < u_gd - self.VTO < -u_ds:
                I_diff = -self.BETA * (-self.LAMBDA * u_ds + 1) * (-2 * self.VTO + 2 * u_gd)

        return I_diff

    def minus_I_drain_diff_u_g(self, args):
        return -self.I_drain_diff_u_g(args)

    def I_gd_diff_u_s(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)

        if 0 < u_gs - self.VTO < u_ds:
            I_diff = -self.ALPHA * self.BETA * self.LAMBDA * (-self.VTO + u_gs) ** 2 * \
                    (self.VTO - u_gd) * np.exp(-self.VK / (self.VTO - u_gd)) + \
                    self.ALPHA * self.BETA * (self.LAMBDA * u_ds + 1) * (self.VTO - u_gd) * \
                    (2 * self.VTO - 2 * u_gs) * np.exp(-self.VK / (self.VTO - u_gd))
        else:
            I_diff = 0

        return I_diff

    def minus_I_gd_diff_u_s(self, args):
        return -self.I_gd_diff_u_s(args)

    def I_gd_diff_u_d(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)

        if 0 < u_gs - self.VTO < u_ds:
            I_diff = self.ALPHA * self.BETA * self.LAMBDA * (-self.VTO + u_gs) ** 2 * \
                     (self.VTO - u_gd) * np.exp(-self.VK / (self.VTO - u_gd)) + \
                     self.ALPHA * self.BETA * self.VK * (self.LAMBDA * u_ds + 1) * \
                     (-self.VTO + u_gs) ** 2 * np.exp(-self.VK / (self.VTO - u_gd)) / \
                     (self.VTO - u_gd) + self.ALPHA * self.BETA * (self.LAMBDA * u_ds + 1) * \
                     (-self.VTO + u_gs) ** 2 * np.exp(-self.VK / (self.VTO - u_gd)) - self.IS * \
                     np.exp(u_gd / (self.N * self.V_t)) / (self.N * self.V_t) + self.ISR * self.M * \
                     (1 - u_gd / self.PB) * ((1 - u_gd / self.PB) ** 2 + 0.005) ** (self.M / 2) * \
                     (np.exp(u_gd / (self.NR * self.V_t)) - 1) / \
                     (self.PB * ((1 - u_gd / self.PB) ** 2 + 0.005)) - self.ISR * \
                     ((1 - u_gd / self.PB) ** 2 + 0.005) ** (self.M / 2) * \
                     np.exp(u_gd / (self.NR * self.V_t)) / (self.NR * self.V_t)
        else:
            I_diff = -self.IS * np.exp(u_gd / (self.N * self.V_t)) / (self.N * self.V_t) + self.ISR * \
                     self.M * (1 - u_gd / self.PB) * ((1 - u_gd / self.PB) ** 2 + 0.005) ** \
                     (self.M / 2) * (np.exp(u_gd / (self.NR * self.V_t)) - 1) / \
                     (self.PB * ((1 - u_gd / self.PB) ** 2 + 0.005)) - self.ISR * \
                     ((1 - u_gd / self.PB) ** 2 + 0.005) ** (self.M / 2) * \
                     np.exp(u_gd / (self.NR * self.V_t)) / (self.NR * self.V_t)

        return I_diff

    def minus_I_gd_diff_u_d(self, args):
        return -self.I_gd_diff_u_d(args)

    def I_gd_diff_u_g(self, args):
        u_gs, u_gd, u_ds = self.get_u(args)

        if 0 < u_gs - self.VTO < u_ds:
            I_diff = -self.ALPHA * self.BETA * self.VK * (self.LAMBDA * u_ds + 1) * \
                     (-self.VTO + u_gs) ** 2 * np.exp(-self.VK / (self.VTO - u_gd)) / \
                     (self.VTO - u_gd) + self.ALPHA * self.BETA * (self.LAMBDA * u_ds + 1) * \
                     (-2 * self.VTO + 2 * u_gs) * (self.VTO - u_gd) * \
                     np.exp(-self.VK / (self.VTO - u_gd)) - self.ALPHA * self.BETA * \
                     (self.LAMBDA * u_ds + 1) * (-self.VTO + u_gs) ** 2 * \
                     np.exp(-self.VK / (self.VTO - u_gd)) + self.IS * \
                     np.exp(u_gd / (self.N * self.V_t)) / (self.N * self.V_t) - self.ISR * self.M * \
                     (1 - u_gd / self.PB) * ((1 - u_gd / self.PB) ** 2 + 0.005) ** (self.M / 2) * \
                     (np.exp(u_gd / (self.NR * self.V_t)) - 1) / \
                     (self.PB * ((1 - u_gd / self.PB) ** 2 + 0.005)) + self.ISR * \
                     ((1 - u_gd / self.PB) ** 2 + 0.005) ** (self.M / 2) * \
                     np.exp(u_gd / (self.NR * self.V_t)) / (self.NR * self.V_t)
        else:
            I_diff = self.IS * np.exp(u_gd / (self.N * self.V_t)) / (self.N * self.V_t) - \
                     self.ISR * self.M * (1 - u_gd / self.PB) * ((1 - u_gd / self.PB) ** 2 + 0.005) ** \
                     (self.M / 2) * (np.exp(u_gd / (self.NR * self.V_t)) - 1) / \
                     (self.PB * ((1 - u_gd / self.PB) ** 2 + 0.005)) + self.ISR * \
                     ((1 - u_gd / self.PB) ** 2 + 0.005) ** (self.M/2) * \
                     np.exp(u_gd / (self.NR * self.V_t)) / (self.NR * self.V_t)

        return I_diff

    def minus_I_gd_diff_u_g(self, args):
        return -self.I_gd_diff_u_g(args)

    def form_matrices(self, B, A, J, g, f, df_dt, nodes_index, current_index):
        self.source_node_index, self.drain_node_index, self.gate_node_index = nodes_index
        source_current_index, drain_current_index, gate_current_index = current_index

        # voltage
        if self.source_node_index != -1:
            g[self.source_node_index].append(self.minus_I_drain)
            g[self.source_node_index].append(self.minus_I_gs)
            J[self.source_node_index][self.source_node_index].append(self.minus_I_drain_diff_u_s)
            J[self.source_node_index][self.source_node_index].append(self.minus_I_gs_diff_u_s)

        if self.drain_node_index != -1:
            g[self.drain_node_index].append(self.I_drain)
            g[self.drain_node_index].append(self.minus_I_gd)
            J[self.drain_node_index][self.drain_node_index].append(self.I_drain_diff_u_d)
            J[self.drain_node_index][self.drain_node_index].append(self.minus_I_gd_diff_u_d)

        if self.gate_node_index != -1:
            g[self.gate_node_index].append(self.I_gs)
            g[self.gate_node_index].append(self.I_gd)
            J[self.gate_node_index][self.gate_node_index].append(self.I_gs_diff_u_g)
            J[self.gate_node_index][self.gate_node_index].append(self.I_gd_diff_u_g)

        if self.source_node_index != -1 and self.drain_node_index != -1:
            J[self.source_node_index][self.drain_node_index].append(self.minus_I_drain_diff_u_d)
            J[self.drain_node_index][self.source_node_index].append(self.I_drain_diff_u_s)
            J[self.drain_node_index][self.source_node_index].append(self.minus_I_gd_diff_u_s)

        if self.source_node_index != -1 and self.gate_node_index != -1:
            J[self.source_node_index][self.gate_node_index].append(self.minus_I_drain_diff_u_g)
            J[self.source_node_index][self.gate_node_index].append(self.minus_I_gs_diff_u_g)
            J[self.gate_node_index][self.source_node_index].append(self.I_gs_diff_u_s)
            J[self.gate_node_index][self.source_node_index].append(self.I_gd_diff_u_s)

        if self.drain_node_index != -1 and self.gate_node_index != -1:
            J[self.drain_node_index][self.gate_node_index].append(self.I_drain_diff_u_g)
            J[self.drain_node_index][self.gate_node_index].append(self.minus_I_gd_diff_u_g)
            J[self.gate_node_index][self.drain_node_index].append(self.I_gd_diff_u_d)

        # current
        if source_current_index != -1:
            A[source_current_index][source_current_index] += -1
            g[source_current_index].append(self.minus_I_drain)
            g[source_current_index].append(self.minus_I_gs)

        if self.source_node_index != -1 and source_current_index != -1:
            J[source_current_index][self.source_node_index].append(self.minus_I_drain_diff_u_s)
            J[source_current_index][self.source_node_index].append(self.minus_I_gs_diff_u_s)

        if drain_current_index != -1:
            A[drain_current_index][drain_current_index] += -1
            g[drain_current_index].append(self.I_drain)
            g[drain_current_index].append(self.minus_I_gd)

        if self.drain_node_index != -1 and drain_current_index != -1:
            J[drain_current_index][self.drain_node_index].append(self.I_drain_diff_u_d)
            J[drain_current_index][self.drain_node_index].append(self.minus_I_gd_diff_u_d)

        if gate_current_index != -1:
            A[gate_current_index][gate_current_index] += -1
            g[gate_current_index].append(self.I_gs)
            g[gate_current_index].append(self.I_gd)

        if self.gate_node_index != -1 and gate_current_index != -1:
            J[gate_current_index][self.gate_node_index].append(self.I_gs_diff_u_g)
            J[gate_current_index][self.gate_node_index].append(self.I_gd_diff_u_g)
