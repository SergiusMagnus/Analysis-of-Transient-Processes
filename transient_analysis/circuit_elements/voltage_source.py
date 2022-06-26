import sympy as smp


class VoltageSource:
    """Voltage Source"""

    def __init__(self, value):
        self.value = value
        self.smb_t = smp.symbols('t')
        self.value_diff = self.signal(self.smb_t).diff(self.smb_t)

    def __repr__(self):
        return f'{self.__class__.__name__}(signal={self.value(self.smb_t)})'

    def signal(self, t):
        return -self.value(t)

    def signal_diff(self, t):
        signal_diff = self.value_diff.subs(self.smb_t, t)
        return -signal_diff

    def form_matrices(self, B, A, J, g, f, df_dt, nodes_index, current_index):
        output_node_index, input_node_index = nodes_index

        # voltage
        if input_node_index != -1:
            A[input_node_index][current_index] += 1

        if output_node_index != -1:
            A[output_node_index][current_index] += -1

        # current
        f[current_index].append(self.signal)
        df_dt[current_index].append(self.signal_diff)

        if input_node_index != -1:
            A[current_index][input_node_index] += 1

        if output_node_index != -1:
            A[current_index][output_node_index] += -1
