class Capacitance:
    """Capacitance"""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'{self.__class__.__name__}(farad={self.value})'

    def form_matrices(self, B, A, J, g, f, df_dt, nodes_index, current_index):
        output_node_index, input_node_index = nodes_index

        # voltage
        if output_node_index != -1 and input_node_index != -1:
            B[output_node_index][input_node_index] += -self.value
            B[input_node_index][output_node_index] += -self.value

        if output_node_index != -1:
            B[output_node_index][output_node_index] += self.value

        if input_node_index != -1:
            B[input_node_index][input_node_index] += self.value

        # current
        if current_index != -1:
            A[current_index][current_index] += 1

        if output_node_index != -1 and current_index != -1:
            B[current_index][output_node_index] += self.value

        if input_node_index != -1 and current_index != -1:
            B[current_index][input_node_index] += -self.value
