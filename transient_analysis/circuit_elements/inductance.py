class Inductance:
    """Inductance"""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'{self.__class__.__name__}(henry={self.value})'

    def form_matrices(self, B, A, J, g, f, df_dt, nodes_index, current_index):
        output_node_index, input_node_index = nodes_index

        # voltage
        if input_node_index != -1:
            A[input_node_index][current_index] += -1

        if output_node_index != -1:
            A[output_node_index][current_index] += 1

        # current
        B[current_index][current_index] += -self.value

        if input_node_index != -1:
            A[current_index][input_node_index] += -1

        if output_node_index != -1:
            A[current_index][output_node_index] += 1
