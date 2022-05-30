import numpy as np


class ElectricalCircuit:
    """Electrical Circuit"""

    def __init__(self):
        self.elements_count = 0
        self.elements = []
        self.variables = []
        self.nodes_and_variables = [(0, -1)]
        self.voltage_number = 0
        self.current_number = 0
        self.variables_number = 0
        self.elements_and_variables = []

    def add_variables(self, element):
        def check_node_exist(num):
            for node in self.nodes_and_variables:
                if node[0] == num:
                    return False
            return True

        for node in element['nodes']:
            if node != 0 and check_node_exist(node):
                self.voltage_number += 1
                self.variables.append('u' + str(self.voltage_number))
                self.nodes_and_variables.append((node, self.variables_number))
                self.variables_number += 1

        for current in element['calculate_current']:
            if current:
                self.current_number += 1
                self.variables.append('i' + str(self.current_number))
                self.elements_and_variables.append((self.elements_count - 1, self.variables_number))
                self.variables_number += 1
            else:
                self.elements_and_variables.append((self.elements_count - 1, -1))

    def add_element(self, circuit_element, nodes, calculate_current):
        if circuit_element.__class__.__name__ in ['VoltageSource', 'Inductance']:
            calculate_current = [True]

        self.elements.append({'element_id': self.elements_count,
                              'element': circuit_element,
                              'nodes': nodes,
                              'calculate_current': calculate_current})
        self.elements_count += 1
        self.add_variables(self.elements[-1])

    def print_elements(self):
        for element in self.elements:
            print(element)

    def print_variables(self):
        print(self.variables)
        print(self.nodes_and_variables)
        print(self.elements_and_variables)

    def get_nodes_index(self, element):
        nodes_index = []

        for node in element['nodes']:
            for node_and_variable in self.nodes_and_variables:
                if node == node_and_variable[0]:
                    nodes_index.append(node_and_variable[1])
                    break

        return nodes_index

    def get_current_index(self, element):
        current_index = []

        for i in self.elements_and_variables:
            if element['element_id'] == i[0]:
                current_index.append(i[1])

        if len(current_index) == 1:
            return current_index[0]
        else:
            return current_index

    def get_matrices_of_electric_circuit(self):
        B = [[0 for _ in range(len(self.variables))] for _ in range(len(self.variables))]
        A = [[0 for _ in range(len(self.variables))] for _ in range(len(self.variables))]
        J = [[[] for _ in range(len(self.variables))] for _ in range(len(self.variables))]
        g = [[] for _ in range(len(self.variables))]
        f = [[] for _ in range(len(self.variables))]
        df_dt = [[] for _ in range(len(self.variables))]

        for element in self.elements:
            nodes_index = self.get_nodes_index(element)
            current_index = self.get_current_index(element)

            element['element'].form_matrices(B, A, J, g, f, df_dt, nodes_index, current_index)

        B = np.array(B)
        A = np.array(A)

        return B, A, J, g, f, df_dt
