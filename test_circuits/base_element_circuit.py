import sympy as smp

from analysis_of_transient_processes.circuit_elements import Resistance, \
                                                             Inductance, \
                                                             Capacitance, \
                                                             VoltageSource

from analysis_of_transient_processes.electrical_circuits import ElectricalCircuit
from analysis_of_transient_processes.calculation_of_equations import calculate_solution
from analysis_of_transient_processes.visualization import visualize_solution


def base_element_circuit(start_t=0, end_t=1):
    circuit = ElectricalCircuit()

    circuit.add_element(VoltageSource(lambda t: smp.sin(2 * smp.pi * t)), [0, 1], [False])
    circuit.add_element(Resistance(100), [1, 2], [False])
    circuit.add_element(Resistance(50), [2, 3], [False])
    circuit.add_element(Capacitance(1e-12), [2, 0], [False])
    circuit.add_element(Inductance(1e-9), [0, 3], [False])

    circuit.print_elements()
    circuit.print_variables()

    initial_values = None

    t, y = calculate_solution(circuit, start_t, end_t, initial_values)

    solution_path = 'test_circuits_solutions/base_element_circuit/'

    for i in range(len(y)):
        visualize_solution(t, y[i], circuit.variables[i], t.size - 1, 'Solution ' + circuit.variables[i], solution_path)


if __name__ == '__main__':
    base_element_circuit()
    print('The transient analysis has been completed.')
