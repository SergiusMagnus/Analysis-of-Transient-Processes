import sympy as smp

from transient_analysis.circuit_elements import Resistance, \
                                                             Inductance, \
                                                             Capacitance, \
                                                             VoltageSource, \
                                                             NJFET

from transient_analysis.electrical_circuits import ElectricalCircuit
from transient_analysis.calculation_of_equations import calculate_solution
from transient_analysis.visualization import visualize_solution


def sel(start_t=0, end_t=1):
    circuit = ElectricalCircuit()

    circuit.add_element(VoltageSource(lambda t: smp.sin(2 * smp.pi * t)), [0, 11], [True])
    circuit.add_element(Capacitance(47e-9), [11, 0], [False])
    circuit.add_element(Resistance(50), [11, 9], [False])
    circuit.add_element(Capacitance(47e-9), [9, 0], [False])
    circuit.add_element(Resistance(1000), [9, 5], [False])
    circuit.add_element(Inductance(0.2e-6), [5, 9], [False])
    circuit.add_element(Capacitance(13e-12), [9, 5], [False])
    circuit.add_element(Resistance(50), [9, 4], [False])
    circuit.add_element(Capacitance(47e-9), [4, 0], [False])
    circuit.add_element(Resistance(1000), [4, 1], [False])
    circuit.add_element(Inductance(0.2e-6), [1, 4], [False])
    circuit.add_element(Capacitance(13e-12), [4, 1], [False])

    circuit.add_element(NJFET(), [2, 1, 14], [False, False, False])
    circuit.add_element(Capacitance(47e-9), [1, 10], [False])
    circuit.add_element(NJFET(), [6, 5, 10], [False, False, False])
    circuit.add_element(Capacitance(47e-9), [5, 50], [False])

    circuit.add_element(Capacitance(47e-9), [21, 14], [False])

    circuit.add_element(VoltageSource(lambda t: t * 0), [0, 21], [False])
    circuit.add_element(Capacitance(13e-12), [14, 0], [False])
    circuit.add_element(Inductance(0.2e-6), [0, 14], [False])
    circuit.add_element(Resistance(1000), [14, 0], [False])
    circuit.add_element(Resistance(1e5), [14, 0], [False])
    circuit.add_element(Capacitance(47e-9), [2, 0], [False])
    circuit.add_element(Resistance(150), [2, 0], [False])
    circuit.add_element(Resistance(1e5), [10, 0], [False])
    circuit.add_element(Capacitance(47e-9), [6, 0], [False])
    circuit.add_element(Resistance(150), [6, 0], [False])
    circuit.add_element(Resistance(1e6), [50, 0], [False])

    circuit.print_elements()
    circuit.print_variables()

    initial_values = None

    t, y = calculate_solution(circuit, start_t, end_t, initial_values)

    solution_path = 'test_circuits_solutions/sel/'

    for i in range(len(y)):
        visualize_solution(t, y[i], circuit.variables[i], t.size - 1, 'Solution ' + circuit.variables[i], solution_path)


if __name__ == '__main__':
    sel()
    print('The transient analysis has been completed.')
