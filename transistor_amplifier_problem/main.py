from calculation_of_equations import calculate_solution
from initial_data import M, DAE, initial_values
from visualization import visualize_solution

if __name__ == '__main__':
    t, y = calculate_solution(M, DAE, 0, 0.2, initial_values, 1e-4)
    visualize_solution(t, y[4], 'U5', t.size - 1, 'Solution U5', 'transistor_amplifier_problem/')
