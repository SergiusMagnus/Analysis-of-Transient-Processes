# Transient Analysis

The transient analysis is implemented using the Rosenbrock method. Based on the sequence of elements of the electrical circuit, and to which nodes they are connected, and whether it is required to count the current passing through the corresponding element, matrices and vectors are formed that define a system of differential equations that simulates an electrical circuit. Then, using the 6-stage stiffly accurate embedded Rosenbrock method of order 4(3), a numerical solution is found. The solutions are then visualized.

