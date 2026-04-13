from math import sqrt




pauli_x = [[0, 1], [1, 0]]
pauli_y = [[0, -1j], [1j, 0]]
pauli_z = [[1, 0], [0, -1]]
paulis = [pauli_x, pauli_y, pauli_z]

H_gate = Hadamard_gate = [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]]
S_gate = phase_gate = [[1, 0], [0, 1j]]
T_gate = [[1, 0], [0, 1 / sqrt(2) + 1j / sqrt(2)]]
cNOT_gate = cNOT = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]

DATA_PATH = './data/'

PHI = GOLDEN_RATIO = (1 + sqrt(5)) / 2
