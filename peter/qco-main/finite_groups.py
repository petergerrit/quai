from __future__ import annotations

from math import sqrt

import numpy as np

from consts import DATA_PATH, PHI
from generate_group import from_generators
from scripts import get_gates_from_file as gates_from_file
from utils import hc




class FiniteGroup():
    def __init__(self, name: str, generators: list[np.ndarray],
        super_golden: np.ndarray, size: int, file_name: str, code: str,
        alternative_names: list[str] | None = None):
        self.name = name

        self.generators: list[np.ndarray] = [
            g / np.sqrt(np.linalg.det(g)+0j) for g in generators
        ]
        self.generators += [hc(g) for g in self.generators]

        self.super_golden = super_golden
        self.super_golden /= np.sqrt(np.linalg.det(super_golden))

        self.size = size
        self.file_name = file_name
        self.code = code

        self._elements: list[np.ndarray] | None = None

        an = alternative_names
        self.alt_names = [] if an is None else an


    def load(self, path: str | None = None):
        if path is None:
            path = f'{DATA_PATH}generated_groups\\{self.file_name}.txt'
        _, self._elements = gates_from_file(path)


    def generate(self, path: str | None = None):
        if path is None:
            path = f'{DATA_PATH}generated_groups\\{self.file_name}.txt'

        elements = from_generators(self.generators)

        if len(elements) != self.size:
            raise RuntimeError(
                f'Generated group has wrong size: {len(elements)}. '\
                f'Expected: {self.size}.'
            )

        self._elements = elements

        with open(path, 'w') as f:
            f.write('2\n')
            for g in elements:
                for line in g:
                    for x in line:
                        f.write(f'{x} ')
                    f.write('\n')


    @property
    def elements(self) -> list[np.ndarray]:
        if self._elements is not None:
            return self._elements

        try:
            self.load()
            el = self._elements
        except FileNotFoundError:
            self.generate()
            el = self._elements
        return el


    def __str__(self) -> str:
        return self.name


    def __hash__(self) -> int:
        return str.__hash__(str(self))


    def __iter__(self):
        return iter(self.elements)


    def __getitem__(self, i) -> np.ndarray:
        return self.elements[i]


    def __len__(self) -> int:
        return self.size




CLIFFORD = FiniteGroup(
    name='Clifford', size=24, file_name='cliff_2', code='cliff',
    generators = [
        np.array([
            [1, 0],
            [0, 1j]
        ]),
        np.array([
            [1, 1],
            [-1, 1]
        ])
    ],
    super_golden=np.array([
        [-1-np.sqrt(2), 2-np.sqrt(2)+1j],
        [2-np.sqrt(2)-1j, 1+np.sqrt(2)]
    ]),
    alternative_names=['C24', 'octahedral', 'symmetric 4']
)

PAULI = FiniteGroup(
    name='Pauli', size=4, file_name='pauli', code='pauli',
    generators = [
        np.array([
            [1j, 0],
            [0, -1j]
        ]),
        np.array([
            [0, 1],
            [-1, 0]
        ])
    ],
    super_golden=np.array([
        [1, 1-1j],
        [1+1j, -1]
    ]),
    alternative_names=['C4', 'quaternion 8']
)

MIN_CLIFF = FiniteGroup(
    name='minimal Clifford', size=3, file_name='min_cliff',
    code='min_cliff',
    generators = [
        np.array([
            [1, 0],
            [0, 1]
        ]),
        np.array([
            [1, 1],
            [1j, -1j]
        ]),
        np.array([
            [1, -1j],
            [1, 1j]
        ])
    ],
    super_golden=np.array([
        [0, np.sqrt(2)],
        [1+1j, 0]
    ]),
    alternative_names=['C3', 'cyclic 3']
)

HURWITZ = FiniteGroup(
    name='Hurwitz', size=12, file_name='hurwitz', code='hurwitz',
    generators = [
        np.array([
            [1j, 0],
            [0, -1j]
        ]),
        np.array([
            [1, 1],
            [1j, -1j]
        ])
    ],
    super_golden=np.array([
        [3, 1-1j],
        [1+1j, -3]
    ]),
    alternative_names=['C12', 'alternating 4']
)

KLEIN_ICO = FiniteGroup(
    name="Klein's Icosahedral", size=60, file_name='klein_ico',
    code='klein_ico',
    generators = [
        np.array([
            [1, 1],
            [1j, -1j]
        ]),
        np.array([
            [1, PHI - 1j/PHI],
            [PHI + 1j/PHI, -1]
        ])
    ],
    super_golden=np.array([
        [2 + PHI, 1 - 1j],
        [1 + 1j, -2 - PHI]
    ]),
    alternative_names=['C60']
)

DIH4 = FiniteGroup(
    name='dihedral 4', size=8, file_name='dih4', code='dih4',
    generators = [
        np.array([
            [0, 1],
            [1, 0]
        ]),
        np.array([
            [1, 0],
            [0, 1j]
        ])
    ],
    super_golden=np.array([
        [sqrt(2)-1, 1-sqrt(2)*1j],
        [1+sqrt(2)*1j, 1-sqrt(2)]
    ]),
    alternative_names=['C8']
)

HYBRID_V = FiniteGroup(
    name='hybrid V-gates', size=6, file_name='hybV', code='hybV',
    generators = [
        np.array([
            [1, 1],
            [1j, -1j]
        ]),
        np.array([
            [0, 1j],
            [1, 0]
        ])
    ],
    super_golden=np.array([
        [0, 2-1j],
        [2+1j, 0]
    ]),
    alternative_names=['C6', 'symmetric 3']
)

ALT4 = FiniteGroup(
    name='alternating 4', size=12, file_name='alt4', code='alt4',
    generators = [
        np.array([
            [1, 1],
            [1j, -1j]
        ]),
        np.array([
            [1, PHI+1j/PHI],
            [PHI-1j/PHI, -1]
        ])
    ],
    super_golden=np.array([
        [PHI-1, 1-1j],
        [1+1j, 1-PHI]
    ]),
    alternative_names=["C12'", 'Icosahedral 12']
)

CYC5 = FiniteGroup(
    name='cyclic 5', size=5, file_name='cyc5', code='cyc5',
    generators = [
        np.array([
            [1+PHI+1j, PHI],
            [-PHI, 1+PHI-1j]
        ])
    ],
    super_golden=np.array([
        [0, 1],
        [1j, 0]
    ]),
    alternative_names=['C5']
)
