from __future__ import annotations

import copy
from math import sqrt, log10
import numpy as np
import os
import time

from utils import flatten, transpose
from representation import uRepresentation, URepresentation


def export1D(path, data, mode='w'):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, mode) as f:
        for element in data:
            f.write('%s\n' % element)


def export2D(path, data, labels=[], mode='w'):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, mode) as f:
        if labels and mode == 'w':
            for l in labels:
                f.write('%s ' % l)
            f.write('\n')

        for line in data:
            for element in line:
                f.write('%s ' % element)
            f.write('\n')


def import1D(path, type):
    elements = []
    with open(path) as f:
        for element in f:
            elements.append(type(element))

    return np.array(elements)


def import2D(path, with_label=True, mathematica=False):
    # TODO: numpy.loadtxt
    elementss = []
    labels = []
    with open(path) as f:
        lines = f.readlines()

        if with_label:
            labels = lines[0].split()
            lines = lines[1:]

        for line in lines:
            if mathematica:
                line = line[1:-1]
            
            elementss.append([float(x[:-1]) if mathematica else float(x) for x in line.split()])

    return labels, np.array(elementss)


def get_file_name(name, dictionary, preserve_order=False):
    def myStr(x):
        if type(x) is bool:
            x = int(x)
        
        if type(x) is float:
            x = round(x, 2)

        return str(x)

    out = name
    used_abbreviations = []

    dictionary_keys = dictionary.keys()
    if not preserve_order:
        dictionary_keys = sorted(dictionary)

    for x in dictionary_keys:
        keys = [x]
        for key in keys:
            if key not in used_abbreviations:
                out += key + myStr(dictionary[x])
                used_abbreviations.append(key)
                break
    return out


class QcoData:
    default_name_prefix = 'qco'

    file_sufixes = {'norms': '', 'gates': '-gates'}

    directory_prefix = './'


    def __init__(self, d, n_of_generators, options,
        representations: list[uRepresentation | URepresentation],
        rank, name=default_name_prefix, abs_val=False, preserve_order=False):
        self.d = d
        self.set_size = n_of_generators
        self.options = options
        self.abs_val = abs_val

        self.symmetric = bool(options['s']) if 's' in options else None

        set_size = 2 * n_of_generators if self.symmetric else n_of_generators
        self.optimal_norm = 2 * sqrt(set_size - 1) / set_size

        self.file_name = get_file_name(name, options, preserve_order)

        _options = copy.deepcopy(options)
        if 's' in options:
            del _options['s']
        self.gates_file_name = get_file_name(name, _options, preserve_order)

        self.directory = self.directory_prefix

        self.gates_directory = self.directory_prefix

        self.weights = [Pi.weight for Pi in representations]

        self.rank = rank # Rank of the process.

        # Eigenvalue with biggest absolute value by operator by representation.
        self.max_eigss = []
        self.norms = [] # Norm by operator.
        self.best_norm = float('inf') # Minimal norm.
        self.gatess = [] # Gate-set generators by operator.
        self.best_gates = [] # Best gate-set generators.
        # Operator norm up to i-th weight.
        # self.cum_max_eigs[i][j] = max(abs(self.max_eigs[i][:i])).
        self.cum_max_eigss = []


    @staticmethod
    def _max_eigs_to_norm(max_eigs):
        return max(abs(l) for l in max_eigs)


    def _max_eigs_to_cum_max_eigs(self, max_eigs):
        cum_max_eigs = []
        tmp_max = -1
        for i in range(len(max_eigs)):
            tmp_max = max(tmp_max, abs(max_eigs[i]))
            cum_max_eigs.append(tmp_max)

        return cum_max_eigs


    def _make_cum_max_eigss(self):
        return [self._max_eigs_to_cum_max_eigs(op) for op in self.max_eigss]


    def make_tables(self):
        self.max_eigss = np.array(self.max_eigss)
        self.cum_max_eigss = np.array(self._make_cum_max_eigss())


    def add_max_eigs(self, max_eigs, gates=None):
        if gates:
            self.gatess.append(gates)

        if self.abs_val:
            max_eigs = [abs(l) for l in max_eigs]
        self.max_eigss.append(max_eigs)

        norm = QcoData._max_eigs_to_norm(max_eigs) if max_eigs else 0
        self.norms.append(norm)

        if norm < self.best_norm:
            self.best_norm = norm
            self.best_gates = gates


    def __iadd__(self, data: QcoData):
        self.max_eigss += copy.copy(data.max_eigss)
        if data.gatess:
            self.gatess += copy.deepcopy(data.gatess)
        self.norms += copy.copy(data.norms)

        if self.best_norm > data.best_norm:
            self.best_norm = data.best_norm
            self.best_gates = copy.deepcopy(data.best_gates)

        return self


    def save(self, mode='w', dir='', name=''):
        directory = dir if dir else self.directory
        name = name if name else self.file_name

        flag = time.time()

        export2D(
            directory + name + self.file_sufixes['norms'] + '.txt', 
            self.max_eigss, self.weights, mode
        )

        return time.time() - flag

    
    def save_gates(self, mode='w', dir='', name=''):
        directory = dir if dir else self.gates_directory
        name = name if name else self.file_name

        flag = time.time()
        
        if self.gatess:
            with open(directory + name + self.file_sufixes['gates'] + '.txt', mode) as f:
                gatess = self.gatess
                for i, gates in enumerate(gatess):
                    f.write('-\n')
                    for gate in gates:
                        gate = gate.A
                        for line in gate:
                            for el in line:
                                f.write('%s ' % el)
                            f.write('\n')
        
        return time.time() - flag


    @staticmethod
    def load(d, G, options, name=default_name_prefix, dir='', file_name='',
        with_label=True, with_gates=False, norms_only=True, preserve_order=False,
        gates_dir=''):
        data = QcoData(d, G, options, [], 0, name, norms_only, preserve_order=preserve_order)
        directory = dir if dir else data.directory
        gates_directory = gates_dir if gates_dir else data.gates_directory
        file_name = file_name if file_name else data.file_name

        ops = []
        with open(directory + file_name + data.file_sufixes['norms'] + '.txt') as f:
            lines = f.readlines()

            if with_label:
                label = lines[0]
                lines = lines[1:]

                weight = []
                number = ''
                for x in label:
                    if x == '[' or x == ',':
                        continue
                    elif x == ']':
                        weight.append(int(number))
                        number = ''
                        data.weights.append(copy.copy(weight))
                        weight = []
                    elif x == ' ':
                        if number:
                            weight.append(int(number))
                            number = ''
                    else:
                        number += x

            for line in lines:
                ops.append([float(x) for x in line.split()])

        opgs = []
        if with_gates:
            gates = []
            gate = []
            j = 0
            with open(gates_directory + data.gates_file_name + data.file_sufixes['gates'] + '.txt') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()

                    if len(line) == 1:
                        j = i
                        if gates:
                            opgs.append(copy.deepcopy(gates))
                            gates = []
                    else:
                        gate.append([complex(x) for x in line])

                        if (i - j) % d == 0:
                            gates.append(copy.deepcopy(gate))
                            gate = []
                opgs.append(copy.deepcopy(gates))
        else:
            opgs = [[] for _ in ops]

        for i in range(len(ops)):
            data.add_max_eigs(ops[i], opgs[i])

        return data

    
    def erase(self):
        self.max_eigss = []
        self.norms = []
        self.gatess = []
        self.cum_max_eigss = []


    def open(self, mode, suffix='', extension='.txt'):
        return open(self.directory + self.file_name + suffix + extension, mode)
