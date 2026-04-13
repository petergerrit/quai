"""Module with Gelfand-Tsetlin patterns.

Classes:
    GTPattern: class of G-T patterns.
"""
from __future__ import annotations

import copy
from typing import Iterable




class GTPattern:
    """Class of Gelfand-Tsetlin patterns.
    Gelfand-Tsetlin pattern is an array of nonnnegative integers:

    m[n-1][0]   m[n-1][1]   ...   m[n-1][n-2]   m[n-1][n-1]
          m[n-2][0]         ...         m[n-2][n-2]
                            ...
                     m[1][0]   m[1][1]
                          m[0][0]

    such that m[i][j] >= m[i-1][j] >= m[i][j+1].

    Class is implemented with linear order of patterns.
    m > l iff their concatenated rows hold:
    m[n-1]...m[0] > l[n-1]...l[0] in lexicographic order.

    Args:
        top_row (Iterable[int]): Row at the top of the pattern.
    """


    def __init__(self, top_row: Iterable[int]):
        self.n_of_rows: int = len(top_row) # Number of rows.
        self.top_row: tuple[int, ...] = tuple(top_row)
        self.rows: list[list[int]] = [] # TODO: Make it immutable.

        # Answer to the question "Do inequalities between pattern's elements hold?".
        self.correct: bool = True

        # Dictionary of indices. See comment to getIndex() method below.
        self.indices: dict[str, int] = {}


    def __str__(self) -> str:
        out = ""
        for i in range(self.n_of_rows - 1, -1, -1):
            for j in range(self.n_of_rows - 1 - i):
                out += " "

            for j in range(i + 1):
                out += str(self.rows[i][j]) + " "

            out += "\n"

        return out[:-1]


    def __eq__(self, other: GTPattern) -> bool:
        if self.n_of_rows != other.n_of_rows:
            return False

        for i in range(self.n_of_rows):
            for j in range(i + 1):
                if self[i][j] != other[i][j]:
                    return False

        return True


    def set_to_highest(self):
        """Set all entries of the pattern to be the biggest possible.
        """
        self.rows = []
        self.rows.append(self.top_row)

        for i in range(1, self.n_of_rows):
            self.rows.insert(0, [])

            for j in range(self.n_of_rows-i):
                self.rows[0].append(self.rows[1][j])


    def __getitem__(self, key: int | slice) -> list[list[int]] | list[int] | int:
        return self.rows[key]


    def add_one(self, i: int, j: int) -> GTPattern:
        """Add one to the j-th element in the i-th row, then checks if the new pattern is
        correct and stores this information in its correct attribute.

        Args:
            i (int): Row number.
            j (int): Column number.

        Returns:
            GTPattern: New pattern, copy of self with m[i][j] + 1 instead of m[i][j].
        """
        out = copy.deepcopy(self)

        if i >= out.n_of_rows or i < j:
            out.correct = False # TODO: Raise ValueError.
            return out

        out.rows[i][j] += 1

        if out.rows[i][j] > out.rows[i + 1][j]:
            out.correct = False

        if  i > 0 and j > 0 and out.rows[i][j] > out.rows[i - 1][j - 1]:
            out.correct = False

        return out


    def subtract_one(self, i: int, j: int) -> GTPattern:
        """Subtract one to the j-th element in the i-th row, then checks if the new pattern is
        correct and stores this information in its correct attribute.

        Args:
            i (int): Row number.
            j (int): Column number.

        Returns:
            GTPattern: New pattern, copy of self with m[i][j] - 1 instead of m[i][j].
        """
        out = copy.deepcopy(self)

        if i >= out.n_of_rows or i < j:
            out.correct = False # TODO: Raise ValueError.
            return out

        out.rows[i][j] -= 1

        if out.rows[i][j] < out.rows[i + 1][j + 1]:
            out.correct = False

        if i > 0 and i > j and out.rows[i][j] < out.rows[i - 1][j]:
            out.correct = False

        return out


    def one_smaller(self) -> bool:
        """Transform pattern into the biggest possible smaller one i.e. one smaller.

        Returns:
            bool: False if pattern is the smallest possible and True otherwise.
        """
        i, j = 0, 0
        changed = False

        while i < self.n_of_rows - 1:
            if self.rows[i][j] - 1 >= self.rows[i + 1][j + 1]:
                self.rows[i][j] -= 1
                changed = True
                break

            if j > 0:
                j -= 1
            else:
                i += 1
                j = i

        if not changed:
            return False

        if j < i:
            j += 1
        else:
            i -= 1
            j = 0

        while i >= 0:
            while j <= i:
                self.rows[i][j] = self.rows[i + 1][j]
                j += 1

            i -= 1
            j = 0

        return True


    def one_bigger(self) -> bool:
        """Transform pattern into the smallest possible bigger one i.e. one bigger.

        Returns:
            bool: False if pattern is biggest possible and True otherwise.
        """
        i, j = 0, 0
        changed = False

        while i < self.n_of_rows - 1:
            if self.rows[i][j] + 1 <= self.rows[i + 1][j]:
                self.rows[i][j] += 1
                changed = True
                break

            if j > 0:
                j -= 1
            else:
                i += 1
                j = i

        if not changed:
            return False

        if j < i:
            j += 1
        else:
            i -= 1
            j = 0

        while i >= 0:
            while j <= i:
                self.rows[i][j] = self.rows[i + 1][j + 1]
                j += 1

            i -= 1
            j = 0

        return True


    def get_index(self) -> int:
        """Get index of the pattern i.e. quantity of smaller patterns.

        Returns:
            int: index
        """
        # TODO: implement __hash__ and simplify this method.
        if str(self) in self.indices:
            return self.indices[str(self)]

        tmp = copy.deepcopy(self)
        out: int = 0
        last: int = 0

        while str(tmp) not in self.indices and tmp.one_smaller():
            out += 1

        if str(tmp) in self.indices:
            last = self.indices[str(tmp)]
            out += last

        for i in range(last, out + 1):
            self.indices[str(tmp)] = i
            tmp.one_bigger()

        return out


    def get_all_rows(self, i: int) -> list[tuple[int, ...]]:
        self.set_to_highest()

        if 0 > i or i >= self.n_of_rows - 1:
            raise ValueError('Row index should be gigger than 0 and smaller than number of rows'\
                             f' minus on which is equal: {self.n_of_rows - 1} but {i} was given.')

        last_upper_row = tuple(self[i + 1])
        last_row = tuple(self[i])
        rows = [last_row]
        while self.one_smaller():
            row = tuple(self[i])
            upper_row = tuple(self[i + 1])
            if upper_row != last_upper_row or row != last_row:
                last_upper_row = upper_row
                last_row = row
                rows.append(row)

        return rows
