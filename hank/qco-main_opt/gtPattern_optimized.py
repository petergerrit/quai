"""
Optimized Gelfand-Tsetlin pattern implementation.

Key optimizations:
1. Pre-compute all patterns and indices upfront (avoids repeated deepcopy)
2. Use tuples (immutable, hashable) instead of lists
3. Direct index computation without string conversions
4. Validity checks without object creation
"""
from __future__ import annotations
import numpy as np
from typing import Iterable, Optional
from functools import lru_cache


class GTPatternOptimized:
    """Optimized Gelfand-Tsetlin pattern using pre-computed enumeration."""

    def __init__(self, top_row: Iterable[int]):
        self.top_row = tuple(top_row)
        self.n = len(self.top_row)
        
        # Pre-compute all valid patterns and their indices
        self._patterns: list[tuple] = []  # list of patterns as nested tuples
        self._pattern_to_idx: dict[tuple, int] = {}
        self._enumerate_all_patterns()
        
        self.dim = len(self._patterns)
        
    def _enumerate_all_patterns(self):
        """Enumerate all valid GT patterns with the given top row."""
        # A GT pattern has rows: row[n-1] = top_row, row[n-2], ..., row[0]
        # where row[k] has k+1 elements satisfying interlacing conditions
        
        def generate_rows(upper_row: tuple, level: int, acc: list[tuple]):
            """Generate all valid rows at given level, given the row above."""
            if level < 0:
                # Convert accumulated rows to a single hashable pattern
                pattern = tuple(acc)
                idx = len(self._patterns)
                self._patterns.append(pattern)
                self._pattern_to_idx[pattern] = idx
                return
            
            # Generate all valid rows at this level
            width = level + 1
            
            def gen_row(pos: int, current: list[int]):
                if pos == width:
                    row = tuple(current)
                    generate_rows(row, level - 1, acc + [row])
                    return
                
                # Constraints: upper_row[pos] >= current[pos] >= upper_row[pos+1]
                max_val = upper_row[pos]
                min_val = upper_row[pos + 1]
                
                # Also: current[pos] >= current[pos-1] (decreasing within row)
                if pos > 0:
                    min_val = max(min_val, current[pos - 1])
                    
                # Wait, GT patterns have: m[k][j] >= m[k-1][j] >= m[k][j+1]
                # So if upper_row is row k+1, and we're generating row k:
                # m[k+1][j] >= m[k][j] >= m[k+1][j+1]
                
                for val in range(min_val, max_val + 1):
                    gen_row(pos + 1, current + [val])
            
            gen_row(0, [])
        
        # Start from the top row
        generate_rows(self.top_row, self.n - 2, [self.top_row])
    
    def get_pattern(self, idx: int) -> tuple:
        """Get pattern by index."""
        return self._patterns[idx]
    
    def get_index(self, pattern: tuple) -> int:
        """Get index of a pattern."""
        return self._pattern_to_idx[pattern]
    
    def pattern_add_one(self, pattern: tuple, level: int, pos: int) -> Optional[tuple]:
        """
        Try to add 1 to position (level, pos) in pattern.
        Returns new pattern if valid, None otherwise.
        
        Pattern is a tuple of rows: (row_n-1, row_n-2, ..., row_0)
        where row_k has k+1 elements.
        Level 0 means the bottom row, level n-2 is the row just below top.
        """
        # In the original code, rows are indexed from 0 (bottom) to n-1 (top)
        # level corresponds to row index in the original scheme
        # The pattern tuple here is (top_row, row_n-2, row_n-3, ..., row_0)
        
        # Convert to correct indexing
        row_idx = self.n - 1 - level  # index in our tuple
        
        if row_idx <= 0 or row_idx >= self.n:
            return None  # Can't modify top row
        
        if pos < 0 or pos > level:
            return None
            
        row = pattern[row_idx]
        new_val = row[pos] + 1
        
        # Check upper constraint: new_val <= upper_row[pos]
        upper_row = pattern[row_idx - 1]
        if new_val > upper_row[pos]:
            return None
        
        # Check lower constraint (if there is a lower row)
        if row_idx < self.n - 1:
            lower_row = pattern[row_idx + 1]
            # Check: new_val >= lower_row[pos-1] (if pos > 0)
            if pos > 0 and new_val > lower_row[pos - 1]:
                return None
        
        # Create new pattern
        new_row = row[:pos] + (new_val,) + row[pos + 1:]
        new_pattern = pattern[:row_idx] + (new_row,) + pattern[row_idx + 1:]
        
        # Verify it's a valid pattern
        if new_pattern in self._pattern_to_idx:
            return new_pattern
        return None
    
    def pattern_subtract_one(self, pattern: tuple, level: int, pos: int) -> Optional[tuple]:
        """
        Try to subtract 1 from position (level, pos) in pattern.
        Returns new pattern if valid, None otherwise.
        """
        row_idx = self.n - 1 - level
        
        if row_idx <= 0 or row_idx >= self.n:
            return None
        
        if pos < 0 or pos > level:
            return None
            
        row = pattern[row_idx]
        new_val = row[pos] - 1
        
        # Check lower constraint: new_val >= upper_row[pos+1]
        upper_row = pattern[row_idx - 1]
        if new_val < upper_row[pos + 1]:
            return None
        
        # Check constraint with lower row
        if row_idx < self.n - 1:
            lower_row = pattern[row_idx + 1]
            if pos < len(lower_row) and new_val < lower_row[pos]:
                return None
        
        # Create new pattern
        new_row = row[:pos] + (new_val,) + row[pos + 1:]
        new_pattern = pattern[:row_idx] + (new_row,) + pattern[row_idx + 1:]
        
        if new_pattern in self._pattern_to_idx:
            return new_pattern
        return None


def precompute_gt_data(top_row: tuple) -> dict:
    """
    Pre-compute all GT pattern data needed for representation construction.
    
    Returns a dictionary with:
    - 'patterns': list of all patterns
    - 'dim': dimension (number of patterns)
    - 'transitions_add': dict mapping (pattern_idx, level, pos) -> new_pattern_idx or -1
    - 'transitions_sub': dict mapping (pattern_idx, level, pos) -> new_pattern_idx or -1
    """
    n = len(top_row)
    
    # Enumerate all patterns
    patterns = []
    pattern_to_idx = {}
    
    def enumerate_patterns():
        def generate(upper_row: tuple, level: int, acc: list[tuple]):
            if level < 0:
                pattern = tuple(acc)
                idx = len(patterns)
                patterns.append(pattern)
                pattern_to_idx[pattern] = idx
                return
            
            width = level + 1
            
            def gen_row(pos: int, current: list[int]):
                if pos == width:
                    generate(tuple(current), level - 1, acc + [tuple(current)])
                    return
                
                max_val = upper_row[pos]
                min_val = upper_row[pos + 1] if pos + 1 < len(upper_row) else 0
                
                for val in range(max_val, min_val - 1, -1):
                    gen_row(pos + 1, current + [val])
            
            gen_row(0, [])
        
        generate(top_row, n - 2, [top_row])
    
    enumerate_patterns()
    dim = len(patterns)
    
    # Pre-compute all transitions
    # transitions_add[idx, level, pos] = new_idx or -1
    # We'll store as numpy array for efficiency
    
    # Max level is n-2 (row just below top), max pos at each level is level
    max_level = n - 1  # levels 0 to n-2 are modifiable
    
    transitions_add = {}
    transitions_sub = {}
    
    for idx, pattern in enumerate(patterns):
        for level in range(n - 1):  # level is row index (0 = bottom)
            row_tuple_idx = n - 1 - level  # index in pattern tuple
            if row_tuple_idx <= 0:
                continue
            row = pattern[row_tuple_idx]
            upper_row = pattern[row_tuple_idx - 1]
            lower_row = pattern[row_tuple_idx + 1] if row_tuple_idx + 1 < n else None
            
            for pos in range(level + 1):
                # Try add_one
                new_val = row[pos] + 1
                valid_add = True
                if new_val > upper_row[pos]:
                    valid_add = False
                # The interlacing condition also requires checking lower row
                if valid_add and lower_row is not None and pos > 0:
                    if new_val > lower_row[pos - 1]:
                        valid_add = False
                
                if valid_add:
                    new_row = row[:pos] + (new_val,) + row[pos + 1:]
                    new_pattern = pattern[:row_tuple_idx] + (new_row,) + pattern[row_tuple_idx + 1:]
                    new_idx = pattern_to_idx.get(new_pattern, -1)
                    transitions_add[(idx, level, pos)] = new_idx
                else:
                    transitions_add[(idx, level, pos)] = -1
                
                # Try subtract_one
                new_val = row[pos] - 1
                valid_sub = True
                if pos + 1 < len(upper_row) and new_val < upper_row[pos + 1]:
                    valid_sub = False
                if valid_sub and lower_row is not None and pos < len(lower_row):
                    if new_val < lower_row[pos]:
                        valid_sub = False
                
                if valid_sub:
                    new_row = row[:pos] + (new_val,) + row[pos + 1:]
                    new_pattern = pattern[:row_tuple_idx] + (new_row,) + pattern[row_tuple_idx + 1:]
                    new_idx = pattern_to_idx.get(new_pattern, -1)
                    transitions_sub[(idx, level, pos)] = new_idx
                else:
                    transitions_sub[(idx, level, pos)] = -1
    
    return {
        'patterns': patterns,
        'pattern_to_idx': pattern_to_idx,
        'dim': dim,
        'n': n,
        'top_row': top_row,
        'transitions_add': transitions_add,
        'transitions_sub': transitions_sub,
    }
