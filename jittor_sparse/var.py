#!/usr/bin/env python
# coding: utf-8

import jittor as jt
from typing import Optional, Tuple
from jittor import Var
from jittor_sparse.storage import SparseStorage

class SparseVar:
    storage : SparseStorage
    def __init__(self, row: Optional[Var] = None,
            rowptr: Optional[Var] = None,
            col: Optional[Var] = None,
            value: Optional[Var] = None,
            sparse_sizes: Optional[Tuple[int, int]] = None,
            is_sorted: bool = False):
        self.storage = SparseStorage(row=row, rowptr=rowptr, col=col,
                value=value, sparse_sizes=sparse_sizes,
                is_sorted=is_sorted)

    def sparse_size(self, idx):
        return self.storage._sparse_sizes[idx]

