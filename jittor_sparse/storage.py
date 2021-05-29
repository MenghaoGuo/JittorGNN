#!/usr/bin/env python
# coding: utf-8

import jittor as jt
from typing import Optional, Tuple

class SparseStorage:
    def __init__(self, row: Optional[jt.Var] = None,
            rowptr: Optional[jt.Var] = None,
            col: Optional[jt.Var] = None,
            value: Optional[jt.Var] = None,
            sparse_sizes: Optional[Tuple[int, int]] = None,
            is_sorted: bool = False):
        self._row = row
        self._rowptr = rowptr
        self._col = col
        self._value = value
        self._sparse_sizes = sparse_sizes
        self._is_sorted = is_sorted

    def row(self):
        if self._row is not None:
            return self._row

        if self._rowptr is None:
            raise ValueError

        self._row = jt.code([self._rowptr[-1].item()], self._rowptr.dtype, [self._rowptr],
            cpu_src='''
                int p=0, size=@in0(in0_shape0-1);
                for (int i=0; i<size; i++){
                    while(i>=@in0(p+1)) p++;
                    @out(i)=p;
                }
            ''')
        return self._row

    def rowptr(self):
        return self._rowptr

    def col(self):
        return self._col

    def value(self):
        return self._value

    def sparse_sizes(self):
        return self._sparse_sizes

def gather_csr(src, rowptr):
    return jt.code([rowptr[-1].item()], src.dtype, [src, rowptr],
            cpu_src='''
                int p=0, size=@in1(in1_shape0-1);
                for (int i=0; i<size; i++){
                    while(i>=@in1(p+1)) p++;
                    @out(i)=@in0(p);
                }
            ''')
