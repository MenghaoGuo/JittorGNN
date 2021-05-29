#!/usr/bin/env python
# coding: utf-8

import jittor as jt
from typing import Optional, Tuple
from jittor import Var
from jittor_sparse.var import SparseVar

def spmm_sum(mat1: SparseVar, x: Var) -> Var:
    row = mat1.storage.row()
    col = mat1.storage.col()
    value = mat1.storage.value()

    mid = jt.transpose((jt.transpose(jt.gather(x, 0, col)) * value))
    scatter_index = jt.transpose(jt.broadcast(row, shape = (mid.shape[1], row.shape[0])))
    output = jt.zeros((mat1.sparse_size(0), x.shape[1])).scatter_(0, scatter_index, mid, reduce = 'add')
    return output

def spmm(mat1: SparseVar, x: Var, reduce="sum") -> Var:
    if reduce == "sum":
        return spmm_sum(mat1, x)
    else:
        raise NotImplementedError

def matmul(mat1, x, reduce="sum"):
    '''
    Perform a matrix multiplication of the 2-D csr_sparse matrix input and the dense matrix mat2
    If input is a (n x m) sparse matrix, mat2 is a (m x p) dense matrix, it will return a (n x p) dense matrix

Parameters:
    * mat1 (SparseVar) - input sparse matrix
    * mat2 (Var) - input dense matrix

Example:
    rowptr = jt.array([0, 2, 4, 4])
    col = jt.array([0, 1, 0, 2])
    value = jt.array([1, 1, 2, 1])

    a = SparseVar(rowptr = rowptr, col = col, value = value, sparse_sizes = (2, 3))
    b = jt.array([[1., 2.],
                  [3., 4.],
                  [5., 6.]])
    data = matmul(a, b)
    assert(data.data == [[4., 6.],
                         [7., 10.]]).all()
    '''
    if isinstance(x, Var):
        return spmm(mat1, x, reduce)
    elif isinstance(x, SparseVar):
        raise NotImplementedError
    else:
        raise ValueError
