from .degree import degree
from .loop import (contains_self_loops, remove_self_loops,
                   segregate_self_loops, add_self_loops,
                   add_remaining_self_loops)
from .get_laplacian import get_laplacian

__all__ = [
    'degree',
    'contains_self_loops',
    'remove_self_loops',
    'segregate_self_loops',
    'add_self_loops',
    'add_remaining_self_loops',
    'get_laplacian',
]

classes = __all__
