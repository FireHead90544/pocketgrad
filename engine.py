from __future__ import annotations
from typing import Union, List, Tuple, Set, Callable

class Tensor:
    grad: float
    _backward: Callable[[], None]
    _dependencies: Set[Tensor]

    def __init__(self, data: Union[int, float], _children: Tuple[Tensor, ...] = (), _op: str = '') -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._dependencies = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

if __name__ == "__main__":
    pass
