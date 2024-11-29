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

    def __add__(self, other: Union[int, float, Tensor]) -> Tensor: # self + other
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for +: 'Tensor' and '{type(other).__name__}'")

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Union[int, float, Tensor]) -> Tensor: # other + self
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for +: '{type(other).__name__}' and 'Tensor'")

        return self + other

    def __mul__(self, other: Union[int, float, Tensor]) -> Tensor: # self * other
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for *: 'Tensor' and '{type(other).__name__}'")

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: Union[int, float, Tensor]) -> Tensor: # other * self
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'Tensor'")

        return self * other

    def __neg__(self) -> Tensor: # -self
        return self * -1

    def __sub__(self, other: Union[int, float, Tensor]) -> Tensor: # self - other
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for -: 'Tensor' and '{type(other).__name__}'")

        return self + (-other)

    def __rsub__(self, other: Union[int, float, Tensor]) -> Tensor: # other - self
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(other).__name__}' and 'Tensor'")

        return other + (-self)

    def __pow__(self, other: Union[int, float]) -> Tensor: # self ** other
        if not isinstance(other, (int, float)):
            raise TypeError(f"Unsupported operand type(s) for **: 'Tensor' and '{type(other).__name__}'")

        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other: Union[int, float, Tensor]) -> Tensor: # self / other
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for /: 'Tensor' and '{type(other).__name__}'")

        return self * (other ** -1)

    def __rtruediv__(self, other: Union[int, float, Tensor]) -> Tensor: # other / self
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'Tensor'")

        return other * (self ** -1)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

if __name__ == "__main__":
    pass
