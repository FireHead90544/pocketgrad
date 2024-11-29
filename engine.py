from __future__ import annotations
from typing import Union, List, Tuple, Set, Callable

class Tensor:
    """
    A scalar-valued tensor with basic autograd functionality.
    Builds a computation graph to track dependencies between tensors.
    Computes gradients using backpropagation by chaining gradients.
    """

    grad: float
    _backward: Callable[[], None]
    _dependencies: Set[Tensor]

    def __init__(self, data: Union[int, float], _children: Tuple[Tensor, ...] = (), _op: str = '') -> None:
        """
        Initialize a Tensor.

        Args:
            data (Union[int, float]): The value of the tensor.
            _children (Tuple[Tensor, ...], optional): The tensors this tensor depends on. Defaults to ().
            _op (str, optional): The operation that created this tensor. Defaults to ''.
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._dependencies = set(_children)
        self._op = _op

    def backward(self) -> None:
        """
        Perform backpropagation to compute gradients for all dependent tensors.
        """
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def _build_topo(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._dependencies:
                    _build_topo(child)
                topo.append(node)

        _build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __add__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise addition with another tensor or scalar.

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to add.

        Returns:
            Tensor: The resulting tensor after addition.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for +: 'Tensor' and '{type(other).__name__}'")

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise addition (reversed operands).

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to add.

        Returns:
            Tensor: The resulting tensor after addition.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for +: '{type(other).__name__}' and 'Tensor'")

        return self + other

    def __mul__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise multiplication with another tensor or scalar.

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to multiply.

        Returns:
            Tensor: The resulting tensor after multiplication.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for *: 'Tensor' and '{type(other).__name__}'")

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise multiplication (reversed operands).

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to multiply.

        Returns:
            Tensor: The resulting tensor after multiplication.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'Tensor'")

        return self * other

    def __neg__(self) -> Tensor:
        """
        Negate the tensor.

        Returns:
            Tensor: The resulting tensor after negation.
        """
        return self * -1

    def __sub__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise subtraction with another tensor or scalar.

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to subtract.

        Returns:
            Tensor: The resulting tensor after subtraction.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for -: 'Tensor' and '{type(other).__name__}'")

        return self + (-other)

    def __rsub__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise subtraction (reversed operands).

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to subtract.

        Returns:
            Tensor: The resulting tensor after subtraction.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(other).__name__}' and 'Tensor'")

        return other + (-self)

    def __pow__(self, other: Union[int, float]) -> Tensor:
        """
        Perform element-wise exponentiation with a scalar exponent.

        Args:
            other (Union[int, float]): The scalar exponent.

        Returns:
            Tensor: The resulting tensor after exponentiation.
        """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Unsupported operand type(s) for **: 'Tensor' and '{type(other).__name__}'")

        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise division with another tensor or scalar.

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to divide.

        Returns:
            Tensor: The resulting tensor after division.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for /: 'Tensor' and '{type(other).__name__}'")

        return self * (other ** -1)

    def __rtruediv__(self, other: Union[int, float, Tensor]) -> Tensor:
        """
        Perform element-wise division (reversed operands).

        Args:
            other (Union[int, float, Tensor]): The tensor or scalar to divide.

        Returns:
            Tensor: The resulting tensor after division.
        """
        if not isinstance(other, (int, float, Tensor)):
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'Tensor'")

        return other * (self ** -1)

    def __repr__(self) -> str:
        """
        Return a string representation of the tensor.

        Returns:
            str: The string representation of the tensor.
        """
        return f"Tensor(data={self.data}, grad={self.grad})"
