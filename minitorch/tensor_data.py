from __future__ import annotations

import builtins
import random
from collections.abc import Iterable, Sequence
from itertools import zip_longest
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from numpy import array, float64

try:
    from numba import cuda as numba_cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    numba_cuda = None  # type: ignore

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Convert an index to a position.

    Converts multidimensional tensor `index` into a single-dimensional
    position in storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage

    """
    return sum(i * s for i, s in zip(index, strides, strict=True))


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.

    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    remainder = ordinal
    start = len(shape) - 1
    stop = -1
    step = -1
    for dimension in range(start, stop, step):
        size = shape[dimension]
        out_index[dimension] = remainder % size
        remainder = remainder // size


def broadcast_index(
    big_index: Index,
    big_shape: Shape,
    shape: Shape,
    out_index: OutIndex,
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None

    """
    for i in range(len(shape)):
        big_dim = len(big_shape) - len(shape) + i
        out_index[i] = big_index[big_dim] if shape[i] > 1 else 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast

    """
    dimensions = zip_longest(reversed(shape1), reversed(shape2), fillvalue=1)

    def broadcast_pair(dim1: int, dim2: int) -> int:
        if dim1 != dim2 and 1 not in (dim1, dim2):
            raise IndexingError("Cannot broadcast")
        return max(dim1, dim2)

    broadcast = [broadcast_pair(dim1, dim2) for dim1, dim2 in dimensions]
    return tuple(reversed(broadcast))


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Sequence[float] | Storage,
        shape: UserShape,
        strides: UserStrides | None = None,
    ) -> None:
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(
                f"Len of strides {strides} must match {shape}.",
            )
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if CUDA_AVAILABLE and numba_cuda is not None:
            if not numba_cuda.is_cuda_array(self._storage):
                self._storage = numba_cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: int | UserIndex) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        elif isinstance(index, tuple):
            aindex = array(index)
        else:
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(
                f"Index {aindex} must be size of {self.shape}.",
            )
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(
                    f"Index {aindex} out of range {self.shape}.",
                )
            if ind < 0:
                raise IndexingError(
                    f"Negative indexing for {aindex} not supported.",
                )

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple(random.randint(0, s - 1) for s in self.shape)

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> builtins.tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.

        """
        if sorted(order) != list(range(len(self.shape))):
            msg = (
                "Must give a position to each dimension. "
                f"Shape: {self.shape} Order: {order}"
            )
            raise IndexingError(msg)

        shape = tuple(self.shape[o] for o in order)
        strides = tuple(self.strides[o] for o in order)
        return TensorData(self._storage, shape, strides)

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            left_bracket = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    left_bracket = "\n%s[" % ("\t" * i) + left_bracket
                else:
                    break
            s += left_bracket
            v = self.get(index)
            s += f"{v:3.2f}"
            right_bracket = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    right_bracket += "]"
                else:
                    break
            if right_bracket:
                s += right_bracket
            else:
                s += " "
        return s
