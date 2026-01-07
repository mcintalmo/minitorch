import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

import minitorch
from minitorch import TensorData

from .tensor_strategies import indices, tensor_data

# ## Tasks 2.1

# Check basic properties of layout and strides.


@pytest.mark.task2_1
def test_layout() -> None:
    """Test basis properties of layout and strides"""
    data = [0] * 3 * 5
    tensor_data = minitorch.TensorData(data, (3, 5), (5, 1))

    assert tensor_data.is_contiguous()
    assert tensor_data.shape == (3, 5)
    assert tensor_data.index((1, 0)) == 5
    assert tensor_data.index((1, 2)) == 7

    tensor_data = minitorch.TensorData(data, (5, 3), (1, 5))
    assert tensor_data.shape == (5, 3)
    assert not tensor_data.is_contiguous()

    data = [0] * 4 * 2 * 2
    tensor_data = minitorch.TensorData(data, (4, 2, 2))
    assert tensor_data.strides == (4, 2, 1)


@pytest.mark.xfail
def test_layout_bad() -> None:
    """Test basis properties of layout and strides"""
    data = [0] * 3 * 5
    minitorch.TensorData(data, (3, 5), (6,))


@pytest.mark.task2_1
@given(tensor_data())
def test_enumeration(tensor_data: TensorData) -> None:
    """Test enumeration of tensor_datas."""
    indices = list(tensor_data.indices())

    # Check that enough positions are enumerated.
    assert len(indices) == tensor_data.size

    # Check that enough positions are enumerated only once.
    assert len(set(tensor_data.indices())) == len(indices)

    # Check that all indices are within the shape.
    for ind in tensor_data.indices():
        for i, p in enumerate(ind):
            assert p >= 0 and p < tensor_data.shape[i]


@pytest.mark.task2_1
@given(tensor_data())
def test_index(tensor_data: TensorData) -> None:
    """Test enumeration of tensor_data."""
    # Check that all indices are within the size.
    for ind in tensor_data.indices():
        pos = tensor_data.index(ind)
        assert pos >= 0 and pos < tensor_data.size

    base = [0] * tensor_data.dims
    with pytest.raises(minitorch.IndexingError):
        base[0] = -1
        tensor_data.index(tuple(base))

    if tensor_data.dims > 1:
        with pytest.raises(minitorch.IndexingError):
            base = [0] * (tensor_data.dims - 1)
            tensor_data.index(tuple(base))


@pytest.mark.task2_1
@given(data())
def test_permute(data: DataObject) -> None:
    td = data.draw(tensor_data())
    ind = data.draw(indices(td))
    td_rev = td.permute(*list(reversed(range(td.dims))))
    assert td.index(ind) == td_rev.index(tuple(reversed(ind)))

    td2 = td_rev.permute(*list(reversed(range(td_rev.dims))))
    assert td.index(ind) == td2.index(ind)


# ## Tasks 2.2

# Check basic properties of broadcasting.


@pytest.mark.task2_2
def test_shape_broadcast() -> None:
    c = minitorch.shape_broadcast((1,), (5, 5))
    assert c == (5, 5)

    c = minitorch.shape_broadcast((5, 5), (1,))
    assert c == (5, 5)

    c = minitorch.shape_broadcast((1, 5, 5), (5, 5))
    assert c == (1, 5, 5)

    c = minitorch.shape_broadcast((5, 1, 5, 1), (1, 5, 1, 5))
    assert c == (5, 5, 5, 5)

    with pytest.raises(minitorch.IndexingError):
        c = minitorch.shape_broadcast((5, 7, 5, 1), (1, 5, 1, 5))
        print(c)

    with pytest.raises(minitorch.IndexingError):
        c = minitorch.shape_broadcast((5, 2), (5,))
        print(c)

    c = minitorch.shape_broadcast((2, 5), (5,))
    assert c == (2, 5)


def test_broadcast_index():
    """When shapes are the same, index should be unchanged."""
    big_index = np.array([1, 2, 3], dtype=np.int32)
    big_shape = np.array([4, 5, 6], dtype=np.int32)
    shape = np.array([4, 5, 6], dtype=np.int32)
    out_index = np.zeros(3, dtype=np.int32)

    minitorch.broadcast_index(big_index, big_shape, shape, out_index)

    assert np.array_equal(out_index, [1, 2, 3])

    big_index = np.array([2, 3], dtype=np.int32)
    big_shape = np.array([5, 6], dtype=np.int32)
    shape = np.array([1, 6], dtype=np.int32)  # First dimension is 1
    out_index = np.zeros(2, dtype=np.int32)

    minitorch.broadcast_index(big_index, big_shape, shape, out_index)

    # First dimension maps to 0 (broadcast dimension)
    # Second dimension stays the same
    assert np.array_equal(out_index, [0, 3])

    big_index = np.array([1, 2, 3, 4], dtype=np.int32)
    big_shape = np.array([2, 3, 5, 6], dtype=np.int32)
    shape = np.array([5, 6], dtype=np.int32)  # Missing first 2 dimensions
    out_index = np.zeros(2, dtype=np.int32)

    minitorch.broadcast_index(big_index, big_shape, shape, out_index)

    # First two dimensions of big_index are ignored
    # Last two dimensions map directly
    assert np.array_equal(out_index, [3, 4])

    big_index = np.array([1, 2, 3, 4], dtype=np.int32)
    big_shape = np.array([2, 3, 4, 5], dtype=np.int32)
    shape = np.array([3, 1, 5], dtype=np.int32)
    out_index = np.zeros(3, dtype=np.int32)

    minitorch.broadcast_index(big_index, big_shape, shape, out_index)

    # Dimension 0 of big (idx=1) is extra → skip
    # Dimension 1 of big (idx=2) → Dimension 0 of small (idx=2)
    # Dimension 2 of big (idx=3) → Dimension 1 of small (1 is broadcast) → 0
    # Dimension 3 of big (idx=4) → Dimension 2 of small (idx=4)
    assert np.array_equal(out_index, [2, 0, 4])

    big_index = np.array([1, 2, 3], dtype=np.int32)
    big_shape = np.array([4, 5, 6], dtype=np.int32)
    shape = np.array([], dtype=np.int32)  # Scalar
    out_index = np.array([], dtype=np.int32)

    minitorch.broadcast_index(big_index, big_shape, shape, out_index)

    # All dimensions map to nothing (scalar has no dimensions)
    assert len(out_index) == 0

    big_index = np.array([2, 3, 4], dtype=np.int32)
    big_shape = np.array([5, 6, 7], dtype=np.int32)
    shape = np.array([1, 1, 1], dtype=np.int32)
    out_index = np.zeros(3, dtype=np.int32)

    minitorch.broadcast_index(big_index, big_shape, shape, out_index)

    # All dimensions are broadcast, so all indices map to 0
    assert np.array_equal(out_index, [0, 0, 0])


@given(tensor_data())
def test_string(tensor_data: TensorData) -> None:
    tensor_data.to_string()
