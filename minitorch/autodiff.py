from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Any,
    *vals: Any,
    arg: int = 0,
    epsilon: float = 1e-6,
) -> Any:
    r"""Compute an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    val_list = list(vals)

    x_i = val_list[arg]

    val_list[arg] = x_i + epsilon / 2
    forward = f(*val_list)

    val_list[arg] = x_i - epsilon / 2
    backward = f(*val_list)

    return (forward - backward) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:  # pyright: ignore[reportReturnType]
        pass

    def is_leaf(self) -> bool:  # pyright: ignore[reportReturnType]
        pass

    def is_constant(self) -> bool:  # pyright: ignore[reportReturnType]
        pass

    @property
    def parents(self) -> Iterable["Variable"]:  # pyright: ignore[reportReturnType]
        pass

    def chain_rule(self, d_output: Any) -> Iterable[tuple["Variable", Any]]:  # pyright: ignore[reportReturnType]
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Compute the topological order of the computation graph.

    Uses depth-first search from
    https://en.wikipedia.org/wiki/Topological_sorting

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.

    """
    sorted_nodes = []
    visited = set()
    visiting = set()

    def visit(node: Variable) -> None:
        if node.unique_id in visited or node.is_constant():
            return

        if node.unique_id in visiting:
            raise RuntimeError(
                "Graph has a cycle - cannot compute topological sort",
            )

        visiting.add(node.unique_id)

        for parent in node.parents:
            visit(parent)

        visited.add(node.unique_id)
        sorted_nodes.append(node)

    visit(variable)

    return sorted_nodes


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Run backpropagation on the computation graph.

    Compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of
    each leaf through `accumulate_derivative`.

    """
    # Call topological sort to get an ordered queue
    topology = list(topological_sort(variable))
    # Create a dictionary of Scalars and current derivatives
    derivatives = {variable.unique_id: deriv}
    # For each node in backward order, pull a completed Scalar and
    # derivative from the queue:
    for node in reversed(topology):
        node_deriv = derivatives[node.unique_id]
        if node.is_leaf():
            # a. if the Scalar is a leaf, add its final derivative
            # (accumulate_derivative) and loop to (1)
            node.accumulate_derivative(node_deriv)
        else:
            # if the Scalar is not a leaf,
            # call .chain_rule on the last function with d_out
            for parent_node, parent_deriv in node.chain_rule(node_deriv):
                # loop through all the Scalars+derivative produced by
                # the chain rule accumulate derivatives for the Scalar
                # in a dictionary
                if parent_node.unique_id not in derivatives:
                    derivatives[parent_node.unique_id] = 0
                derivatives[parent_node.unique_id] += parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        return self.saved_values
