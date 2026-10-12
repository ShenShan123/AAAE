"""Fast exact non-dominated filtering for three minimization objectives."""

from __future__ import annotations

import numpy as np


def nondominated_mask_3d(values: np.ndarray) -> np.ndarray:
    """Return an O(n log n) Pareto mask, retaining rows with equal objectives.

    Columns can be any three minimization objectives. Rows must be finite.
    Sorting on the first objective leaves only a prefix query on the second
    objective and a minimum check on the third. The Fenwick tree also stores
    the first two values to distinguish a strict dominator from an exact tie.
    """
    objectives = np.asarray(values, dtype=np.float64)
    if objectives.ndim != 2 or objectives.shape[1] != 3:
        raise ValueError(f"expected an (n, 3) objective array, got {objectives.shape}")
    if not np.isfinite(objectives).all():
        raise ValueError("Pareto objectives must all be finite")

    count = len(objectives)
    selected = np.zeros(count, dtype=bool)
    if count == 0:
        return selected

    order = np.lexsort((np.arange(count), objectives[:, 2], objectives[:, 1], objectives[:, 0]))
    second_values = np.unique(objectives[:, 1])
    second_indices = np.searchsorted(second_values, objectives[:, 1]) + 1
    sentinel = (float("inf"), float("inf"), float("inf"))
    prefix_minimum = [sentinel] * (len(second_values) + 1)

    for row_index in order:
        first, second, third = objectives[row_index]
        tree_index = int(second_indices[row_index])
        best = sentinel
        query_index = tree_index
        while query_index:
            best = min(best, prefix_minimum[query_index])
            query_index -= query_index & -query_index
        dominated = best[0] < third or (best[0] == third and (best[1] < first or best[2] < second))
        if dominated:
            continue
        selected[row_index] = True
        value = (third, first, second)
        while tree_index <= len(second_values):
            prefix_minimum[tree_index] = min(prefix_minimum[tree_index], value)
            tree_index += tree_index & -tree_index

    return selected
