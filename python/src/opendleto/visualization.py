"""Interactive three-dimensional tensor plots (requires the lab extra)."""

import numpy as np


def _trace(value, tol, color, name):
    from plotly import graph_objects as go

    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("tensor plotting requires exactly three axes")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("tensor must be numeric")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    magnitudes = np.abs(array)
    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("tensor must contain only finite values")
    coords = np.nonzero(magnitudes > tol)
    values = magnitudes[coords]
    # Marker area is proportional to coefficient magnitude.
    sizes = 3 + 9 * np.sqrt(values / values.max()) if values.size else []
    return go.Scatter3d(x=coords[0], y=coords[1], z=coords[2],
                        mode="markers", name=name,
                        marker=dict(size=sizes, color=color),
                        customdata=values, hovertemplate=(
                            "(%{x}, %{y}, %{z})<br>|value|=%{customdata:.4g}<extra></extra>"))


def _scene(value):
    array = np.asarray(value)
    return {
        "aspectmode": "data",
        "xaxis": {"autorange": False, "range": [-0.5, array.shape[0] - 0.5]},
        "yaxis": {"autorange": False, "range": [-0.5, array.shape[1] - 0.5]},
        "zaxis": {"autorange": False, "range": [-0.5, array.shape[2] - 0.5]},
    }


def plot_tensor(array, tol=1e-4, *, title="Tensor", color="blue"):
    """Plot nonzero entries of a 3D tensor, using coefficient magnitudes.

    Indices are zero-based. Entries with magnitude at most ``tol`` are omitted.
    Returns a Plotly figure, which can be displayed in a notebook with ``show()``.
    """
    from plotly import graph_objects as go

    figure = go.Figure(data=[_trace(array, tol, color, title)])
    figure.update_layout(title=title, scene=_scene(array), showlegend=False)
    return figure


def compare(left, right, tol=1e-4, *, left_title="Left", right_title="Right",
            left_color="blue", right_color="red", vertical=False):
    """Compare two 3D tensors in an interactive side-by-side figure."""
    from plotly.subplots import make_subplots

    rows, cols = (2, 1) if vertical else (1, 2)
    figure = make_subplots(rows=rows, cols=cols,
                           specs=[[{"type": "scene"}] * cols for _ in range(rows)],
                           subplot_titles=(left_title, right_title))
    figure.add_trace(_trace(left, tol, left_color, left_title), row=1, col=1)
    figure.add_trace(_trace(right, tol, right_color, right_title),
                     row=2 if vertical else 1, col=1 if vertical else 2)
    figure.update_layout(scene=_scene(left), scene2=_scene(right),
                         showlegend=False)
    return figure


def side_by_side(*arrays, titles=None, colors=None, tol=1e-4):
    """Show two or more 3D tensors in one row, as in the Julia labs."""
    from plotly.subplots import make_subplots

    if len(arrays) < 2:
        raise ValueError("provide at least two tensors")
    if titles is None:
        titles = tuple(f"Tensor {i + 1}" for i in range(len(arrays)))
    if colors is None:
        colors = tuple(("blue", "red", "green")[i % 3] for i in range(len(arrays)))
    if len(titles) != len(arrays) or len(colors) != len(arrays):
        raise ValueError("titles and colors must match the number of tensors")
    fig = make_subplots(rows=1, cols=len(arrays),
                        specs=[[{"type": "scene"}] * len(arrays)],
                        subplot_titles=titles)
    for i, (array, title, color) in enumerate(zip(arrays, titles, colors), start=1):
        fig.add_trace(_trace(array, tol, color, title), row=1, col=i)
        fig.update_layout(**{("scene" if i == 1 else f"scene{i}"): _scene(array)})
    fig.update_layout(showlegend=False, width=max(900, 360 * len(arrays)))
    return fig
