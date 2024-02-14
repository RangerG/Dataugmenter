import numpy as np
import matplotlib.pyplot as plt


def plot2d(x, y, x2=None, y2=None, x3=None, y3=None, xlim=(-1, 1), ylim=(-1, 1), save_file=""):
    """
    Plot a 2D graph with up to three lines.

    Args:
        x, y: Coordinates for the first line.
        x2, y2: Coordinates for the second line (optional).
        x3, y3: Coordinates for the third line (optional).
        xlim: Tuple for x-axis limits.
        ylim: Tuple for y-axis limits.
        save_file: Path to save the figure, displays plot if empty.

    Returns:
        None
    """
    plt.figure(figsize=(4, 4))
    plt.plot(x, y)
    if x2 is not None and y2 is not None:
        plt.plot(x2, y2)
    if x3 is not None and y3 is not None:
        plt.plot(x3, y3)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()


def plot1d(x, x2=None, x3=None, ylim=(-1, 1), save_file="", return_axes=False):
    """
    Plot a 1D graph with up to three lines.

    Args:
        x: Data for the first line.
        x2: Data for the second line (optional).
        x3: Data for the third line (optional).
        ylim: Tuple for y-axis limits.
        save_file: Path to save the figure, displays plot if empty.
        return_axes: Return the matplotlib axes object if True.

    Returns:
        Axes object if return_axes is True, otherwise None.
    """
    fig = plt.figure(figsize=(6, 3))
    steps = np.arange(x.shape[0])
    plt.plot(steps, x)
    if x2 is not None:
        plt.plot(steps, x2)
    if x3 is not None:
        plt.plot(steps, x3)
    plt.xlim(0, x.shape[0])
    plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    elif not return_axes:
        plt.show()
    if return_axes:
        return plt.gca()
    else:
        plt.close(fig)
