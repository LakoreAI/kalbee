from typing import Optional, Tuple
import numpy as np


def plot_trajectory(
    true_positions: Optional[np.ndarray] = None,
    measured_positions: Optional[np.ndarray] = None,
    estimated_positions: Optional[np.ndarray] = None,
    title: str = "Trajectory",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot trajectory with true, measured, and estimated positions.

    Args:
        true_positions: Ground truth positions (T x 2).
        measured_positions: Noisy measurements (T x 2).
        estimated_positions: Filter estimates (T x 2).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size.
        save_path: Path to save figure. If None, displays interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install kalbee[viz]")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if true_positions is not None:
        ax.plot(true_positions[:, 0], true_positions[:, 1],
                'g-', linewidth=2, label='True')

    if measured_positions is not None:
        ax.scatter(measured_positions[:, 0], measured_positions[:, 1],
                   c='red', s=10, alpha=0.5, label='Measured')

    if estimated_positions is not None:
        ax.plot(estimated_positions[:, 0], estimated_positions[:, 1],
                'b--', linewidth=2, label='Estimated')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax


def plot_covariance(
    covariance_history: list,
    state_names: Optional[list] = None,
    title: str = "State Uncertainty",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot covariance diagonal (uncertainty) over time.

    Args:
        covariance_history: List of covariance matrices.
        state_names: Names for each state dimension.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required. Install with: pip install kalbee[viz]")

    n = covariance_history[0].shape[0]
    T = len(covariance_history)

    if state_names is None:
        state_names = [f"State {i}" for i in range(n)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(n):
        std_devs = [np.sqrt(max(0.0, P[i, i])) for P in covariance_history]
        ax.plot(range(T), std_devs, label=state_names[i] if i < len(state_names) else f"State {i}")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Standard Deviation (sqrt(P_ii))")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, ax


def plot_innovations(
    innovations: np.ndarray,
    confidence: float = 0.95,
    title: str = "Innovation Sequence",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    """
    Plot innovation sequence with confidence bounds.

    Args:
        innovations: Innovation array (T x m).
        confidence: Confidence level for bounds.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        raise ImportError("matplotlib and scipy are required.")

    T, m = innovations.shape
    fig, axes = plt.subplots(m, 1, figsize=figsize, sharex=True)

    if m == 1:
        axes = [axes]

    z_score = stats.norm.ppf((1 + confidence) / 2)

    for i in range(m):
        ax = axes[i]
        ax.plot(innovations[:, i], 'b-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        std = np.std(innovations[:, i])
        ax.axhline(y=z_score * std, color='r', linestyle='--', alpha=0.5,
                   label=f'{confidence*100:.0f}% bounds')
        ax.axhline(y=-z_score * std, color='r', linestyle='--', alpha=0.5)

        ax.set_ylabel(f"Innovation {i+1}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    return fig, axes
