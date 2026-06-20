import numpy as np
from numpy.typing import NDArray


def sinkhorn_distance(
    r: NDArray[np.float64],
    c: NDArray[np.float64],
    Lambda: float,
    max_iter: int = 50,
    eps: float = 1e-12
) -> float:
    """
    Compute the Sinkhorn transportation cost between two flattened MNIST images.

    The two images are interpreted as discrete probability distributions on the
    28x28 pixel grid. Only nonzero pixels are kept in order to reduce the size
    of the optimal transport problem.

    Parameters
    ----------
    r : NDArray[np.float64]
        First flattened MNIST image of shape (784,). It must be nonnegative.
    c : NDArray[np.float64]
        Second flattened MNIST image of shape (784,). It must be nonnegative.
    Lambda : float
        Entropic regularization parameter. Larger values make the transport
        plan closer to the classical optimal transport plan.
    max_iter : int, optional
        Number of Sinkhorn iterations. Default is 50.
    eps : float, optional
        Small numerical constant used to avoid division by zero. Default is 1e-12.

    Returns
    -------
    float
        Sinkhorn transportation cost between r and c.
    """

    # Convert images to float arrays
    r = r.astype(float)
    c = c.astype(float)

    # Normalize images so that they become probability distributions
    r = r / (r.sum() + eps)
    c = c / (c.sum() + eps)

    # Keep only nonzero pixels to reduce the size of the problem
    r_idx = np.where(r > 0)[0]
    c_idx = np.where(c > 0)[0]

    # Keep the corresponding masses
    r = r[r_idx]
    c = c[c_idx]

    # Convert pixel indices into 2D coordinates on the 28x28 grid
    r_x = r_idx // 28
    r_y = np.mod(r_idx, 28)

    c_x = c_idx // 28
    c_y = np.mod(c_idx, 28)

    # Build the cost matrix M
    # M[i, j] is the Euclidean distance between pixel i of r and pixel j of c
    M = np.sqrt(
        (r_x[:, None] - c_x[None, :])**2
        +
        (r_y[:, None] - c_y[None, :])**2
    )

    # Compute the Gibbs kernel K = exp(-Lambda * M)
    K = np.exp(-Lambda * M)

    # Initialize x
    x = (1 / len(r)) * np.ones(len(r))

    # Sinkhorn iterations
    for _ in range(max_iter):
        # Update u
        u = 1 / (x + eps)

        # Update v using the target marginal c
        v = c / (np.matmul(K.T, u) + eps)

        # Update x using the source marginal r
        x = np.matmul(K, v) / (r + eps)

    # Recompute u and v after the last update
    u = 1 / (x + eps)
    v = c / (np.matmul(K.T, u) + eps)

    # Compute the optimal transport plan P = diag(u) K diag(v)
    P = np.matmul(np.matmul(np.diag(u), K), np.diag(v))

    # Return the transportation cost <P, M>
    return float(np.sum(P * M))


def hellinger(
    p: NDArray[np.float64],
    q: NDArray[np.float64]
) -> float:
    """
    Compute the Hellinger distance between two discrete distributions.

    The Hellinger distance is defined by

        H(p, q) = (1 / sqrt(2)) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2)).

    Parameters
    ----------
    p : NDArray[np.float64]
        First nonnegative vector.
    q : NDArray[np.float64]
        Second nonnegative vector.

    Returns
    -------
    float
        Hellinger distance between p and q.
    """

    p = p.astype(float)
    q = q.astype(float)

    return float((1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)))


def chi2_distance(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    eps: float = 1e-12
) -> float:
    """
    Compute the chi-square distance between two discrete distributions.

    The chi-square distance is defined by

        chi2(p, q) = sum((p_i - q_i)^2 / (p_i + q_i)).

    A small epsilon is added to the denominator to avoid division by zero.

    Parameters
    ----------
    p : NDArray[np.float64]
        First nonnegative vector.
    q : NDArray[np.float64]
        Second nonnegative vector.
    eps : float, optional
        Small numerical constant used to avoid division by zero. Default is 1e-12.

    Returns
    -------
    float
        Chi-square distance between p and q.
    """

    p = p.astype(float)
    q = q.astype(float)

    return float(np.sum((p - q)**2 / (p + q + eps)))