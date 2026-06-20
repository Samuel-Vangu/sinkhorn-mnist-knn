from functools import partial

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_test(
    distance,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    sinkhorn=False,
    Lambda=9,
    max_iter=20,
    eps=1e-12,
    cv=3
):
    """
    Train a k-NN classifier using a given distance, select the best k by
    cross-validation, and evaluate the best model on the test set.

    Parameters
    ----------
    distance : str or callable
        Distance used by KNeighborsClassifier. It can be a scikit-learn metric
        string such as "euclidean" or "manhattan", or a custom function.
    param_grid : dict
        Grid of parameters for GridSearchCV. Example:
        {"n_neighbors": [1, 3, 5, 7]}.
    X_train, y_train
        Training data and labels.
    X_test, y_test
        Test data and labels.
    sinkhorn : bool, optional
        If True, the distance is assumed to be the Sinkhorn distance and the
        parameters Lambda, max_iter and eps are fixed using functools.partial.
    Lambda : float, optional
        Regularization parameter for the Sinkhorn distance.
    max_iter : int, optional
        Number of Sinkhorn iterations.
    eps : float, optional
        Small numerical constant used to avoid division by zero.
    cv : int, optional
        Number of folds for cross-validation.

    Returns
    -------
    float
        Best cross-validation score.
    float
        Test score of the best model.
    dict
        Best parameters found by cross-validation.
    """

    # Fix the Sinkhorn parameters if we use the Sinkhorn distance
    if sinkhorn:
        distance = partial(
            distance,
            Lambda=Lambda,
            max_iter=max_iter,
            eps=eps
        )

    # Use brute force because we may use custom distances
    knn = KNeighborsClassifier(
        metric=distance,
        algorithm="brute"
    )

    # Cross-validation over k
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=cv
    )

    # Fit the models and select the best one
    grid_search.fit(X_train, y_train)

    # Best k-NN model after cross-validation
    best_model = grid_search.best_estimator_

    # Validation score and test score
    validation_score = grid_search.best_score_
    test_score = best_model.score(X_test, y_test)

    return validation_score, test_score, grid_search.best_params_
