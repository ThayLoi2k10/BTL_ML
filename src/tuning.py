"""Shared, train-only hyperparameter tuning utilities for Week 3.

Example
-------
    cv = make_cv_splitter("classification")
    result = tune_model(
        estimator=model,
        param_grid=params,
        cv_splitter=cv,
        scoring_metric="f1_macro",
        X=X_train,
        y=y_train,
    )

This module never loads project data. Callers must pass training features and
training labels explicitly; held-out test data belongs to later evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)


SearchObject = GridSearchCV | RandomizedSearchCV


@dataclass(slots=True)
class TuningResult:
    """Result returned after a completed cross-validation search.

    ``best_estimator`` is ``None`` when ``refit=False`` because sklearn does
    not fit a final estimator in that mode.
    """

    best_estimator: BaseEstimator | None
    best_params: dict[str, Any]
    best_score: float
    elapsed_seconds: float
    search: SearchObject


def make_cv_splitter(
    task_type: str = "classification",
    n_splits: int = 5,
    random_state: int | None = 42,
    shuffle: bool = True,
) -> StratifiedKFold | KFold:
    """Create a reproducible cross-validation splitter.

    Parameters
    ----------
    task_type:
        ``"classification"`` returns :class:`StratifiedKFold`, while
        ``"regression"`` returns :class:`KFold`.
    n_splits:
        Number of folds; must be at least two.
    random_state:
        Seed used only when ``shuffle=True``. It is set to ``None`` when
        shuffling is disabled, as required by sklearn's splitter semantics.
    shuffle:
        Whether to shuffle samples before constructing folds.
    """

    if not isinstance(task_type, str):
        raise TypeError("task_type must be 'classification' or 'regression'.")
    normalized_task = task_type.strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise ValueError(
            "task_type must be either 'classification' or 'regression'; "
            f"received {task_type!r}."
        )
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer greater than or equal to 2.")
    if n_splits < 2:
        raise ValueError("n_splits must be greater than or equal to 2.")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a boolean.")

    splitter_kwargs = {
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state if shuffle else None,
    }
    if normalized_task == "classification":
        return StratifiedKFold(**splitter_kwargs)
    return KFold(**splitter_kwargs)


def tune_model(
    estimator: BaseEstimator,
    param_grid: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    cv_splitter: Any,
    scoring_metric: str | Callable[..., float],
    *,
    search_type: str = "grid",
    n_iter: int = 20,
    n_jobs: int | None = -1,
    random_state: int | None = 42,
    verbose: int = 0,
    X: Any = None,
    y: Any = None,
    refit: bool = True,
    return_train_score: bool = False,
    error_score: str | float = "raise",
) -> TuningResult:
    """Tune any sklearn-compatible estimator using only supplied train data.

    ``search_type="grid"`` uses :class:`GridSearchCV` and ignores ``n_iter``
    and ``random_state`` because an exhaustive grid is deterministic.
    ``search_type="random"`` uses :class:`RandomizedSearchCV` with those two
    arguments. The function never loads or references a held-out test set.
    """

    if estimator is None:
        raise ValueError("estimator must not be None.")
    _validate_parameter_space(param_grid)
    if cv_splitter is None:
        raise ValueError("cv_splitter must not be None.")
    if scoring_metric is None or scoring_metric == "":
        raise ValueError("scoring_metric must be supplied by the caller.")
    if X is None or y is None:
        raise ValueError("X and y training data must both be supplied.")
    if not isinstance(search_type, str):
        raise TypeError("search_type must be 'grid' or 'random'.")

    normalized_search = search_type.strip().lower()
    if normalized_search not in {"grid", "random"}:
        raise ValueError(
            "search_type must be either 'grid' or 'random'; "
            f"received {search_type!r}."
        )

    common_arguments = {
        "estimator": estimator,
        "scoring": scoring_metric,
        "cv": cv_splitter,
        "n_jobs": n_jobs,
        "verbose": verbose,
        "refit": refit,
        "return_train_score": return_train_score,
        "error_score": error_score,
    }

    if normalized_search == "grid":
        search: SearchObject = GridSearchCV(
            param_grid=param_grid,
            **common_arguments,
        )
    else:
        if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer for random search.")
        search = RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=random_state,
            **common_arguments,
        )

    started_at = perf_counter()
    search.fit(X, y)
    elapsed_seconds = perf_counter() - started_at

    return TuningResult(
        best_estimator=getattr(search, "best_estimator_", None),
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        elapsed_seconds=float(elapsed_seconds),
        search=search,
    )


def cv_results_dataframe(search_or_result: SearchObject | TuningResult) -> pd.DataFrame:
    """Return the main reporting columns from a fitted SearchCV result.

    Training-score columns are included only when the search was configured
    with ``return_train_score=True``. No file is written.
    """

    search = (
        search_or_result.search
        if isinstance(search_or_result, TuningResult)
        else search_or_result
    )
    if not hasattr(search, "cv_results_"):
        raise ValueError("The supplied search object has not been fitted.")

    preferred_columns = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_train_score",
        "std_train_score",
    ]
    available_columns = [
        column for column in preferred_columns if column in search.cv_results_
    ]
    results = pd.DataFrame(search.cv_results_).loc[:, available_columns]
    if "rank_test_score" in results.columns:
        results = results.sort_values("rank_test_score", kind="mergesort")
    return results.reset_index(drop=True)


def _validate_parameter_space(
    param_grid: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> None:
    """Reject obviously empty or malformed search spaces."""

    if isinstance(param_grid, Mapping):
        spaces = [param_grid]
    elif isinstance(param_grid, Sequence) and not isinstance(param_grid, (str, bytes)):
        spaces = list(param_grid)
        if not all(isinstance(space, Mapping) for space in spaces):
            raise TypeError("Every parameter-space entry must be a mapping.")
    else:
        raise TypeError("param_grid must be a mapping or a sequence of mappings.")

    if not spaces or any(len(space) == 0 for space in spaces):
        raise ValueError("param_grid must contain at least one parameter.")


__all__ = [
    "TuningResult",
    "cv_results_dataframe",
    "make_cv_splitter",
    "tune_model",
]
