"""Mô-đun tối ưu hóa siêu tham số và phân chia kiểm định chéo dùng chung."""

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

from src.config import RANDOM_STATE

SearchObject = GridSearchCV | RandomizedSearchCV


@dataclass(slots=True)
class TuningResult:
    """Lưu trữ kết quả tối ưu siêu tham số bao gồm mô hình, tham số tốt nhất và thời gian thực thi."""

    best_estimator: BaseEstimator | None
    best_params: dict[str, Any]
    best_score: float
    elapsed_seconds: float
    search: SearchObject


def make_cv_splitter(
    task_type: str = "classification",
    n_splits: int = 5,
    random_state: int | None = RANDOM_STATE,
    shuffle: bool = True,
) -> StratifiedKFold | KFold:
    """Tạo đối tượng phân chia dữ liệu kiểm định chéo StratifiedKFold hoặc KFold có thể tái lập."""
    if not isinstance(task_type, str):
        raise TypeError("task_type phải là 'classification' hoặc 'regression'.")
    normalized_task = task_type.strip().lower()
    if normalized_task not in {"classification", "regression"}:
        raise ValueError(f"task_type không hợp lệ: {task_type!r}.")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits phải là số nguyên >= 2.")
    if n_splits < 2:
        raise ValueError("n_splits phải >= 2.")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle phải là kiểu boolean.")

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
    random_state: int | None = RANDOM_STATE,
    verbose: int = 0,
    X: Any = None,
    y: Any = None,
    refit: bool = True,
    return_train_score: bool = False,
    error_score: str | float = "raise",
) -> TuningResult:
    """Thực thi tìm kiếm siêu tham số tối ưu (GridSearchCV hoặc RandomizedSearchCV) trên tập Train."""
    if estimator is None:
        raise ValueError("estimator không được để trống.")
    _validate_parameter_space(param_grid)
    if cv_splitter is None:
        raise ValueError("cv_splitter không được để trống.")
    if not scoring_metric:
        raise ValueError("scoring_metric phải được cung cấp.")
    if X is None or y is None:
        raise ValueError("X và y phải được cung cấp đầy đủ.")
    if not isinstance(search_type, str):
        raise TypeError("search_type phải là 'grid' hoặc 'random'.")

    normalized_search = search_type.strip().lower()
    if normalized_search not in {"grid", "random"}:
        raise ValueError(f"search_type không hợp lệ: {search_type!r}.")

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
            raise ValueError("n_iter phải là số nguyên dương cho random search.")
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
    """Trích xuất và sắp xếp bảng kết quả chi tiết từ đối tượng tìm kiếm kiểm định chéo đã fit."""
    search = (
        search_or_result.search
        if isinstance(search_or_result, TuningResult)
        else search_or_result
    )
    if not hasattr(search, "cv_results_"):
        raise ValueError("Đối tượng tìm kiếm chưa được fit.")

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
    """Xác thực tính hợp lệ của không gian tham số đầu vào trước khi tiến hành tìm kiếm."""
    if isinstance(param_grid, Mapping):
        spaces = [param_grid]
    elif isinstance(param_grid, Sequence) and not isinstance(param_grid, (str, bytes)):
        spaces = list(param_grid)
        if not all(isinstance(space, Mapping) for space in spaces):
            raise TypeError("Mỗi phần tử trong không gian tham số phải là một mapping dictionary.")
    else:
        raise TypeError("param_grid phải là mapping hoặc chuỗi các mapping.")

    if not spaces or any(len(space) == 0 for space in spaces):
        raise ValueError("param_grid phải chứa ít nhất một tham số.")


__all__ = [
    "TuningResult",
    "cv_results_dataframe",
    "make_cv_splitter",
    "tune_model",
]
