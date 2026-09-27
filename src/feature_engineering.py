"""Feature engineering and one feature-selection strategy for Week 2.

The module is intentionally model-agnostic.  It adds exactly 24 features that
are available before the current marketing call, then provides one selector:
Mutual Information top-K selection for the fully encoded numeric matrix.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted


TARGET_COLUMN = "deposit"
RANDOM_STATE = 42

REQUIRED_SOURCE_COLUMNS = (
    "age",
    "balance",
    "housing",
    "loan",
    "default",
    "pdays",
    "previous",
    "campaign",
    "poutcome",
    "contact",
    "month",
)

MONTH_TO_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


FEATURE_DEFINITIONS = (
    {
        "feature": "age_group",
        "formula": "cut(age, [-inf, 30, 40, 60, inf))",
        "sources": "age",
        "interpretation": "Nhóm giai đoạn tuổi: dưới 30, 30–39, 40–59, từ 60.",
        "rationale": "Cho phép quan hệ theo giai đoạn tuổi thay vì chỉ tuyến tính.",
    },
    {
        "feature": "age_squared",
        "formula": "(age / 10)^2",
        "sources": "age",
        "interpretation": "Thành phần bậc hai của tuổi trên thang thập niên.",
        "rationale": "Bổ sung độ cong mà một hệ số age tuyến tính không biểu diễn được.",
    },
    {
        "feature": "balance_positive_log",
        "formula": "log1p(max(balance, 0))",
        "sources": "balance",
        "interpretation": "Quy mô số dư dương trên thang log.",
        "rationale": "Week 1 ghi nhận balance lệch phải mạnh; log giảm ảnh hưởng đuôi dài.",
    },
    {
        "feature": "balance_debt_log",
        "formula": "log1p(max(-balance, 0))",
        "sources": "balance",
        "interpretation": "Quy mô thấu chi/số dư âm trên thang log.",
        "rationale": "Giữ riêng ý nghĩa nghiệp vụ của 688 số dư âm thay vì coi là lỗi.",
    },
    {
        "feature": "has_negative_balance",
        "formula": "1(balance < 0)",
        "sources": "balance",
        "interpretation": "Cờ khách hàng có số dư âm.",
        "rationale": "Phân biệt trạng thái thấu chi với số dư không âm.",
    },
    {
        "feature": "balance_per_age",
        "formula": "balance / max(age, 1)",
        "sources": "balance, age",
        "interpretation": "Số dư chuẩn hóa thô theo tuổi.",
        "rationale": "Biểu diễn tương tác giữa quy mô tài chính và giai đoạn vòng đời.",
    },
    {
        "feature": "any_loan",
        "formula": "1(housing == 'yes' or loan == 'yes')",
        "sources": "housing, loan",
        "interpretation": "Có ít nhất một trong hai loại khoản vay.",
        "rationale": "Tóm tắt trạng thái đang có nghĩa vụ vay.",
    },
    {
        "feature": "both_loans",
        "formula": "1(housing == 'yes' and loan == 'yes')",
        "sources": "housing, loan",
        "interpretation": "Đồng thời có vay mua nhà và vay cá nhân.",
        "rationale": "Nắm bắt tương tác gánh nặng từ hai khoản vay.",
    },
    {
        "feature": "credit_obligation_count",
        "formula": "1(default=yes) + 1(housing=yes) + 1(loan=yes)",
        "sources": "default, housing, loan",
        "interpretation": "Số trạng thái nghĩa vụ/rủi ro tín dụng đang được ghi nhận.",
        "rationale": "Tổng hợp ba cờ tài chính liên quan thay vì xét hoàn toàn tách biệt.",
    },
    {
        "feature": "financial_stress_flag",
        "formula": "1(default == 'yes' or balance < 0)",
        "sources": "default, balance",
        "interpretation": "Có dấu hiệu default hoặc thấu chi.",
        "rationale": "Kết hợp hai tín hiệu khó khăn tài chính có ý nghĩa nghiệp vụ.",
    },
    {
        "feature": "loan_with_negative_balance",
        "formula": "1(any loan and balance < 0)",
        "sources": "housing, loan, balance",
        "interpretation": "Đang có khoản vay đồng thời số dư âm.",
        "rationale": "Tạo tương tác giữa nghĩa vụ vay và trạng thái thấu chi.",
    },
    {
        "feature": "prior_contacted",
        "formula": "1(pdays >= 0 or previous > 0)",
        "sources": "pdays, previous",
        "interpretation": "Đã có lịch sử liên hệ trước chiến dịch hiện tại.",
        "rationale": "Tách sentinel pdays=-1 khỏi số ngày thực, đúng kết luận Week 1.",
    },
    {
        "feature": "prior_contact_recency_log",
        "formula": "log1p(pdays) if pdays >= 0 else 0",
        "sources": "pdays",
        "interpretation": "Độ trễ từ lần liên hệ trước trên thang log.",
        "rationale": "Giữ ngữ nghĩa khoảng cách ngày và giảm độ lệch phải.",
    },
    {
        "feature": "prior_contact_recent_30d",
        "formula": "1(0 <= pdays <= 30)",
        "sources": "pdays",
        "interpretation": "Đã được liên hệ trong vòng 30 ngày.",
        "rationale": "Biểu diễn tính gần đây bằng một mốc thời gian nghiệp vụ dễ hiểu.",
    },
    {
        "feature": "previous_contacts_log",
        "formula": "log1p(max(previous, 0))",
        "sources": "previous",
        "interpretation": "Số lần liên hệ ở chiến dịch trước trên thang log.",
        "rationale": "Week 1 ghi nhận previous zero-inflated và lệch phải mạnh.",
    },
    {
        "feature": "previous_contact_intensity",
        "formula": "previous / (pdays + 1) if pdays >= 0 else 0",
        "sources": "previous, pdays",
        "interpretation": "Mật độ liên hệ lịch sử tương đối theo độ trễ ngày.",
        "rationale": "Kết hợp tần suất và độ gần đây của lịch sử liên hệ.",
    },
    {
        "feature": "campaign_contacts_log",
        "formula": "log1p(max(campaign, 0))",
        "sources": "campaign",
        "interpretation": "Số lần liên hệ trong chiến dịch hiện tại trên thang log.",
        "rationale": "Campaign lệch phải mạnh trong Week 1; log giảm đuôi dài.",
    },
    {
        "feature": "campaign_repeat_contact",
        "formula": "1(campaign > 1)",
        "sources": "campaign",
        "interpretation": "Đã liên hệ lặp lại trong chiến dịch hiện tại.",
        "rationale": "Phân biệt lần liên hệ đầu với các lần theo đuổi tiếp theo.",
    },
    {
        "feature": "campaign_vs_previous_ratio",
        "formula": "campaign / (previous + 1)",
        "sources": "campaign, previous",
        "interpretation": "Cường độ chiến dịch hiện tại so với lịch sử liên hệ.",
        "rationale": "Biểu diễn sự thay đổi nỗ lực liên hệ giữa hiện tại và quá khứ.",
    },
    {
        "feature": "successful_prior_contact_recency",
        "formula": "1 / (pdays + 1) if poutcome == 'success' and pdays >= 0 else 0",
        "sources": "poutcome, pdays",
        "interpretation": "Độ gần đây có trọng số của một chiến dịch trước thành công.",
        "rationale": "Kết hợp kết quả lịch sử với độ gần đây thay vì sao chép one-hot poutcome.",
    },
    {
        "feature": "failed_prior_campaign_recontact",
        "formula": "1(poutcome == 'failure' and campaign > 1)",
        "sources": "poutcome, campaign",
        "interpretation": "Chiến dịch trước thất bại nhưng hiện tại đã liên hệ lặp lại.",
        "rationale": "Biểu diễn tương tác giữa kết quả quá khứ và nỗ lực hiện tại.",
    },
    {
        "feature": "contact_known",
        "formula": "1(contact not in {'unknown', 'missing', null})",
        "sources": "contact",
        "interpretation": "Phương thức liên hệ đã được ghi nhận.",
        "rationale": "Week 1 ghi nhận 21,02% contact là unknown và có chênh lệch target.",
    },
    {
        "feature": "contact_month_sin",
        "formula": "sin(2*pi*(month_number-1)/12)",
        "sources": "month",
        "interpretation": "Thành phần sin của chu kỳ tháng.",
        "rationale": "Month có association lớn nhất trong nhóm category; mã hóa chu kỳ tránh đứt gãy Dec–Jan.",
    },
    {
        "feature": "contact_month_cos",
        "formula": "cos(2*pi*(month_number-1)/12)",
        "sources": "month",
        "interpretation": "Thành phần cos của chu kỳ tháng.",
        "rationale": "Kết hợp với thành phần sin để biểu diễn đầy đủ vị trí trong năm.",
    },
)

ENGINEERED_FEATURES = tuple(item["feature"] for item in FEATURE_DEFINITIONS)

if len(ENGINEERED_FEATURES) != 24 or len(set(ENGINEERED_FEATURES)) != 24:
    raise RuntimeError("Feature engineering must define exactly 24 unique features.")


def get_feature_documentation() -> pd.DataFrame:
    """Return formulas, sources and Week-1 justification for all 24 features."""

    return pd.DataFrame(FEATURE_DEFINITIONS).copy()


class BankMarketingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add 24 deterministic, pre-call features to a pandas DataFrame.

    The target is rejected explicitly.  Leakage-prone ``duration`` is expected
    to be removed by the integration schema guard before this transformer.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = self._validate_dataframe(X)
        self._validate_source_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)

        conflicts = sorted(set(ENGINEERED_FEATURES).intersection(X.columns))
        if conflicts:
            raise ValueError(
                "Engineered feature names already exist in input: "
                + ", ".join(conflicts)
            )

        self.feature_names_out_ = np.asarray(
            [*self.feature_names_in_, *ENGINEERED_FEATURES], dtype=object
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        X = self._validate_dataframe(X)

        fitted_columns = list(self.feature_names_in_)
        missing = sorted(set(fitted_columns).difference(X.columns))
        unexpected = sorted(set(X.columns).difference(fitted_columns))
        if missing or unexpected:
            raise ValueError(
                "Input schema differs from fit schema. "
                f"Missing={missing}; unexpected={unexpected}."
            )

        output = X.loc[:, fitted_columns].copy()
        self._validate_source_columns(output)

        age = self._numeric(output, "age")
        balance = self._numeric(output, "balance")
        pdays = self._numeric(output, "pdays")
        previous = self._numeric(output, "previous")
        campaign = self._numeric(output, "campaign")

        housing_yes = self._text(output["housing"]).eq("yes")
        personal_loan_yes = self._text(output["loan"]).eq("yes")
        default_yes = self._text(output["default"]).eq("yes")
        poutcome = self._text(output["poutcome"])
        contact = self._text(output["contact"])
        month = self._text(output["month"])

        output["age_group"] = pd.cut(
            age,
            bins=[-np.inf, 30, 40, 60, np.inf],
            labels=["under_30", "30_39", "40_59", "60_plus"],
            right=False,
        ).astype("string")
        output["age_squared"] = (age / 10.0) ** 2

        output["balance_positive_log"] = np.log1p(balance.clip(lower=0))
        output["balance_debt_log"] = np.log1p((-balance).clip(lower=0))
        output["has_negative_balance"] = balance.lt(0).astype("int8")
        output["balance_per_age"] = balance / age.clip(lower=1)

        output["any_loan"] = (housing_yes | personal_loan_yes).astype("int8")
        output["both_loans"] = (housing_yes & personal_loan_yes).astype("int8")
        output["credit_obligation_count"] = (
            housing_yes.astype("int8")
            + personal_loan_yes.astype("int8")
            + default_yes.astype("int8")
        )
        output["financial_stress_flag"] = (
            default_yes | balance.lt(0)
        ).astype("int8")
        output["loan_with_negative_balance"] = (
            (housing_yes | personal_loan_yes) & balance.lt(0)
        ).astype("int8")

        prior_contacted = pdays.ge(0) | previous.gt(0)
        valid_pdays = pdays.where(pdays.ge(0), 0)
        output["prior_contacted"] = prior_contacted.astype("int8")
        output["prior_contact_recency_log"] = np.log1p(valid_pdays)
        output["prior_contact_recent_30d"] = (
            pdays.ge(0) & pdays.le(30)
        ).astype("int8")
        output["previous_contacts_log"] = np.log1p(previous.clip(lower=0))
        output["previous_contact_intensity"] = np.where(
            pdays.ge(0), previous.clip(lower=0) / (valid_pdays + 1), 0.0
        )

        output["campaign_contacts_log"] = np.log1p(campaign.clip(lower=0))
        output["campaign_repeat_contact"] = campaign.gt(1).astype("int8")
        output["campaign_vs_previous_ratio"] = (
            campaign.clip(lower=0) / (previous.clip(lower=0) + 1)
        )

        output["successful_prior_contact_recency"] = np.where(
            poutcome.eq("success") & pdays.ge(0), 1.0 / (valid_pdays + 1), 0.0
        )
        output["failed_prior_campaign_recontact"] = (
            poutcome.eq("failure") & campaign.gt(1)
        ).astype("int8")
        output["contact_known"] = (
            contact.notna()
            & ~contact.isin(["", "unknown", "missing", "<na>"])
        ).astype("int8")

        month_number = month.map(MONTH_TO_NUMBER)
        valid_month = month_number.notna()
        angle = 2 * np.pi * (month_number.fillna(1).astype(float) - 1) / 12
        output["contact_month_sin"] = np.where(valid_month, np.sin(angle), 0.0)
        output["contact_month_cos"] = np.where(valid_month, np.cos(angle), 0.0)

        numeric_engineered = [
            feature for feature in ENGINEERED_FEATURES if feature != "age_group"
        ]
        values = output[numeric_engineered].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Feature engineering introduced NaN or infinity.")

        return output.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None):
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and list(input_features) != list(
            self.feature_names_in_
        ):
            raise ValueError("input_features do not match the fitted input schema.")
        return self.feature_names_out_.copy()

    @staticmethod
    def _validate_dataframe(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "BankMarketingFeatureEngineer requires a pandas DataFrame input."
            )
        if TARGET_COLUMN in X.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' must not enter feature engineering."
            )
        return X

    @staticmethod
    def _validate_source_columns(X: pd.DataFrame) -> None:
        missing = sorted(set(REQUIRED_SOURCE_COLUMNS).difference(X.columns))
        if missing:
            raise ValueError(
                "Missing source columns required for feature engineering: "
                + ", ".join(missing)
            )

    @staticmethod
    def _numeric(X: pd.DataFrame, column: str) -> pd.Series:
        try:
            values = pd.to_numeric(X[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{column}' must be numeric.") from exc
        if values.isna().any():
            raise ValueError(
                f"Column '{column}' still contains missing values before engineering."
            )
        return values

    @staticmethod
    def _text(values: pd.Series) -> pd.Series:
        return values.astype("string").str.strip().str.lower()


class MutualInformationSelector(BaseEstimator, TransformerMixin):
    """Select exactly top-K encoded features by training-only mutual information.

    This is the sole feature-selection strategy used in Week 2.  MI is suited
    to a binary target and can capture non-linear associations after mixed
    numerical/categorical features have been encoded.  ``random_state`` makes
    sklearn's continuous-variable estimation deterministic.
    """

    def __init__(
        self,
        k: int | str = 32,
        random_state: int = RANDOM_STATE,
        discrete_prefixes: tuple[str, ...] = (
            "ordinal_education__",
            "onehot_nominal__",
            "onehot_engineered__",
            "engineered_binary__",
        ),
    ):
        self.k = k
        self.random_state = random_state
        self.discrete_prefixes = discrete_prefixes

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("MutualInformationSelector.fit requires y_train.")

        frame, was_dataframe = self._as_frame(X)
        if TARGET_COLUMN in frame.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' is present in X.")
        if len(frame) != len(y):
            raise ValueError("X and y have different row counts.")

        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Mutual information input contains NaN or infinity.")

        n_features = frame.shape[1]
        if self.k == "all":
            k = n_features
        elif isinstance(self.k, int) and not isinstance(self.k, bool):
            if not 1 <= self.k <= n_features:
                raise ValueError(
                    f"k must be between 1 and {n_features}; received {self.k}."
                )
            k = self.k
        else:
            raise ValueError("k must be a positive integer or 'all'.")

        target = pd.Series(y).reset_index(drop=True)
        if target.isna().any():
            raise ValueError("y_train contains missing labels.")
        target_codes, classes = pd.factorize(target, sort=True)
        if len(classes) < 2:
            raise ValueError("Mutual information requires at least two target classes.")

        feature_names = np.asarray(frame.columns.astype(str), dtype=object)
        discrete_mask = np.asarray(
            [
                any(name.startswith(prefix) for prefix in self.discrete_prefixes)
                for name in feature_names
            ],
            dtype=bool,
        )
        scores = mutual_info_classif(
            values,
            target_codes,
            discrete_features=discrete_mask,
            random_state=self.random_state,
        )
        if not np.isfinite(scores).all():
            raise ValueError("Mutual information produced a non-finite score.")

        ranking = np.lexsort((np.arange(n_features), -scores))
        selected_indices = np.sort(ranking[:k])

        self.n_features_in_ = n_features
        self.feature_names_in_ = feature_names
        self.input_was_dataframe_ = was_dataframe
        self.fit_row_count_ = len(frame)
        self.target_class_counts_ = target.value_counts().sort_index().to_dict()
        self.discrete_mask_ = discrete_mask
        self.scores_ = np.asarray(scores, dtype=float)
        self.selected_indices_ = selected_indices
        self.selected_feature_names_ = feature_names[selected_indices]
        self.classes_ = np.asarray(classes, dtype=object)
        self.k_ = k
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        frame, was_dataframe = self._as_frame(X)

        current_names = list(frame.columns.astype(str))
        fitted_names = list(self.feature_names_in_)
        if set(current_names) != set(fitted_names):
            missing = sorted(set(fitted_names).difference(current_names))
            unexpected = sorted(set(current_names).difference(fitted_names))
            raise ValueError(
                "Encoded schema differs from fit schema. "
                f"Missing={missing}; unexpected={unexpected}."
            )
        frame = frame.loc[:, fitted_names]

        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Selector transform input contains NaN or infinity.")

        selected = frame.iloc[:, self.selected_indices_]
        if was_dataframe:
            return selected.copy()
        return selected.to_numpy()

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "selected_indices_")
        if indices:
            return self.selected_indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_feature_names_out(self, input_features: Sequence[str] | None = None):
        check_is_fitted(self, "selected_feature_names_")
        if input_features is not None and list(input_features) != list(
            self.feature_names_in_
        ):
            raise ValueError("input_features do not match the fitted encoded schema.")
        return self.selected_feature_names_.copy()

    def get_selection_summary(self) -> pd.DataFrame:
        """Return all MI scores and selection flags, highest score first."""

        check_is_fitted(self, "scores_")
        selected = set(self.selected_feature_names_)
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_in_,
                    "mutual_information": self.scores_,
                    "discrete": self.discrete_mask_,
                    "selected": [name in selected for name in self.feature_names_in_],
                }
            )
            .sort_values(
                ["mutual_information", "feature"],
                ascending=[False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _as_frame(X) -> tuple[pd.DataFrame, bool]:
        if isinstance(X, pd.DataFrame):
            return X.copy(), True
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("Selector input must be a two-dimensional matrix.")
        columns = [f"x{i}" for i in range(values.shape[1])]
        return pd.DataFrame(values, columns=columns), False


__all__ = [
    "BankMarketingFeatureEngineer",
    "ENGINEERED_FEATURES",
    "FEATURE_DEFINITIONS",
    "MutualInformationSelector",
    "get_feature_documentation",
]
