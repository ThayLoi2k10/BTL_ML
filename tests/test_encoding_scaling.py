import unittest

import pandas as pd

from src.encoding_scaling import (
    build_preprocessor,
    get_feature_configuration_summary,
    get_transformer_configurations,
    split_data,
)


def make_sample_data():
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 30, 40, 50, 60, 28, 38],
            "job": ["admin.", "technician", "services", "management", "admin.", "blue-collar", "retired", "student", "admin.", "technician"],
            "marital": ["single", "married", "married", "divorced", "single", "married", "married", "single", "single", "married"],
            "education": ["primary", "secondary", "tertiary", "secondary", "primary", "tertiary", "secondary", "primary", "tertiary", "secondary"],
            "default": ["no", "no", "no", "yes", "no", "no", "yes", "no", "no", "no"],
            "balance": [100, 2000, -50, 50000, 300, 700, 9000, 120, -300, 4500],
            "housing": ["yes", "no", "yes", "no", "yes", "yes", "no", "no", "yes", "no"],
            "loan": ["no", "yes", "no", "no", "yes", "no", "yes", "no", "no", "yes"],
            "contact": ["cellular", "unknown", "telephone", "cellular", "unknown", "cellular", "telephone", "cellular", "unknown", "cellular"],
            "day": [1, 5, 10, 15, 20, 25, 30, 3, 8, 12],
            "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"],
            "duration": [50, 100, 300, 600, 90, 250, 1000, 40, 180, 500],
            "campaign": [1, 2, 1, 5, 3, 1, 10, 2, 1, 4],
            "pdays": [-1, 10, -1, 200, 30, -1, 400, 15, -1, 90],
            "previous": [0, 1, 0, 3, 2, 0, 5, 1, 0, 2],
            "poutcome": ["unknown", "success", "failure", "other", "unknown", "success", "failure", "other", "unknown", "success"],
            "deposit": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        }
    )


class EncodingScalingTests(unittest.TestCase):
    def test_split_data_uses_reproducible_stratified_80_20_split(self):
        data = make_sample_data()

        X_train, X_test, y_train, y_test = split_data(data)

        self.assertEqual(len(X_train), 8)
        self.assertEqual(len(X_test), 2)
        self.assertNotIn("deposit", X_train.columns)
        self.assertEqual(y_train.value_counts().to_dict(), {"yes": 4, "no": 4})
        self.assertEqual(y_test.value_counts().to_dict(), {"yes": 1, "no": 1})

    def test_transformer_configurations_are_column_transformer_compatible(self):
        configs = get_transformer_configurations()

        self.assertEqual(
            [name for name, _, _ in configs],
            [
                "ordinal_education",
                "onehot_nominal",
                "robust_numeric",
                "standard_numeric",
            ],
        )
        self.assertEqual(configs[1][1].handle_unknown, "ignore")
        self.assertEqual(configs[0][1].unknown_value, -1)

    def test_preprocessor_fits_train_and_transforms_unseen_test_labels(self):
        data = make_sample_data()
        X_train, X_test, _, _ = split_data(data)
        preprocessor = build_preprocessor()

        transformed_train = preprocessor.fit_transform(X_train)
        X_test = X_test.copy()
        X_test.loc[X_test.index[0], "job"] = "new-job-label"
        X_test.loc[X_test.index[0], "education"] = "unknown-education"
        transformed_test = preprocessor.transform(X_test)

        self.assertEqual(transformed_train.shape[0], len(X_train))
        self.assertEqual(transformed_test.shape[0], len(X_test))
        self.assertEqual(transformed_train.shape[1], transformed_test.shape[1])

    def test_feature_configuration_summary_is_human_readable(self):
        summary = get_feature_configuration_summary()

        self.assertEqual(list(summary.columns), ["Nhóm cột", "Cột", "Bộ chuyển đổi", "Lý do"])
        self.assertEqual(len(summary), 4)


if __name__ == "__main__":
    unittest.main()
