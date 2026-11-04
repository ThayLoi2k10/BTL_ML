"""Kiểm thử tự động cho mô-đun suy luận dòng lệnh src/predict.py."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.config import CHAMPION_MODEL_PATH, PREPROCESSING_PIPELINE_PATH, SAMPLE_TEST_PATH
from src.predict import load_artifacts, parse_arguments, predict


class PredictTests(unittest.TestCase):
    def test_parse_arguments_defaults(self):
        args = parse_arguments([])
        self.assertEqual(args.input, str(SAMPLE_TEST_PATH))
        self.assertEqual(args.pipeline, str(PREPROCESSING_PIPELINE_PATH))
        self.assertEqual(args.model, str(CHAMPION_MODEL_PATH))

    def test_load_artifacts_success(self):
        pipeline, model = load_artifacts()
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(pipeline, "transform"))
        self.assertTrue(hasattr(model, "predict"))

    def test_predict_on_sample_data(self):
        pipeline, model = load_artifacts()
        sample_df = pd.read_csv(SAMPLE_TEST_PATH)
        predictions_df = predict(sample_df, pipeline, model)

        self.assertEqual(len(predictions_df), len(sample_df))
        self.assertIn("predicted_deposit", predictions_df.columns)
        self.assertIn("probability_yes", predictions_df.columns)
        self.assertIn("confidence_score", predictions_df.columns)

        # Kiểm tra giá trị nhãn hợp lệ (chỉ có 'yes' hoặc 'no')
        unique_labels = set(predictions_df["predicted_deposit"].unique())
        self.assertTrue(unique_labels.issubset({"yes", "no"}))

        # Kiểm tra xác suất và độ tin cậy nằm trong khoảng [0, 1]
        self.assertTrue((predictions_df["probability_yes"] >= 0.0).all())
        self.assertTrue((predictions_df["probability_yes"] <= 1.0).all())
        self.assertTrue((predictions_df["confidence_score"] >= 0.5).all())
        self.assertTrue((predictions_df["confidence_score"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main()

