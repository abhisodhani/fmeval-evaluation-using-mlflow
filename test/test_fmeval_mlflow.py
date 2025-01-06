import os
from pathlib import Path
import tempfile
from typing import Any
import pytest
import pandas as pd
import json
from unittest.mock import Mock, patch
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.eval_algorithms import EvalOutput, EvalScore
from fmeval.data_loaders.data_config import DataConfig
from fmeval_mlflow.fmeval_mlflow import (
    _get_parameters_from_runner,
    _create_dataset_input,
    get_metrics_from_experiment,
    get_metrics_from_runs,
    log_input_dataset,
)
from fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from fmeval.constants import MIME_TYPE_JSONLINES
from mlflow.data.pandas_dataset import PandasDataset


def test_get_parameters_from_runner():
    # Setup
    mock_runner = Mock(spec=ModelRunner)
    mock_runner._model_id = "test-model"
    mock_runner._content_template = {"temperature": 0.7, "maxTokens": 100}
    mock_runner._composer = None

    # Test basic parameters extraction
    params = _get_parameters_from_runner(mock_runner)
    assert params["model_id"] == "test-model"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 100

    # Test with custom parameters map
    custom_map = {"maxTokens": "max_length"}
    params = _get_parameters_from_runner(mock_runner, custom_map)
    assert params["max_length"] == 100

    # Test with JumpStartModelRunner
    mock_jumpstart = Mock(spec=JumpStartModelRunner)
    mock_jumpstart._model_id = "jumpstart-model"
    mock_jumpstart._endpoint_name = "test-endpoint"
    mock_jumpstart._content_template = {"temperature": 0.8}
    mock_jumpstart._composer = None

    params = _get_parameters_from_runner(mock_jumpstart)
    assert params["model_id"] == "jumpstart-model"
    assert params["endpoint_name"] == "test-endpoint"
    assert params["temperature"] == 0.8


def test_get_parameters_basic():
    """Test basic parameter extraction with minimal model runner setup."""
    mock_runner = Mock(spec=ModelRunner)
    mock_runner._model_id = "test-model-123"
    mock_runner._content_template = {"temperature": 0.7}
    mock_runner._composer = None

    params = _get_parameters_from_runner(mock_runner)

    assert params["model_id"] == "test-model-123"
    assert params["temperature"] == 0.7


def test_get_parameters_with_custom_map():
    """Test parameter extraction with custom parameter mapping."""
    mock_runner = Mock(spec=ModelRunner)
    mock_runner._model_id = "test-model-123"
    mock_runner._content_template = {"temperature": 0.7, "top_p": 0.9}
    mock_runner._composer = None

    custom_map = {"temperature": "temp", "top_p": "nucleus_sampling"}

    params = _get_parameters_from_runner(mock_runner, custom_map)

    assert params["model_id"] == "test-model-123"
    assert params["temp"] == 0.7
    assert params["nucleus_sampling"] == 0.9


def test_get_parameters_jumpstart_runner():
    """Test parameter extraction with JumpStartModelRunner."""
    mock_runner = Mock(spec=JumpStartModelRunner)
    mock_runner._model_id = "test-model-123"
    mock_runner._content_template = {"temperature": 0.7}
    mock_runner._endpoint_name = "test-endpoint"
    mock_runner._composer = None

    params = _get_parameters_from_runner(mock_runner)

    assert params["model_id"] == "test-model-123"
    assert params["temperature"] == 0.7
    assert params["endpoint_name"] == "test-endpoint"


def test_get_parameters_with_composer():
    """Test parameter extraction when using composer."""
    mock_runner = Mock(spec=ModelRunner)
    mock_runner._model_id = "test-model-123"
    mock_runner._content_template = None

    mock_composer = Mock()
    mock_composer.compose.return_value = Mock(body={"temperature": 0.7})
    mock_runner._composer = mock_composer

    params = _get_parameters_from_runner(mock_runner)

    assert params["model_id"] == "test-model-123"
    assert params["temperature"] == 0.7
    assert mock_composer.compose.called_with("prompt", None)



@pytest.fixture
def sample_eval_output(sample_jsonl_file: str):
    return EvalOutput(
        eval_name="test_eval",
        dataset_name="test_dataset",
        dataset_scores=[
            EvalScore(name="score1", value=0.8),
            EvalScore(name="score2", value=0.9),
        ],
        output_path=sample_jsonl_file,
    )


@pytest.fixture
def sample_jsonl_file():
    """Fixture to create a temporary JSONL file with test data."""
    data = [
        {
            "id": 1,
            "target_output": "yes",
            "model_output": "yes",
            "scores": [
                {"name": "accuracy", "value": 1.0},
                {"name": "f1", "value": 0.95},
            ],
        },
        {
            "id": 2,
            "target_output": "no",
            "model_output": "no",
            "scores": [
                {"name": "accuracy", "value": 1.0},
                {"name": "f1", "value": 0.92},
            ],
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


def test_basic_dataset_creation(sample_jsonl_file: str):
    """Test basic dataset creation with default column names."""
    eval_output = EvalOutput(
        eval_name="test_eval",
        output_path=sample_jsonl_file,
        dataset_name="test_dataset",
        dataset_scores=[
            EvalScore(name="score1", value=0.8),
            EvalScore(name="score2", value=0.9),
        ],
    )

    result = _create_dataset_input(eval_output)

    assert isinstance(result, PandasDataset)
    assert "target_output" in result.df.columns
    assert "model_output" in result.df.columns
    assert "score_accuracy" in result.df.columns
    assert "score_f1" in result.df.columns
    assert (
        len(result.df.columns) == 5
    )  #'id', 'target_output', 'model_output', 'score_accuracy', 'score_f1'


def test_custom_column_names(sample_eval_output: EvalOutput):
    """Test dataset creation with custom column names."""
    eval_output = sample_eval_output

    result = _create_dataset_input(
        eval_output,
        target_column_name="target_output",
        model_output_column_name="model_output",
    )

    assert isinstance(result, PandasDataset)
    assert result.targets == "target_output"
    assert result.predictions == "model_output"


def test_missing_file():
    """Test handling of non-existent file."""

    eval_output = EvalOutput(
        eval_name="test_eval",
        output_path="nonexistent.jsonl",
        dataset_name="test_dataset",
        dataset_scores=[
            EvalScore(name="score1", value=0.8),
            EvalScore(name="score2", value=0.9),
        ],
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        _create_dataset_input(eval_output)
    assert "Could not find file" in str(exc_info.value)


def test_invalid_json(tmp_path: Path):
    """Test handling of invalid JSON content."""
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text("invalid json content\n")

    eval_output = EvalOutput(
        eval_name="test_eval",
        output_path=str(invalid_file),
        dataset_name="test_dataset",
        dataset_scores=[
            EvalScore(name="score1", value=0.8),
            EvalScore(name="score2", value=0.9),
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        _create_dataset_input(eval_output)
    assert "Invalid JSON format" in str(exc_info.value)


def test_missing_columns(tmp_path: Path):
    """Test handling of missing required columns."""
    data = [{"id": 1, "other_column": "value"}]
    test_file = tmp_path / "test.jsonl"

    with open(test_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    eval_output = EvalOutput(
        eval_name="test_eval",
        output_path=str(test_file),
        dataset_name="test_dataset",
        dataset_scores=[
            EvalScore(name="score1", value=0.8),
            EvalScore(name="score2", value=0.9),
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        _create_dataset_input(eval_output, target_column_name="missing_column")
    assert "Target column 'missing_column' not found" in str(exc_info.value)


def test_score_processing(sample_jsonl_file: str):
    """Test proper processing of scores into separate columns."""
    eval_output = EvalOutput(
        eval_name="test_eval",
        output_path=sample_jsonl_file,
        dataset_name="test_dataset",
        dataset_scores=[
            EvalScore(name="score1", value=0.8),
            EvalScore(name="score2", value=0.9),
        ],
    )

    result = _create_dataset_input(eval_output)

    assert "score_accuracy" in result.df.columns
    assert "score_f1" in result.df.columns
    assert result.df["score_accuracy"].iloc[0] == 1.0
    assert result.df["score_f1"].iloc[0] == 0.95


def test_invalid_output_path_type():
    """Test handling of invalid output_path type."""
    eval_output = EvalOutput(
        eval_name="test_eval",
        output_path=123,  # type: ignore
        dataset_name="test_dataset",
        dataset_scores=[
            EvalScore(name="score1", value=0.8),
            EvalScore(name="score2", value=0.9),
        ],
    )

    with pytest.raises(ValueError) as exc_info:
        _create_dataset_input(eval_output)
    assert "Output path must be a string" in str(exc_info.value)


@pytest.mark.parametrize(
    "scores,expected_columns",
    [
        ([], {"id", "target_output", "model_output"}),
        (
            [{"name": "metric1", "value": 0.5}],
            {"id", "target_output", "model_output", "score_metric1"},
        ),
    ],
)
def test_different_score_configurations(
    tmp_path: Path, scores: list[dict[str, Any]], expected_columns: set[str]
):
    """Test dataset creation with different score configurations."""
    data = [{"id": 1, "target_output": "yes", "model_output": "yes", "scores": scores}]

    test_file = tmp_path / "test.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    eval_output = EvalOutput(
        eval_name="test_eval",
        output_path=str(test_file),
        dataset_name="test_dataset",
        dataset_scores=scores, # type: ignore
    )

    result = _create_dataset_input(eval_output)

    assert set(result.df.columns) - {"scores"} == expected_columns


@pytest.fixture
def sample_data_config():
    return DataConfig(
        dataset_name="test_dataset",
        dataset_uri="test.jsonl",
        target_output_location="target_output",
        model_output_location="model_output",
        dataset_mime_type=MIME_TYPE_JSONLINES,
    )


def test_log_input_dataset(sample_data_config: DataConfig):
    mock_df = pd.DataFrame(
        {
            "input": ["test"],
            "target_output": ["expected"],
            "model_output": ["predicted"],
        }
    )

    with patch("pandas.read_json") as mock_read_json, patch(
        "mlflow.log_input"
    ) as mock_log_input:
        mock_read_json.return_value = mock_df

        log_input_dataset(sample_data_config)

        mock_read_json.assert_called_once_with(
            sample_data_config.dataset_uri, lines=True
        )
        mock_log_input.assert_called_once()


def test_get_metrics_from_runs():
    mock_runs = pd.DataFrame(
        {
            "metrics.eval1.score1": [0.8],
            "metrics.eval1.score2": [0.9],
            "params.model_id": ["test-model"],
        }
    )

    with patch("mlflow.search_runs") as mock_search:
        mock_search.return_value = mock_runs

        result = get_metrics_from_runs("test-parent-run")

        assert result.index[0] == "test-model"
        assert "eval1.score1" in result.columns
        assert "eval1.score2" in result.columns

        mock_search.assert_called_once_with(
            filter_string="tags.mlflow.parentRunId = 'test-parent-run'"
        )


def test_get_metrics_from_experiment():
    mock_runs = pd.DataFrame(
        {
            "metrics.eval1.score1": [0.8],
            "metrics.eval1.score2": [0.9],
            "params.model_id": ["test-model"],
            "run_id": ["test-run"],
        }
    )

    with patch("mlflow.search_runs") as mock_search:
        mock_search.return_value = mock_runs

        result = get_metrics_from_experiment("test-experiment")

        assert "evaluation" in result.columns
        assert "metric" in result.columns
        assert "value" in result.columns

        mock_search.assert_called_once_with(experiment_names=["test-experiment"])
