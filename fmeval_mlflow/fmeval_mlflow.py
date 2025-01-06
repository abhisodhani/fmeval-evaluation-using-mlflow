import json
import re
from collections import ChainMap
from typing import Any, Generator

import mlflow
import mlflow.data.filesystem_dataset_source
import pandas as pd
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalOutput
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from fmeval.model_runners.sm_model_runner import SageMakerModelRunner
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner
from mlflow.data.pandas_dataset import PandasDataset

# Derived from https://stackoverflow.com/a/1176023/2109965
_camel_pattern = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def to_snake_case(name):
    """
    Convert a string from camel case to snake case.
    """
    return _camel_pattern.sub("_", name).lower()


# Define common aliases for common parameters
_default_parameters_map = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_k": "top_k",
    "top_p": "top_p",
    "stop_sequences": "stop_sequences",
    "system_prompt": "system_prompt",
    "system": "system_prompt",
    "tools": "tools",
    "max_length": "max_tokens",
    "max_new_tokens": "max_tokens",
    "max_token_count": "max_tokens",
    "do_sample": "do_sample",
    "stop": "stop_sequences",
}


def _extract_params(
    keys: str | list[str], var: dict | list | Any
) -> Generator[dict[str, Any], None, None]:
    """
    Recursively extract values from a nested dictionary based on provided keys.

    Args:
        keys (str | List[str]): A string or sequence of strings representing the keys to extract.
        var: The variable to extract parameters from. Can be a dictionary, list, or any other type

    Yields:
        Generator: A dictionary containing the extracted key-value pairs.

    Examples:
        >>> data = {"temperature": 0.7, "nested": {"max_tokens": 100}}
        >>> list(_extract_params(["temperature", "max_tokens"], data))
        [{'temperature': 0.7}, {'max_tokens': 100}]

    Raises:
        TypeError: If keys is neither a string nor a list

    """
    if not isinstance(keys, (str, list)):
        raise TypeError("keys must be either a string or list of strings")

    if isinstance(keys, str):
        keys = [keys]

    if isinstance(var, dict):
        for k, v in var.items():
            if (k := to_snake_case(k)) in keys:
                yield {k: v}
            if isinstance(v, dict):
                yield from _extract_params(keys, v)
            elif isinstance(v, list):
                yield from _extract_params(keys, v)


def _runner_parameters(
    content_template: str | dict[str, Any],
    custom_parameters_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Generate a dictionary of parameters for a runner based on a content template and custom parameters.

    Args:
        content_template (str): The content template string.
        custom_parameters_map (Dict[str, str] | None, optional): A dictionary of custom parameters to include.

    Returns:
        dict[str, Any]: A dictionary of normalized parameters for the runner.

    Raises:
        json.JSONDecodeError: If content_template is a string and contains invalid JSON
        KeyError: If a required parameter key is missing in the parameters map
    """
    parameters_map = _default_parameters_map.copy()
    if custom_parameters_map is not None:
        parameters_map.update(
            {to_snake_case(k): v for k, v in custom_parameters_map.items()}
        )
    ctj = content_template
    if isinstance(content_template, str):
        if "$prompt" in content_template:
            content_template = content_template.replace("$prompt", '"prompt"')
            ctj = json.loads(content_template)
    parameters = dict(ChainMap(*_extract_params(list(parameters_map.keys()), ctj)))
    normalized_parameters = {parameters_map[k]: v for k, v in parameters.items()}

    return normalized_parameters


def _get_parameters_from_runner(
    model_runner: ModelRunner,
    custom_parameters_map: dict | None = None,
    model_id: str | None = None,
) -> dict:
    """
    Retrieve the parameters associated with a given ModelRunner instance.

    Args:
        model_runner (ModelRunner): An instance of the ModelRunner class or its subclasses.
        custom_parameters_map (dict | None, optional): A dictionary of custom parameters
            to override or extend the default parameters. If provided, the custom parameters will
            take precedence over the default parameters.
        model_id (str | None, optional): The model identifier to override the default model ID
            from the model runner.

    Returns:
        dict[str, Any]: A dictionary containing the parameters for the given model runner.
            The dictionary includes the following keys:
                - 'model_id': The ID of the model associated with the runner.
                - 'endpoint_name' (optional): The name of the endpoint, if the runner is a
                    JumpStartModelRunner instance.
                - parameters extracted from the content template, using the default and custom
                    parameter mappings. For example, if the content template contains a key
                    'temperature', the returned dictionary will include a key 'temperature'
                    with the corresponding value from the content template.

    Raises:
        AttributeError: If the model_runner instance doesn't have required attributes or
            if the _composer attribute is missing.
        TypeError: If model_runner is not an instance of ModelRunner or its subclasses.
    """
    if not isinstance(model_runner, ModelRunner):
        raise TypeError(
            "model_runner must be an instance of ModelRunner or its subclasses"
        )

    runner_parameters = {
        k.lstrip("_"): v
        for k, v in model_runner.__dict__.items()
        if isinstance(v, (dict, str))
    }

    if model_id:
        runner_parameters.update({"model_id": model_id})

    if (
        val := runner_parameters.get("content_template")
        or model_runner._composer.compose("prompt").body  # type: ignore
    ):
        runner_parameters.update({"content_template": val})
        runner_parameters.update(_runner_parameters(val, custom_parameters_map) or {})

        # Determine endpoint type
        endpoint_type = "unknown"
        if isinstance(model_runner, JumpStartModelRunner):
            endpoint_type = "SageMaker Jumpstart"
        elif isinstance(model_runner, BedrockModelRunner):
            endpoint_type = "Bedrock"
        elif isinstance(model_runner, SageMakerModelRunner):
            endpoint_type = "SageMaker"

        runner_parameters["endpoint_type"] = endpoint_type

    return runner_parameters


def log_runner_parameters(
    model_runner: ModelRunner,
    custom_parameters_map: dict | None = None,
    model_id: str | None = None,
):
    """
    Log the parameters associated with a given ModelRunner instance to MLflow.

    Args:
        model_runner (ModelRunner): An instance of the ModelRunner class or its subclasses.
            Used to extract model parameters and configuration.
        custom_parameters_map (dict | None, optional): A dictionary of custom parameters
            to override or extend the default parameters. If provided, these parameters
            will take precedence over the default ones. Defaults to None.
        model_id (str | None, optional): The model identifier to override the default
            model ID from the model runner. If not provided, uses the model_id from
            the model_runner. Defaults to None.

    Returns:
        None

    Raises:
        RuntimeError: If there is no active MLflow run when attempting to log parameters.
        AttributeError: If the model_runner instance doesn't have required attributes.
        TypeError: If model_runner is not an instance of ModelRunner or its subclasses.

    """

    if mlflow.active_run():
        params = _get_parameters_from_runner(
            model_runner, custom_parameters_map, model_id=model_id
        )
        mlflow.log_params(params)


def _create_dataset_input(
    single_eval_output: EvalOutput,
    target_column_name: str | None = None,
    model_output_column_name: str | None = None,
) -> PandasDataset:
    """
    Create a MLflow PandasDataset from an evaluation output, with support for scores and metrics.

    This function processes an EvalOutput instance by:
    1. Reading and parsing the JSON lines output file
    2. Converting scores into separate columns with appropriate prefixes
    3. Creating a MLflow dataset with specified target and prediction columns

    Args:
        single_eval_output (EvalOutput): An evaluation output instance containing:
            - output_path: Path to the JSON lines file with evaluation results
            - dataset_name: Name identifier for the dataset
            - scores: List of dictionaries containing metric names and values
        target_column_name (Optional[str]): Name of the column containing ground truth/target values.
            If None, defaults to "target_output" if it exists in the data.
        model_output_column_name (Optional[str]): Name of the column containing model predictions.
            If None, defaults to "model_output" if it exists in the data.

    Returns:
        PandasDataset: A MLflow dataset containing:
            - All original columns from the evaluation output
            - Score columns prefixed with "score_"
            - Specified target and prediction columns

    Raises:
        AssertionError: If output_path is not a string or doesn't exist
        FileNotFoundError: If the output file cannot be read
        JSONDecodeError: If the file contains invalid JSON
        ValueError: If required columns are missing from the data
    """
    # Validate input
    if not isinstance(single_eval_output.output_path, str):
        raise ValueError("Output path must be a string")

    try:
        # Read and parse JSON lines file
        with open(single_eval_output.output_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find file: {single_eval_output.output_path}"
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON format in file: {single_eval_output.output_path}"
        )

    df = pd.DataFrame(data)

    # Process scores if they exist
    if "scores" in df.columns:
        try:
            # Convert scores to separate columns
            scores_df = pd.DataFrame.from_records(
                df.pop("scores")
                .apply(lambda x: {k["name"]: k["value"] for k in x})
                .tolist()
            ).add_prefix("score_")

            # Combine with main DataFrame
            df = pd.concat([df, scores_df], axis=1)
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Error processing scores: {str(e)}")

    # Determine column names
    target_column_name = target_column_name or (
        "target_output" if "target_output" in df.columns else None
    )

    model_output_column_name = model_output_column_name or (
        "model_output" if "model_output" in df.columns else None
    )

    # Validate required columns exist
    if target_column_name and target_column_name not in df.columns:
        raise ValueError(f"Target column '{target_column_name}' not found in data")
    if model_output_column_name and model_output_column_name not in df.columns:
        raise ValueError(
            f"Model output column '{model_output_column_name}' not found in data"
        )

    # Create MLflow dataset
    mlf_data = mlflow.data.from_pandas(  # type: ignore
        df=df,
        targets=target_column_name,
        predictions=model_output_column_name,
        name=single_eval_output.dataset_name,
        source=single_eval_output.output_path,
    )
    return mlf_data


def log_metrics(eval_output: list[EvalOutput], log_eval_output_artifact: bool = False):
    """
    Log metrics and artifacts for a list of SingleEvalOutput instances to MLflow.

    Args:
        eval_output (List[EvalOutput]): A list of EvalOutput instances containing
            evaluation results.

    Returns:
        None
    """

    if log_eval_output_artifact:
        for eval_output_instance in eval_output:
            df = _create_dataset_input(eval_output_instance).df
            dataset_name = eval_output_instance.dataset_name.replace(" ", "_")

            mlflow.log_table(data=df, artifact_file=f"{dataset_name}.json")

    metrics = {
        f"{eval_output_instance.eval_name}.{evaluation_score.name}": evaluation_score.value
        for eval_output_instance in eval_output
        if eval_output_instance.dataset_scores is not None
        for evaluation_score in eval_output_instance.dataset_scores
        if evaluation_score.value is not None
    }
    mlflow.log_metrics(metrics)


def log_input_dataset(data_config: DataConfig | list[DataConfig]):
    """
    Log one or more input datasets to MLflow for evaluation purposes.

    This function reads dataset(s) specified in the DataConfig(s) and logs them to MLflow
    along with their associated target and model outputs. Each dataset is logged as a
    pandas DataFrame with specified target and prediction columns.

    Args:
        data_config: Configuration for the dataset(s) to be logged. Can be either:
            - A single DataConfig instance containing:
                * dataset_uri: Path to the dataset file (JSON lines format)
                * target_output_location: Column name for ground truth/target values
                * model_output_location: Column name for model predictions
                * dataset_name: Name to identify the dataset in MLflow
            - A list of DataConfig instances to log multiple datasets

    Returns:
        None

    Raises:
        FileNotFoundError: If the dataset file specified in dataset_uri cannot be found
        JSONDecodeError: If the dataset file is not valid JSON lines format
        ValueError: If required fields in DataConfig are missing or invalid

    Example:
        >>> # Single dataset
        >>> config = DataConfig(
        ...     dataset_uri="data/test.jsonl",
        ...     target_output_location="target",
        ...     model_output_location="prediction",
        ...     dataset_name="test_data"
        ... )
        >>> log_input_dataset(config)

        >>> # Multiple datasets
        >>> configs = [
        ...     DataConfig(dataset_uri="data/train.jsonl", dataset_name="train"),
        ...     DataConfig(dataset_uri="data/test.jsonl", dataset_name="test")
        ... ]
        >>> log_input_dataset(configs)
    """
    if not isinstance(data_config, list):
        data_config = [data_config]

    for data_config in data_config:
        df = pd.read_json(data_config.dataset_uri, lines=True)
        mlflow.log_input(
            mlflow.data.from_pandas(  # type: ignore
                df,
                targets=data_config.target_output_location,
                predictions=data_config.model_output_location,
                name=data_config.dataset_name,
                source=data_config.dataset_uri,
            ),
            context="evaluation",
        )


def track_evaluation(
    data_config: DataConfig | list[DataConfig] | None = None,
    model_runner: ModelRunner | None = None,
    eval_output: list[EvalOutput] | None = None,
    custom_parameters_map: dict[str, str] | None = None,
    log_eval_output_artifact: bool = False,
    model_id: str | None = None,
):
    """
    Track the evaluation of a model on a dataset using MLflow.

    This function logs various aspects of a model evaluation to MLflow, including:
    - Input dataset configuration
    - Model runner parameters
    - Evaluation metrics and outputs

    Args:
        data_config: Configuration for the dataset(s) used in evaluation.
            Can be either a single DataConfig instance or a list of DataConfig instances.
            If None, dataset tracking is skipped.

        model_runner: The ModelRunner instance used for generating predictions.
            Contains information about the model configuration and parameters.
            If None, model parameter tracking is skipped.

        eval_output: List of evaluation results from running evaluation algorithms.
            Each EvalOutput instance contains metrics and other evaluation information.
            If None, metrics tracking is skipped.

        custom_parameters_map: Additional custom parameters to log with the evaluation.
            Keys and values should be strings. These parameters will be logged
            alongside the model runner parameters.

        log_eval_output_artifact: Whether to log the complete eval_output as an MLflow artifact.
            Defaults to False. When True, the entire evaluation output will be saved
            as a trackable artifact in MLflow.

        model_id: Optional model identifier to override the default model ID from the model runner.

    Returns:
        None

    Raises:
        RuntimeError: If there is no active MLflow run.
        TypeError: If input parameters are of incorrect type.
        ValueError: If input parameters contain invalid values.
    """
    if not mlflow.active_run():
        raise RuntimeError(
            "No active MLflow run found. Please start a run using 'mlflow.start_run()'"
        )

    if data_config is not None:
        log_input_dataset(data_config)
    if model_runner is not None:
        log_runner_parameters(model_runner, custom_parameters_map, model_id=model_id)
    if eval_output is not None:
        log_metrics(eval_output, log_eval_output_artifact)


def get_metrics_from_runs(parent_run_id: str) -> pd.DataFrame:
    """
    Extract metrics from all MLflow child runs associated with a specified parent run ID.

    This function searches for all child runs associated with a given parent run ID,
    extracts their metrics and the model_id parameter, and returns them in a structured
    DataFrame format. The metrics and parameter prefixes ('metrics.' and 'params.') are
    removed from the column names for clarity.

    Args:
        parent_run_id (str): The unique identifier of the parent MLflow run. All child
            runs associated with this parent run will be included in the results.

    Returns:
        pd.DataFrame: A DataFrame containing metrics from all child runs, with:
            - Index: model_id from the run parameters
            - Columns: All metrics from the runs, with 'metrics.' prefix removed

    Raises:
        AssertionError: If the MLflow search results cannot be converted to a DataFrame

    Example:
        >>> metrics_df = get_metrics_from_runs("parent_run_123")
        >>> print(metrics_df.head())
              accuracy  precision  recall
    model_id
    model_1   0.95      0.94      0.93
    model_2   0.92      0.91      0.90
    """
    filter_string = f"tags.mlflow.parentRunId = '{parent_run_id}'"
    runs = mlflow.search_runs(filter_string=filter_string)
    assert isinstance(runs, pd.DataFrame)

    df = pd.DataFrame()
    for run in runs.iterrows():
        df = pd.concat(
            [
                df,
                run[1]
                .filter(regex=("^(metrics\\..+|params\\.model_id)$"))
                .to_frame()
                .T,
            ]
        )

    # Remove prefix from column names for clarity
    df.rename(columns=lambda x: x.replace("metrics.", ""), inplace=True)
    df.rename(columns=lambda x: x.replace("params.", ""), inplace=True)

    return df.set_index("model_id")


def get_metrics_from_experiment(experiment_name: str) -> pd.DataFrame:
    """
    Retrieve and transform metrics from an MLflow experiment into a structured DataFrame.

    This function searches for runs in the specified MLflow experiment and transforms
    the metrics data into a melted DataFrame format, splitting metric names into
    evaluation and metric components.

    Args:
        experiment_name: The name of the MLflow experiment to query

    Returns:
        pd.DataFrame: A transformed DataFrame containing the experiment metrics with columns:
            - All original run columns except metrics
            - 'evaluation': The evaluation component from the metric name
            - 'metric': The specific metric name
            - 'value': The metric value

    Raises:
        AssertionError: If the MLflow search results cannot be converted to a DataFrame

    Example:
        >>> metrics_df = get_metrics_from_experiment("my_experiment")
        >>> print(metrics_df.columns)
        Index(['run_id', 'evaluation', 'metric', 'value', ...])
    """
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    assert isinstance(runs, pd.DataFrame)

    # Identify columns that aren't metrics to use as ID variables
    id_vars = list(
        set(runs.columns) - set(runs.filter(regex=("^(metrics\\.)")).columns)
    )

    # Melt the DataFrame to transform metrics columns into rows
    runs = runs.melt(
        id_vars=id_vars,
    )

    # Split the metric names into evaluation and metric components
    runs[["evaluation", "metric"]] = (
        runs["variable"].str.split(".", expand=True).iloc[:, 1:]
    )

    # Clean up the DataFrame
    runs.drop("variable", axis=1, inplace=True)
    runs.columns = runs.columns.str.removeprefix("params.")

    return runs
