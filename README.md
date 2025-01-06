# Track FMeval evaluation using MLflow

[Foundation Model Evaluations Library, of fmeval](https://github.com/aws/fmeval) is a Python library to evaluate Large Language Models (LLMs). This sample shows how to simplify the tracking and recording into MLflow of the evaluations performed using fmeval.

The sample consist of a collection of utility functions in [fmeval_mlflow.py](fmeval_mlflow/fmeval_mlflow.py) and example notebooks demonstrating the use of the functions.

## Examples

The examples notebooks have been developed on and are expected to be executed in a [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/) space, either [Jupyterlab](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl.html) or [CodeEditor](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html). The examples also leverage [Amazon SageMaker with MLflow](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html) as tracking server, and access the large language models to evaluate via either [Amazon Bedrock](https://aws.amazon.com/bedrock/) or [SageMaker Jumpstart](https://aws.amazon.com/sagemaker/jumpstart/).

While these services are an effective way to deploy, access, work with, and evaluate large language models, this evaluation and tracking is independent of the environment. Fmeval and [MLflow](https://mlflow.org/) are open source projects, and can be integrated in different context. Check the [examples](https://github.com/aws/fmeval/tree/main/examples) in fmeval repository to see how to customize the LLM connectors and metrics. This sample ha snot been tested extensively outside of the conditions mentioned above, but it should not be too hard to adapt to a different environment if necessary.

## Utility functions

There is one main functions that help track fmeval evaluation into an MLflow run, `track\_evaluation()`, and two functions to retrieve the metrics from a tracking server, used for example to create custom visualizations.

#### track\_evaluation

```python
def track_evaluation(data_config: DataConfig | list[DataConfig] | None = None,
                     model_runner: ModelRunner | None = None,
                     eval_output: list[EvalOutput] | None = None,
                     custom_parameters_map: dict[str, str] | None = None,
                     log_eval_output_artifact: bool = False,
                     model_id: str | None = None)
```

Track the evaluation of a model on a dataset using MLflow.

This function logs various aspects of a model evaluation to MLflow, including:

- Input dataset configuration
- Model runner parameters
- Evaluation metrics and outputs

**Arguments**:

- `data_config` - Configuration for the dataset(s) used in evaluation.
  Can be either a single DataConfig instance or a list of DataConfig instances.
  If None, dataset tracking is skipped.
  
- `model_runner` - The ModelRunner instance used for generating predictions.
  Contains information about the model configuration and parameters.
  If None, model parameter tracking is skipped.
  
- `eval_output` - List of evaluation results from running evaluation algorithms.
  Each EvalOutput instance contains metrics and other evaluation information.
  If None, metrics tracking is skipped.
  
- `custom_parameters_map` - Additional custom parameters to log with the evaluation.
  Keys and values should be strings. These parameters will be logged
  alongside the model runner parameters.
  
- `log_eval_output_artifact` - Whether to log the complete eval_output as an MLflow artifact.
  Defaults to False. When True, the entire evaluation output will be saved
  as a trackable artifact in MLflow.
  
- `model_id` - Optional model identifier to override the default model ID from the model runner.
  
**Raises**:

- `RuntimeError` - If there is no active MLflow run.
- `TypeError` - If input parameters are of incorrect type.
- `ValueError` - If input parameters contain invalid values.

**Example**:

``` terminal
>>> data_config = DataConfig(
>>>    dataset_name="custom_dataset",
>>>    dataset_uri="./custom_dataset.jsonl",
>>>    dataset_mime_type="application/jsonlines",
>>>    model_input_location="question",
>>>    target_output_location="answer",
>>> )
>>> eval_algo = Toxicity(ToxicityConfig())
>>> with mlflow.start_run():
>>>   eval_output = eval_algo.evaluate(model=model_runner, dataset_config=config)
>>>   track_evaluation(data_config, model_runner, eval_output)
```

#### get\_metrics\_from\_runs

```python
def get_metrics_from_runs(parent_run_id: str) -> pd.DataFrame
```

Extract metrics from all MLflow child runs associated with a specified parent run ID.

This function searches for all child runs associated with a given parent run ID,
extracts their metrics and the model_id parameter, and returns them in a structured
DataFrame format. The metrics and parameter prefixes ('metrics.' and 'params.') are
removed from the column names for clarity.

**Arguments**:

- `parent_run_id` _str_ - The unique identifier of the parent MLflow run. All child
  runs associated with this parent run will be included in the results.
  
**Returns**:

- `pd.DataFrame` - A DataFrame containing metrics from all child runs, with:
  - Index: model_id from the run parameters
  - Columns: All metrics from the runs, with 'metrics.' prefix removed
  
**Raises**:

- `AssertionError` - If the MLflow search results cannot be converted to a DataFrame
  
**Example**:

``` terminal
>>> metrics_df = get_metrics_from_runs("parent_run_123")
>>> print(metrics_df.head())
accuracy  precision  recall
model_id
model_1   0.95      0.94      0.93
model_2   0.92      0.91      0.90
```

#### get\_metrics\_from\_experiment

```python
def get_metrics_from_experiment(experiment_name: str) -> pd.DataFrame
```

Retrieve and transform metrics from an MLflow experiment into a structured DataFrame.

This function searches for runs in the specified MLflow experiment and transforms
the metrics data into a melted DataFrame format, splitting metric names into
evaluation and metric components.

**Arguments**:

- `experiment_name` - The name of the MLflow experiment to query
  
**Returns**:

- `pd.DataFrame` - A transformed DataFrame containing the experiment metrics with columns:
  - All original run columns except metrics
  - 'evaluation': The evaluation component from the metric name
  - 'metric': The specific metric name
  - 'value': The metric value
  
**Raises**:

- `AssertionError` - If the MLflow search results cannot be converted to a DataFrame
  
**Example**:

```terminal
>>> metrics_df = get_metrics_from_experiment("my_experiment")
>>> print(metrics_df.columns)
Index(['run_id', 'evaluation', 'metric', 'value', ...])
```
