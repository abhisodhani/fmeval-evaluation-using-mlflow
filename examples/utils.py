from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import List, Sequence

import mlflow
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from fmeval.model_runners.model_runner import ModelRunner
from packaging import version

from fmeval_mlflow import track_evaluation


@dataclass
class EvaluationSet:
    data_config: DataConfig | List[DataConfig]
    eval_algo: EvalAlgorithmInterface
    prompt_template: str


def run_evaluation_sets(
    model_runner: ModelRunner,
    evaluation_sets: Sequence[EvaluationSet],
) -> None:
    """
    Runs a battery of evaluations on a given model runner and evaluation set.

    Args:
        model_runner (ModelRunner): The model runner instance to be evaluated.
        evaluation_set (Sequence[Tuple[DataConfig, EvalAlgorithmInterface, str]]):
            A sequence of tuples containing the data configuration, evaluation algorithm,
            and prompt template for each evaluation.

    Returns:
        None
    """
    for evaluation_set in evaluation_sets:
        with redirect_stderr(open("/dev/null", "w", encoding="utf-8")), redirect_stdout(
            open("/dev/null", "w", encoding="utf-8")
        ):
            eval_output = evaluation_set.eval_algo.evaluate(
                model=model_runner,
                dataset_config=evaluation_set.data_config,
                prompt_template=evaluation_set.prompt_template,
                save=True,
            )
        if mlflow.active_run():
            track_evaluation(
                evaluation_set.data_config,
                model_runner,
                eval_output,
                log_eval_output_artifact=True,
            )


def run_evaluation_sets_nested(
    model_runner: ModelRunner,
    evaluation_sets: Sequence[EvaluationSet],
) -> None:
    """
    Runs a battery of evaluations on a given model runner and evaluation set. Separate the runs per evaluation set and organize them as nested runs within a parent run

    Args:
        model_runner (ModelRunner): The model runner instance to be evaluated.
        evaluation_set (Sequence[Tuple[DataConfig, EvalAlgorithmInterface, str]]):
            A sequence of tuples containing the data configuration, evaluation algorithm,
            and prompt template for each evaluation.

    Returns:
        None
    """
    if version.parse(mlflow.__version__) < version.parse("2.15.0"):
        raise ValueError(
            "MLflow version is lower than 2.15.0. `parent_run_id` is not supported in this version. Please upgrade to a newer version."
        )

    for evaluation_set in evaluation_sets:
        with redirect_stderr(open("/dev/null", "w", encoding="utf-8")), redirect_stdout(
            open("/dev/null", "w", encoding="utf-8")
        ):
            eval_output = evaluation_set.eval_algo.evaluate(
                model=model_runner,
                dataset_config=evaluation_set.data_config,
                prompt_template=evaluation_set.prompt_template,
                save=True,
            )
        if parent_run := mlflow.active_run():
            evaluation_set_name = type(evaluation_set.eval_algo).__name__
            parent_run_id = parent_run.info.run_id
            print(f"MLflow parent run ID: {parent_run_id}")
            # with mlflow.start_run(run_id=parent_run_id, nested=True):
            with mlflow.start_run(
                run_name=evaluation_set_name,
                nested=True,
                parent_run_id=parent_run_id,  # type: ignore
            ):
                track_evaluation(
                    evaluation_set.data_config,
                    model_runner,
                    eval_output,
                    log_eval_output_artifact=True,
                )
