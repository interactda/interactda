import itertools
from pprint import pprint
import random
import pandas as pd
from typing import List, Dict, Tuple
from pydantic import ValidationError
from tabular.dataset import TabularDataset, TabularDataType
from utils.tabular.preprocessing.dimension_reduction.discriminant_analysis import (
    PrepDimensionReductionDiscriminantAnalysis,
)
from utils.tabular.preprocessing.dimension_reduction.feature_selection import (
    PrepDimensionReductionFeatureSelection,
)
from utils.tabular.preprocessing.dimension_reduction.matrix_decomposition import (
    PrepDimensionReductionMatrixDecomposition,
)
from utils.tabular.preprocessing.feature_engineering.discretizing import (
    PrepFeatureEngineeringDiscretizing,
)
from utils.tabular.preprocessing.feature_engineering.encoding import (
    PrepFeatureEngineeringEncoding,
)
from utils.tabular.preprocessing.feature_engineering.scaling import (
    PrepFeatureEngineeringScaling,
)
from utils.tabular.preprocessing.missing_values.predict_imputer import (
    PrepMissingValuesPredictImputer,
)
from utils.tabular.preprocessing.missing_values.simple_imputer import (
    PrepMissingValuesSimpleImputer,
)
from utils.tabular.preprocessing.sampling.over_sampling import PrepSamplingOverSampling
from utils.tabular.preprocessing.sampling.under_sampling import (
    PrepSamplingUnderSampling,
)
from utils.tabular.preprocessing.validation_base import PrepStepValidationBase


PREP_METHOD_MAPPING = {
    "feature_engineering": {
        "scaling": PrepFeatureEngineeringScaling,
        "encoding": PrepFeatureEngineeringEncoding,
        "discretizing": PrepFeatureEngineeringDiscretizing,
    },
    "missing_values": {
        "simple_imputer": PrepMissingValuesSimpleImputer,
        "predict_imputer": PrepMissingValuesPredictImputer,
    },
    "sampling": {
        "over_sampling": PrepSamplingOverSampling,
        "under_sampling": PrepSamplingUnderSampling,
    },
    "dimension_reduction": {
        "feature_selection": PrepDimensionReductionFeatureSelection,
        "matrix_decomposition": PrepDimensionReductionMatrixDecomposition,
        "discriminant_analysis": PrepDimensionReductionDiscriminantAnalysis,
    },
}


class Preprocessing:
    def __init__(self):
        self._pipeline = None
        self._fitted = False
        self._fitted_pipeline = None

    def setup_pipeline(self, pipeline: List[Dict]):
        if self._validate_pipeline(pipeline):
            self._pipeline = pipeline

    def fit(self, df: pd.DataFrame):
        assert self._pipeline, "Pipeline is not set"
        assert len(self._pipeline) > 0, "Pipeline is empty"
        assert isinstance(
            df, pd.DataFrame
        ), f"Input must be pandas DataFrame, not {str(type(df))}"

        self._fitted_pipeline = []
        for step in self._pipeline:
            action, method, strategy, args = (
                step["action"],
                step["method"],
                step["strategy"],
                step["args"],
            )
            targeted_cols = self._get_actual_targeted_cols(df=df, step=step)

            method_prep = PREP_METHOD_MAPPING[action][method](
                targeted_cols=targeted_cols, strategy=strategy, args=args
            )

            method_prep.fit(df)
            df = method_prep.transform(df)
            self._fitted_pipeline.append(method_prep)
        self._fitted = True

    def transform(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        assert self._fitted, "Pipeline is not fitted"
        assert self._fitted_pipeline, "Pipeline is empty"
        assert len(dfs) > 0, "No data to transform"
        assert all(
            [isinstance(df, pd.DataFrame) for df in dfs]
        ), "All inputs must be DataFrame"

        transformed_dfs = []
        for df in dfs:
            for step in self._fitted_pipeline:
                df = step.transform(df)
            transformed_dfs.append(df)
        return transformed_dfs

    def get_tuning_pipelines(
        self, tuning_cfg: List[Dict | List[Dict]], n_pipelines: int, random_state: int
    ) -> List[Tuple[Dict]]:
        if self._validate_tuning_cfg(tuning_cfg):
            all_step_opts = []
            for step_tuning in tuning_cfg:
                if isinstance(step_tuning, list):
                    step_opts = [
                        opt
                        for step in step_tuning
                        for opt in self.expand_step_tunning(step)
                    ]
                elif isinstance(step_tuning, dict):
                    step_opts = self.expand_step_tunning(step_tuning)
                all_step_opts.append(step_opts)
            all_pipelines = list(itertools.product(*all_step_opts))
            random.seed(random_state)
            selected_pipelines = random.sample(
                all_pipelines, min(n_pipelines, len(all_pipelines))
            )
            return selected_pipelines

    @staticmethod
    def expand_step_tunning(step: Dict) -> List[Dict]:
        args = step["args"]
        options = [
            value["options"] if isinstance(value, dict) else [value]
            for value in args.values()
        ]
        keys = list(args.keys())
        combinations = list(itertools.product(*options))
        all_combinations = []
        for combo in combinations:
            step_dict = step.copy()
            step_dict["args"] = dict(zip(keys, combo))
            all_combinations.append(step_dict)
        return all_combinations

    @staticmethod
    def _get_actual_targeted_cols(df: pd.DataFrame, step: Dict) -> List[str]:
        targeted_cols = step["targeted_cols"]
        all_cols = list(df.columns)
        if targeted_cols == "all":
            targeted_cols = all_cols
        elif targeted_cols in [
            TabularDataType.CATEGORY.value,
            TabularDataType.NUMBER.value,
        ]:
            dtypes = TabularDataset(df).dtypes
            targeted_cols = [col for col in all_cols if dtypes[col] == targeted_cols]
        elif type(targeted_cols) == list:
            targeted_cols = [col for col in targeted_cols if col in all_cols]

        exclude = step.get("exclude")
        if exclude:
            targeted_cols = [col for col in targeted_cols if col not in exclude]

        include = step.get("include")
        if include:
            targeted_cols = targeted_cols + include
        return targeted_cols

    @staticmethod
    def _validate_pipeline(pipeline: List[Dict]) -> bool:
        validated = True
        for idx, step in enumerate(pipeline):
            try:
                action, method, strategy, args = (
                    step["action"],
                    step["method"],
                    step["strategy"],
                    step["args"],
                )
                preprocessor_class = PREP_METHOD_MAPPING[action][
                    method
                ].PREPROCESSOR_MAPPING[strategy]
                preprocessor_class.Args(**args)
                PrepStepValidationBase(**step)

            except ValidationError as e:
                print("Step {} is not valid".format(idx + 1))
                print(e)
                print("\n**************\n")
                validated = False
        return validated

    @staticmethod
    def _validate_tuning_cfg(tuning_cfg: List[Dict | List[Dict]]) -> bool:
        validated = True
        return validated


if __name__ == "__main__":
    df = pd.read_csv("../playgrounds/housing-prices-dataset_train.csv")
    dtypes = TabularDataset(df).dtypes
    num_cols = [
        col for col in df.columns if dtypes[col] == TabularDataType.NUMBER.value
    ]
    prep = Preprocessing()
    step0 = {
        "action": "dimension_reduction",
        "targeted_cols": "number",
        "method": "feature_selection",
        "strategy": "variance_threshold",
        "args": {"threshold": {"options": [0.1, 0.2, 0.3]}},
    }

    step1 = [
        {
            "action": "feature_engineering",
            "targeted_cols": "category",
            "exclude": ["MSZoning"],
            "method": "encoding",
            "strategy": "kfold_target_encoder",
            "args": {"target": "SalePrice"},
        },
        {
            "action": "missing_values",
            "targeted_cols": "category",
            "method": "simple_imputer",
            "strategy": "constant",
            "args": {"fill_value": {"options": ["missing", "unknown"]}},
        },
    ]
    # step1_5 = {
    #     "action": "feature_engineering",
    #     "targeted_cols": ["MSZoning"],
    #     "method": "encoding",
    #     "strategy": "label_encoder",
    #     "args": {},
    # }
    # step2 = {
    #     "action": "dimension_reduction",
    #     "targeted_cols": "all",
    #     "method": "discriminant_analysis",
    #     "strategy": "lda",
    #     "args": {
    #         "target": "SalePrice",
    #         "n_components": 10,
    #         # "beta_loss": "frobenius",
    #     },
    # }
    pipelines = prep.get_tuning_pipelines(
        [step0, step1], n_pipelines=6, random_state=40
    )
    pprint(pipelines)
    # prep.setup_pipeline([step0])
    # prep.fit(df)
    # transformed_df = prep.transform([df])
    # print(transformed_df[0])
