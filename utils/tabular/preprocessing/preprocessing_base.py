from abc import ABC, abstractmethod

import pandas as pd
from pydantic import ValidationError
from typing import List, Dict

from tabular.dataset import TabularDataset


class PrepBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @staticmethod
    def _validate_preprocessing(
        df: pd.DataFrame,
        preprocessor: object,
        targeted_cols: List[str],
        args: Dict,
    ):
        dtypes = TabularDataset(df).dtypes
        if not preprocessor._missing_values_allowed:
            missing_stats = df[targeted_cols].isnull().sum()
            assert (
                missing_stats.sum() == 0
            ), f"{type(preprocessor).__name__} requires no missing values, please preprocess it before. {missing_stats[missing_stats>0]}"

        if preprocessor._allowed_data_types != "all":
            for col in targeted_cols:
                assert (
                    dtypes[col] in preprocessor._allowed_data_types
                ), f"{type(preprocessor).__name__} does not support [{dtypes[col]}] data type"

        if preprocessor._dependent_variable_required:
            assert args.get(
                "target"
            ), f"{type(preprocessor).__name__} requires key 'target' inside args"

        try:
            preprocessor.Args(**args)
        except ValidationError as e:
            raise ValidationError(
                f"Invalid args for {type(preprocessor).__name__}:\n {e}"
            )
