from pydantic import Field
import pandas as pd
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    SMOTENC,
    SMOTEN,
    BorderlineSMOTE,
    KMeansSMOTE,
    SVMSMOTE,
    ADASYN,
)
from tabular.dataset import TabularDataType, TabularDataset
from utils.tabular.preprocessing.preprocessing_base import PrepBase
from typing import List, Dict, Callable
from utils.tabular.preprocessing.validation_base import ArgsBase


class DARandomOverSampler:
    _extend = False
    _allowed_data_types = [
        TabularDataType.CATEGORY.value,
        TabularDataType.NUMBER.value,
        TabularDataType.DATETIME.value,
        TabularDataType.TEXT.value,
    ]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        shrinkage: float | dict | None = None
        random_state: int | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = RandomOverSampler(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DASMOTE:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        k_neighbors: int | object = 5
        random_state: int | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = SMOTE(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DASMOTENC:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value, TabularDataType.CATEGORY.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        categorical_encoder: Callable | None = None
        sampling_strategy: str | float | dict | Callable = "auto"
        k_neighbors: int | object = 5
        random_state: int | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        dtypes = TabularDataset(df).dtypes
        cat_cols = [
            col
            for col in df.columns
            if dtypes[col] == TabularDataType.CATEGORY.value and col != target.name
        ]
        self.args["categorical_features"] = cat_cols

        self.sampler = SMOTENC(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DASMOTEN:
    _extend = False
    _allowed_data_types = [TabularDataType.CATEGORY.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        category_encoder: Callable | None = None
        sampling_strategy: str | float | dict | Callable = "auto"
        k_neighbors: int | object = 5
        random_state: int | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = SMOTEN(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DABorderlineSMOTE:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        k_neighbors: int | object = 5
        m_neighbors: int | object = 10
        kind: str = Field("borderline-1", pattern="^(borderline-1|borderline-2)$")
        random_state: int | None = None

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = BorderlineSMOTE(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DAKMeansSMOTE:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        k_neighbors: int | object = 2
        random_state: int | None = None
        kmeans_estimator: object | None = None
        cluster_balance_threshold: float | str = "auto"
        density_exponent: float | str = "auto"

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = KMeansSMOTE(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DASVMSMOTE:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        k_neighbors: int | object = 5
        m_neighbors: int | object = 10
        svm_estimator: object | None = None
        random_state: int | None = None
        out_step: float = 0.5

    def __init__(self, **kwargs):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = SVMSMOTE(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DAADASYN:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        n_neighbors: int | object = 5
        random_state: int | None = None

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = ADASYN(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class PrepSamplingOverSampling(PrepBase):
    PREPROCESSOR_MAPPING = {
        "random_over_sampler": DARandomOverSampler,
        "smote": DASMOTE,
        "smotenc": DASMOTENC,
        "smoten": DASMOTEN,
        "borderline_smote": DABorderlineSMOTE,
        "kmeans_smote": DAKMeansSMOTE,
        "svm_smote": DASVMSMOTE,
        "adasyn": DAADASYN,
    }

    def __init__(self, targeted_cols: List[str], strategy: str, args: Dict):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None

    def fit(self, df):
        target = self.args.get("target")
        self.preprocessor = self.PREPROCESSOR_MAPPING[self.strategy](**self.args)
        self._validate_preprocessing(
            df=df,
            preprocessor=self.preprocessor,
            targeted_cols=self.targeted_cols,
            args=self.args,
        )
        self.preprocessor.fit(df, target=df[target])

    def transform(self, df):
        target = self.args.get("target")

        return self.preprocessor.transform(
            (
                df[self.targeted_cols].drop(columns=[target])
                if target in self.targeted_cols
                else df[self.targeted_cols]
            ),
            df[target],
        )
