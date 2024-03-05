from imblearn.under_sampling import (
    RandomUnderSampler,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    AllKNN,
    NearMiss,
    TomekLinks,
)
from pydantic import Field
from tabular.dataset import TabularDataType
from utils.tabular.preprocessing.preprocessing_base import PrepBase
import pandas as pd
from utils.tabular.preprocessing.validation_base import ArgsBase
from typing import Callable


class DAClusterCentroids:
    _extend = False
    _allowed_data_types = [
        TabularDataType.NUMBER.value,
    ]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        random_state: int | None = None
        estimator: object | None = None
        voting: str = Field("auto", pattern="auto|hard|soft")

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = ClusterCentroids(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DARandomUnderSampler:
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
        random_state: int | None = None
        replacement: bool = False

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = RandomUnderSampler(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DACondensedNearestNeighbour:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | list | Callable = "auto"
        random_state: int | None = None
        n_neighbors: int | object | None = None
        n_seeds_S: int | None = 1

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = CondensedNearestNeighbour(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DAEditedNearestNeighbours:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | list | Callable = "auto"
        kind_sel: str = Field("all", pattern="all|mode")
        n_neighbors: int | object = 3

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = EditedNearestNeighbours(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DAAllKNN:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | list | Callable = "auto"
        kind_sel: str = Field("all", pattern="all|mode")
        n_neighbors: int | object = 3
        allow_minority: bool = False

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = AllKNN(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DANearMiss:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | float | dict | Callable = "auto"
        version: int = Field(1, ge=1, le=3)
        n_neighbors: int | object = 3
        n_neighbors_ver3: int | object = 3

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = NearMiss(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class DATomekLinks:
    _extend = False
    _allowed_data_types = [TabularDataType.NUMBER.value, TabularDataType.CATEGORY.value]
    _missing_values_allowed = True
    _dependent_variable_required = True
    _only_on_training_set = True

    class Args(ArgsBase):
        target: str = Field(...)
        sampling_strategy: str | list | Callable = "auto"

    def __init__(self, **kwargs: str):
        self.args = self.Args(**kwargs).dict()
        self.args.pop("target")
        self.sampler = None

    def fit(self, df, target):
        self.sampler = TomekLinks(**self.args)

    def transform(self, df, target):
        df_x, ds_y = self.sampler.fit_resample(df, target)
        return pd.concat([df_x, ds_y], axis=1)


class PrepSamplingUnderSampling(PrepBase):
    PREPROCESSOR_MAPPING = {
        "random_under_sampler": DARandomUnderSampler,
        "cluster_centroids": DAClusterCentroids,
        "condensed_nearest_neighbour": DACondensedNearestNeighbour,
        "edited_nearest_neighbour": DAEditedNearestNeighbours,
        "all_knn": DAAllKNN,
        "nearmiss": DANearMiss,
        "tomek_links": DATomekLinks,
    }

    def __init__(self, targeted_cols, strategy, args):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.dtypes = None
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
