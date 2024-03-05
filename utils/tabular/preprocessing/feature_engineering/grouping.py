from sklearn.base import TransformerMixin, BaseEstimator

from utils.tabular.preprocessing.preprocessing_base import PrepBase


# class GroupLeastFrequent(TransformerMixin, BaseEstimator):
#     def fit(self):
#         pass
#
#
class DAGroupLeastFrequent(TransformerMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.args
        self.least_frequent = None

    def fit(self, X, y=None):
        self.least_frequent = (
            X.value_counts().tail(len(X.unique()) - self.n_keep).index.tolist()
        )
        return self

    def transform(self, X):
        X = X.apply(lambda x: "Other" if x in self.least_frequent else x)
        return X


class PrepFeatureEngineeringGrouping(PrepBase):
    PREPROCESSOR_MAPPING = {
        "least_frequent_grouper": None,
        "manual_grouper": None,
    }

    def __init__(self, targeted_cols, strategy, args):
        self.targeted_cols = targeted_cols
        self.strategy = strategy
        self.args = args
        self.preprocessor = None
        self.dtypes = None
