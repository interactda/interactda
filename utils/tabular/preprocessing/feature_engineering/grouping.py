from utils.tabular.preprocessing.preprocessing_base import PrepBase


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
