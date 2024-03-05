PREPROCESSING_OPTIONS = {
    "feature_engineering": {
        "scaling": [
            "min_max_scaler",
            "standard_scaler",
            "robust_scaler",
            "max_abs_scaler",
            "normalizer",
        ],
        "discretizing": ["kbins_discretizer", "binarizer"],
        "encoding": [
            "one_hot_encoder",
            "kfold_target_encoder",
            "label_encoder",
            "frequency_encoder",
        ],
        "grouping": ["grouping"],
    },
    "missing_values": {
        "simple_imputer": ["mean", "median", "most_frequent", "constant"],
        "predict_imputer": ["linear_regression", "knn"],
    },
    "sampling": {
        "over_sampling": [
            "random_over_sampler",
            "smote",
            "smotenc",
            "smoten",
            "borderline_smote",
            "kmeans_smote",
            "svm_smote",
            "adasyn",
        ],
        "under_sampling": [
            "random_under_sampler",
            "cluster_centroids",
            "condensed_nearest_neighbour",
            "edited_nearest_neighbour",
            "all_knn",
            "nearmiss",
            "tomek_links",
        ],
    },
    "dimension_reduction": {
        "feature_selection": [
            "drop_cols",
            "keep_cols",
            "variance_threshold",
            "unique_threshold",
            "missing_threshold",
        ],
        "matrix_decomposition": ["pca", "nmf", "kernel_pca", "truncate_svd"],
        "discriminant_analysis": ["lda"],
    },
}
