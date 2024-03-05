![InteractDA OpenSource](.assets/interactda_github.png)

## About InteractDA Open Source


InteractDA Open Source is a **declarative ML library** that aims to specify every step, down to the smallest data operations, in ML workflows. (EDA, Preprocessing, Model Tuning, Evaluation, Deployment, etc). 

This approach facilitates a simple, unified, consistent, manageable, and transferrable(json, api) Data Science experience across various platforms and environments.

*🚀 We are just getting started, and we're rapidly adding new features. Stay tuned for updates!*

## About InteractDA

[InteractDA]("https://interactda.com") is an interactive no-code Data Science Platform designed to streamline and simplify the conventional workflow of Data Scientists across various topics. We're transforming the traditional notebook-style workflow into a more fun and interactive data playground.

Check our blog [From Jupyter Notebook to an Interactive Data Playground](https://interactda.com/blogs/blog_77364850) for more details.

## Usage Example
- **Declarative Preprocessing**
```py
import pandas as pd
from interactda.tabular.prep import Preprocessing

df = pd.read_csv(".assets/example-kaggle_house_price_prediction.csv")

prep = Preprocessing()
pipeline = [
        {
            "action": "missing_values",
            "targeted_cols": "number",
            "method": "simple_imputer",
            "strategy": "mean",
            "args": {}
        },
        {
            "action": "sampling",
            "targeted_cols": "all",
            "method": "over_sampling",
            "strategy": "smotenc",
            "args": {
                "target": "MSZoning",
                "k_neighbors": 5,
            }
        }
    ]

prep.setup_pipeline(pipeline)
prep.fit(df)
transfromed_dfs = prep.transform([df])
```
- **Preprocessing Tuning**
```py
import pandas as pd
from interactda.tabular.prep import Preprocessing

df = pd.read_csv(".assets/example-kaggle_house_price_prediction.csv")

prep = Preprocessing()
tuning_cfg = [
       {
            "action": "feature_engineering",
            "targeted_cols": "category",
            "method": "encoding",
            "strategy": "kfold_target_encoder",
            "args": {
                "target": "SalePrice",
                "cv": {
                    "options" : [4, 6, 8, 10]
                }
            }
        },
        [
            {
                "action": "missing_values",
                "targeted_cols": "number",
                "method": "simple_imputer",
                "strategy": "mean",
                "args": {}
            },
            {
                "action": "missing_values",
                "targeted_cols": ["LotFrontage", "MasVnrArea"],
                "method": "predict_imputer",
                "strategy": "knn",
                "args": {
                    "independent_variables":["LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF"],
                    "n_neighbors": {
                        "options" : [5, 10, 20]
                    }
                }
            }
        ]  
    ]

pipelines = prep.get_tuning_pipelines(tuning_cfg,
                                      n_pipelines=10,
                                      random_state=42)
all_preps=[]
for pipeline in pipelines:
    prep=Preprocessing()
    prep.setup_pipeline(pipeline)
    all_preps.append(prep)
```
## Installation
    
    `pip install -U interactda`

## InteractDA Open Source V0.0.1

#### Feature Updates:
- Declarative Preprocessing
- Preprocessing Tuning

#### Supported Methods V0.0.1:

  - **feature_engineering**
    - scaling
      - min_max_scaler
      - standard_scaler
      - robust_scaler
      - max_abs_scaler
      - normalizer
    - discretizing
      - kbins_discretizer
      - binarizer
    - encoding
      - one_hot_encoder
      - kfold_target_encoder
      - label_encoder
      - frequency_encoder


  - **missing_values**
    - simple_imputer
      - mean
      - median
      - most_frequent
      - constant
    - predict_imputer
      - linear_regression
      - knn

  
- **sampling**
    - over_sampling
      - random_over_sampler
      - smote
      - smotenc
      - smoten
      - borderline_smote
      - kmeans_smote
      - svm_smote
      - adasyn
    - under_sampling
      - random_under_sampler
      - cluster_centroids
      - condensed_nearest_neighbour
      - edited_nearest_neighbour
      - all_knn
      - nearmiss
      - tomek_links

      
  - **dimension_reduction**
    - feature_selection
      - drop_cols
      - keep_cols
      - variance_threshold
      - unique_threshold
      - missing_threshold
    - matrix_decomposition
      - pca
      - nmf
      - kernel_pca
      - truncate_svd
    - discriminant_analysis
      - lda



## Future Updates
- Declarative Model Training & Tuning
- Interactive visualizations for pipeline workflows
- Integration with Optuna
- Expand to other topics like NLP, Time Series, etc
- QA and testing

### 📢 Calling for Contributors

We are seeking contributors who share our vision and can help enhance the library. If you have profound experience in areas such as classical ML, deep learning, NLP, time series analysis, computer vision, and more, and are interested in contributing to the project, please don't hesitate to submit a pull request.

Feel free to contact us at [interactda.com]("https://interactda.com"), and follow our updates.