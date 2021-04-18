from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest, SelectPercentile, SelectFdr
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LassoLarsIC
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRFRegressor
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
import umap
import data


def display_feature_importance(feature_importance_df, fs_algorithm_name):
    print('Column importance scores:')
    print(feature_importance_df)

    ax = feature_importance_df.plot.bar(title='Scores Bar Plot - ' + fs_algorithm_name)
    ax.set_xlabel('Features')
    ax.set_ylabel('Scores')
    plt.show()


# Reading data from file.
data_df = data.read_data('wine_quality.csv')
inputs, targets = data.generate_dataset(data_df)
columns = inputs.columns

# Removing outliers. This will increase the accuracy of feature importance.
print('Removing outliers...')

mapper = umap.UMAP(set_op_mix_ratio=0.25, random_state=42)
mapper.fit(inputs, y=targets)
data_clusters = hdbscan.HDBSCAN().fit_predict(mapper.embedding_)

print('Initial samples:', inputs.shape)

inputs = inputs[data_clusters != 1]
targets = targets[data_clusters != 1]

print('Samples after outlier removal:', inputs.shape)

# Feature selection with Variance Threshold.
print('\n----- Algorithm used: Variance Threshold:')
fs_algorithm = VarianceThreshold(threshold=0.9 * (1-0.9))

selected_features = fs_algorithm.fit_transform(inputs, y=targets)
print(selected_features.shape)
print('Features removed:', inputs.shape[1] - selected_features.shape[1])

importance_df = pd.DataFrame(fs_algorithm.variances_, columns)
display_feature_importance(importance_df, 'Variance Threshold')

# Feature selection with Univariate Feature Selection.
univariate_selection_algorithms = {
    'Select K Best': SelectKBest(score_func=mutual_info_regression, k=10),
    'Select Percentile': SelectPercentile(score_func=mutual_info_regression, percentile=90),
    'Select FDR': SelectFdr(score_func=f_regression, alpha=0.05),
}

for algorithm_name, fs_algorithm in univariate_selection_algorithms.items():
    print('\n----- Algorithm used:', algorithm_name)
    selected_features = fs_algorithm.fit_transform(inputs, y=targets)

    print(selected_features.shape)
    print('Features removed:', inputs.shape[1] - selected_features.shape[1])

    importance_df = pd.DataFrame(fs_algorithm.scores_, columns)
    display_feature_importance(importance_df, algorithm_name)

# Feature selection with Recursive Feature Elimination.
tree = XGBRFRegressor(learning_rate=0.1)
min_features = 8
rfecv = RFECV(
    estimator=tree,
    step=1,
    cv=StratifiedKFold(5),
    scoring='neg_mean_absolute_percentage_error',
    min_features_to_select=min_features,
)
rfecv.fit(inputs, y=targets)
selected_features = rfecv.transform(inputs)

print('\n----- Algorithm used: RFECV - XGBRF')
print(selected_features.shape)
print('Optimal number of features:  ', rfecv.n_features_)
ax = plt.subplot()
plt.plot(range(min_features, len(rfecv.grid_scores_) + min_features),
         rfecv.grid_scores_)
ax.set_xlabel('Number of Features Selected')
ax.set_ylabel('Cross Score Validation')
ax.set_title('Optimal Number of Features with RFECV - XGBRF')
plt.show()

# Feature Selection using Selection Model.
estimators = {
    'LassoLarsIC': LassoLarsIC(criterion='bic', precompute=True),
    'XGBRF': XGBRFRegressor(learning_rate=0.1)
}

for estimator_name, estimator in estimators.items():
    print('\n----- Model Selection - Estimator used:', estimator_name)
    model = SelectFromModel(estimator)
    selected_features = model.fit_transform(inputs, y=targets)

    print(selected_features.shape)

# Feature Selection using Sequential Model Selection
estimators = {
    'LassoLarsIC': LassoLarsIC(criterion='bic', precompute=True),
    'XGBRF': XGBRFRegressor(learning_rate=0.1)
}

for estimator_name, estimator in estimators.items():
    print('\n----- Sequential Model Selection - Estimator used:', estimator_name)
    model = SequentialFeatureSelector(
        estimator,
        n_features_to_select=min_features,
        scoring='neg_mean_absolute_percentage_error',
        cv=StratifiedKFold(5),
        n_jobs=-1
    )
    selected_features = model.fit_transform(inputs, y=targets)

    print(selected_features.shape)


