# Feature-Selection-Algorithms---Wine-Quality
In this project, I used Feature Selection algorithms to find the importance of wine features.

# Dataset
The dataset i used can be found here: https://www.kaggle.com/rajyellow46/wine-quality

It contains 6500 measurements of the features of wine and their quality score. Specifically, It contains the following columns:

1. Type
2. Fixed Acidity
3. Volatile Aciditi
4. Citric Acid
5. Residual Sugar
6. Chlorides
7. Free Sulfur Dioxide
8. Total Sulfur Dioxide
9. Density
10. pH
11. Sulphates
12. Alcohol
13. Quality Score

# Regressor Construction
First, I built a regressor, in order try and predict the score of a wine, based on its measurements. The Regressor that i have used is a Deep Neural Network with [128, 128, 128] hidden units. Before I train the regressor, I used UMAP + HDBSCAN method to remove any outlying measurements. Then, I scaled the data to range (0, 1) to make the training faster and trained the regressor model.

# Regression Results
After some minutes of training, my model managed to score 1 - 0.0956 or **90% accuracy**. The error 0.0956 was measured using Mean Absolute Percentage Error function.
Here is the results of the first 5 validation samples:

| Prediction | Actual |
| ---------- | ------ |
| 5.7370844  | 5.0  |
| 5.6009750  | 6.0  |
| 5.7100430  | 4.0  |
| 5.9350880  | 6.0  |
| 5.7370844  | 6.99 |

The validation log can be found here: https://github.com/kochlisGit/Feature-Selection-Algorithms---Wine-Quality/blob/main/Regression_Log 

The log contains 650 tested samples.

# Feature Importance

To improve the regressor accuracy it is impotant to understand what are the most important features in the dataset. Maybe the dataset contains unecessary features that act as **"noise"** for out model. Different algorithms were used to compute feature importance:

The entire log can be found here: https://github.com/kochlisGit/Feature-Selection-Algorithms---Wine-Quality/blob/main/feature_selection_log

![](https://github.com/kochlisGit/Feature-Selection-Algorithms---Wine-Quality/blob/main/plots/RFECV%20-%20XGBRF.png)

![](https://github.com/kochlisGit/Feature-Selection-Algorithms---Wine-Quality/blob/main/plots/select_fdr.png)

![](https://github.com/kochlisGit/Feature-Selection-Algorithms---Wine-Quality/blob/main/plots/variance_threshold.png)

![](https://github.com/kochlisGit/Feature-Selection-Algorithms---Wine-Quality/blob/main/plots/select_k_best.png)

![](https://github.com/kochlisGit/Feature-Selection-Algorithms---Wine-Quality/blob/main/plots/select_percentile.png)
