# Wine Preference Analysis

The purpose of this work is the modeling of the wine preferences by physicochemical properties.
Wine quality is a key element within this context. Certification prevents the illegal adulteration of wines
and assures quality for the wine market. Quality evaluation is often part of the certification process 
and can be used to improve wine making and to stratify wines such as premium brands (useful for setting prices).
Such model is useful to support the oenologist wine tasting evaluations, improve and speed-up the wine production.

# Features

According to the oenology theory, the quality of wine is dependent on
**Climate & Weather, Temperature and Sunlight, Growing Practices and Winemaking Practices**.
However, this dataset provides only the physicochemical properties of wine samples, without
the environmental conditions or growing/winemaking procedures. Also according to the oenology
theory, a wine can be described by the columns below:

1. **Type** (*Red, White, Rosé, Sparkling and Fortified (Dessert)*): The different types of wine.
2. **Fixed Acidity** (*1000 mg/L - 4000 mg/L*): Organic acids of wines whose volatilities are so low that cannot be separated from wine by distillation
3. **Volatile Acidity** (*0 - 1700 mg/L*): A measure of the low molecular weight fatty acids in wine and is generally perceived as the odour of vinegar
4. **Citric Acid** (*< 500 mg/L*): Citric acid is used as supplement during the fermentation process to boost the acidity of wine
5. **Residual Sugar** (*> 35 mg/L*): Natural grape sugars leftover in a wine after the alcoholic fermentation
6. **Chlorides** (*< 606 mg/L*): A major contributor to saltiness in wine
7. **Free sulfur Dioxide** (*< 350 mg/L*): A wine additive that preserves wine's freshness and fruit characters by virtue of antioxidant, antimicrobial and anti-enzymatic properties
8. **Total Sulfur Dioxide** (*< 350 mg/L*): The portion of $SO_{2}$ that is free in the wine plus the portion that is bound to other chemicals such as aldehydes, pigments, or sugars
9. **Density** (*1080 - 1090 mg/*$cm^3$): The density of the wine
10. **pH** (*2.5 pH - 4.5 pH*): A scale used to specify the acidity of the wine
11. **Sulphites** (*5 mg/l - 2000 mg/L*): chemical by-products created during the fermentation process, also used in food preservation
12. **Alcohol** (*5.5% - 25%*): The total concentration of alchohol 

# Quality

The quality of each sample is a numerical value between 0.0 and 10.0. The quality is dependent by the values of each of the features that are decrribed above. The goal of this project is:
1. Build a Regression model to accurately predict the quality score of a given wine sample, based on its physicochemical properties.
2. Analyze the regression model and interpret its decision making system.

# Regression Accuracy

A Neural Network was built with 0.41 mean absolute error, after it was tuned using `Optuna` package. The first 10 predictions are shown in the table below.

| Target Quality | Model Prediction | ERROR |
|----------------|------------------|-------|
| 6              | 6                | 0     |
| 6              | 6                | 0     |
| 5              | 4                | 1     |
| 6              | 5                | 1     |
| 5              | 5                | 0     |
| 7              | 6                | 1     |
| 5              | 5                | 0     |
| 4              | 5                | 1     |
| 6              | 6                | 0     |
| 5              | 5                | 0     |

# Interpretability
