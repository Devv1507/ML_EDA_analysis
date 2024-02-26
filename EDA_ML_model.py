import os
os.chdir('C:\Python39\Scripts')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from lime import lime_tabular

col=["qubit","depth","reads"]
dtf = pd.read_csv("INS_resum.csv", header=0, delimiter=";")
dtf.head()

x, y = "reads", "depth"

## split data
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)

print(dtf_train)
print(dtf_test)

## scale X
scalerX = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X = scalerX.fit_transform(dtf_train.drop("depth", axis=1))
dtf_scaled= pd.DataFrame(X, columns=dtf_train.drop("depth", 
                        axis=1).columns, index=dtf_train.index)
## scale Y
scalerY = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
dtf_scaled[y] = scalerY.fit_transform(
                    dtf_train[y].values.reshape(-1,1))
print(dtf_scaled.head())

## Selected features
X_names = ['qubit', 'ct','reads']
X_train = dtf_train[X_names].values
y_train = dtf_train["depth"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["depth"].values

## call model
model = linear_model.LinearRegression()

# train
model.fit(X_train, y_train)
## test
predicted = model.predict(X_test)

## Kpi
print("R2 (explained variance):", round(metrics.r2_score(y_test, predicted), 2))
print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", round(np.mean(np.abs((y_test-predicted)/predicted)), 2))
print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
## residuals
residuals = y_test - predicted
max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
max_true, max_pred = y_test[max_idx], predicted[max_idx]
print("Max Error:", "{:,.0f}".format(max_error))

## Explainibility

print("True:", "{:,.0f}".format(y_test[1]), "--> Pred:", "{:,.0f}".format(predicted[1]))

explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names="depth", mode="regression")
explained = explainer.explain_instance(X_test[1], model.predict, num_features=3)
explained.as_pyplot_figure()

plt.show()
