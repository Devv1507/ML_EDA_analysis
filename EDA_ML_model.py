#!/usr/bin/env python
# coding: utf-8

import os
os.chdir("data")

# Importing statistics, data analysis and graph design librarys (pandas, numpy, matplotlib, scipy):
import pandas as pd
from pandas import DataFrame,Series 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns # plotting library
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm 
from lime import lime_tabular
from math import sqrt

# Importing sklearn, open library specializing in Machine Learning models:
import sklearn
from sklearn import tree, metrics, linear_model, svm, metrics, model_selection, preprocessing, feature_selection, ensemble, decomposition
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict, RepeatedKFold, KFold, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import r2_score, get_scorer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# load dataset of 4 continuous variables
names = ['qubit','depth','reads','cov']
dtf = pd.read_csv("dataset.csv", header=0, delimiter=";", names=names)


### Exploratory Data Analysis
x = "reads"  # target variable
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=(12, 6))
fig.suptitle(x, fontsize=20)

# Target variable distribution
ax[0].title.set_text('distribution')
variable = dtf[x].fillna(dtf[x].mean())
breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
variable = variable[ (variable > breaks[0]) & (variable <  breaks[10]) ]
sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
des = dtf[x].describe()
ax[0].axvline(des["25%"], ls='--')
ax[0].axvline(des["mean"], ls='--')
ax[0].axvline(des["75%"], ls='--')
ax[0].grid(axis='x')
des = round(des, 2).apply(lambda x: str(x))
box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))

# Target variable outliers boxplot
ax[1].title.set_text('outliers (log scale)')
tmp_dtf = pd.DataFrame(dtf[x])
tmp_dtf[x] = np.log(tmp_dtf[x])
tmp_dtf.boxplot(column=x, ax=ax[1])
plt.savefig('reads_dtf1_2.svg')
plt.show()

### Descriptive comparison reads vs depth
x, y = "reads", "depth"

### bin plot
figsize = (6, 6)
dtf_noNan = dtf[dtf[x].notnull()]
breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, 
           duplicates='drop')])[y].agg(['mean','median','size'])
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle(x+"   vs   "+y, fontsize=20)
groups[["mean", "median"]].plot(kind="line", ax=ax)
groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True,
                    color="grey", alpha=0.3, grid=True)
ax.set(ylabel=y)
ax.right_ax.set_ylabel("Observazions in each bin")
plt.savefig('bins_dtf1.svg')
plt.show()

### scatter plot
sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='reg', 
              height=int((figsize[0]+figsize[1])/2) )
plt.savefig('cor_dtf1.svg')
plt.show()

dtf_noNan = dtf[dtf[x].notnull()]
coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
coeff, p = round(coeff, 3), round(p, 3)
conclusion = "Significant" if p < 0.05 else "Non-Significant"
print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")


### univariate analysis
fig = plt.figure(figsize=(16, 8))  
i = 0
for column in dtf:
    sub = fig.add_subplot(2,4 , i + 1)
    sub.set_xlabel(column)
    dtf[column].plot(kind = 'hist')
    i = i + 1

dtf.describe()


# Creating basic correlation plots with the data
sns.pairplot(dtf[['qubit','depth','reads','cov']], diag_kind="kde")


# Define seed for reproducibility
seed = 27

# Splitting data
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.25, random_state=seed)
print(dtf_train)
print(dtf_test)

# Heatmap
corr_matrix = dtf_train.corr(method="pearson")
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")


## Scalling
scaler = preprocessing.RobustScaler()
X_names = ['qubit','depth','cov']
X = scaler.fit_transform(dtf_train[X_names])
dtf_scaled= pd.DataFrame(X, columns=dtf_train[X_names].columns, index=dtf_train[X_names].index)
print(dtf_scaled)
dtf_scaled['reads'] = scaler.fit_transform(dtf_train['reads'].values.reshape(-1,1))
print(dtf_scaled)

dtf_test_scaled = pd.DataFrame(scaler.fit_transform(dtf_test[X_names]), columns=dtf_test[X_names].columns, index=dtf_test[X_names].index)
print(dtf_test_scaled)
dtf_test_scaled['reads'] = scaler.fit_transform(dtf_test['reads'].values.reshape(-1,1))
print(dtf_test_scaled)


## Selected features
#dataset1
X_names = ['qubit','depth','cov']
X_train = dtf_train[X_names].values
y_train = dtf_train["reads"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["reads"].values


#dataset models comparison
from sklearn.pipeline import make_pipeline

Lss_params = [{"alpha": 2.0970464013232393}]
GB_params =  [{"random_state": seed}]
RF_params =  [{"n_estimators": 1000 ,  "random_state": seed}]
SV_params = [{"random_state": seed, "max_iter": 100000}]

models = []
models.append(['Lasso', Lasso, Lss_params])
models.append(['GBR', GradientBoostingRegressor, GB_params])
models.append(['RFR', RandomForestRegressor, RF_params])
models.append(['SVR', LinearSVR, SV_params])

for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train, y_train)    
        # Predict y_pred
        y_pred = model.predict(X_test)
        # Regressin performance metrics
        print('R2','%s: %s' %(Model.__name__, metrics.r2_score(y_test, y_pred)))
        print('MAE','%s: %s' %(Model.__name__, metrics.mean_absolute_error(y_test, y_pred)))
        print('MSE','%s: %s' %(Model.__name__, metrics.mean_squared_error(y_test, y_pred)))
        print('RMSE','%s: %s' %(Model.__name__, metrics.mean_squared_error(y_test, y_pred, squared=False)))
        print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", '%s: %s' %(Model.__name__, round(np.mean(np.abs((y_test-y_pred)/y_pred)), 2)))


## K fold validation #########
scores = []
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state = seed)
for modelname, Model, params_list in models:
    for params in params_list:
        fig = plt.figure(figsize=(6, 6))
        i = 1
        for train, test in cv.split(X_train, y_train):
            prediction = Model(**params).fit(X_train[train],y_train[train]).predict(X_train[test])
            true = y_train[test]
            score = metrics.r2_score(true, prediction)
            scores.append(score)
            plt.scatter(prediction, true, lw=2, alpha=0.3, label='Fold %d (R2 = %0.2f)' % (i,score))
            i = i+1
        plt.plot([min(y_train),max(y_train)], [min(y_train),max(y_train)], linestyle='--', lw=2, color='black')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('%s: %s' %(Model.__name__,'K-Fold Validation'))
        plt.legend()
        plt.savefig('figs2' + Model.__name__ + '.svg')
        plt.show()
print(scores)


## K fold validation - one graphic #########

fig = plt.figure(figsize=(12, 6))
for modelname, Model, params_list in models:
        for params in params_list:
            prediction = Model(**params).fit(X_train,y_train).predict(X_test)
            true = y_test
            score = metrics.r2_score(true, prediction)
            scores.append(score)
            plt.scatter(prediction, true, lw=2, alpha=0.3, label= '%s: %s' %(modelname, round(score, 4)))
plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], linestyle='--', lw=2, color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.legend()
plt.savefig('comparison_dtf1.svg')
plt.show()


## Residuals and error plots 
from statsmodels.graphics.api import abline_plot

for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        ## residuals ############
        residuals = y_test - y_pred
        max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
        max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
        max_true = y_test[max_idx]
        max_pred = y_pred[max_idx]
        print(Model.__name__, "Max Error:", "{:,.0f}".format(max_error))
        col_residuals = pd.DataFrame(residuals)
        print(col_residuals)
        col_residuals["type"] = np.sign(col_residuals[0]).map({0:'Zero', 1: 'Positive', -1: 'Negative'})
        print(col_residuals)
        print(col_residuals["type"].value_counts(normalize=True))
        
        ## Plot predicted vs true ##############
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        ax[0].scatter(y_pred, y_test, color="black")
        abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
        ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
        ax[0].grid(True)
        ax[0].set(xlabel="Predicted", ylabel="True", title= "Predicted vs True")
        ax[0].legend()
        
        ## Plot predicted vs residuals
        ax[1].scatter(y_pred, residuals, color="red")
        ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
        ax[1].grid(True)
        ax[1].set(xlabel="Predicted", ylabel="Residuals", title= "Predicted vs Residuals")
        ax[1].hlines(y=0, xmin=np.min(y_pred), xmax=np.max(y_pred))
        ax[1].legend()
        fig.suptitle(Model.__name__, fontsize=13)
        plt.subplots_adjust(wspace=0.2)
        plt.savefig('figs4' + Model.__name__ + '.svg')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
        ax.grid(axis='x')
        ax.set(title= '%s: %s' %(Model.__name__, "Residuals distribution"))
        plt.savefig('figs6' + Model.__name__ + '.svg')
        plt.show()


## Explainibility
for modelname, Model, params_list in models:
    for params in params_list:
        # Fit model to the training set
        model = Model(**params)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        print(Model.__name__,"True:", "{:,.0f}".format(y_test[1]), "--> Pred:", "{:,.0f}".format(y_pred[1]))
        
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names="reads", mode="regression")
        explained = explainer.explain_instance(X_test[1], model.predict, num_features=3)
        explained.as_pyplot_figure()
        plt.savefig(Model.__name__ + '.svg')
        plt.show()
