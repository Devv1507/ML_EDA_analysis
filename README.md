# ML_EDA_analysis
Basis of Exploratory Data Analysis (EDA) and comparation of Supervised Machine Learning models for numeral variables designed in Python.

EDA analysis allows a better understanding of the predictive power of the variables before modelling. In this case, the target variable data were grouped into bins to comparing its behavior and, since all variables are continuoys, a Pearson correlation coefficient was calculared as a measure of the strength of their relationship. Also, scatterplots, distrubution graphs and heatmaps were defined. At this stage, the libraries “pandas”, “NumPy”, “Matplotlib” and “Seaborn” were used through the programming language Python v3.9.9. 

In the next step, the target variable was isolated from the rest of explanatory variables to design, train, test, and evaluation of the SML model. For both dataset a scaling process was applied using the RobustScaler function from sklearn, then the datasets were splitted into Train (75%) and test (25%) set.  For reproducibility purposes, a random seed of 27 was set for all models that uses a random state.

To select the best performance estimator, several models were trained. In this case, SML algorithms to predict continuous-valued outputs, both linear model regression as Lasso (LssR), and ensemble methods as Gradient Boosting Regressor (GBR) and Random Forest Regressor (RFR), in adition of Support Vector Regressor (SVR) were used.

Models Hyperparameters were modified in order to achieve a better performance for models. For LssR was trained with different values for alpha regularization, those values were 200 numbers in a log scale from -10 to 3. For RFR and GBR seed was set randomly as 27. For SVR all hyperparameters were used by default. Finally, to compare the models, R squared metrics were used for models performance with k-fold cross-validation splitting the dataset into 5 random folds to avoid the model overfitting. Additionally, the average of the 5 scores provided by the cross-validation process and explanatory graphics of the predictions were made.


