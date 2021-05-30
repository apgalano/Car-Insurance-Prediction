# Car-Insurance-Prediction
A logistic regression problem with the goal of predicting whether a customer is likely to buy car insurance, given a set of features.

The dataset found in [kaggle](https://www.kaggle.com/kondla/carinsurance) is processed and visualised to reveal interesting insights about the data and their impact on predicting the CarInsurance binary variable. The figures can be found in the 'Figures' folder, while `data_description.csv` contains statistical analysis results. Logistic regression is used to create a predictive model using the given features. The resulted model coefficients are orderly output to `summary.csv` for easier inspection.

Several expected and unexpected conclusions can be reached about the correlation of each feature to the prediction variable. For instance, the most important positive predictor is call duration of previous sales calls, followed by sales outcome on previous call and specific months of the year. On the other hand, important negative predictors are the customer already having a house/car loan or the number of times contacted before. A large number of features has a coefficient near 0, meaning they are probably inconsequential to the predictive power of the model and could possibly be dropped.

The test set given (20% of all data) is unfortunatelly not labeled. Training the regressor on the entire training set results in 82.82% accuracy. To verify we have no overfitting and essentially test the model on unseen data, I make a 80-20 split on the training set, train on the 80% and test on the remaining 20%.
