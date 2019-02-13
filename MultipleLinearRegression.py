import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize,StandardScaler,scale
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


dataframe = pd.read_excel("C:/Users/vsrivas3/Desktop/vivek/Python/Data Science Tutorial/LinearData.xlsx")
#### Describe the data Frame and check all data point distribution
print(dataframe.describe())
#### Check the null values in dataframe
print(dataframe.isnull().sum())
#### droping null values fro data frame
dataframe[['sqft_living','bedrooms','bathrooms','floors','price']].dropna()
#### checking duplicate values in data frame
print(dataframe.duplicated(keep='first'))
### Droping a column if not required
dataframe.drop(['price'],inplace=True, axis=1)
X = dataframe.iloc[:,1:5]
y = dataframe.iloc[:,1]
#print(X)

### SNS plot for Correlation matrix
sns.pairplot(dataframe)
plt.show()

#### Pandas corr function for correlation between all the columns
correlation = dataframe.corr(method ='pearson')
print(correlation)

## checking multicollenearity betwwen all the independent valriable by VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
#print(vif.round(1))

#### Normalization of X
X = scale(X)
## import train test split from model selection module
### took 80 % data as training data and 20 % as testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting multiple lineaar regression to the training set
regressor = LinearRegression()

## checking fitment of model
regressor.fit(X_train, y_train)
### checking score on test values
regressor.score(X_test,y_test)
### predict the new value of y by using x test
y_pred = regressor.predict(X_test)
print(y_pred)

### Check the cofficient and RMSE and variance
### y = b0+b1x+b2x-----bnx
##intercept b0
print(regressor.intercept_)

##cofficient is b1,b2--bn
#  The coefficients
print('Coefficients: \n', regressor.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

## plot graph for y test value and y predicted value
plt.scatter(y_pred,y_test)
#plt.show()
#################################################################################
## By using stats model for summary of model
lobj=sm.OLS(y,X)
reg_sum=lobj.fit()
print(reg_sum.summary())











