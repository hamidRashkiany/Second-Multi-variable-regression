'''
Date: 2021-07-29
Author: Hamid Rashkiany
Description: This exercise include the Bosting House Dataset. This dataset involves 13 features and 506 samples.
The dataset is availabel in scikit-learn.
'''
# Step1: import all the library that we need for this exercise
from csv import Sniffer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import *
import sklearn

# Step2: In this section we need to load our dataset. In previous example (Multiple variable linear regression), we 
# created out dataset, but here in this example, we will directely load our dataset from from one domain as scikit-learn, 
# thus we do not need to even download the dataset to our PC or our server. We directly load it here to our project.
from sklearn.datasets import load_boston
bostonHouse_dataset=load_boston()
# Step3: Now we need to know what is this dataset about. Thus we print it out the dataset.
print(bostonHouse_dataset)
# Step4: After we print the dataset, we get some information about collecting date and authors, but it does not give practical 
#information about data. Thus let's print out head of our dataset to observe which sort of data are collected by this dataset.
print(bostonHouse_dataset.keys())
# Note: This dataset is created as a dictionary. Thus for invoke the header of dictionary, we just need to call the keys() of our 
# dictionay as above: print(bostonHouse_dataset.keys()). By invoke the header, we can see that there is five keys in this dataset as:
#dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename']). To access to any of those keys, easily write the name of key 
# after the name of dataset. For instace below code call DESCR key which gives us some information about dataset.
print(bostonHouse_dataset.DESCR)
"""
    Printing description reveals below info:
    **Data Set Characteristics:**

    :Number of Instances: 506

    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.
As you observe, there are 14 atributes. 13 atributes such as 1. CRIM (per capita crime rate by town) 
2. - ZN (proportion of residential land zoned for lots over 25,000 sq.ft.) to 13.- LSTAT (% lower status
 of the population) are independent featurs. But the last feature as - MEDV (Median value of owner-occupied 
 homes in $1000's) is our dependent feature and this will be our target or in other word our Y. 
 Thus our linear equation will be something like:
 Y = X0 + theta1*X1 + theta2*X2 + ... + theta13*X13
 This is only one hypothesis. Because we still do not know is there a linear relationship between dependent 
variable Y (MEDV) and each single independent variable or not. So as pervious example, lets observe the data.

"""
#Step5: Make the dataframe
boston_DataFrame=pd.DataFrame(bostonHouse_dataset.data,columns=bostonHouse_dataset.feature_names)
print(boston_DataFrame)
#Step 6: Add our data target (MEDV) to our dataframe. After print dataframe, we can see the target is not show
#in our dataset, thus we have to add it by ourselve.This is like ad one key to one dictionary and then 
# define its value. Implementation as below:
boston_DataFrame["MEDV"]=bostonHouse_dataset.target
print(boston_DataFrame)
#Step 6: Data processing: after put all data and make our dataframe, this is time for data processing. 
# 6.1: First check if there is any missing data. isnull().sum() funtion that return missing data in our data frame.
#if there is any missing data, then it will the number of missing data in each feature.
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html
#dataFrame.isnull() : returns a bolean same-sized object indicating if the values are NA.NA values, such as None or numpy.
# NaN, gets mapped to True values. Everything else gets mapped to False values. Characters such as empty strings '' or 
# numpy.inf are not considered NA values (unless you set pandas.options.mode.use_inf_as_na = True)   
print(boston_DataFrame.isnull().sum())
#Step7: Data exploratory: 
"""
The first step of data exploratory is visualizing data. we need to know how does the data is 
spread in our target. in other language, we need to know where does located each amount of data in our data. For example 
we would like to know how many houses with price of 200 is in our dataset. One way is check all house prices one by one 
and categorize the prices in diferent groups. the group of prices between 0-100, 100-200, 200-300 and etc. This will be 
imposibile for big dataset. The practical and possible method is draw one histogram plot of our data. Histogram plot can
shows us the probabilty or frequency of data. It says the frequency of data point x (the number of repeataion) in our dataset
is equal to which probability. For example if we plot the data histogram for bins with length of 10 in each intervals, the 
histogram will say the probability of data between 0-10 is eual to P1, the probability of data between 10-20 is equal to P2
and etc. There can be three different histograms for data as Normal distribution, Skewness and Kurtosis. You can find more 
about each distribution in:https://tekmarathon.com/2015/11/13/importance-of-data-distribution-in-training-machine-learning-models/
but it is important to know that Normal distribution is one of the most significant data distribution in machine learning models.
All normal distributions are symmetric and have bell-shaped curves with a single peak (aka Gaussian Distribution). 
Creating a histogram on variable (variable values on Xaxis and its frequency on Yaxis) would get you a normal distribution.
When the distribution is normal then it obeys 68-95-99.7% rule. Which means
68% of data points/observations fall within +1*(Standard Deviation) to -1*(Standard Deviation) of mean
95% of data points/observations fall within +2*(Standard Deviation) to -2*(Standard Deviation) of mean
7% of data points/observations fall within +13*(Standard Deviation) to -3*(Standard Deviation) of mean
"""
# 7.1 : Boston House Price dataset Histogram for target data:
# The hitogram of Boston House Data for target data as MEDV can plot as below:
sns.distplot(boston_DataFrame["MEDV"],bins=30,kde=True)
plt.show()
#sns.histplot(boston_DataFrame["MEDV"])# This command also plot the histogram but without curve on the graph
#plt.show()
"""
We can see that all prices in house dataset has a kind of normal distribution and they are symmetric.
Another critical step to estimate our model is to check which feature should be use to design our model? Should we 
utilize all features to estimate the target value? Which features have higher influence to our target and which one 
have less effect? We can answer to all these question by visualize another plot as correlation plot or corelation matrix.
Indeed corellation in machine learning means what is the relationship between different data in our dataset. 
"""
#7.2: Correlation matrix:
correlation_matrix=boston_DataFrame.corr().round(2)
#Now plot correlation matrix by correlation numbers in each house
sns.heatmap(data=correlation_matrix,annot=True)
plt.show()
"""
The elements of correlation matrix are numbers between 1 and -1. As number is more close to 1, it means there is one strong
correlation between two variables (Features) and as it close to -1, it means there is one strong correlation between two features
but in inverse direction. It means by increasing the value in one variable, the value of other variable will decrease.
According to this interpretation of correlation matrix, we can see there is one strong relationship between MEDV and LSTAT (-0.74).
Also there another strong correlation between AGE and DIS (-0.75). Thus between these two independent features, we only use one of them in 
our model. Because thses two models have and strong relation and change in one of them has same result to dependent variabl.
https://stats.stackexchange.com/questions/1149/is-there-an-intuitive-explanation-why-multicollinearity-is-a-problem-in-linear-r/1150#1150
RM also has a positive correlation with MEDV. RAD and TAX also has 0.91 correlation so we will choose only one of them.
In this paper, there are only two features opted to implement in our model: RM with correlation 0.7 and LSTAT with correlation -0.71.
Thus here we will go plot scatter to see how is the distribution of price according to RM (average number of rooms per dwelling) and 
how is the distribution of price based on LSTAT (% lower status of the population).
Hence the linear regression equation that explained in step 4 will alter as below:
        Y=X0+theta1*X1+theta2*X2
"""
plt.subplot(2,1,1)
plt.scatter(boston_DataFrame["RM"],boston_DataFrame["MEDV"])
plt.xlabel("RM (average number of rooms per dwelling)")
plt.ylabel("MEDV (House's price in $1000's")
plt.subplot(2,1,2)
plt.scatter(boston_DataFrame["LSTAT"],boston_DataFrame["MEDV"])
plt.xlabel("LSTAT (lower status of the population)")
plt.ylabel("MEDV (House's price in $1000's")
plt.show()
"""
After plot sccater we can observe that there is one linear relation between both independents variables and dependent variable.
But the hypothesis line for MEDV have positive slop thus the coeficient sign for its theta parametter will be positive. But the slop's line 
for LSTAT is negative, which it means its coeficient theta parametter will be negative. 
"""
#8.Training our model
#8.1: First define our input and output data. Two features are utilizing as input: LSTAT and RM. We will concatenate these two data together.
#and MEDV will define as target.
x=pd.DataFrame(np.c_[boston_DataFrame["LSTAT"],boston_DataFrame["RM"]],columns=["LSTAT","RM"])
y=boston_DataFrame["MEDV"]
#8.2: Splitting dataset
# Second split data to two sets: training set and testing set. The dataset will divide two to categories. 80% of data utilizes for training our
#model and 20 percent of data utilizes for testing our dataset. The train_test_split function from scikit-learn apply to split our dataset.
from sklearn.model_selection import train_test_split
x_Train,x_Test,y_Train,y_Test=train_test_split(x,y,train_size=0.8,test_size=0.2)
print("The shape of input training dataset: ",x_Train.shape)
print("The shape of output training dataset: ",y_Train.shape)
print("The shape of input testing dataset: ",x_Test.shape)
print("The shape of output testing dataset: ",y_Test.shape)
#8.3: Extract the optimum value of parameters (X0,theta1,theta2)
#The details of how to extract the optimum values explained in other training example. You can check the example of estimation the index_rate
#_value example. Here we directly utilize the LinearRegression function from scikit-learn to extract the optimum values
from sklearn.linear_model import LinearRegression
myLinearModel=LinearRegression()
myLinearModel.fit(x_Train,y_Train)
print("The optimum values of coeficients theta1 and theta for training data:","\n",myLinearModel.coef_)
print("The optimum value of intercept for training data:","\n",myLinearModel.intercept_)
print("----------------------------------------------------------------")
#After runnig the above code, the result will be as below:
# The optimum values of coeficients theta1 and theta:
#  [-0.65586902  5.00704338]
# The optimum value of intercept:
#  -0.6466500458710946
#These results disclose that the optimum value for slop line related to LSTAT is -0.65586902 and the optimum value 
#for slop line related to RM is 5.00704338. Also intercept point will be at -0.6466500458710946
#Thus the linear equation related to this dataset according to LSTAt and RM will be as below:
# estimation of MEDV=-0.6466500458710946-0.65586902*LSTAT+5.00704338*RM
#9. Evaluate our model:
#There are different ways to evaluate our system. Most three methods that utilizes in Ordinary Least Square are R-2 
# Square, Root Mean Square Error (RMSE) and the overal F-test. Three approaches are based on two sum of squares:
# Sum of Squares Error (SSE) and Sum of Square Total (SST). The combination of these two sums gives us of scale to 
#evaluate our algorithm. In intuitive meaning, SSE calculates how much the actual value of data is far from the estimated
#value by model. SST is measuring how much is data far from the mean. 
# In this practical problem we are using the R2 square and RMSE.
#In machine learning regression algorithm, a model has good result if  the estimation value be close to real value 
#of output. R2 square has one intuitive concept. It has a value between [0,1]. The mathematical equation can be donted
#as below:
# R2Square=((sum(Mean of y-Estimated value of y))**2)/((sum(The actual value of y-Mean value of y))**2)
#According to the previous defintion, the numerator of above equation is the variance of predicted value and denominator
#is the variance of actual value. Reference: https://www.theanalysisfactor.com/r-squared-for-mixed-effects-models/
#9.1: Evaluate the model for training data
y_Train_Model_Estimation=myLinearModel.predict(x_Train)#This line will call predict() function of our Linear Regression model
r2Square_TrainingData=sklearn.metrics.r2_score(y_Train,y_Train_Model_Estimation)
rmse_TrainingData=(np.sqrt(sklearn.metrics.mean_squared_error(y_Train,y_Train_Model_Estimation)))
print("The evaluation result of our model according to R2 Square and RMSE for training data:")
print("-------------------------------------------------------------------\n")
print("R2 Square is equal to {}".format(r2Square_TrainingData))
print("RMSE is equal to {}".format(rmse_TrainingData))

#9.2: Evaluate the model for training data
y_Test_Model_Estimation=myLinearModel.predict(x_Test)
r2Square_TestData=sklearn.metrics.r2_score(y_Test,y_Test_Model_Estimation)
rmse_TestData=(np.sqrt(sklearn.metrics.mean_squared_error(y_Test,y_Test_Model_Estimation)))
print("-------------------------------------------------------------------")
print("The evaluation result of our model according to R2 Square and RMSE for testing date:")
print("-------------------------------------------------------------------\n")
print("R2 Square is equal to {}".format(r2Square_TestData))
print("RMSE is equal to {}".format(rmse_TestData))




