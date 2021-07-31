# Second-Multi-variable-regression
<h2>Boston House Price estimation according to Boston House price data-set in sklearn</h2>
Date: 2021-07-29</br>
Author: <em>Hamid Rashkiany</em></br>
<b>Description:</b> This exercise include the Bosting House Dataset. This dataset involves 13 features and 506 samples.
The dataset is availabel in scikit-learn.
<p>
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
</p>
